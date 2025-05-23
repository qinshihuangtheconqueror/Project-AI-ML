import gradio as gr
import pandas as pd
import numpy as np
import joblib
from scipy.stats import boxcox

numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
target_label = ['NObeyesdad']

def load_model(model):
    if model == "Logistic regression":
        pred_model = joblib.load("New folder (3)/checkpoint/logistic_regression.joblib")
    elif model == "K nearest neighbors":
        pred_model = joblib.load("New folder (3)/checkpoint/knn.joblib")
    elif model == "Decision tree":
        pred_model = joblib.load("New folder (3)/checkpoint/decision_tree.joblib")
    elif model == "Random forest":
        pred_model = joblib.load("New folder (3)/checkpoint/random_forest.joblib")
    elif model == "XGBoost":
        pred_model = joblib.load("New folder (3)/checkpoint/xg_boost.joblib")
    elif model == "Voting classifier":
        pred_model = joblib.load("New folder (3)/checkpoint/votingClassifier.joblib")
    return pred_model

preprocessing = joblib.load("New folder (3)/checkpoint/preprocessing.joblib")

label_mapping = {
    0: 'Insufficient_Weight',
    1: 'Normal_Weight',
    2: 'Overweight_Level_I',
    3: 'Overweight_Level_II',
    4: 'Obesity_Type_I',
    5: 'Obesity_Type_II',
    6: 'Obesity_Type_III'
}

recommendations = {
    'Insufficient_Weight': "Bạn nên tăng cường dinh dưỡng và tham khảo ý kiến chuyên gia để cải thiện cân nặng.",
    'Normal_Weight': "Bạn đang có cân nặng hợp lý, hãy duy trì lối sống lành mạnh và tiếp tục theo dõi sức khỏe.",
    'Overweight_Level_I': "Bạn nên chú ý chế độ ăn, tăng cường vận động và kiểm soát cân nặng.",
    'Overweight_Level_II': "Bạn nên giảm lượng calo, tập thể dục đều đặn và theo dõi sức khỏe thường xuyên.",
    'Obesity_Type_I': "Bạn nên tham khảo ý kiến bác sĩ để có phác đồ giảm cân phù hợp và an toàn.",
    'Obesity_Type_II': "Bạn cần có sự can thiệp chuyên sâu từ chuyên gia dinh dưỡng và bác sĩ.",
    'Obesity_Type_III': "Bạn nên đến cơ sở y tế để được tư vấn và hỗ trợ giảm cân an toàn, tránh biến chứng nguy hiểm."
}

def predict_obesity_level(model_name, age, height, weight, fcvc, ncp, ch2o, faf, tue, 
                          gender, family_history_with_overweight, favc, caec, smoke, 
                          scc, calc, mtrans):
    model = load_model(model_name)
    x = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Height': [height],
        'Weight': [weight],
        'family_history_with_overweight': [family_history_with_overweight],
        'FAVC': [favc],
        'FCVC': [fcvc],
        'NCP': [ncp],
        'CAEC': [caec],
        'SMOKE': [smoke],
        'CH2O': [ch2o],
        'SCC': [scc],
        'FAF': [faf],
        'TUE': [tue],
        'CALC': [calc],
        'MTRANS': [mtrans]
    })
    if age > 0:
        try:
            x['Age'], _ = boxcox(x['Age'])
        except ValueError:
            x['Age'] = np.log1p(x['Age'])
    else:
        x['Age'] = np.log1p(x['Age'])
    x['FCVC'] = pd.cut(x['FCVC'], bins=[0.5,1.5,2.5,3.5], labels=[1,2,3]).astype('float64')
    x['NCP'] = pd.cut(x['NCP'], bins=[0.5,1.5,2.5,3.5,4.5], labels=[1,2,3,4]).astype('float64')
    x['CH2O'] = pd.cut(x['CH2O'], bins=[0.5,1.5,2.5,3.5], labels=[1,2,3]).astype('float64')
    x['FAF'] = pd.cut(x['FAF'], bins=[-0.5,0.5,1.5,2.5,3.5], labels=[0,1,2,3]).astype('float64')
    x['TUE'] = pd.cut(x['TUE'], bins=[-0.5,0.5,1.5,2.5], labels=[0,1,2]).astype('float64')
    int64_columns = x.select_dtypes(include='int64').columns
    x[int64_columns] = x[int64_columns].astype('float64')
    x = preprocessing.transform(x)
    x = pd.DataFrame(x, columns=preprocessing.get_feature_names_out())
    y = model.predict(x)
    label = label_mapping[y[0]]
    advice = recommendations[label]
    return label, advice

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo")) as app:
    gr.Markdown("# Obesity Level Classification")
    gr.Markdown("Predict the level of obesity based on various health and lifestyle factors.")
    gr.Markdown("Note: ")
    gr.Markdown("The value of Consumption of vegetables (FCVC) ranges from 1 to 3.")
    gr.Markdown("The value of Number of main meals (NCP) ranges from 1 to 4.")
    gr.Markdown("The value of Consumption of water daily (CH2O) ranges from 1 to 3.")
    gr.Markdown("The value of Physical activity frequency (FAF) ranges from 0 to 3.")
    gr.Markdown("The value of Time using tech devices (TUE) ranges from 0 to 2.")

    with gr.Group():
        gr.Markdown("## Personal Status")
        with gr.Row():
            Age = gr.Number(label="Age", info="Nhập tuổi của bạn (năm)")
            Height = gr.Number(label="Height", info="Nhập chiều cao (mét)")
            Weight = gr.Number(label="Weight", info="Nhập cân nặng (kg)")
        with gr.Row():
            Gender = gr.Dropdown(label="Gender", choices=["Male", "Female"], info="Chọn giới tính của bạn")
            Family_history = gr.Dropdown(label="Family history with overweight", choices=["yes", "no"], info="Gia đình bạn có ai từng thừa cân/béo phì không?")

    with gr.Group():
        gr.Markdown("## Routine")
        with gr.Row():
            FAF = gr.Number(label="Physical activity frequency (FAF)", minimum=0, maximum=3, step=1, info="Tần suất hoạt động thể chất (0: không, 3: thường xuyên)")
            TUE = gr.Number(label="Time using tech devices (TUE)", minimum=0, maximum=2, step=1, info="Thời gian sử dụng thiết bị công nghệ (0: ít, 2: nhiều)")
        with gr.Row():
            MTRANS = gr.Dropdown(label="Mode of transportation (MTRANS)", choices=["Automobile", "Motorbike", "Bike", "Public Transportation", "Walking"], info="Phương tiện di chuyển chính của bạn")

    with gr.Group():
        gr.Markdown("## Eating Habits")
        with gr.Row():
            FCVC = gr.Number(label="Consumption of vegetables (FCVC)", minimum=1, maximum=3, step=1, info="Tần suất ăn rau củ (1: ít, 3: nhiều)")
            NCP = gr.Number(label="Number of main meals (NCP)", minimum=1, maximum=4, step=1, info="Số bữa ăn chính mỗi ngày (1-4)")
        with gr.Row():
            FAVC = gr.Dropdown(label="Consumption of high caloric food (FAVC)", choices=["yes", "no"], info="Bạn có thường ăn đồ ăn nhiều calo không?")
            CAEC = gr.Dropdown(label="Consumption of food between meals (CAEC)", choices=["no", "Sometimes", "Frequently", "Always"], info="Bạn có ăn vặt giữa các bữa chính không?")
            CH2O = gr.Number(label="Consumption of water daily (CH2O)", minimum=1, maximum=3, step=1, info="Lượng nước uống mỗi ngày (1: ít, 3: nhiều)")

    with gr.Group():
        gr.Markdown("## Health-related Factors")
        with gr.Row():
            SMOKE = gr.Dropdown(label="Smokes (SMOKE)", choices=["yes", "no"], info="Bạn có hút thuốc không?")
            SCC = gr.Dropdown(label="Monitor calories consumption (SCC)", choices=["yes", "no"], info="Bạn có kiểm soát lượng calo nạp vào không?")
        with gr.Row():
            CALC = gr.Dropdown(label="Consumption of alcohol (CALC)", choices=["no", "Sometimes", "Frequently", "Always"], info="Tần suất uống rượu/bia")

    Model = gr.Dropdown(
        label="Model",
        choices=[
            "Logistic regression",
            "K nearest neighbors",
            "Decision tree",
            "Random forest",
            "XGBoost",
            "Voting classifier"
        ],
        info="Chọn mô hình dự đoán"
    )

    Prediction = gr.Textbox(label="Obesity Level Classification")
    Advice = gr.Textbox(label="Lời khuyên cho bạn", interactive=False)

    with gr.Row():
        submit_button = gr.Button("Predict")
        submit_button.click(fn=predict_obesity_level,
                            outputs=[Prediction, Advice],
                            inputs=[Model, Age, Height, Weight, FCVC, NCP, CH2O, FAF, TUE,
                                    Gender, Family_history, FAVC, CAEC, SMOKE, SCC, CALC, MTRANS
                                    ],
                            queue=True)
        clear_button = gr.ClearButton(components=[Prediction, Advice], value="Clear")
        
    app.launch()