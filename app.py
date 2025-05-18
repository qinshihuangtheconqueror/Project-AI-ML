import gradio as gr
import pandas as pd
import joblib
import numpy as np

# Load model và preprocessing
model = joblib.load("project/checkpoint/random_forest.joblib")
preprocessing = joblib.load("project/checkpoint/preprocessing.joblib")

# Đọc danh sách triệu chứng từ file Train.csv
df = pd.read_csv("project/disease prediction/Train.csv")
symptoms = df.columns[:-1].tolist()  # Bỏ cột prognosis

# Lấy danh sách bệnh hợp lệ (chuẩn hóa)
def normalize(s):
    # Chuyển về chữ thường và loại bỏ khoảng trắng thừa
    s = str(s).strip().lower()
    # Loại bỏ dấu ngoặc đơn và nội dung trong đó
    s = s.split('(')[0].strip()
    # Loại bỏ các ký tự đặc biệt
    s = ''.join(c for c in s if c.isalnum() or c.isspace())
    return s.strip()

train_diseases = set(normalize(x) for x in df['prognosis'].unique())
desc_df = pd.read_csv("project/disease prediction/symptom_Description.csv")
desc_df['Disease_norm'] = desc_df['Disease'].apply(normalize)
desc_diseases = set(desc_df['Disease_norm'].unique())
precaution_df = pd.read_csv("project/disease prediction/symptom_precaution.csv")
precaution_df['Disease_norm'] = precaution_df['Disease'].apply(normalize)
precaution_diseases = set(precaution_df['Disease_norm'].unique())

valid_diseases = train_diseases & desc_diseases & precaution_diseases

# Thử load LabelEncoder nếu có
try:
    from sklearn.preprocessing import LabelEncoder
    label_encoder = None
    # Nếu preprocessing có LabelEncoder, lấy ra
    if hasattr(preprocessing, 'named_transformers_'):
        for v in preprocessing.named_transformers_.values():
            if isinstance(v, LabelEncoder):
                label_encoder = v
    elif hasattr(preprocessing, 'classes_'):
        # Trường hợp lưu trực tiếp LabelEncoder
        label_encoder = preprocessing
except Exception:
    label_encoder = None

def predict_disease(*symptom_values):
    # Convert True/False to 1/0
    symptom_values = [1 if v else 0 for v in symptom_values]
    input_data = pd.DataFrame([symptom_values], columns=symptoms)

    # Preprocess input data
    X_transformed = preprocessing.transform(input_data)
    
    # Predict
    prediction = model.predict(X_transformed)[0]
    
    # Debug: Print prediction and label_encoder info
    print(f"Raw prediction: {prediction}")
    print(f"Label encoder classes: {label_encoder.classes_ if label_encoder else 'No label encoder'}")
    
    # Convert index to disease name
    if isinstance(prediction, (int, np.integer)):
        if label_encoder is not None:
            prediction = label_encoder.inverse_transform([prediction])[0]
        else:
            prediction = df['prognosis'].unique()[prediction]
    
    prediction_norm = normalize(str(prediction))
    
    # Debug: Print info for checking
    print(f"Prediction after conversion: {prediction}")
    print(f"Normalized prediction: {prediction_norm}")
    print(f"Valid diseases: {sorted(valid_diseases)}")
    
    # Only return if disease is valid
    if prediction_norm not in valid_diseases:
        return f"Disease '{prediction}' does not have description or precaution information."
    
    # Get disease description
    description = desc_df[desc_df['Disease_norm'] == prediction_norm]['Description'].values[0]
    
    # Get precautions
    precaution_list = precaution_df[precaution_df['Disease_norm'] == prediction_norm].iloc[0, 1:5].tolist()
    
    return f"""
<div class='diagnosis-card'>
    <h2>🦠 {prediction}</h2>
    <p style='font-size:1.15em; margin-bottom:20px; color:#222;'>{description}</p>
    <b style='font-size:1.1em;'>🛡️ Precautionary measures:</b>
    <ul style='font-size:1.1em; margin-top:10px;'>
        {''.join([f'<li>{p.strip()}</li>' for p in precaution_list])}
    </ul>
</div>
"""

# Gradio UI in English
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as app:
    gr.Markdown("""
    <style>
    /* Checkbox style */
    .gr-checkbox label {
        background: #f1f5f9;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        margin: 10px 0 !important;
        font-size: 1.08em;
        transition: background 0.2s, box-shadow 0.2s;
        box-shadow: 0 2px 8px #2563eb11;
        border: 1.5px solid #e0e7ef;
    }
    .gr-checkbox label:hover {
        background: #e0e7ff;
        border: 1.5px solid #2563eb;
        box-shadow: 0 4px 16px #2563eb22;
    }
    /* Nút Diagnose */
    #diagnose-btn button {
        background: linear-gradient(90deg,#2563eb,#60a5fa) !important;
        color: white !important;
        font-weight: bold;
        font-size: 1.2em;
        border-radius: 10px;
        margin: 28px 0 18px 0;
        padding: 14px 36px;
        box-shadow: 0 4px 16px #2563eb33;
        transition: background 0.2s, box-shadow 0.2s;
    }
    #diagnose-btn button:hover {
        background: linear-gradient(90deg,#1e40af,#60a5fa) !important;
        box-shadow: 0 8px 32px #2563eb44;
    }
    /* Card kết quả */
    #result-card {
        margin-top: 24px;
    }
    #result-card .diagnosis-card {
        border: 3px solid #2563eb;
        border-radius: 18px;
        padding: 32px;
        background: linear-gradient(90deg,#e0e7ff,#f8fafc);
        box-shadow: 0 8px 32px #1e40af22;
        font-size: 1.15em;
        animation: fadeIn 0.7s;
    }
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(30px);}
        to {opacity: 1; transform: none;}
    }
    #result-card h2 {
        color: #1e40af;
        font-size: 2em;
        font-weight: bold;
        margin-bottom: 16px;
    }
    #result-card ul {
        margin-top: 12px;
        margin-bottom: 0;
    }
    </style>
    <div style='text-align:center; padding: 20px 0;'>
        <h1 style='color:#2563eb; font-size:2.5em;'>🩺 Disease Diagnosis from Symptoms</h1>
        <p style='font-size:1.2em;'>Please select the symptoms you are experiencing:</p>
    </div>
    """)
    
    # Create checkboxes for each symptom, chia thành 4 cột
    symptom_inputs = []
    num_cols = 4
    with gr.Row():
        cols = [gr.Column() for _ in range(num_cols)]
        for idx, symptom in enumerate(symptoms):
            with cols[idx % num_cols]:
                symptom_inputs.append(gr.Checkbox(label=symptom))
    
    # Predict button and output
    predict_btn = gr.Button("Diagnose", elem_id="diagnose-btn")
    output = gr.Markdown(elem_id="result-card")
    
    # Event handler
    def predict_and_format(*args):
        result = predict_disease(*args)
        # Định dạng lại kết quả cho đẹp
        if result.startswith("Predicted disease:"):
            lines = result.strip().split("\n")
            disease = lines[0].replace("Predicted disease:", "").strip()
            desc = lines[2].replace("Description:", "").strip()
            precautions = [l.strip() for l in lines[4:] if l.strip() and l.strip()[0].isdigit()]
            return f"""
<div class='diagnosis-card'>
    <h2>🦠 {disease}</h2>
    <p style='font-size:1.15em; margin-bottom:20px; color:#222;'>{desc}</p>
    <b style='font-size:1.1em;'>🛡️ Precautionary measures:</b>
    <ul style='font-size:1.1em; margin-top:10px;'>
        {''.join([f'<li>{p.strip()}</li>' for p in precautions])}
    </ul>
</div>
"""
        else:
            return f"<div style='color:red; font-weight:bold; padding:16px'>{result}</div>"
    
    predict_btn.click(
        fn=predict_and_format,
        inputs=symptom_inputs,
        outputs=output
    )

# Launch app
app.launch()