import streamlit as st
import requests
import os
from dotenv import load_dotenv
import google.generativeai as genai

# === Config ===
st.set_page_config(page_title="AutoOps AI", page_icon="🤖", layout="centered")
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# === Helper: LLM Explanation ===
def get_explanation(input_data, prediction):
    prompt = f"""
    A machine learning model predicted: {"Churn" if prediction == 1 else "No Churn"}.

    Here are the customer details:
    {input_data}

    Based on this, explain in simple terms why the prediction was made. Highlight key features like tenure, contract type, services used, and payment method.
    """
    model = genai.GenerativeModel("models/gemini-1.5-pro")
    response = model.generate_content(prompt)
    return response.text.strip()

# === UI Header ===
st.markdown("<h1 style='text-align: center;'>AutoOps AI – Churn Predictor</h1>", unsafe_allow_html=True)
st.write("Predict customer churn and explain it using a language model.")

# === Input Form ===
with st.form("churn_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0)
    TotalCharges = st.number_input("Total Charges ($)", min_value=0.0)

    submit = st.form_submit_button("🔍 Predict")

# === On Submit: Store Input + Get Prediction ===
if submit:
    st.session_state.user_data = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=st.session_state.user_data)
        if response.status_code == 200:
            result = response.json()["prediction"]
            st.session_state.prediction = result
        else:
            st.warning(f"Unexpected API response: {response.text}")
    except Exception as e:
        st.error(f"Failed to connect to API: {e}")

# === Display Prediction (if exists) ===
if "prediction" in st.session_state:
    result = st.session_state.prediction
    if result == 1:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is unlikely to churn.")

    if st.checkbox("🧠 Explain this prediction using Gemini Pro"):
        with st.spinner("Generating explanation..."):
            explanation = get_explanation(st.session_state.user_data, result)
            st.info(explanation)

