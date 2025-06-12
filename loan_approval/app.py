import streamlit as st
import pandas as pd
import joblib
import os

# Get absolute path to the folder where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build full absolute path to the model file
model_path = os.path.join(BASE_DIR, 'loan_approval_model.sav')
scaler_path = os.path.join(BASE_DIR, 'loan_approval_model.sav')

# Load model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Custom CSS
st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
        }
        .title {
            text-align: center;
            color: #2c3e50;
        }
        .stButton>button {
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='title'>🏦 Loan Approval Prediction App</h1>", unsafe_allow_html=True)
st.markdown("---")

# Input layout
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])

with col2:
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    applicant_income = st.number_input("Applicant Income", min_value=0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0)

st.markdown("### 📋 Loan Details")
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
loan_term = st.number_input("Loan Term (in days)", value=360)
credit_history = st.selectbox("Credit History", [1.0, 0.0])

# Predict
if st.button("🔍 Predict"):
    input_data = {
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Gender_Male': 1 if gender == "Male" else 0,
        'Married_Yes': 1 if married == "Yes" else 0,
        'Dependents_1': 1 if dependents == "1" else 0,
        'Dependents_2': 1 if dependents == "2" else 0,
        'Dependents_3+': 1 if dependents == "3+" else 0,
        'Education_Not Graduate': 1 if education == "Not Graduate" else 0,
        'Self_Employed_Yes': 1 if self_employed == "Yes" else 0,
        'Property_Area_Semiurban': 1 if property_area == "Semiurban" else 0,
        'Property_Area_Urban': 1 if property_area == "Urban" else 0
    }

    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    st.markdown("### 🧾 Prediction Result")
    if prediction[0] == 1:
        st.success("✅ Congratulations! Your loan is likely to be approved.")
    else:
        st.error("❌ Sorry, your loan is not likely to be approved.")

