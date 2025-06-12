import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("./data/train_cleaned.csv")

le = LabelEncoder()
df['Loan_Status'] = le.fit_transform(df['Loan_Status']) # Y -> 1, N - > 0

# One hot encoding remaining categorical features
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df.drop('Loan_Status', axis = 1)
Y = df['Loan_Status']

# Spliting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state= 42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Initialize the model
model = LogisticRegression ()

# Train the model
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

# Evaluate performace
# print("Accuracy: ", accuracy_score(Y_test, Y_pred))
# print("Classification report", classification_report(Y_pred , Y_test))


# # Load model and scaler
# model = joblib.load('loan_approval_model.pkl')
# scaler = joblib.load('scaler.pkl')

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

