import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# LOAD MODEL
# -----------------------------
pipeline = joblib.load('models/pipeline.pkl')

st.set_page_config(page_title="Customer Churn App", layout="wide")

st.title("Customer Churn Prediction App")
st.markdown("Developed by Priya Pathak")
st.write("Enter customer details to predict churn probability")

# -----------------------------
# INPUT UI
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 1)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

with col2:
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

MonthlyCharges = st.number_input("Monthly Charges", value=50.0)
TotalCharges = st.number_input("Total Charges", value=100.0)

# -----------------------------
# PREDICT BUTTON
# -----------------------------
if st.button("Predict Churn"):

    # Create dataframe
    input_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [SeniorCitizen],
        'Partner': [Partner],
        'Dependents': [Dependents],
        'tenure': [tenure],
        'PhoneService': [PhoneService],
        'MultipleLines': [MultipleLines],
        'InternetService': [InternetService],
        'OnlineSecurity': [OnlineSecurity],
        'OnlineBackup': [OnlineBackup],
        'DeviceProtection': [DeviceProtection],
        'TechSupport': [TechSupport],
        'StreamingTV': [StreamingTV],
        'StreamingMovies': [StreamingMovies],
        'Contract': [Contract],
        'PaperlessBilling': [PaperlessBilling],
        'PaymentMethod': [PaymentMethod],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges]
    })
    input_data['customerID'] = '0000'

    input_data['MonthlyCharges'] = input_data['MonthlyCharges'].astype(float)
    input_data['TotalCharges'] = input_data['TotalCharges'].astype(float)
    
    # Prediction
    prediction = pipeline.predict(input_data)[0]
    prob = pipeline.predict_proba(input_data)[0][1]

    # Output
    if prediction == 1:
        st.error(f"Customer likely to churn (Probability: {prob:.2f})")
    else:
        st.success(f"Customer not likely to churn (Probability: {prob:.2f})")
