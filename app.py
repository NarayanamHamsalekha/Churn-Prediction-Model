# app.py
import streamlit as st
import pandas as pd
import pickle

# 1️⃣ Load saved model
with open("best_xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load CSV file with customer data
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Preprocess CSV (like during training)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype('category')

# Title with emoji
st.title("📊 Customer Churn Prediction App")

# 2️⃣ Sidebar - Prediction Settings
st.sidebar.header("Prediction Settings")
threshold = st.sidebar.slider("Prediction Threshold (%)", 0, 100, 50)
st.sidebar.markdown(f"**Threshold Explanation:** Probability ≥ {threshold}% predicts CHURN")
input_method = st.sidebar.radio("Select Input Method", ("Manual Input", "Customer ID"))

# 3️⃣ Main Area - Customer Details
st.subheader("Customer Details")

def user_input_features():
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    MonthlyCharges = st.slider("Monthly Charges", 0, 200, 70)
    TotalCharges = st.slider("Total Charges", 0, 10000, 2000)

    data = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }
    features = pd.DataFrame(data, index=[0])
    for col in features.select_dtypes(include='object').columns:
        features[col] = features[col].astype('category')
    return features

# Select input method
if input_method == "Manual Input":
    input_df = user_input_features()
else:
    # Select Customer ID from CSV
    # If selecting customer from CSV
    customer_id = st.selectbox("Select Customer ID", df['customerID'].tolist())
    input_df = df[df['customerID'] == customer_id].drop(['Churn', 'customerID'], axis=1, errors='ignore')

# 4️⃣ Predict button
if st.button("Predict"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader("Prediction")
    if prediction[0] == 1 or (prediction_proba[0][1]*100 >= threshold):
        st.markdown("<span style='color:red;font-weight:bold'>Customer will churn</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:green;font-weight:bold'>Customer will stay</span>", unsafe_allow_html=True)

    st.subheader("Prediction Probability")
    st.write(f"Stay: {prediction_proba[0][0]*100:.2f}%")
    st.write(f"Churn: {prediction_proba[0][1]*100:.2f}%")