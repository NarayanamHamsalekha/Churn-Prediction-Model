from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# 1️⃣ Load saved model (same as Streamlit)
with open("best_xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load CSV
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Preprocess CSV (like your Streamlit code)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype('category')


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = ""
    prediction_prob = ""
    # Set default threshold
    threshold = 50

    if request.method == "POST":
        # Get input from HTML form (all fields same as your Streamlit)
        input_data = {
            'gender': request.form['gender'],
            'SeniorCitizen': int(request.form['SeniorCitizen']),
            'Partner': request.form['Partner'],
            'Dependents': request.form['Dependents'],
            'tenure': int(request.form['tenure']),
            'PhoneService': request.form['PhoneService'],
            'MultipleLines': request.form['MultipleLines'],
            'InternetService': request.form['InternetService'],
            'OnlineSecurity': request.form['OnlineSecurity'],
            'OnlineBackup': request.form['OnlineBackup'],
            'DeviceProtection': request.form['DeviceProtection'],
            'TechSupport': request.form['TechSupport'],
            'StreamingTV': request.form['StreamingTV'],
            'StreamingMovies': request.form['StreamingMovies'],
            'Contract': request.form['Contract'],
            'PaperlessBilling': request.form['PaperlessBilling'],
            'PaymentMethod': request.form['PaymentMethod'],
            'MonthlyCharges': float(request.form['MonthlyCharges']),
            'TotalCharges': float(request.form['TotalCharges'])
        }

        threshold = int(request.form.get('threshold', 50))

        features = pd.DataFrame(input_data, index=[0])
        for col in features.select_dtypes(include='object').columns:
            features[col] = features[col].astype('category')

# 4️⃣ Predict button
if st.button("Predict"):
    # Make prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # Set prediction text with color
    if prediction[0] == 1 or (prediction_proba[0][1]*100 >= threshold):
        prediction_text = "<span style='color:red;font-weight:bold'>Customer will churn</span>"
    else:
        prediction_text = "<span style='color:green;font-weight:bold'>Customer will stay</span>"

    # Display prediction
    st.subheader("Prediction")
    st.markdown(prediction_text, unsafe_allow_html=True)

    # Display prediction probabilities
    st.subheader("Prediction Probability")
    st.write(f"Stay: {prediction_proba[0][0]*100:.2f}%")
    st.write(f"Churn: {prediction_proba[0][1]*100:.2f}%")
