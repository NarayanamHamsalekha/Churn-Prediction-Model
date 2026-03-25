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

        # Prediction
        prediction = model.predict(features)
        prediction_proba = model.predict_proba(features)

        # Same logic as your Streamlit code
        if prediction[0] == 1 or (prediction_proba[0][1]*100 >= threshold):
            prediction_text = "Customer will churn"
        else:
            prediction_text = "Customer will stay"

        prediction_prob = f"Stay: {prediction_proba[0][0]*100:.2f}%, Churn: {prediction_proba[0][1]*100:.2f}%"

    return render_template("index.html", prediction_text=prediction_text, prediction_prob=prediction_prob, threshold=threshold)


if __name__ == "__main__":
    app.run(debug=True)