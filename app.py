from flask import Flask, render_template, request
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# --- DATA PREPARATION & MODEL TRAINING ---
df_original = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = df_original.copy()

# Clean TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Encode categorical columns
encoders = {}
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Features & Target
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn']

# Train model once at startup
model = xgb.XGBClassifier(eval_metric='logloss')
model.fit(X, y)

def get_risk_level(prob):
    """Categorizes risk based on probability """
    if prob < 0.3:
        return "Low risk"
    elif prob < 0.6:
        return "Medium risk"
    else:
        return "High risk"

@app.route('/', methods=['GET', 'POST'])
def index():
    customer_ids = df_original['customerID'].tolist()
    prediction_results = None

    if request.method == 'POST':
        # Get threshold from form (defaults to 0.5 as per Page 1) [cite: 6]
        threshold = float(request.form.get('threshold', 0.5)) / 100 
        
        # Capture form data
        input_data = {
            'gender': request.form.get('gender'),
            'SeniorCitizen': int(request.form.get('SeniorCitizen', 0)),
            'Partner': request.form.get('Partner'),
            'Dependents': request.form.get('Dependents'),
            'tenure': float(request.form.get('tenure', 0)),
            'PhoneService': request.form.get('PhoneService'),
            'MultipleLines': request.form.get('MultipleLines'),
            'InternetService': request.form.get('InternetService'),
            'OnlineSecurity': request.form.get('OnlineSecurity'),
            'OnlineBackup': request.form.get('OnlineBackup'),
            'DeviceProtection': request.form.get('DeviceProtection'),
            'TechSupport': request.form.get('TechSupport'),
            'StreamingTV': request.form.get('StreamingTV'),
            'StreamingMovies': request.form.get('StreamingMovies'),
            'Contract': request.form.get('Contract'),
            'PaperlessBilling': request.form.get('PaperlessBilling'),
            'PaymentMethod': request.form.get('PaymentMethod'),
            'MonthlyCharges': float(request.form.get('MonthlyCharges', 0)),
            'TotalCharges': float(request.form.get('TotalCharges', 0))
        }

        input_df = pd.DataFrame([input_data])

        # Encode categorical inputs safely
        for col in input_df.columns:
            if col in encoders:
                try:
                    input_df[col] = encoders[col].transform(input_df[col])
                except:
                    input_df[col] = 0 # Default fallback

        # Align columns with training data X
        input_df = input_df[X.columns]

        # Generate Probabilities
        prob_churn = model.predict_proba(input_df)[0][1]
        prob_stay = 1 - prob_churn

        # Map to the format seen in Output File [cite: 17, 18, 19, 20]
        prediction_results = {
            'status': "CHURN" if prob_churn >= threshold else "STAY",
            'stay_prob': round(prob_stay * 100, 2),
            'churn_prob': round(prob_churn * 100, 2),
            'risk_level': get_risk_level(prob_churn),
            'raw_prob': round(prob_churn, 2)
        }

    return render_template('index.html', 
                           customer_ids=customer_ids, 
                           results=prediction_results)
    
    @app.route('/get_customer_data/<customer_id>')
    def get_customer_data(customer_id):
        """API to fetch customer data for auto-fill"""
    # Filter original dataframe for the selected ID
    row = df_original[df_original['customerID'] == customer_id]
    
    if not row.empty:
        # Convert the row to a dictionary and return as JSON
        data = row.to_dict(orient='records')[0]
        
        # Ensure numeric values are JSON serializable
        for key, value in data.items():
            if pd.api.types.is_number(value) and pd.isna(value):
                data[key] = 0
                
        return jsonify(data)
    else:
        return jsonify({"error": "Customer not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)

    # Display prediction probabilities
    st.subheader("Prediction Probability")
    st.write(f"Stay: {prediction_proba[0][0]*100:.2f}%")
    st.write(f"Churn: {prediction_proba[0][1]*100:.2f}%")
