# 📊 Customer Churn Prediction App

This project predicts whether a telecom customer is likely to churn (leave the service) based on their account and usage information. The app provides both **probability-based prediction** and **easy-to-use web interface**, making it suitable for business analysts or telecom managers to quickly identify at-risk customers.

The backend is built using **Flask** and machine learning with **XGBoost**, and the frontend uses **HTML/CSS**. The app is deployed live using **Render**.

---

## 🔗 Live App Link

You can access the deployed app here:  

[Click to Open Live App](https://predictive-customer-churn-model-telecom.onrender.com)

---

## 🧩 Project Overview

### 1. Problem Statement
Customer churn is a major challenge in the telecom industry, causing revenue loss and increased customer acquisition costs. Predicting churn allows companies to implement retention strategies proactively.

### 2. Dataset
-Source: Telco Customer Churn (Kaggle)

Features: 19 features including Demographics (Gender, Seniority), Services (Internet, Streaming, Security), and Financials (Tenure, Monthly/Total Charges).

Target: Churn (Yes/No)

### 3. Model
- Algorithm: **XGBoost Classifier**  
- Preprocessing steps:
  - Convert categorical columns to numeric via category encoding  
  - Fill missing `TotalCharges` values with median  
  - Scale/normalize numeric features (if needed)  
- Model outputs:  
  - **Prediction:** Will churn / Will stay  
  - **Probability:** Likelihood of churn  
- Technical Implementation:
  - **Algorithm:** XGBoost Classifier (Extreme Gradient Boosting).
  - **Preprocessing:**  Handled missing TotalCharges values via median imputation.
    Categorical variable encoding using LabelEncoder.
-Key Features:
- **Threshold Slider:** Users can adjust the probability sensitivity (0-100%) to change how strictly the model classifies "Churn."
- **Doughnut Chart:** Real-time visualization of Stay vs. Churn confidence using Chart.js.
- **Auto-Fill:** Fetching real-world customer data by ID for instant testing.

---

## ⚙️ App Features

- **Manual Input:** Enter customer information via web form  
- **Customer ID Input:** Select a customer ID from CSV to predict  
- **Threshold Slider:** Adjust probability threshold for classification  
- **Prediction Probability Display:** See exact likelihood of churn/stay  

---

## 🗂️ Project Structure
Predictive-Churn-Model/
├─ app.py # Main Flask application
├─ best_xgb_model.pkl # Pre-trained XGBoost model
├─ WA_Fn-UseC_-Telco-Customer-Churn.csv # Dataset
├─ requirements.txt # Python dependencies
├─ templates/
│ └─ index.html # HTML template for Flask
├─ static/ # CSS/JS files (optional)
├─ notebooks/ # Jupyter notebooks for model development
├─ Minor Project.ppt/ # Project presentation
└─ Customer_Churn_Prediction_Outputs(1).pd/ # Screenshots of app/model outputs


---

## 💻 How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/NarayanamHamsalekha/Churn-Prediction-Model.git
cd Churn-Prediction-Model

2. Install dependencies:
  pip install -r requirements.txt

3. Run the Flask app:
      python app.py

4. Open a browser and go to:
      https://churn-prediction-model-1-evan.onrender.com

5. Enter customer details or select a Customer ID and click Predict.

---

🛠️ Tech Stack
Backend: Python, Flask, Gunicorn
Machine Learning: XGBoost, Scikit-Learn, Pandas, NumPy
Frontend: HTML5, CSS3 (Bootstrap 5), JavaScript (Chart.js)
Deployment: Render

---

📊 How It Works
User selects Manual Input or Customer ID.
Input features are processed to match the format used during model training.
XGBoost model predicts churn probability.
App compares probability with threshold to give final prediction.
Displays results on the web page with probability breakdown.

---

🛠️ Dependencies
Python 3.x
Flask
pandas
numpy
scikit-learn
XGBoost
gunicorn
jinja2
markupsafe

---


📈 Potential Improvements
Add data visualization dashboards for insights
Integrate user authentication for secure access
Implement API endpoint for external systems to call predictions
Use Docker for containerized deployment

---

📈 Model Performance
The model focuses on high Recall, ensuring that we identify as many potential churners as possible to minimize business loss.
Accuracy: ~80% (Average for Telco Dataset)
Visualization: Integrated Doughnut charts to provide transparency on prediction confidence.

---

## Live App Link

You can access the deployed Customer Churn Prediction App here:

[Click to Open Live App](https://predictive-customer-churn-model-telecom.onrender.com))

---

👤 Author
Narayanam Hamsalekha 
📧 Email: nhamsalekhahamsa@gmail.com
📞 Phone: +91 8431017029
🔗 GitHub Profile: github.com/NarayanamHamsalekha/Churn-Prediction-Model

📄 References
Telco Customer Churn Dataset (Kaggle)
Flask Documentation
Render Deployment
