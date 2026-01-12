# Customer Churn Prediction System – Project Summary
This project focuses on predicting customer churn probability using Machine Learning and deploying the best-performing model as an interactive web application. The goal is to help businesses identify customers at risk of leaving and take proactive retention actions based on data-driven insights.
### Dataset & Problem Understanding
1) Used the Telco Customer Churn dataset
2) Target variable: Churn (Yes/No)
3) Features include:
   1) Customer demographics
   2) Service usage details
   3) Contract type and billing information

The problem was formulated as a binary classification task.
### Model Development & Evaluation
Three models were trained and evaluated:
1) Logistic Regression
2) Random Forest
3) XGBoost
### Model Comparison (ROC-AUC):
<img width="600" height="227" alt="image" src="https://github.com/user-attachments/assets/20c790e2-b87c-4ff8-9c84-4c1289d4e90a" />

### Why Logistic Regression Was Selected
Although XGBoost is a powerful algorithm, Logistic Regression achieved the highest ROC-AUC score on this dataset.

It also provided:
1) Better generalization
2) Stable probability outputs
3) High interpretability for business decisions

Hence, Logistic Regression was chosen as the final production model.
### ROC Curve Analysis
<img width="770" height="584" alt="image" src="https://github.com/user-attachments/assets/e66dccdf-b7ae-4cf8-98ca-ab00fa6ed89b" />

1) Logistic Regression consistently showed a higher True Positive Rate at lower False Positive Rates
2) This indicates better separation between churned and retained customers
3) ROC analysis confirmed it as the best-performing model
### Web Application (Deployment)
The trained model was deployed using Streamlit with a professional and interactive UI:
### Key Features:
1) User-friendly customer input form
2) Real-time churn probability prediction
3) Color-coded risk classification:

🟢 Low Risk

🟡 Medium Risk

🔴 High Risk

4) Animated KPI cards
5) Automatically generated business insights
### How to Run the Project
1) Clone the repository

git clone https://github.com/Dharshini-V26/FUTURE_ML_02.git

2) Install dependencies

pip install -r requirements.txt

3) Run the web app
   
streamlit run app.py

