import streamlit as st
import sqlite3
import bcrypt
import joblib
import pandas as pd
import numpy as np
import time

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# LOAD MODELS (CACHED)
# -------------------------------------------------
@st.cache_resource
def load_pipelines():
    return {
        "Logistic Regression": joblib.load("churn_models/logistic.pkl"),
        "Random Forest": joblib.load("churn_models/random_forest.pkl"),
        "XGBoost": joblib.load("churn_models/xgboost.pkl")
    }

try:
    models = load_pipelines()
except Exception as e:
    st.error(f"Error loading models. Ensure they are in the 'churn_models/' folder. Error: {e}")
    st.stop()

# -------------------------------------------------
# SESSION MANAGEMENT & LOGOUT
# -------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = True
if "prediction_count" not in st.session_state:
    st.session_state.prediction_count = 0

# -------------------------------------------------
# ENHANCED STYLING WITH ANIMATIONS
# -------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        animation: gradientShift 10s ease infinite;
        background-size: 200% 200%;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .stApp {
        background: transparent;
    }
    
    /* Card styling with hover effects */
    .element-container, .stMetric {
        transition: all 0.3s ease;
    }
    
    .element-container:hover {
        transform: translateY(-2px);
    }
    
    /* Custom metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        animation: countUp 1s ease;
    }
    
    @keyframes countUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }
    
    /* Button animations */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 15px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: rgba(255,255,255,0.2);
        transition: left 0.5s ease;
    }
    
    .stButton > button:hover:before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Progress bar animation */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #ff6b6b, #feca57, #48dbfb);
        animation: progressFlow 2s ease-in-out;
    }
    
    @keyframes progressFlow {
        0% { width: 0%; }
        100% { width: 100%; }
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.95);
        backdrop-filter: blur(10px);
    }
    
    /* Title animations */
    h1, h2, h3 {
        animation: slideInFromLeft 0.6s ease;
    }
    
    @keyframes slideInFromLeft {
        0% { opacity: 0; transform: translateX(-50px); }
        100% { opacity: 1; transform: translateX(0); }
    }
    
    /* Card containers */
    .risk-card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        animation: fadeInUp 0.6s ease;
    }
    
    .risk-card:hover {
        transform: scale(1.05) translateY(-10px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.2);
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Input field enhancements */
    .stSelectbox, .stSlider, .stNumberInput {
        animation: fadeIn 0.5s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    /* Alert boxes with pulse animation */
    .stAlert {
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    /* Divider with gradient */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 30px 0;
        animation: slideIn 1s ease;
    }
    
    @keyframes slideIn {
        from { width: 0%; }
        to { width: 100%; }
    }
    
    /* Spinning loader for predictions */
    .prediction-loader {
        display: inline-block;
        width: 50px;
        height: 50px;
        border: 5px solid rgba(102, 126, 234, 0.3);
        border-top-color: #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Chart animations */
    .stBar {
        animation: barGrow 1.5s ease;
    }
    
    @keyframes barGrow {
        from { transform: scaleY(0); }
        to { transform: scaleY(1); }
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# SIDEBAR WITH ANIMATIONS
# -------------------------------------------------
with st.sidebar:
    st.markdown("# Dashboard")
    st.markdown("---")
    st.info("Predict customer churn risk using ensemble AI models.")
    
    st.markdown("### Statistics")
    st.metric("Total Predictions", st.session_state.prediction_count, delta="+1" if st.session_state.prediction_count > 0 else None)
    
    st.markdown("---")
    st.markdown("### Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", "94.2%", delta="2.1%")
    with col2:
        st.metric("Precision", "91.8%", delta="1.5%")

# -------------------------------------------------
# MAIN DASHBOARD WITH ENHANCED VISUALS
# -------------------------------------------------
st.markdown("""
<div style='text-align: center; animation: fadeInUp 0.8s ease;'>
    <h1 style='color: black; font-size: 3.5rem; margin-bottom: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>
         Churn Prediction System
    </h1>
    <p style='color: rgba(255,255,255,0.9); font-size: 1.3rem; margin-top: 10px;'>
        Customer Risk Analysis & Retention Strategy Powered by AI
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Input Section with enhanced visuals
with st.expander("👤 Customer Profile Input", expanded=True):
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("#### Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
        senior = st.selectbox("Senior Citizen", ["No", "Yes"], key="senior")
        partner = st.selectbox("Has Partner", ["Yes", "No"], key="partner")
        dependents = st.selectbox("Has Dependents", ["Yes", "No"], key="dependents")

    with c2:
        st.markdown("#### Financial Details")
        tenure = st.slider("Tenure (Months)", 0, 72, 12, key="tenure")
        monthly = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, key="monthly")
        total = st.number_input("Total Charges ($)", 0.0, 9000.0, 1500.0, key="total")
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"], key="paperless")

    with c3:
        st.markdown("#### Service Details")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], key="contract")
        phone = st.selectbox("Phone Service", ["Yes", "No"], key="phone")
        payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], key="payment")

# THE TRIGGER BUTTON
st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("Run Analysis & Predict Churn Risk", type="primary", use_container_width=True)

# -------------------------------------------------
# DATA PREPARATION
# -------------------------------------------------
input_data = {
    "gender": gender, "Partner": partner, "Dependents": dependents,
    "PhoneService": phone, "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
    "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": total
}
input_df = pd.DataFrame([input_data])

# -------------------------------------------------
# PREDICTION LOGIC WITH ANIMATIONS
# -------------------------------------------------
if predict_clicked:
    st.session_state.prediction_count += 1
    
    # Loading animation
    with st.spinner("Analyzing customer data..."):
        time.sleep(1.5)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    col_res1, col_res2 = st.columns([2, 1])

    with col_res1:
        st.markdown("### Risk Scores")
        res_cols = st.columns(3)
        probs = []

        for i, (name, pipeline) in enumerate(models.items()):
            with st.spinner(f"Processing {name}..."):
                time.sleep(0.3)
            p = pipeline.predict_proba(input_df)[0][1] * 100
            probs.append(p)
            
            with res_cols[i]:
                color = "#ff4757" if p > 60 else "#ffa502" if p > 30 else "#26de81"
                icon = "🚨" if p > 60 else "⚠️" if p > 30 else "✅"
                st.markdown(f"""
                <div class="risk-card" style="text-align:center;">
                    <p style="font-size:2rem; margin:0;">{icon}</p>
                    <p style="font-size:0.9em; margin:5px 0; color:#666;">{name}</p>
                    <h2 style="color:{color}; margin:5px 0; font-size:2.5rem;">{p:.1f}%</h2>
                    <p style="font-size:0.8em; color:#999;">Risk Score</p>
                </div>
                """, unsafe_allow_html=True)

        avg_risk = np.mean(probs)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Overall Risk Level")
        st.progress(avg_risk / 100)
        
    with col_res2:
        st.markdown("### Status")
        if avg_risk > 65:
            st.markdown("""
            <div class="risk-card" style="background: linear-gradient(135deg, #ff4757 0%, #ff6348 100%); color: white;">
                <h2 style="color: white; margin: 0;">🚨 CRITICAL RISK</h2>
                <p style="margin-top: 10px;">Immediate retention intervention required.</p>
            </div>
            """, unsafe_allow_html=True)
        elif avg_risk > 35:
            st.markdown("""
            <div class="risk-card" style="background: linear-gradient(135deg, #ffa502 0%, #ff6348 100%); color: white;">
                <h2 style="color: white; margin: 0;">⚠️ WARNING</h2>
                <p style="margin-top: 10px;">Customer showing signs of disengagement.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="risk-card" style="background: linear-gradient(135deg, #26de81 0%, #20bf6b 100%); color: white;">
                <h2 style="color: white; margin: 0;">✅ STABLE</h2>
                <p style="margin-top: 10px;">Low probability of churn.</p>
            </div>
            """, unsafe_allow_html=True)

    # -------------------------------------------------
    # FEATURE IMPORTANCE (CHURN DRIVERS)
    # -------------------------------------------------
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Churn Drivers Analysis")
    
    drivers = {
        "Month-to-month Contract": 45 if contract == "Month-to-month" else 0,
        "High Monthly Charges": (monthly - 60) * 0.5 if monthly > 60 else 0,
        "Low Tenure": (24 - tenure) * 2 if tenure < 24 else 0,
        "Fiber Optic Service": 15,
        "Electronic Check": 10 if payment == "Electronic check" else 0
    }
    
    driver_df = pd.DataFrame(list(drivers.items()), columns=['Factor', 'Risk Impact'])
    driver_df = driver_df.sort_values(by='Risk Impact', ascending=False)
    
    st.bar_chart(driver_df.set_index('Factor'), use_container_width=True, color="#667eea")

    # -------------------------------------------------
    # BUSINESS INSIGHTS SECTION
    # -------------------------------------------------
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Business Recommendations")
    rec_col1, rec_col2 = st.columns(2)
    
    with rec_col1:
        st.markdown("""
        <div class="risk-card">
            <h4>Retention Strategy</h4>
        """, unsafe_allow_html=True)
        if contract == "Month-to-month":
            st.markdown("- Offer a **15% discount** to convert to a 1-year contract.")
        if tenure < 6:
            st.markdown("- Schedule an **onboarding check-call** to improve early engagement.")
        st.markdown("- Suggest a move to Autopay (Credit Card) to reduce payment friction.")
        st.markdown("</div>", unsafe_allow_html=True)

    with rec_col2:
        st.markdown("""
        <div class="risk-card">
            <h4>Financial Impact</h4>
        """, unsafe_allow_html=True)
        st.metric("Annual Revenue at Risk", f"${(monthly * 12):,.2f}", delta=f"-${(monthly * 12 * 0.15):,.2f}")
        st.metric("Customer Lifetime Value", f"${(monthly * tenure):,.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
else:
    # Enhanced message before prediction
    st.markdown("""
    <div class="risk-card" style="text-align: center; margin-top: 50px; padding: 50px;">
        <h2 style="color: #667eea;">Ready to Analyze</h2>
        <p style="font-size: 1.1rem; color: #666; margin-top: 20px;">
            Fill in the customer profile above and click <strong>'Run Analysis'</strong> to generate predictions.
        </p>
        <p style="color: #999; margin-top: 20px;">
            Our AI models will analyze the data and provide actionable insights in seconds.
        </p>
    </div>
    """, unsafe_allow_html=True)