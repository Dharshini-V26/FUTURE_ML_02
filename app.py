import streamlit as st
import joblib
import pandas as pd
import time

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

# -------------------------------------------------
# LOAD TRAINED MODEL (BEST MODEL)
# -------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/churn_model.pkl")

model = load_model()

# -------------------------------------------------
# PREMIUM WHITE UI STYLING
# -------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background-color: #ffffff;
    color: #000000;
}

/* Titles */
.title {
    font-size: 3rem;
    font-weight: 700;
    text-align: center;
}
.subtitle {
    text-align: center;
    color: #555;
    margin-bottom: 30px;
}

/* Cards */
.card {
    background: #ffffff;
    padding: 25px;
    border-radius: 16px;
    border: 1px solid #eaeaea;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    transition: all 0.4s ease;
}
.card:hover {
    transform: translateY(-8px);
    box-shadow: 0 22px 45px rgba(0,0,0,0.15);
}

/* KPI */
.kpi {
    text-align: center;
}
.kpi h1 {
    font-size: 2.2rem;
}
.kpi p {
    color: #666;
}

/* Risk styles */
.low { border-left: 16px solid #2ecc71; }
.medium { border-left: 16px solid #f1c40f; }
.high { border-left: 16px solid #e74c3c; }

/* Buttons */
.stButton>button {
    background: #000000;
    color: #ffffff;
    border-radius: 9px;
    padding: 9px 18px;
    font-weight: 600;
    border: none;
    transition: 0.3s ease;
}
.stButton>button:hover {
    background: #333333;
    transform: scale(1.05);
}

/* Animation */
.fade {
    animation: fadeUp 0.9s ease-in;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Divider */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(to right, transparent, #ccc, transparent);
    margin: 40px 0;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown("""
<div class="title fade">Customer Churn Prediction System</div>
<div class="subtitle fade">
AI-powered churn risk analysis for business decision making
</div>
<hr>
""", unsafe_allow_html=True)

# -------------------------------------------------
# INPUT SECTION
# -------------------------------------------------
st.markdown("## Customer Information")

c1, c2, c3 = st.columns(3)

with c1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    monthly = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0)
    total = st.number_input("Total Charges ($)", 0.0, 9000.0, 1500.0)
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    payment = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# MODEL INPUT FORMAT (MATCHES TRAINING)
# -------------------------------------------------
input_df = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": total
}])

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("Predict Churn Risk"):
    with st.spinner("Analyzing customer behavior..."):
        time.sleep(1.2)

    churn_prob = model.predict_proba(input_df)[0][1] * 100

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("## Prediction Result")

    # KPI SUMMARY
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(f"<div class='card kpi fade'><h1>{churn_prob:.1f}%</h1><p>Churn Probability</p></div>", unsafe_allow_html=True)
    with k2:
        st.markdown(f"<div class='card kpi fade'><h1>{tenure} mo</h1><p>Tenure</p></div>", unsafe_allow_html=True)
    with k3:
        st.markdown(f"<div class='card kpi fade'><h1>${monthly:.0f}</h1><p>Monthly Charges</p></div>", unsafe_allow_html=True)

    # Risk Classification
    if churn_prob >= 65:
        risk, css, msg = "High", "high", "Immediate retention action required."
    elif churn_prob >= 35:
        risk, css, msg = "Medium", "medium", "Customer engagement recommended."
    else:
        risk, css, msg = "Low", "low", "Customer likely to stay."

    st.markdown(f"""
    <div class="card {css} fade">
        <h2>{risk} Churn Risk</h2>
        <h1>{churn_prob:.2f}%</h1>
        <p>{msg}</p>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------
    # BUSINESS INSIGHTS
    # -------------------------------------------------
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("## Key Business Insights")

    i1, i2 = st.columns(2)

    with i1:
        if contract == "Month-to-month":
            st.markdown("<div class='card fade'><h4>📄 Contract Risk</h4><p>Month-to-month contracts have higher churn rates.</p></div>", unsafe_allow_html=True)
        if tenure < 12:
            st.markdown("<div class='card fade'><h4>⏳ New Customer Risk</h4><p>Low-tenure customers churn more frequently.</p></div>", unsafe_allow_html=True)

    with i2:
        if monthly > 70:
            st.markdown("<div class='card fade'><h4>💸 Pricing Impact</h4><p>Higher monthly charges increase churn probability.</p></div>", unsafe_allow_html=True)
        st.markdown("<div class='card fade'><h4>🤖 Model Insight</h4><p>Prediction generated using a trained Logistic Regression churn model.</p></div>", unsafe_allow_html=True)


