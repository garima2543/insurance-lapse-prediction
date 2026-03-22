import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Lapse Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Syne', sans-serif !important;
    }
    .main { background-color: #0f1117; }

    .risk-high {
        background: linear-gradient(135deg, #ff4b4b22, #ff4b4b11);
        border: 1.5px solid #ff4b4b;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .risk-medium {
        background: linear-gradient(135deg, #ffa50022, #ffa50011);
        border: 1.5px solid #ffa500;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .risk-low {
        background: linear-gradient(135deg, #00c85322, #00c85311);
        border: 1.5px solid #00c853;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        border: 1px solid #2e3250;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 32px;
        font-family: 'Syne', sans-serif;
        font-weight: 600;
        font-size: 16px;
        width: 100%;
        cursor: pointer;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2, #667eea);
    }
</style>
""", unsafe_allow_html=True)


# ── LOAD MODEL ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load("xgb_insurance_lapse.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_model()
    model_loaded = True
except:
    model_loaded = False


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("<h1 style='text-align:center; font-size:2.8rem;'>🛡️ Insurance Lapse Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#aaa; font-size:1.1rem;'>Predict which policyholders are at risk of lapsing — powered by XGBoost</p>", unsafe_allow_html=True)
st.markdown("---")

if not model_loaded:
    st.error("⚠️ Model files not found. Make sure `xgb_insurance_lapse.pkl` and `scaler.pkl` are in the same directory as `app.py`.")
    st.stop()


# ── SIDEBAR — INPUT FORM ──────────────────────────────────────────────────────
st.sidebar.markdown("## 👤 Customer Details")
st.sidebar.markdown("Fill in the policyholder information below:")

with st.sidebar:
    age             = st.slider("Age", 18, 75, 35)
    gender          = st.selectbox("Gender", ["Male", "Female"])
    marital_status  = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    region          = st.selectbox("Region", ["Urban", "Semi-Urban", "Rural"])
    income_bracket  = st.selectbox("Income Bracket", ["Low", "Medium", "High"])

    st.markdown("---")
    st.markdown("**📋 Policy Details**")
    policy_type     = st.selectbox("Policy Type", ["Term", "Endowment", "ULIP", "Whole Life"])
    policy_tenure   = st.slider("Policy Tenure (years)", 1, 20, 5)
    premium_amount  = st.number_input("Annual Premium (₹)", 5000, 100000, 15000, step=1000)
    sum_assured     = st.number_input("Sum Assured (₹)", 50000, 2000000, 300000, step=10000)
    payment_mode    = st.selectbox("Payment Mode", ["Monthly", "Quarterly", "Yearly"])

    st.markdown("---")
    st.markdown("**⚠️ Risk Indicators**")
    missed_payments = st.slider("Missed Payments", 0, 10, 0)
    num_claims      = st.slider("Number of Claims", 0, 10, 0)
    agent_contact   = st.selectbox("Agent Contacted?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    loan_on_policy  = st.selectbox("Loan on Policy?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    predict_btn = st.button("🔍 Predict Lapse Risk")


# ── FEATURE ENGINEERING + ENCODING ───────────────────────────────────────────
def prepare_input(age, gender, marital_status, region, income_bracket,
                  policy_type, policy_tenure, premium_amount, sum_assured,
                  payment_mode, missed_payments, num_claims, agent_contact, loan_on_policy):

    income_map = {"Low": 200000, "Medium": 600000, "High": 1500000}
    premium_to_income_ratio = premium_amount / income_map[income_bracket]
    claim_rate = num_claims / (policy_tenure + 1)
    high_risk_payment = int(missed_payments >= 2 and payment_mode == "Monthly")

    cat_map = {
        "gender":         {"Female": 0, "Male": 1},
        "payment_mode":   {"Monthly": 0, "Quarterly": 1, "Yearly": 2},
        "income_bracket": {"High": 0, "Low": 1, "Medium": 2},
        "policy_type":    {"Endowment": 0, "Term": 1, "ULIP": 2, "Whole Life": 3},
        "marital_status": {"Divorced": 0, "Married": 1, "Single": 2},
        "region":         {"Rural": 0, "Semi-Urban": 1, "Urban": 2},
    }

    data = {
        "age":                     age,
        "gender":                  cat_map["gender"][gender],
        "marital_status":          cat_map["marital_status"][marital_status],
        "region":                  cat_map["region"][region],
        "income_bracket":          cat_map["income_bracket"][income_bracket],
        "policy_type":             cat_map["policy_type"][policy_type],
        "policy_tenure":           policy_tenure,
        "premium_amount":          premium_amount,
        "sum_assured":             sum_assured,
        "payment_mode":            cat_map["payment_mode"][payment_mode],
        "missed_payments":         missed_payments,
        "num_claims":              num_claims,
        "agent_contact":           agent_contact,
        "loan_on_policy":          loan_on_policy,
        "premium_to_income_ratio": premium_to_income_ratio,
        "claim_rate":              claim_rate,
        "high_risk_payment":       high_risk_payment,
    }
    return pd.DataFrame([data])


# ── MAIN PANEL ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("<div class='metric-card'><h4>🤖 Model</h4><p>XGBoost Classifier</p></div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div class='metric-card'><h4>📊 Dataset</h4><p>5,000 Policies</p></div>", unsafe_allow_html=True)
with col3:
    st.markdown("<div class='metric-card'><h4>🎯 ROC-AUC</h4><p>~0.91</p></div>", unsafe_allow_html=True)

st.markdown("###")

if predict_btn:
    input_df = prepare_input(
        age, gender, marital_status, region, income_bracket,
        policy_type, policy_tenure, premium_amount, sum_assured,
        payment_mode, missed_payments, num_claims, agent_contact, loan_on_policy
    )

    prob       = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]
    percent    = round(prob * 100, 1)

    if prob >= 0.6:
        risk_label = "HIGH RISK"
        risk_class = "risk-high"
        risk_color = "#ff4b4b"
        action     = "🚨 Immediate retention action required — assign agent, offer premium relief or loyalty bonus."
    elif prob >= 0.3:
        risk_label = "MEDIUM RISK"
        risk_class = "risk-medium"
        risk_color = "#ffa500"
        action     = "⚠️ Monitor closely — schedule agent follow-up call within 2 weeks."
    else:
        risk_label = "LOW RISK"
        risk_class = "risk-low"
        risk_color = "#00c853"
        action     = "✅ Customer is stable — continue regular engagement."

    # Result card
    st.markdown(f"""
    <div class='{risk_class}'>
        <h2 style='color:{risk_color}; font-family:Syne,sans-serif; margin:0;'>{risk_label}</h2>
        <h1 style='font-size:3.5rem; margin:8px 0; font-family:Syne,sans-serif;'>{percent}%</h1>
        <p style='color:#ccc; margin:0;'>Lapse Probability</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"### 💼 Recommended Action\n{action}")
    st.markdown("---")

    # Gauge chart
    fig, ax = plt.subplots(figsize=(6, 1.2))
    fig.patch.set_facecolor("#1e2130")
    ax.set_facecolor("#1e2130")
    ax.barh(0, 100, color="#2e3250", height=0.5, edgecolor="none")
    ax.barh(0, percent, color=risk_color, height=0.5, edgecolor="none")
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xticks([0, 30, 60, 100])
    ax.set_xticklabels(["0%", "30%\nLow→Med", "60%\nMed→High", "100%"], color="#aaa", fontsize=8)
    ax.spines[:].set_visible(False)
    ax.set_title(f"Lapse Risk Gauge — {percent}%", color="white", fontsize=10, pad=8)
    st.pyplot(fig)
    plt.close()

    # Key risk factors
    st.markdown("### 🔑 Key Risk Factors for This Customer")
    factors = []
    if missed_payments >= 2:   factors.append(("❌ Missed Payments ≥ 2", "High impact"))
    if income_bracket == "Low": factors.append(("❌ Low Income Bracket", "High impact"))
    if payment_mode == "Monthly": factors.append(("⚠️ Monthly Payment Mode", "Medium impact"))
    if loan_on_policy == 1:    factors.append(("⚠️ Loan on Policy", "Medium impact"))
    if agent_contact == 0:     factors.append(("⚠️ No Agent Contact", "Medium impact"))
    if policy_tenure <= 3:     factors.append(("⚠️ Short Policy Tenure", "Medium impact"))
    if not factors:            factors.append(("✅ No major risk factors detected", "Low risk profile"))

    for f, impact in factors:
        st.markdown(f"- **{f}** — *{impact}*")

else:
    st.info("👈 Fill in the customer details in the sidebar and click **Predict Lapse Risk**")
    st.markdown("### 📖 How It Works")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 1️⃣ Enter Details\nFill in the policyholder's demographic and policy information in the sidebar.")
    with c2:
        st.markdown("#### 2️⃣ Get Prediction\nThe XGBoost model calculates the lapse probability instantly.")
    with c3:
        st.markdown("#### 3️⃣ Take Action\nUse the risk segment and recommended action to retain the customer.")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<p style='text-align:center; color:#555; font-size:0.85rem;'>Insurance Lapse Predictor • Built with Streamlit & XGBoost</p>", unsafe_allow_html=True)
