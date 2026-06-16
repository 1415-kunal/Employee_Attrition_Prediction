"""
=========================================================
 HR Analytics — Employee Attrition Prediction Dashboard
 Built with Streamlit | Model: Logistic Regression
=========================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AttritionIQ — HR Attrition Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ───────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

/* ── Reset & base ───────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ─────────────────────────────── */
.stApp {
    background: #0d1117;
    color: #e6edf3;
}

/* ── Sidebar ────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * {
    color: #c9d1d9 !important;
}

/* ── Main header ────────────────────────────── */
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #58a6ff 0%, #a371f7 60%, #f78166 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-size: 1.05rem;
    color: #8b949e;
    margin-bottom: 0;
    font-weight: 400;
}
.pill {
    display: inline-block;
    background: #1f2937;
    border: 1px solid #374151;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.75rem;
    color: #9ca3af;
    margin: 4px 4px 4px 0;
}

/* ── Section headers ────────────────────────── */
.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: #e6edf3;
    padding: 10px 0 6px 0;
    border-bottom: 2px solid #21262d;
    margin-bottom: 14px;
    letter-spacing: 0.01em;
}
.section-icon {
    margin-right: 8px;
}

/* ── Cards ──────────────────────────────────── */
.card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 22px 24px;
    margin-bottom: 16px;
}
.card-accent {
    border-left: 3px solid #58a6ff;
}

/* ── Streamlit Tabs override ─────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #161b22;
    border-radius: 12px 12px 0 0;
    border: 1px solid #21262d;
    border-bottom: none;
    padding: 6px 8px 0 8px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px 8px 0 0;
    color: #8b949e;
    font-size: 0.85rem;
    font-weight: 500;
    padding: 8px 16px;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: #0d1117 !important;
    color: #58a6ff !important;
    font-weight: 600 !important;
    border-bottom: 2px solid #58a6ff;
}
.stTabs [data-baseweb="tab-panel"] {
    background: #0d1117;
    border: 1px solid #21262d;
    border-top: none;
    border-radius: 0 0 12px 12px;
    padding: 20px 16px;
}

/* ── Result block ───────────────────────────── */
.result-stay {
    background: linear-gradient(135deg, #0d2318 0%, #0f2922 100%);
    border: 1px solid #238636;
    border-radius: 14px;
    padding: 28px 30px;
}
.result-leave {
    background: linear-gradient(135deg, #2d1516 0%, #3b1e1e 100%);
    border: 1px solid #da3633;
    border-radius: 14px;
    padding: 28px 30px;
}
.result-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 4px;
}
.result-prob {
    font-size: 3.6rem;
    font-weight: 700;
    font-family: 'Space Grotesk', sans-serif;
    line-height: 1;
}

/* ── Risk badge ─────────────────────────────── */
.badge-low    { background:#0f2922; border:1px solid #238636; color:#3fb950; border-radius:20px; padding:5px 16px; font-size:0.85rem; font-weight:600; display:inline-block; }
.badge-medium { background:#2d2208; border:1px solid #9e6a03; color:#e3b341; border-radius:20px; padding:5px 16px; font-size:0.85rem; font-weight:600; display:inline-block; }
.badge-high   { background:#2d1516; border:1px solid #da3633; color:#f85149; border-radius:20px; padding:5px 16px; font-size:0.85rem; font-weight:600; display:inline-block; }

/* ── Recommendation cards ───────────────────── */
.rec-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
}
.rec-icon {
    font-size: 1.3rem;
    flex-shrink: 0;
    margin-top: 1px;
}
.rec-text {
    font-size: 0.9rem;
    color: #c9d1d9;
    line-height: 1.5;
}
.rec-text strong {
    color: #e6edf3;
    display: block;
    margin-bottom: 2px;
}

/* ── Progress bar override ───────────────────── */
.stProgress > div > div {
    border-radius: 8px;
}

/* ── Divider ────────────────────────────────── */
.divider {
    border: none;
    border-top: 1px solid #21262d;
    margin: 24px 0;
}

/* ── Footer ─────────────────────────────────── */
.footer {
    text-align: center;
    color: #6e7681;
    font-size: 0.8rem;
    padding: 32px 0 16px 0;
    border-top: 1px solid #21262d;
    margin-top: 40px;
}
.footer a { color: #58a6ff; text-decoration: none; }
.footer a:hover { text-decoration: underline; }

/* ── Streamlit widget overrides ─────────────── */
.stSelectbox > div, .stNumberInput > div {
    background: #0d1117 !important;
}
label {
    color: #8b949e !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #1f6feb 0%, #a371f7 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 14px 40px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 1.05rem;
    letter-spacing: 0.02em;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover {
    opacity: 0.88;
    color: white;
    border: none;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "attrition_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding:18px 0 8px 0'>"
        "<div style='font-family:Space Grotesk,sans-serif;font-size:1.5rem;font-weight:700;"
        "background:linear-gradient(135deg,#58a6ff,#a371f7);-webkit-background-clip:text;"
        "-webkit-text-fill-color:transparent;background-clip:text'>🧠 AttritionIQ</div>"
        "<div style='font-size:0.78rem;color:#6e7681;margin-top:4px'>HR Intelligence Platform</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # About section
    st.markdown(
        "<div style='font-size:0.8rem;font-weight:600;color:#c9d1d9;text-transform:uppercase;"
        "letter-spacing:0.08em;margin-bottom:10px'>📌 About This Tool</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:0.82rem;color:#8b949e;line-height:1.7'>"
        "This dashboard uses machine learning to predict employee attrition risk, "
        "helping HR teams take proactive steps to retain top talent before it's too late."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # How to use
    st.markdown(
        "<div style='font-size:0.8rem;font-weight:600;color:#c9d1d9;text-transform:uppercase;"
        "letter-spacing:0.08em;margin-bottom:10px'>📋 How to Use</div>",
        unsafe_allow_html=True,
    )
    steps = [
        ("1", "Fill in employee details across the 5 input tabs"),
        ("2", "Click the Predict Attrition Risk button"),
        ("3", "Review the risk score and probability"),
        ("4", "Act on the HR recommendations provided"),
    ]
    for num, txt in steps:
        st.markdown(
            f"<div style='display:flex;gap:10px;align-items:flex-start;margin-bottom:8px'>"
            f"<span style='background:#1f6feb;color:white;border-radius:50%;width:20px;height:20px;"
            f"font-size:0.7rem;font-weight:700;display:flex;align-items:center;justify-content:center;"
            f"flex-shrink:0'>{num}</span>"
            f"<span style='font-size:0.82rem;color:#8b949e;line-height:1.5'>{txt}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Risk legend
    st.markdown(
        "<div style='font-size:0.8rem;font-weight:600;color:#c9d1d9;text-transform:uppercase;"
        "letter-spacing:0.08em;margin-bottom:10px'>🎯 Risk Levels</div>",
        unsafe_allow_html=True,
    )
    for color, label, rng in [
        ("#3fb950", "Low Risk",    "0 – 30%"),
        ("#e3b341", "Medium Risk", "30 – 60%"),
        ("#f85149", "High Risk",   "60 – 100%"),
    ]:
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"padding:6px 0;border-bottom:1px solid #21262d'>"
            f"<span style='color:{color};font-size:0.82rem;font-weight:600'>● {label}</span>"
            f"<span style='color:#6e7681;font-size:0.78rem'>{rng}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    if model is None:
        st.warning("⚠️ Model file not found.\nPlace `attrition_model.pkl` in the app directory.")
    else:
        st.success("✅ Model loaded successfully")


# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown(
    "<div class='hero-title'>Employee Attrition Risk Predictor</div>"
    "<div class='hero-sub'>AI-powered workforce retention intelligence — "
    "identify flight risk early and act before talent walks out the door</div>"
    "<div style='margin-top:14px'>"
    "<span class='pill'>🤖 Machine Learning</span>"
    "<span class='pill'>📦 IBM HR Dataset</span>"
    "<span class='pill'>⚡ Real-time Prediction</span>"
    "<span class='pill'>🎯 HR Analytics</span>"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# INPUT FORM  — Tab-based layout
# ─────────────────────────────────────────────
st.markdown(
    "<div class='section-header'>"
    "<span class='section-icon'>📋</span>Employee Profile"
    "</div>",
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "👤 Employee Info",
    "💼 Job Information",
    "💰 Compensation",
    "📅 Experience",
    "😊 Satisfaction",
])

def satisfaction_slider(label, key):
    return st.select_slider(
        label,
        options=[1, 2, 3, 4],
        value=3,
        key=key,
        format_func=lambda x: {1: "1 · Low", 2: "2 · Medium", 3: "3 · High", 4: "4 · Very High"}[x],
    )

# ── TAB 1 · Employee Info ───────────────────
with tab1:
    st.markdown("<div class='card card-accent'>", unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        age        = st.number_input("Age", min_value=18, max_value=65, value=35, step=1)
        gender     = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    with c2:
        age_group  = st.selectbox("Age Group", ["18-25", "26-35", "36-45", "46-60"])
        distance   = st.number_input("Distance from Home (km)", min_value=1, max_value=100, value=10, step=1)
    st.markdown("</div>", unsafe_allow_html=True)

# ── TAB 2 · Job Information ─────────────────
with tab2:
    st.markdown("<div class='card card-accent'>", unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
        job_role   = st.selectbox(
            "Job Role",
            ["Sales Executive", "Research Scientist", "Laboratory Technician",
             "Manufacturing Director", "Healthcare Representative", "Manager",
             "Sales Representative", "Research Director", "Human Resources"],
        )
        education_field = st.selectbox(
            "Education Field",
            ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"],
        )
    with c2:
        business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
        overtime        = st.selectbox("Overtime", ["Yes", "No"])
    st.markdown("</div>", unsafe_allow_html=True)

# ── TAB 3 · Compensation ────────────────────
with tab3:
    st.markdown("<div class='card card-accent'>", unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=20000, value=5000, step=100)
        daily_rate     = st.number_input("Daily Rate ($)", min_value=100, max_value=1500, value=800, step=10)
    with c2:
        job_level    = st.number_input("Job Level (1–5)", min_value=1, max_value=5, value=2, step=1)
        stock_option = st.number_input("Stock Option Level (0–3)", min_value=0, max_value=3, value=1, step=1)
    st.markdown("</div>", unsafe_allow_html=True)

# ── TAB 4 · Experience ──────────────────────
with tab4:
    st.markdown("<div class='card card-accent'>", unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        total_working_years  = st.number_input("Total Working Years",        min_value=0, max_value=40, value=10, step=1)
        num_companies        = st.number_input("Number of Companies Worked", min_value=0, max_value=15, value=3,  step=1)
        years_at_company     = st.number_input("Years at Company",           min_value=0, max_value=40, value=5,  step=1)
        years_in_role        = st.number_input("Years in Current Role",      min_value=0, max_value=20, value=3,  step=1)
    with c2:
        years_since_promotion = st.number_input("Years Since Last Promotion",   min_value=0, max_value=15, value=2, step=1)
        years_with_manager    = st.number_input("Years with Current Manager",   min_value=0, max_value=20, value=3, step=1)
        training_times        = st.number_input("Training Times Last Year",     min_value=0, max_value=10, value=3, step=1)
        tenure_group          = st.selectbox("Tenure Group", ["0-2", "3-5", "6-10", "10+"])
    st.markdown("</div>", unsafe_allow_html=True)

# ── TAB 5 · Satisfaction ────────────────────
with tab5:
    st.markdown("<div class='card card-accent'>", unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        env_satisfaction  = satisfaction_slider("Environment Satisfaction", "env")
        job_satisfaction  = satisfaction_slider("Job Satisfaction",         "job")
        rel_satisfaction  = satisfaction_slider("Relationship Satisfaction","rel")
    with c2:
        job_involvement   = satisfaction_slider("Job Involvement",          "inv")
        work_life_balance = satisfaction_slider("Work-Life Balance",        "wlb")
    st.markdown("</div>", unsafe_allow_html=True)

# ── Engineered features (internal — not displayed) ──
promo_ratio   = round(years_since_promotion / (years_at_company + 1), 4)
manager_ratio = round(years_with_manager   / (years_at_company + 1), 4)
avg_yrs_co    = round(total_working_years  / (num_companies     + 1), 4)
income_per_jl = round(monthly_income       / (job_level         + 1), 4)


# ─────────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_clicked = st.button("🔍  Analyze Attrition Risk", use_container_width=True)


# ─────────────────────────────────────────────
# PREDICTION  LOGIC
# ─────────────────────────────────────────────
def get_recommendations(risk: str, prob: float, inputs: dict) -> list[dict]:
    """Return personalised HR recommendations based on risk profile."""
    recs = []

    if risk == "High":
        recs.append({
            "icon": "💰",
            "title": "Compensation Review",
            "text": "Schedule an immediate compensation benchmarking session. "
                    f"Current monthly income of ${inputs['monthly_income']:,} may be below market rate for this role.",
        })
        recs.append({
            "icon": "🚀",
            "title": "Career Path Conversation",
            "text": f"With {inputs['years_since_promotion']} year(s) since last promotion, "
                    "initiate a structured career development conversation and create a clear advancement roadmap.",
        })
        recs.append({
            "icon": "🤝",
            "title": "Retention Interview",
            "text": "Conduct a confidential stay interview within the next 2 weeks to identify "
                    "specific pain points and co-create solutions before the employee disengages further.",
        })
        if inputs['overtime'] == "Yes":
            recs.append({
                "icon": "⚖️",
                "title": "Workload Rebalancing",
                "text": "Persistent overtime is a strong attrition driver. Audit workload distribution, "
                        "explore delegation or headcount additions, and set clear boundaries on working hours.",
            })

    elif risk == "Medium":
        recs.append({
            "icon": "📈",
            "title": "Engagement Check-in",
            "text": "Schedule a quarterly one-on-one focused on growth, recognition, and team dynamics "
                    "to proactively address any emerging disengagement.",
        })
        recs.append({
            "icon": "🎓",
            "title": "Learning & Development",
            "text": f"With {inputs['training_times']} training session(s) last year, consider increasing "
                    "L&D investment — skill-building directly correlates with retention.",
        })
        if inputs['business_travel'] == "Travel_Frequently":
            recs.append({
                "icon": "✈️",
                "title": "Travel Policy Review",
                "text": "Frequent travel is linked to burnout and attrition. Consider renegotiating travel "
                        "frequency or offering remote/hybrid options where possible.",
            })

    else:  # Low
        recs.append({
            "icon": "✅",
            "title": "Continue Current Practices",
            "text": "This employee shows low attrition risk. Maintain current engagement, "
                    "recognition, and growth practices.",
        })
        recs.append({
            "icon": "🌟",
            "title": "Mentorship Opportunity",
            "text": "High-engagement, low-risk employees make excellent mentors. "
                    "Consider nominating this individual for internal mentorship or leadership programs.",
        })

    # Universal recommendation
    if inputs['stock_option'] == 0:
        recs.append({
            "icon": "📊",
            "title": "Equity Incentives",
            "text": "Offering stock options — even at Level 1 — significantly improves retention. "
                    "Consider including equity as part of the next compensation review.",
        })

    return recs


if predict_clicked:
    if model is None:
        st.error(
            "⚠️ **Model not found.** Place `attrition_model.pkl` in the same directory as this app and restart.",
            icon="🚨",
        )
    else:
        # ── Build feature dict ──────────────────
        input_data = {
            # Numerical
            "Age":                      age,
            "DailyRate":                daily_rate,
            "DistanceFromHome":         distance,
            "EnvironmentSatisfaction":  env_satisfaction,
            "JobInvolvement":           job_involvement,
            "JobLevel":                 job_level,
            "JobSatisfaction":          job_satisfaction,
            "MonthlyIncome":            monthly_income,
            "NumCompaniesWorked":       num_companies,
            "RelationshipSatisfaction": rel_satisfaction,
            "StockOptionLevel":         stock_option,
            "TotalWorkingYears":        total_working_years,
            "TrainingTimesLastYear":    training_times,
            "WorkLifeBalance":          work_life_balance,
            "YearsAtCompany":           years_at_company,
            "YearsInCurrentRole":       years_in_role,
            "YearsSinceLastPromotion":  years_since_promotion,
            "YearsWithCurrManager":     years_with_manager,
            # Engineered
            "YearSincePromotionRatio":  promo_ratio,
            "YearsWithManagerRatio":    manager_ratio,
            "AverageYearsperCompany":   avg_yrs_co,
            "IncomePerJoblevel":        income_per_jl,
            # Categorical
            "BusinessTravel":     business_travel,
            "Department":         department,
            "EducationField":     education_field,
            "Gender":             gender,
            "JobRole":            job_role,
            "MaritalStatus":      marital_status,
            "OverTime":           overtime,
            "AgeGroup":           age_group,
            "TenureGroup":        tenure_group,
        }

        df_input = pd.DataFrame([input_data])

        # ── Predict ─────────────────────────────
        try:
            proba       = model.predict_proba(df_input)[0][1]
            prediction  = model.predict(df_input)[0]
            prob_pct    = round(proba * 100, 1)
            confidence  = round(max(proba, 1 - proba) * 100, 1)

            # Risk category
            if prob_pct < 30:
                risk_level = "Low"
                risk_color = "#3fb950"
                badge_cls  = "badge-low"
                result_cls = "result-stay"
                verdict    = "✅ Likely to Stay"
                verdict_color = "#3fb950"
                bar_color  = "normal"
            elif prob_pct < 60:
                risk_level = "Medium"
                risk_color = "#e3b341"
                badge_cls  = "badge-medium"
                result_cls = "result-stay"
                verdict    = "⚠️ At Risk — Monitor Closely"
                verdict_color = "#e3b341"
                bar_color  = "normal"
            else:
                risk_level = "High"
                risk_color = "#f85149"
                badge_cls  = "badge-high"
                result_cls = "result-leave"
                verdict    = "🚨 Likely to Leave"
                verdict_color = "#f85149"
                bar_color  = "normal"

            # ── Results Section ─────────────────
            st.markdown(
                "<div class='section-header'>"
                "<span class='section-icon'>📊</span>Prediction Results"
                "</div>",
                unsafe_allow_html=True,
            )

            res_col1, res_col2 = st.columns([1, 1], gap="medium")

            with res_col1:
                st.markdown(
                    f"<div class='{result_cls}'>"
                    f"<div style='font-size:0.8rem;color:#8b949e;text-transform:uppercase;"
                    f"letter-spacing:0.08em;margin-bottom:8px'>Prediction</div>"
                    f"<div class='result-title' style='color:{verdict_color}'>{verdict}</div>"
                    f"<div style='margin:16px 0 8px 0'>"
                    f"  <div style='font-size:0.75rem;color:#8b949e;text-transform:uppercase;letter-spacing:0.08em'>Attrition Probability</div>"
                    f"  <div class='result-prob' style='color:{risk_color}'>{prob_pct}%</div>"
                    f"</div>"
                    f"<div style='margin-top:14px'>"
                    f"  <span style='font-size:0.8rem;color:#8b949e;margin-right:10px'>Risk Level</span>"
                    f"  <span class='{badge_cls}'>{risk_level} Risk</span>"
                    f"</div>"
                    f"<div style='margin-top:14px'>"
                    f"  <span style='font-size:0.8rem;color:#8b949e;margin-right:10px'>Model Confidence</span>"
                    f"  <span style='font-size:0.95rem;font-weight:600;color:#c9d1d9'>{confidence}%</span>"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with res_col2:
                st.markdown(
                    "<div style='background:#161b22;border:1px solid #21262d;border-radius:14px;padding:28px 30px;height:100%'>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<div style='font-size:0.8rem;color:#8b949e;text-transform:uppercase;"
                    "letter-spacing:0.08em;margin-bottom:16px'>Risk Gauge</div>",
                    unsafe_allow_html=True,
                )

                # Three zone labels
                g1, g2, g3 = st.columns(3)
                g1.markdown("<div style='font-size:0.72rem;color:#3fb950;text-align:left'>🟢 Low<br><span style='color:#6e7681'>0–30%</span></div>", unsafe_allow_html=True)
                g2.markdown("<div style='font-size:0.72rem;color:#e3b341;text-align:center'>🟡 Medium<br><span style='color:#6e7681'>30–60%</span></div>", unsafe_allow_html=True)
                g3.markdown("<div style='font-size:0.72rem;color:#f85149;text-align:right'>🔴 High<br><span style='color:#6e7681'>60–100%</span></div>", unsafe_allow_html=True)

                st.progress(int(prob_pct))

                st.markdown(
                    f"<div style='text-align:center;margin-top:6px;"
                    f"font-size:0.85rem;color:{risk_color};font-weight:600'>"
                    f"Probability: {prob_pct}%</div>",
                    unsafe_allow_html=True,
                )

                st.markdown("<hr style='border-color:#21262d;margin:18px 0'>", unsafe_allow_html=True)

                # Key factors summary
                st.markdown(
                    "<div style='font-size:0.8rem;color:#8b949e;text-transform:uppercase;"
                    "letter-spacing:0.08em;margin-bottom:12px'>Key Input Summary</div>",
                    unsafe_allow_html=True,
                )
                summary_items = [
                    ("Overtime",        overtime),
                    ("Business Travel", business_travel.replace("_", " ")),
                    ("Job Satisfaction",f"{job_satisfaction}/4"),
                    ("Work-Life Balance",f"{work_life_balance}/4"),
                    ("Years at Company",f"{years_at_company} yrs"),
                    ("Monthly Income",  f"${monthly_income:,}"),
                ]
                for lbl, val in summary_items:
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;"
                        f"padding:4px 0;border-bottom:1px solid #21262d'>"
                        f"<span style='color:#8b949e;font-size:0.78rem'>{lbl}</span>"
                        f"<span style='color:#c9d1d9;font-size:0.78rem;font-weight:500'>{val}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

            # ── Recommendations ─────────────────
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.markdown(
                "<div class='section-header'>"
                "<span class='section-icon'>💡</span>HR Recommendations"
                "</div>",
                unsafe_allow_html=True,
            )

            recs = get_recommendations(
                risk_level, proba,
                {"monthly_income": monthly_income, "years_since_promotion": years_since_promotion,
                 "overtime": overtime, "training_times": training_times,
                 "business_travel": business_travel, "stock_option": stock_option},
            )

            rec_cols = st.columns(2, gap="medium")
            for i, rec in enumerate(recs):
                with rec_cols[i % 2]:
                    st.markdown(
                        f"<div class='rec-card'>"
                        f"<div class='rec-icon'>{rec['icon']}</div>"
                        f"<div class='rec-text'><strong>{rec['title']}</strong>{rec['text']}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info(
                "Ensure `attrition_model.pkl` was trained with the exact same feature set "
                "and that all categorical encodings match."
            )


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown(
    "<div class='footer'>"
    "Built with ❤️ using <strong>Streamlit</strong> &nbsp;|&nbsp; "
    "Model: <strong>Logistic Regression</strong> on IBM HR Analytics Dataset<br>"
    "<span style='font-size:0.75rem;color:#484f58'>"
    "This tool is intended for HR analytics and decision-support only — "
    "predictions should complement, not replace, human judgment."
    "</span>"
    "</div>",
    unsafe_allow_html=True,
)
