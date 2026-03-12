import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import io
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Assessor | Indian Bank",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Header */
.main-header {
    background: linear-gradient(135deg, #0f1923 0%, #1a2d3d 50%, #0f1923 100%);
    border-bottom: 2px solid #00d4aa;
    padding: 28px 32px 20px;
    margin: -1rem -1rem 2rem -1rem;
}
.main-header h1 {
    font-family: 'IBM Plex Mono', monospace;
    color: #00d4aa;
    font-size: 1.6rem;
    font-weight: 600;
    margin: 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #7a9bb5;
    font-size: 0.82rem;
    margin: 6px 0 0;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.5px;
}

/* Decision banner */
.decision-banner {
    border-radius: 6px;
    padding: 22px 28px;
    margin-bottom: 24px;
    border-left: 5px solid;
    font-family: 'IBM Plex Sans', sans-serif;
}
.decision-banner h2 {
    margin: 0 0 6px;
    font-size: 1.5rem;
    font-weight: 700;
    font-family: 'IBM Plex Mono', monospace;
}
.decision-banner p {
    margin: 0;
    font-size: 0.95rem;
    opacity: 0.9;
}

/* Metric cards */
.metric-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 16px 20px;
    text-align: center;
}
.metric-card .value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    line-height: 1;
}
.metric-card .label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 4px;
}

/* Prob bar */
.prob-row { margin-bottom: 10px; }
.prob-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 4px;
    color: #334155;
}
.prob-track {
    background: #e8edf2;
    border-radius: 3px;
    height: 10px;
    overflow: hidden;
}
.prob-fill {
    height: 10px;
    border-radius: 3px;
    transition: width 0.4s ease;
}

/* CIBIL score display */
.cibil-display {
    text-align: center;
    padding: 28px 20px;
    border-radius: 8px;
    border: 2px solid;
}
.cibil-display .score {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 3rem;
    font-weight: 600;
    line-height: 1;
}
.cibil-display .band {
    font-size: 1rem;
    font-weight: 600;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.cibil-display .sublabel {
    font-size: 0.72rem;
    color: #64748b;
    margin-top: 4px;
    letter-spacing: 0.5px;
}

/* Section headers */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #94a3b8;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 8px;
    margin-bottom: 16px;
}

/* Model info box */
.model-info {
    background: #0f1923;
    border: 1px solid #1e3a4a;
    border-radius: 6px;
    padding: 16px 20px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #7a9bb5;
    line-height: 1.8;
}
.model-info .highlight { color: #00d4aa; font-weight: 600; }

/* Batch results table */
.stDataFrame { font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #f8fafc;
    border-right: 1px solid #e2e8f0;
}
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stCheckbox label {
    font-size: 0.8rem !important;
    color: #475569 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 2px solid #e2e8f0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.5px;
    padding: 8px 20px;
}

button[kind="primary"] {
    background: #00d4aa !important;
    color: #0f1923 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
TIER_MAP    = {0: "P1", 1: "P2", 2: "P3", 3: "P4"}
TIER_COLORS = {
    "P1": "#16a34a", "P2": "#2563eb",
    "P3": "#d97706", "P4": "#dc2626"
}
TIER_BG = {
    "P1": "#f0fdf4", "P2": "#eff6ff",
    "P3": "#fffbeb", "P4": "#fef2f2"
}
TIER_LABELS = {
    "P1": "P1 — Premium Applicant",
    "P2": "P2 — Standard Approval",
    "P3": "P3 — Review Required",
    "P4": "P4 — High Risk / Decline"
}
TIER_ADVICE = {
    "P1": "Excellent credit profile. Eligible for best available rate and maximum loan amount.",
    "P2": "Strong profile. Standard approval recommended with normal terms.",
    "P3": "Moderate risk indicators present. Manual underwriter review recommended before approval.",
    "P4": "Significant default risk factors. Consider declining or requiring additional collateral / guarantor."
}
EDU_MAP = {
    "SSC": 0, "12TH": 1, "UNDER GRADUATE": 2,
    "GRADUATE": 2, "POST-GRADUATE": 3, "PROFESSIONAL": 4
}

# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    missing = [f for f in ["shap_model.pkl", "feature_cols.pkl", "feature_normalizer.pkl", "X_train_res.parquet"]
               if not os.path.exists(f)]
    if missing:
        return None, None, None, None, f"Missing files: {', '.join(missing)}"
    try:
        with open("shap_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("feature_cols.pkl", "rb") as f:
            features = pickle.load(f)
        with open("feature_normalizer.pkl", "rb") as f:
            normalizer = pickle.load(f)
        X_tr = pd.read_parquet("X_train_res.parquet")
        return model, features, normalizer, X_tr, None
    except Exception as e:
        return None, None, None, None, str(e)

model, feat_names, normalizer, X_train_res, load_error = load_artifacts()

# ── SHAP explainer — cached so it doesn't recompute on every slider move ──────
@st.cache_resource(show_spinner="Initialising SHAP explainer...")
def get_explainer(_model):
    return shap.TreeExplainer(_model)

# ── Feature builder ────────────────────────────────────────────────────────────
def build_features(cibil, income, age, emp_years, education, gender, married,
                   total_tl, active_tl, missed_pmnt, deliq_6m, deliq_12m,
                   dpd_60, enq_6m, enq_12m, hl, pl, cc, gl):
    row = {f: 0 for f in feat_names}

    # Raw fields
    row["Credit_Score"]        = cibil
    row["NETMONTHLYINCOME"]    = income
    row["AGE"]                 = age
    row["Time_With_Curr_Empr"] = emp_years * 12
    row["EDUCATION_ENC"]       = EDU_MAP.get(education, 1)
    row["GENDER_ENC"]          = int(gender == "M")
    row["MARRIED"]             = int(married == "Married")
    row["Total_TL"]            = total_tl
    row["Tot_Active_TL"]       = active_tl
    row["Tot_Closed_TL"]       = max(0, total_tl - active_tl)
    row["Tot_Missed_Pmnt"]     = missed_pmnt
    row["num_deliq_6mts"]      = deliq_6m
    row["num_deliq_12mts"]     = deliq_12m
    row["num_deliq_6_12mts"]   = max(0, deliq_12m - deliq_6m)
    row["num_times_60p_dpd"]   = dpd_60
    row["num_times_30p_dpd"]   = missed_pmnt
    row["enq_L6m"]             = enq_6m
    row["enq_L12m"]            = enq_12m
    row["enq_L3m"]             = max(0, enq_6m - 1)
    row["HL_Flag"]             = int(hl)
    row["PL_Flag"]             = int(pl)
    row["CC_Flag"]             = int(cc)
    row["GL_Flag"]             = int(gl)
    row["has_CC"]              = int(cc)
    row["has_PL"]              = int(pl)
    row["Age_Oldest_TL"]       = max(12, emp_years * 10)
    row["Age_Newest_TL"]       = 6

    # Engineered features
    row["CIBIL_Band_Num"]      = float(
        pd.cut([cibil], bins=[0, 600, 650, 700, 750, 900], labels=[0, 1, 2, 3, 4])[0]
    )
    row["score_above_700"]     = max(0, cibil - 700)
    row["score_below_650"]     = max(0, 650 - cibil)
    row["loan_diversity"]      = sum([hl, pl, cc, gl])
    row["active_tl_ratio"]     = active_tl / max(total_tl, 1)
    row["TL_age_spread"]       = row["Age_Oldest_TL"] - row["Age_Newest_TL"]
    row["unsecured_ratio"]     = (int(pl) + int(cc)) / max(total_tl, 1)
    row["deliq_intensity"]     = deliq_6m * 3 + deliq_12m * 2
    row["has_60dpd"]           = int(dpd_60 > 0)
    row["has_30dpd"]           = int(missed_pmnt > 0)
    row["enq_acceleration"]    = enq_6m / max(enq_12m, 1)
    row["employment_years"]    = emp_years
    row["stable_employment"]   = int(emp_years >= 2)
    # FOIR proxy — uses PL flag as binary obligation indicator
    row["foir_proxy"]          = (int(pl) * 0.3 * income * 0.4) / (income + 1) if income > 0 else 0
    row["pct_active_TLs_ever"] = active_tl / max(total_tl, 1)

    # Advanced features from notebook training pipeline
    row["delinq_severity_score"] = deliq_6m * 5 + deliq_12m * 2 + dpd_60 * 8
    row["deliq_trend"]           = deliq_6m / deliq_12m if deliq_12m > 0 else 0
    row["enquiry_velocity"]      = enq_6m - row["enq_L3m"]
    row["enquiry_concentration"] = (enq_6m ** 2) / (enq_12m + 1) if enq_12m > 0 else 0
    row["income_scaled"]         = np.log1p(income / 1000) if income >= 0 else 0
    row["income_foir_burden"]    = 0
    row["tl_quality_score"]      = row["active_tl_ratio"] * 3 + row["loan_diversity"] * 2 + np.log1p(row["Age_Oldest_TL"]) * 0.5
    row["payment_stress"]        = missed_pmnt * 2 + row["has_60dpd"] * 5 + row["has_30dpd"] * 2
    row["cibil_poly2"]           = (650 - cibil) ** 2 if cibil < 650 else 0
    row["distress_signal"]       = row["delinq_severity_score"] * (row["enq_acceleration"] + 0.1)

    return pd.DataFrame([row])[feat_names].fillna(0)


def normalize_features(df):
    normalized = pd.DataFrame(normalizer.transform(df), columns=feat_names, index=df.index)
    return normalized.fillna(0)


def get_cibil_band(score):
    if score < 600:   return "Poor",      "#dc2626"
    if score < 650:   return "Fair",      "#d97706"
    if score < 700:   return "Good",      "#ca8a04"
    if score < 750:   return "Very Good", "#16a34a"
    return                   "Excellent", "#2563eb"


# ── Preprocess uploaded batch CSV/Excel ───────────────────────────────────────
def preprocess_batch(df_raw):
    df = df_raw.copy()
    sentinel_cols = [
        'time_since_first_deliquency', 'time_since_recent_deliquency',
        'max_delinquency_level', 'max_recent_level_of_deliq', 'recent_level_of_deliq',
        'max_deliq_6mts', 'max_deliq_12mts',
        'tot_enq', 'CC_enq', 'CC_enq_L6m', 'CC_enq_L12m',
        'PL_enq', 'PL_enq_L6m', 'PL_enq_L12m',
        'time_since_recent_enq', 'enq_L12m', 'enq_L6m', 'enq_L3m'
    ]
    for col in sentinel_cols:
        if col in df.columns:
            df[col] = df[col].replace(-99999, 0)
    if 'time_since_recent_payment' in df.columns:
        df['time_since_recent_payment'] = df['time_since_recent_payment'].replace(-99999, 9999)

    EDU_MAP_BATCH = {"SSC":0,"12TH":1,"UNDER GRADUATE":2,"GRADUATE":2,
                      "POST-GRADUATE":3,"PROFESSIONAL":4,"OTHERS":1}
    if 'EDUCATION' in df.columns:
        df['EDUCATION_ENC'] = df['EDUCATION'].map(EDU_MAP_BATCH).fillna(1)
    if 'GENDER' in df.columns:
        df['GENDER_ENC'] = (df['GENDER'] == 'M').astype(int)
    if 'MARITALSTATUS' in df.columns:
        df['MARRIED'] = (df['MARITALSTATUS'] == 'Married').astype(int)
    for col in ['last_prod_enq2', 'first_prod_enq2']:
        prefix = 'last_prod' if 'last' in col else 'first_prod'
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], prefix=prefix)

    df['has_CC'] = (df.get('CC_utilization', pd.Series([-99999]*len(df))) != -99999).astype(int)
    df['has_PL'] = (df.get('PL_utilization', pd.Series([-99999]*len(df))) != -99999).astype(int)
    if 'CC_utilization' in df.columns:
        df['CC_utilization'] = df['CC_utilization'].replace(-99999, 0)
    if 'PL_utilization' in df.columns:
        df['PL_utilization'] = df['PL_utilization'].replace(-99999, 0)

    if 'Credit_Score' in df.columns:
        df['CIBIL_Band_Num']  = pd.cut(df['Credit_Score'], bins=[0,600,650,700,750,900],
                                         labels=[0,1,2,3,4]).astype(float)
        df['score_above_700'] = np.maximum(df['Credit_Score'] - 700, 0)
        df['score_below_650'] = np.maximum(650 - df['Credit_Score'], 0)

    for col_set in [['HL_Flag','PL_Flag','CC_Flag','GL_Flag']]:
        present = [c for c in col_set if c in df.columns]
        if present:
            df['loan_diversity'] = df[present].sum(axis=1)

    if 'Tot_Active_TL' in df.columns and 'Total_TL' in df.columns:
        df['active_tl_ratio'] = np.where(df['Total_TL']>0,
                                          df['Tot_Active_TL']/df['Total_TL'], 0)
    if 'Unsecured_TL' in df.columns and 'Total_TL' in df.columns:
        df['unsecured_ratio'] = np.where(df['Total_TL']>0,
                                          df['Unsecured_TL']/df['Total_TL'], 0)
    if 'Age_Oldest_TL' in df.columns and 'Age_Newest_TL' in df.columns:
        df['TL_age_spread'] = df['Age_Oldest_TL'] - df['Age_Newest_TL']

    for col in ['num_deliq_6mts','num_deliq_12mts','num_deliq_6_12mts']:
        if col not in df.columns:
            df[col] = 0
    df['deliq_intensity']  = df['num_deliq_6mts']*3 + df['num_deliq_12mts']*2 + df['num_deliq_6_12mts']
    df['has_60dpd']        = (df.get('num_times_60p_dpd', 0) > 0).astype(int)
    df['has_30dpd']        = (df.get('num_times_30p_dpd', 0) > 0).astype(int)

    if 'enq_L12m' in df.columns and 'enq_L6m' in df.columns:
        df['enq_acceleration'] = np.where(df['enq_L12m']>0,
                                            df['enq_L6m']/(df['enq_L12m']+1), 0)
    if 'Time_With_Curr_Empr' in df.columns:
        df['employment_years']  = df['Time_With_Curr_Empr'] / 12
        df['stable_employment'] = (df['employment_years'] >= 2).astype(int)

    # Advanced features used in notebook training pipeline
    df['delinq_severity_score'] = df.get('num_deliq_6mts', 0) * 5 + df.get('num_deliq_12mts', 0) * 2 + df.get('num_times_60p_dpd', 0) * 8
    df['deliq_trend'] = np.where(df.get('num_deliq_12mts', 0) > 0,
                                 df.get('num_deliq_6mts', 0) / df.get('num_deliq_12mts', 0), 0)
    df['enquiry_velocity'] = df.get('enq_L6m', 0) - df.get('enq_L3m', 0)
    df['enquiry_concentration'] = np.where(df.get('enq_L12m', 0) > 0,
                                           (df.get('enq_L6m', 0) ** 2) / (df.get('enq_L12m', 0) + 1), 0)
    df['income_scaled'] = np.log1p(df.get('NETMONTHLYINCOME', 0).clip(lower=0) / 1000)
    df['income_foir_burden'] = 0
    if 'active_tl_ratio' in df.columns and 'loan_diversity' in df.columns and 'Age_Oldest_TL' in df.columns:
        df['tl_quality_score'] = df['active_tl_ratio'] * 3 + df['loan_diversity'] * 2 + np.log1p(df['Age_Oldest_TL'].clip(lower=0)) * 0.5
    else:
        df['tl_quality_score'] = 0
    df['payment_stress'] = df.get('Tot_Missed_Pmnt', 0) * 2 + df['has_60dpd'] * 5 + df['has_30dpd'] * 2
    df['cibil_poly2'] = np.where(df.get('Credit_Score', 0) < 650,
                                 (650 - df.get('Credit_Score', 0)) ** 2, 0)
    df['distress_signal'] = df['delinq_severity_score'] * (df.get('enq_acceleration', 0) + 0.1)

    return df.reindex(columns=feat_names, fill_value=0)


# ════════════════════════════════════════════════════════════════════════════════
# APP LAYOUT
# ════════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="main-header">
    <h1>🏦 CREDIT RISK ASSESSOR</h1>
    <p>INDIAN BANK + CIBIL BUREAU · XGBoost + SHAP · RBI-ALIGNED RISK TIERS</p>
</div>
""", unsafe_allow_html=True)

# Error state
if load_error:
    st.error(f"⚠️ Could not load model artifacts: {load_error}")
    st.markdown("""
    **To fix this, run the following in your notebook:**
    ```python
    import pickle
    with open("shap_model.pkl", "wb") as f:
        pickle.dump(shap_model, f)
    with open("feature_cols.pkl", "wb") as f:
        pickle.dump(feat_names, f)
    with open("feature_normalizer.pkl", "wb") as f:
        pickle.dump(normalizer, f)
    X_train_res.to_parquet("X_train_res.parquet")
    ```
    Then restart the Streamlit app.
    """)
    st.stop()

explainer = get_explainer(model)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "  🔍  Single Applicant  ",
    "  📂  Batch Prediction  ",
    "  ℹ️  Model Info  "
])

# ════════════════════════════════════════
# TAB 1 — Single Applicant
# ════════════════════════════════════════
with tab1:

    # Sidebar inputs
    st.sidebar.markdown("### 🧾 Applicant Profile")
    st.sidebar.markdown('<p class="section-label">Credit Bureau</p>', unsafe_allow_html=True)
    cibil    = st.sidebar.slider("CIBIL Score", 300, 900, 680, step=1)
    enq_6m   = st.sidebar.slider("Enquiries — last 6 months", 0, 15, 1)
    enq_12m  = st.sidebar.slider("Enquiries — last 12 months", 0, 20, 2)
    deliq_6m = st.sidebar.slider("Delinquencies — last 6M", 0, 10, 0)
    deliq_12m= st.sidebar.slider("Delinquencies — last 12M", 0, 10, 0)
    dpd_60   = st.sidebar.slider("60+ DPD occurrences (ever)", 0, 10, 0)

    st.sidebar.markdown('<p class="section-label">Financial</p>', unsafe_allow_html=True)
    income      = st.sidebar.number_input("Net Monthly Income (₹)", 0, 1000000, 25000, step=500)
    missed_pmnt = st.sidebar.slider("Total Missed Payments", 0, 20, 0)

    st.sidebar.markdown('<p class="section-label">Trade Lines</p>', unsafe_allow_html=True)
    total_tl = st.sidebar.slider("Total Loans (ever)", 0, 20, 4)
    active_tl= st.sidebar.slider("Active Loans", 0, 10, 1)
    hl = st.sidebar.checkbox("Home Loan", value=False)
    pl = st.sidebar.checkbox("Personal Loan", value=True)
    cc = st.sidebar.checkbox("Credit Card", value=False)
    gl = st.sidebar.checkbox("Gold Loan", value=False)

    st.sidebar.markdown('<p class="section-label">Demographics</p>', unsafe_allow_html=True)
    age       = st.sidebar.slider("Age", 21, 65, 35)
    emp_years = st.sidebar.slider("Years with Current Employer", 0, 20, 3)
    education = st.sidebar.selectbox("Education",
                    ["SSC","12TH","UNDER GRADUATE","GRADUATE","POST-GRADUATE","PROFESSIONAL"])
    gender    = st.sidebar.selectbox("Gender", ["M", "F"])
    married   = st.sidebar.selectbox("Marital Status", ["Married", "Single"])

    # Predict
    X_input = build_features(cibil, income, age, emp_years, education, gender, married,
                               total_tl, active_tl, missed_pmnt, deliq_6m, deliq_12m,
                               dpd_60, enq_6m, enq_12m, hl, pl, cc, gl)
    X_input_norm = normalize_features(X_input)
    pred  = int(model.predict(X_input_norm)[0])
    proba = model.predict_proba(X_input_norm)[0]
    tier  = TIER_MAP[pred]
    color = TIER_COLORS[tier]
    bg    = TIER_BG[tier]

    # ── Decision banner ──
    st.markdown(f"""
    <div class="decision-banner" style="background:{bg}; border-color:{color}; color:{color}">
        <h2>{TIER_LABELS[tier]}</h2>
        <p style="color:#374151">{TIER_ADVICE[tier]}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics row ──
    c1, c2, c3, c4 = st.columns(4)
    band_name, band_color = get_cibil_band(cibil)
    for col, val, label, vcolor in [
        (c1, f"{cibil}", "CIBIL Score", band_color),
        (c2, band_name, "Risk Band", band_color),
        (c3, f"₹{income:,}", "Monthly Income", "#374151"),
        (c4, f"{proba[pred]*100:.1f}%", "Model Confidence", color),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="value" style="color:{vcolor}">{val}</div>
            <div class="label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Probabilities + CIBIL band ──
    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        st.markdown('<p class="section-label">Prediction Probabilities</p>', unsafe_allow_html=True)
        tier_colors_list = [TIER_COLORS[t] for t in ["P1","P2","P3","P4"]]
        for i, (t, p, c) in enumerate(zip(["P1","P2","P3","P4"], proba, tier_colors_list)):
            bold = "font-weight:700" if i == pred else ""
            st.markdown(f"""
            <div class="prob-row">
                <div class="prob-label">
                    <span style="{bold}">{t} — {["Premium","Standard","Review","Decline"][i]}</span>
                    <span style="{bold}; color:{c}">{p*100:.1f}%</span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill" style="width:{p*100:.1f}%; background:{c}"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<p class="section-label">CIBIL Score Band (RBI Aligned)</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="cibil-display" style="border-color:{band_color}; background:{band_color}11">
            <div class="score" style="color:{band_color}">{cibil}</div>
            <div class="band" style="color:{band_color}">{band_name}</div>
            <div class="sublabel">
                {'Below lending threshold' if cibil < 650 else
                 'Standard approval zone' if cibil < 700 else
                 'Preferred borrower range' if cibil < 750 else
                 'Premium tier eligible'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SHAP waterfall ──
    st.markdown('<p class="section-label">SHAP Explanation — Why this decision?</p>',
                unsafe_allow_html=True)
    with st.spinner("Computing explanation..."):
        try:
            shap_exp   = explainer(X_input_norm)
            shap_class = shap_exp[:, :, pred]
            fig, _ = plt.subplots(figsize=(11, 5))
            shap.waterfall_plot(shap_class[0], max_display=12, show=False)
            plt.title(f"Feature contributions toward {TIER_LABELS[tier]}",
                       fontweight="bold", fontsize=11, pad=12)
            plt.tight_layout()
            st.pyplot(fig, width='stretch')
            plt.close()
            st.caption("🔴 Red bars push toward this tier  ·  🔵 Blue bars push away  ·  "
                        "E[f(x)] = baseline prediction  ·  f(x) = this applicant's score")
        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")


# ════════════════════════════════════════
# TAB 2 — Batch Prediction
# ════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-label">Batch Prediction — Upload & Score</p>',
                unsafe_allow_html=True)

    st.markdown("""
    Upload the **data/Unseen_Dataset.xlsx** (or any preprocessed file with the same columns).
    The model will score every applicant and return a downloadable results file.
    """)

    uploaded = st.file_uploader(
        "Upload applicant file",
        type=["xlsx", "xls", "csv"],
        help="Accepts Excel or CSV files. Must contain CIBIL bureau + bank trade line columns."
    )

    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                df_raw = pd.read_csv(uploaded)
            else:
                df_raw = pd.read_excel(uploaded)

            st.success(f"✅ Loaded {len(df_raw):,} applicants · {df_raw.shape[1]} columns")

            with st.spinner(f"Scoring {len(df_raw):,} applicants..."):
                X_batch  = preprocess_batch(df_raw)
                X_batch_norm = normalize_features(X_batch)
                preds_b  = model.predict(X_batch_norm)
                probas_b = model.predict_proba(X_batch_norm)

            results = df_raw.copy()
            results.insert(0, "Predicted_Tier", [TIER_MAP[p] for p in preds_b])
            results.insert(1, "P1_Probability", probas_b[:, 0].round(3))
            results.insert(2, "P2_Probability", probas_b[:, 1].round(3))
            results.insert(3, "P3_Probability", probas_b[:, 2].round(3))
            results.insert(4, "P4_Probability", probas_b[:, 3].round(3))
            results.insert(5, "Confidence",
                            [f"{probas_b[i, p]*100:.1f}%" for i, p in enumerate(preds_b)])

            # Summary stats
            tier_counts = pd.Series([TIER_MAP[p] for p in preds_b]).value_counts()
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p class="section-label">Portfolio Summary</p>', unsafe_allow_html=True)
            cols = st.columns(4)
            for col, t in zip(cols, ["P1","P2","P3","P4"]):
                count = tier_counts.get(t, 0)
                pct   = count / len(preds_b) * 100
                c     = TIER_COLORS[t]
                col.markdown(f"""
                <div class="metric-card" style="border-top: 3px solid {c}">
                    <div class="value" style="color:{c}">{count}</div>
                    <div class="label">{t} · {pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Results preview
            st.markdown('<p class="section-label">Results Preview (first 20 rows)</p>',
                        unsafe_allow_html=True)
            display_cols = ["Predicted_Tier","P1_Probability","P2_Probability",
                             "P3_Probability","P4_Probability","Confidence"]
            if "PROSPECTID" in results.columns:
                display_cols = ["PROSPECTID"] + display_cols
            st.dataframe(results[display_cols].head(20), width='stretch')

            # Download
            st.markdown("<br>", unsafe_allow_html=True)
            buf = io.BytesIO()
            results.to_excel(buf, index=False, engine="openpyxl")
            buf.seek(0)
            st.download_button(
                label="⬇️  Download Full Results (.xlsx)",
                data=buf,
                file_name="credit_risk_predictions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )

        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.markdown("Make sure the file contains the expected CIBIL + bank columns.")

    else:
        st.info("👆 Upload a file to begin batch scoring. "
                "The data/Unseen_Dataset.xlsx from your project folder works directly.")


# ════════════════════════════════════════
# TAB 3 — Model Info
# ════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-label">Model & Project Information</p>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        model_type = type(model).__name__
        st.markdown(f"""
        <div class="model-info">
            <span class="highlight">MODEL</span><br>
            Type &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: {model_type}<br>
            Classes &nbsp;&nbsp;: P1 · P2 · P3 · P4<br>
            Features &nbsp;: {len(feat_names)}<br>
            Preprocess : notebook-matched feature engineering + MinMaxScaler<br>
            Tuning &nbsp;&nbsp;&nbsp;: Optuna (40 trials)<br>
            Imbalance : SMOTE (4-class balanced)<br>
            <br>
            <span class="highlight">DATASET</span><br>
            Source &nbsp;&nbsp;&nbsp;: Kaggle — saurabhbadole<br>
            Records &nbsp;&nbsp;: 51,336 applicants<br>
            Train &nbsp;&nbsp;&nbsp;&nbsp;: 80% + SMOTE<br>
            Test &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: 20% holdout (stratified)<br>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div class="model-info">
            <span class="highlight">XAI METHODS</span><br>
            Global &nbsp;&nbsp;&nbsp;: SHAP TreeExplainer<br>
            Local &nbsp;&nbsp;&nbsp;&nbsp;: SHAP Waterfall + LIME<br>
            Fallback &nbsp;: Permutation Importance<br>
            <br>
            <span class="highlight">INDIA-SPECIFIC FEATURES</span><br>
            · CIBIL Band (RBI-aligned: 5 tiers)<br>
            · FOIR Proxy (obligation-to-income)<br>
            · Delinquency Intensity (recency-weighted)<br>
            · Enquiry Acceleration (stress signal)<br>
            · Loan Product Mix (HL/PL/CC/GL flags)<br>
            · Employment Stability (2yr threshold)<br>
            · Score Threshold Distance (650/700)<br>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">Risk Tier Reference</p>', unsafe_allow_html=True)

    for t in ["P1","P2","P3","P4"]:
        c = TIER_COLORS[t]
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:14px; padding:12px 16px;
                    background:{TIER_BG[t]}; border-radius:6px; margin-bottom:8px;
                    border-left: 4px solid {c}">
            <span style="font-family:'IBM Plex Mono',monospace; font-weight:700;
                          color:{c}; min-width:28px">{t}</span>
            <span style="color:#374151; font-size:0.9rem">{TIER_ADVICE[t]}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.75rem; color:#94a3b8; font-family:'IBM Plex Mono',monospace;
                border-top:1px solid #e2e8f0; padding-top:12px">
    References: Yang et al. (2025) · Oyeyemi et al. (2025) · Wang & Liang (2024) · Lin & Wang (2025)<br>
    Built for portfolio demonstration purposes. Not intended for production lending decisions.
    </div>
    """, unsafe_allow_html=True)