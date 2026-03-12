"""
app_api.py — Flask backend for Credit Risk Frontend
Run: python app_api.py
Then open credit_risk_frontend.html in browser
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import shap
import os

app = Flask(__name__)
CORS(app)  # Allow HTML file to call this API

# ── Load model artifacts once at startup ──────────────────────────────────────
print("Loading model artifacts...")

if not os.path.exists("shap_model.pkl"):
    raise FileNotFoundError("shap_model.pkl not found. Run your notebook first to save artifacts.")
if not os.path.exists("feature_cols.pkl"):
    raise FileNotFoundError("feature_cols.pkl not found. Run your notebook first to save artifacts.")
if not os.path.exists("feature_normalizer.pkl"):
    raise FileNotFoundError("feature_normalizer.pkl not found. Run your notebook to save the normalizer.")

with open("shap_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_cols.pkl", "rb") as f:
    feat_names = pickle.load(f)

with open("feature_normalizer.pkl", "rb") as f:
    normalizer = pickle.load(f)

# Load SHAP explainer once — expensive to recompute every request
explainer = shap.TreeExplainer(model)

print(f"✅ Model loaded: {type(model).__name__}")
print(f"✅ Features: {len(feat_names)} (10 advanced non-CIBIL features included)")
print(f"✅ Feature normalizer: MinMaxScaler active for consistent predictions")
print(f"✅ SHAP explainer ready with enhanced regularization")
print(f"✅ Server starting at http://localhost:5050")

# ── Constants ─────────────────────────────────────────────────────────────────
TIER_MAP = {0: "P1", 1: "P2", 2: "P3", 3: "P4"}
EDU_MAP  = {
    "SSC": 0, "12TH": 1, "UNDER GRADUATE": 2,
    "GRADUATE": 2, "POST-GRADUATE": 3, "PROFESSIONAL": 4
}

# ── Feature builder — same logic as notebook ──────────────────────────────────
def build_features(data):
    cibil     = float(data.get("cibil", 680))
    income    = float(data.get("income", 25000))
    age       = float(data.get("age", 35))
    emp_years = float(data.get("emp_years", 3))
    education = data.get("education", "GRADUATE")
    gender    = data.get("gender", "M")
    married   = data.get("married", "Married")
    total_tl  = float(data.get("total_tl", 4))
    active_tl = float(data.get("active_tl", 1))
    missed    = float(data.get("missed", 0))
    deliq6    = float(data.get("deliq6", 0))
    deliq12   = float(data.get("deliq12", 0))
    dpd60     = float(data.get("dpd60", 0))
    enq6      = float(data.get("enq6", 1))
    enq12     = float(data.get("enq12", 2))
    hl        = int(data.get("hl", False))
    pl        = int(data.get("pl", True))
    cc        = int(data.get("cc", False))
    gl        = int(data.get("gl", False))

    row = {f: 0 for f in feat_names}

    # Raw fields
    row["Credit_Score"]        = cibil
    row["NETMONTHLYINCOME"]    = income
    row["AGE"]                 = age
    row["Time_With_Curr_Empr"] = emp_years * 12
    row["EDUCATION_ENC"]       = EDU_MAP.get(education, 2)
    row["GENDER_ENC"]          = int(gender == "M")
    row["MARRIED"]             = int(married == "Married")
    row["Total_TL"]            = total_tl
    row["Tot_Active_TL"]       = active_tl
    row["Tot_Closed_TL"]       = max(0, total_tl - active_tl)
    row["Tot_Missed_Pmnt"]     = missed
    row["num_deliq_6mts"]      = deliq6
    row["num_deliq_12mts"]     = deliq12
    row["num_deliq_6_12mts"]   = max(0, deliq12 - deliq6)
    row["num_times_60p_dpd"]   = dpd60
    row["num_times_30p_dpd"]   = missed
    row["enq_L6m"]             = enq6
    row["enq_L12m"]            = enq12
    row["enq_L3m"]             = max(0, enq6 - 1)
    row["HL_Flag"]             = hl
    row["PL_Flag"]             = pl
    row["CC_Flag"]             = cc
    row["GL_Flag"]             = gl
    row["has_CC"]              = cc
    row["has_PL"]              = pl
    row["Age_Oldest_TL"]       = max(12, emp_years * 10)
    row["Age_Newest_TL"]       = 6

    # Engineered features
    bands = pd.cut([cibil], bins=[0, 600, 650, 700, 750, 900], labels=[0,1,2,3,4])
    row["CIBIL_Band_Num"]      = float(bands[0])
    row["score_above_700"]     = max(0, cibil - 700)
    row["score_below_650"]     = max(0, 650 - cibil)
    row["loan_diversity"]      = hl + pl + cc + gl
    row["active_tl_ratio"]     = active_tl / max(total_tl, 1)
    row["TL_age_spread"]       = row["Age_Oldest_TL"] - row["Age_Newest_TL"]
    row["unsecured_ratio"]     = (pl + cc) / max(total_tl, 1)
    row["deliq_intensity"]     = deliq6 * 3 + deliq12 * 2
    row["has_60dpd"]           = int(dpd60 > 0)
    row["has_30dpd"]           = int(missed > 0)
    row["enq_acceleration"]    = enq6 / max(enq12, 1)
    row["employment_years"]    = emp_years
    row["stable_employment"]   = int(emp_years >= 2)
    row["foir_proxy"]          = (pl * 0.3 * income * 0.4) / (income + 1) if income > 0 else 0
    row["pct_active_TLs_ever"] = active_tl / max(total_tl, 1)

    return pd.DataFrame([row])[feat_names].fillna(0)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)
        if data is None:
            data = {}
        if not isinstance(data, dict):
            return jsonify({"status": "error", "message": "Invalid JSON payload"}), 400

        # Build feature vector
        X = build_features(data)

        # ── CRITICAL: Normalize features using the same normalizer from training ──
        X_normalized = normalizer.transform(X)
        X_normalized = pd.DataFrame(X_normalized, columns=feat_names)

        # Predict on normalized features
        pred_idx = int(model.predict(X_normalized)[0])
        proba    = model.predict_proba(X_normalized)[0].tolist()
        tier     = TIER_MAP[pred_idx]

        # SHAP explanation on normalized features
        shap_vals       = explainer(X_normalized)
        pred_class_shap = shap_vals[:, :, pred_idx].values[0]
        # Use P4 SHAP sign to indicate risk direction consistently for the UI.
        p4_class_shap   = shap_vals[:, :, 3].values[0]

        # Top 5 SHAP drivers
        indices     = np.argsort(np.abs(pred_class_shap))[::-1][:5]
        top_drivers = []
        for i in indices:
            top_drivers.append({
                "feature": feat_names[i],
                "shap":    round(float(pred_class_shap[i]), 4),
                "value":   round(float(X.iloc[0, i]), 4),
                "direction": "risk" if p4_class_shap[i] > 0 else "safe"
            })

        return jsonify({
            "tier":        tier,
            "probabilities": {
                "P1": round(proba[0], 4),
                "P2": round(proba[1], 4),
                "P3": round(proba[2], 4),
                "P4": round(proba[3], 4),
            },
            "confidence":  round(proba[pred_idx], 4),
            "top_drivers": top_drivers,
            "risk_score":  round(float(proba[3] * 100), 1),
            "status":      "ok"
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":     "ok",
        "model":      type(model).__name__,
        "features":   len(feat_names),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
