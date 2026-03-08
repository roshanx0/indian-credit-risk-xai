# Credit Risk XAI System 🏦

An end-to-end **Explainable AI (XAI) credit risk classification system** using real-world Indian bank + CIBIL bureau data. Predicts loan approval risk tiers (P1-P4) with full transparency using SHAP and LIME explanations.

> ⚠️ **Educational Project**: This demonstrates XAI techniques on a Kaggle dataset that achieves 99%+ AUC (synthetic/educational). Real-world credit risk models typically achieve 75-90% AUC. Built for portfolio/learning purposes.

---

## 🎯 Features

- **4-Tier Risk Classification**: P1 (Premium) → P4 (Decline)
- **XAI Methods**: SHAP TreeExplainer, LIME, Permutation Importance
- **RBI-Aligned**: CIBIL band tiers, regulatory transparency framework
- **Interactive Web App**: Streamlit UI with real-time explanations
- **India-Specific Engineering**: FOIR proxy, delinquency intensity, enquiry acceleration

---

## 📊 Tech Stack

**ML/XAI**: scikit-learn, XGBoost, LightGBM, SHAP, LIME, Optuna  
**Data**: pandas, numpy, imbalanced-learn (SMOTE)  
**Deployment**: Streamlit  
**Visualization**: matplotlib, seaborn

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone <your-repo-url>
cd credit-risk-xai-new
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
# source venv/bin/activate    # Linux/Mac
pip install -r requirements.txt
```

### 2. Download Dataset

**⚠️ IMPORTANT**: The dataset files are NOT included in this repo (gitignored due to size).

**Option A: From Kaggle** (Recommended)

1. Visit: [Kaggle - Leading Indian Bank & CIBIL Dataset](https://www.kaggle.com/datasets/saurabhbadole/leading-indian-bank-cibil-real-world-dataset)
2. Download the dataset (requires free Kaggle account)
3. Extract the 3 Excel files:
   - `Internal_Bank_Dataset.xlsx`
   - `External_Cibil_Dataset.xlsx`
   - `Unseen_Dataset.xlsx`
4. Create a `data/` folder in the project root
5. Place all 3 files inside `data/`

**Option B: Contact Repository Owner**

- If the Kaggle link is unavailable, contact me for dataset access

**Verify Setup:**

```bash
# Check data folder exists with all 3 files
ls data/
# Should show: Internal_Bank_Dataset.xlsx, External_Cibil_Dataset.xlsx, Unseen_Dataset.xlsx
```

### 3. Run the Notebook

Open `indian_credit_risk.ipynb` in Jupyter/VS Code and execute all cells to:

- Process Indian bank + CIBIL data
- Engineer 15+ credit risk features
- Train 5 models (Logistic Regression → LightGBM)
- Generate SHAP/LIME explanations
- Export model artifacts

### 4. Launch the App

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` to interact with the credit risk classifier.

---

## 📁 Project Structure

```
credit-risk-xai-new/
│
├── data/                        # Dataset folder (gitignored)
│   ├── Internal_Bank_Dataset.xlsx
│   ├── External_Cibil_Dataset.xlsx
│   └── Unseen_Dataset.xlsx
│
├── indian_credit_risk.ipynb    # Main analysis notebook (16 sections)
├── app.py                       # Streamlit web application
│
├── shap_model.pkl               # Trained model (generated, gitignored)
├── feature_cols.pkl             # Feature names (generated, gitignored)
├── X_train_res.parquet          # SHAP background (generated, gitignored)
│
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git exclusions
└── README.md                    # This file
```

---

## 📋 Requirements

Create `requirements.txt` with:

```
streamlit>=1.31.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.1.0
shap>=0.43.0
lime>=0.2.0
imbalanced-learn>=0.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
optuna>=3.4.0
openpyxl>=3.1.0
pyarrow>=14.0.0
```

---

## 🔍 Model Performance

| Model               | AUC-ROC | F1 Weighted |
| ------------------- | ------- | ----------- |
| Logistic Regression | 0.9996  | 0.9951      |
| Random Forest       | 0.9999  | 0.9973      |
| Gradient Boosting   | 0.9999  | 0.9975      |
| XGBoost             | 0.9999  | 0.9976      |
| LightGBM            | 0.9999  | 0.9976      |

**⚠️ Note**: 99% metrics indicate educational dataset (near-perfect predictors). Production credit risk: 75-85% AUC typical, 85-90% excellent.

---

## 🧠 Key Features Engineered

1. **CIBIL_Band_Num** – RBI-aligned risk tiers (5 bands)
2. **deliq_intensity** – Recency-weighted delinquency (6M × 3, 12M × 1)
3. **enq_acceleration** – Credit stress signal (L3M - L12M enquiries)
4. **FOIR_proxy** – Obligation-to-income ratio estimate
5. **score_threshold_dist** – Distance from 650/700 CIBIL cutoffs
6. **loan_product_mix** – HL/PL/CC/GL flags (stability signals)
7. **employment_stability** – 2+ years tenure binary

---

## 📚 Dataset

**Source**: [Kaggle - Leading Indian Bank & CIBIL Real-World Dataset](https://www.kaggle.com/datasets/saurabhbadole/leading-indian-bank-cibil-real-world-dataset)  
**Records**: 51,336 loan applicants  
**Features**: 87 (post-merge: bank + bureau)  
**Target**: `Approved_Flag` (P1/P2/P3/P4 risk tiers)

**⚠️ Dataset Quality**: Synthetic/educational with 99.98% baseline AUC (without SMOTE). Valuable for learning XAI techniques, not representative of production complexity.

**📥 Download Instructions**: See [Quick Start → Step 2](#2-download-dataset) above for how to obtain the dataset files.

---

## 🎓 Learning Outcomes

This project demonstrates:

- ✅ Feature engineering for credit risk (India-specific)
- ✅ Handling imbalanced classification (SMOTE)
- ✅ Hyperparameter tuning (Optuna)
- ✅ Model interpretability (SHAP global + local, LIME)
- ✅ Production deployment (Streamlit)
- ✅ RBI regulatory alignment (transparency framework)
- ✅ Critical thinking (recognizing synthetic data limitations)

---

## 📖 References

- Yang et al. (2025) – Ensemble methods for credit scoring
- Oyeyemi et al. (2025) – XAI in financial risk
- Wang & Liang (2024) – SHAP applications in lending
- Lin & Wang (2025) – Regulatory compliance in ML

---

## ⚖️ License & Disclaimer

**For educational/portfolio purposes only.**  
Not intended for production lending decisions.  
Built with Kaggle educational dataset (synthetic data).

**Dataset**: Available on [Kaggle](https://www.kaggle.com/datasets/saurabhbadole/leading-indian-bank-cibil-real-world-dataset) (free account required).

---

## 👤 Author

[Your Name]  
[GitHub Profile] | [LinkedIn] | [Portfolio]

---

## ❓ FAQ

**Q: Where do I get the dataset files?**  
A: Download from [Kaggle](https://www.kaggle.com/datasets/saurabhbadole/leading-indian-bank-cibil-real-world-dataset). See [Step 2 in Quick Start](#2-download-dataset) for detailed instructions.

**Q: Can I run the Streamlit app without running the notebook?**  
A: No. You must run the notebook first to generate `shap_model.pkl`, `feature_cols.pkl`, and `X_train_res.parquet`.

**Q: Why 99% AUC when real credit models get 75-85%?**  
A: This is a synthetic/educational Kaggle dataset with near-perfect predictors. It's designed for learning XAI techniques, not simulating production complexity.

---

**⭐ If this helped you learn XAI, consider starring the repo!**
