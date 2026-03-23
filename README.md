#  Insurance Policy Lapse Prediction
https://insurance-lapse-prediction-dhojv2tqiyqqdxyvktdnkx.streamlit.app/
A Machine Learning project to predict which policyholders are likely to **lapse (cancel or stop paying premiums)**, enabling insurance companies to take proactive retention actions.

---

##  Business Problem

Policy lapse is one of the biggest challenges in the insurance industry. When a customer lapses, the company loses:
- Future premium revenue
- Customer lifetime value
- Acquisition cost already spent

This project builds a predictive model to **identify high-risk customers before they lapse**, so the retention team can intervene with targeted offers or follow-ups.

---

##  Project Structure

```
insurance-lapse-prediction/
│
├── insurance_lapse_prediction.py   # Main ML script
├── insurance_lapse_data.csv        # Dataset (5000 records)
├── eda_plots.png                   # Exploratory Data Analysis plots
├── model_comparison.png            # ROC & Precision-Recall curves
├── feature_importance.png          # Top features from XGBoost
├── high_risk_customers.csv         # Exported high-risk customer list
├── xgb_insurance_lapse.pkl         # Saved XGBoost model
├── rf_insurance_lapse.pkl          # Saved Random Forest model
├── scaler.pkl                      # Saved StandardScaler
└── README.md                       # Project documentation
```

---

##  Dataset

| Feature | Description |
|---|---|
| `age` | Age of the policyholder |
| `gender` | Male / Female |
| `marital_status` | Single / Married / Divorced |
| `region` | Urban / Semi-Urban / Rural |
| `income_bracket` | Low / Medium / High |
| `policy_type` | Term / Endowment / ULIP / Whole Life |
| `policy_tenure` | Number of years policy has been active |
| `premium_amount` | Annual premium in INR |
| `sum_assured` | Total coverage amount |
| `payment_mode` | Monthly / Quarterly / Yearly |
| `missed_payments` | Number of missed premium payments |
| `num_claims` | Number of claims filed |
| `agent_contact` | Whether agent contacted the customer (0/1) |
| `loan_on_policy` | Whether a loan is taken on the policy (0/1) |
| `lapsed` | **Target variable** — 0 = Active, 1 = Lapsed |

- **Total Records:** 5,000
- **Lapse Rate:** ~19%

---

##  Tech Stack

- **Language:** Python 3.11
- **Libraries:** pandas, numpy, scikit-learn, XGBoost, imbalanced-learn, matplotlib, seaborn, joblib

---

##  Models Used

| Model | Description |
|---|---|
| Logistic Regression | Baseline linear model |
| Random Forest | Ensemble of 200 decision trees |
| XGBoost | Gradient boosted trees (best performer) |

**Class Imbalance** handled using **SMOTE** (Synthetic Minority Oversampling Technique)

---

##  Results

| Model | ROC-AUC | Avg Precision |
|---|---|---|
| Logistic Regression | ~0.78 | ~0.52 |
| Random Forest | ~0.88 | ~0.68 |
| **XGBoost** | **~0.91** | **~0.74** |

> XGBoost performed best and was selected as the final model.

---

##  Top Features (XGBoost)

1. `missed_payments` — strongest predictor of lapse
2. `premium_to_income_ratio` — affordability signal
3. `policy_tenure` — longer tenure = lower lapse risk
4. `income_bracket` — low income = higher lapse risk
5. `agent_contact` — customers contacted by agents lapse less

---

##  How to Run

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/insurance-lapse-prediction.git
cd insurance-lapse-prediction
```

**2. Install dependencies**
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn imbalanced-learn
```

**3. Run the project**
```bash
python insurance_lapse_prediction.py
```

---

##  Output

After running, the following files are generated:

-  `eda_plots.png` — Data analysis visualizations
-  `model_comparison.png` — ROC & PR curves for all models
-  `feature_importance.png` — Top 15 important features
-  `high_risk_customers.csv` — List of customers with lapse probability ≥ 0.6
-  `xgb_insurance_lapse.pkl` — Saved XGBoost model for deployment

---

##  Business Use

The `high_risk_customers.csv` file can be directly used by the **retention team** to:
- Send personalized offers
- Schedule agent follow-up calls
- Offer premium payment flexibility
- Provide policy loan options

---

##  Author

**Garima Choudhary**  
[GitHub](https://github.com/YOUR_USERNAME) • [LinkedIn](https://linkedin.com/in/YOUR_USERNAME)
