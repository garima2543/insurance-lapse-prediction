# =============================================================================
# INSURANCE LAPSE PREDICTION - ML PROJECT
# Business Use Case: Predict which policyholders are likely to lapse (cancel /
# stop paying premiums) so the retention team can intervene proactively.
# Models: Logistic Regression | Random Forest | XGBoost
# =============================================================================

# ── 1. INSTALL DEPENDENCIES ──────────────────────────────────────────────────
# Run this once in your terminal:
#   pip install pandas numpy scikit-learn xgboost matplotlib seaborn imbalanced-learn

# ── 2. IMPORTS ───────────────────────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ── 3. LOAD DATA ─────────────────────────────────────────────────────────────
df = pd.read_csv(r"C:\Users\Dell\OneDrive\Desktop\insurance_project\insurance_lapse_data.csv")
df.drop(columns=["policy_id"], inplace=True, errors="ignore")  # drop ID column if present

print("Dataset shape:", df.shape)
print("\nLapse Rate: {:.2%}".format(df["lapsed"].mean()))
print("\nFirst 5 rows:")
print(df.head())

# ── 4. EXPLORATORY DATA ANALYSIS (EDA) ──────────────────────────────────────
print("\n========== EDA ==========")
print(df.describe())
print("\nMissing values:\n", df.isnull().sum())

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Insurance Lapse - EDA", fontsize=16, fontweight="bold")

# Lapse distribution
df["lapsed"].value_counts().plot.pie(
    ax=axes[0, 0], autopct="%1.1f%%",
    labels=["Active", "Lapsed"], colors=["#2ecc71", "#e74c3c"]
)
axes[0, 0].set_title("Lapse Distribution")

# Age vs Lapse
df.groupby("lapsed")["age"].plot.kde(ax=axes[0, 1], legend=True)
axes[0, 1].set_title("Age Distribution by Lapse")
axes[0, 1].legend(["Active", "Lapsed"])

# Missed payments vs Lapse
sns.barplot(data=df, x="missed_payments", y="lapsed",
            ax=axes[0, 2], palette="Reds")
axes[0, 2].set_title("Missed Payments vs Lapse Rate")

# Policy type vs Lapse
lapse_by_type = df.groupby("policy_type")["lapsed"].mean().sort_values(ascending=False)
lapse_by_type.plot.bar(ax=axes[1, 0], color="#e67e22", edgecolor="black")
axes[1, 0].set_title("Lapse Rate by Policy Type")
axes[1, 0].tick_params(axis="x", rotation=15)

# Income bracket vs Lapse
lapse_by_income = df.groupby("income_bracket")["lapsed"].mean()
lapse_by_income.plot.bar(ax=axes[1, 1], color="#3498db", edgecolor="black")
axes[1, 1].set_title("Lapse Rate by Income Bracket")

# Premium amount vs Lapse
df.boxplot(column="premium_amount", by="lapsed", ax=axes[1, 2])
axes[1, 2].set_title("Premium Amount vs Lapse")
axes[1, 2].set_xticklabels(["Active", "Lapsed"])

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print("EDA plots saved as eda_plots.png")

# ── 5. FEATURE ENGINEERING ───────────────────────────────────────────────────
df["premium_to_income_ratio"] = df["premium_amount"] / (
    df["income_bracket"].map({"Low": 200000, "Medium": 600000, "High": 1500000})
)
df["claim_rate"] = df["num_claims"] / (df["policy_tenure"] + 1)
df["high_risk_payment"] = ((df["missed_payments"] >= 2) & (df["payment_mode"] == "Monthly")).astype(int)

# ── 6. ENCODING ──────────────────────────────────────────────────────────────
cat_cols = ["gender", "payment_mode", "income_bracket", "policy_type",
            "marital_status", "region"]
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# ── 7. TRAIN / TEST SPLIT ────────────────────────────────────────────────────
FEATURES = [c for c in df.columns if c != "lapsed"]
X = df[FEATURES]
y = df["lapsed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Handle class imbalance with SMOTE
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
print(f"\nAfter SMOTE — Train size: {X_train_sm.shape[0]} | Class balance: {pd.Series(y_train_sm).value_counts().to_dict()}")

# Scale for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sm)
X_test_scaled  = scaler.transform(X_test)

# ── 8. MODEL TRAINING ────────────────────────────────────────────────────────
print("\n========== TRAINING MODELS ==========")

# --- Logistic Regression ---
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train_sm)
lr_pred  = lr.predict(X_test_scaled)
lr_proba = lr.predict_proba(X_test_scaled)[:, 1]

# --- Random Forest ---
rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                             min_samples_leaf=5, random_state=42, n_jobs=-1)
rf.fit(X_train_sm, y_train_sm)
rf_pred  = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

# --- XGBoost ---
xgb = XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric="logloss",
    random_state=42, n_jobs=-1
)
xgb.fit(X_train_sm, y_train_sm,
        eval_set=[(X_test, y_test)], verbose=False)
xgb_pred  = xgb.predict(X_test)
xgb_proba = xgb.predict_proba(X_test)[:, 1]

# ── 9. EVALUATION ────────────────────────────────────────────────────────────
def evaluate(name, y_true, y_pred, y_proba):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, target_names=["Active", "Lapsed"]))
    print(f"  ROC-AUC  : {roc_auc_score(y_true, y_proba):.4f}")
    print(f"  Avg Prec : {average_precision_score(y_true, y_proba):.4f}")

evaluate("Logistic Regression", y_test, lr_pred,  lr_proba)
evaluate("Random Forest",       y_test, rf_pred,  rf_proba)
evaluate("XGBoost",             y_test, xgb_pred, xgb_proba)

# ── 10. COMPARISON PLOTS ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Model Comparison", fontsize=15, fontweight="bold")

models = [
    ("Logistic Regression", lr_proba,  lr_pred,  "royalblue"),
    ("Random Forest",       rf_proba,  rf_pred,  "darkorange"),
    ("XGBoost",             xgb_proba, xgb_pred, "crimson"),
]

# ROC curves
for name, proba, _, color in models:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color)
axes[0].plot([0,1],[0,1],"k--")
axes[0].set_title("ROC Curves"); axes[0].legend(); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")

# Precision-Recall curves
for name, proba, _, color in models:
    prec, rec, _ = precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)
    axes[1].plot(rec, prec, label=f"{name} (AP={ap:.3f})", color=color)
axes[1].set_title("Precision-Recall Curves"); axes[1].legend()
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")

# Confusion matrix — XGBoost (best model)
cm = confusion_matrix(y_test, xgb_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=axes[2],
            xticklabels=["Active","Lapsed"], yticklabels=["Active","Lapsed"])
axes[2].set_title("XGBoost Confusion Matrix")
axes[2].set_xlabel("Predicted"); axes[2].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Model comparison plots saved as model_comparison.png")

# ── 11. FEATURE IMPORTANCE (XGBoost) ─────────────────────────────────────────
fi = pd.Series(xgb.feature_importances_, index=FEATURES).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
fi.head(15).plot.barh(color="crimson", edgecolor="black")
plt.gca().invert_yaxis()
plt.title("Top 15 Feature Importances (XGBoost)", fontsize=13)
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("Feature importance plot saved as feature_importance.png")
print("\nTop 10 features:\n", fi.head(10))

# ── 12. BUSINESS SCORING — Risk Segmentation ─────────────────────────────────
X_test_copy = X_test.copy()
X_test_copy["lapse_probability"] = xgb_proba
X_test_copy["actual_lapsed"]     = y_test.values
X_test_copy["risk_segment"] = pd.cut(
    xgb_proba,
    bins=[0, 0.3, 0.6, 1.0],
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

print("\n========== BUSINESS RISK SEGMENTATION ==========")
seg_summary = X_test_copy.groupby("risk_segment").agg(
    count=("lapse_probability", "count"),
    avg_lapse_prob=("lapse_probability", "mean"),
    actual_lapse_rate=("actual_lapsed", "mean")
).round(3)
print(seg_summary)

# Export high-risk customers for retention team
high_risk = X_test_copy[X_test_copy["risk_segment"] == "High Risk"].sort_values(
    "lapse_probability", ascending=False
)
high_risk.to_csv("high_risk_customers.csv", index=False)
print(f"\n✅ {len(high_risk)} high-risk customers exported to high_risk_customers.csv")

# ── 13. CROSS VALIDATION (XGBoost) ───────────────────────────────────────────
print("\n========== 5-FOLD CROSS VALIDATION (XGBoost) ==========")
cv_scores = cross_val_score(
    XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5,
                  use_label_encoder=False, eval_metric="logloss", random_state=42),
    X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="roc_auc", n_jobs=-1
)
print(f"CV ROC-AUC scores : {cv_scores.round(4)}")
print(f"Mean ± Std        : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── 14. SAVE MODELS ──────────────────────────────────────────────────────────
import joblib
joblib.dump(xgb,    "xgb_insurance_lapse.pkl")
joblib.dump(rf,     "rf_insurance_lapse.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\n✅ Models saved: xgb_insurance_lapse.pkl | rf_insurance_lapse.pkl | scaler.pkl")

# ── 15. QUICK PREDICTION FUNCTION (for deployment) ───────────────────────────
def predict_lapse(input_dict: dict) -> dict:
    """
    Predict lapse probability for a single customer.
    input_dict keys must match FEATURES list.
    Returns lapse probability and risk segment.
    """
    model = joblib.load("xgb_insurance_lapse.pkl")
    row   = pd.DataFrame([input_dict])[FEATURES]
    prob  = model.predict_proba(row)[0][1]
    risk  = "High Risk" if prob >= 0.6 else ("Medium Risk" if prob >= 0.3 else "Low Risk")
    return {"lapse_probability": round(prob, 4), "risk_segment": risk}

# Example usage
sample_customer = dict(zip(FEATURES, X_test.iloc[0].values))
print("\n========== SAMPLE PREDICTION ==========")
print("Input:", sample_customer)
print("Output:", predict_lapse(sample_customer))

print("\n\n✅ PROJECT COMPLETE — all outputs saved.")
print("Files generated:")
print("  📊 eda_plots.png")
print("  📊 model_comparison.png")
print("  📊 feature_importance.png")
print("  📁 high_risk_customers.csv")
print("  🤖 xgb_insurance_lapse.pkl")
print("  🤖 rf_insurance_lapse.pkl")
print("  🔧 scaler.pkl")
