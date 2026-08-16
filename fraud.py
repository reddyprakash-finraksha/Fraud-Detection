"""
TASK 2
- Credit Card Fraud Detection (Advanced)
--------------------------------------------------
Same dataset and core algorithms as Task 2 (Logistic Regression, Decision
Tree, Random Forest) -- but going deeper:
  1. SMOTE (synthetic oversampling) compared against class_weight='balanced'
  2. ROC curves for all models, plotted together
  3. Feature importance -- which columns actually drive fraud predictions
  4. SHAP explainability -- WHY a specific transaction was flagged
     (this is exactly what a real bank / FinRaksha-style system needs to
     justify a fraud decision to a customer or auditor)

Dataset: same as Task 2 -- creditcard.csv (Kaggle, ULB Machine Learning Group)
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import shap

DATA_PATH = "creditcard.csv"

# ---------- 1. Load + Scale (same as Task 2) ----------
df = pd.read_csv(DATA_PATH)
fraud_count = df["Class"].sum()
print(f"Rows: {len(df)} | Fraud: {fraud_count} ({fraud_count/len(df)*100:.3f}%)")

scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
df["Time_scaled"] = scaler.fit_transform(df[["Time"]])
df = df.drop(["Amount", "Time"], axis=1)

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- 2. SMOTE: create synthetic fraud examples (train set only!) ----------
print("\nBefore SMOTE:", y_train.value_counts().to_dict())
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("After SMOTE :", pd.Series(y_train_smote).value_counts().to_dict())
print("Note: SMOTE is applied ONLY to training data. Test data stays untouched")
print("-- otherwise we'd be evaluating on fake transactions.\n")

# ---------- 3. Train 3 required models on SMOTE data ----------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
}

results = {}
roc_curves = {}
best_name, best_score, best_model = None, 0, None

for name, model in models.items():
    print(f"Training {name} (on SMOTE-balanced data) ...")
    model.fit(X_train_smote, y_train_smote)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, probs)
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_curves[name] = (fpr, tpr, roc_auc)

    results[name] = {"precision": prec, "recall": rec, "f1": f1, "roc_auc": roc_auc}
    print(f"  Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

    if roc_auc > best_score:
        best_score, best_name, best_model = roc_auc, name, model

print("\n===== RESULTS (SMOTE-trained models, ranked by ROC-AUC) =====")
for name, r in sorted(results.items(), key=lambda x: x[1]["roc_auc"], reverse=True):
    print(f"{name:22s} | Prec: {r['precision']:.4f} | Recall: {r['recall']:.4f} | "
          f"F1: {r['f1']:.4f} | ROC-AUC: {r['roc_auc']:.4f}")
print(f"\nBest model: {best_name}")
print("Compare these Recall numbers to your Task 2 (class_weight='balanced') results --")
print("that's the real test of whether SMOTE helped more than simple reweighting.")

# ---------- 4. ROC Curve plot (all 3 models together) ----------
plt.figure(figsize=(7, 6))
for name, (fpr, tpr, auc) in roc_curves.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curves - Task 4 Fraud Detection")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150)
print("\nSaved: roc_curves.png")

# ---------- 5. Feature Importance (Random Forest) ----------
rf_model = models["Random Forest"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
top10 = importances.head(10)

plt.figure(figsize=(8, 5))
top10[::-1].plot(kind="barh", color="#1a3c6e")
plt.xlabel("Importance")
plt.title("Top 10 Most Important Features (Random Forest)")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
print("Saved: feature_importance.png")
print("\nTop 5 features driving fraud predictions:")
print(top10.head(5))

# ---------- 6. SHAP explainability (WHY a transaction was flagged) ----------
print("\nComputing SHAP values for Random Forest (this explains individual predictions)...")
explainer = shap.TreeExplainer(rf_model)
sample = X_test.sample(min(200, len(X_test)), random_state=42)  # sample for speed
shap_values = explainer.shap_values(sample)

# shap_values shape depends on the shap library version -- handle all 3 formats:
#   list of 2 arrays (older versions), 3D array (samples, features, classes),
#   or already a plain 2D array (samples, features)
if isinstance(shap_values, list):
    sv = shap_values[1]                       # class 1 = fraud
elif shap_values.ndim == 3:
    sv = shap_values[:, :, 1]                 # class 1 = fraud
else:
    sv = shap_values

plt.figure()
shap.summary_plot(sv, sample, show=False, plot_size=(8, 6))
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150)
plt.close()
print("Saved: shap_summary.png -- shows which features push predictions toward 'fraud'")

# ---------- 7. Save Best Model ----------
joblib.dump(best_model, "fraud_model_advanced.pkl")
joblib.dump(scaler, "amount_time_scaler_advanced.pkl")
print(f"\nSaved: fraud_model_advanced.pkl ({best_name}), amount_time_scaler_advanced.pkl")