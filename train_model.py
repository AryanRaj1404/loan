import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score)
import joblib
import json
import os

os.makedirs('/home/claude/ml/artifacts', exist_ok=True)

# ── 1. Load ──────────────────────────────────────────────────────────────────
df = pd.read_csv('/mnt/user-data/uploads/credit_risk_dataset.csv')
print(f"Shape before cleaning: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")

# ── 2. Clean ─────────────────────────────────────────────────────────────────
# Fill missing Income with median (robust to outliers)
df['Income'] = df['Income'].fillna(df['Income'].median())

# Remove any remaining rows with nulls
df = df.dropna()
print(f"Shape after cleaning: {df.shape}")

# ── 3. Encode categoricals ───────────────────────────────────────────────────
edu_order = {'High School': 0, 'Bachelors': 1, 'Masters': 2, 'PhD': 3}
df['Education_Level_enc'] = df['Education_Level'].map(edu_order)

housing_order = {'Rent': 0, 'Mortgage': 1, 'Own': 2}
df['Housing_Status_enc'] = df['Housing_Status'].map(housing_order)

# ── 4. Feature engineering ───────────────────────────────────────────────────
df['Loan_to_Income']   = df['Loan_Amount'] / (df['Income'] + 1)
df['Income_per_Year']  = df['Income'] / (df['Employment_Years'] + 1)
df['Credit_Age_Ratio'] = df['Credit_Score'] / (df['Age'] + 1)

# ── 5. Define features & target ──────────────────────────────────────────────
FEATURES = [
    'Age', 'Income', 'Loan_Amount', 'Credit_Score', 'Employment_Years',
    'Education_Level_enc', 'Housing_Status_enc',
    'Loan_to_Income', 'Income_per_Year', 'Credit_Age_Ratio'
]
TARGET = 'Default'

X = df[FEATURES].values
y = df[TARGET].values

# ── 6. Scale ─────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 7. Train / test split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ── 8. Decision Tree (tuned) ─────────────────────────────────────────────────
model = DecisionTreeClassifier(
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# ── 9. Evaluate ──────────────────────────────────────────────────────────────
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc     = accuracy_score(y_test, y_pred)
auc     = roc_auc_score(y_test, y_proba)
cv_acc  = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy').mean()

print(f"\nAccuracy : {acc:.4f}")
print(f"ROC-AUC  : {auc:.4f}")
print(f"CV Acc   : {cv_acc:.4f}")
print(f"\n{classification_report(y_test, y_pred)}")

# ── 10. Persist artifacts ────────────────────────────────────────────────────
joblib.dump(model,  '/home/claude/ml/artifacts/model.pkl')
joblib.dump(scaler, '/home/claude/ml/artifacts/scaler.pkl')

meta = {
    "features": FEATURES,
    "education_order": edu_order,
    "housing_order": housing_order,
    "metrics": {
        "accuracy": round(acc, 4),
        "roc_auc":  round(auc, 4),
        "cv_accuracy": round(cv_acc, 4)
    }
}
with open('/home/claude/ml/artifacts/meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

print("\nArtifacts saved ✓")
