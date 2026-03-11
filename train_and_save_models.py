"""
train_and_save_models.py
────────────────────────
Trains 3 classifiers and saves them to models/.
Requires: creditcard.csv in the project root.
Download from → https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Usage:
    python train_and_save_models.py
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

os.makedirs("models", exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading dataset…")
df = pd.read_csv("creditcard.csv")
print(f"  Shape  : {df.shape}")
print(f"  Fraud  : {df['Class'].sum()} ({df['Class'].mean()*100:.4f}%)")
print(f"  Legit  : {(df['Class']==0).sum()}")

# ── Preprocessing ─────────────────────────────────────────────────────────────
scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df[["Amount"]])
df["Time"]   = scaler.fit_transform(df[["Time"]])
joblib.dump(scaler, "models/scaler.pkl")
print("\n  Scaler saved → models/scaler.pkl")

X = df.drop("Class", axis=1).values
y = df["Class"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── SMOTE ─────────────────────────────────────────────────────────────────────
print("\nApplying SMOTE to balance classes…")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print(f"  Before SMOTE : Legit={y_train.sum()==0}, Fraud={y_train.sum()}")
print(f"  After  SMOTE : Legit={(y_train_res==0).sum():,}  Fraud={(y_train_res==1).sum():,}")

# ── Helper ────────────────────────────────────────────────────────────────────
def train_eval(name, clf, path):
    print(f"\n{'─'*52}")
    print(f"  {name}")
    print(f"{'─'*52}")
    clf.fit(X_train_res, y_train_res)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    joblib.dump(clf, path)
    print(f"  ✅  Saved → {path}")

# ── 1. Logistic Regression ────────────────────────────────────────────────────
train_eval(
    "1️⃣  Logistic Regression",
    LogisticRegression(
        max_iter = 1000,
        C        = 0.01,
        solver   = "lbfgs",
        random_state = 42
    ),
    "models/lr_model.pkl"
)

# ── 2. Decision Tree ──────────────────────────────────────────────────────────
train_eval(
    "2️⃣  Decision Tree",
    DecisionTreeClassifier(
        max_depth        = 10,
        min_samples_split = 10,
        min_samples_leaf  = 2,
        random_state     = 42
    ),
    "models/dt_model.pkl"
)

# ── 3. K-Nearest Neighbor ────────────────────────────────────────────────────
train_eval(
    "3️⃣  K-Nearest Neighbor",
    KNeighborsClassifier(
        n_neighbors = 5,
        metric      = "minkowski",
        n_jobs      = -1
    ),
    "models/knn_model.pkl"
)

print("\n" + "="*52)
print("  ✅  All 3 models saved to models/")
print("="*52)
print()
print("  Models produced:")
print("    models/lr_model.pkl   ← Logistic Regression")
print("    models/dt_model.pkl   ← Decision Tree")
print("    models/knn_model.pkl  ← K-Nearest Neighbor")
print("    models/scaler.pkl     ← StandardScaler")