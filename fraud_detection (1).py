


import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
import joblib

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
CSV_PATH = r"C:\Users\ADMIN\Downloads\transactions.csv"   # put your real dataset path here
TARGET_COL = "Class"           # 1 = fraud, 0 = legitimate
RANDOM_STATE = 42
MODEL_OUT = "fraud_model.joblib"
SCALER_OUT = "fraud_scaler.joblib"


# ---------------------------------------------------------------------
# 1. LOAD DATA  (real file if present, otherwise synthetic demo data)
# ---------------------------------------------------------------------
def load_data(path: str, target_col: str) -> pd.DataFrame:
    if os.path.exists(path):
        print(f"Loading real dataset from {path} ...")
        df = pd.read_csv(path)
        return df

    print("No dataset found — generating synthetic demo data instead.")
    rng = np.random.default_rng(RANDOM_STATE)
    n_samples = 20000
    fraud_ratio = 0.015  # ~1.5% fraud, typical of real-world imbalance

    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # Legitimate transactions: smaller amounts, normal hours, low velocity
    legit = pd.DataFrame({
        "amount": rng.gamma(2.0, 50, n_legit),
        "hour_of_day": rng.normal(14, 4, n_legit).clip(0, 23),
        "txn_count_last_24h": rng.poisson(3, n_legit),
        "distance_from_home_km": rng.exponential(5, n_legit),
        "is_foreign_country": rng.binomial(1, 0.02, n_legit),
        "is_new_merchant": rng.binomial(1, 0.10, n_legit),
        TARGET_COL: 0,
    })

    # Fraudulent transactions: larger/odd amounts, odd hours, high velocity
    fraud = pd.DataFrame({
        "amount": rng.gamma(3.0, 250, n_fraud),
        "hour_of_day": rng.normal(3, 3, n_fraud).clip(0, 23),
        "txn_count_last_24h": rng.poisson(12, n_fraud),
        "distance_from_home_km": rng.exponential(300, n_fraud),
        "is_foreign_country": rng.binomial(1, 0.55, n_fraud),
        "is_new_merchant": rng.binomial(1, 0.70, n_fraud),
        TARGET_COL: 1,
    })

    df = pd.concat([legit, fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------
# 2. TRAIN / EVALUATE
# ---------------------------------------------------------------------
def main():
    df = load_data(CSV_PATH, TARGET_COL)

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    print(f"\nDataset shape: {df.shape}")
    print(f"Fraud cases: {df[TARGET_COL].sum()} "
          f"({df[TARGET_COL].mean() * 100:.3f}% of all transactions)")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Drop obvious ID columns (not real predictive features)
    id_like_cols = [c for c in X.columns if c.lower() in
                    ("id", "transaction_id", "txn_id", "index")]
    X = X.drop(columns=id_like_cols, errors="ignore")

    # Keep only numeric features for simplicity
    X = X.select_dtypes(include=[np.number])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    # Scale features (helps some models; harmless for tree models too)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # class_weight='balanced' handles the fraud class imbalance without
    # needing to oversample/undersample manually
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("\nTraining model...")
    model.fit(X_train_scaled, y_train)

    # ----- Evaluation -----
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    print("=== Confusion Matrix ===")
    cm = confusion_matrix(y_test, y_pred)
    print(pd.DataFrame(
        cm,
        index=["Actual Legit", "Actual Fraud"],
        columns=["Predicted Legit", "Predicted Fraud"],
    ))

    roc_auc = roc_auc_score(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    print(f"\nROC-AUC: {roc_auc:.4f}")
    print(f"Average Precision (PR-AUC): {avg_precision:.4f}")

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=X.columns)
    print("\n=== Top Feature Importances ===")
    print(importances.sort_values(ascending=False).to_string())

    # ----- Choosing a decision threshold -----
    # In fraud detection, the default 0.5 threshold is rarely optimal.
    # Banks usually pick a threshold that balances catching fraud (recall)
    # against the cost of flagging too many genuine transactions.
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    target_recall = 0.85
    idx = np.argmin(np.abs(recalls - target_recall))
    chosen_threshold = thresholds[max(idx - 1, 0)]
    print(f"\nExample threshold to catch ~{target_recall*100:.0f}% of fraud: "
          f"{chosen_threshold:.3f} "
          f"(precision at that point: {precisions[idx]:.3f})")

    # ----- Save model + scaler for later use -----
    joblib.dump(model, MODEL_OUT)
    joblib.dump(scaler, SCALER_OUT)
    print(f"\nSaved model to '{MODEL_OUT}' and scaler to '{SCALER_OUT}'.")

    # ----- Example: scoring a brand-new transaction -----
    print("\n=== Example: scoring a new transaction ===")
    new_txn = pd.DataFrame([{
        "amount": 980.0,
        "hour_of_day": 2,
        "txn_count_last_24h": 15,
        "distance_from_home_km": 540,
        "is_foreign_country": 1,
        "is_new_merchant": 1,
    }])
    new_txn = new_txn[X.columns]  # ensure same column order
    new_scaled = scaler.transform(new_txn)
    fraud_probability = model.predict_proba(new_scaled)[0, 1]
    print(f"Fraud probability: {fraud_probability:.2%}")
    print("Flagged as FRAUD" if fraud_probability >= chosen_threshold
          else "Looks legitimate")


if __name__ == "__main__":
    main()
