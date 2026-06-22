# Bank Transaction Fraud Detection

A complete, runnable machine learning pipeline that detects fraudulent
bank transactions using a Random Forest classifier.

## Files

| File | Description |
|---|---|
| `fraud_detection.py` | Main script: loads data, trains model, evaluates, saves model |
| `transactions.csv` | Sample dataset (20,000 transactions, 1.5% fraud) |
| `fraud_model.joblib` | Trained model (generated after running the script) |
| `fraud_scaler.joblib` | Feature scaler used at training time (generated after running) |

## Requirements

- Python 3.8+
- Packages: `pandas`, `numpy`, `scikit-learn`, `joblib`

Install them with:

```bash
pip install pandas numpy scikit-learn joblib
```

## How to Run

1. Put `fraud_detection.py` and `transactions.csv` in the same folder.
2. Run:

```bash
python fraud_detection.py
```

If `transactions.csv` is present, the script trains on it. If it's
missing, the script automatically generates a synthetic dataset so it
still runs end-to-end.

## Dataset (`transactions.csv`)

20,000 rows, 300 of which are fraud (~1.5%, realistic for real banking
data). Columns:

| Column | Description |
|---|---|
| `transaction_id` | Unique transaction identifier (dropped before training — not predictive) |
| `amount` | Transaction amount |
| `hour_of_day` | Hour the transaction occurred (0–23) |
| `txn_count_last_24h` | Number of transactions by this account in the last 24 hours |
| `distance_from_home_km` | Distance between transaction location and home address |
| `is_foreign_country` | 1 if the transaction occurred abroad, else 0 |
| `is_new_merchant` | 1 if this is a merchant the account hasn't used before, else 0 |
| `Class` | Target label: 1 = fraud, 0 = legitimate |

To use your own data, replace `transactions.csv` with your file (or
change `CSV_PATH` in the script) and make sure it has numeric features
plus a binary target column. By default the target column is named
`Class`; change `TARGET_COL` in the script if yours is named differently.

## What the Script Does

1. **Loads data** — real CSV if found, otherwise generates synthetic data.
2. **Cleans features** — drops ID columns, keeps only numeric features.
3. **Splits data** — 75% train / 25% test, stratified by fraud label.
4. **Scales features** — `StandardScaler`.
5. **Trains a Random Forest** with `class_weight="balanced"` to handle
   the natural rarity of fraud cases.
6. **Evaluates the model** using:
   - Precision, recall, F1-score
   - Confusion matrix
   - ROC-AUC and PR-AUC (more meaningful than accuracy on imbalanced data)
   - Feature importances
7. **Picks an example decision threshold** that catches ~85% of fraud,
   showing the precision trade-off at that point.
8. **Saves the trained model and scaler** (`fraud_model.joblib`,
   `fraud_scaler.joblib`) so they can be reused without retraining.
9. **Scores a sample new transaction** to demonstrate real-time use.

## Using the Saved Model Later

```python
import joblib
import pandas as pd

model = joblib.load("fraud_model.joblib")
scaler = joblib.load("fraud_scaler.joblib")

new_txn = pd.DataFrame([{
    "amount": 980.0,
    "hour_of_day": 2,
    "txn_count_last_24h": 15,
    "distance_from_home_km": 540,
    "is_foreign_country": 1,
    "is_new_merchant": 1,
}])

scaled = scaler.transform(new_txn)
fraud_probability = model.predict_proba(scaled)[0, 1]
print(f"Fraud probability: {fraud_probability:.2%}")
```

## Notes on Choosing a Threshold

Fraud detection rarely uses the default 0.5 probability cutoff. Banks
typically pick a threshold based on business cost: missing fraud is
expensive, but flagging too many legitimate transactions frustrates
customers. The script shows one example threshold (tuned for ~85%
recall) — adjust `target_recall` in `fraud_detection.py` to match your
own risk tolerance.

## Customizing Further

- **Real-world data**: try the public Kaggle "Credit Card Fraud
  Detection" dataset (search "Kaggle credit card fraud detection").
- **Better models**: swap `RandomForestClassifier` for `XGBClassifier`
  or `LGBMClassifier` for typically stronger performance.
- **Imbalance handling**: try `imbalanced-learn`'s `SMOTE` as an
  alternative (or addition) to `class_weight="balanced"`.
- **More features**: real banking systems also use merchant category,
  device fingerprint, IP address/geolocation mismatches, and historical
  spending patterns per account.
