import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

PROCESSED_DIR = Path("data/processed")
TEST_DIR = Path("data/test") if (Path("data/test") / "X_test_causal.csv").exists() else PROCESSED_DIR
CAUSAL_DIR = Path("data/causal")
MODEL_DIR = Path("models")

selected = json.load(open(CAUSAL_DIR / "selected_features.json"))
X_test = pd.read_csv(TEST_DIR / "X_test_causal.csv")
y_test = pd.read_csv(TEST_DIR / "y_test.csv").squeeze().values

for col in selected:
    if col not in X_test.columns:
        X_test[col] = 0.0
X_test = X_test[selected]

booster = xgb.Booster()
booster.load_model(str(MODEL_DIR / "xgb_model.json"))
with open(MODEL_DIR / "calibrator.pkl", "rb") as f:
    calibrator = pickle.load(f)
meta = json.load(open(MODEL_DIR / "model_meta.json"))
threshold = meta["threshold"]

dtest = xgb.DMatrix(X_test, label=y_test, feature_names=selected)
raw_test_probs = booster.predict(dtest)
y_prob = calibrator.transform(raw_test_probs)
y_pred = (y_prob >= threshold).astype(int)

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
f1_attack = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
f1_benign = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)


def report(label, actual, expected, tolerance=1e-4):
    if isinstance(actual, (int, np.integer)):
        matches = actual == expected
        actual_text = str(int(actual))
        expected_text = str(int(expected))
    else:
        matches = abs(float(actual) - float(expected)) <= tolerance
        actual_text = f"{actual:.4f}"
        expected_text = f"{expected:.4f}"
    status = "MATCH" if matches else "MISMATCH"
    print(f"{label}: actual={actual_text} | paper={expected_text} | {status}")


print("CausalShield test metric verification")
print(f"Selected features: {len(selected)} -> {selected}")
report("ROC-AUC", roc_auc, 0.9393)
report("PR-AUC", pr_auc, 0.9875)
print(f"Threshold: actual={threshold:.2f} | paper=0.61 | {'MATCH' if abs(threshold - 0.61) <= 1e-4 else 'MISMATCH'}")
print("Confusion matrix order: TN, FP, FN, TP")
report("TN", cm[0, 0], 7886)
report("FP", cm[0, 1], 2114)
report("FN", cm[1, 0], 5058)
report("TP", cm[1, 1], 44380)
report("F1 attack", f1_attack, 0.9252)
report("F1 benign", f1_benign, 0.6874)
report("Macro F1", macro_f1, 0.8063)
report("Test rows", len(y_test), 59438)
