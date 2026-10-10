"""
verification/check_shap_means.py
Computes mean|SHAP| over the FULL X_test_causal.csv using XGBoost native
pred_contribs in chunks of 10,000 rows. Prints and saves results/shap_check.txt.

Run from the project root:
    python verification/check_shap_means.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH  = ROOT / "models" / "xgb_model.json"
DATA_DIR    = ROOT / "data" / "processed"
CAUSAL_DIR  = ROOT / "data" / "causal"
RESULTS_DIR = ROOT / "results"
OUT_PATH    = RESULTS_DIR / "shap_check.txt"

CHUNK = 10_000

selected = json.load(open(CAUSAL_DIR / "selected_features.json", encoding="utf-8"))

booster = xgb.Booster()
booster.load_model(str(MODEL_PATH))

# ── patch base_score if needed (XGBoost 3.x bracket notation) ─────────────
try:
    cfg = json.loads(booster.save_config())
    bs  = cfg["learner"]["learner_model_param"]["base_score"]
    if isinstance(bs, str) and bs.startswith("["):
        cfg["learner"]["learner_model_param"]["base_score"] = str(float(bs.strip("[]")))
        booster.load_config(json.dumps(cfg))
        print(f"base_score patched: {bs} → {cfg['learner']['learner_model_param']['base_score']}")
except Exception as e:
    print(f"[WARN] base_score patch skipped: {e}")

print("Loading X_test_causal.csv …")
X_test = pd.read_csv(DATA_DIR / "X_test_causal.csv")[selected]
n_total = len(X_test)
print(f"  n_rows={n_total}  n_features={len(selected)}")

# ── chunk-wise pred_contribs ───────────────────────────────────────────────
sum_abs_shap = np.zeros(len(selected), dtype=np.float64)
rows_processed = 0

for start in range(0, n_total, CHUNK):
    chunk = X_test.iloc[start : start + CHUNK]
    dm    = xgb.DMatrix(chunk, feature_names=selected)
    contribs = booster.predict(dm, pred_contribs=True)  # shape (n, n_feat+1)
    sum_abs_shap += np.abs(contribs[:, :-1]).sum(axis=0)
    rows_processed += len(chunk)
    print(f"  processed {rows_processed}/{n_total} rows …", end="\r")

print()
mean_abs_shap = sum_abs_shap / n_total

lines = [
    f"n_rows={n_total}  n_features={len(selected)}",
    "",
    f"{'Feature':<22} {'mean|SHAP|':>10}",
    f"{'-'*22} {'-'*10}",
]
for feat, val in sorted(zip(selected, mean_abs_shap), key=lambda x: -x[1]):
    lines.append(f"{feat:<22} {val:>10.4f}")

output = "\n".join(lines)
print("\n--- Full-test-set mean|SHAP| ---")
print(output)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(output + "\n", encoding="utf-8")
print(f"\nWrote: {OUT_PATH}")
