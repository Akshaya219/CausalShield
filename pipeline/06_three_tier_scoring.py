"""
CausalShield - Phase 4d: Three-Tier Risk Scoring
=================================================
Converts calibrated probability -> actionable 3-tier risk level:

  LOW    : prob < low_thresh   -> Likely benign, monitor only
  MEDIUM : low_thresh <= prob < high_thresh -> Suspicious, investigate
  HIGH   : prob >= high_thresh -> Confirmed threat, block/alert

Thresholds are tuned on the validation set to maximise:
  - LOW tier    : high specificity (avoid false alerts on benign)
  - HIGH tier   : high sensitivity (catch real attacks)
  - MEDIUM tier : transition zone

Outputs:
  - models/tier_thresholds.json   (LOW/HIGH cutoffs)
  - results/tier_analysis.png
  - results/tier_report.txt
"""

import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

PROCESSED_DIR = Path("data/processed")
CAUSAL_DIR    = Path("data/causal")
MODEL_DIR     = Path("models")
RESULTS_DIR   = Path("results")

print("=" * 60)
print("CausalShield Phase 4d -- Three-Tier Risk Scoring")
print("=" * 60)

# ── Load model and data ────────────────────────────────────────────────────────
import xgboost as xgb

selected   = json.load(open(CAUSAL_DIR / "selected_features.json"))
model_meta = json.load(open(MODEL_DIR  / "model_meta.json"))

booster = xgb.Booster()
booster.load_model(str(MODEL_DIR / "xgb_model.json"))

with open(MODEL_DIR / "calibrator.pkl", "rb") as f:
    cal = pickle.load(f)

TRAIN_DIR = Path("data/train") if (Path("data/train") / "X_val_causal.csv").exists() else PROCESSED_DIR
TEST_DIR  = Path("data/test") if (Path("data/test") / "X_test_causal.csv").exists() else PROCESSED_DIR

X_val  = pd.read_csv(TRAIN_DIR / "X_val_causal.csv")[selected]
y_val  = pd.read_csv(TRAIN_DIR / "y_val.csv").squeeze().values
X_test = pd.read_csv(TEST_DIR / "X_test_causal.csv")[selected]
y_test = pd.read_csv(TEST_DIR / "y_test.csv").squeeze().values

dval  = xgb.DMatrix(X_val,  feature_names=selected)
dtest = xgb.DMatrix(X_test, feature_names=selected)

p_val  = cal.transform(booster.predict(dval))
p_test = cal.transform(booster.predict(dtest))

print(f"      Val  size : {len(p_val)}")
print(f"      Test size : {len(p_test)}")

# ── Tune thresholds on validation set ─────────────────────────────────────────
print("\n[1/3] Tuning tier thresholds on validation set ...")

# Strategy:
#   LOW threshold  : highest prob where false positive rate < 5%
#                    (almost all below this are truly benign)
#   HIGH threshold : lowest prob where true positive rate > 95%
#                    (almost all above this are truly attacks)

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_val, p_val)

# LOW cutoff: FPR <= 0.05
low_candidates  = thresholds[fpr <= 0.05]
low_thresh      = float(low_candidates.min()) if len(low_candidates) > 0 else 0.2

# HIGH cutoff: TPR >= 0.95
high_candidates = thresholds[tpr >= 0.95]
high_thresh     = float(high_candidates.max()) if len(high_candidates) > 0 else 0.7

# Sanity: low < high
if low_thresh >= high_thresh:
    low_thresh  = 0.25
    high_thresh = 0.65

print(f"      LOW  threshold  : {low_thresh:.3f}  (FPR <= 5%)")
print(f"      HIGH threshold  : {high_thresh:.3f}  (TPR >= 95%)")

def assign_tier(prob, low_t, high_t):
    if prob >= high_t:  return "HIGH"
    if prob >= low_t:   return "MEDIUM"
    return "LOW"

# ── Evaluate tiers on test set ────────────────────────────────────────────────
print("\n[2/3] Evaluating tier distribution on test set ...")

tiers_test = np.array([assign_tier(p, low_thresh, high_thresh) for p in p_test])

for tier in ["LOW", "MEDIUM", "HIGH"]:
    mask       = tiers_test == tier
    n_total    = mask.sum()
    n_attack   = y_test[mask].sum()
    n_benign   = n_total - n_attack
    attack_pct = 100 * n_attack / n_total if n_total > 0 else 0
    print(f"      {tier:<8}: {n_total:>6} flows  |  "
          f"{n_attack:>5} attacks ({attack_pct:.1f}%)  |  "
          f"{n_benign:>5} benign")

# Save thresholds
tier_meta = {
    "low_threshold"  : round(low_thresh,  4),
    "high_threshold" : round(high_thresh, 4),
    "tier_labels"    : {
        "LOW"   : f"prob < {low_thresh:.3f}  -- Monitor only",
        "MEDIUM": f"{low_thresh:.3f} <= prob < {high_thresh:.3f}  -- Investigate",
        "HIGH"  : f"prob >= {high_thresh:.3f}  -- Block / Alert",
    },
    "tier_colors"    : {"LOW": "#2ecc71", "MEDIUM": "#f39c12", "HIGH": "#e74c3c"},
}
json.dump(tier_meta, open(MODEL_DIR / "tier_thresholds.json", "w"), indent=2)

# ── Report ────────────────────────────────────────────────────────────────────
print("\n[3/3] Saving tier report ...")

with open(RESULTS_DIR / "tier_report.txt", "w") as f:
    f.write("CausalShield -- Three-Tier Risk Score Report\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"LOW  threshold : {low_thresh:.4f}  (FPR <= 5% on val)\n")
    f.write(f"HIGH threshold : {high_thresh:.4f}  (TPR >= 95% on val)\n\n")
    f.write(f"{'Tier':<8} {'Count':>7} {'Attacks':>8} {'Benign':>8} {'Attack%':>8}\n")
    f.write("-" * 45 + "\n")
    for tier in ["LOW", "MEDIUM", "HIGH"]:
        mask       = tiers_test == tier
        n_total    = mask.sum()
        n_attack   = int(y_test[mask].sum())
        n_benign   = n_total - n_attack
        attack_pct = 100 * n_attack / n_total if n_total > 0 else 0
        f.write(f"{tier:<8} {n_total:>7} {n_attack:>8} {n_benign:>8} {attack_pct:>7.1f}%\n")

# ── Visualisation ──────────────────────────────────────────────────────────────
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: probability distribution coloured by tier
    ax = axes[0]
    tier_colors = {"LOW": "#2ecc71", "MEDIUM": "#f39c12", "HIGH": "#e74c3c"}
    for tier, color in tier_colors.items():
        mask = tiers_test == tier
        if mask.sum() > 0:
            ax.hist(p_test[mask], bins=40, alpha=0.6, color=color, label=tier)
    ax.axvline(low_thresh,  color="#f39c12", linestyle="--", lw=1.5,
               label=f"LOW/MED ({low_thresh:.2f})")
    ax.axvline(high_thresh, color="#e74c3c", linestyle="--", lw=1.5,
               label=f"MED/HIGH ({high_thresh:.2f})")
    ax.set_xlabel("Risk Score"); ax.set_ylabel("Count")
    ax.set_title("Risk Score Distribution by Tier"); ax.legend(fontsize=9)

    # Right: tier composition (attack vs benign)
    ax = axes[1]
    tier_names = ["LOW", "MEDIUM", "HIGH"]
    n_attacks  = []
    n_benigns  = []
    for tier in tier_names:
        mask = tiers_test == tier
        n_attacks.append(int(y_test[mask].sum()))
        n_benigns.append(int((y_test[mask] == 0).sum()))

    x = np.arange(3)
    ax.bar(x, n_benigns, label="Benign", color="#2ecc71", alpha=0.8)
    ax.bar(x, n_attacks, bottom=n_benigns, label="Attack", color="#e74c3c", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(tier_names)
    ax.set_ylabel("Flow count"); ax.set_title("Tier Composition")
    ax.legend(fontsize=9)

    plt.suptitle("CausalShield -- Three-Tier Risk Scoring", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "tier_analysis.png", dpi=150); plt.close()
    print(f"      tier_analysis.png -> {RESULTS_DIR}")
except Exception as e:
    print(f"      [WARN] Plot skipped: {e}")

print("\n" + "=" * 60)
print("Phase 4d complete -- Three-Tier Scoring")
print(f"  LOW  (<{low_thresh:.3f})  : monitor")
print(f"  MED  ({low_thresh:.3f}-{high_thresh:.3f}) : investigate")
print(f"  HIGH (>{high_thresh:.3f}) : block/alert")
print("=" * 60)
