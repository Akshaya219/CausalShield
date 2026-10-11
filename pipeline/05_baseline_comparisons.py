"""
CausalShield - Phase 4c: Baseline Comparisons
==============================================
Trains and evaluates three models for fair comparison:

  Baseline 1: Full-feature XGBoost (all 46 features, no selection)
  Baseline 2: Mutual-Information XGBoost (top-k MI features)
  Proposed  : CausalShield XGBoost (ACE-selected causal features)

Metrics reported: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
All models use identical XGBoost hyperparameters and the same
train/val/test splits for fair comparison.

Outputs:
  - results/baseline_comparison.csv
  - results/baseline_comparison.png
  - results/baseline_report.txt
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
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("CausalShield Phase 4c -- Baseline Comparisons")
print("=" * 60)

# ── Load full data ─────────────────────────────────────────────────────────────
print("\n[1/5] Loading data ...")

TRAIN_DIR = Path("data/train") if (Path("data/train") / "X_train.csv").exists() else PROCESSED_DIR
TEST_DIR  = Path("data/test") if (Path("data/test") / "X_test.csv").exists() else PROCESSED_DIR

X_train_full = pd.read_csv(TRAIN_DIR / "X_train.csv")
X_val_full   = pd.read_csv(TRAIN_DIR / "X_val.csv")
X_test_full  = pd.read_csv(TEST_DIR / "X_test.csv")

y_train = pd.read_csv(TRAIN_DIR / "y_train.csv").squeeze().values
y_val   = pd.read_csv(TRAIN_DIR / "y_val.csv").squeeze().values
y_test  = pd.read_csv(TEST_DIR / "y_test.csv").squeeze().values

selected_causal = json.load(open(CAUSAL_DIR / "selected_features.json"))

print(f"      Full features    : {X_train_full.shape[1]}")
print(f"      Causal features  : {len(selected_causal)}")
print(f"      Train size       : {len(X_train_full)}")

import xgboost as xgb
from calibration_utils import NumpyIsotonicRegression as IsotonicRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    classification_report,
)

# ── Helper: train + evaluate one model ────────────────────────────────────────
def train_eval(X_tr, X_va, X_te, y_tr, y_va, y_te, feat_names, label):
    print(f"\n      --- {label} ({len(feat_names)} features) ---")

    neg = int((y_tr == 0).sum()); pos = int((y_tr == 1).sum())
    spw = neg / pos if pos > 0 else 1.0

    dtr = xgb.DMatrix(X_tr, label=y_tr, feature_names=feat_names)
    dva = xgb.DMatrix(X_va, label=y_va, feature_names=feat_names)
    dte = xgb.DMatrix(X_te, label=y_te, feature_names=feat_names)

    params = {
        "objective"        : "binary:logistic",
        "eval_metric"      : ["logloss", "auc"],
        "eta"              : 0.05,
        "max_depth"        : 6,
        "min_child_weight" : 5,
        "subsample"        : 0.8,
        "colsample_bytree" : 0.8,
        "gamma"            : 0.1,
        "reg_alpha"        : 0.1,
        "reg_lambda"       : 1.0,
        "scale_pos_weight" : spw,
        "seed"             : 42,
        "tree_method"      : "hist",
    }

    er = {}
    booster = xgb.train(
        params, dtr,
        num_boost_round       = 500,
        evals                 = [(dtr, "train"), (dva, "val")],
        evals_result          = er,
        early_stopping_rounds = 30,
        verbose_eval          = False,
    )
    best_round = booster.best_iteration

    # Calibrate
    raw_va  = booster.predict(dva)
    cal     = IsotonicRegression(out_of_bounds="clip")
    cal.fit(raw_va, y_va)

    raw_te  = booster.predict(dte)
    cal_te  = cal.transform(raw_te)

    # Threshold from val F1
    cal_va  = cal.transform(raw_va)
    best_t  = 0.5; best_f1 = 0.0
    for t in np.arange(0.1, 0.9, 0.01):
        f1 = f1_score(y_va, (cal_va >= t).astype(int), zero_division=0)
        if f1 > best_f1: best_f1 = f1; best_t = round(t, 2)

    y_pred = (cal_te >= best_t).astype(int)

    metrics = {
        "model"       : label,
        "n_features"  : len(feat_names),
        "best_round"  : best_round,
        "threshold"   : best_t,
        "accuracy"    : round(accuracy_score(y_te, y_pred), 4),
        "precision"   : round(precision_score(y_te, y_pred, zero_division=0), 4),
        "recall"      : round(recall_score(y_te, y_pred, zero_division=0), 4),
        "f1"          : round(f1_score(y_te, y_pred, zero_division=0), 4),
        "roc_auc"     : round(roc_auc_score(y_te, cal_te), 4),
        "pr_auc"      : round(average_precision_score(y_te, cal_te), 4),
    }

    print(f"        Acc={metrics['accuracy']}  Prec={metrics['precision']}  "
          f"Rec={metrics['recall']}  F1={metrics['f1']}  "
          f"AUC={metrics['roc_auc']}  PR={metrics['pr_auc']}")

    return metrics, booster, cal

# ── Baseline 1: Full-feature ──────────────────────────────────────────────────
print("\n[2/5] Baseline 1: Full-feature XGBoost ...")

# Drop same degenerate columns that Phase 2 dropped for fairness
feat_names_full = list(X_train_full.columns)
stds            = X_train_full.std()
const_cols      = stds[stds < 1e-10].index.tolist()
X_tr_full       = X_train_full.drop(columns=const_cols, errors="ignore")
X_va_full       = X_val_full.drop(columns=const_cols, errors="ignore")
X_te_full       = X_test_full.drop(columns=const_cols, errors="ignore")
feat_full       = list(X_tr_full.columns)

m1, b1, c1 = train_eval(
    X_tr_full.values, X_va_full.values, X_te_full.values,
    y_train, y_val, y_test, feat_full, "Full-Feature XGBoost"
)

# ── Baseline 2: Mutual Information selection ───────────────────────────────────
print("\n[3/5] Baseline 2: Mutual-Information XGBoost ...")

# Use same number of features as causal selection for fair comparison
k_mi = len(selected_causal)
print(f"      Computing MI scores for {X_tr_full.shape[1]} features (k={k_mi}) ...")

mi_scores  = mutual_info_classif(X_tr_full, y_train, random_state=42, n_neighbors=5)
mi_ranking = np.argsort(mi_scores)[::-1]
mi_feats   = [feat_full[i] for i in mi_ranking[:k_mi]]
# Print MI selected features with their scores
print(f"\n      MI Selected Features (top {k_mi} by KSG mutual information):")
print(f"      {'Rank':<5} {'Feature':<35} {'MI Score':>10}")
print(f"      {'-'*5} {'-'*35} {'-'*10}")
for rank, idx in enumerate(mi_ranking[:k_mi], 1):
    print(f"      {rank:<5} {feat_full[idx]:<35} {mi_scores[idx]:>10.6f}")

X_tr_mi = X_tr_full[mi_feats].values
X_va_mi = X_va_full[mi_feats].values
X_te_mi = X_te_full[mi_feats].values

m2, b2, c2 = train_eval(
    X_tr_mi, X_va_mi, X_te_mi,
    y_train, y_val, y_test, mi_feats, "MI-Selected XGBoost"
)

# Save MI feature list for reference
json.dump(
    {"mi_features": mi_feats, "mi_scores": {feat_full[i]: round(float(mi_scores[i]), 6)
                                              for i in mi_ranking[:k_mi]}},
    open(CAUSAL_DIR / "mi_selected_features.json", "w"), indent=2
)

# ── Proposed: CausalShield ────────────────────────────────────────────────────
print("\n[4/5] Proposed: CausalShield (ACE-selected features) ...")

X_tr_c = pd.read_csv(PROCESSED_DIR / "X_train_causal.csv")[selected_causal].values
X_va_c = pd.read_csv(PROCESSED_DIR / "X_val_causal.csv")[selected_causal].values
X_te_c = pd.read_csv(PROCESSED_DIR / "X_test_causal.csv")[selected_causal].values

m3, b3, c3 = train_eval(
    X_tr_c, X_va_c, X_te_c,
    y_train, y_val, y_test, selected_causal, "CausalShield (ACE-Selected)"
)

# ── 5. Comparison table ────────────────────────────────────────────────────────
print("\n[5/5] Saving comparison results ...")

results = pd.DataFrame([m1, m2, m3])
results.to_csv(RESULTS_DIR / "baseline_comparison.csv", index=False)

print("\n      =" * 30)
print(f"\n      {'Model':<30} {'Feats':>5} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6} {'PR':>6}")
print(f"      {'-'*30} {'-'*5} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
for _, row in results.iterrows():
    print(f"      {row['model']:<30} {row['n_features']:>5} "
          f"{row['accuracy']:>6.4f} {row['precision']:>6.4f} "
          f"{row['recall']:>6.4f} {row['f1']:>6.4f} "
          f"{row['roc_auc']:>6.4f} {row['pr_auc']:>6.4f}")

# Save text report
with open(RESULTS_DIR / "baseline_report.txt", "w") as f:
    f.write("CausalShield -- Baseline Comparison Report\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"{'Model':<30} {'Feats':>5} {'Acc':>6} {'Prec':>6} "
            f"{'Rec':>6} {'F1':>6} {'AUC':>6} {'PR':>6}\n")
    f.write("-" * 70 + "\n")
    for _, row in results.iterrows():
        f.write(f"{row['model']:<30} {row['n_features']:>5} "
                f"{row['accuracy']:>6.4f} {row['precision']:>6.4f} "
                f"{row['recall']:>6.4f} {row['f1']:>6.4f} "
                f"{row['roc_auc']:>6.4f} {row['pr_auc']:>6.4f}\n")
    f.write("\n\nNote: All models use identical XGBoost hyperparameters, "
            "isotonic calibration,\nand F1-optimal threshold on validation set.\n")

# ── Bar chart comparison ───────────────────────────────────────────────────────
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    model_labels    = ["Full-Feature", "MI-Selected", "CausalShield"]
    colors          = ["#95a5a6", "#3498db", "#e74c3c"]
    x               = np.arange(len(metrics_to_plot))
    width           = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (row, color, label) in enumerate(zip([m1, m2, m3], colors, model_labels)):
        vals = [row[m] for m in metrics_to_plot]
        bars = ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels([m.upper().replace("_", "\n") for m in metrics_to_plot], fontsize=10)
    # ax.set_ylim(max(0, min(m1["f1"], m2["f1"], m3["f1"]) - 0.05), 1.02)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("CausalShield vs Baselines -- Performance Comparison\n"
                 "(all models: same hyperparams, causal vs correlation vs full features)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "baseline_comparison.png", dpi=150)
    plt.close()
    print(f"\n      baseline_comparison.png -> {RESULTS_DIR}")
except Exception as e:
    print(f"      [WARN] Plot skipped: {e}")

print("\n" + "=" * 60)
print("Phase 4c complete -- Baseline Comparisons")
print(f"  Full-Feature  : F1={m1['f1']:.4f}  AUC={m1['roc_auc']:.4f}")
print(f"  MI-Selected   : F1={m2['f1']:.4f}  AUC={m2['roc_auc']:.4f}")
print(f"  CausalShield  : F1={m3['f1']:.4f}  AUC={m3['roc_auc']:.4f}")
print("=" * 60)