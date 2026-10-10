"""
CausalShield - Phase 4: XGBoost Risk Scoring Model
===================================================
Trains an XGBoost classifier on the causally-selected features.
Outputs a calibrated risk score ∈ [0, 1] per flow.

Steps:
  1. Load causal train/val/test splits + selected feature list
  2. 5-Fold Stratified Cross-Validation on train+val (statistical robustness)
  3. Train final XGBoost on full train set with early stopping on val
  4. Calibrate probabilities with Isotonic Regression
  5. Evaluate on held-out test set — honest reporting, no inflation
  6. Save model artifacts

Outputs:
  - models/xgb_model.json             (XGBoost booster — final model)
  - models/calibrator.pkl             (sklearn calibrator)
  - models/model_meta.json            (thresholds, feature list, scores)
  - results/cv_results.json           (per-fold + mean/std metrics)
  - results/cv_boxplot.png            (cross-validation metric distribution)
  - results/classification_report.txt
  - results/confusion_matrix.png
  - results/roc_curve.png
  - results/calibration_curve.png
  - results/training_history.png
"""

import json
import pickle
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Paths ──────────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
CAUSAL_DIR    = Path("data/causal")
MODEL_DIR     = Path("models")
RESULTS_DIR   = Path("results")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("CausalShield Phase 4 — XGBoost Risk Scoring (+ 5-Fold CV)")
print("=" * 65)

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("\n[1/7] Loading causal feature datasets …")

import xgboost as xgb
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
    roc_curve, precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold
from calibration_utils import NumpyIsotonicRegression as IsotonicRegression
from calibration_utils import calibration_curve

selected = json.load(open(CAUSAL_DIR / "selected_features.json"))
print(f"      Selected features : {len(selected)}  →  {selected}")

X_train = pd.read_csv(PROCESSED_DIR / "X_train_causal.csv")
y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze().values

X_val   = pd.read_csv(PROCESSED_DIR / "X_val_causal.csv")
y_val   = pd.read_csv(PROCESSED_DIR / "y_val.csv").squeeze().values

X_test  = pd.read_csv(PROCESSED_DIR / "X_test_causal.csv")
y_test  = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze().values

# Align columns to selected feature list (zero-fill any missing)
for df in [X_train, X_val, X_test]:
    for col in selected:
        if col not in df.columns:
            df[col] = 0.0

X_train = X_train[selected]
X_val   = X_val[selected]
X_test  = X_test[selected]

print(f"      Train : {X_train.shape}  y={np.bincount(y_train)}")
print(f"      Val   : {X_val.shape}    y={np.bincount(y_val)}")
print(f"      Test  : {X_test.shape}   y={np.bincount(y_test)}")

# Class weight
neg = int((y_train == 0).sum())
pos = int((y_train == 1).sum())
spw = neg / pos if pos > 0 else 1.0
print(f"      scale_pos_weight = {spw:.4f}  (neg={neg}, pos={pos})")

# XGBoost params — shared across CV folds and final model
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

# ── 2. 5-Fold Stratified Cross-Validation ─────────────────────────────────────
print("\n[2/7] 5-Fold Stratified Cross-Validation …")
print("      Purpose : Prove results are stable across data splits,")
print("                not an artefact of a single train/val partition.")
print("      Data    : train + val combined  (test held out — never touched here)")

# Combine train + val for CV (test stays completely untouched)
X_cv = pd.concat([X_train, X_val], ignore_index=True)
y_cv = np.concatenate([y_train, y_val])

print(f"      CV pool : {X_cv.shape}  y={np.bincount(y_cv)}")

N_FOLDS    = 5
CV_ROUNDS  = 300          # fixed rounds for CV (faster than early stopping per fold)
CV_VERBOSE = False

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

fold_metrics = {
    "fold"   : [],
    "auc_roc": [],
    "auc_pr" : [],
    "f1"     : [],
    "acc"    : [],
}

print(f"\n      {'Fold':<6} {'ROC-AUC':<10} {'PR-AUC':<10} {'F1':<10} {'Acc':<10}")
print("      " + "-" * 44)

for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_cv, y_cv), 1):

    X_f_tr, X_f_va = X_cv.iloc[tr_idx], X_cv.iloc[va_idx]
    y_f_tr, y_f_va = y_cv[tr_idx],      y_cv[va_idx]

    # Recompute SPW per fold (class ratio may shift slightly)
    fold_neg = int((y_f_tr == 0).sum())
    fold_pos = int((y_f_tr == 1).sum())
    fold_spw = fold_neg / fold_pos if fold_pos > 0 else 1.0

    fold_params = {**params, "scale_pos_weight": fold_spw}

    d_tr = xgb.DMatrix(X_f_tr, label=y_f_tr, feature_names=selected)
    d_va = xgb.DMatrix(X_f_va, label=y_f_va, feature_names=selected)

    bst_fold = xgb.train(
        fold_params, d_tr,
        num_boost_round       = CV_ROUNDS,
        evals                 = [(d_va, "val")],
        early_stopping_rounds = 20,
        verbose_eval          = CV_VERBOSE,
    )

    # Threshold tuning on this fold's val set
    probs_va = bst_fold.predict(d_va)

    # Calibrate within fold (isotonic on fold val)
    cal_fold = IsotonicRegression(out_of_bounds="clip")
    cal_fold.fit(probs_va, y_f_va)
    probs_cal = cal_fold.transform(probs_va)

    # Find best F1 threshold on this fold's val
    best_f1_fold, best_th_fold = 0.0, 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        preds_t = (probs_cal >= t).astype(int)
        sc      = f1_score(y_f_va, preds_t, zero_division=0)
        if sc > best_f1_fold:
            best_f1_fold = sc
            best_th_fold = t

    preds_fold = (probs_cal >= best_th_fold).astype(int)

    auc  = roc_auc_score(y_f_va, probs_cal)
    pr   = average_precision_score(y_f_va, probs_cal)
    f1   = f1_score(y_f_va, preds_fold, zero_division=0)
    acc  = (preds_fold == y_f_va).mean()

    fold_metrics["fold"].append(fold_idx)
    fold_metrics["auc_roc"].append(round(auc, 4))
    fold_metrics["auc_pr"].append(round(pr,  4))
    fold_metrics["f1"].append(round(f1,      4))
    fold_metrics["acc"].append(round(acc,    4))

    print(f"      {fold_idx:<6} {auc:<10.4f} {pr:<10.4f} {f1:<10.4f} {acc:<10.4f}")

# ── CV Summary ─────────────────────────────────────────────────────────────────
print("\n      " + "=" * 44)
cv_auc = np.array(fold_metrics["auc_roc"])
cv_pr  = np.array(fold_metrics["auc_pr"])
cv_f1  = np.array(fold_metrics["f1"])
cv_acc = np.array(fold_metrics["acc"])

print(f"      {'Mean':<6} {cv_auc.mean():<10.4f} {cv_pr.mean():<10.4f} {cv_f1.mean():<10.4f} {cv_acc.mean():<10.4f}")
print(f"      {'Std':<6} {cv_auc.std():<10.4f} {cv_pr.std():<10.4f} {cv_f1.std():<10.4f} {cv_acc.std():<10.4f}")
print(f"      {'95% CI AUC'} : {cv_auc.mean():.4f} ± {1.96 * cv_auc.std() / np.sqrt(N_FOLDS):.4f}")
print(f"      {'95% CI F1 '} : {cv_f1.mean():.4f} ± {1.96 * cv_f1.std()  / np.sqrt(N_FOLDS):.4f}")
print("      " + "=" * 44)

# Stability check
auc_cv  = cv_auc.std()
if auc_cv < 0.005:
    print(f"\n      ✅ STABLE  — AUC std = {auc_cv:.4f} (< 0.005). Results are not a lucky split.")
elif auc_cv < 0.015:
    print(f"\n      ⚠️  MODERATE variance — AUC std = {auc_cv:.4f}. Acceptable for network datasets.")
else:
    print(f"\n      ❌ HIGH variance — AUC std = {auc_cv:.4f}. Investigate class imbalance per fold.")

# Save CV results
cv_summary = {
    "n_folds"       : N_FOLDS,
    "cv_rounds"     : CV_ROUNDS,
    "per_fold"      : fold_metrics,
    "mean_auc_roc"  : round(float(cv_auc.mean()), 4),
    "std_auc_roc"   : round(float(cv_auc.std()),  4),
    "ci95_auc_roc"  : round(float(1.96 * cv_auc.std() / np.sqrt(N_FOLDS)), 4),
    "mean_pr_auc"   : round(float(cv_pr.mean()),  4),
    "std_pr_auc"    : round(float(cv_pr.std()),   4),
    "mean_f1"       : round(float(cv_f1.mean()),  4),
    "std_f1"        : round(float(cv_f1.std()),   4),
    "ci95_f1"       : round(float(1.96 * cv_f1.std() / np.sqrt(N_FOLDS)), 4),
    "mean_acc"      : round(float(cv_acc.mean()), 4),
    "std_acc"       : round(float(cv_acc.std()),  4),
}
with open(RESULTS_DIR / "cv_results.json", "w") as f:
    json.dump(cv_summary, f, indent=2)
print(f"\n      cv_results.json → {RESULTS_DIR}")

# CV boxplot
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    metrics_for_plot = {
        "ROC-AUC": fold_metrics["auc_roc"],
        "PR-AUC" : fold_metrics["auc_pr"],
        "F1"     : fold_metrics["f1"],
        "Accuracy": fold_metrics["acc"],
    }
    labels = list(metrics_for_plot.keys())
    data   = list(metrics_for_plot.values())

    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False,
                    medianprops=dict(color="white", linewidth=2))
    colors = ["#2E5FA3", "#1F3864", "#e74c3c", "#27ae60"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.8)

    # Overlay individual fold points
    for i, d in enumerate(data, 1):
        ax.scatter([i] * len(d), d, color="white", zorder=5, s=40, edgecolors="black")

    # Annotate mean ± std
    for i, (d, label) in enumerate(zip(data, labels), 1):
        m, s = np.mean(d), np.std(d)
        ax.text(i, min(d) - 0.003, f"{m:.4f}±{s:.4f}",
                ha="center", va="top", fontsize=8, color="#333333")

    ax.set_ylim(max(0, min(min(d) for d in data) - 0.02), 1.02)
    ax.set_ylabel("Score")
    ax.set_title(
        f"CausalShield — 5-Fold Stratified CV Results\n"
        f"ROC-AUC = {cv_auc.mean():.4f} ± {cv_auc.std():.4f}   "
        f"F1 = {cv_f1.mean():.4f} ± {cv_f1.std():.4f}",
        fontweight="bold"
    )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "cv_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      cv_boxplot.png  → {RESULTS_DIR}")
except Exception as e:
    print(f"      [WARN] CV plot skipped: {e}")

# ── 3. Final Model Training ────────────────────────────────────────────────────
print("\n[3/7] Training final model on full train set …")
print("      (Using val set for early stopping — same as original Phase 4)")

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=selected)
dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=selected)
dtest  = xgb.DMatrix(X_test,  label=y_test,  feature_names=selected)

evals_result = {}
booster = xgb.train(
    params,
    dtrain,
    num_boost_round       = 500,
    evals                 = [(dtrain, "train"), (dval, "val")],
    evals_result          = evals_result,
    early_stopping_rounds = 30,
    verbose_eval          = 50,
)

best_round = booster.best_iteration
print(f"\n      Best round : {best_round}")
print(f"      Val AUC    : {max(evals_result['val']['auc']):.4f}")
print(f"      Val LogLoss: {min(evals_result['val']['logloss']):.4f}")

# ── 4. Calibrate probabilities ────────────────────────────────────────────────
print("\n[4/7] Calibrating probabilities (Isotonic Regression) …")

raw_val_probs  = booster.predict(dval)
calibrator     = IsotonicRegression(out_of_bounds="clip")
calibrator.fit(raw_val_probs, y_val)

raw_test_probs = booster.predict(dtest)
cal_test_probs = calibrator.transform(raw_test_probs)

print(f"      Raw  prob range : [{raw_test_probs.min():.4f}, {raw_test_probs.max():.4f}]")
print(f"      Cal  prob range : [{cal_test_probs.min():.4f}, {cal_test_probs.max():.4f}]")

# ── 5. Threshold tuning ───────────────────────────────────────────────────────
print("\n[5/7] Finding optimal classification threshold …")

raw_val_cal = calibrator.transform(raw_val_probs)
threshold_rows = []
for tau in np.arange(0.01, 1.00, 0.01):
    tau = round(float(tau), 2)
    preds = (raw_val_cal >= tau).astype(int)
    threshold_rows.append({
        "threshold": tau,
        "attack_f1": f1_score(y_val, preds, pos_label=1, zero_division=0),
        "benign_f1": f1_score(y_val, preds, pos_label=0, zero_division=0),
        "macro_f1": f1_score(y_val, preds, average="macro", zero_division=0),
    })

attack_optimal = max(threshold_rows, key=lambda row: row["attack_f1"])
macro_optimal = max(threshold_rows, key=lambda row: row["macro_f1"])
best_thresh = macro_optimal["threshold"]
best_f1 = macro_optimal["attack_f1"]

print("\n      Threshold comparison (validation set):")
print("      " + "-" * 62)
print(f"      {'Selection':<22} {'Threshold':<10} {'Attack F1':<12} {'Benign F1':<12} {'Macro F1':<10}")
print("      " + "-" * 62)
for label, row in [
    ("Attack-optimal (original)", attack_optimal),
    ("Macro-optimal (selected)", macro_optimal),
]:
    print(
        f"      {label:<22} {row['threshold']:<10.2f} "
        f"{row['attack_f1']:<12.4f} {row['benign_f1']:<12.4f} "
        f"{row['macro_f1']:<10.4f}"
    )
print("      " + "-" * 62)
print(f"      Selected threshold : {best_thresh} (validation macro F1 = {macro_optimal['macro_f1']:.4f})")

# ── 6. Test-set evaluation ────────────────────────────────────────────────────
print("\n[6/7] Evaluating on test set (HONEST results) …")

y_pred     = (cal_test_probs >= best_thresh).astype(int)
y_prob     = cal_test_probs

auc_roc    = roc_auc_score(y_test, y_prob)
auc_pr     = average_precision_score(y_test, y_prob)
f1_test    = f1_score(y_test, y_pred, zero_division=0)
cm         = confusion_matrix(y_test, y_pred)
report_str = classification_report(y_test, y_pred, target_names=["Benign", "Attack"])

print("\n      Classification Report:")
print("      " + report_str.replace("\n", "\n      "))
print(f"      ROC-AUC  : {auc_roc:.4f}")
print(f"      PR-AUC   : {auc_pr:.4f}")
print(f"      F1       : {f1_test:.4f}")
print(f"      Threshold: {best_thresh}")

# Cross-validation vs test-set consistency check
print("\n      Cross-Validation vs Test-Set Consistency:")
print(f"        CV  ROC-AUC : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
print(f"        Test ROC-AUC: {auc_roc:.4f}  {'✅ within CV range' if abs(auc_roc - cv_auc.mean()) <= 2*cv_auc.std() else '⚠️ outside CV range — check for data leakage'}")
print(f"        CV  F1      : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
print(f"        Test F1     : {f1_test:.4f}  {'✅ within CV range' if abs(f1_test - cv_f1.mean()) <= 2*cv_f1.std() else '⚠️ outside CV range'}")

if auc_roc > 0.99:
    print("\n      [NOTE] AUC > 0.99 — verify no data leakage in preprocessing.")

# Save classification report
report_path = RESULTS_DIR / "classification_report.txt"
with open(report_path, "w") as f:
    f.write("CausalShield — Test Set Evaluation\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Features used        : {len(selected)}\n")
    f.write(f"Threshold            : {best_thresh}\n")
    f.write("\nValidation threshold comparison\n")
    f.write("-" * 50 + "\n")
    f.write(f"{'Selection':<28} {'Threshold':>9} {'Attack F1':>11} {'Benign F1':>11} {'Macro F1':>10}\n")
    for label, row in [
        ("Attack-optimal (original)", attack_optimal),
        ("Macro-optimal (selected)", macro_optimal),
    ]:
        f.write(
            f"{label:<28} {row['threshold']:>9.2f} {row['attack_f1']:>11.4f} "
            f"{row['benign_f1']:>11.4f} {row['macro_f1']:>10.4f}\n"
        )
    f.write("\nTest confusion matrix\n")
    f.write("-" * 50 + "\n")
    f.write("              Predicted Benign  Predicted Attack\n")
    f.write(f"Actual Benign  {cm[0, 0]:>16}  {cm[0, 1]:>16}\n")
    f.write(f"Actual Attack  {cm[1, 0]:>16}  {cm[1, 1]:>16}\n\n")
    f.write(f"ROC-AUC (test)       : {auc_roc:.4f}\n")
    f.write(f"PR-AUC  (test)       : {auc_pr:.4f}\n")
    f.write(f"F1      (test)       : {f1_test:.4f}\n\n")
    f.write(f"5-Fold CV ROC-AUC    : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}\n")
    f.write(f"5-Fold CV F1         : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}\n")
    f.write(f"95% CI AUC           : {cv_auc.mean():.4f} ± {cv_summary['ci95_auc_roc']:.4f}\n")
    f.write(f"95% CI F1            : {cv_f1.mean():.4f} ± {cv_summary['ci95_f1']:.4f}\n\n")
    f.write(report_str)
print(f"\n      classification_report.txt → {RESULTS_DIR}")

# ── 7. Save model artifacts ───────────────────────────────────────────────────
print("\n[7/7] Saving model artifacts …")

booster.save_model(str(MODEL_DIR / "xgb_model.json"))

with open(MODEL_DIR / "calibrator.pkl", "wb") as f:
    pickle.dump(calibrator, f)

model_meta = {
    "selected_features"  : selected,
    "n_features"         : len(selected),
    "best_round"         : best_round,
    "threshold"          : best_thresh,
    "val_auc"            : round(max(evals_result["val"]["auc"]), 4),
    "test_roc_auc"       : round(auc_roc, 4),
    "test_pr_auc"        : round(auc_pr,  4),
    "test_f1"            : round(f1_test, 4),
    "scale_pos_weight"   : round(spw, 4),
    "cv_mean_auc"        : round(float(cv_auc.mean()), 4),
    "cv_std_auc"         : round(float(cv_auc.std()),  4),
    "cv_ci95_auc"        : round(float(cv_summary["ci95_auc_roc"]), 4),
    "cv_mean_f1"         : round(float(cv_f1.mean()),  4),
    "cv_std_f1"          : round(float(cv_f1.std()),   4),
    "cv_ci95_f1"         : round(float(cv_summary["ci95_f1"]), 4),
    "xgb_params"         : params,
}
with open(MODEL_DIR / "model_meta.json", "w") as f:
    json.dump(model_meta, f, indent=2)

print(f"      xgb_model.json    → {MODEL_DIR}")
print(f"      calibrator.pkl    → {MODEL_DIR}")
print(f"      model_meta.json   → {MODEL_DIR}")

# ── Visualisations ─────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#e74c3c", lw=2,
            label=f"XGBoost (AUC = {auc_roc:.4f})\nCV AUC = {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — CausalShield")
    ax.legend(loc="lower right"); plt.tight_layout()
    fig.savefig(RESULTS_DIR / "roc_curve.png", dpi=150); plt.close()

    # PR curve
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec, prec, color="#3498db", lw=2, label=f"XGBoost (PR-AUC = {auc_pr:.4f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — CausalShield")
    ax.legend(); plt.tight_layout()
    fig.savefig(RESULTS_DIR / "pr_curve.png", dpi=150); plt.close()

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign", "Attack"]); ax.set_yticklabels(["Benign", "Attack"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix  (threshold={best_thresh})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    color="white" if cm[i,j] > cm.max() / 2 else "black", fontsize=14)
    plt.colorbar(im, ax=ax); plt.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150); plt.close()

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, "o-", color="#e74c3c", label="CausalShield")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("True fraction positives")
    ax.set_title("Calibration Curve — CausalShield")
    ax.legend(); plt.tight_layout()
    fig.savefig(RESULTS_DIR / "calibration_curve.png", dpi=150); plt.close()

    # Training history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(evals_result["train"]["logloss"], label="Train")
    axes[0].plot(evals_result["val"]["logloss"],   label="Val")
    axes[0].axvline(best_round, color="red", linestyle="--", label=f"Best ({best_round})")
    axes[0].set_title("Log Loss"); axes[0].legend()
    axes[1].plot(evals_result["train"]["auc"], label="Train")
    axes[1].plot(evals_result["val"]["auc"],   label="Val")
    axes[1].axvline(best_round, color="red", linestyle="--")
    axes[1].set_title("AUC"); axes[1].legend()
    plt.suptitle("XGBoost Training History — CausalShield", fontweight="bold")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "training_history.png", dpi=150); plt.close()

    print(f"      Plots saved → {RESULTS_DIR}")

except Exception as viz_err:
    print(f"      [WARN] Visualisation skipped: {viz_err}")

# ── Final Summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Phase 4 complete.")
print(f"  Features          : {len(selected)}")
print(f"  Best round        : {best_round}")
print(f"  Threshold         : {best_thresh}")
print(f"  Test  ROC-AUC     : {auc_roc:.4f}")
print(f"  Test  PR-AUC      : {auc_pr:.4f}")
print(f"  Test  F1          : {f1_test:.4f}")
print(f"  CV    ROC-AUC     : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}  "
      f"(95% CI: ±{cv_summary['ci95_auc_roc']:.4f})")
print(f"  CV    F1          : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}  "
      f"(95% CI: ±{cv_summary['ci95_f1']:.4f})")
print("=" * 65)