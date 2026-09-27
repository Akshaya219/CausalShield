import os
import json
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

warnings.filterwarnings("ignore")

MODEL_PATH = "C:/Users/aksha/Documents/v2/models/xgb_model.json"
X_PATH = "data/cross_val/X_toniot_cross.csv"
Y_PATH = "data/cross_val/y_toniot_cross.csv"
ATK_PATH = "data/cross_val/labels_toniot_raw.csv"
OUT_DIR = "outputs/threshold_recalib"
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_PATH = os.path.join("results", "ton_iot_fewshot_transfer.json")
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
# Diagnostic only: target-space normalization is incompatible with frozen tree splits.
RUN_QUANTILE_NORMALIZATION_EXPERIMENT = False
CONTINUOUS_FEATURES = ["Duration", "Variance"]


def compute_threshold_metrics(y_true, scores, threshold):
    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

    attack_f1 = f1_score(y_true, preds, pos_label=1, zero_division=0)
    benign_f1 = f1_score(y_true, preds, pos_label=0, zero_division=0)
    macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "attack_f1": float(attack_f1),
        "benign_f1": float(benign_f1),
        "macro_f1": float(macro_f1),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def find_tau_star(scores, y_true):
    best_tau = 0.5
    best_macro_f1 = -1.0
    for tau in np.linspace(0.001, 0.999, 1000):
        preds = (scores >= tau).astype(int)
        macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
        if macro_f1 > best_macro_f1:
            best_macro_f1 = float(macro_f1)
            best_tau = float(tau)
    return best_tau, best_macro_f1


def find_tau_fpr05(scores, y_true):
    best_tau = 0.5
    best_tpr = -1.0
    best_fpr = 1.0

    for tau in np.linspace(0.001, 0.999, 1000):
        preds = (scores >= tau).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if fpr <= 0.05 and tpr > best_tpr:
            best_tpr = float(tpr)
            best_tau = float(tau)
            best_fpr = float(fpr)

    if best_tpr < 0:
        for tau in np.linspace(0.001, 0.999, 1000):
            preds = (scores >= tau).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            if fpr < best_fpr:
                best_fpr = float(fpr)
                best_tau = float(tau)
                best_tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    return best_tau, best_fpr, best_tpr


def per_attack_metrics(scores, y_true, labels):
    records = []
    for attack_type in sorted(set(labels)):
        mask = labels == attack_type
        if np.sum(mask) == 0:
            continue

        y_mask = y_true[mask]
        s_mask = scores[mask]
        preds = (s_mask >= 0.5).astype(int)
        acc = float(np.mean(preds == y_mask))
        if np.unique(y_mask).size > 1:
            f1_attack = float(f1_score(y_mask, preds, pos_label=1, zero_division=0))
        elif y_mask[0] == 1:
            f1_attack = float(accuracy_score(y_mask, preds))
        else:
            f1_attack = 0.0

        records.append(
            {
                "attack_type": str(attack_type),
                "n_flows": int(mask.sum()),
                "accuracy": round(acc, 6),
                "attack_f1": round(f1_attack, 6),
            }
        )
    return records


def print_metric_block(label, metrics):
    print(f"\n  [{label}]")
    print(f"    Attack F1 : {metrics['attack_f1']:.4f}")
    print(f"    Benign F1 : {metrics['benign_f1']:.4f}")
    print(f"    Macro F1  : {metrics['macro_f1']:.4f}")
    print(f"    FPR       : {metrics['fpr']:.4f}")


print("=" * 80)
print("STEP 1 — Load frozen CICIoT2023 XGBoost model and ToN-IoT data")
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

X = pd.read_csv(X_PATH)
y = pd.read_csv(Y_PATH).values.ravel()
attack_labels = pd.read_csv(ATK_PATH).values.ravel()

print(f"  X shape       : {X.shape}")
print(f"  Benign count  : {(y == 0).sum():,}")
print(f"  Attack count  : {(y == 1).sum():,}")

print("\n" + "=" * 80)
print("STEP 2 — 1% calibration split and 99% held-out evaluation split")
X_cal, X_eval, y_cal, y_eval, attack_cal, attack_eval = train_test_split(
    X,
    y,
    attack_labels,
    train_size=0.01,
    random_state=42,
    stratify=y,
)

print(f"  Calibration size : {len(X_cal)} ({len(X_cal) / len(X) * 100:.2f}%)")
print(f"  Evaluation size  : {len(X_eval)} ({len(X_eval) / len(X) * 100:.2f}%)")
print(f"  Calibration dist : benign={int((y_cal == 0).sum())}, attack={int((y_cal == 1).sum())}")
print(f"  Eval dist        : benign={int((y_eval == 0).sum())}, attack={int((y_eval == 1).sum())}")

print("\n" + "=" * 80)
print("STEP 3 — Calibration-only inversion detection and threshold search")
raw_cal_scores = model.predict_proba(X_cal)[:, 1]
calibration_roc_auc = roc_auc_score(y_cal, raw_cal_scores)
calibration_pr_auc = average_precision_score(y_cal, raw_cal_scores)
score_inversion_flag = calibration_roc_auc < 0.5
cal_scores = 1.0 - raw_cal_scores if score_inversion_flag else raw_cal_scores

tau_star, tau_star_macro_f1 = find_tau_star(cal_scores, y_cal)
tau_fpr05, tau_fpr05_fpr, tau_fpr05_tpr = find_tau_fpr05(cal_scores, y_cal)

print(f"  Raw calibration ROC-AUC : {calibration_roc_auc:.4f}")
print(f"  Raw calibration PR-AUC  : {calibration_pr_auc:.4f}")
print(f"  Score inversion flag    : {score_inversion_flag}")
print(f"  tau_star                : {tau_star:.4f}  (macro-F1={tau_star_macro_f1:.4f})")
print(f"  tau_fpr05               : {tau_fpr05:.4f}  (FPR={tau_fpr05_fpr:.4f}, TPR={tau_fpr05_tpr:.4f})")

calibration_summary = {
    "size": int(len(X_cal)),
    "share_percent": round(len(X_cal) / len(X) * 100, 4),
    "raw_roc_auc": round(float(calibration_roc_auc), 6),
    "raw_pr_auc": round(float(calibration_pr_auc), 6),
    "score_inversion_detected": bool(score_inversion_flag),
    "tau_star": round(float(tau_star), 6),
    "tau_star_macro_f1": round(float(tau_star_macro_f1), 6),
    "tau_fpr05": round(float(tau_fpr05), 6),
    "tau_fpr05_fpr": round(float(tau_fpr05_fpr), 6),
    "tau_fpr05_tpr": round(float(tau_fpr05_tpr), 6),
}

print("\n" + "=" * 80)
print("STEP 4 — Evaluate on the held-out 99% split using calibration-fixed thresholds")
raw_eval_scores = model.predict_proba(X_eval)[:, 1]
reflected_eval_scores = 1.0 - raw_eval_scores if score_inversion_flag else raw_eval_scores

roc_auc_eval = roc_auc_score(y_eval, reflected_eval_scores)
pr_auc_eval = average_precision_score(y_eval, reflected_eval_scores)
print(f"  Reflected ROC-AUC : {roc_auc_eval:.4f}")
print(f"  Reflected PR-AUC  : {pr_auc_eval:.4f}")

metrics_tau_star = compute_threshold_metrics(y_eval, reflected_eval_scores, tau_star)
metrics_tau_fpr05 = compute_threshold_metrics(y_eval, reflected_eval_scores, tau_fpr05)

print_metric_block("tau_star", metrics_tau_star)
print_metric_block("tau_fpr05", metrics_tau_fpr05)

quantile_metrics = None
if RUN_QUANTILE_NORMALIZATION_EXPERIMENT:
    print("\n" + "=" * 80)
    print("OPTIONAL EXPERIMENT — Quantile-normalized continuous features")
    missing_features = [feature for feature in CONTINUOUS_FEATURES if feature not in X_cal.columns]
    if missing_features:
        raise ValueError(f"Missing continuous feature columns: {missing_features}")

    quantile_transformer = QuantileTransformer(
        output_distribution="uniform",
        random_state=42,
    )
    quantile_transformer.fit(X_cal[CONTINUOUS_FEATURES])
    X_eval_quantile = X_eval.copy()
    X_eval_quantile[CONTINUOUS_FEATURES] = quantile_transformer.transform(
        X_eval_quantile[CONTINUOUS_FEATURES]
    )

    raw_quantile_scores = model.predict_proba(X_eval_quantile)[:, 1]
    reflected_quantile_scores = (
        1.0 - raw_quantile_scores if score_inversion_flag else raw_quantile_scores
    )
    quantile_metrics = compute_threshold_metrics(
        y_eval,
        reflected_quantile_scores,
        tau_star,
    )

    print("\n  Before/after comparison at the calibration-fixed tau_star:")
    print("  " + "-" * 58)
    print(f"  {'Inference':<24} {'FPR':<12} {'Macro F1':<12} {'Threshold':<10}")
    print("  " + "-" * 58)
    print(
        f"  {'Original features':<24} {metrics_tau_star['fpr']:<12.4f} "
        f"{metrics_tau_star['macro_f1']:<12.4f} {tau_star:<10.4f}"
    )
    print(
        f"  {'Quantile-normalized':<24} {quantile_metrics['fpr']:<12.4f} "
        f"{quantile_metrics['macro_f1']:<12.4f} {tau_star:<10.4f}"
    )
    print("  " + "-" * 58)
    print(f"  FPR change: {quantile_metrics['fpr'] - metrics_tau_star['fpr']:+.4f}")

per_attack = per_attack_metrics(reflected_eval_scores, y_eval, attack_eval)
print("\n  Per-attack-type performance on held-out split:")
for item in per_attack:
    print(f"    {item['attack_type']:20s}  n={item['n_flows']:>6}  Acc={item['accuracy']:.4f}  F1={item['attack_f1']:.4f}")

results = {
    "strict_split": {
        "calibration_fraction": 0.01,
        "evaluation_fraction": 0.99,
        "random_state": 42,
        "stratified": True,
        "score_inversion_detected": bool(score_inversion_flag),
    },
    "calibration": calibration_summary,
    "evaluation": {
        "roc_auc": round(float(roc_auc_eval), 6),
        "pr_auc": round(float(pr_auc_eval), 6),
        "tau_star": {
            "threshold": round(float(tau_star), 6),
            "metrics": {
                "attack_f1": round(float(metrics_tau_star["attack_f1"]), 6),
                "benign_f1": round(float(metrics_tau_star["benign_f1"]), 6),
                "macro_f1": round(float(metrics_tau_star["macro_f1"]), 6),
                "fpr": round(float(metrics_tau_star["fpr"]), 6),
                "tpr": round(float(metrics_tau_star["tpr"]), 6),
            },
        },
        "tau_fpr05": {
            "threshold": round(float(tau_fpr05), 6),
            "metrics": {
                "attack_f1": round(float(metrics_tau_fpr05["attack_f1"]), 6),
                "benign_f1": round(float(metrics_tau_fpr05["benign_f1"]), 6),
                "macro_f1": round(float(metrics_tau_fpr05["macro_f1"]), 6),
                "fpr": round(float(metrics_tau_fpr05["fpr"]), 6),
                "tpr": round(float(metrics_tau_fpr05["tpr"]), 6),
            },
        },
        "quantile_normalization_experiment": {
            "enabled": bool(RUN_QUANTILE_NORMALIZATION_EXPERIMENT),
            "fit_source": "1% calibration split",
            "features": CONTINUOUS_FEATURES,
            "output_distribution": "uniform",
            "random_state": 42,
            "threshold": round(float(tau_star), 6),
            "metrics": (
                {
                    "fpr": round(float(quantile_metrics["fpr"]), 6),
                    "macro_f1": round(float(quantile_metrics["macro_f1"]), 6),
                }
                if quantile_metrics is not None
                else None
            ),
        },
    },
    "per_attack_type": per_attack,
}

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n  Saved metrics to: {RESULTS_PATH}")

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"  Calibration ROC-AUC / PR-AUC : {calibration_roc_auc:.4f} / {calibration_pr_auc:.4f}")
print(f"  Reflection applied          : {score_inversion_flag}")
print(f"  tau_star                    : {tau_star:.4f}")
print(f"  tau_fpr05                   : {tau_fpr05:.4f}")
print(f"  Held-out ROC-AUC            : {roc_auc_eval:.4f}")
print(f"  Held-out PR-AUC             : {pr_auc_eval:.4f}")
print(f"  Macro F1 @ tau_star         : {metrics_tau_star['macro_f1']:.4f}")
print(f"  FPR @ tau_star              : {metrics_tau_star['fpr']:.4f}")
print(f"  Macro F1 @ tau_fpr05        : {metrics_tau_fpr05['macro_f1']:.4f}")
print(f"  FPR @ tau_fpr05             : {metrics_tau_fpr05['fpr']:.4f}")
print("\n  No data leakage: threshold selection and score inversion detection were performed using only the 1% calibration split.")
