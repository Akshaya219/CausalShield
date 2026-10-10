from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "xgb_model.json"
DATA_DIR = ROOT / "data" / "cross_val"
OUT_DIR = ROOT / "results" / "transfer_repro"
OUT_PATH = OUT_DIR / "transfer_reproduction.txt"


def compute_threshold_metrics(y_true, scores, threshold):
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "attack_f1": float(f1_score(y_true, predictions, pos_label=1, zero_division=0)),
        "benign_f1": float(f1_score(y_true, predictions, pos_label=0, zero_division=0)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "tpr": float(tp / (tp + fn)) if (tp + fn) else 0.0,
    }


def find_tau_star(scores, labels):
    best_tau = 0.5
    best_macro_f1 = -1.0
    for tau in np.linspace(0.001, 0.999, 1000):
        predictions = (scores >= tau).astype(int)
        macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
        if macro_f1 > best_macro_f1:
            best_macro_f1 = float(macro_f1)
            best_tau = float(tau)
    return best_tau, best_macro_f1


def find_tau_fpr05(scores, labels):
    best_tau = 0.5
    best_tpr = -1.0
    best_fpr = 1.0
    for tau in np.linspace(0.001, 0.999, 1000):
        predictions = (scores >= tau).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        if fpr <= 0.05 and tpr > best_tpr:
            best_tau = float(tau)
            best_tpr = float(tpr)
            best_fpr = float(fpr)
    if best_tpr < 0:
        for tau in np.linspace(0.001, 0.999, 1000):
            predictions = (scores >= tau).astype(int)
            tn, fp, _, _ = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            if fpr < best_fpr:
                best_tau = float(tau)
                best_fpr = float(fpr)
    return best_tau, best_fpr


def per_attack_metrics(scores, labels, attack_labels, threshold):
    rows = []
    for attack_type in sorted(set(attack_labels)):
        mask = attack_labels == attack_type
        y_group = labels[mask]
        scores_group = scores[mask]
        predictions = (scores_group >= threshold).astype(int)
        n_flows = int(mask.sum())
        
        is_benign = (str(attack_type).lower() == "benign")
        if is_benign:
            rate = float(np.mean(predictions == 0))
            metric_name = "specificity"
        else:
            rate = float(np.mean(predictions == 1))
            metric_name = "recall"
            
        paper_f1 = float((2 * rate) / (1 + rate)) if (1 + rate) > 0 else 0.0
        rows.append({
            "attack_type": str(attack_type),
            "n_flows": n_flows,
            "metric_name": metric_name,
            "rate": rate,
            "paper_f1": paper_f1,
        })
    return rows


lines = []

def emit(message=""):
    print(message, flush=True)
    lines.append(message)


X = pd.read_csv(DATA_DIR / "X_toniot_cross.csv")
y = pd.read_csv(DATA_DIR / "y_toniot_cross.csv").values.ravel()
attack_labels = pd.read_csv(DATA_DIR / "labels_toniot_raw.csv").values.ravel()
model = xgb.XGBClassifier()
model.load_model(str(MODEL_PATH))

X_cal, X_eval, y_cal, y_eval, attack_cal, attack_eval = train_test_split(
    X,
    y,
    attack_labels,
    train_size=0.01,
    random_state=42,
    stratify=y,
)


def evaluate_condition(label, X_cal_condition, X_eval_condition):
    raw_cal_scores = model.predict_proba(X_cal_condition)[:, 1]
    raw_eval_scores = model.predict_proba(X_eval_condition)[:, 1]
    raw_cal_auc = roc_auc_score(y_cal, raw_cal_scores)
    polarity_reflected = raw_cal_auc < 0.5
    cal_scores = 1.0 - raw_cal_scores if polarity_reflected else raw_cal_scores
    eval_scores = 1.0 - raw_eval_scores if polarity_reflected else raw_eval_scores
    tau_star, _ = find_tau_star(cal_scores, y_cal)
    enterprise_tau, _, = find_tau_fpr05(cal_scores, y_cal)
    tau_metrics = compute_threshold_metrics(y_eval, eval_scores, tau_star)
    enterprise_metrics = compute_threshold_metrics(y_eval, eval_scores, enterprise_tau)
    emit(f"[{label}]")
    emit(f"n_cal={len(y_cal)} n_heldout={len(y_eval)} polarity_from_calibration_labels={polarity_reflected}")
    emit(f"raw_roc_auc={roc_auc_score(y_eval, raw_eval_scores):.8f}")
    emit(f"reflected_roc_auc={roc_auc_score(y_eval, eval_scores):.8f}")
    emit(f"heldout_pr_auc={average_precision_score(y_eval, eval_scores):.8f}")
    emit(f"tau_star={tau_star:.8f} attack_f1={tau_metrics['attack_f1']:.8f} benign_f1={tau_metrics['benign_f1']:.8f} macro_f1={tau_metrics['macro_f1']:.8f} fpr={tau_metrics['fpr']:.8f}")
    emit(f"enterprise_threshold={enterprise_tau:.8f} attack_f1={enterprise_metrics['attack_f1']:.8f} benign_f1={enterprise_metrics['benign_f1']:.8f} macro_f1={enterprise_metrics['macro_f1']:.8f} fpr={enterprise_metrics['fpr']:.8f}")
    return eval_scores, tau_star


emit(f"n_total={len(y)} n_cal={len(y_cal)} n_heldout={len(y_eval)}")
conditions = {
    "as_is": (X_cal.copy(), X_eval.copy()),
    "duration_nan": (X_cal.copy(), X_eval.copy()),
    "duration_variance_nan": (X_cal.copy(), X_eval.copy()),
    "https_duration_variance_nan": (X_cal.copy(), X_eval.copy()),
}
conditions["duration_nan"][0]["Duration"] = np.nan
conditions["duration_nan"][1]["Duration"] = np.nan
for frame in conditions["duration_variance_nan"]:
    frame[["Duration", "Variance"]] = np.nan
for frame in conditions["https_duration_variance_nan"]:
    frame[["HTTPS", "Duration", "Variance"]] = np.nan

# XGBoost routes NaN values through the learned default branch.
condition_results = {}
for label, (X_cal_condition, X_eval_condition) in conditions.items():
    condition_results[label] = evaluate_condition(label, X_cal_condition, X_eval_condition)

as_is_scores, as_is_tau_star = condition_results["as_is"]
emit(f"[Table X per-attack-category metrics on held-out split; predictions thresholded at tau_star={as_is_tau_star:.8f}]")
emit(f"Held-out label totals: Benign={(y_eval == 0).sum()}, Attack={(y_eval == 1).sum()}, Total={len(y_eval)}")
n_sum = sum((attack_eval == at).sum() for at in sorted(set(attack_eval)))
emit(f"Per-category n sum: {n_sum} (equals n_heldout: {n_sum == len(y_eval)})")

for item in per_attack_metrics(as_is_scores, y_eval, attack_eval, as_is_tau_star):
    emit(
        f"{item['attack_type']}: n={item['n_flows']} "
        f"{item['metric_name']}={item['rate']:.8f} "
        f"paper-style 2r/(1+r), precision assumed 1, comparison only={item['paper_f1']:.8f}"
    )

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote={OUT_PATH}")
