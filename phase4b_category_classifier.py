"""
CausalShield - Phase 4b: Second-Stage Attack Category Classifier
================================================================
Trains a multi-class XGBoost on the same causal features to identify
which of the 7 attack categories a flagged flow belongs to.

Pipeline:
  Binary classifier (Phase 4) -> flags flow as Attack
  Category classifier (this)  -> identifies DDoS / DoS / Mirai /
                                  Reconnaissance / BruteForce /
                                  Spoofing / WebBased

Only attack flows are passed to this classifier in production.
Trained on cat_train.csv (7-category labels from Phase 1).

Outputs:
  - models/cat_model.json         (XGBoost multi-class booster)
  - models/cat_label_map.json     (int -> category name)
  - models/cat_meta.json          (metrics per class)
  - results/cat_confusion.png
  - results/cat_report.txt
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

PROCESSED_DIR = Path("data/processed")
CAUSAL_DIR    = Path("data/causal")
MODEL_DIR     = Path("models")
RESULTS_DIR   = Path("results")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("CausalShield Phase 4b -- Attack Category Classifier")
print("=" * 60)

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("\n[1/5] Loading data ...")

selected = json.load(open(CAUSAL_DIR / "selected_features.json"))

# Category labels (7-class) saved by Phase 1
cat_train = pd.read_csv(PROCESSED_DIR / "cat_train.csv").squeeze()
cat_val   = pd.read_csv(PROCESSED_DIR / "cat_val.csv").squeeze()
cat_test  = pd.read_csv(PROCESSED_DIR / "cat_test.csv").squeeze()

X_train   = pd.read_csv(PROCESSED_DIR / "X_train_causal.csv")[selected]
X_val     = pd.read_csv(PROCESSED_DIR / "X_val_causal.csv")[selected]
X_test    = pd.read_csv(PROCESSED_DIR / "X_test_causal.csv")[selected]

y_bin_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze().values
y_bin_val   = pd.read_csv(PROCESSED_DIR / "y_val.csv").squeeze().values
y_bin_test  = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze().values

print(f"      Causal features  : {len(selected)}")
print(f"      Train size       : {len(X_train)}")
print(f"      Category dist    : {cat_train.value_counts().to_dict()}")

# ── 2. Encode category labels ──────────────────────────────────────────────────
print("\n[2/5] Encoding category labels ...")

# Map string categories to integers
all_cats   = sorted(cat_train.unique().tolist())
cat2idx    = {c: i for i, c in enumerate(all_cats)}
idx2cat    = {i: c for c, i in cat2idx.items()}

y_cat_train = cat_train.map(cat2idx).values
y_cat_val   = cat_val.map(cat2idx).values
y_cat_test  = cat_test.map(cat2idx).values

print(f"      Categories ({len(all_cats)}) : {all_cats}")

# ── 3. Train on attack flows only ─────────────────────────────────────────────
print("\n[3/5] Training category classifier on attack flows ...")

# In production only attack flows are passed here
# Training: use all labelled flows (including benign=label 0 maps to
# the category "Benign" if present, otherwise filter)
# Filter to attack-only rows for training
attack_mask_tr = y_bin_train == 1
attack_mask_va = y_bin_val   == 1
attack_mask_te = y_bin_test  == 1

X_cat_tr = X_train[attack_mask_tr]
y_cat_tr = y_cat_train[attack_mask_tr]
X_cat_va = X_val[attack_mask_va]
y_cat_va = y_cat_val[attack_mask_va]
X_cat_te = X_test[attack_mask_te]
y_cat_te = y_cat_test[attack_mask_te]

print(f"      Attack train size : {len(X_cat_tr)}")
print(f"      Attack val size   : {len(X_cat_va)}")
print(f"      Attack test size  : {len(X_cat_te)}")

import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix

# Re-encode labels to be 0-based contiguous (in case benign removed)
unique_train_labels = np.unique(y_cat_tr)
reenc  = {old: new for new, old in enumerate(unique_train_labels)}
re_idx = {new: idx2cat[old] for old, new in reenc.items()}

y_tr_enc = np.array([reenc[y] for y in y_cat_tr])
y_va_enc = np.array([reenc.get(y, -1) for y in y_cat_va])
y_te_enc = np.array([reenc.get(y, -1) for y in y_cat_te])

# Filter out any val/test labels not seen in train
va_mask = y_va_enc >= 0
te_mask = y_te_enc >= 0
X_cat_va = X_cat_va[va_mask]; y_va_enc = y_va_enc[va_mask]
X_cat_te = X_cat_te[te_mask]; y_te_enc = y_te_enc[te_mask]

n_classes = len(unique_train_labels)
print(f"      Classes in train  : {n_classes}")

dtr = xgb.DMatrix(X_cat_tr, label=y_tr_enc, feature_names=selected)
dva = xgb.DMatrix(X_cat_va, label=y_va_enc, feature_names=selected)
dte = xgb.DMatrix(X_cat_te, label=y_te_enc, feature_names=selected)

params = {
    "objective"        : "multi:softprob",
    "num_class"        : n_classes,
    "eval_metric"      : ["mlogloss", "merror"],
    "eta"              : 0.05,
    "max_depth"        : 6,
    "min_child_weight" : 3,
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "seed"             : 42,
    "tree_method"      : "hist",
}

evals_result = {}
cat_booster  = xgb.train(
    params,
    dtr,
    num_boost_round       = 400,
    evals                 = [(dtr, "train"), (dva, "val")],
    evals_result          = evals_result,
    early_stopping_rounds = 30,
    verbose_eval          = 50,
)

best_round = cat_booster.best_iteration
print(f"\n      Best round : {best_round}")

# ── 4. Evaluate ────────────────────────────────────────────────────────────────
print("\n[4/5] Evaluating on test set ...")

probs_te  = cat_booster.predict(dte).reshape(-1, n_classes)
y_pred_te = np.argmax(probs_te, axis=1)

class_names = [re_idx[i] for i in range(n_classes)]
report      = classification_report(y_te_enc, y_pred_te, target_names=class_names)
cm          = confusion_matrix(y_te_enc, y_pred_te)

print("\n      Category Classification Report:")
print("      " + report.replace("\n", "\n      "))

# Per-class F1
from sklearn.metrics import f1_score
f1_per_class = f1_score(y_te_enc, y_pred_te, average=None)
macro_f1     = f1_score(y_te_enc, y_pred_te, average="macro")
print(f"      Macro F1 (7-class) : {macro_f1:.4f}")
# 5-class Macro F1 (excluding BruteForce and WebBased — too few samples)
tiny_classes = {"BruteForce", "WebBased"}
mask_5 = np.array([re_idx[int(y)] not in tiny_classes for y in y_te_enc])
if mask_5.sum() > 0:
    macro_f1_5 = f1_score(y_te_enc[mask_5], y_pred_te[mask_5], average="macro")
    print(f"      5-class Macro F1 (excl. BruteForce/WebBased) : {macro_f1_5:.4f}")
    bf_idx  = reenc.get(cat2idx.get("BruteForce", -1), -1)
    wb_idx  = reenc.get(cat2idx.get("WebBased",   -1), -1)
    n_bf    = int((y_te_enc == bf_idx).sum()) if bf_idx >= 0 else 0
    n_wb    = int((y_te_enc == wb_idx).sum()) if wb_idx >= 0 else 0
    print(f"      [Note] BruteForce={n_bf} samples, WebBased={n_wb} samples in test — data limitation")

# Save report
with open(RESULTS_DIR / "cat_report.txt", "w") as f:
    f.write("CausalShield -- Attack Category Classifier\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Causal features used : {len(selected)}\n")
    f.write(f"Categories           : {class_names}\n")
    f.write(f"Macro F1             : {macro_f1:.4f}\n\n")
    f.write(report)

# ── 5. Save model ──────────────────────────────────────────────────────────────
print("\n[5/5] Saving category model ...")

cat_booster.save_model(str(MODEL_DIR / "cat_model.json"))

cat_meta = {
    "selected_features" : selected,
    "n_features"        : len(selected),
    "categories"        : class_names,
    "n_classes"         : n_classes,
    "label_map"         : re_idx,
    "best_round"        : best_round,
    "macro_f1"          : round(macro_f1, 4),
    "f1_per_class"      : {class_names[i]: round(float(f1_per_class[i]), 4)
                           for i in range(n_classes)},
}
json.dump(cat_meta,  open(MODEL_DIR / "cat_meta.json",      "w"), indent=2)
json.dump(re_idx,    open(MODEL_DIR / "cat_label_map.json", "w"), indent=2)

print(f"      cat_model.json     -> {MODEL_DIR}")
print(f"      cat_meta.json      -> {MODEL_DIR}")
print(f"      cat_label_map.json -> {MODEL_DIR}")
print(f"      cat_report.txt     -> {RESULTS_DIR}")

# ── Confusion matrix plot ─────────────────────────────────────────────────────
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True",      fontsize=11)
    ax.set_title("CausalShield -- Attack Category Confusion Matrix", fontsize=12, fontweight="bold")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    fig.savefig(RESULTS_DIR / "cat_confusion.png", dpi=150); plt.close()
    print(f"      cat_confusion.png  -> {RESULTS_DIR}")
except Exception as e:
    print(f"      [WARN] Plot skipped: {e}")

print("\n" + "=" * 60)
print("Phase 4b complete -- Category Classifier")
print(f"  Categories : {class_names}")
# print(f"  Macro F1   : {macro_f1:.4f}")
if 'macro_f1_5' in dir():
    print(f"  Macro F1 (7-class) : {macro_f1:.4f}")
    print(f"  5-class F1         : {macro_f1_5:.4f}  (excl. BruteForce/WebBased)")
else:
    print(f"  Macro F1 (7-class) : {macro_f1:.4f}")
print(f"  Per-class  :")
for c, f in zip(class_names, f1_per_class):
    print(f"    {c:<20} F1={f:.4f}")
print("=" * 60)
