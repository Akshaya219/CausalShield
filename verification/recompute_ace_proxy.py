import json
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")
CAUSAL_DIR = Path("data/causal")
RESULTS_DIR = Path("results")
SAMPLE_SIZE = 20_000
SEED = 42
FEATURES = [
    "HTTPS",
    "ack_flag_number",
    "psh_flag_number",
    "rst_flag_number",
    "rst_count",
    "Variance",
    "Duration",
]

X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze().values
feature_names = json.load(open(CAUSAL_DIR / "causal_feature_names.json"))
available = [feature for feature in feature_names if feature in X_train.columns]
X_train_c = X_train[available].copy()

if len(X_train_c) > SAMPLE_SIZE:
    idx = np.random.RandomState(SEED).choice(len(X_train_c), SAMPLE_SIZE, replace=False)
    Xs = X_train_c.iloc[idx].reset_index(drop=True)
    ys = y_train[idx]
else:
    Xs = X_train_c.reset_index(drop=True)
    ys = y_train.copy()

with open(CAUSAL_DIR / "ace_scores.json", encoding="utf-8") as handle:
    stored = json.load(handle)

lines = [
    f"sample_rows={len(Xs)} seed={SEED}",
    f"mean_y_full={y_train.mean():.8f}",
    f"mean_y_sample={ys.mean():.8f}",
]

for feature in FEATURES:
    values = Xs[feature].values.astype(float)
    if feature in {"HTTPS", "ack_flag_number", "psh_flag_number", "rst_flag_number", "rst_count"}:
        x_bin = values > 0
        rule = "x>0"
    else:
        threshold = np.median(values)
        x_bin = values >= threshold
        rule = f"x>=median({threshold:.8f})"

    group_one = x_bin == 1
    group_zero = x_bin == 0
    p1 = ys[group_one].mean() if group_one.sum() > 0 else 0.0
    p0 = ys[group_zero].mean() if group_zero.sum() > 0 else 0.0
    ace = float(p1 - p0)
    passes = abs(ace) >= 0.05
    lines.append(
        f"{feature}: rule={rule} old={stored[feature]['ace']:.8f} "
        f"new={ace:.8f} group0={int(group_zero.sum())} group1={int(group_one.sum())} "
        f"passes_abs_ace_0.05={passes}"
    )

output = "\n".join(lines)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
with open(RESULTS_DIR / "ace_proxy_recomputed.txt", "w", encoding="utf-8") as handle:
    handle.write(output + "\n")
print(output)
