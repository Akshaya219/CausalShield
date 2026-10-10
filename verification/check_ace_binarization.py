import json
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")
CAUSAL_DIR = Path("data/causal")
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

print(f"sample_rows={len(Xs)} seed={SEED}")
for feature in FEATURES:
    values = Xs[feature].values.astype(float)
    median = np.median(values)
    x_bin = values >= median
    all_integers = bool(np.all(values == np.floor(values)))
    print(
        f"{feature}: median={median:.8f} min={values.min():.8f} "
        f"max={values.max():.8f} unique={np.unique(values).size} "
        f"all_integers={all_integers} share_x_bin_1={x_bin.mean():.8f}"
    )

rst_values = Xs["rst_count"].values.astype(float)
rst_median_bin = rst_values >= np.median(rst_values)
rst_positive_bin = rst_values > 0
print(f"rst_count median_vs_positive_diff={np.count_nonzero(rst_median_bin != rst_positive_bin)}")

binary_features = ["HTTPS", "ack_flag_number", "psh_flag_number", "rst_flag_number"]
binary_columns = {
    feature: Xs[feature].values.astype(float) >= np.median(Xs[feature].values.astype(float))
    for feature in binary_features
}
for index, left in enumerate(binary_features):
    for right in binary_features[index + 1:]:
        difference = np.count_nonzero(binary_columns[left] != binary_columns[right])
        print(f"{left} vs {right} binarized_diff={difference}")

rst_gt_zero = rst_values > 0
p1 = ys[rst_gt_zero].mean() if rst_gt_zero.sum() > 0 else 0.0
p0 = ys[~rst_gt_zero].mean() if (~rst_gt_zero).sum() > 0 else 0.0
rst_positive_ace = float(p1 - p0)
print(f"rst_count ACE x_gt_0={rst_positive_ace:.8f} stored=-0.31217100")

meta = json.load(open(CAUSAL_DIR / "causal_graph_meta.json"))
print("Attack-incident edges:")
for edge in meta["edges"]:
    if edge["from"] == "Attack" or edge["to"] == "Attack":
        print(
            f"from={edge['from']} to={edge['to']} "
            f"adj_ij={edge['adj_ij']} adj_ji={edge['adj_ji']}"
        )
