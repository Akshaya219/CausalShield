"""
CausalShield - Phase 3: Do-Calculus ACE Estimation & Feature Filtering
=======================================================================
Estimates Average Causal Effect (ACE) of each candidate feature on the
binary Attack target using the backdoor adjustment formula.

ACE(X_i → Y) = E[Y | do(X_i = 1)] − E[Y | do(X_i = 0)]

For continuous features we discretise into binary (median split) before
applying the adjustment.  Confounders are identified as common parents
in the PC graph (nodes pointing into both X_i and Y).

Selection rule:
  - Keep feature if |ACE| >= ACE_THRESHOLD  (default 0.05)
  - Fallback: if fewer than MIN_FEATURES pass, keep top-k by |ACE|

Outputs:
  - data/causal/ace_scores.json          (all scores)
  - data/causal/selected_features.json   (final feature list)
  - data/causal/ace_scores.png           (bar chart)
  - data/processed/X_train_causal.csv    (training set, causal features only)
  - data/processed/X_val_causal.csv
  - data/processed/X_test_causal.csv
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ── config ────────────────────────────────────────────────────────────────────
ACE_THRESHOLD = 0.05   # minimum |ACE| to keep a feature
MIN_FEATURES  = 8      # guarantee at least this many features
MAX_FEATURES  = 30     # hard cap (for efficiency)

PROCESSED_DIR = Path("data/processed")
CAUSAL_DIR    = Path("data/causal")

# ── 1. Load causal graph outputs ──────────────────────────────────────────────
print("=" * 60)
print("CausalShield Phase 3 — ACE Estimation & Feature Filtering")
print("=" * 60)

print("\n[1/6] Loading causal graph metadata …")

adj           = np.load(CAUSAL_DIR / "causal_adjacency.npy")
feature_names = json.load(open(CAUSAL_DIR / "causal_feature_names.json"))
causal_feats  = json.load(open(CAUSAL_DIR / "causal_target_features.json"))
meta          = json.load(open(CAUSAL_DIR / "causal_graph_meta.json"))

all_names   = feature_names + ["Attack"]
TARGET_IDX  = len(feature_names)
n_nodes     = len(all_names)

print(f"      Features in clean graph : {len(feature_names)}")
print(f"      FCI-linked to target     : {len(causal_feats)}")

# ── 2. Load data ──────────────────────────────────────────────────────────────
print("\n[2/6] Loading preprocessed data …")
X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze().values

X_val   = pd.read_csv(PROCESSED_DIR / "X_val.csv")
y_val   = pd.read_csv(PROCESSED_DIR / "y_val.csv").squeeze().values

X_test  = pd.read_csv(PROCESSED_DIR / "X_test.csv")
y_test  = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze().values

print(f"      Train : {X_train.shape}  |  Val : {X_val.shape}  |  Test : {X_test.shape}")

# Keep only features that survived the PC cleaning step
available = [c for c in feature_names if c in X_train.columns]
X_train_c = X_train[available].copy()

print(f"      Features available in training data : {len(available)}")

# ── 3. Helper: identify backdoor adjustment set ────────────────────────────────
def get_parents(node_idx, adj_matrix):
    """
    Return indices of nodes with a directed edge INTO node_idx.
    Convention: adj[i,j]=1 and adj[j,i]=-1  →  i → j  (j has parent i)
    """
    parents = []
    n = adj_matrix.shape[0]
    for i in range(n):
        if i == node_idx:
            continue
        if adj_matrix[i, node_idx] == 1 and adj_matrix[node_idx, i] == -1:
            parents.append(i)
    return parents

def get_adjustment_set(feature_idx, target_idx, adj_matrix, all_node_names):
    """
    Backdoor adjustment set = parents of feature_idx that are NOT
    descendants of feature_idx (simple heuristic for PC output).
    We use parents(feature) ∩ parents(target) as confounders.
    """
    parents_feat   = set(get_parents(feature_idx, adj_matrix))
    parents_target = set(get_parents(target_idx,  adj_matrix))
    confounders    = parents_feat & parents_target
    return list(confounders)

# ── 4. ACE estimation ─────────────────────────────────────────────────────────
print("\n[3/6] Estimating ACE for each candidate feature …")

# Use a stratified 20k-sample for speed (full train can be large)
SAMPLE_SIZE = 20_000
if len(X_train_c) > SAMPLE_SIZE:
    idx  = np.random.RandomState(42).choice(len(X_train_c), SAMPLE_SIZE, replace=False)
    Xs   = X_train_c.iloc[idx].reset_index(drop=True)
    ys   = y_train[idx]
else:
    Xs   = X_train_c.reset_index(drop=True)
    ys   = y_train.copy()

print(f"      ACE estimation subset size : {len(Xs)}")

ace_results = {}

# Candidate set: PC-linked features OR all features if none found
candidates = causal_feats if causal_feats else available
print(f"      Candidates : {len(candidates)}")

for feat in candidates:
    if feat not in available:
        continue

    feat_idx = feature_names.index(feat) if feat in feature_names else -1
    if feat_idx == -1:
        continue

    x_vals = Xs[feat].values.astype(float)

    # Binarise: 0 if below median, 1 if at/above median
    median_val = np.median(x_vals)
    x_bin = (x_vals >= median_val).astype(int)

    # Adjustment set (confounders)
    adj_set_idx = get_adjustment_set(feat_idx, TARGET_IDX, adj, all_names)
    adj_set_names = [
        all_names[i] for i in adj_set_idx
        if all_names[i] != "Attack" and all_names[i] in Xs.columns
    ]

    if len(adj_set_names) == 0:
        # No confounders → naive ACE = P(Y=1|X=1) - P(Y=1|X=0)
        mask1 = x_bin == 1
        mask0 = x_bin == 0
        p1 = ys[mask1].mean() if mask1.sum() > 0 else 0.0
        p0 = ys[mask0].mean() if mask0.sum() > 0 else 0.0
        ace = float(p1 - p0)
        method = "naive"
    else:
        # Backdoor adjustment: stratify over confounders (discretised)
        # ACE = Σ_z [P(Y=1|X=1,Z=z) - P(Y=1|X=0,Z=z)] · P(Z=z)
        # Discretise each confounder to median-split binary
        Z_df = pd.DataFrame()
        for z_col in adj_set_names[:4]:   # cap at 4 confounders
            z_vals = Xs[z_col].values.astype(float)
            Z_df[z_col] = (z_vals >= np.median(z_vals)).astype(int)

        # Encode confounder combination as tuple
        z_combos = [tuple(row) for row in Z_df.values]
        unique_z = list(set(z_combos))
        n_total  = len(Xs)

        ace_weighted = 0.0
        total_weight = 0.0
        for z_val in unique_z:
            mask_z  = np.array([c == z_val for c in z_combos])
            pz      = mask_z.sum() / n_total
            mask_z1 = mask_z & (x_bin == 1)
            mask_z0 = mask_z & (x_bin == 0)
            p1_z = ys[mask_z1].mean() if mask_z1.sum() > 2 else float(ys[mask_z].mean())
            p0_z = ys[mask_z0].mean() if mask_z0.sum() > 2 else float(ys[mask_z].mean())
            ace_weighted += pz * (p1_z - p0_z)
            total_weight += pz

        ace = float(ace_weighted / total_weight) if total_weight > 0 else 0.0
        method = "backdoor"

    ace_results[feat] = {
        "ace"          : round(ace, 6),
        "abs_ace"      : round(abs(ace), 6),
        "method"       : method,
        "confounders"  : adj_set_names,
        "median_split" : round(float(median_val), 6),
    }

# Sort by |ACE| descending
ace_sorted = dict(sorted(ace_results.items(), key=lambda x: x[1]["abs_ace"], reverse=True))

print(f"\n      ACE results (top 20):")
print(f"      {'Feature':<35} {'ACE':>8}  {'|ACE|':>8}  Method")
print(f"      {'-'*35} {'-'*8}  {'-'*8}  {'-'*10}")
for feat, vals in list(ace_sorted.items())[:20]:
    print(f"      {feat:<35} {vals['ace']:>8.4f}  {vals['abs_ace']:>8.4f}  {vals['method']}")

# ── 5. Feature selection ──────────────────────────────────────────────────────
print("\n[4/6] Selecting features by ACE threshold …")

selected = [f for f, v in ace_sorted.items() if v["abs_ace"] >= ACE_THRESHOLD]

# Fallback: guarantee MIN_FEATURES
if len(selected) < MIN_FEATURES:
    all_by_ace = list(ace_sorted.keys())
    selected   = all_by_ace[:max(MIN_FEATURES, len(selected))]
    print(f"      [Fallback] Fewer than {MIN_FEATURES} features passed threshold.")
    print(f"      Keeping top {len(selected)} by |ACE|.")

# Cap at MAX_FEATURES
if len(selected) > MAX_FEATURES:
    selected = selected[:MAX_FEATURES]
    print(f"      [Cap] Trimmed to {MAX_FEATURES} features.")

print(f"\n      Final selected features ({len(selected)}) :")
for f in selected:
    print(f"        {f:<35}  |ACE| = {ace_sorted[f]['abs_ace']:.4f}")

# ── 6. Save outputs ────────────────────────────────────────────────────────────
print("\n[5/6] Saving ACE scores and selected features …")

with open(CAUSAL_DIR / "ace_scores.json", "w") as f:
    json.dump(ace_sorted, f, indent=2)

with open(CAUSAL_DIR / "selected_features.json", "w") as f:
    json.dump(selected, f, indent=2)

print(f"      ace_scores.json       → {CAUSAL_DIR}")
print(f"      selected_features.json → {CAUSAL_DIR}")

# ── 7. Produce causal-feature datasets ────────────────────────────────────────
print("\n[6/6] Writing causal-feature train/val/test CSVs …")

for split_name, X_split in [("train", X_train), ("val", X_val), ("test", X_test)]:
    cols_in_split = [c for c in selected if c in X_split.columns]
    missing       = [c for c in selected if c not in X_split.columns]
    if missing:
        print(f"      [WARN] {split_name}: missing {missing}  — filling with 0")
        for mc in missing:
            X_split[mc] = 0.0
    X_out = X_split[selected].copy()
    out_path = PROCESSED_DIR / f"X_{split_name}_causal.csv"
    X_out.to_csv(out_path, index=False)
    print(f"      X_{split_name}_causal.csv  → {out_path}  {X_out.shape}")

# ── Visualisation ─────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feats   = list(ace_sorted.keys())[:25]
    values  = [ace_sorted[f]["ace"] for f in feats]
    colors  = ["#e74c3c" if v > 0 else "#3498db" for v in values]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(feats[::-1], values[::-1], color=colors[::-1], edgecolor="white")
    ax.axvline(x=ACE_THRESHOLD,  color="green",  linestyle="--", linewidth=1.2, label=f"+threshold ({ACE_THRESHOLD})")
    ax.axvline(x=-ACE_THRESHOLD, color="green",  linestyle="--", linewidth=1.2)
    ax.axvline(x=0,              color="black",  linestyle="-",  linewidth=0.8)
    ax.set_xlabel("Average Causal Effect (ACE)", fontsize=11)
    ax.set_title("CausalShield Phase 3 — ACE Scores (top 25 features)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)

    # Highlight selected
    selected_set = set(selected)
    for bar, feat in zip(bars[::-1], feats[::-1]):
        if feat in selected_set:
            bar.set_edgecolor("#2ecc71")
            bar.set_linewidth(2)

    plt.tight_layout()
    fig.savefig(CAUSAL_DIR / "ace_scores.png", dpi=150)
    plt.close()
    print(f"\n      ace_scores.png  → {CAUSAL_DIR}")
except Exception as viz_err:
    print(f"      [WARN] Visualisation skipped: {viz_err}")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Phase 3 complete.")
print(f"  Candidates evaluated : {len(ace_results)}")
print(f"  ACE threshold        : ±{ACE_THRESHOLD}")
print(f"  Selected features    : {len(selected)}")
print(f"  Feature list         : {selected}")
print("=" * 60)