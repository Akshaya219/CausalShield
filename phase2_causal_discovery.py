"""
CausalShield - Phase 2: Causal Graph Discovery via FCI Algorithm
=================================================================
Uses FCI (Fast Causal Inference) instead of PC because FCI:
  - Handles latent confounders (unobserved variables)
  - Outputs a PAG (Partial Ancestral Graph) not a CPDAG
  - Uses o->, ->, o-o edge marks to represent uncertainty
  - More appropriate for network traffic where unobserved
    system-level variables (CPU load, routing state) may
    simultaneously influence multiple features

PAG edge semantics:
  X -> Y  : X is an ancestor of Y (direct cause)
  X <-> Y : hidden common cause (latent confounder)
  X o-> Y : X may or may not be an ancestor of Y
  X o-o Y : relationship uncertain

Robustness fixes:
  - Drop near-constant columns (std < 1e-10)
  - Drop highly correlated columns (|r| > 0.99)
  - Drop duplicate columns
  - Alpha fallback: 0.05 -> 0.10 if zero neighbours found
  - Subset: 5,000 stratified samples for speed

Outputs:
  - data/causal/causal_adjacency.npy
  - data/causal/causal_feature_names.json
  - data/causal/causal_target_features.json
  - data/causal/causal_graph_meta.json
  - data/causal/causal_graph.png
"""

import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

PROCESSED_DIR = Path("data/processed")
CAUSAL_DIR    = Path("data/causal")
CAUSAL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("CausalShield Phase 2 - Causal Discovery (FCI Algorithm)")
print("=" * 60)

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("\n[1/6] Loading preprocessed data ...")
X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()
print(f"      X_train shape : {X_train.shape}")
print(f"      y_train dist  : {y_train.value_counts().to_dict()}")

# ── 2. Stratified subset ───────────────────────────────────────────────────────
print("\n[2/6] Creating stratified subset (5,000 samples) ...")
SUBSET_SIZE = 5_000
df_full = X_train.copy()
df_full["__target__"] = y_train.values

n_per   = SUBSET_SIZE // 2
benign  = df_full[df_full["__target__"] == 0].sample(
            n=min(n_per, (df_full["__target__"] == 0).sum()), random_state=42)
attack  = df_full[df_full["__target__"] == 1].sample(
            n=min(n_per, (df_full["__target__"] == 1).sum()), random_state=42)
subset  = pd.concat([benign, attack]).sample(frac=1, random_state=42).reset_index(drop=True)
y_sub   = subset["__target__"].values
X_sub   = subset.drop(columns=["__target__"])
print(f"      Subset size   : {len(subset)}")
print(f"      Class balance : {dict(zip(*np.unique(y_sub, return_counts=True)))}")

# ── 3. Pre-FCI cleaning ────────────────────────────────────────────────────────
print("\n[3/6] Pre-FCI cleaning ...")
orig_cols  = list(X_sub.columns)

stds       = X_sub.std()
const_cols = stds[stds < 1e-10].index.tolist()
X_sub      = X_sub.drop(columns=const_cols)
print(f"      Dropped near-constant ({len(const_cols)})")

X_sub      = X_sub.T.drop_duplicates().T
dup_drop   = len(orig_cols) - len(const_cols) - X_sub.shape[1]
print(f"      Dropped duplicates  ({dup_drop})")

corr_m     = X_sub.corr().abs()
upper      = corr_m.where(np.triu(np.ones(corr_m.shape), k=1).astype(bool))
high_corr  = [c for c in upper.columns if any(upper[c] > 0.99)]
X_sub      = X_sub.drop(columns=high_corr)
print(f"      Dropped high-corr   ({len(high_corr)})")
print(f"      Features remaining  : {X_sub.shape[1]}")

feature_names = list(X_sub.columns)
TARGET_IDX    = len(feature_names)
data_matrix   = np.hstack([X_sub.values, y_sub.reshape(-1, 1)]).astype(np.float64)
all_names     = feature_names + ["Attack"]
print(f"      Data matrix shape   : {data_matrix.shape}")

# ── 4. Run FCI ─────────────────────────────────────────────────────────────────
print("\n[4/6] Running FCI algorithm ...")
print("      FCI handles latent confounders -> outputs PAG")

try:
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.utils.cit import fisherz
except ImportError:
    raise ImportError("Run: pip install causal-learn")

def run_fci(data, alpha):
    t0 = time.time()
    G, edges = fci(
        data,
        independence_test_method = fisherz,
        alpha                    = alpha,
        depth                    = -1,
        max_path_length          = -1,
        verbose                  = False,
        show_progress            = False,
    )
    return G, edges, time.time() - t0

ALPHA = 0.05
print(f"      Trying alpha = {ALPHA} ...")
G, edges, elapsed = run_fci(data_matrix, ALPHA)
adj = G.graph
target_neighbors = int(np.sum(np.abs(adj[TARGET_IDX, :]) > 0))
print(f"      FCI done in {elapsed:.1f}s  |  target neighbors = {target_neighbors}")

if target_neighbors == 0:
    ALPHA = 0.10
    print(f"      Zero neighbors -- retrying with alpha = {ALPHA} ...")
    G, edges, elapsed = run_fci(data_matrix, ALPHA)
    adj = G.graph
    target_neighbors = int(np.sum(np.abs(adj[TARGET_IDX, :]) > 0))
    print(f"      FCI done in {elapsed:.1f}s  |  target neighbors = {target_neighbors}")

# ── 5. Extract PAG structure ───────────────────────────────────────────────────
print("\n[5/6] Extracting PAG structure ...")

# causal-learn PAG mark conventions:
#   adj[i,j] = -1 : tail at i
#   adj[i,j] =  1 : arrowhead at i's end (G[a,b] is the mark at a's end of the a-b edge)
#   adj[i,j] =  2 : circle mark (uncertainty)
#
# i->j  : adj[i,j]=-1, adj[j,i]=1
# i<->j : adj[i,j]=1,  adj[j,i]=1   (bidirected = latent confounder)
# io->j : adj[i,j]=2,  adj[j,i]=1   (possible ancestor)
# io-oj : adj[i,j]=2,  adj[j,i]=2   (uncertain)

connected = [
    i for i in range(len(all_names))
    if i != TARGET_IDX and (adj[i, TARGET_IDX] != 0 or adj[TARGET_IDX, i] != 0)
]
causal_features = [feature_names[i] for i in connected if i < len(feature_names)]

print(f"      Causal features linked to Attack ({len(causal_features)}):")
for f in causal_features:
    i   = feature_names.index(f)
    aij = adj[i, TARGET_IDX]
    aji = adj[TARGET_IDX, i]
    if   aij == -1 and aji == 1: rel = f"{f} -> Attack  (direct cause)"
    elif aij == 1 and aji == -1: rel = f"Attack -> {f}  (effect)"
    elif aij == 1 and aji == 1:  rel = f"{f} <-> Attack  (latent confounder)"
    elif aij == 2 and aji == 1:  rel = f"{f} o-> Attack  (possible cause)"
    elif aij == 1 and aji == 2:  rel = f"{f} <-o Attack  (possible effect)"
    else:                         rel = f"{f} o-o Attack  (uncertain)"
    print(f"        {rel}")

edge_list = []
n_nodes   = adj.shape[0]
for i in range(n_nodes):
    for j in range(i + 1, n_nodes):
        if adj[i, j] != 0 or adj[j, i] != 0:
            aij = int(adj[i, j])
            aji = int(adj[j, i])
            if   aij == 1 and aji == 1:  etype = "bidirected_latent_confounder"
            elif aij == 1 or aji == 1:   etype = "directed_or_possible_ancestor"
            else:                         etype = "uncertain"
            edge_list.append({
                "from": all_names[i], "to": all_names[j],
                "adj_ij": aij, "adj_ji": aji, "type": etype,
            })

bidirected = [e for e in edge_list if e["type"] == "bidirected_latent_confounder"]
print(f"\n      Total PAG edges          : {len(edge_list)}")
print(f"      Bidirected (latent conf) : {len(bidirected)}")

# ── 6. Save ────────────────────────────────────────────────────────────────────
print("\n[6/6] Saving outputs ...")

np.save(CAUSAL_DIR / "causal_adjacency.npy", adj)
json.dump(feature_names,  open(CAUSAL_DIR / "causal_feature_names.json",   "w"), indent=2)
json.dump(causal_features,open(CAUSAL_DIR / "causal_target_features.json", "w"), indent=2)

meta = {
    "algorithm"                  : "FCI",
    "pag_not_cpdag"              : True,
    "handles_latent_confounders" : True,
    "alpha_used"                 : ALPHA,
    "subset_size"                : int(len(subset)),
    "original_features"          : len(orig_cols),
    "clean_features"             : len(feature_names),
    "dropped_constant"           : const_cols,
    "dropped_high_corr"          : high_corr,
    "n_edges_total"              : len(edge_list),
    "n_bidirected_edges"         : len(bidirected),
    "n_causal_features"          : len(causal_features),
    "causal_features"            : causal_features,
    "edges"                      : edge_list,
    "elapsed_seconds"            : round(elapsed, 2),
}
json.dump(meta, open(CAUSAL_DIR / "causal_graph_meta.json", "w"), indent=2)
print(f"      All files -> {CAUSAL_DIR}")

# ── Visualisation ──────────────────────────────────────────────────────────────
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx

    Gnx = nx.DiGraph()
    Gnx.add_nodes_from(all_names)
    for e in edge_list:
        if e["adj_ij"] == -1 and e["adj_ji"] == 1: Gnx.add_edge(e["from"], e["to"])
        elif e["adj_ji"] == -1 and e["adj_ij"] == 1: Gnx.add_edge(e["to"], e["from"])
        else: Gnx.add_edge(e["from"], e["to"])

    colors = ["#e74c3c" if n == "Attack" else "#f39c12" if n in causal_features else "#95a5a6"
              for n in Gnx.nodes()]
    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(Gnx, seed=42, k=1.8)
    nx.draw_networkx(Gnx, pos, ax=ax, node_color=colors, node_size=800,
                     font_size=7, arrows=True, arrowsize=15, edge_color="#555", width=0.8)
    patches = [
        mpatches.Patch(color="#e74c3c", label="Target (Attack)"),
        mpatches.Patch(color="#f39c12", label="PAG-linked Causal Feature"),
        mpatches.Patch(color="#95a5a6", label="Other Feature"),
    ]
    ax.legend(handles=patches, loc="upper left", fontsize=9)
    ax.set_title(
        f"CausalShield -- FCI Partial Ancestral Graph  (alpha={ALPHA}, n={len(subset):,})\n"
        f"{len(causal_features)} causal features  |  {len(bidirected)} bidirected edges (latent confounders)",
        fontsize=11, fontweight="bold")
    ax.axis("off"); plt.tight_layout()
    fig.savefig(CAUSAL_DIR / "causal_graph.png", dpi=150); plt.close()
    print(f"      causal_graph.png -> {CAUSAL_DIR}")
except Exception as e:
    print(f"      [WARN] Visualisation skipped: {e}")

print("\n" + "=" * 60)
print("Phase 2 complete -- FCI PAG")
print(f"  Algorithm     : FCI (latent confounders handled)")
print(f"  Alpha         : {ALPHA}")
print(f"  PAG edges     : {len(edge_list)}")
print(f"  Bidirected    : {len(bidirected)}")
print(f"  Causal feats  : {len(causal_features)}")
print("=" * 60)
