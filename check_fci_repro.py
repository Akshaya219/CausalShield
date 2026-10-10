import time
from pathlib import Path

import numpy as np
import pandas as pd

from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz

PROCESSED_DIR = Path("data/processed")
SUBSET_SIZE = 5_000
SEED = 42
ALPHA = 0.05

X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()

df_full = X_train.copy()
df_full["__target__"] = y_train.values
n_per = SUBSET_SIZE // 2
benign = df_full[df_full["__target__"] == 0].sample(
    n=min(n_per, (df_full["__target__"] == 0).sum()),
    random_state=SEED,
)
attack = df_full[df_full["__target__"] == 1].sample(
    n=min(n_per, (df_full["__target__"] == 1).sum()),
    random_state=SEED,
)
subset = pd.concat([benign, attack]).sample(frac=1, random_state=SEED).reset_index(drop=True)
y_sub = subset["__target__"].values
X_sub = subset.drop(columns=["__target__"])

stds = X_sub.std()
const_cols = stds[stds < 1e-10].index.tolist()
X_sub = X_sub.drop(columns=const_cols)
X_sub = X_sub.T.drop_duplicates().T
corr_m = X_sub.corr().abs()
upper = corr_m.where(np.triu(np.ones(corr_m.shape), k=1).astype(bool))
high_corr = [c for c in upper.columns if any(upper[c] > 0.99)]
X_sub = X_sub.drop(columns=high_corr)

data_matrix = np.hstack([X_sub.values, y_sub.reshape(-1, 1)]).astype(np.float64)

start = time.time()
graph, _ = fci(
    data_matrix,
    independence_test_method=fisherz,
    alpha=ALPHA,
    depth=-1,
    max_path_length=-1,
    verbose=False,
    show_progress=False,
)
adj = graph.graph

edge_count = 0
bidirected_count = 0
for i in range(adj.shape[0]):
    for j in range(i + 1, adj.shape[1]):
        x, y = int(adj[i, j]), int(adj[j, i])
        if x != 0 or y != 0:
            edge_count += 1
            if x == 1 and y == 1:
                bidirected_count += 1

print(f"subset_size={len(subset)} seed={SEED} alpha={ALPHA}")
print(f"total_edges={edge_count}")
print(f"bidirected_edges={bidirected_count}")
print(f"elapsed_seconds={time.time() - start:.1f}")
print(f"comparison_70_47={'MATCH' if (edge_count, bidirected_count) == (70, 47) else 'NO'}")
print(f"comparison_68_46={'MATCH' if (edge_count, bidirected_count) == (68, 46) else 'NO'}")
