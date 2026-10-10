import json
from pathlib import Path

import numpy as np

CAUSAL_DIR = Path("data/causal")


def decode_draw_edges(adjacency, names):
    edges = []
    bidirected = 0
    edge_records = []
    for i in range(adjacency.shape[0]):
        for j in range(i + 1, adjacency.shape[1]):
            a_ij = int(adjacency[i, j])
            a_ji = int(adjacency[j, i])
            if a_ij == 0 and a_ji == 0:
                continue
            edge_records.append((names[i], names[j], a_ij, a_ji))
            if a_ij == 1 and a_ji == 1:
                bidirected += 1
                edges.extend([(names[i], names[j]), (names[j], names[i])])
            elif a_ij == -1 and a_ji == 1:
                edges.append((names[i], names[j]))
            elif a_ij == 1 and a_ji == -1:
                edges.append((names[j], names[i]))
            elif a_ij == 2 and a_ji == 1:
                edges.append((names[i], names[j]))
            elif a_ij == 1 and a_ji == 2:
                edges.append((names[j], names[i]))
            else:
                edges.append((names[i], names[j]))
    return edge_records, edges, bidirected


adjacency = np.load(CAUSAL_DIR / "causal_adjacency.npy")
feature_names = json.load(open(CAUSAL_DIR / "causal_feature_names.json"))
all_names = feature_names + ["Attack"]
edge_records, draw_edges, bidirected = decode_draw_edges(adjacency, all_names)
causal_features = {
    name
    for name_a, name_b, _, _ in edge_records
    for name in (name_a, name_b)
    if name != "Attack"
    and (name_a == "Attack" or name_b == "Attack")
}

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx

Gnx = nx.DiGraph()
Gnx.add_nodes_from(all_names)
Gnx.add_edges_from(draw_edges)
colors = [
    "#e74c3c" if name == "Attack" else
    "#f39c12" if name in causal_features else
    "#95a5a6"
    for name in Gnx.nodes()
]
pos = nx.spring_layout(Gnx, seed=42, k=1.8)
fig, ax = plt.subplots(figsize=(14, 10))
nx.draw_networkx(
    Gnx,
    pos,
    ax=ax,
    node_color=colors,
    node_size=800,
    font_size=7,
    arrows=True,
    arrowsize=15,
    edge_color="#555",
    width=0.8,
    arrowstyle="-|>",
)
patches = [
    mpatches.Patch(color="#e74c3c", label="Target (Attack)"),
    mpatches.Patch(color="#f39c12", label="PAG-linked Causal Feature"),
    mpatches.Patch(color="#95a5a6", label="Other Feature"),
]
ax.legend(handles=patches, loc="upper left", fontsize=9)
ax.set_title(
    f"CausalShield -- FCI Partial Ancestral Graph  (alpha=0.05, n=5,000)\n"
    f"{len(causal_features)} causal features  |  {len(edge_records)} edges ({bidirected} bidirected)",
    fontsize=11,
    fontweight="bold",
)
ax.axis("off")
plt.tight_layout()
fig.savefig(CAUSAL_DIR / "causal_graph.png", dpi=150)
plt.close(fig)
print(f"Wrote {CAUSAL_DIR / 'causal_graph.png'}")
print(f"Edges: {len(edge_records)} ({bidirected} bidirected)")
