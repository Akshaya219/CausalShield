import json
from pathlib import Path

import numpy as np

CAUSAL_DIR = Path("data/causal")
RESULTS_DIR = Path("results")


def decode_edge(a_ij, a_ji):
    """Decode marks at i and j under causal-learn's confirmed convention."""
    a_ij = int(a_ij)
    a_ji = int(a_ji)
    if (a_ij, a_ji) == (-1, 1):
        return "directed", "i", "j"
    if (a_ij, a_ji) == (1, -1):
        return "directed", "j", "i"
    if (a_ij, a_ji) == (1, 1):
        return "bidirected", None, None
    if (a_ij, a_ji) == (2, 1):
        return "o->", None, "j"
    if (a_ij, a_ji) == (1, 2):
        return "o->", None, "i"
    if (a_ij, a_ji) == (2, 2):
        return "o-o", None, None
    if (a_ij, a_ji) == (0, 0):
        return "none", None, None
    return "other", None, None


def named_edge(edge, names):
    i, j, a_ij, a_ji = edge
    edge_type, tail_side, head_side = decode_edge(a_ij, a_ji)
    if edge_type == "directed":
        return f"{names[i]} -> {names[j]}" if tail_side == "i" else f"{names[j]} -> {names[i]}"
    if edge_type == "bidirected":
        return f"{names[i]} <-> {names[j]}"
    if edge_type == "o->":
        source = names[i] if head_side == "j" else names[j]
        target = names[j] if head_side == "j" else names[i]
        return f"{source} o-> {target}"
    if edge_type == "o-o":
        return f"{names[i]} o-o {names[j]}"
    return f"{names[i]} ? {names[j]} ({a_ij}, {a_ji})"


def collect_edges(adjacency):
    return [
        (i, j, int(adjacency[i, j]), int(adjacency[j, i]))
        for i in range(adjacency.shape[0])
        for j in range(i + 1, adjacency.shape[1])
        if adjacency[i, j] != 0 or adjacency[j, i] != 0
    ]


# Self-test for the confirmed mark convention.
raw = np.array([
    [0, 0, 2, 0],
    [0, 0, 2, 0],
    [1, 1, 0, -1],
    [0, 0, 1, 0],
])
assert decode_edge(raw[0, 2], raw[2, 0]) == ("o->", None, "j")
assert decode_edge(raw[1, 2], raw[2, 1]) == ("o->", None, "j")
assert decode_edge(raw[2, 3], raw[3, 2]) == ("directed", "i", "j")

adjacency = np.load(CAUSAL_DIR / "causal_adjacency.npy")
feature_names = json.load(open(CAUSAL_DIR / "causal_feature_names.json"))
names = feature_names + ["Attack"]
edges = collect_edges(adjacency)
directed = [edge for edge in edges if decode_edge(edge[2], edge[3])[0] == "directed"]
possible = [edge for edge in edges if decode_edge(edge[2], edge[3])[0] == "o->"]
uncertain = [edge for edge in edges if decode_edge(edge[2], edge[3])[0] == "o-o"]
bidirected = [edge for edge in edges if decode_edge(edge[2], edge[3])[0] == "bidirected"]
attack_edges = [edge for edge in edges if "Attack" in (names[edge[0]], names[edge[1]])]

lines = []
lines.append("Saved graph decoded with causal-learn convention")
lines.append("Convention: G[i,j] == -1 and G[j,i] == 1 means i -> j")
lines.append("")
lines.append(f"Total edges: {len(edges)}")
lines.append(f"Directed edges ({len(directed)}):")
lines.extend(f"  {named_edge(edge, names)}" for edge in directed)
lines.append(f"Possible o-> edges ({len(possible)}):")
lines.extend(f"  {named_edge(edge, names)}" for edge in possible)
lines.append(f"Uncertain o-o edges ({len(uncertain)}):")
lines.extend(f"  {named_edge(edge, names)}" for edge in uncertain)
lines.append(f"Bidirected edges: {len(bidirected)}")
lines.append(f"Attack-incident edges ({len(attack_edges)}):")
lines.extend(
    f"  {named_edge(edge, names)}"
    for edge in attack_edges
)

output = "\n".join(lines)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
with open(RESULTS_DIR / "saved_graph_decoded.txt", "w", encoding="utf-8") as log:
    log.write(output + "\n")
print(output)
