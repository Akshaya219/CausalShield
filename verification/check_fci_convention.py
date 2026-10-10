import numpy as np
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz

SEED = 0
N_SAMPLES = 5_000
rng = np.random.default_rng(SEED)

X = rng.normal(size=N_SAMPLES)
Y = rng.normal(size=N_SAMPLES)
Z = X + Y + rng.normal(scale=0.5, size=N_SAMPLES)
W = Z + rng.normal(scale=0.5, size=N_SAMPLES)
data = np.column_stack([X, Y, Z, W])
labels = ["X", "Y", "Z", "W"]

graph, _ = fci(
    data,
    independence_test_method=fisherz,
    alpha=0.05,
    depth=-1,
    max_path_length=-1,
    verbose=False,
    show_progress=False,
)
adjacency = graph.graph

print("Raw G.graph:")
print(adjacency)
print()
for left, right in [(0, 2), (2, 0), (2, 3), (3, 2)]:
    print(f"G.graph[{labels[left]},{labels[right]}] = {int(adjacency[left, right])}")


def describe(left, right):
    forward = int(adjacency[left, right])
    reverse = int(adjacency[right, left])
    if forward == 1 and reverse == -1:
        return f"{labels[left]} -> {labels[right]}"
    if forward == -1 and reverse == 1:
        return f"{labels[right]} -> {labels[left]}"
    if forward == 1 and reverse == 1:
        return f"{labels[left]} <-> {labels[right]}"
    if forward == 1 and reverse == 2:
        return f"{labels[left]} o-> {labels[right]}"
    if forward == 2 and reverse == 1:
        return f"{labels[right]} o-> {labels[left]}"
    if forward == 2 and reverse == 2:
        return f"{labels[left]} o-o {labels[right]}"
    return f"{labels[left]} / {labels[right]} marks ({forward}, {reverse})"


print()
print(f"Decoded X/Z relation: {describe(0, 2)}")
print(f"Decoded Z/W relation: {describe(2, 3)}")
print("Convention: (1, -1) in (G.graph[i,j], G.graph[j,i]) means i -> j;")
print("the reversed pair (-1, 1) means j -> i.")
