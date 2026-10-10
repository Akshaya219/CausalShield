import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz

PROCESSED_DIR = Path("data/processed")
CAUSAL_DIR = Path("data/causal")
RESULTS_DIR = Path("results")
LOG_PATH = RESULTS_DIR / "fci_stability_log.txt"
SUBSET_SIZE = 5_000
N_RUNS = 5
SEED = 42
ALPHA = 0.05

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_PATH.open("w", encoding="utf-8")


def emit(message=""):
    print(message, flush=True)
    log_file.write(message + "\n")
    log_file.flush()


def clean_for_fci(X_sub):
    stds = X_sub.std()
    const_cols = stds[stds < 1e-10].index.tolist()
    X_sub = X_sub.drop(columns=const_cols)
    X_sub = X_sub.T.drop_duplicates().T
    corr_m = X_sub.corr().abs()
    upper = corr_m.where(np.triu(np.ones(corr_m.shape), k=1).astype(bool))
    high_corr = [c for c in upper.columns if any(upper[c] > 0.99)]
    return X_sub.drop(columns=high_corr)


def edge_map(adjacency, names):
    edges = {}
    for i in range(adjacency.shape[0]):
        for j in range(i + 1, adjacency.shape[1]):
            marks = (int(adjacency[i, j]), int(adjacency[j, i]))
            if marks != (0, 0):
                edges[(names[i], names[j])] = marks
    return edges


def incident_to_attack(edge):
    return "Attack" in edge


try:
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()
    saved_adjacency = np.load(CAUSAL_DIR / "causal_adjacency.npy")
    saved_features = json.load(open(CAUSAL_DIR / "causal_feature_names.json"))
    all_names = saved_features + ["Attack"]
    saved_edges = edge_map(saved_adjacency, all_names)

    df_full = X_train.copy()
    df_full["__target__"] = y_train.values
    n_per_class = SUBSET_SIZE // 2
    benign = df_full[df_full["__target__"] == 0].sample(frac=1, random_state=SEED)
    attack = df_full[df_full["__target__"] == 1].sample(frac=1, random_state=SEED)
    required = N_RUNS * n_per_class
    if len(benign) < required or len(attack) < required:
        raise ValueError("Development pool does not contain five non-overlapping balanced subsets.")

    run_edges = []
    emit(f"FCI stability check: {N_RUNS} non-overlapping subsets, alpha={ALPHA}, seed={SEED}")
    emit(f"Saved graph edges: {len(saved_edges)}")

    for run_index in range(N_RUNS):
        start = run_index * n_per_class
        stop = start + n_per_class
        benign_chunk = benign.iloc[start:stop]
        attack_chunk = attack.iloc[start:stop]
        subset = pd.concat([benign_chunk, attack_chunk]).sample(
            frac=1,
            random_state=SEED + run_index,
        ).reset_index(drop=True)
        y_sub = subset["__target__"].values
        X_sub = clean_for_fci(subset.drop(columns=["__target__"]))
        data_matrix = np.hstack([X_sub.values, y_sub.reshape(-1, 1)]).astype(np.float64)
        feature_names = list(X_sub.columns)
        names = feature_names + ["Attack"]

        started = time.time()
        graph, _ = fci(
            data_matrix,
            independence_test_method=fisherz,
            alpha=ALPHA,
            depth=-1,
            max_path_length=-1,
            verbose=False,
            show_progress=False,
        )
        adjacency = graph.graph
        current_edges = edge_map(adjacency, names)
        run_edges.append(current_edges)

        attack_adjacent = {
            feature
            for feature in feature_names
            if adjacency[feature_names.index(feature), len(feature_names)] != 0
            or adjacency[len(feature_names), feature_names.index(feature)] != 0
        }
        expected_attack_features = {
            "HTTPS",
            "ack_flag_number",
            "psh_flag_number",
            "rst_flag_number",
            "rst_count",
            "Variance",
            "Duration",
        }
        missing_attack_features = expected_attack_features - attack_adjacent
        extra_attack_features = attack_adjacent - expected_attack_features
        all_expected_attack_features_adjacent = attack_adjacent == expected_attack_features
        identical = sum(
            edge in saved_edges and saved_edges[edge] == marks
            for edge, marks in current_edges.items()
        )
        same_endpoints_different_marks = sum(
            edge in saved_edges and saved_edges[edge] != marks
            for edge, marks in current_edges.items()
        )
        run_only = set(current_edges) - set(saved_edges)
        saved_only = set(saved_edges) - set(current_edges)
        only_one = len(saved_only | run_only)
        changed_edges = {
            edge
            for edge in set(current_edges) | set(saved_edges)
            if current_edges.get(edge) != saved_edges.get(edge)
        }
        attack_change = any(incident_to_attack(edge) for edge in changed_edges)

        emit(
            f"run={run_index + 1} edges={len(current_edges)} "
            f"bidirected={sum(marks == (1, 1) for marks in current_edges.values())} "
            f"all_expected_attack_features_adjacent={all_expected_attack_features_adjacent} "
            f"missing_attack_features={sorted(missing_attack_features)} "
            f"extra_attack_features={sorted(extra_attack_features)} "
            f"identical={identical} "
            f"same_endpoints_different_marks={same_endpoints_different_marks} "
            f"only_one={only_one} saved_only={len(saved_only)} run_only={len(run_only)} "
            f"attack_incident_change={attack_change} "
            f"elapsed_seconds={time.time() - started:.1f}"
        )

    stable_edges = set(saved_edges)
    for edges in run_edges:
        stable_edges = {
            edge
            for edge in stable_edges
            if edge in edges and edges[edge] == saved_edges[edge]
        }
    percentage = 100 * len(stable_edges) / len(saved_edges)
    emit(
        f"summary_edges_identical_in_all_5_and_saved={len(stable_edges)} "
        f"of_saved={len(saved_edges)} percentage={percentage:.2f}%"
    )
finally:
    log_file.close()
