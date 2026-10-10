from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DATA_DIR = Path("data/cross_val")
PARQUET_PATH = Path("CIC-ToN-IoT-V2.parquet")
X_PATH = DATA_DIR / "X_toniot_cross.csv"
Y_PATH = DATA_DIR / "y_toniot_cross.csv"

X = pd.read_csv(X_PATH)
y = pd.read_csv(Y_PATH).squeeze().to_numpy()
schema = pq.ParquetFile(PARQUET_PATH).schema_arrow.names
label_column = "Attack" if "Attack" in schema else None
candidate_columns = [
    "Flow Duration",
    "Packet Length Variance",
    "RST Flag Count",
    "ACK Flag Count",
    "PSH Flag Count",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
]
columns = [column for column in candidate_columns if column in schema]
if label_column is not None:
    columns.append(label_column)
source = pd.read_parquet(PARQUET_PATH, columns=columns, engine="pyarrow")

if label_column is None:
    raise ValueError("The parquet has no Attack label column.")

source_label = pd.to_numeric(source[label_column], errors="coerce").to_numpy()
if np.isnan(source_label).any():
    source_label = (source[label_column].astype(str).str.lower() != "benign").astype(int).to_numpy()
else:
    source_label = source_label.astype(int)

print(f"parquet_rows={len(source)} x_rows={len(X)}")
print(f"candidate_columns={columns}")

feature_candidates = {
    "Duration": ["Flow Duration"],
    "Variance": ["Packet Length Variance"],
    "rst_count": ["RST Flag Count"],
    "rst_flag_number": ["RST Flag Count"],
    "ack_flag_number": ["ACK Flag Count"],
    "psh_flag_number": ["PSH Flag Count", "Fwd PSH Flags", "Bwd PSH Flags"],
    "HTTPS": [],
}

for feature, candidates in feature_candidates.items():
    target = X[feature].astype(float).to_numpy()
    print(f"{feature}:")
    if not candidates:
        print("  source_column=None found in parquet")
        continue
    for candidate in candidates:
        if candidate not in source.columns:
            continue
        values = pd.to_numeric(source[candidate], errors="coerce").to_numpy()
        aligned = np.array_equal(target[:1000], values[:1000])
        means = []
        ratios = []
        for label in [0, 1]:
            target_mean = target[y == label].mean()
            source_mean = values[source_label == label].mean()
            means.append((label, target_mean, source_mean))
            ratios.append(source_mean / target_mean if target_mean != 0 else np.nan)
        same_group_means = all(
            np.isclose(target_mean, source_mean, rtol=1e-6, atol=1e-6)
            for _, target_mean, source_mean in means
        )
        constant_factor = (
            np.isfinite(ratios).all()
            and np.isclose(ratios[0], ratios[1], rtol=1e-6, atol=1e-6)
        )
        print(
            f"  source_column={candidate} first_1000_equal={aligned} "
            f"group_means_equal={same_group_means} "
            f"constant_source_over_target_factor={ratios[0]:.8f} "
            f"factor_consistent={constant_factor}"
        )
        for label, target_mean, source_mean in means:
            print(
                f"    label={label} target_mean={target_mean:.8f} "
                f"source_mean={source_mean:.8f}"
            )

print("TTL-like columns:")
ttl_columns = [column for column in schema if "ttl" in column.lower()]
if not ttl_columns:
    print("  none found by column name")
else:
    for column in ttl_columns:
        ttl = pd.to_numeric(pd.read_parquet(PARQUET_PATH, columns=[column])[column], errors="coerce")
        print(
            f"  {column}: benign_mean={ttl[source_label == 0].mean():.8f} "
            f"attack_mean={ttl[source_label == 1].mean():.8f}"
        )
