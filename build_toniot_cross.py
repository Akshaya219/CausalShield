from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PARQUET_PATH = Path("CIC-ToN-IoT-V2.parquet")
EXISTING_DIR = Path("data/cross_val")
REBUILT_DIR = Path("data/cross_val_rebuilt")
SELECTED_FEATURES = [
    "HTTPS",
    "ack_flag_number",
    "psh_flag_number",
    "rst_flag_number",
    "rst_count",
    "Variance",
    "Duration",
]
SOURCE_COLUMNS = [
    "Flow Duration",
    "Packet Length Variance",
    "RST Flag Count",
    "ACK Flag Count",
    "PSH Flag Count",
    "Attack",
]

schema = pq.ParquetFile(PARQUET_PATH).schema_arrow.names
columns = [column for column in SOURCE_COLUMNS if column in schema]
source = pd.read_parquet(PARQUET_PATH, columns=columns, engine="pyarrow")

raw_labels = source.pop("Attack")
numeric_labels = pd.to_numeric(raw_labels, errors="coerce")
if numeric_labels.notna().all():
    y = numeric_labels.astype(int).to_numpy()
else:
    labels = raw_labels.astype(str).str.strip().str.lower()
    y = (~labels.isin({"0", "benign", "benigntraffic", "normal"})).astype(int).to_numpy()

X = pd.DataFrame(index=source.index)
X["HTTPS"] = 0.0
X["ack_flag_number"] = source["ACK Flag Count"]
X["psh_flag_number"] = source["PSH Flag Count"]
X["rst_flag_number"] = source["RST Flag Count"]
X["rst_count"] = source["RST Flag Count"]
X["Variance"] = source["Packet Length Variance"]
X["Duration"] = source["Flow Duration"]
X = X[SELECTED_FEATURES].astype(float)

finite_mask = np.isfinite(X.to_numpy()).all(axis=1) & np.isfinite(y)
X = X.loc[finite_mask].reset_index(drop=True)
y = y[finite_mask]

REBUILT_DIR.mkdir(parents=True, exist_ok=True)
X.to_csv(REBUILT_DIR / "X_toniot_cross.csv", index=False)
pd.Series(y, name="Attack").to_csv(REBUILT_DIR / "y_toniot_cross.csv", index=False)

existing_X = pd.read_csv(EXISTING_DIR / "X_toniot_cross.csv")[SELECTED_FEATURES].astype(float)
existing_y = pd.read_csv(EXISTING_DIR / "y_toniot_cross.csv").squeeze().to_numpy()

print(f"rebuilt_rows={len(X)} expected=4847201 match={len(X) == 4847201}")
print(f"existing_rows={len(existing_X)}")
print(f"X_shape_equal={X.shape == existing_X.shape}")
if X.shape == existing_X.shape:
    print(f"X_allclose={np.allclose(X.to_numpy(), existing_X.to_numpy())}")
else:
    print("X_allclose=False")
print(f"y_shape_equal={y.shape == existing_y.shape}")
if y.shape == existing_y.shape:
    print(f"y_array_equal={np.array_equal(y, existing_y)}")
else:
    print("y_array_equal=False")

for label, frame in [("rebuilt", X), ("existing", existing_X)]:
    variance = frame["Variance"]
    print(
        f"{label}_Variance_min={variance.min():.8f} "
        f"median={variance.median():.8f} max={variance.max():.8f}"
    )

ciciot_variance = pd.read_csv("data/processed/X_train.csv", usecols=["Variance"])["Variance"]
print(
    f"ciciot_variance_max={ciciot_variance.max():.8f} "
    f"all_values_le_1={bool((ciciot_variance <= 1).all())}"
)
