from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data/cross_val")
X_PATH = DATA_DIR / "X_toniot_cross.csv"
Y_PATH = DATA_DIR / "y_toniot_cross.csv"

X = pd.read_csv(X_PATH)
y = pd.read_csv(Y_PATH).squeeze().to_numpy()

print(f"rows={len(X)}")
for feature in ["Duration", "Variance"]:
    print(f"{feature}:")
    for label, label_value in [("benign", 0), ("attack", 1)]:
        values = X.loc[y == label_value, feature].astype(float)
        print(
            f"  {label}: min={values.min():.8f} median={values.median():.8f} "
            f"mean={values.mean():.8f} max={values.max():.8f} n={len(values)}"
        )
    benign_mean = X.loc[y == 0, feature].astype(float).mean()
    attack_mean = X.loc[y == 1, feature].astype(float).mean()
    print(f"  mean_ratio_benign_attack={benign_mean / attack_mean:.8f}")

rst_zero_share = (X["rst_count"] == 0).mean()
https_all_zero = bool((X["HTTPS"] == 0).all())
print(f"rst_count_zero_share={rst_zero_share:.8f}")
print(f"HTTPS_all_zero={https_all_zero}")
