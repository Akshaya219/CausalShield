from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data/cross_val")
X = pd.read_csv(DATA_DIR / "X_toniot_cross.csv")
y = pd.read_csv(DATA_DIR / "y_toniot_cross.csv").values.ravel()

_, _, y_cal, y_heldout = train_test_split(
    X,
    y,
    train_size=0.01,
    random_state=42,
    stratify=y,
)

print(f"n_total={len(y)}")
print(f"n_cal={len(y_cal)}")
print(f"n_heldout={len(y_heldout)}")
print(f"cal_benign={(y_cal == 0).sum()} cal_attack={(y_cal == 1).sum()}")
print(f"heldout_benign={(y_heldout == 0).sum()} heldout_attack={(y_heldout == 1).sum()}")
print(f"table_x_count_sum={len(y)} expected=4847201 match={len(y) == 4847201}")
print("Table IX target: heldout=6904917 calibration=69747 reproducible_from_this_file=False")
