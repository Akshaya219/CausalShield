import json
from pathlib import Path
import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")
CAUSAL_DIR = Path("data/causal")
SAMPLE_SIZE = 20_000
SEED = 42
FEATURES = [
    "HTTPS",
    "ack_flag_number",
    "psh_flag_number",
    "rst_flag_number",
    "rst_count",
    "Variance",
    "Duration",
]

def main():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze().values
    feature_names = json.load(open(CAUSAL_DIR / "causal_feature_names.json"))
    available = [feature for feature in feature_names if feature in X_train.columns]
    X_train_c = X_train[available].copy()

    if len(X_train_c) > SAMPLE_SIZE:
        idx = np.random.RandomState(SEED).choice(len(X_train_c), SAMPLE_SIZE, replace=False)
        Xs = X_train_c.iloc[idx].reset_index(drop=True)
        ys = y_train[idx]
    else:
        Xs = X_train_c.reset_index(drop=True)
        ys = y_train.copy()

    ace_file = CAUSAL_DIR / "ace_scores.json"
    with open(ace_file, "r", encoding="utf-8") as f:
        old_data = json.load(f)

    new_data = {}
    
    print("Feature Verification (Old vs New at 8 decimals):")
    expected_values = {
        "HTTPS": -0.37515204,
        "ack_flag_number": -0.31593641,
        "psh_flag_number": 0.09362934,
        "rst_flag_number": 0.16656181,
        "rst_count": -0.26600063
    }
    
    check_failed = False

    for feature in FEATURES:
        values = Xs[feature].values.astype(float)
        if feature in {"HTTPS", "ack_flag_number", "psh_flag_number", "rst_flag_number", "rst_count"}:
            x_bin = values > 0
            bin_rule = "x > 0"
        else:
            threshold = np.median(values)
            x_bin = values >= threshold
            bin_rule = "x >= median"

        group_one = x_bin == 1
        group_zero = x_bin == 0
        p1 = ys[group_one].mean() if group_one.sum() > 0 else 0.0
        p0 = ys[group_zero].mean() if group_zero.sum() > 0 else 0.0
        new_ace = float(p1 - p0)
        old_ace = float(old_data[feature]["ace"])

        feat_entry = old_data[feature].copy()
        feat_entry["ace"] = round(new_ace, 8)
        feat_entry["abs_ace"] = round(abs(new_ace), 8)
        feat_entry["method"] = "observational_proxy"
        feat_entry["binarization"] = bin_rule
        new_data[feature] = feat_entry

        print(f"{feature:18s} | Old: {old_ace:12.8f} | New: {new_ace:12.8f}")

        # Checks
        if feature in expected_values:
            target = expected_values[feature]
            if abs(new_ace - target) > 1e-6:
                print(f"ERROR: {feature} new ACE {new_ace:.8f} does not match target {target:.8f}")
                check_failed = True
        else:
            diff = abs(new_ace - old_ace)
            if diff >= 1e-5:
                print(f"ERROR: {feature} diff {diff:.8f} >= 1e-5")
                check_failed = True

    if check_failed:
        raise ValueError("PART 1 Check failed!")

    with open(ace_file, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2)

    print("\ndata/causal/ace_scores.json successfully updated.")

if __name__ == "__main__":
    main()
