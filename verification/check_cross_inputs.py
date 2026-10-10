import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REBUILT_DIR = ROOT / "data" / "cross_val_rebuilt"
EXISTING_DIR = ROOT / "data" / "cross_val"
PARQUET_PATH = ROOT / "CIC-ToN-IoT-V2.parquet"
OUT_DIR = ROOT / "results"
OUT_PATH = OUT_DIR / "cross_input_repro.txt"

lines = []

def emit(msg=""):
    print(msg, flush=True)
    lines.append(msg)

def main():
    emit("=== STEP 19 Part B: Checking Cross Inputs Reproducibility ===")
    
    # 1. Read rebuilt CSVs and parquet Attack column
    emit("Reading rebuilt X_toniot_cross.csv, y_toniot_cross.csv, and parquet Attack column...")
    X_rebuilt = pd.read_csv(REBUILT_DIR / "X_toniot_cross.csv")
    y_rebuilt = pd.read_csv(REBUILT_DIR / "y_toniot_cross.csv").values.ravel()
    attack_raw = pd.read_parquet(PARQUET_PATH, columns=["Attack"])["Attack"].values
    
    n_before = len(X_rebuilt)
    emit(f"Rows before dropping negative Duration: {n_before}")
    
    # 2. Drop rows with Duration < 0
    mask_valid = X_rebuilt["Duration"] >= 0
    X_filtered = X_rebuilt[mask_valid].reset_index(drop=True)
    y_filtered = y_rebuilt[mask_valid]
    attack_filtered = attack_raw[mask_valid]
    
    n_after = len(X_filtered)
    emit(f"Rows after dropping negative Duration (< 0): {n_after}")
    emit(f"Dropped rows count: {n_before - n_after}")
    
    # 3. Compare with data/cross_val/
    emit("\nReading existing data/cross_val/ files...")
    X_existing = pd.read_csv(EXISTING_DIR / "X_toniot_cross.csv")
    y_existing = pd.read_csv(EXISTING_DIR / "y_toniot_cross.csv").values.ravel()
    labels_existing = pd.read_csv(EXISTING_DIR / "labels_toniot_raw.csv").values.ravel()
    
    emit(f"Existing X shape: {X_existing.shape}")
    emit(f"Existing y shape: {y_existing.shape}")
    emit(f"Existing labels shape: {labels_existing.shape}")
    
    # Compare X
    # Ensure same column order
    X_filtered = X_filtered[X_existing.columns]
    
    x_close = np.allclose(X_filtered.values, X_existing.values, rtol=1e-5, atol=1e-6)
    emit(f"\nX_toniot_cross.csv match (np.allclose rtol=1e-5 atol=1e-6): {x_close}")
    if not x_close:
        # Find first mismatching index and location
        diff_mask = ~np.isclose(X_filtered.values, X_existing.values, rtol=1e-5, atol=1e-6)
        mismatch_rows, mismatch_cols = np.where(diff_mask)
        if len(mismatch_rows) > 0:
            r, c = mismatch_rows[0], mismatch_cols[0]
            col_name = X_existing.columns[c]
            val_filt = X_filtered.iloc[r, c]
            val_exist = X_existing.iloc[r, c]
            emit(f"  First mismatch in X at row index {r}, column '{col_name}': filtered={val_filt}, existing={val_exist}")
            
    # Compare y
    y_equal = np.array_equal(y_filtered, y_existing)
    emit(f"y_toniot_cross.csv array equality match: {y_equal}")
    if not y_equal:
        diff_idx = np.where(y_filtered != y_existing)[0]
        if len(diff_idx) > 0:
            idx = diff_idx[0]
            emit(f"  First mismatch in y at row index {idx}: filtered={y_filtered[idx]}, existing={y_existing[idx]}")
            
    # Compare labels_toniot_raw.csv
    labels_equal_exact = np.array_equal(attack_filtered, labels_existing)
    emit(f"labels_toniot_raw.csv exact equality match: {labels_equal_exact}")
    
    labels_equal_lower = np.array_equal([str(a).lower() for a in attack_filtered], [str(l).lower() for l in labels_existing])
    emit(f"labels_toniot_raw.csv lowercase equality match: {labels_equal_lower}")
    
    if not labels_equal_exact:
        diff_idx = np.where(attack_filtered != labels_existing)[0]
        if len(diff_idx) > 0:
            idx = diff_idx[0]
            emit(f"  First mismatch in labels at row index {idx}: parquet='{attack_filtered[idx]}', existing='{labels_existing[idx]}'")
            
    # Case / format differences analysis
    case_diffs = []
    for a, l in zip(attack_filtered, labels_existing):
        if a != l:
            case_diffs.append((a, l))
    if case_diffs:
        unique_diffs = sorted(set(case_diffs))
        emit(f"Unique label format/case differences (parquet_value vs existing_value): {unique_diffs}")
    else:
        emit("No label format/case differences detected.")

    # 4. Final Conclusion
    all_reproducible = x_close and y_equal and (labels_equal_exact or labels_equal_lower)
    emit("\n--- REPRODUCIBILITY CONCLUSION ---")
    if all_reproducible:
        emit("The existing cross files ARE EXACTLY REPRODUCIBLE from the raw parquet using the single rule 'drop rows with negative Flow Duration (Duration < 0)'.")
    else:
        emit("The existing cross files ARE NOT EXACTLY REPRODUCIBLE from the raw parquet.")
        
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    emit(f"\nWrote results to {OUT_PATH}")

if __name__ == "__main__":
    main()
