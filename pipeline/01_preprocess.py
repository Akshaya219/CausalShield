import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import os

BASE       = "D:/causal_ai/CICIOT23"
TRAIN_PATH = os.path.join(BASE, "train/train.csv")
VAL_PATH   = os.path.join(BASE, "validation/validation.csv")
TEST_PATH  = os.path.join(BASE, "test/test.csv")

os.makedirs("data/processed1", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# 7-category map — still needed for IRM later
category_map = {
    'DDoS-ICMP_Flood'         : 'DDoS',
    'DDoS-UDP_Flood'          : 'DDoS',
    'DDoS-TCP_SYN_Flood'      : 'DDoS',
    'DDoS-PSHACK_Flood'       : 'DDoS',
    'DDoS-SYN_Flood'          : 'DDoS',
    'DDoS-RSTFINFlood'        : 'DDoS',
    'DDoS-HTTP_Flood'         : 'DDoS',
    'DDoS-SlowLoris'          : 'DDoS',
    'DDoS-ICMP_Fragmentation' : 'DDoS',
    'DDoS-UDP_Fragmentation'  : 'DDoS',
    'DDoS-ACK_Fragmentation'  : 'DDoS',
    'DoS-UDP_Flood'           : 'DoS',
    'DoS-SYN_Flood'           : 'DoS',
    'DoS-TCP_Flood'           : 'DoS',
    'DoS-HTTP_Flood'          : 'DoS',
    'Mirai-greeth_flood'      : 'Mirai',
    'Mirai-udpplain'          : 'Mirai',
    'Mirai-hostbruteforce'    : 'Mirai',
    'Recon-HostDiscovery'     : 'Reconnaissance',
    'Recon-OSScan'            : 'Reconnaissance',
    'Recon-PortScan'          : 'Reconnaissance',
    'Recon-PingSweep'         : 'Reconnaissance',
    'VulnerabilityScan'       : 'Reconnaissance',
    'BruteForce-Web'          : 'BruteForce',
    'BruteForce-XSS'          : 'BruteForce',
    'CommandInjection'        : 'BruteForce',
    'BrowserHijacking'        : 'BruteForce',
    'MITM-ArpSpoofing'        : 'Spoofing',
    'DNS_Spoofing'            : 'Spoofing',
    'SqlInjection'            : 'WebBased',
    'Uploading_Attack'        : 'WebBased',
    'XSS'                     : 'WebBased',
    'Backdoor_Malware'        : 'WebBased',
    'BenignTraffic'           : 'Benign'
}

# These 8 features are always kept — final CSVs will have ONLY these columns
REQUIRED_FEATURES = [
    'HTTPS', 'ack_flag_number', 'psh_flag_number', 'rst_flag_number',
    'rst_count', 'Variance', 'Duration'
]

FEATURE_COLS = [
    'flow_duration', 'Header_Length', 'Protocol Type', 'Duration',
    'Rate', 'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number',
    'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
    'ece_flag_number', 'cwr_flag_number', 'ack_count', 'syn_count',
    'fin_count', 'urg_count', 'rst_count', 'HTTP', 'HTTPS', 'DNS',
    'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 'ARP',
    'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std',
    'Tot size', 'IAT', 'Number', 'Magnitue', 'Radius', 'Covariance',
    'Variance', 'Weight'
]

def process_split(path, split_name, max_per_class):
    print(f"\n{'='*50}")
    print(f"Processing {split_name}")

    df = pd.read_csv(path, low_memory=False)
    print(f"Raw shape: {df.shape}")

    # ── Map to 7 categories ───────────────────────────────────────
    df['category'] = df['label'].map(category_map)
    df = df[df['category'].notna()].reset_index(drop=True)

    # ── Binary target: 0=Benign, 1=Attack ────────────────────────
    df['binary_label'] = (df['category'] != 'Benign').astype(int)

    print(f"Binary distribution (before balancing):")
    print(f"  Benign : {(df['binary_label']==0).sum():,}")
    print(f"  Attack : {(df['binary_label']==1).sum():,}")

    # ── Balance Benign vs Attack ──────────────────────────────────
    balanced_dfs = []
    for cat in df['category'].unique():
        cat_df = df[df['category'] == cat]
        if len(cat_df) > max_per_class:
            cat_df = resample(cat_df,
                             n_samples=max_per_class,
                             random_state=42,
                             replace=False)
        balanced_dfs.append(cat_df)

    df_bal = pd.concat(balanced_dfs, ignore_index=True)
    df_bal = df_bal.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nAfter balancing:")
    print(f"  Binary distribution:")
    print(f"    Benign : {(df_bal['binary_label']==0).sum():,}")
    print(f"    Attack : {(df_bal['binary_label']==1).sum():,}")
    print(f"  Category breakdown:")
    print(df_bal['category'].value_counts())

    # ── Features ──────────────────────────────────────────────────
    available = [c for c in FEATURE_COLS if c in df_bal.columns]
    X = df_bal[available].copy()

    # Binary target
    y_binary = df_bal['binary_label'].values

    # Category labels — saved separately for IRM later
    y_category = df_bal['category'].values

    # ── Clean ─────────────────────────────────────────────────────
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(numeric_only=True), inplace=True)

    # Drop zero-variance columns BUT always keep REQUIRED_FEATURES
    present_required = [c for c in REQUIRED_FEATURES if c in X.columns]
    missing_required = [c for c in REQUIRED_FEATURES if c not in X.columns]
    if missing_required:
        print(f"  WARNING: required features missing from source data: {missing_required}")
    non_required = [c for c in X.columns if c not in present_required]
    keep_non_required = [c for c in non_required if X[c].nunique() > 1]
    X = X[keep_non_required + present_required]

    print(f"\nFinal shape: X={X.shape}, y_binary={y_binary.shape}")
    print(f"  Required features present: {present_required}")

    return X, y_binary, y_category

# ── Run all splits ────────────────────────────────────────────────
X_train, y_train, cat_train = process_split(TRAIN_PATH, "TRAIN", max_per_class=30000)
X_val,   y_val,   cat_val   = process_split(VAL_PATH,   "VALIDATION", max_per_class=10000)
X_test,  y_test,  cat_test  = process_split(TEST_PATH,  "TEST", max_per_class=10000)

# ── Keep ONLY the 8 required features ────────────────────────────
final_cols = [f for f in REQUIRED_FEATURES if f in X_train.columns
                                            and f in X_val.columns
                                            and f in X_test.columns]
missing = [f for f in REQUIRED_FEATURES if f not in final_cols]
if missing:
    print(f"\nWARNING: These required features missing from ≥1 split: {missing}")

X_train = X_train[final_cols]
X_val   = X_val[final_cols]
X_test  = X_test[final_cols]

print(f"\n{'='*50}")
print(f"FINAL SUMMARY")
print(f"{'='*50}")
print(f"Final columns ({len(final_cols)}): {final_cols}")
print(f"Train      : {X_train.shape} | Benign={(y_train==0).sum():,} | Attack={(y_train==1).sum():,}")
print(f"Validation : {X_val.shape}   | Benign={(y_val==0).sum():,}   | Attack={(y_val==1).sum():,}")
print(f"Test       : {X_test.shape}  | Benign={(y_test==0).sum():,}   | Attack={(y_test==1).sum():,}")

# ── Save ──────────────────────────────────────────────────────────
for target_dir in ["data/train", "data/test", "data/processed"]:
    os.makedirs(target_dir, exist_ok=True)

# Training & validation splits
X_train.to_csv("data/train/X_train.csv", index=False)
X_val.to_csv(  "data/train/X_val.csv",   index=False)
pd.Series(y_train).to_csv("data/train/y_train.csv", index=False)
pd.Series(y_val).to_csv(  "data/train/y_val.csv",   index=False)
pd.Series(cat_train).to_csv("data/train/cat_train.csv", index=False)
pd.Series(cat_val).to_csv(  "data/train/cat_val.csv",   index=False)

# Testing splits
X_test.to_csv( "data/test/X_test.csv",  index=False)
pd.Series(y_test).to_csv( "data/test/y_test.csv",  index=False)
pd.Series(cat_test).to_csv( "data/test/cat_test.csv",  index=False)
pd.Series(final_cols).to_csv("data/test/feature_names.csv", index=False)

# Processed backwards-compatibility
X_train.to_csv("data/processed/X_train.csv", index=False)
X_val.to_csv(  "data/processed/X_val.csv",   index=False)
X_test.to_csv( "data/processed/X_test.csv",  index=False)
pd.Series(y_train).to_csv("data/processed/y_train.csv", index=False)
pd.Series(y_val).to_csv(  "data/processed/y_val.csv",   index=False)
pd.Series(y_test).to_csv( "data/processed/y_test.csv",  index=False)
pd.Series(cat_train).to_csv("data/processed/cat_train.csv", index=False)
pd.Series(cat_val).to_csv(  "data/processed/cat_val.csv",   index=False)
pd.Series(cat_test).to_csv( "data/processed/cat_test.csv",  index=False)
pd.Series(final_cols).to_csv("data/processed/feature_names.csv", index=False)