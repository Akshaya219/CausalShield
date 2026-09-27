"""
CausalShield - Phase 5: SHAP Explainability
============================================
Computes SHAP values using TreeExplainer (exact, fast for XGBoost).
Outputs causally-grounded feature attributions.

Key design:
  - SHAP values computed on test set (subset of 2000 for speed)
  - Summary plot shows global importance ranked by mean|SHAP|
  - Waterfall plots for representative high/low risk flows
  - Causal context: annotates whether each feature is a direct cause,
    confounder, or indirect cause of Attack in the PC graph
  - All SHAP values saved for use in dashboard

Outputs:
  - results/shap_values.npy          (SHAP values, test subset)
  - results/shap_test_subset.csv     (corresponding feature values)
  - results/shap_summary.png
  - results/shap_waterfall_high.png
  - results/shap_waterfall_low.png
  - results/shap_beeswarm.png
  - results/shap_meta.json           (feature importance rankings)
"""

import json
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── paths ──────────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
CAUSAL_DIR    = Path("data/causal")
MODEL_DIR     = Path("models")
RESULTS_DIR   = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("CausalShield Phase 5 — SHAP Explainability")
print("=" * 60)

# ── 1. Load artifacts ──────────────────────────────────────────────────────────
print("\n[1/5] Loading model and data …")

import xgboost as xgb
import pickle

selected   = json.load(open(CAUSAL_DIR / "selected_features.json"))
ace_data   = json.load(open(CAUSAL_DIR / "ace_scores.json"))
meta       = json.load(open(MODEL_DIR  / "model_meta.json"))
graph_meta = json.load(open(CAUSAL_DIR / "causal_graph_meta.json"))

booster = xgb.Booster()
booster.load_model(str(MODEL_DIR / "xgb_model.json"))

# ── XGBoost 3.x / SHAP base_score fix ────────────────────────────────────────
# XGBoost 3.x stores base_score as '[5E-1]' in its internal config.
# SHAP's XGBTreeModelLoader calls float() on it directly and crashes.
# Fix: strip brackets, patch the config, re-save so it stays clean.

def _is_float(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False

print("      Checking base_score compatibility …")
try:
    cfg = json.loads(booster.save_config())
    bs  = cfg["learner"]["learner_model_param"]["base_score"]
    print(f"      base_score raw value : {bs}")
    if isinstance(bs, str) and (bs.startswith("[") or not _is_float(bs)):
        fixed = str(float(bs.strip("[]")))
        cfg["learner"]["learner_model_param"]["base_score"] = fixed
        booster.load_config(json.dumps(cfg))
        booster.save_model(str(MODEL_DIR / "xgb_model.json"))
        print(f"      base_score patched   : {bs} → {fixed}  (model re-saved)")
    else:
        print(f"      base_score OK        : {bs}")
except Exception as patch_err:
    print(f"      [WARN] base_score patch skipped: {patch_err}")

with open(MODEL_DIR / "calibrator.pkl", "rb") as f:
    calibrator = pickle.load(f)

X_test = pd.read_csv(PROCESSED_DIR / "X_test_causal.csv")[selected]
y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze().values

threshold = meta["threshold"]

# Get predictions
dtest     = xgb.DMatrix(X_test, feature_names=selected)
raw_probs = booster.predict(dtest)
cal_probs = calibrator.transform(raw_probs)
y_pred    = (cal_probs >= threshold).astype(int)

print(f"      Features  : {len(selected)}")
print(f"      Test size : {len(X_test)}")

# ── 2. SHAP values ─────────────────────────────────────────────────────────────
print("\n[2/5] Computing SHAP values …")

try:
    import shap
except ImportError as shap_err:
    shap = None
    print(f"      [WARN] SHAP package unavailable ({shap_err})")

if shap is not None:
    print(f"      SHAP version  : {shap.__version__}")
else:
    print("      SHAP version  : native XGBoost contributions")
print(f"      XGBoost version: {xgb.__version__}")

# Stratified subset for SHAP (2000 samples)
SHAP_SIZE = 2000
rng       = np.random.RandomState(42)
idx_0     = np.where(y_test == 0)[0]
idx_1     = np.where(y_test == 1)[0]
n0        = min(SHAP_SIZE // 2, len(idx_0))
n1        = min(SHAP_SIZE // 2, len(idx_1))
shap_idx  = np.concatenate([
    rng.choice(idx_0, n0, replace=False),
    rng.choice(idx_1, n1, replace=False),
])
shap_idx  = rng.permutation(shap_idx)

X_shap = X_test.iloc[shap_idx].reset_index(drop=True)
y_shap = y_test[shap_idx]
p_shap = cal_probs[shap_idx]

print(f"      SHAP subset   : {len(X_shap)} samples  "
      f"({(y_shap==0).sum()} benign / {(y_shap==1).sum()} attack)")

# Build exact tree contributions. Native XGBoost contributions are SHAP values
# in raw-margin space and avoid optional SHAP/scikit-learn imports entirely.
explainer = None
native_base_value = None
if shap is not None:
    try:
        explainer = shap.TreeExplainer(booster)
        shap_vals = explainer.shap_values(X_shap)
    except Exception as e1:
        print(f"      [WARN] SHAP TreeExplainer failed: {e1}")

if explainer is None:
    print("      Using XGBoost native predict contributions …")
    dm_shap = xgb.DMatrix(X_shap, feature_names=selected)
    contribs = booster.predict(dm_shap, pred_contribs=True)
    shap_vals = contribs[:, :-1]
    native_base_value = float(np.mean(contribs[:, -1]))

print(f"      SHAP values shape : {shap_vals.shape}")

# ── 3. Causal context annotations ─────────────────────────────────────────────
print("\n[3/5] Building causal context for features …")

causal_target_features = graph_meta.get("causal_features", [])
edge_list              = graph_meta.get("edges", [])

def get_causal_role(feat, causal_feats, edges):
    for e in edges:
        if e["from"] == feat and e["to"] == "Attack":
            if e["adj_ij"] == 1 and e["adj_ji"] == -1:
                return "direct_cause"
        if e["from"] == "Attack" and e["to"] == feat:
            if e["adj_ij"] == 1 and e["adj_ji"] == -1:
                return "effect"
    if feat in causal_feats:
        return "associated"
    return "indirect"

feature_roles = {f: get_causal_role(f, causal_target_features, edge_list) for f in selected}

# ── 4. Global feature importance ──────────────────────────────────────────────
mean_abs_shap = np.abs(shap_vals).mean(axis=0)
importance_df = pd.DataFrame({
    "feature"      : selected,
    "mean_abs_shap": mean_abs_shap,
    "ace"          : [ace_data.get(f, {}).get("ace",     0.0) for f in selected],
    "abs_ace"      : [ace_data.get(f, {}).get("abs_ace", 0.0) for f in selected],
    "causal_role"  : [feature_roles[f] for f in selected],
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

importance_df["rank_shap"] = range(1, len(importance_df) + 1)

print("\n      Top 15 features by mean |SHAP|:")
print(f"      {'Rank':<5} {'Feature':<35} {'SHAP':>8}  {'ACE':>8}  Role")
print(f"      {'-'*5} {'-'*35} {'-'*8}  {'-'*8}  {'-'*12}")
for _, row in importance_df.head(15).iterrows():
    print(f"      {int(row.rank_shap):<5} {row.feature:<35} "
          f"{row.mean_abs_shap:>8.4f}  {row.abs_ace:>8.4f}  {row.causal_role}")

# ── 5. Save outputs ────────────────────────────────────────────────────────────
print("\n[4/5] Saving SHAP outputs …")

np.save(RESULTS_DIR / "shap_values.npy", shap_vals)
X_shap.to_csv(RESULTS_DIR / "shap_test_subset.csv", index=False)
np.save(RESULTS_DIR / "shap_probs.npy",  p_shap)
np.save(RESULTS_DIR / "shap_labels.npy", y_shap)

shap_meta = {
    "features"         : selected,
    "importance_rank"  : importance_df[
        ["feature", "mean_abs_shap", "ace", "causal_role", "rank_shap"]
    ].to_dict(orient="records"),
    "shap_subset_size" : int(len(X_shap)),
}
with open(RESULTS_DIR / "shap_meta.json", "w") as f:
    json.dump(shap_meta, f, indent=2)

importance_df.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

print(f"      shap_values.npy        → {RESULTS_DIR}")
print(f"      shap_test_subset.csv   → {RESULTS_DIR}")
print(f"      shap_meta.json         → {RESULTS_DIR}")
print(f"      feature_importance.csv → {RESULTS_DIR}")

# ── 6. Visualisations ─────────────────────────────────────────────────────────
print("\n[5/5] Generating SHAP plots …")

role_colors = {
    "direct_cause": "#e74c3c",
    "effect"      : "#9b59b6",
    "associated"  : "#f39c12",
    "indirect"    : "#95a5a6",
    "confounder"  : "#3498db",
}

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # ── 6a. Summary bar plot ──────────────────────────────────────────────────
    top_n  = min(20, len(importance_df))
    top_df = importance_df.head(top_n)
    colors = [role_colors.get(r, "#95a5a6") for r in top_df["causal_role"]]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top_df["feature"][::-1], top_df["mean_abs_shap"][::-1], color=colors[::-1])
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(
        "CausalShield — Global Feature Importance (SHAP)\nColoured by causal role",
        fontsize=12, fontweight="bold"
    )
    legend_handles = [mpatches.Patch(color=v, label=k) for k, v in role_colors.items()]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "shap_summary.png", dpi=150)
    plt.close()
    print("      shap_summary.png saved")

    # ── 6b. Beeswarm ─────────────────────────────────────────────────────────
    try:
        expected_val = (
            explainer.expected_value if explainer is not None
            else native_base_value
        )
        if shap is not None:
            shap_exp = shap.Explanation(
                values        = shap_vals,
                base_values   = expected_val,
                data          = X_shap.values,
                feature_names = selected,
            )
            plt.figure(figsize=(10, 8))
            shap.plots.beeswarm(shap_exp, max_display=15, show=False)
        else:
            top_idx = np.argsort(np.abs(shap_vals).mean(axis=0))[::-1][:15]
            fig, ax = plt.subplots(figsize=(10, 8))
            for row, feature_idx in enumerate(top_idx):
                ax.scatter(
                    shap_vals[:, feature_idx],
                    np.full(len(shap_vals), row),
                    c=X_shap.iloc[:, feature_idx], cmap="coolwarm",
                    s=8, alpha=0.35, linewidths=0,
                )
            ax.set_yticks(range(len(top_idx)))
            ax.set_yticklabels([selected[i] for i in top_idx])
            ax.set_xlabel("SHAP value (raw margin)")
            ax.set_title("CausalShield — SHAP Beeswarm Plot", fontsize=12, fontweight="bold")
            ax.axvline(0, color="black", linewidth=0.7)
            plt.tight_layout()
        plt.savefig(RESULTS_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("      shap_beeswarm.png saved")
    except Exception as bee_err:
        print(f"      [WARN] Beeswarm skipped: {bee_err}")

    # ── 6c. Waterfall plots ───────────────────────────────────────────────────
    high_risk_idx = int(np.argmax(p_shap))
    low_risk_idx  = int(np.argmin(p_shap))

    for label, sample_idx, fname in [
        ("high_risk", high_risk_idx, "shap_waterfall_high.png"),
        ("low_risk",  low_risk_idx,  "shap_waterfall_low.png"),
    ]:
        risk_pct = p_shap[sample_idx] * 100
        true_lab = "Attack" if y_shap[sample_idx] == 1 else "Benign"

        try:
            expected_val = explainer.expected_value if explainer is not None else float(np.mean(raw_probs))
            sample_exp = shap.Explanation(
                values        = shap_vals[sample_idx],
                base_values   = expected_val,
                data          = X_shap.iloc[sample_idx].values,
                feature_names = selected,
            )
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(sample_exp, max_display=12, show=False)
            plt.title(
                f"CausalShield — {label.replace('_', ' ').title()} Flow\n"
                f"Risk Score: {risk_pct:.1f}%  |  True Label: {true_lab}",
                fontsize=11, fontweight="bold"
            )
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / fname, dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as wf_err:
            # Manual fallback bar chart
            shap_row = shap_vals[sample_idx]
            top_idx  = np.argsort(np.abs(shap_row))[::-1][:12]
            fig, ax  = plt.subplots(figsize=(9, 5))
            ax.barh(
                [selected[i] for i in top_idx[::-1]],
                [shap_row[i] for i in top_idx[::-1]],
                color=["#e74c3c" if shap_row[i] > 0 else "#3498db" for i in top_idx[::-1]],
            )
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("SHAP value")
            ax.set_title(
                f"CausalShield — {label.replace('_', ' ').title()} Flow\n"
                f"Risk Score: {risk_pct:.1f}%  |  True Label: {true_lab}",
                fontsize=11, fontweight="bold"
            )
            plt.tight_layout()
            fig.savefig(RESULTS_DIR / fname, dpi=150)
            plt.close()

        print(f"      {fname} saved")

    # ── 6d. ACE × SHAP scatter ────────────────────────────────────────────────
    scatter_df = importance_df.copy()
    colors_s   = [role_colors.get(r, "#95a5a6") for r in scatter_df["causal_role"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        scatter_df["abs_ace"], scatter_df["mean_abs_shap"],
        c=colors_s, s=80, edgecolors="white", linewidth=0.5
    )
    for _, row in scatter_df.head(10).iterrows():
        ax.annotate(
            row["feature"],
            (row["abs_ace"], row["mean_abs_shap"]),
            fontsize=6.5, textcoords="offset points", xytext=(3, 2)
        )
    ax.set_xlabel("|ACE| (do-calculus)", fontsize=11)
    ax.set_ylabel("Mean |SHAP|", fontsize=11)
    ax.set_title(
        "Causal Coherence: ACE vs SHAP\n"
        "(top-right = both causally & predictively important)",
        fontsize=11, fontweight="bold"
    )
    legend_handles = [mpatches.Patch(color=v, label=k) for k, v in role_colors.items()]
    ax.legend(handles=legend_handles, fontsize=8)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "shap_ace_scatter.png", dpi=150)
    plt.close()
    print("      shap_ace_scatter.png saved")

except Exception as viz_err:
    print(f"\n      [WARN] Visualisation error: {viz_err}")
    import traceback; traceback.print_exc()

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Phase 5 complete.")
print(f"  SHAP subset  : {len(X_shap)} samples")
print(f"  Top features by |SHAP| :")
for _, row in importance_df.head(5).iterrows():
    print(f"    {row['feature']:<35}  SHAP={row['mean_abs_shap']:.4f}  ({row['causal_role']})")
print("=" * 60)
