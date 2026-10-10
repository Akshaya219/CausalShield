"""
CausalShield - Phase 6: Streamlit Dashboard
============================================
Run with:  streamlit run phase6_dashboard.py

Tabs:
  1. Risk Scoring      — paste/upload a CSV row → get risk score + SHAP waterfall
  2. Causal Graph      — visualise PC graph, highlight causal features
    3. ACE Analysis      — bar chart of observational ACE proxy scores
  4. Model Report      — performance metrics, calibration, confusion matrix
  5. Feature Explorer  — SHAP beeswarm, ACE×SHAP scatter
    6. Baselines         — in-domain comparison vs Full-Feature & MI-Selected
    """

import json
import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "CausalShield",
    page_icon  = "🛡️",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
# __file__ is not defined in Jupyter/Kaggle notebooks — use cwd instead.
try:
    BASE = Path(__file__).parent
except NameError:
    BASE = Path.cwd()

CAUSAL_DIR    = BASE / "data/causal"
MODEL_DIR     = BASE / "models"
RESULTS_DIR   = BASE / "results"

# ── Load artifacts (cached) ───────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(str(MODEL_DIR / "xgb_model.json"))

    with open(MODEL_DIR / "calibrator.pkl", "rb") as f:
        calibrator = pickle.load(f)

    meta          = json.load(open(MODEL_DIR / "model_meta.json"))
    selected      = json.load(open(CAUSAL_DIR / "selected_features.json"))
    ace_data      = json.load(open(CAUSAL_DIR / "ace_scores.json"))
    graph_meta    = json.load(open(CAUSAL_DIR / "causal_graph_meta.json"))
    shap_meta     = json.load(open(RESULTS_DIR / "shap_meta.json"))
    shap_vals     = np.load(RESULTS_DIR / "shap_values.npy")
    shap_probs    = np.load(RESULTS_DIR / "shap_probs.npy")
    shap_labels   = np.load(RESULTS_DIR / "shap_labels.npy")
    shap_X        = pd.read_csv(RESULTS_DIR / "shap_test_subset.csv")
    feat_imp      = pd.read_csv(RESULTS_DIR / "feature_importance.csv")

    tier_meta     = json.load(open(MODEL_DIR / "tier_thresholds.json")) if (MODEL_DIR / "tier_thresholds.json").exists() else {}
    baseline_df   = pd.read_csv(RESULTS_DIR / "baseline_comparison.csv") if (RESULTS_DIR / "baseline_comparison.csv").exists() else pd.DataFrame()

    return {
        "booster"    : booster,
        "calibrator" : calibrator,
        "meta"       : meta,
        "selected"   : selected,
        "ace_data"   : ace_data,
        "graph_meta" : graph_meta,
        "shap_meta"  : shap_meta,
        "shap_vals"  : shap_vals,
        "shap_probs" : shap_probs,
        "shap_labels": shap_labels,
        "shap_X"     : shap_X,
        "feat_imp"   : feat_imp,
        "tier_meta"  : tier_meta,
        "baseline_df": baseline_df,
    }

@st.cache_resource
def load_shap_explainer(_booster):
    import json as _json
    try:
        cfg = _json.loads(_booster.save_config())
        bs  = cfg["learner"]["learner_model_param"]["base_score"]
        if isinstance(bs, str) and bs.startswith("["):
            cfg["learner"]["learner_model_param"]["base_score"] = str(float(bs.strip("[]")))
            _booster.load_config(_json.dumps(cfg))
    except Exception:
        pass
    try:
        import shap
        return shap.TreeExplainer(_booster)
    except ImportError:
        class NativeTreeExplainer:
            def __init__(self, booster):
                self.booster = booster
                self.expected_value = 0.0

            def shap_values(self, data):
                return self.booster.predict(data, pred_contribs=True)[:, :-1]

        return NativeTreeExplainer(_booster)

def predict_row(row_df, artifacts):
    import xgboost as xgb
    sel      = artifacts["selected"]
    booster  = artifacts["booster"]
    cal      = artifacts["calibrator"]
    thresh   = artifacts["meta"]["threshold"]

    # Ensure all columns exist
    for c in sel:
        if c not in row_df.columns:
            row_df[c] = 0.0

    X  = row_df[sel].astype(float)
    dm = xgb.DMatrix(X, feature_names=sel)
    raw = booster.predict(dm)
    prob = float(cal.transform(raw)[0])
    pred = int(prob >= thresh)
    return prob, pred, X

def shap_waterfall_fig(shap_row, feature_names, base_val, x_row, risk_pct, true_label=None):
    """Manual waterfall using matplotlib — avoids shap display issues in Streamlit."""
    import matplotlib.pyplot as plt

    top_idx = np.argsort(np.abs(shap_row))[::-1][:12]
    top_idx = top_idx[::-1]   # smallest to largest for barh

    fig, ax = plt.subplots(figsize=(9, 5))
    vals    = [shap_row[i] for i in top_idx]
    feats   = [f"{feature_names[i]}={x_row[i]:.3g}" for i in top_idx]
    colors  = ["#e74c3c" if v > 0 else "#3498db" for v in vals]

    ax.barh(feats, vals, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on log-odds of Attack)")

    title = f"Risk Score: {risk_pct:.1f}%"
    if true_label is not None:
        title += f"  |  True: {'Attack' if true_label == 1 else 'Benign'}"
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=60)
    st.title("CausalShield 🛡️")
    st.markdown("""
    **Explainable IoT Intrusion Risk Scoring**
    via Causal Graph-Guided Feature Selection

    ---
    **Dataset**: CICIoT2023 (binary: Benign vs Attack)
    **Causal discovery**: FCI (Fast Causal Inference), α = 0.05, PAG output
    **Feature scoring**: observational ACE proxy (ranking heuristic)
    **Model**: XGBoost + isotonic calibration
    **Explainability**: Tree SHAP

    ---
    *Dept. of Information Technology, Sona College of Technology, Salem*
    """)
    st.markdown("---")

    try:
        artifacts = load_artifacts()
        st.success(f"✅ Model loaded  \n{artifacts['meta']['n_features']} causal features")
        st.metric("Test ROC-AUC", f"{artifacts['meta']['test_roc_auc']:.4f}")
        macro_f1 = artifacts["meta"].get("test_macro_f1")
        st.metric("Test macro F1", f"{macro_f1:.4f}" if macro_f1 is not None else "N/A")
        st.metric("Threshold τ", f"{artifacts['meta']['threshold']:.2f}")
    except Exception as e:
        st.error(f"❌ Load error: {e}")
        st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🎯 Risk Scoring",
    "🔗 Causal Graph",
    "📊 ACE Analysis",
    "📈 Model Report",
    "🔍 Feature Explorer",
    "⚖️ Baselines",
])

# ═══════════════════════════════════════════════════════════════════
# TAB 1 — Risk Scoring
# ═══════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("🎯 Single Flow Risk Scoring")
    st.markdown("""
    Enter network flow feature values below.
    CausalShield will output a **calibrated intrusion risk score** and
    explain which causal features drove the prediction.
    """)

    sel      = artifacts["selected"]
    ace_data = artifacts["ace_data"]

    # Default values: median from SHAP subset
    shap_X = artifacts["shap_X"]
    defaults = {c: float(shap_X[c].median()) if c in shap_X.columns else 0.0 for c in sel}

    col_l, col_r = st.columns([1, 1])
    input_vals = {}

    n_half = len(sel) // 2
    with col_l:
        st.markdown("**Feature inputs (left)**")
        for feat in sel[:n_half]:
            ace_val = ace_data.get(feat, {}).get("ace", 0.0)
            label   = f"{feat}  *(ACE={ace_val:+.3f})*"
            input_vals[feat] = st.number_input(label, value=defaults[feat], key=f"f_{feat}", format="%.4f")

    with col_r:
        st.markdown("**Feature inputs (right)**")
        for feat in sel[n_half:]:
            ace_val = ace_data.get(feat, {}).get("ace", 0.0)
            label   = f"{feat}  *(ACE={ace_val:+.3f})*"
            input_vals[feat] = st.number_input(label, value=defaults[feat], key=f"f2_{feat}", format="%.4f")

    if st.button("🔍 Compute Risk Score", type="primary"):
        row_df = pd.DataFrame([input_vals])
        prob, pred, X_used = predict_row(row_df, artifacts)

        risk_pct = prob * 100
        col1, col2, col3 = st.columns(3)
        col1.metric("Risk Score", f"{risk_pct:.1f}%")
        col2.metric("Prediction", "🔴 Attack" if pred == 1 else "🟢 Benign")
        col3.metric("Threshold",  f"{artifacts['meta']['threshold']:.2f}")

        # Three-tier risk scoring
        tier_meta = artifacts.get("tier_meta", {})
        low_t     = tier_meta.get("low_threshold",  0.25)
        high_t    = tier_meta.get("high_threshold", 0.65)

        if prob >= high_t:
            tier = "HIGH"; color = "#e74c3c"; icon = "🔴"; action = "BLOCK / ALERT immediately"
        elif prob >= low_t:
            tier = "MEDIUM"; color = "#f39c12"; icon = "🟡"; action = "INVESTIGATE — suspicious flow"
        else:
            tier = "LOW"; color = "#2ecc71"; icon = "🟢"; action = "MONITOR — likely benign"

        col1b, col2b = st.columns(2)
        col1b.metric("Risk Tier", f"{icon} {tier}")
        col2b.metric("Action",    action)

        st.markdown(f"""
        <div style="background:linear-gradient(to right, {color} {risk_pct:.0f}%, #ecf0f1 {risk_pct:.0f}%);
                    height:20px; border-radius:10px; margin:10px 0;"></div>
        <p style="text-align:center; font-size:16px; color:{color}; font-weight:bold;">
            {icon} {tier} RISK — {action}
        </p>
        """, unsafe_allow_html=True)

        # SHAP explanation
        try:
            explainer = load_shap_explainer(artifacts["booster"])
            import xgboost as xgb
            X_arr   = X_used.values.astype(float)
            sv      = explainer.shap_values(xgb.DMatrix(X_used, feature_names=sel))
            if sv.ndim > 1:
                sv = sv[0]

            st.subheader("📊 SHAP Explanation — Why this score?")
            fig = shap_waterfall_fig(sv, sel, explainer.expected_value, X_arr[0], risk_pct)
            st.pyplot(fig)

            # Top drivers table
            top_idx = np.argsort(np.abs(sv))[::-1][:8]
            exp_df  = pd.DataFrame({
                "Feature"     : [sel[i] for i in top_idx],
                "Value"       : [f"{X_arr[0][i]:.4g}" for i in top_idx],
                "SHAP Impact" : [f"{sv[i]:+.4f}" for i in top_idx],
                "Direction"   : ["↑ Risk" if sv[i] > 0 else "↓ Risk" for i in top_idx],
                "ACE"         : [f"{ace_data.get(sel[i],{}).get('ace',0):+.4f}" for i in top_idx],
            })
            st.dataframe(exp_df, use_container_width=True)

        except Exception as shap_err:
            st.warning(f"SHAP explanation error: {shap_err}")

    # ── Sample flows ──────────────────────────────────────────────────────────
    with st.expander("📋 Load a sample flow from the test set"):
        sample_type = st.radio("Pick a sample", ["High-risk (Attack)", "Low-risk (Benign)"], horizontal=True)
        if st.button("Load sample"):
            probs  = artifacts["shap_probs"]
            labels = artifacts["shap_labels"]
            shap_X = artifacts["shap_X"]
            if "Attack" in sample_type:
                idx = int(np.argmax(probs))
            else:
                idx = int(np.argmin(probs))
            sample_row = shap_X.iloc[idx]
            st.write(f"Sample index: {idx}  |  True label: {'Attack' if labels[idx]==1 else 'Benign'}  |  Risk prob: {probs[idx]:.3f}")
            st.dataframe(sample_row.to_frame().T, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — Causal Graph
# ═══════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("🔗 Causal Graph — FCI Partial Ancestral Graph (PAG)")

    gm = artifacts["graph_meta"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alpha used",       gm.get("alpha_used"))
    c2.metric("Subset size",      f"{gm.get('subset_size',0):,}")
    c3.metric("Graph edges",      gm.get("n_edges_total", 0))
    c4.metric("Causal features",  gm.get("n_causal_features", 0))

    graph_img = CAUSAL_DIR / "causal_graph.png"
    if graph_img.exists():
        st.image(str(graph_img), caption="FCI PAG — Orange=Causal Feature, Red=Attack Target", width="stretch")
    else:
        st.info("causal_graph.png not found — run Phase 2 first.")

    st.subheader("Causally-linked Features")
    causal_feats = gm.get("causal_features", [])
    if causal_feats:
        st.success(f"**{len(causal_feats)} features** are causally linked to the Attack node:")
        st.write(", ".join(causal_feats))
    else:
        st.warning("No features directly linked to Attack in the graph.")

    st.subheader("Edge List")
    edges = gm.get("edges", [])
    if edges:
        edge_df = pd.DataFrame(edges)
        st.dataframe(edge_df, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — ACE Analysis
# ═══════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("📊 ACE Proxy Analysis")
    st.markdown("""
    **Observational ACE proxy:** E[Y|X=1] − E[Y|X=0] (continuous features binarized at the training median). This is a ranking heuristic, not an identified causal effect, because latent confounding rules out backdoor adjustment.

    Positive ACE → feature increases attack probability when intervened on.
    Negative ACE → feature decreases attack probability.
    """)

    ace_img = CAUSAL_DIR / "ace_scores.png"
    if ace_img.exists():
        st.image(str(ace_img), width="stretch")

    ace_data = artifacts["ace_data"]
    ace_df = pd.DataFrame([
        {
            "Feature"   : k,
            "ACE"       : round(v["ace"], 5),
            "|ACE|"     : round(v["abs_ace"], 5),
            "Method"    : v["method"],
            "Confounders": ", ".join(v["confounders"]) if v["confounders"] else "none",
        }
        for k, v in ace_data.items()
    ]).sort_values("|ACE|", ascending=False).reset_index(drop=True)

    st.dataframe(ace_df, use_container_width=True)

    threshold = st.slider("ACE proxy cutoff (filter)", 0.0, 0.3, 0.05, 0.01)
    filtered  = ace_df[ace_df["|ACE|"] >= threshold]
    st.info(f"**{len(filtered)} features** pass |ACE_obs| ≥ {threshold}")
    st.caption("Cutoff: |ACE_obs| ≥ 0.05. Duration (|proxy| = 0.0430) is retained because of its directed edge (→).")
    st.dataframe(filtered, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 4 — Model Report
# ═══════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("📈 Model Performance Report")

    meta = artifacts["meta"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC",  f"{meta['test_roc_auc']:.4f}")
    c2.metric("PR-AUC",   f"{meta['test_pr_auc']:.4f}")
    c3.metric("F1 Score", f"{meta['test_f1']:.4f}")
    c4.metric("Threshold", f"{meta['threshold']:.2f}")

    st.markdown("---")

    report_path = RESULTS_DIR / "classification_report.txt"
    if report_path.exists():
        st.subheader("Classification Report")
        st.code(open(report_path).read())

    col1, col2 = st.columns(2)
    for fname, caption, col in [
        ("roc_curve.png",       "ROC Curve",        col1),
        ("confusion_matrix.png","Confusion Matrix",  col2),
    ]:
        p = RESULTS_DIR / fname
        if p.exists():
            col.image(str(p), caption=caption, width="stretch")

    col3, col4 = st.columns(2)
    for fname, caption, col in [
        ("calibration_curve.png", "Calibration Curve",  col3),
        ("training_history.png",  "Training History",   col4),
    ]:
        p = RESULTS_DIR / fname
        if p.exists():
            col.image(str(p), caption=caption, width="stretch")

    if meta["test_roc_auc"] > 0.99:
        st.warning("""
        **Note:** AUC > 0.99 is common for CICIoT2023 because network flow statistics
        (Rate, IAT, packet counts) are computed per-flow and are naturally discriminative.
        This does **not** mean the model is overfit — but verify there is no feature
        derived *after* knowing the label. The causal filtering makes this model
        more trustworthy than a naive feature-selection approach.
        """)
    with st.expander("Model Parameters"):
        st.json(meta["xgb_params"])


# ═══════════════════════════════════════════════════════════════════
# TAB 5 — Feature Explorer
# ═══════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("🔍 Feature Explorer")
    feat_imp = artifacts["feat_imp"]

    st.subheader("SHAP vs ACE — Causal Coherence Scatter")
    scatter_img = RESULTS_DIR / "shap_ace_scatter.png"
    if scatter_img.exists():
        st.image(str(scatter_img), width="stretch")

    col1, col2 = st.columns(2)
    bee_img = RESULTS_DIR / "shap_beeswarm.png"
    sum_img = RESULTS_DIR / "shap_summary.png"
    if sum_img.exists():
        col1.image(str(sum_img), caption="SHAP Summary (mean |SHAP|)", width="stretch")
    if bee_img.exists():
        col2.image(str(bee_img), caption="SHAP Beeswarm", width="stretch")

    st.subheader("Waterfall Plots — Extreme Flows")
    col3, col4 = st.columns(2)
    wf_high = RESULTS_DIR / "shap_waterfall_high.png"
    wf_low  = RESULTS_DIR / "shap_waterfall_low.png"
    if wf_high.exists():
        col3.image(str(wf_high), caption="Highest-risk flow", width="stretch")
    if wf_low.exists():
        col4.image(str(wf_low), caption="Lowest-risk flow", width="stretch")

    st.subheader("Feature Importance Table")
    st.dataframe(
        feat_imp[["feature", "mean_abs_shap", "ace", "abs_ace", "causal_role", "rank_shap"]],
        use_container_width=True
    )


# ═══════════════════════════════════════════════════════════════════
# TAB 6 — Baseline Comparison
# ═══════════════════════════════════════════════════════════════════
with tabs[5]:
    st.header("⚖️ Baseline Comparison")
    st.markdown("""
    CausalShield compared against two baselines using identical XGBoost
    hyperparameters, isotonic calibration, and F1-optimal thresholding.

    | Model | Feature Selection |
    |---|---|
    | **Full-Feature XGBoost** | All features (no selection) |
    | **MI-Selected XGBoost** | Top-k mutual information features |
    | **CausalShield** | ACE-guided causal feature selection |
    """)

    baseline_df = artifacts.get("baseline_df", pd.DataFrame())
    if not baseline_df.empty:
        st.subheader("Results Table")
        st.dataframe(baseline_df, use_container_width=True)

        # Highlight CausalShield row
        best_f1_model = baseline_df.loc[baseline_df["f1"].idxmax(), "model"]
        st.success(f"Best F1: **{best_f1_model}**  ({baseline_df['f1'].max():.4f})")

        img = RESULTS_DIR / "baseline_comparison.png"
        if img.exists():
            st.image(str(img), width="stretch")
    else:
        st.warning("Run `phase4c_baselines.py` first to generate baseline results.")

    with st.expander("How baselines are computed"):
        st.markdown("""
        - **Same hyperparameters**: eta=0.05, max_depth=6, subsample=0.8 for all models
        - **Same calibration**: isotonic regression on validation set
        - **Same threshold**: F1-optimal on validation set per model
        - **Same data splits**: identical train/val/test CSVs
        - MI features: top-k by `mutual_info_classif` where k = number of causal features
        """)

