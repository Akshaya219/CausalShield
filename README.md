# CausalShield: Causal Graph-Guided Feature Selection for Explainable IoT Intrusion Risk Scoring

CausalShield is a causally-grounded machine learning framework for IoT network intrusion detection. By utilizing causal discovery and observational effect proxies, CausalShield selects invariant, domain-relevant features from high-dimensional network flow data (CICIoT2023), training calibrated risk-scoring models that generalize across domain shifts (ToN-IoT).

---

## 📌 Technical Pipeline

```
Raw Flow Data (CICIoT2023)
       │
       ▼
1. Preprocessing & Resampling
       │
       ▼
2. Causal Discovery (FCI Algorithm -> PAG)
       │
       ▼
3. ACE Proxy Estimation (Observational Proxy Selection)
       │
       ▼
4. XGBoost Scoring + Isotonic Calibration
       │
       ▼
5. Tree SHAP Explainability & Risk Attribution
       │
       ▼
6. Three-Tier Risk Thresholding (Low / Medium / High)
       │
       ▼
7. Cross-Dataset Transfer & Evaluation (ToN-IoT)
```

---

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Running the Pipeline

Run the full end-to-end pipeline:

```bash
python run_pipeline.py
```

Or run individual phases:

```bash
python run_pipeline.py --from-phase 3
python run_pipeline.py --skip-phase 4c
```

### Launching the Interactive Dashboard

```bash
streamlit run dashboard/app.py
```

---

## 📁 Repository Structure

```
├── dashboard/
│   └── app.py                      # Interactive Streamlit dashboard
├── docs/
│   └── causal_graph.png            # Visualisation of FCI Partial Ancestral Graph
├── models/
│   ├── xgb_model.json              # Trained XGBoost binary classifier
│   ├── calibrator.pkl              # Isotonic calibrator object
│   ├── model_meta.json             # Hyperparameters, metrics, threshold metadata
│   └── tier_thresholds.json        # Three-tier risk thresholds (low/medium/high)
├── archive/
│   └── phase4b_category_classifier.py # (archived) Multi-class attack classifier
├── pipeline/
│   ├── preprocess.py               # Data cleaning and train/val/test splits
│   ├── phase2_causal_discovery.py  # Fast Causal Inference (FCI) execution
│   ├── phase3_ace_filtering.py     # ACE proxy evaluation and feature selection
│   ├── phase4_xgboost_scoring.py   # Model training with isotonic calibration
│   ├── phase4c_baselines.py        # Benchmark against Full-Feature & MI models
│   ├── phase4d_three_tier_scoring.py  # Risk tier assignment and coverage analysis
│   ├── phase5_shap_explainability.py  # Tree SHAP attribution & coherence plots
│   └── phase7_transfer_toniot.py   # ToN-IoT few-shot cross-dataset transfer
├── verification/
│   ├── build_toniot_cross.py       # Prepares ToN-IoT transfer evaluation dataset
│   ├── check_ace_binarization.py   # Validates binarization thresholds
│   ├── check_cross_inputs.py       # Validates transfer dataset schema mapping
│   ├── check_fci_convention.py     # Verifies FCI PAG edge convention
│   ├── check_fci_repro.py          # Verifies FCI graph reproduction
│   ├── check_fci_stability.py      # Resampling stability checks across 5 seeds
│   ├── check_toniot_duration.py    # Checks duration feature scaling on ToN-IoT
│   ├── check_toniot_source.py      # Audits raw ToN-IoT source files
│   ├── check_toniot_split.py       # Checks 1% calibration / 99% heldout split
│   ├── decode_saved_graph.py       # Decodes adjacency matrices into edge lists
│   ├── recompute_ace_proxy.py      # Computes observational proxy ACE values
│   ├── recompute_ace_scores.py     # Standardized ACE recomputation script
│   ├── regenerate_graph_png.py     # Graph rendering script
│   ├── transfer_reproduce.py       # ToN-IoT transfer reproduction & ablation
│   └── verify_test_metrics.py      # Independent evaluation of test set metrics
├── data/                           # Data artifacts (processed splits & causal metadata)
├── results/                        # Evaluation metrics, SHAP outputs, & figures
├── requirements.txt                # Python package dependencies
└── run_pipeline.py                 # Pipeline orchestrator CLI
```

---

## 📊 Paper Artifact Mapping

The table below maps each table and figure from the research paper to its corresponding producing script in this repository:

| Paper Item | Description | Producing Script / Command |
|---|---|---|
| **Table I** | CICIoT2023 Feature Summary & Preprocessing | `pipeline/preprocess.py` |
| **Figure 3** | ACE Proxy Bar Chart | `verification/regenerate_ace_figure.py` |
| **Figure 4 / Table II** | FCI PAG Causal Graph & Tree SHAP Attributions | `pipeline/phase2_causal_discovery.py`, `pipeline/phase5_shap_explainability.py` |
| **Table III** | Observational ACE Proxy Scores | `verification/recompute_ace_scores.py`, `pipeline/phase3_ace_filtering.py` |
| **Table IV / VI** | In-Domain Model Metrics & Baseline Comparison | `pipeline/phase4_xgboost_scoring.py`, `pipeline/phase4c_baselines.py` |
| **Table V** | Three-Tier Risk Scoring Thresholds & Coverage | `pipeline/phase4d_three_tier_scoring.py` |
| **Table VII - X** | ToN-IoT Few-Shot Transfer & Ablation Study | `pipeline/phase7_transfer_toniot.py`, `verification/transfer_reproduce.py` |
| **Dashboard** | Web Interactive Operations & Risk Scoring | `streamlit run dashboard/app.py` |

---

## ⚠️ Known Limitations

1. **Causal Graph Stability across Subsets**: Across 5 resampled training subsets, 6 of 7 Attack-adjacent features appear in 5/5 runs, while `rst_flag_number` appears adjacent in 3 of 5 runs (60% stability). Overall, 22 of 70 PAG edges remain completely identical across all subsamples.
2. **Approximate Cross-Dataset Feature Mapping**: Cross-dataset evaluation on ToN-IoT utilizes an approximate feature schema mapping (`Duration` <- `Flow Duration`, `Variance` <- `Packet Length Variance`, `RST features` <- `RST Flag Count`, `HTTPS` set to 0).
3. **Observational Proxy Binarization**: Binary flag features (`HTTPS`, `ack_flag_number`, `psh_flag_number`, `rst_flag_number`) and `rst_count` are binarized at `x > 0`. Continuous features (`Variance` and `Duration`) are binarized at the sample median.
4. **Dormant Inverted Parent Decoding**: Graph decoding tools handle legacy orientation conventions where `parents` arrays in causal discovery outputs represent inverted edge directions.
