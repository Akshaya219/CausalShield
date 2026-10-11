# CausalShield

**Explainable IoT Intrusion Risk Scoring via Causal Graph-Guided Feature Selection**

CausalShield is an intrusion detection framework for IoT networks. Instead of picking features by statistical correlation, it uses **causal discovery (FCI)** to discover a compact set of protocol-level invariant features, scores every network flow with an **XGBoost** model paired with **isotonic calibration**, and explains each decision with **Tree SHAP** and operational three-tier triage policies.

---

## ⚡ The Idea in 60 Seconds

| Dimension | Summary |
|---|---|
| **Problem** | ML intrusion detectors achieve near-perfect scores in-domain but degrade dramatically under domain shifts on unseen networks. |
| **Root Cause** | Correlation-based feature selection (Pearson, mutual information, ANOVA) picks up dataset capture artifacts: bandwidth, packet timing, and host-specific behaviors. |
| **Our Approach** | Run **FCI** (Fast Causal Inference) to learn a Partial Ancestral Graph (PAG), retain features causally connected to the attack label, and train a compact, calibrated classifier. |
| **Result** | 7 of 39 features kept (**82% fewer**), achieving **ROC-AUC 0.9393** in-domain on CICIoT2023, while generalizing zero-shot/few-shot to ToN-IoT (**Reflected ROC-AUC 0.8125**). |
| **Operational Impact** | Every flow risk score is paired with a local SHAP waterfall explanation and mapped to SOC risk tiers (Low / Medium / High). |

---

## 🔄 Technical Pipeline

```
CICIoT2023 Raw Flows (6.36M)
            │
            ▼
1. Preprocessing       39 features → 34 clean candidates, stratified train/val/test split
            │
            ▼
2. Causal Discovery    FCI on 5,000 balanced flows → PAG (70 edges, 47 bidirected)
            │
            ▼
3. Feature Selection   Attack-adjacent features + observational ACE proxy ranking → 7 features
            │
            ▼
4. Model Training      XGBoost + Isotonic Calibration (threshold τ = 0.61)
            │
            ▼
5. Risk Attribution    Calibrated Score → Three-Tier SOC Policy → SHAP Waterfall Explanation
            │
            ▼
6. Domain Transfer     Frozen model evaluation on ToN-IoT (1% calibration split, 99% held-out)
```

### The 7 Selected Causal Features

| Feature | Meaning | PAG Edge | Causal Interpretation |
|---|---|---|---|
| `rst_count` | Number of TCP RST flags | Latent confounding ($\leftrightarrow$) | Common unobserved network state |
| `Variance` | Packet size variance | Latent confounding ($\leftrightarrow$) | Payload distribution pattern |
| `Duration` | Flow duration | Direct ancestor ($\rightarrow$) | Causal ancestor of intrusion event |
| `ack_flag_number` | TCP ACK flag | Latent confounding ($\leftrightarrow$) | Protocol handshake behavior |
| `HTTPS` | Encrypted web traffic indicator | Latent confounding ($\leftrightarrow$) | Service type confounding |
| `rst_flag_number` | TCP RST flag | Latent confounding ($\leftrightarrow$) | Abrupt connection termination |
| `psh_flag_number` | TCP PSH flag | Latent confounding ($\leftrightarrow$) | Push flag signaling data transfer |

> A **PAG** (Partial Ancestral Graph) is the output of FCI. A bidirected edge ($\leftrightarrow$) indicates an unmeasured common cause (e.g., host/device load or topology state not recorded in the flow record).

---

## 📈 Key Results

### 1. In-Domain Performance (CICIoT2023 Held-Out Test Set, $n = 59,438$)

| Model | Features | F1 Attack | F1 Benign | Macro F1 | ROC-AUC | PR-AUC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Full-Feature XGBoost | 39 | 0.9813 | 0.9021 | 0.9417 | 0.9944 | 0.9989 |
| Mutual-Information XGBoost | 7 | 0.9786 | 0.8914 | 0.9350 | 0.9925 | 0.9986 |
| **CausalShield (Proposed)** | **7** | **0.9252** | **0.6874** | **0.8063** | **0.9393** | **0.9875** |

- **5-Fold Cross-Validation**: ROC-AUC = $0.9381 \pm 0.0009$ (demonstrating cross-fold stability).
- **Inference Latency**: **1.42–1.64 ms** per flow (over 2× faster throughput than the full 39-feature model).

### 2. Cross-Dataset Transfer (ToN-IoT, 6.90M Held-Out Flows, Frozen Model)

#### Standard / Normal Models vs. CausalShield on Out-of-Domain ToN-IoT Test Set

| Model Architecture | Features | Feature Selection Paradigm | Cross-Dataset ROC-AUC | Macro F1 | Benign F1 (Reliability) | Attack F1 | Cross-Domain Stability |
|---|:---:|---|:---:|:---:|:---:|:---:|:---:|
| **Full-Feature XGBoost (Standard)** | 39 | None (All captured features) | **0.4976** | 0.5601 | 0.3035 | 0.8168 | ❌ **Collapsed** (Random Guess) |
| **Mutual Information XGBoost** | 7 | Statistical Correlation (MI) | **0.3688** | 0.4668 | 0.2034 | 0.7302 | ❌ **Inverted** (Severe False Alarms) |
| **CausalShield (Proposed)** | **7** | **FCI Causal Discovery + ACE Proxy** | **0.8125** | **0.7817** | **0.7484** | **0.8149** | ✅ **High Generalization (2.5× to 3.7× higher Benign F1)** |

#### Operating Points on ToN-IoT (CausalShield Frozen Model)

| Operating Point | Threshold ($\tau$) | F1 Attack | F1 Macro | Benign False-Positive Rate |
|---|:---:|:---:|:---:|:---:|
| **Zero-Shot (Default)** | 0.610 | 0.7195 | 0.3598 | 100.0% |
| **Recalibrated ($\tau^*$)** | 0.023 | 0.8149 | 0.7817 | 27.58% |
| **Enterprise ($FPR \le 5\%$)** | 0.080 | 0.3567 | 0.5061 | 2.56% |

- **Reflected ROC-AUC**: **0.8125** on held-out evaluation flows.
- **Few-Shot Split**: Only a strict 1% calibration split ($n = 48,472$) was used to fix score polarity and tune thresholds; the remaining 99% ($n = 4.80\text{M}$) was completely held out.


### 3. Three-Tier SOC Risk Policy

| Risk Tier | Score Range | SOC Action | Attack Share (Test Set) |
|---|:---:|:---:|:---:|
| **LOW** | $< 0.250$ | Monitor | 16.2% – 17.2% |
| **MEDIUM** | $0.250 – 0.650$ | Investigate | 45.1% – 46.3% |
| **HIGH** | $\ge 0.650$ | Block / Immediate Alert | 95.2% – 95.5% |

---

## 💡 Honest Takeaways & Limitations

- **FCI vs. Hand-Crafted Selection**: Retraining on the same 7 features without FCI yields identical performance ($\Delta\text{AUC} = 0.000$). FCI discovers features consistent with domain expertise, but provides an automated, **reproducible, traceable, and auditable** selection mechanism.
- **Accuracy Trade-Off**: Mutual-information selection achieves higher in-domain AUC (0.9925 vs 0.9393), but its top features (packet inter-arrival times, total sizes) heavily overfit testbed capture mechanics. We intentionally trade ~5.32 pp AUC for causal stability.
- **Observational ACE Proxy**: ACE proxy scores serve as ranking heuristics rather than identified interventional effects, as latent confounding prevents formal backdoor adjustment.
- **SHAP Interpretations**: SHAP attribution values show what the trained model relies upon, which may differ from interventional causal effects.
- **Stability**: Across 5 resampled training subsets, 6 of 7 features appear adjacent to `Attack` in 5/5 runs, while `rst_flag_number` appears in 3/5 runs.
- **Cross-Dataset Mapping**: `HTTPS` does not exist in ToN-IoT and is zero-filled; flow duration scale shifts contribute to the elevated benign FPR prior to enterprise thresholding.

---

## 🗄️ Datasets

| Dataset | Role | Details |
|---|---|---|
| **CICIoT2023** | In-domain training, validation, testing, causal discovery | 33 attack types grouped into 7 categories + benign network traffic. Binary target (0 = benign, 1 = attack). |
| **ToN-IoT** | Out-of-domain cross-dataset transfer | Heterogeneous IoT/IIoT network captures with 4.85M attack and 2.12M benign flows. Never used for model training. |

*Datasets are downloaded separately from their original sources and processed into `data/`.*

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd v2

# Create virtual environment and install dependencies
python -m venv venv
.\venv\Scripts\activate          # Windows PowerShell/CMD
# source venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
```

### 2. Launch the Interactive Dashboard

Launch the Streamlit web dashboard to inspect risk scoring, SHAP waterfall plots, causal graphs, and transfer reports:

```bash
streamlit run dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 3. Running the Pipeline

Run the complete pipeline end-to-end:

```bash
python run_pipeline.py
```

Or execute or resume specific phases:

```bash
python run_pipeline.py --from-phase 3
python run_pipeline.py --skip-phase 4c
```

---

## 📁 Repository Structure

```
├── data/
│   ├── train/                         # Training & validation splits (X_train, y_train, X_val, y_val, etc.)
│   ├── test/                          # Held-out testing splits (in-domain CICIoT2023 test set)
│   │   └── toniot/                    # Out-of-domain ToN-IoT few-shot transfer evaluation set
│   └── causal/                        # FCI PAG graph metadata, adjacency, and ACE scores
├── dashboard/
│   └── app.py                         # Interactive Streamlit dashboard
├── docs/
│   └── causal_graph.png               # Rendered FCI Partial Ancestral Graph
├── models/
│   ├── xgb_model.json                 # Trained XGBoost binary booster
│   ├── calibrator.pkl                 # Fitted isotonic calibrator
│   ├── model_meta.json                # Hyperparameters, evaluation metrics, thresholds
│   └── tier_thresholds.json           # Three-tier threshold cutoffs and labels
├── pipeline/
│   ├── 01_preprocess.py               # Step 1: Preprocessing, cleaning, stratified splitting
│   ├── 02_causal_discovery.py         # Step 2: Fast Causal Inference (FCI) algorithm
│   ├── 03_ace_filtering.py            # Step 3: Observational ACE proxy feature ranking
│   ├── 04_xgboost_training.py         # Step 4: XGBoost training and isotonic calibration
│   ├── 05_baseline_comparisons.py     # Step 5: Benchmark models (Full-Feature & MI-Selected)
│   ├── 06_three_tier_scoring.py       # Step 6: Three-tier SOC risk threshold tuning
│   ├── 07_shap_explainability.py      # Step 7: Tree SHAP explanations and attribution plots
│   └── 08_toniot_transfer.py          # Step 8: Few-shot cross-dataset transfer evaluation
├── verification/
│   ├── build_toniot_cross.py          # Prepares ToN-IoT cross-evaluation dataset
│   ├── check_ace_binarization.py      # Validates proxy binarization thresholds
│   ├── check_cross_inputs.py          # Verifies cross-dataset schema mapping
│   ├── check_fci_convention.py        # Verifies FCI PAG edge conventions
│   ├── check_fci_repro.py             # Reproduces FCI graph edges
│   ├── check_fci_stability.py         # Evaluates graph stability across 5 random seeds
│   ├── check_toniot_duration.py       # Audits feature scaling differences
│   ├── check_toniot_source.py         # Inspects raw ToN-IoT source distributions
│   ├── check_toniot_split.py          # Validates 1% calibration / 99% evaluation split
│   ├── decode_saved_graph.py          # Decodes adjacency matrix into legible edge lists
│   ├── generate_cross_dataset_comparison_chart.py # Cross-dataset model benchmark chart
│   ├── recompute_ace_proxy.py         # Observational proxy recomputation
│   ├── recompute_ace_scores.py        # Standardized ACE scoring verification
│   ├── regenerate_ace_figure.py       # Regenerates Figure 3 (ACE bar chart)
│   ├── regenerate_graph_png.py        # Regenerates Figure 4 (causal graph)
│   ├── transfer_reproduce.py          # Reproduces transfer and ablation metrics
│   └── verify_test_metrics.py         # Automated verification of test set metrics
├── results/                           # Metric logs, SHAP matrices, evaluation plots
├── calibration_utils.py               # Custom calibration utilities for model loading
├── run_pipeline.py                    # Master pipeline runner CLI
└── requirements.txt                   # Dependency requirements
```

---

## 📊 Paper Artifact Mapping

| Paper Item | Description | Producing Script / Command |
|---|---|---|
| **Table I** | CICIoT2023 Feature Summary & Cleaning | `pipeline/01_preprocess.py` |
| **Figure 3** | ACE Proxy Ranking Bar Chart | `verification/regenerate_ace_figure.py` |
| **Figure 4 / Table II** | FCI PAG Causal Graph & Edge Attributions | `pipeline/02_causal_discovery.py`, `pipeline/07_shap_explainability.py` |
| **Table III** | Observational ACE Proxy Scores | `verification/recompute_ace_scores.py`, `pipeline/03_ace_filtering.py` |
| **Table IV / VI** | In-Domain Metrics & Baseline Benchmark | `pipeline/04_xgboost_training.py`, `pipeline/05_baseline_comparisons.py` |
| **Table V** | Three-Tier Risk Thresholds & SOC Triage | `pipeline/06_three_tier_scoring.py` |
| **Table VII–X** | ToN-IoT Few-Shot Transfer & Ablation Study | `pipeline/08_toniot_transfer.py`, `verification/transfer_reproduce.py` |
| **Figure 10–12 / Benchmark** | Cross-Dataset Transfer Benchmark Chart | `verification/generate_cross_dataset_comparison_chart.py` |
| **Dashboard** | Interactive Single-Flow Risk Scoring & Explainability | `streamlit run dashboard/app.py` |

---

## 👥 Authors

Gokul Raj, Giridharan K, Akshaya P, Harini S  
*Sona College of Technology, Salem, India*

## 📜 Citation

If you use this work, please cite the CausalShield paper:

```bibtex
@article{causalshield2026,
  title={CausalShield: Causal Graph-Guided Feature Selection for Explainable IoT Intrusion Risk Scoring},
  author={Raj, Gokul and K, Giridharan and P, Akshaya and S, Harini},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License. Dataset licenses are governed by their respective original authors (CICIoT2023 and ToN-IoT).
