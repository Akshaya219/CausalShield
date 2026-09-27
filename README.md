# CausalShield

CausalShield is an explainable IoT intrusion detection system that combines causal graph discovery, average causal effect (ACE) feature filtering, XGBoost-based risk scoring, and SHAP-based explainability to detect malicious traffic with improved robustness and interpretability.

The project is built around the idea that not all network-flow features are equally useful. Instead of using a large raw feature set, it first identifies which features are causally relevant to the Attack target and then trains a compact, transparent model on those features.

## Repository Setup

This repository contains the source code and documentation for the CausalShield pipeline. Large datasets, local model files, generated reports, and virtual environments are intentionally excluded from Git through `.gitignore`. Keep those artifacts locally or provide them through a separate data store before running the pipeline.

Create an environment and install the dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
 .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The expected local data layout is:

```text
CICIOT23/{train,validation,test}/
CICIDS2017.csv
CIC-ToN-IoT-V2.parquet
data/cross_val/
```

The pipeline creates `data/processed/`, `models/`, `results/`, and `outputs/` during execution. These folders are generated artifacts and are not part of the Git upload.

---

## Project Goal

The main goals of this project are:

- detect malicious IoT traffic from network flow features,
- reduce dimensionality using causal feature selection,
- improve interpretability through causal reasoning and SHAP values,
- evaluate model performance under in-domain and cross-dataset settings,
- provide a dashboard for interactive model inspection and scoring.

---

## Core Idea

This project follows a causal-inference-driven machine learning pipeline:

1. Preprocess network flow data.
2. Learn a causal graph from the training data using FCI (Fast Causal Inference).
3. Identify features that are causally linked to the Attack node.
4. Estimate the Average Causal Effect (ACE) of each candidate feature.
5. Keep only the most causally important features.
6. Train an XGBoost classifier on those features.
7. Calibrate output probabilities.
8. Explain model outputs using SHAP.
9. Evaluate transferability on unseen datasets.

This makes the system more resilient to spurious correlations and more understandable than a black-box model trained on all features.

---

## Project Workflow

### Phase 2: Causal Graph Discovery
File: `phase2_causal_discovery.py`

- Loads processed training data.
- Removes near-constant and duplicate columns.
- Builds a causal graph using the FCI algorithm.
- Identifies which features are causally connected to the Attack target.
- Saves causal artifacts in `data/causal/`.

Generated outputs:
- `data/causal/causal_adjacency.npy`
- `data/causal/causal_feature_names.json`
- `data/causal/causal_target_features.json`
- `data/causal/causal_graph_meta.json`
- `data/causal/causal_graph.png`

### Phase 3: ACE Feature Filtering
File: `phase3_ace_filtering.py`

- Loads the causal graph metadata.
- Estimates ACE for each feature using a backdoor-adjustment-style approach.
- Keeps only features whose causal effect passes the threshold.
- Saves the selected feature list and filtered datasets.

Generated outputs:
- `data/causal/ace_scores.json`
- `data/causal/selected_features.json`
- `data/processed/X_train_causal.csv`
- `data/processed/X_val_causal.csv`
- `data/processed/X_test_causal.csv`

### Phase 4: XGBoost Risk Scoring
File: `phase4_xgboost_scoring.py`

- Trains a binary XGBoost intrusion risk model on the causally selected features.
- Uses 5-fold cross-validation for stability checks.
- Uses probability calibration with isotonic regression.
- Evaluates on a held-out test set.
- Saves a trained model and evaluation metrics.

Generated outputs:
- `models/xgb_model.json`
- `models/calibrator.pkl`
- `models/model_meta.json`
- `results/cv_results.json`
- `results/classification_report.txt`
- `results/roc_curve.png`
- `results/calibration_curve.png`

### Phase 4b: Attack Category Classifier
File: `phase4b_category_classifier.py`

- Trains a second XGBoost model to classify the attack family or category.
- Uses the same causal feature set.

### Phase 4c: Baseline Comparisons
File: `phase4c_baselines.py`

- Compares the causal model against baseline models.
- Includes a full-feature model and MI-based feature selection baseline.

### Phase 4d: Three-Tier Risk Scoring
File: `phase4d_three_tier_scoring.py`

- Produces a three-tier risk score system (low / moderate / high risk).
- Useful for operational monitoring and threshold-based alerting.

### Phase 5: SHAP Explainability
File: `phase5_shap_explainability.py`

- Computes SHAP values for the trained model.
- Explains feature influence and causal role.
- Produces feature importance summaries and plots.

Generated outputs:
- `results/shap_values.npy`
- `results/shap_meta.json`
- `results/feature_importance.csv`
- `results/shap_test_subset.csv`

### Phase 6: Streamlit Dashboard
File: `phase6_dashboard.py`

- Launches a dashboard for:
  - single-row risk scoring,
  - causal graph inspection,
  - ACE analysis,
  - model metrics,
  - SHAP feature explorer,
  - batch scoring,
  - baseline comparison,
  - category classification,
  - cross-dataset transfer analysis.

Run with:

```bash
streamlit run phase6_dashboard.py
```

---

## Datasets Used

### CICIoT2023
This is the main training and evaluation dataset for the project.

The pipeline is designed around network-flow data from CICIoT2023, where flow features such as packet counts, timing metrics, flags, and protocol indicators are used to identify attacks.

### Cross-dataset evaluation
The project also evaluates transfer to other network datasets, including:

- CICIDS2017
- ToN-IoT

The cross-dataset analysis tests how well the causal features generalize when the environment changes.

---

## Key Modeling Strategy

The project uses a combination of:

- causal discovery (FCI)
- backdoor-based ACE estimation
- feature selection based on causal importance
- strong tree-based classification (XGBoost)
- post-hoc explainability via SHAP

This is different from conventional intrusion detection systems that simply train on all features and accept high-dimensional but less interpretable models.

---

## Why Causal Selection Matters

The main benefit is that the selected features are not only predictive, but also structurally meaningful. This helps the project:

- reduce noise from correlated or redundant features,
- avoid relying on shortcuts or dataset-specific patterns,
- improve interpretability of the model,
- support better transfer under domain shift.

---

## Main Files

- `run_pipeline.py` — runs the full pipeline from causal discovery to explainability
- `preprocess.py` — preprocessing and data balancing utilities
- `phase2_causal_discovery.py` — FCI causal graph generation
- `phase3_ace_filtering.py` — ACE filtering and causal feature selection
- `phase4_xgboost_scoring.py` — binary intrusion risk scoring model
- `phase4b_category_classifier.py` — attack category classifier
- `phase4c_baselines.py` — baseline evaluation
- `phase4d_three_tier_scoring.py` — risk-tier model
- `phase5_shap_explainability.py` — SHAP explanation generation
- `phase6_dashboard.py` — interactive dashboard
- `test.py` — transfer/recalibration analysis on ToN-IoT

---

## How to Run the Full Pipeline

From the project root:

```bash
python run_pipeline.py
```

To start from a specific phase:

```bash
python run_pipeline.py --from-phase 3
```

To skip a particular phase:

```bash
python run_pipeline.py --skip-phase 4c
```

Then launch the dashboard:

```bash
streamlit run phase6_dashboard.py
```

---

## Folder Structure

```text
.
├── data/
│   ├── causal/
│   ├── cross_val/
│   └── processed/
├── models/
├── results/
├── outputs/
├── phase2_causal_discovery.py
├── phase3_ace_filtering.py
├── phase4_xgboost_scoring.py
├── phase4b_category_classifier.py
├── phase4c_baselines.py
├── phase4d_three_tier_scoring.py
├── phase5_shap_explainability.py
├── phase6_dashboard.py
├── preprocess.py
├── run_pipeline.py
├── test.py
├── README.md
└── ...
```

---

## Typical Outputs

The project generates artifacts such as:

- trained model files in `models/`
- causal metadata in `data/causal/`
- filtered datasets in `data/processed/`
- performance reports in `results/`
- visual plots in `results/` and `outputs/`

These outputs help validate both the model and the causal interpretation.

---

## Evaluation Summary

The project reports both in-domain and cross-domain performance. The model is evaluated using metrics such as:

- ROC-AUC
- PR-AUC
- F1-score
- accuracy
- macro average metrics
- confusion matrix

Cross-dataset tests examine whether a model trained on CICIoT2023 still performs sensibly on other datasets and whether recalibration or threshold adjustment is needed.

---

## Practical Use

This project can be used for:

- intrusion detection in IoT network traffic,
- explainable security monitoring,
- model auditing with causal and feature-based explanations,
- domain shift analysis across different network environments,
- threshold tuning for low false alarm operation.

---

## Summary

CausalShield is a full research-style AI security pipeline that combines causality, machine learning, explanation, and deployment-oriented evaluation. Instead of treating features as independent signals, it first asks which features actually affect attack behavior and uses that knowledge to build a compact, interpretable intrusion detection model. The project is suitable for security analytics, network intrusion monitoring, and explainable AI research in cyber-defense applications.

---

## Notes

This repository is structured as a multi-phase research pipeline, so the system is intended to be run in sequence. Each phase generates artifacts used by later phases, and the dashboard visualizes the final outputs in an easy-to-use interface.
