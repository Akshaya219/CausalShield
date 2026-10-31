# CausalShield: Explainable IoT Intrusion Risk Scoring via Causal Graph-Guided Feature Selection

**Gokul Raj** (gokulraj@gmail.com), **Giridharan K** (giridharan1ads@gmail.com), **Akshaya P** (aksh.learn.lead@gmail.com), **Harini S** (harini11022006@gmail.com)

Department of Information Technology, Sona College of Technology, Salem, India

---

## Abstract

Intrusion detection systems (IDS) built on machine learning often break when they are moved from the network they were trained on to a different one. A large part of the reason is that correlation-based feature selection picks up statistics that belong to the source capture, and these do not survive changes in topology or device mix. We present CausalShield, an explainable risk-scoring framework for IoT intrusion detection. It runs the Fast Causal Inference (FCI) algorithm, adds an observational proxy score, and uses the result to choose features that are tied to the structure of the causal graph. On CICIoT2023 [9], CausalShield keeps seven of 39 features (an 82% reduction) and passes them to an XGBoost classifier. On the held-out test set the model reaches attack-class F1 = 0.9252, benign-class F1 = 0.6874 (macro F1 = 0.8063), and ROC-AUC = 0.9393. Five-fold cross-validation gives ROC-AUC = 0.9380 ± 0.0006, which indicates that the model is stable in-domain under fixed modeling choices. In a three-condition ablation, retraining on the same seven features without FCI gives identical results (ΔAUC = 0.000), so the graph agrees with expert domain knowledge; mutual-information selection scores higher in-domain (AUC 0.9925 vs 0.9393) but relies on features with no causal grounding. Per-flow SHAP explanations and a three-tier risk policy let analysts audit each decision. We also test the frozen model on ToN-IoT [5] without retraining. A 1% calibration split, used only to correct score polarity and set the threshold (τ* = 0.023), yields ROC-AUC = 0.8125 and macro F1 = 0.7817 on the remaining 6.90 M flows. This operating point has a benign false positive rate of 27.6%; a stricter enterprise threshold (τ = 0.080) lowers it to 2.56% but reduces attack F1 to 0.3567. Overall, the protocol-level features keep their rank-ordering across the two networks.

**Index Terms** — IoT intrusion detection, causal inference, FCI algorithm, XGBoost, SHAP, CICIoT2023, ToN-IoT, cross-dataset transfer, explainable AI

---

## I. Introduction

IoT deployments now cover industrial sensors, healthcare telemetry, and consumer appliances, and together they expose a wide attack surface: volumetric DDoS, botnet spread, reconnaissance scans, and web exploits. Machine learning IDS regularly report near-perfect scores on benchmark data, but their performance drops sharply once they are moved to a network they have not seen [1], [2]. Much of this fragility comes from how features are chosen: correlation-based methods such as Pearson correlation, mutual information, and ANOVA F-tests tend to reward quantities that reflect how the training data was captured (testbed bandwidth, packet pacing, host-specific inter-arrival times), and these change as soon as the topology does [3].

CausalShield replaces empirical correlation with structural connectivity in a learned causal ancestral graph as the basis for feature selection. Candidate features are picked according to how they connect to the attack label in the graph, including the possibility of hidden common causes, and not according to raw statistical association alone (Section VI). We combine the constraint-based Fast Causal Inference (FCI) algorithm [4] with an observational Average Causal Effect (ACE) proxy score. The result is a small classifier whose inputs are tied to transport-layer control behavior and are easy to interpret. This paper makes five contributions:

1. We apply constraint-based FCI discovery to IoT telemetry from CICIoT2023 [9] and identify seven protocol-level features adjacent to the attack label, while allowing for unmeasured device states;
2. We run a three-condition ablation showing that FCI recovers the same transport-level features that domain knowledge selects (ΔAUC = 0.000 against domain-knowledge retraining), and we measure the accuracy cost relative to correlation-based selection;
3. We build an operational risk pipeline that combines causal discovery, calibrated gradient-boosted trees, per-flow SHAP explanations, and a three-tier SOC alerting policy;
4. We report an in-domain stability audit using 5-fold cross-validation (ROC-AUC = 0.9380 ± 0.0006) across stratified folds of the development pool; and
5. We evaluate cross-dataset transfer to ToN-IoT [5] with a strict 1% calibration / 99% held-out protocol. The frozen model reaches ROC-AUC = 0.8125 and F1_mac = 0.7817 (τ* = 0.023), and we quantify the trade-off between stealth-attack recall and false alarm suppression under an enterprise constraint (τ = 0.080, FPR = 2.56%, F1_atk = 0.3567).

---

## II. Related Work

### A. IoT Intrusion Detection and Generalisation

Ensemble methods such as Random Forests and gradient-boosted trees score well on older benchmarks including UNSW-NB15 [6] and CICIDS2017 [7]. They lose much of that accuracy when the traffic distribution shifts, whether because of sampling bias or protocol drift [1], [2]. Anomaly-based detectors handle unseen attacks better because they model normal behavior, but they usually raise too many false alarms for busy enterprise networks [8]. CICIoT2023 [9] was built to address the limits of earlier datasets and covers 33 IoT attack types collected on a physical testbed. Even so, generalizing across environments is still an open problem [2], and statistical feature selection does not solve it [3], [10].

### B. Causal Inference and Feature Selection

Pearl's Structural Causal Models (SCM) [11], [12], Invariant Risk Minimisation (IRM) [3], and causal representation learning [13] give formal ways to separate stable mechanisms from spurious correlations tied to the environment. Network telemetry rarely satisfies causal sufficiency, since hardware variants, private topologies, and encrypted payloads are not observed, which makes algorithms such as PC unsuitable. FCI [4] allows for hidden confounders and returns a Partial Ancestral Graph (PAG) [14], in which a bidirected edge signals an unmeasured common cause. Score-based methods such as CAM [15], Additive Noise Models (ANM) [16], and LiNGAM rely on strong assumptions (linearity, non-Gaussianity, additive noise) that packet-level data are unlikely to meet, and none of them handles latent confounding directly. We use the constraint-based FCI implementation in causal-learn [17].

### C. Model Explainability in Network Security

Shapley Additive Explanations (SHAP) [18], [19] give local feature attributions grounded in cooperative game theory, and they fit the triage and compliance needs of Security Operations Centers (SOC) [20], [21]. One caveat applies: a SHAP value reflects how much a feature matters to the trained model, and it should not be read as an interventional causal effect [22].

---

## III. Dataset and Preprocessing

### A. CICIoT2023 (Training and Evaluation Domain)

CICIoT2023 [9] contains 33 IoT attack types grouped into seven classes (DDoS, DoS, Reconnaissance, Web-based, Brute Force, Spoofing, Mirai), plus Benign traffic. Flows were extracted with CICFlowMeter, giving 46 flow-level features. We use a binary target (0 = Benign, 1 = Attack).

Preprocessing removes zero-variance features, non-finite values, and fields with more than 30% missing entries, which leaves 39 features. We then drop highly collinear features (|r| > 0.95) and quasi-constant ones (mode frequency > 0.95), giving 34 candidate variables for causal discovery. The 6.36 M raw flows are heavily imbalanced (for example, 4.35 M DDoS flows against 1,828 Brute Force records), so we apply stratified undersampling (seed = 42) and obtain three mutually exclusive partitions (Table I). No flow appears in more than one split.

**TABLE I — CICIoT2023 Dataset Partitioning Summary**

| Split | Classes | Target Cap† | Actual Total | Operational Role |
|---|---|---|---|---|
| Train | 8 | 30,000 | 182,821 | Model training |
| Validation | 8 | 10,000 | 59,672 | Early stopping / threshold tuning‡ |
| Test | 8 | 10,000 | 59,438 | Clean held-out evaluation |
| Dev Pool | 8 | — | 242,493 | 5-fold CV audit (Train ∪ Val) |

† Minority classes contribute fewer instances. ‡ Validation metrics are inherently optimistic due to dual-use; final reported performance is evaluated strictly on the test set.

![Fig. 1](fig1_pipeline.png)

**Fig. 1.** Overall architecture of the CausalShield pipeline, from data preparation to the audited risk output. The pipeline has eight stages:

1. **Data preparation** — CICIoT2023 (6.36M flows, imbalanced) → cleaning (39 features, then 34) → stratified split (182.8k / 59.7k / 59.4k).
2. **Causal discovery (FCI)** — balanced sample (2,500 per class) → FCI run (α = 0.05, 12.1 s) → PAG output (68 edges, 46 bidirected).
3. **Causal feature selection** — adjacency (7 of 34 nodes) → ACE proxy (six pass 0.05) → Duration kept (directed edge) → stability (5 runs, 7/7).
4. **Modeling and calibration** — 7 features (82% reduction) → XGBoost (η = 0.05, depth 6) → early stop (round 497) → calibration (isotonic, τ = 0.61).
5. **Scoring and explainability** — risk score calibrated 0 to 1 → three tiers (cuts 0.25, 0.65) → Tree SHAP per-flow waterfall → SOC action (monitor to block).
6. **In-domain evaluation** — test set AUC 0.9393 → 5-fold CV 0.9380 ± 0.0006 → FCI ablation (ΔAUC 0.000 vs DK) → latency 1.42 ms per flow.
7. **Cross-dataset transfer (ToN-IoT)** — frozen model (HTTPS set to zero) → inversion (raw AUC 0.1867) → calibration (1% split, reflect) → held-out test (6.90M flows) → recalibrated (τ* 0.023, FPR 27.6%, F1 0.78) and enterprise (τ 0.080, FPR 2.56%, F1_atk 0.36).
8. **Output** — risk verdict (tier and SOC action), explanation (SHAP waterfall), audit trail (PAG edge, 7 features).

---

## IV. Methodology

### A. FCI Causal Discovery and Graph Stability

We run FCI [4] on a class-balanced sample of 5,000 flows from the development partition (n = 2,500 benign, n = 2,500 attack; seed = 42) using 34 input features. The subsample is needed because conditional independence tests in constraint-based discovery scale super-linearly with sample size; 5,000 samples were enough to detect moderate conditional dependencies at α = 0.05, and discovery finished in 12.1 s.

![Fig. 2](fig2_pag.png)

**Fig. 2.** FCI PAG (α = 0.05, n = 5,000). Red: Attack; Orange: 7 causal features; Grey: background nodes. Bidirected edges indicate latent confounding.

The resulting PAG [14] has 68 edges, and **46 of them are bidirected (↔)**, which points to substantial latent confounding from unmeasured device context.

Seven features are adjacent to the Attack node (Fig. 2): HTTPS, ack_flag_number, psh_flag_number, rst_flag_number, rst_count, Variance, and Duration. Six of these connect through bidirected edges (↔). Only Duration has a directed edge (→), which under FCI assumptions indicates that it is an ancestor of Attack (Section VI).

**Graph Stability Verification:** To check that the seven features are not just a product of the seed-42 sample, we repeated FCI on five non-overlapping 5,000-flow stratified subsets from the development pool. In all five runs the same seven features stayed adjacent to the Attack node (7/7 features in 5/5 runs). Across the full 68-edge PAG, 59 edges (86.8%) were identical in every run. The other 9 edges changed only in arrowhead/tail orientation, and only between background protocol features. No edges appeared or disappeared around the Attack node's Markov boundary.

### B. ACE Proxy Scoring and Selection

Each feature linked to the Attack node receives an observational proxy score. For the binary indicators (ack, HTTPS, psh, rst_flag_number, and rst_count > 0) it is

```
ACE_obs(X_i) = E[Y | X_i = 1] − E[Y | X_i = 0].        (1)
```

The continuous features (Duration, Variance) are binarized at the training-set median before applying (1) [23]. These scores are only an empirical ranking heuristic and not unconfounded interventional effects, because latent confounding rules out a formal do-calculus backdoor adjustment.

The four binary TCP flags all receive the same proxy score (+0.8411), because they are active together in attack traffic. Six features satisfy |ACE_obs| ≥ 0.05 (Fig. 3). Duration falls just below this cutoff (|proxy| = 0.0430) but is kept because of its directed ancestral edge (→). Table II lists all seven selected features.

![Fig. 3](fig3_ace.png)

**Fig. 3.** Signed observational ACE proxy scores. Green-bordered bars satisfy |ACE_obs| ≥ 0.05. *Duration* is retained on structural grounds.

**TABLE II — PAG-Guided Feature Properties and Attributions**

| Feature | PAG Edge | Signed Proxy | \|Proxy\| | Mean \|SHAP\| |
|---|---|---|---|---|
| rst_count | ↔ | −0.3122 | 0.3122 | 2.8008 |
| Variance | ↔ | −0.2842 | 0.2842 | 1.8483 |
| Duration | → | −0.0430 | 0.0430 | 0.4126 |
| ack_flag_number | ↔ | +0.8411 | 0.8411 | 0.2866 |
| HTTPS | ↔ | +0.8411 | 0.8411 | 0.1495 |
| rst_flag_number | ↔ | +0.8411 | 0.8411 | 0.0859 |
| psh_flag_number | ↔ | +0.8411 | 0.8411 | 0.0628 |

Edge types: ↔ (latent confounding), → (ancestral). Mean |SHAP| computed across test set (n = 59,438, Fig. 6).

### C. XGBoost Scoring, Calibration, and Validation Scope

We train an XGBoost model [24] on the seven features, with early stopping (patience = 30) on validation AUC. The main settings are η = 0.05, max_depth = 6, tree_method = hist, and scale_pos_weight = 0.196. Training stops at round 497 with a validation AUC of 0.9411. Probabilities are calibrated with isotonic regression. To balance detection sensitivity against disruption to benign traffic, we sweep τ ∈ [0.01, 0.99] on calibrated validation probabilities and pick the value that maximizes macro-averaged F1. This gives τ = 0.61 (F1_mac = 0.8106, F1_atk = 0.9271, F1_ben = 0.6940).

![Fig. 4](fig4_training.png)

**Fig. 4.** XGBoost training history (Log Loss). Convergence at round 497.

**Split Isolation and CV Audit:** The validation set (n = 59,672) was used for both early stopping and threshold selection, so its metrics are optimistic. All clean results therefore come from the held-out test set (n = 59,438). Five-fold cross-validation over the development pool (n = 242,493) gives ROC-AUC = 0.9380 ± 0.0006, F1_atk = 0.9345 ± 0.0003, F1_ben = 0.6720 ± 0.0008, and F1_mac = 0.8032 ± 0.0005 (fold SD). This shows that the decision surface is stable across folds; it is not an estimate of out-of-fold generalization.

### D. Three-Tier Risk Scoring and Local Explainability

Calibrated probabilities are grouped into three operational tiers (Table III). The LOW tier is bounded by FPR ≤ 5% on validation data, and the HIGH tier is designed for attack precision of at least 95%. We compute Tree SHAP values [18], [19] for all 59,438 test flows. Individual waterfall plots (Fig. 5) show that rst_count and Variance contribute most to the model's confidence.

**TABLE III — Three-Tier Risk Scoring Policy (Observed on Test Set)**

| Tier | Threshold (τ) | Action | Flows | Attack % |
|---|---|---|---|---|
| LOW | < 0.250 | Monitor | 2,871 | 16.2% |
| MEDIUM | 0.250–0.650 | Investigate | 9,732 | 45.1% |
| HIGH | > 0.650 | Block | 46,835 | 95.2% |

![Fig. 5](fig5_waterfall.png)

**Fig. 5.** SHAP waterfall for a HIGH-risk flow. Dominant signals: rst_count (+4.25) and Variance (+3.08). Post-calibration score: 100.0%.

![Fig. 6](fig6_shap_global.png)

**Fig. 6.** Global feature attribution (Mean |SHAP|) across 59,438 test flows.

---

## V. Experimental Results

### A. In-Domain Performance on CICIoT2023

Table IV gives results on the held-out test set (n = 59,438). At τ = 0.61, CausalShield reaches attack-class F1 = 0.9252 (recall 89.77%), benign-class F1 = 0.6874 (specificity 78.86%), macro F1 = 0.8063, and ROC-AUC = 0.9393 (TP = 44,380, FN = 5,058, TN = 7,886, FP = 2,114). The benign false positive rate is 21.1% (2,114 of 10,000 benign flows), partly a side effect of scale_pos_weight = 0.196, which shifts the decision boundary toward catching attacks at some cost to benign precision. The 5.32 pp AUC gap to mutual information (0.9925 → 0.9393) is the price of leaving out dataset-specific correlation shortcuts [21] in favor of features with causal support [20].

**Why benign F1 is lower:** The difference between attack F1 (0.9252) and benign F1 (0.6874) follows from the selected features. Four of the seven are TCP control signals (ack, psh, rst flags, rst_count) that are mostly active during scanning and flooding. Benign IoT traffic, by contrast, is varied (periodic MQTT, bursty HTTP, and so on) and has no comparable signature in these features; since application-level payload information was dropped during graph construction, benign flows are more sensitive to where the threshold is placed.

**TABLE IV — In-Domain Performance on CICIoT2023 Test Set (n = 59,438)**

| Model | Feats | F1_atk | F1_ben | F1_mac | AUC |
|---|---|---|---|---|---|
| Full-Feature XGBoost | 39 | 0.9813 | 0.9021 | 0.9417 | 0.9944 |
| MI-Selected XGBoost | 7 | 0.9786 | 0.8914 | 0.9350 | 0.9925 |
| **CausalShield (Test Set)** | 7 | 0.9252 | 0.6874 | 0.8063 | 0.9393 |
| *In-Domain Stability Audit (n = 242,493):* | | | | | |
| CausalShield (5-Fold CV, stability)† | 7 | 0.9345 | 0.6720 | 0.8032 | 0.9380 |

Evaluated at τ = 0.61. Test CM: TN = 7,886, FP = 2,114, FN = 5,058, TP = 44,380. PR-AUC = 0.9875. † Dev-pool CV stability audit (fold SD: F1_atk = ±0.0003, F1_ben = ±0.0008, F1_mac = ±0.0005, AUC = ±0.0006; in-domain sensitivity audit, not out-of-fold generalization).

### B. Comparison with Published Benchmarks

Table V places CausalShield next to recent work on CICIoT2023 [25], [26]. The published baselines solve multiclass problems and do not report binary ROC-AUC, so the table gives context and is not a like-for-like comparison.

**TABLE V — Contextual Comparison with Published Models on CICIoT2023**

| Method | Task | Feats | Macro F1 | ROC-AUC | Ref. |
|---|---|---|---|---|---|
| RF (tuned) | 7-class | All | 0.92 | — | [25] |
| VGGNet (tuned) | 7-class | All | 0.89 | — | [25] |
| DNN (tuned) | 7-class | All | 0.87 | — | [25] |
| RF | 34-class | All | — | — | [26] |
| XGBoost | 34-class | All | — | — | [26] |
| Full-Feat. XGB (ours) | Binary | 39 | 0.9417 | 0.9944 | — |
| MI-Selected XGB (ours) | Binary | 7 | 0.9350 | 0.9925 | — |
| **CausalShield (ours)** | **Binary** | **7** | **0.8063** | **0.9393** | — |

### C. FCI Ablation Study

To see what causal discovery contributes, we compare three pipelines: (A) CausalShield (FCI + PAG + ACE); (B) Domain Knowledge Only (DK-Only), retrained on the same 7 features but without the graph; and (C) Mutual Information, using the top-7 statistical features (IAT, urg_count, rst_count, Max, Tot size, flow_duration, Duration).

Table VI shows the results. Conditions A and B give identical performance (ΔAUC = 0.0000, ΔF1_mac = 0.0000). Since both use the same seven features, this is expected; it shows that FCI, working only from observational data, arrives at the protocol-level features an experienced network security engineer would choose by hand. That agreement supports the learned graph, because in transport-layer IoT traffic any valid causal mechanism should be consistent with the underlying protocol specifications. It does not show that FCI improves accuracy; the benefit of Condition A is procedural, since the features come from conditional independence tests at α = 0.05 and not from manual trial and error.

Condition C (MI-Selected) has a higher in-domain AUC (0.9925 vs 0.9393, Δ = +0.0532). Its top-ranked features include total flow volume (Tot size) and inter-arrival time (IAT), timing and throughput measures that most likely reflect how CICIoT2023 traffic was generated on the testbed and not properties of the attacks themselves. We accept the 5.32 pp AUC difference on purpose. In return we get an 82% smaller feature set, a documented ancestral graph for each feature, and features that keep their ranking ability on another dataset.

**TABLE VI — FCI Ablation on CICIoT2023 Test Set (n = 59,438)**

| Condition | Feats | F1_mac | F1_atk | ROC-AUC | Retrained |
|---|---|---|---|---|---|
| (A) CausalShield (FCI) | 7 | 0.8063 | 0.9252 | 0.9393 | No |
| (B) DK-Only (no FCI) | 7 | 0.8063 | 0.9252 | 0.9393 | Yes |
| (C) MI-Selected | 7 | 0.9350 | 0.9786 | 0.9925 | Yes |

Δ(A−B): ΔAUC = 0.0000. Δ(A−C): ΔAUC = −0.0532, ΔF1_mac = −0.1287.

**TABLE VII — CausalShield vs Mutual Information Feature Selection**

| # | CausalShield | \|Proxy\| | MI Feature | MI Score |
|---|---|---|---|---|
| 1 | HTTPS | 0.8411 | IAT | 0.2922 |
| 2 | ack_flag_number | 0.8411 | urg_count | 0.1837 |
| 3 | psh_flag_number | 0.8411 | rst_count | 0.1796 |
| 4 | rst_flag_number | 0.8411 | Max | 0.1698 |
| 5 | rst_count | 0.3122 | Tot size | 0.1536 |
| 6 | Variance | 0.2842 | flow_duration | 0.1500 |
| 7 | Duration | 0.0430 | Duration | 0.1471 |

### D. Feature Space Comparison and Inference Latency

Table VII lists the two feature sets. The MI features describe traffic volume and timing, while the CausalShield features describe protocol behavior. On an Intel i7 workstation (Python 3.12, XGBoost 3.2.0), single-sample latency averages 1.42 ms (P95 = 1.81 ms), which is 2.3× faster than the 39-feature model (3.29 ms, Table VIII). End-to-end latency falls from 11.04 ms to 2.65 ms (4.2×), and throughput reaches 319,190 flows/s at batch size 1,000.

**TABLE VIII — Single-Sample Inference Latency Benchmark (500 Trials)**

| Model | Feats | Mean (ms) | P95 (ms) | Speedup |
|---|---|---|---|---|
| Full-Feature XGBoost | 39 | 3.2904 | 3.9082 | 1.0× |
| MI-Selected XGBoost | 7 | 1.5205 | 1.9036 | 2.2× |
| CausalShield | 7 | 1.4201 | 1.8122 | 2.3× |

### E. Cross-Dataset Transfer Analysis (ToN-IoT)

To test transfer to a different network, we applied the frozen CICIoT2023 model to ToN-IoT [5] (4.85 M attack and 2.12 M benign flows). HTTPS is not available in ToN-IoT, so it was set to zero. Its in-domain importance is low (mean |SHAP| = 0.1495), so this acts as a forced ablation of one feature and leaves the core protocol features (rst_count, Variance, Duration) untouched.

**Score Inversion and Calibration:** The raw scores were inverted, with probabilities bunched near one (raw ROC-AUC = 0.1867). Two properties of ToN-IoT explain this: Duration differs in scale by a factor of 7.5× (11.54 M µs for benign vs 1.54 M µs for attack), and rst_count is zero for 98.9% of flows. To avoid leakage, we split ToN-IoT into a 1% calibration set (n = 69,747) and a 99% held-out evaluation set (n = 6,904,917). We fixed the polarity by reflecting the score (s′ = 1 − s). A grid search on the calibration set gave τ* = 0.023, along with an enterprise threshold of τ_fpr05 = 0.080 (FPR ≤ 5%).

On the 6.90 M held-out flows (Table IX), the model reaches ROC-AUC = 0.8125 and F1_mac = 0.7817 at τ* = 0.023. At τ = 0.080, the benign false alarm rate falls to 2.56% and F1_atk drops to 0.3567. The per-attack results (Table X) show perfect transfer for volumetric attacks (DDoS and DoS F1 = 1.000) and good detection of application-level exploits (F1 ≈ 0.87–0.88).

**TABLE IX — Cross-Dataset Transfer on ToN-IoT (Strict 1% Calibration)**

| Operating Point | τ | F1_atk | F1_ben | F1_mac | FPR |
|---|---|---|---|---|---|
| In-domain (CICIoT2023 test) | 0.610 | 0.9252 | 0.6874 | 0.8063 | 0.2114 |
| Zero-shot, default threshold | 0.610 | 0.7195 | 0.0000 | 0.3598 | 1.0000 |
| Recalibrated (τ*) | 0.023 | 0.8149 | 0.7484 | 0.7817 | 0.2758 |
| Enterprise (FPR ≤ 5%) | 0.080 | 0.3567 | 0.6555 | 0.5061 | 0.0256 |

*Held-Out Discriminability (n = 6,904,917 flows):*

- Reflected ROC-AUC: 0.8125 (Δ = −0.1268 vs in-domain)
- Reflected PR-AUC: 0.7984 (Δ = −0.1891 vs in-domain)

**TABLE X — Per-Attack Performance on Held-Out ToN-IoT (τ* = 0.023)**

| Attack Category | Flows | Accuracy | F1_atk |
|---|---|---|---|
| DDoS | 202 | 1.0000 | 1.0000 |
| DoS | 145 | 1.0000 | 1.0000 |
| Backdoor | 27,144 | 0.9948 | 0.9974 |
| XSS | 2,057,122 | 0.8529 | 0.9206 |
| Scanning | 35,564 | 0.8160 | 0.8987 |
| MITM | 517 | 0.8279 | 0.9058 |
| Ransomware | 4,788 | 0.8116 | 0.8960 |
| Password | 328,648 | 0.7814 | 0.8773 |
| Injection | 269,499 | 0.7716 | 0.8711 |
| Benign | 2,123,572 | 0.7242 | — |

Rescaling Duration and Variance with target-only quantiles also broke the learned split boundaries (F1_mac → 0.3300), which suggests that harmonization needs a mapping anchored to the source data and not simple rescaling on the target.

---

## VI. Discussion and Limitations

The ablation (Section V-C) shows that FCI-guided selection and expert domain knowledge end up with the same feature set and the same discrimination (ΔAUC = 0.0000). The 5.32 pp AUC gap to mutual information is an example of the usual robustness–performance trade-off in security machine learning [21]. Correlation-based selection fits artifacts of the capture environment that have no structural basis, whereas each of the seven CausalShield features has a PAG edge, a proxy score, and a per-flow SHAP attribution that analysts can inspect [20].

Since domain knowledge and FCI give the same features here, it is fair to ask whether running FCI is worthwhile; we see three reasons.

1. **Reproducibility:** Expert selection is subjective and shaped by the analyst's prior beliefs, whereas FCI produces the feature set from observational data and a significance level α alone, so an external auditor can repeat the derivation and trace each feature.
2. **Portability to unfamiliar deployments:** Expert intuition works well for familiar TCP flooding attacks but is less reliable in heterogeneous deployments with proprietary industrial protocols (e.g., Modbus, BACnet, Zigbee) or new attack types. Automated discovery can propose a starting feature set there without a protocol specialist on site, though we have not yet tested this.
3. **Auditability:** Frameworks such as the NIST AI RMF and SOC audit practices call for decision boundaries that can be checked, and every CausalShield risk score traces back to a specific graph and feature set that reviewers can inspect.

The proxy scores are empirical ranking heuristics and should not be read as identified causal effects, since bidirected edges indicate latent confounding.

**Limitations:**

- **(i) Shared validation set:** The validation set was used for both early stopping and threshold selection, so validation results are optimistic. All reported benchmarks come from the separate test set.
- **(ii) Subsampled discovery:** FCI ran on 5,000-flow subsets because of its super-linear cost. Discovery on the full data with faster kernel-based independence tests is left for future work.
- **(iii) False alarms on transfer:** At τ* = 0.023 the benign FPR on ToN-IoT is 27.58%, caused by the difference in flow duration scale. The enterprise threshold (τ = 0.080) reduces false alarms to 2.56% but lowers stealth-attack recall (F1_atk = 0.3567).
- **(iv) Feature extractor mismatch:** ToN-IoT has no encryption indicators, so HTTPS had to be set to zero. Multi-dataset causal benchmarking will need a common feature extractor.

---

## VII. Conclusion

We showed that constraint-based causal discovery can pick out a small set of protocol-level features for IoT intrusion detection. With 82% of the features removed, the model still reaches ROC-AUC = 0.9393 and F1_mac = 0.8063 on CICIoT2023, at a workstation latency of 1.42 ms (profiling on gateway hardware is left for deployment). Without retraining, the frozen model keeps a useful ranking of flows on ToN-IoT (ROC-AUC = 0.8125, F1_mac = 0.7817 over 6.90 M flows). Limiting false alarms to an enterprise budget (τ = 0.080, FPR ≤ 2.6%) cuts false positives but lowers stealth-attack sensitivity (F1_atk = 0.3567), a trade-off that comes from the scale differences between the two datasets. Even so, the PAG-guided protocol features continue to separate attack from benign traffic across both networks. Future work will evaluate on UNSW-NB15 [6], apply Rosenbaum bounds [23] to test sensitivity to confounding, and study calibration anchored to the source distribution.

---

## References

[1] A. Abdulhammed, M. Faezipour, A. Abuzneid, and A. Abuzneid, "Deep and machine learning approaches for anomaly-based intrusion detection of imbalanced network traffic," *IEEE Sensors Lett.*, vol. 3, no. 1, pp. 1–4, 2019.

[2] Z. Ahmad, A. S. Khan, C. W. Shiang, J. Abdullah, and F. Ahmad, "Network intrusion detection system: A systematic study of machine learning and deep learning approaches," *Trans. Emerg. Telecommun. Technol.*, vol. 32, no. 1, p. e4150, 2021.

[3] M. Arjovsky, L. Bottou, I. Gulrajani, and D. Lopez-Paz, "Invariant risk minimization," arXiv preprint arXiv:1907.02893, 2019.

[4] P. Spirtes, C. Glymour, and R. Scheines, *Causation, Prediction, and Search*, 2nd ed. Cambridge, MA: MIT Press, 2000.

[5] N. Moustafa, "A new distributed architecture for evaluating AI-based security systems at the edge: Network ToN-IoT datasets," in *Proc. IEEE Symp. Ser. Comput. Intell. (SSCI)*, 2021, pp. 1–6.

[6] N. Moustafa and J. Slay, "UNSW-NB15: A comprehensive data set for network intrusion detection systems," in *Proc. MilCIS*, 2015, pp. 1–6.

[7] I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, "Toward generating a new intrusion detection dataset and intrusion traffic characterization," in *Proc. ICISSP*, 2018, pp. 108–116.

[8] P. Garcia-Teodoro, J. Diaz-Verdejo, G. Macia-Fernandez, and E. Vazquez, "Anomaly-based network intrusion detection: Techniques, systems and challenges," *Comput. Secur.*, vol. 28, no. 1–2, pp. 18–28, 2009.

[9] E. C. P. Neto et al., "CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment," *Sensors*, vol. 23, no. 13, p. 5941, 2023.

[10] R. Wieringa and M. Daneva, "Six strategies for generalizing software engineering theories," *Sci. Comput. Program.*, vol. 101, pp. 136–152, 2015.

[11] J. Pearl, *Causality: Models, Reasoning and Inference*, 2nd ed. Cambridge: Cambridge Univ. Press, 2009.

[12] J. Peters, D. Janzing, and B. Schölkopf, *Elements of Causal Inference: Foundations and Learning Algorithms*. Cambridge, MA: MIT Press, 2017.

[13] B. Schölkopf et al., "Toward causal representation learning," *Proc. IEEE*, vol. 109, no. 5, pp. 612–634, 2021.

[14] T. Richardson and P. Spirtes, "Ancestral graph Markov models," *Ann. Statist.*, vol. 30, no. 4, pp. 962–1030, 2002.

[15] P. Bühlmann, J. Peters, and J. Ernest, "CAM: Causal additive models, high-dimensional order search and penalized regression," *Ann. Statist.*, vol. 42, no. 6, pp. 2526–2556, 2014.

[16] J. Peters, J. M. Mooij, D. Janzing, and B. Schölkopf, "Causal discovery with continuous additive noise models," *J. Mach. Learn. Res.*, vol. 15, pp. 2009–2053, 2014.

[17] Y. Zheng et al., "Causal-learn: Causal discovery in Python," *J. Mach. Learn. Res.*, vol. 25, no. 60, pp. 1–8, 2024.

[18] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 30, 2017.

[19] S. M. Lundberg et al., "From local explanations to global understanding with explainable AI for trees," *Nature Mach. Intell.*, vol. 2, no. 1, pp. 56–67, 2020.

[20] M. T. Ribeiro, S. Singh, and C. Guestrin, "Why should I trust you?: Explaining the predictions of any classifier," in *Proc. ACM SIGKDD Int. Conf. Knowl. Discov. Data Min.*, 2016, pp. 1135–1144.

[21] C. Molnar, *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable*, 2nd ed. Munich, Germany: Leanpub, 2022.

[22] Z. C. Lipton, "The mythos of model interpretability," *Queue*, vol. 16, no. 3, pp. 31–57, 2018.

[23] P. R. Rosenbaum, *Observational Studies*, 2nd ed. New York, NY: Springer, 2002.

[24] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. ACM SIGKDD Int. Conf. Knowl. Discov. Data Min.*, 2016, pp. 785–794.

[25] H. Fares, N. Aknin, G. Lazrek, and M. Zeroual, "Intrusion detection in IoT environment using hyperparameters tuned machine and deep learning models on the CICIoT2023 dataset," *Informatica*, vol. 49, no. 5, pp. 325–344, 2025.

[26] J. P. Ntayagabiri, Y. Bentaleb, J. Ndikumagenge, and H. El Makhtoum, "A comparative analysis of supervised machine learning algorithms for IoT attack detection and classification," *J. Comput. Theor. Appl.*, vol. 2, no. 3, pp. 395–409, 2025.
