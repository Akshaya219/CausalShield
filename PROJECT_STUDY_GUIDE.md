# 🛡️ CausalShield: Complete Project Study Guide & Reference Notes
> **A Comprehensive, Step-by-Step Layman's Guide to Explainable, Causal-Guided IoT Intrusion Risk Scoring**  
> *Written as a complete conceptual textbook, reference guide, and study companion.*

---

## 🧭 Executive Navigation Map

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            THE GRAND ARCHITECTURE                            │
└──────────────────────────────────────────────────────────────────────────────┘
       1. THE PROBLEM: Why Traditional Cyber AI Fails in the Real World
                                      │
                                      ▼
       2. THE CORE PHILOSOPHY: From Statistical Correlation to True Causation
                                      │
                                      ▼
       3. STEP 1: Raw Data & Preprocessing (Cleaning the Digital Clutter)
                                      │
                                      ▼
       4. STEP 2: Causal Discovery with FCI (Finding the Master Puppet Strings)
                                      │
                                      ▼
       5. STEP 3: ACE Filtering & Backdoor Adjustment (Selecting the Elite 7)
                                      │
                                      ▼
       6. STEP 4: Risk Scoring Model (XGBoost + Isotonic Probability Calibration)
                                      │
                                      ▼
       7. STEP 5: Operational Three-Tier SOC Policy (Eliminating Alert Fatigue)
                                      │
                                      ▼
       8. STEP 6: Explainability via Tree SHAP (Opening the AI Black Box)
                                      │
                                      ▼
       9. STEP 7: Cross-Dataset Domain Transfer (The Acid Test on ToN-IoT)
                                      │
                                      ▼
      10. STEP 8: The Interactive SOC Dashboard (The Human-in-the-Loop Cockpit)
                                      │
                                      ▼
      11. MASTER SUMMARY TABLES: "What If We Don't Include It?"
                                      │
                                      ▼
      12. REVISION GLOSSARY & FREQUENTLY ASKED QUESTIONS
```

---

# 📖 Module 1: The Detective Story — Why Traditional Cyber AI Fails

### 🎭 The Analogy: The Detective and the Yellow Raincoats
Imagine a rookie detective trying to predict whether a crime is about to happen in a city.  
He looks at historical records and notices that on 95% of days when muggings occurred, people on the street were carrying umbrellas and wearing yellow raincoats.  

The rookie builds an automated alarm system: *“Whenever a person opens an umbrella, ring the city crime siren!”*  

- Inside that specific rainy neighborhood, his alarm looks **brilliant**. His statistical accuracy is 99%!
- But the moment he moves his system to a sunny desert city where it never rains, the system **collapses**.
- Even worse, on a rainy day in the new city, it arrests every innocent grandmother holding an umbrella!

What went wrong?  
The detective found a **correlation** (raincoats occur during the same weather when muggers operate), but **not a cause** (carrying an umbrella does *not* make someone commit a crime). Rain was a **spurious artifact** of where the data was gathered.

---

### 🌐 The Real Problem in IoT Intrusion Detection
Today’s Internet of Things (IoT) devices—smart cameras, medical monitors, industrial sensors, smart meters—are under constant attack from botnets (like Mirai), Denial of Service (DoS) floods, and brute-force hackers.

Engineers usually train Machine Learning (ML) models like Random Forests, Neural Networks, or standard XGBoost to inspect network traffic.
In academic test labs, these models score **99.5% accuracy**. But when installed in real corporate or industrial networks, their performance drops to **worse than a coin toss (36% to 49%)**.

#### Why do traditional models collapse?
Traditional feature selection algorithms (like Pearson Correlation, ANOVA, or Mutual Information) look only for statistical patterns. In doing so, they latch onto **dataset testbed artifacts**:
1. **Network Hardware Quirks:** Specific cable speeds, switch delays, and clock jitter of the lab where the dataset was recorded.
2. **Packet Timers & Inter-Arrival Times:** How fast packets arrived on that specific Tuesday in the testbed.
3. **Total Data Volume:** The exact size of files downloaded during the experiment.

The moment you deploy the model to an unseen network with different internet speeds or different router brands, all those superficial timing numbers change completely. The model panics, hallucinates, and triggers thousands of false alarms.

---

### 💡 The Solution: CausalShield
**CausalShield** throws out correlation-based guesswork. Instead of asking:
> *"Which numbers happen to bounce together with attack labels in this specific CSV file?"*

CausalShield asks:
> *"Which features are **causally linked** to the fundamental mechanics of a network attack across all possible networks?"*

By using **Causal Discovery (Fast Causal Inference - FCI)** and **Average Causal Effect (ACE)**, CausalShield reduces 39 noisy features down to **just 7 protocol-level causal invariants**.

| Dimension | Standard Correlation AI | CausalShield (Our System) |
|---|---|---|
| **Feature Selection** | Picks features with highest math correlation (e.g. packet arrival speed) | Discovers underlying causal graphs (PAG) using physics & protocol rules |
| **Feature Count** | 39–80 bloated features | **7 compact, rock-solid features (82% reduction)** |
| **In-Domain Lab Score** | 99.4% ROC-AUC | 93.9% ROC-AUC (honest, un-bloated score) |
| **Unseen New Network (ToN-IoT)** | **36.8% – 49.7% (Complete Collapse / Inverted)** | **81.25% ROC-AUC (High Real-World Generalization)** |
| **Inference Latency** | 3.5+ milliseconds per flow | **1.42–1.64 milliseconds (Over 2× faster)** |
| **Explainability** | Black box probability number | Exact SHAP waterfall + causal role attribution |

---

# 📦 Module 2: The Data Foundation (Phase 1 — Preprocessing)

### 🔬 What is a "Network Flow"?
When two computers talk over the Internet, they exchange packets (chunks of digital data). A **Network Flow** is like the **phone bill summary** of that conversation. It doesn't contain your private passwords; instead, it records:
- How long the call lasted (`Duration`).
- How many times the line was reset or hung up (`rst_flag_number`, `rst_count`).
- Whether the data was pushed immediately (`psh_flag_number`).
- Whether it was an encrypted web session (`HTTPS`).
- How uneven the packet sizes were (`Variance`).

### 🗄️ The Two Datasets Used in this Project

```
┌──────────────────────────────────────────────┐       ┌──────────────────────────────────────────────┐
│           CICIoT2023 (Source Domain)         │       │            ToN-IoT (Target Domain)           │
├──────────────────────────────────────────────┤       ├──────────────────────────────────────────────┤
│ • 6.36 Million raw network flows             │       │ • 6.90 Million held-out network flows        │
│ • 33 distinct attack types + Benign traffic  │       │ • Heterogeneous IoT/IIoT industrial testbed  │
│ • Grouped into 7 major attack categories     │       │ • Different hardware, routers, and protocols │
│ • USED FOR: Discovery, Training & Tuning     │       │ • USED ONLY FOR: Zero/Few-shot Acid Test     │
└──────────────────────────────────────────────┘       └──────────────────────────────────────────────┘
```

---

### ⚙️ What Phase 1 Does Step-by-Step
1. **Mapping 33 Attacks to 7 Clean Categories:**  
   Attacks like `DDoS-SlowLoris`, `DDoS-SYN_Flood`, and `Mirai-greeth_flood` are organized into high-level categories: *DDoS, DoS, Mirai, Reconnaissance, BruteForce, Spoofing, and WebBased*.
2. **Binary Conversion:**  
   Every flow is tagged simply as `0 = Benign (Normal Traffic)` or `1 = Attack`.
3. **Stratified Balancing & Sampling:**  
   Network captures often contain 95% attacks and only 5% normal traffic (or vice versa). If a model sees 99% attacks during training, it gets lazy and simply guesses "Attack" every time! Stratified sampling ensures both normal and malicious behaviors are balanced across training, validation, and testing splits.
4. **Dropping Useless Columns:**  
   Columns where every single row has the exact same number (zero variance) or columns that are 100% duplicate mirrors of other columns are purged.

---

### ❓ What Happens If We DON'T Include This Phase?
- **Class Imbalance Blindness:** The model learns a shortcut heuristic: *"Always guess the majority class."* It achieves 95% apparent accuracy while failing to detect the minority class completely.
- **Data Leakage:** If training and testing splits are shuffled randomly without stratifying or separating captured sessions, the test set contains identical duplicate flows from the training set, giving a fake illusion of high performance.

---

# 🕸️ Module 3: Causal Discovery (Phase 2 — The FCI Algorithm)

### 🧩 The Core Concept: What is Causal Discovery?
Machine learning normally calculates conditional probabilities: $P(Y \mid X)$ (*"When I see $X$, do I also see $Y$?"*).  
Causal discovery seeks **structural equations**: does changing $X$ actually cause $Y$ to change, or is there an unseen ghost pulling the strings behind both?

### 👥 The Ghost in the Machine: Latent Confounders
In real computer networks, there are dozens of hidden variables that network monitors never record:
- The CPU temperature of a smart camera.
- The queue buffer congestion inside an intermediate internet router.
- Background operating system updates running on the target machine.

If an unmeasured factor $Z$ (e.g., router buffer overflow) causes *both* high packet variance ($X$) and dropped packets ($Y$), standard algorithms assume $X$ caused $Y$. In causal mathematics, $Z$ is called an **unobserved (latent) confounder**.

---

### 🔍 Why FCI instead of the standard PC Algorithm?

| Feature | Standard PC Algorithm | FCI (Fast Causal Inference) |
|---|---|---|
| **Core Assumption** | Assumes **Causal Sufficiency** (assumes you have measured every single cause in the universe; NO hidden variables allowed). | **Allows Latent Confounders** (acknowledges that unmeasured variables exist in real networks). |
| **Graph Output** | CPDAG (Completed Partially Directed Acyclic Graph). | **PAG (Partial Ancestral Graph)**. |
| **Real-World Truth** | Unrealistic for networking. Ignores router load and hardware state. | **Designed for real-world IoT systems** where you cannot monitor everything. |

---

### 🎨 How to Read a PAG (Partial Ancestral Graph) in Plain English
When FCI finishes analyzing 5,000 balanced flows, it outputs a graph with special arrow marks. Here is what they mean:

```
        X ────────► Y         "Direct Ancestor"
                              Feature X definitely influences Y directly.

        X ◄───────► Y         "Latent Confounder"
                              Neither causes the other directly. There is an
                              unmeasured hidden cause affecting both.

        X o───────► Y         "Partially Directed"
                              X may be an ancestor of Y, but Y is definitely
                              NOT an ancestor of X.

        X o───────o Y         "Uncertain Relationship"
                              The direction cannot be determined from observational data alone.
```

---

### ❓ What Happens If We DON'T Include Causal Discovery?
- Without FCI, you rely on correlation ranking (like Mutual Information). Mutual Information selects variables like packet inter-arrival times (`IAT`) and total traffic volume (`Tot size`).
- In the lab where the dataset was captured, the attacker used a script that blasted packets every 0.002 seconds. The ML model memorized that timing.
- On a different network, a legitimate business application sends packets every 0.002 seconds, while an attack sends them slowly. The non-causal model fails completely.

---

# ⚖️ Module 4: Feature Selection via ACE (Phase 3 — The Elite 7)

### 🎯 What is ACE (Average Causal Effect)?
In causal inference, there is a fundamental difference between **Seeing** and **Doing**:
- **Observational Probability:** $P(\text{Attack} \mid \text{Feature} = 1)$ — "I observed that this flag was set; is it an attack?"
- **Interventional Probability (Do-Calculus):** $P(\text{Attack} \mid do(\text{Feature} = 1))$ — "If I physically force this flag to be set, how does the probability of an attack change?"

The **Average Causal Effect (ACE)** is defined as:
$$\text{ACE}(X \to Y) = \mathbb{E}[Y \mid do(X = 1)] - \mathbb{E}[Y \mid do(X = 0)]$$

### 🚪 Pearl's Backdoor Criterion (Layman's Terms)
Imagine testing whether drinking coffee causes heart attacks. If people who drink coffee also smoke cigarettes more often, smoking is a "backdoor path" confounding the relationship.  
To find the true effect of coffee, you must **close the backdoor** by comparing coffee drinkers against non-coffee drinkers *who smoke the exact same amount*.  
Pearl's **Backdoor Adjustment** math mathematically conditions on the parents in the causal graph to block all non-causal paths.

---

### 🏆 The Elite 7 Features Selected by CausalShield
Out of 39 candidates, only **7 features** survived the FCI causal graph link and ACE thresholding ($|\text{ACE}| \ge 0.05$):

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        THE 7 PROTOCOL-LEVEL CAUSAL INVARIANTS                          │
├───────────────────┬───────────────────────────────────┬────────────────────────────────┤
│ Feature Name      │ Layman Meaning                    │ Why it Causally Matters        │
├───────────────────┼───────────────────────────────────┼────────────────────────────────┤
│ 1. Duration       │ How long the connection lasted    │ Direct ancestor (→). Attacks   │
│                   │                                   │ either blast in micro-bursts   │
│                   │                                   │ or hang open indefinitely.     │
├───────────────────┼───────────────────────────────────┼────────────────────────────────┤
│ 2. Variance       │ Packet size variation             │ Latent confounding (↔). Normal │
│                   │                                   │ browsing has erratic sizes;    │
│                   │                                   │ automated botnets send fixed   │
│                   │                                   │ repetitive packet payloads.    │
├───────────────────┼───────────────────────────────────┼────────────────────────────────┤
│ 3. rst_count      │ Total number of TCP RST flags     │ Latent confounding (↔). RST    │
│                   │                                   │ forces a connection to abort;  │
│                   │                                   │ scanners generate storms of    │
│                   │                                   │ broken connections.            │
├───────────────────┼───────────────────────────────────┼────────────────────────────────┤
│ 4. rst_flag_number│ Presence of TCP RST flag          │ Latent confounding (↔). Flags  │
│                   │                                   │ sudden aborted handshakes.     │
├───────────────────┼───────────────────────────────────┼────────────────────────────────┤
│ 5. ack_flag_number│ Presence of TCP ACK flag          │ Latent confounding (↔). Hand-  │
│                   │                                   │ shake confirmation. Floods     │
│                   │                                   │ abuse ACK sequence mechanics.  │
├───────────────────┼───────────────────────────────────┼────────────────────────────────┤
│ 6. psh_flag_number│ Presence of TCP PSH (Push) flag   │ Latent confounding (↔). Forces │
│                   │                                   │ immediate buffer push; common  │
│                   │                                   │ in command-injection attacks.  │
├───────────────────┼───────────────────────────────────┼────────────────────────────────┤
│ 7. HTTPS          │ Indicator for encrypted port 443  │ Latent confounding (↔). Service│
│                   │                                   │ protocol context confounder.   │
└───────────────────┴───────────────────────────────────┴────────────────────────────────┘
```

---

### ❓ What Happens If We DON'T Include ACE Filtering?
- You are left with an unranked list of graph neighbors. Without calculating ACE proxy strengths, you wouldn't know which features have strong effects and which are negligible.
- Keeping all 39 features forces the model to learn 5x more decision tree splits, slowing down detection latency and opening the door to overfitting.

---

# 🌲 Module 5: Risk Scoring (Phase 4 — XGBoost & Calibration)

### 🚀 Why XGBoost (Extreme Gradient Boosting)?
XGBoost is the premier algorithm for tabular structured data:
- It builds an ensemble of shallow decision trees sequentially, where each new tree specifically corrects the mistakes made by the previous trees.
- It handles complex non-linear thresholds (e.g., *"If duration $< 0.1\text{s}$ AND variance $= 0$, high risk"*).
- It is lightning-fast at inference time (sub-2 milliseconds).

---

### ⚖️ The Hidden Flaw of Decision Trees: Uncalibrated Probabilities
Most engineers assume that if an ML model outputs `0.85`, it means there is an **85% chance** that the flow is an attack.  
**That is completely false!**  
Decision trees compute scores based on how many training samples landed in a leaf node. They are notorious for producing **overconfident or distorted scores** (e.g., clustering scores around 0.1 and 0.9, but rarely representing true physical probabilities).

```
         UNRELIABLE (Raw Tree Output)             RELIABLE (Isotonically Calibrated)
         ┌───────────────────────────┐            ┌────────────────────────────────┐
         │ Raw Score = 0.80          │            │ Calibrated Probability = 0.80  │
         │ Real Attack Rate = 52%    │    ───►    │ Real Attack Rate = 80%         │
         │ (Overconfident & Misleading)│          │ (Mathematically Trustworthy!)  │
         └───────────────────────────┘            └────────────────────────────────┘
```

### 📈 The Fix: Isotonic Regression Calibration
CausalShield passes raw XGBoost predictions through **Isotonic Regression** fitted on the validation set.  
Isotonic regression fits a non-parametric, monotonic step function that straightens the calibration curve. If you take 100 flows with a calibrated score of 0.80, exactly 80 of them are guaranteed to be true attacks!

---

### 🛡️ 5-Fold Stratified Cross-Validation
To guarantee that these results were not a lucky fluke on one split, CausalShield runs **5-Fold Cross-Validation**:
- The dataset is split into 5 equal parts. The model is trained on 4 parts and evaluated on the 5th, repeated 5 times.
- Result: **ROC-AUC = $0.9381 \pm 0.0009$**. The tiny standard deviation ($< 0.001$) proves that the model's performance is stable across different splits.

---

### ❓ What Happens If We DON'T Include Calibration?
- Security teams cannot trust the risk scores.
- A SOC manager cannot set automated firewall rules based on thresholds because a score of 0.70 might mean 90% risk today and 40% risk tomorrow.

---

# 🚦 Module 6: Operational Three-Tier Policy (Phase 5 — Eliminating Alert Fatigue)

### 😫 The Crisis in Modern SOCs: Alert Fatigue
In a modern Security Operations Center (SOC), human security analysts monitor network alarms all day.  
If your AI uses a simple binary threshold (e.g., $\ge 0.5 \to \text{Alert}$), the SOC team receives **10,000 notifications an hour**.  
Over 90% of them are false alarms! Analysts burn out, ignore the sirens, and miss the actual intrusion.

---

### 🚥 The Three-Tier Solution
CausalShield translates continuous calibrated probabilities into three operationally clear buckets:

```
    0.0 ────────────── 0.250 ──────────────────────── 0.650 ────────────── 1.0
             LOW                     MEDIUM                    HIGH
      (Monitor Silently)        (Investigate)           (Block & Alarm)
```

| Tier Name | Risk Range | Target Guarantee | SOC Operational Action | Actual Attack Share in Test Set |
|---|---|---|---|:---:|
| 🟢 **LOW RISK** | $< 0.250$ | False Positive Rate $< 5\%$ | **Silent Monitoring.** Log in background. Do not wake up human analysts. | **16.2% – 17.2%** |
| 🟡 **MEDIUM RISK** | $0.250 \le p < 0.650$ | Gray transition zone | **Investigate / Rate-Limit.** Queue for secondary review or challenge with CAPTCHA. | **45.1% – 46.3%** |
| 🔴 **HIGH RISK** | $\ge 0.650$ | True Positive Rate $> 95\%$ | **Immediate Automated Block.** Drop packet, sever TCP socket, trigger emergency siren. | **95.2% – 95.5%** |

---

### ❓ What Happens If We DON'T Include Three-Tier Triage?
- Security teams are forced into binary all-or-nothing thinking.
- If you lower the threshold to catch every attack, your system floods analysts with false alarms.
- If you raise the threshold to avoid false alarms, stealthy intrusions slip past undetected.

---

# 🔍 Module 7: Explainability via Tree SHAP (Phase 6 — No More Black Boxes)

### ❓ The Problem: "Why Did You Block My CEO's Computer?"
If an automated system blocks a network connection and the network administrator asks: *"Why was this connection blocked?"*, an engineer cannot simply answer: *"Because tree #47 branched left."*  
Security operations require **explainability, accountability, and auditability**.

---

### 🎁 What is SHAP in Layman's Terms?
**SHAP (Shapley Additive exPlanations)** comes from Nobel Prize-winning cooperative game theory.  
Imagine 7 soccer players on a team who win a match 3–0. How much prize money does each individual player deserve?
- Did the striker win it alone?
- Or did the goalkeeper make key saves that enabled the win?

SHAP evaluates every combination of players (features) with and without each member to compute their exact mathematical contribution to the final score.

$$\text{Final Flow Risk Score} = \text{Base Expected Value} + \sum_{i=1}^{7} \text{SHAP Value}_i$$

---

### 💧 The Waterfall Plot: Dissecting a Single Decision
For every single network packet flow, CausalShield generates an instant **SHAP Waterfall Plot**:

```
[Base Baseline Risk: 0.50]
   │
   ├─► (+) rst_count = 142       [+0.28]  ─── Storm of aborts detected
   ├─► (+) Variance = 0.00       [+0.14]  ─── Robotically identical packet sizes
   ├─► (+) Duration = 0.002s     [+0.09]  ─── Microsecond burst connection
   ├─► (-) HTTPS = 1             [-0.08]  ─── Standard encrypted port behavior
   │
   ▼
[Final Calibrated Risk: 0.93 ── HIGH TIER ── AUTO BLOCK!]
```

Analyst takeaway: In under 5 seconds, an analyst sees that `rst_count` and zero `Variance` pushed this flow into the danger zone.

---

### ❓ What Happens If We DON'T Include SHAP Explainability?
- The AI remains an unaccountable black box.
- When an executive’s videoconference or an industrial valve communication is mistakenly blocked, nobody knows why.
- Regulatory frameworks (like the EU AI Act) prohibit high-risk unexplainable automated decision systems in critical infrastructure.

---

# 🌍 Module 8: The Acid Test — Domain Transfer to ToN-IoT (Phase 7)

### 🧪 What is Cross-Dataset Evaluation?
Testing an AI model on new samples from the *same* dataset is like testing a student with the exact same practice exam questions they memorized last night.  
To prove genuine intelligence, you must give the student a **completely different exam written by a different professor**.

- **Training/Tuning Domain:** CICIoT2023 (Canadian Institute for Cybersecurity testbed).
- **Evaluation Domain:** **ToN-IoT** (UNSW Canberra testbed with industrial sensors, telemetry, and different operating systems).
- **The Condition:** The trained XGBoost model was **completely frozen**—not a single tree branch or weight was retrained!

---

### 💥 The Benchmark Showdown: What Happened on 6.90 Million Held-Out Flows?

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   CROSS-DATASET GENERALIZATION (HELD-OUT ToN-IoT)                      │
├───────────────────────────────┬──────────────┬──────────────────┬──────────────────────┤
│ Model Architecture            │ Feature Type │ ROC-AUC Score    │ Real-World Outcome   │
├───────────────────────────────┼──────────────┼──────────────────┼──────────────────────┤
│ 1. Full-Feature XGBoost       │ 39 Features  │ 0.4976 (49.7%)   │ ❌ TOTAL COLLAPSE     │
│    (Standard Cyber AI)        │ (No Causal)  │                  │ Worse than coin flip │
├───────────────────────────────┼──────────────┼──────────────────┼──────────────────────┤
│ 2. Mutual Information XGBoost │ 7 Features   │ 0.3688 (36.8%)   │ ❌ INVERTED SCORE     │
│    (Correlation Selection)    │ (Correlated) │                  │ Complete Chaos       │
├───────────────────────────────┼──────────────┼──────────────────┼──────────────────────┤
│ 3. CausalShield (Proposed)    │ 7 Features   │ 0.8125 (81.3%)   │ ✅ RESILIENT & STABLE│
│    (FCI Causal Invariants)    │ (Causal)     │                  │ High Generalization  │
└───────────────────────────────┴──────────────┴──────────────────┴──────────────────────┘
```

#### Why did the standard models crash?
- **Full-Feature XGBoost (AUC 0.4976):** Memorized all 39 hardware timing quirks of CICIoT2023. When placed in ToN-IoT, those timings were meaningless. It became a random coin generator.
- **Mutual Information (AUC 0.3688):** It latched so hard onto superficial lab correlations that on ToN-IoT, it classified benign traffic as attacks and attacks as benign!
- **CausalShield (AUC 0.8125):** Because TCP connection resets (`rst_count`), packet size variances (`Variance`), and flow durations (`Duration`) represent **the physics of computer networking**, they remain true indicators of attacks regardless of whose lab or router runs the network!

---

### 🎚️ The 1% Few-Shot Calibration Split
While the causal features remained invariant, the absolute numerical scale differed across testbeds (e.g., ToN-IoT captured traffic at different link speeds).  
CausalShield evaluated an operational **Few-Shot setup**:
- A tiny **1% calibration slice** ($48,472$ flows) was used solely to recalibrate the score threshold ($\tau^* = 0.023$).
- The remaining **99% of flows ($4.80\text{ Million flows}$)** were strictly held out.
- Result: **Attack F1 reached 0.8149**, with Benign F1 reaching **0.7484** (over **3.7× higher benign reliability** than the standard baseline).

---

### ❓ What Happens If We DON'T Perform Cross-Dataset Evaluation?
- You publish an academic paper claiming "99.9% accuracy" that is completely useless in real deployments.
- You deploy a system into a hospital or manufacturing plant that fails on day one.

---

# 🖥️ Module 9: The Interactive SOC Dashboard (Phase 8)

### 🕹️ The Human-in-the-Loop Cockpit
Code in a terminal is not how security operators work during a midnight cyber incident. CausalShield includes a complete web application built with **Streamlit** (`dashboard/app.py`):

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    CAUSALSHIELD REAL-TIME SOC DASHBOARD                      │
├──────────────────────────────────────────────────────────────────────────────┤
│ [FLOW INSPECTION PANEL]               [RISK SCORE GAUGE]                     │
│ Duration:       0.042 s               ┌────────────────────────┐             │
│ Packet Var:     0.000                 │     0.887 / 1.000      │             │
│ RST Count:      87                    │  [ 🔴 HIGH RISK TIER ] │             │
│ ACK Flag:       1                     └────────────────────────┘             │
│ HTTPS Service:  0                     Action: BLOCK FLOW IMMEDIATELY         │
├──────────────────────────────────────────────────────────────────────────────┤
│ [EXPLANATION WATERFALL]               [CAUSAL GRAPH VIEW]                    │
│ • rst_count pushed risk up (+0.31)    Displaying Partial Ancestral Graph     │
│ • Variance pushed risk up (+0.18)     7 nodes, 70 edges                      │
│ • Duration pushed risk down (-0.05)   Causal path: Duration ──► Attack       │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

# 📊 Module 10: Master Comparison Matrix ("What If We Remove It?")

| Component / Technique | What it Does (In Layman Terms) | What Happens If We DON'T Include It? |
|---|---|---|
| **Phase 1: Stratified Class Balancing** | Ensures equal representation of attacks and benign flows during training. | The model predicts the majority class every time; rare attacks are missed completely. |
| **Phase 2: FCI Causal Discovery** | Maps the network graph while accounting for unmeasured variables (router load, CPU state). | The model falls back on correlation and picks up dataset-specific capture artifacts. |
| **Phase 3: ACE / Do-Calculus** | Measures the true causal impact by closing backdoor confounding paths. | We cannot rank causal features by strength and end up guessing which features matter. |
| **Phase 4: Isotonic Probability Calibration** | Converts raw decision tree scores into true physical probabilities. | A score of "0.80" cannot be trusted; risk numbers become erratic and unpredictable. |
| **Phase 5: Three-Tier SOC Policy** | Separates flows into Low (monitor), Medium (investigate), and High (block). | Analysts suffer severe alert fatigue, leading to ignored alarms and missed breaches. |
| **Phase 6: Tree SHAP Explainability** | Calculates exact feature contributions for every single decision. | The AI remains an untrustworthy black box; audits and troubleshooting become impossible. |
| **Phase 7: Cross-Dataset Evaluation (ToN-IoT)** | Tests the frozen model on millions of flows from an unseen network. | You are left with a model that looks great in the lab but fails completely in production. |
| **Phase 8: Interactive Dashboard** | Translates mathematics into an intuitive operational control room. | The technology stays trapped in Python scripts, inaccessible to real security staff. |

---

# 📚 Module 11: Quick Revision Glossary & Exam Prep

### 🧠 Core Terminology in Plain English

- **Causation vs. Correlation:**  
  *Correlation:* Roosters crow and the sun rises together.  
  *Causation:* The sun rising causes daylight; killing the rooster does not stop the sunrise.
- **PAG (Partial Ancestral Graph):**  
  A causal graph that explicitly shows what is certain, what is uncertain, and where hidden variables exist.
- **Latent Confounder:**  
  A hidden, unmeasured common cause that influences two observed variables simultaneously (e.g., router queue congestion).
- **Do-Calculus / Interventional Effect:**  
  The mathematical calculation of what happens when you physically intervene and change a variable, rather than just passively observing it.
- **Backdoor Path:**  
  A misleading non-causal statistical correlation between two variables caused by a shared common ancestor.
- **Tree SHAP:**  
  An algorithm using game theory to compute feature contributions across decision tree paths.
- **ROC-AUC (Receiver Operating Characteristic - Area Under Curve):**  
  A metric from 0.0 to 1.0 showing how well a model separates positive and negative classes across all possible thresholds. 0.5 is a coin toss; 1.0 is perfection.
- **PR-AUC (Precision-Recall Area Under Curve):**  
  The gold standard metric when dealing with imbalanced datasets. Measures detection quality without being inflated by abundant easy negatives.

---

### ❓ Top 5 Interview / Viva Questions About This Project

#### 1. Why didn't you use deep learning (LSTMs or Transformers)?
> **Answer:** Network intrusion detection happens at line rate (gigabits per second). Deep learning models have high computational latency and function as opaque black boxes. CausalShield uses a 7-feature XGBoost model that executes in **1.42–1.64 milliseconds per flow**, requires minimal compute, and is fully explainable via Tree SHAP.

#### 2. Why does your model score 93.9% in-domain while traditional models score 99.4%? Is your model worse?
> **Answer:** Traditional models achieve 99.4% by overfitting to testbed artifacts (like packet timing jitter). Our model trades ~5.5 percentage points of lab accuracy to purge non-causal variables. When tested on unseen networks (ToN-IoT), the traditional models collapse to **36.8%–49.7%**, while CausalShield maintains **81.25% ROC-AUC**.

#### 3. Why did you use FCI instead of the classic PC algorithm?
> **Answer:** The PC algorithm assumes causal sufficiency (that all causes are observed). Real network traffic always involves unmeasured variables (router queuing delays, CPU spikes, background traffic). FCI explicitly models these latent confounders through bidirected edges in a Partial Ancestral Graph (PAG).

#### 4. How did you handle the scale differences between CICIoT2023 and ToN-IoT?
> **Answer:** The causal invariants (like TCP RST and ACK flags) maintain their qualitative relationships across domains, but absolute numbers (like duration distributions) can shift. We used a strict **1% few-shot calibration split** solely to tune the decision threshold ($\tau^* = 0.023$) without retraining the frozen model weights, preserving out-of-domain integrity on the remaining 99% of data.

#### 5. Why are packet inter-arrival times (IAT) excluded from your 7 features?
> **Answer:** IAT measures the delay between arriving packets. In a controlled testbed, attack scripts send packets at rigid, artificial intervals. In production, network congestion, user activity, and internet routing alter IAT continuously. FCI and ACE recognize that IAT is an artifact of the capture environment rather than an invariant cause of an intrusion.

---
*Created as study reference notes for the CausalShield research project.*
