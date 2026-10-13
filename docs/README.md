# FedMPS — Capstone Review & Documentation Guide

This document indexes all documentation, architectural specifications, benchmark metrics, and demo procedures for **Faculty Guide Verification (First Review & Second Review)**.

---

## 📋 Review Structure Overview

| Review Phase | Primary Deliverables | Reference Files |
| :--- | :--- | :--- |
| **First Review** | Problem statement, literature survey, threat model, system architecture, mathematical formulations (ArcFace, FedProx, DP-SGD, Cancelable Biometrics), and 12-week execution roadmap. | [First_Review_Documentation.pdf](file:///home/kumaresh/Desktop/Dev/fedmps-capstone/docs/First_Review_Documentation.pdf)<br>[01-design-report.html](file:///home/kumaresh/Desktop/Dev/fedmps-capstone/docs/01-design-report.html)<br>[02-twelve-week-plan.html](file:///home/kumaresh/Desktop/Dev/fedmps-capstone/docs/02-twelve-week-plan.html)<br>[03-implementation-spec.html](file:///home/kumaresh/Desktop/Dev/fedmps-capstone/docs/03-implementation-spec.html) |
| **Second Review** | Complete implementation, VGGFace2 empirical benchmark, ROC/DET/CMC curves, score histograms, privacy-utility analysis, multi-node video pipeline, and live interactive demo. | [Second_Review_Documentation.pdf](file:///home/kumaresh/Desktop/Dev/fedmps-capstone/docs/Second_Review_Documentation.pdf)<br>[04-demo-runbook.html](file:///home/kumaresh/Desktop/Dev/fedmps-capstone/docs/04-demo-runbook.html)<br>[benchmark_comparison_table.md](file:///home/kumaresh/Desktop/Dev/fedmps-capstone/results/benchmark_comparison_table.md)<br>[figures/](file:///home/kumaresh/Desktop/Dev/fedmps-capstone/figures/) |
| **Third Review** | Deep-dive code walkthrough, complete file-by-file defense, plot generation scripts, AWS GPU cloud training proof, full quantitative benchmark analysis, and top 10 tough faculty viva questions. | [01 Code Files Guide](file:///home/kumaresh/Desktop/Dev/fedmps-capstone/docs/Third_Review_01_Code_Files.md)<br>[02 Plots & Code Guide](file:///home/kumaresh/Desktop/Dev/fedmps-capstone/docs/Third_Review_02_Plots_and_Code.md)<br>[03 AWS Training Guide](file:///home/kumaresh/Desktop/Dev/fedmps-capstone/docs/Third_Review_03_AWS_Training_Results.md)<br>[04 Metrics & Results Guide](file:///home/kumaresh/Desktop/Dev/fedmps-capstone/docs/Third_Review_04_Metrics_and_Results.md)<br>[05 Faculty Viva Questions](file:///home/kumaresh/Desktop/Dev/fedmps-capstone/docs/Third_Review_05_Faculty_Viva_Questions.md)<br>[06 Complete Files Defense](file:///home/kumaresh/Desktop/Dev/fedmps-capstone/docs/Third_Review_06_Complete_Files_Defense_Guide.md) |

---

## 🏛️ First Review: Architecture & Theoretical Foundations

### 1. Problem Formulation
Surveillance-based missing person search faces two simultaneous challenges:
1. **Severe Domain Shift**: Real CCTV footage suffers from low resolution ($<30\text{px}$ inter-pupillary distance), motion blur, oblique camera angles, and poor lighting.
2. **Privacy & Regulatory Constraints**: Privacy laws prohibit centralizing live CCTV video streams from distributed physical locations (airports, transit hubs, retail centers) onto a single server.

### 2. Proposed System Architecture
FedMPS solves this by decoupling feature extraction from domain adaptation:
- **Frozen Backbone**: A pre-trained, high-capacity feature extractor (ResNet-50 ArcFace `buffalo_l` / `w600k_r50`) remains completely static on edge nodes.
- **Lightweight Edge Adapters**: A small, parameter-efficient adaptation module (MLP adapter) is trained locally on edge nodes to adjust feature distributions without fine-tuning the heavy backbone.
- **Secured Federated Aggregation**: Edge nodes compute gradient updates and share only model deltas via secure masked aggregation with FedProx optimization.
- **Cancelable Watchlist Distribution**: Missing person reference photos are never sent to nodes. Instead, irreversible pseudo-random biometric projections (BioHash / 512-bit Random Projections) are distributed.

### 3. Key Design Documentation Links
- **Comprehensive Design Report**: [docs/01-design-report.html](file:///home/kumaresh/Desktop/Dev/fedmps-capstone/docs/01-design-report.html)
- **12-Week Implementation Plan**: [docs/02-twelve-week-plan.html](file:///home/kumaresh/Desktop/Dev/fedmps-capstone/docs/02-twelve-week-plan.html)
- **Implementation Specification & Module Contracts**: [docs/03-implementation-spec.html](file:///home/kumaresh/Desktop/Dev/fedmps-capstone/docs/03-implementation-spec.html)
- **Official First Review Document**: [docs/First_Review_Documentation.pdf](file:///home/kumaresh/Desktop/Dev/fedmps-capstone/docs/First_Review_Documentation.pdf)

---

## 📊 Second Review: Empirical Results, Metrics & Demo

### 1. Benchmark Dataset & Evaluation Protocol
- **Dataset**: Full **VGGFace2 CCTV Surveillance Benchmark** (8,631 total identities, 172,620 images total).
- **Partitioning**: Non-IID Dirichlet distribution across **5 federated edge camera nodes** (Node 0: 258 IDs, Node 1: 388 IDs, Node 2: 517 IDs, Node 3: 647 IDs, Node 4: 779 IDs).
- **Evaluation Gallery**: Held-out set of **3,021 identities** with 15,105 mate probes, 18,126 open-set non-mate crowd probes, and 6,000 1:1 verification pairs.
- **Reference Log**: [full_run_vgg_big.log](file:///home/kumaresh/Desktop/Dev/fedmps-capstone/full_run_vgg_big.log)

---

### 2. Verified Benchmark Comparison Tables

#### Table 1: Primary Comparison across Surveillance Learning Paradigms
*(Matches the standard academic biometric benchmark reporting format)*

| Training Paradigm / Method | Backbone Arch. | 1:1 Verification TAR @ FAR | | | 1:N Identification TPIR @ FPIR | | 1:N Closed-Set Rank Retrieval | | |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| | | **FAR = 0.001** | **FAR = 0.01** | **FAR = 0.1** | **FPIR = 0.01** | **FPIR = 0.1** | **Rank-1** | **Rank-5** | **Rank-10** |
| **Clean Probe Benchmark** *(Upper Bound)* | ResNet-50 ArcFace | 0.850 | 0.940 | 0.982 | 0.892 | 0.954 | **0.996** | 0.999 | 1.000 |
| **Surveillance Baseline** *(Frozen, No Adapter)* | ResNet-50 ArcFace | 0.629 | 0.744 | 0.871 | 0.797 | 0.890 | **0.947** | 0.978 | 0.988 |
| **Centralised Adapter** *(Pooled Data)* | ResNet-50 + Adapter | 0.604 | 0.738 | 0.868 | 0.798 | 0.891 | **0.945** | 0.977 | 0.988 |
| **FedMPS (Federated Adapter)** *(Ours)* | ResNet-50 + Adapter | **0.618** | **0.740** | **0.870** | **0.804** | **0.893** | **0.945** | **0.976** | **0.987** |
| **Local-Only Training** *(Mean across 5 edge cameras)* | ResNet-50 + Adapter | 0.589 ± 0.018 | 0.715 ± 0.017 | 0.842 ± 0.015 | 0.770 ± 0.016 | 0.865 ± 0.013 | 0.928 ± 0.008 | 0.961 ± 0.006 | 0.975 ± 0.005 |
| **Cancelable Template** *(512-bit BioHash / RP)* | ResNet-50 + 512-bit | 0.442 | 0.605 | 0.781 | 0.594 | 0.742 | 0.857 | 0.914 | 0.938 |
| **Cancelable Template** *(96-bit Prior Literature)* | ResNet-50 + 96-bit | 0.118 | 0.224 | 0.485 | 0.108 | 0.284 | 0.396 | 0.521 | 0.589 |

---

#### Table 2: 1:1 Verification & 1:N Identification at Strict Operating Points

| Training Paradigm | Arch. | TAR (FAR = 1e-4) | TAR (FAR = 1e-3) | TAR (FAR = 1e-2) | TPIR (FPIR = 0.01) | TPIR (FPIR = 0.1) | Rank-1 | Rank-5 | Rank-10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **VGGFace2 (Clean Probe Benchmark)** | ResNet-50 | 0.782 | 0.850 | 0.940 | 0.892 | 0.954 | 0.996 | 0.999 | 1.000 |
| **Surveillance Baseline (Degraded)**| ResNet-50 | 0.512 | 0.629 | 0.744 | 0.797 | 0.890 | 0.947 | 0.978 | 0.988 |
| **FedMPS (Federated Adapter)** | ResNet-50 + Adapter | **0.507** | **0.618** | **0.740** | **0.804** | **0.893** | **0.945** | **0.976** | **0.987** |
| **Centralised (Pooled Upper Bound)**| ResNet-50 + Adapter | 0.501 | 0.604 | 0.738 | 0.798 | 0.891 | 0.945 | 0.977 | 0.988 |
| **Local Only (Mean across 5 Sites)** | ResNet-50 + Adapter | 0.465 ± 0.022 | 0.589 ± 0.018 | 0.715 ± 0.017 | 0.770 ± 0.016 | 0.865 ± 0.013 | 0.928 ± 0.008 | 0.961 ± 0.006 | 0.975 ± 0.005 |
| **Maze et al. [14] (IJB-A Benchmark)**| — | 0.750 | 0.860 | 0.950 | 0.450 | 0.620 | 0.790 | 0.870 | 0.900 |

---

### 3. Publication Figures (Location: `figures/`)

1. **[figures/fig_benchmark_roc_det_cmc.png](file:///home/kumaresh/Desktop/Dev/fedmps-capstone/figures/fig_benchmark_roc_det_cmc.png)**:
   - **Row 1 Left (ROC)**: 1:1 Verification ROC Curve ($10^{-4}$ to $1.0$).
   - **Row 1 Right (DET)**: 1:N Identification DET Curve ($10^{-4}$ to $1.0$).
   - **Row 2 Wide (CMC)**: 1:N Cumulative Match Characteristic from Rank 1 to 50.
2. **[figures/fig_similarity_histograms.png](file:///home/kumaresh/Desktop/Dev/fedmps-capstone/figures/fig_similarity_histograms.png)**:
   - Genuine probe similarity score distributions (Clean vs. Surveillance CCTV vs. FedMPS Domain Adaptation).
3. **[figures/fig2_three_way.png](file:///home/kumaresh/Desktop/Dev/fedmps-capstone/figures/fig2_three_way.png)**:
   - Three-way bound comparison: No Adapter (0.947) vs. Local-only (0.928) vs. Federated (0.945) vs. Centralised (0.945).
4. **[figures/fig5_det.png](file:///home/kumaresh/Desktop/Dev/fedmps-capstone/figures/fig5_det.png)**:
   - Open-set surveillance identification DET curve comparing edge models.
5. **[figures/fig7_fedprox.png](file:///home/kumaresh/Desktop/Dev/fedmps-capstone/figures/fig7_fedprox.png)**:
   - Non-IID convergence comparing FedAvg ($\mu=0$) vs. FedProx ($\mu \in \{0.01, 0.1, 1.0\}$).
6. **[figures/fig6_templates.png](file:///home/kumaresh/Desktop/Dev/fedmps-capstone/figures/fig6_templates.png)**:
   - Accuracy vs. template width curve illustrating why 512-bit BioHash is optimal over 96-bit compact codes.

---

## 🚀 Live Demo Execution Guide for Faculty

The project provides an automated, two-tier demonstration of the real-time CCTV surveillance pipeline and privacy architecture.

### Step 1: Start the Web Dashboards
Open two terminal windows:

**Terminal 1 — Edge Node Operator Review Queue:**
```bash
source .venv/bin/activate
streamlit run fedmps/apps/node_app.py --server.port 8502
```
*Access in browser:* `http://localhost:8502`

**Terminal 2 — Central Law Enforcement Console:**
```bash
source .venv/bin/activate
streamlit run fedmps/apps/server_app.py --server.port 8501
```
*Access in browser:* `http://localhost:8501`

---

### Step 2: Clean Reset for Fresh Demo
Before demonstrating to your guide, execute:
```bash
python demo_reset.py
```
*Action*: Empties the SQLite store, wipes past case alerts, clear previous cropped faces, and ensures fresh state.

---

### Step 3: End-to-End Search Demo Flow
1. **Central Enrolment**:
   - In `http://localhost:8501`, navigate to **"Open a case"**.
   - Upload `media/person-1.jpg` (or any reference photograph), enter Subject Name (e.g., *"Alice"*), and click **"Create case and distribute"**.
   - Notice: Photographs remain confidential; cancelable template vectors are generated and dispatched.
2. **Execute Video Scanning on Edge Nodes**:
   Run the edge node CCTV scan from terminal:
   ```bash
   python demo_trigger.py --node 0
   ```
   *(Or run across all five simulated camera sites: `python demo_trigger.py --all`)*
3. **Edge Node Verification**:
   - Switch to `http://localhost:8502` (Node Review Queue).
   - An alert appears with face detection bounding box, similarity score, frame timestamp, and camera location.
   - Click **"Confirm Match"** or **"Reject False Alarm"**.
4. **Central Audit Synchronization**:
   - Refresh `http://localhost:8501` (Central Dashboard).
   - The verified alert appears in the case log with an immutable cryptographic audit record, confirming cross-site localization.

---

## 🔍 Code & Directory Verification Map

```
fedmps-capstone/
├── fedmps/                     # Core system package
│   ├── models/                 # ArcFace backbone & Domain Adapter architectures
│   │   ├── backbone.py         # RetinaFace + ArcFace feature extraction (buffalo_l)
│   │   └── adapter.py          # Residual MLP adapter for CCTV domain adaptation
│   ├── fl/                     # Federated Learning Engine
│   │   ├── server.py           # Secure aggregation server (FedAvg / FedProx)
│   │   ├── node.py             # Local edge node training loop
│   │   └── loop.py             # Synchronous & asynchronous FL orchestration
│   ├── privacy/                # Privacy Mechanisms
│   │   ├── template.py         # BioHash / Random Projection cancelable templates
│   │   ├── mask.py             # Cryptographic additive masking for model updates
│   │   └── dp.py               # Differential Privacy (DP-SGD & Rényi accounting)
│   ├── pipeline/               # CCTV Video Processing Pipeline
│   │   ├── detect.py           # 5-point landmark face alignment & detection
│   │   ├── quality.py          # Blur, illumination, and pose quality assessment
│   │   └── match.py            # Multi-frame fused gallery matching
│   ├── apps/                   # Interactive Web Dashboards
│   │   ├── server_app.py       # Central Police / Law Enforcement Console (Port 8501)
│   │   └── node_app.py         # Edge Station / CCTV Review Queue (Port 8502)
│   └── eval/                   # Evaluation & Verification Modules
│       ├── metrics.py          # TAR@FAR, TPIR@FPIR, Rank-N, EER, AUC calculations
│       └── experiments.py      # Experiment suites & benchmark visualization
├── docs/                       # Review Documentation & Reports
│   ├── README.md               # Master review roadmap (this file)
│   ├── First_Review_Documentation.pdf
│   ├── Second_Review_Documentation.pdf
│   ├── 01-design-report.html
│   ├── 02-twelve-week-plan.html
│   ├── 03-implementation-spec.html
│   └── 04-demo-runbook.html
├── figures/                    # Generated Benchmark Plots
│   ├── fig_benchmark_roc_det_cmc.png  # Primary 3-panel benchmark curves
│   ├── fig_similarity_histograms.png  # Genuine similarity score distributions
│   └── fig1_degradation.png ... fig8_operating_points.png
├── results/                    # Numerical Benchmark JSONs & Tables
│   ├── benchmark_comparison_table.md  # Official comparative markdown table
│   └── experiments_vgg_big.json       # Raw empirical evaluation record
├── media/                      # Demo surveillance video feeds & test photos
├── demo_trigger.py             # Interactive demo trigger script
├── demo_reset.py               # One-click state reset script
└── run_all_nodes.sh            # Automated batch execution script
```
