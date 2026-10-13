# Third Review Guide — Complete Project Files Defense

If the faculty guide asks:
> *"Why are there so many files in this project? What is each file for? Why didn't you just write everything in 1 or 2 scripts?"*

Use this guide. It maps **every single file in the repository**, explains its software engineering justification, and gives you a 1-sentence defense line for each file.

---

## 🏛️ High-Level Architectural Justification (What to say first)

> *"Sir/Ma'am, a production surveillance system that combines Deep Learning, Federated Optimization, Cryptographic Privacy, and Real-Time Video Streaming cannot be built as a monolithic script.*
>
> *Our codebase follows **Enterprise Clean Architecture & Separation of Concerns (SoC)** divided into 8 decoupled subsystems:
> 1. **Core Configuration & Contracts** (`config.py`, `contracts.py`, `store.py`)
> 2. **Surveillance Data & Degradation Layer** (`fedmps/data/`)
> 3. **Deep Learning Model Architectures** (`fedmps/models/`)
> 4. **Federated Optimization Engine** (`fedmps/fl/`)
> 5. **Cryptographic Privacy & Biometric Protection** (`fedmps/privacy/`)
> 6. **Real-Time Video Surveillance Pipeline** (`fedmps/pipeline/`)
> 7. **Interactive Law Enforcement Web Dashboards** (`fedmps/apps/`)
> 8. **Automated Evaluation, Metrics & Verification Suite** (`fedmps/eval/`)
>
> *Each file has a single responsibility, enabling independent unit testing, modular deployment on low-power edge devices, and strict privacy isolation."*

---

## 📂 Subsystem 1: Core Foundation & Storage

### 1. `fedmps/config.py` — Centralized Type-Safe Configuration
* **Purpose:** Single source of truth for all system hyperparameters using Python dataclasses.
* **Key Components:** `Config`, `DataConfig`, `BackboneConfig`, `AdapterConfig`, `FLConfig`, `PrivacyConfig`, `PipelineConfig`, `TrainConfig`.
* **Why it's essential:** Eliminates hardcoded magic numbers across the project. Changing training epochs, learning rate ($10^{-3}$), or template width (512 bits) happens here without touching core algorithmic logic.
* **Defense line:** *"It enforces strict type safety and centralized hyperparameter management across the entire federated network."*

### 2. `fedmps/contracts.py` — Pipeline Data Contracts & Schema Validation
* **Purpose:** Defines the exact disk layout and schema contracts for artifacts passed between data preparation, model training, and video inference.
* **Key Components:** `class Artefacts`, file constants (`F_EMB_CLEAN`, `F_EMB_DEG`, `F_LABELS`, `F_PARTITION`, `F_SAMPLES`).
* **Why it's essential:** Decouples Phase 1 (data prep) from Phase 2 (federated training). If data formats change, only `contracts.py` validates them, preventing silent corruption of embeddings.
* **Defense line:** *"It acts as an architectural boundary ensuring that data preparation and federated learning communicate through strict, validated data contracts."*

### 3. `fedmps/store.py` — SQLite Surveillance Database & Cryptographic Audit
* **Purpose:** Manages the persistent relational state for active missing person cases, cancelable watchlist templates, edge alerts, and immutable audit logs.
* **Key Components:** `class Store`, `open_store()`, tables: `cases`, `templates`, `alerts`, `audit`.
* **Why it's essential:** Edge nodes and police headquarters need a thread-safe, ACID-compliant database to track case lifecycles, template revocations, and operator approvals without requiring heavy external database servers like PostgreSQL.
* **Defense line:** *"It provides an embedded, ACID-compliant audit store tracking every alert and template lifecycle for legal accountability."*

---

## 📂 Subsystem 2: Surveillance Data & Degradation Layer (`fedmps/data/`)

### 4. `fedmps/data/loader.py` — Universal Face Dataset Ingestion
* **Purpose:** Ingests facial datasets (VGGFace2, LFW, or custom directory hierarchies) into unified identity mappings.
* **Key Components:** `load_identities_folder()`, `load_identities_lfw()`, `class FaceSample`.
* **Why it's essential:** Normalizes diverse folder structures, filters identities with too few images, and parses metadata before passing images to the embedder.
* **Defense line:** *"It normalizes raw surveillance image directories into verified identity batches."*

### 5. `fedmps/data/degrade.py` — CCTV Realistic Degradation Model
* **Purpose:** Mathematically simulates real-world CCTV camera distortions on clean facial imagery.
* **Key Components:** `degrade_image()`, `CameraProfile` (Gaussian blur $\sigma \in [0.5, 2.5]$, bilinear downsampling to $<30\text{px}$ IPD, JPEG compression artifacts $Q \in [10, 50]$, sensor Poisson noise, illumination shifts).
* **Why it's essential:** Real surveillance cameras produce degraded footage, but public face recognition models are trained on studio portraits. This module creates the domain shift that the federated adapter is trained to correct.
* **Defense line:** *"It accurately models the optical and sensor degradation of physical CCTV cameras to generate realistic training data."*

### 6. `fedmps/data/partition.py` — Non-IID Dirichlet Network Partitioning
* **Purpose:** Partitions identities and images across 5 federated edge nodes using an unbalanced Dirichlet-multinomial distribution.
* **Key Components:** `partition_dirichlet()`, `class NodePartition`.
* **Why it's essential:** In real life, CCTV cameras do not observe identical crowds (a train station sees $10\times$ more people than a side alley). This file creates realistic non-IID conditions to validate FedProx.
* **Defense line:** *"It synthetically partitions data into non-IID distributions reflecting real-world camera traffic imbalances."*

### 7. `fedmps/data/celeba.py` — CelebA Benchmark Loader
* **Purpose:** Data adapter used during initial Step 0 baseline experiments on the CelebA dataset.
* **Why it's essential:** Preserved for reproducibility of the ablation history table (Step 0: 48.07% Rank-1).
* **Defense line:** *"It preserves the experimental reproducibility of our initial baseline CelebA ablation benchmark."*

---

## 📂 Subsystem 3: Deep Learning Model Architectures (`fedmps/models/`)

### 8. `fedmps/models/backbone.py` — Face Detection & ArcFace Feature Extraction
* **Purpose:** Wraps RetinaFace (`det_10g.onnx`) and ArcFace ResNet-50 (`w600k_r50.onnx`), with fallback embedders for CPU/testing.
* **Key Components:** `class Backbone`, `ArcFaceBackbone`, `DummyBackbone`, 5-point facial landmark similarity transform alignment.
* **Why it's essential:** Extracts high-dimensional 512-d facial embeddings. Its landmark alignment places eyes, nose, and mouth on exact coordinates, which is critical for ArcFace accuracy.
* **Defense line:** *"It wraps the frozen 43.6M-parameter ArcFace backbone and performs 5-point facial landmark alignment."*

### 9. `fedmps/models/adapter.py` — Residual MLP Domain Adapter
* **Purpose:** The parameter-efficient neural network that is trained and federated across edge nodes.
* **Key Components:** `class Adapter` ($512 \rightarrow 64 \rightarrow 512$), `_init_weights()`, `forward()`, state dict delta arithmetic (`sub_states`, `add_states`, `weighted_mean_states`).
* **Why it's essential:** Adapts degraded CCTV embeddings toward clean gallery space with only 66k parameters, reducing bandwidth by $650\times$.
* **Defense line:** *"It implements our core lightweight residual adapter that is federated across edge cameras."*

---

## 📂 Subsystem 4: Federated Optimization Engine (`fedmps/fl/`)

### 10. `fedmps/fl/node.py` — Local Edge Client Training
* **Purpose:** Executes local gradient descent on each edge camera's private data slice.
* **Key Components:** `local_train()`, InfoNCE contrastive loss, FedProx proximal regularization ($\frac{\mu}{2}\|w - w_{\text{global}}\|^2$), delta extraction.
* **Why it's essential:** Ensures raw surveillance frames never leave the edge camera; only weight deltas $\Delta w$ are computed and returned.
* **Defense line:** *"It executes privacy-preserving local training at the camera node and calculates FedProx weight deltas."*

### 11. `fedmps/fl/server.py` — Central Federated Aggregation Server
* **Purpose:** Coordinates federated rounds, collects deltas, applies sample-weighted averaging, and integrates privacy hooks.
* **Key Components:** `aggregate()`, `apply_privacy()` (DP hook), `secure_aggregate()` (masking hook).
* **Why it's essential:** Coordinates the federated learning protocol without ever inspecting client images or facial embeddings.
* **Defense line:** *"It performs secure, sample-weighted aggregation of client model updates at the central server."*

### 12. `fedmps/fl/loop.py` — Federated Orchestration & Convergence Loop
* **Purpose:** Runs multi-round training loops, evaluates models on held-out validation identities, and tests the three-way bounds.
* **Key Components:** `run_federated()`, `run_three_way()`, early stopping validation checks.
* **Why it's essential:** Automates the full 30-round federated training cycle and validates that federated learning beats local-only training.
* **Defense line:** *"It coordinates multi-round federated convergence, early stopping, and three-way bound validation."*

---

## 📂 Subsystem 5: Privacy & Biometric Security (`fedmps/privacy/`)

### 13. `fedmps/privacy/template.py` — Cancelable Biometric Templates
* **Purpose:** Converts raw 512-d float embeddings into 512-bit revocable, distance-preserving binary templates.
* **Key Components:** `TemplateSelector`, `protect()`, `binarise()`, `derive_key()`, Hamming distance matching.
* **Why it's essential:** Prevents facial photographs from being sent to edge nodes; distributes irreversibly masked templates that can be reissued if a camera is stolen.
* **Defense line:** *"It implements our 512-bit BioHash cancelable template engine for GDPR-compliant watchlist distribution."*

### 14. `fedmps/privacy/mask.py` — Cryptographic Pairwise Masked Aggregation
* **Purpose:** Implements additive secret sharing for model updates.
* **Key Components:** `class SecureAggregator`, pairwise pseudo-random seed negotiation, zero-sum masking ($\sum M_{ij} = 0$).
* **Why it's essential:** Ensures the central server cannot inspect an individual camera's model update; individual updates are masked, but masks cancel out when summed.
* **Defense line:** *"It provides cryptographic additive masking so the server only sees the aggregate model, not individual camera deltas."*

### 15. `fedmps/privacy/dp.py` — Differential Privacy (DP-SGD & Rényi Accountant)
* **Purpose:** Implements $(\epsilon, \delta)$-Differential Privacy via gradient L2-norm clipping and Gaussian noise injection.
* **Key Components:** `privatise()`, `clip_gradient()`, `compute_epsilon()`, Rényi DP accountant.
* **Why it's essential:** Bounds the maximum influence of any single person's face on the federated model update and computes formal privacy budgets.
* **Defense line:** *"It implements formal $(\epsilon, \delta)$-Differential Privacy through L2 gradient clipping and Gaussian noise calibration."*

---

## 📂 Subsystem 6: Real-Time CCTV Video Pipeline (`fedmps/pipeline/`)

### 16. `fedmps/pipeline/detect.py` — Real-Time Video Detection & Face Tracking
* **Purpose:** Detects faces in streaming video frames, extracts 5 landmarks, and maintains persistent object tracks.
* **Key Components:** `class FaceDetector`, `class Tracker`, `class Track`, IoU/SORT tracking.
* **Why it's essential:** Bridges the gap between static image models and real CCTV video streams by tracking moving persons across continuous frames.
* **Defense line:** *"It handles real-time face detection, 5-point landmark alignment, and spatio-temporal tracklet tracking in video."*

### 17. `fedmps/pipeline/quality.py` — Surveillance Quality Gate
* **Purpose:** Evaluates visual quality of detected face crops before wasting neural network compute.
* **Key Components:** `class QualityGate`, Laplacian variance blur score, illumination analysis, yaw/pitch pose estimation.
* **Why it's essential:** Rejects extremely blurry faces or sideways profiles ($>45^\circ$), eliminating false alarms before embedding.
* **Defense line:** *"It acts as a pre-inference quality filter, discarding severe blur and oblique poses to prevent false alarms."*

### 18. `fedmps/pipeline/match.py` — Video Matcher & Multi-Frame Voting
* **Purpose:** Matches tracked video faces against the cancelable watchlist and enforces multi-frame voting rules.
* **Key Components:** `class Matcher`, `process_frame()`, $3\times 3$ temporal voting logic, alert generation.
* **Why it's essential:** Integrates detection, quality gating, adapter inference, template XOR matching, and voting into a unified video pipeline.
* **Defense line:** *"It executes end-to-end multi-frame tracklet voting against the watchlist to raise high-confidence alerts."*

---

## 📂 Subsystem 7: Interactive Law Enforcement Web Dashboards (`fedmps/apps/`)

### 19. `fedmps/apps/server_app.py` — Central Police Headquarters Console (Port 8501)
* **Purpose:** Law enforcement management console built with Streamlit.
* **Key Components:** "Open a Case" UI, reference photo upload, automatic cancelable template generation, confirmed alert visualization, cryptographic audit log viewer.
* **Why it's essential:** Provides the command-and-control dashboard where police open cases and view confirmed sightings.
* **Defense line:** *"It provides the central law enforcement interface for case creation, watchlist distribution, and audit verification."*

### 20. `fedmps/apps/node_app.py` — Edge Station CCTV Review Queue (Port 8502)
* **Purpose:** Edge station review queue implementing the Human-in-the-Loop policy.
* **Key Components:** Alert candidate viewer, crop comparison, similarity score gauge, operator "Confirm Match" and "Reject False Alarm" buttons.
* **Why it's essential:** Ensures edge cameras cannot dispatch police autonomously; an edge operator must verify candidate alerts locally.
* **Defense line:** *"It provides the edge human-in-the-loop review interface where local operators verify candidate detections."*

---

## 📂 Subsystem 8: Evaluation & Verification Suite (`fedmps/eval/`)

### 21. `fedmps/eval/metrics.py` — Biometric Verification & Identification Metrics
* **Purpose:** Implements official NIST/biometric evaluation math.
* **Key Components:** `compute_tar_at_far()`, `compute_tpir_at_fpir()`, `compute_cmc()`, `compute_eer()`, `compute_auc()`.
* **Why it's essential:** Standardizes evaluation so TAR@FAR and TPIR@FPIR are computed identically across all experiment runs.
* **Defense line:** *"It calculates the standardized biometric evaluation metrics (TAR, TPIR, Rank-N, EER, AUC)."*

### 22. `fedmps/eval/calibrate.py` — Scoring Layer Calibration
* **Purpose:** Implements post-processing vector calibration (mean centering, PCA whitening/rotation, and hubness reduction).
* **Key Components:** `fit_calibration()`, `apply_calibration()`, hubness mitigation.
* **Why it's essential:** Mitigates the "hubness" problem in high-dimensional embedding spaces, boosting retrieval accuracy.
* **Defense line:** *"It calibrates embedding geometry through centering, whitening, and hubness mitigation."*

### 23. `fedmps/eval/experiments.py` — Automated Benchmark Suite & Plot Generation
* **Purpose:** Runs the complete experimental battery (degradation, three-way bounds, DP sweep, template sweep, FedProx curves) and generates Figures 1–8.
* **Key Components:** `run_all()`, `exp_degradation()`, `exp_three_way()`, `exp_dp_sweep()`, `exp_templates()`, `make_figures()`.
* **Why it's essential:** Coordinates the scientific experiments and ensures all research claims are backed by automated scripts.
* **Defense line:** *"It orchestrates the full experimental benchmark suite and programmatically renders publication figures."*

---

## 📂 Subsystem 9: Execution & CLI Tooling

### 24. `fedmps/scripts/01_prepare_data.py` — Phase 1 Pipeline CLI
* **Purpose:** Ingests dataset, extracts clean/degraded ArcFace embeddings, applies Dirichlet partitioning, and writes `artifacts/`.
* **Defense line:** *"It runs the Phase 1 data pipeline, generating embeddings and edge node partitions."*

### 25. `fedmps/scripts/02_run_experiments.py` — Phase 3 Experiment Suite CLI
* **Purpose:** Command-line entry point to run full or quick experiment sweeps.
* **Defense line:** *"It executes the Phase 3 experiment battery across federated, centralised, and baseline models."*

### 26. `fedmps/scripts/03_run_video.py` — Phase 4 Live Video Surveillance CLI
* **Purpose:** Runs the video matching pipeline on a single node's MP4 video feed or live camera stream.
* **Defense line:** *"It executes live surveillance video analysis on an edge camera stream."*

### 27. `fedmps/scripts/make_demo_video.py` — Demo Video Synthesizer
* **Purpose:** Utility script that generates synthetic surveillance test videos with moving faces for rapid testing.
* **Defense line:** *"It generates controlled synthetic surveillance footage for offline node testing."*

### 28. `fedmps/scripts/make_sample_data.py` — Synthetic Data Generator
* **Purpose:** Generates lightweight synthetic embeddings so the entire codebase can be verified in 5 seconds without multi-GB downloads.
* **Defense line:** *"It creates minimal synthetic test vectors enabling instant offline verification of all algorithms."*

---

## 📂 Subsystem 10: Root Demo Drivers & Shell Scripts

### 29. `demo_trigger.py` — Interactive Demo Scanner
* **Purpose:** Command-line trigger that runs edge nodes against cases opened in the central dashboard.
* **Defense line:** *"It triggers edge camera matching against active law enforcement cases during live demos."*

### 30. `demo_reset.py` — Zero-Trace State Reset
* **Purpose:** Clears SQLite cases, templates, alerts, video annotations, and cropped snapshots for clean testing.
* **Defense line:** *"It resets the database and filesystem to a clean state before faculty demonstrations."*

### 31. `run_all_nodes.sh` — Batch Multi-Node Video Runner
* **Purpose:** Bash script that executes surveillance matching sequentially across all 5 edge camera nodes.
* **Defense line:** *"It runs automated sequential video processing across all 5 simulated camera sites."*

### 32. `generate_capstone_benchmarks.py` — Publication Visualizer & Table Generator
* **Purpose:** Reads `experiments_vgg_big.json` and produces the high-resolution 3-panel ROC/DET/CMC curves, score histograms, and markdown tables.
* **Defense line:** *"It generates our publication-grade benchmark figures and LaTeX/Markdown evaluation tables."*

### 33. `requirements.txt` — Package Dependency Specifications
* **Purpose:** Lists required Python libraries (`torch`, `insightface`, `onnxruntime`, `opencv-python`, `streamlit`, `scipy`, `matplotlib`).
* **Defense line:** *"It specifies verified package dependencies and execution environments."*

---

## 💡 Quick Defense Cheat Sheet for Common Faculty Doubts

| If Faculty Asks... | Point to this File | Exact Defense Line |
| :--- | :--- | :--- |
| *"Where is the actual federated learning model?"* | `fedmps/models/adapter.py` | *"Here: a 66k parameter residual MLP bottleneck trained on edge cameras."* |
| *"How do you prevent raw camera footage from leaking?"* | `fedmps/fl/node.py` | *"Local training only outputs weight deltas $\Delta w$; raw images never leave the node."* |
| *"How do you protect the missing person's reference photo?"* | `fedmps/privacy/template.py` | *"We convert embeddings into 512-bit BioHash binary codes masked with session XOR keys."* |
| *"What if camera angles are completely unbalanced?"* | `fedmps/fl/node.py` (L202) | *"We use FedProx with proximal regularization ($\frac{\mu}{2}\|w - w_{\text{global}}\|^2$) to stop client drift."* |
| *"How do you prevent false alarms on innocent crowds?"* | `fedmps/pipeline/match.py` | *"We require multi-frame tracklet voting ($3\times 3$ fusion) before an alert is raised."* |
| *"Who stops false alerts from dispatching police?"* | `fedmps/apps/node_app.py` | *"A local operator must verify the detection in the Node Review Queue first (Human-in-the-Loop)."* |
| *"Where are all the experimental numbers stored?"* | `results/experiments_vgg_big.json` | *"Here: 359 KB of raw empirical evaluation logs on 3,021 gallery identities."* |
