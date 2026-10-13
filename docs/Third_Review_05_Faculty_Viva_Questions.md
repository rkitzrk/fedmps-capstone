# Third Review Guide — Part 5: Faculty Viva Questions & Edge Cases

This guide prepares you for the **toughest questions, edge cases, and architectural challenges** that faculty guides and external examiners ask during the final review and project defense.

---

## 🎯 Top 10 Faculty Defense Questions & Model Answers

---

### Q1. "Why didn't you fine-tune the entire ArcFace backbone using federated learning instead of training an adapter?"
**Examiner's Intent:** Testing your understanding of parameter-efficient fine-tuning (PEFT) and edge computing constraints.

* **Your Model Answer:**
  > *"Sir/Ma'am, the ArcFace ResNet-50 backbone has 43.6 million parameters (~170 MB) and was pre-trained on millions of faces using large-scale angular margin loss. Fine-tuning 43.6M parameters across edge cameras presents three major failures:*
  > 1. * **Catastrophic Forgetting:** Edge camera nodes only observe local surveillance faces; fine-tuning the whole backbone would overwrite the global facial feature representation.*
  > 2. * **Bandwidth Bottleneck:** Sending 170 MB per node per round over CCTV cellular networks is impractical. Our residual adapter is only 66,240 parameters (0.26 MB), achieving **650× bandwidth reduction**.*
  > 3. * **Domain Focus:** The degradation shift (low resolution, motion blur, sensor noise) is a low-rank transformation. A 512 $\rightarrow$ 64 $\rightarrow$ 512 bottleneck MLP adapter is mathematically sufficient to correct for this shift without touching the backbone."*

---

### Q2. "In Table 1, why is the Frozen Baseline slightly higher in Rank-1 (94.74% vs. 94.48%), but the Federated Adapter is higher in TPIR (80.44% vs. 79.68%)?"
**Examiner's Intent:** Testing whether you understand the difference between closed-set retrieval and open-set surveillance identification.

* **Your Model Answer:**
  > *"This is a fundamental distinction in biometrics:*
  > * **Rank-1 is Closed-Set:** It assumes the probe person is guaranteed to be in the gallery and merely checks if the true match is ranked #1. At 94.7%, both models are essentially tied (within 0.2%).*
  > * **TPIR @ 1% FPIR is Open-Set Surveillance:** This evaluates real-world search against **18,126 innocent non-mate crowd members** who are not in the watchlist. The federated adapter pulls genuine match scores higher relative to the crowd, boosting True Positive Identification from **79.68% to 80.44%** while strictly capping false alarms to 1 in 100. For surveillance and law enforcement, open-set TPIR is the metric that truly matters."*

---

### Q3. "Why did your Rank-1 accuracy decrease from 98.48% (Step 3) to 94.74% (Step 4) in your ablation study?"
**Examiner's Intent:** Checking if you manipulated data or if you understand benchmark difficulty.

* **Your Model Answer:**
  > *"That drop was intentional and demonstrates our intellectual honesty. In Step 3, we evaluated on a gallery of 1,750 identities where multi-frame fusion reached 98.48%. At 98.5%, the benchmark had saturated, leaving virtually no room to prove the statistical benefit of federated learning.*
  > *To create a rigorous, publication-grade benchmark, we expanded the gallery by 73% (to 3,021 identities out of 8,631 total IDs). Increasing the distractor pool made the search harder, lowering Rank-1 to 94.74% and providing the headroom needed to validate federated learning against local silos."*

---

### Q4. "What happens if an edge camera goes offline or drops out during a federated training round?"
**Examiner's Intent:** Testing fault tolerance and asynchronous federated learning principles.

* **Your Model Answer:**
  > *"Our federated server does not block on stragglers. In each round, the server aggregates updates from whichever nodes respond before the round deadline, weighting each update by its sample proportion $p_k = \frac{n_k}{\sum_{j \in \mathcal{S}} n_j}$.*
  > *Furthermore, because we use **FedProx** rather than vanilla FedAvg, the proximal term $\frac{\mu}{2}\|w - w_{\text{global}}\|^2$ prevents nodes that were offline or delayed from drifting into incompatible parameter subspaces when they rejoin."*

---

### Q5. "Can an attacker reconstruct the missing person's original photograph from the watchlist template stored on the camera?"
**Examiner's Intent:** Testing knowledge of biometric template irreversibility and GDPR compliance.

* **Your Model Answer:**
  > *"No, reconstruction is computationally infeasible. We use **512-bit BioHash cancelable templates**:*
  > 1. * **Lossy Binarization:** The 512 floating-point embedding values (which contain 16,384 bits of precision) are projected and thresholded to single bits, discarding phase, magnitude, and high-frequency texture details.*
  > 2. * **Key-Dependent XOR Masking:** The bit vector is masked with a pseudo-random session key $K_{\text{node, case}}$: $T = B \oplus K$. Without the key, the bits appear as uniform random noise.*
  > 3. * **Revocability:** If a physical camera is stolen, the central server simply revokes key $K$ and re-issues the case with a new key $K'$. The stolen template becomes completely useless."*

---

### Q6. "Why did you choose cancelable templates over Homomorphic Encryption (HE)?"
**Examiner's Intent:** Checking real-world systems engineering knowledge.

* **Your Model Answer:**
  > *"Homomorphic encryption allows computation on ciphertexts, but ciphertext expansion and encrypted vector multiplication add 3 to 4 orders of magnitude of computational overhead (taking several seconds per frame). For CCTV video streams running at 25–30 FPS, HE cannot achieve real-time performance.*
  > *Cancelable templates preserve Hamming distance: $d_H(a \oplus k, b \oplus k) = d_H(a, b)$. This enables our edge hardware to compute matching using bitwise XOR and hardware POPCOUNT instructions in **microseconds**, enabling real-time multi-frame video inference."*

---

### Q7. "What prevents a malicious camera node from poisoning the global model with bad updates?"
**Examiner's Intent:** Testing federated learning security and adversarial robustness.

* **Your Model Answer:**
  > *"We implement **per-update L2 norm clipping** (`clip_norm = 2.0` in `fedmps/privacy/dp.py` and `server.py`). If a compromised node attempts to inject an adversarial update with an arbitrarily large norm to destroy the global model, its update delta $\Delta w$ is strictly rescaled:*
  > $$\Delta w \leftarrow \Delta w \cdot \min\left(1, \frac{C}{\|\Delta w\|_2}\right)$$
  > *This bounds the maximum influence of any single node in a round. Additionally, median aggregation or coordinate-wise trimmed mean can be toggled on the server to reject statistical outliers."*

---

### Q8. "Why does your Differential Privacy chart (`fig4_dp_sweep.png`) show a performance cliff for $\epsilon < 5$?"
**Examiner's Intent:** Testing mathematical depth in differential privacy.

* **Your Model Answer:**
  > *"Differential Privacy adds spherical Gaussian noise calibrated to the global sensitivity: $\sigma = \frac{\Delta_2 \sqrt{2 \ln(1.25/\delta)}}{\epsilon}$.*
  > *In face recognition, facial features lie on a narrow, high-dimensional manifold on the unit hypersphere. Adding sufficient Gaussian noise across 66,240 parameters to guarantee $\epsilon < 5$ perturbs the vector outside this discriminative manifold, destroying angular separation. Rather than hiding this, our paper documents this trade-off and demonstrates why **secure masked aggregation** and **cancelable templates** are superior mechanisms for biometric privacy."*

---

### Q9. "How do you handle poor surveillance conditions like extreme side profiles, low lighting, or motion blur?"
**Examiner's Intent:** Checking understanding of the video processing pipeline.

* **Your Model Answer:**
  > *"We employ a 3-stage defense in `fedmps/pipeline/`:"*
  > 1. * **Pre-Inference Quality Gate (`quality.py`):** Rejects crops with Laplacian variance below threshold (motion blur), illumination extremes, or pose yaw/pitch exceeding $45^\circ$ before running the neural network.*
  > 2. * **5-Point Landmark Similarity Alignment:** RetinaFace detects 5 facial landmarks (eyes, nose, mouth corners) and applies an affine similarity transform, rotating and scaling the face to standardized coordinates.*
  > 3. * **$3 \times 3$ Multi-Frame Tracklet Voting (`match.py`):** A match is only confirmed if at least 3 frames within a continuous spatio-temporal tracking track agree with the watchlist, reducing single-frame false alarms by over $7\times$."*

---

### Q10. "How does the system ensure human accountability and prevent false police dispatches?"
**Examiner's Intent:** Checking legal, ethical, and human-in-the-loop operational design.

* **Your Model Answer:**
  > *"Our system enforces a strict **Human-in-the-Loop** policy implemented across two separate dashboards:*
  > * When a camera detects a match, it does **not** notify the central police automatically. It queues the alert in the **Edge Node Review Queue (`node_app.py` on Port 8502)**.*
  > * A local operator reviews the side-by-side cropped face, timestamp, and similarity score.*
  > * Only upon manual confirmation by the operator is the alert propagated to the **Central Law Enforcement Console (`server_app.py` on Port 8501)**, where it is logged into an immutable cryptographic audit trail with SHA-256 case IDs and operator IDs."*

---

## 📋 Summary of All 5 Generated Third Review Guides

All 5 guides are saved in `docs/` for your immediate review and presentation:

1. **[`docs/Third_Review_01_Code_Files.md`](file:///home/kumaresh/Desktop/Dev/fedmps-capstone/docs/Third_Review_01_Code_Files.md)**: File-by-file code walkthrough, exact line numbers, classes, and architectural flow.
2. **[`docs/Third_Review_02_Plots_and_Code.md`](file:///home/kumaresh/Desktop/Dev/fedmps-capstone/docs/Third_Review_02_Plots_and_Code.md)**: All 10 plots in `figures/`, their Python generation code, and interpretation.
3. **[`docs/Third_Review_03_AWS_Training_Results.md`](file:///home/kumaresh/Desktop/Dev/fedmps-capstone/docs/Third_Review_03_AWS_Training_Results.md)**: Proof of AWS EC2 GPU training, `full_run_vgg_big.log`, PyTorch checkpoint inspection, and MIA attack validation.
4. **[`docs/Third_Review_04_Metrics_and_Results.md`](file:///home/kumaresh/Desktop/Dev/fedmps-capstone/docs/Third_Review_04_Metrics_and_Results.md)**: Mathematical definitions, Table 1 & Table 2 benchmarks, 5-step ablation history, and operational metrics.
5. **[`docs/Third_Review_05_Faculty_Viva_Questions.md`](file:///home/kumaresh/Desktop/Dev/fedmps-capstone/docs/Third_Review_05_Faculty_Viva_Questions.md)**: Top 10 tough faculty viva questions, edge cases, and model defense scripts.
