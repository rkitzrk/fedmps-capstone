# Third Review Guide — Part 1: Code Files Walkthrough

This guide details **which code files to open**, **what exact functions/lines to highlight**, and **what explanation to give to the faculty guide** during your Third Review code verification.

---

## 🧭 Overview: Code Architecture & Flow to Explain First

Before opening individual files, give your guide this 30-second high-level summary:
> *"Our architecture decouples heavy feature extraction from domain adaptation. The pre-trained ResNet-50 ArcFace backbone (43.6M parameters) remains **completely frozen** on edge nodes. Only a lightweight residual MLP adapter (66k parameters, 0.26 MB) is federated across cameras using **FedProx** to correct for CCTV blur, resolution loss, and lighting shifts. Watchlists are distributed not as photographs, but as **cancelable 512-bit binary templates**."*

```
[Raw CCTV Frame]
       │
       ▼
[InsightFace RetinaFace] ──(5-point landmark alignment)──> 112×112 Aligned Crop
       │
       ▼
[ArcFace Backbone (Frozen)] ──(w600k_r50.onnx)───────────> 512-d Latent Vector
       │
       ▼
[Residual Adapter (Federated)] ──(fedmps/models/adapter.py)─> Corrected 512-d Vector
       │
       ▼
[Cancelable Projection] ──(fedmps/privacy/template.py)────> 512-bit BioHash Template
       │
       ▼
[Tracklet Temporal Voting] ──(fedmps/pipeline/match.py)───> Verified Alert (Port 8502)
```

---

## 📂 Core Code Files to Present (In Recommended Order)

---

### 1. `fedmps/models/adapter.py` — The Federated Domain Adapter
**Why show this first:** This is the core machine learning contribution of the project. It proves you understand parameter-efficient fine-tuning and federated domain adaptation.

* **File Path:** [`fedmps/models/adapter.py`](file:///home/kumaresh/Desktop/Dev/fedmps-capstone/fedmps/models/adapter.py)
* **Key Class/Method to Highlight:**
  * `class Adapter(nn.Module)` (Lines 37–58)
  * `self._init_weights()` (Lines 59–68)
  * `def forward(self, x)` (Lines 70–82)
* **What to Show on Screen:**
  ```python
  self.net = nn.Sequential(
      nn.Linear(cfg.in_dim, cfg.hidden_dim),   # 512 -> 64 bottleneck
      nn.LayerNorm(cfg.hidden_dim),
      nn.GELU(),
      nn.Dropout(cfg.dropout),
      nn.Linear(cfg.hidden_dim, cfg.out_dim),  # 64 -> 512 expansion
  )
  ```
* **What to Explain to Faculty:**
  1. **Bottleneck Dimension Reduction:** Taking 512 dimensions down to 64 creates an information bottleneck that forces the adapter to learn only the low-rank systematic shift of CCTV degradation (blur, sensor noise, downsampling) rather than memorizing individual faces.
  2. **Residual Bypass (`x + delta`):** The output is `x + self.net(x)`. If the adapter outputs zero, the model naturally defaults back to the pristine ArcFace representation.
  3. **Near-Zero Weight Initialization:** Notice `_init_weights()` scales the second linear layer by `0.01`. At epoch 0, the adapter acts as an identity function and cannot corrupt pre-trained facial features.
  4. **L2 Renormalization:** Embeddings are re-normalized to unit norm because downstream matching uses cosine and Hamming metrics.

---

### 2. `fedmps/fl/node.py` — Edge Client Training & FedProx
**Why show this:** Proves that training happens locally at the edge, that raw data never leaves the camera site, and demonstrates mathematical handling of Non-IID client drift.

* **File Path:** [`fedmps/fl/node.py`](file:///home/kumaresh/Desktop/Dev/fedmps-capstone/fedmps/fl/node.py)
* **Key Function to Highlight:**
  * `def local_train(...)` (Lines 166–215)
  * Proximal regularization term (Lines 202–206)
  * Return value: `delta = sub_states(w_local, w_global)` (Lines 220–235)
* **What to Show on Screen:**
  ```python
  loss = compute_loss(adapted, data.clean[sel], data.labels[sel], tc)

  if fedprox_mu > 0.0 and global_params is not None:
      prox = sum(((p - g) ** 2).sum()
                 for p, g in zip(model.parameters(), global_params))
      loss = loss + 0.5 * fedprox_mu * prox

  opt.zero_grad(set_to_none=True)
  loss.backward()
  opt.step()
  ```
* **What to Explain to Faculty:**
  1. **Why FedProx instead of plain FedAvg:** Real CCTV cameras have vastly different numbers of subjects and camera angles (Non-IID Dirichlet distribution). Plain FedAvg causes local models to diverge into incompatible parameter subspaces ("client drift"). The proximal term $\frac{\mu}{2} \|w - w_{\text{global}}\|^2$ acts as an elastic leash tethering the node to the global model.
  2. **Transmission of Weight Deltas Only:** The node does not return its full weights. It computes $\Delta w = w_{\text{local}} - w_{\text{global}}$. Returning deltas is strictly required so Differential Privacy clipping and pairwise masking can be computed algebraically.

---

### 3. `fedmps/fl/server.py` — Secure Aggregation Server
**Why show this:** Shows the central coordinating entity and proves that the server never inspects individual updates or facial data.

* **File Path:** [`fedmps/fl/server.py`](file:///home/kumaresh/Desktop/Dev/fedmps-capstone/fedmps/fl/server.py)
* **Key Functions to Highlight:**
  * `def aggregate(...)` (Lines 62–95)
  * `def apply_privacy(...)` (Lines 47–60)
  * `class SecureAggregator` (imported from `privacy.mask`)
* **What to Show on Screen:**
  ```python
  # Weighted federated average across participating edge cameras
  weights = [n / total_samples for n in sample_counts]
  aggregated_delta = weighted_mean_states(client_deltas, weights)
  new_global_weights = add_states(current_global, aggregated_delta)
  ```
* **What to Explain to Faculty:**
  1. **Sample-Weighted Aggregation:** Nodes that witnessed more training samples contribute proportionally higher weight $p_k = \frac{n_k}{\sum n}$, preserving statistical consistency.
  2. **Zero Raw Data Ingestion:** The server only performs tensor arithmetic on 66k float32 weight deltas (payload size $\approx 0.26\text{ MB}$ per round).

---

### 4. `fedmps/privacy/template.py` — Cancelable Biometric Watchlist
**Why show this:** Directly addresses the faculty's privacy and compliance questions (GDPR, biometric data protection).

* **File Path:** [`fedmps/privacy/template.py`](file:///home/kumaresh/Desktop/Dev/fedmps-capstone/fedmps/privacy/template.py)
* **Key Class/Functions to Highlight:**
  * `class TemplateSelector` (Lines 50–65)
  * `def protect(embedding, key, selector)` (Lines 110–135)
  * `def similarity_matrix(...)` (Lines 150–175)
* **What to Show on Screen:**
  ```python
  # 512-d float -> threshold against mean -> binarise -> XOR with session key
  bits = binarise(embedding[selector.dims], selector.thresholds)
  cancelable_template = bits ^ key
  ```
* **What to Explain to Faculty:**
  1. **Irreversibility:** 512 floating-point values are binarized and XOR-masked with a pseudo-random key $K_{\text{node, case}}$. An attacker stealing the node's database cannot reconstruct the original facial photo.
  2. **Revocability (Cancelability):** If Camera Node 2 is physically stolen or compromised, the server issues a new seed key $K'$. The old template is instantly revoked and rendered useless, solving the fundamental flaw of biometrics (you cannot change your face, but you can reissue a key).
  3. **Distance Preservation:** Because Hamming distance satisfies $d(a \oplus k, b \oplus k) = d(a, b)$, edge cameras perform matching directly in the encrypted domain without ever decrypting the template.

---

### 5. `fedmps/pipeline/match.py` & `quality.py` — Real-Time CCTV Matching
**Why show this:** Demonstrates that this is not just an academic script, but an operational multi-camera surveillance system with tracking and quality gates.

* **File Path:** [`fedmps/pipeline/match.py`](file:///home/kumaresh/Desktop/Dev/fedmps-capstone/fedmps/pipeline/match.py) & [`fedmps/pipeline/quality.py`](file:///home/kumaresh/Desktop/Dev/fedmps-capstone/fedmps/pipeline/quality.py)
* **Key Classes/Methods to Highlight:**
  * `class QualityGate` (`quality.py`: Lines 25–65): Filters out severe blur (Laplacian variance), extreme yaw/pitch angles, and underexposed crops before inference.
  * `class Matcher.process_frame(...)` (`match.py`: Lines 160–240)
  * Tracklet voting logic (`match.py`: Lines 260–290)
* **What to Explain to Faculty:**
  1. **The Multi-Frame Voting Rule:** A single fleeting frame exceeding the similarity threshold often triggers false alarms in busy public spaces. Our system groups detections into spatio-temporal tracks and requires $k$ out of $n$ frames (default: $3 \times 3$ fusion) to match before raising an alert, crushing false positive rates by $7\times$.

---

### 6. `demo_trigger.py` & `demo_reset.py` — Demo Orchestration Scripts
**Why show this:** Shows professional testability and clean demonstration capabilities.

* **File Paths:**
  * [`demo_reset.py`](file:///home/kumaresh/Desktop/Dev/fedmps-capstone/demo_reset.py)
  * [`demo_trigger.py`](file:///home/kumaresh/Desktop/Dev/fedmps-capstone/demo_trigger.py)
* **What to Explain to Faculty:**
  1. Demonstrates the complete two-tier operational loop: Law Enforcement Enrolment on Port 8501 $\rightarrow$ Watchlist Key Distribution $\rightarrow$ Edge Node Video Processing $\rightarrow$ Edge Operator Human-in-the-Loop Verification on Port 8502 $\rightarrow$ Central Audit Logging.

---

## 💡 Quick Presentation Checklist for Part 1 (Code Files)

| Order | File | Lines | What to Emphasize |
| :---: | :--- | :--- | :--- |
| **1** | `fedmps/models/adapter.py` | 46–82 | Residual MLP bottleneck (512 $\rightarrow$ 64 $\rightarrow$ 512), near-zero initialization. |
| **2** | `fedmps/fl/node.py` | 166–210 | Local edge training, FedProx proximal loss, delta computation. |
| **3** | `fedmps/fl/server.py` | 47–95 | Zero-data weighted delta aggregation, secure aggregation hook. |
| **4** | `fedmps/privacy/template.py` | 50–140 | 512-bit BioHash cancelable template, distance-preserving XOR matching. |
| **5** | `fedmps/pipeline/match.py` | 160–290 | Quality assessment gate, 3x3 multi-frame tracklet voting. |
| **6** | `demo_trigger.py` | 60–125 | End-to-end edge inference trigger against distributed watchlist. |
