# FedMPS — Federated Missing-Person Search

Privacy-preserving federated learning for missing person identification.

A public face-recognition backbone is **frozen**. Only a small domain adapter
is federated across nodes, secured with masked aggregation, with the watchlist
distributed as cancelable templates rather than photographs. Differential
privacy is implemented and measured; the [Results](#results) explain why its
curve does not yield a usable operating point at this model size.

---

## Build status

The codebase is written in four sequential phases. Testing happens after
Phase 4, so the phases are joined by written contracts (`fedmps/contracts.py`)
rather than by passing tests.

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Foundation & data | **Complete** |
| 2 | Learning core (adapter, FedAvg, metrics) | **Complete** |
| 3 | Privacy & experiments | **Complete** |
| 4 | Video pipeline & dashboards | **Complete** |

All four phases are written **and have been run end to end** on VGGFace2 with
a real ArcFace backbone. See [Results](#results) for the numbers and
`results/experiments_vgg_big.json` for the record.

---

## 🎓 Faculty Guide Review & Demo Quickstart

For faculty evaluation and project defense, consult the complete documentation in [`docs/README.md`](docs/README.md):
- **First Review Deliverables**: Architecture, threat model, mathematical formulations, and 12-week roadmap ([`docs/First_Review_Documentation.pdf`](docs/First_Review_Documentation.pdf)).
- **Second Review Deliverables**: Full VGGFace2 evaluation, academic benchmark tables ([`results/benchmark_comparison_table.md`](results/benchmark_comparison_table.md)), publication figures ([`figures/`](figures/)), and interactive surveillance demo ([`docs/Second_Review_Documentation.pdf`](docs/Second_Review_Documentation.pdf)).

### Live Demo in 3 Commands:
1. **Reset Demo State**:
   ```bash
   python demo_reset.py
   ```
2. **Start Interactive Dashboards** (Two terminals):
   ```bash
   streamlit run fedmps/apps/node_app.py --server.port 8502    # Edge Node Review Queue (http://localhost:8502)
   streamlit run fedmps/apps/server_app.py --server.port 8501  # Central Law Enforcement Console (http://localhost:8501)
   ```
3. **Trigger Video Scan**:
   - Open `http://localhost:8501`, go to **"Open a case"**, upload a reference photo (e.g. `media/person-1.jpg`), and click **"Create case and distribute"**.
   - Run video matching on edge node 0:
     ```bash
     python demo_trigger.py --node 0
     ```
   - Review detection alert and cryptographic audit records at `http://localhost:8502` and `http://localhost:8501`.

---

## Test plan

Run bottom-up so a failure localises to one file instead of arriving as one
traceback from the top.

```bash
# T1 — every module's self-check. No dataset, no model, no network.
python -m fedmps.config
python -m fedmps.contracts
python -m fedmps.data.degrade
python -m fedmps.data.loader
python -m fedmps.data.partition
python -m fedmps.models.backbone
python -m fedmps.models.adapter
python -m fedmps.eval.metrics
python -m fedmps.fl.node
python -m fedmps.fl.server
python -m fedmps.fl.loop
python -m fedmps.privacy.dp
python -m fedmps.privacy.mask
python -m fedmps.privacy.template
python -m fedmps.eval.experiments
python -m fedmps.store
python -m fedmps.pipeline.detect
python -m fedmps.pipeline.quality
python -m fedmps.pipeline.match
```

```bash
# T2 — the data layer on the synthetic backbone. Seconds, no downloads.
python fedmps/scripts/01_prepare_data.py --backbone dummy --max-identities 60
```

```bash
# T3 — the whole learning and privacy stack, still with no real model.
python fedmps/scripts/02_run_experiments.py --quick
```

```bash
# T4 — the real backbone. The ONLY step that can fail on environment
#      rather than logic. Downloads ~200 MB of LFW on first run.
python fedmps/scripts/01_prepare_data.py
```

```bash
# T5 — real numbers and figures. See Results for the recorded run.
python fedmps/scripts/02_run_experiments.py --all     --artifacts-suffix _vgg_big --out experiments_vgg_big.json
```

```bash
# T6 — end to end on video, enrolling a face from a still first.
python fedmps/scripts/03_run_video.py --video media/corridor.mp4 --enrol media/person.jpg
```

```bash
# T7 — the two dashboards.
streamlit run fedmps/apps/server_app.py
streamlit run fedmps/apps/node_app.py
```

T1–T3 need no dataset and no face model, so most of the codebase is
verifiable within minutes. T4 is deliberately isolated: failing it invalidates
nothing built before it.

---

## Install

```bash
pip install -r requirements.txt
```

A face backbone is **optional**. See the notes at the bottom of
`requirements.txt`: if neither `insightface` nor `facenet-pytorch` installs,
the code falls back to a synthetic embedder and every phase still executes.

---

## Run order

```bash
# Phase 1 — Data preparation & feature extraction (writes artifacts/)
python fedmps/scripts/01_prepare_data.py

# Phase 3 — Full experiment suite & ablation studies (writes results/ and figures/)
python fedmps/scripts/02_run_experiments.py --all

# Phase 4 — Multi-camera edge video pipeline (processes media/ and raises alerts)
python fedmps/scripts/03_run_video.py --video media/node0.mp4 --node 0
```


Useful flags on Phase 1:

```bash
# force the synthetic backbone — no download, no model, runs in seconds
python fedmps/scripts/01_prepare_data.py --backbone dummy --max-identities 60

# use your own corridor footage, arranged as person_name/frame.jpg
python fedmps/scripts/01_prepare_data.py --source folder --folder ./media/corridor
```

---

## What Phase 1 produces

Written to `artifacts/`:

| File | Shape / type | Meaning |
|------|--------------|---------|
| `embeddings_clean.npy` | float32 `[N, 512]` | frozen-backbone embedding of the original image |
| `embeddings_deg.npy` | float32 `[N, 512]` | same image after the CCTV degradation model |
| `labels.npy` | int32 `[N]` | identity id per image |
| `partition.json` | dict | node split, held-out identities, per-node camera profiles |
| `identity_names.json` | list | identity id → name |
| `samples.npz` | arrays | 32 clean/degraded pairs, kept for report figures |
| `meta.json` | dict | backbone used, health stats, full config snapshot |

Every row is L2-normalised, and every array is aligned on `N`. Phases 2 and 3
load all of it in one call:

```python
from fedmps.config import default_config
from fedmps.contracts import Artefacts

cfg = default_config()
art = Artefacts.load(cfg)   # validates on the way in
print(art.describe())
```

---

## Layout

```
fedmps/
  config.py          every tunable number, in one place
  contracts.py       cross-phase shapes, assertions, load/save helpers
  data/
    loader.py        LFW, CelebA, or a folder of identities
    degrade.py       blur → downscale → gamma → noise → JPEG → upscale
    partition.py     disjoint identities + per-node camera profiles
  models/
    backbone.py      frozen embedder + ArcFace landmark alignment
    adapter.py       the federated module + state-dict arithmetic
  fl/
    node.py          local training, contrastive losses, the update delta
    server.py        FedAvg + the two Phase 3 privacy hooks
    loop.py          local_only vs federated vs centralised
  privacy/
    dp.py            clip + Gaussian noise, Rényi DP accounting
    mask.py          Diffie-Hellman + pairwise cancelling masks
    template.py      cancelable watchlist templates (see the cost table)
  eval/
    metrics.py       open-set TPIR@FPIR, rank-1, verification
    calibrate.py     the scoring layer: centring, rotation, hubness
    experiments.py   eight experiments, eight figures
  pipeline/
    detect.py        SCRFD or Haar, ByteTrack or IoU
    quality.py       size / blur / exposure / pose gate
    match.py         keyed matching, track voting, video pipeline
  apps/
    server_app.py    case console — confirmed sightings only
    node_app.py      review queue — the human gate
  store.py           cases, templates, alerts, audit (SQLite)
  scripts/
    01_prepare_data.py
    02_run_experiments.py
    03_run_video.py
```

### Running the comparison (after Phase 1 has been run)

```python
from fedmps.config import default_config
from fedmps.contracts import Artefacts
from fedmps.fl.loop import run_three_way

cfg = default_config()
art = Artefacts.load(cfg)
out = run_three_way(art, cfg, save_to="three_way.json")
print(out["summary"]["rank1"])
```

`run_three_way` prints an explicit warning if federated fails to beat
local-only, if it implausibly exceeds centralised, or if the adapter fails to
beat the frozen backbone — the three red flags from the build plan, checked
automatically rather than left for you to notice.

---

## Results

Current benchmark: **VGGFace2**, 8,631 identities, 172,620 images, ArcFace-aligned,
evaluated on a held-out gallery of **3,021 identities** with 15,105 mate probes
and 18,126 non-mates. Frozen `buffalo_l` / `w600k_r50` throughout.

| | base (13 Sep) | now |
|---|---|---|
| rank-1 | 0.4807 | **0.9474** |
| TPIR @ FPIR = 1e-2 | 0.2091 | **0.7968** |
| TPIR @ FPIR = 1e-3 | 0.0770 | **0.6725** |
| TPIR @ FPIR = 1e-4 | 0.0109 | **0.5276** |
| clean-probe ceiling | 0.9214 | 0.9962 |
| gallery | 1,750 | **3,021** |

TPIR@FPIR is the number to lead with. Rank-1 asks only whether the right person
is top of the list; TPIR@FPIR asks how many people on the watchlist are both
flagged *and* correctly named while false alarms are held to one in a thousand.
It moved roughly sevenfold, on a gallery 73% larger, because the original system
was flagging almost nobody at any usable false-alarm rate.

Three separate changes produced that, and they are worth keeping separate
because two of them make the benchmark *harder*, not easier.

| # | change | gallery | rank-1 |
|---|---|---|---|
| 0 | base: CelebA, unaligned, 1 photo x 1 frame, plain cosine | 1,750 | 0.4807 |
| 1 | + scoring layer (centring, rotation, hubness) | 1,750 | 0.5029 |
| 2 | + operating point 3x3, matching what `pipeline.match` already does | 1,750 | 0.9186 |
| 3 | + VGGFace2 and ArcFace alignment | 1,750 | 0.9848 |
| 4 | + harder gallery, 8,631 identities — *costs* accuracy, deliberately | 3,021 | **0.9474** |

Step 4 lowered the number on purpose. At 1,750 identities the 3x3 cell had
saturated at 0.985 and the federated arm had no headroom left to demonstrate
anything; a 73% larger gallery restored about five points of room.

---

### 1. Alignment — the defect that cost the most

`ArcFaceONNX.get_feat` resizes its input to 112x112 and embeds it, which reads
like it handles framing. It does not. ArcFace is trained exclusively on crops
produced by a similarity transform that places five landmarks on fixed pixel
coordinates. Anything else arrives at the wrong scale, offset, and roll angle,
and the embedding degrades accordingly — silently, with no error.

Measured on identical data — same 5,000 identities, same seed, same degradation,
same scoring layer, one variable changed (`results/alignment_ab.json`):

| enrol x frames | unaligned | aligned | gain |
|---|---|---|---|
| 1 x 1 | 0.0444 | **0.7003** | +0.656 |
| 1 x 3 | 0.0999 | **0.9199** | +0.820 |
| 3 x 1 | 0.0755 | **0.8142** | +0.739 |
| 3 x 3 | 0.2064 | **0.9848** | +0.778 |
| 5 x 5 | 0.3804 | **0.9979** | +0.618 |

| | same-identity | cross-identity | separation |
|---|---|---|---|
| unaligned | +0.2920 | +0.1026 | +0.189 |
| aligned | +0.5138 | +0.0038 | **+0.510** |

**Do not quote "+0.78 rank-1" as what this bug cost the original pipeline.** The
old CelebA run scored 0.4807 unaligned, not 0.044, because `img_align_celeba`
ships pre-aligned and was masking the defect. VGGFace2 uses raw, loosely-framed
crops, and there the same missing warp is catastrophic. The table above measures
what alignment is worth *on realistically framed data*, which is the honest
claim.

One diagnostic worth keeping: unaligned same-identity cosine is 0.2920 clean
against 0.2956 degraded. Degradation made no difference, because the embeddings
were already noise. If those two numbers ever converge again, the front end is
broken.

Guards, both of which fired during development: `calibrate.fit` refuses to build
the rotation below `MIN_ROWS_PER_DIM` (10) training rows per dimension, and
`fit_hubness` refuses the hubness term when the cohort is smaller than
`MIN_COHORT_PER_K` (5) times its own `k`. On a 180-row set the underdetermined
rotation cost 17 rank-1 points; a 36-member cohort cost 3. Prep also aborts if
the detector misses more than `backbone.max_align_miss_rate` (5%) of faces —
past that the embeddings are a mixture of aligned faces and plain resizes. The
real run missed 14 of 172,620 (0.008%).

### 2. Operating point — what the single number was hiding

`eval.n_enrol` and `eval.n_probe_frames` default to 3 and 3.
`experiments.operating_points` prints the whole surface
(`results/operating_points_big.json`, `figures/fig8_operating_points.png`):

| enrol x frames | cosine | + centring | + rotation | + hubness |
|---|---|---|---|---|
| 1 x 1 *(the floor)* | 0.5240 | 0.5301 | 0.5344 | **0.5419** |
| 1 x 3 | 0.8238 | 0.8332 | 0.8393 | **0.8511** |
| 2 x 3 | 0.9016 | 0.9075 | 0.9115 | **0.9214** |
| **3 x 3** *(the default)* | 0.9294 | 0.9355 | 0.9391 | **0.9474** |
| 3 x 5 | 0.9775 | 0.9798 | 0.9814 | **0.9863** |
| 5 x 5 | 0.9861 | 0.9882 | 0.9895 | **0.9918** |

The 1x1 cell stays on the record because it is the honest floor and someone will
ask. But it is not what this system does: `pipeline.match` already votes over a
track of `match.track_vote_n` frames before raising anything, and a real
missing-person case file arrives with several photographs. Degradation costs
0.394 at 1x1 and 0.067 at 3x3 — track fusion recovers most of it, which is the
point of doing it.

The scoring layer (`eval/calibrate.py`) adds +1.4 to +2.2 points at every cell,
fitted on node-training identities only.

### 3. Privacy

**Secure aggregation is free.** Masked and unmasked runs are bit-identical:

```
rank1 plain=0.9376  masked=0.9376  difference=+0.00000   overhead 1.008x
```

The Diffie-Hellman pairwise masks cancel exactly on summation, so the server
learns the aggregate and never an individual node's update, at zero accuracy
cost. This is the strongest claim in the project.

**Membership inference finds no leakage.**

| level | AUC | what it measures |
|---|---|---|
| sample | **0.4963** | was this image in training? (chance = 0.5) |
| identity | 0.5456 | was this person enrolled? |

Sample-level is indistinguishable from a coin flip; [4] reported 0.507-0.532.
The identity-level number measures enrolment rather than training-set
membership and is not comparable.

**Differential privacy has no usable operating point at this scale.**

| sigma | epsilon | rank-1 |
|---|---|---|
| 0.0 | inf | 0.9391 |
| 0.3 | 251.9 | 0.8751 |
| 0.6 | 83.6 | 0.6254 |
| 1.0 | 40.0 | 0.2477 |
| 2.0 | 15.9 | 0.0216 |

Every setting with a meaningful epsilon (below 10) is unusable, and the only
tolerable accuracy sits at epsilon = 252, which guarantees nothing. Given that
MIA already shows no measurable leakage, DP here is defending against a threat
the measurements do not find. Report the curve; do not claim an operating point
on it.

**Cancelable templates cost more than they are worth at the configured width.**

| bits | rank-1 | cost vs float |
|---|---|---|
| float (unprotected) | 0.9355 | — |
| 256 | 0.6483 | -0.287 |
| 128 | 0.3593 | -0.576 |
| 96 *(current default)* | 0.2502 | **-0.685** |
| 32 | 0.0383 | -0.897 |

At `privacy.template_bits = 96` protection costs 69 rank-1 points. This got
*worse* after the alignment fix, and that is the expected direction: binarising
to 96 bits destroys a fixed quantity of information, so the better the
embeddings, the more there is to lose. The old CelebA run appeared to show
templates *beating* float only because the float baseline was broken. Either
report at 256 bits and state the cost, or run float and present templates as
measured future work.

### 4. The federated adapter does not earn its place

| arm | rank-1 | vs frozen backbone |
|---|---|---|
| no adapter | **0.9474** | — |
| centralised | 0.9452 | -0.0022 |
| federated | 0.9448 | -0.0026 |
| local only (mean) | 0.9285 | -0.0189 |

Two findings pointing opposite ways, and both belong in the write-up.

**The federated methodology works.** Federated beats local-only by +1.63 points,
and sits within 0.0004 of centralised — so pooling everyone's biometric data
into one place buys essentially nothing over keeping it distributed. The price
of privacy is 0.04%.

**The adapter itself contributes nothing.** Every trained arm sits below the
frozen backbone. `run_three_way` says so rather than hiding it:

```
[three_way] The adapter did not improve on the frozen backbone alone.
```

This was already true in the original CelebA run — federated 0.4813 against a
0.4807 baseline, a gain of +0.0006 — but nothing flagged it, because the old
per-node `val_idx` shared its identities with `train_idx` and so measured
memorisation rather than generalisation.

Two changes address that. `fl.val_identity_frac` (default 0.2) carves identities
out of every node's training data entirely and scores each round on them,
gallery-and-probe; they are disjoint from training *and* from test, so selecting
on them is not leakage. `fl.early_stop` (default true) then keeps the best round
by that score and reports `early_stop.beat_frozen_backbone`.

Be careful how much you claim for early stopping here. On this run it selected
round 26 (val 0.9529) when round 1 scored better on test (0.9478 against
0.9448) — the validation set cannot resolve differences this small. That is not
a fault in the mechanism; it is evidence that the adapter's effect is below the
measurement floor. Where it does earn its keep is under DP, where it correctly
keeps round 1 instead of a model wrecked by thirty rounds of accumulated noise.

Set `fl.early_stop = False` and `fl.val_identity_frac = 0.0` to reproduce the
old behaviour.

**Why no embedding-space map helps.** Measured against held-out identities,
every unconstrained map loses ground, monotonically with training:

| | rank-1 |
|---|---|
| frozen backbone | 0.4807 |
| in-batch InfoNCE adapter, best early stop | 0.4852 |
| in-batch InfoNCE adapter, 40 epochs | 0.4337 |
| centroid InfoNCE adapter, 40 epochs | 0.3720 |
| ridge least-squares map degraded -> clean | 0.4572 |
| PCA whitening (alpha = 0.5) | 0.4027 |
| orthogonal Procrustes, half-blended | **0.4986** |

Raising the training identity count from 150 to 1,200 moves held-out rank-1 by
under half a point, so this is not a data-volume problem. A 30-pixel face has
lost information no 512-dimensional map can put back. Only rigid corrections — a
bias and a rotation, neither of which can distort the angular geometry ArcFace
depends on — transfer to people the adapter never saw. That is what
`eval/calibrate.py` implements, and why it is a scoring layer rather than a
learned module.

### Not yet wired: the live matcher

`eval/calibrate.py` is an evaluation-layer change. `pipeline/match.py` still
scores raw cosine against `match.match_threshold`, so the live video path does
not get the scoring layer's points. Wiring it there means re-deriving both
thresholds from the calibrated DET curve — the scores move onto a different
scale and `match_threshold = 0.24` does not carry over. Worth doing; do not do
it without re-deriving the thresholds.

### Reproducing

```bash
# Phase 1 — align, degrade, embed. ~38 min on a T4, peak RSS 19 GB.
python fedmps/scripts/01_prepare_data.py \
    --source folder --folder /path/to/vggface2 \
    --backbone insightface --device cuda \
    --max-identities 8631 --max-images-per-identity 20 \
    --out-suffix _vgg_big

# Phase 3 — every experiment and figure. ~25 min.
python fedmps/scripts/02_run_experiments.py --all \
    --artifacts-suffix _vgg_big --out experiments_vgg_big.json
```

`--out-suffix` and `--artifacts-suffix` keep runs from overwriting each other.
Add `--no-align` to reproduce the pre-fix numbers for the A/B.

Source images are expected as `identity_name/image.jpg`; the VGGFace2 tarball
already has that layout.

---

## Two design decisions worth knowing before reading the code

**The backbone is never trained.** ArcFace needs millions of images and a
classifier over the full identity set; a node has neither. Freezing it and
federating a 0.4 M-parameter adapter cuts the per-round payload from ~166 MB
to ~2 MB, which is what makes masked aggregation and differential privacy
affordable at all. This is the project's central claim.

**The synthetic backbone is a real component, not a stub.** It gives every
identity a fixed centroid, scatters blurrier inputs further from it, *and*
pushes similarly-degraded images in a common learnable direction. So a full
run on the dummy backbone should still show degradation hurting accuracy and
the adapter recovering some of it. The numbers mean nothing; the directions
mean the plumbing works.

---

## Smoke checks

Every module has a self-check that runs on synthetic input, needing no
dataset, no model and no network:

```bash
python -m fedmps.config
python -m fedmps.contracts
python -m fedmps.data.degrade
python -m fedmps.data.loader
python -m fedmps.data.partition
python -m fedmps.models.backbone
```

These are step **T1** of the test plan and are meant to be run bottom-up after
Phase 4, so that a failure localises to one file instead of arriving as one
traceback from the top.
