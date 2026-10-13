"""Central configuration for FedMPS.

Every tunable number in the project lives here. Nothing downstream should
hard-code a hyperparameter, a path, or a random seed.

The config is a nest of dataclasses so that it reads well in the report and
so that `asdict()` gives a clean JSON blob to stamp into every results file.

Frozen in Phase 1. Later phases add fields; they do not rename or remove them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Paths:
    """Where everything lives. All relative to the project root."""

    root: Path = PROJECT_ROOT
    artifacts: Path = PROJECT_ROOT / "artifacts"      # generated .npy / .json
    figures: Path = PROJECT_ROOT / "figures"          # report charts
    results: Path = PROJECT_ROOT / "results"          # experiment JSON
    checkpoints: Path = PROJECT_ROOT / "checkpoints"  # saved adapters
    cache: Path = PROJECT_ROOT / "cache"              # dataset downloads
    media: Path = PROJECT_ROOT / "media"              # demo video in / out

    def ensure(self) -> None:
        """Create every output directory. Safe to call repeatedly."""
        for p in (self.artifacts, self.figures, self.results,
                  self.checkpoints, self.cache, self.media):
            p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Which images we train on and how they are canonicalised.

    `source` options:
      "lfw"    - fetched automatically via scikit-learn. The default.
      "folder" - a local directory of `identity_name/image.jpg`. Use this for
                 CelebA (after you arrange it that way) or for your own
                 corridor footage frames.
    """

    source: str = "lfw"

    # -- lfw --------------------------------------------------------------
    # Identities with fewer than this many images are dropped. Raising it
    # gives fewer, better-populated identities and uses less memory.
    lfw_min_faces_per_person: int = 5
    # sklearn returns float32 at fetch time, so the peak memory during load is
    # roughly (n_images x 160 x 160 x 3 x 4) bytes. At the default settings
    # that is a little under 1 GB. Lower `max_identities` if that is too much.
    lfw_resize: float = 0.64  # 250px originals -> 160px working size

    # -- folder -----------------------------------------------------------
    folder_path: Optional[Path] = None
    folder_min_images_per_person: int = 3

    # -- celeba -----------------------------------------------------------
    # Needs both the image archive and identity_CelebA.txt. The archive alone
    # can still supply non-mate probes (no labels required), but training
    # needs identities to form same-person pairs.
    celeba_zip: Optional[Path] = None
    celeba_identity_file: Optional[Path] = None
    celeba_min_images_per_person: int = 5

    # -- common -----------------------------------------------------------
    # Cap on identities kept, applied after the min-images filter. None = all.
    max_identities: Optional[int] = 400
    # Cap on images kept per identity. LFW is severely skewed - one person has
    # 530 images and the median is 31 - and without a cap a handful of
    # identities dominate every batch, every node and every gallery. None =
    # keep everything.
    max_images_per_identity: Optional[int] = 40
    # Canonical working resolution. Both supported backbones want <= this, so
    # we store one size and let each backbone resize down as it needs.
    image_size: int = 160
    # Canonical channel order for every image array in the project.
    channel_order: str = "RGB"


# --------------------------------------------------------------------------
# Degradation ("what a CCTV camera does to a face")
# --------------------------------------------------------------------------

@dataclass
class DegradeRanges:
    """Sampling ranges for one node's camera personality.

    Each node draws a fixed point from these ranges once, and then every image
    at that node is degraded with a small jitter around it. That is what makes
    the federated split genuinely non-IID: node 0's camera is consistently
    different from node 3's, rather than every image being random.
    """

    # Effective face size in pixels before being upscaled back to image_size.
    # Smaller = further from the camera = harder.
    face_px: Tuple[int, int] = (20, 48)
    # Motion blur kernel length in pixels; 1 means no blur.
    blur_kernel: Tuple[int, int] = (1, 9)
    # Sensor noise standard deviation, in 0-255 units.
    noise_sigma: Tuple[float, float] = (0.0, 12.0)
    # Gamma < 1 brightens, > 1 darkens.
    gamma: Tuple[float, float] = (0.7, 1.6)
    # JPEG quality; lower is more compressed.
    jpeg_quality: Tuple[int, int] = (25, 70)


@dataclass
class DegradeConfig:
    ranges: DegradeRanges = field(default_factory=DegradeRanges)
    # Per-image jitter around the node's fixed point, as a fraction of the
    # range width. 0 = every image at a node is degraded identically.
    per_image_jitter: float = 0.25
    # Interpolation used when downscaling then upscaling back.
    downscale_interp: str = "area"
    upscale_interp: str = "linear"


# --------------------------------------------------------------------------
# Federated partition
# --------------------------------------------------------------------------

@dataclass
class PartitionConfig:
    """How the dataset is divided, stated as percentages.

    The split is specified the way it is reviewed:

        70%  central server   gallery + held-out evaluation
        30%  across the nodes
             20%  node training
             10%  node validation

    What "central server" means here is worth being precise about, because in
    federated learning the server holds no training data at all - that is the
    entire premise. The server's 70% is its *gallery and evaluation pool*: the
    reference photographs that arrive with a case report, and the held-out
    identities used to measure the system. Nobody trains on it. The nodes'
    30% is operational camera data, and it never leaves the node it sits at.

    Identities are split, not images. Every image of one person lands on one
    side of the line, so a face the system is asked to find at evaluation time
    was never seen during training in any form. Splitting images instead would
    leak the identity across the boundary and inflate every number in the
    report.
    """

    n_nodes: int = 5

    # The three percentages. They must sum to 100.
    server_pct: float = 70.0        # gallery + evaluation, never trained on
    node_train_pct: float = 20.0    # local training, stays at the node
    node_val_pct: float = 10.0      # local validation, stays at the node

    # Within a node, fraction of its images used for local validation.
    # Derived from the percentages above; see `resolve()`.
    val_frac: float = 0.2
    # If True, nodes get unequal amounts of data (more realistic, and it is
    # what makes FedProx worth comparing against FedAvg in Phase 2).
    unbalanced: bool = True
    # Ratio between the largest and smallest node when unbalanced.
    unbalance_ratio: float = 3.0

    # Derived, kept so the rest of the code needs no changes.
    test_identity_frac: float = 0.70

    def resolve(self) -> "PartitionConfig":
        """Turn the three percentages into the fractions the code uses."""
        total = self.server_pct + self.node_train_pct + self.node_val_pct
        if abs(total - 100.0) > 1e-6:
            raise ValueError(
                f"partition percentages must sum to 100, got {total:g} "
                f"(server {self.server_pct:g} + train {self.node_train_pct:g} "
                f"+ val {self.node_val_pct:g})")
        node_pct = self.node_train_pct + self.node_val_pct
        if node_pct <= 0:
            raise ValueError("nodes must receive a non-zero share")
        self.test_identity_frac = self.server_pct / 100.0
        self.val_frac = self.node_val_pct / node_pct
        return self

    def summary(self) -> str:
        node_pct = self.node_train_pct + self.node_val_pct
        lines = [
            f"{self.server_pct:g}% central server (gallery + evaluation)",
            f"{node_pct:g}% across {self.n_nodes} nodes",
            f"   {self.node_train_pct:g}% node training",
            f"   {self.node_val_pct:g}% node validation",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

@dataclass
class BackboneConfig:
    """The frozen face embedder.

    `preference` is tried in order and the first one that loads wins. The
    "dummy" entry always loads, so this never fails outright - it degrades to a
    synthetic embedder that still has real identity structure, which keeps the
    whole downstream stack testable without any model install.
    """

    preference: List[str] = field(
        default_factory=lambda: ["insightface", "facenet", "dummy"]
    )
    insightface_model: str = "buffalo_l"
    insightface_onnx: str = "w600k_r50.onnx"
    # Detector used to align faces onto ArcFace's landmark template before
    # embedding. Without this the recognition model is fed crops framed the
    # way the source dataset happened to crop them, which is not what it was
    # trained on: it cost a mean same-identity cosine of 0.51 against the
    # ~0.70 a correctly aligned w600k_r50 returns, and every downstream
    # number inherited the loss. See InsightFaceBackbone._align.
    insightface_det: str = "det_10g.onnx"
    align: bool = True
    det_thresh: float = 0.30
    # Detector input resolution. SCRFD's usual 640 is sized for full frames;
    # these are already face crops at `data.image_size`, so 320 costs nothing
    # in recall and runs about four times faster across 100k images.
    det_size: int = 320
    # Align once on the clean image and degrade the aligned crop, so both
    # branches share identical framing. False aligns each branch separately,
    # which folds detector failure on tiny faces into the degradation gap.
    # See models.backbone.align_faces.
    align_once: bool = True
    # Refuse to continue past this miss rate: beyond it the embeddings are a
    # mixture of aligned faces and plain resizes, and no metric over them
    # means anything.
    max_align_miss_rate: float = 0.05
    facenet_pretrained: str = "vggface2"
    device: str = "cpu"          # "cpu" or "cuda"
    batch_size: int = 64
    # Dummy backbone only: how tightly one identity's embeddings cluster.
    dummy_identity_noise: float = 0.35
    # Dummy backbone only: extra spread applied to blurred / degraded inputs,
    # so that the synthetic embedder responds to degradation the way a real
    # one does. Without this, Phase 2 on the dummy backbone would show no
    # effect to correct for.
    dummy_degradation_sensitivity: float = 0.55


@dataclass
class AdapterConfig:
    """The federated module. Phase 2 builds this; the shape is fixed here.

    hidden_dim was 256 until the regularisation sweep. At 256 the adapter has
    263k parameters against 264-890 training images per node, and it overfits
    hard: held-out rank-1 peaked at round 1 and then fell for 29 more rounds
    while the training objective kept improving. At 64 it holds its gain.
    See results/regularisation_sweep.json.
    """

    in_dim: int = 512
    hidden_dim: int = 64
    out_dim: int = 512
    dropout: float = 0.1
    residual: bool = True


# --------------------------------------------------------------------------
# Training and federation (consumed in Phase 2)
# --------------------------------------------------------------------------

@dataclass
class TrainConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-4
    local_epochs: int = 1          # per federated round
    centralised_epochs: int = 40   # for the pooled upper bound
    batch_size: int = 128

    # "infonce"  - symmetric contrastive loss; in-batch negatives are the
    #              node's own other people, which is exactly the hard-negative
    #              story the report makes. The default.
    # "triplet"  - explicit hardest-in-batch negative mining.
    # "cosine"   - plain cosine embedding loss on positive/negative pairs.
    loss: str = "infonce"
    temperature: float = 0.07      # infonce
    margin: float = 0.3            # triplet / cosine
    hard_negative_ratio: float = 0.5   # triplet: how often to take the hardest


@dataclass
class FLConfig:
    rounds: int = 30
    clients_per_round: Optional[int] = None  # None = all nodes every round
    # FedProx proximal term. 0.0 disables it, which makes this plain FedAvg.
    fedprox_mu: float = 3.0

    # -- early stopping ----------------------------------------------------
    # The per-node val_idx split shares its identities with train_idx, so it
    # measures whether the adapter has memorised this node's people, not
    # whether it generalises to people it has never seen. It cannot detect the
    # failure that actually happens: held-out rank-1 peaks in the first round
    # or two and then falls for the rest of the run, from 0.4897 at round 1 to
    # 0.4809 by round 30 on the recorded GPU run - below the frozen backbone
    # it started from.
    #
    # So carve this fraction of each node's identities out of training
    # entirely and score the global adapter on them, gallery-and-probe, the
    # same way the test set is scored. Nothing here touches a held-out
    # identity, so selecting the best round on it is not leakage.
    val_identity_frac: float = 0.2
    # Keep the best round by that score rather than whatever the last round
    # happened to produce. Set false to reproduce the old behaviour.
    early_stop: bool = True


# --------------------------------------------------------------------------
# Privacy (consumed in Phase 3)
# --------------------------------------------------------------------------

@dataclass
class PrivacyConfig:
    # -- differential privacy ---------------------------------------------
    enable_dp: bool = False
    clip_norm: float = 2.0
    noise_multiplier: float = 0.0     # 0 disables noise but keeps clipping
    delta: float = 1e-5
    # Renyi orders searched when converting the RDP bound to (eps, delta).
    dp_orders: List[float] = field(default_factory=lambda: [
        1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0,
        16.0, 24.0, 32.0, 48.0, 64.0, 128.0, 256.0,
    ])

    # -- secure aggregation -----------------------------------------------
    enable_masking: bool = False
    # Diffie-Hellman group used for pairwise key agreement between nodes.
    # 2048 is RFC 3526 group 14; 1024 is faster and fine for a demo.
    dh_bits: int = 2048

    # -- cancelable templates ---------------------------------------------
    # 96 bits, the width [2] uses, does not survive surveillance degradation:
    # it cost 53 points of rank-1 here (0.9226 -> 0.3962) and left the genuine
    # score below the impostor tail, so the pipeline could not raise an alert
    # on a person who was present. A wide sweep isolated quantisation, not
    # dimension reduction, as the cause. 512 bits costs 9 points instead of 53
    # and still fits in 64 bytes. See results/template_sweep_wide.json.
    template_bits: int = 512          # must be a multiple of 8
    template_key_bits: int = 512
    # Master secret the per-case, per-node keys are derived from. In a real
    # deployment this lives in the server's key store, never in a config file.
    template_master_key: str = "fedmps-demo-master-key"


# --------------------------------------------------------------------------
# Video pipeline (consumed in Phase 4)
# --------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    sample_fps: int = 8
    min_face_px: int = 24
    max_blur_ratio: float = 0.35
    max_yaw_deg: float = 45.0
    # How a live face is compared against the watchlist.
    #   "template" - keyed 512-bit codes, Hamming distance. Revocable and
    #                unlinkable across sites, but measurably lossy: on this
    #                data it finds far fewer people at the same false-alarm
    #                rate than raw embeddings do.
    #   "cosine"   - the adapted float embedding directly. More accurate, no
    #                template protection.
    # Running both and reporting the gap is the honest way to present the
    # cost of the privacy mechanism, rather than quoting whichever flatters.
    match_mode: str = "cosine"
    # Operating points, each derived from the DET curve at 1% FPIR on the
    # current artefacts. They are not interchangeable - the two scores live on
    # different scales - so each mode carries its own.
    # A threshold is only meaningful against a stated watchlist size. The
    # benchmark enrols 1,750 identities and needs ~0.30 to hold 1% FPIR; a
    # node carrying a handful of open cases has far fewer chances to collide
    # and can run lower. 0.24 is the 1%-FPIR point for a small watchlist:
    # genuine matches on the demo footage reach ~0.33 while other people in
    # frame sit near 0.05. Re-derive it from the DET curve if the deployed
    # watchlist grows - do not carry this number over blindly.
    match_threshold: float = 0.24          # cosine mode, small watchlist
    match_threshold_template: float = 0.2656   # template mode

    def threshold(self) -> float:
        return (self.match_threshold_template
                if self.match_mode == "template" else self.match_threshold)
    track_vote_k: int = 3
    track_vote_n: int = 5
    stop_on_alert: bool = False


@dataclass
class EvalConfig:
    fpir_points: List[float] = field(
        default_factory=lambda: [1e-2, 1e-3, 1e-4]
    )
    far_points: List[float] = field(
        default_factory=lambda: [1e-2, 1e-3, 1e-4]
    )
    # Held-out identities are split: this fraction are enrolled in the gallery
    # (they are the "missing persons"), the rest act as non-mate probes (the
    # crowd walking past the camera who are in no watchlist at all). Without
    # non-mates there is no open-set evaluation, only a closed-set one.
    gallery_frac: float = 0.5
    # How many verification pairs to sample for the LFW-comparable number.
    n_verification_pairs: int = 6000
    n_bootstrap: int = 0  # 0 = no confidence intervals
    eval_every: int = 1   # federated rounds between global evaluations

    # -- operating point ---------------------------------------------------
    # How many enrolment photographs make up one gallery entry, and how many
    # consecutive camera frames are fused into one probe. Both were fixed at 1
    # in the original protocol, which is the single most pessimistic point on
    # the whole surface and does not describe what this system actually does:
    # `pipeline.match` already votes over a track of `match.track_vote_n`
    # frames, and a real missing-person case file carries several photographs,
    # not one. Measured on the held-out identities, with the 1,750-identity
    # gallery unchanged:
    #
    #     enrol x frames     1x1     1x3     3x1     3x3     5x5
    #     rank-1          0.4876  0.7764  0.6019  0.8865  0.9598
    #
    # The 1x1 row stays in the reported table - it is the honest worst case,
    # and `experiments.operating_points` prints the whole surface - but it is
    # not the number to quote as the system's accuracy.
    n_enrol: int = 3
    n_probe_frames: int = 3

    # -- scoring layer -----------------------------------------------------
    # Centering, a blended orthogonal rotation and a hubness penalty, all
    # fitted on node-training identities only. Worth about +2.4 rank-1 points
    # at every operating point above. See eval/calibrate.py for why this
    # rigid correction beats the federated MLP adapter, which measures worse
    # than doing nothing once trained to convergence.
    calibrate: bool = True
    calibrate_blend: float = 0.5   # identity <- 0 .. 1 -> full Procrustes
    hub_k: int = 200               # cohort neighbours per gallery entry
    hub_beta: float = 1.0          # 0 disables the hubness penalty


# --------------------------------------------------------------------------
# Root config
# --------------------------------------------------------------------------

@dataclass
class Config:
    seed: int = 1337
    paths: Paths = field(default_factory=Paths)
    data: DataConfig = field(default_factory=DataConfig)
    degrade: DegradeConfig = field(default_factory=DegradeConfig)
    partition: PartitionConfig = field(default_factory=PartitionConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    fl: FLConfig = field(default_factory=FLConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # ---- serialisation ---------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict. Paths become strings."""

        def _clean(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_clean(v) for v in obj]
            return obj

        return _clean(asdict(self))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def summary(self) -> str:
        """One-line-per-item summary for logs and the report appendix."""
        d = self.to_dict()
        lines = [f"seed = {self.seed}"]
        for section in ("data", "partition", "backbone", "adapter", "fl", "privacy"):
            for k, v in d[section].items():
                lines.append(f"{section}.{k} = {v}")
        return "\n".join(lines)


def default_config() -> Config:
    """The single entry point everything else should use."""
    cfg = Config()
    cfg.partition.resolve()
    cfg.paths.ensure()
    return cfg


if __name__ == "__main__":  # smoke check
    cfg = default_config()
    print(cfg.summary())
    print()
    print("adapter params (approx):",
          cfg.adapter.in_dim * cfg.adapter.hidden_dim
          + cfg.adapter.hidden_dim * cfg.adapter.out_dim)
    assert cfg.adapter.in_dim == cfg.adapter.out_dim == 512
    assert cfg.privacy.template_bits % 8 == 0
    print("config OK")
