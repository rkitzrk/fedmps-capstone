"""The frozen face embedder.

Nothing in this project ever trains a backbone. It is downloaded, frozen, and
used purely to turn a face crop into a 512-d vector. The federated
contribution sits in the adapter on top (Phase 2), which is precisely the
design decision the report defends.

Three interchangeable implementations behind one interface, tried in the order
given by `cfg.backbone.preference`:

  insightface - ArcFace w600k_r50 from the buffalo_l pack. The real thing, and
                what the report should quote. Awkward to install on Windows.
  facenet     - InceptionResnetV1 / VGGFace2 via facenet-pytorch. Also 512-d,
                noticeably easier to install, entirely respectable to use.
  dummy       - a synthetic embedder. Always available.

The dummy exists because the build is written in four phases and tested only
at the end. Without it, a failed `insightface` install would leave the whole
codebase unexercisable. With it, Phases 2 and 3 can be run end to end on
synthetic vectors that have real identity structure and that genuinely
degrade - so plumbing bugs surface immediately and only the final numbers wait
on a working install.
"""

from __future__ import annotations

import zlib
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from ..config import BackboneConfig, Config
from ..contracts import EMBED_DIM, assert_embeddings, l2_normalise
from ..data.degrade import sharpness


def _log(msg: str) -> None:
    print(f"[backbone] {msg}", flush=True)


# --------------------------------------------------------------------------
# Alignment as a separate pass
# --------------------------------------------------------------------------

def align_faces(images: np.ndarray, cfg: Config,
                out_size: int = 112) -> "tuple[np.ndarray, Dict[str, Any]]":
    """Warp every image onto ArcFace's landmark template, once, up front.

    Phase 1 embeds each image twice - clean, and again after the degradation
    model - and those two passes have to be framed identically or the
    comparison measures the wrong thing. Detecting separately on each would
    not achieve that: a face degraded to 30 pixels frequently defeats the
    detector, the degraded branch silently falls back to a plain resize, and
    the resulting gap between clean and degraded is part resolution loss and
    part framing mismatch, with no way to tell which.

    So align once, here, on the clean image, and let `degrade_all` work on the
    aligned crop. Both branches then share identical geometry and the only
    variable left between them is pixel quality, which is the variable the
    project is actually about. Detector behaviour on real low-resolution
    frames is a separate question, and `pipeline/detect.py` is where it is
    answered.

    Images with no detectable face are returned resized rather than dropped,
    so the array stays index-aligned with labels. The returned report carries
    the miss rate; the caller should refuse to continue if it is high, because
    past a few percent the embeddings are a mixture of two populations.
    """
    from insightface.model_zoo import get_model
    from insightface.utils import face_align, storage

    import os

    model_dir = storage.ensure_available("models", cfg.backbone.insightface_model)
    det_path = os.path.join(str(model_dir), cfg.backbone.insightface_det)
    want_cuda = cfg.backbone.device == "cuda"
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if want_cuda else ["CPUExecutionProvider"])
    try:
        det = get_model(det_path, providers=providers)
    except TypeError:
        det = get_model(det_path)
    det.prepare(ctx_id=0 if want_cuda else -1,
                input_size=(cfg.backbone.det_size, cfg.backbone.det_size),
                det_thresh=cfg.backbone.det_thresh)

    session = getattr(det, "session", None)
    actual = list(session.get_providers()) if session is not None else []
    _log(f"aligning {images.shape[0]} images via "
         f"{cfg.backbone.insightface_det} at {cfg.backbone.det_size}px "
         f"providers={actual or 'unknown'}")

    n = int(images.shape[0])
    out = np.empty((n, out_size, out_size, 3), dtype=np.uint8)
    misses = 0

    for i in range(n):
        bgr = cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR)
        try:
            boxes, kpss = det.detect(bgr, max_num=0, metric="default")
        except Exception:
            boxes, kpss = None, None

        if boxes is None or len(boxes) == 0 or kpss is None or len(kpss) == 0:
            misses += 1
            crop = cv2.resize(bgr, (out_size, out_size),
                              interpolation=cv2.INTER_AREA)
        else:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            crop = face_align.norm_crop(bgr, kpss[int(np.argmax(areas))],
                                        image_size=out_size)
        out[i] = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        if i % 5000 == 0 and i:
            _log(f"aligned {i}/{n} ({misses} missed)")

    report = {
        "detector": cfg.backbone.insightface_det,
        "det_size": cfg.backbone.det_size,
        "det_thresh": cfg.backbone.det_thresh,
        "out_size": out_size,
        "aligned": n - misses,
        "missed": misses,
        "miss_rate": misses / max(n, 1),
    }
    _log(f"alignment done: {report['aligned']}/{n} "
         f"({100 * report['miss_rate']:.2f}% missed)")
    return out, report


class Backbone(ABC):
    """RGB uint8 face crops in, L2-normalised 512-d vectors out."""

    name: str = "abstract"
    dim: int = EMBED_DIM

    @abstractmethod
    def _embed_batch(self, images: np.ndarray,
                     labels: Optional[np.ndarray]) -> np.ndarray:
        ...

    def embed(self, images: np.ndarray,
              labels: Optional[np.ndarray] = None,
              batch_size: int = 64,
              progress: bool = True) -> np.ndarray:
        """Embed [N, S, S, 3] uint8 RGB. Returns [N, 512] float32, L2-normed.

        `labels` is used only by the dummy backbone, which needs identity hints
        to synthesise structure. Real backbones ignore it.
        """
        n = int(images.shape[0])
        out = np.empty((n, self.dim), dtype=np.float32)
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            lab = None if labels is None else labels[start:stop]
            out[start:stop] = self._embed_batch(images[start:stop], lab)
            if progress and (start // batch_size) % 10 == 0:
                _log(f"{self.name}: {stop}/{n}")
        out = l2_normalise(out)
        return assert_embeddings(out, f"{self.name} output")


# --------------------------------------------------------------------------
# InsightFace / ArcFace
# --------------------------------------------------------------------------

class InsightFaceBackbone(Backbone):
    name = "insightface"

    def __init__(self, cfg: BackboneConfig) -> None:
        import os

        from insightface.model_zoo import get_model
        from insightface.utils import storage

        model_dir = storage.ensure_available("models", cfg.insightface_model)
        onnx_path = os.path.join(str(model_dir), cfg.insightface_onnx)
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"{cfg.insightface_onnx} not found in {model_dir}. "
                f"The buffalo_l pack may have a different layout in your version."
            )
        want_cuda = cfg.device == "cuda"
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if want_cuda else ["CPUExecutionProvider"])
        try:
            self.model = get_model(onnx_path, providers=providers)
        except TypeError:
            # Older insightface builds do not accept a providers argument.
            self.model = get_model(onnx_path)
        self.model.prepare(ctx_id=0 if want_cuda else -1)

        # Report the provider actually in use, not the one requested. Asking
        # for CUDA and silently getting CPU is the expensive failure here: it
        # looks identical in the logs and runs fifty times slower, so a GPU
        # instance quietly bills for hours of CPU work.
        session = getattr(self.model, "session", None)
        actual = list(session.get_providers()) if session is not None else []
        _log(f"loaded {cfg.insightface_model}/{cfg.insightface_onnx} "
             f"providers={actual or 'unknown'}")

        if want_cuda and "CUDAExecutionProvider" not in actual:
            import onnxruntime as ort

            raise RuntimeError(
                "CUDA was requested but onnxruntime is running on CPU.\n"
                f"  requested : {providers}\n"
                f"  active    : {actual}\n"
                f"  available : {ort.get_available_providers()}\n"
                f"  onnxruntime {ort.__version__}\n\n"
                "Usually one of:\n"
                "  1. the CPU package is installed, or both are:\n"
                "       pip uninstall -y onnxruntime onnxruntime-gpu\n"
                "       pip install onnxruntime-gpu\n"
                "  2. onnxruntime-gpu does not match the CUDA/cuDNN on this\n"
                "     machine - check `nvidia-smi` for the CUDA version and\n"
                "     install the matching onnxruntime-gpu build.\n\n"
                "Pass --device cpu to proceed without a GPU."
            )

        # -- the detector, for alignment ----------------------------------
        self.detector = None
        self.align_hits = 0
        self.align_misses = 0
        if cfg.align:
            det_path = os.path.join(str(model_dir), cfg.insightface_det)
            if not os.path.exists(det_path):
                raise FileNotFoundError(
                    f"{cfg.insightface_det} not found in {model_dir}; it is "
                    f"needed for alignment. Set backbone.align = False to "
                    f"embed unaligned crops, but read the note in _align "
                    f"before you do."
                )
            try:
                self.detector = get_model(det_path, providers=providers)
            except TypeError:
                self.detector = get_model(det_path)
            self.detector.prepare(ctx_id=0 if want_cuda else -1,
                                  input_size=(640, 640),
                                  det_thresh=cfg.det_thresh)
            _log(f"alignment on via {cfg.insightface_det} "
                 f"(det_thresh={cfg.det_thresh})")
        else:
            _log("alignment OFF - embeddings will be weaker; see _align")

    def _embed_batch(self, images: np.ndarray,
                     labels: Optional[np.ndarray]) -> np.ndarray:
        bgr = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in images]
        if self.detector is not None:
            bgr = [self._align(img) for img in bgr]
        feats = self.model.get_feat(bgr)
        return np.asarray(feats, dtype=np.float32).reshape(len(bgr), -1)

    # -- alignment ---------------------------------------------------------

    def _align(self, bgr: np.ndarray) -> np.ndarray:
        """Detect the largest face and warp it onto ArcFace's own template.

        This is the step whose absence cost the project more than every
        other defect combined. `ArcFaceONNX.get_feat` resizes its input to
        112x112 and embeds it, which reads like it handles framing - it does
        not. ArcFace is trained exclusively on crops produced by a similarity
        transform that puts five landmarks on fixed pixel coordinates. Feed it
        anything else and the face is the wrong size, in the wrong place, at
        the wrong roll angle, and the embedding degrades accordingly.

        The symptom is quiet: no error, no warning, just weak embeddings. On
        CelebA's own aligned crops it showed up only as a mean same-identity
        cosine of 0.51, where a correctly aligned w600k_r50 returns ~0.70.

        A face that cannot be detected is passed through as a plain resize -
        the old behaviour - rather than dropped, so the array stays aligned
        with `labels`. `align_misses` counts them; the prepare script reports
        it, because a high miss rate means the detector threshold is wrong for
        this data and the numbers are back to being untrustworthy.
        """
        from insightface.utils import face_align

        try:
            boxes, kpss = self.detector.detect(bgr, max_num=0, metric="default")
        except Exception:
            boxes, kpss = None, None

        if boxes is None or len(boxes) == 0 or kpss is None or len(kpss) == 0:
            self.align_misses += 1
            return bgr

        # Largest detection: on a single-subject crop the extra faces are
        # bystanders, and on a gallery photo the subject is the big one.
        areas = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
        best = int(np.argmax(areas))
        self.align_hits += 1
        return face_align.norm_crop(bgr, kpss[best], image_size=112)

    def alignment_report(self) -> Dict[str, Any]:
        total = self.align_hits + self.align_misses
        return {
            "enabled": self.detector is not None,
            "aligned": self.align_hits,
            "missed": self.align_misses,
            "miss_rate": (self.align_misses / total) if total else 0.0,
        }


# --------------------------------------------------------------------------
# facenet-pytorch
# --------------------------------------------------------------------------

class FacenetBackbone(Backbone):
    name = "facenet"

    def __init__(self, cfg: BackboneConfig) -> None:
        import torch
        from facenet_pytorch import InceptionResnetV1

        self.torch = torch
        self.device = torch.device(
            "cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.model = (InceptionResnetV1(pretrained=cfg.facenet_pretrained)
                      .eval().to(self.device))
        for p in self.model.parameters():
            p.requires_grad_(False)
        _log(f"loaded facenet/{cfg.facenet_pretrained} on {self.device}")

    def _embed_batch(self, images: np.ndarray,
                     labels: Optional[np.ndarray]) -> np.ndarray:
        torch = self.torch
        resized = np.stack([
            cv2.resize(img, (160, 160), interpolation=cv2.INTER_AREA)
            for img in images
        ])
        x = torch.from_numpy(resized).float().permute(0, 3, 1, 2)
        x = (x - 127.5) / 128.0                      # facenet standardisation
        with torch.no_grad():
            feats = self.model(x.to(self.device)).cpu().numpy()
        return np.asarray(feats, dtype=np.float32)


# --------------------------------------------------------------------------
# Dummy
# --------------------------------------------------------------------------

class DummyBackbone(Backbone):
    """A synthetic embedder with the properties that matter downstream.

    Three behaviours are modelled, because Phase 2 depends on all three:

      1. Identity structure. Every identity owns a fixed random unit vector.
         Images of that identity land near it, so matching works at all.

      2. Degradation costs accuracy. Blurrier inputs get more scatter around
         the identity centroid, so a degraded gallery scores worse than a
         clean one - which is the project's problem statement.

      3. Degradation is systematic, not just noisy. Images at a similar blur
         level are pushed in a common direction, chosen per blur bucket. This
         is the part that matters most: a purely random perturbation would be
         unlearnable, and the Phase 2 adapter would correctly find nothing.
         Real low-resolution degradation does shift embeddings consistently,
         and the adapter's whole job is to undo that shift.

    So a full dummy-backbone run should show degraded accuracy dropping and
    the adapter recovering part of it. The numbers mean nothing; the fact that
    the pipeline moves them in the right direction means the plumbing works.
    """

    name = "dummy"

    def __init__(self, cfg: BackboneConfig, seed: int = 1337) -> None:
        self.cfg = cfg
        self.seed = seed
        self._centroids: Dict[int, np.ndarray] = {}
        self._biases: Dict[int, np.ndarray] = {}
        self._sharp_ref = 100.0   # var-of-Laplacian scale for a crisp crop
        _log("using the synthetic backbone - results are structural only, "
             "not reportable")

    def _unit(self, key: int, salt: int) -> np.ndarray:
        rng = np.random.default_rng((self.seed * 1000003 + key * 7919 + salt)
                                    % (2 ** 63))
        v = rng.normal(size=EMBED_DIM)
        return (v / np.linalg.norm(v)).astype(np.float32)

    def _centroid(self, ident: int) -> np.ndarray:
        if ident not in self._centroids:
            self._centroids[ident] = self._unit(ident, salt=11)
        return self._centroids[ident]

    def _bias(self, bucket: int) -> np.ndarray:
        if bucket not in self._biases:
            self._biases[bucket] = self._unit(bucket, salt=977)
        return self._biases[bucket]

    def _embed_batch(self, images: np.ndarray,
                     labels: Optional[np.ndarray]) -> np.ndarray:
        n = images.shape[0]
        out = np.empty((n, EMBED_DIM), dtype=np.float32)

        for i in range(n):
            img = images[i]

            if labels is not None:
                ident = int(labels[i])
            else:
                # No hint: fall back to a stable hash of the pixels. Identity
                # structure is then lost, but shapes stay correct.
                ident = zlib.crc32(np.ascontiguousarray(img).tobytes()) % 100000

            s = sharpness(img)
            sharp_norm = float(s / (s + self._sharp_ref))     # in [0, 1)
            damage = 1.0 - sharp_norm

            bucket = int(np.clip(sharp_norm * 8.0, 0, 7))
            shift = self.cfg.dummy_degradation_sensitivity * damage
            noise_scale = self.cfg.dummy_identity_noise * (0.5 + damage)

            rng = np.random.default_rng(
                (self.seed + ident * 31 + zlib.crc32(
                    np.ascontiguousarray(img[::8, ::8]).tobytes())) % (2 ** 63)
            )
            noise_vec = rng.normal(size=EMBED_DIM).astype(np.float32)
            noise_norm = np.linalg.norm(noise_vec)
            if noise_norm > 0:
                noise_vec = noise_vec / noise_norm
            v = (self._centroid(ident)
                 + shift * self._bias(bucket)
                 + noise_scale * noise_vec)
            out[i] = v

        return out


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

_BUILDERS = {
    "insightface": lambda cfg, seed: InsightFaceBackbone(cfg),
    "facenet": lambda cfg, seed: FacenetBackbone(cfg),
    "dummy": lambda cfg, seed: DummyBackbone(cfg, seed),
}


def get_backbone(cfg: Config) -> Backbone:
    """Return the first backbone in the preference list that loads.

    Never raises for a missing dependency - the dummy is always reachable. It
    does raise if the preference list names something unknown, since that is a
    config typo rather than an environment problem.
    """
    errors: List[str] = []
    for name in cfg.backbone.preference:
        if name not in _BUILDERS:
            raise ValueError(
                f"unknown backbone {name!r}; expected one of {sorted(_BUILDERS)}"
            )
        try:
            return _BUILDERS[name](cfg.backbone, cfg.seed)
        except Exception as exc:                      # noqa: BLE001
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            _log(f"{name} unavailable ({type(exc).__name__}), trying next")

    raise RuntimeError("no backbone could be loaded:\n  " + "\n  ".join(errors))


if __name__ == "__main__":  # smoke check
    from ..config import default_config

    cfg = default_config()
    cfg.backbone.preference = ["dummy"]           # keep the check offline
    bb = get_backbone(cfg)

    rng = np.random.default_rng(0)
    labels = np.array([0, 0, 1, 1], dtype=np.int32)

    crisp = np.stack([
        rng.integers(0, 255, size=(160, 160, 3), dtype=np.uint8) for _ in range(4)
    ])
    blurry = np.stack([cv2.GaussianBlur(im, (21, 21), 0) for im in crisp])

    e_crisp = bb.embed(crisp, labels, progress=False)
    e_blur = bb.embed(blurry, labels, progress=False)

    assert e_crisp.shape == (4, EMBED_DIM)
    assert np.allclose(np.linalg.norm(e_crisp, axis=1), 1.0, atol=1e-4)

    def same_id_sim(e):
        return float(e[0] @ e[1])

    def diff_id_sim(e):
        return float(e[0] @ e[2])

    print(f"crisp : same-id {same_id_sim(e_crisp):+.3f}   "
          f"diff-id {diff_id_sim(e_crisp):+.3f}")
    print(f"blurry: same-id {same_id_sim(e_blur):+.3f}   "
          f"diff-id {diff_id_sim(e_blur):+.3f}")

    assert same_id_sim(e_crisp) > diff_id_sim(e_crisp), \
        "dummy backbone has no identity structure"
    assert same_id_sim(e_blur) < same_id_sim(e_crisp), \
        "dummy backbone does not respond to degradation"

    # Determinism: the same input must give the same vector.
    again = bb.embed(crisp, labels, progress=False)
    assert np.allclose(e_crisp, again, atol=1e-6), "dummy backbone is not deterministic"

    print("backbone OK")
