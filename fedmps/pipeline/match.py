"""Matching a live camera feed against the protected watchlist.

The node holds no photographs. Its watchlist is a set of 96-bit keyed codes,
and it protects its own live embedding with the same key before comparing, so
matching happens entirely between protected representations.

The full path for one face:

    crop -> frozen backbone -> 512-d
         -> federated adapter -> corrected 512-d
         -> select, binarise, XOR key -> 96 bits
         -> Hamming distance against each watchlist entry
         -> track-level vote -> alert

The vote is what makes the output usable. A single frame crossing the
threshold means very little: a camera sees thousands of people a day, and at
any per-comparison false-positive rate that a benchmark would call excellent,
frame-level alerting buries an operator. Requiring k of the last n frames of
one track to agree collapses that, and the pipeline records both the
frame-level and the track-level counts so the report can show by how much.

Note the frames within a track are correlated, so the reduction is
sub-geometric. Measure it, never assume independence.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..config import Config
from ..contracts import F_EMB_DEG, F_LABELS, Artefacts, load_npy
from ..models.adapter import Adapter, build_adapter
from ..models.backbone import Backbone, get_backbone
from ..privacy.template import (
    TemplateSelector, binarise, derive_key, fit_selector, load_selector,
    protect, save_selector, similarity_matrix, unpack,
)
from ..store import Store, key_fingerprint
from .detect import Detection, FaceDetector, Track, Tracker, get_detector, get_tracker
from .quality import QualityGate

SELECTOR_FILE = "selector.json"


def log(msg: str) -> None:
    print(f"[match] {msg}", flush=True)


# --------------------------------------------------------------------------
# Shared components
# --------------------------------------------------------------------------

def get_selector(cfg: Config) -> TemplateSelector:
    """Load the fitted dimension selector, fitting it once if it is missing.

    Fitted on Phase 1's training embeddings, never on evaluation data. It
    holds only indices and per-dimension means, so it is safe to ship in the
    clear to every node.
    """
    path = cfg.paths.artifacts / SELECTOR_FILE
    if path.exists():
        return load_selector(path)

    log("no selector on disk; fitting one from the Phase 1 embeddings")
    emb = load_npy(cfg.paths.artifacts / F_EMB_DEG, "embeddings_deg")
    lab = load_npy(cfg.paths.artifacts / F_LABELS, "labels")
    sel = fit_selector(emb, cfg.privacy.template_bits, lab)
    save_selector(sel, path)
    log(f"fitted and saved {sel.n_bits}-bit selector by {sel.criterion}")
    return sel


def load_adapter(cfg: Config) -> Optional[Adapter]:
    """The federated adapter from Phase 2/3, if one has been trained."""
    import torch

    path = cfg.paths.checkpoints / "adapter_final.pt"
    if not path.exists():
        log("no trained adapter found; running on raw backbone embeddings. "
            "Matching will be measurably worse - run the Phase 3 experiments "
            "first for the real pipeline.")
        return None
    blob = torch.load(path, map_location=cfg.backbone.device, weights_only=False)
    model = build_adapter(cfg.adapter, cfg.backbone.device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    log(f"loaded the federated adapter (trained on backbone "
        f"'{blob.get('backbone')}')")
    return model


@dataclass
class WatchlistEntry:
    """One case, as this node holds it.

    `stored` is what sits on disk and what crossed the network: keyed, and
    unlinkable to the same person's template at any other node. `comparable`
    is the de-keyed form the node needs in memory to actually match against.

    That the node can produce the second from the first is not a flaw in the
    scheme, it is the scheme: cancelable templates defend a database at rest
    and defeat cross-site linkage. A live node holds the key and sees its own
    camera feed regardless, so it was never the thing being defended against.
    The report should say exactly this rather than implying more.
    """

    case_id: str
    case_name: str
    stored: np.ndarray        # [k] bool, keyed - the at-rest form
    comparable: np.ndarray    # [k] bool, de-keyed - held only in memory
    key_fp: str
    # Float embedding, present only when the server distributed one for
    # unprotected cosine matching. None in template mode.
    reference: Optional[np.ndarray] = None


def load_watchlist(store: Store, node_id: int, cfg: Config
                   ) -> List[WatchlistEntry]:
    """This node's active cases, ready for matching.

    Each entry carries its own per-case, per-node key, so closing one case
    retires exactly one key and nothing else. The key is re-derived locally
    from the shared master secret; a real deployment would have the server
    deliver it over the secure channel instead. That shortcut is a demo
    convenience and belongs in the limitations section, not glossed over.
    """
    n_bits = cfg.privacy.template_bits
    out: List[WatchlistEntry] = []
    for row in store.templates_for_node(node_id):
        packed = np.frombuffer(row["template"], dtype=np.uint8)
        stored = unpack(packed, n_bits)
        key = derive_key(cfg.privacy.template_master_key,
                         row["case_id"], node_id, n_bits)
        expected_fp = key_fingerprint(key)
        if expected_fp != row["key_fp"]:
            log(f"  skipping {row['case_id']}: key fingerprint mismatch "
                f"(expected {expected_fp}, template says {row['key_fp']}). "
                f"The case was probably reissued under a new key.")
            continue
        ref = row.get("reference_embedding")
        out.append(WatchlistEntry(
            case_id=row["case_id"],
            case_name=row.get("name") or row["case_id"],
            stored=stored,
            comparable=protect(stored, key),   # XOR is self-inverse
            key_fp=row["key_fp"],
            reference=(np.frombuffer(ref, dtype=np.float32)
                       if ref else None),
        ))
    return out


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------

class Matcher:
    """Embeds crops and scores them against the protected watchlist."""

    def __init__(self, cfg: Config, node_id: int,
                 watchlist: Sequence[WatchlistEntry],
                 backbone: Optional[Backbone] = None,
                 adapter: Optional[Adapter] = None,
                 selector: Optional[TemplateSelector] = None) -> None:
        self.cfg = cfg
        self.node_id = node_id
        self.watchlist = list(watchlist)
        self.backbone = backbone or get_backbone(cfg)
        self.adapter = adapter
        self.selector = selector or get_selector(cfg)
        # De-keyed watchlist codes, held in memory only. Each entry was keyed
        # differently on disk; matching happens in the common comparison space.
        self.mode = cfg.pipeline.match_mode
        self._gallery = (np.stack([e.comparable for e in self.watchlist])
                         if self.watchlist else None)
        # Cosine mode compares adapted float embeddings directly, so the node
        # needs the enrolled embedding rather than its binarised code. The
        # template still governs what crossed the network; this is the local
        # comparison space.
        self._gallery_float = (np.stack([e.reference for e in self.watchlist])
                               if self.watchlist
                               and getattr(self.watchlist[0], "reference", None)
                               is not None else None)

    def embed_crops(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Crops -> corrected 512-d embeddings."""
        if not crops:
            return np.zeros((0, self.cfg.adapter.in_dim), dtype=np.float32)
        size = self.cfg.data.image_size
        batch = np.stack([
            cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA)
            for c in crops
        ]).astype(np.uint8)

        emb = self.backbone.embed(batch, batch_size=self.cfg.backbone.batch_size,
                                  progress=False)
        if self.adapter is not None:
            emb = self.adapter.transform_numpy(emb)
        return emb

    def score(self, crops: Sequence[np.ndarray]
              ) -> List[List[Tuple[str, float]]]:
        """One (case_id, score) list per crop, best first."""
        if not crops or self._gallery is None:
            return [[] for _ in crops]

        emb = self.embed_crops(crops)
        if self.mode == "cosine" and self._gallery_float is not None:
            sims = emb @ self._gallery_float.T
        else:
            sims = similarity_matrix(binarise(emb, self.selector),
                                     self._gallery)

        out: List[List[Tuple[str, float]]] = []
        for row in sims:
            order = np.argsort(-row)
            out.append([(self.watchlist[i].case_id, float(row[i]))
                        for i in order])
        return out


# --------------------------------------------------------------------------
# Track-level voting
# --------------------------------------------------------------------------

class TrackVoter:
    """Requires k of the last n frames of a track to agree before alerting."""

    def __init__(self, cfg: Config) -> None:
        p = cfg.pipeline
        self.k, self.n = p.track_vote_k, p.track_vote_n
        self.threshold = p.threshold()
        self._window: Dict[Tuple[int, str], Deque[bool]] = defaultdict(
            lambda: deque(maxlen=self.n))
        self._fired: set = set()
        self.frame_hits = 0
        self.track_alerts = 0

    def observe(self, track_id: int, case_id: str, score: float
                ) -> Optional[Dict[str, Any]]:
        """Record one frame's score. Returns alert details when it fires."""
        hit = score >= self.threshold
        if hit:
            self.frame_hits += 1

        key = (track_id, case_id)
        window = self._window[key]
        window.append(hit)

        if key in self._fired or sum(window) < self.k:
            return None

        self._fired.add(key)
        self.track_alerts += 1
        return {
            "track_id": track_id, "case_id": case_id, "score": score,
            "frames_voted": int(sum(window)), "frames_seen": len(window),
        }

    def report(self) -> Dict[str, Any]:
        """The number the report needs: how much the vote suppressed."""
        return {
            "rule": f"{self.k} of the last {self.n} frames",
            "threshold": self.threshold,
            "frame_level_hits": self.frame_hits,
            "track_level_alerts": self.track_alerts,
            "suppression_factor": (self.frame_hits / self.track_alerts
                                   if self.track_alerts else float("inf")),
            "note": ("frames within a track are correlated, so this reduction "
                     "is sub-geometric and must be measured, never assumed"),
        }


# --------------------------------------------------------------------------
# The video pipeline
# --------------------------------------------------------------------------

_GREY = (140, 140, 140)
_GREEN = (60, 160, 90)
_RED = (40, 50, 200)


class VideoPipeline:
    """Detect, track, gate, embed, match, vote, alert - on a video file."""

    def __init__(self, cfg: Config, node_id: int, store: Store,
                 detector: Optional[FaceDetector] = None,
                 tracker: Optional[Tracker] = None,
                 matcher: Optional[Matcher] = None) -> None:
        self.cfg = cfg
        self.node_id = node_id
        self.store = store
        self.detector = detector or get_detector(cfg)
        self.tracker = tracker or get_tracker(cfg)
        self.gate = QualityGate(cfg)
        self.voter = TrackVoter(cfg)

        watchlist = load_watchlist(store, node_id, cfg)
        log(f"node {node_id}: {len(watchlist)} active case(s) on the watchlist")
        self.matcher = matcher or Matcher(cfg, node_id, watchlist,
                                          adapter=load_adapter(cfg))

    def process_video(self, video_path: Path, out_path: Optional[Path] = None,
                      camera: str = "cam-A",
                      max_frames: Optional[int] = None,
                      fps_cap: Optional[float] = None) -> Dict[str, Any]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"could not open video: {video_path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        stride = max(1, int(round(src_fps / max(1, self.cfg.pipeline.sample_fps))))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                max(1, self.cfg.pipeline.sample_fps), (width, height))

        crops_dir = self.cfg.paths.media / "alerts"
        crops_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        frame_idx = processed = 0
        alerts: List[Dict[str, Any]] = []
        _min_frame_gap = (1.0 / fps_cap) if fps_cap and fps_cap > 0 else 0.0
        _last_frame_wall = 0.0

        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frame_idx += 1
            if (frame_idx - 1) % stride:
                # Seek forward instead of decoding the skipped frame — saves
                # the JPEG decode cost on every skipped frame.
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                continue
            if max_frames and processed >= max_frames:
                break
            processed += 1

            # Wall-clock FPS cap: yield to the OS so we don't pin a core.
            if _min_frame_gap > 0:
                now = time.time()
                wait = _min_frame_gap - (now - _last_frame_wall)
                if wait > 0:
                    time.sleep(wait)
                _last_frame_wall = time.time()

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tracks = self.tracker.update(self.detector.detect(rgb))

            passed: List[Track] = []
            crops: List[np.ndarray] = []
            annotations: List[Tuple[Track, str, Tuple[int, int, int]]] = []

            for tr in tracks:
                det = tr.detection or Detection(tr.bbox, tr.score)
                q = self.gate.assess(rgb, det)
                if not q.passed:
                    annotations.append((tr, q.reason, _GREY))
                    continue
                passed.append(tr)
                crops.append(det.crop(rgb))

            for tr, ranked in zip(passed, self.matcher.score(crops)):
                if not ranked:
                    annotations.append((tr, f"#{tr.track_id}", _GREEN))
                    continue
                case_id, score = ranked[0]
                fired = self.voter.observe(tr.track_id, case_id, score)

                if fired:
                    crop_path = crops_dir / (
                        f"node{self.node_id}_t{tr.track_id}_{case_id}.png")
                    idx = passed.index(tr)
                    cv2.imwrite(str(crop_path),
                                cv2.cvtColor(crops[idx], cv2.COLOR_RGB2BGR))
                    alert_id = self.store.add_alert(
                        case_id=case_id, node_id=self.node_id, score=score,
                        camera=camera, track_id=tr.track_id,
                        frames_voted=fired["frames_voted"],
                        frames_seen=fired["frames_seen"],
                        crop_path=str(crop_path))
                    fired["alert_id"] = alert_id
                    alerts.append(fired)
                    log(f"  ALERT #{alert_id}: {case_id} on track "
                        f"{tr.track_id}, score {score:.3f} "
                        f"({fired['frames_voted']}/{fired['frames_seen']} frames)"
                        f" - PENDING local review")
                    if getattr(self.cfg.pipeline, "stop_on_alert", False):
                        log("  [fast mode] sighting confirmed — stopping scan immediately!")
                        annotations.append((tr, f"#{tr.track_id} {score:+.2f}", _RED))
                        break

                colour = _RED if score >= self.cfg.pipeline.match_threshold \
                    else _GREEN
                annotations.append((tr, f"#{tr.track_id} {score:+.2f}", colour))

            if writer is not None:
                writer.write(self._annotate(bgr, annotations, processed))

            if alerts and getattr(self.cfg.pipeline, "stop_on_alert", False):
                break

        cap.release()
        if writer is not None:
            writer.release()

        elapsed = time.time() - t0
        stats = {
            "video": str(video_path),
            "node_id": self.node_id,
            "camera": camera,
            "frames_in_file": frame_idx,
            "frames_processed": processed,
            "sample_stride": stride,
            "seconds": round(elapsed, 1),
            "fps": round(processed / elapsed, 2) if elapsed > 0 else 0.0,
            "quality_gate": self.gate.report(),
            "voting": self.voter.report(),
            "alerts_raised": len(alerts),
            "alerts": alerts,
            "annotated_output": str(out_path) if out_path else None,
        }

        g = stats["quality_gate"]
        log(f"processed {processed} frames at {stats['fps']:.1f} fps")
        log(f"quality gate discarded {g['rejection_rate']:.0%} of faces "
            f"({g['by_reason']})")
        v = stats["voting"]
        log(f"voting: {v['frame_level_hits']} frame hits -> "
            f"{v['track_level_alerts']} track alerts")
        return stats

    def _annotate(self, bgr: np.ndarray,
                  annotations: Sequence[Tuple[Track, str, Tuple[int, int, int]]],
                  frame_no: int) -> np.ndarray:
        out = bgr.copy()
        for tr, label, colour in annotations:
            x1, y1, x2, y2 = [int(v) for v in tr.bbox]
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
            cv2.rectangle(out, (x1, max(0, y1 - 18)), (x1 + 8 * len(label) + 6, y1),
                          colour, -1)
            cv2.putText(out, label, (x1 + 3, max(11, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                        cv2.LINE_AA)
        banner = (f"node {self.node_id} | frame {frame_no} | "
                  f"raw video never leaves this device")
        cv2.putText(out, banner, (8, out.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1,
                    cv2.LINE_AA)
        return out


if __name__ == "__main__":  # smoke check
    from ..config import default_config

    cfg = default_config()

    # -- the voter, which is the piece with real logic --------------------
    cfg.pipeline.track_vote_k = 3
    cfg.pipeline.track_vote_n = 5
    cfg.pipeline.match_threshold = 0.4
    v = TrackVoter(cfg)

    # Two isolated spikes must not fire.
    assert v.observe(1, "c1", 0.9) is None
    assert v.observe(1, "c1", 0.1) is None
    assert v.observe(1, "c1", 0.9) is None
    print("isolated spikes suppressed OK")

    # A third hit inside the window fires exactly once.
    fired = v.observe(1, "c1", 0.9)
    assert fired is not None, "three hits in the window failed to fire"
    assert fired["frames_voted"] == 3
    assert v.observe(1, "c1", 0.9) is None, "the same track alerted twice"
    print(f"fired once: {fired}")

    # A different track is tracked independently.
    assert v.observe(2, "c1", 0.9) is None
    rep = v.report()
    print(f"voting report: {rep['frame_level_hits']} frame hits -> "
          f"{rep['track_level_alerts']} alerts "
          f"({rep['suppression_factor']:.1f}x suppression)")
    assert rep["track_level_alerts"] == 1 and rep["frame_level_hits"] == 5

    # -- the keyed comparison round-trip ----------------------------------
    from ..contracts import l2_normalise
    from ..privacy.template import fit_selector

    rng = np.random.default_rng(0)
    emb = l2_normalise(rng.normal(size=(50, cfg.adapter.in_dim)))
    sel = fit_selector(emb, cfg.privacy.template_bits)
    key = derive_key("m", "case-shared", 0, cfg.privacy.template_bits)

    enrolled = protect(binarise(emb[:1], sel), key)
    live_same = protect(binarise(emb[:1], sel), key)
    live_other = protect(binarise(emb[1:2], sel), key)

    s_same = float(similarity_matrix(live_same, enrolled)[0, 0])
    s_other = float(similarity_matrix(live_other, enrolled)[0, 0])
    print(f"keyed match: same person {s_same:+.3f}, "
          f"different person {s_other:+.3f}")
    assert s_same > 0.99, "a face failed to match its own enrolled template"
    assert s_other < s_same, "an impostor scored as high as the genuine match"

    print("match OK")
