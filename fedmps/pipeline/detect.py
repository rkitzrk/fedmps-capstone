"""Face detection and tracking at the node.

Tracking is what turns this from the frame-by-frame system in [1] into
something an operator could use. A person walking past a camera generates
twenty or thirty frames; deciding once per track instead of once per frame is
both more accurate and far quieter, and it is the gap [1] explicitly left open.

Both stages follow the same fallback discipline as the backbone: a good
implementation first, then one that always works.

  detection   SCRFD via insightface, else OpenCV's bundled Haar cascade.
              Haar is weaker - more misses, no landmarks - but it ships inside
              opencv-python, needs no download, and keeps the demo runnable on
              a machine where insightface would not install.

  tracking    ByteTrack via supervision, else a small IoU tracker written
              here. For a corridor demo with a handful of people the IoU
              tracker is genuinely adequate; it is not a stub.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..config import Config


def _log(msg: str) -> None:
    print(f"[detect] {msg}", flush=True)


@dataclass
class Detection:
    """One face in one frame. Coordinates are pixels, (x1, y1, x2, y2)."""

    bbox: Tuple[float, float, float, float]
    score: float = 1.0
    landmarks: Optional[np.ndarray] = None   # [5, 2] eyes, nose, mouth corners

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    def crop(self, frame: np.ndarray, margin: float = 0.15) -> np.ndarray:
        """Square crop with a little context, clamped to the frame."""
        x1, y1, x2, y2 = self.bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half = max(x2 - x1, y2 - y1) * (0.5 + margin)
        h, w = frame.shape[:2]
        a = int(max(0, cx - half))
        b = int(max(0, cy - half))
        c = int(min(w, cx + half))
        d = int(min(h, cy + half))
        if c <= a or d <= b:
            return np.zeros((8, 8, 3), dtype=np.uint8)
        return frame[b:d, a:c]


@dataclass
class Track:
    """A person followed across frames."""

    track_id: int
    bbox: Tuple[float, float, float, float]
    score: float
    hits: int = 1
    age: int = 0
    detection: Optional[Detection] = None


# --------------------------------------------------------------------------
# Detectors
# --------------------------------------------------------------------------

class FaceDetector:
    name = "abstract"

    def detect(self, frame_rgb: np.ndarray) -> List[Detection]:
        raise NotImplementedError


class SCRFDDetector(FaceDetector):
    name = "scrfd"

    def __init__(self, cfg: Config) -> None:
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(name=cfg.backbone.insightface_model,
                                allowed_modules=["detection"])
        det_sz = getattr(cfg.backbone, "det_size", 320)
        self.app.prepare(ctx_id=0 if cfg.backbone.device == "cuda" else -1,
                         det_size=(det_sz, det_sz))
        _log(f"using SCRFD (insightface) at {det_sz}x{det_sz}")

    def detect(self, frame_rgb: np.ndarray) -> List[Detection]:
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        out: List[Detection] = []
        for f in self.app.get(bgr):
            x1, y1, x2, y2 = [float(v) for v in f.bbox]
            kps = getattr(f, "kps", None)
            out.append(Detection((x1, y1, x2, y2), float(f.det_score),
                                 np.asarray(kps) if kps is not None else None))
        return out


class HaarDetector(FaceDetector):
    name = "haar"

    def __init__(self, cfg: Config) -> None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(path)
        if self.cascade.empty():
            raise RuntimeError(f"could not load the Haar cascade from {path}")
        self.min_size = max(24, cfg.pipeline.min_face_px)
        _log("using the Haar cascade - weaker than SCRFD, and no landmarks, "
             "so the pose check in the quality gate is skipped")

    def detect(self, frame_rgb: np.ndarray) -> List[Detection]:
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        rects = self.cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(self.min_size, self.min_size))
        return [Detection((float(x), float(y), float(x + w), float(y + h)), 1.0)
                for (x, y, w, h) in rects]


def get_detector(cfg: Config) -> FaceDetector:
    for builder in (SCRFDDetector, HaarDetector):
        try:
            return builder(cfg)
        except Exception as exc:                      # noqa: BLE001
            _log(f"{builder.name} unavailable ({type(exc).__name__}), "
                 f"trying next")
    raise RuntimeError("no face detector could be loaded")


# --------------------------------------------------------------------------
# Tracking
# --------------------------------------------------------------------------

def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class Tracker:
    name = "abstract"

    def update(self, detections: List[Detection]) -> List[Track]:
        raise NotImplementedError


class IouTracker(Tracker):
    """Greedy IoU association with a short memory.

    A track survives `max_age` frames without a detection, which bridges the
    brief misses that any detector produces when someone turns their head.
    """

    name = "iou"

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 8) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: List[Track] = []
        self._next_id = 1

    def update(self, detections: List[Detection]) -> List[Track]:
        for t in self.tracks:
            t.age += 1

        pairs: List[Tuple[float, int, int]] = []
        for di, det in enumerate(detections):
            for ti, tr in enumerate(self.tracks):
                score = iou(det.bbox, tr.bbox)
                if score >= self.iou_threshold:
                    pairs.append((score, di, ti))
        pairs.sort(reverse=True)

        used_det: set = set()
        used_trk: set = set()
        for _, di, ti in pairs:
            if di in used_det or ti in used_trk:
                continue
            det = detections[di]
            tr = self.tracks[ti]
            tr.bbox, tr.score, tr.detection = det.bbox, det.score, det
            tr.hits += 1
            tr.age = 0
            used_det.add(di)
            used_trk.add(ti)

        for di, det in enumerate(detections):
            if di in used_det:
                continue
            self.tracks.append(Track(self._next_id, det.bbox, det.score,
                                     hits=1, age=0, detection=det))
            self._next_id += 1

        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
        return [t for t in self.tracks if t.age == 0]


class ByteTrackTracker(Tracker):
    name = "bytetrack"

    def __init__(self, cfg: Config) -> None:
        import supervision as sv

        self.sv = sv
        self.tracker = sv.ByteTrack(frame_rate=max(1, cfg.pipeline.sample_fps))
        _log("using ByteTrack (supervision)")

    def update(self, detections: List[Detection]) -> List[Track]:
        sv = self.sv
        if not detections:
            self.tracker.update_with_detections(sv.Detections.empty())
            return []

        boxes = np.array([d.bbox for d in detections], dtype=np.float32)
        scores = np.array([d.score for d in detections], dtype=np.float32)
        dets = sv.Detections(xyxy=boxes, confidence=scores,
                             class_id=np.zeros(len(detections), dtype=int))
        tracked = self.tracker.update_with_detections(dets)

        out: List[Track] = []
        for i in range(len(tracked)):
            box = tuple(float(v) for v in tracked.xyxy[i])
            tid = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
            # Re-attach the original detection, so landmarks survive tracking.
            best = max(detections, key=lambda d: iou(d.bbox, box))
            out.append(Track(tid, box, float(tracked.confidence[i]),
                             detection=best))
        return out


def get_tracker(cfg: Config) -> Tracker:
    try:
        return ByteTrackTracker(cfg)
    except Exception as exc:                          # noqa: BLE001
        _log(f"ByteTrack unavailable ({type(exc).__name__}), "
             f"using the built-in IoU tracker")
        return IouTracker()


if __name__ == "__main__":  # smoke check
    from ..config import default_config

    cfg = default_config()

    # IoU geometry.
    assert abs(iou((0, 0, 10, 10), (0, 0, 10, 10)) - 1.0) < 1e-9
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - (50 / 150)) < 1e-9
    print("IoU geometry OK")

    # The tracker must hold one identity across a walk and number two people
    # separately.
    tr = IouTracker(iou_threshold=0.3, max_age=4)
    ids_seen = []
    for step in range(12):
        x = 10 + step * 4          # a person moving slowly right
        dets = [Detection((x, 20, x + 40, 60), 0.9)]
        if step >= 4:              # a second person joins
            dets.append(Detection((200, 20, 240, 60), 0.9))
        out = tr.update(dets)
        ids_seen.append(sorted(t.track_id for t in out))

    assert ids_seen[-1] == [1, 2], f"expected two tracks, got {ids_seen[-1]}"
    assert all(1 in s for s in ids_seen), "the first track changed identity"
    print(f"tracking: final ids {ids_seen[-1]}, first track held for "
          f"{sum(1 for s in ids_seen if 1 in s)} frames")

    # A track must survive a brief detection gap, then expire.
    tr2 = IouTracker(max_age=3)
    tr2.update([Detection((0, 0, 40, 40), 0.9)])
    for _ in range(3):
        tr2.update([])
    assert len(tr2.tracks) == 1, "the track expired too early"
    for _ in range(3):
        tr2.update([])
    assert len(tr2.tracks) == 0, "the track never expired"
    print("track ageing OK")

    # Cropping must stay inside the frame even at the edge.
    frame = np.random.default_rng(0).integers(
        0, 255, size=(120, 160, 3), dtype=np.uint8)
    edge = Detection((150, 110, 200, 160), 0.9)
    c = edge.crop(frame)
    assert c.size > 0 and c.shape[2] == 3, c.shape
    print(f"edge crop shape {c.shape} OK")

    det = get_detector(cfg)
    print(f"detector selected: {det.name}")
    found = det.detect(frame)
    print(f"detections on noise: {len(found)} (zero or few is expected)")

    print("detect OK")
