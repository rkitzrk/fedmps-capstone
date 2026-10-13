"""The quality gate.

Roughly seventy per cent of detected faces are thrown away here, before they
ever reach the embedding model. That is not an optimisation - it is the
single cheapest accuracy mechanism in the whole pipeline.

The reason is worth stating carefully in the report, because it is
counter-intuitive: face recognition on a twelve-pixel, heavily blurred,
eighty-degree crop does not produce a *weak* match. It produces a **confident
wrong** match. Those crops are the largest single source of false alarms, and
discarding them outright removes far more error than any amount of extra
training would.

Four checks, in increasing order of cost:

  size        inter-ocular distance, from landmarks where the detector gives
              them and estimated from face width where it does not
  blur        variance of the Laplacian, normalised by crop area so the
              threshold does not drift with face size
  exposure    mean brightness, to reject silhouettes and blown-out frames
  pose        yaw estimated from landmark asymmetry; skipped entirely when the
              detector supplies no landmarks

Every rejection is counted by reason, so the report can show which check is
doing the work rather than just quoting a total.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from ..config import Config
from .detect import Detection

# An adult face is about 2.4 inter-ocular distances wide. Used only when the
# detector gives no landmarks, and it is an approximation - say so if asked.
_IOD_FROM_WIDTH = 0.42


@dataclass
class QualityScore:
    """The verdict on one crop, with the numbers behind it."""

    passed: bool
    reason: str
    face_px: float
    iod_px: float
    blur: float
    brightness: float
    yaw_deg: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed, "reason": self.reason,
            "face_px": round(self.face_px, 1), "iod_px": round(self.iod_px, 1),
            "blur": round(self.blur, 2),
            "brightness": round(self.brightness, 1),
            "yaw_deg": None if self.yaw_deg is None else round(self.yaw_deg, 1),
        }


def inter_ocular_distance(det: Detection) -> float:
    """Pixels between the eye centres - the standard face-size measure.

    Bounding-box width is a poor substitute because it grows with hair, hats
    and detector padding, none of which help recognition.
    """
    if det.landmarks is not None and len(det.landmarks) >= 2:
        left, right = det.landmarks[0], det.landmarks[1]
        return float(np.linalg.norm(np.asarray(right) - np.asarray(left)))
    return float(det.width * _IOD_FROM_WIDTH)


def blur_score(crop: np.ndarray) -> float:
    """Variance of the Laplacian, normalised so it does not track crop size."""
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
    # Resize to a fixed size first: a large sharp crop and a small sharp crop
    # should score alike, and raw Laplacian variance does not oblige.
    fixed = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(fixed, cv2.CV_64F).var())


def estimate_yaw(det: Detection) -> Optional[float]:
    """Rough yaw from landmark asymmetry, in degrees. None without landmarks.

    The nose sits midway between the eyes on a frontal face and drifts toward
    one of them as the head turns. Crude, but it costs nothing and it catches
    the profile shots that are worth rejecting.
    """
    if det.landmarks is None or len(det.landmarks) < 3:
        return None
    kps = np.asarray(det.landmarks, dtype=np.float64)
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    eye_mid = (left_eye + right_eye) / 2.0
    eye_dist = float(np.linalg.norm(right_eye - left_eye))
    if eye_dist < 1e-6:
        return None
    offset = float(nose[0] - eye_mid[0]) / eye_dist
    return float(np.clip(offset * 90.0, -90.0, 90.0))


class QualityGate:
    """Applies the checks and keeps the rejection tally."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg.pipeline
        self.counts: Counter = Counter()
        self.total = 0

    def assess(self, frame_rgb: np.ndarray, det: Detection) -> QualityScore:
        self.total += 1
        crop = det.crop(frame_rgb, margin=0.0)

        iod = inter_ocular_distance(det)
        blur = blur_score(crop)
        brightness = float(crop.mean()) if crop.size else 0.0
        yaw = estimate_yaw(det)

        def verdict(ok: bool, reason: str) -> QualityScore:
            self.counts[reason if not ok else "passed"] += 1
            return QualityScore(ok, reason, det.width, iod, blur, brightness, yaw)

        if iod < self.cfg.min_face_px:
            return verdict(False, "too_small")
        # Threshold expressed as a fraction of a nominal sharp score, so the
        # config reads as "reject the blurriest 35%" rather than as a raw
        # Laplacian variance nobody can interpret.
        if blur < self.cfg.max_blur_ratio * 100.0:
            return verdict(False, "too_blurry")
        if brightness < 25.0 or brightness > 235.0:
            return verdict(False, "bad_exposure")
        if yaw is not None and abs(yaw) > self.cfg.max_yaw_deg:
            return verdict(False, "extreme_pose")
        return verdict(True, "passed")

    def report(self) -> Dict[str, Any]:
        passed = self.counts.get("passed", 0)
        return {
            "total_faces": self.total,
            "passed": passed,
            "rejected": self.total - passed,
            "rejection_rate": ((self.total - passed) / self.total
                               if self.total else 0.0),
            "by_reason": dict(self.counts),
            "thresholds": {
                "min_iod_px": self.cfg.min_face_px,
                "min_blur": self.cfg.max_blur_ratio * 100.0,
                "max_yaw_deg": self.cfg.max_yaw_deg,
            },
        }

    def reset(self) -> None:
        self.counts.clear()
        self.total = 0


if __name__ == "__main__":  # smoke check
    from ..config import default_config

    cfg = default_config()
    gate = QualityGate(cfg)
    rng = np.random.default_rng(0)

    def face_frame(size: int, blur: int = 0) -> np.ndarray:
        """A synthetic face with enough texture to have a real blur score."""
        img = np.full((240, 320, 3), 40, dtype=np.uint8)
        cx, cy = 160, 120
        h = size // 2
        cv2.rectangle(img, (cx - h, cy - h), (cx + h, cy + h), (200, 175, 155), -1)
        step = max(2, size // 10)
        for i in range(cx - h, cx + h, step):
            cv2.line(img, (i, cy - h), (i, cy + h), (120, 100, 90), 1)
        if blur:
            img = cv2.GaussianBlur(img, (blur * 2 + 1, blur * 2 + 1), 0)
        return img

    def det_for(size: int, with_kps: bool = True, yaw_shift: float = 0.0):
        cx, cy, h = 160, 120, size // 2
        box = (cx - h, cy - h, cx + h, cy + h)
        kps = None
        if with_kps:
            eye = size * 0.42 / 2
            kps = np.array([
                [cx - eye, cy - size * 0.1],
                [cx + eye, cy - size * 0.1],
                [cx + yaw_shift * eye * 2, cy + size * 0.05],
                [cx - eye * 0.6, cy + size * 0.25],
                [cx + eye * 0.6, cy + size * 0.25],
            ])
        return Detection(box, 0.95, kps)

    # A big sharp face passes.
    q = gate.assess(face_frame(120), det_for(120))
    print(f"large sharp : {q.to_dict()}")
    assert q.passed, f"a good crop was rejected as {q.reason}"

    # A tiny one is rejected on size.
    q = gate.assess(face_frame(30), det_for(30))
    print(f"tiny        : {q.reason}, iod={q.iod_px:.1f}px")
    assert not q.passed and q.reason == "too_small"

    # A heavily blurred one is rejected on blur.
    q = gate.assess(face_frame(120, blur=25), det_for(120))
    print(f"blurred     : {q.reason}, blur={q.blur:.1f}")
    assert not q.passed and q.reason == "too_blurry"

    # A dark frame is rejected on exposure.
    dark = (face_frame(120) * 0.05).astype(np.uint8)
    q = gate.assess(dark, det_for(120))
    print(f"dark        : {q.reason}, brightness={q.brightness:.1f}")
    assert not q.passed and q.reason in ("bad_exposure", "too_blurry")

    # A profile shot is rejected on pose.
    q = gate.assess(face_frame(120), det_for(120, yaw_shift=1.2))
    print(f"profile     : {q.reason}, yaw={q.yaw_deg}")
    assert not q.passed and q.reason == "extreme_pose"

    # Without landmarks the pose check must be skipped, not failed.
    q = gate.assess(face_frame(120), det_for(120, with_kps=False))
    assert q.yaw_deg is None, "yaw was estimated without landmarks"
    assert q.passed, f"a landmark-free crop was rejected as {q.reason}"
    print("landmark-free path OK")

    rep = gate.report()
    print(f"\ngate report: {rep['passed']}/{rep['total_faces']} passed, "
          f"reasons {rep['by_reason']}")
    assert rep["total_faces"] == 6
    print("quality OK")
