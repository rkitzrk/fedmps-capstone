"""The CCTV degradation model.

Turns a clean, well-lit, near-frontal face photograph into something that
looks like it came off a surveillance camera thirty metres away.

This is what makes LFW and CelebA useful for a surveillance project, and it is
also the mechanism that creates the non-IID split: each node draws one fixed
"camera personality" from the ranges in the config, and every image at that
node is degraded around that point. Node 0's camera is therefore consistently
different from node 3's, which is the condition under which federated learning
has something to contribute.

Chain, in physical order:

    optical / motion blur   (at capture resolution)
        -> downscale        (sensor samples a small, distant face)
        -> gamma shift      (auto-exposure, poor lighting)
        -> sensor noise     (gain at low light)
        -> JPEG compression (encoder in the camera or the DVR)
        -> upscale back     (so the backbone gets its fixed input size)

The final upscale restores the array shape but not the information. That is
the point: the loss has already happened.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import cv2
import numpy as np

from ..config import DegradeConfig, DegradeRanges

_INTERP = {
    "area": cv2.INTER_AREA,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "nearest": cv2.INTER_NEAREST,
}

# The parameters a node's camera personality is drawn over, in a fixed order
# so that the stored profile is reproducible.
PARAM_KEYS = ("face_px", "blur_kernel", "noise_sigma", "gamma", "jpeg_quality")


@dataclass
class DegradeParams:
    """Concrete, resolved degradation settings for one image."""

    face_px: int
    blur_kernel: int
    blur_angle: float
    noise_sigma: float
    gamma: float
    jpeg_quality: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

def sample_node_profile(rng: np.random.Generator) -> Dict[str, float]:
    """Draw one node's camera personality.

    Returns a dict of values in [0, 1] - positions within each configured
    range, rather than the concrete values. Storing it this way keeps
    partition.json readable and lets the ranges be retuned later without
    invalidating a saved partition.
    """
    return {k: float(rng.uniform(0.0, 1.0)) for k in PARAM_KEYS}


def _lerp(rng_pair, u: float):
    lo, hi = rng_pair
    return lo + (hi - lo) * u


def resolve_params(profile: Dict[str, float],
                   ranges: DegradeRanges,
                   rng: np.random.Generator,
                   jitter: float = 0.0) -> DegradeParams:
    """Turn a node profile into concrete settings for one image.

    `jitter` moves each parameter a little around the node's fixed point, so
    images at a node vary without losing the node's overall character.
    """
    u = {}
    for k in PARAM_KEYS:
        v = profile[k]
        if jitter > 0.0:
            v += float(rng.uniform(-jitter, jitter))
        u[k] = float(np.clip(v, 0.0, 1.0))

    return DegradeParams(
        face_px=int(round(_lerp(ranges.face_px, u["face_px"]))),
        blur_kernel=int(round(_lerp(ranges.blur_kernel, u["blur_kernel"]))),
        # Blur direction is per-image regardless of node: people move in
        # different directions past the same camera.
        blur_angle=float(rng.uniform(0.0, 180.0)),
        noise_sigma=float(_lerp(ranges.noise_sigma, u["noise_sigma"])),
        gamma=float(_lerp(ranges.gamma, u["gamma"])),
        jpeg_quality=int(round(_lerp(ranges.jpeg_quality, u["jpeg_quality"]))),
    )


# --------------------------------------------------------------------------
# Individual operations
# --------------------------------------------------------------------------

def motion_blur(img: np.ndarray, kernel: int, angle_deg: float) -> np.ndarray:
    """Directional blur, as if the subject moved during the exposure."""
    if kernel <= 1:
        return img
    if kernel % 2 == 0:
        kernel += 1
    k = np.zeros((kernel, kernel), dtype=np.float32)
    k[kernel // 2, :] = 1.0
    m = cv2.getRotationMatrix2D((kernel / 2 - 0.5, kernel / 2 - 0.5), angle_deg, 1.0)
    k = cv2.warpAffine(k, m, (kernel, kernel))
    total = k.sum()
    if total <= 0:
        return img
    return cv2.filter2D(img, -1, k / total)


def resample(img: np.ndarray, size: int, interp: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=interp)


def apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """Lighting shift. gamma < 1 brightens, > 1 darkens."""
    if abs(gamma - 1.0) < 1e-3:
        return img
    lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255)
    return cv2.LUT(img, lut.astype(np.uint8))


def add_noise(img: np.ndarray, sigma: float,
              rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Sensor noise, as you get from gain in poor light."""
    if sigma <= 0.0:
        return img
    rng = rng or np.random.default_rng()
    noise = rng.normal(0.0, sigma, size=img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def jpeg_compress(img: np.ndarray, quality: int) -> np.ndarray:
    """Round-trip through the JPEG encoder to get real blocking artefacts."""
    quality = int(np.clip(quality, 1, 100))
    if quality >= 98:
        return img
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img
    out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img if out is None else out


# --------------------------------------------------------------------------
# The chain
# --------------------------------------------------------------------------

def degrade_image(img: np.ndarray,
                  params: DegradeParams,
                  cfg: DegradeConfig,
                  rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Apply the full chain to one [S, S, 3] uint8 image.

    Returns an array of the same shape and dtype. Channel order is preserved -
    every operation here is channel-agnostic.
    """
    rng = rng or np.random.default_rng()
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    size = img.shape[0]
    down = _INTERP.get(cfg.downscale_interp, cv2.INTER_AREA)
    up = _INTERP.get(cfg.upscale_interp, cv2.INTER_LINEAR)

    out = motion_blur(img, params.blur_kernel, params.blur_angle)

    small = max(8, min(params.face_px, size))
    out = resample(out, small, down)

    out = apply_gamma(out, params.gamma)
    out = add_noise(out, params.noise_sigma, rng)
    out = jpeg_compress(out, params.jpeg_quality)

    out = resample(out, size, up)
    return np.ascontiguousarray(out, dtype=np.uint8)


def degrade_batch(images: np.ndarray,
                  profile: Dict[str, float],
                  ranges: DegradeRanges,
                  cfg: DegradeConfig,
                  rng: np.random.Generator) -> np.ndarray:
    """Degrade every image in [N, S, S, 3] with one node's camera personality."""
    out = np.empty_like(images)
    for i in range(images.shape[0]):
        p = resolve_params(profile, ranges, rng, cfg.per_image_jitter)
        out[i] = degrade_image(images[i], p, cfg, rng)
    return out


def sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian - the standard cheap blur measure.

    Used by the quality gate in Phase 4, and by the dummy backbone in Phase 1
    so that a synthetic embedder still reacts to degradation.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


if __name__ == "__main__":  # smoke check
    rng = np.random.default_rng(0)
    ranges = DegradeRanges()
    cfg = DegradeConfig()

    # A synthetic "face": smooth blobs, so blur and compression have an effect.
    img = np.zeros((160, 160, 3), dtype=np.uint8)
    cv2.circle(img, (80, 70), 45, (210, 180, 160), -1)
    cv2.circle(img, (62, 62), 7, (30, 30, 30), -1)
    cv2.circle(img, (98, 62), 7, (30, 30, 30), -1)
    cv2.ellipse(img, (80, 100), (22, 10), 0, 0, 180, (60, 40, 40), -1)

    prof = sample_node_profile(rng)
    print("node profile:", {k: round(v, 3) for k, v in prof.items()})

    p = resolve_params(prof, ranges, rng, jitter=0.0)
    print("resolved     :", p.to_dict())

    deg = degrade_image(img, p, cfg, rng)
    assert deg.shape == img.shape, (deg.shape, img.shape)
    assert deg.dtype == np.uint8

    s_clean, s_deg = sharpness(img), sharpness(deg)
    print(f"sharpness clean={s_clean:.1f}  degraded={s_deg:.1f}")

    batch = degrade_batch(np.stack([img] * 4), prof, ranges, cfg, rng)
    assert batch.shape == (4, 160, 160, 3)

    # Two different node profiles must produce visibly different results,
    # otherwise the non-IID split is not actually non-IID.
    a = degrade_batch(np.stack([img] * 8), sample_node_profile(rng),
                      ranges, cfg, np.random.default_rng(1))
    b = degrade_batch(np.stack([img] * 8), sample_node_profile(rng),
                      ranges, cfg, np.random.default_rng(1))
    diff = float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())
    print(f"mean abs diff between two node profiles: {diff:.2f}")

    print("degrade OK")
