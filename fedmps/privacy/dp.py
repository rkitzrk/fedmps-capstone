"""Differential privacy on the federated updates.

Two operations, applied at the node before an update is released:

    clip   bound the update's L2 norm to C. This caps how much any single
           node - and therefore any single person at that node - can move the
           global model. Without a bound there is no sensitivity, and without
           sensitivity there is no privacy guarantee at all.

    noise  add Gaussian noise calibrated to C.

This is **update-level** (sometimes "user-level") DP rather than the
per-example DP-SGD that Opacus implements. It is the right granularity here:
the thing being protected is a node's contribution, and per-example gradients
would be both fiddlier and a weaker statement about the entity that actually
holds the data.

Where the noise goes matters, and the choice depends on masking:

    masking ON   Each node adds noise of scale sigma*C/sqrt(n). Variances add,
                 so the aggregate carries exactly sigma*C - central-DP utility.
                 The server never sees an individual update, so no node has to
                 trust it. This is the intended configuration, and it is why
                 the two mechanisms are designed together rather than bolted
                 on separately.

    masking OFF  Each node must add the full sigma*C, because the server sees
                 its update in the clear. This is local DP: a much weaker
                 utility trade for the same guarantee. Useful as a comparison
                 row in the report, not as the deployed design.

Accounting is exact Renyi DP for the Gaussian mechanism under full
participation. When clients are subsampled the reported epsilon ignores
privacy amplification, which makes it an upper bound - loose, but never a
claim that overstates the protection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from ..config import Config, PrivacyConfig
from ..models.adapter import (
    StateDict, state_to_vector, vector_to_state,
)


# --------------------------------------------------------------------------
# Clipping
# --------------------------------------------------------------------------

def clip_update(update: StateDict, clip_norm: float) -> tuple:
    """Scale the update down if its L2 norm exceeds `clip_norm`.

    Returns (clipped_update, original_norm, was_clipped). The original norm is
    worth logging: if almost nothing is being clipped, C is too loose and the
    noise is larger than it needs to be; if everything is clipped hard, C is
    throttling learning. The median update norm is the number to set C near.
    """
    vec = state_to_vector(update)
    norm = float(torch.linalg.vector_norm(vec).item())

    if clip_norm <= 0 or norm <= clip_norm or norm == 0.0:
        return update, norm, False

    scale = clip_norm / norm
    clipped = vector_to_state(vec * scale, update)
    return clipped, norm, True


def add_gaussian_noise(update: StateDict, std: float,
                       generator: Optional[torch.Generator] = None) -> StateDict:
    """Add zero-mean Gaussian noise of the given standard deviation."""
    if std <= 0:
        return update
    out = type(update)()
    for k, v in update.items():
        noise = torch.normal(mean=0.0, std=std, size=v.shape,
                             generator=generator, dtype=torch.float32)
        out[k] = v + noise.to(v.dtype)
    return out


def noise_std(cfg: PrivacyConfig, n_participants: int) -> float:
    """Per-node noise standard deviation.

    Divided across participants only when masking guarantees the server sees
    nothing but the sum - see the module docstring.
    """
    if cfg.noise_multiplier <= 0:
        return 0.0
    base = cfg.noise_multiplier * cfg.clip_norm
    if cfg.enable_masking and n_participants > 1:
        return base / math.sqrt(n_participants)
    return base


def privatise(update: StateDict, cfg: Config, n_participants: int = 1,
              generator: Optional[torch.Generator] = None) -> StateDict:
    """Clip, then noise. This is the body of `server.apply_privacy`."""
    clipped, _, _ = clip_update(update, cfg.privacy.clip_norm)
    return add_gaussian_noise(clipped, noise_std(cfg.privacy, n_participants),
                              generator)


# --------------------------------------------------------------------------
# Renyi DP accounting
# --------------------------------------------------------------------------

def rdp_gaussian(alpha: float, noise_multiplier: float, steps: int) -> float:
    """RDP of `steps` compositions of the Gaussian mechanism at order alpha.

    For sensitivity C and noise sigma*C, one release is (alpha, alpha/(2 sigma^2))
    -RDP. RDP composes by addition, so T rounds multiply it by T.
    """
    if noise_multiplier <= 0:
        return float("inf")
    return steps * alpha / (2.0 * noise_multiplier ** 2)


def rdp_to_dp(rdp: float, alpha: float, delta: float) -> float:
    """Convert an (alpha, rdp) guarantee to (epsilon, delta).

    Uses the tightened conversion of Balle et al. (2020) rather than the
    original Mironov bound, which is looser by roughly log(1/delta)/(alpha-1).
    """
    if not math.isfinite(rdp):
        return float("inf")
    if alpha <= 1.0:
        return float("inf")
    return (rdp
            + math.log1p(-1.0 / alpha)
            - (math.log(delta) + math.log(alpha)) / (alpha - 1.0))


def compute_epsilon(cfg: PrivacyConfig, rounds: int,
                    subsampled: bool = False) -> Dict[str, Any]:
    """Privacy budget spent over a full training run.

    Returns the best (smallest) epsilon over the searched Renyi orders, the
    order that achieved it, and an honest note about what the number does and
    does not account for.
    """
    if cfg.noise_multiplier <= 0:
        return {
            "epsilon": float("inf"),
            "delta": cfg.delta,
            "rounds": rounds,
            "noise_multiplier": cfg.noise_multiplier,
            "note": "no noise added; clipping alone gives no formal guarantee",
        }

    best_eps, best_alpha = float("inf"), float("nan")
    for alpha in cfg.dp_orders:
        eps = rdp_to_dp(rdp_gaussian(alpha, cfg.noise_multiplier, rounds),
                        alpha, cfg.delta)
        if eps < best_eps:
            best_eps, best_alpha = eps, alpha

    note = ("exact RDP for the Gaussian mechanism under full participation; "
            "update-level (not per-example) guarantee")
    if subsampled:
        note += ("; clients were subsampled but amplification is NOT credited, "
                 "so this epsilon is a valid upper bound and looser than the "
                 "true one")

    return {
        "epsilon": best_eps,
        "delta": cfg.delta,
        "best_order": best_alpha,
        "rounds": rounds,
        "noise_multiplier": cfg.noise_multiplier,
        "clip_norm": cfg.clip_norm,
        "note": note,
    }


def suggest_clip_norm(norms: Sequence[float], quantile: float = 0.7) -> float:
    """Pick a clipping bound from update norms actually observed.

    A magic constant is the wrong way to set C. Too loose and nothing is
    clipped, so the noise calibrated to C is larger than it needed to be. Too
    tight and every update is throttled, which is what happened here at C=1.0
    against a median update norm of 1.05: 80% of updates clipped every round,
    learning strangled, and the privacy-utility curve measuring the clipping
    rather than the noise.

    Clipping around the 70th percentile leaves most updates intact while
    still bounding the tail that sensitivity depends on.

    One honest caveat for the report: the norms used here come from real
    updates, so calibration leaks a little information of its own. A strict
    treatment spends a slice of the privacy budget estimating the quantile,
    or takes it from public data. At capstone scale it is enough to say so.
    """
    if not len(norms):
        raise ValueError("no update norms supplied")
    return float(np.quantile(np.asarray(norms, dtype=np.float64), quantile))


def clipping_report(norms: Sequence[float], clip_norm: float) -> Dict[str, Any]:
    """How a given C sits against the observed norms."""
    arr = np.asarray(norms, dtype=np.float64)
    return {
        "clip_norm": float(clip_norm),
        "n_updates": int(arr.size),
        "median_norm": float(np.median(arr)),
        "p70_norm": float(np.quantile(arr, 0.7)),
        "p95_norm": float(np.quantile(arr, 0.95)),
        "fraction_clipped": float((arr > clip_norm).mean()),
    }


def noise_for_target_epsilon(cfg: PrivacyConfig, rounds: int,
                             target_epsilon: float,
                             lo: float = 0.05, hi: float = 200.0,
                             iters: int = 60) -> float:
    """Smallest noise multiplier reaching `target_epsilon`, by bisection.

    Handy for the report: quoting "we ran at epsilon = 3" is easier to defend
    than "we ran at sigma = 1.4", because epsilon is the quantity that means
    something to a reader.
    """
    probe = PrivacyConfig(**{**cfg.__dict__})
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        probe.noise_multiplier = mid
        eps = compute_epsilon(probe, rounds)["epsilon"]
        if eps > target_epsilon:
            lo = mid
        else:
            hi = mid
    return hi


if __name__ == "__main__":  # smoke check
    from ..config import default_config
    from ..models.adapter import build_adapter, clone_state, state_norm

    cfg = default_config()
    torch.manual_seed(0)

    model = build_adapter(cfg.adapter)
    update = clone_state(model.state_dict())
    for k in update:
        update[k] = torch.randn_like(update[k]) * 0.3
    before = state_norm(update)
    print(f"raw update norm: {before:.4f}")

    # Clipping must bound the norm exactly.
    clipped, orig, did = clip_update(update, 1.0)
    after = state_norm(clipped)
    print(f"clipped to 1.0 : {after:.6f}  (was {orig:.4f}, clipped={did})")
    assert did and abs(after - 1.0) < 1e-4, "clipping did not bound the norm"

    # An update already under the bound must pass through untouched.
    small = {k: v * 0.001 for k, v in update.items()}
    same, _, did2 = clip_update(small, 1.0)
    assert not did2 and abs(state_norm(same) - state_norm(small)) < 1e-8

    # Noise must actually move the update, and grow with the multiplier.
    g = torch.Generator().manual_seed(0)
    noisy = add_gaussian_noise(clipped, 0.1, g)
    diff = state_norm({k: noisy[k] - clipped[k] for k in clipped})
    print(f"noise displacement at std=0.1: {diff:.4f}")
    assert diff > 0.0, "noise had no effect"

    # Masking splits the noise across participants.
    cfg.privacy.noise_multiplier = 1.0
    cfg.privacy.enable_masking = False
    local_std = noise_std(cfg.privacy, 5)
    cfg.privacy.enable_masking = True
    dist_std = noise_std(cfg.privacy, 5)
    print(f"noise std  local-DP={local_std:.4f}  distributed={dist_std:.4f}")
    assert abs(dist_std - local_std / math.sqrt(5)) < 1e-9

    # Accounting: epsilon must fall as noise rises, and rise with rounds.
    print()
    print("  sigma   rounds   epsilon (delta=1e-5)")
    prev = float("inf")
    for sigma in (0.5, 0.8, 1.2, 2.0, 4.0):
        cfg.privacy.noise_multiplier = sigma
        acc = compute_epsilon(cfg.privacy, rounds=30)
        print(f"  {sigma:<7.2f} {30:<8} {acc['epsilon']:.3f} "
              f"(order {acc['best_order']:g})")
        assert acc["epsilon"] < prev, "epsilon did not decrease with more noise"
        prev = acc["epsilon"]

    cfg.privacy.noise_multiplier = 1.0
    e30 = compute_epsilon(cfg.privacy, 30)["epsilon"]
    e60 = compute_epsilon(cfg.privacy, 60)["epsilon"]
    assert e60 > e30, "epsilon did not grow with more rounds"

    cfg.privacy.noise_multiplier = 0.0
    assert math.isinf(compute_epsilon(cfg.privacy, 30)["epsilon"]), \
        "zero noise must report infinite epsilon"

    # Inverse lookup round-trips.
    cfg.privacy.noise_multiplier = 1.0
    sigma3 = noise_for_target_epsilon(cfg.privacy, rounds=30, target_epsilon=3.0)
    probe = PrivacyConfig(**{**cfg.privacy.__dict__})
    probe.noise_multiplier = sigma3
    got = compute_epsilon(probe, 30)["epsilon"]
    print(f"\nsigma for epsilon=3 over 30 rounds: {sigma3:.4f} -> eps={got:.4f}")
    assert got <= 3.0 + 1e-3, "inverse lookup overshot the target"

    print("dp OK")
