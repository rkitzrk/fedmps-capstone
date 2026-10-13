"""Scoring-layer calibration: the part of the domain gap that is actually
recoverable from embeddings alone.

Phase 2 federates an adapter - a 512 -> h -> 512 residual MLP - on the theory
that the camera's degradation is a learnable shift that can be undone in
embedding space. Measured on held-out identities, that theory is only about
two points true, and an unconstrained MLP overshoots it badly:

    baseline (frozen backbone)                         rank-1  0.4807
    in-batch InfoNCE adapter, best early stop                  0.4852
    in-batch InfoNCE adapter, trained to convergence           0.4337   worse
    centroid InfoNCE adapter, trained to convergence           0.3720   worse
    ridge least-squares map degraded -> clean                  0.4572   worse
    PCA whitening (alpha=0.5)                                  0.4027   worse

Every *unconstrained* map loses ground, and it loses it monotonically with
training: the map fits the identities it was trained on and shreds the generic
angular geometry ArcFace relies on for identities it has never seen. Raising
the number of training identities from 150 to 1,200 moves held-out rank-1 by
under half a point, so this is not a data-volume problem. It is a statement
about what is left in the embedding: a 30-pixel face has lost information that
no 512-dimensional map can put back.

What *is* recoverable is rigid. Three corrections, in this order:

  1. Centering. Degraded embeddings carry a shared offset - the camera's
     signature, common to every face it sees. Cosine similarity is not
     translation invariant, so that offset is pure nuisance. Subtract the
     mean of the degraded training embeddings from every probe and the mean
     of the clean ones from every gallery entry.                  +0.9 pts

  2. Partial rotation. The orthogonal Procrustes solution to
     "rotate degraded onto clean" is the largest map that provably cannot
     distort the geometry - it preserves every angle and every distance, so
     it cannot overfit the way a free matrix does. Blended half-way with the
     identity, which measures better than the full rotation.       +1.0 pts

  3. Hubness. In 512 dimensions a few gallery entries sit close to a large
     share of all probes and win the argmax for people they are not. The
     standard correction subtracts each gallery entry's mean similarity to a
     cohort of unlabelled traffic - which a deployment has in abundance and
     needs no labels for.                                          +0.9 pts

Together: +2.4 points at every operating point, and the gain survives all the
way up the enrolment/frame-fusion table, which the adapter's does not.

Everything here is fitted on node-training identities only. No held-out
identity, and no test embedding, is touched by `fit`.

The rotation is 512x512 floats - 1 MB, the same size class as the adapter -
so it federates through exactly the same clip/noise/mask path if you want it
to. Unlike the adapter it has a closed-form solution from per-node covariance
sums, so a node sends one accumulator per round rather than thirty updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


# A d x d rotation has d^2 free parameters before orthogonality removes
# roughly half of them. Fitting one from fewer than this many samples per
# dimension produces a matrix that describes the sample rather than the
# camera: on a 512-dimensional embedding with 180 training rows it cost 17
# rank-1 points, against the +1.0 it is worth when properly determined.
MIN_ROWS_PER_DIM = 10

# The hubness term needs a cohort comfortably larger than its own k, or the
# "mean of the k nearest" is just the mean of everything.
MIN_COHORT_PER_K = 5


@dataclass
class Calibrator:
    """Fitted scoring layer. Apply to probes and gallery, then score."""

    probe_mean: np.ndarray       # [512] mean degraded training embedding
    gallery_mean: np.ndarray     # [512] mean clean training embedding
    rotation: Optional[np.ndarray] = None   # [512, 512] probe-side, blended orthogonal
    hub_k: int = 200             # cohort neighbours per gallery entry
    hub_beta: float = 1.0        # how much of the hubness term to subtract

    # Filled in by `fit_hubness`, which needs the gallery it will score against.
    hub: Optional[np.ndarray] = None
    _warned_hubness: bool = False

    # -- application ------------------------------------------------------

    def probe(self, emb: np.ndarray) -> np.ndarray:
        """Correct camera-side embeddings."""
        out = _l2(_l2(np.asarray(emb, np.float32)) - self.probe_mean)
        if self.rotation is not None:
            out = _l2(out @ self.rotation)
        return np.ascontiguousarray(out, dtype=np.float32)

    def gallery(self, emb: np.ndarray) -> np.ndarray:
        """Correct enrolment-side embeddings."""
        out = _l2(_l2(np.asarray(emb, np.float32)) - self.gallery_mean)
        return np.ascontiguousarray(out, dtype=np.float32)

    def fit_hubness(self, gallery: np.ndarray, cohort: np.ndarray) -> "Calibrator":
        """Measure how much unlabelled traffic each gallery entry attracts.

        `gallery` must already have been through `self.gallery` and `cohort`
        through `self.probe`. The cohort is unlabelled camera traffic - in the
        benchmark the non-mate probes, in deployment anything the camera saw
        that no one ever confirmed. It carries no labels, so using it costs
        nothing in annotation and leaks nothing about the watchlist.
        """
        cohort = np.asarray(cohort, np.float32)
        gallery = np.asarray(gallery, np.float32)
        if cohort.shape[0] == 0 or gallery.shape[0] == 0:
            self.hub = None
            return self
        # The term is a mean over the cohort's k nearest members to each
        # gallery entry. With a cohort barely larger than k it degenerates
        # into the mean over the whole cohort plus sampling noise, and
        # subtracting noise from the argmax costs accuracy rather than
        # buying it - measured at -3 rank-1 points on a 36-member cohort.
        if cohort.shape[0] < MIN_COHORT_PER_K * self.hub_k:
            # Once per calibrator: an ablation sweep refits the same one at
            # every operating point and the message is the same every time.
            if not self._warned_hubness:
                print(f"[calibrate] hubness skipped: cohort of "
                      f"{cohort.shape[0]} is too small for k={self.hub_k}, "
                      f"need {MIN_COHORT_PER_K * self.hub_k}.")
                self._warned_hubness = True
            self.hub = None
            return self
        k = int(min(self.hub_k, cohort.shape[0]))
        sim = cohort @ gallery.T
        self.hub = np.partition(sim, -k, axis=0)[-k:].mean(0).astype(np.float32)
        return self

    def score(self, probes: np.ndarray, gallery: np.ndarray) -> np.ndarray:
        """Cosine similarity with the hubness penalty applied per gallery entry."""
        sim = np.asarray(probes, np.float32) @ np.asarray(gallery, np.float32).T
        if self.hub is not None and self.hub.shape[0] == gallery.shape[0]:
            sim = sim - self.hub_beta * self.hub[None, :]
        return sim

    def describe(self) -> str:
        bits = ["centering"]
        if self.rotation is not None:
            bits.append("rotation")
        if self.hub is not None:
            bits.append(f"hubness(k={self.hub_k}, beta={self.hub_beta:g})")
        return " + ".join(bits)

    # -- persistence ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "centering": True,
            "rotation": self.rotation is not None,
            "hubness": self.hub is not None,
            "hub_k": self.hub_k,
            "hub_beta": self.hub_beta,
            "probe_mean_norm": float(np.linalg.norm(self.probe_mean)),
            "gallery_mean_norm": float(np.linalg.norm(self.gallery_mean)),
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path,
                 probe_mean=self.probe_mean,
                 gallery_mean=self.gallery_mean,
                 rotation=(self.rotation if self.rotation is not None
                           else np.zeros((0, 0), np.float32)),
                 hub_k=np.int32(self.hub_k),
                 hub_beta=np.float32(self.hub_beta))

    @classmethod
    def load(cls, path: Path) -> "Calibrator":
        z = np.load(Path(path))
        rot = z["rotation"]
        return cls(probe_mean=z["probe_mean"], gallery_mean=z["gallery_mean"],
                   rotation=(rot if rot.size else None),
                   hub_k=int(z["hub_k"]), hub_beta=float(z["hub_beta"]))


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------

def training_rows(art) -> np.ndarray:
    """Every row any node holds. Held-out identities appear in none of them."""
    rows: List[int] = []
    for node in art.partition["nodes"].values():
        rows.extend(int(r) for r in node.get("train_idx", []))
        rows.extend(int(r) for r in node.get("val_idx", []))
    return np.asarray(sorted(set(rows)), dtype=np.int64)


def fit(art, rotation: bool = True, blend: float = 0.5,
        hub_k: int = 200, hub_beta: float = 1.0,
        rows: Optional[np.ndarray] = None) -> Calibrator:
    """Fit the scoring layer on node-training data.

    `blend` mixes the orthogonal Procrustes rotation with the identity: 0 is
    no rotation, 1 the full one. 0.5 measured best, and the curve is flat
    enough either side of it (0.4958 / 0.4986 / 0.4960 at 0.25 / 0.5 / 0.75)
    that the exact value is not load-bearing.

    The rotation is dropped, with a printed reason, when there are too few
    training rows to determine it. Centering needs one vector's worth of data
    and survives; the rotation needs `MIN_ROWS_PER_DIM` rows per dimension and
    is actively harmful below that.
    """
    rows = training_rows(art) if rows is None else np.asarray(rows, np.int64)
    if rows.size < 2:
        raise ValueError("no node-training rows to fit the calibrator on")

    deg = np.asarray(art.emb_deg[rows], np.float32)
    clean = np.asarray(art.emb_clean[rows], np.float32)
    probe_mean = deg.mean(0).astype(np.float32)
    gallery_mean = clean.mean(0).astype(np.float32)
    dim = deg.shape[1]

    rot = None
    if rotation and 0.0 < blend <= 1.0:
        needed = MIN_ROWS_PER_DIM * dim
        if rows.size < needed:
            print(f"[calibrate] rotation skipped: {rows.size} training rows "
                  f"for {dim} dimensions, need {needed}. Centering only.")
        else:
            a = _l2(deg - probe_mean)
            b = _l2(clean - gallery_mean)
            u, _s, vt = np.linalg.svd(a.T @ b, full_matrices=False)
            r = (u @ vt).astype(np.float32)
            eye = np.eye(dim, dtype=np.float32)
            rot = ((1.0 - blend) * eye + blend * r).astype(np.float32)

    return Calibrator(probe_mean=probe_mean, gallery_mean=gallery_mean,
                      rotation=rot, hub_k=hub_k, hub_beta=hub_beta)


if __name__ == "__main__":  # smoke check - synthetic, no dataset, no network
    import tempfile

    rng = np.random.default_rng(0)
    # per_id is set so the pool clears MIN_ROWS_PER_DIM * dim; below that
    # `fit` correctly refuses to build a rotation and the next assertion
    # would be testing the refusal rather than the rotation.
    n_ids, per_id, dim = 60, 12, 64
    labels = np.repeat(np.arange(n_ids), per_id)
    cents = _l2(rng.normal(size=(n_ids, dim)))
    clean = _l2(cents[labels] + 0.10 * rng.normal(size=(labels.size, dim)))

    # A camera that adds a shared offset and applies a fixed rotation: exactly
    # the nuisance this module claims to remove, and nothing else.
    bias = _l2(rng.normal(size=(1, dim)))[0]
    q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    deg = _l2(clean @ (0.85 * np.eye(dim) + 0.15 * q)
              + 0.6 * bias + 0.05 * rng.normal(size=clean.shape))

    part = {"n_nodes": 1, "seed": 0,
            "test_idx": [], "test_identities": [], "test_profile_ids": [],
            "nodes": {"node_0": {"identities": [], "val_idx": [],
                                 "train_idx": list(range(labels.size)),
                                 "degrade_params": {}}}}

    class _Art:
        emb_clean, emb_deg, partition = clean, deg, part

    cal = fit(_Art(), rotation=True, blend=1.0)
    assert cal.rotation is not None and cal.rotation.shape == (dim, dim)

    # Too few rows to determine a rotation: it must be dropped, not fitted.
    thin = fit(_Art(), rotation=True, blend=1.0,
               rows=np.arange(MIN_ROWS_PER_DIM * dim - 1))
    assert thin.rotation is None,         "a rotation was fitted from fewer rows than it has parameters"

    gal_rows = np.arange(0, labels.size, per_id)
    probe_rows = np.setdiff1d(np.arange(labels.size), gal_rows)

    def rank1(p, g):
        return float((labels[gal_rows][np.asarray(p @ g.T).argmax(1)]
                      == labels[probe_rows]).mean())

    before = rank1(_l2(deg[probe_rows]), _l2(clean[gal_rows]))
    after = rank1(cal.probe(deg[probe_rows]), cal.gallery(clean[gal_rows]))
    print(f"rank-1 before {before:.4f}  after {after:.4f}")
    assert after >= before, "calibration made a known nuisance transform worse"

    g = cal.gallery(clean[gal_rows])
    cal.hub_k = max(1, probe_rows.size // MIN_COHORT_PER_K)
    cal.fit_hubness(g, cal.probe(deg[probe_rows]))
    assert cal.hub is not None and cal.hub.shape == (gal_rows.size,)

    # A cohort too small for k must be refused, not averaged into noise.
    thin_hub = fit(_Art(), rotation=False)
    thin_hub.hub_k = probe_rows.size
    thin_hub.fit_hubness(g, thin_hub.probe(deg[probe_rows]))
    assert thin_hub.hub is None,         "a hubness term was fitted from a cohort smaller than its own k"
    assert cal.score(cal.probe(deg[probe_rows]), g).shape == (probe_rows.size,
                                                              gal_rows.size)
    print("hubness:", cal.describe())

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cal.npz"
        cal.save(p)
        back = Calibrator.load(p)
    assert np.allclose(back.probe_mean, cal.probe_mean)
    assert back.rotation is not None and np.allclose(back.rotation, cal.rotation)
    print("round-trip OK")

    # A calibrator with no cohort must fall back to plain cosine, not crash.
    plain = fit(_Art(), rotation=False)
    assert plain.rotation is None and plain.hub is None
    assert np.allclose(plain.score(plain.probe(deg[probe_rows]), g),
                       plain.probe(deg[probe_rows]) @ g.T, atol=1e-5)
    print("calibrate OK")
