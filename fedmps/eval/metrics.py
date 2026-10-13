"""Evaluation, done the way an operator would read it.

The literature this project builds on reports balanced-set accuracy: 97.50%
in [1], 99.75% in [2], 94.68% in [4]. Those numbers say nothing about whether
a system is usable, because the real task is open-set: almost everyone who
walks past the camera is in no watchlist at all, and the system has to stay
quiet for all of them.

So the primary metric here is **TPIR at a fixed FPIR**:

  FPIR - false positive identification rate. Of the people who are in no
         watchlist, what fraction still triggered an alert?
  TPIR - true positive identification rate. Of the people who ARE on the
         watchlist, what fraction were both flagged and correctly named?

Closed-set rank-1 and an LFW-style verification number are also computed, the
first because it is intuitive and the second so there is one figure directly
comparable with [2] and [6].

Evaluation sets are built only from identities held out of every node, so no
result here is contaminated by anything any node trained on.

One asymmetry worth noting: the adapter is applied to **probes only**, never
to the gallery. The gallery is a clean enrolment photograph and needs no
correction; the adapter's entire job is to move a degraded camera embedding
toward where that clean one already sits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..config import Config
from ..contracts import Artefacts, assert_embeddings, l2_normalise
from .calibrate import Calibrator

Transform = Optional[Callable[[np.ndarray], np.ndarray]]


# --------------------------------------------------------------------------
# Evaluation sets
# --------------------------------------------------------------------------

@dataclass
class EvalSets:
    """Fixed once per run so every model is scored on identical data."""

    gallery_emb: np.ndarray      # [G, 512] clean enrolment photos
    gallery_ids: np.ndarray      # [G] identity id

    mate_emb: np.ndarray         # [M, 512] degraded, person IS on the watchlist
    mate_ids: np.ndarray         # [M]
    mate_profiles: np.ndarray    # [M] which camera degraded it

    nonmate_emb: np.ndarray      # [K, 512] degraded, person is NOT enrolled
    nonmate_profiles: np.ndarray  # [K]

    ver_clean: np.ndarray        # [P, 512] verification pair, gallery side
    ver_degraded: np.ndarray     # [P, 512] verification pair, camera side
    ver_same: np.ndarray         # [P] bool, same identity?

    # The operating point these sets were built at. Carried on the object so
    # a result can never be quoted without the protocol that produced it.
    n_enrol: int = 1
    n_probe_frames: int = 1

    def describe(self) -> str:
        return (f"gallery {len(self.gallery_ids)} identities "
                f"x {self.n_enrol} enrolment photo(s) | "
                f"{len(self.mate_ids)} mate probes "
                f"x {self.n_probe_frames} frame(s) | "
                f"{len(self.nonmate_emb)} non-mate probes | "
                f"{len(self.ver_same)} verification pairs")


def build_eval_sets(art: Artefacts, cfg: Config,
                    probe_source: str = "degraded",
                    n_enrol: Optional[int] = None,
                    n_probe_frames: Optional[int] = None) -> EvalSets:
    """Construct the held-out evaluation protocol.

    Held-out identities are split in two. One half is enrolled in the gallery
    - these are the "missing persons". The other half never appears in the
    gallery at all and supplies the non-mate probes: the crowd. Without them
    there is no FPIR to measure and no open-set evaluation, only a closed-set
    one that flatters the system.

    `probe_source` selects which embeddings the probes are drawn from.
    "degraded" is the real task. "clean" answers a different question - how
    would the frozen backbone do if the camera were perfect? - and the gap
    between the two is the problem statement the whole project addresses.
    The identity splits are seeded identically either way, so the two are
    directly comparable.

    `n_enrol` photographs are averaged into one gallery entry and
    `n_probe_frames` consecutive camera frames into one probe; both default to
    `cfg.eval`. Neither is a concession - the deployed pipeline already votes
    over a track, and a case file already carries several photographs - but
    both change the number materially, so they are recorded in the results
    JSON and swept in full by `experiments.operating_points`.

    The identity split itself is drawn before either is applied, from the same
    seeded stream as before, so every operating point scores the same
    identities in the same gallery and the rows stay comparable.
    """
    if probe_source not in ("degraded", "clean"):
        raise ValueError(f"probe_source must be 'degraded' or 'clean', "
                         f"got {probe_source!r}")
    n_enrol = int(cfg.eval.n_enrol if n_enrol is None else n_enrol)
    n_probe_frames = int(cfg.eval.n_probe_frames if n_probe_frames is None
                         else n_probe_frames)
    if n_enrol < 1 or n_probe_frames < 1:
        raise ValueError(f"n_enrol and n_probe_frames must be >= 1, got "
                         f"{n_enrol} and {n_probe_frames}")
    probe_emb = art.emb_deg if probe_source == "degraded" else art.emb_clean
    rng = np.random.default_rng(cfg.seed + 991)

    test_idx = np.asarray(art.partition["test_idx"], dtype=np.int64)
    profiles = np.asarray(art.partition["test_profile_ids"], dtype=np.int32)
    test_ids = np.asarray(art.partition["test_identities"], dtype=np.int64)

    if test_idx.size == 0:
        raise ValueError("no held-out images; check partition.test_identity_frac")

    shuffled = test_ids.copy()
    rng.shuffle(shuffled)
    n_gal = int(round(cfg.eval.gallery_frac * shuffled.size))
    n_gal = int(np.clip(n_gal, 1, shuffled.size - 1))
    gal_ids = set(int(i) for i in shuffled[:n_gal])
    imp_ids = set(int(i) for i in shuffled[n_gal:])

    labels = art.labels
    prof_of = {int(i): int(p) for i, p in zip(test_idx, profiles)}

    gallery_vecs: List[np.ndarray] = []
    gallery_id_list: List[int] = []
    mate_vecs: List[np.ndarray] = []
    mate_id_list: List[int] = []
    mate_prof_list: List[int] = []
    nonmate_rows: List[int] = []
    skipped = 0

    for ident in sorted(gal_ids):
        rows = test_idx[labels[test_idx] == ident]
        if rows.size < n_enrol + n_probe_frames:
            # Not enough images to enrol this identity AND still probe it.
            # Dropping it is the only honest option: enrolling someone with
            # no probe inflates the gallery without testing anything, and
            # reusing an enrolment photo as its own probe is leakage.
            skipped += 1
            continue
        rows = rows[rng.permutation(rows.size)]
        # The first n_enrol images are the case file; the rest is camera
        # traffic, cut into non-overlapping tracks of n_probe_frames.
        gallery_vecs.append(art.emb_clean[rows[:n_enrol]].mean(axis=0))
        gallery_id_list.append(int(ident))
        rest = rows[n_enrol:]
        for s in range(0, rest.size - n_probe_frames + 1, n_probe_frames):
            track = rest[s:s + n_probe_frames]
            mate_vecs.append(probe_emb[track].mean(axis=0))
            mate_id_list.append(int(ident))
            mate_prof_list.append(prof_of[int(track[0])])

    for ident in sorted(imp_ids):
        nonmate_rows.extend(int(r) for r in test_idx[labels[test_idx] == ident])

    if not gallery_vecs or not mate_vecs:
        raise ValueError(
            f"evaluation set is empty; with n_enrol={n_enrol} and "
            f"n_probe_frames={n_probe_frames} every held-out identity needs "
            f"at least {n_enrol + n_probe_frames} images"
        )
    if not nonmate_rows:
        raise ValueError(
            "no non-mate probes; lower eval.gallery_frac so some identities "
            "stay out of the gallery"
        )
    if skipped:
        print(f"[eval] {skipped} of {len(gal_ids)} gallery identities dropped: "
              f"fewer than {n_enrol + n_probe_frames} images")

    # Non-mates are fused the same way as mates, or the two populations sit on
    # different noise scales and every threshold read off the DET curve is
    # wrong. Tracks are cut within an identity so a fused non-mate is still
    # one person, exactly as a fused mate is.
    nm_vecs: List[np.ndarray] = []
    nm_prof: List[int] = []
    for ident in sorted(imp_ids):
        rows = test_idx[labels[test_idx] == ident]
        for s in range(0, rows.size - n_probe_frames + 1, n_probe_frames):
            track = rows[s:s + n_probe_frames]
            nm_vecs.append(probe_emb[track].mean(axis=0))
            nm_prof.append(prof_of[int(track[0])])

    extra = art.emb_nonmate
    if extra is not None and extra.shape[0]:
        # The extra crowd has no identity labels, so it cannot be cut into
        # per-person tracks. Fuse consecutive rows anyway - they share a
        # camera profile - to keep it on the same scale as everything else.
        _ex_prof_raw = art.meta.get("nonmate_profiles", [])
        if not _ex_prof_raw or len(_ex_prof_raw) != extra.shape[0]:
            _ex_prof_raw = [0] * extra.shape[0]
        ex_prof = np.asarray(_ex_prof_raw, dtype=np.int32)
        usable = (extra.shape[0] // n_probe_frames) * n_probe_frames
        if usable:
            fused = extra[:usable].reshape(-1, n_probe_frames,
                                           extra.shape[1]).mean(axis=1)
            nm_vecs.extend(fused)
            nm_prof.extend(ex_prof[:usable:n_probe_frames].tolist())

    if not nm_vecs:
        raise ValueError(
            f"no non-mate probes survive n_probe_frames={n_probe_frames}; "
            f"lower it or enlarge the non-mate pool"
        )

    k_emb = l2_normalise(np.asarray(nm_vecs, dtype=np.float32))
    k_prof = np.asarray(nm_prof, dtype=np.int32)

    ver_clean, ver_deg, ver_same = _sample_verification_pairs(
        art, test_idx, rng, cfg.eval.n_verification_pairs, probe_emb)

    return EvalSets(
        gallery_emb=l2_normalise(np.asarray(gallery_vecs, dtype=np.float32)),
        gallery_ids=np.asarray(gallery_id_list, dtype=np.int64),
        mate_emb=l2_normalise(np.asarray(mate_vecs, dtype=np.float32)),
        mate_ids=np.asarray(mate_id_list, dtype=np.int64),
        mate_profiles=np.asarray(mate_prof_list, dtype=np.int32),
        nonmate_emb=k_emb,
        nonmate_profiles=k_prof,
        ver_clean=ver_clean,
        ver_degraded=ver_deg,
        ver_same=ver_same,
        n_enrol=n_enrol,
        n_probe_frames=n_probe_frames,
    )


def _sample_verification_pairs(art: Artefacts, test_idx: np.ndarray,
                               rng: np.random.Generator,
                               n_pairs: int,
                               probe_emb: Optional[np.ndarray] = None
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clean-vs-degraded pairs, half genuine and half impostor.

    This mirrors the deployment comparison (enrolment photo against camera
    frame) rather than the clean-vs-clean pairing LFW normally uses, so the
    number is comparable in form but harder in substance. Say so in the report.
    """
    labels = art.labels
    probe_emb = art.emb_deg if probe_emb is None else probe_emb
    by_id: Dict[int, np.ndarray] = {}
    for ident in np.unique(labels[test_idx]):
        rows = test_idx[labels[test_idx] == ident]
        if rows.size >= 2:
            by_id[int(ident)] = rows
    ids = sorted(by_id)
    if len(ids) < 2:
        empty = np.zeros((0, art.emb_clean.shape[1]), dtype=np.float32)
        return empty, empty, np.zeros(0, dtype=bool)

    half = max(1, n_pairs // 2)
    a_rows: List[int] = []
    b_rows: List[int] = []
    same: List[bool] = []

    for _ in range(half):
        ident = ids[int(rng.integers(len(ids)))]
        r1, r2 = rng.choice(by_id[ident], size=2, replace=False)
        a_rows.append(int(r1))
        b_rows.append(int(r2))
        same.append(True)

    for _ in range(half):
        i1, i2 = rng.choice(len(ids), size=2, replace=False)
        r1 = int(rng.choice(by_id[ids[i1]]))
        r2 = int(rng.choice(by_id[ids[i2]]))
        a_rows.append(r1)
        b_rows.append(r2)
        same.append(False)

    return (art.emb_clean[np.asarray(a_rows)],
            probe_emb[np.asarray(b_rows)],
            np.asarray(same, dtype=bool))


# --------------------------------------------------------------------------
# Curve helpers
# --------------------------------------------------------------------------

def _roc(scores: np.ndarray, positive: np.ndarray
         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (fpr, tpr, thresholds), descending by score."""
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    y = positive[order].astype(np.float64)
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    n_pos = max(float(y.sum()), 1.0)
    n_neg = max(float((1.0 - y).sum()), 1.0)
    return fp / n_neg, tp / n_pos, s


def _auc(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(_trapz(y, x))


def _eer(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Equal error rate: where FPR crosses FNR."""
    fnr = 1.0 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2.0)


def _rate_at_target(fpr: np.ndarray, tpr: np.ndarray, target: float,
                    n_neg: int) -> Tuple[float, bool]:
    """TPR at the largest threshold whose FPR does not exceed `target`.

    Also returns whether the target is measurable at all. With `n_neg`
    negatives the finest FPR that can be resolved is 1/n_neg; asking for
    1e-4 with 500 negatives is asking for a number the data cannot support,
    and reporting one anyway would be dishonest.
    """
    if n_neg <= 0:
        return float("nan"), False
    measurable = target >= (1.0 / n_neg)
    ok = np.where(fpr <= target)[0]
    if ok.size == 0:
        return 0.0, measurable
    return float(tpr[ok[-1]]), measurable


# --------------------------------------------------------------------------
# The metrics
# --------------------------------------------------------------------------

def cosine_scores(probes: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Default scoring: cosine similarity, since everything is L2-normalised."""
    return probes @ gallery.T


def open_set_identification(gallery_emb: np.ndarray, gallery_ids: np.ndarray,
                            mate_emb: np.ndarray, mate_ids: np.ndarray,
                            nonmate_emb: np.ndarray,
                            fpir_points: List[float],
                            score_fn: Optional[Callable[
                                [np.ndarray, np.ndarray], np.ndarray]] = None
                            ) -> Dict[str, Any]:
    """The primary metric. Scored against the whole gallery.

    A mate probe counts as a true positive only if it both crossed the
    threshold AND the top gallery hit was the right person. Flagging the
    correct alert against the wrong name is not a success.

    `score_fn` defaults to cosine similarity. Phase 3 passes a Hamming-based
    one so that cancelable templates are evaluated through exactly the same
    protocol as raw embeddings, which is what makes the accuracy cost of
    template protection a like-for-like number.
    """
    if gallery_emb.shape[0] == 0 or mate_emb.shape[0] == 0:
        return {"error": "empty gallery or mate set"}

    score_fn = score_fn or cosine_scores
    mate_sim = score_fn(mate_emb, gallery_emb)   # [M, G]
    mate_best = mate_sim.max(axis=1)
    mate_pred = gallery_ids[mate_sim.argmax(axis=1)]
    correct = mate_pred == mate_ids

    rank1 = float(correct.mean())

    if nonmate_emb.shape[0] == 0:
        return {"rank1": rank1, "error": "no non-mate probes"}

    nonmate_best = score_fn(nonmate_emb, gallery_emb).max(axis=1)

    # A mate that matched the wrong identity is not a true positive at any
    # threshold, so its score is pushed below every threshold.
    mate_score = np.where(correct, mate_best, -np.inf)

    scores = np.concatenate([mate_score, nonmate_best])
    positive = np.concatenate([
        np.ones(mate_score.size, dtype=bool),
        np.zeros(nonmate_best.size, dtype=bool),
    ])
    fpir, tpir, thresh = _roc(scores, positive)

    n_neg = int(nonmate_best.size)
    at: Dict[str, Any] = {}
    for f in fpir_points:
        value, measurable = _rate_at_target(fpir, tpir, f, n_neg)
        at[f"tpir@fpir={f:g}"] = value if measurable else float("nan")
        if not measurable:
            at[f"tpir@fpir={f:g}_note"] = (
                f"unmeasurable: {n_neg} non-mate probes resolve no finer "
                f"than {1.0 / n_neg:.2g}"
            )

    # Sub-sample the curve so results JSON stays readable.
    step = max(1, fpir.size // 400)
    return {
        "rank1": rank1,
        **at,
        "n_gallery": int(gallery_emb.shape[0]),
        "n_mate": int(mate_emb.shape[0]),
        "n_nonmate": n_neg,
        "mean_mate_score": float(mate_best.mean()),
        "mean_nonmate_score": float(nonmate_best.mean()),
        "det_curve": {
            "fpir": fpir[::step].tolist(),
            "tpir": tpir[::step].tolist(),
            "threshold": np.where(np.isfinite(thresh[::step]),
                                  thresh[::step], -1.0).tolist(),
        },
    }


def verification(clean: np.ndarray, degraded: np.ndarray, same: np.ndarray,
                 far_points: List[float]) -> Dict[str, Any]:
    """LFW-style 1:1 verification, for one number comparable with [2] and [6]."""
    if same.size == 0:
        return {"error": "no verification pairs"}

    scores = np.sum(clean * degraded, axis=1)
    fpr, tpr, _ = _roc(scores, same)
    n_neg = int((~same).sum())

    out: Dict[str, Any] = {
        "auc": _auc(fpr, tpr),
        "eer": _eer(fpr, tpr),
        "n_pairs": int(same.size),
        "accuracy_best_threshold": float(np.max(
            (tpr * same.sum() + (1.0 - fpr) * n_neg) / max(same.size, 1)
        )),
    }
    for f in far_points:
        value, measurable = _rate_at_target(fpr, tpr, f, n_neg)
        out[f"tar@far={f:g}"] = value if measurable else float("nan")
    return out


def evaluate(sets: EvalSets, cfg: Config,
             transform: Transform = None,
             per_camera: bool = False,
             calibrator: Optional[Calibrator] = None) -> Dict[str, Any]:
    """Score one model. `transform` is the adapter, or None for the baseline.

    The adapter is applied to probes only - see the module docstring. The
    calibrator is applied to both sides, because centering the probes without
    centering the gallery leaves the two on different origins and makes cosine
    similarity meaningless. It runs after the adapter, so the two compose and
    can be ablated independently.

    `calibrator` defaults to None. `run_calibrated` is the entry point that
    fits one from node-training data and threads it through everything.
    """
    def tf(x: np.ndarray) -> np.ndarray:
        if x.shape[0] == 0:
            return x
        out = x
        if transform is not None:
            out = np.ascontiguousarray(transform(out), dtype=np.float32)
            out = assert_embeddings(out, "adapter output")
        if calibrator is not None:
            out = calibrator.probe(out)
        return out

    mate = tf(sets.mate_emb)
    nonmate = tf(sets.nonmate_emb)
    ver_deg = tf(sets.ver_degraded)

    gallery = sets.gallery_emb
    ver_clean = sets.ver_clean
    score_fn = None
    if calibrator is not None:
        gallery = calibrator.gallery(gallery)
        ver_clean = calibrator.gallery(ver_clean)
        calibrator.fit_hubness(gallery, nonmate)
        score_fn = calibrator.score

    result: Dict[str, Any] = {
        "identification": open_set_identification(
            gallery, sets.gallery_ids,
            mate, sets.mate_ids, nonmate,
            cfg.eval.fpir_points, score_fn=score_fn),
        "verification": verification(
            ver_clean, ver_deg, sets.ver_same, cfg.eval.far_points),
        "protocol": {
            "n_enrol": sets.n_enrol,
            "n_probe_frames": sets.n_probe_frames,
            "calibrated": calibrator.describe() if calibrator else "none",
        },
    }

    if per_camera:
        per: Dict[str, Any] = {}
        for k in sorted(set(sets.mate_profiles.tolist())):
            mm = sets.mate_profiles == k
            nn = sets.nonmate_profiles == k
            if mm.sum() == 0 or nn.sum() == 0:
                continue
            per[f"node_{k}"] = open_set_identification(
                gallery, sets.gallery_ids,
                mate[mm], sets.mate_ids[mm], nonmate[nn],
                cfg.eval.fpir_points, score_fn=score_fn)
            per[f"node_{k}"].pop("det_curve", None)
        result["per_camera"] = per

    return result


def fit_calibrator(art: Artefacts, cfg: Config) -> Optional[Calibrator]:
    """The calibrator `cfg.eval` asks for, or None if it is switched off."""
    if not cfg.eval.calibrate:
        return None
    from . import calibrate as _cal
    return _cal.fit(art, rotation=cfg.eval.calibrate_blend > 0.0,
                    blend=cfg.eval.calibrate_blend,
                    hub_k=cfg.eval.hub_k, hub_beta=cfg.eval.hub_beta)


def headline(result: Dict[str, Any]) -> str:
    """One-line summary for progress logs."""
    ident = result.get("identification", {})
    ver = result.get("verification", {})
    bits = [f"rank1={ident.get('rank1', float('nan')):.4f}"]
    for key in sorted(k for k in ident if k.startswith("tpir@")
                      and not k.endswith("_note")):
        bits.append(f"{key}={ident[key]:.4f}")
    if "auc" in ver:
        bits.append(f"ver_auc={ver['auc']:.4f}")
    return "  ".join(bits)


if __name__ == "__main__":  # smoke check
    from ..config import default_config
    from ..contracts import l2_normalise

    cfg = default_config()
    cfg.eval.n_verification_pairs = 400
    # The single-shot point, so the oracle assertion below has something to
    # bite on. Track fusion saturates this synthetic set at rank-1 = 1.0.
    cfg.eval.n_enrol = 1
    cfg.eval.n_probe_frames = 1
    rng = np.random.default_rng(0)

    # Synthetic artefacts: identity centroids spread across the sphere so
    # all pairs are roughly orthogonal (easy baseline).  A shared bias vector
    # is added to every degraded probe.  Because the bias is the same for
    # every probe the rank-1 ordering is preserved, but the 1-vs-1
    # verification AUC drops — the oracle that removes the bias must restore it.
    n_ids, per_id, dim = 40, 6, 512
    labels = np.repeat(np.arange(n_ids), per_id).astype(np.int32)
    cents = l2_normalise(rng.normal(size=(n_ids, dim)))
    scale = 3.0 / np.sqrt(dim)
    clean = l2_normalise(cents[labels] + 0.15 * scale * rng.normal(size=(labels.size, dim)))

    bias = l2_normalise(rng.normal(size=(1, dim)))[0]
    deg = l2_normalise(clean + 0.8 * bias + 0.15 * scale * rng.normal(size=clean.shape))

    part = {
        "n_nodes": 2, "seed": 0,
        "test_idx": list(range(labels.size)),
        "test_identities": list(range(n_ids)),
        "test_profile_ids": [i % 2 for i in range(labels.size)],
        "nodes": {
            "node_0": {"identities": [], "train_idx": [], "val_idx": [],
                       "degrade_params": {}},
            "node_1": {"identities": [], "train_idx": [], "val_idx": [],
                       "degrade_params": {}},
        },
    }
    art = Artefacts(clean, deg, labels, part, {"backbone": "synthetic"})

    sets = build_eval_sets(art, cfg)
    print(sets.describe())

    base = evaluate(sets, cfg, transform=None)
    print("no adapter :", headline(base))

    # An oracle that removes the known bias should restore verification AUC.
    oracle = evaluate(sets, cfg,
                      transform=lambda x: l2_normalise(x - 0.8 * bias))
    print("oracle     :", headline(oracle))

    assert oracle["verification"]["auc"] >= base["verification"]["auc"], \
        "oracle correction did not improve verification AUC; scoring is wrong"
    assert 0.0 <= base["identification"]["rank1"] <= 1.0
    assert base["identification"]["n_nonmate"] > 0

    # FPIR finer than the non-mate count allows must come back as NaN.
    tiny = open_set_identification(
        sets.gallery_emb, sets.gallery_ids,
        sets.mate_emb, sets.mate_ids, sets.nonmate_emb[:20], [1e-4])
    assert np.isnan(tiny["tpir@fpir=0.0001"]), \
        "an unmeasurable FPIR was reported as a real number"
    print("unmeasurable-FPIR guard OK")

    per = evaluate(sets, cfg, transform=None, per_camera=True)
    assert "per_camera" in per and len(per["per_camera"]) == 2

    # Track fusion and deeper enrolment must both move rank-1 the right way.
    fused = build_eval_sets(art, cfg, n_enrol=2, n_probe_frames=2)
    assert fused.n_enrol == 2 and fused.n_probe_frames == 2
    assert fused.mate_emb.shape[0] < sets.mate_emb.shape[0], \
        "fusing frames did not reduce the probe count; tracks are not being cut"
    fused_r1 = evaluate(fused, cfg)["identification"]["rank1"]
    assert fused_r1 >= base["identification"]["rank1"], \
        f"fusion made rank-1 worse: {fused_r1} < {base['identification']['rank1']}"
    print(f"fusion     : {fused.describe()}")

    # The calibrator removes exactly the bias this synthetic set injects, so
    # it must not lose ground against plain cosine.
    cfg.eval.calibrate = True
    part["nodes"]["node_0"]["train_idx"] = list(range(0, labels.size, 2))
    cal = fit_calibrator(art, cfg)
    assert cal is not None
    cal_r1 = evaluate(sets, cfg, calibrator=cal)["identification"]["rank1"]
    assert cal_r1 >= base["identification"]["rank1"] - 1e-9, \
        f"calibration made rank-1 worse: {cal_r1} < {base['identification']['rank1']}"
    print(f"calibrated : rank1={cal_r1:.4f} via {cal.describe()}")

    cfg.eval.calibrate = False
    assert fit_calibrator(art, cfg) is None, "calibrate=False still built one"

    print("metrics OK")

