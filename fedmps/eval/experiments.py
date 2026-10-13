"""Every experiment in the report, and the figures they produce.

Each function returns a plain dict that goes straight into results/*.json, so
the report is written from recorded numbers rather than from whatever was last
printed to a terminal.

The set:

  degradation   clean vs degraded probes, no adapter. The problem statement.
  operating_points
                rank-1 across enrolment depth, track length and the scoring
                layer. The single-photo, single-frame cell the project used
                to quote is the worst point on this surface by about forty
                rank-1 points; the table is what should be reported.
  three_way     local-only vs federated vs centralised. The headline.
  dp_sweep      accuracy against noise, with the epsilon each setting buys.
  masking       secure aggregation: correctness, overhead, and the preview
                screenshot showing an individual update is unreadable.
  templates     accuracy cost of cancelable protection, swept over bit width.
  fedprox       FedAvg against FedProx under the non-IID split.
  mia           membership inference. Does the model leak who trained it?

Runtime warning: dp_sweep and fedprox each retrain the federation once per
setting. Use `rounds_override` to shorten them while you are still checking
that things work.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from ..config import Config
from ..contracts import Artefacts, save_json
from ..fl.loop import run_federated, run_three_way, _transform_of
from ..fl.node import make_node_data
from ..fl.server import Server
from ..models.adapter import build_adapter, clone_state
from ..privacy.dp import (
    clipping_report, compute_epsilon, suggest_clip_norm,
)
from ..privacy.mask import SecureAggregator
from ..privacy.template import (
    binarise, derive_key, fit_selector, protect, similarity_matrix,
    template_report,
)
from .metrics import (
    EvalSets, _auc, _roc, build_eval_sets, evaluate, fit_calibrator, headline,
    open_set_identification,
)


def log(msg: str) -> None:
    print(f"[exp] {msg}", flush=True)


def _cfg_copy(cfg: Config, **overrides: Any) -> Config:
    """A deep copy with dotted overrides, e.g. `privacy__noise_multiplier=1.0`."""
    c = copy.deepcopy(cfg)
    for key, value in overrides.items():
        section, _, field = key.partition("__")
        if field:
            setattr(getattr(c, section), field, value)
        else:
            setattr(c, section, value)
    return c


def _rank1(result: Dict[str, Any]) -> float:
    return float(result.get("identification", {}).get("rank1", float("nan")))


# --------------------------------------------------------------------------
# 1. Degradation - the problem statement
# --------------------------------------------------------------------------

def exp_degradation(art: Artefacts, cfg: Config) -> Dict[str, Any]:
    """How much does surveillance degradation cost a state-of-the-art model?

    Same gallery, same identities, same protocol - only the probe quality
    changes. If this gap is small, the degradation model is too gentle and
    nothing downstream is worth reporting.
    """
    log("degradation: clean vs degraded probes, frozen backbone alone")
    clean_sets = build_eval_sets(art, cfg, probe_source="clean")
    deg_sets = build_eval_sets(art, cfg, probe_source="degraded")

    clean = evaluate(clean_sets, cfg, transform=None)
    degraded = evaluate(deg_sets, cfg, transform=None, per_camera=True)

    gap = _rank1(clean) - _rank1(degraded)
    log(f"  clean    {headline(clean)}")
    log(f"  degraded {headline(degraded)}")
    log(f"  rank-1 lost to degradation: {gap:+.4f} "
        f"(at {cfg.eval.n_enrol} photos x {cfg.eval.n_probe_frames} frames)")

    # Track fusion hides a large part of the damage - which is the point of
    # doing it, but it makes this the wrong place to judge whether the
    # degradation model is strong enough. Measure that at the floor, where
    # nothing is averaging the noise away.
    floor_clean = _rank1(evaluate(
        build_eval_sets(art, cfg, probe_source="clean",
                        n_enrol=1, n_probe_frames=1), cfg))
    floor_deg = _rank1(evaluate(
        build_eval_sets(art, cfg, probe_source="degraded",
                        n_enrol=1, n_probe_frames=1), cfg))
    floor_gap = floor_clean - floor_deg
    log(f"  single photo, single frame: {floor_clean:.4f} -> {floor_deg:.4f} "
        f"({floor_gap:+.4f})")

    warnings: List[str] = []
    if floor_gap < 0.05:
        warnings.append(
            "Degradation barely hurt accuracy. The chain in data/degrade.py is "
            "too gentle: push face_px lower, blur_kernel higher and "
            "jpeg_quality down, then re-run Phase 1. Without a real gap there "
            "is no problem for the adapter to solve."
        )

    return {
        "clean": clean, "degraded": degraded,
        "rank1_clean": _rank1(clean), "rank1_degraded": _rank1(degraded),
        "rank1_gap": gap,
        # The same comparison with no enrolment depth and no track fusion, so
        # the size of the problem is on the record independently of how much
        # of it the operating point papers over.
        "rank1_clean_1x1": floor_clean,
        "rank1_degraded_1x1": floor_deg,
        "rank1_gap_1x1": floor_gap,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------
# 2. Operating points - what the single number was hiding
# --------------------------------------------------------------------------

def exp_operating_points(
        art: Artefacts, cfg: Config,
        enrol: Sequence[int] = (1, 2, 3, 5),
        frames: Sequence[int] = (1, 3, 5)) -> Dict[str, Any]:
    """Rank-1 across enrolment depth, track length and the scoring layer.

    The original protocol reported one cell of this table - one enrolment
    photograph, one camera frame, plain cosine - and called it the system's
    accuracy. It is the worst cell on the surface, and it does not describe
    what this system does. `pipeline.match` already votes over a track of
    `cfg.match.track_vote_n` frames before it raises anything, and a real
    missing-person case file arrives with several photographs, not one.

    So report the surface. The 1x1 cell stays in it, because it is the honest
    floor and someone will ask; but quoting it alone understates the deployed
    system by about forty points, which is not honesty either.

    Every row is the frozen backbone with no adapter, on the same gallery.
    Only the protocol and the scoring layer change, so every difference in
    the table is attributable to one of those two.
    """
    log("operating_points: enrolment depth x track length x scoring layer")
    cal = fit_calibrator(art, cfg)
    if cal is None:
        from .calibrate import fit as _fit
        cal = _fit(art, blend=cfg.eval.calibrate_blend,
                   hub_k=cfg.eval.hub_k, hub_beta=cfg.eval.hub_beta)
    # Three calibrators, built once and reused at every operating point:
    # `evaluate` refits the hubness term against each new gallery anyway, and
    # rebuilding them per cell would re-print every "not enough data" notice
    # a dozen times. hub_beta=0 isolates the first two columns from the third.
    cen = copy.copy(cal)
    cen.rotation, cen.hub_beta = None, 0.0
    rot = copy.copy(cal)
    rot.hub_beta = 0.0

    rows: List[Dict[str, Any]] = []
    log(f"  {'enrol':>5} {'frames':>6} | {'cosine':>7} {'+centre':>8} "
        f"{'+rotate':>8} {'+hubness':>9}")
    for ne in enrol:
        for nf in frames:
            sets = build_eval_sets(art, cfg, n_enrol=ne, n_probe_frames=nf)
            raw = _rank1(evaluate(sets, cfg))
            r_cen = _rank1(evaluate(sets, cfg, calibrator=cen))
            r_rot = _rank1(evaluate(sets, cfg, calibrator=rot))
            r_hub = _rank1(evaluate(sets, cfg, calibrator=cal))
            log(f"  {ne:>5} {nf:>6} | {raw:>7.4f} {r_cen:>8.4f} "
                f"{r_rot:>8.4f} {r_hub:>9.4f}")
            rows.append({
                "n_enrol": ne, "n_probe_frames": nf,
                "rank1_cosine": raw, "rank1_centred": r_cen,
                "rank1_rotated": r_rot, "rank1_hubness": r_hub,
                "n_gallery": int(sets.gallery_ids.size),
                "n_mate_probes": int(sets.mate_ids.size),
                "n_nonmate_probes": int(sets.nonmate_emb.shape[0]),
            })

    default = next((r for r in rows
                    if r["n_enrol"] == cfg.eval.n_enrol
                    and r["n_probe_frames"] == cfg.eval.n_probe_frames), None)
    floor = next((r for r in rows
                  if r["n_enrol"] == 1 and r["n_probe_frames"] == 1), None)

    warnings: List[str] = []
    if any(r["rank1_hubness"] < r["rank1_cosine"] for r in rows):
        warnings.append(
            "The scoring layer lost ground at one or more operating points. "
            "It is fitted on node-training identities only, so this is not "
            "leakage - it means the fitted rotation is not transferring. "
            "Lower eval.calibrate_blend, or set eval.calibrate false and "
            "report plain cosine."
        )

    return {
        "rows": rows,
        "floor_1x1": floor,
        "default": default,
        "note": (f"every row is the frozen backbone, no adapter, on the "
                 f"same {rows[0]['n_gallery']:,}-identity gallery; only the "
                 f"protocol and the scoring layer differ"),
        "warnings": warnings,
    }


# --------------------------------------------------------------------------
# 3. Three-way - the headline
# --------------------------------------------------------------------------

def exp_three_way(art: Artefacts, cfg: Config) -> Dict[str, Any]:
    log("three_way: local-only vs federated vs centralised")
    return run_three_way(art, cfg)


# --------------------------------------------------------------------------
# 3. DP sweep - the privacy-utility curve
# --------------------------------------------------------------------------

def calibrate_clip_norm(art: Artefacts, cfg: Config, rounds: int = 3,
                        quantile: float = 0.7) -> Dict[str, Any]:
    """Observe real update norms, then choose C from them.

    Runs a few rounds with privacy off purely to collect the norms each node
    produces, and takes a quantile of those. Without this the clipping bound
    is a guess, and a bad guess silently turns the privacy-utility curve into
    a measurement of clipping rather than of noise.
    """
    probe = _cfg_copy(cfg, privacy__enable_dp=False,
                      privacy__enable_masking=False, fl__rounds=rounds)
    torch.manual_seed(cfg.seed)
    res = run_federated(art, probe, build_eval_sets(art, cfg))
    res.pop("state", None)

    norms = [m["delta_norm"] for h in res["history"] for m in h["nodes"]]
    chosen = suggest_clip_norm(norms, quantile)
    report = clipping_report(norms, chosen)
    log(f"  calibrated clip_norm={chosen:.3f} from {len(norms)} observed "
        f"updates (median {report['median_norm']:.3f}, "
        f"p95 {report['p95_norm']:.3f}) -> "
        f"{report['fraction_clipped']:.0%} clipped")
    return {"clip_norm": chosen, "quantile": quantile,
            "probe_rounds": rounds, **report}


def exp_dp_sweep(art: Artefacts, cfg: Config,
                 sigmas: Sequence[float] = (0.0, 0.3, 0.6, 1.0, 2.0),
                 rounds_override: Optional[int] = None,
                 calibrate: bool = True) -> Dict[str, Any]:
    """Accuracy against noise, with the epsilon each setting actually buys.

    The curve is the finding. Do not chase epsilon = 1 if it destroys the
    model - report where the knee sits and let the reader judge the trade.

    The clipping bound is calibrated from observed update norms first, so the
    curve measures the effect of noise rather than the effect of a badly
    chosen constant.
    """
    rounds = rounds_override or cfg.fl.rounds
    log(f"dp_sweep: {len(sigmas)} settings x {rounds} rounds")

    calibration: Optional[Dict[str, Any]] = None
    clip_norm = cfg.privacy.clip_norm
    if calibrate:
        calibration = calibrate_clip_norm(art, cfg, rounds=min(3, rounds))
        clip_norm = calibration["clip_norm"]

    sets = build_eval_sets(art, cfg)
    points: List[Dict[str, Any]] = []

    for sigma in sigmas:
        c = _cfg_copy(cfg,
                      privacy__enable_dp=True,
                      privacy__clip_norm=clip_norm,
                      privacy__noise_multiplier=float(sigma),
                      privacy__enable_masking=True,
                      fl__rounds=rounds)
        t0 = time.time()
        res = run_federated(art, c, sets)
        res.pop("state", None)
        acc = compute_epsilon(c.privacy, rounds)

        point = {
            "noise_multiplier": float(sigma),
            "clip_norm": c.privacy.clip_norm,
            "epsilon": acc["epsilon"],
            "delta": acc["delta"],
            "rank1": _rank1(res),
            "verification_auc": res["verification"].get("auc"),
            "fraction_clipped": res["history"][-1].get("fraction_clipped"),
            "seconds": round(time.time() - t0, 1),
        }
        eps = point["epsilon"]
        log(f"  sigma={sigma:<5.2f} eps={eps:>8.3f}  "
            f"rank1={point['rank1']:.4f}  "
            f"clipped={point['fraction_clipped']}")
        points.append(point)

    finite = [p for p in points if np.isfinite(p["epsilon"])]
    return {
        "calibration": calibration,
        "points": points,
        "no_noise_rank1": points[0]["rank1"] if points else float("nan"),
        "note": ("epsilon is update-level and assumes full participation; "
                 "sigma=0 means clipping only, which carries no formal "
                 "guarantee and is reported as infinite epsilon"),
        "knee": min(finite, key=lambda p: -p["rank1"]) if finite else None,
    }


# --------------------------------------------------------------------------
# 4. Secure aggregation
# --------------------------------------------------------------------------

def exp_masking(art: Artefacts, cfg: Config,
                rounds_override: Optional[int] = None) -> Dict[str, Any]:
    """Correctness, overhead, and the unreadable-update screenshot.

    The claim being tested is that masking changes nothing except latency. So
    the same federation is run twice from the same seed, once masked and once
    not, and the two results are compared directly.
    """
    rounds = rounds_override or min(cfg.fl.rounds, 10)
    log(f"masking: {rounds} rounds, with and without")
    sets = build_eval_sets(art, cfg)

    plain_cfg = _cfg_copy(cfg, privacy__enable_masking=False,
                          privacy__enable_dp=False, fl__rounds=rounds)
    mask_cfg = _cfg_copy(cfg, privacy__enable_masking=True,
                         privacy__enable_dp=False, fl__rounds=rounds)

    torch.manual_seed(cfg.seed)
    t0 = time.time()
    plain = run_federated(art, plain_cfg, sets)
    plain_sec = time.time() - t0
    plain.pop("state", None)

    torch.manual_seed(cfg.seed)
    t0 = time.time()
    masked = run_federated(art, mask_cfg, sets)
    masked_sec = time.time() - t0
    masked.pop("state", None)

    # The preview that goes in the report.
    agg = SecureAggregator(art.n_nodes, mask_cfg)
    model = build_adapter(cfg.adapter, cfg.backbone.device)
    fake = clone_state(model.state_dict())
    for k in fake:
        fake[k] = torch.randn_like(fake[k]) * 0.01
    preview = agg.masked_update_preview(fake, node=0,
                                        participants=list(range(art.n_nodes)))

    delta_rank1 = _rank1(masked) - _rank1(plain)
    log(f"  rank1 plain={_rank1(plain):.4f}  masked={_rank1(masked):.4f}  "
        f"difference={delta_rank1:+.5f}")
    log(f"  wall clock {plain_sec:.1f}s -> {masked_sec:.1f}s "
        f"({masked_sec / max(plain_sec, 1e-6):.2f}x)")

    warnings: List[str] = []
    if abs(delta_rank1) > 0.01:
        warnings.append(
            "Masked and unmasked runs diverged by more than 1% rank-1. Masks "
            "should cancel exactly, so this points at a bug - check that "
            "node_ids are passed consistently and that no node was silently "
            "dropped."
        )

    return {
        "rank1_plain": _rank1(plain),
        "rank1_masked": _rank1(masked),
        "rank1_difference": delta_rank1,
        "seconds_plain": round(plain_sec, 1),
        "seconds_masked": round(masked_sec, 1),
        "overhead_factor": round(masked_sec / max(plain_sec, 1e-6), 3),
        "rounds": rounds,
        "dh_bits": mask_cfg.privacy.dh_bits,
        "masked_update_preview": preview,
        "protocol": ("pairwise Diffie-Hellman key agreement; lower-indexed "
                     "node adds the mask, higher-indexed subtracts it; masks "
                     "cancel in the sum"),
        "limitation": ("single-masked, so dropout recovery reveals pairwise "
                       "masks; a production deployment adds a Shamir-shared "
                       "self-mask to close that gap"),
        "warnings": warnings,
    }


# --------------------------------------------------------------------------
# 5. Cancelable templates
# --------------------------------------------------------------------------

def exp_templates(art: Artefacts, cfg: Config,
                  bit_widths: Sequence[int] = (32, 64, 96, 128, 256),
                  adapter_state: Optional[Dict[str, Any]] = None
                  ) -> Dict[str, Any]:
    """What does protecting the watchlist cost in accuracy?

    Evaluated through the identical open-set protocol used for raw
    embeddings - only the scoring function changes, from cosine to Hamming -
    so the comparison is like for like.
    """
    log(f"templates: sweeping {list(bit_widths)} bits")
    sets = build_eval_sets(art, cfg)

    transform = None
    if adapter_state is not None:
        model = build_adapter(cfg.adapter, cfg.backbone.device)
        model.load_state_dict(adapter_state)
        transform = _transform_of(model)

    def apply(x: np.ndarray) -> np.ndarray:
        return x if transform is None else transform(x)

    gallery = sets.gallery_emb
    mate = apply(sets.mate_emb)
    nonmate = apply(sets.nonmate_emb)

    baseline = open_set_identification(
        gallery, sets.gallery_ids, mate, sets.mate_ids, nonmate,
        cfg.eval.fpir_points)
    baseline.pop("det_curve", None)
    log(f"  float embeddings (unprotected): rank1={baseline['rank1']:.4f}")

    # Fit the selector on training data only - never on the evaluation set.
    train_rows: List[int] = []
    for k in range(art.n_nodes):
        train_rows.extend(art.node(k)["train_idx"])
    fit_emb = apply(art.emb_deg[np.asarray(train_rows)])
    fit_lab = art.labels[np.asarray(train_rows)]

    points: List[Dict[str, Any]] = []
    for bits in bit_widths:
        sel = fit_selector(fit_emb, int(bits), fit_lab)
        key = derive_key(cfg.privacy.template_master_key, "case-0001", 0, int(bits))

        g_bits = protect(binarise(gallery, sel), key)
        m_bits = protect(binarise(mate, sel), key)
        n_bits = protect(binarise(nonmate, sel), key)

        res = open_set_identification(
            g_bits, sets.gallery_ids, m_bits, sets.mate_ids, n_bits,
            cfg.eval.fpir_points, score_fn=similarity_matrix)
        res.pop("det_curve", None)

        point = {
            "bits": int(bits),
            "bytes_on_wire": int(bits) // 8,
            "rank1": res["rank1"],
            "rank1_cost": baseline["rank1"] - res["rank1"],
            "selection_criterion": sel.criterion,
            **{k: v for k, v in res.items()
               if k.startswith("tpir@") and not k.endswith("_note")},
            **template_report(sel),
        }
        log(f"  {bits:>4} bits ({bits // 8:>2} bytes): rank1={res['rank1']:.4f}  "
            f"cost={point['rank1_cost']:+.4f}")
        points.append(point)

    default_bits = cfg.privacy.template_bits
    chosen = next((p for p in points if p["bits"] == default_bits), None)

    return {
        "baseline_float": baseline,
        "points": points,
        "chosen": chosen,
        "note": ("templates protect a watchlist at rest and defeat cross-site "
                 "linkage; they do not defend a live compromised node, which "
                 "holds the key and sees its own camera feed regardless"),
        "used_adapter": adapter_state is not None,
    }


# --------------------------------------------------------------------------
# 6. FedAvg vs FedProx
# --------------------------------------------------------------------------

def exp_fedprox(art: Artefacts, cfg: Config,
                mus: Sequence[float] = (0.0, 0.01, 0.1, 1.0),
                rounds_override: Optional[int] = None) -> Dict[str, Any]:
    """Does the proximal term help under this non-IID split?

    [6] found it did, in exactly this setting. mu = 0 is plain FedAvg.
    """
    rounds = rounds_override or cfg.fl.rounds
    log(f"fedprox: mu in {list(mus)}, {rounds} rounds each")
    sets = build_eval_sets(art, cfg)
    points: List[Dict[str, Any]] = []

    for mu in mus:
        c = _cfg_copy(cfg, fl__fedprox_mu=float(mu), fl__rounds=rounds)
        torch.manual_seed(cfg.seed)
        res = run_federated(art, c, sets)
        res.pop("state", None)
        curve = [h["eval"]["rank1"] for h in res["history"] if "eval" in h]
        points.append({
            "mu": float(mu),
            "aggregation": "FedAvg" if mu == 0 else "FedProx",
            "rank1": _rank1(res),
            "verification_auc": res["verification"].get("auc"),
            "final_delta_norm": res["history"][-1]["aggregated_delta_norm"],
            "rank1_curve": curve,
        })
        log(f"  mu={mu:<5.3g} rank1={_rank1(res):.4f}  "
            f"|delta|={points[-1]['final_delta_norm']:.4f}")

    best = max(points, key=lambda p: p["rank1"])
    return {
        "points": points,
        "best_mu": best["mu"],
        "fedavg_rank1": points[0]["rank1"],
        "best_rank1": best["rank1"],
        "improvement": best["rank1"] - points[0]["rank1"],
    }


# --------------------------------------------------------------------------
# 7. Membership inference
# --------------------------------------------------------------------------

def exp_membership_inference(art: Artefacts, cfg: Config,
                             adapter_state: Optional[Dict[str, Any]] = None
                             ) -> Dict[str, Any]:
    """Can an attacker tell whether a face was in a node's training data?

    A loss-threshold attack: a model fits its own training data slightly
    better, so per-sample similarity is itself a membership signal. AUC near
    0.5 means the attack is no better than a coin flip.

    Two variants are reported, because they answer different questions and
    conflating them inflates the result badly:

      sample_level   members are a node's training images; non-members are
                     held-out images OF THE SAME PEOPLE. Membership is the
                     only thing that differs, so this isolates the leak. It is
                     the number comparable with [4]'s 0.507-0.532, and the one
                     to quote.

      identity_level members are training images; non-members are images of
                     people the model never saw at all. This is a far easier
                     question - it largely measures whether an identity was
                     enrolled, not whether a sample was trained on - and it
                     will score close to 1.0 on almost any working model.
                     Reported for completeness, clearly labelled, never as the
                     headline.
    """
    log("mia: loss-threshold membership inference")

    if adapter_state is None:
        sets = build_eval_sets(art, cfg)
        res = run_federated(art, cfg, sets)
        adapter_state = res.pop("state")

    model = build_adapter(cfg.adapter, cfg.backbone.device)
    model.load_state_dict(adapter_state)
    model.eval()

    def scores(rows: Sequence[int]) -> np.ndarray:
        if not len(rows):
            return np.zeros(0, dtype=np.float32)
        data = make_node_data(art, rows, cfg.backbone.device)
        with torch.no_grad():
            adapted = model(data.degraded)
            return (adapted * data.clean).sum(dim=1).cpu().numpy()

    train_rows: List[int] = []
    val_rows: List[int] = []
    for k in range(art.n_nodes):
        train_rows.extend(art.node(k)["train_idx"])
        val_rows.extend(art.node(k)["val_idx"])
    heldout_rows = list(art.partition["test_idx"])

    member = scores(train_rows)
    rng = np.random.default_rng(cfg.seed + 5)

    def attack(nonmember: np.ndarray, label: str) -> Dict[str, Any]:
        n = int(min(member.size, nonmember.size))
        if n == 0:
            return {"error": "not enough data"}
        m = rng.choice(member, size=n, replace=False)
        nm = rng.choice(nonmember, size=n, replace=False)
        fpr, tpr, _ = _roc(np.concatenate([m, nm]),
                           np.concatenate([np.ones(n, dtype=bool),
                                           np.zeros(n, dtype=bool)]))
        auc = _auc(fpr, tpr)
        log(f"  {label:<15} AUC = {auc:.4f}  ({n} per class)")
        return {
            "attack_auc": auc,
            "n_per_class": n,
            "mean_member_similarity": float(m.mean()),
            "mean_nonmember_similarity": float(nm.mean()),
        }

    sample_level = attack(scores(val_rows), "sample level")
    identity_level = attack(scores(heldout_rows), "identity level")

    auc = sample_level.get("attack_auc", float("nan"))
    verdict = ("no measurable leakage" if abs(auc - 0.5) < 0.03
               else "mild leakage" if abs(auc - 0.5) < 0.10
               else "substantial leakage - report this honestly and check "
                    "whether DP closes it")

    return {
        "attack_auc": auc,                 # the headline: sample level
        "verdict": verdict,
        "sample_level": {
            **sample_level,
            "nonmembers": "held-out images of training identities",
            "comparable_to": "[4] reported MIA AUC 0.507-0.532",
        },
        "identity_level": {
            **identity_level,
            "nonmembers": "images of identities never seen by any node",
            "caveat": ("this measures enrolment, not training-set membership; "
                       "it scores near 1.0 on almost any working model and is "
                       "NOT comparable with [4]"),
        },
        "attack": "loss-threshold on per-sample probe-to-gallery similarity",
    }


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

_PALETTE = {
    "ink": "#14191A", "muted": "#6B7679", "rule": "#DADED7",
    "accent": "#0F5F57", "flag": "#8E5A11", "stop": "#93332A",
    "alt": "#2C5C86",
}


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(_PALETTE["rule"])
    ax.tick_params(colors=_PALETTE["muted"], labelsize=9)
    ax.grid(axis="y", color=_PALETTE["rule"], linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def make_figures(results: Dict[str, Any], cfg: Config) -> List[str]:
    """Render every chart the report needs. Returns the paths written."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib not installed - skipping figures "
            "(pip install matplotlib)")
        return []

    out_dir: Path = cfg.paths.figures
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    def save(fig, name: str) -> None:
        path = out_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        written.append(str(path))
        log(f"  wrote {path.name}")

    # -- the problem statement --------------------------------------------
    if "degradation" in results:
        d = results["degradation"]
        fig, ax = plt.subplots(figsize=(5, 3.4))
        bars = ax.bar(["clean probes", "degraded probes"],
                      [d["rank1_clean"], d["rank1_degraded"]],
                      color=[_PALETTE["alt"], _PALETTE["stop"]], width=0.55)
        for b, v in zip(bars, [d["rank1_clean"], d["rank1_degraded"]]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                    ha="center", fontsize=10, color=_PALETTE["ink"])
        ax.set_ylabel("rank-1 identification")
        ax.set_ylim(0, 1.05)
        ax.set_title("What surveillance degradation costs\n"
                     "(frozen backbone, no adapter)", fontsize=11,
                     color=_PALETTE["ink"])
        _style(ax)
        save(fig, "fig1_degradation.png")

    # -- the operating surface --------------------------------------------
    if "operating_points" in results:
        rows = results["operating_points"]["rows"]
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

        enrols = sorted({r["n_enrol"] for r in rows})
        frames = sorted({r["n_probe_frames"] for r in rows})
        colours = [_PALETTE["stop"], _PALETTE["ink"], _PALETTE["alt"],
                   _PALETTE["muted"]]
        for c, ne in zip(colours, enrols):
            ys = [next(r["rank1_hubness"] for r in rows
                       if r["n_enrol"] == ne and r["n_probe_frames"] == nf)
                  for nf in frames]
            ax.plot(frames, ys, marker="o", color=c, linewidth=1.8,
                    label=f"{ne} enrolment photo{'s' if ne > 1 else ''}")
        ax.set_xlabel("camera frames fused into one probe")
        ax.set_ylabel("rank-1 identification")
        ax.set_xticks(frames)
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=False, fontsize=8)
        # Read the gallery size off the data rather than hardcoding it. The
        # hardcoded version shipped a figure captioned "1,750-identity
        # gallery" for a run whose gallery held 3,021, and a caption that is
        # confidently wrong is worse than no caption at all.
        n_gal = rows[0].get("n_gallery", 0)
        ax.set_title(f"Rank-1 across the operating surface\n"
                     f"({n_gal:,}-identity gallery, frozen backbone)",
                     fontsize=11, color=_PALETTE["ink"])
        _style(ax)

        # What the scoring layer adds, cumulatively, at the default point.
        d = results["operating_points"].get("default") or rows[-1]
        keys = [("rank1_cosine", "cosine"), ("rank1_centred", "+ centring"),
                ("rank1_rotated", "+ rotation"), ("rank1_hubness", "+ hubness")]
        vals = [d[k] for k, _ in keys]
        bars = ax2.bar([lbl for _, lbl in keys], vals,
                       color=[_PALETTE["muted"], _PALETTE["alt"],
                              _PALETTE["alt"], _PALETTE["ink"]], width=0.6)
        for b, v in zip(bars, vals):
            ax2.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
                     ha="center", fontsize=9, color=_PALETTE["ink"])
        ax2.set_ylabel("rank-1 identification")
        ax2.set_ylim(min(vals) - 0.05, max(vals) + 0.04)
        ax2.set_title(f"Scoring layer, fitted on training identities only\n"
                      f"({d['n_enrol']} photos x {d['n_probe_frames']} frames)",
                      fontsize=11, color=_PALETTE["ink"])
        _style(ax2)
        fig.tight_layout()
        save(fig, "fig8_operating_points.png")

    # -- the headline ------------------------------------------------------
    if "three_way" in results:
        r = results["three_way"]["summary"]["rank1"]
        labels = ["no adapter", "local only", "federated", "centralised"]
        vals = [r["no_adapter"], r["local_only_mean"],
                r["federated"], r["centralised"]]
        colors = [_PALETTE["muted"], _PALETTE["flag"],
                  _PALETTE["accent"], _PALETTE["alt"]]
        fig, ax = plt.subplots(figsize=(6, 3.6))
        bars = ax.bar(labels, vals, color=colors, width=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                    ha="center", fontsize=10, color=_PALETTE["ink"])
        ax.set_ylabel("rank-1 identification")
        ax.set_ylim(0, min(1.05, max(vals) * 1.25))
        ax.set_title("Federated learning against its bounds", fontsize=11,
                     color=_PALETTE["ink"])
        _style(ax)
        save(fig, "fig2_three_way.png")

        hist = results["three_way"]["federated"].get("history", [])
        curve = [(i + 1, h["eval"]["rank1"])
                 for i, h in enumerate(hist) if "eval" in h]
        if len(curve) > 1:
            fig, ax = plt.subplots(figsize=(6, 3.4))
            ax.plot([c[0] for c in curve], [c[1] for c in curve],
                    color=_PALETTE["accent"], marker="o", markersize=4,
                    linewidth=1.8)
            ax.axhline(r["local_only_mean"], color=_PALETTE["flag"],
                       linestyle="--", linewidth=1.2, label="local only")
            ax.axhline(r["centralised"], color=_PALETTE["alt"],
                       linestyle=":", linewidth=1.2, label="centralised")
            ax.set_xlabel("federated round")
            ax.set_ylabel("rank-1 identification")
            ax.set_title("Convergence", fontsize=11, color=_PALETTE["ink"])
            ax.legend(frameon=False, fontsize=9)
            _style(ax)
            save(fig, "fig3_convergence.png")

    # -- privacy-utility ---------------------------------------------------
    if "dp_sweep" in results:
        pts = [p for p in results["dp_sweep"]["points"]
               if np.isfinite(p["epsilon"])]
        if pts:
            fig, ax = plt.subplots(figsize=(6, 3.6))
            eps = [p["epsilon"] for p in pts]
            r1 = [p["rank1"] for p in pts]
            ax.plot(eps, r1, color=_PALETTE["accent"], marker="o",
                    markersize=5, linewidth=1.8)
            for p in pts:
                ax.annotate(f"σ={p['noise_multiplier']:g}",
                            (p["epsilon"], p["rank1"]),
                            textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=8,
                            color=_PALETTE["muted"])
            no_noise = results["dp_sweep"]["no_noise_rank1"]
            if np.isfinite(no_noise):
                ax.axhline(no_noise, color=_PALETTE["muted"], linestyle="--",
                           linewidth=1.1, label="no noise (no guarantee)")
                ax.legend(frameon=False, fontsize=9)
            ax.set_xscale("log")
            ax.set_xlabel(f"privacy budget ε  (δ = {pts[0]['delta']:g}, "
                          f"lower is more private)")
            ax.set_ylabel("rank-1 identification")
            ax.set_title("The privacy–utility trade", fontsize=11,
                         color=_PALETTE["ink"])
            _style(ax)
            save(fig, "fig4_dp_sweep.png")

    # -- DET curve ---------------------------------------------------------
    det_src = None
    if "three_way" in results:
        det_src = results["three_way"]["federated"]["identification"]
    if det_src and "det_curve" in det_src:
        base = results.get("three_way", {}).get(
            "baseline_no_adapter", {}).get("identification", {})
        fig, ax = plt.subplots(figsize=(5.4, 4))
        for src, label, color in (
                (base, "no adapter", _PALETTE["muted"]),
                (det_src, "federated adapter", _PALETTE["accent"])):
            c = src.get("det_curve")
            if not c:
                continue
            f = np.asarray(c["fpir"])
            t = np.asarray(c["tpir"])
            keep = f > 0
            ax.plot(f[keep], t[keep], label=label, color=color, linewidth=1.8)
        ax.set_xscale("log")
        ax.set_xlabel("FPIR  (false alerts on people in no watchlist)")
        ax.set_ylabel("TPIR  (watchlisted people correctly found)")
        ax.set_title("Open-set identification", fontsize=11,
                     color=_PALETTE["ink"])
        ax.legend(frameon=False, fontsize=9, loc="lower right")
        _style(ax)
        ax.grid(True, which="both", color=_PALETTE["rule"], linewidth=0.5,
                alpha=0.6)
        save(fig, "fig5_det.png")

    # -- templates ---------------------------------------------------------
    if "templates" in results:
        pts = results["templates"]["points"]
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.plot([p["bits"] for p in pts], [p["rank1"] for p in pts],
                color=_PALETTE["accent"], marker="o", markersize=5,
                linewidth=1.8, label="cancelable template")
        base = results["templates"]["baseline_float"]["rank1"]
        ax.axhline(base, color=_PALETTE["muted"], linestyle="--",
                   linewidth=1.2, label="unprotected 512-d float")
        chosen = cfg.privacy.template_bits
        ax.axvline(chosen, color=_PALETTE["flag"], linestyle=":",
                   linewidth=1.2)
        ax.annotate(f"deployed: {chosen} bits", (chosen, ax.get_ylim()[0]),
                    textcoords="offset points", xytext=(6, 14), fontsize=8,
                    color=_PALETTE["flag"])
        ax.set_xlabel("template width (bits)")
        ax.set_ylabel("rank-1 identification")
        ax.set_title("Accuracy cost of protecting the watchlist", fontsize=11,
                     color=_PALETTE["ink"])
        ax.legend(frameon=False, fontsize=9)
        _style(ax)
        save(fig, "fig6_templates.png")

    # -- fedavg vs fedprox -------------------------------------------------
    if "fedprox" in results:
        pts = results["fedprox"]["points"]
        fig, ax = plt.subplots(figsize=(6, 3.4))
        for p in pts:
            curve = p["rank1_curve"]
            if not curve:
                continue
            ax.plot(range(1, len(curve) + 1), curve, marker="o", markersize=3,
                    linewidth=1.6,
                    label=f"{p['aggregation']} μ={p['mu']:g}")
        ax.set_xlabel("evaluation point")
        ax.set_ylabel("rank-1 identification")
        ax.set_title("FedAvg against FedProx under non-IID data", fontsize=11,
                     color=_PALETTE["ink"])
        ax.legend(frameon=False, fontsize=9)
        _style(ax)
        save(fig, "fig7_fedprox.png")

    return written


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

ALL_EXPERIMENTS = ("degradation", "operating_points", "three_way", "dp_sweep",
                   "masking", "templates", "fedprox", "mia")

DEFAULT_EXPERIMENTS = ("degradation", "operating_points", "three_way",
                       "dp_sweep", "masking", "templates")


def run_all(art: Artefacts, cfg: Config,
            which: Sequence[str] = DEFAULT_EXPERIMENTS,
            rounds_override: Optional[int] = None,
            figures: bool = True) -> Dict[str, Any]:
    """Run the requested experiments and collect everything into one dict."""
    unknown = set(which) - set(ALL_EXPERIMENTS)
    if unknown:
        raise ValueError(f"unknown experiments: {sorted(unknown)}; "
                         f"choose from {ALL_EXPERIMENTS}")

    t0 = time.time()
    results: Dict[str, Any] = {"config": cfg.to_dict(),
                               "backbone": art.meta.get("backbone"),
                               "synthetic_backbone":
                                   art.meta.get("backbone") == "dummy"}
    adapter_state = None

    if "degradation" in which:
        results["degradation"] = exp_degradation(art, cfg)

    if "operating_points" in which:
        results["operating_points"] = exp_operating_points(art, cfg)

    if "three_way" in which:
        out = exp_three_way(art, cfg)
        results["three_way"] = out
        ckpt = cfg.paths.checkpoints / "adapter_final.pt"
        if ckpt.exists():
            adapter_state = torch.load(ckpt, map_location="cpu",
                                       weights_only=False)["state_dict"]

    if "dp_sweep" in which:
        results["dp_sweep"] = exp_dp_sweep(art, cfg,
                                           rounds_override=rounds_override)
    if "masking" in which:
        results["masking"] = exp_masking(art, cfg,
                                         rounds_override=rounds_override)
    if "templates" in which:
        results["templates"] = exp_templates(art, cfg,
                                             adapter_state=adapter_state)
    if "fedprox" in which:
        results["fedprox"] = exp_fedprox(art, cfg,
                                         rounds_override=rounds_override)
    if "mia" in which:
        results["mia"] = exp_membership_inference(art, cfg,
                                                  adapter_state=adapter_state)

    results["elapsed_sec"] = round(time.time() - t0, 1)

    warnings: List[str] = []
    for name, block in results.items():
        if isinstance(block, dict):
            warnings.extend(f"[{name}] {w}" for w in block.get("warnings", []))
            summary = block.get("summary", {})
            if isinstance(summary, dict):
                warnings.extend(f"[{name}] {w}"
                                for w in summary.get("warnings", []))
    results["warnings"] = warnings

    if figures:
        log("rendering figures")
        results["figures"] = make_figures(results, cfg)

    return results


if __name__ == "__main__":  # smoke check
    from ..config import default_config
    from ..contracts import l2_normalise

    cfg = default_config()
    cfg.fl.rounds = 6
    cfg.train.local_epochs = 2
    cfg.train.centralised_epochs = 12
    cfg.eval.eval_every = 2
    cfg.eval.n_verification_pairs = 300
    cfg.privacy.dh_bits = 1024

    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    n_nodes, ids_per_node, per_id, dim = 3, 6, 10, cfg.adapter.in_dim
    n_test = 24
    total = n_nodes * ids_per_node + n_test
    labels = np.repeat(np.arange(total), per_id).astype(np.int32)
    cents = l2_normalise(rng.normal(size=(total, dim)))

    # A unit vector in 512 dimensions has components of size 1/sqrt(512) =
    # 0.044, so the plain `0.15 * normal()` this used to add was three times
    # the identity signal it was perturbing: within-identity cosine came out
    # at 0.09, every identity was indistinguishable from every other, and the
    # whole smoke check ran at chance while asserting things about rank-1.
    # Scale the noise to the signal, the way metrics.py already does.
    #
    # The constants are then set so this synthetic set sits where the real one
    # does - clean 1.00, degraded 0.46 at one photo and one frame, rising to
    # 0.96 at three and three. Too little noise and every assertion below
    # passes at the ceiling and tests nothing; too much and it passes at
    # chance, which is how this check got into that state to begin with.
    scale = 3.0 / np.sqrt(dim)
    clean = l2_normalise(cents[labels]
                         + 0.60 * scale * rng.normal(size=(labels.size, dim)))
    shared = l2_normalise(rng.normal(size=(1, dim)))[0]
    deg = l2_normalise(clean + 0.5 * shared
                       + 1.20 * scale * rng.normal(size=clean.shape))

    nodes_meta = {}
    for k in range(n_nodes):
        ids = list(range(k * ids_per_node, (k + 1) * ids_per_node))
        rows = [i for i in range(labels.size) if int(labels[i]) in ids]
        cut = int(len(rows) * 0.8)
        nodes_meta[f"node_{k}"] = {"identities": ids, "train_idx": rows[:cut],
                                   "val_idx": rows[cut:], "degrade_params": {}}
    test_ids = list(range(n_nodes * ids_per_node, total))
    test_idx = [i for i in range(labels.size) if int(labels[i]) in test_ids]
    part = {"n_nodes": n_nodes, "seed": 0, "test_idx": test_idx,
            "test_identities": test_ids,
            "test_profile_ids": [i % n_nodes for i in range(len(test_idx))],
            "nodes": nodes_meta}
    art = Artefacts(clean, deg, labels, part, {"backbone": "synthetic"})

    out = run_all(art, cfg,
                  which=("degradation", "operating_points", "three_way",
                         "masking", "templates"),
                  figures=True)

    assert "degradation" in out and "three_way" in out
    assert abs(out["masking"]["rank1_difference"]) < 0.02, \
        "secure aggregation changed the result"
    assert out["templates"]["points"][0]["bits"] == 32

    # Degradation must cost something, or nothing downstream means anything.
    assert out["degradation"]["rank1_gap_1x1"] > 0.2, \
        (f"synthetic degradation is too gentle: "
         f"{out['degradation']['rank1_gap_1x1']:.3f} at one photo, one frame")

    # The operating surface must rise with both enrolment depth and track
    # length. If it does not, the fusion in build_eval_sets is not working.
    ops = {(r["n_enrol"], r["n_probe_frames"]): r
           for r in out["operating_points"]["rows"]}
    assert ops[(1, 3)]["rank1_cosine"] > ops[(1, 1)]["rank1_cosine"], \
        "fusing frames did not help"
    assert ops[(3, 1)]["rank1_cosine"] > ops[(1, 1)]["rank1_cosine"], \
        "deeper enrolment did not help"
    assert ops[(3, 3)]["rank1_cosine"] > ops[(3, 1)]["rank1_cosine"], \
        "the two do not compose"
    floor, default = out["operating_points"]["floor_1x1"], out["operating_points"]["default"]
    print(f"\noperating surface: 1x1 {floor['rank1_cosine']:.3f} -> "
          f"{default['n_enrol']}x{default['n_probe_frames']} "
          f"{default['rank1_cosine']:.3f} "
          f"(calibrated {default['rank1_hubness']:.3f})")

    print(f"figures written: {len(out.get('figures', []))}")
    print(f"warnings: {out['warnings'] or 'none'}")
    print("experiments OK")
