"""The three training modes, and the comparison the whole project rests on.

    local_only   each node trains alone on its own data          lower bound
    federated    nodes collaborate through weight averaging       the system
    centralised  all data pooled into one training set            upper bound

Federated must beat local_only. If it does not, there is no argument for the
methodology and the project needs to react - which is why this comparison is
scheduled for week 6 of twelve rather than week eleven.

Centralised is the bound federated is allowed to approach but never exceed.
If federated comes out ahead of centralised, that is a bug, almost always
test-set contamination, and the run should be treated as invalid.

Fairness note: local_only is given exactly the same number of local epochs
over its own data as federated gets in total (rounds x local_epochs). The two
therefore differ only in whether weights are shared, which is the single
variable under test.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..config import Config
from ..contracts import Artefacts, F_ADAPTER, l2_normalise, save_json
from ..eval.calibrate import Calibrator
from ..eval.metrics import (
    EvalSets, build_eval_sets, evaluate, fit_calibrator, headline,
)
from ..models.adapter import Adapter, StateDict, build_adapter, clone_state
from .node import Node, build_nodes, local_eval, local_train, make_node_data
from .server import Server


def log(msg: str) -> None:
    print(f"[fl] {msg}", flush=True)


def _transform_of(model: Adapter):
    """Wrap a trained adapter as the probe transform evaluation expects."""
    model.eval()
    return lambda emb: model.transform_numpy(emb)


# --------------------------------------------------------------------------
# Identity-disjoint validation
# --------------------------------------------------------------------------

def split_node_identities(art: Artefacts, cfg: Config
                          ) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """Carve a held-back slice of identities out of every node.

    Returns (train_rows, val_rows) keyed by node id. The identities in the
    second never appear in the first, so an adapter selected on the second has
    been selected on people it did not train on - which is the whole point.
    Held-out test identities appear in neither, so this is not the test set by
    another name.
    """
    frac = float(cfg.fl.val_identity_frac)
    labels = art.labels
    train_rows: Dict[int, List[int]] = {}
    val_rows: Dict[int, List[int]] = {}

    for k in range(art.n_nodes):
        entry = art.node(k)
        rows = np.asarray(list(entry["train_idx"]) + list(entry["val_idx"]),
                          dtype=np.int64)
        ids = np.unique(labels[rows])
        n_val = int(round(frac * ids.size))
        if frac <= 0.0 or n_val < 1 or n_val >= ids.size:
            train_rows[k] = rows.tolist()
            val_rows[k] = []
            continue
        rng = np.random.default_rng(cfg.seed + 4242 + k)
        held = set(int(i) for i in rng.choice(ids, size=n_val, replace=False))
        is_val = np.array([int(l) in held for l in labels[rows]])
        train_rows[k] = rows[~is_val].tolist()
        val_rows[k] = rows[is_val].tolist()

    return train_rows, val_rows


def build_val_sets(art: Artefacts, cfg: Config,
                   val_rows: Dict[int, List[int]]) -> Optional[EvalSets]:
    """A gallery-and-probe protocol over the held-back node identities.

    Scored exactly like the test protocol - `cfg.eval.n_enrol` clean photos
    enrolled, the rest fused into tracks - so a round that looks better here
    looks better for the same reason it would there.
    """
    rows = np.asarray(sorted({int(r) for rs in val_rows.values() for r in rs}),
                      dtype=np.int64)
    if rows.size == 0:
        return None

    labels = art.labels
    rng = np.random.default_rng(cfg.seed + 8123)
    n_enrol, n_frames = cfg.eval.n_enrol, cfg.eval.n_probe_frames

    gal, gal_ids, probes, probe_ids, profiles = [], [], [], [], []
    for ident in np.unique(labels[rows]):
        r = rows[labels[rows] == ident]
        if r.size < n_enrol + n_frames:
            continue
        r = r[rng.permutation(r.size)]
        gal.append(art.emb_clean[r[:n_enrol]].mean(axis=0))
        gal_ids.append(int(ident))
        rest = r[n_enrol:]
        for s in range(0, rest.size - n_frames + 1, n_frames):
            probes.append(art.emb_deg[rest[s:s + n_frames]].mean(axis=0))
            probe_ids.append(int(ident))
            profiles.append(0)

    if len(gal) < 2 or not probes:
        return None

    empty = np.zeros((0, art.emb_clean.shape[1]), dtype=np.float32)
    return EvalSets(
        gallery_emb=l2_normalise(np.asarray(gal, dtype=np.float32)),
        gallery_ids=np.asarray(gal_ids, dtype=np.int64),
        mate_emb=l2_normalise(np.asarray(probes, dtype=np.float32)),
        mate_ids=np.asarray(probe_ids, dtype=np.int64),
        mate_profiles=np.asarray(profiles, dtype=np.int32),
        nonmate_emb=empty, nonmate_profiles=np.zeros(0, dtype=np.int32),
        ver_clean=empty, ver_degraded=empty, ver_same=np.zeros(0, dtype=bool),
        n_enrol=n_enrol, n_probe_frames=n_frames,
    )


def _val_rank1(sets: Optional[EvalSets], model: Adapter,
               calibrator: Optional[Calibrator]) -> float:
    """Closed-set rank-1 on the held-back identities. No non-mates needed."""
    if sets is None:
        return float("nan")
    probes = model.transform_numpy(sets.mate_emb)
    gallery = sets.gallery_emb
    if calibrator is not None:
        probes = calibrator.probe(probes)
        gallery = calibrator.gallery(gallery)
    hit = 0
    for i in range(0, probes.shape[0], 4096):
        sim = probes[i:i + 4096] @ gallery.T
        hit += int((sets.gallery_ids[sim.argmax(1)]
                    == sets.mate_ids[i:i + 4096]).sum())
    return hit / probes.shape[0]


# --------------------------------------------------------------------------
# Mode 1 - centralised (upper bound)
# --------------------------------------------------------------------------

def run_centralised(art: Artefacts, cfg: Config,
                    sets: EvalSets,
                    calibrator: Optional[Calibrator] = None) -> Dict[str, Any]:
    """Pool every node's training data and train one adapter on all of it.

    Implemented as **federated learning with a single client that holds
    everything** - same rounds, same local epochs, same proximal term, same
    optimiser. Only the partitioning differs, which is the one variable under
    test.

    That matters, and it was wrong before. The earlier version trained a fixed
    number of unconstrained epochs while the federated arm ran under a FedProx
    penalty. Once the federated arm was properly regularised it became the
    *more* constrained of the two, overtook this one, and tripped the
    "federated beat centralised, therefore leakage" alarm - when the real
    cause was that the baseline was under-regularised and had stopped being an
    upper bound at all. A bound is only a bound if it is tuned at least as
    well as the thing it bounds.

    Note what this row costs in reality: it requires centralising exactly the
    biometric data the project exists to keep local. The gap between this row
    and the federated row is the actual price of privacy.
    """
    t0 = time.time()
    # Same identity-disjoint split the federated arm trains under. Pooling the
    # full partition here while the federated arm trains on a subset would
    # hand this row extra data and stop it being a like-for-like bound.
    node_train, node_val = split_node_identities(art, cfg)
    rows: List[int] = []
    val_rows: List[int] = []
    for k in range(art.n_nodes):
        rows.extend(node_train[k])
        val_rows.extend(art.node(k)["val_idx"])

    pooled = Node(
        node_id=0,
        train_data=make_node_data(art, rows, cfg.backbone.device),
        val_data=make_node_data(art, val_rows, cfg.backbone.device),
        cfg=cfg,
    )
    log(f"centralised: {pooled.train_data.n} images, "
        f"{pooled.train_data.n_identities} identities, pooled into one client "
        f"({cfg.fl.rounds} rounds x {cfg.train.local_epochs} epoch, "
        f"mu={cfg.fl.fedprox_mu})")

    result = run_federated(art, cfg, sets, nodes=[pooled], calibrator=calibrator,
                           val_sets=build_val_sets(art, cfg, node_val))
    result["elapsed_sec"] = round(time.time() - t0, 1)
    result["matched_budget"] = {
        "rounds": cfg.fl.rounds,
        "local_epochs": cfg.train.local_epochs,
        "fedprox_mu": cfg.fl.fedprox_mu,
        "lr": cfg.train.lr,
        "note": ("identical to the federated arm in every respect except that "
                 "one client holds all the data"),
    }
    log(f"centralised: {headline(result)}")
    return result


# --------------------------------------------------------------------------
# Mode 2 - local only (lower bound)
# --------------------------------------------------------------------------

def run_local_only(art: Artefacts, cfg: Config,
                   sets: EvalSets,
                   calibrator: Optional[Calibrator] = None) -> Dict[str, Any]:
    """Every node trains in isolation. No weights are ever shared.

    Each node's adapter is then scored on the same global evaluation set, so
    the question being asked is the operational one: could this node, working
    alone, find a missing person it had never seen?
    """
    t0 = time.time()
    epochs = cfg.fl.rounds * cfg.train.local_epochs
    log(f"local_only: {art.n_nodes} nodes x {epochs} epochs each")

    per_node: Dict[str, Any] = {}
    rank1s: List[float] = []

    node_train, _node_val = split_node_identities(art, cfg)
    for k in range(art.n_nodes):
        entry = art.node(k)
        data = make_node_data(art, node_train[k], cfg.backbone.device)
        val = make_node_data(art, entry["val_idx"], cfg.backbone.device)

        model = build_adapter(cfg.adapter, cfg.backbone.device)
        stats = local_train(model, data, cfg, epochs=epochs,
                            seed=cfg.seed + k)
        val_stats = local_eval(model, val, cfg)

        res = evaluate(sets, cfg, transform=_transform_of(model),
                       calibrator=calibrator)
        res["train"] = stats
        res["val"] = val_stats
        res.pop("per_camera", None)
        res["identification"].pop("det_curve", None)
        per_node[f"node_{k}"] = res

        r1 = res["identification"]["rank1"]
        rank1s.append(r1)
        log(f"  node_{k}: n={data.n:<5} rank1={r1:.4f}  "
            f"sim_gain={val_stats['sim_gain']:+.4f}")

    summary = {
        "per_node": per_node,
        "mean_rank1": float(np.mean(rank1s)) if rank1s else float("nan"),
        "best_rank1": float(np.max(rank1s)) if rank1s else float("nan"),
        "worst_rank1": float(np.min(rank1s)) if rank1s else float("nan"),
        "epochs_each": epochs,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    log(f"local_only: mean rank1={summary['mean_rank1']:.4f} "
        f"(worst {summary['worst_rank1']:.4f}, best {summary['best_rank1']:.4f})")
    return summary


# --------------------------------------------------------------------------
# Mode 3 - federated (the system)
# --------------------------------------------------------------------------

def run_federated(art: Artefacts, cfg: Config,
                  sets: EvalSets,
                  nodes: Optional[List[Node]] = None,
                  calibrator: Optional[Calibrator] = None,
                  val_sets: Optional[EvalSets] = None) -> Dict[str, Any]:
    """FedAvg (or FedProx, if fl.fedprox_mu > 0) across the nodes.

    With `cfg.fl.early_stop` the round with the best identity-disjoint
    validation rank-1 is kept, not the last one. On the recorded runs the last
    round is reliably the worst: held-out rank-1 peaks within one or two
    rounds and decays for the remaining twenty-eight, ending below the frozen
    backbone the adapter started from. Training longer was making the system
    worse and the old per-node validation split - which shares its identities
    with training - could not see it.
    """
    t0 = time.time()
    if nodes is None:
        train_rows, val_rows = split_node_identities(art, cfg)
        nodes = build_nodes(art, cfg, train_rows=train_rows)
        if val_sets is None:
            val_sets = build_val_sets(art, cfg, val_rows)
    server = Server(cfg, n_nodes=len(nodes))
    rng = np.random.default_rng(cfg.seed + 77)

    if val_sets is not None:
        log(f"early stopping on {val_sets.gallery_ids.size} held-back "
            f"identities ({val_sets.mate_ids.size} probes), disjoint from "
            f"both training and test")
    elif cfg.fl.early_stop:
        log("no identity-disjoint validation set could be built; "
            "keeping the final round")

    n_per_round = cfg.fl.clients_per_round or len(nodes)
    n_per_round = int(np.clip(n_per_round, 1, len(nodes)))

    mode = "FedProx" if cfg.fl.fedprox_mu > 0 else "FedAvg"
    log(f"federated: {mode}, {len(nodes)} nodes, {cfg.fl.rounds} rounds, "
        f"{n_per_round} sampled per round")

    history: List[Dict[str, Any]] = []
    best_val, best_state, best_round = -1.0, None, None
    base_val = _val_rank1(val_sets, build_adapter(cfg.adapter,
                                                  cfg.backbone.device),
                          calibrator)

    for r in range(cfg.fl.rounds):
        chosen = (list(range(len(nodes))) if n_per_round == len(nodes)
                  else rng.choice(len(nodes), size=n_per_round,
                                  replace=False).tolist())

        global_state = server.broadcast()
        updates = []
        node_metrics = []
        for k in chosen:
            delta, n, m = nodes[k].train_round(global_state, r)
            updates.append((delta, n))
            node_metrics.append(m)

        agg = server.aggregate(updates, node_ids=chosen)

        entry: Dict[str, Any] = {
            **agg,
            "nodes": node_metrics,
            "mean_val_loss": float(np.nanmean(
                [m["val_loss"] for m in node_metrics])),
            "mean_sim_gain": float(np.nanmean(
                [m["sim_gain"] for m in node_metrics])),
        }

        model = server.load_into_model()
        val_r1 = _val_rank1(val_sets, model, calibrator)
        entry["val_rank1"] = val_r1
        if val_r1 == val_r1 and val_r1 > best_val:
            best_val, best_state, best_round = val_r1, clone_state(
                server.global_state), r

        is_last = (r == cfg.fl.rounds - 1)
        if is_last or (r + 1) % max(1, cfg.eval.eval_every) == 0:
            res = evaluate(sets, cfg, transform=_transform_of(model),
                           calibrator=calibrator)
            entry["eval"] = {
                "rank1": res["identification"]["rank1"],
                **{k: v for k, v in res["identification"].items()
                   if k.startswith("tpir@") and not k.endswith("_note")},
                "verification_auc": res["verification"].get("auc"),
            }
            log(f"  round {r + 1:>3}/{cfg.fl.rounds}  "
                f"rank1={entry['eval']['rank1']:.4f}  "
                f"val_rank1={val_r1:.4f}  "
                f"sim_gain={entry['mean_sim_gain']:+.4f}  "
                f"|delta|={agg['aggregated_delta_norm']:.4f}")

        history.append(entry)

    selected = cfg.fl.rounds - 1
    if cfg.fl.early_stop and best_state is not None and best_round is not None:
        if best_round != cfg.fl.rounds - 1:
            log(f"early stop: keeping round {best_round + 1}/{cfg.fl.rounds} "
                f"(val rank1={best_val:.4f}) instead of the final round")
        server.global_state = best_state
        selected = best_round

    model = server.load_into_model()
    result = evaluate(sets, cfg, transform=_transform_of(model),
                      per_camera=True, calibrator=calibrator)
    result["history"] = history
    result["payload"] = server.payload_report()
    result["privacy"] = server.privacy_report(rounds=cfg.fl.rounds)
    result["aggregation"] = mode
    result["early_stop"] = {
        "enabled": bool(cfg.fl.early_stop),
        "selected_round": int(selected) + 1,
        "total_rounds": int(cfg.fl.rounds),
        "val_rank1_selected": (float(best_val) if best_state is not None
                               else float("nan")),
        # The untrained adapter is an exact pass-through, so this is the frozen
        # backbone's own score on the same held-back identities. If no round
        # beat it, the adapter earned nothing and the report should say so.
        "val_rank1_no_adapter": float(base_val),
        "beat_frozen_backbone": bool(best_state is not None
                                     and base_val == base_val
                                     and best_val > base_val),
    }
    result["elapsed_sec"] = round(time.time() - t0, 1)
    result["state"] = clone_state(server.global_state)
    log(f"federated: {headline(result)}")
    return result


# --------------------------------------------------------------------------
# The comparison
# --------------------------------------------------------------------------

def run_three_way(art: Artefacts, cfg: Config,
                  save_to: Optional[str] = None,
                  calibrator: Optional[Calibrator] = None) -> Dict[str, Any]:
    """Run all three modes on identical data and report them side by side.

    Every arm, the no-adapter baseline included, is scored through the same
    calibrator and the same operating point. That is the only way the three
    rows compare: a calibrated federated arm against an uncalibrated baseline
    would credit the adapter with the scoring layer's +2.4 points, which is
    most of what the adapter appears to earn in the first place.
    """
    torch.manual_seed(cfg.seed)
    sets = build_eval_sets(art, cfg)
    calibrator = fit_calibrator(art, cfg) if calibrator is None else calibrator
    log(sets.describe())
    log(f"scoring layer: {calibrator.describe() if calibrator else 'none'}")

    baseline = evaluate(sets, cfg, transform=None, per_camera=True,
                        calibrator=calibrator)
    log(f"no adapter (frozen backbone alone): {headline(baseline)}")

    local = run_local_only(art, cfg, sets, calibrator)
    fed = run_federated(art, cfg, sets, calibrator=calibrator)
    cen = run_centralised(art, cfg, sets, calibrator)

    fed_state = fed.pop("state")
    cen.pop("state", None)

    b1 = baseline["identification"]["rank1"]
    l1 = local["mean_rank1"]
    f1 = fed["identification"]["rank1"]
    c1 = cen["identification"]["rank1"]

    verdict: List[str] = []
    if f1 <= l1:
        verdict.append(
            "FEDERATED DID NOT BEAT LOCAL-ONLY. The methodology is "
            "unjustified as it stands. Most likely the nodes are too similar "
            "to each other, so there is nothing worth sharing - widen the "
            "per-node degradation ranges and shrink each node's data, then "
            "re-run. See the red-flag table in the build plan."
        )
    if f1 > c1 + 1e-6:
        verdict.append(
            "FEDERATED BEAT CENTRALISED. Check two things, in this order, "
            "before concluding anything.\n"
            "      (1) Leakage. Confirm no identity and no image index is "
            "shared between any node and the held-out set. If they are, the "
            "run is invalid.\n"
            "      (2) Matched tuning. Centralised is only an upper bound if "
            "it is regularised at least as well as the federated arm. An "
            "under-regularised baseline overfits and stops bounding anything, "
            "which looks identical to leakage in this table but is not. "
            "run_centralised trains one pooled client under the same rounds, "
            "epochs and proximal term, so this should hold - but if you have "
            "changed either arm's hyperparameters independently, re-check it.\n"
            "      A small margin after both checks pass is a real, reportable "
            "result: averaging across heterogeneous clients is itself a "
            "regulariser, and on small non-IID data it can beat pooling."
        )
    if f1 <= b1:
        verdict.append(
            "The adapter did not improve on the frozen backbone alone. "
            "Check that the degradation model is strong enough to matter and "
            "that the shift it induces is systematic rather than pure noise."
        )

    summary = {
        "rank1": {
            "no_adapter": b1,
            "local_only_mean": l1,
            "federated": f1,
            "centralised": c1,
        },
        "federated_gain_over_local": f1 - l1,
        "federated_gap_to_centralised": c1 - f1,
        "adapter_gain_over_baseline": f1 - b1,
        "warnings": verdict,
    }

    print()
    log("three-way comparison, rank-1 on held-out identities")
    log(f"  no adapter        {b1:.4f}")
    log(f"  local only (mean) {l1:.4f}")
    log(f"  federated         {f1:.4f}   "
        f"({f1 - l1:+.4f} vs local only)")
    log(f"  centralised       {c1:.4f}   "
        f"({c1 - f1:+.4f} headroom remaining)")
    for w in verdict:
        print()
        log(f"WARNING: {w}")
    print()

    out = {
        "summary": summary,
        "baseline_no_adapter": baseline,
        "local_only": local,
        "federated": fed,
        "centralised": cen,
        "eval_sets": sets.describe(),
        "config": cfg.to_dict(),
    }

    ckpt = cfg.paths.checkpoints / F_ADAPTER
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": fed_state,
                "adapter_config": cfg.adapter.__dict__,
                "backbone": art.meta.get("backbone")}, ckpt)
    log(f"saved the federated adapter to {ckpt}")

    if save_to:
        save_json(cfg.paths.results / save_to, out)
        log(f"saved results to {cfg.paths.results / save_to}")

    return out


if __name__ == "__main__":  # smoke check
    # A synthetic five-node federation. Each node gets its own systematic
    # shift plus a shared component, so collaboration should genuinely help:
    # no node can learn the shared part well from its own handful of people.
    from ..config import default_config
    from ..contracts import l2_normalise

    cfg = default_config()
    cfg.fl.rounds = 12
    cfg.train.local_epochs = 3
    cfg.train.centralised_epochs = 36
    cfg.eval.eval_every = 4
    cfg.eval.n_verification_pairs = 400

    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    n_nodes, ids_per_node, per_id, dim = 5, 8, 8, cfg.adapter.in_dim
    n_test_ids = 30

    total_ids = n_nodes * ids_per_node + n_test_ids
    labels = np.repeat(np.arange(total_ids), per_id).astype(np.int32)
    cents = l2_normalise(rng.normal(size=(total_ids, dim)))
    # Moderate noise: harder task than 1/sqrt(dim) gives room to improve.
    scale = 3.0 / np.sqrt(dim)
    clean = l2_normalise(cents[labels] + 0.15 * scale * rng.normal(size=(labels.size, dim)))

    shared = l2_normalise(rng.normal(size=(1, dim)))[0]
    node_bias = l2_normalise(rng.normal(size=(n_nodes, dim)))

    owner = np.zeros(labels.size, dtype=np.int32)
    for i, lab in enumerate(labels):
        owner[i] = int(lab) // ids_per_node if int(lab) < n_nodes * ids_per_node \
            else int(lab) % n_nodes

    deg = np.empty_like(clean)
    for i in range(labels.size):
        deg[i] = clean[i] + 0.45 * shared + 0.30 * node_bias[owner[i]]
    deg = l2_normalise(deg + 0.18 * scale * rng.normal(size=deg.shape))

    nodes_meta, test_idx = {}, []
    for k in range(n_nodes):
        ids = list(range(k * ids_per_node, (k + 1) * ids_per_node))
        rows = [i for i in range(labels.size) if int(labels[i]) in ids]
        cut = int(len(rows) * 0.8)
        nodes_meta[f"node_{k}"] = {
            "identities": ids, "train_idx": rows[:cut],
            "val_idx": rows[cut:], "degrade_params": {},
        }
    test_ids = list(range(n_nodes * ids_per_node, total_ids))
    test_idx = [i for i in range(labels.size) if int(labels[i]) in test_ids]

    part = {
        "n_nodes": n_nodes, "seed": 0,
        "test_idx": test_idx, "test_identities": test_ids,
        "test_profile_ids": [owner[i] for i in test_idx],
        "nodes": nodes_meta,
    }
    art = Artefacts(clean, deg, labels, part, {"backbone": "synthetic"})

    out = run_three_way(art, cfg)
    s_rank1 = out["summary"]["rank1"]
    s_auc = out["summary"].get("ver_auc", s_rank1)   # fall back to rank1 if no auc key

    # The self-check only verifies the pipeline runs end-to-end and that the
    # federated result is within a plausible range of the centralised bound.
    # On a toy synthetic dataset the adapter may not always beat the frozen
    # baseline on rank-1 (which can be trivially 1.0 for well-separated
    # identities), so we only assert the sanity-check direction.
    assert s_rank1["centralised"] >= s_rank1["federated"] - 0.05, \
        "federated implausibly exceeded the centralised upper bound"

    print("loop OK")
