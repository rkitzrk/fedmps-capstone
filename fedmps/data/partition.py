"""Non-IID partitioning across federated nodes.

Two axes of heterogeneity, applied together, following the setup in [6]:

  Disjoint identities - no person appears at more than one node. This is both
  realistic (a missing person is not simultaneously enrolled everywhere) and
  it is what makes the federated problem honest: a node genuinely cannot learn
  another node's people.

  Distinct camera profiles - each node draws one fixed degradation
  personality. Node 0's camera is consistently softer, darker or more
  compressed than node 3's, so the embedding shift each node has to correct
  for is different.

A quarter of identities are held out of every node entirely and reserved for
evaluation. Nothing trains on them, at any node, in any mode - which is what
lets the three-way comparison in Phase 2 mean anything.

Optionally the nodes are unbalanced in size, because that is the condition
under which FedProx is worth comparing against FedAvg.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from ..config import Config
from ..contracts import assert_partition
from .degrade import sample_node_profile


def _identity_shares(n_nodes: int, unbalanced: bool, ratio: float) -> np.ndarray:
    """Fraction of identities each node receives. Sums to 1."""
    if not unbalanced or n_nodes == 1:
        return np.full(n_nodes, 1.0 / n_nodes)
    weights = np.linspace(1.0, max(1.0, ratio), n_nodes)
    return weights / weights.sum()


def _split_train_val(indices: np.ndarray, val_frac: float,
                     rng: np.random.Generator) -> Tuple[List[int], List[int]]:
    """Shuffle and cut. Always leaves at least one image on each side."""
    idx = indices.copy()
    rng.shuffle(idx)
    if idx.size < 2:
        return idx.tolist(), []
    n_val = int(round(val_frac * idx.size))
    n_val = int(np.clip(n_val, 1, idx.size - 1))
    return idx[n_val:].tolist(), idx[:n_val].tolist()


def build_partition(labels: np.ndarray, cfg: Config) -> Dict[str, Any]:
    """Build partition.json's contents.

    `labels` is the [N] int32 identity array from the loader. Returns a dict
    that satisfies `contracts.assert_partition`.
    """
    pc = cfg.partition
    rng = np.random.default_rng(cfg.seed)

    n_total = int(labels.shape[0])
    identities = np.unique(labels)
    n_ids = identities.size

    if n_ids < pc.n_nodes + 2:
        raise ValueError(
            f"only {n_ids} identities available but {pc.n_nodes} nodes "
            f"requested; lower partition.n_nodes or loosen the dataset filters"
        )

    # ---- hold out evaluation identities ---------------------------------
    shuffled = identities.copy()
    rng.shuffle(shuffled)

    n_test_ids = int(round(pc.test_identity_frac * n_ids))
    n_test_ids = int(np.clip(n_test_ids, 1, n_ids - pc.n_nodes))
    test_ids = np.sort(shuffled[:n_test_ids])
    node_ids_pool = shuffled[n_test_ids:]

    # ---- distribute the rest across nodes -------------------------------
    shares = _identity_shares(pc.n_nodes, pc.unbalanced, pc.unbalance_ratio)
    counts = np.maximum(1, np.floor(shares * node_ids_pool.size).astype(int))
    # Hand any rounding remainder to the largest node.
    counts[-1] += node_ids_pool.size - counts.sum()
    if counts.min() < 1:
        raise ValueError("partition produced an empty node; reduce n_nodes")

    by_label: Dict[int, np.ndarray] = {
        int(i): np.where(labels == i)[0] for i in identities
    }

    nodes: Dict[str, Any] = {}
    cursor = 0
    for k in range(pc.n_nodes):
        ids_k = np.sort(node_ids_pool[cursor:cursor + counts[k]])
        cursor += counts[k]

        img_idx = np.concatenate([by_label[int(i)] for i in ids_k])
        train_idx, val_idx = _split_train_val(img_idx, pc.val_frac, rng)

        nodes[f"node_{k}"] = {
            "identities": ids_k.tolist(),
            "train_idx": sorted(train_idx),
            "val_idx": sorted(val_idx),
            "degrade_params": sample_node_profile(rng),
        }

    # ---- evaluation set --------------------------------------------------
    test_idx = np.concatenate([by_label[int(i)] for i in test_ids])
    test_idx = np.sort(test_idx)

    # Each held-out image is degraded with one node's camera, assigned round
    # robin, so the evaluation set spans every camera type rather than
    # favouring one. Phase 2 can slice on this to report per-camera results.
    test_profile_ids = (np.arange(test_idx.size) % pc.n_nodes).tolist()

    part: Dict[str, Any] = {
        "n_nodes": pc.n_nodes,
        "seed": cfg.seed,
        "test_idx": test_idx.tolist(),
        "test_identities": test_ids.tolist(),
        "test_profile_ids": test_profile_ids,
        "nodes": nodes,
    }

    return assert_partition(part, n_total=n_total)


def profile_for_each_image(part: Dict[str, Any], n_total: int) -> np.ndarray:
    """Map every image index to the node profile it should be degraded with.

    Returns [N] int32. Node-owned images take their own node's camera; held-out
    images take the round-robin assignment. Any image not covered by the
    partition (there should be none) falls back to node 0.
    """
    out = np.zeros(n_total, dtype=np.int32)
    for k in range(part["n_nodes"]):
        entry = part["nodes"][f"node_{k}"]
        for i in entry["train_idx"]:
            out[i] = k
        for i in entry["val_idx"]:
            out[i] = k
    for i, k in zip(part["test_idx"], part["test_profile_ids"]):
        out[i] = k
    return out


def describe_partition(part: Dict[str, Any]) -> str:
    lines = [f"nodes: {part['n_nodes']}   held-out identities: "
             f"{len(part['test_identities'])} "
             f"({len(part['test_idx'])} images)"]
    for k in range(part["n_nodes"]):
        e = part["nodes"][f"node_{k}"]
        prof = e["degrade_params"]
        face = prof.get("face_px", 0.0)
        lines.append(
            f"  node_{k}: {len(e['identities']):>4} ids  "
            f"{len(e['train_idx']):>5} train  {len(e['val_idx']):>4} val   "
            f"camera[face={face:.2f} blur={prof.get('blur_kernel', 0):.2f} "
            f"jpeg={prof.get('jpeg_quality', 0):.2f}]"
        )
    return "\n".join(lines)


if __name__ == "__main__":  # smoke check
    from ..config import default_config

    cfg = default_config()
    cfg.partition.n_nodes = 5

    # 60 identities, 6 images each
    labels = np.repeat(np.arange(60), 6).astype(np.int32)
    part = build_partition(labels, cfg)

    print(describe_partition(part))

    # Disjointness across nodes, and against the held-out set.
    seen: set = set(part["test_idx"])
    node_ids: set = set()
    for k in range(part["n_nodes"]):
        e = part["nodes"][f"node_{k}"]
        idx = set(e["train_idx"]) | set(e["val_idx"])
        assert not (idx & seen), f"node_{k} overlaps an earlier split"
        seen |= idx
        ids = set(e["identities"])
        assert not (ids & node_ids), f"node_{k} shares identities with another node"
        assert not (ids & set(part["test_identities"])), "node holds a test identity"
        node_ids |= ids

    assert len(seen) == labels.size, f"{len(seen)} of {labels.size} images assigned"

    prof = profile_for_each_image(part, labels.size)
    assert prof.shape == (labels.size,)
    assert prof.min() >= 0 and prof.max() < part["n_nodes"]

    # Unbalanced mode really should produce unequal nodes.
    sizes = [len(part["nodes"][f"node_{k}"]["identities"])
             for k in range(part["n_nodes"])]
    print("identities per node:", sizes)
    assert max(sizes) > min(sizes), "unbalanced partition came out balanced"

    print("partition OK")
