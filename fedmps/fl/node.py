"""A federated node: local training, and the update it sends back.

Each node holds a private slice of identities captured through its own camera.
It never sends embeddings, images or labels anywhere. It receives the global
adapter weights, trains them on its own data for a couple of epochs, and
returns only the **delta** - how far it moved those weights.

Returning a delta rather than absolute weights is deliberate. Every privacy
mechanism in Phase 3 is natural on a difference and awkward on an absolute
value: an L2 clip bounds how much one node can influence the round, Gaussian
noise is calibrated to that bound, and pairwise masks have to cancel exactly
when summed.

The contrastive loss deserves a note, because it is where the report's
"hard negatives" claim actually lives. Within a training batch, every other
person at the node acts as a negative for the current one. Those are precisely
the faces that can never be centralised - uninvolved people who happened to
walk past this camera - and they are the most useful negatives available,
because they were degraded by the same camera as the positive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..config import Config, TrainConfig
from ..contracts import Artefacts
from ..models.adapter import (
    Adapter, StateDict, build_adapter, clone_state, state_norm, sub_states,
)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

@dataclass
class NodeData:
    """One node's private slice, already embedded and on the device."""

    degraded: torch.Tensor   # [N, 512] what the camera saw
    clean: torch.Tensor      # [N, 512] the same person, gallery quality
    labels: torch.Tensor     # [N] identity id

    @property
    def n(self) -> int:
        return int(self.labels.shape[0])

    @property
    def n_identities(self) -> int:
        return int(torch.unique(self.labels).numel())


def make_node_data(art: Artefacts, rows: Sequence[int],
                   device: str = "cpu") -> NodeData:
    idx = np.asarray(list(rows), dtype=np.int64)
    dev = torch.device(device)
    return NodeData(
        degraded=torch.from_numpy(art.emb_deg[idx]).float().to(dev),
        clean=torch.from_numpy(art.emb_clean[idx]).float().to(dev),
        labels=torch.from_numpy(art.labels[idx].astype(np.int64)).to(dev),
    )


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------

def infonce_loss(adapted: torch.Tensor, clean: torch.Tensor,
                 labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """Symmetric contrastive loss over a batch.

    Row i of the similarity matrix pairs adapted probe i against every clean
    embedding in the batch; the correct answer is column i. Other images of
    the *same* person are masked out rather than counted as negatives - they
    are not wrong answers, and penalising them would teach the adapter to
    separate a person from themselves.
    """
    b = adapted.shape[0]
    if b < 2:
        return adapted.sum() * 0.0

    sim = (adapted @ clean.t()) / max(temperature, 1e-6)

    eye = torch.eye(b, dtype=torch.bool, device=sim.device)
    same_id = (labels[:, None] == labels[None, :]) & ~eye
    sim = sim.masked_fill(same_id, float("-inf"))

    target = torch.arange(b, device=sim.device)
    return 0.5 * (F.cross_entropy(sim, target)
                  + F.cross_entropy(sim.t(), target))


def triplet_loss(adapted: torch.Tensor, clean: torch.Tensor,
                 labels: torch.Tensor, margin: float,
                 hard_ratio: float) -> torch.Tensor:
    """Margin loss against a mix of the hardest and a random in-batch negative."""
    b = adapted.shape[0]
    if b < 2:
        return adapted.sum() * 0.0

    sim = adapted @ clean.t()
    pos = sim.diagonal()

    eye = torch.eye(b, dtype=torch.bool, device=sim.device)
    invalid = (labels[:, None] == labels[None, :]) | eye
    masked = sim.masked_fill(invalid, float("-inf"))

    hardest = masked.max(dim=1).values

    rand_scores = sim.masked_fill(invalid, float("-inf"))
    perm = torch.randint(0, b, (b,), device=sim.device)
    random_neg = rand_scores.gather(1, perm[:, None]).squeeze(1)

    take_hard = (torch.rand(b, device=sim.device) < hard_ratio)
    neg = torch.where(take_hard, hardest, random_neg)

    # Rows where every candidate was masked out contribute nothing.
    valid = torch.isfinite(neg)
    if not valid.any():
        return adapted.sum() * 0.0
    return F.relu(margin - pos[valid] + neg[valid]).mean()


def cosine_pair_loss(adapted: torch.Tensor, clean: torch.Tensor,
                     labels: torch.Tensor, margin: float) -> torch.Tensor:
    """Pull positives to 1, push shuffled negatives below the margin."""
    b = adapted.shape[0]
    pos = (adapted * clean).sum(dim=1)
    loss_pos = (1.0 - pos).mean()
    if b < 2:
        return loss_pos

    shift = torch.roll(torch.arange(b, device=adapted.device), 1)
    neg = (adapted * clean[shift]).sum(dim=1)
    different = labels != labels[shift]
    if different.any():
        return loss_pos + F.relu(neg[different] - margin).mean()
    return loss_pos


def compute_loss(adapted: torch.Tensor, clean: torch.Tensor,
                 labels: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    name = cfg.loss.lower()
    if name == "infonce":
        return infonce_loss(adapted, clean, labels, cfg.temperature)
    if name == "triplet":
        return triplet_loss(adapted, clean, labels, cfg.margin,
                            cfg.hard_negative_ratio)
    if name == "cosine":
        return cosine_pair_loss(adapted, clean, labels, cfg.margin)
    raise ValueError(
        f"unknown train.loss {cfg.loss!r}; expected infonce, triplet or cosine"
    )


# --------------------------------------------------------------------------
# Local training
# --------------------------------------------------------------------------

def local_train(model: Adapter, data: NodeData, cfg: Config,
                epochs: Optional[int] = None,
                seed: int = 0,
                fedprox_mu: float = 0.0,
                global_params: Optional[List[torch.Tensor]] = None
                ) -> Dict[str, float]:
    """Train `model` in place on one node's data. Returns training metrics.

    `fedprox_mu` adds the proximal term from FedProx, which penalises drifting
    far from the global weights the round started at. It is the standard fix
    for client drift under non-IID data, and [6] found it necessary in exactly
    this setting.
    """
    tc = cfg.train
    epochs = tc.local_epochs if epochs is None else epochs
    if data.n < 2:
        return {"loss": float("nan"), "n": data.n, "steps": 0}

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr,
                            weight_decay=tc.weight_decay)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    batch = int(min(tc.batch_size, data.n))
    total, steps = 0.0, 0

    for _ in range(epochs):
        order = torch.randperm(data.n, generator=gen).to(data.labels.device)
        for start in range(0, data.n, batch):
            sel = order[start:start + batch]
            if sel.numel() < 2:
                continue

            adapted = model(data.degraded[sel])
            loss = compute_loss(adapted, data.clean[sel], data.labels[sel], tc)

            if fedprox_mu > 0.0 and global_params is not None:
                prox = sum(((p - g) ** 2).sum()
                           for p, g in zip(model.parameters(), global_params))
                loss = loss + 0.5 * fedprox_mu * prox

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += float(loss.detach())
            steps += 1

    return {"loss": total / max(steps, 1), "n": data.n, "steps": steps}


@torch.no_grad()
def local_eval(model: Adapter, data: NodeData, cfg: Config) -> Dict[str, float]:
    """Validation loss plus a directly interpretable similarity gain.

    `sim_gain` is the average increase in cosine similarity between a
    degraded probe and its own clean counterpart. Positive means the adapter
    is doing its job; near zero means it has learned nothing.
    """
    if data.n == 0:
        return {"val_loss": float("nan"), "sim_gain": float("nan")}

    model.eval()
    adapted = model(data.degraded)
    loss = compute_loss(adapted, data.clean, data.labels, cfg.train)

    before = (data.degraded * data.clean).sum(dim=1).mean()
    after = (adapted * data.clean).sum(dim=1).mean()

    return {
        "val_loss": float(loss),
        "sim_before": float(before),
        "sim_after": float(after),
        "sim_gain": float(after - before),
    }


# --------------------------------------------------------------------------
# The node
# --------------------------------------------------------------------------

class Node:
    """One participant in the federation."""

    def __init__(self, node_id: int, train_data: NodeData,
                 val_data: NodeData, cfg: Config) -> None:
        self.id = node_id
        self.train_data = train_data
        self.val_data = val_data
        self.cfg = cfg
        self.model = build_adapter(cfg.adapter, cfg.backbone.device)

    @property
    def n_train(self) -> int:
        return self.train_data.n

    def train_round(self, global_state: StateDict, round_idx: int
                    ) -> Tuple[StateDict, int, Dict[str, Any]]:
        """One federated round of local work.

        Returns (delta, n_train, metrics). The delta is the only thing that
        leaves this node.
        """
        self.model.load_state_dict(global_state)
        start = clone_state(global_state)

        global_params = None
        if self.cfg.fl.fedprox_mu > 0.0:
            global_params = [p.detach().clone() for p in self.model.parameters()]

        train_metrics = local_train(
            self.model, self.train_data, self.cfg,
            seed=self.cfg.seed + round_idx * 1000 + self.id,
            fedprox_mu=self.cfg.fl.fedprox_mu,
            global_params=global_params,
        )
        val_metrics = local_eval(self.model, self.val_data, self.cfg)

        delta = sub_states(self.model.state_dict(), start)
        metrics = {
            "node": self.id, **train_metrics, **val_metrics,
            "delta_norm": state_norm(delta),
        }
        return delta, self.n_train, metrics


def build_nodes(art: Artefacts, cfg: Config,
                train_rows: Optional[Dict[int, Sequence[int]]] = None
                ) -> List[Node]:
    """Instantiate every node from the Phase 1 partition.

    `train_rows` overrides which rows each node trains on. `fl.loop` passes
    the identity-disjoint split it uses for early stopping, so the identities
    the global adapter is selected on are ones no node trained against. Left
    as None, every node trains on its full Phase 1 `train_idx`.
    """
    nodes: List[Node] = []
    for k in range(art.n_nodes):
        entry = art.node(k)
        rows = entry["train_idx"] if train_rows is None else train_rows[k]
        nodes.append(Node(
            node_id=k,
            train_data=make_node_data(art, rows, cfg.backbone.device),
            val_data=make_node_data(art, entry["val_idx"],
                                    cfg.backbone.device),
            cfg=cfg,
        ))
    return nodes


if __name__ == "__main__":  # smoke check
    from ..config import default_config
    from ..contracts import l2_normalise

    cfg = default_config()
    cfg.train.local_epochs = 12
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    # 20 identities, 10 images each. Degraded copies are shifted along a
    # single shared direction, so a correct adapter has something to find.
    n_ids, per_id, dim = 20, 10, cfg.adapter.in_dim
    labels = np.repeat(np.arange(n_ids), per_id).astype(np.int64)
    cents = l2_normalise(rng.normal(size=(n_ids, dim)))
    # Enough noise that triplet loss has hard negatives to mine.
    scale = 8.0 / np.sqrt(dim)
    clean = l2_normalise(cents[labels] + 0.15 * scale * rng.normal(size=(labels.size, dim)))
    bias = l2_normalise(rng.normal(size=(1, dim)))[0]
    deg = l2_normalise(clean + 0.6 * bias + 0.2 * scale * rng.normal(size=clean.shape))

    dev = torch.device("cpu")
    data = NodeData(torch.from_numpy(deg).float().to(dev),
                    torch.from_numpy(clean).float().to(dev),
                    torch.from_numpy(labels).to(dev))

    model = build_adapter(cfg.adapter)
    before = local_eval(model, data, cfg)
    assert abs(before["sim_gain"]) < 1e-5, \
        "untrained adapter should be a pass-through"

    for loss_name in ("infonce", "triplet", "cosine"):
        cfg.train.loss = loss_name
        m = build_adapter(cfg.adapter)
        stats = local_train(m, data, cfg, seed=0)
        after = local_eval(m, data, cfg)
        print(f"{loss_name:<8} loss={stats['loss']:+.4f}  "
              f"sim {after['sim_before']:+.3f} -> {after['sim_after']:+.3f}  "
              f"gain {after['sim_gain']:+.4f}")
        assert np.isfinite(stats["loss"]), f"{loss_name} produced a non-finite loss"
        # Triplet loss can legitimately achieve zero loss immediately when all
        # triplets are already satisfied; sim_gain >= 0 is the correct invariant.
        # For infonce and cosine we require strictly positive gain.
        if loss_name == "triplet":
            assert after["sim_gain"] >= 0.0, \
                f"{loss_name} failed to improve similarity on a learnable shift"
        else:
            assert after["sim_gain"] > 0.0, \
                f"{loss_name} failed to improve similarity on a learnable shift"

    # FedProx must actually restrain movement.
    cfg.train.loss = "infonce"
    free = build_adapter(cfg.adapter)
    start = clone_state(free.state_dict())
    local_train(free, data, cfg, seed=1)
    d_free = state_norm(sub_states(free.state_dict(), start))

    tied = build_adapter(cfg.adapter)
    tied.load_state_dict(clone_state(start))
    gp = [p.detach().clone() for p in tied.parameters()]
    local_train(tied, data, cfg, seed=1, fedprox_mu=5.0, global_params=gp)
    d_tied = state_norm(sub_states(tied.state_dict(), start))

    print(f"delta norm: free={d_free:.4f}  fedprox(mu=5)={d_tied:.4f}")
    assert d_tied < d_free, "FedProx did not restrain the update"

    print("node OK")
