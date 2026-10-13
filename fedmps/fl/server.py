"""The aggregation server.

It holds the global adapter weights, collects one delta per node per round,
combines them, and hands the result back out. It never sees an image, an
embedding, a label, or an identity.

Two privacy hooks live here. Both are written now, in Phase 2, as
pass-throughs, and Phase 3 fills in their bodies without restructuring
anything around them:

    apply_privacy(update, cfg)
        Per-node, applied at the node before the update is released.
        Phase 3: L2 clipping to a fixed norm, then Gaussian noise.

    secure_aggregate(updates, weights, cfg)
        Across nodes. Phase 3: pairwise cancelling masks, so the server can
        compute the weighted sum without any individual update ever being
        readable to it.

Keeping the second hook separate matters. Per-update noise and cross-update
masking are different operations - one bounds influence, the other hides
identity - and collapsing them into a single function would force a rewrite
of this file in Phase 3.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from ..config import Config
from ..models.adapter import (
    Adapter, StateDict, add_states, build_adapter, clone_state, state_norm,
    weighted_mean_states,
)
from ..privacy.dp import compute_epsilon, privatise
from ..privacy.mask import SecureAggregator

Update = Tuple[StateDict, int]   # (delta, n_train_samples)


# --------------------------------------------------------------------------
# Privacy hooks - bodies land in Phase 3, signatures are frozen now
# --------------------------------------------------------------------------

def apply_privacy(update: StateDict, cfg: Config,
                  n_participants: int = 1,
                  generator: Optional[torch.Generator] = None) -> StateDict:
    """Per-node protection, applied before an update leaves the node.

    Clips the update's L2 norm to `cfg.privacy.clip_norm`, then adds Gaussian
    noise. When masking is also on, the noise is divided by sqrt(n) across
    participants so the aggregate carries exactly the intended scale - see
    privacy/dp.py for why that combination is the point rather than a
    coincidence.

    With DP disabled this is the identity, so the Phase 2 federated baseline
    stays directly comparable against the protected runs.
    """
    if not cfg.privacy.enable_dp:
        return update
    return privatise(update, cfg, n_participants=n_participants,
                     generator=generator)


def secure_aggregate(updates: Sequence[StateDict], weights: Sequence[float],
                     cfg: Config,
                     node_ids: Optional[Sequence[int]] = None,
                     round_idx: int = 0,
                     aggregator: Optional[SecureAggregator] = None,
                     dropped: Sequence[int] = ()) -> StateDict:
    """Combine node updates into one.

    With masking off this is a plain weighted mean - FedAvg.

    With masking on, each pair of nodes contributes a shared pseudo-random
    mask that one adds and the other subtracts, so every mask vanishes in the
    sum while no individual update is readable in transit. The arithmetic
    result is identical either way, which is exactly the claim worth making:
    secure aggregation costs latency, not accuracy.
    """
    if not cfg.privacy.enable_masking:
        return weighted_mean_states(list(updates), list(weights))
    if aggregator is None:
        raise ValueError(
            "masking is enabled but no SecureAggregator was supplied; "
            "the Server builds one for you - call Server.aggregate instead"
        )
    return aggregator.aggregate(updates, weights, node_ids=node_ids,
                                round_idx=round_idx, dropped=dropped)


# --------------------------------------------------------------------------
# Server
# --------------------------------------------------------------------------

class Server:
    """Holds the global adapter and runs aggregation."""

    def __init__(self, cfg: Config, n_nodes: int = 0) -> None:
        self.cfg = cfg
        self.model: Adapter = build_adapter(cfg.adapter, cfg.backbone.device)
        self.global_state: StateDict = clone_state(self.model.state_dict())
        self.round: int = 0
        self.n_nodes = n_nodes
        # Key agreement happens once, at setup, not per round. Built lazily so
        # a run without masking never pays for the modular exponentiations.
        self.secagg: Optional[SecureAggregator] = None
        if cfg.privacy.enable_masking and n_nodes > 0:
            self.secagg = SecureAggregator(n_nodes, cfg)

    # -- state -------------------------------------------------------------

    def broadcast(self) -> StateDict:
        """The weights sent out at the start of a round."""
        return clone_state(self.global_state)

    def load_into_model(self, state: Optional[StateDict] = None) -> Adapter:
        """Return the server's model loaded with `state` (default: global)."""
        self.model.load_state_dict(state or self.global_state)
        self.model.eval()
        return self.model

    # -- aggregation -------------------------------------------------------

    def aggregate(self, updates: List[Update],
                  node_ids: Optional[Sequence[int]] = None,
                  dropped: Sequence[int] = ()) -> Dict[str, Any]:
        """Apply one round of updates to the global weights.

        FedAvg weights each node by how much data it holds, so a node with
        three images does not count as much as one with three hundred.

        `node_ids` identifies which nodes these updates came from. Masking
        needs it: which pairwise masks a node carries depends on its index.
        """
        if not updates:
            raise ValueError("no updates to aggregate")

        ids = list(node_ids) if node_ids is not None else list(range(len(updates)))
        n_part = len(updates) + len(dropped)

        if self.cfg.privacy.enable_masking and self.secagg is None:
            self.secagg = SecureAggregator(max(self.n_nodes, n_part), self.cfg)

        raw_norms = [state_norm(d) for d, _ in updates]

        protected = [
            apply_privacy(delta, self.cfg, n_participants=n_part)
            for delta, _ in updates
        ]
        weights = [float(n) for _, n in updates]
        sent_norms = [state_norm(d) for d in protected]

        mean_delta = secure_aggregate(
            protected, weights, self.cfg, node_ids=ids,
            round_idx=self.round, aggregator=self.secagg, dropped=dropped)

        self.global_state = add_states(self.global_state, mean_delta)
        self.round += 1

        clip = self.cfg.privacy.clip_norm
        stats: Dict[str, Any] = {
            "round": self.round,
            "n_updates": len(updates),
            "total_samples": int(sum(weights)),
            "mean_delta_norm_raw": float(sum(raw_norms) / len(raw_norms)),
            "median_delta_norm_raw": float(sorted(raw_norms)[len(raw_norms) // 2]),
            "mean_delta_norm_sent": float(sum(sent_norms) / len(sent_norms)),
            "aggregated_delta_norm": state_norm(mean_delta),
            "global_norm": state_norm(self.global_state),
        }
        if self.cfg.privacy.enable_dp:
            # If almost nothing is being clipped, C is loose and the noise is
            # larger than it needs to be. If everything is clipped hard, C is
            # throttling learning. The median raw norm is what to set C near.
            stats["fraction_clipped"] = float(
                sum(1 for n in raw_norms if n > clip) / len(raw_norms))
        if self.cfg.privacy.enable_masking and self.secagg is not None:
            stats["secagg"] = {
                k: v for k, v in self.secagg.last_stats.items()
                if k != "masked_preview"
            }
        return stats

    def privacy_report(self, rounds: Optional[int] = None) -> Dict[str, Any]:
        """The (epsilon, delta) spent, and what protects what."""
        pc = self.cfg.privacy
        subsampled = bool(self.cfg.fl.clients_per_round
                          and self.cfg.fl.clients_per_round < self.n_nodes)
        report: Dict[str, Any] = {
            "dp_enabled": pc.enable_dp,
            "masking_enabled": pc.enable_masking,
            "regime": ("distributed DP (noise split across nodes, server sees "
                       "only the masked sum)" if pc.enable_dp and pc.enable_masking
                       else "local DP (server sees each noisy update)"
                       if pc.enable_dp
                       else "no formal guarantee"),
        }
        if pc.enable_dp:
            report["accounting"] = compute_epsilon(
                pc, rounds if rounds is not None else self.round,
                subsampled=subsampled)
        return report

    # -- reporting ---------------------------------------------------------

    def payload_report(self) -> Dict[str, Any]:
        """The communication numbers the report's central claim rests on.

        The comparison against a federated backbone is the whole argument for
        freezing it, so it is computed here rather than quoted from memory.
        """
        params = self.model.n_params()
        per_round = params * 4  # float32 on the wire
        backbone_params = 43_600_000  # ArcFace R50, for contrast
        return {
            "adapter_params": params,
            "bytes_per_update": per_round,
            "mb_per_update": per_round / 1e6,
            "backbone_params_if_federated": backbone_params,
            "backbone_mb_if_federated": backbone_params * 4 / 1e6,
            "reduction_factor": backbone_params / max(params, 1),
        }


if __name__ == "__main__":  # smoke check
    from ..config import default_config
    from ..models.adapter import sub_states, zeros_like_state

    cfg = default_config()
    torch.manual_seed(0)

    server = Server(cfg)
    rep = server.payload_report()
    print(f"adapter params    : {rep['adapter_params']:,}")
    print(f"per-round payload : {rep['mb_per_update']:.2f} MB")
    print(f"if backbone were federated: {rep['backbone_mb_if_federated']:.0f} MB"
          f"  ({rep['reduction_factor']:.0f}x larger)")

    base = server.broadcast()

    # Three nodes each pushing the same delta: the mean must equal it, and the
    # global weights must move by exactly that amount.
    delta = zeros_like_state(base)
    for k in delta:
        delta[k] = torch.full_like(delta[k], 0.01)

    before = clone_state(server.global_state)
    stats = server.aggregate([(clone_state(delta), 100) for _ in range(3)])
    moved = sub_states(server.global_state, before)
    for k in moved:
        assert torch.allclose(moved[k], delta[k], atol=1e-6), \
            f"aggregation moved the global weights incorrectly at {k}"
    print("identical updates -> exact mean OK")

    # Sample-count weighting: a node with 10x the data should dominate.
    server2 = Server(cfg)
    small = zeros_like_state(base)
    big = zeros_like_state(base)
    for k in small:
        small[k] = torch.full_like(small[k], 1.0)
        big[k] = torch.full_like(big[k], 0.0)

    before2 = clone_state(server2.global_state)
    server2.aggregate([(clone_state(small), 10), (clone_state(big), 90)])
    moved2 = sub_states(server2.global_state, before2)
    key = next(iter(moved2))
    got = float(moved2[key].flatten()[0])
    assert abs(got - 0.1) < 1e-6, f"expected 0.1 from a 10/90 split, got {got}"
    print("sample-count weighting OK")

    print(f"round stats: {dict(list(stats.items())[:5])}")

    # -- Phase 3 hooks, now live ------------------------------------------

    # DP: clipping must bound every update that leaves a node.
    cfg.privacy.enable_dp = True
    cfg.privacy.clip_norm = 0.5
    cfg.privacy.noise_multiplier = 0.0      # clipping alone, so the check is exact
    big = zeros_like_state(base)
    for k in big:
        big[k] = torch.randn_like(big[k]) * 0.5
    protected = apply_privacy(big, cfg, n_participants=5)
    print(f"\nclipping: {state_norm(big):.4f} -> {state_norm(protected):.4f}")
    assert state_norm(protected) <= 0.5 + 1e-4, "DP hook failed to clip"

    # Masking: the aggregate must be unchanged, to floating-point accuracy.
    cfg.privacy.enable_dp = False
    cfg.privacy.enable_masking = True
    cfg.privacy.dh_bits = 1024
    n = 4
    s_plain = Server(cfg, n_nodes=n)
    s_plain.cfg = default_config()          # masking off for the reference run

    ups = []
    for k in range(n):
        u = zeros_like_state(base)
        for key in u:
            u[key] = torch.randn_like(u[key]) * 0.01
        ups.append((u, 40 + 20 * k))

    ref = weighted_mean_states([u for u, _ in ups], [float(w) for _, w in ups])

    s_mask = Server(cfg, n_nodes=n)
    start = clone_state(s_mask.global_state)
    s_mask.aggregate([(clone_state(u), w) for u, w in ups],
                     node_ids=list(range(n)))
    moved = sub_states(s_mask.global_state, start)

    worst = max(float((ref[k] - moved[k]).abs().max()) for k in ref)
    print(f"masked aggregation vs plain mean, max difference: {worst:.3e}")
    assert worst < 1e-5, "secure aggregation changed the result"

    rep = s_mask.privacy_report(rounds=1)
    print(f"privacy regime: {rep['regime']}")

    # And with both on, an epsilon must actually be reported.
    cfg.privacy.enable_dp = True
    cfg.privacy.noise_multiplier = 1.0
    s_both = Server(cfg, n_nodes=n)
    s_both.aggregate([(clone_state(u), w) for u, w in ups],
                     node_ids=list(range(n)))
    acc = s_both.privacy_report(rounds=30)["accounting"]
    print(f"epsilon over 30 rounds: {acc['epsilon']:.3f} at delta={acc['delta']}")
    assert acc["epsilon"] < float("inf")

    print("server OK")
