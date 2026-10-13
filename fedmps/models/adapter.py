"""The surveillance adapter - the only thing that is ever federated.

A small residual MLP sitting on top of the frozen backbone. It takes the
512-d embedding of a degraded surveillance crop and nudges it back toward
where the same person's clean, gallery-quality embedding lives.

    frozen backbone (43.6 M, never trained)
        -> 512-d degraded embedding
            -> Adapter (this file, ~0.26 M, federated)
                -> 512-d corrected embedding

Why this and not the backbone: ArcFace is trained with a classifier spanning
millions of identities and needs data at that scale. A node has neither. The
degradation shift, on the other hand, is site-specific, low-dimensional, and
learnable from the handful of images a node actually has - which is exactly
the kind of thing federation is for.

This module also owns the state-dict arithmetic (delta, add, scale, flatten)
used by the federated loop in Phase 2 and by the clipping, noise and masking
in Phase 3. Keeping it in one place means those phases never hand-roll it.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AdapterConfig

StateDict = Dict[str, torch.Tensor]


class Adapter(nn.Module):
    """512 -> hidden -> 512, residual, L2-normalised output.

    The output is renormalised because everything downstream compares
    embeddings by cosine similarity. Without it the adapter could trivially
    lower the loss by inflating vector norms rather than by learning anything
    about direction.
    """

    def __init__(self, cfg: AdapterConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.in_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.out_dim),
        )
        self.residual = cfg.residual
        self._init_weights()

    def _init_weights(self) -> None:
        """Start close to the identity map.

        The final layer begins at zero, so an untrained adapter is exactly a
        pass-through. That matters for the report: round 0 of federated
        training reproduces the no-adapter baseline by construction, so any
        improvement afterwards is unambiguously the adapter's doing.
        """
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        out = x + h if self.residual else h
        return F.normalize(out, p=2, dim=-1)

    @torch.no_grad()
    def transform_numpy(self, emb, batch_size: int = 4096):
        """Apply the adapter to a numpy [N, 512] array. Used by evaluation."""
        import numpy as np

        self.eval()
        device = next(self.parameters()).device
        out = np.empty_like(emb, dtype="float32")
        for i in range(0, emb.shape[0], batch_size):
            chunk = torch.from_numpy(emb[i:i + batch_size]).float().to(device)
            out[i:i + batch_size] = self(chunk).cpu().numpy()
        return out

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def payload_bytes(self) -> int:
        """Size of one federated update, as float32 on the wire."""
        return self.n_params() * 4


def build_adapter(cfg: AdapterConfig, device: str = "cpu") -> Adapter:
    return Adapter(cfg).to(torch.device(device))


# --------------------------------------------------------------------------
# State-dict arithmetic
#
# The federated loop moves *deltas* (local weights minus the global weights
# they started from), not absolute weights. Phase 3 clips, noises and masks
# those deltas, all of which are natural on a difference and awkward on an
# absolute value.
# --------------------------------------------------------------------------

def clone_state(state: StateDict) -> StateDict:
    return OrderedDict((k, v.detach().clone()) for k, v in state.items())


def sub_states(a: StateDict, b: StateDict) -> StateDict:
    """a - b, key by key."""
    return OrderedDict((k, a[k].detach() - b[k].detach()) for k in a)


def add_states(a: StateDict, b: StateDict, scale: float = 1.0) -> StateDict:
    """a + scale * b, key by key."""
    return OrderedDict((k, a[k].detach() + scale * b[k].detach()) for k in a)


def scale_state(a: StateDict, scale: float) -> StateDict:
    return OrderedDict((k, a[k].detach() * scale) for k in a)


def zeros_like_state(a: StateDict) -> StateDict:
    return OrderedDict((k, torch.zeros_like(v)) for k, v in a.items())


def weighted_mean_states(states: List[StateDict],
                         weights: List[float]) -> StateDict:
    """Weighted average. This is FedAvg's one line of actual mathematics."""
    if not states:
        raise ValueError("no states to average")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weights must sum to a positive number")

    out = zeros_like_state(states[0])
    for state, w in zip(states, weights):
        f = w / total
        for k in out:
            out[k] = out[k] + state[k].detach() * f
    return out


def state_to_vector(state: StateDict) -> torch.Tensor:
    """Flatten to a single 1-D tensor, in sorted key order.

    Phase 3 needs this: an L2 clip is defined over the whole update, not
    per-tensor, and a masking scheme needs one contiguous buffer.
    """
    return torch.cat([state[k].detach().reshape(-1) for k in sorted(state)])


def vector_to_state(vec: torch.Tensor, reference: StateDict) -> StateDict:
    """Inverse of `state_to_vector`, using `reference` for shapes."""
    out: StateDict = OrderedDict()
    offset = 0
    for k in sorted(reference):
        ref = reference[k]
        n = ref.numel()
        out[k] = vec[offset:offset + n].reshape(ref.shape).to(ref.dtype)
        offset += n
    if offset != vec.numel():
        raise ValueError(
            f"vector has {vec.numel()} elements but reference needs {offset}"
        )
    # Restore the reference's key order, which state_dict loading expects.
    return OrderedDict((k, out[k]) for k in reference)


def state_norm(state: StateDict) -> float:
    """L2 norm of the whole update. The quantity DP clipping bounds."""
    return float(torch.linalg.vector_norm(state_to_vector(state)).item())


if __name__ == "__main__":  # smoke check
    import numpy as np

    from ..config import default_config

    cfg = default_config()
    torch.manual_seed(0)

    model = build_adapter(cfg.adapter)
    print(f"adapter params : {model.n_params():,}")
    print(f"payload        : {model.payload_bytes() / 1e6:.2f} MB per round")

    x = F.normalize(torch.randn(8, cfg.adapter.in_dim), dim=-1)
    y = model(x)
    assert y.shape == x.shape
    assert torch.allclose(y.norm(dim=-1), torch.ones(8), atol=1e-5), \
        "adapter output is not L2-normalised"

    # Zero-initialised final layer => an untrained adapter is a pass-through.
    assert torch.allclose(y, x, atol=1e-5), \
        "untrained adapter is not the identity map"

    # State arithmetic round-trip.
    s0 = clone_state(model.state_dict())
    for p in model.parameters():
        with torch.no_grad():
            p.add_(torch.randn_like(p) * 0.01)
    s1 = clone_state(model.state_dict())

    delta = sub_states(s1, s0)
    restored = add_states(s0, delta)
    for k in s1:
        assert torch.allclose(s1[k], restored[k], atol=1e-6), f"round trip failed at {k}"

    vec = state_to_vector(delta)
    back = vector_to_state(vec, delta)
    for k in delta:
        assert torch.allclose(delta[k], back[k], atol=1e-8), f"flatten failed at {k}"
    assert list(back.keys()) == list(delta.keys()), "key order not preserved"
    print(f"delta norm     : {state_norm(delta):.4f}")

    # FedAvg of identical states must be that state.
    avg = weighted_mean_states([s1, s1, s1], [1.0, 2.0, 3.0])
    for k in s1:
        assert torch.allclose(avg[k], s1[k], atol=1e-6), "weighted mean is wrong"

    # numpy transform path
    emb = np.ascontiguousarray(
        F.normalize(torch.randn(20, cfg.adapter.in_dim), dim=-1).numpy())
    out = model.transform_numpy(emb)
    assert out.shape == emb.shape and out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-4)

    print("adapter OK")
