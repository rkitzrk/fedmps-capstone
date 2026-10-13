"""Secure aggregation by pairwise cancelling masks.

The server needs the *sum* of the nodes' updates. It has no business seeing
any individual one. Pairwise masking gives exactly that, and it is a genuine
secure multi-party computation rather than a gesture at one.

The construction:

  1. Every pair of nodes (i, j) agrees a shared secret over Diffie-Hellman.
     Neither the server nor any third node can derive it.
  2. From that secret and the round number they derive one pseudo-random mask
     vector m_ij, the same on both sides.
  3. The lower-indexed node adds +m_ij to its update; the higher-indexed node
     subtracts it. Across the full set every mask appears once with each sign.
  4. The server sums what it receives. Every mask cancels, and the sum is
     exact - while any single masked update, viewed alone, is indistinguishable
     from noise.

Crucially the arithmetic result is identical to the unmasked mean. Secure
aggregation costs latency and a key-exchange round, not accuracy. The
experiments should show precisely that, and it is a cleaner claim than a
hand-wave about cryptography being "roughly free".

Weighting is handled by masking the *weighted* update w_i * u_i rather than
u_i itself. Masks then cancel unweighted, and the server divides by the total
weight at the end. Masking the unweighted update instead would leave a
residue of (w_i - w_j) * m_ij, which does not vanish.

Honest limitation, and it belongs in the report's threat model: this is
single-masked aggregation. The dropout recovery below asks survivors to reveal
the masks they shared with a node that vanished. A server that accepts a
node's update and *then* declares it dropped could therefore unmask it. Real
deployments add a second self-mask, shared through Shamir secret sharing, so
that recovery reveals either the pairwise masks or the self-mask but never
both. That is a known extension, deliberately out of scope here, and saying so
is stronger than pretending the gap is not there.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..config import Config
from ..models.adapter import StateDict, state_to_vector, vector_to_state

# RFC 3526 group 14 (2048-bit MODP) and RFC 2409 group 2 (1024-bit MODP).
# The smoke check below runs Miller-Rabin on whichever is selected, so a
# mistyped digit fails loudly rather than silently weakening the exchange.
_P2048 = int(
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1"
    "29024E088A67CC74020BBEA63B139B22514A08798E3404DD"
    "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245"
    "E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
    "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3D"
    "C2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F"
    "83655D23DCA3AD961C62F356208552BB9ED529077096966D"
    "670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B"
    "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9"
    "DE2BCBF6955817183995497CEA956AE515D2261898FA0510"
    "15728E5A8AACAA68FFFFFFFFFFFFFFFF", 16)

_P1024 = int(
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1"
    "29024E088A67CC74020BBEA63B139B22514A08798E3404DD"
    "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245"
    "E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
    "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE65381"
    "FFFFFFFFFFFFFFFF", 16)

_GENERATOR = 2


def _prime(bits: int) -> int:
    return _P1024 if bits <= 1024 else _P2048


# --------------------------------------------------------------------------
# Key agreement
# --------------------------------------------------------------------------

@dataclass
class NodeKeypair:
    """One node's Diffie-Hellman keypair. The private half never leaves it."""

    node_id: int
    private: int
    public: int
    prime: int

    def shared_secret(self, other_public: int) -> int:
        return pow(other_public, self.private, self.prime)


def generate_keypair(node_id: int, bits: int, rng: np.random.Generator
                     ) -> NodeKeypair:
    p = _prime(bits)
    # A 256-bit exponent is ample against the best known discrete-log attacks
    # on these groups, and keeps the modular exponentiation quick.
    private = int.from_bytes(rng.bytes(32), "big") % (p - 2) + 2
    return NodeKeypair(node_id, private, pow(_GENERATOR, private, p), p)


def pair_seed(secret: int, round_idx: int, lo: int, hi: int) -> int:
    """Derive a mask seed from a shared secret, the round, and the pair.

    Including the round number means masks never repeat across rounds, which
    matters: a mask reused between two rounds would let the server difference
    them and recover the update.
    """
    material = (secret.to_bytes((secret.bit_length() + 7) // 8 or 1, "big")
                + round_idx.to_bytes(8, "big")
                + lo.to_bytes(4, "big") + hi.to_bytes(4, "big"))
    return int.from_bytes(hashlib.sha256(material).digest(), "big")


def mask_vector(seed: int, size: int) -> np.ndarray:
    """The pseudo-random pad. float64 so cancellation is exact to ~1e-16.

    Element magnitudes are around 1.0 while a clipped update's elements are
    around 0.002, so a masked update carries no visible trace of the update.
    """
    return np.random.default_rng(seed).standard_normal(size)


# --------------------------------------------------------------------------
# The aggregator
# --------------------------------------------------------------------------

class SecureAggregator:
    """Runs the masking protocol across a fixed set of nodes."""

    def __init__(self, n_nodes: int, cfg: Config) -> None:
        self.n_nodes = n_nodes
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed + 4242)
        self.keys: List[NodeKeypair] = [
            generate_keypair(i, cfg.privacy.dh_bits, rng) for i in range(n_nodes)
        ]
        self.publics: List[int] = [k.public for k in self.keys]
        self._secret_cache: Dict[Tuple[int, int], int] = {}
        self.last_stats: Dict[str, Any] = {}

    # -- secrets -----------------------------------------------------------

    def secret(self, i: int, j: int) -> int:
        """Shared secret for the pair, computed from i's side.

        Both sides derive the same value; the cache just avoids recomputing a
        modular exponentiation we have already done.
        """
        key = (min(i, j), max(i, j))
        if key not in self._secret_cache:
            self._secret_cache[key] = self.keys[key[0]].shared_secret(
                self.publics[key[1]])
        return self._secret_cache[key]

    def node_mask(self, node: int, participants: Sequence[int],
                  round_idx: int, size: int) -> np.ndarray:
        """Sum of this node's pairwise masks, signed by index order."""
        total = np.zeros(size, dtype=np.float64)
        for other in participants:
            if other == node:
                continue
            lo, hi = min(node, other), max(node, other)
            m = mask_vector(pair_seed(self.secret(lo, hi), round_idx, lo, hi),
                            size)
            total += m if node == lo else -m
        return total

    # -- the protocol ------------------------------------------------------

    def aggregate(self, updates: Sequence[StateDict],
                  weights: Sequence[float],
                  node_ids: Optional[Sequence[int]] = None,
                  round_idx: int = 0,
                  dropped: Sequence[int] = ()) -> StateDict:
        """Masked weighted mean. Numerically identical to the plain mean.

        `dropped` names nodes that were expected but never delivered. Their
        pairwise masks are reconstructed and removed - see the limitation in
        the module docstring.
        """
        if not updates:
            raise ValueError("no updates to aggregate")
        t0 = time.perf_counter()

        ids = list(node_ids) if node_ids is not None else list(range(len(updates)))
        participants = list(ids) + list(dropped)
        reference = updates[0]
        size = int(state_to_vector(reference).numel())

        total_weight = float(sum(weights))
        if total_weight <= 0:
            raise ValueError("weights must sum to a positive number")

        masked_sum = np.zeros(size, dtype=np.float64)
        masked_examples: List[np.ndarray] = []

        for update, w, node in zip(updates, weights, ids):
            weighted = state_to_vector(update).double().numpy() * float(w)
            masked = weighted + self.node_mask(node, participants, round_idx, size)
            masked_sum += masked
            if len(masked_examples) < 2:
                masked_examples.append(masked[:8].copy())

        # Recovery: a node that never delivered still had masks assigned to it
        # by everyone else, so those must be cancelled by hand.
        for d in dropped:
            masked_sum += self.node_mask(d, participants, round_idx, size)

        mean = masked_sum / total_weight
        out = vector_to_state(
            torch.from_numpy(mean).float(), reference)

        self.last_stats = {
            "n_updates": len(updates),
            "n_dropped": len(dropped),
            "vector_size": size,
            "pairwise_masks": len(participants) * (len(participants) - 1) // 2,
            "seconds": round(time.perf_counter() - t0, 4),
            "masked_preview": [list(np.round(e, 4)) for e in masked_examples],
        }
        return out

    # -- reporting ---------------------------------------------------------

    def masked_update_preview(self, update: StateDict, node: int,
                              participants: Sequence[int],
                              round_idx: int = 0, n: int = 8) -> Dict[str, Any]:
        """Side-by-side raw and masked values, for the report.

        A screenshot of this is worth a paragraph of prose about secure
        aggregation: the masked row is what the server actually receives.
        """
        vec = state_to_vector(update).double().numpy()
        masked = vec + self.node_mask(node, participants, round_idx, vec.size)
        return {
            "node": node,
            "raw": [round(float(x), 6) for x in vec[:n]],
            "masked": [round(float(x), 6) for x in masked[:n]],
            "raw_norm": float(np.linalg.norm(vec)),
            "masked_norm": float(np.linalg.norm(masked)),
        }


def _is_probable_prime(n: int, rounds: int = 16) -> bool:
    """Miller-Rabin. Guards against a mistyped digit in the group constants."""
    if n < 2 or n % 2 == 0:
        return n == 2
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)[:rounds]:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


if __name__ == "__main__":  # smoke check
    from ..config import default_config
    from ..models.adapter import build_adapter, clone_state, weighted_mean_states

    cfg = default_config()
    torch.manual_seed(0)

    print("verifying the Diffie-Hellman group constants...")
    assert _is_probable_prime(_P1024), "the 1024-bit modulus is not prime"
    assert _is_probable_prime(_P2048), "the 2048-bit modulus is not prime"
    print(f"  1024-bit modulus prime, {_P1024.bit_length()} bits")
    print(f"  2048-bit modulus prime, {_P2048.bit_length()} bits")

    n_nodes = 5
    cfg.privacy.dh_bits = 1024          # keep the smoke check quick
    agg = SecureAggregator(n_nodes, cfg)

    # Key agreement: both sides of a pair must land on the same secret.
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            si = agg.keys[i].shared_secret(agg.publics[j])
            sj = agg.keys[j].shared_secret(agg.publics[i])
            assert si == sj, f"key agreement failed for pair ({i},{j})"
    print("pairwise key agreement OK")

    model = build_adapter(cfg.adapter)
    base = clone_state(model.state_dict())
    updates, weights = [], []
    for k in range(n_nodes):
        u = clone_state(base)
        for key in u:
            u[key] = torch.randn_like(u[key]) * 0.01
        updates.append(u)
        weights.append(float(50 + 30 * k))

    plain = weighted_mean_states(updates, weights)
    masked = agg.aggregate(updates, weights, round_idx=1)

    worst = max(float((plain[k] - masked[k]).abs().max()) for k in plain)
    print(f"masked vs plain mean, max element difference: {worst:.3e}")
    assert worst < 1e-5, "masks did not cancel; the aggregate is wrong"

    # A masked update must look like noise on its own.
    prev = agg.masked_update_preview(updates[0], 0, list(range(n_nodes)),
                                     round_idx=1)
    print(f"\nnode 0 raw    : {prev['raw'][:4]}")
    print(f"node 0 masked : {prev['masked'][:4]}")
    print(f"norm {prev['raw_norm']:.4f} -> {prev['masked_norm']:.1f}")
    assert prev["masked_norm"] > 50 * prev["raw_norm"], \
        "the mask does not dominate the update; it would be readable"

    # Masks must differ between rounds, or they could be differenced away.
    m1 = agg.node_mask(0, list(range(n_nodes)), 1, 16)
    m2 = agg.node_mask(0, list(range(n_nodes)), 2, 16)
    assert not np.allclose(m1, m2), "masks repeated across rounds"
    print("per-round mask freshness OK")

    # Dropout recovery.
    survivors = updates[:4]
    surv_w = weights[:4]
    plain_s = weighted_mean_states(survivors, surv_w)
    masked_s = agg.aggregate(survivors, surv_w, node_ids=[0, 1, 2, 3],
                             round_idx=3, dropped=[4])
    worst_s = max(float((plain_s[k] - masked_s[k]).abs().max()) for k in plain_s)
    print(f"after one dropout, max difference: {worst_s:.3e}")
    assert worst_s < 1e-5, "dropout recovery failed to cancel the masks"

    print(f"\nlast round stats: {dict(list(agg.last_stats.items())[:5])}")
    print("mask OK")
