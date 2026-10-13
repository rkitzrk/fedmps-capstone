"""Cancelable biometric templates for the watchlist.

The server never sends a missing person's photograph to a node. It sends a
96-bit keyed code, derived like this, following [2]:

    512-d embedding
      -> keep the k most discriminative dimensions   (dimension selection)
      -> threshold each against its mean             (binarisation)
      -> XOR with a key derived from case and node   (the cancelable part)
      -> 96 bits, 12 bytes on the wire

Three properties come out of it:

  Revocable    If a node is compromised, reissue the case under a new key.
               Every template that node holds becomes useless. A leaked face
               embedding, by contrast, is leaked forever - you cannot reissue
               someone's face.

  Unlinkable   The same person enrolled at two nodes yields two unrelated bit
               patterns, because the keys differ. Nobody can correlate
               watchlists across sites to learn who is being searched for.

  Hard to      96 thresholded bits out of 512 floats discards most of the
  invert       information, and the key hides what is left.

Matching still works because XOR with a common key cancels inside a Hamming
distance: d(a^k, b^k) = d(a, b). The node applies the same key to its live
embedding, compares, and never needs to undo the protection.

The honest limitation, which belongs in the report's threat model: this
defends a template database at rest and defeats cross-site linkage. It does
not defend a *live, fully compromised node*, which holds the key and sees its
own camera feed anyway. Cancelable templates are the right tool for the first
two threats and the wrong tool for the third.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------
# Dimension selection
# --------------------------------------------------------------------------

@dataclass
class TemplateSelector:
    """Which embedding dimensions to keep, and where to threshold them.

    Fitted once on training data and then fixed. It carries no personal
    information - just indices and per-dimension means - so it can be shipped
    to every node in the clear.
    """

    dims: np.ndarray        # [k] int32, the chosen dimension indices
    thresholds: np.ndarray  # [k] float32, per-dimension cut point
    n_bits: int
    criterion: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dims": self.dims.tolist(),
            "thresholds": [round(float(t), 6) for t in self.thresholds],
            "n_bits": self.n_bits,
            "criterion": self.criterion,
        }


def fit_selector(embeddings: np.ndarray, n_bits: int,
                 labels: Optional[np.ndarray] = None) -> TemplateSelector:
    """Choose the `n_bits` most discriminative dimensions.

    With labels, dimensions are ranked by a Fisher-style ratio: how much the
    identity means differ, relative to the scatter within each identity. That
    is a much better criterion than raw variance, which happily selects a
    dimension that varies wildly for reasons unrelated to who the person is.
    Without labels it falls back to variance.
    """
    if n_bits % 8 != 0:
        raise ValueError(f"n_bits must be a multiple of 8, got {n_bits}")
    if n_bits > embeddings.shape[1]:
        raise ValueError(
            f"cannot select {n_bits} bits from {embeddings.shape[1]} dimensions")

    if labels is not None and np.unique(labels).size >= 2:
        idents = np.unique(labels)
        means = np.stack([embeddings[labels == i].mean(axis=0) for i in idents])
        within = np.stack([embeddings[labels == i].var(axis=0) for i in idents])
        score = means.var(axis=0) / (within.mean(axis=0) + 1e-8)
        criterion = "fisher"
    else:
        score = embeddings.var(axis=0)
        criterion = "variance"

    dims = np.sort(np.argsort(-score)[:n_bits]).astype(np.int32)
    thresholds = embeddings[:, dims].mean(axis=0).astype(np.float32)
    return TemplateSelector(dims, thresholds, n_bits, criterion)


# --------------------------------------------------------------------------
# Binarisation and keying
# --------------------------------------------------------------------------

def binarise(embeddings: np.ndarray, selector: TemplateSelector) -> np.ndarray:
    """[N, 512] float -> [N, k] bool."""
    emb = np.atleast_2d(embeddings)
    return emb[:, selector.dims] > selector.thresholds[None, :]


def derive_key(master: str, case_id: str, node_id: int, n_bits: int
               ) -> np.ndarray:
    """Per-case, per-node key as a [n_bits] bool array.

    Both halves matter. Per-case means closing a case retires its key. Per-node
    means two sites holding the same person store unrelated bit patterns.
    """
    material = f"{master}|{case_id}|{node_id}".encode("utf-8")
    need = (n_bits + 7) // 8
    stream = b""
    counter = 0
    while len(stream) < need:
        stream += hashlib.sha256(material + counter.to_bytes(4, "big")).digest()
        counter += 1
    bits = np.unpackbits(np.frombuffer(stream[:need], dtype=np.uint8))
    return bits[:n_bits].astype(bool)


def protect(bits: np.ndarray, key: np.ndarray) -> np.ndarray:
    """XOR the bits with the key. Self-inverse, so the same call undoes it."""
    return np.logical_xor(bits, key[None, :] if bits.ndim == 2 else key)


def pack(bits: np.ndarray) -> np.ndarray:
    """Bool bits -> the uint8 wire format. [k] -> [k/8], or [N,k] -> [N,k/8]."""
    return np.packbits(bits.astype(np.uint8), axis=-1)


def unpack(packed: np.ndarray, n_bits: int) -> np.ndarray:
    return np.unpackbits(packed, axis=-1)[..., :n_bits].astype(bool)


def make_template(embedding: np.ndarray, selector: TemplateSelector,
                  key: np.ndarray) -> np.ndarray:
    """One embedding to one protected, packed template. [12] uint8 at 96 bits."""
    bits = binarise(np.atleast_2d(embedding), selector)
    return pack(protect(bits, key))[0]


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------

def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances between two sets of bit vectors.

    [N, k] and [M, k] bool -> [N, M] int. Computed as a pair of matrix
    products, which is far quicker than looping once k and N get real.
    """
    af = a.astype(np.float32)
    bf = b.astype(np.float32)
    agree = af @ bf.T + (1.0 - af) @ (1.0 - bf).T
    return (a.shape[1] - agree).astype(np.int32)


def similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamming distance mapped onto the [-1, 1] scale cosine uses.

    Identical codes score 1.0, unrelated codes score about 0.0. That means the
    open-set evaluation in eval/metrics.py can score templates without knowing
    it is looking at bits rather than floats.
    """
    k = a.shape[1]
    return (1.0 - 2.0 * hamming_matrix(a, b).astype(np.float32) / k)


def match_score(a: np.ndarray, b: np.ndarray) -> float:
    """Single-pair convenience wrapper, for the node pipeline in Phase 4."""
    return float(similarity_matrix(np.atleast_2d(a), np.atleast_2d(b))[0, 0])


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def save_selector(selector: TemplateSelector, path) -> None:
    """Persist a fitted selector. It holds no personal data - just indices
    and per-dimension means - so it ships to every node in the clear."""
    import json
    from pathlib import Path as _Path

    p = _Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(selector.to_dict(), indent=2), encoding="utf-8")


def load_selector(path) -> TemplateSelector:
    import json
    from pathlib import Path as _Path

    d = json.loads(_Path(path).read_text(encoding="utf-8"))
    return TemplateSelector(
        dims=np.asarray(d["dims"], dtype=np.int32),
        thresholds=np.asarray(d["thresholds"], dtype=np.float32),
        n_bits=int(d["n_bits"]),
        criterion=str(d["criterion"]),
    )


def template_report(selector: TemplateSelector) -> Dict[str, Any]:
    return {
        "n_bits": selector.n_bits,
        "bytes_on_wire": selector.n_bits // 8,
        "selection_criterion": selector.criterion,
        "source_dims": 512,
        "compression_ratio": (512 * 32) / selector.n_bits,
    }


if __name__ == "__main__":  # smoke check
    from ..contracts import l2_normalise

    rng = np.random.default_rng(0)
    n_ids, per_id, dim, k = 40, 8, 512, 96

    labels = np.repeat(np.arange(n_ids), per_id)
    cents = l2_normalise(rng.normal(size=(n_ids, dim)))
    emb = l2_normalise(cents[labels] + 0.25 * rng.normal(size=(labels.size, dim)))

    sel = fit_selector(emb, k, labels)
    print(f"selector: {sel.n_bits} bits by {sel.criterion}, "
          f"{template_report(sel)['bytes_on_wire']} bytes on the wire")
    assert sel.dims.size == k and np.unique(sel.dims).size == k

    bits = binarise(emb, sel)
    assert bits.shape == (labels.size, k) and bits.dtype == bool

    # Identity structure must survive binarisation.
    sim = similarity_matrix(bits, bits)
    same = labels[:, None] == labels[None, :]
    eye = np.eye(labels.size, dtype=bool)
    mean_same = float(sim[same & ~eye].mean())
    mean_diff = float(sim[~same].mean())
    print(f"template similarity: same-id {mean_same:+.3f}  "
          f"diff-id {mean_diff:+.3f}")
    assert mean_same > mean_diff + 0.05, \
        "binarisation destroyed the identity signal"

    # The key must cancel inside the distance.
    key = derive_key("master", "case-0001", node_id=2, n_bits=k)
    pa, pb = protect(bits, key), protect(bits, key)
    assert np.array_equal(hamming_matrix(bits, bits), hamming_matrix(pa, pb)), \
        "keying changed the Hamming distance; matching would break"
    print("keyed matching is distance-preserving OK")

    # XOR is self-inverse.
    assert np.array_equal(protect(protect(bits, key), key), bits)

    # Unlinkability: the same person at two nodes must look unrelated.
    k1 = derive_key("master", "case-0001", node_id=1, n_bits=k)
    k2 = derive_key("master", "case-0001", node_id=2, n_bits=k)
    t1, t2 = protect(bits[:1], k1), protect(bits[:1], k2)
    cross = float(similarity_matrix(t1, t2)[0, 0])
    print(f"same person, two nodes: cross-node similarity {cross:+.3f} "
          f"(should sit near 0)")
    assert abs(cross) < 0.35, "templates are linkable across nodes"

    # Revocation: a new key must not match the old template.
    k_new = derive_key("master", "case-0001-reissued", node_id=1, n_bits=k)
    stale = float(similarity_matrix(protect(bits[:1], k1),
                                    protect(bits[:1], k_new))[0, 0])
    print(f"after reissue, old template scores {stale:+.3f}")
    assert abs(stale) < 0.35, "revocation did not invalidate the old template"

    # Wire format round-trip.
    packed = pack(protect(bits, key))
    assert packed.shape == (labels.size, k // 8) and packed.dtype == np.uint8
    assert np.array_equal(unpack(packed, k), protect(bits, key))

    one = make_template(emb[0], sel, key)
    assert one.shape == (k // 8,) and one.dtype == np.uint8
    print(f"single template: {one.tolist()}")

    # More bits should retain more signal.
    print()
    print("  bits   same-id   diff-id   separation")
    prev_sep = -1.0
    for kb in (32, 64, 96, 128, 256):
        s = fit_selector(emb, kb, labels)
        b = binarise(emb, s)
        m = similarity_matrix(b, b)
        sep = float(m[same & ~eye].mean() - m[~same].mean())
        print(f"  {kb:<6} {float(m[same & ~eye].mean()):+.3f}    "
              f"{float(m[~same].mean()):+.3f}    {sep:+.3f}")
        prev_sep = sep

    print("template OK")
