"""Frozen interface contracts between the four build phases.

Because the build is written in four passes and nothing is executed until the
last one, the boundaries between phases are written contracts rather than
passing tests. This module is that contract, made executable.

Every phase asserts on the way in and on the way out. A shape mistake then
surfaces at the boundary that broke it, instead of as a confusing traceback
three modules downstream.

Nothing here changes after Phase 1 without going back to the spec first.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------
# The nine artefacts
# --------------------------------------------------------------------------

EMBED_DIM = 512            # width of every embedding in the project
TEMPLATE_BITS = 96         # cancelable watchlist template
TEMPLATE_BYTES = TEMPLATE_BITS // 8   # 12
CHANNELS = 3
CHANNEL_ORDER = "RGB"      # canonical; backbones convert internally

EMB_DTYPE = np.float32
LABEL_DTYPE = np.int32
IMG_DTYPE = np.uint8

# Filenames, relative to cfg.paths.artifacts
F_IMAGES = "images.npy"
F_EMB_CLEAN = "embeddings_clean.npy"
F_EMB_DEG = "embeddings_deg.npy"
F_LABELS = "labels.npy"
F_IDENTITY_NAMES = "identity_names.json"
F_PARTITION = "partition.json"
F_META = "meta.json"
# A handful of clean/degraded pairs kept for report figures. The full image
# stacks are not persisted: Phases 2 and 3 work on embeddings alone, and
# Phase 4 gets its crops from video.
F_SAMPLES = "samples.npz"
# Extra non-mate probes drawn from an external face set (CelebA). Identity
# labels are not needed - a non-mate only has to be someone the gallery does
# not contain - so this works with distributions that ship without them.
F_EMB_NONMATE = "embeddings_nonmate.npy"

# Filenames elsewhere
F_ADAPTER = "adapter_final.pt"          # in cfg.paths.checkpoints
F_ALERTS_DB = "fedmps.sqlite"           # in cfg.paths.artifacts


class ContractError(AssertionError):
    """Raised when a cross-phase artefact does not match its contract."""


def _fail(msg: str) -> None:
    raise ContractError(msg)


# --------------------------------------------------------------------------
# Assertions
# --------------------------------------------------------------------------

def assert_images(arr: np.ndarray, image_size: int, name: str = "images") -> np.ndarray:
    """[N, S, S, 3] uint8, RGB."""
    if arr.ndim != 4:
        _fail(f"{name}: expected 4 dims [N,S,S,3], got shape {arr.shape}")
    if arr.shape[1] != image_size or arr.shape[2] != image_size:
        _fail(f"{name}: expected {image_size}x{image_size}, got "
              f"{arr.shape[1]}x{arr.shape[2]}")
    if arr.shape[3] != CHANNELS:
        _fail(f"{name}: expected {CHANNELS} channels, got {arr.shape[3]}")
    if arr.dtype != IMG_DTYPE:
        _fail(f"{name}: expected dtype {IMG_DTYPE}, got {arr.dtype}")
    return arr


def assert_embeddings(arr: np.ndarray, name: str = "embeddings",
                      check_normed: bool = True) -> np.ndarray:
    """[N, 512] float32, L2-normalised rows."""
    if arr.ndim != 2:
        _fail(f"{name}: expected 2 dims [N,{EMBED_DIM}], got shape {arr.shape}")
    if arr.shape[1] != EMBED_DIM:
        _fail(f"{name}: expected width {EMBED_DIM}, got {arr.shape[1]}")
    if arr.dtype != EMB_DTYPE:
        _fail(f"{name}: expected dtype {EMB_DTYPE}, got {arr.dtype}")
    if not np.isfinite(arr).all():
        _fail(f"{name}: contains NaN or inf")
    if check_normed and arr.shape[0] > 0:
        norms = np.linalg.norm(arr, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            worst = float(np.abs(norms - 1.0).max())
            _fail(f"{name}: rows are not L2-normalised (max deviation {worst:.4f})")
    return arr


def assert_labels(arr: np.ndarray, n_expected: Optional[int] = None,
                  name: str = "labels") -> np.ndarray:
    """[N] int32, non-negative identity ids."""
    if arr.ndim != 1:
        _fail(f"{name}: expected 1 dim [N], got shape {arr.shape}")
    if arr.dtype != LABEL_DTYPE:
        _fail(f"{name}: expected dtype {LABEL_DTYPE}, got {arr.dtype}")
    if arr.size and arr.min() < 0:
        _fail(f"{name}: negative identity id present")
    if n_expected is not None and arr.shape[0] != n_expected:
        _fail(f"{name}: expected {n_expected} rows, got {arr.shape[0]}")
    return arr


def assert_aligned(*arrays: np.ndarray, names: Optional[List[str]] = None) -> None:
    """Every array must have the same leading dimension."""
    if not arrays:
        return
    n = arrays[0].shape[0]
    for i, a in enumerate(arrays[1:], start=1):
        if a.shape[0] != n:
            label = names[i] if names else f"array[{i}]"
            first = names[0] if names else "array[0]"
            _fail(f"{label} has {a.shape[0]} rows but {first} has {n}")


def assert_template(arr: np.ndarray, name: str = "template") -> np.ndarray:
    """[TEMPLATE_BYTES] uint8 - 96 bits packed. Produced in Phase 3."""
    if arr.dtype != np.uint8:
        _fail(f"{name}: expected uint8, got {arr.dtype}")
    if arr.shape != (TEMPLATE_BYTES,):
        _fail(f"{name}: expected shape ({TEMPLATE_BYTES},), got {arr.shape}")
    return arr


def assert_partition(part: Dict[str, Any], n_total: int,
                     name: str = "partition") -> Dict[str, Any]:
    """Structural check on partition.json.

    Guarantees Phase 2 can rely on:
      - every node index is present and contiguous from 0
      - no image index appears at two nodes, or in both a node and the test set
      - every index is in range
    """
    for key in ("n_nodes", "seed", "test_idx", "nodes"):
        if key not in part:
            _fail(f"{name}: missing top-level key '{key}'")

    n_nodes = part["n_nodes"]
    nodes = part["nodes"]
    if len(nodes) != n_nodes:
        _fail(f"{name}: n_nodes={n_nodes} but {len(nodes)} node entries present")

    seen: set = set()

    def _take(idx_list: List[int], where: str) -> None:
        for i in idx_list:
            if not (0 <= i < n_total):
                _fail(f"{name}: index {i} in {where} is out of range [0,{n_total})")
            if i in seen:
                _fail(f"{name}: index {i} appears more than once (at {where})")
            seen.add(i)

    _take(part["test_idx"], "test_idx")

    for k in range(n_nodes):
        nk = f"node_{k}"
        if nk not in nodes:
            _fail(f"{name}: missing '{nk}'")
        entry = nodes[nk]
        for key in ("identities", "train_idx", "val_idx", "degrade_params"):
            if key not in entry:
                _fail(f"{name}: {nk} missing key '{key}'")
        _take(entry["train_idx"], f"{nk}.train_idx")
        _take(entry["val_idx"], f"{nk}.val_idx")

    return part


# --------------------------------------------------------------------------
# Typed load / save helpers
# --------------------------------------------------------------------------

def save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def load_npy(path: Path, name: Optional[str] = None) -> np.ndarray:
    if not path.exists():
        _fail(f"{name or path.name}: not found at {path}. "
              f"Run scripts/01_prepare_data.py first.")
    return np.load(path)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_json(path: Path, name: Optional[str] = None) -> Any:
    if not path.exists():
        _fail(f"{name or path.name}: not found at {path}. "
              f"Run scripts/01_prepare_data.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def l2_normalise(arr: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Row-wise L2 normalisation. Every embedding in the project passes here."""
    arr = np.asarray(arr, dtype=EMB_DTYPE)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return (arr / np.maximum(norms, eps)).astype(EMB_DTYPE)


# --------------------------------------------------------------------------
# Convenience bundle used by Phases 2 and 3
# --------------------------------------------------------------------------

class Artefacts:
    """Everything Phase 1 produced, loaded and validated in one call.

    Phases 2 and 3 start with `Artefacts.load(cfg)` and never touch an image
    file or the disk layout directly.
    """

    def __init__(self, emb_clean: np.ndarray, emb_deg: np.ndarray,
                 labels: np.ndarray, partition: Dict[str, Any],
                 meta: Dict[str, Any],
                 identity_names: Optional[List[str]] = None,
                 emb_nonmate: Optional[np.ndarray] = None) -> None:
        self.emb_clean = emb_clean
        self.emb_deg = emb_deg
        self.labels = labels
        self.partition = partition
        self.meta = meta
        self.identity_names = identity_names or []
        # Optional extra crowd; None when only the held-out identities are used.
        self.emb_nonmate = emb_nonmate

    @property
    def n(self) -> int:
        return int(self.labels.shape[0])

    @property
    def n_nodes(self) -> int:
        return int(self.partition["n_nodes"])

    def node(self, k: int) -> Dict[str, Any]:
        return self.partition["nodes"][f"node_{k}"]

    @classmethod
    def load(cls, cfg) -> "Artefacts":
        a = cfg.paths.artifacts
        emb_clean = assert_embeddings(load_npy(a / F_EMB_CLEAN, "embeddings_clean"),
                                      "embeddings_clean")
        emb_deg = assert_embeddings(load_npy(a / F_EMB_DEG, "embeddings_deg"),
                                    "embeddings_deg")
        labels = assert_labels(load_npy(a / F_LABELS, "labels"), name="labels")
        assert_aligned(emb_clean, emb_deg, labels,
                       names=["embeddings_clean", "embeddings_deg", "labels"])
        meta = load_json(a / F_META, "meta")
        partition = assert_partition(load_json(a / F_PARTITION, "partition"),
                                     n_total=labels.shape[0])
        names_path = a / F_IDENTITY_NAMES
        names = load_json(names_path) if names_path.exists() else []

        nonmate = None
        nm_path = a / F_EMB_NONMATE
        if nm_path.exists():
            nonmate = assert_embeddings(load_npy(nm_path, "embeddings_nonmate"),
                                        "embeddings_nonmate")
        return cls(emb_clean, emb_deg, labels, partition, meta, names, nonmate)

    def describe(self) -> str:
        lines = [
            f"images        : {self.n}",
            f"identities    : {len(set(self.labels.tolist()))}",
            f"backbone      : {self.meta.get('backbone', '?')}",
            f"nodes         : {self.n_nodes}",
            f"test images   : {len(self.partition['test_idx'])}",
            f"extra non-mates: {0 if self.emb_nonmate is None else len(self.emb_nonmate)}",
        ]
        for k in range(self.n_nodes):
            e = self.node(k)
            lines.append(
                f"  node_{k}: {len(e['identities']):>4} ids  "
                f"{len(e['train_idx']):>5} train  {len(e['val_idx']):>4} val"
            )
        return "\n".join(lines)


if __name__ == "__main__":  # smoke check
    rng = np.random.default_rng(0)
    emb = l2_normalise(rng.normal(size=(10, EMBED_DIM)))
    assert_embeddings(emb)
    lab = np.arange(10, dtype=LABEL_DTYPE)
    assert_labels(lab, n_expected=10)
    assert_aligned(emb, lab)

    part = {
        "n_nodes": 2, "seed": 0, "test_idx": [8, 9],
        "nodes": {
            "node_0": {"identities": [0, 1], "train_idx": [0, 1, 2],
                       "val_idx": [3], "degrade_params": {}},
            "node_1": {"identities": [2, 3], "train_idx": [4, 5, 6],
                       "val_idx": [7], "degrade_params": {}},
        },
    }
    assert_partition(part, n_total=10)

    tpl = np.zeros(TEMPLATE_BYTES, dtype=np.uint8)
    assert_template(tpl)

    # negative case: an unnormalised embedding must be rejected
    try:
        assert_embeddings((emb * 3.0).astype(EMB_DTYPE))
    except ContractError:
        pass
    else:
        raise SystemExit("contract check failed to catch unnormalised input")

    print("contracts OK")
