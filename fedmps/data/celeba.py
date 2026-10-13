"""CelebA as the non-mate population - the crowd walking past the camera.

Open-set identification needs two kinds of probe. Mates are the people on the
watchlist. Non-mates are everyone else, and they are what FPIR is measured on:
of the people in no watchlist at all, how many still triggered an alert?

The held-out LFW identities give about 500 non-mates, which resolves an FPIR
no finer than 1/500. Every metric below that reads NaN, including the 1e-4 and
1e-5 operating points that matter for missing-person search - one camera
against a 50-person watchlist makes 100,000 comparisons a day, so a rate of
1e-3 means a hundred false alarms a day and a rate of 1e-5 means one.

CelebA supplies 202,599 faces. Crucially, **non-mates need no identity
labels** - they only have to be people who are not in the gallery. So the
Kaggle distribution of CelebA works here even though it ships without
identity_CelebA.txt and is therefore useless for training.

Two things have to be right or the resulting FPIR flatters the system:

  Same degradation. A clean CelebA face compared against a degraded gallery
  would be trivially separable, and the false-alarm rate would come out far
  too low. Non-mates go through the identical degradation chain, with camera
  profiles drawn from the same nodes.

  Same framing. Our LFW crops keep a square, face-centred region. CelebA's
  aligned images are framed differently, so a naive crop puts the face at a
  different scale and the embeddings separate on framing rather than on
  identity. `validate_against` below checks for exactly that.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from ..contracts import IMG_DTYPE

# CelebA aligned images are 218x178 with the face roughly centred. This window
# is chosen so the face fills about the same fraction of the frame as our LFW
# crop does, which is what keeps the two embedding populations comparable.
CELEBA_CROP = (slice(38, 196), slice(10, 168))      # 158 x 158

IMAGE_DIR_HINTS = ("img_align_celeba", "img_celeba")


def _log(msg: str) -> None:
    print(f"[celeba] {msg}", flush=True)


def list_images(zip_path: Path) -> List[str]:
    """Every JPEG inside the archive, sorted for reproducibility."""
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist()
                 if n.lower().endswith(".jpg")
                 and any(h in n for h in IMAGE_DIR_HINTS)]
    if not names:
        raise RuntimeError(
            f"no CelebA images found in {zip_path}. Expected paths containing "
            f"one of {IMAGE_DIR_HINTS}.")
    return sorted(names)


def has_identity_labels(zip_path: Path) -> bool:
    """True if this archive can also be used for training.

    The Kaggle distribution ships attributes, bounding boxes, landmarks and an
    image-level partition, but not identity_CelebA.txt. Without identities
    CelebA cannot form same-person pairs, so it can serve as a crowd and
    nothing more.
    """
    with zipfile.ZipFile(zip_path) as z:
        return any("identity" in n.lower() for n in z.namelist())


def iter_crops(zip_path: Path, names: List[str], image_size: int,
               chunk: int = 2000) -> Iterator[np.ndarray]:
    """Yield [chunk, S, S, 3] uint8 RGB batches.

    Chunked because the full set at working resolution would be well over a
    gigabyte; only the embeddings are kept.
    """
    with zipfile.ZipFile(zip_path) as z:
        buf: List[np.ndarray] = []
        for name in names:
            try:
                raw = np.frombuffer(z.read(name), dtype=np.uint8)
            except KeyError:
                continue
            bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            if bgr.shape[0] >= 196 and bgr.shape[1] >= 168:
                bgr = bgr[CELEBA_CROP[0], CELEBA_CROP[1]]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            buf.append(cv2.resize(rgb, (image_size, image_size),
                                  interpolation=cv2.INTER_AREA))
            if len(buf) >= chunk:
                yield np.stack(buf).astype(IMG_DTYPE)
                buf = []
        if buf:
            yield np.stack(buf).astype(IMG_DTYPE)


def sample_names(zip_path: Path, n: int, seed: int) -> List[str]:
    """Pick n image names at random, without replacement."""
    names = list_images(zip_path)
    rng = np.random.default_rng(seed)
    if n >= len(names):
        _log(f"requested {n} but the archive holds {len(names)}; using all")
        return names
    idx = rng.choice(len(names), size=n, replace=False)
    return [names[i] for i in np.sort(idx)]


# --------------------------------------------------------------------------
# Sanity check
# --------------------------------------------------------------------------

def validate_against(celeba_emb: np.ndarray, lfw_nonmate_emb: np.ndarray,
                     gallery_emb: np.ndarray) -> Dict[str, Any]:
    """Do CelebA non-mates behave like the LFW non-mates they are joining?

    Both are strangers to the gallery, so both should score about the same
    against it. If CelebA scores markedly lower, the two populations differ
    for some reason other than identity - framing, alignment or degradation -
    and every FPIR computed on them would be optimistic. That is a quiet,
    plausible-looking failure, so it is checked rather than assumed.
    """
    if celeba_emb.shape[0] == 0 or lfw_nonmate_emb.shape[0] == 0:
        return {"ok": False, "reason": "one population is empty"}

    c = (celeba_emb @ gallery_emb.T).max(axis=1)
    l = (lfw_nonmate_emb @ gallery_emb.T).max(axis=1)

    gap = float(l.mean() - c.mean())
    pooled_sd = float(np.sqrt(0.5 * (c.var() + l.var())) + 1e-9)
    effect = gap / pooled_sd

    out = {
        "celeba_mean_top_score": float(c.mean()),
        "lfw_mean_top_score": float(l.mean()),
        "gap": gap,
        "standardised_gap": effect,
        "celeba_p99": float(np.quantile(c, 0.99)),
        "lfw_p99": float(np.quantile(l, 0.99)),
        "n_celeba": int(c.size), "n_lfw": int(l.size),
    }
    # Half a standard deviation is a generous bar; beyond it the populations
    # are clearly not interchangeable.
    out["ok"] = abs(effect) < 0.5
    out["reason"] = ("populations are comparable" if out["ok"] else
                     "CelebA non-mates score too differently from LFW "
                     "non-mates; check CELEBA_CROP framing and that the same "
                     "degradation was applied. FPIR would be misleading.")
    return out


def resolvable_fpir(n_nonmates: int) -> float:
    """Finest false-positive rate this many non-mates can measure."""
    return 1.0 / max(n_nonmates, 1)


if __name__ == "__main__":  # smoke check
    import sys

    from ..config import default_config

    cfg = default_config()
    zip_path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path.home() / "Downloads" / "archive.zip"

    if not zip_path.exists():
        print(f"CelebA archive not found at {zip_path}; skipping live check")
        # Still exercise the pure functions.
        assert abs(resolvable_fpir(500) - 0.002) < 1e-9
        assert abs(resolvable_fpir(20000) - 5e-5) < 1e-12
        rng = np.random.default_rng(0)
        from ..contracts import l2_normalise
        g = l2_normalise(rng.normal(size=(50, 512)))
        a = l2_normalise(rng.normal(size=(200, 512)))
        b = l2_normalise(rng.normal(size=(200, 512)))
        v = validate_against(a, b, g)
        assert v["ok"], v
        print("pure functions OK")
        raise SystemExit(0)

    print(f"archive: {zip_path}")
    names = list_images(zip_path)
    print(f"images in archive        : {len(names):,}")
    print(f"has identity labels      : {has_identity_labels(zip_path)}")
    print(f"  -> usable for training : {has_identity_labels(zip_path)}")
    print(f"  -> usable as non-mates : True (no labels needed)")

    picked = sample_names(zip_path, 8, cfg.seed)
    batch = next(iter_crops(zip_path, picked, cfg.data.image_size, chunk=8))
    print(f"\nsampled batch shape      : {batch.shape} {batch.dtype}")
    assert batch.shape == (8, cfg.data.image_size, cfg.data.image_size, 3)
    assert batch.dtype == IMG_DTYPE

    print(f"\nFPIR resolvable with 500 non-mates   : {resolvable_fpir(500):.1e}")
    print(f"FPIR resolvable with 20,000 non-mates: {resolvable_fpir(20000):.1e}")
    print("celeba OK")


# --------------------------------------------------------------------------
# CelebA as a training source
#
# With identity_CelebA.txt present the archive becomes a far better training
# set than LFW: 10,177 identities at a median of 21 images each, against LFW's
# 400 at a median of 8. That matters directly for the 70/20/10 split, where
# the nodes' 30% was leaving roughly 200 training images per node - too little
# for the adapter to learn a camera correction that generalises.
#
# This path streams. Twenty thousand images at working resolution is about
# 1.5 GB, and the batch flow holds clean and degraded copies at once, so
# decoding everything up front would need three. Instead each chunk is
# decoded, degraded, embedded and thrown away, and only the embeddings are
# kept - 512 floats per image, a few tens of megabytes.
# --------------------------------------------------------------------------

def load_identity_map(txt_path: Path) -> Dict[str, int]:
    """Parse identity_CelebA.txt: `000001.jpg 2880` per line."""
    mapping: Dict[str, int] = {}
    with open(txt_path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 2:
                mapping[parts[0]] = int(parts[1])
    if not mapping:
        raise RuntimeError(f"no identities parsed from {txt_path}")
    return mapping


@dataclass
class CelebAIndex:
    """Which files to use, resolved before a single pixel is decoded."""

    entries: List[Tuple[int, str]]      # (identity index, zip entry path)
    names: List[str]                    # identity index -> display name
    pids: List[int]                     # identity index -> CelebA person id

    @property
    def n(self) -> int:
        return len(self.entries)

    @property
    def labels(self) -> np.ndarray:
        from ..contracts import LABEL_DTYPE
        return np.asarray([e[0] for e in self.entries], dtype=LABEL_DTYPE)


def build_index(zip_path: Path, identity_txt: Path,
                min_images: int = 5,
                max_identities: Optional[int] = None,
                max_images_per_identity: Optional[int] = None,
                seed: int = 1337) -> CelebAIndex:
    """Choose identities and files up front, on metadata alone.

    Every filter runs on the identity table, so the decode cost is paid only
    for images that survive - the same lesson the LFW loader learned.
    """
    ident = load_identity_map(identity_txt)
    by_name = {Path(n).name: n for n in list_images(zip_path)}

    groups: Dict[int, List[str]] = {}
    for fname, pid in ident.items():
        entry = by_name.get(fname)
        if entry is not None:
            groups.setdefault(pid, []).append(entry)

    eligible = {p: sorted(f) for p, f in groups.items() if len(f) >= min_images}
    if not eligible:
        raise RuntimeError(
            f"no CelebA identity has at least {min_images} images present")

    rng = np.random.default_rng(seed)
    chosen = sorted(eligible, key=lambda p: (-len(eligible[p]), p))
    if max_identities is not None:
        chosen = chosen[:max_identities]
    chosen = sorted(chosen)

    entries: List[Tuple[int, str]] = []
    names: List[str] = []
    for idx, pid in enumerate(chosen):
        files = eligible[pid]
        if max_images_per_identity is not None and len(files) > max_images_per_identity:
            keep = rng.choice(len(files), size=max_images_per_identity,
                              replace=False)
            files = [files[i] for i in sorted(keep)]
        names.append(f"celeb_{pid:05d}")
        entries.extend((idx, f) for f in files)

    _log(f"selected {len(entries):,} images from {len(chosen):,} identities "
         f"(of {len(groups):,} in the archive, {len(eligible):,} eligible)")
    return CelebAIndex(entries, names, list(chosen))


def sample_nonmate_names(zip_path: Path, identity_txt: Optional[Path],
                         n: int, seed: int,
                         exclude_pids: Optional[List[int]] = None
                         ) -> List[str]:
    """Sample non-mate images, excluding people used for training or gallery.

    This exclusion is not optional once CelebA supplies both sides. Drawing
    non-mates at random from the same archive would quietly include photographs
    of enrolled people, and a "non-mate" who is in fact in the gallery *should*
    match. Those become counted false alarms, the measured FPIR comes out too
    high, and the system looks worse than it is - or, if the threshold is then
    tuned against that inflated rate, genuinely worse.
    """
    if not exclude_pids or identity_txt is None:
        return sample_names(zip_path, n, seed)

    ident = load_identity_map(identity_txt)
    blocked = set(exclude_pids)
    by_name = {Path(x).name: x for x in list_images(zip_path)}

    pool = [by_name[f] for f, pid in ident.items()
            if pid not in blocked and f in by_name]
    pool.sort()
    _log(f"non-mate pool: {len(pool):,} images from identities outside the "
         f"{len(blocked):,} used for training and gallery")

    rng = np.random.default_rng(seed + 1)
    if n >= len(pool):
        return pool
    idx = rng.choice(len(pool), size=n, replace=False)
    return [pool[i] for i in np.sort(idx)]


def stream_embed(zip_path: Path, index: CelebAIndex, profile_ids: np.ndarray,
                 partition: Dict[str, Any], cfg, backbone,
                 chunk: int = 1000,
                 checkpoint_dir: Optional[Path] = None
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Decode, degrade, embed and discard, chunk by chunk.

    Returns (clean embeddings, degraded embeddings). Peak memory is one chunk
    of images rather than the whole set.

    Every chunk is also written to `checkpoint_dir` as it completes, and an
    interrupted run picks up from the last completed chunk. A CelebA pass at
    useful scale takes hours on a laptop CPU, and losing all of it to a closed
    lid, a sleeping machine or a killed shell is not an acceptable failure
    mode. Delete the directory to force a clean re-run.
    """
    from .degrade import degrade_image, resolve_params

    ranges = cfg.degrade.ranges
    profiles = {k: partition["nodes"][f"node_{k}"]["degrade_params"]
                for k in range(partition["n_nodes"])}

    clean_parts: List[np.ndarray] = []
    deg_parts: List[np.ndarray] = []
    size = cfg.data.image_size
    done = 0

    # Resume from whatever previous chunks completed.
    ckpt = Path(checkpoint_dir) if checkpoint_dir else None
    skip_until = 0
    if ckpt is not None:
        ckpt.mkdir(parents=True, exist_ok=True)
        idx = 0
        while (ckpt / f"clean_{idx:05d}.npy").exists() and               (ckpt / f"deg_{idx:05d}.npy").exists():
            c = np.load(ckpt / f"clean_{idx:05d}.npy")
            d = np.load(ckpt / f"deg_{idx:05d}.npy")
            clean_parts.append(c)
            deg_parts.append(d)
            skip_until += c.shape[0]
            idx += 1
        if skip_until:
            _log(f"resuming: {skip_until:,} images already embedded in "
                 f"{idx} checkpoint(s)")
        done = skip_until

    with zipfile.ZipFile(zip_path) as z:
        buf: List[np.ndarray] = []
        rows: List[int] = []

        def flush() -> None:
            nonlocal buf, rows, done
            if not buf:
                return
            images = np.stack(buf).astype(IMG_DTYPE)
            labels = index.labels[np.asarray(rows)]

            degraded = np.empty_like(images)
            for j, row in enumerate(rows):
                rng_j = np.random.default_rng((cfg.seed * 7919 + row) % (2 ** 63))
                params = resolve_params(profiles[int(profile_ids[row])], ranges,
                                        rng_j, cfg.degrade.per_image_jitter)
                degraded[j] = degrade_image(images[j], params, cfg.degrade, rng_j)

            ce = backbone.embed(images, labels,
                                batch_size=cfg.backbone.batch_size,
                                progress=False)
            de = backbone.embed(degraded, labels,
                                batch_size=cfg.backbone.batch_size,
                                progress=False)
            clean_parts.append(ce)
            deg_parts.append(de)
            if ckpt is not None:
                n = len(clean_parts) - 1
                np.save(ckpt / f"clean_{n:05d}.npy", ce)
                np.save(ckpt / f"deg_{n:05d}.npy", de)
            done += len(buf)
            pct = 100.0 * done / max(index.n, 1)
            _log(f"  embedded {done:,}/{index.n:,}  ({pct:.0f}%)")
            buf, rows = [], []

        for row, (_, entry) in enumerate(index.entries):
            if row < skip_until:
                continue
            bgr = cv2.imdecode(np.frombuffer(z.read(entry), dtype=np.uint8),
                               cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            if bgr.shape[0] >= 196 and bgr.shape[1] >= 168:
                bgr = bgr[CELEBA_CROP[0], CELEBA_CROP[1]]
            buf.append(cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                                  (size, size), interpolation=cv2.INTER_AREA))
            rows.append(row)
            if len(buf) >= chunk:
                flush()
        flush()

    return (np.concatenate(clean_parts, axis=0),
            np.concatenate(deg_parts, axis=0))


def estimate_runtime(n_images: int, images_per_sec: float = 7.2) -> str:
    """Rough wall clock, from the rate measured on this project's CPU runs."""
    minutes = (n_images * 2) / images_per_sec / 60.0
    return (f"{minutes:.0f} min" if minutes < 90
            else f"{minutes / 60:.1f} hours")
