"""Dataset loading.

Produces the one thing the rest of Phase 1 needs: a stack of uint8 RGB face
crops at the canonical working size, plus an identity label per image.

Two sources:

  "lfw"    - fetched automatically through scikit-learn. No account, no
             licence form, no manual download. This is the default and it is
             what the 12-week plan assumes.

  "folder" - any directory laid out as `identity_name/image.jpg`. Use this for
             CelebA once you have arranged it that way, or for frames pulled
             out of your own corridor footage.

CelebA has deliberately no automatic downloader here. Its usual route is a
Google Drive link that rate-limits and fails silently, which is not something
worth putting on a twelve-week critical path. Arrange it as folders and use
"folder" instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..config import Config, DataConfig
from ..contracts import IMG_DTYPE, LABEL_DTYPE, assert_images, assert_labels

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# LFW ships 13,233 images of 5,749 people. Used only to recognise an
# interrupted download, which is by far the most common way this step fails.
LFW_EXPECTED_IMAGES = 13233


def _log(msg: str) -> None:
    print(f"[loader] {msg}", flush=True)


@dataclass
class Dataset:
    """A loaded, canonicalised image set."""

    images: np.ndarray        # [N, S, S, 3] uint8, RGB
    labels: np.ndarray        # [N] int32, identity id
    identity_names: List[str]  # index by identity id

    @property
    def n(self) -> int:
        return int(self.images.shape[0])

    @property
    def n_identities(self) -> int:
        return len(self.identity_names)

    def describe(self) -> str:
        counts = np.bincount(self.labels)
        return (f"{self.n} images, {self.n_identities} identities, "
                f"{counts.min()}-{counts.max()} images each "
                f"(median {int(np.median(counts))})")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """scikit-learn hands back float images; which scale depends on version."""
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if float(arr.max()) <= 1.5:      # [0, 1] scaled
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _resize_stack(images: np.ndarray, size: int) -> np.ndarray:
    """Resize [N, H, W, 3] to [N, size, size, 3], one image at a time."""
    if images.shape[1] == size and images.shape[2] == size:
        return np.ascontiguousarray(images, dtype=IMG_DTYPE)
    out = np.empty((images.shape[0], size, size, 3), dtype=IMG_DTYPE)
    for i in range(images.shape[0]):
        interp = cv2.INTER_AREA if images.shape[1] > size else cv2.INTER_LINEAR
        out[i] = cv2.resize(images[i], (size, size), interpolation=interp)
    return out


def _cap_identities(images: np.ndarray, labels: np.ndarray,
                    names: List[str], max_identities: Optional[int],
                    seed: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Keep at most `max_identities`, chosen by image count then shuffled.

    Preferring well-populated identities matters: an identity with two images
    cannot form a useful (degraded, clean) training pair.
    """
    n_ids = len(names)
    if max_identities is None or n_ids <= max_identities:
        return images, labels, names

    counts = np.bincount(labels, minlength=n_ids)
    keep = np.argsort(-counts)[:max_identities]
    keep = np.sort(keep)

    remap = -np.ones(n_ids, dtype=np.int64)
    remap[keep] = np.arange(len(keep))

    mask = remap[labels] >= 0
    new_labels = remap[labels[mask]].astype(LABEL_DTYPE)
    new_names = [names[i] for i in keep]
    return images[mask], new_labels, new_names


def _drop_sparse_identities(images: np.ndarray, labels: np.ndarray,
                            names: List[str],
                            min_images: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Remove identities with too few images to form training pairs."""
    counts = np.bincount(labels, minlength=len(names))
    keep = np.where(counts >= min_images)[0]
    if len(keep) == len(names):
        return images, labels, names

    remap = -np.ones(len(names), dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    mask = remap[labels] >= 0
    return (images[mask],
            remap[labels[mask]].astype(LABEL_DTYPE),
            [names[i] for i in keep])


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------

def _jpeg_is_intact(path: Path) -> bool:
    """Cheap truncation check: JPEG magic at the head, EOI marker at the tail.

    Reading four bytes per file is fast enough to run on every load, and it
    catches the failure mode that actually happens - a download cut off
    mid-file - without decoding 13,000 images.
    """
    try:
        size = path.stat().st_size
        if size < 4:
            return False
        with path.open("rb") as fh:
            if fh.read(2) != b"\xff\xd8":        # SOI
                return False
            fh.seek(-2, 2)
            return fh.read(2) == b"\xff\xd9"     # EOI
    except OSError:
        return False


def validate_lfw_cache(cache_dir: Path, remove_corrupt: bool = True
                       ) -> Dict[str, Any]:
    """Check the extracted LFW tree before handing it to scikit-learn.

    scikit-learn's loader opens every file and dies on the first bad one, from
    deep inside a joblib-cached call, with a traceback that points at PIL
    rather than at the real problem. Catching it here turns an interrupted
    download into a sentence someone can act on.
    """
    home = Path(cache_dir) / "lfw_home"
    folder = home / "lfw_funneled"
    if not folder.is_dir():
        return {"present": False, "total": 0, "corrupt": 0}

    corrupt: List[Path] = []
    total = 0
    for p in folder.rglob("*.jpg"):
        total += 1
        if not _jpeg_is_intact(p):
            corrupt.append(p)

    if corrupt and remove_corrupt:
        for p in corrupt:
            try:
                p.unlink()
            except OSError:
                pass
        # joblib caches the result of the previous load, so it has to go too
        # or the repaired tree is never actually re-read.
        joblib_dir = home / "joblib"
        if joblib_dir.is_dir():
            import shutil
            shutil.rmtree(joblib_dir, ignore_errors=True)
        _log(f"removed {len(corrupt)} corrupt image(s) and cleared the "
             f"joblib cache")

    return {"present": True, "total": total, "corrupt": len(corrupt),
            "folder": folder, "home": home}


def ensure_lfw_downloaded(cache_dir: Path) -> Path:
    """Make sure the funneled LFW tree exists on disk, downloading if not.

    scikit-learn owns the download and extraction, which is worth keeping -
    the archive URL moves occasionally and they track it. What we do not use
    is their *loader*: see `load_lfw` for why.
    """
    folder = Path(cache_dir) / "lfw_home" / "lfw_funneled"
    if folder.is_dir() and any(folder.iterdir()):
        return folder

    _log("LFW not found locally; downloading (~200 MB, do not interrupt)")
    try:
        from sklearn.datasets._lfw import _check_fetch_lfw

        _check_fetch_lfw(data_home=str(cache_dir), funneled=True,
                         download_if_missing=True)
    except Exception:                                  # noqa: BLE001
        # Private API moved. Fall back to the public one, asking for a tiny
        # subset so it downloads and extracts without decoding the world.
        from sklearn.datasets import fetch_lfw_people

        fetch_lfw_people(data_home=str(cache_dir), min_faces_per_person=100,
                         download_if_missing=True)

    if not folder.is_dir():
        raise RuntimeError(f"LFW download finished but {folder} is missing")
    return folder


def _index_lfw(folder: Path) -> Dict[str, List[Path]]:
    """identity name -> its image files. Reads directory entries, not pixels."""
    index: Dict[str, List[Path]] = {}
    for person in sorted(p for p in folder.iterdir() if p.is_dir()):
        files = sorted(f for f in person.iterdir() if f.suffix.lower() == ".jpg")
        if files:
            index[person.name] = files
    return index


def load_lfw(cfg: DataConfig, cache_dir: Path, seed: int) -> Dataset:
    """Labelled Faces in the Wild, read directly from the extracted tree.

    scikit-learn's `fetch_lfw_people` is deliberately not used for loading.
    It decodes every eligible image into one float32 array before any of our
    filters apply - 5,985 images at the default settings, a 1.18 GB peak - and
    then joblib pickles all of it back to disk. With `max_identities` set we
    were discarding most of that immediately, paying about fourteen minutes
    and a gigabyte for it.

    Reading the tree ourselves lets every filter run on directory entries
    first, so pixels are decoded only for images that survive. The peak
    becomes the size of what we actually keep, in uint8, and there is no
    joblib layer at all.

    scikit-learn still owns the download - see `ensure_lfw_downloaded`.
    """
    folder = ensure_lfw_downloaded(cache_dir)

    state = validate_lfw_cache(cache_dir)
    if state["present"]:
        remaining = state["total"] - state["corrupt"]
        if remaining < LFW_EXPECTED_IMAGES * 0.9:
            raise RuntimeError(
                f"The LFW cache at {state['home']} is incomplete: {remaining} "
                f"images present, about {LFW_EXPECTED_IMAGES} expected. A "
                f"previous download was interrupted.\n\n"
                f"Delete the cache and let it download again:\n"
                f"    python fedmps/scripts/01_prepare_data.py --repair-cache\n\n"
                f"It fetches roughly 200 MB, so give it a few minutes on a "
                f"slow connection and do not stop it part way."
            )
        _log(f"LFW cache looks complete: {remaining} images")

    rng = np.random.default_rng(seed)
    index = _index_lfw(folder)

    # -- filter on counts, before decoding anything ------------------------
    eligible = {name: files for name, files in index.items()
                if len(files) >= cfg.lfw_min_faces_per_person}
    if not eligible:
        raise RuntimeError(
            f"no identity in {folder} has at least "
            f"{cfg.lfw_min_faces_per_person} images")

    chosen = sorted(eligible, key=lambda n: (-len(eligible[n]), n))
    if cfg.max_identities is not None:
        chosen = chosen[:cfg.max_identities]
    chosen = sorted(chosen)

    # LFW's long tail is extreme: one identity has 530 images while the median
    # is 31. Left uncapped, a few people dominate every batch, every node and
    # every gallery, and the federated split stops meaning much.
    selected: List[Tuple[str, Path]] = []
    cap = cfg.max_images_per_identity
    for name in chosen:
        files = eligible[name]
        if cap is not None and len(files) > cap:
            keep = rng.choice(len(files), size=cap, replace=False)
            files = [files[i] for i in sorted(keep)]
        selected.extend((name, f) for f in files)

    _log(f"selected {len(selected)} images from {len(chosen)} identities "
         f"(of {len(index)} in the archive, {len(eligible)} eligible)")

    # -- decode only what survived ----------------------------------------
    size = cfg.image_size
    images = np.empty((len(selected), size, size, 3), dtype=IMG_DTYPE)
    labels = np.empty(len(selected), dtype=LABEL_DTYPE)
    names = chosen
    name_to_id = {n: i for i, n in enumerate(names)}

    kept = 0
    for name, path in selected:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        # The same square, face-centred crop out of the 250x250 funneled
        # originals that the previous implementation asked sklearn for.
        if bgr.shape[0] >= 189 and bgr.shape[1] >= 189:
            bgr = bgr[61:189, 61:189]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        images[kept] = cv2.resize(rgb, (size, size),
                                  interpolation=cv2.INTER_AREA)
        labels[kept] = name_to_id[name]
        kept += 1

    images = images[:kept]
    labels = labels[:kept]

    # A per-identity cap can drop someone below the pair threshold.
    images, labels, names = _drop_sparse_identities(
        images, labels, names, min_images=2)

    return Dataset(images, labels, names)


def load_folder(cfg: DataConfig, seed: int) -> Dataset:
    """A directory of `identity_name/image.jpg`.

    Sub-directories are identities. Files directly in the root are ignored.
    """
    if cfg.folder_path is None:
        raise ValueError("data.folder_path must be set when data.source == 'folder'")
    root = Path(cfg.folder_path)
    if not root.is_dir():
        raise FileNotFoundError(f"data.folder_path does not exist: {root}")

    people = sorted(p for p in root.iterdir() if p.is_dir())
    if not people:
        raise FileNotFoundError(f"no identity sub-directories under {root}")

    # Decide what to read BEFORE reading it. The earlier version loaded every
    # image in the tree, stacked one array out of all of them, and only then
    # applied max_identities - so pointing it at a corpus larger than the cap
    # allocated the whole corpus first. On VGGFace2 that is 172,620 images at
    # 160x160x3 = 13.3 GB, and the process was killed by the OOM reaper before
    # it reached the line that would have discarded most of it. It also never
    # applied max_images_per_identity at all, which the LFW and CelebA paths
    # both honour.
    #
    # Directory listings are cheap; pixels are not. So do the selection on the
    # listing and read only the files that survive it.
    eligible: List[Tuple[Path, List[Path]]] = []
    for person in people:
        files = sorted(f for f in person.iterdir()
                       if f.suffix.lower() in IMAGE_SUFFIXES)
        if len(files) < cfg.folder_min_images_per_person:
            continue
        eligible.append((person, files))

    if not eligible:
        raise RuntimeError(
            f"no identity under {root} has the required "
            f"{cfg.folder_min_images_per_person} images"
        )

    rng = np.random.default_rng(seed)
    if cfg.max_identities is not None and len(eligible) > cfg.max_identities:
        keep = rng.choice(len(eligible), size=cfg.max_identities, replace=False)
        eligible = [eligible[i] for i in sorted(keep)]

    per_id = cfg.max_images_per_identity
    planned = sum(min(len(f), per_id) if per_id else len(f)
                  for _, f in eligible)
    _log(f"folder: {len(eligible)} identities, reading {planned} images "
         f"at {cfg.image_size}px "
         f"(~{planned * cfg.image_size ** 2 * 3 / 2 ** 30:.1f} GB)")

    # Allocate once, up front, at the size the plan says. Appending to a list
    # and stacking at the end briefly holds two copies of everything.
    images = np.empty((planned, cfg.image_size, cfg.image_size, 3),
                      dtype=IMG_DTYPE)
    labs: List[int] = []
    names: List[str] = []
    n = 0

    for person, files in eligible:
        if per_id is not None and len(files) > per_id:
            pick = rng.choice(len(files), size=per_id, replace=False)
            files = [files[i] for i in sorted(pick)]
        ident = len(names)
        loaded = 0
        for f in files:
            bgr = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            images[n] = cv2.resize(rgb, (cfg.image_size, cfg.image_size),
                                   interpolation=cv2.INTER_AREA)
            labs.append(ident)
            n += 1
            loaded += 1
        if loaded == 0:
            continue
        names.append(person.name)
        if len(names) % 500 == 0:
            _log(f"folder: {len(names)}/{len(eligible)} identities, {n} images")

    if n == 0:
        raise RuntimeError(
            f"no usable images under {root} "
            f"(need >= {cfg.folder_min_images_per_person} per identity)"
        )

    # Unreadable files leave the tail unused; trim rather than ship zeros.
    images = images[:n]
    labels = np.asarray(labs, dtype=LABEL_DTYPE)
    return Dataset(images, labels, names)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def load_dataset(cfg: Config) -> Dataset:
    """Load whichever source the config names, and validate the contract."""
    source = cfg.data.source.lower()
    if source == "lfw":
        ds = load_lfw(cfg.data, cfg.paths.cache, cfg.seed)
    elif source == "folder":
        ds = load_folder(cfg.data, cfg.seed)
    else:
        raise ValueError(
            f"unknown data.source {cfg.data.source!r}; expected 'lfw' or 'folder'"
        )

    assert_images(ds.images, cfg.data.image_size, "images")
    assert_labels(ds.labels, n_expected=ds.images.shape[0], name="labels")
    if ds.labels.size and int(ds.labels.max()) >= len(ds.identity_names):
        raise ValueError("label id exceeds the identity name table")
    return ds


if __name__ == "__main__":  # smoke check
    # Runs without touching the network: builds a tiny folder dataset in a
    # temp directory and loads it back through the real code path.
    import tempfile

    from ..config import default_config

    cfg = default_config()
    rng = np.random.default_rng(0)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for pid in range(4):
            d = root / f"person_{pid:02d}"
            d.mkdir()
            base = rng.integers(60, 200, size=(3,), dtype=np.int16)
            for j in range(5):
                img = np.clip(
                    base + rng.normal(0, 12, size=(96, 96, 3)), 0, 255
                ).astype(np.uint8)
                cv2.imwrite(str(d / f"{j}.png"), img)

        cfg.data.source = "folder"
        cfg.data.folder_path = root
        cfg.data.folder_min_images_per_person = 3
        ds = load_dataset(cfg)

    print(ds.describe())
    assert ds.n == 20, ds.n
    assert ds.n_identities == 4
    assert ds.images.shape == (20, cfg.data.image_size, cfg.data.image_size, 3)
    assert ds.images.dtype == np.uint8
    print("loader OK")
