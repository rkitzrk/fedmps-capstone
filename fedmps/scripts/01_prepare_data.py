"""Phase 1 entry point: images in, embeddings and a federated partition out.

Run from the project root:

    python fedmps/scripts/01_prepare_data.py

Everything after this reads the artefacts this script writes and never touches
an image file again.

Steps:
  1. load a face dataset and canonicalise it
  2. split identities across nodes, disjointly, and give each node a camera
  3. degrade every image with its owning node's camera
  4. embed both the clean and the degraded version with the frozen backbone
  5. write the artefacts and a meta.json recording exactly how it was made
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Allow `python fedmps/scripts/01_prepare_data.py` from the project root
# without needing the package installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fedmps.config import default_config                       # noqa: E402
from fedmps.contracts import (                                 # noqa: E402
    F_EMB_CLEAN, F_EMB_DEG, F_EMB_NONMATE, F_IDENTITY_NAMES, F_LABELS,
    F_META, F_PARTITION, F_SAMPLES, assert_aligned, assert_embeddings,
    save_json, save_npy,
)
from fedmps.data.degrade import degrade_image, resolve_params  # noqa: E402
from fedmps.data.loader import load_dataset                    # noqa: E402
from fedmps.data.partition import (                            # noqa: E402
    build_partition, describe_partition, profile_for_each_image,
)
from fedmps.data.celeba import (                               # noqa: E402
    has_identity_labels, iter_crops, resolvable_fpir, sample_names,
    sample_nonmate_names, validate_against,
)
from fedmps.models.backbone import get_backbone                # noqa: E402

N_SAMPLES_KEPT = 32


def log(msg: str) -> None:
    print(f"[prepare] {msg}", flush=True)


def degrade_all(images: np.ndarray, profile_ids: np.ndarray,
                partition: Dict[str, Any], cfg) -> np.ndarray:
    """Degrade every image with the camera of the node that owns it."""
    out = np.empty_like(images)
    ranges = cfg.degrade.ranges
    profiles = {
        k: partition["nodes"][f"node_{k}"]["degrade_params"]
        for k in range(partition["n_nodes"])
    }
    for i in range(images.shape[0]):
        # Per-image seed so the whole degradation pass is reproducible
        # regardless of ordering or batching.
        rng = np.random.default_rng((cfg.seed * 7919 + i) % (2 ** 63))
        params = resolve_params(profiles[int(profile_ids[i])], ranges, rng,
                                cfg.degrade.per_image_jitter)
        out[i] = degrade_image(images[i], params, cfg.degrade, rng)
        if i % 500 == 0:
            log(f"degrading {i}/{images.shape[0]}")
    return out


def mean_same_identity_similarity(emb: np.ndarray, labels: np.ndarray,
                                  rng: np.random.Generator,
                                  n_pairs: int = 2000) -> float:
    """Average cosine similarity between two images of the same person.

    A cheap health check. If this collapses toward the different-identity
    number, the embeddings carry no usable identity signal.
    """
    by_id: Dict[int, np.ndarray] = {}
    for ident in np.unique(labels):
        idx = np.where(labels == ident)[0]
        if idx.size >= 2:
            by_id[int(ident)] = idx
    if not by_id:
        return float("nan")

    ids = list(by_id)
    sims = []
    for _ in range(n_pairs):
        ident = ids[int(rng.integers(len(ids)))]
        a, b = rng.choice(by_id[ident], size=2, replace=False)
        sims.append(float(emb[a] @ emb[b]))
    return float(np.mean(sims))


def mean_cross_identity_similarity(emb: np.ndarray, labels: np.ndarray,
                                   rng: np.random.Generator,
                                   n_pairs: int = 2000) -> float:
    """Average cosine similarity between two different people."""
    n = emb.shape[0]
    sims = []
    tries = 0
    while len(sims) < n_pairs and tries < n_pairs * 10:
        a, b = rng.integers(0, n, size=2)
        tries += 1
        if labels[a] == labels[b]:
            continue
        sims.append(float(emb[a] @ emb[b]))
    return float(np.mean(sims)) if sims else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description="FedMPS Phase 1 data preparation")
    ap.add_argument("--source", choices=["lfw", "folder", "celeba"],
                    default=None)
    ap.add_argument("--folder", type=Path, default=None,
                    help="root directory when --source folder")
    ap.add_argument("--backbone", default=None,
                    help="force one backbone: insightface | facenet | dummy")
    ap.add_argument("--nodes", type=int, default=None)
    ap.add_argument("--max-identities", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", choices=["cpu", "cuda"], default=None,
                    help="cuda needs onnxruntime-gpu installed; the embedding "
                         "step is the whole cost of this script and a GPU "
                         "turns hours into minutes")
    ap.add_argument("--max-images-per-identity", type=int, default=None,
                    help="cap images kept per person; fewer per person and "
                         "more people is the better trade for recognition")
    ap.add_argument("--celeba-identities", type=Path, default=None,
                    help="identity_CelebA.txt; required for --source celeba")
    ap.add_argument("--celeba-zip", type=Path, default=None,
                    help="CelebA archive to draw extra non-mate probes from")
    ap.add_argument("--celeba-nonmates", type=int, default=20000,
                    help="how many non-mate probes to add from CelebA")
    ap.add_argument("--no-align", action="store_true",
                    help="skip SCRFD detection and ArcFace landmark "
                         "alignment before embedding. Only for reproducing "
                         "the pre-fix numbers: unaligned crops cost roughly "
                         "0.2 of mean same-identity cosine and drag every "
                         "downstream metric with them")
    ap.add_argument("--out-suffix", default="",
                    help="suffix for the artifacts directory, e.g. "
                         "--out-suffix _noalign writes artifacts_noalign/ so "
                         "two runs can be compared side by side")
    ap.add_argument("--repair-cache", action="store_true",
                    help="delete the LFW cache and download it again; use "
                         "this after an interrupted download")
    args = ap.parse_args()

    align_report: Dict[str, Any] = {"enabled": False,
                                    "reason": "not an insightface run"}
    cfg = default_config()
    if args.no_align:
        cfg.backbone.align = False
    if args.out_suffix:
        cfg.paths.artifacts = (cfg.paths.artifacts.parent
                               / f"artifacts{args.out_suffix}")
    if args.seed is not None:
        cfg.seed = args.seed
    if args.source:
        cfg.data.source = args.source
    if args.folder:
        cfg.data.source = "folder"
        cfg.data.folder_path = args.folder
    if args.backbone:
        cfg.backbone.preference = [args.backbone]
    if args.nodes:
        cfg.partition.n_nodes = args.nodes
    if args.max_identities is not None:
        cfg.data.max_identities = args.max_identities
    if args.device:
        cfg.backbone.device = args.device
    if args.max_images_per_identity is not None:
        cfg.data.max_images_per_identity = args.max_images_per_identity
    if args.celeba_zip:
        cfg.data.celeba_zip = args.celeba_zip
    if args.celeba_identities:
        cfg.data.celeba_identity_file = args.celeba_identities

    if args.repair_cache:
        import shutil

        home = cfg.paths.cache / "lfw_home"
        if home.is_dir():
            size_mb = sum(f.stat().st_size for f in home.rglob("*")
                          if f.is_file()) / 1e6
            log(f"deleting the LFW cache at {home} ({size_mb:.1f} MB)")
            shutil.rmtree(home, ignore_errors=True)
        else:
            log(f"no LFW cache at {home}; nothing to repair")
        log("it will be downloaded again below (~200 MB) - do not interrupt")

    t0 = time.time()
    rng = np.random.default_rng(cfg.seed)

    if cfg.data.source == "celeba":
        # Streaming path: decode, degrade, embed and discard chunk by chunk.
        # Holding the whole set twice at working resolution would need several
        # gigabytes; this needs one chunk.
        from fedmps.data.celeba import build_index, estimate_runtime, stream_embed

        if not cfg.data.celeba_zip or not cfg.data.celeba_identity_file:
            log("--source celeba needs both --celeba-zip and "
                "--celeba-identities")
            return 1

        log("indexing CelebA (metadata only, no pixels decoded yet)")
        index = build_index(
            cfg.data.celeba_zip, cfg.data.celeba_identity_file,
            min_images=cfg.data.celeba_min_images_per_person,
            max_identities=cfg.data.max_identities,
            max_images_per_identity=cfg.data.max_images_per_identity,
            seed=cfg.seed)
        labels = index.labels
        identity_names = index.names
        n_images = index.n
        log(f"{n_images:,} images, {len(identity_names):,} identities")
        log(f"estimated embedding time: {estimate_runtime(n_images)}")

        log(f"partitioning across {cfg.partition.n_nodes} nodes")
        partition = build_partition(labels, cfg)
        print(describe_partition(partition))
        profile_ids = profile_for_each_image(partition, n_images)

        backbone = get_backbone(cfg)
        log("streaming: decode -> degrade -> embed -> discard")
        emb_clean, emb_deg = stream_embed(
            cfg.data.celeba_zip, index, profile_ids, partition, cfg, backbone,
            checkpoint_dir=cfg.paths.artifacts / "_stream")

        class _DS:
            pass
        ds = _DS()
        ds.labels = labels
        ds.identity_names = identity_names
        ds.n = n_images
        ds.n_identities = len(identity_names)
        ds.images = None
        degraded = None
        celeba_train_pids = index.pids
    else:
        # ---- 1. images --------------------------------------------------
        log(f"loading source={cfg.data.source}")
        ds = load_dataset(cfg)
        log(ds.describe())

        # ---- 1b. alignment ----------------------------------------------
        # Before anything is degraded or embedded, put every face on
        # ArcFace's own landmark template. Doing it here rather than inside
        # the embedder means the clean and degraded branches are framed
        # identically and differ only in pixel quality.
        if (cfg.backbone.align and cfg.backbone.align_once
                and cfg.backbone.preference[:1] == ["insightface"]):
            from fedmps.models.backbone import align_faces

            ds.images, align_report = align_faces(ds.images, cfg)
            cfg.data.image_size = int(align_report["out_size"])
            # The embedder must not detect again - these are already crops.
            cfg.backbone.align = False
            if align_report["miss_rate"] > cfg.backbone.max_align_miss_rate:
                log(f"ABORT: the detector missed "
                    f"{100 * align_report['miss_rate']:.1f}% of faces, over "
                    f"the {100 * cfg.backbone.max_align_miss_rate:.1f}% "
                    f"limit. Those rows fall back to a plain resize, so the "
                    f"embeddings would be a mix of aligned faces and "
                    f"unaligned ones and nothing measured on them would "
                    f"mean much. Lower backbone.det_thresh, raise "
                    f"backbone.det_size, or check the source crops.")
                return 2

        # ---- 2. partition -----------------------------------------------
        log(f"partitioning across {cfg.partition.n_nodes} nodes")
        partition = build_partition(ds.labels, cfg)
        print(describe_partition(partition))
        profile_ids = profile_for_each_image(partition, ds.n)

        # ---- 3. degradation ---------------------------------------------
        log("applying the degradation model")
        degraded = degrade_all(ds.images, profile_ids, partition, cfg)

        # ---- 4. embeddings ----------------------------------------------
        backbone = get_backbone(cfg)
        log(f"embedding {ds.n} clean images with {backbone.name}")
        emb_clean = backbone.embed(ds.images, ds.labels,
                                   batch_size=cfg.backbone.batch_size)
        log(f"embedding {ds.n} degraded images with {backbone.name}")
        emb_deg = backbone.embed(degraded, ds.labels,
                                 batch_size=cfg.backbone.batch_size)
        celeba_train_pids = None

    assert_embeddings(emb_clean, "embeddings_clean")
    assert_embeddings(emb_deg, "embeddings_deg")
    assert_aligned(emb_clean, emb_deg, ds.labels,
                   names=["embeddings_clean", "embeddings_deg", "labels"])

    # ---- 5. health check ------------------------------------------------
    same_clean = mean_same_identity_similarity(emb_clean, ds.labels, rng)
    same_deg = mean_same_identity_similarity(emb_deg, ds.labels, rng)
    cross_clean = mean_cross_identity_similarity(emb_clean, ds.labels, rng)
    cross_deg = mean_cross_identity_similarity(emb_deg, ds.labels, rng)

    print()
    print("  mean cosine similarity        clean     degraded")
    print(f"    same identity            {same_clean:+7.3f}     {same_deg:+7.3f}")
    print(f"    different identities     {cross_clean:+7.3f}     {cross_deg:+7.3f}")
    print(f"    separation               {same_clean - cross_clean:+7.3f}     "
          f"{same_deg - cross_deg:+7.3f}")
    print()
    if same_deg - cross_deg >= same_clean - cross_clean:
        log("WARNING: degradation did not reduce identity separation. "
            "The degradation model is too gentle - widen degrade.ranges "
            "before relying on any Phase 2 result.")

    # ---- 5b. extra non-mate population ----------------------------------
    nonmate_emb = None
    nonmate_profiles: list = []
    celeba_info: Dict[str, Any] = {"used": False}

    if args.celeba_zip and args.celeba_nonmates > 0:
        if not args.celeba_zip.exists():
            log(f"CelebA archive not found at {args.celeba_zip}; skipping")
        else:
            n_want = int(args.celeba_nonmates)
            log(f"adding {n_want} non-mate probes from {args.celeba_zip.name}")
            if not has_identity_labels(args.celeba_zip):
                log("  this archive has no identity labels, which is fine - a "
                    "non-mate only has to be someone the gallery does not "
                    "contain. It cannot be used for training.")

            names = sample_nonmate_names(
                args.celeba_zip, cfg.data.celeba_identity_file, n_want,
                cfg.seed, exclude_pids=celeba_train_pids)
            ranges = cfg.degrade.ranges
            profiles = {
                k: partition["nodes"][f"node_{k}"]["degrade_params"]
                for k in range(partition["n_nodes"])
            }

            chunks = []
            done = 0
            for batch in iter_crops(args.celeba_zip, names,
                                    cfg.data.image_size, chunk=2000):
                # Same degradation chain as every other probe, with camera
                # profiles drawn round robin from the same nodes. Clean faces
                # here would be trivially separable and FPIR would flatter us.
                deg = np.empty_like(batch)
                for j in range(batch.shape[0]):
                    k = (done + j) % partition["n_nodes"]
                    rng_j = np.random.default_rng(
                        (cfg.seed * 31337 + done + j) % (2 ** 63))
                    params = resolve_params(profiles[k], ranges, rng_j,
                                            cfg.degrade.per_image_jitter)
                    deg[j] = degrade_image(batch[j], params, cfg.degrade, rng_j)
                    nonmate_profiles.append(k)
                chunks.append(backbone.embed(
                    deg, batch_size=cfg.backbone.batch_size, progress=False))
                done += batch.shape[0]
                log(f"  non-mates {done}/{len(names)}")

            nonmate_emb = np.concatenate(chunks, axis=0) if chunks else None

    if nonmate_emb is not None:
        assert_embeddings(nonmate_emb, "embeddings_nonmate")
        held_out_nonmates = emb_deg[np.asarray(partition["test_idx"])]
        gallery_probe = emb_clean[np.asarray(partition["test_idx"])][:200]
        check = validate_against(nonmate_emb, held_out_nonmates, gallery_probe)
        total_nm = nonmate_emb.shape[0] + len(partition["test_idx"])
        celeba_info = {
            "used": True,
            "archive": str(args.celeba_zip),
            "n_added": int(nonmate_emb.shape[0]),
            "total_nonmates": int(total_nm),
            "fpir_resolvable_before": resolvable_fpir(len(partition["test_idx"])),
            "fpir_resolvable_after": resolvable_fpir(total_nm),
            "validation": check,
        }
        print()
        print(f"  non-mate population: {len(partition['test_idx'])} held-out "
              f"+ {nonmate_emb.shape[0]} CelebA = {total_nm}")
        print(f"  finest measurable FPIR: "
              f"{celeba_info['fpir_resolvable_before']:.1e} -> "
              f"{celeba_info['fpir_resolvable_after']:.1e}")
        print(f"  population check: {check['reason']}")
        print(f"    top-score mean  CelebA {check['celeba_mean_top_score']:+.4f}"
              f"   held-out {check['lfw_mean_top_score']:+.4f}"
              f"   standardised gap {check['standardised_gap']:+.3f}")
        if not check["ok"]:
            log("WARNING: the two non-mate populations are not comparable. "
                "Any FPIR computed from them would be misleading. Check "
                "CELEBA_CROP in fedmps/data/celeba.py before reporting.")
        print()

    # ---- 6. write -------------------------------------------------------
    a = cfg.paths.artifacts
    save_npy(a / F_EMB_CLEAN, emb_clean)
    save_npy(a / F_EMB_DEG, emb_deg)
    save_npy(a / F_LABELS, ds.labels)
    save_json(a / F_PARTITION, partition)
    save_json(a / F_IDENTITY_NAMES, ds.identity_names)
    if nonmate_emb is not None:
        save_npy(a / F_EMB_NONMATE, nonmate_emb)

    if ds.images is not None and degraded is not None:
        keep = rng.choice(ds.n, size=min(N_SAMPLES_KEPT, ds.n), replace=False)
        np.savez_compressed(a / F_SAMPLES,
                            clean=ds.images[keep], degraded=degraded[keep],
                            labels=ds.labels[keep], index=keep.astype(np.int32))

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backbone": backbone.name,
        "backbone_is_synthetic": backbone.name == "dummy",
        "n_images": int(ds.n),
        "n_identities": int(ds.n_identities),
        "image_size": cfg.data.image_size,
        "elapsed_sec": round(time.time() - t0, 1),
        "celeba_nonmates": celeba_info,
        "nonmate_profiles": nonmate_profiles,
        "alignment": align_report,
        "health": {
            "same_identity_clean": same_clean,
            "same_identity_degraded": same_deg,
            "cross_identity_clean": cross_clean,
            "cross_identity_degraded": cross_deg,
        },
        "config": cfg.to_dict(),
    }
    save_json(a / F_META, meta)

    log(f"wrote artefacts to {a}")
    log(f"done in {meta['elapsed_sec']}s")
    if backbone.name == "dummy":
        print()
        log("NOTE: the synthetic backbone was used. Downstream phases will "
            "run correctly but the numbers are not reportable. Install "
            "insightface or facenet-pytorch and re-run before writing results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
