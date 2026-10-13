"""Build one surveillance clip per node, from that node's own data.

Your own corridor footage is what belongs in the final demo. This is for
testing and rehearsal: deterministic clips, built from the same faces the
system was actually evaluated on.

One clip per node, and they differ in two ways that matter:

  Different people. A node's distractors are drawn from the identities that
  node holds. Site 3's camera sees site 3's regulars.

  Different camera. Each node's degradation profile - face size, blur, noise,
  gamma, JPEG quality - is applied to the whole frame, so node 1's footage is
  visibly softer or more compressed than node 4's. That heterogeneity is the
  thing federated learning is supposed to cope with, so the demo should show
  it rather than assert it.

The missing person is the same in every clip and is drawn from the held-out
identities, so no node trained on them. That is the honest case: a stranger to
every site, which is what a real missing-person search looks like.

    python fedmps/scripts/make_demo_video.py --celeba-identities <path>

Writes media/node0.mp4 ... media/nodeN.mp4 and media/person.jpg.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2                                                       # noqa: E402
import numpy as np                                               # noqa: E402

from fedmps.config import default_config                         # noqa: E402
from fedmps.contracts import Artefacts                           # noqa: E402
from fedmps.data.degrade import degrade_image, resolve_params    # noqa: E402

W, H = 640, 480


def log(msg: str) -> None:
    print(f"[demo-video] {msg}", flush=True)


# --------------------------------------------------------------------------
# Face sources
# --------------------------------------------------------------------------

def faces_from_celeba(zip_path: Path, identity_txt: Path,
                      pids: List[int], per_id: int = 3
                      ) -> Dict[int, List[np.ndarray]]:
    """Load a few images for each CelebA person id."""
    from fedmps.data.celeba import CELEBA_CROP, list_images, load_identity_map

    ident = load_identity_map(identity_txt)
    by_name = {Path(n).name: n for n in list_images(zip_path)}
    wanted = set(pids)

    groups: Dict[int, List[str]] = {}
    for fname, pid in ident.items():
        if pid in wanted and fname in by_name:
            groups.setdefault(pid, []).append(by_name[fname])

    out: Dict[int, List[np.ndarray]] = {}
    with zipfile.ZipFile(zip_path) as z:
        for pid, entries in groups.items():
            imgs = []
            for entry in sorted(entries)[:per_id]:
                bgr = cv2.imdecode(np.frombuffer(z.read(entry), np.uint8),
                                   cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                if bgr.shape[0] >= 196 and bgr.shape[1] >= 168:
                    bgr = bgr[CELEBA_CROP[0], CELEBA_CROP[1]]
                imgs.append(bgr)
            if imgs:
                out[pid] = imgs
    return out


def faces_from_lfw(folder: Path, names: List[str], per_id: int = 3
                   ) -> Dict[str, List[np.ndarray]]:
    out: Dict[str, List[np.ndarray]] = {}
    for name in names:
        files = sorted(folder.joinpath(name).glob("*.jpg"))[:per_id]
        imgs = []
        for f in files:
            bgr = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if bgr is not None:
                imgs.append(bgr[61:189, 61:189])
        if imgs:
            out[name] = imgs
    return out


# --------------------------------------------------------------------------
# Frame composition
# --------------------------------------------------------------------------

def corridor_background(rng: np.random.Generator, node_id: int) -> np.ndarray:
    """A dull indoor scene. Each site gets a slightly different room."""
    bg = np.zeros((H, W, 3), dtype=np.uint8)
    tint = 6 * (node_id % 3)
    for y in range(H):
        shade = int(70 + 60 * (y / H))
        bg[y, :] = (shade - 8 + tint, shade - 4, shade - tint)
    cv2.rectangle(bg, (0, int(H * 0.62)), (W, H), (92, 96, 104), -1)
    for x in range(0, W, 80):
        cv2.line(bg, (x, int(H * 0.62)), (int(x * 1.35) - 120, H),
                 (84, 88, 96), 1)
    door_x = 40 + 30 * (node_id % 4)
    cv2.rectangle(bg, (door_x, 60), (door_x + 110, 300), (120, 124, 130), -1)
    cv2.rectangle(bg, (470, 40), (610, 210), (140, 150, 160), -1)
    return np.clip(bg.astype(np.int16) + rng.normal(0, 3, bg.shape),
                   0, 255).astype(np.uint8)


def paste(frame: np.ndarray, face: np.ndarray, cx: int, cy: int,
          size: int) -> None:
    size = max(16, size)
    f = cv2.resize(face, (size, size), interpolation=cv2.INTER_AREA)
    x1, y1 = cx - size // 2, cy - size // 2
    x2, y2 = x1 + size, y1 + size
    fx1, fy1 = max(0, -x1), max(0, -y1)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return
    frame[y1:y2, x1:x2] = f[fy1:fy1 + (y2 - y1), fx1:fx1 + (x2 - x1)]


def face_size_for(params) -> tuple:
    """How large this node's camera renders a face, in frame pixels.

    The degradation profile describes a *tight crop* being resampled to
    `face_px`. A detected bounding box in a frame is roughly 2.4x that crop,
    and the quality gate needs at least `min_face_px` of inter-ocular
    distance, which is about 0.42 of the box. Sizing the pasted face from the
    profile - rather than shrinking the whole frame by the same ratio - is
    what keeps a poor camera visibly poor while still producing footage a
    detector can work with. Applying the crop ratio to a 640x480 frame was
    the earlier mistake: it rejected 87-100% of faces as too small.
    """
    near = int(np.clip(params.face_px * 3.6, 90, 240))
    far = int(np.clip(params.face_px * 2.2, 60, 160))
    return far, near


def apply_camera(frame: np.ndarray, params, cfg, rng,
                 face_px_in_frame: int) -> np.ndarray:
    """Put the whole frame through this node's camera.

    Blur, exposure, sensor noise and compression are applied at frame
    resolution. Face *size* is handled when the face is pasted, not by
    shrinking the scene, so each site's footage carries its own character
    without the detector losing every face.
    """
    small = frame

    if params.blur_kernel > 1:
        k = int(params.blur_kernel) | 1
        small = cv2.GaussianBlur(small, (k, k), 0)
    if abs(params.gamma - 1.0) > 1e-3:
        lut = np.clip(((np.arange(256) / 255.0) ** params.gamma) * 255, 0, 255)
        small = cv2.LUT(small, lut.astype(np.uint8))
    if params.noise_sigma > 0:
        small = np.clip(small.astype(np.float32)
                        + rng.normal(0, params.noise_sigma, small.shape),
                        0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", small,
                           [int(cv2.IMWRITE_JPEG_QUALITY),
                            int(np.clip(params.jpeg_quality, 15, 100))])
    if ok:
        small = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return small


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Per-node synthetic test clips")
    ap.add_argument("--celeba-zip", type=Path, default=None)
    ap.add_argument("--celeba-identities", type=Path, default=None)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--distractors", type=int, default=3)
    args = ap.parse_args()

    cfg = default_config()
    art = Artefacts.load(cfg)
    rng = np.random.default_rng(cfg.seed)
    names = art.identity_names
    held_out = [int(i) for i in art.partition["test_identities"]]
    if not held_out:
        log("no held-out identities in the partition")
        return 1

    is_celeba = bool(names) and names[0].startswith("celeb_")
    zip_path = args.celeba_zip or (Path.home() / "Downloads" / "archive.zip")
    id_txt = args.celeba_identities or (Path.home() / "Downloads"
                                        / "identity_CelebA.txt")

    # One missing person, unknown to every node.
    target_idx = held_out[len(held_out) // 2]
    target_name = names[target_idx]
    log(f"missing person : {target_name} (held-out identity {target_idx})")

    # Each node's distractors come from that node's own people.
    node_ids = {}
    for k in range(art.n_nodes):
        own = [int(i) for i in art.node(k)["identities"]][:args.distractors]
        node_ids[k] = own

    if is_celeba:
        if not zip_path.exists() or not id_txt.exists():
            log(f"need CelebA at {zip_path} and {id_txt}")
            return 1
        wanted = [int(target_name.split("_")[1])]
        for k, ids in node_ids.items():
            wanted += [int(names[i].split("_")[1]) for i in ids]
        faces = faces_from_celeba(zip_path, id_txt, wanted)

        def get(idx: int) -> List[np.ndarray]:
            return faces.get(int(names[idx].split("_")[1]), [])
    else:
        folder = cfg.paths.cache / "lfw_home" / "lfw_funneled"
        wanted = [target_name] + [names[i] for ids in node_ids.values()
                                  for i in ids]
        faces = faces_from_lfw(folder, wanted)

        def get(idx: int) -> List[np.ndarray]:
            return faces.get(names[idx], [])

    target_imgs = get(target_idx)
    if len(target_imgs) < 2:
        log(f"need at least 2 images of {target_name}; found {len(target_imgs)}")
        return 1

    cfg.paths.media.mkdir(parents=True, exist_ok=True)
    ref = cfg.paths.media / "person.jpg"
    # Enrol a *different* photograph than the one that appears in the footage.
    cv2.imwrite(str(ref), cv2.resize(target_imgs[1], (256, 256)))
    log(f"reference photograph -> {ref}")

    ranges = cfg.degrade.ranges
    written = []

    for k in range(art.n_nodes):
        profile = art.node(k)["degrade_params"]
        params = resolve_params(profile, ranges, np.random.default_rng(k), 0.0)

        distract = [img for i in node_ids[k] for img in get(i)[:1]]
        if not distract:
            distract = [target_imgs[0]]

        out_path = cfg.paths.media / f"node{k}.mp4"
        writer = cv2.VideoWriter(str(out_path),
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 args.fps, (W, H))
        bg = corridor_background(rng, k)
        frng = np.random.default_rng(cfg.seed + k)

        for i in range(args.frames):
            frame = bg.copy()
            t = i / max(1, args.frames - 1)

            far, _ = face_size_for(params)
            for j, face in enumerate(distract):
                phase = (t + 0.25 * (j + 1)) % 1.0
                paste(frame, face, int(W * phase),
                      int(H * 0.42 + 18 * np.sin(6.0 * phase + j)),
                      far + 10 * j)

            face_px = 0
            if t > 0.15:
                tt = (t - 0.15) / 0.85
                far, near = face_size_for(params)
                face_px = int(far + (near - far) * tt)
                paste(frame, target_imgs[0], int(W * (0.30 + 0.40 * tt)),
                      int(H * (0.34 + 0.14 * tt)), face_px)

            writer.write(apply_camera(frame, params, cfg, frng,
                                      max(face_px, 90)))
        writer.release()
        written.append(out_path)
        far, near = face_size_for(params)
        log(f"node {k}: {len(distract)} local distractor(s), "
            f"camera[blur={params.blur_kernel} jpeg={params.jpeg_quality} "
            f"face {far}->{near}px] -> {out_path.name}")

    print()
    log("enrol the case once, then run each node against its own footage:")
    log(f"  python fedmps/scripts/03_run_video.py --video media/node0.mp4 "
        f"--node 0 --enrol media/person.jpg --enrol-name \"{target_name}\"")
    for k in range(1, art.n_nodes):
        log(f"  python fedmps/scripts/03_run_video.py "
            f"--video media/node{k}.mp4 --node {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
