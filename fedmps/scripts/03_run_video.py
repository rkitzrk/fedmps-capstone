"""Phase 4 entry point: run a node's pipeline over a video file.

    python fedmps/scripts/03_run_video.py --video media/corridor.mp4

Detects faces, tracks them, gates on quality, embeds through the frozen
backbone and the federated adapter, matches against this node's protected
watchlist, votes across each track, and raises alerts as PENDING for local
review.

Writes an annotated copy of the video to media/, which doubles as the backup
demo recording - record one, and a live demo failing in the room stops being a
problem.

Before this does anything useful you need a case on the watchlist. Either open
one in the server console, or use --enrol to enrol a face directly from a
still image for testing.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2                                                       # noqa: E402
import numpy as np                                               # noqa: E402

from fedmps.config import default_config                         # noqa: E402
from fedmps.contracts import save_json                           # noqa: E402
from fedmps.models.backbone import get_backbone                  # noqa: E402
from fedmps.pipeline.match import (                              # noqa: E402
    VideoPipeline, get_selector, load_adapter,
)
from fedmps.privacy.template import (                            # noqa: E402
    binarise, derive_key, pack, protect,
)
from fedmps.store import key_fingerprint, open_store             # noqa: E402


def log(msg: str) -> None:
    print(f"[video] {msg}", flush=True)


def enrol_from_image(cfg, store, image_path: Path, name: str,
                     n_nodes: int, ttl_days: int) -> str:
    """Open a case straight from a still image, for testing without the UI."""
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read {image_path}")

    # Build a small JPEG thumbnail for the node dashboard's reference photo.
    # The operator needs to see who they are being asked to identify.
    thumb_size = 160
    h, w = bgr.shape[:2]
    scale = thumb_size / max(h, w)
    thumb_bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", thumb_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    reference_thumb: bytes | None = bytes(buf) if ok else None

    size = cfg.data.image_size
    rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (size, size),
                     interpolation=cv2.INTER_AREA)

    backbone = get_backbone(cfg)
    selector = get_selector(cfg)
    emb = backbone.embed(rgb[None, ...], progress=False)
    bits = binarise(emb, selector)
    ref_bytes = emb[0].astype("float32").tobytes()

    case_id = f"case-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    store.create_case(case_id, name, f"enrolled from {image_path.name}",
                      ttl_days=ttl_days, reference_thumb=reference_thumb)
    for node in range(n_nodes):
        key = derive_key(cfg.privacy.template_master_key, case_id, node,
                         cfg.privacy.template_bits)
        template = pack(protect(bits, key))[0]
        store.issue_template(case_id, node, template.tobytes(),
                             key_fingerprint(key),
                             reference_embedding=ref_bytes)

    log(f"enrolled '{name}' as {case_id}, distributed to {n_nodes} node(s)")
    log(f"  the photograph itself was not stored; "
        f"{cfg.privacy.template_bits // 8} bytes went to each node")
    return case_id


def main() -> int:
    ap = argparse.ArgumentParser(description="FedMPS node video pipeline")
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--node", type=int, default=0)
    ap.add_argument("--camera", default="cam-A")
    ap.add_argument("--out", type=Path, default=None,
                    help="annotated output video (default: media/<name>_annotated.mp4)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=None,
                    help="override pipeline.match_threshold")
    ap.add_argument("--vote", type=str, default=None,
                    help="override the track vote as K/N, e.g. 3/5")
    ap.add_argument("--enrol", type=Path, default=None,
                    help="enrol this still image as a case before running")
    ap.add_argument("--enrol-name", default="Test Subject")
    ap.add_argument("--fps-cap", type=float, default=4.0,
                    help="max wall-clock frames/sec to process (default 4); "
                         "prevents the detector from pinning 100%% CPU. "
                         "Set 0 to run as fast as possible.")
    ap.add_argument("--stop-on-alert", action="store_true", default=False,
                    help="Stop processing immediately once an alert is raised")
    ap.add_argument("--sample-fps", type=int, default=None,
                    help="Override pipeline.sample_fps (default 8, set 4 for faster scan)")
    args = ap.parse_args()

    cfg = default_config()
    if args.stop_on_alert:
        cfg.pipeline.stop_on_alert = True
    if args.sample_fps:
        cfg.pipeline.sample_fps = args.sample_fps
    if args.threshold is not None:
        cfg.pipeline.match_threshold = args.threshold
    if args.vote:
        k, _, n = args.vote.partition("/")
        cfg.pipeline.track_vote_k = int(k)
        cfg.pipeline.track_vote_n = int(n)

    if not args.video.exists():
        log(f"video not found: {args.video}")
        return 1

    store = open_store(cfg)

    if args.enrol:
        enrol_from_image(cfg, store, args.enrol, args.enrol_name,
                         n_nodes=cfg.partition.n_nodes, ttl_days=30)

    watchlist = store.templates_for_node(args.node)
    if not watchlist:
        log(f"node {args.node} has an empty watchlist. Open a case in the "
            f"server console, or pass --enrol <photo.jpg>. Running anyway: "
            f"detection, tracking and the quality gate will still be "
            f"exercised, but nothing can match.")

    out_path = args.out or (cfg.paths.media /
                            f"{args.video.stem}_annotated.mp4")

    pipeline = VideoPipeline(cfg, node_id=args.node, store=store)
    stats = pipeline.process_video(args.video, out_path=out_path,
                                   camera=args.camera,
                                   max_frames=args.max_frames,
                                   fps_cap=args.fps_cap or None)

    results_path = cfg.paths.results / f"video_node{args.node}.json"
    save_json(results_path, stats)

    print()
    log(f"annotated video : {out_path}")
    log(f"run statistics  : {results_path}")

    g, v = stats["quality_gate"], stats["voting"]
    print()
    log("what the quality gate did")
    log(f"  {g['passed']} of {g['total_faces']} faces passed "
        f"({g['rejection_rate']:.0%} discarded)")
    for reason, count in sorted(g["by_reason"].items(),
                                key=lambda kv: -kv[1]):
        if reason != "passed":
            log(f"    {reason:<14} {count}")

    print()
    log("what track-level voting did")
    log(f"  rule: {v['rule']} above {v['threshold']}")
    log(f"  {v['frame_level_hits']} frame-level hits -> "
        f"{v['track_level_alerts']} track-level alerts")
    if stats["alerts_raised"]:
        print()
        log(f"{stats['alerts_raised']} alert(s) are PENDING review. Open the "
            f"node console to confirm or reject them:")
        log("  streamlit run fedmps/apps/node_app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
