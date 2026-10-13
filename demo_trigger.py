#!/usr/bin/env python3
"""
demo_trigger.py — trigger node video analysis against watchlist.

Usage (Recommended Demo Flow):
    1. Upload photo in Central Dashboard (http://localhost:8501)
    2. Run node(s) against the video feed:
         python demo_trigger.py --node 0
         python demo_trigger.py --all

Usage (CLI-only standalone):
    python demo_trigger.py --image media/person-1.jpg --name "Alice" --node 0
    python demo_trigger.py --image media/person-1.jpg --name "Alice" --all
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from fedmps.config import default_config
from fedmps.store import CASE_OPEN, open_store


def run_node(node: int, image: Path | None, name: str, enrol: bool, fps: float,
             max_frames: int | None = None, stop_on_alert: bool = True,
             sample_fps: int = 4) -> int:
    cmd = [
        sys.executable,
        "fedmps/scripts/03_run_video.py",
        "--video",  f"media/node{node}.mp4",
        "--node",   str(node),
        "--camera", f"cam-{'ABCDE'[node]} (site {node})",
        "--fps-cap", str(fps),
        "--sample-fps", str(sample_fps),
    ]
    if stop_on_alert:
        cmd.append("--stop-on-alert")
    if max_frames:
        cmd += ["--max-frames", str(max_frames)]
    if enrol and image:
        cmd += ["--enrol", str(image), "--enrol-name", name]

    print()
    print(f"{'='*58}")
    print(f"  Node {node} {'(enrol + scan)' if (enrol and image) else '(scan against distributed watchlist)'}")
    print(f"  Video : media/node{node}.mp4")
    if enrol and image:
        print(f"  Photo : {image}")
        print(f"  Name  : {name}")
    print(f"{'='*58}")

    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode


def main() -> int:
    ap = argparse.ArgumentParser(
        description="FedMPS demo trigger: run edge nodes against video feeds"
    )
    ap.add_argument("--node", type=int, default=None,
                    help="Which node to run (0-4). If omitted with --all, runs all.")
    ap.add_argument("--all", action="store_true",
                    help="Run all 5 nodes in sequence")
    ap.add_argument("--image", type=Path, default=None,
                    help="Optional: enrolment photo (if enrolling via CLI instead of Central Dashboard)")
    ap.add_argument("--name", default="Missing Person",
                    help="Optional: Subject name (used only if --image is supplied)")
    ap.add_argument("--fps", type=float, default=0.0,
                    help="FPS cap per node (default 0 = max speed; set >0 to throttle)")
    ap.add_argument("--max-frames", type=int, default=80,
                    help="Max frames to process (default 80 for fast demo; set 0 for full video)")
    args = ap.parse_args()

    max_frames_arg = None if args.max_frames == 0 else args.max_frames

    cfg = default_config()
    store = open_store(cfg)
    active_cases = store.list_cases(status=CASE_OPEN)

    # Determine whether we need to enrol from CLI or if we already have a case from Central Dashboard
    enrol_cli = False
    if args.image is not None:
        if not args.image.exists():
            print(f"[trigger] ERROR: image not found: {args.image}")
            return 1
        enrol_cli = True
    else:
        # No image supplied via CLI — check if case exists in Central Dashboard
        if not active_cases:
            print("[trigger] ERROR: No active cases found in the system, and no --image was supplied.")
            print("\nRecommended Demo Flow:")
            print("  1. Open Central Dashboard at http://localhost:8501")
            print("  2. Go to 'Open a case' tab, upload photo, and click 'Create case and distribute'")
            print("  3. Run: python demo_trigger.py --node 0  (or --all)")
            print("\nAlternative (CLI-only flow):")
            print("  python demo_trigger.py --image media/person-1.jpg --name 'Alice' --node 0")
            return 1

        print("[trigger] Found active case from Central Dashboard:")
        for c in active_cases:
            print(f"          • [{c.case_id}] {c.name}")
        print("[trigger] Running video pipeline against distributed watchlist.")

    if args.all:
        for node in range(5):
            video = ROOT / f"media/node{node}.mp4"
            if not video.exists():
                print(f"[trigger] skipping node {node} — no video at {video}")
                continue
            # If enrol_cli is True, node 0 does the enrolment; otherwise neither does
            run_node(node, args.image, args.name,
                     enrol=(enrol_cli and node == 0), fps=args.fps,
                     max_frames=max_frames_arg)

        print()
        print("=" * 58)
        print("  All nodes complete.")
        print("  Open the dashboards to review alerts:")
        print("    http://localhost:8502  ← Node review queue")
        print("    http://localhost:8501  ← Central server console")
        print("=" * 58)
        return 0

    if args.node is None:
        ap.print_help()
        print("\n[trigger] Please specify which node to run: --node <0-4> or --all")
        return 1

    node = args.node
    video = ROOT / f"media/node{node}.mp4"
    if not video.exists():
        print(f"[trigger] ERROR: no video at {video}")
        return 1

    rc = run_node(node, args.image, args.name,
                  enrol=(enrol_cli and node == 0), fps=args.fps,
                  max_frames=max_frames_arg)
    if rc == 0:
        print()
        print("[trigger] Scan complete. Check alerts in Node dashboard:")
        print("          http://localhost:8502")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

