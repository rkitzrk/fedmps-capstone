#!/usr/bin/env python3
"""
demo_reset.py — wipe all cases, templates, alerts, and audit entries
from the FedMPS store so the dashboards start clean for a fresh demo.

Usage:
    python demo_reset.py
"""
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB   = ROOT / "artifacts" / "fedmps.sqlite"

# Also clean up annotated videos, per-node result files, and alert snapshots
ANNOTATED  = sorted(ROOT.glob("media/*_annotated.mp4"))
NODE_JSONS = sorted(ROOT.glob("results/video_node*.json"))
CROP_DIR   = ROOT / "artifacts" / "crops"
ALERTS_DIR = ROOT / "media" / "alerts"

def wipe_store(db_path: Path) -> None:
    if not db_path.exists():
        print(f"[reset] no database found at {db_path} — nothing to clear")
        return
    with closing(sqlite3.connect(db_path)) as con:
        tables = ["alerts", "templates", "cases", "audit"]
        for t in tables:
            try:
                n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                con.execute(f"DELETE FROM {t}")
                print(f"[reset]   {t:<12} {n} row(s) deleted")
            except sqlite3.OperationalError:
                pass          # table may not exist yet
        con.commit()
    print(f"[reset] ✅ store wiped: {db_path}")

def remove_files(paths: list, label: str) -> None:
    removed = 0
    for p in paths:
        try:
            p.unlink()
            removed += 1
        except FileNotFoundError:
            pass
    if removed:
        print(f"[reset] ✅ {label}: {removed} file(s) removed")
    else:
        print(f"[reset]    {label}: nothing to remove")

def wipe_crops(crops_dir: Path) -> None:
    if not crops_dir.exists():
        return
    files = list(crops_dir.glob("*"))
    for f in files:
        try:
            f.unlink()
        except Exception:
            pass
    print(f"[reset] ✅ crops: {len(files)} file(s) removed")

if __name__ == "__main__":
    print()
    print("=" * 52)
    print("  FedMPS Demo Reset")
    print("=" * 52)
    wipe_store(DB)
    remove_files(ANNOTATED,  "annotated videos")
    remove_files(NODE_JSONS, "node result JSONs")
    wipe_crops(CROP_DIR)
    wipe_crops(ALERTS_DIR)
    print()
    print("[reset] dashboards will refresh automatically on next page load.")
    print("        Reload http://localhost:8501 and http://localhost:8502")
    print("=" * 52)
    print()
