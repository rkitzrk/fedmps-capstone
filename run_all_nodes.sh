#!/usr/bin/env bash
# run_all_nodes.sh
# ------------------------------------------------------------------
# Runs the FedMPS video pipeline for all 5 nodes in sequence.
# Node 0 enrolls the missing person; nodes 1-4 pick up the same case
# from the shared SQLite store.
#
# Usage:
#   bash run_all_nodes.sh [enrol_image] [name] [fps_cap]
#
# Example:
#   bash run_all_nodes.sh media/person.jpg "Missing Person" 4
#
# After this finishes, open the dashboards:
#   streamlit run fedmps/apps/node_app.py   --server.port 8502
#   streamlit run fedmps/apps/server_app.py --server.port 8501
# ------------------------------------------------------------------

set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$PROJ/.venv"

source "$VENV/bin/activate"
cd "$PROJ"

ENROL_IMAGE="${1:-media/person.jpg}"
ENROL_NAME="${2:-Missing Person}"
FPS_CAP="${3:-4}"

echo ""
echo "=================================================="
echo "  FedMPS — All-node video pipeline"
echo "=================================================="
echo "  Enrolment photo : $ENROL_IMAGE"
echo "  Subject name    : $ENROL_NAME"
echo "  Wall-clock cap  : ${FPS_CAP} fps/node (CPU throttle)"
echo ""

# ── Node 0: enrol the case + scan footage
echo "---------- Node 0 (enrolment + scan) ----------"
python fedmps/scripts/03_run_video.py \
    --video  media/node0.mp4 \
    --node   0 \
    --camera "cam-A (site 0)" \
    --enrol  "$ENROL_IMAGE" \
    --enrol-name "$ENROL_NAME" \
    --fps-cap "$FPS_CAP"
echo ""

# ── Nodes 1-4: case already in store; just scan
for NODE in 1 2 3 4; do
    VIDEO="media/node${NODE}.mp4"
    if [ ! -f "$VIDEO" ]; then
        echo "---------- Node $NODE — skipped (no video at $VIDEO) ----------"
        continue
    fi
    echo "---------- Node $NODE (scan only) ----------"
    python fedmps/scripts/03_run_video.py \
        --video  "$VIDEO" \
        --node   "$NODE" \
        --camera "cam-$(printf '%s' 'BCDE' | cut -c$NODE) (site $NODE)" \
        --fps-cap "$FPS_CAP"
    echo ""
done

# ── Summary
echo "=================================================="
echo "  All nodes complete."
echo ""
echo "  Annotated videos:"
ls -1 media/*_annotated.mp4 2>/dev/null || echo "  (none found)"
echo ""
echo "  Alerts per node:"
for NODE in 0 1 2 3 4; do
    RFILE="results/video_node${NODE}.json"
    if [ -f "$RFILE" ]; then
        COUNT=$(python3 -c "import json; d=json.load(open('$RFILE')); print(d.get('alerts_raised',0))")
        echo "    node $NODE : $COUNT alert(s) PENDING review"
    fi
done

echo ""
echo "  Open the dashboards (two terminals):"
echo ""
echo "    Terminal 1 — Node review queue:"
echo "      source .venv/bin/activate"
echo "      streamlit run fedmps/apps/node_app.py --server.port 8502"
echo "      → http://localhost:8502"
echo ""
echo "    Terminal 2 — Server case console:"
echo "      source .venv/bin/activate"
echo "      streamlit run fedmps/apps/server_app.py --server.port 8501"
echo "      → http://localhost:8501"
echo "=================================================="
