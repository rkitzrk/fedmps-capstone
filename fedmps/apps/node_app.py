"""Node console: the human gate, and the label buffer behind it.

    streamlit run fedmps/apps/node_app.py

This screen does three jobs at once, which is why the design report argues it
is the most important component in the system rather than an optional extra:

  It is the false-positive control. Nothing reaches the coordinating server
  until a named person here has looked at the crop and said yes.

  It is the legal safeguard. Every disclosure has a human attached to it, and
  every decision is written to an audit log that both sides can read.

  It is the label source. Confirm and reject are exactly the binary labels the
  federated verifier would train on. Without an operator reviewing alerts
  there is no supervised signal at a node at all, and the learning loop the
  project describes would have nothing to learn from.

Note what the watchlist tab shows: case references and key fingerprints. Not
photographs. This node cannot see the faces it is searching for, and does not
need to.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st                                          # noqa: E402

from fedmps.config import default_config                        # noqa: E402
from fedmps.store import (                                       # noqa: E402
    STATUS_CONFIRMED, STATUS_PENDING, STATUS_REJECTED, Store,
)

st.set_page_config(page_title="FedMPS · Node console",
                   page_icon="🎥", layout="wide")


@st.cache_resource
def _cfg():
    return default_config()


@st.cache_resource
def _store(_path: str) -> Store:
    return Store(Path(_path))


cfg = _cfg()
store = _store(str(cfg.paths.artifacts / "fedmps.sqlite"))

with st.sidebar:
    st.header("This node")
    node_id = st.number_input("Node id", min_value=0, max_value=31, value=0)
    operator = st.text_input("Reviewing as", value="operator@node0",
                             help="Recorded against every decision you make.")
    st.divider()
    st.caption(
        "Raw video, face crops of non-matches, embeddings and the label "
        "buffer never leave this device. Only a confirmed alert does, and "
        "only when you say so.")
    st.divider()
    st.subheader("Camera Feed Simulation")
    video_file = cfg.paths.media / f"node{int(node_id)}.mp4"
    if video_file.exists():
        if st.button(f"▶ Scan video (media/node{int(node_id)}.mp4)",
                     help="Process this node's camera video against the active watchlist"):
            with st.spinner(f"Scanning camera video for Node {int(node_id)}..."):
                import subprocess
                cmd = [
                    sys.executable,
                    "fedmps/scripts/03_run_video.py",
                    "--video", str(video_file),
                    "--node", str(int(node_id)),
                    "--camera", f"cam-{'ABCDE'[int(node_id) % 5]} (site {int(node_id)})",
                    "--fps-cap", "0",
                    "--sample-fps", "4",
                    "--stop-on-alert",
                ]
                subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2])
            st.success("Scan complete!")
            st.rerun()
    else:
        st.caption(f"No video file found at {video_file}")

node_id = int(node_id)
st.title(f"Node {node_id} review console")

pending = store.list_alerts(node_id=node_id, status=STATUS_PENDING)
confirmed = store.list_alerts(node_id=node_id, status=STATUS_CONFIRMED)
rejected = store.list_alerts(node_id=node_id, status=STATUS_REJECTED)
watchlist = store.templates_for_node(node_id)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Awaiting review", len(pending))
c2.metric("Confirmed", len(confirmed))
c3.metric("Rejected", len(rejected))
c4.metric("Cases watched", len(watchlist))

tab_queue, tab_watchlist, tab_labels = st.tabs(
    ["Review queue", "Watchlist", "Training labels"])


# --------------------------------------------------------------------------
# Review queue
# --------------------------------------------------------------------------

with tab_queue:
    if not pending:
        st.success("Nothing awaiting review.")
    else:
        st.caption(
            f"{len(pending)} candidate(s). Compare the reported photograph "
            f"against what the camera saw. Confirming releases this one crop "
            f"upstream; rejecting keeps it here and discards it. Opening an "
            f"alert pulls the reference photograph from the server and is "
            f"recorded in the audit log.")

    for a in pending:
        with st.container(border=True):
            cols = st.columns([1.1, 1.1, 2.4, 1.6, 1.8])

            # Who we are looking for. Pulled from the server only now, when a
            # human is actually reviewing, and the pull is written to the
            # audit log. Without this the operator is asked to confirm a face
            # they have never seen, which makes the human gate meaningless.
            ref = store.reference_thumb(a["case_id"], actor=operator)
            if ref:
                cols[0].image(ref, width=130, caption="Reported missing")
            else:
                cols[0].info("no reference photograph")

            crop = Path(a["crop_path"] or "")
            if crop.exists():
                cols[1].image(str(crop), width=130, caption="Seen by camera")
            else:
                cols[1].warning("crop missing")

            cols[2].markdown(
                f"**{a['case_name'] or a['case_id']}**  \n"
                f"`{a['case_id']}`  \n"
                f"camera **{a['camera']}** · track **{a['track_id']}**  \n"
                f"{a['created_utc'][:19].replace('T', ' ')} UTC")
            cols[3].metric("Match score", f"{a['score']:.3f}")
            cols[3].caption(f"{a['frames_voted']} of {a['frames_seen']} "
                            f"frames in this track agreed")

            if cols[4].button("Confirm — send upstream",
                              key=f"ok-{a['alert_id']}", type="primary"):
                store.review_alert(a["alert_id"], STATUS_CONFIRMED, operator)
                st.rerun()
            if cols[4].button("Reject — not this person",
                              key=f"no-{a['alert_id']}"):
                store.review_alert(a["alert_id"], STATUS_REJECTED, operator)
                st.rerun()

    if confirmed or rejected:
        with st.expander(f"Reviewed here ({len(confirmed) + len(rejected)})"):
            rows = [{
                "alert": a["alert_id"], "case": a["case_id"],
                "score": round(a["score"], 3), "status": a["status"],
                "by": a["reviewed_by"], "at": (a["reviewed_utc"] or "")[:19],
            } for a in (confirmed + rejected)]
            st.dataframe(rows, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------
# Watchlist
# --------------------------------------------------------------------------

with tab_watchlist:
    st.subheader("What this node is searching for")
    st.caption(
        f"{cfg.privacy.template_bits}-bit keyed codes. No photographs are "
        f"held here, and the same person enrolled at another node has an "
        f"entirely different pattern - so these watchlists cannot be "
        f"correlated across sites.")

    if not watchlist:
        st.info("No active cases. The server distributes them; closed and "
                "expired cases drop off automatically.")
    else:
        rows = [{
            "case": w["case_id"],
            "name": w["name"],
            "template (hex)": bytes(w["template"]).hex(),
            "key": w["key_fp"],
            "issued": w["issued_utc"][:19],
            "expires": (w["expires_utc"] or "—")[:10],
        } for w in watchlist]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption(f"{len(watchlist)} case(s) × "
                   f"{cfg.privacy.template_bits // 8} bytes = "
                   f"{len(watchlist) * cfg.privacy.template_bits // 8} bytes "
                   f"of watchlist in total.")


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------

with tab_labels:
    st.subheader("Training labels produced here")
    reviewed = len(confirmed) + len(rejected)
    st.write(
        "Every decision on the review queue is a labelled example. Confirms "
        "are positives; rejects are hard negatives — near-misses that this "
        "camera actually produced, which is the most useful negative there "
        "is. These stay on this device and only ever leave as averaged, "
        "clipped, noised and masked weight updates."
    )

    l1, l2, l3 = st.columns(3)
    l1.metric("Labelled examples", reviewed)
    l2.metric("Positives", len(confirmed))
    l3.metric("Hard negatives", len(rejected))

    if reviewed:
        precision = len(confirmed) / reviewed
        st.metric("Local precision", f"{precision:.0%}",
                  help="Of the alerts you reviewed, how many were real. This "
                       "is the number that decides whether the system is "
                       "usable, far more than any benchmark accuracy.")
        if precision < 0.3:
            st.warning(
                "Most alerts are false. Raise `pipeline.match_threshold` or "
                "`track_vote_k` in the config — an operator drowning in false "
                "positives stops reading them, and the human gate stops "
                "working.")

    st.divider()
    st.caption("Audit trail for this node")
    entries = [e for e in store.list_audit(400)
               if f"node_{node_id}" in e["actor"] or operator in e["actor"]]
    if entries:
        st.dataframe(entries[:100], use_container_width=True, hide_index=True)
    else:
        st.info("Nothing logged for this node yet.")
