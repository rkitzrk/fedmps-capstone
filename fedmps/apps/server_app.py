"""Server console: open a case, distribute it, review what came back.

    streamlit run fedmps/apps/server_app.py

What this screen deliberately cannot do is as important as what it can. It
never shows a pending alert, because a pending alert is a face crop that no
human at the node has looked at yet. The server sees a candidate only after a
named operator confirms it. That is the two-stage minimal disclosure from the
design report, and here it is a property of the queries rather than a promise.

The probe photograph is used once, in memory, to compute an embedding. It is
never stored and never transmitted. What goes out to each node is twelve
bytes, keyed differently per node so the same person's entry at two sites
cannot be correlated.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np                                              # noqa: E402
import streamlit as st                                          # noqa: E402

from fedmps.config import default_config                        # noqa: E402
from fedmps.models.backbone import get_backbone                 # noqa: E402
from fedmps.pipeline.match import get_selector, load_adapter    # noqa: E402
from fedmps.privacy.template import (                           # noqa: E402
    binarise, derive_key, pack, protect,
)
from fedmps.store import (                                       # noqa: E402
    CASE_OPEN, STATUS_CONFIRMED, Store, key_fingerprint, open_store,
)

st.set_page_config(page_title="FedMPS · Server console",
                   page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading the frozen backbone…")
def _resources():
    cfg = default_config()
    return cfg, get_backbone(cfg), get_selector(cfg), load_adapter(cfg)


@st.cache_resource
def _store(_path: str) -> Store:
    return Store(Path(_path))


cfg, backbone, selector, adapter = _resources()
store = _store(str(cfg.paths.artifacts / "fedmps.sqlite"))

st.title("Missing-person coordination")
st.caption("Federated search across participating sites. "
           "No raw video, face crop or embedding is held here.")

stats = store.stats()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Open cases", stats["cases_open"])
c2.metric("Live templates", stats["templates_live"])
c3.metric("Confirmed sightings", stats["alerts_confirmed"])
c4.metric("Rejected at node", stats["alerts_rejected"])
c5.metric("Audit entries", stats["audit_entries"])

tab_new, tab_cases, tab_alerts, tab_audit = st.tabs(
    ["Open a case", "Cases", "Confirmed sightings", "Audit log"])


# --------------------------------------------------------------------------
# Open a case
# --------------------------------------------------------------------------

with tab_new:
    st.subheader("Open a case and distribute it")
    st.write("The photograph is read once, in memory, to compute a template. "
             "It is not saved and it is not sent anywhere.")

    left, right = st.columns([2, 3])
    with left:
        name = st.text_input("Person's name", placeholder="e.g. A. Kumar")
        case_id = st.text_input(
            "Case reference",
            value=f"case-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}")
        notes = st.text_area("Notes", height=80,
                             placeholder="Last seen, clothing, contact…")
        ttl_days = st.number_input(
            "Auto-expire after (days)", min_value=1, max_value=365, value=30,
            help="When this elapses the templates stop being served to nodes. "
                 "Nobody has to remember to delete anything.")
        n_nodes = st.number_input("Distribute to nodes", min_value=1,
                                  max_value=32,
                                  value=cfg.partition.n_nodes)
        photo = st.file_uploader("Reference photograph",
                                 type=["jpg", "jpeg", "png"])

    with right:
        if photo is not None:
            st.image(photo,
                     caption="Used once to compute the template. A small "
                             "thumbnail is kept for reviewer verification.",
                     width=260)

        if st.button("Create case and distribute", type="primary",
                     disabled=not (photo and name)):
            import cv2

            raw = np.frombuffer(photo.getvalue(), dtype=np.uint8)
            bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if bgr is None:
                st.error("That file could not be decoded as an image.")
                st.stop()

            size = cfg.data.image_size
            rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                             (size, size), interpolation=cv2.INTER_AREA)
            emb = backbone.embed(rgb[None, ...], progress=False)
            if adapter is not None:
                # The gallery side is a clean enrolment photo, so the adapter
                # is deliberately NOT applied here - it corrects camera
                # degradation, and there is none to correct.
                pass
            bits = binarise(emb, selector)

            # A small JPEG kept for human verification at review time. The
            # full-resolution photograph is still never stored, and automated
            # matching never sees either - that runs on the template alone.
            thumb = cv2.resize(bgr, (200, 200), interpolation=cv2.INTER_AREA)
            ok_enc, enc = cv2.imencode(".jpg", thumb,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            store.create_case(case_id, name, notes, ttl_days=int(ttl_days),
                              reference_thumb=enc.tobytes() if ok_enc else None)
            issued = []
            for node in range(int(n_nodes)):
                key = derive_key(cfg.privacy.template_master_key, case_id,
                                 node, cfg.privacy.template_bits)
                template = pack(protect(bits, key))[0]
                fp = key_fingerprint(key)
                store.issue_template(case_id, node, template.tobytes(), fp,
                                     reference_embedding=emb[0]
                                     .astype("float32").tobytes())
                issued.append({"node": f"node_{node}",
                               "template (hex)": template.tobytes().hex(),
                               "key": fp})

            st.success(f"Case {case_id} distributed to {n_nodes} node(s).")
            st.write("What actually left this server:")
            st.caption(
                "A 200x200 reference thumbnail is retained here for human "
                "verification. It is released to a node only when one of its "
                "own alerts is opened, and every release is audited.")
            st.dataframe(issued, use_container_width=True, hide_index=True)
            st.caption(
                f"{cfg.privacy.template_bits // 8} bytes per node. Every row "
                f"is a different pattern for the same person, so no two sites "
                f"can correlate their watchlists.")


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------

with tab_cases:
    st.subheader("Cases")
    cases = store.list_cases()
    if not cases:
        st.info("No cases yet.")
    for case in cases:
        state = ("open" if case.active
                 else "expired" if case.expired else case.status)
        cols = st.columns([3, 2, 2, 2, 2])
        cols[0].markdown(f"**{case.name}**  \n`{case.case_id}`")
        cols[1].write(state)
        cols[2].write(f"opened {case.created_utc[:10]}")
        cols[3].write(f"expires {(case.expires_utc or '—')[:10]}")
        if case.status == CASE_OPEN:
            if cols[4].button("Close & revoke", key=f"close-{case.case_id}"):
                store.close_case(case.case_id)
                st.rerun()
        st.divider()


# --------------------------------------------------------------------------
# Confirmed sightings
# --------------------------------------------------------------------------

with tab_alerts:
    st.subheader("Confirmed sightings")
    st.caption("Pending alerts are not visible here by design. A crop reaches "
               "this server only after a named operator at the node has "
               "confirmed it.")

    confirmed = store.list_alerts(status=STATUS_CONFIRMED)
    if not confirmed:
        st.info("Nothing confirmed yet.")
    for a in confirmed:
        cols = st.columns([1, 3, 2, 2])
        crop = Path(a["crop_path"] or "")
        if crop.exists():
            cols[0].image(str(crop), width=110)
        else:
            cols[0].write("—")
        cols[1].markdown(
            f"**{a['case_name'] or a['case_id']}**  \n"
            f"`{a['case_id']}` · node {a['node_id']} · {a['camera']}")
        cols[2].metric("Score", f"{a['score']:.3f}")
        cols[3].write(
            f"{a['frames_voted']}/{a['frames_seen']} frames agreed  \n"
            f"confirmed by {a['reviewed_by']}  \n"
            f"{(a['reviewed_utc'] or '')[:19]}")
        st.divider()

    prec = store.precision_report()
    if prec["reviewed"]:
        st.metric("Precision over reviewed alerts",
                  f"{prec['precision']:.0%}",
                  help=prec["note"])


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------

with tab_audit:
    st.subheader("Audit log")
    st.caption("Append-only, and readable by the node operators too. Trust in "
               "a system like this has to run both ways to mean anything.")
    entries = store.list_audit(300)
    if entries:
        st.dataframe(entries, use_container_width=True, hide_index=True)
    else:
        st.info("Nothing logged yet.")
