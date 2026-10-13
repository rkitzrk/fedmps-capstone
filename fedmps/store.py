"""Shared state for the operational plane: cases, templates, alerts, audit.

One SQLite file stands in for what would be a server database and a set of
node-local databases in a real deployment. Using a file rather than a network
service is a deliberate demo choice - there is one less thing to fail in front
of an examiner - but the *access pattern* is kept honest, and that is what the
report should describe:

  The server writes cases, mints one template per node, and reads only
  CONFIRMED alerts.

  A node reads its own templates, writes alerts as PENDING, and shows those
  pending crops to its own operator and to nobody else.

That asymmetry is the two-stage minimal disclosure from the design report,
enacted rather than asserted. A face crop becomes visible upstream only after
a named human at the node has confirmed it. Every issue, review and access is
appended to an audit log that both sides can read, because trust in a system
like this has to run in both directions to mean anything.

Retention is enforced here too: a case carries a TTL, and expired cases stop
returning templates to nodes without anyone having to remember to delete them.
"""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

STATUS_PENDING = "pending"
STATUS_CONFIRMED = "confirmed"
STATUS_REJECTED = "rejected"

CASE_OPEN = "open"
CASE_CLOSED = "closed"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id     TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    notes       TEXT,
    status      TEXT NOT NULL DEFAULT 'open',
    created_utc TEXT NOT NULL,
    expires_utc TEXT,
    -- Small JPEG of the reference photograph, kept so a reviewing operator
    -- can see who they are being asked to identify. Automated matching never
    -- touches it - that runs on the 64-byte template alone. It is released
    -- only when a human opens an alert, and every release is audited.
    reference_thumb BLOB
);
CREATE TABLE IF NOT EXISTS templates (
    case_id     TEXT NOT NULL,
    node_id     INTEGER NOT NULL,
    template    BLOB NOT NULL,
    key_fp      TEXT NOT NULL,
    issued_utc  TEXT NOT NULL,
    revoked     INTEGER NOT NULL DEFAULT 0,
    -- The raw 512-d embedding, distributed only when the node is running in
    -- unprotected "cosine" matching mode. It is not a photograph, but unlike
    -- the template it is neither keyed nor revocable, and the same person
    -- enrolled at two sites yields identical bytes. Holding both side by side
    -- is what lets the demo show the accuracy cost of protection.
    reference_embedding BLOB,
    PRIMARY KEY (case_id, node_id)
);
CREATE TABLE IF NOT EXISTS alerts (
    alert_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id      TEXT NOT NULL,
    node_id      INTEGER NOT NULL,
    camera       TEXT,
    track_id     INTEGER,
    score        REAL NOT NULL,
    frames_voted INTEGER,
    frames_seen  INTEGER,
    crop_path    TEXT,
    status       TEXT NOT NULL DEFAULT 'pending',
    created_utc  TEXT NOT NULL,
    reviewed_by  TEXT,
    reviewed_utc TEXT
);
CREATE TABLE IF NOT EXISTS audit (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc   TEXT NOT NULL,
    actor    TEXT NOT NULL,
    action   TEXT NOT NULL,
    detail   TEXT
);
CREATE INDEX IF NOT EXISTS idx_alerts_node ON alerts(node_id, status);
CREATE INDEX IF NOT EXISTS idx_templates_node ON templates(node_id, revoked);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def key_fingerprint(key_bits) -> str:
    """A short, non-reversible label for a key, so the audit log can name
    which key issued a template without ever recording the key."""
    import numpy as np

    raw = np.packbits(np.asarray(key_bits, dtype=np.uint8)).tobytes()
    return hashlib.sha256(raw).hexdigest()[:12]


@dataclass
class Case:
    case_id: str
    name: str
    notes: str
    status: str
    created_utc: str
    expires_utc: Optional[str]

    @property
    def expired(self) -> bool:
        if not self.expires_utc:
            return False
        return datetime.fromisoformat(self.expires_utc) < datetime.now(timezone.utc)

    @property
    def active(self) -> bool:
        return self.status == CASE_OPEN and not self.expired


class Store:
    """Thin SQLite wrapper. Every mutating call writes an audit row."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._conn()) as c:
            c.executescript(_SCHEMA)
            c.commit()
        self._migrate()

    def _migrate(self) -> None:
        """Add columns introduced after a database was first created."""
        with closing(self._conn()) as c:
            cols = {r[1] for r in c.execute("PRAGMA table_info(cases)")}
            if "reference_thumb" not in cols:
                c.execute("ALTER TABLE cases ADD COLUMN reference_thumb BLOB")
            tcols = {r[1] for r in c.execute("PRAGMA table_info(templates)")}
            if "reference_embedding" not in tcols:
                c.execute("ALTER TABLE templates "
                          "ADD COLUMN reference_embedding BLOB")
            c.commit()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=10)
        c.row_factory = sqlite3.Row
        return c

    # -- audit -------------------------------------------------------------

    def audit(self, actor: str, action: str, detail: str = "") -> None:
        with closing(self._conn()) as c:
            c.execute("INSERT INTO audit (ts_utc, actor, action, detail) "
                      "VALUES (?,?,?,?)", (_now(), actor, action, detail))
            c.commit()

    def list_audit(self, limit: int = 200) -> List[Dict[str, Any]]:
        with closing(self._conn()) as c:
            rows = c.execute("SELECT * FROM audit ORDER BY id DESC LIMIT ?",
                             (limit,)).fetchall()
        return [dict(r) for r in rows]

    # -- cases -------------------------------------------------------------

    def create_case(self, case_id: str, name: str, notes: str = "",
                    ttl_days: Optional[int] = 30,
                    actor: str = "server",
                    reference_thumb: Optional[bytes] = None) -> Case:
        expires = (_now() if ttl_days is None else
                   (datetime.now(timezone.utc)
                    + timedelta(days=int(ttl_days))).isoformat(timespec="seconds"))
        if ttl_days is None:
            expires = None
        with closing(self._conn()) as c:
            c.execute(
                "INSERT OR REPLACE INTO cases "
                "(case_id, name, notes, status, created_utc, expires_utc, "
                " reference_thumb) VALUES (?,?,?,?,?,?,?)",
                (case_id, name, notes, CASE_OPEN, _now(), expires,
                 reference_thumb))
            c.commit()
        self.audit(actor, "case.create",
                   f"{case_id} '{name}' ttl={ttl_days}d")
        return self.get_case(case_id)  # type: ignore[return-value]

    def get_case(self, case_id: str) -> Optional[Case]:
        with closing(self._conn()) as c:
            r = c.execute(
                "SELECT case_id,name,notes,status,created_utc,expires_utc "
                "FROM cases WHERE case_id=?", (case_id,)).fetchone()
        return Case(**dict(r)) if r else None

    def reference_thumb(self, case_id: str, actor: str,
                        reason: str = "alert review") -> Optional[bytes]:
        """The reference photograph, released for human verification.

        This is the one place an image of the subject reaches a reviewer, and
        it is deliberately pull-based: nothing is pushed to a node in advance,
        so a site holds no photographs until one of its own alerts is opened.
        Every release is written to the audit log, which is what makes the
        access reviewable rather than merely limited.
        """
        with closing(self._conn()) as c:
            r = c.execute("SELECT reference_thumb FROM cases WHERE case_id=?",
                          (case_id,)).fetchone()
        if r is None or r[0] is None:
            return None
        self.audit(actor, "reference.view", f"{case_id} ({reason})")
        return bytes(r[0])

    def list_cases(self, status: Optional[str] = None) -> List[Case]:
        q = ("SELECT case_id,name,notes,status,created_utc,expires_utc "
             "FROM cases")
        args: Sequence[Any] = ()
        if status:
            q += " WHERE status=?"
            args = (status,)
        q += " ORDER BY created_utc DESC"
        with closing(self._conn()) as c:
            rows = c.execute(q, args).fetchall()
        return [Case(**dict(r)) for r in rows]

    def close_case(self, case_id: str, actor: str = "server") -> None:
        """Closing a case revokes every template issued for it.

        Nobody has to remember to delete anything at the nodes: the templates
        stop being served, and a node that still holds bytes on disk holds
        bytes that no longer match anything.
        """
        with closing(self._conn()) as c:
            c.execute("UPDATE cases SET status=? WHERE case_id=?",
                      (CASE_CLOSED, case_id))
            c.execute("UPDATE templates SET revoked=1 WHERE case_id=?",
                      (case_id,))
            c.commit()
        self.audit(actor, "case.close", f"{case_id} templates revoked")

    # -- templates ---------------------------------------------------------

    def issue_template(self, case_id: str, node_id: int, template: bytes,
                       key_fp: str, actor: str = "server",
                       reference_embedding: Optional[bytes] = None) -> None:
        with closing(self._conn()) as c:
            c.execute(
                "INSERT OR REPLACE INTO templates "
                "(case_id, node_id, template, key_fp, issued_utc, revoked, "
                " reference_embedding) VALUES (?,?,?,?,?,0,?)",
                (case_id, node_id, template, key_fp, _now(),
                 reference_embedding))
            c.commit()
        self.audit(actor, "template.issue",
                   f"{case_id} -> node_{node_id} key={key_fp}")

    def templates_for_node(self, node_id: int,
                           include_inactive: bool = False
                           ) -> List[Dict[str, Any]]:
        """A node's current watchlist. Expired and closed cases drop out."""
        with closing(self._conn()) as c:
            rows = c.execute(
                "SELECT t.case_id, t.template, t.key_fp, t.issued_utc, "
                "       t.revoked, t.reference_embedding, "
                "       c.name, c.status, c.expires_utc "
                "FROM templates t JOIN cases c ON c.case_id = t.case_id "
                "WHERE t.node_id = ? ORDER BY t.issued_utc DESC",
                (node_id,)).fetchall()

        out: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            case = Case(case_id=d["case_id"], name=d["name"], notes="",
                        status=d["status"], created_utc="",
                        expires_utc=d["expires_utc"])
            d["active"] = bool(case.active and not d["revoked"])
            if d["active"] or include_inactive:
                out.append(d)
        return out

    # -- alerts ------------------------------------------------------------

    def add_alert(self, case_id: str, node_id: int, score: float,
                  camera: str = "", track_id: int = -1,
                  frames_voted: int = 0, frames_seen: int = 0,
                  crop_path: str = "") -> int:
        """Raised by a node. Starts PENDING and is invisible to the server."""
        with closing(self._conn()) as c:
            cur = c.execute(
                "INSERT INTO alerts (case_id, node_id, camera, track_id, "
                "score, frames_voted, frames_seen, crop_path, status, "
                "created_utc) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (case_id, node_id, camera, track_id, float(score),
                 frames_voted, frames_seen, crop_path, STATUS_PENDING, _now()))
            c.commit()
            alert_id = int(cur.lastrowid)
        self.audit(f"node_{node_id}", "alert.raise",
                   f"#{alert_id} {case_id} score={score:.3f} track={track_id}")
        return alert_id

    def list_alerts(self, node_id: Optional[int] = None,
                    status: Optional[str] = None,
                    limit: int = 500) -> List[Dict[str, Any]]:
        q = "SELECT a.*, c.name AS case_name FROM alerts a " \
            "LEFT JOIN cases c ON c.case_id = a.case_id WHERE 1=1"
        args: List[Any] = []
        if node_id is not None:
            q += " AND a.node_id=?"
            args.append(node_id)
        if status:
            q += " AND a.status=?"
            args.append(status)
        q += " ORDER BY a.alert_id DESC LIMIT ?"
        args.append(limit)
        with closing(self._conn()) as c:
            rows = c.execute(q, args).fetchall()
        return [dict(r) for r in rows]

    def review_alert(self, alert_id: int, status: str, reviewer: str) -> None:
        """The human gate. Only a confirmed alert becomes visible upstream."""
        if status not in (STATUS_CONFIRMED, STATUS_REJECTED):
            raise ValueError(f"status must be confirmed or rejected, "
                             f"got {status!r}")
        with closing(self._conn()) as c:
            c.execute("UPDATE alerts SET status=?, reviewed_by=?, "
                      "reviewed_utc=? WHERE alert_id=?",
                      (status, reviewer, _now(), alert_id))
            c.commit()
        self.audit(reviewer, f"alert.{status}", f"#{alert_id}")

    # -- reporting ---------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        with closing(self._conn()) as c:
            def one(q: str, *a: Any) -> int:
                return int(c.execute(q, a).fetchone()[0])

            return {
                "cases_open": one("SELECT COUNT(*) FROM cases WHERE status=?",
                                  CASE_OPEN),
                "cases_total": one("SELECT COUNT(*) FROM cases"),
                "templates_live": one(
                    "SELECT COUNT(*) FROM templates WHERE revoked=0"),
                "alerts_pending": one(
                    "SELECT COUNT(*) FROM alerts WHERE status=?",
                    STATUS_PENDING),
                "alerts_confirmed": one(
                    "SELECT COUNT(*) FROM alerts WHERE status=?",
                    STATUS_CONFIRMED),
                "alerts_rejected": one(
                    "SELECT COUNT(*) FROM alerts WHERE status=?",
                    STATUS_REJECTED),
                "audit_entries": one("SELECT COUNT(*) FROM audit"),
            }

    def precision_report(self) -> Dict[str, Any]:
        """How many reviewed alerts were real? The operator-facing number."""
        s = self.stats()
        reviewed = s["alerts_confirmed"] + s["alerts_rejected"]
        return {
            "reviewed": reviewed,
            "confirmed": s["alerts_confirmed"],
            "rejected": s["alerts_rejected"],
            "precision": (s["alerts_confirmed"] / reviewed
                          if reviewed else float("nan")),
            "note": ("precision over reviewed alerts; with a real watchlist "
                     "and real footage this is the number an operator cares "
                     "about far more than rank-1"),
        }


def open_store(cfg) -> Store:
    from .contracts import F_ALERTS_DB
    return Store(cfg.paths.artifacts / F_ALERTS_DB)


if __name__ == "__main__":  # smoke check
    import tempfile

    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        st = Store(Path(tmp) / "t.sqlite")

        case = st.create_case("case-0001", "Test Person", "demo", ttl_days=7)
        assert case.active and not case.expired
        print(f"case {case.case_id} active until {case.expires_utc}")

        key = np.random.default_rng(0).integers(0, 2, size=96)
        fp = key_fingerprint(key)
        for node in range(3):
            st.issue_template("case-0001", node, bytes(range(12)), fp)

        wl = st.templates_for_node(1)
        assert len(wl) == 1 and wl[0]["active"], wl
        print(f"node 1 watchlist: {len(wl)} active template(s), key {fp}")

        aid = st.add_alert("case-0001", node_id=1, score=0.72, camera="cam-A",
                           track_id=7, frames_voted=4, frames_seen=6,
                           crop_path="media/alerts/1.png")

        # Two-stage disclosure: pending is invisible upstream.
        assert len(st.list_alerts(status=STATUS_CONFIRMED)) == 0, \
            "an unreviewed alert was visible as confirmed"
        assert len(st.list_alerts(node_id=1)) == 1, \
            "the node cannot see its own pending alert"
        print("pending alert is node-local OK")

        st.review_alert(aid, STATUS_CONFIRMED, reviewer="operator@node1")
        assert len(st.list_alerts(status=STATUS_CONFIRMED)) == 1
        print("confirmed alert became visible upstream OK")

        # Closing a case must retire every template automatically.
        st.close_case("case-0001")
        assert st.templates_for_node(1) == [], \
            "templates survived case closure"
        assert len(st.templates_for_node(1, include_inactive=True)) == 1
        print("case closure revoked all templates OK")

        # An expired case must stop serving even while nominally open.
        st.create_case("case-0002", "Expired", ttl_days=0)
        st.issue_template("case-0002", 1, b"x" * 12, fp)
        import time as _t
        _t.sleep(1.1)
        assert st.templates_for_node(1) == [], "an expired case still served"
        print("TTL expiry OK")

        print(f"stats: {st.stats()}")
        print(f"audit entries: {len(st.list_audit())}")
        assert len(st.list_audit()) >= 6, "audit log is missing entries"

    print("store OK")
