"""SQLite quarantine queue with auditable review decisions and export."""
from __future__ import annotations

import json
import random
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ragdef.utils import write_csv

STATES = {"ALLOWED", "QUARANTINED", "RELEASED", "REJECTED"}


class QuarantineStore:
    """Persist quarantine state, reasons, review decisions, and original payloads."""
    def __init__(self, path: str | Path = "results/quarantine.sqlite",
                 reingest_callback: Callable[[dict[str, Any]], None] | None = None) -> None:
        location = str(path)
        if location != ":memory:":
            Path(location).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(location, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.reingest_callback = reingest_callback
        self.connection.execute("""CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY, status TEXT NOT NULL, score REAL NOT NULL,
            reason_codes TEXT NOT NULL, reasons TEXT NOT NULL, created_at TEXT NOT NULL,
            reviewer_decision TEXT, reviewed_at TEXT, payload TEXT NOT NULL)""")
        self.connection.commit()

    def put(self, document: dict[str, Any], status: str = "QUARANTINED", score: float = 0.0,
            reason_codes: list[str] | None = None, reasons: list[str] | None = None) -> None:
        """Insert/update a document's triage record, retaining the full input payload."""
        if status not in STATES:
            raise ValueError(f"Invalid quarantine state: {status}")
        doc_id = str(document.get("doc_id", ""))
        if not doc_id:
            raise ValueError("Quarantine records require doc_id")
        now = datetime.now(timezone.utc).isoformat()
        self.connection.execute("""INSERT INTO documents
            (doc_id,status,score,reason_codes,reasons,created_at,reviewer_decision,reviewed_at,payload)
            VALUES (?,?,?,?,?,?,NULL,NULL,?)
            ON CONFLICT(doc_id) DO UPDATE SET status=excluded.status, score=excluded.score,
            reason_codes=excluded.reason_codes, reasons=excluded.reasons, payload=excluded.payload""",
            (doc_id, status, float(score), json.dumps(reason_codes or []), json.dumps(reasons or []), now,
             json.dumps(document, ensure_ascii=False, sort_keys=True)))
        self.connection.commit()

    def status(self, doc_id: str) -> str | None:
        """Return the current state of one document, or None if untracked."""
        row = self.connection.execute("SELECT status FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        return str(row["status"]) if row else None

    def get(self, doc_id: str) -> dict[str, Any] | None:
        """Return a full review record, decoded to ordinary Python values."""
        row = self.connection.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        result["reason_codes"] = json.loads(result["reason_codes"])
        result["reasons"] = json.loads(result["reasons"])
        result["payload"] = json.loads(result["payload"])
        return result

    def release(self, doc_id: str) -> bool:
        """Mark a document released and optionally invoke the caller's re-ingestion hook."""
        record = self.get(doc_id)
        if not record:
            return False
        now = datetime.now(timezone.utc).isoformat()
        self.connection.execute("UPDATE documents SET status='RELEASED', reviewer_decision='release', reviewed_at=? WHERE doc_id=?",
                                (now, doc_id))
        self.connection.commit()
        if self.reingest_callback:
            self.reingest_callback(record["payload"])
        return True

    def reject(self, doc_id: str) -> bool:
        """Mark a document rejected by a reviewer."""
        if not self.get(doc_id):
            return False
        now = datetime.now(timezone.utc).isoformat()
        self.connection.execute("UPDATE documents SET status='REJECTED', reviewer_decision='reject', reviewed_at=? WHERE doc_id=?",
                                (now, doc_id))
        self.connection.commit()
        return True

    def review_queue(self) -> list[dict[str, Any]]:
        """List items still waiting for human review."""
        rows = self.connection.execute("SELECT * FROM documents WHERE status='QUARANTINED' ORDER BY created_at").fetchall()
        output = []
        for row in rows:
            item = dict(row)
            item["reason_codes"] = json.loads(item["reason_codes"])
            item["reasons"] = json.loads(item["reasons"])
            item["payload"] = json.loads(item["payload"])
            output.append(item)
        return output

    def export_csv(self, path: str | Path = "results/review_queue.csv") -> int:
        """Export current quarantine queue to a portable CSV for analyst review."""
        queue = self.review_queue()
        rows = [{"doc_id": r["doc_id"], "status": r["status"], "score": r["score"],
                 "reason_codes": ";".join(r["reason_codes"]), "reasons": ";".join(r["reasons"]),
                 "created_at": r["created_at"]} for r in queue]
        write_csv(path, rows, ["doc_id", "status", "score", "reason_codes", "reasons", "created_at"])
        return len(rows)

    @staticmethod
    def simulated_review(actual_malicious: bool, accuracy: float = 0.92, seed: int = 42) -> str:
        """Model a reviewer decision with configured accuracy and deterministic seed."""
        rng = random.Random(seed)
        decision_malicious = actual_malicious if rng.random() < min(1.0, max(0.0, accuracy)) else not actual_malicious
        return "REJECTED" if decision_malicious else "RELEASED"

    def close(self) -> None:
        """Close the SQLite connection."""
        self.connection.close()
