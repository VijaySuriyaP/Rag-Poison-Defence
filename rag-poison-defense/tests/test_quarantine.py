"""SQLite quarantine state transitions and review queue export."""
from __future__ import annotations

import csv

from ragdef.defenses.quarantine import QuarantineStore


def test_quarantine_lifecycle_and_export(tmp_path):
    reingested = []
    store = QuarantineStore(tmp_path / "review.sqlite", reingest_callback=reingested.append)
    doc = {"doc_id": "d1", "text": "example text"}
    store.put(doc, score=0.8, reason_codes=["override"], reasons=["test rule"])
    assert store.status("d1") == "QUARANTINED"
    assert store.export_csv(tmp_path / "queue.csv") == 1
    with (tmp_path / "queue.csv").open(encoding="utf-8", newline="") as stream:
        assert next(csv.DictReader(stream))["doc_id"] == "d1"
    assert store.release("d1") and store.status("d1") == "RELEASED"
    assert reingested[0]["doc_id"] == "d1"
    store.put({"doc_id": "d2", "text": "x"})
    assert store.reject("d2") and store.status("d2") == "REJECTED"
    assert QuarantineStore.simulated_review(True, accuracy=1.0, seed=3) == "REJECTED"
    store.close()
