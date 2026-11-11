"""HMAC, trust tier, spoofing, timestamp, and replay checks."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ragdef.defenses.provenance import ProvenanceVerifier, sign_document


def test_signature_hash_and_replay_checks():
    key = "test-secret"
    base = {"doc_id": "signed-1", "text": "trusted internal notes", "source": "internal",
            "author": "research", "timestamp": datetime.now(timezone.utc).isoformat()}
    signed = sign_document(base, key)
    verifier = ProvenanceVerifier(key=key)
    result = verifier.verify(signed)
    assert result.valid and result.trust_tier == "T0"
    replay = verifier.verify(signed)
    assert not replay.valid and "replayed_manifest" in replay.flags
    tampered = dict(signed, text="changed body")
    assert "content_hash_mismatch" in ProvenanceVerifier(key=key).verify(tampered).flags


def test_spoofed_source_and_future_timestamp_are_flagged():
    future = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
    doc = {"doc_id": "spoof", "text": "hello", "source": "user-upload", "url": "https://attacker.invalid/x",
           "claimed_source": "trusted.internal", "timestamp": future}
    result = ProvenanceVerifier(key="k").verify(doc)
    assert result.trust_tier == "T3"
    assert "claimed_source_mismatch" in result.flags and "future_timestamp" in result.flags
