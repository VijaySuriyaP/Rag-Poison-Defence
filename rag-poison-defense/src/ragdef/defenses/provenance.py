"""HMAC-signed ingestion manifests, source allowlists, and trust-tier assignment."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse

from ragdef.config import load_config
from ragdef.utils import sha256_text


@dataclass
class ProvenanceResult:
    """Machine-readable provenance verification and trust classification."""
    valid: bool
    trust_tier: str
    flags: list[str]
    reasons: list[str]
    content_hash: str
    signature: str


def _manifest_fields(document: dict[str, Any], content_hash: str) -> dict[str, str]:
    """Return stable HMAC-covered fields from a document."""
    return {"doc_id": str(document.get("doc_id", "")), "content_hash": content_hash,
            "source": str(document.get("source", "")), "author": str(document.get("author", "")),
            "ts": str(document.get("timestamp", document.get("ingest_ts", document.get("ts", ""))))}


def sign_document(document: dict[str, Any], key: str) -> dict[str, Any]:
    """Attach content hash and HMAC-SHA256 signature to a document copy."""
    result = dict(document)
    content_hash = sha256_text(str(result.get("text", "")))
    fields = _manifest_fields(result, content_hash)
    payload = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
    result["content_hash"] = content_hash
    result["signature"] = hmac.new(key.encode(), payload, hashlib.sha256).hexdigest()
    result["signature_valid"] = True
    return result


class ProvenanceVerifier:
    """Verify content hashes, HMAC signatures, source claims, timestamps, and replays."""
    def __init__(self, key: str | None = None, allowed_domains: set[str] | None = None,
                 future_skew_seconds: int = 300) -> None:
        config = load_config().get("provenance", {})
        env_name = str(config.get("hmac_env", "RAGDEF_HMAC_KEY"))
        self.key = key or os.getenv(env_name) or str(config.get("default_key", "local-demo-key"))
        self.allowed_domains = {d.lower() for d in (allowed_domains or set(config.get("allowed_domains", [])))}
        self.future_skew = future_skew_seconds
        self.seen_manifests: set[str] = set()

    def verify(self, document: dict[str, Any]) -> ProvenanceResult:
        """Verify a document and assign T0–T3 based on verified origin evidence."""
        text_hash = sha256_text(str(document.get("text", "")))
        stored_hash = str(document.get("content_hash", ""))
        signature = str(document.get("signature", ""))
        flags: list[str] = []
        reasons: list[str] = []
        valid = False
        if stored_hash and not hmac.compare_digest(stored_hash, text_hash):
            flags.append("content_hash_mismatch"); reasons.append("document text does not match the stored content hash")
        elif signature and stored_hash:
            fields = _manifest_fields(document, stored_hash)
            payload = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
            expected = hmac.new(self.key.encode(), payload, hashlib.sha256).hexdigest()
            valid = hmac.compare_digest(expected, signature)
            if not valid:
                flags.append("invalid_signature"); reasons.append("HMAC signature does not match manifest fields")
        elif signature or stored_hash:
            flags.append("incomplete_manifest"); reasons.append("signature and content hash must both be present")

        manifest_id = sha256_text(f"{document.get('doc_id', '')}:{stored_hash}:{signature}")
        if valid and manifest_id in self.seen_manifests:
            flags.append("replayed_manifest"); reasons.append("the same signed manifest was already observed")
            valid = False
        elif valid:
            self.seen_manifests.add(manifest_id)

        timestamp = str(document.get("timestamp", document.get("ingest_ts", document.get("ts", ""))))
        timestamp_ok = True
        if timestamp:
            try:
                ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                now = datetime.now(timezone.utc)
                if ts > now + timedelta(seconds=self.future_skew):
                    timestamp_ok = False
                    flags.append("future_timestamp"); reasons.append("manifest timestamp is implausibly far in the future")
            except ValueError:
                timestamp_ok = False
                flags.append("invalid_timestamp"); reasons.append("manifest timestamp could not be parsed")
        claimed = str(document.get("claimed_source", "")).lower().strip()
        url = str(document.get("url", ""))
        host = (urlparse(url).hostname or "").lower()
        source = str(document.get("source", "")).lower().strip()
        if claimed and host and claimed not in {host, f"https://{host}", f"http://{host}"} and claimed not in source:
            flags.append("claimed_source_mismatch"); reasons.append("claimed source does not match recorded URL/source")
        elif claimed and source and claimed not in source and not host:
            flags.append("claimed_source_mismatch"); reasons.append("claimed source differs from recorded source")

        explicitly_uploaded = any(word in source for word in ("upload", "anonymous", "user", "attack")) or not source
        allowlisted = any(host == domain or host.endswith("." + domain) for domain in self.allowed_domains)
        if valid and timestamp_ok:
            tier = "T0"
        elif explicitly_uploaded or (host and host.endswith(".invalid")):
            tier = "T3"
        elif allowlisted:
            tier = "T1"
        else:
            tier = "T2"
        if not reasons:
            reasons.append("signed manifest verified" if tier == "T0" else f"source assigned to trust tier {tier}")
        return ProvenanceResult(valid and timestamp_ok, tier, list(dict.fromkeys(flags)),
                                list(dict.fromkeys(reasons)), text_hash, signature)

    def annotate(self, document: dict[str, Any]) -> tuple[dict[str, Any], ProvenanceResult]:
        """Return a copied document annotated with trust tier and provenance signals."""
        result = dict(document)
        proof = self.verify(result)
        result.update({"trust_tier": proof.trust_tier, "provenance_valid": proof.valid,
                       "provenance_flags": proof.flags, "content_hash": proof.content_hash})
        return result, proof
