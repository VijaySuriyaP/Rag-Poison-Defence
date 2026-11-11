"""Composable defended RAG pipeline wiring D1–D6 into ingestion and generation."""
from __future__ import annotations

import time
from typing import Any

from ragdef.config import load_config
from ragdef.defenses.ml_scanner import MLScanner
from ragdef.defenses.normalize import normalize_content
from ragdef.defenses.output_guard import OutputGuard
from ragdef.defenses.provenance import ProvenanceVerifier
from ragdef.defenses.prompt_hardening import harden_context
from ragdef.defenses.quarantine import QuarantineStore
from ragdef.defenses.rule_scanner import RuleScanner
from ragdef.defenses.types import Verdict
from ragdef.rag.pipeline import RAGPipeline

ALL_DEFENSES = {"D1", "D2", "D3", "D4", "D5", "D6"}


class DefendedPipeline:
    """Apply configurable scanners/provenance/quarantine/hardening/output guard."""
    def __init__(self, enabled: set[str] | None = None, config: dict[str, Any] | None = None,
                 victim: Any | None = None, quarantine_store: QuarantineStore | None = None) -> None:
        self.config = config or load_config()
        self.enabled = set(enabled or set())
        invalid = self.enabled - ALL_DEFENSES
        if invalid:
            raise ValueError(f"Unknown defenses: {sorted(invalid)}")
        scanner_cfg = self.config.get("scanner", {})
        provenance_cfg = self.config.get("provenance", {})
        self.rule_scanner = RuleScanner(float(scanner_cfg.get("t_review", 0.35)),
                                        float(scanner_cfg.get("t_block", 0.75)),
                                        set(provenance_cfg.get("allowed_domains", [])))
        self.provenance = ProvenanceVerifier(allowed_domains=set(provenance_cfg.get("allowed_domains", [])))
        self.ml_scanner: MLScanner | None = None
        if "D2" in self.enabled:
            self.ml_scanner = MLScanner(float(scanner_cfg.get("t_review", 0.35)),
                                        float(scanner_cfg.get("t_block", 0.75)))
            self.ml_scanner.fit_from_files("models/prompt_injection_scanner.joblib")
        self.quarantine = quarantine_store or QuarantineStore(":memory:")
        self.output_guard = OutputGuard(set(provenance_cfg.get("allowed_domains", [])))
        self.base = RAGPipeline(self.config, victim=victim)
        self.stats: dict[str, float] = {"documents_seen": 0, "documents_indexed": 0,
                                        "quarantined": 0, "blocked": 0, "scan_ms": 0.0}
        self.scan_log: list[dict[str, Any]] = []

    def _tier_block_threshold(self, tier: str) -> float:
        """Look up risk threshold for a trust tier (lower is stricter)."""
        return float(self.config.get("scanner", {}).get(f"threshold_{tier.lower()}", 0.75))

    def scan_document(self, input_document: dict[str, Any]) -> tuple[dict[str, Any], Verdict, str]:
        """Normalize, assign provenance tier, and combine enabled scanner verdicts."""
        document = dict(input_document)
        if not document.get("text") and document.get("full_text"):
            document["text"] = document["full_text"]
        if self.enabled.intersection({"D1", "D2"}):
            normalized = normalize_content(str(document.get("text", "")))
            document["text"] = normalized.text
            document["normalization_signals"] = normalized.signals
            document["normalization_removed_count"] = normalized.removed_count
        else:
            normalized = type("NoopNormalization", (), {"signals": [], "removed_count": 0})()
        proof = self.provenance.verify(document)
        tier = proof.trust_tier
        if "D3" in self.enabled:
            document["trust_tier"] = tier
            document["provenance_valid"] = proof.valid
            document["provenance_flags"] = proof.flags
        elif not document.get("trust_tier"):
            document["trust_tier"] = tier

        scores: list[float] = []
        flags = list(normalized.signals) + list(proof.flags if "D3" in self.enabled else [])
        reasons = []
        if "D1" in self.enabled:
            verdict = self.rule_scanner.scan(document)
            scores.append(verdict.score); flags.extend(verdict.flags); reasons.extend(verdict.reasons)
        if "D2" in self.enabled and self.ml_scanner is not None:
            verdict = self.ml_scanner.scan(document)
            scores.append(verdict.score); flags.extend(verdict.flags); reasons.extend(verdict.reasons)
        combined = 0.0
        if scores:
            remaining = 1.0
            for value in scores:
                remaining *= 1.0 - value
            combined = 1.0 - remaining
        if "D3" in self.enabled and proof.flags:
            combined = 1.0 - (1.0 - combined) * 0.8
            reasons.extend(proof.reasons)
        scanner_cfg = self.config.get("scanner", {})
        review_threshold = float(scanner_cfg.get("t_review", 0.35))
        block_threshold = (self._tier_block_threshold(tier) if "D3" in self.enabled
                           else float(scanner_cfg.get("t_block", 0.75)))
        if combined >= block_threshold:
            action = "block"
        elif combined >= review_threshold:
            action = "quarantine"
        else:
            action = "allow"
        return document, Verdict(combined, flags, reasons, action), tier

    def ingest(self, documents: list[dict[str, Any]],
               prechecked: dict[str, tuple[dict[str, Any], Verdict, str]] | None = None) -> int:
        """Filter or quarantine documents, optionally reusing same-config scan results.

        `prechecked` is an experiment optimization for clean documents scanned once per
        defense configuration and threshold. Signed/provenance replay checks should not
        use this cache; callers must only reuse results for the same exact document/config.
        """
        start = time.perf_counter()
        allowed: list[dict[str, Any]] = []
        self.stats = {"documents_seen": len(documents), "documents_indexed": 0,
                      "quarantined": 0, "blocked": 0, "scan_ms": 0.0}
        if self.ml_scanner is not None:
            # Warm the per-document score cache in one vectorizer call. scan_document
            # then reuses these exact normalized-text scores during policy evaluation.
            batch_texts = []
            for original in documents:
                key = str(original.get("doc_id", "unknown"))
                if prechecked is not None and key in prechecked:
                    continue
                raw = str(original.get("text", original.get("full_text", "")))
                batch_texts.append(normalize_content(raw).text)
            if batch_texts:
                self.ml_scanner.score_many(batch_texts)
        for original in documents:
            doc_key = str(original.get("doc_id", "unknown"))
            cached = (prechecked or {}).get(doc_key)
            if cached is not None:
                cached_doc, cached_verdict, tier = cached
                doc = dict(cached_doc)
                verdict = Verdict(cached_verdict.score, cached_verdict.flags,
                                  cached_verdict.reasons, cached_verdict.action)
            else:
                doc, verdict, tier = self.scan_document(original)
            doc_id = str(doc.get("doc_id", "unknown"))
            # D4-only is deliberately conservative: all anonymous/untrusted T3 content is queued.
            if self.enabled == {"D4"} and tier == "T3":
                self.quarantine.put(doc, "QUARANTINED", 1.0, ["untrusted_t3"], ["D4 policy quarantines all T3 documents"])
                self.stats["quarantined"] += 1
                self.scan_log.append({"doc_id": doc_id, "tier": tier, "score": 1.0, "action": "quarantine"})
                continue
            if verdict.action in {"block", "quarantine"} and self.enabled.intersection({"D1", "D2", "D3"}):
                if verdict.action == "quarantine" and "D4" in self.enabled:
                    self.quarantine.put(doc, "QUARANTINED", verdict.score, verdict.flags, verdict.reasons)
                    self.stats["quarantined"] += 1
                else:
                    self.stats["blocked"] += 1
                self.scan_log.append({"doc_id": doc_id, "tier": tier, **verdict.as_dict()})
                continue
            if "D4" in self.enabled and tier == "T3" and verdict.action != "allow":
                self.quarantine.put(doc, "QUARANTINED", verdict.score, verdict.flags, verdict.reasons)
                self.stats["quarantined"] += 1
                continue
            allowed.append(doc)
        self.stats["scan_ms"] = (time.perf_counter() - start) * 1000
        self.stats["documents_indexed"] = self.base.ingest(allowed)
        return int(self.stats["documents_indexed"])

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        """Retrieve context, applying T3 filtering when provenance defense is enabled."""
        allowed_tiers = {"T0", "T1", "T2"} if "D3" in self.enabled else None
        return self.base.retrieve(question, allowed_tiers)

    def ask(self, question: str) -> str:
        """Generate answer with optional context hardening and output redaction."""
        context = self.retrieve(question)
        if "D5" in self.enabled:
            context = harden_context(context)
        answer = self.base.victim.generate(question, context)
        if "D6" in self.enabled:
            return self.output_guard.guard(answer, "query").text
        return answer
