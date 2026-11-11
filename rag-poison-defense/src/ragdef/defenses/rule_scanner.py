"""Explainable weighted-rule scanner for indirect prompt injection."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any
from urllib.parse import urlparse

from ragdef.defenses.normalize import normalize_content
from ragdef.defenses.types import Verdict

RULES: tuple[tuple[str, str, float, str], ...] = (
    ("direct_override", r"\b(ignore|disregard|override|forget)\s+(all\s+)?(the\s+)?(previous|prior|above|normal)\s+(instructions?|rules?|content)", 0.65, "override wording targets instruction hierarchy"),
    ("answer_override", r"\b(output|print|return|say|emit)\s+(only\s+)?(?:the\s+)?(?:final\s+)?(?:answer|marker|token|canary|response)\b", 0.50, "imperative requests a fixed model output"),
    ("role_impersonation", r"(?:^|\n)\s*(system|assistant|developer)\s*:\s*|<\|(?:system|assistant|developer)\|>", 0.45, "content imitates a privileged conversation role"),
    ("fake_tool_call", r"function_call|tool_call|mock_echo|\{\s*['\"]tool['\"]\s*:", 0.42, "content resembles a tool/function invocation"),
    ("multilingual_instruction", r"(?:निर्देशों?\s+को\s+अनदेखा|उत्तर\s+में|മുൻ\s+നിർദ്ദേശങ്ങൾ|അവഗണിച്ച്|മറുപടിയായി)", 0.60, "instruction-like Hindi/Malayalam phrase detected"),
    ("metadata_spoof", r"(?:signature\s*=\s*valid|signed\s*=\s*true|trusted_source\s*=\s*true|origin\s*=\s*internal)", 0.58, "document self-asserts trusted provenance"),
    ("canary_marker", r"CANARY-[A-F0-9]{8}", 0.35, "evaluation canary marker appears in document"),
)


class RuleScanner:
    """Return a bounded risk score and reasons for rule-based content analysis."""
    def __init__(self, t_review: float = 0.35, t_block: float = 0.75,
                 allowed_domains: set[str] | None = None) -> None:
        self.t_review = t_review
        self.t_block = t_block
        self.allowed_domains = allowed_domains or {"example.org", "wikipedia.org"}

    @staticmethod
    def _text(document: str | dict[str, Any]) -> str:
        return str(document.get("text", "")) if isinstance(document, dict) else str(document)

    def scan(self, document: str | dict[str, Any]) -> Verdict:
        """Scan one document and emit score, flags, reasons, and triage action."""
        raw_text = self._text(document)
        normalized = normalize_content(raw_text)
        text = normalized.text
        weights: list[float] = []
        flags: list[str] = []
        reasons: list[str] = []

        def add(flag: str, weight: float, reason: str) -> None:
            flags.append(flag)
            weights.append(weight)
            reasons.append(reason)

        for flag, pattern, weight, reason in RULES:
            if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                add(flag, weight, reason)
        signal_weights = {
            "hidden_html_comment": (0.70, "HTML comment contains hidden text"),
            "hidden_html_style": (0.72, "CSS/HTML-hidden text was exposed for inspection"),
            "zero_width_or_bidi_removed": (0.62, "zero-width or bidirectional controls were removed"),
            "confusable_folded": (0.20, "look-alike Unicode characters were folded"),
            "base64_candidate": (0.38, "bounded base64 decoding exposed printable text"),
            "hex_candidate": (0.35, "bounded hex decoding exposed printable text"),
            "rot13_candidate": (0.30, "ROT13-like text was decoded for inspection"),
        }
        for signal in normalized.signals:
            if signal in signal_weights:
                weight, reason = signal_weights[signal]
                add(signal, weight, reason)

        # Only markdown/images that point to a non-allowlisted host are a strong link signal.
        for match in re.finditer(r"!?\[[^\]]*\]\((https?://[^\s)]+)", text, re.IGNORECASE):
            host = (urlparse(match.group(1)).hostname or "").lower()
            if host and not any(host == domain or host.endswith("." + domain) for domain in self.allowed_domains):
                add("non_allowlisted_markdown_url", 0.72, f"markdown target host is not allowlisted: {host}")
                break
        if re.search(r"https?://attacker\.invalid", text, re.IGNORECASE):
            add("dummy_exfil_url", 0.72, "reserved dummy exfiltration domain appears in content")
        if re.search(r"(?:assistant|model|AI|you)\s*[, :]?.{0,50}\bmust\b", text, re.IGNORECASE | re.DOTALL):
            add("imperative_density", 0.28, "imperative wording is close to an assistant/model reference")
        scripts = Counter("latin" if ord(c) < 128 else "other" for c in text if c.isalpha())
        alpha_total = sum(scripts.values())
        if alpha_total > 20 and scripts["other"] / alpha_total > 0.20:
            add("mixed_script_ratio", 0.25, "non-Latin script ratio is high; context determines whether benign")
        words = re.findall(r"[a-z]{3,}", text.lower())
        counts = Counter(words)
        if len(words) >= 16 and sum(v for v in counts.values() if v >= 4) / len(words) > 0.30:
            add("keyword_stuffing", 0.35, "repeated terms may be intended to bias retrieval ranking")
        metadata = document if isinstance(document, dict) else {}
        claimed_source = str(metadata.get("claimed_source", ""))
        source = str(metadata.get("source", ""))
        if claimed_source and source and claimed_source.casefold() not in source.casefold():
            add("claimed_source_mismatch", 0.58, "claimed source differs from recorded source")
        score = 1.0
        for weight in weights:
            score *= 1.0 - weight
        score = 1.0 - score
        action = "block" if score >= self.t_block else "quarantine" if score >= self.t_review else "allow"
        return Verdict(score, flags, reasons, action)
