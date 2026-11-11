"""Generation output guard for canaries and non-allowlisted markdown URLs."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse


@dataclass
class GuardResult:
    """Redaction decision and reasons for unsafe generated output."""
    text: str
    blocked: bool
    flags: list[str]


class OutputGuard:
    """Detect canary leakage, dummy exfil URLs, and unexpected markdown links/images."""
    def __init__(self, allowed_domains: set[str] | None = None) -> None:
        self.allowed_domains = {d.lower() for d in (allowed_domains or {"example.org", "wikipedia.org"})}
        self.events: list[dict[str, Any]] = []

    def guard(self, output: str, doc_id: str = "") -> GuardResult:
        """Redact output when a canary or suspicious link/image is emitted."""
        flags: list[str] = []
        if re.search(r"CANARY-[A-F0-9]{8}", output):
            flags.append("canary_output")
        if re.search(r"https?://attacker\.invalid", output, re.IGNORECASE):
            flags.append("dummy_exfil_url")
        for match in re.finditer(r"!?\[[^\]]*\]\((https?://[^\s)]+)", output, re.IGNORECASE):
            host = (urlparse(match.group(1)).hostname or "").lower()
            if host and not any(host == d or host.endswith("." + d) for d in self.allowed_domains):
                flags.append("non_allowlisted_markdown_url")
                break
        blocked = bool(flags)
        text = "[REDACTED: unsafe output blocked by output guard]" if blocked else output
        if blocked:
            self.events.append({"doc_id": doc_id, "flags": flags, "output_hash": __import__("hashlib").sha256(output.encode()).hexdigest()})
        return GuardResult(text, blocked, list(dict.fromkeys(flags)))
