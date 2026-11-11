"""Shared decision object emitted by independent defense scanners."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Verdict:
    """Explainable scanner result with normalized score, flags, reasons, and action."""
    score: float
    flags: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    action: str = "allow"

    def __post_init__(self) -> None:
        self.score = min(1.0, max(0.0, float(self.score)))
        self.flags = list(dict.fromkeys(self.flags))
        self.reasons = list(dict.fromkeys(self.reasons))

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation."""
        return {"score": self.score, "flags": self.flags, "reasons": self.reasons, "action": self.action}
