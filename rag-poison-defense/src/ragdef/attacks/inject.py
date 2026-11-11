"""Insert a supplied benchmark payload into a source document for a controlled test."""
from __future__ import annotations

from typing import Any


def inject_payload(document: dict[str, Any], payload: str, position: str) -> str:
    """Return full document text with payload at start, middle, end, or own chunk."""
    text = str(document.get("text", ""))
    if position == "start":
        return f"{payload}\n\n{text}"
    if position == "middle":
        midpoint = max(0, len(text) // 2)
        return f"{text[:midpoint]}\n\n{payload}\n\n{text[midpoint:]}"
    if position == "end":
        return f"{text}\n\n{payload}"
    if position == "own-chunk":
        return f"{text}\n\n--- separate untrusted chunk ---\n{payload}"
    raise ValueError(f"Unsupported injection position: {position}")
