"""Sentence-aware text chunking with configurable overlap."""
from __future__ import annotations

import re
from typing import Any

_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def chunk_text(text: str, chunk_size: int = 480, overlap: int = 80) -> list[str]:
    """Split text into approximately sized chunks without dropping sentence text."""
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size >= 1 and 0 <= overlap < chunk_size")
    sentences = [part.strip() for part in _SENTENCE.split(text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        while len(sentence) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sentence[:chunk_size])
            sentence = sentence[chunk_size:]
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= chunk_size:
            current += " " + sentence
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = (tail + " " if tail else "") + sentence
    if current:
        chunks.append(current)
    return chunks


def chunk_document(document: dict[str, Any], chunk_size: int = 480, overlap: int = 80) -> list[dict[str, Any]]:
    """Create text chunks retaining document provenance metadata."""
    chunks = chunk_text(str(document.get("text", "")), chunk_size, overlap)
    results = []
    for i, text in enumerate(chunks):
        results.append({"chunk_id": f"{document.get('doc_id', 'doc')}::{i}", "text": text,
                        "doc_id": document.get("doc_id", "doc"), "source": document.get("source", "unknown"),
                        "signature": document.get("signature", ""), "trust_tier": document.get("trust_tier", "T2"),
                        "quarantine_status": document.get("quarantine_status", "ALLOWED"),
                        "metadata": {k: v for k, v in document.items() if k not in {"text"}}})
    return results
