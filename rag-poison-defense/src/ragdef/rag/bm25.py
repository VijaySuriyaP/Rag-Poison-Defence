"""Deterministic BM25 scoring over an in-memory chunk corpus."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class BM25Index:
    """Small corpus-fitted BM25 index; scores chunks without densifying term matrices."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 1),
                                          token_pattern=r"(?u)\b\w\w+\b")
        self.counts = None
        self.doc_lengths = np.empty(0, dtype=np.float32)
        self.average_length = 0.0
        self.idf = np.empty(0, dtype=np.float32)
        self.fitted = False

    def fit(self, chunks: list[dict[str, Any]]) -> None:
        """Fit document frequencies and lengths to the indexed chunks."""
        texts = [str(chunk.get("text", "")) for chunk in chunks]
        if not texts:
            self.counts = None
            self.doc_lengths = np.empty(0, dtype=np.float32)
            self.average_length = 0.0
            self.idf = np.empty(0, dtype=np.float32)
            self.fitted = True
            return
        self.counts = self.vectorizer.fit_transform(texts).tocsr()
        self.doc_lengths = np.asarray(self.counts.sum(axis=1)).ravel().astype(np.float32)
        self.average_length = float(self.doc_lengths.mean()) if len(self.doc_lengths) else 0.0
        document_frequency = np.asarray((self.counts > 0).sum(axis=0)).ravel()
        total_documents = len(texts)
        self.idf = np.log1p((total_documents - document_frequency + 0.5) /
                            (document_frequency + 0.5)).astype(np.float32)
        self.fitted = True

    def search(self, query: str, chunks: list[dict[str, Any]], top_k: int,
               allowed_tiers: set[str] | None = None,
               exclude_quarantined: bool = True) -> list[dict[str, Any]]:
        """Return BM25-ranked chunks, respecting trust and quarantine filters."""
        if not self.fitted:
            raise RuntimeError("BM25 index is not fitted; ingest documents before retrieval")
        if self.counts is None or top_k <= 0 or not query.strip():
            return []
        query_vector = self.vectorizer.transform([query])
        term_indices = query_vector.indices
        if not len(term_indices):
            return []
        scores = np.zeros(len(chunks), dtype=np.float32)
        lengths = self.doc_lengths
        length_norm = self.k1 * (1 - self.b + self.b * lengths / max(self.average_length, 1e-12))
        for term_index in term_indices:
            term_frequency = self.counts[:, term_index].toarray().ravel().astype(np.float32)
            contribution = (self.idf[term_index] * term_frequency * (self.k1 + 1) /
                            (term_frequency + length_norm))
            scores += contribution
        order = np.argsort(-scores, kind="stable")
        output: list[dict[str, Any]] = []
        for index in order:
            chunk = chunks[int(index)]
            if allowed_tiers is not None and chunk.get("trust_tier", "T2") not in allowed_tiers:
                continue
            if exclude_quarantined and chunk.get("quarantine_status", "ALLOWED") == "QUARANTINED":
                continue
            if scores[int(index)] <= 0:
                break
            item = dict(chunk)
            item["score"] = float(scores[int(index)])
            output.append(item)
            if len(output) >= top_k:
                break
        return output
