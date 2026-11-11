"""Small cosine-similarity index using NumPy with metadata-aware filtering."""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np


class NumpyVectorStore:
    """Deterministic in-memory dense-vector index for local experiments."""
    def __init__(self) -> None:
        self._vectors = np.empty((0, 0), dtype=np.float32)
        self._chunks: list[dict[str, Any]] = []

    def add(self, vectors: np.ndarray, chunks: Iterable[dict[str, Any]]) -> None:
        """Append normalized vectors and matching chunk metadata records."""
        rows = list(chunks)
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(rows):
            raise ValueError("Vector rows and chunk metadata must have matching lengths")
        if not rows:
            return
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.maximum(norms, 1e-12)
        if self._vectors.size == 0:
            self._vectors = matrix
        else:
            if matrix.shape[1] != self._vectors.shape[1]:
                raise ValueError("Embedding dimensions do not match existing index")
            self._vectors = np.vstack([self._vectors, matrix])
        self._chunks.extend(rows)

    @property
    def chunks(self) -> list[dict[str, Any]]:
        """Expose indexed chunk metadata for alternate lexical retrieval backends."""
        return self._chunks

    def search(self, query_vector: np.ndarray, top_k: int = 5,
               allowed_tiers: set[str] | None = None,
               exclude_quarantined: bool = True) -> list[dict[str, Any]]:
        """Return the highest cosine-scoring eligible chunks."""
        if self._vectors.size == 0 or top_k <= 0:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.size != self._vectors.shape[1]:
            raise ValueError("Query embedding dimension does not match index")
        query = query / max(float(np.linalg.norm(query)), 1e-12)
        scores = self._vectors @ query
        order = np.argsort(-scores)
        output: list[dict[str, Any]] = []
        for idx in order:
            chunk = self._chunks[int(idx)]
            if allowed_tiers is not None and chunk.get("trust_tier", "T2") not in allowed_tiers:
                continue
            if exclude_quarantined and chunk.get("quarantine_status", "ALLOWED") == "QUARANTINED":
                continue
            item = dict(chunk)
            item["score"] = float(scores[int(idx)])
            output.append(item)
            if len(output) >= top_k:
                break
        return output

    def __len__(self) -> int:
        """Return the number of indexed chunks."""
        return len(self._chunks)
