"""Query encoding and retrieval facade."""
from __future__ import annotations

from typing import Any

from ragdef.rag.embedder import HashingEmbedder
from ragdef.rag.bm25 import BM25Index
from ragdef.rag.vector_store import NumpyVectorStore


class Retriever:
    """Retrieve top-k chunks from an existing vector store."""
    def __init__(self, store: NumpyVectorStore, embedder: Any | None = None, top_k: int = 5,
                 mode: str = "dense", bm25_index: BM25Index | None = None,
                 rrf_constant: int = 60) -> None:
        self.store = store
        self.embedder = embedder or HashingEmbedder()
        self.top_k = top_k
        self.mode = mode.lower()
        self.bm25_index = bm25_index
        self.rrf_constant = max(1, int(rrf_constant))

    def retrieve(self, query: str, allowed_tiers: set[str] | None = None) -> list[dict[str, Any]]:
        """Return top-k relevant chunks for a natural-language query."""
        if self.mode == "bm25":
            if self.bm25_index is None:
                raise RuntimeError("BM25 retrieval was selected without an index")
            return self.bm25_index.search(query, self.store.chunks, self.top_k,
                                          allowed_tiers=allowed_tiers)
        if self.mode == "hybrid":
            if self.bm25_index is None:
                raise RuntimeError("Hybrid retrieval was selected without a BM25 index")
            dense = self.store.search(self.embedder.encode([query])[0], len(self.store),
                                      allowed_tiers=allowed_tiers)
            lexical = self.bm25_index.search(query, self.store.chunks, len(self.store),
                                             allowed_tiers=allowed_tiers)
            fused: dict[str, tuple[float, dict[str, Any]]] = {}
            for ranked in (dense, lexical):
                for rank, chunk in enumerate(ranked, start=1):
                    identity = str(chunk.get("chunk_id", id(chunk)))
                    score, _ = fused.get(identity, (0.0, chunk))
                    fused[identity] = (score + 1.0 / (self.rrf_constant + rank), chunk)
            output = []
            for score, chunk in sorted(fused.values(), key=lambda item: item[0], reverse=True)[:self.top_k]:
                result = dict(chunk)
                result["score"] = score
                output.append(result)
            return output
        vector = self.embedder.encode([query])[0]
        return self.store.search(vector, self.top_k, allowed_tiers=allowed_tiers)
