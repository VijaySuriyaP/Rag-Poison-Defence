"""Document ingestion with ordered extension hook points for defenses."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

from ragdef.rag.chunker import chunk_document
from ragdef.rag.embedder import HashingEmbedder
from ragdef.rag.vector_store import NumpyVectorStore

Hook = Callable[[dict[str, Any]], dict[str, Any] | None]
HOOK_STAGES = ("pre_parse", "post_parse", "pre_embed", "pre_index")


class Ingestor:
    """Parse, chunk, embed and index documents while allowing defense hooks."""
    def __init__(self, store: NumpyVectorStore, embedder: Any | None = None,
                 chunk_size: int = 480, overlap: int = 80,
                 hooks: dict[str, list[Hook]] | None = None) -> None:
        self.store = store
        self.embedder = embedder or HashingEmbedder()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.hooks: dict[str, list[Hook]] = defaultdict(list)
        for stage, callbacks in (hooks or {}).items():
            if stage not in HOOK_STAGES:
                raise ValueError(f"Unknown ingestion hook stage: {stage}")
            self.hooks[stage].extend(callbacks)
        self.last_stats = {"documents_seen": 0, "documents_indexed": 0, "chunks_indexed": 0}

    def _run_hooks(self, stage: str, document: dict[str, Any]) -> dict[str, Any] | None:
        current: dict[str, Any] | None = document
        for hook in self.hooks[stage]:
            if current is None:
                break
            current = hook(current)
        return current

    def ingest(self, documents: list[dict[str, Any]]) -> int:
        """Apply hook stages to input docs then append chunks to vector store."""
        all_chunks: list[dict[str, Any]] = []
        self.last_stats = {"documents_seen": len(documents), "documents_indexed": 0, "chunks_indexed": 0}
        for original in documents:
            document = dict(original)
            for stage in ("pre_parse", "post_parse", "pre_embed"):
                document = self._run_hooks(stage, document)
                if document is None:
                    break
            if document is None:
                continue
            chunks = chunk_document(document, self.chunk_size, self.overlap)
            kept_chunks: list[dict[str, Any]] = []
            for chunk in chunks:
                filtered = self._run_hooks("pre_index", chunk)
                if filtered is not None:
                    kept_chunks.append(filtered)
            if kept_chunks:
                all_chunks.extend(kept_chunks)
                self.last_stats["documents_indexed"] += 1
        if all_chunks:
            vectors = self.embedder.encode([c["text"] for c in all_chunks])
            self.store.add(vectors, all_chunks)
        self.last_stats["chunks_indexed"] = len(all_chunks)
        return len(all_chunks)
