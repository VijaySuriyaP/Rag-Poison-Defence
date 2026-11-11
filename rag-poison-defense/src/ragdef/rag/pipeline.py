"""Baseline RAG pipeline used by reproducible local experiments."""
from __future__ import annotations

import os
from typing import Any

from ragdef.config import load_config
from ragdef.rag.bm25 import BM25Index
from ragdef.rag.embedder import get_embedder
from ragdef.rag.ingest import Ingestor
from ragdef.rag.retriever import Retriever
from ragdef.rag.vector_store import NumpyVectorStore
from ragdef.rag.victim_llm import APIVictim, OllamaVictim, SimulatedVictim, VictimLLM


class RAGPipeline:
    """Combine document ingestion, retrieval and victim generation."""
    def __init__(self, config: dict[str, Any] | None = None, victim: VictimLLM | None = None) -> None:
        self.config = config or load_config()
        retrieval_cfg = self.config.get("retrieval", {})
        self.retrieval_mode = str(retrieval_cfg.get("embedder", "hashing")).lower()
        self.bm25_index = BM25Index() if self.retrieval_mode in {"bm25", "hybrid"} else None
        dense_name = str(retrieval_cfg.get("dense_embedder", "tfidf" if self.retrieval_mode == "hybrid" else "hashing"))
        self.embedder = get_embedder(dense_name if self.retrieval_mode in {"bm25", "hybrid"}
                                     else self.retrieval_mode)
        self.store = NumpyVectorStore()
        corpus_cfg = self.config.get("corpus", {})
        self.ingestor = Ingestor(self.store, self.embedder,
                                 int(corpus_cfg.get("chunk_size", 480)),
                                 int(corpus_cfg.get("chunk_overlap", 80)))
        self.retriever = Retriever(self.store, self.embedder,
                                   int(retrieval_cfg.get("top_k", 5)),
                                   mode=self.retrieval_mode, bm25_index=self.bm25_index,
                                   rrf_constant=int(retrieval_cfg.get("rrf_constant", 60)))
        if victim is not None:
            self.victim = victim
        else:
            settings = self.config.get("victim", {})
            mode = str(settings.get("mode", "simulated")).lower()
            if mode == "ollama":
                self.victim = OllamaVictim(os.getenv("RAGDEF_OLLAMA_MODEL", "llama3.2"))
            elif mode == "api":
                self.victim = APIVictim()
            else:
                self.victim = SimulatedVictim(float(settings.get("obey_prob", 0.70)),
                                              int(settings.get("seed", 42)),
                                              float(settings.get("encoded_obey_factor", 0.45)))

    def ingest(self, documents: list[dict[str, Any]]) -> int:
        """Index the supplied corpus and return number of chunks indexed."""
        count = self.ingestor.ingest(documents)
        if self.bm25_index is not None:
            self.bm25_index.fit(self.store.chunks)
        return count

    def retrieve(self, question: str, allowed_tiers: set[str] | None = None) -> list[dict[str, Any]]:
        """Return retrieved context chunks for evaluation or inspection."""
        return self.retriever.retrieve(question, allowed_tiers)

    def ask(self, question: str) -> str:
        """Retrieve context and generate an answer."""
        return self.victim.generate(question, self.retrieve(question))
