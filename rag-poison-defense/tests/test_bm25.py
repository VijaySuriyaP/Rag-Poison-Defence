"""Regression tests for BM25 and rank-fusion retrieval modes."""
from __future__ import annotations

import pytest

from ragdef.rag.bm25 import BM25Index
from ragdef.rag.pipeline import RAGPipeline
from ragdef.rag.victim_llm import SimulatedVictim


def test_bm25_ranks_matching_chunk_first():
    chunks = [
        {"chunk_id": "a", "text": "The rare quasar spectrum was measured carefully.", "doc_id": "a"},
        {"chunk_id": "b", "text": "The quasar was photographed from an observatory.", "doc_id": "b"},
    ]
    index = BM25Index()
    index.fit(chunks)
    results = index.search("quasar spectrum", chunks, 2)
    assert results[0]["doc_id"] == "a"
    assert results[0]["score"] > results[1]["score"]


@pytest.mark.parametrize("mode", ["bm25", "hybrid"])
def test_pipeline_supports_lexical_and_hybrid_retrieval(mode: str):
    config = {
        "seed": 42,
        "corpus": {"chunk_size": 120, "chunk_overlap": 0},
        "retrieval": {"embedder": mode, "dense_embedder": "hashing", "top_k": 1},
    }
    pipeline = RAGPipeline(config, SimulatedVictim(0.0, seed=42))
    pipeline.ingest([
        {"doc_id": "quasar", "text": "The rare quasar spectrum was measured carefully."},
        {"doc_id": "recipe", "text": "A recipe for apple pie with cinnamon."},
    ])
    result = pipeline.retrieve("quasar spectrum")
    assert result
    assert result[0]["doc_id"] == "quasar"
