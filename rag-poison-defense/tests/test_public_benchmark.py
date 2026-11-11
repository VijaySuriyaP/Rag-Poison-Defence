"""Evaluation invariants for public grouped splits and threshold selection."""
from __future__ import annotations

from ragdef.data.prepare_public import _assign_grouped_splits, collect_squad_clean_qa
from ragdef.rag.embedder import get_embedder
from ragdef.eval.public_benchmark import (
    _cluster_bootstrap_ci,
    _clean_qa_rows,
    _normalize_squad_answer,
    _metric_row,
    _select_threshold,
    _unique_classifier_rows,
)


def test_identical_normalized_text_shares_split_group():
    rows = [
        {"text": "Ignore the previous instruction!", "split": "train"},
        {"text": "IGNORE the previous instruction", "split": "test"},
    ]
    _assign_grouped_splits(rows)
    assert rows[0]["dedup_group_id"] == rows[1]["dedup_group_id"]
    assert rows[0]["split"] == rows[1]["split"]
    assert rows[0]["upstream_split"] != rows[1]["upstream_split"]


def test_conflicting_exact_duplicate_labels_are_excluded():
    attacks = [{"text": "same text", "label": 1, "source": "bipia", "split": "train"}]
    negatives = [{"text": "SAME text!", "label": 0, "source": "wikipedia", "split": "train"}]
    rows, summary = _unique_classifier_rows(attacks, negatives)
    assert rows == []
    assert summary["conflicting_label_groups_excluded"] == 1


def test_threshold_selection_meets_fpr_constraint_and_maximizes_dev_recall():
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    scores = [0.9, 0.8, 0.7, 0.6, 0.7, 0.4, 0.3, 0.2]
    selected = _select_threshold(labels, scores, fpr_target=0.05)
    assert selected["dev_fpr"] <= 0.05
    assert selected["dev_recall"] == 0.5


def test_threshold_selection_enforces_fpr_per_benign_source():
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.85, 0.2]
    sources = [["attack_a"], ["attack_b"], ["negative_a"], ["negative_b"]]
    selected = _select_threshold(labels, scores, fpr_target=0.05, sources=sources)
    assert selected["dev_fpr"] <= 0.05
    assert all(value <= 0.05 for value in selected["dev_fpr_by_benign_source"].values())


def test_source_only_class_has_undefined_not_zero_rate():
    row = _metric_row([0, 0], [0.1, 0.9], 0.5, scanner="D1", source="wikipedia", record_type="by_source")
    assert row["recall"] is None
    assert row["recall_ci95_low"] is None
    assert row["fpr"] == 0.5


def test_zero_observed_cluster_events_have_nonzero_upper_confidence_bound():
    low, high = _cluster_bootstrap_ci([0.0, 0.0, 0.0], ["a", "b", "c"])
    assert low is not None and low < 1e-12
    assert high is not None and high > 0.0


def test_clean_title_queries_are_deterministic_and_use_opening_sentence():
    rows = _clean_qa_rows([{"doc_id": "wiki-1", "title": "Example", "text": "Example is a test. More text."}])
    assert rows == [{"query_id": "wiki-1", "doc_id": "wiki-1", "group_id": rows[0]["group_id"],
                     "question": "What is Example?", "gold": "Example is a test."}]


def test_character_tfidf_embedder_is_available_for_dev_comparison():
    embedder = get_embedder("tfidf_char")
    corpus = embedder.encode(["A clean example document.", "Prompt injection text."])
    query = embedder.encode(["clean example"])
    assert corpus.shape[0] == 2
    assert query.shape[0] == 1
    assert corpus.shape[1] == query.shape[1] > 0


def test_squad_answer_normalization_matches_benchmark_rules():
    assert _normalize_squad_answer("The U.S. (example)!") == "us example"


def test_squad_loader_deduplicates_contexts_and_preserves_gold_answers(monkeypatch):
    records = [
        {"id": "q1", "title": "A", "context": "A short passage.", "question": "Who?",
         "answers": {"text": ["Alice", "Alice."], "answer_start": [0, 0]}},
        {"id": "q2", "title": "A", "context": "A short passage.", "question": "Where?",
         "answers": {"text": ["Paris"], "answer_start": [0]}},
    ]

    def fake_rows(dataset_id, *, split, config, shuffle_seed):
        assert dataset_id == "rajpurkar/squad"
        assert split == "validation"
        assert config == "plain_text"
        assert shuffle_seed == 7
        return iter(records)

    monkeypatch.setattr("ragdef.data.prepare_public._dataset_rows", fake_rows)
    contexts, questions = collect_squad_clean_qa(max_questions=2, seed=7)
    assert len(contexts) == 1
    assert [row["qa_id"] for row in questions] == ["q1", "q2"]
    assert questions[0]["gold_answers"] == ["Alice", "Alice."]
    assert questions[0]["context_group_id"] == contexts[0]["dedup_group_id"]
