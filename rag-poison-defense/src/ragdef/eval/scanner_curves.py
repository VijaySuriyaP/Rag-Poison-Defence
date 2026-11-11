"""Select scanner thresholds on dev data and create held-out ROC/PR artifacts."""
from __future__ import annotations

from typing import Any, Sequence

from ragdef.defenses.ml_scanner import MLScanner
from ragdef.defenses.rule_scanner import RuleScanner
from ragdef.eval.metrics import classification_metrics
from ragdef.utils import read_jsonl, write_csv


def curve_points(scores: Sequence[float], labels: Sequence[int],
                 thresholds: Sequence[float]) -> list[dict[str, float]]:
    """Convert continuous risk scores to FPR/TPR/precision/recall points."""
    points = []
    for threshold in thresholds:
        metrics = classification_metrics(labels, [int(score >= threshold) for score in scores])
        points.append({"threshold": float(threshold), "fpr": metrics["fpr"], "tpr": metrics["recall"],
                       "precision": metrics["precision"], "recall": metrics["recall"], "f1": metrics["f1"]})
    return points


def select_threshold(scores: Sequence[float], labels: Sequence[int]) -> dict[str, float]:
    """Choose the threshold with the best dev F1, breaking ties by precision then threshold."""
    if not scores or len(scores) != len(labels):
        raise ValueError("Threshold selection requires non-empty matched scores and labels")
    candidates = sorted({0.0, 1.000001, *(round(float(score), 6) for score in scores)})
    best: dict[str, float] | None = None
    best_key = (-1.0, -1.0, -1.0)
    for threshold in candidates:
        metrics = classification_metrics(labels, [int(score >= threshold) for score in scores])
        row = {"selected_threshold": threshold, "dev_fpr": metrics["fpr"],
               "dev_precision": metrics["precision"], "dev_recall": metrics["recall"],
               "dev_f1": metrics["f1"]}
        key = (metrics["f1"], metrics["precision"], threshold)
        if key > best_key:
            best_key = key
            best = row
    assert best is not None
    return best


def _text(row: dict[str, Any]) -> str:
    """Select the normalized string field used by the scanner."""
    return str(row.get("full_text", row.get("text", "")))


def _split(rows: list[dict[str, Any]], name: str) -> tuple[list[dict[str, Any]], str]:
    """Return the requested family split, explicitly recording a fallback when absent."""
    selected = [row for row in rows if row.get("split") == name]
    if selected:
        return selected, name
    # Small smoke fixtures have only one family and cannot populate dev/test partitions.
    return rows, f"fallback_all_rows_no_{name}"


def run_scanner_curves(seed: int = 42, smoke: bool = False) -> list[dict[str, Any]]:
    """Evaluate D1/D2 on dev for threshold choice and on test for reported metrics/curves."""
    del seed  # The model is trained deterministically from the configured training split.
    attacks = read_jsonl("data/attacks/attack_set.jsonl")
    negatives = read_jsonl("data/attacks/benign_hard_negatives.jsonl")
    clean = read_jsonl("data/clean/clean_docs.jsonl")
    if not attacks or not negatives or not clean:
        return []
    attack_dev, attack_dev_source = _split(attacks, "dev")
    negative_dev, negative_dev_source = _split(negatives, "dev")
    attack_test, attack_test_source = _split(attacks, "test")
    negative_test, negative_test_source = _split(negatives, "test")
    clean_count = 15 if smoke else 150
    clean_dev = clean[:clean_count]
    clean_test = clean[clean_count:clean_count * 2]
    limit = 30 if smoke else 240
    attack_dev = attack_dev[:limit]
    negative_dev = negative_dev[:max(limit, 300)]
    attack_test = attack_test[:limit]
    negative_test = negative_test[:max(limit, 300)]
    dev_texts = [_text(row) for row in attack_dev] + [_text(row) for row in negative_dev] + [_text(row) for row in clean_dev]
    dev_labels = [1] * len(attack_dev) + [0] * (len(negative_dev) + len(clean_dev))
    test_texts = [_text(row) for row in attack_test] + [_text(row) for row in negative_test] + [_text(row) for row in clean_test]
    test_labels = [1] * len(attack_test) + [0] * (len(negative_test) + len(clean_test))
    rule = RuleScanner()
    ml = MLScanner().fit_from_files("models/prompt_injection_scanner.joblib", force=True)
    scanners = [("D1_rule", rule), ("D2_ml", ml)]
    grid = [round(i / 20, 3) for i in range(21)]
    rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for name, scanner in scanners:
        dev_scores = _scores(scanner, dev_texts)
        test_scores = _scores(scanner, test_texts)
        chosen = select_threshold(dev_scores, dev_labels)
        final = classification_metrics(test_labels, [int(score >= chosen["selected_threshold"]) for score in test_scores])
        selection_rows.append({"scanner": name, **chosen, "test_fpr": final["fpr"],
                               "test_precision": final["precision"], "test_recall": final["recall"],
                               "test_f1": final["f1"], "n_dev_attacks": len(attack_dev),
                               "n_dev_hard_negative": len(negative_dev), "n_dev_clean": len(clean_dev),
                               "n_test_attacks": len(attack_test), "n_test_hard_negative": len(negative_test),
                               "n_test_clean": len(clean_test), "attack_dev_source": attack_dev_source,
                               "negative_dev_source": negative_dev_source, "attack_test_source": attack_test_source,
                               "negative_test_source": negative_test_source,
                               "selection_method": "max_dev_f1_then_precision_then_threshold"})
        for point in curve_points(test_scores, test_labels, grid):
            rows.append({"scanner": name, "n_attack_test": len(attack_test),
                         "n_hard_negative": len(negative_test), "n_clean": len(clean_test), **point})
    write_csv("results/tables/scanner_roc_pr.csv", rows)
    write_csv("results/tables/scanner_threshold_selection.csv", selection_rows)
    return rows


def _score(scanner: Any, text: str) -> float:
    """Return the scanner's normalized risk score."""
    return float(scanner.scan(text).score)


def _scores(scanner: Any, texts: list[str]) -> list[float]:
    """Use batch inference for the ML model and normal scanning for rules."""
    if isinstance(scanner, MLScanner):
        return scanner.score_many(texts)
    return [_score(scanner, text) for text in texts]


def main() -> None:
    """CLI entry point for ROC/PR and dev-selected threshold generation."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    points = run_scanner_curves(args.seed, args.smoke)
    print(f"Wrote {len(points)} ROC/PR points and the dev-selected threshold table.")


if __name__ == "__main__":
    main()
