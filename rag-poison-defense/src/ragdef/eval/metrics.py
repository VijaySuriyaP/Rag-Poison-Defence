"""Metric definitions for attack success, retrieval, clean utility, and scan cost."""
from __future__ import annotations

import re
from typing import Any, Sequence

from ragdef.eval.stats import bootstrap_ci


def contains_gold(answer: str, gold: str) -> float:
    """Return 1 when a normalized gold answer appears in the generated answer."""
    normalize = lambda s: re.sub(r"\s+", " ", s.casefold()).strip().strip(" .,!?:;")
    target = normalize(gold)
    return float(bool(target) and target in normalize(answer))


def token_f1(answer: str, gold: str) -> float:
    """Compute token-overlap F1 for a generated answer versus its reference."""
    a = re.findall(r"\w+", answer.casefold())
    g = re.findall(r"\w+", gold.casefold())
    if not a or not g:
        return float(a == g)
    from collections import Counter
    overlap = sum((Counter(a) & Counter(g)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(a), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def classification_metrics(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, float]:
    """Compute precision, recall, F1, FPR, and confusion counts for binary labels."""
    tp = sum(int(y == 1 and p == 1) for y, p in zip(labels, predictions))
    tn = sum(int(y == 0 and p == 0) for y, p in zip(labels, predictions))
    fp = sum(int(y == 0 and p == 1) for y, p in zip(labels, predictions))
    fn = sum(int(y == 1 and p == 0) for y, p in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": float(tp), "tn": float(tn), "fp": float(fp), "fn": float(fn),
            "precision": precision, "recall": recall, "f1": f1,
            "fpr": fp / (fp + tn) if fp + tn else 0.0}


def utility_summary(scores: Sequence[float], bootstrap_samples: int = 500, seed: int = 42) -> dict[str, float]:
    """Summarize clean QA success and its bootstrap 95% confidence interval."""
    values = list(scores)
    mean = sum(values) / len(values) if values else 0.0
    low, high = bootstrap_ci(values, samples=bootstrap_samples, seed=seed)
    return {"utility": mean, "utility_ci_low": low, "utility_ci_high": high, "n": float(len(values))}
