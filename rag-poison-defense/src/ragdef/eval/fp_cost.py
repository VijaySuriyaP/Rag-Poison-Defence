"""False-positive, reviewer-burden, latency, utility-loss, and operations cost model."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

from ragdef.config import load_config
from ragdef.defenses.ml_scanner import MLScanner
from ragdef.defenses.rule_scanner import RuleScanner
from ragdef.eval.metrics import classification_metrics
from ragdef.utils import read_jsonl, write_csv


def cost_per_10k(fp_rate: float, quarantine_rate: float, docs_per_10k: int = 10000,
                 review_minutes: float = 4.0, review_hour_cost: float = 45.0,
                 incident_cost: float = 2500.0, poison_rate: float = 0.01,
                 asr: float = 0.2, legit_block_cost: float = 0.0) -> dict[str, float]:
    """Estimate reviewer hours and expected cost per 10,000 ingested documents.

    Assumptions: review applies only to quarantined docs; expected incident count is
    docs_per_10k * poison_rate * ASR. This simplified model excludes fixed infrastructure
    and treats incident cost as a configurable expected loss, not an observed payment.
    """
    hours = max(0.0, quarantine_rate) * docs_per_10k * max(0.0, review_minutes) / 60.0
    reviews = max(0.0, quarantine_rate) * docs_per_10k
    incidents = docs_per_10k * max(0.0, poison_rate) * max(0.0, asr)
    review_cost = hours * max(0.0, review_hour_cost)
    incident_loss = incidents * max(0.0, incident_cost)
    block_loss = max(0.0, fp_rate) * docs_per_10k * max(0.0, legit_block_cost)
    return {"reviewer_hours_per_10k": hours, "reviews_per_10k": reviews,
            "expected_incidents_per_10k": incidents, "review_cost_per_10k": review_cost,
            "expected_incident_loss_per_10k": incident_loss,
            "false_positive_block_cost_per_10k": block_loss,
            "expected_total_cost_per_10k": review_cost + incident_loss + block_loss,
            "false_positive_blocked_per_10k": max(0.0, fp_rate) * docs_per_10k}


def _score(scanner: Any, text: str) -> float:
    result = scanner.scan(text)
    return float(result.score)


def _scores(scanner: Any, texts: list[str]) -> list[float]:
    """Score a corpus efficiently, using vectorized ML inference when available."""
    if isinstance(scanner, MLScanner):
        return scanner.score_many(texts)
    return [_score(scanner, text) for text in texts]


def _classify_scores(scores: list[float], labels: list[int], threshold: float) -> dict[str, float]:
    return classification_metrics(labels, [int(s >= threshold) for s in scores])


def _read_baseline_asr(default: float = 0.2) -> float:
    path = Path("results/tables/baseline_asr.csv")
    if not path.exists():
        return default
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    vals = [float(r["asr"]) for r in rows if r.get("asr") not in (None, "")]
    return sum(vals) / len(vals) if vals else default


def run_fp_cost(seed: int = 42, smoke: bool = False) -> list[dict[str, Any]]:
    """Evaluate scanner metrics on test attacks, hard negatives, and normal clean docs."""
    config = load_config()
    clean = read_jsonl("data/clean/clean_docs.jsonl")
    negatives = read_jsonl("data/attacks/benign_hard_negatives.jsonl")
    attacks = read_jsonl("data/attacks/attack_set.jsonl")
    if not negatives or not attacks:
        raise RuntimeError("Public attack and benign datasets missing; run `make data`. No synthetic fallback exists.")
    if not clean:
        raise RuntimeError("Public Wikipedia corpus missing; run `make data`. No synthetic fallback exists.")
    negatives = [r for r in negatives if r.get("split") == "test"] or negatives
    attack_test = [r for r in attacks if r.get("split") == "test"] or attacks
    clean = clean[:50 if smoke else 500]
    negatives = negatives[:50 if smoke else 600]
    attack_test = attack_test[:60 if smoke else 240]
    scanners: list[tuple[str, Any]] = [("D1_rule", RuleScanner(
        float(config["scanner"]["t_review"]), float(config["scanner"]["t_block"]),
        set(config["provenance"]["allowed_domains"])))]
    ml = MLScanner(float(config["scanner"]["t_review"]), float(config["scanner"]["t_block"]))
    ml.fit_from_files("models/prompt_injection_scanner.joblib")
    scanners.append(("D2_ml", ml))
    qa = read_jsonl("data/clean/qa.jsonl")
    if not qa:
        raise RuntimeError("Public benchmark QA rows missing; run `make data`.")
    baseline_asr = _read_baseline_asr()
    selected_thresholds = {}
    threshold_path = Path("results/tables/scanner_threshold_selection.csv")
    if threshold_path.exists():
        with threshold_path.open(encoding="utf-8", newline="") as stream:
            selected_thresholds = {
                row.get("scanner", ""): float(row["selected_threshold"])
                for row in csv.DictReader(stream)
                if row.get("selected_threshold") not in (None, "")
            }
    output: list[dict[str, Any]] = []
    review_mins = float(config["review"]["minutes_per_doc"])
    review_cost = float(config["cost"]["cost_per_review_hour"])
    incident_cost = float(config["cost"]["cost_per_incident"])
    legit_block_cost = float(config["cost"].get("cost_per_legit_block", 0.0))
    poison_rate = max(config.get("poison_rates", [0.01]))
    for name, scanner in scanners:
        start = time.perf_counter()
        hard_scores = _scores(scanner, [str(row.get("text", row.get("full_text", ""))) for row in negatives])
        clean_scores = _scores(scanner, [str(row.get("text", "")) for row in clean])
        attack_scores = _scores(scanner, [str(row.get("full_text", row.get("text", ""))) for row in attack_test])
        total_ms = (time.perf_counter() - start) * 1000
        threshold_block = selected_thresholds.get(name, float(config["scanner"]["t_block"]))
        threshold_review = min(float(config["scanner"]["t_review"]), max(0.0, threshold_block * 0.8))
        hard_metrics = _classify_scores(hard_scores + attack_scores, [0] * len(hard_scores) + [1] * len(attack_scores), threshold_block)
        hard_fpr = sum(s >= threshold_block for s in hard_scores) / max(1, len(hard_scores))
        clean_fpr = sum(s >= threshold_block for s in clean_scores) / max(1, len(clean_scores))
        hard_quarantine = sum(threshold_review <= s < threshold_block for s in hard_scores) / max(1, len(hard_scores))
        clean_quarantine = sum(threshold_review <= s < threshold_block for s in clean_scores) / max(1, len(clean_scores))
        block_ids = {row["doc_id"] for row, score in zip(clean, clean_scores) if score >= threshold_block}
        review_ids = {row["doc_id"] for row, score in zip(clean, clean_scores)
                      if threshold_review <= score < threshold_block}
        affected = sum(row.get("gold_doc_id") in (block_ids | review_ids) for row in qa)
        utility_loss = affected / max(1, len(qa))
        # Monetary assumptions use the normal clean corpus for expected operational rates;
        # the deliberately adversarial hard-negative suite remains a separate stress metric.
        assumptions = cost_per_10k(clean_fpr, clean_quarantine, 10000, review_mins,
                                   review_cost, incident_cost, poison_rate, baseline_asr,
                                   legit_block_cost)
        output.append({"defense": name, "threshold_block": threshold_block, "threshold_review": threshold_review,
                       "hard_negative_fpr": hard_fpr, "clean_corpus_fpr": clean_fpr,
                       "hard_negative_quarantine_rate": hard_quarantine,
                       "clean_corpus_quarantine_rate": clean_quarantine,
                       "precision_test": hard_metrics["precision"], "recall_test": hard_metrics["recall"],
                       "f1_test": hard_metrics["f1"], "legit_docs_blocked_rate": clean_fpr,
                       "legit_docs_quarantined_rate": clean_quarantine, "clean_qa_utility_loss_proxy": utility_loss,
                       "clean_docs_blocked_in_sample": len(block_ids), "scan_ms_per_doc": total_ms / max(1, len(negatives) + len(clean) + len(attack_test)),
                       "expected_asr_used": baseline_asr, "poison_rate_assumption": poison_rate,
                       "cost_per_legit_block": legit_block_cost,
                       "threshold_source": "dev-selected" if name in selected_thresholds else "config-fallback",
                       **assumptions})
    write_csv("results/tables/fp_cost_by_defense.csv", output)
    return output


def main() -> None:
    """Run false-positive and cost analysis."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    rows = run_fp_cost(args.seed, args.smoke)
    print(f"Wrote false-positive/cost table for {len(rows)} scanners.")


if __name__ == "__main__":
    main()
