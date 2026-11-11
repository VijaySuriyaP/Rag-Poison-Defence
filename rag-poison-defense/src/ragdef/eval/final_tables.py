"""Consolidate raw run checkpoints into comparison, ASR, cost, and trade-off tables."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from ragdef.config import load_config
from ragdef.eval.fp_cost import cost_per_10k
from ragdef.eval.stats import paired_binary_test
from ragdef.utils import write_csv


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def consolidate_tables() -> dict[str, int]:
    """Build final comparison tables and operating-cost estimates from raw CSVs."""
    raw_files = sorted(Path("results/raw").glob("defense_*.csv"))
    raw = [row for path in raw_files for row in _rows(path)]
    overall = [r for r in raw if r.get("record_type") == "overall"]
    classes = [r for r in raw if r.get("record_type") == "class"]
    trials = [r for r in raw if r.get("record_type") == "trial"]

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in overall:
        groups[(str(row.get("defense", "")), str(row.get("poison_rate", "")))].append(row)
    asr_table = []
    for (defense, rate), rows in sorted(groups.items()):
        vals = [_float(r, "asr") for r in rows]
        asr_table.append({"defense": defense, "poison_rate": rate, "threshold_count": len(rows),
                          "mean_asr_over_thresholds": sum(vals) / max(1, len(vals)),
                          "min_asr": min(vals, default=0.0), "max_asr": max(vals, default=0.0),
                          "mean_retrieval_rate": sum(_float(r, "retrieval_rate") for r in rows) / max(1, len(rows)),
                          "mean_asr_ci_low": sum(_float(r, "asr_ci_low") for r in rows) / max(1, len(rows)),
                          "mean_asr_ci_high": sum(_float(r, "asr_ci_high") for r in rows) / max(1, len(rows))})
    write_csv("results/tables/asr_by_defense_poison_rate.csv", asr_table)

    class_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in classes:
        class_groups[(str(row.get("defense", "")), str(row.get("attack_class", "")))].append(row)
    class_table = []
    for (defense, attack_class), rows in sorted(class_groups.items()):
        class_table.append({"defense": defense, "attack_class": attack_class,
                           "mean_asr": sum(_float(r, "asr") for r in rows) / max(1, len(rows)),
                           "mean_retrieval_rate": sum(_float(r, "retrieval_rate") for r in rows) / max(1, len(rows)),
                           "n_runs": len(rows)})
    write_csv("results/tables/asr_by_class_defense.csv", class_table)

    # Pair baseline `none` with each configuration for the same rate, threshold, attack, query.
    indexed: dict[tuple[str, str, str, str, str], int] = {}
    for row in trials:
        key = (str(row.get("defense", "")), str(row.get("poison_rate", "")),
               str(row.get("threshold", "")), str(row.get("attack_doc_id", "")),
               str(row.get("query_index", "")))
        indexed[key] = int(str(row.get("success", "False")).lower() in {"true", "1"})
    trial_keys = {(rate, threshold, doc, query) for defense, rate, threshold, doc, query in indexed if defense == "none"}
    comparison_rows = []
    defenses = sorted({str(r.get("defense", "")) for r in overall if r.get("defense") not in (None, "none", "")})
    for defense in defenses:
        matched = [(rate, threshold, doc, query) for rate, threshold, doc, query in sorted(trial_keys)
                   if (defense, rate, threshold, doc, query) in indexed]
        base_values = [indexed[("none", rate, threshold, doc, query)] for rate, threshold, doc, query in matched]
        defense_values = [indexed[(defense, rate, threshold, doc, query)] for rate, threshold, doc, query in matched]
        stats = paired_binary_test(base_values, defense_values, samples=500)
        comparison_rows.append({"baseline": "none", "defense": defense, **stats})
    write_csv("results/tables/defense_comparisons.csv", comparison_rows)

    # Consolidate measured clean FPR, quarantine rate, latency and expected cost per 10k.
    config = load_config()
    cost_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in overall:
        cost_groups[str(row.get("defense", ""))].append(row)
    cost_rows = []
    cost_cfg = config.get("cost", {})
    review_cfg = config.get("review", {})
    for defense, rows in sorted(cost_groups.items()):
        fp = sum(_float(r, "legit_docs_blocked_rate") for r in rows) / max(1, len(rows))
        quarantine = sum(_float(r, "legit_docs_quarantined_rate") for r in rows) / max(1, len(rows))
        asr = sum(_float(r, "asr") for r in rows) / max(1, len(rows))
        hard_fpr = sum(_float(r, "hard_negative_fpr") for r in rows) / max(1, len(rows))
        latency = sum(_float(r, "ingest_ms_per_doc") for r in rows) / max(1, len(rows))
        query_latency = sum(_float(r, "query_ms_mean") for r in rows) / max(1, len(rows))
        cost = cost_per_10k(fp, quarantine, 10000, float(review_cfg.get("minutes_per_doc", 4.0)),
                            float(cost_cfg.get("cost_per_review_hour", 45.0)),
                            float(cost_cfg.get("cost_per_incident", 2500.0)),
                            max(config.get("poison_rates", [0.01])), asr,
                            float(cost_cfg.get("cost_per_legit_block", 0.0)))
        cost_rows.append({"defense": defense, "mean_clean_fpr_blocked": fp,
                          "mean_clean_quarantine_rate": quarantine, "mean_hard_negative_fpr": hard_fpr,
                          "mean_ingest_ms_per_doc": latency, "mean_query_ms": query_latency,
                          "mean_asr": asr, **cost})
    write_csv("results/tables/fp_cost_by_defense_config.csv", cost_rows)
    # Bounded scenarios make configured-cost uncertainty visible rather than implying
    # that a single illustrative dollar estimate is a forecast.
    cost_sensitivity = []
    scenario_multipliers = {
        "low": {"review_minutes": 0.5, "hourly_rate": 0.5, "incident_cost": 0.5, "block_cost": 0.5},
        "base": {"review_minutes": 1.0, "hourly_rate": 1.0, "incident_cost": 1.0, "block_cost": 1.0},
        "high": {"review_minutes": 2.0, "hourly_rate": 2.0, "incident_cost": 4.0, "block_cost": 2.0},
    }
    for row in cost_rows:
        for scenario, factors in scenario_multipliers.items():
            assumptions = cost_per_10k(
                _float(row, "mean_clean_fpr_blocked"),
                _float(row, "mean_clean_quarantine_rate"), 10000,
                float(review_cfg.get("minutes_per_doc", 4.0)) * factors["review_minutes"],
                float(cost_cfg.get("cost_per_review_hour", 45.0)) * factors["hourly_rate"],
                float(cost_cfg.get("cost_per_incident", 2500.0)) * factors["incident_cost"],
                max(config.get("poison_rates", [0.01])), _float(row, "mean_asr"),
                float(cost_cfg.get("cost_per_legit_block", 0.0)) * factors["block_cost"])
            cost_sensitivity.append({
                "scenario": scenario, "defense": row["defense"],
                "review_minutes_per_doc": float(review_cfg.get("minutes_per_doc", 4.0)) * factors["review_minutes"],
                "review_cost_per_hour": float(cost_cfg.get("cost_per_review_hour", 45.0)) * factors["hourly_rate"],
                "incident_cost": float(cost_cfg.get("cost_per_incident", 2500.0)) * factors["incident_cost"],
                "cost_per_legit_block": float(cost_cfg.get("cost_per_legit_block", 0.0)) * factors["block_cost"],
                "poison_rate_assumption": max(config.get("poison_rates", [0.01])),
                "mean_asr": _float(row, "mean_asr"),
                "mean_clean_blocked_rate": _float(row, "mean_clean_fpr_blocked"),
                "mean_clean_quarantine_rate": _float(row, "mean_clean_quarantine_rate"),
                **assumptions})
    write_csv("results/tables/cost_sensitivity.csv", cost_sensitivity)
    write_csv("results/tables/roc_pr_thresholds.csv", [
        {k: row.get(k, "") for k in ("defense", "threshold", "poison_rate", "asr", "asr_ci_low", "asr_ci_high",
                                      "hard_negative_fpr", "legit_docs_blocked_rate", "legit_docs_quarantined_rate")}
        for row in overall])
    return {"defense_runs": len(overall), "class_runs": len(classes), "trial_rows": len(trials),
            "asr_rate_rows": len(asr_table), "class_summary_rows": len(class_table),
            "paired_comparisons": len(comparison_rows), "cost_configs": len(cost_rows),
            "cost_sensitivity_rows": len(cost_sensitivity)}


def main() -> None:
    """CLI entry point for final table consolidation."""
    summary = consolidate_tables()
    print("Consolidated tables: " + ", ".join(f"{k}={v}" for k, v in summary.items()))


if __name__ == "__main__":
    main()
