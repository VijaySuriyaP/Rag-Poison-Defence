"""Formatting helpers shared by report generation."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV artifact, returning an empty list when it has not been generated."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt(value: Any, field: str = "") -> str:
    number = _number(value, float("nan"))
    if number != number:
        return str(value if value not in (None, "") else "—")
    if any(token in field for token in ("rate", "asr", "fpr", "recall", "precision", "f1", "utility", "delta", "tpr", "fpr", "gap")):
        return f"{number:.1%}"
    if field in {"count", "n", "n_trials", "n_attacks", "poison_docs", "clean_docs", "threshold_count", "n_runs", "defense_runs", "class_runs", "trial_rows", "paired_comparisons", "cost_configs"}:
        return f"{number:,.0f}"
    if "p_" in field or field.endswith("_p"):
        return f"{number:.3g}"
    if "ms" in field:
        return f"{number:.2f}"
    if "cost" in field or "hours" in field or "incidents" in field or "reviews" in field:
        return f"{number:,.2f}"
    return f"{number:.3f}"


def markdown_table(rows: Iterable[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    """Render a concise Markdown table from dictionaries and field/label pairs."""
    materialized = list(rows)
    if not materialized:
        return "_No result artifact was available for this section yet._"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in materialized:
        cells = []
        for field, _label in columns:
            value = _fmt(row.get(field, ""), field)
            cells.append(value.replace("|", "\\|").replace("\n", " "))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *body])


