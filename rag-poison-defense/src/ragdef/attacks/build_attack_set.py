"""Validate and summarize attack rows downloaded from the selected public datasets."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from ragdef.utils import read_jsonl


def build_attack_set(count_per_class: int | None = None, hard_negative_count: int | None = None,
                     seed: int = 42) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read the public benchmark corpus; never fabricate attacks or benign examples."""
    attacks = read_jsonl("data/attacks/attack_set.jsonl")
    negatives = read_jsonl("data/attacks/benign_hard_negatives.jsonl")
    if not attacks or not negatives:
        raise RuntimeError("Public datasets are missing. Run `make data`; synthetic fallback is disabled.")
    if any(not row.get("source_dataset") for row in attacks + negatives):
        raise RuntimeError("Unattributed rows found; public dataset provenance is required for every record.")
    return attacks, negatives


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count-per-class", type=int, default=None, help="deprecated; corpus sampling is set by make data")
    parser.add_argument("--hard-negatives", type=int, default=None, help="deprecated; corpus sampling is set by make data")
    parser.add_argument("--seed", type=int, default=42)
    parser.parse_args()
    attacks, negatives = build_attack_set()
    counts = Counter(row["source"] for row in attacks)
    print(f"Validated {len(attacks)} public attacks and {len(negatives)} public benign records: {dict(counts)}")


if __name__ == "__main__":
    main()
