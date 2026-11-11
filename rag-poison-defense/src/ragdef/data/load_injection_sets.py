"""Expose normalized examples from the downloaded public attack datasets."""
from __future__ import annotations

import argparse
from typing import Any

from ragdef.utils import read_jsonl, write_jsonl


def try_external() -> list[dict[str, Any]]:
    """Read the normalized public dataset corpus already downloaded by prepare_public."""
    return [{"text": row.get("payload_text") or row.get("text", ""),
             "label": int(row.get("label", 1)), "source": row.get("source", "public-dataset"),
             "lang": "unknown", "split": row.get("split", "")}
            for row in read_jsonl("data/attacks/attack_set.jsonl") if row.get("text")]


def main() -> None:
    """Write the normalized injection pool to the raw data directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--try-online", action="store_true", help="deprecated; public datasets are now required")
    args = parser.parse_args()
    rows = try_external()
    if not rows:
        raise RuntimeError("Public dataset records are missing. Run `make data`; synthetic fallback is disabled.")
    write_jsonl("data/raw/injection_pool.jsonl", rows)
    print(f"Wrote {len(rows)} normalized injection examples.")


if __name__ == "__main__":
    main()
