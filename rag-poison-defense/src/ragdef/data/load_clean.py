"""Load real Wikimedia articles; there is deliberately no generated-data fallback."""
from __future__ import annotations

import argparse
from typing import Any

from ragdef.config import load_config
from ragdef.utils import write_jsonl


def try_wikipedia(size: int, seed: int = 42) -> list[dict[str, Any]]:
    """Try a public Wikipedia dataset; return an empty list if unavailable."""
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        documents: list[dict[str, Any]] = []
        for i, item in enumerate(dataset):
            if i >= size:
                break
            text = str(item.get("text", "")).strip()
            if len(text) < 100:
                continue
            documents.append({
                "doc_id": f"wikipedia-{i:06d}", "title": str(item.get("title", "Untitled")),
                "text": text[:12000], "source": "wikipedia", "url": "",
                "license": "See Wikimedia dataset terms and article attribution",
                "ingest_ts": "2024-01-01T00:00:00+00:00",
            })
        return documents
    except Exception:
        return []


def main() -> None:
    """Generate the clean corpus and QA files from CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    size = args.size if args.size is not None else int(load_config()["corpus"]["size"])
    docs = try_wikipedia(size, args.seed)
    if not docs:
        raise RuntimeError("Wikipedia download returned no rows; synthetic fallback is disabled.")
    write_jsonl("data/clean/clean_docs.jsonl", docs)
    write_jsonl("data/clean/qa.jsonl", [])
    print(f"Wrote {len(docs)} real Wikimedia documents; no generated QA rows were added.")


if __name__ == "__main__":
    main()
