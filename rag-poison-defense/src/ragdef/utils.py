"""Shared deterministic utilities for data, evaluation, and artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any, Iterable


def seed_everything(seed: int = 42) -> None:
    """Seed Python and available numerical libraries for reproducibility."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def get_logger(name: str = "ragdef", level: int = logging.INFO) -> logging.Logger:
    """Return a consistently formatted console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def sha256_text(text: str) -> str:
    """Return a stable SHA-256 hex digest for UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write dictionaries to JSON Lines, creating parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSON Lines while ignoring blank lines."""
    target = Path(path)
    if not target.exists():
        return []
    with target.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    """Write iterable records to CSV; empty output still creates a valid file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    records = list(rows)
    keys = fieldnames or list(dict.fromkeys(key for record in records for key in record))
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        if keys:
            writer.writeheader()
            writer.writerows(records)
