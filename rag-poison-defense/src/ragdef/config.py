"""Configuration loading with a safe default for minimal environments."""
from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "corpus": {"size": 2000, "chunk_size": 480, "chunk_overlap": 80},
    "retrieval": {"top_k": 5, "embedder": "tfidf"},
    "victim": {"mode": "simulated", "obey_prob": 0.7, "encoded_obey_factor": 0.45, "seed": 42},
    "poison_rates": [0.001, 0.005, 0.01, 0.05],
    "scanner": {"t_review": 0.35, "t_block": 0.75, "threshold_t0": 0.8, "threshold_t1": 0.7, "threshold_t2": 0.55, "threshold_t3": 0.35},
    "provenance": {"hmac_env": "RAGDEF_HMAC_KEY", "default_key": "local-research-demo-key-not-for-production", "allowed_domains": ["wikipedia.org", "example.org"]},
    "review": {"minutes_per_doc": 4.0, "accuracy": 0.92},
    "cost": {"cost_per_review_hour": 45.0, "cost_per_incident": 2500.0, "docs_per_day": 10000},
}


def load_config(path: str | Path = "configs/default.yaml", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load YAML config and apply top-level overrides; use defaults if absent."""
    config: dict[str, Any] = dict(DEFAULT_CONFIG)
    config = {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v) for k, v in config.items()}
    file_path = Path(path)
    if file_path.exists():
        try:
            import yaml
            loaded = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
            for key, value in loaded.items():
                if isinstance(value, dict) and isinstance(config.get(key), dict):
                    config[key].update(value)
                else:
                    config[key] = value
        except ImportError:
            pass
    if overrides:
        config.update(overrides)
    return config
