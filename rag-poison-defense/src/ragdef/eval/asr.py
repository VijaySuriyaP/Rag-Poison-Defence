"""Precise end-to-end and retrieval-conditional ASR calculations."""
from __future__ import annotations

from typing import Any

from ragdef.eval.stats import bootstrap_ci


def summarize_trials(trials: list[dict[str, Any]], seed: int = 42,
                     bootstrap_samples: int = 500) -> dict[str, float]:
    """Compute retrieval rate, end-to-end ASR, conditional ASR and CIs.

    Each trial has `retrieved` and `success` booleans. Empty collections return zeroes.
    """
    if not trials:
        return {"n_trials": 0, "retrieval_rate": 0.0, "asr": 0.0,
                "asr_conditional": 0.0, "asr_ci_low": 0.0, "asr_ci_high": 0.0}
    retrieval = [float(bool(t.get("retrieved"))) for t in trials]
    success = [float(bool(t.get("success"))) for t in trials]
    conditional = [float(bool(t.get("success"))) for t in trials if t.get("retrieved")]
    low, high = bootstrap_ci(success, samples=bootstrap_samples, seed=seed)
    return {"n_trials": float(len(trials)), "retrieval_rate": sum(retrieval) / len(retrieval),
            "asr": sum(success) / len(success),
            "asr_conditional": sum(conditional) / len(conditional) if conditional else 0.0,
            "asr_ci_low": low, "asr_ci_high": high}
