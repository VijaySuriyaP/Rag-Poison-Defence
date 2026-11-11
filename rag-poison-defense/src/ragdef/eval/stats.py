"""Bootstrap uncertainty and paired-comparison helpers."""
from __future__ import annotations

import random
from typing import Callable, Sequence

import numpy as np


def bootstrap_ci(values: Sequence[float], statistic: Callable[[Sequence[float]], float] | None = None,
                 confidence: float = 0.95, samples: int = 500, seed: int = 42) -> tuple[float, float]:
    """Return a percentile bootstrap interval for a statistic (mean by default)."""
    if not values:
        return 0.0, 0.0
    stat = statistic or (lambda xs: sum(xs) / len(xs))
    rng = random.Random(seed)
    estimates: list[float] = []
    data = list(values)
    for _ in range(max(1, samples)):
        sample = [data[rng.randrange(len(data))] for _ in range(len(data))]
        estimates.append(float(stat(sample)))
    estimates.sort()
    alpha = (1.0 - confidence) / 2.0
    low = estimates[min(len(estimates) - 1, int(alpha * len(estimates)))]
    high = estimates[min(len(estimates) - 1, int((1.0 - alpha) * len(estimates)))]
    return low, high


def paired_effect(left: Sequence[float], right: Sequence[float]) -> dict[str, float]:
    """Summarize paired deltas with mean difference and standardized effect."""
    n = min(len(left), len(right))
    if n == 0:
        return {"n": 0, "mean_delta": 0.0, "cohens_d": 0.0}
    deltas = [float(right[i]) - float(left[i]) for i in range(n)]
    mean = sum(deltas) / n
    variance = sum((x - mean) ** 2 for x in deltas) / max(1, n - 1)
    sd = variance ** 0.5
    return {"n": float(n), "mean_delta": mean, "cohens_d": mean / sd if sd else 0.0}


def mcnemar_p_approx(a_only: int, b_only: int) -> float:
    """Return a conservative normal-approximation two-sided McNemar p-value."""
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    z = max(0.0, abs(a_only - b_only) - 1.0) / (discordant ** 0.5)
    # Approximate two-sided normal tail without a scipy dependency.
    import math
    return min(1.0, math.erfc(z / (2.0 ** 0.5)))


def paired_binary_test(left: Sequence[int], right: Sequence[int], seed: int = 42,
                       samples: int = 1000) -> dict[str, float]:
    """Return paired mean delta, bootstrap interval, effect size, and McNemar counts."""
    n = min(len(left), len(right))
    if n == 0:
        return {"n": 0.0, "mean_delta": 0.0, "delta_ci_low": 0.0, "delta_ci_high": 0.0,
                "cohens_d": 0.0, "left_only": 0.0, "right_only": 0.0, "mcnemar_p": 1.0}
    a = [int(bool(v)) for v in left[:n]]
    b = [int(bool(v)) for v in right[:n]]
    deltas = [float(y - x) for x, y in zip(a, b)]
    # For binary paired outcomes each delta is -1, 0, or +1. Sampling category
    # counts via NumPy's multinomial is equivalent to resampling paired rows, but
    # avoids Python loops over n * samples for every comparison.
    negative = sum(value < 0 for value in deltas)
    positive = sum(value > 0 for value in deltas)
    unchanged = n - negative - positive
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n, [negative / n, unchanged / n, positive / n],
                            size=max(1, samples))
    estimates = np.sort((draws[:, 2] - draws[:, 0]) / n)
    alpha = 0.025
    low = float(estimates[min(len(estimates) - 1, int(alpha * len(estimates)))])
    high = float(estimates[min(len(estimates) - 1, int((1.0 - alpha) * len(estimates)))])
    effect = paired_effect(a, b)
    left_only = sum(x == 1 and y == 0 for x, y in zip(a, b))
    right_only = sum(x == 0 and y == 1 for x, y in zip(a, b))
    return {"n": float(n), "mean_delta": effect["mean_delta"], "delta_ci_low": low,
            "delta_ci_high": high, "cohens_d": effect["cohens_d"],
            "left_only": float(left_only), "right_only": float(right_only),
            "mcnemar_p": mcnemar_p_approx(left_only, right_only)}
