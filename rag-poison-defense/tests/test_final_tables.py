"""Paired comparison calculations used by final experiment tables."""
from __future__ import annotations

from ragdef.eval.stats import paired_binary_test


def test_paired_binary_test_effect_and_mcnemar_counts():
    result = paired_binary_test([1, 1, 0, 1], [0, 0, 0, 1], samples=100)
    assert result["n"] == 4
    assert result["mean_delta"] == -0.5
    assert result["left_only"] == 2 and result["right_only"] == 0
    assert 0.0 <= result["mcnemar_p"] <= 1.0


def test_paired_binary_bootstrap_ci_is_deterministic_and_bounded():
    left = [1, 1, 0, 1, 0, 0]
    right = [0, 0, 0, 1, 1, 0]
    first = paired_binary_test(left, right, seed=7, samples=200)
    second = paired_binary_test(left, right, seed=7, samples=200)
    assert first == second
    assert first["delta_ci_low"] <= first["mean_delta"] <= first["delta_ci_high"]
    assert -1.0 <= first["delta_ci_low"] <= first["delta_ci_high"] <= 1.0
