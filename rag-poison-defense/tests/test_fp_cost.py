"""False-positive cost model assumptions and output checks."""
from __future__ import annotations

from ragdef.eval.fp_cost import cost_per_10k


def test_cost_model_reviewer_and_incident_costs_are_explicit():
    costs = cost_per_10k(0.02, 0.1, review_minutes=6, review_hour_cost=30,
                         incident_cost=1000, poison_rate=0.01, asr=0.5)
    assert costs["reviewer_hours_per_10k"] == 100.0
    assert costs["reviews_per_10k"] == 1000.0
    assert costs["expected_incidents_per_10k"] == 50.0
    assert costs["expected_total_cost_per_10k"] == 53000.0


def test_legitimate_block_cost_is_included_in_total():
    costs = cost_per_10k(0.1, 0.0, docs_per_10k=10000, poison_rate=0.0,
                         asr=0.0, legit_block_cost=2.5)
    assert costs["false_positive_block_cost_per_10k"] == 2500.0
    assert costs["expected_total_cost_per_10k"] == 2500.0
