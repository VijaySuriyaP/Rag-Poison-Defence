"""ROC/PR curve calculations are bounded and monotonic on simple data."""
from __future__ import annotations

from ragdef.eval.scanner_curves import curve_points, select_threshold


def test_curve_points_expected_metrics():
    points = curve_points([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0], [0.5, 0.95])
    assert points[0]["tpr"] == 1.0 and points[0]["fpr"] == 0.0
    assert points[1]["tpr"] == 0.0 and points[1]["fpr"] == 0.0
    assert all(0.0 <= row["precision"] <= 1.0 for row in points)


def test_threshold_selection_uses_dev_labels_and_maximizes_f1():
    """A separated dev sample chooses a score cut without looking at test metrics."""
    selected = select_threshold([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    assert selected["selected_threshold"] == 0.8
    assert selected["dev_f1"] == 1.0
    assert selected["dev_fpr"] == 0.0
