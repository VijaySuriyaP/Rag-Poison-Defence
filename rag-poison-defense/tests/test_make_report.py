"""Legacy report entry point refuses to produce results without public artifacts."""
from __future__ import annotations

import pytest

from ragdef.eval.make_report import build_report


def test_report_requires_public_dataset_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="Public dataset report is missing"):
        build_report(tmp_path / "results", tmp_path / "data")
