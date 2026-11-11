"""Public-data ingestion refuses to invent attack or benign rows."""
from __future__ import annotations

import pytest

from ragdef.attacks.build_attack_set import build_attack_set


def test_missing_public_data_fails_without_synthetic_fallback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="synthetic fallback is disabled"):
        build_attack_set()
