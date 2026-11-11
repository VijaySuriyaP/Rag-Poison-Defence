"""Tests for shared artifact helpers."""
from __future__ import annotations

import csv

from ragdef.utils import write_csv


def test_write_csv_preserves_union_of_record_fields(tmp_path):
    """A mixed aggregate/detail output retains fields from every row."""
    path = tmp_path / "mixed.csv"
    write_csv(path, [{"record_type": "overall", "asr": 0.5},
                     {"record_type": "class", "attack_class": "A01", "asr": 0.2}])
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[1]["attack_class"] == "A01"
