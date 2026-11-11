"""Compatibility entry point for the public-data benchmark report."""
from __future__ import annotations

import argparse
from pathlib import Path


def build_report(results_dir: str | Path = "results", data_dir: str | Path = "data") -> str:
    """Read the report generated from the public-data evaluation; never rebuild synthetic metrics."""
    dataset_manifest = Path(data_dir) / "attacks" / "DATASET_MANIFEST.json"
    report_path = Path("reports/effectiveness_report.md")
    if not dataset_manifest.exists() or not report_path.exists():
        raise RuntimeError("Public dataset report is missing; run `python -m ragdef.eval.public_benchmark`.")
    return report_path.read_text(encoding="utf-8")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    from ragdef.eval.public_benchmark import run
    run()
    print("Generated reports/effectiveness_report.md from public dataset evaluation.")


if __name__ == "__main__":
    main()
