"""Regenerate evaluation figures from CSV result artifacts."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def plot_baseline(results_dir: str | Path = "results") -> list[str]:
    """Create ASR-by-class and ASR-vs-poison-rate figures if matplotlib is available."""
    root = Path(results_dir)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    baseline = _read_csv(root / "tables" / "baseline_asr.csv")
    if not baseline:
        baseline = [r for p in sorted((root / "raw").glob("baseline_poison_*.csv"))
                    for r in _read_csv(p) if r.get("record_type") == "overall"]
    if not baseline:
        return []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    made: list[str] = []
    rates = sorted(baseline, key=lambda row: float(row.get("poison_rate", 0)))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([float(r["poison_rate"]) * 100 for r in rates], [float(r.get("asr", 0)) for r in rates], marker="o")
    ax.set_xlabel("Poisoned documents (%)")
    ax.set_ylabel("End-to-end ASR")
    ax.set_ylim(0, 1)
    ax.set_title("Baseline ASR vs. poison rate")
    ax.grid(True, alpha=0.25)
    path = figures / "baseline_asr_vs_poison_rate.png"
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig); made.append(str(path))

    by_class: dict[str, list[float]] = {}
    for csv_path in sorted((root / "raw").glob("baseline_poison_*.csv")):
        for row in _read_csv(csv_path):
            if row.get("record_type") == "attack_class" and row.get("attack_class"):
                by_class.setdefault(row["attack_class"], []).append(float(row.get("asr", 0)))
    if by_class:
        names = sorted(by_class)
        means = [sum(by_class[n]) / len(by_class[n]) for n in names]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(names, means)
        ax.set_ylabel("End-to-end ASR")
        ax.set_ylim(0, 1)
        ax.set_title("Baseline ASR by attack class (available poison rates)")
        ax.tick_params(axis="x", labelrotation=65)
        path = figures / "baseline_asr_by_class.png"
        fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig); made.append(str(path))
    return made


def main() -> None:
    """CLI entry point for plotting existing results."""
    made = plot_baseline()
    print("Created figures: " + (", ".join(made) if made else "no data or matplotlib unavailable"))



def plot_supplementary(results_dir: str | Path = "results") -> list[str]:
    """Create final class, defense, ROC/PR, FPR/cost, latency, and sensitivity figures."""
    root = Path(results_dir)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    made: list[str] = []

    def save(fig: Any, filename: str) -> None:
        path = figures / filename
        fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig); made.append(str(path))

    # Baseline versus full defense, aggregated by attack class.
    baseline_by_class: dict[str, list[float]] = {}
    for path in sorted((root / "raw").glob("baseline_poison_*.csv")):
        for row in _read_csv(path):
            if row.get("record_type") == "attack_class" and row.get("attack_class"):
                baseline_by_class.setdefault(row["attack_class"], []).append(float(row.get("asr", 0)))
    defended_by_class: dict[str, float] = {}
    for row in _read_csv(root / "tables" / "asr_by_class_defense.csv"):
        if row.get("defense") == "all_D1_D6":
            defended_by_class[row["attack_class"]] = float(row.get("mean_asr", 0))
    common = sorted(set(baseline_by_class) & set(defended_by_class))
    if common:
        x = list(range(len(common)))
        base_values = [sum(baseline_by_class[c]) / len(baseline_by_class[c]) for c in common]
        defense_values = [defended_by_class[c] for c in common]
        fig, ax = plt.subplots(figsize=(11, 5))
        width = 0.38
        ax.bar([i - width / 2 for i in x], base_values, width, label="Baseline")
        ax.bar([i + width / 2 for i in x], defense_values, width, label="All defenses")
        ax.set_xticks(x, common, rotation=65, ha="right"); ax.set_ylabel("ASR")
        ax.set_ylim(0, 1); ax.set_title("Attack success rate by class: baseline vs. all defenses"); ax.legend()
        save(fig, "asr_by_class_before_after.png")

    by_rate = _read_csv(root / "tables" / "asr_by_defense_poison_rate.csv")
    if by_rate:
        fig, ax = plt.subplots(figsize=(9, 5))
        configs = sorted({row["defense"] for row in by_rate if row.get("defense")})
        for name in configs:
            vals = sorted([r for r in by_rate if r.get("defense") == name], key=lambda r: float(r["poison_rate"]))
            ax.plot([float(r["poison_rate"]) * 100 for r in vals],
                    [float(r["mean_asr_over_thresholds"]) for r in vals], marker="o", label=name)
        ax.set_xlabel("Poisoned documents (%)"); ax.set_ylabel("Mean ASR across threshold sweep")
        ax.set_ylim(0, 1); ax.set_title("ASR versus poison rate by defense configuration")
        ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.25)
        save(fig, "asr_vs_poison_rate_per_defense.png")

    curve_rows = _read_csv(root / "tables" / "scanner_roc_pr.csv")
    if curve_rows:
        fig, ax = plt.subplots(figsize=(6, 5))
        for name in sorted({r["scanner"] for r in curve_rows}):
            vals = sorted([r for r in curve_rows if r["scanner"] == name], key=lambda r: float(r["fpr"]))
            ax.plot([float(r["fpr"]) for r in vals], [float(r["tpr"]) for r in vals], label=name)
        ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
        ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate / recall")
        ax.set_title("Scanner ROC curves"); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend()
        save(fig, "scanner_roc.png")
        fig, ax = plt.subplots(figsize=(6, 5))
        for name in sorted({r["scanner"] for r in curve_rows}):
            vals = sorted([r for r in curve_rows if r["scanner"] == name], key=lambda r: float(r["recall"]))
            ax.plot([float(r["recall"]) for r in vals], [float(r["precision"]) for r in vals], label=name)
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Scanner precision–recall curves")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend()
        save(fig, "scanner_pr.png")

    trade = _read_csv(root / "tables" / "roc_pr_thresholds.csv")
    if trade:
        fig, ax = plt.subplots(figsize=(8, 5))
        for name in sorted({r.get("defense", "") for r in trade if r.get("defense")}):
            vals = [r for r in trade if r.get("defense") == name]
            ax.scatter([float(r.get("hard_negative_fpr", 0) or 0) for r in vals],
                       [float(r.get("asr", 0) or 0) for r in vals], label=name, alpha=0.65)
        ax.set_xlabel("Hard-negative FPR"); ax.set_ylabel("End-to-end ASR")
        ax.set_title("ASR–false-positive trade-off (points span thresholds and poison rates)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=7, ncol=2)
        save(fig, "asr_vs_fpr_tradeoff.png")

    costs = _read_csv(root / "tables" / "fp_cost_by_defense_config.csv")
    if costs:
        names = [r["defense"] for r in costs]
        values = [float(r.get("expected_total_cost_per_10k", 0) or 0) for r in costs]
        fig, ax = plt.subplots(figsize=(10, 5)); ax.bar(names, values)
        ax.set_ylabel("Expected cost per 10,000 documents"); ax.set_title("Modeled operating cost by defense")
        ax.tick_params(axis="x", labelrotation=45); save(fig, "cost_per_10k_docs.png")
        ingest = [float(r.get("mean_ingest_ms_per_doc", 0) or 0) for r in costs]
        query = [float(r.get("mean_query_ms", 0) or 0) for r in costs]
        x = list(range(len(names))); width = 0.38
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar([i - width / 2 for i in x], ingest, width, label="Ingestion ms/doc")
        ax.bar([i + width / 2 for i in x], query, width, label="Query ms")
        ax.set_xticks(x, names, rotation=45, ha="right"); ax.set_ylabel("Milliseconds")
        ax.set_title("Measured latency overhead"); ax.legend()
        save(fig, "latency_overhead.png")

    sensitivity = _read_csv(root / "tables" / "obey_prob_sensitivity.csv")
    if sensitivity:
        vals = sorted(sensitivity, key=lambda r: float(r["obey_prob"]))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([float(r["obey_prob"]) for r in vals], [float(r.get("asr", 0)) for r in vals], marker="o")
        ax.fill_between([float(r["obey_prob"]) for r in vals],
                        [float(r.get("asr_ci_low", 0)) for r in vals],
                        [float(r.get("asr_ci_high", 0)) for r in vals], alpha=0.2)
        ax.set_xlabel("Simulated victim obey probability"); ax.set_ylabel("ASR")
        ax.set_title("Sensitivity of ASR to simulated victim behavior"); ax.set_ylim(0, 1)
        save(fig, "obey_prob_sensitivity.png")
    return made


def plot_all(results_dir: str | Path = "results") -> list[str]:
    """Regenerate every figure supported by available result tables."""
    return plot_baseline(results_dir) + plot_supplementary(results_dir)


def main() -> None:
    """CLI entry point for regenerating baseline and final figures."""
    made = plot_all()
    print("Created figures: " + (", ".join(made) if made else "no data or matplotlib unavailable"))


if __name__ == "__main__":
    main()
