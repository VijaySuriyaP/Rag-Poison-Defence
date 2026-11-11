"""Empirical baseline sensitivity of ASR to the simulated victim's obey probability."""
from __future__ import annotations

import argparse
import time
from typing import Any

from ragdef.config import load_config
from ragdef.eval.asr import summarize_trials
from ragdef.eval.run_experiments import balanced_sample, target_queries
from ragdef.rag.pipeline import RAGPipeline
from ragdef.rag.victim_llm import SimulatedVictim
from ragdef.utils import read_jsonl, write_csv


def run_sensitivity(seed: int = 42, smoke: bool = False,
                    probabilities: list[float] | None = None) -> list[dict[str, Any]]:
    """Re-run a fixed set of attack trials at each configured obey probability."""
    clean = read_jsonl("data/clean/clean_docs.jsonl")
    if not clean:
        raise RuntimeError("Public Wikipedia corpus missing; run `make data`. Synthetic fallback is disabled.")
    attacks = read_jsonl("data/attacks/attack_set.jsonl")
    selected = balanced_sample(attacks, 12 if smoke else 36, seed)
    if not selected:
        raise RuntimeError("Public attack corpus missing; run `make data`. Synthetic fallback is disabled.")
    rates = probabilities or [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    base = load_config()
    docs = list(clean) + [{**row, "text": row.get("full_text", row.get("payload_text", "")),
                           "source": row.get("source", "public-attack-dataset")} for row in selected]
    rows: list[dict[str, Any]] = []
    encoded_factor = float(base.get("victim", {}).get("encoded_obey_factor", 0.45))
    # The corpus/index do not depend on victim obedience. Build once and swap only the
    # deterministic victim adapter so this sensitivity analysis isolates that parameter.
    pipeline = RAGPipeline(base, SimulatedVictim(rates[0], seed, encoded_factor))
    ingest_start = time.perf_counter()
    pipeline.ingest(docs)
    ingest_ms = (time.perf_counter() - ingest_start) * 1000 / max(1, len(docs))
    for probability in rates:
        pipeline.victim = SimulatedVictim(probability, seed, encoded_factor)
        trials = []
        query_ms = []
        for attack in selected:
            for query in target_queries(attack):
                start = time.perf_counter()
                context = pipeline.retrieve(query)
                answer = pipeline.victim.generate(query, context)
                query_ms.append((time.perf_counter() - start) * 1000)
                trials.append({"retrieved": any(c.get("doc_id") == attack["doc_id"] for c in context),
                               "success": attack["canary"] in answer or "attacker.invalid" in answer})
        summary = summarize_trials(trials, seed, 100 if smoke else 500)
        rows.append({"obey_prob": probability, **summary, "ingest_ms_per_doc": ingest_ms,
                     "query_ms_mean": sum(query_ms) / max(1, len(query_ms))})
    write_csv("results/tables/obey_prob_sensitivity.csv", rows)
    return rows


def main() -> None:
    """Run simulator sensitivity sweep."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    rows = run_sensitivity(args.seed, args.smoke)
    print(f"Wrote {len(rows)} obey-probability sensitivity rows.")


if __name__ == "__main__":
    main()
