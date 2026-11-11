# Progress

- [x] Replace synthetic corpus/attack generation with the Topic 38 public sources.
- [x] Download and normalize Wikimedia, deepset prompt-injections, BIPIA, and Tensor Trust records.
- [x] Preserve dataset source/split metadata and redact Tensor Trust access codes.
- [x] Prepare full-size public-data samples and apply deterministic grouped 60/20/20 deduplication splits.
- [x] Train scanners on train only; select scanner thresholds and the retriever on dev only.
- [x] Implement sealed held-out scanner metrics, per-source confidence intervals, PR curves, retrieval Recall@k/MRR, simulator-only diagnostics, and low/base/high cost sensitivity.
- [x] Add regression coverage for grouped splits, conflict exclusion, pooled/source FPR constraints, undefined single-class metrics, and zero-event intervals.
- [x] Run the full regression suite (34 passed) and Ruff (clean).
- [x] Regenerate final benchmark artifacts after the latest proxy top-5 scoring, query alignment, and confidence-interval fixes.
- [x] Record final output status and remaining limitations below.
- [x] Add a source-held-out BIPIA/Tensor Trust generalization evaluation, withholding all duplicate aliases from fitting/tuning.
- [x] Add held-out clean Wikipedia title-query retrieval and deterministic extractive answer-utility metrics under the defense policies.
- [x] Compare hashing, word TF-IDF, and character TF-IDF on dev and report sealed BIPIA/clean test comparisons without test-driven selection.
- [x] Audit Tensor Trust reuse/redistribution evidence and document the unresolved-license restriction.
- [x] Regenerate the final benchmark report/tables/figures with sealed retriever candidate comparison; run the final full suite.
- [x] Add BM25 and a BM25 + word-TF-IDF reciprocal-rank-fusion hybrid with pipeline regression tests.
- [x] Expand dev-only selection and sealed test comparisons to include BM25 and hybrid retrieval.
- [x] Run the expanded full test suite (37 passed).
- [x] Regenerate the full benchmark outputs with BM25/hybrid candidate evaluation and run final checks.
- [x] Add SQuAD v1.1 validation QA ingestion with crowd-authored questions, gold answer spans, and matched clean passages.
- [x] Add retrieval context Recall@k, gold-answer evidence coverage, and explicitly simulator-only extractive EM/F1 on SQuAD QA.
- [x] Run the full regression suite after SQuAD additions (39 passed).
- [x] Reprepare the public data with SQuAD and regenerate the full report/tables.

Use `make data` and `make all`; download failures are fatal rather than triggering synthetic fallback.



- [x] Remove the explainer video and the local Ollama/Qwen ASR pilot; the simulated victim is the only ASR diagnostic.
- [x] Reduce report graphs from eight to three (attack success, scanner detection, false-positive cost) and fix the previously empty attack-success chart.
- [x] Ablate D2 features (binary BoW, word TF-IDF, char TF-IDF, word+char). Char n-grams overfit source style: BIPIA source-held-out recall 45.8% (word+char) vs 70.1% (word only). Switched D2 to word 1-2-gram TF-IDF and reran the full benchmark (226 s). New D2 test: recall 96.0%, FPR 0.6% (3/505); source-held-out recall BIPIA 70.1%, Tensor Trust 91.7%. Earlier D2 figures in this file refer to the previous word+char model.
- [x] Add README architecture diagram, `notebooks/results.ipynb` (executed) and `reports/technical_report.md`.
