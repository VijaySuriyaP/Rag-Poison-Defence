# Project state

## Current implementation

- Default input data comes from Wikimedia Wikipedia, deepset prompt-injections, BIPIA, and Tensor Trust.
- Synthetic corpus, attack generation, and fallback code were removed from the default project pipeline.
- Dataset preparation fails loudly if a required public source cannot be downloaded or normalized.
- The public benchmark groups normalized exact duplicates before deterministic train/dev/test assignment, selects thresholds on dev data under pooled and per-benign-source FPR caps, and reports pooled/per-source metrics with confidence intervals.
- The evaluation includes BIPIA paired-context Recall@k/MRR, a clean-only simulator control, and a clearly labeled simulator diagnostic (not ASR).
- The report also includes source-held-out scanner tests (BIPIA and Tensor Trust) and clean Wikipedia title-query retrieval/extractive utility diagnostics under each ingestion policy.
- The selected BM25 retriever is additionally evaluated on crowd-authored SQuAD validation questions with gold answer spans: paired-context retrieval, gold-span evidence coverage, and separately labeled deterministic-extractor EM/F1.
- Tensor Trust rights were rechecked on 2026-11-02; no explicit license was located, so the source remains local/private pending clarification.
- Graphs and the effectiveness report are generated from current public-data outputs.

## Limitations

- No real-model evaluation was run: no model, credentials, or explicit authorization was supplied. The deterministic victim diagnostic is not an ASR claim.
- Tensor Trust mirror metadata omits a license field. Verify upstream terms before redistributing its records.
- Public datasets vary by task and label definition; no combined metric is representative of all production RAG traffic.

## Reproduce

```bash
python -m pip install -e ".[dev]"
make data
make all
make test
```
