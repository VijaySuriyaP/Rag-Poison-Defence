# Submission index — Topic 38

**Project:** Prompt Injection and Poisoned Documents in RAG Ingestion  
**Data policy:** Public datasets only; no synthetic dataset or fallback.  
**Primary command:** `make all`

## Deliverables

1. **Attack set and clean corpus:** `data/attacks/attack_set.jsonl`, `data/attacks/benign_hard_negatives.jsonl`, and `data/clean/clean_docs.jsonl`. Sources are deepset prompt-injections, BIPIA, Tensor Trust, and Wikimedia Wikipedia. The manifest is `data/attacks/DATASET_MANIFEST.json`.
   - Crowd-authored clean QA and matched passages: `data/clean/squad_qa.jsonl`, `data/clean/squad_contexts.jsonl` (SQuAD validation split).
2. **Defense code:** `src/ragdef/defenses/` includes rule scanning, TF-IDF classification, HMAC provenance, quarantine/review, context hardening, and output guarding.
3. **Effectiveness report:** `reports/technical_report.md` (written 3–5 page technical report, including the D2 TF-IDF design rationale and ablation) and `reports/effectiveness_report.md` (full generated tables).
4. **Results notebook:** `notebooks/results.ipynb`.
5. **Architecture diagram and setup:** `README.md`.

## Evaluation artifacts

- CSV tables: `results/tables/public_*.csv`
- Graphs: `results/figures/public_attack_success.png`, `public_scanner_detection.png`, `public_false_positive_cost.png`
- Scanner model trained from public benchmark splits: `models/prompt_injection_scanner.joblib`
- Dataset card, ethics policy, and threat model: `DATACARD.md`, `ETHICS.md`, `THREAT_MODEL.md`
- Dataset rights audit: `DATA_RIGHTS_AUDIT.md`

The report includes source-aware grouped splits, per-source held-out metrics and confidence intervals, retrieval Recall@k, and cost sensitivity. The deterministic-victim output is a simulator diagnostic, not ASR. No real-model call was made because no model, credentials, or explicit authorization was provided.

## Reproduce

```bash
python -m pip install -e ".[dev]"
make data
make all
python scripts/tfidf_ablation.py
make test
```

Dataset access/download failures stop the pipeline; no generated substitute is used.
