# Dataset card

## Dataset sources and preprocessing

The default project data are downloaded from the public sources named for Topic 38. There is no synthetic corpus generator or synthetic-data fallback in the default pipeline. Dataset download, schema, or access failures stop the build and are reported to the user.

| Dataset | Source ID | Use | License/access notes |
|---|---|---|---|
| Wikipedia (English, 2023-11-01) | `wikimedia/wikipedia`, config `20231101.en` | Real clean documents; held-out benign false-positive examples | CC BY-SA 3.0 / GFDL; preserve article attribution |
| deepset prompt-injections | `deepset/prompt-injections` | Labeled injection/benign scanner examples | Apache-2.0 according to dataset card |
| BIPIA | `geodesic-research/bipia` public mirror of Microsoft BIPIA | Indirect-injection contexts and paired questions | Mirror card states CC BY-SA 4.0; confirm upstream terms |
| Tensor Trust | `qxcv/tensor-trust`, hijacking-robustness v1 | Human-authored prompt-injection examples | No explicit license file/metadata found in the mirror/repository during the 2026-11-02 audit; do not redistribute without written clarification |
| SQuAD 1.1 | `rajpurkar/squad`, config `plain_text` | Crowd-authored clean questions, answer spans, and source passages for held-out clean QA utility | Hugging Face dataset card declares CC BY-SA 4.0; preserve source attribution |

Exact fetched sample counts and dataset metadata are written to `data/attacks/DATASET_MANIFEST.json`. The default sample limit is 2,000 clean Wikipedia articles, 600 examples per prompt-injection source (deepset samples both train and test splits), and 1,200 SQuAD validation QA pairs. Public source rows retain their source, source record ID, split, and license note. BIPIA `question` and `ideal` values are preserved for the paired-question evaluation. SQuAD QA and contexts are stored separately from scanner training data. Tensor Trust `access_code` is excluded; occurrences in pre/attack/post prompt text are redacted.

## Evaluation labels and splits

- Upstream splits are retained as `upstream_split` metadata, but model evaluation uses a separate deterministic 60/20/20 split of normalized exact-content groups.
- Every duplicate normalized text receives the same split, preventing exact duplicate leakage across train/dev/test.
- Conflicting-label duplicate groups are excluded from scanner scoring and counted in the generated report.
- The combined detector train/dev/test sets are source-aware and must not be interpreted as a uniform population sample; source-level counts and split sizes are included in the manifest.
- `data/attacks/attack_set.jsonl` contains only positive public attack records; `benign_hard_negatives.jsonl` contains deepset negative records and unmodified Wikipedia articles.

## Sensitive content and limitations

Tensor Trust contains user-authored security prompts and benchmark access-code material. Access-code fields are dropped, exact occurrences in saved prompt text are replaced with `[REDACTED_ACCESS_CODE]`, and raw access codes are never logged by the loader. Do not attempt to identify authors or infer real-world credentials. The upstream paper describes a planned permissive release, but the upstream repository has no license file and its Hugging Face card has no license metadata; a 2024 upstream issue asks for a license to be added. Treat redistribution rights as unresolved, keep fetched records local/private, and request written permission or an explicit license from the maintainers before sharing data-derived artifacts containing Tensor Trust text. See `DATA_RIGHTS_AUDIT.md` for sources and the audit date.

The sources differ in task, style, and label definition. Wikipedia may not represent a deployment's benign traffic. BIPIA paired-context retrieval and SQuAD clean-QA utility are separate from scanner evaluation. SQuAD context retrieval and gold-answer evidence are genuine benchmark metrics; answer EM/F1 in this project come from a deterministic extractor simulator, not a real QA model. The attack simulator diagnostic is not ASR and does not estimate a real model's susceptibility. The false-positive cost uses assumptions documented in the report and is not an observed financial cost.
