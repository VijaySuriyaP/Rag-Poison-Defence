"""Download and normalize the public datasets specified for project topic 38.

This command has no generated-data fallback: a missing dataset or network error is
fatal so a run can never silently claim to use public data when it did not.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ragdef.utils import write_jsonl


DATASETS = {
    "wikipedia": {"id": "wikimedia/wikipedia", "config": "20231101.en", "license": "CC-BY-SA-3.0/GFDL"},
    "deepset": {"id": "deepset/prompt-injections", "license": "Apache-2.0"},
    "bipia": {"id": "geodesic-research/bipia", "license": "CC-BY-SA-4.0"},
    "tensor_trust": {"id": "qxcv/tensor-trust", "license": "See upstream dataset terms; no license metadata in mirror card"},
    "squad": {"id": "rajpurkar/squad", "config": "plain_text", "license": "CC-BY-SA-4.0"},
}
STAMP = datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _dataset_rows(dataset_id: str, *, split: str = "train", config: str | None = None,
                  data_files: str | None = None, shuffle_seed: int = 42) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the project data extra: pip install -e '.[data]'") from exc
    kwargs: dict[str, Any] = {"split": split, "streaming": True}
    if config:
        kwargs["name"] = config
    if data_files:
        kwargs["data_files"] = data_files
    stream = load_dataset(dataset_id, **kwargs)
    if hasattr(stream, "shuffle"):
        stream = stream.shuffle(seed=shuffle_seed, buffer_size=10_000)
    yield from stream


def _row_id(source: str, source_id: Any, text: str) -> str:
    value = str(source_id or hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:18])
    return f"{source}-{value}"


def _dedup_group(text: str) -> str:
    """Hash a case/whitespace/punctuation-normalized document for split grouping."""
    normalized = re.sub(r"[^\w]+", " ", text.casefold(), flags=re.UNICODE).strip()
    return hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()


def _group_split(group_id: str) -> str:
    """Assign duplicate groups deterministically to 60/20/20 train/dev/test partitions."""
    bucket = int(group_id[:8], 16) % 100
    return "train" if bucket < 60 else "dev" if bucket < 80 else "test"


def _assign_grouped_splits(rows: list[dict[str, Any]]) -> None:
    """Make every normalized duplicate share a split, retaining upstream split metadata."""
    for row in rows:
        text = str(row.get("text", ""))
        group_id = _dedup_group(text)
        row["upstream_split"] = row.get("split", "")
        row["dedup_group_id"] = group_id
        row["split"] = _group_split(group_id)


def _standard_attack(source: str, row: dict[str, Any], split: str, text: str,
                     *, question: str = "", ideal: str = "", attack_text: str = "") -> dict[str, Any]:
    return {
        "doc_id": _row_id(source, row.get("id", row.get("sample_id")), text),
        "text": text[:20000], "full_text": text[:20000], "payload_text": attack_text[:6000],
        "question": question, "ideal": ideal, "label": 1,
        "attack_class": source, "split": split, "source": source,
        "source_dataset": DATASETS[source]["id"], "source_record_id": str(row.get("id", row.get("sample_id", ""))),
        "claimed_source": "untrusted-public-benchmark", "signature_valid": False,
        "license": DATASETS[source]["license"],
    }


def _collect_wikipedia(size: int, seed: int) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    for row in _dataset_rows(DATASETS["wikipedia"]["id"], config=DATASETS["wikipedia"]["config"], shuffle_seed=seed):
        text, title = str(row.get("text", "")).strip(), str(row.get("title", "Untitled")).strip()
        if len(text) < 100:
            continue
        content_group = _dedup_group(text)
        if content_group in seen_text:
            continue
        seen_text.add(content_group)
        docs.append({
            "doc_id": f"wikipedia-{row.get('id', len(docs))}", "title": title,
            "text": text[:12000], "source": "wikipedia", "source_dataset": DATASETS["wikipedia"]["id"],
            "source_record_id": str(row.get("id", "")), "url": str(row.get("url", "")),
            "license": DATASETS["wikipedia"]["license"], "ingest_ts": STAMP,
            "label": 0, "split": "train" if len(docs) < size * 0.8 else "test",
        })
        if len(docs) >= size:
            break
    if len(docs) < size:
        raise RuntimeError(f"Wikipedia yielded {len(docs)} usable documents; {size} required. No fallback is enabled.")
    return docs


def collect_public_datasets(size: int = 2000, per_attack_source: int = 600,
                            seed: int = 42) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch Wikipedia clean docs and actual attack/negative examples from all three benchmarks."""
    clean = _collect_wikipedia(size, seed)
    attacks: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []

    # deepset provides actual labeled train and test rows; retain the original labels.
    for split in ("train", "test"):
        count = 0
        for row in _dataset_rows(DATASETS["deepset"]["id"], split=split, shuffle_seed=seed):
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            label = int(row.get("label", 0))
            normalized = _standard_attack("deepset", row, split, text, attack_text=text)
            normalized["label"] = label
            if label:
                attacks.append(normalized)
            else:
                negatives.append({**normalized, "source": "deepset-benign", "attack_class": "benign",
                                  "claimed_source": "deepset/prompt-injections", "signature_valid": True})
            count += 1
            if count >= per_attack_source:
                break
        if count == 0:
            raise RuntimeError(f"deepset dataset returned no rows for split={split!r}.")

    # BIPIA rows contain benchmark-provided poisoned context and its actual question.
    bipia_count = 0
    for row in _dataset_rows(DATASETS["bipia"]["id"], shuffle_seed=seed):
        context = str(row.get("context", "")).strip()
        if not context:
            continue
        split = str(row.get("split_name", "train")).lower()
        if split not in {"train", "dev", "test"}:
            split = "train"
        question = str(row.get("question", "")).strip()
        ideal = str(row.get("ideal", "")).strip()
        attacks.append(_standard_attack("bipia", row, split, context, question=question,
                                        ideal=ideal, attack_text=str(row.get("attack_str", ""))))
        bipia_count += 1
        if bipia_count >= per_attack_source:
            break
    if not bipia_count:
        raise RuntimeError("BIPIA yielded no usable poisoned-context rows.")

    # Tensor Trust examples are used only for the attack text; access_code is deliberately
    # omitted and redacted from the surrounding prompt before it is saved.
    tensor_rows = _dataset_rows(
        DATASETS["tensor_trust"]["id"],
        data_files="benchmarks/hijacking-robustness/v1/hijacking_robustness_dataset.jsonl",
        shuffle_seed=seed,
    )
    tensor_count = 0
    for row in tensor_rows:
        attack_text = str(row.get("attack", "")).strip()
        access_code = str(row.get("access_code", ""))
        parts = [str(row.get(key, "")).strip() for key in ("pre_prompt", "attack", "post_prompt")]
        text = "\n\n".join(part for part in parts if part)
        if access_code:
            text = re.sub(re.escape(access_code), "[REDACTED_ACCESS_CODE]", text, flags=re.IGNORECASE)
            attack_text = re.sub(re.escape(access_code), "[REDACTED_ACCESS_CODE]", attack_text, flags=re.IGNORECASE)
        if not attack_text:
            continue
        sample_id = str(row.get("sample_id", tensor_count))
        digest = int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16) % 10
        split = "train" if digest < 6 else "dev" if digest < 8 else "test"
        safe_row = {**row, "access_code": ""}
        attacks.append(_standard_attack("tensor_trust", safe_row, split, text, attack_text=attack_text))
        tensor_count += 1
        if tensor_count >= per_attack_source:
            break
    if not tensor_count:
        raise RuntimeError("Tensor Trust yielded no usable attack rows.")

    # Real, unmodified Wikipedia articles are benign ingestion examples and hard negatives.
    negatives.extend({**doc, "attack_class": "benign-wikipedia", "split": doc["split"]} for doc in clean)
    if not attacks or not negatives:
        raise RuntimeError("Public source data must provide both attacks and benign examples.")
    return clean, attacks, negatives


def collect_squad_clean_qa(max_questions: int = 1200, seed: int = 42):
    """Fetch crowd-authored SQuAD validation QA and its clean Wikipedia paragraphs."""
    questions: list[dict[str, Any]] = []
    contexts: dict[str, dict[str, Any]] = {}
    seen_question_ids: set[str] = set()
    for row in _dataset_rows(DATASETS["squad"]["id"], split="validation",
                            config=DATASETS["squad"]["config"], shuffle_seed=seed):
        context = str(row.get("context", "")).strip()
        question = str(row.get("question", "")).strip()
        raw_answers = row.get("answers", {}) or {}
        answers = raw_answers.get("text", []) if isinstance(raw_answers, dict) else []
        answers = list(dict.fromkeys(str(answer).strip() for answer in answers if str(answer).strip()))
        question_id = str(row.get("id", ""))
        if not context or not question or not answers or not question_id or question_id in seen_question_ids:
            continue
        group_id = _dedup_group(context)
        doc_id = f"squad-{group_id[:20]}"
        contexts.setdefault(group_id, {
            "doc_id": doc_id, "dedup_group_id": group_id, "title": str(row.get("title", "")),
            "text": context[:20000], "source": "squad", "source_dataset": DATASETS["squad"]["id"],
            "source_record_id": doc_id, "license": DATASETS["squad"]["license"],
            "label": 0, "split": "test", "upstream_split": "validation",
        })
        questions.append({
            "qa_id": question_id, "question": question, "gold_answers": answers,
            "gold_answer": answers[0], "gold_doc_id": doc_id, "context_group_id": group_id,
            "source": "SQuAD", "source_dataset": DATASETS["squad"]["id"],
            "license": DATASETS["squad"]["license"], "split": "test",
            "upstream_split": "validation",
        })
        seen_question_ids.add(question_id)
        if len(questions) >= max_questions:
            break
    if len(questions) < max_questions:
        raise RuntimeError(f"SQuAD validation yielded {len(questions)} QA rows; {max_questions} required.")
    return list(contexts.values()), questions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=2000, help="number of Wikimedia clean articles")
    parser.add_argument("--per-attack-source", type=int, default=600,
                        help="maximum examples from each prompt-injection dataset")
    parser.add_argument("--squad-qa-rows", type=int, default=1200,
                        help="number of human-authored SQuAD validation QA rows for clean utility")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.size < 1 or args.per_attack_source < 1 or args.squad_qa_rows < 1:
        parser.error("size, per-attack-source, and squad-qa-rows must be positive")
    clean, attacks, negatives = collect_public_datasets(args.size, args.per_attack_source, args.seed)
    squad_contexts, squad_questions = collect_squad_clean_qa(args.squad_qa_rows, args.seed)
    _assign_grouped_splits(clean)
    _assign_grouped_splits(attacks)
    _assign_grouped_splits(negatives)
    write_jsonl("data/clean/clean_docs.jsonl", clean)
    bipia_qa = [{"qa_id": row["doc_id"], "question": row["question"],
                 "gold_answer": row["ideal"], "gold_doc_id": row["doc_id"],
                 "source": "BIPIA", "split": row["split"]}
                for row in attacks if row["source"] == "bipia" and row.get("question")]
    write_jsonl("data/clean/qa.jsonl", bipia_qa)
    write_jsonl("data/clean/squad_contexts.jsonl", squad_contexts)
    write_jsonl("data/clean/squad_qa.jsonl", squad_questions)
    write_jsonl("data/attacks/attack_set.jsonl", attacks)
    write_jsonl("data/attacks/benign_hard_negatives.jsonl", negatives)
    write_jsonl("data/raw/injection_pool.jsonl", [
        {"text": row.get("payload_text") or row["text"], "label": row["label"],
         "source": row["source"], "split": row["split"]} for row in attacks
    ] + [
        {"text": row["text"], "label": 0, "source": row["source"], "split": row["split"]}
        for row in negatives if row.get("source") == "deepset-benign"
    ])
    counts = Counter(row["source"] for row in attacks)
    negative_counts = Counter(row["source"] for row in negatives)
    manifest = {
        "seed": args.seed, "created_utc": STAMP,
        "dataset_sources": DATASETS,
        "counts": {"wikipedia_clean": len(clean), "attack_examples": len(attacks),
                   "benign_examples": len(negatives), "bipia_questions": len(bipia_qa),
                   "squad_clean_qa_questions": len(squad_questions),
                   "squad_clean_contexts": len(squad_contexts),
                   "attacks_by_source": dict(counts), "benign_by_source": dict(negative_counts)},
        "squad_upstream_splits": {"qa_source_split": "validation",
                                  "sampled_questions": len(squad_questions)},
        "split_rows": {
            "attacks": dict(Counter(row["split"] for row in attacks)),
            "benign": dict(Counter(row["split"] for row in negatives)),
            "wikipedia": dict(Counter(row["split"] for row in clean)),
        },
        "split_policy": "60/20/20 deterministic SHA-256 split by normalized document content; duplicate texts share a split",
        "notes": ["All documents, attacks, and QA questions originate from listed public datasets.",
                  "Tensor Trust access_code fields are excluded and any occurrences in prompt text are redacted.",
                  "SQuAD validation QA and its gold Wikipedia paragraphs are stored separately from scanner training data.",
                  "No local synthetic corpus, generated attack, or synthetic-data fallback is used."],
    }
    Path("data/attacks").mkdir(parents=True, exist_ok=True)
    Path("data/attacks/DATASET_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    attack_readme = [
        "# Public attack-set manifest", "",
        "This file is built exclusively from the public datasets listed below; it does not create attack text.", "",
        f"- Public attack rows: {len(attacks)}", f"- Public benign rows: {len(negatives)}",
         f"- Clean Wikimedia articles: {len(clean)}", f"- Paired BIPIA questions: {len(bipia_qa)}",
         f"- Human-authored SQuAD validation QA rows: {len(squad_questions)}", "",
        "| Dataset source | Attack rows |", "|---|---:|",
    ]
    attack_readme.extend(f"| {source} | {count} |" for source, count in sorted(counts.items()))
    attack_readme += ["", "See `DATASET_MANIFEST.json` for dataset IDs, license notes, and preprocessing details.",
                      "Tensor Trust access codes are omitted and redacted from prompt text.",
                      "No synthetic examples or synthetic fallback are used.", ""]
    Path("data/attacks/ATTACK_SET_README.md").write_text("\n".join(attack_readme), encoding="utf-8")
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
