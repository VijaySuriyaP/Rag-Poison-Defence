"""Feature ablation for the D2 scanner: why word+character TF-IDF.

Reuses the benchmark's grouped train/dev/test splits, dev-only threshold rule
(FPR <= 5% overall and per benign source) and sealed test set. Each variant is
the same calibrated, class-balanced logistic regression; only the text
features change. Also measures the BIPIA source-held-out setting, scoring
latency, and the highest-weight n-grams of the shipped model.

Run from the repository root:  python scripts/tfidf_ablation.py
Writes results/tables/tfidf_ablation.csv and results/tables/tfidf_top_features.csv.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from sklearn.pipeline import FeatureUnion, Pipeline  # noqa: E402

from ragdef.eval import public_benchmark as pb  # noqa: E402
from ragdef.utils import read_jsonl, write_csv  # noqa: E402


def word_tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), max_features=24000, sublinear_tf=True)


def char_tfidf():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=32000, sublinear_tf=True)


VARIANTS = {
    "binary bag-of-words (no TF-IDF)": lambda: CountVectorizer(ngram_range=(1, 2), max_features=24000,
                                                               binary=True),
    "word TF-IDF only (shipped D2)": word_tfidf,
    "char TF-IDF only (3-5 grams)": char_tfidf,
    "word + char TF-IDF (previous D2)": lambda: FeatureUnion([("word", word_tfidf()), ("char", char_tfidf())]),
}


def build(features):
    estimator = Pipeline([("features", features), ("model", LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42))])
    return CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=3)


def sources(rows):
    return [row.get("evaluation_sources", [pb._source_family(row)]) for row in rows]


def evaluate(make_features, train, dev, test):
    model = build(make_features())
    start = time.perf_counter()
    model.fit([r["text"] for r in train], [int(r["label"]) for r in train])
    fit_seconds = time.perf_counter() - start
    dev_scores = list(model.predict_proba([r["text"] for r in dev])[:, 1])
    info = pb._select_threshold([int(r["label"]) for r in dev], dev_scores, sources=sources(dev))
    texts = [r["text"] for r in test]
    start = time.perf_counter()
    scores = list(model.predict_proba(texts)[:, 1])
    ms_per_doc = (time.perf_counter() - start) * 1000 / len(texts)
    labels = [int(r["label"]) for r in test]
    tp = sum(s >= info["threshold"] and y == 1 for s, y in zip(scores, labels))
    fp = sum(s >= info["threshold"] and y == 0 for s, y in zip(scores, labels))
    pos, neg = sum(labels), len(labels) - sum(labels)
    deepset = [i for i, r in enumerate(test) if "deepset" in sources([r])[0] and labels[i] == 1]
    deepset_tp = sum(scores[i] >= info["threshold"] for i in deepset)
    return {
        "recall": tp / pos, "fpr": fp / neg, "average_precision": average_precision_score(labels, scores),
        "tp": tp, "fp": fp, "n_attack": pos, "n_benign": neg,
        "deepset_recall": deepset_tp / max(1, len(deepset)), "dev_threshold": info["threshold"],
        "fit_seconds": fit_seconds, "ms_per_doc": ms_per_doc,
    }


def main() -> None:
    examples, _ = pb._unique_classifier_rows(read_jsonl(ROOT / "data/attacks/attack_set.jsonl"),
                                             read_jsonl(ROOT / "data/attacks/benign_hard_negatives.jsonl"))
    split = {name: [r for r in examples if r.get("split") == name] for name in ("train", "dev", "test")}

    heldout_ids = {pb._group_id(r) for r in examples
                   if int(r["label"]) == 1 and "bipia" in sources([r])[0]}
    remaining = [r for r in examples if pb._group_id(r) not in heldout_ids]
    holdout_train = [r for r in remaining if r.get("split") == "train"]
    holdout_dev = [r for r in remaining if r.get("split") == "dev"]
    holdout_test = ([r for r in examples if pb._group_id(r) in heldout_ids]
                    + [r for r in examples if int(r["label"]) == 0 and r.get("split") == "test"])

    rows = []
    for name, make_features in VARIANTS.items():
        random_split = evaluate(make_features, split["train"], split["dev"], split["test"])
        holdout = evaluate(make_features, holdout_train, holdout_dev, holdout_test)
        rows.append({
            "variant": name,
            "test_recall": round(random_split["recall"], 4),
            "test_fpr": round(random_split["fpr"], 4),
            "test_average_precision": round(random_split["average_precision"], 4),
            "test_tp_fp": f"{random_split['tp']}/{random_split['fp']}",
            "deepset_recall": round(random_split["deepset_recall"], 4),
            "bipia_heldout_recall": round(holdout["recall"], 4),
            "bipia_heldout_fpr": round(holdout["fpr"], 4),
            "fit_seconds": round(random_split["fit_seconds"], 2),
            "ms_per_doc": round(random_split["ms_per_doc"], 3),
        })
        print(rows[-1])
    write_csv(ROOT / "results/tables/tfidf_ablation.csv", rows)

    # Highest-weight injection n-grams of the shipped feature set (averaged over calibration folds).
    model = build(VARIANTS["word TF-IDF only (shipped D2)"]())
    model.fit([r["text"] for r in split["train"]], [int(r["label"]) for r in split["train"]])
    weights: dict[str, float] = {}
    for calibrated in model.calibrated_classifiers_:
        pipe = calibrated.estimator
        names = pipe.named_steps["features"].get_feature_names_out()
        for feature, coef in zip(names, pipe.named_steps["model"].coef_[0]):
            weights[feature] = weights.get(feature, 0.0) + coef / len(model.calibrated_classifiers_)
    top = sorted(weights.items(), key=lambda item: item[1], reverse=True)
    write_csv(ROOT / "results/tables/tfidf_top_features.csv",
              [{"direction": "injection", "feature": f, "weight": round(w, 3)} for f, w in top[:25]]
              + [{"direction": "benign", "feature": f, "weight": round(w, 3)} for f, w in top[-15:]])
    print("top injection features:", [f for f, _ in top[:25]])


if __name__ == "__main__":
    main()
