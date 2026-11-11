"""Trainable calibrated TF-IDF classifier for prompt-injection triage."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import hashlib
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ragdef.defenses.types import Verdict
from ragdef.eval.metrics import classification_metrics
from ragdef.utils import read_jsonl


def _classifier() -> CalibratedClassifierCV:
    """Create a word 1-2 gram TF-IDF model with sigmoid probability calibration.

    Word-level features were chosen over word+character n-grams by the ablation in
    scripts/tfidf_ablation.py: equal held-out accuracy, the lowest false-positive
    rate, and much better recall on an attack source withheld from training.
    Character n-grams mostly memorized each dataset's surface style.
    """
    features = TfidfVectorizer(ngram_range=(1, 2), max_features=24000, min_df=1, sublinear_tf=True)
    estimator = Pipeline([("features", features), ("model", LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42))])
    return CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=3)


class MLScanner:
    """Calibrated ML scanner with train/dev/test-aware dataset loading."""
    _SCORE_CACHE: dict[tuple[str, str], float] = {}

    def __init__(self, t_review: float = 0.35, t_block: float = 0.75) -> None:
        self.t_review = float(t_review)
        self.t_block = float(t_block)
        self.model: CalibratedClassifierCV | None = None
        self.model_key = "untrained"

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> "MLScanner":
        """Fit the calibrated classifier on labeled training text."""
        if len(texts) != len(labels) or len(texts) < 6 or len(set(labels)) < 2:
            raise ValueError("Need matching text/label arrays with at least 6 examples and both labels")
        self.model = _classifier()
        self.model.fit(list(texts), list(map(int, labels)))
        digest = hashlib.sha256(("\n".join(texts) + repr(list(labels))).encode()).hexdigest()
        self.model_key = f"trained:{digest}"
        return self

    def fit_from_files(self, model_path: str | Path | None = None, force: bool = False) -> "MLScanner":
        """Train on train splits only, reusing a previously saved model when available."""
        target = Path(model_path) if model_path else None
        if target and target.exists() and not force:
            self.model = joblib.load(target)
            self.model_key = f"file:{target.resolve()}:{target.stat().st_mtime_ns}"
            return self
        attacks = read_jsonl("data/attacks/attack_set.jsonl")
        negatives = read_jsonl("data/attacks/benign_hard_negatives.jsonl")
        injections = read_jsonl("data/raw/injection_pool.jsonl")
        texts = [str(r.get("full_text", r.get("text", ""))) for r in attacks if r.get("split") == "train"]
        labels = [1] * len(texts)
        neg_texts = [str(r.get("text", "")) for r in negatives if r.get("split") == "train"]
        texts.extend(neg_texts); labels.extend([0] * len(neg_texts))
        for row in injections:
            if int(row.get("label", 1)) == 1:
                texts.append(str(row.get("text", ""))); labels.append(1)
        if len(set(labels)) < 2 or len(texts) < 6:
            raise RuntimeError("Public train/negative data is unavailable; synthetic scanner fallback is disabled.")
        self.fit(texts, labels)
        if target:
            target.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, target)
            self.model_key = f"file:{target.resolve()}:{target.stat().st_mtime_ns}"
        return self

    def score(self, document: str | dict[str, Any]) -> float:
        """Return calibrated injection probability for one document."""
        if self.model is None:
            raise RuntimeError("MLScanner is not fitted; call fit() or fit_from_files() first")
        text = str(document.get("text", "")) if isinstance(document, dict) else str(document)
        text_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
        cache_key = (self.model_key, text_hash)
        if cache_key in self._SCORE_CACHE:
            return self._SCORE_CACHE[cache_key]
        score = float(self.model.predict_proba([text])[0][1])
        if len(self._SCORE_CACHE) > 50000:
            self._SCORE_CACHE.clear()
        self._SCORE_CACHE[cache_key] = score
        return score

    def score_many(self, documents: Sequence[str | dict[str, Any]]) -> list[float]:
        """Score many documents in one vectorizer pass while preserving the cache."""
        if self.model is None:
            raise RuntimeError("MLScanner is not fitted; call fit() or fit_from_files() first")
        texts = [str(item.get("text", "")) if isinstance(item, dict) else str(item)
                 for item in documents]
        hashes = [hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
                  for text in texts]
        keys = [(self.model_key, digest) for digest in hashes]
        scores: list[float | None] = [self._SCORE_CACHE.get(key) for key in keys]
        missing_indices = [index for index, value in enumerate(scores) if value is None]
        if missing_indices:
            predictions = self.model.predict_proba([texts[index] for index in missing_indices])[:, 1]
            for index, prediction in zip(missing_indices, predictions):
                score = float(prediction)
                scores[index] = score
                self._SCORE_CACHE[keys[index]] = score
        if len(self._SCORE_CACHE) > 50000:
            # Keep current results intact; a later call may rebuild evicted cache entries.
            self._SCORE_CACHE.clear()
            for key, score in zip(keys, scores):
                self._SCORE_CACHE[key] = float(score)
        return [float(score) for score in scores]

    def scan(self, document: str | dict[str, Any]) -> Verdict:
        """Emit a verdict with calibrated score and a short explanation."""
        score = self.score(document)
        action = "block" if score >= self.t_block else "quarantine" if score >= self.t_review else "allow"
        flags = ["ml_injection_probability"] if score >= self.t_review else []
        return Verdict(score, flags, [f"calibrated TF-IDF injection probability={score:.3f}"], action)

    def evaluate(self, texts: Sequence[str], labels: Sequence[int], threshold: float = 0.5) -> dict[str, float]:
        """Compute binary classification metrics for a held-out split."""
        scores = self.score_many(texts)
        return classification_metrics(labels, [int(score >= threshold) for score in scores])

    def save(self, path: str | Path = "models/prompt_injection_scanner.joblib") -> None:
        """Persist a fitted model for reproducible reuse."""
        if self.model is None:
            raise RuntimeError("Cannot save an unfitted scanner")
        target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, target)


class TransformerScanner:
    """Optional local-only transformer classifier with graceful ML fallback."""
    def __init__(self, model_name: str | None = None, fallback: MLScanner | None = None) -> None:
        self.fallback = fallback
        self.available = False
        self.tokenizer = None
        self.model = None
        if model_name:
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
                self.model.eval()
                self.available = True
            except Exception:
                self.available = False

    def scan(self, document: str | dict[str, Any]) -> Verdict:
        """Use a local transformer when available; otherwise delegate to fallback."""
        if not self.available:
            if self.fallback is not None:
                return self.fallback.scan(document)
            return Verdict(0.0, ["transformer_unavailable"], ["optional local transformer is unavailable"], "allow")
        import torch
        text = str(document.get("text", "")) if isinstance(document, dict) else str(document)
        inputs = self.tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
            score = float(torch.softmax(logits, dim=-1)[0][-1].item())
        action = "block" if score >= 0.75 else "quarantine" if score >= 0.35 else "allow"
        return Verdict(score, ["transformer_score"], [f"local transformer probability={score:.3f}"], action)
