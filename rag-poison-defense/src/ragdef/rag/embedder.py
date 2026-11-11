"""Embedding backends: deterministic hashing by default, optional MiniLM."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


class HashingEmbedder:
    """Offline feature-hashing text encoder with stable dimensions and no fit step."""
    def __init__(self, n_features: int = 2048) -> None:
        self.vectorizer = HashingVectorizer(n_features=n_features, alternate_sign=False,
                                           ngram_range=(1, 2), norm="l2", lowercase=True,
                                           strip_accents="unicode")

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode text strings into normalized dense vectors."""
        if not texts:
            return np.empty((0, self.vectorizer.n_features), dtype=np.float32)
        return self.vectorizer.transform(list(texts)).toarray().astype(np.float32)


class CorpusTfidfEmbedder:
    """Corpus-fitted sparse TF-IDF encoder exposed as dense vectors for NumPy search."""
    def __init__(self, max_features: int = 8192, analyzer: str = "word") -> None:
        if analyzer == "char":
            self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                              max_features=max_features, min_df=1,
                                              sublinear_tf=True)
        else:
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features,
                                              min_df=1, sublinear_tf=True, stop_words="english")
        self.fitted = False

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Fit on the first indexed corpus, then transform retrieval queries with same vocabulary."""
        values = list(texts)
        if not values:
            return np.empty((0, 0), dtype=np.float32)
        if not self.fitted:
            vectors = self.vectorizer.fit_transform(values)
            self.fitted = True
        else:
            vectors = self.vectorizer.transform(values)
        return vectors.toarray().astype(np.float32)


class SentenceTransformerEmbedder:
    """Lazy wrapper around a locally cached sentence-transformers model."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode text with normalized MiniLM embeddings."""
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True), dtype=np.float32)


def get_embedder(name: str = "hashing"):
    """Build the requested embedding backend, falling back offline on error."""
    if name.lower() in {"minilm", "sentence-transformers", "all-minilm-l6-v2"}:
        try:
            return SentenceTransformerEmbedder()
        except Exception:
            return HashingEmbedder()
    if name.lower() in {"tfidf", "corpus-tfidf"}:
        return CorpusTfidfEmbedder()
    if name.lower() in {"tfidf_char", "char-tfidf"}:
        return CorpusTfidfEmbedder(max_features=2048, analyzer="char")
    return HashingEmbedder()
