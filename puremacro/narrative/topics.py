"""Pure-Python Topic Modeling & Dynamic Topic Extraction (NMF + TF-IDF).

Provides a zero-compiled-dependency, Pyodide-compatible topic modeling engine
specifically designed for macroeconomic textual corpora (central bank minutes,
Beige Book district summaries, and policy communications).

Features:
- :class:`TfidfVectorizer`: Pure-Python TF-IDF vectorizer with English/Spanish stopwords.
- :class:`NMF`: Non-Negative Matrix Factorization via multiplicative updates.
- :class:`DynamicTopicModel`: Aggregates document-topic shares across dates into time series.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from ..text.stopwords import EN as _STOPWORDS_EN, ES as _STOPWORDS_ES

# The tokenizer this vectorizer has always used: letters only (numbers are
# dropped), at least two characters. Kept as the default so existing topic
# models keep their vocabularies; pass ``tokenizer=puremacro.text.RegexTokenizer()``
# for the number-preserving, multilingual one.
_LEGACY_WORD_RE = re.compile(r"\b[a-zA-ZáéíóúÁÉÍÓÚñÑüÜ]{2,}\b")


class TfidfVectorizer:
    """TF-IDF vectorizer with n-gram support and sublinear TF.

    The matrix is built sparse (:func:`puremacro.text.build_dtm`), so memory
    grows with the number of non-zero cells. ``transform`` returns a dense
    array by default for backward compatibility; pass ``sparse_output=True``
    to get the ``scipy.sparse.csr_matrix`` (what :class:`DynamicTopicModel`
    uses).

    Parameters
    ----------
    max_features : int, default 1000
        Maximum number of vocabulary terms by document frequency.
    min_df : int, default 2
        Minimum document frequency for a term to be retained.
    stop_words : str or set of str, default 'english'
        Stop words to remove ('english', 'spanish', or a custom set). Ignored
        when ``tokenizer`` is given (the tokenizer owns its stop words).
    ngram_range : tuple of (int, int), default (1, 1)
        Lower and upper boundary of the range of n-values for n-grams.
    sublinear_tf : bool, default True
        Apply sublinear scaling: 1 + log(tf) if tf > 0.
    tokenizer : callable, optional
        ``tokenizer(text, language) -> list[str]``, e.g.
        :class:`puremacro.text.RegexTokenizer`. Default: the legacy
        letters-only regex.
    sparse_output : bool, default False
        Return ``scipy.sparse.csr_matrix`` from ``transform``.
    """

    def __init__(
        self,
        max_features: int = 1000,
        min_df: int = 2,
        stop_words: str | Sequence[str] | set[str] | None = "english",
        ngram_range: tuple[int, int] = (1, 1),
        sublinear_tf: bool = True,
        tokenizer: Callable[..., list[str]] | None = None,
        sparse_output: bool = False,
    ):
        self.max_features = max_features
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.tokenizer = tokenizer
        self.sparse_output = sparse_output

        if isinstance(stop_words, str):
            if stop_words.lower() == "english":
                self.stop_words_ = set(_STOPWORDS_EN)
            elif stop_words.lower() in ("spanish", "es"):
                self.stop_words_ = set(_STOPWORDS_ES)
            else:
                self.stop_words_ = set()
        elif stop_words is not None:
            self.stop_words_ = set(stop_words)
        else:
            self.stop_words_ = set()

        self.vocabulary_: dict[str, int] = {}
        self.feature_names_: list[str] = []
        self.idf_: np.ndarray = np.array([])

    def _base_tokens(self, text: str, language: str | None = None) -> list[str]:
        if self.tokenizer is not None:
            return list(self.tokenizer(text, language))
        words = _LEGACY_WORD_RE.findall(text.lower())
        return [w for w in words if w not in self.stop_words_]

    def _tokenize(self, text: str) -> list[str]:
        from ..text.tokenize import ngrams
        return ngrams(self._base_tokens(text), self.ngram_range)

    def _counts(self, raw_documents: Sequence[str], vocabulary=None):
        from ..text.dtm import build_dtm
        return build_dtm(
            list(raw_documents), tokenizer=self._base_tokens,
            ngram_range=self.ngram_range, vocabulary=vocabulary,
        )

    def fit(self, raw_documents: Sequence[str]) -> TfidfVectorizer:
        """Fit vocabulary and IDF weights on raw documents."""
        N = len(raw_documents)
        m = self._counts(raw_documents)
        df = np.diff(m.X.tocsc().indptr)
        valid = [(t, int(c)) for t, c in zip(m.vocab, df) if c >= self.min_df]

        # Sort by document frequency descending, take top max_features
        sorted_terms = sorted(valid, key=lambda x: (-x[1], x[0]))[: self.max_features]

        self.vocabulary_ = {t: idx for idx, (t, _) in enumerate(sorted_terms)}
        self.feature_names_ = [t for t, _ in sorted_terms]

        # Smooth IDF: log((1 + N) / (1 + df)) + 1
        df_array = np.array([c for _, c in sorted_terms], dtype=float)
        self.idf_ = np.log((1.0 + N) / (1.0 + df_array)) + 1.0
        return self

    def transform(self, raw_documents: Sequence[str]) -> np.ndarray | sparse.csr_matrix:
        """Transform raw documents into the L2-normalized TF-IDF matrix."""
        if not self.vocabulary_:
            raise RuntimeError("TfidfVectorizer is not fitted.")

        X = self._counts(raw_documents, vocabulary=self.feature_names_).X.astype(float)
        if self.sublinear_tf:
            X.data = 1.0 + np.log(X.data)
        X = sparse.csr_matrix(X @ sparse.diags(self.idf_))

        # L2 normalize rows
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        X = sparse.csr_matrix(sparse.diags(1.0 / norms) @ X)
        return X if self.sparse_output else X.toarray()

    def fit_transform(self, raw_documents: Sequence[str]) -> np.ndarray | sparse.csr_matrix:
        """Fit and transform raw documents."""
        return self.fit(raw_documents).transform(raw_documents)

    def get_feature_names_out(self) -> list[str]:
        return self.feature_names_


class NMF:
    """Non-Negative Matrix Factorization (Lee & Seung Multiplicative Updates).

    Factorizes non-negative matrix X ≈ W @ H with W >= 0 (document-topic)
    and H >= 0 (topic-term).

    Parameters
    ----------
    n_components : int, default 5
        Number of topics.
    max_iter : int, default 200
        Maximum number of iterations.
    tol : float, default 1e-4
        Tolerance for stopping criteria.
    random_state : int, default 0
        Random seed for initialization.
    """

    def __init__(
        self,
        n_components: int = 5,
        max_iter: int = 200,
        tol: float = 1e-4,
        random_state: int = 0,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.components_: np.ndarray = np.array([])
        self.reconstruction_err_: float = 0.0

    def fit(self, X: np.ndarray) -> NMF:
        """Fit NMF model on matrix X."""
        self.fit_transform(X)
        return self

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit NMF model and return document-topic matrix W."""
        X = _as_float_matrix(X)
        if _has_negative(X):
            raise ValueError("NMF requires non-negative inputs.")
            
        N, V = X.shape
        K = min(self.n_components, V)
        rng = np.random.default_rng(self.random_state)
        
        # Initialize W and H with non-negative random values
        avg = np.sqrt(X.mean() / K)
        W = np.abs(rng.normal(loc=avg, scale=avg * 0.1, size=(N, K)))
        H = np.abs(rng.normal(loc=avg, scale=avg * 0.1, size=(K, V)))
        
        eps = 1e-10
        prev_loss = np.inf
        
        for _ in range(self.max_iter):
            # Update H
            Ht_num = W.T @ X
            Ht_den = W.T @ W @ H + eps
            H *= (Ht_num / Ht_den)
            
            # Update W
            Wt_num = X @ H.T
            Wt_den = W @ H @ H.T + eps
            W *= (Wt_num / Wt_den)
            
            # Check loss convergence: ||X - W H||_F
            loss = _frobenius_residual(X, W, H)
            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

        # Normalize H rows to sum to 1
        H_sums = H.sum(axis=1, keepdims=True)
        H_sums[H_sums == 0] = 1.0
        H = H / H_sums
        
        self.components_ = H
        self.reconstruction_err_ = float(prev_loss)
        return W

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new documents into topic space given fitted H."""
        X = _as_float_matrix(X)
        N = X.shape[0]
        K = self.components_.shape[0]
        rng = np.random.default_rng(self.random_state)
        avg = np.sqrt(X.mean() / K)
        W = np.abs(rng.normal(loc=avg, scale=avg * 0.1, size=(N, K)))
        H = self.components_
        eps = 1e-10
        
        for _ in range(self.max_iter // 2):
            Wt_num = X @ H.T
            Wt_den = W @ H @ H.T + eps
            W *= (Wt_num / Wt_den)
        return W


def _as_float_matrix(X):
    """Dense float array, or a float CSR matrix when ``X`` is sparse."""
    if sparse.issparse(X):
        return sparse.csr_matrix(X, dtype=float)
    return np.asarray(X, dtype=float)


def _has_negative(X) -> bool:
    if sparse.issparse(X):
        return bool(X.nnz) and bool(X.data.min() < 0)
    return bool(np.any(X < 0))


def _frobenius_residual(X, W: np.ndarray, H: np.ndarray) -> float:
    """``||X - W H||_F``; for sparse ``X`` without forming the dense residual."""
    if not sparse.issparse(X):
        return float(np.linalg.norm(X - W @ H))
    # ||X||^2 - 2 tr(W' X H') + tr((W'W)(HH'))
    xx = float(X.multiply(X).sum())
    cross = float(np.sum(np.asarray(W.T @ X) * H))
    wh = float(np.sum((W.T @ W) * (H @ H.T)))
    return float(np.sqrt(max(xx - 2.0 * cross + wh, 0.0)))


@dataclass
class DynamicTopicModelResult:
    """Results from DynamicTopicModel estimation."""
    topic_shares: pd.DataFrame
    topic_keywords: dict[str, list[tuple[str, float]]]
    document_topics: np.ndarray
    feature_names: list[str]
    components: np.ndarray


class DynamicTopicModel:
    """Extracts macroeconomic topics and aggregates them into time series.

    Parameters
    ----------
    n_topics : int, default 5
        Number of latent topics.
    max_features : int, default 1000
        Maximum vocabulary terms.
    stop_words : str or sequence, default 'english'
        Stopwords language ('english' or 'spanish').
    ngram_range : tuple of (int, int), default (1, 2)
        N-gram range.
    random_state : int, default 42
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_topics: int = 5,
        max_features: int = 1000,
        stop_words: str | Sequence[str] = "english",
        ngram_range: tuple[int, int] = (1, 2),
        random_state: int = 42,
    ):
        self.n_topics = n_topics
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=2,
            stop_words=stop_words,
            ngram_range=ngram_range,
            sparse_output=True,
        )
        self.nmf = NMF(
            n_components=n_topics,
            random_state=random_state,
        )

    def fit_transform_corpus(
        self,
        texts: Sequence[str],
        dates: Sequence[Any],
        freq: str = "MS",
    ) -> DynamicTopicModelResult:
        """Fit topic model on dated texts and aggregate into monthly/quarterly series.

        Parameters
        ----------
        texts : sequence of str
            Document texts.
        dates : sequence of datetime-like
            Date for each document.
        freq : str, default 'MS'
            Pandas frequency string for aggregation (e.g. 'MS' for monthly, 'QS' for quarterly).

        Returns
        -------
        DynamicTopicModelResult
        """
        X = self.vectorizer.fit_transform(texts)
        W = self.nmf.fit_transform(X)  # (D, K)
        
        # Row-normalize W so each document's topic distribution sums to 1
        W_norm = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-10)
        
        feature_names = self.vectorizer.get_feature_names_out()
        H = self.nmf.components_
        
        # Build keywords dictionary
        topic_keywords: dict[str, list[tuple[str, float]]] = {}
        for k in range(self.n_topics):
            top_indices = np.argsort(H[k])[::-1][:10]
            topic_name = f"Topic_{k+1}"
            topic_keywords[topic_name] = [
                (feature_names[idx], float(H[k, idx])) for idx in top_indices
            ]

        # Group by date and average topic shares
        dt_index = pd.DatetimeIndex(pd.to_datetime(list(dates)))
        cols = [f"Topic_{k+1}" for k in range(self.n_topics)]
        df_docs = pd.DataFrame(W_norm, index=dt_index, columns=cols)
        
        # Resample to specified frequency
        df_shares = df_docs.resample(freq).mean().dropna(how="all")
        
        return DynamicTopicModelResult(
            topic_shares=df_shares,
            topic_keywords=topic_keywords,
            document_topics=W_norm,
            feature_names=feature_names,
            components=H,
        )


__all__ = [
    "TfidfVectorizer",
    "NMF",
    "DynamicTopicModel",
    "DynamicTopicModelResult",
]
