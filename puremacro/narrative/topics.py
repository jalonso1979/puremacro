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
from typing import Any, Sequence

import numpy as np
import pandas as pd

# Built-in lightweight stop words for zero-dependency execution
_STOPWORDS_EN = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "also", "may", "said", "one", "two", "well", "will"
}

_STOPWORDS_ES = {
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un",
    "para", "con", "no", "una", "su", "al", "lo", "como", "más", "pero", "sus", "le",
    "ya", "o", "este", "sí", "porque", "esta", "entre", "cuando", "muy", "sin", "sobre",
    "también", "me", "hasta", "hay", "donde", "quien", "desde", "todo", "nos", "durante",
    "todos", "uno", "les", "ni", "contra", "otros", "ese", "eso", "ante", "ellos",
    "e", "esto", "mí", "antes", "algunos", "qué", "unos", "yo", "otro", "otras",
    "otra", "él", "tanto", "esa", "estos", "mucho", "quienes", "nada", "muchos", "cual",
    "sea", "poco", "ella", "estar", "estas", "algunas", "algo", "nosotros", "mi", "mis",
    "tú", "te", "ti", "tu", "tus", "ellas", "nosotras", "vosotros", "vosotras", "os",
    "mío", "mía", "míos", "mías", "tuyo", "tuya", "tuyos", "tuyas", "suyo", "suya",
    "suyos", "suyas", "nuestro", "nuestra", "nuestros", "nuestras", "vuestro", "vuestra",
    "vuestros", "vuestras", "esos", "esas", "estoy", "estás", "está", "estamos",
    "estáis", "están", "esté", "estés", "estemos", "estéis", "estén", "asimismo", "dicho"
}


class TfidfVectorizer:
    """Pure-Python TF-IDF Vectorizer with n-gram support and sublinear TF.

    Parameters
    ----------
    max_features : int, default 1000
        Maximum number of vocabulary terms by document frequency.
    min_df : int, default 2
        Minimum document frequency for a term to be retained.
    stop_words : str or set of str, default 'english'
        Stop words to remove ('english', 'spanish', or a custom set).
    ngram_range : tuple of (int, int), default (1, 1)
        Lower and upper boundary of the range of n-values for n-grams.
    sublinear_tf : bool, default True
        Apply sublinear scaling: 1 + log(tf) if tf > 0.
    """

    def __init__(
        self,
        max_features: int = 1000,
        min_df: int = 2,
        stop_words: str | Sequence[str] | set[str] | None = "english",
        ngram_range: tuple[int, int] = (1, 1),
        sublinear_tf: bool = True,
    ):
        self.max_features = max_features
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        
        if isinstance(stop_words, str):
            if stop_words.lower() == "english":
                self.stop_words_ = _STOPWORDS_EN
            elif stop_words.lower() in ("spanish", "es"):
                self.stop_words_ = _STOPWORDS_ES
            else:
                self.stop_words_ = set()
        elif stop_words is not None:
            self.stop_words_ = set(stop_words)
        else:
            self.stop_words_ = set()

        self.vocabulary_: dict[str, int] = {}
        self.feature_names_: list[str] = []
        self.idf_: np.ndarray = np.array([])

    def _tokenize(self, text: str) -> list[str]:
        words = re.findall(r"\b[a-zA-ZáéíóúÁÉÍÓÚñÑüÜ]{2,}\b", text.lower())
        tokens = [w for w in words if w not in self.stop_words_]
        
        min_n, max_n = self.ngram_range
        if min_n == 1 and max_n == 1:
            return tokens
            
        ngrams = []
        for n in range(min_n, max_n + 1):
            if n == 1:
                ngrams.extend(tokens)
            else:
                for i in range(len(tokens) - n + 1):
                    ngrams.append(" ".join(tokens[i : i + n]))
        return ngrams

    def fit(self, raw_documents: Sequence[str]) -> TfidfVectorizer:
        """Fit vocabulary and IDF weights on raw documents."""
        N = len(raw_documents)
        df_counts: dict[str, int] = {}
        
        for doc in raw_documents:
            tokens = set(self._tokenize(doc))
            for t in tokens:
                df_counts[t] = df_counts.get(t, 0) + 1

        # Filter by min_df
        valid_terms = {t: c for t, c in df_counts.items() if c >= self.min_df}
        
        # Sort by document frequency descending, take top max_features
        sorted_terms = sorted(valid_terms.items(), key=lambda x: (-x[1], x[0]))[: self.max_features]
        
        self.vocabulary_ = {t: idx for idx, (t, _) in enumerate(sorted_terms)}
        self.feature_names_ = [t for t, _ in sorted_terms]
        
        # Smooth IDF: log((1 + N) / (1 + df)) + 1
        df_array = np.array([valid_terms[t] for t in self.feature_names_], dtype=float)
        self.idf_ = np.log((1.0 + N) / (1.0 + df_array)) + 1.0
        return self

    def transform(self, raw_documents: Sequence[str]) -> np.ndarray:
        """Transform raw documents into normalized TF-IDF matrix."""
        if not self.vocabulary_:
            raise RuntimeError("TfidfVectorizer is not fitted.")
            
        N = len(raw_documents)
        V = len(self.feature_names_)
        X = np.zeros((N, V), dtype=float)
        
        for i, doc in enumerate(raw_documents):
            tokens = self._tokenize(doc)
            for t in tokens:
                if t in self.vocabulary_:
                    X[i, self.vocabulary_[t]] += 1.0

        if self.sublinear_tf:
            mask = X > 0
            X[mask] = 1.0 + np.log(X[mask])

        # Multiply by IDF
        X = X * self.idf_[None, :]

        # L2 normalize rows
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    def fit_transform(self, raw_documents: Sequence[str]) -> np.ndarray:
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
        X = np.asarray(X, dtype=float)
        if np.any(X < 0):
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
            loss = np.linalg.norm(X - W @ H)
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
        X = np.asarray(X, dtype=float)
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
