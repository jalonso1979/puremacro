"""Sparse document-term matrix with row-aligned document metadata.

:func:`build_dtm` turns texts into a :class:`DocumentTermMatrix` whose ``X`` is
a ``scipy.sparse.csr_matrix`` (documents x terms). Memory grows with the
number of non-zero cells, not with ``n_docs * n_terms``: 50,000 documents x
20,000 terms is 8 GB dense but typically tens of MB sparse.

The matrix is the hand-off point to other tools:

* scikit-learn and scipy take ``dtm.X`` as is;
* :meth:`DocumentTermMatrix.save` writes ``X.mtx`` (Matrix Market),
  ``vocab.txt`` and ``docs.csv``, which R reads with ``Matrix::readMM`` and
  passes to quanteda (``as.dfm``), stm or text2vec;
* :meth:`DocumentTermMatrix.to_tidy` gives the long ``doc, term, value`` frame
  tidytext expects;
* :meth:`DocumentTermMatrix.aggregate` sums rows into a country x month panel
  of term counts that feeds local projections or VARs directly.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Collection, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from .tokenize import RegexTokenizer, Tokenizer, ngrams

_WEIGHTINGS = ("count", "binary", "tfidf")


@dataclass
class DocumentTermMatrix:
    """Documents x terms sparse matrix.

    Attributes
    ----------
    X : scipy.sparse.csr_matrix
        Shape ``(n_docs, n_terms)``.
    vocab : list of str
        Column labels, ``X[:, j]`` is ``vocab[j]``.
    docs : pandas.DataFrame
        One row per row of ``X``, in the same order (``doc_id``, ``date``, ...).
    weighting : {"count", "binary", "tfidf"}
    tokenizer, ngram_range
        Kept so :meth:`transform` tokenizes new texts the same way.
    """

    X: sparse.csr_matrix
    vocab: list[str]
    docs: pd.DataFrame
    weighting: str = "count"
    tokenizer: Any = field(default=None, repr=False)
    ngram_range: tuple[int, int] = (1, 1)

    def __post_init__(self):
        self.X = sparse.csr_matrix(self.X)
        self.vocab = list(self.vocab)
        if self.X.shape != (len(self.docs), len(self.vocab)):
            raise ValueError(
                f"X has shape {self.X.shape} but there are {len(self.docs)} docs "
                f"and {len(self.vocab)} terms"
            )
        if self.weighting not in _WEIGHTINGS:
            raise ValueError(f"weighting must be one of {_WEIGHTINGS}, got {self.weighting!r}")

    # ── basics ─────────────────────────────────────────────────────────────
    @property
    def shape(self) -> tuple[int, int]:
        return self.X.shape

    @property
    def n_docs(self) -> int:
        return self.X.shape[0]

    @property
    def n_terms(self) -> int:
        return self.X.shape[1]

    def __repr__(self) -> str:
        nnz = self.X.nnz
        dens = nnz / max(1, self.n_docs * self.n_terms)
        return (f"DocumentTermMatrix({self.n_docs} docs x {self.n_terms} terms, "
                f"{nnz} non-zero ({dens:.2%}), weighting={self.weighting!r})")

    @property
    def term_index(self) -> dict[str, int]:
        return {t: j for j, t in enumerate(self.vocab)}

    def document_frequency(self) -> pd.Series:
        """Number of documents containing each term."""
        return pd.Series(np.diff(self.X.tocsc().indptr), index=self.vocab, name="df")

    def term_totals(self) -> pd.Series:
        """Column sums (total count of each term when ``weighting="count"``)."""
        return pd.Series(np.asarray(self.X.sum(axis=0)).ravel(), index=self.vocab, name="total")

    def _with(self, **changes) -> "DocumentTermMatrix":
        return replace(self, **changes)

    def _cols(self, keep: np.ndarray) -> "DocumentTermMatrix":
        keep = np.asarray(keep)
        if keep.dtype == bool:
            keep = np.flatnonzero(keep)
        return self._with(X=self.X[:, keep], vocab=[self.vocab[j] for j in keep])

    # ── column selection ───────────────────────────────────────────────────
    def filter(
        self,
        *,
        min_df: int | float | None = None,
        max_df: int | float | None = None,
        max_features: int | None = None,
        terms: Collection[str] | None = None,
        drop_terms: Collection[str] | None = None,
    ) -> "DocumentTermMatrix":
        """Keep a subset of columns.

        ``min_df``/``max_df``: an int is a number of documents, a float in
        (0, 1] a share of documents (scikit-learn's convention).
        ``max_features`` keeps the most frequent terms by column total, ties
        broken alphabetically. ``terms`` keeps only those terms (in the order
        given; unknown terms are ignored); ``drop_terms`` removes terms.
        """
        keep = np.ones(self.n_terms, dtype=bool)
        df = self.document_frequency().to_numpy()
        if min_df is not None:
            keep &= df >= _df_threshold(min_df, self.n_docs)
        if max_df is not None:
            keep &= df <= _df_threshold(max_df, self.n_docs)
        if drop_terms:
            drop = set(drop_terms)
            keep &= np.array([t not in drop for t in self.vocab], dtype=bool)
        idx = np.flatnonzero(keep)
        if max_features is not None and len(idx) > max_features:
            totals = np.asarray(self.X[:, idx].sum(axis=0)).ravel()
            order = sorted(range(len(idx)), key=lambda k: (-totals[k], self.vocab[idx[k]]))
            idx = np.sort(idx[order[:max_features]])
        out = self._cols(idx)
        if terms is not None:
            ti = out.term_index
            out = out._cols(np.array([ti[t] for t in terms if t in ti], dtype=int))
        return out

    # ── weighting ──────────────────────────────────────────────────────────
    def binary(self) -> "DocumentTermMatrix":
        X = self.X.copy()
        X.data = np.ones_like(X.data, dtype=np.int64)
        return self._with(X=X, weighting="binary")

    def tfidf(
        self,
        *,
        sublinear_tf: bool = False,
        smooth_idf: bool = True,
        norm: str | None = "l2",
    ) -> "DocumentTermMatrix":
        """TF-IDF re-weighting of a count matrix.

        Same definition as scikit-learn's ``TfidfTransformer``:
        ``idf = ln((1 + n) / (1 + df)) + 1`` with ``smooth_idf`` (without it,
        ``ln(n / df) + 1``), optional ``1 + ln(tf)``, then row normalisation.
        """
        if self.weighting != "count":
            raise ValueError("tfidf() needs a count matrix; this one is "
                             f"{self.weighting!r}")
        n = self.n_docs
        df = self.document_frequency().to_numpy().astype(float)
        idf = (np.log((1.0 + n) / (1.0 + df)) if smooth_idf
               else np.log(n / np.maximum(df, 1.0))) + 1.0
        X = self.X.astype(np.float64)
        if sublinear_tf:
            X.data = 1.0 + np.log(X.data)
        X = sparse.csr_matrix(X @ sparse.diags(idf))
        X = _normalize_rows(X, norm)
        return self._with(X=X, weighting="tfidf")

    # ── rows ───────────────────────────────────────────────────────────────
    def aggregate(
        self,
        by: str | Sequence[str] | None = None,
        *,
        freq: str | None = None,
        date_col: str = "date",
    ) -> "DocumentTermMatrix":
        """Sum rows within groups of ``docs`` columns and/or calendar periods.

        ``aggregate(by="country", freq="MS")`` returns one row per country and
        month with the summed term counts, plus an ``n_docs`` column in
        ``docs``. ``freq`` is any pandas period alias ("MS", "QS", "YS", "W").
        Only count or binary matrices can be summed meaningfully.
        """
        if self.weighting == "tfidf":
            raise ValueError("aggregate the count matrix, then call tfidf() on the result")
        keys = [by] if isinstance(by, str) else list(by or [])
        frame = self.docs[keys].copy() if keys else pd.DataFrame(index=self.docs.index)
        if freq is not None:
            dates = pd.to_datetime(self.docs[date_col], utc=_has_tz(self.docs[date_col]))
            if getattr(dates.dt, "tz", None) is not None:
                dates = dates.dt.tz_convert("UTC").dt.tz_localize(None)
            frame[date_col] = dates.dt.to_period(_period_alias(freq)).dt.start_time
            keys = keys + [date_col]
        if not keys:
            raise ValueError("give `by`, `freq`, or both")
        frame = frame.reset_index(drop=True)
        codes, uniques = _group_codes(frame, keys)
        G = sparse.csr_matrix(
            (np.ones(len(codes), dtype=np.int64), (codes, np.arange(len(codes)))),
            shape=(len(uniques), len(codes)),
        )
        X = sparse.csr_matrix(G @ self.X)
        groups = uniques.copy()
        groups["n_docs"] = np.bincount(codes, minlength=len(uniques))
        return self._with(X=X, docs=groups.reset_index(drop=True))

    # ── new documents ──────────────────────────────────────────────────────
    def transform(
        self,
        texts: Sequence[str],
        *,
        languages: Sequence[str | None] | str | None = None,
        docs: pd.DataFrame | None = None,
    ) -> "DocumentTermMatrix":
        """Count matrix of new texts on this vocabulary (unseen terms dropped)."""
        return build_dtm(
            texts, languages=languages, docs=docs, tokenizer=self.tokenizer,
            ngram_range=self.ngram_range, vocabulary=self.vocab,
        )

    # ── export ─────────────────────────────────────────────────────────────
    def to_tidy(self, id_col: str | None = "doc_id") -> pd.DataFrame:
        """Long frame with one row per non-zero cell: ``<id_col>, term, value``."""
        coo = self.X.tocoo()
        if id_col is not None and id_col in self.docs.columns:
            ids = self.docs[id_col].to_numpy()[coo.row]
        else:
            id_col, ids = "row", coo.row
        out = pd.DataFrame({
            id_col: ids,
            "term": np.asarray(self.vocab, dtype=object)[coo.col],
            "value": coo.data,
        })
        return out.sort_values([id_col, "term"], kind="stable").reset_index(drop=True)

    def to_frame(self, id_col: str | None = "doc_id") -> pd.DataFrame:
        """Wide frame with pandas sparse columns (cheap; densify deliberately)."""
        index = (self.docs[id_col] if id_col is not None and id_col in self.docs
                 else self.docs.index)
        return pd.DataFrame.sparse.from_spmatrix(self.X, index=index, columns=self.vocab)

    def save(self, directory: str | Path) -> Path:
        """Write ``X.mtx``, ``vocab.txt``, ``docs.csv`` and ``dtm.json``.

        In R: ``X <- Matrix::readMM("X.mtx"); colnames(X) <- readLines("vocab.txt")``.
        """
        from scipy.io import mmwrite

        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        field_ = "integer" if self.weighting in ("count", "binary") else "real"
        mmwrite(str(d / "X.mtx"), sparse.coo_matrix(self.X), field=field_)
        (d / "vocab.txt").write_text("\n".join(self.vocab) + "\n", encoding="utf-8")
        self.docs.to_csv(d / "docs.csv", index=False, encoding="utf-8")
        meta = {"weighting": self.weighting, "ngram_range": list(self.ngram_range),
                "n_docs": self.n_docs, "n_terms": self.n_terms,
                "tokenizer": repr(self.tokenizer)}
        (d / "dtm.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return d

    @classmethod
    def load(cls, directory: str | Path) -> "DocumentTermMatrix":
        """Read a directory written by :meth:`save` (the tokenizer is not restored)."""
        from scipy.io import mmread

        d = Path(directory)
        meta = json.loads((d / "dtm.json").read_text(encoding="utf-8"))
        X = sparse.csr_matrix(mmread(str(d / "X.mtx")))
        vocab = (d / "vocab.txt").read_text(encoding="utf-8").split("\n")[: X.shape[1]]
        docs = pd.read_csv(d / "docs.csv", encoding="utf-8")
        return cls(X=X, vocab=vocab, docs=docs, weighting=meta["weighting"],
                   ngram_range=tuple(meta.get("ngram_range", (1, 1))))


def build_dtm(
    texts: Sequence[str],
    *,
    languages: Sequence[str | None] | str | None = None,
    docs: pd.DataFrame | None = None,
    tokenizer: Tokenizer | None = None,
    ngram_range: tuple[int, int] = (1, 1),
    min_df: int | float = 1,
    max_df: int | float = 1.0,
    max_features: int | None = None,
    vocabulary: Sequence[str] | None = None,
) -> DocumentTermMatrix:
    """Count matrix of ``texts``.

    Parameters
    ----------
    texts : sequence of str
    languages : str or sequence, optional
        Language of each text (or one for all), passed to the tokenizer so
        stop words follow the document's language.
    docs : DataFrame, optional
        Row-aligned metadata; defaults to ``doc_id = 0..n-1``.
    tokenizer : callable, optional
        ``tokenizer(text, language) -> list[str]``; default :class:`RegexTokenizer`.
    ngram_range : (int, int)
        n-grams are formed after stop-word removal.
    min_df, max_df, max_features
        Vocabulary pruning, see :meth:`DocumentTermMatrix.filter`.
    vocabulary : sequence of str, optional
        Fixed columns; terms outside it are not counted and no pruning applies.
    """
    texts = list(texts)
    n = len(texts)
    if isinstance(languages, str) or languages is None:
        langs: Sequence[str | None] = [languages] * n
    else:
        langs = list(languages)
        if len(langs) != n:
            raise ValueError(f"{len(langs)} languages for {n} texts")
    if docs is None:
        docs = pd.DataFrame({"doc_id": np.arange(n)})
    elif len(docs) != n:
        raise ValueError(f"docs has {len(docs)} rows for {n} texts")
    docs = docs.reset_index(drop=True)
    tok = tokenizer if tokenizer is not None else RegexTokenizer()

    fixed = vocabulary is not None
    index: dict[str, int] = ({t: j for j, t in enumerate(vocabulary)} if fixed else {})
    indptr = [0]
    indices: list[int] = []
    data: list[int] = []
    for text, lang in zip(texts, langs):
        counts = Counter(ngrams(tok(text or "", lang), ngram_range))
        for term, c in counts.items():
            j = index.get(term)
            if j is None:
                if fixed:
                    continue
                j = index[term] = len(index)
            indices.append(j)
            data.append(c)
        indptr.append(len(indices))
    X = sparse.csr_matrix(
        (np.asarray(data, dtype=np.int64), np.asarray(indices, dtype=np.int64),
         np.asarray(indptr, dtype=np.int64)),
        shape=(n, len(index)),
    )
    X.sum_duplicates()
    vocab = list(index)
    out = DocumentTermMatrix(X=X, vocab=vocab, docs=docs, tokenizer=tok,
                             ngram_range=tuple(ngram_range))
    if fixed:
        return out
    # Alphabetical columns make the matrix independent of document order.
    out = out._cols(np.argsort(np.asarray(vocab, dtype=object), kind="stable"))
    if min_df != 1 or max_df != 1.0 or max_features is not None:
        out = out.filter(min_df=min_df, max_df=max_df, max_features=max_features)
    return out


def _df_threshold(v: int | float, n: int) -> float:
    # scikit-learn's convention: a float is a share of documents, an int a count.
    if isinstance(v, (float, np.floating)):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"a float document frequency must be in [0, 1], got {v}")
        return float(v) * n
    return float(v)


def _normalize_rows(X: sparse.csr_matrix, norm: str | None) -> sparse.csr_matrix:
    if norm is None:
        return X
    if norm == "l2":
        s = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    elif norm == "l1":
        s = np.asarray(abs(X).sum(axis=1)).ravel()
    else:
        raise ValueError(f"norm must be 'l1', 'l2' or None, got {norm!r}")
    s[s == 0] = 1.0
    return sparse.csr_matrix(sparse.diags(1.0 / s) @ X)


def _has_tz(col: pd.Series) -> bool:
    return any(getattr(v, "tzinfo", None) is not None for v in col)


def _period_alias(freq: str) -> str:
    # Period aliases differ from offset aliases: "MS"/"ME" -> "M", "QS" -> "Q".
    f = freq.upper()
    for suffix in ("S", "E"):
        if len(f) > 1 and f.endswith(suffix) and f[:-1] in ("M", "Q", "Y", "A", "BM", "BQ"):
            f = f[:-1]
    return {"A": "Y", "BM": "M", "BQ": "Q"}.get(f, f)


def _group_codes(frame: pd.DataFrame, keys: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    grouped = frame.groupby(keys, sort=True, dropna=False)
    codes = grouped.ngroup().to_numpy()
    uniques = (frame.assign(_code=codes).drop_duplicates("_code")
               .sort_values("_code")[keys].reset_index(drop=True))
    return codes, uniques


__all__ = ["DocumentTermMatrix", "build_dtm"]
