"""Text-as-data building blocks shared across puremacro.

Not specific to policy narratives: any sequence of strings works. The
narrative corpus (:class:`puremacro.narrative.harvest.NarrativeCorpus`) builds
on it through ``corpus.dtm()`` and ``corpus.segment()``.

* :mod:`~puremacro.text.tokenize`: :class:`RegexTokenizer` (keeps numbers,
  per-language stop words) and the :class:`Tokenizer` protocol that spaCy or
  NLTK wrappers satisfy as plain functions.
* :mod:`~puremacro.text.segment`: paragraph and sentence splitters.
* :mod:`~puremacro.text.dtm`: sparse :class:`DocumentTermMatrix` with TF-IDF,
  grouping into panels, and export to Matrix Market (R), tidy frames and
  scikit-learn.
* :mod:`~puremacro.text.stopwords`: en, es, pt, de, fr, it lists.

Only numpy, scipy and pandas are imported, so the package runs in Pyodide.
"""
from __future__ import annotations

from .dtm import DocumentTermMatrix, build_dtm
from .segment import segment, split_paragraphs, split_sentences
from .stopwords import STOPWORDS, stopwords
from .tokenize import RegexTokenizer, Tokenizer, ngrams

__all__ = [
    "DocumentTermMatrix", "build_dtm",
    "RegexTokenizer", "Tokenizer", "ngrams",
    "segment", "split_paragraphs", "split_sentences",
    "STOPWORDS", "stopwords",
]
