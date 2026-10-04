"""Tokenizers: text -> list of terms, language-aware.

A tokenizer is any callable ``tokenizer(text, language) -> list[str]``
(:class:`Tokenizer`). :class:`RegexTokenizer` is the zero-dependency default;
spaCy, NLTK or a subword tokenizer plug in as a plain function with the same
signature, so puremacro does not ship its own lemmatizer.

Unlike the legacy :class:`puremacro.narrative.topics.TfidfVectorizer` regex,
the default keeps numbers and percentages ("50", "2.5%"): in policy text the
number is often the content ("raised the rate by 25 basis points").
"""
from __future__ import annotations

import re
import unicodedata
from typing import Collection, Protocol, Sequence, runtime_checkable

from .stopwords import resolve_stop_words, stopwords

# A number (optionally with decimal or thousands separators and a percent
# sign, "2.5%" or "2,5 %"), or a word: letters in any script, with internal
# apostrophes ("don't", "l'économie" stays one token and is split below).
_TOKEN_RE = re.compile(
    r"\d+(?:[.,]\d+)*(?:\s?%)?"
    r"|[^\W\d_]+(?:'[^\W\d_]+)*"
)
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'"})
# French/Italian elision: l'économie -> économie, dell'inflazione -> inflazione.
_ELISION_RE = re.compile(
    r"^(?:l|d|j|m|n|s|t|c|qu|jusqu|lorsqu|puisqu|dell|nell|all|dall|sull|un)'(?=[^\W\d_])"
)


@runtime_checkable
class Tokenizer(Protocol):
    """Anything callable as ``tokenizer(text, language=None) -> list[str]``."""

    def __call__(self, text: str, language: str | None = None) -> list[str]: ...


class RegexTokenizer:
    """Unicode regex tokenizer with per-language stop words.

    Parameters
    ----------
    lowercase : bool, default True
    numbers : {"keep", "mask", "drop"}, default "keep"
        ``"mask"`` replaces every number with ``"<num>"`` and every percentage
        with ``"<pct>"``, which keeps their position for n-grams ("cut by
        <num> basis points") without one column per value.
    stop_words : "auto", str, collection or None, default "auto"
        ``"auto"`` picks the list from the ``language`` passed with each text
        (see :mod:`puremacro.text.stopwords`); a language name forces one list
        for every text; a collection is used as given; ``None`` keeps all words.
    extra_stop_words : collection of str, optional
        Added to whichever list applies, e.g. boilerplate such as "committee".
    min_length : int, default 2
        Shortest word kept. Numbers are kept whatever their length.
    strip_accents : bool, default False
        Fold "inflación" and "inflacion" together. Off by default because
        accents distinguish words in Spanish and Portuguese ("está"/"esta").
    split_elisions : bool, default True
        Drop French/Italian elided articles ("l'économie" -> "économie").
    default_language : str, default "en"
        Used for stop words when ``stop_words="auto"`` and no language is given.
    """

    def __init__(
        self,
        *,
        lowercase: bool = True,
        numbers: str = "keep",
        stop_words: str | Collection[str] | None = "auto",
        extra_stop_words: Collection[str] = (),
        min_length: int = 2,
        strip_accents: bool = False,
        split_elisions: bool = True,
        default_language: str = "en",
    ):
        if numbers not in ("keep", "mask", "drop"):
            raise ValueError(f"numbers must be 'keep', 'mask' or 'drop', got {numbers!r}")
        self.lowercase = lowercase
        self.numbers = numbers
        self.stop_words = stop_words
        self.extra_stop_words = frozenset(
            w.lower() if lowercase else w for w in extra_stop_words
        )
        self.min_length = int(min_length)
        self.strip_accents = strip_accents
        self.split_elisions = split_elisions
        self.default_language = default_language
        self._fixed_stop = resolve_stop_words(stop_words)

    def __repr__(self) -> str:
        return (
            f"RegexTokenizer(numbers={self.numbers!r}, stop_words={self.stop_words!r}, "
            f"min_length={self.min_length}, strip_accents={self.strip_accents})"
        )

    def stop_words_for(self, language: str | None) -> frozenset[str]:
        base = self._fixed_stop
        if base is None:
            base = stopwords(language or self.default_language)
        if self.strip_accents:
            base = frozenset(_fold(w) for w in base)
        return base | self.extra_stop_words if self.extra_stop_words else base

    def __call__(self, text: str, language: str | None = None) -> list[str]:
        if not text:
            return []
        text = text.translate(_APOSTROPHES)
        if self.lowercase:
            text = text.lower()
        if self.strip_accents:
            text = _fold(text)
        stop = self.stop_words_for(language)
        out: list[str] = []
        for tok in _TOKEN_RE.findall(text):
            if tok[0].isdigit():
                if self.numbers == "drop":
                    continue
                is_pct = tok.endswith("%")
                if self.numbers == "mask":
                    out.append("<pct>" if is_pct else "<num>")
                else:
                    out.append(tok.replace(" ", "") if is_pct else tok)
                continue
            if self.split_elisions and "'" in tok:
                tok = _ELISION_RE.sub("", tok)
            if len(tok) < self.min_length or tok in stop:
                continue
            out.append(tok)
        return out


def _fold(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def ngrams(tokens: Sequence[str], ngram_range: tuple[int, int] = (1, 1)) -> list[str]:
    """All n-grams of ``tokens`` for n in ``ngram_range``, joined by a space."""
    lo, hi = ngram_range
    if lo < 1 or hi < lo:
        raise ValueError(f"invalid ngram_range {ngram_range!r}")
    if lo == hi == 1:
        return list(tokens)
    out: list[str] = []
    for n in range(lo, hi + 1):
        if n == 1:
            out.extend(tokens)
        else:
            out.extend(" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    return out


__all__ = ["Tokenizer", "RegexTokenizer", "ngrams"]
