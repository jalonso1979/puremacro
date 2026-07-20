"""Per-document scoring kernels for the indices layer.

Slice 2 introduced three kernels: ``keyword_count_kernel``,
``cooccurrence_kernel``, ``tone_kernel``.

Slice 6a adds:
  - ``_split_sentences(text, language)`` — language-aware splitter.
  - ``sentence_cooccurrence_kernel`` — score = fraction of sentences that
    contain ≥1 term from each group OR ≥1 curated phrase. Designed for
    long multi-topic documents (e.g. FOMC minutes) where document-level
    co-occurrence is too coarse.
  - ``length_normalize`` parameter on ``keyword_count_kernel`` —
    Ahir-Bloom-Furceri WUI normalization (hits / total_words × 1000).
"""
from __future__ import annotations

import re
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from ..types import VALID_RISKINDEX_NORMALIZATION as _VALID_NORMALIZATIONS

_NEEDS_SUBSTRING = {"ja", "zh"}
_TOKEN_RX = re.compile(r"\w+", flags=re.UNICODE)
_SENT_LATIN_RX = re.compile(r"[.!?]+(?:\s+|$)")
_SENT_CJK_RX = re.compile(r"[。！？]+")
_PARA_LATIN_RX = re.compile(r"\n\s*\n+")
_PARA_CJK_RX = re.compile(r"(?:\n\s*\n+)|　{2,}")

# Per-language negation markers. A keyword hit is suppressed if any of
# these tokens appears within ``_NEG_WINDOW`` tokens preceding the hit.
# Coverage: English + the seven other supported languages. CJK uses
# substring matching (no word boundaries available); Latin scripts use
# whole-word matching to avoid e.g. "noted" → "no".
_NEG_WINDOW = 4
_NEG_MARKERS: dict[str, frozenset[str]] = {
    "en": frozenset({
        "not", "no", "never", "neither", "nor", "without",
        "hardly", "barely", "scarcely", "rarely", "seldom",
        "doesn't", "doesnt", "don't", "dont",
        "isn't", "isnt", "aren't", "arent", "wasn't", "wasnt", "weren't", "werent",
        "won't", "wont", "wouldn't", "wouldnt", "shouldn't", "shouldnt",
        "couldn't", "couldnt", "can't", "cant", "cannot",
        "fail", "fails", "failed", "failing",
    }),
    "es": frozenset({
        "no", "nunca", "jamás", "ni", "sin", "tampoco",
        "apenas", "difícilmente", "raramente",
        "deja", "deje", "dejó", "dejan",
        "fracas", "fracasa", "fracasó",
    }),
    "pt": frozenset({
        "não", "nao", "nunca", "jamais", "nem", "sem", "tampouco",
        "mal", "raramente", "dificilmente",
        "deixa", "deixe", "deixou",
        "fracass", "fracassa", "fracassou",
    }),
    "de": frozenset({
        "nicht", "kein", "keine", "keiner", "keinen", "keinem", "keines",
        "nie", "niemals", "weder", "noch", "ohne",
        "kaum", "schwerlich",
    }),
    "fr": frozenset({
        "pas", "ne", "non", "ni", "sans", "jamais", "aucun", "aucune",
        "guère", "peu",
    }),
    "it": frozenset({
        "non", "no", "né", "ne", "mai", "senza", "nessun", "nessuno",
        "appena", "raramente",
    }),
    "ja": frozenset({
        "ない", "なく", "なかっ", "ません", "ぬ", "ぬる",
    }),
    "zh": frozenset({
        "不", "没", "无", "未", "非", "莫", "勿",
    }),
}


def _normalize_text_for_match(text: str, language: str) -> str:
    if language in _NEEDS_SUBSTRING:
        return text
    return text.lower()


def _is_negated(norm: str, hit_start: int, language: str) -> bool:
    """Return True iff a negation marker appears within ``_NEG_WINDOW``
    tokens preceding the match starting at ``hit_start`` in ``norm``."""
    markers = _NEG_MARKERS.get(language)
    if not markers:
        return False
    if language in _NEEDS_SUBSTRING:
        # CJK: scan the ~30 chars before the hit for a substring marker.
        window = norm[max(0, hit_start - 30):hit_start]
        return any(m in window for m in markers)
    # Latin: take the preceding chunk, tokenize, take last N tokens.
    window = norm[max(0, hit_start - 80):hit_start]
    tokens = _TOKEN_RX.findall(window)
    if not tokens:
        return False
    return any(t in markers for t in tokens[-_NEG_WINDOW:])


def count_keywords(
    text: str,
    terms: frozenset[str],
    language: str = "en",
    *,
    negation: bool = False,
) -> int:
    """Count word-level matches of ``terms`` in ``text``.

    When ``negation=True``, a hit is suppressed if a negation marker
    appears within ``_NEG_WINDOW`` (4) tokens before it (e.g. "*not*
    uncertain" → 0 instead of +1). Negation markers are per-language
    (see ``_NEG_MARKERS``). For CJK, the preceding ~30-character window
    is scanned for substring markers.
    """
    if not text:
        return 0
    norm = _normalize_text_for_match(text, language)
    if not negation:
        if language in _NEEDS_SUBSTRING:
            return sum(norm.count(t) for t in terms)
        n = 0
        for t in terms:
            rx = r"\b" + re.escape(t.lower()) + r"\b"
            n += len(re.findall(rx, norm))
        return n
    # Negation-aware count: iterate matches and check the preceding window.
    n = 0
    if language in _NEEDS_SUBSTRING:
        for t in terms:
            start = 0
            while True:
                i = norm.find(t, start)
                if i < 0:
                    break
                if not _is_negated(norm, i, language):
                    n += 1
                start = i + len(t)
        return n
    for t in terms:
        rx = r"\b" + re.escape(t.lower()) + r"\b"
        for m in re.finditer(rx, norm):
            if not _is_negated(norm, m.start(), language):
                n += 1
    return n


def _split_sentences(text: str, language: str) -> list[str]:
    """Split text into sentences. Latin scripts use [.!?]+ boundary;
    CJK uses [。！？]+. Empty strings are stripped from the result;
    a doc with no punctuation is one sentence (or empty)."""
    if not text:
        return []
    if language in _NEEDS_SUBSTRING:
        parts = _SENT_CJK_RX.split(text)
    else:
        parts = _SENT_LATIN_RX.split(text)
    return [p.strip() for p in parts if p.strip()]


def _split_paragraphs(text: str, language: str) -> list[str]:
    """Split text into paragraphs. Latin scripts use blank-line boundary;
    CJK additionally accepts two-or-more full-width spaces. Empty/whitespace
    fragments are stripped. A text with no paragraph break returns a
    single-element list."""
    if not text or not text.strip():
        return []
    rx = _PARA_CJK_RX if language in _NEEDS_SUBSTRING else _PARA_LATIN_RX
    parts = rx.split(text)
    return [p.strip() for p in parts if p.strip()]


def keyword_count_kernel(
    records: Iterable[tuple],
    *,
    terms: frozenset[str],
    language: str = "en",
    length_normalize: bool = False,
    negation: bool = False,
) -> Iterator[tuple]:
    """Yield ``(date, score)`` per record.

    If ``length_normalize=False`` (default): score = total keyword hits.
    If ``length_normalize=True``: score = (hits / total_words) * 1000
        (Ahir-Bloom-Furceri WUI normalization). Empty docs → 0.0.
    If ``negation=True``: hits within ``_NEG_WINDOW`` tokens after a
        per-language negation marker are suppressed.
    """
    for record in records:
        if len(record) == 4:
            date, text, _, meta = record
            lang = (meta or {}).get("language", language)
        else:
            date, text, _ = record
            lang = language
        hits = count_keywords(text, terms, language=lang, negation=negation)
        if not length_normalize:
            yield (pd.Timestamp(date), float(hits))
            continue
        total = len(_TOKEN_RX.findall(text or ""))
        if total == 0:
            yield (pd.Timestamp(date), 0.0)
        else:
            yield (pd.Timestamp(date), float(hits) / total * 1000.0)


def cooccurrence_kernel(
    records: Iterable[tuple],
    *,
    term_groups: list[frozenset[str]],
    language: str = "en",
    negation: bool = False,
) -> Iterator[tuple]:
    """Yield ``(date, 1.0)`` if the document contains ≥1 term from EVERY
    group, else ``(date, 0.0)``. Used by BBD-EPU."""
    for record in records:
        if len(record) == 4:
            date, text, _, meta = record
            lang = (meta or {}).get("language", language)
        else:
            date, text, _ = record
            lang = language
        score = 1.0 if all(
            count_keywords(text, group, language=lang, negation=negation) > 0
            for group in term_groups
        ) else 0.0
        yield (pd.Timestamp(date), score)


def sentence_cooccurrence_kernel(
    records: Iterable[tuple],
    *,
    term_groups: list[frozenset[str]],
    phrases: frozenset[str] | None = None,
    language: str = "en",
    negation: bool = False,
) -> Iterator[tuple]:
    """Yield ``(date, fraction-of-sentences-matching)`` per record.

    A sentence "matches" iff:
      (∀ group in term_groups: count_keywords(s, group) > 0)
      OR
      (phrases is not None AND count_keywords(s, phrases) > 0)

    Score = matched_sentences / total_sentences ∈ [0, 1].
    Empty docs and docs with zero parseable sentences yield 0.0.

    Implementation note (slice b): the multi-group co-occurrence path
    delegates to ``triple_cooccurrence_kernel``; the phrase-shortcut
    path is handled here because phrases are an LUI-only feature.
    """
    if phrases is None:
        yield from triple_cooccurrence_kernel(
            records,
            term_groups=term_groups,
            weight_group_idx=None,
            window="sentence",
            language=language,
            negation=negation,
        )
        return

    for record in records:
        if len(record) == 4:
            date, text, _, meta = record
            lang = (meta or {}).get("language", language)
        else:
            date, text, _ = record
            lang = language
        sentences = _split_sentences(text or "", lang)
        if not sentences:
            yield (pd.Timestamp(date), 0.0)
            continue
        matched = 0
        for s in sentences:
            cooc = all(
                count_keywords(s, group, language=lang, negation=negation) > 0
                for group in term_groups
            )
            phrase_hit = count_keywords(s, phrases, language=lang, negation=negation) > 0
            if cooc or phrase_hit:
                matched += 1
        yield (pd.Timestamp(date), float(matched) / len(sentences))


def triple_cooccurrence_kernel(
    records: Iterable[tuple],
    *,
    term_groups: list[frozenset[str]],
    weight_group_idx: int | None = None,
    window: str = "paragraph",
    language: str = "en",
    negation: bool = False,
) -> Iterator[tuple]:
    """Generalised K-group co-occurrence kernel.

    For each window (sentence or paragraph) the score is:

        1[all K groups have ≥1 hit] · (1 + density(group_w))

    where ``density(group_w) = hits_in_window / max(1, n_tokens_in_window)``
    when ``weight_group_idx`` is not None, else density factor is 1.0
    (the kernel reduces to the matched-fraction-of-windows metric).

    Doc-level score = sum(window_scores) / total_windows ∈ [0, ∞).
    Empty docs and docs with zero parseable windows yield 0.0.

    ``window`` accepts ``"sentence"`` | ``"paragraph"``. When K=2 and
    ``weight_group_idx=None``, results match ``sentence_cooccurrence_kernel``
    (with ``phrases=None``) on the same input.
    """
    if window not in ("sentence", "paragraph"):
        raise ValueError(f"window must be 'sentence' or 'paragraph', got {window!r}")
    if weight_group_idx is not None and not (0 <= weight_group_idx < len(term_groups)):
        raise ValueError(
            f"weight_group_idx={weight_group_idx} out of range for "
            f"{len(term_groups)} term_groups"
        )

    for record in records:
        if len(record) == 4:
            date, text, _, meta = record
            lang = (meta or {}).get("language", language)
        else:
            date, text, _ = record
            lang = language

        splitter = _split_sentences if window == "sentence" else _split_paragraphs
        windows = splitter(text or "", lang)
        if not windows:
            yield (pd.Timestamp(date), 0.0)
            continue

        total = 0.0
        for w in windows:
            cooc = all(
                count_keywords(w, group, language=lang, negation=negation) > 0
                for group in term_groups
            )
            if not cooc:
                continue
            if weight_group_idx is None:
                total += 1.0
            else:
                n_tokens = max(1, len(_TOKEN_RX.findall(w)))
                density = count_keywords(
                    w, term_groups[weight_group_idx], language=lang,
                    negation=negation,
                ) / n_tokens
                total += 1.0 + density
        yield (pd.Timestamp(date), float(total) / len(windows))


def tone_kernel(
    records: Iterable[tuple],
    *,
    hawkish_terms: frozenset[str],
    dovish_terms: frozenset[str],
    language: str = "en",
    negation: bool = False,
) -> Iterator[tuple]:
    """Yield ``(date, net_tone)`` per doc.

    ``net_tone = (h - d) / (h + d)`` for documents with at least one hit;
    ``0.0`` if neither lexicon matches. Per-doc score in ``[-1, +1]``.
    """
    for record in records:
        if len(record) == 4:
            date, text, _, meta = record
            lang = (meta or {}).get("language", language)
        else:
            date, text, _ = record
            lang = language
        h = count_keywords(text, hawkish_terms, language=lang, negation=negation)
        d = count_keywords(text, dovish_terms, language=lang, negation=negation)
        total = h + d
        score = 0.0 if total == 0 else (h - d) / total
        yield (pd.Timestamp(date), float(score))


def normalize_series(
    series: pd.Series,
    normalization: str,
    *,
    base_period: tuple[str, str] | None = None,
) -> pd.Series:
    if normalization not in _VALID_NORMALIZATIONS:
        raise ValueError(
            f"normalization {normalization!r} not in {_VALID_NORMALIZATIONS}"
        )
    if normalization == "raw":
        return series.copy()

    if base_period is None:
        ref = series.dropna()
    else:
        start, end = base_period
        ref = series.loc[start:end].dropna()

    if ref.empty:
        return series.copy()

    mu = float(ref.mean())
    sigma = float(ref.std(ddof=0))
    if sigma == 0.0:
        sigma = 1.0  # avoid divide-by-zero on degenerate base periods

    z = (series - mu) / sigma
    if normalization == "zscore":
        return z
    # bbd_100: target mean 100, std 50
    return 100.0 + 50.0 * z


__all__ = [
    "count_keywords",
    "_split_sentences",
    "_split_paragraphs",
    "keyword_count_kernel",
    "cooccurrence_kernel",
    "sentence_cooccurrence_kernel",
    "triple_cooccurrence_kernel",
    "tone_kernel",
    "normalize_series",
]
