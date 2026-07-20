"""Labor-Market Uncertainty Index — sentence-level co-occurrence
methodology (Slice 6a).

A document's LUI score is the fraction of sentences that contain
either:
  (i) ≥1 labor-domain term AND ≥1 uncertainty-tone term, OR
  (ii) ≥1 high-precision pre-formed phrase (e.g. "rising unemployment").

This generalizes BBD-EPU's document-level co-occurrence to long
multi-topic documents (FOMC minutes, ECB bulletins) where document-
level co-occurrence is too coarse.

Score is in [0, 1]: "fraction of doc that's labor-uncertainty-flavored".
"""
from __future__ import annotations

from typing import Iterable

from ..types import RiskIndex
from ..aggregate import index_to_quarterly
from ._lexicons import LEXICONS
from ._kernels import sentence_cooccurrence_kernel


def lui(
    text_iter: Iterable[tuple],
    *,
    country: str,
    language: str = "en",
    lexicon: dict | None = None,
    normalize: str = "zscore",
    base_period: tuple[str, str] | None = None,
    agg: str = "mean",
    negation: bool = True,
    with_quality: bool = False,
) -> RiskIndex:
    """Build a labor-market uncertainty series from a custom corpus.

    Parameters
    ----------
    text_iter : iterable of ``(date, text, source_url, metadata)`` records.
    country : ISO3 country tag stamped onto the resulting RiskIndex.
    language : ISO-639-1; selects the default lexicon if ``lexicon=None``.
    lexicon : optional override of the form
        ``{"labor_domain": frozenset, "uncertainty_tone": frozenset,
           "phrases": frozenset | None}``.
    normalize : ``"raw"`` | ``"zscore"`` | ``"bbd_100"``.
    base_period : ``(start_iso, end_iso)`` for normalisation statistics.
    agg : ``"mean"`` (default) | ``"max"`` | ``"dispersion"``.
    """
    lex = lexicon if lexicon is not None else LEXICONS["lui"][language]
    term_groups = [lex["labor_domain"], lex["uncertainty_tone"]]
    phrases = lex.get("phrases")

    def _kernel(records):
        return sentence_cooccurrence_kernel(
            records,
            term_groups=term_groups,
            phrases=phrases,
            language=language,
            negation=negation,
        )

    return index_to_quarterly(
        text_iter, kernel=_kernel,
        country=country, language=language,
        name=f"lui_{country.lower()}",
        method="sentence_cooccurrence", corpus="custom",
        normalization=normalize, agg=agg,
        metadata={"index": "lui", "base_period": base_period},
        with_quality=with_quality,
    )


__all__ = ["lui"]
