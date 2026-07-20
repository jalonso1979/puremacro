"""Ahir-Bloom-Furceri World Uncertainty Index.

Counts uncertainty-term mentions per document and normalises by
document length: ``score = (hits / total_words) * 1000``.

Reference
---------
Ahir, H., Bloom, N., Furceri, D. (2022). The World Uncertainty Index.
NBER WP 29763.
"""
from __future__ import annotations

from typing import Iterable

from ..types import RiskIndex
from ..aggregate import index_to_quarterly
from ._lexicons import LEXICONS
from ._kernels import keyword_count_kernel


def wui(
    text_iter: Iterable[tuple],
    *,
    country: str,
    language: str = "en",
    lexicon: frozenset | None = None,
    normalize: str = "zscore",
    base_period: tuple[str, str] | None = None,
    agg: str = "mean",
    with_quality: bool = False,
) -> RiskIndex:
    """Build a length-normalised WUI series from a custom corpus.

    Score is hits per 1000 words per Ahir-Bloom-Furceri.
    """
    terms = lexicon if lexicon is not None else LEXICONS["wui"][language]

    def _kernel(records):
        return keyword_count_kernel(
            records, terms=terms, language=language, length_normalize=True,
        )

    return index_to_quarterly(
        text_iter, kernel=_kernel,
        country=country, language=language,
        name=f"wui_{country.lower()}",
        method="length_normalized_count", corpus="custom",
        normalization=normalize, agg=agg,
        metadata={"index": "wui", "base_period": base_period},
        with_quality=with_quality,
    )


__all__ = ["wui"]
