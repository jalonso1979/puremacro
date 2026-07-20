"""Husted-Rogers-Sun monetary-policy uncertainty index.

Counts documents containing monetary-policy-uncertainty terms,
aggregates per-quarter, optionally normalises (default ``zscore``).

Reference
---------
Husted, L., Rogers, J., Sun, B. (2020). Monetary policy uncertainty.
J. Monetary Economics 115, 20-36.
"""
from __future__ import annotations

from typing import Iterable

from ..types import RiskIndex
from ..aggregate import index_to_quarterly
from ._lexicons import LEXICONS
from ._kernels import keyword_count_kernel


def mpu(
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
    """Build a monetary-policy uncertainty series from a custom corpus.

    Parameters
    ----------
    text_iter : iterable of ``(date, text, source_url, metadata)`` records.
    country : ISO3 country tag.
    language : ISO-639-1; selects the default flat term-list lexicon if
        ``lexicon=None``.
    lexicon : optional ``frozenset[str]`` override.
    normalize : ``"raw"`` | ``"zscore"`` (default) | ``"bbd_100"``.
    base_period : optional ``(start_iso, end_iso)`` for normalisation
        statistics. When supplied, mean/std are computed on the slice
        ``series.loc[start:end]`` rather than the full series. Default
        ``None`` uses the full series.
    agg : aggregator across documents in a quarter.
    """
    terms = lexicon if lexicon is not None else LEXICONS["mpu"][language]

    def _kernel(records):
        return keyword_count_kernel(records, terms=terms, language=language)

    return index_to_quarterly(
        text_iter, kernel=_kernel,
        country=country, language=language,
        name=f"mpu_{country.lower()}",
        method="keyword_count", corpus="custom",
        normalization=normalize, agg=agg,
        metadata={"index": "mpu", "base_period": base_period},
        with_quality=with_quality,
    )


__all__ = ["mpu"]
