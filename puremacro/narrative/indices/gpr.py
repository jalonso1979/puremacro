"""Caldara-Iacoviello geopolitical risk index.

Counts documents containing geopolitical-risk terms, aggregates per
quarter, optionally normalises.

Reference
---------
Caldara, D., Iacoviello, M. (2022). Measuring geopolitical risk.
American Economic Review 112(4), 1194-1225.
"""
from __future__ import annotations

from typing import Iterable

from ..types import RiskIndex
from ..aggregate import index_to_quarterly
from ._lexicons import LEXICONS
from ._kernels import keyword_count_kernel


def gpr(
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
    """Build a Caldara-Iacoviello GPR series from a custom corpus.

    Note: ``base_period`` (when supplied) is plumbed through
    ``index_to_quarterly`` to ``normalize_series`` so that
    ``"zscore"`` / ``"bbd_100"`` statistics are computed on the slice
    ``series.loc[start:end]`` rather than the full series. The default
    ``None`` uses the full series for normalisation stats.
    """
    terms = lexicon if lexicon is not None else LEXICONS["gpr"][language]

    def _kernel(records):
        return keyword_count_kernel(records, terms=terms, language=language)

    return index_to_quarterly(
        text_iter, kernel=_kernel,
        country=country, language=language,
        name=f"gpr_{country.lower()}",
        method="keyword_count", corpus="custom",
        normalization=normalize, agg=agg,
        metadata={"index": "gpr", "base_period": base_period},
        with_quality=with_quality,
    )


__all__ = ["gpr"]
