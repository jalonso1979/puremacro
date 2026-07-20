"""Baker-Bloom-Davis Economic Policy Uncertainty index.

Constructs a count of documents that contain at least one term from
each of three groups (Economy, Policy, Uncertainty), aggregates by
quarter, and optionally normalises (z-score or BBD's 100/50 scale).

Reference
---------
Baker, S.R., Bloom, N., Davis, S.J. (2016). Measuring economic policy
uncertainty. QJE 131(4), 1593-1636.
"""
from __future__ import annotations

from typing import Iterable

from ..types import RiskIndex
from ..aggregate import index_to_quarterly
from ._lexicons import LEXICONS
from ._kernels import cooccurrence_kernel


def epu(
    text_iter: Iterable[tuple],
    *,
    country: str,
    language: str = "en",
    lexicon: dict | None = None,
    normalize: str = "bbd_100",
    base_period: tuple[str, str] | None = None,
    agg: str = "mean",
    with_quality: bool = False,
) -> RiskIndex:
    """Build a Baker-Bloom-Davis EPU series from a custom corpus.

    Parameters
    ----------
    text_iter : iterable of ``(date, text, source_url, metadata)`` records.
    country : ISO3 country tag stamped onto the resulting RiskIndex.
    language : ISO-639-1; selects the default lexicon if ``lexicon=None``.
    lexicon : optional override of the form
        ``{"economy": frozenset, "policy": frozenset, "uncertainty": frozenset}``.
    normalize : ``"raw"`` | ``"zscore"`` | ``"bbd_100"``.
    base_period : ``(start_iso, end_iso)`` for normalisation statistics.
        When supplied, mean/std are computed on the slice
        ``series.loc[start:end]`` (e.g. BBD's published 1985–2009 base)
        rather than the full series. Default ``None`` uses the full series.
    agg : ``"mean"`` (default) | ``"max"`` | ``"dispersion"`` aggregator
        applied across documents within each quarter.
    """
    lex = lexicon if lexicon is not None else LEXICONS["epu"][language]
    term_groups = [lex["economy"], lex["policy"], lex["uncertainty"]]

    def _kernel(records):
        return cooccurrence_kernel(
            records, term_groups=term_groups, language=language,
        )

    return index_to_quarterly(
        text_iter, kernel=_kernel,
        country=country, language=language,
        name=f"epu_{country.lower()}",
        method="keyword_count", corpus="custom",
        normalization=normalize, agg=agg,
        metadata={
            "index": "epu", "base_period": base_period,
        },
        with_quality=with_quality,
    )


__all__ = ["epu"]
