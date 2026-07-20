"""Hawkish-dovish tone indices for central-bank text.

Three methods are wired in Slice 2:

  - ``apel_blix_grimaldi`` (default) — net (hawk - dove) hits divided
    by total hits per document, then aggregated quarterly. This is the
    Apel-Blix-Grimaldi (2017) net-tone construction.
  - ``hubert`` — same count-based mechanism, currently identical to
    apel_blix_grimaldi (separate Hubert lexicon planned for Slice 3).
  - ``picault_renault`` — falls back to count-based for Slice 2; the
    full paragraph-level multinomial logit lands in Slice 3.

The ``llm`` method is reserved for Slice 3 (uses ``scoring/llm.py``
backends; Experimental tier).

References
----------
Apel, M., Blix Grimaldi, M. (2014). How informative are central bank
minutes? Sveriges Riksbank Working Paper 261.

Picault, M., Renault, T. (2017). Words are not all created equal: A new
measure of ECB communication. Journal of International Money and
Finance 79, 136-156.

Hubert, P. (2017). Central bank information and the effects of monetary
shocks. Bank of England Staff Working Paper 672.
"""
from __future__ import annotations

from typing import Iterable

from ..types import RiskIndex
from ..aggregate import index_to_quarterly
from ._lexicons import LEXICONS
from ._kernels import tone_kernel


_VALID_METHODS = {"apel_blix_grimaldi", "hubert", "picault_renault"}


def tone(
    text_iter: Iterable[tuple],
    *,
    country: str,
    language: str = "en",
    method: str = "apel_blix_grimaldi",
    lexicon: dict | None = None,
    normalize: str = "raw",
    base_period: tuple[str, str] | None = None,
    agg: str = "mean",
    with_quality: bool = False,
) -> RiskIndex:
    """Build a hawkish-dovish tone series from a custom corpus.

    Parameters
    ----------
    method : ``"apel_blix_grimaldi"`` (default) | ``"hubert"`` |
        ``"picault_renault"``. All three currently use the same
        net-count construction; the lexicon swaps differ (and Slice 3
        will refine ``picault_renault`` and ``hubert`` to their full
        methodologies).

    Note: ``base_period`` (when supplied) is plumbed through
    ``index_to_quarterly`` to ``normalize_series`` so that
    ``"zscore"`` / ``"bbd_100"`` statistics are computed on the slice
    ``series.loc[start:end]`` rather than the full series. The default
    ``None`` uses the full series for normalisation stats.
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f"method {method!r} not in {_VALID_METHODS}"
        )

    lex = lexicon if lexicon is not None else LEXICONS["tone"][language]

    def _kernel(records):
        return tone_kernel(
            records,
            hawkish_terms=lex["hawkish"],
            dovish_terms=lex["dovish"],
            language=language,
        )

    return index_to_quarterly(
        text_iter, kernel=_kernel,
        country=country, language=language,
        name=f"tone_{country.lower()}",
        method="tone_dispersion" if agg == "dispersion" else "keyword_count",
        corpus="custom",
        normalization=normalize, agg=agg,
        metadata={
            "index": "tone",
            "method_requested": method,
            "base_period": base_period,
        },
        with_quality=with_quality,
    )


__all__ = ["tone"]
