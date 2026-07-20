"""Labor-War-Uncertainty Index (LWUI) — paragraph-level triple
co-occurrence (Slice e).

Score is the (density-weighted) fraction of paragraphs that contain at
least one term from each of three groups:

    labor_domain × uncertainty_tone × war_domain

Parallel construction to LTUI (slice b). Built on top of
``triple_cooccurrence_kernel`` with war-domain density weighting.
"""
from __future__ import annotations

from typing import Iterable

from ..types import RiskIndex
from ..aggregate import index_to_quarterly
from ._lexicons import LEXICONS
from ._kernels import triple_cooccurrence_kernel


_DEFAULT_BASE_PERIOD: tuple[str, str] = ("2014-01-01", "2026-04-01")


def lwui(
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
    """Build a labor-war-uncertainty series from a custom corpus.

    Mirrors the ``ltui()`` API. Default ``base_period`` is
    2014-01-01 → 2026-04-01 — covers the major post-Crimea / Ukraine / Gaza
    geopolitical-conflict window where war-labor talk is non-trivial.

    Parameters
    ----------
    text_iter : iterable of ``(date, text, source_url, metadata)`` records.
    country : ISO3 country tag stamped onto the resulting RiskIndex.
    language : ISO-639-1; selects the default lexicon if ``lexicon=None``.
    lexicon : optional override of the form
        ``{"labor_domain": frozenset, "uncertainty_tone": frozenset,
           "war_domain": frozenset}``.
    normalize : ``"raw"`` | ``"zscore"`` | ``"bbd_100"``.
    base_period : ``(start_iso, end_iso)`` for normalisation statistics.
    agg : ``"mean"`` (default) | ``"max"`` | ``"dispersion"``.
    """
    lex = lexicon if lexicon is not None else LEXICONS["lwui"][language]
    term_groups = [
        lex["labor_domain"],
        lex["uncertainty_tone"],
        lex["war_domain"],
    ]
    bp = base_period if base_period is not None else _DEFAULT_BASE_PERIOD

    def _kernel(records):
        return triple_cooccurrence_kernel(
            records,
            term_groups=term_groups,
            weight_group_idx=2,
            window="paragraph",
            language=language,
            negation=negation,
        )

    return index_to_quarterly(
        text_iter, kernel=_kernel,
        country=country, language=language,
        name=f"lwui_{country.lower()}",
        method="triple_cooccurrence_paragraph", corpus="custom",
        normalization=normalize, agg=agg,
        metadata={"index": "lwui", "base_period": bp},
        with_quality=with_quality,
    )


_DEFAULT_BASE_PERIOD_WAGE: tuple[str, str] = ("2006-01-01", "2026-04-01")


def lwui_wage(
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
    """Labor-Wage-Uncertainty Index — paragraph-level triple co-occurrence:

        labor_domain × uncertainty_tone × wage_domain

    Parallel construction to ``lwui`` (war flavour) and ``ltui`` (tech
    flavour); same kernel, same multilingual lexicons. Default
    ``base_period`` covers 2006-2026 (the broader inflation cycle where
    wage-uncertainty talk is dense). Use this variant when the question
    is about *price-of-labor* uncertainty rather than geopolitical labour
    risk.

    Parameters mirror :func:`lwui` exactly except that the third term
    group is ``wage_domain`` and the lexicon comes from
    ``LEXICONS["lwui_wage"][language]``.
    """
    lex = lexicon if lexicon is not None else LEXICONS["lwui_wage"][language]
    term_groups = [
        lex["labor_domain"],
        lex["uncertainty_tone"],
        lex["wage_domain"],
    ]
    bp = base_period if base_period is not None else _DEFAULT_BASE_PERIOD_WAGE

    def _kernel(records):
        return triple_cooccurrence_kernel(
            records,
            term_groups=term_groups,
            weight_group_idx=2,
            window="paragraph",
            language=language,
            negation=negation,
        )

    return index_to_quarterly(
        text_iter, kernel=_kernel,
        country=country, language=language,
        name=f"lwui_wage_{country.lower()}",
        method="triple_cooccurrence_paragraph", corpus="custom",
        normalization=normalize, agg=agg,
        metadata={"index": "lwui_wage", "base_period": bp},
        with_quality=with_quality,
    )


__all__ = ["lwui", "lwui_wage"]
