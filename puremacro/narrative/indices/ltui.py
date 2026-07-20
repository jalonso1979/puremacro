"""Labor-Technological-Uncertainty Index (LTUI) — paragraph-level triple
co-occurrence (Slice b).

Score is the (density-weighted) fraction of paragraphs that contain at
least one term from each of three groups:

    labor_domain × uncertainty_tone × tech_domain

Built on top of ``triple_cooccurrence_kernel``. By default the score is
weighted by the tech-domain density of the matched paragraph (so a
paragraph saying "ai automation digital ..." outweighs a paragraph that
mentions "ai" once).
"""
from __future__ import annotations

from typing import Iterable

from ..types import RiskIndex
from ..aggregate import index_to_quarterly
from ._lexicons import LEXICONS
from ._kernels import triple_cooccurrence_kernel


_DEFAULT_BASE_PERIOD: tuple[str, str] = ("2017-01-01", "2026-04-01")


def ltui(
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
    """Build a labor-technological-uncertainty series from a custom corpus.

    Parameters
    ----------
    text_iter : iterable of ``(date, text, source_url, metadata)`` records.
    country : ISO3 country tag stamped onto the resulting RiskIndex.
    language : ISO-639-1; selects the default lexicon if ``lexicon=None``.
    lexicon : optional override of the form
        ``{"labor_domain": frozenset, "uncertainty_tone": frozenset,
           "tech_domain": frozenset}``.
    normalize : ``"raw"`` | ``"zscore"`` | ``"bbd_100"``.
    base_period : ``(start_iso, end_iso)`` for normalisation statistics.
        Defaults to the GenAI window 2017-01-01 → 2026-04-01 (see spec).
    agg : ``"mean"`` (default) | ``"max"`` | ``"dispersion"``.
    """
    lex = lexicon if lexicon is not None else LEXICONS["ltui"][language]
    term_groups = [
        lex["labor_domain"],
        lex["uncertainty_tone"],
        lex["tech_domain"],
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
        name=f"ltui_{country.lower()}",
        method="triple_cooccurrence_paragraph", corpus="custom",
        normalization=normalize, agg=agg,
        metadata={"index": "ltui", "base_period": bp},
        with_quality=with_quality,
    )


def _ltui_directional(
    text_iter: Iterable[tuple],
    *,
    direction: str,
    country: str,
    language: str = "en",
    lexicon: dict | None = None,
    normalize: str = "zscore",
    base_period: tuple[str, str] | None = None,
    agg: str = "mean",
    negation: bool = True,
    with_quality: bool = False,
) -> RiskIndex:
    """Internal: 4-group co-occurrence kernel (labor x uncertainty x tech x polarity).

    Used by ltui_up()/ltui_down() — see those for parameter docs.
    """
    assert direction in ("up", "down")
    key = "ltui_up" if direction == "up" else "ltui_down"
    lex = lexicon if lexicon is not None else LEXICONS[key][language]
    term_groups = [
        lex["labor_domain"],
        lex["uncertainty_tone"],
        lex["tech_domain"],
        lex["polarity"],
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
        name=f"{key}_{country.lower()}",
        method="triple_cooccurrence_paragraph", corpus="custom",
        normalization=normalize, agg=agg,
        metadata={"index": key, "base_period": bp},
        with_quality=with_quality,
    )


def ltui_up(
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
    """LTUI sub-index conditional on upside (tech-labor productivity / augmentation) framing.

    4-group co-occurrence: labor x uncertainty x tech x upside-polarity. Returns
    a RiskIndex with the same shape as ltui(). The optional ``lexicon`` override
    accepts a 4-key dict ``{labor_domain, uncertainty_tone, tech_domain, polarity}``.
    """
    return _ltui_directional(
        text_iter, direction="up",
        country=country, language=language, lexicon=lexicon,
        normalize=normalize, base_period=base_period, agg=agg,
        negation=negation, with_quality=with_quality,
    )


def ltui_down(
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
    """LTUI sub-index conditional on downside (displacement / job loss) framing.

    4-group co-occurrence: labor x uncertainty x tech x downside-polarity.
    """
    return _ltui_directional(
        text_iter, direction="down",
        country=country, language=language, lexicon=lexicon,
        normalize=normalize, base_period=base_period, agg=agg,
        negation=negation, with_quality=with_quality,
    )


__all__ = ["ltui", "ltui_up", "ltui_down"]
