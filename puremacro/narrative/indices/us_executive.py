"""US executive narrative uncertainty indices (ERPUI, SOTUUI, CBOUI).

Three thin wrappers over ``puremacro.narrative.indices.lui``. Each
filters records (optionally by chapter/topic) and delegates to the
LUI sentence-cooccurrence kernel. The LUI lexicon transfers cleanly
to US executive narrative (large macroeconomic vocabulary overlap);
pass ``lexicon=...`` for a domain-specific term set.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from ..types import RiskIndex
from .lui import lui


def erpui(
    records: Iterable[tuple] | pd.DataFrame,
    *,
    chapter: str | None = None,
    country: str = "USA",
    language: str = "en",
    lexicon: dict | None = None,
    normalize: str = "zscore",
    base_period: tuple[str, str] | None = None,
    agg: str = "mean",
    negation: bool = True,
    with_quality: bool = False,
) -> RiskIndex:
    """ERP Uncertainty Index. Optional chapter substring filter."""
    df = _ensure_dataframe(records)
    if chapter is not None:
        ch_lower = chapter.lower()
        df = df[df["metadata"].apply(
            lambda m: ch_lower in str(m.get("chapter", "")).lower())]
    text_iter = (
        (row.date, row.text, row.source_url, row.metadata)
        for row in df.itertuples()
    )
    return lui(text_iter, country=country, language=language,
               lexicon=lexicon, normalize=normalize,
               base_period=base_period, agg=agg, negation=negation,
               with_quality=with_quality)


def sotuui(
    records: Iterable[tuple] | pd.DataFrame,
    *,
    country: str = "USA",
    language: str = "en",
    lexicon: dict | None = None,
    normalize: str = "zscore",
    base_period: tuple[str, str] | None = None,
    agg: str = "mean",
    negation: bool = True,
    with_quality: bool = False,
) -> RiskIndex:
    """SOTU Uncertainty Index. Per-document only."""
    df = _ensure_dataframe(records)
    text_iter = (
        (row.date, row.text, row.source_url, row.metadata)
        for row in df.itertuples()
    )
    return lui(text_iter, country=country, language=language,
               lexicon=lexicon, normalize=normalize,
               base_period=base_period, agg=agg, negation=negation,
               with_quality=with_quality)


def cboui(
    records: Iterable[tuple] | pd.DataFrame,
    *,
    topic: str | None = None,
    country: str = "USA",
    language: str = "en",
    lexicon: dict | None = None,
    normalize: str = "zscore",
    base_period: tuple[str, str] | None = None,
    agg: str = "mean",
    negation: bool = True,
    with_quality: bool = False,
) -> RiskIndex:
    """CBO Uncertainty Index. Optional topic filter (substring on title)."""
    df = _ensure_dataframe(records)
    if topic is not None:
        topic_lower = topic.lower()
        df = df[df["metadata"].apply(
            lambda m: topic_lower in str(m.get("title", "")).lower())]
    text_iter = (
        (row.date, row.text, row.source_url, row.metadata)
        for row in df.itertuples()
    )
    return lui(text_iter, country=country, language=language,
               lexicon=lexicon, normalize=normalize,
               base_period=base_period, agg=agg, negation=negation,
               with_quality=with_quality)


def _ensure_dataframe(records: Iterable[tuple] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        required = {"date", "text", "source_url", "metadata"}
        missing = required - set(records.columns)
        if missing:
            raise ValueError(
                f"records DataFrame missing columns: {sorted(missing)}")
        return records
    rows = list(records)
    if not rows:
        return pd.DataFrame(columns=["date", "text", "source_url", "metadata"])
    cols = ["date", "text", "source_url", "metadata"]
    return pd.DataFrame(rows, columns=cols)


__all__ = ["erpui", "sotuui", "cboui"]
