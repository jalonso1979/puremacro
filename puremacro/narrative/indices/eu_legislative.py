"""EU legislative narrative uncertainty indices (EURLEX_UI, EP_UI).

Two thin wrappers over ``puremacro.narrative.indices.lui``. Each
filters records by language (mandatory — connectors emit multi-
language records) plus optional source-specific fields, then
delegates to the LUI sentence-cooccurrence kernel.

The LUI lexicon's en / de / fr sub-dictionaries handle language-
specific term sets.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from ..types import RiskIndex
from .lui import lui


def eurlex_ui(
    records: Iterable[tuple] | pd.DataFrame,
    *,
    language: str = "en",
    act_type: str | None = None,
    country: str = "EU",
    lexicon: dict | None = None,
    normalize: str = "zscore",
    base_period: tuple[str, str] | None = None,
    agg: str = "mean",
    negation: bool = True,
    with_quality: bool = False,
) -> RiskIndex:
    """EUR-Lex Uncertainty Index. Required language filter; optional act_type."""
    df = _ensure_dataframe(records)
    lang_lower = language.lower()
    df = df[df["metadata"].apply(
        lambda m: str(m.get("language", "")).lower() == lang_lower)]
    if act_type is not None:
        df = df[df["metadata"].apply(
            lambda m: str(m.get("act_type", "")) == act_type)]
    text_iter = (
        (row.date, row.text, row.source_url, row.metadata)
        for row in df.itertuples()
    )
    return lui(text_iter, country=country, language=language,
               lexicon=lexicon, normalize=normalize,
               base_period=base_period, agg=agg, negation=negation,
               with_quality=with_quality)


def ep_ui(
    records: Iterable[tuple] | pd.DataFrame,
    *,
    language: str = "en",
    country: str = "EU",
    lexicon: dict | None = None,
    normalize: str = "zscore",
    base_period: tuple[str, str] | None = None,
    agg: str = "mean",
    negation: bool = True,
    with_quality: bool = False,
) -> RiskIndex:
    """EU Parliament Uncertainty Index. Required language filter."""
    df = _ensure_dataframe(records)
    lang_lower = language.lower()
    df = df[df["metadata"].apply(
        lambda m: str(m.get("language", "")).lower() == lang_lower)]
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


__all__ = ["eurlex_ui", "ep_ui"]
