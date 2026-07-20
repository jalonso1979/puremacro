"""Bluesky narrative uncertainty index (BLUESKY_UI).

Thin wrapper over ``puremacro.narrative.indices.lui``. Optional filters:
``actor_class`` ('institution' / 'governor' / 'minister') and
``country`` (ISO3 code).
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from ..types import RiskIndex
from .lui import lui


def bluesky_ui(
    records: Iterable[tuple] | pd.DataFrame,
    *,
    actor_class: str | None = None,
    country: str | None = None,
    country_arg: str = "WORLD",
    language: str = "en",
    aggregate_to: str | None = "actor_month",
    lexicon: dict | None = None,
    normalize: str = "zscore",
    base_period: tuple[str, str] | None = None,
    agg: str = "mean",
    negation: bool = True,
    with_quality: bool = False,
) -> RiskIndex:
    """Bluesky Uncertainty Index.

    Parameters
    ----------
    records : iterable of 4-tuples or DataFrame from iter_bluesky_posts.
    actor_class : optional filter ('institution', 'governor', 'minister').
    country : optional ISO3 country filter on metadata['country'].
    country_arg : country tag passed to lui (default 'WORLD').
    aggregate_to : aggregation granularity before LUI scoring. Default
        ``"actor_month"`` concatenates posts per (handle × calendar
        month) — recommended since Bluesky posts are ≤300 chars and
        LUI's sentence-cooccurrence kernel underperforms on short text.
        Pass ``None`` to disable aggregation (raw per-post records).
    Remaining kwargs forwarded to lui().

    Returns
    -------
    RiskIndex computed via LUI sentence-cooccurrence on filtered records.
    """
    df = _ensure_dataframe(records)
    if actor_class is not None:
        df = df[df["metadata"].apply(
            lambda m: str(m.get("actor_class", "")) == actor_class)]
    if country is not None:
        df = df[df["metadata"].apply(
            lambda m: str(m.get("country", "")) == country)]
    if aggregate_to == "actor_month":
        df = _aggregate_to_actor_month(df)
    elif aggregate_to is None:
        pass
    else:
        raise ValueError(
            f"unknown aggregate_to: {aggregate_to!r}; "
            f"expected 'actor_month' or None"
        )
    text_iter = (
        (row.date, row.text, row.source_url, row.metadata)
        for row in df.itertuples()
    )
    return lui(text_iter, country=country_arg, language=language,
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


def _aggregate_to_actor_month(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate posts per (handle × calendar month) by concatenating text.

    Bluesky posts are ≤300 chars; LUI's sentence-cooccurrence kernel
    underperforms on short text. Aggregating to monthly granularity
    per actor mitigates that. Date becomes the first-of-month;
    source_url is the first one in the group (4-tuple contract only
    permits a single URL); metadata preserves handle/country/role/
    actor_class from the first row and adds ``n_posts`` + ``aggregation``.
    """
    if df.empty:
        return df

    df = df.copy()
    df["_month"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
    df["_handle"] = df["metadata"].apply(lambda m: m.get("handle", ""))

    grouped = []
    for (handle, month), grp in df.groupby(["_handle", "_month"]):
        first = grp.iloc[0]
        meta = dict(first["metadata"])
        meta["n_posts"] = len(grp)
        meta["aggregation"] = "actor_month"
        grouped.append({
            "date": month.date(),
            "text": " ".join(grp["text"].astype(str)),
            "source_url": first["source_url"],
            "metadata": meta,
        })
    return pd.DataFrame(grouped, columns=["date", "text", "source_url", "metadata"])


__all__ = ["bluesky_ui"]
