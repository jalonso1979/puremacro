"""Beige Book Uncertainty Index (BBUI).

A thin wrapper over ``puremacro.narrative.indices.lui``: filters records
emitted by ``iter_beige_book()`` to a canonical section (or all sections)
and computes the LUI sentence-level co-occurrence index. The LUI lexicon
transfers cleanly to Beige Book commentary, which is overwhelmingly
labor + macro narrative — though callers can pass a custom lexicon if a
domain-specific term set is needed.
"""
from __future__ import annotations

import warnings
from typing import Iterable, Literal

import pandas as pd

from ..types import RiskIndex
from .lui import lui


def bbui(
    records: Iterable[tuple] | pd.DataFrame,
    *,
    section: str | None = None,
    level: Literal["national", "district"] = "national",
    output: Literal["wide", "tidy"] = "wide",
    country: str = "USA",
    language: str = "en",
    lexicon: dict | None = None,
    normalize: str = "zscore",
    base_period: tuple[str, str] | None = None,
    agg: str = "mean",
    negation: bool = True,
    with_quality: bool = False,
) -> RiskIndex | pd.DataFrame:
    """Beige Book Uncertainty Index.

    Parameters
    ----------
    records : iterable of 6-tuples ``(date, district, section, text,
        source_url, metadata)`` (the output of ``iter_beige_book``),
        or a DataFrame with the same columns.
    section : optional filter to one canonical section
        (``'overall'``, ``'labor'``, ``'prices'``, ``'consumer'``,
        ``'manufacturing'``, ``'realestate'``). ``None`` keeps all
        sections.
    level : ``'national'`` returns a single RiskIndex; ``'district'``
        returns a DataFrame (see ``output``).
    output : shape of the ``level='district'`` DataFrame.
        ``'wide'`` (default, backward compatible): quarter-start date
        index × one float column per district.
        ``'tidy'``: long DataFrame with one row per (quarter, district)
        and columns ``date``, ``district``, ``value``, ``n_docs``
        (input records scored into that district-quarter — one record
        per canonical section at the connector's default granularity)
        and ``n_sections`` (distinct canonical sections covered).
        Only valid with ``level='district'``.
    country : passed to underlying ``lui``; defaults to USA.
    lexicon : optional override; defaults to the LUI lexicon.
    Remaining kwargs are forwarded verbatim to ``lui``.

    Notes
    -----
    District labels are the connector's canonical names
    (``puremacro.narrative.sources.beige_book.DISTRICTS``) plus
    ``"National"`` for the national-summary pseudo-district, which is
    kept in district-level output so callers can compare each district
    against the summary; filter it out for a strict 12-district panel.
    A state-level crosswalk for merging district output onto state data
    is provided by ``puremacro.narrative.indices._fed_districts``.
    With ``normalize='zscore'`` (default) each district is z-scored
    against its own history — use ``normalize='raw'`` for level
    comparisons across districts.

    Returns
    -------
    ``RiskIndex`` when ``level='national'``; ``pd.DataFrame`` when
    ``level='district'``.
    """
    if output not in ("wide", "tidy"):
        raise ValueError(f"output must be 'wide' or 'tidy', got {output!r}")
    if level == "national" and output != "wide":
        raise ValueError(
            "output='tidy' is only meaningful with level='district'; "
            "level='national' returns a RiskIndex"
        )
    df = _ensure_dataframe(records)
    if section is not None:
        df = df[df["section"] == section]

    if level == "national":
        text_iter = (
            (row.date, row.text, row.source_url,
             {"district": row.district, "section": row.section})
            for row in df.itertuples()
        )
        return lui(text_iter, country=country, language=language,
                   lexicon=lexicon, normalize=normalize,
                   base_period=base_period, agg=agg, negation=negation,
                   with_quality=with_quality)

    if level == "district":
        if with_quality:
            warnings.warn(
                "bbui(level='district') does not return a RiskIndex; per-period "
                "SignalQualityReport is only available in level='national' mode. "
                "Ignoring with_quality=True for this district-level call.",
                UserWarning,
                stacklevel=2,
            )
        if df.empty:
            if output == "tidy":
                return pd.DataFrame(columns=["date", "district", "value",
                                             "n_docs", "n_sections"])
            return pd.DataFrame()
        out: dict[str, pd.Series] = {}
        tidy_frames: list[pd.DataFrame] = []
        for district, group in df.groupby("district"):
            text_iter = (
                (row.date, row.text, row.source_url, {})
                for row in group.itertuples()
            )
            ri = lui(text_iter, country=country, language=language,
                     lexicon=lexicon, normalize=normalize,
                     base_period=base_period, agg=agg, negation=negation,
                     with_quality=False)
            out[district] = ri.series  # RiskIndex.series is the pd.Series
            if output == "tidy":
                cov = group.copy()
                cov["date"] = (
                    pd.to_datetime(cov["date"])
                    .dt.to_period("Q").dt.to_timestamp()
                )
                coverage = cov.groupby("date").agg(
                    n_docs=("text", "size"),
                    n_sections=("section", "nunique"),
                )
                tidy = ri.series.rename("value").to_frame()
                tidy = tidy.join(coverage)
                # Quarters present in the reindexed series but with no
                # underlying documents (gaps) carry zero coverage.
                tidy[["n_docs", "n_sections"]] = (
                    tidy[["n_docs", "n_sections"]].fillna(0).astype(int)
                )
                tidy.insert(0, "district", district)
                tidy_frames.append(tidy.reset_index())
        if output == "tidy":
            result = pd.concat(tidy_frames, ignore_index=True)
            result = result[["date", "district", "value",
                             "n_docs", "n_sections"]]
            return result.sort_values(["date", "district"],
                                      ignore_index=True)
        return pd.DataFrame(out)

    raise ValueError(f"level must be 'national' or 'district', got {level!r}")


def _ensure_dataframe(records: Iterable[tuple] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        required = {"date", "district", "section", "text", "source_url"}
        missing = required - set(records.columns)
        if missing:
            raise ValueError(
                f"records DataFrame missing columns: {sorted(missing)}")
        return records
    rows = list(records)
    if not rows:
        return pd.DataFrame(columns=["date", "district", "section",
                                      "text", "source_url", "metadata"])
    if len(rows[0]) == 5:
        cols = ["date", "district", "section", "text", "source_url"]
    elif len(rows[0]) == 6:
        cols = ["date", "district", "section", "text", "source_url",
                "metadata"]
    else:
        raise ValueError(
            f"records tuples must be 5 or 6 long, got {len(rows[0])}")
    return pd.DataFrame(rows, columns=cols)


__all__ = ["bbui"]
