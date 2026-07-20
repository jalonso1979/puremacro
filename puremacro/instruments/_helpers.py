"""Shared adapters for instrument-loader CSV/JSON → Instrument conversion.

This module is the canonical home of ``_csv_to_instrument`` (originally in
``literature/_helpers.py`` and promoted here in 0.5.2 so the new
``external/`` subpackage can share it) and ``_json_to_instrument`` (new in
0.5.2 to support FRED's JSON observations format). Used by both
``literature/`` and ``external/`` loaders.
"""
from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from ._core import Instrument


def _csv_to_instrument(
    df: pd.DataFrame,
    *,
    name: str,
    source: str,
    frequency: str,
    value_col: str,
    date_col: str | None = None,
    year_col: str | None = None,
    month_col: str | None = None,
    metadata: dict[str, Any] | None = None,
    category: str = "literature",
) -> Instrument:
    """Convert a parsed DataFrame into an :class:`Instrument`.

    The DataFrame may carry the date in two forms:

    * a single ``date_col`` column (parsed via ``pd.to_datetime``); or
    * a ``year_col`` + ``month_col`` pair (combined into month-start dates).

    Pass either ``date_col`` OR (``year_col`` + ``month_col``), not both.
    """
    if date_col is not None and (year_col is not None or month_col is not None):
        raise ValueError(
            "pass either date_col or (year_col + month_col), not both"
        )
    if date_col is not None:
        dates = pd.to_datetime(df[date_col])
    elif year_col is not None and month_col is not None:
        dates = pd.to_datetime(
            df[year_col].astype(int).astype(str) + "-"
            + df[month_col].astype(int).astype(str).str.zfill(2) + "-01"
        )
    else:
        raise ValueError(
            "must provide either date_col or both year_col + month_col"
        )
    values = df[value_col].astype(float).values
    series = pd.Series(values, index=dates, name=name).sort_index()
    return Instrument(
        series=series,
        name=name,
        source=source,
        category=category,
        frequency=frequency,
        metadata=metadata or {},
    )


def _json_to_instrument(
    observations: Iterable[dict[str, Any]],
    *,
    name: str,
    source: str,
    frequency: str,
    date_field: str,
    value_field: str,
    metadata: dict[str, Any] | None = None,
    category: str = "external_csv",
    missing_markers: tuple[str, ...] = (".", "", "NaN", "nan"),
) -> Instrument:
    """Convert a list of observation dicts (e.g. FRED JSON response) into an
    :class:`Instrument`.

    Parameters
    ----------
    observations : iterable of dicts
        Each dict has at least the ``date_field`` and ``value_field`` keys.
    date_field : str
        Key name for the date string in each observation.
    value_field : str
        Key name for the value (string-encoded; will be coerced to float).
    missing_markers : tuple of str, default ``(".", "", "NaN", "nan")``
        String values to treat as NaN. FRED uses ``"."``.
    """
    obs_list = list(observations)
    if not obs_list:
        raise ValueError("empty observations list")
    dates = pd.to_datetime([o[date_field] for o in obs_list])
    raw_values = [o[value_field] for o in obs_list]
    values = []
    for v in raw_values:
        if isinstance(v, str) and v in missing_markers:
            values.append(float("nan"))
        else:
            try:
                values.append(float(v))
            except (TypeError, ValueError):
                values.append(float("nan"))
    series = pd.Series(values, index=dates, name=name).sort_index()
    return Instrument(
        series=series,
        name=name,
        source=source,
        category=category,
        frequency=frequency,
        metadata=metadata or {},
    )


__all__ = ["_csv_to_instrument", "_json_to_instrument"]
