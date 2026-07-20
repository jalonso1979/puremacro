"""Real-time / vintage-aware panel utilities.

A real-time panel is a long DataFrame with three keys:
    date    : the calendar period the value refers to
    vintage : the publication date (when this value became known)
    value   : the published number

Macro releases get revised, so for any reference date there is a
sequence of vintage values, each more "final" than the last. The
helpers here let you (a) build a "real-time" snapshot — what was
*known* on a given vintage date, (b) re-run a forecast model across
vintages and track how the forecast revises with new information.

This pairs naturally with FRED-ALFRED-style data delivered by
``puremacro.fetch.fetch_fred_alfred``.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def as_of(
    panel_long: pd.DataFrame,
    vintage_date,
    date_col: str = "date",
    vintage_col: str = "vintage",
    value_col: str = "value",
) -> pd.Series:
    """Slice a real-time panel to the latest value known *as of* vintage_date.

    Parameters
    ----------
    panel_long : long-form DataFrame with (date, vintage, value) columns.
    vintage_date : timestamp; only rows with vintage <= vintage_date are
        considered. For each ``date``, the most recent surviving vintage
        is kept.

    Returns
    -------
    pd.Series indexed by ``date`` with the as-of values.
    """
    sub = panel_long[panel_long[vintage_col] <= vintage_date]
    if sub.empty:
        return pd.Series(dtype=float)
    sub = sub.sort_values([date_col, vintage_col])
    out = sub.groupby(date_col, sort=True).last()[value_col]
    out.name = value_col
    return out


def align_vintages(
    panel_long: pd.DataFrame,
    vintage_dates,
    date_col: str = "date",
    vintage_col: str = "vintage",
    value_col: str = "value",
) -> pd.DataFrame:
    """Build a (date × vintage) wide matrix of as-of snapshots.

    Each column is the panel as it was known on a particular vintage
    date; missing entries are NaN where no observation had been
    published yet.
    """
    cols = {}
    for v in vintage_dates:
        cols[pd.Timestamp(v)] = as_of(panel_long, v,
                                       date_col=date_col,
                                       vintage_col=vintage_col,
                                       value_col=value_col)
    out = pd.DataFrame(cols).sort_index()
    out.columns.name = "vintage"
    return out


def forecast_revision(
    panel_long: pd.DataFrame,
    vintage_dates,
    forecast_fn: Callable[[pd.Series], float],
    *,
    date_col: str = "date",
    vintage_col: str = "vintage",
    value_col: str = "value",
) -> pd.Series:
    """Track how a single-number forecast revises across vintages.

    For each vintage in ``vintage_dates``, build the as-of snapshot,
    pass it to ``forecast_fn`` (a function taking a Series and returning
    a scalar — e.g., next-period AR(1) forecast), and collect the
    sequence of forecasts.

    Returns
    -------
    pd.Series indexed by vintage date.
    """
    out = {}
    for v in vintage_dates:
        snap = as_of(panel_long, v, date_col=date_col,
                     vintage_col=vintage_col, value_col=value_col)
        if snap.empty:
            out[pd.Timestamp(v)] = np.nan
            continue
        try:
            out[pd.Timestamp(v)] = float(forecast_fn(snap))
        except Exception:
            out[pd.Timestamp(v)] = np.nan
    s = pd.Series(out).sort_index()
    s.index.name = "vintage"
    return s


__all__ = ["as_of", "align_vintages", "forecast_revision", "AlfredVintageStore", "as_of_from_store"]


# ── AlfredVintageStore (0.66.0+) ────────────────────────────────────
# Persistent local store for FRED-ALFRED vintage observations, backed
# by the shared SQLite cache DB (see puremacro._cache_db). Existing
# in-memory helpers (as_of, align_vintages, forecast_revision) are
# untouched; this class adds the store-backed counterpart so research
# notebooks don't refetch ALFRED on every kernel restart.

import sqlite3 as _sqlite3
import warnings as _warnings
from pathlib import Path as _Path

import pandas as _pd


class AlfredVintageStore:
    """Persistent store for FRED-ALFRED vintage panels.

    Backed by the ``alfred_vintages`` table in the shared SQLite cache
    DB (``~/.cache/puremacro/cache.db`` by default). Failures (DB
    locked, disk full, etc.) emit a ``UserWarning`` and degrade
    gracefully — ``get()`` returns an empty DataFrame, ``put_many()``
    no-ops. Research notebooks never crash on store errors; they just
    fall through to the API.
    """

    def __init__(self, db_path: "_Path | None" = None):
        from . import _cache_db
        self._cache_db = _cache_db
        self._db_path = db_path

    def _conn(self) -> _sqlite3.Connection:
        return self._cache_db.get_conn(self._db_path)

    def put(
        self,
        series_id: str,
        observation_date: str,
        vintage_date: str,
        value: float | None,
    ) -> None:
        """Insert (or replace) a single vintage observation."""
        try:
            self._conn().execute(
                "INSERT OR REPLACE INTO alfred_vintages "
                "(series_id, observation_date, vintage_date, value) "
                "VALUES (?, ?, ?, ?)",
                (series_id, observation_date, vintage_date, value),
            )
        except (_sqlite3.OperationalError, _sqlite3.DatabaseError) as e:
            _warnings.warn(
                f"AlfredVintageStore.put({series_id!r}, ...) failed: {e}",
                UserWarning, stacklevel=2,
            )

    def put_many(self, df: _pd.DataFrame) -> int:
        """Bulk insert. Required columns: ['series_id', 'observation_date',
        'vintage_date', 'value']. Returns count of rows inserted/replaced.
        Returns 0 on DB error."""
        required = {"series_id", "observation_date", "vintage_date", "value"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"AlfredVintageStore.put_many: missing columns {sorted(missing)}"
            )
        rows = [
            (str(r["series_id"]),
             str(r["observation_date"])[:10],
             str(r["vintage_date"])[:10],
             None if _pd.isna(r["value"]) else float(r["value"]))
            for _, r in df.iterrows()
        ]
        try:
            self._conn().executemany(
                "INSERT OR REPLACE INTO alfred_vintages "
                "(series_id, observation_date, vintage_date, value) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )
            return len(rows)
        except (_sqlite3.OperationalError, _sqlite3.DatabaseError) as e:
            _warnings.warn(
                f"AlfredVintageStore.put_many failed: {e}",
                UserWarning, stacklevel=2,
            )
            return 0

    def get(
        self,
        series_id: str,
        *,
        vintage_until: str | None = None,
    ) -> _pd.DataFrame:
        """Return long-form DataFrame ['observation_date', 'vintage_date',
        'value'] for ``series_id``. Empty DataFrame on missing series or
        DB error. ``observation_date`` and ``vintage_date`` are
        ``pd.Timestamp``-typed."""
        sql = (
            "SELECT observation_date, vintage_date, value "
            "FROM alfred_vintages WHERE series_id = ?"
        )
        params: list = [series_id]
        if vintage_until is not None:
            sql += " AND vintage_date <= ?"
            params.append(str(vintage_until)[:10])
        sql += " ORDER BY observation_date, vintage_date"
        try:
            rows = self._conn().execute(sql, params).fetchall()
        except (_sqlite3.OperationalError, _sqlite3.DatabaseError) as e:
            _warnings.warn(
                f"AlfredVintageStore.get({series_id!r}) failed: {e}",
                UserWarning, stacklevel=2,
            )
            return _pd.DataFrame(
                columns=["observation_date", "vintage_date", "value"]
            )
        if not rows:
            return _pd.DataFrame(
                columns=["observation_date", "vintage_date", "value"]
            )
        df = _pd.DataFrame(
            rows, columns=["observation_date", "vintage_date", "value"]
        )
        df["observation_date"] = _pd.to_datetime(df["observation_date"])
        df["vintage_date"] = _pd.to_datetime(df["vintage_date"])
        return df

    def has_series(self, series_id: str) -> bool:
        try:
            row = self._conn().execute(
                "SELECT 1 FROM alfred_vintages WHERE series_id = ? LIMIT 1",
                (series_id,),
            ).fetchone()
            return row is not None
        except (_sqlite3.OperationalError, _sqlite3.DatabaseError):
            return False

    def series_list(self) -> list[str]:
        try:
            rows = self._conn().execute(
                "SELECT DISTINCT series_id FROM alfred_vintages ORDER BY series_id"
            ).fetchall()
            return [r[0] for r in rows]
        except (_sqlite3.OperationalError, _sqlite3.DatabaseError):
            return []

    def coverage(self, series_id: str) -> dict | None:
        """Diagnostic: counts + first/last observation/vintage dates,
        or None if has_series(series_id) is False."""
        if not self.has_series(series_id):
            return None
        row = self._conn().execute(
            "SELECT COUNT(*), MIN(observation_date), MAX(observation_date), "
            "MIN(vintage_date), MAX(vintage_date), "
            "COUNT(DISTINCT vintage_date) "
            "FROM alfred_vintages WHERE series_id = ?",
            (series_id,),
        ).fetchone()
        n_rows, first_obs, last_obs, first_v, last_v, n_v = row
        return {
            "n_rows":         int(n_rows),
            "first_obs":      _pd.to_datetime(first_obs),
            "last_obs":       _pd.to_datetime(last_obs),
            "first_vintage":  _pd.to_datetime(first_v),
            "last_vintage":   _pd.to_datetime(last_v),
            "n_vintages":     int(n_v),
        }


def as_of_from_store(
    series_id: str,
    vintage_date: str,
    store: "AlfredVintageStore",
) -> _pd.Series:
    """Pull ``series_id`` from ``store`` (vintages on or before
    ``vintage_date``), then apply :func:`as_of` to produce a
    pd.Series indexed by observation_date with the latest-known
    value as of ``vintage_date``.
    """
    df = store.get(series_id, vintage_until=vintage_date)
    if df.empty:
        return _pd.Series(dtype="float64")
    return as_of(
        df, vintage_date,
        date_col="observation_date",
        vintage_col="vintage_date",
        value_col="value",
    )
