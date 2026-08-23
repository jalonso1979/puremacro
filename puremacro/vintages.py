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


__all__ = [
    "as_of",
    "align_vintages",
    "forecast_revision",
    "AlfredVintageStore",
    "as_of_from_store",
    "QNAVintagePanel",
    "fetch_qna_vintages",
    "get_qna_vintage_catalog",
    # Revision econometrics (1.7.0)
    "revision_triangle",
    "revision_frame",
    "first_release",
    "latest_release",
    "mankiw_shapiro",
    "MankiwShapiroResult",
    "revision_test",
]


def __getattr__(name: str):
    if name in {"QNAVintagePanel", "fetch_qna_vintages", "get_qna_vintage_catalog"}:
        from .fetch.qna_vintages import (
            QNAVintagePanel as _QNAVintagePanel,
            fetch_qna_vintages as _fetch_qna_vintages,
            get_qna_vintage_catalog as _get_qna_vintage_catalog,
        )
        mapping = {
            "QNAVintagePanel": _QNAVintagePanel,
            "fetch_qna_vintages": _fetch_qna_vintages,
            "get_qna_vintage_catalog": _get_qna_vintage_catalog,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ── AlfredVintageStore (0.66.0+) ────────────────────────────────────
# Persistent local store for FRED-ALFRED vintage observations, backed
# by the shared SQLite cache DB (see puremacro._cache_db). Existing
# in-memory helpers (as_of, align_vintages, forecast_revision) are
# untouched; this class adds the store-backed counterpart so research
# notebooks don't refetch ALFRED on every kernel restart.

import sqlite3 as _sqlite3
import warnings as _warnings
from pathlib import Path as _Path

from dataclasses import dataclass as _dataclass

import numpy as _np
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


# ── Revision econometrics: news vs noise (1.7.0) ────────────────────
#
# Mankiw & Shapiro (1986), "News or Noise? An Analysis of GNP
# Revisions", Survey of Current Business 66(5), 20-25.
#
# Two competing accounts of why a published macro aggregate gets
# revised, and the regression that separates them. Write y_p for the
# preliminary (first-published) estimate of some quarter, y_f for the
# final (latest-available) one, and r = y_f - y_p for the revision.
#
#   NEWS  — the statistical office publishes an efficient forecast of
#           the truth given the information it had. Then y_p is a
#           conditional expectation, the revision is the forecast
#           error, and r is orthogonal to y_p.
#             => regress r on y_p, test beta = 0.
#
#   NOISE — the published number is the truth plus independent
#           measurement error, y_p = y_f + u with u orthogonal to y_f.
#           Then r = -u is orthogonal to y_f.
#             => regress r on y_f, test beta = 0.
#
# The two regressions are both run and both reported, because failing
# to reject one is only informative alongside the other. A common
# reading error is to test the y_p regression alone and call a
# significantly negative beta "noise". Under pure noise that beta is
#   -sigma_u^2 / (sigma_f^2 + sigma_u^2),
# which lies strictly inside (-1, 0) and equals -1 only in the
# degenerate case sigma_f^2 = 0 -- so its magnitude identifies the
# noise share, not the hypothesis. The verdict below therefore comes
# from the pair of tests, never from one coefficient's size.
#
# UNITS MATTER. Real GDP *levels* are I(1) and trending, so regressing
# a revision on a level is a spurious regression: the estimated beta
# is driven by the common trend, not by the revision process. The
# literature runs these tests on growth rates, and so does the default
# here (``transform="log_diff_pct"``). ``transform="level"`` is
# available for series that are genuinely stationary (rates, shares)
# and warns otherwise.

_REVISION_TRANSFORMS: dict[str, str] = {
    "level": "published level, untransformed",
    "diff": "first difference of the published level",
    "log_diff_pct": "100 x change in log level (approximately % growth)",
    "pct_change": "% change on the previous period",
    "pct_change_ann": "% change on the previous period, annualised",
    "yoy": "% change on the same period one year earlier",
}


def _apply_revision_transform(
    wide: _pd.DataFrame,
    transform: str,
    periods_per_year: int = 4,
) -> _pd.DataFrame:
    """Apply ``transform`` down the date axis of a (date x vintage) frame.

    The transform is applied **within** each vintage column, never
    across columns. That is the whole point: the growth rate a
    forecaster saw in vintage v is built from vintage v's own level
    for t and for t-1. Mixing a numerator from one vintage with a
    denominator from another manufactures revisions that nobody ever
    published.
    """
    if transform not in _REVISION_TRANSFORMS:
        raise ValueError(
            f"unknown transform {transform!r}; available: "
            f"{sorted(_REVISION_TRANSFORMS)}"
        )
    if wide.empty:
        return wide
    w = wide.sort_index()
    if transform == "level":
        return w
    if transform == "diff":
        return w.diff()
    if transform == "log_diff_pct":
        nonpos = int((w <= 0).to_numpy().sum())
        if nonpos:
            _warnings.warn(
                f"log_diff_pct dropped {nonpos} non-positive value(s): the "
                "log transform is undefined there. Series that legitimately "
                "go negative or through zero (net exports, the current "
                "account, a balance) should use transform='diff' or "
                "'pct_change' instead -- otherwise this silently returns a "
                "sparser, or entirely empty, revision sample.",
                UserWarning, stacklevel=3,
            )
        positive = w.where(w > 0)
        return 100.0 * _np.log(positive).diff()
    if transform == "pct_change":
        return 100.0 * w.pct_change(fill_method=None)
    if transform == "pct_change_ann":
        ratio = w / w.shift(1)
        return 100.0 * (ratio.where(ratio > 0) ** periods_per_year - 1.0)
    # yoy
    return 100.0 * w.pct_change(periods_per_year, fill_method=None)


def revision_triangle(
    panel_long: _pd.DataFrame,
    *,
    transform: str = "level",
    periods_per_year: int = 4,
    carry_forward: bool = False,
    date_col: str = "date",
    vintage_col: str = "vintage",
    value_col: str = "value",
) -> _pd.DataFrame:
    """Build the (reference date x vintage) revision triangle.

    Row ``t``, column ``v`` is the value that edition ``v`` published
    for period ``t``; NaN means that edition did not carry ``t``.

    ``carry_forward`` chooses between two genuinely different objects,
    and the default matters:

    ``False`` (default) -- each column is **one edition, and only that
        edition**. This is what a revision is defined against, and it
        is the only construction under which a within-column transform
        is meaningful.
    ``True`` -- real-time as-of semantics (:func:`align_vintages`): a
        period a later edition dropped keeps its last published value,
        so the column reflects everything a researcher knew on date
        ``v``. Right for reconstructing an information set, **wrong for
        differencing**, because a column is then a patchwork of
        editions.

    That distinction is not academic. Editions truncate their own
    history: the OECD's 2005 snapshot of Spain starts later than its
    1999 one, so under as-of semantics the 2005 column holds 1980Q1 in
    euros next to 1979Q4 still carried from the 1999 edition in
    pesetas. Differencing down that column yields a quarterly "growth
    rate" of -594%. With ``carry_forward=False`` the cell is simply
    NaN, which is the truth: that edition published no such growth
    rate.

    ``transform`` is applied within each column; see
    :data:`_REVISION_TRANSFORMS` for the menu.
    """
    if panel_long is None or len(panel_long) == 0:
        return _pd.DataFrame()
    if carry_forward:
        vints = sorted(_pd.to_datetime(panel_long[vintage_col]).unique())
        wide = align_vintages(
            panel_long, vints,
            date_col=date_col, vintage_col=vintage_col, value_col=value_col,
        )
    else:
        wide = panel_long.pivot_table(
            index=date_col, columns=vintage_col, values=value_col,
            aggfunc="last",
        ).sort_index()
        wide = wide.reindex(columns=sorted(wide.columns))
        wide.columns.name = "vintage"
    return _apply_revision_transform(wide, transform, periods_per_year)


def _kth_edition(row: _pd.Series, k: int = 0) -> float:
    """The k-th *distinct successive* value in a triangle row.

    Not the k-th column. Triangle columns carry as-of values, so an
    edition that did not restate a period repeats the previous
    edition's number; counting columns would return the same estimate
    several times over and call them different releases.
    """
    vals = row.dropna().to_numpy()
    if vals.size == 0:
        return _np.nan
    distinct = [vals[0]]
    for v in vals[1:]:
        if v != distinct[-1]:
            distinct.append(v)
    return distinct[k] if len(distinct) > k else _np.nan


def _first_last_from_triangle(
    tri: _pd.DataFrame,
    *,
    require_observable_first: bool = True,
) -> tuple[_pd.Series, _pd.Series]:
    """Return (first published, latest published) per row of a triangle.

    ``require_observable_first`` censors the rows where the *first
    release* cannot be observed at all. A reference period is published
    after it ends, so if the period ends before the archive's earliest
    edition, that edition already carries a revised number — and taking
    the earliest available column as the "first release" understates
    every revision computed from it.

    This bites hard in practice: the OECD archive begins in 1999 but
    carries reference periods from 1980, and the ONS workbook begins in
    1961 with periods from 1955. Without this guard roughly two decades
    of each panel silently report a near-final estimate as the initial
    one.
    """
    if tri.empty:
        return _pd.Series(dtype=float), _pd.Series(dtype=float)
    ordered = tri.reindex(columns=sorted(tri.columns))
    first = ordered.apply(_kth_edition, axis=1, k=0)
    last = ordered.apply(
        lambda row: row.dropna().iloc[-1] if row.notna().any() else _np.nan,
        axis=1,
    )
    if require_observable_first and len(ordered.columns):
        earliest = min(ordered.columns)
        first = first.where(_pd.Index(ordered.index) >= earliest, _np.nan)
    first.name, last.name = "preliminary", "final"
    return first, last


def revision_frame(
    panel_long: _pd.DataFrame,
    *,
    transform: str = "log_diff_pct",
    periods_per_year: int = 4,
    release: int = 0,
    require_observable_first: bool = True,
    carry_forward: bool = False,
    date_col: str = "date",
    vintage_col: str = "vintage",
    value_col: str = "value",
) -> _pd.DataFrame:
    """Reduce a real-time panel to ``[preliminary, final, revision]``.

    ``revision`` is ``final - preliminary`` -- the quantity Mankiw and
    Shapiro call ``r_t``, and the ``r_t = y_f - y_p`` the issue asks
    for.

    Parameters
    ----------
    release : which release counts as "preliminary". ``0`` (default) is
        the first published estimate; ``1`` the second, and so on. Use
        a later release to skip a flash estimate that a country
        publishes on a different schedule from its peers.
    transform : see :data:`_REVISION_TRANSFORMS`. The default is
        ``"log_diff_pct"`` (100 x dlog), because revisions to I(1)
        *levels* regressed on levels are spurious -- see the module
        note above.
    require_observable_first : drop reference periods that ended before
        the archive's earliest edition, whose first release therefore
        cannot be observed. On by default; setting it False restores
        the naive "earliest column available" reading, which
        understates revisions for those periods.

    Returns
    -------
    pd.DataFrame indexed by reference date, columns
    ``["preliminary", "final", "revision"]``, rows with either side
    missing dropped.
    """
    if release < 0:
        raise ValueError(f"release must be >= 0, got {release}")
    tri = revision_triangle(
        panel_long,
        transform=transform, periods_per_year=periods_per_year,
        carry_forward=carry_forward,
        date_col=date_col, vintage_col=vintage_col, value_col=value_col,
    )
    if tri.empty:
        return _pd.DataFrame(columns=["preliminary", "final", "revision"])
    prelim, final = _first_last_from_triangle(
        tri, require_observable_first=require_observable_first)
    if release > 0:
        ordered = tri.reindex(columns=sorted(tri.columns))
        prelim = ordered.apply(_kth_edition, axis=1, k=release)
        prelim.name = "preliminary"
        if require_observable_first and len(ordered.columns):
            earliest = min(ordered.columns)
            prelim = prelim.where(_pd.Index(ordered.index) >= earliest, _np.nan)
    out = _pd.concat([prelim, final], axis=1)
    out["revision"] = out["final"] - out["preliminary"]
    return out.dropna(subset=["preliminary", "final"])


def first_release(
    panel_long: _pd.DataFrame,
    *,
    transform: str = "level",
    periods_per_year: int = 4,
    date_col: str = "date",
    vintage_col: str = "vintage",
    value_col: str = "value",
) -> _pd.Series:
    """The first published estimate of each reference period."""
    tri = revision_triangle(
        panel_long, transform=transform, periods_per_year=periods_per_year,
        date_col=date_col, vintage_col=vintage_col, value_col=value_col,
    )
    return _first_last_from_triangle(tri)[0]


def latest_release(
    panel_long: _pd.DataFrame,
    *,
    transform: str = "level",
    periods_per_year: int = 4,
    date_col: str = "date",
    vintage_col: str = "vintage",
    value_col: str = "value",
) -> _pd.Series:
    """The most recent estimate of each reference period in the panel."""
    tri = revision_triangle(
        panel_long, transform=transform, periods_per_year=periods_per_year,
        date_col=date_col, vintage_col=vintage_col, value_col=value_col,
    )
    return _first_last_from_triangle(tri)[1]


@_dataclass(frozen=True)
class MankiwShapiroResult:
    """Outcome of the news-vs-noise test pair on one revision sample.

    Both regressions share the dependent variable ``r = final -
    preliminary``. They differ in the regressor, and each one tests a
    different hypothesis:

    ``beta_on_preliminary``
        from ``r = a + b * preliminary``. Tests **news**: under an
        efficient preliminary estimate the revision is a forecast
        error, orthogonal to what was published, so ``b = 0``.
    ``beta_on_final``
        from ``r = a + b * final``. Tests **noise**: under
        ``preliminary = final + u`` with ``u`` orthogonal to the
        truth, ``r = -u`` is orthogonal to ``final``, so ``b = 0``.

    ``noise_share`` is ``max(0, -beta_on_preliminary)``: under the pure
    noise model that coefficient equals
    ``-var(u) / var(preliminary)``, so its negative is the share of
    the preliminary estimate's variance attributable to measurement
    error. It is a magnitude, not a verdict -- read it only alongside
    the two p-values.
    """
    n_obs: int
    transform: str
    hac_lags: int
    significance: float
    mean_revision: float
    se_mean_revision: float
    t_mean_revision: float
    p_mean_revision: float
    std_revision: float
    alpha_on_preliminary: float
    beta_on_preliminary: float
    se_beta_on_preliminary: float
    t_beta_on_preliminary: float
    p_beta_on_preliminary: float
    alpha_on_final: float
    beta_on_final: float
    se_beta_on_final: float
    t_beta_on_final: float
    p_beta_on_final: float
    noise_share: float
    rejects_news: bool
    rejects_noise: bool
    verdict: str


def _nw_auto_lags(n: int) -> int:
    """Newey-West plug-in bandwidth, floor(4 (n/100)^(2/9))."""
    if n <= 1:
        return 0
    return int(_np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def _hac_slope(y: _np.ndarray, x: _np.ndarray, lags: int) -> tuple[float, float, float, float]:
    """OLS of y on [1, x] with HAC SE. Returns (alpha, beta, se_beta, t_beta)."""
    from .inference._ols_helpers import ols_hac
    X = _np.column_stack([_np.ones(len(x)), x])
    fit = ols_hac(y, X, lags)
    return (
        float(fit["beta"][0]), float(fit["beta"][1]),
        float(fit["se"][1]), float(fit["t"][1]),
    )


def mankiw_shapiro(
    preliminary,
    final,
    *,
    hac_lags: "int | str" = 0,
    significance: float = 0.05,
    transform: str = "(pre-transformed)",
) -> MankiwShapiroResult:
    """Run the Mankiw-Shapiro (1986) news-vs-noise test pair.

    Parameters
    ----------
    preliminary, final : array-like or pd.Series of equal length. If
        both are Series they are inner-joined on their index first, so
        passing two differently-sampled series is safe. **Pass growth
        rates, not I(1) levels** -- see the module note; a revision
        regressed on a trending level is a spurious regression.
    hac_lags : Newey-West bandwidth. ``0`` (default) gives
        heteroskedasticity-robust (White) errors, appropriate when
        revisions are serially uncorrelated. Pass ``"auto"`` for the
        ``floor(4 (n/100)^(2/9))`` plug-in rule, or an int. Annual
        benchmark revisions induce serial correlation in ``r``, so
        ``"auto"`` is the safer choice on long samples.
    significance : level for the two rejection flags and ``verdict``.

    Returns
    -------
    MankiwShapiroResult

    Raises
    ------
    ValueError
        If fewer than 3 usable observations remain after alignment --
        the regression is not identified and silently returning NaN
        would read as a result.
    """
    if isinstance(preliminary, _pd.Series) and isinstance(final, _pd.Series):
        joined = _pd.concat(
            [preliminary.rename("preliminary"), final.rename("final")],
            axis=1, join="inner",
        ).dropna()
        y_p = joined["preliminary"].to_numpy(dtype=float)
        y_f = joined["final"].to_numpy(dtype=float)
    else:
        y_p = _np.asarray(preliminary, dtype=float).reshape(-1)
        y_f = _np.asarray(final, dtype=float).reshape(-1)
        if y_p.shape != y_f.shape:
            raise ValueError(
                f"preliminary and final must have the same length; got "
                f"{y_p.shape[0]} and {y_f.shape[0]}"
            )
        keep = _np.isfinite(y_p) & _np.isfinite(y_f)
        y_p, y_f = y_p[keep], y_f[keep]

    n = int(y_p.size)
    if n < 3:
        raise ValueError(
            f"mankiw_shapiro needs at least 3 aligned observations, got {n}. "
            "The slope is not identified on fewer; check that the vintage "
            "panel actually contains more than one edition per period."
        )

    lags = _nw_auto_lags(n) if hac_lags == "auto" else int(hac_lags)
    if lags < 0:
        raise ValueError(f"hac_lags must be >= 0, got {lags}")

    r = y_f - y_p
    if not _np.any(r != 0.0):
        raise ValueError(
            "every revision is exactly zero, so the news/noise regressions "
            "are degenerate. The usual cause is a panel carrying a single "
            "edition per period -- check `n_vintages` in coverage(); with "
            "one edition the preliminary and final estimates are the same "
            "number by construction and there is nothing to test."
        )

    from scipy import stats as _stats
    from .inference._ols_helpers import ols_hac

    # Bias test: is the mean revision zero? (regression of r on a constant)
    const_fit = ols_hac(r, _np.ones((n, 1)), lags)
    mean_rev = float(const_fit["beta"][0])
    se_mean = float(const_fit["se"][0])
    t_mean = float(const_fit["t"][0])
    p_mean = float(2.0 * _stats.t.sf(abs(t_mean), df=max(1, n - 1)))

    a_p, b_p, se_p, t_p = _hac_slope(r, y_p, lags)
    a_f, b_f, se_f, t_f = _hac_slope(r, y_f, lags)
    df_slope = max(1, n - 2)
    p_p = float(2.0 * _stats.t.sf(abs(t_p), df=df_slope))
    p_f = float(2.0 * _stats.t.sf(abs(t_f), df=df_slope))

    rejects_news = bool(p_p < significance)
    rejects_noise = bool(p_f < significance)
    if not rejects_news and rejects_noise:
        verdict = "news"
    elif rejects_news and not rejects_noise:
        verdict = "noise"
    elif not rejects_news and not rejects_noise:
        verdict = "indeterminate"
    else:
        verdict = "neither"

    return MankiwShapiroResult(
        n_obs=n,
        transform=transform,
        hac_lags=lags,
        significance=float(significance),
        mean_revision=mean_rev,
        se_mean_revision=se_mean,
        t_mean_revision=t_mean,
        p_mean_revision=p_mean,
        std_revision=float(_np.std(r, ddof=1)) if n > 1 else float("nan"),
        alpha_on_preliminary=a_p,
        beta_on_preliminary=b_p,
        se_beta_on_preliminary=se_p,
        t_beta_on_preliminary=t_p,
        p_beta_on_preliminary=p_p,
        alpha_on_final=a_f,
        beta_on_final=b_f,
        se_beta_on_final=se_f,
        t_beta_on_final=t_f,
        p_beta_on_final=p_f,
        noise_share=float(max(0.0, -b_p)),
        rejects_news=rejects_news,
        rejects_noise=rejects_noise,
        verdict=verdict,
    )


def revision_test(
    panel_long: _pd.DataFrame,
    *,
    transform: str = "log_diff_pct",
    periods_per_year: int = 4,
    release: int = 0,
    hac_lags: "int | str" = 0,
    significance: float = 0.05,
    date_col: str = "date",
    vintage_col: str = "vintage",
    value_col: str = "value",
) -> MankiwShapiroResult:
    """Go from a real-time panel straight to the news-vs-noise verdict.

    Convenience wrapper: :func:`revision_frame` then
    :func:`mankiw_shapiro`. This is the one-liner the teaching notebook
    wants::

        rev = fetch.vintage_panel(["USA", "DEU", "ESP"])
        res = revision_test(rev.long("DEU"))
        print(res.beta_on_preliminary, res.verdict)
    """
    frame = revision_frame(
        panel_long,
        transform=transform, periods_per_year=periods_per_year,
        release=release,
        date_col=date_col, vintage_col=vintage_col, value_col=value_col,
    )
    return mankiw_shapiro(
        frame["preliminary"], frame["final"],
        hac_lags=hac_lags, significance=significance, transform=transform,
    )
