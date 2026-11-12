"""Shared keyword and input-contract helpers for ``puremacro.lp`` estimators.

Every estimator in :mod:`puremacro.lp` accepts the canonical arguments
``horizons`` / ``n_lags`` / ``alpha`` **and** the 2.0 aliases
``horizon`` / ``lags`` / ``ci``.  :func:`resolve_lp_kwargs` applies the
aliases, rejects conflicting canonical + alias pairs, and validates the
result so that wrong scales (``ci=90`` instead of ``ci=0.90``,
``alpha=1.5``) or ambiguous overrides raise immediately.
"""
from __future__ import annotations

import warnings
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd


class _DefaultInt(int):
    """Sentinel ``int`` subclass distinguishing omitted default from explicit int."""

    def __repr__(self) -> str:
        return int.__repr__(self)


class _DefaultFloat(float):
    """Sentinel ``float`` subclass distinguishing omitted default from explicit float."""

    def __repr__(self) -> str:
        return float.__repr__(self)


class _DefaultHorizons:
    """Default ``range(0, stop)`` wrapper preserving ``Iterable[int]`` and ``repr``."""

    def __init__(self, stop: int = 21) -> None:
        self._r = range(0, stop)

    def __iter__(self) -> Iterator[int]:
        return iter(self._r)

    def __len__(self) -> int:
        return len(self._r)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, range):
            return self._r == other
        if isinstance(other, _DefaultHorizons):
            return self._r == other._r
        return NotImplemented

    def __repr__(self) -> str:
        return repr(self._r)


_DEFAULT_HORIZONS_21: Iterable[int] = _DefaultHorizons(21)
_DEFAULT_HORIZONS_13: Iterable[int] = _DefaultHorizons(13)
_DEFAULT_N_LAGS_2: int = _DefaultInt(2)
_DEFAULT_N_LAGS_4: int = _DefaultInt(4)
_DEFAULT_N_LAGS_0: int = _DefaultInt(0)
_DEFAULT_ALPHA_10: float = _DefaultFloat(0.10)


def _is_int_like(v: object) -> bool:
    if isinstance(v, bool):
        return False
    try:
        return float(v).is_integer()  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def resolve_lp_kwargs(
    horizons: Iterable[int],
    n_lags: int,
    alpha: float,
    *,
    lags: int | None = None,
    horizon: int | None = None,
    ci: float | None = None,
    name: str = "lp",
    strict_aliases: bool = False,
    default_horizons: Sequence[int] | range = range(0, 21),
    default_n_lags: int = 2,
    default_alpha: float = 0.10,
) -> tuple[list[int], int, float]:
    """Apply the ``lags`` / ``horizon`` / ``ci`` aliases and validate.

    Parameters
    ----------
    horizons, n_lags, alpha
        Canonical arguments as received by the estimator.
    lags
        Alias for ``n_lags``. Raises ``ValueError`` if both ``n_lags`` and
        ``lags`` are explicitly supplied with different values.
    horizon
        Maximum horizon ``H``; sets ``horizons = range(0, H + 1)``. Raises
        ``ValueError`` if both ``horizons`` and ``horizon`` are explicitly
        supplied with disagreeing horizon grids.
    ci
        Coverage of the confidence bands in ``(0, 1)``; sets
        ``alpha = 1 - ci``. Raises ``ValueError`` if both ``alpha`` and
        ``ci`` are explicitly supplied with ``alpha != 1 - ci``.
    name
        Estimator name used in error messages.
    strict_aliases
        True when the caller uses ``_DefaultHorizons`` / ``_DefaultInt`` /
        ``_DefaultFloat`` defaults so even explicit default-matching values
        (e.g. ``n_lags=2, lags=4``) are detected as conflicts.

    Returns
    -------
    (horizons, n_lags, alpha)
        ``horizons`` as a list of ints, ``n_lags`` as an int and
        ``alpha`` as a float in ``(0, 1)``.
    """
    horizons_explicit = bool(strict_aliases) and (type(horizons) is not _DefaultHorizons)
    n_lags_explicit = bool(strict_aliases) and (type(n_lags) is not _DefaultInt)
    alpha_explicit = bool(strict_aliases) and (type(alpha) is not _DefaultFloat)

    # Validate canonical values when explicitly supplied (or before alias override)
    if not _is_int_like(n_lags) or int(n_lags) < 0:
        raise ValueError(
            f"{name}: lags must be a non-negative integer; got {n_lags!r}"
        )
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(
            f"{name}: alpha must be a two-sided significance level in (0, 1), "
            f"e.g. alpha=0.10 for 90% bands; got {alpha!r}"
        )
    hs_canon = [int(h) for h in horizons]

    if lags is not None:
        if not _is_int_like(lags) or int(lags) < 0:
            raise ValueError(
                f"{name}: lags must be a non-negative integer; got {lags!r}"
            )
        if n_lags_explicit and int(n_lags) != int(lags):
            raise ValueError(
                f"{name}: received conflicting lag arguments n_lags={int(n_lags)} "
                f"and lags={int(lags)}; pass only one."
            )
        n_lags = int(lags)

    if horizon is not None:
        if not _is_int_like(horizon) or int(horizon) < 0:
            raise ValueError(
                f"{name}: horizon must be a non-negative integer (the maximum "
                f"horizon H, estimating h = 0..H); got {horizon!r}"
            )
        alias_hs = list(range(0, int(horizon) + 1))
        if horizons_explicit and hs_canon != alias_hs:
            raise ValueError(
                f"{name}: received conflicting horizon arguments "
                f"horizons={hs_canon!r} and horizon={int(horizon)}; pass only one."
            )
        hs_canon = alias_hs

    if ci is not None:
        if not (0.0 < float(ci) < 1.0):
            raise ValueError(
                f"{name}: ci must be a coverage probability in (0, 1), e.g. "
                f"ci=0.90 for 90% bands; got {ci!r}"
            )
        implied_alpha = 1.0 - float(ci)
        if alpha_explicit and abs(float(alpha) - implied_alpha) > 1e-12:
            raise ValueError(
                f"{name}: received conflicting confidence arguments "
                f"alpha={float(alpha)} and ci={float(ci)} "
                f"(implies alpha={implied_alpha:.6g}); pass only one."
            )
        alpha = implied_alpha

    return hs_canon, int(n_lags), float(alpha)


__all__ = ["resolve_lp_kwargs"]


def cumulative_windows(
    y: pd.Series, x: pd.Series, h: int
) -> tuple[pd.Series, pd.Series]:
    """Window sums for cumulative (integral) multipliers.

    Returns ``(Y_w, X_w)`` on the input index, where
    ``Y_w[t] = sum_{j=0..h} (y[t+j] - y[t-1])`` and
    ``X_w[t] = sum_{j=0..h} x[t+j]``.  Built by explicit shift-and-add so a
    missing value inside a window makes exactly the rows touching it NaN
    (``cumsum`` would skip it) and ``h = 0`` equals ``y.shift(0) - y.shift(1)``
    bit for bit (``rolling`` would accumulate rounding error).
    """
    y_sum = sum(y.shift(-j) for j in range(h + 1))
    x_sum = sum(x.shift(-j) for j in range(h + 1))
    return y_sum - (h + 1) * y.shift(1), x_sum


def _fmt_ts(ts: object) -> str:
    if isinstance(ts, pd.Timestamp):
        return str(ts.date()) if ts == ts.normalize() else str(ts)
    return str(ts)


def format_sample_bounds(index: pd.Index) -> tuple[str | int | None, str | int | None]:
    """Return ``(start, end)`` labels for an estimation sample index."""
    if len(index) == 0:
        return None, None
    if isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex)):
        return _fmt_ts(index[0]), _fmt_ts(index[-1])
    try:
        return int(index[0]), int(index[-1])
    except (TypeError, ValueError):
        return str(index[0]), str(index[-1])


def validate_lp_columns_finite(
    df: pd.DataFrame, cols: Sequence[str], func: str
) -> None:
    """Validate that required LP columns exist, are numeric 1-D series, and contain no ±inf."""
    if df.columns.has_duplicates:
        dups = [str(c) for c in df.columns[df.columns.duplicated()].unique()]
        raise ValueError(
            f"{func}: DataFrame has duplicate column names {dups!r}; "
            "rename or deduplicate columns before estimation."
        )
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"{func}: column(s) {missing!r} not found in DataFrame "
            f"(available: {list(df.columns)!r})."
        )
    for col in dict.fromkeys(cols):
        ser = df[col]
        try:
            vals = np.asarray(ser, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{func}: column {col!r} must be numeric; could not convert to float."
            ) from exc
        if vals.ndim != 1:
            raise ValueError(
                f"{func}: column {col!r} must be 1-dimensional, got shape {vals.shape}."
            )
        if np.isinf(vals).any():
            bad_pos = int(np.flatnonzero(np.isinf(vals))[0])
            bad_idx = _fmt_ts(df.index[bad_pos])
            raise ValueError(
                f"{func}: column {col!r} contains non-finite (±inf) value at "
                f"index {bad_idx}; replace ±inf with NaN or finite values before estimation."
            )


def check_regular_index(
    index: pd.Index,
    func: str,
    *,
    cumulative: bool = True,
    allow_gaps: bool = False,
) -> None:
    """Raise ``ValueError`` if a date or period index violates chronological regularity.

    Checks ``DatetimeIndex`` and ``PeriodIndex`` inputs of length ``>= 2``:
    - Duplicate dates/periods always raise ``ValueError``.
    - Non-increasing (unsorted or reversed) dates/periods always raise
      ``ValueError`` with an explicit ``df.sort_index()`` hint.
    - Irregular spacing (skipped periods on ``len(index) >= 3``) raises
      ``ValueError`` unless ``allow_gaps=True`` and ``cumulative=False``.
    Plain positional indexes (``RangeIndex``, integer indexes) are not checked.
    """
    if not isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex)):
        return
    if index.isna().any():
        raise ValueError(
            f"{func}: dated index contains NaT/missing timestamps; drop or fix "
            "missing index labels before estimation."
        )
    if len(index) < 2:
        return

    def _fail_gap(i: int) -> None:
        prev_s = _fmt_ts(index[i])
        next_s = _fmt_ts(index[i + 1])
        if cumulative:
            raise ValueError(
                f"{func}: cumulative=True needs a regularly spaced index, but "
                f"{prev_s} is followed by {next_s}; "
                "to exclude a period, set y, x and the instrument to NaN there "
                "and keep the rows"
            )
        raise ValueError(
            f"{func}: dated index has an irregular gap ({prev_s} followed by "
            f"{next_s}); positional lag/horizon shifts would bridge the missing "
            "period(s). Set excluded periods to NaN while keeping the rows, or "
            "pass allow_index_gaps=True if lag/horizon columns were pre-constructed."
        )

    def _fail_order(i: int, is_dup: bool) -> None:
        prev_s = _fmt_ts(index[i])
        next_s = _fmt_ts(index[i + 1])
        if cumulative:
            _fail_gap(i)
        if is_dup:
            raise ValueError(
                f"{func}: dated index contains duplicate period/timestamp {prev_s!r} "
                "at adjacent rows; deduplicate or aggregate before estimation."
            )
        raise ValueError(
            f"{func}: dated index must be strictly increasing in chronological order, "
            f"but {prev_s} is followed by {next_s}; sort explicitly with "
            "df = df.sort_index() before estimation."
        )

    raw_steps = np.diff(index.asi8)
    non_inc = np.flatnonzero(raw_steps <= 0)
    if non_inc.size:
        idx0 = int(non_inc[0])
        _fail_order(idx0, bool(raw_steps[idx0] == 0))

    if len(index) < 3 or (allow_gaps and not cumulative):
        return

    if isinstance(index, pd.PeriodIndex):
        bad = np.flatnonzero(raw_steps != 1)
    else:
        # pandas >= 3.1 warns that infer_freq will return a BaseOffset; only None vs not-None is used here
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inferred = pd.infer_freq(index)
        if inferred is not None:
            return
        naive = index.tz_localize(None) if index.tz is not None else index
        steps = np.diff(naive.to_period("M").asi8)
        if int(pd.Series(steps).mode().iloc[0]) < 1:  # sub-monthly data: use days
            steps = np.diff(naive.to_period("D").asi8)
        modal = int(pd.Series(steps).mode().iloc[0])
        bad = np.flatnonzero(steps != modal)
    if bad.size:
        _fail_gap(int(bad[0]))
