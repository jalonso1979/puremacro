"""Time-domain trend-cycle decompositions.

Provides:
    - :func:`hamilton_filter` — Hamilton (2018) regression filter.
    - :func:`baxter_king_filter` — Baxter & King (1999) bandpass filter.
    - :func:`christiano_fitzgerald_filter` — Christiano & Fitzgerald (2003) asymmetric bandpass filter.
    - :func:`beveridge_nelson_filter` — Beveridge & Nelson (1981) permanent-transitory decomposition.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CycleResult:
    """Result of a time-domain trend-cycle decomposition.

    Implements tuple unpacking ``cycle, trend = res`` for seamless
    backward-compatibility.

    Attributes
    ----------
    cycle : np.ndarray | pd.Series
        Estimated cyclical component.
    trend : np.ndarray | pd.Series
        Estimated trend / permanent component.
    method : str
        Decomposition method ('hamilton', 'baxter_king', 'christiano_fitzgerald', 'beveridge_nelson').
    params : dict
        Parameters supplied to the filter.
    """

    cycle: np.ndarray | pd.Series
    trend: np.ndarray | pd.Series
    method: str
    params: dict

    def __iter__(self):
        return iter((self.cycle, self.trend))

    def __getitem__(self, idx: int):
        if idx == 0:
            return self.cycle
        elif idx == 1:
            return self.trend
        raise IndexError("CycleResult only has index 0 (cycle) and 1 (trend).")

    def __len__(self) -> int:
        return 2

    def plot(self, *, title: str = "", ax=None):
        """Plot cyclical and trend components.

        Lazily creates a 2-panel figure showing the cycle and trend.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, (ax_cyc, ax_tr) = plt.subplots(2, 1, figsize=(7, 4.5), sharex=True)
        else:
            fig = ax.figure
            ax_cyc = ax
            ax_tr = None

        c = self.cycle
        t = self.trend
        idx = getattr(c, "index", np.arange(len(c)))

        ax_cyc.plot(idx, c, color="0.0", lw=1.3, label="Cycle")
        ax_cyc.axhline(0, color="0.6", ls="--", lw=0.8)
        ax_cyc.set_ylabel("Cycle")
        ax_cyc.set_title(title or f"Decomposition ({self.method})")
        ax_cyc.legend(loc="best", frameon=False)

        if ax_tr is not None:
            ax_tr.plot(idx, t, color="0.2", lw=1.3, label="Trend")
            ax_tr.set_ylabel("Trend")
            ax_tr.legend(loc="best", frameon=False)
            fig.tight_layout()

        return fig

    def summary(self) -> str:
        n_obs = len(self.cycle)
        valid = int(np.sum(~np.isnan(np.asarray(self.cycle))))
        return (
            f"CycleResult ({self.method})\n"
            f"  total observations : {n_obs}\n"
            f"  valid cycle values : {valid}\n"
            f"  parameters         : {self.params}\n"
        )


def hamilton_filter(
    y,
    h: int = 8,
    p: int = 4,
    *,
    horizon: int | None = None,
    lags: int | None = None,
) -> CycleResult:
    """Hamilton (2018) regression filter for trend-cycle decomposition.

    Projects ``y_{t+h}`` on a constant and ``(y_t, y_{t-1}, ..., y_{t-p+1})``
    via OLS. The fitted value is the trend; the residual is the cycle. This
    replaces the HP filter, which Hamilton (2018) shows produces spurious
    dynamics and end-of-sample distortions.

    Parameters
    ----------
    y : array_like, shape (T,)
        Time series. ``pandas.Series`` is accepted; index is preserved if Series.
    h : int, default 8
        Forecast horizon. Default 8 is Hamilton's quarterly convention
        (project 2 years ahead from the most recent year).
    p : int, default 4
        Number of lags on the right-hand side. Default 4 is Hamilton's
        quarterly convention (1 year of lags).
    horizon : int, optional
        Standardized alias for ``h``.
    lags : int, optional
        Standardized alias for ``p``.

    Returns
    -------
    CycleResult
        Dataclass unpacking to ``(cycle, trend)`` with ``.cycle``, ``.trend``,
        ``.summary()``, and ``.plot()``.
    """
    if horizon is not None:
        h = horizon
    if lags is not None:
        p = lags

    is_series = isinstance(y, pd.Series)
    idx = y.index if is_series else None
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    T = y_arr.shape[0]
    lags_needed = h + p
    if T < lags_needed + 1:
        raise ValueError(
            f"hamilton_filter: sample size {T} is too short for h={h}, p={p} "
            f"(need at least {lags_needed + 1})."
        )

    # Build the design matrix: constant + p lags of y.
    # Row t in the regression predicts y[t + h] from (1, y[t], ..., y[t - p + 1]).
    # Usable baseline indices run from t = p - 1 to t = T - h - 1.
    n_reg = T - h - p + 1
    X = np.empty((n_reg, p + 1))
    X[:, 0] = 1.0
    for j in range(p):
        X[:, 1 + j] = y_arr[(p - 1 - j):(T - h - j)]

    # Target: y_{t+h}, matching the row indexing above
    y_target = y_arr[(h + p - 1):T]

    beta = np.linalg.pinv(X) @ y_target
    fitted = X @ beta
    residual = y_target - fitted
    cycle = np.full(T, np.nan)
    trend = np.full(T, np.nan)
    cycle[h + p - 1:] = residual
    trend[h + p - 1:] = fitted
    if is_series and idx is not None:
        c_out, t_out = pd.Series(cycle, index=idx), pd.Series(trend, index=idx)
    else:
        c_out, t_out = cycle, trend
    return CycleResult(
        cycle=c_out,
        trend=t_out,
        method="hamilton",
        params={"h": int(h), "p": int(p)},
    )


def baxter_king_filter(
    y,
    low: int = 6,
    high: int = 32,
    K: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Baxter-King (1999) bandpass filter for isolating business-cycle fluctuations.

    Constructs a symmetric approximation of the ideal bandpass filter with lead-lag
    truncation parameter ``K`` and zero-frequency gain constraint.

    Parameters
    ----------
    y : array_like, shape (T,)
        Time series.
    low : int, default 6
        Minimum periodicity of the cycle (e.g. 6 quarters = 1.5 years).
    high : int, default 32
        Maximum periodicity of the cycle (e.g. 32 quarters = 8 years).
    K : int, default 12
        Lead-lag truncation order (default 12 for quarterly data = 3 years).

    Returns
    -------
    cycle : ndarray or pd.Series, shape (T,)
        Cyclical component with ``K`` leading and trailing NaNs.
    trend : ndarray or pd.Series, shape (T,)
        Trend / non-cyclical component (``y - cycle``).

    References
    ----------
    Baxter, M. and King, R.G. (1999). Measuring business cycles: approximate
        band-pass filters for economic time series. Review of Economics and
        Statistics 81(4), 575-593.
    """
    is_series = isinstance(y, pd.Series)
    idx = y.index if is_series else None
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    T = len(y_arr)
    if T < 2 * K + 1:
        raise ValueError(
            f"baxter_king_filter: sample size {T} is too short for K={K} (need >= {2*K + 1})."
        )
    if low <= 1 or high <= low:
        raise ValueError(f"baxter_king_filter: require 1 < low < high (got low={low}, high={high}).")

    w1 = 2.0 * np.pi / high
    w2 = 2.0 * np.pi / low

    a = np.empty(K + 1)
    a[0] = (w2 - w1) / np.pi
    k_vec = np.arange(1, K + 1)
    a[1:] = (np.sin(w2 * k_vec) - np.sin(w1 * k_vec)) / (np.pi * k_vec)

    total_sum = a[0] + 2.0 * np.sum(a[1:])
    theta = total_sum / (2 * K + 1)
    a_hat = a - theta

    cycle = np.full(T, np.nan)
    weights = np.concatenate([a_hat[:0:-1], [a_hat[0]], a_hat[1:]])
    for t in range(K, T - K):
        chunk = y_arr[t - K : t + K + 1]
        cycle[t] = np.dot(weights, chunk)

    trend = np.full(T, np.nan)
    valid = ~np.isnan(cycle)
    trend[valid] = y_arr[valid] - cycle[valid]
    if is_series and idx is not None:
        c_out, t_out = pd.Series(cycle, index=idx), pd.Series(trend, index=idx)
    else:
        c_out, t_out = cycle, trend
    return CycleResult(
        cycle=c_out,
        trend=t_out,
        method="baxter_king",
        params={"low": int(low), "high": int(high), "K": int(K)},
    )


def christiano_fitzgerald_filter(
    y,
    low: int = 6,
    high: int = 32,
    drift: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Christiano-Fitzgerald (2003) asymmetric bandpass filter.

    Computes optimal asymmetric finite-sample approximations to the ideal bandpass
    filter under a random-walk plus drift data-generating process. Unlike Baxter-King,
    no observations are lost at the sample endpoints.

    Parameters
    ----------
    y : array_like, shape (T,)
        Time series.
    low : int, default 6
        Minimum periodicity of the cycle (e.g. 6 quarters).
    high : int, default 32
        Maximum periodicity of the cycle (e.g. 32 quarters).
    drift : bool, default True
        If True, detrend with deterministic linear drift prior to filtering.

    Returns
    -------
    cycle : ndarray or pd.Series, shape (T,)
        Cyclical component for all ``T`` observations (no NaNs).
    trend : ndarray or pd.Series, shape (T,)
        Trend component (``y - cycle``).

    References
    ----------
    Christiano, L.J. and Fitzgerald, T.J. (2003). The band pass filter.
        International Economic Review 44(2), 435-465.
    """
    is_series = isinstance(y, pd.Series)
    idx = y.index if is_series else None
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    T = len(y_arr)
    if T < 4:
        raise ValueError(f"christiano_fitzgerald_filter: sample size {T} too short (need >= 4).")
    if low <= 1 or high <= low:
        raise ValueError(f"christiano_fitzgerald_filter: require 1 < low < high (got low={low}, high={high}).")

    if drift:
        slope = (y_arr[-1] - y_arr[0]) / (T - 1)
        drift_trend = y_arr[0] + slope * np.arange(T)
        y_star = y_arr - drift_trend
    else:
        y_star = y_arr - np.mean(y_arr)

    w1 = 2.0 * np.pi / high
    w2 = 2.0 * np.pi / low

    b0 = (w2 - w1) / np.pi
    j_vec = np.arange(1, T + 1)
    b = np.empty(T + 1)
    b[0] = b0
    b[1:] = (np.sin(w2 * j_vec) - np.sin(w1 * j_vec)) / (np.pi * j_vec)

    cycle = np.empty(T)
    for t in range(T):
        k_past = t
        k_fut = T - 1 - t
        val = b0 * y_star[t]
        if k_past > 0:
            val += np.dot(b[1 : k_past + 1], y_star[t - 1 : : -1])
            A_t = -0.5 * b0 - np.sum(b[1 : k_past + 1])
            val += A_t * y_star[0]
        else:
            A_t = -0.5 * b0
            val += A_t * y_star[0]

        if k_fut > 0:
            val += np.dot(b[1 : k_fut + 1], y_star[t + 1 :])
            B_t = -0.5 * b0 - np.sum(b[1 : k_fut + 1])
            val += B_t * y_star[-1]
        else:
            B_t = -0.5 * b0
            val += B_t * y_star[-1]

        cycle[t] = val

    trend = y_arr - cycle
    if is_series and idx is not None:
        c_out, t_out = pd.Series(cycle, index=idx), pd.Series(trend, index=idx)
    else:
        c_out, t_out = cycle, trend
    return CycleResult(
        cycle=c_out,
        trend=t_out,
        method="christiano_fitzgerald",
        params={"low": int(low), "high": int(high), "drift": bool(drift)},
    )


def beveridge_nelson_filter(
    y,
    p: int = 4,
    *,
    lags: int | None = None,
) -> CycleResult:
    """Beveridge-Nelson (1981) permanent-transitory decomposition.

    Decomposes an I(1) series into a random walk with drift (the permanent component)
    and a stationary cyclical component via AR(p) forecasting of first differences.

    Parameters
    ----------
    y : array_like, shape (T,)
        Integrated time series.
    p : int, default 4
        Autoregressive lag order for the first-differenced series.
    lags : int, optional
        Standardized alias for ``p``.

    Returns
    -------
    CycleResult
        Dataclass unpacking to ``(cycle, trend)`` with ``.cycle``, ``.trend``,
        ``.summary()``, and ``.plot()``.

    References
    ----------
    Beveridge, S. and Nelson, C.R. (1981). A new approach to decomposition of
        economic time series into permanent and transitory components with
        particular attention to measurement of the business cycle. Journal of
        Monetary Economics 7(2), 151-174.
    """
    if lags is not None:
        p = lags

    is_series = isinstance(y, pd.Series)
    idx = y.index if is_series else None
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    T = len(y_arr)
    if T < p + 4:
        raise ValueError(
            f"beveridge_nelson_filter: sample size {T} is too short for AR({p}) (need >= {p+4})."
        )

    dy = np.diff(y_arr)
    N = len(dy)
    X = np.empty((N - p, p + 1))
    X[:, 0] = 1.0
    for j in range(p):
        X[:, 1 + j] = dy[p - 1 - j : N - 1 - j]
    y_target = dy[p:]

    beta = np.linalg.lstsq(X, y_target, rcond=None)[0]
    mu_const = beta[0]
    phi = beta[1:]

    F = np.zeros((p, p))
    F[0, :] = phi
    if p > 1:
        F[1:, :-1] = np.eye(p - 1)

    eigvals = np.linalg.eigvals(F)
    max_eig = np.max(np.abs(eigvals))
    if max_eig >= 1.0:
        F *= 0.99 / max_eig

    I = np.eye(p)
    M = F @ np.linalg.inv(I - F)
    e1_M = M[0, :]

    denom = 1.0 - np.sum(phi)
    mu = mu_const / denom if abs(denom) > 1e-4 else np.mean(dy)

    cycle = np.full(T, np.nan)
    for t in range(p, T):
        dy_state = dy[t - p : t][::-1] - mu
        c_t = -float(np.dot(e1_M, dy_state))
        cycle[t] = c_t

    trend = np.full(T, np.nan)
    valid = ~np.isnan(cycle)
    trend[valid] = y_arr[valid] - cycle[valid]
    if is_series and idx is not None:
        c_out, t_out = pd.Series(cycle, index=idx), pd.Series(trend, index=idx)
    else:
        c_out, t_out = cycle, trend
    return CycleResult(
        cycle=c_out,
        trend=t_out,
        method="beveridge_nelson",
        params={"p": int(p)},
    )


def hp_filter(
    y,
    lamb: float = 1600.0,
) -> CycleResult:
    """Hodrick-Prescott (1997) filter returning :class:`CycleResult`.

    Parameters
    ----------
    y : array_like or pd.Series
        Time series data.
    lamb : float, default 1600.0
        Smoothing parameter (1600 for quarterly, 6.25 for annual, 129600 for monthly).

    Returns
    -------
    CycleResult
        Dataclass with ``cycle``, ``trend``, ``method='hp'``, and ``params={'lamb': lamb}``.
    """
    from .data import hp_filter as _data_hp_filter
    c, t = _data_hp_filter(y, lamb=lamb)
    is_series = isinstance(y, pd.Series)
    cycle_out = c if is_series else np.asarray(c)
    trend_out = t if is_series else np.asarray(t)
    return CycleResult(cycle=cycle_out, trend=trend_out, method="hp", params={"lamb": float(lamb)})


__all__ = [
    "CycleResult",
    "hamilton_filter",
    "baxter_king_filter",
    "christiano_fitzgerald_filter",
    "beveridge_nelson_filter",
    "hp_filter",
]
