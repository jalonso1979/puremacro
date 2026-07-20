"""Time-domain trend-cycle decompositions.

Frequency-domain tools (Welch periodogram, cross-spectrum, coherence) live in
:mod:`puremacro.spectral`; this module is for time-domain regression-based
decompositions.

Currently provides:
    - :func:`hamilton_filter` — Hamilton (2018) regression filter.

Future expansion (tracked for 0.5.0+): Christiano-Fitzgerald, Baxter-King,
Beveridge-Nelson decomposition.
"""
from __future__ import annotations

import numpy as np


def hamilton_filter(
    y,
    h: int = 8,
    p: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Hamilton (2018) regression filter for trend-cycle decomposition.

    Projects ``y_{t+h}`` on a constant and ``(y_t, y_{t-1}, ..., y_{t-p+1})``
    via OLS. The fitted value is the trend; the residual is the cycle. This
    replaces the HP filter, which Hamilton (2018) shows produces spurious
    dynamics and end-of-sample distortions.

    Parameters
    ----------
    y : array_like, shape (T,)
        Time series. ``pandas.Series`` is accepted; index is ignored.
    h : int, default 8
        Forecast horizon. Default 8 is Hamilton's quarterly convention
        (project 2 years ahead from the most recent year).
    p : int, default 4
        Number of lags on the right-hand side. Default 4 is Hamilton's
        quarterly convention (1 year of lags).

    Returns
    -------
    cycle : ndarray, shape (T,)
        Cyclical component. The first ``h + p - 1`` entries are ``NaN`` —
        the regression has no value at those positions.
    trend : ndarray, shape (T,)
        Trend / projection component. Same NaN convention as cycle.

    Notes
    -----
    Convention: ``cycle[t] + trend[t] == y[t]`` for all ``t >= h + p - 1``.

    References
    ----------
    Hamilton, J.D. (2018). Why you should never use the Hodrick-Prescott
        filter. Review of Economics and Statistics 100(5), 831-843.
    """
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    T = y_arr.shape[0]
    lags_needed = h + p
    if T < lags_needed:
        raise ValueError(
            f"hamilton_filter: input length {T} is shorter than h + p = {lags_needed}; "
            f"need at least {lags_needed} observations."
        )
    n_obs = T - h - (p - 1)
    # X[:, 0] = 1; X[:, 1+j] = y at lag j relative to t (j=0,...,p-1)
    # Row i corresponds to t = (p - 1) + i, target y[t + h] = y[(p - 1) + i + h]
    X = np.empty((n_obs, p + 1))
    X[:, 0] = 1.0
    for j in range(p):
        X[:, 1 + j] = y_arr[(p - 1 - j):(T - h - j)]
    y_target = y_arr[(h + p - 1):T]
    # Use Moore-Penrose pseudoinverse: degenerate inputs (constant series,
    # exact deterministic trend) produce collinear lag columns and a singular
    # X'X. The pseudoinverse gives the minimum-norm OLS solution, which still
    # produces the correct cycle=0 result on those inputs. This is also what
    # Hamilton's reference Stata implementation does.
    beta = np.linalg.pinv(X) @ y_target
    fitted = X @ beta
    residual = y_target - fitted
    cycle = np.full(T, np.nan)
    trend = np.full(T, np.nan)
    cycle[h + p - 1:] = residual
    trend[h + p - 1:] = fitted
    return cycle, trend
