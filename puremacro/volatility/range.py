"""Range-based volatility estimators.

When tick data isn't available but daily open / high / low / close are,
range-based estimators recover most of the information realised
volatility would extract from intraday returns. They're 4–14× more
efficient than the close-to-close estimator depending on the variant.

Estimators implemented (canonical normalisations under zero-drift
geometric Brownian motion):

  Parkinson (1980):
      σ²_P = (1 / (4 ln 2)) · (ln(H/L))²

  Garman-Klass (1980):
      σ²_GK = 0.5 · (ln(H/L))² − (2 ln 2 − 1) · (ln(C/O))²

  Rogers-Satchell (1991), drift-robust:
      σ²_RS = ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)

All return per-period variances; pass `np.sqrt(...)` to recover σ.
Inputs may be 1-D (single asset) or 2-D ``(T, n)`` panels.

References
----------
Parkinson, M. (1980). The extreme value method for estimating the
    variance of the rate of return. JoB 53(1), 61-65.
Garman, M. and Klass, M. (1980). On the estimation of security price
    volatilities from historical data. JoB 53(1), 67-78.
Rogers, L.C.G. and Satchell, S.E. (1991). Estimating variance from
    high, low and closing prices. Annals of Applied Probability 1(4),
    504-512.
"""
from __future__ import annotations

import numpy as np


_LN2 = np.log(2.0)


def parkinson(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Parkinson (1980) high-low range estimator.

    Parameters
    ----------
    high, low : ndarray
        Period-by-period high and low prices. May be 1-D ``(T,)`` or
        2-D ``(T, n)``.

    Returns
    -------
    ndarray
        Per-period variance estimates with the same shape as the
        inputs.

    References
    ----------
    Parkinson, M. (1980). The extreme value method for estimating the
    variance of the rate of return. JoB 53(1), 61-65.
    """
    H = np.asarray(high, dtype=float)
    L = np.asarray(low, dtype=float)
    return (np.log(H / L) ** 2) / (4.0 * _LN2)


def garman_klass(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
) -> np.ndarray:
    """Garman-Klass (1980) OHLC variance estimator.

    Parameters
    ----------
    open_, high, low, close : ndarray
        Period-by-period open / high / low / close prices. May be 1-D
        ``(T,)`` or 2-D ``(T, n)``; all four must broadcast.

    Returns
    -------
    ndarray
        Per-period variance estimates with the same shape as the
        inputs.

    References
    ----------
    Garman, M. and Klass, M. (1980). On the estimation of security
    price volatilities from historical data. JoB 53(1), 67-78.
    """
    O = np.asarray(open_, dtype=float)
    H = np.asarray(high, dtype=float)
    L = np.asarray(low, dtype=float)
    C = np.asarray(close, dtype=float)
    return 0.5 * np.log(H / L) ** 2 - (2.0 * _LN2 - 1.0) * np.log(C / O) ** 2


def rogers_satchell(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
) -> np.ndarray:
    """Rogers-Satchell (1991) drift-robust OHLC variance estimator.

    Unlike Parkinson and Garman-Klass, this remains unbiased under a
    non-zero drift in log-prices.

    Parameters
    ----------
    open_, high, low, close : ndarray
        Period-by-period open / high / low / close prices. May be 1-D
        ``(T,)`` or 2-D ``(T, n)``; all four must broadcast.

    Returns
    -------
    ndarray
        Per-period variance estimates with the same shape as the
        inputs.

    References
    ----------
    Rogers, L.C.G. and Satchell, S.E. (1991). Estimating variance from
    high, low and closing prices. Annals of Applied Probability 1(4),
    504-512.
    """
    O = np.asarray(open_, dtype=float)
    H = np.asarray(high, dtype=float)
    L = np.asarray(low, dtype=float)
    C = np.asarray(close, dtype=float)
    return (np.log(H / C) * np.log(H / O)
            + np.log(L / C) * np.log(L / O))


__all__ = ["parkinson", "garman_klass", "rogers_satchell"]
