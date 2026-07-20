"""Driscoll-Kraay (1998) panel HAC: cross-sectional sum then Newey-West."""
from __future__ import annotations

import numpy as np


def driscoll_kraay(score: np.ndarray, time_keys: np.ndarray, lags: int) -> np.ndarray:
    """Compute the meat of the Driscoll-Kraay sandwich.

    Given the score = X * residuals (T_panel x k), aggregate by time, then
    apply Newey-West with bandwidth `lags`. Returns the (k x k) meat S of
    the sandwich; caller wraps it with (X'X)^-1 S (X'X)^-1.

    Parameters
    ----------
    score : np.ndarray (T_panel, k)
        Per-observation moment conditions (X_it * u_it).
    time_keys : np.ndarray (T_panel,)
        Time index for each row of score; rows with the same value are
        summed before applying Newey-West.
    lags : int
        Bartlett kernel bandwidth.
    """
    if len(score) == 0:
        return np.zeros((score.shape[1], score.shape[1]))
    T_unique = np.unique(time_keys)
    by_t = np.array([
        score[time_keys == t].sum(axis=0) for t in T_unique
    ])  # (T, k)
    S = by_t.T @ by_t  # ell=0
    T = len(by_t)
    for ell in range(1, lags + 1):
        if ell >= T:
            break
        w = 1.0 - ell / (lags + 1.0)
        Gamma = by_t[ell:].T @ by_t[:-ell]
        S += w * (Gamma + Gamma.T)
    return S


__all__ = ["driscoll_kraay"]
