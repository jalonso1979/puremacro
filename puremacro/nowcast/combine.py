"""Forecast combinations.

Five combination rules, each takes a (T, M) panel of forecasts and
returns a (T,) combined forecast plus the (M,) weights:

  ``equal_weight``        — w = 1/M.
  ``inverse_mse``         — w_m ∝ 1 / MSE_m.
  ``bates_granger``       — w solves min Var(w' e) subject to w'1 = 1,
                              given the empirical error-covariance matrix
                              (Bates-Granger 1969 minimum-variance combo).
  ``rank_weight``         — weight by inverse rank (Stock-Watson 2004
                              "trimmed-mean" approximation).
  ``model_confidence_set`` — equal-weight over the Hansen-Lunde-Nason
                              (2011) confidence set of statistically-best
                              models. (Implemented as a thin filter over
                              MSE here, not the full elimination test.)

Each helper accepts forecasts and realised values and returns
``{ "combined": (T,), "weights": (M,), "method": str }``.

References
----------
Bates, J.M. and Granger, C.W.J. (1969). The combination of forecasts.
    Operational Research Quarterly 20(4), 451-468.
Stock, J.H. and Watson, M.W. (2004). Combination forecasts of output
    growth in a seven-country data set. Journal of Forecasting 23.
"""
from __future__ import annotations

import numpy as np


def _errors(forecasts: np.ndarray, realised: np.ndarray) -> np.ndarray:
    e = forecasts - realised[:, None]
    return e


def equal_weight(forecasts: np.ndarray, realised: np.ndarray) -> dict:
    """Simple average across models. Always a useful benchmark.

    Parameters
    ----------
    forecasts : (T, M) ndarray
        Panel of forecasts: ``T`` periods × ``M`` models.
    realised : (T,) ndarray
        Realised values of the target. Unused for this rule but kept in
        the signature for a uniform combiner API.

    Returns
    -------
    dict
        ``{"combined": (T,), "weights": (M,), "method": "equal_weight"}``.
    """
    forecasts = np.asarray(forecasts, dtype=float)
    M = forecasts.shape[1]
    w = np.ones(M) / M
    combined = forecasts @ w
    return {"combined": combined, "weights": w, "method": "equal_weight"}


def inverse_mse(forecasts: np.ndarray, realised: np.ndarray) -> dict:
    """Inverse-MSE weighting (Granger-Ramanathan style).

    ``w_m ∝ 1 / MSE_m`` where ``MSE_m`` is the in-sample mean squared
    error of model ``m``.

    Parameters
    ----------
    forecasts : (T, M) ndarray
        Panel of forecasts.
    realised : (T,) ndarray
        Realised values of the target.

    Returns
    -------
    dict
        ``{"combined": (T,), "weights": (M,), "method": "inverse_mse"}``.
    """
    forecasts = np.asarray(forecasts, dtype=float)
    realised = np.asarray(realised, dtype=float).ravel()
    e = _errors(forecasts, realised)
    mse = (e ** 2).mean(axis=0)
    inv = 1.0 / np.maximum(mse, 1e-12)
    w = inv / inv.sum()
    return {"combined": forecasts @ w, "weights": w, "method": "inverse_mse"}


def bates_granger(
    forecasts: np.ndarray, realised: np.ndarray, *, ridge: float = 1e-10,
) -> dict:
    """Bates-Granger (1969) minimum-variance combination.

    Solves ``w = (1' Σ_e^{-1} 1)^{-1} Σ_e^{-1} 1`` where ``Σ_e`` is the
    empirical error covariance. A small ridge keeps ``Σ_e``
    well-conditioned when forecasts are highly correlated (almost
    always the case in practice).

    Parameters
    ----------
    forecasts : (T, M) ndarray
        Panel of forecasts.
    realised : (T,) ndarray
        Realised values of the target.
    ridge : float, default 1e-10
        Diagonal loading added to ``Σ_e`` for numerical stability.

    Returns
    -------
    dict
        ``{"combined": (T,), "weights": (M,), "method": "bates_granger"}``.
    """
    forecasts = np.asarray(forecasts, dtype=float)
    realised = np.asarray(realised, dtype=float).ravel()
    M = forecasts.shape[1]
    e = _errors(forecasts, realised)
    Sigma_e = (e.T @ e) / e.shape[0] + ridge * np.eye(M)
    ones = np.ones(M)
    Sinv_ones = np.linalg.solve(Sigma_e, ones)
    w = Sinv_ones / float(ones @ Sinv_ones)
    return {"combined": forecasts @ w, "weights": w, "method": "bates_granger"}


def rank_weight(forecasts: np.ndarray, realised: np.ndarray) -> dict:
    """Inverse-rank weighting (Stock-Watson 2004 trimmed-mean cousin).

    Models are ranked by in-sample MSE; weights are proportional to
    ``1 / rank``.

    Parameters
    ----------
    forecasts : (T, M) ndarray
        Panel of forecasts.
    realised : (T,) ndarray
        Realised values of the target.

    Returns
    -------
    dict
        ``{"combined": (T,), "weights": (M,), "method": "rank_weight"}``.
    """
    forecasts = np.asarray(forecasts, dtype=float)
    realised = np.asarray(realised, dtype=float).ravel()
    mse = ((forecasts - realised[:, None]) ** 2).mean(axis=0)
    order = np.argsort(mse)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    inv_rank = 1.0 / ranks
    w = inv_rank / inv_rank.sum()
    return {"combined": forecasts @ w, "weights": w, "method": "rank_weight"}


def model_confidence_set(
    forecasts: np.ndarray, realised: np.ndarray, *, alpha: float = 0.10,
) -> dict:
    """Equal-weight average over a thresholded model confidence set.

    Keeps models whose MSE is within ``(1 + alpha)`` of the best and
    averages those equally. Lightweight stand-in for the
    Hansen-Lunde-Nason (2011) elimination procedure; useful when you
    want a robust subset average without running the full bootstrap.

    Parameters
    ----------
    forecasts : (T, M) ndarray
        Panel of forecasts.
    realised : (T,) ndarray
        Realised values of the target.
    alpha : float, default 0.10
        Tolerance above the best MSE — keep model ``m`` iff
        ``MSE_m ≤ (1 + alpha) · min_m MSE_m``.

    Returns
    -------
    dict
        ``{"combined": (T,), "weights": (M,), "method":
        "model_confidence_set", "kept": int}``.
    """
    forecasts = np.asarray(forecasts, dtype=float)
    realised = np.asarray(realised, dtype=float).ravel()
    mse = ((forecasts - realised[:, None]) ** 2).mean(axis=0)
    threshold = (1.0 + alpha) * mse.min()
    keep = mse <= threshold
    w = np.zeros_like(mse)
    w[keep] = 1.0 / keep.sum()
    return {"combined": forecasts @ w, "weights": w,
             "method": "model_confidence_set", "kept": int(keep.sum())}


__all__ = [
    "equal_weight", "inverse_mse", "bates_granger",
    "rank_weight", "model_confidence_set",
]
