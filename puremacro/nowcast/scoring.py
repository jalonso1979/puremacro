"""Probabilistic-forecast scoring rules.

Companion to ``puremacro.forecast.compare`` (DM / GW for point
forecasts). The functions here score *predictive distributions* —
either parametric (mean + std) or sample-based (ensemble of draws).

Implemented:
  ``crps_gaussian``    — closed-form CRPS for a Normal(μ, σ) forecast.
  ``crps_ensemble``    — sample-based CRPS via the Gneiting-Raftery
                          (2007) ensemble formula.
  ``log_score_gaussian`` — log-pdf score for a Normal forecast.
  ``brier_score``      — squared-error of probability forecasts of a
                          binary event.
  ``pit_histogram``    — Probability Integral Transform diagnostic;
                          uniform under correct calibration.

References
----------
Gneiting, T. and Raftery, A.E. (2007). Strictly proper scoring rules,
    prediction, and estimation. JASA 102(477), 359-378.
Diebold, F.X., Gunther, T.A. and Tay, A.S. (1998). Evaluating density
    forecasts with applications to financial risk management. IER 39(4).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def crps_gaussian(
    realised: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
) -> np.ndarray:
    """Closed-form CRPS when the predictive distribution is N(μ, σ²).

    ``CRPS(F, y) = σ · ( z (2 Φ(z) - 1) + 2 φ(z) - 1/√π )``,
    ``z = (y - μ) / σ``.

    Parameters
    ----------
    realised : ndarray
        Realised values ``y``.
    mu : ndarray
        Predictive means.
    sigma : ndarray
        Predictive standard deviations.

    Returns
    -------
    ndarray
        Per-observation CRPS values (same shape as the broadcast inputs).
        Smaller is better.
    """
    realised = np.asarray(realised, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (realised - mu) / sigma_safe
    return sigma_safe * (z * (2 * norm.cdf(z) - 1)
                          + 2 * norm.pdf(z)
                          - 1.0 / np.sqrt(np.pi))


def crps_ensemble(realised: np.ndarray, ensemble: np.ndarray) -> np.ndarray:
    """Sample-based CRPS for an ensemble of forecast draws.

    For each observation t,

        ``CRPS(F̂_t, y_t) = (1/M) Σ_i |x_i - y_t|
                            - 1/(2 M²) Σ_i Σ_j |x_i - x_j|``.

    Parameters
    ----------
    realised : (T,) ndarray
        Observed values.
    ensemble : (T, M) ndarray
        ``M`` forecast draws per period.

    Returns
    -------
    ndarray
        ``(T,)`` array of per-period CRPS values. Smaller is better.
    """
    realised = np.asarray(realised, dtype=float).ravel()
    ensemble = np.asarray(ensemble, dtype=float)
    T, M = ensemble.shape
    crps = np.empty(T)
    for t in range(T):
        x = ensemble[t]
        crps[t] = (np.abs(x - realised[t]).mean()
                    - 0.5 * np.abs(x[:, None] - x[None, :]).mean())
    return crps


def log_score_gaussian(
    realised: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
) -> np.ndarray:
    """Per-period log predictive density of a Normal forecast.

    Parameters
    ----------
    realised : ndarray
        Realised values.
    mu : ndarray
        Predictive means.
    sigma : ndarray
        Predictive standard deviations.

    Returns
    -------
    ndarray
        ``log f(y_t | μ_t, σ_t)`` for each ``t`` (larger is better).
    """
    realised = np.asarray(realised, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma_safe = np.maximum(sigma, 1e-12)
    return norm.logpdf(realised, loc=mu, scale=sigma_safe)


def brier_score(
    realised_binary: np.ndarray, predicted_prob: np.ndarray,
) -> np.ndarray:
    """Brier score for binary-event probability forecasts.

    ``BS_t = (p_t - y_t)²``, ``y_t ∈ {0, 1}``.

    Parameters
    ----------
    realised_binary : ndarray
        Observed 0/1 outcomes.
    predicted_prob : ndarray
        Forecast probabilities of the event.

    Returns
    -------
    ndarray
        ``(T,)`` array of per-period Brier scores. Smaller is better.
    """
    y = np.asarray(realised_binary, dtype=float).ravel()
    p = np.asarray(predicted_prob, dtype=float).ravel()
    return (p - y) ** 2


def pit_histogram(
    realised: np.ndarray, ensemble: np.ndarray, *, n_bins: int = 10,
) -> dict:
    """Probability Integral Transform diagnostic for ensemble forecasts.

    Computes per-period ``PIT_t = (1/M) Σ 𝟙{x_i ≤ y_t}``
    (Diebold-Gunther-Tay 1998). Under correct calibration, the histogram
    of PITs across ``t`` is uniform on ``[0, 1]``. Departures (U-shape
    ⇒ underdispersed, inverted-U ⇒ overdispersed) flag mis-specification.

    Parameters
    ----------
    realised : (T,) ndarray
        Observed values.
    ensemble : (T, M) ndarray
        ``M`` forecast draws per period.
    n_bins : int, default 10
        Number of histogram bins on ``[0, 1]``.

    Returns
    -------
    dict
        ``{"pit": (T,), "hist": (n_bins,), "edges": (n_bins + 1,)}``.
    """
    realised = np.asarray(realised, dtype=float).ravel()
    ensemble = np.asarray(ensemble, dtype=float)
    T = ensemble.shape[0]
    pit = np.empty(T)
    for t in range(T):
        pit[t] = float((ensemble[t] <= realised[t]).mean())
    hist, edges = np.histogram(pit, bins=n_bins, range=(0.0, 1.0))
    return {"pit": pit, "hist": hist, "edges": edges}


__all__ = [
    "crps_gaussian", "crps_ensemble",
    "log_score_gaussian", "brier_score",
    "pit_histogram",
]
