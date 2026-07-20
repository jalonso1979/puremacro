"""Tests for forecast-combination and probabilistic-scoring helpers."""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.nowcast import (
    equal_weight, inverse_mse, bates_granger, rank_weight,
    model_confidence_set,
    crps_gaussian, crps_ensemble,
    log_score_gaussian, brier_score, pit_histogram,
)


# ---------------------------------------------------------------------------
# Combinations
# ---------------------------------------------------------------------------


def _toy_forecasts(rng, T=200, M=4, true_var=1.0):
    """Three forecasters with increasing accuracy + a noisy outlier."""
    y = rng.standard_normal(T) * np.sqrt(true_var)
    F = np.empty((T, M))
    F[:, 0] = y + 0.1 * rng.standard_normal(T)   # very accurate
    F[:, 1] = y + 0.4 * rng.standard_normal(T)
    F[:, 2] = y + 0.9 * rng.standard_normal(T)
    F[:, 3] = y + 3.0 * rng.standard_normal(T)   # noisy outlier
    return F, y


def test_equal_weight_is_simple_mean(rng):
    F, y = _toy_forecasts(rng)
    out = equal_weight(F, y)
    np.testing.assert_allclose(out["weights"], 0.25 * np.ones(4))
    np.testing.assert_allclose(out["combined"], F.mean(axis=1))


def test_inverse_mse_puts_more_weight_on_accurate(rng):
    F, y = _toy_forecasts(rng)
    out = inverse_mse(F, y)
    # Forecaster 0 (most accurate) should get the largest weight.
    assert out["weights"][0] == out["weights"].max()
    np.testing.assert_allclose(out["weights"].sum(), 1.0)


def test_bates_granger_weights_sum_to_one(rng):
    F, y = _toy_forecasts(rng)
    out = bates_granger(F, y)
    np.testing.assert_allclose(out["weights"].sum(), 1.0, atol=1e-8)


def test_rank_weight_orders_correctly(rng):
    F, y = _toy_forecasts(rng)
    out = rank_weight(F, y)
    # Rank 1 (most accurate) gets the highest inverse-rank weight.
    assert out["weights"][0] == out["weights"].max()
    assert out["weights"][3] == out["weights"].min()


def test_model_confidence_set_drops_outlier(rng):
    F, y = _toy_forecasts(rng)
    out = model_confidence_set(F, y, alpha=0.20)
    # The outlier (forecaster 3) has MSE >> all others; should be dropped.
    assert out["weights"][3] == 0.0
    assert out["kept"] >= 1


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_crps_gaussian_zero_when_perfect():
    """For a degenerate forecast σ → 0 with correct μ, CRPS → 0."""
    y = np.array([1.0, 2.0])
    mu = np.array([1.0, 2.0])
    sigma = np.full(2, 1e-10)
    crps = crps_gaussian(y, mu, sigma)
    assert np.all(crps < 1e-6)


def test_crps_gaussian_sigma_only():
    """When y = μ exactly, CRPS reduces to σ · (2 φ(0) - 1/√π)
    = σ · (√(2/π) - 1/√π). Test this identity."""
    y = np.array([0.0])
    mu = np.array([0.0])
    sigma = np.array([2.0])
    crps = crps_gaussian(y, mu, sigma)
    expected = 2.0 * (np.sqrt(2.0 / np.pi) - 1.0 / np.sqrt(np.pi))
    assert abs(float(crps[0]) - expected) < 1e-12


def test_crps_ensemble_matches_gaussian_in_large_M(rng):
    """For a Gaussian forecast, ensemble CRPS converges to closed-form
    CRPS as M → ∞."""
    T, M = 5, 5000
    mu = rng.standard_normal(T)
    sigma = np.full(T, 1.0)
    y = rng.standard_normal(T)
    ensemble = mu[:, None] + sigma[:, None] * rng.standard_normal((T, M))
    crps_e = crps_ensemble(y, ensemble)
    crps_g = crps_gaussian(y, mu, sigma)
    np.testing.assert_allclose(crps_e, crps_g, rtol=0.10, atol=0.05)


def test_log_score_gaussian_proper(rng):
    """The expected log score of N(μ, σ²) under itself equals
    -0.5 log(2π σ²) - 0.5 (theoretical entropy)."""
    sigma = 1.5
    samples = sigma * rng.standard_normal(50000)
    ls = log_score_gaussian(samples, np.zeros_like(samples), np.full_like(samples, sigma))
    # The empirical mean log-density under the data-generating process.
    expected = -0.5 * np.log(2 * np.pi * sigma ** 2) - 0.5
    assert abs(ls.mean() - expected) < 0.05


def test_brier_score_perfect_zero():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    p = y.copy()
    assert np.all(brier_score(y, p) == 0.0)


def test_brier_score_uniform_random_about_quarter():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    p = np.array([0.5, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(brier_score(y, p), 0.25)


def test_pit_histogram_uniform_under_correct_calibration(rng):
    T, M = 1000, 200
    mu = rng.standard_normal(T)
    sigma = np.full(T, 1.0)
    ensemble = mu[:, None] + sigma[:, None] * rng.standard_normal((T, M))
    y = mu + sigma * rng.standard_normal(T)
    out = pit_histogram(y, ensemble, n_bins=10)
    # Histogram should be approximately uniform — coefficient of variation
    # of bin counts under the null is ~ 1/√(T/n_bins) = 1/10.
    assert out["pit"].shape == (T,)
    assert out["hist"].shape == (10,)
    # Each bin has expected count T/n_bins = 100; observed mean across bins
    # always equals 100 by construction. We check the spread is reasonable.
    cv = float(out["hist"].std()) / float(out["hist"].mean())
    assert cv < 0.30
