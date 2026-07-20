"""Tests for BEKK / CCC GARCH and ARCH-LM / Ljung-Box diagnostics."""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.volatility import (
    bekk_fit, ccc_fit, arch_lm_test, ljung_box_squared,
)


def _simulate_garch(T, omega, alpha, beta, rng):
    """Simple GARCH(1,1) DGP for one series."""
    h = np.empty(T); u = np.empty(T)
    h[0] = omega / max(1.0 - alpha - beta, 1e-3)
    u[0] = np.sqrt(h[0]) * rng.standard_normal()
    for t in range(1, T):
        h[t] = omega + alpha * u[t - 1] ** 2 + beta * h[t - 1]
        u[t] = np.sqrt(h[t]) * rng.standard_normal()
    return u


def test_ccc_fit_documented_keys(rng):
    n, T = 2, 500
    U = np.column_stack([
        _simulate_garch(T, 0.05, 0.10, 0.85, rng),
        _simulate_garch(T, 0.05, 0.10, 0.85, rng),
    ])
    out = ccc_fit(U)
    for k in ("omega", "alpha", "beta", "R", "H", "loglik"):
        assert k in out
    assert out["R"].shape == (n, n)
    assert out["H"].shape == (T, n, n)
    np.testing.assert_allclose(np.diag(out["R"]), 1.0, atol=1e-10)


def test_ccc_fit_recovers_persistence(rng):
    T = 800
    U = np.column_stack([
        _simulate_garch(T, 0.05, 0.10, 0.85, rng),
        _simulate_garch(T, 0.05, 0.10, 0.85, rng),
    ])
    out = ccc_fit(U)
    # Per-equation α + β should land near the true 0.95 persistence
    # within QMLE tolerance.
    assert all(0.85 < a + b < 1.0 for a, b in zip(out["alpha"], out["beta"]))


def test_bekk_fit_documented_keys(rng):
    """BEKK has a hairy parameter space, but the documented output
    schema should always be present even on a short series."""
    n, T = 2, 250
    U = np.column_stack([
        _simulate_garch(T, 0.05, 0.10, 0.85, rng),
        _simulate_garch(T, 0.05, 0.10, 0.85, rng),
    ])
    out = bekk_fit(U, max_iter=80)
    for k in ("C", "A", "B", "H", "loglik", "converged"):
        assert k in out
    assert out["H"].shape == (T, n, n)


def test_arch_lm_detects_arch(rng):
    """ARCH-LM should reject H0 when ARCH effects are present."""
    u = _simulate_garch(800, 0.05, 0.20, 0.75, rng)
    out = arch_lm_test(u, q=5)
    assert out["p_value"] < 0.01


def test_arch_lm_does_not_falsely_reject_iid(rng):
    """Pure Gaussian noise has no ARCH effect; the LM test should
    not reject at α = 1% with high probability."""
    u = rng.standard_normal(1000)
    out = arch_lm_test(u, q=5)
    assert out["p_value"] > 0.01


def test_ljung_box_squared_detects_remaining_arch(rng):
    """A naive constant-volatility model (residuals = raw returns)
    leaves all ARCH structure untouched; Ljung-Box on the squared
    residuals should reject."""
    u = _simulate_garch(800, 0.05, 0.20, 0.75, rng)
    out = ljung_box_squared(u, lags=10)
    assert out["p_value"] < 0.01


def test_ljung_box_squared_documented_keys(rng):
    out = ljung_box_squared(rng.standard_normal(200), lags=5)
    for k in ("q_stat", "p_value", "df"):
        assert k in out
