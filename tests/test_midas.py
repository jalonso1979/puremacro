"""Tests for puremacro.midas result objects."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from puremacro.midas import (
    u_midas, beta_midas,
    UMidasResult, BetaMidasResult,
)


def _simulate_midas(n_low=120, K=3, seed=0):
    rng = np.random.default_rng(seed)
    x_hf = rng.standard_normal(n_low * K)
    for i in range(1, len(x_hf)):
        x_hf[i] = 0.4 * x_hf[i - 1] + x_hf[i]
    true_w = np.array([0.7, 0.2, 0.1])
    y_lf = np.zeros(n_low)
    for i in range(n_low):
        seg = x_hf[i * K:(i + 1) * K][::-1]
        y_lf[i] = 0.5 + 0.8 * (true_w @ seg) + rng.standard_normal() * 0.4
    return y_lf, x_hf, K, true_w


# --------------------------------------------------------------------------
# u_midas
# --------------------------------------------------------------------------
def test_u_midas_returns_UMidasResult():
    y, x, K, _ = _simulate_midas()
    res = u_midas(y, x, K=K)
    assert isinstance(res, UMidasResult)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.intercept = 0.0


def test_u_midas_has_documented_fields():
    y, x, K, _ = _simulate_midas()
    res = u_midas(y, x, K=K)
    assert isinstance(res.intercept, float)
    assert res.beta.shape == (K,)
    assert res.fitted.shape == res.residuals.shape
    assert 0.0 <= res.R2 <= 1.0
    assert res.n_obs == len(y)


def test_u_midas_summary_runs():
    y, x, K, _ = _simulate_midas()
    s = u_midas(y, x, K=K).summary()
    assert isinstance(s, str)
    assert "U-MIDAS" in s


def test_u_midas_recovers_true_weights_in_proportion():
    """U-MIDAS coefficients (rescaled to sum to 1) should approximate
    the true weights [0.7, 0.2, 0.1] within 0.15."""
    y, x, K, true_w = _simulate_midas(n_low=200, K=3, seed=0)
    res = u_midas(y, x, K=K)
    # Rescale unrestricted coefficients to be comparable with normalized weights.
    norm = res.beta / max(np.abs(res.beta).sum(), 1e-9)
    assert np.allclose(norm, true_w, atol=0.15), (
        f"U-MIDAS normalised weights {norm} vs true {true_w}"
    )


# --------------------------------------------------------------------------
# beta_midas
# --------------------------------------------------------------------------
def test_beta_midas_returns_BetaMidasResult():
    y, x, K, _ = _simulate_midas()
    res = beta_midas(y, x, K=K)
    assert isinstance(res, BetaMidasResult)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.beta = 0.0


def test_beta_midas_has_documented_fields():
    y, x, K, _ = _simulate_midas()
    res = beta_midas(y, x, K=K)
    assert isinstance(res.intercept, float)
    assert isinstance(res.beta, float)
    assert isinstance(res.theta1, float) and res.theta1 > 0
    assert isinstance(res.theta2, float) and res.theta2 > 0
    assert res.weights.shape == (K,)
    assert np.isclose(res.weights.sum(), 1.0)
    assert res.fitted.shape == res.residuals.shape == y.shape
    assert 0.0 <= res.R2 <= 1.0
    assert isinstance(res.converged, bool)


def test_beta_midas_summary_runs():
    y, x, K, _ = _simulate_midas()
    s = beta_midas(y, x, K=K).summary()
    assert isinstance(s, str)
    assert "Beta-MIDAS" in s


def test_beta_midas_converges_and_recovers_decreasing_weights():
    """Beta-MIDAS should converge on a clean DGP and produce a
    monotonically decreasing weight schedule (true DGP weights
    [0.7, 0.2, 0.1] are strictly decreasing)."""
    y, x, K, _ = _simulate_midas(n_low=200, K=3, seed=0)
    res = beta_midas(y, x, K=K)
    assert res.converged
    # Most-recent weight should exceed each later one.
    assert res.weights[0] > res.weights[1]
    assert res.weights[1] > res.weights[2]
