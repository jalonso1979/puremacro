"""Tests for the SIMEX errors-in-variables correction (`puremacro.inference.simex_ols`).

Classical measurement error in a regressor attenuates the OLS slope toward
zero by exactly the reliability ratio, which gives every test here a closed
form to check against rather than a snapshot.
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.inference import SIMEXResult, simex_ols


def _eiv_sample(seed, *, n=4000, beta=1.5, sigma_u=0.8, sigma_e=0.5):
    """y = 2 + beta*x_true + e, with x observed as x_true + N(0, sigma_u^2).

    Var(x_true) = 1, so the reliability ratio is 1 / (1 + sigma_u^2) and the
    naive OLS slope converges to beta * reliability.
    """
    rng = np.random.default_rng(seed)
    x_true = rng.standard_normal(n)
    y = 2.0 + beta * x_true + sigma_e * rng.standard_normal(n)
    x = x_true + sigma_u * rng.standard_normal(n)
    return y, x


def test_naive_slope_matches_the_attenuation_formula():
    """The premise the whole estimator rests on: with classical measurement
    error the naive slope is beta * Var(x_true)/(Var(x_true) + sigma_u^2)."""
    beta, sigma_u = 1.5, 0.8
    naive = []
    for seed in range(40):
        y, x = _eiv_sample(seed, beta=beta, sigma_u=sigma_u)
        naive.append(simex_ols(y, x, sigma_u=sigma_u, B=10, rng=seed).beta_naive)
    expected = beta * 1.0 / (1.0 + sigma_u ** 2)
    assert abs(float(np.mean(naive)) - expected) < 0.01


def test_reliability_matches_its_closed_form():
    beta, sigma_u = 1.5, 0.8
    rel = [simex_ols(*_eiv_sample(s, beta=beta, sigma_u=sigma_u),
                     sigma_u=sigma_u, B=10, rng=s).reliability for s in range(40)]
    assert abs(float(np.mean(rel)) - 1.0 / (1.0 + sigma_u ** 2)) < 0.01


def test_moment_correction_recovers_the_planted_slope():
    """beta_mom = beta_naive / reliability undoes the attenuation exactly in
    expectation, so it is the sharpest available check on the plumbing."""
    beta, sigma_u = 1.5, 0.8
    mom = [simex_ols(*_eiv_sample(s, beta=beta, sigma_u=sigma_u),
                     sigma_u=sigma_u, B=10, rng=s).beta_mom for s in range(40)]
    assert abs(float(np.mean(mom)) - beta) < 0.03


def test_simex_moves_the_naive_estimate_toward_the_truth():
    """The quadratic extrapolant is an approximation, so SIMEX is not expected
    to land on beta — but it must beat the naive estimate every time."""
    beta, sigma_u = 1.5, 0.8
    wins = 0
    for seed in range(40):
        y, x = _eiv_sample(seed, beta=beta, sigma_u=sigma_u)
        r = simex_ols(y, x, sigma_u=sigma_u, B=60, rng=seed)
        wins += abs(r.beta_simex - beta) < abs(r.beta_naive - beta)
    assert wins == 40


def test_zero_measurement_error_is_a_no_op():
    """With sigma_u = 0 there is nothing to correct: every estimate collapses
    onto the naive OLS slope and the reliability ratio is one."""
    y, x = _eiv_sample(0, sigma_u=0.0)
    r = simex_ols(y, x, sigma_u=0.0, B=20, rng=0)
    assert r.reliability == pytest.approx(1.0, abs=1e-12)
    assert r.beta_mom == pytest.approx(r.beta_naive, rel=1e-12)
    assert r.beta_simex == pytest.approx(r.beta_naive, abs=1e-8)
    # lambda = 0 is the naive fit by construction, whatever the grid
    assert float(r.curve[0]) == pytest.approx(r.beta_naive, abs=1e-8)


def test_seeded_runs_are_reproducible_and_unseeded_ones_differ():
    y, x = _eiv_sample(1)
    a = simex_ols(y, x, sigma_u=0.8, B=30, rng=7)
    b = simex_ols(y, x, sigma_u=0.8, B=30, rng=7)
    c = simex_ols(y, x, sigma_u=0.8, B=30, rng=8)
    assert a.beta_simex == b.beta_simex
    np.testing.assert_array_equal(a.curve, b.curve)
    assert a.beta_simex != c.beta_simex
    # a Generator is accepted as well as an int seed
    d = simex_ols(y, x, sigma_u=0.8, B=30, rng=np.random.default_rng(7))
    assert d.beta_simex == a.beta_simex


def test_result_shape_and_metadata():
    y, x = _eiv_sample(2)
    r = simex_ols(y, x, sigma_u=0.8, B=25, deg=2, rng=0)
    assert isinstance(r, SIMEXResult)
    assert r.B == 25 and r.deg == 2
    assert r.curve.shape == r.lambdas.shape
    assert r.lambdas[0] == 0.0
    assert np.all(np.diff(r.lambdas) > 0)
    assert len(r.extrapolant_coefs) == r.deg + 1
    assert "SIMEX" in r.summary() and "reliability" in r.summary()


def test_the_simulation_curve_attenuates_with_lambda():
    """Adding more measurement error must push the slope further toward zero —
    the monotonicity the extrapolation back to lambda = -1 relies on."""
    y, x = _eiv_sample(3, beta=1.5)
    r = simex_ols(y, x, sigma_u=0.8, B=200, lambdas=[0.0, 0.5, 1.0, 1.5, 2.0], rng=0)
    assert np.all(np.diff(r.curve) < 0), r.curve
    assert np.all(r.curve > 0)


def test_a_custom_lambda_grid_and_a_linear_extrapolant_are_honoured():
    y, x = _eiv_sample(4)
    r = simex_ols(y, x, sigma_u=0.8, B=30, lambdas=[0.0, 1.0, 2.0], deg=1, rng=0)
    np.testing.assert_allclose(r.lambdas, [0.0, 1.0, 2.0])
    assert r.curve.shape == (3,)
    assert len(r.extrapolant_coefs) == 2


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(y=np.zeros(10), x=np.zeros(9)), "same length"),
        (dict(y=np.zeros(2), x=np.zeros(2)), "at least 3"),
    ],
)
def test_bad_input_raises(kwargs, match):
    with pytest.raises(ValueError, match=match):
        simex_ols(sigma_u=0.5, **kwargs)
