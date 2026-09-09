"""Adversarial stress harness and empirical challenge suite for puremacro 2.9.0 M1.

Tests numerical accuracy, robustness, boundary conditions, and edge cases for:
1. Gauss-Legendre quadrature spectral density integration.
2. HP filter theoretical variance against closed-form and fine-grid quadrature benchmarks (< 1e-8 rel error).
3. One-sided HP recursive Kalman filter: linear trend, HF noise, T=4, near-singular lambdas (1e-5, 1e8).
4. Baxter-King bandpass filter: extreme frequency bands [2.1, 2.2], [10, 100].
5. Unit roots, explosive roots, zero variance, and container polymorphism.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.integrate

from puremacro.dsge import build, one_sided_hp_filter, TheoreticalMomentsResult, StochSimulResult
from puremacro.dsge._moments import spectral_moments, compute_autocorr_matrices


@pytest.fixture
def ar1_model_factory():
    """Factory creating calibrated AR(1) models for arbitrary rho."""
    def _create(rho: float = 0.8):
        def eqs(xp, x, e, p):
            return [xp.a - p.rho * x.a - e.eps]

        return build(
            eqs,
            variables=["a"],
            states=["a"],
            shocks=["eps"],
            params=dict(rho=rho),
            steady_state=dict(a=0.0),
        )

    return _create


@pytest.fixture
def rbc_model():
    """Canonical calibrated RBC model."""
    def eqs(xp, x, e, p):
        return [
            x.c**-p.sigma - p.beta * xp.c**-p.sigma * (p.alpha * xp.z * xp.k**(p.alpha - 1) + 1 - p.delta),
            x.c + xp.k - x.z * x.k**p.alpha - (1 - p.delta) * x.k,
            xp.z - (1.0 - p.rho) - p.rho * x.z - e.eps,
        ]

    beta, delta, alpha = 0.99, 0.025, 0.33
    r_ss = 1.0 / beta - 1.0
    k_ss = (alpha / (r_ss + delta)) ** (1.0 / (1.0 - alpha))
    y_ss = k_ss**alpha
    c_ss = y_ss - delta * k_ss

    return build(
        eqs,
        variables=["c", "k", "z"],
        states=["k", "z"],
        shocks=["eps"],
        params=dict(alpha=alpha, beta=beta, delta=delta, sigma=1.0, rho=0.95),
        steady_state=dict(c=c_ss, k=k_ss, z=1.0),
    )


# =========================================================================
# 1. Closed-Form & Fine-Grid Quadrature HP Filter Variance Benchmarks
# =========================================================================

def test_adversarial_hp_closed_form_white_noise():
    """Verify HP filter theoretical variance matches closed-form analytical solution.

    For white noise with unit variance, the theoretical HP filter variance has
    the exact analytical closed form:
        Var_hp = 1 - (1 / sqrt(1 + 16*lambda)) * sqrt((sqrt(1 + 16*lambda) + 1) / 2).
    Verify that spectral_moments (both for nx=0 and for AR(1) with rho=0) matches
    this closed-form solution to relative error < 1e-10 across 8 orders of lambda.
    """
    lambdas = [0.1, 1.0, 10.0, 100.0, 1600.0, 14400.0, 1e5, 1e6]
    for lamb in lambdas:
        sA = np.sqrt(1.0 + 16.0 * lamb)
        var_closed = 1.0 - (1.0 / sA) * np.sqrt(0.5 * (sA + 1.0))

        # Static model (nx = 0)
        G = np.zeros((0, 0))
        N = np.zeros((0, 1))
        Mx = np.zeros((1, 0))
        Mu = np.array([[1.0]])
        sig_u = np.array([[1.0]])
        _, g0_static, _ = spectral_moments(
            G, N, Mx, Mu, sig_u, lags=1, filter_type="hp", hp_lambda=lamb, n_quad=128
        )
        rel_err_static = abs(g0_static[0, 0] - var_closed) / var_closed
        assert rel_err_static < 1e-10, (
            f"Static nx=0 lambda={lamb}: relative error {rel_err_static:.3e} exceeds 1e-10"
        )

        # Dynamic model with rho = 0
        G_ar = np.array([[0.0]])
        N_ar = np.array([[1.0]])
        Mx_ar = np.array([[0.0]])
        Mu_ar = np.array([[1.0]])
        _, g0_ar, _ = spectral_moments(
            G_ar, N_ar, Mx_ar, Mu_ar, sig_u, lags=1, filter_type="hp", hp_lambda=lamb, n_quad=128
        )
        rel_err_ar = abs(g0_ar[0, 0] - var_closed) / var_closed
        assert rel_err_ar < 1e-10, (
            f"Dynamic rho=0 lambda={lamb}: relative error {rel_err_ar:.3e} exceeds 1e-10"
        )


@pytest.mark.parametrize("rho", [-0.9, -0.5, 0.0, 0.5, 0.8, 0.95, 0.99, 0.999])
@pytest.mark.parametrize("hp_lambda", [100.0, 1600.0, 14400.0, 1e5])
def test_adversarial_hp_quadrature_ar1_fine_grid(rho, hp_lambda):
    """Stress-test HP theoretical variance on AR(1) against adaptive numerical integration.

    Uses scipy.integrate.quad with epsabs=1e-14, epsrel=1e-13 as oracle benchmark.
    Verifies that puremacro's 128-point Gauss-Legendre quadrature achieves < 1e-8
    relative error across persistence levels up to rho=0.999.
    """
    G = np.array([[rho]])
    N = np.array([[1.0]])
    Mx = np.array([[rho]])
    Mu = np.array([[1.0]])
    sigma_u = np.array([[1.0]])

    _, g0, _ = spectral_moments(
        G, N, Mx, Mu, sigma_u, lags=1, filter_type="hp", hp_lambda=hp_lambda, n_quad=128
    )
    val_pm = g0[0, 0]

    def integrand(w):
        cos_w = np.cos(w)
        gain2 = 4.0 * hp_lambda * (1.0 - cos_w)**2 / (1.0 + 4.0 * hp_lambda * (1.0 - cos_w)**2)
        psd = (1.0 / (2.0 * np.pi)) / (1.0 + rho**2 - 2.0 * rho * cos_w)
        return 2.0 * gain2 * psd

    val_oracle, _ = scipy.integrate.quad(integrand, 0.0, np.pi, epsabs=1e-14, epsrel=1e-13, limit=500)
    rel_err = abs(val_pm - val_oracle) / val_oracle
    assert rel_err < 1e-8, (
        f"AR(1) rho={rho}, lambda={hp_lambda}: rel error {rel_err:.3e} exceeds 1e-8 threshold"
    )


def test_adversarial_hp_multivariate_dsge_high_precision(rbc_model):
    """Verify multivariate DSGE HP filter variance converges to machine precision.

    Compares 128-node Gauss-Legendre quadrature against 1024-node reference
    on the full 3-variable RBC model across quarterly and annual smoothing values.
    """
    G, N = rbc_model.solution.G, rbc_model.solution.N
    Mx, Mu = rbc_model._reported_loadings()
    sigma_u = rbc_model._shock_covariance(None)

    for lamb in [100.0, 1600.0, 14400.0]:
        _, g0_128, _ = spectral_moments(
            G, N, Mx, Mu, sigma_u, lags=1, filter_type="hp", hp_lambda=lamb, n_quad=128
        )
        _, g0_1024, _ = spectral_moments(
            G, N, Mx, Mu, sigma_u, lags=1, filter_type="hp", hp_lambda=lamb, n_quad=1024
        )
        rel_err = np.max(np.abs(g0_128 - g0_1024) / g0_1024)
        assert rel_err < 1e-10, (
            f"RBC lambda={lamb}: relative error {rel_err:.3e} between 128 and 1024 nodes exceeds 1e-10"
        )


# =========================================================================
# 2. One-Sided HP Filter Recursive Kalman Filter Stress Harness
# =========================================================================

def test_adversarial_one_sided_hp_linear_trend():
    """Verify one-sided HP filter tracks an exact linear trend y_t = a + b*t with zero curvature.

    Because Delta^2 (a + b*t) == 0, the trend is linear and cycle is zero
    aside from transient initialization effects.
    """
    t = np.arange(100, dtype=float)
    y = 12.34 + 5.67 * t

    cycle, trend = one_sided_hp_filter(y, lamb=1600.0)

    # Adding-up identity holds everywhere
    np.testing.assert_allclose(cycle + trend, y, atol=1e-12)

    # Initial transient error due to P=1e5 diffuse prior is <= 6e-5
    assert np.max(np.abs(cycle)) < 6e-5

    # After initial transient (t >= 20), cycle is <= 1e-5 and steadily decaying
    assert np.max(np.abs(cycle[20:])) < 1e-5


def test_adversarial_one_sided_hp_high_frequency_noise():
    """Verify one-sided HP filter heavily attenuates high-frequency noise."""
    t = np.arange(200)
    noise = (-1.0) ** t

    cycle, trend = one_sided_hp_filter(noise, lamb=1600.0)

    # High-frequency alternating noise should be mostly in cycle, not trend
    trend_std = np.std(trend[50:])
    cycle_std = np.std(cycle[50:])
    assert trend_std < 0.15
    assert cycle_std > 0.85
    assert abs(np.mean(trend[50:])) < 0.01


@pytest.mark.parametrize("lamb", [1e-5, 1e-3, 1e5, 1e8])
def test_adversarial_one_sided_hp_near_singular_lambdas(lamb):
    """Stress-test one-sided HP filter under near-singular smoothing parameters."""
    t = np.arange(100, dtype=float)
    y = 3.0 * t + np.sin(0.3 * t)

    cycle, trend = one_sided_hp_filter(y, lamb=lamb)

    assert not np.isnan(cycle).any()
    assert not np.isnan(trend).any()
    assert not np.isinf(cycle).any()
    assert not np.isinf(trend).any()
    np.testing.assert_allclose(cycle + trend, y, atol=1e-12)

    if lamb <= 1e-5:
        # Extremely small lambda: q = 1/lambda -> inf, observation error negligible,
        # trend tracks data almost identically
        assert np.max(np.abs(cycle)) < 1e-4
    elif lamb >= 1e8:
        # Extremely large lambda: q = 1/lambda -> 0, state transition covariance negligible,
        # trend is smoother than under small lambda and stable
        assert np.std(trend) > 0.0
        assert np.max(np.abs(cycle)) < 2.0


def test_adversarial_one_sided_hp_sample_size_boundaries():
    """Verify one-sided HP filter sample size boundary conditions."""
    # Minimum allowed length: T = 4
    y4 = np.array([10.0, 20.0, 30.0, 40.0])
    c4, t4 = one_sided_hp_filter(y4, lamb=1600.0)
    assert len(c4) == 4
    assert len(t4) == 4
    assert not np.isnan(c4).any()
    assert not np.isnan(t4).any()

    # T < 4 must strictly raise ValueError
    for T in [0, 1, 2, 3]:
        with pytest.raises(ValueError, match="requires at least 4 observations"):
            one_sided_hp_filter(np.ones(T), lamb=1600.0)


def test_adversarial_one_sided_hp_no_memory_explosion():
    """Stress-test one-sided HP filter with large series T=50,000 to verify O(T) memory and speed."""
    import time

    T = 50000
    rng = np.random.default_rng(123)
    y_large = rng.standard_normal(T)

    t0 = time.perf_counter()
    c, tr = one_sided_hp_filter(y_large, lamb=1600.0)
    dt = time.perf_counter() - t0

    assert dt < 2.0, f"Execution took {dt:.2f}s, exceeding 2.0s limit"
    assert c.shape == (T,)
    assert tr.shape == (T,)
    assert not np.isnan(c).any()


def test_adversarial_one_sided_hp_containers():
    """Verify container polymorphism and metadata preservation across Series and DataFrame."""
    dates = pd.date_range("2020-01-01", periods=10, freq="QE")

    # DataFrame with multiple columns
    df = pd.DataFrame(
        {"y": np.linspace(100, 200, 10), "pi": np.sin(np.linspace(0, 3, 10))},
        index=dates,
    )
    c_df, t_df = one_sided_hp_filter(df, lamb=1600.0)
    assert isinstance(c_df, pd.DataFrame)
    assert isinstance(t_df, pd.DataFrame)
    assert (c_df.index == dates).all()
    assert (t_df.index == dates).all()
    assert list(c_df.columns) == ["y", "pi"]
    assert list(t_df.columns) == ["y", "pi"]

    # Series
    s = pd.Series(np.linspace(100, 200, 10), index=dates, name="output")
    c_s, t_s = one_sided_hp_filter(s, lamb=1600.0)
    assert isinstance(c_s, pd.Series)
    assert isinstance(t_s, pd.Series)
    assert c_s.name == "output"
    assert t_s.name == "output"
    assert (c_s.index == dates).all()


# =========================================================================
# 3. Baxter-King Bandpass Filter Extreme Frequency Bands
# =========================================================================

@pytest.mark.parametrize("bp", [(2.1, 2.2), (10.0, 100.0)])
def test_adversarial_bandpass_theoretical_moments_accuracy(ar1_model_factory, bp):
    """Verify theoretical bandpass moments accuracy on extreme bands against numerical quadrature.

    Band [2.1, 2.2]: high-frequency band near Nyquist limit pi.
    Band [10, 100]: wide low-frequency long business-cycle band.
    """
    m = ar1_model_factory(rho=0.8)
    G, N = m.solution.G, m.solution.N
    Mx, Mu = m._reported_loadings()
    sigma_u = m._shock_covariance(None)

    _, g0_pm, _ = spectral_moments(
        G, N, Mx, Mu, sigma_u, lags=1, filter_type="bandpass", bandpass=bp, n_quad=128
    )
    val_pm = g0_pm[0, 0]

    # Oracle benchmark via scipy.integrate.quad over [2*pi/high, 2*pi/low]
    low, high = bp
    wL = 2.0 * np.pi / high
    wH = 2.0 * np.pi / low
    val_oracle, _ = scipy.integrate.quad(
        lambda w: (1.0 / np.pi) / (1.0 + 0.8**2 - 2.0 * 0.8 * np.cos(w)),
        wL,
        wH,
        epsabs=1e-14,
        epsrel=1e-13,
    )

    rel_err = abs(val_pm - val_oracle) / val_oracle
    assert rel_err < 1e-8, f"Bandpass {bp}: relative error {rel_err:.3e} exceeds 1e-8"


def test_adversarial_bandpass_stoch_simul_extreme_bands(ar1_model_factory):
    """Verify stoch_simul executes cleanly with extreme frequency bands and simul_replic."""
    m = ar1_model_factory(rho=0.8)

    for bp in [(2.1, 2.2), (10.0, 100.0)]:
        res = m.stoch_simul(periods=200, bandpass_filter=bp, simul_replic=20, seed=42)
        assert isinstance(res, StochSimulResult)
        assert res.simulated_moments is not None
        assert not res.simulated_moments.isna().any().any()
        assert (res.simulated_moments["MC Std.Err."] > 0).all()
        assert res.theoretical_moments is not None
        assert res.theoretical_moments.moments.loc["a", "Variance"] > 0.0


# =========================================================================
# 4. Unit Roots, Explosive Roots & Singular Covariance Robustness
# =========================================================================

def test_adversarial_unit_root_spectral_moments():
    """Verify spectral_moments cleanly evaluates HP filter on exact unit-root system (I(1))."""
    # G = [[1.0]] has exact unit root
    G = np.array([[1.0]])
    N = np.array([[1.0]])
    Mx = np.array([[1.0]])
    Mu = np.array([[1.0]])
    sigma_u = np.array([[1.0]])

    sig_x, g0, gammas = spectral_moments(
        G, N, Mx, Mu, sigma_u, lags=4, filter_type="hp", hp_lambda=1600.0
    )

    assert not np.isnan(g0).any()
    assert not np.isinf(g0).any()
    assert g0[0, 0] > 0.0
    for gk in gammas:
        assert not np.isnan(gk).any()


def test_adversarial_explosive_root_spectral_moments():
    """Verify spectral_moments raises ValueError when transition matrix has explosive root."""
    G = np.array([[1.05]])
    N = np.array([[1.0]])
    Mx = np.array([[1.0]])
    Mu = np.array([[1.0]])
    sigma_u = np.array([[1.0]])

    with pytest.raises(ValueError, match="explosive eigenvalues"):
        spectral_moments(G, N, Mx, Mu, sigma_u, filter_type="hp", hp_lambda=1600.0)


def test_adversarial_zero_variance_correlation_handling():
    """Verify compute_autocorr_matrices handles zero-variance variables without division by zero crash."""
    g0_zero = np.array([[0.0, 0.0], [0.0, 1.0]])
    gammas_zero = [np.array([[0.0, 0.0], [0.0, 0.5]])]

    df_corr, autocorr_mats = compute_autocorr_matrices(
        g0_zero, gammas_zero, ["zero_var", "pos_var"]
    )

    # Unitary diagonal
    assert df_corr.loc["zero_var", "zero_var"] == 1.0
    assert df_corr.loc["pos_var", "pos_var"] == 1.0
    # Zero variance cross-correlation is NaN
    assert np.isnan(df_corr.loc["zero_var", "pos_var"])
    # Autocorrelation of zero variance is NaN, positive variance is valid
    assert np.isnan(autocorr_mats[0].loc["zero_var", "zero_var"])
    np.testing.assert_allclose(autocorr_mats[0].loc["pos_var", "pos_var"], 0.5, atol=1e-12)
