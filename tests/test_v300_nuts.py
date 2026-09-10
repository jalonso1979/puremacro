"""Unit and integration tests for Pillar 2: Pure-Python Hamiltonian Monte Carlo & NUTS.

Tests:
1. Symplectic leapfrog energy conservation on harmonic oscillator.
2. Step size heuristic (Hoffman-Gelman Algorithm 4).
3. Dual averaging step size adaptation targeting delta* = 0.80.
4. Welford online diagonal variance accumulation and Stan shrinkage.
5. Split-R_hat, bulk ESS, tail ESS, and E-BFMI diagnostic calculations.
6. Multi-chain NUTS sampling on 2D correlated Gaussian (R_hat < 1.05, 0 divergences).
7. NUTS sampling on nonlinear 2D Rosenbrock / Banana distribution.
8. NUTSResult methods (.summary(), .to_markdown(), .to_latex(), plotting methods).
9. End-to-end DSGE model estimation with method="nuts" and analytic likelihood gradients.
"""
from __future__ import annotations

import math
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from puremacro.dsge.nuts import (
    NUTSResult,
    leapfrog,
    nuts_step,
    nuts_sample,
    find_reasonable_step_size,
    DualAveraging,
    WelfordVariance,
    compute_split_rhat,
    compute_bulk_ess,
    compute_tail_ess,
    compute_ebfmi,
)


# ---------------------------------------------------------------------------
# 1. Symplectic Leapfrog Integrator Tests
# ---------------------------------------------------------------------------

def test_leapfrog_symplectic_energy_conservation():
    """Harmonic oscillator H(q, p) = 0.5*q^2 + 0.5*p^2.
    Symplectic leapfrog preserves shadow Hamiltonian with bounded energy error."""
    def target(theta):
        # log p(q) = -0.5 * q^2 => grad = -q
        return -0.5 * float(theta[0] ** 2), -theta

    M_inv = np.array([1.0])
    q = np.array([1.5])
    p = np.array([0.0])
    eps = 0.05
    H0 = 0.5 * q[0]**2 + 0.5 * p[0]**2

    energies = []
    for _ in range(500):
        q, p, lp, g = leapfrog(q, p, -q, eps, target, M_inv)
        H = -lp + 0.5 * p[0]**2
        energies.append(H)

    # Symplectic integrators do not drift; maximum energy deviation is O(eps^2)
    max_dev = np.max(np.abs(np.array(energies) - H0))
    assert max_dev < 0.01, f"Symplectic leapfrog energy drift too large: {max_dev}"


# ---------------------------------------------------------------------------
# 2. Step Size Heuristic & Dual Averaging
# ---------------------------------------------------------------------------

def test_find_reasonable_step_size():
    """Verify Hoffman-Gelman initial step size finder finds a stable step size."""
    rng = np.random.default_rng(123)
    def target(theta):
        return -0.5 * float(np.sum(theta ** 2)), -theta

    theta = np.array([0.5, -0.5])
    lp, g = target(theta)
    M_inv = np.array([1.0, 1.0])

    eps = find_reasonable_step_size(theta, lp, g, target, M_inv, rng)
    assert 0.01 <= eps <= 5.0, f"Unexpected step size: {eps}"


def test_dual_averaging_adaptation():
    """Verify DualAveraging adapts towards target acceptance probability."""
    da = DualAveraging(eps0=0.1, target_accept=0.80)
    # If acceptance is high (1.0), step size should increase
    for _ in range(50):
        eps = da.step(1.0)
    assert eps > 0.1

    # If acceptance is low (0.1), step size should decrease
    for _ in range(100):
        eps = da.step(0.1)
    assert eps < 0.5


# ---------------------------------------------------------------------------
# 3. Welford Online Variance Accumulation
# ---------------------------------------------------------------------------

def test_welford_variance_accumulator():
    """Verify Welford online algorithm matches sample variance."""
    rng = np.random.default_rng(456)
    samples = rng.normal(loc=[1.0, -2.0], scale=[0.5, 2.0], size=(300, 2))

    welford = WelfordVariance(dim=2)
    for x in samples:
        welford.add_sample(x)

    var_np = np.var(samples, axis=0, ddof=1)
    var_welford = welford.variance()
    # Stan shrinkage count / (count + 5)
    shrinkage = 300.0 / 305.0
    expected_var = shrinkage * var_np + (5.0 / 305.0) * 1.0
    np.testing.assert_allclose(var_welford, expected_var, rtol=1e-4)


# ---------------------------------------------------------------------------
# 4. MCMC Diagnostics (Split-R_hat, Bulk/Tail ESS, E-BFMI)
# ---------------------------------------------------------------------------

def test_split_rhat_and_ess_diagnostics():
    """Verify split Gelman-Rubin R_hat and ESS functions."""
    rng = np.random.default_rng(789)
    # Stationary well-mixed chains of i.i.d. draws
    chains_good = rng.normal(0, 1, size=(2, 1000))
    r_hat_good = compute_split_rhat(chains_good)
    assert 0.99 <= r_hat_good <= 1.05

    bulk_ess = compute_bulk_ess(chains_good)
    assert bulk_ess > 500  # Should be close to 2000 for i.i.d.

    tail_ess = compute_tail_ess(chains_good)
    assert tail_ess > 200

    # Non-converged chains with different means
    chains_bad = np.array([
        rng.normal(0, 1, size=500),
        rng.normal(5, 1, size=500),
    ])
    r_hat_bad = compute_split_rhat(chains_bad)
    assert r_hat_bad > 1.20

    # E-BFMI
    energy = np.random.normal(10, 2, size=500)
    ebfmi_val = compute_ebfmi(energy)
    assert ebfmi_val > 0.5


# ---------------------------------------------------------------------------
# 5. Multi-Chain NUTS Sampling: 2D Correlated Gaussian
# ---------------------------------------------------------------------------

def test_nuts_sample_2d_correlated_gaussian():
    """Gate 2.1: Sample correlated 2D Gaussian; verify R_hat < 1.05 and zero divergences."""
    mu = np.array([0.5, -1.2])
    cov = np.array([[1.0, 0.85], [0.85, 2.0]])
    prec = np.linalg.inv(cov)

    def target(theta):
        diff = theta - mu
        lp = -0.5 * float(diff @ prec @ diff)
        g = -prec @ diff
        return lp, g

    res = nuts_sample(
        target,
        init_params=np.array([0.0, 0.0]),
        n_draws=600,
        n_chains=2,
        warmup=300,
        target_accept=0.80,
        seed=101,
    )

    summary = res.summary()
    assert np.all(summary["r_hat"] < 1.05), f"R_hat failed: {summary['r_hat'].to_dict()}"
    assert res.diagnostics["n_divergences"] == 0, f"Divergences: {res.diagnostics['n_divergences']}"
    assert np.all(np.array(res.diagnostics["ebfmi"]) > 0.3)

    # Check parameter recovery within 3 standard errors
    flat = res.draws.reshape(-1, 2)
    means = flat.mean(axis=0)
    np.testing.assert_allclose(means, mu, atol=0.25)


# ---------------------------------------------------------------------------
# 6. NUTS Sampling: Nonlinear Rosenbrock / Banana Distribution
# ---------------------------------------------------------------------------

def test_nuts_sample_rosenbrock_banana():
    """Verify NUTS samples nonlinear high-curvature Rosenbrock banana target."""
    # y ~ N(0, 1), x ~ N(y^2, 0.5^2)
    def target(theta):
        x, y = theta[0], theta[1]
        lp_y = -0.5 * (y ** 2)
        diff_x = x - y**2
        lp_x = -0.5 * (diff_x / 0.5)**2
        lp = float(lp_y + lp_x)

        g_x = -diff_x / 0.25
        g_y = -y - (-diff_x / 0.25) * (2.0 * y)
        return lp, np.array([g_x, g_y])

    res = nuts_sample(
        target,
        init_params=np.array([1.0, 1.0]),
        n_draws=500,
        n_chains=2,
        warmup=250,
        target_accept=0.90,
        seed=202,
    )

    assert res.diagnostics["divergence_rate"] <= 0.005
    assert np.all(res.summary()["r_hat"] < 1.08)


# ---------------------------------------------------------------------------
# 7. NUTSResult Container Methods
# ---------------------------------------------------------------------------

def test_nuts_result_methods():
    """Verify summary, formatting, and plotting methods on NUTSResult."""
    def target(theta):
        return -0.5 * float(theta[0]**2), -theta

    res = nuts_sample(
        target,
        init_params=np.array([0.0]),
        n_draws=150,
        n_chains=2,
        warmup=50,
        param_names=["alpha"],
        seed=303,
    )

    # DataFrame summary
    df = res.summary()
    assert "alpha" in df.index
    assert "r_hat" in df.columns
    assert "ess_bulk" in df.columns
    assert "ess_tail" in df.columns

    # Formatting
    md = res.to_markdown()
    assert "alpha" in md
    lat = res.to_latex()
    assert "alpha" in lat

    # Plotting methods
    fig1, axes1 = res.plot_trace()
    assert fig1 is not None
    fig2, axes2 = res.plot_posterior()
    assert fig2 is not None
    fig3, axes3 = res.plot_autocorr()
    assert fig3 is not None
    diag, fig4, ax4 = res.energy_diagnostics()
    assert "mean_ebfmi" in diag
    assert fig4 is not None


# ---------------------------------------------------------------------------
# 8. End-to-End DSGE Model Estimation with NUTS
# ---------------------------------------------------------------------------

def test_dsge_nuts_estimation_end_to_end():
    """Gate 2.1 & 2.3: Estimate DSGE model with method='nuts' and exact analytic gradients."""
    from puremacro.dsge.dynare import load_mod

    mod_code = """
    var y c a;
    varexo e_a;
    parameters beta sigma rho_a;
    beta = 0.99;
    sigma = 1.0;
    rho_a = 0.8;

    model;
      c = c(+1) - (1/sigma)*y;
      y = c + a;
      a = rho_a * a(-1) + e_a;
    end;

    steady_state_model;
      y = 0;
      c = 0;
      a = 0;
    end;

    varobs y;
    estimated_params;
      sigma, gamma_pdf, 1.0, 0.25;
    end;
    """

    m = load_mod(mod_code)
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"y": rng.normal(0, 0.5, size=40)})

    res = m.estimate(df, method="nuts", n_draws=150, n_chains=2, burn_in=75, seed=42)
    assert isinstance(res, NUTSResult)
    assert "sigma" in res.param_names
    assert res.draws.shape == (2, 150, 1)
    assert res.diagnostics["n_divergences"] == 0
    assert res.summary()["r_hat"].iloc[0] < 1.05
