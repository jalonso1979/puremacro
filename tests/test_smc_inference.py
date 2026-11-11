"""Unit tests for puremacro.inference.smc (Sequential Monte Carlo).

Verifies:
1. Linear-Gaussian state-space benchmark with analytical marginal data density ln p(Y):
   smc_sample recovers ln p(Y) within 0.15 log-points.
2. Adaptive tempering maintains ESS >= 0.5 * N across stages.
3. Gradient mutation operators (MALA / HMC) outperform random-walk in ESS/sec or mixing efficiency.
4. AdaptiveTemperingSchedule root-finding and monotonicity.
5. SMCResult export and visualization methods (.to_dataframe, .summary, .to_interactive_html, plots).
6. End-to-end integration with DifferentiableModel protocol.
"""
from __future__ import annotations

import math
from typing import Any
import time
import numpy as np
import pandas as pd
import pytest
import scipy.integrate

from puremacro._autodiff import value_and_grad
from puremacro.core.protocol import DSGEDifferentiableAdapter, wrap_differentiable_model
from puremacro.dsge.priors import NormalPrior, UniformPrior
from puremacro.inference.smc import (
    AdaptiveTemperingSchedule,
    SMCResult,
    smc_sample,
)


# ===========================================================================
# Linear-Gaussian State-Space Benchmark Helpers
# ===========================================================================

def _simulate_ar1_state_space(
    rho: float = 0.60,
    sig_e: float = 0.50,
    sig_m: float = 0.20,
    T_obs: int = 40,
    seed: int = 123,
) -> np.ndarray:
    """Simulate unobserved AR(1) state with Gaussian measurement error."""
    rng = np.random.default_rng(seed)
    x = np.zeros(T_obs)
    y = np.zeros(T_obs)
    curr_x = 0.0
    for t in range(T_obs):
        curr_x = rho * curr_x + sig_e * rng.standard_normal()
        x[t] = curr_x
        y[t] = curr_x + sig_m * rng.standard_normal()
    return y


def _kalman_loglik_scalar(
    rho: float,
    y: np.ndarray,
    sig_e: float = 0.50,
    sig_m: float = 0.20,
) -> float:
    """Exact Kalman filter log-likelihood for scalar state-space model."""
    Q = sig_e ** 2
    H = sig_m ** 2
    P = Q / (1.0 - rho ** 2) if abs(rho) < 0.999 else Q / 0.01
    a = 0.0
    ll = 0.0
    log2pi = math.log(2.0 * math.pi)

    for yt in y:
        a_pred = rho * a
        P_pred = (rho ** 2) * P + Q
        v = yt - a_pred
        F = P_pred + H
        ll -= 0.5 * (log2pi + math.log(F) + (v * v) / F)
        K = P_pred / F
        a = a_pred + K * v
        P = P_pred - K * P_pred
    return ll


def _exact_marginal_likelihood(
    y: np.ndarray,
    sig_e: float = 0.50,
    sig_m: float = 0.20,
    lb: float = -0.85,
    ub: float = 0.85,
    grid_points: int = 1000,
) -> float:
    """Compute exact analytical/quadrature marginal data density ln p(Y)."""
    grid = np.linspace(lb, ub, grid_points)
    ll_vals = np.array([_kalman_loglik_scalar(r, y, sig_e, sig_m) for r in grid])
    ll_max = np.max(ll_vals)
    prior_density = 1.0 / (ub - lb)
    # Numerical quadrature using trapezoidal rule on exp(ll - ll_max)
    integrated = scipy.integrate.trapezoid(np.exp(ll_vals - ll_max) * prior_density, grid)
    exact_log_mdd = ll_max + math.log(integrated)
    return float(exact_log_mdd)


# ===========================================================================
# 1. Analytical MDD Recovery & ESS Stability Tests
# ===========================================================================

def test_smc_recovers_analytical_mdd():
    """Verify smc_sample recovers known analytical ln p(Y) within 0.15 log-points."""
    sig_e = 0.50
    sig_m = 0.20
    lb, ub = -0.85, 0.85
    y_data = _simulate_ar1_state_space(rho=0.55, sig_e=sig_e, sig_m=sig_m, T_obs=35, seed=42)
    exact_log_mdd = _exact_marginal_likelihood(y_data, sig_e, sig_m, lb=lb, ub=ub)

    # Likelihood + DualArray gradient function
    def _loglik_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        rho_val = float(theta[0])

        def _objective(p: Any) -> Any:
            r = p[0]
            Q = sig_e ** 2
            H = sig_m ** 2
            P = Q / (1.0 - r ** 2 + 1e-6)
            a = 0.0 * r
            total_ll = 0.0 * r
            log2pi = math.log(2.0 * math.pi)
            for yt in y_data:
                a_pred = r * a
                P_pred = r * P * r + Q
                v = yt - a_pred
                F = P_pred + H
                total_ll = total_ll - 0.5 * (log2pi + np.log(F) + (v * v) / F)
                K = P_pred / F
                a = a_pred + K * v
                P = P_pred - K * P_pred
            return total_ll

        val, grad = value_and_grad(_objective, np.array([rho_val]))
        return float(val), np.asarray(grad, dtype=float)

    priors = {"rho": {"dist": "uniform", "lb": lb, "ub": ub}}

    res = smc_sample(
        _loglik_and_grad,
        priors,
        n_particles=1000,
        mutation="mala",
        target_ess=0.5,
        n_steps=2,
        seed=101,
    )

    # 1. MDD recovery within 0.15 log-points
    err = abs(res.log_mdd - exact_log_mdd)
    assert err <= 0.15, f"MDD recovery error {err:.4f} exceeds 0.15 (est={res.log_mdd:.4f}, exact={exact_log_mdd:.4f})"

    # 2. ESS stability across stages (ESS >= 0.5 * N)
    assert np.all(res.ess_history >= 0.5 * 1000 - 1e-6), f"ESS dropped below 0.5*N: {res.ess_history}"

    # 3. Final stage tempering reaches 1.0
    assert res.stage_tempering[-1] >= 1.0 - 1e-6


def test_smc_mala_and_hmc_efficiency_vs_rw():
    """Verify MALA and HMC achieve higher ESS/sec or acceptance efficiency than RW."""
    sig_e = 0.50
    sig_m = 0.20
    y_data = _simulate_ar1_state_space(rho=0.50, sig_e=sig_e, sig_m=sig_m, T_obs=30, seed=77)

    def _loglik_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        rho_val = float(theta[0])

        def _objective(p: Any) -> Any:
            r = p[0]
            Q = sig_e ** 2
            H = sig_m ** 2
            P = Q / (1.0 - r ** 2 + 1e-6)
            a = 0.0 * r
            total_ll = 0.0 * r
            log2pi = math.log(2.0 * math.pi)
            for yt in y_data:
                a_pred = r * a
                P_pred = r * P * r + Q
                v = yt - a_pred
                F = P_pred + H
                total_ll = total_ll - 0.5 * (log2pi + np.log(F) + (v * v) / F)
                K = P_pred / F
                a = a_pred + K * v
                P = P_pred - K * P_pred
            return total_ll

        val, grad = value_and_grad(_objective, np.array([rho_val]))
        return float(val), np.asarray(grad, dtype=float)

    priors = {"rho": {"dist": "uniform", "lb": -0.80, "ub": 0.80}}

    # Run RW
    res_rw = smc_sample(_loglik_and_grad, priors, n_particles=500, mutation="rw", seed=42)
    # Run MALA
    res_mala = smc_sample(_loglik_and_grad, priors, n_particles=500, mutation="mala", seed=42)
    # Run HMC
    res_hmc = smc_sample(_loglik_and_grad, priors, n_particles=500, mutation="hmc", seed=42)

    # Acceptance rates of gradient-based methods should match their higher optimal targets
    assert np.mean(res_mala.acceptance_rates) > np.mean(res_rw.acceptance_rates)
    assert np.mean(res_hmc.acceptance_rates) > np.mean(res_rw.acceptance_rates)

    # Posterior means close to true value 0.50
    w_mala = res_mala.weights / np.sum(res_mala.weights)
    mean_mala = float(np.sum(res_mala.particles[:, 0] * w_mala))
    assert 0.30 <= mean_mala <= 0.70


# ===========================================================================
# 2. AdaptiveTemperingSchedule Unit Tests
# ===========================================================================

def test_adaptive_tempering_schedule_monotonicity():
    """Adaptive schedule monotonically increases and terminates at 1.0."""
    sched = AdaptiveTemperingSchedule(target_ess_ratio=0.6, min_step=1e-3, max_stages=20)
    N = 200
    logliks = -np.random.default_rng(0).exponential(scale=10.0, size=N)
    weights = np.ones(N) / N

    phi_curr = 0.0
    stages = [phi_curr]
    for st in range(1, 25):
        phi_next = sched.next_phi(phi_curr, logliks, weights, st)
        assert phi_next > phi_curr, f"Stage {st}: non-increasing phi {phi_next} <= {phi_curr}"
        assert phi_next <= 1.0
        stages.append(phi_next)
        phi_curr = phi_next
        if sched.is_finished(phi_curr):
            break

    assert stages[-1] >= 1.0 - 1e-8


# ===========================================================================
# 3. SMCResult Presentation & Diagnostics
# ===========================================================================

def test_smc_result_presentation_methods():
    """Verify summary, DataFrame, LaTeX, Typst, HTML, and plot exports."""
    particles = np.random.default_rng(42).normal(loc=0.5, scale=0.1, size=(200, 2))
    weights = np.ones(200) / 200.0
    stage_phi = np.array([0.0, 0.3, 0.7, 1.0])
    res = SMCResult(
        particles=particles,
        weights=weights,
        stage_tempering=stage_phi,
        log_mdd=-125.4,
        mdd_se=0.035,
        acceptance_rates=np.array([0.55, 0.58, 0.52]),
        ess_history=np.array([200.0, 160.0, 140.0, 180.0]),
        mutation_type="mala",
        param_names=("param_a", "param_b"),
        elapsed_seconds=1.25,
    )

    # 1. Summary
    summ = res.summary()
    assert "MALA" in summ
    assert "-125.4000" in summ

    # 2. DataFrame
    df = res.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"Mean", "Std", "Median", "P05", "P95", "ESS"}
    assert len(df) == 2

    # 3. Markdown / LaTeX / Typst
    md = res.to_markdown()
    assert "param_a" in md
    latex = res.to_latex()
    assert "tabular" in latex
    typst = res.to_typst()
    assert "#table" in typst

    # 4. Interactive HTML
    html = res.to_interactive_html()
    assert "<html" in html or "<!DOCTYPE html>" in html
    assert "http://" not in html and "https://" not in html

    # 5. Plots
    fig1 = res.plot_stages()
    assert fig1 is not None
    fig2 = res.plot_posterior()
    assert fig2 is not None


# ===========================================================================
# 4. Composable with DifferentiableModel Protocol
# ===========================================================================

def test_smc_integration_with_differentiable_model():
    """smc_sample accepts DifferentiableModel directly."""
    adapter = DSGEDifferentiableAdapter(None)
    priors = {
        "rho": {"dist": "uniform", "lb": 0.5, "ub": 0.9},
        "sig_e": {"dist": "normal", "mean": 0.1, "std": 0.05, "lb": 0.01, "ub": 0.5},
        "sig_m": {"dist": "normal", "mean": 0.05, "std": 0.02, "lb": 0.01, "ub": 0.2},
    }

    res = smc_sample(
        adapter,
        priors,
        n_particles=100,
        mutation="mala",
        n_steps=1,
        seed=1,
    )

    assert isinstance(res, SMCResult)
    assert len(res.param_names) == 3
    assert res.stage_tempering[-1] >= 1.0 - 1e-6


# ===========================================================================
# 5. Exact Truncated Prior Sampling Across All 6 DSGE Families
# ===========================================================================

def test_smc_prior_sampling_all_families_exact_truncation_and_finite_logprior():
    """Stage-0 prior sampler must support all 6 families without boundary atoms or -inf log-priors."""
    import scipy.stats as st
    from puremacro.dsge.priors import (
        BetaPrior,
        GammaPrior,
        InvGammaPrior,
        NormalPrior,
        UniformPrior,
        WeibullPrior,
        ensure_prior,
        log_prior,
    )
    from puremacro.inference.smc import _sample_from_prior

    rng = np.random.default_rng(202611)
    n = 4096

    # 1. InvGamma type1 & type2 (class and dict without explicit lb)
    ig1 = InvGammaPrior(mean=0.10, std=2.00)
    d_ig1 = _sample_from_prior(ig1, n, rng)
    assert np.sum(d_ig1 == ig1.lb) == 0
    assert np.all(np.isfinite(ig1.logpdf(d_ig1)))
    # Check exact CDF: 1/X^2 ~ Gamma(nu/2, scale=2/s) truncated to (0, 1/lb^2)
    y_hi = ig1.lb ** -2.0
    c_hi = st.gamma.cdf(y_hi, a=ig1.nu / 2.0, scale=2.0 / ig1.s)
    ks_ig1 = st.kstest(
        d_ig1,
        lambda x: 1.0 - st.gamma.cdf(np.asarray(x) ** -2.0, a=ig1.nu / 2.0, scale=2.0 / ig1.s) / c_hi,
    )
    assert ks_ig1.pvalue > 0.01

    ig_dict = {"dist": "invgamma", "mean": 0.10, "std": 2.00}
    d_ig_dict = _sample_from_prior(ig_dict, n, rng)
    ig_parsed = ensure_prior(ig_dict)
    assert np.all(d_ig_dict > 0.0)
    assert all(math.isfinite(log_prior({"s": float(v)}, {"s": ig_parsed})) for v in d_ig_dict[:256])

    ig2 = InvGammaPrior(s=0.5, nu=5.0, kind="type2", lb=0.05, ub=2.0)
    d_ig2 = _sample_from_prior(ig2, n, rng)
    assert np.all((d_ig2 > 0.05) & (d_ig2 < 2.0))
    assert np.all(np.isfinite(ig2.logpdf(d_ig2)))

    # 2. Truncated Normal on [0, inf): half-normal with zero point-mass at 0
    tn = NormalPrior(mean=0.0, std=1.0, lb=0.0, ub=math.inf)
    d_tn = _sample_from_prior(tn, n, rng)
    assert np.sum(d_tn == 0.0) == 0
    assert np.all(d_tn > 0.0)
    ks_tn = st.kstest(d_tn, lambda x: 2.0 * st.norm.cdf(x) - 1.0)
    assert ks_tn.pvalue > 0.01

    # 3. Truncated Beta, Gamma, Weibull, and Uniform
    families = [
        BetaPrior(mean=0.5, std=0.2, lb=0.3, ub=0.8),
        GammaPrior(mean=1.5, std=0.6, lb=0.8, ub=3.0, shift=0.2),
        WeibullPrior(mean=1.2, std=0.5, lb=0.4, ub=2.5, shift=0.1),
        UniformPrior(lb=-1.5, ub=2.5),
    ]
    for p_obj in families:
        draws = _sample_from_prior(p_obj, 2048, rng)
        assert np.all((draws > p_obj.lb) & (draws < p_obj.ub))
        assert np.all(np.isfinite(p_obj.logpdf(draws)))


def test_smc_truncated_normal_analytical_mdd():
    """Under a half-normal prior N+(0, 1) and Gaussian likelihood, SMC recovers exact truncated MDD."""
    import scipy.stats as st
    from puremacro.dsge.priors import NormalPrior

    y_obs = 1.2
    sig = 0.5
    # Prior: theta ~ N(0, 1) truncated to [0, inf), density 2 * phi(theta) on theta >= 0.
    # Likelihood: L(theta) = (2*pi*sig^2)^(-1/2) exp(-(y - theta)^2 / (2*sig^2)).
    # Exact marginal likelihood = int_0^inf L(theta) * 2 * phi(theta) d theta.
    v_post = 1.0 / (1.0 + 1.0 / (sig ** 2))
    m_post = v_post * (y_obs / (sig ** 2))
    log_norm_lik = st.norm.logpdf(y_obs, loc=0.0, scale=math.sqrt(1.0 + sig ** 2))
    exact_log_mdd = math.log(2.0) + float(log_norm_lik) + float(st.norm.logsf(0.0, loc=m_post, scale=math.sqrt(v_post)))

    def _ll_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        th = float(theta[0])
        ll = -0.5 * math.log(2.0 * math.pi * sig * sig) - 0.5 * ((y_obs - th) / sig) ** 2
        g = np.array([(y_obs - th) / (sig * sig)], dtype=float)
        return ll, g

    res = smc_sample(
        _ll_grad,
        {"theta": NormalPrior(mean=0.0, std=1.0, lb=0.0, ub=math.inf)},
        n_particles=1500,
        mutation="rw",
        n_steps=3,
        seed=123,
    )
    assert abs(res.log_mdd - exact_log_mdd) < 0.08

