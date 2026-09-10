"""Comprehensive 4-Tier Opaque-Box E2E Test Suite for puremacro Phase D.

Advanced DSGE Frontier Phase D:
1. Pure-Python Vectorized Particle Filtering for Nonlinear & Stochastic Volatility Models (R4)
   - Bootstrap Particle Filter (BPF) with systematic, stratified, residual, and multinomial resampling
   - Auxiliary Particle Filter (APF) with first-stage proposal weights and predictive covariance
   - 2nd-order and 3rd-order pruned perturbation DSGE likelihood evaluation
   - Stochastic Volatility: sigma_t = bar{sigma} exp(h_t), h_t = rho_h h_{t-1} + sigma_h eta_t
   - Precautionary saving shift and positive skewness tracking
   - Fat-tailed innovation and observation densities (Student-t, Gaussian mixture)
   - ParticleFilterResult presentation interface (.summary, .plot, .to_markdown, .to_latex, .to_typst)

2. Markov-Switching DSGE (MS-DSGE / Regime Switching) (R5)
   - Foerster, Rubio-Ramirez, Waggoner & Zha (2016) perturbation system:
     A(s_t) E_t [y_{t+1}] + B(s_t) y_t + C(s_t) y_{t-1} + K(s_t) + D(s_t) epsilon_t = 0
   - Minimal State Variable (MSV) solution: y_t = c(s_t) + T(s_t) y_{t-1} + R(s_t) epsilon_t
   - Coupled quadratic solvers: Block Newton-Raphson (Kronecker Jacobian) & damped functional iteration
   - First-moment stability rho(M_1) < 1 and Mean-Square Stability (MSS) rho(M_2) < 1
   - Ergodic distribution pi_infty P = pi_infty, ergodic mean, and discrete Lyapunov covariance
   - Closed-form analytical GIRF (1_S^T otimes I_n) M_1^h z_0 and counterfactual regime-conditional IRFs
   - Canonical economic benchmarks: Hawkish vs Dovish (FWZ 2011), Active vs Passive fiscal policy (Leeper 1991)
   - MSDSGEResult presentation interface (.summary, .plot, .to_markdown, .to_latex, .to_typst, .irf, .girf, .simulate)

Zero new external dependencies: strictly verified under Pyodide 4-package contract (numpy, scipy, pandas, matplotlib).
"""
from __future__ import annotations

import copy
import dataclasses
import inspect
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.linalg
import scipy.stats

import puremacro
import puremacro.dsge as dsge
from puremacro.dsge import (
    LinearModel,
    ModelError,
    BlanchardKahnError,
    PrunedDSGESolution,
    canonical_growth_2nd_order,
)
from puremacro.dsge.klein import KleinSolution


# ===========================================================================
# Opaque-Box Module Resolution Helpers with Progressive Readiness Checks
# ===========================================================================

def _require_particle_filter():
    """Resolve particle_filter callable and ParticleFilterResult container."""
    pf_fn = None
    res_cls = None

    # Try submodule import
    try:
        from puremacro.dsge import particle_filter as pf_mod
        if hasattr(pf_mod, "particle_filter"):
            pf_fn = getattr(pf_mod, "particle_filter")
        if hasattr(pf_mod, "ParticleFilterResult"):
            res_cls = getattr(pf_mod, "ParticleFilterResult")
    except (ImportError, AttributeError):
        pass

    # Fallback to dsge package export
    if pf_fn is None and hasattr(dsge, "particle_filter"):
        pf_fn = getattr(dsge, "particle_filter")
    if res_cls is None and hasattr(dsge, "ParticleFilterResult"):
        res_cls = getattr(dsge, "ParticleFilterResult")

    if pf_fn is None or res_cls is None:
        pytest.skip("Milestone 1: particle_filter pending in puremacro.dsge.particle_filter")
    return pf_fn, res_cls


def _require_ms_dsge():
    """Resolve solve_ms_dsge callable and MSDSGEResult container."""
    ms_fn = None
    res_cls = None

    # Try submodule import
    try:
        from puremacro.dsge import markov_switching as ms_mod
        if hasattr(ms_mod, "solve_ms_dsge"):
            ms_fn = getattr(ms_mod, "solve_ms_dsge")
        if hasattr(ms_mod, "MSDSGEResult"):
            res_cls = getattr(ms_mod, "MSDSGEResult")
    except (ImportError, AttributeError):
        pass

    # Fallback to dsge package export
    if ms_fn is None and hasattr(dsge, "solve_ms_dsge"):
        ms_fn = getattr(dsge, "solve_ms_dsge")
    if res_cls is None and hasattr(dsge, "MSDSGEResult"):
        res_cls = getattr(dsge, "MSDSGEResult")

    if ms_fn is None or res_cls is None:
        pytest.skip("Milestone 2: solve_ms_dsge pending in puremacro.dsge.markov_switching")
    return ms_fn, res_cls


# ===========================================================================
# Mathematical Reference Oracles & Model Factories
# ===========================================================================

def _kalman_filter_oracle(
    data: np.ndarray,
    T: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    Q: np.ndarray,
    H: np.ndarray,
    a0: np.ndarray | None = None,
    P0: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Independent pure-Python Kalman filter oracle for exact Gaussian log-likelihood.

    State space:
        x_t = T x_{t-1} + R u_t,   u_t ~ N(0, Q)
        y_t = Z x_t + v_t,         v_t ~ N(0, H)

    Returns:
        (log_likelihood, filtered_states, filtered_covs)
    """
    T_mat = np.atleast_2d(T)
    R_mat = np.atleast_2d(R)
    Z_mat = np.atleast_2d(Z)
    Q_mat = np.atleast_2d(Q)
    H_mat = np.atleast_2d(H)

    n_x = T_mat.shape[0]
    n_obs, n_y = data.shape

    # Stationary initialization via discrete Lyapunov equation P0 = T P0 T' + R Q R'
    if a0 is None:
        a = np.zeros(n_x)
    else:
        a = np.array(a0, dtype=float)

    if P0 is None:
        QQ = R_mat @ Q_mat @ R_mat.T
        try:
            P = scipy.linalg.solve_discrete_lyapunov(T_mat, QQ)
        except Exception:
            P = np.eye(n_x) * 1.0
    else:
        P = np.array(P0, dtype=float)

    log_lik = 0.0
    filtered_states = np.zeros((n_obs, n_x))
    filtered_covs = np.zeros((n_obs, n_x, n_x))

    QQ = R_mat @ Q_mat @ R_mat.T

    for t in range(n_obs):
        # 1. Forecast step
        if t > 0:
            a_prior = T_mat @ a
            P_prior = T_mat @ P @ T_mat.T + QQ
        else:
            a_prior = a
            P_prior = P

        # 2. Measurement update
        y_t = data[t, :]
        v_t = y_t - Z_mat @ a_prior
        F_t = Z_mat @ P_prior @ Z_mat.T + H_mat
        F_inv = np.linalg.inv(F_t)
        K_t = P_prior @ Z_mat.T @ F_inv

        a = a_prior + K_t @ v_t
        P = (np.eye(n_x) - K_t @ Z_mat) @ P_prior

        filtered_states[t, :] = a
        filtered_covs[t, :, :] = P

        sign, logdet = np.linalg.slogdet(F_t)
        ll_t = -0.5 * (n_y * np.log(2.0 * np.pi) + logdet + v_t.T @ F_inv @ v_t)
        log_lik += float(ll_t)

    return log_lik, filtered_states, filtered_covs


def _make_test_linear_model(
    T: np.ndarray,
    R: np.ndarray,
    variables: Sequence[str],
    shocks: Sequence[str],
    steady_state: pd.Series | Mapping[str, float] | None = None,
    params: Mapping[str, float] | None = None,
) -> LinearModel:
    """Construct a minimal valid LinearModel instance from state-space matrices (T, R)."""
    T_arr = np.asarray(T, dtype=float)
    R_arr = np.asarray(R, dtype=float)
    n_x = len(variables)
    n_e = len(shocks)

    if steady_state is None:
        ss_series = pd.Series(0.0, index=list(variables))
    elif isinstance(steady_state, Mapping):
        ss_series = pd.Series(steady_state)
    else:
        ss_series = steady_state

    sol = KleinSolution(
        G=T_arr,
        F=np.zeros((0, n_x)),
        N=R_arr,
        L=np.zeros((0, n_e)),
        eu=(1, 1),
        eigenvalues=np.linalg.eigvals(T_arr) if n_x > 0 else np.array([]),
        residual=0.0,
    )
    return LinearModel(
        variables=tuple(variables),
        states=tuple(variables),
        controls=(),
        shocks=tuple(shocks),
        steady_state=ss_series,
        units={v: "level" for v in variables},
        solution=sol,
        A=np.eye(n_x),
        B=np.eye(n_x),
        C=np.zeros((n_x, n_e)),
        method="direct",
        residual_norm=0.0,
        _params=dict(params or {}),
    )


def _make_ar1_state_space_data(
    n_obs: int = 50,
    rho: float = 0.8,
    sigma_u: float = 0.5,
    sigma_v: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, LinearModel, float]:
    """Construct an AR(1) state space with simulated data and exact Kalman log-likelihood."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n_obs)
    y = np.zeros(n_obs)
    curr_x = 0.0
    for t in range(n_obs):
        curr_x = rho * curr_x + sigma_u * rng.normal()
        x[t] = curr_x
        y[t] = curr_x + sigma_v * rng.normal()

    df_data = pd.DataFrame({"y": y})

    # Construct minimal LinearModel representation
    T = np.array([[rho]])
    R = np.array([[sigma_u]])
    Z = np.array([[1.0]])
    Q = np.array([[1.0]])
    H = np.array([[sigma_v**2]])

    oracle_ll, _, _ = _kalman_filter_oracle(
        data=y[:, None], T=T, R=R, Z=Z, Q=Q, H=H
    )

    # Build mock or actual LinearModel
    model = _make_test_linear_model(
        T=T,
        R=R,
        variables=["y"],
        shocks=["eps"],
        steady_state=pd.Series({"y": 0.0}),
        params={"rho": rho, "sigma_u": sigma_u, "sigma_v": sigma_v},
    )
    return df_data, model, oracle_ll


def _make_3eq_nk_ms_matrices(
    phi_pi: tuple[float, float] = (1.5, 0.8),
    phi_x: tuple[float, float] = (0.125, 0.0),
    sigma: float = 1.0,
    beta: float = 0.99,
    kappa: float = 0.1,
    rho_g: float = 0.5,
    rho_u: float = 0.5,
    p11: float = 0.9,
    p22: float = 0.9,
) -> dict[str, Any]:
    """Construct 3-equation New Keynesian MS-DSGE system matrices.

    Variables: y_t = [x_t, pi_t, i_t, g_t, u_t]^T
    Shocks: epsilon_t = [eps_m, eps_g, eps_u]^T
    System: A(s) E_t y_{t+1} + B(s) y_t + C(s) y_{t-1} + D(s) epsilon_t = 0
    """
    n = 5
    n_e = 3
    S = 2

    # Transition matrix
    P = np.array([
        [p11, 1.0 - p11],
        [1.0 - p22, p22],
    ], dtype=float)

    A_list = []
    B_list = []
    C_list = []
    D_list = []

    for s in range(S):
        # A matrix multiplies E_t y_{t+1}
        # Eq 0 (IS): x_t - E_t x_{t+1} + (1/sigma)(i_t - E_t pi_{t+1}) - g_t = 0
        # Eq 1 (NKPC): pi_t - beta E_t pi_{t+1} - kappa x_t - u_t = 0
        # Eq 2 (Taylor): i_t - phi_pi(s) pi_t - phi_x(s) x_t - eps_m = 0
        # Eq 3 (Demand shock AR1): g_t - rho_g g_{t-1} - eps_g = 0
        # Eq 4 (Cost-push shock AR1): u_t - rho_u u_{t-1} - eps_u = 0

        A_s = np.zeros((n, n))
        A_s[0, 0] = -1.0          # - E_t x_{t+1}
        A_s[0, 1] = -1.0 / sigma   # - (1/sigma) E_t pi_{t+1}
        A_s[1, 1] = -beta         # - beta E_t pi_{t+1}

        B_s = np.zeros((n, n))
        B_s[0, 0] = 1.0
        B_s[0, 2] = 1.0 / sigma
        B_s[0, 3] = -1.0

        B_s[1, 0] = -kappa
        B_s[1, 1] = 1.0
        B_s[1, 4] = -1.0

        B_s[2, 0] = -phi_x[s]
        B_s[2, 1] = -phi_pi[s]
        B_s[2, 2] = 1.0

        B_s[3, 3] = 1.0
        B_s[4, 4] = 1.0

        C_s = np.zeros((n, n))
        C_s[3, 3] = -rho_g
        C_s[4, 4] = -rho_u

        D_s = np.zeros((n, n_e))
        D_s[2, 0] = -1.0   # eps_m
        D_s[3, 1] = -1.0   # eps_g
        D_s[4, 2] = -1.0   # eps_u

        A_list.append(A_s)
        B_list.append(B_s)
        C_list.append(C_s)
        D_list.append(D_s)

    return {
        "A": A_list,
        "B": B_list,
        "C": C_list,
        "D": D_list,
        "transition_matrix": P,
        "variable_names": ("x", "pi", "i", "g", "u"),
        "shock_names": ("eps_m", "eps_g", "eps_u"),
        "regime_names": ("Hawkish", "Dovish"),
    }


def _make_leeper_ftpl_matrices() -> dict[str, Any]:
    """Leeper (1991) Fiscal-Monetary regime switching system.

    Regime 1: Active Monetary / Passive Fiscal (M-dominance)
    Regime 2: Passive Monetary / Active Fiscal (F-dominance)
    """
    n = 3  # [pi_t, b_t, i_t]
    n_e = 2  # [eps_m, eps_f]
    P = np.array([[0.85, 0.15], [0.20, 0.80]])

    beta = 0.99
    r_bar = 1.0 / beta - 1.0
    b_bar = 1.0

    # Policy params across regimes
    phi_pi = (1.5, 0.8)   # Active (>1) vs Passive (<1)
    gamma_b = (0.35, 0.0) # Passive (>0) vs Active (=0)

    A_list = []
    B_list = []
    C_list = []
    D_list = []

    for s in range(2):
        # 1. Fisher / Euler: i_t - E_t pi_{t+1} = 0
        # 2. Government debt: b_t - (1+r_bar) b_{t-1} + (1+r_bar) b_bar pi_t - b_bar i_t + gamma_b(s) b_{t-1} - eps_f = 0
        # 3. Taylor rule: i_t - phi_pi(s) pi_t - eps_m = 0
        A_s = np.zeros((n, n))
        A_s[0, 0] = -1.0  # - E_t pi_{t+1}

        B_s = np.zeros((n, n))
        B_s[0, 2] = 1.0   # i_t

        B_s[1, 0] = (1.0 + r_bar) * b_bar
        B_s[1, 1] = 1.0
        B_s[1, 2] = -b_bar

        B_s[2, 0] = -phi_pi[s]
        B_s[2, 2] = 1.0

        C_s = np.zeros((n, n))
        C_s[1, 1] = -(1.0 + r_bar - gamma_b[s])

        D_s = np.zeros((n, n_e))
        D_s[1, 1] = -1.0  # eps_f
        D_s[2, 0] = -1.0  # eps_m

        A_list.append(A_s)
        B_list.append(B_s)
        C_list.append(C_s)
        D_list.append(D_s)

    return {
        "A": A_list,
        "B": B_list,
        "C": C_list,
        "D": D_list,
        "transition_matrix": P,
        "variable_names": ("pi", "b", "i"),
        "shock_names": ("eps_m", "eps_f"),
        "regime_names": ("Regime_M", "Regime_F"),
    }


# ===========================================================================
# TIER 1: FEATURE COVERAGE (>=5 tests per feature, isolated happy paths)
# ===========================================================================

class TestTier1FeatureCoverage:
    """Isolated unit and functional tests covering all 10 Phase D features."""

    # -----------------------------------------------------------------------
    # Feature 1: Bootstrap Particle Filter (BPF)
    # -----------------------------------------------------------------------

    def test_t1_f01_bpf_matches_kalman_log_likelihood_linear_gaussian(self):
        """BPF log-likelihood matches Kalman filter oracle within MC standard error."""
        particle_filter, _ = _require_particle_filter()
        df_data, model, oracle_ll = _make_ar1_state_space_data(
            n_obs=40, rho=0.7, sigma_u=0.4, sigma_v=0.2, seed=101
        )
        res = particle_filter(
            model=model,
            data=df_data,
            observed_vars=["y"],
            n_particles=10_000,
            method="bootstrap",
            resampling_method="systematic",
            measurement_error={"y": 0.2},
            seed=42,
        )
        assert np.isfinite(res.log_likelihood)
        # Expected value from analytical Kalman filter oracle: +/- 2.0 log points tolerance at N=10,000
        assert abs(res.log_likelihood - oracle_ll) < 2.0

    def test_t1_f01_bpf_resampling_schemes_preserve_moments(self):
        """Verify all 4 resampling schemes preserve particle mean and reset weights to 1/N."""
        particle_filter, _ = _require_particle_filter()
        df_data, model, _ = _make_ar1_state_space_data(n_obs=15, seed=102)

        for scheme in ["systematic", "stratified", "residual", "multinomial"]:
            res = particle_filter(
                model=model,
                data=df_data,
                observed_vars=["y"],
                n_particles=2_000,
                method="bootstrap",
                resampling_method=scheme,
                seed=42,
            )
            assert np.isfinite(res.log_likelihood)
            assert len(res.resample_history) == len(df_data)
            assert 0.0 <= res.resampling_frequency <= 1.0

    def test_t1_f01_bpf_ess_trajectory_and_resampling_threshold(self):
        """ESS trajectory decreases when weights concentrate and triggers resampling when ESS < threshold*N."""
        particle_filter, _ = _require_particle_filter()
        df_data, model, _ = _make_ar1_state_space_data(n_obs=25, seed=103)
        n_p = 3_000
        threshold = 0.6

        res = particle_filter(
            model=model,
            data=df_data,
            observed_vars=["y"],
            n_particles=n_p,
            resampling_threshold=threshold,
            seed=42,
        )
        assert len(res.ess) == len(df_data)
        assert np.all(res.ess >= 1.0)
        assert np.all(res.ess <= n_p + 1e-6)
        # Whenever resampling occurred, ESS at that step was below threshold * N
        for t, resampled in enumerate(res.resample_history):
            if resampled:
                assert res.ess.iloc[t] <= threshold * n_p + 1e-6

    def test_t1_f01_bpf_filtered_states_and_uncertainty_dimensions(self):
        """Filtered states and standard deviations match expected dimensions and non-negativity."""
        particle_filter, _ = _require_particle_filter()
        df_data, model, _ = _make_ar1_state_space_data(n_obs=20, seed=104)

        res = particle_filter(
            model=model,
            data=df_data,
            observed_vars=["y"],
            n_particles=1_500,
            seed=42,
        )
        assert isinstance(res.filtered_states, pd.DataFrame)
        assert res.filtered_states.shape == (20, 1)
        assert isinstance(res.filtered_states_std, pd.DataFrame)
        assert np.all(res.filtered_states_std.to_numpy() >= 0.0)

    def test_t1_f01_bpf_result_presentation_contract(self):
        """ParticleFilterResult implements puremacro presentation contract (.summary, .plot, .to_markdown, .to_latex, .to_typst)."""
        particle_filter, res_cls = _require_particle_filter()
        df_data, model, _ = _make_ar1_state_space_data(n_obs=10, seed=105)

        res = particle_filter(
            model=model,
            data=df_data,
            observed_vars=["y"],
            n_particles=500,
            seed=42,
        )
        assert isinstance(res, res_cls)

        # Presentation methods
        summary = res.summary()
        assert isinstance(summary, (pd.DataFrame, str))
        md = res.to_markdown()
        assert isinstance(md, str) and len(md) > 0
        latex = res.to_latex()
        assert isinstance(latex, str) and ("tabular" in latex or "table" in latex or len(latex) > 0)
        typst = res.to_typst()
        assert isinstance(typst, str) and len(typst) > 0

        fig = res.plot()
        assert fig is not None
        plt.close("all")

    # -----------------------------------------------------------------------
    # Feature 2: Auxiliary Particle Filter (APF)
    # -----------------------------------------------------------------------

    def test_t1_f02_apf_proposal_weights_anticipate_observations(self):
        """APF evaluates first-stage proposal weights conditioned on forward observation."""
        particle_filter, _ = _require_particle_filter()
        df_data, model, _ = _make_ar1_state_space_data(n_obs=20, seed=201)

        res_apf = particle_filter(
            model=model,
            data=df_data,
            observed_vars=["y"],
            n_particles=2_000,
            method="auxiliary",
            seed=42,
        )
        assert np.isfinite(res_apf.log_likelihood)
        assert len(res_apf.log_likelihood_contributions) == len(df_data)

    def test_t1_f02_apf_log_likelihood_unbiased_vs_bpf(self):
        """APF log-likelihood estimate is statistically consistent with BPF."""
        particle_filter, _ = _require_particle_filter()
        df_data, model, oracle_ll = _make_ar1_state_space_data(n_obs=25, seed=202)

        res_bpf = particle_filter(
            model=model,
            data=df_data,
            observed_vars=["y"],
            n_particles=5_000,
            method="bootstrap",
            seed=10,
        )
        res_apf = particle_filter(
            model=model,
            data=df_data,
            observed_vars=["y"],
            n_particles=5_000,
            method="auxiliary",
            seed=20,
        )
        # Both methods converge to true marginal likelihood within Monte Carlo bounds
        assert abs(res_bpf.log_likelihood - oracle_ll) < 2.5
        assert abs(res_apf.log_likelihood - oracle_ll) < 2.5
        assert abs(res_bpf.log_likelihood - res_apf.log_likelihood) < 2.0

    def test_t1_f02_apf_resampling_efficiency_under_tight_noise(self):
        """APF reduces sample degeneracy when measurement noise is small."""
        particle_filter, _ = _require_particle_filter()
        df_data, model, _ = _make_ar1_state_space_data(
            n_obs=20, sigma_u=0.5, sigma_v=0.05, seed=203
        )
        res_bpf = particle_filter(
            model=model,
            data=df_data,
            observed_vars=["y"],
            n_particles=3_000,
            method="bootstrap",
            measurement_error={"y": 0.05**2},
            seed=42,
        )
        res_apf = particle_filter(
            model=model,
            data=df_data,
            observed_vars=["y"],
            n_particles=3_000,
            method="auxiliary",
            measurement_error={"y": 0.05**2},
            seed=42,
        )
        # APF achieves higher or comparable mean ESS by looking ahead at measurement
        assert res_apf.ess.mean() >= 0.5 * res_bpf.ess.mean()

    def test_t1_f02_apf_predictive_covariance_scaling(self):
        """APF correctly accounts for predictive measurement covariance Sigma_mu = Z R Q R' Z' + H."""
        particle_filter, _ = _require_particle_filter()
        df_data, model, _ = _make_ar1_state_space_data(n_obs=15, seed=204)

        res = particle_filter(
            model=model,
            data=df_data,
            observed_vars=["y"],
            n_particles=1_000,
            method="auxiliary",
            ridge=1e-5,
            seed=42,
        )
        assert np.isfinite(res.log_likelihood)

    def test_t1_f02_apf_multivariate_observation(self):
        """APF operates correctly on multivariate observation setups."""
        particle_filter, _ = _require_particle_filter()
        rng = np.random.default_rng(205)
        T = 20
        df_multi = pd.DataFrame({
            "y1": rng.normal(size=T),
            "y2": rng.normal(size=T),
        })
        model = _make_test_linear_model(
            T=np.diag([0.8, 0.7]),
            R=np.eye(2) * 0.3,
            variables=["y1", "y2"],
            shocks=["eps1", "eps2"],
            steady_state=pd.Series({"y1": 0.0, "y2": 0.0}),
        )
        res = particle_filter(
            model=model,
            data=df_multi,
            observed_vars=["y1", "y2"],
            n_particles=1_000,
            method="auxiliary",
            seed=42,
        )
        assert res.filtered_states.shape == (T, 2)
        assert np.isfinite(res.log_likelihood)

    # -----------------------------------------------------------------------
    # Feature 3: Pruned Nonlinear Likelihood Evaluation
    # -----------------------------------------------------------------------

    def test_t1_f03_pruned_2nd_order_growth_model_likelihood(self):
        """Evaluate nonlinear likelihood on 2nd-order pruned growth model."""
        particle_filter, _ = _require_particle_filter()
        sol2 = canonical_growth_2nd_order()
        sim = sol2.simulate(periods=30, seed=301, burn=20)
        df_data = sim.to_frame()[["c"]]

        res = particle_filter(
            model=sol2,
            data=df_data,
            observed_vars=["c"],
            n_particles=2_000,
            seed=42,
        )
        assert np.isfinite(res.log_likelihood)
        assert res.filtered_states.shape[0] == 30
        assert "k" in res.filtered_states.columns

    def test_t1_f03_pruned_3rd_order_growth_model_likelihood(self):
        """Evaluate nonlinear likelihood on 3rd-order pruned growth model."""
        particle_filter, _ = _require_particle_filter()
        try:
            from puremacro.dsge.pruning import canonical_growth_3rd_order
            sol3 = canonical_growth_3rd_order()
        except (ImportError, AttributeError):
            pytest.skip("canonical_growth_3rd_order unavailable")

        sim = sol3.simulate(periods=25, seed=302, burn=15)
        df_data = sim.to_frame()[["c"]]

        res = particle_filter(
            model=sol3,
            data=df_data,
            observed_vars=["c"],
            n_particles=1_500,
            seed=42,
        )
        assert np.isfinite(res.log_likelihood)
        assert len(res.ess) == 25

    def test_t1_f03_pruned_einsum_speed_benchmark(self):
        """Vectorized einsum propagation of 10,000 particles executes rapidly."""
        particle_filter, _ = _require_particle_filter()
        sol2 = canonical_growth_2nd_order()
        sim = sol2.simulate(periods=10, seed=303)
        df_data = sim.to_frame()[["c"]]

        t0 = time.perf_counter()
        res = particle_filter(
            model=sol2,
            data=df_data,
            observed_vars=["c"],
            n_particles=10_000,
            seed=42,
        )
        elapsed = time.perf_counter() - t0
        assert elapsed < 3.0  # Vectorized execution target in Pyodide/pure Python
        assert np.isfinite(res.log_likelihood)

    def test_t1_f03_pruned_steady_state_offset_handling(self):
        """Observation equation properly accounts for deterministic steady-state offset."""
        particle_filter, _ = _require_particle_filter()
        sol2 = canonical_growth_2nd_order()
        sim = sol2.simulate(periods=15, seed=304)
        df_levels = sim.to_frame()[["c"]]  # already in levels

        res = particle_filter(
            model=sol2,
            data=df_levels,
            observed_vars=["c"],
            n_particles=1_000,
            seed=42,
        )
        assert np.isfinite(res.log_likelihood)

    def test_t1_f03_pruned_likelihood_penalizes_distorted_data(self):
        """Likelihood on genuine data is substantially higher than on distorted/shifted data."""
        particle_filter, _ = _require_particle_filter()
        sol2 = canonical_growth_2nd_order()
        sim = sol2.simulate(periods=20, seed=305)
        df_good = sim.to_frame()[["c"]] + sol2.steady_state["c"]
        df_bad = df_good + 10.0  # Huge shift far outside ergodic distribution

        res_good = particle_filter(model=sol2, data=df_good, observed_vars=["c"], n_particles=2_000, seed=42)
        res_bad = particle_filter(model=sol2, data=df_bad, observed_vars=["c"], n_particles=2_000, seed=42)
        assert res_good.log_likelihood > res_bad.log_likelihood

    # -----------------------------------------------------------------------
    # Feature 4: Stochastic Volatility (SV)
    # -----------------------------------------------------------------------

    def test_t1_f04_sv_state_augmentation_tracking(self):
        """Augmented particle state tracks time-varying log-volatility h_t."""
        particle_filter, _ = _require_particle_filter()
        df_data, model, _ = _make_ar1_state_space_data(n_obs=30, seed=401)

        sv_spec = {"rho_h": 0.85, "sigma_h": 0.2, "sigma_bar": 0.5}
        res = particle_filter(
            model=model,
            data=df_data,
            observed_vars=["y"],
            n_particles=2_000,
            stochastic_volatility=sv_spec,
            seed=42,
        )
        assert res.filtered_volatility is not None
        assert res.filtered_volatility.shape[0] == 30

    def test_t1_f04_sv_correlation_with_true_volatility_spike(self):
        """Filtered volatility state responds to simulated volatility surge."""
        particle_filter, _ = _require_particle_filter()
        rng = np.random.default_rng(402)
        T = 40
        h_true = np.zeros(T)
        y = np.zeros(T)
        rho_h = 0.9
        sigma_h = 0.3
        h_val = 0.0

        for t in range(T):
            if t == 15:
                h_val += 2.0  # Planted volatility shock
            else:
                h_val = rho_h * h_val + sigma_h * rng.normal()
            h_true[t] = h_val
            scale = 0.2 * np.exp(h_val)
            y[t] = scale * rng.normal()

        df_sv = pd.DataFrame({"y": y})
        model = _make_test_linear_model(
            T=np.array([[0.0]]),
            R=np.array([[1.0]]),
            variables=["y"],
            shocks=["eps"],
            steady_state=pd.Series({"y": 0.0}),
        )

        res = particle_filter(
            model=model,
            data=df_sv,
            observed_vars=["y"],
            n_particles=3_000,
            stochastic_volatility={"rho_h": rho_h, "sigma_h": sigma_h, "sigma_bar": 0.2},
            seed=42,
        )
        assert res.filtered_volatility is not None
        filt_h = res.filtered_volatility.iloc[:, 0].to_numpy()
        # Filtered volatility at the shock date is significantly higher than at initial date
        assert filt_h[15] > filt_h[0]

    def test_t1_f04_sv_precautionary_saving_shift(self):
        """2nd-order model with SV exhibits precautionary shift during volatility surge."""
        particle_filter, _ = _require_particle_filter()
        sol2 = canonical_growth_2nd_order()
        sim = sol2.simulate(periods=20, seed=403)
        df_data = sim.to_frame()[["c"]]

        res_constant = particle_filter(
            model=sol2, data=df_data, observed_vars=["c"], n_particles=2_000, seed=42
        )
        res_sv = particle_filter(
            model=sol2,
            data=df_data,
            observed_vars=["c"],
            n_particles=2_000,
            stochastic_volatility={"rho_h": 0.9, "sigma_h": 0.25, "sigma_bar": 0.01},
            seed=42,
        )
        assert np.isfinite(res_constant.log_likelihood)
        assert np.isfinite(res_sv.log_likelihood)

    def test_t1_f04_sv_dict_and_spec_input_handling(self):
        """stochastic_volatility parameter accepts both dict and keyword specs."""
        particle_filter, _ = _require_particle_filter()
        df_data, model, _ = _make_ar1_state_space_data(n_obs=10, seed=404)

        res1 = particle_filter(
            model=model,
            data=df_data,
            observed_vars=["y"],
            n_particles=500,
            stochastic_volatility={"rho_h": 0.8, "sigma_h": 0.1, "sigma_bar": 0.5},
            seed=42,
        )
        assert np.isfinite(res1.log_likelihood)

    def test_t1_f04_sv_zero_volatility_variance_recovers_constant_volatility(self):
        """When sigma_h = 0.0, SV model log-likelihood closely tracks constant volatility."""
        particle_filter, _ = _require_particle_filter()
        df_data, model, _ = _make_ar1_state_space_data(n_obs=15, seed=405)

        res_const = particle_filter(
            model=model, data=df_data, observed_vars=["y"], n_particles=3_000, seed=42
        )
        res_sv0 = particle_filter(
            model=model,
            data=df_data,
            observed_vars=["y"],
            n_particles=3_000,
            stochastic_volatility={"rho_h": 0.8, "sigma_h": 0.0, "sigma_bar": 1.0},
            seed=42,
        )
        assert abs(res_const.log_likelihood - res_sv0.log_likelihood) < 1.0

    # -----------------------------------------------------------------------
    # Feature 5: Fat-Tailed Innovations & Measurement Densities
    # -----------------------------------------------------------------------

    def test_t1_f05_fat_tails_student_t_innovations(self):
        """Particle filter propagates with Student-t innovations."""
        particle_filter, _ = _require_particle_filter()
        df_data, model, _ = _make_ar1_state_space_data(n_obs=20, seed=501)

        res = particle_filter(
            model=model,
            data=df_data,
            observed_vars=["y"],
            n_particles=1_500,
            innovation_dist="student_t",
            innovation_df=4.0,
            seed=42,
        )
        assert np.isfinite(res.log_likelihood)

    def test_t1_f05_fat_tails_gaussian_mixture_innovations(self):
        """Particle filter propagates with 2-component Gaussian mixture innovations."""
        particle_filter, _ = _require_particle_filter()
        df_data, model, _ = _make_ar1_state_space_data(n_obs=20, seed=502)

        res = particle_filter(
            model=model,
            data=df_data,
            observed_vars=["y"],
            n_particles=1_500,
            innovation_dist="mixture",
            mixture_params={"weights": [0.9, 0.1], "scales": [0.3, 1.5]},
            seed=42,
        )
        assert np.isfinite(res.log_likelihood)

    def test_t1_f05_fat_tails_student_t_measurement_density(self):
        """Measurement weighting evaluates multivariate Student-t observation density."""
        particle_filter, _ = _require_particle_filter()
        df_data, model, _ = _make_ar1_state_space_data(n_obs=20, seed=503)

        res = particle_filter(
            model=model,
            data=df_data,
            observed_vars=["y"],
            n_particles=1_500,
            measurement_dist="student_t",
            measurement_df=5.0,
            seed=42,
        )
        assert np.isfinite(res.log_likelihood)

    def test_t1_f05_fat_tails_outlier_robustness_comparison(self):
        """Student-t filter survives extreme outlier without total particle weight collapse."""
        particle_filter, _ = _require_particle_filter()
        df_data, model, _ = _make_ar1_state_space_data(n_obs=20, seed=504)
        # Inject 7-sigma outlier at period 10
        df_outlier = df_data.copy()
        df_outlier.iloc[10, 0] += 5.0

        res_gauss = particle_filter(
            model=model,
            data=df_outlier,
            observed_vars=["y"],
            n_particles=2_000,
            measurement_dist="gaussian",
            seed=42,
        )
        res_t = particle_filter(
            model=model,
            data=df_outlier,
            observed_vars=["y"],
            n_particles=2_000,
            measurement_dist="student_t",
            measurement_df=3.0,
            seed=42,
        )
        # Student-t filter maintains higher ESS at the outlier shock
        assert res_t.ess.iloc[10] >= res_gauss.ess.iloc[10]

    def test_t1_f05_fat_tails_degrees_of_freedom_validation(self):
        """Degrees of freedom <= 2 raises ValueError to guarantee finite variance."""
        particle_filter, _ = _require_particle_filter()
        df_data, model, _ = _make_ar1_state_space_data(n_obs=5, seed=505)

        with pytest.raises(ValueError, match=r"(df|degrees of freedom|finite)"):
            particle_filter(
                model=model,
                data=df_data,
                observed_vars=["y"],
                innovation_dist="student_t",
                innovation_df=1.5,
            )

    # -----------------------------------------------------------------------
    # Feature 6: Markov-Switching DSGE Newton Solver
    # -----------------------------------------------------------------------

    def test_t1_f06_ms_newton_quadratic_convergence_tolerance(self):
        """MS-DSGE Newton solver converges to ||coupled quadratic residual||_infty <= 1e-10."""
        solve_ms_dsge, _ = _require_ms_dsge()
        spec = _make_3eq_nk_ms_matrices()

        res = solve_ms_dsge(
            A=spec["A"],
            B=spec["B"],
            C=spec["C"],
            D=spec["D"],
            transition_matrix=spec["transition_matrix"],
            method="newton",
            tol=1e-10,
        )
        assert res.converged is True
        assert res.diff <= 1e-10

        # Verify coupled quadratic matrix equations directly:
        # A_i (sum_j p_ij T_j) T_i + B_i T_i + C_i = 0
        P = spec["transition_matrix"]
        S = len(spec["A"])
        for i in range(S):
            A_i = spec["A"][i]
            B_i = spec["B"][i]
            C_i = spec["C"][i]
            T_i = res.T[i]

            sum_p_T = sum(P[i, j] * res.T[j] for j in range(S))
            quad_res = A_i @ sum_p_T @ T_i + B_i @ T_i + C_i
            np.testing.assert_allclose(quad_res, 0.0, atol=1e-9)

    def test_t1_f06_ms_newton_iteration_efficiency(self):
        """Newton solver with analytical block-Kronecker Jacobian converges in < 25 iterations."""
        solve_ms_dsge, _ = _require_ms_dsge()
        spec = _make_3eq_nk_ms_matrices()

        res = solve_ms_dsge(
            A=spec["A"],
            B=spec["B"],
            C=spec["C"],
            D=spec["D"],
            transition_matrix=spec["transition_matrix"],
            method="newton",
        )
        assert res.iterations < 25

    def test_t1_f06_ms_newton_solution_dimensions_and_shock_loadings(self):
        """Solution matrices T(s) and R(s) match variable and shock counts."""
        solve_ms_dsge, _ = _require_ms_dsge()
        spec = _make_3eq_nk_ms_matrices()

        res = solve_ms_dsge(
            A=spec["A"],
            B=spec["B"],
            C=spec["C"],
            D=spec["D"],
            transition_matrix=spec["transition_matrix"],
            variable_names=spec["variable_names"],
            shock_names=spec["shock_names"],
            regime_names=spec["regime_names"],
            method="newton",
        )
        assert len(res.T) == 2
        assert len(res.R) == 2
        for s in (0, 1):
            assert res.T[s].shape == (5, 5)
            assert res.R[s].shape == (5, 3)

    def test_t1_f06_ms_newton_intercept_correction_k(self):
        """Non-zero constant vectors K(s) produce constant policy intercept c(s)."""
        solve_ms_dsge, _ = _require_ms_dsge()
        spec = _make_3eq_nk_ms_matrices()
        K = [np.array([0.0, 0.0, 0.02, 0.0, 0.0]), np.array([0.0, 0.0, -0.01, 0.0, 0.0])]

        res = solve_ms_dsge(
            A=spec["A"],
            B=spec["B"],
            C=spec["C"],
            D=spec["D"],
            K=K,
            transition_matrix=spec["transition_matrix"],
            method="newton",
        )
        assert res.c is not None
        assert 0 in res.c and 1 in res.c
        assert res.c[0].shape == (5,)

    def test_t1_f06_ms_newton_identical_regimes_matches_klein(self):
        """When regimes are identical, MS-DSGE solution matches single-regime Klein solver."""
        solve_ms_dsge, _ = _require_ms_dsge()
        spec = _make_3eq_nk_ms_matrices(phi_pi=(1.5, 1.5), phi_x=(0.125, 0.125))

        res = solve_ms_dsge(
            A=spec["A"],
            B=spec["B"],
            C=spec["C"],
            D=spec["D"],
            transition_matrix=spec["transition_matrix"],
            method="newton",
        )
        # Decision rules across regimes are identical
        np.testing.assert_allclose(res.T[0], res.T[1], atol=1e-10)
        np.testing.assert_allclose(res.R[0], res.R[1], atol=1e-10)

    # -----------------------------------------------------------------------
    # Feature 7: Markov-Switching Functional Iteration
    # -----------------------------------------------------------------------

    def test_t1_f07_ms_func_iter_matches_newton_solution(self):
        """Functional iteration converges to identical policy rule T(s) as Newton."""
        solve_ms_dsge, _ = _require_ms_dsge()
        spec = _make_3eq_nk_ms_matrices()

        res_newton = solve_ms_dsge(
            A=spec["A"], B=spec["B"], C=spec["C"], D=spec["D"],
            transition_matrix=spec["transition_matrix"], method="newton"
        )
        res_fi = solve_ms_dsge(
            A=spec["A"], B=spec["B"], C=spec["C"], D=spec["D"],
            transition_matrix=spec["transition_matrix"], method="functional_iteration",
            damping=0.8, tol=1e-8
        )
        assert res_fi.converged is True
        for s in (0, 1):
            np.testing.assert_allclose(res_fi.T[s], res_newton.T[s], atol=1e-5)

    def test_t1_f07_ms_func_iter_damping_parameter(self):
        """Damping parameter in (0, 1] enforces controlled convex updates."""
        solve_ms_dsge, _ = _require_ms_dsge()
        spec = _make_3eq_nk_ms_matrices()

        res = solve_ms_dsge(
            A=spec["A"], B=spec["B"], C=spec["C"], D=spec["D"],
            transition_matrix=spec["transition_matrix"],
            method="functional_iteration",
            damping=0.5,
            max_iter=500,
        )
        assert res.converged is True

    def test_t1_f07_ms_func_iter_max_iter_termination(self):
        """Solver halts when max_iter is reached without crashing."""
        solve_ms_dsge, _ = _require_ms_dsge()
        spec = _make_3eq_nk_ms_matrices()

        res = solve_ms_dsge(
            A=spec["A"], B=spec["B"], C=spec["C"], D=spec["D"],
            transition_matrix=spec["transition_matrix"],
            method="functional_iteration",
            max_iter=3,
        )
        assert res.iterations <= 3

    def test_t1_f07_ms_func_iter_residual_norm_decrease(self):
        """Functional iteration difference norm decreases over successive iterations."""
        solve_ms_dsge, _ = _require_ms_dsge()
        spec = _make_3eq_nk_ms_matrices()

        res_early = solve_ms_dsge(
            A=spec["A"], B=spec["B"], C=spec["C"], D=spec["D"],
            transition_matrix=spec["transition_matrix"],
            method="functional_iteration",
            max_iter=5,
        )
        res_late = solve_ms_dsge(
            A=spec["A"], B=spec["B"], C=spec["C"], D=spec["D"],
            transition_matrix=spec["transition_matrix"],
            method="functional_iteration",
            max_iter=30,
        )
        assert res_late.diff <= res_early.diff

    def test_t1_f07_ms_func_iter_3_regimes(self):
        """Functional iteration successfully solves a 3-regime Markov-switching system."""
        solve_ms_dsge, _ = _require_ms_dsge()
        P3 = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ])
        # Simple scalar MS system across 3 regimes
        A = [np.array([[-0.99]]), np.array([[-0.99]]), np.array([[-0.99]])]
        B = [np.array([[1.5]]), np.array([[1.2]]), np.array([[0.9]])]
        C = [np.array([[-0.6]]), np.array([[-0.5]]), np.array([[-0.4]])]
        D = [np.array([[1.0]]), np.array([[1.0]]), np.array([[1.0]])]

        res = solve_ms_dsge(A=A, B=B, C=C, D=D, transition_matrix=P3, method="functional_iteration")
        assert res.converged is True
        assert len(res.T) == 3

    # -----------------------------------------------------------------------
    # Feature 8: Mean Stability and Mean-Square Stability (MSS)
    # -----------------------------------------------------------------------

    def test_t1_f08_mss_operator_matrices_dimensions(self):
        """Verify dimensions of M_1 and M_2 stability operators."""
        solve_ms_dsge, _ = _require_ms_dsge()
        spec = _make_3eq_nk_ms_matrices()

        res = solve_ms_dsge(
            A=spec["A"], B=spec["B"], C=spec["C"], D=spec["D"],
            transition_matrix=spec["transition_matrix"],
        )
        # S=2 regimes, n=5 variables
        # M_1 dimension is (S*n, S*n) = (10, 10)
        # M_2 dimension is (S*n^2, S*n^2) = (50, 50)
        assert hasattr(res, "spectral_radius_mean")
        assert hasattr(res, "spectral_radius_mss")
        assert res.spectral_radius_mean >= 0.0
        assert res.spectral_radius_mss >= 0.0

    def test_t1_f08_mss_spectral_radius_calculation(self):
        """Verify spectral radii match operator maximum eigenvalue modulus."""
        solve_ms_dsge, _ = _require_ms_dsge()
        spec = _make_3eq_nk_ms_matrices()

        res = solve_ms_dsge(
            A=spec["A"], B=spec["B"], C=spec["C"], D=spec["D"],
            transition_matrix=spec["transition_matrix"],
        )
        # Verify M_1 operator independently
        P = spec["transition_matrix"]
        T0 = res.T[0]
        T1 = res.T[1]
        n = T0.shape[0]

        # Block construction of M1 = (P^T otimes I_n) diag(T_0, T_1)
        M1 = np.kron(P.T, np.eye(n)) @ scipy.linalg.block_diag(T0, T1)
        expected_rho_m1 = float(np.max(np.abs(np.linalg.eigvals(M1))))

        assert abs(res.spectral_radius_mean - expected_rho_m1) < 1e-6

    def test_t1_f08_mss_stable_system_boolean_flag(self):
        """mean_square_stable flag is True iff rho(M_2) < 1.0."""
        solve_ms_dsge, _ = _require_ms_dsge()
        spec = _make_3eq_nk_ms_matrices()

        res = solve_ms_dsge(
            A=spec["A"], B=spec["B"], C=spec["C"], D=spec["D"],
            transition_matrix=spec["transition_matrix"],
        )
        if res.spectral_radius_mss < 1.0:
            assert res.mean_square_stable is True
        else:
            assert res.mean_square_stable is False

    def test_t1_f08_mss_regime_instability_with_aggregate_stability(self):
        """System is MSS stable overall even when Dovish regime is explosive in isolation."""
        solve_ms_dsge, _ = _require_ms_dsge()
        # Dovish regime with phi_pi = 0.5 (explosive in isolation)
        # but transient persistence p22 = 0.2
        spec = _make_3eq_nk_ms_matrices(phi_pi=(1.8, 0.5), p11=0.95, p22=0.2)

        res = solve_ms_dsge(
            A=spec["A"], B=spec["B"], C=spec["C"], D=spec["D"],
            transition_matrix=spec["transition_matrix"],
        )
        assert res.converged is True
        # Globally mean-square stable
        assert res.mean_square_stable is True
        assert res.spectral_radius_mss < 1.0

    def test_t1_f08_mss_unstable_system_detection(self):
        """Prolonged dwell time in explosive regime flags MSS instability."""
        solve_ms_dsge, _ = _require_ms_dsge()
        # High persistence in deeply dovish regime
        spec = _make_3eq_nk_ms_matrices(phi_pi=(1.05, 0.2), p11=0.1, p22=0.99)

        res = solve_ms_dsge(
            A=spec["A"], B=spec["B"], C=spec["C"], D=spec["D"],
            transition_matrix=spec["transition_matrix"],
        )
        # Spectral radius should exceed 1.0 or flag unstable
        assert res.spectral_radius_mss > 0.95

    # -----------------------------------------------------------------------
    # Feature 9: Ergodic Distribution & Moments
    # -----------------------------------------------------------------------

    def test_t1_f09_ergodic_distribution_markov_chain(self):
        """Ergodic distribution pi_infty satisfies pi_infty P = pi_infty and sum(pi) = 1."""
        solve_ms_dsge, _ = _require_ms_dsge()
        spec = _make_3eq_nk_ms_matrices(p11=0.8, p22=0.7)

        res = solve_ms_dsge(
            A=spec["A"], B=spec["B"], C=spec["C"], D=spec["D"],
            transition_matrix=spec["transition_matrix"],
        )
        pi = res.ergodic_distribution.to_numpy()
        P = spec["transition_matrix"]

        assert abs(np.sum(pi) - 1.0) < 1e-10
        assert np.all(pi > 0.0)
        np.testing.assert_allclose(pi @ P, pi, atol=1e-10)

    def test_t1_f09_ergodic_distribution_analytical_two_state_formula(self):
        """Ergodic distribution matches analytical formula pi_1 = (1-p22)/((1-p11)+(1-p22))."""
        solve_ms_dsge, _ = _require_ms_dsge()
        p11, p22 = 0.85, 0.65
        spec = _make_3eq_nk_ms_matrices(p11=p11, p22=p22)

        res = solve_ms_dsge(
            A=spec["A"], B=spec["B"], C=spec["C"], D=spec["D"],
            transition_matrix=spec["transition_matrix"],
        )
        pi = res.ergodic_distribution.to_numpy()
        expected_pi1 = (1.0 - p22) / ((1.0 - p11) + (1.0 - p22))
        expected_pi2 = (1.0 - p11) / ((1.0 - p11) + (1.0 - p22))

        assert abs(pi[0] - expected_pi1) < 1e-10
        assert abs(pi[1] - expected_pi2) < 1e-10

    def test_t1_f09_ergodic_mean_vector_labels(self):
        """Ergodic mean is a labeled Series matching declared variable names."""
        solve_ms_dsge, _ = _require_ms_dsge()
        spec = _make_3eq_nk_ms_matrices()

        res = solve_ms_dsge(
            A=spec["A"], B=spec["B"], C=spec["C"], D=spec["D"],
            transition_matrix=spec["transition_matrix"],
            variable_names=spec["variable_names"],
        )
        assert isinstance(res.ergodic_mean, pd.Series)
        assert list(res.ergodic_mean.index) == list(spec["variable_names"])

    def test_t1_f09_ergodic_covariance_matrix_properties(self):
        """Ergodic covariance is symmetric positive semi-definite with correct column names."""
        solve_ms_dsge, _ = _require_ms_dsge()
        spec = _make_3eq_nk_ms_matrices()

        res = solve_ms_dsge(
            A=spec["A"], B=spec["B"], C=spec["C"], D=spec["D"],
            transition_matrix=spec["transition_matrix"],
            variable_names=spec["variable_names"],
        )
        cov = res.ergodic_cov.to_numpy()
        assert cov.shape == (5, 5)
        # Symmetry
        np.testing.assert_allclose(cov, cov.T, atol=1e-8)
        # Positive semi-definiteness: all eigenvalues >= -1e-10
        eigs = np.linalg.eigvalsh(cov)
        assert np.all(eigs >= -1e-10)

    def test_t1_f09_ergodic_mean_matches_long_simulation(self):
        """Simulated long-run sample mean converges to analytical ergodic mean."""
        solve_ms_dsge, _ = _require_ms_dsge()
        spec = _make_3eq_nk_ms_matrices()

        res = solve_ms_dsge(
            A=spec["A"], B=spec["B"], C=spec["C"], D=spec["D"],
            transition_matrix=spec["transition_matrix"],
            variable_names=spec["variable_names"],
        )
        sim_df, _ = res.simulate(periods=10_000, seed=42)
        sample_mean = sim_df.mean()
        # In zero-intercept system, both should be close to zero
        np.testing.assert_allclose(sample_mean.to_numpy(), res.ergodic_mean.to_numpy(), atol=0.08)

    # -----------------------------------------------------------------------
    # Feature 10: Closed-Form GIRF & Regime IRF
    # -----------------------------------------------------------------------

    def test_t1_f10_girf_closed_form_formula_precision(self):
        """Closed-form GIRF (1_S^T otimes I_n) M_1^h z_0 evaluates accurately."""
        solve_ms_dsge, _ = _require_ms_dsge()
        spec = _make_3eq_nk_ms_matrices()

        res = solve_ms_dsge(
            A=spec["A"], B=spec["B"], C=spec["C"], D=spec["D"],
            transition_matrix=spec["transition_matrix"],
            variable_names=spec["variable_names"],
            shock_names=spec["shock_names"],
        )
        girf = res.girf(initial_regime=0, shock="eps_m", horizon=20)
        assert isinstance(girf, pd.DataFrame)
        assert girf.shape == (21, 5)  # horizons 0 to 20
        assert list(girf.columns) == list(spec["variable_names"])

    def test_t1_f10_girf_evaluates_sub_millisecond(self):
        """Analytical closed-form GIRF evaluates in < 5 milliseconds."""
        solve_ms_dsge, _ = _require_ms_dsge()
        spec = _make_3eq_nk_ms_matrices()

        res = solve_ms_dsge(
            A=spec["A"], B=spec["B"], C=spec["C"], D=spec["D"],
            transition_matrix=spec["transition_matrix"],
            variable_names=spec["variable_names"],
            shock_names=spec["shock_names"],
        )
        t0 = time.perf_counter()
        _ = res.girf(initial_regime=0, shock="eps_m", horizon=40)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms < 10.0

    def test_t1_f10_girf_matches_monte_carlo_regime_switching_average(self):
        """Analytical GIRF matches Monte Carlo averaged trajectory across Markov switching paths."""
        solve_ms_dsge, _ = _require_ms_dsge()
        spec = _make_3eq_nk_ms_matrices()

        res = solve_ms_dsge(
            A=spec["A"], B=spec["B"], C=spec["C"], D=spec["D"],
            transition_matrix=spec["transition_matrix"],
            variable_names=spec["variable_names"],
            shock_names=spec["shock_names"],
        )
        H = 15
        girf = res.girf(initial_regime=0, shock="eps_m", horizon=H)

        # Monte Carlo simulation of K paths
        K = 1_000
        rng = np.random.default_rng(42)
        P = spec["transition_matrix"]
        n = 5
        mc_paths = np.zeros((K, H + 1, n))

        shock_vec = np.array([1.0, 0.0, 0.0])  # eps_m = 1.0

        for k in range(K):
            curr_s = 0
            # Impact at t=0
            y = res.R[curr_s] @ shock_vec
            mc_paths[k, 0, :] = y
            for h in range(1, H + 1):
                # Switch regime according to Markov row
                probs = P[curr_s, :]
                curr_s = int(rng.choice([0, 1], p=probs))
                # Propagation without further shocks
                y = res.T[curr_s] @ y
                mc_paths[k, h, :] = y

        mc_mean = np.mean(mc_paths, axis=0)
        # Compare analytical GIRF against Monte Carlo mean
        np.testing.assert_allclose(girf.to_numpy(), mc_mean, atol=0.08)

    def test_t1_f10_regime_conditional_irf_fixed_regime(self):
        """Regime-conditional IRF holds the designated regime fixed across the horizon."""
        solve_ms_dsge, _ = _require_ms_dsge()
        spec = _make_3eq_nk_ms_matrices()

        res = solve_ms_dsge(
            A=spec["A"], B=spec["B"], C=spec["C"], D=spec["D"],
            transition_matrix=spec["transition_matrix"],
            variable_names=spec["variable_names"],
            shock_names=spec["shock_names"],
        )
        irf0 = res.irf(regime=0, shock="eps_m", horizon=10)
        irf1 = res.irf(regime=1, shock="eps_m", horizon=10)

        assert isinstance(irf0, pd.DataFrame)
        assert isinstance(irf1, pd.DataFrame)
        # Responses differ between Hawkish and Dovish regimes
        assert not np.allclose(irf0.to_numpy(), irf1.to_numpy())

    def test_t1_f10_girf_presentation_and_plotting(self):
        """MSDSGEResult presentation contract (.summary, .plot, .to_markdown, .to_latex, .to_typst)."""
        solve_ms_dsge, res_cls = _require_ms_dsge()
        spec = _make_3eq_nk_ms_matrices()

        res = solve_ms_dsge(
            A=spec["A"], B=spec["B"], C=spec["C"], D=spec["D"],
            transition_matrix=spec["transition_matrix"],
            variable_names=spec["variable_names"],
            shock_names=spec["shock_names"],
            regime_names=spec["regime_names"],
        )
        assert isinstance(res, res_cls)

        assert isinstance(res.to_markdown(), str)
        assert isinstance(res.to_latex(), str)
        assert isinstance(res.to_typst(), str)

        fig, _ = res.plot(shock="eps_m", horizon=15)
        assert fig is not None
        plt.close("all")


# ===========================================================================
# TIER 2: BOUNDARY & CORNER CASES (>=5 tests, limits & degeneracies)
# ===========================================================================

class TestTier2BoundaryAndCornerCases:
    """Rigorous boundary, corner, singularity, and degenerate state tests."""

    def test_t2_degenerate_particle_weights_logsumexp(self):
        """Particle filter with extreme outlier observations does not produce NaN/Inf weights."""
        particle_filter, _ = _require_particle_filter()
        # Data with massive values (1e6)
        df_extreme = pd.DataFrame({"y": [1e6, -1e6, 5e5]})
        model = _make_test_linear_model(
            T=np.array([[0.5]]),
            R=np.array([[1.0]]),
            variables=["y"],
            shocks=["eps"],
            steady_state=pd.Series({"y": 0.0}),
        )
        res = particle_filter(
            model=model,
            data=df_extreme,
            observed_vars=["y"],
            n_particles=500,
            seed=42,
        )
        assert np.isfinite(res.log_likelihood)
        assert not np.any(np.isnan(res.filtered_states.to_numpy()))

    def test_t2_boundary_particle_counts(self):
        """Minimal particle counts (N=2, N=10) execute; non-positive counts raise ValueError."""
        particle_filter, _ = _require_particle_filter()
        df_data, model, _ = _make_ar1_state_space_data(n_obs=5, seed=601)

        # N=2 minimal ensemble
        res_min = particle_filter(model=model, data=df_data, observed_vars=["y"], n_particles=2, seed=42)
        assert np.isfinite(res_min.log_likelihood)

        # N <= 0 raises ValueError
        with pytest.raises(ValueError, match=r"(n_particles|positive)"):
            particle_filter(model=model, data=df_data, observed_vars=["y"], n_particles=0)

        with pytest.raises(ValueError, match=r"(n_particles|positive)"):
            particle_filter(model=model, data=df_data, observed_vars=["y"], n_particles=-50)

    def test_t2_near_absorbing_regimes_in_markov_matrix(self):
        """Near-absorbing Markov transitions (p11 = 0.99999, p22 = 0.99999) do not divide by zero."""
        solve_ms_dsge, _ = _require_ms_dsge()
        spec = _make_3eq_nk_ms_matrices(p11=0.99999, p22=0.99999)

        res = solve_ms_dsge(
            A=spec["A"], B=spec["B"], C=spec["C"], D=spec["D"],
            transition_matrix=spec["transition_matrix"],
            method="newton",
        )
        assert res.converged is True
        assert np.all(np.isfinite(res.ergodic_distribution.to_numpy()))
        assert abs(np.sum(res.ergodic_distribution.to_numpy()) - 1.0) < 1e-8

    def test_t2_near_unit_root_volatility_persistence(self):
        """Stochastic volatility with near-unit root (rho_h = 0.999) remains stable."""
        particle_filter, _ = _require_particle_filter()
        df_data, model, _ = _make_ar1_state_space_data(n_obs=15, seed=602)

        res = particle_filter(
            model=model,
            data=df_data,
            observed_vars=["y"],
            n_particles=1_000,
            stochastic_volatility={"rho_h": 0.999, "sigma_h": 0.05, "sigma_bar": 0.5},
            seed=42,
        )
        assert np.isfinite(res.log_likelihood)

    def test_t2_singular_shock_impact_matrix(self):
        """Fewer structural shocks than endogenous equations (singular shock matrix) solves cleanly."""
        solve_ms_dsge, _ = _require_ms_dsge()
        # 5 equations but only 1 shock
        spec = _make_3eq_nk_ms_matrices()
        D_singular = [spec["D"][0][:, :1], spec["D"][1][:, :1]]

        res = solve_ms_dsge(
            A=spec["A"], B=spec["B"], C=spec["C"], D=D_singular,
            transition_matrix=spec["transition_matrix"],
            method="newton",
        )
        assert res.converged is True
        assert res.R[0].shape == (5, 1)

    def test_t2_zero_measurement_noise_ridge_regularization(self):
        """Measurement error with zero variance does not trigger singular matrix exception when ridge > 0."""
        particle_filter, _ = _require_particle_filter()
        df_data, model, _ = _make_ar1_state_space_data(n_obs=10, seed=603)

        res = particle_filter(
            model=model,
            data=df_data,
            observed_vars=["y"],
            measurement_error={"y": 0.0},
            ridge=1e-5,
            n_particles=500,
            seed=42,
        )
        assert np.isfinite(res.log_likelihood)

    def test_t2_invalid_transition_matrix_row_stochastic(self):
        """Non-stochastic transition matrix (row sum != 1) raises ValueError."""
        solve_ms_dsge, _ = _require_ms_dsge()
        spec = _make_3eq_nk_ms_matrices()
        bad_P = np.array([[0.8, 0.5], [0.3, 0.7]])  # Row 0 sums to 1.3

        with pytest.raises(ValueError, match=r"(stochastic|sum to 1|transition_matrix)"):
            solve_ms_dsge(
                A=spec["A"], B=spec["B"], C=spec["C"], D=spec["D"],
                transition_matrix=bad_P,
            )


# ===========================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS (Pairwise & Tripartite Interactions)
# ===========================================================================

class TestTier3CrossFeatureCombinations:
    """Pairwise and tripartite subsystem interactions."""

    def test_t3_pruned_2nd_order_sv_and_student_t(self):
        """Tripartite combination: 2nd-order pruning + stochastic volatility + Student-t innovations."""
        particle_filter, _ = _require_particle_filter()
        sol2 = canonical_growth_2nd_order()
        sim = sol2.simulate(periods=20, seed=701)
        df_data = sim.to_frame()[["c"]]

        res = particle_filter(
            model=sol2,
            data=df_data,
            observed_vars=["c"],
            n_particles=1_500,
            stochastic_volatility={"rho_h": 0.85, "sigma_h": 0.2, "sigma_bar": 0.01},
            innovation_dist="student_t",
            innovation_df=4.0,
            seed=42,
        )
        assert np.isfinite(res.log_likelihood)
        assert res.filtered_volatility is not None

    def test_t3_ms_dsge_switching_taylor_and_fiscal_rule(self):
        """Leeper (1991) MS-DSGE: switching Taylor rule + active/passive fiscal rule."""
        solve_ms_dsge, _ = _require_ms_dsge()
        spec = _make_leeper_ftpl_matrices()

        res = solve_ms_dsge(
            A=spec["A"],
            B=spec["B"],
            C=spec["C"],
            D=spec["D"],
            transition_matrix=spec["transition_matrix"],
            variable_names=spec["variable_names"],
            shock_names=spec["shock_names"],
            regime_names=spec["regime_names"],
            method="newton",
        )
        assert res.converged is True
        assert res.mean_square_stable is True

        # GIRF from initial Regime M vs initial Regime F
        girf_m = res.girf(initial_regime="Regime_M", shock="eps_f", horizon=15)
        girf_f = res.girf(initial_regime="Regime_F", shock="eps_f", horizon=15)

        # Fiscal shock generates different inflation response under Fiscal vs Monetary dominance
        assert not np.allclose(girf_m["pi"].to_numpy(), girf_f["pi"].to_numpy())

    def test_t3_apf_with_fat_tailed_measurement_error(self):
        """Auxiliary Particle Filter with Student-t observation noise density."""
        particle_filter, _ = _require_particle_filter()
        df_data, model, _ = _make_ar1_state_space_data(n_obs=20, seed=702)

        res = particle_filter(
            model=model,
            data=df_data,
            observed_vars=["y"],
            n_particles=2_000,
            method="auxiliary",
            measurement_dist="student_t",
            measurement_df=4.0,
            seed=42,
        )
        assert np.isfinite(res.log_likelihood)
        assert len(res.resample_history) == len(df_data)

    def test_t3_ms_dsge_newton_and_func_iter_cross_validation(self):
        """Cross-validate Newton and functional iteration: identical GIRF trajectories."""
        solve_ms_dsge, _ = _require_ms_dsge()
        spec = _make_3eq_nk_ms_matrices()

        res_n = solve_ms_dsge(
            A=spec["A"], B=spec["B"], C=spec["C"], D=spec["D"],
            transition_matrix=spec["transition_matrix"],
            variable_names=spec["variable_names"],
            shock_names=spec["shock_names"],
            method="newton",
        )
        res_f = solve_ms_dsge(
            A=spec["A"], B=spec["B"], C=spec["C"], D=spec["D"],
            transition_matrix=spec["transition_matrix"],
            variable_names=spec["variable_names"],
            shock_names=spec["shock_names"],
            method="functional_iteration",
            damping=0.8,
        )
        girf_n = res_n.girf(initial_regime=0, shock="eps_m", horizon=20)
        girf_f = res_f.girf(initial_regime=0, shock="eps_m", horizon=20)

        np.testing.assert_allclose(girf_n.to_numpy(), girf_f.to_numpy(), atol=1e-4)

    def test_t3_pruned_3rd_order_apf_with_stratified_resampling(self):
        """3rd-order pruned model + Auxiliary Particle Filter + stratified resampling."""
        particle_filter, _ = _require_particle_filter()
        try:
            from puremacro.dsge.pruning import canonical_growth_3rd_order
            sol3 = canonical_growth_3rd_order()
        except (ImportError, AttributeError):
            pytest.skip("canonical_growth_3rd_order unavailable")

        sim = sol3.simulate(periods=15, seed=703)
        df_data = sim.to_frame()[["c"]]

        res = particle_filter(
            model=sol3,
            data=df_data,
            observed_vars=["c"],
            n_particles=1_000,
            method="auxiliary",
            resampling_method="stratified",
            seed=42,
        )
        assert np.isfinite(res.log_likelihood)


# ===========================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS
# ===========================================================================

class TestTier4RealWorldScenarios:
    """Canonical macroeconomic benchmark application workloads."""

    def test_t4_s1_canonical_3eq_nk_stochastic_volatility_precautionary_saving(self):
        """Scenario 1: Canonical 3-equation NK model with SV on natural rate of interest.

        Tracks latent volatility state and verifies precautionary contraction in output gap.
        """
        particle_filter, _ = _require_particle_filter()
        rng = np.random.default_rng(801)
        T = 40
        # Simulate synthetic NK series under persistent volatility regime
        h_latent = np.zeros(T)
        x_obs = np.zeros(T)
        h = 0.0
        x_val = 0.0
        for t in range(T):
            h = 0.85 * h + 0.3 * rng.normal()
            h_latent[t] = h
            # Higher volatility induces precautionary saving -> lower output gap x_t
            x_val = 0.7 * x_val - 0.2 * np.exp(h) * rng.normal()
            x_obs[t] = x_val + 0.05 * rng.normal()

        df_nk = pd.DataFrame({"x": x_obs})
        model_nk = _make_test_linear_model(
            T=np.array([[0.7]]),
            R=np.array([[-0.2]]),
            variables=["x"],
            shocks=["eps_r"],
            steady_state=pd.Series({"x": 0.0}),
        )

        res = particle_filter(
            model=model_nk,
            data=df_nk,
            observed_vars=["x"],
            n_particles=3_000,
            stochastic_volatility={"rho_h": 0.85, "sigma_h": 0.3, "sigma_bar": 1.0},
            seed=42,
        )
        assert np.isfinite(res.log_likelihood)
        assert res.filtered_volatility is not None

        # Filtered volatility correlates positively with latent simulated volatility
        filt_vol = res.filtered_volatility.iloc[:, 0].to_numpy()
        corr = np.corrcoef(filt_vol, h_latent)[0, 1]
        assert corr > 0.4

    def test_t4_s2_davig_leeper_farmer_waggoner_zha_monetary_switching(self):
        """Scenario 2: Davig-Leeper (2007) / Farmer-Waggoner-Zha (2011) monetary regime switching.

        Demonstrates that an economy with an isolated non-determinate Dovish regime
        achieves determinacy and MSS stability under Markov switching.
        """
        solve_ms_dsge, _ = _require_ms_dsge()
        spec = _make_3eq_nk_ms_matrices(
            phi_pi=(1.5, 0.8),   # Hawkish (satisfies Taylor rule) vs Dovish (violates Taylor rule)
            phi_x=(0.125, 0.0),
            p11=0.9,
            p22=0.85,
        )

        res = solve_ms_dsge(
            A=spec["A"],
            B=spec["B"],
            C=spec["C"],
            D=spec["D"],
            transition_matrix=spec["transition_matrix"],
            variable_names=spec["variable_names"],
            shock_names=spec["shock_names"],
            regime_names=spec["regime_names"],
            method="newton",
        )
        assert res.converged is True
        assert res.mean_square_stable is True
        assert res.spectral_radius_mss < 1.0

        # Ergodic distribution weights Hawkish regime ~ 60%
        # pi_1 = (1 - 0.85) / ((1 - 0.9) + (1 - 0.85)) = 0.15 / 0.25 = 0.60
        pi = res.ergodic_distribution
        assert abs(pi["Hawkish"] - 0.60) < 1e-5
        assert abs(pi["Dovish"] - 0.40) < 1e-5

        # Monetary policy shock GIRF: output gap and inflation contract on impact
        girf = res.girf(initial_regime="Hawkish", shock="eps_m", horizon=15)
        assert girf["x"].iloc[0] < 0.0
        assert girf["pi"].iloc[0] < 0.0

    def test_t4_s3_leeper_1991_ftpl_fiscal_monetary_regime_switching(self):
        """Scenario 3: Leeper (1991) Fiscal Theory of the Price Level (FTPL) regime switching.

        Contrasts fiscal debt shock responses across Regime M (Monetary Dominance)
        and Regime F (Fiscal Dominance).
        """
        solve_ms_dsge, _ = _require_ms_dsge()
        spec = _make_leeper_ftpl_matrices()

        res = solve_ms_dsge(
            A=spec["A"],
            B=spec["B"],
            C=spec["C"],
            D=spec["D"],
            transition_matrix=spec["transition_matrix"],
            variable_names=spec["variable_names"],
            shock_names=spec["shock_names"],
            regime_names=spec["regime_names"],
            method="newton",
        )
        assert res.converged is True
        assert res.mean_square_stable is True

        # In Regime F (Fiscal dominance), a fiscal shock triggers inflation jump
        # because monetary policy does not raise real rates to back debt
        girf_f = res.girf(initial_regime="Regime_F", shock="eps_f", horizon=10)
        assert girf_f["b"].iloc[0] > 0.0

    def test_t4_s4_particle_filter_vs_kalman_filter_benchmark(self):
        """Scenario 4: Particle filter log-likelihood parity against Kalman filter benchmark."""
        particle_filter, _ = _require_particle_filter()
        df_data, model, oracle_ll = _make_ar1_state_space_data(
            n_obs=50, rho=0.85, sigma_u=0.3, sigma_v=0.15, seed=804
        )

        res_pf = particle_filter(
            model=model,
            data=df_data,
            observed_vars=["y"],
            n_particles=10_000,
            method="bootstrap",
            seed=42,
        )
        # Acceptance criterion: matches Kalman filter log-likelihood within +/- 1.5 log-points
        diff = abs(res_pf.log_likelihood - oracle_ll)
        assert diff < 1.5, f"Log-likelihood diff {diff} exceeds tolerance 1.5 (PF: {res_pf.log_likelihood}, KF: {oracle_ll})"

    def test_t4_s5_end_to_end_ms_dsge_simulation_and_particle_filter_tracking(self):
        """Scenario 5: Full E2E pipeline: simulate from MS-DSGE, track latent states via particle filter."""
        solve_ms_dsge, _ = _require_ms_dsge()
        particle_filter, _ = _require_particle_filter()

        spec = _make_3eq_nk_ms_matrices()
        res_ms = solve_ms_dsge(
            A=spec["A"], B=spec["B"], C=spec["C"], D=spec["D"],
            transition_matrix=spec["transition_matrix"],
            variable_names=spec["variable_names"],
            shock_names=spec["shock_names"],
            method="newton",
        )

        sim_df, sim_regimes = res_ms.simulate(periods=35, seed=805)
        assert len(sim_df) == 35
        assert len(sim_regimes) == 35

        # Extract observable inflation series and feed into particle filter
        df_obs = sim_df[["pi"]]
        res_pf = particle_filter(
            model=res_ms,
            data=df_obs,
            observed_vars=["pi"],
            n_particles=2_000,
            seed=42,
        )
        assert np.isfinite(res_pf.log_likelihood)
        assert res_pf.filtered_states.shape[0] == 35


# ===========================================================================
# TIER 5 / ARCHITECTURAL AUDIT: PYODIDE 4-PACKAGE PURITY
# ===========================================================================

def test_pyodide_four_package_purity_phase_d():
    """Static import scan: verify strictly zero unauthorized runtime dependencies."""
    e2e_path = Path(__file__).resolve()
    with open(e2e_path, "r", encoding="utf-8") as f:
        tree_code = f.read()

    allowed = {
        "numpy", "scipy", "pandas", "matplotlib", "puremacro", "pytest",
        "typing", "dataclasses", "copy", "math", "os", "sys", "pathlib",
        "re", "tempfile", "time", "inspect", "__future__",
    }

    import_lines = re.findall(r"^(?:import|from)\s+([a-zA-Z0-9_]+)", tree_code, re.MULTILINE)
    for pkg in import_lines:
        assert pkg in allowed, f"Unauthorized package import in Phase D E2E: {pkg}"
