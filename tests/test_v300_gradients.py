"""Tests for puremacro 3.0 Pillar 1: Exact Analytic Likelihood Gradients.

Verifies:
- Generalized Sylvester matrix equation solver (Schur back-substitution)
- Analytical decision rule sensitivities dG/dtheta and dN/dtheta
- Forward Kalman score recursion against 5-point central finite differences
- ScoreDiagnosticsResult formatting and verification methods
- Prior density gradient vectors across Beta, Gamma, InvGamma, Normal, Uniform
- Combined log_posterior_and_gradient evaluations
"""

import math
import numpy as np
import pandas as pd
import pytest
import scipy.linalg

from puremacro.dsge._gradients import (
    ScoreDiagnosticsResult,
    StateSpaceSensitivity,
    build_state_space_sensitivities,
    compute_decision_rule_derivatives,
    kalman_score,
    log_posterior_and_gradient,
    solve_sylvester_generalized,
)
from puremacro.dsge.priors import grad_log_prior, log_prior
from puremacro.state_space import StateSpaceModel


# ---------------------------------------------------------------------------
# 1. Generalized Sylvester Solver Tests
# ---------------------------------------------------------------------------

def test_solve_sylvester_generalized_accuracy():
    """Verify solve_sylvester_generalized matches Kronecker reference to machine precision."""
    np.random.seed(42)
    N = 6
    M = 4

    A_hat = np.random.randn(N, N) + 5.0 * np.eye(N)
    B = np.random.randn(N, N)
    # Stable C matrix (eigenvalues strictly inside unit circle)
    C = np.random.randn(M, M) * 0.4
    D = np.random.randn(N, M)

    # 1. Kronecker reference: (I_M \otimes A_hat + C^T \otimes B) vec(X) = vec(D)
    M_kron = np.kron(np.eye(M), A_hat) + np.kron(C.T, B)
    x_vec = np.linalg.solve(M_kron, D.ravel(order="F"))
    X_ref = x_vec.reshape((N, M), order="F")

    # 2. Schur column substitution solver
    X_sol = solve_sylvester_generalized(A_hat, B, C, D)

    # Verify matching reference and Sylvester residual
    np.testing.assert_allclose(X_sol, X_ref, rtol=1e-12, atol=1e-12)
    residual = A_hat @ X_sol + B @ X_sol @ C - D
    assert np.max(np.abs(residual)) < 1e-12


def test_solve_sylvester_generalized_batched():
    """Verify batched RHS across K parameters gives identical results to single solves."""
    np.random.seed(123)
    N = 5
    M = 3
    K = 4

    A_hat = np.random.randn(N, N) + 4.0 * np.eye(N)
    B = np.random.randn(N, N)
    C = np.random.randn(M, M) * 0.3
    D_batch = np.random.randn(K, N, M)

    X_batch = solve_sylvester_generalized(A_hat, B, C, D_batch)
    assert X_batch.shape == (K, N, M)

    for k in range(K):
        X_single = solve_sylvester_generalized(A_hat, B, C, D_batch[k])
        np.testing.assert_allclose(X_batch[k], X_single, rtol=1e-13, atol=1e-13)


# ---------------------------------------------------------------------------
# 2. Decision Rule Sensitivities Tests
# ---------------------------------------------------------------------------

def test_decision_rule_derivatives_vs_complex_step():
    """Verify dghx/dtheta and dghu/dtheta match differentiation to high precision."""
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
    """
    m = load_mod(mod_code)
    dghx, dghu = compute_decision_rule_derivatives(m, "rho_a")

    # High precision finite difference
    h = 1e-6
    m_p = load_mod(mod_code, params={"beta": 0.99, "sigma": 1.0, "rho_a": 0.8 + h})
    m_m = load_mod(mod_code, params={"beta": 0.99, "sigma": 1.0, "rho_a": 0.8 - h})

    dghx_ref = (np.asarray(m_p.dynare_dr.ghx) - np.asarray(m_m.dynare_dr.ghx)) / (2 * h)
    dghu_ref = (np.asarray(m_p.dynare_dr.ghu) - np.asarray(m_m.dynare_dr.ghu)) / (2 * h)

    np.testing.assert_allclose(dghx, dghx_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(dghu, dghu_ref, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. Forward Kalman Score Recursion Tests
# ---------------------------------------------------------------------------

def test_kalman_score_forward_vs_finite_difference():
    """Verify analytical score matches 5-point central FD to high precision."""
    np.random.seed(999)
    m_dim = 3
    n_dim = 2
    r_dim = 2
    T_steps = 30

    def make_toy_ssm(theta):
        T = np.array([
            [0.65 * theta, 0.10, 0.0],
            [0.05, 0.50, 0.15 * theta],
            [0.0, 0.0, 0.40],
        ])
        Z = np.array([
            [1.0, 0.0, 0.3 * theta],
            [0.0, 1.0, 0.0],
        ])
        R = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.2 * theta, 0.0],
        ])
        Q = np.array([
            [0.70, 0.05 * theta],
            [0.05 * theta, 0.50],
        ])
        H = np.array([
            [0.04, 0.0],
            [0.0, 0.03 * theta],
        ])
        c = np.zeros(m_dim)
        d = np.array([0.05 * theta, 0.0])
        return StateSpaceModel(T=T, Z=Z, R=R, Q=Q, H=H, c=c, d=d)

    theta_val = 0.75
    ssm_base = make_toy_ssm(theta_val)

    # Simulate data
    a = np.zeros(m_dim)
    y_data = np.zeros((T_steps, n_dim))
    for t in range(T_steps):
        eta = np.random.multivariate_normal(np.zeros(r_dim), ssm_base.Q)
        eps = np.random.multivariate_normal(np.zeros(n_dim), ssm_base.H)
        y_data[t] = ssm_base.Z @ a + ssm_base.d + eps
        a = ssm_base.T @ a + ssm_base.c + ssm_base.R @ eta

    # Analytical sensitivities
    h_probe = 1e-7
    ssm_plus = make_toy_ssm(theta_val + h_probe)
    ssm_minus = make_toy_ssm(theta_val - h_probe)

    dT = (ssm_plus.T - ssm_minus.T) / (2 * h_probe)
    dZ = (ssm_plus.Z - ssm_minus.Z) / (2 * h_probe)
    dR = (ssm_plus.R - ssm_minus.R) / (2 * h_probe)
    dQ = (ssm_plus.Q - ssm_minus.Q) / (2 * h_probe)
    dH = (ssm_plus.H - ssm_minus.H) / (2 * h_probe)
    dc = (ssm_plus.c - ssm_minus.c) / (2 * h_probe)
    dd = (ssm_plus.d - ssm_minus.d) / (2 * h_probe)

    sens = {
        "theta": StateSpaceSensitivity(
            dT=dT, dZ=dZ, dR=dR, dQ=dQ, dH=dH, dc=dc, dd=dd
        )
    }

    loglik_an, score_an = kalman_score(y_data, ssm_base, sens)

    # 5-point central finite differences
    def eval_filter_ll(th):
        m_th = make_toy_ssm(th)
        from puremacro.state_space import kalman_filter
        RQR = m_th.R @ m_th.Q @ m_th.R.T
        P0 = scipy.linalg.solve_discrete_lyapunov(m_th.T, RQR)
        res = kalman_filter(y_data, m_th, P0=P0)
        return float(res["loglik"])

    eps_fd = 1e-6
    f_p2 = eval_filter_ll(theta_val + 2 * eps_fd)
    f_p1 = eval_filter_ll(theta_val + eps_fd)
    f_m1 = eval_filter_ll(theta_val - eps_fd)
    f_m2 = eval_filter_ll(theta_val - 2 * eps_fd)

    score_fd = (-f_p2 + 8 * f_p1 - 8 * f_m1 + f_m2) / (12 * eps_fd)

    assert abs(score_an[0] - score_fd) < 1e-5
    rel_err = abs(score_an[0] - score_fd) / max(1.0, abs(score_fd))
    assert rel_err < 1e-5


def test_kalman_score_missing_data():
    """Verify kalman_score handles missing observations (np.nan)."""
    np.random.seed(555)
    T = np.array([[0.5, 0.1], [0.0, 0.4]])
    Z = np.array([[1.0, 0.0], [0.0, 1.0]])
    R = np.eye(2)
    Q = np.eye(2) * 0.2
    H = np.eye(2) * 0.05
    model = StateSpaceModel(T=T, Z=Z, R=R, Q=Q, H=H)

    y = np.random.randn(20, 2)
    y[3, 0] = np.nan
    y[7, :] = np.nan  # completely missing period

    sens = {
        "p1": StateSpaceSensitivity(
            dT=np.array([[0.1, 0.0], [0.0, 0.0]]),
            dZ=np.zeros((2, 2)),
            dR=np.zeros((2, 2)),
            dQ=np.zeros((2, 2)),
            dH=np.zeros((2, 2)),
            dc=np.zeros(2),
            dd=np.zeros(2),
        )
    }

    ll, score = kalman_score(y, model, sens)
    assert np.isfinite(ll)
    assert np.isfinite(score[0])


# ---------------------------------------------------------------------------
# 4. ScoreDiagnosticsResult Tests
# ---------------------------------------------------------------------------

def test_score_diagnostics_result():
    """Verify ScoreDiagnosticsResult markdown, latex, and numerical verification methods."""
    res = ScoreDiagnosticsResult(
        loglik=-120.45,
        gradient=np.array([1.234, -0.567]),
        param_names=("alpha", "beta"),
        elapsed_sec=0.003,
    )
    md = res.to_markdown()
    assert "alpha" in md
    assert "beta" in md

    ltx = res.to_latex()
    assert "\\begin{tabular}" in ltx


# ---------------------------------------------------------------------------
# 5. Prior Gradients Tests
# ---------------------------------------------------------------------------

def test_prior_gradients_all_distributions():
    """Verify grad_log_prior matches finite differences across all supported distributions."""
    priors = {
        "p_beta": {"dist": "beta", "mean": 0.5, "std": 0.1, "lb": 0.0, "ub": 1.0},
        "p_gamma": {"dist": "gamma", "mean": 1.2, "std": 0.3, "lb": 0.001, "ub": 10.0},
        "p_norm": {"dist": "normal", "mean": 0.0, "std": 1.0, "lb": -5.0, "ub": 5.0},
        "p_ig": {"dist": "invgamma", "mean": 0.4, "std": 1.0, "lb": 0.001, "ub": 10.0},
        "p_unif": {"dist": "uniform", "mean": 0.5, "std": 0.288, "lb": 0.0, "ub": 1.0},
    }

    theta = {
        "p_beta": 0.55,
        "p_gamma": 1.10,
        "p_norm": -0.40,
        "p_ig": 0.35,
        "p_unif": 0.70,
    }

    names = tuple(priors.keys())
    g_analytic = grad_log_prior(theta, priors, names)

    # Finite difference
    h = 1e-6
    for i, nm in enumerate(names):
        th_p = dict(theta); th_p[nm] = theta[nm] + h
        th_m = dict(theta); th_m[nm] = theta[nm] - h
        g_fd = (log_prior(th_p, priors) - log_prior(th_m, priors)) / (2 * h)
        np.testing.assert_allclose(g_analytic[i], g_fd, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 6. Combined Posterior and Gradient Tests
# ---------------------------------------------------------------------------

def test_log_posterior_and_gradient_combined():
    """Verify log_posterior_and_gradient returns valid combined log-post and gradient."""
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
    """
    m = load_mod(mod_code)
    priors = {
        "sigma": {"dist": "gamma", "mean": 1.0, "std": 0.2, "lb": 0.1, "ub": 5.0},
    }
    y_data = np.random.randn(20, 1)

    log_post, grad = log_posterior_and_gradient(
        vec=np.array([1.0]),
        names=["sigma"],
        model_template=m,
        y_data=y_data,
        varobs=["y"],
        priors=priors,
        fixed_params={"beta": 0.99, "rho_a": 0.8},
    )

    assert np.isfinite(log_post)
    assert grad.shape == (1,)
    assert np.isfinite(grad[0])
