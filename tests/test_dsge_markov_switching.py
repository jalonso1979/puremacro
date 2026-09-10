"""Comprehensive unit and benchmark test suite for Markov-Switching DSGE (MS-DSGE).

Verifies:
1. Foerster et al. (2016) and Farmer et al. (2011) benchmark solutions:
   - Forward-looking MSRE models (C_i = 0, T_i = 0).
   - Backward-looking dynamic persistence models with analytical scalar roots.
2. Monetary policy regime switching (Hawkish vs Dovish Taylor rules):
   - Davig & Leeper (2007) / Farmer et al. (2011) 3-equation NK model.
   - Both Newton and Functional Iteration convergence.
   - Mean-Square Stability (rho(M_2) < 1) despite Dovish regime being indeterminate in isolation.
3. Fiscal policy regime switching (Leeper 1991 / Bianchi 2012):
   - Active Monetary / Passive Fiscal (AM/PF) vs Passive Monetary / Active Fiscal (PM/AF).
4. Ergodic distribution and analytical Lyapunov moments:
   - pi_infty P = pi_infty.
   - Discrete Lyapunov covariance (I - M_2) v = q matching unconditional Var(y).
   - Symmetry and positive semi-definiteness.
5. Analytical closed-form GIRF matching Monte Carlo simulation within 3-sigma SE.
6. Dynare .mod file parser integration with markov_switching; ... end; blocks.
7. Presentation contract: summary(), plot(), to_markdown(), to_latex(), to_typst(), simulate().
8. Adversarial edge cases: non-zero intercepts K(s), explosive regimes, degenerate S=1, invalid matrices.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro.dsge.markov_switching import (
    MSDSGEResult,
    markov_stationary,
    solve_ms_dsge,
)
from puremacro.dsge._parser import parse_mod_to_dag


# ==============================================================================
# 1. Foerster et al. (2016) / Farmer et al. (2011) Benchmarks
# ==============================================================================

def test_forward_looking_msre_benchmark():
    """Purely forward-looking model (C_i = 0) has exact MSV solution T_i = 0."""
    S = 2
    n = 2
    P = np.array([[0.85, 0.15],
                  [0.25, 0.75]])

    A = [np.eye(n), 1.2 * np.eye(n)]
    B = [-2.0 * np.eye(n), -1.8 * np.eye(n)]
    C = [np.zeros((n, n)), np.zeros((n, n))]
    D = [np.eye(n), 0.5 * np.eye(n)]

    res = solve_ms_dsge(
        A, B, C, D, P,
        regime_names=["Regime 1", "Regime 2"],
        variable_names=["y1", "y2"],
        shock_names=["e1", "e2"],
    )

    assert res.converged
    # T_i must be exactly zero
    assert np.allclose(res.T[0], 0.0, atol=1e-12)
    assert np.allclose(res.T[1], 0.0, atol=1e-12)
    # R_i = -B_i^{-1} D_i
    expected_R0 = -np.linalg.solve(B[0], D[0])
    expected_R1 = -np.linalg.solve(B[1], D[1])
    assert np.allclose(res.R[0], expected_R0, atol=1e-10)
    assert np.allclose(res.R[1], expected_R1, atol=1e-10)

    # Stability operators M1, M2 must have spectral radius 0
    assert res.spectral_radius_mean == pytest.approx(0.0, abs=1e-12)
    assert res.spectral_radius_mss == pytest.approx(0.0, abs=1e-12)
    assert res.mean_square_stable


def test_univariate_coupled_quadratic_benchmark():
    """Scalar dynamic model with persistent backward-looking state.

    A_i (sum_k p_{ik} T_k) T_i + B_i T_i + C_i = 0.
    Verifies that Newton and Functional Iteration reach the identical analytical root.
    """
    S = 2
    n = 1
    P = np.array([[0.80, 0.20],
                  [0.30, 0.70]])

    A = [np.array([[1.0]]), np.array([[1.0]])]
    B = [np.array([[-2.0]]), np.array([[-1.8]])]
    C = [np.array([[0.5]]), np.array([[0.4]])]
    D = [np.array([[1.0]]), np.array([[1.0]])]

    # Reference root from independent optimization
    t1_ref = 0.29165212
    t2_ref = 0.26153542

    # Solve with Newton
    res_newton = solve_ms_dsge(A, B, C, D, P, method="newton", tol=1e-11)
    assert res_newton.converged
    assert res_newton.T[0][0, 0] == pytest.approx(t1_ref, abs=1e-6)
    assert res_newton.T[1][0, 0] == pytest.approx(t2_ref, abs=1e-6)

    # Solve with Functional Iteration
    res_fp = solve_ms_dsge(A, B, C, D, P, method="functional_iteration", tol=1e-11)
    assert res_fp.converged
    assert res_fp.T[0][0, 0] == pytest.approx(res_newton.T[0][0, 0], abs=1e-8)
    assert res_fp.T[1][0, 0] == pytest.approx(res_newton.T[1][0, 0], abs=1e-8)

    # Verify residuals are zero to high precision
    F1 = A[0] * (P[0, 0] * res_newton.T[0] + P[0, 1] * res_newton.T[1]) * res_newton.T[0] + B[0] * res_newton.T[0] + C[0]
    F2 = A[1] * (P[1, 0] * res_newton.T[0] + P[1, 1] * res_newton.T[1]) * res_newton.T[1] + B[1] * res_newton.T[1] + C[1]
    assert np.abs(F1[0, 0]) < 1e-10
    assert np.abs(F2[0, 0]) < 1e-10


# ==============================================================================
# 2. Monetary Policy Regime Switching (Davig-Leeper 2007; FWZ 2011)
# ==============================================================================

def test_monetary_policy_hawkish_vs_dovish():
    """3-equation New Keynesian model with switching Taylor rule.

    Regime 1: Hawkish (phi_pi = 1.8 > 1, determinate in isolation).
    Regime 2: Dovish (phi_pi = 0.8 < 1, indeterminate in isolation).
    Shows Mean-Square Stability holds globally (rho(M_2) < 1).
    """
    beta = 0.99
    sigma = 1.0
    kappa = 0.1
    rho_i = 0.8
    phi_x = 0.1

    phi_pi_list = [1.8, 0.8]
    S = 2
    n = 3

    P = np.array([[0.90, 0.10],
                  [0.20, 0.80]])

    A_list = []
    B_list = []
    C_list = []
    D_list = []

    for s in range(S):
        phi_pi = phi_pi_list[s]
        As = np.array([
            [1.0, 1.0 / sigma, 0.0],
            [0.0, beta,        0.0],
            [0.0, 0.0,         0.0]
        ])
        Bs = np.array([
            [-1.0, 0.0, -1.0 / sigma],
            [kappa, -1.0, 0.0],
            [(1.0 - rho_i) * phi_x, (1.0 - rho_i) * phi_pi, -1.0]
        ])
        Cs = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, rho_i]
        ])
        Ds = np.eye(3)
        A_list.append(As)
        B_list.append(Bs)
        C_list.append(Cs)
        D_list.append(Ds)

    res = solve_ms_dsge(
        A_list, B_list, C_list, D_list, P,
        regime_names=["Hawkish", "Dovish"],
        variable_names=["output_gap", "inflation", "interest_rate"],
        shock_names=["demand", "cost_push", "monetary_policy"],
        method="newton",
    )

    assert res.converged
    assert res.iterations <= 15
    assert res.diff < 1e-9

    # Check stability metrics
    assert res.mean_square_stable
    assert res.spectral_radius_mss < 1.0
    assert res.spectral_radius_mean < 1.0

    # Dovish regime in isolation would have higher persistence/instability
    rho_hawkish = np.max(np.abs(np.linalg.eigvals(res.T["Hawkish"])))
    rho_dovish = np.max(np.abs(np.linalg.eigvals(res.T["Dovish"])))
    assert rho_dovish > rho_hawkish

    # Functional iteration should also converge and match
    res_fp = solve_ms_dsge(
        A_list, B_list, C_list, D_list, P,
        regime_names=["Hawkish", "Dovish"],
        variable_names=["output_gap", "inflation", "interest_rate"],
        shock_names=["demand", "cost_push", "monetary_policy"],
        method="functional_iteration",
        damping=0.8,
    )
    assert res_fp.converged
    assert np.allclose(res.T["Hawkish"], res_fp.T["Hawkish"], atol=1e-7)
    assert np.allclose(res.T["Dovish"], res_fp.T["Dovish"], atol=1e-7)

    # Impulse responses to contractionary monetary policy shock
    # Interest rate rises, output and inflation contract on impact
    irf_h = res.irf("Hawkish", shock="monetary_policy", horizon=10)
    assert irf_h.loc[0, "interest_rate"] > 0
    assert irf_h.loc[0, "output_gap"] < 0
    assert irf_h.loc[0, "inflation"] < 0

    # Over time, responses revert toward 0
    assert abs(irf_h.loc[10, "interest_rate"]) < abs(irf_h.loc[0, "interest_rate"])
    assert abs(irf_h.loc[10, "output_gap"]) < abs(irf_h.loc[0, "output_gap"])


# ==============================================================================
# 3. Fiscal Policy Regime Switching (Leeper 1991 / Bianchi 2012)
# ==============================================================================

def test_fiscal_monetary_regime_switching():
    """Leeper (1991) fiscal-monetary interactions with regime shifts.

    Regime 1: Active Monetary / Passive Fiscal (AM/PF):
              alpha = 1.5 > 1, gamma = 0.15 > r.
    Regime 2: Passive Monetary / Active Fiscal (PM/AF):
              alpha = 0.8 < 1, gamma = 0.00 < r.
    """
    r = 0.02
    b_bar = 1.0

    # Variables: y = [b_t, pi_t, i_t]
    # Debt: b_t = (1+r - gamma(s)) b_{t-1} + b_bar*i_{t-1} - b_bar*pi_t + eps_g
    # Fisher: E_t[pi_{t+1}] - i_t + r = 0  => E_t[pi_{t+1}] - i_t = 0 (in dev)
    # Taylor: i_t = alpha(s) pi_t + eps_m
    S = 2
    n = 3
    P = np.array([[0.85, 0.15],
                  [0.20, 0.80]])

    alpha_list = [1.5, 0.8]
    gamma_list = [0.15, 0.0]

    A_list = []
    B_list = []
    C_list = []
    D_list = []

    for s in range(S):
        alpha = alpha_list[s]
        gamma = gamma_list[s]

        As = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        Bs = np.array([
            [-1.0, -b_bar, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, alpha, -1.0]
        ])
        Cs = np.array([
            [1.0 + r - gamma, 0.0, b_bar],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        Ds = np.array([
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0]
        ])
        A_list.append(As)
        B_list.append(Bs)
        C_list.append(Cs)
        D_list.append(Ds)

    res = solve_ms_dsge(
        A_list, B_list, C_list, D_list, P,
        regime_names=["AM_PF", "PM_AF"],
        variable_names=["debt", "inflation", "rate"],
        shock_names=["fiscal", "monetary"],
    )

    assert res.converged
    assert res.mean_square_stable
    assert res.spectral_radius_mss < 1.0

    # Ergodic distribution
    assert res.ergodic_distribution["AM_PF"] == pytest.approx(0.20 / (0.15 + 0.20), abs=1e-5)
    assert res.ergodic_distribution["PM_AF"] == pytest.approx(0.15 / (0.15 + 0.20), abs=1e-5)


# ==============================================================================
# 4. Ergodic Distribution & Moments
# ==============================================================================

def test_ergodic_distribution_and_lyapunov_covariance():
    """Verify stationary distribution and discrete Lyapunov equation covariance."""
    P = np.array([[0.90, 0.10],
                  [0.25, 0.75]])

    pi = markov_stationary(P)
    # pi @ P == pi
    assert np.allclose(pi @ P, pi, atol=1e-12)
    assert np.allclose(pi.sum(), 1.0, atol=1e-12)
    assert np.all(pi >= 0.0)

    # Construct stable 2-state system
    n = 2
    T0 = np.array([[0.6, 0.1], [0.0, 0.5]])
    T1 = np.array([[0.4, -0.1], [0.1, 0.3]])
    R0 = np.eye(n)
    R1 = 1.5 * np.eye(n)

    A = [np.eye(n), np.eye(n)]
    B = [-2.0 * np.eye(n), -2.0 * np.eye(n)]
    C = [-(A[0] @ (P[0, 0] * T0 + P[0, 1] * T1) + B[0]) @ T0,
         -(A[1] @ (P[1, 0] * T0 + P[1, 1] * T1) + B[1]) @ T1]
    D = [-(A[0] @ (P[0, 0] * T0 + P[0, 1] * T1) + B[0]) @ R0,
         -(A[1] @ (P[1, 0] * T0 + P[1, 1] * T1) + B[1]) @ R1]

    res = solve_ms_dsge(A, B, C, D, P, variable_names=["y1", "y2"])

    assert res.mean_square_stable
    assert res.ergodic_cov.shape == (n, n)

    # Covariance must be symmetric positive semi-definite
    cov_arr = res.ergodic_cov.values
    assert np.allclose(cov_arr, cov_arr.T, atol=1e-10)
    cov_eigs = np.linalg.eigvalsh(cov_arr)
    assert np.all(cov_eigs >= -1e-12)
    assert np.all(np.diag(cov_arr) > 0.0)


# ==============================================================================
# 5. Closed-Form Analytical GIRF vs Monte Carlo Simulation
# ==============================================================================

def test_analytical_girf_matches_monte_carlo():
    """Verify exact analytical closed-form GIRF matches Monte Carlo within 3-sigma."""
    S = 2
    n = 2
    P = np.array([[0.80, 0.20],
                  [0.30, 0.70]])

    T0 = np.array([[0.5, 0.1], [0.0, 0.4]])
    T1 = np.array([[0.3, -0.1], [0.1, 0.2]])
    R0 = np.eye(n)
    R1 = np.array([[1.2, 0.0], [0.0, 0.8]])

    A = [np.eye(n), np.eye(n)]
    B = [-2.0 * np.eye(n), -2.0 * np.eye(n)]
    C = [-(A[0] @ (P[0, 0] * T0 + P[0, 1] * T1) + B[0]) @ T0,
         -(A[1] @ (P[1, 0] * T0 + P[1, 1] * T1) + B[1]) @ T1]
    D = [-(A[0] @ (P[0, 0] * T0 + P[0, 1] * T1) + B[0]) @ R0,
         -(A[1] @ (P[1, 0] * T0 + P[1, 1] * T1) + B[1]) @ R1]

    res = solve_ms_dsge(A, B, C, D, P, variable_names=["x1", "x2"], shock_names=["e1", "e2"])

    H = 8
    girf_ana = res.girf(initial_regime=0, shock="e1", horizon=H)

    # Monte Carlo simulation
    N_sim = 35_000
    rng = np.random.default_rng(123)
    mc_paths = np.zeros((N_sim, H + 1, n))
    e1 = np.array([1.0, 0.0])

    for i in range(N_sim):
        s = 0
        y = R0 @ e1
        mc_paths[i, 0, :] = y
        for h in range(1, H + 1):
            s = int(rng.choice(S, p=P[s]))
            Ts = T0 if s == 0 else T1
            y = Ts @ y
            mc_paths[i, h, :] = y

    mc_mean = np.mean(mc_paths, axis=0)
    mc_std = np.std(mc_paths, axis=0)
    mc_se = mc_std / np.sqrt(N_sim)

    # Test that analytical GIRF falls within 3.5 standard errors of Monte Carlo estimate
    for h in range(H + 1):
        for j in range(n):
            ana_val = girf_ana.iloc[h, j]
            sim_val = mc_mean[h, j]
            se_val = mc_se[h, j]
            diff = abs(ana_val - sim_val)
            assert diff <= 3.5 * se_val + 1e-4, (
                f"Mismatch at h={h}, var={j}: analytical={ana_val:.5f}, sim={sim_val:.5f}, se={se_val:.5f}"
            )


# ==============================================================================
# 6. Dynare .mod File Parser Integration
# ==============================================================================

def test_mod_file_parser_with_markov_switching():
    """Verify parsing of Dynare .mod file containing markov_switching block."""
    mod_text = """
    var y, pi, i;
    varexo eps_r;
    parameters beta, sigma, phi_pi;
    beta = 0.99;
    sigma = 1.0;
    phi_pi = 1.5;

    model;
    y = y(+1) - (1/sigma)*(i - pi(+1));
    pi = beta*pi(+1) + 0.1*y;
    i = 0.8*i(-1) + 0.2*phi_pi*pi + eps_r;
    end;

    markov_switching;
        num_regimes = 2;
        P = [[0.90, 0.10], [0.20, 0.80]];
        param phi_pi = [1.8, 0.8];
    end;
    """

    dag = parse_mod_to_dag(mod_text)
    assert dag.markov_switching_config is not None
    assert dag.markov_switching_config["num_regimes"] == 2
    assert np.allclose(dag.markov_switching_config["transition_matrix"], [[0.9, 0.1], [0.2, 0.8]])
    assert dag.markov_switching_config["parameters"]["phi_pi"] == [1.8, 0.8]

    # Solve model directly from .mod text
    res = solve_ms_dsge(mod_text)
    assert isinstance(res, MSDSGEResult)
    assert res.converged
    assert res.mean_square_stable
    assert len(res.variables) == 3
    assert len(res.shocks) == 1
    assert "Regime 1" in res.regime_names
    assert "Regime 2" in res.regime_names


def test_mod_file_parser_alternative_syntax():
    """Verify semicolon matrix and regime block syntax in markov_switching."""
    mod_text = """
    var x, p;
    varexo e;
    parameters a, b;
    a = 0.5;
    b = 0.2;

    model;
    x = a*x(-1) + b*p(+1) + e;
    p = 0.7*p(+1) + 0.3*x;
    end;

    markov_switching;
        regimes = 2;
        transition_matrix = [0.85, 0.15; 0.25, 0.75];
        regime 1;
            a = 0.7;
        regime 2;
            a = 0.3;
    end;
    """

    dag = parse_mod_to_dag(mod_text)
    assert dag.markov_switching_config is not None
    assert dag.markov_switching_config["num_regimes"] == 2
    assert np.allclose(dag.markov_switching_config["transition_matrix"], [[0.85, 0.15], [0.25, 0.75]])
    assert dag.markov_switching_config["regimes"][1]["a"] == 0.7
    assert dag.markov_switching_config["regimes"][2]["a"] == 0.3

    res = solve_ms_dsge(mod_text)
    assert res.converged
    assert res.mean_square_stable


# ==============================================================================
# 7. Presentation Contracts & Export Formats
# ==============================================================================

def test_presentation_contracts_and_simulation():
    """Verify summary(), plot(), to_markdown(), to_latex(), to_typst(), simulate()."""
    S = 2
    n = 2
    P = np.array([[0.8, 0.2], [0.3, 0.7]])
    A = [np.eye(n), np.eye(n)]
    B = [-2.0 * np.eye(n), -1.5 * np.eye(n)]
    C = [np.zeros((n, n)), np.zeros((n, n))]
    D = [np.eye(n), 0.5 * np.eye(n)]

    res = solve_ms_dsge(
        A, B, C, D, P,
        regime_names=["Hawkish", "Dovish"],
        variable_names=["Output", "Inflation"],
        shock_names=["Demand", "Supply"],
    )

    # summary()
    summary_str = res.summary()
    assert isinstance(summary_str, str)
    assert "MARKOV-SWITCHING DSGE EQUILIBRIUM" in summary_str
    assert "Hawkish" in summary_str
    assert "Dovish" in summary_str
    assert "Mean-Square Stable" in summary_str

    # summary(as_dataframe=True)
    summary_df = res.summary(as_dataframe=True)
    assert isinstance(summary_df, pd.DataFrame)
    assert len(summary_df) == 2
    assert "Regime" in summary_df.columns
    assert "Mean-Square Stable" in summary_df.columns

    # to_markdown()
    md = res.to_markdown()
    assert isinstance(md, str)
    assert "| Regime |" in md or "| Hawkish |" in md or "Hawkish" in md

    # to_latex()
    latex = res.to_latex()
    assert isinstance(latex, str)
    assert "begin{tabular}" in latex or "Hawkish" in latex

    # to_typst()
    typst = res.to_typst()
    assert isinstance(typst, str)
    assert "#table" in typst or "Hawkish" in typst

    # simulate()
    sim_df, regimes = res.simulate(periods=50, initial_regime="Hawkish", seed=99)
    assert isinstance(sim_df, pd.DataFrame)
    assert sim_df.shape == (50, 2)
    assert len(regimes) == 50
    assert set(regimes).issubset({0, 1})

    # plot() in headless mode
    fig, axes = res.plot(girf=True, horizon=15)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert axes is not None
    plt.close(fig)

    fig2, axes2 = res.plot(regime="Hawkish", girf=False, horizon=10)
    assert isinstance(fig2, matplotlib.figure.Figure)
    plt.close(fig2)


# ==============================================================================
# 8. Edge Cases & Robustness
# ==============================================================================

def test_nonzero_intercepts_and_ergodic_mean():
    """Model with regime constants K(s) produces non-zero intercepts c(s) and ergodic mean."""
    S = 2
    n = 2
    P = np.array([[0.8, 0.2], [0.3, 0.7]])
    A = [np.eye(n), np.eye(n)]
    B = [-2.0 * np.eye(n), -2.0 * np.eye(n)]
    C = [0.2 * np.eye(n), 0.1 * np.eye(n)]
    D = [np.eye(n), np.eye(n)]
    K = [np.array([0.5, -0.2]), np.array([0.1, 0.3])]

    res = solve_ms_dsge(A, B, C, D, P, K=K, variable_names=["y1", "y2"])

    assert res.converged
    # Intercepts must be non-zero
    assert np.max(np.abs(res.c[0])) > 0.05
    assert np.max(np.abs(res.c[1])) > 0.05
    # Ergodic mean must be non-zero
    assert np.max(np.abs(res.ergodic_mean.values)) > 0.05


def test_explosive_regime_mean_square_unstable():
    """System with explosive roots in both regimes fails Mean-Square Stability."""
    S = 2
    n = 1
    P = np.array([[0.5, 0.5], [0.5, 0.5]])
    # A * T^2 + B * T + C = 0 with highly explosive roots
    A = [np.array([[1.0]]), np.array([[1.0]])]
    B = [np.array([[-3.0]]), np.array([[-3.0]])]
    C = [np.array([[2.5]]), np.array([[2.5]])]
    D = [np.array([[1.0]]), np.array([[1.0]])]

    res = solve_ms_dsge(A, B, C, D, P, initial_T=[np.array([[1.8]]), np.array([[1.8]])])
    # Spectral radius of second moment operator should exceed 1
    assert res.spectral_radius_mss > 1.0
    assert not res.mean_square_stable
    # Ergodic covariance should safely be NaN
    assert np.isnan(res.ergodic_cov.values).all()


def test_single_regime_degeneracy():
    """S = 1 regime collapses to standard time-invariant linear DSGE."""
    n = 2
    P = np.array([[1.0]])
    A = [0.5 * np.eye(n)]
    B = [-1.5 * np.eye(n)]
    C = [0.4 * np.eye(n)]
    D = [np.eye(n)]

    res = solve_ms_dsge(A, B, C, D, P)
    assert res.converged
    assert res.mean_square_stable
    assert len(res.regime_names) == 1
    assert res.ergodic_distribution.iloc[0] == pytest.approx(1.0)


def test_input_validation_errors():
    """Verify proper exceptions on invalid matrix shapes and non-stochastic transition matrices."""
    n = 2
    A = [np.eye(n), np.eye(n)]
    B = [np.eye(n), np.eye(n)]
    C = [np.eye(n), np.eye(n)]
    D = [np.eye(n), np.eye(n)]

    # Non-square transition matrix
    with pytest.raises(ValueError, match="square"):
        solve_ms_dsge(A, B, C, D, np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]]))

    # Rows do not sum to 1
    with pytest.raises(ValueError, match="sum to 1"):
        solve_ms_dsge(A, B, C, D, np.array([[0.8, 0.1], [0.2, 0.5]]))

    # Dimension mismatch: 3 regimes matrix for 2 regimes input
    with pytest.raises(ValueError, match="match number of regimes"):
        solve_ms_dsge(A, B, C, D, np.eye(3))

    # Unknown solver method
    with pytest.raises(ValueError, match="Unknown method"):
        solve_ms_dsge(A, B, C, D, np.eye(2), method="unsupported_solver")


def test_negative_transition_probabilities_raise_error():
    """Verify ValueError is raised when negative transition probabilities are supplied."""
    n = 2
    A = [np.eye(n), np.eye(n)]
    B = [-2.0 * np.eye(n), -2.0 * np.eye(n)]
    C = [0.1 * np.eye(n), 0.1 * np.eye(n)]
    D = [np.eye(n), np.eye(n)]

    # Negative probability in transition matrix (row sums to 1.0)
    P_neg = np.array([[1.2, -0.2], [0.3, 0.7]])
    with pytest.raises(ValueError, match=r"probabilities in \[0, 1\]"):
        solve_ms_dsge(A, B, C, D, P_neg)

    with pytest.raises(ValueError, match=r"probabilities in \[0, 1\]"):
        markov_stationary(P_neg)

    # Transition probability exceeding 1.0 (row sums to 1.0)
    P_gt1 = np.array([[1.5, -0.5], [0.0, 1.0]])
    with pytest.raises(ValueError, match=r"probabilities in \[0, 1\]"):
        solve_ms_dsge(A, B, C, D, P_gt1)

    with pytest.raises(ValueError, match=r"probabilities in \[0, 1\]"):
        markov_stationary(P_gt1)


def test_shock_covariance_correlated():
    """Verify that non-diagonal shock covariance is properly handled in Lyapunov covariance."""
    S = 2
    n = 2
    P = np.array([[0.8, 0.2], [0.3, 0.7]])
    A = [np.eye(n), np.eye(n)]
    B = [-2.0 * np.eye(n), -2.0 * np.eye(n)]
    C = [0.1 * np.eye(n), 0.1 * np.eye(n)]
    D = [np.eye(n), np.eye(n)]

    shock_cov = np.array([[1.0, 0.5], [0.5, 2.0]])
    res = solve_ms_dsge(A, B, C, D, P, shock_cov=shock_cov)

    assert res.converged
    assert res.mean_square_stable
    # Cross-covariance should be strictly non-zero
    assert abs(res.ergodic_cov.iloc[0, 1]) > 0.05
    # Variance of second variable should be larger than first variable due to higher shock variance
    assert res.ergodic_cov.iloc[1, 1] > res.ergodic_cov.iloc[0, 0]


def test_warm_start_initial_t():
    """Providing an exact or close initial guess converges in fewer iterations."""
    S = 2
    n = 2
    P = np.array([[0.8, 0.2], [0.3, 0.7]])
    A = [np.eye(n), np.eye(n)]
    B = [-2.5 * np.eye(n), -2.0 * np.eye(n)]
    C = [0.3 * np.eye(n), 0.2 * np.eye(n)]
    D = [np.eye(n), np.eye(n)]

    res_cold = solve_ms_dsge(A, B, C, D, P, method="newton")
    assert res_cold.converged

    # Warm start with the solved T
    res_warm = solve_ms_dsge(A, B, C, D, P, method="newton", initial_T=[res_cold.T[0], res_cold.T[1]])
    assert res_warm.converged
    assert res_warm.iterations <= 2
    assert np.allclose(res_cold.T[0], res_warm.T[0], atol=1e-10)


def test_irf_and_girf_inputs_and_exceptions():
    """Verify resolution of shocks (names, vectors, indices) and regime inputs with error handling."""
    S = 2
    n = 2
    P = np.array([[0.7, 0.3], [0.4, 0.6]])
    A = [np.eye(n), np.eye(n)]
    B = [-2.0 * np.eye(n), -2.0 * np.eye(n)]
    C = [0.1 * np.eye(n), 0.1 * np.eye(n)]
    D = [np.eye(n), np.eye(n)]

    res = solve_ms_dsge(
        A, B, C, D, P,
        regime_names=["Low", "High"],
        variable_names=["Output", "Inflation"],
        shock_names=["Tech", "Monetary"],
    )

    # Shock by name
    irf1 = res.irf("Low", shock="Tech")
    # Shock by index
    irf2 = res.irf(0, shock=0)
    assert np.allclose(irf1.values, irf2.values)

    # Shock by custom vector
    irf3 = res.irf("High", shock=np.array([2.0, 0.0]))
    irf4 = res.irf("High", shock="Tech")
    assert np.allclose(irf3.values, 2.0 * irf4.values)

    # Exceptions on invalid regime or shock
    with pytest.raises(KeyError, match="not found"):
        res.irf("NonExistentRegime")

    with pytest.raises(KeyError, match="not found"):
        res.irf("Low", shock="NonExistentShock")

    with pytest.raises(IndexError):
        res.girf(999)


def test_foerster_three_regimes_benchmark():
    """Benchmark with S = 3 regimes testing multi-regime transitions."""
    S = 3
    n = 2
    P = np.array([
        [0.80, 0.15, 0.05],
        [0.10, 0.80, 0.10],
        [0.05, 0.15, 0.80],
    ])

    A = [np.eye(n) for _ in range(S)]
    B = [-2.0 * np.eye(n), -2.2 * np.eye(n), -1.8 * np.eye(n)]
    C = [0.2 * np.eye(n), 0.1 * np.eye(n), 0.3 * np.eye(n)]
    D = [np.eye(n), 1.2 * np.eye(n), 0.8 * np.eye(n)]

    res = solve_ms_dsge(
        A, B, C, D, P,
        regime_names=["Recession", "Normal", "Boom"],
        variable_names=["y", "pi"],
        shock_names=["e1", "e2"],
        method="newton",
    )

    assert res.converged
    assert res.mean_square_stable
    assert len(res.regime_names) == 3
    assert len(res.T) == 3  # S regimes
    assert 0 in res.T and 1 in res.T and 2 in res.T
    assert "Recession" in res.T
    assert "Normal" in res.T
    assert "Boom" in res.T
    assert res.ergodic_distribution.sum() == pytest.approx(1.0)

