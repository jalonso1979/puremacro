"""Empirical Challenger Test Suite for Markov-Switching DSGE (MS-DSGE).

Author: challenger_2
Target: puremacro/dsge/markov_switching.py

Verification Objectives (Adversarial Empirical Challenge):
1. Coupled quadratic matrix solver convergence under near-absorbing Markov regimes (p_11 = 0.9999, 0.99999).
2. Convergence comparison: Block Newton-Raphson (quadratic convergence) vs Damped Functional Iteration.
3. Monetary policy Hawkish vs Dovish regimes where the Dovish regime exhibits extreme
   indeterminacy in isolation (phi_pi = 0.1, 0.0, -0.2) while coupled MSRE system achieves Mean-Square Stability.
4. Closed-form analytical GIRF vs 25,000+ trajectory Monte Carlo simulation verified within 3-sigma standard error.
5. Stationary distribution pi_infty P = pi_infty and discrete Lyapunov covariance (I - M2) v = q positive semi-definiteness.
6. Mean-Square Stability operator M2 spectral radius rho(M2) < 1 vs mean stability operator M1.
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
from puremacro.dsge.dynare import build_dynare
from puremacro.dsge.klein import BlanchardKahnError


# ==============================================================================
# Helper DGP Fixtures
# ==============================================================================

def make_monetary_policy_ms_model(
    phi_pi_hawkish: float = 1.8,
    phi_pi_dovish: float = 0.1,
    rho_i: float = 0.8,
    beta: float = 0.99,
    sigma: float = 1.0,
    kappa: float = 0.1,
    phi_x: float = 0.1,
):
    """Construct 3-equation New Keynesian model with switching Taylor rule."""
    S = 2
    n = 3
    phi_pi_list = [phi_pi_hawkish, phi_pi_dovish]

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

    return A_list, B_list, C_list, D_list


# ==============================================================================
# 1. Stress-Test Near-Absorbing Markov Regimes
# ==============================================================================

@pytest.mark.parametrize("p11, p22", [
    (0.9999, 0.95),
    (0.99999, 0.99999),
    (0.999999, 0.80),
    (0.90, 0.9999),
])
def test_stress_near_absorbing_markov_regimes(p11: float, p22: float):
    """Stress-test coupled quadratic solver convergence under near-absorbing Markov regimes."""
    A, B, C, D = make_monetary_policy_ms_model(phi_pi_hawkish=1.8, phi_pi_dovish=0.8)
    P = np.array([[p11, 1.0 - p11],
                  [1.0 - p22, p22]])

    # Newton solve
    res_newton = solve_ms_dsge(
        A, B, C, D, P,
        regime_names=["Regime_1", "Regime_2"],
        variable_names=["output", "inflation", "interest"],
        shock_names=["shk_demand", "shk_supply", "shk_monpol"],
        method="newton",
        tol=1e-10,
    )

    assert res_newton.converged, f"Newton failed to converge for near-absorbing P with p11={p11}, p22={p22}"
    assert res_newton.iterations <= 15, f"Newton took excessive iterations ({res_newton.iterations})"
    assert res_newton.diff < 1e-10, f"Residual norm {res_newton.diff} exceeds tolerance"

    # Verify quadratic residuals directly
    for i in range(2):
        T_sum = sum(P[i, k] * res_newton.T[k] for k in range(2))
        Omega_i = A[i] @ T_sum + B[i]
        Fi = Omega_i @ res_newton.T[i] + C[i]
        assert np.max(np.abs(Fi)) < 1e-9

    # Stationary distribution must remain well-behaved
    pi = res_newton.ergodic_distribution.to_numpy()
    assert np.all(pi >= 0.0)
    assert np.isclose(np.sum(pi), 1.0, atol=1e-12)
    assert np.allclose(pi @ P, pi, atol=1e-10)


# ==============================================================================
# 2. Block Newton-Raphson vs Damped Functional Iteration
# ==============================================================================

def test_stress_newton_vs_damped_functional_iteration():
    """Compare Newton-Raphson vs Damped Functional Iteration on convergence, speed, and precision."""
    A, B, C, D = make_monetary_policy_ms_model(phi_pi_hawkish=2.0, phi_pi_dovish=0.2, rho_i=0.7)
    P = np.array([[0.85, 0.15],
                  [0.20, 0.80]])

    # Newton-Raphson
    res_newton = solve_ms_dsge(A, B, C, D, P, method="newton", tol=1e-11)
    assert res_newton.converged
    assert res_newton.iterations <= 8, f"Newton took {res_newton.iterations} iter"
    assert res_newton.diff < 1e-12

    # Functional Iteration with damping = 0.8
    res_fp_80 = solve_ms_dsge(A, B, C, D, P, method="functional_iteration", damping=0.8, tol=1e-10)
    assert res_fp_80.converged
    # Functional iteration requires significantly more iterations (linear vs quadratic convergence)
    assert res_fp_80.iterations > res_newton.iterations
    assert res_fp_80.iterations <= 150

    # Verify both converge to the identical MSV decision rules
    max_diff_0 = np.max(np.abs(res_newton.T[0] - res_fp_80.T[0]))
    max_diff_1 = np.max(np.abs(res_newton.T[1] - res_fp_80.T[1]))
    assert max_diff_0 < 1e-9, f"T[0] diff between Newton and FP is {max_diff_0:.2e}"
    assert max_diff_1 < 1e-9, f"T[1] diff between Newton and FP is {max_diff_1:.2e}"

    # Verify functional iteration under different damping factors
    for damping in [0.4, 0.6, 0.9]:
        res_fp_d = solve_ms_dsge(A, B, C, D, P, method="functional_iteration", damping=damping, tol=1e-8)
        assert res_fp_d.converged
        diff_d = max(np.max(np.abs(res_newton.T[k] - res_fp_d.T[k])) for k in range(2))
        assert diff_d < 1e-7


# ==============================================================================
# 3. Monetary Policy Hawkish vs Dovish (Extreme Indeterminacy in Isolation)
# ==============================================================================

def test_stress_hawkish_vs_dovish_extreme_indeterminacy():
    """Stress-test Hawkish vs Dovish regimes where Dovish has extreme indeterminacy in isolation (phi_pi = 0.1)."""
    # 1. First verify that the Dovish regime IN ISOLATION fails Blanchard-Kahn
    mod_dovish = """
    var y pi r;
    varexo e_r;
    parameters beta sigma kappa phi_pi phi_y rho_r;
    beta = 0.99;
    sigma = 1.0;
    kappa = 0.1;
    phi_pi = 0.1;
    phi_y = 0.1;
    rho_r = 0.8;
    model(linear);
    y = y(+1) - 1/sigma * (r - pi(+1));
    pi = beta * pi(+1) + kappa * y;
    r = rho_r * r(-1) + (1 - rho_r) * phi_pi * pi + (1 - rho_r) * phi_y * y + e_r;
    end;
    """
    with pytest.raises(BlanchardKahnError) as exc_info:
        build_dynare(mod_dovish)
    assert "indeterminacy" in str(exc_info.value).lower()

    # 2. Now verify that the coupled MSRE system stabilizes and achieves Mean-Square Stability
    A, B, C, D = make_monetary_policy_ms_model(phi_pi_hawkish=1.8, phi_pi_dovish=0.1, rho_i=0.8)
    P = np.array([[0.90, 0.10],
                  [0.20, 0.80]])

    res = solve_ms_dsge(
        A, B, C, D, P,
        regime_names=["Hawkish", "Dovish"],
        variable_names=["output", "inflation", "interest"],
        shock_names=["shk_demand", "shk_supply", "shk_monpol"],
        method="newton",
    )

    assert res.converged
    assert res.iterations <= 10
    assert res.diff < 1e-12

    # Verify Mean-Square Stability holds globally
    assert res.mean_square_stable
    assert res.spectral_radius_mss < 1.0
    assert res.spectral_radius_mean < 1.0

    # Dovish regime in isolation has higher persistence than Hawkish
    rho_hawk = np.max(np.abs(np.linalg.eigvals(res.T["Hawkish"])))
    rho_dov = np.max(np.abs(np.linalg.eigvals(res.T["Dovish"])))
    assert rho_dov > rho_hawk

    # Test extreme dovish grid: phi_pi = 0.05, 0.01, 0.0, -0.2
    for phi_dov in [0.05, 0.01, 0.0, -0.2]:
        A_ext, B_ext, C_ext, D_ext = make_monetary_policy_ms_model(phi_pi_hawkish=1.8, phi_pi_dovish=phi_dov, rho_i=0.8)
        res_ext = solve_ms_dsge(A_ext, B_ext, C_ext, D_ext, P, method="newton")
        assert res_ext.converged
        assert res_ext.mean_square_stable
        assert res_ext.spectral_radius_mss < 1.0


# ==============================================================================
# 4. Analytical Closed-Form GIRF vs 25,000-Trajectory Monte Carlo
# ==============================================================================

def test_stress_girf_vs_monte_carlo_simulation():
    """Verify analytical closed-form GIRF against 25,000 Monte Carlo trajectories within 3-sigma SE."""
    A, B, C, D = make_monetary_policy_ms_model(phi_pi_hawkish=1.8, phi_pi_dovish=0.1, rho_i=0.7)
    P = np.array([[0.85, 0.15],
                  [0.25, 0.75]])

    res = solve_ms_dsge(
        A, B, C, D, P,
        regime_names=["Hawkish", "Dovish"],
        variable_names=["output", "inflation", "interest"],
        shock_names=["demand", "supply", "monpol"],
    )

    H = 15
    N_sim = 25000
    rng = np.random.default_rng(20260910)
    n = len(res.variables)

    # Test across both initial regimes
    for init_s, init_name in enumerate(["Hawkish", "Dovish"]):
        # Analytical GIRF following a unit monetary policy shock
        girf_an = res.girf(init_name, shock="monpol", horizon=H).to_numpy()

        # Empirical Monte Carlo simulation over N_sim Markov regime paths
        y_mc = np.zeros((N_sim, H + 1, n))
        y0 = res.R[init_s] @ np.array([0.0, 0.0, 1.0])
        y_mc[:, 0, :] = y0

        s_paths = np.zeros((N_sim, H + 1), dtype=int)
        s_paths[:, 0] = init_s

        for h in range(H):
            curr_s = s_paths[:, h]
            u = rng.uniform(0.0, 1.0, size=N_sim)
            next_s = np.where(curr_s == 0, (u < P[0, 1]).astype(int), (u < P[1, 1]).astype(int))
            s_paths[:, h + 1] = next_s

            mask0 = (next_s == 0)
            mask1 = (next_s == 1)
            if np.any(mask0):
                y_mc[mask0, h + 1, :] = y_mc[mask0, h, :] @ res.T[0].T
            if np.any(mask1):
                y_mc[mask1, h + 1, :] = y_mc[mask1, h, :] @ res.T[1].T

        mean_mc = np.mean(y_mc, axis=0)
        std_mc = np.std(y_mc, axis=0)
        se_mc = std_mc / np.sqrt(N_sim)

        # Discrepancy between analytical GIRF and Monte Carlo mean
        diff = np.abs(girf_an - mean_mc)
        z_scores = diff / np.maximum(se_mc, 1e-12)
        max_z = np.max(z_scores)

        # Must strictly be within 3-sigma standard error
        assert max_z < 3.0, f"Max Z-score {max_z:.3f} exceeded 3.0 for initial regime {init_name}"
        assert np.all(diff <= 3.0 * se_mc + 1e-12)


# ==============================================================================
# 5. Ergodic Distribution & Discrete Lyapunov Covariance PSD
# ==============================================================================

def test_stress_ergodic_distribution_and_lyapunov_covariance_psd():
    """Verify ergodic distribution pi_infty P = pi_infty and discrete Lyapunov covariance (I - M2) v = q is PSD."""
    # 1. Test stationary distribution properties across diverse Markov transition matrices
    for S in [2, 3, 4]:
        rng = np.random.default_rng(S * 42)
        raw = rng.uniform(0.05, 1.0, size=(S, S))
        P = raw / raw.sum(axis=1, keepdims=True)
        pi = markov_stationary(P)
        assert np.all(pi >= 0.0)
        assert np.isclose(np.sum(pi), 1.0, atol=1e-12)
        assert np.allclose(pi @ P, pi, atol=1e-10)

    # 2. Discrete Lyapunov covariance in 3-equation DSGE
    A, B, C, D = make_monetary_policy_ms_model(phi_pi_hawkish=1.8, phi_pi_dovish=0.8, rho_i=0.7)
    P = np.array([[0.85, 0.15],
                  [0.25, 0.75]])

    res = solve_ms_dsge(A, B, C, D, P)

    cov_mat = res.ergodic_cov.to_numpy()

    # Exact symmetry
    symm_err = np.max(np.abs(cov_mat - cov_mat.T))
    assert symm_err < 1e-12, f"Covariance matrix not symmetric: max asymmetry = {symm_err:.2e}"

    # Positive Semi-Definiteness (all eigenvalues >= 0)
    eigs = np.linalg.eigvalsh(cov_mat)
    assert np.all(eigs >= -1e-12), f"Negative eigenvalue found in covariance: {eigs}"

    # Verify against long-run 100,000-period simulation
    df_sim, _ = res.simulate(periods=100_000, seed=42)
    emp_cov = df_sim.cov().to_numpy()
    rel_err = np.max(np.abs(cov_mat - emp_cov) / np.maximum(np.abs(cov_mat), 1e-3))
    assert rel_err < 0.06, f"Analytical Lyapunov covariance differs from empirical by {rel_err:.2%}"


# ==============================================================================
# 6. Mean-Square Stability Operator Spectral Radius rho(M2) < 1
# ==============================================================================

def test_stress_mean_square_stability_spectral_radius():
    """Verify Mean-Square Stability operator M2 and distinguish from first-moment operator M1."""
    # Construct a scalar system: y_t = T(s_t) y_{t-1} + eps_t
    # In regime 1: T1 = 0.2
    # In regime 2: T2 = 2.5
    # P = [[0.8, 0.2], [0.8, 0.2]]
    # First moment operator M1 has spectral radius 0.8 * 0.2 + 0.2 * 2.5 = 0.66 < 1.0 (Mean Stable!)
    # Second moment operator M2 has spectral radius 0.8 * 0.04 + 0.2 * 6.25 = 1.282 > 1.0 (MSS False!)
    P = np.array([[0.8, 0.2],
                  [0.8, 0.2]])
    A = [np.array([[0.0]]), np.array([[0.0]])]
    B = [np.array([[-1.0]]), np.array([[-1.0]])]
    C = [np.array([[0.2]]), np.array([[2.5]])]
    D = [np.array([[1.0]]), np.array([[1.0]])]

    res_unstable = solve_ms_dsge(A, B, C, D, P)

    # First moment is stable
    assert res_unstable.spectral_radius_mean == pytest.approx(0.66, abs=1e-4)
    assert res_unstable.spectral_radius_mean < 1.0

    # Second moment is explosive
    assert res_unstable.spectral_radius_mss == pytest.approx(1.282, abs=1e-4)
    assert not res_unstable.mean_square_stable
    assert res_unstable.spectral_radius_mss >= 1.0

    # Discrete Lyapunov covariance correctly returns NaN when MSS fails
    assert np.isnan(res_unstable.ergodic_cov.values[0, 0])

    # Now verify continuous behavior near boundary: rho(M2) = 0.98
    # 0.032 + 0.2 * t2^2 = 0.98 => t2 = sqrt(0.948 / 0.2) = 2.1771541
    t2_98 = float(np.sqrt((0.98 - 0.032) / 0.2))
    C_98 = [np.array([[0.2]]), np.array([[t2_98]])]
    res_98 = solve_ms_dsge(A, B, C_98, D, P)

    assert res_98.mean_square_stable
    assert res_98.spectral_radius_mss == pytest.approx(0.98, abs=1e-4)
    # Theoretical variance for unit shock: q / (1 - rho(M2)) = 1.0 / (1 - 0.98) = 50.0
    assert res_98.ergodic_cov.values[0, 0] == pytest.approx(50.0, rel=1e-6)


# ==============================================================================
# 7. Presentation & API Contract
# ==============================================================================

def test_stress_presentation_contract_and_summary():
    """Verify presentation contract: summary, plot, to_markdown, to_latex, to_typst, simulate."""
    A, B, C, D = make_monetary_policy_ms_model(phi_pi_hawkish=1.8, phi_pi_dovish=0.1)
    P = np.array([[0.9, 0.1], [0.2, 0.8]])

    res = solve_ms_dsge(
        A, B, C, D, P,
        regime_names=["Hawkish", "Dovish"],
        variable_names=["output", "inflation", "interest"],
        shock_names=["demand", "supply", "monpol"],
    )

    # summary
    summary_str = res.summary()
    assert "MARKOV-SWITCHING DSGE EQUILIBRIUM" in summary_str
    assert "Hawkish" in summary_str
    assert "Dovish" in summary_str
    assert "STABILITY DIAGNOSTICS:" in summary_str

    # summary as dataframe
    sum_df = res.summary(as_dataframe=True)
    assert isinstance(sum_df, pd.DataFrame)
    assert len(sum_df) == 2

    # export tables
    md = res.to_markdown()
    assert "Hawkish" in md
    latex = res.to_latex()
    assert "\\begin{tabular}" in latex
    typst = res.to_typst()
    assert "#table(" in typst

    # plotting (headless Agg mode)
    fig, axes = res.plot(girf=True, horizon=10)
    assert fig is not None
    plt.close(fig)

    fig_irf, axes_irf = res.plot(regime="Hawkish", girf=False, horizon=10)
    assert fig_irf is not None
    plt.close(fig_irf)

    # simulate with seed
    df_sim1, reg1 = res.simulate(periods=50, seed=123)
    df_sim2, reg2 = res.simulate(periods=50, seed=123)
    pd.testing.assert_frame_equal(df_sim1, df_sim2)
    np.testing.assert_array_equal(reg1, reg2)


# ==============================================================================
# 8. Multi-Regime Higher-Dimension & Adversarial Stress Tests
# ==============================================================================

def test_stress_high_dimension_and_multiregime():
    """Stress-test S=4 regimes with 5 variables (3 endogenous + 2 AR(1) exogenous shocks)."""
    S = 4
    n = 5
    # y = [output, inflation, interest, a_tech, u_cost]
    # a_tech_t = rho_a * a_tech_{t-1} + eps_a
    # u_cost_t = rho_u * u_cost_{t-1} + eps_u
    # y_t = E_t y_{t+1} - 1/sigma (i_t - E_t pi_{t+1}) + a_tech_t
    # pi_t = beta E_t pi_{t+1} + kappa y_t + u_cost_t
    # i_t = rho_i * i_{t-1} + (1 - rho_i) * (phi_pi(s) * pi_t + phi_y * y_t) + eps_r

    P = np.array([
        [0.70, 0.15, 0.10, 0.05],
        [0.10, 0.70, 0.10, 0.10],
        [0.05, 0.15, 0.70, 0.10],
        [0.10, 0.10, 0.10, 0.70]
    ])

    phi_pi_grid = [2.5, 1.5, 0.8, 0.1]
    rho_i = 0.75
    rho_a = 0.85
    rho_u = 0.50
    beta = 0.99
    sigma = 1.0
    kappa = 0.15
    phi_y = 0.10

    A_list = []
    B_list = []
    C_list = []
    D_list = []

    for s in range(S):
        phi_pi_s = phi_pi_grid[s]
        # Endogenous equations
        As = np.zeros((n, n))
        As[0, 0] = 1.0
        As[0, 1] = 1.0 / sigma
        As[1, 1] = beta

        Bs = np.zeros((n, n))
        Bs[0, 0] = -1.0
        Bs[0, 2] = -1.0 / sigma
        Bs[0, 3] = 1.0  # technology shock in IS
        Bs[1, 0] = kappa
        Bs[1, 1] = -1.0
        Bs[1, 4] = 1.0  # cost-push shock in NKPC
        Bs[2, 0] = (1.0 - rho_i) * phi_y
        Bs[2, 1] = (1.0 - rho_i) * phi_pi_s
        Bs[2, 2] = -1.0
        Bs[3, 3] = -1.0  # a_tech
        Bs[4, 4] = -1.0  # u_cost

        Cs = np.zeros((n, n))
        Cs[2, 2] = rho_i
        Cs[3, 3] = rho_a
        Cs[4, 4] = rho_u

        Ds = np.zeros((n, 3))
        Ds[2, 0] = 1.0  # eps_r
        Ds[3, 1] = 1.0  # eps_a
        Ds[4, 2] = 1.0  # eps_u

        A_list.append(As)
        B_list.append(Bs)
        C_list.append(Cs)
        D_list.append(Ds)

    res = solve_ms_dsge(
        A_list, B_list, C_list, D_list, P,
        regime_names=["Hawkish_Strong", "Hawkish_Standard", "Dovish_Mild", "Dovish_Extreme"],
        variable_names=["y", "pi", "r", "a", "u"],
        shock_names=["e_r", "e_a", "e_u"],
        method="newton",
        tol=1e-10,
    )

    assert res.converged
    assert res.iterations <= 12
    assert res.diff < 1e-10
    assert res.mean_square_stable
    assert res.spectral_radius_mss < 1.0

    # Covariance PSD
    eigs = np.linalg.eigvalsh(res.ergodic_cov.to_numpy())
    assert np.all(eigs >= -1e-12)


def test_stress_near_singular_shock_covariance():
    """Verify that discrete Lyapunov covariance remains robust under near-singular shock covariance."""
    A, B, C, D = make_monetary_policy_ms_model(phi_pi_hawkish=1.8, phi_pi_dovish=0.8)
    P = np.array([[0.85, 0.15],
                  [0.25, 0.75]])

    # Rank-1 shock covariance (perfect collinearity between shocks)
    v = np.array([1.0, 0.5, -0.5])
    Sigma_rank1 = np.outer(v, v)
    assert np.linalg.matrix_rank(Sigma_rank1) == 1

    res = solve_ms_dsge(A, B, C, D, P, shock_cov=Sigma_rank1)
    assert res.converged
    assert res.mean_square_stable

    cov_mat = res.ergodic_cov.to_numpy()
    assert np.max(np.abs(cov_mat - cov_mat.T)) < 1e-12
    eigs = np.linalg.eigvalsh(cov_mat)
    assert np.all(eigs >= -1e-12)


def test_stress_invalid_inputs_and_exception_handling():
    """Verify robust exception handling on malformed transition matrices and invalid dimensions."""
    A, B, C, D = make_monetary_policy_ms_model()

    # Rows do not sum to 1.0
    P_invalid_sum = np.array([[0.5, 0.2], [0.3, 0.4]])
    with pytest.raises(ValueError, match="rows must sum to 1"):
        solve_ms_dsge(A, B, C, D, P_invalid_sum)

    # Non-square transition matrix
    P_nonsquare = np.array([[0.5, 0.5, 0.0], [0.2, 0.8, 0.0]])
    with pytest.raises(ValueError, match="must be square"):
        solve_ms_dsge(A, B, C, D, P_nonsquare)

    # Incompatible regime count (S=3 transition matrix for 2-regime model)
    P_mismatch = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    with pytest.raises(ValueError, match="must match number of regimes"):
        solve_ms_dsge(A, B, C, D, P_mismatch)

    # Unknown solver method
    P_valid = np.array([[0.9, 0.1], [0.2, 0.8]])
    with pytest.raises(ValueError, match="Unknown method"):
        solve_ms_dsge(A, B, C, D, P_valid, method="invalid_algorithm")

