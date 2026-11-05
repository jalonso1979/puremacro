"""Comprehensive tests for semi-structural policy counterfactuals and OPP evaluation.

Covers:
- McKay & Wolf (2023, REStud): `mckay_wolf_counterfactual` and `PolicyCounterfactualResult`
- Barnichon & Mesters (2020, AER; 2023, JPE): `barnichon_mesters_opp` and `OptimalPolicyPerturbationResult`
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro.lp.counterfactual import (
    OptimalPolicyPerturbationResult,
    PolicyCounterfactualResult,
    barnichon_mesters_opp,
    mckay_wolf_counterfactual,
)


# ---------------------------------------------------------------------------
# Helpers: Structural New Keynesian Sequence-Space Model for Analytical Tests
# ---------------------------------------------------------------------------
def _solve_structural_nk_single_target(
    H: int,
    phi_pi: float,
    beta: float = 0.99,
    kappa: float = 0.15,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct sequence-space Jacobians for a 3-equation NK model over horizons 0..H.

    Structural equations (for h = 0, ..., H with terminal condition at H+1 = 0):
      IS curve:  x_h = x_{h+1} - (1/sigma) * (i_h - pi_{h+1})
      NKPC:      pi_h = beta * pi_{h+1} + kappa * x_h + u_h
    Eliminating output gap x gives a linear sequence-space map:
      pi = M_{pi, i} @ i + M_{pi, u} @ u
    Under a Taylor rule i = phi_pi * pi + nu:
      (I - phi_pi * M_{pi, i}) @ pi = M_{pi, u} @ u + M_{pi, i} @ nu.
    """
    T_h = H + 1
    # Upper shift matrix S where (S @ v)_h = v_{h+1}
    S = np.eye(T_h, k=1, dtype=np.float64)
    I_mat = np.eye(T_h, dtype=np.float64)

    # IS: (I - S) x = -(1/sigma) * (i - S @ pi)  =>  x = -(1/sigma) * (I - S)^{-1} (i - S @ pi)
    inv_I_minus_S = np.linalg.inv(I_mat - S)
    # NKPC: (I - beta * S) pi = kappa * x + u
    # => [ (I - beta * S) - (kappa / sigma) * inv_I_minus_S @ S ] pi
    #    = -(kappa / sigma) * inv_I_minus_S @ i + u
    A_pi = (I_mat - beta * S) - (kappa / sigma) * (inv_I_minus_S @ S)
    A_pi_inv = np.linalg.inv(A_pi)

    M_pi_i = -(kappa / sigma) * (A_pi_inv @ inv_I_minus_S)
    M_pi_u = A_pi_inv

    Gamma = np.linalg.inv(I_mat - phi_pi * M_pi_i)
    return M_pi_i, M_pi_u, Gamma


def _solve_structural_nk_dual_target(
    H: int,
    phi_pi: float,
    phi_x: float,
    beta: float = 0.99,
    kappa: float = 0.15,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve structural NK model with dual-mandate rule i_h = phi_pi * pi_h + phi_x * x_h + nu_h."""
    T_h = H + 1
    S = np.eye(T_h, k=1, dtype=np.float64)
    I_mat = np.eye(T_h, dtype=np.float64)
    inv_I_minus_S = np.linalg.inv(I_mat - S)

    # x = -(1/sigma) * inv_I_minus_S @ i + (1/sigma) * inv_I_minus_S @ S @ pi
    M_x_i = -(1.0 / sigma) * inv_I_minus_S
    M_x_pi = (1.0 / sigma) * (inv_I_minus_S @ S)

    # pi = (I - beta*S)^{-1} (kappa * x + u)
    inv_nkpc = np.linalg.inv(I_mat - beta * S)
    # Substitute x: (I - inv_nkpc * kappa * M_x_pi) pi = inv_nkpc * kappa * M_x_i @ i + inv_nkpc @ u
    G_pi = np.linalg.inv(I_mat - kappa * (inv_nkpc @ M_x_pi))
    J_pi_i = G_pi @ (kappa * (inv_nkpc @ M_x_i))
    J_pi_u = G_pi @ inv_nkpc

    J_x_i = M_x_i + M_x_pi @ J_pi_i
    J_x_u = M_x_pi @ J_pi_u

    # Policy rule: i = phi_pi * pi + phi_x * x + nu
    # => (I - phi_pi * J_pi_i - phi_x * J_x_i) i = (phi_pi * J_pi_u + phi_x * J_x_u) u + nu
    inv_policy = np.linalg.inv(I_mat - phi_pi * J_pi_i - phi_x * J_x_i)
    return J_pi_i, J_pi_u, J_x_i, J_x_u, inv_policy


# ===========================================================================
# Part 1: McKay & Wolf (2023) Counterfactual Tests
# ===========================================================================
def test_mckay_wolf_exact_zero_rule_residual_full_shock_span() -> None:
    """When n_shocks == H + 1 and ridge == 0, counterfactual rule residual is zero (< 1e-12)."""
    H = 8
    T_h = H + 1
    M_pi_i, M_pi_u, Gamma_base = _solve_structural_nk_single_target(H=H, phi_pi=1.5)

    # Cost-push shock u_h = 0.8^h
    u_shock = 0.8 ** np.arange(T_h)
    pi_base = Gamma_base @ (M_pi_u @ u_shock)
    i_base = 1.5 * pi_base

    # Empirical IRFs to H+1 monetary policy news shocks
    theta_pi = Gamma_base @ M_pi_i
    theta_i = np.eye(T_h) + 1.5 * theta_pi

    res = mckay_wolf_counterfactual(
        baseline_target=pi_base,
        baseline_instrument=i_base,
        policy_target_irfs=theta_pi,
        policy_instrument_irfs=theta_i,
        rule_target_weights=2.5,
        ridge=0.0,
    )

    assert isinstance(res, PolicyCounterfactualResult)
    assert res.horizon == H
    assert res.rule_rmse < 1e-12
    assert np.max(np.abs(res.rule_residual)) < 1e-12
    np.testing.assert_allclose(
        res.counterfactual_instrument, 2.5 * res.counterfactual_target, atol=1e-12
    )


def test_mckay_wolf_equivalence_theorem_2_period_closed_form() -> None:
    """Verify McKay-Wolf Proposition 1 on an exact 2-period (H=1) NK model."""
    H = 1
    phi_base = 1.25
    phi_cf = 2.20
    M_pi_i, M_pi_u, Gamma_base = _solve_structural_nk_single_target(
        H=H, phi_pi=phi_base, beta=0.98, kappa=0.20, sigma=1.0
    )
    _, _, Gamma_cf = _solve_structural_nk_single_target(
        H=H, phi_pi=phi_cf, beta=0.98, kappa=0.20, sigma=1.0
    )

    u_shock = np.array([1.0, 0.5])
    # Baseline paths under phi_base
    pi_base = Gamma_base @ (M_pi_u @ u_shock)
    i_base = phi_base * pi_base

    # True structural equilibrium under phi_cf
    pi_true_cf = Gamma_cf @ (M_pi_u @ u_shock)
    i_true_cf = phi_cf * pi_true_cf

    # Policy news shock IRFs under baseline regime
    theta_pi = Gamma_base @ M_pi_i
    theta_i = Gamma_base

    res = mckay_wolf_counterfactual(
        baseline_target=pi_base,
        baseline_instrument=i_base,
        policy_target_irfs=theta_pi,
        policy_instrument_irfs=theta_i,
        rule_target_weights=phi_cf,
        ridge=0.0,
    )

    np.testing.assert_allclose(res.counterfactual_target, pi_true_cf, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(res.counterfactual_instrument, i_true_cf, atol=1e-12, rtol=1e-12)
    assert res.rule_rmse < 1e-13


def test_mckay_wolf_equivalence_theorem_multi_horizon_nk() -> None:
    """Verify McKay-Wolf equivalence theorem over H=15 horizons in a forward-looking NK model."""
    H = 15
    T_h = H + 1
    phi_base = 1.5
    phi_cf = 3.0

    M_pi_i, M_pi_u, Gamma_base = _solve_structural_nk_single_target(H=H, phi_pi=phi_base)
    _, _, Gamma_cf = _solve_structural_nk_single_target(H=H, phi_pi=phi_cf)

    u_shock = 0.75 ** np.arange(T_h)
    pi_base = Gamma_base @ (M_pi_u @ u_shock)
    i_base = phi_base * pi_base

    pi_true_cf = Gamma_cf @ (M_pi_u @ u_shock)
    i_true_cf = phi_cf * pi_true_cf

    theta_pi = Gamma_base @ M_pi_i
    theta_i = Gamma_base

    res = mckay_wolf_counterfactual(
        baseline_target=pi_base,
        baseline_instrument=i_base,
        policy_target_irfs=theta_pi,
        policy_instrument_irfs=theta_i,
        rule_target_weights=phi_cf,
    )

    np.testing.assert_allclose(res.counterfactual_target, pi_true_cf, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(res.counterfactual_instrument, i_true_cf, atol=1e-11, rtol=1e-11)
    assert res.rule_rmse < 1e-12


def test_mckay_wolf_interest_rate_peg_counterfactual() -> None:
    """Counterfactual interest rate peg (phi_pi = 0 => i_cf = 0) recovers exact structural peg."""
    H = 10
    T_h = H + 1
    phi_base = 1.5
    M_pi_i, M_pi_u, Gamma_base = _solve_structural_nk_single_target(H=H, phi_pi=phi_base)

    u_shock = 0.6 ** np.arange(T_h)
    pi_base = Gamma_base @ (M_pi_u @ u_shock)
    i_base = phi_base * pi_base

    # Under an interest rate peg i_cf = 0, true structural inflation is M_{pi, u} @ u_shock
    pi_peg_true = M_pi_u @ u_shock

    theta_pi = Gamma_base @ M_pi_i
    theta_i = Gamma_base

    res = mckay_wolf_counterfactual(
        baseline_target=pi_base,
        baseline_instrument=i_base,
        policy_target_irfs=theta_pi,
        policy_instrument_irfs=theta_i,
        rule_target_weights=0.0,
    )

    np.testing.assert_allclose(res.counterfactual_instrument, np.zeros(T_h), atol=1e-12)
    np.testing.assert_allclose(res.counterfactual_target, pi_peg_true, atol=1e-12)
    assert res.rule_rmse < 1e-12


def test_mckay_wolf_inertial_policy_rule_matrices() -> None:
    """Verify general rule matrices A_i and A_y for an inertial Taylor rule i_t = rho i_{t-1} + (1-rho) phi pi_t."""
    H = 8
    T_h = H + 1
    phi_base = 1.5
    M_pi_i, M_pi_u, Gamma_base = _solve_structural_nk_single_target(H=H, phi_pi=phi_base)

    u_shock = 0.7 ** np.arange(T_h)
    pi_base = Gamma_base @ (M_pi_u @ u_shock)
    i_base = phi_base * pi_base

    theta_pi = Gamma_base @ M_pi_i
    theta_i = Gamma_base

    rho_i = 0.6
    phi_cf = 2.0
    # A_i i^cf + A_y pi^cf = 0 where (A_i i^cf)_h = i_h - rho_i i_{h-1}
    A_i = np.eye(T_h) - rho_i * np.eye(T_h, k=-1)
    A_y = -(1.0 - rho_i) * phi_cf * np.eye(T_h)

    res = mckay_wolf_counterfactual(
        baseline_target=pi_base,
        baseline_instrument=i_base,
        policy_target_irfs=theta_pi,
        policy_instrument_irfs=theta_i,
        rule_instrument_matrix=A_i,
        rule_target_matrix=A_y,
    )

    # Direct structural solution with A_i i + A_y pi = 0  =>  i = -A_i^{-1} A_y pi
    K_rule = -np.linalg.solve(A_i, A_y)
    pi_true = np.linalg.solve(np.eye(T_h) - M_pi_i @ K_rule, M_pi_u @ u_shock)
    i_true = K_rule @ pi_true

    np.testing.assert_allclose(res.counterfactual_target, pi_true, atol=1e-11)
    np.testing.assert_allclose(res.counterfactual_instrument, i_true, atol=1e-11)
    assert res.rule_rmse < 1e-12


def test_mckay_wolf_fewer_shocks_least_squares_projection() -> None:
    """When 1 <= n_shocks < H + 1, shock_weights solve the exact least-squares rule projection."""
    H = 10
    T_h = H + 1
    n_shocks = 3  # e.g., contemporaneous, 1-year forward guidance, 2-year forward guidance
    M_pi_i, M_pi_u, Gamma_base = _solve_structural_nk_single_target(H=H, phi_pi=1.5)

    u_shock = 0.75 ** np.arange(T_h)
    pi_base = Gamma_base @ (M_pi_u @ u_shock)
    i_base = 1.5 * pi_base

    full_theta_pi = Gamma_base @ M_pi_i
    full_theta_i = Gamma_base

    shock_cols = [0, 2, 5]
    theta_pi_sub = full_theta_pi[:, shock_cols]
    theta_i_sub = full_theta_i[:, shock_cols]

    phi_cf = 2.2
    res = mckay_wolf_counterfactual(
        baseline_target=pi_base,
        baseline_instrument=i_base,
        policy_target_irfs=theta_pi_sub,
        policy_instrument_irfs=theta_i_sub,
        rule_target_weights=phi_cf,
        ridge=0.0,
    )

    assert res.shock_weights.shape == (n_shocks,)
    # Verify normal equations M' r(w*) = 0
    M = theta_i_sub - phi_cf * theta_pi_sub
    np.testing.assert_allclose(M.T @ res.rule_residual, np.zeros(n_shocks), atol=1e-11)


def test_mckay_wolf_ridge_monotonic_shrinkage() -> None:
    """Increasing ridge penalty strictly shrinks ||w*||_2 while increasing rule_rmse."""
    H = 8
    T_h = H + 1
    M_pi_i, M_pi_u, Gamma_base = _solve_structural_nk_single_target(H=H, phi_pi=1.5)

    u_shock = 0.8 ** np.arange(T_h)
    pi_base = Gamma_base @ (M_pi_u @ u_shock)
    i_base = 1.5 * pi_base

    theta_pi = Gamma_base @ M_pi_i
    theta_i = Gamma_base

    ridges = [0.0, 1e-3, 1e-1, 1.0, 10.0]
    w_norms = []
    rmses = []
    for lam in ridges:
        res = mckay_wolf_counterfactual(
            baseline_target=pi_base,
            baseline_instrument=i_base,
            policy_target_irfs=theta_pi,
            policy_instrument_irfs=theta_i,
            rule_target_weights=2.5,
            ridge=lam,
        )
        w_norms.append(float(np.linalg.norm(res.shock_weights)))
        rmses.append(res.rule_rmse)

    # Strict monotonic shrinkage of weight norm
    for k in range(len(ridges) - 1):
        assert w_norms[k] > w_norms[k + 1]
        assert rmses[k] <= rmses[k + 1]


def test_mckay_wolf_dual_mandate_multi_target_exact() -> None:
    """Multi-target (H+1, 2) Taylor rule recovers exact dual-mandate structural equilibrium."""
    H = 8
    T_h = H + 1
    phi_pi_base, phi_x_base = 1.5, 0.25
    phi_pi_cf, phi_x_cf = 2.2, 0.60

    J_pi_i, J_pi_u, J_x_i, J_x_u, inv_pol_base = _solve_structural_nk_dual_target(
        H=H, phi_pi=phi_pi_base, phi_x=phi_x_base
    )
    _, _, _, _, inv_pol_cf = _solve_structural_nk_dual_target(
        H=H, phi_pi=phi_pi_cf, phi_x=phi_x_cf
    )

    u_shock = 0.7 ** np.arange(T_h)
    # Baseline paths
    i_base = inv_pol_base @ ((phi_pi_base * J_pi_u + phi_x_base * J_x_u) @ u_shock)
    pi_base = J_pi_u @ u_shock + J_pi_i @ i_base
    x_base = J_x_u @ u_shock + J_x_i @ i_base
    y_base_2d = np.column_stack([pi_base, x_base])  # shape (H+1, 2)

    # True counterfactual paths
    i_true_cf = inv_pol_cf @ ((phi_pi_cf * J_pi_u + phi_x_cf * J_x_u) @ u_shock)
    pi_true_cf = J_pi_u @ u_shock + J_pi_i @ i_true_cf
    x_true_cf = J_x_u @ u_shock + J_x_i @ i_true_cf
    y_true_cf_2d = np.column_stack([pi_true_cf, x_true_cf])

    # Policy shock IRFs under baseline regime: shape (H+1, 2, H+1)
    theta_i = inv_pol_base
    theta_pi = J_pi_i @ theta_i
    theta_x = J_x_i @ theta_i
    theta_y_3d = np.stack([theta_pi, theta_x], axis=1)

    res = mckay_wolf_counterfactual(
        baseline_target=y_base_2d,
        baseline_instrument=i_base,
        policy_target_irfs=theta_y_3d,
        policy_instrument_irfs=theta_i,
        rule_target_weights=[phi_pi_cf, phi_x_cf],
        ridge=0.0,
    )

    assert res.counterfactual_target.shape == (T_h, 2)
    assert res.rule_rmse < 1e-12
    np.testing.assert_allclose(res.counterfactual_target, y_true_cf_2d, atol=1e-11)
    np.testing.assert_allclose(res.counterfactual_instrument, i_true_cf, atol=1e-11)


def test_mckay_wolf_confidence_intervals_and_alpha_scaling() -> None:
    """Delta-method confidence bands are centered at counterfactual_target and widen as alpha decreases."""
    H = 6
    T_h = H + 1
    M_pi_i, M_pi_u, Gamma_base = _solve_structural_nk_single_target(H=H, phi_pi=1.5)
    u_shock = 0.7 ** np.arange(T_h)
    pi_base = Gamma_base @ (M_pi_u @ u_shock)
    i_base = 1.5 * pi_base
    theta_pi = Gamma_base @ M_pi_i
    theta_i = Gamma_base

    se_mat = np.full_like(theta_pi, 0.05)

    res_90 = mckay_wolf_counterfactual(
        baseline_target=pi_base,
        baseline_instrument=i_base,
        policy_target_irfs=theta_pi,
        policy_instrument_irfs=theta_i,
        rule_target_weights=2.0,
        target_irf_se=se_mat,
        alpha=0.10,
    )
    res_95 = mckay_wolf_counterfactual(
        baseline_target=pi_base,
        baseline_instrument=i_base,
        policy_target_irfs=theta_pi,
        policy_instrument_irfs=theta_i,
        rule_target_weights=2.0,
        target_irf_se=se_mat,
        alpha=0.05,
    )

    assert res_90.ci_lower is not None and res_90.ci_upper is not None
    assert res_95.ci_lower is not None and res_95.ci_upper is not None
    np.testing.assert_allclose(
        0.5 * (res_90.ci_lower + res_90.ci_upper), res_90.counterfactual_target, atol=1e-12
    )
    width_90 = res_90.ci_upper - res_90.ci_lower
    width_95 = res_95.ci_upper - res_95.ci_lower
    assert np.all(width_95 > width_90)


def test_mckay_wolf_single_shock_1d_inputs() -> None:
    """1D policy_target_irfs and policy_instrument_irfs work seamlessly for n_shocks=1."""
    H = 5
    T_h = H + 1
    pi_base = np.linspace(1.0, 0.2, T_h)
    i_base = np.linspace(0.5, 0.1, T_h)
    theta_pi = -0.4 * (0.8 ** np.arange(T_h))
    theta_i = 0.8 ** np.arange(T_h)

    res = mckay_wolf_counterfactual(
        baseline_target=pi_base,
        baseline_instrument=i_base,
        policy_target_irfs=theta_pi,
        policy_instrument_irfs=theta_i,
        rule_target_weights=1.5,
        target_irf_se=np.full(T_h, 0.02),
    )

    assert res.shock_weights.shape == (1,)
    assert res.counterfactual_target.shape == (T_h,)
    assert res.ci_lower is not None and res.ci_upper is not None


def test_mckay_wolf_presentation_standard_methods() -> None:
    """Verify all 7 Result-Object Presentation Standard 2.0+ methods on PolicyCounterfactualResult."""
    H = 6
    T_h = H + 1
    M_pi_i, M_pi_u, Gamma_base = _solve_structural_nk_single_target(H=H, phi_pi=1.5)
    u_shock = 0.75 ** np.arange(T_h)
    pi_base = Gamma_base @ (M_pi_u @ u_shock)
    i_base = 1.5 * pi_base
    theta_pi = Gamma_base @ M_pi_i
    theta_i = Gamma_base

    res = mckay_wolf_counterfactual(
        baseline_target=pi_base,
        baseline_instrument=i_base,
        policy_target_irfs=theta_pi,
        policy_instrument_irfs=theta_i,
        rule_target_weights=2.2,
        target_irf_se=np.full_like(theta_pi, 0.03),
    )

    df = res.to_dataframe()
    df_frame = res.to_frame()
    df_sum = res.summary()
    md = res.to_markdown()
    tex = res.to_latex()
    typ = res.to_typst()

    assert isinstance(df, pd.DataFrame) and len(df) == T_h
    assert "counterfactual_target" in df.columns
    assert "counterfactual_instrument" in df.columns
    pd.testing.assert_frame_equal(df, df_frame)
    assert isinstance(df_sum, pd.DataFrame) and "rule_rmse" in df_sum.index
    assert isinstance(md, str) and "counterfactual_target" in md
    assert isinstance(tex, str) and "\\begin{tabular}" in tex
    assert isinstance(typ, str) and "#table(" in typ

    fig = res.plot(show=False)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_mckay_wolf_multi_target_presentation_and_custom_ax() -> None:
    """Verify DataFrame columns and plotting with multi-target (H+1, 2) result and custom Axes."""
    H = 4
    T_h = H + 1
    y_base = np.ones((T_h, 2))
    i_base = np.ones(T_h)
    theta_y = np.zeros((T_h, 2, T_h))
    theta_y[:, 0, :] = -0.3 * np.eye(T_h)
    theta_y[:, 1, :] = -0.2 * np.eye(T_h)
    theta_i = np.eye(T_h)

    res = mckay_wolf_counterfactual(
        baseline_target=y_base,
        baseline_instrument=i_base,
        policy_target_irfs=theta_y,
        policy_instrument_irfs=theta_i,
        rule_target_weights=[1.5, 0.5],
        target_irf_se=np.full_like(theta_y, 0.01),
    )

    df = res.to_dataframe()
    assert "counterfactual_target_0" in df.columns
    assert "counterfactual_target_1" in df.columns

    fig, axes = plt.subplots(1, 2)
    out_fig = res.plot(ax=axes, show=False)
    assert out_fig is fig
    plt.close(fig)


def test_mckay_wolf_validation_errors() -> None:
    """Verify informative ValueErrors on invalid inputs to mckay_wolf_counterfactual."""
    y = np.ones(5)
    i = np.ones(5)
    theta_y = np.eye(5)
    theta_i = np.eye(5)

    with pytest.raises(ValueError, match="ridge"):
        mckay_wolf_counterfactual(y, i, theta_y, theta_i, ridge=-0.1)

    with pytest.raises(ValueError, match="alpha"):
        mckay_wolf_counterfactual(y, i, theta_y, theta_i, alpha=1.2)

    with pytest.raises(ValueError, match="baseline_target length"):
        mckay_wolf_counterfactual(np.ones(4), i, theta_y, theta_i)

    with pytest.raises(ValueError, match="policy_instrument_irfs"):
        mckay_wolf_counterfactual(y, i, theta_y, np.eye(4))

    with pytest.raises(ValueError, match="target_irf_se"):
        mckay_wolf_counterfactual(y, i, theta_y, theta_i, target_irf_se=-np.ones((5, 5)))


# ===========================================================================
# Part 2: Barnichon & Mesters (2020, 2023) Optimal Policy Perturbation Tests
# ===========================================================================
def test_barnichon_mesters_divine_coincidence_exact_zero_opp() -> None:
    """When target gaps satisfy the optimal targeting rule pi_h + (kappa/lambda)*x_h = 0, OPP == 0 and delta* == 0."""
    H = 12
    T_h = H + 1
    kappa = 0.20
    lam_x = 0.50

    # A contractionary monetary shock lowers output gap by -1 unit and inflation by -kappa units
    r_x = -(0.8 ** np.arange(T_h))
    r_pi = kappa * r_x
    policy_irfs = np.column_stack([r_pi, r_x])

    # Under optimal commitment / targeting rule: w_pi * r_pi * pi + w_x * r_x * x = 0
    # With w_pi = 1.0, w_x = lam_x, kappa * pi_h + lam_x * x_h = 0  =>  pi_h = -(lam_x / kappa) x_h
    x_gap = -0.6 * (0.75 ** np.arange(T_h))
    pi_gap = -(lam_x / kappa) * x_gap
    target_gaps = np.column_stack([pi_gap, x_gap])

    res = barnichon_mesters_opp(
        target_gaps=target_gaps,
        policy_irfs=policy_irfs,
        target_weights=[1.0, lam_x],
        beta=0.99,
    )

    assert isinstance(res, OptimalPolicyPerturbationResult)
    assert abs(res.opp_gradient) < 1e-14
    assert abs(res.optimal_adjustment) < 1e-14
    assert abs(res.opp_tstat) < 1e-14
    assert np.isclose(res.opp_pvalue, 1.0, atol=1e-12)
    assert abs(res.ar_stat) < 1e-14
    assert np.isclose(res.ar_pvalue, 1.0, atol=1e-12)
    assert np.isclose(res.loss_baseline, res.loss_adjusted, atol=1e-14)
    np.testing.assert_allclose(res.target_paths_optimal, res.target_paths_baseline, atol=1e-14)


def test_barnichon_mesters_excess_inflation_positive_tightening() -> None:
    """When inflation is above target and contraction lowers inflation (R_pi < 0), OPP < 0 and delta* > 0."""
    H = 8
    T_h = H + 1
    # Inflation is +2% above target, output gap is +1% above potential
    pi_gap = 2.0 * (0.85 ** np.arange(T_h))
    x_gap = 1.0 * (0.85 ** np.arange(T_h))
    target_gaps = np.column_stack([pi_gap, x_gap])

    # 100 bps tightening lowers inflation by -0.5 * 0.85^h and output gap by -0.25 * 0.85^h
    r_pi = -0.5 * (0.85 ** np.arange(T_h))
    r_x = -0.25 * (0.85 ** np.arange(T_h))
    policy_irfs = np.column_stack([r_pi, r_x])

    res = barnichon_mesters_opp(
        target_gaps=target_gaps,
        policy_irfs=policy_irfs,
        target_weights=[1.0, 1.0],
        beta=0.99,
    )

    # Analytical check: y_{h,k} = -4.0 * R_{h,k}  =>  delta* = +4.0 eliminates 100% of the gap!
    assert res.opp_gradient < 0.0
    assert np.isclose(res.optimal_adjustment, 4.0, atol=1e-12)
    assert res.loss_adjusted < 1e-14
    assert res.loss_baseline > 0.0
    np.testing.assert_allclose(res.target_paths_optimal, np.zeros_like(target_gaps), atol=1e-12)


def test_barnichon_mesters_orthogonality_and_exact_loss_reduction() -> None:
    """Optimal target paths are strictly orthogonal to policy_irfs under the (beta, w)-inner product."""
    rng = np.random.default_rng(42)
    H = 15
    T_h = H + 1
    K = 3
    target_gaps = rng.normal(loc=0.5, scale=1.0, size=(T_h, K))
    policy_irfs = rng.normal(loc=-0.3, scale=0.4, size=(T_h, K))
    weights = np.array([1.0, 0.5, 0.25])
    beta = 0.96

    res = barnichon_mesters_opp(
        target_gaps=target_gaps,
        policy_irfs=policy_irfs,
        target_weights=weights,
        beta=beta,
    )

    discount = beta ** np.arange(T_h)
    w_hk = discount[:, None] * weights[None, :]

    # 1. Orthogonality of optimal target paths to policy IRFs
    inner_prod = float(np.sum(w_hk * policy_irfs * res.target_paths_optimal))
    assert abs(inner_prod) < 1e-12

    # 2. Strict quadratic loss identity: L(0) - L(delta*) == 0.5 * OPP^2 / ||R||_{beta, w}^2
    r_norm_sq = float(np.sum(w_hk * (policy_irfs ** 2)))
    expected_reduction = 0.5 * (res.opp_gradient ** 2) / r_norm_sq
    assert res.loss_adjusted <= res.loss_baseline
    assert np.isclose(res.loss_baseline - res.loss_adjusted, expected_reduction, atol=1e-12)


def test_barnichon_mesters_1d_single_target() -> None:
    """1D (H+1,) target_gaps and policy_irfs work cleanly and return 1D target paths."""
    H = 6
    T_h = H + 1
    y_gap = np.linspace(1.5, 0.3, T_h)
    r_irf = -np.linspace(0.5, 0.1, T_h)

    res = barnichon_mesters_opp(target_gaps=y_gap, policy_irfs=r_irf, beta=0.99)

    assert res.target_paths_baseline.ndim == 1
    assert res.target_paths_optimal.ndim == 1
    assert np.isclose(res.optimal_adjustment, 3.0, atol=1e-12)
    assert res.loss_adjusted < 1e-14


def test_barnichon_mesters_custom_covariances_and_ar_stat() -> None:
    """Verify exact delta-method SE and Anderson-Rubin statistic under custom irf_cov and target_cov."""
    H = 4
    T_h = H + 1
    K = 2
    N = T_h * K

    target_gaps = np.ones((T_h, K)) * 1.2
    policy_irfs = np.ones((T_h, K)) * (-0.4)
    irf_cov = 0.04 * np.eye(N)
    target_cov = 0.01 * np.eye(N)
    beta = 0.95
    w_k = np.array([1.0, 0.5])

    res = barnichon_mesters_opp(
        target_gaps=target_gaps,
        policy_irfs=policy_irfs,
        target_weights=w_k,
        beta=beta,
        irf_cov=irf_cov,
        target_cov=target_cov,
    )

    d_vec = ((beta ** np.arange(T_h))[:, None] * w_k[None, :]).ravel()
    a_y = d_vec * target_gaps.ravel()
    a_r = d_vec * policy_irfs.ravel()
    expected_var_delta = float(a_y @ irf_cov @ a_y + a_r @ target_cov @ a_r)
    expected_se = np.sqrt(expected_var_delta)

    assert np.isclose(res.opp_se, expected_se, atol=1e-12)
    assert np.isclose(res.opp_tstat, res.opp_gradient / expected_se, atol=1e-12)
    assert 0.0 <= res.opp_pvalue <= 1.0
    assert res.ar_stat > 0.0
    assert 0.0 <= res.ar_pvalue <= 1.0


def test_barnichon_mesters_diagonal_2d_covariance_input() -> None:
    """Passing (H+1, K) diagonal variance arrays matches passing a full diagonal covariance matrix."""
    H = 5
    T_h = H + 1
    K = 2
    target_gaps = np.ones((T_h, K))
    policy_irfs = -0.5 * np.ones((T_h, K))
    diag_var = np.full((T_h, K), 0.02)

    res_2d = barnichon_mesters_opp(
        target_gaps=target_gaps,
        policy_irfs=policy_irfs,
        irf_cov=diag_var,
    )
    res_full = barnichon_mesters_opp(
        target_gaps=target_gaps,
        policy_irfs=policy_irfs,
        irf_cov=np.diag(diag_var.ravel()),
    )

    assert np.isclose(res_2d.opp_se, res_full.opp_se, atol=1e-14)
    assert np.isclose(res_2d.ar_stat, res_full.ar_stat, atol=1e-14)


def test_barnichon_mesters_presentation_standard_methods() -> None:
    """Verify all 7 Result-Object Presentation Standard 2.0+ methods on OptimalPolicyPerturbationResult."""
    H = 6
    T_h = H + 1
    target_gaps = np.column_stack([np.linspace(1.0, 0.2, T_h), np.linspace(0.5, -0.1, T_h)])
    policy_irfs = np.column_stack([-0.4 * (0.85 ** np.arange(T_h)), -0.3 * (0.8 ** np.arange(T_h))])

    res = barnichon_mesters_opp(
        target_gaps=target_gaps,
        policy_irfs=policy_irfs,
        target_weights=[1.0, 0.5],
        beta=0.99,
    )

    df = res.to_dataframe()
    df_frame = res.to_frame()
    df_sum = res.summary()
    md = res.to_markdown()
    tex = res.to_latex()
    typ = res.to_typst()

    assert isinstance(df, pd.DataFrame) and len(df) == T_h
    assert "target_0_baseline" in df.columns
    assert "target_0_optimal" in df.columns
    pd.testing.assert_frame_equal(df, df_frame)
    assert isinstance(df_sum, pd.DataFrame) and "opp_gradient" in df_sum.index
    assert isinstance(md, str) and "opp_gradient" in md
    assert isinstance(tex, str) and "\\begin{tabular}" in tex
    assert isinstance(typ, str) and "#table(" in typ

    fig = res.plot(show=False)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_barnichon_mesters_1d_plot_and_custom_ax() -> None:
    """Verify 1D target plotting and custom Axes support on OptimalPolicyPerturbationResult."""
    y_gap = np.array([1.0, 0.8, 0.5, 0.2])
    r_irf = np.array([-0.3, -0.4, -0.25, -0.1])
    res = barnichon_mesters_opp(target_gaps=y_gap, policy_irfs=r_irf)

    df = res.to_dataframe()
    assert "target_baseline" in df.columns and "target_optimal" in df.columns

    fig, ax = plt.subplots()
    out_fig = res.plot(ax=ax, show=False)
    assert out_fig is fig
    plt.close(fig)


def test_barnichon_mesters_validation_errors() -> None:
    """Verify informative ValueErrors on invalid inputs to barnichon_mesters_opp."""
    y = np.ones((5, 2))
    r = -0.5 * np.ones((5, 2))

    with pytest.raises(ValueError, match="beta"):
        barnichon_mesters_opp(y, r, beta=0.0)

    with pytest.raises(ValueError, match="beta"):
        barnichon_mesters_opp(y, r, beta=1.05)

    with pytest.raises(ValueError, match="policy_irfs shape"):
        barnichon_mesters_opp(y, np.ones((4, 2)))

    with pytest.raises(ValueError, match="target_weights"):
        barnichon_mesters_opp(y, r, target_weights=[1.0, -0.5])

    with pytest.raises(ValueError, match="zero weighted energy"):
        barnichon_mesters_opp(y, np.zeros((5, 2)))


def test_mckay_wolf_custom_3d_and_flattened_rule_target_matrix() -> None:
    """Verify 3D (n_eq, n_y, H+1) and 2D (n_eq, (H+1)*n_y) rule_target_matrix formats."""
    H = 3
    T_h = H + 1
    n_y = 2
    y_base = np.ones((T_h, n_y))
    i_base = np.ones(T_h)
    theta_i = np.eye(T_h)
    theta_y = np.zeros((T_h, n_y, T_h))
    theta_y[:, 0, :] = -0.5 * np.eye(T_h)
    theta_y[:, 1, :] = -0.25 * np.eye(T_h)

    A_y_3d = np.zeros((T_h, n_y, T_h))
    A_y_3d[:, 0, :] = -1.5 * np.eye(T_h)
    A_y_3d[:, 1, :] = -0.5 * np.eye(T_h)

    res_3d = mckay_wolf_counterfactual(
        baseline_target=y_base,
        baseline_instrument=i_base,
        policy_target_irfs=theta_y,
        policy_instrument_irfs=theta_i,
        rule_target_matrix=A_y_3d,
    )
    res_weights = mckay_wolf_counterfactual(
        baseline_target=y_base,
        baseline_instrument=i_base,
        policy_target_irfs=theta_y,
        policy_instrument_irfs=theta_i,
        rule_target_weights=[1.5, 0.5],
    )

    np.testing.assert_allclose(
        res_3d.counterfactual_target, res_weights.counterfactual_target, atol=1e-12
    )
    np.testing.assert_allclose(
        res_3d.counterfactual_instrument, res_weights.counterfactual_instrument, atol=1e-12
    )


def test_mckay_wolf_horizon_varying_target_weights() -> None:
    """Verify horizon-varying Taylor rule weights phi_h of length H+1 for 1D target."""
    H = 5
    T_h = H + 1
    M_pi_i, M_pi_u, Gamma_base = _solve_structural_nk_single_target(H=H, phi_pi=1.5)
    u_shock = 0.8 ** np.arange(T_h)
    pi_base = Gamma_base @ (M_pi_u @ u_shock)
    i_base = 1.5 * pi_base
    theta_pi = Gamma_base @ M_pi_i
    theta_i = Gamma_base

    phi_h = np.linspace(1.2, 2.4, T_h)
    res = mckay_wolf_counterfactual(
        baseline_target=pi_base,
        baseline_instrument=i_base,
        policy_target_irfs=theta_pi,
        policy_instrument_irfs=theta_i,
        rule_target_weights=phi_h,
    )

    assert res.rule_rmse < 1e-12
    np.testing.assert_allclose(
        res.counterfactual_instrument, phi_h * res.counterfactual_target, atol=1e-12
    )


def test_barnichon_mesters_shared_horizon_covariance_kron() -> None:
    """Verify (H+1, H+1) horizon covariance matrix is expanded via Kronecker product when K > 1."""
    H = 3
    T_h = H + 1
    K = 2
    y = np.ones((T_h, K))
    r = -0.4 * np.ones((T_h, K))
    cov_h = 0.05 * np.eye(T_h)

    res_kron = barnichon_mesters_opp(y, r, irf_cov=cov_h)
    res_full = barnichon_mesters_opp(y, r, irf_cov=np.kron(cov_h, np.eye(K)))

    assert np.isclose(res_kron.opp_se, res_full.opp_se, atol=1e-14)
    assert np.isclose(res_kron.ar_stat, res_full.ar_stat, atol=1e-14)


def test_mckay_wolf_same_policy_rule_zero_weights() -> None:
    """When counterfactual Taylor rule matches baseline rule, w* == 0 and counterfactual == baseline."""
    H = 6
    T_h = H + 1
    phi_base = 1.5
    M_pi_i, M_pi_u, Gamma_base = _solve_structural_nk_single_target(H=H, phi_pi=phi_base)
    u_shock = 0.75 ** np.arange(T_h)
    pi_base = Gamma_base @ (M_pi_u @ u_shock)
    i_base = phi_base * pi_base
    theta_pi = Gamma_base @ M_pi_i
    theta_i = Gamma_base

    res = mckay_wolf_counterfactual(
        baseline_target=pi_base,
        baseline_instrument=i_base,
        policy_target_irfs=theta_pi,
        policy_instrument_irfs=theta_i,
        rule_target_weights=phi_base,
    )

    np.testing.assert_allclose(res.shock_weights, np.zeros(T_h), atol=1e-14)
    np.testing.assert_allclose(res.counterfactual_target, pi_base, atol=1e-14)
    np.testing.assert_allclose(res.counterfactual_instrument, i_base, atol=1e-14)


def test_mckay_wolf_2d_horizon_by_target_weights() -> None:
    """Verify (H+1, n_y) horizon-by-target weight array in mckay_wolf_counterfactual."""
    H = 4
    T_h = H + 1
    n_y = 2
    y_base = np.ones((T_h, n_y))
    i_base = np.ones(T_h)
    theta_i = np.eye(T_h)
    theta_y = np.zeros((T_h, n_y, T_h))
    theta_y[:, 0, :] = -0.4 * np.eye(T_h)
    theta_y[:, 1, :] = -0.2 * np.eye(T_h)

    w_2d = np.column_stack([np.linspace(1.2, 2.0, T_h), np.linspace(0.3, 0.7, T_h)])
    res = mckay_wolf_counterfactual(
        baseline_target=y_base,
        baseline_instrument=i_base,
        policy_target_irfs=theta_y,
        policy_instrument_irfs=theta_i,
        rule_target_weights=w_2d,
    )

    expected_inst = np.sum(w_2d * res.counterfactual_target, axis=1)
    np.testing.assert_allclose(res.counterfactual_instrument, expected_inst, atol=1e-12)
    assert res.rule_rmse < 1e-12


def test_mckay_wolf_2d_flattened_rule_target_matrix() -> None:
    """Verify 2D (n_eq, (H+1)*n_y) flattened rule_target_matrix format."""
    H = 3
    T_h = H + 1
    n_y = 2
    y_base = np.ones((T_h, n_y))
    i_base = np.ones(T_h)
    theta_i = np.eye(T_h)
    theta_y = np.zeros((T_h, n_y, T_h))
    theta_y[:, 0, :] = -0.5 * np.eye(T_h)
    theta_y[:, 1, :] = -0.25 * np.eye(T_h)

    # Flattened row-major order over (T_h, n_y): index 2*h is target 0, 2*h + 1 is target 1
    A_y_flat = np.zeros((T_h, T_h * n_y))
    for h in range(T_h):
        A_y_flat[h, 2 * h] = -1.5
        A_y_flat[h, 2 * h + 1] = -0.5

    res_flat = mckay_wolf_counterfactual(
        baseline_target=y_base,
        baseline_instrument=i_base,
        policy_target_irfs=theta_y,
        policy_instrument_irfs=theta_i,
        rule_target_matrix=A_y_flat,
    )
    res_weights = mckay_wolf_counterfactual(
        baseline_target=y_base,
        baseline_instrument=i_base,
        policy_target_irfs=theta_y,
        policy_instrument_irfs=theta_i,
        rule_target_weights=[1.5, 0.5],
    )

    np.testing.assert_allclose(
        res_flat.counterfactual_target, res_weights.counterfactual_target, atol=1e-12
    )


def test_mckay_wolf_baseline_shape_se() -> None:
    """Verify target_irf_se with shape (H+1,) when n_shocks > 1."""
    H = 4
    T_h = H + 1
    y_base = np.ones(T_h)
    i_base = np.ones(T_h)
    theta_y = -0.5 * np.eye(T_h)[:, :2]
    theta_i = np.eye(T_h)[:, :2]

    res = mckay_wolf_counterfactual(
        baseline_target=y_base,
        baseline_instrument=i_base,
        policy_target_irfs=theta_y,
        policy_instrument_irfs=theta_i,
        target_irf_se=np.full(T_h, 0.05),
    )

    assert res.ci_lower is not None and res.ci_upper is not None
    assert res.ci_lower.shape == (T_h,)


def test_barnichon_mesters_zero_target_gaps() -> None:
    """When target gaps are identically zero, OPP == 0, delta* == 0, and statistics are finite."""
    H = 5
    T_h = H + 1
    y = np.zeros((T_h, 2))
    r = -0.4 * np.ones((T_h, 2))

    res = barnichon_mesters_opp(y, r)

    assert res.opp_gradient == 0.0
    assert res.optimal_adjustment == 0.0
    assert res.loss_baseline == 0.0
    assert res.loss_adjusted == 0.0
    assert np.isfinite(res.opp_se) and res.opp_se > 0.0
    assert res.opp_tstat == 0.0
    assert np.isclose(res.opp_pvalue, 1.0)
    assert res.ar_stat == 0.0
    assert np.isclose(res.ar_pvalue, 1.0)


def test_barnichon_mesters_excess_deflation_negative_easing() -> None:
    """When inflation is below target (y < 0) and tightening lowers inflation (R < 0), OPP > 0 and delta* < 0."""
    H = 6
    T_h = H + 1
    y_gap = -1.5 * (0.8 ** np.arange(T_h))
    r_irf = -0.5 * (0.8 ** np.arange(T_h))

    res = barnichon_mesters_opp(y_gap, r_irf)

    assert res.opp_gradient > 0.0
    assert np.isclose(res.optimal_adjustment, -3.0, atol=1e-12)
    assert res.loss_adjusted < 1e-14


def test_barnichon_mesters_discount_factor_sensitivity() -> None:
    """Discount factor beta < 1 shifts optimal adjustment toward near-term horizons relative to beta = 1."""
    # Horizon 0 needs delta = 1.0, Horizon 10 needs delta = 5.0
    H = 10
    T_h = H + 1
    r_irf = -np.ones(T_h)
    y_gap = np.linspace(1.0, 5.0, T_h)

    res_undiscounted = barnichon_mesters_opp(y_gap, r_irf, beta=1.0)
    res_discounted = barnichon_mesters_opp(y_gap, r_irf, beta=0.80)

    assert np.isclose(res_undiscounted.optimal_adjustment, 3.0, atol=1e-12)
    assert res_discounted.optimal_adjustment < res_undiscounted.optimal_adjustment

