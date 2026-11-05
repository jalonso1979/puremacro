"""Tests for Sequence-Space Optimal Policy (`ssj_policy`) and Risky Steady State / Precautionary OBC (`risky_steady_state`)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro.dsge.ssj_policy import (
    SSJOSRResult,
    SSJPolicyResult,
    ssj_osr,
    ssj_ramsey,
    ssj_welfare_decomposition,
)
from puremacro.dsge.risky_steady_state import (
    PrecautionaryOBCResult,
    RiskySteadyStateResult,
    solve_precautionary_obc,
    solve_risky_steady_state,
)


# ===========================================================================
# Fixtures & Helpers for Sequence-Space Policy
# ===========================================================================

def _make_nk_ssj(T: int = 40, beta: float = 0.99, kappa: float = 0.1, sigma: float = 2.0) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Construct analytical sequence-space Jacobians J_y, J_pi with respect to nominal rate i_t."""
    # Forward-looking IS curve: y_t = -1/sigma * sum_{s >= t} (i_s - pi_{s+1})
    # Forward-looking NKPC: pi_t = kappa * sum_{s >= t} beta^{s - t} y_s
    M_is = np.triu(np.ones((T, T), dtype=np.float64))
    M_pc = np.zeros((T, T), dtype=np.float64)
    for t in range(T):
        for s in range(t, T):
            M_pc[t, s] = kappa * (beta ** (s - t))

    # Lead shift operator S_{+1}
    S_plus = np.zeros((T, T), dtype=np.float64)
    S_plus[:-1, 1:] = np.eye(T - 1, dtype=np.float64)

    # Coupled linear system:
    # y = -(1/sigma) * M_is @ (i - S_plus @ pi)
    # pi = M_pc @ y
    A_y = np.eye(T, dtype=np.float64) - (1.0 / sigma) * (M_is @ S_plus @ M_pc)
    J_y = -np.linalg.solve(A_y, (1.0 / sigma) * M_is)
    J_pi = M_pc @ J_y

    t_grid = np.arange(T, dtype=np.float64)
    base_pi = 0.015 * (0.8 ** t_grid)
    base_y = -0.010 * (0.75 ** t_grid)
    return {"pi": J_pi, "y": J_y}, {"pi": base_pi, "y": base_y}


# ===========================================================================
# 1. Tests for `ssj_ramsey` & `SSJPolicyResult`
# ===========================================================================

def test_ssj_ramsey_foc_and_loss_reduction() -> None:
    jacs, base = _make_nk_ssj(T=40)
    res = ssj_ramsey(
        jacs,
        base,
        target_weights={"pi": 1.0, "y": 0.5},
        instrument_weight=0.1,
        beta=0.99,
    )
    assert isinstance(res, SSJPolicyResult)
    assert res.horizon == 40
    assert res.loss_optimal < res.loss_baseline
    assert np.isclose(res.welfare_gain, res.loss_baseline - res.loss_optimal, atol=1e-14)

    # Analytical FOC check: || (sum_k w_k J_k' B J_k + lambda_i B) i* + sum_k w_k J_k' B y_bar_k ||_inf < 1e-11
    b_vec = 0.99 ** np.arange(40, dtype=np.float64)
    B = np.diag(b_vec)
    H = 1.0 * (jacs["pi"].T @ B @ jacs["pi"]) + 0.5 * (jacs["y"].T @ B @ jacs["y"]) + 0.1 * B
    rhs = 1.0 * (jacs["pi"].T @ B @ base["pi"]) + 0.5 * (jacs["y"].T @ B @ base["y"])
    foc_grad = H @ res.instrument_path + rhs
    assert float(np.max(np.abs(foc_grad))) < 1e-11
    assert res.metadata["foc_grad_inf_norm"] < 1e-11


def test_ssj_ramsey_zero_target_weights_limit() -> None:
    jacs, base = _make_nk_ssj(T=30)
    res = ssj_ramsey(
        jacs,
        base,
        target_weights={"pi": 0.0, "y": 0.0},
        instrument_weight=0.2,
        beta=0.99,
    )
    assert np.allclose(res.instrument_path, 0.0, atol=1e-14)
    assert np.isclose(res.loss_optimal, 0.0, atol=1e-14)
    assert np.isclose(res.loss_baseline, 0.0, atol=1e-14)


def test_ssj_ramsey_infinite_instrument_cost_limit() -> None:
    jacs, base = _make_nk_ssj(T=30)
    res_high_cost = ssj_ramsey(
        jacs,
        base,
        target_weights={"pi": 1.0, "y": 0.5},
        instrument_weight=1e14,
        beta=0.99,
    )
    assert float(np.max(np.abs(res_high_cost.instrument_path))) < 1e-12
    assert np.isclose(res_high_cost.loss_optimal, res_high_cost.loss_baseline, rtol=1e-9, atol=1e-12)


def test_ssj_ramsey_zero_instrument_cost_exact_target_stabilization() -> None:
    jacs, base = _make_nk_ssj(T=25)
    # Single target with invertible J_pi and zero instrument penalty achieves zero target deviation
    res = ssj_ramsey(
        {"pi": jacs["pi"]},
        {"pi": base["pi"]},
        target_weights={"pi": 1.0},
        instrument_weight=0.0,
        beta=0.99,
    )
    assert float(np.max(np.abs(res.target_paths["pi"]))) < 1e-11
    assert res.loss_optimal < 1e-22


def test_ssj_ramsey_timeless_perspective_path() -> None:
    jacs, base = _make_nk_ssj(T=35)
    res = ssj_ramsey(jacs, base, compute_timeless=True)
    assert res.timeless_instrument_path is not None
    assert res.timeless_instrument_path.shape == (35,)
    # Because J_pi and J_y are upper-triangular (forward-looking anticipation),
    # the timeless path differs from the date-0 Ramsey plan at t=0
    assert abs(res.timeless_instrument_path[0] - res.instrument_path[0]) > 1e-6


def test_ssj_ramsey_distributional_jacobian_and_gradient() -> None:
    T = 30
    jacs, base = _make_nk_ssj(T=T)
    n_groups = 3
    dist_J = np.zeros((n_groups, T, T), dtype=np.float64)
    # Group 0 (high-MPC borrowers) reacts strongly negatively to interest rate hikes;
    # Group 2 (savers) benefits from higher interest income
    dist_J[0] = -0.8 * np.eye(T)
    dist_J[1] = -0.3 * np.eye(T)
    dist_J[2] = +0.2 * np.eye(T)
    w_groups = np.array([0.5, 0.3, 0.2], dtype=np.float64)

    res1 = ssj_ramsey(
        jacs,
        base,
        distributional_jacobian=dist_J,
        welfare_weights=w_groups,
        incorporate_welfare_gradient=False,
    )
    assert res1.distributional_gains is not None
    assert res1.distributional_gains.shape == (n_groups,)

    res2 = ssj_ramsey(
        jacs,
        base,
        distributional_jacobian=dist_J,
        welfare_weights=w_groups,
        incorporate_welfare_gradient=True,
    )
    assert not np.allclose(res1.instrument_path, res2.instrument_path)
    assert res2.metadata["foc_grad_inf_norm"] < 1e-11


def test_ssj_ramsey_array_inputs_and_validation() -> None:
    jacs, base = _make_nk_ssj(T=20)
    J_3d = np.stack([jacs["pi"], jacs["y"]], axis=0)
    base_2d = np.stack([base["pi"], base["y"]], axis=0)
    res = ssj_ramsey(J_3d, base_2d, target_weights=[1.0, 0.5])
    assert res.instrument_path.shape == (20,)
    assert set(res.target_paths.keys()) == {"y0", "y1"}

    with pytest.raises(ValueError, match="beta"):
        ssj_ramsey(jacs, base, beta=1.5)
    with pytest.raises(ValueError, match="instrument_weight"):
        ssj_ramsey(jacs, base, instrument_weight=-0.1)


def test_ssj_policy_result_presentation_standard() -> None:
    jacs, base = _make_nk_ssj(T=20)
    res = ssj_ramsey(jacs, base)

    df_sum = res.summary()
    assert isinstance(df_sum, pd.DataFrame)
    assert "Optimal Loss" in df_sum.index

    df_full = res.to_dataframe()
    df_frame = res.to_frame()
    assert isinstance(df_full, pd.DataFrame)
    assert df_full.equals(df_frame)
    assert "instrument" in df_full.columns
    assert "pi_optimal" in df_full.columns

    md = res.to_markdown()
    tex = res.to_latex()
    typ = res.to_typst()
    assert "|" in md
    assert "\\begin{tabular}" in tex
    assert "#table(" in typ

    fig, ax = plt.subplots()
    out_fig = res.plot(ax=ax, show=False)
    assert out_fig is fig
    assert len(ax.lines) >= 3
    plt.close(fig)

    fig2 = res.plot(show=False)
    assert fig2 is not None
    plt.close(fig2)


# ===========================================================================
# 2. Tests for `ssj_osr` & `SSJOSRResult`
# ===========================================================================

def test_ssj_osr_beats_baseline_and_naive_rule() -> None:
    jacs, base = _make_nk_ssj(T=30)
    res = ssj_osr(
        jacs,
        base,
        target_weights={"pi": 1.0, "y": 0.5},
        instrument_weight=0.1,
        beta=0.99,
        lags=(0,),
    )
    assert isinstance(res, SSJOSRResult)
    assert res.converged
    assert res.loss_optimal < res.loss_baseline
    assert res.loss_optimal <= res.loss_naive + 1e-12
    assert res.welfare_gain > 0.0
    assert "pi" in res.coefficients and "y" in res.coefficients
    assert res.metadata["grad_inf_norm"] < 1e-6


def test_ssj_osr_multi_lag_improvement() -> None:
    jacs, base = _make_nk_ssj(T=25)
    res_lag0 = ssj_osr(jacs, base, lags=(0,))
    res_lag01 = ssj_osr(jacs, base, lags=(0, 1))
    # Adding a lag weakly improves (reduces) the optimal simple rule loss
    assert res_lag01.loss_optimal <= res_lag0.loss_optimal + 1e-9
    assert len(res_lag01.coefficient_vector) == 4


def test_ssj_osr_presentation_standard() -> None:
    jacs, base = _make_nk_ssj(T=20)
    res = ssj_osr(jacs, base)

    assert isinstance(res.summary(), pd.DataFrame)
    assert isinstance(res.to_dataframe(), pd.DataFrame)
    assert res.to_dataframe().equals(res.to_frame())
    assert "|" in res.to_markdown()
    assert "\\begin{tabular}" in res.to_latex()
    assert "#table(" in res.to_typst()

    fig = res.plot(show=False)
    assert fig is not None
    plt.close(fig)


# ===========================================================================
# 3. Tests for `ssj_welfare_decomposition`
# ===========================================================================

def test_ssj_welfare_decomposition_identity_and_equal_weights() -> None:
    T = 25
    n_groups = 4
    rng = np.random.default_rng(42)
    group_J = rng.normal(0.0, 0.2, size=(n_groups, T, T))
    i_path = 0.01 * (0.85 ** np.arange(T))

    # 1. Equal marginal utilities -> redistribution component is identically 0
    df_eq = ssj_welfare_decomposition(
        group_J,
        i_path,
        group_shares=[0.25, 0.25, 0.25, 0.25],
        group_marginal_utilities=[1.5, 1.5, 1.5, 1.5],
        beta=0.99,
    )
    assert np.allclose(df_eq["redistribution"], 0.0, atol=1e-14)
    assert np.allclose(df_eq["efficiency"], df_eq["total_welfare"], atol=1e-14)
    assert abs(df_eq.attrs["redistribution_total"]) < 1e-14

    # 2. Heterogeneous marginal utilities -> exact additive decomposition
    df_het = ssj_welfare_decomposition(
        group_J,
        i_path,
        group_shares=[0.4, 0.3, 0.2, 0.1],
        group_marginal_utilities=[3.0, 1.8, 1.0, 0.5],
        beta=0.99,
    )
    assert np.allclose(
        df_het["efficiency"] + df_het["redistribution"],
        df_het["total_welfare"],
        atol=1e-14,
    )
    assert np.isclose(
        df_het.attrs["efficiency_total"] + df_het.attrs["redistribution_total"],
        df_het.attrs["total_welfare"],
        atol=1e-14,
    )


def test_ssj_welfare_decomposition_positive_redistribution_sign() -> None:
    T = 15
    # Two groups: Group 0 (poor, high marginal utility = 2.0) gains +1,
    # Group 1 (rich, low marginal utility = 0.5) loses -1 (pure zero-sum transfer)
    group_J = np.zeros((2, T, T), dtype=np.float64)
    group_J[0] = np.eye(T)
    group_J[1] = -np.eye(T)
    i_path = np.ones(T, dtype=np.float64) * 0.01

    df = ssj_welfare_decomposition(
        group_J,
        i_path,
        group_shares=[0.5, 0.5],
        group_marginal_utilities=[2.0, 0.5],
        beta=0.99,
    )
    # Aggregate consumption change is 0 -> aggregate efficiency total is 0
    assert abs(df.attrs["efficiency_total"]) < 1e-14
    # Transfer to high-MU group yields strictly positive redistributive welfare gain
    assert df.attrs["redistribution_total"] > 0.0
    assert np.isclose(df.attrs["total_welfare"], df.attrs["redistribution_total"], atol=1e-14)


# ===========================================================================
# 4. Tests for `solve_risky_steady_state` & `RiskySteadyStateResult`
# ===========================================================================

def _crra_bond_residual(
    b_curr: np.ndarray,
    b_next: np.ndarray,
    u_shock: np.ndarray,
    *,
    beta: float = 0.96,
    r: float = 0.041666666666666664,
    gamma: float = 3.0,
    y_bar: float = 1.0,
    psi: float = 0.02,
) -> np.ndarray:
    """CRRA bond Euler equation with portfolio adjustment cost:
    (y_bar + r * b_t - psi * b_t)^{-gamma} - beta * (1 + r) * (y_bar + r * b_{t+1} + u_{t+1})^{-gamma} = 0.
    At deterministic steady state (beta*(1+r) = 1), b_det = 0.
    Because u'''(c) > 0, future consumption/bond volatility raises E_t[u'(c_{t+1})],
    inducing precautionary asset accumulation b_rss > b_det = 0.
    """
    c_t = y_bar + (r - psi) * b_curr
    c_next = y_bar + r * b_next + u_shock
    return (c_t ** (-gamma)) - beta * (1.0 + r) * (c_next ** (-gamma))


def test_risky_steady_state_zero_shock_cov_recovers_deterministic() -> None:
    y_det = np.array([0.0])
    res = solve_risky_steady_state(
        _crra_bond_residual,
        y_det,
        shock_cov=np.array([[0.0]]),
        state_names=("bonds",),
    )
    assert isinstance(res, RiskySteadyStateResult)
    assert res.converged
    assert np.allclose(res.y_rss, y_det, atol=1e-12)
    assert np.allclose(res.precautionary_shift, 0.0, atol=1e-12)
    assert res.residual <= 1e-10


def test_risky_steady_state_positive_precautionary_buffer_and_quadratic_scaling() -> None:
    y_det = np.array([0.0])
    sigma1 = 0.05
    sigma2 = 0.10
    res1 = solve_risky_steady_state(
        _crra_bond_residual,
        y_det,
        shock_cov=np.array([[sigma1 ** 2]]),
        gu=np.array([[1.0]]),
        state_names=("bonds",),
    )
    res2 = solve_risky_steady_state(
        _crra_bond_residual,
        y_det,
        shock_cov=np.array([[sigma2 ** 2]]),
        gu=np.array([[1.0]]),
        state_names=("bonds",),
    )
    assert res1.converged and res2.converged
    # Strictly positive precautionary bond holdings
    assert float(res1.y_rss[0]) > 0.0
    assert float(res1.precautionary_shift[0]) > 0.0
    # Doubling volatility quadruples precautionary shift to second order
    ratio = float(res2.precautionary_shift[0] / res1.precautionary_shift[0])
    assert 3.85 < ratio < 4.15


def test_risky_steady_state_multivariate_and_custom_hessian() -> None:
    # 2-variable system: y0 has quadratic forward curvature, y1 is linear in y0
    def f_res_2d(y_curr: np.ndarray, y_next: np.ndarray, u: np.ndarray) -> np.ndarray:
        r0 = y_curr[0] - 0.5 * (y_next[0] ** 2)
        r1 = y_curr[1] - 0.8 * y_curr[0] - 0.2 * (y_next[1] ** 2)
        return np.array([r0, r1])

    y_det = np.array([0.0, 0.0])
    cov = np.diag([0.04, 0.09])

    # Analytical Hessian tensor d^2 f / d y_{t+1}^2:
    # d^2 f_0 / dy_next_0^2 = -1.0, d^2 f_1 / dy_next_1^2 = -0.4
    H_exact = np.zeros((2, 2, 2), dtype=np.float64)
    H_exact[0, 0, 0] = -1.0
    H_exact[1, 1, 1] = -0.4

    res_ad = solve_risky_steady_state(f_res_2d, y_det, cov, state_names=("k", "c"))
    res_custom = solve_risky_steady_state(
        f_res_2d, y_det, cov, f_y_plus_hessian=H_exact, state_names=("k", "c")
    )
    assert res_ad.converged and res_custom.converged
    assert np.allclose(res_ad.y_rss, res_custom.y_rss, atol=1e-10)
    # Exact nonlinear root of f(y_rss, y_rss, 0) + 0.5 * Psi = 0:
    # y0 - 0.5 * y0^2 - 0.02 = 0, y1 - 0.8 * y0 - 0.2 * y1^2 - 0.018 = 0
    y0_exact = 1.0 - np.sqrt(1.0 - 2.0 * 0.02)
    y1_exact = (1.0 - np.sqrt(1.0 - 0.8 * (0.8 * y0_exact + 0.018))) / 0.4
    assert np.allclose(res_ad.y_rss, np.array([y0_exact, y1_exact]), atol=1e-10)


def test_risky_steady_state_presentation_standard() -> None:
    res = solve_risky_steady_state(
        _crra_bond_residual,
        [0.0],
        [[0.01]],
        state_names=("bonds",),
    )
    df_sum = res.summary()
    assert isinstance(df_sum, pd.DataFrame)
    assert "precautionary_shift" in df_sum.columns
    assert res.to_dataframe().equals(res.to_frame())
    assert "|" in res.to_markdown()
    assert "\\begin{tabular}" in res.to_latex()
    assert "#table(" in res.to_typst()

    fig, ax = plt.subplots()
    out_fig = res.plot(ax=ax, show=False)
    assert out_fig is fig
    plt.close(fig)


# ===========================================================================
# 5. Tests for `solve_precautionary_obc` & `PrecautionaryOBCResult`
# ===========================================================================

def test_precautionary_obc_convergence_and_strict_zlb_enforcement() -> None:
    res = solve_precautionary_obc(
        beta=0.99,
        sigma=2.0,
        kappa_pc=0.1,
        phi_pi=1.5,
        phi_x=0.5,
        i_floor=-0.005,
        rho_r=0.8,
        sigma_r=0.005,
        n_grid=41,
        n_quad=7,
    )
    assert isinstance(res, PrecautionaryOBCResult)
    assert res.converged
    assert res.residual <= 1e-8
    # Strict enforcement of i_policy >= i_floor everywhere on the state grid
    assert np.all(res.i_policy >= res.i_floor - 1e-15)
    assert np.all(res.i_ce >= res.i_floor - 1e-15)
    # ZLB binds on negative demand states and is slack on high positive demand states
    assert bool(res.binding_mask[0]) is True
    assert bool(res.binding_mask[-1]) is False


def test_precautionary_obc_deflationary_and_output_bias_before_zlb() -> None:
    res = solve_precautionary_obc(
        i_floor=-0.005,
        rho_r=0.8,
        sigma_r=0.006,
        n_grid=41,
        n_quad=7,
    )
    # At the deterministic steady state r^n = 0 (middle of symmetric grid),
    # the ZLB does not bind under certainty equivalence (i_ce(0) = 0 > i_floor),
    # yet stochastic risk of hitting the ZLB in future periods induces a strictly
    # negative precautionary deflationary bias and output-gap bias!
    mid_idx = len(res.state_grid) // 2
    assert np.isclose(res.state_grid[mid_idx], 0.0, atol=1e-14)
    assert not bool(res.ce_binding_mask[mid_idx])
    assert res.precautionary_pi_bias[mid_idx] < -1e-5
    assert res.precautionary_x_bias[mid_idx] < -1e-5

    # Across all non-CE-binding points near the ZLB, precautionary bias is non-positive
    near_zlb_slack = (~res.ce_binding_mask) & (res.state_grid <= 0.005)
    assert np.all(res.precautionary_pi_bias[near_zlb_slack] < 0.0)
    assert np.all(res.precautionary_x_bias[near_zlb_slack] < 0.0)


def test_precautionary_obc_zero_volatility_matches_certainty_equivalence() -> None:
    res_zero = solve_precautionary_obc(
        i_floor=-0.005,
        rho_r=0.8,
        sigma_r=0.0,
        n_grid=31,
        n_quad=5,
    )
    assert res_zero.converged
    assert np.allclose(res_zero.x_policy, res_zero.x_ce, atol=1e-14)
    assert np.allclose(res_zero.pi_policy, res_zero.pi_ce, atol=1e-14)
    assert np.allclose(res_zero.i_policy, res_zero.i_ce, atol=1e-14)
    assert np.allclose(res_zero.precautionary_pi_bias, 0.0, atol=1e-14)


def test_precautionary_obc_bias_increases_with_volatility() -> None:
    res_low = solve_precautionary_obc(sigma_r=0.003, n_grid=41)
    res_high = solve_precautionary_obc(sigma_r=0.007, n_grid=41)
    mid = len(res_low.state_grid) // 2
    # Higher shock volatility deepens the precautionary deflationary bias at r^n = 0
    assert res_high.precautionary_pi_bias[mid] < res_low.precautionary_pi_bias[mid] < 0.0
    assert res_high.precautionary_x_bias[mid] < res_low.precautionary_x_bias[mid] < 0.0


def test_precautionary_obc_presentation_standard_and_validation() -> None:
    res = solve_precautionary_obc(n_grid=25, n_quad=5)
    df_sum = res.summary()
    df_grid = res.to_dataframe()
    assert isinstance(df_sum, pd.DataFrame)
    assert isinstance(df_grid, pd.DataFrame)
    assert df_grid.equals(res.to_frame())
    assert "precautionary_pi_bias" in df_grid.columns
    assert "|" in res.to_markdown()
    assert "\\begin{tabular}" in res.to_latex()
    assert "#table(" in res.to_typst()

    fig, ax = plt.subplots()
    out_fig = res.plot(ax=ax, show=False)
    assert out_fig is fig
    assert len(ax.lines) >= 4
    plt.close(fig)

    with pytest.raises(ValueError, match="n_grid"):
        solve_precautionary_obc(n_grid=3)
    with pytest.raises(ValueError, match="sigma_r"):
        solve_precautionary_obc(sigma_r=-0.01)


# ===========================================================================
# 6. Additional Edge-Case, Accuracy & Validation Tests (25+ total tests)
# ===========================================================================

def test_ssj_ramsey_2d_single_target_input() -> None:
    jacs, base = _make_nk_ssj(T=20)
    res = ssj_ramsey(jacs["pi"], base["pi"], instrument_weight=0.05)
    assert res.instrument_path.shape == (20,)
    assert "y0" in res.target_paths
    assert res.loss_optimal < res.loss_baseline


def test_ssj_ramsey_lower_triangular_timeless_coincides() -> None:
    # For a purely backward-looking (lower-triangular) Jacobian with no forward anticipation,
    # the timeless perspective instrument path coincides identically with the date-0 Ramsey plan.
    T = 20
    J_lower = np.tril(0.5 ** (np.arange(T)[:, None] - np.arange(T)[None, :]))
    base_y = 0.02 * (0.8 ** np.arange(T))
    res = ssj_ramsey({"y": J_lower}, {"y": base_y}, compute_timeless=True)
    assert res.timeless_instrument_path is not None
    assert np.allclose(res.timeless_instrument_path, res.instrument_path, atol=1e-12)


def test_ssj_ramsey_discount_factor_sensitivity() -> None:
    jacs, base = _make_nk_ssj(T=25)
    res_high_beta = ssj_ramsey(jacs, base, beta=0.99)
    res_low_beta = ssj_ramsey(jacs, base, beta=0.80)
    assert not np.allclose(res_high_beta.instrument_path, res_low_beta.instrument_path)
    assert res_high_beta.metadata["foc_grad_inf_norm"] < 1e-11
    assert res_low_beta.metadata["foc_grad_inf_norm"] < 1e-11


def test_ssj_osr_single_shock_vector_alignment() -> None:
    jacs, base = _make_nk_ssj(T=20)
    # Passing a single shock path in shock_paths that does not match target keys
    res = ssj_osr(jacs, {"demand_shock": base["y"]}, lags=(0,))
    assert res.converged
    assert res.loss_optimal <= res.loss_baseline


def test_ssj_osr_validation_errors() -> None:
    jacs, base = _make_nk_ssj(T=15)
    with pytest.raises(ValueError, match="beta"):
        ssj_osr(jacs, base, beta=0.0)
    with pytest.raises(ValueError, match="instrument_weight"):
        ssj_osr(jacs, base, instrument_weight=-1.0)
    with pytest.raises(ValueError, match="lags"):
        ssj_osr(jacs, base, lags=())


def test_ssj_welfare_decomposition_validation_errors() -> None:
    T = 10
    J_grp = np.zeros((2, T, T))
    i_vec = np.zeros(T)
    with pytest.raises(ValueError, match="3D array"):
        ssj_welfare_decomposition(np.zeros((T, T)), i_vec)
    with pytest.raises(ValueError, match="group_shares"):
        ssj_welfare_decomposition(J_grp, i_vec, group_shares=[0.5, 0.3, 0.2])
    with pytest.raises(ValueError, match="group_marginal_utilities"):
        ssj_welfare_decomposition(J_grp, i_vec, group_marginal_utilities=[1.0])


def test_risky_steady_state_callable_hessian_and_gu_matrix() -> None:
    def f_res(y_curr: np.ndarray, y_next: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.array([y_curr[0] - 0.25 * (y_next[0] ** 2)])

    res = solve_risky_steady_state(
        f_res,
        [0.0],
        shock_cov=[[0.04]],
        gu=np.array([[2.0]]),
        f_y_plus_hessian=lambda y: np.array([[[-0.5]]]),
        state_names=("x",),
    )
    # Sigma_y = gu @ shock_cov @ gu.T = 4 * 0.04 = 0.16
    # y_rss - 0.25 * y_rss^2 + 0.5 * (-0.5) * 0.16 = 0 => y_rss - 0.25 * y_rss^2 = 0.04
    exact_root = 2.0 * (1.0 - np.sqrt(1.0 - 0.04))
    assert res.converged
    assert np.isclose(res.y_rss[0], exact_root, atol=1e-11)
    assert np.isclose(res.conditional_cov[0, 0], 0.16, atol=1e-12)


def test_risky_steady_state_lead_first_argument_convention() -> None:
    # Function where first argument is named y_next and second is y_curr
    def f_lead_first(y_next: np.ndarray, y_curr: np.ndarray, u: np.ndarray) -> np.ndarray:
        return y_curr - 0.5 * (y_next ** 2)

    res = solve_risky_steady_state(f_lead_first, [0.0], [[0.08]])
    exact_root = 1.0 - np.sqrt(1.0 - 2.0 * 0.04)
    assert res.converged
    assert np.isclose(res.y_rss[0], exact_root, atol=1e-10)


def test_precautionary_obc_earlier_binding_threshold_under_risk() -> None:
    res = solve_precautionary_obc(
        i_floor=-0.005,
        sigma_r=0.008,
        n_grid=51,
        n_quad=9,
    )
    # Because of precautionary deflationary expectations, the ZLB binds at least as often
    # (on at least as many grid points) under stochastic risk as under certainty equivalence
    assert int(np.sum(res.binding_mask)) >= int(np.sum(res.ce_binding_mask))
    assert np.all(res.i_notional[res.binding_mask] <= res.i_floor + 1e-10)


def test_precautionary_obc_quadrature_order_stability() -> None:
    res_q5 = solve_precautionary_obc(n_grid=31, n_quad=5)
    res_q9 = solve_precautionary_obc(n_grid=31, n_quad=9)
    assert res_q5.converged and res_q9.converged
    assert np.allclose(res_q5.pi_policy, res_q9.pi_policy, atol=5e-4)
    assert np.allclose(res_q5.x_policy, res_q9.x_policy, atol=5e-4)

