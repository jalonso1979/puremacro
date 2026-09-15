"""Tests for canonical implicit upwind HJB solver, adjoint KFE, and continuous Aiyagari GE."""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np
import pytest

from puremacro.vfi import (
    AiyagariContinuousHJBResult,
    HJBSolution,
    solve_aiyagari_continuous_hjb,
    solve_hjb_achdou,
    solve_kfe_achdou,
)


def test_hjb_implicit_convergence_and_speed():
    """Verify implicit upwind HJB solver converges in <= 20 iterations to tol 1e-8."""
    sol = solve_hjb_achdou(
        r_rate=0.03,
        w_rate=1.0,
        rho_val=0.05,
        gamma_r=2.0,
        Na=50,
        tol=1e-8,
        max_iter=100,
    )
    assert sol.converged
    assert sol.n_iter <= 20
    assert sol.V.shape == (50, 2)
    assert sol.c_policy.shape == (50, 2)
    assert sol.s_drift.shape == (50, 2)
    # Borrowing constraint: at a=0, drift cannot be negative into the infeasible region
    assert np.all(sol.s_drift[0, :] >= -1e-12)
    # Monotonicity of value function in wealth: V(a) is strictly increasing
    assert np.all(np.diff(sol.V[:, 0]) > 0)
    assert np.all(np.diff(sol.V[:, 1]) > 0)
    # Higher productivity yields higher value
    assert np.all(sol.V[:, 1] > sol.V[:, 0])


def test_hjb_solution_mapping_protocol():
    """Verify HJBSolution satisfies collections.abc.Mapping contract."""
    sol = solve_hjb_achdou(Na=30, max_iter=30)
    assert isinstance(sol, Mapping)
    assert len(sol) == len(sol.keys())
    assert "V" in sol
    assert "c_policy" in sol
    assert "s_drift" in sol
    assert "n_iter" in sol
    assert "g_dist" in sol
    assert "mass_residual" in sol

    # Subscripting
    assert sol["n_iter"] == sol.n_iter
    np.testing.assert_array_equal(sol["V"], sol.V)

    # Conversion to dict
    d = dict(sol)
    assert isinstance(d, dict)
    assert d["converged"] is True

    # Iteration
    keys_list = list(sol)
    assert "V" in keys_list
    assert len(keys_list) == len(sol)


def test_hjb_presentation_methods():
    """Verify HJBSolution presentation methods: summary, to_frame, to_markdown, to_latex, to_typst, plot."""
    sol = solve_hjb_achdou(Na=30, max_iter=30)

    summary_df = sol.summary()
    assert not summary_df.empty
    assert "Asset Grid Points (Na)" in summary_df.index

    frame_df = sol.to_frame()
    assert len(frame_df) == 30 * 2
    assert "asset_a" in frame_df.columns
    assert "consumption_c" in frame_df.columns

    md_str = sol.to_markdown()
    assert "Asset Grid Points (Na)" in md_str

    latex_str = sol.to_latex()
    assert "tabular" in latex_str

    typst_str = sol.to_typst()
    assert "table(" in typst_str

    # Headless plot
    fig = sol.plot(show=False)
    assert fig is not None


def test_kfe_strict_mass_conservation():
    """Verify adjoint continuous-time KFE achieves |sum g_i Delta a_i - 1.0| <= 10^-12 and g >= 0."""
    sol = solve_hjb_achdou(Na=80, tol=1e-8)
    assert sol.g_dist is not None
    assert sol.A_generator is not None

    g = sol.g_dist
    assert g.shape == (80, 2)
    assert np.all(g >= 0.0)

    da = sol.a_grid[1] - sol.a_grid[0]
    total_mass = float(np.sum(g * da))
    assert abs(total_mass - 1.0) <= 1e-12
    assert sol.mass_residual <= 1e-12

    # Direct call to solve_kfe_achdou
    g_direct, mass_res = solve_kfe_achdou(
        sol.A_generator, sol.a_grid, sol.e_grid, return_residual=True
    )
    assert mass_res <= 1e-12
    np.testing.assert_allclose(g_direct, g, atol=1e-14)


def test_analytical_crra_cake_eating_benchmark():
    """Verify numerical HJB solver converges in < 25 iter and matches analytical CRRA benchmark."""
    # Analytical cake eating / unconstrained consumption benchmark:
    # Under rho=0.05, gamma=2.0, r=0, w=0, exact consumption is c(a) = mu * a where mu = rho / gamma.
    # Value function is V(a) = (mu^(-gamma) / (1 - gamma)) * a^(1 - gamma).
    rho = 0.05
    gamma = 2.0
    mu = rho / gamma
    a_min, a_max = 1.0, 5.0
    Na = 500

    # 1. Execute solve_hjb_achdou directly on cake-eating calibration (r=0, w=0)
    sol = solve_hjb_achdou(
        r_rate=0.0,
        w_rate=0.0,
        rho_val=rho,
        gamma_r=gamma,
        a_min=a_min,
        a_max=a_max,
        Na=Na,
        tol=1e-8,
    )
    assert sol.converged, "HJB solver failed to converge on cake-eating benchmark"
    assert sol.n_iter < 25, f"HJB solver required {sol.n_iter} iterations (expected < 25)"

    c_exact = mu * sol.a_grid
    K = (mu ** (-gamma)) / (1.0 - gamma)
    V_exact = K * (sol.a_grid ** (1.0 - gamma))

    c_num = sol.c_policy[:, 0]
    V_num = sol.V[:, 0]

    # Verify numerical consumption policy matches analytical closed-form solution
    max_abs_c_err = float(np.max(np.abs(c_num - c_exact)))
    max_rel_c_err = float(np.max(np.abs(c_num - c_exact) / c_exact))
    assert max_abs_c_err < 1e-3, f"Max absolute consumption error {max_abs_c_err} exceeds 1e-3"
    assert max_rel_c_err < 0.01, f"Max relative consumption error {max_rel_c_err} exceeds 1%"

    # Verify interior relative error (Reviewer 2 check) is well below 1%
    rel_err_interior = float(np.max(np.abs(c_num[20:180] - c_exact[20:180]) / c_exact[20:180]))
    assert rel_err_interior < 0.01, f"Interior relative error {rel_err_interior} exceeds 1%"

    # Verify numerical value function matches analytical closed-form value function
    max_rel_V_err = float(np.max(np.abs(V_num - V_exact) / np.abs(V_exact)))
    assert max_rel_V_err < 0.01, f"Max relative value function error {max_rel_V_err} exceeds 1%"

    # 2. Verify custom v_prime_boundary parameter produces identical consistent policy
    v_bwd_exact = float((mu * a_min) ** (-gamma))
    v_fwd_exact = float((mu * a_max) ** (-gamma))
    sol_custom = solve_hjb_achdou(
        r_rate=0.0,
        w_rate=0.0,
        rho_val=rho,
        gamma_r=gamma,
        a_min=a_min,
        a_max=a_max,
        Na=Na,
        tol=1e-8,
        v_prime_boundary=(v_bwd_exact, v_fwd_exact),
    )
    assert sol_custom.converged
    assert sol_custom.n_iter < 25
    np.testing.assert_allclose(sol_custom.c_policy, sol.c_policy, atol=1e-12)

    # 3. Verify analytical unconstrained CRRA with positive interest rate r=0.02
    r_pos = 0.02
    mu_pos = (rho - (1.0 - gamma) * r_pos) / gamma
    sol_r = solve_hjb_achdou(
        r_rate=r_pos,
        w_rate=0.0,
        rho_val=rho,
        gamma_r=gamma,
        a_min=a_min,
        a_max=a_max,
        Na=Na,
        tol=1e-8,
    )
    assert sol_r.converged
    assert sol_r.n_iter < 25
    c_exact_r = mu_pos * sol_r.a_grid
    rel_c_r = float(np.max(np.abs(sol_r.c_policy[:, 0] - c_exact_r) / c_exact_r))
    assert rel_c_r < 0.01, f"Positive-r unconstrained relative error {rel_c_r} exceeds 1%"


def test_continuous_aiyagari_general_equilibrium():
    """Verify solve_aiyagari_continuous_hjb converges to market clearing |Ks - Kd| < 10^-4."""
    res = solve_aiyagari_continuous_hjb(
        alpha=0.33,
        delta=0.05,
        rho_val=0.05,
        gamma_r=2.0,
        Na=40,
        a_max=25.0,
        tol_ge=1e-4,
        max_iter_ge=30,
        max_iter_hjb=50,
        tol_hjb=1e-7,
    )
    assert isinstance(res, Mapping)
    assert res.converged
    assert abs(res.excess_capital) < 1e-4
    assert abs(res.Ks_star - res.Kd_star) < 1e-4
    assert res.r_star > 0.005
    assert res.r_star < 0.05
    assert res.w_star > 0
    assert res.K_star > 0
    assert res.L_star > 0

    # Test dictionary indexing
    assert res["r_star"] == res.r_star
    assert res["K_star"] == res.K_star

    # Presentation methods
    assert not res.summary().empty
    assert "tabular" in res.to_latex()
    assert "table(" in res.to_typst()
    assert "Equilibrium Interest Rate" in res.to_markdown()


def test_custom_markov_jump_generator():
    """Verify solve_hjb_achdou supports custom income grids and generator matrix A_z."""
    e_grid = np.array([0.5, 1.0, 1.5])
    Ne = len(e_grid)
    # 3-state Poisson jump generator with zero row sums
    A_z = np.array([
        [-0.2, 0.2, 0.0],
        [0.1, -0.2, 0.1],
        [0.0, 0.2, -0.2],
    ])
    sol = solve_hjb_achdou(
        Na=40,
        e_grid=e_grid,
        A_z=A_z,
        max_iter=50,
        tol=1e-7,
    )
    assert sol.converged
    assert sol.V.shape == (40, 3)
    assert sol.c_policy.shape == (40, 3)
    assert sol.g_dist is not None
    assert sol.g_dist.shape == (40, 3)
    assert sol.mass_residual <= 1e-12
