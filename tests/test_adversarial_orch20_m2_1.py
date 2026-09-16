"""Adversarial Empirical Stress Tests for Milestone 2: Implicit Solvers & Acceleration.

Challenger 1 Suite targeting:
1. Implicit HJB solver across extreme calibrations:
   - High risk aversion (gamma = 5.0)
   - Tiny discount rate (rho = 0.01)
   - Large asset domain (a in [0, 100])
   - Fine grids (Na = 500)
   - Asymmetric jump rates (lambda_1 = 0.5, lambda_2 = 0.05)
   - Verifying convergence in < 25 iterations, monotonicity, concavity, and borrowing constraints.
2. Analytical cake-eating and CRRA unconstrained benchmarks:
   - Deterministic-income CRRA problem with w > 0 and r != rho (closed form
     c = mu (a + w e / r)) under exact Neumann boundary data: the solver has no
     closed-form branch to echo, so the check is falsifiable (first-order grid
     convergence, and the v_prime_boundary parameter is shown to be live).
   - Empirical validation of consumption slope dc/da ~ rho/gamma in cake eating.
3. Adjoint KFE mass conservation & multi-state non-uniform grids:
   - Strongly non-uniform asset grid (power grid with aspect ratio > 50).
   - 4-state asymmetric Markov income process.
   - Exact mass conservation |sum g_i w_i - 1| < 10^-12 and strict non-negativity min g >= 0.
   - Density invariants: weighted marginal sum_i g_ik w_i equals the generator's
     stationary distribution and the node mass g * w solves A^T pi = 0.
4. Continuous Aiyagari General Equilibrium:
   - Strict monotonicity of excess capital demand bracket Z(r) = Ks(r) - Kd(r) on [r_min, rho).
   - Market clearing root clearance |Ks - Kd| < 10^-4.
5. Hardware acceleration fallback & boundary robustness.
6. WASM-aware threading fallback across all bootstrap modules (including lp_block_bootstrap).
"""
from __future__ import annotations

import os
from unittest.mock import patch
import numpy as np
import pytest

from puremacro._backend import backend_available, get_array_namespace, to_numpy
from puremacro.inference.block_bootstrap import block_bootstrap
from puremacro.inference.lp_block_bootstrap import _can_use_threads as lp_can_use_threads
from puremacro.inference.wild_bootstrap import wild_bootstrap
from puremacro.runtime import capabilities, refresh
from puremacro.var.bootstrap import bootstrap_bands
from puremacro.vfi import (
    AiyagariContinuousHJBResult,
    HJBSolution,
    solve_aiyagari_continuous_hjb,
    solve_hjb_achdou,
    solve_kfe_achdou,
)
from puremacro.vfi.hjb_achdou import _stationary_markov_distribution


# ===========================================================================
# 1. Extreme Calibration Stress Tests for Implicit HJB Solver
# ===========================================================================


def test_hjb_implicit_extreme_calibration():
    """Empirically challenge implicit HJB solver under extreme parameters:

    - gamma = 5.0 (high relative risk aversion)
    - rho = 0.01 (patient households)
    - a in [0, 100] (large wealth domain)
    - Na = 500 (fine grid)
    - Asymmetric jump intensities: lambda_1 = 0.5 (fast exit from low), lambda_2 = 0.05 (slow exit from high).
    """
    rho = 0.01
    gamma = 5.0
    r = 0.008  # r < rho
    w = 1.0
    Na = 500
    a_min = 0.0
    a_max = 100.0

    # Asymmetric jump generator:
    # State 1 (e=0.2) jumps to State 2 with intensity 0.5 (expected duration 2.0 periods)
    # State 2 (e=1.0) jumps to State 1 with intensity 0.05 (expected duration 20.0 periods)
    A_z = np.array([[-0.5, 0.5], [0.05, -0.05]])
    e_grid = np.array([0.2, 1.0])

    sol = solve_hjb_achdou(
        r_rate=r,
        w_rate=w,
        rho_val=rho,
        gamma_r=gamma,
        Na=Na,
        a_min=a_min,
        a_max=a_max,
        e_grid=e_grid,
        A_z=A_z,
        max_iter=50,
        tol=1e-8,
        Delta=1e4,
    )

    # 1. Fast convergence in < 25 iterations
    assert sol.converged is True
    assert sol.n_iter < 25, f"Expected n_iter < 25, got {sol.n_iter}"

    # 2. Strict monotonicity in wealth: V'(a) > 0
    dV_0 = np.diff(sol.V[:, 0])
    dV_1 = np.diff(sol.V[:, 1])
    assert np.all(dV_0 > 0), "Value function must be strictly increasing in assets for state 0"
    assert np.all(dV_1 > 0), "Value function must be strictly increasing in assets for state 1"

    # 3. Strict concavity in wealth: V''(a) < 0
    d2V_0 = np.diff(sol.V[:, 0], n=2)
    d2V_1 = np.diff(sol.V[:, 1], n=2)
    assert np.all(d2V_0 < 0), f"Value function must be strictly concave (max d2V_0={np.max(d2V_0)})"
    assert np.all(d2V_1 < 0), f"Value function must be strictly concave (max d2V_1={np.max(d2V_1)})"

    # 4. Strict monotonicity across productivity states: V(a, e_high) > V(a, e_low)
    assert np.all(sol.V[:, 1] > sol.V[:, 0]), "Higher productivity must yield strictly higher value"

    # 5. Borrowing constraint at a_min = 0: drift cannot push below 0
    assert np.all(sol.s_drift[0, :] >= -1e-12), "Savings drift at lower boundary cannot violate borrowing constraint"

    # 6. Consumption policy positivity and monotonicity in assets
    assert np.all(sol.c_policy > 0.0), "Consumption must be strictly positive everywhere"
    assert np.all(np.diff(sol.c_policy[:, 0]) > 0), "Consumption must be strictly increasing in assets (state 0)"
    assert np.all(np.diff(sol.c_policy[:, 1]) > 0), "Consumption must be strictly increasing in assets (state 1)"


# ===========================================================================
# 2. Analytical Benchmarks: Closed-Form CRRA & Cake-Eating Propensity
# ===========================================================================


def test_analytical_crra_unconstrained_benchmark():
    """Verify solve_hjb_achdou against a closed form the solver cannot short-circuit.

    Deterministic income (Ne = 1, no jumps), w > 0 and r != rho. Absent a binding
    constraint the CRRA household consumes a constant fraction of total wealth
    X(a) = a + w e / r:
        c(a) = mu X,   mu = (rho - (1 - gamma) r) / gamma,
        V(a) = mu^(-gamma) X^(1 - gamma) / (1 - gamma),   s(a) = (r - rho) / gamma * X.
    The exact marginal utilities at a_min and a_max are imposed as Neumann data via
    ``v_prime_boundary``. With w > 0 the solver has no closed-form branch to echo
    (its initial guess u(r a + w e) / rho is not the fixed point), so the policy
    must converge to the closed form at first order in the grid spacing.
    """
    rho, gamma, r, w = 0.05, 2.0, 0.03, 1.0
    mu = (rho - (1.0 - gamma) * r) / gamma
    rel_c_err = {}
    for Na in (100, 400):
        a_grid = np.linspace(1.0, 20.0, Na)
        X = a_grid + w / r
        c_exact = mu * X
        V_exact = mu ** (-gamma) * X ** (1.0 - gamma) / (1.0 - gamma)
        s_exact = (r - rho) / gamma * X
        sol = solve_hjb_achdou(
            r_rate=r,
            w_rate=w,
            rho_val=rho,
            gamma_r=gamma,
            a_grid=a_grid,
            e_grid=np.array([1.0]),
            A_z=np.zeros((1, 1)),
            compute_kfe=False,
            v_prime_boundary=((mu * X[0]) ** (-gamma), (mu * X[-1]) ** (-gamma)),
            max_iter=50,
            tol=1e-9,
        )
        assert sol.converged is True
        # The solver genuinely iterates: the initial guess is not the fixed point
        assert 1 < sol.n_iter <= 25

        rel_c_err[Na] = float(np.max(np.abs(sol.c_policy[:, 0] - c_exact) / c_exact))
        rel_V_err = float(np.max(np.abs(sol.V[:, 0] - V_exact) / np.abs(V_exact)))
        assert rel_c_err[Na] < 5e-3, f"Na={Na}: consumption error {rel_c_err[Na]:.2e}"
        assert rel_V_err < 2e-3, f"Na={Na}: value error {rel_V_err:.2e}"
        # r < rho: wealth decumulates everywhere at the closed-form rate
        assert np.all(sol.s_drift[:, 0] < 0.0)
        np.testing.assert_allclose(sol.s_drift[:, 0], s_exact, rtol=2e-2)

    # First-order convergence: quadrupling Na cuts the error by ~4
    assert rel_c_err[400] < 0.35 * rel_c_err[100], rel_c_err

    # The boundary data are live: doubling both marginal utilities makes the household
    # consume less at the constraint (higher v' <=> lower c = (v')^(-1/gamma))
    a_grid = np.linspace(1.0, 20.0, 200)
    X = a_grid + w / r
    common = dict(
        r_rate=r, w_rate=w, rho_val=rho, gamma_r=gamma, a_grid=a_grid,
        e_grid=np.array([1.0]), A_z=np.zeros((1, 1)), compute_kfe=False, tol=1e-9,
    )
    v_lo, v_hi = (mu * X[0]) ** (-gamma), (mu * X[-1]) ** (-gamma)
    sol_exact = solve_hjb_achdou(v_prime_boundary=(v_lo, v_hi), **common)
    sol_doubled = solve_hjb_achdou(v_prime_boundary=(2.0 * v_lo, 2.0 * v_hi), **common)
    assert sol_doubled.converged is True
    assert sol_doubled.c_policy[0, 0] < sol_exact.c_policy[0, 0] - 0.1
    assert float(np.max(np.abs(sol_doubled.c_policy - sol_exact.c_policy))) > 0.1


def test_cake_eating_marginal_propensity():
    """Verify numerical HJB solver recovers the theoretical marginal propensity to consume out of cake:

    In continuous-time cake eating with r = 0, w = 0:
    Theoretical consumption rule: c(a) = (rho / gamma) * a
    Marginal propensity to consume: dc / da = rho / gamma.
    """
    rho = 0.04
    gamma = 2.0
    expected_mpc = rho / gamma  # 0.02

    Na = 200
    a_grid = np.linspace(0.1, 20.0, Na)
    da = a_grid[1] - a_grid[0]

    sol = solve_hjb_achdou(
        r_rate=0.0,
        w_rate=0.0,
        rho_val=rho,
        gamma_r=gamma,
        a_grid=a_grid,
        e_grid=np.array([1.0]),
        A_z=np.zeros((1, 1)),
        compute_kfe=False,
        max_iter=50,
        tol=1e-7,
    )
    assert sol.converged is True
    assert sol.n_iter < 25

    # Interior derivative away from boundary truncation effects (indices 50 to 170)
    c_interior = sol.c_policy[50:170, 0]
    empirical_mpc = np.gradient(c_interior, da)

    # Verify empirical MPC matches theoretical rho / gamma within 3% tolerance
    np.testing.assert_allclose(empirical_mpc, expected_mpc, rtol=0.03)


# ===========================================================================
# 3. Adjoint KFE Solver on Non-Uniform Grids & Multi-State Processes
# ===========================================================================


def test_kfe_mass_conservation_nonuniform_grid_and_multi_state():
    """Stress-test solve_kfe_achdou on highly non-uniform asset grid and 4-state Markov jump process:

    - Grid spacing ratio max(Delta a) / min(Delta a) > 50.
    - 4 asymmetric productivity states.
    - Assert |sum g_i Delta a_i - 1| < 10^-12.
    - Assert min g >= 0.
    - Assert marginal distribution across productivity matches Markov chain stationary distribution.
    """
    Na = 160
    # Power grid: concentrated near a_min = 0, stretching to 50
    u = np.linspace(0.0, 1.0, Na)
    a_grid = 50.0 * (u ** 3.0)
    da = np.diff(a_grid)
    grid_aspect_ratio = da[-1] / da[0]
    assert grid_aspect_ratio > 50, f"Grid aspect ratio should be > 50, got {grid_aspect_ratio:.1f}"

    # 4-state asymmetric productivity process
    e_grid = np.array([0.2, 0.6, 1.2, 2.5])
    Ne = len(e_grid)
    A_z = np.array([
        [-0.40, 0.25, 0.15, 0.00],
        [0.10, -0.45, 0.25, 0.10],
        [0.05, 0.15, -0.40, 0.20],
        [0.00, 0.10, 0.20, -0.30],
    ])
    # Zero row sums check
    np.testing.assert_allclose(np.sum(A_z, axis=1), 0.0, atol=1e-15)

    sol = solve_hjb_achdou(
        r_rate=0.02,
        w_rate=1.0,
        rho_val=0.04,
        gamma_r=2.0,
        a_grid=a_grid,
        e_grid=e_grid,
        A_z=A_z,
        max_iter=60,
        tol=1e-8,
        compute_kfe=True,
    )
    assert sol.converged is True

    # Check solver returned distribution
    g = sol.g_dist
    assert g is not None
    assert g.shape == (Na, Ne)

    # 1. Strict non-negativity
    assert np.min(g) >= 0.0, f"KFE density must be non-negative, got min {np.min(g)}"

    # 2. Strict mass conservation on non-uniform grid (< 10^-12)
    assert sol.mass_residual < 1e-12, f"Mass residual {sol.mass_residual} exceeds 1e-12"

    # 3. Direct call to solve_kfe_achdou
    g_direct, res = solve_kfe_achdou(sol.A_generator, a_grid, e_grid, return_residual=True)
    assert res < 1e-12
    np.testing.assert_allclose(g_direct, g, atol=1e-14)

    # 4. Density invariants on the non-uniform grid. solve_kfe_achdou returns a DENSITY
    # (node mass / cell width), so the weighted marginal sum_i g_{i,k} w_i with the
    # cell-width quadrature weights w_i = (Delta a_{i-1} + Delta a_i) / 2 (end spacings
    # repeated; w == Delta a on a uniform grid) must equal the stationary distribution of
    # A_z on ANY grid, and the node mass pi = g * w must solve the KFE A^T pi = 0 itself,
    # not merely the post-solve normalisation.
    w = 0.5 * (np.insert(da, 0, da[0]) + np.append(da, da[-1]))
    p_z_expected = _stationary_markov_distribution(A_z)
    weighted_marginal = np.sum(g * w[:, None], axis=0)
    np.testing.assert_allclose(weighted_marginal, p_z_expected, atol=1e-12)
    pi_vec = (g * w[:, None]).ravel(order="F")
    assert abs(np.sum(pi_vec) - 1.0) < 1e-12
    assert np.max(np.abs(sol.A_generator.T @ pi_vec)) < 1e-12
    # The unweighted column sum is NOT the marginal on a non-uniform grid: that identity
    # only holds for a node mass, which is what the pre-fix solver wrongly returned as g
    unweighted = np.sum(g, axis=0) / np.sum(g)
    assert np.max(np.abs(unweighted - p_z_expected)) > 1e-2

    # On uniform grid, the continuous integral sum_i g_{i, k} Delta a matches p_z to machine precision
    a_uniform = np.linspace(0.0, 50.0, Na)
    da_u = a_uniform[1] - a_uniform[0]
    sol_u = solve_hjb_achdou(
        r_rate=0.02,
        w_rate=1.0,
        rho_val=0.04,
        gamma_r=2.0,
        a_grid=a_uniform,
        e_grid=e_grid,
        A_z=A_z,
        max_iter=60,
        tol=1e-8,
        compute_kfe=True,
    )
    marginal_u = np.sum(sol_u.g_dist * da_u, axis=0)
    np.testing.assert_allclose(marginal_u, p_z_expected, atol=1e-12)


# ===========================================================================
# 4. Continuous Aiyagari General Equilibrium: Monotonic Bracket & Root Clearance
# ===========================================================================


def test_aiyagari_continuous_excess_capital_monotonic_bracket():
    """Verify excess capital supply function Z(r) = Ks(r) - Kd(r) is strictly monotonic on [r_min, rho).

    Also verifies market clearance |Ks - Kd| < 10^-4.
    """
    alpha = 0.33
    delta = 0.05
    rho = 0.05
    gamma = 2.0
    Na = 50

    # 1. Full GE Solve
    ge_res = solve_aiyagari_continuous_hjb(
        alpha=alpha,
        delta=delta,
        rho_val=rho,
        gamma_r=gamma,
        Na=Na,
        a_max=30.0,
        r_min=0.005,
        r_max=rho - 0.002,
        tol_ge=1e-4,
        max_iter_ge=35,
        max_iter_hjb=50,
    )

    assert isinstance(ge_res, AiyagariContinuousHJBResult)
    assert ge_res.converged is True
    # Clearance requirement: |Ks - Kd| < 1e-4
    assert abs(ge_res.excess_capital) < 1e-4, f"Excess capital {ge_res.excess_capital} exceeds 1e-4"
    assert abs(ge_res.Ks_star - ge_res.Kd_star) < 1e-4
    assert 0.005 < ge_res.r_star < rho
    assert ge_res.w_star > 0.0
    assert ge_res.K_star > 0.0
    assert ge_res.Y_star > 0.0

    # 2. Monotonicity of excess capital supply Z(r)
    # Sample r across [0.010, 0.045]
    r_samples = np.array([0.010, 0.018, 0.026, 0.034, 0.042])
    excess_values = []

    for r_val in r_samples:
        k_over_l = (alpha / (r_val + delta)) ** (1.0 / (1.0 - alpha))
        Kd = float(ge_res.L_star * k_over_l)
        w = float((1.0 - alpha) * (k_over_l ** alpha))
        sol = solve_hjb_achdou(
            r_rate=r_val,
            w_rate=w,
            rho_val=rho,
            gamma_r=gamma,
            Na=Na,
            a_max=30.0,
            max_iter=35,
            tol=1e-7,
            compute_kfe=True,
        )
        assert sol.g_dist is not None
        da = sol.a_grid[1] - sol.a_grid[0]
        Ks = float(np.sum(sol.a_grid[:, None] * sol.g_dist * da))
        excess_values.append(Ks - Kd)

    excess_arr = np.array(excess_values)
    diffs = np.diff(excess_arr)

    # Monotonicity check: excess capital must be strictly increasing in r
    assert np.all(diffs > 0), f"Excess capital function Z(r) must be strictly increasing: {excess_arr}"

    # Sign change check: negative at low r, positive at high r
    assert excess_arr[0] < 0.0, f"Expected Z(r_min) < 0, got {excess_arr[0]}"
    assert excess_arr[-1] > 0.0, f"Expected Z(r_max) > 0, got {excess_arr[-1]}"


# ===========================================================================
# 5. Hardware Acceleration Robustness & Edge Cases
# ===========================================================================


def test_backend_edge_cases():
    """Verify array namespace handling for scalar, 1D, and invalid backend parameters."""
    assert backend_available("numpy") is True

    with pytest.raises(ValueError, match="Unknown backend"):
        backend_available("invalid_accelerator_xyz")

    xp = get_array_namespace("numpy")
    assert xp is np

    # Test conversion of empty array
    empty = np.array([])
    np.testing.assert_array_equal(to_numpy(empty), empty)

    # Test conversion of Python float/int
    assert to_numpy(3.14159) == 3.14159
    assert to_numpy(100) == 100


# ===========================================================================
# 6. WASM Threading Fallback Comprehensive Coverage
# ===========================================================================


def test_lp_block_bootstrap_wasm_fallback():
    """Verify puremacro.inference.lp_block_bootstrap adheres to WASM threading fallback."""
    # When PUREMACRO_THREADS=0, lp_can_use_threads() must return False
    os.environ["PUREMACRO_THREADS"] = "0"
    try:
        refresh()
        assert capabilities().threads is False
        assert lp_can_use_threads() is False
    finally:
        os.environ.pop("PUREMACRO_THREADS", None)
        refresh()
        assert capabilities().threads is True
        assert lp_can_use_threads() is True
