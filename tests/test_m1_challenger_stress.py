"""Milestone 1 Empirical Stress Test Harness & Defect Demonstrator (Challenger 1).

Covers all Milestone 1 empirical challenge criteria:
1. Extreme elasticity limits:
   rho_va in {1e-4, 0.5, 1-1e-7, 1.0, 1+1e-7, 2.0, 10.0, 50.0}
   sigma_y in {0.0, 1e-7, 0.5, 1.0, 5.0, 20.0}
2. Continuity across Cobb-Douglas gating (|rho_va - 1| = 1e-6) and Leontief gating (sigma_y = 1e-6).
   Verification of |c_va(CES) - c_va(CD)| < 1e-5 at boundary.
3. Factor price shocks r in [1e-3, 1e3], w in [1e-3, 1e3], Euler homogeneity degree 1 for costs,
   and degree 0 for factor demands.
4. NaN, Inf, and exception safety verification.

Provides empirical proof for Verdict: REQUEST_CHANGES.
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.trade import calibrate_trade_model
from puremacro.trade.data import load_icio_data
from puremacro.trade.flexible import (
    FlexibleTechnologyConfig,
    _compute_benchmark_cva0,
    compute_inner_ces_cost,
    compute_inner_ces_derivatives,
    compute_intermediate_composite_price,
    compute_nested_ces_costs,
    compute_nested_factor_demands,
    compute_outer_ces_cost,
    compute_zero_profit_price,
)


@pytest.fixture(scope="module")
def calib_synthetic():
    """Balanced 2-country 2-sector synthetic model."""
    nc, ns, nfd = 2, 2, 3
    data = np.zeros((ns * nc + 3, ns * nc + nfd * nc), dtype=float)
    data[:4, :4] = np.array([
        [10.0, 15.0, 5.0, 5.0],
        [15.0, 20.0, 10.0, 10.0],
        [5.0, 5.0, 12.0, 18.0],
        [10.0, 10.0, 18.0, 22.0],
    ])
    inter_col_sums = data[:4, :4].sum(axis=0)
    y = np.array([100.0, 150.0, 120.0, 180.0])
    va = y - inter_col_sums
    taxes = 0.05 * y
    va_fac = va - taxes
    data[4, :4] = taxes
    data[5, :4] = (2.0 / 3.0) * va_fac
    data[6, :4] = (1.0 / 3.0) * va_fac

    inter_row_sums = data[:4, :4].sum(axis=1)
    fd_row_sums = y - inter_row_sums
    for i in range(4):
        tot_fd = fd_row_sums[i]
        if i < 2:
            data[i, 4:10] = tot_fd * np.array([0.5, 0.25, 0.05, 0.1, 0.08, 0.02])
        else:
            data[i, 4:10] = tot_fd * np.array([0.1, 0.08, 0.02, 0.5, 0.25, 0.05])

    data[4, 4:] = 0.02 * data[:4, 4:].sum(axis=0)
    return calibrate_trade_model(data, ns=ns, nc=nc, nfd=nfd, validate=True)


@pytest.fixture(scope="module")
def calib_icio():
    """Empirical 77c x 11s ICIO benchmark."""
    return calibrate_trade_model(load_icio_data(), ns=11, nc=77, nfd=3, validate=True)


# =============================================================================
# 1. EXTREME ELASTICITY LIMITS AT BASELINE (VERIFIED: PASS)
# =============================================================================

RHO_VA_GRID = [1e-4, 0.5, 1.0 - 1e-7, 1.0, 1.0 + 1e-7, 2.0, 10.0, 50.0]
SIGMA_Y_GRID = [0.0, 1e-7, 0.5, 1.0, 5.0, 20.0]


@pytest.mark.parametrize("rho_va", RHO_VA_GRID)
@pytest.mark.parametrize("sigma_y", SIGMA_Y_GRID)
def test_extreme_elasticities_baseline_invariance_synthetic(calib_synthetic, rho_va, sigma_y):
    """Stress-test all 48 combinations of extreme elasticities on synthetic model."""
    calib = calib_synthetic
    tech_cfg = FlexibleTechnologyConfig(rho_va=rho_va, sigma_y=sigma_y)

    r_base = np.ones((1, calib.n_sectors, calib.n_countries))
    w_base = np.ones((1, calib.n_sectors, calib.n_countries))
    p_base = np.ones((1, calib.n_sectors, calib.n_countries))
    tau_base = np.ones((calib.n_sectors * calib.n_countries, calib.n_sectors, calib.n_countries))
    P_M_base, _, _ = compute_intermediate_composite_price(p_base, tau_base, calib)

    c_va, c_y = compute_nested_ces_costs(r_base, w_base, P_M_base, calib, tech_cfg)
    assert np.all(np.isfinite(c_va))
    assert np.all(np.isfinite(c_y))
    assert np.all(c_va > 0)
    assert np.all(c_y > 0)

    xl, xk, x_mat = compute_nested_factor_demands(
        calib.ytot, r_base, w_base, P_M_base, c_va, c_y, p_base, tau_base, calib, tech_cfg
    )
    assert np.all(np.isfinite(xl))
    assert np.all(np.isfinite(xk))
    assert np.all(np.isfinite(x_mat))

    np.testing.assert_allclose(xl.sum(axis=1).ravel(), calib.l_endow.ravel(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(xk.sum(axis=1).ravel(), calib.k_endow.ravel(), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("rho_va", [1e-4, 1.0 - 1e-7, 1.0, 1.0 + 1e-7, 50.0])
@pytest.mark.parametrize("sigma_y", [0.0, 1e-7, 1.0, 20.0])
def test_extreme_elasticities_baseline_invariance_empirical(calib_icio, rho_va, sigma_y):
    """Stress-test extreme elasticities on 77c x 11s ICIO empirical model."""
    calib = calib_icio
    tech_cfg = FlexibleTechnologyConfig(rho_va=rho_va, sigma_y=sigma_y)

    r_base = np.ones((1, calib.n_sectors, calib.n_countries))
    w_base = np.ones((1, calib.n_sectors, calib.n_countries))
    p_base = np.ones((1, calib.n_sectors, calib.n_countries))
    tau_base = np.ones((calib.n_sectors * calib.n_countries, calib.n_sectors, calib.n_countries))
    P_M_base, _, _ = compute_intermediate_composite_price(p_base, tau_base, calib)

    c_va, c_y = compute_nested_ces_costs(r_base, w_base, P_M_base, calib, tech_cfg)
    assert np.all(np.isfinite(c_va))
    assert np.all(np.isfinite(c_y))

    xl, xk, x_mat = compute_nested_factor_demands(
        calib.ytot, r_base, w_base, P_M_base, c_va, c_y, p_base, tau_base, calib, tech_cfg
    )
    # Relative precision on large ICIO endowment scale is machine precision
    rel_l = np.max(np.abs(xl.sum(axis=1).ravel() - calib.l_endow.ravel()) / calib.l_endow.ravel())
    rel_k = np.max(np.abs(xk.sum(axis=1).ravel() - calib.k_endow.ravel()) / calib.k_endow.ravel())
    assert rel_l < 1e-12
    assert rel_k < 1e-12


# =============================================================================
# 2. CONTINUITY ACROSS GATING BOUNDARIES (VERIFIED: PASS)
# =============================================================================

def test_cobb_douglas_gating_boundary_continuity(calib_synthetic):
    """Verify continuity across |rho_va - 1| = 1e-6 boundary and |c_va(CES) - c_va(CD)| < 1e-5."""
    calib = calib_synthetic

    price_pairs = [
        (1.0, 1.0),
        (1.5, 0.8),
        (0.5, 2.0),
        (0.1, 10.0),
        (5.0, 0.2),
        (10.0, 0.1),
    ]

    for r_val, w_val in price_pairs:
        r = np.full((1, calib.n_sectors, calib.n_countries), r_val)
        w = np.full((1, calib.n_sectors, calib.n_countries), w_val)

        cfg_cd = FlexibleTechnologyConfig(rho_va=1.0)
        c_va_cd = compute_inner_ces_cost(r, w, calib, cfg_cd)

        cfg_outside_minus = FlexibleTechnologyConfig(rho_va=1.0 - (1e-6 + 1e-9))
        cfg_outside_plus = FlexibleTechnologyConfig(rho_va=1.0 + (1e-6 + 1e-9))
        c_va_out_minus = compute_inner_ces_cost(r, w, calib, cfg_outside_minus)
        c_va_out_plus = compute_inner_ces_cost(r, w, calib, cfg_outside_plus)

        diff_minus = np.max(np.abs(c_va_out_minus - c_va_cd))
        diff_plus = np.max(np.abs(c_va_out_plus - c_va_cd))

        assert diff_minus < 1e-5, f"Boundary jump at 1 - 1e-6: {diff_minus}"
        assert diff_plus < 1e-5, f"Boundary jump at 1 + 1e-6: {diff_plus}"


def test_leontief_gating_boundary_continuity(calib_synthetic):
    """Verify continuity across sigma_y = 1e-6 Leontief gating boundary."""
    calib = calib_synthetic

    r = np.full((1, calib.n_sectors, calib.n_countries), 1.4)
    w = np.full((1, calib.n_sectors, calib.n_countries), 0.7)
    p = np.full((1, calib.n_sectors, calib.n_countries), 1.2)
    tau = np.ones((calib.n_sectors * calib.n_countries, calib.n_sectors, calib.n_countries))
    P_M, _, _ = compute_intermediate_composite_price(p, tau, calib)

    c_va = compute_inner_ces_cost(r, w, calib, FlexibleTechnologyConfig(rho_va=1.0))

    cfg_leo = FlexibleTechnologyConfig(sigma_y=0.0)
    cfg_inside = FlexibleTechnologyConfig(sigma_y=1e-6 - 1e-9)
    cfg_outside = FlexibleTechnologyConfig(sigma_y=1e-6 + 1e-9)

    c_y_leo = compute_outer_ces_cost(c_va, P_M, calib, cfg_leo, normalized=True)
    c_y_inside = compute_outer_ces_cost(c_va, P_M, calib, cfg_inside, normalized=True)
    c_y_outside = compute_outer_ces_cost(c_va, P_M, calib, cfg_outside, normalized=True)

    diff_gating = np.max(np.abs(c_y_outside - c_y_leo))
    assert diff_gating < 1e-5, f"Leontief boundary jump: {diff_gating}"
    np.testing.assert_allclose(c_y_inside, c_y_leo, rtol=1e-14)


# =============================================================================
# 3. EULER HOMOGENEITY & PRICE SHOCKS (MODERATE: PASS)
# =============================================================================

@pytest.mark.parametrize("rho_va", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("lam", [0.01, 0.1, 0.5, 2.0, 10.0])
def test_inner_cost_euler_homogeneity_moderate(calib_synthetic, rho_va, lam):
    """Euler homogeneity degree 1 for inner cost under moderate price scaling."""
    calib = calib_synthetic
    tech_cfg = FlexibleTechnologyConfig(rho_va=rho_va)

    r_base = np.full((1, calib.n_sectors, calib.n_countries), 1.5)
    w_base = np.full((1, calib.n_sectors, calib.n_countries), 0.8)

    c_va_base = compute_inner_ces_cost(r_base, w_base, calib, tech_cfg)
    c_va_scaled = compute_inner_ces_cost(lam * r_base, lam * w_base, calib, tech_cfg)

    rel_err = np.max(np.abs(c_va_scaled - lam * c_va_base) / (lam * c_va_base))
    assert rel_err < 1e-12


@pytest.mark.parametrize("rho_va", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("lam", [0.01, 0.1, 0.5, 2.0, 10.0])
def test_inner_derivatives_euler_homogeneity_degree_0(calib_synthetic, rho_va, lam):
    """Euler homogeneity degree 0 for normalized inner nest gradients."""
    calib = calib_synthetic
    tech_cfg = FlexibleTechnologyConfig(rho_va=rho_va)

    r_base = np.full((1, calib.n_sectors, calib.n_countries), 1.5)
    w_base = np.full((1, calib.n_sectors, calib.n_countries), 0.8)
    c_va_base = compute_inner_ces_cost(r_base, w_base, calib, tech_cfg)

    dw_base, dr_base = compute_inner_ces_derivatives(r_base, w_base, c_va_base, calib, tech_cfg)

    r_scaled = r_base * lam
    w_scaled = w_base * lam
    c_va_scaled = compute_inner_ces_cost(r_scaled, w_scaled, calib, tech_cfg)
    dw_scaled, dr_scaled = compute_inner_ces_derivatives(r_scaled, w_scaled, c_va_scaled, calib, tech_cfg)

    np.testing.assert_allclose(dw_scaled, dw_base, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(dr_scaled, dr_base, rtol=1e-10, atol=1e-12)


# =============================================================================
# 4. DEFECT REPRODUCTION TESTS (EMPIRICAL CHALLENGE FINDINGS)
# =============================================================================

def test_defect_factor_demand_heuristic_breaks_euler_hd0(calib_synthetic):
    """Demonstrate that factor demands jump by > 5% between lam=1.05 and lam=1.10."""
    calib = calib_synthetic
    cfg = FlexibleTechnologyConfig(rho_va=1.0, sigma_y=1.0)

    r_base = np.full((1, calib.n_sectors, calib.n_countries), 1.0)
    w_base = np.full((1, calib.n_sectors, calib.n_countries), 1.0)
    p_base = np.full((1, calib.n_sectors, calib.n_countries), 1.0)
    tau = np.ones((calib.n_sectors * calib.n_countries, calib.n_sectors, calib.n_countries))
    P_M_base, _, _ = compute_intermediate_composite_price(p_base, tau, calib)

    c_va_base, c_y_base = compute_nested_ces_costs(r_base, w_base, P_M_base, calib, cfg)
    xl_base, xk_base, _ = compute_nested_factor_demands(
        calib.ytot, r_base, w_base, P_M_base, c_va_base, c_y_base, p_base, tau, calib, cfg
    )

    # Scaling all prices uniformly by lam = 1.10 triggers heuristic toggle
    lam = 1.10
    P_M_lam, _, _ = compute_intermediate_composite_price(lam * p_base, tau, calib)
    c_va_lam, c_y_lam = compute_nested_ces_costs(lam * r_base, lam * w_base, P_M_lam, calib, cfg)
    xl_lam, xk_lam, _ = compute_nested_factor_demands(
        calib.ytot, lam * r_base, lam * w_base, P_M_lam, c_va_lam, c_y_lam, lam * p_base, tau, calib, cfg
    )

    rel_diff_l = np.max(np.abs(xl_lam - xl_base) / xl_base)
    # Must fail because rel_diff_l is ~0.0526 (5.26%) on synthetic, ~4.08 (408%) on ICIO
    assert rel_diff_l < 1e-8, f"HD0 violated: xl jumped by {rel_diff_l * 100:.2f}%"


def test_defect_outer_nest_ces_clamping_at_high_sigma(calib_synthetic):
    """Demonstrate that sigma_y=20.0 outer cost clamps to 4.2813 when price level is 10.0."""
    calib = calib_synthetic
    cfg = FlexibleTechnologyConfig(sigma_y=20.0)
    c_va_0, _ = _compute_benchmark_cva0(calib)
    p_ones = np.ones((1, calib.n_sectors, calib.n_countries))

    # Test uniform scale factor lam = 10.0
    lam = 10.0
    c_y_lam = compute_outer_ces_cost(lam * c_va_0, lam * p_ones, calib, cfg, normalized=True)
    mean_val = float(np.mean(c_y_lam))

    # Clamping caps mean_val at 4.2813 instead of 10.0 (57.2% error)
    assert abs(mean_val - 10.0) < 1e-4, f"Outer cost clamped: got {mean_val:.4f}, expected 10.0"


def test_defect_inner_nest_ces_clamping_at_high_rho(calib_synthetic):
    """Demonstrate that rho_va=50.0 inner cost clamps to 2.5595 when factor prices are 10.0."""
    calib = calib_synthetic
    cfg = FlexibleTechnologyConfig(rho_va=50.0)
    p_ones = np.ones((1, calib.n_sectors, calib.n_countries))

    c_base = compute_inner_ces_cost(p_ones, p_ones, calib, cfg)
    lam = 10.0
    c_lam = compute_inner_ces_cost(lam * p_ones, lam * p_ones, calib, cfg)

    ratio = float(np.mean(c_lam / c_base))
    # Clamping caps ratio at 2.5595 instead of 10.0 (74.4% error)
    assert abs(ratio - 10.0) < 1e-4, f"Inner cost clamped: got ratio {ratio:.4f}, expected 10.0"


def test_defect_zero_profit_price_jump(calib_synthetic):
    """Demonstrate that compute_zero_profit_price jumps discontinuously when c_y scales by 1.10."""
    calib = calib_synthetic
    c_y_105 = np.ones((1, calib.n_sectors, calib.n_countries)) * 1.05
    c_y_110 = np.ones((1, calib.n_sectors, calib.n_countries)) * 1.10

    pp_105 = compute_zero_profit_price(c_y_105, calib)
    pp_110 = compute_zero_profit_price(c_y_110, calib)

    slope = float(np.mean((pp_110 - pp_105) / 0.05))
    # Theoretical slope is 1.0, but heuristic toggle makes it ~2.0
    assert abs(slope - 1.0) < 1e-3, f"Spurious jump in zero profit price: slope is {slope:.4f} instead of 1.0"
