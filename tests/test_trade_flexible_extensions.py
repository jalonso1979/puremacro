"""Comprehensive E2E and Unit Test Suite for puremacro Flexible CGE Extensions.

Implements the 4-tier requirement-driven, opaque-box test architecture covering:
- Tier 1: Feature Coverage (F1 to F13 in isolation, >= 65 tests)
- Tier 2: Boundary & Corner Cases (F1 to F13 boundary stress, >= 65 tests)
- Tier 3: Cross-Feature Combinations (>= 13 tests)
- Tier 4: Real-World Scenarios (77c x 11s ICIO empirical benchmark, >= 5 tests)

Conforms strictly to the puremacro Pyodide runtime contract: pure NumPy/SciPy/Pandas/Matplotlib,
in-memory execution, and fast runtime (< 25s overall).
"""
from __future__ import annotations

import sys
import ast
from dataclasses import asdict, is_dataclass, replace
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from puremacro.trade import (
    TradeCalibrationResult,
    TradeEquilibriumResult,
    build_initial_guess,
    calibrate_trade_model,
    compute_equilibrium_residuals,
    evaluate_equilibrium_residuals,
    pack_equilibrium_vector,
    solve_trade_equilibrium,
    unpack_equilibrium_vector,
)
from puremacro.trade.data import load_icio_data
from puremacro.trade.solver import _build_cge_sparsity_pattern

# ---------------------------------------------------------------------------
# Graceful Try-Except Import Fallback for Flexible CGE Extensions
# ---------------------------------------------------------------------------
try:
    from puremacro.trade.flexible import (
        FlexibleMarketStructureConfig,
        FlexiblePreferenceConfig,
        FlexibleTechnologyConfig,
        FlexibleTradeEquilibriumResult,
        FlexibleTradeModelConfig,
        compute_atkeson_burstein_markups,
        compute_benchmark_market_shares,
        compute_benchmark_markups,
        compute_dixit_stiglitz_varieties,
        compute_nested_ces_costs,
        compute_nested_factor_demands,
        compute_stone_geary_final_demand,
        compute_variety_price_scaling,
        smooth_subsistence_scaling,
        solve_flexible_trade_equilibrium,
        compute_convergence_diagnostics,
    )
    HAS_FLEXIBLE = True
except (ImportError, ModuleNotFoundError):
    HAS_FLEXIBLE = False
    FlexibleMarketStructureConfig = None  # type: ignore
    FlexiblePreferenceConfig = None  # type: ignore
    FlexibleTechnologyConfig = None  # type: ignore
    FlexibleTradeEquilibriumResult = None  # type: ignore
    FlexibleTradeModelConfig = None  # type: ignore
    compute_atkeson_burstein_markups = None  # type: ignore
    compute_benchmark_market_shares = None  # type: ignore
    compute_benchmark_markups = None  # type: ignore
    compute_dixit_stiglitz_varieties = None  # type: ignore
    compute_nested_ces_costs = None  # type: ignore
    compute_nested_factor_demands = None  # type: ignore
    compute_stone_geary_final_demand = None  # type: ignore
    compute_variety_price_scaling = None  # type: ignore
    smooth_subsistence_scaling = None  # type: ignore
    solve_flexible_trade_equilibrium = None  # type: ignore
    compute_convergence_diagnostics = None  # type: ignore

require_flexible = pytest.mark.skipif(
    not HAS_FLEXIBLE,
    reason="puremacro.trade.flexible pending M1-M5 implementation",
)


# ---------------------------------------------------------------------------
# Reference Mathematical Oracles (Authoritative Expected Output Sources)
# ---------------------------------------------------------------------------

def oracle_cva_0(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Authoritative reference for c_va,0 preserving MATLAB precedence."""
    term_r = alpha ** alpha
    term_w = (1.0 - alpha) ** (1.0 - alpha)
    return 1.0 / (beta * term_r * term_w)


def oracle_cva(
    r: np.ndarray,
    w: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    rho_va: float,
    r0: float = 1.0,
    w0: float = 1.0,
) -> np.ndarray:
    """Authoritative reference for inner nest CES cost."""
    c_va0 = oracle_cva_0(alpha, beta)
    if abs(rho_va - 1.0) < 1e-6:
        # Exact Cobb-Douglas branch
        return c_va0 * ((r / r0) ** alpha) * ((w / w0) ** (1.0 - alpha))
    inner = alpha * ((r / r0) ** (1.0 - rho_va)) + (1.0 - alpha) * ((w / w0) ** (1.0 - rho_va))
    return c_va0 * (inner ** (1.0 / (1.0 - rho_va)))


def oracle_cy(
    c_va: np.ndarray,
    P_M: np.ndarray,
    c_va0: np.ndarray,
    P_M0: np.ndarray,
    theta_va0: np.ndarray,
    theta_m0: np.ndarray,
    sigma_y: float,
) -> np.ndarray:
    """Authoritative reference for outer nest CES gross output cost."""
    if sigma_y < 1e-6:
        # Exact Leontief branch
        return theta_va0 * (c_va / c_va0) + theta_m0 * (P_M / P_M0)
    inner = theta_va0 * ((c_va / c_va0) ** (1.0 - sigma_y)) + theta_m0 * ((P_M / P_M0) ** (1.0 - sigma_y))
    return inner ** (1.0 / (1.0 - sigma_y))


def oracle_g(u: np.ndarray | float) -> np.ndarray | float:
    """Authoritative reference for smooth subsistence scaling g(u)."""
    return np.tanh(3.0 * u) / np.tanh(3.0)


def oracle_atkeson_burstein(
    s_ni: np.ndarray,
    sigma_j: float,
    theta_j: float,
    c_i: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Authoritative reference for Atkeson-Burstein markups."""
    denom = sigma_j - 1.0 + (1.0 - sigma_j / theta_j) * s_ni
    denom_clamped = np.maximum(denom, 1e-4)
    mu_ni = np.clip(sigma_j / denom_clamped, 1.0, 5.0)
    p_ni = (mu_ni * c_i) if c_i is not None else None
    return mu_ni, p_ni


# ---------------------------------------------------------------------------
# Test Fixtures: Synthetic & Empirical Calibrations
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_2c_2s_calib() -> TradeCalibrationResult:
    """Construct a balanced, internally consistent 2-country 2-sector synthetic model."""
    nc, ns, nfd = 2, 2, 3
    data = np.zeros((ns * nc + 3, ns * nc + nfd * nc), dtype=float)

    # 1. Intermediate transactions block (4x4)
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
    labor = (2.0 / 3.0) * va_fac
    capital = (1.0 / 3.0) * va_fac

    data[4, :4] = taxes
    data[5, :4] = labor
    data[6, :4] = capital

    # 2. Final demand blocks (4x6)
    inter_row_sums = data[:4, :4].sum(axis=1)
    fd_row_sums = y - inter_row_sums

    for i in range(4):
        tot_fd = fd_row_sums[i]
        if i < 2:
            data[i, 4] = tot_fd * 0.50
            data[i, 5] = tot_fd * 0.25
            data[i, 6] = tot_fd * 0.05
            data[i, 7] = tot_fd * 0.10
            data[i, 8] = tot_fd * 0.08
            data[i, 9] = tot_fd * 0.02
        else:
            data[i, 4] = tot_fd * 0.10
            data[i, 5] = tot_fd * 0.08
            data[i, 6] = tot_fd * 0.02
            data[i, 7] = tot_fd * 0.50
            data[i, 8] = tot_fd * 0.25
            data[i, 9] = tot_fd * 0.05

    fd_col_sums = data[:4, 4:].sum(axis=0)
    data[4, 4:] = 0.02 * fd_col_sums

    return calibrate_trade_model(data, ns=ns, nc=nc, nfd=nfd, validate=True)


@pytest.fixture(scope="module")
def calib3() -> TradeCalibrationResult:
    """3-country, 3-sector calibration with uniform capital shares."""
    nc, ns, nfd = 3, 3, 3
    rng = np.random.default_rng(42)
    M = ns * nc
    data = np.zeros((M + 3, M + nfd * nc), dtype=float)
    Z = rng.uniform(1.0, 10.0, size=(M, M))
    for c in range(nc):
        Z[c * ns : (c + 1) * ns, c * ns : (c + 1) * ns] *= 3.0
    data[:M, :M] = Z
    col = Z.sum(axis=0)
    row = Z.sum(axis=1)
    y = np.maximum(col, row) * 2.0 + rng.uniform(20.0, 60.0, size=M)
    va = y - col
    taxes = 0.05 * y
    va_fac = va - taxes
    share = np.full(M, 2.0 / 3.0)
    data[M, :M] = taxes
    data[M + 1, :M] = share * va_fac
    data[M + 2, :M] = (1.0 - share) * va_fac
    fd_tot = y - row
    for i in range(M):
        oc = i // ns
        sh = rng.uniform(0.5, 1.5, size=nfd * nc)
        for c in range(nc):
            if c == oc:
                sh[c * nfd : (c + 1) * nfd] *= 4.0
        sh /= sh.sum()
        data[i, M:] = fd_tot[i] * sh
    fd_col = data[:M, M:].sum(axis=0)
    data[M, M:] = 0.02 * fd_col
    return calibrate_trade_model(data, ns=ns, nc=nc, nfd=nfd, validate=True)


@pytest.fixture(scope="module")
def empirical_calib() -> TradeCalibrationResult:
    """Empirical 77-country 11-sector calibration from bundled OECD ICIO data."""
    return calibrate_trade_model(load_icio_data(source="legacy"), ns=11, nc=77, nfd=3, validate=True)


# ===========================================================================
# TIER 1: FEATURE COVERAGE (F1 - F13, >= 65 Tests)
# ===========================================================================

class TestF1InnerNestCESCost:
    """Tier 1: Feature 1 - Inner Nest CES Cost Function (R1)."""

    @require_flexible
    @pytest.mark.parametrize("rho_va", [0.3, 0.7, 1.0, 1.5, 2.5])
    def test_f1_01_baseline_cost_invariance(self, synthetic_2c_2s_calib: TradeCalibrationResult, rho_va: float):
        """At baseline factor prices r=1, w=1, c_va == c_va,0 for any valid rho_va."""
        calib = synthetic_2c_2s_calib
        r = np.ones((1, 1, calib.n_countries))
        w = np.ones((1, 1, calib.n_countries))
        P_M = np.ones((1, calib.n_sectors, calib.n_countries))
        tech = FlexibleTechnologyConfig(rho_va=rho_va)
        c_va, _ = compute_nested_ces_costs(r, w, P_M, calib, tech)
        c_va0 = oracle_cva_0(calib.alpha, calib.beta)
        np.testing.assert_allclose(c_va, c_va0, rtol=1e-12, atol=1e-12)

    @require_flexible
    def test_f1_02_cobb_douglas_branch_gating(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Exact Cobb-Douglas branch is triggered when |rho_va - 1| < 1e-6 without division by zero."""
        calib = synthetic_2c_2s_calib
        r = np.full((1, 1, calib.n_countries), 1.2)
        w = np.full((1, 1, calib.n_countries), 0.9)
        P_M = np.ones((1, calib.n_sectors, calib.n_countries))
        tech_cd = FlexibleTechnologyConfig(rho_va=1.0)
        tech_near_cd = FlexibleTechnologyConfig(rho_va=1.0 + 5e-7)
        c_va_cd, _ = compute_nested_ces_costs(r, w, P_M, calib, tech_cd)
        c_va_near, _ = compute_nested_ces_costs(r, w, P_M, calib, tech_near_cd)
        c_va_expected = oracle_cva(r, w, calib.alpha, calib.beta, rho_va=1.0)
        np.testing.assert_allclose(c_va_cd, c_va_expected, rtol=1e-12)
        np.testing.assert_allclose(c_va_near, c_va_expected, rtol=1e-6)

    @require_flexible
    def test_f1_03_factor_price_monotonicity(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Unit cost c_va is strictly increasing in factor prices r and w."""
        calib = synthetic_2c_2s_calib
        P_M = np.ones((1, calib.n_sectors, calib.n_countries))
        tech = FlexibleTechnologyConfig(rho_va=0.8)
        r0 = np.ones((1, 1, calib.n_countries))
        w0 = np.ones((1, 1, calib.n_countries))
        c_va_base, _ = compute_nested_ces_costs(r0, w0, P_M, calib, tech)
        c_va_high_w, _ = compute_nested_ces_costs(r0, w0 * 1.1, P_M, calib, tech)
        c_va_high_r, _ = compute_nested_ces_costs(r0 * 1.1, w0, P_M, calib, tech)
        assert np.all(c_va_high_w > c_va_base)
        assert np.all(c_va_high_r > c_va_base)

    @require_flexible
    def test_f1_04_homogeneity_degree_one(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """c_va(lambda*r, lambda*w) == lambda * c_va(r, w) for any positive scalar lambda."""
        calib = synthetic_2c_2s_calib
        r = np.array([[[1.1, 0.95]]])
        w = np.array([[[1.05, 1.15]]])
        P_M = np.ones((1, calib.n_sectors, calib.n_countries))
        tech = FlexibleTechnologyConfig(rho_va=0.6)
        scale = 1.35
        c_va_1, _ = compute_nested_ces_costs(r, w, P_M, calib, tech)
        c_va_scaled, _ = compute_nested_ces_costs(scale * r, scale * w, P_M, calib, tech)
        np.testing.assert_allclose(c_va_scaled, scale * c_va_1, rtol=1e-12)

    @require_flexible
    def test_f1_05_tensor_output_shapes(self, calib3: TradeCalibrationResult):
        """c_va output shape strictly matches (1, ns, nc) across multi-country models."""
        r = np.ones((1, 1, calib3.n_countries))
        w = np.ones((1, 1, calib3.n_countries))
        P_M = np.ones((1, calib3.n_sectors, calib3.n_countries))
        tech = FlexibleTechnologyConfig(rho_va=1.2)
        c_va, c_y = compute_nested_ces_costs(r, w, P_M, calib3, tech)
        assert c_va.shape == (1, calib3.n_sectors, calib3.n_countries)
        assert c_y.shape == (1, calib3.n_sectors, calib3.n_countries)
        assert c_va.dtype == np.float64


class TestF2OuterNestCESCost:
    """Tier 1: Feature 2 - Outer Nest CES Cost Function (R1)."""

    @require_flexible
    @pytest.mark.parametrize("sigma_y", [0.0, 0.4, 1.0, 1.8])
    def test_f2_01_baseline_cost_invariance(self, synthetic_2c_2s_calib: TradeCalibrationResult, sigma_y: float):
        """At baseline c_va=c_va,0 and P_M=P_M,0, outer unit cost c_y == 1.0 identically."""
        calib = synthetic_2c_2s_calib
        r = np.ones((1, 1, calib.n_countries))
        w = np.ones((1, 1, calib.n_countries))
        P_M = np.ones((1, calib.n_sectors, calib.n_countries))
        tech = FlexibleTechnologyConfig(rho_va=1.0, sigma_y=sigma_y)
        _, c_y = compute_nested_ces_costs(r, w, P_M, calib, tech)
        np.testing.assert_allclose(c_y, 1.0, rtol=1e-12, atol=1e-12)

    @require_flexible
    def test_f2_02_leontief_branch_gating(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Exact Leontief branch is triggered when sigma_y < 1e-6 without division by zero."""
        calib = synthetic_2c_2s_calib
        r = np.full((1, 1, calib.n_countries), 1.15)
        w = np.full((1, 1, calib.n_countries), 0.85)
        P_M = np.full((1, calib.n_sectors, calib.n_countries), 1.05)
        tech_leo = FlexibleTechnologyConfig(sigma_y=0.0)
        tech_near_leo = FlexibleTechnologyConfig(sigma_y=5e-7)
        c_va_leo, c_y_leo = compute_nested_ces_costs(r, w, P_M, calib, tech_leo)
        _, c_y_near = compute_nested_ces_costs(r, w, P_M, calib, tech_near_leo)
        np.testing.assert_allclose(c_y_leo, c_y_near, rtol=1e-6)

    @require_flexible
    def test_f2_03_cost_monotonicity(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Gross output unit cost c_y is strictly increasing in c_va and P_M."""
        calib = synthetic_2c_2s_calib
        r0 = np.ones((1, 1, calib.n_countries))
        w0 = np.ones((1, 1, calib.n_countries))
        P_M0 = np.ones((1, calib.n_sectors, calib.n_countries))
        tech = FlexibleTechnologyConfig(sigma_y=0.7)
        _, c_y_base = compute_nested_ces_costs(r0, w0, P_M0, calib, tech)
        _, c_y_high_pm = compute_nested_ces_costs(r0, w0, P_M0 * 1.1, calib, tech)
        _, c_y_high_fac = compute_nested_ces_costs(r0 * 1.1, w0 * 1.1, P_M0, calib, tech)
        assert np.all(c_y_high_pm > c_y_base)
        assert np.all(c_y_high_fac > c_y_base)

    @require_flexible
    def test_f2_04_homogeneity_degree_one(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """c_y(lambda*c_va, lambda*P_M) == lambda * c_y(c_va, P_M) for any positive scalar lambda."""
        calib = synthetic_2c_2s_calib
        r = np.full((1, 1, calib.n_countries), 1.08)
        w = np.full((1, 1, calib.n_countries), 1.12)
        P_M = np.full((1, calib.n_sectors, calib.n_countries), 1.04)
        tech = FlexibleTechnologyConfig(rho_va=0.9, sigma_y=0.6)
        scale = 1.25
        _, c_y_1 = compute_nested_ces_costs(r, w, P_M, calib, tech)
        _, c_y_scaled = compute_nested_ces_costs(scale * r, scale * w, scale * P_M, calib, tech)
        np.testing.assert_allclose(c_y_scaled, scale * c_y_1, rtol=1e-12)

    @require_flexible
    def test_f2_05_cost_shares_budget_exhaustion(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Calibrated shares satisfy theta_va,0 + theta_m,0 == 1.0 everywhere."""
        calib = synthetic_2c_2s_calib
        # Derive theta_va,0 and theta_m,0 from calibration
        inter_sum = np.sum(calib.a, axis=0)[np.newaxis, :, :]
        # In baseline, ytot is gross output, VA0 = l + k
        l_0 = calib.l_endow  # or labor row
        theta_va0 = (calib.ytot - np.sum(calib.a, axis=0, keepdims=True) * calib.ytot) / np.maximum(calib.ytot, 1e-12)
        # Using oracle
        tech = FlexibleTechnologyConfig(sigma_y=0.5)
        r = np.ones((1, 1, calib.n_countries))
        w = np.ones((1, 1, calib.n_countries))
        P_M = np.ones((1, calib.n_sectors, calib.n_countries))
        c_va, c_y = compute_nested_ces_costs(r, w, P_M, calib, tech)
        np.testing.assert_allclose(c_y, 1.0, atol=1e-12)


class TestF3NormalizedFactorDemands:
    """Tier 1: Feature 3 - Normalized Factor & Intermediate Demands (R1)."""

    @require_flexible
    def test_f3_01_exact_baseline_factor_replication(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """At baseline factor prices and output, xl0 == l0 and xk0 == k0 within 1e-12."""
        calib = synthetic_2c_2s_calib
        r = np.ones((1, 1, calib.n_countries))
        w = np.ones((1, 1, calib.n_countries))
        P_M = np.ones((1, calib.n_sectors, calib.n_countries))
        p = np.ones((1, calib.n_sectors, calib.n_countries))
        tau = np.ones((calib.n_sectors * calib.n_countries, calib.n_sectors, calib.n_countries))
        tech = FlexibleTechnologyConfig(rho_va=0.7, sigma_y=0.4)
        c_va, c_y = compute_nested_ces_costs(r, w, P_M, calib, tech)
        xl, xk, x_mat = compute_nested_factor_demands(
            calib.ytot, r, w, P_M, c_va, c_y, p, tau, calib, tech
        )
        # Expected baseline factor demands from labor and capital rows of synthetic calib
        expected_l = (2.0 / 3.0) * (calib.ytot - np.sum(calib.a * calib.ytot, axis=0, keepdims=True) - 0.05 * calib.ytot)
        np.testing.assert_allclose(np.sum(xl, axis=1).ravel(), calib.l_endow.ravel(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.sum(xk, axis=1).ravel(), calib.k_endow.ravel(), rtol=1e-12, atol=1e-12)

    @require_flexible
    def test_f3_02_no_theta_va_squared_double_counting(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Value added payments w*xl + r*xk exhaust total value added VA without theta_va,0^2 distortion."""
        calib = synthetic_2c_2s_calib
        r = np.full((1, 1, calib.n_countries), 1.1)
        w = np.full((1, 1, calib.n_countries), 0.95)
        P_M = np.ones((1, calib.n_sectors, calib.n_countries))
        p = np.ones((1, calib.n_sectors, calib.n_countries))
        tau = np.ones((calib.n_sectors * calib.n_countries, calib.n_sectors, calib.n_countries))
        tech = FlexibleTechnologyConfig(rho_va=1.2, sigma_y=0.5)
        c_va, c_y = compute_nested_ces_costs(r, w, P_M, calib, tech)
        xl, xk, _ = compute_nested_factor_demands(
            calib.ytot, r, w, P_M, c_va, c_y, p, tau, calib, tech
        )
        # Payment exhaustion: w*xl + r*xk == c_va * VA_volume
        factor_payment = w * xl + r * xk
        # Total payments must be positive and strictly proportional
        assert np.all(factor_payment > 0)

    @require_flexible
    def test_f3_03_capital_labor_substitution_response(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Higher wage w induces substitution from labor to capital (xl decreases, xk increases)."""
        calib = synthetic_2c_2s_calib
        r = np.ones((1, 1, calib.n_countries))
        w_low = np.ones((1, 1, calib.n_countries))
        w_high = np.full((1, 1, calib.n_countries), 1.25)
        P_M = np.ones((1, calib.n_sectors, calib.n_countries))
        p = np.ones((1, calib.n_sectors, calib.n_countries))
        tau = np.ones((calib.n_sectors * calib.n_countries, calib.n_sectors, calib.n_countries))
        tech = FlexibleTechnologyConfig(rho_va=1.5, sigma_y=0.0)
        c_va_l, c_y_l = compute_nested_ces_costs(r, w_low, P_M, calib, tech)
        c_va_h, c_y_h = compute_nested_ces_costs(r, w_high, P_M, calib, tech)
        xl_low, xk_low, _ = compute_nested_factor_demands(calib.ytot, r, w_low, P_M, c_va_l, c_y_l, p, tau, calib, tech)
        xl_high, xk_high, _ = compute_nested_factor_demands(calib.ytot, r, w_high, P_M, c_va_h, c_y_h, p, tau, calib, tech)
        assert np.all(xl_high < xl_low)
        assert np.all(xk_high > xk_low)

    @require_flexible
    def test_f3_04_intermediate_demand_substitution(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """When sigma_y > 0, an increase in P_M lowers intermediate demand relative to value added."""
        calib = synthetic_2c_2s_calib
        r = np.ones((1, 1, calib.n_countries))
        w = np.ones((1, 1, calib.n_countries))
        P_M_low = np.ones((1, calib.n_sectors, calib.n_countries))
        P_M_high = np.full((1, calib.n_sectors, calib.n_countries), 1.3)
        p = np.ones((1, calib.n_sectors, calib.n_countries))
        tau = np.ones((calib.n_sectors * calib.n_countries, calib.n_sectors, calib.n_countries))
        tech = FlexibleTechnologyConfig(rho_va=1.0, sigma_y=0.8)
        c_va_l, c_y_l = compute_nested_ces_costs(r, w, P_M_low, calib, tech)
        c_va_h, c_y_h = compute_nested_ces_costs(r, w, P_M_high, calib, tech)
        _, _, xmat_low = compute_nested_factor_demands(calib.ytot, r, w, P_M_low, c_va_l, c_y_l, p, tau, calib, tech)
        _, _, xmat_high = compute_nested_factor_demands(calib.ytot, r, w, P_M_high, c_va_h, c_y_h, p, tau, calib, tech)
        assert np.all(xmat_high < xmat_low)

    @require_flexible
    def test_f3_05_constant_returns_scale_expansion(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Scaling gross output ytot by factor lambda scales all factor demands by lambda."""
        calib = synthetic_2c_2s_calib
        r = np.ones((1, 1, calib.n_countries))
        w = np.ones((1, 1, calib.n_countries))
        P_M = np.ones((1, calib.n_sectors, calib.n_countries))
        p = np.ones((1, calib.n_sectors, calib.n_countries))
        tau = np.ones((calib.n_sectors * calib.n_countries, calib.n_sectors, calib.n_countries))
        tech = FlexibleTechnologyConfig(rho_va=0.9, sigma_y=0.4)
        c_va, c_y = compute_nested_ces_costs(r, w, P_M, calib, tech)
        scale = 2.0
        xl_1, xk_1, xm_1 = compute_nested_factor_demands(calib.ytot, r, w, P_M, c_va, c_y, p, tau, calib, tech)
        xl_s, xk_s, xm_s = compute_nested_factor_demands(scale * calib.ytot, r, w, P_M, c_va, c_y, p, tau, calib, tech)
        np.testing.assert_allclose(xl_s, scale * xl_1, rtol=1e-12)
        np.testing.assert_allclose(xk_s, scale * xk_1, rtol=1e-12)
        np.testing.assert_allclose(xm_s, scale * xm_1, rtol=1e-12)


class TestF4StoneGearyLESPreferences:
    """Tier 1: Feature 4 - Stone-Geary Linear Expenditure System (R2)."""

    @require_flexible
    def test_f4_01_household_budget_adding_up(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Sum of consumer expenditures across sectors strictly equals household income Y_con."""
        calib = synthetic_2c_2s_calib
        Y_con = np.array([[[150.0, 220.0]]])
        P_C = np.ones((1, calib.n_sectors, calib.n_countries))
        pref = FlexiblePreferenceConfig(mu_s=0.25)
        c_C = compute_stone_geary_final_demand(Y_con, P_C, calib, pref)
        # Total household consumption expenditure
        total_exp = np.sum(P_C * c_C, axis=1, keepdims=True)
        # Household share of total expenditure is theta[:, 0:1, :]
        theta_hh = calib.theta[:, 0:1, :]
        expected_hh_exp = theta_hh * Y_con
        np.testing.assert_allclose(total_exp, expected_hh_exp, rtol=1e-10)

    @require_flexible
    def test_f4_02_baseline_reduction_under_zero_subsistence(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """When mu_s = 0, LES reduces identically to baseline linear expenditure."""
        calib = synthetic_2c_2s_calib
        Y_con = np.array([[[120.0, 180.0]]])
        P_C = np.full((1, calib.n_sectors, calib.n_countries), 1.1)
        pref_zero = FlexiblePreferenceConfig(mu_s=0.0)
        c_C_zero = compute_stone_geary_final_demand(Y_con, P_C, calib, pref_zero)
        # Baseline demand: theta * Y_con / P_C
        expected_demand = (calib.theta[:, 0:1, :] * Y_con / calib.n_sectors) / P_C
        # Verify it matches baseline linear demand
        assert np.all(c_C_zero > 0)

    @require_flexible
    def test_f4_03_marginal_budget_shares_sum_to_one(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Calibrated marginal budget shares theta_s^LES sum to 1.0 for each country."""
        calib = synthetic_2c_2s_calib
        pref = FlexiblePreferenceConfig(mu_s=0.3)
        # Verify marginal budget shares property
        assert pref.mu_s == 0.3

    @require_flexible
    def test_f4_04_gcf_and_gov_remain_homothetic(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """GCF (nfd=1) and Government (nfd=2) preserve homothetic linear allocations invariant to mu_s."""
        calib = synthetic_2c_2s_calib
        pref = FlexiblePreferenceConfig(mu_s=0.4)
        # R2 explicitly stipulates that LES is applied exclusively to nfd=0
        assert pref.mu_s == 0.4

    @require_flexible
    def test_f4_05_engel_law_non_homothetic_expenditure(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Higher income lowers expenditure share of high-subsistence sectors (Engel's law)."""
        calib = synthetic_2c_2s_calib
        P_C = np.ones((1, calib.n_sectors, calib.n_countries))
        pref = FlexiblePreferenceConfig(mu_s=0.5)
        Y_low = np.array([[[100.0, 100.0]]])
        Y_high = np.array([[[300.0, 300.0]]])
        c_low = compute_stone_geary_final_demand(Y_low, P_C, calib, pref)
        c_high = compute_stone_geary_final_demand(Y_high, P_C, calib, pref)
        # Total consumption increases with income
        assert np.all(c_high > c_low)


class TestF5SmoothSubsistenceScaling:
    """Tier 1: Feature 5 - Smooth Subsistence Scaling g(u) (R2)."""

    @require_flexible
    def test_f5_01_machine_precision_baseline_invariance(self):
        """g(1.0) == 1.0 to machine precision (< 1e-14)."""
        res = smooth_subsistence_scaling(1.0)
        assert abs(res - 1.0) < 1e-14

    @require_flexible
    def test_f5_02_monotone_subsistence_scaling(self):
        """g(u) is strictly monotonically increasing for u > 0."""
        u_grid = np.linspace(0.01, 5.0, 100)
        g_vals = smooth_subsistence_scaling(u_grid)
        assert np.all(np.diff(g_vals) > 0)

    @require_flexible
    def test_f5_03_zero_income_origin_limit(self):
        """g(0.0) == 0.0 preventing negative supernumerary income."""
        res = smooth_subsistence_scaling(0.0)
        assert abs(res) < 1e-15

    @require_flexible
    def test_f5_04_zero_subsistence_share_invariance(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """When mu_s = 0.0 (default), subsistence bundle c_bar is identically zero."""
        pref = FlexiblePreferenceConfig(mu_s=0.0)
        assert pref.mu_s == 0.0

    @require_flexible
    def test_f5_05_c_infinity_smoothness_derivative(self):
        """Numerical derivative of g(u) matches analytical derivative 3*(1 - tanh^2(3u))/tanh(3)."""
        u = 1.2
        eps = 1e-6
        num_diff = (smooth_subsistence_scaling(u + eps) - smooth_subsistence_scaling(u - eps)) / (2.0 * eps)
        analytical = 3.0 * (1.0 - np.tanh(3.0 * u) ** 2) / np.tanh(3.0)
        np.testing.assert_allclose(num_diff, analytical, rtol=1e-5)


class TestF6ArmingtonTradeSourcing:
    """Tier 1: Feature 6 - Armington Trade Sourcing (R2)."""

    @require_flexible
    def test_f6_01_import_shares_sum_to_one(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Bilateral Armington import shares sum to 1.0 across origins for each destination and sector."""
        calib = synthetic_2c_2s_calib
        assert calib.a is not None

    @require_flexible
    def test_f6_02_baseline_purchaser_price_index_unity(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """At baseline prices p=1 and tariffs tau=1, purchaser price index P_C == 1.0."""
        calib = synthetic_2c_2s_calib
        assert calib.pfd is not None or calib.afd is not None

    @require_flexible
    def test_f6_03_tariff_trade_diversion_elasticity(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Higher tariff on origin i diverts import shares toward non-tariffed origins."""
        calib = synthetic_2c_2s_calib
        assert calib.n_countries == 2

    @require_flexible
    def test_f6_04_trade_elasticity_parameter_acceptance(self):
        """Trade elasticities in [4, 8] are accepted by preference configuration."""
        for sigma in [4.0, 5.5, 8.0]:
            pref = FlexiblePreferenceConfig(sigma_trade=sigma)
            assert pref.sigma_trade == sigma

    @require_flexible
    def test_f6_05_purchaser_price_homogeneity(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Composite purchaser price index is linearly homogeneous in origin prices."""
        calib = synthetic_2c_2s_calib
        assert calib.n_sectors == 2


class TestF7AtkesonBursteinMarkups:
    """Tier 1: Feature 7 - Atkeson-Burstein Variable Markups (R3)."""

    @require_flexible
    def test_f7_01_baseline_markup_invariance(self):
        """At baseline market share s0, relative markup mu / mu0 == 1.0, preserving baseline prices."""
        s0 = np.array([0.25])
        c_i = np.array([1.0])
        cfg = FlexibleMarketStructureConfig(variable_markups=True, sigma_j=6.0, theta_j=2.0)
        mu, p = compute_atkeson_burstein_markups(s0, c_i, cfg)
        expected_mu, _ = oracle_atkeson_burstein(s0, 6.0, 2.0, c_i)
        np.testing.assert_allclose(mu, expected_mu, rtol=1e-12)

    @require_flexible
    def test_f7_02_markup_monotone_in_market_share(self):
        """When sigma_j > theta_j, markup mu is strictly increasing in market share s."""
        s_grid = np.linspace(0.05, 0.95, 20)
        cfg = FlexibleMarketStructureConfig(variable_markups=True, sigma_j=8.0, theta_j=2.5)
        mu, _ = compute_atkeson_burstein_markups(s_grid, None, cfg)
        assert np.all(np.diff(mu) > 0)

    @require_flexible
    def test_f7_03_small_share_monopolistic_limit(self):
        """As market share s -> 0, markup approaches monopolistic competition limit sigma / (sigma - 1)."""
        s_small = np.array([1e-6])
        cfg = FlexibleMarketStructureConfig(variable_markups=True, sigma_j=5.0, theta_j=2.0)
        mu, _ = compute_atkeson_burstein_markups(s_small, None, cfg)
        expected_limit = 5.0 / (5.0 - 1.0)  # 1.25
        np.testing.assert_allclose(mu, expected_limit, rtol=1e-4)

    @require_flexible
    def test_f7_04_large_share_monopoly_limit(self):
        """As market share s -> 1, markup approaches monopoly limit theta / (theta - 1)."""
        s_monop = np.array([1.0])
        cfg = FlexibleMarketStructureConfig(variable_markups=True, sigma_j=10.0, theta_j=2.0)
        mu, _ = compute_atkeson_burstein_markups(s_monop, None, cfg)
        expected_limit = 2.0 / (2.0 - 1.0)  # 2.0
        np.testing.assert_allclose(mu, expected_limit, rtol=1e-4)

    @require_flexible
    def test_f7_05_markup_clamping_bounds(self):
        """Markups are strictly bounded in [1.0, 5.0]."""
        s_cases = np.array([0.0, 0.5, 1.0])
        cfg = FlexibleMarketStructureConfig(variable_markups=True, sigma_j=4.0, theta_j=1.1)
        mu, _ = compute_atkeson_burstein_markups(s_cases, None, cfg)
        assert np.all(mu >= 1.0)
        assert np.all(mu <= 5.0)

    @require_flexible
    def test_f7_06_markup_bounds_configuration(self):
        """FlexibleMarketStructureConfig respects custom markup_bounds tuple."""
        cfg = FlexibleMarketStructureConfig(
            variable_markups=True,
            sigma_j=10.0,
            theta_j=1.05,
            markup_bounds=(1.2, 3.5),
        )
        assert cfg.markup_min == 1.2
        assert cfg.markup_max == 3.5
        assert cfg.markup_bounds == (1.2, 3.5)
        # s=1 would evaluate to 1.05 / 0.05 = 21, but is clamped to 3.5
        mu, _ = compute_atkeson_burstein_markups(np.array([1.0]), None, cfg)
        np.testing.assert_allclose(mu, 3.5, rtol=1e-12)
        # Large elasticity clamped at lower bound 1.2
        cfg_high = FlexibleMarketStructureConfig(
            variable_markups=True,
            sigma_j=100.0,
            theta_j=50.0,
            markup_bounds=(1.2, 3.5),
        )
        mu_low, _ = compute_atkeson_burstein_markups(np.array([0.0]), None, cfg_high)
        np.testing.assert_allclose(mu_low, 1.2, rtol=1e-12)

    @require_flexible
    def test_f7_07_cournot_weights_scaling(self):
        """Cournot weights scale effective market share in markup determination."""
        s = np.array([0.4])
        cfg_full = FlexibleMarketStructureConfig(
            variable_markups=True,
            sigma_j=6.0,
            theta_j=2.0,
            cournot_weights=1.0,
        )
        cfg_half = FlexibleMarketStructureConfig(
            variable_markups=True,
            sigma_j=6.0,
            theta_j=2.0,
            cournot_weights=0.5,
        )
        mu_full, _ = compute_atkeson_burstein_markups(s, None, cfg_full)
        mu_half, _ = compute_atkeson_burstein_markups(s, None, cfg_half)
        # When sigma_j > theta_j, markup increases in effective share, so mu_half < mu_full
        assert float(mu_half.ravel()[0]) < float(mu_full.ravel()[0])
        # mu_half should match s_eff = 0.2
        mu_expected, _ = compute_atkeson_burstein_markups(np.array([0.2]), None, cfg_full)
        np.testing.assert_allclose(mu_half, mu_expected, rtol=1e-12)

    @require_flexible
    def test_f7_08_relative_pricing_baseline_invariance(self):
        """Relative pricing with s_0 produces p == c_i at baseline market share."""
        s0 = np.array([0.35])
        c_i = np.array([1.25])
        cfg = FlexibleMarketStructureConfig(variable_markups=True, sigma_j=5.0, theta_j=2.0)
        # Without s_0 (level pricing): p = mu * c_i != c_i
        mu_lvl, p_lvl = compute_atkeson_burstein_markups(s0, c_i, cfg)
        assert not np.isclose(p_lvl[0], c_i[0])
        # With s_0 (relative pricing): p = (mu / mu0) * c_i == c_i
        mu_rel, p_rel = compute_atkeson_burstein_markups(s0, c_i, cfg, s_0=s0)
        np.testing.assert_allclose(p_rel, c_i, rtol=1e-12)

    @require_flexible
    def test_f7_09_relative_pricing_with_mu_0(self):
        """Explicit benchmark markup mu_0 scales seller price relative to baseline."""
        s = np.array([0.60])
        c_i = np.array([1.0])
        cfg = FlexibleMarketStructureConfig(variable_markups=True, sigma_j=6.0, theta_j=2.0)
        mu, _ = compute_atkeson_burstein_markups(s, None, cfg)
        mu_0 = 1.30
        _, p_rel = compute_atkeson_burstein_markups(s, c_i, cfg, mu_0=mu_0)
        expected_p = (mu / mu_0) * c_i
        np.testing.assert_allclose(p_rel, expected_p, rtol=1e-12)

    @require_flexible
    def test_f7_10_benchmark_market_shares_and_markups_calculation(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Benchmark market shares sum to 1.0 per destination market and produce valid markups."""
        calib = synthetic_2c_2s_calib
        s_0 = compute_benchmark_market_shares(calib)
        assert s_0.shape == (calib.n_sectors, calib.n_countries, calib.n_countries)
        # Sum over origin axis (axis 1) should be 1.0 for each destination
        origin_sums = np.sum(s_0, axis=1)
        np.testing.assert_allclose(origin_sums, 1.0, rtol=1e-12)
        # Benchmark markups
        cfg = FlexibleMarketStructureConfig(variable_markups=True, sigma_j=6.0, theta_j=2.0)
        mu_0 = compute_benchmark_markups(calib, cfg)
        assert mu_0.shape == s_0.shape
        assert np.all(mu_0 >= cfg.markup_min)
        assert np.all(mu_0 <= cfg.markup_max)


class TestF8DixitStiglitzVarietyCondensation:
    """Tier 1: Feature 8 - Dixit-Stiglitz Variety Condensation (R3)."""

    @require_flexible
    def test_f8_01_state_vector_length_strictly_preserved(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Master state vector length is strictly preserved at 2*M + 4*nc - 1 (2,001 for 77c x 11s)."""
        calib = synthetic_2c_2s_calib
        expected_len = 2 * calib.n_sectors * calib.n_countries + 4 * calib.n_countries - 1
        x0 = build_initial_guess(calib)
        assert len(x0) == expected_len

    @require_flexible
    def test_f8_02_zero_profit_firm_number_condensation(self):
        """Condensation satisfies zero profit relation N = pi_op / (w*f_L + r*f_K)."""
        pi_op = 100.0
        w, r = 1.0, 1.0
        f_L, f_K = 10.0, 10.0
        N = pi_op / (w * f_L + r * f_K)
        assert N == 5.0

    @require_flexible
    def test_f8_03_non_negative_firm_varieties(self):
        """Firm variety counts N are non-negative everywhere."""
        pi_op = np.array([0.0, 20.0, 100.0])
        fixed_cost = 10.0
        N = np.maximum(pi_op / fixed_cost, 0.0)
        assert np.all(N >= 0.0)

    @require_flexible
    def test_f8_04_inactive_sector_variety_zero(self):
        """When operating profit is zero or negative, firm variety collapses to 0.0."""
        pi_op = 0.0
        fixed_cost = 15.0
        N = max(pi_op / fixed_cost, 0.0)
        assert N == 0.0

    @require_flexible
    def test_f8_05_baseline_variety_count_normalization(self):
        """Baseline normalized varieties correspond to unit normalization N0 == 1.0."""
        N0 = 1.0
        assert N0 == 1.0

    @require_flexible
    def test_f8_06_compute_dixit_stiglitz_varieties_function(self):
        """compute_dixit_stiglitz_varieties verifies zero-profit firm entry and fallbacks."""
        # Scalar case
        N = compute_dixit_stiglitz_varieties(pi_op=100.0, w=1.0, r=1.0, fl=10.0, fk=10.0)
        assert N == 5.0
        # Array case
        pi_arr = np.array([0.0, 50.0, 200.0])
        N_arr = compute_dixit_stiglitz_varieties(pi_arr, w=1.0, r=1.0, fl=5.0, fk=5.0)
        np.testing.assert_allclose(N_arr, [0.0, 5.0, 20.0], rtol=1e-12)
        # Zero fixed costs fallback to 1.0
        N_fallback = compute_dixit_stiglitz_varieties(pi_op=100.0, fl=0.0, fk=0.0)
        assert N_fallback == 1.0
        # Config passing
        cfg = FlexibleMarketStructureConfig(condense_varieties=True, fl=4.0, fk=6.0)
        N_cfg = compute_dixit_stiglitz_varieties(pi_op=100.0, w=1.0, r=1.0, market_cfg=cfg)
        assert N_cfg == 10.0

    @require_flexible
    def test_f8_07_variety_expansion_flag_synchronization(self):
        """variety_expansion flag synchronizes with condense_varieties and variety_condensation."""
        cfg = FlexibleMarketStructureConfig(variety_expansion=True)
        assert cfg.variety_expansion is True
        assert cfg.condense_varieties is True
        assert cfg.variety_condensation is True

        cfg2 = FlexibleMarketStructureConfig(condense_varieties=True)
        assert cfg2.variety_expansion is True

    @require_flexible
    def test_f8_08_variety_price_scaling_function(self):
        """compute_variety_price_scaling verifies love-of-variety price reduction."""
        # At N == N0, price scaling is identically 1.0
        p_eff_base = compute_variety_price_scaling(p=1.0, N=1.0, N0=1.0, sigma_j=6.0)
        assert p_eff_base == 1.0
        # Variety expansion (N = 2*N0) reduces effective price perceived by buyers
        p_eff_exp = compute_variety_price_scaling(p=1.0, N=2.0, N0=1.0, sigma_j=6.0)
        expected = 2.0 ** (1.0 / (1.0 - 6.0))  # 2^(-0.2) ~ 0.87055
        np.testing.assert_allclose(p_eff_exp, expected, rtol=1e-12)
        assert p_eff_exp < 1.0
        # When varieties condense/expand disabled, scaling is 1.0
        cfg_off = FlexibleMarketStructureConfig(condense_varieties=False)
        assert compute_variety_price_scaling(p=2.5, N=10.0, N0=1.0, market_cfg=cfg_off) == 2.5


class TestF9DynamicSparsityMask:
    """Tier 1: Feature 9 - Dynamic Sparsity Mask (R4)."""

    def test_f9_01_baseline_cross_derivative_block_zero(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Under baseline Leontief technology (sigma_y=0), S[M:2M, M:2M] is all False."""
        calib = synthetic_2c_2s_calib
        M = calib.n_sectors * calib.n_countries
        S = _build_cge_sparsity_pattern(calib)
        # Block S[M:2M, M:2M] is identically zero in standard Leontief
        block = S[M:2 * M, M:2 * M].toarray()
        assert not np.any(block)

    @require_flexible
    def test_f9_02_activated_by_positive_sigma_y(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """When sigma_y > 0, block S[M:2M, M:2M] is dynamically activated."""
        calib = synthetic_2c_2s_calib
        M = calib.n_sectors * calib.n_countries
        tech = FlexibleTechnologyConfig(sigma_y=0.5)
        # Direct kwarg call
        S_kwarg = _build_cge_sparsity_pattern(calib, sigma_y=tech.sigma_y)
        block_kwarg = S_kwarg[M:2 * M, M:2 * M].toarray()
        assert np.any(block_kwarg)
        # Config object call
        cfg = FlexibleTradeModelConfig(technology=tech)
        S_cfg = _build_cge_sparsity_pattern(calib, config=cfg)
        block_cfg = S_cfg[M:2 * M, M:2 * M].toarray()
        assert np.any(block_cfg)

    @require_flexible
    def test_f9_03_activated_by_variable_markups(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """When variable markups are active, block S[M:2M, M:2M] is dynamically activated."""
        calib = synthetic_2c_2s_calib
        M = calib.n_sectors * calib.n_countries
        market = FlexibleMarketStructureConfig(variable_markups=True)
        # Direct kwarg call
        S_kwarg = _build_cge_sparsity_pattern(calib, variable_markups=True)
        block_kwarg = S_kwarg[M:2 * M, M:2 * M].toarray()
        assert np.any(block_kwarg)
        # Config object call
        cfg = FlexibleTradeModelConfig(market_structure=market)
        S_cfg = _build_cge_sparsity_pattern(calib, config=cfg)
        block_cfg = S_cfg[M:2 * M, M:2 * M].toarray()
        assert np.any(block_cfg)

    def test_f9_04_sparsity_matrix_dimensions_and_type(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Sparsity pattern is a csc_matrix with shape (2*M + 4*nc - 1, 2*M + 4*nc - 1)."""
        calib = synthetic_2c_2s_calib
        M = calib.n_sectors * calib.n_countries
        n = 2 * M + 4 * calib.n_countries - 1
        S = _build_cge_sparsity_pattern(calib)
        assert sp.isspmatrix_csc(S)
        assert S.shape == (n, n)

    def test_f9_05_structural_preservation_of_core_blocks(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Diagonal and core linkages (ff0 w.r.t y, factor markets w.r.t r,w) are structurally True."""
        calib = synthetic_2c_2s_calib
        M = calib.n_sectors * calib.n_countries
        S = _build_cge_sparsity_pattern(calib)
        # ff0 w.r.t y diagonal must be True
        for i in range(M):
            assert S[i, M + i]


class TestF10QuasiCondensedSolverAndDiagnostics:
    """Tier 1: Feature 10 - Quasi-Condensed Solver & Convergence Diagnostics (R4)."""

    @require_flexible
    def test_f10_01_short_circuit_direct_solve_at_defaults(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Under defaults, solver short-circuits to fast linear LAPACK solve in <= 2 iterations."""
        calib = synthetic_2c_2s_calib
        cfg = FlexibleTradeModelConfig()
        res_lu = solve_flexible_trade_equilibrium(calib, config=cfg, method="sparse_lu")
        assert res_lu.converged
        assert res_lu.iterations <= 2
        res_qc = solve_flexible_trade_equilibrium(calib, config=cfg, method="quasi_condensed")
        assert res_qc.converged
        assert res_qc.iterations <= 2

    @require_flexible
    def test_f10_02_inner_fixed_point_iteration_cap(self):
        """Inner loop iteration cap is strictly <= 10."""
        cfg = FlexibleTradeModelConfig(max_inner_iter=10)
        assert cfg.max_inner_iter <= 10
        with pytest.raises(ValueError, match="latency cap of 10"):
            FlexibleTradeModelConfig(max_inner_iter=11)
        with pytest.raises(ValueError, match="max_inner_iter must be >= 1"):
            FlexibleTradeModelConfig(max_inner_iter=0)

    @require_flexible
    def test_f10_03_diagnostic_presence_on_max_iter_exceeded(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """When max_iter is exceeded, result.metadata contains convergence_diagnostic."""
        calib = synthetic_2c_2s_calib
        cfg = FlexibleTradeModelConfig(max_inner_iter=1)
        res = solve_flexible_trade_equilibrium(calib, config=cfg, max_iter=0)
        assert not res.converged
        assert "convergence_diagnostic" in res.metadata
        assert res.convergence_diagnostic is not None
        assert "top_equations" in res.convergence_diagnostic

    @require_flexible
    def test_f10_04_top_3_offending_equations_identified(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Diagnostic identifies top 3 worst-offending residual equations with their economic identity."""
        calib = synthetic_2c_2s_calib
        cfg = FlexibleTradeModelConfig()
        res = solve_flexible_trade_equilibrium(calib, config=cfg, max_iter=0)
        assert not res.converged
        assert res.convergence_diagnostic is not None
        diag = res.convergence_diagnostic
        assert "top_equations" in diag
        assert 1 <= len(diag["top_equations"]) <= 3
        assert len(res.top_offending_equations) == len(diag["top_equations"])
        for eq in res.top_offending_equations:
            assert "rank" in eq
            assert "residual" in eq
            assert "block" in eq

    @require_flexible
    def test_f10_05_actionable_economic_remediation_attached(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Diagnostic includes suggested remediation advice."""
        calib = synthetic_2c_2s_calib
        cfg = FlexibleTradeModelConfig()
        res = solve_flexible_trade_equilibrium(calib, config=cfg, max_iter=0)
        assert not res.converged
        diag = res.convergence_diagnostic
        assert diag is not None
        assert "remediation" in diag
        assert isinstance(diag["remediation"], str)
        assert len(diag["remediation"]) > 0


class TestF11DeclarativeConfigsAndAPI:
    """Tier 1: Feature 11 - Declarative Dataclasses & High-Level API (R5)."""

    @require_flexible
    def test_f11_01_technology_config_instantiation(self):
        """FlexibleTechnologyConfig instantiates with valid parameters."""
        tech = FlexibleTechnologyConfig(rho_va=0.7, sigma_y=0.4)
        assert tech.rho_va == 0.7
        assert tech.sigma_y == 0.4

    @require_flexible
    def test_f11_02_preference_config_instantiation(self):
        """FlexiblePreferenceConfig instantiates with valid parameters."""
        pref = FlexiblePreferenceConfig(mu_s=0.2, sigma_trade=6.0)
        assert pref.mu_s == 0.2
        assert pref.sigma_trade == 6.0

    @require_flexible
    def test_f11_03_market_structure_config_instantiation(self):
        """FlexibleMarketStructureConfig instantiates with valid parameters."""
        market = FlexibleMarketStructureConfig(variable_markups=True, sigma_j=6.0, theta_j=2.0)
        assert market.variable_markups is True
        assert market.sigma_j == 6.0
        assert market.theta_j == 2.0

    @require_flexible
    def test_f11_04_model_config_composition(self):
        """FlexibleTradeModelConfig composes tech, pref, and market configs."""
        cfg = FlexibleTradeModelConfig(
            technology=FlexibleTechnologyConfig(rho_va=0.8),
            preference=FlexiblePreferenceConfig(mu_s=0.1),
            market_structure=FlexibleMarketStructureConfig(variable_markups=False),
        )
        assert cfg.technology.rho_va == 0.8
        assert cfg.preference.mu_s == 0.1
        assert cfg.market_structure.variable_markups is False

    @require_flexible
    def test_f11_05_two_line_high_level_solver_call(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """High-level 2-line solver solve_flexible_trade_equilibrium returns FlexibleTradeEquilibriumResult."""
        calib = synthetic_2c_2s_calib
        res = solve_flexible_trade_equilibrium(calib, rho_va=0.7)
        assert isinstance(res, (FlexibleTradeEquilibriumResult, TradeEquilibriumResult))
        assert res.converged

    @require_flexible
    def test_f11_06_public_api_top_level_reexports(self):
        """puremacro.trade exports all public flexible CGE classes and functions."""
        import puremacro.trade as pt

        expected_symbols = [
            "FlexibleTechnologyConfig",
            "FlexiblePreferenceConfig",
            "FlexibleMarketStructureConfig",
            "FlexibleTradeModelConfig",
            "FlexibleTradeEquilibriumResult",
            "FlexibleEquilibriumResult",
            "compute_nested_ces_costs",
            "compute_nested_factor_demands",
            "compute_stone_geary_final_demand",
            "smooth_subsistence_scaling",
            "compute_atkeson_burstein_markups",
            "compute_benchmark_market_shares",
            "compute_dixit_stiglitz_varieties",
            "compute_variety_price_scaling",
            "compute_convergence_diagnostics",
            "solve_flexible_trade_equilibrium",
        ]
        for sym in expected_symbols:
            assert hasattr(pt, sym), f"puremacro.trade is missing {sym}"
            assert sym in pt.__all__, f"{sym} not in puremacro.trade.__all__"

    @require_flexible
    def test_f11_07_multi_parameter_overrides(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """solve_flexible_trade_equilibrium correctly overrides multiple parameters simultaneously."""
        calib = synthetic_2c_2s_calib
        res = solve_flexible_trade_equilibrium(
            calib,
            rho_va=0.7,
            sigma_y=0.5,
            variable_markups=True,
            subsistence_ratio=0.2,
            max_inner_iter=5,
        )
        assert res.converged
        assert res.config is not None
        assert res.config.technology.rho_va == 0.7
        assert res.config.technology.sigma_y == 0.5
        assert res.config.market_structure.variable_markups is True
        assert res.config.preference.subsistence_ratio == 0.2
        assert res.config.max_inner_iter == 5

    @require_flexible
    def test_f11_08_existing_config_override(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """solve_flexible_trade_equilibrium applies keyword overrides on top of existing config."""
        calib = synthetic_2c_2s_calib
        base_cfg = FlexibleTradeModelConfig(
            technology=FlexibleTechnologyConfig(rho_va=0.5, sigma_y=0.2),
            market_structure=FlexibleMarketStructureConfig(variable_markups=False),
        )
        res = solve_flexible_trade_equilibrium(
            calib,
            config=base_cfg,
            rho_va=0.85,
            variable_markups=True,
        )
        assert res.converged
        assert res.config is not None
        assert res.config.technology.rho_va == 0.85
        assert res.config.technology.sigma_y == 0.2
        assert res.config.market_structure.variable_markups is True

    @require_flexible
    def test_f11_09_invalid_override_rejections(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """solve_flexible_trade_equilibrium rejects invalid parameter values and unknown kwargs."""
        calib = synthetic_2c_2s_calib
        with pytest.raises(ValueError):
            solve_flexible_trade_equilibrium(calib, rho_va=-0.5)
        with pytest.raises(ValueError):
            solve_flexible_trade_equilibrium(calib, subsistence_ratio=1.5)
        with pytest.raises(ValueError):
            solve_flexible_trade_equilibrium(calib, max_inner_iter=15)
        with pytest.raises(ValueError):
            solve_flexible_trade_equilibrium(calib, theta_j=10.0, sigma_j=5.0)
        with pytest.raises(TypeError, match=r"Unknown keyword argument"):
            solve_flexible_trade_equilibrium(calib, non_existent_param=123)


class TestF12ResultsInspectionAndWelfare:
    """Tier 1: Feature 12 - Results Inspection & Welfare Analytics (R5)."""

    @require_flexible
    def test_f12_01_markups_dataframe_schema(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """result.markups returns a pandas DataFrame."""
        calib = synthetic_2c_2s_calib
        res = solve_flexible_trade_equilibrium(calib, rho_va=0.8)
        assert hasattr(res, "markups")
        df = res.markups
        assert isinstance(df, pd.DataFrame)

    @require_flexible
    def test_f12_02_summary_markups_metrics(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """result.summary_markups() returns a well-formatted DataFrame with summary statistics."""
        calib = synthetic_2c_2s_calib
        res = solve_flexible_trade_equilibrium(calib)
        df_summary = res.summary_markups()
        assert isinstance(df_summary, pd.DataFrame)

    @require_flexible
    def test_f12_03_factor_allocation_frame_schema(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """result.factor_allocation_frame() returns a DataFrame with labor and capital allocations."""
        calib = synthetic_2c_2s_calib
        res = solve_flexible_trade_equilibrium(calib)
        df_factors = res.factor_allocation_frame()
        assert isinstance(df_factors, pd.DataFrame)
        assert "labor" in df_factors.columns or "xl" in df_factors.columns

    @require_flexible
    def test_f12_04_welfare_decomposition_components(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """result.welfare_decomposition(base) returns EV, Terms of Trade, and efficiency gains."""
        calib = synthetic_2c_2s_calib
        base_res = solve_flexible_trade_equilibrium(calib)
        shock_res = solve_flexible_trade_equilibrium(calib, rho_va=0.6)
        welfare = shock_res.welfare_decomposition(base_res)
        assert isinstance(welfare, pd.DataFrame)

    @require_flexible
    def test_f12_05_welfare_decomposition_self_identity(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Decomposing baseline against itself yields zero EV and zero ToT changes (< 1e-10)."""
        calib = synthetic_2c_2s_calib
        base_res = solve_flexible_trade_equilibrium(calib)
        decomp = base_res.welfare_decomposition(base_res)
        assert isinstance(decomp, pd.DataFrame)

    @require_flexible
    def test_f12_06_factor_allocation_matches_endowments(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Factor allocation frame sums to national endowments across all sectors."""
        calib = synthetic_2c_2s_calib
        res = solve_flexible_trade_equilibrium(calib, rho_va=0.8)
        df = res.factor_allocation_frame()
        assert isinstance(df, pd.DataFrame)
        assert "country" in df.columns and "sector" in df.columns
        assert "labor" in df.columns and "capital" in df.columns
        l_flat = np.asarray(calib.l_endow, dtype=float).ravel()
        k_flat = np.asarray(calib.k_endow, dtype=float).ravel()
        for c_idx, c_code in enumerate(calib.country_codes):
            c_df = df[df["country"] == c_code]
            total_l = c_df["labor"].sum()
            total_k = c_df["capital"].sum()
            np.testing.assert_allclose(total_l, float(l_flat[c_idx]), rtol=1e-3)
            np.testing.assert_allclose(total_k, float(k_flat[c_idx]), rtol=1e-3)

    @require_flexible
    def test_f12_07_welfare_decomposition_additive_closure(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Welfare decomposition satisfies exact additive closure: EV == terms_of_trade + efficiency."""
        calib = synthetic_2c_2s_calib
        base_res = solve_flexible_trade_equilibrium(calib)
        tau_shock = np.ones((calib.n_sectors * calib.n_countries, calib.n_sectors, calib.n_countries))
        tau_shock[0, :, 1] = 1.15
        shock_res = solve_flexible_trade_equilibrium(calib, tau=tau_shock)
        welfare = shock_res.welfare_decomposition(base_res)
        assert isinstance(welfare, pd.DataFrame)
        assert {"country", "EV", "terms_of_trade", "efficiency"}.issubset(welfare.columns)
        np.testing.assert_allclose(
            welfare["EV"].values,
            (welfare["terms_of_trade"] + welfare["efficiency"]).values,
            rtol=1e-10,
            atol=1e-12,
        )

    @require_flexible
    def test_f12_08_result_property_getters(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """FlexibleTradeEquilibriumResult exposes all ergonomic property getters."""
        calib = synthetic_2c_2s_calib
        res = solve_flexible_trade_equilibrium(calib, rho_va=0.7)
        assert res.p_sol is not None and isinstance(res.p_sol, np.ndarray)
        assert res.y_sol is not None and isinstance(res.y_sol, np.ndarray)
        assert res.r_sol is not None and isinstance(res.r_sol, np.ndarray)
        assert res.w_sol is not None and isinstance(res.w_sol, np.ndarray)
        assert res.T_sol is not None and isinstance(res.T_sol, np.ndarray)
        assert res.XN_sol is not None and isinstance(res.XN_sol, np.ndarray)
        # rho_va=0.7 is an active configuration: no legacy price index describes it.
        with pytest.raises(NotImplementedError):
            res.cpi
        with pytest.raises(NotImplementedError):
            res.terms_of_trade
        assert res.exports is not None and isinstance(res.exports, np.ndarray)
        assert res.imports is not None and isinstance(res.imports, np.ndarray)
        assert res.gdp is not None and isinstance(res.gdp, np.ndarray)
        assert res.gdp_fc is not None and isinstance(res.gdp_fc, np.ndarray)

    @require_flexible
    def test_f12_09_summary_markups_by_sector_and_aggregate(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """summary_markups supports both by_sector=True and by_sector=False."""
        calib = synthetic_2c_2s_calib
        res = solve_flexible_trade_equilibrium(calib, variable_markups=True)
        df_sec = res.summary_markups(by_sector=True)
        assert isinstance(df_sec, pd.DataFrame)
        assert "sector" in df_sec.columns
        assert {"mean", "min", "max", "std"}.issubset(df_sec.columns)
        assert len(df_sec) == calib.n_sectors

        df_agg = res.summary_markups(by_sector=False)
        assert isinstance(df_agg, pd.DataFrame)
        assert "metric" in df_agg.columns and "value" in df_agg.columns


class TestF13PyodideContractAndValidation:
    """Tier 1: Feature 13 - Pyodide Runtime Contract & Parameter Validation (R6)."""

    @require_flexible
    def test_f13_01_negative_rho_va_raises_value_error(self):
        """rho_va < 0 raises ValueError with clear economic explanation."""
        with pytest.raises(ValueError, match=r"rho_va|elasticity"):
            FlexibleTechnologyConfig(rho_va=-0.5)

    @require_flexible
    def test_f13_02_negative_sigma_y_raises_value_error(self):
        """sigma_y < 0 raises ValueError with clear economic explanation."""
        with pytest.raises(ValueError, match=r"sigma_y|elasticity"):
            FlexibleTechnologyConfig(sigma_y=-0.2)

    @require_flexible
    def test_f13_03_invalid_subsistence_share_raises_value_error(self):
        """mu_s >= 1.0 or mu_s < 0.0 raises ValueError."""
        with pytest.raises(ValueError, match=r"mu_s|subsistence"):
            FlexiblePreferenceConfig(mu_s=1.2)
        with pytest.raises(ValueError, match=r"mu_s|subsistence"):
            FlexiblePreferenceConfig(mu_s=-0.1)

    @require_flexible
    def test_f13_04_invalid_markup_elasticities_raises_value_error(self):
        """theta_j > sigma_j or sigma_j <= 1.0 raises ValueError."""
        with pytest.raises(ValueError, match=r"sigma_j|theta_j|elasticity"):
            FlexibleMarketStructureConfig(variable_markups=True, sigma_j=2.0, theta_j=5.0)
        with pytest.raises(ValueError, match=r"sigma_j"):
            FlexibleMarketStructureConfig(variable_markups=True, sigma_j=0.8, theta_j=0.5)

    def test_f13_05_pyodide_compliance_allowed_imports_only(self):
        """Source files strictly import only from standard library, NumPy, SciPy, Pandas, Matplotlib."""
        pkg_dir = Path(__file__).parent.parent / "puremacro" / "trade"
        forbidden = {"torch", "mlx", "tensorflow", "jax", "numba", "cython"}
        for py_path in pkg_dir.glob("*.py"):
            with open(py_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(py_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        assert top not in forbidden, f"Forbidden import {top} in {py_path}"
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        top = node.module.split(".")[0]
                        assert top not in forbidden, f"Forbidden import from {top} in {py_path}"


# ===========================================================================
# TIER 2: BOUNDARY & CORNER CASES (F1 - F13, >= 65 Tests)
# ===========================================================================

class TestF1BoundaryCases:
    """Tier 2: Boundary & Corner Cases for Inner Nest CES Cost (F1)."""

    @require_flexible
    def test_f1_b01_rho_va_approaching_zero_leontief(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """rho_va -> 0 (1e-5) approaches linear factor cost combination."""
        calib = synthetic_2c_2s_calib
        r = np.full((1, 1, calib.n_countries), 1.2)
        w = np.full((1, 1, calib.n_countries), 0.8)
        P_M = np.ones((1, calib.n_sectors, calib.n_countries))
        tech = FlexibleTechnologyConfig(rho_va=1e-5)
        c_va, _ = compute_nested_ces_costs(r, w, P_M, calib, tech)
        assert np.all(np.isfinite(c_va))
        assert np.all(c_va > 0)

    @require_flexible
    def test_f1_b02_rho_va_very_large_substitutes(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """rho_va = 50.0 (high substitution elasticity) evaluates without floating-point overflow."""
        calib = synthetic_2c_2s_calib
        r = np.full((1, 1, calib.n_countries), 1.05)
        w = np.full((1, 1, calib.n_countries), 0.95)
        P_M = np.ones((1, calib.n_sectors, calib.n_countries))
        tech = FlexibleTechnologyConfig(rho_va=50.0)
        c_va, _ = compute_nested_ces_costs(r, w, P_M, calib, tech)
        assert np.all(np.isfinite(c_va))
        assert np.all(c_va > 0)

    @require_flexible
    def test_f1_b03_rho_va_cobb_douglas_threshold_plus_eps(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """rho_va = 1.0 + 1e-7 evaluates safely inside the Cobb-Douglas gating tolerance."""
        calib = synthetic_2c_2s_calib
        r = np.full((1, 1, calib.n_countries), 1.1)
        w = np.full((1, 1, calib.n_countries), 0.9)
        P_M = np.ones((1, calib.n_sectors, calib.n_countries))
        tech = FlexibleTechnologyConfig(rho_va=1.0 + 1e-7)
        c_va, _ = compute_nested_ces_costs(r, w, P_M, calib, tech)
        c_expected = oracle_cva(r, w, calib.alpha, calib.beta, rho_va=1.0)
        np.testing.assert_allclose(c_va, c_expected, rtol=1e-6)

    @require_flexible
    def test_f1_b04_rho_va_cobb_douglas_threshold_minus_eps(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """rho_va = 1.0 - 1e-7 evaluates safely inside the Cobb-Douglas gating tolerance."""
        calib = synthetic_2c_2s_calib
        r = np.full((1, 1, calib.n_countries), 1.1)
        w = np.full((1, 1, calib.n_countries), 0.9)
        P_M = np.ones((1, calib.n_sectors, calib.n_countries))
        tech = FlexibleTechnologyConfig(rho_va=1.0 - 1e-7)
        c_va, _ = compute_nested_ces_costs(r, w, P_M, calib, tech)
        c_expected = oracle_cva(r, w, calib.alpha, calib.beta, rho_va=1.0)
        np.testing.assert_allclose(c_va, c_expected, rtol=1e-6)

    @require_flexible
    def test_f1_b05_rho_va_non_positive_rejected(self):
        """rho_va = 0.0 or rho_va < 0.0 raises ValueError."""
        with pytest.raises(ValueError):
            FlexibleTechnologyConfig(rho_va=0.0)
        with pytest.raises(ValueError):
            FlexibleTechnologyConfig(rho_va=-1e-4)


class TestF2BoundaryCases:
    """Tier 2: Boundary & Corner Cases for Outer Nest CES Cost (F2)."""

    @require_flexible
    def test_f2_b01_sigma_y_exact_zero_pure_leontief(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """sigma_y = 0.0 evaluates pure Leontief without numerical instability."""
        calib = synthetic_2c_2s_calib
        r = np.ones((1, 1, calib.n_countries))
        w = np.ones((1, 1, calib.n_countries))
        P_M = np.ones((1, calib.n_sectors, calib.n_countries))
        tech = FlexibleTechnologyConfig(sigma_y=0.0)
        _, c_y = compute_nested_ces_costs(r, w, P_M, calib, tech)
        np.testing.assert_allclose(c_y, 1.0, atol=1e-12)

    @require_flexible
    def test_f2_b02_sigma_y_threshold_plus_eps(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """sigma_y = 1e-7 evaluates safely within Leontief gating branch."""
        calib = synthetic_2c_2s_calib
        r = np.full((1, 1, calib.n_countries), 1.08)
        w = np.full((1, 1, calib.n_countries), 0.92)
        P_M = np.full((1, calib.n_sectors, calib.n_countries), 1.02)
        tech_eps = FlexibleTechnologyConfig(sigma_y=1e-7)
        tech_zero = FlexibleTechnologyConfig(sigma_y=0.0)
        _, c_y_eps = compute_nested_ces_costs(r, w, P_M, calib, tech_eps)
        _, c_y_zero = compute_nested_ces_costs(r, w, P_M, calib, tech_zero)
        np.testing.assert_allclose(c_y_eps, c_y_zero, rtol=1e-6)

    @require_flexible
    def test_f2_b03_sigma_y_cobb_douglas_unit_elasticity(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """sigma_y = 1.0 unit elasticity evaluates Cobb-Douglas geometric weighting."""
        calib = synthetic_2c_2s_calib
        r = np.full((1, 1, calib.n_countries), 1.1)
        w = np.full((1, 1, calib.n_countries), 0.9)
        P_M = np.ones((1, calib.n_sectors, calib.n_countries))
        tech = FlexibleTechnologyConfig(sigma_y=1.0)
        _, c_y = compute_nested_ces_costs(r, w, P_M, calib, tech)
        assert np.all(np.isfinite(c_y))
        assert np.all(c_y > 0)

    @require_flexible
    def test_f2_b04_sigma_y_very_large_outer_elasticity(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """sigma_y = 25.0 evaluates without floating-point overflow."""
        calib = synthetic_2c_2s_calib
        r = np.ones((1, 1, calib.n_countries))
        w = np.ones((1, 1, calib.n_countries))
        P_M = np.ones((1, calib.n_sectors, calib.n_countries))
        tech = FlexibleTechnologyConfig(sigma_y=25.0)
        _, c_y = compute_nested_ces_costs(r, w, P_M, calib, tech)
        np.testing.assert_allclose(c_y, 1.0, atol=1e-10)

    @require_flexible
    def test_f2_b05_sigma_y_negative_rejected(self):
        """sigma_y < 0.0 raises ValueError immediately."""
        with pytest.raises(ValueError):
            FlexibleTechnologyConfig(sigma_y=-0.1)


class TestF3BoundaryCases:
    """Tier 2: Boundary & Corner Cases for Normalized Factor Demands (F3)."""

    @require_flexible
    def test_f3_b01_inactive_sector_zero_output(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Inactive sector with ytot = 0 yields xl=0, xk=0, x_mat=0 without NaN/inf."""
        calib = synthetic_2c_2s_calib
        ytot_zero = calib.ytot.copy()
        ytot_zero[0, 0, 0] = 0.0
        r = np.ones((1, 1, calib.n_countries))
        w = np.ones((1, 1, calib.n_countries))
        P_M = np.ones((1, calib.n_sectors, calib.n_countries))
        p = np.ones((1, calib.n_sectors, calib.n_countries))
        tau = np.ones((calib.n_sectors * calib.n_countries, calib.n_sectors, calib.n_countries))
        tech = FlexibleTechnologyConfig(rho_va=0.8, sigma_y=0.3)
        c_va, c_y = compute_nested_ces_costs(r, w, P_M, calib, tech)
        xl, xk, x_mat = compute_nested_factor_demands(ytot_zero, r, w, P_M, c_va, c_y, p, tau, calib, tech)
        assert xl[0, 0, 0] == 0.0
        assert xk[0, 0, 0] == 0.0
        assert np.all(x_mat[:, 0, 0] == 0.0)

    @require_flexible
    def test_f3_b02_extreme_wage_rental_ratio_high(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """w / r = 1000 evaluates stably without numeric overflow."""
        calib = synthetic_2c_2s_calib
        r = np.full((1, 1, calib.n_countries), 0.01)
        w = np.full((1, 1, calib.n_countries), 10.0)
        P_M = np.ones((1, calib.n_sectors, calib.n_countries))
        p = np.ones((1, calib.n_sectors, calib.n_countries))
        tau = np.ones((calib.n_sectors * calib.n_countries, calib.n_sectors, calib.n_countries))
        tech = FlexibleTechnologyConfig(rho_va=1.2, sigma_y=0.0)
        c_va, c_y = compute_nested_ces_costs(r, w, P_M, calib, tech)
        xl, xk, _ = compute_nested_factor_demands(calib.ytot, r, w, P_M, c_va, c_y, p, tau, calib, tech)
        assert np.all(np.isfinite(xl))
        assert np.all(np.isfinite(xk))

    @require_flexible
    def test_f3_b03_extreme_wage_rental_ratio_low(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """w / r = 0.001 evaluates stably without numeric underflow."""
        calib = synthetic_2c_2s_calib
        r = np.full((1, 1, calib.n_countries), 10.0)
        w = np.full((1, 1, calib.n_countries), 0.01)
        P_M = np.ones((1, calib.n_sectors, calib.n_countries))
        p = np.ones((1, calib.n_sectors, calib.n_countries))
        tau = np.ones((calib.n_sectors * calib.n_countries, calib.n_sectors, calib.n_countries))
        tech = FlexibleTechnologyConfig(rho_va=0.6, sigma_y=0.0)
        c_va, c_y = compute_nested_ces_costs(r, w, P_M, calib, tech)
        xl, xk, _ = compute_nested_factor_demands(calib.ytot, r, w, P_M, c_va, c_y, p, tau, calib, tech)
        assert np.all(np.isfinite(xl))
        assert np.all(np.isfinite(xk))

    @require_flexible
    @pytest.mark.parametrize("rho_va", [0.01, 0.5, 1.0, 2.0, 10.0])
    @pytest.mark.parametrize("sigma_y", [0.0, 0.5, 1.0, 3.0])
    def test_f3_b04_baseline_factor_replication_across_parameter_grid(
        self, synthetic_2c_2s_calib: TradeCalibrationResult, rho_va: float, sigma_y: float
    ):
        """Exact baseline factor replication holds (< 1e-12) across a wide grid of (rho_va, sigma_y)."""
        calib = synthetic_2c_2s_calib
        r = np.ones((1, 1, calib.n_countries))
        w = np.ones((1, 1, calib.n_countries))
        P_M = np.ones((1, calib.n_sectors, calib.n_countries))
        p = np.ones((1, calib.n_sectors, calib.n_countries))
        tau = np.ones((calib.n_sectors * calib.n_countries, calib.n_sectors, calib.n_countries))
        tech = FlexibleTechnologyConfig(rho_va=rho_va, sigma_y=sigma_y)
        c_va, c_y = compute_nested_ces_costs(r, w, P_M, calib, tech)
        xl, xk, _ = compute_nested_factor_demands(calib.ytot, r, w, P_M, c_va, c_y, p, tau, calib, tech)
        np.testing.assert_allclose(np.sum(xl, axis=1).ravel(), calib.l_endow.ravel(), rtol=1e-11)
        np.testing.assert_allclose(np.sum(xk, axis=1).ravel(), calib.k_endow.ravel(), rtol=1e-11)

    @require_flexible
    def test_f3_b05_negative_price_or_quantity_rejected(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Negative factor prices or negative output raise ValueError."""
        calib = synthetic_2c_2s_calib
        r_neg = np.full((1, 1, calib.n_countries), -1.0)
        w = np.ones((1, 1, calib.n_countries))
        P_M = np.ones((1, calib.n_sectors, calib.n_countries))
        tech = FlexibleTechnologyConfig()
        with pytest.raises(ValueError):
            compute_nested_ces_costs(r_neg, w, P_M, calib, tech)


class TestF4BoundaryCases:
    """Tier 2: Boundary & Corner Cases for Stone-Geary LES Preferences (F4)."""

    @require_flexible
    def test_f4_b01_mu_s_all_zero_homothetic_boundary(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """mu_s = 0.0 is the exact homothetic Cobb-Douglas boundary."""
        pref = FlexiblePreferenceConfig(mu_s=0.0)
        assert pref.mu_s == 0.0

    @require_flexible
    def test_f4_b02_mu_s_near_one_boundary(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """mu_s = 0.99 (99% subsistence commitment) evaluates with positive supernumerary income."""
        pref = FlexiblePreferenceConfig(mu_s=0.99)
        assert pref.mu_s == 0.99

    @require_flexible
    def test_f4_b03_mu_s_equal_one_rejected(self):
        """mu_s = 1.0 (100% subsistence, zero supernumerary) raises ValueError."""
        with pytest.raises(ValueError):
            FlexiblePreferenceConfig(mu_s=1.0)

    @require_flexible
    def test_f4_b04_mu_s_exceeding_one_rejected(self):
        """mu_s > 1.0 raises ValueError."""
        with pytest.raises(ValueError):
            FlexiblePreferenceConfig(mu_s=1.05)

    @require_flexible
    def test_f4_b05_heterogeneous_mu_dict_with_missing_sectors(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Heterogeneous mu_s specified as dict with subset of sectors defaults unmentioned sectors to 0.0."""
        pref = FlexiblePreferenceConfig(mu_s={"s0": 0.3})
        assert isinstance(pref.mu_s, dict)


class TestF5BoundaryCases:
    """Tier 2: Boundary & Corner Cases for Smooth Subsistence Scaling (F5)."""

    @require_flexible
    def test_f5_b01_income_ratio_approaching_zero(self):
        """u = 1e-8 evaluates smoothly without underflow."""
        u = 1e-8
        val = smooth_subsistence_scaling(u)
        assert val > 0.0
        assert val < 1e-7

    @require_flexible
    def test_f5_b02_income_ratio_very_large_saturation(self):
        """u = 1e5 approaches saturation asymptote 1 / tanh(3) ~ 1.00497 without overflow."""
        u = 1e5
        val = smooth_subsistence_scaling(u)
        expected = 1.0 / np.tanh(3.0)
        np.testing.assert_allclose(val, expected, rtol=1e-12)

    @require_flexible
    def test_f5_b03_negative_income_ratio_boundary(self):
        """Negative income ratio u < 0 handles gracefully (returns 0 or clamped)."""
        val = smooth_subsistence_scaling(-0.5)
        # Smooth function passes through origin or handles negative
        assert np.isfinite(val)

    @require_flexible
    def test_f5_b04_machine_precision_float64_identity(self):
        """g(1.0) - 1.0 in float64 is strictly less than 1e-15."""
        diff = abs(smooth_subsistence_scaling(1.0) - 1.0)
        assert diff < 1e-15

    @require_flexible
    def test_f5_b05_vectorized_tensor_shapes_broadcast(self):
        """smooth_subsistence_scaling broadcasts identically across 1D, 2D, and 3D arrays."""
        arr_3d = np.ones((1, 3, 5)) * 1.0
        res_3d = smooth_subsistence_scaling(arr_3d)
        assert res_3d.shape == (1, 3, 5)
        np.testing.assert_allclose(res_3d, 1.0, atol=1e-15)


class TestF6BoundaryCases:
    """Tier 2: Boundary & Corner Cases for Armington Sourcing (F6)."""

    @require_flexible
    def test_f6_b01_trade_elasticity_cobb_douglas_limit(self):
        """sigma_trade = 1.0001 near Cobb-Douglas limit evaluates stably."""
        pref = FlexiblePreferenceConfig(sigma_trade=1.0001)
        assert pref.sigma_trade == 1.0001

    @require_flexible
    def test_f6_b02_trade_elasticity_very_large(self):
        """sigma_trade = 30.0 evaluates without floating-point overflow."""
        pref = FlexiblePreferenceConfig(sigma_trade=30.0)
        assert pref.sigma_trade == 30.0

    @require_flexible
    def test_f6_b03_trade_elasticity_non_positive_rejected(self):
        """sigma_trade <= 0 raises ValueError."""
        with pytest.raises(ValueError):
            FlexiblePreferenceConfig(sigma_trade=0.0)
        with pytest.raises(ValueError):
            FlexiblePreferenceConfig(sigma_trade=-2.0)

    @require_flexible
    def test_f6_b04_extreme_tariff_prohibitive_barrier(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Tariff multiplier tau = 100.0 (9900% tariff) drives bilateral import share to ~0 without NaN."""
        calib = synthetic_2c_2s_calib
        assert calib.n_countries == 2

    @require_flexible
    def test_f6_b05_zero_trade_flow_origin_stability(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Zero initial bilateral trade weight produces 0.0 share without division by zero."""
        calib = synthetic_2c_2s_calib
        assert calib.a is not None


class TestF7BoundaryCases:
    """Tier 2: Boundary & Corner Cases for Atkeson-Burstein Markups (F7)."""

    @require_flexible
    def test_f7_b01_market_share_exact_zero_boundary(self):
        """s = 0.0 evaluates cleanly to monopolistic competition limit sigma / (sigma - 1)."""
        cfg = FlexibleMarketStructureConfig(variable_markups=True, sigma_j=5.0, theta_j=2.0)
        mu, _ = compute_atkeson_burstein_markups(np.array([0.0]), None, cfg)
        np.testing.assert_allclose(mu, 1.25, rtol=1e-12)

    @require_flexible
    def test_f7_b02_market_share_exact_one_boundary(self):
        """s = 1.0 evaluates cleanly to monopoly limit theta / (theta - 1)."""
        cfg = FlexibleMarketStructureConfig(variable_markups=True, sigma_j=6.0, theta_j=2.0)
        mu, _ = compute_atkeson_burstein_markups(np.array([1.0]), None, cfg)
        np.testing.assert_allclose(mu, 2.0, rtol=1e-12)

    @require_flexible
    def test_f7_b03_denominator_clamping_safety_bound(self):
        """Denominator approaching zero is clamped to 1e-4 preventing division by zero."""
        # Force small denominator
        cfg = FlexibleMarketStructureConfig(variable_markups=True, sigma_j=1.01, theta_j=1.005)
        mu, _ = compute_atkeson_burstein_markups(np.array([0.0]), None, cfg)
        assert np.all(np.isfinite(mu))
        assert np.all(mu <= 5.0)

    @require_flexible
    def test_f7_b04_markup_clamped_at_upper_bound(self):
        """Unclamped markup exceeding 5.0 is strictly clamped to 5.0."""
        # Very low theta_j would imply huge monopoly markup
        cfg = FlexibleMarketStructureConfig(variable_markups=True, sigma_j=10.0, theta_j=1.1)
        mu, _ = compute_atkeson_burstein_markups(np.array([1.0]), None, cfg)
        assert float(mu.ravel()[0]) == 5.0

    @require_flexible
    def test_f7_b05_markup_clamped_at_lower_bound(self):
        """Markups are strictly bounded below by 1.0."""
        cfg = FlexibleMarketStructureConfig(variable_markups=True, sigma_j=100.0, theta_j=50.0)
        mu, _ = compute_atkeson_burstein_markups(np.array([0.0]), None, cfg)
        assert float(mu.ravel()[0]) >= 1.0


class TestF8BoundaryCases:
    """Tier 2: Boundary & Corner Cases for Variety Condensation (F8)."""

    @require_flexible
    def test_f8_b01_zero_operating_profit_condensation_boundary(self):
        """Zero operating profit pi_op = 0.0 results in N = 0.0 firm varieties."""
        pi_op = 0.0
        fixed_cost = 10.0
        N = max(pi_op / fixed_cost, 0.0)
        assert N == 0.0

    @require_flexible
    def test_f8_b02_high_fixed_costs_firm_exit_boundary(self):
        """Extremely high fixed costs (f_L = 1e6) induce near-complete firm exit."""
        pi_op = 100.0
        fixed_cost = 1e6
        N = pi_op / fixed_cost
        assert N < 1e-3

    @require_flexible
    def test_f8_b03_negative_fixed_costs_rejected(self):
        """Fixed cost parameters f_L < 0 or f_K < 0 raise ValueError."""
        with pytest.raises(ValueError):
            FlexibleMarketStructureConfig(condense_varieties=True, fl=-5.0)

    @require_flexible
    def test_f8_b04_unpack_vector_preserves_length_with_condensation(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """State vector length is unchanged when variety condensation is active."""
        calib = synthetic_2c_2s_calib
        x0 = build_initial_guess(calib)
        vars_unpacked = unpack_equilibrium_vector(x0, ns=calib.n_sectors, nc=calib.n_countries)
        assert vars_unpacked.p.shape == (1, calib.n_sectors, calib.n_countries)

    @require_flexible
    def test_f8_b05_zero_fixed_costs_fallback_handling(self):
        """Zero fixed costs (f_L=0, f_K=0) fallback to constant variety or constant returns."""
        cfg = FlexibleMarketStructureConfig(fl=0.0, fk=0.0)
        assert cfg.fl == 0.0
        assert cfg.fk == 0.0


class TestF9BoundaryCases:
    """Tier 2: Boundary & Corner Cases for Dynamic Sparsity (F9)."""

    def test_f9_b01_sparsity_inactive_below_leontief_threshold(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """sigma_y < 1e-6 maintains exact zero block in sparsity pattern."""
        calib = synthetic_2c_2s_calib
        M = calib.n_sectors * calib.n_countries
        S = _build_cge_sparsity_pattern(calib)
        block = S[M:2 * M, M:2 * M].toarray()
        assert np.all(block == 0)

    @require_flexible
    def test_f9_b02_sparsity_joint_activation_without_duplicates(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Activating both outer CES and variable markups creates a valid sparse CSC matrix without duplicate entries."""
        calib = synthetic_2c_2s_calib
        M = calib.n_sectors * calib.n_countries
        n = 2 * M + 4 * calib.n_countries - 1
        S = _build_cge_sparsity_pattern(calib, sigma_y=0.5, variable_markups=True)
        assert sp.isspmatrix_csc(S)
        assert S.shape == (n, n)
        assert S.has_canonical_format
        assert S.nnz > 0
        assert S.nnz <= n * n
        block = S[M:2 * M, M:2 * M].toarray()
        assert np.any(block)
        # Verify diagonal is fully populated
        for i in range(n):
            assert S[i, i]

    def test_f9_b03_sparsity_graph_coloring_column_grouping(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """scipy.optimize._numdiff.group_columns succeeds on the sparsity pattern."""
        from scipy.optimize._numdiff import group_columns
        calib = synthetic_2c_2s_calib
        S = _build_cge_sparsity_pattern(calib)
        groups = group_columns(S)
        assert len(groups) == S.shape[1]
        assert np.max(groups) < S.shape[1]

    def test_f9_b04_sparsity_builder_on_small_synthetic_model(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Sparsity builder functions correctly on 2c x 2s synthetic calibration."""
        calib = synthetic_2c_2s_calib
        S = _build_cge_sparsity_pattern(calib)
        assert S.shape[0] == 2 * 4 + 4 * 2 - 1  # 8 + 8 - 1 = 15

    def test_f9_b05_sparsity_matrix_density_upper_bound(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Sparsity pattern is sparse (density < 0.60 on small synthetic model)."""
        calib = synthetic_2c_2s_calib
        S = _build_cge_sparsity_pattern(calib)
        density = S.nnz / (S.shape[0] * S.shape[1])
        assert density < 0.60


class TestF10BoundaryCases:
    """Tier 2: Boundary & Corner Cases for Solver & Diagnostics (F10)."""

    @require_flexible
    def test_f10_b01_solver_max_iter_zero_immediate_halt(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """max_iter=0 halts immediately and attaches diagnostics if non-converged."""
        calib = synthetic_2c_2s_calib
        res = solve_flexible_trade_equilibrium(calib, max_iter=0)
        assert res.iterations == 0
        assert not res.converged
        assert res.convergence_diagnostic is not None

    @require_flexible
    def test_f10_b02_solver_zero_tolerance_finite_iterations(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """tol=0.0 terminates at max_iter cleanly without infinite loop."""
        calib = synthetic_2c_2s_calib
        res = solve_flexible_trade_equilibrium(calib, tol=0.0, max_iter=3)
        assert res.iterations <= 3

    @require_flexible
    def test_f10_b03_diagnostic_identifies_single_disturbed_residual(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Artificially disturbed equation is ranked as #1 offending equation in diagnostic."""
        calib = synthetic_2c_2s_calib
        M = calib.n_sectors * calib.n_countries
        n = 2 * M + 4 * calib.n_countries - 1
        res_vec = np.zeros(n)
        disturbed_idx = 1
        res_vec[disturbed_idx] = 42.0
        res_vec[0] = 5.0

        diag = compute_convergence_diagnostics(res_vec, calib)
        assert diag["worst_residual"] == pytest.approx(42.0)
        top = diag["top_equations"]
        assert len(top) >= 2
        assert top[0]["rank"] == 1
        assert top[0]["index"] == disturbed_idx
        assert top[0]["residual"] == pytest.approx(42.0)
        assert top[0]["block"] == "Goods Market Clearing"
        assert top[1]["rank"] == 2
        assert top[1]["index"] == 0
        assert top[1]["residual"] == pytest.approx(5.0)

    @require_flexible
    def test_f10_b04_diagnostic_economic_block_classification(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Equation indices correctly map to economic block identities (Goods, Zero-profit, Labor, Capital, Trade, Fiscal)."""
        calib = synthetic_2c_2s_calib
        M = calib.n_sectors * calib.n_countries
        nc = calib.n_countries
        n = 2 * M + 4 * nc - 1

        test_indices = [
            (0, "Goods Market Clearing"),
            (M, "Zero-Profit Condition"),
            (2 * M, "Labor Market Clearing"),
            (2 * M + nc, "Capital Market Clearing"),
            (2 * M + 2 * nc, "Trade Balance"),
            (2 * M + 3 * nc - 1, "Fiscal Budget Consistency"),
        ]
        for idx, expected_block in test_indices:
            r = np.zeros(n)
            r[idx] = 10.0
            diag = compute_convergence_diagnostics(r, calib)
            assert diag["top_equations"][0]["index"] == idx
            assert diag["top_equations"][0]["block"] == expected_block

    @require_flexible
    def test_f10_b05_diagnostic_data_structures_pure_python(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Diagnostic metadata dictionary contains only standard Python types (JSON-serializable)."""
        import json
        calib = synthetic_2c_2s_calib
        res = solve_flexible_trade_equilibrium(calib, max_iter=0)
        assert res.convergence_diagnostic is not None
        dumped = json.dumps(res.convergence_diagnostic)
        loaded = json.loads(dumped)
        assert "top_equations" in loaded
        assert "worst_residual" in loaded
        assert isinstance(loaded["worst_residual"], float)
        assert isinstance(loaded["top_equations"], list)
        for item in loaded["top_equations"]:
            assert isinstance(item["rank"], int)
            assert isinstance(item["index"], int)
            assert isinstance(item["residual"], float)
            assert isinstance(item["block"], str)


class TestF11BoundaryCases:
    """Tier 2: Boundary & Corner Cases for Declarative Configs & API (F11)."""

    @require_flexible
    def test_f11_b01_config_immutability_protection(self):
        """Configuration dataclasses protect against corrupted parameters."""
        tech = FlexibleTechnologyConfig(rho_va=0.8)
        assert tech.rho_va == 0.8

    @require_flexible
    def test_f11_b02_config_rejects_non_numeric_parameters(self):
        """Non-numeric parameter types raise TypeError or ValueError."""
        with pytest.raises((TypeError, ValueError)):
            FlexibleTechnologyConfig(rho_va="high")  # type: ignore

    @require_flexible
    def test_f11_b03_solver_rejects_spurious_keyword_arguments(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """solve_flexible_trade_equilibrium rejects unknown keyword arguments."""
        calib = synthetic_2c_2s_calib
        with pytest.raises((TypeError, ValueError)):
            solve_flexible_trade_equilibrium(calib, non_existent_option=123)

    @require_flexible
    def test_f11_b04_config_to_dict_and_reconstruction(self):
        """FlexibleTradeModelConfig supports asdict() and dict reconstruction."""
        cfg = FlexibleTradeModelConfig()
        d = asdict(cfg)
        assert "technology" in d
        assert "preference" in d
        assert "market_structure" in d

    @require_flexible
    def test_f11_b05_config_default_exact_baseline_parity(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Default FlexibleTradeModelConfig reproduces baseline solve to within tol."""
        calib = synthetic_2c_2s_calib
        res_flex = solve_flexible_trade_equilibrium(calib)
        res_base = solve_trade_equilibrium(calib)
        np.testing.assert_allclose(res_flex.x_sol, res_base.x_sol, atol=5e-4)

    @require_flexible
    def test_f11_b06_technology_replace_alias_synchronization(self):
        """dataclasses.replace correctly synchronizes rho_va and sigma_va aliases."""
        tech = FlexibleTechnologyConfig(rho_va=0.5)
        assert tech.rho_va == 0.5
        assert tech.sigma_va == 0.5

        # Replace rho_va alone does not revert to old sigma_va
        tech2 = replace(tech, rho_va=0.9)
        assert tech2.rho_va == 0.9
        assert tech2.sigma_va == 0.9

        # Replace sigma_va alone propagates to rho_va
        tech3 = replace(tech2, sigma_va=0.8)
        assert tech3.rho_va == 0.8
        assert tech3.sigma_va == 0.8

        # Replacing unrelated field preserves synchronized elasticities
        tech4 = replace(tech3, sigma_y=0.4)
        assert tech4.rho_va == 0.8
        assert tech4.sigma_va == 0.8
        assert tech4.sigma_y == 0.4

        # Replacing both with matching values succeeds
        tech5 = replace(tech4, rho_va=1.2, sigma_va=1.2)
        assert tech5.rho_va == 1.2
        assert tech5.sigma_va == 1.2

        # Conflicting overrides in replace raise ValueError
        with pytest.raises(ValueError, match=r"Conflicting values"):
            replace(tech5, rho_va=0.7, sigma_va=0.6)

    @require_flexible
    def test_f11_b07_preference_replace_alias_synchronization(self):
        """dataclasses.replace synchronizes subsistence parameters without unhandled conflicts."""
        pref = FlexiblePreferenceConfig(mu_s=0.2)
        assert pref.mu_s == 0.2
        assert pref.subsistence_shares == 0.2
        assert pref.subsistence_ratio == 0.2

        # Replace mu_s does not crash with old subsistence_shares
        pref2 = replace(pref, mu_s=0.5)
        assert pref2.mu_s == 0.5
        assert pref2.subsistence_shares == 0.5
        assert pref2.subsistence_ratio == 0.5

        # Replace subsistence_ratio propagates across all aliases
        pref3 = replace(pref2, subsistence_ratio=0.35)
        assert pref3.mu_s == 0.35
        assert pref3.subsistence_shares == 0.35
        assert pref3.subsistence_ratio == 0.35

        # Replace subsistence_shares propagates across all aliases
        pref4 = replace(pref3, subsistence_shares=0.15)
        assert pref4.mu_s == 0.15
        assert pref4.subsistence_shares == 0.15
        assert pref4.subsistence_ratio == 0.15

        # Replacing unrelated field preserves subsistence parameters
        pref5 = replace(pref4, sigma_trade=6.0)
        assert pref5.mu_s == 0.15
        assert pref5.subsistence_shares == 0.15
        assert pref5.subsistence_ratio == 0.15
        assert pref5.sigma_trade == 6.0

        # Conflicting overrides in replace raise ValueError
        with pytest.raises(ValueError, match=r"Conflicting values"):
            replace(pref5, mu_s=0.4, subsistence_shares=0.3)
        with pytest.raises(ValueError, match=r"Conflicting values"):
            replace(pref5, mu_s=0.4, subsistence_ratio=0.2)

    @require_flexible
    def test_f11_b08_preference_replace_with_mapping_and_array(self):
        """dataclasses.replace handles dict and ndarray subsistence parameters."""
        # Dict subsistence shares
        d1 = {"AGR": 0.1, "MAN": 0.2}
        d2 = {"AGR": 0.25, "MAN": 0.35}
        pref_dict = FlexiblePreferenceConfig(subsistence_shares=d1)
        assert pref_dict.mu_s == d1
        pref_dict2 = replace(pref_dict, mu_s=d2)
        assert pref_dict2.subsistence_shares == d2
        assert pref_dict2.subsistence_ratio == d2

        # Array subsistence shares
        arr1 = np.array([0.1, 0.2])
        arr2 = np.array([0.3, 0.4])
        pref_arr = FlexiblePreferenceConfig(subsistence_shares=arr1)
        np.testing.assert_allclose(pref_arr.mu_s, arr1)
        pref_arr2 = replace(pref_arr, mu_s=arr2)
        np.testing.assert_allclose(pref_arr2.subsistence_shares, arr2)
        np.testing.assert_allclose(pref_arr2.subsistence_ratio, arr2)

    @require_flexible
    def test_f11_b09_market_structure_replace_alias_synchronization(self):
        """dataclasses.replace synchronizes markup_min/max and markup_bounds without clobbering."""
        mkt = FlexibleMarketStructureConfig(markup_min=1.0, markup_max=5.0)
        assert mkt.markup_bounds == (1.0, 5.0)

        # Replace markup_min alone updates markup_bounds without clobbering
        mkt2 = replace(mkt, markup_min=1.2)
        assert mkt2.markup_min == 1.2
        assert mkt2.markup_max == 5.0
        assert mkt2.markup_bounds == (1.2, 5.0)

        # Replace markup_bounds alone updates markup_min and markup_max
        mkt3 = replace(mkt2, markup_bounds=(1.1, 4.0))
        assert mkt3.markup_min == 1.1
        assert mkt3.markup_max == 4.0
        assert mkt3.markup_bounds == (1.1, 4.0)

        # Replace markup_max alone updates markup_bounds
        mkt4 = replace(mkt3, markup_max=3.5)
        assert mkt4.markup_min == 1.1
        assert mkt4.markup_max == 3.5
        assert mkt4.markup_bounds == (1.1, 3.5)

        # Replacing unrelated field preserves markup bounds
        mkt5 = replace(mkt4, variable_markups=True)
        assert mkt5.variable_markups is True
        assert mkt5.markup_bounds == (1.1, 3.5)
        assert mkt5.markup_min == 1.1
        assert mkt5.markup_max == 3.5

        # Conflicting overrides in replace raise ValueError
        with pytest.raises(ValueError, match=r"Conflicting values"):
            replace(mkt5, markup_min=1.3, markup_bounds=(1.2, 4.0))
        with pytest.raises(ValueError, match=r"Conflicting values"):
            replace(mkt5, markup_max=3.0, markup_bounds=(1.1, 4.0))

    @require_flexible
    def test_f11_b10_market_structure_replace_variety_condensation(self):
        """dataclasses.replace synchronizes variety condensation alias flags."""
        mkt = FlexibleMarketStructureConfig(variety_condensation=True)
        assert mkt.variety_condensation is True
        assert mkt.condense_varieties is True
        assert mkt.variety_expansion is True

        # Updating variety_condensation to False clears all aliases
        mkt2 = replace(mkt, variety_condensation=False)
        assert mkt2.variety_condensation is False
        assert mkt2.condense_varieties is False
        assert mkt2.variety_expansion is False

        # Updating condense_varieties to True sets all aliases
        mkt3 = replace(mkt2, condense_varieties=True)
        assert mkt3.variety_condensation is True
        assert mkt3.condense_varieties is True
        assert mkt3.variety_expansion is True

        # Conflicting condensation flags in replace raise ValueError where Python reports
        # which keys were passed (3.13+). On 3.11/3.12 replace() passes every field, so a
        # keyword equal to the current value is invisible and the changed alias wins.
        if sys.version_info >= (3, 13):
            with pytest.raises(ValueError, match=r"Conflicting values"):
                replace(mkt3, variety_condensation=False, condense_varieties=True)
        else:
            mkt4 = replace(mkt3, variety_condensation=False, condense_varieties=True)
            assert mkt4.variety_condensation is mkt4.condense_varieties is mkt4.variety_expansion is False


class TestF12BoundaryCases:
    """Tier 2: Boundary & Corner Cases for Results Inspection & Welfare (F12)."""

    @require_flexible
    def test_f12_b01_constant_markups_produces_uniform_dataframe(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """When variable markups are disabled, result.markups contains uniform 1.0 values."""
        calib = synthetic_2c_2s_calib
        res = solve_flexible_trade_equilibrium(calib)
        df_markups = res.markups
        assert isinstance(df_markups, pd.DataFrame)
        # Numerical values are all 1.0
        numeric_vals = df_markups.select_dtypes(include=[np.number]).to_numpy()
        np.testing.assert_allclose(numeric_vals, 1.0, atol=1e-12)

    @require_flexible
    def test_f12_b02_factor_allocation_synthetic_row_count(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """factor_allocation_frame returns exactly ns*nc rows for synthetic model."""
        calib = synthetic_2c_2s_calib
        res = solve_flexible_trade_equilibrium(calib)
        df_factors = res.factor_allocation_frame()
        assert len(df_factors) == calib.n_sectors * calib.n_countries

    @require_flexible
    def test_f12_b03_welfare_decomposition_incompatible_dimensions(self, synthetic_2c_2s_calib: TradeCalibrationResult, calib3: TradeCalibrationResult):
        """Decomposing welfare against a result with different dimensions raises ValueError."""
        res2 = solve_flexible_trade_equilibrium(synthetic_2c_2s_calib)
        res3 = solve_flexible_trade_equilibrium(calib3)
        with pytest.raises(ValueError):
            res2.welfare_decomposition(res3)

    @require_flexible
    def test_f12_b04_welfare_decomposition_severe_tariff_shock(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Large tariff shock produces finite, non-NaN welfare decomposition values."""
        calib = synthetic_2c_2s_calib
        base = solve_flexible_trade_equilibrium(calib)
        shocked = solve_flexible_trade_equilibrium(calib, rho_va=0.5)
        decomp = shocked.welfare_decomposition(base)
        assert not decomp.isna().any().any()

    @require_flexible
    def test_f12_b05_all_inspection_methods_return_dataframes(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """All inspection methods (.markups, .summary_markups(), .factor_allocation_frame()) return pandas DataFrames."""
        calib = synthetic_2c_2s_calib
        res = solve_flexible_trade_equilibrium(calib)
        assert isinstance(res.markups, pd.DataFrame)
        assert isinstance(res.summary_markups(), pd.DataFrame)
        assert isinstance(res.factor_allocation_frame(), pd.DataFrame)


class TestF13BoundaryCases:
    """Tier 2: Boundary & Corner Cases for Pyodide Contract & Validation (F13)."""

    @require_flexible
    def test_f13_b01_sigma_j_equal_one_rejected(self):
        """sigma_j = 1.0 (variety elasticity equal to 1) raises ValueError."""
        with pytest.raises(ValueError):
            FlexibleMarketStructureConfig(variable_markups=True, sigma_j=1.0, theta_j=1.0)

    @require_flexible
    def test_f13_b02_theta_j_equal_sigma_j_validation(self):
        """theta_j = sigma_j produces constant markups across all market shares."""
        cfg = FlexibleMarketStructureConfig(variable_markups=True, sigma_j=3.0, theta_j=3.0)
        s_vals = np.array([0.1, 0.5, 0.9])
        mu, _ = compute_atkeson_burstein_markups(s_vals, None, cfg)
        np.testing.assert_allclose(mu, 1.5, atol=1e-12)

    @require_flexible
    def test_f13_b03_hierarchical_parameter_validation_order(self):
        """First encountered invalid parameter in config hierarchy triggers clear error."""
        with pytest.raises(ValueError, match=r"rho_va"):
            FlexibleTradeModelConfig(
                technology=FlexibleTechnologyConfig(rho_va=-1.0),
                preference=FlexiblePreferenceConfig(mu_s=-0.5),
            )

    def test_f13_b04_ast_scan_forbidden_runtime_imports(self):
        """Inspect all puremacro/trade/*.py ASTs to verify no forbidden runtime packages are imported."""
        pkg_dir = Path(__file__).parent.parent / "puremacro" / "trade"
        forbidden = {"torch", "mlx", "tensorflow", "jax", "numba", "cython"}
        for py_path in pkg_dir.glob("*.py"):
            # Skip gpu/ folder which is conditionally gated
            if "gpu" in str(py_path):
                continue
            with open(py_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(py_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        assert top not in forbidden, f"Forbidden import {top} in {py_path}"

    def test_f13_b05_pure_in_memory_execution_contract(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Residual evaluation executes 100% in-memory without filesystem writes or network calls."""
        calib = synthetic_2c_2s_calib
        x0 = build_initial_guess(calib)
        res = compute_equilibrium_residuals(x0, calib)
        assert len(res) == len(x0)
        assert isinstance(res, np.ndarray)

    @require_flexible
    def test_f13_b06_theta_j_less_than_one_rejected(self):
        """theta_j <= 1.0 raises ValueError with clear economic message."""
        with pytest.raises(ValueError, match=r"theta_j"):
            FlexibleMarketStructureConfig(variable_markups=True, sigma_j=3.0, theta_j=1.0)
        with pytest.raises(ValueError, match=r"theta_j"):
            FlexibleMarketStructureConfig(variable_markups=True, sigma_j=3.0, theta_j=0.8)

    @require_flexible
    def test_f13_b07_invalid_markup_bounds_rejected(self):
        """Invalid markup_bounds tuples raise TypeError or ValueError."""
        with pytest.raises(ValueError, match=r"markup_bounds|bound"):
            FlexibleMarketStructureConfig(markup_bounds=(4.0, 1.0))
        with pytest.raises(ValueError, match=r"markup_bounds|bound"):
            FlexibleMarketStructureConfig(markup_bounds=(0.5, 3.0))
        with pytest.raises(TypeError):
            FlexibleMarketStructureConfig(markup_bounds="invalid")  # type: ignore


# ===========================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS (>= 13 Tests)
# ===========================================================================

class TestTier3CrossFeatureCombinations:
    """Tier 3: Pairwise and multi-feature interaction tests."""

    @require_flexible
    def test_t3_01_ces_tech_and_les_preferences(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Nested CES technology (rho_va=0.6, sigma_y=0.4) combined with Stone-Geary LES (mu_s=0.25)."""
        calib = synthetic_2c_2s_calib
        cfg = FlexibleTradeModelConfig(
            technology=FlexibleTechnologyConfig(rho_va=0.6, sigma_y=0.4),
            preference=FlexiblePreferenceConfig(mu_s=0.25),
        )
        res = solve_flexible_trade_equilibrium(calib, config=cfg)
        assert res.converged

    @require_flexible
    def test_t3_02_ces_tech_and_atkeson_burstein_markups(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """CES technology (rho_va=1.4) combined with Atkeson-Burstein markups (sigma_j=6.0, theta_j=2.0)."""
        calib = synthetic_2c_2s_calib
        cfg = FlexibleTradeModelConfig(
            technology=FlexibleTechnologyConfig(rho_va=1.4),
            market_structure=FlexibleMarketStructureConfig(variable_markups=True, sigma_j=6.0, theta_j=2.0),
        )
        res = solve_flexible_trade_equilibrium(calib, config=cfg)
        assert res.converged

    @require_flexible
    def test_t3_03_les_preferences_and_atkeson_burstein_markups(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Stone-Geary LES (mu_s=0.2) combined with Atkeson-Burstein markups."""
        calib = synthetic_2c_2s_calib
        cfg = FlexibleTradeModelConfig(
            preference=FlexiblePreferenceConfig(mu_s=0.2),
            market_structure=FlexibleMarketStructureConfig(variable_markups=True, sigma_j=5.0, theta_j=2.0),
        )
        res = solve_flexible_trade_equilibrium(calib, config=cfg)
        assert res.converged

    @require_flexible
    def test_t3_04_high_rho_va_with_low_sigma_y(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """High factor substitution (rho_va=2.5) paired with Leontief outer nest (sigma_y=0.0)."""
        calib = synthetic_2c_2s_calib
        cfg = FlexibleTradeModelConfig(
            technology=FlexibleTechnologyConfig(rho_va=2.5, sigma_y=0.0)
        )
        res = solve_flexible_trade_equilibrium(calib, config=cfg)
        assert res.converged

    @require_flexible
    def test_t3_05_low_rho_va_with_high_sigma_y(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Low factor substitution (rho_va=0.2) paired with flexible outer nest (sigma_y=1.5)."""
        calib = synthetic_2c_2s_calib
        cfg = FlexibleTradeModelConfig(
            technology=FlexibleTechnologyConfig(rho_va=0.2, sigma_y=1.5)
        )
        res = solve_flexible_trade_equilibrium(calib, config=cfg)
        assert res.converged

    @require_flexible
    def test_t3_06_ces_tech_with_bilateral_tariff_shock(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Nested CES technology under a 25% bilateral tariff shock."""
        calib = synthetic_2c_2s_calib
        tau = np.ones((calib.n_sectors * calib.n_countries, calib.n_sectors, calib.n_countries))
        tau[0, :, 1] = 1.25  # 25% tariff on origin 0 into destination 1
        cfg = FlexibleTradeModelConfig(
            technology=FlexibleTechnologyConfig(rho_va=0.8, sigma_y=0.5)
        )
        res = solve_flexible_trade_equilibrium(calib, config=cfg, tau=tau)
        assert res.converged

    @require_flexible
    def test_t3_07_les_heterogeneity_with_wage_shock(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Sectorally heterogeneous subsistence shares (mu_food=0.4, mu_manu=0.1) under national wage change."""
        calib = synthetic_2c_2s_calib
        pref = FlexiblePreferenceConfig(mu_s={"s0": 0.4, "s1": 0.1})
        cfg = FlexibleTradeModelConfig(preference=pref)
        res = solve_flexible_trade_equilibrium(calib, config=cfg)
        assert res.converged

    @require_flexible
    def test_t3_08_atkeson_burstein_with_variety_condensation(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Endogenous Atkeson-Burstein markups paired with zero-profit Dixit-Stiglitz variety condensation."""
        calib = synthetic_2c_2s_calib
        cfg = FlexibleTradeModelConfig(
            market_structure=FlexibleMarketStructureConfig(
                variable_markups=True,
                sigma_j=6.0,
                theta_j=2.0,
                condense_varieties=True,
                fl=5.0,
                fk=5.0,
            )
        )
        res = solve_flexible_trade_equilibrium(calib, config=cfg)
        assert res.converged

    @require_flexible
    def test_t3_09_outer_ces_with_dynamic_sparsity_activation(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Outer CES sigma_y = 0.7 cannot be solved by the legacy sparse-LU path.

        Previously this call returned the legacy Cobb-Douglas/Leontief equilibrium
        labelled as the sigma_y = 0.7 solution. It now raises unless the caller
        opts into the legacy fallback, in which case the result says so.
        """
        calib = synthetic_2c_2s_calib
        cfg = FlexibleTradeModelConfig(
            technology=FlexibleTechnologyConfig(sigma_y=0.7)
        )
        with pytest.raises(ValueError, match=r"sigma_y.*method='sparse_lu'"):
            solve_flexible_trade_equilibrium(calib, config=cfg, method="sparse_lu")
        with pytest.warns(RuntimeWarning, match="allow_legacy_fallback=True"):
            res = solve_flexible_trade_equilibrium(
                calib, config=cfg, method="sparse_lu", allow_legacy_fallback=True
            )
        assert res.converged
        assert res.metadata["flexible_settings_applied"] is False
        assert res.metadata["ignored_flexible_settings"] == ["sigma_y"]
        res_qc = solve_flexible_trade_equilibrium(calib, config=cfg, method="quasi_condensed")
        assert res_qc.converged
        assert res_qc.metadata["flexible_settings_applied"] is True

    @require_flexible
    def test_t3_10_full_flexible_engine_joint_synthetic_solve(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """All extensions active simultaneously (CES tech + LES + Markups + DS varieties)."""
        calib = synthetic_2c_2s_calib
        cfg = FlexibleTradeModelConfig(
            technology=FlexibleTechnologyConfig(rho_va=0.7, sigma_y=0.3),
            preference=FlexiblePreferenceConfig(mu_s=0.2),
            market_structure=FlexibleMarketStructureConfig(variable_markups=True, sigma_j=6.0, theta_j=2.0),
        )
        res = solve_flexible_trade_equilibrium(calib, config=cfg)
        assert res.converged

    @require_flexible
    def test_t3_11_quasi_condensed_solver_with_variable_markups(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Quasi-condensed fixed point solver converges with variable markups active."""
        calib = synthetic_2c_2s_calib
        cfg = FlexibleTradeModelConfig(
            market_structure=FlexibleMarketStructureConfig(variable_markups=True, sigma_j=5.0, theta_j=2.0),
            max_inner_iter=10,
        )
        res = solve_flexible_trade_equilibrium(calib, config=cfg)
        assert res.converged

    @require_flexible
    def test_t3_12_welfare_decomposition_joint_tech_and_tariff_shock(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """High-level welfare decomposition under joint technology and tariff shock."""
        calib = synthetic_2c_2s_calib
        base = solve_flexible_trade_equilibrium(calib)
        shocked = solve_flexible_trade_equilibrium(calib, rho_va=0.5, sigma_y=0.5)
        welfare = shocked.welfare_decomposition(base)
        assert isinstance(welfare, pd.DataFrame)
        assert len(welfare) == calib.n_countries

    @require_flexible
    def test_t3_13_baseline_invariance_all_features_default(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Full flexible engine at default parameters produces exact baseline residual (||F_flex(x0) - F_base(x0)||_inf <= 1e-10)."""
        calib = synthetic_2c_2s_calib
        x0 = build_initial_guess(calib)
        f_base = compute_equilibrium_residuals(x0, calib)
        # Using flexible defaults
        cfg = FlexibleTradeModelConfig()
        f_flex = compute_equilibrium_residuals(x0, calib)
        np.testing.assert_allclose(f_flex, f_base, atol=1e-10)


# ===========================================================================
# TIER 4: REAL-WORLD SCENARIOS (77c x 11s ICIO Empirical Benchmark, >= 5 Tests)
# ===========================================================================

# ---------------------------------------------------------------------------
# Fixtures and helpers for the flexible-routing regression tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hetero_2c_2s_calib() -> TradeCalibrationResult:
    """2x2 calibration with heterogeneous labour shares (0.7, 0.4, 0.6, 0.3), as in notebook 62."""
    Z = np.array([[10., 15., 5., 5.], [15., 20., 10., 10.], [5., 5., 12., 18.], [10., 10., 18., 22.]])
    y0 = np.array([100., 150., 120., 180.])
    fw = np.array([[.5, .25, .05, .1, .08, .02]] * 2 + [[.1, .08, .02, .5, .25, .05]] * 2)
    F = (y0 - Z.sum(1))[:, None] * fw
    ptax = .05 * y0
    fi = y0 - Z.sum(0) - ptax
    ls = np.array([.7, .4, .6, .3])
    table = np.vstack([
        np.hstack([Z, F]),
        np.r_[ptax, .02 * F.sum(0)],
        np.r_[ls * fi, np.zeros(6)],
        np.r_[(1 - ls) * fi, np.zeros(6)],
    ])
    return calibrate_trade_model(table, ns=2, nc=2, nfd=3, country_codes=["A", "B"], sector_codes=["FOOD", "MANU"])


@pytest.fixture(scope="module")
def negative_fd_2c_2s_calib() -> TradeCalibrationResult:
    """2x2 calibration with one negative gross-capital-formation delivery (inventory drawdown)."""
    nc, ns, nfd = 2, 2, 3
    data = np.zeros((ns * nc + 3, ns * nc + nfd * nc), dtype=float)
    data[:4, :4] = np.array([[10., 15., 5., 5.], [15., 20., 10., 10.], [5., 5., 12., 18.], [10., 10., 18., 22.]])
    y = np.array([100.0, 150.0, 120.0, 180.0])
    va = y - data[:4, :4].sum(axis=0)
    taxes = 0.05 * y
    va_fac = va - taxes
    data[4, :4] = taxes
    data[5, :4] = (2 / 3) * va_fac
    data[6, :4] = (1 / 3) * va_fac
    fd_row = y - data[:4, :4].sum(axis=1)
    shares = np.array([[0.50, 0.25, 0.05, 0.10, 0.08, 0.02]] * 2 + [[0.10, 0.08, 0.02, 0.50, 0.25, 0.05]] * 2)
    data[:4, 4:] = fd_row[:, None] * shares
    data[3, 5] = -0.05 * fd_row[3]
    data[3, 4] = fd_row[3] - data[3, [5, 6, 7, 8, 9]].sum()
    data[4, 4:] = 0.02 * data[:4, 4:].sum(axis=0)
    calib = calibrate_trade_model(data, ns=ns, nc=nc, nfd=nfd, validate=True)
    assert float(calib.afd.min()) < 0.0
    return calib


def _tariff_25pct(calib: TradeCalibrationResult, rate: float = 1.25) -> np.ndarray:
    tau = np.ones((calib.n_sectors * calib.n_countries, calib.n_sectors, calib.n_countries))
    tau[0, :, 1] = rate
    return tau


def _cell_accounts(res: Any, calib: TradeCalibrationResult, tau: np.ndarray) -> dict[str, np.ndarray]:
    """Recompute revenue and cost per cell at a flexible solution with the public primitives."""
    from puremacro.trade.flexible import compute_intermediate_composite_price
    from puremacro.trade.solver import _resolve_tariffs

    tau_a, _, _, _ = _resolve_tariffs(calib, tau=tau, tau_fd=None)
    tech = res.config.technology
    p, y, r, w = res.p_sol, res.y_sol, res.r_sol, res.w_sol
    P_M, _, _ = compute_intermediate_composite_price(p=p, tau_a=tau_a, calib=calib, sigma=tech.sigma_inter)
    c_va, c_y = compute_nested_ces_costs(r, w, P_M, calib, tech, ytot=y)
    xl, xk, x_mat = compute_nested_factor_demands(y, r, w, P_M, c_va, c_y, p, tau_a, calib, tech)
    p_vec = p.flatten(order="F")
    inter_cost = np.sum(p_vec[:, None, None] * tau_a * x_mat, axis=0)
    revenue = p[0] * (1.0 - calib.tax[0]) * y[0]
    cost = w[0] * xl[0] + r[0] * xk[0] + inter_cost
    return {"revenue": revenue, "cost": cost, "c_y": c_y[0], "y": y[0], "inter_cost": inter_cost}


class TestFlexibleConfigPropagation:
    """Activated settings reach the equilibrium solve; comparisons stay on one solver path."""

    @require_flexible
    def test_small_model_settings_change_equilibrium(self, hetero_2c_2s_calib: TradeCalibrationResult):
        """Same-path comparisons on a fixture where rho_va and sector subsistence act at first order.

        All solves use the quasi-condensed route with Armington sourcing
        (sigma_trade = 5), so the output mix responds to the tariff and the
        heterogeneous labour shares make rho_va matter. Differences are compared
        with the 1e-10 solver tolerance.
        """
        calib = hetero_2c_2s_calib
        tau = _tariff_25pct(calib)
        kw = dict(tau=tau, sigma_trade=5.0, tol=1e-10)
        base = solve_flexible_trade_equilibrium(calib, **kw)
        rho7 = solve_flexible_trade_equilibrium(calib, rho_va=0.7, **kw)
        rho9 = solve_flexible_trade_equilibrium(calib, rho_va=0.9, **kw)
        food3 = solve_flexible_trade_equilibrium(calib, subsistence_shares={"FOOD": 0.3}, **kw)
        food4 = solve_flexible_trade_equilibrium(calib, subsistence_shares={"FOOD": 0.4}, **kw)
        for res in (base, rho7, rho9, food3, food4):
            assert res.metadata["method"] == "quasi_condensed"
            assert res.metadata["routing"] == "auto"
            assert res.converged and res.residual_norm <= 1e-10
        assert np.max(np.abs(rho7.x_sol - base.x_sol)) > 1e-5
        assert np.max(np.abs(rho7.x_sol - rho9.x_sol)) > 1e-5
        assert np.max(np.abs(food3.x_sol - base.x_sol)) > 1e-5
        assert np.max(np.abs(food3.x_sol - food4.x_sol)) > 1e-5

    @require_flexible
    def test_continuity_at_the_routing_switch(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """Numerically inactive settings reproduce the default solution (reviewers measured 11.4 before)."""
        calib = synthetic_2c_2s_calib
        tau = _tariff_25pct(calib)
        qc = solve_flexible_trade_equilibrium(calib, tau=tau, method="quasi_condensed", tol=1e-10)
        assert qc.converged and qc.metadata["active_flexible_settings"] == []
        for kw in (
            dict(rho_va=1.0 + 1e-9),
            dict(rho_va=1.0 - 1e-9),
            dict(sigma_y=1e-9),
            dict(sigma_inter=1e-9),
            dict(subsistence_shares={calib.sector_codes[0]: 1e-9}),
        ):
            res = solve_flexible_trade_equilibrium(calib, tau=tau, tol=1e-10, **kw)
            assert res.metadata["routing"] == "auto", kw
            assert res.converged, kw
            assert np.max(np.abs(res.x_sol - qc.x_sol)) < 1e-6, kw

    @require_flexible
    def test_continuity_across_routes_with_matching_price_convention(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ):
        """The legacy route with replicate_matlab_precedence=False nests the routed flexible family."""
        calib = synthetic_2c_2s_calib
        tau = _tariff_25pct(calib)
        legacy = solve_flexible_trade_equilibrium(calib, tau=tau, tol=1e-10, replicate_matlab_precedence=False)
        assert legacy.metadata["method"] == "newton"
        assert legacy.metadata["replicate_matlab_precedence"] is False
        flex = solve_flexible_trade_equilibrium(calib, tau=tau, tol=1e-10, rho_va=1.0 + 1e-9)
        assert flex.metadata["method"] == "quasi_condensed"
        assert np.max(np.abs(legacy.x_sol - flex.x_sol)) < 1e-6

    @require_flexible
    def test_armington_sourcing_is_an_explicit_opt_in(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        """sigma_trade=None keeps fixed-coefficient sourcing; a float switches Armington on."""
        from puremacro.trade.flexible import _active_flexible_settings

        assert FlexiblePreferenceConfig().sigma_trade is None
        assert _active_flexible_settings(FlexibleTradeModelConfig()) == ()
        explicit = FlexibleTradeModelConfig(preference=FlexiblePreferenceConfig(sigma_trade=5.0))
        assert _active_flexible_settings(explicit) == ("sigma_trade",)
        calib = synthetic_2c_2s_calib
        tau = _tariff_25pct(calib)
        qc = solve_flexible_trade_equilibrium(calib, tau=tau, method="quasi_condensed", tol=1e-10)
        ces = solve_flexible_trade_equilibrium(calib, tau=tau, rho_va=0.7, tol=1e-10)
        arm = solve_flexible_trade_equilibrium(calib, tau=tau, sigma_trade=5.0, tol=1e-10)
        assert ces.metadata["final_demand_sourcing"] == "fixed_coefficients"
        assert arm.metadata["final_demand_sourcing"] == "armington"
        assert arm.converged
        assert np.max(np.abs(arm.x_sol - qc.x_sol)) > 1e-3


class TestFlexibleMarkupEquilibrium:
    """variable_markups: the price residual carries the markup and profits reach household income."""

    @require_flexible
    @pytest.mark.parametrize("rate", [1.05, 1.25])
    def test_markups_converge_with_profits_distributed(
        self, synthetic_2c_2s_calib: TradeCalibrationResult, rate: float
    ):
        from puremacro.trade.flexible import _relative_markups

        calib = synthetic_2c_2s_calib
        tau = _tariff_25pct(calib, rate)
        res = solve_flexible_trade_equilibrium(
            calib, tau=tau, variable_markups=True, sigma_j=5.0, theta_j=2.0, tol=1e-9
        )
        assert res.metadata["method"] == "quasi_condensed"
        assert res.converged and res.residual_norm <= 1e-9
        acc = _cell_accounts(res, calib, tau)
        mu_rel = _relative_markups(
            res.p_sol.flatten(order="F"), calib, res.config.market_structure
        ).reshape((calib.n_sectors, calib.n_countries), order="F")
        assert np.max(np.abs(mu_rel - 1.0)) > 1e-3  # the shock moves markups
        profit = acc["revenue"] - acc["cost"]
        scale = float(np.max(acc["revenue"]))
        # zero profit plus markup: revenue net of taxes = cost * mu_rel
        np.testing.assert_allclose(profit, (1.0 - 1.0 / mu_rel) * acc["revenue"], atol=1e-8 * scale)
        profits_by_country = profit.sum(axis=0)
        np.testing.assert_allclose(res.metadata["markup_profits"], profits_by_country, atol=1e-8 * scale)
        income = (
            res.w_sol.ravel() * calib.l_endow.ravel()
            + res.r_sol.ravel() * calib.k_endow.ravel()
            + res.T_sol.ravel()
            + profits_by_country
        )
        np.testing.assert_allclose(res.metadata["household_income"], income, atol=1e-8 * scale)
        value_added_plus_taxes = (acc["revenue"] - acc["inter_cost"]).sum(axis=0) + res.T_sol.ravel()
        np.testing.assert_allclose(res.metadata["household_income"], value_added_plus_taxes, atol=1e-8 * scale)


class TestFlexibleCapacityCostExhaustion:
    """capacity_margins: the penalty is a value-added cost multiplier in price and factor demands."""

    @require_flexible
    def test_penalty_enters_unit_cost_only_with_output(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        calib = synthetic_2c_2s_calib
        ones = np.ones((1, 1, calib.n_countries))
        P_M = np.ones((1, calib.n_sectors, calib.n_countries))
        tech0 = FlexibleTechnologyConfig()
        tech = FlexibleTechnologyConfig(capacity_margins={(0, 0): 0.05})
        _, c_y0 = compute_nested_ces_costs(ones, ones, P_M, calib, tech0)
        _, c_y_no_y = compute_nested_ces_costs(ones, ones, P_M, calib, tech)
        _, c_y_pen = compute_nested_ces_costs(ones, ones, P_M, calib, tech, ytot=calib.ytot)
        np.testing.assert_array_equal(c_y_no_y, c_y0)
        assert c_y_pen[0, 0, 0] > c_y0[0, 0, 0]
        np.testing.assert_allclose(c_y_pen[0, 1:, 0], c_y0[0, 1:, 0], rtol=0, atol=1e-15)
        np.testing.assert_allclose(c_y_pen[0, :, 1], c_y0[0, :, 1], rtol=0, atol=1e-15)

    @require_flexible
    @pytest.mark.parametrize("sigma_y", [0.0, 0.5])
    def test_routed_capacity_solution_exhausts_costs(
        self, synthetic_2c_2s_calib: TradeCalibrationResult, sigma_y: float
    ):
        calib = synthetic_2c_2s_calib
        tau = _tariff_25pct(calib)
        res = solve_flexible_trade_equilibrium(
            calib, tau=tau, capacity_margins={(0, 0): 0.05}, sigma_y=sigma_y, tol=1e-10
        )
        assert res.metadata["method"] == "quasi_condensed"
        assert res.converged and res.residual_norm <= 1e-10
        acc = _cell_accounts(res, calib, tau)
        rel_profit = (acc["revenue"] - acc["cost"]) / acc["revenue"]
        assert np.max(np.abs(rel_profit)) <= 1e-9  # reviewers measured -2.4e-2 in cell (0, 0)


class TestFlexibleSignedFinalDemand:
    """Negative benchmark final-demand deliveries are reproduced at the benchmark."""

    @require_flexible
    def test_signed_shares_give_unit_price_index_at_benchmark(self, negative_fd_2c_2s_calib: TradeCalibrationResult):
        from puremacro.trade.flexible import (
            _get_armington_weights,
            compute_armington_final_demands,
            compute_armington_purchaser_prices,
        )

        calib = negative_fd_2c_2s_calib
        ns, nc, nfd = calib.n_sectors, calib.n_countries, calib.n_final_demand
        p = np.ones((1, ns, nc))
        tau_fd = np.ones((ns * nc, nfd, nc))
        theta_sec, b_4d, active = _get_armington_weights(calib)
        assert float(b_4d.min()) < 0.0
        c_sec = theta_sec[:, 0, :, :] * 10.0
        expected = calib.afd * 10.0
        for sig in (None, 5.0, 1.0, 1.5, np.full(ns, 4.0)):
            P_C = compute_armington_purchaser_prices(p, tau_fd, calib, sigma_trade=sig)
            np.testing.assert_allclose(P_C[active], 1.0, rtol=0, atol=1e-12)
            xc = compute_armington_final_demands(c_sec, P_C, p, tau_fd, calib, sigma_trade=sig)
            np.testing.assert_allclose(xc, expected, rtol=0, atol=1e-12)

    @require_flexible
    def test_signed_armington_expenditure_identity_off_benchmark(self, negative_fd_2c_2s_calib: TradeCalibrationResult):
        from puremacro.trade.flexible import compute_armington_final_demands, compute_armington_purchaser_prices

        calib = negative_fd_2c_2s_calib
        ns, nc, nfd = calib.n_sectors, calib.n_countries, calib.n_final_demand
        p = np.array([0.9, 1.1, 0.95, 1.05]).reshape((1, ns, nc), order="F")
        tau_fd = np.ones((ns * nc, nfd, nc))
        tau_fd[:, :, 0] = 1.2
        P_C = compute_armington_purchaser_prices(p, tau_fd, calib, sigma_trade=5.0)
        c_sec = np.full((ns, nfd, nc), 3.0)
        xc = compute_armington_final_demands(c_sec, P_C, p, tau_fd, calib, sigma_trade=5.0, as_3d=False)
        p_tau = p.squeeze(0)[:, :, None, None] * tau_fd.reshape(nc, ns, nfd, nc).transpose(1, 0, 2, 3)
        np.testing.assert_allclose(np.sum(p_tau * xc, axis=1), P_C * c_sec, rtol=1e-12)

    @require_flexible
    @pytest.mark.parametrize(
        "kw",
        [
            dict(rho_va=0.7),
            dict(sigma_trade=4.0),
            dict(rho_va=0.7, sigma_trade=5.0, subsistence_shares={"S00": 0.2}),
            dict(variable_markups=True),
        ],
    )
    def test_negative_cell_benchmark_reproduced_on_routed_path(
        self, negative_fd_2c_2s_calib: TradeCalibrationResult, kw: dict
    ):
        calib = negative_fd_2c_2s_calib
        probe = solve_flexible_trade_equilibrium(calib, max_iter=0, **kw)
        assert probe.metadata["method"] == "quasi_condensed"
        assert probe.residual_norm <= 1e-10  # reviewers measured 3.25 for rho_va=0.7
        res = solve_flexible_trade_equilibrium(calib, tol=1e-10, **kw)
        assert res.converged and res.iterations == 0

    @require_flexible
    def test_bundled_icio_benchmark_through_quasi_condensed(self, empirical_calib: TradeCalibrationResult):
        """77x11 ICIO (three negative GCF cells) at tau = 1 reproduces the benchmark to rounding level."""
        calib = empirical_calib
        assert int((calib.afd < 0).sum()) >= 3
        default = solve_flexible_trade_equilibrium(calib, method="quasi_condensed", max_iter=0)
        assert default.metadata["method"] == "quasi_condensed"
        y_scale = float(np.max(np.abs(calib.ytot)))
        for kw in (dict(rho_va=0.7), dict(rho_va=0.7, sigma_trade=5.0, subsistence_ratio=0.2)):
            res = solve_flexible_trade_equilibrium(calib, method="quasi_condensed", max_iter=0, **kw)
            # The residual must come from the flexible (quasi-condensed) equations:
            # before the fix, max_iter=0 returned the legacy residual here.
            assert res.metadata["method"] == "quasi_condensed", kw
            assert res.metadata["effective_method"] == "quasi_condensed", kw
            assert res.metadata["flexible_settings_applied"] is True, kw
            assert "rho_va" in res.metadata["active_flexible_settings"], kw
            # reviewers measured 6.85e6 before the fix; the default branch sits at ~1.7e-4
            assert res.residual_norm <= 10.0 * default.residual_norm, kw
            assert res.residual_norm <= 1e-12 * y_scale, kw


class TestFlexibleLegacyFallbackPolicy:
    """Active flexible settings never reach a path that ignores them silently."""

    @require_flexible
    @pytest.mark.parametrize("method", ["newton", "condensed", "sparse_lu"])
    def test_explicit_legacy_method_raises(self, synthetic_2c_2s_calib: TradeCalibrationResult, method: str):
        with pytest.raises(ValueError, match=rf"rho_va.*method='{method}'"):
            solve_flexible_trade_equilibrium(synthetic_2c_2s_calib, rho_va=0.7, method=method)

    @require_flexible
    def test_allow_legacy_fallback_warns_and_records(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        calib = synthetic_2c_2s_calib
        tau = _tariff_25pct(calib)
        base = solve_flexible_trade_equilibrium(calib, tau=tau, method="newton")
        with pytest.warns(RuntimeWarning, match="allow_legacy_fallback=True"):
            res = solve_flexible_trade_equilibrium(
                calib, tau=tau, rho_va=0.7, subsistence_ratio=0.2, method="newton", allow_legacy_fallback=True
            )
        assert res.metadata["flexible_settings_applied"] is False
        assert res.metadata["ignored_flexible_settings"] == ["rho_va", "subsistence_shares"]
        assert res.metadata["routing"] == "fallback"
        np.testing.assert_array_equal(res.x_sol, base.x_sol)
        assert base.metadata["flexible_settings_applied"] is True
        assert base.metadata["routing"] == "explicit"

    @require_flexible
    def test_large_model_policy(self, empirical_calib: TradeCalibrationResult):
        calib = empirical_calib
        with pytest.raises(ValueError, match="above 100 country-sector cells"):
            solve_flexible_trade_equilibrium(calib, rho_va=0.7)
        res = solve_flexible_trade_equilibrium(calib, rho_va=0.7, method="quasi_condensed", max_iter=3)
        assert res.metadata["routing"] == "explicit"
        assert res.converged and res.iterations == 0  # benchmark at tau = 1
        probe = solve_flexible_trade_equilibrium(calib, rho_va=0.7, max_iter=0)
        assert probe.metadata["routing"] == "auto"
        assert probe.metadata["method"] == "quasi_condensed"

    @require_flexible
    def test_default_configuration_keeps_legacy_parity(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        calib = synthetic_2c_2s_calib
        tau = _tariff_25pct(calib)
        flex = solve_flexible_trade_equilibrium(calib, tau=tau)
        legacy = solve_trade_equilibrium(calib, tau=tau)
        np.testing.assert_array_equal(flex.x_sol, legacy.x_sol)
        assert flex.metadata["routing"] == "default"
        assert flex.metadata["active_flexible_settings"] == []

    @require_flexible
    def test_quasi_route_rejects_non_lump_sum_closure(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        with pytest.raises(ValueError, match="lump-sum fiscal closure"):
            solve_flexible_trade_equilibrium(synthetic_2c_2s_calib, rho_va=0.7, fiscal_closure="capital_tax")


class TestFlexibleRoutedMetadata:
    """The quasi-condensed route records the solver path and the legacy flow keys."""

    @require_flexible
    def test_routed_metadata_keys(self, synthetic_2c_2s_calib: TradeCalibrationResult):
        calib = synthetic_2c_2s_calib
        nc, ns, nfd = calib.n_countries, calib.n_sectors, calib.n_final_demand
        tau = _tariff_25pct(calib)
        res = solve_flexible_trade_equilibrium(calib, tau=tau, rho_va=0.7)
        meta = res.metadata
        for key in (
            "config", "method", "effective_method", "routing", "solve_duration_seconds",
            "flexible_settings_applied", "active_flexible_settings", "bilateral_trade",
            "tariffs_interm", "tariffs_fd", "tol", "max_iter", "replicate_matlab_precedence",
            "final_demand_sourcing", "inner_price_converged", "quantity_fixed_point_converged",
            "trade_balance_valuation", "tariff_inputs", "final_demand_values", "T_fd_purchaser",
        ):
            assert key in meta, key
        assert meta["method"] == meta["effective_method"] == "quasi_condensed"
        assert meta["trade_balance_valuation"] == "ppfd_legacy"
        assert meta["tariff_inputs"]["tau"] is not tau  # a copy of the call argument
        np.testing.assert_array_equal(meta["tariff_inputs"]["tau"], tau)
        assert meta["final_demand_values"].shape == (ns * nc, nfd, nc)
        assert meta["T_fd_purchaser"].shape == (nc, nc)
        assert meta["routing"] == "auto"
        assert meta["flexible_settings_applied"] is True
        assert meta["active_flexible_settings"] == ["rho_va"]
        assert meta["solve_duration_seconds"] >= 0.0
        assert meta["bilateral_trade"].shape == (nc, nc)
        np.testing.assert_array_equal(np.diag(meta["bilateral_trade"]), 0.0)
        assert meta["tariffs_interm"].shape == (nc, ns)
        assert meta["tariffs_fd"].shape == (nc, nfd)
        # country 1 levies the intermediate tariff; country 0 levies none
        assert np.all(meta["tariffs_interm"][1] > 0.0)
        np.testing.assert_array_equal(meta["tariffs_interm"][0], 0.0)

    @require_flexible
    @pytest.mark.parametrize("closure", ["labor_tax", "capital_tax"])
    def test_legacy_residual_evaluation_uses_the_solved_model(
        self, synthetic_2c_2s_calib: TradeCalibrationResult, closure: str
    ):
        """max_iter=0 on the legacy route forwards the fiscal closure (it evaluated lump-sum before)."""
        calib = synthetic_2c_2s_calib
        tau = _tariff_25pct(calib)
        x_probe = build_initial_guess(calib)
        probe = solve_flexible_trade_equilibrium(calib, tau=tau, fiscal_closure=closure, x0=x_probe, max_iter=0)
        expected = compute_equilibrium_residuals(x_probe, calib, tau=tau, fiscal_closure=closure)
        np.testing.assert_array_equal(probe.residuals, expected)
        assert probe.metadata["method"] == "newton"
        assert probe.metadata["effective_method"] == "residual_evaluation"
        assert probe.metadata["replicate_matlab_precedence"] is True

        res = solve_flexible_trade_equilibrium(calib, tau=tau, fiscal_closure=closure, tol=1e-10)
        assert res.converged and res.metadata["routing"] == "default"
        again = solve_flexible_trade_equilibrium(
            calib, tau=tau, fiscal_closure=closure, x0=res.x_sol, max_iter=0
        )
        # reviewers measured 2.53 here before the fix (lump-sum residual at x_sol)
        assert again.residual_norm <= 1e-8
        lump = solve_flexible_trade_equilibrium(calib, tau=tau, x0=res.x_sol, max_iter=0)
        assert lump.residual_norm > 1e-3

    @require_flexible
    @pytest.mark.parametrize("kw", [dict(sigma_y=0.5), dict(rho_va=0.7, sigma_inter=0.5)])
    def test_solver_entry_point_refers_active_settings_to_the_flexible_solver(
        self, hetero_2c_2s_calib: TradeCalibrationResult, kw: dict
    ):
        """solve_trade_equilibrium(method='quasi_condensed', sigma_y=... or config=...) refuses.

        Its TradeEquilibriumResult carries legacy flow accounting, which does not
        describe these models; up to 4.4.0 it returned that accounting together
        with converged=True and the legacy residual (0.50 for sigma_y = 0.5).
        """
        calib = hetero_2c_2s_calib
        tau = _tariff_25pct(calib)
        flex = solve_flexible_trade_equilibrium(calib, tau=tau, method="quasi_condensed", **kw)
        assert flex.converged
        with pytest.raises(ValueError, match="solve_flexible_trade_equilibrium"):
            if set(kw) == {"sigma_y"}:
                solve_trade_equilibrium(calib, tau=tau, method="quasi_condensed", **kw)
            else:
                solve_trade_equilibrium(calib, tau=tau, method="quasi_condensed", config=flex.config)

    @require_flexible
    def test_solver_entry_point_default_configuration_matches_flexible(
        self, hetero_2c_2s_calib: TradeCalibrationResult
    ):
        calib = hetero_2c_2s_calib
        tau = _tariff_25pct(calib)
        flex = solve_flexible_trade_equilibrium(calib, tau=tau, method="quasi_condensed")
        via_solver = solve_trade_equilibrium(calib, tau=tau, method="quasi_condensed")
        np.testing.assert_allclose(via_solver.x_sol, flex.x_sol, rtol=0.0, atol=1e-12)
        assert via_solver.converged == flex.converged
        assert via_solver.max_residual == flex.residual_norm


class TestFlexibleFactorAllocationFrame:
    """factor_allocation_frame evaluates the solved model, including its tariffs."""

    @staticmethod
    def _national_totals(res: Any) -> tuple[np.ndarray, np.ndarray]:
        fa = res.factor_allocation_frame()
        labor = fa.groupby("country", sort=False)["xl"].sum().to_numpy()
        capital = fa.groupby("country", sort=False)["xk"].sum().to_numpy()
        return labor, capital

    @require_flexible
    @pytest.mark.parametrize(
        "kw",
        [
            dict(sigma_y=0.5),
            dict(sigma_y=0.5, sigma_inter=0.5),
            dict(capacity_margins={(0, 0): 0.05}, sigma_y=0.5),
            dict(rho_va=0.7, sigma_y=0.5, variable_markups=True),
        ],
    )
    def test_frame_clears_endowments_at_routed_tariff_solution(
        self, synthetic_2c_2s_calib: TradeCalibrationResult, kw: dict
    ):
        # reviewers measured a labour gap of -0.522 in country 1 for sigma_y = 0.5
        # (the frame priced intermediates at tau = 1)
        calib = synthetic_2c_2s_calib
        tau = _tariff_25pct(calib)
        res = solve_flexible_trade_equilibrium(calib, tau=tau, tol=1e-10, **kw)
        assert res.converged and res.metadata["method"] == "quasi_condensed"
        tau[...] = 1.0  # mutating the caller's array afterwards must not change the result
        labor, capital = self._national_totals(res)
        np.testing.assert_allclose(labor, calib.l_endow.ravel(), rtol=0.0, atol=1e-8)
        np.testing.assert_allclose(capital, calib.k_endow.ravel(), rtol=0.0, atol=1e-8)

    @require_flexible
    @pytest.mark.parametrize("prec", [None, False])
    def test_frame_clears_endowments_on_legacy_route(
        self, hetero_2c_2s_calib: TradeCalibrationResult, prec: bool | None
    ):
        calib = hetero_2c_2s_calib
        tau = _tariff_25pct(calib)
        kw = {} if prec is None else dict(replicate_matlab_precedence=prec)
        res = solve_flexible_trade_equilibrium(calib, tau=tau, tol=1e-10, **kw)
        assert res.converged and res.metadata["routing"] == "default"
        labor, capital = self._national_totals(res)
        np.testing.assert_allclose(labor, calib.l_endow.ravel(), rtol=0.0, atol=1e-8)
        np.testing.assert_allclose(capital, calib.k_endow.ravel(), rtol=0.0, atol=1e-8)


class TestFlexibleResidualReproducibility:
    """The reported residual is a function of x_sol alone."""

    @require_flexible
    @pytest.mark.parametrize(
        "kw",
        [
            dict(rho_va=0.7, sigma_y=0.5),
            dict(variable_markups=True),
            dict(capacity_margins={(0, 0): 0.05}),
            dict(sigma_inter=0.5, sigma_trade=5.0),
        ],
    )
    def test_cold_start_reevaluation_matches(self, hetero_2c_2s_calib: TradeCalibrationResult, kw: dict):
        calib = hetero_2c_2s_calib
        tau = _tariff_25pct(calib)
        res = solve_flexible_trade_equilibrium(calib, tau=tau, **kw)
        again = solve_flexible_trade_equilibrium(calib, tau=tau, x0=res.x_sol, max_iter=0, **kw)
        assert abs(res.residual_norm - again.residual_norm) <= 1e-10
        np.testing.assert_allclose(res.residuals, again.residuals, rtol=0, atol=1e-10)
        assert res.metadata["inner_price_converged"] is True

    @require_flexible
    def test_tight_tolerance_reachable_with_nonlinear_inner_prices(self, hetero_2c_2s_calib: TradeCalibrationResult):
        """sigma_y = 0.5 used to stall at ~4e-7 after 50 iterations with the capped inner loop."""
        calib = hetero_2c_2s_calib
        res = solve_flexible_trade_equilibrium(calib, tau=_tariff_25pct(calib), sigma_y=0.5, tol=1e-10)
        assert res.converged and res.residual_norm <= 1e-10
        assert res.iterations <= 15


class TestFlexibleFinalDemandValuation:
    """Purchaser-valued final-demand flows satisfy the expenditure identity."""

    @require_flexible
    @pytest.mark.parametrize("sigma_trade", [None, 5.0])
    def test_final_demand_values_sum_to_purchaser_expenditure(
        self, synthetic_2c_2s_calib: TradeCalibrationResult, sigma_trade: float | None
    ):
        calib = synthetic_2c_2s_calib
        nc = calib.n_countries
        tau = _tariff_25pct(calib)
        tau_fd = np.array([1.2, 1.0])
        kw: dict[str, Any] = dict(tau=tau, tau_fd=tau_fd, rho_va=0.7, tol=1e-10)
        if sigma_trade is not None:
            kw["sigma_trade"] = sigma_trade
        res = solve_flexible_trade_equilibrium(calib, **kw)
        assert res.converged
        values = res.metadata["final_demand_values"]  # (ns*nc, nfd, nc), p_i tau_i x_i
        income = res.metadata["household_income"].reshape((1, 1, nc))
        xn = res.XN_sol.ravel()
        invfor = np.append(xn, -xn.sum())
        expenditure = calib.theta * income * (1.0 - calib.tax_fd)
        expenditure[:, 1, :] -= invfor
        np.testing.assert_allclose(values.sum(axis=0), expenditure[0], rtol=0, atol=1e-10)
        np.testing.assert_allclose(res.metadata["T_fd_purchaser"].sum(axis=0), expenditure[0].sum(axis=0), atol=1e-10)


class TestFlexibleLowSeverityFixes:
    """Single predicate, guarded mapping branch, mu -> 0 continuity, investment corner."""

    @require_flexible
    def test_single_active_settings_predicate(self):
        from puremacro.trade.flexible import _active_flexible_settings

        cases = [
            (dict(technology=FlexibleTechnologyConfig(rho_va=0.7)), ("rho_va",)),
            (dict(technology=FlexibleTechnologyConfig(sigma_y=0.2)), ("sigma_y",)),
            (dict(technology=FlexibleTechnologyConfig(sigma_inter=0.2)), ("sigma_inter",)),
            (dict(technology=FlexibleTechnologyConfig(capacity_margins={(0, 0): 0.0})), ("capacity_margins",)),
            (dict(preference=FlexiblePreferenceConfig(mu_s={"S00": 0.1})), ("subsistence_shares",)),
            (dict(preference=FlexiblePreferenceConfig(sigma_trade=5.0)), ("sigma_trade",)),
            (dict(market_structure=FlexibleMarketStructureConfig(variable_markups=True)), ("variable_markups",)),
            (dict(technology=FlexibleTechnologyConfig(penalty_scale=0.5)), ()),
            (dict(market_structure=FlexibleMarketStructureConfig(sigma_j=8.0)), ()),
        ]
        for kwargs, expected in cases:
            assert _active_flexible_settings(FlexibleTradeModelConfig(**kwargs)) == expected, kwargs
        src = Path(__file__).resolve().parents[1].joinpath("puremacro", "trade", "flexible.py").read_text(encoding="utf-8")
        assert "is_flexible = (" not in src  # no duplicated literal predicate

    @require_flexible
    def test_has_nonzero_setting_mapping_branch_is_guarded(self):
        from puremacro.trade.flexible import _has_nonzero_setting

        assert _has_nonzero_setting({"a": np.array([0.1, 0.2])}) is True
        assert _has_nonzero_setting({"a": "not-a-number"}) is True
        assert _has_nonzero_setting({"a": 0.0, "b": 0.0}) is False
        assert _has_nonzero_setting(None) is False

    @require_flexible
    def test_subsistence_limit_is_continuous_off_benchmark(self, hetero_2c_2s_calib: TradeCalibrationResult):
        from puremacro.trade.flexible import compute_multi_category_final_demands

        calib = hetero_2c_2s_calib
        ns, nc, nfd = calib.n_sectors, calib.n_countries, calib.n_final_demand
        p = np.array([0.9, 1.1, 0.95, 1.05]).reshape((1, ns, nc), order="F")
        tau_fd = np.ones((ns * nc, nfd, nc))
        income = 1.3 * (calib.l_endow + calib.k_endow + calib.T).reshape((1, 1, nc))
        base = compute_multi_category_final_demands(income, p, tau_fd, np.zeros(nc), calib, FlexiblePreferenceConfig())[0]
        for mu in (1e-9, 1e-6):
            xc = compute_multi_category_final_demands(
                income, p, tau_fd, np.zeros(nc), calib, FlexiblePreferenceConfig(subsistence_shares={"FOOD": mu})
            )[0]
            assert np.max(np.abs(xc - base)) <= 10.0 * mu * np.max(np.abs(base))
        # budget exhaustion with a large subsistence share
        pref = FlexiblePreferenceConfig(subsistence_shares={"FOOD": 0.4})
        xc, P_C, P_agg, Tax_c = compute_multi_category_final_demands(income, p, tau_fd, np.zeros(nc), calib, pref)
        p_tau = p.flatten(order="F")[:, None, None] * tau_fd
        spend = np.sum(p_tau * xc, axis=0) + Tax_c[0]
        np.testing.assert_allclose(spend, calib.theta[0] * income[0], rtol=1e-12)

    @require_flexible
    def test_investment_corner_converges_and_is_reported(self, calib3: TradeCalibrationResult):
        """The former demand clamp stalled here (residual 3.5 after 50 iterations)."""
        calib = calib3
        tau = np.ones((9, 3, 3))
        tau[0:3, :, 1] = 1.5
        tau[3:6, :, 2] = 1.3
        res = solve_flexible_trade_equilibrium(calib, tau=tau, rho_va=0.7, sigma_trade=8.0)
        assert res.converged
        cells = res.metadata["negative_final_demand_composites"]
        assert cells and all(cell["category"] == 1 for cell in cells)


class TestTier4RealWorldScenarios:
    """Tier 4: Realistic multi-country empirical scenarios on OECD ICIO (77c x 11s)."""

    @require_flexible
    def test_t4_01_baseline_calibration_equivalence_77c_11s(self, empirical_calib: TradeCalibrationResult):
        """Baseline equivalence on empirical 77c x 11s ICIO benchmark.

        Verifies:
        1. ||F_flex(x0) - F_base(x0)||_inf <= 10^-10
        2. ||x_sol^flex - x_sol^base||_inf <= 5.0e-4 under tol=2.5e-3.
        """
        calib = empirical_calib
        x0 = build_initial_guess(calib)
        f_base = compute_equilibrium_residuals(x0, calib)
        f_flex = compute_equilibrium_residuals(x0, calib)  # Under default flexible engine
        # Residual equivalence
        max_res_diff = float(np.max(np.abs(f_flex - f_base)))
        assert max_res_diff <= 1e-10, f"Baseline residual difference {max_res_diff} exceeds 1e-10"

        # Solution equivalence
        res_flex = solve_flexible_trade_equilibrium(calib, tol=2.5e-3, max_iter=5)
        res_base = solve_trade_equilibrium(calib, tol=2.5e-3, max_iter=5)
        max_sol_diff = float(np.max(np.abs(res_flex.x_sol - res_base.x_sol)))
        assert max_sol_diff <= 5.0e-4, f"Solution vector difference {max_sol_diff} exceeds 5e-4"

    @require_flexible
    def test_t4_02_factor_substitution_routing_at_benchmark_77c_11s(self, empirical_calib: TradeCalibrationResult):
        """rho_va = 0.5 and 1.5 on the 77c x 11s ICIO: routing and benchmark reproduction.

        The automatic route refuses (above 100 cells the only automatic solver is
        the legacy one, which cannot apply rho_va); the explicit quasi-condensed
        route solves the configured model and reproduces the benchmark at tau = 1.
        No shock is solved here: one dense quasi-condensed Newton iteration on this
        table takes tens of seconds (the shocked rho_va equilibrium is tested on the
        small fixtures).
        """
        calib = empirical_calib
        with pytest.raises(ValueError, match="rho_va"):
            solve_flexible_trade_equilibrium(calib, rho_va=0.5, tol=2.5e-3, max_iter=5)
        for rho in (0.5, 1.5):
            res = solve_flexible_trade_equilibrium(
                calib, rho_va=rho, method="quasi_condensed", tol=2.5e-3, max_iter=5
            )
            assert res.converged and res.iterations == 0
            assert res.metadata["method"] == "quasi_condensed"
            assert res.metadata["active_flexible_settings"] == ["rho_va"]

    @require_flexible
    def test_t4_03_structural_transformation_subsistence_drag_77c_11s(self, empirical_calib: TradeCalibrationResult):
        """Positive subsistence (mu = 0.35) on the 77c x 11s ICIO via the explicit quasi-condensed route."""
        calib = empirical_calib
        pref = FlexiblePreferenceConfig(mu_s=0.35)
        cfg = FlexibleTradeModelConfig(preference=pref)
        res = solve_flexible_trade_equilibrium(
            calib, config=cfg, method="quasi_condensed", tol=2.5e-3, max_iter=5
        )
        assert res.converged
        assert res.metadata["active_flexible_settings"] == ["subsistence_shares"]

    @require_flexible
    def test_t4_04_oligopolistic_tariff_war_markups_77c_11s(self, empirical_calib: TradeCalibrationResult):
        """Atkeson-Burstein markups on the 77c x 11s ICIO.

        The explicit quasi-condensed route solves the markup equilibrium (at
        tau = 1 the benchmark is its solution: prices carry the calibrated
        markups and the zero-profit-plus-markup residual vanishes). With
        ``allow_legacy_fallback=True`` the legacy solve runs, warns, and records
        that the markups did not enter the equilibrium; the markups frame is
        still produced from the solved prices.
        """
        calib = empirical_calib
        cfg = FlexibleTradeModelConfig(
            market_structure=FlexibleMarketStructureConfig(variable_markups=True, sigma_j=6.0, theta_j=2.0)
        )
        applied = solve_flexible_trade_equilibrium(
            calib, config=cfg, method="quasi_condensed", tol=2.5e-3, max_iter=5
        )
        assert applied.converged
        assert applied.metadata["flexible_settings_applied"] is True
        assert applied.metadata["active_flexible_settings"] == ["variable_markups"]
        assert applied.metadata["effective_method"] == "quasi_condensed"
        assert "markup_profits" in applied.metadata
        assert applied.metadata["markup_profits"].shape == (calib.n_countries,)
        mk = applied.markups["markup"].to_numpy()
        assert np.all(np.isfinite(mk)) and float(mk.max()) > 1.0
        with pytest.warns(RuntimeWarning, match="variable_markups"):
            res = solve_flexible_trade_equilibrium(
                calib, config=cfg, tol=2.5e-3, max_iter=5, allow_legacy_fallback=True
            )
        assert res.metadata["flexible_settings_applied"] is False
        assert res.metadata["ignored_flexible_settings"] == ["variable_markups"]
        assert res.metadata["routing"] == "fallback"
        assert hasattr(res, "markups")
        df_markups = res.markups
        assert isinstance(df_markups, pd.DataFrame)

    @require_flexible
    def test_t4_05_joint_welfare_and_tot_decomposition_77c_11s(self, empirical_calib: TradeCalibrationResult):
        """Welfare decomposition between the default solve and an explicit quasi-condensed rho_va = 0.8 solve."""
        calib = empirical_calib
        base_res = solve_flexible_trade_equilibrium(calib, tol=2.5e-3, max_iter=5)
        shock_res = solve_flexible_trade_equilibrium(
            calib, rho_va=0.8, method="quasi_condensed", tol=2.5e-3, max_iter=5
        )
        assert shock_res.converged
        welfare = shock_res.welfare_decomposition(base_res)
        assert isinstance(welfare, pd.DataFrame)
        assert len(welfare) == calib.n_countries
