"""Tier 5 White-Box Adversarial Coverage & Micro-foundations Hardening Test Suite.

Author: orch26_challenger_m6_tier5_1 (Milestone 6 Phase 2 Challenger 1)
Scope: puremacro.trade.flexible micro-foundations, boundary limits, and numerical stability:
1. Nested CES Technology: extreme substitution (rho_va -> 0, rho_va -> infty), asymmetric
   factor shares (alpha -> 0, alpha -> 1), single-factor sectors, extreme w/r in [10^-8, 10^8],
   gating continuity.
2. Stone-Geary LES Preferences: near-subsistence threshold, sub-subsistence income collapse,
   high-income asymptote (expenditure shares -> theta_LES), smooth scaling axioms g(u).
3. Atkeson-Burstein Markups: single domestic monopolist limit (s_ni -> 1.0), infinitesimal
   fringe (s_ni -> 0.0), markup bounds clipping, autarky trade prohibitions (tau -> infty).
4. Variety Condensation: fixed cost collapse and explosion (fl, fk), deep recessions (pi_op <= 0),
   effective price scaling under variety changes.
5. Exact Baseline Invariance & Systemic Verification: residual invariance <= 10^-10, factor
   demand replication <= 10^-12 relative error, Hicksian EV self-identity, convergence diagnostics.

Strictly conforms to Pyodide 4-package runtime contract (NumPy, SciPy, Pandas only).
"""
from __future__ import annotations

from dataclasses import replace
import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from puremacro.trade import (
    TradeCalibrationResult,
    calibrate_trade_model,
    compute_equilibrium_residuals,
)
from puremacro.trade.data import load_icio_data
from puremacro.trade.flexible import (
    FlexibleMarketStructureConfig,
    FlexiblePreferenceConfig,
    FlexibleTechnologyConfig,
    FlexibleTradeEquilibriumResult,
    FlexibleTradeModelConfig,
    compute_armington_final_demands,
    compute_armington_purchaser_prices,
    compute_atkeson_burstein_markups,
    compute_benchmark_market_shares,
    compute_benchmark_markups,
    compute_capacity_penalty,
    compute_convergence_diagnostics,
    compute_dixit_stiglitz_varieties,
    compute_intermediate_composite_price,
    compute_les_marginal_budget_shares,
    compute_nested_ces_costs,
    compute_nested_factor_demands,
    compute_outer_ces_cost,
    compute_inner_ces_cost,
    compute_inner_ces_derivatives,
    compute_stone_geary_final_demand,
    compute_variety_price_scaling,
    smooth_subsistence_scaling,
    solve_flexible_trade_equilibrium,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def empirical_calib() -> TradeCalibrationResult:
    """Calibrate empirical 77-country 11-sector OECD ICIO benchmark model."""
    data = load_icio_data()
    return calibrate_trade_model(data)


# =============================================================================
# 1. Nested CES Technology Adversarial Tests
# =============================================================================

class TestNestedCESTechnologyAdversarial:
    """Stress-tests for Nested CES production technology and normalized factor demands."""

    def test_rho_va_leontief_limit_linear_unit_cost(self, empirical_calib: TradeCalibrationResult) -> None:
        """As rho_va -> 0 (Leontief limit), CES unit cost must converge to linear cost alpha*r + (1-alpha)*w."""
        r = np.full((1, 11, 77), 1.8)
        w = np.full((1, 11, 77), 0.6)

        # Theoretical Leontief limit cost
        alpha = empirical_calib.alpha
        beta = empirical_calib.beta
        term_alpha = (alpha ** alpha) * ((1.0 - alpha) ** (1.0 - alpha))
        c_va_0 = 1.0 / (beta * term_alpha)
        c_va_leontief = c_va_0 * (alpha * 1.8 + (1.0 - alpha) * 0.6)

        for rho in [1e-6, 1e-4, 1e-3]:
            cfg = FlexibleTechnologyConfig(rho_va=rho)
            c_va = compute_inner_ces_cost(r, w, empirical_calib, cfg)
            rel_err = np.max(np.abs(c_va - c_va_leontief) / c_va_leontief)
            assert rel_err < 5e-3, f"Leontief limit failed for rho_va={rho}: rel_err={rel_err}"

    def test_rho_va_linear_production_limit_min_cost(self, empirical_calib: TradeCalibrationResult) -> None:
        """As rho_va -> infty (linear technology / perfect substitutes), unit cost converges to min(r, w)."""
        alpha = empirical_calib.alpha
        beta = empirical_calib.beta
        term_alpha = (alpha ** alpha) * ((1.0 - alpha) ** (1.0 - alpha))
        c_va_0 = 1.0 / (beta * term_alpha)

        # Case 1: w < r
        r1 = np.full((1, 11, 77), 2.5)
        w1 = np.full((1, 11, 77), 0.5)
        cfg_hi = FlexibleTechnologyConfig(rho_va=100.0)
        c_va1 = compute_inner_ces_cost(r1, w1, empirical_calib, cfg_hi)
        rel_cost1 = c_va1 / c_va_0
        assert np.allclose(rel_cost1, 0.5, atol=0.05)

        # Case 2: r < w
        r2 = np.full((1, 11, 77), 0.4)
        w2 = np.full((1, 11, 77), 3.0)
        c_va2 = compute_inner_ces_cost(r2, w2, empirical_calib, cfg_hi)
        rel_cost2 = c_va2 / c_va_0
        assert np.allclose(rel_cost2, 0.4, atol=0.05)

    @pytest.mark.parametrize("w_val, r_val", [
        (1e-8, 1e8),
        (1e8, 1e-8),
        (1e-12, 1e12),
        (1e12, 1e-12),
    ])
    @pytest.mark.parametrize("rho", [0.01, 0.5, 1.0, 2.0, 25.0])
    def test_extreme_wage_rental_ratios_factor_demands(
        self,
        empirical_calib: TradeCalibrationResult,
        w_val: float,
        r_val: float,
        rho: float,
    ) -> None:
        """Verify numerical stability of factor demands under extreme factor price ratios w/r in [10^-12, 10^12]."""
        cfg = FlexibleTechnologyConfig(rho_va=rho)
        r = np.full((1, 11, 77), r_val)
        w = np.full((1, 11, 77), w_val)

        c_va = compute_inner_ces_cost(r, w, empirical_calib, cfg)
        assert np.all(np.isfinite(c_va)), f"c_va non-finite for w={w_val}, r={r_val}, rho={rho}"
        assert np.all(c_va > 0.0), f"c_va non-positive for w={w_val}, r={r_val}, rho={rho}"

        norm_dw, norm_dr = compute_inner_ces_derivatives(r, w, c_va, empirical_calib, cfg)
        assert np.all(np.isfinite(norm_dw)), f"norm_dw non-finite for w={w_val}, r={r_val}, rho={rho}"
        assert np.all(np.isfinite(norm_dr)), f"norm_dr non-finite for w={w_val}, r={r_val}, rho={rho}"
        assert np.all(norm_dw >= 0.0), f"norm_dw negative for w={w_val}, r={r_val}, rho={rho}"
        assert np.all(norm_dr >= 0.0), f"norm_dr negative for w={w_val}, r={r_val}, rho={rho}"

        # Endogenous factor demands xl, xk
        P_M = np.ones((1, 11, 77))
        c_y = np.ones((1, 11, 77))
        p = np.ones((1, 11, 77))
        tau = np.ones((11 * 77, 11, 77))
        xl, xk, x_mat = compute_nested_factor_demands(
            ytot=empirical_calib.ytot,
            r=r,
            w=w,
            P_M=P_M,
            c_va=c_va,
            c_y=c_y,
            p=p,
            tau=tau,
            calib=empirical_calib,
            tech_cfg=cfg,
        )
        assert np.all(np.isfinite(xl))
        assert np.all(np.isfinite(xk))
        assert np.all(xl >= 0.0)
        assert np.all(xk >= 0.0)

    @pytest.mark.parametrize("alpha_val", [1e-12, 1e-6, 0.001, 0.999, 1 - 1e-6, 1 - 1e-12])
    def test_asymmetric_factor_shares_near_boundaries(
        self,
        empirical_calib: TradeCalibrationResult,
        alpha_val: float,
    ) -> None:
        """Asymmetric factor shares alpha -> 0 or alpha -> 1 must evaluate without numerical failure."""
        calib_mod = replace(empirical_calib, alpha=np.full_like(empirical_calib.alpha, alpha_val))
        r = np.full((1, 11, 77), 1.2)
        w = np.full((1, 11, 77), 0.9)

        for rho in [0.2, 1.0, 3.0]:
            cfg = FlexibleTechnologyConfig(rho_va=rho)
            c_va = compute_inner_ces_cost(r, w, calib_mod, cfg)
            dw, dr = compute_inner_ces_derivatives(r, w, c_va, calib_mod, cfg)
            assert np.all(np.isfinite(c_va)), f"c_va not finite for alpha={alpha_val}, rho={rho}"
            assert np.all(c_va > 0.0), f"c_va not positive for alpha={alpha_val}, rho={rho}"
            assert np.all(np.isfinite(dw))
            assert np.all(np.isfinite(dr))

    def test_single_factor_sector_boundary_masking(self, empirical_calib: TradeCalibrationResult) -> None:
        """Verify single-factor sector boundary behavior (alpha=0.0 pure labor and alpha=1.0 pure capital)."""
        # Case 1: alpha=0.0 (pure labor) -> c_va > 0, dw > 0, dr == 0
        calib_labor = replace(empirical_calib, alpha=np.full_like(empirical_calib.alpha, 0.0))
        cfg = FlexibleTechnologyConfig(rho_va=1.0)
        c_va_l = compute_inner_ces_cost(np.ones((1, 11, 77)), np.ones((1, 11, 77)), calib_labor, cfg)
        dw_l, dr_l = compute_inner_ces_derivatives(np.ones((1, 11, 77)), np.ones((1, 11, 77)), c_va_l, calib_labor, cfg)
        assert np.all(np.isfinite(c_va_l))
        assert np.all(c_va_l > 0.0)
        assert np.all(dw_l > 0.0)
        assert np.all(dr_l == 0.0)

        # Case 2: alpha=1.0 (pure capital) -> c_va > 0, dw == 0, dr > 0
        calib_cap = replace(empirical_calib, alpha=np.full_like(empirical_calib.alpha, 1.0))
        c_va_k = compute_inner_ces_cost(np.ones((1, 11, 77)), np.ones((1, 11, 77)), calib_cap, cfg)
        dw_k, dr_k = compute_inner_ces_derivatives(np.ones((1, 11, 77)), np.ones((1, 11, 77)), c_va_k, calib_cap, cfg)
        assert np.all(np.isfinite(c_va_k))
        assert np.all(c_va_k > 0.0)
        assert np.all(dw_k == 0.0)
        assert np.all(dr_k > 0.0)

    def test_cobb_douglas_gating_continuity(self, empirical_calib: TradeCalibrationResult) -> None:
        """Verify smooth C^0 and C^1 continuity across the |rho_va - 1.0| < 1e-6 gating threshold."""
        r = np.full((1, 11, 77), 1.4)
        w = np.full((1, 11, 77), 0.7)

        c_cd = compute_inner_ces_cost(r, w, empirical_calib, FlexibleTechnologyConfig(rho_va=1.0))
        c_inside = compute_inner_ces_cost(r, w, empirical_calib, FlexibleTechnologyConfig(rho_va=1.0 + 5e-7))
        c_outside = compute_inner_ces_cost(r, w, empirical_calib, FlexibleTechnologyConfig(rho_va=1.0 + 2e-6))

        assert np.max(np.abs(c_cd - c_inside)) == 0.0, "Inside gating branch deviated from Cobb-Douglas"
        diff_cost = np.max(np.abs(c_cd - c_outside))
        assert diff_cost < 1e-5, f"Cost discontinuity across Cobb-Douglas gate: {diff_cost}"

        dw_cd, dr_cd = compute_inner_ces_derivatives(r, w, c_cd, empirical_calib, FlexibleTechnologyConfig(rho_va=1.0))
        dw_out, dr_out = compute_inner_ces_derivatives(r, w, c_outside, empirical_calib, FlexibleTechnologyConfig(rho_va=1.0 + 2e-6))
        assert np.max(np.abs(dw_cd - dw_out)) < 1e-5, "dw derivative discontinuity across gate"
        assert np.max(np.abs(dr_cd - dr_out)) < 1e-5, "dr derivative discontinuity across gate"

    def test_outer_ces_leontief_gating_continuity(self, empirical_calib: TradeCalibrationResult) -> None:
        """Verify smooth continuity across the sigma_y < 1e-6 Leontief outer nest gate."""
        c_va = np.full((1, 11, 77), 1.25)
        P_M = np.full((1, 11, 77), 0.85)

        cy_leo = compute_outer_ces_cost(c_va, P_M, empirical_calib, FlexibleTechnologyConfig(sigma_y=0.0))
        cy_near = compute_outer_ces_cost(c_va, P_M, empirical_calib, FlexibleTechnologyConfig(sigma_y=2e-6))
        diff_outer = np.max(np.abs(cy_leo - cy_near))
        assert diff_outer < 5e-4, f"Outer nest Leontief gate discontinuity: {diff_outer}"

    def test_outer_ces_extreme_cost_ratio_overflow(self, empirical_calib: TradeCalibrationResult) -> None:
        """Reproduction of Defect 1: outer CES cost calculation under extreme cost dispersion and sigma_y > 1."""
        cfg = FlexibleTechnologyConfig(sigma_y=50.0)  # e = -49
        c_va = np.full((1, 11, 77), 1e-8)
        P_M = np.full((1, 11, 77), 1e8)
        c_y = compute_outer_ces_cost(c_va, P_M, empirical_calib, cfg)
        assert np.all(np.isfinite(c_y)), "c_y contains inf or nan under extreme outer cost ratio"
        assert np.all(c_y > 0.0), "c_y collapsed to non-positive value due to power overflow"

    def test_intermediate_composite_small_price_overflow(self, empirical_calib: TradeCalibrationResult) -> None:
        """Reproduction of Defect 3: intermediate composite price under small p and high sigma."""
        tau_a = np.ones((11 * 77, 11, 77), dtype=float)
        p = np.full((1, 11, 77), 1e-8)
        P_M, _, _ = compute_intermediate_composite_price(p, tau_a, empirical_calib, sigma=45.0)
        assert np.all(np.isfinite(P_M)), "P_M contains inf or nan under small prices and high elasticity"

    def test_capacity_barrier_penalty_extreme_spikes(self, empirical_calib: TradeCalibrationResult) -> None:
        """Capacity barrier penalty must remain non-negative, finite, and strictly convex under 10x output spikes."""
        tech_cfg = FlexibleTechnologyConfig(
            capacity_margins={"MANU": 0.20},
            penalty_scale=0.05,
            penalty_exponent=2.0,
        )
        y_normal = empirical_calib.ytot.copy()
        pen_normal = compute_capacity_penalty(y_normal, empirical_calib, tech_cfg)
        assert np.all(pen_normal >= 0.0)

        # 10x spike
        y_spike = y_normal * 10.0
        pen_spike = compute_capacity_penalty(y_spike, empirical_calib, tech_cfg)
        assert np.all(np.isfinite(pen_spike))
        assert np.max(pen_spike) > np.max(pen_normal)


# =============================================================================
# 2. Stone-Geary LES Preferences Adversarial Tests
# =============================================================================

class TestStoneGearyPreferencesAdversarial:
    """Stress-tests for Stone-Geary Linear Expenditure System (LES) and Armington sourcing."""

    def test_subsistence_smooth_scaling_axioms(self) -> None:
        """Exhaustively verify mathematical properties of g(u) = tanh(3u) / tanh(3)."""
        # Exact calibration invariance at baseline
        assert abs(smooth_subsistence_scaling(1.0) - 1.0) < 1e-15, "g(1.0) != 1.0"
        # Zero-income vanishing
        assert smooth_subsistence_scaling(0.0) == 0.0, "g(0.0) != 0.0"

        # Check monotonicity and concavity within float64 resolution (u in [0.0, 2.5])
        u_dense = np.linspace(0.0, 2.5, 500)
        g_vals = smooth_subsistence_scaling(u_dense)

        # Strict monotonicity: g'(u) > 0
        diffs = np.diff(g_vals)
        assert np.all(diffs > 0), "g(u) violated strict monotonicity"

        # Strict concavity for u > 0: g''(u) < 0
        diffs2 = np.diff(diffs)
        assert np.all(diffs2 < 0), "g(u) violated strict concavity"

        # Asymptotic ceiling
        asymptote = 1.0 / np.tanh(3.0)
        assert np.all(g_vals <= asymptote + 1e-15), "g(u) exceeded asymptotic upper bound"

    def test_high_income_asymptote_expenditure_shares(self, empirical_calib: TradeCalibrationResult) -> None:
        """As consumer income Y -> infty, Stone-Geary expenditure shares must converge to marginal shares theta_LES."""
        pref_cfg = FlexiblePreferenceConfig(mu_s=0.4)
        P_C = np.ones((1, 11, 77))

        Y_con_0 = empirical_calib.l_endow + empirical_calib.k_endow + (
            empirical_calib.T if empirical_calib.T is not None else 0.0
        )
        # Deep asymptote: 10^8 x baseline income
        Y_asymptote = Y_con_0 * 1e8
        c_high = compute_stone_geary_final_demand(Y_asymptote, P_C, empirical_calib, pref_cfg)
        exp_shares_high = (P_C * c_high) / np.sum(P_C * c_high, axis=1, keepdims=True)

        theta_les = compute_les_marginal_budget_shares(empirical_calib, pref_cfg)
        max_dist = np.max(np.abs(exp_shares_high - theta_les))
        assert max_dist < 1e-10, f"High-income expenditure shares failed to converge to theta_LES: {max_dist}"

    @pytest.mark.parametrize("income_multiplier", [1.0, 0.5, 0.1, 0.01, 1e-6, 1e-12, 0.0])
    def test_near_and_sub_subsistence_income_resilience(
        self,
        empirical_calib: TradeCalibrationResult,
        income_multiplier: float,
    ) -> None:
        """Ensure non-negativity and numerical stability when income drops near or below subsistence threshold."""
        pref_cfg = FlexiblePreferenceConfig(mu_s=0.6)
        P_C = np.ones((1, 11, 77))

        Y_con_0 = empirical_calib.l_endow + empirical_calib.k_endow + (
            empirical_calib.T if empirical_calib.T is not None else 0.0
        )
        Y_test = Y_con_0 * income_multiplier

        c_dem = compute_stone_geary_final_demand(Y_test, P_C, empirical_calib, pref_cfg)
        assert np.all(np.isfinite(c_dem)), f"c_dem not finite for multiplier={income_multiplier}"
        assert np.all(c_dem >= 0.0), f"c_dem negative for multiplier={income_multiplier}: min={np.min(c_dem)}"

    def test_budget_identity_exactness(self, empirical_calib: TradeCalibrationResult) -> None:
        """Verify Walrasian budget exhaustion sum_s P_C,s * c_C,s == Y_C^con across income levels."""
        pref_cfg = FlexiblePreferenceConfig(mu_s=0.20)
        P_C = np.full((1, 11, 77), 1.25)
        Y_con_0 = empirical_calib.l_endow + empirical_calib.k_endow + (
            empirical_calib.T if empirical_calib.T is not None else 0.0
        )

        for mult in np.logspace(-2, 4, 15):
            Y_curr = Y_con_0 * mult
            c_dem = compute_stone_geary_final_demand(Y_curr, P_C, empirical_calib, pref_cfg)
            tot_exp = np.sum(P_C * c_dem, axis=1, keepdims=True)

            theta_hh = empirical_calib.theta[:, 0:1, :] if empirical_calib.theta is not None else 1.0 / 3.0
            Y_C_expected = theta_hh * Y_curr.reshape((1, 1, 77))
            rel_budget_err = np.max(np.abs(tot_exp - Y_C_expected) / np.maximum(Y_C_expected, 1e-12))
            assert rel_budget_err < 1e-12, f"Budget identity violated at mult={mult}: err={rel_budget_err}"

    def test_armington_purchaser_prices_autarky_overflow(self, empirical_calib: TradeCalibrationResult) -> None:
        """Reproduction of Defect 2: Armington purchaser prices under autarky tariffs and high trade elasticity."""
        p = np.ones((1, 11, 77))
        tau_fd = np.ones((11 * 77, 3, 77), dtype=float)
        # Impose prohibitive import tariffs on destination country 0
        for c_orig in range(1, 77):
            tau_fd[c_orig * 11 : (c_orig + 1) * 11, :, 0] = 1e7

        P_C = compute_armington_purchaser_prices(p, tau_fd, empirical_calib, sigma_trade=50.0)
        assert np.all(np.isfinite(P_C)), "P_C contains NaN due to max reference price factorization"

    def test_armington_final_demands_relative_price_clamping(self, empirical_calib: TradeCalibrationResult) -> None:
        """Verify that household bilateral final demand deliveries remain finite and non-negative."""
        p = np.ones((1, 11, 77))
        tau_fd = np.ones((11 * 77, 3, 77), dtype=float)
        # Disparate price spread: some tariffs 1e4, others 1e-2
        tau_fd[:11, :, 0] = 1e4
        tau_fd[11:22, :, 0] = 1e-2

        c_sec = np.ones((11, 3, 77))
        P_C = np.ones((11, 3, 77))
        xc = compute_armington_final_demands(c_sec, P_C, p, tau_fd, empirical_calib, sigma_trade=8.0)
        assert np.all(np.isfinite(xc))
        # Household final demand (category 0) must be strictly non-negative
        assert np.all(xc[:, 0, :] >= 0.0)


# =============================================================================
# 3. Atkeson-Burstein Markups Adversarial Tests
# =============================================================================

class TestAtkesonBursteinMarkupsAdversarial:
    """Stress-tests for Cournot-Armington imperfect competition and variable markups."""

    def test_single_domestic_monopolist_limit(self) -> None:
        """When destination market share s_ni -> 1.0, markup must converge to monopoly limit theta_j / (theta_j - 1)."""
        sigma_j = 6.0
        theta_j = 2.0
        cfg = FlexibleMarketStructureConfig(sigma_j=sigma_j, theta_j=theta_j, markup_max=10.0)

        s_mono = np.array([1.0])
        mu_mono, _ = compute_atkeson_burstein_markups(s_ni=s_mono, market_cfg=cfg)
        expected_mono = theta_j / (theta_j - 1.0)  # 2.0
        assert np.isclose(mu_mono[0], expected_mono), f"Monopolist markup was {mu_mono[0]}, expected {expected_mono}"

    def test_infinitesimal_fringe_limit(self) -> None:
        """When destination market share s_ni -> 0.0, markup must converge to fringe limit sigma_j / (sigma_j - 1)."""
        sigma_j = 6.0
        theta_j = 2.0
        cfg = FlexibleMarketStructureConfig(sigma_j=sigma_j, theta_j=theta_j, markup_max=10.0)

        s_fringe = np.array([0.0])
        mu_fringe, _ = compute_atkeson_burstein_markups(s_ni=s_fringe, market_cfg=cfg)
        expected_fringe = sigma_j / (sigma_j - 1.0)  # 1.20
        assert np.isclose(mu_fringe[0], expected_fringe), f"Fringe markup was {mu_fringe[0]}, expected {expected_fringe}"

    def test_markup_monotonicity_in_market_share(self) -> None:
        """Under theta_j < sigma_j, markups must be strictly monotonically increasing in market share."""
        cfg = FlexibleMarketStructureConfig(sigma_j=8.0, theta_j=2.5, markup_max=10.0)
        s_grid = np.linspace(0.0, 1.0, 100)
        mu_vals, _ = compute_atkeson_burstein_markups(s_ni=s_grid, market_cfg=cfg)

        diffs = np.diff(mu_vals)
        assert np.all(diffs > 0), "Markup was not strictly increasing in market share"

    def test_unnormalized_and_negative_market_shares_clamping(self) -> None:
        """Adversarial unnormalized or negative market shares from solver divergence must be safely clamped."""
        cfg = FlexibleMarketStructureConfig(
            sigma_j=6.0,
            theta_j=2.0,
            markup_min=1.0,
            markup_max=5.0,
            clamping_threshold=1e-4,
        )
        s_adversarial = np.array([-10.0, -1.0, 0.0, 0.5, 1.0, 2.0, 10.0, 100.0])
        mu_res, _ = compute_atkeson_burstein_markups(s_ni=s_adversarial, market_cfg=cfg)

        assert np.all(np.isfinite(mu_res))
        assert np.all(mu_res >= 1.0), "Markup violated minimum lower bound 1.0"
        assert np.all(mu_res <= 5.0), "Markup violated maximum upper bound 5.0"

    def test_markups_frame_structure_and_bounds(self, empirical_calib: TradeCalibrationResult) -> None:
        """Verify post-solve markups DataFrame generation and numerical bounds."""
        cfg = FlexibleTradeModelConfig(
            market_structure=FlexibleMarketStructureConfig(variable_markups=True)
        )
        res = solve_flexible_trade_equilibrium(empirical_calib, config=cfg, max_iter=0)
        df_m = res.markups
        assert isinstance(df_m, pd.DataFrame)
        assert not df_m.empty
        assert set(df_m.columns) == {"origin", "destination", "sector", "markup"}
        assert np.all(df_m["markup"] >= 1.0)
        assert np.all(df_m["markup"] <= 5.0)

        df_summary = res.summary_markups()
        assert isinstance(df_summary, pd.DataFrame)
        assert "sector" in df_summary.columns
        assert "mean" in df_summary.columns


# =============================================================================
# 4. Dixit-Stiglitz Variety Condensation Adversarial Tests
# =============================================================================

class TestVarietyCondensationAdversarial:
    """Stress-tests for zero-profit Dixit-Stiglitz variety condensation and price scaling."""

    def test_extreme_fixed_cost_escalation(self) -> None:
        """Massive fixed costs fl, fk -> 10^8 must drive variety count N -> 0 smoothly."""
        N = compute_dixit_stiglitz_varieties(pi_op=50.0, w=1.0, r=1.0, fl=1e8, fk=1e8)
        assert np.isclose(N, 2.5e-7)
        p_eff = compute_variety_price_scaling(p=1.0, N=N, N0=1.0, sigma_j=6.0)
        assert np.isfinite(p_eff)
        assert p_eff > 1.0, "Love of variety effect requires price index increase when N drops"

    def test_microscopic_fixed_cost_explosion(self) -> None:
        """Microscopic fixed overhead fl = 10^-12 must produce huge variety expansion without numerical overflow."""
        N = compute_dixit_stiglitz_varieties(pi_op=100.0, w=1.0, r=1.0, fl=1e-12, fk=0.0)
        assert N == 1e14
        p_eff = compute_variety_price_scaling(p=1.0, N=N, N0=1.0, sigma_j=6.0)
        assert np.isfinite(p_eff)
        assert 0.0 < p_eff < 1.0, "Effective price must fall under variety expansion"

    def test_zero_fixed_cost_invariance(self) -> None:
        """When fixed costs fl <= 0 and fk <= 0, variety count must fall back to constant N = 1.0."""
        assert compute_dixit_stiglitz_varieties(pi_op=100.0, fl=0.0, fk=0.0) == 1.0
        assert compute_dixit_stiglitz_varieties(pi_op=100.0, fl=-1.0, fk=-1.0) == 1.0

    def test_deep_recession_operating_loss(self) -> None:
        """Operating profits pi_op <= 0 must condense firm numbers N to exactly 0.0."""
        for pi in [0.0, -10.0, -1e6]:
            N = compute_dixit_stiglitz_varieties(pi_op=pi, w=1.0, r=1.0, fl=1.0, fk=1.0)
            assert N == 0.0, f"Firm count was non-zero for pi={pi}: {N}"

    def test_variety_price_scaling_collapse_clamping(self) -> None:
        """When N = 0.0, love of variety effective price index must clamp at 10^-12 ratio without inf/nan."""
        p_eff = compute_variety_price_scaling(p=1.0, N=0.0, N0=1.0, sigma_j=6.0)
        assert np.isfinite(p_eff)
        expected = (1e-12) ** (1.0 / (1.0 - 6.0))  # 10^2.4 ~ 251.19
        assert np.isclose(p_eff, expected)

    def test_variety_price_scaling_baseline_identity(self) -> None:
        """When N == N0, price scaling factor must equal 1.0 to float64 machine precision."""
        scale = compute_variety_price_scaling(p=1.0, N=1.0, N0=1.0, sigma_j=6.0)
        assert scale == 1.0


# =============================================================================
# 5. Baseline Calibration Invariance & Systemic Verification Tests
# =============================================================================

class TestCalibrationInvarianceAndSystemicVerification:
    """Stress-tests for general equilibrium residual invariance, factor balance, and diagnostics."""

    def test_baseline_calibration_exact_invariance(self, empirical_calib: TradeCalibrationResult) -> None:
        """Baseline CGE residuals must match between baseline and flexible engine to <= 10^-10."""
        from puremacro.trade.solver import build_initial_guess

        x0 = build_initial_guess(empirical_calib)
        f_base = compute_equilibrium_residuals(x0, empirical_calib)

        cfg_default = FlexibleTradeModelConfig()
        res_flex = solve_flexible_trade_equilibrium(empirical_calib, config=cfg_default, max_iter=0)
        f_flex = res_flex.residuals

        max_dev = np.max(np.abs(f_flex - f_base))
        assert max_dev <= 1e-10, f"Baseline residual invariance violated: max_dev={max_dev}"

    def test_baseline_factor_demands_exact_replication(self, empirical_calib: TradeCalibrationResult) -> None:
        """At baseline prices, normalized factor demands must match calibrated endowments to machine relative error (< 10^-12)."""
        cfg = FlexibleTechnologyConfig()
        c_va, c_y = compute_nested_ces_costs(
            r=np.ones((1, 11, 77)),
            w=np.ones((1, 11, 77)),
            P_M=np.ones((1, 11, 77)),
            calib=empirical_calib,
            tech_cfg=cfg,
        )
        xl, xk, _ = compute_nested_factor_demands(
            ytot=empirical_calib.ytot,
            r=np.ones((1, 11, 77)),
            w=np.ones((1, 11, 77)),
            P_M=np.ones((1, 11, 77)),
            c_va=c_va,
            c_y=c_y,
            p=np.ones((1, 11, 77)),
            tau=np.ones((11 * 77, 11, 77)),
            calib=empirical_calib,
            tech_cfg=cfg,
        )
        l_sum = np.sum(xl, axis=1).ravel()
        k_sum = np.sum(xk, axis=1).ravel()

        rel_err_l = np.max(np.abs(l_sum - empirical_calib.l_endow.ravel()) / empirical_calib.l_endow.ravel())
        rel_err_k = np.max(np.abs(k_sum - empirical_calib.k_endow.ravel()) / empirical_calib.k_endow.ravel())
        assert rel_err_l < 1e-12, f"Baseline labor demand relative mismatch: {rel_err_l}"
        assert rel_err_k < 1e-12, f"Baseline capital demand relative mismatch: {rel_err_k}"

    def test_factor_allocation_frame_endowment_balance(self, empirical_calib: TradeCalibrationResult) -> None:
        """Factor allocation DataFrame must aggregate to national labor and capital endowments."""
        res = solve_flexible_trade_equilibrium(empirical_calib, max_iter=0)
        df_factors = res.factor_allocation_frame()
        assert not df_factors.empty
        assert set(df_factors.columns) == {"country", "sector", "labor", "capital", "xl", "xk"}

        c_codes = empirical_calib.country_codes
        for i, code in enumerate(c_codes):
            sub = df_factors[df_factors["country"] == code]
            tot_l = sub["labor"].sum()
            tot_k = sub["capital"].sum()
            expected_l = float(empirical_calib.l_endow.ravel()[i])
            expected_k = float(empirical_calib.k_endow.ravel()[i])
            assert np.isclose(tot_l, expected_l, rtol=1e-12)
            assert np.isclose(tot_k, expected_k, rtol=1e-12)

    def test_welfare_decomposition_self_identity(self, empirical_calib: TradeCalibrationResult) -> None:
        """Hicksian Equivalent Variation under self-comparison must be identically zero."""
        res = solve_flexible_trade_equilibrium(empirical_calib, max_iter=0)
        df_w = res.welfare_decomposition(res)
        assert np.all(df_w["EV"] == 0.0)
        assert np.all(df_w["terms_of_trade"] == 0.0)
        assert np.all(df_w["efficiency"] == 0.0)

    def test_convergence_diagnostics_market_mapping(self, empirical_calib: TradeCalibrationResult) -> None:
        """Verify convergence diagnostics identify worst-offending equation and remediation."""
        from puremacro.trade.solver import build_initial_guess

        x0 = build_initial_guess(empirical_calib)
        f_vec = compute_equilibrium_residuals(x0, empirical_calib)
        diag = compute_convergence_diagnostics(f_vec, empirical_calib)

        assert "top_equations" in diag
        assert "max_residual" in diag
        assert "remediation" in diag
        assert len(diag["top_equations"]) <= 3

        top1 = diag["top_equations"][0]
        assert "block" in top1
        assert "country" in top1
        assert "equation" in top1
        assert isinstance(top1["residual"], float)
