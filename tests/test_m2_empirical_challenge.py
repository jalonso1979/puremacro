"""Empirical Stress-Testing Suite for Milestone 2: Preferences & Armington Sourcing.

Authored by Empirical Challenger (Challenger 1, Milestone 2).
Covers:
1. Extreme subsistence parameters: mu_s in {0.0, 1e-6, 0.5, 0.9, 0.999} and expenditure adding-up (< 1e-14).
2. Income scaling: u in [1e-4, 1e4], monotonicity of g(u), and saturation at 1/tanh(3) ~ 1.00497.
3. Engel curves: Engel's Law verification for luxury vs necessity sectors.
4. Extreme trade elasticities: sigma_trade in {1.001, 2.0, 5.0, 8.0, 20.0} and price stability.
5. Parameter validation and boundary guards.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from puremacro.trade import calibrate_trade_model
from puremacro.trade.data import load_icio_data
from puremacro.trade.flexible import (
    FlexiblePreferenceConfig,
    _extract_benchmark_household_data,
    compute_armington_final_demands,
    compute_armington_purchaser_prices,
    compute_les_marginal_budget_shares,
    compute_multi_category_final_demands,
    compute_stone_geary_final_demand,
    smooth_subsistence_scaling,
)


@pytest.fixture(scope="module")
def synthetic_2c_2s_calib():
    """Construct balanced 2-country 2-sector synthetic calibration."""
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
    labor = (2.0 / 3.0) * va_fac
    capital = (1.0 / 3.0) * va_fac
    data[4, :4] = taxes
    data[5, :4] = labor
    data[6, :4] = capital
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
def icio_77c_11s_calib():
    """Calibrate empirical 77-country 11-sector OECD ICIO benchmark."""
    return calibrate_trade_model(load_icio_data(), ns=11, nc=77, nfd=3, validate=True)


class TestExtremeSubsistenceAddingUp:
    """Stress-test expenditure adding-up under extreme subsistence shares mu_s."""

    @pytest.mark.parametrize("mu_s", [0.0, 1e-6, 0.5, 0.9, 0.999])
    def test_adding_up_at_baseline_income(self, synthetic_2c_2s_calib, mu_s):
        """Expenditure adding-up holds to < 1e-14 at baseline income on synthetic model."""
        calib = synthetic_2c_2s_calib
        Ycon_0, E_C_0, _, _, _ = _extract_benchmark_household_data(calib)
        P_C = np.ones((1, calib.n_sectors, calib.n_countries))
        pref = FlexiblePreferenceConfig(mu_s=mu_s)

        c_C = compute_stone_geary_final_demand(Ycon_0, P_C, calib, pref)
        tot_exp = np.sum(P_C * c_C, axis=1, keepdims=True)

        rel_err = np.max(np.abs(tot_exp - E_C_0) / E_C_0)
        assert rel_err < 1e-14, f"Relative error {rel_err} exceeds 1e-14 for mu_s={mu_s}"

    @pytest.mark.parametrize("mu_s", [0.0, 1e-6, 0.5, 0.9, 0.999])
    def test_adding_up_on_icio_benchmark(self, icio_77c_11s_calib, mu_s):
        """Expenditure adding-up holds to < 1e-14 on full 77c x 11s ICIO dataset at baseline."""
        calib = icio_77c_11s_calib
        Ycon_0, E_C_0, _, _, _ = _extract_benchmark_household_data(calib)
        P_C = np.ones((1, calib.n_sectors, calib.n_countries))
        pref = FlexiblePreferenceConfig(mu_s=mu_s)

        c_C = compute_stone_geary_final_demand(Ycon_0, P_C, calib, pref)
        tot_exp = np.sum(P_C * c_C, axis=1, keepdims=True)

        rel_err = np.max(np.abs(tot_exp - E_C_0) / E_C_0)
        assert rel_err < 1e-14, f"Relative error {rel_err} exceeds 1e-14 for mu_s={mu_s}"

    @pytest.mark.parametrize("mu_s", [0.0, 1e-6, 0.5, 0.9, 0.999])
    def test_adding_up_for_expanded_income(self, icio_77c_11s_calib, mu_s):
        """Expenditure adding-up holds to < 1e-14 across u in [1.0, 100.0]."""
        calib = icio_77c_11s_calib
        Ycon_0, E_C_0, _, _, _ = _extract_benchmark_household_data(calib)
        P_C = np.ones((1, calib.n_sectors, calib.n_countries))
        pref = FlexiblePreferenceConfig(mu_s=mu_s)

        for u in [1.0, 1.2, 2.0, 5.0, 20.0, 100.0]:
            Y_u = u * Ycon_0
            E_C_u = u * E_C_0
            c_C = compute_stone_geary_final_demand(Y_u, P_C, calib, pref)
            tot_exp = np.sum(P_C * c_C, axis=1, keepdims=True)
            rel_err = np.max(np.abs(tot_exp - E_C_u) / E_C_u)
            assert rel_err < 1e-14, f"Adding-up relative error {rel_err} at u={u} exceeds 1e-14"

    @pytest.mark.parametrize("mu_s", [0.0, 1e-6, 0.33])
    def test_adding_up_low_subsistence_sub_unity(self, icio_77c_11s_calib, mu_s):
        """When mu_s <= tanh(3)/3 ~ 0.33168, adding-up holds across all u in [1e-4, 1.0]."""
        calib = icio_77c_11s_calib
        Ycon_0, E_C_0, _, _, _ = _extract_benchmark_household_data(calib)
        P_C = np.ones((1, calib.n_sectors, calib.n_countries))
        pref = FlexiblePreferenceConfig(mu_s=mu_s)

        for u in np.logspace(-4, 0, 10):
            Y_u = u * Ycon_0
            E_C_u = u * E_C_0
            c_C = compute_stone_geary_final_demand(Y_u, P_C, calib, pref)
            tot_exp = np.sum(P_C * c_C, axis=1, keepdims=True)
            rel_err = np.max(np.abs(tot_exp - E_C_u) / E_C_u)
            assert rel_err < 1e-14, f"Adding-up relative error {rel_err} at u={u} exceeds 1e-14"

    def test_marginal_budget_shares_sum_to_one(self, icio_77c_11s_calib):
        """Marginal budget shares theta_s^LES sum strictly to 1.0 across sectors."""
        calib = icio_77c_11s_calib
        for mu_s in [0.0, 1e-6, 0.5, 0.9, 0.999]:
            pref = FlexiblePreferenceConfig(mu_s=mu_s)
            theta_LES = compute_les_marginal_budget_shares(calib, pref)
            sum_shares = np.sum(theta_LES, axis=1)
            err = np.max(np.abs(sum_shares - 1.0))
            assert err < 1e-14, f"theta_LES sum error {err} exceeds 1e-14 for mu_s={mu_s}"


class TestIncomeScalingAndSaturation:
    """Stress-test smooth subsistence scaling function g(u) over [1e-4, 1e4]."""

    def test_monotonicity_across_eight_decades(self):
        """g(u) is monotonically non-decreasing over 8 orders of magnitude [1e-4, 1e4]."""
        u_grid = np.logspace(-4, 4, 10000)
        g_vals = smooth_subsistence_scaling(u_grid)
        diffs = np.diff(g_vals)
        assert np.all(diffs >= 0.0), "Monotonicity violated in g(u)"

    def test_strict_monotonicity_pre_saturation(self):
        """g(u) is strictly increasing for u in [1e-4, 5.0] before IEEE-754 saturation."""
        u_grid = np.linspace(1e-4, 5.0, 5000)
        g_vals = smooth_subsistence_scaling(u_grid)
        diffs = np.diff(g_vals)
        assert np.all(diffs > 0.0), "Strict monotonicity violated in pre-saturation regime"

    def test_exact_asymptotic_saturation(self):
        """As u -> inf, g(u) smoothly saturates to 1/tanh(3) ~ 1.0049698233."""
        target_sat = 1.0 / np.tanh(3.0)
        assert abs(target_sat - 1.0049698233136894) < 1e-12

        g_inf = smooth_subsistence_scaling(1e4)
        assert abs(g_inf - target_sat) < 1e-15, f"g(1e4)={g_inf} deviates from {target_sat}"

    def test_machine_precision_baseline_invariance(self):
        """g(1.0) == 1.0 to float64 machine precision (< 1e-16)."""
        res = smooth_subsistence_scaling(1.0)
        assert abs(res - 1.0) < 1e-16, f"g(1.0)={res} not identically 1.0"

    def test_origin_vanishing_limit(self):
        """g(0.0) == 0.0 to float64 machine precision."""
        res = smooth_subsistence_scaling(0.0)
        assert abs(res) < 1e-16

    def test_derivative_smoothness_oracle(self):
        """Numerical derivative matches analytical 3*sech^2(3u)/tanh(3) across u grid."""
        u_samples = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 3.0])
        eps = 1e-7
        g_plus = smooth_subsistence_scaling(u_samples + eps)
        g_minus = smooth_subsistence_scaling(u_samples - eps)
        num_deriv = (g_plus - g_minus) / (2.0 * eps)
        analytical = 3.0 * (1.0 - np.tanh(3.0 * u_samples) ** 2) / np.tanh(3.0)
        np.testing.assert_allclose(num_deriv, analytical, rtol=1e-4, atol=1e-8)


class TestEngelCurvesLuxuryNecessity:
    """Stress-test non-homothetic consumption dynamics and Engel's Law."""

    def test_engel_law_sectoral_shares_monotonicity(self, icio_77c_11s_calib):
        """Necessity sectors decrease and luxury sectors increase expenditure share as income rises."""
        calib = icio_77c_11s_calib
        Ycon_0, E_C_0, _, _, _ = _extract_benchmark_household_data(calib)
        P_C = np.ones((1, calib.n_sectors, calib.n_countries))

        # Designate sector 0 as necessity (high mu_s=0.7) and sector 1 as luxury (low mu_s=0.1)
        mu_dict = {calib.sector_codes[0]: 0.7, calib.sector_codes[1]: 0.1}
        for s in calib.sector_codes[2:]:
            mu_dict[s] = 0.3
        pref = FlexiblePreferenceConfig(mu_s=mu_dict)

        u_grid = [0.8, 1.0, 1.5, 2.5, 5.0, 10.0]
        c_idx = 0  # Test country 0

        shares_nec = []
        shares_lux = []

        for u in u_grid:
            Y_u = u * Ycon_0
            c_C = compute_stone_geary_final_demand(Y_u, P_C, calib, pref)
            tot_exp = np.sum(P_C * c_C, axis=1, keepdims=True)
            w_nec = float((P_C[0, 0, c_idx] * c_C[0, 0, c_idx]) / tot_exp[0, 0, c_idx])
            w_lux = float((P_C[0, 1, c_idx] * c_C[0, 1, c_idx]) / tot_exp[0, 0, c_idx])
            shares_nec.append(w_nec)
            shares_lux.append(w_lux)

        # Engel's law: necessity expenditure share is strictly decreasing
        diff_nec = np.diff(shares_nec)
        assert np.all(diff_nec < 0.0), f"Necessity shares not strictly decreasing: {shares_nec}"

        # Luxury expenditure share is strictly increasing
        diff_lux = np.diff(shares_lux)
        assert np.all(diff_lux > 0.0), f"Luxury shares not strictly increasing: {shares_lux}"

    def test_asymptotic_convergence_to_marginal_budget_shares(self, icio_77c_11s_calib):
        """As income u -> inf, expenditure shares w_s converge to theta_s^LES."""
        calib = icio_77c_11s_calib
        Ycon_0, _, _, _, _ = _extract_benchmark_household_data(calib)
        P_C = np.ones((1, calib.n_sectors, calib.n_countries))

        mu_dict = {calib.sector_codes[0]: 0.6, calib.sector_codes[1]: 0.1}
        pref = FlexiblePreferenceConfig(mu_s=mu_dict)
        theta_LES = compute_les_marginal_budget_shares(calib, pref)

        # Very high income: u = 1000.0
        Y_huge = 1000.0 * Ycon_0
        c_C = compute_stone_geary_final_demand(Y_huge, P_C, calib, pref)
        tot_exp = np.sum(P_C * c_C, axis=1, keepdims=True)
        w_emp = (P_C * c_C) / tot_exp

        # Difference between empirical share and theta_LES should be < 0.002
        diff = np.max(np.abs(w_emp - theta_LES))
        assert diff < 0.002, f"Asymptotic convergence error {diff} too large"


class TestExtremeTradeElasticitiesAndArmingtonStability:
    """Stress-test Tier 2 Armington price index and sourcing across extreme sigma_trade."""

    @pytest.mark.parametrize("sig", [1.001, 2.0, 5.0, 8.0, 20.0])
    def test_linear_homogeneity_of_degree_one(self, icio_77c_11s_calib, sig):
        """Armington price index P_C(lambda * p) == lambda * P_C(p) for lambda in [0.01, 100]."""
        calib = icio_77c_11s_calib
        tau_fd = np.ones((calib.n_sectors * calib.n_countries, calib.n_final_demand, calib.n_countries))

        for scale in [0.01, 0.1, 1.0, 10.0, 100.0]:
            p = np.full((1, calib.n_sectors, calib.n_countries), scale)
            P = compute_armington_purchaser_prices(p, tau_fd, calib, sigma_trade=sig)
            P_hh = P[:, 0, :]
            rel_err = np.max(np.abs(P_hh - scale) / scale)
            assert rel_err < 1e-11, f"Homogeneity violated for sig={sig}, scale={scale}: {rel_err}"

    @pytest.mark.parametrize("sig", [1.001, 2.0, 5.0, 8.0, 20.0])
    def test_price_boundedness_under_heterogeneous_prices(self, icio_77c_11s_calib, sig):
        """Household purchaser price P_C is bounded strictly in [min(p), max(p)]."""
        calib = icio_77c_11s_calib
        tau_fd = np.ones((calib.n_sectors * calib.n_countries, calib.n_final_demand, calib.n_countries))
        rng = np.random.default_rng(999)
        p_rand = 0.5 + 1.5 * rng.random((1, calib.n_sectors, calib.n_countries))

        P = compute_armington_purchaser_prices(p_rand, tau_fd, calib, sigma_trade=sig)
        P_hh = P[:, 0, :]

        assert not np.isnan(P_hh).any(), f"NaN in P_hh for sig={sig}"
        assert not np.isinf(P_hh).any(), f"Inf in P_hh for sig={sig}"
        assert np.min(P_hh) >= 0.5 - 1e-12, f"P_hh dropped below min price for sig={sig}"
        assert np.max(P_hh) <= 2.0 + 1e-12, f"P_hh exceeded max price for sig={sig}"

    @pytest.mark.parametrize("sig", [1.001, 2.0, 5.0, 8.0, 20.0])
    def test_bilateral_armington_expenditure_adding_up(self, icio_77c_11s_calib, sig):
        """Bilateral trade deliveries satisfy sum_i p_i * tau_i * x_i == P_C * c_sec to < 1e-14."""
        calib = icio_77c_11s_calib
        nc, ns, nfd = calib.n_countries, calib.n_sectors, calib.n_final_demand
        tau_fd = np.ones((ns * nc, nfd, nc))
        rng = np.random.default_rng(101)
        p_rand = 0.8 + 0.4 * rng.random((1, ns, nc))
        c_sec = np.full((ns, nfd, nc), 50.0)

        P_C = compute_armington_purchaser_prices(p_rand, tau_fd, calib, sigma_trade=sig)
        xfd_4d = compute_armington_final_demands(
            c_sec, P_C, p_rand, tau_fd, calib, sigma_trade=sig, as_3d=False
        )

        p_4d = p_rand.squeeze(0)[:, :, np.newaxis, np.newaxis]
        tau_4d = tau_fd.reshape(nc, ns, nfd, nc).transpose(1, 0, 2, 3)
        p_tau = p_4d * tau_4d

        exp_origins = np.sum(p_tau * xfd_4d, axis=1)
        expected_exp = P_C * c_sec

        # Test Household category (cat 0)
        rel_err = np.max(
            np.abs(exp_origins[:, 0, :] - expected_exp[:, 0, :]) / expected_exp[:, 0, :]
        )
        assert rel_err < 1e-14, f"Bilateral adding-up relative error {rel_err} for sig={sig} exceeds 1e-14"

    def test_sigma_trade_substitution_effect(self, icio_77c_11s_calib):
        """As sigma_trade rises under price dispersion, mean purchaser price strictly falls."""
        calib = icio_77c_11s_calib
        tau_fd = np.ones((calib.n_sectors * calib.n_countries, calib.n_final_demand, calib.n_countries))
        rng = np.random.default_rng(42)
        p_rand = 0.6 + 1.2 * rng.random((1, calib.n_sectors, calib.n_countries))

        mean_prices = []
        for sig in [1.001, 2.0, 5.0, 8.0, 20.0]:
            P = compute_armington_purchaser_prices(p_rand, tau_fd, calib, sigma_trade=sig)
            mean_prices.append(float(np.mean(P[:, 0, :])))

        diffs = np.diff(mean_prices)
        assert np.all(diffs < 0.0), f"Mean price not strictly decreasing with sigma: {mean_prices}"


class TestParameterValidationAndGuards:
    """Stress-test input validation and error rejection."""

    def test_reject_sigma_trade_le_one(self):
        """sigma_trade <= 1.0 raises ValueError."""
        with pytest.raises(ValueError, match=r"sigma_trade.*strictly greater than 1\.0"):
            FlexiblePreferenceConfig(sigma_trade=1.0)
        with pytest.raises(ValueError, match=r"sigma_trade.*strictly greater than 1\.0"):
            FlexiblePreferenceConfig(sigma_trade=0.8)

    def test_reject_negative_subsistence_share(self):
        """mu_s < 0.0 raises ValueError."""
        with pytest.raises(ValueError, match=r"mu_s.*non-negative"):
            FlexiblePreferenceConfig(mu_s=-0.05)

    def test_reject_subsistence_share_ge_one(self):
        """mu_s >= 1.0 raises ValueError."""
        with pytest.raises(ValueError, match=r"mu_s.*strictly less than 1\.0"):
            FlexiblePreferenceConfig(mu_s=1.0)
        with pytest.raises(ValueError, match=r"mu_s.*strictly less than 1\.0"):
            FlexiblePreferenceConfig(mu_s=1.5)

    def test_reject_boolean_parameters(self):
        """Boolean values for sigma_trade or mu_s raise TypeError."""
        with pytest.raises(TypeError, match=r"sigma_trade cannot be a boolean"):
            FlexiblePreferenceConfig(sigma_trade=True)
        with pytest.raises(TypeError, match=r"mu_s cannot be a boolean"):
            FlexiblePreferenceConfig(mu_s=False)
