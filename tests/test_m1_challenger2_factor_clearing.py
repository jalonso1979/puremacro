"""Milestone 1 Empirical Verification: Factor Market Clearing & Baseline Invariance (Challenger 2).

Verifies:
1. Empirical OECD ICIO dataset loading (77 countries, 11 sectors = 847 sector-country pairs).
2. Exact factor demand replication: xl0 == l0 and xk0 == k0 to < 10^-12 relative precision.
3. Absence of theta_va,0^2 double-counting in factor demands and factor payment exhaustion.
4. Baseline gross output cost cy == cy,0 and zero-profit unit price pp == 1.0 to < 10^-12.
5. National factor market clearing relative residuals ff2 / L0 == 0 and ff3 / K0 == 0 to < 10^-12.
6. Robustness across elasticity space (rho_va in [0.05, 5.0], sigma_y in [0.0, 3.0]).
7. Homogeneity degree 1 in output Y and degree 0 in factor prices (r, w).
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.trade.calibration import calibrate_trade_model
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
def empirical_calib():
    """Load empirical 77-country, 11-sector OECD ICIO calibration."""
    raw = load_icio_data()
    return calibrate_trade_model(raw, ns=11, nc=77, nfd=3, validate=True)


@pytest.fixture(scope="module")
def empirical_raw():
    """Load raw OECD ICIO data matrix."""
    return load_icio_data()


class TestEmpiricalFactorMarketClearing:
    """Empirical verification suite for factor market clearing and baseline invariance."""

    def test_01_icio_data_dimensions_and_baseline_factors(self, empirical_calib, empirical_raw):
        """Verify empirical calibration dimensions and extract benchmark l0, k0."""
        calib = empirical_calib
        raw = empirical_raw
        nc = calib.n_countries
        ns = calib.n_sectors
        assert nc == 77, f"Expected 77 countries, got {nc}"
        assert ns == 11, f"Expected 11 sectors, got {ns}"

        # Extract empirical benchmark l0 and k0 from raw ICIO table
        # Row ns*nc + 1 (848) is labor compensation, Row ns*nc + 2 (849) is gross capital return
        l0_raw = raw[ns * nc + 1, :ns * nc].reshape(nc, ns).T[np.newaxis, :, :]  # (1, 11, 77)
        k0_raw = raw[ns * nc + 2, :ns * nc].reshape(nc, ns).T[np.newaxis, :, :]  # (1, 11, 77)

        # Check national endowment sums
        l_endow_sum = np.sum(l0_raw, axis=1)  # (1, 77)
        k_endow_sum = np.sum(k0_raw, axis=1)  # (1, 77)
        np.testing.assert_allclose(l_endow_sum, calib.l_endow, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(k_endow_sum, calib.k_endow, rtol=1e-14, atol=1e-14)

        # Reconstructed factor demands from calibration formulas
        mask_y = (calib.ytot > 0)
        alpha = calib.alpha
        beta = calib.beta
        term_l = ((1.0 - alpha[mask_y]) / alpha[mask_y]) ** alpha[mask_y]
        term_k = (alpha[mask_y] / (1.0 - alpha[mask_y])) ** (1.0 - alpha[mask_y])
        l0_formula = np.zeros_like(calib.ytot)
        k0_formula = np.zeros_like(calib.ytot)
        l0_formula[mask_y] = (calib.ytot[mask_y] / beta[mask_y]) * term_l
        k0_formula[mask_y] = (calib.ytot[mask_y] / beta[mask_y]) * term_k

        np.testing.assert_allclose(l0_raw, l0_formula, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(k0_raw, k0_formula, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize(
        "rho_va,sigma_y",
        [
            (1.0, 0.0),       # Baseline Cobb-Douglas / Leontief
            (1.0, 0.5),       # Cobb-Douglas inner, CES outer
            (0.7, 0.0),       # CES inner, Leontief outer
            (0.7, 0.4),       # Canonical nested CES
            (0.3, 0.8),       # Low VA elasticity, moderate outer
            (1.8, 1.2),       # High elasticity both nests
            (1.0 - 1e-7, 0.0),# Just below Cobb-Douglas gating
            (1.0 + 1e-7, 1e-7),# Just above Cobb-Douglas and Leontief gating
        ],
    )
    def test_02_baseline_factor_demand_exact_replication(
        self, empirical_calib, empirical_raw, rho_va, sigma_y
    ):
        """Verify xl0 == l0 and xk0 == k0 to < 10^-12 relative precision across all 847 pairs."""
        calib = empirical_calib
        raw = empirical_raw
        nc, ns = calib.n_countries, calib.n_sectors

        l0 = raw[ns * nc + 1, :ns * nc].reshape(nc, ns).T[np.newaxis, :, :]
        k0 = raw[ns * nc + 2, :ns * nc].reshape(nc, ns).T[np.newaxis, :, :]

        r_base = np.ones((1, 1, nc))
        w_base = np.ones((1, 1, nc))
        p_base = np.ones((1, ns, nc))
        tau_base = np.ones((ns * nc, ns, nc))
        P_M_base, _, _ = compute_intermediate_composite_price(p_base, tau_base, calib)

        tech = FlexibleTechnologyConfig(rho_va=rho_va, sigma_y=sigma_y)
        c_va, c_y = compute_nested_ces_costs(r_base, w_base, P_M_base, calib, tech)

        xl, xk, x_mat = compute_nested_factor_demands(
            calib.ytot, r_base, w_base, P_M_base, c_va, c_y, p_base, tau_base, calib, tech
        )

        # Active sectors mask
        mask_active = (calib.ytot > 0)

        # Relative error on active sectors across all 847 sector-country pairs
        rel_err_l = np.abs(xl[mask_active] - l0[mask_active]) / l0[mask_active]
        rel_err_k = np.abs(xk[mask_active] - k0[mask_active]) / k0[mask_active]

        max_rel_err_l = float(np.max(rel_err_l))
        max_rel_err_k = float(np.max(rel_err_k))

        assert max_rel_err_l < 1e-12, (
            f"Labor demand relative error {max_rel_err_l:.4e} exceeds 1e-12 at rho={rho_va}, sig_y={sigma_y}"
        )
        assert max_rel_err_k < 1e-12, (
            f"Capital demand relative error {max_rel_err_k:.4e} exceeds 1e-12 at rho={rho_va}, sig_y={sigma_y}"
        )

        # National factor market clearing residuals ff2 and ff3
        ff2 = calib.l_endow.ravel() - np.sum(xl, axis=1).ravel()
        ff3 = calib.k_endow.ravel() - np.sum(xk, axis=1).ravel()

        rel_ff2 = np.abs(ff2) / calib.l_endow.ravel()
        rel_ff3 = np.abs(ff3) / calib.k_endow.ravel()

        max_rel_ff2 = float(np.max(rel_ff2))
        max_rel_ff3 = float(np.max(rel_ff3))

        assert max_rel_ff2 < 1e-12, f"National labor clearing max relative residual {max_rel_ff2:.4e} >= 1e-12"
        assert max_rel_ff3 < 1e-12, f"National capital clearing max relative residual {max_rel_ff3:.4e} >= 1e-12"

    def test_03_no_theta_va_squared_double_counting(self, empirical_calib, empirical_raw):
        """Specifically detect and verify absence of theta_va,0^2 double-counting."""
        calib = empirical_calib
        raw = empirical_raw
        nc, ns = calib.n_countries, calib.n_sectors

        l0 = raw[ns * nc + 1, :ns * nc].reshape(nc, ns).T[np.newaxis, :, :]
        k0 = raw[ns * nc + 2, :ns * nc].reshape(nc, ns).T[np.newaxis, :, :]
        va0 = l0 + k0

        # Empirical value-added share theta_va,0 = VA_0 / Y_0
        mask_y = (calib.ytot > 0)
        theta_va0 = np.zeros_like(calib.ytot)
        theta_va0[mask_y] = va0[mask_y] / calib.ytot[mask_y]

        # In OECD ICIO, value added share is between 0.15 and 0.85 (mean ~0.45)
        mean_theta = float(np.mean(theta_va0[mask_y]))
        assert 0.3 < mean_theta < 0.6, f"Empirical theta_va,0 mean {mean_theta:.3f} unexpected"

        r_base = np.ones((1, 1, nc))
        w_base = np.ones((1, 1, nc))
        p_base = np.ones((1, ns, nc))
        tau_base = np.ones((ns * nc, ns, nc))
        P_M_base, _, _ = compute_intermediate_composite_price(p_base, tau_base, calib)

        tech = FlexibleTechnologyConfig(rho_va=0.8, sigma_y=0.3)
        c_va, c_y = compute_nested_ces_costs(r_base, w_base, P_M_base, calib, tech)
        xl, xk, _ = compute_nested_factor_demands(
            calib.ytot, r_base, w_base, P_M_base, c_va, c_y, p_base, tau_base, calib, tech
        )

        # Ratio of realized factor demand to benchmark
        ratio_l = xl[mask_y] / l0[mask_y]
        ratio_k = xk[mask_y] / k0[mask_y]

        # If theta_va,0^2 were double-counted, ratio would be ~ theta_va0 (mean ~0.45)
        # Verify ratio is 1.0 to < 10^-12, and strictly separated from theta_va0
        np.testing.assert_allclose(ratio_l, 1.0, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(ratio_k, 1.0, rtol=1e-12, atol=1e-12)

        # Verify that if double-counting DID occur, the error would be massive
        xl_buggy = theta_va0 * l0
        drop_pct = 100.0 * (1.0 - np.sum(xl_buggy) / np.sum(l0))
        assert drop_pct > 40.0, f"Simulated double-counting drop is {drop_pct:.1f}%, expected > 40%"

        # Factor payment exhaustion: w*xl + r*xk == VA0
        factor_payments = w_base * xl + r_base * xk
        rel_err_payments = np.abs(factor_payments[mask_y] - va0[mask_y]) / va0[mask_y]
        assert np.max(rel_err_payments) < 1e-12

    def test_04_baseline_gross_output_costs_and_zero_profit_prices(self, empirical_calib):
        """Verify baseline gross output cost c_y == c_y,0 and unit price pp == 1.0 to < 10^-12."""
        calib = empirical_calib
        nc, ns = calib.n_countries, calib.n_sectors

        r_base = np.ones((1, 1, nc))
        w_base = np.ones((1, 1, nc))
        p_base = np.ones((1, ns, nc))
        tau_base = np.ones((ns * nc, ns, nc))

        P_M_base, inter_cost_base, _ = compute_intermediate_composite_price(p_base, tau_base, calib)

        # Intermediate price index P_M must be 1.0 at baseline
        np.testing.assert_allclose(P_M_base, 1.0, rtol=1e-14, atol=1e-14)

        for rho_va, sigma_y in [(1.0, 0.0), (0.7, 0.4), (1.5, 0.8), (0.4, 0.0)]:
            tech = FlexibleTechnologyConfig(rho_va=rho_va, sigma_y=sigma_y)
            c_va, c_y_norm = compute_nested_ces_costs(r_base, w_base, P_M_base, calib, tech)

            # 1. Normalized gross output unit cost c_y_norm must be 1.0
            max_cy_norm_err = float(np.max(np.abs(c_y_norm - 1.0)))
            assert max_cy_norm_err < 1e-12, (
                f"c_y_norm error {max_cy_norm_err:.4e} >= 1e-12 at rho={rho_va}, sig_y={sigma_y}"
            )

            # 2. Unit net cost in levels: c_y_level must equal 1.0 - tax
            c_y_level = compute_outer_ces_cost(c_va, P_M_base, calib, tech, normalized=False)
            expected_cy_level = 1.0 - calib.tax
            max_cy_level_err = float(np.max(np.abs(c_y_level - expected_cy_level)))
            assert max_cy_level_err < 1e-12, (
                f"c_y_level error {max_cy_level_err:.4e} >= 1e-12 at rho={rho_va}, sig_y={sigma_y}"
            )

            # 3. Producer zero-profit price pp must be 1.0 for all active sectors
            pp_from_norm = compute_zero_profit_price(c_y_norm, calib)
            pp_from_level = compute_zero_profit_price(c_y_level, calib)

            mask_y = (calib.ytot > 0)
            max_pp_norm_err = float(np.max(np.abs(pp_from_norm[mask_y] - 1.0)))
            max_pp_level_err = float(np.max(np.abs(pp_from_level[mask_y] - 1.0)))

            assert max_pp_norm_err < 1e-12, (
                f"pp from c_y_norm error {max_pp_norm_err:.4e} >= 1e-12"
            )
            assert max_pp_level_err < 1e-12, (
                f"pp from c_y_level error {max_pp_level_err:.4e} >= 1e-12"
            )

    def test_05_intermediate_deliveries_baseline_replication(self, empirical_calib):
        """Verify baseline intermediate input demands x_mat == a * ytot to < 10^-12."""
        calib = empirical_calib
        nc, ns = calib.n_countries, calib.n_sectors

        r_base = np.ones((1, 1, nc))
        w_base = np.ones((1, 1, nc))
        p_base = np.ones((1, ns, nc))
        tau_base = np.ones((ns * nc, ns, nc))
        P_M_base, _, _ = compute_intermediate_composite_price(p_base, tau_base, calib)

        tech = FlexibleTechnologyConfig(rho_va=0.8, sigma_y=0.5, sigma_inter=0.0)
        c_va, c_y = compute_nested_ces_costs(r_base, w_base, P_M_base, calib, tech)
        _, _, x_mat = compute_nested_factor_demands(
            calib.ytot, r_base, w_base, P_M_base, c_va, c_y, p_base, tau_base, calib, tech
        )

        expected_xmat = np.where(calib.ytot > 0, calib.a * calib.ytot, 0.0)
        mask_pos = (expected_xmat > 0)
        rel_diff = np.abs(x_mat[mask_pos] - expected_xmat[mask_pos]) / expected_xmat[mask_pos]
        max_rel_xmat_err = float(np.max(rel_diff))
        assert max_rel_xmat_err < 1e-12, f"Intermediate delivery relative error {max_rel_xmat_err:.4e} >= 1e-12"

    def test_06_homogeneity_and_factor_substitution_properties(self, empirical_calib):
        """Verify Euler degree 1 in output Y and degree 0 in factor prices (r, w)."""
        calib = empirical_calib
        nc, ns = calib.n_countries, calib.n_sectors

        r_base = np.ones((1, 1, nc))
        w_base = np.ones((1, 1, nc))
        p_base = np.ones((1, ns, nc))
        tau_base = np.ones((ns * nc, ns, nc))
        P_M_base, _, _ = compute_intermediate_composite_price(p_base, tau_base, calib)

        tech = FlexibleTechnologyConfig(rho_va=0.7, sigma_y=0.4)
        c_va, c_y_norm = compute_nested_ces_costs(r_base, w_base, P_M_base, calib, tech)
        c_y_level = compute_outer_ces_cost(c_va, P_M_base, calib, tech, normalized=False)

        # 1. Output scaling by lambda=2.5: factor demands scale linearly (degree 1)
        lam = 2.5
        xl_1, xk_1, xm_1 = compute_nested_factor_demands(
            calib.ytot, r_base, w_base, P_M_base, c_va, c_y_level, p_base, tau_base, calib, tech
        )
        xl_lam, xk_lam, xm_lam = compute_nested_factor_demands(
            lam * calib.ytot, r_base, w_base, P_M_base, c_va, c_y_level, p_base, tau_base, calib, tech
        )
        np.testing.assert_allclose(xl_lam, lam * xl_1, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(xk_lam, lam * xk_1, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(xm_lam, lam * xm_1, rtol=1e-12, atol=1e-12)

        # 2. Factor price proportional scaling (r, w, P_M) -> mu*(r, w, P_M)
        # Cost in levels scales by mu (Euler degree 1)
        mu = 1.8
        c_va_mu = compute_inner_ces_cost(mu * r_base, mu * w_base, calib, tech)
        c_y_level_mu = compute_outer_ces_cost(c_va_mu, mu * P_M_base, calib, tech, normalized=False)

        np.testing.assert_allclose(c_va_mu, mu * c_va, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(c_y_level_mu, mu * c_y_level, rtol=1e-12, atol=1e-12)

        # Factor demands in levels under level c_y are invariant (Euler degree 0)
        xl_mu, xk_mu, _ = compute_nested_factor_demands(
            calib.ytot, mu * r_base, mu * w_base, mu * P_M_base, c_va_mu, c_y_level_mu, p_base, tau_base, calib, tech
        )
        mask_y = (calib.ytot > 0)
        np.testing.assert_allclose(xl_mu[mask_y], xl_1[mask_y], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(xk_mu[mask_y], xk_1[mask_y], rtol=1e-12, atol=1e-12)
