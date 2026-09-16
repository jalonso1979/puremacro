"""Empirical adversarial numerical precision test suite for CGE trade calibration.

Authored by m2_challenger_r2_1 (Numerical Precision Challenger) to independently
and adversarially stress-test:
1. Bit-level and machine-precision (< 1e-12 relative discrepancy) parity for all 9
   calibrated parameter arrays against reference MATLAB results_77c_11s_base.mat:
   - a (847, 11, 77)
   - afd (847, 3, 77)
   - alpha (1, 11, 77)
   - beta (1, 11, 77)
   - k_endow (1, 77)
   - l_endow (1, 77)
   - invforT (1, 77)
   - tax (1, 11, 77)
   - ytot (1, 11, 77)
2. Macro balance / market-clearing supply-demand identity across all 847 sector-country pairs.
3. Cost and value-added column share balance across all 847 sector-country pairs.
4. Robustness of masked division refactoring against inactive sectors, zeros,
   extreme factor ratios, and inventory disaccumulation.
5. Adversarial input validation sensitivity (asserting validate() flags corruption).
"""
from __future__ import annotations

from dataclasses import replace
import sys
import warnings

import numpy as np
import pytest

from puremacro.trade.calibration import calibrate_trade_model
from puremacro.trade.data import (
    get_country_codes,
    get_sector_codes,
    load_icio_data,
    load_reference_solution,
)

from conftest import load_or_skip


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# The 77x11 ICIO matrix ships inside ``puremacro.trade``, so ``load_icio_data``
# needs no file outside the installation.  A broken install could still lose it,
# and a setup ERROR on every test is the wrong signal there, so the loader keeps
# turning an unavailable-data failure into a skip.
#
# The bundled MATLAB reference solutions carry the *equilibrium* arrays (XN_sol,
# c_sol, pfd_sol, xx_sol, w_sol, ytot_sol, T_sol, r_sol, p_sol, tau_a, taufd_a).
# They do NOT carry the baseline *calibration* arrays this suite compares against
# (a, afd, alpha, beta, KT, LT, invforT, tax, ytot), which are an external check
# on puremacro and must not be regenerated with puremacro itself.  Every test that
# needs them takes ``matlab_base_arrays`` and skips by name until they are bundled.
_MATLAB_CALIBRATION_ARRAYS = (
    "a", "afd", "alpha", "beta", "KT", "LT", "invforT", "tax", "ytot",
)


def _load_icio_or_skip():
    """Return the bundled 77x11 ICIO matrix, or skip if it cannot be read.

    Without it ``load_icio_data`` raises FileNotFoundError, which a module
    fixture turns into a setup ERROR on all sixteen tests; a skip is the honest
    outcome for an installation that is missing its own data file.
    """
    return load_or_skip(load_icio_data)


@pytest.fixture(scope="module")
def reference_data():
    """Bundled raw ICIO matrix and the Python calibration derived from it."""
    raw_data = _load_icio_or_skip()
    calib = calibrate_trade_model(raw_data, nc=77, ns=11, nfd=3)
    return raw_data, calib


@pytest.fixture(scope="module")
def matlab_base_arrays():
    """Baseline MATLAB *calibration* arrays, or a skip naming what is absent."""
    arrays = load_reference_solution("base")
    missing = [n for n in _MATLAB_CALIBRATION_ARRAYS if n not in arrays]
    if missing:
        pytest.skip(
            f"MATLAB baseline calibration array(s) {', '.join(missing)} are not part of "
            "the reference solutions bundled with puremacro "
            f"(bundled: {', '.join(sorted(arrays))})"
        )
    return arrays


def test_reference_fixture_skips_without_the_private_icio_data(monkeypatch):
    """An installation without the bundled ICIO data must skip, not ERROR sixteen times."""

    def _missing(*_args, **_kwargs):
        raise FileNotFoundError("Could not automatically locate data_77c_11s.mat")

    monkeypatch.setattr(sys.modules[__name__], "load_icio_data", _missing)
    with pytest.raises(pytest.skip.Exception):
        _load_icio_or_skip()


# ---------------------------------------------------------------------------
# 1. Calibrated Parameter Arrays Numerical Parity Suite
# ---------------------------------------------------------------------------

class TestCalibratedArrayPrecisionParity:
    """Rigorous precision assertions for all 9 calibrated parameter arrays against MATLAB."""

    @pytest.mark.parametrize("arr_name,mat_key,max_abs_tol,max_rel_tol", [
        ("a", "a", 1e-14, 1e-12),
        ("afd", "afd", 1e-14, 1e-12),
        ("alpha", "alpha", 1e-15, 1e-12),
        ("beta", "beta", 1e-10, 1e-12),
        ("k_endow", "KT", 1e-4, 1e-12),
        ("l_endow", "LT", 1e-4, 1e-12),
        ("invforT", "invforT", 1e-4, 1e-12),
        ("tax", "tax", 1e-14, 1e-12),
        ("ytot", "ytot", 1e-4, 1e-12),
    ])
    def test_calibrated_array_precision(
        self, reference_data, matlab_base_arrays, arr_name, mat_key, max_abs_tol, max_rel_tol
    ):
        """Assert both absolute and relative discrepancies are strictly below thresholds."""
        _, calib = reference_data
        mat = matlab_base_arrays
        py_arr = getattr(calib, arr_name)
        mat_arr = mat[mat_key]

        assert py_arr.shape == mat_arr.shape, (
            f"Shape mismatch for {arr_name}: Python {py_arr.shape} vs MATLAB {mat_arr.shape}"
        )

        diff = np.abs(py_arr - mat_arr)
        max_abs = float(np.max(diff))
        denom = float(np.max(np.abs(mat_arr)))
        rel_diff = max_abs / denom if denom > 0 else max_abs

        assert max_abs < max_abs_tol, (
            f"Array '{arr_name}' max absolute error {max_abs:.4e} exceeds {max_abs_tol:.4e}"
        )
        assert rel_diff < max_rel_tol, (
            f"Array '{arr_name}' relative error {rel_diff:.4e} exceeds {max_rel_tol:.4e}"
        )

    def test_pointwise_relative_error_all_9_arrays(self, reference_data, matlab_base_arrays):
        """Assert pointwise relative error is strictly < 1e-12 across all non-zero elements."""
        _, calib = reference_data
        mat = matlab_base_arrays
        array_map = {
            "a": (calib.a, mat["a"]),
            "afd": (calib.afd, mat["afd"]),
            "alpha": (calib.alpha, mat["alpha"]),
            "beta": (calib.beta, mat["beta"]),
            "k_endow": (calib.k_endow, mat["KT"]),
            "l_endow": (calib.l_endow, mat["LT"]),
            "invforT": (calib.invforT, mat["invforT"]),
            "tax": (calib.tax, mat["tax"]),
            "ytot": (calib.ytot, mat["ytot"]),
        }

        for name, (py_arr, mat_arr) in array_map.items():
            non_zero = np.abs(mat_arr) > 0.0
            if np.any(non_zero):
                pointwise_rel = np.abs(py_arr[non_zero] - mat_arr[non_zero]) / np.abs(mat_arr[non_zero])
                max_pointwise_rel = float(np.max(pointwise_rel))
                assert max_pointwise_rel < 1e-12, (
                    f"Pointwise relative error for '{name}' {max_pointwise_rel:.4e} exceeds 1e-12 threshold."
                )

    def test_cobb_douglas_alpha_exact_identity(self, reference_data, matlab_base_arrays):
        """Verify capital share alpha is exactly bit-level identical to MATLAB."""
        _, calib = reference_data
        mat = matlab_base_arrays
        np.testing.assert_array_equal(calib.alpha, mat["alpha"])


# ---------------------------------------------------------------------------
# 2. Macro Balance Across all 847 Sector-Country Pairs
# ---------------------------------------------------------------------------

class TestMacroBalanceAcross847Pairs:
    """Verify Leontief market-clearing and cost accounting identities across all 847 pairs."""

    def test_market_clearing_supply_equals_demand_847(self, reference_data):
        """Assert supply equals intermediate plus final demand within 1e-12 relative discrepancy."""
        raw_data, calib = reference_data
        nc, ns, nfd = 77, 11, 3
        n_ind = ns * nc

        # 1. Gross output / supply for all 847 sector-country pairs
        y_supply = calib.ytot.flatten(order="F")  # shape (847,)

        # 2. Intermediate demand across all 847 purchasing industries
        interm_demand = np.sum(calib.a * calib.ytot, axis=(1, 2))  # shape (847,)

        # 3. Final demand across all categories and destination countries
        # In baseline calibration:
        xfd_4d = raw_data[:n_ind, n_ind:].reshape(nc, ns, nc, nfd).transpose(1, 0, 3, 2)
        sum_xc = np.sum(xfd_4d.transpose(1, 0, 2, 3).reshape(n_ind, nfd, nc), axis=0, keepdims=True)
        final_demand = np.sum(calib.afd * sum_xc, axis=(1, 2))  # shape (847,)

        total_demand = interm_demand + final_demand
        residual = y_supply - total_demand

        # Relative discrepancy per sector-country pair
        rel_residual = np.abs(residual) / y_supply

        max_rel = float(np.max(rel_residual))
        mean_rel = float(np.mean(rel_residual))

        assert max_rel < 1e-12, (
            f"Market-clearing discrepancy across 847 pairs: max relative error {max_rel:.4e} exceeds 1e-12."
        )
        assert mean_rel < 1e-14, (
            f"Mean market-clearing relative discrepancy {mean_rel:.4e} exceeds 1e-14."
        )

        # Zero violations of 1e-12 across all 847 pairs
        violators = np.where(rel_residual > 1e-12)[0]
        assert len(violators) == 0, f"Found {len(violators)} pairs with relative residual > 1e-12: {violators}"

    def test_cost_value_added_column_share_balance_847(self, reference_data):
        """Assert total outlay shares (intermediates + taxes + labor + capital) sum to 1.0."""
        raw_data, calib = reference_data
        nc, ns = 77, 11
        n_ind = ns * nc

        # Sum of technical intermediate coefficients for each column (j, k)
        interm_shares = np.sum(calib.a, axis=0)  # shape (11, 77)
        tax_shares = calib.tax[0]                # shape (11, 77)

        l_data = raw_data[n_ind + 1, :n_ind].reshape(nc, ns).T
        k_data = raw_data[n_ind + 2, :n_ind].reshape(nc, ns).T
        y_mat = calib.ytot[0]

        active = y_mat > 0
        l_shares = np.zeros_like(y_mat)
        k_shares = np.zeros_like(y_mat)
        l_shares[active] = l_data[active] / y_mat[active]
        k_shares[active] = k_data[active] / y_mat[active]

        total_cost_shares = interm_shares + tax_shares + l_shares + k_shares
        cost_diff = np.abs(total_cost_shares[active] - 1.0)

        max_cost_diff = float(np.max(cost_diff))
        assert max_cost_diff < 1e-14, (
            f"Cost share adding-up discrepancy {max_cost_diff:.4e} exceeds 1e-14 across 847 pairs."
        )

    def test_world_current_account_zero_balance(self, reference_data):
        """Assert world current account (sum of invforT) equals zero to machine precision."""
        _, calib = reference_data
        world_ca = float(np.sum(calib.invforT))
        world_gdp = float(np.sum(calib.ytot))
        rel_ca = abs(world_ca) / world_gdp

        assert rel_ca < 1e-12, (
            f"Global current account balance {world_ca:.4e} (relative {rel_ca:.4e}) exceeds 1e-12."
        )


# ---------------------------------------------------------------------------
# 3. Masked Division & Adversarial Stress Tests
# ---------------------------------------------------------------------------

class TestMaskedDivisionAdversarialStress:
    """Adversarial stress-testing of masked division, zeros, and boundary conditions."""

    def test_inactive_sectors_warning_free_under_werror(self):
        """Verify multiple inactive sectors across multiple countries emit zero RuntimeWarnings."""
        nc, ns, nfd = 4, 3, 2
        n_ind = ns * nc
        toy_data = np.ones((n_ind + 3, n_ind + nfd * nc), dtype=np.float64) * 5.0

        # Entire country 1 inactive
        c1_slice = slice(ns * 1, ns * 2)
        toy_data[:, c1_slice] = 0.0
        toy_data[c1_slice, :] = 0.0
        toy_data[n_ind:, c1_slice] = 0.0
        toy_data[:, n_ind + nfd * 1 : n_ind + nfd * 2] = 0.0
        toy_data[n_ind, n_ind + nfd * 1 : n_ind + nfd * 2] = 0.0

        # Sector 0 in country 3 inactive
        c3_s0 = 0 + ns * 3
        toy_data[:, c3_s0] = 0.0
        toy_data[c3_s0, :] = 0.0
        toy_data[n_ind:, c3_s0] = 0.0

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            calib = calibrate_trade_model(toy_data, ns=ns, nc=nc, nfd=nfd, validate=False)

        runtime_warnings = [w for w in record if issubclass(w.category, RuntimeWarning)]
        assert len(runtime_warnings) == 0, f"Expected 0 RuntimeWarnings, got: {runtime_warnings}"

        # Inactive elements must be zero, not NaN or Inf
        assert not np.isnan(calib.ytot).any()
        assert not np.isnan(calib.alpha).any()
        assert not np.isnan(calib.beta).any()
        assert not np.isnan(calib.a).any()
        assert not np.isnan(calib.afd).any()
        assert not np.isnan(calib.tax).any()
        assert not np.isinf(calib.beta).any()

        # Inactive country has zero output and parameters
        assert np.all(calib.ytot[0, :, 1] == 0.0)
        assert np.all(calib.alpha[0, :, 1] == 0.0)
        assert np.all(calib.beta[0, :, 1] == 0.0)

    def test_extreme_capital_labor_ratios(self):
        """Verify calibration handles extreme capital/labor ratios without NaN or overflow."""
        nc, ns, nfd = 2, 2, 1
        n_ind = ns * nc
        toy_data = np.ones((n_ind + 3, n_ind + nfd * nc), dtype=np.float64) * 10.0

        # High capital intensity at (s=0, c=0): K = 1e10, L = 1e-5
        toy_data[n_ind + 1, 0] = 1e-5
        toy_data[n_ind + 2, 0] = 1e10

        # High labor intensity at (s=1, c=0): L = 1e10, K = 1e-5
        toy_data[n_ind + 1, 1] = 1e10
        toy_data[n_ind + 2, 1] = 1e-5

        calib = calibrate_trade_model(toy_data, ns=ns, nc=nc, nfd=nfd, validate=False)
        assert not np.isnan(calib.alpha).any()
        assert not np.isnan(calib.beta).any()
        assert 0.999 < calib.alpha[0, 0, 0] <= 1.0
        assert 0.0 <= calib.alpha[0, 1, 0] < 0.001

    def test_negative_investment_drawdown_exact_locations(self, reference_data, matlab_base_arrays):
        """Verify negative afd entries occur strictly in Category 1 for LTU, UKR, VNM."""
        _, calib = reference_data
        mat = matlab_base_arrays
        neg_mask = calib.afd < 0.0

        # Exactly 3 negative entries in the 847 x 3 x 77 tensor
        assert np.sum(neg_mask) == 3

        # Must occur strictly in Category 1 (Investment)
        categories = np.where(neg_mask)[1]
        assert np.all(categories == 1)

        # Categories 0 and 2 must be strictly non-negative
        assert np.all(calib.afd[:, 0, :] >= 0.0)
        assert np.all(calib.afd[:, 2, :] >= 0.0)

        # Bit-level parity with MATLAB benchmark
        np.testing.assert_allclose(calib.afd[neg_mask], mat["afd"][neg_mask], rtol=1e-14, atol=1e-15)

    def test_validation_catches_adversarial_corruptions(self, reference_data):
        """Assert TradeCalibrationResult.validate() actively rejects corrupted parameter states."""
        _, calib = reference_data

        # 1. Negative entry in Consumption (Category 0)
        bad_afd_c0 = calib.afd.copy()
        bad_afd_c0[0, 0, 0] = -1e-4
        assert not replace(calib, afd=bad_afd_c0).validate()["nonneg_afd"]

        # 2. Negative entry in Direct Purchases (Category 2)
        bad_afd_c2 = calib.afd.copy()
        bad_afd_c2[0, 2, 0] = -1e-4
        assert not replace(calib, afd=bad_afd_c2).validate()["nonneg_afd"]

        # 3. Category 1 sum violation
        bad_afd_sum = calib.afd.copy()
        bad_afd_sum[0, 1, 0] += 0.1
        assert not replace(calib, afd=bad_afd_sum).validate()["nonneg_afd"]

        # 4. Alpha out of bounds
        bad_alpha = calib.alpha.copy()
        bad_alpha[0, 0, 0] = 1.05
        assert not replace(calib, alpha=bad_alpha).validate()["alpha_bounds"]

        # 5. Global transfer imbalance
        bad_inv = calib.invforT.copy()
        bad_inv[0, 0] += 1000.0
        assert not replace(calib, invforT=bad_inv).validate()["global_transfer_balance"]
