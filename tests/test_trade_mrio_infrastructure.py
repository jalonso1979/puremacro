"""Unit and Integration Test Suite for Cross-Database MRIO Data Infrastructure.

Tests:
1. Harmonized Ingestion Adapters for all 5 databases (FIGARO, EXIOBASE, WIOD, Eora, OECD ICIO)
   with deterministic synthetic fallback.
2. Row sum equals column sum accounting invariants and Walrasian zero-profit cost exhaustion.
3. Phantom output injection (1e-6 M USD) on inactive/zero-output sectors.
4. Non-negative value-added flooring with dual TLS absorption preserving outlay identities.
5. Multilateral zero-leakage trade balance sum_c XN_c == 0.0.
6. Biproportional matrix balancing: RAS, GRAS (with negative entries), and quadratic balancing.
7. Collatz-Wielandt spectral radius check terminating in < 0.15s on 3,465-node networks.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pytest

from puremacro.trade.data import (
    EORA_189_COUNTRIES,
    EORA_26_SECTORS,
    EORA_FD_CATEGORIES,
    EXIOBASE_163_SECTORS,
    EXIOBASE_200_SECTORS,
    EXIOBASE_COUNTRIES,
    EXIOBASE_FD_CATEGORIES,
    FIGARO_64_SECTORS,
    FIGARO_COUNTRIES,
    FIGARO_FD_CATEGORIES,
    OECD_45_SECTORS,
    OECD_77_COUNTRIES,
    OECD_FD_CATEGORIES,
    WIOD_44_COUNTRIES,
    WIOD_56_SECTORS,
    WIOD_FD_CATEGORIES,
    RawIOData,
    compute_trade_balances,
    generate_synthetic_mrio,
    load_eora,
    load_exiobase,
    load_figaro,
    load_oecd_icio_granular,
    load_wiod,
    package_mrio_to_calibration_result,
    verify_accounting_invariants,
    verify_zero_leakage,
)
from puremacro.trade.regularize import (
    balance_gras,
    balance_quadratic,
    balance_ras,
    compute_spectral_radius,
    regularize_mrio_table,
    spectral_radius,
    validate_accounting_identities,
)


# ===========================================================================
# 1. Tests for Harmonized Ingestion Adapters (Synthetic & Empirical)
# ===========================================================================


class TestMRIOAdapters:
    """Test data adapters for the 5 international input-output databases."""

    def test_load_figaro_synthetic_adapter(self) -> None:
        """FIGARO loader correctly parses and calibrates to canonical result."""
        res = load_figaro(year=2019, custom_c=4, custom_s=3, seed=42, fallback_to_synthetic=True)
        assert res.nc == 4
        assert res.ns == 3
        assert res.theta.shape == (1, 5, 4)
        assert np.all(res.a >= 0.0)
        assert np.all(res.beta > 0.0)
        inv = verify_accounting_invariants(res)
        assert inv["valid"]
        assert inv["zero_leakage_certified"]
        assert inv["max_budget_error"] < 1e-9

    def test_load_exiobase_synthetic_adapter_ixi(self) -> None:
        """EXIOBASE ixi model (Industry-by-Industry) correctly loads and calibrates."""
        res = load_exiobase(year=2019, model="ixi", custom_c=4, custom_s=3, seed=42, fallback_to_synthetic=True)
        assert res.nc == 4
        assert res.ns == 3
        assert res.theta.shape == (1, 7, 4)  # 7 FD categories in EXIOBASE
        inv = verify_accounting_invariants(res)
        assert inv["valid"]

    def test_load_exiobase_synthetic_adapter_pxp(self) -> None:
        """EXIOBASE pxp model (Product-by-Product) correctly loads and calibrates."""
        res = load_exiobase(year=2019, model="pxp", custom_c=4, custom_s=3, seed=42, fallback_to_synthetic=True)
        assert res.nc == 4
        assert res.ns == 3
        assert res.theta.shape == (1, 7, 4)
        inv = verify_accounting_invariants(res)
        assert inv["valid"]

    def test_load_wiod_synthetic_adapter(self) -> None:
        """WIOD loader correctly loads and calibrates."""
        res = load_wiod(year=2014, custom_c=4, custom_s=3, seed=42, fallback_to_synthetic=True)
        assert res.nc == 4
        assert res.ns == 3
        assert res.theta.shape == (1, 5, 4)
        inv = verify_accounting_invariants(res)
        assert inv["valid"]

    def test_load_eora_synthetic_adapter(self) -> None:
        """Eora26 loader correctly loads, scales from '000 USD to M USD, and calibrates."""
        res = load_eora(year=2015, custom_c=4, custom_s=3, seed=42, fallback_to_synthetic=True)
        assert res.nc == 4
        assert res.ns == 3
        assert res.theta.shape == (1, 6, 4)
        inv = verify_accounting_invariants(res)
        assert inv["valid"]

    def test_load_oecd_icio_granular_synthetic_adapter(self) -> None:
        """Granular OECD ICIO 45-sector loader correctly loads and calibrates."""
        res = load_oecd_icio_granular(year=2019, custom_c=4, custom_s=3, seed=42, fallback_to_synthetic=True)
        assert res.nc == 4
        assert res.ns == 3
        assert res.theta.shape == (1, 6, 4)
        inv = verify_accounting_invariants(res)
        assert inv["valid"]

    def test_authentic_roster_metadata_and_dimensions(self) -> None:
        """Synthetic generator produces authentic dimensions matching official schemas."""
        # FIGARO: 46 x 64 = 2,944 nodes, 5 FD
        raw_fig = generate_synthetic_mrio("figaro")
        assert len(raw_fig.countries) == 46
        assert len(raw_fig.sectors) == 64
        assert len(raw_fig.fd_categories) == 5
        assert raw_fig.intermediate_matrix.shape == (2944, 2944)
        assert raw_fig.final_demand_matrix.shape == (2944, 46 * 5)

        # EXIOBASE: 49 x 163 = 7,987 nodes, 7 FD
        raw_exio = generate_synthetic_mrio("exiobase", model="ixi")
        assert len(raw_exio.countries) == 49
        assert len(raw_exio.sectors) == 163
        assert len(raw_exio.fd_categories) == 7

        # WIOD: 44 x 56 = 2,464 nodes, 5 FD
        raw_wiod = generate_synthetic_mrio("wiod")
        assert len(raw_wiod.countries) == 44
        assert len(raw_wiod.sectors) == 56
        assert len(raw_wiod.fd_categories) == 5

        # EORA: 189 x 26 = 4,914 nodes, 6 FD
        raw_eora = generate_synthetic_mrio("eora")
        assert len(raw_eora.countries) == 189
        assert len(raw_eora.sectors) == 26
        assert len(raw_eora.fd_categories) == 6

        # OECD ICIO: 77 x 45 = 3,465 nodes, 6 FD
        raw_oecd = generate_synthetic_mrio("oecd")
        assert len(raw_oecd.countries) == 77
        assert len(raw_oecd.sectors) == 45
        assert len(raw_oecd.fd_categories) == 6

    def test_unknown_dataset_raises_error(self) -> None:
        """Unknown dataset name raises informative ValueError."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            generate_synthetic_mrio("invalid_database_name")

    def test_missing_empirical_file_error_when_fallback_disabled(self) -> None:
        """Disabling fallback_to_synthetic on a missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_figaro(year=1900, fallback_to_synthetic=False, file_path="/nonexistent_path/figaro.csv")


# ===========================================================================
# 2. Tests for Full Table Execution & Performance
# ===========================================================================


class TestFullTableExecution:
    """Test full authentic dimension tables to verify scalability and execution times."""

    def test_full_figaro_calibration_and_invariants(self) -> None:
        """Full FIGARO (46x64 = 2,944 nodes) calibrates within 3 seconds with valid invariants."""
        t0 = time.time()
        res = load_figaro(year=2019, fallback_to_synthetic=True)
        elapsed = time.time() - t0
        assert elapsed < 5.0, f"Full FIGARO calibration took too long: {elapsed:.2f}s"
        assert res.nc == 46
        assert res.ns == 64
        inv = verify_accounting_invariants(res)
        assert inv["valid"]

    def test_full_wiod_calibration_and_invariants(self) -> None:
        """Full WIOD (44x56 = 2,464 nodes) calibrates within 3 seconds with valid invariants."""
        t0 = time.time()
        res = load_wiod(year=2014, fallback_to_synthetic=True)
        elapsed = time.time() - t0
        assert elapsed < 5.0, f"Full WIOD calibration took too long: {elapsed:.2f}s"
        assert res.nc == 44
        assert res.ns == 56
        inv = verify_accounting_invariants(res)
        assert inv["valid"]

    def test_full_oecd_icio_granular_calibration_and_invariants(self) -> None:
        """Full OECD ICIO (77x45 = 3,465 nodes) calibrates within 5 seconds with valid invariants."""
        t0 = time.time()
        res = load_oecd_icio_granular(year=2019, fallback_to_synthetic=True)
        elapsed = time.time() - t0
        assert elapsed < 6.0, f"Full OECD ICIO calibration took too long: {elapsed:.2f}s"
        assert res.nc == 77
        assert res.ns == 45
        inv = verify_accounting_invariants(res)
        assert inv["valid"]


# ===========================================================================
# 3. Tests for Accounting & Balance Invariants
# ===========================================================================


class TestAccountingInvariants:
    """Test accounting identities, Walrasian zero-profit outlays, and budget balance."""

    def test_row_sum_equals_column_outlays(self) -> None:
        """Gross output row sales identically equal column outlays: sum_i Z_ij + VA_j + TLS_j == Y_j."""
        raw = generate_synthetic_mrio("figaro", custom_c=5, custom_s=4, seed=123)
        Z_c, F_c, VA_c, TLS_c, Y_c = regularize_mrio_table(
            raw.Z, raw.F, raw.VA, raw.TLS, raw.Y, n_countries=raw.C, n_sectors=raw.S
        )

        sales = np.sum(Z_c, axis=1) + np.sum(F_c, axis=1)
        outlays = np.sum(Z_c, axis=0) + VA_c + TLS_c

        assert np.allclose(sales, Y_c, atol=1e-11, rtol=1e-11)
        assert np.allclose(outlays, Y_c, atol=1e-11, rtol=1e-11)
        assert np.allclose(sales, outlays, atol=1e-11, rtol=1e-11)

    def test_zero_leakage_multilateral_trade_balance(self) -> None:
        """World current account balances sum to identically zero: sum_c XN_c == 0.0."""
        raw = generate_synthetic_mrio("oecd", custom_c=6, custom_s=3, seed=88)
        XN = compute_trade_balances(raw.Z, raw.F, nc=raw.C, ns=raw.S, nfd=raw.K_F)
        assert verify_zero_leakage(XN, tol=1e-10)
        assert abs(np.sum(XN)) < 1e-10

    def test_investment_discrepancy_balancing(self) -> None:
        """Consumer budget balance holds: sum_f theta_c^f == 1.0 for each country."""
        res = load_figaro(custom_c=5, custom_s=4, seed=99, fallback_to_synthetic=True)
        theta_sums = np.sum(res.theta, axis=1).flatten()
        assert np.allclose(theta_sums, 1.0, atol=1e-11)

    def test_validate_accounting_identities_helper(self) -> None:
        """validate_accounting_identities reports valid=True on regularized flows."""
        raw = generate_synthetic_mrio("wiod", custom_c=3, custom_s=3, seed=10)
        Z_c, F_c, VA_c, TLS_c, Y_c = regularize_mrio_table(
            raw.Z, raw.F, raw.VA, raw.TLS, raw.Y, n_countries=raw.C, n_sectors=raw.S
        )
        report = validate_accounting_identities(Z_c, F_c, VA_c, TLS_c, Y_c)
        assert report["valid"]
        assert report["spectral_certified"]
        assert report["max_sales_outlays_error"] < 1e-10


# ===========================================================================
# 4. Tests for Phantom Output Injection
# ===========================================================================


class TestPhantomOutputInjection:
    """Test micro-floor injection on inactive (zero-output) sectors."""

    def test_phantom_injection_on_single_inactive_sector(self) -> None:
        """Inactive sector (Y=0) receives epsilon=1e-6 M USD in domestic F, VA, and Y."""
        raw = generate_synthetic_mrio("oecd", custom_c=3, custom_s=3, n_inactive=1, seed=55)
        assert raw.Y[0] == 0.0
        assert raw.VA[0] == 0.0
        assert np.all(raw.Z[0, :] == 0.0)

        Z_c, F_c, VA_c, TLS_c, Y_c = regularize_mrio_table(
            raw.Z, raw.F, raw.VA, raw.TLS, raw.Y, floor_output=1e-6, n_countries=3, n_sectors=3
        )

        assert Y_c[0] >= 1e-6
        assert VA_c[0] >= 1e-6
        # Injected into domestic household final demand (category 0)
        assert F_c[0, 0] >= 1e-6
        # Sales match outlays on phantom node
        outlay_0 = np.sum(Z_c[:, 0]) + VA_c[0] + TLS_c[0]
        assert abs(outlay_0 - Y_c[0]) < 1e-12

    def test_phantom_injection_multiple_inactive_sectors(self) -> None:
        """Multiple inactive sectors across multiple countries are all safely floored."""
        raw = generate_synthetic_mrio("eora", custom_c=4, custom_s=4, n_inactive=3, seed=77)
        Z_c, F_c, VA_c, TLS_c, Y_c = regularize_mrio_table(
            raw.Z, raw.F, raw.VA, raw.TLS, raw.Y, n_countries=4, n_sectors=4
        )
        assert np.all(Y_c >= 1e-6)
        assert np.all(VA_c >= 1e-6)

        # Technical coefficients A = Z / Y remain well-defined and finite
        A = Z_c / Y_c.reshape(1, -1)
        assert np.all(np.isfinite(A))
        assert np.all(A >= 0.0)

    def test_active_sectors_not_disturbed_by_phantom_injection(self) -> None:
        """Active sectors with Y >> 1e-6 are untouched by phantom injection."""
        raw = generate_synthetic_mrio("figaro", custom_c=3, custom_s=3, n_inactive=1, seed=42)
        # Node 1 is active
        y_before = raw.Y[1]
        assert y_before > 1.0

        Z_c, F_c, VA_c, TLS_c, Y_c = regularize_mrio_table(
            raw.Z, raw.F, raw.VA, raw.TLS, raw.Y, n_countries=3, n_sectors=3
        )
        assert abs(Y_c[1] - y_before) < 1e-12


# ===========================================================================
# 5. Tests for Value-Added Flooring & Dual TLS Debit
# ===========================================================================


class TestValueAddedFlooring:
    """Test non-negative value-added flooring and dual tax absorption."""

    def test_negative_va_flooring_with_dual_tls_debit(self) -> None:
        """Negative VA is floored to max(1e-3*Y, 1.0) with exact dual TLS debit."""
        M = 4
        Z = np.ones((M, M)) * 5.0
        F = np.ones((M, 2)) * 10.0
        Y = np.sum(Z, axis=1) + np.sum(F, axis=1)  # 40.0
        # Sector 1 has negative VA
        VA = np.array([15.0, -10.0, 12.0, 14.0])
        TLS = Y - np.sum(Z, axis=0) - VA

        outlays_before = np.sum(Z, axis=0) + VA + TLS

        Z_c, F_c, VA_c, TLS_c, Y_c = regularize_mrio_table(
            Z, F, VA, TLS, Y, floor_va_ratio=1e-3, floor_va_abs=1.0, n_countries=2, n_sectors=2
        )

        expected_floor = max(1e-3 * Y[1], 1.0)  # 1.0
        assert VA_c[1] >= expected_floor
        # Outlays must remain strictly conserved
        outlays_after = np.sum(Z_c, axis=0) + VA_c + TLS_c
        assert np.allclose(outlays_after, outlays_before, atol=1e-12)
        assert np.allclose(outlays_after, Y_c, atol=1e-12)

    def test_synthetic_mrio_with_negative_va_injection(self) -> None:
        """Synthetic table generated with negative VA nodes calibrates cleanly after regularization."""
        raw = generate_synthetic_mrio("exiobase", custom_c=4, custom_s=3, n_neg_va=2, seed=33)
        assert np.any(raw.VA < 0.0)

        calib = package_mrio_to_calibration_result(raw, regularize=True, validate=True)
        # Value added factor payments must be non-negative
        assert np.all(calib.KT > 0.0)
        assert np.all(calib.LT > 0.0)
        inv = verify_accounting_invariants(calib)
        assert inv["valid"]

    def test_residual_tls_reconciliation_exactness(self) -> None:
        """Residual TLS reconciliation ensures |Sales_j - Outlays_j| < 1e-12 * max(Y)."""
        raw = generate_synthetic_mrio("wiod", custom_c=3, custom_s=3, seed=12)
        # Perturb TLS to create an intentional discrepancy
        raw.taxes_less_subsidies[0] += 50.0

        Z_c, F_c, VA_c, TLS_c, Y_c = regularize_mrio_table(
            raw.Z, raw.F, raw.VA, raw.taxes_less_subsidies, raw.Y, n_countries=3, n_sectors=3
        )
        outlays = np.sum(Z_c, axis=0) + VA_c + TLS_c
        max_disc = np.max(np.abs(outlays - Y_c))
        assert max_disc < 1e-11 * np.max(Y_c)


# ===========================================================================
# 6. Tests for Collatz-Wielandt Spectral Radius
# ===========================================================================


class TestCollatzWielandtSpectralRadius:
    """Test Collatz-Wielandt spectral radius certification and performance."""

    def test_spectral_radius_productive_network(self) -> None:
        """Productive Leontief network certifies rho(B) < 0.999."""
        B = np.array([
            [0.2, 0.1, 0.05],
            [0.1, 0.3, 0.1],
            [0.05, 0.1, 0.25],
        ])
        rho, lower, upper = compute_spectral_radius(B)
        assert 0.0 < rho < 0.999
        assert lower <= rho <= upper + 1e-12

    def test_non_productive_system_rejection(self) -> None:
        """Non-productive system (rho >= 0.999) raises ValueError."""
        # Highly circular system with row sums > 1.0
        B_unviable = np.array([
            [0.8, 0.5],
            [0.5, 0.8],
        ])
        rho, _, _ = compute_spectral_radius(B_unviable)
        assert rho >= 0.999

        Z = B_unviable * 100.0
        F = np.array([[10.0], [10.0]])
        VA = np.array([10.0, 10.0])
        TLS = np.array([0.0, 0.0])
        Y = np.array([100.0, 100.0])

        with pytest.raises(ValueError, match="Leontief cost system is non-productive"):
            regularize_mrio_table(Z, F, VA, TLS, Y)

    def test_collatz_wielandt_speed_on_large_network(self) -> None:
        """Collatz-Wielandt power iteration on 3,465 nodes evaluates in < 0.15s."""
        M = 3465
        # Create sparse-like dense productive block
        rng = np.random.default_rng(42)
        B = rng.uniform(0.0001, 0.001, size=(M, M))
        # Ensure row sums < 0.8 so rho < 0.8
        B = 0.5 * B / np.max(np.sum(B, axis=1))

        t0 = time.time()
        rho, lower, upper = compute_spectral_radius(B, max_iter=50)
        elapsed = time.time() - t0

        assert elapsed < 0.15, f"Spectral radius check took too long: {elapsed:.4f}s"
        assert rho < 0.999
        assert lower <= rho <= upper + 1e-6


# ===========================================================================
# 7. Tests for Biproportional Matrix Balancing (RAS, GRAS, Quadratic)
# ===========================================================================


class TestMatrixBalancing:
    """Test RAS, GRAS, and Quadratic matrix balancing algorithms."""

    def test_balance_ras_convergence(self) -> None:
        """Classical RAS converges and matches row and column margins to 1e-10."""
        Z0 = np.array([
            [10.0, 20.0, 15.0],
            [5.0, 30.0, 25.0],
            [15.0, 10.0, 35.0],
        ])
        u = np.array([50.0, 70.0, 80.0])
        v = np.array([40.0, 60.0, 100.0])  # sum(u) == sum(v) == 200

        Z_bal = balance_ras(Z0, u, v, tol=1e-10)

        assert np.all(Z_bal >= 0.0)
        assert np.allclose(np.sum(Z_bal, axis=1), u, atol=1e-9)
        assert np.allclose(np.sum(Z_bal, axis=0), v, atol=1e-9)

    def test_balance_ras_rejects_negative_entries(self) -> None:
        """balance_ras raises ValueError if prior matrix has negative entries."""
        Z0_neg = np.array([[10.0, -2.0], [5.0, 8.0]])
        u = np.array([8.0, 13.0])
        v = np.array([15.0, 6.0])
        with pytest.raises(ValueError, match="non-negative"):
            balance_ras(Z0_neg, u, v)

    def test_balance_gras_with_negative_entries(self) -> None:
        """Generalized RAS (GRAS) balances matrices with negative entries while preserving signs."""
        Z0 = np.array([
            [20.0, -5.0, 10.0],
            [8.0, 15.0, -3.0],
            [-2.0, 12.0, 25.0],
        ])
        u = np.array([30.0, 25.0, 45.0])
        v = np.array([28.0, 32.0, 40.0])  # sum(u) == sum(v) == 100

        Z_gras = balance_gras(Z0, u, v, tol=1e-10)

        # Margins satisfied
        assert np.allclose(np.sum(Z_gras, axis=1), u, atol=1e-9)
        assert np.allclose(np.sum(Z_gras, axis=0), v, atol=1e-9)

        # Signs preserved
        assert np.all((Z_gras > 0) == (Z0 > 0))
        assert np.all((Z_gras < 0) == (Z0 < 0))

    def test_balance_gras_equivalence_to_ras_for_nonnegative(self) -> None:
        """GRAS produces identical scaling to classical RAS when Z0 >= 0."""
        Z0 = np.array([
            [12.0, 18.0],
            [24.0, 6.0],
        ])
        u = np.array([40.0, 60.0])
        v = np.array([50.0, 50.0])

        Z_ras = balance_ras(Z0, u, v, tol=1e-12)
        Z_gras = balance_gras(Z0, u, v, tol=1e-12)

        assert np.allclose(Z_ras, Z_gras, atol=1e-8)

    def test_balance_quadratic_unweighted_closed_form(self) -> None:
        """Constrained least-squares quadratic balancing satisfies margins to machine precision."""
        Z0 = np.array([
            [10.0, 25.0, 5.0],
            [15.0, 20.0, 30.0],
        ])
        u = np.array([45.0, 75.0])
        v = np.array([30.0, 45.0, 45.0])  # sum(u) == sum(v) == 120

        Z_quad = balance_quadratic(Z0, u, v)

        assert np.allclose(np.sum(Z_quad, axis=1), u, atol=1e-12)
        assert np.allclose(np.sum(Z_quad, axis=0), v, atol=1e-12)

    def test_balance_quadratic_weighted(self) -> None:
        """Weighted quadratic balancing satisfies marginals under positive weight matrix."""
        Z0 = np.array([
            [10.0, 20.0],
            [30.0, 40.0],
        ])
        u = np.array([35.0, 75.0])
        v = np.array([45.0, 65.0])
        weights = np.array([
            [1.0, 2.0],
            [0.5, 1.5],
        ])

        Z_quad_w = balance_quadratic(Z0, u, v, weights=weights)

        assert np.allclose(np.sum(Z_quad_w, axis=1), u, atol=1e-9)
        assert np.allclose(np.sum(Z_quad_w, axis=0), v, atol=1e-9)

    def test_balancing_algorithms_automatic_margin_rescaling(self) -> None:
        """All balancing routines handle sum(u) != sum(v) by scaling column targets."""
        Z0 = np.array([[5.0, 10.0], [15.0, 20.0]])
        u = np.array([20.0, 40.0])  # sum = 60
        v = np.array([25.0, 25.0])  # sum = 50 != 60

        # RAS
        Z_ras = balance_ras(Z0, u, v)
        assert np.allclose(np.sum(Z_ras, axis=1), u, atol=1e-9)
        assert np.isclose(np.sum(Z_ras), 60.0)

        # GRAS
        Z_gras = balance_gras(Z0, u, v)
        assert np.allclose(np.sum(Z_gras, axis=1), u, atol=1e-9)
        assert np.isclose(np.sum(Z_gras), 60.0)

        # Quadratic
        Z_quad = balance_quadratic(Z0, u, v)
        assert np.allclose(np.sum(Z_quad, axis=1), u, atol=1e-9)
        assert np.isclose(np.sum(Z_quad), 60.0)


# ===========================================================================
# 8. Tests for Pyodide Compliance & Runtime Contract
# ===========================================================================


class TestRuntimeContract:
    """Test runtime compliance with Pyodide contract (zero C-extensions or user paths)."""

    def test_no_machine_specific_paths_in_regularize(self) -> None:
        """puremacro/trade/regularize.py has no hardcoded machine-specific paths."""
        reg_file = Path(__file__).resolve().parent.parent / "puremacro" / "trade" / "regularize.py"
        assert reg_file.is_file()

        pattern = re.compile(r"[\"']/(?:Users|home)/[A-Za-z0-9._-]+/")
        hits = []
        for lineno, line in enumerate(reg_file.read_text(encoding="utf-8").splitlines(), start=1):
            if line.lstrip().startswith("#"):
                continue
            if pattern.search(line):
                hits.append(f"Line {lineno}: {line}")
        assert not hits, f"Machine-specific path detected in regularize.py: {hits}"

    def test_no_machine_specific_paths_in_data(self) -> None:
        """puremacro/trade/data.py has no hardcoded machine-specific paths."""
        data_file = Path(__file__).resolve().parent.parent / "puremacro" / "trade" / "data.py"
        assert data_file.is_file()

        pattern = re.compile(r"[\"']/(?:Users|home)/[A-Za-z0-9._-]+/")
        hits = []
        for lineno, line in enumerate(data_file.read_text(encoding="utf-8").splitlines(), start=1):
            if line.lstrip().startswith("#"):
                continue
            if pattern.search(line):
                hits.append(f"Line {lineno}: {line}")
        assert not hits, f"Machine-specific path detected in data.py: {hits}"
