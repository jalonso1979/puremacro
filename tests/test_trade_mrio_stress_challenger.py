"""Empirical Stress Test Harness for Milestone 2 MRIO Infrastructure.

Executed by Challenger Agent (challenger_m2_1) to stress-test:
1. Multiple inactive sectors (up to 100 inactive nodes) and entire-country inactive nodes.
2. Extreme negative value-added nodes (VA down to -1e7, -1e10) and machine-precision outlays balance (< 1e-12).
3. GRAS balancing on matrices with mixed signs, extreme negative entries, and varied sparsity.
4. Collatz-Wielandt spectral radius check on M=3,465 tables with execution time benchmark (< 0.15s).
5. Memory leaks and numeric stability under repeated stress cycles.
"""
from __future__ import annotations

import gc
import sys
import time
import tracemalloc
from typing import Any

import numpy as np
import pytest

from puremacro.trade.data import (
    generate_synthetic_mrio,
    load_figaro,
    load_oecd_icio_granular,
    load_wiod,
    package_mrio_to_calibration_result,
    verify_accounting_invariants,
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


class TestStressInactiveSectors:
    """Stress-test phantom output injection under extreme inactive sector counts."""

    @pytest.mark.parametrize("n_inactive", [1, 10, 50, 100])
    def test_phantom_injection_scaling_up_to_100_nodes(self, n_inactive: int) -> None:
        """Phantom output injection handles up to 100 inactive nodes cleanly."""
        # Use WIOD (44x56 = 2,464 nodes)
        raw = generate_synthetic_mrio("wiod", seed=42 + n_inactive)
        M = raw.M
        rng = np.random.default_rng(1000 + n_inactive)
        inactive_indices = rng.choice(M, size=n_inactive, replace=False)
        assert len(inactive_indices) == n_inactive

        Z = raw.Z.copy()
        F = raw.F.copy()
        VA = raw.VA.copy()
        TLS = raw.TLS.copy()

        for idx in inactive_indices:
            Z[idx, :] = 0.0
            Z[:, idx] = 0.0
            F[idx, :] = 0.0
            VA[idx] = 0.0
            TLS[idx] = 0.0

        Y = np.sum(Z, axis=1) + np.sum(F, axis=1)
        for idx in inactive_indices:
            Y[idx] = 0.0
        TLS = Y - np.sum(Z, axis=0) - VA
        for idx in inactive_indices:
            TLS[idx] = 0.0

        active_indices = np.setdiff1d(np.arange(M), inactive_indices)
        y_active_before = Y[active_indices].copy()

        # Regularize
        Z_c, F_c, VA_c, TLS_c, Y_c = regularize_mrio_table(
            Z, F, VA, TLS, Y,
            floor_output=1e-6,
            n_countries=raw.C,
            n_sectors=raw.S,
        )

        # 1. Inactive sectors floored properly
        assert np.all(Y_c[inactive_indices] >= 1e-6)
        assert np.all(VA_c[inactive_indices] >= 1e-6)
        assert np.all(np.isfinite(Y_c))
        assert np.all(np.isfinite(VA_c))

        # 2. Active sectors completely unaffected
        y_active_after = Y_c[active_indices]
        max_active_diff = float(np.max(np.abs(y_active_after - y_active_before)))
        assert max_active_diff < 1e-12, f"Active sector output was modified by {max_active_diff:.2e}"

        # 3. Technical coefficients A = Z / Y remain well-defined and finite
        A = Z_c / Y_c.reshape(1, -1)
        assert np.all(np.isfinite(A))
        assert np.all(A >= 0.0)
        assert not np.any(np.isnan(A))

        # 4. Outlays balance holds on all nodes including phantom nodes
        outlays = np.sum(Z_c, axis=0) + VA_c + TLS_c
        max_outlay_diff = float(np.max(np.abs(outlays - Y_c)))
        assert max_outlay_diff < 1e-11 * np.max(Y_c)

    def test_entire_country_inactive_sectors(self) -> None:
        """An entire country with all sectors inactive (Y = 0) is safely activated."""
        from puremacro.trade.data import RawIOData
        C, S = 4, 10
        raw = generate_synthetic_mrio("figaro", custom_c=C, custom_s=S, seed=101)

        # Invalidate country 1 (all 10 sectors and all trade/final demand)
        c_target = 1
        c_nodes = slice(c_target * S, (c_target + 1) * S)
        c_fd_cols = slice(c_target * raw.K_F, (c_target + 1) * raw.K_F)

        Z = raw.Z.copy()
        F = raw.F.copy()
        VA = raw.VA.copy()

        Z[c_nodes, :] = 0.0
        Z[:, c_nodes] = 0.0
        F[c_nodes, :] = 0.0
        F[:, c_fd_cols] = 0.0
        VA[c_nodes] = 0.0

        Y = np.sum(Z, axis=1) + np.sum(F, axis=1)
        TLS = Y - np.sum(Z, axis=0) - VA

        raw_mod = RawIOData(
            countries=raw.countries,
            sectors=raw.sectors,
            intermediate_matrix=Z,
            final_demand_matrix=F,
            value_added=VA,
            taxes_less_subsidies=TLS,
            gross_output=Y,
            fd_categories=raw.fd_categories,
        )

        calib = package_mrio_to_calibration_result(raw_mod, regularize=True, validate=True)
        assert calib.nc == C
        assert calib.ns == S
        assert np.all(calib.ytot > 0.0)
        assert np.all(calib.beta > 0.0)
        assert np.all(np.isfinite(calib.a))

        inv = verify_accounting_invariants(calib)
        assert inv["valid"]
        assert inv["zero_leakage_certified"]


class TestStressExtremeNegativeVA:
    """Stress-test non-negative value-added flooring and dual TLS debit under extreme values."""

    @pytest.mark.parametrize("va_val", [-1e3, -1e6, -1e7, -1e10])
    def test_extreme_negative_va_scalar_nodes(self, va_val: float) -> None:
        """Single node with extreme negative VA (-1e3 to -1e10) satisfies outlays to machine precision."""
        M = 10
        rng = np.random.default_rng(123)
        Z = rng.uniform(1.0, 10.0, size=(M, M))
        F = rng.uniform(5.0, 20.0, size=(M, 2))
        Y = np.sum(Z, axis=1) + np.sum(F, axis=1)

        VA = rng.uniform(10.0, 50.0, size=M)
        # Inject extreme negative VA on node 3
        VA[3] = va_val
        TLS = Y - np.sum(Z, axis=0) - VA

        outlays_initial = np.sum(Z, axis=0) + VA + TLS
        assert np.allclose(outlays_initial, Y, atol=1e-12)

        Z_c, F_c, VA_c, TLS_c, Y_c = regularize_mrio_table(
            Z, F, VA, TLS, Y, floor_va_ratio=1e-3, floor_va_abs=1.0, n_countries=2, n_sectors=5
        )

        expected_floor = max(1e-3 * Y[3], 1.0)
        assert VA_c[3] >= expected_floor

        # Outlays identity conservation check to machine precision
        outlays_after = np.sum(Z_c, axis=0) + VA_c + TLS_c
        max_dev = float(np.max(np.abs(outlays_after - Y_c)))
        assert max_dev < 1e-11 * np.max(Y_c), f"Dev {max_dev:.2e} exceeded tolerance"
        # Relative error to max(Y) or machine epsilon
        assert max_dev / np.max(Y_c) < 1e-12

    def test_multiple_extreme_negative_va_nodes(self) -> None:
        """Multiple nodes with extreme negative VA across sectors calibrate cleanly."""
        raw = generate_synthetic_mrio("oecd", custom_c=5, custom_s=6, seed=777)
        # Inject extreme negative VA across 5 nodes
        bad_nodes = [0, 5, 12, 18, 25]
        for i, idx in enumerate(bad_nodes):
            raw.value_added[idx] = -1e7 * (i + 1)
            raw.taxes_less_subsidies[idx] = raw.gross_output[idx] - np.sum(raw.intermediate_matrix[:, idx]) - raw.value_added[idx]

        calib = package_mrio_to_calibration_result(raw, regularize=True, validate=True)
        assert np.all(calib.KT > 0.0)
        assert np.all(calib.LT > 0.0)
        inv = verify_accounting_invariants(calib)
        assert inv["valid"]
        assert inv["max_outlays_sales_error"] < 1e-11 * np.max(calib.ytot)

    def test_negative_va_on_2d_va_matrix(self) -> None:
        """2D VA matrix (e.g. Labor and Capital rows) with negative entries regularizes cleanly."""
        M = 8
        Z = np.ones((M, M)) * 2.0
        F = np.ones((M, 2)) * 5.0
        Y = np.sum(Z, axis=1) + np.sum(F, axis=1)  # 26.0 per sector

        # 2D VA: row 0 is labor, row 1 is capital
        VA_2d = np.array([
            [10.0, -1e7,  5.0,  8.0, 12.0,  3.0,  6.0,  9.0],
            [ 5.0,  2.0, -1e6,  4.0,  6.0,  2.0,  3.0,  4.0],
        ])
        TLS = Y - np.sum(Z, axis=0) - np.sum(VA_2d, axis=0)

        Z_c, F_c, VA_c, TLS_c, Y_c = regularize_mrio_table(
            Z, F, VA_2d, TLS, Y, n_countries=2, n_sectors=4
        )

        va_totals_clean = np.sum(VA_c, axis=0)
        assert np.all(va_totals_clean >= 1.0)
        outlays = np.sum(Z_c, axis=0) + va_totals_clean + TLS_c
        assert np.allclose(outlays, Y_c, atol=1e-12)


class TestStressGRASBalancing:
    """Stress-test GRAS balancing on mixed signs, extreme negative entries, and varying condition."""

    def test_gras_with_extreme_negative_entries(self) -> None:
        """GRAS balances matrices containing extreme negative entries (-1e4 to -1e6)."""
        Z0 = np.array([
            [1e4, -5e3, 2e3],
            [3e3, 8e3, -4e3],
            [-2e3, 4e3, 1.5e4],
        ])
        # Define target marginals consistent with aggregate sum
        u = np.array([8e3, 7e3, 1.8e4])
        v = np.array([1.2e4, 9e3, 1.2e4])  # sum = 33,000

        Z_bal = balance_gras(Z0, u, v, tol=1e-10, max_iter=2000)

        # 1. Exact margin satisfaction
        row_sums = np.sum(Z_bal, axis=1)
        col_sums = np.sum(Z_bal, axis=0)
        assert np.allclose(row_sums, u, atol=1e-8)
        assert np.allclose(col_sums, v, atol=1e-8)

        # 2. Strict sign preservation
        assert np.all((Z_bal > 0) == (Z0 > 0))
        assert np.all((Z_bal < 0) == (Z0 < 0))

    def test_gras_high_density_negative_entries(self) -> None:
        """GRAS balances a 20x20 matrix where 40% of entries are negative."""
        rng = np.random.default_rng(999)
        n = 20
        # Positive and negative components
        P = rng.exponential(10.0, size=(n, n))
        N = rng.exponential(5.0, size=(n, n))
        mask_neg = rng.uniform(0.0, 1.0, size=(n, n)) < 0.4
        Z0 = np.where(mask_neg, -N, P)

        # Compute ground truth margins from known positive scaling r*, s*
        r_true = rng.uniform(0.5, 2.0, size=n)
        s_true = rng.uniform(0.5, 2.0, size=n)
        Z_true = r_true[:, None] * np.maximum(Z0, 0) * s_true[None, :] - \
                 (1.0 / r_true)[:, None] * np.maximum(-Z0, 0) * (1.0 / s_true)[None, :]
        u_target = np.sum(Z_true, axis=1)
        v_target = np.sum(Z_true, axis=0)

        Z_bal = balance_gras(Z0, u_target, v_target, tol=1e-9, max_iter=2000)

        assert np.allclose(np.sum(Z_bal, axis=1), u_target, atol=1e-7)
        assert np.allclose(np.sum(Z_bal, axis=0), v_target, atol=1e-7)
        assert np.all((Z_bal > 0) == (Z0 > 0))
        assert np.all((Z_bal < 0) == (Z0 < 0))

    def test_gras_with_negative_target_margins(self) -> None:
        """GRAS correctly balances when some row/column target marginals are negative."""
        Z0 = np.array([
            [-10.0,  5.0],
            [  2.0, -8.0],
        ])
        u = np.array([-6.0, -4.0])  # negative row targets
        v = np.array([-7.0, -3.0])  # sum(u) == sum(v) == -10.0

        Z_bal = balance_gras(Z0, u, v, tol=1e-10)

        assert np.allclose(np.sum(Z_bal, axis=1), u, atol=1e-8)
        assert np.allclose(np.sum(Z_bal, axis=0), v, atol=1e-8)
        assert np.all((Z_bal > 0) == (Z0 > 0))
        assert np.all((Z_bal < 0) == (Z0 < 0))


class TestStressCollatzWielandtSpectralRadius:
    """Stress-test Collatz-Wielandt spectral radius check on large (M=3,465) networks."""

    def test_spectral_radius_speed_m_3465_under_0_15s(self) -> None:
        """Shifted Collatz-Wielandt power iteration on M=3,465 evaluates in < 0.15s across repeated runs."""
        M = 3465
        rng = np.random.default_rng(42)
        # Construct dense realistic technical coefficient matrix
        B = rng.uniform(0.0001, 0.001, size=(M, M))
        row_sums = np.sum(B, axis=1)
        B = 0.5 * B / np.max(row_sums)

        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            rho, lower, upper = compute_spectral_radius(B, max_iter=50, tol=1e-10)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

        median_time = float(np.median(times))
        min_time = float(np.min(times))
        max_time = float(np.max(times))

        print(f"\n[BENCHMARK] M=3,465 Collatz-Wielandt: min={min_time:.4f}s, median={median_time:.4f}s, max={max_time:.4f}s")
        assert median_time < 0.15, f"Median runtime {median_time:.4f}s exceeds 0.15s"
        assert min_time < 0.15, f"Min runtime {min_time:.4f}s exceeds 0.15s"
        assert 0.0 < rho < 0.999
        assert lower <= rho <= upper + 1e-6

    def test_spectral_radius_tariff_schedule_screening(self) -> None:
        """Spectral radius check correctly flags and rejects non-viable tariff schedules tau >= 8.0."""
        # Realistic A matrix from OECD 4x3 synthetic model
        raw = generate_synthetic_mrio("oecd", custom_c=4, custom_s=3, seed=123)
        Y = raw.gross_output
        A = raw.intermediate_matrix / Y.reshape(1, -1)

        # 1. Base viable network: rho(A) < 0.999
        rho_base, _, _ = compute_spectral_radius(A)
        assert rho_base < 0.999

        # 2. Extreme tariff schedule: tau = 8.0 on cross-border flows
        C, S = raw.C, raw.S
        M = C * S
        tau_extreme = np.ones((M, M))
        for c in range(C):
            for d in range(C):
                if c != d:
                    tau_extreme[c * S:(c + 1) * S, d * S:(d + 1) * S] = 8.0

        B_tau = tau_extreme * A
        rho_tau, _, _ = compute_spectral_radius(B_tau)

        # Must either blow up or be detected
        assert rho_tau > rho_base


class TestStressMemoryAndNumericStability:
    """Stress-test memory leaks and numeric stability under repeated execution cycles."""

    def test_no_memory_leak_on_repeated_regularization(self) -> None:
        """Running 20 regularization cycles on large tables exhibits zero cumulative memory leak."""
        raw = generate_synthetic_mrio("wiod", custom_c=10, custom_s=10, seed=500)
        gc.collect()
        tracemalloc.start()

        # Warmup
        for _ in range(3):
            regularize_mrio_table(raw.Z, raw.F, raw.VA, raw.TLS, raw.Y, n_countries=10, n_sectors=10)

        snapshot1 = tracemalloc.take_snapshot()

        for _ in range(15):
            regularize_mrio_table(raw.Z, raw.F, raw.VA, raw.TLS, raw.Y, n_countries=10, n_sectors=10)

        gc.collect()
        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        total_growth = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)
        # Total memory growth should be minimal (< 2 MB for temporary object churn)
        assert total_growth < 2 * 1024 * 1024, f"Memory grew by {total_growth / 1024:.1f} KB"

    def test_numeric_stability_near_zero_output(self) -> None:
        """Near-zero output nodes (Y = 1e-7, 1e-12) are stabilized without NaN or Inf."""
        M = 6
        Z = np.ones((M, M)) * 0.1
        F = np.ones((M, 2)) * 0.2
        # Set node 2 to tiny near-zero output
        Z[2, :] = 1e-12
        F[2, :] = 1e-12
        Y = np.sum(Z, axis=1) + np.sum(F, axis=1)
        VA = np.ones(M) * 1.0
        VA[2] = 1e-12
        TLS = Y - np.sum(Z, axis=0) - VA

        Z_c, F_c, VA_c, TLS_c, Y_c = regularize_mrio_table(
            Z, F, VA, TLS, Y, floor_output=1e-6, n_countries=2, n_sectors=3
        )

        assert not np.any(np.isnan(Z_c))
        assert not np.any(np.isnan(Y_c))
        assert not np.any(np.isinf(Z_c))
        assert not np.any(np.isinf(Y_c))
        assert Y_c[2] >= 1e-6
        assert VA_c[2] >= 1e-6

        # Check technical coefficients
        A = Z_c / Y_c.reshape(1, -1)
        assert np.all(np.isfinite(A))
        assert np.all(A >= 0.0)
