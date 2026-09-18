"""Tier 5 White-Box Adversarial Coverage Hardening Test Suite: Solver & Diagnostics.

Author: orch26_challenger_m6_tier5_2 (Milestone 6 Phase 2 Challenger 2)
Scope:
1. Dynamic sparsity block S[M:2M, M:2M] activation at boundary values (sigma_y = 1e-6,
   sigma_y = 1e-7, alternating variable markup flags, full_block vs diagonal).
2. Quasi-condensed solver behavior under extreme shocks (500% tariff increases, 90%
   productivity collapse, asymmetric sector shutdowns), verifying latency cap <= 10
   inner iterations is strictly enforced and solver terminates cleanly without hanging.
3. Convergence diagnostics mapping under extreme residuals across all 6 block identities,
   OECD ICIO 2,001 system boundary equations, and 100% airtight JSON serialization
   with zero NumPy scalar leaks.
4. API ergonomics & serialization: copy, deepcopy, pickle, and JSON export of
   FlexibleTradeEquilibriumResult and declarative config objects.

Strictly conforms to the puremacro Pyodide runtime contract (NumPy, SciPy, Pandas only).
"""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
import json
import pickle
from typing import Any

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from puremacro.trade import (
    TradeCalibrationResult,
    calibrate_trade_model,
    compute_equilibrium_residuals,
)
from puremacro.trade.flexible import (
    FlexibleMarketStructureConfig,
    FlexiblePreferenceConfig,
    FlexibleTechnologyConfig,
    FlexibleTradeEquilibriumResult,
    FlexibleTradeModelConfig,
    _quasi_condensed_solve,
    _solve_inner_prices,
    compute_convergence_diagnostics,
    solve_flexible_trade_equilibrium,
)
from puremacro.trade.solver import (
    _build_cge_sparsity_pattern,
    build_initial_guess,
    solve_trade_equilibrium,
)


# =============================================================================
# Fixtures
# =============================================================================

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


# =============================================================================
# 1. Dynamic Sparsity Boundary Transitions
# =============================================================================

class TestDynamicSparsityBoundaryTransitions:
    """Adversarial stress-testing of dynamic sparsity block S[M:2M, M:2M] activation."""

    def test_sigma_y_boundary_threshold_activation(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """Verify strict gating transition at sigma_y = 1e-6 when variable_markups=False."""
        calib = synthetic_2c_2s_calib
        M = calib.n_sectors * calib.n_countries  # 4

        # Sub-threshold values: block must remain inactive (0 non-zeros)
        for sig in [0.0, 1e-9, 1e-8, 5e-7, 9.99e-7]:
            S = _build_cge_sparsity_pattern(calib, sigma_y=sig, variable_markups=False)
            block = S[M : 2 * M, M : 2 * M]
            assert block.nnz == 0, f"Sparsity block unexpectedly active for sub-threshold sigma_y={sig}"

        # At and above threshold: block must be activated (M non-zeros on diagonal)
        for sig in [1e-6, 1.001e-6, 1e-5, 0.1, 0.5, 1.0]:
            S = _build_cge_sparsity_pattern(calib, sigma_y=sig, variable_markups=False)
            block = S[M : 2 * M, M : 2 * M]
            assert block.nnz == M, f"Sparsity block not activated for supra-threshold sigma_y={sig}"
            # Check it is strictly diagonal
            diag_eye = sp.eye(M, dtype=bool)
            assert (block != diag_eye).nnz == 0

    def test_variable_markups_overrides_subthreshold_sigma_y(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """When variable_markups=True, block S[M:2M, M:2M] is active even at sigma_y=0 or 1e-7."""
        calib = synthetic_2c_2s_calib
        M = calib.n_sectors * calib.n_countries

        for sig in [0.0, 1e-10, 1e-7]:
            S_vm = _build_cge_sparsity_pattern(calib, sigma_y=sig, variable_markups=True)
            block_vm = S_vm[M : 2 * M, M : 2 * M]
            assert block_vm.nnz == M, f"variable_markups failed to activate block at sigma_y={sig}"

    def test_alternating_variable_markups_flag_idempotence(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """Toggling variable_markups flag activates and deactivates block idempotently."""
        calib = synthetic_2c_2s_calib
        M = calib.n_sectors * calib.n_countries

        for _ in range(3):
            # Inactive state
            S_off = _build_cge_sparsity_pattern(calib, sigma_y=1e-7, variable_markups=False)
            assert S_off[M : 2 * M, M : 2 * M].nnz == 0
            # Active state
            S_on = _build_cge_sparsity_pattern(calib, sigma_y=1e-7, variable_markups=True)
            assert S_on[M : 2 * M, M : 2 * M].nnz == M

    def test_full_block_activation_vs_diagonal(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """When full_block=True, active block is dense M x M (M^2 non-zeros)."""
        calib = synthetic_2c_2s_calib
        M = calib.n_sectors * calib.n_countries

        # Active with full_block=True
        S_full = _build_cge_sparsity_pattern(calib, sigma_y=0.5, full_block=True)
        block_full = S_full[M : 2 * M, M : 2 * M]
        assert block_full.nnz == M * M, f"full_block did not produce dense block: {block_full.nnz} != {M*M}"

        # Active with full_block=False
        S_diag = _build_cge_sparsity_pattern(calib, sigma_y=0.5, full_block=False)
        block_diag = S_diag[M : 2 * M, M : 2 * M]
        assert block_diag.nnz == M

        # Inactive with full_block=True (inactive state remains 0 non-zeros)
        S_inact = _build_cge_sparsity_pattern(calib, sigma_y=1e-7, full_block=True)
        assert S_inact[M : 2 * M, M : 2 * M].nnz == 0

    def test_sparsity_pattern_config_argument_plumbing(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """All configuration pathways to _build_cge_sparsity_pattern correctly propagate parameters."""
        calib = synthetic_2c_2s_calib
        M = calib.n_sectors * calib.n_countries

        # Pathway 1: via config=FlexibleTradeModelConfig
        cfg1 = FlexibleTradeModelConfig(technology=FlexibleTechnologyConfig(sigma_y=0.4))
        S1 = _build_cge_sparsity_pattern(calib, config=cfg1)
        assert S1[M : 2 * M, M : 2 * M].nnz == M

        cfg2 = FlexibleTradeModelConfig(
            market_structure=FlexibleMarketStructureConfig(variable_markups=True)
        )
        S2 = _build_cge_sparsity_pattern(calib, config=cfg2)
        assert S2[M : 2 * M, M : 2 * M].nnz == M

        # Pathway 2: via kwargs technology / tech_cfg
        S3 = _build_cge_sparsity_pattern(calib, technology=FlexibleTechnologyConfig(sigma_y=0.2))
        assert S3[M : 2 * M, M : 2 * M].nnz == M

        S4 = _build_cge_sparsity_pattern(calib, tech_cfg=FlexibleTechnologyConfig(sigma_y=0.2))
        assert S4[M : 2 * M, M : 2 * M].nnz == M

        # Pathway 3: via kwargs market_structure / market_cfg
        S5 = _build_cge_sparsity_pattern(
            calib, market_structure=FlexibleMarketStructureConfig(variable_markups=True)
        )
        assert S5[M : 2 * M, M : 2 * M].nnz == M

    def test_sparse_lu_solver_across_boundary_transition(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """Sparse LU solver factorizes and converges smoothly across the 1e-6 boundary."""
        calib = synthetic_2c_2s_calib
        tau_shock = np.array([0.15, 0.05])

        # Solve below boundary (sigma_y = 1e-7, inactive block)
        res_below = solve_trade_equilibrium(
            calib, tau=tau_shock, method="sparse_lu", sigma_y=1e-7, variable_markups=False
        )
        assert res_below.converged
        assert res_below.iterations > 0

        # Solve at boundary (sigma_y = 1e-6, active block)
        res_at = solve_trade_equilibrium(
            calib, tau=tau_shock, method="sparse_lu", sigma_y=1e-6, variable_markups=False
        )
        assert res_at.converged
        assert res_at.iterations > 0

        # Verify solutions match closely across boundary
        assert np.allclose(res_below.x_sol, res_at.x_sol, atol=1e-4)


# =============================================================================
# 2. Quasi-Condensed Solver Latency Caps & Fixed-Point Verification
# =============================================================================

class TestQuasiCondensedSolverLatencyCaps:
    """Adversarial stress-testing of quasi-condensed solver iteration bounds and short-circuits."""

    def test_max_inner_iter_latency_cap_validation(self) -> None:
        """Enforces max_inner_iter <= 10 with clear, informative ValueError."""
        # Valid inner iterations
        for k in [1, 5, 10]:
            cfg = FlexibleTradeModelConfig(max_inner_iter=k)
            assert cfg.max_inner_iter == k

        # Exceeding latency cap (e.g. 11, 20) must raise ValueError
        for k_invalid in [11, 15, 50]:
            with pytest.raises(ValueError, match="exceeds the latency cap of 10"):
                FlexibleTradeModelConfig(max_inner_iter=k_invalid)

        # Lower bound violation (k < 1) must raise ValueError
        for k_low in [0, -1, -5]:
            with pytest.raises(ValueError, match="must be >= 1"):
                FlexibleTradeModelConfig(max_inner_iter=k_low)

        # Non-integer must raise TypeError
        for non_int in [5.5, "10", True, None]:
            with pytest.raises(TypeError, match="must be an integer"):
                FlexibleTradeModelConfig(max_inner_iter=non_int)  # type: ignore

    def test_inner_iteration_count_strictly_capped_at_10(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """Contractive inner loop cannot execute more than 10 iterations under any input."""
        calib = synthetic_2c_2s_calib
        ns, nc = calib.n_sectors, calib.n_countries
        M = ns * nc

        # Setup state vector with large factor price disruption
        x_init = build_initial_guess(calib)
        xm = np.zeros(4 * nc - 1)
        xm[:nc] = 0.8        # r
        xm[nc : 2 * nc] = -0.5  # w

        # Non-linear technology requiring contractive iterations
        cfg = FlexibleTradeModelConfig(
            technology=FlexibleTechnologyConfig(sigma_y=0.8, sigma_inter=0.5),
            max_inner_iter=10,
        )

        tau_a = np.ones((M, ns, nc))
        tax_flat = calib.tax.flatten(order="F")
        a_eff_2d = (calib.a * tau_a).reshape((M, M), order="F")
        denom = np.maximum(1.0 - tax_flat[:, np.newaxis], 1e-12)
        B_T_mat = a_eff_2d.T / denom
        import scipy.linalg as la
        M_P_dense = np.eye(M, dtype=float) - B_T_mat
        lu_P_piv = la.lu_factor(M_P_dense)

        class _LUSolver:
            def __init__(self, piv: Any) -> None:
                self._piv = piv
            def solve(self, b: np.ndarray) -> np.ndarray:
                return la.lu_solve(self._piv, b)

        lu_P = _LUSolver(lu_P_piv)

        p_sol, inner_iters = _solve_inner_prices(
            xm_curr=xm,
            calib=calib,
            config=cfg,
            tau_a=tau_a,
            lu_P=lu_P,
        )
        assert inner_iters <= 10, f"Inner iterations {inner_iters} exceeded cap of 10"
        assert inner_iters >= 1
        assert np.all(np.isfinite(p_sol))

    def test_linear_technology_short_circuits_to_lapack(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """Linear technology settings (sigma_y=0, sigma_inter=0, no VM) short-circuit with 0 inner iterations."""
        calib = synthetic_2c_2s_calib
        ns, nc = calib.n_sectors, calib.n_countries
        M = ns * nc

        xm = np.zeros(4 * nc - 1)
        cfg_linear = FlexibleTradeModelConfig(
            technology=FlexibleTechnologyConfig(sigma_y=0.0, sigma_inter=0.0),
            market_structure=FlexibleMarketStructureConfig(variable_markups=False),
        )

        tau_a = np.ones((M, ns, nc))
        tax_flat = calib.tax.flatten(order="F")
        a_eff_2d = (calib.a * tau_a).reshape((M, M), order="F")
        denom = np.maximum(1.0 - tax_flat[:, np.newaxis], 1e-12)
        B_T_mat = a_eff_2d.T / denom
        import scipy.linalg as la
        M_P_dense = np.eye(M, dtype=float) - B_T_mat
        lu_P_piv = la.lu_factor(M_P_dense)

        class _LUSolver:
            def __init__(self, piv: Any) -> None:
                self._piv = piv
            def solve(self, b: np.ndarray) -> np.ndarray:
                return la.lu_solve(self._piv, b)

        lu_P = _LUSolver(lu_P_piv)

        p_sol, inner_iters = _solve_inner_prices(
            xm_curr=xm,
            calib=calib,
            config=cfg_linear,
            tau_a=tau_a,
            lu_P=lu_P,
        )
        assert inner_iters == 0, f"Expected 0 inner iterations under linear short-circuit, got {inner_iters}"
        assert np.all(np.isfinite(p_sol))


# =============================================================================
# 3. Quasi-Condensed Extreme Multi-Shocks & Clean Failure Handling
# =============================================================================

class TestQuasiCondensedExtremeMultiShocks:
    """Adversarial stress-testing of quasi-condensed solver under extreme multi-shock scenarios."""

    def test_extreme_500pct_tariff_shock_clean_termination(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """500% tariff shock terminates cleanly in <= max_iter without hanging or NaN crashes."""
        calib = synthetic_2c_2s_calib
        tau_500 = np.array([5.0, 5.0])  # 500% ad-valorem tariff

        # Run with max_iter=15 to test clean non-converged termination
        res = solve_flexible_trade_equilibrium(
            calib, tau=tau_500, method="quasi_condensed", max_iter=15
        )
        assert res.iterations <= 15
        assert isinstance(res.converged, bool)
        assert np.all(np.isfinite(res.residuals))
        assert np.all(np.isfinite(res.x_sol))

        # If not converged, convergence diagnostic must be attached
        if not res.converged:
            diag = res.convergence_diagnostic
            assert isinstance(diag, dict)
            assert "top_equations" in diag
            assert len(diag["top_equations"]) >= 1

    def test_extreme_90pct_factor_endowment_collapse(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """90% collapse in national factor endowments evaluates with finite residuals."""
        calib = synthetic_2c_2s_calib
        # Severe supply destruction: 90% drop in labor and capital endowments
        calib_collapse = replace(
            calib,
            l_endow=calib.l_endow * 0.10,
            k_endow=calib.k_endow * 0.10,
        )

        res = solve_flexible_trade_equilibrium(
            calib_collapse, method="quasi_condensed", max_iter=20
        )
        assert res.iterations <= 20
        assert np.all(np.isfinite(res.residuals))
        assert np.all(np.isfinite(res.x_sol))
        assert res.residual_norm >= 0.0

    def test_asymmetric_country_labor_collapse_convergence(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """Country 0 suffers 90% labor collapse while Country 1 remains intact: converges cleanly."""
        calib = synthetic_2c_2s_calib
        l_endow_asym = calib.l_endow.copy()
        l_endow_asym[0, 0] *= 0.10  # Country 0 labor shock
        calib_asym = replace(calib, l_endow=l_endow_asym)

        res = solve_flexible_trade_equilibrium(
            calib_asym, method="quasi_condensed", max_iter=25, tol=1e-3
        )
        assert res.converged, f"Asymmetric labor collapse failed to converge: max_res={res.residual_norm}"
        assert res.iterations <= 25
        assert np.all(np.isfinite(res.residuals))
        assert res.residual_norm < 1e-3

    def test_asymmetric_sector_specific_prohibitive_tariff(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """Country 0 imposes 500% tariff on Sector 0 imports from Country 1: converges cleanly."""
        calib = synthetic_2c_2s_calib
        nc, ns = calib.n_countries, calib.n_sectors
        tau_asym = np.ones((ns * nc, ns, nc), dtype=float)
        # Import of sector 0 from origin country 1 into destination country 0
        tau_asym[2:, 0, 0] = 6.0  # (1 + 5.0)

        res = solve_flexible_trade_equilibrium(
            calib, tau=tau_asym, method="quasi_condensed", max_iter=25, tol=1e-3
        )
        assert res.converged, f"Asymmetric sector tariff failed to converge: max_res={res.residual_norm}"
        assert res.iterations <= 25
        assert np.all(np.isfinite(res.residuals))

    def test_compound_triple_extreme_shock_resilience(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """Compound shock (100% tariff + 50% labor shock + flexible markups + non-homothetic prefs)."""
        calib = synthetic_2c_2s_calib
        l_endow_compound = calib.l_endow.copy()
        l_endow_compound[0, 0] *= 0.50
        calib_compound = replace(calib, l_endow=l_endow_compound)

        cfg = FlexibleTradeModelConfig(
            technology=FlexibleTechnologyConfig(rho_va=0.7, sigma_y=0.3),
            preference=FlexiblePreferenceConfig(subsistence_shares={"C00": 0.15}),
            market_structure=FlexibleMarketStructureConfig(variable_markups=True),
            max_inner_iter=10,
        )

        tau_100 = np.array([1.0, 0.5])
        res = solve_flexible_trade_equilibrium(
            calib_compound,
            config=cfg,
            tau=tau_100,
            method="quasi_condensed",
            max_iter=15,
        )
        # Verify finite evaluation and clean termination
        assert res.iterations <= 15
        assert np.all(np.isfinite(res.residuals))
        assert np.all(np.isfinite(res.x_sol))
        if not res.converged:
            assert "top_equations" in res.convergence_diagnostic


# =============================================================================
# 3. Convergence Diagnostics Mapping & Airtight JSON Serialization
# =============================================================================

class TestConvergenceDiagnosticsAdversarial:
    """Adversarial stress-testing of convergence diagnostics across all 6 block identities."""

    def test_all_six_blocks_mapped_to_correct_economic_identities(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """Verify each of the 6 economic blocks is accurately mapped when disturbed."""
        calib = synthetic_2c_2s_calib
        nc, ns = calib.n_countries, calib.n_sectors
        M = ns * nc
        n = 2 * M + 4 * nc - 1

        # Test index in each of the 6 blocks
        blocks_data = [
            ("Goods Market Clearing", 1, 1e12, "S01", "C00"),
            ("Zero-Profit Condition", M + 2, -5e11, "S00", "C01"),
            ("Labor Market Clearing", 2 * M + 1, 8e10, None, "C01"),
            ("Capital Market Clearing", 2 * M + nc + 0, -9e10, None, "C00"),
            ("Trade Balance", 2 * M + 2 * nc + 0, 7e9, None, "C00"),
            ("Fiscal Budget Consistency", 2 * M + 3 * nc - 1, -6e9, None, "C00"),
        ]

        for expected_block, idx, val, expected_sec, expected_cntry in blocks_data:
            r = np.zeros(n)
            r[idx] = val
            diag = compute_convergence_diagnostics(r, calib, n_top=3)
            assert len(diag["top_equations"]) >= 1
            top1 = diag["top_equations"][0]
            assert top1["rank"] == 1
            assert top1["index"] == idx
            assert top1["block"] == expected_block, (
                f"Expected block '{expected_block}' for idx={idx}, got '{top1['block']}'"
            )
            assert top1["country"] == expected_cntry
            if expected_sec is not None:
                assert top1["sector"] == expected_sec
            assert top1["residual"] == pytest.approx(val)
            assert top1["abs_residual"] == pytest.approx(abs(val))

    def test_competing_multi_block_extreme_residuals_ranking(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """Multiple competing blocks with distinct residual magnitudes are sorted strictly descending."""
        calib = synthetic_2c_2s_calib
        nc, ns = calib.n_countries, calib.n_sectors
        M = ns * nc
        n = 2 * M + 4 * nc - 1

        r = np.zeros(n)
        r[2 * M + 3 * nc - 1] = 1e11   # Rank 1: Fiscal Budget Consistency
        r[2 * M + nc + 0] = -1e9        # Rank 2: Capital Market Clearing
        r[M + 2] = 1e7                  # Rank 3: Zero-Profit Condition
        r[0] = 50.0                     # Rank 4: Goods Market Clearing

        diag = compute_convergence_diagnostics(r, calib, n_top=3)
        top = diag["top_equations"]
        assert len(top) == 3

        assert top[0]["rank"] == 1
        assert top[0]["block"] == "Fiscal Budget Consistency"
        assert top[0]["abs_residual"] == pytest.approx(1e11)

        assert top[1]["rank"] == 2
        assert top[1]["block"] == "Capital Market Clearing"
        assert top[1]["abs_residual"] == pytest.approx(1e9)

        assert top[2]["rank"] == 3
        assert top[2]["block"] == "Zero-Profit Condition"
        assert top[2]["abs_residual"] == pytest.approx(1e7)

    def test_canonical_2001_equation_system_boundary_equations(self) -> None:
        """All 12 boundary equations (start/end of each block) in 2,001 OECD system map without error."""
        boundaries = [
            (0, "Goods Market Clearing", "C00", "S00"),
            (846, "Goods Market Clearing", "C76", "S10"),
            (847, "Zero-Profit Condition", "C00", "S00"),
            (1693, "Zero-Profit Condition", "C76", "S10"),
            (1694, "Labor Market Clearing", "C00", None),
            (1770, "Labor Market Clearing", "C76", None),
            (1771, "Capital Market Clearing", "C00", None),
            (1847, "Capital Market Clearing", "C76", None),
            (1848, "Trade Balance", "C00", None),
            (1923, "Trade Balance", "C75", None),
            (1924, "Fiscal Budget Consistency", "C00", None),
            (2000, "Fiscal Budget Consistency", "C76", None),
        ]

        for idx, expected_block, exp_c, exp_s in boundaries:
            r = np.zeros(2001)
            r[idx] = 12345.67
            diag = compute_convergence_diagnostics(r, calib=None, n_top=1)
            top1 = diag["top_equations"][0]
            assert top1["index"] == idx
            assert top1["block"] == expected_block, f"Index {idx} expected {expected_block}, got {top1['block']}"
            assert top1["country"] == exp_c, f"Index {idx} expected {exp_c}, got {top1['country']}"
            if exp_s is not None:
                assert top1["sector"] == exp_s, f"Index {idx} expected {exp_s}, got {top1['sector']}"

    def test_airtight_json_serialization_zero_numpy_scalar_leaks(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """Diagnostics dictionary is 100% airtight JSON-serializable with zero NumPy scalar types."""
        calib = synthetic_2c_2s_calib
        M = calib.n_sectors * calib.n_countries
        n = 2 * M + 4 * calib.n_countries - 1

        # Use float64 NumPy array
        r = np.array([float(i * 1.5 - 10.0) for i in range(n)], dtype=np.float64)
        diag = compute_convergence_diagnostics(r, calib, n_top=5)

        # Recursive check to verify NO numpy types anywhere in the tree
        def _assert_no_numpy(obj: Any, path: str = "root") -> None:
            assert not isinstance(obj, (np.generic, np.ndarray)), (
                f"NumPy type leak detected: {type(obj)} at path '{path}'"
            )
            if isinstance(obj, dict):
                for k, v in obj.items():
                    assert isinstance(k, str), f"Dictionary key is not string at {path}.{k}"
                    _assert_no_numpy(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, elem in enumerate(obj):
                    _assert_no_numpy(elem, f"{path}[{i}]")

        _assert_no_numpy(diag)

        # Roundtrip JSON dump and load
        dumped_json = json.dumps(diag, allow_nan=False)
        loaded = json.loads(dumped_json)
        assert loaded["worst_residual"] == pytest.approx(diag["worst_residual"])
        assert len(loaded["top_equations"]) == 5

    def test_diagnostics_edge_case_inputs(self) -> None:
        """compute_convergence_diagnostics handles None, empty, and boundary n_top robustly."""
        # 1. None input
        diag_none = compute_convergence_diagnostics(None)
        assert diag_none["top_equations"] == []
        assert diag_none["max_residual"] == 0.0

        # 2. Empty input
        diag_empty = compute_convergence_diagnostics(np.array([]))
        assert diag_empty["top_equations"] == []
        assert diag_empty["max_residual"] == 0.0

        # 3. All-zero residuals
        diag_zero = compute_convergence_diagnostics(np.zeros(15), n_top=3)
        assert len(diag_zero["top_equations"]) == 3
        assert diag_zero["worst_residual"] == 0.0
        assert diag_zero["l1_norm"] == 0.0

        # 4. n_top bounds
        r = np.array([1.0, 2.0, 3.0])
        assert len(compute_convergence_diagnostics(r, n_top=0)["top_equations"]) == 0
        assert len(compute_convergence_diagnostics(r, n_top=1)["top_equations"]) == 1
        assert len(compute_convergence_diagnostics(r, n_top=100)["top_equations"]) == 3

    def test_spec_layout_vs_codebase_layout(self) -> None:
        """layout='spec' correctly maps indices to theoretical dual ordering."""
        # Index 0 in 'spec' is Zero-Profit Condition, while in 'codebase' it is Goods Market Clearing
        r = np.zeros(2001)
        r[0] = 50.0

        diag_spec = compute_convergence_diagnostics(r, layout="spec", n_top=1)
        assert diag_spec["top_equations"][0]["block"] == "Zero-Profit Condition"

        diag_code = compute_convergence_diagnostics(r, layout="codebase", n_top=1)
        assert diag_code["top_equations"][0]["block"] == "Goods Market Clearing"


# =============================================================================
# 4. API Ergonomics & Serialization
# =============================================================================

class TestAPIErgonomicsAndSerialization:
    """Adversarial testing of copy, deepcopy, pickle, and JSON export for result and configs."""

    def test_result_copy_and_deepcopy_independence(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """copy and deepcopy preserve all result fields, deepcopy ensures full array isolation."""
        calib = synthetic_2c_2s_calib
        res = solve_flexible_trade_equilibrium(calib, rho_va=0.8, sigma_y=0.2)
        assert res.converged

        # Shallow copy
        res_copy = copy.copy(res)
        assert np.allclose(res_copy.p_sol, res.p_sol)
        assert res_copy.converged == res.converged

        # Deep copy
        res_deep = copy.deepcopy(res)
        assert np.allclose(res_deep.p_sol, res.p_sol)

        # Mutate deep copy's state vector: original must remain completely untouched
        res_deep.x_sol[0] += 999.0
        assert not np.isclose(res_deep.x_sol[0], res.x_sol[0])

    def test_result_pickle_roundtrip_all_methods(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """Pickle roundtrip preserves result container and all progressive inspection methods."""
        calib = synthetic_2c_2s_calib
        res = solve_flexible_trade_equilibrium(calib, rho_va=0.8, sigma_y=0.2)

        # Pre-compute frames
        df_factors_orig = res.factor_allocation_frame()
        df_markups_orig = res.summary_markups()
        df_welfare_orig = res.welfare_decomposition()

        # Pickle dump and load
        serialized = pickle.dumps(res)
        res_unpickled = pickle.loads(serialized)

        # Verify properties
        assert np.allclose(res_unpickled.p_sol, res.p_sol)
        assert np.allclose(res_unpickled.y_sol, res.y_sol)
        assert np.allclose(res_unpickled.w_sol, res.w_sol)
        assert np.allclose(res_unpickled.r_sol, res.r_sol)

        # Verify inspection DataFrame methods
        df_factors_unpickled = res_unpickled.factor_allocation_frame()
        pd.testing.assert_frame_equal(df_factors_orig, df_factors_unpickled)

        df_markups_unpickled = res_unpickled.summary_markups()
        pd.testing.assert_frame_equal(df_markups_orig, df_markups_unpickled)

        df_welfare_unpickled = res_unpickled.welfare_decomposition()
        pd.testing.assert_frame_equal(df_welfare_orig, df_welfare_unpickled)

    def test_all_config_classes_copy_deepcopy_pickle(self) -> None:
        """All four configuration dataclasses support copy, deepcopy, and pickle seamlessly."""
        configs = [
            FlexibleTechnologyConfig(rho_va=0.8, sigma_y=0.2, sigma_inter=0.1),
            FlexiblePreferenceConfig(subsistence_shares={"C00": 0.1}, sigma_trade=6.0),
            FlexibleMarketStructureConfig(variable_markups=True, variety_condensation=True),
            FlexibleTradeModelConfig(
                technology=FlexibleTechnologyConfig(rho_va=0.7),
                preference=FlexiblePreferenceConfig(sigma_trade=5.5),
                market_structure=FlexibleMarketStructureConfig(variable_markups=True),
                max_inner_iter=5,
            ),
        ]

        for cfg in configs:
            # Copy
            cfg_c = copy.copy(cfg)
            assert cfg_c == cfg
            # Deepcopy
            cfg_dc = copy.deepcopy(cfg)
            assert cfg_dc == cfg
            # Pickle
            cfg_p = pickle.loads(pickle.dumps(cfg))
            assert cfg_p == cfg

    def test_config_dataclass_asdict_and_json_roundtrip(self) -> None:
        """asdict serialization of FlexibleTradeModelConfig produces airtight JSON."""
        cfg = FlexibleTradeModelConfig(
            technology=FlexibleTechnologyConfig(rho_va=0.8, sigma_y=0.2),
            preference=FlexiblePreferenceConfig(subsistence_shares={"C00": 0.1}),
            market_structure=FlexibleMarketStructureConfig(variable_markups=True),
            max_inner_iter=6,
        )

        d = asdict(cfg)
        assert isinstance(d, dict)

        # Strict JSON dump (allow_nan=False)
        json_str = json.dumps(d, allow_nan=False)
        loaded = json.loads(json_str)

        # Reconstruct config from dictionary
        cfg_reconstructed = FlexibleTradeModelConfig(
            technology=FlexibleTechnologyConfig(**loaded["technology"]),
            preference=FlexiblePreferenceConfig(**loaded["preference"]),
            market_structure=FlexibleMarketStructureConfig(**loaded["market_structure"]),
            max_inner_iter=loaded["max_inner_iter"],
        )
        assert cfg_reconstructed == cfg

    def test_dataframe_export_methods_json(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """Result DataFrame inspection methods export valid JSON records."""
        calib = synthetic_2c_2s_calib
        res = solve_flexible_trade_equilibrium(calib, rho_va=0.8)

        # 1. factor_allocation_frame
        factors_json = res.factor_allocation_frame().to_json(orient="records")
        assert len(json.loads(factors_json)) == 4

        # 2. summary_markups
        markups_json = res.summary_markups().to_json(orient="records")
        assert len(json.loads(markups_json)) >= 1

        # 3. welfare_decomposition
        welfare_json = res.welfare_decomposition().to_json(orient="records")
        assert len(json.loads(welfare_json)) == 2

    def test_config_frozen_immutability_protection(self) -> None:
        """Attempting to mutate frozen configuration dataclasses raises error."""
        tech = FlexibleTechnologyConfig(rho_va=0.8)
        with pytest.raises(Exception):  # FrozenInstanceError
            tech.rho_va = 0.5  # type: ignore

        pref = FlexiblePreferenceConfig(sigma_trade=5.0)
        with pytest.raises(Exception):
            pref.sigma_trade = 4.0  # type: ignore

        mkt = FlexibleMarketStructureConfig(variable_markups=False)
        with pytest.raises(Exception):
            mkt.variable_markups = True  # type: ignore

    def test_invalid_parameters_raise_economic_value_errors(self) -> None:
        """Passing invalid economic parameters raises immediate, informative ValueError."""
        with pytest.raises(ValueError, match="violates strict quasi-concavity"):
            FlexibleTechnologyConfig(rho_va=0.0)

        with pytest.raises(ValueError, match="violates strict quasi-concavity"):
            FlexibleTechnologyConfig(rho_va=-0.5)

        with pytest.raises(ValueError, match="must be non-negative"):
            FlexibleTechnologyConfig(sigma_y=-0.1)

        with pytest.raises(ValueError, match="must be strictly less than 1.0"):
            FlexiblePreferenceConfig(subsistence_shares={"C00": 1.0})

        with pytest.raises(ValueError, match="cannot exceed sigma_j"):
            FlexibleMarketStructureConfig(theta_j=10.0, sigma_j=5.0)

        with pytest.raises(ValueError, match="sigma_j must be strictly greater than 1.0"):
            FlexibleMarketStructureConfig(sigma_j=0.5)

        with pytest.raises(ValueError, match="theta_j must be strictly greater than 1.0"):
            FlexibleMarketStructureConfig(theta_j=0.5)

    def test_dataclass_replace_on_all_configs(self) -> None:
        """dataclasses.replace functions correctly and synchronizes alias fields."""
        tech = FlexibleTechnologyConfig(rho_va=0.8)
        tech_replaced = replace(tech, rho_va=0.6)
        assert tech_replaced.rho_va == 0.6

        # Alias synchronization
        tech_alias = replace(tech, sigma_va=0.4)
        assert tech_alias.rho_va == 0.4

        pref = FlexiblePreferenceConfig(sigma_trade=5.0)
        pref_replaced = replace(pref, sigma_trade=7.0)
        assert pref_replaced.sigma_trade == 7.0

        mkt = FlexibleMarketStructureConfig(variable_markups=False)
        mkt_replaced = replace(mkt, variable_markups=True)
        assert mkt_replaced.variable_markups

        model_cfg = FlexibleTradeModelConfig(max_inner_iter=10)
        model_replaced = replace(model_cfg, max_inner_iter=4)
        assert model_replaced.max_inner_iter == 4

    def test_factor_allocation_frame_endowment_conservation(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """Sum of sector factor demands xl, xk matches national endowments."""
        calib = synthetic_2c_2s_calib
        res = solve_flexible_trade_equilibrium(calib, rho_va=0.8, sigma_y=0.2)
        df_factors = res.factor_allocation_frame()

        for c_idx, c_code in enumerate(calib.country_codes):
            sub = df_factors[df_factors["country"] == c_code]
            total_l = sub["labor"].sum()
            total_k = sub["capital"].sum()
            assert total_l == pytest.approx(float(calib.l_endow[0, c_idx]), rel=1e-3)
            assert total_k == pytest.approx(float(calib.k_endow[0, c_idx]), rel=1e-3)

    def test_welfare_decomposition_self_identity(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """Welfare decomposition comparing solution to itself produces zero EV and zero TOT effect."""
        calib = synthetic_2c_2s_calib
        res = solve_flexible_trade_equilibrium(calib, rho_va=0.8)
        df_self = res.welfare_decomposition(res)
        assert np.allclose(df_self["EV"], 0.0, atol=1e-12)
        assert np.allclose(df_self["terms_of_trade"], 0.0, atol=1e-12)
        assert np.allclose(df_self["efficiency"], 0.0, atol=1e-12)

        # welfare_summary alias
        df_alias = res.welfare_summary(res)
        pd.testing.assert_frame_equal(df_self, df_alias)

    def test_progressive_two_line_solver_dispatch(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """Progressive disclosure 2-line solver works with diverse keyword argument permutations."""
        calib = synthetic_2c_2s_calib

        # Permutation 1: rho_va only
        res1 = solve_flexible_trade_equilibrium(calib, rho_va=0.7)
        assert res1.converged
        assert res1.config is not None
        assert res1.config.technology.rho_va == 0.7

        # Permutation 2: sigma_y and variable_markups
        res2 = solve_flexible_trade_equilibrium(calib, sigma_y=0.3, variable_markups=True)
        assert res2.converged
        assert res2.config.technology.sigma_y == 0.3
        assert res2.config.market_structure.variable_markups

        # Permutation 3: quasi_condensed method
        res3 = solve_flexible_trade_equilibrium(calib, rho_va=0.9, method="quasi_condensed")
        assert res3.converged
        assert res3.iterations >= 0


class TestAdditionalSolverAndDiagnosticsHardening:
    """Additional edge cases for solver density, NaN diagnostics, and custom labels."""

    def test_total_cge_sparsity_density_bounds(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """Total structural sparsity pattern density is bounded strictly < 0.60 for inactive and < 0.70 for full block."""
        calib = synthetic_2c_2s_calib
        # Inactive configuration
        S_inactive = _build_cge_sparsity_pattern(calib, sigma_y=1e-7, variable_markups=False)
        dens_inactive = S_inactive.nnz / (S_inactive.shape[0] * S_inactive.shape[1])
        assert dens_inactive < 0.60

        # Active diagonal configuration
        S_diag = _build_cge_sparsity_pattern(calib, sigma_y=0.5, variable_markups=True, full_block=False)
        dens_diag = S_diag.nnz / (S_diag.shape[0] * S_diag.shape[1])
        assert dens_diag < 0.62

        # Active full block configuration
        S_full = _build_cge_sparsity_pattern(calib, sigma_y=0.5, variable_markups=True, full_block=True)
        dens_full = S_full.nnz / (S_full.shape[0] * S_full.shape[1])
        assert dens_full < 0.70

    def test_both_sigma_y_and_variable_markups_active(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """When both sigma_y >= 1e-6 and variable_markups=True, block is activated with M nonzeros."""
        calib = synthetic_2c_2s_calib
        M = calib.n_sectors * calib.n_countries
        S = _build_cge_sparsity_pattern(calib, sigma_y=0.2, variable_markups=True)
        assert S[M : 2 * M, M : 2 * M].nnz == M

    def test_full_block_with_variable_markups(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """full_block=True produces dense M x M block when variable_markups=True even with sigma_y=0."""
        calib = synthetic_2c_2s_calib
        M = calib.n_sectors * calib.n_countries
        S = _build_cge_sparsity_pattern(calib, sigma_y=0.0, variable_markups=True, full_block=True)
        assert S[M : 2 * M, M : 2 * M].nnz == M * M

    def test_max_iter_zero_immediate_halt_with_diagnostics(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """max_iter=0 halts immediately and attaches diagnostics identifying top residual equations."""
        calib = synthetic_2c_2s_calib
        res = solve_flexible_trade_equilibrium(calib, max_iter=0)
        assert res.iterations == 0
        assert not res.converged
        assert "convergence_diagnostic" in res.metadata
        assert len(res.top_offending_equations) >= 1
        assert "Increase solver max_iter" in res.convergence_diagnostic.get("remediation", "")

    def test_diagnostics_nan_inf_divergence_detection(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """Diagnostics correctly identify numerical divergence when residuals contain NaN or Inf."""
        calib = synthetic_2c_2s_calib
        n = 2 * (calib.n_sectors * calib.n_countries) + 4 * calib.n_countries - 1

        r = np.zeros(n)
        r[0] = np.nan
        r[1] = 100.0

        diag = compute_convergence_diagnostics(r, calib)
        assert "Severe numerical divergence detected" in diag["remediation"]
        # JSON dump must succeed with default json.dumps
        dumped = json.dumps(diag)
        assert len(dumped) > 0

    def test_diagnostics_custom_country_and_sector_codes(self) -> None:
        """Custom country and sector codes are properly reflected in equation labels."""
        class _MockCalib:
            n_countries = 2
            n_sectors = 2
            country_codes = ("USA", "CHN")
            sector_codes = ("MANU", "SERV")

        mock_calib = _MockCalib()
        M = 4
        n = 2 * M + 4 * 2 - 1

        r = np.zeros(n)
        # Sector MANU (s=0), Country CHN (c=1) in Goods Market Clearing: idx = 0 + 1*2 + 0 = 2
        r[2] = 55.5
        diag = compute_convergence_diagnostics(r, mock_calib, n_top=1)
        top1 = diag["top_equations"][0]
        assert top1["country"] == "CHN"
        assert top1["sector"] == "MANU"
        assert "Sector MANU, Country CHN" in top1["equation"]

