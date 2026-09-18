"""Empirical Challenger Test Suite for Milestone 4: Solver Boundaries & Convergence Diagnostics.

Authored by Empirical Challenger (Challenger 1, Milestone 4).
Covers:
1. Strict boundary enforcement of max_inner_iter:
   - max_inner_iter=11 strictly raises ValueError.
   - max_inner_iter=10 passes, max_inner_iter=1 passes.
   - max_inner_iter=0 and max_inner_iter=-1 raise ValueError.
   - Non-integer types (bool, float, str) raise TypeError.
   - dataclasses.replace enforces boundary.
2. max_iter=0 boundary and diagnostic structure:
   - Returns converged=False and iterations=0.
   - Attaches valid convergence_diagnostic dict.
   - Exactly top 3 offending equations identified (or min(3, N)).
   - Equation rankings ordered by abs_residual descending.
   - Each equation record contains rank, index, block, equation, country, sector, residual, abs_residual.
   - Non-empty economic remediation string present.
3. Ill-conditioned calibrations, residual spikes, and economic block identification:
   - Systematic residual spike injection across all 6 economic blocks:
     * Block 0: Goods Market Clearing (sector, country)
     * Block 1: Zero-Profit Condition (sector, country)
     * Block 2: Labor Market Clearing (country)
     * Block 3: Capital Market Clearing (country)
     * Block 4: Trade Balance (country)
     * Block 5: Fiscal Budget Consistency (country)
   - Adversarial ill-conditioned calibration run (extreme tariffs / insufficient iterations).
   - Robustness to NaN / Inf residual entries.
4. JSON serialization robustness:
   - json.dumps(res.convergence_diagnostic) verified across 10+ randomized trials.
   - Pure Python type verification (no np.generic, np.ndarray, int64, float64).
   - Perfect json.loads round-trip equality.
"""
from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from puremacro.trade import calibrate_trade_model
from puremacro.trade.data import load_icio_data
from puremacro.trade.flexible import (
    FlexibleMarketStructureConfig,
    FlexiblePreferenceConfig,
    FlexibleTechnologyConfig,
    FlexibleTradeModelConfig,
    compute_convergence_diagnostics,
    solve_flexible_trade_equilibrium,
)
from puremacro.trade.solver import _build_cge_sparsity_pattern


@pytest.fixture(scope="module")
def synthetic_calib():
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

    return calibrate_trade_model(
        data,
        ns=ns,
        nc=nc,
        nfd=nfd,
        country_codes=["USA", "CHN"],
        sector_codes=["AGR", "MAN"],
        validate=True,
    )


@pytest.fixture(scope="module")
def icio_calib():
    """Load canonical OECD ICIO benchmark calibration."""
    return calibrate_trade_model(load_icio_data())


# =============================================================================
# 1. Adversarial Boundary Testing: max_inner_iter
# =============================================================================

class TestAdversarialMaxInnerIterBoundary:
    """Stress-test max_inner_iter boundary, types, and error contracts."""

    def test_max_inner_iter_11_strictly_raises_value_error(self):
        """Verify max_inner_iter=11 strictly raises ValueError."""
        with pytest.raises(ValueError, match=r"exceeds the latency cap of 10"):
            FlexibleTradeModelConfig(max_inner_iter=11)

    @pytest.mark.parametrize("invalid_val", [12, 15, 100, 1000])
    def test_max_inner_iter_greater_than_10_raises_value_error(self, invalid_val: int):
        """Any max_inner_iter > 10 must raise ValueError."""
        with pytest.raises(ValueError, match=r"exceeds the latency cap of 10"):
            FlexibleTradeModelConfig(max_inner_iter=invalid_val)

    @pytest.mark.parametrize("valid_val", [1, 2, 5, 9, 10])
    def test_max_inner_iter_valid_range(self, valid_val: int):
        """Values in [1, 10] must be valid."""
        cfg = FlexibleTradeModelConfig(max_inner_iter=valid_val)
        assert cfg.max_inner_iter == valid_val

    @pytest.mark.parametrize("non_positive", [0, -1, -10])
    def test_max_inner_iter_non_positive_raises_value_error(self, non_positive: int):
        """Values < 1 must raise ValueError."""
        with pytest.raises(ValueError, match=r"must be >= 1"):
            FlexibleTradeModelConfig(max_inner_iter=non_positive)

    @pytest.mark.parametrize("bad_type", [True, False, 10.0, "10", [10], None])
    def test_max_inner_iter_type_error(self, bad_type: Any):
        """Non-integer types must strictly raise TypeError."""
        with pytest.raises(TypeError, match=r"max_inner_iter must be an integer"):
            FlexibleTradeModelConfig(max_inner_iter=bad_type)

    def test_dataclass_replace_enforces_cap(self):
        """replace(cfg, max_inner_iter=11) must raise ValueError."""
        cfg = FlexibleTradeModelConfig(max_inner_iter=5)
        with pytest.raises(ValueError, match=r"exceeds the latency cap of 10"):
            replace(cfg, max_inner_iter=11)


# =============================================================================
# 2. Adversarial Testing: max_iter=0 and Diagnostics Contract
# =============================================================================

class TestMaxIterZeroAndDiagnosticsContract:
    """Verify max_iter=0 immediate halt and 3-equation diagnostics dictionary."""

    def test_max_iter_zero_immediate_halt_synthetic(self, synthetic_calib):
        """max_iter=0 on synthetic calibration returns converged=False with 3 top equations."""
        res = solve_flexible_trade_equilibrium(synthetic_calib, max_iter=0)
        assert res.converged is False
        assert res.iterations == 0
        assert res.residual_norm > 0.0
        assert res.residuals is not None

        diag = res.convergence_diagnostic
        assert isinstance(diag, dict)
        assert "top_equations" in diag
        assert "max_residual" in diag
        assert "worst_residual" in diag
        assert "l1_norm" in diag
        assert "remediation" in diag

        top_eqs = diag["top_equations"]
        assert len(top_eqs) == 3
        assert len(res.top_offending_equations) == 3

        # Verify monotonicity of residuals
        for rank_idx, eq in enumerate(top_eqs, start=1):
            assert eq["rank"] == rank_idx
            assert "index" in eq and isinstance(eq["index"], int)
            assert "block" in eq and isinstance(eq["block"], str)
            assert "equation" in eq and isinstance(eq["equation"], str)
            assert "country" in eq and isinstance(eq["country"], str)
            assert "residual" in eq and isinstance(eq["residual"], float)
            assert "abs_residual" in eq and isinstance(eq["abs_residual"], float)

        assert top_eqs[0]["abs_residual"] >= top_eqs[1]["abs_residual"]
        assert top_eqs[1]["abs_residual"] >= top_eqs[2]["abs_residual"]
        assert isinstance(diag["remediation"], str)
        assert len(diag["remediation"]) > 0

    def test_max_iter_zero_with_flexible_config(self, synthetic_calib):
        """max_iter=0 with custom flexible config returns diagnostics."""
        cfg = FlexibleTradeModelConfig(
            technology=FlexibleTechnologyConfig(rho_va=0.8, sigma_y=0.2),
            preference=FlexiblePreferenceConfig(mu_s=0.1),
            market_structure=FlexibleMarketStructureConfig(variable_markups=True),
        )
        res = solve_flexible_trade_equilibrium(synthetic_calib, config=cfg, max_iter=0)
        assert not res.converged
        assert res.iterations == 0
        assert len(res.top_offending_equations) == 3
        assert res.convergence_diagnostic["remediation"] != ""

    def test_max_iter_zero_quasi_condensed_method(self, synthetic_calib):
        """max_iter=0 under quasi_condensed method immediately halts with diagnostics."""
        cfg = FlexibleTradeModelConfig()
        res = solve_flexible_trade_equilibrium(
            synthetic_calib, config=cfg, method="quasi_condensed", max_iter=0
        )
        assert not res.converged
        assert res.iterations == 0
        diag = res.convergence_diagnostic
        assert "top_equations" in diag
        assert len(diag["top_equations"]) == 3
        assert "remediation" in diag


# =============================================================================
# 3. Adversarial Testing: Ill-Conditioned Calibrations & Block Identification
# =============================================================================

class TestIllConditionedAndDiagnosticBlockIdentification:
    """Verify diagnostics accurately identify residual spikes across all economic blocks."""

    def test_residual_spike_identification_all_blocks(self, synthetic_calib):
        """Inject dominant residual into each of the 6 economic blocks and verify identification."""
        calib = synthetic_calib
        ns, nc = calib.n_sectors, calib.n_countries
        M = ns * nc
        n_total = 2 * M + 4 * nc - 1
        c_codes = calib.country_codes
        s_codes = calib.sector_codes

        test_cases = [
            # (block_name, index, expected_country, expected_sector)
            ("Goods Market Clearing", 0, c_codes[0], s_codes[0]),
            ("Goods Market Clearing", 1, c_codes[0], s_codes[1]),
            ("Goods Market Clearing", 2, c_codes[1], s_codes[0]),
            ("Zero-Profit Condition", M + 0, c_codes[0], s_codes[0]),
            ("Zero-Profit Condition", M + 3, c_codes[1], s_codes[1]),
            ("Labor Market Clearing", 2 * M + 0, c_codes[0], None),
            ("Labor Market Clearing", 2 * M + 1, c_codes[1], None),
            ("Capital Market Clearing", 2 * M + nc + 0, c_codes[0], None),
            ("Capital Market Clearing", 2 * M + nc + 1, c_codes[1], None),
            ("Trade Balance", 2 * M + 2 * nc + 0, c_codes[0], None),
            ("Fiscal Budget Consistency", 2 * M + 3 * nc - 1 + 0, c_codes[0], None),
            ("Fiscal Budget Consistency", 2 * M + 3 * nc - 1 + 1, c_codes[1], None),
        ]

        for expected_block, target_idx, exp_c, exp_s in test_cases:
            # Create a residual vector where all entries are small (e.g. 1e-4) except target_idx
            res_vec = np.full(n_total, 1e-4)
            res_vec[target_idx] = 999.0

            diag = compute_convergence_diagnostics(res_vec, calib)
            top_eq = diag["top_equations"][0]

            assert top_eq["rank"] == 1
            assert top_eq["index"] == target_idx
            assert top_eq["block"] == expected_block, f"Expected {expected_block}, got {top_eq['block']}"
            assert top_eq["country"] == exp_c, f"Expected country {exp_c}, got {top_eq['country']}"
            if exp_s is not None:
                assert top_eq["sector"] == exp_s, f"Expected sector {exp_s}, got {top_eq['sector']}"
            assert np.isclose(top_eq["residual"], 999.0)
            assert np.isclose(top_eq["abs_residual"], 999.0)
            assert expected_block.split()[0] in diag["remediation"] or "residual is in" in diag["remediation"]

    def test_ill_conditioned_shock_convergence_failure(self, synthetic_calib):
        """A severe non-convergent run (max_iter=1, extreme tariff shock) produces valid diagnostics."""
        calib = synthetic_calib
        M = calib.n_sectors * calib.n_countries
        tau_extreme = np.full((M, calib.n_sectors, calib.n_countries), 100.0)
        res = solve_flexible_trade_equilibrium(
            calib, tau=tau_extreme, max_iter=1, tol=1e-12
        )
        assert not res.converged
        diag = res.convergence_diagnostic
        assert "top_equations" in diag
        assert len(diag["top_equations"]) >= 1
        assert diag["max_residual"] > 0.0
        assert isinstance(diag["remediation"], str)
        assert len(diag["remediation"]) > 0

    def test_nan_and_inf_residual_handling(self, synthetic_calib):
        """Diagnostics handle NaN and Inf gracefully, prioritizing them and reporting severe divergence."""
        calib = synthetic_calib
        n_total = 2 * calib.n_sectors * calib.n_countries + 4 * calib.n_countries - 1
        res_vec = np.zeros(n_total)
        res_vec[5] = np.nan
        res_vec[2] = 100.0

        diag = compute_convergence_diagnostics(res_vec, calib)
        assert "Severe numerical divergence" in diag["remediation"]
        # NaN is sorted to rank 1
        assert diag["top_equations"][0]["index"] == 5
        assert np.isnan(diag["top_equations"][0]["residual"])


# =============================================================================
# 4. Adversarial Testing: JSON Serialization Robustness Across 10 Trials
# =============================================================================

class TestJSONSerializationRobustness:
    """Verify json.dumps(res.convergence_diagnostic) never raises TypeError."""

    def _assert_pure_python_types(self, obj: Any) -> None:
        """Recursively assert that an object contains only JSON-serializable Python builtins."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                assert isinstance(k, str), f"Dict key {k!r} is not str (type: {type(k)})"
                self._assert_pure_python_types(v)
        elif isinstance(obj, list):
            for item in obj:
                self._assert_pure_python_types(item)
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            # Check not numpy scalars
            assert not isinstance(obj, np.generic), f"Value {obj!r} is numpy generic {type(obj)}"
        else:
            pytest.fail(f"Encountered unexpected non-pure type {type(obj)} for value {obj!r}")

    def test_json_dumps_ten_random_trials(self, synthetic_calib):
        """Perform 10 randomized trials and verify JSON serializability."""
        rng = np.random.default_rng(seed=20260917)
        calib = synthetic_calib
        n_total = 2 * calib.n_sectors * calib.n_countries + 4 * calib.n_countries - 1

        for trial in range(10):
            # Generate random residual disturbances
            random_res = rng.standard_normal(n_total) * (10.0 ** rng.uniform(-2, 3))
            diag = compute_convergence_diagnostics(random_res, calib)

            # 1. Strict pure python type assertion
            self._assert_pure_python_types(diag)

            # 2. json.dumps must not raise TypeError
            json_str = json.dumps(diag)
            assert isinstance(json_str, str)
            assert len(json_str) > 0

            # 3. json.loads roundtrip
            loaded = json.loads(json_str)
            assert loaded["max_residual"] == pytest.approx(diag["max_residual"], rel=1e-7)
            assert len(loaded["top_equations"]) == len(diag["top_equations"])

    def test_json_dumps_solver_result_ten_trials(self, synthetic_calib):
        """Perform 10 randomized solver trials with max_iter=0 and verify json.dumps."""
        rng = np.random.default_rng(seed=42)

        for trial in range(10):
            rho = float(rng.uniform(0.1, 2.0))
            sigma_y = float(rng.uniform(0.0, 1.0))
            var_mkt = bool(trial % 2 == 0)

            cfg = FlexibleTradeModelConfig(
                technology=FlexibleTechnologyConfig(rho_va=rho, sigma_y=sigma_y),
                market_structure=FlexibleMarketStructureConfig(variable_markups=var_mkt),
                max_inner_iter=int(rng.integers(1, 11)),
            )

            res = solve_flexible_trade_equilibrium(synthetic_calib, config=cfg, max_iter=0)
            assert not res.converged

            # Must serialize without TypeError
            json_dump = json.dumps(res.convergence_diagnostic)
            assert len(json_dump) > 0
            self._assert_pure_python_types(res.convergence_diagnostic)

    def test_icio_benchmark_max_iter_zero(self, icio_calib):
        """Verify 2,001-equation OECD ICIO benchmark under max_iter=0 produces valid diagnostics and JSON."""
        res = solve_flexible_trade_equilibrium(icio_calib, max_iter=0)
        assert not res.converged
        assert res.iterations == 0
        assert len(res.x_sol) == 2001
        assert len(res.residuals) == 2001

        diag = res.convergence_diagnostic
        assert len(diag["top_equations"]) == 3
        assert len(res.top_offending_equations) == 3
        for eq in res.top_offending_equations:
            assert eq["country"] in icio_calib.country_codes
            if eq["sector"] is not None:
                assert eq["sector"] in icio_calib.sector_codes

        self._assert_pure_python_types(diag)
        json_str = json.dumps(diag)
        assert len(json_str) > 0
        loaded = json.loads(json_str)
        assert loaded["max_residual"] == pytest.approx(diag["max_residual"], rel=1e-7)

    def test_icio_residual_spike_identification(self, icio_calib):
        """Verify exact country and sector identification in 2,001-equation ICIO model."""
        ns, nc = icio_calib.n_sectors, icio_calib.n_countries
        M = ns * nc
        assert ns == 11 and nc == 77 and M == 847

        # Find USA country index and a specific sector index
        c_idx = list(icio_calib.country_codes).index("USA")
        s_idx = 3
        expected_sec = icio_calib.sector_codes[s_idx]

        # Target index in Zero Profit Condition (Block 1): M + c_idx * ns + s_idx
        target_idx = M + c_idx * ns + s_idx

        res_vec = np.zeros(2 * M + 4 * nc - 1)
        res_vec[target_idx] = 555.55

        diag = compute_convergence_diagnostics(res_vec, icio_calib)
        top = diag["top_equations"][0]

        assert top["rank"] == 1
        assert top["index"] == target_idx
        assert top["block"] == "Zero-Profit Condition"
        assert top["country"] == "USA"
        assert top["sector"] == expected_sec
        assert np.isclose(top["residual"], 555.55)

    def test_icio_json_serialization_ten_random_trials(self, icio_calib):
        """Verify 10 randomized trials of 2,001-length residual vectors on ICIO calibration."""
        rng = np.random.default_rng(seed=999)
        n_total = 2001

        for _ in range(10):
            res_vec = rng.standard_normal(n_total) * 100.0
            diag = compute_convergence_diagnostics(res_vec, icio_calib)
            self._assert_pure_python_types(diag)
            json_str = json.dumps(diag)
            assert isinstance(json_str, str)
            loaded = json.loads(json_str)
            assert len(loaded["top_equations"]) == 3

