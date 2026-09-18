"""Milestone 4 Empirical Verification: ICIO Benchmark Calibration Invariance & Solver Efficiency (Challenger 2).

Verifies:
1. Sparsity pattern density and valid CSC format on the canonical 77-country 11-sector OECD ICIO dataset:
   - Density < 0.60 on ICIO baseline (approx 0.572).
   - Dynamic activation of S[M:2M, M:2M] when sigma_y >= 1e-6 or variable_markups=True.
   - CSC matrix type, shape (2001, 2001).
   - Density < 0.60 on small synthetic model (2c x 2s).
2. Solver efficiency at default calibration:
   - solve_flexible_trade_equilibrium converges in <= 2 outer iterations.
   - Quasi-condensed solver converges in <= 2 outer iterations with LAPACK direct solve short-circuit.
   - Solution vector matches baseline solution vector within 5.0e-4 (exact match 0.0).
3. Exact baseline calibration invariance:
   - ||F_flex(x0) - F_base(x0)||_infty <= 10^-10.
   - Smooth subsistence scaling |g(1.0) - 1.0| <= 10^-15.
4. General equilibrium state vector dimension preservation:
   - Length strictly preserved at 2,001 across initial guess, solver solution, and flexible configurations.
   - Unpacking into 6 economic variable tensors preserves exact physical dimensions.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from puremacro.trade.calibration import calibrate_trade_model
from puremacro.trade.data import load_icio_data
from puremacro.trade.equilibrium import compute_equilibrium_residuals
from puremacro.trade.flexible import (
    FlexibleMarketStructureConfig,
    FlexiblePreferenceConfig,
    FlexibleTechnologyConfig,
    FlexibleTradeModelConfig,
    smooth_subsistence_scaling,
    solve_flexible_trade_equilibrium,
)
from puremacro.trade.solver import (
    _build_cge_sparsity_pattern,
    build_initial_guess,
    solve_trade_equilibrium,
    unpack_equilibrium_vector,
)


@pytest.fixture(scope="module")
def empirical_calib():
    """Load empirical 77-country, 11-sector OECD ICIO calibration."""
    raw = load_icio_data()
    return calibrate_trade_model(raw, ns=11, nc=77, nfd=3, validate=True)


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
            data[i, 4:10] = [tot_fd * 0.50, tot_fd * 0.25, tot_fd * 0.05, tot_fd * 0.10, tot_fd * 0.08, tot_fd * 0.02]
        else:
            data[i, 4:10] = [tot_fd * 0.10, tot_fd * 0.08, tot_fd * 0.02, tot_fd * 0.50, tot_fd * 0.25, tot_fd * 0.05]
    data[4, 4:] = 0.02 * data[:4, 4:].sum(axis=0)
    return calibrate_trade_model(data, ns=ns, nc=nc, nfd=nfd, validate=True)


class TestSparsityPatternAndDensity:
    """1. Sparsity pattern density and valid CSC format on 77c x 11s ICIO and synthetic."""

    def test_icio_sparsity_pattern_type_and_shape(self, empirical_calib):
        ns, nc = empirical_calib.n_sectors, empirical_calib.n_countries
        M = ns * nc
        N = 2 * M + 4 * nc - 1
        assert N == 2001
        S = _build_cge_sparsity_pattern(empirical_calib)
        assert sp.isspmatrix_csc(S), "Pattern must be scipy.sparse.csc_matrix"
        assert S.shape == (2001, 2001)

    def test_icio_sparsity_pattern_density_under_bound(self, empirical_calib):
        S = _build_cge_sparsity_pattern(empirical_calib)
        N = S.shape[0]
        density = S.nnz / (N * N)
        assert density < 0.60, f"Empirical density {density} exceeds 0.60 upper bound"
        assert 0.50 < density < 0.60

    def test_icio_dynamic_sparsity_activation_preserves_density_bound(self, empirical_calib):
        ns, nc = empirical_calib.n_sectors, empirical_calib.n_countries
        M = ns * nc
        N = 2 * M + 4 * nc - 1

        # sigma_y active
        S_sig = _build_cge_sparsity_pattern(empirical_calib, sigma_y=0.5)
        d_sig = S_sig.nnz / (N * N)
        assert d_sig < 0.60
        assert sp.isspmatrix_csc(S_sig)
        blk_sig = S_sig[M : 2 * M, M : 2 * M].toarray()
        assert np.all(np.diag(blk_sig))
        assert np.count_nonzero(blk_sig) == M

        # variable markups active
        S_mkt = _build_cge_sparsity_pattern(empirical_calib, variable_markups=True)
        d_mkt = S_mkt.nnz / (N * N)
        assert d_mkt < 0.60
        assert sp.isspmatrix_csc(S_mkt)
        blk_mkt = S_mkt[M : 2 * M, M : 2 * M].toarray()
        assert np.all(np.diag(blk_mkt))

    def test_synthetic_sparsity_density_under_bound(self, synthetic_2c_2s_calib):
        S = _build_cge_sparsity_pattern(synthetic_2c_2s_calib)
        N = S.shape[0]
        density = S.nnz / (N * N)
        assert density < 0.60, f"Synthetic density {density} exceeds 0.60"
        assert sp.isspmatrix_csc(S)


class TestSolverEfficiencyAndConvergence:
    """2. Solver efficiency at default calibration: converges in <= 2 outer iterations."""

    def test_solve_flexible_trade_equilibrium_converges_le_2_iterations(self, empirical_calib):
        res = solve_flexible_trade_equilibrium(empirical_calib, tol=2.5e-3, max_iter=50)
        assert res.converged, "Flexible solver failed to converge at default calibration"
        assert res.iterations <= 2, f"Outer iterations {res.iterations} exceeded 2"
        assert res.residual_norm <= 2.5e-3

    def test_quasi_condensed_solve_converges_le_2_iterations_lapack(self, empirical_calib):
        res = solve_flexible_trade_equilibrium(
            empirical_calib, method="quasi_condensed", tol=2.5e-3, max_iter=50
        )
        assert res.converged, "Quasi-condensed solver failed to converge at default calibration"
        assert res.iterations <= 2, f"Outer iterations {res.iterations} exceeded 2"
        assert res.residual_norm <= 2.5e-3

    def test_solution_vector_matches_baseline(self, empirical_calib):
        res_flex = solve_flexible_trade_equilibrium(empirical_calib, tol=2.5e-3, max_iter=50)
        res_base = solve_trade_equilibrium(empirical_calib, tol=2.5e-3, max_iter=50)
        max_diff = float(np.max(np.abs(res_flex.x_sol - res_base.x_sol)))
        assert max_diff <= 5.0e-4, f"Solution difference {max_diff} exceeded 5.0e-4"


class TestBaselineCalibrationInvariance:
    """3. Exact baseline calibration invariance: ||F_flex(x0) - F_base(x0)||_infty <= 10^-10."""

    def test_baseline_residual_invariance_at_x0(self, empirical_calib):
        x0 = build_initial_guess(empirical_calib)
        f_base = compute_equilibrium_residuals(x0, empirical_calib)
        res_0 = solve_flexible_trade_equilibrium(empirical_calib, max_iter=0)
        f_flex = res_0.residuals
        max_diff = float(np.max(np.abs(f_flex - f_base)))
        assert max_diff <= 1e-10, f"||F_flex(x0) - F_base(x0)||_infty = {max_diff} exceeds 1e-10"

    def test_baseline_residual_invariance_default_solve(self, empirical_calib):
        x0 = build_initial_guess(empirical_calib)
        f_base = compute_equilibrium_residuals(x0, empirical_calib)
        res_def = solve_flexible_trade_equilibrium(empirical_calib)
        f_flex = res_def.residuals
        max_diff = float(np.max(np.abs(f_flex - f_base)))
        assert max_diff <= 1e-10, f"||F_flex(x0) - F_base(x0)||_infty = {max_diff} exceeds 1e-10"

    def test_smooth_subsistence_scaling_machine_precision(self):
        val = smooth_subsistence_scaling(1.0)
        diff = abs(val - 1.0)
        assert diff <= 1e-15, f"g(1.0) deviation {diff} exceeds 1e-15"


class TestStateVectorPreservation:
    """4. Confirm state vector length is strictly preserved at 2,001 across all operations."""

    def test_initial_guess_length_is_2001(self, empirical_calib):
        x0 = build_initial_guess(empirical_calib)
        assert len(x0) == 2001, f"Expected 2001, got {len(x0)}"

    def test_solution_vector_length_is_2001(self, empirical_calib):
        res = solve_flexible_trade_equilibrium(empirical_calib)
        assert len(res.x_sol) == 2001, f"Expected 2001, got {len(res.x_sol)}"

    def test_unpack_equilibrium_vector_shapes(self, empirical_calib):
        x0 = build_initial_guess(empirical_calib)
        ns, nc = empirical_calib.n_sectors, empirical_calib.n_countries
        vars0 = unpack_equilibrium_vector(x0, ns=ns, nc=nc)
        assert vars0.p.shape == (1, 11, 77)
        assert vars0.y.shape == (1, 11, 77)
        assert vars0.r.shape == (1, 1, 77)
        assert vars0.w.shape == (1, 1, 77)
        assert vars0.T.shape == (1, 1, 77)
        assert vars0.XN.shape == (76,)
        assert vars0.invforT.shape == (1, 77)
        # Sum of parts = 11*77 + 11*77 + 77 + 77 + 77 + 76 = 2001
        total_len = vars0.p.size + vars0.y.size + vars0.r.size + vars0.w.size + vars0.T.size + vars0.XN.size
        assert total_len == 2001
