"""Comprehensive Unit, Integration, and Parity Tests for puremacro.trade general equilibrium.

Covers:
- Tier 1: Fast synthetic 2-country 2-sector closed model unit tests (< 0.1s execution).
- Tier 2: Residual verification tests asserting that xx_sol_77c_11s_base.mat and
  xx_sol_77c_11s_t10.mat yield max residuals < 2.5e-3 and price residuals < 1e-14
  under replicate_matlab_precedence=True.
- Tier 3: Solve parity tests asserting solved equilibrium reproduces
  xx_sol_77c_11s_base.mat and xx_sol_77c_11s_t10.mat within relative error < 1e-4.

Conforms strictly to the puremacro Pyodide runtime contract and has zero
machine-specific hardcoded paths.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.trade import (
    TradeCalibrationResult,
    TradeEquilibriumResult,
    build_initial_guess,
    calibrate_trade_model,
    compute_equilibrium_residuals,
    compute_postprocessing_flows,
    evaluate_equilibrium_residuals,
    pack_equilibrium_vector,
    solve_trade_equilibrium,
    unpack_equilibrium_vector,
)
from puremacro.trade.data import load_icio_data, load_reference_solution


# ---------------------------------------------------------------------------
# Bundled MATLAB reference solutions
# ---------------------------------------------------------------------------
#
# The ICIO source matrix and the reference equilibria ship inside
# ``puremacro.trade`` as verbatim copies of the MATLAB inputs and outputs, so
# these checks need neither MATLAB nor any file outside the installation.
#
# The bundle carries XN_sol, c_sol, pfd_sol, xx_sol, w_sol, ytot_sol, T_sol,
# r_sol, p_sol, tau_a and taufd_a per scenario.  It does NOT carry ``ff`` (the
# MATLAB residual vector) or the retaliatory tariff tensors ``tauf`` /
# ``tauf_fd``, so the residual- and post-processing-parity checks below skip by
# name.  Those arrays are an external check on puremacro and must not be
# regenerated with puremacro itself; bundling them alongside the others is what
# re-enables these tests.

def reference_or_skip(scenario: str, *required: str) -> dict[str, np.ndarray]:
    """Bundled reference arrays for ``scenario``, or a skip naming what is absent."""
    try:
        arrays = load_reference_solution(scenario)
    except KeyError:
        pytest.skip(f"No bundled reference solution for scenario {scenario!r}")
    missing = [name for name in required if name not in arrays]
    if missing:
        pytest.skip(
            f"Reference array(s) {', '.join(missing)} for scenario {scenario!r} are not "
            "part of the reference solutions bundled with puremacro "
            f"(bundled: {', '.join(sorted(arrays))})"
        )
    return arrays


# ---------------------------------------------------------------------------
# Fixtures
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
def empirical_calib() -> TradeCalibrationResult:
    """Calibrate full 77-country 11-sector empirical model from the bundled ICIO data."""
    return calibrate_trade_model(load_icio_data(), ns=11, nc=77, nfd=3, validate=True)


# ---------------------------------------------------------------------------
# Tier 1: Fast Synthetic 2x2 Unit Tests (< 0.1s)
# ---------------------------------------------------------------------------
class TestSyntheticEquilibrium:
    """Unit tests on synthetic 2-country 2-sector fixture."""

    def test_synthetic_initial_guess_residual(self, synthetic_2c_2s_calib: TradeCalibrationResult) -> None:
        """Verify that calibrated baseline parameters yield near-zero initial residual."""
        calib = synthetic_2c_2s_calib
        x0 = build_initial_guess(calib)
        # Length: 2*2*2 + 3*2 + (2-1) = 8 + 6 + 1 = 15
        assert len(x0) == 15

        res0 = compute_equilibrium_residuals(x0, calib)
        assert len(res0) == 15
        assert np.max(np.abs(res0)) < 1e-10

    def test_synthetic_newton_solve_baseline(self, synthetic_2c_2s_calib: TradeCalibrationResult) -> None:
        """Verify that pure NumPy Newton converges immediately on baseline."""
        calib = synthetic_2c_2s_calib
        res = solve_trade_equilibrium(calib, method="newton", tol=2.5e-3)
        assert res.converged is True
        assert res.iterations <= 1
        assert res.max_residual < 1e-10
        assert np.allclose(res.p_sol, 1.0, atol=1e-10)
        assert np.allclose(res.w_sol, 1.0, atol=1e-10)
        assert np.allclose(res.r_sol, 1.0, atol=1e-10)
        assert res.intermediate_flows is not None
        assert res.final_demand_flows is not None

    def test_synthetic_broyden_solve_baseline(self, synthetic_2c_2s_calib: TradeCalibrationResult) -> None:
        """Verify that pure NumPy Broyden converges on baseline."""
        calib = synthetic_2c_2s_calib
        res = solve_trade_equilibrium(calib, method="broyden", tol=2.5e-3)
        assert res.converged is True
        assert res.max_residual < 1e-10

    def test_synthetic_scipy_hybr_baseline(self, synthetic_2c_2s_calib: TradeCalibrationResult) -> None:
        """Verify that SciPy hybr wrapper converges on baseline."""
        calib = synthetic_2c_2s_calib
        res = solve_trade_equilibrium(calib, method="hybr", tol=1e-6)
        assert res.converged is True
        assert res.max_residual < 1e-8

    def test_synthetic_scipy_lm_baseline(self, synthetic_2c_2s_calib: TradeCalibrationResult) -> None:
        """Verify that SciPy Levenberg-Marquardt wrapper converges on baseline."""
        calib = synthetic_2c_2s_calib
        res = solve_trade_equilibrium(calib, method="lm", tol=1e-6)
        assert res.converged is True
        assert res.max_residual < 1e-8

    def test_synthetic_newton_and_hybr_parity_under_shock(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """Verify that Newton and SciPy hybr converge to identical solutions under 10% tariff."""
        calib = synthetic_2c_2s_calib
        nc, ns = calib.n_countries, calib.n_sectors

        # 10% tariff on country 0 imports from country 1
        tau = np.ones((ns * nc, ns, nc), dtype=float)
        tau_fd = np.ones((ns * nc, calib.n_final_demand, nc), dtype=float)
        tau[2:, :, 0] = 1.10
        tau_fd[2:, :, 0] = 1.10

        res_newt = solve_trade_equilibrium(calib, tau=tau, tau_fd=tau_fd, method="newton", tol=2.5e-3)
        assert res_newt.converged is True
        assert res_newt.max_residual < 2.5e-3
        assert res_newt.iterations < 10

        res_hybr = solve_trade_equilibrium(calib, tau=tau, tau_fd=tau_fd, method="hybr", tol=1e-6)
        assert res_hybr.converged is True

        # Relative discrepancy between Newton and SciPy solutions < 1e-4
        max_rel_diff = np.max(np.abs(res_newt.x_sol - res_hybr.x_sol) / (np.abs(res_hybr.x_sol) + 1.0))
        assert max_rel_diff < 1e-4

    def test_operator_precedence_toggle(self, synthetic_2c_2s_calib: TradeCalibrationResult) -> None:
        """Verify replicate_matlab_precedence toggle propagates correctly."""
        calib = synthetic_2c_2s_calib
        res_true = solve_trade_equilibrium(calib, replicate_matlab_precedence=True)
        res_false = solve_trade_equilibrium(calib, replicate_matlab_precedence=False)
        assert res_true.metadata["replicate_matlab_precedence"] is True
        assert res_false.metadata["replicate_matlab_precedence"] is False

    def test_packing_unpacking_roundtrip(self, synthetic_2c_2s_calib: TradeCalibrationResult) -> None:
        """Verify pack_equilibrium_vector and unpack_equilibrium_vector roundtrip."""
        calib = synthetic_2c_2s_calib
        nc, ns, nfd = calib.n_countries, calib.n_sectors, calib.n_final_demand
        x0 = build_initial_guess(calib)

        unpacked = unpack_equilibrium_vector(x0, ns=ns, nc=nc, nfd=nfd, return_levels=True)
        assert unpacked.p.shape == (1, ns, nc)
        assert unpacked.y.shape == (1, ns, nc)
        assert unpacked.r.shape == (1, 1, nc)
        assert unpacked.w.shape == (1, 1, nc)
        assert unpacked.T.shape == (1, 1, nc)
        assert unpacked.XN.shape == (nc - 1,)
        assert unpacked.invforT.shape == (1, nc)

        # Repack
        repacked = pack_equilibrium_vector(
            p=unpacked.p,
            y=unpacked.y,
            r=unpacked.r,
            w=unpacked.w,
            T=unpacked.T,
            XN=unpacked.XN,
            log_transformed=True,
        )
        assert np.allclose(x0, repacked, atol=1e-12)

    def test_postprocessing_synthetic(self, synthetic_2c_2s_calib: TradeCalibrationResult) -> None:
        """Verify post-processing reconstruction on synthetic model."""
        calib = synthetic_2c_2s_calib
        x0 = build_initial_guess(calib)
        flows = compute_postprocessing_flows(x0, calib)

        assert flows["intermediate_flows"].shape == (2, 2, 2, 2)
        assert flows["intermediate_matrix"].shape == (4, 4)
        assert flows["final_demand_flows"].shape == (2, 2, 3, 2)
        assert flows["final_demand_matrix"].shape == (4, 6)
        assert flows["gdp"].shape == (2,)
        assert flows["gdp_fc"].shape == (2,)
        assert flows["exports"].shape == (2,)
        assert flows["imports"].shape == (2,)
        assert flows["tariffs"].shape == (2,)
        assert flows["cpi"].shape == (2,)
        assert flows["terms_of_trade"].shape == (2,)

    def test_trade_equilibrium_result_presentation(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """Verify presentation methods (.summary, .to_frame, .to_markdown, .to_latex)."""
        calib = synthetic_2c_2s_calib
        res = solve_trade_equilibrium(calib, method="newton")

        # Default summary table
        df_diag = res.summary(detailed=False)
        assert isinstance(df_diag, pd.DataFrame)
        assert "CONVERGED" in res.to_markdown(detailed=False)
        assert "\\begin{tabular}" in res.to_latex(detailed=False)

        # Detailed summary table
        df_detail = res.summary(detailed=True)
        assert isinstance(df_detail, pd.DataFrame)
        assert "GDP_Market_Prices" in df_detail.columns
        assert "Exports" in df_detail.columns
        assert "Imports" in df_detail.columns

        # Frame and table emitters
        assert isinstance(res.bilateral_trade_frame(), pd.DataFrame)
        assert isinstance(res.sectoral_tariffs_frame(), pd.DataFrame)


# ---------------------------------------------------------------------------
# Tier 2: Residual Validation on MATLAB Reference Solutions
# ---------------------------------------------------------------------------
class TestResidualValidation:
    """Validate that MATLAB .mat solution vectors yield near-zero residuals in Python."""

    def test_baseline_residual_parity(
        self, empirical_calib: TradeCalibrationResult
    ) -> None:
        """Verify xx_sol_77c_11s_base.mat evaluated in Python satisfies tolerance < 2.5e-3."""
        calib = empirical_calib
        mat = reference_or_skip("base", "ff", "tau_a", "taufd_a", "tauf", "tauf_fd", "xx_sol")
        xx_mat = mat["xx_sol"].ravel()
        ff_mat = mat["ff"].ravel()

        ff_py = compute_equilibrium_residuals(
            xx_mat,
            calib,
            tau=mat["tau_a"],
            tau_fd=mat["taufd_a"],
            tauf=mat["tauf"],
            tauf_fd=mat["tauf_fd"],
            replicate_matlab_precedence=True,
        )

        assert len(ff_py) == 2001
        # Max discrepancy vs MATLAB ff is < 1e-4 (dominated by machine precision on 10^11 factor endowments)
        assert np.max(np.abs(ff_py - ff_mat)) < 1e-4

        # Zero-profit price residual (pp - p) machine precision < 1e-14
        ff1_py = ff_py[847:1694]
        assert np.max(np.abs(ff1_py)) < 1e-14

        # Absolute and L1 residual norms satisfy convergence tolerance < 2.5e-3
        assert np.max(np.abs(ff_py)) < 2.5e-3
        assert np.sum(np.abs(ff_py)) < 2.5e-3

    def test_t10_residual_parity(
        self, empirical_calib: TradeCalibrationResult
    ) -> None:
        """Verify xx_sol_77c_11s_t10.mat evaluated in Python satisfies tolerance < 2.5e-3."""
        calib = empirical_calib
        mat = reference_or_skip("t10", "ff", "tau_a", "taufd_a", "tauf", "tauf_fd", "xx_sol")
        xx_mat = mat["xx_sol"].ravel()
        ff_mat = mat["ff"].ravel()

        ff_py = compute_equilibrium_residuals(
            xx_mat,
            calib,
            tau=mat["tau_a"],
            tau_fd=mat["taufd_a"],
            tauf=mat["tauf"],
            tauf_fd=mat["tauf_fd"],
            replicate_matlab_precedence=True,
        )

        assert len(ff_py) == 2001
        assert np.max(np.abs(ff_py - ff_mat)) < 1e-4

        # Price residual satisfies < 1e-14 under replicate_matlab_precedence=True
        ff1_py = ff_py[847:1694]
        assert np.max(np.abs(ff1_py)) < 1e-14

        assert np.max(np.abs(ff_py)) < 2.5e-3
        assert np.sum(np.abs(ff_py)) < 2.5e-3

    def test_t10_precedence_divergence(
        self, empirical_calib: TradeCalibrationResult
    ) -> None:
        """Verify that replicate_matlab_precedence=False diverges from MATLAB solution."""
        calib = empirical_calib
        mat = reference_or_skip("t10", "tau_a", "taufd_a", "tauf", "tauf_fd", "xx_sol")
        xx_mat = mat["xx_sol"].ravel()

        ff_false = compute_equilibrium_residuals(
            xx_mat,
            calib,
            tau=mat["tau_a"],
            tau_fd=mat["taufd_a"],
            tauf=mat["tauf"],
            tauf_fd=mat["tauf_fd"],
            replicate_matlab_precedence=False,
        )
        ff1_false = ff_false[847:1694]
        # Price residual diverges to > 0.02, exceeding tolerance
        assert np.max(np.abs(ff1_false)) > 0.02


# ---------------------------------------------------------------------------
# Tier 3: Solve Parity Tests (< 10^-4 Parity with MATLAB Solutions)
# ---------------------------------------------------------------------------
class TestSolveParity:
    """Verify solve_trade_equilibrium reproduces MATLAB benchmark solutions within 10^-4."""

    def test_solve_baseline_parity(
        self, empirical_calib: TradeCalibrationResult
    ) -> None:
        """Verify baseline solve reproduces xx_sol_77c_11s_base.mat within relative error < 1e-4."""
        calib = empirical_calib
        mat = reference_or_skip("base", "xx_sol")
        xx_expected = mat["xx_sol"].ravel()

        res = solve_trade_equilibrium(calib, method="newton", tol=2.5e-3)
        assert res.converged is True
        assert res.iterations <= 1
        assert res.max_residual < 2.5e-3

        max_rel_error = np.max(np.abs(res.x_sol - xx_expected) / (np.abs(xx_expected) + 1.0))
        assert max_rel_error < 1e-4

        assert np.allclose(res.p_sol, 1.0, atol=1e-10)
        assert np.allclose(res.w_sol, 1.0, atol=1e-10)
        assert np.allclose(res.r_sol, 1.0, atol=1e-10)

    def test_solve_t10_warm_start_parity(
        self, empirical_calib: TradeCalibrationResult
    ) -> None:
        """Verify t10 solve starting from xx_sol_77c_11s_t10 confirms convergence < 1e-4."""
        calib = empirical_calib
        mat = reference_or_skip("t10", "tau_a", "taufd_a", "tauf", "tauf_fd", "xx_sol")
        xx_expected = mat["xx_sol"].ravel()

        res = solve_trade_equilibrium(
            calib,
            tau=mat["tau_a"],
            tau_fd=mat["taufd_a"],
            tauf=mat["tauf"],
            tauf_fd=mat["tauf_fd"],
            x0=xx_expected,
            method="newton",
            tol=2.5e-3,
        )
        assert res.converged is True
        assert res.max_residual < 2.5e-3

        max_rel_error = np.max(np.abs(res.x_sol - xx_expected) / (np.abs(xx_expected) + 1.0))
        assert max_rel_error < 1e-4

    def test_postprocessing_t10_parity(
        self, empirical_calib: TradeCalibrationResult
    ) -> None:
        """Verify post-processing reproduces US tariff collections and terms of trade."""
        calib = empirical_calib
        mat = reference_or_skip("t10", "tau_a", "taufd_a", "tauf", "tauf_fd", "xx_sol")
        xx_sol = mat["xx_sol"].ravel()

        flows = compute_postprocessing_flows(
            x_sol=xx_sol,
            calib=calib,
            tau=mat["tau_a"],
            tau_fd=mat["taufd_a"],
            tauf=mat["tauf"],
            tauf_fd=mat["tauf_fd"],
            replicate_matlab_precedence=True,
        )

        # US (country index 73) tariff collections total $2,221,703,028.20
        us_tariffs = flows["tariffs"][73]
        expected_us_tariffs = 2221703028.2023954
        assert abs(us_tariffs - expected_us_tariffs) / expected_us_tariffs < 1e-4

        # Non-US tariffs are 0.0
        non_us_tariffs = np.delete(flows["tariffs"], 73)
        assert np.max(np.abs(non_us_tariffs)) < 1e-6

        # US terms of trade drops to 0.9451 (5.49% deterioration)
        us_tot = flows["terms_of_trade"][73]
        assert abs(us_tot - 0.94509759) < 1e-4
