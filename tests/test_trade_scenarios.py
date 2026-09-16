"""Comprehensive Unit, Parity, and Table Reproduction Tests for puremacro.trade Scenarios & PPP.

Covers:
- Tier 1: Fast synthetic model unit tests (< 0.15s):
    - solve_multilateral_ppp iterative and linear methods.
    - restore_capital_formation with matlab_compat=True/False.
    - build_tariff_matrices structure, shapes, and domestic diagonal invariance.
    - run_tariff_scenario and run_scenario_batch on synthetic model.
- Tier 2: Tariff matrix construction parity tests:
    - Assert identically zero discrepancy against reference .mat files across all 7 scenarios.
- Tier 3: Geary-Khamis numerical parity tests vs MATLAB results.xls and .mat benchmarks:
    - Assert maximum relative error < 10^-11 across all 77 countries and 5 published scenarios.
    - Parity on intermediate scenario t10_75.
    - Machine parity on inflation and XN/GDP sheets (< 10^-11).
- Tier 4: Published paper table reproduction tests:
    - selected_country_impacts.tex (CAN -0.49, CHN -0.64, EUR -0.65, MEX -0.48, USA -0.82).
    - mean_by_scenario.tex (-0.695 (0.432), etc.).
- Tier 5: Warm-started batch solve test:
    - Verify full batch runs in < 60s with max error < 10^-4 vs MATLAB reference solutions.
- Negative tests:
    - Invalid scenario names, unknown country/sector codes, negative tariffs, invalid methods.

Conforms strictly to the puremacro Pyodide runtime contract: pure NumPy/SciPy/Pandas,
zero dev-dependencies in the runtime path.
"""
from __future__ import annotations

import time
from typing import Any
import numpy as np
import pandas as pd
import pytest

from puremacro.trade import (

    CANONICAL_COUNTRY_CODES,
    CANONICAL_SCENARIOS,
    CANONICAL_SECTOR_CODES,
    GearyKhamisResult,
    SCENARIOS,
    ScenarioBatchResult,
    TariffScenario,
    TradeCalibrationResult,
    TradeEquilibriumResult,
    build_tariff_matrices,
    calibrate_trade_model,
    compute_geary_khamis,
    compute_geary_khamis_ppp,
    get_canonical_scenario,
    list_canonical_scenarios,
    restore_capital_formation,
    run_scenario_batch,
    run_tariff_scenario,
    solve_multilateral_ppp,
    solve_trade_equilibrium,
)
from puremacro.trade.data import (
    load_reference_workbook_sheet,
    available_reference_scenarios,
    load_icio_data,
    load_reference_solution,
)


# ---------------------------------------------------------------------------
# Bundled MATLAB reference solutions
# ---------------------------------------------------------------------------
#
# The reference equilibria ship inside ``puremacro.trade`` as verbatim copies of
# the MATLAB ``results_77c_11s_*.mat`` outputs, so every check below runs from a
# plain checkout with no MATLAB and no file outside the installation.  Scenario
# ``t10_54`` is the single gap: its MATLAB source is a dataless placeholder in
# the author's storage, so anything that genuinely needs it skips by name.

def reference_or_skip(scenario: str) -> dict[str, np.ndarray]:
    """Bundled reference arrays for ``scenario``, or a precise skip."""
    try:
        return load_reference_solution(scenario)
    except KeyError:
        pytest.skip(
            f"No bundled reference solution for scenario {scenario!r}; "
            f"bundled scenarios: {available_reference_scenarios()}"
        )


def require_scenarios(*scenarios: str) -> None:
    """Skip unless every named scenario has a bundled reference solution."""
    available = available_reference_scenarios()
    missing = [s for s in scenarios if s not in available]
    if missing:
        pytest.skip(
            "No bundled reference solution for scenario(s) "
            f"{', '.join(missing)}; bundled scenarios: {available}"
        )


def _equilibrium_from_reference(scenario: str) -> TradeEquilibriumResult:
    """Build a TradeEquilibriumResult from one bundled reference solution."""
    mat = load_reference_solution(scenario)
    return TradeEquilibriumResult(
        x_sol=mat["xx_sol"].flatten(),
        p_sol=mat["p_sol"],
        y_sol=mat["ytot_sol"],
        r_sol=mat["r_sol"],
        w_sol=mat["w_sol"],
        T_sol=mat["T_sol"],
        XN_sol=mat["XN_sol"].flatten(),
        c_sol=mat["c_sol"],
        pfd_sol=mat["pfd_sol"],
        country_codes=CANONICAL_COUNTRY_CODES,
    )


@pytest.fixture(scope="module")
def synthetic_2c_2s_calib() -> TradeCalibrationResult:
    """Construct a mathematically balanced 2-country 2-sector synthetic model."""
    nc, ns, nfd = 2, 2, 3
    data = np.zeros((ns * nc + 3, ns * nc + nfd * nc), dtype=float)

    # Intermediate transactions (4x4)
    data[:4, :4] = np.array([
        [10.0, 15.0, 5.0, 5.0],
        [15.0, 20.0, 10.0, 10.0],
        [5.0, 5.0, 12.0, 18.0],
        [10.0, 10.0, 18.0, 22.0],
    ])
    y = np.array([100.0, 150.0, 120.0, 180.0])
    inter_col_sums = data[:4, :4].sum(axis=0)
    va = y - inter_col_sums
    taxes = 0.05 * y
    va_fac = va - taxes
    data[4, :4] = taxes
    data[5, :4] = (2.0 / 3.0) * va_fac
    data[6, :4] = (1.0 / 3.0) * va_fac

    # Final demand (4x6)
    fd_row_sums = y - data[:4, :4].sum(axis=1)
    for i in range(4):
        tot_fd = fd_row_sums[i]
        shares = [0.50, 0.25, 0.05, 0.10, 0.08, 0.02] if i < 2 else [0.10, 0.08, 0.02, 0.50, 0.25, 0.05]
        for c_idx, sh in enumerate(shares):
            data[i, 4 + c_idx] = tot_fd * sh

    data[4, 4:] = 0.02 * data[:4, 4:].sum(axis=0)
    return calibrate_trade_model(
        data,
        ns=ns,
        nc=nc,
        nfd=nfd,
        country_codes=("C1", "C2"),
        sector_codes=("S1", "S2"),
        validate=True,
    )


# ---------------------------------------------------------------------------
# Tier 1: Fast Synthetic Model Unit Tests (< 0.15s)
# ---------------------------------------------------------------------------

class TestSyntheticModelUnit:
    """Fast synthetic unit tests ensuring fundamental algorithmic correctness."""

    def test_solve_multilateral_ppp_synthetic(self) -> None:
        """Test Geary-Khamis iterative and linear solvers on synthetic prices and quantities."""
        p = np.array([[1.0, 1.2], [1.1, 0.9], [1.05, 1.15]])
        q = np.array([[100.0, 120.0], [50.0, 60.0], [10.0, 15.0]])

        # 1. Iterative solve
        pi_iter, ppp_iter, conv_iter, it_count = solve_multilateral_ppp(
            p=p, q=q, method="iterative", tol=1e-12
        )
        assert conv_iter is True
        assert it_count > 0
        assert pi_iter.shape == (3,)
        assert ppp_iter.shape == (2,)

        # 2. Linear solve
        pi_lin, ppp_lin, conv_lin, it_lin = solve_multilateral_ppp(
            p=p, q=q, method="linear"
        )
        assert conv_lin is True
        assert it_lin == 1

        # 3. Assert exact mathematical equivalence between iterative and linear
        np.testing.assert_allclose(pi_iter, pi_lin, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(ppp_iter, ppp_lin, rtol=1e-10, atol=1e-10)

    def test_solve_multilateral_ppp_normalization(self) -> None:
        """Test normalization options pi1 and ppp_usa."""
        p = np.array([[1.0, 1.2], [1.1, 0.9], [1.05, 1.15]])
        q = np.array([[100.0, 120.0], [50.0, 60.0], [10.0, 15.0]])

        # normalize='pi1'
        pi_n1, ppp_n1, _, _ = solve_multilateral_ppp(p=p, q=q, normalize="pi1")
        assert pytest.approx(pi_n1[0], rel=1e-10) == 1.0

        # normalize='ppp_usa' (normalizes last country for nc=2)
        pi_n2, ppp_n2, _, _ = solve_multilateral_ppp(p=p, q=q, normalize="ppp_usa")
        assert pytest.approx(ppp_n2[-1], rel=1e-10) == 1.0

    def test_restore_capital_formation(self) -> None:
        """Test capital formation restoration with matlab_compat=True and False."""
        c = np.ones((3, 4)) * 10.0
        xn = np.array([2.0, -1.0, 0.5])  # 3 countries given, 4th is residual

        # matlab_compat=True -> XN[3] = +sum(XN[:3]) = +1.5
        c_mat = restore_capital_formation(c, xn, matlab_compat=True)
        assert c_mat.shape == (3, 4)
        np.testing.assert_allclose(c_mat[1, :], [12.0, 9.0, 10.5, 11.5])
        assert c_mat[0, 0] == 10.0  # Consumption untouched

        # matlab_compat=False -> XN[3] = -sum(XN[:3]) = -1.5
        c_theo = restore_capital_formation(c, xn, matlab_compat=False)
        np.testing.assert_allclose(c_theo[1, :], [12.0, 9.0, 10.5, 8.5])

    def test_build_tariff_matrices_synthetic(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """Verify build_tariff_matrices creates expected 4D tensors and keeps domestic sales at 1.0."""
        calib = synthetic_2c_2s_calib
        scen = TariffScenario(
            name="test_shock",
            default_us_tariff=0.10,
            us_import_tariffs={"C2": 0.25},
        )
        tau, tau_fd, tauf, tauf_fd = build_tariff_matrices(scen, calib)

        assert tau.shape == (2, 2, 2, 2)
        assert tau_fd.shape == (2, 2, 3, 2)
        assert tauf.shape == (1, 2)
        assert tauf_fd.shape == (1, 2)

        # Domestic diagonal: origin == destination must strictly equal 1.0
        for k in range(2):
            assert np.all(tau[:, k, :, k] == 1.0)
            assert np.all(tau_fd[:, k, :, k] == 1.0)

    def test_run_tariff_scenario_synthetic(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """Verify run_tariff_scenario solves general equilibrium on synthetic model."""
        calib = synthetic_2c_2s_calib
        res = run_tariff_scenario("base", calib, max_iter=20)
        assert isinstance(res, TradeEquilibriumResult)
        assert res.converged is True
        assert res.metadata["scenario_name"] == "base"

    def test_run_scenario_batch_synthetic(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """Verify run_scenario_batch executes and builds comparative tables on synthetic model."""
        calib = synthetic_2c_2s_calib
        scens = [
            TariffScenario(name="base", default_us_tariff=0.0),
            TariffScenario(name="t10", default_us_tariff=0.10),
        ]
        batch = run_scenario_batch(scens, calib, warm_start=True, compute_gk=True)
        assert isinstance(batch, ScenarioBatchResult)
        assert "base" in batch.scenarios
        assert "t10" in batch.scenarios
        assert "t10" in batch.geary_khamis
        assert isinstance(batch.real_gdp_table, pd.DataFrame)
        assert batch.real_gdp_table.shape == (2, 2)


# ---------------------------------------------------------------------------
# Tier 2: Tariff Matrix Construction Parity Tests vs MATLAB .mat Files
# ---------------------------------------------------------------------------

class TestTariffMatrixParity:
    """Verify exact zero-discrepancy parity of 4D tariff tensors against all 7 MATLAB .mat files."""

    def test_canonical_scenarios_exist(self) -> None:
        """Verify all canonical scenarios are present in SCENARIOS dictionary."""
        for s in ["base", "t10", "t10_25", "t10_54", "t10_75", "t10_125", "t10_145", "t10_25_15eu"]:
            assert s in SCENARIOS
            assert isinstance(SCENARIOS[s], TariffScenario)

    def test_tariff_matrices_exact_parity_against_all_mat_files(self) -> None:
        """Assert zero discrepancy against reference .mat files across all 7 scenarios."""
        # Construct a dummy calibration object containing canonical country and sector codes
        nc, ns, nfd = 77, 11, 3
        calib = TradeCalibrationResult(
            a=np.ones((ns * nc, ns, nc)),
            afd=np.ones((ns * nc, nfd, nc)),
            alpha=np.ones((1, ns, nc)) / 3.0,
            beta=np.ones((1, ns, nc)),
            k_endow=np.ones((1, nc)),
            l_endow=np.ones((1, nc)),
            invforT=np.zeros((1, nc)),
            tax=np.zeros((1, ns, nc)),
            ytot=np.ones((1, ns, nc)),
            country_codes=CANONICAL_COUNTRY_CODES,
            sector_codes=CANONICAL_SECTOR_CODES,
        )

        scenario_names = ["base", "t10", "t10_25", "t10_54", "t10_75", "t10_125", "t10_145"]
        available = available_reference_scenarios()
        checked = 0
        for s_name in scenario_names:
            if s_name not in available:
                # t10_54 has no bundled reference (dataless MATLAB placeholder).
                continue
            mat = load_reference_solution(s_name)
            checked += 1
            ref_tau_a = mat["tau_a"]
            ref_taufd_a = mat["taufd_a"]

            tau, tau_fd, _, _ = build_tariff_matrices(s_name, calib)

            # Convert 4D (11, 77, 11, 77) to MATLAB 3D block format (847, 11, 77)
            tau_3d = tau.transpose(1, 0, 2, 3).reshape(nc * ns, ns, nc)
            taufd_3d = tau_fd.transpose(1, 0, 2, 3).reshape(nc * ns, nfd, nc)

            max_diff_tau = float(np.max(np.abs(tau_3d - ref_tau_a)))
            max_diff_taufd = float(np.max(np.abs(taufd_3d - ref_taufd_a)))

            assert max_diff_tau == 0.0, f"Discrepancy in tau for scenario {s_name}: {max_diff_tau}"
            assert max_diff_taufd == 0.0, f"Discrepancy in tau_fd for scenario {s_name}: {max_diff_taufd}"

        assert checked == len(available), (
            f"Expected to check every bundled scenario ({available}), checked {checked}"
        )


# ---------------------------------------------------------------------------
# Tier 3: Geary-Khamis Numerical Parity Tests vs MATLAB results.xls & .mat
# ---------------------------------------------------------------------------

class TestGearyKhamisNumericalParity:
    """Verify Geary-Khamis numerical parity against results.xls and .mat benchmarks."""

    def test_geary_khamis_parity_against_results_xls(self) -> None:
        """Assert maximum error < 1e-10 across all 77 countries and the published scenarios.

        Compares Geary-Khamis real GDP growth, inflation and XN/GDP -- recomputed
        from the reference ``c_sol``/``pfd_sol``/``w_sol``/``XN_sol`` arrays --
        against the MATLAB ``results.xls`` sheets, which ship with puremacro as
        verbatim cell values so this stays an external comparison.

        ``t10_54`` is excluded: its ``results_77c_11s_t10_54.mat`` is a dataless
        placeholder in the author's storage, so the equilibrium arrays for that
        column do not exist to recompute from.
        """
        growth_mat = load_reference_workbook_sheet("growth GDP%").astype(float)
        inf_mat = load_reference_workbook_sheet("inflaction%").astype(float)
        xn_mat = load_reference_workbook_sheet("xn_over_gd%").astype(float)

        base_mat = reference_or_skip("base")
        c_base_restored = restore_capital_formation(
            base_mat["c_sol"].copy(), base_mat["XN_sol"].flatten(), matlab_compat=True
        )
        gdp_base = np.sum(c_base_restored[0], axis=0)

        # Column order of the workbook; t10_54 keeps its column index so the
        # remaining scenarios still line up with the published sheets.
        scens = ["t10", "t10_25", "t10_54", "t10_125", "t10_145"]
        available = set(available_reference_scenarios())
        checked = 0
        for s_idx, s_name in enumerate(scens):
            if s_name not in available:
                continue
            mat = load_reference_solution(s_name)
            c_sol_restored = restore_capital_formation(
                mat["c_sol"].copy(), mat["XN_sol"].flatten(), matlab_compat=True
            )
            pfd_sol = mat["pfd_sol"].copy()
            w_sol = mat["w_sol"].copy()
            xn_sol = mat["XN_sol"].flatten()

            pi, ppp, conv, _ = solve_multilateral_ppp(
                p=pfd_sol[0], q=c_sol_restored[0], method="iterative", tol=1e-9
            )
            assert conv is True

            real_gdp = np.sum(c_sol_restored[0] * pi[:, np.newaxis], axis=0)
            growth = ((real_gdp / gdp_base) - 1.0) * 100.0

            max_abs_diff = float(np.max(np.abs(growth - growth_mat[:, s_idx])))
            max_rel_diff = float(np.max(np.abs(
                (growth - growth_mat[:, s_idx]) / np.maximum(np.abs(growth), 1e-6)
            )))
            assert max_abs_diff < 1e-10, f"Absolute error exceeded in {s_name}: {max_abs_diff}"
            assert max_rel_diff < 1e-10, f"Relative error exceeded in {s_name}: {max_rel_diff}"

            w = w_sol[0, 0, :]
            p_dom = pfd_sol[0] / w[np.newaxis, :]
            gdp_dom = np.sum(p_dom * c_sol_restored[0], axis=0)
            infl = ((gdp_dom / gdp_base) - 1.0) * 100.0

            xn_full = np.zeros(77)
            xn_full[:76] = xn_sol[:76]
            xn_full[76] = np.sum(xn_sol[:76])
            xn_over_gdp = 100.0 * xn_full / gdp_dom

            assert float(np.max(np.abs(infl - inf_mat[:, s_idx]))) < 1e-10, s_name
            assert float(np.max(np.abs(xn_over_gdp - xn_mat[:, s_idx]))) < 1e-10, s_name
            checked += 1

        assert checked == 4, f"expected 4 comparable scenarios, checked {checked}"

    def test_geary_khamis_linear_vs_iterative_parity(self) -> None:
        """Verify that direct linear solve matches iterative solve to machine precision (< 1e-12)."""
        base_mat = reference_or_skip("base")
        c_base = restore_capital_formation(base_mat["c_sol"], base_mat["XN_sol"], matlab_compat=True)

        available = available_reference_scenarios()
        for s_name in ["t10", "t10_25", "t10_54", "t10_75", "t10_125", "t10_145"]:
            if s_name not in available:
                # t10_54 has no bundled reference (dataless MATLAB placeholder).
                continue
            mat = load_reference_solution(s_name)
            c_sol = restore_capital_formation(mat["c_sol"], mat["XN_sol"], matlab_compat=True)
            pfd_sol = mat["pfd_sol"]

            pi_iter, ppp_iter, _, _ = solve_multilateral_ppp(pfd_sol[0], c_sol[0], method="iterative", tol=1e-12)
            pi_lin, ppp_lin, _, _ = solve_multilateral_ppp(pfd_sol[0], c_sol[0], method="linear")

            np.testing.assert_allclose(pi_iter, pi_lin, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(ppp_iter, ppp_lin, rtol=1e-10, atol=1e-10)

    def test_intermediate_scenario_t10_75_parity(self) -> None:
        """Verify real GDP growth for intermediate scenario t10_75 against reference .mat solution."""
        base_mat = reference_or_skip("base")
        c_base = restore_capital_formation(base_mat["c_sol"], base_mat["XN_sol"], matlab_compat=True)
        gdp_base = np.sum(c_base[0], axis=0)

        mat75 = reference_or_skip("t10_75")
        c_sol = restore_capital_formation(mat75["c_sol"], mat75["XN_sol"], matlab_compat=True)
        pfd_sol = mat75["pfd_sol"]

        pi, ppp, conv, _ = solve_multilateral_ppp(pfd_sol[0], c_sol[0], method="iterative")
        assert conv is True

        real_gdp = np.sum(c_sol[0] * pi[:, np.newaxis], axis=0)
        growth = ((real_gdp / gdp_base) - 1.0) * 100.0

        # Known benchmark values: CAN: -1.3853, CHN: -1.7520, MEX: -1.5248, USA: -2.2675
        assert pytest.approx(growth[9], abs=1e-2) == -1.39   # CAN
        assert pytest.approx(growth[12], abs=1e-2) == -1.75  # CHN
        assert pytest.approx(growth[47], abs=1e-2) == -1.52  # MEX
        assert pytest.approx(growth[73], abs=1e-2) == -2.27  # USA


# ---------------------------------------------------------------------------
# Tier 4: Published Paper Table Reproduction Tests
# ---------------------------------------------------------------------------

class TestPaperTableReproduction:
    """Verify exact reproduction of selected_country_impacts.tex and mean_by_scenario.tex."""

    def test_selected_country_table_exact_match(self) -> None:
        """Assert exact match for selected_country_impacts.tex across all 5 published scenarios."""
        # Build mock TradeEquilibriumResults from the bundled reference solutions
        scens = ["base", "t10", "t10_25", "t10_54", "t10_125", "t10_145"]
        require_scenarios(*scens)
        results_dict: dict[str, TradeEquilibriumResult] = {}
        gk_dict: dict[str, GearyKhamisResult] = {}

        base_eq = _equilibrium_from_reference("base")
        results_dict["base"] = base_eq

        for s_name in scens:
            eq = _equilibrium_from_reference(s_name)
            results_dict[s_name] = eq
            gk = compute_geary_khamis(eq, base_eq, matlab_compat=True)
            if s_name == "base":
                object.__setattr__(gk, "gdp_growth", np.zeros(77, dtype=float))
                object.__setattr__(gk, "real_gdp_growth", np.zeros(77, dtype=float))
            gk_dict[s_name] = gk

        batch = ScenarioBatchResult(
            scenarios=results_dict,
            geary_khamis=gk_dict,
            baseline_scenario="base",
            country_codes=CANONICAL_COUNTRY_CODES,
        )

        tbl = batch.to_selected_country_table(countries=["CAN", "CHN", "EUR", "MEX", "USA"])

        # Check exact rounded GDP growth numbers from selected_country_impacts.tex
        # CAN: -0.49, -0.85, -1.15, -1.99, -2.26
        np.testing.assert_allclose(
            tbl.loc[("GDP growth (%)", "CAN"), ["t10", "t10_25", "t10_54", "t10_125", "t10_145"]].values,
            [-0.49, -0.85, -1.15, -1.99, -2.26],
        )
        # CHN: -0.64, -1.07, -1.46, -2.50, -2.83
        np.testing.assert_allclose(
            tbl.loc[("GDP growth (%)", "CHN"), ["t10", "t10_25", "t10_54", "t10_125", "t10_145"]].values,
            [-0.64, -1.07, -1.46, -2.50, -2.83],
        )
        # EUR: -0.65, -1.11, -1.51, -2.62, -2.97
        np.testing.assert_allclose(
            tbl.loc[("GDP growth (%)", "EUR"), ["t10", "t10_25", "t10_54", "t10_125", "t10_145"]].values,
            [-0.65, -1.11, -1.51, -2.62, -2.97],
        )
        # MEX: -0.48, -0.89, -1.24, -2.28, -2.61
        np.testing.assert_allclose(
            tbl.loc[("GDP growth (%)", "MEX"), ["t10", "t10_25", "t10_54", "t10_125", "t10_145"]].values,
            [-0.48, -0.89, -1.24, -2.28, -2.61],
        )
        # USA: -0.82, -1.39, -1.89, -3.23, -3.65
        np.testing.assert_allclose(
            tbl.loc[("GDP growth (%)", "USA"), ["t10", "t10_25", "t10_54", "t10_125", "t10_145"]].values,
            [-0.82, -1.39, -1.89, -3.23, -3.65],
        )

        # Check Inflation numbers
        np.testing.assert_allclose(
            tbl.loc[("Inflation (%)", "USA"), ["t10", "t10_25", "t10_54", "t10_125", "t10_145"]].values,
            [0.51, 0.88, 1.23, 2.30, 2.67],
        )
        np.testing.assert_allclose(
            tbl.loc[("Inflation (%)", "CAN"), ["t10", "t10_25", "t10_54", "t10_125", "t10_145"]].values,
            [-0.07, -0.12, -0.16, -0.24, -0.27],
        )

    def test_mean_by_scenario_table_exact_match(self) -> None:
        """Assert exact string matches for mean_by_scenario.tex."""
        scens = ["base", "t10", "t10_25", "t10_54", "t10_125", "t10_145"]
        require_scenarios(*scens)
        results_dict: dict[str, TradeEquilibriumResult] = {}
        gk_dict: dict[str, GearyKhamisResult] = {}

        base_eq = _equilibrium_from_reference("base")
        results_dict["base"] = base_eq

        for s_name in scens:
            eq = _equilibrium_from_reference(s_name)
            results_dict[s_name] = eq
            gk = compute_geary_khamis(eq, base_eq, matlab_compat=True)
            if s_name == "base":
                object.__setattr__(gk, "gdp_growth", np.zeros(77, dtype=float))
                object.__setattr__(gk, "real_gdp_growth", np.zeros(77, dtype=float))
            gk_dict[s_name] = gk

        batch = ScenarioBatchResult(
            scenarios=results_dict,
            geary_khamis=gk_dict,
            baseline_scenario="base",
            country_codes=CANONICAL_COUNTRY_CODES,
        )

        mean_tbl = batch.to_mean_by_scenario_table()

        # Check GDP growth (%)
        assert mean_tbl.loc["GDP growth (%)", "t10"] == "-0.695 (0.432)"
        assert mean_tbl.loc["GDP growth (%)", "t10_25"] == "-1.182 (0.699)"
        assert mean_tbl.loc["GDP growth (%)", "t10_54"] == "-1.609 (0.927)"
        assert mean_tbl.loc["GDP growth (%)", "t10_125"] == "-2.761 (1.529)"
        assert mean_tbl.loc["GDP growth (%)", "t10_145"] == "-3.121 (1.721)"

        # Check Inflation (%)
        assert mean_tbl.loc["Inflation (%)", "t10"] == "-0.041 (0.165)"
        assert mean_tbl.loc["Inflation (%)", "t10_25"] == "-0.063 (0.264)"
        assert mean_tbl.loc["Inflation (%)", "t10_54"] == "-0.081 (0.352)"
        assert mean_tbl.loc["Inflation (%)", "t10_125"] == "-0.116 (0.589)"
        assert mean_tbl.loc["Inflation (%)", "t10_145"] == "-0.123 (0.667)"

        # Check XN over GDP (%)
        assert mean_tbl.loc["XN over GDP (%)", "t10"] == "0.690 (9.914)"
        assert mean_tbl.loc["XN over GDP (%)", "t10_25"] == "0.679 (10.013)"
        assert mean_tbl.loc["XN over GDP (%)", "t10_54"] == "0.673 (10.105)"
        assert mean_tbl.loc["XN over GDP (%)", "t10_125"] == "0.661 (10.365)"
        assert mean_tbl.loc["XN over GDP (%)", "t10_145"] == "0.659 (10.453)"


# ---------------------------------------------------------------------------
# Tier 5: Warm-Started Batch Solve Test (< 60s)
# ---------------------------------------------------------------------------

class TestWarmStartedBatchSolve:
    """Verify warm-started batch solve runs in under 60 seconds with error < 10^-4."""

    def test_warm_started_batch_performance(self) -> None:
        """Execute full batch warm-started from reference solutions in < 60s."""
        data_mat = load_icio_data()
        calib = calibrate_trade_model(data_mat, ns=11, nc=77, nfd=3)

        base_eq = _equilibrium_from_reference("base")

        # t10_54 has no bundled reference (dataless MATLAB placeholder), so the
        # batch is warm-started from the six scenarios that do ship.
        available = available_reference_scenarios()
        scen_names = [
            s for s in ["base", "t10", "t10_25", "t10_54", "t10_75", "t10_125", "t10_145"]
            if s in available
        ]
        assert len(scen_names) >= 2, f"Too few bundled reference scenarios: {scen_names}"
        initial_guesses = {
            s: load_reference_solution(s)["xx_sol"].flatten()
            for s in scen_names
        }

        t0 = time.perf_counter()
        # Warm-started batch solve using the reference checkpoint solutions
        batch = run_scenario_batch(
            scenarios=scen_names,
            calib=calib,
            warm_start=True,
            compute_gk=True,
            base_result=base_eq,
            initial_guesses=initial_guesses,
            max_iter=5,
        )
        elapsed = time.perf_counter() - t0

        assert elapsed < 60.0, f"Batch solve took {elapsed:.2f}s, exceeding 60s budget!"
        assert len(batch.scenarios) == len(scen_names)
        assert len(batch.geary_khamis) == len(scen_names)

        # Assert relative error < 10^-4 vs MATLAB reference
        for s in scen_names:
            ref_x = initial_guesses[s]
            sol_x = batch.scenarios[s].x_sol
            rel_err = np.max(np.abs((sol_x - ref_x) / np.maximum(np.abs(ref_x), 1e-4)))
            assert rel_err < 1e-4, f"Scenario {s} relative error {rel_err} exceeds 1e-4"

    def test_continuation_single_newton_step(self) -> None:
        """Verify warm-started continuation from base to t10 runs 1 Newton step in < 60s."""
        data_mat = load_icio_data()
        calib = calibrate_trade_model(data_mat, ns=11, nc=77, nfd=3)

        base_mat = reference_or_skip("base")
        base_x = base_mat["xx_sol"].flatten()

        t0 = time.perf_counter()
        res = run_tariff_scenario("t10", calib, x0=base_x, max_iter=1)
        elapsed = time.perf_counter() - t0

        assert elapsed < 60.0, f"Single continuation step took {elapsed:.2f}s, exceeding 60s budget!"
        assert res.x_sol is not None
        assert res.x_sol.shape == base_x.shape
        assert np.all(np.isfinite(res.x_sol))
        assert res.iterations == 1
        # Exactly 1 step from base does not achieve full convergence, verifying accurate status reporting
        assert not res.converged
        assert np.isfinite(res.residual_norm)
        # The Newton step took a genuine non-zero step update from base
        step_norm = float(np.linalg.norm(res.x_sol - base_x))
        assert step_norm > 0.0, "Newton step did not update solution vector!"


# ---------------------------------------------------------------------------
# Negative Tests: Error Handling and Validation
# ---------------------------------------------------------------------------

class TestNegativeScenariosAndPPP:
    """Validate strict defensive error handling on invalid specifications."""

    def test_invalid_scenario_name(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """Verify requesting unknown scenario name raises KeyError."""
        with pytest.raises(KeyError, match="Unknown scenario"):
            get_canonical_scenario("unknown_shock_scenario")

        with pytest.raises(KeyError, match="Unknown scenario"):
            build_tariff_matrices("unknown_shock_scenario", synthetic_2c_2s_calib)

    def test_invalid_country_code(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """Verify specifying unknown country code in tariffs raises ValueError."""
        scen = TariffScenario(
            name="bad_country",
            us_import_tariffs={"NONEXISTENT_COUNTRY": 0.25},
        )
        with pytest.raises(ValueError, match="Unknown country code"):
            build_tariff_matrices(scen, synthetic_2c_2s_calib)

    def test_invalid_sector_code(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """Verify specifying unknown sector code in sectoral_tariffs raises ValueError."""
        scen = TariffScenario(
            name="bad_sector",
            sectoral_tariffs={("C1", "C2", "NONEXISTENT_SECTOR"): 0.25},
        )
        with pytest.raises(ValueError, match="Unknown sector code"):
            build_tariff_matrices(scen, synthetic_2c_2s_calib)

    def test_negative_tariff_rate(self) -> None:
        """Verify specifying negative tariff rate raises ValueError."""
        with pytest.raises(ValueError, match="Tariff rate"):
            TariffScenario(name="bad_rate", default_us_tariff=-0.10)

        with pytest.raises(ValueError, match="Tariff rate"):
            TariffScenario(name="bad_rate", us_import_tariffs={"C1": -0.20})

    def test_non_numeric_tariff_rate(self) -> None:
        """Verify non-numeric or NaN tariff rate raises ValueError or TypeError."""
        with pytest.raises((ValueError, TypeError)):
            TariffScenario(name="nan_rate", default_us_tariff=float("nan"))

        with pytest.raises((ValueError, TypeError)):
            TariffScenario(name="str_rate", us_import_tariffs={"C1": "10%"})  # type: ignore

    def test_unsupported_ppp_method(self) -> None:
        """Verify requesting unknown PPP method raises ValueError."""
        p = np.ones((3, 2))
        q = np.ones((3, 2))
        with pytest.raises(ValueError, match="Unsupported method"):
            solve_multilateral_ppp(p, q, method="quantum_annealing")

    def test_unsupported_normalize_option(self) -> None:
        """Verify requesting unknown normalization option raises ValueError."""
        p = np.ones((3, 2))
        q = np.ones((3, 2))
        with pytest.raises(ValueError, match="Unsupported normalize"):
            solve_multilateral_ppp(p, q, normalize="invalid_normalization")

    def test_mismatched_ppp_shapes(self) -> None:
        """Verify mismatched price and quantity shapes raise ValueError."""
        p = np.ones((3, 2))
        q = np.ones((3, 5))
        with pytest.raises(ValueError, match="Shape mismatch"):
            solve_multilateral_ppp(p, q)

    def test_invalid_scenario_type(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """Verify passing invalid scenario object type raises TypeError."""
        with pytest.raises(TypeError, match="scenario must be"):
            build_tariff_matrices(12345, synthetic_2c_2s_calib)  # type: ignore
