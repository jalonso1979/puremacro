"""Tests for puremacro.trade data ingestion, calibration, result containers, and numerical parity."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import os
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import scipy.io as sio

from puremacro.trade._results import (
    GearyKhamisResult,
    ScenarioBatchResult,
    TradeCalibrationResult,
    TradeEquilibriumResult,
)
from puremacro.trade.calibration import calibrate_trade_model
from puremacro.trade.data import (
    CANONICAL_COUNTRY_CODES,
    CANONICAL_FINAL_DEMAND_CODES,
    CANONICAL_SECTOR_CODES,
    EU_COUNTRY_CODES,
    ICIOData,
    get_country_codes,
    get_eu_country_codes,
    get_final_demand_codes,
    get_sector_codes,
    get_sector_names,
    load_icio_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def toy_icio_matrix() -> tuple[np.ndarray, int, int, int]:
    """Construct a mathematically balanced 3-country, 2-sector, 1-final-demand ICIO matrix.

    Dimensions:
      nc = 3, ns = 2, nfd = 1
      Rows: ns*nc + 3 = 6 + 3 = 9
      Cols: ns*nc + nfd*nc = 6 + 3 = 9

    Execution time: < 0.001s.
    """
    nc = 3
    ns = 2
    nfd = 1
    n_ind = ns * nc  # 6

    data = np.zeros((n_ind + 3, n_ind + nfd * nc), dtype=np.float64)

    # Balanced intermediate inputs (domestic and cross-border trade)
    for k1 in range(nc):
        for k2 in range(nc):
            base_val = 10.0 if k1 == k2 else 2.0
            r_slice = slice(ns * k1, ns * (k1 + 1))
            c_slice = slice(ns * k2, ns * (k2 + 1))
            data[r_slice, c_slice] = np.array([
                [base_val, base_val * 0.8],
                [base_val * 0.6, base_val * 1.2],
            ])

    # Final demand deliveries (columns 6..8)
    for k1 in range(nc):
        for k2 in range(nc):
            fd_val = 15.0 if k1 == k2 else 3.0
            r_slice = slice(ns * k1, ns * (k1 + 1))
            c_idx = n_ind + k2
            data[r_slice, c_idx] = np.array([fd_val, fd_val * 1.5])

    # Enforce Leontief double-entry accounting balance: Row sum == Column sum
    row_sales = np.sum(data[:n_ind, :], axis=1)
    col_interm = np.sum(data[:n_ind, :n_ind], axis=0)

    for j in range(n_ind):
        tax_val = 2.0
        va_val = row_sales[j] - col_interm[j] - tax_val
        data[n_ind, j] = tax_val
        data[n_ind + 1, j] = (2.0 / 3.0) * va_val  # Labor compensation (2/3 VA)
        data[n_ind + 2, j] = (1.0 / 3.0) * va_val  # Capital compensation (1/3 VA)

    # For final demand columns (6..8), row 6 carries final demand taxes
    for j in range(nc):
        data[n_ind, n_ind + j] = 1.2

    return data, nc, ns, nfd


@pytest.fixture
def toy_calibration_result(toy_icio_matrix) -> TradeCalibrationResult:
    """Calibrate the toy ICIO matrix into a TradeCalibrationResult."""
    data, nc, ns, nfd = toy_icio_matrix
    return calibrate_trade_model(
        data,
        nc=nc,
        ns=ns,
        nfd=nfd,
        country_codes=("CT1", "CT2", "CT3"),
        sector_codes=("SEC1", "SEC2"),
    )


@pytest.fixture(scope="module")
def matlab_benchmark_dir() -> Path | None:
    """Locate the reference MATLAB computation directory if available."""
    candidates = [
        Path(os.environ.get("IO_COMPUTATION_DIR", "")),
        Path(__file__).resolve().parent.parent.parent / "IO" / "computation" / "7_TIO_77c_vf",
        Path(__file__).resolve().parents[2] / "IO" / "computation" / "7_TIO_77c_vf",
        Path.cwd() / "computation" / "7_TIO_77c_vf",
        Path.cwd() / "IO" / "computation" / "7_TIO_77c_vf",
    ]
    for c in candidates:
        if c.exists() and (c / "results_77c_11s_base.mat").exists():
            return c
    return None


@pytest.fixture(scope="module")
def matlab_base_results(matlab_benchmark_dir) -> dict[str, np.ndarray] | None:
    """Load results_77c_11s_base.mat if available; skips integration test otherwise."""
    if matlab_benchmark_dir is None:
        pytest.skip("Reference results_77c_11s_base.mat not found in workspace.")
    mat_path = matlab_benchmark_dir / "results_77c_11s_base.mat"
    return sio.loadmat(str(mat_path))


@pytest.fixture(scope="module")
def matlab_raw_data(matlab_benchmark_dir) -> np.ndarray | None:
    """Load data_77c_11s.mat if available; skips integration test otherwise."""
    if matlab_benchmark_dir is None:
        pytest.skip("Reference data_77c_11s.mat not found in workspace.")
    data_path = matlab_benchmark_dir / "data_77c_11s.mat"
    mat = sio.loadmat(str(data_path))
    return mat["data"]


# ---------------------------------------------------------------------------
# Tier 1: Fast Unit Tests on Toy Fixtures (< 0.05s total)
# ---------------------------------------------------------------------------

class TestTradeCalibrationResultUnit:
    """Unit tests validating TradeCalibrationResult schema, immutability, and presentation."""

    def test_required_arrays_presence_validation(self, toy_calibration_result):
        """Verify that omitting or passing None for any of the 9 required arrays raises ValueError."""
        res = toy_calibration_result
        required_names = [
            "a", "afd", "alpha", "beta", "k_endow", "l_endow", "invforT", "tax", "ytot"
        ]

        for name in required_names:
            kwargs = {k: getattr(res, k) for k in required_names}
            kwargs[name] = None
            with pytest.raises(ValueError, match=f"requires array '{name}'"):
                TradeCalibrationResult(**kwargs)

    def test_required_arrays_type_validation(self, toy_calibration_result):
        """Verify that passing non-numpy arrays raises TypeError."""
        res = toy_calibration_result
        kwargs = {
            "a": res.a, "afd": res.afd, "alpha": res.alpha, "beta": res.beta,
            "k_endow": res.k_endow, "l_endow": res.l_endow, "invforT": res.invforT,
            "tax": res.tax, "ytot": res.ytot,
        }
        kwargs["alpha"] = [1.0, 2.0, 3.0]  # list instead of np.ndarray
        with pytest.raises(TypeError, match="must be a numpy.ndarray"):
            TradeCalibrationResult(**kwargs)

    def test_required_arrays_empty_validation(self, toy_calibration_result):
        """Verify that passing empty array (size 0) raises ValueError."""
        res = toy_calibration_result
        kwargs = {
            "a": res.a, "afd": res.afd, "alpha": res.alpha, "beta": res.beta,
            "k_endow": res.k_endow, "l_endow": res.l_endow, "invforT": res.invforT,
            "tax": res.tax, "ytot": res.ytot,
        }
        kwargs["ytot"] = np.array([])
        with pytest.raises(ValueError, match="cannot be empty"):
            TradeCalibrationResult(**kwargs)

    def test_frozen_immutability(self, toy_calibration_result):
        """Verify that TradeCalibrationResult is strictly immutable."""
        res = toy_calibration_result
        with pytest.raises(FrozenInstanceError):
            res.alpha = np.zeros_like(res.alpha)

        with pytest.raises(FrozenInstanceError):
            res.n_countries = 99

    def test_property_aliases_matching_matlab(self, toy_calibration_result):
        """Verify KT, LT, and ytot_base aliases resolve to exact identical arrays."""
        res = toy_calibration_result
        assert res.KT is res.k_endow
        assert res.LT is res.l_endow
        assert res.ytot_base is res.ytot

    def test_summary_dataframe_structure(self, toy_calibration_result):
        """Verify .summary() produces valid DataFrame with expected columns and index."""
        res = toy_calibration_result
        df = res.summary(detailed=True)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == res.n_countries
        assert list(df.index) == list(res.country_codes)
        expected_cols = {
            "Capital_Endow", "Labor_Endow", "Factor_Income",
            "Net_Transfers", "Gross_Output", "Mean_TFP", "Mean_TaxRate"
        }
        assert expected_cols.issubset(set(df.columns))

    def test_summary_aggregate_mode(self, toy_calibration_result):
        """Verify .summary(detailed=False) produces world overview table."""
        res = toy_calibration_result
        df = res.summary(detailed=False)
        assert isinstance(df, pd.DataFrame)
        assert "Global Gross Output" in df.index

    def test_to_frame_alias(self, toy_calibration_result):
        """Verify .to_frame() returns the same DataFrame as .summary()."""
        res = toy_calibration_result
        pd.testing.assert_frame_equal(res.to_frame(), res.summary())

    def test_to_markdown_formatting(self, toy_calibration_result):
        """Verify .to_markdown() generates valid table with pipe separators."""
        res = toy_calibration_result
        md = res.to_markdown()
        assert isinstance(md, str)
        assert "| Country |" in md or "| Country" in md
        assert "|---" in md
        for code in res.country_codes:
            assert code in md

    def test_to_latex_formatting(self, toy_calibration_result):
        """Verify .to_latex() generates valid tabular string with escaping."""
        res = toy_calibration_result
        latex = res.to_latex()
        assert isinstance(latex, str)
        assert "\\begin{tabular}" in latex
        assert "\\end{tabular}" in latex
        assert "\\hline" in latex
        for code in res.country_codes:
            assert code in latex

    def test_downstream_result_stubs(self):
        """Verify stub result dataclasses instantiate and provide summary/markdown/latex."""
        x = np.ones(10)
        p = np.ones((1, 2, 2))
        eq = TradeEquilibriumResult(
            x_sol=x, p_sol=p, y_sol=p, r_sol=p, w_sol=p, T_sol=p, XN_sol=x
        )
        assert isinstance(eq.summary(), pd.DataFrame)
        assert "CONVERGED" in eq.to_markdown()
        assert "\\begin{tabular}" in eq.to_latex()

        gk = GearyKhamisResult(
            ppp=np.ones(3), pi=np.ones(2), real_gdp=np.ones(3), nominal_gdp=np.ones(3)
        )
        assert isinstance(gk.summary(), pd.DataFrame)
        assert "Country" in gk.to_markdown()
        assert "\\begin{tabular}" in gk.to_latex()

        batch = ScenarioBatchResult(
            scenarios={"base": eq},
            ppp_results={"base": gk},
            real_gdp_table=pd.DataFrame({"base": [0.0]}, index=["USA"]),
            trade_balance_table=pd.DataFrame({"base": [0.0]}, index=["USA"]),
            cpi_table=pd.DataFrame({"base": [0.0]}, index=["USA"]),
        )
        assert isinstance(batch.summary(), pd.DataFrame)
        assert "USA" in batch.to_markdown()
        assert "\\begin{tabular}" in batch.to_latex()


class TestToyCalibrationEconomics:
    """Tests verifying economic accounting identities on the calibrated toy model."""

    def test_cobb_douglas_alpha_is_one_third(self, toy_calibration_result):
        """Verify capital share alpha equals 1/3 (factor share convention)."""
        res = toy_calibration_result
        np.testing.assert_allclose(res.alpha, 1.0 / 3.0, rtol=1e-12, atol=1e-12)

    def test_validation_checklist_passes(self, toy_calibration_result):
        """Verify .validate() reports all dimensional and non-negativity checks passing."""
        res = toy_calibration_result
        checks = res.validate()
        assert checks["dim_a"]
        assert checks["dim_afd"]
        assert checks["dim_alpha"]
        assert checks["dim_beta"]
        assert checks["dim_tax"]
        assert checks["dim_ytot"]
        assert checks["dim_k_endow"]
        assert checks["dim_l_endow"]
        assert checks["dim_invforT"]
        assert checks["nonneg_ytot"]
        assert checks["nonneg_k_endow"]
        assert checks["nonneg_l_endow"]
        assert checks["nonneg_a"]
        assert checks["nonneg_afd"]
        assert checks["alpha_bounds"]
        assert checks["global_transfer_balance"]

    def test_technical_coefficients_bounds(self, toy_calibration_result):
        """Verify 0 <= a_ij < 1 and final demand shares sum to 1."""
        res = toy_calibration_result
        assert np.all(res.a >= 0.0)
        assert np.all(res.afd >= 0.0)
        afd_sum = res.afd.sum(axis=0)
        np.testing.assert_allclose(afd_sum, 1.0, rtol=1e-10)

    def test_tensors_3d_and_4d(self, toy_calibration_result):
        """Verify both 3D and 4D tensors are accessible and have correct dimensions."""
        res = toy_calibration_result
        assert res.a.shape == (6, 2, 3)
        assert res.afd.shape == (6, 1, 3)
        assert res.a_3d.shape == (6, 2, 3)
        assert res.afd_3d.shape == (6, 1, 3)
        assert res.a_4d.shape == (2, 3, 2, 3)
        assert res.afd_4d.shape == (2, 3, 1, 3)

    def test_inactive_sectors_division_warning_free(self):
        """Verify that synthetic datasets with completely inactive sectors (y=0, l=0) emit zero warnings."""
        import warnings
        nc, ns, nfd = 3, 2, 1
        n_ind = ns * nc
        toy_data = np.ones((n_ind + 3, n_ind + nfd * nc), dtype=np.float64) * 10.0
        # Sector 1 in country 0 is completely inactive
        toy_data[:, 1] = 0.0
        toy_data[1, :] = 0.0
        toy_data[n_ind:, 1] = 0.0

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            calib = calibrate_trade_model(toy_data, ns=ns, nc=nc, nfd=nfd, validate=False)

        runtime_warnings = [w for w in record if issubclass(w.category, RuntimeWarning)]
        assert len(runtime_warnings) == 0, f"Expected 0 RuntimeWarnings on inactive sectors, got: {runtime_warnings}"
        assert calib.ytot[0, 1, 0] == 0.0
        assert calib.alpha[0, 1, 0] == 0.0
        assert calib.beta[0, 1, 0] == 0.0


class TestTradeDataModule:
    """Unit tests for puremacro.trade.data constants, functions, and ICIOData container."""

    def test_canonical_counts(self):
        """Assert standard country, sector, and final demand dimension counts."""
        countries = get_country_codes()
        assert len(countries) == 77
        assert countries[0] == "ARG"
        assert countries[9] == "CAN"
        assert countries[12] == "CHN"
        assert countries[47] == "MEX"
        assert countries[73] == "USA"
        assert countries[76] == "ROW"

        eu = get_eu_country_codes()
        assert len(eu) == 27
        assert "DEU" in eu and "FRA" in eu and "ITA" in eu

        sectors = get_sector_codes()
        assert len(sectors) == 11
        assert sectors[0] == "AGRI"
        assert sectors[2] == "MANU"
        assert sectors[10] == "OTHS"

        names = get_sector_names()
        assert len(names) == 11
        assert names["MANU"] == "Manufacturing"

        fds = get_final_demand_codes()
        assert fds == ["C", "I", "Cx"]

    def test_icio_data_container_and_properties(self, matlab_raw_data):
        """Verify ICIOData container slices and accounting identities."""
        icio = load_icio_data(return_structured=True)
        assert isinstance(icio, ICIOData)
        assert icio.matrix.shape == (850, 1078)
        assert icio.intermediate_matrix.shape == (847, 847)
        assert icio.final_demand_matrix.shape == (847, 231)
        assert icio.net_taxes_intermediate.shape == (847,)
        assert icio.net_taxes_final_demand.shape == (231,)
        assert icio.labor_va.shape == (847,)
        assert icio.capital_va.shape == (847,)
        assert icio.gross_output.shape == (847,)

        # Capital to labor value added ratio is 1/2
        valid = icio.labor_va > 0
        np.testing.assert_allclose(
            icio.capital_va[valid] / icio.labor_va[valid], 0.5, rtol=1e-12
        )

        # Output outlay column sum matches sales row sum within floating point tolerance
        col_outlay = icio.gross_output
        row_sales = np.sum(icio.matrix[:847, :], axis=1)
        max_diff = np.max(np.abs(col_outlay - row_sales))
        assert max_diff < 2e-4

    def test_load_icio_data_calibration_validation(self, matlab_raw_data):
        """Verify calibrate_trade_model(load_icio_data()).validate() passes cleanly."""
        raw = load_icio_data()
        calib = calibrate_trade_model(raw)
        checks = calib.validate()
        assert isinstance(checks, dict)
        failing = [k for k, v in checks.items() if not v]
        assert not failing, f"calibrate_trade_model(load_icio_data()) failed validation checks: {failing}"
        assert all(checks.values()), f"Expected all validation checks to be True, got: {checks}"


# ---------------------------------------------------------------------------
# Tier 2: Machine-Precision Numerical Parity vs MATLAB (< 1e-12 relative diff)
# ---------------------------------------------------------------------------

class TestMatlabCalibrationParity:
    """Integration test suite asserting < 1e-12 relative difference against results_77c_11s_base.mat."""

    @pytest.fixture(autouse=True)
    def setup_parity(self, matlab_raw_data, matlab_base_results):
        """Run Python calibration on 77c x 11s ICIO data."""
        self.py_calib = calibrate_trade_model(matlab_raw_data, nc=77, ns=11, nfd=3)
        self.mat = matlab_base_results

    def _assert_relative_error(self, actual: np.ndarray, expected: np.ndarray, name: str, tol: float = 1e-12):
        """Assert maximum relative discrepancy is strictly below machine-precision threshold."""
        assert actual.shape == expected.shape, (
            f"Shape mismatch for {name}: Python {actual.shape} vs MATLAB {expected.shape}"
        )
        max_abs = float(np.max(np.abs(actual - expected)))
        denom = float(np.max(np.abs(expected)))
        rel_diff = max_abs / denom if denom > 0 else max_abs
        assert rel_diff < tol, (
            f"Parity failure for '{name}': relative difference {rel_diff:.3e} exceeds tolerance {tol:.3e} "
            f"(max_abs={max_abs:.3e}, max_val={denom:.3e})"
        )

    def test_parity_alpha_capital_share(self):
        """Assert exact machine parity for alpha (capital share = 1/3)."""
        self._assert_relative_error(self.py_calib.alpha, self.mat["alpha"], "alpha", tol=1e-12)

    def test_parity_tax_production_rates(self):
        """Assert machine precision parity for net production tax rates."""
        self._assert_relative_error(self.py_calib.tax, self.mat["tax"], "tax", tol=1e-12)

    def test_parity_beta_tfp_scale(self):
        """Assert machine precision parity for Cobb-Douglas TFP scale beta."""
        self._assert_relative_error(self.py_calib.beta, self.mat["beta"], "beta", tol=1e-12)

    def test_parity_capital_endowment_KT(self):
        """Assert exact machine parity for capital endowments KT."""
        self._assert_relative_error(self.py_calib.k_endow, self.mat["KT"], "k_endow (KT)", tol=1e-12)

    def test_parity_labor_endowment_LT(self):
        """Assert exact machine parity for labor endowments LT."""
        self._assert_relative_error(self.py_calib.l_endow, self.mat["LT"], "l_endow (LT)", tol=1e-12)

    def test_parity_invforT_current_account(self):
        """Assert machine precision parity for net foreign transfers invforT."""
        self._assert_relative_error(self.py_calib.invforT, self.mat["invforT"], "invforT", tol=1e-12)

    def test_parity_ytot_gross_output(self):
        """Assert machine precision parity for baseline gross output ytot."""
        self._assert_relative_error(self.py_calib.ytot, self.mat["ytot"], "ytot", tol=1e-12)

    def test_parity_intermediate_coefficients_a(self):
        """Assert machine precision parity for input-output technical coefficients a."""
        self._assert_relative_error(self.py_calib.a, self.mat["a"], "a", tol=1e-12)

    def test_parity_final_demand_coefficients_afd(self):
        """Assert machine precision parity for final demand sourcing coefficients afd."""
        self._assert_relative_error(self.py_calib.afd, self.mat["afd"], "afd", tol=1e-12)

    def test_parity_expenditure_shares_theta(self):
        """Assert machine precision parity for final demand expenditure shares theta."""
        if self.py_calib.theta is not None and "theta" in self.mat:
            self._assert_relative_error(self.py_calib.theta, self.mat["theta"], "theta", tol=1e-12)

    def test_parity_final_demand_tax_rates(self):
        """Assert machine precision parity for final demand tax rates tax_fd."""
        if self.py_calib.tax_fd is not None and "tax_fd" in self.mat:
            self._assert_relative_error(self.py_calib.tax_fd, self.mat["tax_fd"], "tax_fd", tol=1e-12)

    def test_parity_reconstructed_table_data_calibra(self):
        """Assert machine precision parity for reconstructed table data_calibra."""
        if self.py_calib.data_calibra is not None and "data_calibra" in self.mat:
            self._assert_relative_error(
                self.py_calib.data_calibra, self.mat["data_calibra"], "data_calibra", tol=1e-12
            )

    def test_negative_investment_division_edge_case(self):
        """Verify that negative investment consumption (Bulgaria) does not cause zeroed tax rate."""
        # Bulgaria is index 5 (ARG=0, AUS=1, AUT=2, BEL=3, BGD=4, BGR=5)
        bgr_idx = 5
        tax_fd_inv_bgr = self.py_calib.tax_fd[0, 1, bgr_idx]
        assert tax_fd_inv_bgr != 0.0, "Bulgaria investment tax rate should not be zeroed out!"
        assert tax_fd_inv_bgr < 0.0, "Bulgaria investment tax rate should be negative due to c < 0."
        expected_tax_fd = self.mat["tax_fd"][0, 1, bgr_idx]
        np.testing.assert_allclose(tax_fd_inv_bgr, expected_tax_fd, rtol=1e-12)

    def test_empirical_calibration_validation_clean(self):
        """Verify TradeCalibrationResult.validate() passes cleanly on empirical ICIO calibration."""
        checks = self.py_calib.validate()
        assert isinstance(checks, dict), "validate() must return a dict[str, bool]"
        failing_checks = [k for k, v in checks.items() if not v]
        assert not failing_checks, f"Empirical calibration failed validate() checks: {failing_checks}"
        assert all(checks.values()), "All validation checks must evaluate to True"

        # Explicitly verify critical accounting identities
        assert checks["dim_a"] is True
        assert checks["dim_afd"] is True
        assert checks["nonneg_a"] is True
        assert checks["nonneg_afd"] is True
        assert checks["alpha_bounds"] is True
        assert checks["global_transfer_balance"] is True

    def test_inventory_disinvestment_edge_case(self):
        """Verify that negative afd entries exist in Category 1 for LTU, UKR, VNM and match MATLAB."""
        neg_mask = self.py_calib.afd < 0.0
        assert np.any(neg_mask), "Empirical afd must contain negative entries due to inventory disinvestment"

        # Verify negative entries occur ONLY in Category 1 (Investment)
        neg_cats = np.where(neg_mask)[1]
        assert np.all(neg_cats == 1), "Negative afd entries must only occur in Category 1 (Investment)"

        # Verify non-negativity in consumption categories 0 and 2
        assert np.all(self.py_calib.afd[:, [0, 2], :] >= 0.0)

        # Verify unit sum property holds across all categories and countries
        np.testing.assert_allclose(np.sum(self.py_calib.afd, axis=0), 1.0, atol=1e-12)

        # Verify exact machine parity against MATLAB benchmark afd
        np.testing.assert_allclose(self.py_calib.afd[neg_mask], self.mat["afd"][neg_mask], atol=1e-12)


# ---------------------------------------------------------------------------
# Tier 3: Subpackage Integrity & Non-Regression Tests
# ---------------------------------------------------------------------------

class TestTradeSubpackageIntegrity:
    """Regression and integrity tests for puremacro.trade modules."""

    def test_no_machine_specific_paths_in_trade_modules(self):
        """Confirm that puremacro.trade modules contain zero machine-specific paths.

        Mirrors the project release gate in tests/test_no_machine_specific_paths.py.
        Strictly forbids hardcoded user home directories, Windows user paths,
        cloud account storage, or personal email addresses in shipped trade code.
        """
        import re
        import puremacro.trade

        forbidden = {
            "unix home directory": re.compile(r"[\"']/(?:Users|home)/[A-Za-z0-9._-]+/"),
            "windows user directory": re.compile(r"[A-Za-z]:\\\\Users\\\\"),
            "cloud account in a path": re.compile(r"GoogleDrive-[^\"'\s/]+|OneDrive-[^\"'\s/]+"),
            "bare email address": re.compile(r"[\"'][^\"'\s]*@(?:gmail|hotmail|outlook|yahoo)\.[a-z]+"),
        }

        trade_pkg_dir = Path(puremacro.trade.__file__).resolve().parent
        sources = sorted(trade_pkg_dir.glob("*.py"))
        assert len(sources) >= 4, f"Expected at least 4 trade modules, found {len(sources)}"

        hits = []
        for path in sources:
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if line.lstrip().startswith("#"):
                    continue
                for label, pattern in forbidden.items():
                    if pattern.search(line):
                        hits.append(f"{path.name}:{lineno} [{label}]: {line.strip()}")

        assert not hits, (
            "Machine-specific path detected in shipped puremacro.trade modules:\n  "
            + "\n  ".join(hits)
            + "\n\nUse relative path resolution (e.g. here.parents[3]) or environment variables "
            "(IO_DATA_PATH, IO_COMPUTATION_DIR)."
        )
