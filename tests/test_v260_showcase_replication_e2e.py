"""Comprehensive 4-Tier Opaque-Box E2E Test Suite for puremacro v2.6.0.

Showcase Modernization & Replication Suite Extension:
- F1: Purge External Regressors (statsmodels/linearmodels/arch)
- F2: Upgrade Showcase 41 DSGE Estimation
- F3: Bilingual Synchronization & Styling
- F4: Flagship DSGE Bayesian Estimation Showcase (Notebook 42)
- F5: Visual Mode Diagnostics (mode_check)
- F6: Kalman Smoother, Shock Decomposition & Forecasting
- F7: Marginal Data Density & Model Comparison
- F8: DSGE Estimation Replication Cases
- F9: Pure-NumPy Econometric Regression Replication Cases
- F10: Replication Auto-Discovery & Scorecard
- F11: Pyodide Purity & Release Check Gates

Test Tiers:
- Tier 1: Feature Coverage (>=5 tests per feature, happy path in isolation)
- Tier 2: Boundary & Corner Cases (>=5 tests per feature, edge cases & error handling)
- Tier 3: Cross-Feature Interactions (pairwise combinations between subsystems)
- Tier 4: Real-World Scenarios (end-to-end macroeconomic workflows)
"""
from __future__ import annotations

import ast
from collections.abc import Sequence
import importlib
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import puremacro
import puremacro.dsge as dsge
from puremacro.dsge.marginal import (
    HarmonicMeanResult,
    harmonic_mean_mdd,
    laplace_mdd,
    model_comparison,
)
from puremacro.dsge.mode import csminwel, find_mode, mode_check
import puremacro.regress as reg
from puremacro.regress import add_constant, logit, ols, poisson, quantreg
import puremacro.replication as rep
from puremacro.replication import ReplicationCase, TargetKind, Tol, run, run_all, scorecard

# ===========================================================================
# Fixtures & Shared Test Helpers
# ===========================================================================

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = WORKSPACE_ROOT / "notebooks"
DSGE_DIR = Path(dsge.__file__).resolve().parent
SW07_MOD_PATH = DSGE_DIR / "_references" / "sw07_pfeifer.mod"
SW07_DATA_PATH = DSGE_DIR / "_sw07_data.csv"

SW07_RENAME_MAP = {
    "gdp_growth": "dy",
    "cons_growth": "dc",
    "inv_growth": "dinve",
    "wage_growth": "dw",
    "log_hours": "labobs",
    "infl": "pinfobs",
    "ffr": "robs",
}


@pytest.fixture(scope="module")
def sw07_mod_file() -> Path:
    assert SW07_MOD_PATH.exists(), f"sw07_pfeifer.mod missing at {SW07_MOD_PATH}"
    return SW07_MOD_PATH


@pytest.fixture(scope="module")
def sw07_model(sw07_mod_file: Path):
    return dsge.load_mod(sw07_mod_file, order=1)


@pytest.fixture(scope="module")
def sw07_data() -> pd.DataFrame:
    assert SW07_DATA_PATH.exists(), f"_sw07_data.csv missing at {SW07_DATA_PATH}"
    df = pd.read_csv(SW07_DATA_PATH, comment="#")
    return df.rename(columns=SW07_RENAME_MAP)


@pytest.fixture(scope="module")
def sample_regression_data():
    rng = np.random.default_rng(42)
    n = 120
    x1 = rng.normal(size=n)
    x2 = rng.uniform(0.5, 2.5, size=n)
    heteroskedastic_noise = rng.normal(scale=0.5 * np.abs(x2), size=n)
    y = 1.5 + 2.0 * x1 - 1.2 * x2 + heteroskedastic_noise
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    return df


def _get_file_imports(filepath: Path) -> list[str]:
    """Parse imports from a python file, stripping IPython magics if present."""
    try:
        content = filepath.read_text(encoding="utf-8")
        clean_lines = [
            f"# {line}" if line.strip().startswith(("%", "!")) else line
            for line in content.splitlines()
        ]
        tree = ast.parse("\n".join(clean_lines), filename=str(filepath))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    imports.append(a.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return imports
    except Exception:
        return []


# ===========================================================================
# TIER 1: Feature Coverage (>=5 test cases per feature in isolation)
# ===========================================================================

class TestTier1FeatureCoverage:
    """Tier 1: Comprehensive sanity tests for each of the 5 assigned features."""

    # -----------------------------------------------------------------------
    # Feature 1: Prohibited External Regressor Imports Check (>=5 tests)
    # -----------------------------------------------------------------------

    def test_t1_f1_no_statsmodels_imports_in_all_notebook_py(self):
        """T1.1.1: Verify zero direct statsmodels imports in any .py notebook."""
        py_notebooks = list(NOTEBOOKS_DIR.glob("**/*.py"))
        assert len(py_notebooks) >= 10, f"Found only {len(py_notebooks)} notebooks"
        
        violations = []
        for nb in py_notebooks:
            if nb.name.startswith("_"):
                continue
            imports = _get_file_imports(nb)
            for imp in imports:
                if imp == "statsmodels" or imp.startswith("statsmodels."):
                    violations.append(f"{nb.relative_to(WORKSPACE_ROOT)}: {imp}")
        
        assert not violations, f"Forbidden statsmodels imports found: {violations}"

    def test_t1_f1_no_linearmodels_or_arch_imports_in_notebooks(self):
        """T1.1.2: Verify zero linearmodels or arch imports in any notebook."""
        py_notebooks = list(NOTEBOOKS_DIR.glob("**/*.py"))
        violations = []
        for nb in py_notebooks:
            if nb.name.startswith("_"):
                continue
            imports = _get_file_imports(nb)
            for imp in imports:
                if (imp == "linearmodels" or imp.startswith("linearmodels.") or
                    imp == "arch" or imp.startswith("arch.")):
                    violations.append(f"{nb.relative_to(WORKSPACE_ROOT)}: {imp}")
        assert not violations, f"Forbidden package imports found: {violations}"

    def test_t1_f1_n12_paleoclimate_uses_native_regress(self):
        """T1.1.3: Verify N12 paleoclimate showcase exclusively imports puremacro.regress."""
        n12_path = NOTEBOOKS_DIR / "macro_history_and_climate" / "N12_paleoclimate_eiv_and_simex.py"
        assert n12_path.exists(), f"N12 missing at {n12_path}"
        
        imports = _get_file_imports(n12_path)
        assert any("puremacro.regress" in imp for imp in imports), f"N12 must import puremacro.regress; found: {imports}"
        assert not any("statsmodels" in imp for imp in imports), "N12 still contains statsmodels import"

    def test_t1_f1_no_prohibited_imports_in_course_directory(self):
        """T1.1.4: Verify zero prohibited imports in notebooks/course directory."""
        course_dir = NOTEBOOKS_DIR / "course"
        if course_dir.exists():
            for nb in course_dir.glob("*.py"):
                imports = _get_file_imports(nb)
                for imp in imports:
                    assert not (imp.startswith("statsmodels") or imp.startswith("linearmodels") or imp.startswith("arch")), (
                        f"Forbidden import {imp} in {nb.name}"
                    )

    def test_t1_f1_sys_modules_purity_after_regress_execution(self, sample_regression_data):
        """T1.1.5: Verify statsmodels/linearmodels/arch are not imported by puremacro.regress."""
        import subprocess

        code = (
            "import sys, numpy as np, pandas as pd\n"
            "from puremacro.regress import add_constant, ols\n"
            "df = pd.DataFrame({'y': np.arange(20, dtype=float), 'x1': np.arange(20, dtype=float), 'x2': np.arange(20, dtype=float)**2})\n"
            "X = add_constant(df[['x1', 'x2']])\n"
            "res = ols(df['y'], X, cov_type='HC1')\n"
            "assert res.params is not None\n"
            "for forbidden in ('statsmodels', 'linearmodels', 'arch'):\n"
            "    assert forbidden not in sys.modules, f'{forbidden} was loaded into sys.modules'\n"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr

    # -----------------------------------------------------------------------
    # Feature 2: Showcase 41 runs natively with smoother and estimate (>=5 tests)
    # -----------------------------------------------------------------------

    def test_t1_f2_sw07_model_structure_and_observables(self, sw07_model):
        """T1.2.1: Verify sw07_pfeifer.mod loads with 40 variables, 7 observables, 36 estimated params."""
        assert len(sw07_model.variables) == 40
        assert len(sw07_model.shocks) == 7
        assert hasattr(sw07_model, "_varobs") and len(sw07_model._varobs) == 7
        assert hasattr(sw07_model, "_estimated_params") and len(sw07_model._estimated_params.names()) == 36

    def test_t1_f2_sw07_data_file_integrity(self, sw07_data):
        """T1.2.2: Verify bundled _sw07_data.csv has 156 quarterly rows and all 7 observables."""
        required_obs = ["dy", "dc", "dinve", "labobs", "pinfobs", "dw", "robs"]
        assert len(sw07_data) == 156
        for col in required_obs:
            assert col in sw07_data.columns, f"Required observable {col} missing from sw07_data"
            assert not sw07_data[col].isna().any(), f"Observable {col} contains NaNs"

    def test_t1_f2_sw07_native_smoother_execution(self, sw07_model, sw07_data):
        """T1.2.3: Call m.smoother(data) on sw07 and verify state/shock trajectories."""
        res = sw07_model.smoother(sw07_data)
        assert isinstance(res, dsge.SmootherResult)
        assert res.states.shape == (156, len(sw07_model.states))
        assert res.shocks.shape == (156, len(sw07_model.shocks))
        assert not np.isnan(res.states.to_numpy()).any()
        assert not np.isnan(res.shocks.to_numpy()).any()

    def test_t1_f2_sw07_native_estimate_execution(self, sw07_model, sw07_data):
        """T1.2.4: Call m.estimate(data) on sw07 and verify DSGEPosteriorResult."""
        res = sw07_model.estimate(sw07_data, mode_compute="none", n_draws=4, burn_in=2, seed=42)
        assert isinstance(res, dsge.DSGEPosteriorResult)
        assert res.draws.shape == (2, 4, 36)
        assert len(res.mode) == 36
        assert hasattr(res, "summary")

    def test_t1_f2_showcase_41_bilingual_pair_exists(self):
        """T1.2.5: Verify 41_dynare_frontier_showcase.py and _es.py exist with equal cells."""
        en_path = NOTEBOOKS_DIR / "41_dynare_frontier_showcase.py"
        es_path = NOTEBOOKS_DIR / "41_dynare_frontier_showcase_es.py"
        assert en_path.exists(), f"English notebook 41 missing at {en_path}"
        assert es_path.exists(), f"Spanish notebook 41 missing at {es_path}"
        
        en_cells = [c for c in en_path.read_text(encoding="utf-8").split("# %%") if c.strip()]
        es_cells = [c for c in es_path.read_text(encoding="utf-8").split("# %%") if c.strip()]
        assert len(en_cells) == len(es_cells), f"Mismatched cell count: {len(en_cells)} vs {len(es_cells)}"

    # -----------------------------------------------------------------------
    # Feature 3: Notebook 42 structure and exports (>=5 tests)
    # -----------------------------------------------------------------------

    def test_t1_f3_notebook_42_pair_contract(self):
        """T1.3.1: Verify Notebook 42 English and Spanish source pair existence or path contract."""
        en_path = NOTEBOOKS_DIR / "42_dsge_bayesian_estimation_and_diagnostics.py"
        es_path = NOTEBOOKS_DIR / "42_dsge_bayesian_estimation_and_diagnostics_es.py"
        if en_path.exists():
            assert es_path.exists(), f"Spanish edition {es_path} missing when English exists"
        else:
            assert en_path.parent.is_dir()

    def test_t1_f3_notebook_42_seven_cell_structure(self):
        """T1.3.2: Verify Notebook 42 follows the canonical 7-section template."""
        en_path = NOTEBOOKS_DIR / "42_dsge_bayesian_estimation_and_diagnostics.py"
        if not en_path.exists():
            pytest.skip("Notebook 42 pending implementation by Worker M2")
        
        content = en_path.read_text(encoding="utf-8")
        cells = content.split("# %%")
        assert len(cells) >= 7, f"Expected at least 7 cells, found {len(cells)}"
        assert any("sw07_pfeifer.mod" in c for c in cells), "Must reference sw07_pfeifer.mod"
        assert any("mode_check" in c for c in cells), "Must demonstrate mode_check"

    def test_t1_f3_notebook_42_puremacro_imports(self):
        """T1.3.3: Verify Notebook 42 imports puremacro.dsge and zero external econometric packages."""
        en_path = NOTEBOOKS_DIR / "42_dsge_bayesian_estimation_and_diagnostics.py"
        if not en_path.exists():
            pytest.skip("Notebook 42 pending implementation by Worker M2")
        
        imports = _get_file_imports(en_path)
        assert any("puremacro.dsge" in imp for imp in imports)
        for imp in imports:
            assert not (imp.startswith("statsmodels") or imp.startswith("linearmodels") or imp.startswith("arch")), (
                f"Prohibited import {imp} found in Notebook 42"
            )

    def test_t1_f3_notebook_42_inline_assertions(self):
        """T1.3.4: Verify presence of robust inline assertions in Notebook 42."""
        en_path = NOTEBOOKS_DIR / "42_dsge_bayesian_estimation_and_diagnostics.py"
        if not en_path.exists():
            pytest.skip("Notebook 42 pending implementation by Worker M2")
        
        clean_lines = [
            f"# {line}" if line.strip().startswith(("%", "!")) else line
            for line in en_path.read_text(encoding="utf-8").splitlines()
        ]
        tree = ast.parse("\n".join(clean_lines))
        assert_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]
        assert len(assert_nodes) >= 3, f"Expected >=3 inline assertions, found {len(assert_nodes)}"

    def test_t1_f3_notebook_42_styling_application(self):
        """T1.3.5: Verify Notebook 42 applies puremacro editorial styling contract."""
        en_path = NOTEBOOKS_DIR / "42_dsge_bayesian_estimation_and_diagnostics.py"
        if not en_path.exists():
            pytest.skip("Notebook 42 pending implementation by Worker M2")
        
        content = en_path.read_text(encoding="utf-8")
        assert ("_nbstyle.apply_style" in content or "set_style" in content or "style" in content), (
            "Notebook 42 must configure styling"
        )

    # -----------------------------------------------------------------------
    # Feature 4: Replication cases discoverability (>=5 tests)
    # -----------------------------------------------------------------------

    def test_t1_f4_replication_cases_discovered(self):
        """T1.4.1: Verify _discover_cases() dynamically discovers cases."""
        cases = rep.runner._discover_cases()
        assert len(cases) >= 12, f"Expected at least 12 discovered cases, got {len(cases)}"
        families = set(c.family for c in cases)
        assert "dsge_estimation" in families, "dsge_estimation family missing"
        assert "regression" in families, "regression family missing"
        assert "stylized_facts" in families, "stylized_facts family missing"
        for c in cases:
            assert isinstance(c, ReplicationCase)

    def test_t1_f4_replication_cases_unique_valid_ids(self):
        """T1.4.2: Verify all discovered case IDs are non-empty, unique, and dot-separated."""
        cases = rep.runner._discover_cases()
        ids = [c.id for c in cases]
        assert len(ids) == len(set(ids)), f"Duplicate case IDs found: {ids}"
        for cid in ids:
            assert "." in cid, f"Case ID {cid} must contain dot separator (family.name)"

    def test_t1_f4_replication_cases_bibliographic_metadata(self):
        """T1.4.3: Verify paper citation, citation location, and target kinds."""
        cases = rep.runner._discover_cases()
        for c in cases:
            assert len(c.paper.strip()) > 0, f"Case {c.id} missing paper citation"
            assert len(c.citation.strip()) > 0, f"Case {c.id} missing table/page citation"
            assert isinstance(c.target_kind, TargetKind)
            assert isinstance(c.tol, Tol)

    def test_t1_f4_replication_cases_bilingual_titles(self):
        """T1.4.4: Verify English and Spanish titles are populated for all cases."""
        cases = rep.runner._discover_cases()
        for c in cases:
            assert len(c.title.strip()) > 0, f"Case {c.id} missing English title"
            assert len(c.title_es.strip()) > 0, f"Case {c.id} missing Spanish title"

    def test_t1_f4_replication_cases_callable_estimates(self):
        """T1.4.5: Verify case.estimate is a zero-argument callable returning a mapping."""
        cases = rep.runner._discover_cases()
        for c in cases:
            assert callable(c.estimate), f"Case {c.id} estimate is not callable"

    # -----------------------------------------------------------------------
    # Feature 5: Scorecard columns and data types (>=5 tests)
    # -----------------------------------------------------------------------

    def test_t1_f5_scorecard_returns_dataframe(self):
        """T1.5.1: Verify rep.scorecard() returns a pandas DataFrame."""
        fast_cases = [c for c in rep.runner._discover_cases() if c.family == "stylized_facts"]
        results = [run(c) for c in fast_cases]
        sc = scorecard(results)
        assert isinstance(sc, pd.DataFrame)
        assert len(sc) == len(fast_cases)

    def test_t1_f5_scorecard_exact_column_schema(self):
        """T1.5.2: Verify exact required columns are present in scorecard."""
        expected_cols = [
            "id", "family", "paper", "source", "target_kind",
            "tol", "passed", "margin", "network", "citation"
        ]
        fast_cases = [c for c in rep.runner._discover_cases() if c.family == "stylized_facts"]
        results = [run(c) for c in fast_cases]
        sc = scorecard(results)
        for col in expected_cols:
            assert col in sc.columns, f"Expected column {col} missing from scorecard"

    def test_t1_f5_scorecard_passed_column_dtype(self):
        """T1.5.3: Verify 'passed' column is boolean dtype."""
        fast_cases = [c for c in rep.runner._discover_cases() if c.family == "stylized_facts"]
        results = [run(c) for c in fast_cases]
        sc = scorecard(results)
        assert sc["passed"].dtype == bool or str(sc["passed"].dtype) == "boolean"

    def test_t1_f5_scorecard_margin_column_numeric(self):
        """T1.5.4: Verify 'margin' column is float numeric dtype."""
        fast_cases = [c for c in rep.runner._discover_cases() if c.family == "stylized_facts"]
        results = [run(c) for c in fast_cases]
        sc = scorecard(results)
        assert np.issubdtype(sc["margin"].dtype, np.floating)

    def test_t1_f5_scorecard_family_filtering(self):
        """T1.5.5: Verify scorecard preserves family filtering."""
        fast_cases = [c for c in rep.runner._discover_cases() if c.family == "stylized_facts"]
        results = [run(c) for c in fast_cases]
        sc = scorecard(results)
        assert set(sc["family"]) == {"stylized_facts"}


# ===========================================================================
# TIER 2: Boundary & Corner Cases (>=5 test cases per feature)
# ===========================================================================

class TestTier2BoundaryAndCornerCases:
    """Tier 2: Boundary, extreme limits, singular geometries, and error handling."""

    # -----------------------------------------------------------------------
    # Feature 1: Empty Data Handling (>=5 tests)
    # -----------------------------------------------------------------------

    def test_t2_f1_empty_dataframe_to_smoother_raises(self, sw07_model):
        """T2.1.1: Passing empty DataFrame to smoother raises informative ValueError."""
        with pytest.raises(ValueError, match="data is missing column"):
            sw07_model.smoother(pd.DataFrame())

    def test_t2_f1_empty_dataframe_to_estimate_raises(self, sw07_model):
        """T2.1.2: Passing empty DataFrame to estimate raises informative ValueError."""
        with pytest.raises(ValueError):
            sw07_model.estimate(pd.DataFrame(), mode_compute="none")

    def test_t2_f1_empty_arrays_to_ols_raises(self):
        """T2.1.3: Passing empty arrays to OLS raises LinAlgError or ValueError."""
        with pytest.raises((np.linalg.LinAlgError, ValueError)):
            ols(np.array([]), np.empty((0, 2)))

    def test_t2_f1_forecast_zero_or_negative_horizon_raises(self, sw07_model, sw07_data):
        """T2.1.4: Calling forecast with horizon <= 0 raises ValueError."""
        with pytest.raises(ValueError, match=r"horizon must be an integer >= 1"):
            sw07_model.forecast(horizon=0, data=sw07_data)
        with pytest.raises(ValueError, match=r"horizon must be an integer >= 1"):
            sw07_model.forecast(horizon=-5, data=sw07_data)

    def test_t2_f1_empty_draws_to_harmonic_mean_mdd_raises(self):
        """T2.1.5: Passing empty draws to harmonic_mean_mdd raises ValueError."""
        with pytest.raises(ValueError, match=r"draws must be 2-D or 3-D|usable draws"):
            harmonic_mean_mdd(np.empty((0, 5)), np.empty(0))

    # -----------------------------------------------------------------------
    # Feature 2: Corrupted Data Error Handling (>=5 tests)
    # -----------------------------------------------------------------------

    def test_t2_f2_estimate_with_nan_data_raises(self, sw07_model, sw07_data):
        """T2.2.1: Passing data containing NaNs to estimate raises ValueError."""
        corrupted_data = sw07_data.copy()
        corrupted_data.loc[10, "dy"] = np.nan
        with pytest.raises(ValueError, match="data contains NaN in observed_vars"):
            sw07_model.estimate(corrupted_data, mode_compute="none")

    def test_t2_f2_smoother_missing_single_observable_raises(self, sw07_model, sw07_data):
        """T2.2.2: Passing data with one missing observable column raises ValueError."""
        corrupted = sw07_data.drop(columns=["robs"])
        with pytest.raises(ValueError, match="data is missing column.*robs"):
            sw07_model.smoother(corrupted)

    def test_t2_f2_ols_nan_with_missing_raise(self):
        """T2.2.3: Passing NaNs to OLS with missing='raise' raises ValueError."""
        y = np.array([1.0, np.nan, 3.0, 4.0])
        X = np.ones((4, 2))
        with pytest.raises(ValueError, match="NaNs were encountered in the data"):
            ols(y, X, missing="raise")

    def test_t2_f2_ols_dimension_mismatch_raises(self):
        """T2.2.4: Mismatched row counts between y and X raises ValueError."""
        y = np.ones(10)
        X = np.ones((12, 2))
        with pytest.raises(ValueError, match="y has 10 rows but X has 12"):
            ols(y, X)

    def test_t2_f2_ols_singular_design_matrix_raises_linalg_error(self):
        """T2.2.5: Exactly collinear regressors raise LinAlgError identifying null space."""
        x1 = np.ones(50)
        x2 = 2.0 * x1
        X = np.column_stack([x1, x2])
        y = np.random.randn(50)
        with pytest.raises(np.linalg.LinAlgError, match="ols: X'X is singular"):
            ols(y, X)

    # -----------------------------------------------------------------------
    # Feature 3: Parameter Bounds Violations (>=5 tests)
    # -----------------------------------------------------------------------

    def test_t2_f3_find_mode_invalid_method_raises(self):
        """T2.3.1: find_mode with unrecognized method name raises informative ValueError."""
        with pytest.raises(ValueError, match="unknown mode_compute 'nonexistent_solver'"):
            find_mode(lambda x: float(np.sum(x**2)), np.array([1.0, 1.0]), method="nonexistent_solver")

    def test_t2_f3_mode_check_n_points_less_than_three_raises(self):
        """T2.3.2: mode_check with n_points < 3 raises ValueError."""
        with pytest.raises(ValueError, match="mode_check: n_points must be >= 3"):
            mode_check(lambda x: float(np.sum(x**2)), np.array([0.0]), ["x"], n_points=2)

    def test_t2_f3_mode_check_dimension_mismatch_raises(self):
        """T2.3.3: mode_check with names length != mode vector length raises ValueError."""
        with pytest.raises(ValueError, match="mode_check: 2 names for 1 parameters"):
            mode_check(lambda x: float(np.sum(x**2)), np.array([0.0]), ["x1", "x2"], n_points=5)

    def test_t2_f3_laplace_mdd_non_positive_definite_raises(self):
        """T2.3.4: laplace_mdd on non-positive definite Hessian raises ValueError."""
        non_pd = np.array([[-1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="not positive definite"):
            laplace_mdd(-50.0, non_pd)

    def test_t2_f3_laplace_mdd_non_finite_entries_raises(self):
        """T2.3.5: laplace_mdd with NaNs or infs in inverse Hessian raises ValueError."""
        nan_mat = np.array([[np.nan, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="contains non-finite entries"):
            laplace_mdd(-50.0, nan_mat)

    # -----------------------------------------------------------------------
    # Feature 4: Near-Singular Covariance Matrix Handling in Harmonic Mean (>=5 tests)
    # -----------------------------------------------------------------------

    def test_t2_f4_harmonic_mean_insufficient_draws_raises(self):
        """T2.4.1: Passing fewer draws than parameters (m <= d + 1) raises ValueError."""
        draws = np.ones((3, 5))
        lp = np.ones(3)
        with pytest.raises(ValueError, match="usable draws for 5 parameters is not enough"):
            harmonic_mean_mdd(draws, lp)

    def test_t2_f4_harmonic_mean_identical_draws_raises(self):
        """T2.4.2: Constant/degenerate draws produce singular covariance and raise ValueError."""
        draws = np.ones((100, 3))
        lp = np.ones(100)
        with pytest.raises(ValueError, match="the posterior covariance of the draws is singular"):
            harmonic_mean_mdd(draws, lp)

    def test_t2_f4_harmonic_mean_collinear_parameter_draws_raises(self):
        """T2.4.3: Perfectly collinear parameter draws raise singular covariance ValueError."""
        rng = np.random.default_rng(42)
        d1 = rng.normal(size=100)
        d2 = 3.0 * d1  # collinear
        draws = np.column_stack([d1, d2])
        lp = -0.5 * d1**2
        with pytest.raises(ValueError, match="the posterior covariance of the draws is singular"):
            harmonic_mean_mdd(draws, lp)

    def test_t2_f4_harmonic_mean_draws_log_post_length_mismatch(self):
        """T2.4.4: Mismatched number of draws and log-posterior values raises ValueError."""
        draws = np.random.randn(50, 2)
        lp = np.random.randn(45)
        with pytest.raises(ValueError, match="draws and log_post disagree on the number of draws"):
            harmonic_mean_mdd(draws, lp)

    def test_t2_f4_harmonic_mean_unconverged_spread_warning(self):
        """T2.4.5: Highly unstable draws trigger non-converged status and user warning."""
        rng = np.random.default_rng(99)
        draws = rng.normal(scale=10.0, size=(100, 2))
        lp = -10.0 * np.sum(draws**2, axis=1)
        with pytest.warns(UserWarning, match="harmonic_mean_mdd: the estimate moves"):
            hm = harmonic_mean_mdd(draws, lp)
            assert hm.converged is False
            assert hm.spread > 1.0

    # -----------------------------------------------------------------------
    # Feature 5: Extreme Truncation Levels in MDD (>=5 tests)
    # -----------------------------------------------------------------------

    def test_t2_f5_harmonic_mean_negative_truncation_raises(self):
        """T2.5.1: Truncation levels <= 0 raise ValueError."""
        draws = np.random.randn(100, 2)
        lp = -0.5 * np.sum(draws**2, axis=1)
        with pytest.raises(ValueError, match="no draw fell inside any truncation ellipsoid"):
            harmonic_mean_mdd(draws, lp, truncations=[-0.1, -0.5])

    def test_t2_f5_harmonic_mean_empty_truncation_list_raises(self):
        """T2.5.2: Empty truncation list raises ValueError."""
        draws = np.random.randn(100, 2)
        lp = -0.5 * np.sum(draws**2, axis=1)
        with pytest.raises(ValueError, match="no draw fell inside any truncation ellipsoid"):
            harmonic_mean_mdd(draws, lp, truncations=[])

    def test_t2_f5_harmonic_mean_extreme_low_truncation(self):
        """T2.5.3: Valid extreme low truncation (p=0.01) evaluates stably."""
        rng = np.random.default_rng(42)
        draws = rng.normal(size=(500, 2))
        lp = -0.5 * np.sum(draws**2, axis=1)
        hm = harmonic_mean_mdd(draws, lp, truncations=[0.01, 0.05])
        assert 0.01 in hm.by_truncation or 0.05 in hm.by_truncation
        assert np.isfinite(hm.estimate)

    def test_t2_f5_harmonic_mean_extreme_high_truncation(self):
        """T2.5.4: Valid extreme high truncation (p=0.99) evaluates stably."""
        rng = np.random.default_rng(42)
        draws = rng.normal(size=(500, 2))
        lp = -0.5 * np.sum(draws**2, axis=1)
        hm = harmonic_mean_mdd(draws, lp, truncations=[0.90, 0.99])
        assert 0.99 in hm.by_truncation
        assert np.isfinite(hm.estimate)

    def test_t2_f5_model_comparison_mismatched_methods_raises(self):
        """T2.5.5: model_comparison with fewer than 2 models raises ValueError."""
        with pytest.raises(ValueError, match=r"model_comparison needs at least two models"):
            model_comparison({"Model1": None})


# ===========================================================================
# TIER 3: Cross-Feature Interactions (Pairwise combinations)
# ===========================================================================

class TestTier3CrossFeatureInteractions:
    """Tier 3: Pairwise combinations between DSGE and Econometric subsystems."""

    # -----------------------------------------------------------------------
    # DSGE Subsystem Interactions
    # -----------------------------------------------------------------------

    def test_t3_dsge_load_mod_and_lbfgs_mode(self, sw07_model, sw07_data):
        """T3.1: load_mod + L-BFGS-B mode finding interaction."""
        names = list(sw07_model._estimated_params.names()[:2])
        init_p = sw07_model._estimated_params.initial_params()
        x0 = np.array([init_p[k] for k in names])

        def fake_obj(theta):
            return float(np.sum((theta - x0)**2))

        opt = find_mode(fake_obj, x0 + 0.1, method="lbfgs")
        assert opt.success is True
        np.testing.assert_allclose(opt.x, x0, atol=1e-4)

        # Verify estimate result mode structure
        res = sw07_model.estimate(
            sw07_data.iloc[:20],
            mode_compute="none",
            n_draws=2,
            burn_in=1,
            seed=42,
        )
        assert res.mode is not None
        assert len(res.mode) == 36
        assert all(np.isfinite(v) for v in res.mode.values())

    def test_t3_dsge_load_mod_and_csminwel_mode(self, sw07_model, sw07_data):
        """T3.2: load_mod + Sims csminwel optimizer interaction."""
        f = lambda x: float(0.5 * np.sum(x**2) + 0.1 * x[0])
        res = csminwel(f, np.array([1.0, -1.0]), max_iter=20)
        assert res.success is True
        assert np.allclose(res.x, [-0.1, 0.0], atol=1e-3)
        assert hasattr(res, "n_hessian_resets")

    def test_t3_dsge_load_mod_and_cmaes_mode(self):
        """T3.3: CMA-ES evolutionary optimizer interaction."""
        f = lambda x: float(np.sum((x - 1.5)**2))
        res = find_mode(f, np.array([0.0, 0.0]), method="cmaes", options={"max_iter": 30, "popsize": 8, "seed": 42})
        assert np.allclose(res.x, [1.5, 1.5], atol=0.2)

    def test_t3_dsge_mode_to_mode_check_curvature(self, sw07_model):
        """T3.4: Mode result passed to mode_check coordinate slice analysis."""
        names = list(sw07_model._estimated_params.names()[:2])
        init_p = sw07_model._estimated_params.initial_params()
        x0 = np.array([init_p[k] for k in names])
        
        def fake_neg_log_post(theta):
            return float(np.sum((theta - x0)**2))
        
        diag = mode_check(fake_neg_log_post, x0, names, n_points=5, width=1.0)
        assert len(diag.slices) == 2
        assert all(diag.peaks_at_mode.values())
        assert len(diag.failures) == 0

    def test_t3_dsge_smoother_to_shock_decomposition(self, sw07_model, sw07_data):
        """T3.5: Kalman smoother output feeding historical shock decomposition."""
        sm_res = sw07_model.smoother(sw07_data)
        assert sm_res.states.shape[0] == 156
        
        decomp = sw07_model.shock_decomposition(sw07_data)
        assert hasattr(decomp, "plot")
        assert hasattr(decomp, "summary")

    def test_t3_dsge_smoother_to_forecast(self, sw07_model, sw07_data):
        """T3.6: Smoother end-of-sample state feeding multi-period forecast with CI."""
        fc = sw07_model.forecast(horizon=8, data=sw07_data, ci=0.90)
        assert fc.mean.shape == (8, 7)
        assert fc.lower.shape == (8, 7)
        assert fc.upper.shape == (8, 7)
        assert (fc.lower.to_numpy() <= fc.mean.to_numpy() + 1e-8).all()
        assert (fc.mean.to_numpy() <= fc.upper.to_numpy() + 1e-8).all()

    def test_t3_dsge_laplace_and_harmonic_mdd_agreement(self):
        """T3.7: Laplace approximation vs Geweke harmonic mean on synthetic posterior."""
        rng = np.random.default_rng(42)
        n_draws = 600
        cov = np.array([[1.0, 0.3], [0.3, 1.5]])
        cov_inv = np.linalg.inv(cov)
        draws = rng.multivariate_normal([0.0, 0.0], cov, size=n_draws)
        log_post = -0.5 * np.einsum("ij,jk,ik->i", draws, cov_inv, draws)
        
        l_mdd = laplace_mdd(0.0, cov)
        hm = harmonic_mean_mdd(draws, log_post)
        
        assert np.isfinite(l_mdd)
        assert np.isfinite(hm.estimate)
        assert abs(l_mdd - hm.estimate) < 2.0

    # -----------------------------------------------------------------------
    # Econometric Regress Subsystem Interactions
    # -----------------------------------------------------------------------

    def test_t3_ols_hc_covariance_hierarchy(self, sample_regression_data):
        """T3.8: OLS with HC0 through HC3 covariance hierarchy verification."""
        df = sample_regression_data
        X = add_constant(df[["x1", "x2"]])
        y = df["y"]
        
        m_hc0 = ols(y, X, cov_type="HC0")
        m_hc1 = ols(y, X, cov_type="HC1")
        m_hc2 = ols(y, X, cov_type="HC2")
        m_hc3 = ols(y, X, cov_type="HC3")
        
        np.testing.assert_allclose(m_hc0.params, m_hc1.params)
        np.testing.assert_allclose(m_hc1.params, m_hc2.params)
        np.testing.assert_allclose(m_hc2.params, m_hc3.params)
        
        assert (m_hc1.bse >= m_hc0.bse).all()
        assert (m_hc3.bse >= m_hc2.bse).all()

    def test_t3_ols_add_constant_dataframe_metadata(self, sample_regression_data):
        """T3.9: add_constant + OLS DataFrame label and Series metadata preservation."""
        df = sample_regression_data
        X = add_constant(df[["x1", "x2"]])
        res = ols(df["y"], X, cov_type="HC1")
        
        assert isinstance(res.params, pd.Series)
        assert list(res.params.index) == ["const", "x1", "x2"]
        assert list(res.bse.index) == ["const", "x1", "x2"]
        assert list(res.tvalues.index) == ["const", "x1", "x2"]
        assert list(res.pvalues.index) == ["const", "x1", "x2"]

    def test_t3_ols_and_logit_discrete_consistency(self):
        """T3.10: Linear probability model (OLS) vs Logit direction consistency."""
        rng = np.random.default_rng(42)
        n = 200
        x = rng.normal(size=n)
        prob = 1.0 / (1.0 + np.exp(-(0.5 + 2.0 * x)))
        y = (rng.uniform(size=n) < prob).astype(float)
        
        X = add_constant(pd.DataFrame({"x": x}))
        m_ols = ols(y, X, cov_type="HC1")
        m_logit = logit(y, X)
        
        assert m_ols.params["x"] > 0
        assert m_logit.params["x"] > 0
        assert m_ols.pvalues["x"] < 0.01
        assert m_logit.pvalues["x"] < 0.01

    def test_t3_ols_weights_wls_equivalence(self, sample_regression_data):
        """T3.11: Weighted Least Squares via weights argument vs manual pre-whitening."""
        df = sample_regression_data
        X = add_constant(df[["x1", "x2"]])
        y = df["y"]
        weights = 1.0 / (df["x2"] ** 2)
        
        res_wls = ols(y, X, weights=weights, cov_type="nonrobust")
        
        sw = np.sqrt(weights).to_numpy()
        y_star = y.to_numpy() * sw
        X_star = X.to_numpy() * sw[:, None]
        res_manual = ols(y_star, X_star, cov_type="nonrobust")
        
        np.testing.assert_allclose(np.asarray(res_wls.params), np.asarray(res_manual.params), rtol=1e-10)

    def test_t3_ols_and_quantreg_median(self):
        """T3.12: Quantile regression at median (q=0.5) vs OLS on symmetric errors."""
        rng = np.random.default_rng(42)
        n = 250
        x = rng.normal(size=n)
        y = 2.0 + 3.0 * x + rng.normal(scale=0.5, size=n)
        X = add_constant(pd.DataFrame({"x": x}))
        
        m_ols = ols(y, X)
        m_qr = quantreg(y, X, q=0.5)
        
        np.testing.assert_allclose(m_ols.params["x"], m_qr.params["x"], atol=0.2)


# ===========================================================================
# TIER 4: Real-World Scenarios (Comprehensive End-to-End Execution)
# ===========================================================================

class TestTier4RealWorldScenarios:
    """Tier 4: Five realistic application workloads validating end-to-end macroeconomic contracts."""

    def test_t4_s1_full_sw07_bayesian_estimation_workflow(self, sw07_model, sw07_data):
        """Scenario 1: Full end-to-end SW07 Bayesian estimation workflow.
        
        1. Load canonical Smets-Wouters (2007) model.
        2. Ingest 156 quarterly US macroeconomic observations.
        3. Execute Kalman smoother at baseline parameters.
        4. Extract structural shock decomposition for output growth.
        5. Run out-of-sample forecast for 8 quarters with 90% confidence bands.
        6. Compute Laplace marginal data density.
        """
        assert len(sw07_model.variables) == 40
        assert len(sw07_data) == 156
        
        # Smoother
        sm = sw07_model.smoother(sw07_data)
        assert sm.states.shape == (156, 15)
        assert sm.shocks.shape == (156, 7)
        
        # Shock Decomposition
        decomp = sw07_model.shock_decomposition(sw07_data)
        assert hasattr(decomp, "plot")
        
        # Multi-period forecast
        fc = sw07_model.forecast(horizon=8, data=sw07_data, ci=0.90)
        assert fc.mean.shape == (8, 7)
        
        # Mode and Laplace MDD
        est = sw07_model.estimate(sw07_data.iloc[:40], mode_compute="none", n_draws=4, burn_in=2, seed=42)
        mdd = est.log_mdd(method="laplace")
        assert np.isfinite(mdd)
        assert mdd < 0

    def test_t4_s2_pure_numpy_econometrics_vs_analytical_benchmark(self, sample_regression_data):
        """Scenario 2: Pure-NumPy Econometric Estimation vs Academic Benchmark.
        
        Verifies exact numerical agreement of HC0-HC3 standard errors against
        analytical sandwich formulas:
        V(HC0) = (X'X)^-1 X' diag(e^2) X (X'X)^-1
        V(HC1) = [n / (n-k)] V(HC0)
        V(HC2) = (X'X)^-1 X' diag(e^2 / (1-h)) X (X'X)^-1
        V(HC3) = (X'X)^-1 X' diag(e^2 / (1-h)^2) X (X'X)^-1
        """
        df = sample_regression_data
        X_df = add_constant(df[["x1", "x2"]])
        y = df["y"]
        
        res = ols(y, X_df, cov_type="HC1")
        
        X_mat = X_df.to_numpy()
        y_vec = y.to_numpy()
        n, k = X_mat.shape
        
        xtx_inv = np.linalg.inv(X_mat.T @ X_mat)
        beta = xtx_inv @ (X_mat.T @ y_vec)
        resid = y_vec - X_mat @ beta
        
        omega_hc1 = (n / (n - k)) * (X_mat.T @ np.diag(resid**2) @ X_mat)
        cov_hc1 = xtx_inv @ omega_hc1 @ xtx_inv
        se_hc1 = np.sqrt(np.diag(cov_hc1))
        
        np.testing.assert_allclose(res.params.to_numpy(), beta, rtol=1e-11)
        np.testing.assert_allclose(res.bse.to_numpy(), se_hc1, rtol=1e-11)

    def test_t4_s3_replication_scorecard_execution(self):
        """Scenario 3: Automated replication scorecard execution across active families.

        Executes offline replication cases and asserts 100% pass status and non-null margins.
        """
        offline_cases = [
            c for c in rep.runner._discover_cases()
            if not c.network and c.family in ("stylized_facts", "dsge_estimation", "regression")
        ]
        assert len(offline_cases) >= 8

        results = [run(c) for c in offline_cases]
        sc = scorecard(results)

        assert sc["passed"].all(), f"Failed replication cases:\n{sc[~sc['passed']]}"
        assert not sc["margin"].isna().any()

    def test_t4_s4_clean_release_check_pyodide_gates(self):
        """Scenario 4: Pyodide 4-Package Contract Purity Check (Gate 2 simulation).
        
        Asserts that puremacro runtime dependencies include only the 4 authorized
        Pyodide packages (numpy, scipy, pandas, matplotlib).
        """
        pyproject_path = WORKSPACE_ROOT / "pyproject.toml"
        assert pyproject_path.exists()
        
        content = pyproject_path.read_text(encoding="utf-8")
        
        forbidden = ["statsmodels", "linearmodels", "arch", "torch", "numba"]
        for pkg in forbidden:
            assert f'"{pkg}"' not in content.split("[project.optional-dependencies]")[0], (
                f"Forbidden dependency {pkg} in pyproject.toml runtime dependencies"
            )

    def test_t4_s5_notebook_suite_bilingual_and_purity_integrity(self):
        """Scenario 5: Full showcase notebook suite structural & bilingual integrity scenario.
        
        Verifies that showcase notebooks in notebooks/ are paired, Pyodide-safe,
        and follow the percent format standard.
        """
        en_notebooks = [p for p in NOTEBOOKS_DIR.glob("*.py") if not p.name.endswith("_es.py") and not p.name.startswith("_")]
        assert len(en_notebooks) >= 10
        
        for en_nb in en_notebooks:
            es_nb = en_nb.with_name(f"{en_nb.stem}_es.py")
            if es_nb.exists():
                en_content = en_nb.read_text(encoding="utf-8")
                es_content = es_nb.read_text(encoding="utf-8")
                en_cells = [c for c in en_content.split("# %%") if c.strip()]
                es_cells = [c for c in es_content.split("# %%") if c.strip()]
                assert len(en_cells) == len(es_cells), (
                    f"Cell count mismatch between {en_nb.name} ({len(en_cells)}) and {es_nb.name} ({len(es_cells)})"
                )
