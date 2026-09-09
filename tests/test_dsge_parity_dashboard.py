"""Comprehensive unit test suite for Dynare Parity Dashboard and CLI (puremacro 2.9.0 Tier 3 R4)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.io

from puremacro.dsge import (
    Dynare2ndDR,
    DynareDR,
    ModelParityResult,
    ParityDashboardResult,
    compare_model_to_dynare,
    load_dynare_dr,
    load_dynare_moments,
    load_mod,
    run_parity_suite,
    verify_dynare_parity,
)
from puremacro.dsge.cli import DynareParser, create_parser, run_cli


SW07_PATH = Path("puremacro/dsge/_references/sw07_pfeifer.mod")


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def synthetic_first_order_mat(tmp_path: Path):
    """Create a temporary 1st-order Dynare results MAT file."""
    mat_path = tmp_path / "model1_results.mat"
    n_v, n_x, n_u = 3, 2, 1
    ghx = np.array([[0.7, 0.1], [0.0, 0.5], [0.2, 0.3]], dtype=float)
    ghu = np.array([[0.4], [0.3], [0.1]], dtype=float)
    ys = np.array([1.0, 2.0, 0.5], dtype=float)
    order_var = np.array([[1], [2], [3]], dtype=int)

    dr = {
        "ghx": ghx,
        "ghu": ghu,
        "ys": ys,
        "order_var": order_var,
    }
    oo = {
        "dr": dr,
        "mean": np.array([1.0, 2.0, 0.5]),
        "var": np.array([0.25, 0.16, 0.09]),
        "autocorr": np.array([[0.7, 0.49], [0.5, 0.25], [0.6, 0.36]]),
    }
    M = {
        "endo_names": np.array(["y1", "y2", "y3"]),
        "exo_names": np.array(["e"]),
        "state_var": np.array([[1], [2]]),
    }
    scipy.io.savemat(str(mat_path), {"oo_": oo, "M_": M})
    return {
        "path": mat_path,
        "ghx": ghx,
        "ghu": ghu,
        "ys": ys,
        "order_var": order_var,
        "oo": oo,
        "M": M,
    }


@pytest.fixture
def synthetic_second_order_mat(tmp_path: Path):
    """Create a temporary 2nd-order Dynare results MAT file."""
    mat_path = tmp_path / "model2_results.mat"
    n_v, n_x, n_u = 2, 2, 1
    ghx = np.array([[0.8, 0.0], [0.1, 0.6]], dtype=float)
    ghu = np.array([[0.5], [0.2]], dtype=float)
    ghxx = np.array([[0.05, 0.01, 0.01, 0.02], [0.02, 0.01, 0.01, 0.04]], dtype=float)
    ghs2 = np.array([0.01, 0.02], dtype=float)
    ys = np.array([0.0, 0.0], dtype=float)
    order_var = np.array([[1], [2]], dtype=int)

    dr = {
        "ghx": ghx,
        "ghu": ghu,
        "ghxx": ghxx,
        "ghs2": ghs2,
        "ys": ys,
        "order_var": order_var,
    }
    oo = {"dr": dr, "mean": ys, "var": np.array([0.2, 0.1])}
    M = {
        "endo_names": np.array(["y1", "y2"]),
        "exo_names": np.array(["e"]),
        "state_var": np.array([[1], [2]]),
    }
    scipy.io.savemat(str(mat_path), {"oo_": oo, "M_": M})
    return {"path": mat_path, "ghx": ghx, "ghu": ghu, "ghxx": ghxx, "ghs2": ghs2}


# ===========================================================================
# Suite 1: Dynare Output Parsers (load_dynare_dr, load_dynare_moments)
# ===========================================================================

def test_load_dynare_dr_first_order(synthetic_first_order_mat):
    """Verify first-order decision rules parsing from .mat file."""
    data = synthetic_first_order_mat
    dr = load_dynare_dr(data["path"], order=1)
    assert isinstance(dr, DynareDR)
    assert hasattr(dr, "ghx")
    assert hasattr(dr, "ghu")
    assert hasattr(dr, "ys")
    np.testing.assert_allclose(dr.ghx.to_numpy(), data["ghx"], atol=1e-12)
    np.testing.assert_allclose(dr.ghu.to_numpy(), data["ghu"], atol=1e-12)
    np.testing.assert_allclose(dr.ys.to_numpy(), data["ys"], atol=1e-12)
    assert list(dr.variable_names) == ["y1", "y2", "y3"]
    assert list(dr.state_variables) == ["y1", "y2"]
    assert list(dr.shock_names) == ["e"]


def test_load_dynare_dr_permutation_unpermute(tmp_path: Path):
    """Verify load_dynare_dr unpermutes rows from DR order to declaration order."""
    mat_path = tmp_path / "permuted_results.mat"
    # Declaration variables: y1, y2, y3
    # DR ordering puts variable y3 first, then y1, then y2
    # order_var: [3, 1, 2]
    ghx_dr = np.array([
        [0.3, 0.3],  # row for y3
        [0.1, 0.1],  # row for y1
        [0.2, 0.2],  # row for y2
    ], dtype=float)
    ghu_dr = np.array([[0.3], [0.1], [0.2]], dtype=float)
    order_var = np.array([[3], [1], [2]], dtype=int)

    dr_dict = {"ghx": ghx_dr, "ghu": ghu_dr, "order_var": order_var, "ys": np.zeros(3)}
    scipy.io.savemat(str(mat_path), {"oo_": {"dr": dr_dict}})

    dr = load_dynare_dr(mat_path, order=1, var_names=["y1", "y2", "y3"])
    # After unpermuting, rows should be ordered: y1 (0.1), y2 (0.2), y3 (0.3)
    np.testing.assert_allclose(dr.ghx.loc["y1"].to_numpy(), [0.1, 0.1], atol=1e-12)
    np.testing.assert_allclose(dr.ghx.loc["y2"].to_numpy(), [0.2, 0.2], atol=1e-12)
    np.testing.assert_allclose(dr.ghx.loc["y3"].to_numpy(), [0.3, 0.3], atol=1e-12)
    np.testing.assert_allclose(dr.ghu.loc["y1"].to_numpy(), [0.1], atol=1e-12)
    np.testing.assert_allclose(dr.ghu.loc["y2"].to_numpy(), [0.2], atol=1e-12)
    np.testing.assert_allclose(dr.ghu.loc["y3"].to_numpy(), [0.3], atol=1e-12)


def test_load_dynare_dr_second_order_folded_unfold(tmp_path: Path):
    """Verify folded second-order tensor is unfolded symmetrically to (nv, nx^2)."""
    mat_path = tmp_path / "folded_results.mat"
    # n_v = 2, n_x = 2
    # Folded columns: (0,0), (0,1), (1,1) -> 3 columns
    # Unfolded columns: (0,0), (0,1), (1,0), (1,1) -> 4 columns
    ghx = np.array([[0.5, 0.1], [0.0, 0.5]], dtype=float)
    ghu = np.array([[0.1], [0.2]], dtype=float)
    ghxx_folded = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ], dtype=float)
    ghs2 = np.array([0.01, 0.02], dtype=float)

    dr_dict = {
        "ghx": ghx,
        "ghu": ghu,
        "ghxx": ghxx_folded,
        "ghs2": ghs2,
        "order_var": np.array([[1], [2]]),
    }
    scipy.io.savemat(str(mat_path), {"oo_": {"dr": dr_dict}})

    dr2 = load_dynare_dr(mat_path, order=2, var_names=["y1", "y2"], state_names=["y1", "y2"])
    assert isinstance(dr2, Dynare2ndDR)
    assert dr2.ghxx.shape == (2, 4)

    # For variable 1: folded was [1.0, 2.0, 3.0] -> unfolded should be [1.0, 2.0, 2.0, 3.0]
    expected_row0 = np.array([1.0, 2.0, 2.0, 3.0])
    np.testing.assert_allclose(dr2.ghxx.iloc[0].to_numpy(), expected_row0, atol=1e-12)

    # For variable 2: folded was [4.0, 5.0, 6.0] -> unfolded should be [4.0, 5.0, 5.0, 6.0]
    expected_row1 = np.array([4.0, 5.0, 5.0, 6.0])
    np.testing.assert_allclose(dr2.ghxx.iloc[1].to_numpy(), expected_row1, atol=1e-12)


def test_load_dynare_moments_extraction(synthetic_first_order_mat):
    """Verify load_dynare_moments extracts mean, var, and autocorr."""
    data = synthetic_first_order_mat
    mom = load_dynare_moments(data["path"])
    assert "mean" in mom
    assert "var" in mom
    assert "autocorr" in mom
    np.testing.assert_allclose(mom["mean"], [1.0, 2.0, 0.5], atol=1e-12)
    np.testing.assert_allclose(mom["var"], [0.25, 0.16, 0.09], atol=1e-12)
    assert mom["autocorr"].shape == (3, 2)


def test_load_dynare_dr_missing_file_error(tmp_path: Path):
    """Verify load_dynare_dr raises FileNotFoundError on non-existent path."""
    non_existent = tmp_path / "missing_file.mat"
    with pytest.raises(FileNotFoundError):
        load_dynare_dr(non_existent)


def test_load_dynare_dr_missing_oo_key_error(tmp_path: Path):
    """Verify load_dynare_dr raises KeyError when oo_ structure is missing."""
    bad_mat = tmp_path / "not_dynare.mat"
    scipy.io.savemat(str(bad_mat), {"arbitrary_key": [1, 2, 3]})
    with pytest.raises(KeyError, match=r"(?i)(oo_|dynare)"):
        load_dynare_dr(bad_mat)


def test_load_dynare_dr_dict_input(synthetic_first_order_mat):
    """Verify load_dynare_dr accepts a raw dict directly."""
    data = synthetic_first_order_mat
    raw_mat = scipy.io.loadmat(str(data["path"]), squeeze_me=True, struct_as_record=False)
    dr = load_dynare_dr(raw_mat, order=1)
    assert isinstance(dr, DynareDR)
    np.testing.assert_allclose(dr.ghx.to_numpy(), data["ghx"], atol=1e-12)


# ===========================================================================
# Suite 2: Core Parity Verification Engine (verify_dynare_parity)
# ===========================================================================

def test_verify_dynare_parity_exact_match(synthetic_first_order_mat):
    """Verify exact match yields 100% score and 0.0 max deviation."""
    data = synthetic_first_order_mat
    dr = load_dynare_dr(data["path"], order=1)
    res = verify_dynare_parity(dr, dr, order=1)

    assert isinstance(res, ParityDashboardResult)
    assert res.passed is True
    assert res.score == 100.0
    assert res.max_dev_ghx == 0.0
    assert res.max_dev_ghu == 0.0
    assert res.max_dev_dr == 0.0


def test_verify_dynare_parity_within_tolerance(synthetic_first_order_mat):
    """Verify deviations within tolerance pass with 100% score."""
    data = synthetic_first_order_mat
    dr1 = load_dynare_dr(data["path"], order=1)

    # Perturb by 1e-7 (tolerance is 1e-6)
    ghx_noisy = dr1.ghx.to_numpy() + 1e-7
    ghu_noisy = dr1.ghu.to_numpy() + 1e-7
    dr2 = DynareDR(
        ghx=pd.DataFrame(ghx_noisy, index=list(dr1.variable_names), columns=list(dr1.state_variables)),
        ghu=pd.DataFrame(ghu_noisy, index=list(dr1.variable_names), columns=list(dr1.shock_names)),
        ys=dr1.ys,
        state_variables=dr1.state_variables,
        variable_names=dr1.variable_names,
        shock_names=dr1.shock_names,
    )

    res = verify_dynare_parity(dr1, dr2, order=1)
    assert res.passed is True
    assert res.score == 100.0
    assert res.max_dev_ghx <= 1e-6
    assert res.max_dev_ghu <= 1e-6


def test_verify_dynare_parity_exceeding_tolerance(synthetic_first_order_mat):
    """Verify deviations exceeding tolerance fail."""
    data = synthetic_first_order_mat
    dr1 = load_dynare_dr(data["path"], order=1)

    # Perturb first variable by 1e-3 (tolerance is 1e-6)
    ghx_noisy = dr1.ghx.to_numpy().copy()
    ghx_noisy[0, 0] += 1e-3
    dr2 = DynareDR(
        ghx=pd.DataFrame(ghx_noisy, index=list(dr1.variable_names), columns=list(dr1.state_variables)),
        ghu=dr1.ghu,
        ys=dr1.ys,
        state_variables=dr1.state_variables,
        variable_names=dr1.variable_names,
        shock_names=dr1.shock_names,
    )

    res = verify_dynare_parity(dr1, dr2, order=1)
    assert res.passed is False
    assert res.score < 100.0
    assert res.max_dev_ghx >= 1e-3
    assert res.dr_diff.loc["y1", "status"] == "FAIL"


def test_verify_dynare_parity_second_order(synthetic_second_order_mat):
    """Verify order 2 parity comparisons on ghxx and ghs2."""
    data = synthetic_second_order_mat
    dr2 = load_dynare_dr(data["path"], order=2)
    res = verify_dynare_parity(dr2, dr2, order=2)
    assert res.passed is True
    assert res.score == 100.0
    assert res.max_dev_ghxx == 0.0
    assert res.max_dev_ghs2 == 0.0


def test_verify_dynare_parity_variable_name_mismatch(synthetic_first_order_mat):
    """Verify variable name mismatch results in passed=False."""
    data = synthetic_first_order_mat
    dr1 = load_dynare_dr(data["path"], order=1)
    dr_mismatch = DynareDR(
        ghx=pd.DataFrame(dr1.ghx.to_numpy(), index=["a", "b", "c"], columns=list(dr1.state_variables)),
        ghu=pd.DataFrame(dr1.ghu.to_numpy(), index=["a", "b", "c"], columns=list(dr1.shock_names)),
        ys=pd.Series([0.0, 0.0, 0.0], index=["a", "b", "c"]),
        state_variables=dr1.state_variables,
        variable_names=("a", "b", "c"),
        shock_names=dr1.shock_names,
    )
    res = verify_dynare_parity(dr1, dr_mismatch, order=1)
    assert res.passed is False
    assert res.score == 0.0


def test_verify_dynare_parity_zero_tolerance_strictness(synthetic_first_order_mat):
    """Verify tol=0.0 boundary behavior executes cleanly."""
    data = synthetic_first_order_mat
    dr = load_dynare_dr(data["path"], order=1)
    res = verify_dynare_parity(dr, dr, order=1, tol=0.0)
    assert res.passed is True
    assert res.max_dev_ghx == 0.0


def test_verify_dynare_parity_missing_order_2_fallback(synthetic_first_order_mat):
    """Verify requesting order=2 on order 1 results provides graceful failure."""
    data = synthetic_first_order_mat
    res = verify_dynare_parity(data["path"], data["path"], order=2)
    assert hasattr(res, "passed")
    assert res.passed is False


# ===========================================================================
# Suite 3: ParityDashboardResult & ModelParityResult Presentation Contract
# ===========================================================================

def test_parity_dashboard_result_summary_content():
    """Verify ParityDashboardResult summary formatting and required headers."""
    df_scorecard = pd.DataFrame([
        {"model": "sw07", "order": 1, "status": "PASS", "score": "100.0%", "max_dev_dr": 1e-12, "max_dev_moments": 0.0, "time_s": 0.05},
    ])
    res = ParityDashboardResult(
        scorecard=df_scorecard,
        total_models=1,
        passed_models=1,
        failed_models=0,
        max_dev_ghx=1e-12,
        max_dev_ghu=1e-12,
    )
    summary_text = res.summary()
    assert "Parity Dashboard" in summary_text
    assert "PASS" in summary_text
    assert "Tolerances:" in summary_text
    assert "Scorecard Table:" in summary_text


def test_parity_dashboard_result_scorecard_dataframe():
    """Verify .scorecard() returns a DataFrame."""
    df_scorecard = pd.DataFrame([
        {"model": "sw07", "order": 1, "status": "PASS", "time_s": 0.1},
    ])
    res = ParityDashboardResult(scorecard=df_scorecard)
    sc = res.scorecard()
    assert isinstance(sc, pd.DataFrame)
    assert sc["model"].iloc[0] == "sw07"
    assert isinstance(res.to_frame(), pd.DataFrame)


def test_parity_dashboard_result_to_markdown():
    """Verify Markdown export."""
    df = pd.DataFrame([{"model": "sw07", "status": "PASS"}])
    res = ParityDashboardResult(scorecard=df)
    md = res.to_markdown()
    assert isinstance(md, str)
    assert "sw07" in md
    assert "|" in md


def test_parity_dashboard_result_to_latex():
    """Verify LaTeX tabular export."""
    df = pd.DataFrame([{"model": "sw07", "status": "PASS"}])
    res = ParityDashboardResult(scorecard=df)
    tex = res.to_latex()
    assert isinstance(tex, str)
    assert "\\begin{tabular}" in tex or "tabular" in tex


def test_parity_dashboard_result_to_typst():
    """Verify Typst table export."""
    df = pd.DataFrame([{"model": "sw07", "status": "PASS"}])
    res = ParityDashboardResult(scorecard=df)
    typ = res.to_typst()
    assert isinstance(typ, str)
    assert "#table(" in typ


def test_parity_dashboard_result_plot_headless(synthetic_first_order_mat):
    """Verify .plot() returns Matplotlib figure headlessly."""
    data = synthetic_first_order_mat
    dr = load_dynare_dr(data["path"], order=1)
    res = verify_dynare_parity(dr, dr, order=1)

    fig = res.plot(style="publication")
    assert fig is not None
    plt.close(fig)


def test_model_parity_result_presentation():
    """Verify ModelParityResult presentation methods."""
    m_res = ModelParityResult(
        model_name="test_model",
        order=1,
        n_vars=3,
        n_shocks=1,
        passed=True,
        status="PASS",
        max_dev_ghx=1e-12,
        max_dev_ghu=1e-12,
        score=100.0,
        runtime_sec=0.02,
    )
    assert "test_model" in m_res.summary()
    assert isinstance(m_res.to_frame(), pd.DataFrame)
    assert "test_model" in m_res.to_markdown()
    assert "tabular" in m_res.to_latex()
    assert "#table(" in m_res.to_typst()

    fig = m_res.plot()
    assert fig is not None
    plt.close(fig)


# ===========================================================================
# Suite 4: File Comparison & Batch Suite (compare_model_to_dynare, run_parity_suite)
# ===========================================================================

def test_compare_model_to_dynare_file_runner(tmp_path: Path, synthetic_first_order_mat):
    """Verify compare_model_to_dynare evaluates .mod against .mat."""
    data = synthetic_first_order_mat
    mod_text = """
    var y1 y2 y3;
    varexo e;
    model;
    y1 = 0.7*y1(-1) + 0.1*y2(-1) + 0.4*e;
    y2 = 0.5*y2(-1) + 0.3*e;
    y3 = 0.2*y1(-1) + 0.3*y2(-1) + 0.1*e;
    end;
    """
    mod_file = tmp_path / "toy.mod"
    mod_file.write_text(mod_text)

    res = compare_model_to_dynare(mod_file, data["path"], order=1)
    assert isinstance(res, ParityDashboardResult)
    assert res.passed is True
    assert res.max_dev_ghx <= 1e-10


def test_run_parity_suite_directory_discovery(tmp_path: Path, synthetic_first_order_mat):
    """Verify run_parity_suite discovers and evaluates paired model files."""
    data = synthetic_first_order_mat
    mod_text = """
    var y1 y2 y3;
    varexo e;
    model;
    y1 = 0.7*y1(-1) + 0.1*y2(-1) + 0.4*e;
    y2 = 0.5*y2(-1) + 0.3*e;
    y3 = 0.2*y1(-1) + 0.3*y2(-1) + 0.1*e;
    end;
    """
    # Create matching .mod file alongside the .mat
    mod_file = data["path"].with_suffix(".mod")
    mod_file.write_text(mod_text)

    suite_res = run_parity_suite(tmp_path, order=1)
    assert suite_res.total_models == 1
    assert suite_res.passed_models == 1
    assert suite_res.failed_models == 0
    assert suite_res.passed is True


def test_run_parity_suite_empty_directory(tmp_path: Path):
    """Verify run_parity_suite handles directory with no models cleanly."""
    empty_dir = tmp_path / "empty_models"
    empty_dir.mkdir()
    res = run_parity_suite(empty_dir)
    assert res.total_models == 0
    assert res.passed is True


# ===========================================================================
# Suite 5: CLI Subcommand Integration (puremacro-dynare parity)
# ===========================================================================

def test_cli_create_parser_dual_mode(tmp_path: Path):
    """Verify CLI parser handles both model solving and parity subcommands."""
    parser = create_parser()
    assert isinstance(parser, DynareParser)

    # 1. Model mode
    args_solve = parser.parse_args(["model.mod", "--order", "2"])
    assert args_solve.command == "solve"
    assert args_solve.model == "model.mod"
    assert args_solve.order == 2

    # 2. Parity mode
    args_parity = parser.parse_args(["parity", str(tmp_path), "--order", "1", "--tol", "1e-6"])
    assert args_parity.command == "parity"
    assert args_parity.path == str(tmp_path)
    assert args_parity.order == 1
    assert args_parity.tol == 1e-6


def test_cli_parity_execution_clean_exit(tmp_path: Path, synthetic_first_order_mat):
    """Verify CLI parity subcommand executes and exits 0 on passing models."""
    data = synthetic_first_order_mat
    mod_text = """
    var y1 y2 y3;
    varexo e;
    model;
    y1 = 0.7*y1(-1) + 0.1*y2(-1) + 0.4*e;
    y2 = 0.5*y2(-1) + 0.3*e;
    y3 = 0.2*y1(-1) + 0.3*y2(-1) + 0.1*e;
    end;
    """
    mod_file = data["path"].with_suffix(".mod")
    mod_file.write_text(mod_text)

    exit_code = run_cli(["parity", str(mod_file), "--quiet"])
    assert exit_code == 0


def test_cli_parity_missing_path_error():
    """Verify CLI parity subcommand exits with 1 on missing path."""
    exit_code = run_cli(["parity", "non_existent_path_xyz_12345", "--quiet"])
    assert exit_code == 1


def test_cli_parity_format_export(tmp_path: Path, synthetic_first_order_mat):
    """Verify CLI exports markdown, latex, typst scorecards to --outdir."""
    data = synthetic_first_order_mat
    mod_text = """
    var y1 y2 y3;
    varexo e;
    model;
    y1 = 0.7*y1(-1) + 0.1*y2(-1) + 0.4*e;
    y2 = 0.5*y2(-1) + 0.3*e;
    y3 = 0.2*y1(-1) + 0.3*y2(-1) + 0.1*e;
    end;
    """
    mod_file = data["path"].with_suffix(".mod")
    mod_file.write_text(mod_text)

    out_dir = tmp_path / "cli_parity_export"
    exit_code = run_cli(["parity", str(mod_file), "--format", "all", "--outdir", str(out_dir), "--quiet"])
    assert exit_code == 0
    assert (out_dir / "parity_scorecard.md").is_file()
    assert (out_dir / "parity_scorecard.tex").is_file()
    assert (out_dir / "parity_scorecard.typ").is_file()


def test_cli_invalid_flag_system_exit():
    """Verify unrecognized CLI flags trigger SystemExit."""
    parser = create_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--completely-bogus-flag"])

    with pytest.raises(SystemExit):
        parser.parse_args(["parity", "--completely-bogus-flag"])


# ===========================================================================
# Suite 6: Pfeifer Smets-Wouters (2007) Canonical Benchmark Parity
# ===========================================================================

def test_sw07_pfeifer_canonical_parity_audit(tmp_path: Path):
    """Audit canonical Pfeifer benchmark (sw07_pfeifer.mod) linear parity."""
    assert SW07_PATH.exists(), f"SW07 .mod file missing at {SW07_PATH}"
    m = load_mod(SW07_PATH, order=1)
    dr1 = m.decision_rules()

    assert dr1.ghx.shape == (40, 15)
    assert dr1.ghu.shape == (40, 7)

    # Save exact golden .mat representation of SW07 order 1 solution
    mat_path = tmp_path / "sw07_golden_results.mat"
    dr_dict = {
        "ghx": dr1.ghx.to_numpy(),
        "ghu": dr1.ghu.to_numpy(),
        "ys": dr1.ys.to_numpy(),
        "order_var": np.arange(1, 41)[:, None],
    }
    oo_dict = {
        "dr": dr_dict,
        "mean": dr1.ys.to_numpy(),
        "var": np.ones(40),
    }
    M_dict = {
        "endo_names": np.array(dr1.variable_names),
        "exo_names": np.array(dr1.shock_names),
        "state_var": np.arange(1, 16)[:, None],
    }
    scipy.io.savemat(str(mat_path), {"oo_": oo_dict, "M_": M_dict})

    # Evaluate parity
    res = compare_model_to_dynare(SW07_PATH, mat_path, order=1)
    assert res.passed is True
    assert res.max_dev_ghx <= 1e-12
    assert res.max_dev_ghu <= 1e-12


def test_sw07_pfeifer_second_order_parity(tmp_path: Path):
    """Audit canonical Pfeifer benchmark (sw07_pfeifer.mod) second-order parity."""
    assert SW07_PATH.exists(), f"SW07 .mod file missing at {SW07_PATH}"
    m = load_mod(SW07_PATH, order=2)
    dr2 = m.decision_rules()

    assert dr2.ghxx.shape == (40, 225)
    assert dr2.ghs2.shape == (40,)

    # Save golden .mat representation
    mat_path = tmp_path / "sw07_order2_golden.mat"
    dr_dict = {
        "ghx": dr2.ghx.to_numpy(),
        "ghu": dr2.ghu.to_numpy(),
        "ghxx": dr2.ghxx.to_numpy(),
        "ghs2": dr2.ghs2.to_numpy()[:, None],
        "ys": dr2.ys.to_numpy(),
        "order_var": np.arange(1, 41)[:, None],
    }
    oo_dict = {"dr": dr_dict, "mean": dr2.ys.to_numpy()}
    M_dict = {
        "endo_names": np.array(dr2.variable_names),
        "exo_names": np.array(dr2.shock_names),
    }
    scipy.io.savemat(str(mat_path), {"oo_": oo_dict, "M_": M_dict})

    res = compare_model_to_dynare(SW07_PATH, mat_path, order=2)
    assert res.passed is True
    assert res.max_dev_ghxx <= 1e-10


def test_callable_dataframe_copy_and_deepcopy():
    """Verify CallableDataFrame retains callability after shallow and deep copies."""
    import copy
    from puremacro.dsge._results import CallableDataFrame

    cdf = CallableDataFrame([{"model": "test", "score": "100.0%"}])
    assert callable(cdf)
    assert isinstance(cdf(), pd.DataFrame)
    assert cdf()["model"].iloc[0] == "test"

    # Shallow copy via DataFrame.copy()
    cp = cdf.copy()
    assert callable(cp)
    assert isinstance(cp, CallableDataFrame)
    assert cp()["model"].iloc[0] == "test"

    # Deepcopy via copy.deepcopy()
    dcp = copy.deepcopy(cdf)
    assert callable(dcp)
    assert isinstance(dcp, CallableDataFrame)
    assert dcp()["model"].iloc[0] == "test"


def test_run_parity_suite_corrupted_model_isolation(tmp_path: Path):
    """Verify run_parity_suite isolates corrupted models and computes safe max deviations."""
    # Create valid model pair
    valid_mod = tmp_path / "valid.mod"
    valid_mod.write_text("var y; varexo e; model; y = 0.5*y(-1) + e; end;")
    valid_mat = tmp_path / "valid_results.mat"
    scipy.io.savemat(
        str(valid_mat),
        {
            "oo_": {
                "dr": {
                    "ghx": np.array([[0.5]]),
                    "ghu": np.array([[1.0]]),
                    "ys": np.array([[0.0]]),
                    "order_var": np.array([[1]]),
                }
            },
            "M_": {
                "endo_names": np.array(["y"]),
                "exo_names": np.array(["e"]),
            },
        },
    )

    # Create corrupted model pair
    bad_mod = tmp_path / "corrupted.mod"
    bad_mod.write_text("var z; model; z = 2*z(1); end;")
    bad_mat = tmp_path / "corrupted_results.mat"
    bad_mat.write_bytes(b"corrupted_mat_bytes")

    res = run_parity_suite(tmp_path)
    assert res.total_models == 2
    assert res.passed_models == 1
    assert res.failed_models == 1
    assert res.passed is False
    assert not np.isnan(res.max_dev_ghx)
    assert res.max_dev_ghx >= 0.0

    # Verify failed model recorded in results with error in details
    failed_r = [r for r in res.results if not r.passed][0]
    assert failed_r.model_name == "corrupted"
    assert "error" in failed_r.details


def test_verify_dynare_parity_shock_dimension_guard():
    """Verify shock dimension mismatch raises clean ParityDashboardResult failure."""
    from puremacro.dsge._results import DynareDR

    # pm has 2 shocks, dyn has 1 shock
    dr_pm = DynareDR(
        ghx=pd.DataFrame([[0.5]], index=["y"], columns=["y"]),
        ghu=pd.DataFrame([[1.0, 0.2]], index=["y"], columns=["e1", "e2"]),
        ys=pd.Series([0.0], index=["y"]),
        state_variables=("y",),
        variable_names=("y",),
        shock_names=("e1", "e2"),
    )
    dr_dyn = DynareDR(
        ghx=pd.DataFrame([[0.5]], index=["y"], columns=["y"]),
        ghu=pd.DataFrame([[1.0]], index=["y"], columns=["e1"]),
        ys=pd.Series([0.0], index=["y"]),
        state_variables=("y",),
        variable_names=("y",),
        shock_names=("e1",),
    )

    res = verify_dynare_parity(dr_pm, dr_dyn)
    assert res.passed is False
    assert res.score == 0.0
    assert "Shock shape mismatch" in res.details.get("error", "")


def test_verify_dynare_parity_independent_moment_tolerances():
    """Verify moment mean and var tolerances are checked independently."""
    from puremacro.dsge._results import DynareDR

    dr = DynareDR(
        ghx=pd.DataFrame([[0.5]], index=["y"], columns=["y"]),
        ghu=pd.DataFrame([[1.0]], index=["y"], columns=["e"]),
        ys=pd.Series([0.0], index=["y"]),
        state_variables=("y",),
        variable_names=("y",),
        shock_names=("e",),
    )

    class MockModelWithMoments:
        decision_rules = lambda self, **kw: dr

        def theoretical_moments(self, **kw):
            from dataclasses import make_dataclass
            Mom = make_dataclass("Mom", [("mean", np.ndarray), ("variance", np.ndarray)])
            return Mom(mean=np.array([0.00005]), variance=np.array([1.0005]))

    mock_m = MockModelWithMoments()
    dyn_mom = {"mean": np.array([0.0]), "var": np.array([1.0])}
    dyn_dict = {
        "oo_": {
            "dr": {
                "ghx": np.array([[0.5]]),
                "ghu": np.array([[1.0]]),
                "ys": np.array([[0.0]]),
                "order_var": np.array([[1]]),
            },
            "mean": dyn_mom["mean"],
            "var": dyn_mom["var"],
        },
        "M_": {"endo_names": np.array(["y"]), "exo_names": np.array(["e"])},
    }
    # dev_mean = 5e-5, dev_var = 5e-4
    # Case 1: tol allows mean (1e-4) but fails var (1e-4)
    res1 = verify_dynare_parity(
        mock_m,
        dyn_dict,
        tol={"ghx": 1e-4, "ghu": 1e-4, "mean": 1e-4, "var": 1e-4},
    )
    assert res1.passed is False

    # Case 2: tol allows both mean and var
    res2 = verify_dynare_parity(
        mock_m,
        dyn_dict,
        tol={"ghx": 1e-4, "ghu": 1e-4, "mean": 1e-4, "var": 1e-3},
    )
    assert res2.passed is True


def test_load_dynare_dr_ys_declaration_order_non_identity(tmp_path: Path):
    """Verify load_dynare_dr leaves ys in declaration order when order_var is non-identity."""
    mat_path = tmp_path / "permuted_ys.mat"
    # Declaration order: [y1, y2, y3]
    # DR order: y3 (idx 3), y1 (idx 1), y2 (idx 2) -> order_var = [3, 1, 2]
    ghx_dr = np.array([[0.3], [0.1], [0.2]])
    ghu_dr = np.array([[0.3], [0.1], [0.2]])
    order_var = np.array([[3], [1], [2]], dtype=int)
    ys_declaration = np.array([10.0, 20.0, 30.0])  # official Dynare stores ys in declaration order

    scipy.io.savemat(
        str(mat_path),
        {
            "oo_": {
                "dr": {
                    "ghx": ghx_dr,
                    "ghu": ghu_dr,
                    "order_var": order_var,
                    "ys": ys_declaration,
                }
            },
            "M_": {
                "endo_names": np.array(["y1", "y2", "y3"]),
                "exo_names": np.array(["e"]),
            },
        },
    )

    dr = load_dynare_dr(mat_path, order=1)
    # ghx should be unpermuted to declaration order
    np.testing.assert_allclose(dr.ghx.loc["y1"].to_numpy(), [0.1], atol=1e-12)
    np.testing.assert_allclose(dr.ghx.loc["y2"].to_numpy(), [0.2], atol=1e-12)
    np.testing.assert_allclose(dr.ghx.loc["y3"].to_numpy(), [0.3], atol=1e-12)
    # ys should remain in declaration order
    assert dr.ys.loc["y1"] == 10.0
    assert dr.ys.loc["y2"] == 20.0
    assert dr.ys.loc["y3"] == 30.0


def test_load_dynare_dr_0d_scalar_arrays(tmp_path: Path):
    """Verify load_dynare_dr successfully handles 0D scalar arrays for 1x1 models."""
    mat_path = tmp_path / "scalar_1x1.mat"
    scipy.io.savemat(
        str(mat_path),
        {
            "oo_": {
                "dr": {
                    "ghx": np.array([[0.75]]),
                    "ghu": np.array([[0.25]]),
                    "ys": np.array([[1.5]]),
                    "order_var": np.array([[1]]),
                }
            },
            "M_": {
                "endo_names": np.array(["y"]),
                "exo_names": np.array(["e"]),
                "state_var": np.array([[1]]),
            },
        },
    )

    dr = load_dynare_dr(mat_path, order=1)
    assert dr.ghx.shape == (1, 1)
    assert dr.ghu.shape == (1, 1)
    assert dr.ghx.iloc[0, 0] == 0.75
    assert dr.ghu.iloc[0, 0] == 0.25
    assert dr.ys.iloc[0] == 1.5
