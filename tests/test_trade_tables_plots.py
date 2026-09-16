"""Comprehensive Unit and Parity Tests for puremacro.trade Tables and Plots.

Tests:
1. Fast synthetic unit tests (< 0.1s):
   - generate_selected_country_table DataFrame multi-index and column structure.
   - generate_mean_by_scenario_table structure and string formatting.
   - generate_weighted_mean_by_scenario_table structure.
   - plot_country_impacts, plot_scenario_distributions, plot_tariff_escalation_curve,
     plot_terms_of_trade_vs_welfare Figure generation, axes counts, and save_path creation.
2. Exact LaTeX parity tests vs reference benchmark files:
   - Byte-for-byte match against computation/7_TIO_77c_vf/selected_country_impacts.tex.
   - Byte-for-byte match against computation/7_TIO_77c_vf/mean_by_scenario.tex.
   - Byte-for-byte match against computation/7_TIO_77c_vf/weighted_mean_by_scenario.tex.
   - Byte-for-byte match against computation/7_TIO_77c_vf/selected_country_impacts_with_ROW.tex.
   - Exact numerical parity on weighted_mean_by_scenario.tex.
3. Clean LaTeX generation tests:
   - Assert that legacy_compat=False produces valid 7-column headers (& Base).
4. Strict warnings safety under pytest -W error.

Conforms strictly to the puremacro Pyodide runtime contract.
"""
from __future__ import annotations

import os
from pathlib import Path
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.io as sio
from matplotlib.figure import Figure

from puremacro.trade._results import (
    GearyKhamisResult,
    ScenarioBatchResult,
    TradeCalibrationResult,
    TradeEquilibriumResult,
)
from puremacro.trade.data import CANONICAL_COUNTRY_CODES
from puremacro.trade.geary_khamis import compute_geary_khamis
from puremacro.trade.plot import (
    plot_country_impacts,
    plot_scenario_distributions,
    plot_tariff_escalation_curve,
    plot_terms_of_trade_vs_welfare,
)
from puremacro.trade.tables import (

    export_latex_tables,
    generate_mean_by_scenario_table,
    generate_selected_country_table,
    generate_weighted_mean_by_scenario_table,
    to_latex_mean_by_scenario,
    to_latex_mean_by_scenario_table,
    to_latex_selected_country,
    to_latex_selected_country_table,
    to_latex_selected_country_with_row,
    to_latex_weighted_mean_by_scenario,
    to_latex_weighted_mean_by_scenario_table,
)

from conftest import mat_file_is_readable


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def close_figures():
    """Ensure all figures are closed after each test to prevent memory leaks."""
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def matlab_benchmark_dir() -> Path | None:
    """Discover directory containing reference MATLAB .mat and .tex files."""
    candidates = [
        Path(os.environ.get("IO_COMPUTATION_DIR", "")),
        Path(__file__).resolve().parents[2] / "IO" / "computation" / "7_TIO_77c_vf",
        Path(__file__).resolve().parents[1] / "IO" / "computation" / "7_TIO_77c_vf",
        Path.cwd() / "computation" / "7_TIO_77c_vf",
        Path.cwd() / "IO" / "computation" / "7_TIO_77c_vf",
    ]
    for c in candidates:
        if (c.exists() and mat_file_is_readable(c / "results_77c_11s_base.mat")
                and mat_file_is_readable(c / "data_77c_11s.mat")):
            return c
    return None


@pytest.fixture(scope="module")
def synthetic_batch_result() -> ScenarioBatchResult:
    """Fast synthetic 2-country 2-sector batch result for smoke testing (< 0.05s)."""
    nc, ns = 2, 2
    codes = ("CAN", "USA")
    s_codes = ("S01", "S02")

    def _make_eq():
        return TradeEquilibriumResult(
            x_sol=np.zeros(2 * ns * nc + 2 * nc + nc + nc - 1),
            p_sol=np.ones((1, ns, nc)),
            y_sol=np.ones((1, ns, nc)) * 50.0,
            r_sol=np.ones((1, 1, nc)),
            w_sol=np.ones((1, 1, nc)),
            T_sol=np.ones((1, 1, nc)) * 5.0,
            XN_sol=np.array([1.0]),
            c_sol=np.ones((1, 3, nc)) * 20.0,
            pfd_sol=np.ones((1, 3, nc)),
            terms_of_trade=np.array([1.02, 0.98]),
            country_codes=codes,
            sector_codes=s_codes,
        )

    scens = ["base", "t10", "t10_25"]
    eq_dict = {s: _make_eq() for s in scens}
    gk_dict = {}
    for s in scens:
        growth = np.array([-0.5, -0.8]) if s != "base" else np.array([0.0, 0.0])
        gk_dict[s] = GearyKhamisResult(
            pi=np.ones(3),
            ppp=np.ones(2),
            real_gdp=np.array([100.0, 200.0]),
            nominal_gdp=np.array([100.0, 200.0]),
            gdp_growth=growth,
            country_codes=codes,
        )

    return ScenarioBatchResult(
        scenarios=eq_dict,
        geary_khamis=gk_dict,
        baseline_scenario="base",
        country_codes=codes,
        sector_codes=s_codes,
    )


@pytest.fixture(scope="module")
def benchmark_batch_result(matlab_benchmark_dir: Path | None) -> ScenarioBatchResult | None:
    """Full 77-country benchmark batch result loaded from reference .mat files."""
    if matlab_benchmark_dir is None:
        return None

    base_mat = sio.loadmat(str(matlab_benchmark_dir / "results_77c_11s_base.mat"))
    base_eq = TradeEquilibriumResult(
        x_sol=base_mat["xx_sol"].flatten(),
        p_sol=base_mat["p_sol"],
        y_sol=base_mat["ytot_sol"],
        r_sol=base_mat["r_sol"],
        w_sol=base_mat["w_sol"],
        T_sol=base_mat["T_sol"],
        XN_sol=base_mat["XN_sol"].flatten(),
        c_sol=base_mat["c_sol"],
        pfd_sol=base_mat["pfd_sol"],
        country_codes=CANONICAL_COUNTRY_CODES,
    )

    scens = ["t10", "t10_25", "t10_54", "t10_125", "t10_145"]
    res_dict = {"base": base_eq}
    gk_dict = {}

    for s in scens:
        mat = sio.loadmat(str(matlab_benchmark_dir / f"results_77c_11s_{s}.mat"))
        eq = TradeEquilibriumResult(
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
        res_dict[s] = eq
        gk = compute_geary_khamis(eq, base_eq, matlab_compat=True)
        gk_dict[s] = gk

    gk_base = compute_geary_khamis(base_eq, base_eq, matlab_compat=True)
    object.__setattr__(gk_base, "gdp_growth", np.zeros(77, dtype=float))
    object.__setattr__(gk_base, "real_gdp_growth", np.zeros(77, dtype=float))
    gk_dict["base"] = gk_base

    return ScenarioBatchResult(
        scenarios=res_dict,
        geary_khamis=gk_dict,
        baseline_scenario="base",
        country_codes=CANONICAL_COUNTRY_CODES,
    )


# ---------------------------------------------------------------------------
# 1. Fast Synthetic Table Tests
# ---------------------------------------------------------------------------

def test_generate_selected_country_table_synthetic(synthetic_batch_result):
    tbl = generate_selected_country_table(synthetic_batch_result, countries=["CAN", "USA"])
    assert isinstance(tbl, pd.DataFrame)
    assert tbl.index.names == ["Section", "Country"]
    assert "t10" in tbl.columns
    assert "t10_25" in tbl.columns
    assert "Base" in tbl.columns
    assert len(tbl) == 6  # 3 sections x 2 countries


def test_generate_mean_by_scenario_table_synthetic(synthetic_batch_result):
    tbl = generate_mean_by_scenario_table(synthetic_batch_result)
    assert isinstance(tbl, pd.DataFrame)
    assert tbl.index.name == "Variable"
    assert "GDP growth (%)" in tbl.index
    assert "Inflation (%)" in tbl.index
    assert "XN over GDP (%)" in tbl.index
    assert "t10" in tbl.columns
    assert "t10_25" in tbl.columns


def test_generate_weighted_mean_by_scenario_table_synthetic(synthetic_batch_result):
    tbl = generate_weighted_mean_by_scenario_table(synthetic_batch_result)
    assert isinstance(tbl, pd.DataFrame)
    assert tbl.index.name == "Variable"
    assert "GDP growth (%)" in tbl.index


def test_export_latex_tables_file_writing(synthetic_batch_result, tmp_path):
    out_dict = export_latex_tables(synthetic_batch_result, output_dir=tmp_path, prefix="test_")
    assert isinstance(out_dict, dict)
    assert "selected_country_impacts" in out_dict
    assert "mean_by_scenario" in out_dict
    assert "weighted_mean_by_scenario" in out_dict
    for p in out_dict.values():
        assert p.exists()
        assert p.stat().st_size > 0


# ---------------------------------------------------------------------------
# 2. Fast Synthetic Plot Tests
# ---------------------------------------------------------------------------

def test_plot_country_impacts_synthetic(synthetic_batch_result):
    fig = plot_country_impacts(synthetic_batch_result, countries=["CAN", "USA"])
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 3


def test_plot_country_impacts_custom_ax(synthetic_batch_result):
    fig, ax = plt.subplots()
    ret_fig = plot_country_impacts(synthetic_batch_result, countries=["CAN", "USA"], ax=ax)
    assert ret_fig is fig
    assert len(ax.patches) > 0


def test_plot_scenario_distributions_synthetic(synthetic_batch_result):
    fig = plot_scenario_distributions(synthetic_batch_result, metric="real_gdp_growth")
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1


def test_plot_tariff_escalation_curve_synthetic(synthetic_batch_result):
    fig = plot_tariff_escalation_curve(synthetic_batch_result, partner="CAN")
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1
    ax = fig.axes[0]
    assert len(ax.lines) >= 2


def test_plot_terms_of_trade_vs_welfare_synthetic(synthetic_batch_result):
    fig = plot_terms_of_trade_vs_welfare(synthetic_batch_result, scenario="t10")
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1


def test_plot_save_path_functionality(synthetic_batch_result, tmp_path):
    target_img = tmp_path / "impacts.png"
    fig = plot_country_impacts(synthetic_batch_result, save_path=target_img)
    assert target_img.exists()
    assert target_img.stat().st_size > 0


# ---------------------------------------------------------------------------
# 3. Exact Benchmark Parity Tests (Matching headlinePaper LaTeX Files)
# ---------------------------------------------------------------------------

def test_selected_country_impacts_exact_byte_match(
    benchmark_batch_result, matlab_benchmark_dir: Path | None
):
    """Assert byte-for-byte exact equality with selected_country_impacts.tex."""
    if benchmark_batch_result is None or matlab_benchmark_dir is None:
        pytest.skip("Benchmark reference directory not available.")

    ref_file = matlab_benchmark_dir / "selected_country_impacts.tex"
    if not ref_file.exists():
        pytest.skip(f"Reference file {ref_file} not found.")

    expected_tex = ref_file.read_text(encoding="utf-8")
    generated_tex = to_latex_selected_country(
        benchmark_batch_result, countries=["CAN", "CHN", "EU_", "MEX", "USA"], legacy_compat=True
    )

    assert generated_tex == expected_tex


def test_mean_by_scenario_exact_byte_match(
    benchmark_batch_result, matlab_benchmark_dir: Path | None
):
    """Assert byte-for-byte exact equality with mean_by_scenario.tex."""
    if benchmark_batch_result is None or matlab_benchmark_dir is None:
        pytest.skip("Benchmark reference directory not available.")

    ref_file = matlab_benchmark_dir / "mean_by_scenario.tex"
    if not ref_file.exists():
        pytest.skip(f"Reference file {ref_file} not found.")

    expected_tex = ref_file.read_text(encoding="utf-8")
    generated_tex = to_latex_mean_by_scenario(benchmark_batch_result, legacy_compat=True)

    assert generated_tex == expected_tex


def test_weighted_mean_by_scenario_exact_byte_match(
    benchmark_batch_result, matlab_benchmark_dir: Path | None
):
    """Assert byte-for-byte exact equality with weighted_mean_by_scenario.tex."""
    if benchmark_batch_result is None or matlab_benchmark_dir is None:
        pytest.skip("Benchmark reference directory not available.")

    ref_file = matlab_benchmark_dir / "weighted_mean_by_scenario.tex"
    if not ref_file.exists():
        pytest.skip(f"Reference file {ref_file} not found.")

    expected_tex = ref_file.read_text(encoding="utf-8")
    generated_tex = to_latex_weighted_mean_by_scenario(benchmark_batch_result, legacy_compat=True)

    assert generated_tex == expected_tex


def test_selected_country_impacts_with_row_exact_byte_match(
    benchmark_batch_result, matlab_benchmark_dir: Path | None
):
    """Assert byte-for-byte exact equality with selected_country_impacts_with_ROW.tex."""
    if benchmark_batch_result is None or matlab_benchmark_dir is None:
        pytest.skip("Benchmark reference directory not available.")

    ref_file = matlab_benchmark_dir / "selected_country_impacts_with_ROW.tex"
    if not ref_file.exists():
        pytest.skip(f"Reference file {ref_file} not found.")

    expected_tex = ref_file.read_text(encoding="utf-8")
    generated_tex = to_latex_selected_country_with_row(benchmark_batch_result, legacy_compat=True)

    assert generated_tex == expected_tex


def test_weighted_mean_by_scenario_numerical_parity(
    benchmark_batch_result, matlab_benchmark_dir: Path | None
):
    """Assert numerical values match weighted_mean_by_scenario.tex."""
    if benchmark_batch_result is None:
        pytest.skip("Benchmark reference directory not available.")

    tbl = generate_weighted_mean_by_scenario_table(benchmark_batch_result)
    assert tbl.loc["GDP growth (%)", "t10"] == "-0.705 (0.199)"
    assert tbl.loc["GDP growth (%)", "t10_25"] == "-1.197 (0.317)"
    assert tbl.loc["GDP growth (%)", "t10_54"] == "-1.628 (0.417)"
    assert tbl.loc["GDP growth (%)", "t10_125"] == "-2.795 (0.667)"
    assert tbl.loc["GDP growth (%)", "t10_145"] == "-3.162 (0.742)"

    assert tbl.loc["Inflation (%)", "t10"] == "0.092 (0.261)"
    assert tbl.loc["Inflation (%)", "t10_25"] == "0.164 (0.440)"
    assert tbl.loc["Inflation (%)", "t10_54"] == "0.235 (0.606)"
    assert tbl.loc["Inflation (%)", "t10_125"] == "0.458 (1.096)"
    assert tbl.loc["Inflation (%)", "t10_145"] == "0.539 (1.265)"

    assert tbl.loc["XN over GDP (%)", "t10"] == "1.492 (5.496)"
    assert tbl.loc["XN over GDP (%)", "t10_25"] == "1.491 (5.504)"
    assert tbl.loc["XN over GDP (%)", "t10_54"] == "1.490 (5.515)"
    assert tbl.loc["XN over GDP (%)", "t10_125"] == "1.489 (5.547)"
    assert tbl.loc["XN over GDP (%)", "t10_145"] == "1.489 (5.557)"


def test_selected_country_impacts_clean_latex(benchmark_batch_result):
    """Verify that legacy_compat=False produces valid column headers (& Base)."""
    if benchmark_batch_result is None:
        pytest.skip("Benchmark reference directory not available.")

    clean_tex = to_latex_selected_country(benchmark_batch_result, legacy_compat=False)
    assert r"\begin{tabular}{lrrrrrr}" in clean_tex
    assert r"Country & t10 & t10\_25 & t10\_54 & t10\_125 & t10\_145 & Base \\" in clean_tex
