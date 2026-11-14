"""Tests for CompositeTariffScenario, combine_scenarios, ScenarioComparisonResult, and summarize_scenario_batch."""
from __future__ import annotations

from dataclasses import replace
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest

from puremacro.study.robust import RobustPolicyComparison
from puremacro.trade import (
    CompositeTariffScenario,
    ScenarioBatchResult,
    ScenarioComparisonResult,
    TariffScenario,
    build_tariff_matrices,
    calibrate_trade_model,
    combine_scenarios,
    compute_effective_rate_of_protection,
    compute_hicksian_welfare,
    run_scenario,
    run_scenario_batch,
    summarize_scenario_batch,
)
from puremacro.trade.data import synthetic_codes, synthetic_icio_table
from puremacro.trade.solver import solve_keller_pac


def _calib_3x2():
    table = synthetic_icio_table(3, 2, nfd=3, seed=42)
    countries, sectors = synthetic_codes(3, 2)
    return calibrate_trade_model(
        table, ns=2, nc=3, nfd=3, country_codes=countries, sector_codes=sectors
    )


def test_composite_retaliation_matrices_and_importers():
    calib = _calib_3x2()
    countries, sectors = calib.country_codes, calib.sector_codes
    c01_tariffs = TariffScenario("C01", {("C02", "*"): 0.15}, name="c01_hits_c02")
    c02_tariffs = TariffScenario("C02", {("C01", "S01"): 0.25}, name="c02_retaliates")
    mixed = combine_scenarios([c01_tariffs, c02_tariffs], name="trade_war")

    assert isinstance(mixed, CompositeTariffScenario)
    assert mixed.importers == ("C01", "C02")

    rates = mixed.to_rate_tensor(countries, sectors)
    assert rates.shape == (3, 3, 2)
    # C02 -> C01 taxed 15% across all sectors
    assert np.allclose(rates[1, 0, :], [0.15, 0.15])
    # C01 -> C02 taxed 25% on S01 only
    assert np.allclose(rates[0, 1, :], [0.25, 0.0])
    # C03 and domestic diagonals untouched
    assert np.allclose(rates[2, :, :], 0.0)
    assert np.allclose(rates[:, 2, :], 0.0)
    for k in range(3):
        assert np.allclose(rates[k, k, :], 0.0)

    tau, tau_fd = build_tariff_matrices(calib, mixed)
    assert tau.shape == (2, 3, 2, 3)
    assert tau_fd.shape == (2, 3, 3, 3)
    assert np.allclose(tau[:, 1, :, 0], 1.15)
    assert np.allclose(tau_fd[:, 1, :, 0], 1.15)
    assert np.allclose(tau[0, 0, :, 1], 1.25)
    assert np.allclose(tau[1, 0, :, 1], 1.0)
    for k in range(3):
        assert np.allclose(tau[:, k, :, k], 1.0)
        assert np.allclose(tau_fd[:, k, :, k], 1.0)

    df_nz = mixed.to_frame(countries, sectors)
    assert len(df_nz) == 3
    assert set(df_nz["importer"]) == {"C01", "C02"}


def test_sparse_zero_fta_carveout_override_and_conflicts():
    countries, sectors = ["C01", "C02", "C03"], ["S01", "S02"]
    universal = TariffScenario(
        "C01",
        {("C02", "*"): 0.20, ("C03", "*"): 0.15},
        name="broad_tariff",
    )
    fta = TariffScenario("C01", {("C02", "S01"): 0.0}, name="fta_s01")
    mixed = combine_scenarios([universal, fta], name="broad_plus_fta", on_conflict="last_wins")

    rates = mixed.to_rate_tensor(countries, sectors)
    # Explicit 0.0 carve-out must override prior 0.20 on S01 while keeping S02=0.20
    assert rates[1, 0, 0] == pytest.approx(0.0)
    assert rates[1, 0, 1] == pytest.approx(0.20)
    assert np.allclose(rates[2, 0, :], [0.15, 0.15])

    conf = mixed.conflicts(countries, sectors)
    assert len(conf) == 1
    row = conf.iloc[0]
    assert (row["exporter"], row["importer"], row["sector"]) == ("C02", "C01", "S01")
    assert row["sources"] == {"broad_tariff": 0.20, "fta_s01": 0.0}
    assert row["resolved_rate"] == pytest.approx(0.0)

    # Array tariffs only mark nonzero cells as written so orthogonal arrays compose without conflict
    arr_s01 = np.array([[0.0, 0.0], [0.10, 0.0], [0.0, 0.0]])
    s_arr = TariffScenario("C01", arr_s01, name="arr_s01")
    s_map = TariffScenario("C01", {("C02", "S02"): 0.25}, name="map_s02")
    disjoint = combine_scenarios([s_arr, s_map], name="disjoint", on_conflict="error")
    assert disjoint.conflicts(countries, sectors).empty
    assert np.allclose(disjoint.to_rate_tensor(countries, sectors)[1, 0, :], [0.10, 0.25])


def test_composite_conflict_modes_nested_and_validation():
    countries, sectors = ["C01", "C02", "C03"], ["S01", "S02"]
    s1 = TariffScenario("C01", {("C02", "*"): 0.20}, name="layer1")
    s2 = TariffScenario("C01", {("C02", "S01"): 0.0, ("C02", "S02"): 0.35}, name="layer2")

    assert np.allclose(
        combine_scenarios([s1, s2], on_conflict="last_wins").to_rate_tensor(countries, sectors)[1, 0, :],
        [0.0, 0.35],
    )
    assert np.allclose(
        combine_scenarios([s1, s2], on_conflict="first_wins").to_rate_tensor(countries, sectors)[1, 0, :],
        [0.20, 0.20],
    )
    assert np.allclose(
        combine_scenarios([s1, s2], on_conflict="max").to_rate_tensor(countries, sectors)[1, 0, :],
        [0.20, 0.35],
    )
    assert np.allclose(
        combine_scenarios([s1, s2], on_conflict="sum").to_rate_tensor(countries, sectors)[1, 0, :],
        [0.20, 0.55],
    )

    err_comp = combine_scenarios([s1, s2], name="strict", on_conflict="error")
    # Inspecting .conflicts() must not raise even in on_conflict="error"
    conf_df = err_comp.conflicts(countries, sectors)
    assert len(conf_df) == 2
    assert conf_df["resolved_rate"].isna().all()
    with pytest.raises(ValueError, match="Tariff conflict"):
        err_comp.to_rate_tensor(countries, sectors)

    # Nested composite preserves inner resolution
    inner = combine_scenarios([s1, s2], name="inner_sum", on_conflict="sum")
    outer_ret = TariffScenario("C03", {("C01", "S01"): 0.10}, name="c03_layer")
    nested = combine_scenarios([inner, outer_ret], name="nested")
    assert nested.importers == ("C01", "C03")
    ntensor = nested.to_rate_tensor(countries, sectors)
    assert np.allclose(ntensor[1, 0, :], [0.20, 0.55])
    assert np.allclose(ntensor[0, 2, :], [0.10, 0.0])

    # Validation errors
    with pytest.raises(TypeError):
        combine_scenarios("not_a_sequence")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        combine_scenarios(s1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one"):
        combine_scenarios([])
    with pytest.raises(ValueError, match="on_conflict"):
        combine_scenarios([s1], on_conflict="bad_mode")  # type: ignore[arg-type]


def test_run_scenario_keller_pac_and_erp_with_composite():
    calib = _calib_3x2()
    s_a = TariffScenario("C01", {("C02", "S01"): 0.08}, name="c01")
    s_b = TariffScenario("C02", {("C01", "S02"): 0.08}, name="c02")
    mixed = combine_scenarios([s_a, s_b], name="bilateral_mix")

    res = run_scenario(calib, mixed)
    assert res.converged
    assert res.metadata["scenario_name"] == "bilateral_mix"
    assert res.metadata["scenario_importers"] == ("C01", "C02")
    assert res.metadata["scenario_importer"] is None

    # Single-importer composite preserves scenario_importer compatibility
    single_imp = combine_scenarios([s_a], name="single_imp")
    res_single = run_scenario(calib, single_imp)
    assert res_single.converged
    assert res_single.metadata["scenario_importer"] == "C01"

    pac = solve_keller_pac(calib, mixed, max_steps=30)
    assert pac.converged

    erp = compute_effective_rate_of_protection(calib, mixed, target_country="C01")
    assert erp.nominal_tariffs.shape == (2,)


def test_summarize_scenario_batch_robust_comparison_and_exports():
    calib = _calib_3x2()
    unilateral = TariffScenario("C01", {("C02", "*"): 0.12}, name="unilateral")
    retaliation = combine_scenarios(
        [
            TariffScenario("C01", {("C02", "*"): 0.12}, name="us"),
            TariffScenario("C02", {("C01", "*"): 0.12}, name="eu"),
        ],
        name="retaliation",
    )
    batch = run_scenario_batch(calib, [unilateral, retaliation], baseline_name="base")

    comp = summarize_scenario_batch(batch, calib)
    assert isinstance(comp, ScenarioComparisonResult)
    assert comp.baseline == "base"
    assert comp.scenarios == ("base", "unilateral", "retaliation")
    assert comp.countries == ("C01", "C02", "C03")

    full_df = comp.to_frame("full")
    assert len(full_df) == 9
    base_rows = full_df[full_df["scenario"] == "base"]
    assert (base_rows["welfare_status"] == "passed").all()
    assert np.allclose(base_rows["ev"].to_numpy(), 0.0)
    assert np.allclose(base_rows["cv"].to_numpy(), 0.0)
    assert np.allclose(base_rows["ev_pct_consumption"].to_numpy(), 0.0)

    # Counterfactual welfare must match compute_hicksian_welfare directly
    w_direct = compute_hicksian_welfare(
        calib,
        batch["unilateral"],
        base_result=batch["base"],
        target_country=0,
        tol=float(batch["unilateral"].metadata.get("tol", 1e-8)),
    )
    row_c01 = full_df[(full_df["scenario"] == "unilateral") & (full_df["country"] == "C01")].iloc[0]
    assert row_c01["welfare_status"] == "passed"
    assert row_c01["ev"] == pytest.approx(w_direct.ev)
    assert row_c01["ev_pct_consumption"] == pytest.approx(w_direct.ev_pct_consumption)

    # Payoff matrix & robust comparison integration
    payoffs = comp.to_payoff_matrix("ev_pct_consumption")
    assert list(payoffs.index) == ["base", "unilateral", "retaliation"]
    assert list(payoffs.columns) == ["C01", "C02", "C03"]
    assert payoffs.index.name == "scenario"

    rob = comp.compare_robust("ev_pct_consumption")
    assert isinstance(rob, RobustPolicyComparison)
    assert list(rob.regrets.index) == ["base", "unilateral", "retaliation"]
    assert {"maximum_regret", "minimax_regret", "maximin"}.issubset(rob.summary.columns)

    rob_prob = comp.compare_robust(
        "ev_pct_consumption",
        probabilities={"C01": 0.5, "C02": 0.3, "C03": 0.2},
        include_baseline=False,
    )
    assert isinstance(rob_prob, RobustPolicyComparison)
    assert set(rob_prob.regrets.index) == {"unilateral", "retaliation"}
    assert "expected_payoff" in rob_prob.summary.columns

    # Subset countries & scenarios
    sub = summarize_scenario_batch(
        batch,
        calib,
        countries=["C01", "C02"],
        scenarios=["base", "retaliation"],
        include_baseline=False,
    )
    assert sub.countries == ("C01", "C02")
    assert sub.scenarios == ("retaliation",)

    # Result export parity & plot contract
    assert len(comp.to_dataframe("compact")) == 9
    assert len(comp.to_frame("summary")) == 3
    text = comp.summary()
    assert "Scenario Comparison" in text
    assert "|" in comp.to_markdown()
    assert "\\begin{tabular}" in comp.to_latex()
    assert "#table(" in comp.to_typst()

    fig = comp.plot()
    assert isinstance(fig, Figure)
    plt.close(fig)

    fig2, ax = plt.subplots()
    ret_ax = comp.plot(metric="real_gdp_growth_pct", ax=ax)
    assert ret_ax is ax
    plt.close(fig2)


def test_summarize_scenario_batch_failed_solve_and_no_gk():
    calib = _calib_3x2()
    s1 = TariffScenario("C01", {("C02", "*"): 0.10}, name="ok")
    s2 = TariffScenario("C01", {("C02", "*"): 0.20}, name="broken")
    batch = run_scenario_batch(calib, [s1, s2], compute_gk=False)

    # No-GK counterfactuals should have NaN real_gdp_growth_pct instead of misleading 0.0
    comp_no_gk = summarize_scenario_batch(batch, calib)
    df_ngk = comp_no_gk.to_frame()
    assert np.allclose(
        df_ngk[df_ngk["scenario"] == "baseline"]["real_gdp_growth_pct"].to_numpy(), 0.0
    )
    assert df_ngk[df_ngk["scenario"] == "ok"]["real_gdp_growth_pct"].isna().all()

    # Mark one solve as failed -> summarize warns and blocks compare_robust
    bad_scenarios = dict(batch.scenarios)
    bad_scenarios["broken"] = replace(batch.scenarios["broken"], converged=False)
    bad_batch = ScenarioBatchResult(
        scenarios=bad_scenarios,
        geary_khamis={},
        baseline_scenario=batch.baseline_scenario,
        country_codes=batch.country_codes,
        sector_codes=batch.sector_codes,
        metadata=dict(batch.metadata),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        comp_bad = summarize_scenario_batch(bad_batch, calib)
    assert any("did not converge" in str(w.message) for w in caught)
    assert any("did not converge" in msg for msg in comp_bad.warnings)
    broken_rows = comp_bad.table[comp_bad.table["scenario"] == "broken"]
    assert (broken_rows["welfare_status"] == "failed").all()
    assert broken_rows["ev"].isna().all()

    # compare_robust must block both NaN welfare and finite macro metrics via failed statuses
    with pytest.raises(ValueError, match="finite|passed|status"):
        comp_bad.compare_robust("ev_pct_consumption")
    with pytest.raises(ValueError, match="passed"):
        comp_bad.compare_robust("trade_balance")
