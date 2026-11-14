"""Adversarial regression tests for trade scenario batch convergence guards.

Guards against two silent propagation defects in ``puremacro.trade.scenarios``:
1. Warm-start contamination: an unconverged scenario in ``run_scenario_batch``
   must not overwrite ``prev_x`` and poison subsequent scenarios.
2. Unconverged-solve propagation: unconverged equilibria (or unconverged
   Geary-Khamis PPP results / partial welfare failures) must emit explicit
   ``RuntimeWarning`` notices and produce ``NaN``/``failed`` indicators rather
   than silently entering Geary-Khamis tables or summary aggregations.
"""
from __future__ import annotations

from dataclasses import replace
import warnings

import numpy as np
import pandas as pd
import pytest

from puremacro.trade import (
    GearyKhamisResult,
    ScenarioBatchResult,
    TariffScenario,
    calibrate_trade_model,
    run_scenario_batch,
    summarize_scenario_batch,
)
from puremacro.trade.data import synthetic_codes, synthetic_icio_table
import puremacro.trade.scenarios as scenarios_mod
from puremacro.trade.scenarios import _build_scenario_summary_df


def _calib_3x2():
    table = synthetic_icio_table(3, 2, nfd=3, seed=42)
    countries, sectors = synthetic_codes(3, 2)
    return calibrate_trade_model(
        table, ns=2, nc=3, nfd=3, country_codes=countries, sector_codes=sectors
    )


def test_unconverged_scenario_does_not_poison_subsequent_warm_starts_end_to_end():
    """Real solver end-to-end: a mid-batch diverged solve must not break later scenarios."""
    calib = _calib_3x2()
    s_ok = TariffScenario("C01", {("C02", "*"): 0.10}, name="ok")
    s_fail = TariffScenario("C01", {("C02", "*"): 50.0, ("C03", "*"): 50.0}, name="extreme_fail")
    s_after = TariffScenario("C01", {("C02", "*"): 0.12}, name="after")

    clean_batch = run_scenario_batch(
        calib, [s_ok, s_after], warm_start=True, compute_gk=True, max_iter=6, tol=1e-6
    )
    assert all(res.converged for res in clean_batch.scenarios.values())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        batch = run_scenario_batch(
            calib,
            [s_ok, s_fail, s_after],
            warm_start=True,
            compute_gk=True,
            max_iter=6,
            tol=1e-6,
        )

    assert any(
        issubclass(w.category, RuntimeWarning)
        and "extreme_fail" in str(w.message)
        and "did not converge" in str(w.message)
        for w in caught
    )
    assert batch.scenarios["baseline"].converged is True
    assert batch.scenarios["ok"].converged is True
    assert batch.scenarios["extreme_fail"].converged is False
    # Pre-fix, 'after' inherited extreme_fail.x_sol and also failed to converge.
    assert batch.scenarios["after"].converged is True
    assert batch.metadata["converged"] == {
        "baseline": True,
        "ok": True,
        "extreme_fail": False,
        "after": True,
    }

    # 'after' must receive the exact same warm start from 'ok' and reach the identical equilibrium.
    np.testing.assert_allclose(
        batch.scenarios["after"].x_sol,
        clean_batch.scenarios["after"].x_sol,
        rtol=0.0,
        atol=1e-13,
    )
    assert batch.scenarios["after"].iterations == clean_batch.scenarios["after"].iterations


def test_warm_start_x0_skips_first_and_consecutive_failed_solves(monkeypatch):
    """Spy on real run_scenario calls to prove x0 never comes from unconverged solves."""
    calib = _calib_3x2()
    recorded_x0: list[tuple[str, np.ndarray, bool]] = []
    orig_run_scenario = scenarios_mod.run_scenario

    def spy_run_scenario(c, scen, **opts):
        x0 = opts["x0"].copy()
        res = orig_run_scenario(c, scen, **opts)
        recorded_x0.append((scen.name, x0, bool(res.converged)))
        return res

    monkeypatch.setattr(scenarios_mod, "run_scenario", spy_run_scenario)

    scens = [
        TariffScenario("C01", {("C02", "*"): 50.0, ("C03", "*"): 50.0}, name="fail_first"),
        TariffScenario("C01", {("C02", "*"): 0.10}, name="ok_mid"),
        TariffScenario("C01", {("C02", "*"): 50.0, ("C03", "*"): 50.0}, name="fail_2"),
        TariffScenario("C01", {("C02", "*"): 60.0, ("C03", "*"): 60.0}, name="fail_3"),
        TariffScenario("C01", {("C02", "*"): 0.12}, name="ok_last"),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        batch = run_scenario_batch(
            calib, scens, warm_start=True, compute_gk=False, max_iter=5, tol=1e-6
        )

    assert [conv for _, _, conv in recorded_x0] == [False, True, False, False, True]
    # fail_first failed -> ok_mid must still start from baseline's x_sol.
    np.testing.assert_array_equal(recorded_x0[1][1], batch.scenarios["baseline"].x_sol)
    # fail_2 and fail_3 both failed -> ok_last must start from ok_mid's x_sol.
    np.testing.assert_array_equal(recorded_x0[4][1], batch.scenarios["ok_mid"].x_sol)
    assert not np.allclose(recorded_x0[4][1], batch.scenarios["fail_3"].x_sol)


def test_unconverged_baseline_falls_back_from_poisoned_x_and_skips_gk():
    """An unconverged baseline must neither poison counterfactual x0 nor compute GK."""
    calib = _calib_3x2()
    good_base = scenarios_mod.run_scenario(
        calib, TariffScenario("C01", {}, name="zero"), tol=1e-6
    )
    poisoned_base = replace(
        good_base,
        converged=False,
        max_residual=12.5,
        x_sol=np.full_like(good_base.x_sol, 1e4),
    )
    s_ok = TariffScenario("C01", {("C02", "*"): 0.10}, name="ok")

    with warnings.catch_warnings(record=True) as caught_batch:
        warnings.simplefilter("always")
        batch = run_scenario_batch(
            calib, [s_ok], base_result=poisoned_base, warm_start=True, compute_gk=True, tol=1e-6
        )

    assert any(
        issubclass(w.category, RuntimeWarning) and "Baseline" in str(w.message)
        for w in caught_batch
    )
    assert batch.scenarios["ok"].converged is True
    assert batch.geary_khamis == {}
    assert batch.real_gdp_table.isna().all().all()

    with warnings.catch_warnings(record=True) as caught_sum:
        warnings.simplefilter("always")
        comp = summarize_scenario_batch(batch, calib)

    assert any(
        issubclass(w.category, RuntimeWarning) and "Baseline" in str(w.message)
        for w in caught_sum
    )
    assert (comp.table["welfare_status"] == "failed").all()
    assert comp.table["ev"].isna().all()
    assert comp.table["real_gdp_growth_pct"].isna().all()
    for col in (
        "mean_ev_pct_consumption",
        "min_ev_pct_consumption",
        "total_ev",
        "mean_real_gdp_growth_pct",
    ):
        assert comp.scenario_summary[col].isna().all()
    base_summary = comp.scenario_summary[comp.scenario_summary["scenario"] == "baseline"].iloc[0]
    assert bool(base_summary["converged"]) is False
    assert np.isnan(base_summary["mean_cpi_inflation_pct"])


def test_unconverged_scenario_skipped_from_gk_and_summary_aggregates():
    """Unconverged scenarios must have NaN GK, NaN welfare, and NaN summary means."""
    calib = _calib_3x2()
    s_ok = TariffScenario("C01", {("C02", "*"): 0.10}, name="ok")
    s_fail = TariffScenario("C01", {("C02", "*"): 50.0, ("C03", "*"): 50.0}, name="extreme_fail")
    s_after = TariffScenario("C01", {("C02", "*"): 0.12}, name="after")

    clean_batch = run_scenario_batch(
        calib, [s_ok, s_after], warm_start=True, compute_gk=True, max_iter=6, tol=1e-6
    )
    clean_comp = summarize_scenario_batch(clean_batch, calib)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        batch = run_scenario_batch(
            calib,
            [s_ok, s_fail, s_after],
            warm_start=True,
            compute_gk=True,
            max_iter=6,
            tol=1e-6,
        )

    assert set(batch.geary_khamis) == {"baseline", "ok", "after"}
    assert set(batch.ppp_results) == {"baseline", "ok", "after"}
    assert batch.real_gdp_table["extreme_fail"].isna().all()
    assert batch.real_gdp_table["ok"].notna().all()
    assert batch.real_gdp_table["after"].notna().all()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        comp = summarize_scenario_batch(batch, calib)

    assert any(
        issubclass(w.category, RuntimeWarning)
        and "extreme_fail" in str(w.message)
        and "did not converge" in str(w.message)
        for w in caught
    )
    assert any("extreme_fail" in msg and "did not converge" in msg for msg in comp.warnings)

    fail_rows = comp.table[comp.table["scenario"] == "extreme_fail"]
    assert (fail_rows["converged"] == False).all()  # noqa: E712
    assert (fail_rows["welfare_status"] == "failed").all()
    for col in (
        "ev",
        "cv",
        "ev_pct_consumption",
        "ev_pct_gdp",
        "price_effect",
        "factor_income_effect",
        "real_gdp_growth_pct",
    ):
        assert fail_rows[col].isna().all()

    # Pre-fix, summary silently reported finite mean_real_gdp_growth_pct & mean_cpi_inflation_pct.
    sum_df = comp.scenario_summary
    fail_sum = sum_df[sum_df["scenario"] == "extreme_fail"].iloc[0]
    assert bool(fail_sum["converged"]) is False
    assert int(fail_sum["n_welfare_passed"]) == 0
    for col in (
        "mean_ev_pct_consumption",
        "min_ev_pct_consumption",
        "total_ev",
        "mean_real_gdp_growth_pct",
        "mean_cpi_inflation_pct",
    ):
        assert np.isnan(fail_sum[col]), f"Expected NaN for unconverged {col}, got {fail_sum[col]}"

    # Converged scenarios in the same batch must preserve exact values from the clean run.
    for scen in ("baseline", "ok", "after"):
        got_row = sum_df[sum_df["scenario"] == scen].iloc[0]
        exp_row = clean_comp.scenario_summary[clean_comp.scenario_summary["scenario"] == scen].iloc[0]
        assert bool(got_row["converged"]) is True
        assert int(got_row["n_welfare_passed"]) == 3
        for col in (
            "mean_ev_pct_consumption",
            "min_ev_pct_consumption",
            "total_ev",
            "mean_real_gdp_growth_pct",
            "mean_cpi_inflation_pct",
        ):
            assert got_row[col] == pytest.approx(exp_row[col], rel=1e-12, abs=1e-12)

    for metric in ("ev_pct_consumption", "real_gdp_growth_pct", "trade_balance"):
        with pytest.raises(ValueError, match="finite|passed|status"):
            comp.compare_robust(metric)


def test_summarize_guards_stale_unconverged_gk_and_unconverged_gk_result():
    """Even a pre-populated GK entry must be ignored if equilibrium or GK failed to converge."""
    calib = _calib_3x2()
    s_ok = TariffScenario("C01", {("C02", "*"): 0.10}, name="ok")
    s_other = TariffScenario("C01", {("C02", "*"): 0.20}, name="other")
    clean = run_scenario_batch(calib, [s_ok, s_other], compute_gk=True)

    # Case A: scenario has converged=False, yet batch carries a stale GK entry & nonzero table.
    bad_scenarios = dict(clean.scenarios)
    bad_scenarios["other"] = replace(clean.scenarios["other"], converged=False)
    stale_gk = dict(clean.geary_khamis)
    stale_gk["other"] = replace(
        clean.geary_khamis["other"],
        converged=True,
        gdp_growth=np.array([99.0, -50.0, 12.0]),
    )
    stale_batch = ScenarioBatchResult(
        scenarios=bad_scenarios,
        geary_khamis=stale_gk,
        baseline_scenario=clean.baseline_scenario,
        country_codes=clean.country_codes,
        sector_codes=clean.sector_codes,
        metadata=dict(clean.metadata),
    )

    with warnings.catch_warnings(record=True) as caught_a:
        warnings.simplefilter("always")
        comp_a = summarize_scenario_batch(stale_batch, calib)

    assert any(issubclass(w.category, RuntimeWarning) and "other" in str(w.message) for w in caught_a)
    other_rows = comp_a.table[comp_a.table["scenario"] == "other"]
    assert other_rows["real_gdp_growth_pct"].isna().all()
    assert other_rows["ev"].isna().all()
    other_sum = comp_a.scenario_summary[comp_a.scenario_summary["scenario"] == "other"].iloc[0]
    assert np.isnan(other_sum["mean_real_gdp_growth_pct"])
    assert np.isnan(other_sum["mean_cpi_inflation_pct"])

    # Case B: equilibrium converged, but Geary-Khamis fixed point itself did not converge.
    unconv_gk = dict(clean.geary_khamis)
    unconv_gk["other"] = replace(
        clean.geary_khamis["other"],
        converged=False,
        gdp_growth=np.array([42.0, -42.0, 7.0]),
    )
    bad_gk_batch = clean.with_geary_khamis(unconv_gk)

    with warnings.catch_warnings(record=True) as caught_b:
        warnings.simplefilter("always")
        comp_b = summarize_scenario_batch(bad_gk_batch, calib)

    assert any(
        issubclass(w.category, RuntimeWarning)
        and "Geary-Khamis" in str(w.message)
        and "other" in str(w.message)
        for w in caught_b
    )
    assert any("Geary-Khamis" in msg and "other" in msg for msg in comp_b.warnings)
    other_rows_b = comp_b.table[comp_b.table["scenario"] == "other"]
    assert other_rows_b["real_gdp_growth_pct"].isna().all()
    # Hicksian welfare for converged equilibrium remains valid.
    assert (other_rows_b["welfare_status"] == "passed").all()
    other_sum_b = comp_b.scenario_summary[comp_b.scenario_summary["scenario"] == "other"].iloc[0]
    assert np.isnan(other_sum_b["mean_real_gdp_growth_pct"])
    assert np.isfinite(other_sum_b["mean_ev_pct_consumption"])


def test_build_scenario_summary_df_guards_unconverged_and_partial_welfare_failures():
    """_build_scenario_summary_df must never average over unconverged or failed rows."""
    df = pd.DataFrame([
        # Scenario 1: unconverged rows that still contain finite bogus numbers
        {
            "scenario": "unconv",
            "country": "C01",
            "converged": False,
            "welfare_status": "failed",
            "ev": 100.0,
            "ev_pct_consumption": 5.0,
            "real_gdp_growth_pct": 12.0,
            "cpi_inflation_pct": 8.0,
        },
        {
            "scenario": "unconv",
            "country": "C02",
            "converged": False,
            "welfare_status": "failed",
            "ev": -50.0,
            "ev_pct_consumption": -3.0,
            "real_gdp_growth_pct": -6.0,
            "cpi_inflation_pct": 4.0,
        },
        # Scenario 2: converged solve, but partial welfare failure (1 of 3 countries failed)
        {
            "scenario": "partial",
            "country": "C01",
            "converged": True,
            "welfare_status": "passed",
            "ev": 10.0,
            "ev_pct_consumption": 1.0,
            "real_gdp_growth_pct": 0.5,
            "cpi_inflation_pct": 0.2,
        },
        {
            "scenario": "partial",
            "country": "C02",
            "converged": True,
            "welfare_status": "passed",
            "ev": 30.0,
            "ev_pct_consumption": 3.0,
            "real_gdp_growth_pct": 1.5,
            "cpi_inflation_pct": 0.4,
        },
        {
            "scenario": "partial",
            "country": "C03",
            "converged": True,
            "welfare_status": "failed",
            "ev": np.nan,
            "ev_pct_consumption": np.nan,
            "real_gdp_growth_pct": 2.5,
            "cpi_inflation_pct": 0.6,
        },
        # Scenario 3: completely valid baseline
        {
            "scenario": "base",
            "country": "C01",
            "converged": True,
            "welfare_status": "passed",
            "ev": 0.0,
            "ev_pct_consumption": 0.0,
            "real_gdp_growth_pct": 0.0,
            "cpi_inflation_pct": 0.0,
        },
    ])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        summary = _build_scenario_summary_df(df, ["base", "unconv", "partial"], "base")

    assert any(
        issubclass(w.category, RuntimeWarning) and "unconv" in str(w.message)
        for w in caught
    )
    assert any(
        issubclass(w.category, RuntimeWarning) and "partial" in str(w.message)
        for w in caught
    )

    row_unconv = summary[summary["scenario"] == "unconv"].iloc[0]
    assert bool(row_unconv["converged"]) is False
    assert int(row_unconv["n_welfare_passed"]) == 0
    for col in (
        "mean_ev_pct_consumption",
        "min_ev_pct_consumption",
        "total_ev",
        "mean_real_gdp_growth_pct",
        "mean_cpi_inflation_pct",
    ):
        assert np.isnan(row_unconv[col])

    row_partial = summary[summary["scenario"] == "partial"].iloc[0]
    assert bool(row_partial["converged"]) is True
    assert int(row_partial["n_welfare_passed"]) == 2
    assert int(row_partial["n_countries"]) == 3
    # Welfare aggregates must NOT silently average the 2 passing countries.
    assert np.isnan(row_partial["mean_ev_pct_consumption"])
    assert np.isnan(row_partial["min_ev_pct_consumption"])
    assert np.isnan(row_partial["total_ev"])
    # Macro means remain valid because all 3 countries converged and have finite macro metrics.
    assert row_partial["mean_real_gdp_growth_pct"] == pytest.approx(1.5)
    assert row_partial["mean_cpi_inflation_pct"] == pytest.approx(0.4)

    row_base = summary[summary["scenario"] == "base"].iloc[0]
    assert bool(row_base["converged"]) is True
    assert row_base["mean_ev_pct_consumption"] == pytest.approx(0.0)


def test_no_gk_batch_with_unconverged_scenario_keeps_nan_real_gdp_for_all_counterfactuals():
    """With compute_gk=False and one failed scenario (NaN in real_gdp_table),
    converged counterfactuals must still report NaN real_gdp_growth_pct (not 0.0).
    """
    calib = _calib_3x2()
    s_ok = TariffScenario("C01", {("C02", "*"): 0.10}, name="ok")
    s_fail = TariffScenario("C01", {("C02", "*"): 50.0, ("C03", "*"): 50.0}, name="extreme_fail")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        batch = run_scenario_batch(
            calib, [s_ok, s_fail], compute_gk=False, max_iter=6, tol=1e-6
        )
        comp = summarize_scenario_batch(batch, calib)

    df = comp.to_frame("full")
    assert np.allclose(df[df["scenario"] == "baseline"]["real_gdp_growth_pct"].to_numpy(), 0.0)
    assert df[df["scenario"] == "ok"]["real_gdp_growth_pct"].isna().all()
    assert df[df["scenario"] == "extreme_fail"]["real_gdp_growth_pct"].isna().all()
