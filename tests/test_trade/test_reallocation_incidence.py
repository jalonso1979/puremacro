"""Tests for puremacro.trade.reallocation_incidence (Track 3B)."""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from puremacro.solvers.diagnostics import SolverDiagnosticRecord
from puremacro.trade.data import ICIOData, RawIOData, load_icio_data
from puremacro.trade.reallocation import (
    ReallocationEconomy,
    ReallocationTransitionResult,
    audit_reallocation,
    solve_reallocation_transition,
)
from puremacro.trade.reallocation_incidence import (
    HouseholdGroupSpec,
    ICIOCalibrationAudit,
    ReallocationCalibrationResult,
    TransitionError,
    TransitionHouseholdIncidenceResult,
    calibrate_reallocation_from_icio,
    evaluate_transition_household_incidence,
    make_sector_household_groups,
    reallocation_tariffs_from_scenarios,
)
from puremacro.trade.scenarios import CompositeTariffScenario, TariffScenario


def _make_standard_groups(economy: ReallocationEconomy) -> tuple[HouseholdGroupSpec, ...]:
    c0, c1 = economy.countries
    s0, s1 = economy.sectors
    return (
        HouseholdGroupSpec(
            name=f"{c0}_{s0}_Workers",
            country=c0,
            labor_weights=(1.0, 0.0),
            capital_weight=0.0,
            rebate_weight=0.0,
            investment_weight=0.0,
            expenditure_shares=economy.sector_weights[0],
        ),
        HouseholdGroupSpec(
            name=f"{c0}_{s1}_Workers",
            country=c0,
            labor_weights=(0.0, 1.0),
            capital_weight=0.0,
            rebate_weight=0.0,
            investment_weight=0.0,
            expenditure_shares=economy.sector_weights[0],
        ),
        HouseholdGroupSpec(
            name=f"{c0}_Capital_Rebate_Owners",
            country=c0,
            labor_weights=(0.0, 0.0),
            capital_weight=1.0,
            rebate_weight=1.0,
            investment_weight=1.0,
            expenditure_shares=economy.sector_weights[0],
        ),
        HouseholdGroupSpec(
            name=f"{c1}_Representative",
            country=c1,
            labor_weights=(1.0, 1.0),
            non_labor_weight=1.0,
            expenditure_shares=economy.sector_weights[1],
        ),
    )


@pytest.mark.parametrize("rule", ["fixed_baskets", "cobb_douglas", "stone_geary", "ces"])
def test_zero_tariff_shock_exact_identity(rule: str) -> None:
    """Zero tariff shock must recover exact EV=0, CV=0, cost_index=1.0 (< 1e-12)."""
    econ = ReallocationEconomy()
    zero_tariffs = np.zeros((20, 2, 2), dtype=float)
    tr = solve_reallocation_transition(econ, zero_tariffs)
    groups = make_sector_household_groups(econ, include_capital_owners=True, basket_tilt=0.15)

    res = evaluate_transition_household_incidence(
        tr,
        groups,
        rule=rule,  # type: ignore[arg-type]
        ces_elasticity=2.0,
        supernumerary_share=0.55,
    )
    assert isinstance(res, TransitionHouseholdIncidenceResult)
    assert res.audit_metrics["zero_shock_detected"] is True
    assert float(np.max(np.abs(res.ev))) < 1e-12
    assert float(np.max(np.abs(res.cv))) < 1e-12
    assert float(np.max(np.abs(res.ev_pct))) < 1e-12
    assert float(np.max(np.abs(res.cv_pct))) < 1e-12
    assert float(np.max(np.abs(res.cost_index - 1.0))) < 1e-12
    assert float(np.max(np.abs(res.price_index - 1.0))) < 1e-12
    assert float(np.max(np.abs(res.real_income - res.baseline_budget[None, :]))) < 1e-12
    assert float(np.max(np.abs(res.pv_ev))) < 1e-12
    assert float(np.max(np.abs(res.pv_ev_finite))) < 1e-12
    assert float(np.max(np.abs(res.path_cev_pct))) < 1e-12
    assert float(np.max(np.abs(res.ev_pct_spread))) < 1e-12
    assert res.diagnostic.convergence.converged is True
    assert res.diagnostic.accuracy.is_accurate is True
    assert res.diagnostic.overall_status == "PASS"


def test_oecd_icio_calibration_and_transition_reconciliation() -> None:
    """Offline OECD ICIO calibration passes audit, solves transition, and conserves budgets."""
    raw_icio = load_icio_data(return_structured=True)
    assert isinstance(raw_icio, ICIOData)
    raw_va_total = float((raw_icio.labor_va + raw_icio.capital_va).sum())

    cal = calibrate_reallocation_from_icio(raw_icio)
    assert isinstance(cal, ReallocationCalibrationResult)
    assert isinstance(cal.audit, ICIOCalibrationAudit)
    assert isinstance(cal.diagnostic, SolverDiagnosticRecord)
    assert cal.economy.countries == ("USA", "ROW")
    assert cal.economy.sectors == ("Tradable", "Services")
    assert cal.steady_state.converged is True
    assert audit_reallocation(cal.steady_state)["accepted"] is True
    assert cal.stability["stable_graph"] is True
    assert cal.stability["stable_roots"] == 16
    assert cal.diagnostic.convergence.converged is True
    assert cal.diagnostic.accuracy.is_accurate is True
    assert cal.diagnostic.overall_status == "PASS"

    # World VA conserved identically; non-identified parameters transparently disclosed.
    assert cal.audit.world_va_total == pytest.approx(raw_va_total, rel=1e-12)
    for key in (
        "beta",
        "delta",
        "risk_aversion",
        "trade_elasticity",
        "adjustment_costs",
        "productivity_normalization",
        "omitted_intermediates",
        "omitted_bilateral_tariffs",
        "national_financial_autarky",
    ):
        assert key in cal.audit.stylized_assumptions

    # Asymmetric US Tradable import tariff via TariffScenario helper + transition incidence.
    scen = TariffScenario(
        importer="USA",
        tariffs={("ROW", "Tradable"): 0.20},
        name="us_tradable_20pct",
    )
    tariffs = reallocation_tariffs_from_scenarios(scen, cal.economy, horizon=25)
    assert tariffs.shape == (25, 2, 2)
    assert tariffs[0, 0, 0] == pytest.approx(0.20)
    assert float(np.max(np.abs(tariffs[:, 0, 1]))) == 0.0
    assert float(np.max(np.abs(tariffs[:, 1, :]))) == 0.0

    tr = solve_reallocation_transition(cal.economy, tariffs)
    groups = make_sector_household_groups(cal.economy, include_capital_owners=True, basket_tilt=0.05)
    inc = evaluate_transition_household_incidence(tr, groups, rule="cobb_douglas")

    # Exact group spending reconciliation vs national Pc * C and gross income vs Pc * A at every date.
    nat_cons_nom = tr.paths["price_index"] * tr.paths["consumption"]
    nat_abs_nom = tr.paths["price_index"] * tr.paths["gross_absorption"]
    for c_idx, cname in enumerate(cal.economy.countries):
        g_idxs = [i for i, gc in enumerate(inc.group_countries) if gc == cname]
        grp_spending = inc.nominal_income[:, g_idxs].sum(axis=1)
        grp_gross_inc = (
            inc.labor_income[:, g_idxs] + inc.capital_income[:, g_idxs] + inc.rebate_income[:, g_idxs]
        ).sum(axis=1)
        assert np.allclose(grp_spending, nat_cons_nom[:, c_idx], atol=1e-9, rtol=1e-9)
        assert np.allclose(grp_gross_inc, nat_abs_nom[:, c_idx], atol=1e-9, rtol=1e-9)

    assert inc.audit_metrics["max_country_consumption_gap"] < 1e-10
    assert inc.audit_metrics["max_sector_labor_gap"] < 1e-12
    assert inc.audit_metrics["max_national_income_gap"] < 1e-10


def test_custom_icio_partition_raw_io_and_fine_scenario_bridge() -> None:
    """Custom MEX+USA vs ROW aggregation, RawIOData adapter, and fine scenario bridging work."""
    raw_icio = load_icio_data(return_structured=True)
    assert isinstance(raw_icio, ICIOData)
    mex_usa = ("MEX", "USA")
    row = tuple(c for c in raw_icio.country_codes if c not in mex_usa)
    goods = ("A", "B", "C")
    services = tuple(s for s in raw_icio.sector_codes if s not in goods)

    cal = calibrate_reallocation_from_icio(
        raw_icio,
        country_groups={"MEX+USA": mex_usa, "ROW": row},
        sector_groups={"Goods": goods, "Services": services},
        expenditure_basis="consumption",
    )
    assert cal.economy.countries == ("MEX+USA", "ROW")
    assert cal.economy.sectors == ("Goods", "Services")
    assert cal.audit.world_va_total == pytest.approx(
        float((raw_icio.labor_va + raw_icio.capital_va).sum()), rel=1e-12
    )
    assert cal.stability["stable_graph"] is True

    # Fine-country/sector TariffScenario maps to coarse 2x2 using exact expenditure weights.
    fine_scen = CompositeTariffScenario((
        TariffScenario(importer="USA", tariffs={("CHN", "C"): 0.30, ("DEU", "C"): 0.30}, name="us_mfg"),
    ))
    tar_path = reallocation_tariffs_from_scenarios(fine_scen, cal.economy, horizon=20, calibration=cal)
    assert tar_path.shape == (20, 2, 2)
    assert tar_path[0, 1, 0] == 0.0

    # Quantitative expenditure-weighting check vs exact ICIO baseline flows (not unweighted 1/57).
    cal_usa = calibrate_reallocation_from_icio(raw_icio)
    usa_idx = raw_icio.country_codes.index("USA")
    chn_idx = raw_icio.country_codes.index("CHN")
    c_idx = raw_icio.sector_codes.index("C")
    row_idxs = [i for i, code in enumerate(raw_icio.country_codes) if code != "USA"]
    goods_idxs = [raw_icio.sector_codes.index(s) for s in ("A", "B", "C")]
    flows = cal_usa.audit.fine_expenditure_flows
    chn_c_to_usa = float(flows[chn_idx, c_idx, usa_idx])
    total_row_goods_to_usa = float(flows[np.ix_(row_idxs, goods_idxs, [usa_idx])].sum())
    expected_rate = 0.30 * chn_c_to_usa / total_row_goods_to_usa
    chn_scen = TariffScenario(importer="USA", tariffs={("CHN", "C"): 0.30}, name="chn_c_30")
    tar_chn = reallocation_tariffs_from_scenarios(chn_scen, cal_usa.economy, horizon=2, calibration=cal_usa)
    assert tar_chn[0, 0, 0] == pytest.approx(expected_rate, rel=1e-12)
    assert abs(tar_chn[0, 0, 0] - 0.30 / (len(row_idxs) * len(goods_idxs))) > 1e-2

    # Refuse non-zero intra-bloc tariffs inside a coarse region.
    intra_scen = TariffScenario(importer="USA", tariffs={("MEX", "C"): 0.15}, name="intra_na")
    with pytest.raises(ValueError, match="intra-bloc"):
        reallocation_tariffs_from_scenarios(intra_scen, cal.economy, horizon=2, calibration=cal)

    # RawIOData input path.
    raw_obj = RawIOData(
        source="toy_test",
        year=2023,
        countries=("A", "B"),
        sectors=("G", "S"),
        fd_categories=("C", "I"),
        intermediate_matrix=np.array([
            [10.0, 2.0, 3.0, 1.0],
            [2.0, 12.0, 1.0, 2.0],
            [4.0, 1.0, 14.0, 2.0],
            [1.0, 2.0, 2.0, 16.0],
        ]),
        final_demand_matrix=np.array([
            [25.0, 8.0, 5.0, 2.0],
            [18.0, 6.0, 2.0, 1.0],
            [6.0, 2.0, 30.0, 9.0],
            [2.0, 1.0, 22.0, 7.0],
        ]),
        value_added=np.array([30.0, 20.0, 35.0, 25.0]),
        taxes_less_subsidies=np.zeros(4),
        gross_output=np.array([46.0, 39.0, 56.0, 46.0]),
    )
    cal_raw = calibrate_reallocation_from_icio(raw_obj)
    assert cal_raw.economy.countries == ("A", "B")
    assert cal_raw.economy.sectors == ("G", "S")
    assert cal_raw.economy.capital_share == pytest.approx(1.0 / 3.0)


def test_sector_wage_and_spending_share_incidence_directions() -> None:
    """Protected vs hurt sector workers diverge; import-basket group faces higher COLI."""
    econ = ReallocationEconomy()
    tariffs = np.zeros((25, 2, 2), dtype=float)
    tariffs[:, 0, 0] = 0.25  # Home protects sector S01
    tr = solve_reallocation_transition(econ, tariffs)

    # 1. Sector-wage incidence holding spending basket identical (national weights).
    wage_groups = _make_standard_groups(econ)
    inc_w = evaluate_transition_household_incidence(tr, wage_groups, rule="cobb_douglas")
    idx_s01 = inc_w.group_names.index("AAA_S01_Workers")
    idx_s02 = inc_w.group_names.index("AAA_S02_Workers")

    # Protected S01 worker gains in real income & EV throughout transition; S02 worker loses.
    assert np.all(inc_w.ev[:, idx_s01] > 0.0)
    assert np.all(inc_w.ev[:, idx_s02] < 0.0)
    assert inc_w.pv_ev_pct[idx_s01] > 0.0
    assert inc_w.pv_ev_pct[idx_s02] < 0.0
    assert inc_w.path_cev_pct[idx_s01] > 0.0
    assert inc_w.path_cev_pct[idx_s02] < 0.0
    assert np.all(inc_w.ev_pct_spread[:, 0] > 0.0)
    assert inc_w.country_pv_ev_pct_spread[0] > 0.0

    # 2. Foreign retaliatory tariff targeting AAA's S01 reverses S01 workers' real welfare sign.
    retaliate = np.zeros((25, 2, 2), dtype=float)
    retaliate[:, 1, 0] = 0.25
    tr_ret = solve_reallocation_transition(econ, retaliate)
    inc_ret = evaluate_transition_household_incidence(tr_ret, wage_groups, rule="cobb_douglas")
    assert inc_ret.pv_ev_pct[idx_s01] < 0.0
    assert inc_ret.pv_ev_pct[idx_s01] < inc_ret.pv_ev_pct[idx_s02]

    # 3. Spending-share price exposure: identical incomes, different baskets.
    basket_groups = (
        HouseholdGroupSpec(
            name="AAA_High_S01_Spend",
            country="AAA",
            labor_weights=(0.5, 0.5),
            non_labor_weight=0.5,
            expenditure_shares=(0.85, 0.15),
        ),
        HouseholdGroupSpec(
            name="AAA_High_S02_Spend",
            country="AAA",
            labor_weights=(0.5, 0.5),
            non_labor_weight=0.5,
            expenditure_shares=(0.15, 0.85),
        ),
        HouseholdGroupSpec(
            name="BBB_Rep",
            country="BBB",
            labor_weights=(1.0, 1.0),
            non_labor_weight=1.0,
        ),
    )
    inc_b = evaluate_transition_household_incidence(tr, basket_groups, rule="ces", ces_elasticity=2.0)
    # Ps[S01] rises more than Ps[S02] under S01 tariff -> S01-heavy spender faces higher COLI and lower EV%.
    assert np.all(inc_b.cost_index[:, 0] > inc_b.cost_index[:, 1])
    assert np.all(inc_b.ev_pct[:, 0] < inc_b.ev_pct[:, 1])
    assert inc_b.pv_ev_pct[0] < inc_b.pv_ev_pct[1]

    # 4. Mixed heterogeneous per-group rules (fixed_baskets + stone_geary + ces) via dicts.
    mixed = (
        {
            "name": "AAA_Fixed",
            "country": "AAA",
            "labor_weights": (0.6, 0.4),
            "non_labor_weight": 0.5,
            "expenditure_shares": (0.7, 0.3),
            "rule": "fixed_baskets",
        },
        {
            "name": "AAA_SG",
            "country": "AAA",
            "labor_weights": (0.4, 0.6),
            "non_labor_weight": 0.5,
            "expenditure_shares": (0.3, 0.7),
            "rule": "stone_geary",
            "supernumerary_share": 0.45,
        },
        {
            "name": "BBB_CES",
            "country": "BBB",
            "labor_weights": (1.0, 1.0),
            "non_labor_weight": 1.0,
            "rule": "ces",
            "ces_elasticity": 1.8,
        },
    )
    inc_mix = evaluate_transition_household_incidence(tr, mixed)
    assert inc_mix.group_rules == ("fixed_baskets", "stone_geary", "ces")
    assert inc_mix.audit_metrics["max_country_consumption_gap"] < 1e-10


def test_refusals_bilingual_reports_and_serialization() -> None:
    """Invalid weights, negative shares, unconverged/unaudited transitions are refused cleanly."""
    econ = ReallocationEconomy()
    tariffs = np.zeros((20, 2, 2), dtype=float)
    tariffs[:, 0, 0] = 0.15
    tr = solve_reallocation_transition(econ, tariffs)
    valid_groups = make_sector_household_groups(econ)

    # Refuse unconverged transition flag.
    bad_diag = dict(tr.diagnostics)
    bad_diag["accepted"] = False
    unconv = dataclasses.replace(tr, diagnostics=bad_diag)
    with pytest.raises(TransitionError, match="unconverged"):
        evaluate_transition_household_incidence(unconv, valid_groups)

    # Also catchable as ValueError; test corrupted equilibrium values rejected by audit_reallocation.
    corrupt_paths = dict(tr.paths)
    corrupt_paths["consumption"] = tr.paths["consumption"] * 1.25
    corrupt = dataclasses.replace(tr, paths=corrupt_paths)
    with pytest.raises(ValueError, match="audit"):
        evaluate_transition_household_incidence(corrupt, valid_groups)

    # Refuse invalid weight sums (sector labor weights != 1.0, capital/rebate != 1.0).
    bad_labor = (
        HouseholdGroupSpec(name="A1", country="AAA", labor_weights=(0.4, 0.0), non_labor_weight=0.5),
        HouseholdGroupSpec(name="A2", country="AAA", labor_weights=(0.4, 1.0), non_labor_weight=0.5),
        HouseholdGroupSpec(name="B", country="BBB", labor_weights=(1.0, 1.0), non_labor_weight=1.0),
    )
    with pytest.raises(ValueError, match="labor_weights"):
        evaluate_transition_household_incidence(tr, bad_labor)

    bad_rebate = (
        HouseholdGroupSpec(name="A1", country="AAA", labor_weights=(1.0, 0.0), capital_weight=0.5, rebate_weight=0.2),
        HouseholdGroupSpec(name="A2", country="AAA", labor_weights=(0.0, 1.0), capital_weight=0.5, rebate_weight=0.2),
        HouseholdGroupSpec(name="B", country="BBB", labor_weights=(1.0, 1.0), non_labor_weight=1.0),
    )
    with pytest.raises(ValueError, match="rebate_weight"):
        evaluate_transition_household_incidence(tr, bad_rebate)

    # Refuse negative expenditure shares and non-summing shares.
    with pytest.raises(ValueError, match="nonnegative"):
        HouseholdGroupSpec(
            name="Neg", country="AAA", labor_weights=(1.0, 1.0), non_labor_weight=1.0,
            expenditure_shares=(-0.1, 1.1),
        )
    with pytest.raises(ValueError, match="sum to 1.0"):
        HouseholdGroupSpec(
            name="BadSum", country="AAA", labor_weights=(1.0, 1.0), non_labor_weight=1.0,
            expenditure_shares=(0.3, 0.4),
        )

    # Refuse expenditure_basis='consumption' without 'C' column; return None aggregated_consumption otherwise.
    raw_no_c = RawIOData(
        source="no_c_test",
        year=2023,
        countries=("A", "B"),
        sectors=("G", "S"),
        fd_categories=("I", "Cx"),
        intermediate_matrix=np.array([
            [10.0, 2.0, 3.0, 1.0],
            [2.0, 12.0, 1.0, 2.0],
            [4.0, 1.0, 14.0, 2.0],
            [1.0, 2.0, 2.0, 16.0],
        ]),
        final_demand_matrix=np.array([
            [25.0, 8.0, 5.0, 2.0],
            [18.0, 6.0, 2.0, 1.0],
            [6.0, 2.0, 30.0, 9.0],
            [2.0, 1.0, 22.0, 7.0],
        ]),
        value_added=np.array([30.0, 20.0, 35.0, 25.0]),
        taxes_less_subsidies=np.zeros(4),
        gross_output=np.array([46.0, 39.0, 56.0, 46.0]),
    )
    with pytest.raises(ValueError, match="requires final-demand code 'C'"):
        calibrate_reallocation_from_icio(raw_no_c, expenditure_basis="consumption")
    cal_no_c = calibrate_reallocation_from_icio(raw_no_c, expenditure_basis="final_demand")
    assert cal_no_c.audit.aggregated_consumption is None

    # Refuse non-partitioning ICIO group mapping.
    raw_icio = load_icio_data(return_structured=True)
    with pytest.raises(ValueError, match="partition"):
        calibrate_reallocation_from_icio(
            raw_icio,
            country_groups={"USA": ("USA",), "CHN": ("CHN",)},
        )

    # Verify bilingual outputs, tabular exports, and JSON serialization.
    cal = calibrate_reallocation_from_icio(raw_icio)
    assert "Región" in cal.summary(lang="es").columns
    assert "Region" in cal.summary(lang="en").columns
    assert "| País" in evaluate_transition_household_incidence(tr, valid_groups).to_markdown(lang="es")

    inc = evaluate_transition_household_incidence(tr, valid_groups)
    assert "Country" in inc.to_dataframe(lang="en").columns
    assert len(inc.path_dataframe()) == len(tr.tariffs) * len(valid_groups)
    assert len(inc.inequality_dataframe()) == len(tr.tariffs) * 2
    assert isinstance(inc.to_latex(lang="en"), str)
    assert isinstance(inc.to_typst(lang="es"), str)
    json.dumps(cal.to_dict())
    json.dumps(inc.to_dict())

    with pytest.raises(ValueError, match="lang"):
        inc.summary(lang="fr")
