"""Offline tests for :mod:`puremacro.fetch.oecd_qna_tools`.

The panel is built by hand rather than downloaded, so the arithmetic can be
checked against a known answer: a chain-linked country whose components do not
add up in volume terms (USA) and a country carrying a genuine current-price
statistical discrepancy (MEX).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.fetch import (APPROACH_GDP, HOURS_PLAUSIBLE_BAND,
                             IDENTITY_TERMS, INCOME_TERMS, OUTPUT_TERMS,
                             qna_contributions, qna_hours_time_base,
                             qna_identity, qna_rebase, qna_repair_hours)
from puremacro.fetch.oecd_qna_panel import (QNA_AGGREGATES, _fetch_ref_areas,
                                            qna_countries)

_QUARTERS = pd.period_range("2014Q1", "2016Q4", freq="Q").to_timestamp(how="start")
_NAMES = ["gdp", *IDENTITY_TERMS]


def _panel(*, ref_year: dict[str, int], discrepancy: float = 0.0,
           real: bool = True) -> pd.DataFrame:
    """A minimal (code, date) panel with nominal, deflator and real columns.

    Components are built first and each gets its own deflator drifting at its
    own rate. GDP in current prices is their sum scaled by ``1 + discrepancy``,
    so the nominal identity misses by a known amount.

    GDP in volume terms is a **chained Laspeyres index** of the components —
    each quarter's growth weighted by the previous quarter's nominal shares,
    cumulated, then referenced to ``base``. That is how a statistical office
    actually builds it, and it is why the component volumes do not add to it
    except in the neighbourhood of the reference year. Summing them here
    instead would hand every non-additivity test a fixture that passes for the
    wrong reason.
    """
    drift = {"gdp": 0.020, "cons_hh": 0.018, "cons_gov": 0.025,
             "capform": 0.005, "exports": 0.030, "imports": 0.010}
    rows = {}
    for code, base in ref_year.items():
        t = np.arange(len(_QUARTERS), dtype=float)
        offset = t / 4.0 + _QUARTERS[0].year - base
        in_base = np.array([d.year == base for d in _QUARTERS])
        real_lvl, defl = {}, {}
        for i, name in enumerate(_NAMES):
            real_lvl[name] = 1000.0 * (i + 1) * (1.0 + 0.004 * t) * (1.0 + 0.02) ** offset
            defl[name] = 100.0 * (1.0 + drift[name]) ** offset
        nom = {n: real_lvl[n] * defl[n] / 100.0 for n in _NAMES}
        parts = sum(s * nom[n] for n, s in IDENTITY_TERMS.items())
        nom["gdp"] = parts * (1.0 + discrepancy)

        # Chained Laspeyres: previous-quarter nominal weights on this
        # quarter's volume relatives, cumulated from 1.
        rel = np.ones(len(_QUARTERS))
        for k in range(1, len(_QUARTERS)):
            num = sum(s * nom[n][k - 1] * (real_lvl[n][k] / real_lvl[n][k - 1])
                      for n, s in IDENTITY_TERMS.items())
            rel[k] = num / parts[k - 1]
        idx = np.cumprod(rel)
        # Reference the index so the GDP deflator averages 100 in `base`.
        scale = (nom["gdp"][in_base].mean() / idx[in_base].mean()
                 if in_base.any() else nom["gdp"][0] / idx[0])
        real_lvl["gdp"] = idx * scale
        defl["gdp"] = 100.0 * nom["gdp"] / real_lvl["gdp"]
        frame = pd.DataFrame(
            {**nom,
             **{f"{n}_defl": defl[n] for n in _NAMES},
             **{f"{n}_real": real_lvl[n] for n in _NAMES}},
            index=pd.MultiIndex.from_product([[code], _QUARTERS],
                                             names=["code", "date"]))
        rows[code] = frame
    out = pd.concat(rows.values()).sort_index()
    if not real:
        out = out.drop(columns=[c for c in out.columns if c.endswith("_real")])
    out.attrs["meta"] = tuple({"code": c, "price_ref_year": float(y)}
                              for c, y in ref_year.items())
    return out


# --------------------------------------------------------------- qna_rebase

def test_rebase_puts_every_country_on_one_reference_year():
    panel = _panel(ref_year={"USA": 2014, "MEX": 2016})
    out = qna_rebase(panel, 2015)

    in_2015 = out.index.get_level_values("date").year == 2015
    got = out.loc[in_2015].groupby(level="code")[[f"{n}_defl" for n in _NAMES]].mean()
    np.testing.assert_allclose(got.to_numpy(), 100.0, rtol=1e-12)


def test_rebase_preserves_nominal_equals_real_times_deflator():
    panel = _panel(ref_year={"USA": 2014, "MEX": 2016})
    out = qna_rebase(panel, 2015)

    for name in _NAMES:
        np.testing.assert_allclose(
            out[name], out[f"{name}_real"] * out[f"{name}_defl"] / 100.0,
            rtol=1e-10, err_msg=name)


def test_rebase_leaves_current_prices_and_every_growth_rate_alone():
    """Re-referencing is one scalar per country: only levels' units change."""
    panel = _panel(ref_year={"USA": 2014, "MEX": 2016})
    out = qna_rebase(panel, 2015)

    pd.testing.assert_frame_equal(out[_NAMES], panel[_NAMES])
    for col in ("gdp_real", "cons_hh_real", "gdp_defl", "exports_defl"):
        before = panel.groupby(level="code")[col].pct_change(fill_method=None)
        after = out.groupby(level="code")[col].pct_change(fill_method=None)
        np.testing.assert_allclose(after.dropna(), before.dropna(), rtol=1e-10)


def test_rebase_flags_a_country_with_no_data_in_the_reference_year():
    panel = _panel(ref_year={"USA": 2014, "MEX": 2016})
    out = qna_rebase(panel, 1990)

    assert out.attrs["rebase_missing"] == ("MEX", "USA")
    assert out[[f"{n}_defl" for n in _NAMES]].isna().all().all()
    # metadata must not claim a reference year that was never applied
    assert all(r["price_ref_year"] != 1990.0 for r in out.attrs["meta"])

    with pytest.raises(ValueError, match="1990"):
        qna_rebase(panel, 1990, strict=True)


def test_rebase_rejects_a_long_frame():
    panel = _panel(ref_year={"USA": 2015}).reset_index()
    with pytest.raises(ValueError, match="code, date"):
        qna_rebase(panel, 2015)


# ------------------------------------------------------------- qna_identity

def test_identity_separates_the_discrepancy_from_the_chain_linking_gap():
    """USA closes in current prices; MEX misses by a planted 2%."""
    panel = _panel(ref_year={"USA": 2015, "MEX": 2015}, discrepancy=0.0)
    clean = qna_identity(panel)
    assert clean.loc["USA", "nominal_absmax"] == pytest.approx(0.0, abs=1e-9)

    dirty = qna_identity(_panel(ref_year={"MEX": 2015}, discrepancy=0.02))
    # gap = Y - sum = 0.02 * sum, and Y = 1.02 * sum, so gap/Y = 2/102 %.
    assert dirty.loc["MEX", "nominal_mean"] == pytest.approx(100 * 0.02 / 1.02, rel=1e-6)


def test_identity_reports_a_volume_gap_that_widens_away_from_the_base_year():
    panel = _panel(ref_year={"USA": 2014})
    scored = qna_identity(panel)

    # Volumes are additive *at* the reference year and nowhere else.
    assert scored.loc["USA", "real_absmax"] > 0.05
    gap = (panel["gdp_real"]
           - sum(s * panel[f"{n}_real"] for n, s in IDENTITY_TERMS.items()))
    assert gap.abs().iloc[0] < gap.abs().iloc[-1]


def _approach_panel(*, code: str = "USA", output_gap: float = 0.0,
                    income_gap: float = 0.0, crossflow: float = 0.0,
                    chainlink: bool = True, drop_income: bool = False,
                    drop_output: bool = False) -> pd.DataFrame:
    """Panel carrying all three approaches, each with a planted residual.

    ``crossflow`` shifts the output/income flows' *own* GDP away from the
    headline expenditure GDP, which is what the real OECD data does (Japan
    0.61%, Germany 1.77%). The point of the tests below is that this must land
    in ``crossflow_*`` and must NOT be charged to the approach's components.
    """
    base = _panel(ref_year={code: 2015})
    gdp = base["gdp"]

    if not drop_output:
        base["gdp_output"] = gdp * (1.0 + crossflow)
        base["taxes_prod"] = 0.10 * gdp
        base["chainlink_disc"] = (0.01 * gdp) if chainlink else np.nan
        # va_total absorbs whatever the other output terms do not cover, so
        # the identity misses by exactly `output_gap` of that flow's own GDP.
        base["va_total"] = (base["gdp_output"] * (1.0 - output_gap)
                            - base["taxes_prod"]
                            - (base["chainlink_disc"] if chainlink else 0.0))
    if not drop_income:
        base["gdp_income"] = gdp * (1.0 + crossflow)
        base["comp_emp"] = 0.52 * base["gdp_income"]
        base["taxes_prod_imp_net"] = 0.11 * base["gdp_income"]
        base["surplus_mixed"] = (base["gdp_income"] * (1.0 - income_gap)
                                 - base["comp_emp"]
                                 - base["taxes_prod_imp_net"])
    return base


def test_identity_scores_all_three_approaches():
    scored = qna_identity(_approach_panel(output_gap=0.02, income_gap=-0.01))

    assert scored.loc["USA", "output_mean"] == pytest.approx(2.0, rel=1e-6)
    assert scored.loc["USA", "income_mean"] == pytest.approx(-1.0, rel=1e-6)
    # the expenditure identity is untouched by any of it
    assert scored.loc["USA", "nominal_absmax"] == pytest.approx(0.0, abs=1e-9)


def test_each_approach_is_scored_against_its_own_flows_gdp():
    """A flow whose GDP differs from the headline must not be charged for it.

    This is the Japan/Germany case: the OECD publishes GDP separately in each
    QNA flow and the numbers disagree. Scoring against the expenditure flow's
    GDP would report a 5% output-side discrepancy that does not exist.
    """
    scored = qna_identity(_approach_panel(crossflow=0.05))

    assert scored.loc["USA", "output_absmax"] == pytest.approx(0.0, abs=1e-6)
    assert scored.loc["USA", "income_absmax"] == pytest.approx(0.0, abs=1e-6)
    assert scored.loc["USA", "crossflow_output"] == pytest.approx(5.0, rel=1e-6)
    assert scored.loc["USA", "crossflow_income"] == pytest.approx(5.0, rel=1e-6)


def test_chainlink_discrepancy_is_optional_but_used_when_published():
    """Japan closes only with YA1; Germany does not publish it at all."""
    with_ya1 = qna_identity(_approach_panel(chainlink=True))
    assert with_ya1.loc["USA", "output_absmax"] == pytest.approx(0.0, abs=1e-6)

    # Same accounts, YA1 absent: treated as zero rather than making the whole
    # score NaN, so the country is still comparable.
    without = qna_identity(_approach_panel(chainlink=False))
    assert np.isfinite(without.loc["USA", "output_absmax"])


def test_a_country_that_does_not_publish_an_approach_comes_back_nan():
    """The real mixed case: the US is absent from the OECD by-activity flow.

    The columns exist because *other* countries populate them, and the
    non-publisher must read NaN rather than a spurious 100% gap.
    """
    usa = _approach_panel(code="USA", drop_output=True)
    esp = _approach_panel(code="ESP")
    panel = pd.concat([usa, esp]).sort_index()
    scored = qna_identity(panel)

    assert "output_mean" in scored.columns          # ESP supplies the column
    assert np.isnan(scored.loc["USA", "output_mean"])
    assert np.isnan(scored.loc["USA", "output_absmax"])
    assert scored.loc["ESP", "output_absmax"] == pytest.approx(0.0, abs=1e-6)
    # the approach it *does* publish, and the expenditure identity, still score
    assert np.isfinite(scored.loc["USA", "income_mean"])
    assert scored.loc["USA", "nominal_absmax"] == pytest.approx(0.0, abs=1e-9)


def test_approach_columns_are_absent_from_an_expenditure_only_panel():
    scored = qna_identity(_panel(ref_year={"USA": 2015}))
    assert not [c for c in scored.columns
                if c.startswith(("output_", "income_", "crossflow_"))]


def test_approach_registries_agree_with_the_gdp_map():
    assert set(APPROACH_GDP) == {"output", "income"}
    assert "va_total" in OUTPUT_TERMS and "comp_emp" in INCOME_TERMS
    # every addend is a plain column name, never a gdp column
    assert not (set(OUTPUT_TERMS) | set(INCOME_TERMS)) & set(APPROACH_GDP.values())


def test_identity_returns_nan_real_columns_without_volumes():
    scored = qna_identity(_panel(ref_year={"USA": 2015}, real=False))
    assert scored[["real_mean", "real_absmax", "real_last"]].isna().all().all()
    assert scored["n_obs"].iloc[0] == len(_QUARTERS)


# -------------------------------------------------------- qna_contributions

def test_contributions_sum_to_growth_when_the_accounts_close():
    panel = _panel(ref_year={"USA": 2015}, discrepancy=0.0)
    contrib = qna_contributions(panel)

    assert contrib["residual"].abs().max() < 0.05
    np.testing.assert_allclose(
        contrib["gdp"].dropna(),
        100 * panel.groupby(level="code")["gdp_real"].pct_change(fill_method=None).dropna(),
        rtol=1e-10)


def test_contribution_weight_is_the_previous_period_nominal_share():
    panel = _panel(ref_year={"USA": 2015})
    contrib = qna_contributions(panel)
    g = panel.loc["USA"]

    growth = g["cons_hh_real"].iloc[3] / g["cons_hh_real"].iloc[2] - 1.0
    weight = g["cons_hh"].iloc[2] / g["gdp"].iloc[2]
    assert contrib.loc[("USA", _QUARTERS[3]), "cons_hh"] == pytest.approx(
        100 * growth * weight, rel=1e-10)


def test_imports_contribute_negatively_when_they_grow():
    panel = _panel(ref_year={"USA": 2015})
    contrib = qna_contributions(panel).dropna()
    assert (contrib["imports"] < 0).all()          # import volumes are rising
    assert (contrib["exports"] > 0).all()


def test_contributions_annualise_the_aggregate_and_rescale_the_parts():
    panel = _panel(ref_year={"USA": 2015})
    plain, ann = qna_contributions(panel), qna_contributions(panel, annualise=True)

    q = plain["gdp"].dropna()
    np.testing.assert_allclose(
        ann["gdp"].dropna(), 100 * ((1 + q / 100) ** 4 - 1), rtol=1e-10)
    # the parts still add to the whole after rescaling
    assert ann["residual"].abs().max() < 0.2
    assert ann.attrs["annualised"] is True


def test_contributions_over_four_quarters():
    panel = _panel(ref_year={"USA": 2015})
    contrib = qna_contributions(panel, periods=4)
    assert contrib.attrs["periods"] == 4
    assert contrib["gdp"].isna().sum() == 4          # first year has no base
    assert contrib["residual"].abs().max() < 0.2


def test_contributions_needs_volumes():
    panel = _panel(ref_year={"USA": 2015}, real=False)
    with pytest.raises(ValueError, match="real=True"):
        qna_contributions(panel)


def test_contributions_rejects_a_zero_horizon():
    with pytest.raises(ValueError, match="positive"):
        qna_contributions(_panel(ref_year={"USA": 2015}), periods=0)


# --------------------------------------------------------- qna_countries

def test_countries_drops_aggregates_and_sorts(monkeypatch):
    monkeypatch.setattr(
        "puremacro.fetch.oecd_qna_panel._fetch_ref_areas",
        lambda flow, refresh: ["USA", "OECD", "MEX", "EA20", "ESP"])

    assert qna_countries() == ["ESP", "MEX", "USA"]
    assert qna_countries(include_aggregates=True) == [
        "EA20", "ESP", "MEX", "OECD", "USA"]


def test_countries_falls_back_when_the_endpoint_is_unreachable(monkeypatch):
    monkeypatch.setattr("puremacro.fetch.oecd_qna_panel._fetch_ref_areas",
                        lambda flow, refresh: [])

    codes = qna_countries()
    assert len(codes) >= 45 and "USA" in codes and "MEX" in codes
    assert not (set(codes) & QNA_AGGREGATES)
    assert set(qna_countries(include_aggregates=True)) >= QNA_AGGREGATES


def test_fetch_ref_areas_swallows_a_broken_response(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("rate limited")

    monkeypatch.setattr("puremacro.fetch._http.cached_get", boom)
    assert _fetch_ref_areas("OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA,", False) == []


def test_fetch_ref_areas_parses_an_availability_payload(monkeypatch):
    payload = b"""{"data": {"contentConstraints": [{"cubeRegions": [{"keyValues":
        [{"id": "FREQ", "values": ["Q"]},
         {"id": "REF_AREA", "values": ["USA", "MEX", "OECD"]}]}]}]}}"""
    monkeypatch.setattr("puremacro.fetch._http.cached_get",
                        lambda *a, **k: payload)

    assert _fetch_ref_areas("OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA,", False) == [
        "USA", "MEX", "OECD"]


# ---------------------------------------------------------------------------
# The hours time base the OECD does not declare
# ---------------------------------------------------------------------------
def _labour_panel(weeks: dict[str, float]) -> pd.DataFrame:
    """A (code, date) labour block with a chosen implied working week.

    ``weeks`` maps a country to the hours per week per employed person its
    published series *implies*. A country filing the right units implies a
    plausible week; one filing a weekly figure as a quarterly total implies
    that week divided by thirteen.
    """
    rows = []
    for code, week in weeks.items():
        heads = np.linspace(1_000.0, 1_100.0, len(_QUARTERS))   # thousands
        hours = week * 13.0 * heads * 1e3 / 1e6                 # millions
        rows.append(pd.DataFrame({
            "code": code, "date": _QUARTERS,
            "emp": heads, "hours": hours,
            "hours_employees": hours * 0.8, "hours_selfemp": hours * 0.2,
        }))
    return pd.concat(rows).set_index(["code", "date"]).sort_index()


def test_hours_time_base_leaves_plausible_countries_alone():
    panel = _labour_panel({"DEU": 27.3, "MEX": 43.0})
    diag = qna_hours_time_base(panel)
    assert diag["ok"].all()
    assert (diag["factor"] == 1.0).all()
    assert np.allclose(diag["repaired_week"], diag["implied_week"])


@pytest.mark.parametrize("filed_as, factor", [
    (1 / 13.0, 13.0),     # a weekly figure filed as a quarterly total (Chile)
    (4.0, 0.25),          # an annual figure filed as a quarterly total (Costa Rica)
])
def test_hours_time_base_identifies_the_calendar_factor(filed_as, factor):
    truth = 41.4
    panel = _labour_panel({"XXX": truth * filed_as})
    diag = qna_hours_time_base(panel)
    assert not diag.loc["XXX", "ok"]
    assert diag.loc["XXX", "factor"] == pytest.approx(factor)
    assert diag.loc["XXX", "repaired_week"] == pytest.approx(truth, rel=1e-9)
    assert diag.loc["XXX", "reason"]


def test_hours_time_base_refuses_when_the_factor_is_not_identified():
    """A week no calendar ratio can rescue is reported, not guessed at."""
    panel = _labour_panel({"XXX": 1e6})
    diag = qna_hours_time_base(panel)
    assert not diag.loc["XXX", "ok"]
    assert np.isnan(diag.loc["XXX", "factor"])
    assert "not repaired" in diag.loc["XXX", "reason"]


def test_repair_hours_rescales_every_hours_column_and_records_the_factor():
    panel = _labour_panel({"DEU": 27.3, "CHL": 41.4 / 13.0})
    fixed, diag = qna_repair_hours(panel)
    week = ((fixed["hours"] * 1e6) / (fixed["emp"] * 1e3) / 13.0)
    assert week.groupby(level="code").median().between(*HOURS_PLAUSIBLE_BAND).all()
    # the split legs move with the total, or they stop summing to it
    assert np.allclose(fixed["hours_employees"] + fixed["hours_selfemp"],
                       fixed["hours"])
    assert fixed.loc["CHL", "hours_factor"].eq(13.0).all()
    assert fixed.loc["DEU", "hours_factor"].eq(1.0).all()
    assert set(diag.index) == {"CHL", "DEU"}


def test_repair_hours_leaves_growth_rates_untouched():
    """The whole repair is a constant, so dlog h cannot move — only levels."""
    panel = _labour_panel({"CHL": 41.4 / 13.0})
    fixed, _ = qna_repair_hours(panel)
    before = np.log(panel.loc["CHL", "hours"]).diff().dropna()
    after = np.log(fixed.loc["CHL", "hours"]).diff().dropna()
    assert np.allclose(before.to_numpy(), after.to_numpy())


def test_repair_hours_is_idempotent():
    """Applying it twice must not square the factor."""
    panel = _labour_panel({"CHL": 41.4 / 13.0})
    once, _ = qna_repair_hours(panel)
    twice, _ = qna_repair_hours(once)
    assert np.allclose(once["hours"].to_numpy(), twice["hours"].to_numpy())


def test_hours_time_base_needs_the_columns_it_uses():
    panel = _labour_panel({"DEU": 27.3}).drop(columns=["hours"])
    with pytest.raises(KeyError, match="hours"):
        qna_hours_time_base(panel)
