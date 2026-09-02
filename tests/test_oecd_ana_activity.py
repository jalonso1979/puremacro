"""Offline tests for :mod:`puremacro.fetch.oecd_ana_activity`.

The SDMX round-trip is mocked with synthetic ``csvfilewithlabels`` frames that
reproduce the two shapes this module exists for: a country that publishes every
ISIC section and total hours by activity (DEU), and one that publishes neither —
sections E/N/S/T/U missing and hours only for employees, which is Japan and the
United States. If the second one survives the aggregation, so do they.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from puremacro.fetch import oecd_ana_activity as mod
from puremacro.fetch.oecd_ana_activity import (ANA_LABOR, ana_by_activity,
                                               ana_hours_wedge, ana_meta,
                                               chain_volume)

_YEARS = list(range(2010, 2021))
_SECTIONS = list("ABCDEFGHIJKLMNOPQRSTU")
#: what a country like Japan does not publish at all
_MISSING = {"E", "N", "S", "T", "U"}
_GROUPINGS = ["_T", "A", "OTQ"]

# The synthetic economy: value added grows 3% a year everywhere and prices 2%,
# except agriculture, which is flat, and O-Q, which grows 1%. Hours grow 1%.
_VA_GROWTH = {"A": 0.00, "O": 0.01, "P": 0.01, "Q": 0.01}
_HOURS_GROWTH = {"A": -0.01, "OTQ": 0.005}


def _va_rows(code: str) -> list[dict]:
    sections = [s for s in _SECTIONS
                if not (code == "JPN" and s in _MISSING)]
    out = []
    for s in sections + ["_T"]:
        base = 100.0 + 10.0 * (_SECTIONS.index(s) if s in _SECTIONS else 30)
        g = _VA_GROWTH.get(s, 0.03)
        for t, year in enumerate(_YEARS):
            vol = base * (1.0 + g) ** t
            price = 1.02 ** t
            for price_base, value in (("V", vol * price),
                                      ("Y", vol * 1.02 ** max(t - 1, 0))):
                out.append({
                    "FREQ": "A", "REF_AREA": code, "SECTOR": "S1",
                    "COUNTERPART_SECTOR": "_Z", "TRANSACTION": "B1G",
                    "INSTR_ASSET": "_Z", "ACTIVITY": s, "EXPENDITURE": "_Z",
                    "UNIT_MEASURE": "XDC", "PRICE_BASE": price_base,
                    "TRANSFORMATION": "N", "TABLE_IDENTIFIER": "T0600",
                    "TIME_PERIOD": str(year), "OBS_VALUE": value,
                    "UNIT_MULT": 6, "CURRENCY": "EUR",
                })
    # _T is the sum of the sections, so subtraction has something to be right
    # about. Overwrite the placeholder rows built above.
    out = [r for r in out if r["ACTIVITY"] != "_T"]
    for price_base in ("V", "Y"):
        for year in _YEARS:
            tot = sum(r["OBS_VALUE"] for r in out
                      if r["TIME_PERIOD"] == str(year)
                      and r["PRICE_BASE"] == price_base)
            out.append({**out[0], "ACTIVITY": "_T", "PRICE_BASE": price_base,
                        "TIME_PERIOD": str(year), "OBS_VALUE": tot})
    return out


def _labor_rows(code: str) -> list[dict]:
    """DEU publishes every stem; JPN/USA publish employee hours only."""
    stems = ANA_LABOR if code == "DEU" else {
        k: v for k, v in ANA_LABOR.items()
        if v[1] == "PS" or k == "hours_employees"}
    out = []
    for stem, (txn, unit) in stems.items():
        for act in _GROUPINGS:
            base = 1000.0 if act == "_T" else 100.0
            g = _HOURS_GROWTH.get(act, 0.01)
            for t, year in enumerate(_YEARS):
                out.append({
                    "FREQ": "A", "REF_AREA": code, "SECTOR": "S1",
                    "COUNTERPART_SECTOR": "_Z", "TRANSACTION": txn,
                    "INSTR_ASSET": "_Z", "ACTIVITY": act, "EXPENDITURE": "_Z",
                    "UNIT_MEASURE": unit, "PRICE_BASE": "_Z",
                    "TRANSFORMATION": "N", "TABLE_IDENTIFIER": "T0300",
                    "TIME_PERIOD": str(year),
                    "OBS_VALUE": base * (1.0 + g) ** t,
                    "UNIT_MULT": 3 if unit == "PS" else 6, "CURRENCY": "",
                })
    return out


@pytest.fixture
def fake_sdmx(monkeypatch):
    def _get(agency_flow, key, start_period, *, refresh=False):
        codes = key.split(".")[1]
        wanted = [c for c in codes.split("+") if c] or ["DEU", "JPN"]
        rows: list[dict] = []
        for code in wanted:
            rows += (_va_rows(code) if "TABLE6" in agency_flow
                     else _labor_rows(code))
        df = pd.DataFrame(rows)
        if "TABLE6" in agency_flow:
            # the caller pins PRICE_BASE in the key; honour it
            pb = key.split(".")[-3]
            if pb:
                df = df[df["PRICE_BASE"] == pb]
            acts = key.split(".")[6].split("+")
            df = df[df["ACTIVITY"].isin(acts)]
        else:
            acts = key.split(".")[6].split("+")
            df = df[df["ACTIVITY"].isin(acts)]
        return df.reset_index(drop=True)

    monkeypatch.setattr(mod, "get_sdmx_csv", _get)
    return _get


def test_columns_and_countries(fake_sdmx):
    p = ana_by_activity(["DEU", "JPN"])
    assert set(p.index.get_level_values("code")) == {"DEU", "JPN"}
    for c in ("va_total", "va_total_py", "va_agri", "va_public",
              "hours_employees", "hours_employees_agri", "hours_employees_public"):
        assert c in p.columns, c
    # no deflators here: this module returns two price bases, not a ratio
    assert not any(c.endswith("_defl") for c in p.columns)


def test_japan_survives_missing_sections(fake_sdmx):
    """The whole point: aggregate by subtraction, so E/N/S/T/U can be absent."""
    p = ana_by_activity(["DEU", "JPN"])
    jpn = p.loc["JPN"]
    assert jpn["va_total"].notna().all()
    assert jpn["va_agri"].notna().all()
    assert jpn["va_public"].notna().all()
    market = jpn["va_total"] - jpn["va_agri"] - jpn["va_public"]
    assert (market > 0).all()


def test_public_is_o_plus_p_plus_q(fake_sdmx):
    p = ana_by_activity(["DEU"])
    va = _va_rows("DEU")
    year = str(_YEARS[3])
    expect = sum(r["OBS_VALUE"] for r in va
                 if r["TIME_PERIOD"] == year and r["PRICE_BASE"] == "V"
                 and r["ACTIVITY"] in ("O", "P", "Q"))
    got = p.loc[("DEU", int(year)), "va_public"]
    assert got == pytest.approx(expect, rel=1e-12)


def test_chain_volume_recovers_known_growth(fake_sdmx):
    """Agriculture is flat in volume by construction, so its chain must be flat."""
    p = ana_by_activity(["DEU"]).loc["DEU"]
    q = chain_volume(p["va_agri"], p["va_agri_py"])
    g = np.log(q).diff().dropna()
    assert g.abs().max() < 1e-12
    # and the total grows at the volume-weighted rate, not the nominal one
    qt = chain_volume(p["va_total"], p["va_total_py"])
    assert 0.0 < float(np.log(qt).diff().mean()) < 0.03


def test_chain_volume_breaks_on_a_gap():
    """A break costs its own link and nothing else — and never bridges."""
    yrs = [2010, 2011, 2015, 2016, 2017]
    v = pd.Series([100.0, 103.0, 106.0, 109.0, 112.0], index=yrs)
    y = pd.Series([100.0, 103.0, 106.0, 109.0, 112.0], index=yrs)
    q = chain_volume(v, y)
    assert q.loc[2010] == pytest.approx(1.0)          # base of the first run
    assert q.loc[2011] == pytest.approx(1.03)
    assert np.isnan(q.loc[2015])                      # the gap: no link
    assert q.notna().loc[[2016, 2017]].all()          # the run after it lives
    # the bug this guards: growth must never bridge the hole
    g = np.log(q).diff()
    assert np.isnan(g.loc[2015]) and np.isnan(g.loc[2016])
    assert g.loc[2017] == pytest.approx(np.log(112.0 / 109.0))


def test_meta_reports_the_market_window_not_the_longest_column(fake_sdmx):
    p = ana_by_activity(["DEU", "JPN"])
    m = ana_meta(p).set_index("code")
    assert set(m.columns) >= {"hours_by_activity", "market_first", "market_n"}
    assert bool(m.loc["DEU", "hours_by_activity"]) is True
    assert bool(m.loc["JPN", "hours_by_activity"]) is False
    assert int(m.loc["JPN", "market_n"]) == len(_YEARS)


def test_wedge_only_where_both_denominators_exist(fake_sdmx):
    p = ana_by_activity(["DEU", "JPN"])
    w = ana_hours_wedge(p)
    assert np.isfinite(w.loc["DEU", "g_all"])
    assert "JPN" in w.index and np.isnan(w.loc["JPN", "g_all"])
    assert np.isfinite(w.loc["JPN", "g_employees"])


def test_total_is_always_requested(fake_sdmx):
    p = ana_by_activity(["DEU"], activities=["agri"])
    assert "va_total" in p.columns and "va_agri" in p.columns
    assert "va_public" not in p.columns


def test_unknown_activity_is_an_error(fake_sdmx):
    with pytest.raises(ValueError, match="unknown activity"):
        ana_by_activity(["DEU"], activities=["mining"])


def test_empty_response_is_an_empty_frame(monkeypatch):
    monkeypatch.setattr(mod, "get_sdmx_csv",
                        lambda *a, **k: pd.DataFrame())
    p = ana_by_activity(["XXX"])
    assert p.empty
    assert list(p.index.names) == ["code", "year"]
    assert ana_meta(p).empty


# ---------------------------------------------------------------------------
# Regressions. Each of these was green against the pre-fix tree for the wrong
# reason, so each is written to fail there: see CONTRIBUTING.md, "Making sure a
# test can fail".
# ---------------------------------------------------------------------------

def _rows(code, *, va_sections=None, hours_acts=("_T", "A", "OTQ"),
          hours_mult=6, years=None, hours_total=1000.0, hours_part=100.0,
          hpw=1000.0, hours_factor=1.0):
    """One country's raw rows, with every knob a regression here needs."""
    years = list(years or _YEARS)
    sections = list(va_sections if va_sections is not None else _SECTIONS)
    out = []
    for s in sections:
        for y in years:
            for pb in ("V", "Y"):
                out.append({"FREQ": "A", "REF_AREA": code, "TRANSACTION": "B1G",
                            "ACTIVITY": s, "UNIT_MEASURE": "XDC",
                            "PRICE_BASE": pb, "TIME_PERIOD": str(y),
                            "OBS_VALUE": 100.0 + 10.0 * _SECTIONS.index(s),
                            "UNIT_MULT": 6, "CURRENCY": "EUR"})
    for pb in ("V", "Y"):
        for y in years:
            out.append({"FREQ": "A", "REF_AREA": code, "TRANSACTION": "B1G",
                        "ACTIVITY": "_T", "UNIT_MEASURE": "XDC",
                        "PRICE_BASE": pb, "TIME_PERIOD": str(y),
                        "OBS_VALUE": sum(100.0 + 10.0 * _SECTIONS.index(s)
                                         for s in _SECTIONS),
                        "UNIT_MULT": 6, "CURRENCY": "EUR"})
    for txn, unit in ANA_LABOR.values():
        acts = _GROUPINGS if unit == "PS" else list(hours_acts)
        for a in acts:
            # Heads are the base; hours are the base times the target hours per
            # worker per year, so `hpw` sets the ratio the plausibility guard
            # reads and `hours_factor` is the mislabelled time base to detect.
            base = hours_total if a == "_T" else hours_part
            value = base if unit == "PS" else base * hpw / 1e3 * hours_factor
            for y in years:
                out.append({"FREQ": "A", "REF_AREA": code, "TRANSACTION": txn,
                            "ACTIVITY": a, "UNIT_MEASURE": unit,
                            "PRICE_BASE": "_Z", "TIME_PERIOD": str(y),
                            "OBS_VALUE": value,
                            "UNIT_MULT": 3 if unit == "PS" else hours_mult,
                            "CURRENCY": ""})
    return out


def _serve(per_code):
    """A `get_sdmx_csv` stub over ``{code: rows}``, honouring the key's filters."""
    def _get(agency_flow, key, start_period, *, refresh=False):
        parts = key.split(".")
        codes = [c for c in parts[1].split("+") if c] or list(per_code)
        rows = []
        for c in codes:
            rows += per_code.get(c, [])
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df = df[df["ACTIVITY"].isin(parts[6].split("+"))]
        if "TABLE6" in agency_flow:
            df = df[df["PRICE_BASE"] == parts[-3]]
            df = df[df["TRANSACTION"] == "B1G"]
        else:
            df = df[df["TRANSACTION"] != "B1G"]
        return df.reset_index(drop=True)
    return _get


def test_market_window_does_not_depend_on_the_other_countries_in_the_call(monkeypatch):
    """The honest sample is a property of the country, never of its company.

    Korea publishes hours for the whole economy only. Asked for alone it used
    to report a full market window, because the branch columns nothing in the
    request published were dropped from the requirement instead of failing it;
    adding Germany to the same call made the same country report zero.
    """
    served = _serve({"KOR": _rows("KOR", hours_acts=("_T",)),
                     "DEU": _rows("DEU")})
    monkeypatch.setattr(mod, "get_sdmx_csv", served)

    alone = ana_meta(ana_by_activity(["KOR"])).set_index("code")
    joint = ana_meta(ana_by_activity(["KOR", "DEU"])).set_index("code")
    assert int(alone.loc["KOR", "market_n"]) == 0
    assert int(joint.loc["KOR", "market_n"]) == 0
    assert alone.loc["KOR", "market_first"] is None or pd.isna(alone.loc["KOR", "market_first"])
    # and Germany, which does publish the branches, keeps its window
    assert int(joint.loc["DEU", "market_n"]) == len(_YEARS)


def test_missing_columns_says_why_the_window_is_zero(monkeypatch):
    monkeypatch.setattr(mod, "get_sdmx_csv",
                        _serve({"KOR": _rows("KOR", hours_acts=("_T",))}))
    p = ana_by_activity(["KOR"])
    assert set(p.attrs["missing_columns"]) == {"hours_employees_agri",
                                               "hours_employees_public"}


def test_negative_retained_sector_is_excluded_and_named(monkeypatch):
    """Australia publishes O-Q hours identical to its own total.

    Presence is not consistency: every column exists, so the window used to
    certify all of them while `hours - hours_public` was negative throughout.
    """
    monkeypatch.setattr(mod, "get_sdmx_csv",
                        _serve({"AUS": _rows("AUS", hours_total=1000.0,
                                             hours_part=1000.0)}))
    with pytest.warns(UserWarning, match="retained sector is negative"):
        p = ana_by_activity(["AUS"])
    m = ana_meta(p).set_index("code")
    assert int(m.loc["AUS", "market_n"]) == 0
    assert bool(m.loc["AUS", "hours_by_activity"]) is False


def test_hours_published_weekly_are_put_back_on_an_annual_basis(monkeypatch):
    """New Zealand publishes these hours per week, labelled like everyone else.

    The quarterly sibling has had this guard since Chile and Costa Rica; the
    annual flow shipped without it and NZ came back 52x too small.
    """
    # hours per week, heads unchanged: the ratio is 52x too small and nothing
    # in the message says so. The baseline is 1600 h/worker/yr so that the
    # corrected figure lands inside the acceptance band.
    weekly = _rows("NZL", hpw=1600.0, hours_factor=1.0 / 52.0)
    monkeypatch.setattr(mod, "get_sdmx_csv",
                        _serve({"NZL": weekly, "DEU": _rows("DEU", hpw=1600.0)}))
    p = ana_by_activity(["NZL", "DEU"])
    hpw = (p["hours"] * 1e3 / p["emp"]).groupby(level="code").median()
    assert hpw.loc["NZL"] == pytest.approx(hpw.loc["DEU"], rel=1e-9)
    assert ana_meta(p).set_index("code").loc["NZL", "hours_scale"] == 52.0
    assert p.attrs["hours_scale"] == {"NZL": 52.0}


def test_a_partial_public_sector_is_refused_not_summed(monkeypatch):
    """O+P with no Q is a smaller government, not a missing one."""
    no_q = [s for s in _SECTIONS if s != "Q"]
    monkeypatch.setattr(mod, "get_sdmx_csv",
                        _serve({"XXX": _rows("XXX", va_sections=no_q)}))
    p = ana_by_activity(["XXX"])
    assert "va_public" not in p.columns
    assert "va_total" in p.columns


def test_duplicate_labour_rows_do_not_duplicate_the_index(monkeypatch):
    """The key leaves dimensions open, so one series can arrive twice."""
    doubled = _rows("DEU") + _rows("DEU")
    monkeypatch.setattr(mod, "get_sdmx_csv", _serve({"DEU": doubled}))
    p = ana_by_activity(["DEU"])
    assert p.index.is_unique
    assert len(p) == len(_YEARS)


def test_chunks_empty_survives_a_total_outage(monkeypatch):
    """The one case the attribute exists for is the one it used to skip."""
    monkeypatch.setattr(mod, "get_sdmx_csv", lambda *a, **k: pd.DataFrame())
    p = ana_by_activity(["DEU", "JPN"])
    assert p.empty
    assert p.attrs["chunks_empty"], "an empty frame must say what came back empty"
    assert any(c.startswith("labor:") for c in p.attrs["chunks_empty"])


def test_a_lost_price_base_is_recorded_not_hidden(monkeypatch):
    """Two price bases are two requests; losing the second loses all volumes."""
    served = _serve({"DEU": _rows("DEU")})

    def _v_only(agency_flow, key, start_period, *, refresh=False):
        if "TABLE6" in agency_flow and key.split(".")[-3] == "Y":
            return pd.DataFrame()
        return served(agency_flow, key, start_period, refresh=refresh)

    monkeypatch.setattr(mod, "get_sdmx_csv", _v_only)
    p = ana_by_activity(["DEU"])
    assert not [c for c in p.columns if c.endswith("_py")]
    assert any(c.endswith(":Y") for c in p.attrs["chunks_empty"])
    # and with no volume to chain, the market window is empty rather than full
    assert int(ana_meta(p).set_index("code").loc["DEU", "market_n"]) == 0


def test_chain_volume_refuses_a_non_positive_link():
    """A sign change is finite, so it used to pass the mask and be skipped.

    `Series.cumsum` skips NaN, so `np.log` of a negative link contributed
    nothing and the level continued on the far side as if it had grown at
    zero — verbatim what the docstring says cannot happen.
    """
    yrs = list(range(2000, 2008))
    v = pd.Series([100.0, 102.0, 104.0, -3.0, 105.0, 107.0, 109.0, 111.0],
                  index=yrs)
    q = chain_volume(v, v)
    assert q.loc[2002] == pytest.approx(1.04)
    assert np.isnan(q.loc[2003]) and np.isnan(q.loc[2004])
    # the run after the break starts its own base, and never continues 1.04
    assert q.loc[2005] == pytest.approx(107.0 / 105.0)
    assert np.isnan(np.log(q).diff().loc[2005])


def test_chain_volume_zero_link_is_a_break_not_a_zero_level():
    v = pd.Series([100.0, 0.0, 104.0, 106.0], index=range(2000, 2004))
    with warnings.catch_warnings():
        warnings.simplefilter("error")     # no bare divide-by-zero RuntimeWarning
        q = chain_volume(v, v)
    assert np.isnan(q.loc[2000]) and np.isnan(q.loc[2001])
    assert q.loc[2002] == pytest.approx(1.0)


def test_chain_volume_multiindex_tolerates_a_country_with_no_volumes():
    idx = pd.MultiIndex.from_product([["AAA", "BBB"], [2010, 2011, 2012]],
                                     names=["code", "year"])
    cur = pd.Series(100.0, index=idx)
    prev = cur.loc[["AAA"]]                # BBB publishes no previous-year prices
    q = chain_volume(cur, prev)
    assert q.loc["AAA"].notna().any()
    assert q.loc["BBB"].isna().all()


def test_wedge_never_differences_across_a_chain_volume_break():
    """`.dropna()` before `.diff()` closed the hole and bridged two runs.

    Each run of `chain_volume` carries its own arbitrary base, so a bridged
    link is not a two-year growth rate — it is a meaningless one. Here the
    truth is exactly 100*log(1.03) per year on both denominators.
    """
    yrs = [2010, 2011, 2012, 2015, 2016, 2017, 2018, 2019]
    va = pd.Series([100.0 * 1.03 ** i for i in range(len(yrs))], index=yrs)
    idx = pd.MultiIndex.from_product([["AAA"], yrs], names=["code", "year"])
    g = pd.DataFrame(index=idx)
    g["va_total"] = va.to_numpy()
    g["va_total_py"] = va.to_numpy()
    for c in ("va_agri", "va_agri_py", "va_public", "va_public_py"):
        g[c] = 0.0
    for c in ("hours", "hours_agri", "hours_public"):
        g[c] = 1000.0 if c == "hours" else 0.0
    for c in ("hours_employees", "hours_employees_agri", "hours_employees_public"):
        g[c] = 900.0 if c == "hours_employees" else 0.0

    w = ana_hours_wedge(g)
    assert w.loc["AAA", "g_all"] == pytest.approx(100 * np.log(1.03), rel=1e-9)
    assert w.loc["AAA", "wedge"] == pytest.approx(0.0, abs=1e-12)
    # the two links the break costs, and only those two
    assert int(w.loc["AAA", "n_all"]) == 5


def test_wedge_averages_both_denominators_over_the_same_years():
    """Employee hours run decades longer; two windows put a regime change in.

    Volume grows 4%/yr to 1999 and 1%/yr after, both denominators grow at
    1%/yr, and `hours` starts only in 2000 — the real OECD shape. The true
    substitution cost is exactly zero.
    """
    yrs = list(range(1980, 2020))
    vol, lvl = [], 100.0
    for y in yrs:
        vol.append(lvl)
        lvl *= 1.04 if y < 1999 else 1.01
    idx = pd.MultiIndex.from_product([["AAA"], yrs], names=["code", "year"])
    g = pd.DataFrame(index=idx)
    g["va_total"] = vol
    g["va_total_py"] = [v * (1.04 if y < 1999 else 1.01)
                        for v, y in zip(vol, yrs)]
    for c in ("va_agri", "va_agri_py", "va_public", "va_public_py"):
        g[c] = 0.0
    h = [1000.0 * 1.01 ** i for i in range(len(yrs))]
    g["hours_employees"] = h
    g["hours"] = [np.nan if y < 2000 else v for v, y in zip(h, yrs)]
    for c in ("hours_agri", "hours_public",
              "hours_employees_agri", "hours_employees_public"):
        g[c] = 0.0

    w = ana_hours_wedge(g)
    assert int(w.loc["AAA", "n_all"]) == int(w.loc["AAA", "n_employees"])
    assert w.loc["AAA", "wedge"] == pytest.approx(0.0, abs=1e-9)


def test_wedge_window_and_minimum_sample(monkeypatch):
    monkeypatch.setattr(mod, "get_sdmx_csv", _serve({"DEU": _rows("DEU")}))
    p = ana_by_activity(["DEU"])
    assert ana_hours_wedge(p, start=2010, end=2014).empty   # 4 links < 5
    assert not ana_hours_wedge(p, start=2010, end=2016).empty


def test_labour_units_are_rescaled_from_whatever_the_source_published(monkeypatch):
    """UNIT_MULT equalled the target in the original fixture, so nothing ran."""
    raw = _rows("DEU", hours_mult=3)        # hours published in thousands
    monkeypatch.setattr(mod, "get_sdmx_csv", _serve({"DEU": raw}))
    p = ana_by_activity(["DEU"])
    # 10**(3-6) of the published figure, i.e. normalised to millions
    assert p["hours"].iloc[0] == pytest.approx(1000.0 * 10 ** -3)


def test_chunks_batches_wide_country_lists():
    assert mod._chunks(None) == [""]
    assert mod._chunks([f"C{i:02d}" for i in range(20)]) == [
        "+".join(f"C{i:02d}" for i in range(0, 8)),
        "+".join(f"C{i:02d}" for i in range(8, 16)),
        "+".join(f"C{i:02d}" for i in range(16, 20)),
    ]
