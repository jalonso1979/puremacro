"""Offline tests for ``puremacro.fetch.oecd_ana_panel`` (annual OECD national accounts).

Every test patches the module's HTTP seam ``_csv`` (and ``_availability``) with
tiny synthetic SDMX ``csvfile`` frames, built in the column layout the OECD
actually returns for ``DSD_NAMAIN10`` (12 dimensions) and ``DSD_NASEC10`` (13).
"""
from __future__ import annotations

import http.client
import warnings

import numpy as np
import pandas as pd
import pytest

import puremacro.fetch.oecd_ana_panel as ap

NAMAIN = list(ap.NAMAIN10_DIMS)
NASEC = list(ap.NASEC10_DIMS)


def _row(dims, *, year, value, mult=6, currency="XXX", **pins):
    base = {d: "_Z" for d in dims}
    base.update(FREQ="A", COUNTERPART_SECTOR="S1", UNIT_MEASURE="XDC",
                PRICE_BASE="V", TRANSFORMATION="N", SECTOR="S1")
    if "ACCOUNTING_ENTRY" in dims:
        base.update(ACCOUNTING_ENTRY="D", VALUATION="S", TABLE_IDENTIFIER="T0800")
    else:
        base.update(TABLE_IDENTIFIER="T0102")
    base.update(pins)
    base.update(TIME_PERIOD=year, OBS_VALUE=value, UNIT_MULT=mult, CURRENCY=currency,
                REF_YEAR_PRICE=np.nan, OBS_STATUS="A")
    return base


def _frame(rows):
    d = pd.DataFrame(rows)
    d.insert(0, "DATAFLOW", "OECD.SDD.NAD:SYNTHETIC")
    d.attrs["status"] = "ok"
    return d


def _empty(status="HTTP 404"):
    d = pd.DataFrame()
    d.attrs["status"] = status
    return d


def _main_rows(code, years=(2019, 2020, 2021), scale=1.0, currency="XXX",
               lr=True, l_only=False):
    """Expenditure side: V for 7 aggregates, LR (or L) volumes, YA0."""
    rows = []
    comps = {"gdp": ("B1GQ", "S1", "_Z", "_Z", 1000.0),
             "cons_hh": ("P3", "S1M", "_Z", "_Z", 600.0),
             "cons_gov": ("P3", "S13", "_Z", "_Z", 200.0),
             "inv": ("P51G", "S1", "N11G", "_T", 200.0),
             "capform": ("P5", "S1", "N1G", "_T", 210.0),
             "exports": ("P6", "S1", "_Z", "_Z", 300.0),
             "imports": ("P7", "S1", "_Z", "_Z", 290.0)}
    for i, y in enumerate(years):
        price = 1.0 + 0.1 * (y - 2020)            # deflator 2020 = 1
        for trans, sec, asset, act, lvl in comps.values():
            v = lvl * scale * (1 + 0.05 * i)
            exp = "_T" if trans == "P3" else "_Z"
            rows.append(_row(NAMAIN, year=y, value=v, currency=currency, TRANSACTION=trans,
                             SECTOR=sec, INSTR_ASSET=asset, ACTIVITY=act, EXPENDITURE=exp))
            if lr and not l_only:
                rows.append(_row(NAMAIN, year=y, value=v / price, currency=currency,
                                 TRANSACTION=trans, SECTOR=sec, INSTR_ASSET=asset,
                                 ACTIVITY=act, EXPENDITURE=exp, PRICE_BASE="LR"))
            if l_only:
                # own reference year: volume = 2 * (V / price)
                rows.append(_row(NAMAIN, year=y, value=2 * v / price, currency=currency,
                                 TRANSACTION=trans, SECTOR=sec, INSTR_ASSET=asset,
                                 ACTIVITY=act, EXPENDITURE=exp, PRICE_BASE="L"))
        rows.append(_row(NAMAIN, year=y, value=-20.0 * scale * (1 + 0.05 * i),
                         currency=currency, TRANSACTION="YA0", SECTOR="_Z"))
    for r in rows:
        r["REF_AREA"] = code
    return rows


class FakeOECD:
    """Dispatch on the flow name; record every (flow, key) asked for."""

    def __init__(self, by_flow):
        self.by_flow = by_flow
        self.calls: list[tuple[str, str]] = []

    def __call__(self, agency_flow, key, *, start, refresh):
        self.calls.append((agency_flow, key))
        short = agency_flow.split("@")[1].split(",")[0]
        hit = self.by_flow.get(short)
        if hit is None:
            return _empty()
        if callable(hit):
            return hit(key)
        return hit.copy() if hasattr(hit, "copy") else hit


@pytest.fixture
def patch(monkeypatch):
    def _install(by_flow, countries=None):
        fake = FakeOECD(by_flow)
        monkeypatch.setattr(ap, "_csv", fake)
        if countries is not None:
            payload = {"data": {"contentConstraints": [{"cubeRegions": [{"keyValues": [
                {"id": "REF_AREA", "values": countries}]}]}]}}
            monkeypatch.setattr(ap, "_availability", lambda *a, **k: payload)
        return fake
    return _install


def test_main_block_has_period_start_dates_millions_and_2020_based_deflators(patch):
    bbb = _main_rows("BBB")
    # BBB publishes in thousands: UNIT_MULT 3 must come back divided by 1000.
    for r in bbb:
        r["UNIT_MULT"] = 3
    rows = _main_rows("AAA") + bbb
    fake = patch({"DF_TABLE1": _frame(rows)})
    p = ap.ana_panel(["AAA", "BBB"], start="2019")
    assert list(p.index.names) == ["code", "date"]
    assert pd.api.types.is_datetime64_any_dtype(p.index.get_level_values("date"))
    dates = p.index.get_level_values("date")
    assert ((dates.month == 1) & (dates.day == 1)).all()
    assert sorted(set(dates.year)) == [2019, 2020, 2021]
    assert p.loc[("AAA", pd.Timestamp("2020-01-01")), "gdp"] == pytest.approx(1050.0)
    assert p.loc[("BBB", pd.Timestamp("2020-01-01")), "gdp"] == pytest.approx(1.05)
    assert (p.xs(pd.Timestamp("2020-01-01"), level="date")["gdp_defl"] == 100.0).all()
    assert p.loc[("AAA", pd.Timestamp("2021-01-01")), "gdp_defl"] == pytest.approx(110.0)
    assert p.loc[("AAA", pd.Timestamp("2021-01-01")), "gdp_real"] == pytest.approx(1100.0 / 1.1)
    assert "discrepancy_exp" in p and "discrepancy_exp_defl" not in p
    x = p.loc["AAA"]
    gap = x.cons_hh + x.cons_gov + x.capform + x.exports - x.imports + x.discrepancy_exp - x.gdp
    assert gap.abs().max() == pytest.approx(0.0)
    assert all(p[c].dtype == np.float64 for c in p.columns)
    meta = ap.ana_panel_meta(p).set_index("code")
    assert meta.loc["AAA", "volume_base"] == "LR"
    assert meta.loc["AAA", "currency"] == "XXX"
    assert (meta["first"] == 2019).all() and (meta["last"] == 2021).all()
    assert p.attrs["missing"] == ()
    assert p.attrs["fetched_at"].endswith("+00:00")
    assert "DF_TABLE1,2.0" in p.attrs["source"]
    # one request, a 12-dimension key with both countries in it
    assert len(fake.calls) == 1
    flow, key = fake.calls[0]
    assert key.count(".") == 11 and key.split(".")[1] == "AAA+BBB"


def test_gdp_is_taken_from_the_expenditure_table_and_output_gdp_kept_apart(patch):
    rows = _main_rows("AAA")
    for r in rows:
        r["REF_AREA"] = "AAA"
    # same B1GQ in the output table (T0101) with a different value
    extra = [_row(NAMAIN, year=y, value=999.0, TRANSACTION="B1GQ", TABLE_IDENTIFIER="T0101",
                  REF_AREA="AAA") for y in (2019, 2020, 2021)]
    extra += [_row(NAMAIN, year=y, value=400.0, TRANSACTION="B1G", ACTIVITY=a,
                   TABLE_IDENTIFIER="T0101", REF_AREA="AAA")
              for y in (2019, 2020, 2021)
              for a in ("_T", "A", "BTE", "C", "F", "GTI", "J", "K", "L", "M_N", "OTQ", "RTU")]
    patch({"DF_TABLE1": _frame(extra + rows)})
    p = ap.ana_panel(["AAA"], output=True)
    assert (p["gdp"] != 999.0).all()
    assert (p["gdp_output"] == 999.0).all()
    assert (p["va_services"] == 7 * 400.0).all()
    assert "va_mfg" in p and "taxes_prod" not in p      # not published -> absent
    meta = ap.ana_panel_meta(p).iloc[0]
    assert "taxes_prod" in meta["absent"]


def test_aggregates_are_dropped_and_codes_none_requests_ten_countries_at_a_time(patch):
    countries = [f"Q{chr(65 + i // 26)}{chr(65 + i % 26)}" for i in range(23)]
    rows = [r for c in countries[:2] + ["OECD", "EA20"] for r in _main_rows(c)]
    fake = patch({"DF_TABLE1": _frame(rows)}, countries=countries + ["OECD", "EA20", "DEU_F"])
    with pytest.warns(UserWarning, match="QAC: no GDP"):
        p = ap.ana_panel(None)
    codes = set(p.index.get_level_values("code"))
    assert codes == set(countries[:2])
    assert len(fake.calls) == 3                          # 23 countries -> 10 + 10 + 3
    assert all("OECD" not in k and "EA20" not in k for _, k in fake.calls)
    assert all(len(k.split(".")[1].split("+")) <= 10 for _, k in fake.calls)


def test_chain_volume_at_own_reference_year_is_rebased_to_2020(patch):
    rows = _main_rows("AAA", l_only=True)
    for r in rows:
        r["REF_AREA"] = "AAA"
    patch({"DF_TABLE1": _frame(rows)})
    p = ap.ana_panel(["AAA"])
    t20 = pd.Timestamp("2020-01-01")
    assert p.loc[("AAA", t20), "gdp_real"] == pytest.approx(p.loc[("AAA", t20), "gdp"])
    assert p.loc[("AAA", t20), "gdp_defl"] == pytest.approx(100.0)
    meta = ap.ana_panel_meta(p).iloc[0]
    assert meta["volume_base"] == "L->2020" and "gdp" in meta["volume_from_l"]


def test_provider_failure_returns_an_empty_frame_with_missing_and_a_warning(patch):
    patch({"DF_TABLE1": _empty("HTTP 429")})
    with pytest.warns(UserWarning, match="HTTP 429"):
        p = ap.ana_panel(["AAA", "BBB"], income=True, labor=True)
    assert p.empty
    assert list(p.index.names) == ["code", "date"]
    for key in ("meta", "variables", "missing", "source", "fetched_at"):
        assert key in p.attrs
    assert p.attrs["missing"][0]["status"] == "HTTP 429"
    assert p.attrs["missing"][0]["codes"] == ("AAA", "BBB")
    long = None
    with pytest.warns(UserWarning):
        long = ap.ana_panel(["AAA"], long=True)
    assert list(long.columns) == ["code", "date", "variable", "value"] and long.empty


def test_a_failed_secondary_block_keeps_the_main_block_and_is_listed(patch):
    rows = _main_rows("AAA")
    for r in rows:
        r["REF_AREA"] = "AAA"
    patch({"DF_TABLE1": _frame(rows), "DF_TABLE3_EMPDC": _empty("timeout")})
    with pytest.warns(UserWarning, match="timeout"):
        p = ap.ana_panel(["AAA"], labor=True)
    assert p["gdp"].notna().all()
    blocks = {(m["block"], m["status"]) for m in p.attrs["missing"]}
    assert blocks == {("labor", "timeout")}               # EMPDC failed: re-run fixes it
    empty = {(m["block"], m["status"]) for m in p.attrs["chunks_empty"]}
    assert empty == {("labor", "HTTP 404")}               # POP: nothing published
    sent = [(r["block"], r["status"], r["rows"]) for r in p.attrs["requests"]]
    assert sent[0][:2] == ("main", "ok") and sent[0][2] > 0 and len(sent) == 3
    gdp = next(v for v in p.attrs["variables"] if v["name"] == "gdp")
    assert gdp["keys"] and gdp["keys"][0].startswith("A.AAA.")
    meta = ap.ana_panel_meta(p).iloc[0]
    assert meta["unit_mult"] is None and meta["units"] is None


def test_a_header_only_answer_is_an_empty_chunk_and_does_not_warn(patch):
    patch({"DF_TABLE1": _frame(_main_rows("AAA")),
           "DF_TABLE3_EMPDC": _frame([]), "DF_TABLE3_POP_EMPNC": _empty("HTTP 404")})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = ap.ana_panel(["AAA"], labor=True)
    assert p.attrs["missing"] == ()
    assert {m["status"] for m in p.attrs["chunks_empty"]} == {"no rows", "HTTP 404"}


def test_a_body_that_is_not_sdmx_csv_is_a_failure_that_warns(patch):
    html = pd.DataFrame({"<html><body>Service under maintenance</body></html>": []})
    html.loc[0] = ["x"]
    html.attrs["status"] = "ok"
    patch({"DF_TABLE1": _frame(_main_rows("AAA")), "DF_TABLE9A": html})
    with pytest.warns(UserWarning, match="no OBS_VALUE"):
        p = ap.ana_panel(["AAA"], stocks=True)
    assert p["gdp"].notna().all()
    assert ("stocks", "unexpected response (no OBS_VALUE)") in {
        (m["block"], m["status"]) for m in p.attrs["missing"]}


@pytest.mark.parametrize("exc", [http.client.IncompleteRead(b""), EOFError("gzip"),
                                 RuntimeError("anything else")])
def test_a_truncated_transfer_never_raises_out_of_the_panel(monkeypatch, exc):
    def boom(*a, **k):
        raise exc
    monkeypatch.setattr(ap, "oecd_csv", boom)
    with pytest.warns(UserWarning, match=type(exc).__name__):
        p = ap.ana_panel(["USA"])
    assert p.empty and list(p.index.names) == ["code", "date"]
    assert p.attrs["missing"][0]["status"] == type(exc).__name__
    assert {m["what"] for m in p.attrs["missing"]} == {"request", "country"}


def test_a_truncated_read_deep_in_the_transport_lands_in_missing(monkeypatch):
    import time as _time

    import puremacro._http as http_mod

    def boom(*a, **k):
        raise http.client.IncompleteRead(b"partial")
    monkeypatch.setattr(http_mod, "safe_get_bytes_cached", boom)
    monkeypatch.setattr(_time, "sleep", lambda s: None)
    monkeypatch.setattr(ap, "_PAUSE", 0.0)
    monkeypatch.setattr(ap, "_availability", lambda *a, **k: (_ for _ in ()).throw(EOFError()))
    with pytest.warns(UserWarning):
        p = ap.ana_panel(["USA"])
    assert p.empty and p.attrs["missing"][0]["status"] == "IncompleteRead"


def test_availability_that_raises_falls_back_to_the_frozen_list(monkeypatch):
    def boom(*a, **k):
        raise EOFError("truncated gzip")
    monkeypatch.setattr(ap, "oecd_availability", boom)
    with pytest.warns(UserWarning, match="frozen"):
        codes = ap.ana_countries()
    assert len(codes) == 64 and "USA" in codes


def test_a_failed_table5a_does_not_fall_back_to_the_stale_table5(patch):
    fake = patch({"DF_TABLE1": _frame(_main_rows("AAA")),
                  "DF_TABLE5A_T117": _empty("HTTP 429")})
    with pytest.warns(UserWarning, match="HTTP 429"):
        p = ap.ana_panel(["AAA"], durability=True)
    assert not any("DF_TABLE5_T117" in f for f, _ in fake.calls)
    assert "cons_dur" not in p.columns
    meta = ap.ana_panel_meta(p).set_index("code")
    assert meta.loc["AAA", "durability_table"] == ""


def test_each_series_comes_whole_from_one_table_and_exceptions_are_in_meta(patch):
    rows = [r for r in _main_rows("AAA") if not (r["TRANSACTION"] == "B1GQ"
                                                 and r["PRICE_BASE"] == "LR")]
    # GDP LR: output table 0101 for every year, expenditure table 0102 for 2021 only
    for y, v in ((2019, 900.0), (2020, 1000.0), (2021, 1100.0)):
        rows.append(_row(NAMAIN, year=y, value=v, REF_AREA="AAA", TRANSACTION="B1GQ",
                         PRICE_BASE="LR", TABLE_IDENTIFIER="T0101"))
    rows.append(_row(NAMAIN, year=2021, value=5555.0, REF_AREA="AAA", TRANSACTION="B1GQ",
                     PRICE_BASE="LR", TABLE_IDENTIFIER="T0102"))
    patch({"DF_TABLE1": _frame(rows)})
    p = ap.ana_panel(["AAA"])
    real = p.loc["AAA", "gdp_real"]
    assert real.notna().sum() == 1 and real.iloc[-1] == 5555.0   # T0102 only, no splice
    rows = [r for r in rows if not (r["TABLE_IDENTIFIER"] == "T0102"
                                    and r["TRANSACTION"] == "B1GQ" and r["PRICE_BASE"] == "LR")]
    patch({"DF_TABLE1": _frame(rows)})
    p = ap.ana_panel(["AAA"])
    assert list(p.loc["AAA", "gdp_real"]) == [900.0, 1000.0, 1100.0]
    meta = ap.ana_panel_meta(p).set_index("code")
    assert meta.loc["AAA", "tables_not_t0102"] == ("gdp:LR=T0101",)


@pytest.mark.parametrize("kwargs", [
    {"codes": ["US"]}, {"codes": ["OECD"]}, {"codes": []},
    {"codes": ["USA"], "start": "19x0"}, {"codes": ["USA"], "start": "1850"}])
def test_caller_errors_raise_value_error_before_any_request(patch, kwargs):
    fake = patch({})
    with pytest.raises(ValueError):
        ap.ana_panel(**kwargs)
    assert fake.calls == []


def test_durability_takes_table5a_first_and_table5_only_for_the_rest(patch):
    main = []
    for c in ("AAA", "BBB"):
        for r in _main_rows(c):
            r["REF_AREA"] = c
            main.append(r)

    def dur(code, table):
        return [_row(NAMAIN, year=y, value=v, REF_AREA=code, SECTOR="S14", TRANSACTION=t,
                     EXPENDITURE="_T", TABLE_IDENTIFIER=table)
                for y in (2019, 2020, 2021)
                for t, v in (("P311", 50.0), ("P312", 40.0), ("P313", 200.0), ("P314", 300.0))]

    fake = patch({"DF_TABLE1": _frame(main),
                  "DF_TABLE5A_T117": _frame(dur("AAA", "T0500")),
                  "DF_TABLE5_T117": _frame(dur("BBB", "T0117"))})
    p = ap.ana_panel(["AAA", "BBB"], durability=True)
    assert (p["cons_dur"] == 50.0).all()
    meta = ap.ana_panel_meta(p).set_index("code")
    assert meta.loc["AAA", "durability_table"] == "TABLE5A_T117"
    assert meta.loc["BBB", "durability_table"] == "TABLE5_T117"
    t5_keys = [k for f, k in fake.calls if "DF_TABLE5_T117" in f]
    assert t5_keys and t5_keys[0].split(".")[1] == "BBB"


def test_income_split_reads_the_uses_side_and_vat_from_resources(patch):
    main = _main_rows("AAA")
    for r in main:
        r["REF_AREA"] = "AAA"
    main += [_row(NAMAIN, year=y, value=v, REF_AREA="AAA", TRANSACTION=t, ACTIVITY=a,
                  TABLE_IDENTIFIER="T0103")
             for y in (2019, 2020, 2021)
             for t, a, v in (("D1", "_T", 500.0), ("B2A3G", "_T", 400.0),
                             ("D2X3", "_Z", 100.0), ("B1GQ", "_Z", 1000.0))]
    t14 = [_row(NASEC, year=y, value=v, REF_AREA="AAA", ACCOUNTING_ENTRY=ae, TRANSACTION=t)
           for y in (2019, 2020, 2021)
           for t, ae, v in (("B2G", "D", 300.0), ("B2G", "C", 300.0), ("B3G", "D", 100.0),
                            ("P51C", "D", 150.0), ("D211", "C", 70.0),
                            ("D211", "D", -1.0))]
    patch({"DF_TABLE1": _frame(main), "DF_TABLE14": _frame(t14)})
    p = ap.ana_panel(["AAA"], income=True)
    assert (p["surplus_gross"] + p["mixed_income"] == p["surplus_mixed"]).all()
    assert (p["vat"] == 70.0).all() and (p["cfc"] == 150.0).all()
    assert (p["gdp_income"] == 1000.0).all()
    assert "comp_emp_defl" not in p and "comp_emp_real" not in p


def test_household_investment_falls_back_to_s1m_where_s14_is_not_published(patch):
    main = []
    for c in ("AAA", "BBB"):
        for r in _main_rows(c):
            r["REF_AREA"] = c
            main.append(r)
    gf = [_row(NASEC, year=y, value=v, REF_AREA=c, SECTOR=s, TRANSACTION="P51G",
               INSTR_ASSET="N11G")
          for y in (2019, 2020, 2021)
          for c, s, v in (("AAA", "S13", 40.0), ("AAA", "S14", 30.0), ("AAA", "S1M", 31.0),
                          ("BBB", "S13", 20.0), ("BBB", "S1M", 25.0))]
    patch({"DF_TABLE1": _frame(main), "DF_TABLE14_GFCF": _frame(gf)})
    p = ap.ana_panel(["AAA", "BBB"], sectors=True)
    assert (p.loc["AAA", "inv_hh"] == 30.0).all() and (p.loc["BBB", "inv_hh"] == 25.0).all()
    assert (p.loc["AAA", "inv_gov"] == 40.0).all()
    meta = ap.ana_panel_meta(p).set_index("code")
    assert meta.loc["AAA", "inv_hh_sector"] == "S14" and meta.loc["BBB", "inv_hh_sector"] == "S1M"
    assert "inv_gov_defl" not in p


def test_labor_block_suffixes_activities_and_puts_weekly_hours_on_an_annual_basis(patch):
    main = []
    for c in ("AAA", "NZL"):
        for r in _main_rows(c):
            r["REF_AREA"] = c
            main.append(r)
    emp = []
    for c, h in (("AAA", 1700.0), ("NZL", 1700.0 / 52)):
        for y in (2019, 2020, 2021):
            for act, share in (("_T", 1.0), ("A", 0.1), ("OTQ", 0.3)):
                emp.append(_row(NAMAIN, year=y, value=1000.0 * share, mult=3, REF_AREA=c,
                                TRANSACTION="EMP", ACTIVITY=act, UNIT_MEASURE="PS",
                                PRICE_BASE="_Z", TABLE_IDENTIFIER="T0111"))
                emp.append(_row(NAMAIN, year=y, value=h * share, mult=6, REF_AREA=c,
                                TRANSACTION="EMP", ACTIVITY=act, UNIT_MEASURE="H",
                                PRICE_BASE="_Z", TABLE_IDENTIFIER="T0111"))
    pop = [_row(NAMAIN, year=y, value=5000.0, mult=3, REF_AREA=c, TRANSACTION="POP",
                UNIT_MEASURE="PS", PRICE_BASE="_Z") for y in (2019, 2020, 2021)
           for c in ("AAA", "NZL")]
    patch({"DF_TABLE1": _frame(main), "DF_TABLE3_EMPDC": _frame(emp),
           "DF_TABLE3_POP_EMPNC": _frame(pop)})
    p = ap.ana_panel(["AAA", "NZL"], labor=True)
    assert {"emp", "emp_agri", "emp_public", "hours", "hours_agri", "pop"} <= set(p.columns)
    hpw = p["hours"] * 1e3 / p["emp"]
    assert hpw.loc["AAA"].iloc[0] == pytest.approx(1700.0)
    assert hpw.loc["NZL"].iloc[0] == pytest.approx(1700.0)
    assert p.loc["NZL", "hours_agri"].iloc[0] == pytest.approx(170.0)
    assert ap.ana_panel_meta(p).set_index("code").loc["NZL", "hours_scale"] == 52.0
    assert "emp_defl" not in p and "pop_real" not in p


def test_capital_stocks_rebase_their_own_reference_year_volumes(patch):
    main = _main_rows("AAA")
    for r in main:
        r["REF_AREA"] = "AAA"
    k = [_row(NAMAIN, year=y, value=v, REF_AREA="AAA", TRANSACTION="LE", INSTR_ASSET=a,
              ACTIVITY="_T", PRICE_BASE=pb, TABLE_IDENTIFIER="T2000")
         for y, f in ((2019, 0.9), (2020, 1.0), (2021, 1.1))
         for a, lvl in (("N11N", 3000.0), ("N11G", 5000.0))
         for pb, v in (("V", lvl * f), ("L", lvl * 0.8))]
    kg = [_row(NASEC, year=y, value=900.0, REF_AREA="AAA", SECTOR="S13", ACCOUNTING_ENTRY="A",
               TRANSACTION="_Z", INSTR_ASSET="N11N", VALUATION="_Z") for y in (2019, 2020, 2021)]
    patch({"DF_TABLE1": _frame(main), "DF_TABLE9A": _frame(k), "DF_TABLE9B": _frame(kg)})
    p = ap.ana_panel(["AAA"], stocks=True)
    t20 = pd.Timestamp("2020-01-01")
    assert p.loc[("AAA", t20), "k_net_real"] == pytest.approx(3000.0)
    assert p.loc[("AAA", t20), "k_net_defl"] == pytest.approx(100.0)
    assert (p["k_net_gov"] == 900.0).all()


def test_long_form_has_one_row_per_observation(patch):
    rows = _main_rows("AAA")
    for r in rows:
        r["REF_AREA"] = "AAA"
    patch({"DF_TABLE1": _frame(rows)})
    long = ap.ana_panel(["AAA"], long=True, real=False)
    assert list(long.columns) == ["code", "date", "variable", "value"]
    assert long["value"].notna().all()
    assert not long["variable"].str.endswith("_real").any()
    assert set(long["variable"]) >= {"gdp", "gdp_defl"}


def test_ana_countries_drops_aggregates_and_falls_back_offline(monkeypatch):
    payload = {"data": {"contentConstraints": [{"cubeRegions": [{"keyValues": [
        {"id": "REF_AREA", "values": ["USA", "EA20", "OECD", "MEX", "DEU_F"]}]}]}]}}
    monkeypatch.setattr(ap, "_availability", lambda *a, **k: payload)
    assert ap.ana_countries() == ["MEX", "USA"]
    monkeypatch.setattr(ap, "_availability", lambda *a, **k: {})
    with pytest.warns(UserWarning, match="frozen"):
        codes = ap.ana_countries()
    assert len(codes) == 64 and "USA" in codes and "OECD" not in codes


def test_every_key_has_the_dimension_count_of_its_dsd(patch):
    rows = _main_rows("AAA")
    for r in rows:
        r["REF_AREA"] = "AAA"
    fake = patch({"DF_TABLE1": _frame(rows)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ap.ana_panel(["AAA"], assets=True, durability=True, income=True, output=True,
                     labor=True, sectors=True, stocks=True)
    for flow, key in fake.calls:
        n = 13 if "NASEC10" in flow else 12
        assert key.count(".") == n - 1, (flow, key)
        assert key.startswith("A.AAA.")


@pytest.mark.network
def test_live_three_country_main_block():
    p = ap.ana_panel(["USA", "FRA", "MEX"], start="2015")
    assert set(p.index.get_level_values("code")) == {"USA", "FRA", "MEX"}
    t20 = pd.Timestamp("2020-01-01")
    assert p.xs(t20, level="date")["gdp_defl"].round(6).eq(100.0).all()


def test_stocks_published_in_thousands_are_rescaled_and_recorded():
    import warnings as _w
    import pandas as _pd
    from puremacro.fetch import oecd_ana_panel as m
    idx = _pd.MultiIndex.from_product([["ISL", "NOR"], _pd.to_datetime(["2020-01-01", "2021-01-01"])],
                                      names=["code", "date"])
    out = _pd.DataFrame({"gdp": [100.0, 110.0, 200.0, 210.0],
                         "k_net": [300000.0, 320000.0, 600.0, 620.0],
                         "k_net_real": [290000.0, 300000.0, 590.0, 600.0]}, index=idx)
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        fixed, corr = m._fix_stock_scale(out.copy())
    assert fixed.loc["ISL", "k_net"].tolist() == [300.0, 320.0]
    assert fixed.loc["ISL", "k_net_real"].tolist() == [290.0, 300.0]
    assert fixed.loc["NOR", "k_net"].tolist() == [600.0, 620.0]          # plausible: untouched
    assert [c["code"] for c in corr] == ["ISL"] and corr[0]["factor"] == 1e-3
    assert any("ISL" in str(r.message) for r in rec)
    # Once corrected at the source, the rule no longer fires.
    again, corr2 = m._fix_stock_scale(fixed.copy())
    assert corr2 == [] and again.equals(fixed)
