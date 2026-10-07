"""Offline tests for ``puremacro.fetch.wdi`` and the two older WDI callers.

Every request goes through the module's ``_get`` seam, patched here with tiny
synthetic World Bank API v2 bodies built in Python; nothing touches the
network except the tests marked ``network``.
"""
from __future__ import annotations

import json
import urllib.error

import numpy as np
import pandas as pd
import pytest

import puremacro.fetch.wdi as wdi
from puremacro._codes import IS_AGGREGATE, WB_AGGREGATES, is_country

LASTUPDATED = "2026-07-13"


# ---------------------------------------------------------------------------
# Synthetic API
# ---------------------------------------------------------------------------

def _country(code, iso2, name, region_id, region, income="High income"):
    return {"id": code, "iso2Code": iso2, "name": name,
            "region": {"id": region_id, "iso2code": "", "value": region},
            "incomeLevel": {"id": "HIC", "value": income}}


COUNTRIES = [
    _country("MEX", "MX", "Mexico", "LCN", "Latin America & Caribbean ", "Upper middle income"),
    _country("USA", "US", "United States", "NAC", "North America"),
    _country("DEU", "DE", "Germany", "ECS", "Europe & Central Asia"),
    _country("EMU", "XC", "Euro area", "NA", "Aggregates", "Aggregates"),
    _country("WLD", "1W", "World", "NA", "Aggregates", "Aggregates"),
    _country("HIC", "XD", "High income", "NA", "Aggregates", "Aggregates"),
]

#: indicator code -> {(iso3, year): value}; WDI publishes levels in full units.
SERIES = {
    "NY.GDP.MKTP.CN": {("MEX", 2020): 2.4e13, ("MEX", 2021): 2.6e13,
                       ("USA", 2020): 2.1e13, ("USA", 2021): 2.3e13,
                       ("EMU", 2020): 1.1e13, ("", 2020): 5.0e13},
    "NY.GDP.MKTP.KN": {("MEX", 2020): 2.0e13, ("MEX", 2021): 2.1e13,
                       ("USA", 2020): 2.1e13, ("USA", 2021): 2.2e13},
    "SP.POP.TOTL": {("MEX", 2020): 126_000_000, ("USA", 2020): 331_000_000,
                    ("DEU", 2020): 83_000_000, ("WLD", 2020): 7.8e9},
    "SL.UEM.TOTL.ZS": {("MEX", 2020): 4.4, ("USA", 2020): 8.1, ("USA", 2021): None},
    "NE.GDI.STKB.CN": {("MEX", 2020): -1.0e11},
    "NE.GDI.STKB.KN": {("MEX", 2020): -0.8e11},
    # capform: a negative nominal (MEX 2020) and a nominal zero (USA 2020, the
    # VEN 1991-2011 pattern); only USA 2021 is a valid deflator cell.
    "NE.GDI.TOTL.CN": {("MEX", 2020): -5.0e10, ("USA", 2020): 0.0, ("USA", 2021): 4.0e12},
    "NE.GDI.TOTL.KN": {("MEX", 2020): 4.0e10, ("USA", 2020): 3.0e12, ("USA", 2021): 3.2e12},
    "NY.TAX.NIND.CN": {("MEX", 2020): -2.0e11, ("USA", 2020): 1.0e12},
    "NY.TAX.NIND.KN": {("MEX", 2020): 1.0e11, ("USA", 2020): 0.9e12},
}


def _rows(code):
    out = []
    for (iso3, year), value in SERIES[code].items():
        out.append({"indicator": {"id": code, "value": f"label of {code}"},
                    "country": {"id": iso3[:2], "value": iso3},
                    "countryiso3code": iso3, "date": str(year), "value": value,
                    "unit": "", "obs_status": "", "decimal": 0})
    return out


def _page(rows, page=1, pages=1, total=None):
    meta = {"page": page, "pages": pages, "per_page": 32767,
            "total": len(rows) if total is None else total,
            "sourceid": "2", "lastupdated": LASTUPDATED}
    return json.dumps([meta, rows]).encode()


METADATA = {
    "page": 1, "pages": 1, "per_page": "20000", "total": 12,
    "source": [{"id": "2", "name": "World Development Indicators", "concept": [{
        "id": "Country", "variable": [
            {"id": "MEX", "metatype": [
                {"id": "CurrencyUnit", "value": "Mexican peso"},
                {"id": "Nationalaccountsbaseyear", "value": "2018"},
                {"id": "SNApricevaluation", "value": "Value added at basic prices (VAB)"},
                {"id": "SystemofNationalAccounts", "value": "Country uses the 2008 System of National Accounts methodology"},
                {"id": "LongName", "value": "United Mexican States"}]},
            {"id": "USA", "metatype": [
                {"id": "CurrencyUnit", "value": "U.S. dollar"},
                {"id": "Nationalaccountsbaseyear", "value": "Original chained constant price data are rescaled."},
                {"id": "Nationalaccountsreferenceyear", "value": "2020"},
                {"id": "SpecialNotes", "value": "Fiscal year ends on September 30."}]},
            {"id": "EMU", "metatype": [{"id": "CurrencyUnit", "value": "Euro"}]},
        ]}]}],
}


class FakeAPI:
    """Answers ``_get(url, ...)`` from the synthetic tables; records every URL."""

    def __init__(self, fail=None, split_pages=(), short=(), in_band=()):
        self.urls: list[str] = []
        self.fail = fail or {}            # substring -> exception to raise
        self.split_pages = set(split_pages)
        self.short = set(short)
        self.in_band = set(in_band)

    def __call__(self, url, **kw):
        self.urls.append(url)
        for key, exc in self.fail.items():
            if key in url:
                raise exc
        if "/sources/2/country/all/metadata" in url:
            return json.dumps(METADATA).encode()
        if "/country?format=json" in url:
            return _page(COUNTRIES)
        code = url.split("/indicator/")[1].split("?")[0]
        if code in self.in_band or code not in SERIES:
            return json.dumps([{"message": [{"id": "120", "key": "Invalid value",
                                             "value": "The provided parameter value is not valid"}]}]).encode()
        rows = _rows(code)
        page = int(url.rsplit("&page=", 1)[1]) if "&page=" in url else 1
        if code in self.short:
            return _page(rows, total=len(rows) + 5)
        if code in self.split_pages:
            half = len(rows) // 2
            return _page(rows[:half] if page == 1 else rows[half:], page=page, pages=2,
                         total=len(rows))
        return _page(rows)


@pytest.fixture
def api(monkeypatch):
    fake = FakeAPI()
    monkeypatch.setattr(wdi, "_get", fake)
    monkeypatch.setattr(wdi, "_RETRY_SLEEP", 0.0)
    return fake


# ---------------------------------------------------------------------------
# Economies and metadata
# ---------------------------------------------------------------------------

def test_wdi_countries_keeps_economies_and_drops_every_aggregate(api):
    c = wdi.wdi_countries()
    assert list(c.index) == ["DEU", "MEX", "USA"]
    assert c.index.name == "code"
    assert list(c.columns) == ["iso2", "name", "region", "income_group"]
    assert c.loc["MEX", "region"] == "Latin America & Caribbean"
    assert set(c.attrs["aggregates"]) == {"EMU", "HIC", "WLD"}
    assert c.attrs["missing"] == ()
    assert isinstance(c.attrs["fetched_at"], str) and c.attrs["fetched_at"].endswith("+00:00")


def test_wdi_countries_returns_an_empty_frame_and_warns_when_the_api_fails(monkeypatch):
    monkeypatch.setattr(wdi, "_get", FakeAPI(fail={"/country?": urllib.error.HTTPError(
        "u", 503, "down", {}, None)}))
    monkeypatch.setattr(wdi, "_RETRY_SLEEP", 0.0)
    with pytest.warns(UserWarning, match="HTTP 503"):
        c = wdi.wdi_countries()
    assert c.empty and c.index.name == "code"
    assert c.attrs["missing"][0]["reason"] == "HTTP 503"


def test_wdi_meta_reads_currency_base_year_and_notes_for_economies_only(api):
    m = wdi.wdi_meta()
    assert list(m.index) == ["MEX", "USA"]            # EMU dropped, DEU has no record
    assert m.loc["MEX", "currency"] == "Mexican peso"
    assert m.loc["MEX", "base_year"] == "2018"
    assert m.loc["MEX", "name"] == "Mexico"
    assert m.loc["USA", "ref_year"] == "2020"
    assert "Fiscal year" in m.loc["USA", "notes"]
    assert "2008" in m.loc["MEX", "sna"]
    assert [d["item"] for d in m.attrs["missing"]] == ["DEU"]


def test_wdi_meta_rejects_an_aggregate_code(api):
    with pytest.raises(ValueError, match="EMU"):
        wdi.wdi_meta(["MEX", "EMU"])


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------

def test_wdi_panel_returns_period_start_annual_dates_and_house_units(api):
    p = wdi.wdi_panel(["gdp", "pop", "urate"], start=2020, end=2021)
    assert p.index.names == ["code", "date"]
    assert str(p.index.get_level_values("date").dtype).startswith("datetime64")
    dates = p.index.get_level_values("date")
    assert ((dates.month == 1) & (dates.day == 1)).all()
    assert list(p.columns) == ["gdp", "gdp_real", "gdp_defl", "pop", "urate"]
    assert all(np.issubdtype(t, np.floating) for t in p.dtypes)
    # millions of LCU, thousands of persons, percent as published
    assert p.loc[("MEX", pd.Timestamp("2020-01-01")), "gdp"] == pytest.approx(2.4e7)
    assert p.loc[("USA", pd.Timestamp("2020-01-01")), "pop"] == pytest.approx(331_000)
    assert p.loc[("USA", pd.Timestamp("2020-01-01")), "urate"] == pytest.approx(8.1)


def test_wdi_panel_derives_the_deflator_as_100_times_nominal_over_real(api):
    p = wdi.wdi_panel(["gdp"], start=2020, end=2021)
    row = p.loc[("MEX", pd.Timestamp("2021-01-01"))]
    assert row["gdp_defl"] == pytest.approx(100 * 2.6e13 / 2.1e13)
    assert p.loc[("USA", pd.Timestamp("2020-01-01")), "gdp_defl"] == pytest.approx(100.0)


def test_wdi_panel_drops_aggregates_and_blank_iso3_income_groups(api):
    p = wdi.wdi_panel(["gdp", "pop"], start=2020, end=2021, real=False)
    codes = set(p.index.get_level_values("code"))
    assert codes == {"DEU", "MEX", "USA"}
    assert not codes & set(WB_AGGREGATES)


def test_wdi_panel_writes_one_meta_dict_per_column_with_the_release_date(api):
    p = wdi.wdi_panel(["gdp", "pop"], start=2020, end=2021)
    meta = p.attrs["meta"]
    assert isinstance(meta, tuple) and all(isinstance(m, dict) for m in meta)
    assert [m["variable"] for m in meta] == list(p.columns)
    gdp = meta[0]
    assert gdp["indicator"] == "NY.GDP.MKTP.CN"
    assert gdp["units"] == "millions of current LCU" and gdp["unit_mult"] == 6
    assert gdp["lastupdated"] == LASTUPDATED
    assert (gdp["first"], gdp["last"], gdp["n"], gdp["n_codes"]) == (2020, 2021, 4, 2)
    assert meta[-1]["unit_mult"] == 3                       # pop in thousands
    assert p.attrs["lastupdated"] == LASTUPDATED
    assert p.attrs["missing"] == ()
    assert "World Bank" in p.attrs["source"]
    pd.concat([p, p])                                      # plain-dict meta survives concat


def test_wdi_panel_sends_one_request_per_indicator_and_names_the_codes(api):
    wdi.wdi_panel(["gdp", "pop"], ["usa", "MEX"], start=2020, end=2021)
    data_urls = [u for u in api.urls if "/indicator/" in u]
    assert len(data_urls) == 3                              # gdp, gdp_real, pop
    assert all("/country/MEX;USA/indicator/" in u for u in data_urls)
    assert all("date=2020:2021" in u and "per_page=32767" in u for u in data_urls)


def test_wdi_panel_follows_every_page(monkeypatch):
    fake = FakeAPI(split_pages={"NY.GDP.MKTP.CN"})
    monkeypatch.setattr(wdi, "_get", fake)
    p = wdi.wdi_panel(["gdp"], start=2020, end=2021, real=False)
    assert p["gdp"].notna().sum() == 4
    assert any(u.endswith("&page=2") for u in fake.urls)


def test_wdi_panel_reports_a_short_read_instead_of_truncating(monkeypatch):
    monkeypatch.setattr(wdi, "_get", FakeAPI(short={"SP.POP.TOTL"}))
    with pytest.warns(UserWarning, match="incomplete pagination"):
        p = wdi.wdi_panel(["gdp", "pop"], start=2020, end=2021, real=False)
    assert "pop" not in p.columns and "gdp" in p.columns
    assert p.attrs["missing"][0]["variable"] == "pop"


def test_wdi_panel_turns_an_in_band_error_into_attrs_missing(monkeypatch):
    monkeypatch.setattr(wdi, "_get", FakeAPI(in_band={"SP.POP.TOTL"}))
    with pytest.warns(UserWarning, match="WDI message 120"):
        p = wdi.wdi_panel(["gdp", "pop"], start=2020, end=2021)
    assert list(p.columns) == ["gdp", "gdp_real", "gdp_defl"]
    (miss,) = p.attrs["missing"]
    assert miss["indicator"] == "SP.POP.TOTL" and "120" in miss["reason"]


def test_wdi_panel_returns_an_empty_frame_when_every_request_fails(monkeypatch):
    monkeypatch.setattr(wdi, "_get", FakeAPI(fail={"/indicator/": TimeoutError("slow")}))
    monkeypatch.setattr(wdi, "_RETRY_SLEEP", 0.0)
    with pytest.warns(UserWarning, match="TimeoutError"):
        p = wdi.wdi_panel(["gdp", "pop"], start=2020, end=2021)
    assert p.empty and p.index.names == ["code", "date"]
    assert list(p.columns) == ["gdp", "gdp_real", "gdp_defl", "pop"]
    assert str(p.index.get_level_values("date").dtype) == "datetime64[ns]"
    assert {m["variable"] for m in p.attrs["missing"]} == {"gdp", "gdp_real", "pop"}
    assert set(p.attrs) >= {"meta", "source", "fetched_at", "missing"}


def test_wdi_panel_returns_an_empty_frame_when_the_country_list_fails(monkeypatch):
    monkeypatch.setattr(wdi, "_get", FakeAPI(fail={"/country?": OSError("offline")}))
    monkeypatch.setattr(wdi, "_RETRY_SLEEP", 0.0)
    with pytest.warns(UserWarning):
        p = wdi.wdi_panel(["pop"], long=True)
    assert p.empty and list(p.columns) == ["code", "date", "variable", "value"]
    assert p.attrs["missing"][0]["reason"] == "country list unavailable"


def test_wdi_panel_retries_once_after_a_timeout(monkeypatch):
    calls = {"n": 0}
    inner = FakeAPI()

    def flaky(url, **kw):
        if "/indicator/" in url and calls["n"] == 0:
            calls["n"] += 1
            raise TimeoutError("stall")
        return inner(url, **kw)

    monkeypatch.setattr(wdi, "_get", flaky)
    monkeypatch.setattr(wdi, "_RETRY_SLEEP", 0.0)
    p = wdi.wdi_panel(["pop"], start=2020, end=2020)
    assert p["pop"].notna().sum() == 3 and p.attrs["missing"] == ()


def test_wdi_panel_skips_the_deflator_of_a_sign_changing_item(api):
    p = wdi.wdi_panel(["inventories"], start=2020, end=2020)
    assert list(p.columns) == ["inventories", "inventories_real"]


def test_wdi_panel_masks_deflators_of_zero_or_negative_levels_and_counts_them(api):
    p = wdi.wdi_panel(["capform", "taxes_prod"], start=2020, end=2021)
    assert "taxes_prod_defl" not in p.columns and "taxes_prod_real" in p.columns
    d = p["capform_defl"]
    assert np.isnan(d.loc[("MEX", pd.Timestamp("2020-01-01"))])
    assert np.isnan(d.loc[("USA", pd.Timestamp("2020-01-01"))])
    assert d.loc[("USA", pd.Timestamp("2021-01-01"))] == pytest.approx(125.0)
    meta = {m["variable"]: m for m in p.attrs["meta"]}
    assert meta["capform_defl"]["n_masked"] == 2 and meta["capform_defl"]["n"] == 1


def test_wdi_panel_full_and_empty_frames_share_the_datetime_unit(api, monkeypatch):
    full = wdi.wdi_panel(["gdp"], start=2020, end=2021)
    monkeypatch.setattr(wdi, "_get", FakeAPI(fail={"/indicator/": OSError("down")}))
    with pytest.warns(UserWarning):
        empty = wdi.wdi_panel(["gdp"], start=2020, end=2021)
    assert full.index.get_level_values("date").dtype == empty.index.get_level_values("date").dtype
    assert list(full.columns) == list(empty.columns) == ["gdp", "gdp_real", "gdp_defl"]


def test_wdi_panel_refetches_a_cached_in_band_error_once_before_reporting_it(monkeypatch):
    inner = FakeAPI()
    seen: list[bool] = []

    def poisoned(url, **kw):
        if "SP.POP.TOTL" in url:
            seen.append(kw.get("refresh", False))
            if not kw.get("refresh", False):     # the stale copy in the cache
                return json.dumps([{"message": [{"id": "120", "key": "x", "value": "y"}]}]).encode()
        return inner(url, **kw)

    monkeypatch.setattr(wdi, "_get", poisoned)
    p = wdi.wdi_panel(["pop"], start=2020, end=2020)
    assert seen == [False, True]
    assert p["pop"].notna().sum() == 3 and p.attrs["missing"] == ()


def test_wdi_panel_refetches_a_cached_non_json_body_once(monkeypatch):
    inner = FakeAPI()
    seen: list[bool] = []

    def poisoned(url, **kw):
        if "SP.POP.TOTL" in url:
            seen.append(kw.get("refresh", False))
            if not kw.get("refresh", False):
                return b"<html>maintenance</html>"
        return inner(url, **kw)

    monkeypatch.setattr(wdi, "_get", poisoned)
    p = wdi.wdi_panel(["pop"], start=2020, end=2020)
    assert seen == [False, True] and p.attrs["missing"] == ()


def test_wdi_panel_long_form_has_one_row_per_observation(api):
    p = wdi.wdi_panel(["gdp", "urate"], start=2020, end=2021, long=True)
    assert list(p.columns) == ["code", "date", "variable", "value"]
    assert p["value"].notna().all()
    assert set(p["variable"]) == {"gdp", "gdp_real", "gdp_defl", "urate"}
    assert len(p) == 4 * 3 + 2                              # USA 2021 urate is null
    assert p.attrs["meta"][0]["variable"] == "gdp"


def test_wdi_panel_accepts_a_mapping_of_raw_codes_unscaled(api):
    p = wdi.wdi_panel({"population_raw": "SP.POP.TOTL"}, ["MEX"], start=2020, end=2020)
    assert p.iloc[0, 0] == 126_000_000
    assert p.attrs["meta"][0]["unit_mult"] == 0


def test_wdi_panel_real_false_skips_the_constant_price_twins(api):
    p = wdi.wdi_panel(["gdp"], start=2020, end=2021, real=False)
    assert list(p.columns) == ["gdp"]
    assert not any("KN" in u for u in api.urls)


def test_wdi_panel_rejects_an_unknown_variable_and_lists_the_registry(api):
    with pytest.raises(ValueError, match="valid names.*cons_hh"):
        wdi.wdi_panel(["gdp", "durables"])


def test_wdi_panel_rejects_codes_that_are_not_economies(api):
    with pytest.raises(ValueError, match="OED"):
        wdi.wdi_panel(["gdp"], ["MEX", "OED"])


def test_wdi_panel_rejects_an_end_before_the_start(api):
    with pytest.raises(ValueError, match="before start"):
        wdi.wdi_panel(["gdp"], start=2000, end=1990)


def test_the_registry_covers_the_probe_table_and_the_core_is_inside_it():
    for name in ("gdp", "cons_hh", "inv", "capform", "va_mfg", "inv_priv", "cfc_usd",
                 "lf", "urate_nat", "family_share", "gdp_ppp", "fx_alt",
                 "gdp_pc_usd_real", "cpi_infl", "credit_priv_banks"):
        assert name in wdi.WDI_INDICATORS
    assert len(wdi.WDI_INDICATORS) == 52
    assert set(wdi.WDI_CORE) <= set(wdi.WDI_INDICATORS)
    assert all(code.endswith(".KN") for code in wdi.WDI_REAL.values())
    assert set(wdi.WDI_REAL) <= set(wdi.WDI_INDICATORS)


def test_module_imports_no_network_library_at_module_scope():
    import sys
    src = open(wdi.__file__, encoding="utf-8").read()
    head = src.split("def _get", 1)[0]
    for lib in ("requests", "ssl", "sqlite3", "urllib.request"):
        assert f"import {lib}" not in head
    assert "environ" not in src
    assert "puremacro.fetch.wdi" in sys.modules


# ---------------------------------------------------------------------------
# The shared aggregate list and the two older WDI callers
# ---------------------------------------------------------------------------

def test_every_world_bank_aggregate_fails_is_country():
    assert len(WB_AGGREGATES) == 79
    assert WB_AGGREGATES <= IS_AGGREGATE
    for code in ("EMU", "EUU", "OED", "ARB", "LCN", "HIC", "WLD", "XZN"):
        assert not is_country(code)
    for code in ("USA", "MEX", "XKX", "PRI", "HKG"):
        assert is_country(code)


def test_fetch_wdi_emissions_reads_every_page_and_drops_world_bank_aggregates(monkeypatch):
    import puremacro.fetch.emissions as em

    rows = [{"indicator": {"id": "EN.GHG.CO2.PC.CE.AR5"}, "countryiso3code": c,
             "date": "2020", "value": v} for c, v in (("USA", 13.0), ("EMU", 6.0))]
    rows2 = [{"indicator": {"id": "EN.GHG.CO2.PC.CE.AR5"}, "countryiso3code": "TUR",
              "date": "2020", "value": 5.0}]
    urls = []

    def fake(url, *, refresh=False, timeout=60):
        urls.append(url)
        if url.endswith("&page=2"):
            return json.dumps([{"page": 2, "pages": 2, "total": 3}, rows2]).encode()
        return json.dumps([{"page": 1, "pages": 2, "total": 3}, rows]).encode()

    monkeypatch.setattr(em, "_cached_get", fake)
    df = em.fetch_wdi_emissions(indicators=["EN.GHG.CO2.PC.CE.AR5"])
    assert set(df["code"]) == {"USA", "TUR"}
    assert "per_page=15000" in urls[0] and urls[1].endswith("&page=2")


def test_energy_transition_wdi_reader_reads_every_page_and_drops_aggregates(monkeypatch):
    import puremacro.fetch.energy_transition as et

    def body(code, page):
        rows = {("EG", 1): [("USA", 6800.0), ("OED", 4000.0)], ("EG", 2): [("TUR", 1700.0)],
                ("SP", 1): [("USA", 3.3e8), ("OED", 1.4e9)], ("SP", 2): [("TUR", 8.4e7)]}[(code, page)]
        recs = [{"countryiso3code": c, "date": "2020", "value": v} for c, v in rows]
        return json.dumps([{"page": page, "pages": 2, "total": 3}, recs]).encode()

    def fake(url, *, refresh=False, timeout=30.0):
        code = "EG" if "EG.USE" in url else "SP"
        return body(code, 2 if url.endswith("&page=2") else 1)

    monkeypatch.setattr(et, "cached_get", fake)
    df = et._fetch_wdi_primary_energy(start_year=2020)
    assert set(df["code"]) == {"USA", "TUR"}


def test_energy_transition_wdi_reader_warns_when_a_later_page_is_malformed(monkeypatch):
    import puremacro.fetch.energy_transition as et

    def fake(url, *, refresh=False, timeout=30.0):
        if url.endswith("&page=2"):
            return json.dumps([{"message": "boom"}]).encode()
        recs = [{"countryiso3code": "USA", "date": "2020", "value": 1.0}]
        return json.dumps([{"page": 1, "pages": 2, "total": 2}, recs]).encode()

    monkeypatch.setattr(et, "cached_get", fake)
    with pytest.warns(UserWarning, match="1 of 2 rows"):
        rows = et._wdi_all_pages("https://example.invalid/x?format=json", refresh=False, timeout=1.0)
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# Live
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_live_wdi_panel_has_the_217_economies_and_gdp_from_1960():
    p = wdi.wdi_panel(["gdp", "pop"], start=1960)
    assert p.attrs["missing"] == ()
    codes = set(p.index.get_level_values("code"))
    assert 210 <= len(codes) <= 217 and "USA" in codes and not codes & WB_AGGREGATES
    assert p.index.get_level_values("date").min() == pd.Timestamp("1960-01-01")
