"""Offline tests for :func:`puremacro.fetch.oecd_stes_panel.stes_panel`.

The SDMX transport is replaced by a fake ``_get_csv`` that builds a tiny
codes-only SDMX-CSV frame for whatever flow is asked for, with the column
layout of the real ``format=csvfile`` responses (DATAFLOW, the flow's
dimensions in DSD order, TIME_PERIOD, OBS_VALUE, OBS_STATUS, UNIT_MULT,
BASE_PER). Every row is generated from the concept's own pinned key, so the
tests also check that the pins and the dimension tables agree.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from puremacro.fetch import oecd_stes_panel as mod
from puremacro.fetch.oecd_stes_panel import (STES_CONCEPTS, stes_meta,
                                             stes_panel)

_MONTHS = ["2025-11", "2025-12", "2026-01"]


def _row(flow: str, concept: str, code: str, period: str, value: float,
         **override) -> dict:
    """One SDMX-CSV row for ``concept`` in ``flow``, built from its pinned key."""
    spec = STES_CONCEPTS[concept]
    dims = mod._FLOW_DIMS[flow]
    row = {"DATAFLOW": flow}
    for dim, val in zip(dims, spec.key.split(".")):
        row[dim] = code if dim == "REF_AREA" else val
    row.update(TIME_PERIOD=period, OBS_VALUE=value, OBS_STATUS="A",
               UNIT_MULT=spec.unit_mult, BASE_PER=spec.base_period or "")
    row.update(override)
    return row


def _value(concept: str, code: str, t: int) -> float:
    return 100.0 + 10.0 * list(STES_CONCEPTS).index(concept) + len(code) + t


def _fake_rows(flow: str, codes: list[str]) -> list[dict]:
    rows = []
    for concept, spec in STES_CONCEPTS.items():
        if flow not in [f for f, _ in mod._flows_of(spec)]:
            continue
        for code in codes:
            for t, period in enumerate(_MONTHS):
                rows.append(_row(flow, concept, code, period, _value(concept, code, t)))
    return rows


class FakeOECD:
    """Stands in for ``oecd_csv``: records calls, serves synthetic frames."""

    def __init__(self, codes=("USA", "MEX"), fail=(), extra=None):
        self.codes = list(codes)
        self.fail = dict.fromkeys(fail, "HTTP 429") if not isinstance(fail, dict) else fail
        self.extra = extra or {}
        self.calls: list[tuple[str, str, dict]] = []

    def __call__(self, agency_flow, key, **kw):
        self.calls.append((agency_flow, key, kw))
        flow = agency_flow.rsplit(",", 1)[0]
        if flow in self.fail:
            out = pd.DataFrame()
            out.attrs.update(status=self.fail[flow], url="x")
            return out
        rows = _fake_rows(flow, self.codes + ["EA20", "OECD"]) + self.extra.get(flow, [])
        areas = key.split(".")[0]
        if areas:                       # the OECD answers only for the areas in the key
            rows = [r for r in rows if r["REF_AREA"] in areas.split("+")]
        if not rows:
            out = pd.DataFrame()
            out.attrs.update(status="HTTP 404", url="x")
            return out
        out = pd.DataFrame(rows)
        out.attrs.update(status="ok", url="x")
        return out


@pytest.fixture
def fake(monkeypatch):
    f = FakeOECD()
    monkeypatch.setattr(mod, "_get_csv", f)
    return f


def test_the_panel_is_indexed_by_code_and_month_start_with_one_float_column_per_concept(fake):
    df = stes_panel(start="2000")
    assert df.index.names == ["code", "date"]
    assert list(df.columns) == list(STES_CONCEPTS)
    assert all(dt == np.float64 for dt in df.dtypes)
    dates = df.index.get_level_values("date")
    assert (dates.day == 1).all()
    assert sorted(set(dates.strftime("%Y-%m"))) == _MONTHS
    assert df.loc[("USA", pd.Timestamp("2025-12-01")), "ip"] == _value("ip", "USA", 1)


def test_aggregates_are_dropped_when_every_area_is_requested(fake):
    df = stes_panel(start="2000")
    assert set(df.index.get_level_values("code")) == {"USA", "MEX"}


def test_an_aggregate_asked_for_by_name_is_kept(fake):
    df = stes_panel(["EA20", "USA"], concepts="m3", start="2000")
    assert set(df.index.get_level_values("code")) == {"EA20", "USA"}


def test_one_request_per_flow_with_the_countries_in_the_first_key_position(fake):
    stes_panel(["usa", "MEX"], concepts=["ip", "retail_vol", "cpi"], start="1990")
    flows = [c[0] for c in fake.calls]
    assert flows == ["OECD.SDD.STES,DSD_STES@DF_INDSERV,4.3",
                     "OECD.SDD.TPS,DSD_PRICES_COICOP2018@DF_PRICES_C2018_ALL,1.0",
                     "OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL,1.0"]
    assert fake.calls[0][1] == "USA+MEX.M.PRVM+TOVM.IX.BTE+C+F+G47.Y._Z._Z.N"
    assert fake.calls[0][2]["start_period"] == "1990"


def test_the_request_key_does_not_depend_on_which_concepts_are_asked_for(monkeypatch):
    a, b = FakeOECD(), FakeOECD()
    monkeypatch.setattr(mod, "_get_csv", a)
    stes_panel(concepts="ip", start="2000")
    monkeypatch.setattr(mod, "_get_csv", b)
    stes_panel(concepts="retail_vol", start="2000")
    assert a.calls[0][:2] == b.calls[0][:2]


def test_persons_are_rescaled_to_thousands_and_rates_are_left_alone(monkeypatch):
    extra = {mod._OLAB: [_row(mod._OLAB, "vacancies", "DEU", "2026-02", 648242.0,
                              UNIT_MULT=0)]}
    monkeypatch.setattr(mod, "_get_csv", FakeOECD(extra=extra))
    df = stes_panel(concepts=["vacancies", "emp", "urate"], start="2000")
    assert df.loc[("DEU", pd.Timestamp("2026-02-01")), "vacancies"] == pytest.approx(648.242)
    assert df.loc[("USA", pd.Timestamp("2025-11-01")), "emp"] == _value("emp", "USA", 0)
    meta = stes_meta(df).set_index("concept")
    assert meta.loc["vacancies", "unit"] == "thousands of persons"
    assert meta.loc["urate", "unit"] == "percent of labour force"


def test_growth_rate_rows_and_subsectors_do_not_leak_into_the_level_series(monkeypatch):
    extra = {
        mod._CLI: [_row(mod._CLI, "bci", "USA", "2025-11", -0.3, TRANSFORMATION="GY")],
        mod._OLAB: [_row(mod._OLAB, "reg_unemp", "USA", "2025-11", 5.0, SECTOR="S1D")],
        mod._UNE_M: [_row(mod._UNE_M, "urate", "USA", "2025-11", 9.9, AGE="Y15T24")],
    }
    monkeypatch.setattr(mod, "_get_csv", FakeOECD(extra=extra))
    df = stes_panel(concepts=["bci", "reg_unemp", "urate", "urate_nsa"], start="2000")
    at = ("USA", pd.Timestamp("2025-11-01"))
    assert df.loc[at, "bci"] == _value("bci", "USA", 0)
    assert df.loc[at, "reg_unemp"] == _value("reg_unemp", "USA", 0)
    assert df.loc[at, "urate"] == _value("urate", "USA", 0)
    assert df.loc[at, "urate_nsa"] == _value("urate_nsa", "USA", 0)


def test_cpi_takes_coicop_2018_first_and_fills_from_coicop_1999(monkeypatch):
    def price_rows(flow, code, months, value):
        return [_row(flow, "cpi", code, m, value) for m in months]

    c18 = price_rows(mod._C2018, "FRA", ["2025-12", "2026-01"], 120.0)
    c99 = ([_row(mod._C1999, "cpi", "FRA", "2025-11", 118.0)]
           + price_rows(mod._C1999, "FRA", ["2025-12"], 119.0)
           + price_rows(mod._C1999, "DEU", ["2025-12", "2026-01"], 130.0))

    def fetch(agency_flow, key, **kw):
        flow = agency_flow.rsplit(",", 1)[0]
        out = pd.DataFrame(c18 if flow == mod._C2018 else c99)
        out.attrs["status"] = "ok"
        return out

    monkeypatch.setattr(mod, "_get_csv", fetch)
    df = stes_panel(["FRA", "DEU"], concepts="cpi", start="2000")["cpi"]
    # only in 1999, ratio-spliced onto the 2018 level at 2025-12 (120 / 119)
    assert df[("FRA", pd.Timestamp("2025-11-01"))] == pytest.approx(118.0 * 120.0 / 119.0)
    assert df[("FRA", pd.Timestamp("2025-12-01"))] == 120.0     # 2018 wins the overlap
    assert df[("FRA", pd.Timestamp("2026-01-01"))] == 120.0
    assert df[("DEU", pd.Timestamp("2026-01-01"))] == 130.0
    assert not df.index.duplicated().any()
    meta = stes_meta(stes_panel(["FRA", "DEU"], concepts="cpi", start="2000"))
    assert meta.loc[0, "countries_fallback"] == ("DEU", "FRA")
    assert meta.loc[0, "splice_factors"] == ("FRA:2025-12:1.0084",)
    # DEU has no COICOP-2018 series to anchor on: published 1999 values, unscaled
    assert df[("DEU", pd.Timestamp("2025-12-01"))] == 130.0


def test_a_concept_with_duplicate_months_after_pinning_is_dropped_with_a_warning(monkeypatch):
    dup = [_row(mod._MONAGG, "m1", "USA", _MONTHS[0], 1.0)]
    monkeypatch.setattr(mod, "_get_csv", FakeOECD(extra={mod._MONAGG: dup}))
    with pytest.warns(UserWarning, match="duplicate"):
        df = stes_panel(concepts=["m1", "m3"], start="2000")
    assert df["m1"].isna().all()
    assert df["m3"].notna().any()
    assert "m1" in df.attrs["missing"]


def test_a_throttled_provider_gives_an_empty_frame_and_a_warning_not_an_exception(monkeypatch):
    every = {spec.flow for spec in STES_CONCEPTS.values()} | {mod._C1999}
    monkeypatch.setattr(mod, "_get_csv", FakeOECD(fail=every))
    with pytest.warns(UserWarning, match="HTTP 429"):
        df = stes_panel(["USA"], concepts=["ip", "cpi"])
    assert df.empty
    assert df.index.names == ["code", "date"]
    assert list(df.columns) == ["ip", "cpi"]
    assert set(df.attrs) >= {"meta", "source", "fetched_at", "missing", "requests"}
    assert df.attrs["meta"] == ()
    assert set(df.attrs["missing"]) == {"ip", "cpi"}
    assert {r["status"] for r in df.attrs["requests"]} == {"HTTP 429"}


def test_one_failed_flow_leaves_the_others_and_lists_what_is_missing(monkeypatch):
    monkeypatch.setattr(mod, "_get_csv", FakeOECD(fail=[mod._FINMARK]))
    with pytest.warns(UserWarning):
        df = stes_panel(["USA", "MEX"], concepts=["ip", "rate_3m"], start="2000")
    assert df["ip"].notna().all()
    assert df["rate_3m"].isna().all()
    assert df.attrs["missing"] == ("rate_3m",)


def test_a_requested_country_without_data_is_listed_as_missing(fake):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        df = stes_panel(["USA", "NZL"], concepts="ip", start="2000")
    assert "ip:NZL" in df.attrs["missing"]
    assert set(df.index.get_level_values("code")) == {"USA"}


def test_meta_records_flow_key_units_and_coverage_per_concept(fake):
    df = stes_panel(concepts=["ip", "fx_usd"], start="2000")
    meta = stes_meta(df).set_index("concept")
    assert meta.loc["ip", "flow"] == "OECD.SDD.STES,DSD_STES@DF_INDSERV,4.3"
    assert meta.loc["ip", "key"] == ".M.PRVM.IX.BTE.Y._Z._Z.N"
    assert meta.loc["ip", "base_period"] == "2015"
    assert meta.loc["ip", "sa"] is True or meta.loc["ip", "sa"] == True  # noqa: E712
    assert meta.loc["fx_usd", "n_countries"] == 2
    assert meta.loc["fx_usd", "first"] == "2025-11-01"
    assert meta.loc["fx_usd", "last"] == "2026-01-01"
    assert meta.loc["fx_usd", "n"] == 6
    assert all(isinstance(m, dict) for m in df.attrs["meta"])
    assert "DF_KEI,4.0" in df.attrs["source"]
    pd.Timestamp(df.attrs["fetched_at"])


def test_start_trims_the_months_before_it(fake):
    df = stes_panel(concepts="ip", start="2025-12")
    assert df.index.get_level_values("date").min() == pd.Timestamp("2025-12-01")


@pytest.mark.parametrize("kwargs, match", [
    ({"concepts": ["ip", "gdp"]}, "unknown concept"),
    ({"concepts": []}, "concepts is empty"),
    ({"start": "1950Q1"}, "start must look like"),
    ({"codes": ["USA", 5]}, "country codes"),
])
def test_caller_errors_raise_value_error_listing_the_choices(fake, kwargs, match):
    with pytest.raises(ValueError, match=match):
        stes_panel(**kwargs)
    assert fake.calls == []


def test_every_registry_key_has_one_position_per_dimension_of_its_flows():
    for name, spec in STES_CONCEPTS.items():
        for flow, _ in mod._flows_of(spec):
            assert len(spec.key.split(".")) == len(mod._FLOW_DIMS[flow]), name
            assert spec.key.split(".")[0] == "", name


@pytest.mark.network
def test_live_three_countries_every_concept():
    df = stes_panel(["USA", "MEX", "DEU"], start="1950", pause=8.0)
    assert {"USA", "MEX", "DEU"} <= set(df.index.get_level_values("code"))
    assert df["cpi"].notna().sum() > 1000


# --- review fixes -----------------------------------------------------------

def _raise(exc):
    def fetch(agency_flow, key, **kw):
        raise exc
    return fetch


@pytest.mark.parametrize("exc, status", [
    (__import__("urllib.error").error.HTTPError("u", 429, "Too Many", {}, None), "HTTP 429"),
    (OSError("timed out"), "OSError"),
    (TimeoutError("read timed out"), "TimeoutError"),
])
def test_a_transport_that_raises_gives_an_empty_frame_and_a_warning(monkeypatch, exc, status):
    monkeypatch.setattr(mod, "_get_csv", _raise(exc))
    with pytest.warns(UserWarning, match=status):
        df = stes_panel(["USA", "MEX"], concepts=["ip", "cpi"])
    assert df.empty
    assert df.index.names == ["code", "date"]
    assert set(df.attrs["missing"]) == {"ip", "cpi"}
    assert {r["status"] for r in df.attrs["requests"]} == {status}


def test_a_failed_primary_price_flow_drops_cpi_instead_of_returning_the_1999_flow_alone(monkeypatch):
    fake = FakeOECD(fail={mod._C2018: "HTTP 429"})
    monkeypatch.setattr(mod, "_get_csv", fake)
    with pytest.warns(UserWarning, match="HTTP 429"):
        df = stes_panel(["USA", "MEX"], concepts=["ip", "cpi"], start="2000")
    assert "cpi" in df.attrs["missing"]
    assert df["cpi"].isna().all()
    assert df["ip"].notna().all()
    assert [m["concept"] for m in df.attrs["meta"]] == ["ip"]


def test_a_failed_fallback_flow_keeps_the_column_but_says_so(monkeypatch):
    fake = FakeOECD(fail={mod._C1999: "HTTP 429"})
    monkeypatch.setattr(mod, "_get_csv", fake)
    with pytest.warns(UserWarning, match="HTTP 429"):
        df = stes_panel(["USA", "MEX"], concepts="cpi", start="2000")
    assert df["cpi"].notna().all()
    assert "cpi@DF_PRICES_ALL" in df.attrs["missing"]
    meta = stes_meta(df).set_index("concept")
    assert meta.loc["cpi", "flows_failed"] == ("OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL,1.0",)


def test_a_404_on_a_whole_flow_is_a_failure_but_on_named_countries_is_not(monkeypatch):
    fake = FakeOECD(fail={mod._C2018: "HTTP 404"})
    monkeypatch.setattr(mod, "_get_csv", fake)
    with pytest.warns(UserWarning, match="HTTP 404"):
        df = stes_panel(concepts="cpi", start="2000")       # every area: unknown flow?
    assert "cpi" in df.attrs["missing"]
    monkeypatch.setattr(mod, "_get_csv", FakeOECD(fail={mod._C2018: "HTTP 404"}))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        df = stes_panel(["USA", "MEX"], concepts="cpi", start="2000")
    assert df["cpi"].notna().all()                         # served by COICOP 1999
    assert stes_meta(df).loc[0, "countries_fallback"] == ("MEX", "USA")


def test_a_blank_unit_mult_takes_the_countrys_usual_multiplier_not_zero(monkeypatch):
    extra = {mod._LFS: [_row(mod._LFS, "emp", "CAN", "2025-11", 20500.0, UNIT_MULT=3),
                        _row(mod._LFS, "emp", "CAN", "2025-12", 20510.0, UNIT_MULT=np.nan)]}
    monkeypatch.setattr(mod, "_get_csv", FakeOECD(extra=extra))
    with pytest.warns(UserWarning, match="blank UNIT_MULT"):
        df = stes_panel(concepts="emp", start="2000")
    assert df.loc[("CAN", pd.Timestamp("2025-12-01")), "emp"] == 20510.0


def test_hicp_takes_the_1999_flow_only_for_european_economies(monkeypatch):
    c99 = [_row(mod._C1999, "hicp", code, m, 100.0)
           for code in ("USA", "GBR") for m in _MONTHS]

    def fetch(agency_flow, key, **kw):
        flow = agency_flow.rsplit(",", 1)[0]
        out = pd.DataFrame(c99 if flow == mod._C1999 else [])
        out.attrs["status"] = "ok" if flow == mod._C1999 else "HTTP 404"
        return out

    monkeypatch.setattr(mod, "_get_csv", fetch)
    df = stes_panel(["USA", "GBR"], concepts="hicp", start="2000")
    assert set(df.index.get_level_values("code")) == {"GBR"}
    assert "hicp:USA" in df.attrs["missing"]


def test_meta_lists_countries_whose_series_stopped_early(monkeypatch):
    extra = {mod._INDSERV: [_row(mod._INDSERV, "ip", "RUS", m, 90.0)
                            for m in ("2022-01", "2022-02", "2022-03")]}
    monkeypatch.setattr(mod, "_get_csv", FakeOECD(extra=extra))
    df = stes_panel(concepts="ip", start="2000")
    assert stes_meta(df).loc[0, "stale_countries"] == ("RUS",)


def test_retries_and_retry_sleep_reach_the_transport(fake):
    stes_panel(["USA"], concepts="ip", start="2000", retries=1, retry_sleep=5.0)
    assert fake.calls[0][2]["retries"] == 1
    assert fake.calls[0][2]["retry_sleep"] == 5.0
