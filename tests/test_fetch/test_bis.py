"""Offline tests for :mod:`puremacro.fetch.bis` (BIS statistics API panels).

Every HTTP call goes through ``bis._get``, which the fixture replaces with a
dispatcher returning tiny synthetic SDMX-CSV bodies shaped like the real ones
(column order and attribute columns copied from live responses, 7 Oct 2026).
"""
from __future__ import annotations

import json
import urllib.error

import pandas as pd
import pytest

from puremacro.fetch import bis

EER = """FREQ,EER_TYPE,EER_BASKET,REF_AREA,TIME_PERIOD,OBS_VALUE
M,R,B,MX,2025-10,100.0
M,R,B,MX,2025-11,102.0
M,R,B,MX,2025-12,104.0
M,R,B,MX,2026-01,110.0
M,R,B,MX,2026-02,112.0
M,R,B,XM,2025-12,95.5
M,R,B,XM,2026-01,96.5
M,R,B,4T,2026-01,90.0
"""

CBPOL = """FREQ,REF_AREA,TIME_PERIOD,OBS_VALUE
M,MX,2025-10,7.5
M,MX,2025-11,7.25
M,MX,2025-12,7.0
M,MX,2026-01,7.0
M,DE,1998-11,3.3
M,DE,1998-12,3.0
M,XM,1999-01,3.0
M,US,2025-12,NaN
"""

XRU = """FREQ,REF_AREA,CURRENCY,COLLECTION,TIME_PERIOD,OBS_VALUE
M,NA,NAD,A,2026-07,16.4588
M,NA,NAD,A,2026-08,16.1794
M,XW,XDR,A,2026-08,0.73
M,WA,XOF,A,2026-08,560.0
"""

TC = """FREQ,BORROWERS_CTY,TC_BORROWERS,TC_LENDERS,VALUATION,UNIT_TYPE,TC_ADJUST,COLLECTION,DECIMALS,UNIT_MULT,UNIT_MEASURE,TITLE_TS,TIME_PERIOD,OBS_VALUE,OBS_STATUS,OBS_PRE_BREAK,OBS_CONF
Q,MX,P,A,M,XDC,A,E,3,9,MXN,Mexico - credit,2025-Q4,13673.305,A,,F
Q,MX,P,A,M,XDC,A,E,3,9,MXN,Mexico - credit,2026-Q1,13738.687,A,,F
Q,5A,P,A,M,XDC,A,E,3,9,USD,All reporting,2026-Q1,1.0,A,,F
"""

TC_GDP = """FREQ,BORROWERS_CTY,TC_BORROWERS,TC_LENDERS,VALUATION,UNIT_TYPE,TC_ADJUST,UNIT_MULT,UNIT_MEASURE,TIME_PERIOD,OBS_VALUE
Q,US,P,A,M,770,A,0,770,2025-Q1,140.0
Q,US,P,A,M,770,A,0,770,2025-Q2,141.0
Q,US,P,A,M,770,A,0,770,2025-Q3,142.0
Q,US,P,A,M,770,A,0,770,2025-Q4,143.0
Q,US,P,A,M,770,A,0,770,2026-Q1,139.7
"""

AVAIL = json.dumps({"data": {"dataConstraints": [{"cubeRegions": [{"include": True, "keyValues": [
    {"id": "FREQ", "values": [{"value": "M"}]},
    {"id": "REF_AREA", "values": [{"value": v} for v in ("MX", "US", "XM", "4T", "TW", "NA")]},
]}]}]}})


def _fake(bodies):
    calls = []

    def get(url, **kw):
        calls.append((url, kw))
        for needle, body in bodies.items():
            if needle in url:
                if isinstance(body, BaseException):
                    raise body
                return body.encode()
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
    get.calls = calls
    return get


@pytest.fixture
def patch_get(monkeypatch):
    def install(bodies):
        fake = _fake(bodies)
        monkeypatch.setattr(bis, "_get", fake)
        return fake
    return install


def test_monthly_panel_has_code_date_index_float_columns_and_month_start_dates(patch_get):
    patch_get({"WS_EER/+/M.R.B.": EER, "WS_CBPOL/+/M.": CBPOL})
    df = bis.bis_panel(variables=["reer", "policy_rate"])
    assert df.index.names == ["code", "date"]
    assert list(df.columns) == ["reer", "policy_rate"]
    assert all(df[c].dtype == "float64" for c in df.columns)
    dates = df.index.get_level_values("date")
    assert pd.api.types.is_datetime64_any_dtype(dates)
    assert (dates.day == 1).all()
    assert df.loc[("MEX", pd.Timestamp("2026-01-01")), "reer"] == 110.0
    assert df.loc[("MEX", pd.Timestamp("2025-11-01")), "policy_rate"] == 7.25


def test_euro_area_and_other_aggregates_are_dropped_by_default(patch_get):
    patch_get({"WS_EER/+/M.R.B.": EER})
    df = bis.bis_panel(variables=["reer"])
    assert set(df.index.get_level_values("code")) == {"MEX"}
    with_agg = bis.bis_panel(variables=["reer"], aggregates=True)
    assert {"EME", "EA", "MEX"} <= set(with_agg.index.get_level_values("code"))


def test_euro_area_named_in_codes_is_kept_as_ea(patch_get):
    fake = patch_get({"WS_EER/+/M.R.B.": EER})
    df = bis.bis_panel(["MEX", "EA"], variables=["reer"])
    assert set(df.index.get_level_values("code")) == {"MEX", "EA"}
    assert "/M.R.B.MX+XM?" in fake.calls[0][0]


def test_codes_is_the_first_positional_argument(patch_get):
    fake = patch_get({"WS_EER/+/M.R.B.": EER})
    df = bis.bis_panel(["MEX"], variables=["reer"])
    assert "/M.R.B.MX?" in fake.calls[0][0]
    assert set(df.index.get_level_values("code")) == {"MEX"}
    with pytest.raises(ValueError, match="variables="):
        bis.bis_panel(["reer", "policy_rate"])


def test_namibia_is_not_read_as_missing_and_xru_aggregates_drop(patch_get):
    patch_get({"WS_XRU/+/M...A": XRU})
    df = bis.bis_panel(variables=["xr_usd"])
    assert set(df.index.get_level_values("code")) == {"NAM"}
    assert df.loc[("NAM", pd.Timestamp("2026-08-01")), "xr_usd"] == pytest.approx(16.1794)


def test_bis_nan_strings_become_missing_and_all_nan_rows_drop(patch_get):
    patch_get({"WS_CBPOL/+/M.": CBPOL})
    df = bis.bis_panel(variables=["policy_rate"])
    assert "USA" not in set(df.index.get_level_values("code"))


def test_meta_records_flow_key_units_and_coverage_per_country(patch_get):
    patch_get({"WS_EER/+/M.R.B.": EER})
    df = bis.bis_panel(variables=["reer"])
    meta = {m["code"]: m for m in df.attrs["meta"]}
    assert isinstance(df.attrs["meta"], tuple)
    m = meta["MEX"]
    assert m["flow"] == "WS_EER" and m["key"] == "M.R.B.MX"
    assert m["units"] == "index, 2020=100" and m["unit_mult"] == 0 and m["sa"] is False
    assert (m["first"], m["last"], m["n"]) == ("2025-10-01", "2026-02-01", 5)
    assert df.attrs["source"].startswith("BIS") and "WS_EER" in df.attrs["source"]
    assert df.attrs["fetched_at"].endswith("+00:00")
    assert df.attrs["missing"] == ()
    assert list(bis.bis_meta(df).columns)[:3] == ["code", "variable", "flow"]


def test_credit_levels_are_rescaled_from_billions_to_millions(patch_get):
    patch_get({"WS_TC/+/Q..P.A.M.XDC.A": TC})
    df = bis.bis_panel(variables=["credit"], freq="Q")
    assert df.loc[("MEX", pd.Timestamp("2026-01-01")), "credit"] == pytest.approx(13738687.0)
    assert set(df.index.get_level_values("code")) == {"MEX"}
    m = df.attrs["meta"][0]
    assert m["units"] == "millions of national currency" and m["unit_mult"] == 6
    assert m["currency"] == "MXN"


def test_quarterly_dates_are_quarter_starts(patch_get):
    patch_get({"WS_TC/+/Q..P.A.M.770.A": TC_GDP})
    df = bis.bis_panel(variables=["credit_gdp"], freq="Q")
    d = df.index.get_level_values("date")
    assert list(d.month) == [1, 4, 7, 10, 1]


def test_quarterly_mean_needs_three_months_and_last_needs_the_final_month(patch_get):
    patch_get({"WS_EER/+/M.R.B.": EER, "WS_CBPOL/+/M.": CBPOL})
    df = bis.bis_panel(variables=["reer", "policy_rate"], freq="Q")
    mex = df.loc["MEX"]
    # 2025Q4 complete: mean of Oct-Dec; 2026Q1 has two months only -> absent
    assert mex.loc[pd.Timestamp("2025-10-01"), "reer"] == pytest.approx(102.0)
    assert mex.loc[pd.Timestamp("2025-10-01"), "policy_rate"] == 7.0
    assert pd.Timestamp("2026-01-01") not in mex.index
    # euro area only has 1999-01: no complete quarter, no Q4 end month
    assert "EA" not in set(df.index.get_level_values("code"))
    assert df.loc[("DEU", pd.Timestamp("1998-10-01")), "policy_rate"] == 3.0


def test_annual_from_quarterly_uses_q4_for_stocks(patch_get):
    patch_get({"WS_TC/+/Q..P.A.M.770.A": TC_GDP})
    df = bis.bis_panel(variables=["credit_gdp"], freq="A")
    assert list(df.index) == [("USA", pd.Timestamp("2025-01-01"))]
    assert df.iloc[0, 0] == 143.0


def test_codes_are_sent_as_bis_iso2_joined_with_plus(patch_get):
    fake = patch_get({"WS_EER/+/M.R.B.": EER})
    df = bis.bis_panel(variables=["reer"], codes=["MEX", "EA", "USA"])
    url = fake.calls[0][0]
    assert "/WS_EER/+/M.R.B.MX+XM+US?" in url
    assert fake.calls[0][1]["headers"] == {"Accept-Encoding": "gzip"}
    assert {"variable": "reer", "code": "USA", "reason": "not published"} in df.attrs["missing"]


def test_start_is_sent_in_the_native_period_format_and_filters_rows(patch_get):
    fake = patch_get({"WS_TC/+/Q..P.A.M.770.A": TC_GDP, "WS_EER/+/M.R.B.": EER})
    q = bis.bis_panel(variables=["credit_gdp"], freq="Q", start="2025-04")
    assert "startPeriod=2025-Q2" in fake.calls[0][0]
    assert q.index.get_level_values("date").min() == pd.Timestamp("2025-04-01")
    bis.bis_panel(variables=["reer"], start="1990")
    assert "startPeriod=1990-01" in fake.calls[1][0]


def test_provider_failure_returns_empty_frame_with_attrs_and_warns(patch_get):
    patch_get({"WS_EER": urllib.error.HTTPError("u", 503, "down", {}, None)})
    with pytest.warns(RuntimeWarning, match="no data"):
        df = bis.bis_panel(variables=["reer", "neer"])
    assert df.empty and df.index.names == ["code", "date"]
    assert list(df.columns) == ["reer", "neer"]
    assert set(df.attrs) >= {"meta", "source", "fetched_at", "missing"}
    assert {m["variable"] for m in df.attrs["missing"]} == {"reer", "neer"}
    assert df.attrs["missing"][0]["reason"] == "HTTP 503"


def test_timeout_and_garbage_body_do_not_raise(patch_get):
    patch_get({"WS_EER/+/M.R.B.": TimeoutError("timed out"), "WS_CBPOL": "<html>oops</html>"})
    with pytest.warns(RuntimeWarning):
        df = bis.bis_panel(variables=["reer", "policy_rate"])
    reasons = {m["variable"]: m["reason"] for m in df.attrs["missing"]}
    assert reasons["reer"] == "timeout"
    assert reasons["policy_rate"].startswith("unexpected response")


def test_one_failing_variable_keeps_the_others(patch_get):
    patch_get({"WS_EER/+/M.R.B.": EER})  # CBPOL -> 404
    with pytest.warns(RuntimeWarning):
        df = bis.bis_panel(variables=["reer", "policy_rate"])
    assert df["reer"].notna().any() and df["policy_rate"].isna().all()
    assert df.attrs["missing"] == ({"variable": "policy_rate", "code": None, "reason": "HTTP 404"},)


@pytest.mark.parametrize("kwargs, match", [
    ({"variables": ["gdp"]}, "unknown BIS variable"),
    ({"freq": "W"}, "freq must be one of"),
    ({"variables": ["credit_gdp"], "freq": "M"}, "not available at freq"),
    ({"variables": ["cpi_a"], "freq": "Q"}, "not available at freq"),
    ({"variables": ["reer"], "codes": ["XYZ"]}, "unknown code"),
])
def test_caller_errors_raise_value_error_listing_choices(patch_get, kwargs, match):
    patch_get({})
    with pytest.raises(ValueError, match=match):
        bis.bis_panel(**kwargs)


def test_bis_eer_maps_kind_and_basket_and_rejects_bad_values(patch_get):
    fake = patch_get({"WS_EER/+/M.N.N.": EER.replace("M,R,B", "M,N,N")})
    df = bis.bis_eer("nominal", "narrow")
    assert list(df.columns) == ["neer_narrow"]
    assert "/M.N.N.?" in fake.calls[0][0]
    with pytest.raises(ValueError, match="kind"):
        bis.bis_eer("effective")
    with pytest.raises(ValueError, match="basket"):
        bis.bis_eer("real", "wide")


def test_bis_get_returns_strings_with_float_obs_value_and_never_raises(patch_get):
    patch_get({"WS_XRU/+/M.NA..A": XRU})
    raw = bis.bis_get("WS_XRU", "M.NA..A", last_n=2, dataonly=True)
    assert raw["REF_AREA"].iloc[0] == "NA"
    assert raw["OBS_VALUE"].dtype == "float64"
    assert "lastNObservations=2" in raw.attrs["url"] and "detail=dataonly" in raw.attrs["url"]
    assert raw.attrs["status"] == "ok"
    with pytest.warns(RuntimeWarning, match="HTTP 404"):
        empty = bis.bis_get("WS_NOPE", "M")
    assert empty.empty and empty.attrs["status"] == "HTTP 404"


def test_bis_countries_reads_availability_maps_iso3_and_drops_aggregates(patch_get):
    fake = patch_get({"/availability/dataflow/BIS/WS_EER/+/*": AVAIL})
    assert bis.bis_countries("WS_EER") == ["MEX", "NAM", "TWN", "USA"]
    assert "mode=exact" in fake.calls[0][0]
    assert {"EME", "EA"} <= set(bis.bis_countries("WS_EER", aggregates=True))


def test_bis_countries_failure_warns_and_returns_empty_list(patch_get):
    patch_get({})
    with pytest.warns(RuntimeWarning):
        assert bis.bis_countries("WS_EER") == []


def test_annual_cpi_before_1678_survives_as_microsecond_dates(patch_get):
    body = "FREQ,REF_AREA,UNIT_MEASURE,TIME_PERIOD,OBS_VALUE\nA,GB,628,1661,1.32\nA,GB,628,2025,154.7\n"
    patch_get({"WS_LONG_CPI/+/A.": body})
    df = bis.bis_panel(variables=["cpi_a"], freq="A")
    assert df.index.get_level_values("date").min() == pd.Timestamp("1661-01-01")


def test_registry_keys_use_version_plus_and_never_the_all_keyword():
    for name, spec in bis.BIS_SERIES.items():
        assert "{a}" in spec["key"], name
        assert "ALL" not in spec["key"], name
        assert spec["freq"] in ("M", "Q", "A") and spec["agg"] in ("mean", "last")
    assert "XM" not in bis.BIS_TO_ISO3 and bis.BIS_AGGREGATES["XM"] == "EA"
    assert bis.BIS_TO_ISO3["NA"] == "NAM"
    assert all(len(c) == 3 for c in bis.BIS_TO_ISO3.values())
    assert len(set(bis.BIS_TO_ISO3.values())) == len(bis.BIS_TO_ISO3)


@pytest.mark.network
def test_live_three_country_reer_policy_rate():
    df = bis.bis_panel(["MEX", "USA", "EA"], variables=["reer", "policy_rate"])
    assert {"MEX", "USA", "EA"} <= set(df.index.get_level_values("code"))
    assert df.loc["USA", "policy_rate"].first_valid_index() <= pd.Timestamp("1955-01-01")
