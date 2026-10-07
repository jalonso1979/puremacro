"""Offline tests for ``puremacro.fetch.imf`` (IMF SDMX 2.1 API).

Every test patches the module's HTTP seam ``_get`` with tiny synthetic
SDMX-CSV bodies; the ``network`` tests at the bottom hit api.imf.org and are
deselected by default.
"""
from __future__ import annotations

import urllib.error
import warnings

import numpy as np
import pandas as pd
import pytest

import puremacro.fetch.imf as imf

ATTRS = {"meta", "source", "fetched_at", "missing"}

ANEA = """DATAFLOW,COUNTRY,INDICATOR,PRICE_TYPE,TYPE_OF_TRANSFORMATION,FREQUENCY,TIME_PERIOD,OBS_VALUE,SCALE,UNIT
IMF.STA:ANEA(6.0.1),USA,B1GQ,V,XDC,A,2023,27720709000000,,
IMF.STA:ANEA(6.0.1),USA,B1GQ,V,XDC,A,2024,29298012800000,,
IMF.STA:ANEA(6.0.1),USA,B1GQ,Q,XDC,A,2023,22671096000000,,
IMF.STA:ANEA(6.0.1),USA,B1GQ,Q,XDC,A,2024,23358434800000,,
IMF.STA:ANEA(6.0.1),USA,P3,V,XDC,A,2024,23000000000000,,
IMF.STA:ANEA(6.0.1),USA,P3_S13,V,XDC,A,2024,4000000000000,,
IMF.STA:ANEA(6.0.1),USA,P52,V,XDC,A,2024,50000000000,,
IMF.STA:ANEA(6.0.1),USA,P52,Q,XDC,A,2024,40000000000,,
IMF.STA:ANEA(6.0.1),G163,B1GQ,V,XDC,A,2024,15000000000000,,
IMF.STA:ANEA(6.0.1),KOS,B1GQ,V,XDC,A,2024,10000000000,,
IMF.STA:ANEA(6.0.1),MEX,B1GQ,V,XDC,A,,,,
"""

QNEA = """DATAFLOW,COUNTRY,INDICATOR,PRICE_TYPE,S_ADJUSTMENT,TYPE_OF_TRANSFORMATION,FREQUENCY,TIME_PERIOD,OBS_VALUE
IMF.STA:QNEA(7.0.0),USA,B1GQ,V,SA,XDC,Q,2024-Q1,7177040300000
IMF.STA:QNEA(7.0.0),USA,B1GQ,V,SA,XDC,Q,2024-Q2,7300000000000
IMF.STA:QNEA(7.0.0),USA,B1GQ,Q,SA,XDC,Q,2024-Q2,5800000000000
"""

CPI = """DATAFLOW,COUNTRY,INDEX_TYPE,COICOP_1999,TYPE_OF_TRANSFORMATION,FREQUENCY,TIME_PERIOD,OBS_VALUE
IMF.STA:CPI(5.0.0),MEX,CPI,_T,IX,M,2025-M01,137.9
IMF.STA:CPI(5.0.0),MEX,CPI,_T,IX,M,2025-M02,138.3
IMF.STA:CPI(5.0.0),MEX,CPI,CP01,IX,M,2025-M02,150.1
IMF.STA:CPI(5.0.0),G163,CPI,_T,IX,M,2025-M02,120.0
"""

LS_M = """DATAFLOW,COUNTRY,INDICATOR,TYPE_OF_TRANSFORMATION,FREQUENCY,TIME_PERIOD,OBS_VALUE
IMF.STA:LS(9.0.0),MEX,U,PT,M,2025-M02,2.5
IMF.STA:LS(9.0.0),MEX,E,PE,M,2025-M02,59500000
"""

ER = """DATAFLOW,COUNTRY,INDICATOR,TYPE_OF_TRANSFORMATION,FREQUENCY,TIME_PERIOD,OBS_VALUE
IMF.STA:ER(4.0.1),MEX,XDC_USD,PA_RT,M,2025-M02,20.5
IMF.STA:ER(4.0.1),MEX,XDC_USD,EOP_RT,M,2025-M02,20.4
IMF.STA:ER(4.0.1),MEX,XDC_USD,PA_RT,M,2099-M03,99.0
IMF.STA:ER(4.0.1),G129,XDC_USD,PA_RT,M,2025-M02,1.1
"""

MFS = """DATAFLOW,COUNTRY,INDICATOR,FREQUENCY,TIME_PERIOD,OBS_VALUE
IMF.STA:MFS_IR(9.0.0),MEX,MFS166_RT_PT_A_PT,M,2025-M02,9.5
"""

LS_Q = """DATAFLOW,COUNTRY,INDICATOR,TYPE_OF_TRANSFORMATION,FREQUENCY,TIME_PERIOD,OBS_VALUE
IMF.STA:LS(9.0.0),USA,U,PT,Q,2024-Q3,4.2
IMF.STA:LS(9.0.0),USA,UP,PE,Q,2024-Q3,7000000
IMF.STA:LS(9.0.0),USA,LF,PE,Q,2024-Q3,168000000
"""

PCPS = """DATAFLOW,COUNTRY,INDICATOR,DATA_TRANSFORMATION,FREQUENCY,TIME_PERIOD,OBS_VALUE
IMF.RES:PCPS(9.0.0),G001,PALLFNF,INDEX,M,2024-M01,159.3
IMF.RES:PCPS(9.0.0),G001,POILAPSP,USD,M,2024-M01,78.9
IMF.RES:PCPS(9.0.0),G001,PALUM,INDEX,M,2024-M01,137.2
IMF.RES:PCPS(9.0.0),G001,PCOBA,USD,M,2024-M01,28691.4
"""

WEO = """DATAFLOW,COUNTRY,INDICATOR,FREQUENCY,TIME_PERIOD,OBS_VALUE,SCALE
IMF.RES:WEO(9.0.0),USA,NGDP,A,2024,29298000000000,
IMF.RES:WEO(9.0.0),USA,NGDP,A,2026,31000000000000,
IMF.RES:WEO(9.0.0),USA,LUR,A,2024,4.0,
IMF.RES:WEO(9.0.0),USA,LUR,A,2026,4.3,
IMF.RES:WEO(9.0.0),G001,LUR,A,2024,5.0,
"""

WEO_NODATA = """DATAFLOW,COUNTRY,INDICATOR,FREQUENCY,TIME_PERIOD,OBS_VALUE,SCALE,LATEST_ACTUAL_ANNUAL_DATA
IMF.RES:WEO(9.0.0),USA,NGDP,A,,,9,
IMF.RES:WEO(9.0.0),USA,NGDP,A,,,,2025
IMF.RES:WEO(9.0.0),USA,LUR,A,,,,2025
"""

ICSD = """DATAFLOW,COUNTRY,INDICATOR,FREQUENCY,TIME_PERIOD,OBS_VALUE
IMF.FAD:ICSD(1.0.0),USA,P51G_S13_V_XDC,A,2019,741.441956
IMF.FAD:ICSD(1.0.0),USA,CAPSTCK_S13_V_XDC,A,2019,12715.46191
IMF.FAD:ICSD(1.0.0),USA,CAPSTCK_S13_Q_POGDP_PT,A,2019,59.489784
"""

DATAFLOWS_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<mes:Structure xmlns:mes="http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message"
 xmlns:str="http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure"
 xmlns:com="http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common">
 <mes:Structures><str:Dataflows>
  <str:Dataflow id="CPI" agencyID="IMF.STA" version="5.0.0"><com:Name xml:lang="en">Consumer Price Index (CPI)</com:Name></str:Dataflow>
  <str:Dataflow id="CPI_2026_JAN_VINTAGE" agencyID="IMF.STA" version="1.0.0"><com:Name xml:lang="en">CPI vintage</com:Name></str:Dataflow>
 </str:Dataflows></mes:Structures></mes:Structure>"""


def _http_error(url: str, code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, "err", {}, None)


@pytest.fixture
def fake_api(monkeypatch):
    """Route ``imf._get`` to fixture bodies by flow; record every call."""
    monkeypatch.setattr(imf, "_RETRY_SLEEP", 0.0)
    monkeypatch.setattr(imf, "_SLEEP_429", 0.0)
    routes: dict[str, object] = {}
    calls: list[dict] = []

    def _get(url, *, timeout=None, pause=None, refresh=False, accept=None):
        calls.append(dict(url=url, accept=accept, refresh=refresh))
        for needle, body in routes.items():
            if needle in url:
                if isinstance(body, list):          # a sequence of answers
                    body = body.pop(0) if len(body) > 1 else body[0]
                if isinstance(body, BaseException):
                    raise body
                if callable(body):
                    return body(url)
                return body.encode() if isinstance(body, str) else body
        raise _http_error(url, 404)

    monkeypatch.setattr(imf, "_get", _get)
    return routes, calls


# --- raw reader -------------------------------------------------------------

def test_imf_get_sends_the_csv_accept_header_and_dataonly(fake_api):
    routes, calls = fake_api
    routes[",CPI/"] = CPI
    raw = imf.imf_get("CPI", "MEX.CPI._T.IX.M", start=2025)
    assert calls[0]["accept"] == "application/vnd.sdmx.data+csv;version=1.0.0"
    assert calls[0]["url"] == ("https://api.imf.org/external/sdmx/2.1/data/IMF.STA,CPI/"
                               "MEX.CPI._T.IX.M?detail=dataonly&startPeriod=2025")
    assert raw.attrs["status"] == "ok"
    assert raw["OBS_VALUE"].dtype == float and raw["TIME_PERIOD"].iloc[0] == "2025-M01"


def test_imf_get_sends_an_all_dots_key_as_all_to_dodge_the_gateway_403(fake_api):
    routes, calls = fake_api
    routes[",MFS_IR/"] = MFS
    imf.imf_get("IMF.STA,MFS_IR", "..")
    assert "/IMF.STA,MFS_IR/all?" in calls[0]["url"] and "/../" not in calls[0]["url"]


def test_imf_get_retries_a_flaky_500_and_then_succeeds(fake_api):
    routes, calls = fake_api
    routes[",CPI/"] = [_http_error("u", 500), _http_error("u", 500), CPI]
    raw = imf.imf_get("CPI", "MEX.CPI._T.IX.M")
    assert len(calls) == 3 and raw.attrs["status"] == "ok" and len(raw) == 4


def test_imf_get_does_not_retry_a_404_and_returns_an_empty_frame(fake_api):
    _, calls = fake_api
    raw = imf.imf_get("CPI", "XXX.CPI._T.IX.M")
    assert len(calls) == 1 and raw.empty and raw.attrs["status"] == "HTTP 404"


def test_imf_get_turns_a_persistent_timeout_into_a_status_and_a_warning(fake_api):
    routes, calls = fake_api
    routes[",CPI/"] = [TimeoutError("timed out")]
    with pytest.warns(UserWarning, match="timeout"):
        raw = imf.imf_get("CPI", "MEX.CPI._T.IX.M", retries=2)
    assert len(calls) == 3 and raw.empty and raw.attrs["status"].startswith("timeout")


def test_imf_get_rejects_unqualified_flows_and_bad_periods():
    with pytest.raises(ValueError, match="agency-qualified"):
        imf.imf_get("NOTAFLOW")
    with pytest.raises(ValueError, match="start"):
        imf.imf_get("CPI", start="90")
    with pytest.raises(ValueError, match="detail"):
        imf.imf_get("CPI", detail="everything")


def test_imf_dataflows_parses_the_structure_message(fake_api):
    routes, calls = fake_api
    routes["/dataflow"] = DATAFLOWS_XML
    flows = imf.imf_dataflows()
    assert calls[0]["accept"] is None
    assert list(flows["flow"]) == ["IMF.STA,CPI", "IMF.STA,CPI_2026_JAN_VINTAGE"]
    assert list(flows["vintage"]) == [False, True]
    assert flows.loc[0, "name"] == "Consumer Price Index (CPI)"


# --- national accounts --------------------------------------------------------

def test_annual_nea_panel_is_in_millions_at_year_start_with_implicit_deflators(fake_api):
    routes, calls = fake_api
    routes[",ANEA/"] = ANEA
    p = imf.imf_nea_panel()
    assert p.index.names == ["code", "date"]
    assert set(p.dtypes) == {np.dtype("float64")}
    assert set(p.index.get_level_values("date")) == {pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01")}
    usa = p.loc["USA"]
    assert usa.loc["2024-01-01", "gdp"] == pytest.approx(29_298_012.8)
    assert usa.loc["2024-01-01", "cons_priv"] == pytest.approx(19_000_000.0)
    assert usa.loc["2024-01-01", "gdp_defl"] == pytest.approx(100 * 29298012.8 / 23358434.8)
    assert "inventories_defl" not in p and "inventories_real" in p
    meta = {m["variable"]: m for m in p.attrs["meta"]}
    assert meta["gdp"]["unit_mult"] == 6 and meta["gdp"]["scale"] == 1e-6
    assert meta["gdp"]["units"] == "millions of national currency"
    assert meta["gdp_defl"]["unit_mult"] == 0
    assert meta["gdp"]["first"] == "2023-01-01" and meta["gdp"]["n"] == 2   # USA + XKX
    assert ATTRS <= set(p.attrs)
    url = calls[0]["url"]
    assert "/IMF.STA,ANEA/.B1GQ+P3+P3_S13+P3_S14+P51G+P5+P52+P6+P7.V+Q.XDC.A?" in url


def test_nea_panel_drops_imf_aggregates_and_renames_kosovo(fake_api):
    routes, _ = fake_api
    routes[",ANEA/"] = ANEA
    p = imf.imf_nea_panel()
    codes = set(p.index.get_level_values("code"))
    assert "G163" not in codes and "KOS" not in codes and "XKX" in codes
    assert p.attrs["aggregates_dropped"] == ("G163",)


def test_quarterly_nea_panel_dates_are_quarter_starts_and_the_key_carries_sa(fake_api):
    routes, calls = fake_api
    routes[",QNEA/"] = QNEA
    p = imf.imf_nea_panel(["usa", "PSE"], freq="Q", sa="SA", real=False)
    assert list(p.index.get_level_values("date")) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-04-01")]
    assert "/IMF.STA,QNEA/USA+WBG." in calls[0]["url"] and ".V.SA.XDC.Q" in calls[0]["url"]
    assert "gdp_real" not in p
    assert {"code": "PSE", "reason": "no observations returned"} in p.attrs["missing"]


def test_nea_panel_provider_failure_returns_an_empty_frame_and_lists_it(fake_api):
    routes, _ = fake_api
    routes[",ANEA/"] = [_http_error("u", 503)]
    with pytest.warns(UserWarning, match="not obtained"):
        p = imf.imf_nea_panel(["USA"])
    assert p.empty and p.index.names == ["code", "date"]
    assert "gdp" in p.columns and ATTRS <= set(p.attrs)
    assert any("HTTP 503" in m["reason"] for m in p.attrs["missing"])


def test_nea_panel_rejects_bad_frequency_sa_and_codes():
    with pytest.raises(ValueError, match="freq"):
        imf.imf_nea_panel(freq="M")
    with pytest.raises(ValueError, match="sa"):
        imf.imf_nea_panel(sa="X13")
    with pytest.raises(ValueError, match="ISO3"):
        imf.imf_nea_panel(["US"])


# --- monthly, labour ----------------------------------------------------------

def _monthly_routes(routes):
    routes[",CPI/"] = CPI
    routes[",LS/"] = LS_M
    routes[",ER/"] = ER
    routes[",MFS_IR/"] = MFS
    routes[",PI/"] = [_http_error("u", 500)]
    routes[",EER/"] = [_http_error("u", 500)]


def test_monthly_panel_joins_flows_at_month_start_and_clips_future_rows(fake_api):
    routes, calls = fake_api
    _monthly_routes(routes)
    with pytest.warns(UserWarning, match="not obtained"):
        m = imf.imf_monthly_panel(["MEX"])
    assert m.index.names == ["code", "date"]
    assert m.loc[("MEX", pd.Timestamp("2025-02-01")), "cpi"] == pytest.approx(138.3)
    assert m.loc[("MEX", pd.Timestamp("2025-02-01")), "emp"] == pytest.approx(59_500.0)
    assert m.loc[("MEX", pd.Timestamp("2025-02-01")), "fx_usd_eop"] == pytest.approx(20.4)
    assert m.loc[("MEX", pd.Timestamp("2025-02-01")), "policy_rate"] == pytest.approx(9.5)
    assert m.index.get_level_values("date").max() == pd.Timestamp("2025-02-01")
    assert m.attrs["future_rows_dropped"] == 1
    assert set(m.index.get_level_values("code")) == {"MEX"}
    assert sorted(set(m.attrs["aggregates_dropped"])) == ["G129", "G163"]
    assert len(calls) == 4 + 2 * 4      # four good flows, two failing flows tried 1 + 3 times


def test_monthly_panel_reports_a_failed_flow_without_raising(fake_api):
    routes, _ = fake_api
    _monthly_routes(routes)
    with pytest.warns(UserWarning):
        m = imf.imf_monthly_panel(["MEX"])
    failed = {x["variable"] for x in m.attrs["missing"] if "500" in x["reason"]}
    assert failed == {"ip", "ip_sa", "ip_mfg", "neer", "reer"}
    assert m["neer"].isna().all() and m["cpi"].notna().any()


def test_monthly_panel_variables_subset_requests_only_the_flows_it_needs(fake_api):
    routes, calls = fake_api
    _monthly_routes(routes)
    m = imf.imf_monthly_panel(["MEX"], variables=["cpi", "cpi_food"], start="2025-01")
    assert list(m.columns) == ["cpi", "cpi_food"]
    assert len(calls) == 1 and "/MEX.CPI._T+CP01.IX.M?" in calls[0]["url"]
    with pytest.raises(ValueError, match="unknown variable"):
        imf.imf_monthly_panel(variables=["cpi_core"])


def test_labour_panel_quarterly_is_in_thousands_of_persons(fake_api):
    routes, calls = fake_api
    routes[",LS/"] = LS_Q
    with pytest.warns(UserWarning, match="emp"):
        p = imf.imf_labour_panel(["USA"], freq="Q")
    row = p.loc[("USA", pd.Timestamp("2024-07-01"))]
    assert row["urate"] == pytest.approx(4.2) and row["lf"] == pytest.approx(168_000.0)
    assert row["unemp"] == pytest.approx(7_000.0)
    assert ".U+E+LF+UP.PT+PE.Q" in calls[0]["url"]
    with pytest.raises(ValueError, match="freq"):
        imf.imf_labour_panel(freq="W")


# --- commodities, WEO, ICSD -------------------------------------------------------

def test_pcps_is_indexed_by_wld_with_named_and_generic_columns(fake_api):
    routes, calls = fake_api
    routes[",PCPS/"] = PCPS
    with pytest.warns(UserWarning):           # most named series are absent from the fixture
        c = imf.imf_pcps()
    assert c.index.names == ["code", "date"]
    assert set(c.index.get_level_values("code")) == {"WLD"}
    assert c.loc[("WLD", pd.Timestamp("2024-01-01")), "pcom_all"] == pytest.approx(159.3)
    assert c.loc[("WLD", pd.Timestamp("2024-01-01")), "poil"] == pytest.approx(78.9)
    assert c.loc[("WLD", pd.Timestamp("2024-01-01")), "pcoba_usd"] == pytest.approx(28691.4)
    assert "/IMF.RES,PCPS/G001..INDEX+USD.M?" in calls[0]["url"]
    with pytest.raises(ValueError, match="freq"):
        imf.imf_pcps(freq="D")


def test_weo_flags_forecast_years_from_latest_actual_annual_data(fake_api):
    routes, calls = fake_api
    routes["detail=nodata"] = WEO_NODATA
    routes[",WEO/"] = WEO
    w = imf.imf_weo(["USA"], indicators=["gdp", "urate"])
    assert w.loc[("USA", pd.Timestamp("2024-01-01")), "gdp"] == pytest.approx(29_298_000.0)
    assert w["forecast"].dtype == "boolean"
    assert w.loc[("USA", pd.Timestamp("2024-01-01")), "forecast"] is np.False_ or \
        not w.loc[("USA", pd.Timestamp("2024-01-01")), "forecast"]
    assert bool(w.loc[("USA", pd.Timestamp("2026-01-01")), "forecast"])
    assert w.attrs["latest_actual"] == {"USA": {"gdp": 2025, "urate": 2025}}
    assert w.attrs["forecast_basis"] == {"USA": "NGDP"}
    # the attribute request always carries the GDP indicators that set the flag
    nodata = [c["url"] for c in calls if "detail=nodata" in c["url"]][0]
    assert "/USA.NGDP+LUR+NGDP_R.A?" in nodata
    assert "G001" not in set(w.index.get_level_values("code"))


def test_weo_without_the_attribute_pull_leaves_the_flag_missing_not_wrong(fake_api):
    routes, _ = fake_api
    routes["detail=nodata"] = [_http_error("u", 500)]
    routes[",WEO/"] = WEO
    with pytest.warns(UserWarning, match="forecast flag"):
        w = imf.imf_weo("USA", indicators=["gdp", "urate"])
    assert w["forecast"].isna().all() and w["gdp"].notna().any()


def test_weo_rejects_unknown_names_and_bad_vintages():
    with pytest.raises(ValueError, match="unknown WEO"):
        imf.imf_weo(indicators=["gdp_nominal"])
    with pytest.raises(ValueError, match="start"):
        imf.imf_weo(start="last year")
    with pytest.raises(ValueError, match="vintage"):
        imf.imf_weo(vintage="October 2025")


def test_icsd_converts_billions_to_millions(fake_api):
    routes, calls = fake_api
    routes[",ICSD/"] = ICSD
    with pytest.warns(UserWarning):
        k = imf.imf_icsd(["USA"])
    row = k.loc[("USA", pd.Timestamp("2019-01-01"))]
    assert row["inv_gov"] == pytest.approx(741_441.956)
    assert row["k_gov"] == pytest.approx(12_715_461.91)
    assert row["k_gov_gdp"] == pytest.approx(59.489784)
    meta = {m["variable"]: m for m in k.attrs["meta"]}
    assert meta["k_gov"]["scale"] == 1e3 and meta["k_gov"]["unit_mult"] == 6
    assert meta["k_gov"]["units"].startswith("millions")
    assert meta["k_gov_gdp"]["unit_mult"] == 0


def test_period_parser_handles_annual_quarterly_and_monthly_codes():
    s = pd.Series(["2024", "2024-Q3", "2025-M11", "2025-07", "junk", ""])
    out = imf._period_start(s)
    assert list(out[:4]) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-07-01"),
                             pd.Timestamp("2025-11-01"), pd.Timestamp("2025-07-01")]
    assert out[4:].isna().all()


def test_old_imf_ifs_fetcher_warns_that_its_endpoint_is_gone(monkeypatch):
    import puremacro.fetch.imf_ifs as old
    monkeypatch.setattr(old, "_http_json", lambda *a, **k: {"CompactData": {"DataSet": {}}})
    with pytest.warns(DeprecationWarning, match="imf_monthly_panel"):
        old.fetch(["USA"])


# --- review round 2: WEO boundaries, zeros, throttling, budget -----------------

WEO_ARG = """DATAFLOW,COUNTRY,INDICATOR,FREQUENCY,TIME_PERIOD,OBS_VALUE
IMF.RES:WEO(9.0.0),ARG,NGDP_R,A,2010,500000000000
IMF.RES:WEO(9.0.0),ARG,NGDP_R,A,2015,600000000000
IMF.RES:WEO(9.0.0),ARG,NGDP_R,A,2026,700000000000
IMF.RES:WEO(9.0.0),ARG,LP,A,2010,40000000
IMF.RES:WEO(9.0.0),ARG,LP,A,2015,43000000
IMF.RES:WEO(9.0.0),ARG,LP,A,2026,47000000
IMF.RES:WEO(9.0.0),TUV,LP,A,2015,11000
"""

WEO_ARG_NODATA = """DATAFLOW,COUNTRY,INDICATOR,FREQUENCY,TIME_PERIOD,OBS_VALUE,LATEST_ACTUAL_ANNUAL_DATA
IMF.RES:WEO(9.0.0),ARG,NGDP_R,A,,,2025
IMF.RES:WEO(9.0.0),ARG,LP,A,,,2010
"""


def test_weo_flag_follows_real_gdp_not_the_stalest_indicator(fake_api):
    routes, _ = fake_api
    routes["detail=nodata"] = WEO_ARG_NODATA
    routes[",WEO/"] = WEO_ARG
    with pytest.warns(UserWarning, match="TUV"):
        w = imf.imf_weo(indicators=["gdp_real", "pop"])
    arg = w.loc["ARG"]
    # population is projected after 2010, but 2015 is still history for GDP
    assert not arg.loc["2015-01-01", "forecast"] and arg.loc["2026-01-01", "forecast"]
    assert w.attrs["forecast_basis"]["ARG"] == "NGDP_R"
    assert w.attrs["latest_actual"]["ARG"] == {"gdp_real": 2025, "pop": 2010}
    # a country with no boundary at all keeps NA and is reported
    assert w.loc["TUV", "forecast"].isna().all()
    assert any(m.get("code") == "TUV" and m.get("item") == "forecast flag" for m in w.attrs["missing"])


def test_weo_history_only_cuts_each_column_at_its_own_boundary(fake_api):
    routes, _ = fake_api
    routes["detail=nodata"] = WEO_ARG_NODATA
    routes[",WEO/"] = WEO_ARG
    with pytest.warns(UserWarning):
        w = imf.imf_weo(["ARG"], indicators=["gdp_real", "pop"], history_only=True)
    arg = w.loc["ARG"]
    assert list(arg.index) == [pd.Timestamp("2010-01-01"), pd.Timestamp("2015-01-01")]
    assert arg.loc["2015-01-01", "gdp_real"] == pytest.approx(600_000.0)
    assert np.isnan(arg.loc["2015-01-01", "pop"]) and arg.loc["2010-01-01", "pop"] == pytest.approx(40_000.0)
    assert not w["forecast"].any() and w.attrs["history_only"] is True


@pytest.mark.parametrize("value, notes, months, year", [
    ("2024", "", "", 2024),
    ("FY2024/25", "Fiscal year data are mapped to calendar year as follows: FY(t/t+1) = CY(t).", "April/March", 2024),
    ("FY2024/25", "Fiscal year data are mapped to calendar year as follows: FY(t-1/t) = CY(t).", "July/June", 2025),
    ("FY2023/24", "", "April/March", 2023),
    ("FY2023/24", "", "October/September", 2024),
    ("Not applicable", "", "", None),
])
def test_latest_actual_parser_maps_fiscal_years_as_the_methodology_note_says(value, notes, months, year):
    got = imf._latest_actual_year(value, notes, months)
    assert (np.isnan(got) if year is None else got == year)


def test_weo_fiscal_year_boundaries_are_parsed_not_dropped(fake_api):
    routes, _ = fake_api
    routes["detail=nodata"] = (
        "DATAFLOW,COUNTRY,INDICATOR,FREQUENCY,LATEST_ACTUAL_ANNUAL_DATA,METHODOLOGY_NOTES,"
        "START_END_MONTHS_OF_REPORTING_YEAR\n"
        'IMF.RES:WEO(9.0.0),IND,NGDP,A,FY2024/25,"Fiscal year data are mapped to calendar year '
        'as follows: FY(t/t+1) = CY(t).",April/March\n'
        'IMF.RES:WEO(9.0.0),PAK,NGDP,A,FY2024/25,"Fiscal year data are mapped to calendar year '
        'as follows: FY(t-1/t) = CY(t).",July/June\n')
    routes[",WEO/"] = ("DATAFLOW,COUNTRY,INDICATOR,FREQUENCY,TIME_PERIOD,OBS_VALUE\n"
                       + "".join(f"IMF.RES:WEO(9.0.0),{c},NGDP,A,{y},1000000\n"
                                 for c in ("IND", "PAK") for y in (2024, 2025)))
    w = imf.imf_weo(["IND", "PAK"], indicators=["gdp"])
    assert w.attrs["latest_actual"] == {"IND": {"gdp": 2024}, "PAK": {"gdp": 2025}}
    assert w["forecast"].notna().all()
    assert bool(w.loc[("IND", pd.Timestamp("2025-01-01")), "forecast"])
    assert not w.loc[("PAK", pd.Timestamp("2025-01-01")), "forecast"]


def test_nea_panel_treats_exact_zero_levels_as_placeholders(fake_api):
    routes, _ = fake_api
    routes[",ANEA/"] = ANEA + (
        "IMF.STA:ANEA(6.0.1),BFA,B1GQ,V,XDC,A,1960,0,,\n"
        "IMF.STA:ANEA(6.0.1),BFA,B1GQ,Q,XDC,A,1960,5000000,,\n"
        "IMF.STA:ANEA(6.0.1),BFA,P52,V,XDC,A,1960,0,,\n")
    with pytest.warns(UserWarning, match="exact zeros"):
        p = imf.imf_nea_panel()
    bfa = p.loc[("BFA", pd.Timestamp("1960-01-01"))]
    assert np.isnan(bfa["gdp"]) and np.isnan(bfa["gdp_defl"]) and bfa["gdp_real"] == 5.0
    assert bfa["inventories"] == 0.0          # a zero change in stocks is real
    assert p.attrs["zeros_dropped"] == {"gdp": 1}
    meta = {m["variable"]: m for m in p.attrs["meta"]}
    assert meta["gdp"]["n"] == 2              # BFA does not count as having GDP


def test_icsd_drops_zero_capital_stocks_but_keeps_zero_ppp_capital(fake_api):
    routes, calls = fake_api
    routes[",ICSD/"] = ICSD + (
        "IMF.FAD:ICSD(1.0.0),ABW,CAPSTCK_S13_V_XDC,A,2019,0\n"
        "IMF.FAD:ICSD(1.0.0),ABW,CAPSTCK_PUPVT_V_XDC,A,2019,0\n")
    with pytest.warns(UserWarning, match="exact zeros"):
        k = imf.imf_icsd(["USA", "ABW"], start=1990)
    assert "startPeriod=1990" in calls[0]["url"]
    abw = k.loc[("ABW", pd.Timestamp("2019-01-01"))]
    assert np.isnan(abw["k_gov"]) and abw["k_pubpriv"] == 0.0
    assert k.attrs["zeros_dropped"] == {"k_gov": 1}


def test_imf_get_waits_out_a_429_and_then_succeeds(fake_api):
    routes, calls = fake_api
    routes[",CPI/"] = [_http_error("u", 429), CPI]
    raw = imf.imf_get("CPI", "MEX.CPI._T.IX.M")
    assert len(calls) == 2 and raw.attrs["status"] == "ok" and len(raw) == 4


def test_a_persistent_429_gives_an_empty_panel_that_says_so(fake_api):
    routes, calls = fake_api
    routes[",LS/"] = [_http_error("u", 429)]
    with pytest.warns(UserWarning, match="HTTP 429"):
        p = imf.imf_labour_panel(["USA"])
    assert p.empty and p.index.names == ["code", "date"] and ATTRS <= set(p.attrs)
    assert len(calls) == 4
    assert any(m.get("reason") == "HTTP 429 after 4 attempts" for m in p.attrs["missing"])


def test_monthly_panel_stops_asking_after_two_flows_fail_in_a_row(fake_api):
    routes, calls = fake_api
    routes["/data/"] = [_http_error("u", 503)]
    with pytest.warns(UserWarning, match="not obtained"):
        m = imf.imf_monthly_panel(["MEX"])
    assert m.empty and m.index.names == ["code", "date"]
    assert len(calls) == 2 * 4                      # two flows x (1 + 3 retries), then stop
    reasons = {x["variable"]: x["reason"] for x in m.attrs["missing"] if "variable" in x}
    assert reasons["cpi"].startswith("HTTP 503") and reasons["fx_usd"].startswith("skipped after 2")


def test_monthly_panel_keeps_going_after_a_404_flow(fake_api):
    routes, calls = fake_api
    routes[",ER/"] = ER                              # every other flow answers 404 (no series)
    with pytest.warns(UserWarning, match="not obtained"):
        m = imf.imf_monthly_panel(["MEX"])
    assert m.loc[("MEX", pd.Timestamp("2025-02-01")), "fx_usd"] == 20.5
    assert len(calls) == 6


def test_dataflows_reports_an_html_error_page_instead_of_an_empty_catalogue(fake_api):
    routes, _ = fake_api
    routes["/dataflow"] = "<html>gateway</html>"
    with pytest.warns(UserWarning, match="no Dataflow"):
        f = imf.imf_dataflows()
    assert f.empty and f.attrs["missing"][0]["reason"] == "no Dataflow elements in response"


# --- live ---------------------------------------------------------------------

@pytest.mark.network
def test_live_annual_nea_panel_for_three_countries():
    p = imf.imf_nea_panel(["USA", "MEX", "KOR"])
    assert set(p.index.get_level_values("code")) == {"USA", "MEX", "KOR"}
    assert p.loc["USA", "gdp"].dropna().iloc[-1] > 2.0e7


@pytest.mark.network
def test_live_monthly_panel_for_two_countries():
    m = imf.imf_monthly_panel(["USA", "MEX"], start=2020)
    assert m["cpi"].notna().sum() > 100 and m["fx_usd"].loc["MEX"].notna().any()
