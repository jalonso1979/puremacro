"""Offline tests for :mod:`puremacro.fetch.eurostat`.

The module's HTTP seam ``_get`` is patched with tiny synthetic SDMX-CSV
responses (gzipped, as Eurostat serves them with ``compressed=true``); the
polling sleep and the cache write used after asynchronous jobs are patched
too, so nothing touches the network or the on-disk cache.
"""
from __future__ import annotations

import gzip
import http.client
import io
import urllib.error

import numpy as np
import pandas as pd
import pytest

import puremacro.fetch.eurostat as es

DIMS = {
    "nama_10_gdp": ("unit", "na_item"),
    "namq_10_gdp": ("unit", "s_adj", "na_item"),
    "nama_10_a10_e": ("unit", "nace_r2", "na_item"),
    "nama_10_pe": ("unit", "na_item"),
    "prc_hicp_minr": ("unit", "coicop18"),
    "irt_st_m": ("int_rt",),
    "irt_h_mr3_m": ("int_rt",),
    "irt_h_ddmr_m": ("int_rt",),
}


def _csv(flow: str, rows: list[tuple], *, gz: bool = True) -> bytes:
    """rows: (dim values..., geo, period, value)."""
    dims = DIMS[flow]
    recs = []
    for r in rows:
        rec = {"DATAFLOW": f"ESTAT:{flow}(1.0)", "LAST UPDATE": "05/10/26 23:00:00",
               "freq": "A"}
        rec.update(dict(zip(dims, r[:len(dims)])))
        rec.update({"geo": r[-3], "TIME_PERIOD": r[-2], "OBS_VALUE": r[-1],
                    "OBS_FLAG": "", "CONF_STATUS": ""})
        recs.append(rec)
    body = pd.DataFrame(recs).to_csv(index=False).encode()
    return gzip.compress(body) if gz else body


def _http_error(code: int, msg: str) -> urllib.error.HTTPError:
    body = (f'<?xml version="1.0"?><S:Fault xmlns:S="x"><faultcode>150</faultcode>'
            f"<faultstring>{msg}</faultstring></S:Fault>").encode()
    return urllib.error.HTTPError("u", code, "err", {}, io.BytesIO(gzip.compress(body)))


class FakeEurostat:
    """Route URLs to canned answers; record every URL asked for."""

    def __init__(self, routes: dict[str, object]):
        self.routes = routes
        self.urls: list[str] = []

    def __call__(self, url, *, timeout=120.0, pause=3.0, refresh=False):
        self.urls.append(url)
        for frag, ans in self.routes.items():
            if frag in url:
                if callable(ans) and not isinstance(ans, bytes):
                    ans = ans(url)
                if isinstance(ans, BaseException):
                    raise ans
                return ans
        raise urllib.error.HTTPError(url, 503, "unrouted", {}, io.BytesIO(b""))


@pytest.fixture
def fake(monkeypatch):
    cached: dict[str, bytes] = {}
    monkeypatch.setattr(es, "_sleep", lambda s: None)
    monkeypatch.setattr(es, "_cache_put", lambda url, body: cached.__setitem__(url, body))

    def install(routes):
        f = FakeEurostat(routes)
        f.cached = cached
        monkeypatch.setattr(es, "_get", f)
        return f
    return install


# --- annual main block: DE (PD20 + CLV20), EL (CLV20 only), UK (CLV15 only), EA (aggregate)
def _main_annual() -> bytes:
    rows = []
    for geo, cp, clv20, pd20, clv15 in (("DE", 4000.0, 3600.0, 111.0, None),
                                        ("EL", 200.0, 180.0, None, None),
                                        ("UK", 2000.0, None, None, 1900.0),
                                        ("EA20", 9e6, 8e6, 110.0, None)):
        for yr, g in (("2023", 1.0), ("2024", 1.1)):
            rows.append(("CP_MNAC", "B1GQ", geo, yr, cp * g))
            rows.append(("CP_MNAC", "P6", geo, yr, cp * g / 2))
            if clv20 is not None:
                rows.append(("CLV20_MNAC", "B1GQ", geo, yr, clv20))
            if pd20 is not None:
                rows.append(("PD20_NAC", "B1GQ", geo, yr, pd20))
            if clv15 is not None:
                rows.append(("CLV15_MNAC", "B1GQ", geo, yr, clv15))
    return _csv("nama_10_gdp", rows)


def test_eurostat_get_unzips_by_magic_bytes_and_keeps_the_release_stamp(fake):
    f = fake({"nama_10_gdp/A.CP_MNAC.B1GQ.": _main_annual()})
    raw = es.eurostat_get("nama_10_gdp", "A.CP_MNAC.B1GQ.", start=2000)
    assert "compressed=true" in f.urls[0] and "startPeriod=2000" in f.urls[0]
    assert raw.attrs["status"] == "ok"
    assert raw.attrs["last_update"] == "05/10/26 23:00:00"
    assert raw.attrs["flow"] == "nama_10_gdp"
    assert raw["OBS_VALUE"].dtype == float
    assert raw["TIME_PERIOD"].iloc[0] == "2023"      # codes stay strings


def test_eurostat_get_follows_a_queued_request_and_caches_the_result(fake):
    envelope = (b'<?xml version="1.0"?><env:Envelope><env:Body><ns0:syncResponse>'
                b"<queued><id>ab12-cd34</id><status>SUBMITTED</status></queued>"
                b"</ns0:syncResponse></env:Body></env:Envelope>")
    polls = iter([b"<ns1:status>PROCESSING</ns1:status>",
                  b"<ns1:status>AVAILABLE</ns1:status>"])
    data = _main_annual()
    f = fake({"/async/status/ab12-cd34": lambda u: next(polls),
              "/async/data/ab12-cd34": data,
              "/data/nama_10_gdp/": envelope})
    raw = es.eurostat_get("nama_10_gdp", "A..B1GQ.")
    assert len(raw) > 10 and raw.attrs["status"] == "ok"
    assert sum("/status/" in u for u in f.urls) == 2
    # the data URL now maps to the finished file, not to the envelope
    (url, body), = f.cached.items()
    assert "/data/nama_10_gdp/A..B1GQ." in url and body == data


def test_eurostat_get_raises_value_error_naming_the_bad_code(fake):
    fake({"nama_10_gdp": _http_error(
        400, "INVALID_QUERY_DIMENSION_VALUE: The following values for "
             "dimension are not allowed: GEO=US.")})
    with pytest.raises(ValueError, match="GEO=US"):
        es.eurostat_get("nama_10_gdp", "A.CP_MNAC.B1GQ.DE+US")


def test_eurostat_get_returns_an_empty_frame_and_warns_on_provider_failure(fake):
    fake({"nama_10_gdp": urllib.error.HTTPError("u", 503, "down", {}, io.BytesIO(b""))})
    with pytest.warns(UserWarning, match="HTTP 503"):
        raw = es.eurostat_get("nama_10_gdp", "A.CP_MNAC.B1GQ.")
    assert raw.empty and raw.attrs["status"] == "HTTP 503"
    fake({"nama_10_gdp": TimeoutError("timed out")})
    with pytest.warns(UserWarning):
        raw = es.eurostat_get("nama_10_gdp", "A.CP_MNAC.B1GQ.")
    assert raw.empty and "TimeoutError" in raw.attrs["status"]


def test_eurostat_codes_returns_dimensions_in_key_order_with_available_codes(fake):
    dsd = (b'<m:Structure xmlns:m="m" xmlns:s="s"><s:DimensionList>'
           b'<s:Dimension id="geo" position="3"/><s:Dimension id="freq" position="1"/>'
           b'<s:Dimension id="unit" position="2"/>'
           b'<s:TimeDimension id="TIME_PERIOD" position="4"/></s:DimensionList></m:Structure>')
    cc = (b'<m:Structure xmlns:m="m" xmlns:c="c" xmlns:s="s"><s:CubeRegion include="true">'
          b'<c:KeyValue id="freq"><c:Value>A</c:Value></c:KeyValue>'
          b'<c:KeyValue id="unit"><c:Value>CP_MNAC</c:Value><c:Value>CLV20_MNAC</c:Value></c:KeyValue>'
          b'<c:KeyValue id="geo"><c:Value>DE</c:Value><c:Value>EA20</c:Value></c:KeyValue>'
          b'<c:KeyValue id="TIME_PERIOD"><c:Value>2024</c:Value></c:KeyValue>'
          b"</s:CubeRegion></m:Structure>")
    fake({"/datastructure/ESTAT/x_flow": gzip.compress(dsd),
          "/contentconstraint/ESTAT/x_flow": cc})
    codes = es.eurostat_codes("x_flow")
    assert list(codes) == ["freq", "unit", "geo", "TIME_PERIOD"]
    assert codes["unit"] == ["CP_MNAC", "CLV20_MNAC"]
    fake({"/datastructure/": _http_error(404, "ERR_NOT_FOUND_2: no such flow")})
    with pytest.raises(ValueError, match="ERR_NOT_FOUND"):
        es.eurostat_codes("nope")


def test_annual_panel_has_iso3_period_start_dates_volumes_and_deflators(fake):
    fake({"nama_10_gdp": _main_annual()})
    df = es.eurostat_na_panel()
    assert df.index.names == ["code", "date"]
    assert set(df.index.get_level_values("code")) == {"DEU", "GRC", "GBR"}  # EA20 dropped
    assert df.index.get_level_values("date").min() == pd.Timestamp("2023-01-01")
    assert all(dt == np.float64 for dt in df.dtypes)
    assert df.loc[("DEU", "2024-01-01"), "gdp"] == pytest.approx(4400.0)
    assert df.loc[("DEU", "2024-01-01"), "exports"] == pytest.approx(2200.0)
    # Germany: published PD20; Greece: implicit 100*CP/CLV20; UK: CLV15 fallback
    assert df.loc[("DEU", "2023-01-01"), "gdp_defl"] == pytest.approx(111.0)
    assert df.loc[("GRC", "2023-01-01"), "gdp_defl"] == pytest.approx(100 * 200 / 180)
    assert df.loc[("GBR", "2023-01-01"), "gdp_real"] == pytest.approx(1900.0)
    meta = {(m["code"], m["variable"]): m for m in df.attrs["meta"]}
    assert meta[("GBR", "gdp")]["volume_base"] == "CLV15_MNAC"
    assert meta[("DEU", "gdp")]["defl_source"] == "PD20_NAC"
    assert meta[("GRC", "gdp")]["defl_source"] == "100*CP_MNAC/CLV20_MNAC"
    assert meta[("DEU", "gdp")]["unit_mult"] == 6
    assert meta[("DEU", "gdp")]["first"] == "2023-01-01" and meta[("DEU", "gdp")]["n"] == 2
    assert set(df.attrs) >= {"meta", "source", "fetched_at", "missing", "last_update"}
    assert "nama_10_gdp" in df.attrs["source"]
    assert "cons_hh" in " ".join(df.attrs["missing"])     # asked for, never published
    assert list(df.columns[:3]) == ["gdp", "cons_hh", "cons_gov"]  # stable schema
    assert df["cons_hh"].isna().all()


def test_quarterly_panel_prefers_sca_then_sa_then_nsa_per_country(fake):
    rows = []
    for geo, adjs in (("FR", ("SCA", "NSA")), ("TR", ("SA", "NSA")), ("XK", ("NSA",))):
        for adj in adjs:
            val = {"SCA": 1.0, "SA": 2.0, "NSA": 3.0}[adj]
            for q in ("2025-Q4", "2026-Q1"):
                rows.append(("CP_MNAC", adj, "B1GQ", geo, q, val))
                rows.append(("CLV20_MNAC", adj, "B1GQ", geo, q, val / 2))
    fake({"namq_10_gdp": _csv("namq_10_gdp", rows)})
    df = es.eurostat_na_panel(freq="Q")
    assert df.loc[("FRA", "2026-01-01"), "gdp"] == 1.0
    assert df.loc[("TUR", "2026-01-01"), "gdp"] == 2.0
    assert df.loc[("XKX", "2025-10-01"), "gdp"] == 3.0
    assert df.loc[("XKX", "2025-10-01"), "gdp_defl"] == pytest.approx(200.0)
    sa = {m["code"]: m["sa"] for m in df.attrs["meta"] if m["variable"] == "gdp"}
    assert sa == {"FRA": "SCA", "TUR": "SA", "XKX": "NSA"}
    strict = es.eurostat_na_panel(freq="Q", sa="NSA")
    assert strict.loc[("FRA", "2026-01-01"), "gdp"] == 3.0


def test_retired_codes_are_dropped_and_the_request_repeated(fake):
    good = _main_annual()

    def answer(url):
        if "D21X31" in url or "P31_S14_S15" in url:
            return _http_error(400, "INVALID_QUERY_DIMENSION_VALUE: Query is invalid. "
                                    "The following values for dimension are not "
                                    "allowed: NA_ITEM=P31_S14_S15.")
        return good
    f = fake({"nama_10_gdp": answer})
    df = es.eurostat_na_panel(["DEU"])
    assert len(f.urls) == 2 and "P31_S14_S15" not in f.urls[1]
    assert df.loc[("DEU", "2023-01-01"), "gdp"] == 4000.0
    assert any("P31_S14_S15" in m for m in df.attrs["missing"])


def test_provider_failure_gives_an_empty_panel_with_the_same_shape(fake):
    fake({})          # every URL answers HTTP 503
    with pytest.warns(UserWarning):
        df = es.eurostat_na_panel(["DEU", "FRA"], durability=True)
    assert df.empty and df.index.names == ["code", "date"]
    assert df.attrs["meta"] == ()
    assert any("HTTP 503" in m for m in df.attrs["missing"])
    assert "DEU: no data" in df.attrs["missing"]
    assert df.attrs["fetched_at"].endswith("+00:00")


def test_a_connection_dropped_mid_body_gives_an_empty_panel_not_an_exception(fake):
    fake({"/data/": http.client.IncompleteRead(b"partial")})
    with pytest.warns(UserWarning):
        df = es.eurostat_na_panel(["DEU"])
    assert df.empty and df.index.names == ["code", "date"]
    assert any("IncompleteRead" in m for m in df.attrs["missing"])
    with pytest.warns(UserWarning):
        m = es.eurostat_monthly_panel(["DEU"], variables=["cpi"])
    assert m.empty and any("IncompleteRead" in x for x in m.attrs["missing"])


def test_a_truncated_gzip_is_fetched_again_once_and_then_reported(fake):
    bad = _main_annual()[:40]
    f = fake({"nama_10_gdp": bad})
    with pytest.warns(UserWarning, match="corrupt gzip"):
        raw = es.eurostat_get("nama_10_gdp", "A.CP_MNAC.B1GQ.")
    assert raw.empty and raw.attrs["status"] == "corrupt gzip"
    assert len(f.urls) == 2                       # the second asks with refresh
    good = iter([bad, _main_annual()])
    fake({"nama_10_gdp": lambda u: next(good)})
    raw = es.eurostat_get("nama_10_gdp", "A.CP_MNAC.B1GQ.")
    assert raw.attrs["status"] == "ok" and len(raw) > 10


def test_an_async_job_unknown_twice_is_reported_as_a_failure_not_as_no_rows(fake):
    envelope = (b"<env:Envelope><queued><id>ab12</id><status>SUBMITTED</status>"
                b"</queued></env:Envelope>")
    fake({"/async/status/ab12": b"<status>UNKNOWN_REQUEST</status>",
          "/data/namq_10_gdp/": envelope})
    with pytest.warns(UserWarning, match="unknown after retry"):
        df = es.eurostat_na_panel(["DEU"], freq="Q")
    assert any("async job unknown after retry" in m for m in df.attrs["missing"])


def test_an_empty_panel_still_carries_the_requested_columns(fake):
    fake({})
    with pytest.warns(UserWarning):
        df = es.eurostat_na_panel(["DEU"])
    assert {"gdp", "gdp_real", "gdp_defl", "imports"} <= set(df.columns)
    assert df["gdp"].empty
    with pytest.warns(UserWarning):
        m = es.eurostat_monthly_panel(variables=["cpi", "urate"])
    assert list(m.columns) == ["cpi", "urate"]


def test_eurostat_meta_tabulates_the_per_country_records(fake):
    fake({"nama_10_gdp/": _main_annual()})
    df = es.eurostat_na_panel(["DEU", "GBR"])
    meta = es.eurostat_meta(df)
    row = meta.set_index(["code", "variable"]).loc[("GBR", "gdp")]
    assert row["volume_base"] == "CLV15_MNAC" and row["unit_mult"] == 6
    assert es.eurostat_meta(es._empty_panel()).empty


def test_caller_errors_raise_value_error_listing_the_choices():
    with pytest.raises(ValueError, match=r"\['A', 'Q'\]"):
        es.eurostat_na_panel(freq="M")
    with pytest.raises(ValueError, match="prefer"):
        es.eurostat_na_panel(sa="X13")
    with pytest.raises(ValueError, match="DEU"):
        es.eurostat_na_panel(["ZZZ"])
    with pytest.raises(ValueError, match="cpi_core"):
        es.eurostat_monthly_panel(variables=["inflation"])


def test_labor_block_reports_hours_in_millions_and_population_in_thousands(fake):
    lab = _csv("nama_10_a10_e", [
        ("THS_PER", "TOTAL", "EMP_DC", "AT", "2024", 4500.0),
        ("THS_PER", "TOTAL", "SAL_DC", "AT", "2024", 3900.0),
        ("THS_HW", "TOTAL", "EMP_DC", "AT", "2024", 7_000_000.0)])
    pop = _csv("nama_10_pe", [("THS_PER", "POP_NC", "AT", "2024", 9150.0),
                              ("THS_PER", "POP_NC", "EU27_2020", "2024", 449000.0)])
    fake({"nama_10_gdp": _main_annual(), "nama_10_a10_e": lab, "nama_10_pe": pop})
    df = es.eurostat_na_panel(labor=True, real=False)
    row = df.loc[("AUT", "2024-01-01")]
    assert row["hours"] == pytest.approx(7000.0)
    assert row["emp"] == 4500.0 and row["emp_employees"] == 3900.0
    assert row["pop"] == 9150.0
    assert "gdp_real" not in df.columns
    units = {m["variable"]: m["units"] for m in df.attrs["meta"] if m["code"] == "AUT"}
    assert units["hours"] == "millions of hours worked"
    assert units["pop"] == "thousands of persons"


def test_monthly_panel_splices_euro_area_rates_after_adoption(fake):
    st = _csv("irt_st_m", [("IRT_M3", "EA", "1998-12", 3.5), ("IRT_M3", "EA", "1999-01", 3.1),
                           ("IRT_M3", "EA", "2001-01", 4.8),
                           ("IRT_M3", "SE", "1999-01", 3.3), ("IRT_M3", "US", "1999-01", 4.9)])
    h3 = _csv("irt_h_mr3_m", [("IRT_M3", "DE", "1998-12", 3.6), ("IRT_M3", "DE", "1999-01", 9.9),
                              ("IRT_M3", "EL", "2000-12", 8.0)])
    hon = _csv("irt_h_ddmr_m", [("IRT_DTD", "DE", "1998-12", 3.0)])
    hicp = _csv("prc_hicp_minr", [("I15", "TOTAL", "DE", "1999-01", 70.1),
                                  ("I15", "TOT_X_NRG_FOOD", "DE", "1999-01", 72.0),
                                  ("I15", "TOTAL", "EA", "1999-01", 71.0)])
    fake({"irt_st_m": st, "irt_h_mr3_m": h3, "irt_h_ddmr_m": hon, "prc_hicp_minr": hicp})
    m = es.eurostat_monthly_panel(variables=["rate_3m", "cpi", "cpi_core"])
    assert m.index.names == ["code", "date"]
    assert m.loc[("DEU", "1998-12-01"), "rate_3m"] == 3.6     # national history
    assert m.loc[("DEU", "1999-01-01"), "rate_3m"] == 3.1     # EA, not the 9.9
    assert m.loc[("GRC", "2000-12-01"), "rate_3m"] == 8.0
    assert m.loc[("GRC", "2001-01-01"), "rate_3m"] == 4.8
    assert m.loc[("USA", "1999-01-01"), "rate_3m"] == 4.9
    assert m.loc[("DEU", "1999-01-01"), "cpi_core"] == 72.0
    assert "EA" not in m.index.get_level_values("code")
    deu = [x for x in m.attrs["meta"] if x["code"] == "DEU" and x["variable"] == "rate_3m"][0]
    assert deu["splice"] == "irt_h_mr3_m before 1999-01; EA from 1999-01"
    sub = es.eurostat_monthly_panel(["SWE"], variables=["rate_3m"])
    assert set(sub.index.get_level_values("code")) == {"SWE"}


@pytest.mark.network
def test_live_annual_main_block_covers_about_forty_reporters():
    df = es.eurostat_na_panel()
    assert df.index.get_level_values("code").nunique() >= 38
    assert df.loc[("FRA", "1975-01-01"), "gdp"] > 0
    assert "EA20" not in df.index.get_level_values("code")
