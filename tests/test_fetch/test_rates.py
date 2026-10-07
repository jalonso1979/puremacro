"""Offline tests for :mod:`puremacro.fetch.rates` and the stdlib Pink Sheet reader.

Every test serves tiny synthetic bodies through the module's single network
seam (``rates._get``, and ``wb_pink_sheet.cached_get`` for the workbook), in
the shapes the providers return: FRED ``fredgraph.csv``, ECB SDMX-CSV with
``detail=dataonly``, the Bundesbank's wide CSV, Yahoo's chart JSON and an
``.xlsx`` built here with ``zipfile`` (shared strings, as the World Bank
writes them).
"""
from __future__ import annotations

import ast
import io
import json
import urllib.error
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from puremacro.fetch import rates
from puremacro.fetch import wb_pink_sheet as pk


# ---------------------------------------------------------------------------
# synthetic provider bodies
# ---------------------------------------------------------------------------

def _fred(sid: str, start: str, n: int, base: float, step: float = 0.01) -> bytes:
    dates = pd.date_range(start, periods=n, freq="MS")
    vals = [f"{base + step * i:.2f}" for i in range(n)]
    vals[1] = "."                                  # FRED's missing-value marker
    lines = ["observation_date," + sid] + [f"{d:%Y-%m-%d},{v}" for d, v in zip(dates, vals)]
    return ("\n".join(lines) + "\n").encode()


_ECB_HEAD = "KEY,FREQ,REF_AREA,CURRENCY,PROVIDER_FM,INSTRUMENT_FM,PROVIDER_FM_ID,DATA_TYPE_FM,TIME_PERIOD,OBS_VALUE"
_IRS_HEAD = ("KEY,FREQ,REF_AREA,IR_TYPE,TR_TYPE,MATURITY_CAT,BS_COUNT_SECTOR,"
             "CURRENCY_TRANS,IR_BUS_COV,IR_FV_TYPE,TIME_PERIOD,OBS_VALUE")


def _fm_series(area, cur, prov, inst, pid, typ, start, n, value):
    key = f"FM.M.{area}.{cur}.{prov}.{inst}.{pid}.{typ}"
    dims = key.split(".")[1:]
    per = pd.period_range(start, periods=n, freq="M")
    return [",".join([key] + dims + [str(p), str(value(i) if callable(value) else value)])
            for i, p in enumerate(per)]


def _irs_series(area, irtype, trtype, mat, cur, start, n, value):
    key = f"IRS.M.{area}.{irtype}.{trtype}.{mat}.0000.{cur}.N.Z"
    dims = key.split(".")[1:]
    per = pd.period_range(start, periods=n, freq="M")
    return [",".join([key] + dims + [str(p), str(value)]) for p in per]


def _ecb(head: str, rows: list[str]) -> bytes:
    return ("\n".join([head] + rows) + "\n").encode()


_MONEY = _ecb(_ECB_HEAD,
              _fm_series("U2", "EUR", "RT", "MM", "EURIBOR3MD_", "HSTA", "1994-01", 400, 3.0)
              + _fm_series("U2", "EUR", "RT", "MM", "EURIBOR1MD_", "HSTA", "1994-01", 400, 2.9)
              + _fm_series("U2", "EUR", "4F", "MM", "EONIA", "HSTA", "1994-01", 336, 2.0)
              + _fm_series("U2", "EUR", "4F", "MM", "UONSTR", "HSTA", "2017-04", 120, 1.0))

_IRS_10Y = _ecb(_IRS_HEAD,
                _irs_series("DE", "L", "L40", "CI", "EUR", "1990-01", 440, 6.5)
                + _irs_series("BG", "L", "L40", "CI", "BGN", "2003-01", 276, 4.0)
                + _irs_series("BG", "L", "L40", "CI", "EUR", "2025-06", 15, 3.5)
                + _irs_series("U2", "L", "L40", "CI", "EUR", "1993-01", 400, 9.9))

_LONG_10Y = _ecb(_ECB_HEAD,
                 _fm_series("US", "USD", "4F", "BB", "US10YT_RR", "YLDA", "1900-01", 700, 3.3)
                 + _fm_series("JP", "JPY", "4F", "BB", "JP10YT_RR", "YLDA", "1972-03", 600, 7.0))

_STOCKS_ECB = _ecb(_ECB_HEAD,
                   _fm_series("US", "USD", "DS", "EI", "S_PCOMP", "HSTA", "1964-01", 120, lambda i: 80 + i))

_BBK = ('\ufeff"",BBSIS.M.I.UMR.RD.EUR.X2000.B.A.A.R.A.A._Z._Z.A,FLAGS\n'
        '"",Yields on debt securities outstanding/Corporate bonds (non-MFIs)/Monthly averages,\n'
        'BBK_UNIT_ENG,percent,\nunit multiplier,One,\nlast update,2026-10-01 10:32:50,\n'
        '1957-01,8.10,\n1957-02,8.00,\n1957-03,.,\n1957-04,7.90,\n').encode()


def _yahoo(ts: list[int], close: list[float | None], *, tz="Asia/Tokyo", gran="1mo") -> bytes:
    doc = {"chart": {"result": [{
        "meta": {"currency": "JPY", "exchangeTimezoneName": tz, "gmtoffset": 32400,
                 "dataGranularity": gran},
        "timestamp": ts,
        "indicators": {"quote": [{"close": close}]},
    }], "error": None}}
    return json.dumps(doc).encode()


def _utc(s: str) -> int:
    return int(pd.Timestamp(s, tz="UTC").timestamp())


# Nikkei-style bars, stamped at Tokyo midnight = 15:00 UTC the previous day.
_N225 = _yahoo([_utc("2019-12-31 15:00"), _utc("2020-01-31 15:00"), _utc("2020-02-29 15:00"),
                _utc("2020-03-05 06:00")],
               [23000.0, 22000.0, None, 18000.0])


class _Server:
    """Route URLs to bodies; anything unrouted answers 404."""

    def __init__(self, routes: dict[str, bytes | Exception]):
        self.routes = routes
        self.calls: list[str] = []

    def __call__(self, url, **kw):
        self.calls.append(url)
        for needle, body in self.routes.items():
            if needle in url:
                if isinstance(body, Exception):
                    raise body
                return body
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, io.BytesIO(b"<html>404</html>"))


def _rates_routes() -> dict:
    return {
        "id=IRLTLT01USM156N": _fred("IRLTLT01USM156N", "1953-04", 40, 2.8),
        "id=IRLTLT01DEM156N": _fred("IRLTLT01DEM156N", "1956-05", 60, 6.0),
        "id=IRLTLT01ITM156N": _fred("IRLTLT01ITM156N", "1991-03", 30, 12.0),
        "id=IR3TIB01USM156N": _fred("IR3TIB01USM156N", "1964-06", 20, 3.6),
        "id=IR3TIB01DEM156N": _fred("IR3TIB01DEM156N", "1960-01", 600, 4.0, 0.0),
        "id=IR3TIB01ITM156N": _fred("IR3TIB01ITM156N", "1978-10", 600, 11.0, 0.0),
        "id=IRSTCI01USM156N": _fred("IRSTCI01USM156N", "1954-07", 20, 0.8),
        "id=IRSTCI01DEM156N": _fred("IRSTCI01DEM156N", "1960-01", 600, 3.8, 0.0),
        "id=IRSTCI01ITM156N": _fred("IRSTCI01ITM156N", "1999-01", 300, 2.9, 0.0),
        "id=AAA": _fred("AAA", "1919-01", 50, 5.35),
        "id=BAA": _fred("BAA", "1919-01", 50, 7.12),
        "EURIBOR1MD_+EURIBOR3MD_": _MONEY,
        "IRS/M..L.L40.CI.0000..N.Z": _IRS_10Y,
        "US10YT_RR+JP10YT_RR": _LONG_10Y,
    }


@pytest.fixture
def server(monkeypatch):
    srv = _Server(_rates_routes())
    monkeypatch.setattr(rates, "_get", srv)
    return srv


# ---------------------------------------------------------------------------
# rates_panel
# ---------------------------------------------------------------------------

def test_the_panel_is_indexed_by_code_and_month_start_with_float_columns(server):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = rates.rates_panel(["USA", "DEU", "ITA", "BGR"], start="1900")
    assert df.index.names == ["code", "date"]
    assert list(df.columns) == ["rate_on", "rate_3m", "yield_10y", "yield_corp_aaa", "yield_corp_baa"]
    assert all(df[c].dtype == np.float64 for c in df.columns)
    dates = df.index.get_level_values("date")
    assert (dates == dates.to_period("M").to_timestamp()).all()
    assert set(df.index.get_level_values("code")) == {"USA", "DEU", "ITA", "BGR"}
    assert set(df.attrs) >= {"meta", "source", "fetched_at", "missing"}
    assert isinstance(df.attrs["meta"], tuple) and all(type(m) is dict for m in df.attrs["meta"])


def test_euro_members_switch_to_euribor_at_entry_and_keep_the_national_rate_before(server):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = rates.rates_panel(["ITA", "DEU"], variables=("rate_3m", "rate_on"), start="1970")
    ita = df.loc["ITA", "rate_3m"].dropna()
    assert ita.loc["1998-12-01"] == pytest.approx(11.0)          # FRED national
    assert ita.loc["1999-01-01"] == pytest.approx(3.0)           # EURIBOR
    on = df.loc["DEU", "rate_on"].dropna()
    assert on.loc["1998-12-01"] == pytest.approx(3.8)
    assert on.loc["2019-09-01"] == pytest.approx(2.0)            # EONIA
    assert on.loc["2019-10-01"] == pytest.approx(1.0)            # €STR
    roles = {(m["code"], m["variable"], m["role"]) for m in df.attrs["meta"]}
    assert ("ITA", "rate_3m", "3-month EURIBOR from euro entry") in roles
    assert ("ITA", "rate_3m", "national rate before euro entry") in roles


def test_late_euro_members_get_euribor_only_from_their_entry_month(server):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = rates.rates_panel(["BGR"], variables=("rate_3m",), start="1990")
    s = df.loc["BGR", "rate_3m"].dropna()
    assert s.index.min() == pd.Timestamp("2026-01-01")


def test_ecb_yields_fill_eu_countries_fred_lacks_and_aggregates_are_dropped(server):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = rates.rates_panel(None, variables=("yield_10y",), start="1990")
    codes = set(df.index.get_level_values("code"))
    assert "BGR" in codes
    assert not codes & {"U2", "EMU", "EA", "EUR", "EZ"}
    bg = df.loc["BGR", "yield_10y"].dropna()
    assert bg.loc["2025-05-01"] == pytest.approx(4.0)            # BGN series
    assert bg.loc["2026-03-01"] == pytest.approx(3.5)            # EUR series wins where both report
    leg = [m for m in df.attrs["meta"] if m["code"] == "BGR"]
    assert leg and leg[0]["source"] == "ECB" and leg[0]["units"] == "percent per annum"
    # no FRED request is spent on an id known not to exist
    assert not any("IRLTLT01BGM156N" in u for u in server.calls)


def test_fred_stays_primary_and_the_ecb_only_adds_months_it_lacks(server):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = rates.rates_panel(["DEU"], variables=("yield_10y",), start="1950")
    de = df.loc["DEU", "yield_10y"].dropna()
    assert de.loc["1956-05-01"] == pytest.approx(6.0)            # FRED
    assert de.loc["1990-01-01"] == pytest.approx(6.5)            # ECB, after FRED's 60 months
    legs = {m["role"]: m for m in df.attrs["meta"] if m["code"] == "DEU"}
    assert legs["primary"]["source"] == "FRED"
    assert legs["second source"]["source"] == "ECB"
    assert legs["primary"]["n"] + legs["second source"]["n"] == len(de)


def test_the_united_states_yield_reaches_back_to_1900_through_the_ecb_history(server):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = rates.rates_panel(["USA"], variables=("yield_10y",), start="1900")
    us = df.loc["USA", "yield_10y"].dropna()
    assert us.index.min() == pd.Timestamp("1900-01-01")
    assert us.loc["1953-04-01"] == pytest.approx(2.8)            # FRED wins where it exists


def test_start_trims_and_corporate_yields_are_usa_only(server):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = rates.rates_panel(["USA", "DEU"], start="1922")
    assert df.index.get_level_values("date").min() >= pd.Timestamp("1922-01-01")
    assert df.loc["DEU", "yield_corp_aaa"].isna().all()
    assert df.loc["USA", "yield_corp_baa"].dropna().iloc[0] == pytest.approx(7.12 + 0.01 * 36)


def test_german_corporate_yield_comes_from_the_bundesbank(monkeypatch):
    srv = _Server({"BBSIS/M.I.UMR.RD.EUR.X2000": _BBK})
    monkeypatch.setattr(rates, "_get", srv)
    df = rates.rates_panel(["DEU"], variables=("yield_corp_de",), start="1950")
    s = df.loc["DEU", "yield_corp_de"]
    assert list(s.index) == list(pd.to_datetime(["1957-01-01", "1957-02-01", "1957-04-01"]))
    assert s.iloc[0] == pytest.approx(8.10)


def test_one_failing_series_is_listed_as_missing_without_raising(monkeypatch):
    routes = _rates_routes()
    routes["id=IRLTLT01ITM156N"] = urllib.error.HTTPError("u", 500, "err", {}, io.BytesIO(b""))
    srv = _Server(routes)
    monkeypatch.setattr(rates, "_get", srv)
    with pytest.warns(UserWarning, match="could not be fetched"):
        df = rates.rates_panel(["ITA", "USA"], variables=("yield_10y",), start="1950")
    miss = df.attrs["missing"]
    assert any(m["code"] == "ITA" and "HTTP 500" in m["reason"] for m in miss)
    assert not df.loc["USA", "yield_10y"].dropna().empty


def test_a_partial_failure_marks_the_panel_incomplete_and_names_the_series(monkeypatch):
    routes = _rates_routes()
    routes["id=IRLTLT01ITM156N"] = urllib.error.HTTPError("u", 429, "slow down", {}, io.BytesIO(b""))
    monkeypatch.setattr(rates, "_get", _Server(routes))
    with pytest.warns(UserWarning, match=r"ITA/yield_10y \(FRED IRLTLT01ITM156N: HTTP 429\)"):
        df = rates.rates_panel(["ITA", "USA"], variables=("yield_10y",), start="1950")
    assert df.attrs["complete"] is False
    assert not df.empty


def test_a_full_pull_is_marked_complete(server):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        df = rates.rates_panel(["USA"], variables=("yield_10y", "yield_corp_aaa"), start="1900")
    assert df.attrs["complete"] is True and df.attrs["missing"] == ()


def test_a_country_with_no_source_does_not_make_the_panel_incomplete(server):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = rates.rates_panel(["ARG", "USA"], variables=("yield_corp_aaa",), start="1900")
    assert df.attrs["missing"] and df.attrs["complete"] is True


def test_a_cached_body_that_does_not_parse_is_downloaded_once_more(monkeypatch):
    calls = []

    def get(url, *, refresh=False, **kw):
        calls.append(refresh)
        if not refresh:
            return b"<html>FRED is down for maintenance</html>"
        return _fred("AAA", "1919-01", 5, 5.0)

    monkeypatch.setattr(rates, "_get", get)
    df = rates.fetch_fred_many(["AAA"])
    assert calls == [False, True]
    assert df["AAA"].notna().sum() == 4 and df.attrs["missing"] == ()


def test_a_body_that_still_fails_after_the_retry_is_reported_once(monkeypatch):
    calls = []

    def get(url, *, refresh=False, **kw):
        calls.append(refresh)
        return b"<html>still down</html>"

    monkeypatch.setattr(rates, "_get", get)
    with pytest.warns(UserWarning, match="unparseable"):
        df = rates.fetch_fred_many(["AAA"])
    assert calls == [False, True]
    assert df.attrs["missing"][0]["reason"].startswith("unparseable")


def test_a_cached_quarterly_yahoo_response_is_refetched(monkeypatch):
    calls = []

    def get(url, *, refresh=False, **kw):
        calls.append(refresh)
        return _yahoo([_utc("2019-12-31 15:00")], [1.0], gran="3mo") if not refresh else _N225

    monkeypatch.setattr(rates, "_get", get)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                    # the ECB leg is not served here
        df = rates.stock_index_monthly(["JPN"], start="2019")
    assert calls[:2] == [False, True]
    assert df.loc["JPN", "stock_idx"].notna().sum() == 3


def test_rate_columns_reuse_the_house_names_of_the_other_fetchers():
    assert {"rate_on", "rate_3m", "yield_10y", "yield_corp_aaa", "yield_corp_baa"} <= set(rates.RATE_VARIABLES)
    assert not {"ir_on", "ir3m", "corp_aaa", "corp_baa", "corp_de"} & set(rates.RATE_VARIABLES)


def test_a_country_without_any_source_is_reported_not_an_error(server):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = rates.rates_panel(["ARG", "USA"], variables=("yield_10y", "yield_corp_aaa"))
    missing = {(m["code"], m["variable"]) for m in df.attrs["missing"]}
    assert {("ARG", "yield_corp_aaa"), ("ARG", "yield_10y")} <= missing
    assert not any("01ARM156N" in u for u in server.calls)       # no request spent on a known 404


def test_a_dead_network_gives_the_empty_frame_with_the_same_shape(monkeypatch):
    srv = _Server({"": OSError("offline")})
    monkeypatch.setattr(rates, "_get", srv)
    with pytest.warns(UserWarning, match="nothing obtained"):
        df = rates.rates_panel(["USA", "DEU"])
    assert df.empty
    assert df.index.names == ["code", "date"]
    assert list(df.columns) == ["rate_on", "rate_3m", "yield_10y", "yield_corp_aaa", "yield_corp_baa"]
    assert set(df.attrs) >= {"meta", "source", "fetched_at", "missing"}
    assert len(df.attrs["missing"]) >= 5


@pytest.mark.parametrize("kwargs, match", [
    ({"variables": ("yield_2y",)}, "unknown variables"),
    ({"codes": ["DE"]}, "ISO3"),
    ({"codes": ["EMU"]}, "aggregate"),
    ({"start": "not a date"}, "start"),
])
def test_caller_errors_raise_value_error_listing_the_choices(server, kwargs, match):
    with pytest.raises(ValueError, match=match):
        rates.rates_panel(**kwargs)
    assert server.calls == []


# ---------------------------------------------------------------------------
# fetch_fred_many and ecb_get
# ---------------------------------------------------------------------------

def test_fetch_fred_many_keeps_good_ids_and_lists_the_bad_one(server):
    with pytest.warns(UserWarning, match="NOPE"):
        df = rates.fetch_fred_many(["AAA", "NOPE", "BAA"], start="1920")
    assert list(df.columns) == ["AAA", "BAA"]
    assert df.index.name == "date"
    assert df.index.min() == pd.Timestamp("1920-01-01")
    assert df.attrs["missing"][0]["id"] == "NOPE" and df.attrs["missing"][0]["reason"] == "HTTP 404"
    assert [m["id"] for m in df.attrs["meta"]] == ["AAA", "BAA"]
    assert sum("id=" in u for u in server.calls) == 3             # one request per id


def test_fetch_fred_many_rejects_an_empty_list():
    with pytest.raises(ValueError):
        rates.fetch_fred_many([])


def test_ecb_get_returns_the_raw_frame_with_numeric_values(server):
    raw = rates.ecb_get("FM", "M.U2.EUR.RT+4F.MM.EURIBOR1MD_+EURIBOR3MD_+EONIA+UONSTR.HSTA",
                        start="1999")
    assert raw.attrs["status"] == "ok"
    assert "startPeriod=1999" in raw.attrs["url"] and "detail=dataonly" in raw.attrs["url"]
    assert raw["OBS_VALUE"].dtype == np.float64
    assert {"KEY", "REF_AREA", "TIME_PERIOD"} <= set(raw.columns)


def test_ecb_get_reports_a_404_as_status_on_an_empty_frame(server):
    with pytest.warns(UserWarning, match="HTTP 404"):
        raw = rates.ecb_get("FM", "M.XX.EUR.RT.MM.EURIBOR3MD_.HSTA")
    assert raw.empty and raw.attrs["status"] == "HTTP 404"


def test_ecb_get_rejects_a_malformed_flow_or_key():
    with pytest.raises(ValueError, match="flow"):
        rates.ecb_get("FM/x", "M.U2")
    with pytest.raises(ValueError, match="key"):
        rates.ecb_get("FM", "")


# ---------------------------------------------------------------------------
# stock_index_monthly
# ---------------------------------------------------------------------------

def test_yahoo_bars_are_dated_by_the_exchange_local_month(monkeypatch):
    srv = _Server({"chart/%5EN225": _N225})
    monkeypatch.setattr(rates, "_get", srv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = rates.stock_index_monthly(["JPN"])
    s = df.loc["JPN", "stock_idx"].dropna()
    # 31 Dec 15:00 UTC is 1 Jan in Tokyo; the empty February bar is dropped
    assert list(s.index) == list(pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]))
    assert s.iloc[0] == 23000.0
    assert "period1=0" in srv.calls[0] and "range=max" not in srv.calls[0]
    meta = [m for m in df.attrs["meta"] if m["variable"] == "stock_idx"][0]
    assert meta["currency"] == "JPY" and meta["key"] == "^N225"


def test_dead_tickers_and_quarterly_bars_are_reported_as_missing(monkeypatch):
    srv = _Server({"chart/%5EGSPC": _yahoo([_utc("2020-01-01 05:00")], [3000.0], gran="3mo"),
                   "S_PCOMP": _STOCKS_ECB})
    monkeypatch.setattr(rates, "_get", srv)
    with pytest.warns(UserWarning, match="missing"):
        df = rates.stock_index_monthly(["USA", "POL"])
    reasons = {m["code"]: m["reason"] for m in df.attrs["missing"]}
    assert "3mo" in reasons["USA"]
    assert "^WIG20" in reasons["POL"]
    avg = df.loc["USA", "stock_idx_avg"].dropna()                 # ECB monthly average still there
    assert avg.index.min() == pd.Timestamp("1964-01-01") and avg.iloc[0] == 80.0
    assert df.index.names == ["code", "date"]


def test_stock_index_monthly_with_no_network_is_empty_and_keeps_its_shape(monkeypatch):
    monkeypatch.setattr(rates, "_get", _Server({"": TimeoutError("slow")}))
    with pytest.warns(UserWarning):
        df = rates.stock_index_monthly(["USA", "MEX"])
    assert df.empty and df.index.names == ["code", "date"]
    assert list(df.columns) == ["stock_idx", "stock_idx_avg"]
    assert any("timeout" in m["reason"] for m in df.attrs["missing"])


# ---------------------------------------------------------------------------
# Pink Sheet: stdlib xlsx reader and commodity_prices_monthly
# ---------------------------------------------------------------------------

def _xlsx(sheets: dict[str, list[list]]) -> bytes:
    """A minimal workbook with shared strings, the way the World Bank writes it."""
    strings: list[str] = []

    def sidx(t: str) -> int:
        if t not in strings:
            strings.append(t)
        return strings.index(t)

    def col(j: int) -> str:
        s = ""
        j += 1
        while j:
            j, r = divmod(j - 1, 26)
            s = chr(65 + r) + s
        return s

    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    rns = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
    parts = {}
    for k, (name, rows) in enumerate(sheets.items(), start=1):
        xml_rows = []
        for i, row in enumerate(rows, start=1):
            cells = []
            for j, v in enumerate(row):
                ref = f"{col(j)}{i}"
                if v is None:
                    continue
                if isinstance(v, str):
                    cells.append(f'<c r="{ref}" t="s"><v>{sidx(v)}</v></c>')
                else:
                    cells.append(f'<c r="{ref}"><v>{v}</v></c>')
            xml_rows.append(f'<row r="{i}">' + "".join(cells) + "</row>")
        parts[f"xl/worksheets/sheet{k}.xml"] = (
            f'<?xml version="1.0"?><worksheet {ns}><sheetData>' + "".join(xml_rows)
            + "</sheetData></worksheet>")
    sheet_tags = "".join(f'<sheet name="{n}" sheetId="{k}" r:id="rId{k}"/>'
                         for k, n in enumerate(sheets, start=1))
    parts["xl/workbook.xml"] = f'<?xml version="1.0"?><workbook {ns} {rns}><sheets>{sheet_tags}</sheets></workbook>'
    rels = "".join(
        f'<Relationship Id="rId{k}" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
        f'relationships/worksheet" Target="worksheets/sheet{k}.xml"/>' for k in range(1, len(sheets) + 1))
    parts["xl/_rels/workbook.xml.rels"] = (
        '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/'
        f'package/2006/relationships">{rels}</Relationships>')
    sst = "".join(f"<si><t>{t}</t></si>" for t in strings)
    parts["xl/sharedStrings.xml"] = f'<?xml version="1.0"?><sst {ns}>{sst}</sst>'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, text in parts.items():
            zf.writestr(name, text)
    return buf.getvalue()


def _pink_workbook() -> bytes:
    periods = ["1960M01", "1960M02", "1960M03"]
    prices = [["World Bank Commodity Price Data (The Pink Sheet)"], ["Monthly prices in nominal US dollars"],
              [None], [None],
              [None, "Crude oil, Brent", "Tin", "Platinum", "Unmapped thing"],
              [None, "($/bbl)", "($/mt)", "($/troy oz)", "($/kg)"]]
    prices += [[p, 1.63 + i, 2100.0, "..", 1.0] for i, p in enumerate(periods)]
    indices = [["Monthly indices"], [None], [None], [None, "Energy"], [None, "iTOTAL", "iENERGY"]]
    indices += [[p, 20.0 + i, 10.0 + i] for i, p in enumerate(periods)]
    return _xlsx({"Monthly Prices": prices, "Monthly Indices": indices})


def test_the_stdlib_reader_reads_shared_strings_and_absolute_columns():
    grid = pk._read_xlsx_sheet(_pink_workbook(), "Monthly Prices")
    assert grid[4][1] == "Crude oil, Brent"
    assert grid[6][0] == "1960M01" and grid[6][1] == pytest.approx(1.63)
    assert grid[2] == [None] * len(grid[2])
    with pytest.raises(KeyError):
        pk._read_xlsx_sheet(_pink_workbook(), "No such sheet")


def test_the_stdlib_reader_matches_the_committed_openpyxl_fixture():
    """The mock workbook in tests/data was written by openpyxl (inline strings)."""
    path = Path(__file__).resolve().parent.parent / "data" / "commodities" / "cmo_historical_mock.xlsx"
    out = pk.fetch_prices(workbook_bytes=path.read_bytes())
    assert {"oil_brent_m", "copper_m", "gold_m", "platinum_m", "tin_m"} <= set(out["variable"])
    assert out["value"].notna().all()


def test_platinum_is_no_longer_swallowed_by_tin():
    out = pk.fetch_prices(workbook_bytes=_pink_workbook())
    assert set(out["variable"]) == {"oil_brent_m", "tin_m"}       # platinum is all '..'
    assert (out.loc[out["variable"] == "tin_m", "value"] == 2100.0).all()


def test_commodity_prices_monthly_is_a_wide_wld_panel_with_units(monkeypatch):
    page = (b'<a href="https://thedocs.worldbank.org/en/doc/X-0050012026/related/'
            b'CMO-Historical-Data-Monthly.xlsx">x</a>')
    book = _pink_workbook()
    monkeypatch.setattr(pk, "cached_get", lambda url, **kw: page if url == pk._LANDING else book)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = rates.commodity_prices_monthly()
    assert df.index.names == ["code", "date"]
    assert set(df.index.get_level_values("code")) == {"WLD"}
    assert list(df.index.get_level_values("date")) == list(pd.to_datetime(["1960-01-01", "1960-02-01", "1960-03-01"]))
    assert {"commodity_brent", "commodity_tin", "commodity_index_total",
            "commodity_index_energy"} <= set(df.columns)
    assert all(c.startswith("commodity_") for c in df.columns)
    assert df.loc[("WLD", pd.Timestamp("1960-02-01")), "commodity_brent"] == pytest.approx(2.63)
    units = {m["variable"]: m["units"] for m in df.attrs["meta"]}
    assert units["commodity_brent"] == "$/bbl"
    assert units["commodity_index_total"] == "index, 2010=100"
    assert df.attrs["missing"] == () and df.attrs["complete"] is True


def test_commodity_prices_monthly_survives_a_failed_download(monkeypatch):
    def _fail(url, **kw):
        raise OSError("offline")
    monkeypatch.setattr(pk, "cached_get", _fail)
    with pytest.warns(UserWarning, match="could not be downloaded"):
        df = rates.commodity_prices_monthly()
    assert df.empty and df.index.names == ["code", "date"]
    assert df.attrs["missing"][0]["code"] == "WLD"
    assert df.attrs["complete"] is False


def test_the_empty_commodity_frame_has_the_same_48_columns_as_a_full_one(monkeypatch):
    def _fail(url, **kw):
        raise OSError("offline")
    monkeypatch.setattr(pk, "cached_get", _fail)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = rates.commodity_prices_monthly()
    assert df.shape == (0, 48)
    assert len(set(df.columns)) == 48
    assert sum(c.startswith("commodity_index_") for c in df.columns) == 16
    assert all(df[c].dtype == float for c in df.columns)


# ---------------------------------------------------------------------------
# import safety
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", ["rates.py", "wb_pink_sheet.py", "yahoo.py"])
def test_no_heavy_or_unsafe_imports_at_module_scope(module):
    src = Path(rates.__file__).with_name(module).read_text()
    tree = ast.parse(src)
    banned = {"requests", "ssl", "sqlite3", "statsmodels", "openpyxl", "pyarrow", "yfinance"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert not {a.name.split(".")[0] for a in node.names} & banned
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in banned
