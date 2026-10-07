"""Offline tests for puremacro.fetch.ilostat (ILOSTAT one-call labour panel).

Every test patches the module's HTTP seam ``_get`` with tiny SDMX-CSV bodies
shaped like the live service's (column order, ``2025-Q3``/``2025-M09``
periods, ``X##`` aggregates, ``OBS_STATUS`` flags); none touches the network.
"""
from __future__ import annotations

import datetime as dt
import http.client
import urllib.error
import warnings
import zlib

import numpy as np
import pandas as pd
import pytest

import puremacro.fetch.ilostat as ilo


_TAIL = "UNIT_MEASURE_TYPE,UNIT_MEASURE,UNIT_MULT,SOURCE,NOTE_SOURCE,NOTE_INDICATOR,NOTE_CLASSIF,DECIMALS,UPPER_BOUND,LOWER_BOUND"


def _csv(flow: str, dims: list[str], rows: list[tuple]) -> bytes:
    """rows: (area, freq, measure, *dims, period, value, status, unit_mult, source)."""
    head = ",".join(["DATAFLOW", "REF_AREA", "FREQ", "MEASURE", *dims,
                     "TIME_PERIOD", "OBS_VALUE", "OBS_STATUS"]) + "," + _TAIL
    lines = [head]
    for r in rows:
        *key, period, value, status, mult, source = r
        lines.append(",".join([f"ILO:{flow}(1.0)", *key, period, str(value), status,
                               "NB", "PS", mult, source, "", "", "", "1", "", ""]))
    return ("\r\n".join(lines) + "\r\n").encode()


_CATALOGUE = b"""<?xml version="1.0"?><message:Structure>
<structure:Dataflow id="DF_EMP_TEMP_SEX_AGE_NB" agencyID="ILO" version="1.0">
<common:Annotations><common:Annotation>
<common:AnnotationTitle>03/10/2026 07:10:53</common:AnnotationTitle>
<common:AnnotationType>LAST_UPDATE</common:AnnotationType>
</common:Annotation></common:Annotations></structure:Dataflow>
<structure:Dataflow id="DF_UNE_DEAP_SEX_AGE_RT" agencyID="ILO" version="1.0">
<common:Annotations><common:Annotation>
<common:AnnotationTitle>24/09/2026 08:00:00</common:AnnotationTitle>
<common:AnnotationType>LAST_UPDATE</common:AnnotationType>
</common:Annotation></common:Annotations></structure:Dataflow>
</message:Structure>"""

_AGE = ["SEX", "AGE"]
_SA = ("SEX_T", "AGE_YTHADULT_YGE15")
_LFS = "LFS - Encuesta Nacional"


def _bodies(this_year: int | None = None) -> dict[str, bytes]:
    y = this_year or dt.datetime.now(dt.timezone.utc).year
    return {
        "DF_EMP_TEMP_SEX_AGE_NB": _csv("DF_EMP_TEMP_SEX_AGE_NB", _AGE, [
            ("MEX", "A", "EMP_TEMP_NB", *_SA, "2023", 58000.5, "", "3", _LFS),
            ("MEX", "A", "EMP_TEMP_NB", *_SA, "2024", 59000.25, "B", "3", _LFS),
            ("USA", "A", "EMP_TEMP_NB", *_SA, "1948", 58343.0, "", "3", "LFS - CPS"),
            ("X01", "A", "EMP_TEMP_NB", *_SA, "2024", 3.4e6, "", "3", "ILO"),
            ("XA1", "A", "EMP_TEMP_NB", *_SA, "2024", 1.0e5, "", "3", "ILO"),
            ("MEX", "Q", "EMP_TEMP_NB", *_SA, "2024-Q3", 59500.0, "", "3", _LFS),
            ("MEX", "Q", "EMP_TEMP_NB", *_SA, "2024-Q4", 59600.0, "", "3", _LFS),
            ("ESP", "M", "EMP_TEMP_NB", *_SA, "2024-M09", 21000.0, "", "3", "LFS-ADJ"),
        ]),
        "DF_UNE_DEAP_SEX_AGE_RT": _csv("DF_UNE_DEAP_SEX_AGE_RT", _AGE, [
            ("MEX", "A", "UNE_DEAP_RT", *_SA, "2023", 2.8, "", "0", _LFS),
            ("MEX", "A", "UNE_DEAP_RT", *_SA, "2024", 2.7, "", "0", _LFS),
            ("X05", "A", "UNE_DEAP_RT", *_SA, "2024", 5.0, "", "0", "ILO"),
            ("MEX", "Q", "UNE_DEAP_RT", *_SA, "2024-Q3", 2.9, "", "0", _LFS),
            ("ESP", "M", "UNE_DEAP_RT", *_SA, "2024-M09", 11.2, "", "0", "LFS-ADJ"),
            ("ESP", "M", "UNE_DEAP_RT", *_SA, "2024-M10", 11.1, "", "0", "LFS-ADJ"),
        ]),
        "DF_EMP_TEMP_SEX_STE_NB": _csv("DF_EMP_TEMP_SEX_STE_NB", ["SEX", "STE"], [
            ("MEX", "A", "EMP_TEMP_NB", "SEX_T", "STE_AGGREGATE_EES", "2024", 40000.0, "", "3", _LFS),
            ("MEX", "A", "EMP_TEMP_NB", "SEX_T", "STE_AGGREGATE_SLF", "2024", 18295.435, "", "3", _LFS),
            ("MEX", "A", "EMP_TEMP_NB", "SEX_T", "STE_ICSE93_3", "2024", 12907.659, "", "3", _LFS),
        ]),
        "DF_HOW_TEMP_SEX_NB": _csv("DF_HOW_TEMP_SEX_NB", ["SEX"], [
            ("MEX", "A", "HOW_TEMP_NB", "SEX_T", "2024", 41.7, "", "0", _LFS),
        ]),
        "DF_LAP_2GDP_NOC_RT": _csv("DF_LAP_2GDP_NOC_RT", [], [
            ("MEX", "A", "LAP_2GDP_RT", str(y - 2), 37.8, "I", "0", "ILO - Modelled Estimates"),
            ("MEX", "A", "LAP_2GDP_RT", str(y - 1), 37.9, "M", "0", "ILO - Modelled Estimates"),
            ("MEX", "A", "LAP_2GDP_RT", str(y), 37.95, "M", "0", "ILO - Modelled Estimates"),
            ("MEX", "A", "LAP_2GDP_RT", str(y + 1), 38.0, "M", "0", "ILO - Modelled Estimates"),
        ]),
        "DF_EMP_2EMP_SEX_AGE_NB": _csv("DF_EMP_2EMP_SEX_AGE_NB", _AGE, [
            ("MEX", "A", "EMP_2EMP_NB", *_SA, str(y - 1), 60000.0, "", "", ""),
            ("MEX", "A", "EMP_2EMP_NB", *_SA, str(y + 1), 61000.0, "", "", ""),
        ]),
    }


class _Server:
    """Answers by dataflow id in the URL; records every URL asked for."""

    def __init__(self, bodies: dict[str, bytes], fail: dict[str, BaseException] | None = None):
        self.bodies, self.fail, self.urls = bodies, fail or {}, []

    def __call__(self, url, *, timeout=None, pause=None, refresh=False):
        self.urls.append(url)
        if url.endswith("/dataflow/ILO"):
            return _CATALOGUE
        flow = url.split("ILO,")[1].split(",")[0]
        if flow in self.fail:
            raise self.fail[flow]
        if flow not in self.bodies:
            raise urllib.error.HTTPError(url, 404, "NoRecordsFound", {}, None)
        return self.bodies[flow]


@pytest.fixture
def server(monkeypatch):
    srv = _Server(_bodies())
    monkeypatch.setattr(ilo, "_get", srv)
    return srv


def test_annual_panel_is_indexed_by_code_and_year_start_with_float_columns(server):
    p = ilo.ilostat_panel(freq="A", variables=["emp", "urate"])
    assert p.index.names == ["code", "date"]
    assert list(p.columns) == ["emp", "urate"]
    assert all(p.dtypes == np.float64)
    assert pd.api.types.is_datetime64_any_dtype(p.index.get_level_values("date"))
    assert p.loc[("MEX", pd.Timestamp("2024-01-01")), "emp"] == pytest.approx(59000.25)
    assert p.loc[("USA", pd.Timestamp("1948-01-01")), "emp"] == pytest.approx(58343.0)
    assert np.isnan(p.loc[("USA", pd.Timestamp("1948-01-01")), "urate"])


def test_values_stay_in_published_units_thousands_and_percent(server):
    p = ilo.ilostat_panel(["MEX"], freq="A", variables=["emp", "urate"])
    meta = {(m["variable"], m["code"]): m for m in p.attrs["meta"]}
    assert meta[("emp", "MEX")]["units"] == "thousands of persons"
    assert meta[("emp", "MEX")]["unit_mult"] == 3
    assert meta[("urate", "MEX")]["units"].startswith("percent")
    assert p.loc[("MEX", pd.Timestamp("2024-01-01")), "urate"] == pytest.approx(2.7)


def test_quarterly_and_monthly_dates_are_period_starts(server):
    q = ilo.ilostat_panel(freq="Q", variables=["emp", "urate"])
    assert ("MEX", pd.Timestamp("2024-07-01")) in q.index
    assert ("MEX", pd.Timestamp("2024-10-01")) in q.index
    m = ilo.ilostat_panel(freq="M", variables=["emp", "urate"])
    assert m.loc[("ESP", pd.Timestamp("2024-09-01")), "urate"] == pytest.approx(11.2)
    assert m.loc[("ESP", pd.Timestamp("2024-10-01")), "urate"] == pytest.approx(11.1)


def test_frequency_and_measure_are_pinned_in_the_key(server):
    ilo.ilostat_panel(["MEX", "USA"], freq="Q", variables=["emp"], start=1995)
    data_urls = [u for u in server.urls if "/data/" in u]
    assert data_urls == ["https://sdmx.ilo.org/rest/data/ILO,DF_EMP_TEMP_SEX_AGE_NB,1.0/"
                         "MEX+USA.Q.EMP_TEMP_NB.SEX_T.AGE_YTHADULT_YGE15?format=csv&startPeriod=1995"]


def test_ilo_aggregates_are_dropped_and_reported(server):
    p = ilo.ilostat_panel(freq="A", variables=["emp", "urate"])
    codes = set(p.index.get_level_values("code"))
    assert codes == {"MEX", "USA"}
    assert set(p.attrs["aggregates_dropped"]) == {"X01", "X05", "XA1"}


def test_status_variables_sharing_a_flow_travel_in_one_request(server):
    p = ilo.ilostat_panel(freq="A", variables=["emp_ees", "emp_self"])
    data_urls = [u for u in server.urls if "/data/" in u]
    assert len(data_urls) == 1
    assert ".EMP_TEMP_NB.SEX_T.STE_AGGREGATE_EES+STE_AGGREGATE_SLF?" in data_urls[0]
    row = p.loc[("MEX", pd.Timestamp("2024-01-01"))]
    assert row["emp_ees"] == pytest.approx(40000.0)
    assert row["emp_self"] == pytest.approx(18295.435)


def test_hours_and_three_dimension_labour_share_keys(server):
    ilo.ilostat_panel(["MEX"], freq="A", variables=["hours", "labour_share"])
    data_urls = [u for u in server.urls if "/data/" in u]
    assert any("/MEX.A.HOW_TEMP_NB.SEX_T?" in u for u in data_urls)
    assert any("/MEX.A.LAP_2GDP_RT?" in u for u in data_urls)


def test_breaks_and_status_flags_are_kept_in_meta(server):
    p = ilo.ilostat_panel(["MEX"], freq="A", variables=["emp"])
    (m,) = [m for m in p.attrs["meta"] if m["code"] == "MEX"]
    assert m["breaks"] == ("2024",)
    assert m["status"] == {"B": 1}
    assert (m["first"], m["last"], m["n"]) == ("2023", "2024", 2)
    assert m["surveys"] == (_LFS,)
    assert m["sa"] == "NSA"
    assert m["flow"] == "DF_EMP_TEMP_SEX_AGE_NB"
    assert m["key"] == "MEX.A.EMP_TEMP_NB.SEX_T.AGE_YTHADULT_YGE15"


def test_eurostat_adjusted_monthly_series_are_marked_seasonally_adjusted(server):
    p = ilo.ilostat_panel(["ESP"], freq="M", variables=["urate"])
    (m,) = p.attrs["meta"]
    assert m["sa"] == "SA"


def test_source_carries_the_catalogue_last_update_per_flow(server):
    p = ilo.ilostat_panel(freq="A", variables=["emp", "urate"])
    assert "ILO,DF_EMP_TEMP_SEX_AGE_NB,1.0 (LAST_UPDATE 2026-10-03T07:10:53)" in p.attrs["source"]
    assert "ILO,DF_UNE_DEAP_SEX_AGE_RT,1.0 (LAST_UPDATE 2026-09-24T08:00:00)" in p.attrs["source"]
    assert {f["flow"]: f["last_update"] for f in p.attrs["flows"]}[
        "DF_EMP_TEMP_SEX_AGE_NB"] == "2026-10-03T07:10:53"
    assert p.attrs["fetched_at"].endswith("+00:00")


def test_projections_of_modelled_flows_are_cut_back_from_the_last_published_year(server):
    # The fixture's modelled flows end at y + 1: labour_share publishes one
    # projected year, the *_model twins two.
    y = dt.datetime.now(dt.timezone.utc).year
    p = ilo.ilostat_panel(["MEX"], freq="A", variables=["labour_share", "emp_model"])
    assert p["labour_share"].dropna().index.get_level_values("date").year.max() == y
    assert p["emp_model"].dropna().index.get_level_values("date").year.max() == y - 1
    cut = {m["variable"]: m["trimmed_after"] for m in p.attrs["meta"]}
    assert cut == {"labour_share": y, "emp_model": y - 1}


def test_the_projection_cut_depends_on_the_data_not_on_the_build_date(monkeypatch):
    rows = [("MEX", "A", "EMP_2EMP_NB", *_SA, str(t), 60000.0 + t, "", "3",
             "ILO - Modelled Estimates") for t in (2024, 2025, 2026, 2027)]
    srv = _Server({"DF_EMP_2EMP_SEX_AGE_NB": _csv("DF_EMP_2EMP_SEX_AGE_NB", _AGE, rows)})
    monkeypatch.setattr(ilo, "_get", srv)
    p = ilo.ilostat_panel(["MEX"], freq="A", variables=["emp_model"])
    assert list(p.index.get_level_values("date").year) == [2024, 2025]
    assert p.attrs["meta"][0]["trimmed_after"] == 2025


def test_survey_flows_are_never_trimmed(server):
    p = ilo.ilostat_panel(["MEX"], freq="A", variables=["emp"])
    assert p.attrs["meta"][0]["trimmed_after"] is None


def test_an_explicit_end_keeps_the_projections(server):
    y = dt.datetime.now(dt.timezone.utc).year
    p = ilo.ilostat_panel(["MEX"], freq="A", variables=["labour_share"], end=y + 5)
    assert p["labour_share"].dropna().index.get_level_values("date").year.max() == y + 1
    assert p.attrs["meta"][0]["trimmed_after"] is None


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_modelled_true_adds_the_model_twins(server):
    p = ilo.ilostat_panel(freq="A", variables=["emp"], modelled=True)
    assert list(p.columns) == ["emp", *ilo.MODEL_VARS]


def test_start_and_end_are_applied_exactly_on_dates(server):
    q = ilo.ilostat_panel(freq="Q", variables=["emp"], start="2024Q4")
    assert list(q.index.get_level_values("date").unique()) == [pd.Timestamp("2024-10-01")]
    assert any("startPeriod=2024" in u for u in server.urls)


def test_provider_failure_returns_empty_frame_with_missing_and_does_not_raise(monkeypatch):
    srv = _Server({}, fail={"DF_EMP_TEMP_SEX_AGE_NB": urllib.error.HTTPError(
        "u", 503, "Service Unavailable", {}, None)})
    monkeypatch.setattr(ilo, "_get", srv)
    with pytest.warns(UserWarning, match="HTTP 503"):
        p = ilo.ilostat_panel(freq="A", variables=["emp"])
    assert p.empty
    assert p.index.names == ["code", "date"]
    assert list(p.columns) == ["emp"]
    assert set(p.attrs) >= {"meta", "source", "fetched_at", "missing", "flows"}
    assert p.attrs["missing"] == ({"variable": "emp", "flow": "DF_EMP_TEMP_SEX_AGE_NB",
                                   "reason": "HTTP 503"},)
    # one retry on a 5xx
    assert sum("/data/" in u for u in srv.urls) == 2


def test_too_many_requests_is_reported_and_not_retried(monkeypatch):
    srv = _Server({}, fail={"DF_EMP_TEMP_SEX_AGE_NB": urllib.error.HTTPError(
        "u", 429, "Too Many Requests", {}, None)})
    monkeypatch.setattr(ilo, "_get", srv)
    with pytest.warns(UserWarning, match="HTTP 429"):
        p = ilo.ilostat_panel(freq="A", variables=["emp"])
    assert p.empty and p.index.names == ["code", "date"]
    assert p.attrs["missing"][0]["reason"] == "HTTP 429"
    assert sum("/data/" in u for u in srv.urls) == 1


@pytest.mark.parametrize("exc", [
    http.client.IncompleteRead(b"", 10),
    EOFError("Compressed file ended before the end-of-stream marker was reached"),
    zlib.error("Error -3 while decompressing data"),
], ids=["incomplete-read", "truncated-gzip", "zlib"])
def test_a_truncated_or_corrupt_transfer_returns_an_empty_frame_without_raising(monkeypatch, exc):
    srv = _Server({}, fail={"DF_EMP_TEMP_SEX_AGE_NB": exc})
    monkeypatch.setattr(ilo, "_get", srv)
    with pytest.warns(UserWarning, match="truncated or corrupt"):
        p = ilo.ilostat_panel(freq="A", variables=["emp"])
    assert p.empty and list(p.columns) == ["emp"]
    assert p.index.names == ["code", "date"]
    assert p.attrs["missing"][0]["variable"] == "emp"
    assert "truncated or corrupt" in p.attrs["missing"][0]["reason"]
    assert sum("/data/" in u for u in srv.urls) == 2      # retried once


def test_a_truncated_catalogue_only_blanks_last_update(monkeypatch):
    srv = _Server(_bodies())

    def get(url, **kw):
        if url.endswith("/dataflow/ILO"):
            raise http.client.IncompleteRead(b"<?xml", 7_000_000)
        return srv(url, **kw)

    monkeypatch.setattr(ilo, "_get", get)
    with pytest.warns(UserWarning, match="catalogue unavailable"):
        p = ilo.ilostat_panel(freq="A", variables=["emp"])
    assert p["emp"].notna().any()
    assert "LAST_UPDATE unknown" in p.attrs["source"]


def test_a_cached_unreadable_body_is_fetched_again_with_refresh(monkeypatch):
    good = _bodies()["DF_EMP_TEMP_SEX_AGE_NB"]
    calls = []

    def get(url, *, timeout=None, pause=None, refresh=False):
        if url.endswith("/dataflow/ILO"):
            return _CATALOGUE
        calls.append(refresh)
        return good if refresh else b"<html><body>Maintenance</body></html>"

    monkeypatch.setattr(ilo, "_get", get)
    p = ilo.ilostat_panel(["MEX"], freq="A", variables=["emp"])
    assert calls == [False, True]
    assert p.loc[("MEX", pd.Timestamp("2024-01-01")), "emp"] == 59000.25
    assert p.attrs["missing"] == ()


def test_a_persistently_unreadable_body_is_missing_after_one_refresh(monkeypatch):
    calls = []

    def get(url, *, timeout=None, pause=None, refresh=False):
        if url.endswith("/dataflow/ILO"):
            return _CATALOGUE
        calls.append(refresh)
        return b""

    monkeypatch.setattr(ilo, "_get", get)
    with pytest.warns(UserWarning, match="unreadable response"):
        p = ilo.ilostat_panel(freq="A", variables=["emp"])
    assert calls == [False, True]
    assert p.empty and p.attrs["missing"][0]["reason"].startswith("unreadable response")


def test_duplicated_observations_warn_keep_the_first_and_are_recorded(monkeypatch):
    rows = [("MEX", "A", "EMP_TEMP_NB", *_SA, "2024", 59000.0, "", "3", _LFS),
            ("MEX", "A", "EMP_TEMP_NB", *_SA, "2024", 61000.0, "", "3", "LFS - other vintage")]
    srv = _Server({"DF_EMP_TEMP_SEX_AGE_NB": _csv("DF_EMP_TEMP_SEX_AGE_NB", _AGE, rows)})
    monkeypatch.setattr(ilo, "_get", srv)
    with pytest.warns(UserWarning, match="duplicated"):
        p = ilo.ilostat_panel(["MEX"], freq="A", variables=["emp"])
    assert p.loc[("MEX", pd.Timestamp("2024-01-01")), "emp"] == 59000.0
    assert p.attrs["duplicates"] == (("MEX", "2024", "emp"),)


def test_a_clean_response_records_no_duplicates(server):
    p = ilo.ilostat_panel(freq="A", variables=["emp", "urate"])
    assert p.attrs["duplicates"] == ()


def test_a_timeout_on_one_flow_keeps_the_other_variables(monkeypatch):
    srv = _Server(_bodies(), fail={"DF_UNE_DEAP_SEX_AGE_RT": TimeoutError("timed out")})
    monkeypatch.setattr(ilo, "_get", srv)
    with pytest.warns(UserWarning, match="timeout"):
        p = ilo.ilostat_panel(freq="A", variables=["emp", "urate"])
    assert p["emp"].notna().any()
    assert p["urate"].isna().all()
    assert [m["variable"] for m in p.attrs["missing"]] == ["urate"]


def test_unknown_codes_are_listed_as_missing(server):
    p = ilo.ilostat_panel(["MEX", "ZZZ"], freq="A", variables=["emp"])
    assert {"code": "ZZZ", "reason": "no observations for any variable"} in p.attrs["missing"]


def test_a_failed_catalogue_only_blanks_last_update(monkeypatch):
    srv = _Server(_bodies())

    def get(url, **kw):
        if url.endswith("/dataflow/ILO"):
            raise OSError("down")
        return srv(url, **kw)

    monkeypatch.setattr(ilo, "_get", get)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = ilo.ilostat_panel(freq="A", variables=["emp"])
    assert p["emp"].notna().any()
    assert "LAST_UPDATE unknown" in p.attrs["source"]


def test_unknown_variable_is_a_caller_error_listing_valid_names(server):
    with pytest.raises(ValueError, match="valid"):
        ilo.ilostat_panel(freq="A", variables=["gdp"])


def test_annual_only_variables_are_rejected_at_quarterly_frequency(server):
    with pytest.raises(ValueError, match="not published at freq='Q'"):
        ilo.ilostat_panel(freq="Q", variables=["labour_share"])
    with pytest.raises(ValueError, match="annual only"):
        ilo.ilostat_panel(freq="Q", modelled=True)


def test_modelled_true_is_rejected_at_quarterly_or_monthly_frequency_even_with_variables(server):
    for f in ("Q", "M"):
        with pytest.raises(ValueError, match="annual only"):
            ilo.ilostat_panel(freq=f, variables=["emp"], modelled=True)
    assert server.urls == []


def test_codes_is_the_only_positional_argument():
    import inspect
    params = inspect.signature(ilo.ilostat_panel).parameters
    assert list(params)[0] == "codes"
    assert params["codes"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in list(params.values())[1:])
    assert "variables" in params and "vars" not in params


def test_bad_frequency_is_a_caller_error(server):
    with pytest.raises(ValueError, match="freq must be one of"):
        ilo.ilostat_panel(freq="W")
    assert server.urls == []


def test_quarterly_default_variables_leave_out_annual_only_ones():
    names = ilo._check_vars(None, "Q", False)
    assert "labour_share" not in names
    assert names[:2] == ["emp", "une"]


def test_period_parser_handles_annual_quarterly_and_monthly_codes():
    s = pd.Series(["2025", "2025-Q3", "2025-M09", "1948-Q1"])
    got = ilo._period_start(s)
    assert list(got) == [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-07-01"),
                         pd.Timestamp("2025-09-01"), pd.Timestamp("1948-01-01")]


def test_aggregate_rule_spares_iso_and_ilo_territory_codes():
    assert ilo.is_ilo_aggregate("X01") and ilo.is_ilo_aggregate("XA1")
    assert not any(ilo.is_ilo_aggregate(c) for c in ("MEX", "KOS", "CHA", "ANT", "XKX"))


def test_meta_reader_returns_one_row_per_country_and_variable(server):
    p = ilo.ilostat_panel(freq="A", variables=["emp", "urate"])
    m = ilo.ilostat_meta(p)
    assert m.index.names == ["variable", "code"]
    assert ("emp", "USA") in m.index and ("urate", "MEX") in m.index


@pytest.mark.network
def test_live_three_country_quarterly_urate():
    p = ilo.ilostat_panel(["MEX", "USA", "ESP"], freq="Q", variables=["urate"], pause=3.0)
    assert set(p.index.get_level_values("code")) == {"MEX", "USA", "ESP"}
    assert p.loc["USA"].index.min() <= pd.Timestamp("1948-01-01")


def test_projection_horizons_match_the_november_2025_edition():
    horizon = {v: s["projected"] for v, s in ilo.ILOSTAT_VARS.items() if s["projected"]}
    assert horizon == {"labour_share": 1, "emp_model": 2, "urate_model": 2, "lf_model": 2,
                       "wap_model": 5, "hours_model": 2}
    assert ilo.ILOSTAT_VARS["emp_self_model"]["projected"] == 0
