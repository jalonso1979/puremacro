"""Offline tests for :mod:`puremacro.fetch.oecd_qna_extras`.

Every OECD request is answered by a synthetic ``csvfile`` frame built here,
with the columns the real flows return (checked against live responses of
6 Oct 2026): the module-level ``_get`` and ``_availability`` are the names
patched, never the functions under test.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from puremacro.fetch import oecd_qna_extras as mod
from puremacro.fetch.oecd_qna_extras import (LFS_INDICATORS, OECD_VACANCIES,
                                             QNA_POPULATION, oecd_lfs_panel,
                                             oecd_vacancies, qna_extras_meta,
                                             qna_population, qna_sector_gfcf)

_Q = [f"{y}-Q{q}" for y in (2019, 2020) for q in (1, 2, 3, 4)]
_M = [f"2020-{m:02d}" for m in range(1, 7)]


def _ok(rows: list[dict]) -> pd.DataFrame:
    out = pd.DataFrame(rows)
    out.attrs["status"] = "ok"
    return out


def _fail(status: str) -> pd.DataFrame:
    out = pd.DataFrame()
    out.attrs["status"] = status
    return out


class _Recorder:
    """A fake ``_get`` that answers by flow and remembers every key it saw."""

    def __init__(self, answers: dict[str, object]) -> None:
        self.answers = answers
        self.calls: list[tuple[str, str]] = []

    def __call__(self, flow, key, *, start_period=None, refresh=False):
        self.calls.append((flow, key))
        for tag, answer in self.answers.items():
            if tag in flow:
                return answer(key) if callable(answer) else answer
        return _fail("HTTP 404")


def _namain(code, sector, adj, price_base, value, *, txn="P51G", mult=6,
            periods=_Q, ref_year=2017):
    return [{"DATAFLOW": "OECD.SDD.NAD:DSD_NAMAIN1@DF_QNA_EXPENDITURE_GFCF_SECTOR(1.1)",
             "FREQ": "Q", "ADJUSTMENT": adj, "REF_AREA": code, "SECTOR": sector,
             "COUNTERPART_SECTOR": "S1", "TRANSACTION": txn, "INSTR_ASSET": "_Z",
             "ACTIVITY": "_T", "EXPENDITURE": "_Z", "UNIT_MEASURE": "XDC",
             "PRICE_BASE": price_base, "TRANSFORMATION": "N",
             "TABLE_IDENTIFIER": "T0102", "TIME_PERIOD": t,
             "OBS_VALUE": value + i, "REF_YEAR_PRICE": ref_year if price_base != "V" else np.nan,
             "UNIT_MULT": mult, "CURRENCY": "USD"}
            for i, t in enumerate(periods)]


def _qsa(code, sector, adj, value, *, entry="D", transformation="N", periods=_Q):
    return [{"DATAFLOW": "OECD.SDD.NAD:DSD_NASEC1@DF_QSA(1.1)", "FREQ": "Q",
             "ADJUSTMENT": adj, "REF_AREA": code, "SECTOR": sector,
             "COUNTERPART_SECTOR": "S1", "ACCOUNTING_ENTRY": entry,
             "TRANSACTION": "P51G", "INSTR_ASSET": "_Z", "EXPENDITURE": "_Z",
             "UNIT_MEASURE": "XDC", "VALUATION": "S", "PRICE_BASE": "V",
             "TRANSFORMATION": transformation, "TABLE_IDENTIFIER": "T0801",
             "TIME_PERIOD": t, "OBS_VALUE": value, "UNIT_MULT": 6, "CURRENCY": "MXN"}
            for t in periods]


def _sector_answers():
    flow1 = (_namain("USA", "S13", "Y", "V", 100.0) + _namain("USA", "S1W", "Y", "V", 400.0)
             + _namain("USA", "S13", "Y", "L", 90.0) + _namain("USA", "S1W", "Y", "L", 380.0)
             + _namain("EA20", "S13", "Y", "V", 999.0))
    qsa = (_qsa("MEX", "S13", "N", 50.0) + _qsa("MEX", "S1", "N", 300.0)
           # the annualised edition (4x) and the credit entry must be ignored
           + _qsa("MEX", "S13", "N", 200.0, transformation="LA")
           + _qsa("MEX", "S13", "N", 7.0, entry="C")
           # Spain: listed by DF_QSA but every row is empty
           + [{**_qsa("ESP", "S13", "N", 1.0)[0], "TIME_PERIOD": np.nan,
               "OBS_VALUE": np.nan}])
    return {"GFCF_SECTOR": _ok(flow1), "DF_QSA": _ok(qsa)}


# --------------------------------------------------------------------------
# qna_sector_gfcf
# --------------------------------------------------------------------------

def test_sector_gfcf_returns_a_code_date_panel_in_millions_at_quarter_start(monkeypatch):
    monkeypatch.setattr(mod, "_get", _Recorder(_sector_answers()))
    out = qna_sector_gfcf(["USA", "MEX", "ESP"], sa="prefer")
    assert out.index.names == ["code", "date"]
    assert list(out.columns) == ["inv_gov", "inv_priv", "inv_gov_real", "inv_priv_real"]
    assert all(out.dtypes == float)
    dates = out.index.get_level_values("date")
    assert (dates.day == 1).all() and set(dates.month) <= {1, 4, 7, 10}
    assert out.loc[("USA", pd.Timestamp("2019-01-01")), "inv_gov"] == 100.0
    assert out.loc[("USA", pd.Timestamp("2019-01-01")), "inv_priv_real"] == 380.0
    assert out.attrs["source"].count("DF_") == 2


def test_sector_gfcf_falls_back_to_sector_accounts_and_derives_private_investment(monkeypatch):
    monkeypatch.setattr(mod, "_get", _Recorder(_sector_answers()))
    out = qna_sector_gfcf(["USA", "MEX", "ESP"], sa="prefer")
    mex = out.loc["MEX"]
    # S13 with entry D and transformation N only: neither LA (200) nor C (7).
    assert (mex["inv_gov"] == 50.0).all()
    assert (mex["inv_priv"] == 250.0).all()          # S1 300 less S13 50
    assert mex["inv_gov_real"].isna().all()           # DF_QSA has no volumes
    meta = qna_extras_meta(out).set_index(["code", "variable"])
    assert meta.loc[("MEX", "inv_gov"), "source"] == "OECD.SDD.NAD,DSD_NASEC1@DF_QSA"
    assert meta.loc[("USA", "inv_gov"), "source"].endswith("DF_QNA_EXPENDITURE_GFCF_SECTOR")
    assert meta.loc[("MEX", "inv_gov"), "sa"] == "none"     # published NSA, sa="prefer"
    assert meta.loc[("USA", "inv_gov"), "sa"] == "oecd"


def test_sector_gfcf_keeps_published_sa_for_government_and_pairs_editions_for_private(monkeypatch):
    # GBR-like: S13 published adjusted and raw, S1 raw only.
    qsa = (_qsa("GBR", "S13", "Y", 60.0) + _qsa("GBR", "S13", "N", 50.0)
           + _qsa("GBR", "S1", "N", 300.0))
    monkeypatch.setattr(mod, "_get", _Recorder({"DF_QSA": _ok(qsa)}))
    out = qna_sector_gfcf(["GBR"], sa="prefer")
    assert (out.loc["GBR", "inv_gov"] == 60.0).all()      # the OECD's own SA
    assert (out.loc["GBR", "inv_priv"] == 250.0).all()    # raw S1 less raw S13
    meta = qna_extras_meta(out).set_index(["code", "variable"])
    assert meta.loc[("GBR", "inv_gov"), "sa"] == "oecd"
    assert meta.loc[("GBR", "inv_priv"), "sa"] == "none"


def test_sector_gfcf_asks_the_sector_accounts_only_for_countries_the_first_flow_lacks(monkeypatch):
    rec = _Recorder(_sector_answers())
    monkeypatch.setattr(mod, "_get", rec)
    qna_sector_gfcf(["USA", "MEX", "ESP"], sa="prefer")
    qsa_keys = [k for f, k in rec.calls if "DF_QSA" in f]
    assert qsa_keys == ["Q..MEX+ESP.S1+S13..D.P51G......N."]
    # 14 dimensions in DSD_NASEC1, 13 in DSD_NAMAIN1
    assert qsa_keys[0].count(".") == 13
    assert [k for f, k in rec.calls if "GFCF_SECTOR" in f][0].count(".") == 12


def test_sector_gfcf_drops_aggregates_and_lists_empty_countries_as_missing(monkeypatch):
    monkeypatch.setattr(mod, "_get", _Recorder(_sector_answers()))
    out = qna_sector_gfcf(None, sa="prefer")
    codes = set(out.index.get_level_values("code"))
    assert codes == {"USA", "MEX"}                      # EA20 dropped, ESP empty
    out = qna_sector_gfcf(["USA", "MEX", "ESP"], sa="prefer")
    missing = {(m["code"], m["variable"]) for m in out.attrs["missing"]}
    assert ("ESP", "inv_gov") in missing
    assert ("MEX", "inv_gov_real") in missing
    assert ("USA", "inv_gov") not in missing


def test_sector_gfcf_x13_adjusts_only_the_series_published_unadjusted(monkeypatch):
    monkeypatch.setattr(mod, "_get", _Recorder(_sector_answers()))
    seen = {}

    def fake_adjust(tidy):
        nsa = tidy["ADJUSTMENT"] != "Y"
        seen["codes"] = set(tidy.loc[nsa, "code"])
        tidy = tidy.copy()
        tidy.loc[nsa, "ADJUSTMENT"] = "X"
        return tidy, {"MEX|inv_gov|V": "native"}

    monkeypatch.setattr(mod, "_adjust_unadjusted", fake_adjust)
    out = qna_sector_gfcf(["USA", "MEX"], sa="x13")
    assert seen["codes"] == {"MEX"}
    meta = qna_extras_meta(out).set_index(["code", "variable"])
    assert meta.loc[("MEX", "inv_gov"), "sa"] == "puremacro"
    assert out.attrs["sa_engines"] == {"MEX|inv_gov|V": "native"}


def test_sector_gfcf_provider_failure_returns_an_empty_frame_and_lists_what_is_missing(monkeypatch):
    monkeypatch.setattr(mod, "_get", _Recorder({"OECD": _fail("HTTP 429")}))
    with pytest.warns(UserWarning, match="HTTP 429"):
        out = qna_sector_gfcf(["USA", "MEX"])
    assert out.empty
    assert out.index.names == ["code", "date"]
    assert list(out.columns) == ["inv_gov", "inv_priv", "inv_gov_real", "inv_priv_real"]
    assert set(out.attrs) >= {"meta", "source", "fetched_at", "missing"}
    reasons = {m["reason"] for m in out.attrs["missing"]}
    assert any("HTTP 429" in r for r in reasons)
    assert {m["code"] for m in out.attrs["missing"]} == {"USA", "MEX"}


def test_sector_gfcf_keeps_a_reference_period_that_is_not_a_single_year(monkeypatch):
    answers = _sector_answers()
    flow1 = answers["GFCF_SECTOR"].copy()
    flow1["REF_YEAR_PRICE"] = flow1["REF_YEAR_PRICE"].astype(object)
    flow1.loc[flow1["PRICE_BASE"] == "L", "REF_YEAR_PRICE"] = "2009-2010"
    flow1.attrs["status"] = "ok"
    answers["GFCF_SECTOR"] = flow1
    monkeypatch.setattr(mod, "_get", _Recorder(answers))
    meta = qna_extras_meta(qna_sector_gfcf(["USA"], sa="prefer"))
    units = meta.set_index("variable").loc["inv_gov_real", "units"]
    assert "(2009-2010 prices)" in units


@pytest.mark.parametrize("fn", [qna_sector_gfcf, qna_population, oecd_lfs_panel,
                                oecd_vacancies])
def test_a_transport_exception_becomes_an_empty_frame_and_a_missing_list(monkeypatch, fn):
    import urllib.error

    def boom(*a, **k):
        raise urllib.error.HTTPError("https://sdmx.oecd.org", 429, "Too Many", {}, None)

    monkeypatch.setattr(mod, "_get", boom)
    with pytest.warns(UserWarning):
        out = fn(["USA", "MEX"])
    assert out.empty and out.index.names == ["code", "date"]
    assert {"meta", "source", "fetched_at", "missing", "requests"} <= set(out.attrs)
    assert out.attrs["missing"]
    assert all("HTTPError" in m["reason"] for m in out.attrs["missing"])


def test_sector_gfcf_rejects_an_unknown_adjustment_mode():
    with pytest.raises(ValueError, match="x13"):
        qna_sector_gfcf(["USA"], sa="stl")


# --------------------------------------------------------------------------
# qna_population
# --------------------------------------------------------------------------

def _pop_rows(code, txn, adj, value, mult=3):
    act = "_Z" if txn == "POP" else "_T"
    return [{"FREQ": "Q", "ADJUSTMENT": adj, "REF_AREA": code, "SECTOR": "S1",
             "COUNTERPART_SECTOR": "S1", "TRANSACTION": txn, "INSTR_ASSET": "_Z",
             "ACTIVITY": act, "EXPENDITURE": "_Z", "UNIT_MEASURE": "PS",
             "PRICE_BASE": "_Z", "TRANSFORMATION": "N", "TABLE_IDENTIFIER": "T0110",
             "TIME_PERIOD": t, "OBS_VALUE": value, "UNIT_MULT": mult, "CURRENCY": "_Z"}
            for t in _Q]


def test_population_is_in_thousands_and_prefers_the_adjusted_edition(monkeypatch):
    rows = (_pop_rows("MEX", "POP", "N", 130000.0) + _pop_rows("MEX", "POP", "Y", 130500.0)
            + _pop_rows("MEX", "EMP", "Y", 60000.0) + _pop_rows("MEX", "SAL", "Y", 40000.0)
            + _pop_rows("MEX", "SELF", "Y", 20000.0)
            # persons published in units (UNIT_MULT 0) are put in thousands
            + _pop_rows("USA", "POP", "N", 342_749_000.0, mult=0)
            + _pop_rows("EU27_2020", "POP", "N", 450000.0))
    rec = _Recorder({"POP_EMPNC": _ok(rows)})
    monkeypatch.setattr(mod, "_get", rec)
    out = qna_population(["MEX", "USA"])
    assert list(out.columns) == list(QNA_POPULATION)
    assert out.loc[("MEX", pd.Timestamp("2020-10-01")), "pop"] == 130500.0
    assert out.loc[("USA", pd.Timestamp("2019-04-01")), "pop"] == 342749.0
    mex = out.loc["MEX"]
    assert np.allclose(mex["emp_employees_nc"] + mex["emp_selfemp_nc"], mex["emp_nc"])
    assert "EU27_2020" not in out.index.get_level_values("code")
    assert rec.calls[0][1] == "Q..MEX+USA...POP+EMP+SAL+SELF......."
    missing = {(m["code"], m["variable"]) for m in out.attrs["missing"]}
    assert ("USA", "emp_nc") in missing
    meta = qna_extras_meta(out)
    assert set(meta["units"]) == {"thousands of persons"}


def test_population_attrs_carry_no_stale_response_status(monkeypatch):
    raw = _ok(_pop_rows("MEX", "POP", "Y", 130500.0))
    raw.attrs["url"] = "https://example.invalid"
    monkeypatch.setattr(mod, "_get", _Recorder({"POP_EMPNC": raw}))
    out = qna_population(["MEX"])
    assert set(out.attrs) == {"meta", "source", "fetched_at", "missing", "requests"}


def test_population_not_published_is_not_a_failure(monkeypatch):
    monkeypatch.setattr(mod, "_get", _Recorder({"POP_EMPNC": _fail("HTTP 404")}))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = qna_population(["ATA"])
    assert out.empty
    assert {m["reason"] for m in out.attrs["missing"]} == {"not published"}


# --------------------------------------------------------------------------
# oecd_lfs_panel
# --------------------------------------------------------------------------

def _lfs(code, measure, age, unit, adj, value, *, freq="Q", mult=None):
    periods = _Q if freq == "Q" else _M
    if mult is None:
        mult = 3 if unit == "PS" else 0
    return [{"DATAFLOW": "OECD.SDD.TPS:DSD_LFS@DF_IALFS_INDIC(1.0)", "REF_AREA": code,
             "MEASURE": measure, "UNIT_MEASURE": unit, "TRANSFORMATION": "_Z",
             "ADJUSTMENT": adj, "SEX": "_T", "AGE": age, "ACTIVITY": "_Z",
             "FREQ": freq, "TIME_PERIOD": t, "OBS_VALUE": value,
             "UNIT_MULT": mult} for t in periods]


def _lfs_rows(freq="Q"):
    rows = []
    for name, (measure, age, unit, _) in LFS_INDICATORS.items():
        rows += _lfs("USA", measure, age, unit, "Y", 10.0 + len(name), freq=freq)
    rows += _lfs("USA", "WAP", "Y15T64", "PS", "N", 1.0, freq=freq)      # NSA loses
    rows += _lfs("USA", "EMP", "Y_GE15", "GR", "Y", 0.3, freq=freq)      # growth: ignored
    rows += _lfs("USA", "UNE_LF", "Y_GE15", "PT_LF_SUB", "Y", 99.0, freq=freq)
    # ZAF publishes only the survey rate: urate falls back to it
    rows += _lfs("ZAF", "UNE_LF", "Y_GE15", "PT_LF_SUB", "Y", 32.0, freq=freq)
    rows += _lfs("ZAF", "WAP", "Y15T64", "PS", "N", 40000.0, freq=freq)  # NSA only: kept
    rows += _lfs("OECD", "UNE_LF_M", "Y_GE15", "PT_LF_SUB", "Y", 5.0, freq=freq)
    return rows


def test_lfs_panel_quarterly_happy_path(monkeypatch):
    rec = _Recorder({"DSD_LFS": _ok(_lfs_rows())})
    monkeypatch.setattr(mod, "_get", rec)
    out = oecd_lfs_panel(["USA", "ZAF"])
    assert out.index.names == ["code", "date"]
    assert list(out.columns) == list(LFS_INDICATORS)
    usa = out.loc["USA"]
    assert (usa["urate"] == 10.0 + len("urate")).all()       # UNE_LF_M, not UNE_LF
    assert (usa["wap_1564"] == 10.0 + len("wap_1564")).all()  # SA beats NSA
    assert (usa["emp_lfs"] == 10.0 + len("emp_lfs")).all()    # GR row not taken
    zaf = out.loc["ZAF"]
    assert (zaf["urate"] == 32.0).all()
    assert (zaf["wap_1564"] == 40000.0).all()
    meta = qna_extras_meta(out).set_index(["code", "variable"])
    assert meta.loc[("ZAF", "urate"), "series"].startswith("UNE_LF Y_GE15")
    assert meta.loc[("ZAF", "wap_1564"), "sa"] == "none"
    assert meta.loc[("USA", "urate"), "units"] == "percent"
    # DSD_LFS key: REF_AREA first, FREQ last, nine dimensions
    key = rec.calls[0][1]
    assert key.startswith("USA+ZAF.") and key.endswith(".Q") and key.count(".") == 8
    assert "OECD" not in out.index.get_level_values("code")


def test_lfs_panel_monthly_dates_are_month_starts(monkeypatch):
    rec = _Recorder({"DSD_LFS": _ok(_lfs_rows("M"))})
    monkeypatch.setattr(mod, "_get", rec)
    out = oecd_lfs_panel(["USA"], freq="M")
    dates = out.loc["USA"].index
    assert list(dates) == list(pd.date_range("2020-01-01", periods=6, freq="MS"))
    assert rec.calls[0][1].endswith(".M")
    assert out.attrs["freq"] == "M"


def test_lfs_panel_all_countries_chunks_the_availability_list(monkeypatch):
    areas = [f"C{i:02d}" for i in range(23)] + ["EA20", "OECD"]
    payload = {"data": {"contentConstraints": [{"cubeRegions": [
        {"keyValues": [{"id": "REF_AREA", "values": areas}]}]}]}}
    monkeypatch.setattr(mod, "_availability", lambda *a, **k: payload)
    rec = _Recorder({"DSD_LFS": _ok(_lfs_rows())})
    monkeypatch.setattr(mod, "_get", rec)
    oecd_lfs_panel(None)
    sent = [k.split(".")[0].split("+") for _, k in rec.calls]
    assert [len(s) for s in sent] == [10, 10, 3]
    assert not {"EA20", "OECD"} & {c for s in sent for c in s}


def test_lfs_panel_without_availability_chunks_the_known_areas_and_says_so(monkeypatch):
    monkeypatch.setattr(mod, "_availability", lambda *a, **k: {})
    rec = _Recorder({"DSD_LFS": _ok(_lfs_rows())})
    monkeypatch.setattr(mod, "_get", rec)
    with pytest.warns(UserWarning, match="availability"):
        out = oecd_lfs_panel(None)
    sent = [k.split(".")[0].split("+") for _, k in rec.calls]
    assert all(0 < len(s) <= 10 for s in sent)
    assert sorted(c for s in sent for c in s) == sorted(mod._LFS_AREAS)
    first = out.attrs["requests"][0]
    assert "availability" in first["flow"] and first["status"] == "failed"


def test_lfs_panel_rejects_an_unknown_frequency():
    with pytest.raises(ValueError, match="'Q', 'M'"):
        oecd_lfs_panel(["USA"], freq="A")


def test_lfs_panel_throttled_returns_empty_frame_without_raising(monkeypatch):
    monkeypatch.setattr(mod, "_get", _Recorder({"DSD_LFS": _fail("HTTP 429")}))
    with pytest.warns(UserWarning):
        out = oecd_lfs_panel(["USA", "MEX"])
    assert out.empty and out.index.names == ["code", "date"]
    assert list(out.columns) == list(LFS_INDICATORS)
    assert len(out.attrs["missing"]) == 2 * len(LFS_INDICATORS)
    assert out.attrs["requests"][0]["status"] == "HTTP 429"


# --------------------------------------------------------------------------
# oecd_vacancies
# --------------------------------------------------------------------------

def _olab(code, measure, adj, value, sector="S1", transformation="_Z"):
    return [{"DATAFLOW": "OECD.SDD.TPS:DSD_OLAB@DF_OIALAB_INDIC(1.1)", "REF_AREA": code,
             "MEASURE": measure, "UNIT_MEASURE": "PS", "TRANSFORMATION": transformation,
             "ADJUSTMENT": adj, "SECTOR": sector, "FREQ": "Q", "TIME_PERIOD": t,
             "OBS_VALUE": value, "UNIT_MULT": 0} for t in _Q]


def test_vacancies_are_in_thousands_and_keep_stock_and_flow_apart(monkeypatch):
    rows = (_olab("DEU", "VAC_U", "Y", 643667.0) + _olab("DEU", "VAC_U", "N", 1.0)
            # the growth rate (G1) shares UNIT_MEASURE=PS with the level and
            # sorts first; it must never be read as a level
            + _olab("DEU", "REG_UNE", "Y", 6.133, transformation="G1")
            + _olab("DEU", "REG_UNE", "Y", 2_900_000.0)
            + _olab("DEU", "VAC_U", "Y", 5.0, sector="S1D")
            + _olab("JPN", "VAC_N", "Y", 800000.0))
    rec = _Recorder({"DSD_OLAB": _ok(rows)})
    monkeypatch.setattr(mod, "_get", rec)
    out = oecd_vacancies(["DEU", "JPN"])
    assert list(out.columns) == list(OECD_VACANCIES)
    deu = out.loc["DEU"]
    assert np.allclose(deu["vacancies"], 643.667)
    assert np.allclose(deu["reg_unemp"], 2900.0)
    assert out.loc["JPN", "vacancies"].isna().all()
    assert np.allclose(out.loc["JPN", "vacancies_new"], 800.0)
    assert rec.calls[0][1] == "DEU+JPN.VAC_U+VAC_N+REG_UNE.PS._Z..S1.Q"


def test_vacancies_rejects_an_unknown_frequency():
    with pytest.raises(ValueError):
        oecd_vacancies(["DEU"], freq="W")


def test_meta_survives_concat_of_slices(monkeypatch):
    monkeypatch.setattr(mod, "_get", _Recorder({"DSD_OLAB": _ok(_olab("DEU", "VAC_U", "Y", 1e5))}))
    out = oecd_vacancies(["DEU"])
    assert isinstance(out.attrs["meta"], tuple)
    assert all(isinstance(r, dict) for r in out.attrs["meta"])
    pd.concat([out.iloc[:2], out.iloc[2:]])
    assert out.attrs["fetched_at"].endswith("+00:00")


# --------------------------------------------------------------------------
# qna_panel transport and aggregates (edits made alongside this module)
# --------------------------------------------------------------------------

def test_qna_panel_wrapper_goes_through_the_urllib_transport(monkeypatch):
    from puremacro.fetch import _oecd_sdmx, oecd_qna_panel

    seen = {}

    def fake_csv(agency_flow, key, **kw):
        seen.update(kw, agency_flow=agency_flow, key=key)
        return _ok([{"A": 1}])

    monkeypatch.setattr(_oecd_sdmx, "oecd_csv", fake_csv)
    out = oecd_qna_panel.get_sdmx_csv("OECD.SDD.NAD,X,", "Q..USA", "1947", refresh=True)
    assert len(out) == 1
    assert seen["start_period"] == "1947" and seen["labels"] is False
    assert seen["refresh"] is True


def test_qna_panel_reports_a_throttled_chunk_instead_of_dropping_it(monkeypatch):
    from puremacro.fetch import oecd_qna_panel

    monkeypatch.setattr(oecd_qna_panel, "get_sdmx_csv",
                        lambda flow, key, start, *, refresh=False: _fail("HTTP 429"))
    with pytest.warns(UserWarning, match="HTTP 429"):
        out = oecd_qna_panel.qna_panel(["USA"], start="2019")
    assert out.empty
    assert out.attrs["missing"]
    assert {m["status"] for m in out.attrs["missing"]} == {"HTTP 429"}
    assert all("USA" in m["key"] for m in out.attrs["missing"])


def test_qna_panel_treats_404_as_no_data_not_as_a_failure(monkeypatch):
    from puremacro.fetch import oecd_qna_panel

    monkeypatch.setattr(oecd_qna_panel, "get_sdmx_csv",
                        lambda flow, key, start, *, refresh=False: _fail("HTTP 404"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = oecd_qna_panel.qna_panel(["ATA"], start="2019")
    assert out.empty and out.attrs["missing"] == ()


def test_qna_aggregates_include_euro_area_and_eu():
    from puremacro.fetch.oecd_qna_panel import QNA_AGGREGATES

    assert {"EA", "EU"} <= QNA_AGGREGATES


# --------------------------------------------------------------------------
# Live (deselected by default)
# --------------------------------------------------------------------------

@pytest.mark.network
def test_live_lfs_panel_usa_starts_in_1955():
    out = oecd_lfs_panel(["USA"])
    if out.empty:
        pytest.skip(f"OECD unreachable: {out.attrs.get('missing')}")
    assert out.loc["USA"].index.min() == pd.Timestamp("1955-01-01")


@pytest.mark.network
def test_live_sector_gfcf_usa_from_1947():
    out = qna_sector_gfcf(["USA"])
    if out.empty:
        pytest.skip(f"OECD unreachable: {out.attrs.get('missing')}")
    assert out.loc["USA", "inv_gov"].first_valid_index() == pd.Timestamp("1947-01-01")
