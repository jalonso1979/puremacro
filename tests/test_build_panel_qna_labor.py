"""The QNA labour route `build_all` uses to fill gaps in the local workbook.

Pins the contract the migration off ``fetch.oecd_qna_labor`` had to preserve —
the panel's long schema, the two variable names ``keep_mask`` matches on
literally, natural logs rather than levels — and the two things it changed: a
reference area publishing the labour block raw now reports a real adjustment
instead of the ``x13_pending`` label nothing ever honoured, and the fetch no
longer drags the expenditure block along with it.

These patch ``get_sdmx_csv``, not ``qna_labor``. Mocking the fetcher would make
the seasonal-adjustment assertion vacuous, since that is precisely the
behaviour the migration moved.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro import build_panel as bp
from puremacro.fetch import oecd_qna_panel as mod

_SCHEMA = ["code", "date", "variable", "value", "sa_source", "source"]
_QUARTERS = [f"{y}-Q{q}" for y in range(2000, 2024) for q in (1, 2, 3, 4)]
_SEASON = {1: 0.94, 2: 1.04, 3: 1.00, 4: 1.02}


def _sdmx_labor(code: str, adjustment: str) -> list[dict]:
    """EMP as persons and hours, SDMX-shaped, with a real seasonal in it."""
    rows = []
    for unit, base in (("PS", 20_000.0), ("H", 8_000.0)):
        for t, period in enumerate(_QUARTERS):
            rows.append({
                "FREQ": "Q", "ADJUSTMENT": adjustment, "REF_AREA": code,
                "SECTOR": "S1", "COUNTERPART_SECTOR": "S1",
                "TRANSACTION": "EMP", "INSTR_ASSET": "_Z", "ACTIVITY": "_T",
                "EXPENDITURE": "_Z", "UNIT_MEASURE": unit, "PRICE_BASE": "_Z",
                "TRANSFORMATION": "N", "TABLE_IDENTIFIER": "T0111",
                "TIME_PERIOD": period,
                "OBS_VALUE": base * (1 + 0.002 * t) * _SEASON[int(period[-1])],
                "REF_YEAR_PRICE": None, "UNIT_MULT": 3 if unit == "PS" else 6,
                "CURRENCY": "_Z",
            })
    return rows


@pytest.fixture
def sdmx_calls(monkeypatch):
    """DEU is adjusted at source, MEX publishes the block raw."""
    calls: list[str] = []

    def _fake(agency_flow, key, start_period, *, refresh=False, **kw):
        calls.append(agency_flow)
        if "EMPDC" not in agency_flow:
            raise AssertionError(
                "the labour gap-fill must not download the expenditure block")
        return pd.DataFrame(_sdmx_labor("DEU", "Y") + _sdmx_labor("MEX", "N"))

    monkeypatch.setattr(mod, "get_sdmx_csv", _fake)
    return calls


def test_adapter_returns_the_panel_long_schema(sdmx_calls):
    assert list(bp._fetch_qna_labor_logs(["DEU", "MEX"]).columns) == _SCHEMA


def test_adapter_keeps_the_variable_names_keep_mask_matches_on(sdmx_calls):
    """`keep_mask` in build_all filters on these two literal strings. Rename
    them and the local workbook silently stops winning over SDMX."""
    df = bp._fetch_qna_labor_logs(["DEU", "MEX"])
    assert set(df["variable"]) == {"log_emp_qna", "log_hours_qna"}


def test_adapter_emits_logs_not_levels(sdmx_calls):
    """qna_labor returns levels; every other producer of these two variables
    in the panel emits natural logs, so the adapter has to convert."""
    df = bp._fetch_qna_labor_logs(["DEU", "MEX"])
    v = df[(df["code"] == "DEU") & (df["variable"] == "log_emp_qna")]["value"]
    assert 9.0 < v.iloc[0] < 11.0                       # log of ~20,000
    assert np.isfinite(v).all()


def test_a_raw_publisher_is_really_adjusted_not_just_relabelled(sdmx_calls):
    """The whole point of the migration. The retired route labelled every
    unadjusted reference area `x13_pending` and never adjusted it; nothing in
    build_panel acted on that label, so those series stayed raw and were not
    even flagged by sa_audit. This runs the real engine."""
    df = bp._fetch_qna_labor_logs(["DEU", "MEX"])
    by_code = dict(zip(df["code"], df["sa_source"]))
    assert by_code["DEU"] == "oecd"
    assert by_code["MEX"] == "x13"
    assert "x13_pending" not in set(df["sa_source"])
    assert set(df["sa_source"]) <= {"oecd", "x13", "none"}

    # and the seasonal really is gone from the series that got adjusted
    mex = df[(df["code"] == "MEX") & (df["variable"] == "log_emp_qna")]
    d = mex.set_index("date")["value"].diff().dropna()
    by_q = d.groupby(d.index.quarter).mean()
    assert (by_q.max() - by_q.min()) < 0.01


def test_the_gap_fill_never_downloads_the_expenditure_block(sdmx_calls):
    """qna_panel(labor=True) would. It also drops any country the expenditure
    flow did not return, and loses everything if that request comes back
    empty — neither acceptable for a source whose whole job is filling gaps."""
    bp._fetch_qna_labor_logs(["DEU", "MEX"])
    assert len(sdmx_calls) == 1 and "EMPDC" in sdmx_calls[0]


def test_a_country_absent_from_the_expenditure_block_still_arrives(monkeypatch):
    """MEX publishes labour here and no expenditure at all. The legacy route
    fetched labour independently and so did not care; the replacement must
    not care either."""
    def _fake(agency_flow, key, start_period, *, refresh=False, **kw):
        if "EMPDC" in agency_flow:
            return pd.DataFrame(_sdmx_labor("MEX", "Y"))
        return pd.DataFrame()

    monkeypatch.setattr(mod, "get_sdmx_csv", _fake)
    df = bp._fetch_qna_labor_logs(["MEX"])
    assert sorted(df["code"].unique()) == ["MEX"]


def test_adapter_degrades_to_empty_rather_than_raising(monkeypatch):
    """A dead network must cost this source's rows, not the whole build."""
    def _boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(mod, "get_sdmx_csv", _boom)
    df = bp._fetch_qna_labor_logs(["DEU"])
    assert df.empty and list(df.columns) == _SCHEMA


def test_adapter_drops_non_positive_values_before_taking_logs(monkeypatch):
    """log(0) is -inf and log(negative) is nan; neither belongs in a panel."""
    rows = _sdmx_labor("DEU", "Y")
    rows[0]["OBS_VALUE"] = 0.0
    monkeypatch.setattr(mod, "get_sdmx_csv",
                        lambda *a, **k: pd.DataFrame(rows))
    df = bp._fetch_qna_labor_logs(["DEU"])
    assert np.isfinite(df["value"]).all()
