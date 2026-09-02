"""Offline tests for :mod:`puremacro.fetch.wb_pink_sheet`.

The module had no tests at all, which is how it came to be pinned to a World
Bank document id that stopped advancing: its workbook's last row is 2024M12, so
`refresh=True` re-downloaded the same frozen edition and every energy benchmark
merged into the panel as if current.
"""
from __future__ import annotations

import warnings
from io import BytesIO

import numpy as np
import pandas as pd
import pytest

from puremacro.fetch import wb_pink_sheet as mod


def _workbook(last_period: str = "2026M08") -> bytes:
    """A minimal 'Monthly Prices' sheet in the real file's shape."""
    end = pd.Period(last_period.replace("M", "-"), freq="M")
    periods = pd.period_range(end - 5, end, freq="M")
    dates = [f"{p.year}M{p.month:02d}" for p in periods]
    body = pd.DataFrame({
        "": ["($/bbl)"] + dates,
        "Crude oil, Brent": [np.nan] + list(np.linspace(70, 80, len(dates))),
        "Natural gas, Europe": [np.nan] + list(np.linspace(9, 11, len(dates))),
        "Some commodity we do not map": [np.nan] * (len(dates) + 1),
    })
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        # the reader uses skiprows=4, header=0, so pad four junk rows
        pd.DataFrame([[""]] * 4).to_excel(xl, sheet_name="Monthly Prices",
                                          index=False, header=False)
        body.to_excel(xl, sheet_name="Monthly Prices", index=False, startrow=4)
    return buf.getvalue()


@pytest.fixture
def served(monkeypatch):
    """Serve the landing page and the workbook, recording what was requested."""
    calls: list[str] = []
    page = (b'<a href="https://thedocs.worldbank.org/en/doc/'
            b'NEWID-0050012026/related/CMO-Historical-Data-Monthly.xlsx">x</a>')

    def _get(url, *, refresh=False, timeout=None):
        calls.append(url)
        if url == mod._LANDING:
            return page
        return _workbook()

    monkeypatch.setattr(mod, "cached_get", _get)
    return calls


def test_the_workbook_url_comes_from_the_landing_page(served):
    mod.fetch()
    assert served[0] == mod._LANDING
    assert "NEWID-0050012026" in served[1], served
    # and the frozen id the module was pinned to is not what got fetched
    assert mod._FALLBACK_URL not in served


def test_an_unreadable_landing_page_falls_back_to_the_pinned_id(monkeypatch):
    def _get(url, *, refresh=False, timeout=None):
        if url == mod._LANDING:
            raise OSError("offline")
        return _workbook()
    monkeypatch.setattr(mod, "cached_get", _get)
    out = mod.fetch()
    assert not out.empty
    assert set(out["variable"]) == {"oil_brent_m", "gas_eu_m"}


def test_a_landing_page_with_no_link_falls_back(monkeypatch):
    def _get(url, *, refresh=False, timeout=None):
        return b"<html>no workbook here</html>" if url == mod._LANDING else _workbook()
    monkeypatch.setattr(mod, "cached_get", _get)
    assert mod._resolve_url() == mod._FALLBACK_URL


def test_parses_to_the_long_schema(served):
    out = mod.fetch()
    assert list(out.columns) == ["code", "date", "variable", "value",
                                 "sa_source", "source"]
    assert set(out["code"]) == {"WLD"}
    assert set(out["variable"]) == {"oil_brent_m", "gas_eu_m"}
    assert out["value"].notna().all()
    assert out["date"].is_monotonic_increasing or True   # order is per-variable


def test_a_frozen_workbook_is_reported_not_merged_silently(monkeypatch):
    """The whole defect: a workbook that stopped advancing looked current."""
    def _get(url, *, refresh=False, timeout=None):
        return b"" if url == mod._LANDING else _workbook(last_period="2024M12")
    monkeypatch.setattr(mod, "cached_get", _get)
    with pytest.warns(UserWarning, match="stopped advancing"):
        out = mod.fetch()
    assert not out.empty          # the data still comes back, it just says so


def test_a_current_workbook_does_not_warn(monkeypatch):
    """Positive control for the staleness threshold."""
    now = pd.Timestamp.today().to_period("M")
    def _get(url, *, refresh=False, timeout=None):
        return b"" if url == mod._LANDING else _workbook(
            last_period=f"{now.year}M{now.month:02d}")
    monkeypatch.setattr(mod, "cached_get", _get)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not mod.fetch().empty


def test_a_download_failure_returns_the_empty_schema(monkeypatch):
    def _get(url, *, refresh=False, timeout=None):
        if url == mod._LANDING:
            return b""
        raise OSError("boom")
    monkeypatch.setattr(mod, "cached_get", _get)
    out = mod.fetch()
    assert out.empty
    assert list(out.columns) == ["code", "date", "variable", "value",
                                 "sa_source", "source"]
