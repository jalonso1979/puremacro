"""Cluster C (item C3) — external benchmark fetchers (mocked + network smokes)."""
from __future__ import annotations

from unittest import mock

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# WUI panel
# ---------------------------------------------------------------------------
def test_fetch_wui_panel_returns_dataframe(tmp_path, monkeypatch):
    """When the fetcher hits a fixture workbook, it returns a country-by-
    quarter DataFrame indexed by quarter-start Timestamps."""
    fixture = pd.DataFrame({
        "year": ["1990q1", "1990q2", "1990q3"],
        "USA": [1.0, 2.0, 3.0],
        "MEX": [0.5, 1.5, 2.5],
    })

    def fake_read_excel(*_args, **_kwargs):
        return fixture.copy()

    monkeypatch.setattr(
        "puremacro.narrative.validation.external_benchmarks.pd.read_excel",
        fake_read_excel,
    )
    from puremacro.narrative.validation import fetch_ahir_bloom_furceri_wui_panel
    df = fetch_ahir_bloom_furceri_wui_panel(cache_dir=tmp_path)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["USA", "MEX"]
    assert df.shape == (3, 2)
    assert df.iloc[0]["USA"] == pytest.approx(1.0)


def test_fetch_wui_panel_country_slice(tmp_path, monkeypatch):
    fixture = pd.DataFrame({
        "year": ["1990q1", "1990q2"],
        "USA": [1.0, 2.0],
        "MEX": [0.5, 1.5],
    })
    monkeypatch.setattr(
        "puremacro.narrative.validation.external_benchmarks.pd.read_excel",
        lambda *_a, **_kw: fixture.copy(),
    )
    from puremacro.narrative.validation import fetch_ahir_bloom_furceri_wui_panel
    ser = fetch_ahir_bloom_furceri_wui_panel(country="USA", cache_dir=tmp_path)
    assert isinstance(ser, pd.Series)
    assert ser.tolist() == [1.0, 2.0]


def test_fetch_wui_panel_unknown_country_raises(tmp_path, monkeypatch):
    fixture = pd.DataFrame({"year": ["1990q1"], "USA": [1.0]})
    monkeypatch.setattr(
        "puremacro.narrative.validation.external_benchmarks.pd.read_excel",
        lambda *_a, **_kw: fixture.copy(),
    )
    from puremacro.narrative.validation import fetch_ahir_bloom_furceri_wui_panel
    with pytest.raises(ValueError, match="not in WUI"):
        fetch_ahir_bloom_furceri_wui_panel(country="ZZZ", cache_dir=tmp_path)


# ---------------------------------------------------------------------------
# BBD country EPU
# ---------------------------------------------------------------------------
def test_fetch_bbd_country_epu_us(tmp_path, monkeypatch):
    fixture = pd.DataFrame({
        "Year": [2022, 2022, 2022],
        "Month": [1.0, 2.0, 3.0],
        "US": [100.0, 110.0, 120.0],
        "UK": [80.0, 85.0, 90.0],
    })
    monkeypatch.setattr(
        "puremacro.narrative.validation.external_benchmarks.pd.read_excel",
        lambda *_a, **_kw: fixture.copy(),
    )
    from puremacro.narrative.validation import fetch_bbd_country_epu_monthly
    ser = fetch_bbd_country_epu_monthly(country="US", cache_dir=tmp_path)
    assert ser.iloc[0] == pytest.approx(100.0)
    assert ser.iloc[2] == pytest.approx(120.0)
    assert ser.name == "bbd_epu_us"


def test_fetch_bbd_country_epu_drops_trailing_notes(tmp_path, monkeypatch):
    """Trailing rows with non-numeric Year (citations, footnotes) must be dropped."""
    fixture = pd.DataFrame({
        "Year": [2022, 2022, "Zalla, R. — note"],
        "Month": [1.0, 2.0, None],
        "US": [100.0, 110.0, None],
    })
    monkeypatch.setattr(
        "puremacro.narrative.validation.external_benchmarks.pd.read_excel",
        lambda *_a, **_kw: fixture.copy(),
    )
    from puremacro.narrative.validation import fetch_bbd_country_epu_monthly
    ser = fetch_bbd_country_epu_monthly(country="US", cache_dir=tmp_path)
    assert len(ser) == 2  # the note row is dropped


def test_fetch_bbd_country_epu_unknown_country_raises(tmp_path, monkeypatch):
    fixture = pd.DataFrame({"Year": [2022], "Month": [1.0], "US": [100.0]})
    monkeypatch.setattr(
        "puremacro.narrative.validation.external_benchmarks.pd.read_excel",
        lambda *_a, **_kw: fixture.copy(),
    )
    from puremacro.narrative.validation import fetch_bbd_country_epu_monthly
    with pytest.raises(ValueError, match="not in BBD"):
        fetch_bbd_country_epu_monthly(country="ZZZ", cache_dir=tmp_path)


@pytest.mark.network
def test_fetch_wui_panel_live_smoke(tmp_path):
    from puremacro.narrative.validation import fetch_ahir_bloom_furceri_wui_panel
    try:
        df = fetch_ahir_bloom_furceri_wui_panel(cache_dir=tmp_path)
    except Exception as e:
        pytest.skip(f"WUI endpoint unreachable: {e}")
    if df.empty:
        pytest.skip("WUI fetcher returned empty")
    assert "USA" in df.columns
    assert df.shape[1] >= 50


@pytest.mark.network
def test_fetch_bbd_country_epu_live_smoke(tmp_path):
    from puremacro.narrative.validation import fetch_bbd_country_epu_monthly
    try:
        ser = fetch_bbd_country_epu_monthly(country="US", cache_dir=tmp_path)
    except Exception as e:
        pytest.skip(f"BBD country-EPU endpoint unreachable: {e}")
    if ser.empty:
        pytest.skip("BBD country-EPU fetcher returned empty")
    assert ser.index.min() <= pd.Timestamp("2000-01-01")
