"""Tests for puremacro.fetch.sdmx."""
from __future__ import annotations

import io

import pandas as pd
import pytest

from puremacro.fetch import sdmx_get, oecd_sdmx_instrument
from puremacro.instruments import Instrument


# Synthetic SDMX-CSV (subset of canonical OECD shape).
_SYNTHETIC_SDMX_CSV = """DATAFLOW,REF_AREA,MEASURE,UNIT_MEASURE,TIME_PERIOD,OBS_VALUE
OECD:DSD_STAN(1.0),USA,VALADD,USD_M,2018,21000.0
OECD:DSD_STAN(1.0),USA,VALADD,USD_M,2019,22000.0
OECD:DSD_STAN(1.0),USA,VALADD,USD_M,2020,21500.0
OECD:DSD_STAN(1.0),GBR,VALADD,USD_M,2018,3000.0
OECD:DSD_STAN(1.0),GBR,VALADD,USD_M,2019,3100.0
"""


def test_sdmx_get_with_csv_path_returns_dataframe(tmp_path):
    csv = tmp_path / "sdmx.csv"
    csv.write_text(_SYNTHETIC_SDMX_CSV)
    df = sdmx_get(provider="oecd", dataflow="DSD_STAN", key="USA",
                  csv_path=csv)
    assert isinstance(df, pd.DataFrame)
    assert "TIME_PERIOD" in df.columns
    assert "OBS_VALUE" in df.columns
    assert len(df) == 5


def test_sdmx_get_unknown_provider_raises():
    with pytest.raises(ValueError, match="provider"):
        sdmx_get(provider="not_a_real_provider", dataflow="X", key="Y",
                 csv_path=None)


def test_sdmx_get_known_providers():
    """Provider whitelist must include the four planned sources."""
    from puremacro.fetch.sdmx import _PROVIDERS
    assert "oecd" in _PROVIDERS
    assert "eurostat" in _PROVIDERS
    assert "ecb" in _PROVIDERS
    assert "imf" in _PROVIDERS


def test_sdmx_get_no_csv_no_network_raises_clear_error(monkeypatch):
    from puremacro.fetch import sdmx as _mod
    def _fail(_url, **kw):
        raise OSError("simulated network failure")
    monkeypatch.setattr(_mod, "safe_get_bytes", _fail)
    with pytest.raises(RuntimeError, match="SDMX"):
        sdmx_get(provider="oecd", dataflow="DSD_STAN", key="USA")


def test_oecd_sdmx_instrument_with_csv_path_returns_instrument(tmp_path):
    csv = tmp_path / "sdmx.csv"
    csv.write_text(_SYNTHETIC_SDMX_CSV)
    inst = oecd_sdmx_instrument(
        dataset="DSD_STAN", country="USA", indicator="VALADD",
        csv_path=csv,
    )
    assert isinstance(inst, Instrument)
    assert inst.category == "external_csv"
    assert inst.frequency == "A"
    assert inst.name == "oecd_DSD_STAN_USA_VALADD"
    assert inst.series.loc[pd.Timestamp("2018-01-01")] == pytest.approx(21000.0)
    assert len(inst.series) == 3  # USA-only, 3 years


def test_oecd_sdmx_instrument_filters_country(tmp_path):
    csv = tmp_path / "sdmx.csv"
    csv.write_text(_SYNTHETIC_SDMX_CSV)
    inst = oecd_sdmx_instrument(
        dataset="DSD_STAN", country="GBR", indicator="VALADD",
        csv_path=csv,
    )
    assert len(inst.series) == 2
    assert inst.series.loc[pd.Timestamp("2018-01-01")] == pytest.approx(3000.0)


def test_oecd_sdmx_instrument_unknown_country_raises(tmp_path):
    csv = tmp_path / "sdmx.csv"
    csv.write_text(_SYNTHETIC_SDMX_CSV)
    with pytest.raises(ValueError, match="ZZZ"):
        oecd_sdmx_instrument(
            dataset="DSD_STAN", country="ZZZ", indicator="VALADD",
            csv_path=csv,
        )


def test_oecd_sdmx_instrument_metadata_includes_provider_and_dataset(tmp_path):
    csv = tmp_path / "sdmx.csv"
    csv.write_text(_SYNTHETIC_SDMX_CSV)
    inst = oecd_sdmx_instrument(
        dataset="DSD_STAN", country="USA", indicator="VALADD",
        csv_path=csv,
    )
    assert inst.metadata.get("provider") == "oecd"
    assert inst.metadata.get("dataset") == "DSD_STAN"
    assert inst.metadata.get("country") == "USA"
    assert inst.metadata.get("indicator") == "VALADD"
    assert "reference" in inst.metadata


def test_oecd_sdmx_instrument_rejects_non_annual_frequency(tmp_path):
    """Phase 1 only handles annual data; non-annual frequency must raise."""
    csv = tmp_path / "sdmx.csv"
    csv.write_text(_SYNTHETIC_SDMX_CSV)
    with pytest.raises(ValueError, match="annual"):
        oecd_sdmx_instrument(
            dataset="DSD_STAN", country="USA", indicator="VALADD",
            csv_path=csv, frequency="Q",
        )


def test_ilostat_provider_registered():
    """ILOSTAT should be a 5th provider with a CSV-format URL template."""
    from puremacro.fetch.sdmx import _PROVIDERS
    assert "ilostat" in _PROVIDERS
    template = _PROVIDERS["ilostat"]
    assert "{dataflow}" in template
    assert "{key}" in template
    assert "ilo.org" in template
    assert "format=csv" in template
