"""Tests for puremacro.instruments.external.bis."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from puremacro.instruments import Instrument
from puremacro.instruments.external.bis import load


# Synthetic BIS-style CSV: long format with country code, period, value.
_SYNTHETIC_CSV = """ISO,date,value
US,1999-Q1,3.2
US,1999-Q2,3.5
US,1999-Q3,3.8
GB,1999-Q1,1.1
GB,1999-Q2,1.4
"""


def test_load_with_csv_path_filters_country(tmp_path):
    csv = tmp_path / "bis.csv"
    csv.write_text(_SYNTHETIC_CSV, encoding="utf-8")
    inst = load(series_id="credit_to_gdp_gap", country="US", csv_path=csv)
    assert isinstance(inst, Instrument)
    assert inst.category == "external_csv"
    assert inst.frequency == "Q"
    assert inst.name == "bis_credit_to_gdp_gap_US"
    assert len(inst.series) == 3
    assert inst.series.iloc[0] == pytest.approx(3.2)


def test_load_filters_to_requested_country(tmp_path):
    csv = tmp_path / "bis.csv"
    csv.write_text(_SYNTHETIC_CSV, encoding="utf-8")
    inst = load(series_id="credit_to_gdp_gap", country="GB", csv_path=csv)
    assert len(inst.series) == 2
    assert inst.series.iloc[0] == pytest.approx(1.1)


def test_load_unknown_country_raises_with_available_list(tmp_path):
    csv = tmp_path / "bis.csv"
    csv.write_text(_SYNTHETIC_CSV, encoding="utf-8")
    with pytest.raises(ValueError, match="ZZ"):
        load(series_id="credit_to_gdp_gap", country="ZZ", csv_path=csv)


def test_load_csv_with_wrong_columns_raises(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text("country,quarter,gap\nUS,1999Q1,3.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing expected columns"):
        load(series_id="credit_to_gdp_gap", country="US", csv_path=csv)


def test_load_metadata_has_reference(tmp_path):
    csv = tmp_path / "bis.csv"
    csv.write_text(_SYNTHETIC_CSV, encoding="utf-8")
    inst = load(series_id="credit_to_gdp_gap", country="US", csv_path=csv)
    assert "reference" in inst.metadata
    assert "BIS" in inst.metadata["reference"] or "Bank for International" in inst.metadata["reference"]


def test_load_no_csv_no_network_raises(monkeypatch):
    from puremacro.instruments.external import bis as _mod
    def _fail(_url):
        raise OSError("simulated network failure")
    monkeypatch.setattr(_mod, "safe_get_bytes", _fail)
    with pytest.raises(RuntimeError, match="bis.org"):
        load(series_id="credit_to_gdp_gap", country="US")


def test_load_handles_q01_leading_zero_date_variant(tmp_path):
    """BIS occasionally publishes Q01 (leading-zero) instead of Q1."""
    csv_text = (
        "ISO,date,value\n"
        "US,1999-Q01,3.2\n"
        "US,1999-Q02,3.5\n"
        "US,2000-Q01,4.0\n"
    )
    csv = tmp_path / "bis_q01.csv"
    csv.write_text(csv_text, encoding="utf-8")
    inst = load(series_id="credit_to_gdp_gap", country="US", csv_path=csv)
    assert len(inst.series) == 3
    assert inst.series.loc[pd.Timestamp("1999-01-01")] == pytest.approx(3.2)
    assert inst.series.loc[pd.Timestamp("2000-01-01")] == pytest.approx(4.0)


def test_load_unknown_series_id_no_csv_path_raises(monkeypatch):
    """If the user requests a series_id with no default mirror and no
    csv_path, raise a helpful error."""
    with pytest.raises(RuntimeError, match="no default mirror"):
        load(series_id="not_a_real_series", country="US")
