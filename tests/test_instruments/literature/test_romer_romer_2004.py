"""Tests for puremacro.instruments.literature.romer_romer_2004."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.instruments import Instrument
from puremacro.instruments.literature.romer_romer_2004 import load


_SYNTHETIC_CSV = """date,RR_shock
1969-01-01,0.12
1969-04-01,-0.05
1969-07-01,0.08
1980-04-01,1.42
"""


def test_load_with_csv_path_returns_instrument(tmp_path):
    csv = tmp_path / "rr2004.csv"
    csv.write_text(_SYNTHETIC_CSV, encoding="utf-8")
    inst = load(csv_path=csv)
    assert isinstance(inst, Instrument)
    assert inst.category == "literature"
    assert inst.frequency == "Q"
    assert inst.name == "rr_2004_monetary"


def test_load_with_csv_path_default_value_col(tmp_path):
    """Default value_col is 'RR_shock' but accepts an override."""
    csv = tmp_path / "rr2004.csv"
    csv.write_text(_SYNTHETIC_CSV, encoding="utf-8")
    inst = load(csv_path=csv)
    assert inst.series.loc[pd.Timestamp("1980-04-01")] == pytest.approx(1.42)
    assert len(inst.series) == 4


def test_load_with_alternative_value_col(tmp_path):
    """The user can specify a non-default value column."""
    csv = tmp_path / "rr2004_alt.csv"
    csv.write_text("date,intended_residual\n1969-01-01,0.12\n1969-04-01,-0.05\n", encoding="utf-8")
    inst = load(csv_path=csv, value_col="intended_residual")
    assert inst.series.loc[pd.Timestamp("1969-01-01")] == pytest.approx(0.12)


def test_metadata_has_reference(tmp_path):
    csv = tmp_path / "rr2004.csv"
    csv.write_text(_SYNTHETIC_CSV, encoding="utf-8")
    inst = load(csv_path=csv)
    assert "reference" in inst.metadata
    assert "Romer" in inst.metadata["reference"]


def test_load_no_csv_no_network_raises_runtime_error(monkeypatch):
    from puremacro.instruments.literature import romer_romer_2004 as _mod
    def _fail_fetch(_url):
        raise OSError("simulated network failure")
    monkeypatch.setattr(_mod, "safe_get_bytes", _fail_fetch)
    with pytest.raises(RuntimeError, match="Romer"):
        load()


def test_load_csv_with_wrong_columns_raises_clear_error(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text("quarter,shock\n1969Q1,0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing expected columns"):
        load(csv_path=csv)
