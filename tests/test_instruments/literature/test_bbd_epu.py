"""Tests for puremacro.instruments.literature.bbd_epu."""
from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import pytest

from puremacro.instruments import Instrument
from puremacro.instruments.literature.bbd_epu import load


_SYNTHETIC_CSV = """Year,Month,News_Based_Policy_Uncert_Index
1985,1,55.42
1985,2,87.16
1985,3,108.06
2001,9,392.58
"""


def test_load_with_csv_path_returns_instrument(tmp_path):
    csv = tmp_path / "epu.csv"
    csv.write_text(_SYNTHETIC_CSV)
    inst = load(csv_path=csv)
    assert isinstance(inst, Instrument)
    assert inst.category == "literature"
    assert inst.frequency == "M"
    assert inst.name == "bbd_epu_us"


def test_load_csv_extracts_correct_values(tmp_path):
    csv = tmp_path / "epu.csv"
    csv.write_text(_SYNTHETIC_CSV)
    inst = load(csv_path=csv)
    assert inst.series.loc[pd.Timestamp("1985-01-01")] == pytest.approx(55.42)
    assert inst.series.loc[pd.Timestamp("2001-09-01")] == pytest.approx(392.58)
    assert len(inst.series) == 4


def test_metadata_has_reference(tmp_path):
    csv = tmp_path / "epu.csv"
    csv.write_text(_SYNTHETIC_CSV)
    inst = load(csv_path=csv)
    assert "reference" in inst.metadata
    assert "Baker" in inst.metadata["reference"]
    assert "Bloom" in inst.metadata["reference"]
    assert "Davis" in inst.metadata["reference"]


def test_load_no_csv_no_network_raises_runtime_error(monkeypatch):
    """If csv_path is None and the network fetch fails, raise a
    RuntimeError pointing at policyuncertainty.com."""
    from puremacro.instruments.literature import bbd_epu as _mod
    def _fail_fetch(_url):
        raise OSError("simulated network failure")
    monkeypatch.setattr(_mod, "safe_get_bytes", _fail_fetch)
    with pytest.raises(RuntimeError, match="policyuncertainty.com"):
        load()


def test_load_csv_with_wrong_columns_raises_clear_error(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text("year,month,value\n2000,1,1.0\n")  # lowercase column names
    with pytest.raises(ValueError, match="missing expected columns"):
        load(csv_path=csv)
