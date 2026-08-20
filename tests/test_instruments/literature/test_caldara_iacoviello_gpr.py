"""Tests for puremacro.instruments.literature.caldara_iacoviello_gpr."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.instruments import Instrument
from puremacro.instruments.literature.caldara_iacoviello_gpr import load


_SYNTHETIC_CSV = """month,GPR
1985-01-01,68.41
1985-02-01,70.22
2001-09-01,295.10
"""


def test_load_with_csv_path_returns_instrument(tmp_path):
    csv = tmp_path / "gpr.csv"
    csv.write_text(_SYNTHETIC_CSV, encoding="utf-8")
    inst = load(csv_path=csv)
    assert isinstance(inst, Instrument)
    assert inst.category == "literature"
    assert inst.frequency == "M"
    assert inst.name == "caldara_iacoviello_gpr"


def test_load_extracts_correct_values(tmp_path):
    csv = tmp_path / "gpr.csv"
    csv.write_text(_SYNTHETIC_CSV, encoding="utf-8")
    inst = load(csv_path=csv)
    assert inst.series.loc[pd.Timestamp("1985-01-01")] == pytest.approx(68.41)
    assert inst.series.loc[pd.Timestamp("2001-09-01")] == pytest.approx(295.10)
    assert len(inst.series) == 3


def test_metadata_has_reference(tmp_path):
    csv = tmp_path / "gpr.csv"
    csv.write_text(_SYNTHETIC_CSV, encoding="utf-8")
    inst = load(csv_path=csv)
    assert "reference" in inst.metadata
    assert "Caldara" in inst.metadata["reference"]
    assert "Iacoviello" in inst.metadata["reference"]


def test_load_no_csv_no_network_raises_runtime_error(monkeypatch):
    from puremacro.instruments.literature import caldara_iacoviello_gpr as _mod
    def _fail_fetch(_url):
        raise OSError("simulated network failure")
    monkeypatch.setattr(_mod, "safe_get_bytes", _fail_fetch)
    with pytest.raises(RuntimeError, match="matteoiacoviello.com"):
        load()


def test_load_csv_with_wrong_columns_raises_clear_error(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text("date,risk_index\n2000-01-01,1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing expected columns"):
        load(csv_path=csv)
