"""Tests for puremacro.instruments.external.imf_weo."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.instruments import Instrument
from puremacro.instruments.external.imf_weo import load


# Synthetic WEO-style CSV (tab-separated): subset of columns for tests.
_SYNTHETIC_WEO = (
    "ISO\tWEO Subject Code\t2015\t2016\t2017\t2018\t2019\t2020\n"
    "USA\tGGXWDG_NGDP\t104.5\t106.2\t105.8\t107.1\t108.4\t135.0\n"
    "USA\tGGXONLB_NGDP\t-2.5\t-3.2\t-3.4\t-4.0\t-4.6\t-12.0\n"
    "GBR\tGGXWDG_NGDP\t87.9\t87.9\t86.2\t85.7\t85.2\t104.5\n"
    "USA\tNGDP_RPCH\t2.7\t1.7\t2.3\t2.9\t2.3\t-3.4\n"
)


def test_load_with_csv_path_returns_instrument(tmp_path):
    csv = tmp_path / "weo.csv"
    csv.write_text(_SYNTHETIC_WEO)
    inst = load(indicator="GGXWDG_NGDP", country="USA", csv_path=csv)
    assert isinstance(inst, Instrument)
    assert inst.category == "external_csv"
    assert inst.frequency == "A"
    assert inst.name == "imf_weo_GGXWDG_NGDP_USA"


def test_load_extracts_correct_year_values(tmp_path):
    csv = tmp_path / "weo.csv"
    csv.write_text(_SYNTHETIC_WEO)
    inst = load(indicator="GGXWDG_NGDP", country="USA", csv_path=csv)
    assert inst.series.loc[pd.Timestamp("2015-01-01")] == pytest.approx(104.5)
    assert inst.series.loc[pd.Timestamp("2020-01-01")] == pytest.approx(135.0)
    assert len(inst.series) == 6


def test_load_different_indicator(tmp_path):
    csv = tmp_path / "weo.csv"
    csv.write_text(_SYNTHETIC_WEO)
    inst = load(indicator="GGXONLB_NGDP", country="USA", csv_path=csv)
    assert inst.series.loc[pd.Timestamp("2020-01-01")] == pytest.approx(-12.0)


def test_load_different_country(tmp_path):
    csv = tmp_path / "weo.csv"
    csv.write_text(_SYNTHETIC_WEO)
    inst = load(indicator="GGXWDG_NGDP", country="GBR", csv_path=csv)
    assert inst.series.loc[pd.Timestamp("2015-01-01")] == pytest.approx(87.9)


def test_load_missing_country_indicator_pair_raises(tmp_path):
    csv = tmp_path / "weo.csv"
    csv.write_text(_SYNTHETIC_WEO)
    with pytest.raises(ValueError, match="not found"):
        load(indicator="GGXONLB_NGDP", country="GBR", csv_path=csv)


def test_load_metadata_has_reference(tmp_path):
    csv = tmp_path / "weo.csv"
    csv.write_text(_SYNTHETIC_WEO)
    inst = load(indicator="GGXWDG_NGDP", country="USA", csv_path=csv)
    assert "reference" in inst.metadata
    assert "WEO" in inst.metadata["reference"] or "World Economic Outlook" in inst.metadata["reference"]


def test_load_csv_with_wrong_columns_raises(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text("country\tcode\tval\nUSA\tX\t1.0\n")
    with pytest.raises(ValueError, match="missing expected columns"):
        load(indicator="X", country="USA", csv_path=csv)


def test_load_no_csv_no_network_raises(monkeypatch):
    from puremacro.instruments.external import imf_weo as _mod
    def _fail(_url):
        raise OSError("simulated network failure")
    monkeypatch.setattr(_mod, "safe_get_bytes", _fail)
    with pytest.raises(RuntimeError, match="imf.org"):
        load(indicator="GGXWDG_NGDP", country="USA")


def test_load_with_latin1_encoded_csv(tmp_path):
    """Real WEO files use Latin-1 encoding (e.g., Côte d'Ivoire). The
    loader must fall back from UTF-8 to Latin-1 transparently."""
    # Build the synthetic file with a Latin-1-only byte (0xE7 = ç)
    weo_with_accent = (
        "ISO\tWEO Subject Code\t2015\t2016\t2017\t2018\t2019\t2020\n"
        "CIV\tGGXWDG_NGDP\t40.0\t41.0\t42.0\t43.0\t44.0\t45.0\n"  # Côte d'Ivoire
    ).encode("latin-1")
    # Inject a non-UTF-8 byte that valid UTF-8 would reject.
    weo_with_accent = weo_with_accent.replace(b"CIV", b"C\xf4te")  # invalid UTF-8 byte
    csv = tmp_path / "weo_latin1.csv"
    csv.write_bytes(weo_with_accent)
    inst = load(indicator="GGXWDG_NGDP", country="C\xf4te", csv_path=csv)
    assert isinstance(inst, Instrument)
    assert inst.series.loc[pd.Timestamp("2015-01-01")] == pytest.approx(40.0)


def test_load_warns_on_duplicate_indicator_country_rows(tmp_path):
    """Duplicate (indicator, country) rows must produce UserWarning."""
    csv_text = (
        "ISO\tWEO Subject Code\t2015\t2016\n"
        "USA\tGGXWDG_NGDP\t100.0\t101.0\n"
        "USA\tGGXWDG_NGDP\t999.0\t999.0\n"  # duplicate
    )
    csv = tmp_path / "weo_dup.csv"
    csv.write_text(csv_text)
    with pytest.warns(UserWarning, match="2 rows"):
        inst = load(indicator="GGXWDG_NGDP", country="USA", csv_path=csv)
    # First row wins.
    assert inst.series.loc[pd.Timestamp("2015-01-01")] == pytest.approx(100.0)
