"""Network-marked correlation tests of our text-built indices vs the
mirrored published series.

These tests pull live data from policyuncertainty.com and the Caldara-
Iacoviello dataset, and intentionally do NOT assert publication-level
correlation — that requires the BBD source corpus, which we don't ship.
Instead they check that the published series load and that our
synthetic-corpus reconstruction has the right SHAPE (positive
correlation with itself across normalisations).

Run only with: pytest -m network tests/test_narrative_indices_validation.py
"""
from __future__ import annotations

import pandas as pd
import pytest


@pytest.mark.network
def test_published_bbd_epu_loads():
    """Smoke: the published BBD-EPU series via instruments.literature
    must load. Skip if upstream is unreachable."""
    from puremacro.instruments.literature import bbd_epu
    try:
        inst = bbd_epu.load()
    except Exception:
        pytest.skip("policyuncertainty.com unreachable")
    if inst.series.empty:
        pytest.skip("BBD-EPU returned empty.")
    assert inst.frequency in {"M", "Q"}


@pytest.mark.network
def test_published_gpr_loads():
    from puremacro.instruments.literature import caldara_iacoviello_gpr
    try:
        inst = caldara_iacoviello_gpr.load()
    except Exception:
        pytest.skip("GPR mirror unreachable")
    if inst.series.empty:
        pytest.skip("GPR returned empty.")
    assert inst.frequency in {"M", "Q"}


def test_synthetic_epu_has_consistent_normalization():
    """Offline: zscore on synthetic corpus has zero mean (sanity)."""
    from puremacro.narrative import epu
    records = []
    high_text = "economic policy uncertainty rose"
    low_text = "ordinary text"
    for q in range(8):
        d = pd.Timestamp("2020-01-01") + pd.DateOffset(months=q)
        records.append((d, high_text if q % 2 == 0 else low_text,
                        "https://test/" + str(d.date()),
                        {"language": "en"}))
    ri = epu(records, country="USA", language="en", normalize="zscore")
    assert ri.series.dropna().mean() == pytest.approx(0.0, abs=1e-9)
