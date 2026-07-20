"""Tests for puremacro.instruments.literature.bloom_2009."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.instruments import Instrument
from puremacro.instruments.literature.bloom_2009 import load, BLOOM_2009_EVENTS


def test_load_returns_instrument():
    inst = load()
    assert isinstance(inst, Instrument)
    assert inst.category == "literature"
    assert inst.frequency == "M"
    assert inst.name == "bloom_2009_uncertainty"


def test_event_count_matches_appendix():
    """Bloom 2009 paper Table A.1 lists 17 large-uncertainty episodes."""
    assert len(BLOOM_2009_EVENTS) == 17


def test_event_months_are_indicator_one():
    """Series value is 1.0 at each event month, 0.0 elsewhere."""
    inst = load()
    for date in BLOOM_2009_EVENTS:
        ts = pd.Timestamp(date).to_period("M").to_timestamp()
        assert inst.series.loc[ts] == 1.0, f"event {date} not marked"


def test_non_event_months_are_zero():
    """Months outside the event list are 0.0 (not NaN)."""
    inst = load()
    quiet = pd.Timestamp("1995-02-01")
    assert inst.series.loc[quiet] == 0.0


def test_series_covers_full_sample():
    """Series spans Jan-1962 to Dec-2008 inclusive."""
    inst = load()
    assert inst.series.index.min() == pd.Timestamp("1962-01-01")
    assert inst.series.index.max() == pd.Timestamp("2008-12-01")
    assert len(inst.series) == 564


def test_series_sum_equals_event_count():
    """Sum of indicator series equals the documented event count."""
    inst = load()
    assert inst.series.sum() == 17.0


def test_metadata_includes_reference_and_event_dates():
    inst = load()
    assert "reference" in inst.metadata
    assert "event_dates" in inst.metadata
    assert len(inst.metadata["event_dates"]) == 17
