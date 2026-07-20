"""Unit tests for puremacro.instruments._helpers (the promoted helpers)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.instruments import Instrument
from puremacro.instruments._helpers import _json_to_instrument, _csv_to_instrument


# --------------------------------------------------------------------------
# _csv_to_instrument promoted location — confirm it imports
# --------------------------------------------------------------------------
def test_csv_to_instrument_importable_from_promoted_path():
    """Confirm _csv_to_instrument lives at puremacro.instruments._helpers."""
    df = pd.DataFrame({"date": ["2000-01-01"], "v": [1.0]})
    inst = _csv_to_instrument(
        df, name="x", source="x", frequency="M",
        value_col="v", date_col="date",
    )
    assert isinstance(inst, Instrument)


def test_csv_to_instrument_legacy_literature_path_still_works():
    """The shim at literature/_helpers.py must continue to re-export."""
    from puremacro.instruments.literature._helpers import _csv_to_instrument as legacy
    df = pd.DataFrame({"date": ["2000-01-01"], "v": [1.0]})
    inst = legacy(df, name="x", source="x", frequency="M",
                  value_col="v", date_col="date")
    assert isinstance(inst, Instrument)


# --------------------------------------------------------------------------
# _json_to_instrument — new helper for FRED-style JSON
# --------------------------------------------------------------------------
def test_json_to_instrument_basic_shape():
    """FRED-style observations list → Instrument."""
    obs = [
        {"date": "2000-01-01", "value": "1.5"},
        {"date": "2000-02-01", "value": "1.7"},
        {"date": "2000-03-01", "value": "1.9"},
    ]
    inst = _json_to_instrument(
        obs, name="test_series", source="synthetic",
        frequency="M",
        date_field="date", value_field="value",
    )
    assert isinstance(inst, Instrument)
    assert inst.frequency == "M"
    assert inst.series.loc[pd.Timestamp("2000-01-01")] == 1.5
    assert len(inst.series) == 3


def test_json_to_instrument_handles_dot_missing_marker():
    """FRED uses '.' to mark missing values; the helper must coerce to NaN."""
    obs = [
        {"date": "2000-01-01", "value": "1.5"},
        {"date": "2000-02-01", "value": "."},
        {"date": "2000-03-01", "value": "1.9"},
    ]
    inst = _json_to_instrument(
        obs, name="test", source="synthetic", frequency="M",
        date_field="date", value_field="value",
    )
    assert pd.isna(inst.series.loc[pd.Timestamp("2000-02-01")])
    assert inst.series.dropna().shape[0] == 2


def test_json_to_instrument_empty_observations_raises():
    """Empty observation list → ValueError (not silent empty Instrument)."""
    with pytest.raises(ValueError, match="empty"):
        _json_to_instrument(
            [], name="x", source="x", frequency="M",
            date_field="date", value_field="value",
        )


def test_json_to_instrument_passes_metadata_through():
    obs = [{"date": "2000-01-01", "value": "1.0"}]
    inst = _json_to_instrument(
        obs, name="x", source="x", frequency="M",
        date_field="date", value_field="value",
        metadata={"reference": "test ref"},
    )
    assert inst.metadata.get("reference") == "test ref"
