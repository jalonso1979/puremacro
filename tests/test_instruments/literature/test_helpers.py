"""Unit tests for puremacro.instruments.literature._helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.instruments import Instrument
from puremacro.instruments.literature._helpers import _csv_to_instrument


def test_rejects_both_date_col_and_year_month_cols():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2000-01-01"]),
        "year": [2000],
        "month": [1],
        "v": [1.0],
    })
    with pytest.raises(ValueError, match="not both"):
        _csv_to_instrument(
            df, name="x", source="x", frequency="M",
            value_col="v", date_col="date", year_col="year",
        )


def test_rejects_neither_date_spec():
    df = pd.DataFrame({"v": [1.0]})
    with pytest.raises(ValueError, match="either date_col"):
        _csv_to_instrument(
            df, name="x", source="x", frequency="M", value_col="v",
        )


def test_year_month_path_zero_pads_single_digit_months():
    df = pd.DataFrame({"y": [2000, 2000], "m": [1, 12], "v": [10.0, 20.0]})
    inst = _csv_to_instrument(
        df, name="x", source="x", frequency="M", value_col="v",
        year_col="y", month_col="m",
    )
    assert inst.series.loc[pd.Timestamp("2000-01-01")] == 10.0
    assert inst.series.loc[pd.Timestamp("2000-12-01")] == 20.0
