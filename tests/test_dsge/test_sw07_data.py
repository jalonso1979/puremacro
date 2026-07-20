"""Test the bundled SW07 dataset loader."""
import importlib.resources

import pandas as pd
import pytest


def test_bundled_sw07_data_loads():
    """The CSV is accessible via importlib.resources and parses cleanly."""
    pkg = importlib.resources.files("puremacro.dsge")
    csv_path = pkg / "_sw07_data.csv"
    assert csv_path.is_file()
    df = pd.read_csv(csv_path, comment="#", parse_dates=["date"], index_col="date")
    assert df.shape[0] >= 150 and df.shape[0] <= 160
    assert set(df.columns) == {
        "gdp_growth", "cons_growth", "inv_growth",
        "wage_growth", "log_hours", "infl", "ffr",
    }
    assert df.notna().all().all()


def test_bundled_sw07_data_date_range():
    pkg = importlib.resources.files("puremacro.dsge")
    df = pd.read_csv(pkg / "_sw07_data.csv", comment="#", parse_dates=["date"], index_col="date")
    assert df.index.min() >= pd.Timestamp("1966-01-01")
    assert df.index.max() <= pd.Timestamp("2005-01-01")
