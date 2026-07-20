"""Tests for puremacro.climate.degree_days."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_at_threshold_both_zero():
    from puremacro.climate.degree_days import compute_monthly_cdd_hdd
    df = pd.DataFrame({"temp_c": [18.0, 18.0, 18.0]})
    out = compute_monthly_cdd_hdd(df, threshold=18.0)
    assert (out["cdd"] == 0.0).all()
    assert (out["hdd"] == 0.0).all()


def test_cooling_above_threshold():
    from puremacro.climate.degree_days import compute_monthly_cdd_hdd
    df = pd.DataFrame({"temp_c": [25.0]})
    out = compute_monthly_cdd_hdd(df, threshold=18.0)
    assert out["cdd"].iloc[0] == pytest.approx(7.0)
    assert out["hdd"].iloc[0] == pytest.approx(0.0)


def test_heating_below_threshold():
    from puremacro.climate.degree_days import compute_monthly_cdd_hdd
    df = pd.DataFrame({"temp_c": [10.0]})
    out = compute_monthly_cdd_hdd(df, threshold=18.0)
    assert out["hdd"].iloc[0] == pytest.approx(8.0)
    assert out["cdd"].iloc[0] == pytest.approx(0.0)


def test_annual_aggregation_sums_monthly():
    from puremacro.climate.degree_days import (
        compute_monthly_cdd_hdd, compute_annual_cdd_hdd
    )
    # Two regions, 12 months each in 2020; temperatures alternating 25 / 10.
    rows = []
    for region in ["A", "B"]:
        for month in range(1, 13):
            temp = 25.0 if month % 2 == 0 else 10.0
            rows.append({"region": region, "year": 2020, "month": month, "temp_c": temp})
    df = pd.DataFrame(rows)
    df = compute_monthly_cdd_hdd(df, threshold=18.0)
    annual = compute_annual_cdd_hdd(df, threshold=18.0)
    # 6 months × 7 cdd = 42 per region; 6 months × 8 hdd = 48 per region.
    assert set(annual["region"]) == {"A", "B"}
    assert (annual["annual_cdd"] == 42.0).all()
    assert (annual["annual_hdd"] == 48.0).all()
    assert set(annual.columns) == {"region", "year", "annual_cdd", "annual_hdd"}
