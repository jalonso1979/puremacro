"""Tests for puremacro.climate.annual_lp."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _synthetic_panel(n_regions: int = 8, n_years: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_regions):
        for y in range(2000, 2000 + n_years):
            cdd = abs(rng.normal(loc=80, scale=15))
            hdd = abs(rng.normal(loc=140, scale=20))
            response = -0.001 * cdd + 0.0005 * hdd + rng.normal(scale=0.05)
            rows.append({
                "region": f"R{r}",
                "year": y,
                "log_lf": response,
                "annual_cdd": cdd,
                "annual_hdd": hdd,
            })
    return pd.DataFrame(rows)


def test_returns_cdd_and_hdd_keys():
    from puremacro.climate.annual_lp import climate_annual_lp
    df = _synthetic_panel(seed=1)
    out = climate_annual_lp(df, response="log_lf", horizons=range(0, 5), n_lags=1)
    assert set(out.keys()) == {"cdd", "hdd"}


def test_each_lp_dataframe_has_expected_columns():
    from puremacro.climate.annual_lp import climate_annual_lp
    df = _synthetic_panel(seed=2)
    out = climate_annual_lp(df, response="log_lf", horizons=range(0, 5), n_lags=1)
    expected = {"h", "beta", "se", "t", "lo", "hi"}
    assert expected.issubset(set(out["cdd"].columns))
    assert expected.issubset(set(out["hdd"].columns))


def test_horizon_count_matches_arg():
    from puremacro.climate.annual_lp import climate_annual_lp
    df = _synthetic_panel(seed=3)
    horizons = list(range(0, 7))
    out = climate_annual_lp(df, response="log_lf", horizons=horizons, n_lags=1)
    assert len(out["cdd"]) == len(horizons)
    assert len(out["hdd"]) == len(horizons)


def test_controls_forwarded_to_panel_lp_dk():
    from puremacro.climate.annual_lp import climate_annual_lp
    df = _synthetic_panel(seed=4)
    df["gdp_growth"] = np.random.default_rng(99).normal(size=len(df))
    out_no_ctrl = climate_annual_lp(
        df, response="log_lf", horizons=range(0, 4), n_lags=1
    )
    out_with_ctrl = climate_annual_lp(
        df, response="log_lf", horizons=range(0, 4), n_lags=1,
        controls=("gdp_growth",),
    )
    diff = (out_no_ctrl["cdd"]["beta"] - out_with_ctrl["cdd"]["beta"]).abs().sum()
    assert diff > 1e-8
