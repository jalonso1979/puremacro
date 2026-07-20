"""Tests for puremacro.climate.monthly_dl."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _synthetic_single_region(
    T: int = 600, cdd_true: float = 0.05, hdd_true: float = -0.03, seed: int = 0,
) -> pd.DataFrame:
    """Monthly DGP with known shock betas + month and year FE noise."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("1970-01-01")
    dates = pd.date_range(start, periods=T, freq="MS")
    cdd = abs(rng.normal(loc=80, scale=15, size=T))
    hdd = abs(rng.normal(loc=140, scale=20, size=T))
    month_fe = np.tile(rng.normal(scale=0.1, size=12), T // 12 + 1)[:T]
    year_fe = np.repeat(rng.normal(scale=0.05, size=T // 12 + 1), 12)[:T]
    eps = rng.normal(scale=0.01, size=T)
    log_births = cdd_true * cdd + hdd_true * hdd + month_fe + year_fe + eps
    return pd.DataFrame({
        "date": dates,
        "calendar_month": dates.month,
        "year": dates.year,
        "cdd": cdd,
        "hdd": hdd,
        "log_births": log_births,
    })


def test_make_dl_lags_creates_correct_columns():
    from puremacro.climate.monthly_dl import make_dl_lags
    df = _synthetic_single_region(T=50)
    out = make_dl_lags(df, cols=["cdd", "hdd"], n_lags=3, sort_by=["date"])
    for col in ["cdd_lag1", "cdd_lag2", "cdd_lag3", "hdd_lag1", "hdd_lag2", "hdd_lag3"]:
        assert col in out.columns, f"missing {col}"
    # First n_lags rows should have NaN in the lag columns.
    assert out["cdd_lag1"].isna().sum() == 1
    assert out["cdd_lag3"].isna().sum() == 3


def test_recovers_known_betas_on_synthetic_data():
    from puremacro.climate.monthly_dl import monthly_dl
    df = _synthetic_single_region(cdd_true=0.05, hdd_true=-0.03)
    out = monthly_dl(
        df, shock_cols=("cdd", "hdd"), response_col="log_births",
        n_lags=0,
    )
    assert abs(out["cdd_betas"][0] - 0.05) < 0.01, f"cdd β recovered: {out['cdd_betas'][0]}"
    assert abs(out["hdd_betas"][0] - (-0.03)) < 0.01, f"hdd β recovered: {out['hdd_betas'][0]}"


def test_n_lags_zero_returns_contemporaneous_only():
    from puremacro.climate.monthly_dl import monthly_dl
    df = _synthetic_single_region(T=200)
    out = monthly_dl(
        df, shock_cols=("cdd", "hdd"), response_col="log_births", n_lags=0,
    )
    assert len(out["cdd_betas"]) == 1
    assert len(out["hdd_betas"]) == 1


def test_panel_mode_with_region_col():
    from puremacro.climate.monthly_dl import monthly_dl
    rng = np.random.default_rng(0)
    rows = []
    for r in range(4):
        for t, d in enumerate(pd.date_range("1970-01-01", periods=600, freq="MS")):
            cdd = abs(rng.normal(loc=80, scale=15))
            hdd = abs(rng.normal(loc=140, scale=20))
            response = 0.04 * cdd - 0.02 * hdd + rng.normal(scale=0.02)
            rows.append({
                "region": f"R{r}",
                "date": d,
                "calendar_month": d.month,
                "year": d.year,
                "cdd": cdd, "hdd": hdd,
                "log_births": response,
            })
    df = pd.DataFrame(rows)
    out = monthly_dl(
        df, shock_cols=("cdd", "hdd"), response_col="log_births",
        n_lags=0, region_col="region", panel_fe="region",
    )
    assert "cdd_betas" in out
    assert "hdd_betas" in out
    assert out["n_regions"] == 4
    # Recovery within reasonable tolerance on T=2400 observations.
    assert abs(out["cdd_betas"][0] - 0.04) < 0.01
    assert abs(out["hdd_betas"][0] - (-0.02)) < 0.01


def test_biological_benchmark_equals_first_shock_sum():
    from puremacro.climate.monthly_dl import monthly_dl
    df = _synthetic_single_region(T=200, seed=7)
    out = monthly_dl(
        df, shock_cols=("cdd", "hdd"), response_col="log_births", n_lags=3,
    )
    assert out["biological_benchmark"] == pytest.approx(sum(out["cdd_betas"]))
