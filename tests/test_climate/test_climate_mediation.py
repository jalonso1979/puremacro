"""Tests for puremacro.climate.mediation."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


def _synthetic_panel(n_regions: int = 25, n_years: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_regions):
        for y in range(2000, 2000 + n_years):
            cdd = abs(rng.normal(loc=80, scale=15))
            hdd = abs(rng.normal(loc=140, scale=20))
            mediator = rng.normal(loc=0.02, scale=0.05)
            response = -0.001 * cdd + 0.0005 * hdd + rng.normal(scale=0.05)
            rows.append({
                "region": f"R{r}",
                "year": y,
                "log_lf": response,
                "annual_cdd": cdd,
                "annual_hdd": hdd,
                "housing_growth": mediator,
            })
    return pd.DataFrame(rows)


def test_returns_expected_keys():
    from puremacro.climate.mediation import climate_mediation_lp
    df = _synthetic_panel(seed=1)
    out = climate_mediation_lp(
        df, mediator_col="housing_growth", response="log_lf",
        horizons=range(0, 4), n_lags=1,
    )
    expected = {"baseline", "interacted", "mediation_share_cdd", "mediation_share_hdd"}
    assert expected.issubset(out.keys())


def test_within_year_quintile_assigns_5_buckets():
    from puremacro.climate.mediation import _within_year_quintile
    df = _synthetic_panel(seed=2)
    q = _within_year_quintile(df, "housing_growth", "region", "year", n_bins=5)
    df = df.assign(_q=q)
    # Year 2000 is the first year per region; pct_change is all-NaN, so the
    # quintile is undefined. Subsequent years should each get 5 bins.
    first_year = df["year"].min()
    for yr, g in df.groupby("year"):
        bins = set(g["_q"].dropna().astype(int))
        if yr == first_year:
            assert bins == set(), f"first year should have no quintiles, got {bins}"
        else:
            assert bins == {0, 1, 2, 3, 4}, (
                f"year {yr}: expected 5 bins, got {sorted(bins)}"
            )


def test_mediation_share_zero_on_zero_baseline():
    """When the baseline IRF is numerically tiny (≪ 1e-12), the share
    is clamped to 0 — never NaN/Inf from divide-by-near-zero.

    Tests ``_mediation_share`` directly with synthetic near-zero beta
    vectors, since triggering |β| < 1e-12 through a live LP requires
    regressors scaled ~1e-12 which is degenerate for numerical OLS.
    """
    from puremacro.climate.mediation import _mediation_share
    # Construct DataFrames where baseline beta is near-zero (|b| < 1e-12)
    # and interacted beta is also tiny but possibly non-zero.
    baseline_df = pd.DataFrame({
        "h": [0, 1, 2, 3],
        "beta": [0.0, 1e-15, -5e-14, 0.0],
        "se": [0.01, 0.01, 0.01, 0.01],
    })
    interacted_df = pd.DataFrame({
        "h": [0, 1, 2, 3],
        "beta": [0.0, 2e-15, 1e-14, 0.0],
        "se": [0.01, 0.01, 0.01, 0.01],
    })
    share = _mediation_share(baseline_df, interacted_df)
    assert np.all(np.isfinite(share)), "shares must be finite"
    # All |baseline beta| < 1e-12 → clamp triggers → share = 0.
    assert np.all(share == 0.0), f"expected all zeros, got {share}"


def test_warns_when_mediator_all_nan_in_a_year():
    """Trigger the few-bins warning by collapsing the mediator's
    cross-region variation in one year. With <n_bins distinct growth
    values, pd.qcut(duplicates='drop') reduces bin count."""
    from puremacro.climate.mediation import _within_year_quintile
    df = _synthetic_panel(seed=4)
    # Make every region's mediator in BOTH 2004 and 2005 identical (0.05)
    # → pct_change into 2005 = 0.05/0.05 - 1 = 0 for every region
    # → exactly 1 distinct growth value in 2005 → qcut produces 1 bin
    # → counts_per_year[2005] > 0 and bins < 5 → warning fires.
    df.loc[df["year"].isin([2004, 2005]), "housing_growth"] = 0.05
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _within_year_quintile(df, "housing_growth", "region", "year", n_bins=5)
    assert any("quintile" in str(wi.message).lower() or "bins" in str(wi.message).lower()
               for wi in w)
