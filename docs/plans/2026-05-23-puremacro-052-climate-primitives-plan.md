# puremacro 0.52.0 Implementation Plan — climate × fertility primitives

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `puremacro.climate` — a new Pyodide-compatible subpackage with degree-days, climate-shock annual LP, mediation LP, and monthly distributed-lag estimators, extracted from `My Drive/Fertility/climate_fertility`.

**Architecture:** Four small files under `puremacro/climate/` with single responsibilities. `degree_days.py` is pure pandas. `annual_lp.py` delegates to existing `puremacro.lp.panel_lp_dk`. `mediation.py` builds on `annual_lp.py`. `monthly_dl.py` is the only standalone numerical estimator (HC1 + cluster SE). Additive — no existing public symbols change.

**Tech Stack:** numpy, pandas. No new dependencies. Python ≥3.10.

**Spec:** `docs/specs/2026-05-23-puremacro-052-climate-primitives-design.md`

---

## File map

### New files
- `puremacro/climate/__init__.py` — exports + `__all__`.
- `puremacro/climate/degree_days.py` (~60 LOC).
- `puremacro/climate/annual_lp.py` (~80 LOC).
- `puremacro/climate/mediation.py` (~80 LOC).
- `puremacro/climate/monthly_dl.py` (~200 LOC).
- `tests/test_climate/__init__.py` (empty).
- `tests/test_climate/test_degree_days.py` (~4 tests).
- `tests/test_climate/test_climate_annual_lp.py` (~4 tests).
- `tests/test_climate/test_climate_mediation.py` (~4 tests).
- `tests/test_climate/test_monthly_dl.py` (~5 tests).

### Modified files
- `puremacro/__init__.py` — bump `__version__` to `"0.52.0"`.
- `pyproject.toml` — bump `version` to `"0.52.0"`.
- `CHANGELOG.md` — add 0.52.0 section at top.
- `tests/test_import.py` — bump pinned version to `"0.52.0"`.
- `tests/fixtures/public_api_snapshot.json` — regenerate.

### Verified API surfaces
- `puremacro.lp.panel_lp_dk(df_wide, y, x, *, horizons, n_lags, controls, alpha, entity_level, time_level) -> DataFrame` — input is `df_wide` keyed by **MultiIndex (entity, time)** with columns `y`, `x`, `controls`. Returns long DataFrame with columns `[h, beta, se, t, lo, hi]`.
- Source reference: `My Drive/Fertility/climate_fertility/{degree_days.py, monthly_regression.py, estimation/annual_lp.py, estimation/mediation.py}`.

---

## Task 1: degree_days + package skeleton

**Files:**
- Create: `puremacro/climate/__init__.py` (empty docstring, will fill in Task 7).
- Create: `puremacro/climate/degree_days.py`.
- Create: `tests/test_climate/__init__.py` (empty).
- Create: `tests/test_climate/test_degree_days.py`.

- [ ] **Step 1: Create the empty package files**

```bash
mkdir -p puremacro/climate tests/test_climate
```

Then create `puremacro/climate/__init__.py`:
```python
"""Climate × fertility primitives extracted from My Drive/Fertility/climate_fertility.

The source project remains the canonical full-pipeline implementation
(including xarray-based weather loaders, geopandas zonal aggregation,
and country-specific runners). This subpackage exposes only the
Pyodide-compatible estimator primitives.
"""
```

And `tests/test_climate/__init__.py`:
```python
```
(empty file)

- [ ] **Step 2: Write 4 failing tests**

Create `tests/test_climate/test_degree_days.py`:
```python
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
```

- [ ] **Step 3: Run tests, expect FAIL**

Run: `pytest tests/test_climate/test_degree_days.py -v`
Expected: `ImportError: cannot import 'compute_monthly_cdd_hdd' from 'puremacro.climate.degree_days'`

- [ ] **Step 4: Implement degree_days.py**

Create `puremacro/climate/degree_days.py`:
```python
"""Cooling- and heating-degree-day construction from monthly temperatures."""
from __future__ import annotations

import math

import pandas as pd


def compute_monthly_cdd_hdd(
    df: pd.DataFrame,
    *,
    temp_col: str = "temp_c",
    threshold: float = 18.0,
) -> pd.DataFrame:
    """Return a copy of df with two new columns:

        cdd = max(temp - threshold, 0)
        hdd = max(threshold - temp, 0)

    Parameters
    ----------
    df : DataFrame containing ``temp_col``.
    temp_col : column name in Celsius.
    threshold : base temperature for the degree-day cut-off.

    Raises
    ------
    KeyError if ``temp_col`` is not a column of df.
    ValueError if ``threshold`` is not finite.
    """
    if temp_col not in df.columns:
        raise KeyError(f"compute_monthly_cdd_hdd: expected column {temp_col!r} in df")
    if not math.isfinite(threshold):
        raise ValueError(f"compute_monthly_cdd_hdd: threshold must be finite, got {threshold}")
    out = df.copy()
    out["cdd"] = (out[temp_col] - threshold).clip(lower=0.0)
    out["hdd"] = (threshold - out[temp_col]).clip(lower=0.0)
    return out


def compute_annual_cdd_hdd(
    df: pd.DataFrame,
    *,
    temp_col: str = "temp_c",
    threshold: float = 18.0,
    region_col: str = "region",
    year_col: str = "year",
    month_col: str = "month",
) -> pd.DataFrame:
    """Aggregate monthly degree-days to annual by summing across months.

    If ``cdd``/``hdd`` columns are not already present, compute them via
    ``compute_monthly_cdd_hdd``.

    Returns
    -------
    DataFrame with columns ``[region_col, year_col, annual_cdd, annual_hdd]``.
    """
    if "cdd" not in df.columns or "hdd" not in df.columns:
        df = compute_monthly_cdd_hdd(df, temp_col=temp_col, threshold=threshold)
    for col in (region_col, year_col, month_col):
        if col not in df.columns:
            raise KeyError(f"compute_annual_cdd_hdd: expected column {col!r} in df")
    annual = (
        df.groupby([region_col, year_col], observed=True)[["cdd", "hdd"]]
        .sum()
        .reset_index()
        .rename(columns={"cdd": "annual_cdd", "hdd": "annual_hdd"})
    )
    return annual


__all__ = ["compute_monthly_cdd_hdd", "compute_annual_cdd_hdd"]
```

- [ ] **Step 5: Run tests, expect PASS**

Run: `pytest tests/test_climate/test_degree_days.py -v`
Expected: 4/4 PASS.

- [ ] **Step 6: Commit**

```bash
git add puremacro/climate/__init__.py puremacro/climate/degree_days.py tests/test_climate/__init__.py tests/test_climate/test_degree_days.py
git commit -m "feat(climate): degree-days package skeleton + CDD/HDD helpers"
```

---

## Task 2: climate_annual_lp

**Files:**
- Create: `puremacro/climate/annual_lp.py`.
- Create: `tests/test_climate/test_climate_annual_lp.py`.

- [ ] **Step 1: Write 4 failing tests**

Create `tests/test_climate/test_climate_annual_lp.py`:
```python
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
            # Outcome: small negative response to cdd, positive to hdd, plus noise.
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
    # Add an extra control column.
    df["gdp_growth"] = np.random.default_rng(99).normal(size=len(df))
    out_no_ctrl = climate_annual_lp(
        df, response="log_lf", horizons=range(0, 4), n_lags=1
    )
    out_with_ctrl = climate_annual_lp(
        df, response="log_lf", horizons=range(0, 4), n_lags=1,
        controls=("gdp_growth",),
    )
    # Coefficients should differ when a real (random) regressor is added.
    diff = (out_no_ctrl["cdd"]["beta"] - out_with_ctrl["cdd"]["beta"]).abs().sum()
    assert diff > 1e-8
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `pytest tests/test_climate/test_climate_annual_lp.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement annual_lp.py**

Create `puremacro/climate/annual_lp.py`:
```python
"""Annual panel LP for climate (CDD, HDD) shocks.

Delegates to ``puremacro.lp.panel_lp_dk`` (Driscoll-Kraay HAC SE) twice
— once for the CDD shock with HDD as a control, once for HDD with CDD
as a control. This partials out the joint heating/cooling comovement
so each reported IRF is the *partial* response to that shock.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

from ..lp.panel_dk import panel_lp_dk


def climate_annual_lp(
    panel: pd.DataFrame,
    *,
    response: str,
    cdd_col: str = "annual_cdd",
    hdd_col: str = "annual_hdd",
    horizons: Iterable[int] = range(0, 11),
    n_lags: int = 2,
    controls: Sequence[str] = (),
    region_col: str = "region",
    year_col: str = "year",
    alpha: float = 0.10,
) -> dict:
    """Run two panel LPs (CDD shock, HDD shock) on the same panel.

    The input ``panel`` is long-form with one row per (region, year) and
    must contain ``response``, ``cdd_col``, ``hdd_col``, and all of
    ``controls``. Internally converted to the MultiIndex shape that
    ``panel_lp_dk`` expects.

    Returns
    -------
    dict with keys ``'cdd'`` and ``'hdd'``. Each value is the long
    DataFrame returned by ``panel_lp_dk`` with columns
    ``[h, beta, se, t, lo, hi]``.
    """
    horizons = list(horizons)
    controls = list(controls)
    needed_cols = {response, cdd_col, hdd_col, region_col, year_col} | set(controls)
    missing = needed_cols - set(panel.columns)
    if missing:
        raise KeyError(f"climate_annual_lp: panel missing columns {sorted(missing)}")
    # Build MultiIndex (entity, time) DataFrame as panel_lp_dk expects.
    wide = (
        panel[list(needed_cols)]
        .dropna()
        .copy()
        .set_index([region_col, year_col])
        .sort_index()
    )
    wide.index.names = ["code", "date"]
    cdd_result = panel_lp_dk(
        wide, y=response, x=cdd_col,
        horizons=horizons, n_lags=n_lags,
        controls=[hdd_col] + controls,
        alpha=alpha,
        entity_level="code", time_level="date",
    )
    hdd_result = panel_lp_dk(
        wide, y=response, x=hdd_col,
        horizons=horizons, n_lags=n_lags,
        controls=[cdd_col] + controls,
        alpha=alpha,
        entity_level="code", time_level="date",
    )
    return {"cdd": cdd_result, "hdd": hdd_result}


__all__ = ["climate_annual_lp"]
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `pytest tests/test_climate/test_climate_annual_lp.py -v`
Expected: 4/4 PASS.

- [ ] **Step 5: Commit**

```bash
git add puremacro/climate/annual_lp.py tests/test_climate/test_climate_annual_lp.py
git commit -m "feat(climate): climate_annual_lp — paired CDD/HDD panel LP"
```

---

## Task 3: climate_mediation_lp

**Files:**
- Create: `puremacro/climate/mediation.py`.
- Create: `tests/test_climate/test_climate_mediation.py`.

- [ ] **Step 1: Write 4 failing tests**

Create `tests/test_climate/test_climate_mediation.py`:
```python
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
            mediator = rng.normal(loc=0.02, scale=0.05)  # housing-growth-like
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
    # In each year, exactly 5 distinct bins should exist (25 regions / 5 = 5 per bin).
    df = df.assign(_q=q)
    for yr, g in df.groupby("year"):
        assert set(g["_q"].dropna().astype(int)) == {0, 1, 2, 3, 4}, (
            f"year {yr}: expected 5 bins, got {sorted(set(g['_q'].dropna().astype(int)))}"
        )


def test_mediation_share_zero_on_zero_baseline():
    from puremacro.climate.mediation import climate_mediation_lp
    # Construct a panel where the baseline IRF is degenerate near zero
    # so the share calculation must clamp without raising.
    df = _synthetic_panel(seed=3)
    df["log_lf"] = 0.0  # all responses identically zero → baseline IRF ~ 0
    out = climate_mediation_lp(
        df, mediator_col="housing_growth", response="log_lf",
        horizons=range(0, 4), n_lags=1,
    )
    # Shares should be finite and zero (not NaN/Inf).
    assert np.all(np.isfinite(out["mediation_share_cdd"]))
    assert np.all(out["mediation_share_cdd"] == 0.0)


def test_warns_when_mediator_all_nan_in_a_year():
    from puremacro.climate.mediation import _within_year_quintile
    df = _synthetic_panel(seed=4)
    # Wipe out the mediator for one year.
    df.loc[df["year"] == 2005, "housing_growth"] = np.nan
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _within_year_quintile(df, "housing_growth", "region", "year", n_bins=5)
    assert any("quintile" in str(wi.message).lower() or "bins" in str(wi.message).lower()
               for wi in w)
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `pytest tests/test_climate/test_climate_mediation.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement mediation.py**

Create `puremacro/climate/mediation.py`:
```python
"""Within-year-quintile mediation LP for climate shocks.

Splits regions into quintiles of a mediator (e.g., housing growth)
computed within each year, then runs two annual LPs: one baseline and
one with top-quintile × shock interactions added as controls. Reports
the per-horizon mediation share.
"""
from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd

from .annual_lp import climate_annual_lp


def _within_year_quintile(
    panel: pd.DataFrame,
    mediator_col: str,
    region_col: str,
    year_col: str,
    n_bins: int = 5,
) -> pd.Series:
    """Quintile bucket of the mediator's year-over-year growth per region,
    ranked within each year across regions.

    Returns a Series aligned with ``panel.index`` whose values are the
    integer bin index in ``[0, n_bins-1]`` or NaN when the mediator is
    NaN. Emits a warning when any year produces fewer than ``n_bins``
    buckets (pd.qcut(duplicates='drop') silently reduces bin count when
    too few distinct values exist).
    """
    df = panel[[region_col, year_col, mediator_col]].copy()
    df = df.sort_values([region_col, year_col])
    df["_g"] = df.groupby(region_col, observed=True)[mediator_col].pct_change()
    # Within-year cross-region quintile of growth.
    df["_q"] = df.groupby(year_col, observed=True)["_g"].transform(
        lambda s: pd.qcut(s, n_bins, labels=False, duplicates="drop")
    )
    # Warn whenever a year has fewer than n_bins distinct buckets.
    by_year = df.groupby(year_col, observed=True)["_q"].nunique(dropna=True)
    short_years = by_year[by_year < n_bins].index.tolist()
    if short_years:
        warnings.warn(
            f"_within_year_quintile: {len(short_years)} year(s) produced "
            f"fewer than {n_bins} quintile bins (first: {short_years[:3]}); "
            "mediator may have too few distinct values.",
            stacklevel=2,
        )
    return df["_q"].reindex(panel.index)


def climate_mediation_lp(
    panel: pd.DataFrame,
    *,
    mediator_col: str,
    response: str,
    cdd_col: str = "annual_cdd",
    hdd_col: str = "annual_hdd",
    horizons: Iterable[int] = range(0, 11),
    n_lags: int = 2,
    region_col: str = "region",
    year_col: str = "year",
    n_bins: int = 5,
    top_quintile_only: bool = True,
) -> dict:
    """Return baseline, interacted, and mediation-share LP results."""
    horizons = list(horizons)
    df = panel.copy()
    df["_q"] = _within_year_quintile(df, mediator_col, region_col, year_col, n_bins=n_bins)
    df["_top"] = (df["_q"] == (n_bins - 1)).astype(float)
    df["_cdd_top"] = df[cdd_col] * df["_top"]
    df["_hdd_top"] = df[hdd_col] * df["_top"]
    baseline = climate_annual_lp(
        df, response=response, cdd_col=cdd_col, hdd_col=hdd_col,
        horizons=horizons, n_lags=n_lags,
        region_col=region_col, year_col=year_col,
    )
    interacted = climate_annual_lp(
        df, response=response, cdd_col=cdd_col, hdd_col=hdd_col,
        horizons=horizons, n_lags=n_lags,
        region_col=region_col, year_col=year_col,
        controls=("_cdd_top", "_hdd_top", mediator_col),
    )
    def _share(baseline_df: pd.DataFrame, interacted_df: pd.DataFrame) -> np.ndarray:
        b = baseline_df["beta"].to_numpy()
        c = interacted_df["beta"].to_numpy()
        near_zero = np.abs(b) < 1e-12
        return np.where(near_zero, 0.0, (b - c) / np.where(near_zero, 1.0, b))
    return {
        "baseline": baseline,
        "interacted": interacted,
        "mediation_share_cdd": _share(baseline["cdd"], interacted["cdd"]),
        "mediation_share_hdd": _share(baseline["hdd"], interacted["hdd"]),
    }


__all__ = ["climate_mediation_lp"]
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `pytest tests/test_climate/test_climate_mediation.py -v`
Expected: 4/4 PASS.

- [ ] **Step 5: Commit**

```bash
git add puremacro/climate/mediation.py tests/test_climate/test_climate_mediation.py
git commit -m "feat(climate): climate_mediation_lp — within-year-quintile mediation LP"
```

---

## Task 4: make_dl_lags + monthly_dl single-region (HC1)

**Files:**
- Create: `puremacro/climate/monthly_dl.py`.
- Create: `tests/test_climate/test_monthly_dl.py`.

- [ ] **Step 1: Write 3 failing tests covering make_dl_lags + single-region monthly_dl**

Create `tests/test_climate/test_monthly_dl.py`:
```python
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
    # Contemporaneous coefficient is index 0 of each shock's betas list.
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
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `pytest tests/test_climate/test_monthly_dl.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement make_dl_lags + single-region monthly_dl (HC1)**

Create `puremacro/climate/monthly_dl.py`:
```python
"""Monthly distributed-lag estimator for climate-fertility analysis.

Two modes:
- Single-region (``region_col=None``): OLS with HC1 SE.
- Panel (``region_col`` set): OLS with cluster-robust SE by region.

Model:
    y_t = α + Σ_k Σ_s β_k^s · shock_s_{t-k}
          + month_FE (optional) + year_FE (optional)
          + region_FE (panel mode) + ε_t

The wide kwarg set lets country pipelines parameterise shock and
response columns without rewriting the estimator.
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd


def make_dl_lags(
    df: pd.DataFrame,
    *,
    cols: Sequence[str],
    n_lags: int,
    sort_by: Sequence[str],
) -> pd.DataFrame:
    """Add ``{col}_lag1..{col}_lag{n_lags}`` columns to a copy of df.

    ``sort_by`` controls within-group shifting. If ``len(sort_by) >= 2``,
    the first element is treated as the group key and shifts are computed
    within each group; otherwise plain ``.shift`` is used. df is sorted
    by ``sort_by`` before shifts.

    Returns a copy of df with the lag columns added.
    """
    out = df.sort_values(list(sort_by)).copy()
    group_key = sort_by[0] if len(sort_by) >= 2 else None
    for col in cols:
        for k in range(1, n_lags + 1):
            if group_key is None:
                out[f"{col}_lag{k}"] = out[col].shift(k)
            else:
                out[f"{col}_lag{k}"] = out.groupby(group_key, observed=True)[col].shift(k)
    return out


def _hc1_sandwich(X: np.ndarray, residuals: np.ndarray, XtX_inv: np.ndarray) -> np.ndarray:
    """HC1 heteroskedasticity-robust covariance."""
    n, k = X.shape
    Xe = X * residuals[:, None]
    meat = Xe.T @ Xe
    return (n / max(n - k, 1)) * XtX_inv @ meat @ XtX_inv


def _design_matrix(
    df: pd.DataFrame,
    *,
    shock_cols: Sequence[str],
    n_lags: int,
    add_month_fe: bool,
    add_year_fe: bool,
    region_col: str | None,
    panel_fe: str,
    month_col: str,
    year_col: str,
) -> tuple[np.ndarray, dict[str, slice]]:
    """Build the regression design matrix and a mapping
    ``{shock_col: slice_into_beta}`` so the estimator can extract each
    shock's coefficient block.
    """
    n = len(df)
    blocks: list[np.ndarray] = [np.ones((n, 1))]  # intercept
    col_slices: dict[str, slice] = {}
    offset = 1  # start past intercept
    for shock in shock_cols:
        cols_for_shock = [shock] + [f"{shock}_lag{k}" for k in range(1, n_lags + 1)]
        block = df[cols_for_shock].to_numpy(dtype=float)
        blocks.append(block)
        col_slices[shock] = slice(offset, offset + block.shape[1])
        offset += block.shape[1]
    # Fixed effects
    if region_col is not None:
        if panel_fe == "region_month":
            rm_key = (
                df[region_col].astype(str) + "_" + df[month_col].astype(int).astype(str)
            )
            rm = pd.get_dummies(rm_key, drop_first=True, dtype=float).to_numpy()
            blocks.append(rm)
        elif panel_fe == "region":
            r = pd.get_dummies(df[region_col], drop_first=True, dtype=float).to_numpy()
            blocks.append(r)
        else:
            raise ValueError(
                f"monthly_dl: panel_fe must be 'region_month' or 'region', got {panel_fe!r}"
            )
        if add_month_fe and panel_fe == "region":
            md = pd.get_dummies(df[month_col].astype(int), prefix="m",
                                drop_first=True, dtype=float).to_numpy()
            blocks.append(md)
    else:
        if add_month_fe:
            md = pd.get_dummies(df[month_col].astype(int), prefix="m",
                                drop_first=True, dtype=float).to_numpy()
            blocks.append(md)
    if add_year_fe:
        yd = pd.get_dummies(df[year_col].astype(int), prefix="y",
                            drop_first=True, dtype=float).to_numpy()
        blocks.append(yd)
    X = np.hstack(blocks)
    return X, col_slices


def monthly_dl(
    df: pd.DataFrame,
    *,
    shock_cols: Sequence[str] = ("cdd", "hdd"),
    response_col: str = "log_births",
    n_lags: int = 12,
    add_month_fe: bool = True,
    add_year_fe: bool = True,
    region_col: str | None = None,
    panel_fe: str = "region_month",
    month_col: str = "calendar_month",
    year_col: str = "year",
) -> dict:
    """Estimate the distributed-lag model.

    Returns dict with per-shock coefficient arrays + SEs, R², n_obs,
    and a ``biological_benchmark`` field (sum of first shock's betas)
    for backward compatibility with the climate_fertility source.
    """
    # 1. Build lags WITHIN regions if panel; else globally.
    sort_by = [region_col, year_col, month_col] if region_col else [year_col, month_col]
    df_lagged = make_dl_lags(df, cols=list(shock_cols), n_lags=n_lags, sort_by=sort_by)
    needed_cols = [response_col, month_col, year_col]
    if region_col is not None:
        needed_cols.append(region_col)
    for shock in shock_cols:
        needed_cols.append(shock)
        for k in range(1, n_lags + 1):
            needed_cols.append(f"{shock}_lag{k}")
    data = df_lagged[needed_cols].dropna()
    if data.empty:
        raise ValueError("monthly_dl: no observations after dropna")
    # 2. Design matrix.
    X, col_slices = _design_matrix(
        data, shock_cols=shock_cols, n_lags=n_lags,
        add_month_fe=add_month_fe, add_year_fe=add_year_fe,
        region_col=region_col, panel_fe=panel_fe,
        month_col=month_col, year_col=year_col,
    )
    y = data[response_col].to_numpy(dtype=float)
    n, k = X.shape
    if n_lags >= len(data):
        raise ValueError(
            f"monthly_dl: n_lags={n_lags} exceeds usable T={len(data)} after lag construction"
        )
    # 3. OLS via pinv (rank-tolerant).
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    residuals = y - X @ beta
    # 4. SE: HC1 for single-region; cluster-by-region for panel.
    if region_col is None:
        V = _hc1_sandwich(X, residuals, XtX_inv)
    else:
        regions = data[region_col].to_numpy()
        unique_regions = np.unique(regions)
        G = len(unique_regions)
        meat = np.zeros((k, k))
        for region in unique_regions:
            mask = regions == region
            score_g = X[mask].T @ residuals[mask]
            meat += np.outer(score_g, score_g)
        correction = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
        V = correction * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    # 5. Compose return dict.
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    out: dict = {
        "r_squared": float(r_squared),
        "n_obs": int(n),
    }
    if region_col is not None:
        out["n_regions"] = int(len(np.unique(data[region_col])))
    first_shock_sum: float | None = None
    for shock in shock_cols:
        sl = col_slices[shock]
        out[f"{shock}_betas"] = beta[sl].tolist()
        out[f"{shock}_ses"] = se[sl].tolist()
        if first_shock_sum is None:
            first_shock_sum = float(sum(beta[sl]))
    out["biological_benchmark"] = float(first_shock_sum) if first_shock_sum is not None else 0.0
    return out


__all__ = ["monthly_dl", "make_dl_lags"]
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `pytest tests/test_climate/test_monthly_dl.py -v`
Expected: 3/3 PASS.

- [ ] **Step 5: Commit**

```bash
git add puremacro/climate/monthly_dl.py tests/test_climate/test_monthly_dl.py
git commit -m "feat(climate): monthly_dl single-region + make_dl_lags"
```

---

## Task 5: monthly_dl panel mode + biological_benchmark backward-compat

**Files:**
- Test: `tests/test_climate/test_monthly_dl.py` (append 2 tests).

- [ ] **Step 1: Append 2 more tests**

Append to `tests/test_climate/test_monthly_dl.py`:
```python
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
```

- [ ] **Step 2: Run tests, expect PASS** (implementation already handles panel mode + biological_benchmark; this is a verification task)

Run: `pytest tests/test_climate/test_monthly_dl.py -v`
Expected: 5/5 PASS.

If anything fails, fix the implementation in `puremacro/climate/monthly_dl.py` and re-run before committing.

- [ ] **Step 3: Commit**

```bash
git add tests/test_climate/test_monthly_dl.py
git commit -m "test(climate): panel monthly_dl + biological_benchmark backward-compat"
```

---

## Task 6: Wire exports in puremacro/climate/__init__.py

**Files:**
- Modify: `puremacro/climate/__init__.py`.
- Test: `tests/test_climate/` (re-run full suite).

- [ ] **Step 1: Update `puremacro/climate/__init__.py`**

Replace the contents with:
```python
"""Climate × fertility primitives extracted from My Drive/Fertility/climate_fertility.

The source project remains the canonical full-pipeline implementation
(including xarray-based weather loaders, geopandas zonal aggregation,
and country-specific runners). This subpackage exposes only the
Pyodide-compatible estimator primitives:

- degree-days (CDD / HDD construction from monthly temperatures)
- annual climate-shock LP (paired CDD + HDD, Driscoll-Kraay HAC SE)
- within-year-quintile mediation LP
- monthly distributed-lag estimator (HC1 single-region; cluster panel)
"""
from .degree_days import compute_monthly_cdd_hdd, compute_annual_cdd_hdd
from .annual_lp import climate_annual_lp
from .mediation import climate_mediation_lp
from .monthly_dl import monthly_dl, make_dl_lags

__all__ = [
    "compute_monthly_cdd_hdd",
    "compute_annual_cdd_hdd",
    "climate_annual_lp",
    "climate_mediation_lp",
    "monthly_dl",
    "make_dl_lags",
]
```

- [ ] **Step 2: Verify the package imports cleanly**

Run: `python -c "import puremacro.climate as c; print(sorted(c.__all__))"`
Expected output:
```
['climate_annual_lp', 'climate_mediation_lp', 'compute_annual_cdd_hdd', 'compute_monthly_cdd_hdd', 'make_dl_lags', 'monthly_dl']
```

- [ ] **Step 3: Run the full test_climate suite**

Run: `pytest tests/test_climate/ -v`
Expected: 17/17 PASS.

- [ ] **Step 4: Commit**

```bash
git add puremacro/climate/__init__.py
git commit -m "feat(climate): wire public exports for puremacro.climate"
```

---

## Task 7: Version bump + CHANGELOG entry

**Files:**
- Modify: `puremacro/__init__.py`.
- Modify: `pyproject.toml`.
- Modify: `CHANGELOG.md`.
- Modify: `tests/test_import.py`.

- [ ] **Step 1: Bump `puremacro/__init__.py`**

Change `__version__ = "0.51.0"` to `__version__ = "0.52.0"`.

- [ ] **Step 2: Bump `pyproject.toml`**

In the `[project]` block, change `version = "0.51.0"` to `version = "0.52.0"`.

- [ ] **Step 3: Bump the pinned version in `tests/test_import.py`**

Change `assert puremacro.__version__ == "0.51.0"` to `assert puremacro.__version__ == "0.52.0"`.

- [ ] **Step 4: Add CHANGELOG entry**

Insert into `CHANGELOG.md` after the `# Changelog` heading + preamble and before the existing `## 0.51.0 — 2026-05-23` entry:

```markdown
## 0.52.0 — 2026-05-23

Climate × fertility primitives. R2 from the 2026-05-23 research-directions
brainstorm: extracted Pyodide-compatible estimators from
``My Drive/Fertility/climate_fertility`` into a new ``puremacro.climate``
subpackage. The source project remains the canonical full-pipeline
implementation (xarray weather loaders, geopandas zonal aggregation,
country-specific runners). This release exposes only the reusable
estimator primitives.

### Added
- `puremacro.climate` subpackage:
  - `compute_monthly_cdd_hdd(df, *, temp_col='temp_c', threshold=18.0)`,
    `compute_annual_cdd_hdd(df, *, temp_col, threshold, region_col,
    year_col, month_col)` — degree-day construction + annual aggregation.
  - `climate_annual_lp(panel, *, response, cdd_col, hdd_col, horizons,
    n_lags, controls, region_col, year_col, alpha) -> dict` — paired
    CDD + HDD panel LP via Driscoll-Kraay (delegates to
    `puremacro.lp.panel_lp_dk`).
  - `climate_mediation_lp(panel, *, mediator_col, response, ..., n_bins,
    top_quintile_only) -> dict` — within-year-quintile mediation LP.
  - `monthly_dl(df, *, shock_cols, response_col, n_lags, add_month_fe,
    add_year_fe, region_col, panel_fe, month_col, year_col) -> dict` —
    distributed-lag estimator (HC1 single-region; cluster-by-region in
    panel mode).
  - `make_dl_lags(df, *, cols, n_lags, sort_by)` — within-group lag
    construction helper.

### Internal
- New module ~420 LOC across four files.
- 17 new unit tests in `tests/test_climate/`.
- No new dependencies (numpy + pandas only).

### Provenance
The primitives are reimplementations (not direct lifts) of analogous
estimators in `My Drive/Fertility/climate_fertility/`. Notable
deliberate differences: `monthly_dl` exposes `shock_cols`,
`response_col`, FE toggles, and `region_col` as wired kwargs (the
source's `estimate_distributed_lag` reserved these as future-Plan
documentation only).
```

- [ ] **Step 5: Smoke check the version bump**

Run: `python -c "import puremacro; assert puremacro.__version__ == '0.52.0'; print(puremacro.__version__)"`
Expected output: `0.52.0`.

- [ ] **Step 6: Commit**

```bash
git add puremacro/__init__.py pyproject.toml CHANGELOG.md tests/test_import.py
git commit -m "chore(puremacro): bump 0.51.0 → 0.52.0 (climate × fertility primitives)"
```

---

## Task 8: Regenerate the public-API snapshot

**Files:**
- Modify: `tests/fixtures/public_api_snapshot.json`.

- [ ] **Step 1: Locate the snapshot test + fail it intentionally**

Run: `pytest tests/ -k "public_api" -v 2>&1 | tail -30`
Expected: FAIL — the snapshot is missing the new `puremacro.climate` entries.

- [ ] **Step 2: Find the snapshot file**

Run: `grep -rln "public_api_snapshot" tests/ tools/ 2>&1 | head -3`
The file is `tests/fixtures/public_api_snapshot.json`.

- [ ] **Step 3: Find the snapshot-regeneration helper (if any)**

Run: `find tools -name "*snapshot*.py" -o -name "*public_api*.py" 2>&1 | head -3`

If a helper script exists, run it. If not, regenerate manually by reading
the snapshot JSON, adding the new entries:

Under the `all` section, add:
```
"puremacro.climate": ["climate_annual_lp", "climate_mediation_lp", "compute_annual_cdd_hdd", "compute_monthly_cdd_hdd", "make_dl_lags", "monthly_dl"],
"puremacro.climate.annual_lp": ["climate_annual_lp"],
"puremacro.climate.degree_days": ["compute_annual_cdd_hdd", "compute_monthly_cdd_hdd"],
"puremacro.climate.mediation": ["climate_mediation_lp"],
"puremacro.climate.monthly_dl": ["make_dl_lags", "monthly_dl"],
```
in alphabetical order matching the existing sort convention.

- [ ] **Step 4: Re-run the snapshot test, expect PASS**

Run: `pytest tests/ -k "public_api" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/public_api_snapshot.json
git commit -m "chore(tests): regenerate public_api_snapshot for 0.52.0 additions"
```

---

## Task 9: Run the 6-gate release check

**Files:** none modified — verification only.

- [ ] **Step 1: Run gates 1-4**

Run: `python tools/release_check.py`
Expected: `all 4 gates PASS`.

- [ ] **Step 2: Run gate 5 (examples gallery)**

Run: `python tools/release_check.py --examples`
Expected: gate 5 PASS (58 PASS, 4 SKIP, 0 FAIL) — likely emits a "stale" advisory because new source files exist; that's a warning, not a fail.

If the stale advisory fires and you want to clear it, regenerate the gallery:
```bash
python tools/render_examples_gallery.py
```
Then if `hfi_gertler_karadi` flakes with a timeout (a known intermittent), restore its previous PASS entry from the prior commit. The integrated commit pattern is documented in commit `e569dee` from the 0.51.0 release.

- [ ] **Step 3: Run gate 6 (Pyodide smoke)**

Run: `python tools/release_check.py --pyodide`
Expected: gate 6 PASS (8 passed in Pyodide).

- [ ] **Step 4: Final integrated 6-gate check**

Run: `python tools/release_check.py --examples --pyodide`
Expected: `all 6 gates PASS`.

If any gate fails:
- Diagnose the failure (don't `--no-verify` or skip hooks).
- Fix the underlying issue.
- Re-run only the failing gate.

---

## Self-review checklist (run AFTER all 9 tasks)

1. **Spec coverage:**
   - Component A (`degree_days`): Task 1 ✓
   - Component B (`annual_lp`): Task 2 ✓
   - Component C (`mediation`): Task 3 ✓
   - Component D (`monthly_dl` + `make_dl_lags`): Tasks 4 + 5 ✓
   - Public exports: Task 6 ✓
   - Version + CHANGELOG: Task 7 ✓
   - Snapshot: Task 8 ✓
   - Release gates: Task 9 ✓
   - All 9 acceptance criteria map to a task.

2. **Placeholder scan:** None — every step has runnable code or a concrete command.

3. **Type consistency:**
   - `monthly_dl` return-dict keys (`{shock}_betas`, `{shock}_ses`, `r_squared`, `n_obs`, `biological_benchmark`, optional `n_regions`) match between Task 4 (implementation) and Tasks 4–5 (test assertions).
   - `climate_annual_lp` returns `{'cdd': df, 'hdd': df}` — both `mediation` (Task 3) and the annual_lp tests (Task 2) consume that shape.
   - `climate_mediation_lp` returns `{'baseline', 'interacted', 'mediation_share_cdd', 'mediation_share_hdd'}` — tests assert this set.
   - `_within_year_quintile` returns a Series with bin indices `[0, n_bins-1]` — `climate_mediation_lp` consumes via `df["_q"] == (n_bins - 1)`.
   - `make_dl_lags(sort_by=[...])` is invoked with `sort_by=["date"]` in single-region tests and `sort_by=["region", "year", "month"]` in panel mode inside `monthly_dl`. Both paths covered.
