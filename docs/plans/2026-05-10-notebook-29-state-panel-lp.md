# Notebook 29 — State-Panel LP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Notebook 29 — state-level panel local projection of US labor outcomes on national LUI shocks. Pooled + per-state IRFs across 3 outcomes (urate, Δ₄ log NFP, LFPR), with Driscoll-Kraay SEs and heterogeneity splits.

**Architecture:** Two new submodules: `puremacro/fetch/bls_state_panel.py` (3 BLS state-panel fetchers) and `puremacro/regress/lp.py` (panel LP estimator with DK SEs). Pure numpy, pyodide-clean. Notebook 29 + paired builder + outputs (parquet/JSON/PDF).

**Tech Stack:** Python 3.10+, numpy, pandas, matplotlib. No new top-level deps.

**Spec reference:** `docs/specs/2026-05-10-notebook-29-state-panel-lp.md`.

**Branching:** Stay on `feature/narrative-extension-slice3` (current head fe8e3ed past v0.8.0).

**Pre-implementation baseline:** `pytest -q` after Slice 6a = **1014 passed, 27 skipped**, plus 1 pre-existing pyodide-compat failure.

---

## File Structure

### Files created
- `puremacro/regress/__init__.py`
- `puremacro/regress/lp.py`
- `puremacro/fetch/bls_state_panel.py`
- `puremacro/tests/test_regress_lp.py`
- `puremacro/tests/test_fetch_bls_state.py`
- `notebooks/29_state_panel_lp_lui.ipynb`
- `tools/make_notebook_29_state_panel_lp.py`
- `notebooks/output_tables/29_*.parquet`, `29_meta.json`
- `notebooks/output_figures/29_*.pdf`

### Files modified
- `puremacro/pyproject.toml`, `puremacro/puremacro/__init__.py`, `tests/test_import.py`, `puremacro/CHANGELOG.md` (0.8.0 → 0.9.0).

---

## Task 0: Branch + baseline

- [ ] **Step 1: Verify branch + baseline**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git branch --show-current   # feature/narrative-extension-slice3
git log --oneline -1        # fe8e3ed (T2 teaching outputs) past v0.8.0
cd puremacro && pytest -q --no-header 2>&1 | tail -3
```
Expected: 1014 passed, 27 skipped.

---

## Task 1: `puremacro/regress/lp.py` — panel LP estimator with Driscoll-Kraay SE

**Files:**
- Create: `puremacro/regress/__init__.py`
- Create: `puremacro/regress/lp.py`
- Create: `puremacro/tests/test_regress_lp.py`

- [ ] **Step 1: Write failing tests**

Create `puremacro/tests/test_regress_lp.py`:

```python
"""Tests for puremacro.regress.lp panel local projection estimator."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest


def _make_synthetic_panel(n_units=20, n_periods=80, beta=0.5, sigma=1.0, seed=0):
    """y_{i,t} = alpha_i + beta * shock_t + e_{i,t},  shock ~ N(0,1) iid."""
    rng = np.random.default_rng(seed)
    units = list(range(n_units))
    dates = pd.period_range("2006Q1", periods=n_periods, freq="Q").to_timestamp(how="end")
    shock = rng.standard_normal(n_periods)
    rows = []
    for i, u in enumerate(units):
        alpha_i = rng.standard_normal()
        e = rng.standard_normal(n_periods) * sigma
        y = alpha_i + beta * shock + e
        for t, d in enumerate(dates):
            rows.append({"unit": u, "date": d, "y": y[t], "shock": shock[t]})
    return pd.DataFrame(rows)


def test_lp_panel_recovers_known_beta_at_h0():
    """y_{i,t+0} = alpha_i + beta*shock_t + e: lp_panel should recover beta at h=0."""
    from puremacro.regress.lp import lp_panel
    df = _make_synthetic_panel(n_units=30, n_periods=200, beta=0.7, seed=1)
    out = lp_panel(df, y="y", shock="shock", horizons=range(0, 1),
                   unit="unit", date="date", se="driscoll_kraay")
    row = out.iloc[0]
    assert abs(row["beta"] - 0.7) < 0.05
    assert row["horizon"] == 0
    assert row["se"] > 0
    assert row["n_obs"] > 0


def test_lp_panel_horizon_decay_with_white_noise_shock():
    """For iid shock, response at h>=1 should be ~0 (within CI)."""
    from puremacro.regress.lp import lp_panel
    df = _make_synthetic_panel(n_units=30, n_periods=200, beta=0.5, seed=2)
    out = lp_panel(df, y="y", shock="shock", horizons=range(0, 5),
                   unit="unit", date="date", se="driscoll_kraay")
    # h=0 recovers beta
    assert abs(out.loc[out["horizon"] == 0, "beta"].iloc[0] - 0.5) < 0.05
    # h>=1: shock is iid, expected response is 0 (within ~3 SE).
    for h in range(1, 5):
        row = out.loc[out["horizon"] == h].iloc[0]
        assert abs(row["beta"]) < 0.10  # tolerance reflects sample noise


def test_lp_panel_unit_fe_absorbs_unit_means():
    """Adding a constant per unit shouldn't change estimated beta when unit_fe=True."""
    from puremacro.regress.lp import lp_panel
    df = _make_synthetic_panel(n_units=20, n_periods=100, beta=0.6, seed=3)
    df_shifted = df.copy()
    rng = np.random.default_rng(7)
    shifts = {u: 10 * rng.standard_normal() for u in df["unit"].unique()}
    df_shifted["y"] = df_shifted.apply(lambda r: r["y"] + shifts[r["unit"]], axis=1)
    out_a = lp_panel(df, y="y", shock="shock", horizons=range(0, 1),
                     unit="unit", date="date", unit_fe=True, se="driscoll_kraay")
    out_b = lp_panel(df_shifted, y="y", shock="shock", horizons=range(0, 1),
                     unit="unit", date="date", unit_fe=True, se="driscoll_kraay")
    assert abs(out_a["beta"].iloc[0] - out_b["beta"].iloc[0]) < 1e-8


def test_lp_panel_returns_required_columns():
    from puremacro.regress.lp import lp_panel
    df = _make_synthetic_panel(n_units=10, n_periods=40, seed=4)
    out = lp_panel(df, y="y", shock="shock", horizons=range(0, 3),
                   unit="unit", date="date", se="driscoll_kraay")
    for col in ("horizon", "beta", "se", "t", "p", "ci_lo", "ci_hi", "n_obs"):
        assert col in out.columns, f"missing column {col}"
    assert len(out) == 3


def test_lp_panel_dk_lag_auto_picks_h_plus_one():
    """When dk_lag=None, lp_panel uses h+1 as the truncation parameter."""
    from puremacro.regress.lp import _dk_default_lag
    assert _dk_default_lag(0) == 1
    assert _dk_default_lag(4) == 5
    assert _dk_default_lag(12) == 13


def test_lp_panel_with_controls():
    """Adding a control variable shouldn't blow up the estimator."""
    from puremacro.regress.lp import lp_panel
    df = _make_synthetic_panel(n_units=15, n_periods=80, beta=0.5, seed=5)
    rng = np.random.default_rng(11)
    df["x1"] = rng.standard_normal(len(df))
    out = lp_panel(df, y="y", shock="shock", horizons=range(0, 2),
                   unit="unit", date="date", controls=["x1"],
                   se="driscoll_kraay")
    assert abs(out.loc[out["horizon"] == 0, "beta"].iloc[0] - 0.5) < 0.10
    assert (out["se"] > 0).all()
```

- [ ] **Step 2: Run tests, expect failure**

```bash
cd puremacro && pytest tests/test_regress_lp.py -v --no-header 2>&1 | tail -10
```
Expected: 6 fail with `ModuleNotFoundError: No module named 'puremacro.regress'`.

- [ ] **Step 3: Create `puremacro/regress/__init__.py`**

```python
"""Econometric estimators (pyodide-clean, pure numpy)."""
from .lp import lp_panel

__all__ = ["lp_panel"]
```

- [ ] **Step 4: Create `puremacro/regress/lp.py`**

```python
"""Panel local projection estimator with Driscoll-Kraay standard errors.

Jordà (2005) local projections in a panel setting. For each horizon h:

    y_{i, t+h} = alpha_i + beta_h * shock_t + gamma' x_{t-1} + u_{i,t+h}

beta_h is the impulse response of y to the shock at horizon h.
Driscoll-Kraay (1998) SEs are robust to cross-sectional dependence and
serial correlation; the truncation lag defaults to h+1.

Pure-numpy implementation, pyodide-clean.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _dk_default_lag(horizon: int) -> int:
    """Default Driscoll-Kraay truncation: h + 1 (covers LP overlap)."""
    return horizon + 1


def _newey_west_weight(j: int, lag: int) -> float:
    """Bartlett kernel weight: 1 - j/(lag+1)."""
    return 1.0 - j / (lag + 1)


def _within_demean(y: np.ndarray, unit_idx: np.ndarray) -> np.ndarray:
    """Subtract per-unit mean from y. unit_idx is integer-coded."""
    n_units = unit_idx.max() + 1
    means = np.zeros(n_units)
    counts = np.zeros(n_units)
    for i, u in enumerate(unit_idx):
        means[u] += y[i]
        counts[u] += 1
    means = means / np.maximum(counts, 1)
    return y - means[unit_idx]


def _ols_with_dk_se(
    Y: np.ndarray,           # (N,)
    X: np.ndarray,           # (N, K)
    time_idx: np.ndarray,    # (N,) integer time-period codes
    dk_lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pooled OLS with Driscoll-Kraay covariance.

    Returns (beta_hat, var_beta_hat) as length-K arrays.
    """
    N, K = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ Y
    resid = Y - X @ beta
    # h_t: time-aggregated moment conditions (1xK per period).
    T = int(time_idx.max() + 1)
    h_t = np.zeros((T, K))
    for i in range(N):
        h_t[time_idx[i]] += resid[i] * X[i]
    # Newey-West on h_t: S = Sum_j w_j * (Gamma_j + Gamma_j')
    Gamma_0 = h_t.T @ h_t
    S = Gamma_0.copy()
    for j in range(1, dk_lag + 1):
        if j >= T:
            break
        Gamma_j = h_t[j:].T @ h_t[:-j]
        w = _newey_west_weight(j, dk_lag)
        S = S + w * (Gamma_j + Gamma_j.T)
    var_beta = XtX_inv @ S @ XtX_inv
    return beta, var_beta


def lp_panel(
    panel: pd.DataFrame,
    *,
    y: str,
    shock: str,
    horizons=range(0, 13),
    unit: str = "unit",
    date: str = "date",
    unit_fe: bool = True,
    controls: list[str] | None = None,
    se: str = "driscoll_kraay",
    dk_lag: int | None = None,
    dummies: list[str] | None = None,
) -> pd.DataFrame:
    """Run panel LP for each horizon h in horizons.

    Returns long-format DataFrame with columns:
        horizon, beta, se, t, p, ci_lo, ci_hi, n_obs
    """
    if se != "driscoll_kraay":
        raise NotImplementedError(f"se={se!r} not implemented")

    df = panel.copy()
    df = df.sort_values([unit, date]).reset_index(drop=True)

    unit_codes, unit_uniq = pd.factorize(df[unit])
    date_codes, date_uniq = pd.factorize(df[date])
    df["_unit_idx"] = unit_codes
    df["_date_idx"] = date_codes

    out_rows = []
    controls = controls or []
    dummies = dummies or []
    regressor_names = [shock] + controls + dummies

    for h in horizons:
        # Build (i, t) -> (i, t+h) mapping. y_lead at row (i,t) = y at (i, t+h).
        df_h = df.copy()
        df_h["y_lead"] = df_h.groupby("_unit_idx")[y].shift(-h)
        df_h = df_h.dropna(subset=["y_lead", shock] + controls + dummies)
        if df_h.empty:
            continue

        Y = df_h["y_lead"].to_numpy(dtype=float)
        X_cols = [df_h[c].to_numpy(dtype=float).reshape(-1, 1) for c in regressor_names]
        X = np.hstack(X_cols)

        if unit_fe:
            Y = _within_demean(Y, df_h["_unit_idx"].to_numpy())
            X = np.column_stack([
                _within_demean(X[:, k], df_h["_unit_idx"].to_numpy())
                for k in range(X.shape[1])
            ])

        lag = dk_lag if dk_lag is not None else _dk_default_lag(h)
        beta, var_beta = _ols_with_dk_se(
            Y, X, df_h["_date_idx"].to_numpy(), lag,
        )
        b = float(beta[0])
        se_b = float(np.sqrt(max(var_beta[0, 0], 0.0)))
        t_stat = b / se_b if se_b > 0 else np.nan
        # 90% CI
        ci_z = 1.6448536269514722
        out_rows.append({
            "horizon": h,
            "beta": b,
            "se": se_b,
            "t": t_stat,
            "p": 2.0 * (1.0 - _norm_cdf(abs(t_stat))) if se_b > 0 else np.nan,
            "ci_lo": b - ci_z * se_b,
            "ci_hi": b + ci_z * se_b,
            "n_obs": int(len(Y)),
        })

    return pd.DataFrame(out_rows)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via erf (math.erf is in stdlib)."""
    import math
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


__all__ = ["lp_panel", "_dk_default_lag"]
```

- [ ] **Step 5: Run tests, expect green**

```bash
cd puremacro && pytest tests/test_regress_lp.py -v --no-header 2>&1 | tail -10
```
Expected: all 6 pass.

- [ ] **Step 6: Run full suite**

```bash
cd puremacro && pytest -q --no-header 2>&1 | tail -3
```
Expected: 1014 + 6 = 1020 passed.

- [ ] **Step 7: Commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git add puremacro/puremacro/regress/ puremacro/tests/test_regress_lp.py
git commit -m "feat(regress): panel LP estimator with Driscoll-Kraay SE (Notebook 29 prep)"
```

---

## Task 2: BLS state-panel fetchers

**Files:**
- Create: `puremacro/fetch/bls_state_panel.py`
- Create: `puremacro/tests/test_fetch_bls_state.py`

- [ ] **Step 1: Write failing tests with offline mocks**

Create `puremacro/tests/test_fetch_bls_state.py`:

```python
"""Tests for BLS state-panel fetchers (LAUS urate, CES employment, CPS LFPR)."""
from __future__ import annotations
import json
import pandas as pd
import pytest


def _laus_canned_response(state_fips: str, n_periods: int = 4) -> str:
    """Canned BLS API JSON for LAUS state urate query."""
    series_id = f"LASST{state_fips}0000000000003"
    obs = [
        {"year": "2024", "period": f"M{m:02d}", "periodName": "x",
         "value": str(3.5 + 0.1 * m), "footnotes": []}
        for m in range(1, n_periods + 1)
    ]
    return json.dumps({
        "status": "REQUEST_SUCCEEDED",
        "Results": {"series": [{"seriesID": series_id, "data": obs}]},
    })


def test_iter_state_urate_q_offline_mock(mock_http):
    """Fetcher parses LAUS monthly response and aggregates to quarterly."""
    mock_http(text={
        "https://api.bls.gov/publicAPI/v2/timeseries/data/LASST010000000000003":
            _laus_canned_response("01", 6),
    })
    from puremacro.fetch.bls_state_panel import iter_state_urate_q
    rows = list(iter_state_urate_q(states=["AL"]))
    assert len(rows) >= 1
    state, qdate, urate, src, meta = rows[0]
    assert state == "AL"
    assert isinstance(qdate, pd.Timestamp)
    assert 0 < float(urate) < 30  # plausible US state urate


def test_iter_state_urate_q_skips_on_empty(mock_http):
    """If BLS returns empty Results, fetcher skips silently per project rule."""
    mock_http(text={
        "https://api.bls.gov/publicAPI/v2/timeseries/data/LASST010000000000003":
            json.dumps({"status": "REQUEST_NOT_PROCESSED"}),
    })
    from puremacro.fetch.bls_state_panel import iter_state_urate_q
    rows = list(iter_state_urate_q(states=["AL"]))
    assert rows == []


def test_iter_state_employment_q_offline_mock(mock_http):
    """CES state seasonally-adjusted total nonfarm employment, monthly→quarterly."""
    series_id = "SMU01000000000000001"  # AL all employees, total nonfarm, SA, thousands
    obs = [
        {"year": "2024", "period": f"M{m:02d}", "periodName": "x",
         "value": str(2000 + m), "footnotes": []}
        for m in range(1, 7)
    ]
    payload = json.dumps({
        "status": "REQUEST_SUCCEEDED",
        "Results": {"series": [{"seriesID": series_id, "data": obs}]},
    })
    mock_http(text={
        f"https://api.bls.gov/publicAPI/v2/timeseries/data/{series_id}": payload,
    })
    from puremacro.fetch.bls_state_panel import iter_state_employment_q
    rows = list(iter_state_employment_q(states=["AL"]))
    assert len(rows) >= 1
    state, qdate, log_emp, src, meta = rows[0]
    assert state == "AL"
    assert log_emp > 0  # log of thousands of jobs


def test_iter_state_participation_a_offline_mock(mock_http):
    """CPS Geographic Profile annual state LFPR."""
    series_id = "LASST010000000000007"
    obs = [
        {"year": str(y), "period": "M13", "periodName": "Annual",
         "value": "60.5", "footnotes": []}
        for y in (2018, 2019, 2020, 2021)
    ]
    payload = json.dumps({
        "status": "REQUEST_SUCCEEDED",
        "Results": {"series": [{"seriesID": series_id, "data": obs}]},
    })
    mock_http(text={
        f"https://api.bls.gov/publicAPI/v2/timeseries/data/{series_id}": payload,
    })
    from puremacro.fetch.bls_state_panel import iter_state_participation_a
    rows = list(iter_state_participation_a(states=["AL"]))
    assert len(rows) >= 1
    state, year, lfpr, src, meta = rows[0]
    assert state == "AL"
    assert isinstance(year, int)
    assert 40 < float(lfpr) < 80


@pytest.mark.network
def test_iter_state_urate_q_live_smoke():
    """One real BLS call to confirm series ID + parsing on a single state.
    Marked @network so it skips by default. Skip on empty per project rule."""
    from puremacro.fetch.bls_state_panel import iter_state_urate_q
    rows = list(iter_state_urate_q(states=["CA"]))
    if not rows:
        pytest.skip("BLS returned empty live response")
    assert all(0 < float(r[2]) < 30 for r in rows[:5])
```

- [ ] **Step 2: Run, verify failure**

```bash
cd puremacro && pytest tests/test_fetch_bls_state.py -v --no-header 2>&1 | tail -10
```
Expected: 4 fail with `ModuleNotFoundError: No module named 'puremacro.fetch.bls_state_panel'`.

- [ ] **Step 3: Create `puremacro/fetch/bls_state_panel.py`**

```python
"""BLS state-panel fetchers — LAUS urate (Q), CES employment (Q), CPS LFPR (A).

Series ID conventions:
  LAUS state urate     : LASST{FIPS}0000000000003  (monthly, SA)
  CES state nonfarm SA : SMU{FIPS}00000000000000001 (monthly, SA)
  CPS state LFPR (ann) : LASST{FIPS}0000000000007  (annual)

Quarterly aggregation: arithmetic mean of monthly values per quarter.
Output records are 5-tuples ``(state_code, qdate|year, value, source_url, metadata)``.

API: https://api.bls.gov/publicAPI/v2/timeseries/data/{series_id}
Optional API key via env var BLS_API_KEY raises the daily limit.
"""
from __future__ import annotations

import json
import math
import os
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from .._http import safe_get_text


_BASE = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
_UA = "puremacro/0.9 (research; +https://github.com/jalonso1979/uncertainty_examples)"

# 2-digit FIPS codes for 50 states + DC. Hard-coded to keep the module
# pyodide-clean (no census/state-info package).
_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08",
    "CT": "09", "DE": "10", "DC": "11", "FL": "12", "GA": "13", "HI": "15",
    "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21",
    "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27",
    "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
    "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39",
    "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46",
    "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53",
    "WV": "54", "WI": "55", "WY": "56",
}


def _fetch_series(series_id: str) -> list[dict]:
    """GET BLS series; return its `data` array (list of {year, period, value, ...})."""
    url = _BASE + series_id
    try:
        text = safe_get_text(url, user_agent=_UA)
    except Exception:
        return []
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return []
    if obj.get("status") != "REQUEST_SUCCEEDED":
        return []
    series = (obj.get("Results") or {}).get("series") or []
    if not series:
        return []
    return series[0].get("data") or []


def _monthly_to_quarterly(records: list[dict]) -> dict[pd.Timestamp, float]:
    """Average monthly observations to quarterly. records: BLS data list."""
    bucket: dict[pd.Timestamp, list[float]] = {}
    for r in records:
        per = r.get("period", "")
        if not per.startswith("M") or per == "M13":
            continue
        m = int(per[1:])
        if m < 1 or m > 12:
            continue
        try:
            year = int(r["year"])
            value = float(r["value"])
        except (KeyError, ValueError):
            continue
        if math.isnan(value):
            continue
        q = (m - 1) // 3 + 1
        qdate = pd.Timestamp(year=year, month=3 * q, day=1) + pd.offsets.MonthEnd(0)
        bucket.setdefault(qdate, []).append(value)
    return {k: float(np.mean(v)) for k, v in bucket.items()}


def iter_state_urate_q(
    states: Iterable[str] | None = None,
) -> Iterator[tuple]:
    """Yield (state_code, qdate, urate, source_url, metadata) per state-quarter."""
    state_list = list(states) if states is not None else list(_FIPS)
    for st in state_list:
        fips = _FIPS.get(st)
        if fips is None:
            continue
        series_id = f"LASST{fips}0000000000003"
        url = _BASE + series_id
        records = _fetch_series(series_id)
        q = _monthly_to_quarterly(records)
        for qdate, urate in sorted(q.items()):
            yield (st, qdate, urate, url, {
                "series_id": series_id, "freq": "Q",
                "source": "BLS LAUS", "measure": "urate_pct_sa",
            })


def iter_state_employment_q(
    states: Iterable[str] | None = None,
) -> Iterator[tuple]:
    """Yield (state_code, qdate, log_emp, source_url, metadata)."""
    state_list = list(states) if states is not None else list(_FIPS)
    for st in state_list:
        fips = _FIPS.get(st)
        if fips is None:
            continue
        series_id = f"SMU{fips}00000000000000001"
        url = _BASE + series_id
        records = _fetch_series(series_id)
        q = _monthly_to_quarterly(records)
        for qdate, emp in sorted(q.items()):
            if emp <= 0:
                continue
            yield (st, qdate, math.log(emp), url, {
                "series_id": series_id, "freq": "Q",
                "source": "BLS CES", "measure": "log_total_nonfarm_sa",
            })


def iter_state_participation_a(
    states: Iterable[str] | None = None,
) -> Iterator[tuple]:
    """Yield (state_code, year, lfpr, source_url, metadata) — annual."""
    state_list = list(states) if states is not None else list(_FIPS)
    for st in state_list:
        fips = _FIPS.get(st)
        if fips is None:
            continue
        series_id = f"LASST{fips}0000000000007"
        url = _BASE + series_id
        records = _fetch_series(series_id)
        for r in records:
            if r.get("period") != "M13":
                continue
            try:
                year = int(r["year"])
                lfpr = float(r["value"])
            except (KeyError, ValueError):
                continue
            if math.isnan(lfpr):
                continue
            yield (st, year, lfpr, url, {
                "series_id": series_id, "freq": "A",
                "source": "BLS LAUS", "measure": "lfpr_pct",
            })


__all__ = [
    "iter_state_urate_q",
    "iter_state_employment_q",
    "iter_state_participation_a",
]
```

**Note for implementer:** the BLS series ID format above is the most common documented convention; if a live probe (`@pytest.mark.network`) returns empty for a state, the implementer should verify the exact ID against the BLS Series ID Builder and adjust. The test `test_iter_state_urate_q_skips_on_empty` already enforces fail-loud-via-skip behavior per the MEMORY rule.

- [ ] **Step 4: Run tests**

```bash
cd puremacro && pytest tests/test_fetch_bls_state.py -v --no-header 2>&1 | tail -10
```
Expected: 4 offline tests pass; live `@pytest.mark.network` test SKIPs.

- [ ] **Step 5: Run full suite**

```bash
cd puremacro && pytest -q --no-header 2>&1 | tail -3
```
Expected: 1020 + 4 = 1024 passed (network test skipped, doesn't count).

- [ ] **Step 6: Commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git add puremacro/puremacro/fetch/bls_state_panel.py puremacro/tests/test_fetch_bls_state.py
git commit -m "feat(fetch): BLS state-panel fetchers — LAUS urate Q, CES emp Q, CPS lfpr A"
```

---

## Task 3: Notebook 29 builder skeleton (data + shock construction)

**Files:**
- Create: `tools/make_notebook_29_state_panel_lp.py`
- Create: `notebooks/29_state_panel_lp_lui.ipynb` (rendered output)

This task builds the notebook skeleton through cell ~5 (data load + shock + merged panel). Estimation cells come in T4.

- [ ] **Step 1: Create the builder**

Create `tools/make_notebook_29_state_panel_lp.py`:

```python
"""Builder for notebooks/29_state_panel_lp_lui.ipynb.

Pattern: write notebook source as a list of cells, render via nbformat,
then optionally execute via jupyter execute.

Run from repo root:
    python tools/make_notebook_29_state_panel_lp.py
"""
from __future__ import annotations

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


_TITLE_CELL = """\
# Notebook 29 — State-Panel LP: National LUI → State Labor Outcomes

Slice 6a's LUI lifted ρ vs urate to +0.331, unblocking this notebook.
We now run a state-level panel local projection (Jordà 2005) with
national LUI shocks and three outcomes: state unemployment rate,
nonfarm employment growth, and labor force participation.

**Spec:** `puremacro/docs/specs/2026-05-10-notebook-29-state-panel-lp.md`.
"""


_SETUP_CELL = """\
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import json
import os
from pathlib import Path

import puremacro
from puremacro.fetch.bls_state_panel import (
    iter_state_urate_q,
    iter_state_employment_q,
    iter_state_participation_a,
)
from puremacro.regress.lp import lp_panel

print(f"puremacro {puremacro.__version__}")
warnings.filterwarnings("ignore", category=FutureWarning)

REFETCH = os.environ.get("PUREMACRO_REFETCH") == "1"
CACHE = Path("notebooks/data_cache")
OUT_TBL = Path("notebooks/output_tables")
OUT_FIG = Path("notebooks/output_figures")
CACHE.mkdir(parents=True, exist_ok=True)
OUT_TBL.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)
"""


_LOAD_LUI_CELL = """\
# Load Slice 6a LUI from notebook 28 outputs.
lui_path = OUT_TBL / "28_lui_us_quarterly.parquet"
lui = pd.read_parquet(lui_path)
lui.index = pd.to_datetime(lui.index)
print("LUI shape:", lui.shape, "range:", lui.index.min(), "→", lui.index.max())
lui.head()
"""


_SHOCK_CELL = """\
# Construct LUI shock as AR(4) residual on the quarterly LUI series.
# Then standardize to unit variance for IRF interpretability.
from numpy.linalg import lstsq

def _ar_residual(series: pd.Series, p: int = 4) -> pd.Series:
    s = series.dropna().astype(float)
    Y = s.iloc[p:].values
    X = np.column_stack([s.shift(k).iloc[p:].values for k in range(1, p + 1)] + [np.ones(len(Y))])
    coef, *_ = lstsq(X, Y, rcond=None)
    fitted = X @ coef
    resid = pd.Series(Y - fitted, index=s.index[p:], name="lui_shock_raw")
    return resid

# Pull primary LUI z-score series (column name lui_<country>).
lui_col = [c for c in lui.columns if c.lower().startswith("lui_")][0]
shock_raw = _ar_residual(lui[lui_col], p=4)
shock = (shock_raw - shock_raw.mean()) / shock_raw.std()
shock.name = "shock"
print("Shock (AR(4) residual, standardized) — range:",
      shock.index.min(), "→", shock.index.max(),
      "n =", len(shock), "mean =", shock.mean(),
      "std =", shock.std())
shock.plot(figsize=(8, 2.5), title="National LUI shock (AR(4) residual, standardized)")
plt.tight_layout()
"""


_FETCH_PANELS_CELL = """\
# Fetch state panels (urate Q, employment Q, LFPR A). Cache locally.
URATE_CACHE = CACHE / "state_urate_q.parquet"
EMP_CACHE = CACHE / "state_emp_q.parquet"
LFPR_CACHE = CACHE / "state_lfpr_a.parquet"

def _refresh_panel(cache, fetcher, columns):
    if cache.exists() and not REFETCH:
        return pd.read_parquet(cache)
    rows = list(fetcher())
    if not rows:
        raise RuntimeError(f"empty panel from {fetcher.__name__}")
    df = pd.DataFrame(rows, columns=columns + ["source", "meta"])[columns]
    df.to_parquet(cache, index=False)
    return df

state_urate = _refresh_panel(
    URATE_CACHE, iter_state_urate_q, ["state", "qdate", "urate"]
)
state_emp = _refresh_panel(
    EMP_CACHE, iter_state_employment_q, ["state", "qdate", "log_emp"]
)
state_lfpr_a = _refresh_panel(
    LFPR_CACHE, iter_state_participation_a, ["state", "year", "lfpr"]
)
print("urate panel:", state_urate.shape, "states:", state_urate['state'].nunique())
print("emp panel:", state_emp.shape)
print("lfpr panel (annual):", state_lfpr_a.shape, "years:", state_lfpr_a['year'].nunique())
"""


_BUILD_PANEL_CELL = """\
# Build merged quarterly panel.
# 1. Outer-join urate + emp on (state, qdate).
# 2. Quarterly-interpolate annual LFPR (constant within year).
# 3. Compute Δ₄ log NFP per state.
# 4. Add national shock (broadcast across states).
# 5. Add COVID dummy (2020Q2-2021Q4).

panel = state_urate.merge(state_emp, on=["state", "qdate"], how="outer")
panel = panel.sort_values(["state", "qdate"])
panel["d4_log_emp"] = panel.groupby("state")["log_emp"].diff(4)

# Quarterly LFPR via constant-by-year fill.
state_lfpr_a["year"] = state_lfpr_a["year"].astype(int)
qrows = []
for _, row in state_lfpr_a.iterrows():
    for q in (1, 2, 3, 4):
        qrows.append({
            "state": row["state"],
            "qdate": pd.Timestamp(row["year"], 3 * q, 1) + pd.offsets.MonthEnd(0),
            "lfpr": row["lfpr"],
        })
state_lfpr_q = pd.DataFrame(qrows)
panel = panel.merge(state_lfpr_q, on=["state", "qdate"], how="left")

# Attach shock.
shock_df = shock.reset_index()
shock_df.columns = ["qdate", "shock"]
shock_df["qdate"] = pd.to_datetime(shock_df["qdate"]) + pd.offsets.QuarterEnd(0)
panel["qdate"] = pd.to_datetime(panel["qdate"]) + pd.offsets.QuarterEnd(0)
panel = panel.merge(shock_df, on="qdate", how="left")

# COVID dummy.
panel["covid"] = (
    (panel["qdate"] >= "2020-04-01") & (panel["qdate"] <= "2021-12-31")
).astype(float)

print("Merged panel:", panel.shape)
print("Date range:", panel["qdate"].min(), "→", panel["qdate"].max())
print("Coverage by outcome (non-null):")
print(panel[["urate", "d4_log_emp", "lfpr", "shock"]].notna().sum())
panel.head()
"""


def main():
    nb = new_notebook(cells=[
        new_markdown_cell(_TITLE_CELL),
        new_code_cell(_SETUP_CELL),
        new_markdown_cell("## 1. Load LUI from notebook 28"),
        new_code_cell(_LOAD_LUI_CELL),
        new_markdown_cell("## 2. Construct LUI shock (AR(4) residual)"),
        new_code_cell(_SHOCK_CELL),
        new_markdown_cell("## 3. Fetch state panels (BLS)"),
        new_code_cell(_FETCH_PANELS_CELL),
        new_markdown_cell("## 4. Build merged panel"),
        new_code_cell(_BUILD_PANEL_CELL),
    ])
    target = "notebooks/29_state_panel_lp_lui.ipynb"
    with open(target, "w") as f:
        nbformat.write(nb, f)
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Render the notebook**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
python3 tools/make_notebook_29_state_panel_lp.py
ls -la notebooks/29_state_panel_lp_lui.ipynb
```

- [ ] **Step 3: Smoke-execute the first cells (no LP yet)**

Don't execute the full notebook in this task — fetching all 51 states from BLS is slow. Instead, smoke-test that the builder produced a parseable notebook:

```bash
python3 -c "import nbformat; nb = nbformat.read('notebooks/29_state_panel_lp_lui.ipynb', as_version=4); print(len(nb.cells), 'cells')"
```
Expected: 9 cells.

- [ ] **Step 4: Commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git add tools/make_notebook_29_state_panel_lp.py notebooks/29_state_panel_lp_lui.ipynb
git commit -m "feat(notebook-29): builder skeleton — load LUI, construct shock, fetch+merge state panels"
```

---

## Task 4: Notebook 29 — LP estimation cells (3 outcomes × pooled IRFs)

**Files:**
- Modify: `tools/make_notebook_29_state_panel_lp.py` (extend with estimation cells)

- [ ] **Step 1: Extend the builder with new cells**

In `tools/make_notebook_29_state_panel_lp.py`, after `_BUILD_PANEL_CELL`, add:

```python
_LP_ESTIMATION_CELL = """\
# Run pooled LP for each outcome.
HORIZONS = list(range(0, 13))
OUTCOMES = [
    ("urate",       "State unemployment rate (pct)"),
    ("d4_log_emp",  "State Δ₄ log nonfarm employment"),
    ("lfpr",        "State labor force participation (pct)"),
]

results = {}
for col, label in OUTCOMES:
    df_lp = panel.dropna(subset=[col, "shock"]).copy()
    res = lp_panel(
        df_lp, y=col, shock="shock",
        unit="state", date="qdate",
        horizons=HORIZONS,
        unit_fe=True,
        dummies=["covid"],
        se="driscoll_kraay",
    )
    results[col] = res
    print(f"\\n=== Pooled IRF — {label} ===")
    print(res.to_string(index=False))
    res.to_parquet(OUT_TBL / f"29_lp_pooled_irf_{col}.parquet", index=False)
"""

_PLOT_POOLED_IRF_CELL = """\
# Plot pooled IRFs as a 3-panel grid.
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
for ax, (col, label) in zip(axes, OUTCOMES):
    res = results[col]
    ax.plot(res["horizon"], res["beta"], color="C0", lw=1.8, label="β")
    ax.fill_between(res["horizon"], res["ci_lo"], res["ci_hi"],
                    alpha=0.2, color="C0", label="90% CI")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("Horizon (quarters)")
ax = axes[0]; ax.set_ylabel("β (response per 1σ LUI shock)")
fig.suptitle("Pooled IRFs — National LUI shock → US state labor outcomes", fontsize=11)
fig.tight_layout()
fig.savefig(OUT_FIG / "29_pooled_irf_grid.pdf", bbox_inches="tight")
fig.savefig(OUT_FIG / "29_pooled_irf_grid.png", bbox_inches="tight", dpi=140)
plt.show()
"""
```

And add to the cells list in `main()`:

```python
        new_markdown_cell("## 5. Run panel LP for 3 outcomes"),
        new_code_cell(_LP_ESTIMATION_CELL),
        new_markdown_cell("## 6. Plot pooled IRFs"),
        new_code_cell(_PLOT_POOLED_IRF_CELL),
```

- [ ] **Step 2: Re-render the notebook**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
python3 tools/make_notebook_29_state_panel_lp.py
```

- [ ] **Step 3: Verify cell count**

```bash
python3 -c "import nbformat; nb = nbformat.read('notebooks/29_state_panel_lp_lui.ipynb', as_version=4); print(len(nb.cells), 'cells')"
```
Expected: 13 cells.

- [ ] **Step 4: Commit**

```bash
git add tools/make_notebook_29_state_panel_lp.py notebooks/29_state_panel_lp_lui.ipynb
git commit -m "feat(notebook-29): pooled LP IRFs for urate / Δ₄log NFP / lfpr (3-panel grid)"
```

---

## Task 5: Notebook 29 — per-state IRFs + visualization (forest + heat map)

**Files:**
- Modify: `tools/make_notebook_29_state_panel_lp.py`

- [ ] **Step 1: Extend builder with per-state estimation + plots**

Add to the builder file:

```python
_PER_STATE_IRF_CELL = """\
# Per-state IRFs at horizon h=8 (peak-response window).
PEAK_H = 8
state_results = {}
for col, label in OUTCOMES:
    rows = []
    for st in panel["state"].unique():
        df_st = panel[(panel["state"] == st)].dropna(subset=[col, "shock"])
        if len(df_st) < 30:  # skip thin panels
            continue
        # State-by-state OLS (no FE; single unit).
        # Build (date, y_lead_h, shock, covid).
        df_h = df_st.sort_values("qdate").copy()
        df_h["y_lead"] = df_h[col].shift(-PEAK_H)
        df_h = df_h.dropna(subset=["y_lead", "shock", "covid"])
        if df_h.empty:
            continue
        Y = df_h["y_lead"].to_numpy(dtype=float)
        X = np.column_stack([df_h["shock"].to_numpy(dtype=float),
                             df_h["covid"].to_numpy(dtype=float),
                             np.ones(len(Y))])
        coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
        rows.append({"state": st, "horizon": PEAK_H, "beta": float(coef[0])})
    res_st = pd.DataFrame(rows).sort_values("beta")
    state_results[col] = res_st
    res_st.to_parquet(OUT_TBL / f"29_lp_state_irf_h{PEAK_H}_{col}.parquet", index=False)
    print(f"{label}: {len(res_st)} states, β@h={PEAK_H} range "
          f"[{res_st['beta'].min():.3f}, {res_st['beta'].max():.3f}]")
"""

_FOREST_PLOTS_CELL = """\
# Forest plot per outcome — states ranked by response at h=8.
for col, label in OUTCOMES:
    res_st = state_results[col]
    fig, ax = plt.subplots(figsize=(5, 11))
    ax.scatter(res_st["beta"], range(len(res_st)), s=20, color="C0")
    ax.axvline(0, color="k", lw=0.5)
    ax.set_yticks(range(len(res_st)))
    ax.set_yticklabels(res_st["state"], fontsize=7)
    ax.set_xlabel(f"β@h={PEAK_H}")
    ax.set_title(f"Per-state response — {label}", fontsize=9)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT_FIG / f"29_forest_{col}_h{PEAK_H}.pdf", bbox_inches="tight")
    plt.show()
"""

_HEATMAP_CELL = """\
# Heat map: state × horizon, sorted by h=8 response.
for col, label in OUTCOMES:
    rows = []
    for st in panel["state"].unique():
        df_st = panel[panel["state"] == st].dropna(subset=[col, "shock"])
        if len(df_st) < 30:
            continue
        for h in HORIZONS:
            df_h = df_st.sort_values("qdate").copy()
            df_h["y_lead"] = df_h[col].shift(-h)
            df_h = df_h.dropna(subset=["y_lead", "shock", "covid"])
            if df_h.empty:
                rows.append({"state": st, "horizon": h, "beta": np.nan})
                continue
            Y = df_h["y_lead"].to_numpy(dtype=float)
            X = np.column_stack([df_h["shock"].to_numpy(dtype=float),
                                 df_h["covid"].to_numpy(dtype=float),
                                 np.ones(len(Y))])
            coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
            rows.append({"state": st, "horizon": h, "beta": float(coef[0])})
    grid = pd.DataFrame(rows).pivot(index="state", columns="horizon", values="beta")
    # Sort by h=8 response.
    grid = grid.loc[grid[PEAK_H].sort_values().index]
    grid.to_parquet(OUT_TBL / f"29_lp_state_heatmap_{col}.parquet")
    fig, ax = plt.subplots(figsize=(8, 11))
    im = ax.imshow(grid.values, aspect="auto", cmap="RdBu_r",
                   vmin=-abs(np.nanpercentile(grid.values, 95)),
                   vmax=abs(np.nanpercentile(grid.values, 95)))
    ax.set_yticks(range(len(grid))); ax.set_yticklabels(grid.index, fontsize=7)
    ax.set_xticks(range(len(grid.columns))); ax.set_xticklabels(grid.columns)
    ax.set_xlabel("Horizon (quarters)"); ax.set_ylabel("State")
    ax.set_title(f"State × horizon — {label}", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(OUT_FIG / f"29_heatmap_{col}.pdf", bbox_inches="tight")
    plt.show()
"""
```

Add to the cells list:
```python
        new_markdown_cell("## 7. Per-state IRFs at h=8"),
        new_code_cell(_PER_STATE_IRF_CELL),
        new_markdown_cell("## 8. Forest plots — per state"),
        new_code_cell(_FOREST_PLOTS_CELL),
        new_markdown_cell("## 9. Heat map — state × horizon"),
        new_code_cell(_HEATMAP_CELL),
```

- [ ] **Step 2: Re-render**

```bash
python3 tools/make_notebook_29_state_panel_lp.py
```

- [ ] **Step 3: Commit**

```bash
git add tools/make_notebook_29_state_panel_lp.py notebooks/29_state_panel_lp_lui.ipynb
git commit -m "feat(notebook-29): per-state IRFs + forest + heatmap visualizations"
```

---

## Task 6: Notebook 29 — heterogeneity + meta + final builder cells

**Files:**
- Modify: `tools/make_notebook_29_state_panel_lp.py`

- [ ] **Step 1: Add heterogeneity + meta cells**

Add to the builder:

```python
_HETEROGENEITY_CELL = """\
# Heterogeneity: split states by manufacturing employment share (median).
# Manufacturing share is a national-level cross-sectional split — we
# approximate by computing each state's average manufacturing-emp /
# total-emp ratio over the sample using public BEA shares (hardcoded
# below from BEA SAEMP25N 2019 snapshot to avoid an additional fetcher).
MFG_SHARE_2019 = {
    "AL": 0.13, "AK": 0.05, "AZ": 0.07, "AR": 0.13, "CA": 0.08, "CO": 0.06,
    "CT": 0.10, "DE": 0.06, "DC": 0.01, "FL": 0.05, "GA": 0.10, "HI": 0.02,
    "ID": 0.10, "IL": 0.10, "IN": 0.18, "IA": 0.16, "KS": 0.13, "KY": 0.13,
    "LA": 0.07, "ME": 0.09, "MD": 0.05, "MA": 0.08, "MI": 0.15, "MN": 0.12,
    "MS": 0.14, "MO": 0.10, "MT": 0.05, "NE": 0.10, "NV": 0.04, "NH": 0.10,
    "NJ": 0.06, "NM": 0.04, "NY": 0.05, "NC": 0.13, "ND": 0.06, "OH": 0.13,
    "OK": 0.08, "OR": 0.10, "PA": 0.10, "RI": 0.09, "SC": 0.13, "SD": 0.10,
    "TN": 0.13, "TX": 0.07, "UT": 0.09, "VT": 0.09, "VA": 0.06, "WA": 0.09,
    "WV": 0.07, "WI": 0.16, "WY": 0.04,
}
median_mfg = float(np.median(list(MFG_SHARE_2019.values())))
panel["high_mfg"] = panel["state"].map(MFG_SHARE_2019).gt(median_mfg).astype(float)

# Pooled LP with mfg-split.
hetero = {}
for col, label in OUTCOMES:
    df_lp = panel.dropna(subset=[col, "shock", "high_mfg"]).copy()
    df_lp["shock_high"] = df_lp["shock"] * df_lp["high_mfg"]
    res = lp_panel(
        df_lp, y=col, shock="shock",
        controls=["shock_high", "high_mfg"],
        unit="state", date="qdate",
        horizons=HORIZONS, unit_fe=True,
        dummies=["covid"], se="driscoll_kraay",
    )
    hetero[col] = res
    res.to_parquet(OUT_TBL / f"29_lp_heterogeneity_mfg_{col}.parquet", index=False)
    print(f"\\n=== Heterogeneity (mfg high vs low) — {label} ===")
    print(res.to_string(index=False))

# Plot baseline + interaction term.
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
for ax, (col, label) in zip(axes, OUTCOMES):
    res = hetero[col]
    base = results[col]
    ax.plot(res["horizon"], res["beta"], color="C0", lw=1.5, label="baseline (β)")
    ax.fill_between(res["horizon"], res["ci_lo"], res["ci_hi"],
                    alpha=0.15, color="C0")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("Horizon (quarters)")
axes[0].set_ylabel("β (response per 1σ LUI shock)")
fig.suptitle("Heterogeneity by mfg share (high vs low) — baseline shown",
             fontsize=11)
fig.tight_layout()
fig.savefig(OUT_FIG / "29_heterogeneity_mfg_grid.pdf", bbox_inches="tight")
plt.show()
"""

_META_CELL = """\
# Write meta.json with run details.
meta = {
    "puremacro_version": puremacro.__version__,
    "computed_at": pd.Timestamp.utcnow().isoformat(),
    "n_states": int(panel['state'].nunique()),
    "n_quarters": int(panel['qdate'].nunique()),
    "shock_n": int(shock.notna().sum()),
    "shock_method": "AR(4) residual, standardized",
    "outcomes": [c for c, _ in OUTCOMES],
    "horizons": HORIZONS,
    "covid_dummy": "2020Q2-2021Q4",
    "se": "driscoll_kraay",
}
with open(OUT_TBL / "29_meta.json", "w") as f:
    json.dump(meta, f, indent=2, default=str)
print(json.dumps(meta, indent=2, default=str))
"""
```

Add cells:
```python
        new_markdown_cell("## 10. Heterogeneity by manufacturing share"),
        new_code_cell(_HETEROGENEITY_CELL),
        new_markdown_cell("## 11. Run metadata"),
        new_code_cell(_META_CELL),
```

- [ ] **Step 2: Re-render + commit**

```bash
python3 tools/make_notebook_29_state_panel_lp.py
git add tools/make_notebook_29_state_panel_lp.py notebooks/29_state_panel_lp_lui.ipynb
git commit -m "feat(notebook-29): mfg-share heterogeneity split + run-meta cell"
```

---

## Task 7: Re-run notebook 29 + validate acceptance criteria

**Files:**
- Notebook 29 outputs (parquet + JSON + PDF)

- [ ] **Step 1: Verify branch state + commits since v0.8.0**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git log --oneline 1437881..HEAD
```
Expected: 6 new commits (T1–T6).

- [ ] **Step 2: Re-run notebook end-to-end**

```bash
PUREMACRO_REFETCH=1 jupyter execute notebooks/29_state_panel_lp_lui.ipynb \
    --output 29_state_panel_lp_lui.executed.ipynb
```
Network-bound; ~3-8 min for 51-state BLS pulls + ~1 min for LP estimation. Use background bash + poll.

- [ ] **Step 3: Inspect outputs**

```bash
cat notebooks/output_tables/29_meta.json
ls -la notebooks/output_tables/29_*.parquet
ls -la notebooks/output_figures/29_*.pdf
python3 -c "import pandas as pd; r = pd.read_parquet('notebooks/output_tables/29_lp_pooled_irf_urate.parquet'); print(r.to_string(index=False))"
python3 -c "import pandas as pd; r = pd.read_parquet('notebooks/output_tables/29_lp_pooled_irf_d4_log_emp.parquet'); print(r.to_string(index=False))"
```

- [ ] **Step 4: Acceptance check**

| Criterion | Target | Action if missed |
|---|---|---|
| State urate IRF positive at h ∈ [2, 6] (β > 0, |t| > 1.65) | required | STOP — investigate sign reversal. Likely shock standardization or control confound. |
| Magnitude reasonable (peak β ≈ 0.05–0.40) | required | If outside range, document and proceed with caveat. |
| State emp growth IRF negative at h ∈ [2, 6] | required | STOP if positive — investigate. |
| Heterogeneity: mfg-high response > mfg-low | preferred | Document if not; not a blocker. |
| All 51 states fetched | required | Fix fetcher / state-code issue if missing states. |

- [ ] **Step 5: REPORT findings — no commit yet**

The release commit comes in T8.

---

## Task 8: 0.9.0 release

**Files:**
- `puremacro/pyproject.toml`, `puremacro/puremacro/__init__.py`, `puremacro/tests/test_import.py`, `puremacro/CHANGELOG.md`

- [ ] **Step 1: Bump version**

In:
- `puremacro/pyproject.toml`: `version = "0.8.0"` → `"0.9.0"`
- `puremacro/puremacro/__init__.py`: `__version__ = "0.8.0"` → `"0.9.0"`
- `puremacro/tests/test_import.py`: `assert puremacro.__version__ == "0.8.0"` → `"0.9.0"`

- [ ] **Step 2: Add CHANGELOG entry**

Insert above the `## 0.8.0 — 2026-05-09` block. Use the actual values from T7:

```markdown
## 0.9.0 — 2026-05-10

Notebook 29 ships: state-panel LP of US labor outcomes on national LUI shocks. Adds two new submodules — `puremacro.fetch.bls_state_panel` (BLS LAUS / CES / CPS state fetchers) and `puremacro.regress.lp` (panel local projection with Driscoll-Kraay SE).

### Added

- `puremacro.regress.lp.lp_panel` — generic panel local projection estimator. Pure numpy. Supports unit fixed effects, controls, dummies, Driscoll-Kraay SEs (1998) with auto-pick truncation lag h+1.
- `puremacro.fetch.bls_state_panel` — three BLS fetchers: `iter_state_urate_q` (LAUS quarterly state urate), `iter_state_employment_q` (CES quarterly nonfarm SA, log), `iter_state_participation_a` (annual state LFPR). Hard-coded 50-state + DC FIPS map; pyodide-clean.
- Notebook 29 (`notebooks/29_state_panel_lp_lui.ipynb`) + paired builder (`tools/make_notebook_29_state_panel_lp.py`).
- Tests: `test_regress_lp.py` (6 unit tests with synthetic panels), `test_fetch_bls_state.py` (4 offline mocks + 1 network smoke).

### Validation (notebook 29 fresh re-run)

| Outcome | Peak β (h=<H>) | t-stat | Sign matches theory |
|---|---:|---:|---|
| State urate (pp) | <VALUE> | <T> | <yes/no> |
| State Δ₄ log NFP | <VALUE> | <T> | <yes/no> |
| State LFPR | <VALUE> | <T> | <yes/no> |

Heterogeneity by manufacturing share: <one-line summary>.

### Pyodide compatibility

- Pure numpy + pandas. No new top-level deps. Same 1 pre-existing pyodide-compat failure (statsmodels.tsa.x13 leak); no new leaks.

### Notes for next iteration

- Slice 6b candidates: `llm_prob_kernel`, Picault-Renault paragraph MNL, stricter sentence tokenizer, per-bank precise extractors, BIS speeches.
- LP-IV (instrumented LP) extension: instrument national LUI with EPU news shock orthogonal to state-level confounders.
- Cross-country state/region panels (e.g., German Länder, UK regions) using corresponding national LUI series.
```

Replace `<VALUE>`, `<T>`, `<yes/no>`, `<H>` placeholders with actual T7 numbers before committing.

- [ ] **Step 3: Final regression sweep**

```bash
cd puremacro && pytest -q --no-header 2>&1 | tail -3
```
Expected: ≥ 1024 passed (1014 baseline + 6 LP + 4 BLS fetcher tests; +1 fewer than 1025 if `test_import.py` count check is in mix).

```bash
cd puremacro && pytest tests/test_pyodide_compat.py -q --no-header 2>&1 | tail -3
```
Expected: same 1 pre-existing failure; no new leaks.

- [ ] **Step 4: Commit + tag**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git status -s notebooks/output_tables/ notebooks/output_figures/ notebooks/data_cache/
git add notebooks/output_tables/29_*.parquet notebooks/output_tables/29_meta.json \
        notebooks/data_cache/state_*.parquet \
        puremacro/pyproject.toml puremacro/puremacro/__init__.py \
        puremacro/tests/test_import.py puremacro/CHANGELOG.md
# output_figures are gitignored — force-add per repo convention
git add -f notebooks/output_figures/29_*.pdf notebooks/output_figures/29_*.png 2>/dev/null
git commit -m "chore(release): puremacro 0.9.0 — Notebook 29 (state-panel LP, national LUI shock)"
git tag -a v0.9.0 -m "puremacro 0.9.0 — Notebook 29: state-panel LP, national LUI shock → state labor outcomes"
```

(Do NOT push.)

---

## Definition of Done

- [ ] All 8 task blocks above checked off.
- [ ] Branch has new commits past v0.8.0, tagged `v0.9.0`.
- [ ] `pytest -q` ≥ 1024 passed.
- [ ] `test_pyodide_compat.py` shows the same 1 pre-existing failure (no new leaks).
- [ ] Notebook 29 re-runs end-to-end without manual intervention.
- [ ] Pooled urate IRF positive and significant at h ∈ [2, 6] (acceptance criterion met).
- [ ] CHANGELOG has actual β / t-stat values from re-run.
- [ ] `puremacro.__version__ == "0.9.0"`.

## Out of scope (deferred)

- LP-IV with EPU / news-based instruments.
- State-level VAR or FAVAR.
- Cross-country state/region panels.
- Real-time forecasting application.
- Slice 6b items (LLM kernel, Picault-Renault, BIS speeches).
- Bayesian LP (Plagborg-Møller-Wolf).
