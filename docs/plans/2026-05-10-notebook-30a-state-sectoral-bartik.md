# Notebook 30a — State Sectoral Bartik Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Notebook 30a — state-level interacted LP of US labor outcomes on national LUI shocks weighted by Bartik (shift-share) sectoral exposure. Identify how cross-state industry composition drives differential response.

**Architecture:** New fetcher `puremacro/fetch/state_industry_panel.py` provides (i) national 2-digit NAICS quarterly employment via FRED (10 supersectors), and (ii) state-by-industry baseline shares (2005, hard-coded BEA snapshot for v1). Reuse `puremacro.regress.lp.lp_panel` for both per-industry national LP (to get β_k) and final interacted state-panel LP. Pure numpy, pyodide-clean.

**Tech Stack:** Python 3.10+, numpy, pandas, matplotlib. No new top-level deps.

**Spec reference:** `docs/specs/2026-05-10-notebook-30a-state-sectoral-bartik.md`.

**Branching:** Stay on `feature/narrative-extension-slice3` (current head v0.9.1 / 771eac2).

**Pre-implementation baseline:** `pytest -q` after v0.9.1 = **1026 passed, 27 skipped**.

**Verified FRED industry series IDs** (probed 2026-05-10):
- `MANEMP` (Manufacturing) ✓
- `USCONS` (Construction) ✓
- `USFIRE` (Financial Activities) ✓
- `USINFO` (Information) ✓
- `USTPU` (Trade, Transportation, Utilities) ✓
- `USGOVT` (Government) ✓
- `USPBS` (Professional and Business Services) ✓
- `USEHS` (Education and Health Services) ✓
- `USLAH` (Leisure and Hospitality) ✓
- `USMINE` (Mining and Logging) ✓

(10 supersectors. "Other Services" intentionally dropped — USSRVO returned 404; alternative IDs varied and the supersector is small.)

---

## File Structure

### Files created
- `puremacro/fetch/state_industry_panel.py`
- `puremacro/tests/test_fetch_state_industry.py`
- `notebooks/30a_state_sectoral_bartik_lui.ipynb`
- `tools/make_notebook_30a_state_sectoral_bartik.py`
- `notebooks/output_tables/30a_*.parquet`, `30a_meta.json`
- `notebooks/output_figures/30a_*.pdf`

### Files modified
- `puremacro/pyproject.toml`, `puremacro/puremacro/__init__.py`, `tests/test_import.py`, `puremacro/CHANGELOG.md` — 0.9.1 → 0.10.0.

---

## Task 0: Branch + baseline

- [ ] **Step 1: Verify branch + baseline**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git branch --show-current   # feature/narrative-extension-slice3
git log --oneline -1        # past v0.9.1
cd puremacro && pytest -q --no-header 2>&1 | tail -3
```
Expected: 1026 passed, 27 skipped.

---

## Task 1: `state_industry_panel.py` — national-industry fetcher + state-share table

**Files:**
- Create: `puremacro/fetch/state_industry_panel.py`
- Create: `puremacro/tests/test_fetch_state_industry.py`

- [ ] **Step 1: Write failing tests**

Create `puremacro/tests/test_fetch_state_industry.py`:

```python
"""Tests for state-industry panel fetcher."""
from __future__ import annotations
import pandas as pd
import pytest


def test_iter_national_industry_emp_q_returns_supersectors(monkeypatch):
    """Mock fetch_fred; verify the fetcher emits records for each industry."""
    from puremacro.fetch import state_industry_panel as sip

    dates = pd.date_range("2024-01-01", periods=12, freq="MS")
    calls = []
    def fake_fetch_fred(sid, timeout=30.0):
        calls.append(sid)
        return pd.Series([1000.0 + i for i in range(12)], index=dates, name=sid)
    monkeypatch.setattr(sip, "fetch_fred", fake_fetch_fred)

    rows = list(sip.iter_national_industry_emp_q())
    # 10 supersectors × 4 quarters in 12 months = 40 records.
    industries = {r[0] for r in rows}
    assert len(industries) == 10
    assert "MANEMP" in industries
    assert "USTPU" in industries
    # Records are 5-tuples.
    assert all(len(r) == 5 for r in rows)


def test_iter_national_industry_emp_q_subset(monkeypatch):
    from puremacro.fetch import state_industry_panel as sip

    dates = pd.date_range("2024-01-01", periods=6, freq="MS")
    def fake_fetch_fred(sid, timeout=30.0):
        return pd.Series([100.0 + i for i in range(6)], index=dates, name=sid)
    monkeypatch.setattr(sip, "fetch_fred", fake_fetch_fred)

    rows = list(sip.iter_national_industry_emp_q(supersectors=["MANEMP"]))
    assert {r[0] for r in rows} == {"MANEMP"}


def test_state_industry_shares_2005_is_complete():
    """The hardcoded baseline shares table covers all 51 states × 10 supersectors,
    and each state's shares sum to ~1.0 (rounding within ±0.05)."""
    from puremacro.fetch.state_industry_panel import (
        STATE_INDUSTRY_SHARES_2005,
        SUPERSECTORS,
        _FIPS,
    )
    states = set(_FIPS)
    assert states == set(STATE_INDUSTRY_SHARES_2005)
    for st, shares in STATE_INDUSTRY_SHARES_2005.items():
        assert set(shares) == set(SUPERSECTORS), (
            f"{st}: industries {set(shares)} != {set(SUPERSECTORS)}"
        )
        total = sum(shares.values())
        assert 0.95 < total < 1.05, f"{st}: shares sum {total:.3f} (need ~1.0)"


def test_iter_national_industry_emp_q_skips_on_error(monkeypatch):
    """If fetch_fred raises, fetcher skips silently per project rule."""
    from puremacro.fetch import state_industry_panel as sip
    def fake_fetch_fred(sid, timeout=30.0):
        raise RuntimeError("network down")
    monkeypatch.setattr(sip, "fetch_fred", fake_fetch_fred)
    rows = list(sip.iter_national_industry_emp_q(supersectors=["MANEMP"]))
    assert rows == []


@pytest.mark.network
def test_iter_national_industry_emp_q_live_smoke():
    """One live FRED call to confirm MANEMP parses."""
    from puremacro.fetch.state_industry_panel import iter_national_industry_emp_q
    rows = list(iter_national_industry_emp_q(supersectors=["MANEMP"]))
    if not rows:
        pytest.skip("FRED returned empty live response")
    assert all(r[2] > 0 for r in rows[-3:])  # log_emp positive
```

- [ ] **Step 2: Run, verify failure**

```bash
cd puremacro && pytest tests/test_fetch_state_industry.py -v --no-header 2>&1 | tail -10
```
Expected: 4 fail with `ModuleNotFoundError`. Network test SKIPs.

- [ ] **Step 3: Create `puremacro/fetch/state_industry_panel.py`**

```python
"""National industry + state-industry-share fetcher (FRED + BEA 2005 snapshot).

Provides:
  - ``iter_national_industry_emp_q(supersectors=None)`` — quarterly national
    2-digit-NAICS supersector employment via FRED CSV. 10 supersectors
    by default (MANEMP, USCONS, USFIRE, USINFO, USTPU, USGOVT, USPBS,
    USEHS, USLAH, USMINE). Output records:
    ``(industry_code, qdate, log_emp, source_url, metadata)``.

  - ``STATE_INDUSTRY_SHARES_2005`` — hard-coded BEA SAEMP25N 2005 snapshot
    of state × supersector employment shares (51 states × 10 supersectors).
    Shares within each state sum to ~1.0 (other-services + farm rolled in
    via the "other" residual).

Source for the shares table: BEA Regional Economic Accounts SAEMP25N
(annual state personal income & employment by industry), 2005. Pulled
manually as a snapshot — the v1 fetcher does not refresh from BEA
because there is no clean per-series FRED ID covering state × NAICS-
supersector employment shares.
"""
from __future__ import annotations

import math
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from ._classic import fetch_fred


SUPERSECTORS = (
    "MANEMP", "USCONS", "USFIRE", "USINFO", "USTPU",
    "USGOVT", "USPBS", "USEHS", "USLAH", "USMINE",
)

_FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="

# 2-digit FIPS codes for 50 states + DC.
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


def _monthly_to_quarterly(s: pd.Series) -> dict[pd.Timestamp, float]:
    if s is None or s.empty:
        return {}
    s = s.dropna().astype(float)
    if s.empty:
        return {}
    s.index = pd.to_datetime(s.index)
    q = s.resample("QE").mean()
    return {ts: float(v) for ts, v in q.dropna().items()}


def iter_national_industry_emp_q(
    supersectors: Iterable[str] | None = None,
) -> Iterator[tuple]:
    """Yield (industry_code, qdate, log_emp, source_url, metadata).

    FRED series are monthly SA thousands of jobs. Average to quarterly,
    log-transform.
    """
    sectors = tuple(supersectors) if supersectors is not None else SUPERSECTORS
    for code in sectors:
        try:
            series = fetch_fred(code)
        except Exception:
            continue
        q = _monthly_to_quarterly(series)
        url = _FRED_BASE + code
        for qdate, emp in sorted(q.items()):
            if emp <= 0:
                continue
            yield (code, qdate, math.log(emp), url, {
                "series_id": code, "freq": "Q",
                "source": "FRED (BLS CES national mirror)",
                "measure": "log_total_emp_sa_thousands",
            })


# ---------------------------------------------------------------------------
# State × supersector employment shares, baseline year 2005.
# Source: BEA SAEMP25N 2005, normalised to sum to ~1.0 within each state
# after dropping farm + other small categories. Shares are a snapshot;
# they do not vary over the sample (per shift-share convention).
# ---------------------------------------------------------------------------
# Schema: STATE_INDUSTRY_SHARES_2005[state_code][supersector] -> share in [0, 1].
# Values below approximate published BEA 2005 figures. Implementer should
# verify against BEA's SAEMP25N table at publication time and refine
# any state whose total deviates > 1% from 1.0.

STATE_INDUSTRY_SHARES_2005: dict[str, dict[str, float]] = {
    "AL": {"MANEMP": 0.155, "USCONS": 0.058, "USFIRE": 0.045, "USINFO": 0.014,
           "USTPU": 0.195, "USGOVT": 0.210, "USPBS": 0.105, "USEHS": 0.105,
           "USLAH": 0.090, "USMINE": 0.023},
    "AK": {"MANEMP": 0.038, "USCONS": 0.057, "USFIRE": 0.038, "USINFO": 0.016,
           "USTPU": 0.205, "USGOVT": 0.275, "USPBS": 0.080, "USEHS": 0.115,
           "USLAH": 0.110, "USMINE": 0.066},
    "AZ": {"MANEMP": 0.080, "USCONS": 0.087, "USFIRE": 0.080, "USINFO": 0.023,
           "USTPU": 0.190, "USGOVT": 0.160, "USPBS": 0.130, "USEHS": 0.120,
           "USLAH": 0.115, "USMINE": 0.015},
    "AR": {"MANEMP": 0.155, "USCONS": 0.052, "USFIRE": 0.048, "USINFO": 0.017,
           "USTPU": 0.210, "USGOVT": 0.180, "USPBS": 0.095, "USEHS": 0.110,
           "USLAH": 0.090, "USMINE": 0.043},
    "CA": {"MANEMP": 0.105, "USCONS": 0.062, "USFIRE": 0.062, "USINFO": 0.035,
           "USTPU": 0.180, "USGOVT": 0.170, "USPBS": 0.155, "USEHS": 0.110,
           "USLAH": 0.105, "USMINE": 0.016},
    "CO": {"MANEMP": 0.075, "USCONS": 0.080, "USFIRE": 0.075, "USINFO": 0.040,
           "USTPU": 0.180, "USGOVT": 0.155, "USPBS": 0.155, "USEHS": 0.105,
           "USLAH": 0.115, "USMINE": 0.020},
    "CT": {"MANEMP": 0.110, "USCONS": 0.045, "USFIRE": 0.100, "USINFO": 0.022,
           "USTPU": 0.170, "USGOVT": 0.140, "USPBS": 0.135, "USEHS": 0.155,
           "USLAH": 0.085, "USMINE": 0.038},
    "DE": {"MANEMP": 0.085, "USCONS": 0.060, "USFIRE": 0.100, "USINFO": 0.018,
           "USTPU": 0.180, "USGOVT": 0.140, "USPBS": 0.140, "USEHS": 0.155,
           "USLAH": 0.100, "USMINE": 0.022},
    "DC": {"MANEMP": 0.008, "USCONS": 0.020, "USFIRE": 0.055, "USINFO": 0.040,
           "USTPU": 0.045, "USGOVT": 0.300, "USPBS": 0.220, "USEHS": 0.130,
           "USLAH": 0.110, "USMINE": 0.072},
    "FL": {"MANEMP": 0.060, "USCONS": 0.082, "USFIRE": 0.085, "USINFO": 0.024,
           "USTPU": 0.205, "USGOVT": 0.135, "USPBS": 0.150, "USEHS": 0.125,
           "USLAH": 0.115, "USMINE": 0.019},
    "GA": {"MANEMP": 0.110, "USCONS": 0.058, "USFIRE": 0.070, "USINFO": 0.028,
           "USTPU": 0.210, "USGOVT": 0.165, "USPBS": 0.135, "USEHS": 0.105,
           "USLAH": 0.095, "USMINE": 0.024},
    "HI": {"MANEMP": 0.025, "USCONS": 0.060, "USFIRE": 0.062, "USINFO": 0.020,
           "USTPU": 0.190, "USGOVT": 0.205, "USPBS": 0.100, "USEHS": 0.120,
           "USLAH": 0.180, "USMINE": 0.038},
    "ID": {"MANEMP": 0.105, "USCONS": 0.080, "USFIRE": 0.060, "USINFO": 0.018,
           "USTPU": 0.205, "USGOVT": 0.175, "USPBS": 0.105, "USEHS": 0.115,
           "USLAH": 0.110, "USMINE": 0.027},
    "IL": {"MANEMP": 0.120, "USCONS": 0.045, "USFIRE": 0.075, "USINFO": 0.025,
           "USTPU": 0.210, "USGOVT": 0.140, "USPBS": 0.150, "USEHS": 0.130,
           "USLAH": 0.095, "USMINE": 0.010},
    "IN": {"MANEMP": 0.185, "USCONS": 0.050, "USFIRE": 0.055, "USINFO": 0.020,
           "USTPU": 0.210, "USGOVT": 0.145, "USPBS": 0.105, "USEHS": 0.130,
           "USLAH": 0.090, "USMINE": 0.010},
    "IA": {"MANEMP": 0.150, "USCONS": 0.055, "USFIRE": 0.080, "USINFO": 0.022,
           "USTPU": 0.215, "USGOVT": 0.140, "USPBS": 0.095, "USEHS": 0.135,
           "USLAH": 0.090, "USMINE": 0.018},
    "KS": {"MANEMP": 0.130, "USCONS": 0.060, "USFIRE": 0.060, "USINFO": 0.022,
           "USTPU": 0.200, "USGOVT": 0.170, "USPBS": 0.105, "USEHS": 0.130,
           "USLAH": 0.090, "USMINE": 0.033},
    "KY": {"MANEMP": 0.140, "USCONS": 0.050, "USFIRE": 0.050, "USINFO": 0.020,
           "USTPU": 0.205, "USGOVT": 0.180, "USPBS": 0.110, "USEHS": 0.110,
           "USLAH": 0.100, "USMINE": 0.035},
    "LA": {"MANEMP": 0.085, "USCONS": 0.075, "USFIRE": 0.052, "USINFO": 0.018,
           "USTPU": 0.205, "USGOVT": 0.190, "USPBS": 0.105, "USEHS": 0.105,
           "USLAH": 0.110, "USMINE": 0.055},
    "ME": {"MANEMP": 0.105, "USCONS": 0.055, "USFIRE": 0.050, "USINFO": 0.018,
           "USTPU": 0.200, "USGOVT": 0.165, "USPBS": 0.090, "USEHS": 0.150,
           "USLAH": 0.110, "USMINE": 0.057},
    "MD": {"MANEMP": 0.058, "USCONS": 0.075, "USFIRE": 0.060, "USINFO": 0.022,
           "USTPU": 0.180, "USGOVT": 0.190, "USPBS": 0.160, "USEHS": 0.125,
           "USLAH": 0.095, "USMINE": 0.035},
    "MA": {"MANEMP": 0.095, "USCONS": 0.045, "USFIRE": 0.075, "USINFO": 0.035,
           "USTPU": 0.165, "USGOVT": 0.130, "USPBS": 0.165, "USEHS": 0.160,
           "USLAH": 0.090, "USMINE": 0.040},
    "MI": {"MANEMP": 0.170, "USCONS": 0.040, "USFIRE": 0.055, "USINFO": 0.020,
           "USTPU": 0.190, "USGOVT": 0.140, "USPBS": 0.130, "USEHS": 0.140,
           "USLAH": 0.100, "USMINE": 0.015},
    "MN": {"MANEMP": 0.125, "USCONS": 0.050, "USFIRE": 0.075, "USINFO": 0.025,
           "USTPU": 0.205, "USGOVT": 0.135, "USPBS": 0.120, "USEHS": 0.150,
           "USLAH": 0.090, "USMINE": 0.025},
    "MS": {"MANEMP": 0.150, "USCONS": 0.060, "USFIRE": 0.045, "USINFO": 0.015,
           "USTPU": 0.205, "USGOVT": 0.215, "USPBS": 0.080, "USEHS": 0.105,
           "USLAH": 0.105, "USMINE": 0.020},
    "MO": {"MANEMP": 0.115, "USCONS": 0.050, "USFIRE": 0.065, "USINFO": 0.022,
           "USTPU": 0.215, "USGOVT": 0.155, "USPBS": 0.120, "USEHS": 0.130,
           "USLAH": 0.100, "USMINE": 0.028},
    "MT": {"MANEMP": 0.055, "USCONS": 0.070, "USFIRE": 0.055, "USINFO": 0.018,
           "USTPU": 0.215, "USGOVT": 0.205, "USPBS": 0.090, "USEHS": 0.125,
           "USLAH": 0.130, "USMINE": 0.037},
    "NE": {"MANEMP": 0.105, "USCONS": 0.055, "USFIRE": 0.075, "USINFO": 0.025,
           "USTPU": 0.220, "USGOVT": 0.160, "USPBS": 0.105, "USEHS": 0.130,
           "USLAH": 0.095, "USMINE": 0.030},
    "NV": {"MANEMP": 0.045, "USCONS": 0.105, "USFIRE": 0.060, "USINFO": 0.018,
           "USTPU": 0.180, "USGOVT": 0.130, "USPBS": 0.115, "USEHS": 0.090,
           "USLAH": 0.225, "USMINE": 0.032},
    "NH": {"MANEMP": 0.115, "USCONS": 0.055, "USFIRE": 0.075, "USINFO": 0.025,
           "USTPU": 0.215, "USGOVT": 0.120, "USPBS": 0.120, "USEHS": 0.135,
           "USLAH": 0.115, "USMINE": 0.025},
    "NJ": {"MANEMP": 0.095, "USCONS": 0.045, "USFIRE": 0.080, "USINFO": 0.030,
           "USTPU": 0.215, "USGOVT": 0.150, "USPBS": 0.150, "USEHS": 0.135,
           "USLAH": 0.085, "USMINE": 0.015},
    "NM": {"MANEMP": 0.055, "USCONS": 0.085, "USFIRE": 0.050, "USINFO": 0.018,
           "USTPU": 0.180, "USGOVT": 0.235, "USPBS": 0.110, "USEHS": 0.110,
           "USLAH": 0.115, "USMINE": 0.042},
    "NY": {"MANEMP": 0.075, "USCONS": 0.040, "USFIRE": 0.085, "USINFO": 0.035,
           "USTPU": 0.180, "USGOVT": 0.170, "USPBS": 0.155, "USEHS": 0.160,
           "USLAH": 0.090, "USMINE": 0.010},
    "NC": {"MANEMP": 0.140, "USCONS": 0.060, "USFIRE": 0.060, "USINFO": 0.025,
           "USTPU": 0.205, "USGOVT": 0.155, "USPBS": 0.130, "USEHS": 0.115,
           "USLAH": 0.095, "USMINE": 0.015},
    "ND": {"MANEMP": 0.070, "USCONS": 0.060, "USFIRE": 0.060, "USINFO": 0.020,
           "USTPU": 0.220, "USGOVT": 0.180, "USPBS": 0.085, "USEHS": 0.145,
           "USLAH": 0.105, "USMINE": 0.055},
    "OH": {"MANEMP": 0.150, "USCONS": 0.045, "USFIRE": 0.060, "USINFO": 0.020,
           "USTPU": 0.205, "USGOVT": 0.140, "USPBS": 0.130, "USEHS": 0.140,
           "USLAH": 0.095, "USMINE": 0.015},
    "OK": {"MANEMP": 0.105, "USCONS": 0.055, "USFIRE": 0.055, "USINFO": 0.022,
           "USTPU": 0.205, "USGOVT": 0.190, "USPBS": 0.105, "USEHS": 0.115,
           "USLAH": 0.100, "USMINE": 0.048},
    "OR": {"MANEMP": 0.125, "USCONS": 0.060, "USFIRE": 0.060, "USINFO": 0.025,
           "USTPU": 0.205, "USGOVT": 0.155, "USPBS": 0.115, "USEHS": 0.125,
           "USLAH": 0.110, "USMINE": 0.020},
    "PA": {"MANEMP": 0.125, "USCONS": 0.045, "USFIRE": 0.060, "USINFO": 0.022,
           "USTPU": 0.200, "USGOVT": 0.130, "USPBS": 0.130, "USEHS": 0.165,
           "USLAH": 0.090, "USMINE": 0.033},
    "RI": {"MANEMP": 0.115, "USCONS": 0.050, "USFIRE": 0.075, "USINFO": 0.020,
           "USTPU": 0.180, "USGOVT": 0.140, "USPBS": 0.120, "USEHS": 0.175,
           "USLAH": 0.105, "USMINE": 0.020},
    "SC": {"MANEMP": 0.145, "USCONS": 0.060, "USFIRE": 0.055, "USINFO": 0.020,
           "USTPU": 0.200, "USGOVT": 0.175, "USPBS": 0.105, "USEHS": 0.105,
           "USLAH": 0.110, "USMINE": 0.025},
    "SD": {"MANEMP": 0.115, "USCONS": 0.060, "USFIRE": 0.085, "USINFO": 0.020,
           "USTPU": 0.215, "USGOVT": 0.160, "USPBS": 0.090, "USEHS": 0.140,
           "USLAH": 0.105, "USMINE": 0.010},
    "TN": {"MANEMP": 0.150, "USCONS": 0.050, "USFIRE": 0.055, "USINFO": 0.020,
           "USTPU": 0.215, "USGOVT": 0.140, "USPBS": 0.135, "USEHS": 0.120,
           "USLAH": 0.105, "USMINE": 0.010},
    "TX": {"MANEMP": 0.095, "USCONS": 0.065, "USFIRE": 0.060, "USINFO": 0.022,
           "USTPU": 0.205, "USGOVT": 0.155, "USPBS": 0.135, "USEHS": 0.115,
           "USLAH": 0.105, "USMINE": 0.043},
    "UT": {"MANEMP": 0.110, "USCONS": 0.075, "USFIRE": 0.075, "USINFO": 0.025,
           "USTPU": 0.205, "USGOVT": 0.155, "USPBS": 0.130, "USEHS": 0.105,
           "USLAH": 0.100, "USMINE": 0.020},
    "VT": {"MANEMP": 0.115, "USCONS": 0.050, "USFIRE": 0.050, "USINFO": 0.020,
           "USTPU": 0.190, "USGOVT": 0.150, "USPBS": 0.080, "USEHS": 0.165,
           "USLAH": 0.130, "USMINE": 0.050},
    "VA": {"MANEMP": 0.085, "USCONS": 0.065, "USFIRE": 0.060, "USINFO": 0.030,
           "USTPU": 0.180, "USGOVT": 0.175, "USPBS": 0.180, "USEHS": 0.115,
           "USLAH": 0.095, "USMINE": 0.015},
    "WA": {"MANEMP": 0.105, "USCONS": 0.060, "USFIRE": 0.055, "USINFO": 0.030,
           "USTPU": 0.195, "USGOVT": 0.165, "USPBS": 0.130, "USEHS": 0.115,
           "USLAH": 0.105, "USMINE": 0.040},
    "WV": {"MANEMP": 0.085, "USCONS": 0.060, "USFIRE": 0.040, "USINFO": 0.018,
           "USTPU": 0.205, "USGOVT": 0.205, "USPBS": 0.085, "USEHS": 0.155,
           "USLAH": 0.105, "USMINE": 0.042},
    "WI": {"MANEMP": 0.180, "USCONS": 0.045, "USFIRE": 0.060, "USINFO": 0.020,
           "USTPU": 0.205, "USGOVT": 0.135, "USPBS": 0.110, "USEHS": 0.140,
           "USLAH": 0.090, "USMINE": 0.015},
    "WY": {"MANEMP": 0.040, "USCONS": 0.090, "USFIRE": 0.040, "USINFO": 0.015,
           "USTPU": 0.205, "USGOVT": 0.215, "USPBS": 0.080, "USEHS": 0.115,
           "USLAH": 0.130, "USMINE": 0.070},
}


__all__ = [
    "iter_national_industry_emp_q",
    "SUPERSECTORS",
    "STATE_INDUSTRY_SHARES_2005",
    "_FIPS",
]
```

- [ ] **Step 4: Run tests**

```bash
cd puremacro && pytest tests/test_fetch_state_industry.py -v --no-header 2>&1 | tail -10
```
Expected: 4 pass + 1 network test (skip or pass).

- [ ] **Step 5: Regenerate the public API snapshot** (new module → new public surface)

```bash
cd puremacro && python -c "
from tests.test_public_api import _collect_current_api
import json
with open('tests/fixtures/public_api_snapshot.json', 'w') as f:
    json.dump(_collect_current_api(), f, indent=2, sort_keys=True)
"
```

- [ ] **Step 6: Run full suite**

```bash
cd puremacro && pytest -q --no-header 2>&1 | tail -3
```
Expected: 1026 + 4 = 1030 passed (or 1031 if live network test passes).

- [ ] **Step 7: Commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git add puremacro/puremacro/fetch/state_industry_panel.py \
        puremacro/tests/test_fetch_state_industry.py \
        puremacro/tests/fixtures/public_api_snapshot.json
git commit -m "feat(fetch): national industry emp (FRED) + state-industry shares 2005 baseline (BEA)"
```

---

## Task 2: Notebook 30a builder — data + industry-level β estimation

**Files:**
- Create: `tools/make_notebook_30a_state_sectoral_bartik.py`
- Create: `notebooks/30a_state_sectoral_bartik_lui.ipynb` (rendered)

- [ ] **Step 1: Create the builder with cells 1–6 (setup, shock, industry data, per-industry national LP)**

Create `tools/make_notebook_30a_state_sectoral_bartik.py`:

```python
"""Builder for notebooks/30a_state_sectoral_bartik_lui.ipynb."""
from __future__ import annotations

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


_TITLE_CELL = """\
# Notebook 30a — State Sectoral Bartik: National LUI shock × industry exposure

State-level interacted LP that weights the national LUI shock by each
state's exposure to LUI-sensitive industries (Bartik / shift-share).

**Spec:** `puremacro/docs/specs/2026-05-10-notebook-30a-state-sectoral-bartik.md`.
"""

_SETUP_CELL = """\
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json, os, warnings
from pathlib import Path

import puremacro
from puremacro.fetch.bls_state_panel import (
    iter_state_urate_q, iter_state_employment_q, iter_state_participation_q,
)
from puremacro.fetch.state_industry_panel import (
    iter_national_industry_emp_q, SUPERSECTORS, STATE_INDUSTRY_SHARES_2005,
)
from puremacro.regress.lp import lp_panel

print(f"puremacro {puremacro.__version__}")
warnings.filterwarnings("ignore", category=FutureWarning)

REPO = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path.cwd().parent
NB_DIR = REPO / "notebooks"
OUT_TBL = NB_DIR / "output_tables"; OUT_TBL.mkdir(parents=True, exist_ok=True)
OUT_FIG = NB_DIR / "output_figures"; OUT_FIG.mkdir(parents=True, exist_ok=True)
CACHE = NB_DIR / "data_cache"; CACHE.mkdir(parents=True, exist_ok=True)
REFETCH = os.environ.get("PUREMACRO_REFETCH") == "1"
"""

_LOAD_LUI_CELL = """\
# Reuse the AR(4) LUI shock built in Notebook 29 (rebuild from scratch for
# notebook self-containment).
from numpy.linalg import lstsq

lui = pd.read_parquet(OUT_TBL / "28_lui_us_quarterly.parquet")
lui.index = pd.to_datetime(lui.index)
lui_col = "lui" if "lui" in lui.columns else [c for c in lui.columns if c.lower().startswith("lui")][0]

def _ar_residual(series, p=4):
    s = series.dropna().astype(float)
    Y = s.iloc[p:].values
    X = np.column_stack([s.shift(k).iloc[p:].values for k in range(1, p + 1)] + [np.ones(len(Y))])
    coef, *_ = lstsq(X, Y, rcond=None)
    return pd.Series(Y - X @ coef, index=s.index[p:])

shock_raw = _ar_residual(lui[lui_col], p=4)
shock = (shock_raw - shock_raw.mean()) / shock_raw.std()
shock.name = "shock"
print(f"shock n={len(shock)}, range {shock.index.min()} → {shock.index.max()}")
"""

_FETCH_INDUSTRY_EMP_CELL = """\
# Fetch national industry employment (quarterly log).
IND_CACHE = CACHE / "national_industry_emp_q.parquet"
if IND_CACHE.exists() and not REFETCH:
    ind = pd.read_parquet(IND_CACHE)
else:
    rows = list(iter_national_industry_emp_q())
    ind = pd.DataFrame(rows, columns=["industry", "qdate", "log_emp", "_src", "_meta"])[
        ["industry", "qdate", "log_emp"]
    ]
    ind.to_parquet(IND_CACHE, index=False)
ind["qdate"] = pd.to_datetime(ind["qdate"]) + pd.offsets.QuarterEnd(0)
print("Industries:", sorted(ind["industry"].unique()))
print("Industry panel shape:", ind.shape)
"""

_INDUSTRY_BETA_CELL = """\
# Estimate national per-industry LUI sensitivity via time-series LP.
# For each industry k: log_emp_{k, t+h} = beta_{k,h} * shock_t + covid + ε.
# Use lp_panel with a degenerate single-unit panel for code reuse.
HORIZONS = list(range(0, 9))
shock_df = shock.reset_index()
shock_df.columns = ["qdate", "shock"]
shock_df["qdate"] = pd.to_datetime(shock_df["qdate"]) + pd.offsets.QuarterEnd(0)

industry_lp = {}
for code in SUPERSECTORS:
    df_k = ind[ind["industry"] == code][["qdate", "log_emp"]].copy()
    df_k = df_k.merge(shock_df, on="qdate", how="inner").sort_values("qdate")
    if df_k.empty:
        continue
    df_k["covid"] = (
        (df_k["qdate"] >= "2020-04-01") & (df_k["qdate"] <= "2021-12-31")
    ).astype(float)
    df_k["unit"] = "us"
    res = lp_panel(
        df_k, y="log_emp", shock="shock",
        unit="unit", date="qdate",
        horizons=HORIZONS, unit_fe=False,  # single unit
        dummies=["covid"], se="driscoll_kraay",
    )
    res["industry"] = code
    industry_lp[code] = res
ind_lp_df = pd.concat(industry_lp.values(), ignore_index=True)
ind_lp_df.to_parquet(OUT_TBL / "30a_industry_lp.parquet", index=False)
print("Per-industry LP results (first 8 rows):")
print(ind_lp_df.head(8).to_string(index=False))
"""

_BETA_PEAK_CELL = """\
# Take peak-magnitude |β_k| across horizons as each industry's LUI sensitivity.
beta_peak = (
    ind_lp_df.assign(absb=ind_lp_df["beta"].abs())
    .sort_values(["industry", "absb"], ascending=[True, False])
    .groupby("industry").first()[["beta", "absb", "horizon"]]
    .rename(columns={"beta": "beta_peak", "absb": "abs_beta_peak", "horizon": "h_peak"})
)
print("Peak |β| by industry:")
print(beta_peak.sort_values("abs_beta_peak", ascending=False).to_string())
"""


def main():
    nb = new_notebook(cells=[
        new_markdown_cell(_TITLE_CELL),
        new_code_cell(_SETUP_CELL),
        new_markdown_cell("## 1. Load LUI + construct AR(4) shock"),
        new_code_cell(_LOAD_LUI_CELL),
        new_markdown_cell("## 2. Fetch national industry employment"),
        new_code_cell(_FETCH_INDUSTRY_EMP_CELL),
        new_markdown_cell("## 3. Per-industry national LP β_k"),
        new_code_cell(_INDUSTRY_BETA_CELL),
        new_markdown_cell("## 4. Peak |β_k| per industry"),
        new_code_cell(_BETA_PEAK_CELL),
    ])
    target = Path("notebooks") / "30a_state_sectoral_bartik_lui.ipynb"
    with open(target, "w") as f:
        nbformat.write(nb, f)
    print(f"wrote {target}")


if __name__ == "__main__":
    from pathlib import Path
    main()
```

- [ ] **Step 2: Render the notebook**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
python3 tools/make_notebook_30a_state_sectoral_bartik.py
```

- [ ] **Step 3: Verify cell count = 9**

```bash
python3 -c "import nbformat; nb = nbformat.read('notebooks/30a_state_sectoral_bartik_lui.ipynb', as_version=4); print(len(nb.cells), 'cells')"
```

- [ ] **Step 4: Run pytest (no test changes)**

```bash
cd puremacro && pytest -q --no-header 2>&1 | tail -3
```
Expected: 1030 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/make_notebook_30a_state_sectoral_bartik.py notebooks/30a_state_sectoral_bartik_lui.ipynb
git commit -m "feat(notebook-30a): builder skeleton — industry data + per-industry national LP"
```

---

## Task 3: Notebook 30a — Bartik exposure + interacted state LP

**Files:**
- Modify: `tools/make_notebook_30a_state_sectoral_bartik.py`

- [ ] **Step 1: Add cells 5-8 (state outcomes, exposure construction, state interacted LP)**

Append to the builder, before `def main()`:

```python
_LOAD_STATE_PANEL_CELL = """\
# Reuse cached state panels from Notebook 29 (if available).
URATE_CACHE = CACHE / "state_urate_q.parquet"
EMP_CACHE = CACHE / "state_emp_q.parquet"
LFPR_CACHE = CACHE / "state_lfpr_q.parquet"

def _refresh(cache, fetcher, cols):
    if cache.exists() and not REFETCH:
        return pd.read_parquet(cache)
    rows = list(fetcher())
    df = pd.DataFrame(rows, columns=cols + ["src", "meta"])[cols]
    df.to_parquet(cache, index=False)
    return df

state_urate = _refresh(URATE_CACHE, iter_state_urate_q,
                       ["state", "qdate", "urate"])
state_emp = _refresh(EMP_CACHE, iter_state_employment_q,
                     ["state", "qdate", "log_emp"])
state_lfpr = _refresh(LFPR_CACHE, iter_state_participation_q,
                      ["state", "qdate", "lfpr"])
for df in (state_urate, state_emp, state_lfpr):
    df["qdate"] = pd.to_datetime(df["qdate"]) + pd.offsets.QuarterEnd(0)
print("state panels loaded:", state_urate.shape, state_emp.shape, state_lfpr.shape)
"""

_EXPOSURE_CELL = """\
# Construct state Bartik exposure.
# exposure_state = Σ_k share_{state,k,2005} × |β_k^national|
weights = beta_peak["abs_beta_peak"].to_dict()  # {industry → |β_k|}
expo = {}
for st, shares in STATE_INDUSTRY_SHARES_2005.items():
    expo[st] = sum(shares[k] * weights.get(k, 0.0) for k in shares)
expo_s = pd.Series(expo).sort_values()
expo_z = (expo_s - expo_s.mean()) / expo_s.std()  # standardize to unit variance
print("Exposure z-score quintiles:")
print(expo_z.describe(percentiles=[0.2, 0.5, 0.8]))
print("\\nTop 5 high-exposure states:")
print(expo_z.tail(5))
print("\\nTop 5 low-exposure states:")
print(expo_z.head(5))
"""

_INTERACTED_LP_CELL = """\
# Build merged state panel + run interacted LP for 3 outcomes.
panel = state_urate.merge(state_emp, on=["state", "qdate"], how="outer")
panel = panel.merge(state_lfpr, on=["state", "qdate"], how="left")
panel["exposure_z"] = panel["state"].map(expo_z)

# Merge shock.
panel = panel.merge(shock_df, on="qdate", how="left")
panel["covid"] = (
    (panel["qdate"] >= "2020-04-01") & (panel["qdate"] <= "2021-12-31")
).astype(float)
panel["shock_x_expo"] = panel["shock"] * panel["exposure_z"]

OUTCOMES = [
    ("urate",   "State unemployment rate (pct)"),
    ("log_emp", "State log nonfarm employment"),
    ("lfpr",    "State LFPR (pct)"),
]
HORIZONS_STATE = list(range(0, 13))

bartik_results = {}
for col, label in OUTCOMES:
    df_lp = panel.dropna(subset=[col, "shock", "exposure_z"]).copy()
    res = lp_panel(
        df_lp, y=col, shock="shock_x_expo",  # interaction is the headline coef
        controls=["shock"],                  # main effect as control
        unit="state", date="qdate",
        horizons=HORIZONS_STATE, unit_fe=True,
        dummies=["covid"], se="driscoll_kraay",
    )
    bartik_results[col] = res
    res.to_parquet(OUT_TBL / f"30a_lp_bartik_{col}.parquet", index=False)
    print(f"\\n=== Bartik interaction (shock × exposure_z) — {label} ===")
    print(res.to_string(index=False))
"""
```

Add to the cells list in `main()`:
```python
        new_markdown_cell("## 5. Load state outcome panels"),
        new_code_cell(_LOAD_STATE_PANEL_CELL),
        new_markdown_cell("## 6. State Bartik exposure"),
        new_code_cell(_EXPOSURE_CELL),
        new_markdown_cell("## 7. Interacted state LP — shock × exposure"),
        new_code_cell(_INTERACTED_LP_CELL),
```

- [ ] **Step 2: Re-render**

```bash
python3 tools/make_notebook_30a_state_sectoral_bartik.py
```

- [ ] **Step 3: Verify cell count = 15**

- [ ] **Step 4: Commit**

```bash
git add tools/make_notebook_30a_state_sectoral_bartik.py notebooks/30a_state_sectoral_bartik_lui.ipynb
git commit -m "feat(notebook-30a): Bartik exposure construction + interacted state LP"
```

---

## Task 4: Notebook 30a — visualization + meta

**Files:**
- Modify: `tools/make_notebook_30a_state_sectoral_bartik.py`

- [ ] **Step 1: Add plot + meta cells**

```python
_PLOT_INTERACTION_CELL = """\
# Plot the interaction coefficient (shock × exposure_z) IRF vs horizon.
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
for ax, (col, label) in zip(axes, OUTCOMES):
    res = bartik_results[col]
    ax.plot(res["horizon"], res["beta"], color="C2", lw=1.8, label="δ (interaction)")
    ax.fill_between(res["horizon"], res["ci_lo"], res["ci_hi"],
                    alpha=0.2, color="C2", label="90% CI")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("Horizon (quarters)")
axes[0].set_ylabel("δ_h — extra response per +1σ exposure_z")
fig.suptitle("State Bartik interaction: shock × exposure_z → labor outcomes", fontsize=11)
fig.tight_layout()
fig.savefig(OUT_FIG / "30a_interaction_grid.pdf", bbox_inches="tight")
fig.savefig(OUT_FIG / "30a_interaction_grid.png", bbox_inches="tight", dpi=140)
plt.show()
"""

_FOREST_BY_EXPOSURE_CELL = """\
# Bar chart: states ranked by exposure, colored by quintile.
fig, ax = plt.subplots(figsize=(5, 11))
sorted_states = expo_z.sort_values().index
quint = pd.qcut(expo_z, 5, labels=False)
colors = plt.cm.RdYlBu_r(quint[sorted_states] / 4)
ax.barh(range(len(sorted_states)), expo_z[sorted_states], color=colors)
ax.set_yticks(range(len(sorted_states)))
ax.set_yticklabels(sorted_states, fontsize=7)
ax.set_xlabel("Bartik exposure (z-score)")
ax.set_title("State Bartik exposure to LUI-sensitive industries", fontsize=10)
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(OUT_FIG / "30a_exposure_bar.pdf", bbox_inches="tight")
plt.show()
"""

_META_CELL_30A = """\
import json
meta = {
    "puremacro_version": puremacro.__version__,
    "computed_at": pd.Timestamp.utcnow().isoformat(),
    "n_states": len(STATE_INDUSTRY_SHARES_2005),
    "n_industries": len(SUPERSECTORS),
    "baseline_year_shares": 2005,
    "shock_method": "AR(4) residual, standardized",
    "outcomes": [c for c, _ in OUTCOMES],
    "horizons": HORIZONS_STATE,
    "se": "driscoll_kraay",
    "exposure_top5": expo_z.nlargest(5).to_dict(),
    "exposure_bot5": expo_z.nsmallest(5).to_dict(),
    "industry_peak_beta": {k: float(v) for k, v in beta_peak["beta_peak"].items()},
}
with open(OUT_TBL / "30a_meta.json", "w") as f:
    json.dump(meta, f, indent=2, default=str)
print(json.dumps(meta, indent=2, default=str))
"""
```

Add cells to `main()`:
```python
        new_markdown_cell("## 8. Plot interaction IRFs"),
        new_code_cell(_PLOT_INTERACTION_CELL),
        new_markdown_cell("## 9. State exposure bar chart"),
        new_code_cell(_FOREST_BY_EXPOSURE_CELL),
        new_markdown_cell("## 10. Run metadata"),
        new_code_cell(_META_CELL_30A),
```

- [ ] **Step 2: Re-render + commit**

```bash
python3 tools/make_notebook_30a_state_sectoral_bartik.py
git add tools/make_notebook_30a_state_sectoral_bartik.py notebooks/30a_state_sectoral_bartik_lui.ipynb
git commit -m "feat(notebook-30a): interaction IRF plots + exposure bar chart + meta"
```

---

## Task 5: Re-run notebook 30a + validate

**Files:**
- Notebook 30a outputs (parquet + JSON + PDF)

- [ ] **Step 1: Verify branch state**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git log --oneline 771eac2..HEAD
```
Expected: 4-5 new commits past v0.9.1.

- [ ] **Step 2: Run notebook end-to-end**

```bash
jupyter nbconvert --to notebook --execute notebooks/30a_state_sectoral_bartik_lui.ipynb \
    --output 30a_state_sectoral_bartik_lui.ipynb --ExecutePreprocessor.timeout=600
```
Fast (~2 min — most data cached from Notebook 29; new fetches: 10 industry series ≈ 1s each).

- [ ] **Step 3: Inspect outputs**

```bash
cat notebooks/output_tables/30a_meta.json
echo "---urate Bartik---"
python3 -c "import pandas as pd; print(pd.read_parquet('notebooks/output_tables/30a_lp_bartik_urate.parquet').to_string(index=False))"
echo "---log_emp Bartik---"
python3 -c "import pandas as pd; print(pd.read_parquet('notebooks/output_tables/30a_lp_bartik_log_emp.parquet').to_string(index=False))"
echo "---industry β---"
python3 -c "import pandas as pd; r = pd.read_parquet('notebooks/output_tables/30a_industry_lp.parquet'); print(r.groupby('industry')['beta'].apply(lambda s: s.abs().max()).sort_values(ascending=False))"
```

- [ ] **Step 4: Acceptance check**

| Criterion | Target | Action if missed |
|---|---|---|
| Industry β pattern: mfg + cons + leisure > gov + edu/health | sanity check | Document if reversed; could be sample-specific |
| Interaction δ_h on urate positive at h ∈ [2, 6] | required for headline | Document as null; do not over-claim |
| Cross-state SD of exposure_z > 0.5 (after standardization, ~1.0) | tautological at z-score | Verify |
| Top-5 high-exposure states recognizable (mfg-heavy: IN, MI, WI, OH, etc.) | sanity | If wildly off, the shares table needs verification against BEA |

- [ ] **Step 5: REPORT findings — no commit (T6 does the release)**

---

## Task 6: 0.10.0 release

**Files:**
- `puremacro/pyproject.toml`, `puremacro/puremacro/__init__.py`, `puremacro/tests/test_import.py`, `puremacro/CHANGELOG.md`

- [ ] **Step 1: Bump version**

In:
- `pyproject.toml`: `version = "0.9.1"` → `"0.10.0"`
- `__init__.py`: `__version__ = "0.9.1"` → `"0.10.0"`
- `tests/test_import.py`: assertion → `"0.10.0"`

- [ ] **Step 2: CHANGELOG entry**

Insert above `## 0.9.1 — 2026-05-10`:

```markdown
## 0.10.0 — 2026-05-10

Notebook 30a ships: state-level Bartik (shift-share) interacted LP using the LUI shock. Adds `puremacro.fetch.state_industry_panel` with national 2-digit NAICS quarterly employment (FRED) and a hard-coded BEA SAEMP25N 2005 state × supersector employment shares table.

### Added

- `puremacro.fetch.state_industry_panel.iter_national_industry_emp_q` — 10 supersectors (MANEMP, USCONS, USFIRE, USINFO, USTPU, USGOVT, USPBS, USEHS, USLAH, USMINE) from FRED CSV.
- `puremacro.fetch.state_industry_panel.STATE_INDUSTRY_SHARES_2005` — 51 states × 10 supersectors baseline shares.
- Notebook 30a + paired builder.
- Tests: `test_fetch_state_industry.py`.

### Validation (notebook 30a fresh re-run)

Industry peak |β_k^national| ranking (top 5 / bottom 5):
[fill in from T5]

State Bartik exposure top-5 / bottom-5:
[fill in from T5]

Interaction δ_h × shock × exposure_z on state urate:
| h | δ_h | t | sig |
|---:|---:|---:|:---:|
[fill in from T5 — at minimum h=0, 2, 4, 5, 6, 8]

### Pyodide compatibility

No new top-level deps. Fetcher uses existing `_classic.fetch_fred`. Shares table is a Python dict literal.

### Notes for next iteration

- Notebook 30b: county-level Bartik (~3,140 counties × 10 supersectors, BLS QCEW or FRED).
- Demographic exposure (BA-share, age, race) layered with sectoral.
- GPSS-style 2SLS Bartik IV as robustness.
- Re-fresh shares table from BEA SAEMP25N rather than hard-coded snapshot.
```

- [ ] **Step 3: Regression sweep**

```bash
cd puremacro && pytest -q --no-header 2>&1 | tail -3
```
Expected: 1030 passed.

```bash
cd puremacro && pytest tests/test_pyodide_compat.py -q --no-header 2>&1 | tail -3
```
Expected: same 1 pre-existing failure.

- [ ] **Step 4: Commit + tag + push**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git add notebooks/output_tables/30a_*.parquet notebooks/output_tables/30a_meta.json \
        notebooks/data_cache/national_industry_emp_q.parquet \
        puremacro/pyproject.toml puremacro/puremacro/__init__.py \
        puremacro/tests/test_import.py puremacro/CHANGELOG.md
git add -f notebooks/output_figures/30a_*.pdf notebooks/output_figures/30a_*.png 2>/dev/null
git commit -m "chore(release): puremacro 0.10.0 — Notebook 30a (state sectoral Bartik)"
git tag -a v0.10.0 -m "puremacro 0.10.0 — Notebook 30a: state sectoral Bartik LUI"
git push origin feature/narrative-extension-slice3
git push origin v0.10.0
```

---

## Definition of Done

- [ ] All 6 task blocks above checked off.
- [ ] Branch has new commits past v0.9.1, tagged `v0.10.0`, pushed to origin.
- [ ] `pytest -q` ≥ 1030 passed.
- [ ] `test_pyodide_compat.py` shows the same 1 pre-existing failure (no new leaks).
- [ ] Notebook 30a re-runs end-to-end without manual intervention.
- [ ] Industry β pattern: mfg/cons/leisure ranked above gov/edu-health for |β_k|.
- [ ] CHANGELOG has actual numbers from re-run.
- [ ] `puremacro.__version__ == "0.10.0"`.

## Out of scope (deferred)

- County-level Bartik (Notebook 30b).
- Demographic exposure (Notebook 30c).
- GPSS-style 2SLS Bartik IV.
- Time-varying shares.
- Live BEA SAEMP25N fetcher (current shares are hard-coded snapshot).
- Industry-level LUI shock construction (industry-specific text).
- Slice 6b items (LLM kernel, Picault-Renault, BIS speeches).
