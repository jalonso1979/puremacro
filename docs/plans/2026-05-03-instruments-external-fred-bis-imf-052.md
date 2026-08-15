# puremacro 0.5.2 — FRED / BIS / IMF External-CSV Loaders Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **This workspace is NOT a git repo (Google Drive sync); skip every "commit" step the meta-skill suggests.**

**Goal:** Ship 3 new external-CSV providers (FRED, BIS, IMF WEO) under `puremacro/instruments/external/`, each with a generic loader (`load(series_id=..., ...)`) plus 1–4 curated catalog entries. Total 7 new catalog entries, growing the registry 29 → 36. Released as **0.5.2** (patch — additive, no breaking changes).

**Architecture:** New `puremacro/instruments/external/` subpackage, sibling to `literature/`. Each provider is a stand-alone module exposing a generic `load(*, series_id, ...) -> Instrument` plus optional convenience kwargs. The shared `_csv_to_instrument` helper moves from `literature/_helpers.py` up to `instruments/_helpers.py` so both subpackages can import it via `from .._helpers import _csv_to_instrument`. A new `_json_to_instrument` helper handles FRED's JSON response shape. All HTTP goes through `narrative.sources._http.safe_get_*` (existing Pyodide-correct path). Each catalog entry is a thin closure that pre-binds a known series_id and routes through the generic loader.

**Tech Stack:** Python 3.10+, `pandas`, `numpy`, existing `puremacro.instruments` (0.5.1), `narrative.sources._http`. Pyodide-compatible (no new runtime deps).

**Pre-implementation baseline:** 474 passing, 9 skipped (puremacro 0.5.1).
**Post-implementation target:** ~510 passing (+~36 new tests), 9 skipped.

---

## File Structure

### Files created
- `puremacro/instruments/_helpers.py` — promoted home of `_csv_to_instrument` + new `_json_to_instrument`
- `puremacro/instruments/external/__init__.py` — re-exports the 3 generic loaders
- `puremacro/instruments/external/fred.py` — `load(*, series_id, api_key=None, observation_start=None, observation_end=None) -> Instrument`
- `puremacro/instruments/external/bis.py` — `load(*, series_id, country="USA", csv_path=None) -> Instrument` (single-series for v1; expand later)
- `puremacro/instruments/external/imf_weo.py` — `load(*, indicator, country, csv_path=None) -> Instrument`
- `tests/test_instruments/external/__init__.py` — empty pytest package marker
- `tests/test_instruments/external/test_helpers.py` — `_json_to_instrument` unit tests
- `tests/test_instruments/external/test_fred.py`
- `tests/test_instruments/external/test_bis.py`
- `tests/test_instruments/external/test_imf_weo.py`

### Files modified
- `puremacro/instruments/literature/_helpers.py` — re-export `_csv_to_instrument` from new home (backwards-compat shim)
- `puremacro/instruments/literature/bbd_epu.py` — update import to `from .._helpers import _csv_to_instrument`
- `puremacro/instruments/literature/caldara_iacoviello_gpr.py` — same import update
- `puremacro/instruments/literature/romer_romer_2004.py` — same import update
- `puremacro/instruments/_catalog.py` — add 7 new `register(InstrumentSpec(...))` entries
- `tests/test_instruments/test_catalog.py` — extend size assertions 29 → 36; add 4 new tests for external-key membership and category flags
- `tests/fixtures/public_api_snapshot.json` — regenerate (new subpackage `puremacro.instruments.external`)
- `pyproject.toml` — `version = "0.5.1" → "0.5.2"`
- `puremacro/__init__.py` — `__version__` bump
- `tests/test_import.py` — bump expected version
- `CHANGELOG.md` — add `## 0.5.2 — 2026-05-03` block at top
- `~/.claude/projects/.../memory/project_puremacro.md` — append iteration entry

---

## Task 1: Promote `_csv_to_instrument` + add `_json_to_instrument`

**Files:**
- Create: `puremacro/instruments/_helpers.py`
- Modify: `puremacro/instruments/literature/_helpers.py` (becomes a re-export shim)
- Modify: `puremacro/instruments/literature/bbd_epu.py` (import update)
- Modify: `puremacro/instruments/literature/caldara_iacoviello_gpr.py` (import update)
- Modify: `puremacro/instruments/literature/romer_romer_2004.py` (import update)
- Create: `tests/test_instruments/external/__init__.py` (empty)
- Create: `tests/test_instruments/external/test_helpers.py`

This task creates the new shared helper module at `puremacro/instruments/_helpers.py` containing both `_csv_to_instrument` (moved up from `literature/`) and a new `_json_to_instrument` for FRED's JSON shape. The literature subpackage's `_helpers.py` becomes a one-line shim that re-exports from the new location to keep any external import paths stable, and the 3 literature loaders that use the helper get one-line import updates.

- [ ] **Step 1: Read the existing helper to confirm exact contents**

Run: `cat puremacro/instruments/literature/_helpers.py`
Expected: see `_csv_to_instrument` definition with `date_col` / `year_col`+`month_col` paths plus the mutual-exclusion guard.

- [ ] **Step 2: Write failing tests for the new `_json_to_instrument` helper**

Create `tests/test_instruments/external/__init__.py` as an empty file.

Create `tests/test_instruments/external/test_helpers.py`:

```python
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
```

- [ ] **Step 3: Run tests to confirm failure**

Run: `pytest tests/test_instruments/external/test_helpers.py -v`
Expected: ImportError on `puremacro.instruments._helpers` (module doesn't exist yet).

- [ ] **Step 4: Create the new shared helper module**

Create `puremacro/instruments/_helpers.py`:

```python
"""Shared adapters for instrument-loader CSV/JSON → Instrument conversion.

This module is the canonical home of ``_csv_to_instrument`` (originally in
``literature/_helpers.py`` and promoted here in 0.5.2 so the new
``external/`` subpackage can share it) and ``_json_to_instrument`` (new in
0.5.2 to support FRED's JSON observations format). Used by both
``literature/`` and ``external/`` loaders.
"""
from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from ._core import Instrument


def _csv_to_instrument(
    df: pd.DataFrame,
    *,
    name: str,
    source: str,
    frequency: str,
    value_col: str,
    date_col: str | None = None,
    year_col: str | None = None,
    month_col: str | None = None,
    metadata: dict[str, Any] | None = None,
    category: str = "literature",
) -> Instrument:
    """Convert a parsed DataFrame into an :class:`Instrument`.

    The DataFrame may carry the date in two forms:

    * a single ``date_col`` column (parsed via ``pd.to_datetime``); or
    * a ``year_col`` + ``month_col`` pair (combined into month-start dates).

    Pass either ``date_col`` OR (``year_col`` + ``month_col``), not both.
    """
    if date_col is not None and (year_col is not None or month_col is not None):
        raise ValueError(
            "pass either date_col or (year_col + month_col), not both"
        )
    if date_col is not None:
        dates = pd.to_datetime(df[date_col])
    elif year_col is not None and month_col is not None:
        dates = pd.to_datetime(
            df[year_col].astype(int).astype(str) + "-"
            + df[month_col].astype(int).astype(str).str.zfill(2) + "-01"
        )
    else:
        raise ValueError(
            "must provide either date_col or both year_col + month_col"
        )
    values = df[value_col].astype(float).values
    series = pd.Series(values, index=dates, name=name).sort_index()
    return Instrument(
        series=series,
        name=name,
        source=source,
        category=category,
        frequency=frequency,
        metadata=metadata or {},
    )


def _json_to_instrument(
    observations: Iterable[dict[str, Any]],
    *,
    name: str,
    source: str,
    frequency: str,
    date_field: str,
    value_field: str,
    metadata: dict[str, Any] | None = None,
    category: str = "external_csv",
    missing_markers: tuple[str, ...] = (".", "", "NaN", "nan"),
) -> Instrument:
    """Convert a list of observation dicts (e.g. FRED JSON response) into an
    :class:`Instrument`.

    Parameters
    ----------
    observations : iterable of dicts
        Each dict has at least the ``date_field`` and ``value_field`` keys.
    date_field : str
        Key name for the date string in each observation.
    value_field : str
        Key name for the value (string-encoded; will be coerced to float).
    missing_markers : tuple of str, default ``(".", "", "NaN", "nan")``
        String values to treat as NaN. FRED uses ``"."``.
    """
    obs_list = list(observations)
    if not obs_list:
        raise ValueError("empty observations list")
    dates = pd.to_datetime([o[date_field] for o in obs_list])
    raw_values = [o[value_field] for o in obs_list]
    values = []
    for v in raw_values:
        if isinstance(v, str) and v in missing_markers:
            values.append(float("nan"))
        else:
            try:
                values.append(float(v))
            except (TypeError, ValueError):
                values.append(float("nan"))
    series = pd.Series(values, index=dates, name=name).sort_index()
    return Instrument(
        series=series,
        name=name,
        source=source,
        category=category,
        frequency=frequency,
        metadata=metadata or {},
    )


__all__ = ["_csv_to_instrument", "_json_to_instrument"]
```

- [ ] **Step 5: Replace `literature/_helpers.py` with a backwards-compat shim**

Edit `puremacro/instruments/literature/_helpers.py`. Replace its entire contents with:

```python
"""Backwards-compat shim: ``_csv_to_instrument`` moved to
:mod:`puremacro.instruments._helpers` in 0.5.2.

This re-export keeps any pre-0.5.2 imports working. New code should
import from the promoted location directly.
"""
from .._helpers import _csv_to_instrument

__all__ = ["_csv_to_instrument"]
```

- [ ] **Step 6: Update the 3 literature loaders' imports**

Edit each of these files. Find the line `from ._helpers import _csv_to_instrument` and change it to `from .._helpers import _csv_to_instrument`:

- `puremacro/instruments/literature/bbd_epu.py`
- `puremacro/instruments/literature/caldara_iacoviello_gpr.py`
- `puremacro/instruments/literature/romer_romer_2004.py`

- [ ] **Step 7: Confirm all tests pass**

Run: `pytest tests/test_instruments/external/test_helpers.py tests/test_instruments/literature/ -v`
Expected: All helper tests pass (5 new) + all 26 existing literature tests still pass (no regression from the import refactor).

Run: `pytest tests/test_instruments/ -v`
Expected: 83 + 5 = 88 tests passing. (Total instruments suite, was 83 in 0.5.1 baseline + 5 new helper tests.)

---

## Task 2: FRED loader + 4 catalog entries

**Files:**
- Create: `puremacro/instruments/external/__init__.py` (initial stub with `load_fred` re-export only)
- Create: `puremacro/instruments/external/fred.py`
- Create: `tests/test_instruments/external/test_fred.py`

FRED's public REST API at `https://api.stlouisfed.org/fred/series/observations` requires a free API key (passed via `?api_key=` query param). The endpoint returns JSON with shape `{"observations": [{"date": "...", "value": "..."}], "count": ..., ...}`. The loader reads the API key from the `FRED_API_KEY` env var if not passed explicitly. We DO NOT register the API key with FRED — users register their own free key.

The 4 curated catalog entries are NFCI (Chicago Fed National Financial Conditions Index, weekly), VIXCLS (CBOE Volatility Index, daily), FEDFUNDS (Effective Federal Funds Rate, monthly), STLFSI4 (St. Louis Fed Financial Stress Index v4, weekly). Each catalog entry is a closure that pre-binds the `series_id` and routes through the generic `fred.load`.

- [ ] **Step 1: Write failing tests**

Create `tests/test_instruments/external/test_fred.py`:

```python
"""Tests for puremacro.instruments.external.fred."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from puremacro.instruments import Instrument
from puremacro.instruments.external.fred import load


_SYNTHETIC_FRED_JSON = json.dumps({
    "realtime_start": "2026-01-01",
    "realtime_end": "2026-01-01",
    "observation_start": "1600-01-01",
    "observation_end": "9999-12-31",
    "units": "lin",
    "output_type": 1,
    "file_type": "json",
    "order_by": "observation_date",
    "sort_order": "asc",
    "count": 4,
    "offset": 0,
    "limit": 100000,
    "observations": [
        {"realtime_start": "2026-01-01", "realtime_end": "2026-01-01",
         "date": "2020-01-01", "value": "1.50"},
        {"realtime_start": "2026-01-01", "realtime_end": "2026-01-01",
         "date": "2020-02-01", "value": "1.75"},
        {"realtime_start": "2026-01-01", "realtime_end": "2026-01-01",
         "date": "2020-03-01", "value": "."},  # missing
        {"realtime_start": "2026-01-01", "realtime_end": "2026-01-01",
         "date": "2020-04-01", "value": "0.25"},
    ],
})


def _patched_safe_get_text(url):
    """Returns synthetic FRED JSON regardless of URL."""
    return _SYNTHETIC_FRED_JSON


def test_load_with_explicit_api_key_returns_instrument(monkeypatch):
    from puremacro.instruments.external import fred as _mod
    monkeypatch.setattr(_mod, "safe_get_text", _patched_safe_get_text)
    inst = load(series_id="FEDFUNDS", api_key="dummy_key", frequency="M")
    assert isinstance(inst, Instrument)
    assert inst.category == "external_csv"
    assert inst.frequency == "M"
    assert inst.name == "fred_FEDFUNDS"


def test_load_uses_env_var_api_key(monkeypatch):
    from puremacro.instruments.external import fred as _mod
    monkeypatch.setattr(_mod, "safe_get_text", _patched_safe_get_text)
    monkeypatch.setenv("FRED_API_KEY", "env_key_value")
    inst = load(series_id="FEDFUNDS", frequency="M")
    assert inst.name == "fred_FEDFUNDS"


def test_load_no_api_key_anywhere_raises(monkeypatch):
    """If neither api_key= nor FRED_API_KEY env var is set, raise."""
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FRED_API_KEY"):
        load(series_id="FEDFUNDS", frequency="M")


def test_load_parses_dot_as_missing(monkeypatch):
    from puremacro.instruments.external import fred as _mod
    monkeypatch.setattr(_mod, "safe_get_text", _patched_safe_get_text)
    inst = load(series_id="FEDFUNDS", api_key="dummy", frequency="M")
    assert pd.isna(inst.series.loc[pd.Timestamp("2020-03-01")])
    assert inst.series.dropna().shape[0] == 3


def test_load_observation_date_range_kwargs(monkeypatch):
    """observation_start and observation_end propagate into the URL."""
    captured_url = {"url": None}
    def _capture(url):
        captured_url["url"] = url
        return _SYNTHETIC_FRED_JSON
    from puremacro.instruments.external import fred as _mod
    monkeypatch.setattr(_mod, "safe_get_text", _capture)
    load(series_id="FEDFUNDS", api_key="dummy", frequency="M",
         observation_start="2010-01-01", observation_end="2020-12-31")
    assert "observation_start=2010-01-01" in captured_url["url"]
    assert "observation_end=2020-12-31" in captured_url["url"]
    assert "series_id=FEDFUNDS" in captured_url["url"]


def test_load_network_failure_raises_clear_error(monkeypatch):
    from puremacro.instruments.external import fred as _mod
    def _fail(_url):
        raise OSError("simulated network failure")
    monkeypatch.setattr(_mod, "safe_get_text", _fail)
    with pytest.raises(RuntimeError, match="FRED"):
        load(series_id="FEDFUNDS", api_key="dummy", frequency="M")


def test_load_instrument_name_includes_series_id(monkeypatch):
    from puremacro.instruments.external import fred as _mod
    monkeypatch.setattr(_mod, "safe_get_text", _patched_safe_get_text)
    inst = load(series_id="NFCI", api_key="dummy", frequency="W")
    assert "NFCI" in inst.name
```

- [ ] **Step 2: Confirm failure**

Run: `pytest tests/test_instruments/external/test_fred.py -v`
Expected: ImportError.

- [ ] **Step 3: Create `external/__init__.py`**

Create `puremacro/instruments/external/__init__.py`:

```python
"""External-CSV instruments — public-data sources outside the
literature/replication ecosystem.

Each provider exposes a generic ``load(*, series_id, ...) -> Instrument``
function for long-tail series, plus 1+ curated catalog entries
(registered in :mod:`puremacro.instruments._catalog`) that pre-bind
common series for easy discovery via :func:`puremacro.instruments.load`.
"""
from .fred import load as load_fred

__all__ = ["load_fred"]
```

- [ ] **Step 4: Implement the FRED loader**

Create `puremacro/instruments/external/fred.py`:

```python
"""FRED (Federal Reserve Economic Data) generic series loader.

FRED is the public economic-data archive of the Federal Reserve Bank of
St. Louis, hosting tens of thousands of macro / financial / regional
series. Access requires a free API key (register at
https://fred.stlouisfed.org/docs/api/api_key.html).

This loader fetches a single series by ID via the
``/fred/series/observations`` endpoint, parses the JSON response, and
returns an :class:`Instrument`. The frequency is supplied by the caller
(FRED's metadata can also report it but we accept it explicitly so
catalog entries are self-documenting).

Reference
---------
Federal Reserve Bank of St. Louis (n.d.). FRED® Economic Data API.
https://fred.stlouisfed.org/docs/api/fred/
"""
from __future__ import annotations

import json
import os
import urllib.parse

import pandas as pd

from .._core import Instrument
from .._helpers import _json_to_instrument
from ...narrative.sources._http import safe_get_text


_BASE = "https://api.stlouisfed.org/fred/series/observations"


def _build_url(
    series_id: str,
    api_key: str,
    *,
    observation_start: str | None = None,
    observation_end: str | None = None,
) -> str:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }
    if observation_start is not None:
        params["observation_start"] = observation_start
    if observation_end is not None:
        params["observation_end"] = observation_end
    return _BASE + "?" + urllib.parse.urlencode(params)


def load(
    *,
    series_id: str,
    api_key: str | None = None,
    frequency: str = "M",
    observation_start: str | None = None,
    observation_end: str | None = None,
) -> Instrument:
    """Load a FRED series as an :class:`Instrument`.

    Parameters
    ----------
    series_id : str
        FRED series identifier (e.g. ``"FEDFUNDS"``, ``"NFCI"``).
    api_key : str | None
        Free FRED API key. If None, read from ``FRED_API_KEY`` env var.
        Raises RuntimeError if neither is set.
    frequency : str, default ``"M"``
        Pandas-style frequency code recorded on the resulting
        Instrument. Pass ``"W"`` for weekly, ``"D"`` for daily, etc.
        FRED returns the series at its native frequency; this kwarg
        is metadata only (no resampling).
    observation_start, observation_end : str | None
        Optional ISO date strings (``"YYYY-MM-DD"``) restricting the
        FRED query window.

    Returns
    -------
    Instrument
        Series indexed by observation date, name ``f"fred_{series_id}"``,
        category ``"external_csv"``.
    """
    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError(
            "FRED API key not provided. Pass api_key= explicitly or set "
            "the FRED_API_KEY environment variable. Free registration: "
            "https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    url = _build_url(
        series_id, key,
        observation_start=observation_start,
        observation_end=observation_end,
    )
    try:
        text = safe_get_text(url)
        payload = json.loads(text)
    except Exception as e:
        raise RuntimeError(
            f"Could not fetch FRED series {series_id!r}. Verify the "
            f"series ID exists at https://fred.stlouisfed.org/series/"
            f"{series_id} and that the API key is valid."
        ) from e
    observations = payload.get("observations", [])
    if not observations:
        raise RuntimeError(
            f"FRED returned no observations for {series_id!r}. Check the "
            f"series ID and the requested date window."
        )
    return _json_to_instrument(
        observations,
        name=f"fred_{series_id}",
        source=f"FRED series {series_id}",
        frequency=frequency,
        date_field="date",
        value_field="value",
        metadata={
            "reference": (
                "Federal Reserve Bank of St. Louis. FRED Economic Data. "
                f"https://fred.stlouisfed.org/series/{series_id}"
            ),
            "series_id": series_id,
        },
    )


__all__ = ["load"]
```

- [ ] **Step 5: Confirm green**

Run: `pytest tests/test_instruments/external/test_fred.py -v`
Expected: 7 passed.

- [ ] **Step 6: Add 4 FRED catalog entries**

(This step lives in Task 5's catalog wiring section. For now, the FRED loader is callable directly via `from puremacro.instruments.external import load_fred; load_fred(series_id="FEDFUNDS")`.)

---

## Task 3: BIS loader + 1 catalog entry

**Files:**
- Create: `puremacro/instruments/external/bis.py`
- Modify: `puremacro/instruments/external/__init__.py` (add `load_bis` re-export)
- Create: `tests/test_instruments/external/test_bis.py`

The Bank for International Settlements publishes statistics in CSV form at https://www.bis.org/statistics/. The credit-to-GDP gap is published as part of the "Total credit to the non-financial sector" statistical release. The bulk CSV at `https://www.bis.org/statistics/totcredit/credit-gap.csv` (verify URL — BIS occasionally changes paths) carries multi-country panel data. For v1 we ship a single-series loader that pulls a country slice; the long tail is reachable by passing `csv_path=` with a user-supplied download.

The single curated catalog entry is `bis_credit_to_gdp_gap_us` — US quarterly credit-to-GDP gap.

- [ ] **Step 1: Write failing tests**

Create `tests/test_instruments/external/test_bis.py`:

```python
"""Tests for puremacro.instruments.external.bis."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from puremacro.instruments import Instrument
from puremacro.instruments.external.bis import load


# Synthetic BIS-style CSV: long format with country code, period, value.
_SYNTHETIC_CSV = """ISO,date,value
US,1999-Q1,3.2
US,1999-Q2,3.5
US,1999-Q3,3.8
GB,1999-Q1,1.1
GB,1999-Q2,1.4
"""


def test_load_with_csv_path_filters_country(tmp_path):
    csv = tmp_path / "bis.csv"
    csv.write_text(_SYNTHETIC_CSV)
    inst = load(series_id="credit_to_gdp_gap", country="US", csv_path=csv)
    assert isinstance(inst, Instrument)
    assert inst.category == "external_csv"
    assert inst.frequency == "Q"
    assert inst.name == "bis_credit_to_gdp_gap_US"
    assert len(inst.series) == 3
    assert inst.series.iloc[0] == pytest.approx(3.2)


def test_load_filters_to_requested_country(tmp_path):
    csv = tmp_path / "bis.csv"
    csv.write_text(_SYNTHETIC_CSV)
    inst = load(series_id="credit_to_gdp_gap", country="GB", csv_path=csv)
    assert len(inst.series) == 2
    assert inst.series.iloc[0] == pytest.approx(1.1)


def test_load_unknown_country_raises_with_available_list(tmp_path):
    csv = tmp_path / "bis.csv"
    csv.write_text(_SYNTHETIC_CSV)
    with pytest.raises(ValueError, match="ZZ"):
        load(series_id="credit_to_gdp_gap", country="ZZ", csv_path=csv)


def test_load_csv_with_wrong_columns_raises(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text("country,quarter,gap\nUS,1999Q1,3.2\n")
    with pytest.raises(ValueError, match="missing expected columns"):
        load(series_id="credit_to_gdp_gap", country="US", csv_path=csv)


def test_load_metadata_has_reference(tmp_path):
    csv = tmp_path / "bis.csv"
    csv.write_text(_SYNTHETIC_CSV)
    inst = load(series_id="credit_to_gdp_gap", country="US", csv_path=csv)
    assert "reference" in inst.metadata
    assert "BIS" in inst.metadata["reference"] or "Bank for International" in inst.metadata["reference"]


def test_load_no_csv_no_network_raises(monkeypatch):
    from puremacro.instruments.external import bis as _mod
    def _fail(_url):
        raise OSError("simulated network failure")
    monkeypatch.setattr(_mod, "safe_get_bytes", _fail)
    with pytest.raises(RuntimeError, match="bis.org"):
        load(series_id="credit_to_gdp_gap", country="US")
```

- [ ] **Step 2: Confirm failure**

Run: `pytest tests/test_instruments/external/test_bis.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement the BIS loader**

Create `puremacro/instruments/external/bis.py`:

```python
"""BIS (Bank for International Settlements) statistics loader.

The BIS publishes cross-country financial statistics — credit-to-GDP
gaps, effective exchange rates, total credit to non-financial sectors,
etc. — at https://www.bis.org/statistics/. Most are quarterly panels
in long-format CSV (one row per country × period).

This v1 loader pulls a single country slice from a single statistical
series. The bulk CSV URL is hardcoded for the credit-to-GDP gap; pass
``csv_path=`` to use a local download or to point at a different BIS
release.

Reference
---------
Bank for International Settlements. BIS Statistics. https://www.bis.org/statistics/
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from .._core import Instrument
from ...narrative.sources._http import safe_get_bytes


# BIS publishes the credit-to-GDP gap as part of the totcredit release.
# The exact mirror path has rotated across BIS website redesigns; the
# user can always pass csv_path= with a manual download.
_DEFAULT_MIRRORS = {
    "credit_to_gdp_gap": "https://www.bis.org/statistics/totcredit/credit-gap.csv",
}

_REFERENCE = (
    "Bank for International Settlements. BIS Statistics — Credit-to-GDP "
    "gap. https://www.bis.org/statistics/totcredit.htm"
)


def load(
    *,
    series_id: str = "credit_to_gdp_gap",
    country: str,
    csv_path: str | Path | None = None,
    frequency: str = "Q",
) -> Instrument:
    """Load a BIS country-slice series as an :class:`Instrument`.

    Parameters
    ----------
    series_id : str, default ``"credit_to_gdp_gap"``
        Identifier of the BIS statistical release. Currently only
        ``"credit_to_gdp_gap"`` has a default mirror URL; for other
        series pass ``csv_path=``.
    country : str
        ISO-2 country code matching the ``ISO`` column of the BIS CSV.
        Required (no sensible default).
    csv_path : str | Path | None
        Optional local path to the BIS CSV. When None, attempt the
        default mirror download.
    frequency : str, default ``"Q"``
        Pandas-style frequency code. Most BIS stats are quarterly.

    Returns
    -------
    Instrument
        Country-filtered series, name ``f"bis_{series_id}_{country}"``.
    """
    if csv_path is not None:
        df = pd.read_csv(csv_path)
    else:
        mirror = _DEFAULT_MIRRORS.get(series_id)
        if mirror is None:
            raise RuntimeError(
                f"BIS series {series_id!r} has no default mirror; pass "
                f"csv_path= with a local download from "
                f"https://www.bis.org/statistics/."
            )
        try:
            raw = safe_get_bytes(mirror)
            df = pd.read_csv(io.BytesIO(raw))
        except Exception as e:
            raise RuntimeError(
                f"Could not fetch BIS {series_id!r} from {mirror}. "
                f"Download a local copy from https://www.bis.org/statistics/ "
                f"and pass csv_path=."
            ) from e

    expected_cols = {"ISO", "date", "value"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"BIS CSV missing expected columns: {sorted(missing)}. "
            f"Got: {sorted(df.columns)}. The default schema is "
            f"long-format with ISO/date/value columns."
        )

    available = sorted(df["ISO"].dropna().unique().tolist())
    if country not in available:
        raise ValueError(
            f"country {country!r} not present in BIS CSV; available: {available}"
        )

    sub = df[df["ISO"] == country].copy()
    sub = sub.dropna(subset=["date", "value"])
    # BIS dates are typically "1999-Q1" or ISO. pd.to_datetime handles "1999Q1"
    # natively; convert "1999-Q1" by stripping the dash.
    raw_dates = sub["date"].astype(str).str.replace("-Q", "Q", regex=False)
    dates = pd.PeriodIndex(raw_dates, freq="Q").to_timestamp(how="start")
    series = pd.Series(
        sub["value"].astype(float).values,
        index=dates,
        name=f"bis_{series_id}_{country}",
    ).sort_index()

    return Instrument(
        series=series,
        name=f"bis_{series_id}_{country}",
        source=f"BIS {series_id} ({country})",
        category="external_csv",
        frequency=frequency,
        metadata={
            "reference": _REFERENCE,
            "series_id": series_id,
            "country": country,
        },
    )


__all__ = ["load"]
```

- [ ] **Step 4: Update `external/__init__.py`**

Edit `puremacro/instruments/external/__init__.py`:

```python
"""External-CSV instruments — public-data sources outside the
literature/replication ecosystem.

Each provider exposes a generic ``load(*, series_id, ...) -> Instrument``
function for long-tail series, plus 1+ curated catalog entries
(registered in :mod:`puremacro.instruments._catalog`) that pre-bind
common series for easy discovery via :func:`puremacro.instruments.load`.
"""
from .fred import load as load_fred
from .bis import load as load_bis

__all__ = ["load_fred", "load_bis"]
```

- [ ] **Step 5: Confirm green**

Run: `pytest tests/test_instruments/external/test_bis.py -v`
Expected: 6 passed.

---

## Task 4: IMF WEO loader + 2 catalog entries

**Files:**
- Create: `puremacro/instruments/external/imf_weo.py`
- Modify: `puremacro/instruments/external/__init__.py` (add `load_imf_weo` re-export)
- Create: `tests/test_instruments/external/test_imf_weo.py`

The IMF World Economic Outlook (WEO) is published twice yearly as a single tab-delimited file containing all countries × all indicators × all years. The schema includes `ISO`, `WEO Subject Code`, and one column per year. Common indicators: `GGXWDG_NGDP` (general government gross debt as % of GDP), `GGXONLB_NGDP` (primary balance as % of GDP), `NGDP_RPCH` (real GDP growth %).

For v1 the loader takes an `indicator` (WEO subject code) and a `country` (ISO3), pulls that one cell-row, melts it into a time series, and returns an `Instrument`. The bulk WEO file is ~5 MB; we accept the bandwidth cost since this is a one-time research data fetch.

- [ ] **Step 1: Write failing tests**

Create `tests/test_instruments/external/test_imf_weo.py`:

```python
"""Tests for puremacro.instruments.external.imf_weo."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.instruments import Instrument
from puremacro.instruments.external.imf_weo import load


# Synthetic WEO-style CSV (tab-separated): subset of columns for tests.
_SYNTHETIC_WEO = (
    "ISO\tWEO Subject Code\t2015\t2016\t2017\t2018\t2019\t2020\n"
    "USA\tGGXWDG_NGDP\t104.5\t106.2\t105.8\t107.1\t108.4\t135.0\n"
    "USA\tGGXONLB_NGDP\t-2.5\t-3.2\t-3.4\t-4.0\t-4.6\t-12.0\n"
    "GBR\tGGXWDG_NGDP\t87.9\t87.9\t86.2\t85.7\t85.2\t104.5\n"
    "USA\tNGDP_RPCH\t2.7\t1.7\t2.3\t2.9\t2.3\t-3.4\n"
)


def test_load_with_csv_path_returns_instrument(tmp_path):
    csv = tmp_path / "weo.csv"
    csv.write_text(_SYNTHETIC_WEO)
    inst = load(indicator="GGXWDG_NGDP", country="USA", csv_path=csv)
    assert isinstance(inst, Instrument)
    assert inst.category == "external_csv"
    assert inst.frequency == "A"
    assert inst.name == "imf_weo_GGXWDG_NGDP_USA"


def test_load_extracts_correct_year_values(tmp_path):
    csv = tmp_path / "weo.csv"
    csv.write_text(_SYNTHETIC_WEO)
    inst = load(indicator="GGXWDG_NGDP", country="USA", csv_path=csv)
    assert inst.series.loc[pd.Timestamp("2015-01-01")] == pytest.approx(104.5)
    assert inst.series.loc[pd.Timestamp("2020-01-01")] == pytest.approx(135.0)
    assert len(inst.series) == 6


def test_load_different_indicator(tmp_path):
    csv = tmp_path / "weo.csv"
    csv.write_text(_SYNTHETIC_WEO)
    inst = load(indicator="GGXONLB_NGDP", country="USA", csv_path=csv)
    assert inst.series.loc[pd.Timestamp("2020-01-01")] == pytest.approx(-12.0)


def test_load_different_country(tmp_path):
    csv = tmp_path / "weo.csv"
    csv.write_text(_SYNTHETIC_WEO)
    inst = load(indicator="GGXWDG_NGDP", country="GBR", csv_path=csv)
    assert inst.series.loc[pd.Timestamp("2015-01-01")] == pytest.approx(87.9)


def test_load_missing_country_indicator_pair_raises(tmp_path):
    csv = tmp_path / "weo.csv"
    csv.write_text(_SYNTHETIC_WEO)
    with pytest.raises(ValueError, match="not found"):
        load(indicator="GGXONLB_NGDP", country="GBR", csv_path=csv)


def test_load_metadata_has_reference(tmp_path):
    csv = tmp_path / "weo.csv"
    csv.write_text(_SYNTHETIC_WEO)
    inst = load(indicator="GGXWDG_NGDP", country="USA", csv_path=csv)
    assert "reference" in inst.metadata
    assert "WEO" in inst.metadata["reference"] or "World Economic Outlook" in inst.metadata["reference"]


def test_load_csv_with_wrong_columns_raises(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text("country\tcode\tval\nUSA\tX\t1.0\n")
    with pytest.raises(ValueError, match="missing expected columns"):
        load(indicator="X", country="USA", csv_path=csv)


def test_load_no_csv_no_network_raises(monkeypatch):
    from puremacro.instruments.external import imf_weo as _mod
    def _fail(_url):
        raise OSError("simulated network failure")
    monkeypatch.setattr(_mod, "safe_get_bytes", _fail)
    with pytest.raises(RuntimeError, match="imf.org"):
        load(indicator="GGXWDG_NGDP", country="USA")
```

- [ ] **Step 2: Confirm failure**

Run: `pytest tests/test_instruments/external/test_imf_weo.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement the IMF WEO loader**

Create `puremacro/instruments/external/imf_weo.py`:

```python
"""IMF World Economic Outlook (WEO) panel loader.

The WEO is published twice yearly as a single tab-delimited file
containing all countries × all macro indicators × all years. The
schema includes ``ISO`` (country code), ``WEO Subject Code`` (indicator
code), and one column per year. Common indicators: ``GGXWDG_NGDP``
(general government gross debt as % of GDP), ``GGXONLB_NGDP``
(primary balance as % of GDP), ``NGDP_RPCH`` (real GDP growth %).

This loader fetches the bulk file, filters to one (indicator, country)
row, and returns the year-by-year time series as an :class:`Instrument`.

Reference
---------
International Monetary Fund. World Economic Outlook Database.
https://www.imf.org/en/Publications/WEO
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .._core import Instrument
from ...narrative.sources._http import safe_get_bytes


# WEO publishes the latest archive at a versioned URL; this is the
# October 2024 release path. The user can pass csv_path= with any
# WEO archive (the schema is stable across releases).
_DEFAULT_MIRROR = (
    "https://www.imf.org/-/media/Files/Publications/WEO/WEO-Database/"
    "2024/October/WEOOct2024all.xls"
)

_REFERENCE = (
    "International Monetary Fund. World Economic Outlook Database. "
    "https://www.imf.org/en/Publications/WEO"
)


def load(
    *,
    indicator: str,
    country: str,
    csv_path: str | Path | None = None,
    frequency: str = "A",
) -> Instrument:
    """Load one (indicator, country) WEO time series as an :class:`Instrument`.

    Parameters
    ----------
    indicator : str
        WEO subject code (e.g. ``"GGXWDG_NGDP"``, ``"GGXONLB_NGDP"``,
        ``"NGDP_RPCH"``).
    country : str
        ISO3 country code (e.g. ``"USA"``, ``"GBR"``).
    csv_path : str | Path | None
        Optional local path to a WEO bulk file (tab-separated). When
        None, attempt the canonical IMF mirror download.
    frequency : str, default ``"A"``
        WEO is published annually.

    Returns
    -------
    Instrument
        Annual series spanning the WEO archive's year range, name
        ``f"imf_weo_{indicator}_{country}"``.
    """
    if csv_path is not None:
        df = pd.read_csv(csv_path, sep="\t", encoding="utf-8")
    else:
        try:
            raw = safe_get_bytes(_DEFAULT_MIRROR)
            df = pd.read_csv(io.BytesIO(raw), sep="\t", encoding="utf-8")
        except Exception as e:
            raise RuntimeError(
                "Could not fetch IMF WEO bulk file. Download a copy from "
                "https://www.imf.org/en/Publications/WEO/weo-database/ "
                "and pass csv_path=."
            ) from e

    expected_cols = {"ISO", "WEO Subject Code"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"WEO file missing expected columns: {sorted(missing)}. "
            f"Got: {sorted(df.columns)[:10]}... "
            f"Expected long-format with ISO and 'WEO Subject Code' columns."
        )

    mask = (df["ISO"] == country) & (df["WEO Subject Code"] == indicator)
    sub = df[mask]
    if sub.empty:
        raise ValueError(
            f"WEO row not found for indicator={indicator!r}, country={country!r}."
        )
    if len(sub) > 1:
        # Defensive: take the first row.
        sub = sub.iloc[[0]]

    # Year columns are integer-typed strings ("2015", "2016", ...). Pick
    # them out via regex.
    year_pattern = re.compile(r"^\d{4}$")
    year_cols = [c for c in sub.columns if year_pattern.match(str(c))]
    if not year_cols:
        raise ValueError(
            "No year columns (4-digit integer-named) found in WEO file."
        )

    # Build a Series indexed by year-start date.
    row = sub.iloc[0]
    raw_values = []
    dates = []
    for c in year_cols:
        val = row[c]
        # WEO uses "n/a" or "--" or NaN for missing; pd.read_csv may
        # return either string or NaN. Coerce to float-or-NaN.
        if pd.isna(val):
            num = float("nan")
        elif isinstance(val, str) and val.strip() in ("n/a", "--", ""):
            num = float("nan")
        else:
            try:
                num = float(val)
            except (TypeError, ValueError):
                num = float("nan")
        raw_values.append(num)
        dates.append(pd.Timestamp(f"{c}-01-01"))

    series = pd.Series(
        raw_values,
        index=pd.DatetimeIndex(dates),
        name=f"imf_weo_{indicator}_{country}",
    ).sort_index()

    return Instrument(
        series=series,
        name=f"imf_weo_{indicator}_{country}",
        source=f"IMF WEO {indicator} ({country})",
        category="external_csv",
        frequency=frequency,
        metadata={
            "reference": _REFERENCE,
            "indicator": indicator,
            "country": country,
        },
    )


__all__ = ["load"]
```

- [ ] **Step 4: Update `external/__init__.py`** (final version)

Edit `puremacro/instruments/external/__init__.py`:

```python
"""External-CSV instruments — public-data sources outside the
literature/replication ecosystem.

Each provider exposes a generic ``load(*, series_id, ...) -> Instrument``
function for long-tail series, plus 1+ curated catalog entries
(registered in :mod:`puremacro.instruments._catalog`) that pre-bind
common series for easy discovery via :func:`puremacro.instruments.load`.
"""
from .fred import load as load_fred
from .bis import load as load_bis
from .imf_weo import load as load_imf_weo

__all__ = ["load_fred", "load_bis", "load_imf_weo"]
```

- [ ] **Step 5: Confirm green**

Run: `pytest tests/test_instruments/external/test_imf_weo.py -v`
Expected: 8 passed.

Run: `pytest tests/test_instruments/external/ -v`
Expected: 5 (helpers) + 7 (fred) + 6 (bis) + 8 (imf_weo) = 26 tests passing.

---

## Task 5: Catalog wiring (7 entries) + 0.5.2 release coordination

**Files:**
- Modify: `puremacro/instruments/_catalog.py` — append a new "External" section
- Modify: `tests/test_instruments/test_catalog.py` — bump size assertions, add external-key tests
- Modify: `tests/fixtures/public_api_snapshot.json` (regenerate)
- Modify: `pyproject.toml` (`0.5.1 → 0.5.2`)
- Modify: `puremacro/__init__.py` (`__version__`)
- Modify: `tests/test_import.py`
- Modify: `CHANGELOG.md`
- Modify: `~/.claude/projects/.../memory/project_puremacro.md`

The 7 new entries are: 4 FRED (NFCI weekly, VIXCLS daily, FEDFUNDS monthly, STLFSI4 weekly), 1 BIS (US credit-to-GDP gap quarterly), 2 IMF WEO (US gross debt/GDP annual, US primary balance/GDP annual). All `requires_network=True` (or `requires_fixture=True` if FRED API key is needed at runtime).

For FRED entries, `requires_network=True` AND `requires_fixture=True` — the user must supply both network access AND an API key (which is conceptually a "fixture" — a user-supplied secret). Both flags `True` is honest.

- [ ] **Step 1: Update test_catalog.py size assertions**

Edit `tests/test_instruments/test_catalog.py`. Find `test_total_catalog_size_is_exactly_29` and update to 36:

```python
def test_total_catalog_size_is_exactly_36():
    """6 replications + 6 connectors + 1 monetary HFI + 12 stubs + 4 literature + 7 external = 36."""
    assert len(_registry._REGISTRY) == 36
```

Find `test_total_catalog_size_at_least_29` and update similarly:

```python
def test_total_catalog_size_at_least_36():
    """6 replications + 6 connectors + 1 monetary HFI + 12 stubs + 4 literature + 7 external = 36."""
    assert len(_registry._REGISTRY) >= 36
```

(Update the test names too — `_29` → `_36`.)

Append at the end of the file:

```python
_EXPECTED_EXTERNAL_KEYS = {
    "fred_nfci",
    "fred_vixcls",
    "fred_fedfunds",
    "fred_stlfsi4",
    "bis_credit_to_gdp_gap_us",
    "imf_weo_debt_gdp_usa",
    "imf_weo_primary_balance_gdp_usa",
}


def test_all_seven_external_entries_registered():
    keys = set(_registry._REGISTRY.keys())
    missing = _EXPECTED_EXTERNAL_KEYS - keys
    assert not missing, f"external entries missing: {missing}"


def test_every_external_entry_is_external_csv_category():
    for key in _EXPECTED_EXTERNAL_KEYS:
        spec = _registry._REGISTRY[key]
        assert spec.category == "external_csv", (
            f"{key} category={spec.category!r}, expected 'external_csv'"
        )


def test_every_external_entry_requires_network_or_fixture():
    for key in _EXPECTED_EXTERNAL_KEYS:
        spec = _registry._REGISTRY[key]
        assert spec.requires_network or spec.requires_fixture


def test_every_external_entry_has_non_empty_reference():
    for key in _EXPECTED_EXTERNAL_KEYS:
        spec = _registry._REGISTRY[key]
        assert spec.reference and len(spec.reference) > 10
```

- [ ] **Step 2: Confirm failure**

Run: `pytest tests/test_instruments/test_catalog.py -v -k "external or 36"`
Expected: AssertionError on size + missing external keys.

- [ ] **Step 3: Append the 7 catalog entries to _catalog.py**

Edit `puremacro/instruments/_catalog.py`. Append at the end (after the existing literature section):

```python
# --------------------------------------------------------------------------
# External-CSV providers (7) — see puremacro/instruments/external/
# --------------------------------------------------------------------------
from .external import load_fred, load_bis, load_imf_weo


# --- FRED (4 entries) ---
def _make_fred_loader(series_id: str, frequency: str):
    def _load(**kwargs):
        return load_fred(series_id=series_id, frequency=frequency, **kwargs)
    return _load


register(InstrumentSpec(
    key="fred_nfci",
    name="Chicago Fed NFCI (weekly)",
    category="external_csv",
    description=(
        "Chicago Fed National Financial Conditions Index. Weekly, "
        "1971-present. Standard composite measure of risk, liquidity, "
        "and leverage in US financial markets. Requires FRED API key "
        "(free; set FRED_API_KEY env var)."
    ),
    reference="Federal Reserve Bank of Chicago. NFCI. https://fred.stlouisfed.org/series/NFCI",
    loader=_make_fred_loader("NFCI", "W"),
    country="USA",
    frequency="W",
    requires_network=True,
    requires_fixture=True,
))


register(InstrumentSpec(
    key="fred_vixcls",
    name="CBOE Volatility Index VIX (daily)",
    category="external_csv",
    description=(
        "CBOE VIX, daily close. The canonical proxy for US stock-market "
        "uncertainty. Requires FRED API key (free; set FRED_API_KEY env "
        "var)."
    ),
    reference="Chicago Board Options Exchange. VIX. https://fred.stlouisfed.org/series/VIXCLS",
    loader=_make_fred_loader("VIXCLS", "D"),
    country="USA",
    frequency="D",
    requires_network=True,
    requires_fixture=True,
))


register(InstrumentSpec(
    key="fred_fedfunds",
    name="Effective Federal Funds Rate (monthly)",
    category="external_csv",
    description=(
        "Monthly effective federal funds rate. The standard policy-rate "
        "series in US monetary VARs. Requires FRED API key (free; set "
        "FRED_API_KEY env var)."
    ),
    reference="Board of Governors of the Federal Reserve. FEDFUNDS. https://fred.stlouisfed.org/series/FEDFUNDS",
    loader=_make_fred_loader("FEDFUNDS", "M"),
    country="USA",
    frequency="M",
    requires_network=True,
    requires_fixture=True,
))


register(InstrumentSpec(
    key="fred_stlfsi4",
    name="St. Louis Fed Financial Stress Index v4 (weekly)",
    category="external_csv",
    description=(
        "St. Louis Fed Financial Stress Index, version 4. Weekly. A "
        "complementary financial-conditions measure to the Chicago Fed "
        "NFCI, with different weighting. Requires FRED API key (free; "
        "set FRED_API_KEY env var)."
    ),
    reference="Federal Reserve Bank of St. Louis. STLFSI4. https://fred.stlouisfed.org/series/STLFSI4",
    loader=_make_fred_loader("STLFSI4", "W"),
    country="USA",
    frequency="W",
    requires_network=True,
    requires_fixture=True,
))


# --- BIS (1 entry) ---
def _load_bis_credit_gap_us(**kwargs):
    return load_bis(series_id="credit_to_gdp_gap", country="US", **kwargs)


register(InstrumentSpec(
    key="bis_credit_to_gdp_gap_us",
    name="BIS US credit-to-GDP gap (quarterly)",
    category="external_csv",
    description=(
        "US credit-to-GDP gap from the BIS total-credit statistics. "
        "Quarterly. The BCBS countercyclical-buffer reference variable. "
        "Pass csv_path= to skip the network call."
    ),
    reference="Bank for International Settlements. Credit-to-GDP gaps. https://www.bis.org/statistics/totcredit.htm",
    loader=_load_bis_credit_gap_us,
    country="USA",
    frequency="Q",
    requires_network=True,
    requires_fixture=False,
))


# --- IMF WEO (2 entries) ---
def _make_weo_loader(indicator: str, country: str):
    def _load(**kwargs):
        return load_imf_weo(indicator=indicator, country=country, **kwargs)
    return _load


register(InstrumentSpec(
    key="imf_weo_debt_gdp_usa",
    name="IMF WEO US general government gross debt (% of GDP, annual)",
    category="external_csv",
    description=(
        "US general government gross debt as % of GDP, from the IMF "
        "World Economic Outlook database. Annual. Pass csv_path= to "
        "skip the network call."
    ),
    reference="International Monetary Fund. World Economic Outlook Database — GGXWDG_NGDP indicator. https://www.imf.org/en/Publications/WEO",
    loader=_make_weo_loader("GGXWDG_NGDP", "USA"),
    country="USA",
    frequency="A",
    requires_network=True,
    requires_fixture=False,
))


register(InstrumentSpec(
    key="imf_weo_primary_balance_gdp_usa",
    name="IMF WEO US general government primary balance (% of GDP, annual)",
    category="external_csv",
    description=(
        "US general government primary balance as % of GDP, from the "
        "IMF World Economic Outlook database. Annual. Pass csv_path= "
        "to skip the network call."
    ),
    reference="International Monetary Fund. World Economic Outlook Database — GGXONLB_NGDP indicator. https://www.imf.org/en/Publications/WEO",
    loader=_make_weo_loader("GGXONLB_NGDP", "USA"),
    country="USA",
    frequency="A",
    requires_network=True,
    requires_fixture=False,
))
```

- [ ] **Step 4: Confirm catalog tests green**

Run: `pytest tests/test_instruments/test_catalog.py -v`
Expected: All catalog tests pass — total entries 36, 7 external keys present, all flagged correctly.

- [ ] **Step 5: Confirm full instruments suite green**

Run: `pytest tests/test_instruments/ -v`
Expected: ~117 passed (88 from end of Task 1 + 7 fred + 6 bis + 8 imf_weo + 4 catalog tests + ... actual count may vary slightly).

- [ ] **Step 6: Verify zero warnings on import**

Run: `python -W error -c "import puremacro.instruments; print(len(puremacro.instruments._registry._REGISTRY))"`
Expected: prints `36`, no warnings.

- [ ] **Step 7: Regenerate the public-API snapshot**

Run from the repo root:

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro" && \
python -c "
import sys; sys.path.insert(0, 'tests')
from test_public_api import _collect_current_api
import json
print(json.dumps(_collect_current_api(), indent=2))
" > tests/fixtures/public_api_snapshot.json
```

Then run `pytest tests/test_public_api.py -v`. Expect PASS.

- [ ] **Step 8: Spot-check the snapshot diff**

Run: `grep -c "external" tests/fixtures/public_api_snapshot.json`
Expected: at least 5 (one for each external loader module + the parent subpackage).

- [ ] **Step 9: Bump pyproject.toml**

Edit `pyproject.toml`. Change `version = "0.5.1"` to `version = "0.5.2"`.

- [ ] **Step 10: Bump `__version__`**

Edit `puremacro/__init__.py`. Change `__version__ = "0.5.1"` to `__version__ = "0.5.2"`.

- [ ] **Step 11: Bump test_import expected version**

Edit `tests/test_import.py`. Change `assert puremacro.__version__ == "0.5.1"` to `assert puremacro.__version__ == "0.5.2"`.

- [ ] **Step 12: Confirm import test green**

Run: `pytest tests/test_import.py -v`
Expected: PASS.

- [ ] **Step 13: Add CHANGELOG entry**

Edit `CHANGELOG.md`. Insert immediately after the file header and before `## 0.5.1 — 2026-05-03`:

```markdown
## 0.5.2 — 2026-05-03

Patch release — adds 3 new external-data providers (FRED, BIS, IMF WEO) under `puremacro.instruments.external`. Catalog grows 29 → 36 with 7 new entries (4 FRED, 1 BIS, 2 IMF WEO). The shared `_csv_to_instrument` helper is promoted from `literature/_helpers.py` to `instruments/_helpers.py` so both subpackages share it; the legacy import path keeps working via a backwards-compat shim. New `_json_to_instrument` helper handles FRED's JSON observation format.

### Added
- `puremacro.instruments.external` — new subpackage with 3 generic provider loaders, all returning `Instrument` directly:
  - `load_fred(*, series_id, api_key=None, frequency="M", observation_start=None, observation_end=None)` — fetches any FRED series via the public REST API. Reads `FRED_API_KEY` env var if `api_key` is not passed; raises RuntimeError if neither is set. Handles FRED's `"."` missing marker.
  - `load_bis(*, series_id, country, csv_path=None, frequency="Q")` — pulls a country slice from BIS statistical CSVs. Default mirror covers `series_id="credit_to_gdp_gap"`; for other series pass `csv_path=` with a manual download.
  - `load_imf_weo(*, indicator, country, csv_path=None, frequency="A")` — pulls one (indicator, country) cell from the IMF WEO bulk archive. Default mirror points at the October 2024 release.
- `puremacro.instruments._helpers._json_to_instrument()` — new shared adapter for JSON observation lists (FRED-style).
- 7 new catalog entries:
  - `fred_nfci` (Chicago Fed NFCI, weekly), `fred_vixcls` (VIX, daily), `fred_fedfunds` (effective FFR, monthly), `fred_stlfsi4` (St. Louis FSI v4, weekly) — all `requires_network=True` AND `requires_fixture=True` (the FRED API key is the "fixture").
  - `bis_credit_to_gdp_gap_us` (BIS US credit-to-GDP gap, quarterly).
  - `imf_weo_debt_gdp_usa` (US gross debt/GDP, annual), `imf_weo_primary_balance_gdp_usa` (US primary balance/GDP, annual).

### Internal
- `puremacro.instruments._helpers` (new module at the subpackage root) becomes the canonical home of `_csv_to_instrument` and the new `_json_to_instrument`. The literature subpackage's `_helpers.py` is now a backwards-compat shim re-exporting from the new location.
- `tests/test_instruments/external/` — new test directory: 26 tests across helpers (5), FRED (7), BIS (6), IMF WEO (8).
- `tests/test_instruments/test_catalog.py` — size assertions tightened to 36 entries; new tests for external-key membership, category flag, network/fixture flags.
- `tests/fixtures/public_api_snapshot.json` regenerated to record the new `puremacro.instruments.external` subpackage and 3 loader modules.

### Out of scope (still deferred)
- `Instrument.compose()` operator for combining shock series.
- Per-record country threading in `score_keyword`.
- JSON serializability of `Instrument.metadata`.
- Additional FRED catalog entries beyond the 4 most-cited series.
- BIS and IMF SDMX API integration (we use bulk-CSV downloads only).

### Tests
- Pre-release baseline: 474 passing, 9 skipped (0.5.1).
- Post-release: ~510+ passing, 9 skipped (~36 new tests).
```

- [ ] **Step 14: Append memory entry**

Edit `/Users/jalonso/.claude/projects/-Users-jalonso-Library-CloudStorage-GoogleDrive-jorge-alonsoortiz-gmail-com-My-Drive-MAV/memory/project_puremacro.md`. Append at the end:

```markdown

**Iteration N+9 step 4 done (2026-05-03) — released as 0.5.2 (patch):**
- New subpackage `puremacro.instruments.external` with 3 generic provider loaders: `load_fred(series_id, api_key=)`, `load_bis(series_id, country, csv_path=)`, `load_imf_weo(indicator, country, csv_path=)`. Each returns `Instrument` directly.
- FRED requires API key — read from `FRED_API_KEY` env var with explicit kwarg override; clear RuntimeError if neither set. Returns native FRED frequency (W/D/M/Q); no resampling.
- BIS loader handles long-format CSV with `ISO/date/value` columns; supports country filter; default mirror for `credit_to_gdp_gap` series, `csv_path=` fallback for everything else.
- IMF WEO loader parses the bulk tab-separated archive (one row per country × indicator with one column per year); pulls one (indicator, country) row and emits annual time series.
- Promoted `_csv_to_instrument` from `literature/_helpers.py` to `instruments/_helpers.py` so both `literature/` and `external/` share it. Legacy import path kept via backwards-compat shim.
- New `_json_to_instrument` helper for FRED's JSON observations format. Handles `"."` and other missing markers, coerces strings to floats safely.
- Catalog grew 29 → 36: +4 FRED (NFCI, VIXCLS, FEDFUNDS, STLFSI4), +1 BIS (US credit-to-GDP gap), +2 IMF WEO (US debt/GDP, US primary balance/GDP).
- All FRED entries flagged `requires_network=True AND requires_fixture=True` (the API key is the "fixture"). BIS/IMF entries flagged `requires_network=True` only.
- 36 new tests across `tests/test_instruments/external/`.
- Plan file: `uncertainty_examples/puremacro/docs/plans/2026-05-03-instruments-external-fred-bis-imf-052.md`.

**0.5.2 still deferred (Trim B / future patches):**
- `Instrument.compose()` operator.
- Per-record country threading in `score_keyword`.
- JSON serializability of `Instrument.metadata`.
- More curated FRED entries (long tail of macro series).
- BIS/IMF SDMX API integration (we ship bulk-CSV download only).

**How to apply:** When the user asks "can I get FED funds rate / VIX / NFCI / debt-to-GDP for VAR work?", they can now `pi.load("fred_fedfunds")` etc. The FRED entries need the user's free API key set as FRED_API_KEY in the environment. For non-catalogued FRED series, call `from puremacro.instruments.external import load_fred; load_fred(series_id="ANYTHING")` directly.
```

- [ ] **Step 15: Final test run**

Run: `pytest -x -q 2>&1 | tail -5`
Expected: ~510 passed, 9 skipped.

- [ ] **Step 16: Pyodide-compat regression check**

Run: `pytest tests/test_pyodide_compat.py -v`
Expected: PASS (no new runtime deps; FRED/BIS/IMF loaders use existing `safe_get_*` helpers).

- [ ] **Step 17: Sanity-check the public surface**

Run: `python -c "from puremacro.instruments.external import load_fred, load_bis, load_imf_weo; print('OK')"`
Expected: `OK`.

Run: `python -c "from puremacro.instruments import list_available; df = list_available(include_unavailable=True, category='external_csv'); print(f'external entries: {len(df)}'); print(df['key'].tolist())"`
Expected: prints `external entries: 7` followed by the 7 keys.

Run: `python -c "from puremacro.instruments import describe; print(describe('fred_nfci'))"`
Expected: multi-line description mentioning Chicago Fed, NFCI, FRED API key.

---

## Self-Review Checklist

After implementation:

1. **Spec coverage:** All 3 providers and 7 catalog entries implemented?
   - [x] `_csv_to_instrument` promotion + `_json_to_instrument` → Task 1
   - [x] FRED loader + 4 catalog entries (NFCI, VIXCLS, FEDFUNDS, STLFSI4) → Tasks 2 & 5
   - [x] BIS loader + 1 catalog entry (US credit-to-GDP gap) → Tasks 3 & 5
   - [x] IMF WEO loader + 2 catalog entries (US debt/GDP, US primary balance/GDP) → Tasks 4 & 5
   - [x] Snapshot regen + version bump + CHANGELOG + memory → Task 5

2. **Placeholder scan:** No "TBD", "implement later", "appropriate error handling" patterns.

3. **Type consistency:**
   - All 3 generic loaders return `Instrument` directly.
   - Catalog `loader=` is always either `_make_fred_loader(...)`, a direct function reference, or a closure that calls the appropriate `load_*`.
   - FRED entries flag both `requires_network=True` AND `requires_fixture=True` consistently.
   - BIS and IMF WEO entries flag `requires_network=True` only.
   - Catalog keys: `fred_<lowercase>`, `bis_<series>_<country>` (US lowercase), `imf_weo_<indicator>_<country>` (USA uppercase). The Bloom catalog precedent uses lowercase descriptors; we follow that for the FRED entries (e.g., `fred_nfci` not `fred_NFCI`).
   - Instrument `name=` field uses the same convention as the catalog key for consistency in `pi.list_available()` output.

4. **Pyodide hygiene:** All HTTP through existing `safe_get_*` helpers. No new runtime deps.

5. **Citation discipline:** Every catalog entry has a non-empty `reference`. FRED entries cite both the source institution AND the FRED URL.

6. **Backwards compatibility:** No changes to `Instrument`, `InstrumentLike`, `InstrumentSpec`, `register`, `list_available`, `load`, `describe`. Literature subpackage's `_helpers.py` becomes a shim that preserves the import path. The 4 literature loaders' import lines change but their behavior does not.

7. **API key UX:** FRED loader gives a clear error message naming both override paths (`api_key=` kwarg AND `FRED_API_KEY` env var) and the registration URL.

8. **Failure-path tests:** Each network loader has a `monkeypatch`-based test confirming the RuntimeError message names the canonical source. FRED additionally has a missing-API-key test.
