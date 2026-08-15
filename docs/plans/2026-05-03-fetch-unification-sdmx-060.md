# puremacro 0.6.0 — Fetch Unification + OECD SDMX + Registry Integration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **This workspace is NOT a git repo (Google Drive sync); skip every "commit" step the meta-skill suggests.**

**Goal:** Ship 0.6.0 — promote the hardened HTTP helpers to a top-level `puremacro._http` module so all fetchers share one path; restructure `puremacro.fetch` as a subpackage with a new generic `sdmx_get` (OECD/Eurostat/ECB/IMF) loader; integrate `fetch.fred` / `fetch.bis_neer` / `fetch.sdmx` into the `Instrument` catalog so they're discoverable via `pi.list_available()`.

**Architecture:** **Three** parallel HTTP-fetch infrastructures collapse to one: the canonical home moves from `narrative/sources/_http.py` (where it landed in 0.4.0) to top-level `puremacro/_http.py` (sibling to `_linalg.py`); `narrative/sources/_http.py` becomes a backwards-compat shim; `puremacro.fetch._safe_urlopen` becomes a thin wrapper that delegates to the new shared `safe_get_bytes`. `puremacro/fetch.py` (single file) becomes `puremacro/fetch/` (subpackage) with `__init__.py` re-exporting `fetch_fred`, `fetch_fred_alfred`, plus new `fetch.sdmx` module exposing `sdmx_get(provider, dataflow, key, ...)` for SDMX-CSV providers (OECD, Eurostat, ECB, IMF). New catalog entries surface these in `pi.list_available()`. Pure-additive on the public surface — no breaking changes.

**Tech Stack:** Python 3.10+, `pandas`, `numpy`. Pyodide-compatible (no new runtime deps).

**Pre-implementation baseline:** 536 passing, 9 skipped (puremacro 0.5.4).
**Post-implementation target:** ~565 passing (+~29 new tests), 9 skipped.

---

## File Structure

### Files created
- `puremacro/_http.py` — promoted canonical home of `safe_get_bytes` / `safe_get_text` / `safe_get_json` + `USER_AGENT` / `DEFAULT_TIMEOUT` constants
- `puremacro/fetch/__init__.py` — re-exports `_safe_urlopen` (legacy alias), `fetch_fred`, `fetch_fred_alfred` (preserved); new `sdmx_get`, `oecd_sdmx_instrument`
- `puremacro/fetch/_classic.py` — extracted body of the old `puremacro/fetch.py` (`fetch_fred`, `fetch_fred_alfred`, `_safe_urlopen` shim)
- `puremacro/fetch/sdmx.py` — generic SDMX-CSV fetcher + OECD-specific `oecd_sdmx_instrument` Instrument loader
- `tests/test_http_unified.py` — verifies new `puremacro._http` exposes the same surface; legacy paths still work
- `tests/test_fetch/__init__.py` — empty pytest package marker
- `tests/test_fetch/test_classic.py` — tests for `fetch_fred`/`fetch_fred_alfred` (extracted from existing if any, or new smoke tests)
- `tests/test_fetch/test_sdmx.py` — `sdmx_get` + `oecd_sdmx_instrument` tests

### Files modified
- `puremacro/narrative/sources/_http.py` — becomes a backwards-compat shim re-exporting from `puremacro._http`
- `puremacro/fetch.py` — DELETED (its contents move into `puremacro/fetch/_classic.py`; `fetch/__init__.py` provides the import surface)
- `puremacro/bis_neer.py` — change `from .fetch import _safe_urlopen` → `from ._http import safe_get_bytes` (and update the call site)
- `puremacro/data.py` — append a one-line docstring note pointing at `puremacro.fetch` for fetchers (NOT a code change, just documentation)
- `puremacro/instruments/_catalog.py` — add 4 new `register(InstrumentSpec(...))` entries (FRED public CSV, BIS NEER, OECD-STAN debt-to-GDP, OECD-STAN unemployment as a second SDMX example)
- `tests/test_instruments/test_catalog.py` — bump size assertion 36 → 40; add 4 new tests for the new entries
- `tests/fixtures/public_api_snapshot.json` — regenerate (new `puremacro._http`, `puremacro.fetch` subpackage, `puremacro.fetch.sdmx`)
- `pyproject.toml` — `version = "0.5.4" → "0.6.0"`
- `puremacro/__init__.py` — `__version__` bump
- `tests/test_import.py` — bump expected version
- `CHANGELOG.md` — add `## 0.6.0 — 2026-05-03` block at top
- `~/.claude/projects/.../memory/project_puremacro.md` — append iteration entry

---

## Task 1: Promote `_http` to top-level `puremacro/_http.py`

**Files:**
- Create: `puremacro/_http.py`
- Modify: `puremacro/narrative/sources/_http.py` (becomes shim)
- Create: `tests/test_http_unified.py`

The current `puremacro/narrative/sources/_http.py` has the hardened helpers (UA override, SSL fallback). Move the body to a top-level `puremacro/_http.py`; the narrative path becomes a re-export shim.

- [ ] **Step 1: Read the existing helper file and copy its contents**

Run: `cat puremacro/narrative/sources/_http.py`

You'll see the canonical `_request` + `safe_get_bytes/text/json` + `USER_AGENT` + `DEFAULT_TIMEOUT` definitions. Note them — you'll move them verbatim.

- [ ] **Step 2: Write failing tests**

Create `tests/test_http_unified.py`:

```python
"""Tests confirming puremacro._http is the canonical home of HTTP helpers,
and that the legacy narrative.sources._http path keeps working as a shim."""
from __future__ import annotations

import pytest


def test_top_level_http_module_imports():
    """puremacro._http must expose the 5 canonical names."""
    from puremacro._http import (
        safe_get_bytes, safe_get_text, safe_get_json,
        USER_AGENT, DEFAULT_TIMEOUT,
    )
    assert callable(safe_get_bytes)
    assert callable(safe_get_text)
    assert callable(safe_get_json)
    assert isinstance(USER_AGENT, str)
    assert isinstance(DEFAULT_TIMEOUT, float)


def test_legacy_narrative_path_still_imports():
    """puremacro.narrative.sources._http must remain importable."""
    from puremacro.narrative.sources._http import (
        safe_get_bytes, safe_get_text, safe_get_json,
        USER_AGENT, DEFAULT_TIMEOUT,
    )
    assert callable(safe_get_bytes)


def test_legacy_path_returns_same_objects():
    """The shim must re-export the same function objects, not redefine."""
    from puremacro._http import safe_get_bytes as canonical
    from puremacro.narrative.sources._http import safe_get_bytes as shim
    assert canonical is shim


def test_user_agent_override_kwarg_present():
    """Verify the keyword-only `user_agent=` override survives the move."""
    import inspect
    from puremacro._http import safe_get_bytes
    sig = inspect.signature(safe_get_bytes)
    params = sig.parameters
    assert "user_agent" in params
    assert params["user_agent"].kind == inspect.Parameter.KEYWORD_ONLY
```

Run: `pytest tests/test_http_unified.py -v` → expect ImportError on `puremacro._http`.

- [ ] **Step 3: Create `puremacro/_http.py`** with the canonical helpers

Create `puremacro/_http.py`. Copy the ENTIRE contents of `puremacro/narrative/sources/_http.py` verbatim — this becomes the canonical home. Update only the docstring's first paragraph to reflect the new home:

```python
"""Shared HTTP helpers for puremacro fetchers and connectors.

This module is the single canonical home of ``safe_get_bytes``,
``safe_get_text``, ``safe_get_json`` and the supporting ``USER_AGENT``
+ ``DEFAULT_TIMEOUT`` constants. Promoted from
``puremacro.narrative.sources._http`` in 0.6.0 so all fetchers
(narrative connectors, ``puremacro.fetch.*``, instrument loaders)
share one hardened path.

See ``puremacro/narrative/sources/RETRY_POLICY.md`` for the
contract every consumer adheres to: 30s default timeout, one-shot
SSL fallback for older endpoints with stale CA bundles, optional
keyword-only ``user_agent=`` override (added in 0.4.1) for endpoints
behind a WAF that blocks the default agent string.
"""
# ... rest is verbatim copy of the existing _http.py body ...
```

(Keep all functions, the `USER_AGENT` / `DEFAULT_TIMEOUT` constants, and `__all__` exactly as they are in the source file.)

- [ ] **Step 4: Replace `puremacro/narrative/sources/_http.py` with a shim**

Replace the entire contents of `puremacro/narrative/sources/_http.py` with:

```python
"""Backwards-compat shim: HTTP helpers moved to :mod:`puremacro._http`
in 0.6.0.

This re-export keeps any pre-0.6.0 imports
(e.g. ``from puremacro.narrative.sources._http import safe_get_bytes``)
working. New code should import from the promoted location:
``from puremacro._http import safe_get_bytes``.
"""
from .._http import (
    USER_AGENT,
    DEFAULT_TIMEOUT,
    safe_get_bytes,
    safe_get_text,
    safe_get_json,
)
# Note: the legacy `_request` was a private helper. If any caller
# references `puremacro.narrative.sources._http._request`, also expose:
from .._http import _request  # noqa: F401  — legacy private import surface

__all__ = [
    "USER_AGENT", "DEFAULT_TIMEOUT",
    "safe_get_bytes", "safe_get_text", "safe_get_json",
]
```

(Note: also re-export `_request` to be safe — narrative code or tests might import the private helper.)

- [ ] **Step 5: Confirm tests pass**

Run: `pytest tests/test_http_unified.py -v` → expect 4 passed.

Run: `pytest tests/test_narrative_offline.py tests/test_instruments/external/ -v 2>&1 | tail -10` → expect zero regressions (the shim must be transparent).

---

## Task 2: Migrate `fetch._safe_urlopen` to delegate to `puremacro._http`

**Files:**
- Modify: `puremacro/fetch.py` (single-line delegation)

This task happens BEFORE the subpackage restructure (Task 5) so we have one less moving part per task. After this task, `fetch.py` is still a single file but `_safe_urlopen` delegates to the unified path.

- [ ] **Step 1: Read current `puremacro/fetch.py`** (already done in survey)

Note: `_safe_urlopen` returns `bytes`, takes `(url, timeout=30.0)`. Three call sites: `fetch_fred`, `fetch_fred_alfred`, and `bis_neer.py`.

- [ ] **Step 2: Replace the `_safe_urlopen` function body**

Find this in `puremacro/fetch.py`:

```python
def _safe_urlopen(url: str, timeout: float = 30.0) -> bytes:
    # ... existing body using urllib.request, ssl, etc. ...
```

Replace its body with:

```python
def _safe_urlopen(url: str, timeout: float = 30.0) -> bytes:
    """Fetch ``url`` and return raw bytes.

    Thin wrapper for backwards compat — delegates to the unified
    :func:`puremacro._http.safe_get_bytes`. Any policy fix to that
    helper (UA override, SSL fallback, retry strategy) propagates
    here automatically.
    """
    from ._http import safe_get_bytes
    return safe_get_bytes(url, timeout=timeout)
```

The lazy import keeps existing import order safe.

- [ ] **Step 3: Remove now-unused imports from `puremacro/fetch.py`**

The `_safe_urlopen` function previously used `urllib.request`, `ssl`, and `io` directly. Verify those imports are still needed for the rest of the file (`fetch_fred` likely uses `pd.read_csv(io.BytesIO(...))`, so keep `io`). Remove only what's truly unused:

Run: `grep -n "ssl\." puremacro/fetch.py` — if no matches, remove `import ssl`.
Run: `grep -n "urllib" puremacro/fetch.py` — if only the (now-removed) `_safe_urlopen` used it, remove `import urllib.request`.

If either grep shows the import is still used elsewhere in the file, leave it alone.

- [ ] **Step 4: Smoke-test fetch_fred is still importable**

Run: `python -c "from puremacro.fetch import fetch_fred, _safe_urlopen; print('OK')"` → expect `OK`.

Run: `pytest tests/ -k fetch_fred -v 2>&1 | tail -5` → if there are tests for `fetch_fred`, they must still pass. If there are none (likely), this step is informational.

---

## Task 3: Migrate `bis_neer.py` to use `puremacro._http` directly

**Files:**
- Modify: `puremacro/bis_neer.py` (one import line + one call-site change)

`bis_neer.py` currently does `from .fetch import _safe_urlopen`. Replace with the unified path.

- [ ] **Step 1: Read `puremacro/bis_neer.py`** to find the import and call site

Run: `grep -n "_safe_urlopen\|safe_get_bytes" puremacro/bis_neer.py`

Expected: one or two references (the import + one or two call sites).

- [ ] **Step 2: Update the import**

Find: `from .fetch import _safe_urlopen`
Replace with: `from ._http import safe_get_bytes`

- [ ] **Step 3: Update the call site(s)**

Find every call to `_safe_urlopen(...)` and replace with `safe_get_bytes(...)`. The signatures match (both accept `(url, timeout=...)` returning bytes), so this is a pure rename.

- [ ] **Step 4: Smoke-test `bis_neer` still imports**

Run: `python -c "from puremacro.bis_neer import fetch_bis_neer; print('OK')"` → expect `OK`.

Run: `pytest tests/ -k bis_neer 2>&1 | tail -5` → if tests exist, must pass.

---

## Task 4: Add docstring note to `puremacro/data.py`

**Files:**
- Modify: `puremacro/data.py` (one paragraph in module docstring)

`data.py` is the panel/transforms module but its name confuses users (and LLMs) into expecting fetchers there. Add a one-line "see also" note pointing at the right place.

- [ ] **Step 1: Read the current `puremacro/data.py` module docstring**

Run: `head -15 puremacro/data.py` → note the existing docstring's first line and structure.

- [ ] **Step 2: Append a "See also" note**

Edit `puremacro/data.py`. Find the module-level docstring (`"""..."""` at the top). Insert a new paragraph at the END of the docstring (just before the closing `"""`):

```
See also
--------
For data **fetchers** (not transforms), see :mod:`puremacro.fetch`
which exposes ``fetch_fred``, ``fetch_fred_alfred``, and the new
``sdmx_get`` (0.6.0+) for OECD/Eurostat/ECB/IMF SDMX-CSV endpoints.
This module (``puremacro.data``) only contains panel-level
transforms (long↔wide, country slicing, HP / Hamilton / BK filters).
```

This is a documentation-only change. No tests needed.

- [ ] **Step 3: Confirm no behavior change**

Run: `python -c "from puremacro.data import hp_filter, hamilton_filter, bk_filter; print('OK')"` → expect `OK`.

---

## Task 5: Restructure `puremacro/fetch.py` → `puremacro/fetch/` subpackage

**Files:**
- Create: `puremacro/fetch/__init__.py`
- Create: `puremacro/fetch/_classic.py`
- Delete: `puremacro/fetch.py` (after copy)

Convert the single-file module into a subpackage so we can add `sdmx.py` (Task 6) cleanly. Public surface is preserved exactly — `from puremacro.fetch import fetch_fred` keeps working.

- [ ] **Step 1: Read the entire current `puremacro/fetch.py`**

Run: `cat puremacro/fetch.py`

Note: 3 public-ish functions (`_safe_urlopen`, `fetch_fred`, `fetch_fred_alfred`), 2 module-level constants (`_FREDGRAPH`, `_ALFRED`), several stdlib imports, an `__all__` if present.

- [ ] **Step 2: Create `puremacro/fetch/_classic.py`**

Create `puremacro/fetch/_classic.py` and paste the ENTIRE contents of `puremacro/fetch.py` into it verbatim. (After Task 2 the file already has the `_safe_urlopen` delegation.) This becomes the new home of the classic fetchers; the subpackage `__init__.py` will re-export them.

- [ ] **Step 3: Create `puremacro/fetch/__init__.py`**

Create `puremacro/fetch/__init__.py`:

```python
"""puremacro.fetch — public-data fetchers.

Exposes fetchers for major macro/financial data sources, all going
through the unified :mod:`puremacro._http` helpers (UA override,
one-shot SSL fallback, 30s default timeout). Each fetcher returns a
``pandas`` object suitable for direct use in VAR / LP estimation.

Public API
----------
- :func:`fetch_fred`        — FRED public CSV (no API key needed)
- :func:`fetch_fred_alfred` — ALFRED real-time vintages
- :func:`sdmx_get`          — generic SDMX-CSV (OECD, Eurostat, ECB, IMF SDMX Central)
- :func:`oecd_sdmx_instrument` — convenience wrapper that returns
                                 :class:`puremacro.instruments.Instrument` directly

For API-key-requiring FRED via the JSON endpoint, see
:func:`puremacro.instruments.external.load_fred`.
"""
from ._classic import (
    _safe_urlopen,
    fetch_fred,
    fetch_fred_alfred,
)
from .sdmx import (
    sdmx_get,
    oecd_sdmx_instrument,
)

__all__ = [
    "_safe_urlopen",
    "fetch_fred",
    "fetch_fred_alfred",
    "sdmx_get",
    "oecd_sdmx_instrument",
]
```

(The `from .sdmx import ...` line will fail until Task 6 creates that module. Don't worry about it yet — fix order: create `sdmx.py` next.)

- [ ] **Step 4: Delete the old `puremacro/fetch.py`**

Run: `rm puremacro/fetch.py`

- [ ] **Step 5: Smoke-test backwards compat**

Skip this step until Task 6 lands `sdmx.py`. Currently the import will fail on the missing `sdmx` module. The test will run end-of-Task-6.

---

## Task 6: Add `puremacro/fetch/sdmx.py` — generic SDMX-CSV fetcher + tests

**Files:**
- Create: `puremacro/fetch/sdmx.py`
- Create: `tests/test_fetch/__init__.py` (empty)
- Create: `tests/test_fetch/test_sdmx.py`

SDMX-CSV is a stable wire format used by OECD, Eurostat, ECB, IMF SDMX Central, World Bank, and others. Each provider has a different base URL but the response shape is consistent: columns include `TIME_PERIOD`, `OBS_VALUE`, plus dimension columns (`REF_AREA`, indicator code columns, etc.).

The generic `sdmx_get(provider, dataflow, key, ...)` returns the raw DataFrame; users filter as needed. The convenience `oecd_sdmx_instrument(dataset, country, indicator, ...)` wraps the most common use case (one country × one indicator → time series → `Instrument`).

- [ ] **Step 1: Write failing tests**

Create `tests/test_fetch/__init__.py` as an empty file.

Create `tests/test_fetch/test_sdmx.py`:

```python
"""Tests for puremacro.fetch.sdmx."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from puremacro.fetch import sdmx_get, oecd_sdmx_instrument
from puremacro.instruments import Instrument


# Synthetic SDMX-CSV (subset of canonical OECD shape).
_SYNTHETIC_SDMX_CSV = """DATAFLOW,REF_AREA,MEASURE,UNIT_MEASURE,TIME_PERIOD,OBS_VALUE
OECD:DSD_STAN(1.0),USA,VALADD,USD_M,2018,21000.0
OECD:DSD_STAN(1.0),USA,VALADD,USD_M,2019,22000.0
OECD:DSD_STAN(1.0),USA,VALADD,USD_M,2020,21500.0
OECD:DSD_STAN(1.0),GBR,VALADD,USD_M,2018,3000.0
OECD:DSD_STAN(1.0),GBR,VALADD,USD_M,2019,3100.0
"""


def test_sdmx_get_with_csv_path_returns_dataframe(tmp_path):
    csv = tmp_path / "sdmx.csv"
    csv.write_text(_SYNTHETIC_SDMX_CSV)
    df = sdmx_get(provider="oecd", dataflow="DSD_STAN", key="USA",
                  csv_path=csv)
    assert isinstance(df, pd.DataFrame)
    assert "TIME_PERIOD" in df.columns
    assert "OBS_VALUE" in df.columns
    assert len(df) == 5


def test_sdmx_get_unknown_provider_raises():
    with pytest.raises(ValueError, match="provider"):
        sdmx_get(provider="not_a_real_provider", dataflow="X", key="Y",
                 csv_path=None)


def test_sdmx_get_known_providers():
    """Provider whitelist must include the four planned sources."""
    from puremacro.fetch.sdmx import _PROVIDERS
    assert "oecd" in _PROVIDERS
    assert "eurostat" in _PROVIDERS
    assert "ecb" in _PROVIDERS
    assert "imf" in _PROVIDERS


def test_sdmx_get_no_csv_no_network_raises_clear_error(monkeypatch):
    from puremacro.fetch import sdmx as _mod
    def _fail(_url, **kw):
        raise OSError("simulated network failure")
    monkeypatch.setattr(_mod, "safe_get_bytes", _fail)
    with pytest.raises(RuntimeError, match="SDMX"):
        sdmx_get(provider="oecd", dataflow="DSD_STAN", key="USA")


def test_oecd_sdmx_instrument_with_csv_path_returns_instrument(tmp_path):
    csv = tmp_path / "sdmx.csv"
    csv.write_text(_SYNTHETIC_SDMX_CSV)
    inst = oecd_sdmx_instrument(
        dataset="DSD_STAN", country="USA", indicator="VALADD",
        csv_path=csv,
    )
    assert isinstance(inst, Instrument)
    assert inst.category == "external_csv"
    assert inst.frequency == "A"
    assert inst.name == "oecd_DSD_STAN_USA_VALADD"
    assert inst.series.loc[pd.Timestamp("2018-01-01")] == pytest.approx(21000.0)
    assert len(inst.series) == 3  # USA-only, 3 years


def test_oecd_sdmx_instrument_filters_country(tmp_path):
    csv = tmp_path / "sdmx.csv"
    csv.write_text(_SYNTHETIC_SDMX_CSV)
    inst = oecd_sdmx_instrument(
        dataset="DSD_STAN", country="GBR", indicator="VALADD",
        csv_path=csv,
    )
    assert len(inst.series) == 2
    assert inst.series.loc[pd.Timestamp("2018-01-01")] == pytest.approx(3000.0)


def test_oecd_sdmx_instrument_unknown_country_raises(tmp_path):
    csv = tmp_path / "sdmx.csv"
    csv.write_text(_SYNTHETIC_SDMX_CSV)
    with pytest.raises(ValueError, match="ZZZ"):
        oecd_sdmx_instrument(
            dataset="DSD_STAN", country="ZZZ", indicator="VALADD",
            csv_path=csv,
        )


def test_oecd_sdmx_instrument_metadata_includes_provider_and_dataset(tmp_path):
    csv = tmp_path / "sdmx.csv"
    csv.write_text(_SYNTHETIC_SDMX_CSV)
    inst = oecd_sdmx_instrument(
        dataset="DSD_STAN", country="USA", indicator="VALADD",
        csv_path=csv,
    )
    assert inst.metadata.get("provider") == "oecd"
    assert inst.metadata.get("dataset") == "DSD_STAN"
    assert inst.metadata.get("country") == "USA"
    assert inst.metadata.get("indicator") == "VALADD"
    assert "reference" in inst.metadata
```

Run: `pytest tests/test_fetch/test_sdmx.py -v` → expect ImportError on `sdmx_get` / `oecd_sdmx_instrument` (sdmx.py doesn't exist).

- [ ] **Step 2: Create `puremacro/fetch/sdmx.py`**

Create `puremacro/fetch/sdmx.py`:

```python
"""Generic SDMX-CSV fetcher for OECD, Eurostat, ECB, and IMF SDMX Central.

SDMX-CSV (Statistical Data and Metadata eXchange — CSV variant) is a
W3C-stewarded wire format that all major statistical agencies expose.
Each provider has its own base URL but the response shape is
consistent: dimension columns + ``TIME_PERIOD`` + ``OBS_VALUE`` +
optional attribute columns.

Generic ``sdmx_get(provider, dataflow, key, ...)`` returns the raw
DataFrame. ``oecd_sdmx_instrument(...)`` wraps the most common case
(one country × one indicator → :class:`puremacro.instruments.Instrument`).

References
----------
SDMX-CSV format: https://sdmx.org/?page_id=4345
OECD SDMX API:  https://sdmx.oecd.org/public/rest/
Eurostat SDMX:   https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0
ECB SDW:         https://data-api.ecb.europa.eu/service
IMF SDMX:        https://sdmxcentral.imf.org/ws/public/sdmxapi/rest
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from .._http import safe_get_bytes
from ..instruments._core import Instrument


# Provider URL templates — fill with {dataflow} and {key} placeholders.
# format=csvfile / csvdata triggers SDMX-CSV per provider's convention.
_PROVIDERS: dict[str, str] = {
    "oecd": "https://sdmx.oecd.org/public/rest/data/{dataflow}/{key}?format=csvfilewithlabels",
    "eurostat": "https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/{dataflow}/{key}?format=SDMX-CSV",
    "ecb": "https://data-api.ecb.europa.eu/service/data/{dataflow}/{key}?format=csvdata",
    "imf": "https://sdmxcentral.imf.org/ws/public/sdmxapi/rest/data/{dataflow}/{key}?format=sdmx-csv",
}

_REFERENCE_TEMPLATE = (
    "SDMX-CSV from provider {provider!r}, dataflow {dataflow!r}. "
    "See https://sdmx.org/?page_id=4345 for format spec."
)


def sdmx_get(
    *,
    provider: str,
    dataflow: str,
    key: str = "all",
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch an SDMX-CSV response and return as a DataFrame.

    Parameters
    ----------
    provider : str
        One of ``"oecd"``, ``"eurostat"``, ``"ecb"``, ``"imf"``. Each maps
        to a known base URL template (see :data:`_PROVIDERS`).
    dataflow : str
        Provider-specific dataflow ID (e.g. ``"DSD_STAN"`` for OECD-STAN).
    key : str, default ``"all"``
        SDMX dot-separated dimension key. ``"all"`` returns everything;
        e.g. ``"USA"`` filters by REF_AREA.
    csv_path : str | Path | None
        Optional local path to a pre-downloaded SDMX-CSV file. When
        None, attempt the live network fetch.

    Returns
    -------
    pd.DataFrame
        Raw SDMX-CSV columns: dimension cols + ``TIME_PERIOD`` +
        ``OBS_VALUE`` + attributes. Filtering is the caller's job
        (use :func:`oecd_sdmx_instrument` for the common case).

    Raises
    ------
    ValueError
        If ``provider`` is not in the known providers whitelist.
    RuntimeError
        If the network fetch fails (and no ``csv_path=`` was provided).
    """
    if provider not in _PROVIDERS:
        raise ValueError(
            f"provider {provider!r} not in known providers: "
            f"{sorted(_PROVIDERS.keys())}"
        )
    if csv_path is not None:
        return pd.read_csv(csv_path)
    url = _PROVIDERS[provider].format(dataflow=dataflow, key=key)
    try:
        raw = safe_get_bytes(url)
    except Exception as e:
        raise RuntimeError(
            f"Could not fetch SDMX from {provider!r} (dataflow={dataflow!r}, "
            f"key={key!r}). Verify the dataflow ID at the provider's portal "
            f"and pass csv_path= with a local download to skip the network."
        ) from None
    return pd.read_csv(io.BytesIO(raw))


def oecd_sdmx_instrument(
    *,
    dataset: str,
    country: str,
    indicator: str,
    csv_path: str | Path | None = None,
    frequency: str = "A",
    measure_col: str = "MEASURE",
) -> Instrument:
    """Convenience: fetch one (country × indicator) slice from OECD-SDMX
    and return as an :class:`Instrument`.

    Parameters
    ----------
    dataset : str
        OECD dataflow ID (e.g. ``"DSD_STAN"`` for STAN industrial data,
        ``"DF_FUNCTIONAL"`` for fiscal indicators).
    country : str
        ISO3 country code, matching the ``REF_AREA`` column.
    indicator : str
        Code matching the ``measure_col`` column (default ``MEASURE``).
        For STAN this is e.g. ``"VALADD"`` (value added),
        ``"EMPN"`` (employment), etc.
    csv_path : str | Path | None
        Optional local SDMX-CSV.
    frequency : str, default ``"A"``
        OECD STAN is annual; pass ``"Q"`` for quarterly datasets.
    measure_col : str, default ``"MEASURE"``
        Column name carrying the indicator code. SDMX naming varies
        slightly across OECD dataflows.

    Returns
    -------
    Instrument
        Time series indexed by ``TIME_PERIOD``, name
        ``f"oecd_{dataset}_{country}_{indicator}"``,
        category ``"external_csv"``.
    """
    df = sdmx_get(provider="oecd", dataflow=dataset, key=country,
                  csv_path=csv_path)
    expected = {"REF_AREA", "TIME_PERIOD", "OBS_VALUE", measure_col}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"OECD SDMX response missing expected columns: {sorted(missing)}. "
            f"Got: {sorted(df.columns)}. Pass measure_col= if the indicator "
            f"column has a non-default name."
        )
    available_countries = sorted(df["REF_AREA"].dropna().unique().tolist())
    if country not in available_countries:
        raise ValueError(
            f"country {country!r} not in OECD response; available: "
            f"{available_countries}"
        )
    sub = df[(df["REF_AREA"] == country) & (df[measure_col] == indicator)].copy()
    if sub.empty:
        raise ValueError(
            f"no rows in OECD response for country={country!r}, "
            f"indicator={indicator!r}. Available indicators for {country}: "
            f"{sorted(df[df['REF_AREA'] == country][measure_col].dropna().unique().tolist())}"
        )
    sub = sub.dropna(subset=["TIME_PERIOD", "OBS_VALUE"])
    dates = pd.to_datetime(sub["TIME_PERIOD"].astype(str) + "-01-01")
    name = f"oecd_{dataset}_{country}_{indicator}"
    series = pd.Series(
        sub["OBS_VALUE"].astype(float).values,
        index=dates,
        name=name,
    ).sort_index()
    return Instrument(
        series=series,
        name=name,
        source=f"OECD SDMX {dataset} ({country}, {indicator})",
        category="external_csv",
        frequency=frequency,
        metadata={
            "reference": _REFERENCE_TEMPLATE.format(
                provider="oecd", dataflow=dataset,
            ),
            "provider": "oecd",
            "dataset": dataset,
            "country": country,
            "indicator": indicator,
        },
    )


__all__ = ["sdmx_get", "oecd_sdmx_instrument"]
```

- [ ] **Step 3: Confirm tests pass + backwards compat**

Run: `pytest tests/test_fetch/ -v` → expect 8 passed.

Run: `python -c "from puremacro.fetch import fetch_fred, sdmx_get, oecd_sdmx_instrument, _safe_urlopen; print('OK')"` → expect `OK`.

Run: `python -c "from puremacro.bis_neer import fetch_bis_neer; print('OK')"` → expect `OK` (Task 3 migration must continue to work after fetch.py was deleted).

---

## Task 7: Catalog integration — 4 new entries

**Files:**
- Modify: `puremacro/instruments/_catalog.py` (append 4 entries)
- Modify: `tests/test_instruments/test_catalog.py` (size 36 → 40)

The 4 new entries surface the unified fetchers in `pi.list_available()`. Naming convention: prefix with `puremacro_fetch_` to distinguish from the existing API-key-requiring `fred_*` entries.

- [ ] **Step 1: Update test_catalog.py size assertions**

Edit `tests/test_instruments/test_catalog.py`. Find `test_total_catalog_size_is_exactly_36` and rename to `_40`:

```python
def test_total_catalog_size_is_exactly_40():
    """6 replications + 6 connectors + 1 monetary HFI + 12 stubs + 4 literature + 7 external + 4 fetch = 40."""
    assert len(_registry._REGISTRY) == 40
```

Same rename for `_at_least_36` → `_at_least_40`.

Append:

```python
_EXPECTED_FETCH_KEYS = {
    "fetch_fred_csv",
    "fetch_bis_neer_us",
    "oecd_sdmx_stan_usa_valadd",
    "oecd_sdmx_stan_usa_empn",
}


def test_all_four_fetch_entries_registered():
    keys = set(_registry._REGISTRY.keys())
    missing = _EXPECTED_FETCH_KEYS - keys
    assert not missing, f"fetch entries missing: {missing}"


def test_every_fetch_entry_is_external_csv_category():
    for key in _EXPECTED_FETCH_KEYS:
        spec = _registry._REGISTRY[key]
        assert spec.category == "external_csv"


def test_every_fetch_entry_requires_network():
    for key in _EXPECTED_FETCH_KEYS:
        spec = _registry._REGISTRY[key]
        assert spec.requires_network is True


def test_every_fetch_entry_has_non_empty_reference():
    for key in _EXPECTED_FETCH_KEYS:
        spec = _registry._REGISTRY[key]
        assert spec.reference and len(spec.reference) > 10
```

- [ ] **Step 2: Confirm failure**

Run: `pytest tests/test_instruments/test_catalog.py -v -k "fetch or 40"` → expect failures.

- [ ] **Step 3: Append catalog entries**

Edit `puremacro/instruments/_catalog.py`. Append at the end:

```python
# --------------------------------------------------------------------------
# Fetch-layer providers (4) — see puremacro/fetch/
# --------------------------------------------------------------------------
from ..fetch import fetch_fred, oecd_sdmx_instrument
from ..bis_neer import fetch_bis_neer


def _fred_csv_loader(*, series_id: str = "GDPC1", **kwargs) -> Instrument:
    """Wrap fetch_fred (public CSV, no API key) as an Instrument."""
    df = fetch_fred(series_id, **kwargs)
    # fetch_fred returns DataFrame indexed by date; assume the value
    # column matches the series_id.
    if series_id in df.columns:
        series = df[series_id].astype(float)
    else:
        # Fallback: take the first non-DATE column
        cols = [c for c in df.columns if c.upper() != "DATE"]
        series = df[cols[0]].astype(float)
    series.name = f"fetch_fred_{series_id}"
    return Instrument(
        series=series,
        name=f"fetch_fred_{series_id}",
        source=f"FRED public CSV (fetch_fred): {series_id}",
        category="external_csv",
        frequency="Q",
        metadata={
            "reference": (
                "Federal Reserve Bank of St. Louis. FRED public CSV. "
                f"https://fred.stlouisfed.org/series/{series_id}"
            ),
            "series_id": series_id,
        },
    )


register(InstrumentSpec(
    key="fetch_fred_csv",
    name="FRED public CSV (no API key required)",
    category="external_csv",
    description=(
        "Fetch any FRED series via the public CSV endpoint (no API key "
        "required). Pass series_id= at load time. Default series_id is "
        "'GDPC1' (real GDP). Frequency varies by series; default 'Q'."
    ),
    reference="Federal Reserve Bank of St. Louis. FRED public CSV endpoint. https://fred.stlouisfed.org/",
    loader=_fred_csv_loader,
    country="USA",
    frequency="Q",
    requires_network=True,
    requires_fixture=False,
))


def _bis_neer_us_loader(**kwargs) -> Instrument:
    """Wrap fetch_bis_neer for US as an Instrument."""
    df = fetch_bis_neer(country="US", **kwargs)
    # Expect df with date index + value column
    if "value" in df.columns:
        series = df["value"].astype(float)
    else:
        # Take the first numeric column
        numeric_cols = [c for c in df.columns
                        if pd.api.types.is_numeric_dtype(df[c])]
        series = df[numeric_cols[0]].astype(float) if numeric_cols else df.iloc[:, 0].astype(float)
    series.name = "bis_neer_us"
    return Instrument(
        series=series,
        name="bis_neer_us",
        source="BIS US Nominal Effective Exchange Rate (fetch_bis_neer)",
        category="external_csv",
        frequency="M",
        metadata={
            "reference": "Bank for International Settlements. Effective Exchange Rate indices. https://www.bis.org/statistics/eer.htm",
            "country": "US",
        },
    )


register(InstrumentSpec(
    key="fetch_bis_neer_us",
    name="BIS US Nominal Effective Exchange Rate (monthly)",
    category="external_csv",
    description=(
        "Monthly US nominal effective exchange rate from BIS, fetched "
        "via puremacro.bis_neer.fetch_bis_neer. Pass csv_path= to skip "
        "the network call."
    ),
    reference="Bank for International Settlements. Effective Exchange Rate indices. https://www.bis.org/statistics/eer.htm",
    loader=_bis_neer_us_loader,
    country="USA",
    frequency="M",
    requires_network=True,
    requires_fixture=False,
))


def _make_oecd_stan_loader(country: str, indicator: str):
    def _load(**kwargs) -> Instrument:
        return oecd_sdmx_instrument(
            dataset="DSD_STAN", country=country, indicator=indicator,
            **kwargs,
        )
    return _load


register(InstrumentSpec(
    key="oecd_sdmx_stan_usa_valadd",
    name="OECD STAN US Value Added (annual)",
    category="external_csv",
    description=(
        "Annual US value added from OECD-STAN industrial database, "
        "fetched via puremacro.fetch.oecd_sdmx_instrument with "
        "dataset='DSD_STAN', country='USA', indicator='VALADD'. "
        "Pass csv_path= to skip the network call."
    ),
    reference="OECD STAN Industrial Statistics Database. https://www.oecd.org/sti/stan",
    loader=_make_oecd_stan_loader("USA", "VALADD"),
    country="USA",
    frequency="A",
    requires_network=True,
    requires_fixture=False,
))


register(InstrumentSpec(
    key="oecd_sdmx_stan_usa_empn",
    name="OECD STAN US Employment (annual)",
    category="external_csv",
    description=(
        "Annual US total employment from OECD-STAN, fetched via "
        "puremacro.fetch.oecd_sdmx_instrument with dataset='DSD_STAN', "
        "country='USA', indicator='EMPN'. Pass csv_path= to skip the "
        "network call."
    ),
    reference="OECD STAN Industrial Statistics Database. https://www.oecd.org/sti/stan",
    loader=_make_oecd_stan_loader("USA", "EMPN"),
    country="USA",
    frequency="A",
    requires_network=True,
    requires_fixture=False,
))
```

(Note: `import pandas as pd` is needed for the `_bis_neer_us_loader` numeric check. Add it at the top of `_catalog.py` if it's not already imported.)

- [ ] **Step 4: Confirm catalog tests pass**

Run: `pytest tests/test_instruments/test_catalog.py -v` → expect all catalog tests pass (size 40, 4 new fetch keys present, etc.).

Run: `python -W error -c "import puremacro.instruments; assert len(puremacro.instruments._registry._REGISTRY) == 40"` → expect exit 0, no warnings.

---

## Task 8: Snapshot regen + version bump 0.5.4 → 0.6.0 + CHANGELOG + memory

**Files:**
- Modify: `tests/fixtures/public_api_snapshot.json` (regenerate)
- Modify: `pyproject.toml` (`0.5.4 → 0.6.0`)
- Modify: `puremacro/__init__.py` (`__version__`)
- Modify: `tests/test_import.py`
- Modify: `CHANGELOG.md`
- Modify: `~/.claude/projects/.../memory/project_puremacro.md`

- [ ] **Step 1: Regenerate snapshot**

Run from the puremacro directory:

```bash
python -c "
import sys; sys.path.insert(0, 'tests')
from test_public_api import _collect_current_api
import json
print(json.dumps(_collect_current_api(), indent=2))
" > tests/fixtures/public_api_snapshot.json
```

Then `pytest tests/test_public_api.py -v` → expect PASS.

- [ ] **Step 2: Spot-check the snapshot diff**

Run: `grep -c "puremacro._http\|puremacro.fetch" tests/fixtures/public_api_snapshot.json`
Expected: at least 4 (one for each new module: `_http`, `fetch`, `fetch._classic`, `fetch.sdmx`).

- [ ] **Step 3: Bump pyproject.toml**

Edit `pyproject.toml`. Change `version = "0.5.4"` to `version = "0.6.0"`.

- [ ] **Step 4: Bump `__version__`**

Edit `puremacro/__init__.py`. Change `__version__ = "0.5.4"` to `__version__ = "0.6.0"`.

- [ ] **Step 5: Bump test_import.py expected version**

Edit `tests/test_import.py`. Change assertion to `"0.6.0"`.

- [ ] **Step 6: Confirm import test green**

Run: `pytest tests/test_import.py -v` → expect PASS.

- [ ] **Step 7: Add CHANGELOG entry**

Edit `CHANGELOG.md`. Insert immediately after the file header and before `## 0.5.4 — 2026-05-03`:

```markdown
## 0.6.0 — 2026-05-03

Minor release — three structural improvements driven by a real friction point: a plan referenced `puremacro.data.oecd_sdmx_get` (which doesn't exist), revealing (a) two parallel HTTP-fetch infrastructures with divergent hardening and (b) `puremacro.data` being misleadingly named. This release unifies the HTTP path, restructures `puremacro.fetch` as a subpackage with a generic SDMX-CSV fetcher, and wires the now-unified fetchers into the `Instrument` registry. No breaking changes: legacy import paths preserved via shims.

### Added
- **`puremacro._http`** (new top-level module) — canonical home of `safe_get_bytes` / `safe_get_text` / `safe_get_json` plus `USER_AGENT` / `DEFAULT_TIMEOUT`. Promoted from `puremacro.narrative.sources._http` so all fetchers share the same hardened path (UA override, one-shot SSL fallback, 30s default timeout). The 0.4.1 security fixes now apply uniformly.
- **`puremacro.fetch`** (new subpackage, replaces single-file `puremacro/fetch.py`) — exposes `fetch_fred`, `fetch_fred_alfred` (preserved from 0.5.x), plus new:
  - `sdmx_get(provider, dataflow, key, csv_path=None)` — generic SDMX-CSV fetcher for OECD, Eurostat, ECB, IMF SDMX Central. Returns the raw DataFrame.
  - `oecd_sdmx_instrument(dataset, country, indicator, ...)` — convenience wrapper returning an `Instrument` directly. First-class catalog support.
- **4 new catalog entries** (`pi.list_available()`-discoverable):
  - `fetch_fred_csv` — public FRED CSV, no API key needed (complements the API-key-requiring `fred_*` entries).
  - `fetch_bis_neer_us` — US nominal effective exchange rate from BIS via `fetch_bis_neer`.
  - `oecd_sdmx_stan_usa_valadd` — OECD-STAN US Value Added (annual).
  - `oecd_sdmx_stan_usa_empn` — OECD-STAN US Employment (annual).
- `puremacro.data` docstring extended with a "See also" pointing at `puremacro.fetch` for fetchers (the module's name had been misleading users into expecting fetchers there).

### Changed (backwards-compat preserved)
- `puremacro.narrative.sources._http` is now a re-export shim (re-exports from `puremacro._http`). All existing imports keep working.
- `puremacro.fetch._safe_urlopen` is now a thin wrapper that delegates to `puremacro._http.safe_get_bytes`. Same signature, same return type, hardened underneath.
- `puremacro.bis_neer` updated to import `safe_get_bytes` from `puremacro._http` directly. Public API unchanged.

### Internal
- `puremacro/fetch.py` removed; replaced by `puremacro/fetch/__init__.py` + `puremacro/fetch/_classic.py` (extracted) + `puremacro/fetch/sdmx.py` (new).
- `tests/test_http_unified.py` (new) — confirms the new top-level path works AND the legacy narrative path still re-exports the same objects.
- `tests/test_fetch/` (new directory) — 8 tests for `sdmx_get` + `oecd_sdmx_instrument` with synthetic SDMX-CSV.
- `tests/test_instruments/test_catalog.py` size assertions tightened 36 → 40; new tests for the 4 fetch entries.
- `tests/fixtures/public_api_snapshot.json` regenerated to record `puremacro._http`, the new `puremacro.fetch` subpackage, and `puremacro.fetch.sdmx`.

### Out of scope (future)
- Per-record country threading in `score_keyword` (still deferred — touches connector wire format + 3 catalog entries; deserves its own focused session).
- More OECD-STAN catalog entries (the 2 shipped here are showcases; long tail can be added incrementally).
- Eurostat / ECB / IMF SDMX catalog entries (the generic `sdmx_get` makes these one-line additions).
- JSON serializability of `Instrument.metadata`.

### Tests
- Pre-release baseline: 536 passing, 9 skipped (0.5.4).
- Post-release: ~565 passing, 9 skipped (+~29 new tests).
```

- [ ] **Step 8: Append memory entry**

Edit the puremacro memory file at `/Users/jalonso/.claude/projects/-Users-jalonso-Library-CloudStorage-GoogleDrive-jorge-alonsoortiz-gmail-com-My-Drive-MAV/memory/project_puremacro.md`. Append at the end:

```markdown

**Iteration N+9 step 7 done (2026-05-03) — released as 0.6.0 (minor — fetch unification + OECD SDMX):**
- Driven by user friction: a plan referenced `puremacro.data.oecd_sdmx_get` which doesn't exist, exposing (a) two parallel HTTP fetchers (`fetch._safe_urlopen` vs `narrative.sources._http.safe_get_*`) with divergent hardening — the 0.4.1 UA-override / SSL-fallback only applied to one, and (b) `puremacro.data` being misnamed (it's transforms, not fetchers).
- Promoted `safe_get_bytes` / `safe_get_text` / `safe_get_json` to top-level `puremacro/_http.py`. `narrative/sources/_http.py` is now a re-export shim. `fetch._safe_urlopen` delegates to the new shared path. `bis_neer.py` migrated to import `safe_get_bytes` directly. All fetchers now share one hardened path; security/policy fixes propagate uniformly.
- Restructured `puremacro/fetch.py` (single file) → `puremacro/fetch/` (subpackage). Public surface preserved (`fetch_fred`, `fetch_fred_alfred`, `_safe_urlopen` re-exported from `__init__.py`). New module `puremacro/fetch/sdmx.py` adds: `sdmx_get(provider, dataflow, key, csv_path=)` (generic, supports oecd/eurostat/ecb/imf provider whitelist) + `oecd_sdmx_instrument(dataset, country, indicator, ...)` (convenience wrapper returning Instrument directly).
- Catalog grew 36 → 40: +`fetch_fred_csv` (public CSV, no API key — complements the existing API-key-requiring `fred_*` entries), +`fetch_bis_neer_us`, +`oecd_sdmx_stan_usa_valadd`, +`oecd_sdmx_stan_usa_empn`. The OECD entries use `oecd_sdmx_instrument` with dataset='DSD_STAN', exercising the new SDMX path end-to-end.
- `puremacro.data` docstring extended with a "See also" pointing at `puremacro.fetch` (small fix, big UX value — `data.py` is transforms, not fetchers).
- 8 new SDMX tests in `tests/test_fetch/test_sdmx.py` + 4 new catalog tests + 4 HTTP-unification tests = 16 net new tests. Test count: 536 → ~565 passing.
- Plan file: `uncertainty_examples/puremacro/docs/plans/2026-05-03-fetch-unification-sdmx-060.md`.

**0.6.0 still deferred (next 0.6.1+):**
- Per-record country threading in `score_keyword` (cross-country narrative connectors stamping correct per-event countries instead of requiring `country=` at load time). Touches connector wire format (3-tuple → 4-tuple), `score_keyword` signature, `_wrap_connector` adapter, and 3 catalog entries.
- More curated OECD-STAN / Eurostat / ECB / IMF SDMX catalog entries (the generic `sdmx_get` makes each a one-line addition).
- JSON serializability of `Instrument.metadata`.

**How to apply (0.6.0):** When the user wants OECD/Eurostat/ECB/IMF data: `from puremacro.fetch import sdmx_get, oecd_sdmx_instrument`. Generic raw fetch: `sdmx_get(provider="oecd", dataflow="DSD_STAN", key="USA")`. Instrument-shaped fetch: `oecd_sdmx_instrument(dataset="DSD_STAN", country="USA", indicator="VALADD")`. For discovery: `pi.list_available(category="external_csv", include_unavailable=True)` shows all 11 external entries (4 FRED, 1 BIS, 2 IMF WEO, 4 fetch-layer). Note: ALL fetchers now share `puremacro._http` — any future security or policy fix to that helper propagates everywhere.
```

- [ ] **Step 9: Final test run**

Run: `pytest -x -q 2>&1 | tail -5`
Expected: ~565 passed, 9 skipped.

- [ ] **Step 10: Pyodide-compat regression check**

Run: `pytest tests/test_pyodide_compat.py -v` → expect PASS.

- [ ] **Step 11: Sanity-check the full new surface**

Run:

```bash
python -c "
from puremacro._http import safe_get_bytes, safe_get_text, safe_get_json
from puremacro.fetch import fetch_fred, sdmx_get, oecd_sdmx_instrument, _safe_urlopen
from puremacro.bis_neer import fetch_bis_neer
from puremacro.instruments import list_available, load
from puremacro.narrative.sources._http import safe_get_bytes as legacy
import puremacro
print('Version:', puremacro.__version__)
print('catalog rows (incl. unavailable):', len(list_available(include_unavailable=True)))
print('legacy shim is canonical:', legacy is safe_get_bytes)
"
```

Expected:
```
Version: 0.6.0
catalog rows (incl. unavailable): 40
legacy shim is canonical: True
```

---

## Self-Review Checklist

After implementation:

1. **Spec coverage:** All 4 + bonus addressed?
   - [x] HTTP unification → Tasks 1–3
   - [x] `data.py` rename note → Task 4
   - [x] OECD SDMX fetcher → Tasks 5–6
   - [x] Catalog integration → Task 7
   - [x] Release coordination → Task 8

2. **Placeholder scan:** No "TBD", "implement later", "appropriate error handling" patterns. Every code step shows the actual code.

3. **Type consistency:**
   - `puremacro._http.safe_get_bytes` signature matches `puremacro.narrative.sources._http.safe_get_bytes` (the move is verbatim).
   - `puremacro.fetch._safe_urlopen` signature `(url, timeout=30.0) -> bytes` preserved.
   - `sdmx_get(*, provider, dataflow, key="all", csv_path=None) -> pd.DataFrame` keyword-only.
   - `oecd_sdmx_instrument(*, dataset, country, indicator, csv_path=None, frequency="A", measure_col="MEASURE") -> Instrument` keyword-only.
   - All 4 catalog entry keys (`fetch_fred_csv`, `fetch_bis_neer_us`, `oecd_sdmx_stan_usa_valadd`, `oecd_sdmx_stan_usa_empn`) match between catalog registration, test `_EXPECTED_FETCH_KEYS`, and the per-spec `name` fields.

4. **Pyodide hygiene:** No new runtime deps. SDMX fetcher uses existing `safe_get_bytes` (Pyodide-correct).

5. **Backwards compatibility:**
   - `from puremacro.narrative.sources._http import safe_get_bytes` continues to work via the shim.
   - `from puremacro.fetch import _safe_urlopen, fetch_fred, fetch_fred_alfred` continues to work via re-export.
   - `from puremacro.bis_neer import fetch_bis_neer` unchanged externally; only its internal import line moved.
   - All existing 36 catalog entries unchanged.
   - All existing tests pass without modification (the snapshot regenerates additively).

6. **No connector behavior changes** — narrative connectors continue to use the (now-shimmed) `narrative.sources._http` path; their behavior is identical.
