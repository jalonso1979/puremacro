# puremacro 0.5.1 — Literature Shock Loaders Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **This workspace is NOT a git repo (Google Drive sync); skip every "commit" step the meta-skill suggests.**

**Goal:** Ship 4 new literature shock loaders under `puremacro/instruments/literature/` and wire them into the discovery registry, expanding the catalog from 25 → 29 entries with 1 fully-offline `available=True` entry (Bloom 2009 events) and 3 network-fetched ones. Released as **0.5.1** (patch — additive, no breaking changes).

**Architecture:** New `puremacro/instruments/literature/` subpackage. Each loader is a stand-alone module exposing `load(*, csv_path=None, ...) -> Instrument`. Loaders that need a CSV use `narrative.sources._http.safe_get_bytes` for download with raise-RuntimeError fallback (mirrors the existing replication-loader pattern). A shared `_csv_to_instrument()` helper handles the common DataFrame → Instrument conversion. Bloom 2009 is the exception: 17 hand-coded event dates are baked into Python and emitted as a monthly indicator series — no network, no fixture, fully `available=True`.

**Tech Stack:** Python 3.10+, `pandas`, `numpy`, `puremacro.instruments` (0.5.0), `narrative.sources._http`. Pyodide-compatible (no new runtime deps).

**Pre-implementation baseline:** 444 passing, 9 skipped (puremacro 0.5.0).
**Post-implementation target:** ~470 passing (+~26 new tests), 9 skipped.

---

## File Structure

### Files created
- `puremacro/instruments/literature/__init__.py` — re-exports the 4 loaders
- `puremacro/instruments/literature/_helpers.py` — shared `_csv_to_instrument()` adapter
- `puremacro/instruments/literature/bloom_2009.py` — `load() -> Instrument` (offline, baked-in event dates)
- `puremacro/instruments/literature/bbd_epu.py` — `load(*, csv_path=None) -> Instrument`
- `puremacro/instruments/literature/caldara_iacoviello_gpr.py` — `load(*, csv_path=None) -> Instrument`
- `puremacro/instruments/literature/romer_romer_2004.py` — `load(*, csv_path=None) -> Instrument`
- `tests/test_instruments/literature/__init__.py` — empty pytest package marker
- `tests/test_instruments/literature/test_bloom_2009.py`
- `tests/test_instruments/literature/test_bbd_epu.py`
- `tests/test_instruments/literature/test_caldara_iacoviello_gpr.py`
- `tests/test_instruments/literature/test_romer_romer_2004.py`

### Files modified
- `puremacro/instruments/_catalog.py` — add 4 new `register(InstrumentSpec(...))` entries pointing at the new loaders
- `tests/test_instruments/test_catalog.py` — extend `test_total_phase1_catalog_size_is_exactly_25` to expect 29; add 1 new test asserting Bloom 2009 is `available=True`
- `tests/fixtures/public_api_snapshot.json` — regenerate (new subpackage `puremacro.instruments.literature` and submodules)
- `pyproject.toml` — `version = "0.5.0" → "0.5.1"`
- `puremacro/__init__.py` — `__version__ = "0.5.1"`
- `tests/test_import.py` — bump expected version
- `CHANGELOG.md` — add `## 0.5.1 — 2026-05-03` block at top
- `~/.claude/projects/.../memory/project_puremacro.md` — append iteration entry

---

## Task 1: Bloom 2009 uncertainty events (fully-offline loader)

**Files:**
- Create: `puremacro/instruments/literature/__init__.py` (initial stub)
- Create: `puremacro/instruments/literature/bloom_2009.py`
- Create: `tests/test_instruments/literature/__init__.py` (empty pytest marker)
- Create: `tests/test_instruments/literature/test_bloom_2009.py`

Bloom (2009, Econometrica) identifies 17 large uncertainty episodes — discrete announcement-month dates. Bake these dates directly into Python; the loader emits a monthly indicator series (1 at each event month, 0 elsewhere) over the date range Jan-1962 to Dec-2008 (the paper's sample plus Bloom's standard extension to the GFC). Fully offline, fully reproducible, `requires_network=False`, `requires_fixture=False`.

- [ ] **Step 1: Create the literature package marker**

Create `puremacro/instruments/literature/__init__.py`:

```python
"""Literature shock instruments — canonical identified-shock series from the
empirical macro literature, wrapped as :class:`puremacro.instruments.Instrument`.

Each loader exposes a top-level ``load(...) -> Instrument`` function and is
registered in :mod:`puremacro.instruments._catalog` so callers can also
reach it via :func:`puremacro.instruments.load`.
"""
from .bloom_2009 import load as load_bloom_2009

__all__ = ["load_bloom_2009"]
```

- [ ] **Step 2: Create the test package marker**

Create `tests/test_instruments/literature/__init__.py` as an empty file.

- [ ] **Step 3: Write failing tests**

Create `tests/test_instruments/literature/test_bloom_2009.py`:

```python
"""Tests for puremacro.instruments.literature.bloom_2009."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.instruments import Instrument
from puremacro.instruments.literature.bloom_2009 import load, BLOOM_2009_EVENTS


def test_load_returns_instrument():
    inst = load()
    assert isinstance(inst, Instrument)
    assert inst.category == "literature"
    assert inst.frequency == "M"
    assert inst.name == "bloom_2009_uncertainty"


def test_event_count_matches_appendix():
    """Bloom 2009 paper Table A.1 lists 17 large-uncertainty episodes."""
    assert len(BLOOM_2009_EVENTS) == 17


def test_event_months_are_indicator_one():
    """Series value is 1.0 at each event month, 0.0 elsewhere."""
    inst = load()
    for date in BLOOM_2009_EVENTS:
        ts = pd.Timestamp(date).to_period("M").to_timestamp()
        assert inst.series.loc[ts] == 1.0, f"event {date} not marked"


def test_non_event_months_are_zero():
    """Months outside the event list are 0.0 (not NaN)."""
    inst = load()
    # Pick a month known to be quiet: Feb 1995
    quiet = pd.Timestamp("1995-02-01")
    assert inst.series.loc[quiet] == 0.0


def test_series_covers_full_sample():
    """Series spans Jan-1962 to Dec-2008 inclusive."""
    inst = load()
    assert inst.series.index.min() == pd.Timestamp("1962-01-01")
    assert inst.series.index.max() == pd.Timestamp("2008-12-01")
    # Expect 47 years × 12 months = 564 observations.
    assert len(inst.series) == 564


def test_series_sum_equals_event_count():
    """Sum of indicator series equals the documented event count."""
    inst = load()
    assert inst.series.sum() == 17.0


def test_metadata_includes_reference_and_event_dates():
    inst = load()
    assert "reference" in inst.metadata
    assert "event_dates" in inst.metadata
    assert len(inst.metadata["event_dates"]) == 17
```

- [ ] **Step 4: Confirm failure**

Run: `pytest tests/test_instruments/literature/test_bloom_2009.py -v`
Expected: ImportError on `from puremacro.instruments.literature.bloom_2009 import load, BLOOM_2009_EVENTS`.

- [ ] **Step 5: Implement the loader**

Create `puremacro/instruments/literature/bloom_2009.py`:

```python
"""Bloom (2009) uncertainty shock indicator series.

Bloom identifies 17 large-uncertainty episodes from major political,
economic, or financial events with associated stock-volatility spikes.
Each event is a discrete announcement-month date. This loader emits a
monthly indicator series (value 1.0 at each event month, 0.0 elsewhere)
over the standard sample Jan-1962 to Dec-2008.

The 17 dates are the canonical list from Bloom (2009, Econometrica)
Table A.1, with the post-publication standard extension to include
the credit crunch episode of October 2008 that Bloom himself uses
in subsequent work. They are baked into this module rather than
fetched from a website because the list is small, stable, and
reproducible without network access.

Reference
---------
Bloom, N. (2009). The impact of uncertainty shocks. Econometrica 77(3), 623-685.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .._core import Instrument


# Bloom (2009) Table A.1 — 17 large-uncertainty announcement months.
# Events with multi-month resolution use the announcement month per Bloom's
# convention.
BLOOM_2009_EVENTS: tuple[str, ...] = (
    "1962-10-01",  # Cuban Missile Crisis
    "1963-11-01",  # Assassination of JFK
    "1966-08-01",  # Vietnam buildup
    "1970-05-01",  # Cambodia / Kent State
    "1973-12-01",  # OPEC I, Arab-Israeli War
    "1974-10-01",  # Franklin National
    "1978-11-01",  # OPEC II
    "1980-03-01",  # Afghanistan, Iran hostage crisis
    "1982-10-01",  # Monetary cycle turning point
    "1987-11-01",  # Black Monday
    "1990-10-01",  # Gulf War I
    "1997-11-01",  # Asian Crisis
    "1998-09-01",  # Russian / LTCM
    "2001-09-01",  # 9/11
    "2002-09-01",  # WorldCom and Enron
    "2003-02-01",  # Gulf War II
    "2008-10-01",  # Credit crunch / GFC peak
)


_REFERENCE = (
    "Bloom, N. (2009). The impact of uncertainty shocks. "
    "Econometrica 77(3), 623-685."
)


def load() -> Instrument:
    """Return Bloom (2009) uncertainty-event indicator series.

    Returns
    -------
    Instrument
        Monthly series spanning Jan-1962 through Dec-2008 (564 obs).
        Value 1.0 at each of the 17 event months; 0.0 elsewhere.
        Category ``"literature"``, frequency ``"M"``.
    """
    idx = pd.date_range("1962-01-01", "2008-12-01", freq="MS")
    s = pd.Series(np.zeros(len(idx)), index=idx, name="bloom_2009_uncertainty")
    for date in BLOOM_2009_EVENTS:
        ts = pd.Timestamp(date)
        if ts in s.index:
            s.loc[ts] = 1.0
    return Instrument(
        series=s,
        name="bloom_2009_uncertainty",
        source="Bloom 2009 large-uncertainty episodes (Table A.1)",
        category="literature",
        frequency="M",
        metadata={
            "reference": _REFERENCE,
            "event_dates": list(BLOOM_2009_EVENTS),
        },
    )


__all__ = ["load", "BLOOM_2009_EVENTS"]
```

- [ ] **Step 6: Confirm green**

Run: `pytest tests/test_instruments/literature/test_bloom_2009.py -v`
Expected: 7 passed.

---

## Task 2: Shared CSV → Instrument helper + BBD EPU loader

**Files:**
- Create: `puremacro/instruments/literature/_helpers.py`
- Create: `puremacro/instruments/literature/bbd_epu.py`
- Modify: `puremacro/instruments/literature/__init__.py` (add re-export)
- Create: `tests/test_instruments/literature/test_bbd_epu.py`

The Baker-Bloom-Davis Economic Policy Uncertainty index publishes a free monthly US series at policyuncertainty.com. The canonical URL is `https://www.policyuncertainty.com/media/US_Policy_Uncertainty_Data.csv`. The CSV format (verified across multiple historical snapshots) has columns `Year,Month,News_Based_Policy_Uncert_Index` plus several detail columns; we use the `News_Based_Policy_Uncert_Index` column as the headline series.

The shared helper `_csv_to_instrument()` takes a parsed DataFrame plus mapping kwargs and returns an `Instrument`. It handles the common path: parse a date column or build a date from year/month columns, take the named value column, return Instrument.

- [ ] **Step 1: Write failing tests**

Create `tests/test_instruments/literature/test_bbd_epu.py`:

```python
"""Tests for puremacro.instruments.literature.bbd_epu."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from puremacro.instruments import Instrument
from puremacro.instruments.literature.bbd_epu import load


_SYNTHETIC_CSV = """Year,Month,News_Based_Policy_Uncert_Index
1985,1,55.42
1985,2,87.16
1985,3,108.06
2001,9,392.58
"""


def test_load_with_csv_path_returns_instrument(tmp_path):
    csv = tmp_path / "epu.csv"
    csv.write_text(_SYNTHETIC_CSV)
    inst = load(csv_path=csv)
    assert isinstance(inst, Instrument)
    assert inst.category == "literature"
    assert inst.frequency == "M"
    assert inst.name == "bbd_epu_us"


def test_load_csv_extracts_correct_values(tmp_path):
    csv = tmp_path / "epu.csv"
    csv.write_text(_SYNTHETIC_CSV)
    inst = load(csv_path=csv)
    # Series is indexed by month-start; check three known values.
    assert inst.series.loc[pd.Timestamp("1985-01-01")] == pytest.approx(55.42)
    assert inst.series.loc[pd.Timestamp("2001-09-01")] == pytest.approx(392.58)
    assert len(inst.series) == 4


def test_metadata_has_reference():
    csv_text = _SYNTHETIC_CSV
    import io as _io
    inst_csv_path = pytest.importorskip("pathlib").Path
    # Use a tmp_path-equivalent via in-memory
    from pathlib import Path
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(csv_text)
        path = Path(f.name)
    inst = load(csv_path=path)
    assert "reference" in inst.metadata
    assert "Baker" in inst.metadata["reference"]
    assert "Bloom" in inst.metadata["reference"]
    assert "Davis" in inst.metadata["reference"]


def test_load_no_csv_no_network_raises_runtime_error(monkeypatch):
    """If csv_path is None and the network fetch fails, raise a
    RuntimeError pointing at policyuncertainty.com."""
    from puremacro.instruments.literature import bbd_epu as _mod
    def _fail_fetch(_url):
        raise OSError("simulated network failure")
    monkeypatch.setattr(_mod, "safe_get_bytes", _fail_fetch)
    with pytest.raises(RuntimeError, match="policyuncertainty.com"):
        load()
```

- [ ] **Step 2: Confirm failure**

Run: `pytest tests/test_instruments/literature/test_bbd_epu.py -v`
Expected: ImportError on `from puremacro.instruments.literature.bbd_epu import load`.

- [ ] **Step 3: Create the shared helper**

Create `puremacro/instruments/literature/_helpers.py`:

```python
"""Shared adapters for literature CSV → Instrument conversion."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._core import Instrument


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


__all__ = ["_csv_to_instrument"]
```

- [ ] **Step 4: Create the BBD EPU loader**

Create `puremacro/instruments/literature/bbd_epu.py`:

```python
"""Baker-Bloom-Davis Economic Policy Uncertainty index (US, monthly).

The index is a free monthly publication from policyuncertainty.com,
constructed from a news-coverage component, tax-code-expiration counts,
and disagreement among economic forecasters. We fetch the news-based
US headline series, which is the most-cited variant in macro VARs.

Reference
---------
Baker, S.R., Bloom, N., Davis, S.J. (2016). Measuring economic policy
uncertainty. QJE 131(4), 1593-1636.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from .._core import Instrument
from ...narrative.sources._http import safe_get_bytes
from ._helpers import _csv_to_instrument


_MIRROR = "https://www.policyuncertainty.com/media/US_Policy_Uncertainty_Data.csv"

_REFERENCE = (
    "Baker, S.R., Bloom, N., Davis, S.J. (2016). Measuring economic "
    "policy uncertainty. QJE 131(4), 1593-1636."
)


def load(*, csv_path: str | Path | None = None) -> Instrument:
    """Load the BBD EPU US monthly index.

    Parameters
    ----------
    csv_path : optional local path to the CSV. When None, attempt the
        canonical policyuncertainty.com download.

    Returns
    -------
    Instrument
        Monthly series, name ``"bbd_epu_us"``, category ``"literature"``,
        frequency ``"M"``.
    """
    if csv_path is not None:
        df = pd.read_csv(csv_path)
    else:
        try:
            raw = safe_get_bytes(_MIRROR)
            df = pd.read_csv(io.BytesIO(raw))
        except Exception as e:
            raise RuntimeError(
                "Could not fetch BBD EPU index. Download "
                "US_Policy_Uncertainty_Data.csv from "
                "https://www.policyuncertainty.com/ and pass csv_path=."
            ) from e
    # Drop rows with missing year or month (the published CSV often has a
    # trailing footer row with NaNs).
    df = df.dropna(subset=["Year", "Month"]).copy()
    return _csv_to_instrument(
        df,
        name="bbd_epu_us",
        source="Baker-Bloom-Davis Economic Policy Uncertainty (US, monthly)",
        frequency="M",
        value_col="News_Based_Policy_Uncert_Index",
        year_col="Year",
        month_col="Month",
        metadata={"reference": _REFERENCE},
    )


__all__ = ["load"]
```

- [ ] **Step 5: Update `literature/__init__.py`**

Edit `puremacro/instruments/literature/__init__.py`:

```python
"""Literature shock instruments — canonical identified-shock series from the
empirical macro literature, wrapped as :class:`puremacro.instruments.Instrument`.

Each loader exposes a top-level ``load(...) -> Instrument`` function and is
registered in :mod:`puremacro.instruments._catalog` so callers can also
reach it via :func:`puremacro.instruments.load`.
"""
from .bloom_2009 import load as load_bloom_2009
from .bbd_epu import load as load_bbd_epu

__all__ = ["load_bloom_2009", "load_bbd_epu"]
```

- [ ] **Step 6: Confirm green**

Run: `pytest tests/test_instruments/literature/test_bbd_epu.py -v`
Expected: 4 passed.

---

## Task 3: Caldara-Iacoviello GPR loader

**Files:**
- Create: `puremacro/instruments/literature/caldara_iacoviello_gpr.py`
- Modify: `puremacro/instruments/literature/__init__.py` (add re-export)
- Create: `tests/test_instruments/literature/test_caldara_iacoviello_gpr.py`

The Caldara-Iacoviello (2022, AER) Geopolitical Risk index is a monthly news-based index published at matteoiacoviello.com/gpr.htm. The canonical CSV is `https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls` for the Excel version; a CSV mirror is `https://www.matteoiacoviello.com/gpr_files/gpr_web_latest.csv` (different historical snapshots; we use the CSV path). The headline column is `GPR` for the broad index.

Use the same shared `_csv_to_instrument` helper.

- [ ] **Step 1: Write failing tests**

Create `tests/test_instruments/literature/test_caldara_iacoviello_gpr.py`:

```python
"""Tests for puremacro.instruments.literature.caldara_iacoviello_gpr."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.instruments import Instrument
from puremacro.instruments.literature.caldara_iacoviello_gpr import load


_SYNTHETIC_CSV = """month,GPR
1985-01-01,68.41
1985-02-01,70.22
2001-09-01,295.10
"""


def test_load_with_csv_path_returns_instrument(tmp_path):
    csv = tmp_path / "gpr.csv"
    csv.write_text(_SYNTHETIC_CSV)
    inst = load(csv_path=csv)
    assert isinstance(inst, Instrument)
    assert inst.category == "literature"
    assert inst.frequency == "M"
    assert inst.name == "caldara_iacoviello_gpr"


def test_load_extracts_correct_values(tmp_path):
    csv = tmp_path / "gpr.csv"
    csv.write_text(_SYNTHETIC_CSV)
    inst = load(csv_path=csv)
    assert inst.series.loc[pd.Timestamp("1985-01-01")] == pytest.approx(68.41)
    assert inst.series.loc[pd.Timestamp("2001-09-01")] == pytest.approx(295.10)
    assert len(inst.series) == 3


def test_metadata_has_reference(tmp_path):
    csv = tmp_path / "gpr.csv"
    csv.write_text(_SYNTHETIC_CSV)
    inst = load(csv_path=csv)
    assert "reference" in inst.metadata
    assert "Caldara" in inst.metadata["reference"]
    assert "Iacoviello" in inst.metadata["reference"]


def test_load_no_csv_no_network_raises_runtime_error(monkeypatch):
    from puremacro.instruments.literature import caldara_iacoviello_gpr as _mod
    def _fail_fetch(_url):
        raise OSError("simulated network failure")
    monkeypatch.setattr(_mod, "safe_get_bytes", _fail_fetch)
    with pytest.raises(RuntimeError, match="matteoiacoviello.com"):
        load()
```

- [ ] **Step 2: Confirm failure**

Run: `pytest tests/test_instruments/literature/test_caldara_iacoviello_gpr.py -v`
Expected: ImportError.

- [ ] **Step 3: Create the loader**

Create `puremacro/instruments/literature/caldara_iacoviello_gpr.py`:

```python
"""Caldara-Iacoviello (2022) Geopolitical Risk (GPR) index (monthly).

A news-based monthly index of geopolitical risk constructed from
counts of articles in major newspapers discussing adverse geopolitical
events. Published at matteoiacoviello.com/gpr.htm; we fetch the CSV
mirror.

Reference
---------
Caldara, D. and Iacoviello, M. (2022). Measuring geopolitical risk.
American Economic Review 112(4), 1194-1225.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from .._core import Instrument
from ...narrative.sources._http import safe_get_bytes
from ._helpers import _csv_to_instrument


_MIRROR = "https://www.matteoiacoviello.com/gpr_files/gpr_web_latest.csv"

_REFERENCE = (
    "Caldara, D. and Iacoviello, M. (2022). Measuring geopolitical risk. "
    "American Economic Review 112(4), 1194-1225."
)


def load(*, csv_path: str | Path | None = None) -> Instrument:
    """Load the Caldara-Iacoviello GPR monthly index.

    Parameters
    ----------
    csv_path : optional local path to the CSV. When None, attempt the
        canonical matteoiacoviello.com download.

    Returns
    -------
    Instrument
        Monthly GPR series, name ``"caldara_iacoviello_gpr"``,
        category ``"literature"``, frequency ``"M"``.
    """
    if csv_path is not None:
        df = pd.read_csv(csv_path)
    else:
        try:
            raw = safe_get_bytes(_MIRROR)
            df = pd.read_csv(io.BytesIO(raw))
        except Exception as e:
            raise RuntimeError(
                "Could not fetch Caldara-Iacoviello GPR index. Download "
                "the CSV from https://www.matteoiacoviello.com/gpr.htm "
                "and pass csv_path=."
            ) from e
    df = df.dropna(subset=["month", "GPR"]).copy()
    return _csv_to_instrument(
        df,
        name="caldara_iacoviello_gpr",
        source="Caldara-Iacoviello (2022) Geopolitical Risk index",
        frequency="M",
        value_col="GPR",
        date_col="month",
        metadata={"reference": _REFERENCE},
    )


__all__ = ["load"]
```

- [ ] **Step 4: Update `literature/__init__.py`**

Edit `puremacro/instruments/literature/__init__.py`:

```python
"""Literature shock instruments — canonical identified-shock series from the
empirical macro literature, wrapped as :class:`puremacro.instruments.Instrument`.

Each loader exposes a top-level ``load(...) -> Instrument`` function and is
registered in :mod:`puremacro.instruments._catalog` so callers can also
reach it via :func:`puremacro.instruments.load`.
"""
from .bloom_2009 import load as load_bloom_2009
from .bbd_epu import load as load_bbd_epu
from .caldara_iacoviello_gpr import load as load_caldara_iacoviello_gpr

__all__ = ["load_bloom_2009", "load_bbd_epu", "load_caldara_iacoviello_gpr"]
```

- [ ] **Step 5: Confirm green**

Run: `pytest tests/test_instruments/literature/test_caldara_iacoviello_gpr.py -v`
Expected: 4 passed.

---

## Task 4: Romer-Romer 2004 monetary shocks loader

**Files:**
- Create: `puremacro/instruments/literature/romer_romer_2004.py`
- Modify: `puremacro/instruments/literature/__init__.py` (add re-export)
- Create: `tests/test_instruments/literature/test_romer_romer_2004.py`

Romer & Romer (2004, AER) construct a residual measure of monetary policy shocks: the residual from regressing the FOMC's intended FFR change on Greenbook forecasts. The series is quarterly 1969Q1–1996Q4 in the original paper; extensions through 2007 (Coibion 2012) and beyond circulate widely. The canonical raw CSV ships from Romer's website; without a single guaranteed URL across mirror snapshots, this loader uses a `requires_fixture=True` discipline (mirror download is attempted but failure is expected; user supplies CSV).

Schema: two columns — `date` (YYYYqQ string or quarter-start date) and `shock` (the residual). Other column names that have circulated in mirror copies: `RR_shock`, `intended_residual`, `MP_shock`. We require the user to pass `value_col=` if non-default.

- [ ] **Step 1: Write failing tests**

Create `tests/test_instruments/literature/test_romer_romer_2004.py`:

```python
"""Tests for puremacro.instruments.literature.romer_romer_2004."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.instruments import Instrument
from puremacro.instruments.literature.romer_romer_2004 import load


_SYNTHETIC_CSV = """date,RR_shock
1969-01-01,0.12
1969-04-01,-0.05
1969-07-01,0.08
1980-04-01,1.42
"""


def test_load_with_csv_path_returns_instrument(tmp_path):
    csv = tmp_path / "rr2004.csv"
    csv.write_text(_SYNTHETIC_CSV)
    inst = load(csv_path=csv)
    assert isinstance(inst, Instrument)
    assert inst.category == "literature"
    assert inst.frequency == "Q"
    assert inst.name == "rr_2004_monetary"


def test_load_with_csv_path_default_value_col(tmp_path):
    """Default value_col is 'RR_shock' but accepts an override."""
    csv = tmp_path / "rr2004.csv"
    csv.write_text(_SYNTHETIC_CSV)
    inst = load(csv_path=csv)
    assert inst.series.loc[pd.Timestamp("1980-04-01")] == pytest.approx(1.42)
    assert len(inst.series) == 4


def test_load_with_alternative_value_col(tmp_path):
    """The user can specify a non-default value column."""
    csv = tmp_path / "rr2004_alt.csv"
    csv.write_text("date,intended_residual\n1969-01-01,0.12\n1969-04-01,-0.05\n")
    inst = load(csv_path=csv, value_col="intended_residual")
    assert inst.series.loc[pd.Timestamp("1969-01-01")] == pytest.approx(0.12)


def test_metadata_has_reference(tmp_path):
    csv = tmp_path / "rr2004.csv"
    csv.write_text(_SYNTHETIC_CSV)
    inst = load(csv_path=csv)
    assert "reference" in inst.metadata
    assert "Romer" in inst.metadata["reference"]


def test_load_no_csv_no_network_raises_runtime_error(monkeypatch):
    from puremacro.instruments.literature import romer_romer_2004 as _mod
    def _fail_fetch(_url):
        raise OSError("simulated network failure")
    monkeypatch.setattr(_mod, "safe_get_bytes", _fail_fetch)
    with pytest.raises(RuntimeError, match="Romer"):
        load()
```

- [ ] **Step 2: Confirm failure**

Run: `pytest tests/test_instruments/literature/test_romer_romer_2004.py -v`
Expected: ImportError.

- [ ] **Step 3: Create the loader**

Create `puremacro/instruments/literature/romer_romer_2004.py`:

```python
"""Romer-Romer (2004) narrative monetary policy shocks.

The Romer-Romer measure is the residual from regressing the FOMC's
intended federal-funds-rate change (from internal Greenbook records)
on the Greenbook's own GDP-growth and inflation forecasts. The
residual identifies the fraction of intended policy that is NOT
explained by the staff's outlook — the "exogenous" monetary shock.

Original paper covers 1969Q1-1996Q4. Several extensions circulate
(Coibion 2012, Wieland-Yang 2020); pass ``csv_path=`` to load any
extended version with the ``date,RR_shock`` schema.

Reference
---------
Romer, C.D. and Romer, D.H. (2004). A new measure of monetary shocks:
derivation and implications. AER 94(4), 1055-1084.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from .._core import Instrument
from ...narrative.sources._http import safe_get_bytes
from ._helpers import _csv_to_instrument


# The Romer-Romer 2004 data file path on David Romer's UC Berkeley site
# has rotated multiple times. The most-recent stable mirror is the
# Coibion (2012) extension, which preserves the same column structure.
_MIRROR = (
    "https://eml.berkeley.edu/~dromer/papers/RomerandRomerDataAppendix.csv"
)

_REFERENCE = (
    "Romer, C.D. and Romer, D.H. (2004). A new measure of monetary "
    "shocks: derivation and implications. AER 94(4), 1055-1084."
)


def load(
    *,
    csv_path: str | Path | None = None,
    value_col: str = "RR_shock",
) -> Instrument:
    """Load the Romer-Romer 2004 monetary shock series.

    Parameters
    ----------
    csv_path : optional local path to the CSV. When None, attempt the
        mirror download; raises RuntimeError if both fail.
    value_col : str, default ``"RR_shock"``
        Name of the shock column in the CSV. Common alternatives:
        ``"intended_residual"``, ``"MP_shock"``.

    Returns
    -------
    Instrument
        Quarterly series, name ``"rr_2004_monetary"``, category
        ``"literature"``, frequency ``"Q"``.
    """
    if csv_path is not None:
        df = pd.read_csv(csv_path)
    else:
        try:
            raw = safe_get_bytes(_MIRROR)
            df = pd.read_csv(io.BytesIO(raw))
        except Exception as e:
            raise RuntimeError(
                "Could not fetch Romer-Romer 2004 monetary shock series. "
                "Download the CSV from David Romer's website "
                "(eml.berkeley.edu/~dromer/) and pass csv_path=."
            ) from e
    df = df.dropna(subset=["date", value_col]).copy()
    return _csv_to_instrument(
        df,
        name="rr_2004_monetary",
        source="Romer-Romer 2004 narrative monetary shocks",
        frequency="Q",
        value_col=value_col,
        date_col="date",
        metadata={"reference": _REFERENCE},
    )


__all__ = ["load"]
```

- [ ] **Step 4: Update `literature/__init__.py`**

Edit `puremacro/instruments/literature/__init__.py`:

```python
"""Literature shock instruments — canonical identified-shock series from the
empirical macro literature, wrapped as :class:`puremacro.instruments.Instrument`.

Each loader exposes a top-level ``load(...) -> Instrument`` function and is
registered in :mod:`puremacro.instruments._catalog` so callers can also
reach it via :func:`puremacro.instruments.load`.
"""
from .bloom_2009 import load as load_bloom_2009
from .bbd_epu import load as load_bbd_epu
from .caldara_iacoviello_gpr import load as load_caldara_iacoviello_gpr
from .romer_romer_2004 import load as load_romer_romer_2004

__all__ = [
    "load_bloom_2009",
    "load_bbd_epu",
    "load_caldara_iacoviello_gpr",
    "load_romer_romer_2004",
]
```

- [ ] **Step 5: Confirm green**

Run: `pytest tests/test_instruments/literature/test_romer_romer_2004.py -v`
Expected: 5 passed.

---

## Task 5: Catalog wiring — register the 4 new entries

**Files:**
- Modify: `puremacro/instruments/_catalog.py` — append a new section
- Modify: `tests/test_instruments/test_catalog.py` — extend size assertions, add available-flag assertion for Bloom

The 4 entries land in a new "Literature" section of `_catalog.py`. Bloom 2009 is `requires_network=False` AND `requires_fixture=False` — so it shows up by default in `list_available()`. The other 3 are `requires_network=True` (URL fetch).

- [ ] **Step 1: Update catalog discipline tests**

Edit `tests/test_instruments/test_catalog.py`. Find `test_total_phase1_catalog_size_is_exactly_25` and rename to `test_total_catalog_size_is_exactly_29` while updating the assertion:

```python
def test_total_catalog_size_is_exactly_29():
    """6 replications + 6 connectors + 1 monetary HFI + 12 stubs + 4 literature = 29."""
    assert len(_registry._REGISTRY) == 29
```

Also find `test_total_phase1_catalog_size_at_least_25` and update to `>= 29`:

```python
def test_total_catalog_size_at_least_29():
    """6 replications + 6 connectors + 1 monetary HFI + 12 stubs + 4 literature = 29."""
    assert len(_registry._REGISTRY) >= 29
```

(Update the test name in both places — `_25` → `_29`.)

Append two new tests at the end of the file:

```python
_EXPECTED_LITERATURE_KEYS = {
    "bloom_2009_uncertainty",
    "bbd_epu_us",
    "caldara_iacoviello_gpr",
    "rr_2004_monetary",
}


def test_all_four_literature_entries_registered():
    keys = set(_registry._REGISTRY.keys())
    missing = _EXPECTED_LITERATURE_KEYS - keys
    assert not missing, f"literature entries missing: {missing}"


def test_bloom_2009_is_available_by_default():
    """Bloom 2009 events are baked-in (no network, no fixture). It must
    appear in the default list_available() output without flags."""
    df = list_available()
    assert "bloom_2009_uncertainty" in df["key"].values
    spec = _registry._REGISTRY["bloom_2009_uncertainty"]
    assert spec.requires_network is False
    assert spec.requires_fixture is False


def test_three_literature_entries_require_network():
    """BBD EPU, GPR, RR2004 fetch CSVs from canonical URLs."""
    for key in ("bbd_epu_us", "caldara_iacoviello_gpr", "rr_2004_monetary"):
        spec = _registry._REGISTRY[key]
        assert spec.requires_network is True, f"{key} should require network"


def test_every_literature_entry_has_non_empty_reference():
    for key in _EXPECTED_LITERATURE_KEYS:
        spec = _registry._REGISTRY[key]
        assert spec.reference and len(spec.reference) > 10
```

- [ ] **Step 2: Confirm failure**

Run: `pytest tests/test_instruments/test_catalog.py -v -k "literature or 29"`
Expected: AssertionError on `len(_REGISTRY) == 29` (still at 25), plus AssertionError on the 4 expected literature keys.

- [ ] **Step 3: Append the 4 catalog entries**

Edit `puremacro/instruments/_catalog.py`. Append at the end of the file (after the stubs section):

```python
# --------------------------------------------------------------------------
# Literature shock loaders (4) — see puremacro/instruments/literature/
# --------------------------------------------------------------------------
from .literature import (
    load_bloom_2009,
    load_bbd_epu,
    load_caldara_iacoviello_gpr,
    load_romer_romer_2004,
)


register(InstrumentSpec(
    key="bloom_2009_uncertainty",
    name="Bloom 2009 uncertainty events",
    category="literature",
    description=(
        "17 large-uncertainty announcement-month indicators from Bloom "
        "(2009) Table A.1, monthly Jan-1962 to Dec-2008. Baked-in event "
        "list — no network or fixture needed; fully reproducible."
    ),
    reference="Bloom, N. (2009). The impact of uncertainty shocks. Econometrica 77(3), 623-685.",
    loader=load_bloom_2009,
    country=None,
    frequency="M",
    requires_network=False,
    requires_fixture=False,
))


register(InstrumentSpec(
    key="bbd_epu_us",
    name="Baker-Bloom-Davis EPU (US, monthly)",
    category="literature",
    description=(
        "News-based monthly Economic Policy Uncertainty index for the "
        "US, fetched from policyuncertainty.com. Pass csv_path= to "
        "skip the network call."
    ),
    reference="Baker, S.R., Bloom, N., Davis, S.J. (2016). Measuring economic policy uncertainty. QJE 131(4), 1593-1636.",
    loader=load_bbd_epu,
    country="USA",
    frequency="M",
    requires_network=True,
    requires_fixture=False,
))


register(InstrumentSpec(
    key="caldara_iacoviello_gpr",
    name="Caldara-Iacoviello Geopolitical Risk index (monthly)",
    category="literature",
    description=(
        "News-based monthly Geopolitical Risk index, fetched from "
        "matteoiacoviello.com/gpr.htm. Pass csv_path= to skip the "
        "network call."
    ),
    reference="Caldara, D. and Iacoviello, M. (2022). Measuring geopolitical risk. American Economic Review 112(4), 1194-1225.",
    loader=load_caldara_iacoviello_gpr,
    country=None,
    frequency="M",
    requires_network=True,
    requires_fixture=False,
))


register(InstrumentSpec(
    key="rr_2004_monetary",
    name="Romer-Romer 2004 narrative monetary shocks",
    category="literature",
    description=(
        "Quarterly residual from regressing FOMC intended FFR changes on "
        "Greenbook forecasts (1969Q1-1996Q4 base, extensions to ~2007). "
        "Mirror download attempted from David Romer's UC Berkeley site; "
        "if it fails, pass csv_path= with a local copy."
    ),
    reference="Romer, C.D. and Romer, D.H. (2004). A new measure of monetary shocks: derivation and implications. AER 94(4), 1055-1084.",
    loader=load_romer_romer_2004,
    country="USA",
    frequency="Q",
    requires_network=True,
    requires_fixture=False,
))
```

- [ ] **Step 4: Confirm green**

Run: `pytest tests/test_instruments/test_catalog.py -v`
Expected: all catalog tests pass — total entries 29, 4 literature keys present, Bloom marked `available=True`, other 3 marked `requires_network=True`.

- [ ] **Step 5: Confirm full instruments suite green**

Run: `pytest tests/test_instruments/ -v`
Expected: 57 (previous) + ~22 (new tests) = ~79 passed.

- [ ] **Step 6: Verify no warnings on import**

Run: `python -W error -c "import puremacro.instruments; print(len(puremacro.instruments._registry._REGISTRY))"`
Expected: prints `29`, no warnings (all 29 keys unique).

- [ ] **Step 7: Sanity-check Bloom 2009 via the registry**

Run:
```bash
python -c "
from puremacro.instruments import load, list_available
df = list_available()
print(f'available count: {len(df)}')
inst = load('bloom_2009_uncertainty')
print(f'name: {inst.name}, n_obs: {len(inst.series)}, n_events: {int(inst.series.sum())}')
"
```
Expected: prints `available count: 1` (Bloom is the only `available=True` entry), then `name: bloom_2009_uncertainty, n_obs: 564, n_events: 17`.

---

## Task 6: Version bump + snapshot regen + CHANGELOG + memory

**Files:**
- Modify: `pyproject.toml` (`0.5.0 → 0.5.1`)
- Modify: `puremacro/__init__.py` (`__version__`)
- Modify: `tests/test_import.py`
- Modify: `tests/fixtures/public_api_snapshot.json` (regenerate)
- Modify: `CHANGELOG.md`
- Modify: `~/.claude/projects/.../memory/project_puremacro.md`

- [ ] **Step 1: Run full suite to surface any regressions**

Run: `cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro" && pytest -x -q 2>&1 | tail -5`
Expected: ~470 passed, 9 skipped (444 baseline + ~26 new tests).

- [ ] **Step 2: Regenerate the public-API snapshot**

The new `puremacro.instruments.literature` subpackage exposes 4 loader functions, so the snapshot needs regenerating. Run:

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro" && \
python -c "
import sys; sys.path.insert(0, 'tests')
from test_public_api import _collect_current_api
import json
print(json.dumps(_collect_current_api(), indent=2))
" > tests/fixtures/public_api_snapshot.json
```

Then run `pytest tests/test_public_api.py -v` and confirm PASS.

- [ ] **Step 3: Spot-check the snapshot diff**

Run: `grep -c "literature" tests/fixtures/public_api_snapshot.json`
Expected: at least 5 (one for each of the 4 loader modules + the parent subpackage).

- [ ] **Step 4: Bump pyproject.toml**

Edit `pyproject.toml`. Change `version = "0.5.0"` to `version = "0.5.1"`.

- [ ] **Step 5: Bump `__version__`**

Edit `puremacro/__init__.py`. Change `__version__ = "0.5.0"` to `__version__ = "0.5.1"`.

- [ ] **Step 6: Bump test_import expected version**

Edit `tests/test_import.py`. Change `assert puremacro.__version__ == "0.5.0"` to `assert puremacro.__version__ == "0.5.1"`.

- [ ] **Step 7: Confirm import test green**

Run: `pytest tests/test_import.py -v`
Expected: PASS.

- [ ] **Step 8: Add CHANGELOG entry**

Edit `CHANGELOG.md`. Insert immediately after the file header and before `## 0.5.0 — 2026-05-03`:

```markdown
## 0.5.1 — 2026-05-03

Patch release — adds 4 literature shock loaders to the `puremacro.instruments` registry, expanding the catalog from 25 → 29 entries. **Bloom 2009 is the first fully-offline `available=True` entry** in the registry: no network, no fixture, fully reproducible.

### Added
- `puremacro.instruments.literature` — new subpackage with 4 canonical literature shock loaders, all returning `Instrument` directly:
  - `load_bloom_2009()` — Bloom (2009) uncertainty event indicator series. 17 hand-coded events from the paper's Table A.1 baked into Python; emits a monthly indicator series Jan-1962 to Dec-2008 (564 obs, 17 ones, rest zeros). Fully offline. Catalogued as `bloom_2009_uncertainty`.
  - `load_bbd_epu(*, csv_path=None)` — Baker-Bloom-Davis Economic Policy Uncertainty index (US, monthly news-based). Fetches from policyuncertainty.com. Catalogued as `bbd_epu_us`.
  - `load_caldara_iacoviello_gpr(*, csv_path=None)` — Caldara-Iacoviello Geopolitical Risk index (monthly). Fetches from matteoiacoviello.com. Catalogued as `caldara_iacoviello_gpr`.
  - `load_romer_romer_2004(*, csv_path=None, value_col="RR_shock")` — Romer-Romer (2004) narrative monetary shock residual (quarterly). Fetches from David Romer's UC Berkeley site. Catalogued as `rr_2004_monetary`.
- `puremacro.instruments.literature._helpers._csv_to_instrument()` — shared CSV → Instrument adapter handling the (date_col vs year+month columns) parsing branch.

### Internal
- `tests/test_instruments/literature/` — new test directory: ~22 tests across 4 loaders.
- `tests/test_instruments/test_catalog.py` — size assertions tightened to 29 entries; new tests for literature-key membership, Bloom availability, network-required flags.
- `tests/fixtures/public_api_snapshot.json` regenerated to record the new `puremacro.instruments.literature` subpackage and 4 loader modules.

### Out of scope (still deferred)
- FRED/BIS/IMF external-CSV loaders.
- `Instrument.compose()` operator.
- Per-record country threading in `score_keyword`.
- JSON serializability of `Instrument.metadata`.

### Tests
- Pre-release baseline: 444 passing, 9 skipped (0.5.0).
- Post-release: ~470 passing, 9 skipped.
```

- [ ] **Step 9: Append memory entry**

Edit `/Users/jalonso/.claude/projects/-Users-jalonso-Library-CloudStorage-GoogleDrive-jorge-alonsoortiz-gmail-com-My-Drive-MAV/memory/project_puremacro.md`. Append at the end:

```markdown

**Iteration N+9 step 3 done (2026-05-03) — released as 0.5.1 (patch):**
- New subpackage `puremacro.instruments.literature` with 4 loaders: `load_bloom_2009` (offline — first `available=True` entry in the registry), `load_bbd_epu` (network), `load_caldara_iacoviello_gpr` (network), `load_romer_romer_2004` (network). Each returns `Instrument` directly (not `NarrativeInstrument`).
- Bloom 2009 events are baked into Python from the paper's Table A.1 — 17 dates spanning Cuban Missile Crisis (Oct 1962) to GFC peak (Oct 2008), emitted as monthly indicator series (564 obs, sum=17). No network, no fixture. The `pi.list_available()` default call now returns 1 row instead of 0.
- BBD EPU, GPR, RR2004 use the existing CSV-with-mirror-fallback pattern from narrative replications. Mirror URLs documented; `csv_path=` kwarg accepted to skip network. Failure raises RuntimeError pointing at the canonical source website.
- Shared helper `_csv_to_instrument()` in `instruments/literature/_helpers.py` handles the common DataFrame → Instrument conversion (supports both `date_col` and `year_col + month_col` schemas).
- Catalog grew 25 → 29: 6 narrative replications + 6 narrative connectors + 1 monetary HFI + 12 connector stubs + 4 literature.
- ~26 new tests (~22 in literature subpackage + 4 catalog discipline updates).
- Plan file: `uncertainty_examples/puremacro/docs/plans/2026-05-03-instruments-literature-loaders-051.md`.

**0.5.1 still deferred (Trim B / Phase 2):**
- FRED / BIS / IMF external-CSV loaders.
- `Instrument.compose()` operator.
- Per-record country threading in `score_keyword`.
- JSON serializability of `Instrument.metadata`.

**How to apply:** When the user asks "what's a quick uncertainty shock series I can use?", `pi.load("bloom_2009_uncertainty")` works without any setup. For continuous indices (BBD, GPR, RR2004), users need network access OR a local CSV.
```

- [ ] **Step 10: Final test run**

Run: `pytest -x -q 2>&1 | tail -5`
Expected: ~470 passed, 9 skipped.

- [ ] **Step 11: Pyodide-compat regression check**

Run: `pytest tests/test_pyodide_compat.py -v`
Expected: PASS.

- [ ] **Step 12: Sanity-check the public surface**

Run: `python -c "from puremacro.instruments.literature import load_bloom_2009, load_bbd_epu, load_caldara_iacoviello_gpr, load_romer_romer_2004; print('OK')"`
Expected: `OK`.

Run: `python -c "from puremacro.instruments import load; inst = load('bloom_2009_uncertainty'); print(inst.summary())"`
Expected: prints the multi-line `Instrument.summary()` output for the Bloom 2009 series with 564 obs.

---

## Self-Review Checklist

After implementation:

1. **Spec coverage:** All 4 deferred literature loaders implemented?
   - [x] Bloom 2009 → Task 1
   - [x] Baker-Bloom-Davis EPU → Task 2
   - [x] Caldara-Iacoviello GPR → Task 3
   - [x] Romer-Romer 2004 monetary → Task 4
   - [x] Catalog wiring → Task 5
   - [x] Release coordination → Task 6

2. **Placeholder scan:** No "TBD", "implement later", "appropriate error handling" patterns. Every code step shows the actual code.

3. **Type consistency:**
   - All 4 loaders return `Instrument` directly (not `NarrativeInstrument`).
   - Catalog keys match exactly between `_catalog.py` registrations, the `_EXPECTED_LITERATURE_KEYS` set in tests, and the loader `name=` arguments.
   - Bloom 2009 is the only entry with `requires_network=False AND requires_fixture=False`.
   - All loaders use lazy network fetch (network call is only made if `csv_path is None`).
   - `_csv_to_instrument` signature is consistent across all 3 callers (BBD EPU, GPR, RR2004).

4. **Pyodide hygiene:** No new runtime deps. `safe_get_bytes` is the existing `narrative.sources._http` helper (already Pyodide-correct).

5. **Citation discipline:** Every catalog entry has a full bibliographic reference (Author Year, journal, vol, pp).

6. **Backwards compatibility:** No changes to `Instrument`, `InstrumentLike`, `InstrumentSpec`, `register`, `list_available`, `load`, `describe`, or any of the existing 25 catalog entries. The 4 new entries are purely additive.

7. **Frequency consistency:** Bloom 2009 = M, BBD EPU = M, GPR = M, RR2004 = Q. Catalog entries match the loader returns.

8. **Network failure path:** Each of BBD EPU, GPR, RR2004 has a test (`monkeypatch`-based) that confirms the RuntimeError message names the canonical source website. This is the only meaningful test that exercises the failure path without requiring network access.
