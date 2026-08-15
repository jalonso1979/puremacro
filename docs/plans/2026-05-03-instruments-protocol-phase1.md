# `puremacro.instruments` Phase 1 Implementation Plan (Trim A)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **This workspace is NOT a git repo (Google Drive sync); skip every "commit" step the meta-skill suggests.**

**Goal:** Ship `puremacro.instruments` — a unified `Instrument` wrapper + `InstrumentLike` Protocol + self-describing registry of available identified-shock series. Phase 1 (Trim A) covers protocol, adapters on existing classes, registry primitives, and a catalog of already-shipped things only. Released as **0.5.0**.

**Architecture:** New top-level subpackage `puremacro/instruments/` with three responsibilities: (1) `_core.py` defines a frozen `Instrument` dataclass + `runtime_checkable` `InstrumentLike` Protocol; (2) `_registry.py` defines `InstrumentSpec` + `list_available` / `load` / `describe`; (3) `_catalog.py` declaratively populates the registry with ~25 entries. Existing `NarrativeInstrument` (mutable) and `JKResult` (frozen) gain a single `as_instrument()` method each — no breaking changes. Downstream `proxy_svar` and `lp_iv` signatures are unchanged; polymorphism lives on `Instrument.to_proxy_svar()` / `Instrument.to_lp_iv()`.

**Tech Stack:** Python 3.10+, `dataclasses`, `typing.Protocol`, `pandas`, `numpy`. Pyodide-compatible (no new runtime deps).

**Spec reference:** `docs/specs/2026-05-03-instruments-protocol-design.md` (read it before starting — single source of truth for category enum, field names, and catalog scope).

**Pre-implementation baseline:** 387 passing, 9 skipped (puremacro 0.4.1).

---

## File Structure

### Files created
- `puremacro/instruments/__init__.py` — public re-exports
- `puremacro/instruments/_core.py` — `Instrument` + `InstrumentLike`
- `puremacro/instruments/_registry.py` — `InstrumentSpec` + `list_available` + `load` + `describe`
- `puremacro/instruments/_catalog.py` — registry population (~25 entries)
- `tests/test_instruments/__init__.py` — empty (pytest package marker)
- `tests/test_instruments/test_core.py` — protocol + Instrument tests
- `tests/test_instruments/test_adapters.py` — `as_instrument()` round-trip tests
- `tests/test_instruments/test_registry.py` — registry primitives tests
- `tests/test_instruments/test_catalog.py` — catalog discipline tests

### Files modified
- `puremacro/narrative/types.py` — add `NarrativeInstrument.as_instrument()` method (~20 lines)
- `puremacro/hfi/_results.py` — add `JKResult.as_instrument(*, component, index)` method (~25 lines)
- `pyproject.toml` — version `0.4.1 → 0.5.0`
- `puremacro/__init__.py` — `__version__` bump
- `tests/test_import.py` — expected-version bump
- `tests/fixtures/public_api_snapshot.json` — regenerate (new subpackage + Instrument fields + new `as_instrument` methods)
- `CHANGELOG.md` — add `## 0.5.0 — 2026-05-03` block
- `~/.claude/projects/.../memory/project_puremacro.md` — append iteration N+9 step 2 entry

---

## Task 1: Create `Instrument` dataclass + `InstrumentLike` Protocol

**Files:**
- Create: `puremacro/instruments/__init__.py` (one-line stub for now)
- Create: `puremacro/instruments/_core.py`
- Create: `tests/test_instruments/__init__.py` (empty)
- Create: `tests/test_instruments/test_core.py`

**Result-object standard reminder:** `@dataclass(frozen=True)`, no `__post_init__`, optional `.summary()`, no `.plot()`. Category validation lives in tests, not in the constructor.

- [ ] **Step 1: Create empty test package marker**

```python
# tests/test_instruments/__init__.py
```

(Empty file — pytest needs it as a package marker for sibling test discovery.)

- [ ] **Step 2: Write the failing tests**

Create `tests/test_instruments/test_core.py`:

```python
"""Tests for puremacro.instruments._core: Instrument + InstrumentLike."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from puremacro.instruments import Instrument, InstrumentLike


def _make_quarterly(n=20, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-01", periods=n, freq="QS")
    return pd.Series(rng.standard_normal(n), index=idx, name="z")


# --------------------------------------------------------------------------
# Instrument: structural
# --------------------------------------------------------------------------
def test_instrument_is_frozen_dataclass():
    inst = Instrument(
        series=_make_quarterly(),
        name="test", source="synthetic",
        category="literature", frequency="Q",
    )
    assert dataclasses.is_dataclass(inst)
    with pytest.raises(dataclasses.FrozenInstanceError):
        inst.name = "other"


def test_instrument_metadata_defaults_to_empty_dict():
    inst = Instrument(
        series=_make_quarterly(),
        name="test", source="synthetic",
        category="literature", frequency="Q",
    )
    assert inst.metadata == {}


def test_instrument_carries_documented_fields():
    s = _make_quarterly()
    inst = Instrument(
        series=s, name="test", source="synthetic",
        category="literature", frequency="Q",
        metadata={"foo": "bar"},
    )
    assert inst.series is s
    assert inst.name == "test"
    assert inst.source == "synthetic"
    assert inst.category == "literature"
    assert inst.frequency == "Q"
    assert inst.metadata == {"foo": "bar"}


# --------------------------------------------------------------------------
# InstrumentLike: protocol
# --------------------------------------------------------------------------
def test_instrument_like_is_runtime_checkable():
    """The Protocol must be @runtime_checkable so isinstance() works."""
    class _MyShock:
        def as_instrument(self) -> Instrument:
            return Instrument(
                series=_make_quarterly(),
                name="mine", source="synthetic",
                category="literature", frequency="Q",
            )

    assert isinstance(_MyShock(), InstrumentLike)


def test_instrument_like_rejects_non_conforming():
    """A class without as_instrument() must NOT satisfy the protocol."""
    class _NotAShock:
        pass
    assert not isinstance(_NotAShock(), InstrumentLike)
```

- [ ] **Step 3: Run tests to confirm failure**

Run: `pytest tests/test_instruments/test_core.py -v`
Expected: ImportError on `from puremacro.instruments import Instrument, InstrumentLike` — module does not exist.

- [ ] **Step 4: Create `_core.py`**

Create `puremacro/instruments/_core.py`:

```python
"""Core types for puremacro.instruments.

Defines the canonical ``Instrument`` wrapper (a frozen dataclass) and
the ``InstrumentLike`` Protocol that any class can satisfy by exposing
an ``as_instrument()`` method. Downstream consumers (proxy_svar,
lp_iv, future SVAR-IV variants) accept ``Instrument`` and dispatch
uniformly; upstream classes (``NarrativeInstrument``, ``JKResult``)
provide adapters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd


VALID_CATEGORIES = {
    "narrative_replication",
    "narrative_connector",
    "monetary_hfi",
    "literature",
    "external_csv",
}


@dataclass(frozen=True)
class Instrument:
    """A single identified shock or instrument series with provenance.

    Constructed via ``as_instrument()`` adapters on existing classes
    (:class:`puremacro.narrative.NarrativeInstrument`,
    :class:`puremacro.hfi.JKResult`) or via
    :func:`puremacro.instruments.load`.

    Attributes
    ----------
    series : pd.Series
        Date-indexed proxy/shock values. Any frequency.
    name : str
        Short identifier. Matches the registry key when loaded via
        :func:`load`.
    source : str
        Human-readable provenance, e.g. ``"Ramey 2011 defense buildup events"``.
    category : str
        One of ``"narrative_replication"``, ``"narrative_connector"``,
        ``"monetary_hfi"``, ``"literature"``, ``"external_csv"``.
        See :data:`VALID_CATEGORIES`.
    frequency : str
        Pandas-style frequency code: ``"M"``, ``"Q"``, ``"A"``.
    metadata : dict
        Free-form additional fields (e.g. country, target, reference).
    """

    series: pd.Series
    name: str
    source: str
    category: str
    frequency: str
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class InstrumentLike(Protocol):
    """Anything that knows how to expose itself as an :class:`Instrument`.

    A single-method protocol. The canonical use is to call
    ``obj.as_instrument()`` and then work with the returned
    :class:`Instrument`. Forward-compatible with future shock types.
    """

    def as_instrument(self) -> Instrument: ...


__all__ = ["Instrument", "InstrumentLike", "VALID_CATEGORIES"]
```

- [ ] **Step 5: Create `__init__.py` stub**

Create `puremacro/instruments/__init__.py`:

```python
"""Unified Instrument protocol + discovery registry.

See :class:`Instrument` for the canonical wrapper, :class:`InstrumentLike`
for the Protocol, and :func:`list_available` / :func:`load` for the
discovery registry. Spec: ``docs/specs/2026-05-03-instruments-protocol-design.md``.
"""
from ._core import Instrument, InstrumentLike, VALID_CATEGORIES

__all__ = ["Instrument", "InstrumentLike", "VALID_CATEGORIES"]
```

- [ ] **Step 6: Run tests to confirm green**

Run: `pytest tests/test_instruments/test_core.py -v`
Expected: 5 passed.

---

## Task 2: Add `Instrument` convenience methods (`diagnostics`, `validate_against`, `summary`)

**Files:**
- Modify: `puremacro/instruments/_core.py`
- Modify: `tests/test_instruments/test_core.py`

These three methods are pure-Python computations on `self.series` — no dependency on `puremacro.var` or `puremacro.lp` (those come in Tasks 3–4). Keeping them in this task lets the protocol surface be self-contained for early adopters.

- [ ] **Step 1: Append failing tests**

Add to `tests/test_instruments/test_core.py`:

```python
# --------------------------------------------------------------------------
# Instrument convenience methods
# --------------------------------------------------------------------------
def test_instrument_diagnostics_shape():
    inst = Instrument(
        series=_make_quarterly(n=24, seed=1),
        name="test", source="synthetic",
        category="literature", frequency="Q",
    )
    d = inst.diagnostics()
    assert set(d.keys()) >= {"n_obs", "mean", "std", "first_date", "last_date"}
    assert d["n_obs"] == 24
    assert d["first_date"] is not None
    assert d["last_date"] is not None


def test_instrument_summary_returns_string_with_name_and_source():
    inst = Instrument(
        series=_make_quarterly(),
        name="ramey_2011_defense", source="Ramey 2011 buildup events",
        category="narrative_replication", frequency="Q",
    )
    s = inst.summary()
    assert isinstance(s, str)
    assert "ramey_2011_defense" in s
    assert "Ramey 2011" in s
    assert "Q" in s


def test_instrument_validate_against_returns_correlation():
    """validate_against returns at least a correlation against the benchmark."""
    rng = np.random.default_rng(42)
    n = 40
    idx = pd.date_range("2000-01-01", periods=n, freq="QS")
    base = rng.standard_normal(n)
    z = pd.Series(base, index=idx)
    bench = pd.Series(base + 0.1 * rng.standard_normal(n), index=idx)
    inst = Instrument(
        series=z, name="test", source="synthetic",
        category="literature", frequency="Q",
    )
    result = inst.validate_against(bench)
    assert "correlation" in result
    # Correlation with near-perfect benchmark should exceed 0.9
    assert result["correlation"] > 0.9
```

- [ ] **Step 2: Confirm failure**

Run: `pytest tests/test_instruments/test_core.py -v -k "diagnostics or summary or validate_against"`
Expected: AttributeError on `inst.diagnostics`, `inst.summary`, `inst.validate_against`.

- [ ] **Step 3: Add the three methods to `Instrument`**

Edit `puremacro/instruments/_core.py`. Inside the `Instrument` dataclass body, after the field declarations, add:

```python
    def diagnostics(self) -> dict[str, Any]:
        """Sample-size, central-tendency, and date-coverage statistics."""
        s = self.series.dropna()
        return {
            "n_obs": int(s.shape[0]),
            "mean": float(s.mean()) if s.shape[0] else float("nan"),
            "std": float(s.std()) if s.shape[0] else float("nan"),
            "first_date": str(s.index.min()) if s.shape[0] else None,
            "last_date": str(s.index.max()) if s.shape[0] else None,
        }

    def validate_against(self, benchmark: pd.Series) -> dict[str, Any]:
        """Correlation + overlap diagnostics against a benchmark series."""
        joined = pd.concat([self.series, benchmark], axis=1, join="inner").dropna()
        if joined.empty:
            return {"correlation": float("nan"), "n_overlap": 0}
        return {
            "correlation": float(joined.iloc[:, 0].corr(joined.iloc[:, 1])),
            "n_overlap": int(joined.shape[0]),
        }

    def summary(self) -> str:
        """One-paragraph human-readable summary."""
        d = self.diagnostics()
        return (
            f"Instrument: {self.name}\n"
            f"  source            : {self.source}\n"
            f"  category          : {self.category}\n"
            f"  frequency         : {self.frequency}\n"
            f"  n_obs             : {d['n_obs']}\n"
            f"  mean (std)        : {d['mean']:+.4f} ({d['std']:.4f})\n"
            f"  date range        : {d['first_date']} → {d['last_date']}\n"
        )
```

- [ ] **Step 4: Confirm green**

Run: `pytest tests/test_instruments/test_core.py -v`
Expected: 8 passed (5 from Task 1 + 3 new).

---

## Task 3: Add `Instrument.to_proxy_svar()`

**Files:**
- Modify: `puremacro/instruments/_core.py`
- Modify: `tests/test_instruments/test_core.py`

`proxy_svar(Y, *, p, horizon, instrument_series, shock_target_idx, n_boot, ci, seed) -> ProxySVARResult` is the existing signature — verified at `puremacro/var/identify/proxy.py:43-53`. The wrapper just hands `self.series.values` as `instrument_series`.

- [ ] **Step 1: Append failing test**

Add to `tests/test_instruments/test_core.py`:

```python
def test_instrument_to_proxy_svar_matches_raw_call():
    """instrument.to_proxy_svar(Y, p=, horizon=) must produce identical output
    to proxy_svar(Y, p=, horizon=, instrument_series=instrument.series.values)."""
    from puremacro.var.identify.proxy import proxy_svar

    rng = np.random.default_rng(0)
    T, n = 60, 2
    Y = rng.standard_normal((T, n)).cumsum(axis=0)
    idx = pd.date_range("2000-01-01", periods=T, freq="QS")
    z_arr = rng.standard_normal(T)
    z_series = pd.Series(z_arr, index=idx)
    inst = Instrument(
        series=z_series, name="test", source="synthetic",
        category="literature", frequency="Q",
    )

    res_via_inst = inst.to_proxy_svar(Y, p=2, horizon=5, n_boot=50, seed=7)
    res_raw = proxy_svar(Y, p=2, horizon=5, instrument_series=z_arr,
                         n_boot=50, seed=7)
    np.testing.assert_array_equal(res_via_inst.irf_point, res_raw.irf_point)
```

- [ ] **Step 2: Confirm failure**

Run: `pytest tests/test_instruments/test_core.py -k to_proxy_svar -v`
Expected: AttributeError on `inst.to_proxy_svar`.

- [ ] **Step 3: Add the method**

Edit `puremacro/instruments/_core.py`. Inside `Instrument`, after `summary()`:

```python
    def to_proxy_svar(self, Y, *, p: int, horizon: int,
                      shock_target_idx: int = 0,
                      n_boot: int = 500, ci: float = 0.9, seed: int = 0):
        """Run :func:`puremacro.var.identify.proxy.proxy_svar` with this
        instrument as the external proxy.

        Returns
        -------
        :class:`puremacro.var.identify._results.ProxySVARResult`
        """
        from ..var.identify.proxy import proxy_svar
        return proxy_svar(
            Y, p=p, horizon=horizon,
            instrument_series=np.asarray(self.series.values, dtype=float),
            shock_target_idx=shock_target_idx,
            n_boot=n_boot, ci=ci, seed=seed,
        )
```

- [ ] **Step 4: Confirm green**

Run: `pytest tests/test_instruments/test_core.py -v`
Expected: 9 passed.

---

## Task 4: Add `Instrument.to_lp_iv()`

**Files:**
- Modify: `puremacro/instruments/_core.py`
- Modify: `tests/test_instruments/test_core.py`

`lp_iv(df, y, x, z, horizons, n_lags, controls, alpha) -> pd.DataFrame` — verified at `puremacro/lp/iv.py:22-31`. Wrapper reindexes `self.series` onto `df.index` as a new column and dispatches.

- [ ] **Step 1: Append failing test**

Add to `tests/test_instruments/test_core.py`:

```python
def test_instrument_to_lp_iv_runs_end_to_end():
    """to_lp_iv must build a DataFrame with the instrument as z and dispatch
    to lp_iv without raising."""
    rng = np.random.default_rng(1)
    T = 120
    idx = pd.date_range("2000-01-01", periods=T, freq="QS")
    z_arr = rng.standard_normal(T)
    x_arr = 0.6 * z_arr + 0.5 * rng.standard_normal(T)
    y_arr = (0.4 * x_arr + 0.3 * rng.standard_normal(T)).cumsum()

    df = pd.DataFrame({"y": y_arr, "x": x_arr}, index=idx)
    z_series = pd.Series(z_arr, index=idx)
    inst = Instrument(
        series=z_series, name="test", source="synthetic",
        category="literature", frequency="Q",
    )

    out = inst.to_lp_iv(df, y="y", x="x", horizons=range(0, 5), n_lags=2)
    assert isinstance(out, pd.DataFrame)
    assert "h" in out.columns
    assert "beta" in out.columns
    assert len(out) == 5
```

- [ ] **Step 2: Confirm failure**

Run: `pytest tests/test_instruments/test_core.py -k to_lp_iv -v`
Expected: AttributeError on `inst.to_lp_iv`.

- [ ] **Step 3: Add the method**

Edit `puremacro/instruments/_core.py`. Inside `Instrument`, after `to_proxy_svar()`:

```python
    def to_lp_iv(self, df: pd.DataFrame, *, y: str, x: str, **kwargs) -> pd.DataFrame:
        """Run :func:`puremacro.lp.iv.lp_iv` with this instrument as ``z``.

        The instrument series is reindexed onto ``df.index`` and added as
        a column with a unique name; this column is then passed as ``z=``
        to ``lp_iv``. Any extra ``kwargs`` are forwarded.
        """
        from ..lp.iv import lp_iv
        z_col = "_instrument_z"
        df2 = df.copy()
        df2[z_col] = self.series.reindex(df2.index)
        return lp_iv(df2, y=y, x=x, z=z_col, **kwargs)
```

- [ ] **Step 4: Confirm green**

Run: `pytest tests/test_instruments/test_core.py -v`
Expected: 10 passed.

---

## Task 5: `NarrativeInstrument.as_instrument()` adapter

**Files:**
- Modify: `puremacro/narrative/types.py` (add method to existing `NarrativeInstrument` class)
- Create: `tests/test_instruments/test_adapters.py`

The adapter wraps `self.quarterly` as an `Instrument`. The category is determined by `self.metadata.get("replication")`: if present, `"narrative_replication"`; else `"narrative_connector"`.

- [ ] **Step 1: Write failing tests**

Create `tests/test_instruments/test_adapters.py`:

```python
"""Round-trip tests for as_instrument() adapters on NarrativeInstrument
and JKResult."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from puremacro.instruments import Instrument, InstrumentLike


# --------------------------------------------------------------------------
# NarrativeInstrument.as_instrument()
# --------------------------------------------------------------------------
def _make_narrative_with_replication_flag():
    from puremacro.narrative import NarrativeEvent, NarrativeInstrument
    events = [
        NarrativeEvent(
            date=pd.Timestamp("2001-01-01"),
            country="USA", magnitude=10.0, magnitude_unit="USD_bn",
            target="investment", subtarget="defense", sign=1,
            confidence=1.0, source_text="t1", source_url="u1",
            scoring_method="manual",
            metadata={"replication": "ramey_2011"},
        ),
        NarrativeEvent(
            date=pd.Timestamp("2002-04-01"),
            country="USA", magnitude=12.0, magnitude_unit="USD_bn",
            target="investment", subtarget="defense", sign=1,
            confidence=1.0, source_text="t2", source_url="u2",
            scoring_method="manual",
            metadata={"replication": "ramey_2011"},
        ),
    ]
    return NarrativeInstrument.from_events(events, target="investment")


def test_narrative_instrument_satisfies_protocol():
    narr = _make_narrative_with_replication_flag()
    assert isinstance(narr, InstrumentLike)


def test_narrative_as_instrument_returns_Instrument():
    narr = _make_narrative_with_replication_flag()
    inst = narr.as_instrument()
    assert isinstance(inst, Instrument)
    assert inst.frequency == "Q"


def test_narrative_as_instrument_preserves_quarterly_series_identity():
    narr = _make_narrative_with_replication_flag()
    inst = narr.as_instrument()
    assert inst.series is narr.quarterly


def test_narrative_as_instrument_picks_connector_category_by_default():
    """When no replication flag is in any event metadata, category is
    'narrative_connector'."""
    from puremacro.narrative import NarrativeEvent, NarrativeInstrument
    events = [
        NarrativeEvent(
            date=pd.Timestamp("2001-01-01"),
            country="USA", magnitude=5.0, magnitude_unit="USD_bn",
            target="investment", subtarget=None, sign=1,
            confidence=1.0, source_text="t", source_url="u",
            scoring_method="keyword",
            metadata={},
        ),
    ]
    narr = NarrativeInstrument.from_events(events)
    inst = narr.as_instrument()
    assert inst.category == "narrative_connector"


def test_narrative_as_instrument_picks_replication_category_when_flagged():
    narr = _make_narrative_with_replication_flag()
    inst = narr.as_instrument()
    assert inst.category == "narrative_replication"


def test_narrative_as_instrument_metadata_includes_n_events():
    narr = _make_narrative_with_replication_flag()
    inst = narr.as_instrument()
    assert inst.metadata.get("n_events") == 2
    assert inst.metadata.get("target") == "investment"
```

- [ ] **Step 2: Confirm failure**

Run: `pytest tests/test_instruments/test_adapters.py -k narrative -v`
Expected: AttributeError — `as_instrument` not yet on `NarrativeInstrument`.

- [ ] **Step 3: Add the method**

Edit `puremacro/narrative/types.py`. Inside the `NarrativeInstrument` class, after `to_frame()`, append:

```python
    def as_instrument(self) -> "Instrument":
        """Wrap as an :class:`puremacro.instruments.Instrument`.

        Category is ``"narrative_replication"`` if any event carries a
        ``"replication"`` key in its metadata, else ``"narrative_connector"``.
        """
        from ..instruments import Instrument
        is_replication = any(
            "replication" in (e.metadata or {}) for e in self.events
        )
        category = "narrative_replication" if is_replication else "narrative_connector"
        return Instrument(
            series=self.quarterly,
            name=self.metadata.get("registry_key", "narrative_instrument"),
            source=self.metadata.get("source", "narrative aggregation"),
            category=category,
            frequency="Q",
            metadata={
                "n_events": len(self.events),
                "target": self.target,
                "aggregation": self.aggregation,
                **self.metadata,
            },
        )
```

- [ ] **Step 4: Confirm green**

Run: `pytest tests/test_instruments/test_adapters.py -k narrative -v`
Expected: 6 passed.

---

## Task 6: `JKResult.as_instrument(*, component, index)` adapter

**Files:**
- Modify: `puremacro/hfi/_results.py`
- Modify: `tests/test_instruments/test_adapters.py`

`JKResult` is `@dataclass(frozen=True)` — adding methods is fine (methods aren't fields). The `index` parameter is required because `JKResult` carries no datetime info (HFI surprises are event-timestamped).

- [ ] **Step 1: Append failing tests**

Add to `tests/test_instruments/test_adapters.py`:

```python
# --------------------------------------------------------------------------
# JKResult.as_instrument()
# --------------------------------------------------------------------------
def _make_jkresult():
    from puremacro.hfi import JKResult
    rng = np.random.default_rng(0)
    n = 30
    return JKResult(
        mp_shock=rng.standard_normal(n),
        info_shock=rng.standard_normal(n),
        rotation=np.eye(2),
        n_admissible=42,
        method="median_target",
    )


def test_jkresult_satisfies_protocol():
    """JKResult.as_instrument() needs the index= kwarg, but the protocol
    check only requires the method to *exist*. We still expect satisfaction."""
    jk = _make_jkresult()
    assert isinstance(jk, InstrumentLike)


def test_jkresult_as_instrument_mp_component():
    jk = _make_jkresult()
    idx = pd.date_range("2000-01-01", periods=30, freq="MS")
    inst = jk.as_instrument(component="mp", index=idx)
    assert isinstance(inst, Instrument)
    assert inst.category == "monetary_hfi"
    assert inst.frequency == "M"
    assert inst.name == "jk2020_mp_shock"
    np.testing.assert_array_equal(inst.series.values, jk.mp_shock)


def test_jkresult_as_instrument_info_component():
    jk = _make_jkresult()
    idx = pd.date_range("2000-01-01", periods=30, freq="MS")
    inst = jk.as_instrument(component="info", index=idx)
    np.testing.assert_array_equal(inst.series.values, jk.info_shock)


def test_jkresult_as_instrument_rejects_bad_component():
    jk = _make_jkresult()
    idx = pd.date_range("2000-01-01", periods=30, freq="MS")
    with pytest.raises(ValueError, match="component"):
        jk.as_instrument(component="bogus", index=idx)


def test_jkresult_as_instrument_rejects_wrong_length_index():
    jk = _make_jkresult()
    bad_idx = pd.date_range("2000-01-01", periods=10, freq="MS")
    with pytest.raises(ValueError, match="length"):
        jk.as_instrument(component="mp", index=bad_idx)


def test_jkresult_as_instrument_method_in_source_string():
    jk = _make_jkresult()
    idx = pd.date_range("2000-01-01", periods=30, freq="MS")
    inst = jk.as_instrument(component="mp", index=idx)
    assert "median_target" in inst.source
```

- [ ] **Step 2: Confirm failure**

Run: `pytest tests/test_instruments/test_adapters.py -k jkresult -v`
Expected: AttributeError — `as_instrument` not yet on `JKResult`.

- [ ] **Step 3: Add the method**

Edit `puremacro/hfi/_results.py`. Inside the `JKResult` dataclass, after the field declarations, add:

```python
    def as_instrument(
        self,
        *,
        component: str = "mp",
        index: pd.DatetimeIndex,
    ):
        """Wrap one component of the decomposition as an
        :class:`puremacro.instruments.Instrument`.

        Parameters
        ----------
        component : ``"mp"`` (default) or ``"info"`` — which shock series.
        index : pd.DatetimeIndex — dates for the shock array (required;
            ``JKResult`` deliberately carries no datetime info).
        """
        from ..instruments import Instrument
        if component not in ("mp", "info"):
            raise ValueError(f"component must be 'mp' or 'info', got {component!r}")
        arr = self.mp_shock if component == "mp" else self.info_shock
        if len(index) != len(arr):
            raise ValueError(
                f"index length {len(index)} does not match shock array length {len(arr)}"
            )
        return Instrument(
            series=pd.Series(arr, index=index, name=f"jk_{component}_shock"),
            name=f"jk2020_{component}_shock",
            source=f"Jarociński-Karadi 2020 {component} component ({self.method})",
            category="monetary_hfi",
            frequency="M",
            metadata={
                "method": self.method,
                "n_admissible": self.n_admissible,
                "rotation": self.rotation,
            },
        )
```

You'll also need to add `import pandas as pd` at the top of the file if not already present. Read the file first to check.

- [ ] **Step 4: Confirm green**

Run: `pytest tests/test_instruments/test_adapters.py -v`
Expected: 12 passed (6 narrative + 6 jkresult).

---

## Task 7: Registry primitives — `InstrumentSpec`, `list_available`, `load`, `describe`

**Files:**
- Create: `puremacro/instruments/_registry.py`
- Create: `tests/test_instruments/test_registry.py`

The registry is a module-level dict `_REGISTRY: dict[str, InstrumentSpec]`. Catalog population (Tasks 8–10) appends to it. This task ships an empty registry + the public functions and tests them with synthetic specs.

- [ ] **Step 1: Write failing tests**

Create `tests/test_instruments/test_registry.py`:

```python
"""Tests for the registry primitives. Catalog discipline tests live in
test_catalog.py."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from puremacro.instruments import (
    Instrument, InstrumentSpec,
    list_available, load, describe, register,
)


def _toy_loader() -> Instrument:
    idx = pd.date_range("2000-01-01", periods=10, freq="QS")
    return Instrument(
        series=pd.Series(np.zeros(10), index=idx),
        name="toy", source="toy synthetic",
        category="literature", frequency="Q",
    )


@pytest.fixture(autouse=True)
def _isolate_registry(monkeypatch):
    """Register a fresh test entry per test, clean up after."""
    from puremacro.instruments import _registry
    saved = dict(_registry._REGISTRY)
    yield
    _registry._REGISTRY.clear()
    _registry._REGISTRY.update(saved)


def test_instrument_spec_is_frozen():
    spec = InstrumentSpec(
        key="toy", name="Toy", category="literature",
        description="syn", reference="N/A",
        loader=_toy_loader, country=None, frequency="Q",
        requires_network=False, requires_fixture=False,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.key = "other"


def test_register_and_load_round_trip():
    spec = InstrumentSpec(
        key="toy", name="Toy", category="literature",
        description="syn", reference="N/A",
        loader=_toy_loader, country=None, frequency="Q",
        requires_network=False, requires_fixture=False,
    )
    register(spec)
    inst = load("toy")
    assert isinstance(inst, Instrument)
    assert inst.name == "toy"


def test_load_missing_key_raises_keyerror_with_help():
    with pytest.raises(KeyError, match="not found"):
        load("definitely_not_in_registry_xyz")


def test_list_available_returns_dataframe_with_documented_columns():
    spec = InstrumentSpec(
        key="toy", name="Toy", category="literature",
        description="syn", reference="N/A",
        loader=_toy_loader, country="USA", frequency="Q",
        requires_network=False, requires_fixture=False,
    )
    register(spec)
    df = list_available()
    assert isinstance(df, pd.DataFrame)
    expected_cols = {
        "key", "name", "category", "country", "frequency",
        "reference", "available", "requires_network", "requires_fixture",
    }
    assert expected_cols <= set(df.columns)
    assert "toy" in df["key"].values


def test_list_available_filters_by_category():
    spec1 = InstrumentSpec(
        key="lit1", name="Lit1", category="literature",
        description="d", reference="r", loader=_toy_loader,
        country=None, frequency="Q",
        requires_network=False, requires_fixture=False,
    )
    spec2 = InstrumentSpec(
        key="narr1", name="Narr1", category="narrative_replication",
        description="d", reference="r", loader=_toy_loader,
        country=None, frequency="Q",
        requires_network=False, requires_fixture=False,
    )
    register(spec1); register(spec2)
    df = list_available(category="literature")
    assert "lit1" in df["key"].values
    assert "narr1" not in df["key"].values


def test_list_available_filters_by_country():
    spec_usa = InstrumentSpec(
        key="usa1", name="USA1", category="literature",
        description="d", reference="r", loader=_toy_loader,
        country="USA", frequency="Q",
        requires_network=False, requires_fixture=False,
    )
    spec_gbr = InstrumentSpec(
        key="gbr1", name="GBR1", category="literature",
        description="d", reference="r", loader=_toy_loader,
        country="GBR", frequency="Q",
        requires_network=False, requires_fixture=False,
    )
    register(spec_usa); register(spec_gbr)
    df = list_available(country="USA")
    assert "usa1" in df["key"].values
    assert "gbr1" not in df["key"].values


def test_list_available_excludes_unavailable_by_default():
    spec_avail = InstrumentSpec(
        key="avail", name="Avail", category="literature",
        description="d", reference="r", loader=_toy_loader,
        country=None, frequency="Q",
        requires_network=False, requires_fixture=False,
    )
    spec_net = InstrumentSpec(
        key="needs_net", name="N", category="literature",
        description="d", reference="r", loader=_toy_loader,
        country=None, frequency="Q",
        requires_network=True, requires_fixture=False,
    )
    register(spec_avail); register(spec_net)
    default = list_available()
    assert "avail" in default["key"].values
    assert "needs_net" not in default["key"].values
    full = list_available(include_unavailable=True)
    assert "needs_net" in full["key"].values


def test_describe_returns_multiline_with_reference():
    spec = InstrumentSpec(
        key="toy", name="Toy", category="literature",
        description="syn description here",
        reference="Author (2020). Journal X 1(1).",
        loader=_toy_loader, country=None, frequency="Q",
        requires_network=False, requires_fixture=False,
    )
    register(spec)
    s = describe("toy")
    assert "syn description here" in s
    assert "Author (2020)" in s
    assert "\n" in s
```

- [ ] **Step 2: Confirm failure**

Run: `pytest tests/test_instruments/test_registry.py -v`
Expected: ImportError on `InstrumentSpec, list_available, load, describe, register`.

- [ ] **Step 3: Create `_registry.py`**

Create `puremacro/instruments/_registry.py`:

```python
"""Self-describing registry of available identified-shock instruments.

The registry is a process-wide dict of :class:`InstrumentSpec` entries
populated by :mod:`._catalog`. Public functions :func:`list_available`,
:func:`load`, and :func:`describe` provide ergonomic access. Use
:func:`register` to add a new spec at runtime (rare — most additions
should be in the catalog file).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from ._core import Instrument, VALID_CATEGORIES


@dataclass(frozen=True)
class InstrumentSpec:
    """Catalog entry describing one identified-shock series.

    Attributes
    ----------
    key : str — unique snake_case identifier.
    name : str — human-readable display name.
    category : str — one of :data:`VALID_CATEGORIES`.
    description : str — one-paragraph what the series represents.
    reference : str — full citation.
    loader : Callable[..., Instrument] — constructs the Instrument.
    country : str | None — ISO3 or None for cross-country.
    frequency : str — pandas-style frequency code: "M" | "Q" | "A".
    requires_network : bool — True if loader needs HTTP.
    requires_fixture : bool — True if loader needs a user-supplied CSV
        (e.g. AB 1991, RR 2004 fed funds shocks).
    """

    key: str
    name: str
    category: str
    description: str
    reference: str
    loader: Callable[..., Instrument]
    country: str | None
    frequency: str
    requires_network: bool
    requires_fixture: bool


_REGISTRY: dict[str, InstrumentSpec] = {}


def register(spec: InstrumentSpec) -> None:
    """Add a spec to the process-wide registry. Overwrites if key exists."""
    if spec.category not in VALID_CATEGORIES:
        raise ValueError(
            f"category {spec.category!r} not in {sorted(VALID_CATEGORIES)}"
        )
    _REGISTRY[spec.key] = spec


def _is_available(spec: InstrumentSpec) -> bool:
    """An entry is 'available' if it needs neither live network nor a
    user-supplied fixture."""
    return not spec.requires_network and not spec.requires_fixture


def list_available(
    *,
    category: str | None = None,
    country: str | None = None,
    include_unavailable: bool = False,
) -> pd.DataFrame:
    """Return a DataFrame of catalogued instruments, one row per spec.

    Columns: ``key``, ``name``, ``category``, ``country``, ``frequency``,
    ``reference``, ``available``, ``requires_network``, ``requires_fixture``.

    Parameters
    ----------
    category : str | None — filter to one category if provided.
    country : str | None — filter to ISO3 country if provided.
    include_unavailable : if False (default), drop entries with
        ``requires_network=True`` or ``requires_fixture=True``.
    """
    rows = []
    for spec in _REGISTRY.values():
        if category is not None and spec.category != category:
            continue
        if country is not None and spec.country != country:
            continue
        avail = _is_available(spec)
        if not include_unavailable and not avail:
            continue
        rows.append({
            "key": spec.key,
            "name": spec.name,
            "category": spec.category,
            "country": spec.country,
            "frequency": spec.frequency,
            "reference": spec.reference,
            "available": avail,
            "requires_network": spec.requires_network,
            "requires_fixture": spec.requires_fixture,
        })
    return pd.DataFrame(rows, columns=[
        "key", "name", "category", "country", "frequency",
        "reference", "available", "requires_network", "requires_fixture",
    ])


def load(key: str, **kwargs: Any) -> Instrument:
    """Construct an Instrument by registry key. Forwards kwargs to the loader."""
    if key not in _REGISTRY:
        raise KeyError(
            f"Instrument key {key!r} not found in registry. "
            f"Use list_available(include_unavailable=True) to see all."
        )
    return _REGISTRY[key].loader(**kwargs)


def describe(key: str) -> str:
    """Return a multi-line human-readable description of the spec at key."""
    if key not in _REGISTRY:
        raise KeyError(f"Instrument key {key!r} not found in registry.")
    s = _REGISTRY[key]
    return (
        f"Instrument: {s.name}  ({s.key})\n"
        f"  category          : {s.category}\n"
        f"  country           : {s.country or '(cross-country)'}\n"
        f"  frequency         : {s.frequency}\n"
        f"  requires_network  : {s.requires_network}\n"
        f"  requires_fixture  : {s.requires_fixture}\n"
        f"  reference         : {s.reference}\n"
        f"  description       : {s.description}\n"
    )


__all__ = [
    "InstrumentSpec", "register",
    "list_available", "load", "describe",
]
```

- [ ] **Step 4: Update package `__init__.py` to re-export registry symbols**

Edit `puremacro/instruments/__init__.py`:

```python
"""Unified Instrument protocol + discovery registry.

See :class:`Instrument` for the canonical wrapper, :class:`InstrumentLike`
for the Protocol, and :func:`list_available` / :func:`load` for the
discovery registry. Spec: ``docs/specs/2026-05-03-instruments-protocol-design.md``.
"""
from ._core import Instrument, InstrumentLike, VALID_CATEGORIES
from ._registry import (
    InstrumentSpec, register,
    list_available, load, describe,
)
from . import _catalog  # noqa: F401  — populates _REGISTRY at import time

__all__ = [
    "Instrument", "InstrumentLike", "VALID_CATEGORIES",
    "InstrumentSpec", "register",
    "list_available", "load", "describe",
]
```

(The `_catalog` import is added now even though Task 8 hasn't created it yet — Step 6 below will create a stub `_catalog.py` with no entries so this import works. Tasks 8–10 then populate it.)

- [ ] **Step 5: Create empty catalog stub**

Create `puremacro/instruments/_catalog.py`:

```python
"""Catalog: declaratively registers Phase-1 instrument entries.

Tasks 8–10 of the implementation plan populate this file. For now it is
empty so the package imports cleanly during Tasks 1–7.
"""
from __future__ import annotations

# Catalog entries are added by direct register(...) calls below.
# Phase-1 scope: see docs/specs/2026-05-03-instruments-protocol-design.md.
```

- [ ] **Step 6: Confirm registry tests green**

Run: `pytest tests/test_instruments/test_registry.py -v`
Expected: 8 passed.

---

## Task 8: Catalog — 6 narrative replication entries

**Files:**
- Modify: `puremacro/instruments/_catalog.py`
- Create: `tests/test_instruments/test_catalog.py`

Each replication has a `load(*, csv_path=None, ...)` function that returns a `NarrativeInstrument` (which has `as_instrument()` from Task 5). The catalog wrapper calls the loader and adapts. Because the loaders need network OR a local CSV, we mark these as `requires_network=True` (default fallback). Users who have local copies pass `csv_path=` through `load()` kwargs.

- [ ] **Step 1: Write failing catalog discipline tests**

Create `tests/test_instruments/test_catalog.py`:

```python
"""Catalog discipline: every Phase-1 entry has the expected shape."""
from __future__ import annotations

import pytest

from puremacro.instruments import list_available, _registry


_EXPECTED_REPLICATION_KEYS = {
    "ramey_2011_defense",
    "romer_romer_2010_fiscal",
    "mertens_ravn_2013_tax",
    "cloyne_2013_uk_tax",
    "romer_romer_2017_fiscal",
    "dglp_2011_consolidations",
}


def test_all_six_replication_entries_registered():
    keys = set(_registry._REGISTRY.keys())
    missing = _EXPECTED_REPLICATION_KEYS - keys
    assert not missing, f"replication entries missing: {missing}"


def test_every_replication_entry_has_non_empty_reference():
    for key in _EXPECTED_REPLICATION_KEYS:
        spec = _registry._REGISTRY[key]
        assert spec.reference and len(spec.reference) > 10, (
            f"{key} has empty/short reference"
        )


def test_every_replication_entry_loader_is_callable():
    for key in _EXPECTED_REPLICATION_KEYS:
        spec = _registry._REGISTRY[key]
        assert callable(spec.loader), f"{key} loader is not callable"


def test_every_replication_entry_country_is_iso3_or_none():
    for key in _EXPECTED_REPLICATION_KEYS:
        spec = _registry._REGISTRY[key]
        assert spec.country is None or (
            isinstance(spec.country, str) and len(spec.country) == 3 and spec.country.isupper()
        ), f"{key} country={spec.country!r} not ISO3 or None"


def test_replication_entries_appear_in_list_available_with_include_flag():
    """They require_network OR require_fixture, so default list_available
    excludes them."""
    df = list_available(include_unavailable=True, category="narrative_replication")
    for key in _EXPECTED_REPLICATION_KEYS:
        assert key in df["key"].values
```

- [ ] **Step 2: Confirm failure**

Run: `pytest tests/test_instruments/test_catalog.py -v`
Expected: AssertionError on the first test — replication entries not registered yet.

- [ ] **Step 3: Populate the 6 replication entries**

Edit `puremacro/instruments/_catalog.py`. Replace contents with:

```python
"""Catalog: declaratively registers Phase-1 instrument entries.

See docs/specs/2026-05-03-instruments-protocol-design.md for scope.
Phase 1: 6 narrative replications + 6 narrative connectors + 1 monetary HFI
+ 12 connector stubs.
"""
from __future__ import annotations

from ._core import Instrument
from ._registry import InstrumentSpec, register


# --------------------------------------------------------------------------
# Helper: wrap a NarrativeInstrument-returning loader as an Instrument loader
# --------------------------------------------------------------------------
def _wrap_narrative(loader_fn, registry_key: str, source: str):
    """Adapter: call loader_fn(**kwargs) → NarrativeInstrument, then
    .as_instrument() and overwrite the name to the registry key."""
    def _load(**kwargs) -> Instrument:
        narr = loader_fn(**kwargs)
        # Inject registry_key so as_instrument()'s name comes through.
        narr.metadata["registry_key"] = registry_key
        narr.metadata["source"] = source
        return narr.as_instrument()
    return _load


# --------------------------------------------------------------------------
# Narrative replications (6)
# --------------------------------------------------------------------------
from ..narrative.replication import (
    load_ramey_2011_defense,
    load_romer_romer_2010,
    load_mertens_ravn_2013,
    load_cloyne_2013_uk,
    load_romer_romer_2017,
    load_dglp_2011,
)


register(InstrumentSpec(
    key="ramey_2011_defense",
    name="Ramey 2011 defense buildup news",
    category="narrative_replication",
    description=(
        "Defense-spending news shocks identified from major military "
        "buildup announcements (Korea, Vietnam, Carter-Reagan, post-9/11)."
    ),
    reference="Ramey, V.A. (2011). Identifying government spending shocks: it's all in the timing. QJE 126(1), 1-50.",
    loader=_wrap_narrative(load_ramey_2011_defense, "ramey_2011_defense",
                           "Ramey 2011 defense buildup events"),
    country="USA",
    frequency="Q",
    requires_network=True,
    requires_fixture=False,
))


register(InstrumentSpec(
    key="romer_romer_2010_fiscal",
    name="Romer-Romer 2010 fiscal shocks",
    category="narrative_replication",
    description=(
        "Exogenous tax-policy changes classified by motivation from "
        "presidential speeches and Congressional records."
    ),
    reference="Romer, C.D. and Romer, D.H. (2010). The macroeconomic effects of tax changes. AER 100(3), 763-801.",
    loader=_wrap_narrative(load_romer_romer_2010, "romer_romer_2010_fiscal",
                           "Romer-Romer 2010 narrative tax shocks"),
    country="USA",
    frequency="Q",
    requires_network=True,
    requires_fixture=False,
))


register(InstrumentSpec(
    key="mertens_ravn_2013_tax",
    name="Mertens-Ravn 2013 tax-rate shocks",
    category="narrative_replication",
    description=(
        "Personal and corporate tax rate change announcements built on "
        "Romer-Romer's narrative coding, used as external instruments."
    ),
    reference="Mertens, K. and Ravn, M.O. (2013). The dynamic effects of personal and corporate income tax changes in the United States. AER 103(4), 1212-1247.",
    loader=_wrap_narrative(load_mertens_ravn_2013, "mertens_ravn_2013_tax",
                           "Mertens-Ravn 2013 narrative tax rate shocks"),
    country="USA",
    frequency="Q",
    requires_network=True,
    requires_fixture=False,
))


register(InstrumentSpec(
    key="cloyne_2013_uk_tax",
    name="Cloyne 2013 UK narrative tax shocks",
    category="narrative_replication",
    description=(
        "UK exogenous tax changes classified by motivation, the British "
        "counterpart to Romer-Romer 2010."
    ),
    reference="Cloyne, J. (2013). Discretionary tax changes and the macroeconomy: new narrative evidence from the United Kingdom. AER 103(4), 1507-1528.",
    loader=_wrap_narrative(load_cloyne_2013_uk, "cloyne_2013_uk_tax",
                           "Cloyne 2013 UK narrative tax shocks"),
    country="GBR",
    frequency="Q",
    requires_network=True,
    requires_fixture=False,
))


register(InstrumentSpec(
    key="romer_romer_2017_fiscal",
    name="Romer-Romer 2017 financial-distress shocks",
    category="narrative_replication",
    description=(
        "Updated narrative tax-shock coding extended to financial-distress "
        "episodes."
    ),
    reference="Romer, C.D. and Romer, D.H. (2017). New evidence on the aftermath of financial crises in advanced countries. AER 107(10), 3072-3118.",
    loader=_wrap_narrative(load_romer_romer_2017, "romer_romer_2017_fiscal",
                           "Romer-Romer 2017 narrative shocks"),
    country=None,
    frequency="Q",
    requires_network=True,
    requires_fixture=False,
))


register(InstrumentSpec(
    key="dglp_2011_consolidations",
    name="Devries-Guajardo-Leigh-Pescatori 2011 consolidations",
    category="narrative_replication",
    description=(
        "OECD-wide narrative coding of fiscal consolidation episodes (17 "
        "advanced economies, 1978-2009)."
    ),
    reference="Devries, P., Guajardo, J., Leigh, D., Pescatori, A. (2011). A new action-based dataset of fiscal consolidations. IMF WP 11/128.",
    loader=_wrap_narrative(load_dglp_2011, "dglp_2011_consolidations",
                           "DGLP 2011 fiscal consolidations"),
    country=None,
    frequency="Q",
    requires_network=True,
    requires_fixture=False,
))
```

- [ ] **Step 4: Confirm catalog tests green**

Run: `pytest tests/test_instruments/test_catalog.py -v`
Expected: 5 passed.

- [ ] **Step 5: Confirm full instruments suite still green**

Run: `pytest tests/test_instruments/ -v`
Expected: 35 passed (10 core + 12 adapters + 8 registry + 5 catalog).

---

## Task 9: Catalog — 6 narrative connector entries

**Files:**
- Modify: `puremacro/instruments/_catalog.py`
- Modify: `tests/test_instruments/test_catalog.py`

Connectors return iterators of `(date, text, link)` tuples — they need to be (a) consumed into `NarrativeEvent` objects via a scoring pass and (b) aggregated into a quarterly series. The cheapest correct loader: thin adapter that takes the connector iterator's output, uses `score_keyword`, aggregates, returns `Instrument`. This is the same recipe used in `narrative.panel.HomogeneousFiscalPanel.add_llm_scored`-style functions.

For Phase 1, we mark these `requires_network=True` (the connector itself fetches HTTP). They're discoverable but not listable by default.

- [ ] **Step 1: Append catalog discipline tests**

Add to `tests/test_instruments/test_catalog.py`:

```python
_EXPECTED_CONNECTOR_KEYS = {
    "us_treasury_press",
    "us_federal_register",
    "us_dod_contracts",
    "oecd_surveys",
    "imf_articleiv",
    "gdelt_v2_news",
}


def test_all_six_connector_entries_registered():
    keys = set(_registry._REGISTRY.keys())
    missing = _EXPECTED_CONNECTOR_KEYS - keys
    assert not missing, f"connector entries missing: {missing}"


def test_every_connector_entry_has_non_empty_reference():
    for key in _EXPECTED_CONNECTOR_KEYS:
        spec = _registry._REGISTRY[key]
        assert spec.reference and len(spec.reference) > 5


def test_every_connector_entry_requires_network_or_fixture():
    for key in _EXPECTED_CONNECTOR_KEYS:
        spec = _registry._REGISTRY[key]
        assert spec.requires_network or spec.requires_fixture, (
            f"{key} should require network OR fixture"
        )
```

- [ ] **Step 2: Confirm failure**

Run: `pytest tests/test_instruments/test_catalog.py::test_all_six_connector_entries_registered -v`
Expected: AssertionError — connector entries missing.

- [ ] **Step 3: Add the 6 connector entries to `_catalog.py`**

Append to `puremacro/instruments/_catalog.py`:

```python
# --------------------------------------------------------------------------
# Helper: wrap a connector iterator as an Instrument loader
# --------------------------------------------------------------------------
def _wrap_connector(iter_fn, *, registry_key: str, source: str,
                    country: str | None, target: str = "both"):
    """Adapter: pull (date, text, link) records from connector, score with
    keyword backend, aggregate to quarterly, return Instrument."""
    def _load(**kwargs) -> Instrument:
        from ..narrative.scoring import score_keyword
        from ..narrative import NarrativeInstrument
        records = list(iter_fn(**kwargs))
        # Convert (date, text, link) into the (text_iter, country=...) form
        # that score_keyword expects. score_keyword yields NarrativeEvents.
        events = list(score_keyword(
            ((date, text, link) for (date, text, link) in records),
            country=country or "USA",
        ))
        narr = NarrativeInstrument.from_events(events, target=target)
        narr.metadata["registry_key"] = registry_key
        narr.metadata["source"] = source
        return narr.as_instrument()
    return _load


# --------------------------------------------------------------------------
# Narrative connectors (6 active)
# --------------------------------------------------------------------------
from ..narrative.sources import (
    us_treasury, us_federal_register, us_dod_contracts,
    oecd_surveys, imf_articleiv, news_api,
)


register(InstrumentSpec(
    key="us_treasury_press",
    name="US Treasury press releases (HTML scrape)",
    category="narrative_connector",
    description="Recent US Treasury press releases scraped from home.treasury.gov/news/press-releases.",
    reference="US Department of the Treasury, public press release listing.",
    loader=_wrap_connector(us_treasury.iter_treasury_press,
                           registry_key="us_treasury_press",
                           source="US Treasury press releases (HTML)",
                           country="USA"),
    country="USA",
    frequency="Q",
    requires_network=True,
    requires_fixture=False,
))


register(InstrumentSpec(
    key="us_federal_register",
    name="US Federal Register documents",
    category="narrative_connector",
    description=(
        "Treasury and DoD presidential documents, rules, and notices from "
        "the Federal Register API."
    ),
    reference="US Government Publishing Office, Federal Register API v1.",
    loader=_wrap_connector(us_federal_register.iter_federal_register,
                           registry_key="us_federal_register",
                           source="US Federal Register API",
                           country="USA"),
    country="USA",
    frequency="Q",
    requires_network=True,
    requires_fixture=False,
))


register(InstrumentSpec(
    key="us_dod_contracts",
    name="US DoD daily contract awards",
    category="narrative_connector",
    description=(
        "Daily defense contract awards (procurement + operating "
        "expenditure) from defense.gov."
    ),
    reference="US Department of Defense, public contracts listing.",
    loader=_wrap_connector(us_dod_contracts.iter_dod_contracts,
                           registry_key="us_dod_contracts",
                           source="US DoD contracts (HTML)",
                           country="USA",
                           target="investment"),
    country="USA",
    frequency="Q",
    requires_network=True,
    requires_fixture=False,
))


register(InstrumentSpec(
    key="oecd_surveys",
    name="OECD Economic Surveys",
    category="narrative_connector",
    description="Country-by-country fiscal-policy assessments from OECD Surveys.",
    reference="OECD Economic Surveys, public PDF/HTML archive.",
    loader=_wrap_connector(oecd_surveys.iter_oecd_surveys,
                           registry_key="oecd_surveys",
                           source="OECD Economic Surveys",
                           country=None),
    country=None,
    frequency="Q",
    requires_network=True,
    requires_fixture=False,
))


register(InstrumentSpec(
    key="imf_articleiv",
    name="IMF Article IV consultations",
    category="narrative_connector",
    description="Member-country fiscal-policy assessments from IMF Article IV consultations.",
    reference="International Monetary Fund, Article IV staff reports.",
    loader=_wrap_connector(imf_articleiv.iter_imf_articleiv,
                           registry_key="imf_articleiv",
                           source="IMF Article IV consultations",
                           country=None),
    country=None,
    frequency="Q",
    requires_network=True,
    requires_fixture=False,
))


register(InstrumentSpec(
    key="gdelt_v2_news",
    name="GDELT v2 fiscal-policy news",
    category="narrative_connector",
    description=(
        "Global news event records from GDELT v2 — free public access, "
        "filtered by fiscal-policy keywords. Module name `news_api.py` "
        "is historical; the actual backend is GDELT."
    ),
    reference="GDELT Project, public v2 events API.",
    loader=_wrap_connector(news_api.iter_gdelt_v2,
                           registry_key="gdelt_v2_news",
                           source="GDELT v2 events",
                           country=None),
    country=None,
    frequency="Q",
    requires_network=True,
    requires_fixture=False,
))
```

**Important note on connector function names:** the actual `iter_*` function names may differ from what's listed above. Verify each before relying on it. If a connector module's function has a different name, update the `_wrap_connector(<actual_name>, ...)` reference. Use `grep -n "^def iter_" puremacro/narrative/sources/*.py` to confirm.

- [ ] **Step 4: Verify connector iter function names match catalog**

Run: `cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro" && grep -n "^def iter_" puremacro/narrative/sources/{us_treasury,us_federal_register,us_dod_contracts,oecd_surveys,imf_articleiv,news_api}.py`

Update the catalog references for any mismatches.

- [ ] **Step 5: Confirm catalog tests green**

Run: `pytest tests/test_instruments/test_catalog.py -v`
Expected: 8 passed (5 from Task 8 + 3 new).

---

## Task 10: Catalog — monetary HFI entry + 12 connector stubs

**Files:**
- Modify: `puremacro/instruments/_catalog.py`
- Modify: `tests/test_instruments/test_catalog.py`

The single monetary HFI entry wraps `gk2015_surprise` (a function that takes announcement-day FFR-futures changes and returns a month-end-adjusted surprise series). Marked `requires_fixture=True` since the user must supply the underlying high-frequency data. The 12 stubs are connector modules without offline tests yet — they appear in `list_available(include_unavailable=True)` only.

- [ ] **Step 1: Append failing tests**

Add to `tests/test_instruments/test_catalog.py`:

```python
def test_monetary_hfi_entry_registered():
    assert "gk2015_ffr_surprise" in _registry._REGISTRY
    spec = _registry._REGISTRY["gk2015_ffr_surprise"]
    assert spec.category == "monetary_hfi"


def test_total_phase1_catalog_size_at_least_25():
    """6 replications + 6 connectors + 1 monetary HFI + 12 stubs = 25."""
    assert len(_registry._REGISTRY) >= 25


def test_stub_entries_appear_in_include_unavailable_listing():
    df = list_available(include_unavailable=True, category="narrative_connector")
    assert len(df) >= 18  # 6 active + 12 stubs


def test_no_two_entries_share_a_key():
    keys = list(_registry._REGISTRY.keys())
    assert len(keys) == len(set(keys))
```

- [ ] **Step 2: Confirm failure**

Run: `pytest tests/test_instruments/test_catalog.py::test_monetary_hfi_entry_registered -v`
Expected: AssertionError — entry missing.

- [ ] **Step 3: Add monetary HFI entry**

Append to `puremacro/instruments/_catalog.py`:

```python
# --------------------------------------------------------------------------
# Monetary HFI (1 active)
# --------------------------------------------------------------------------
def _load_gk2015_ffr_surprise(*, announcement_dates, ffr_futures_changes,
                              freq: str = "M") -> Instrument:
    """Loader for the Gertler-Karadi 2015 month-end-adjusted FFR surprise.

    Parameters
    ----------
    announcement_dates : array-like of pandas Timestamps for FOMC dates.
    ffr_futures_changes : array-like of FFR-futures-implied changes
        (1 per announcement, same length as announcement_dates).
    freq : pandas frequency code for the output series. Default "M".
    """
    from ..hfi import gk2015_surprise, aggregate_to_period
    raw = gk2015_surprise(announcement_dates, ffr_futures_changes)
    series = aggregate_to_period(raw, period=freq)
    return Instrument(
        series=series,
        name="gk2015_ffr_surprise",
        source="Gertler-Karadi 2015 FFR-futures month-end-adjusted surprise",
        category="monetary_hfi",
        frequency=freq,
        metadata={"reference": "Gertler-Karadi 2015"},
    )


register(InstrumentSpec(
    key="gk2015_ffr_surprise",
    name="Gertler-Karadi 2015 FFR surprise",
    category="monetary_hfi",
    description=(
        "Month-end-adjusted FFR-futures monetary policy surprise around "
        "FOMC announcements."
    ),
    reference="Gertler, M. and Karadi, P. (2015). Monetary policy surprises, credit costs, and economic activity. AEJ Macro 7(1), 44-76.",
    loader=_load_gk2015_ffr_surprise,
    country="USA",
    frequency="M",
    requires_network=False,
    requires_fixture=True,
))
```

If `gk2015_surprise` or `aggregate_to_period` accepts different argument names than `(announcement_dates, ffr_futures_changes)`, adjust the loader accordingly. Verify with `grep -A 5 "^def gk2015_surprise\|^def aggregate_to_period" puremacro/hfi/surprises.py` before relying on the names above.

- [ ] **Step 4: Add the 12 connector stubs**

Append to `puremacro/instruments/_catalog.py`:

```python
# --------------------------------------------------------------------------
# Connector stubs (12) — listed for discoverability; not loadable in Phase 1
# --------------------------------------------------------------------------
def _stub_loader(*args, **kwargs) -> Instrument:
    """Stub loader for catalogued-but-not-yet-wrapped connectors."""
    raise NotImplementedError(
        "This connector is catalogued for discoverability only. Phase 1 "
        "ships loaders for the 6 connectors with offline fixtures; this "
        "one will land in a future patch. See "
        "docs/specs/2026-05-03-instruments-protocol-design.md."
    )


_STUB_CONNECTORS = [
    ("uk_obr",      "UK Office for Budget Responsibility forecasts", "GBR"),
    ("uk_hmt",      "UK HM Treasury press notices",                  "GBR"),
    ("de_bmf",      "German Bundesfinanzministerium press",          "DEU"),
    ("fr_tresor",   "French Trésor (DG) press releases",             "FRA"),
    ("it_mef",      "Italian MEF press releases",                    "ITA"),
    ("jp_mof",      "Japanese Ministry of Finance press",            "JPN"),
    ("ca_dof",      "Canadian Department of Finance press",          "CAN"),
    ("ecb_press",   "ECB press releases",                            None),
    ("eu_ecfin",    "European Commission DG ECFIN press",            None),
    ("imf_news",    "IMF news",                                      None),
    ("google_news", "Google News fiscal queries",                    None),
    ("local_csv",   "User-supplied local CSV of events",             None),
]
for stub_key, stub_name, stub_country in _STUB_CONNECTORS:
    register(InstrumentSpec(
        key=stub_key,
        name=stub_name,
        category="narrative_connector",
        description=f"Stub: {stub_name}. Loader to be implemented in a future patch.",
        reference=f"Connector module puremacro.narrative.sources.{stub_key}",
        loader=_stub_loader,
        country=stub_country,
        frequency="Q",
        requires_network=True,
        requires_fixture=False,
    ))
```

- [ ] **Step 5: Confirm full catalog tests green**

Run: `pytest tests/test_instruments/test_catalog.py -v`
Expected: 12 passed (8 from Tasks 8–9 + 4 new).

- [ ] **Step 6: Confirm whole instruments suite green**

Run: `pytest tests/test_instruments/ -v`
Expected: ~42 passed (10 core + 12 adapters + 8 registry + 12 catalog).

---

## Task 11: Wire to public API + snapshot regen + version bump 0.5.0 + CHANGELOG + memory

**Files:**
- Verify: `puremacro/instruments/__init__.py` exports complete (already done in Task 7)
- Modify: `tests/fixtures/public_api_snapshot.json` (regenerate)
- Modify: `pyproject.toml` (`0.4.1 → 0.5.0`)
- Modify: `puremacro/__init__.py` (`__version__`)
- Modify: `tests/test_import.py` (expected version)
- Modify: `CHANGELOG.md` (new `## 0.5.0` block at top)
- Modify: `~/.claude/projects/.../memory/project_puremacro.md` (append iteration entry)

- [ ] **Step 1: Run the full test suite to surface any regressions**

Run: `cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro" && pytest -x -q 2>&1 | tail -10`
Expected: ~429 passed (387 baseline + 42 new instruments tests), 9 skipped.

- [ ] **Step 2: Regenerate the public-API snapshot**

Run from the puremacro directory:

```bash
python -c "from tests.test_public_api import _collect_current_api; \
import json; print(json.dumps(_collect_current_api(), indent=2))" \
> tests/fixtures/public_api_snapshot.json
```

(If that fails because `tests/__init__.py` is absent, fall back to `python -c "import sys; sys.path.insert(0, 'tests'); from test_public_api import _collect_current_api; import json; print(json.dumps(_collect_current_api(), indent=2))" > tests/fixtures/public_api_snapshot.json`.)

- [ ] **Step 3: Confirm snapshot test green**

Run: `pytest tests/test_public_api.py -v`
Expected: PASS.

- [ ] **Step 4: Spot-check the snapshot diff includes the expected new entries**

Run: `grep -c "puremacro.instruments\|Instrument\b\|InstrumentLike\|InstrumentSpec" tests/fixtures/public_api_snapshot.json`
Expected: at least 4 (one for each newly-tracked symbol).

- [ ] **Step 5: Bump pyproject.toml**

Edit `pyproject.toml`. Change `version = "0.4.1"` to `version = "0.5.0"`.

- [ ] **Step 6: Bump `__version__`**

Edit `puremacro/__init__.py`. Change `__version__ = "0.4.1"` to `__version__ = "0.5.0"`.

- [ ] **Step 7: Bump test_import expected version**

Edit `tests/test_import.py`. Change `assert puremacro.__version__ == "0.4.1"` to `assert puremacro.__version__ == "0.5.0"`.

- [ ] **Step 8: Run import test**

Run: `pytest tests/test_import.py -v`
Expected: PASS.

- [ ] **Step 9: Add CHANGELOG entry**

Edit `CHANGELOG.md`. Insert immediately after the file header and before `## 0.4.1 — 2026-05-02`:

```markdown
## 0.5.0 — 2026-05-03

Minor release — new public subpackage `puremacro.instruments` introducing a unified `Instrument` wrapper, an `InstrumentLike` Protocol, and a discovery registry of identified-shock series. No breaking changes: existing `proxy_svar` / `lp_iv` signatures unchanged; `NarrativeInstrument` and `JKResult` gain a single `as_instrument()` method each.

### Added
- **`puremacro.instruments`** — new top-level subpackage.
  - `Instrument` (frozen dataclass) — canonical wrapper for an identified-shock series with provenance metadata. Methods: `to_proxy_svar`, `to_lp_iv`, `diagnostics`, `validate_against`, `summary`.
  - `InstrumentLike` (`@runtime_checkable` Protocol) — single-method protocol any class can satisfy by exposing `as_instrument() -> Instrument`.
  - `InstrumentSpec` (frozen dataclass) — catalog entry describing one shock series (key, category, reference, loader, country, frequency, network/fixture requirements).
  - `list_available()`, `load(key)`, `describe(key)`, `register(spec)` — discovery registry.
  - Phase-1 catalog: 6 narrative replications (Ramey, RR2010, MR2013, Cloyne, RR2017, DGLP), 6 narrative connectors (us_treasury, us_federal_register, us_dod_contracts, oecd_surveys, imf_articleiv, gdelt_v2_news), 1 monetary HFI (Gertler-Karadi 2015 FFR surprise), 12 connector stubs (uk_obr, uk_hmt, de_bmf, fr_tresor, it_mef, jp_mof, ca_dof, ecb_press, eu_ecfin, imf_news, google_news, local_csv).
- `NarrativeInstrument.as_instrument()` — adapter wrapping `self.quarterly` as an `Instrument`. Backwards-compatible.
- `JKResult.as_instrument(*, component, index)` — adapter wrapping one of the two HFI components (`mp_shock` or `info_shock`) as an `Instrument`. Required `index=` kwarg because JKResult deliberately carries no datetime info.

### Internal
- `tests/test_instruments/` — new test directory: 42 tests across protocol, adapters, registry primitives, and catalog discipline.
- `tests/fixtures/public_api_snapshot.json` regenerated to record `puremacro.instruments` and the two new adapter methods.

### Out of scope (deferred to 0.5.1)
- 4 new literature shock loaders (Romer-Romer 2004 monetary, BBD EPU, Caldara-Iacoviello GPR, Bloom 2009 stock-vol uncertainty).
- FRED/BIS/IMF external-CSV loaders.
- `Instrument.compose()` operator.

### Tests
- Pre-release baseline: 387 passing, 9 skipped (0.4.1).
- Post-release: ~429 passing, 9 skipped.
```

- [ ] **Step 10: Append memory entry**

Edit `/Users/jalonso/.claude/projects/-Users-jalonso-Library-CloudStorage-GoogleDrive-jorge-alonsoortiz-gmail-com-My-Drive-MAV/memory/project_puremacro.md`. Append at the end:

```markdown

**Iteration N+9 step 2 done (2026-05-03) — released as 0.5.0 (minor):**
- New top-level subpackage `puremacro.instruments` shipping a unified `Instrument` wrapper + `InstrumentLike` Protocol + self-describing registry of identified-shock series. Solves three friction points at once: type ergonomics (one signature for downstream consumers), API consistency (`NarrativeInstrument` and `JKResult` both expose `.as_instrument()`), and discoverability (`list_available()` returns a DataFrame of catalogued shocks with citations).
- `Instrument` is a frozen dataclass with `series` (date-indexed pd.Series), `name`, `source`, `category` (one of narrative_replication / narrative_connector / monetary_hfi / literature / external_csv), `frequency`, `metadata`. Methods: `to_proxy_svar`, `to_lp_iv`, `diagnostics`, `validate_against`, `summary`.
- `InstrumentLike` is `@runtime_checkable` — single method `as_instrument() -> Instrument`. `NarrativeInstrument` and `JKResult` both satisfy it.
- Adapter on `JKResult`: requires `index=` kwarg because the result carries no datetime info; `component=` selects "mp" or "info".
- Phase-1 catalog (Trim A): 25 entries — 6 narrative replications (Ramey, RR2010, MR2013, Cloyne, RR2017, DGLP), 6 narrative connectors (the same 6 wired up to fixture machinery in 0.4.1), 1 monetary HFI (GK2015), 12 connector stubs for discoverability.
- 42 new tests across `tests/test_instruments/` (10 core + 12 adapters + 8 registry + 12 catalog). Total: 387 → 429 passing, 9 → 9 skipped.
- Plan file: `uncertainty_examples/puremacro/docs/plans/2026-05-03-instruments-protocol-phase1.md`.
- Spec file: `uncertainty_examples/puremacro/docs/specs/2026-05-03-instruments-protocol-design.md`.

**0.5.0 deferred (Trim B / Phase 2 → 0.5.1):**
- 4 new literature shock loaders: Romer-Romer 2004 monetary FFR shocks, Baker-Bloom-Davis EPU, Caldara-Iacoviello GPR, Bloom 2009 stock-vol uncertainty. Each needs either a downloadable CSV or in-package computation from VIX.
- FRED / BIS / IMF external-CSV loaders.
- `Instrument.compose()` operator for adding/concatenating shock series across sources.
- Country-aware automatic filtering inside `proxy_svar`.

**How to apply:** When the user says "discover what shocks are available" or "what instruments do we have?", the answer is `from puremacro.instruments import list_available; list_available()` (or `list_available(include_unavailable=True)` to see stubs). When wiring a new shock-source class to existing puremacro pipelines, the simplest path is: implement `as_instrument()` and you immediately get `to_proxy_svar` / `to_lp_iv` / `diagnostics`.
```

- [ ] **Step 11: Final test run**

Run: `pytest -x -q 2>&1 | tail -5`
Expected: ~429 passed, 9 skipped.

- [ ] **Step 12: Pyodide-compat regression check**

Run: `pytest tests/test_pyodide_compat.py -v`
Expected: PASS.

- [ ] **Step 13: Sanity-check the public surface**

Run: `python -c "from puremacro.instruments import Instrument, InstrumentLike, InstrumentSpec, list_available, load, describe; print('OK')"`
Expected: prints `OK` with no traceback.

Run: `python -c "from puremacro.instruments import list_available; print(list_available(include_unavailable=True).shape)"`
Expected: prints `(25, 9)` (or similar — 25 rows, 9 columns).

---

## Self-Review Checklist

After implementation:

1. **Spec coverage:** Skim each section of `docs/specs/2026-05-03-instruments-protocol-design.md`:
   - [x] `Instrument` frozen dataclass — Task 1–2
   - [x] `InstrumentLike` protocol — Task 1
   - [x] `Instrument.to_proxy_svar` / `to_lp_iv` / `diagnostics` / `validate_against` / `summary` — Tasks 2–4
   - [x] `NarrativeInstrument.as_instrument()` — Task 5
   - [x] `JKResult.as_instrument(*, component, index)` — Task 6
   - [x] `InstrumentSpec` + `register` + `list_available` + `load` + `describe` — Task 7
   - [x] 6 narrative replication catalog entries — Task 8
   - [x] 6 narrative connector catalog entries — Task 9
   - [x] 1 monetary HFI catalog entry + 12 connector stubs — Task 10
   - [x] Public API surface + snapshot + version bump + CHANGELOG + memory — Task 11

2. **Placeholder scan:** No "TBD" / "implement later" / "appropriate error handling" / "similar to Task N" patterns. Every code step shows the actual code. The `_stub_loader` raises `NotImplementedError` — that is the implementation, not a placeholder.

3. **Type consistency:**
   - `Instrument` field order is consistent across the dataclass declaration, the test constructors, and the catalog `_wrap_*` helpers.
   - `InstrumentSpec` field order is consistent across the registration calls, the test fixtures, and the `list_available()` column-order spec.
   - `JKResult.as_instrument(component=, index=)` keyword-only signature is consistent across the test file, the dataclass body addition, and the spec.
   - Catalog category strings (`"narrative_replication"`, `"narrative_connector"`, `"monetary_hfi"`) match `VALID_CATEGORIES` in `_core.py` exactly.
   - The 6 replication keys, 6 connector keys, 1 monetary HFI key, and 12 stub keys total to 25 — matches the test `test_total_phase1_catalog_size_at_least_25`.

4. **Pyodide hygiene:** No new runtime deps. All loaders go through existing `narrative.sources._http` (already Pyodide-correct). The registry is pure-Python data plus pandas DataFrames.

5. **Result-object standard compliance:**
   - `Instrument` and `InstrumentSpec` are both `@dataclass(frozen=True)`.
   - Names end in `Result` for result types — but `Instrument` and `InstrumentSpec` are interface/catalog types, not function returns, so the naming convention does not apply (already explained in the spec).
   - No `__post_init__` (category validation is in `register`, not in the dataclass).
   - `Instrument.summary()` exists; `Instrument.plot()` does not.

6. **Backwards compatibility:** `proxy_svar` and `lp_iv` signatures unchanged. `NarrativeInstrument` keeps all existing methods. `JKResult` adds one method (frozen dataclasses can add methods — methods aren't fields). `puremacro.__init__` unchanged except for the version bump. The 5 dict-return functions migrated in 0.4.1 stay migrated.
