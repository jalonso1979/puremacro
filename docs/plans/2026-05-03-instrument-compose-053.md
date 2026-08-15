# puremacro 0.5.3 — `Instrument.compose()` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **This workspace is NOT a git repo (Google Drive sync); skip every "commit" step the meta-skill suggests.**

**Goal:** Ship `puremacro.instruments.compose()` — a free function (and matching `Instrument.compose()` method) for combining multiple `Instrument`s via pointwise sum / mean / weighted-mean / chronological concat. Catalog stays at 36 entries; no new categories of data, just a new operation. Released as **0.5.3** (patch — additive, no breaking changes).

**Architecture:** New `puremacro/instruments/_compose.py` module houses the core `compose()` function. `Instrument.compose()` becomes a thin method wrapper that delegates. Adds `"composite"` to `VALID_CATEGORIES` so composed results have an honest provenance category. All operations require matching frequencies across inputs (resampling is the caller's responsibility). Date alignment is configurable (`inner` default, `outer` available). Provenance metadata records source-instrument names, the operation, and weights.

**Tech Stack:** Python 3.10+, `pandas`, `numpy`, existing `puremacro.instruments` (0.5.2). Pyodide-compatible (no new runtime deps).

**Pre-implementation baseline:** 510 passing, 9 skipped (puremacro 0.5.2).
**Post-implementation target:** ~525 passing (+~15 new tests), 9 skipped.

---

## File Structure

### Files created
- `puremacro/instruments/_compose.py` — `compose()` function + private alignment / op helpers
- `tests/test_instruments/test_compose.py` — ~15 tests

### Files modified
- `puremacro/instruments/_core.py` — add `"composite"` to `VALID_CATEGORIES`; add `Instrument.compose()` method that delegates to the free function
- `puremacro/instruments/__init__.py` — re-export `compose`
- `tests/fixtures/public_api_snapshot.json` — regenerate (new `_compose` module + `compose` symbol on top-level + `Instrument.compose` method shape — though methods don't show in snapshot)
- `pyproject.toml` — `version = "0.5.2" → "0.5.3"`
- `puremacro/__init__.py` — `__version__` bump
- `tests/test_import.py` — bump expected version
- `CHANGELOG.md` — add `## 0.5.3 — 2026-05-03` block at top
- `~/.claude/projects/.../memory/project_puremacro.md` — append iteration entry

---

## Task 1: Add `"composite"` category + `compose()` function + tests

**Files:**
- Modify: `puremacro/instruments/_core.py` (single-line change to `VALID_CATEGORIES`)
- Create: `puremacro/instruments/_compose.py`
- Create: `tests/test_instruments/test_compose.py`

The `compose()` function takes a list of `Instrument`s and produces a new `Instrument` whose series is a pointwise combination. Operations:

- `"sum"` — pointwise sum (NaN propagates by default; `skipna=True` ignores NaNs at each timestamp)
- `"mean"` — pointwise arithmetic mean
- `"weighted_mean"` — pointwise weighted mean (requires `weights=` matching `len(instruments)`)
- `"concat"` — chronological union; if multiple inputs have a value at the same timestamp, the LAST instrument in the list wins (explicit "later wins" rule documented)

All inputs must share `.frequency`; mismatch raises `ValueError`. Single-instrument lists return a copy. Empty lists raise `ValueError`. Result `category="composite"`. Result `name`, `source` are caller-supplied or auto-generated. Metadata records source-instrument names, op, weights, and align mode.

- [ ] **Step 1: Update `VALID_CATEGORIES` in `_core.py`**

Edit `puremacro/instruments/_core.py`. Find the `VALID_CATEGORIES` set definition and add `"composite"` to it. Result:

```python
VALID_CATEGORIES = {
    "narrative_replication",
    "narrative_connector",
    "monetary_hfi",
    "literature",
    "external_csv",
    "composite",
}
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_instruments/test_compose.py`:

```python
"""Tests for puremacro.instruments._compose."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from puremacro.instruments import Instrument, compose


def _make(values, idx, name, freq="M"):
    return Instrument(
        series=pd.Series(values, index=idx, name=name),
        name=name,
        source="synthetic",
        category="literature",
        frequency=freq,
    )


def _idx(start, n, freq="MS"):
    return pd.date_range(start, periods=n, freq=freq)


# --------------------------------------------------------------------------
# Sum
# --------------------------------------------------------------------------
def test_compose_sum_two_aligned_instruments():
    idx = _idx("2000-01-01", 3)
    a = _make([1.0, 2.0, 3.0], idx, "a")
    b = _make([10.0, 20.0, 30.0], idx, "b")
    out = compose([a, b], op="sum")
    assert isinstance(out, Instrument)
    assert out.category == "composite"
    assert out.frequency == "M"
    assert list(out.series.values) == [11.0, 22.0, 33.0]


def test_compose_sum_propagates_nan_by_default():
    idx = _idx("2000-01-01", 3)
    a = _make([1.0, np.nan, 3.0], idx, "a")
    b = _make([10.0, 20.0, 30.0], idx, "b")
    out = compose([a, b], op="sum")
    assert pd.isna(out.series.iloc[1])


def test_compose_sum_skipna_true_ignores_nan():
    idx = _idx("2000-01-01", 3)
    a = _make([1.0, np.nan, 3.0], idx, "a")
    b = _make([10.0, 20.0, 30.0], idx, "b")
    out = compose([a, b], op="sum", skipna=True)
    assert out.series.iloc[1] == 20.0  # only b contributes


# --------------------------------------------------------------------------
# Mean
# --------------------------------------------------------------------------
def test_compose_mean_three_instruments():
    idx = _idx("2000-01-01", 2)
    a = _make([1.0, 2.0], idx, "a")
    b = _make([3.0, 4.0], idx, "b")
    c = _make([5.0, 6.0], idx, "c")
    out = compose([a, b, c], op="mean")
    assert list(out.series.values) == [3.0, 4.0]


# --------------------------------------------------------------------------
# Weighted mean
# --------------------------------------------------------------------------
def test_compose_weighted_mean_with_explicit_weights():
    idx = _idx("2000-01-01", 2)
    a = _make([1.0, 2.0], idx, "a")
    b = _make([3.0, 4.0], idx, "b")
    out = compose([a, b], op="weighted_mean", weights=[0.25, 0.75])
    # 0.25*1 + 0.75*3 = 2.5; 0.25*2 + 0.75*4 = 3.5
    assert out.series.iloc[0] == pytest.approx(2.5)
    assert out.series.iloc[1] == pytest.approx(3.5)


def test_compose_weighted_mean_requires_weights():
    idx = _idx("2000-01-01", 2)
    a = _make([1.0, 2.0], idx, "a")
    b = _make([3.0, 4.0], idx, "b")
    with pytest.raises(ValueError, match="weights"):
        compose([a, b], op="weighted_mean")


def test_compose_weighted_mean_wrong_length_weights():
    idx = _idx("2000-01-01", 2)
    a = _make([1.0, 2.0], idx, "a")
    b = _make([3.0, 4.0], idx, "b")
    with pytest.raises(ValueError, match="length"):
        compose([a, b], op="weighted_mean", weights=[0.5, 0.3, 0.2])


# --------------------------------------------------------------------------
# Concat
# --------------------------------------------------------------------------
def test_compose_concat_non_overlapping_union():
    idx_a = _idx("2000-01-01", 2)
    idx_b = _idx("2010-01-01", 2)
    a = _make([1.0, 2.0], idx_a, "a")
    b = _make([10.0, 20.0], idx_b, "b")
    out = compose([a, b], op="concat")
    assert len(out.series) == 4
    assert out.series.loc[pd.Timestamp("2000-01-01")] == 1.0
    assert out.series.loc[pd.Timestamp("2010-01-01")] == 10.0


def test_compose_concat_overlapping_later_wins():
    """When two instruments have a value at the same date, the LAST in
    the input list overwrites the earlier."""
    idx = _idx("2000-01-01", 3)
    a = _make([1.0, 2.0, 3.0], idx, "a")
    b = _make([100.0, 200.0, 300.0], idx, "b")
    out = compose([a, b], op="concat")
    # b is last, so its values win at every overlapping date
    assert list(out.series.values) == [100.0, 200.0, 300.0]


# --------------------------------------------------------------------------
# Frequency / alignment / scalar inputs
# --------------------------------------------------------------------------
def test_compose_mismatched_frequencies_raises():
    idx_m = _idx("2000-01-01", 3, freq="MS")
    idx_q = _idx("2000-01-01", 3, freq="QS")
    a = _make([1.0, 2.0, 3.0], idx_m, "a", freq="M")
    b = _make([1.0, 2.0, 3.0], idx_q, "b", freq="Q")
    with pytest.raises(ValueError, match="frequency"):
        compose([a, b], op="sum")


def test_compose_inner_alignment_default():
    """Default alignment is inner-join: drop dates not present in all."""
    idx_a = pd.date_range("2000-01-01", periods=4, freq="MS")
    idx_b = pd.date_range("2000-02-01", periods=4, freq="MS")
    a = _make([1.0, 2.0, 3.0, 4.0], idx_a, "a")
    b = _make([10.0, 20.0, 30.0, 40.0], idx_b, "b")
    out = compose([a, b], op="sum")
    # Common dates: 2000-02, 2000-03, 2000-04 (three months)
    assert len(out.series) == 3


def test_compose_outer_alignment():
    idx_a = pd.date_range("2000-01-01", periods=2, freq="MS")
    idx_b = pd.date_range("2000-02-01", periods=2, freq="MS")
    a = _make([1.0, 2.0], idx_a, "a")
    b = _make([10.0, 20.0], idx_b, "b")
    out = compose([a, b], op="sum", align="outer")
    # Union: 2000-01, 2000-02, 2000-03 (three dates)
    assert len(out.series) == 3
    # Outer-join introduces NaN where one input is missing; sum propagates
    assert pd.isna(out.series.loc[pd.Timestamp("2000-01-01")])
    assert pd.isna(out.series.loc[pd.Timestamp("2000-03-01")])
    assert out.series.loc[pd.Timestamp("2000-02-01")] == 12.0


def test_compose_empty_list_raises():
    with pytest.raises(ValueError, match="empty"):
        compose([], op="sum")


def test_compose_single_instrument_returns_copy():
    idx = _idx("2000-01-01", 3)
    a = _make([1.0, 2.0, 3.0], idx, "a")
    out = compose([a], op="sum")
    assert isinstance(out, Instrument)
    assert out.category == "composite"
    assert list(out.series.values) == [1.0, 2.0, 3.0]
    # Series is a copy (modifying out.series should not touch a.series; both
    # are pandas Series so we just verify they're different objects).
    assert out.series is not a.series


def test_compose_unknown_op_raises():
    idx = _idx("2000-01-01", 2)
    a = _make([1.0, 2.0], idx, "a")
    with pytest.raises(ValueError, match="op"):
        compose([a], op="not_a_real_op")


# --------------------------------------------------------------------------
# Result Instrument shape
# --------------------------------------------------------------------------
def test_compose_result_metadata_records_provenance():
    idx = _idx("2000-01-01", 2)
    a = _make([1.0, 2.0], idx, "a")
    b = _make([3.0, 4.0], idx, "b")
    out = compose([a, b], op="sum")
    assert out.metadata.get("source_instruments") == ["a", "b"]
    assert out.metadata.get("composition_op") == "sum"
    assert out.metadata.get("composition_weights") is None
    assert out.metadata.get("composition_align") == "inner"


def test_compose_result_uses_caller_supplied_name_and_source():
    idx = _idx("2000-01-01", 2)
    a = _make([1.0, 2.0], idx, "a")
    b = _make([3.0, 4.0], idx, "b")
    out = compose([a, b], op="sum", name="my_composite", source="user demo")
    assert out.name == "my_composite"
    assert out.source == "user demo"


def test_compose_result_auto_generates_name_when_none():
    idx = _idx("2000-01-01", 2)
    a = _make([1.0, 2.0], idx, "a")
    b = _make([3.0, 4.0], idx, "b")
    out = compose([a, b], op="sum")
    assert "compose" in out.name or "sum" in out.name
    assert "a" in out.source or "b" in out.source
```

- [ ] **Step 3: Confirm failure**

Run: `pytest tests/test_instruments/test_compose.py -v`
Expected: ImportError on `from puremacro.instruments import compose` (function not yet defined).

- [ ] **Step 4: Create `_compose.py`**

Create `puremacro/instruments/_compose.py`:

```python
"""Combine multiple :class:`Instrument`s into a single composite series.

Use cases
---------
- Sum monetary, fiscal, and uncertainty proxies into a "macro shock index".
- Average two financial-conditions indices (NFCI + STLFSI4) into one.
- Concatenate Bloom 2009 events with a continuous uncertainty index for a
  longer-history series.

All inputs must share ``.frequency``; resampling is the caller's responsibility.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ._core import Instrument


_VALID_OPS = ("sum", "mean", "weighted_mean", "concat")
_VALID_ALIGN = ("inner", "outer")


def _validate_inputs(
    instruments: list[Instrument],
    op: str,
    weights: list[float] | None,
    align: str,
) -> None:
    if not instruments:
        raise ValueError("compose() received an empty instruments list")
    if op not in _VALID_OPS:
        raise ValueError(f"op {op!r} not in {_VALID_OPS}")
    if align not in _VALID_ALIGN:
        raise ValueError(f"align {align!r} not in {_VALID_ALIGN}")
    freqs = {inst.frequency for inst in instruments}
    if len(freqs) > 1:
        raise ValueError(
            f"compose() requires all instruments to share a frequency; "
            f"got {sorted(freqs)}. Resample first."
        )
    if op == "weighted_mean":
        if weights is None:
            raise ValueError("op='weighted_mean' requires weights= kwarg")
        if len(weights) != len(instruments):
            raise ValueError(
                f"weights length {len(weights)} does not match number of "
                f"instruments {len(instruments)}"
            )


def _align_series(
    instruments: list[Instrument], align: str,
) -> pd.DataFrame:
    """Build a DataFrame with one column per instrument, indexed by the
    chosen alignment of the input series indexes.

    For `inner`, only dates present in all inputs are kept. For `outer`,
    the union is kept and missing values are NaN.
    """
    join = "inner" if align == "inner" else "outer"
    cols = {f"_inst_{i}": inst.series for i, inst in enumerate(instruments)}
    df = pd.concat(cols, axis=1, join=join)
    return df


def _apply_op(
    df: pd.DataFrame,
    op: str,
    weights: list[float] | None,
    skipna: bool,
) -> pd.Series:
    if op == "sum":
        return df.sum(axis=1, skipna=skipna)
    if op == "mean":
        return df.mean(axis=1, skipna=skipna)
    if op == "weighted_mean":
        # Pointwise weighted mean. With skipna=True, dynamically renormalize
        # weights per row to exclude NaN columns.
        w = np.asarray(weights, dtype=float)
        if skipna:
            mask = df.notna().to_numpy()  # (T, n)
            vals = df.fillna(0.0).to_numpy()  # (T, n)
            num = (vals * w).sum(axis=1)
            den = (mask * w).sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                out = np.where(den > 0, num / den, np.nan)
            return pd.Series(out, index=df.index)
        return pd.Series((df.to_numpy() * w).sum(axis=1) / w.sum(),
                         index=df.index)
    if op == "concat":
        # Chronological concat: later instruments overwrite earlier values
        # at the same timestamp. Use a fresh union index then iterate the
        # input columns in order, overwriting.
        union_idx = df.index.sort_values().unique()
        out = pd.Series(np.nan, index=union_idx, dtype=float)
        for col in df.columns:
            non_nan = df[col].dropna()
            out.loc[non_nan.index] = non_nan.values
        return out
    raise ValueError(f"unreachable: op={op!r}")  # validated upstream


def compose(
    instruments: list[Instrument],
    *,
    op: str = "sum",
    weights: list[float] | None = None,
    name: str | None = None,
    source: str | None = None,
    align: str = "inner",
    skipna: bool = False,
) -> Instrument:
    """Combine multiple :class:`Instrument` series into one.

    Parameters
    ----------
    instruments : list of Instrument
        Input series. All must share ``.frequency``.
    op : {"sum", "mean", "weighted_mean", "concat"}, default "sum"
        Combination operation.
    weights : list of float, optional
        Required when ``op="weighted_mean"``; must have the same length
        as ``instruments``. Need not sum to 1 (will be normalized).
    name : str, optional
        Result Instrument name. Auto-generated if None.
    source : str, optional
        Result Instrument source. Auto-generated if None.
    align : {"inner", "outer"}, default "inner"
        Date-index alignment. ``"inner"`` keeps only dates present in
        every input; ``"outer"`` keeps the union and fills NaN where
        missing.
    skipna : bool, default False
        For ``op`` in {sum, mean, weighted_mean}: whether to ignore
        NaN values per timestamp when combining. Has no effect on
        ``op="concat"`` (which always skips NaNs by construction).

    Returns
    -------
    Instrument
        A new Instrument with category ``"composite"`` and metadata
        recording the source instrument names, operation, weights,
        and alignment mode.
    """
    _validate_inputs(instruments, op, weights, align)

    # Single-instrument case: return a copy with composite category.
    if len(instruments) == 1:
        inst = instruments[0]
        return Instrument(
            series=inst.series.copy(),
            name=name or f"compose_{op}_{inst.name}",
            source=source or f"compose({op}: {inst.name})",
            category="composite",
            frequency=inst.frequency,
            metadata={
                "source_instruments": [inst.name],
                "composition_op": op,
                "composition_weights": list(weights) if weights is not None else None,
                "composition_align": align,
            },
        )

    df = _align_series(instruments, align)
    series = _apply_op(df, op, weights, skipna)

    src_names = [inst.name for inst in instruments]
    auto_name = f"compose_{op}_{len(instruments)}_inst"
    auto_source = f"compose({op}: {', '.join(src_names)})"

    return Instrument(
        series=pd.Series(
            series.values,
            index=series.index,
            name=name or auto_name,
        ),
        name=name or auto_name,
        source=source or auto_source,
        category="composite",
        frequency=instruments[0].frequency,
        metadata={
            "source_instruments": src_names,
            "composition_op": op,
            "composition_weights": list(weights) if weights is not None else None,
            "composition_align": align,
        },
    )


__all__ = ["compose"]
```

- [ ] **Step 5: Confirm green**

Run: `pytest tests/test_instruments/test_compose.py -v`
Expected: 17 passed.

---

## Task 2: Add `Instrument.compose()` method

**Files:**
- Modify: `puremacro/instruments/_core.py` (add a method to the `Instrument` dataclass)
- Modify: `tests/test_instruments/test_compose.py` (add 2 method-form tests)

The method form is a thin convenience wrapper. `a.compose(b, c, d, op="sum")` is equivalent to `compose([a, b, c, d], op="sum")`. Useful for inline composition where you have one "base" instrument and want to add others.

- [ ] **Step 1: Append failing tests**

Add to `tests/test_instruments/test_compose.py`:

```python
# --------------------------------------------------------------------------
# Method form: Instrument.compose(*others, **kwargs)
# --------------------------------------------------------------------------
def test_method_compose_matches_function_form():
    idx = _idx("2000-01-01", 3)
    a = _make([1.0, 2.0, 3.0], idx, "a")
    b = _make([10.0, 20.0, 30.0], idx, "b")
    c = _make([100.0, 200.0, 300.0], idx, "c")
    func_result = compose([a, b, c], op="sum")
    method_result = a.compose(b, c, op="sum")
    assert list(func_result.series.values) == list(method_result.series.values)
    assert func_result.metadata == method_result.metadata


def test_method_compose_with_no_others_returns_single_instrument_copy():
    idx = _idx("2000-01-01", 2)
    a = _make([1.0, 2.0], idx, "a")
    out = a.compose(op="sum")
    assert isinstance(out, Instrument)
    assert out.category == "composite"
    assert list(out.series.values) == [1.0, 2.0]
```

- [ ] **Step 2: Confirm failure**

Run: `pytest tests/test_instruments/test_compose.py -k method -v`
Expected: AttributeError on `a.compose` (method not yet on Instrument).

- [ ] **Step 3: Add the `Instrument.compose()` method**

Edit `puremacro/instruments/_core.py`. Inside the `Instrument` dataclass, AFTER `to_lp_iv()`, append:

```python
    def compose(
        self,
        *others: "Instrument",
        op: str = "sum",
        weights: list[float] | None = None,
        name: str | None = None,
        source: str | None = None,
        align: str = "inner",
        skipna: bool = False,
    ) -> "Instrument":
        """Compose this instrument with zero or more others.

        Thin method wrapper for :func:`puremacro.instruments.compose`.
        Equivalent to ``compose([self, *others], op=op, ...)``.
        """
        from ._compose import compose as _compose
        return _compose(
            [self, *others],
            op=op,
            weights=weights,
            name=name,
            source=source,
            align=align,
            skipna=skipna,
        )
```

(Lazy import of `_compose` avoids any potential circular import — `_compose.py` imports `Instrument` from `_core.py`.)

- [ ] **Step 4: Confirm green**

Run: `pytest tests/test_instruments/test_compose.py -v`
Expected: 19 passed (17 from Task 1 + 2 method tests).

---

## Task 3: Public surface + 0.5.3 release coordination

**Files:**
- Modify: `puremacro/instruments/__init__.py` (re-export `compose`)
- Modify: `tests/fixtures/public_api_snapshot.json` (regenerate)
- Modify: `pyproject.toml` (`0.5.2 → 0.5.3`)
- Modify: `puremacro/__init__.py` (`__version__`)
- Modify: `tests/test_import.py` (expected version)
- Modify: `CHANGELOG.md` (new `## 0.5.3` block at top)
- Modify: `~/.claude/projects/.../memory/project_puremacro.md`

- [ ] **Step 1: Re-export `compose` from the package `__init__.py`**

Edit `puremacro/instruments/__init__.py`. Add `from ._compose import compose` after the existing `from ._core import ...` line, and add `"compose"` to `__all__`. Result:

```python
"""Unified Instrument protocol + discovery registry.

See :class:`Instrument` for the canonical wrapper, :class:`InstrumentLike`
for the Protocol, and :func:`list_available` / :func:`load` for the
discovery registry. :func:`compose` combines multiple Instruments into
a composite series. Spec: ``docs/specs/2026-05-03-instruments-protocol-design.md``.
"""
from ._core import Instrument, InstrumentLike, VALID_CATEGORIES
from ._compose import compose
from ._registry import (
    InstrumentSpec, register,
    list_available, load, describe,
)
from . import _catalog  # noqa: F401  — populates _REGISTRY at import time

__all__ = [
    "Instrument", "InstrumentLike", "VALID_CATEGORIES",
    "compose",
    "InstrumentSpec", "register",
    "list_available", "load", "describe",
]
```

- [ ] **Step 2: Run the suite to confirm baseline**

Run: `pytest -x -q tests/ 2>&1 | tail -5`
Expected: 510 + 19 = 529 passed (or close), 9 skipped, EXCEPT `test_public_api.py` may fail on snapshot drift. That's expected; next step regenerates.

- [ ] **Step 3: Regenerate the public-API snapshot**

Run from the puremacro directory:

```bash
python -c "
import sys; sys.path.insert(0, 'tests')
from test_public_api import _collect_current_api
import json
print(json.dumps(_collect_current_api(), indent=2))
" > tests/fixtures/public_api_snapshot.json
```

Then run `pytest tests/test_public_api.py -v`. Expect PASS.

- [ ] **Step 4: Spot-check the snapshot diff**

Run: `grep -c "compose" tests/fixtures/public_api_snapshot.json`
Expected: at least 2 (one for the `_compose` module + one for `compose` in the parent `__all__`).

- [ ] **Step 5: Bump pyproject.toml**

Edit `pyproject.toml`. Change `version = "0.5.2"` to `version = "0.5.3"`.

- [ ] **Step 6: Bump `__version__`**

Edit `puremacro/__init__.py`. Change `__version__ = "0.5.2"` to `__version__ = "0.5.3"`.

- [ ] **Step 7: Bump test_import expected version**

Edit `tests/test_import.py`. Change `assert puremacro.__version__ == "0.5.2"` to `assert puremacro.__version__ == "0.5.3"`.

- [ ] **Step 8: Confirm import test green**

Run: `pytest tests/test_import.py -v`
Expected: PASS.

- [ ] **Step 9: Add CHANGELOG entry**

Edit `CHANGELOG.md`. Insert immediately after the file header and before `## 0.5.2 — 2026-05-03`:

```markdown
## 0.5.3 — 2026-05-03

Patch release — adds `puremacro.instruments.compose()` for combining multiple `Instrument` series into a composite. New `"composite"` category records provenance. Catalog unchanged at 36 entries; this is a runtime composition operation, not new data.

### Added
- `puremacro.instruments.compose(instruments, *, op="sum", weights=None, name=None, source=None, align="inner", skipna=False) -> Instrument` — combine multiple instruments via pointwise sum / mean / weighted-mean / chronological concatenation. All inputs must share `.frequency`; resampling is the caller's responsibility.
  - Operations: `"sum"`, `"mean"`, `"weighted_mean"` (requires `weights=`), `"concat"` (later instrument wins on overlapping dates).
  - Alignment: `"inner"` (default; intersect indices) or `"outer"` (union with NaN fill).
  - `skipna=True` for sum/mean/weighted_mean ignores NaN values per timestamp; weighted_mean dynamically renormalizes weights to exclude NaN columns.
- `Instrument.compose(*others, **kwargs) -> Instrument` — convenience method that delegates to the free function.
- New category `"composite"` added to `VALID_CATEGORIES`. Composed Instruments carry this category and record source-instrument names, operation, weights, and alignment mode in metadata.

### Internal
- New `puremacro/instruments/_compose.py` module (~150 lines).
- `tests/test_instruments/test_compose.py` (new) — 19 tests covering all operations, edge cases (empty list, single instrument, mismatched frequencies, mismatched weight lengths, unknown ops), alignment modes, NaN handling, and the method-form delegation.
- `tests/fixtures/public_api_snapshot.json` regenerated to record `puremacro.instruments._compose` and the `compose` symbol on `puremacro.instruments`.

### Out of scope (still deferred)
- Per-record country threading in `score_keyword` (would let cross-country narrative connectors stamp correct per-event countries).
- JSON serializability of `Instrument.metadata`.
- BIS/IMF SDMX API integration.
- More curated FRED entries for the long tail of macro series.

### Tests
- Pre-release baseline: 510 passing, 9 skipped (0.5.2).
- Post-release: ~529 passing, 9 skipped (+19 new tests).
```

- [ ] **Step 10: Append memory entry**

Edit `/Users/jalonso/.claude/projects/-Users-jalonso-Library-CloudStorage-GoogleDrive-jorge-alonsoortiz-gmail-com-My-Drive-MAV/memory/project_puremacro.md`. Append at the end:

```markdown

**Iteration N+9 step 5 done (2026-05-03) — released as 0.5.3 (patch):**
- New `puremacro.instruments.compose(instruments, *, op, weights, name, source, align, skipna)` function for combining multiple Instruments into a composite series. Operations: `"sum"`, `"mean"`, `"weighted_mean"` (requires `weights=`), `"concat"` (later instrument wins on overlapping dates). All inputs must share `.frequency` (resampling is caller's responsibility). Alignment: `"inner"` (default) or `"outer"`. `skipna` controls NaN handling for sum/mean/weighted_mean; for `weighted_mean` with `skipna=True`, weights are dynamically renormalized to exclude NaN columns per timestamp.
- Method form `Instrument.compose(*others, **kwargs)` — thin wrapper that delegates to the free function. `a.compose(b, c, op="sum")` is equivalent to `compose([a, b, c], op="sum")`.
- New `"composite"` category added to `VALID_CATEGORIES`. Composed Instruments carry this category and record source-instrument names, operation, weights, and alignment mode in `metadata` (keys: `source_instruments`, `composition_op`, `composition_weights`, `composition_align`).
- Catalog unchanged at 36 entries — compose() is a runtime operation, not a new data source.
- 19 new tests in `tests/test_instruments/test_compose.py` covering all 4 operations, both alignment modes, NaN handling for both default and skipna paths, single-instrument copy semantics, edge-case errors (empty list, mismatched frequencies, missing weights, wrong-length weights, unknown op), and method-form delegation parity.
- Plan file: `uncertainty_examples/puremacro/docs/plans/2026-05-03-instrument-compose-053.md`.

**0.5.3 still deferred (Trim B / future patches):**
- Per-record country threading in `score_keyword` — would let cross-country narrative connectors (oecd_surveys, imf_articleiv, gdelt_v2_news) stamp correct per-event countries instead of requiring `country=` at load time. Touches connector wire format (3-tuple → 4-tuple), `score_keyword` signature, and 3 catalog entries; deserves its own release. Likely 0.5.4.
- JSON serializability of `Instrument.metadata` (currently stores ndarray for JKResult.rotation).
- BIS/IMF SDMX API integration.
- More curated FRED entries.

**How to apply:** When the user wants to combine shock series — "sum monetary + fiscal + uncertainty into one VAR proxy", "average two FCI variants", "extend Bloom 2009 events with continuous GPR for longer history" — they can call `from puremacro.instruments import compose; compose([a, b, c], op="sum")` or method form `a.compose(b, c, op="sum")`. Result has category `"composite"` and metadata recording the inputs.
```

- [ ] **Step 11: Final test run**

Run: `pytest -x -q 2>&1 | tail -5`
Expected: ~529 passed, 9 skipped.

- [ ] **Step 12: Pyodide-compat regression check**

Run: `pytest tests/test_pyodide_compat.py -v`
Expected: PASS (no new runtime deps).

- [ ] **Step 13: Sanity-check the public surface**

Run: `python -c "from puremacro.instruments import compose, Instrument; print('OK')"` — expect `OK`.

Run:
```bash
python -c "
import pandas as pd, numpy as np
from puremacro.instruments import Instrument, compose
idx = pd.date_range('2000-01-01', periods=3, freq='MS')
a = Instrument(series=pd.Series([1.0, 2.0, 3.0], index=idx), name='a', source='s', category='literature', frequency='M')
b = Instrument(series=pd.Series([10.0, 20.0, 30.0], index=idx), name='b', source='s', category='literature', frequency='M')
out = compose([a, b], op='sum')
print(f'name: {out.name}')
print(f'category: {out.category}')
print(f'sum: {list(out.series.values)}')
print(f'sources: {out.metadata[\"source_instruments\"]}')
"
```
Expected:
```
name: compose_sum_2_inst
category: composite
sum: [11.0, 22.0, 33.0]
sources: ['a', 'b']
```

Run: `python -c "import puremacro; print('Version:', puremacro.__version__)"` — expect `Version: 0.5.3`.

---

## Self-Review Checklist

After implementation:

1. **Spec coverage:**
   - [x] `"composite"` added to `VALID_CATEGORIES` → Task 1 Step 1
   - [x] `compose()` function with 4 ops + alignment + skipna → Task 1 Steps 2–5
   - [x] `Instrument.compose()` method → Task 2
   - [x] Public re-export → Task 3 Step 1
   - [x] Snapshot regen + version bump + CHANGELOG + memory → Task 3 Steps 3–10

2. **Placeholder scan:** No "TBD", "implement later", "appropriate error handling" patterns. Every code step shows complete code.

3. **Type consistency:**
   - `compose()` signature is identical between `_compose.py` definition, the test invocations, and the docstring.
   - `Instrument.compose()` method signature mirrors the function (minus `instruments` since `self` + `*others` builds the list).
   - `VALID_CATEGORIES` set has exactly 6 strings: `narrative_replication`, `narrative_connector`, `monetary_hfi`, `literature`, `external_csv`, `composite`.
   - Metadata keys are consistent: `source_instruments`, `composition_op`, `composition_weights`, `composition_align`.

4. **Pyodide hygiene:** No new runtime deps. Uses pandas + numpy that the package already requires.

5. **Backwards compatibility:** No changes to existing `Instrument` fields, no changes to `InstrumentLike` Protocol, no changes to existing categories. Adding a new category is additive — old code that didn't know about `"composite"` continues to work because nothing in the existing catalog uses it.

6. **`Instrument.compose()` lazy import**: imports `_compose` inside the method body to avoid the circular import (`_compose.py` imports `Instrument` from `_core.py`). This pattern matches the existing `to_proxy_svar` / `to_lp_iv` lazy imports.

7. **The `name=` parameter** on the result Instrument propagates to BOTH `Instrument.name` AND `Series.name` (which `pd.Series(series.values, index=series.index, name=name)` handles in the multi-instrument branch; the single-instrument branch uses `inst.series.copy()` which preserves the original Series name — flag if test catches this and update).

8. **`weighted_mean` with `skipna=True`** dynamically renormalizes weights per row. This is the mathematically correct behavior (a weighted mean over n-1 columns shouldn't shrink because one is missing) and is documented in the docstring.
