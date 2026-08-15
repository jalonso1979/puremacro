# Iteration N+8 Step 2 — `puremacro.cycles.hamilton_filter`

**Goal:** Add Hamilton (2018) regression filter as `puremacro.cycles.hamilton_filter`, retiring the standalone `HamiltonFilter.py` script at the MAV repo root.

**Architecture:** New top-level module `puremacro/cycles.py` (peer of `spectral.py`, `numerics.py`, `realized_vol.py`). `spectral.py` is frequency-domain only; `cycles.py` is the time-domain home for trend-cycle decompositions. Single function for now (`hamilton_filter`); future expansions (Christiano-Fitzgerald, Baxter-King, Beveridge-Nelson) land here.

**Tech Stack:** numpy only. No fixture file ships in step 2 — synthetic tests cover correctness; a real-data fixture is a low-priority follow-up.

**Spec reference:** `docs/specs/2026-05-02-iteration-n8-design.md` § 4 (Section C).

---

## Task 1: Implement `hamilton_filter` with TDD

**Files:**
- Create: `puremacro/cycles.py`
- Create: `tests/test_cycles.py`

### Math

For input series `y` of length T, project `y_{t+h}` on `(1, y_t, y_{t-1}, ..., y_{t-p+1})` via OLS. Cycle = residual; trend = fitted value. Defaults `h=8, p=4` (quarterly: 2-year-ahead projection from most recent 1 year). The first `h + p - 1` output positions are NaN — the regression has no value there.

### Public signature

```python
cycle, trend = hamilton_filter(y, h=8, p=4)
# both arrays have shape (T,); first (h+p-1) entries are NaN.
```

### Tests (in order — TDD)

1. **Constant series → cycle ≈ 0**: input `y = np.ones(50)` should give `cycle ≈ 0` (with NaN prefix).
2. **Linear trend → cycle ≈ 0**: input `y = np.arange(50, dtype=float)` should give `cycle ≈ 0` (the regression fits perfectly).
3. **Pure noise → cycle reproduces innovations approximately**: high-persistence AR(1) input → cycle has lower variance than the input (filter removes persistent component).
4. **Shape correctness**: output length matches input; first `h+p-1` entries are NaN; subsequent entries are finite.
5. **Default kwargs match Hamilton 2018 convention**: `h=8, p=4` produces 12 NaNs at the front.
6. **Custom kwargs**: `h=24, p=12` (monthly) produces 35 NaNs at the front.
7. **Input too short raises**: `T < h + p` raises `ValueError`.
8. **Trend + cycle == y_{t+h}**: for indices where defined, `cycle + trend == y_{h+p-1:}`.

---

## Task 2: CHANGELOG + final verification

- Add `puremacro.cycles.hamilton_filter` to the 0.4.0 in-progress block in `CHANGELOG.md`.
- Run `pytest tests/ -q --tb=short` — confirm green.
- Run `pytest tests/test_pyodide_compat.py -v` — confirm `puremacro.cycles` is auto-discovered.
- Run `python -c "from puremacro.cycles import hamilton_filter; import numpy as np; c, t = hamilton_filter(np.arange(50.0)); print('ok', c.shape, t.shape)"`.
