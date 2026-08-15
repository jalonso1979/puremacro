# Iteration N+8 Step 4 — `puremacro.dynpanel`

**Goal:** Ship dynamic panel GMM (Arellano-Bond 1991 difference GMM, Blundell-Bond 1998 system GMM) with modern best-practice defaults: two-step optimal weighting, Windmeijer (2005) finite-sample SE correction, Roodman (2009) instrument collapse, Hansen J overidentification test, AR(1)/AR(2) Arellano-Bond serial-correlation tests, lag-window control.

**Architecture:** New `puremacro/dynpanel/` subpackage. Long-format input (NT × variables) with `(panel_id, time_id)` index arrays, supports unbalanced panels naturally. Modern defaults ON: `collapse=True`, `two_step=True`, `windmeijer=True`. All inversions through `_linalg.inv_xtx` / `_linalg.safe_cholesky` for the diagnostic-error contract. Result objects per the 0.4.0 standard (`GMMResult` frozen dataclass).

**Tech Stack:** numpy + scipy.stats (chi-square p-values for Hansen J / AR tests), pandas (optional input format). No new runtime dependencies.

**Test strategy:** Tests run on **simulated panels with known DGPs**, NOT on the canonical Arellano-Bond 1991 employment data. Reason: that fixture is not shippable in this environment without network access, and the published numbers are sensitive to small implementation choices (collapse, lag window, two-step). Simulation tests are more robust and validate the mechanics. **Replication of AB 1991 published numbers** is flagged as a follow-up (would land as `tests/test_dynpanel/test_abdata_replication.py` once `tests/fixtures/abdata.csv` is added).

**Spec reference:** `docs/specs/2026-05-02-iteration-n8-design.md` § 5 (Section D).

---

## File layout

```
puremacro/dynpanel/
  __init__.py              # public API + __all__
  _results.py              # GMMResult dataclass
  instruments.py           # instrument-matrix construction (collapsed/uncollapsed, lag windows)
  diagnostics.py           # Hansen J, AR(1)/AR(2), Windmeijer SE correction
  ab_gmm.py                # Arellano-Bond difference GMM
  bb_gmm.py                # Blundell-Bond system GMM (composes on top of ab_gmm)

tests/test_dynpanel/
  __init__.py              # empty
  conftest.py              # shared simulation helpers (panel DGPs)
  test_instruments.py
  test_diagnostics.py
  test_ab_gmm.py
  test_bb_gmm.py
```

---

## Public API (final)

```python
from puremacro.dynpanel import ab_gmm, bb_gmm, GMMResult

result = ab_gmm(
    y,                          # (NT,) outcome, long format
    panel_id, time_id,          # (NT,) integer arrays
    lag_dep_var=1,              # number of lagged y on RHS (typically 1; can be 2+)
    X_endog=None,               # (NT, k_e) endogenous regressors
    X_pred=None,                # (NT, k_p) predetermined regressors
    X_exog=None,                # (NT, k_x) strictly exogenous regressors
    gmm_lag_window=(2, None),   # (min, max) lag for endog/pred GMM-style instruments; None = ∞
    collapse=True,              # Roodman 2009 collapsed instruments (default ON)
    two_step=True,              # two-step optimal weighting (default ON)
    windmeijer=True,            # Windmeijer 2005 SE correction (default ON)
    names=None,                 # optional list[str] for coefficient labels
)

result_bb = bb_gmm(...)         # identical signature; adds level equation + lagged-diff
                                # instruments per Blundell-Bond 1998
```

`GMMResult` (frozen dataclass) defined in `dynpanel/_results.py`:

```python
@dataclass(frozen=True)
class GMMResult:
    coefs: np.ndarray            # (k,)
    se: np.ndarray               # (k,)
    cov: np.ndarray              # (k, k)
    names: tuple[str, ...]
    hansen_j: float
    hansen_j_p: float
    hansen_j_df: int
    ar1_p: float
    ar2_p: float
    n_instruments: int
    n_obs: int                   # total observations used in the moment conditions
    n_panels: int
    step: int                    # 1 or 2
    windmeijer: bool
    estimator: str               # "ab" or "bb"
    converged: bool
    def summary(self) -> str: ...
```

---

## Math reference

### Difference GMM (Arellano-Bond 1991)

Model: `y_{i,t} = ρ y_{i,t-1} + β X_{i,t} + α_i + ε_{i,t}`.

First-difference: `Δy_{i,t} = ρ Δy_{i,t-1} + β ΔX_{i,t} + Δε_{i,t}` (drops `α_i`).

`Δy_{i,t-1}` is correlated with `Δε_{i,t}` (both contain `ε_{i,t-1}`), so we use **lagged levels** as instruments:

- For an endogenous regressor `w` (incl. lagged `y`): use `w_{i,t-2}, w_{i,t-3}, ..., w_{i,t-L}` to instrument `Δw_{i,t}`. Lag window `(2, L)`.
- For a predetermined regressor: lag window `(1, L)`.
- For a strictly exogenous regressor: itself is the instrument (any lag).

Stack across t and i to get instrument matrix `Z`. Moment conditions: `E[Z' Δε] = 0`.

Two-step estimator:
1. **Step 1**: weight matrix `W_1 = (Z' H Z)^{-1}` where `H` is the difference operator's first-difference covariance (T-1 banded with 2 on diagonal, -1 on off-diagonal). Solve `β̂_1 = (X' Z W_1 Z' X)^{-1} X' Z W_1 Z' y`.
2. **Step 2**: weight matrix `W_2 = (Z' Δε̂_1 Δε̂_1' Z)^{-1}` (cluster on panel within Z). Solve again with `W_2`.

**Windmeijer (2005) finite-sample correction** for two-step SE: standard sandwich SE underestimates because `W_2` depends on `β̂_1`. Correction adds a Jacobian term involving `∂W_2/∂β`. See Windmeijer 2005 eq. (2.7).

**Hansen J overidentification test**: `J = N · (Z' Δε̂)' W_2 (Z' Δε̂) ~ χ²(df)` where `df = #instruments - #regressors`.

**AR(1) / AR(2) Arellano-Bond tests**: under correct specification, `Δε_{i,t} = ε_{i,t} - ε_{i,t-1}` is AR(1) by construction (negative lag-1 autocovariance) but should NOT have AR(2). The AR(m) test statistic:

`m_test = (Σ_i Σ_t Δε̂_{i,t} Δε̂_{i,t-m}) / sqrt(V_m) ~ N(0,1)`

where `V_m` is the panel-clustered variance of the numerator. Implementation: see Roodman 2009 eq. (10).

### System GMM (Blundell-Bond 1998)

Stacks two equations: difference equation (as above) AND level equation `y_{i,t} = ρ y_{i,t-1} + β X_{i,t} + α_i + ε_{i,t}`.

Additional moment for level equation: `E[Δw_{i,t-1} (α_i + ε_{i,t})] = 0` for endogenous `w`, requires `α_i` uncorrelated with `Δw`. The level instruments are **lagged differences**: `Δw_{i,t-1}` instruments for `w` in the level equation.

Same two-step + Windmeijer + Hansen + AR tests apply.

### Roodman (2009) instrument collapse

Without collapse: `#instruments` grows as `O(T²)`. Collapse stacks all lags of a single regressor into a single column per lag, reducing to `O(T)`. Implementation: for endog/pred regressor `w` with lag window `(L_min, L_max)`, build `len(window) * 1` columns instead of `len(window) * (T - p_panel - L_min)`.

---

## Implementation strategy notes

1. **Long format input**: convert `(y, panel_id, time_id)` arrays into a balanced/unbalanced 2D structure as needed. Unbalanced panels are routine in macro work; do not require balance.

2. **Build instrument matrix `Z`** in `instruments.py`:
   - For each panel `i` and each time `t` in the regression sample, build the row of `Z` from lag windows.
   - `collapse=True` reduces #columns; `collapse=False` keeps the full set.
   - Strictly-exogenous columns appear as themselves; predetermined / endogenous use lag windows.

3. **First-difference operator `H`**: pure numpy; banded matrix computed once per panel block.

4. **Weight matrices**: route `(Z' Z)`, `(Z' H Z)`, `(Z' Δε̂ Δε̂' Z)` inversions through `_linalg.inv_xtx`. The two-step weight is the canonical place where rank deficiency appears (too many instruments, T small) — `inv_xtx` will surface a named diagnostic instead of silent garbage.

5. **Windmeijer correction**: numerical Jacobian via finite differences on `β̂_1` is acceptable; closed-form is involved. Use a small step size (`1e-5`) and forward differences. Document this choice — analytic Jacobian is a future optimization.

6. **AR(1)/AR(2) variance**: panel-clustered. Use a vectorized loop over panels.

7. **Cluster-robust SE at panel level is automatic**: the moment conditions stack within panel; `W_2` permits arbitrary within-panel correlation. No separate `cluster=` flag.

8. **Edge cases to handle**:
   - Empty `X_endog` / `X_pred` / `X_exog` (only lagged y on RHS) — must work.
   - Unbalanced panels with gaps in `time_id` — must skip the gap rows.
   - Degenerate panel (all NaN, or only 1 time period) — drop with informative warning.
   - All-zero residuals (perfect fit) — `W_2` is singular; raise from `inv_xtx`.

---

## Tests (in TDD order)

### `test_instruments.py`

- `test_uncollapsed_instrument_count` — for known T, p, window, instrument count matches the formula.
- `test_collapsed_reduces_instrument_count` — collapsed has fewer columns by the expected factor.
- `test_lag_window_truncation` — `gmm_lag_window=(2, 3)` produces only 2 columns per regressor (lags 2 and 3).
- `test_unbalanced_panel_skips_gaps` — a panel with a missing time period produces zero rows for the missing slot, not garbage.
- `test_strictly_exogenous_columns_passthrough` — `X_exog` columns appear in `Z` as themselves (no lag wrap).

### `test_diagnostics.py`

- `test_windmeijer_correction_increases_se_in_two_step` — for a known DGP, two-step SE without Windmeijer is too small; with Windmeijer, SE is closer to bootstrap truth.
- `test_hansen_j_under_correct_model_has_uniform_p` — repeated draws from a correctly-specified DGP, p-values uniform on [0, 1].
- `test_ar1_p_low_for_difference_residuals_under_iid` — AR(1) test rejects (low p) because differencing iid creates AR(1) structure.
- `test_ar2_p_high_for_difference_residuals_under_iid` — AR(2) test does NOT reject under iid (the well-specified case).
- `test_ar2_p_low_when_serial_correlation_present` — when true `ε` has AR(1) serial correlation, AR(2) test rejects.

### `test_ab_gmm.py`

- `test_ab_recovers_persistence_low_persistence` — DGP `ρ=0.3`, T=8, N=200 — AB recovers `ρ̂` within 0.1 of truth.
- `test_ab_returns_GMMResult` — isinstance check + frozen.
- `test_ab_summary_runs` — `.summary()` returns a non-empty string mentioning estimator name.
- `test_ab_high_persistence_downward_biased` — DGP `ρ=0.95`, T=8, N=200 — AB known-biased downward (this is the canonical motivation for BB). `ρ̂_AB < 0.85` expected.
- `test_ab_lag_window_reduces_instruments` — `gmm_lag_window=(2, 3)` produces fewer instruments than `(2, None)`.
- `test_ab_with_exogenous_regressor` — DGP includes a strictly-exogenous regressor; AB recovers its coefficient within tolerance.
- `test_ab_unbalanced_panel` — drop ~10% of cells at random; estimator runs and recovers truth.

### `test_bb_gmm.py`

- `test_bb_recovers_persistence_high_persistence` — DGP `ρ=0.95` (where AB fails); BB recovers within 0.05 of truth.
- `test_bb_returns_GMMResult_estimator_label` — `result.estimator == "bb"`.
- `test_bb_more_instruments_than_ab` — BB has additional level-equation moments → more instruments than AB on the same panel.
- `test_bb_with_predetermined_regressor` — DGP includes a predetermined regressor; BB recovers within tolerance.
- `test_bb_summary_runs`.

---

## CHANGELOG entry (to be added in Task C)

Under existing `## 0.4.0 (in progress)` block:

```markdown
- **`puremacro.dynpanel`** — Dynamic panel GMM.
  - `ab_gmm` — Arellano-Bond (1991) difference GMM with two-step optimal weighting.
  - `bb_gmm` — Blundell-Bond (1998) system GMM (composes on top of ab_gmm).
  - Modern best-practice defaults ON: `collapse=True` (Roodman 2009), `two_step=True`, `windmeijer=True` (Windmeijer 2005 finite-sample SE correction).
  - Hansen J overidentification test, Arellano-Bond AR(1)/AR(2) serial-correlation tests, lag-window control.
  - Long-format input `(y, panel_id, time_id)`; supports unbalanced panels.
  - Cluster-robust at panel level by construction.
  - Endogenous / predetermined / strictly-exogenous regressors via separate matrices (numpy-flavoured, not Stata-style string grammars).
  - New `GMMResult` frozen dataclass.
- AB 1991 employment-data replication test (`tests/fixtures/abdata.csv` + `tests/test_dynpanel/test_abdata_replication.py`) DEFERRED — fixture file not shippable in this environment. Simulation tests cover correctness.
```

---

## ARCHITECTURE.md updates (Task C)

In the module map ASCII, add a `dynpanel/` line under the existing entries (between `did/` and `examples/`):

```
├── dynpanel/            ← Dynamic panel GMM: Arellano-Bond + Blundell-Bond (0.4.0)
```

In the stability tiers table, add a row:

| `dynpanel/{ab_gmm, bb_gmm, instruments, diagnostics}` | **Stable** | Two-step Windmeijer + Hansen J + AR(1)/AR(2) + Roodman collapse + lag windows. Sim-tested against canonical DGPs; AB 1991 published-numbers replication deferred to a fixture-shipping patch. |

---

## Out-of-scope (deferred to N+9 or later)

- Anderson-Hsiao IV (older, dominated by AB).
- Continuously-updating GMM.
- Empirical-likelihood / ETEL alternatives.
- Han-Phillips bias-corrected estimator.
- Bun-Carree small-N corrections.
- AB 1991 employment data fixture and exact-numbers replication (would land as a separate patch with the CSV).
