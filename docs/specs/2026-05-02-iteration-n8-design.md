# Iteration N+8 Design — puremacro 0.4.0

**Date:** 2026-05-02
**Status:** Approved (sections 1, 2, 3, 3.5 negotiated interactively; sections C, E, F decided autonomously under auto-mode greenlight)
**Driving lenses:** A (research throughput for MAV pipelines) + B (methodological breadth at the empirical-macro frontier) + E (hardening before piling on more features).

---

## 1. Iteration scope and ordering

Six items, executed in the order below. Feature items gate on prior tests passing. E checkpoints sit between features so polish never starves.

| # | Item | Type | Lens |
|---|------|------|------|
| 0 | Result-object standard added to `ARCHITECTURE.md` (~½ hr, no code) | Doc | E |
| 1 | `puremacro.hfi` (HFI surprises + Jarociński-Karadi 2020 decomposition) | Feature | A+B |
| 2 | `puremacro.cycles.hamilton_filter` | Feature | A |
| 3 | **Combined E checkpoint:** result-object migration sweep + `tests/test_public_api.py` + docstring/type-hint pass on `volatility`/`nowcast`/`gar`/`did` | E | E |
| 4 | `puremacro.dynpanel` (Arellano-Bond + Blundell-Bond GMM) | Feature | B |
| 5 | Narrative connector HTTP fixtures | E | E |
| 6 | DiD completers (de Chaisemartin-D'Haultfœuille 2020 + multi-cohort SDID aggregation) | Feature | B |

**Version target:** `0.4.0` (minor). The result-object migration is technically breaking for any caller that unpacks tuples or reads dict keys today; CHANGELOG entry will be explicit.

**Out of scope (N+9+):** TVP-VAR Bayesian extensions, FAVAR, BCA wedge accounting port, full JK 2020 Bayesian variant, Anderson-Hsiao IV, continuously-updating GMM, Han-Phillips bias correction, Bun-Carree small-N corrections, docs site, performance/benchmark suite.

**Success criteria:**
- All current 237 tests still passing.
- ~70-100 new tests across HFI, cycles, dynpanel, DiD completers, public-API freeze, narrative offline.
- 0 statsmodels/linearmodels/arch in `sys.modules` per `tests/test_pyodide_compat.py`.
- `CHANGELOG.md` 0.4.0 entry; `ARCHITECTURE.md` module map updated; `pyproject.toml` and `puremacro/__init__.py` bumped to `0.4.0`.

---

## 2. Section A — Result-object standard

Established before any new feature ships in 0.4.0.

### Contract

1. **`@dataclass(frozen=True)`** for any return with 3+ fields or carrying non-trivial diagnostics. Frozen prevents accidental mutation post-fit; trivially picklable.
2. **Naming:** `<MethodName>Result` in PascalCase (e.g., `GMMResult`, `IRFResult`, `JKResult`, `CdHResult`, `SDIDMultiResult`). Defined in `<subpackage>/_results.py`; re-exported via `<subpackage>/__init__.py`.
3. **Tuple returns** still allowed for genuinely simple two-value returns (e.g., `cycle, trend = hamilton_filter(y)`). Standard kicks in at 3+ fields or whenever there is a diagnostic to attach.
4. **Common field vocabulary** where applicable: `coefs`, `se`, `cov`, `names: tuple[str, ...]`, `n_obs`, `converged`. Method-specific diagnostics (e.g., `hansen_j` for GMM, `first_stage_F` for IV) live alongside.
5. **`.summary() -> str`** method optional but encouraged. Multi-line pretty-print. Each module writes its own — no autogeneration.
6. **No `.plot()` method.** Plotting stays in `puremacro.plot` and friends; result objects are pure data.
7. **No `__post_init__` validation that raises.** The estimator is responsible for building a valid result; the dataclass just stores it.

### Migration sweep (item #3)

- Inventory every public function in `var/identify/*`, `lp/`, `inference/weak_iv`, `garch/`, `did/*`, `dsge/klein`, `gar/*`, `nowcast/*`, `volatility/*` that returns 3+ fields.
- For each, define a result class in `<subpackage>/_results.py`, update the function to construct it, update tests and examples, expand `__all__`.
- Public-API freeze test (`tests/test_public_api.py`) snapshots both `__all__` per subpackage and result-class field names per dataclass. Failing test prints diff + instructions to regenerate.
- Docstring/type-hint pass focuses on the four newest modules (`volatility`, `nowcast`, `gar`, `did`) — they shipped fastest in 0.3.0 and are likely thinnest. NumPy-style docstrings; Parameters/Returns/Notes sections; type hints on signatures.

**Risk note:** the migration sweep is the biggest unknown of the iteration. Budget 1-1.5 days. If inventory finds more than ~10 affected modules, split the most-callers-affected (probably `var/identify/*`) into its own step.

---

## 3. Section B — `puremacro.hfi`

### File layout

```
puremacro/hfi/
  __init__.py        # public API + __all__
  surprises.py       # GK 2015, NS 2018 surprise construction
  jk2020.py          # Jarociński-Karadi monetary-vs-information decomposition
  _results.py        # JKResult dataclass
  README.md          # quick guide + canonical chain into proxy_svar
```

No `external_iv.py` — composes on top of `var.identify.proxy.proxy_svar`. Implementation step 0 audits `proxy.py` for Olea-Pflueger (2013) effective-F. If missing, it gets added *to `proxy.py`*, not duplicated in `hfi`.

### `surprises.py` — public functions

```python
gk2015_surprise(ff_futures_pre, ff_futures_post, days_remaining_in_month) -> np.ndarray
# Gertler-Karadi 2015 with month-end scaling factor M/(M-d) for FF futures payoff convention.

ns2018_first_pc(surprise_matrix, scale_to="ff_funds_h0") -> tuple[np.ndarray, np.ndarray]
# Nakamura-Steinsson first PC of K policy-sensitive futures' announcement-window changes,
# rescaled so a unit corresponds to 1 pp surprise in the named target contract.
# Returns (pc_series, factor_loadings).

aggregate_to_period(surprises, dates, freq="M") -> pd.Series
# Sum announcement-day surprises into monthly/quarterly bins for merging with macro VARs.
```

### `jk2020.py` — public functions

```python
jk_poor_man(rate_surprise, asset_surprise) -> JKResult
# JK 2020 "simple" version: same-sign (rate, asset) → info shock; opposite-sign → MP shock.

jk_median_target(rate_surprise, asset_surprise, n_rotations=10_000, seed=None) -> JKResult
# JK 2020 preferred spec: median admissible rotation under sign restrictions
#   MP   : rate>0, asset<0
#   Info : rate>0, asset>0
```

### `_results.py`

```python
@dataclass(frozen=True)
class JKResult:
    mp_shock: np.ndarray
    info_shock: np.ndarray
    rotation: np.ndarray | None        # None for poor-man variant
    n_admissible: int | None           # None for poor-man variant
    method: str                         # "poor_man" or "median_target"
    def summary(self) -> str: ...
```

### Tests (`tests/test_hfi/`)

- `test_surprises.py`: GK month-end scaling matches the published formula on synthetic data; NS first PC orthogonal to its residual signal; `aggregate_to_period` sums correctly across month boundaries.
- `test_jk2020.py`: `jk_poor_man` reproduces sign pattern from a JK 2020 Table 2 row; `jk_median_target` returns orthogonal rotation (`U U' = I`); degenerate cases (perfect ±correlation between rate and asset) attribute correctly.
- One end-to-end test that pipes synthetic surprise → `proxy_svar` → IRFs and asserts CI shape.

### Example

`puremacro/examples/hfi_gertler_karadi.py` — fully synthetic monthly panel + synthetic announcement-day surprise series, runs the full chain (surprises → aggregate → `proxy_svar` → IRF + bootstrap CI + Olea-Pflueger F) and plots. No real dataset shipped.

### Pyodide compat

Pure numpy / scipy / pandas. No new dependencies. Module added to the walk in `tests/test_pyodide_compat.py`.

### Deferred

JK 2020 Bayesian sign-restriction variant (median-target only ships in 0.4.0).

---

## 4. Section C — `puremacro.cycles`

New module `puremacro/cycles.py` for time-domain cycle/trend decompositions. `spectral.py` is frequency-domain only (Welch, cross-spectrum, coherence) — wrong home for a regression-based filter.

### Public API

```python
cycle, trend = hamilton_filter(y, h=8, p=4)
# Hamilton 2018: project y_{t+h} on (y_t, ..., y_{t-p+1}); residual is the cycle.
# Defaults h=8, p=4 are quarterly. Two-element tuple stays under the standard.
```

### Tests (`tests/test_cycles.py`)

- Replication of Hamilton 2018 cyclical components on a canonical macro series within tolerance. Ship a small fixture (~150 quarterly observations of US real GDP — public domain via FRED) at `tests/fixtures/us_rgdp_quarterly.csv`.
- Comparison with a no-op HP filter on the same series — agreement on signs of canonical NBER recession dates.

### Future expansion (N+9+)

Christiano-Fitzgerald, Baxter-King, Beveridge-Nelson decomposition all naturally land in `cycles.py`.

---

## 5. Section D — `puremacro.dynpanel`

### File layout

```
puremacro/dynpanel/
  __init__.py
  ab_gmm.py          # Arellano-Bond difference GMM
  bb_gmm.py          # Blundell-Bond system GMM (composes on top of ab_gmm)
  diagnostics.py     # Hansen J, AR(1)/AR(2), Windmeijer SE
  instruments.py     # instrument-matrix construction (collapsed/uncollapsed, lag windows)
  _results.py        # GMMResult dataclass
```

### Public API

```python
from puremacro.dynpanel import ab_gmm, bb_gmm, GMMResult

result = ab_gmm(
    y,                          # (NT,) outcome, long format
    panel_id, time_id,          # (NT,) integer arrays
    lag_dep_var=1,              # number of lagged y on RHS
    X_endog=None,               # (NT, k_e) endogenous regressors
    X_pred=None,                # (NT, k_p) predetermined regressors
    X_exog=None,                # (NT, k_x) strictly exogenous regressors
    gmm_lag_window=(2, None),   # (min, max) lag for endog/pred instruments; None = ∞
    collapse=True,              # Roodman 2009 collapsed instruments (default ON)
    two_step=True,              # two-step optimal weighting (default ON)
    windmeijer=True,            # Windmeijer 2005 SE correction (default ON)
    names=None,                 # optional list[str] for coefficient labels
)

result_bb = bb_gmm(...)         # identical signature; adds level equation + lagged-diff
                                # instruments per Blundell-Bond 1998
```

### `_results.py`

```python
@dataclass(frozen=True)
class GMMResult:
    coefs: np.ndarray
    se: np.ndarray
    cov: np.ndarray
    names: tuple[str, ...]
    hansen_j: float
    hansen_j_p: float
    hansen_j_df: int
    ar1_p: float
    ar2_p: float
    n_instruments: int
    n_obs: int
    n_panels: int
    step: int                   # 1 or 2
    windmeijer: bool
    weights: np.ndarray | None
    converged: bool
    estimator: str              # "ab" or "bb"
    def summary(self) -> str: ...
```

### Design decisions

- **Long format** (NT × variables) with `(panel_id, time_id)` index arrays. Matches how macro panels arrive in pandas; supports unbalanced naturally. Wide-format `(N, T)` converted via small helper if needed.
- **Modern best-practice defaults:** `collapse=True` + `two_step=True` + `windmeijer=True`. Naive AB without these is well-known to be misleading; opt-out is safer than opt-in.
- **Endogenous/predetermined/exogenous distinction** via separate matrices, not a Stata-style string-grammar specification. Numpy-flavoured, consistent with the rest of puremacro.
- **All inversions through `_linalg`.** Two-step weighting and instrument moment matrices blow up under near-collinearity; `inv_xtx` + `safe_cholesky` give named diagnostic errors instead of "Singular matrix".
- **Cluster-robust at panel level is automatic** — moment conditions stack within panel and the optimal weighting matrix permits arbitrary within-panel correlation.

### Tests (`tests/test_dynpanel/`)

- `test_ab_gmm.py`: replication of Arellano-Bond 1991 Table 4 employment results within tolerance. Ship 140-firm × 9-year panel as `tests/fixtures/abdata.csv` (public-domain).
- `test_bb_gmm.py`: replication of the Blundell-Bond 1998 simulation pattern (high-persistence DGP → AB downward biased, BB recovers near-true persistence).
- `test_diagnostics.py`: Hansen J degenerate cases (saturated → df=0); AR(2) on a known AR(1) DGP; Windmeijer correction against closed-form on a small case.
- `test_instruments.py`: `collapse=True` reduces instrument count by the expected factor on a known T; lag-window `(2, 4)` produces correctly windowed instruments.

### In / out of scope

**In:** AB diff-GMM, BB sys-GMM, two-step Windmeijer, Hansen J, AR(1)/AR(2), collapse, lag windows, unbalanced panels.

**Out (N+9 candidates):** Anderson-Hsiao IV, continuously-updating GMM, EL/ETEL alternatives, Han-Phillips bias correction, Bun-Carree small-N.

---

## 6. Section E — Narrative connector HTTP fixtures

### Mechanism

- `tests/_http_fixtures.py` provides an HTTP fixture cache: SHA256 of (URL + sorted headers) → JSON file at `tests/fixtures/http/<sha>.json` containing `{status, body, content_type}`.
- Monkey-patches `puremacro.narrative.sources._http.safe_get_bytes`/`safe_get_text`/`safe_get_json` to read from cache.
- **Record mode** via env var `PUREMACRO_RECORD_HTTP=1`: actual HTTP fires, response written to fixture cache.
- **Replay mode** (default in CI and local pytest): cache miss → test fails with "fixture missing for URL X; rerun with PUREMACRO_RECORD_HTTP=1".

### New test

`tests/test_narrative_offline.py`: runs each connector against fixtures, asserts ≥1 valid event yielded. Covers:
- 9 live sources: `_rss`, `us_treasury`, `us_federal_register`, `us_dod_contracts`, `imf_articleiv`, `oecd_surveys`, `news_api`, `eu_ecfin`.
- 6 replication modules: `dglp`, `ramey`, `romer_romer_2010`, `romer_romer_2017`, `mertens_ravn`, `cloyne`.

Guards against:
- Parser regressions when upstream HTML/JSON shape changes.
- The "yield, don't raise" pattern silently swallowing all results.

### Network tests retention

Existing network tests stay, opt-in via `pytest -m network`. Marker added to `pyproject.toml` `[tool.pytest.ini_options]`.

---

## 7. Section F — DiD completers

### `did/cdh.py` — de Chaisemartin-D'Haultfœuille (2020)

```python
result = cdh_did(y, treatment, panel_id, time_id, placebo=True, n_boot=500, seed=None)
# Returns CdHResult with att_M, att_M_l, se, placebo_p, n_switchers, n_boot, names
```

DID_M (instantaneous) and DID_M^l (long-run) estimators. "Switchers" placebo when `placebo=True`. Unit-resampling bootstrap SEs.

### `did/sdid_multi.py` — Multi-cohort SDID aggregation

```python
result = sdid_multi_cohort(y, treatment, panel_id, time_id, aggregation="att", n_boot=500)
# Returns SDIDMultiResult with att, se, cohort_weights, cohort_atts, names
```

Wraps the single-cohort SDID estimator already in `did/`. Aggregation modes:
- `"att"` — cohort-size-weighted average treatment effect on the treated.
- `"att_g_t"` — full cohort-time grid.

### `_results.py` additions

```python
@dataclass(frozen=True)
class CdHResult:
    att_M: float
    att_M_l: float
    se: tuple[float, float]
    placebo_p: float | None
    n_switchers: int
    n_boot: int
    names: tuple[str, ...]
    def summary(self) -> str: ...

@dataclass(frozen=True)
class SDIDMultiResult:
    att: float
    se: float
    cohort_weights: np.ndarray
    cohort_atts: np.ndarray
    aggregation: str
    n_boot: int
    names: tuple[str, ...]
    def summary(self) -> str: ...
```

### Tests (`tests/test_did/`)

- `test_cdh.py`: directional CdH 2020 result on a known dataset; placebo p-value is uniform under no-effect synthetic DGP.
- `test_sdid_multi.py`: cohort weights sum to 1; on single-cohort input, multi-cohort estimator agrees with single-cohort SDID exactly.

---

## 8. Risks and unknowns

1. **Result-object migration scope** — true cost depends on inventory. Mitigation: dedicated step #3 with explicit budget and split-off plan.
2. **`var/identify/proxy.py` audit** — may turn up gaps (Olea-Pflueger F, weak-IV diagnostics) that are bigger than expected. Mitigation: triage during HFI step 0; if non-trivial, lift into its own sub-step.
3. **Replication tolerances** — Arellano-Bond 1991 and JK 2020 Table 2 use specific data and software; matching their published numbers exactly may be infeasible. Mitigation: tolerances stated per test; "directional agreement + same-sign coefficients within 5%" rather than bit-exact.
4. **Fixture recording cost** — first-time recording of all 15 narrative connectors requires real HTTP and may need rate-limit handling. Mitigation: record incrementally, commit fixtures as each connector lands.

---

## 9. Definition of done

- All items #0-#6 implemented with tests passing.
- `tests/test_public_api.py` snapshot committed; runs green.
- `tests/test_pyodide_compat.py` runs green; new modules included in walk.
- `CHANGELOG.md` 0.4.0 entry written; `ARCHITECTURE.md` updated; `CONTRIBUTING.md` updated if result-object standard changes contribution flow.
- `pyproject.toml` and `puremacro/__init__.py` bumped to `0.4.0`.
- Spec self-review pass (this document).

---

## 10. Next step

Transition to writing-plans skill to produce a detailed implementation plan with subagent-friendly task decomposition.
