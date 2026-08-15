# puremacro 0.52.0 — climate × fertility primitives

**Status:** draft 2026-05-23. Target release: **0.52.0**.

## Why

The R2 pick from the 2026-05-23 research-directions brainstorm (R4 shipped as
0.51.0). The user already maintains a substantial climate × fertility pipeline at
`My Drive/Fertility/climate_fertility/` (`degree_days.py`, `monthly_regression.py`,
`estimation/{annual_lp.py, mediation.py, monthly_dl.py}`, plus heavy xarray/
geopandas weather and zonal-aggregation infrastructure). That source project is
the canonical full-pipeline implementation for empirical work.

What's missing is a reusable, Pyodide-compatible set of estimator primitives that
(a) other research projects can pull from without dragging in the source's
xarray/geopandas/regional_catastrophes deps, (b) ships under the standard
puremacro release discipline (6 gates), and (c) cleans up the source's
"forward-looking kwargs documented but ignored" wart in `monthly_dl` by
generalising the implementation now.

This release ports four primitives — degree-days, annual panel LP for climate
shocks, within-year-quintile mediation LP, monthly distributed-lag — as a new
`puremacro.climate` subpackage built on top of existing puremacro
infrastructure (`puremacro.lp.panel_lp_dk`, `puremacro.inference._ols_helpers`).

## Scope

One release. New subpackage `puremacro/climate/` with four files:

- `degree_days.py` — CDD/HDD from monthly temperature, monthly + annual.
- `annual_lp.py` — climate-shock annual LP wrapper around `panel_lp_dk`.
- `mediation.py` — within-year-quintile mediation LP.
- `monthly_dl.py` — distributed-lag estimator (single-region + panel) with HC1 SE.

Plus tests, version bump, CHANGELOG, public-API snapshot regeneration.

**Out of scope:** xarray-based raster loaders, geopandas-based zonal aggregation,
country-specific data fetchers, HTTP fetchers for weather or birth registries,
HDDD / extreme-temperature thresholds beyond the simple CDD/HDD cutoff, spatial
autocorrelation diagnostics. These remain in `My Drive/Fertility/climate_fertility/`
or get their own follow-on specs.

## Pre-conditions

- 0.51.0 shipped at tag `v0.51.0` (commit `e569dee`), pushed to
  `origin/feature/subnational-labor-uncertainty-us`.
- 6 release-gate gates green at 0.51.0 HEAD.
- `puremacro.lp.panel_lp_dk(df_wide, y, x, *, horizons, n_lags, controls, alpha,
  entity_level, time_level) -> DataFrame` available, returns columns `[h, beta,
  se, t, lo, hi]`.
- `puremacro.inference._ols_helpers.ols_hac(y, X, lags) -> dict` available.
- Existing source pipeline at `My Drive/Fertility/climate_fertility/` is the
  reference for behavioural intent; we re-derive the math rather than re-importing.

## Architecture

```
puremacro/
  climate/
    __init__.py          — exports + __all__
    degree_days.py       — Component A, ~60 LOC
    annual_lp.py         — Component B, ~80 LOC
    mediation.py         — Component C, ~80 LOC
    monthly_dl.py        — Component D, ~200 LOC
  lp/
    panel_dk.py          — (existing, reused by Component B)
  inference/
    _ols_helpers.py      — (existing, reused by Component D)
```

Each file has a single responsibility; cross-file dependencies are confined to
`mediation.py → annual_lp.py` (mediation runs two annual LPs).

## Component A — degree_days.py

Pure pandas/numpy lift of the source file. No structural changes; signature is
identical to `Fertility/climate_fertility/degree_days.py`.

### Public API

```python
def compute_monthly_cdd_hdd(
    df: pd.DataFrame,
    *,
    temp_col: str = "temp_c",
    threshold: float = 18.0,
) -> pd.DataFrame:
    """Return a copy of df with added columns ``cdd`` and ``hdd``.

    cdd = max(temp_c - threshold, 0)
    hdd = max(threshold - temp_c, 0)

    Raises
    ------
    KeyError
        If ``temp_col`` is not a column of df.
    ValueError
        If ``threshold`` is not finite.
    """


def compute_annual_cdd_hdd(
    df: pd.DataFrame,
    *,
    temp_col: str = "temp_c",
    threshold: float = 18.0,
    region_col: str = "region",
    year_col: str = "year",
    month_col: str = "month",
) -> pd.DataFrame:
    """Aggregate monthly degree-days to annual via sum.

    Returns a DataFrame with columns ``[region_col, year_col, annual_cdd,
    annual_hdd]``.
    """
```

## Component B — annual_lp.py

Annual panel LP for climate shocks. Delegates to `puremacro.lp.panel_lp_dk`
(Driscoll-Kraay HAC SE — the canonical inference for climate-econ panel LPs)
twice, once for CDD and once for HDD.

### Public API

```python
def climate_annual_lp(
    panel: pd.DataFrame,
    *,
    response: str,
    cdd_col: str = "annual_cdd",
    hdd_col: str = "annual_hdd",
    horizons: Iterable[int] = range(0, 11),
    n_lags: int = 2,
    controls: Sequence[str] = (),
    region_col: str = "region",
    year_col: str = "year",
    alpha: float = 0.10,
) -> dict[str, pd.DataFrame]:
    """Run two panel LPs (CDD shock, HDD shock) on the same panel.

    Returns
    -------
    dict
        {'cdd': lp_df, 'hdd': lp_df} where each lp_df is the
        long-form DataFrame returned by ``panel_lp_dk``.

    Notes
    -----
    The panel must be wide-form with one row per (region, year) and contain
    ``response``, ``cdd_col``, ``hdd_col``, and all ``controls``.
    """
```

The wrapper:
1. Calls `panel_lp_dk(panel, y=response, x=cdd_col, controls=[hdd_col] + list(controls), ...)`.
2. Calls `panel_lp_dk(panel, y=response, x=hdd_col, controls=[cdd_col] + list(controls), ...)`.
3. Returns both as a dict.

Putting the other shock in `controls` is intentional: it partials out the joint
HDD/CDD comovement so each reported IRF is the *partial* response to that shock.

## Component C — mediation.py

Within-year-quintile mediation LP. Splits regions into quintiles of a mediator
variable (e.g., housing growth) computed per-year, then runs the climate LP with
top-quintile × shock interactions added as controls.

### Public API

```python
def climate_mediation_lp(
    panel: pd.DataFrame,
    *,
    mediator_col: str,
    response: str,
    cdd_col: str = "annual_cdd",
    hdd_col: str = "annual_hdd",
    horizons: Iterable[int] = range(0, 11),
    n_lags: int = 2,
    region_col: str = "region",
    year_col: str = "year",
    n_bins: int = 5,
    top_quintile_only: bool = True,
) -> dict:
    """Return {'baseline', 'interacted', 'mediation_share_cdd',
    'mediation_share_hdd'}.

    mediation_share_h = (baseline_irf_h - interacted_irf_h) / baseline_irf_h,
    clamped to 0 when |baseline_irf_h| < 1e-12.
    """


def _within_year_quintile(
    panel: pd.DataFrame,
    mediator_col: str,
    region_col: str,
    year_col: str,
    n_bins: int = 5,
) -> pd.Series:
    """Compute quintile bucket of mediator's year-over-year growth per region,
    ranked within each year across regions. Emits a warning when any year
    produces fewer than n_bins buckets due to pd.qcut(duplicates='drop')."""
```

## Component D — monthly_dl.py

Monthly distributed-lag estimator. Generalises the source's `estimate_distributed_lag`
+ `estimate_panel_distributed_lag` into a single function with explicit kwargs
for shock columns, response, FE selection, and panel mode.

This is the design opportunity to fix the source's `Plan 2 — kwargs documented
but ignored` wart by actually wiring the kwargs.

### Public API

```python
def make_dl_lags(
    df: pd.DataFrame,
    *,
    cols: Sequence[str],
    n_lags: int,
    sort_by: Sequence[str],
) -> pd.DataFrame:
    """Add ``{col}_lag1..{col}_lag{n_lags}`` columns. df must be pre-sorted
    by ``sort_by`` (region, year, month) so shifts respect group boundaries.

    Internally calls .groupby(first(sort_by)).shift if len(sort_by) > 1; else
    plain .shift.
    """


def monthly_dl(
    df: pd.DataFrame,
    *,
    shock_cols: Sequence[str] = ("cdd", "hdd"),
    response_col: str = "log_births",
    n_lags: int = 12,
    add_month_fe: bool = True,
    add_year_fe: bool = True,
    region_col: str | None = None,
    panel_fe: str = "region_month",
    month_col: str = "calendar_month",
    year_col: str = "year",
) -> dict:
    """Distributed-lag OLS with HC1 SE.

    Model (single-region, ``region_col=None``):
        y_t = α + Σ_k Σ_s β_k^s · shock_s_{t-k} + month_FE + year_FE + ε_t

    Model (panel, ``region_col`` set):
        y_{r,t} = α_{r,?} + δ_y + Σ_k Σ_s β_k^s · shock_s_{r,t-k} + ε_{r,t}
        where α_{r,?} is region×month FE iff panel_fe='region_month' else region FE.

    Returns
    -------
    dict with keys:
        '{shock}_betas' (length n_lags+1) for each shock_col
        '{shock}_ses'   (length n_lags+1)
        'r_squared' : float
        'n_obs'     : int
        'biological_benchmark' : float
            sum(first shock's betas) — back-compat metric.

    Notes
    -----
    Single-region mode uses HC1 (heteroskedasticity-robust). Panel mode
    clusters by region. For serial-correlation robustness on a single
    region, use the panel variant or a future HAC option.
    """
```

The estimator:
1. Calls `make_dl_lags(df, cols=shock_cols, n_lags=n_lags, sort_by=...)`.
2. Builds design matrix: `[intercept | shocks_contemp | shocks_lags | month_dummies | year_dummies | region_FE]`.
3. OLS via `np.linalg.pinv`.
4. HC1 sandwich (single-region) or cluster-by-region sandwich (panel).
5. Extracts shock coefficients per the column slicing.

## Data flow

```
   external pipeline (xarray + geopandas, stays in Fertility/climate_fertility)
                  │
                  ▼
   panel: [region, year, month, temp_c]
                  │
        compute_monthly_cdd_hdd()
                  │
                  ▼
   panel: [..., cdd, hdd]
                  │
        compute_annual_cdd_hdd()
                  │
                  ▼
   annual panel: [region, year, annual_cdd, annual_hdd]
        + response variable joined in
                  │
                  ├──→ climate_annual_lp()  →  {'cdd': lp_df, 'hdd': lp_df}
                  │
                  └──→ climate_mediation_lp(mediator_col='housing_growth')
                                  │
                                  ▼
                       {'baseline', 'interacted', 'mediation_share_cdd', 'mediation_share_hdd'}

   monthly panel: [region, year, month, log_births, cdd, hdd]
                  │
        monthly_dl(region_col=None|'region')
                  │
                  ▼
   dict: {'cdd_betas', 'hdd_betas', 'cdd_ses', 'hdd_ses', 'r_squared', 'n_obs', 'biological_benchmark'}
```

## Error handling

| Failure mode | Component | Handling |
|---|---|---|
| Required column missing (e.g., no `temp_c`) | all | `KeyError` with clear message naming the expected column |
| `threshold` non-finite | degree_days | `ValueError("threshold must be finite")` |
| Empty df after `dropna` | annual_lp / mediation / monthly_dl | `ValueError("no observations after dropna")` |
| Single observation per region | annual_lp | propagate `panel_lp_dk` error; documented as a precondition |
| Mediator column all-NaN within a year | mediation | `pd.qcut` raises → caught → emit warning, skip that year |
| `n_lags >= T` (any region) | monthly_dl | `ValueError("n_lags={n_lags} exceeds usable T={T} after lag construction")` |
| Singular design (FE consume all DoF) | monthly_dl | `np.linalg.pinv` returns least-norm solution; emit warning; SEs may be zero for absorbed columns |
| `n_bins` too large for region count in a year | mediation | `pd.qcut(..., duplicates='drop')` reduces bins silently; `_within_year_quintile` emits warning when fewer than `n_bins` buckets result |
| Mediation share with near-zero baseline (denominator) | mediation | clamp to 0.0 when `|baseline| < 1e-12` (matches existing logic) |
| Bootstrap NaN in any `panel_lp_dk` horizon | annual_lp | propagate as NaN in the LP DataFrame — caller decides how to handle |

## Testing

Total: ~17 new unit tests across four test files at `tests/test_climate/`.

### `test_degree_days.py` (~4 tests)

- `test_at_threshold_both_zero` — temp_c == threshold → cdd=0, hdd=0.
- `test_cooling_above_threshold` — temp 25, threshold 18 → cdd=7, hdd=0.
- `test_heating_below_threshold` — temp 10, threshold 18 → hdd=8, cdd=0.
- `test_annual_aggregation_sums_monthly` — synthetic 12-month region panel → annual_cdd = sum(monthly cdd), per region.

### `test_climate_annual_lp.py` (~4 tests)

- `test_returns_cdd_and_hdd_keys` — `set(out.keys()) == {'cdd', 'hdd'}`.
- `test_each_lp_dataframe_has_expected_columns` — `{'h', 'beta', 'se', 't', 'lo', 'hi'}.issubset(out['cdd'].columns)`.
- `test_horizon_count_matches_arg` — `len(out['cdd']) == len(list(horizons))`.
- `test_controls_forwarded_to_panel_lp_dk` — providing controls produces different coefficients than the no-controls case (smoke check the kwarg is wired).

### `test_climate_mediation.py` (~4 tests)

- `test_returns_expected_keys` — `{'baseline', 'interacted', 'mediation_share_cdd', 'mediation_share_hdd'}.issubset(out.keys())`.
- `test_within_year_quintile_assigns_5_buckets` — synthetic 25-region panel for 5 years → each year has 5 distinct quintile bins.
- `test_mediation_share_zero_on_zero_baseline` — construct case where baseline_irf is ≈0 → resulting share is 0, not NaN/inf.
- `test_warns_when_mediator_all_nan_in_a_year` — force one year to have all-NaN mediator → `warnings.warn` fires.

### `test_monthly_dl.py` (~5 tests)

- `test_recovers_known_betas_on_synthetic_data` — synthetic DGP with `y_t = 0.05·cdd_t - 0.03·hdd_t + month_fe + year_fe + ε`, T=600 → recovered betas within ±0.01.
- `test_panel_mode_with_region_col` — panel of 4 regions × 600 months → estimator runs with `region_col='region'` and returns dict with same key set; recovers known shock effects across regions.
- `test_n_lags_zero_returns_contemporaneous_only` — `n_lags=0` → `len(cdd_betas) == 1`.
- `test_biological_benchmark_equals_first_shock_sum` — `out['biological_benchmark'] == sum(out['cdd_betas'])`.
- `test_make_dl_lags_creates_correct_columns` — `make_dl_lags(df, cols=['cdd', 'hdd'], n_lags=3, sort_by=['date'])` produces columns `cdd_lag1..3, hdd_lag1..3`.

### Markers

- No `@pytest.mark.slow` on any new test (all <2s individually).
- No new `@pytest.mark.pyodide_smoke` tags initially; the existing 8-test Gate 6 set unchanged.

## Acceptance criteria for 0.52.0

1. `puremacro.climate.compute_monthly_cdd_hdd`, `puremacro.climate.compute_annual_cdd_hdd` exported from `puremacro.climate.__init__`.
2. `puremacro.climate.climate_annual_lp` exported; returns `{'cdd': DataFrame, 'hdd': DataFrame}` with columns `[h, beta, se, t, lo, hi]`.
3. `puremacro.climate.climate_mediation_lp` exported; returns dict `{'baseline', 'interacted', 'mediation_share_cdd', 'mediation_share_hdd'}`.
4. `puremacro.climate.monthly_dl` and `puremacro.climate.make_dl_lags` exported; `monthly_dl` accepts `shock_cols`, `response_col`, `n_lags`, `add_month_fe`, `add_year_fe`, `region_col`, `panel_fe`, `month_col`, `year_col` kwargs and uses each of them meaningfully (no documented-but-ignored kwargs).
5. ~17 new unit tests green under CPython.
6. Public-API snapshot regenerated.
7. All 6 release-gate gates green at HEAD.
8. CHANGELOG 0.52.0 entry. **No breaking changes** — `puremacro.climate` is a brand-new subpackage; no existing puremacro symbols change.
9. Version bumped 0.51.0 → 0.52.0.

## Risks and mitigations

1. **Numerical divergence from the source `climate_fertility` pipeline.**
   Reimplementing in puremacro will produce small numerical differences (different
   OLS solver, different SE convention, different FE construction). *Mitigation:*
   parity tests against the source pipeline are out of scope (would re-import the
   geopandas/xarray deps). The recovery tests use synthetic DGPs with known β
   so the test passes/fails on its own merits, decoupled from source-pipeline output.

2. **`pd.qcut` silently drops bins on small region counts.** With fewer than
   `n_bins` distinct values in a year, `qcut(duplicates='drop')` reduces bin count
   without warning. *Mitigation:* `_within_year_quintile` emits a `warnings.warn`
   when fewer than `n_bins` distinct buckets result for any year.

3. **HC1 SEs are conservative under serial correlation.** Single-region
   `monthly_dl` uses HC1, not HAC. *Mitigation:* document in docstring;
   recommend the panel variant (which clusters by region) when serial correlation
   is suspected. A HAC option could ship in 0.53.0 if needed.

4. **Signature change vs. existing `estimate_distributed_lag`.** The new
   `monthly_dl(df, *, shock_cols, response_col, n_lags, ...)` is not drop-in
   compatible with `estimate_distributed_lag(df, n_lags=12)`. *Mitigation:* clearly
   documented as a new API in CHANGELOG; the source pipeline (`Fertility/
   climate_fertility/`) is unaffected — its consumers don't import from puremacro.
   Users porting to the new API get the kwargs they expected from the source's
   docstring.

5. **Mediation share unstable when baseline IRF crosses zero.** `(baseline -
   controlled) / baseline` blows up near zero. *Mitigation:* clamp `share = 0.0`
   when `|baseline_irf[h]| < 1e-12` (matches existing source behaviour); document
   the caveat in the docstring.

6. **`puremacro.climate` overlapping name with `climate_fertility`.** A future
   reader might confuse `puremacro.climate` (this 4-file extraction) with the
   much-larger source project. *Mitigation:* the `puremacro/climate/__init__.py`
   module docstring explicitly says: "extracted Pyodide-compatible primitives from
   My Drive/Fertility/climate_fertility; the source project remains the canonical
   full-pipeline implementation including data fetch, zonal aggregation, and
   country-specific runners."

## Out of scope (deferred)

- Source-pipeline parity tests (would require importing the source's xarray/
  geopandas pipeline).
- Pyodide-Gate-6 expansion (initial 8-test set unchanged).
- HAC option on the single-region `monthly_dl` (consider for 0.53.0).
- HDDD / extreme-temperature thresholds beyond the simple CDD/HDD cutoff.
- Spatial autocorrelation diagnostics (Moran's I etc.).
- xarray-based raster loaders, geopandas-based zonal aggregation.
- Country-specific runners (france/japan/usa).
- HTTP fetchers for weather (ERA5) or birth registries (CDC, INSEE).
- R1 (fertility DSGE + generic Bayesian engine) — queued as the next research direction.
- R3 (paleoclimate VARX / long-run cliometrics) — third in sequence.
