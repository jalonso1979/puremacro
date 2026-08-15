# Notebook 29 — State-Panel LP: National LUI Shock → State Labor Outcomes

**Status:** Drafted 2026-05-10. Triggered by Slice 6a (v0.8.0) lifting LUI vs urate ρ to +0.331, breaking through the +0.30 acceptance threshold and unblocking the dissertation chapter's headline empirical result.

**Driving lens:** Identify how US states respond to national labor-uncertainty shocks. The setup naturally separates national-level identification (LUI is constructed from federal Fed text) from state-level outcomes (urate, employment, participation). Cross-sectional dispersion in state responses identifies which labor markets are most exposed.

## Motivation

LUI is now a research-usable measure of national US labor uncertainty. The natural next step: use it as a shock in a state-level panel local projection (Jordà 2005). National text-based identification + sub-national outcomes is a clean, defensible setup: state-level idiosyncratic shocks don't drive national Fed text, so LUI is approximately exogenous to any single state's labor market.

Three outcomes capture different margins:
- **State unemployment rate** — direct labor-market response, most interpretable, default headline.
- **State nonfarm employment growth (Δ₄ log NFP)** — flow margin, reflects hiring/separation balance.
- **State labor force participation rate** — discouraged-worker margin; thinner data (annual CPS).

## Non-goals

- **No** LP-IV in v1. Pure LP rests on "national LUI is approximately exogenous to state-level idiosyncratic shocks" — credible because the index is built from federal Fed text, not state-level news. LP-IV with EPU or news-based instruments is a future extension.
- **No** continuous-time / VAR / FAVAR setups. LP is the simplest credible identification.
- **No** structural decomposition (sign restrictions, narrative IV beyond LUI). LP is reduced-form.
- **No** state-level VAR. Pure panel LP.
- **No** real-time forecasting application — historical IRFs only.

## Architecture

### Data layer (new)

**`puremacro/fetch/bls_state_panel.py`** — three fetchers wrapping the BLS public API:

1. `iter_state_urate_q()` — quarterly seasonally-adjusted state unemployment rate from LAUS. Series ID pattern: `LASST{STATE_FIPS}0000000000003`. Output panel records: `(state_code, qdate, urate, source_url, metadata)`.
2. `iter_state_employment_q()` — quarterly seasonally-adjusted state nonfarm employment from CES (state). Series ID: `SMU{state}{area}_{industry}{datatype}`. Output: `(state_code, qdate, log_emp, …)` with 4-quarter log growth derived in the notebook.
3. `iter_state_participation_a()` — **annual** state labor force participation rate from CPS state estimates. Quarterly state-level CPS is too thin (small samples). Notebook 29 quarterly-interpolates via constant-by-year for joint estimation with the other two outcomes. Limitation documented.

Coverage: 50 states + DC = 51 cross-sectional units. Sample: 2006Q1 onward (LUI start). Caching: per-series parquet files in `notebooks/data_cache/`.

Endpoint: `https://api.bls.gov/publicAPI/v2/timeseries/data/` (no key for ≤25 series; user can supply BLS_API_KEY for ≤500/day).

### Estimation layer (new)

**`puremacro/regress/lp.py`** — generic panel local projection estimator.

```python
def lp_panel(
    panel: pd.DataFrame,            # cols: 'unit', 'date', y, shock, controls...
    y: str,                         # outcome column name
    shock: str,                     # shock column (national, broadcast across units)
    horizons: range = range(0, 13),
    unit_fe: bool = True,
    controls: list[str] | None = None,
    se: str = "driscoll_kraay",     # or "two_way_cluster"
    dk_lag: int | None = None,      # auto-pick = h + 1 if None
    dummies: list[str] | None = None,
) -> pd.DataFrame:
    """Returns long-format DataFrame: horizon, beta, se, t, p, ci_lo, ci_hi, n_obs."""
```

Pooled OLS with unit FE; Driscoll-Kraay (1998) SEs handle cross-sectional dependence + serial correlation from the LP overlapping-window construction. Pure numpy implementation, pyodide-clean (no statsmodels/linearmodels at top level).

### Notebook 29 (new)

**`notebooks/29_state_panel_lp_lui.ipynb`**, paired with **`tools/make_notebook_29_state_panel_lp.py`** (per the notebooks↔builders convention).

Cell sequence:
1. Setup + imports.
2. Load LUI from `notebooks/output_tables/28_lui_us_quarterly.parquet`.
3. Construct LUI shock: AR(4) residual on quarterly LUI series; standardize to unit variance.
4. Fetch + cache state panels (urate Q, employment Q, participation A→Q).
5. Build merged panel: 51 states × 80 quarters. Add national controls (FFR, CPI, real GDP — quarterly), 4 lags. Add COVID dummies (2020Q2–2021Q4).
6. Run `lp_panel()` for each outcome × horizon ∈ [0, 12]. Pooled IRF plus per-state β.
7. **Plot pooled IRFs** (β_h ± 90% CI band) — 3-panel grid (urate / Δ₄log NFP / lfpr).
8. **Forest plot** per state at h=8 cumulative response — one figure per outcome.
9. **Heat map**: state × horizon, sorted by h=8 response magnitude — one per outcome.
10. **Heterogeneity splits**: manufacturing employment share, BA-or-higher attainment share. Per-split pooled IRFs.
11. **Robustness**: AR(2) shock; h=20 horizon; subsample pre-2020 vs 2020+.

### Acceptance criteria

- **Pooled IRF for state urate**: positive and significant at h ∈ [2, 6] (1 std LUI shock raises state urate by ~10–30 bps at peak). Sign + significance test passes.
- **Pooled IRF for state employment growth**: negative and significant at similar horizons.
- **Pooled IRF for state participation**: negative at intermediate horizons (discouraged-worker effect); may be smaller in magnitude given annual data.
- **Heterogeneity**: states with higher manufacturing share show ≥ 1.5× larger urate response than low-manufacturing states (cyclical-sensitivity literature).
- **Notebook re-runs end-to-end** without manual intervention; outputs reproducible bit-for-bit on a clean cache.

If sign reverses on the urate panel: STOP, investigate (likely shock standardization or control confound).

## Components

| File | Change |
|---|---|
| `puremacro/fetch/bls_state_panel.py` | New: 3 BLS state-panel fetchers + per-series cache. |
| `puremacro/regress/__init__.py` | New module package. |
| `puremacro/regress/lp.py` | New: panel LP estimator with Driscoll-Kraay SE; pure-numpy. |
| `puremacro/tests/test_fetch_bls_state.py` | New: offline mock tests against canned BLS JSON; one `@pytest.mark.network` live probe. |
| `puremacro/tests/test_regress_lp.py` | New: LP unit tests on synthetic panels with known DGP. |
| `notebooks/29_state_panel_lp_lui.ipynb` | New notebook. |
| `tools/make_notebook_29_state_panel_lp.py` | New builder; pair with the .ipynb. |
| `notebooks/output_tables/29_*.parquet`, `29_meta.json` | New outputs. |
| `notebooks/output_figures/29_*.pdf` | New figures. |
| `puremacro/CHANGELOG.md`, `pyproject.toml`, `__init__.py`, `tests/test_import.py` | 0.8.0 → 0.9.0 (new fetch + regress submodules). |

## Failure handling

| Failure | Behavior |
|---|---|
| BLS API rate limit (HTTP 429) | Exponential backoff; aggressive caching; suggest user set `BLS_API_KEY` env var. |
| Missing CPS state quarterly data | Spec already accounts: annual data, quarterly-interpolated by constant-by-year. Documented limitation in notebook. |
| LP IRF reverses sign vs theory (urate decreases on positive LUI shock) | STOP. Investigate likely causes: shock standardization, control confound, sample-period issue. |
| Driscoll-Kraay SEs blow up at long horizons | Switch to two-way clustered (state × time) at h ≥ 8 if observed. |
| State coverage gaps (e.g., DC quarterly LFPR has thin sample) | Drop affected states from heterogeneity splits with note in notebook. |
| Heterogeneity split shows insignificant difference | Document as null result; do not over-interpret. |

## Testing strategy

- **`bls_state_panel.py`** — offline mock tests against canned BLS API JSON for at least 3 states; one live `@pytest.mark.network` test.
- **`regress/lp.py`** — synthetic panel tests:
  - Simulate `y_{i,t} = α_i + β · shock_t + e_{i,t}` with known β; recover β within ±0.05 std error.
  - Driscoll-Kraay SE compared to a manual reference computation on a small (5×20) panel.
  - Edge cases: unbalanced panels, all-zero shock, single horizon.
- **Notebook 29** — end-to-end re-run as integration test; commit outputs.
- **Pyodide compat** — confirm new code adds zero new top-level deps.

## Out of scope (deferred)

- LP-IV with EPU / news-based / Greenbook-IV instruments.
- State-level VAR or FAVAR.
- Cross-country panel (other countries' state/region panels).
- Real-time forecasting application of the IRF.
- Slice 6b items (LLM kernel, Picault-Renault paragraph MNL, full Hubert lexicon, BIS speeches connector).
- Bayesian LP (Plagborg-Møller-Wolf).
