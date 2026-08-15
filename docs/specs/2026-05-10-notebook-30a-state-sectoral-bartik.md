# Notebook 30a — State-Level Sectoral Bartik: LUI × Industry-Composition Exposure

**Status:** Drafted 2026-05-10. Follows Notebook 29 (v0.9.1) — state-panel LP of US labor outcomes on national LUI shocks. Notebook 29 found a significant urate IRF and a persistent-but-marginal log NFP IRF. Notebook 30a explores **whether cross-state heterogeneity in industry composition explains the differential response**.

**Driving lens:** Build a Bartik (shift-share) exposure measure for each US state based on its industry composition × national industry-level LUI sensitivity. Interact this exposure with the LUI shock in the state-panel LP. States more exposed to LUI-sensitive sectors (manufacturing, construction, leisure) should respond more strongly than states dominated by less-sensitive sectors (government, education/health).

## Motivation

Notebook 29 found a pooled urate IRF of +0.466 pp at h=5 (significant). The mfg-share heterogeneity split in 30a's parent slice was a crude single-cut split. A proper Bartik approach:

1. **National-level structural step.** Estimate each 2-digit NAICS supersector's national employment sensitivity to LUI via per-industry LP regressions. Call this `β_k^national`.
2. **State-level exposure step.** Each state's Bartik exposure = Σ_k (share_{state,k,baseline} × |β_k^national|). High-mfg states get high exposure if mfg has a high |β_k^national|.
3. **Identification step.** Interact `shock_t × exposure_state` in the state-panel LP. The interaction's coefficient measures how much more responsive high-exposure states are.

This is **plain interacted LP** (not GPSS 2SLS Bartik). It doesn't require industry-share exogeneity. Interpretable as: "states more exposed to LUI-sensitive industries respond more."

## Non-goals

- **No** county-level analysis in v1. Notebook 30b (next slice) extends to ~3,140 counties.
- **No** formal Goldsmith-Pinkham-Sorkin-Swift 2SLS Bartik IV. Plain interacted LP is the v1 target.
- **No** demographic exposure (BA-share, age, race). Sectoral only in v1. Demographics in Notebook 30b or 30c.
- **No** state-by-industry quarterly time series (51 × 11 = 561 series). Use BEA SAEMP25N annual at baseline (2005) — sufficient for a fixed-share shift-share.
- **No** time-varying Bartik exposure. Single baseline year (2005Q1) per the standard "fixed shift-share" convention.
- **No** structural model. LP only, reduced-form.

## Architecture

### Data layer (new fetchers)

**`puremacro/fetch/state_industry_panel.py`** (new module):

1. `iter_national_industry_emp_q(supersectors=None)` — quarterly national 2-digit NAICS employment from FRED. Default 11 supersectors: Manufacturing, Construction, FIRE, Information, Trade-Transport-Utilities, Government, PBS, Education/Health, Leisure/Hospitality, Other Services, Mining/Logging. Series IDs: `MANEMP`, `USCONS`, `USFIRE`, `USINFO`, `USTRADE`, `USGOVT`, `USPBS`, `USEHS`, `USLAH`, `USSRVO`, `USMINE` (verified IDs at implementation time). Output records: `(industry_code, qdate, log_emp, source_url, metadata)`.
2. `iter_state_industry_shares_annual(year=2005)` — annual state × NAICS supersector employment shares at the baseline year, from BEA SAEMP25N via FRED. Series ID pattern: `SAEMP25N{ST}{SECTOR_ID}` (varies). Output: `(state_code, industry_code, year, emp_share, source_url, metadata)`.

Alternative source if FRED doesn't cover BEA SAEMP25N: a small hardcoded shares table (51 states × 11 sectors = 561 numbers) drawn from a single BEA snapshot. Implementation chooses based on FRED availability at probe time.

### Estimation layer (extend existing)

**Reuse `puremacro.regress.lp.lp_panel`** from Notebook 29 — already supports interactions via `controls=` parameter. No new estimator code needed.

### Notebook 30a (new)

**`notebooks/30a_state_sectoral_bartik_lui.ipynb`**, paired with **`tools/make_notebook_30a_state_sectoral_bartik.py`** (per the notebooks↔builders convention).

Cell sequence:

1. Setup + imports.
2. Load LUI shock (reuse `notebooks/output_tables/28_lui_us_quarterly.parquet` + AR(4) construction from notebook 29).
3. Load state outcomes panel (reuse `notebooks/data_cache/state_*.parquet` from notebook 29; if not cached, recompute via the same fetcher chain).
4. Fetch national 2-digit NAICS quarterly employment (11 series, 1-2 sec via FRED).
5. Estimate national industry-level LUI elasticities. For each industry k, run a time-series LP:
   ```
   log_emp_{k,t+h} = β_k,h · shock_t + γ · controls + ε
   ```
   Single-unit, no panel — just a plain time-series LP. Use the existing `lp_panel` machinery with a degenerate single-unit panel for simplicity. Take peak-magnitude β_k = max_{h ∈ [0, 8]} |β_k,h|.
6. Fetch baseline state × industry shares (2005, BEA SAEMP25N via FRED or fallback table).
7. Compute state Bartik exposure: `exposure_state = Σ_k share_{state,k,2005} × |β_k^national|`. Standardize across states to mean 0, std 1.
8. Merge `exposure_state` into the state outcome panel (broadcast).
9. Run interacted state LP for each outcome (urate, log_emp, lfpr):
   ```
   y_{state, t+h} = α_state + γ_h · shock_t + δ_h · (shock_t × exposure_state) + covid_dummy + ε
   ```
   The `shock_t` main effect captures the average state response; the interaction `shock_t × exposure_state` captures the differential due to Bartik exposure.
10. Plot the interaction coefficient δ_h vs horizon. Positive δ_h (for urate) means high-exposure states see larger urate response.
11. Forest plot states ranked by exposure, colored by mfg-share quintile.
12. Meta JSON.

### Acceptance criteria

- **Headline interaction δ_h on urate positive at h ∈ [2, 6]** (high-exposure states have larger urate response). Magnitude: ~50% of the main effect γ_h, or larger.
- **Industry β_k^national pattern matches intuition**: |β_manufacturing|, |β_construction|, |β_leisure| > |β_government|, |β_education_health|.
- **State exposure variation is sufficient**: cross-state SD of exposure / mean ≥ 0.2.
- **Notebook reproducibility**: end-to-end re-run without manual intervention; outputs committed.

## Components

| File | Change |
|---|---|
| `puremacro/fetch/state_industry_panel.py` | New: national-industry + state-industry-share fetchers. |
| `puremacro/tests/test_fetch_state_industry.py` | New: offline FRED-mock tests + one network smoke. |
| `notebooks/30a_state_sectoral_bartik_lui.ipynb` | New notebook. |
| `tools/make_notebook_30a_state_sectoral_bartik.py` | New builder. |
| Notebook outputs (`30a_*.parquet`, `30a_*.pdf`, `30a_meta.json`) | New. |
| `puremacro/CHANGELOG.md`, `pyproject.toml`, `__init__.py`, `tests/test_import.py` | 0.9.1 → 0.10.0 (new fetcher module). |

## Failure handling

| Failure | Behavior |
|---|---|
| BEA SAEMP25N not on FRED (single-series IDs) | Fall back to hardcoded baseline shares table (51 × 11 = 561 numbers from one BEA snapshot). Documented in the fetcher. |
| Industry-level β_k^national fragile (small N, no significance) | Use absolute value AND a "qualitative" version (sign-only). Document. |
| Interaction δ_h insignificant / wrong-signed | Document as null result. The pooled LP IRFs from Notebook 29 still stand. |
| LUI shock too short to identify industry-level LPs | Use shorter horizons (h ∈ [0, 6]) for industry LPs. |
| State exposure variation too compressed | Add a robustness check using just manufacturing share alone (the simplest "Bartik-lite"). |

## Testing strategy

- **Industry fetcher**: offline mock test + one live FRED probe for `MANEMP`.
- **State-industry-share fetcher**: offline mock + one live probe for one series.
- **Bartik exposure construction**: synthetic test — given a known {share, β} table, verify exposure = Σ.
- **Interacted LP**: relies on existing `lp_panel` tests; no new estimator code.
- **Notebook 30a**: end-to-end re-run + outputs committed.

## Out of scope (deferred)

- County-level Bartik (Notebook 30b — next slice).
- Demographic exposure (BA-share, age, race) — Notebook 30c or 30d.
- GPSS-style 2SLS Bartik IV.
- Time-varying shares (rolling baseline).
- Industry-level LUI shock construction (industry-specific text).
- Slice 6b items (LLM kernel, Picault-Renault).
