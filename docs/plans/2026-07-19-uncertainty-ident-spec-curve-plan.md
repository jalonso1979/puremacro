# Uncertainty-Identification Spec-Curve Paper Plan — Phase 1 (pipeline + skeleton)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A reproducible spec-curve study — *"How Much of the Uncertainty-Shock Literature Is Identification Choice?"* — running the full `puremacro.var.identify` menu (plus `lp.la_lp`) on one fixed monthly US dataset and sweeping proxy / sample / detrending through `puremacro.inference.spec_curve`. Phase 1 delivers the pipeline, frozen data, results, figure, and the paper skeleton with honest numbers.

**Architecture:** One self-contained tool (`tools/run_uncertainty_ident_spec_curve.py`) that (1) assembles and freezes a monthly panel (EPU/WUI/JLN from `data/raw`, VIX + INDPRO/PAYEMS/FEDFUNDS from the no-key FRED fredgraph CSV), (2) runs 9 identification schemes on the same reduced-form VAR per dataset cell, (3) feeds a proxy × sample × detrend × scheme grid into `run_spec_curve`, and (4) emits figure + tables + `summary.json`. No package modules are touched; nothing under `puremacro/examples` or `notebooks/` (owned by parallel streams).

**Tech Stack:** numpy / scipy / pandas / matplotlib + `puremacro` internals only. Tools are not gated by the public-API snapshot; no new public symbols.

**Spec:** this file (Phase 1 was implemented alongside it; see the module docstring of the tool for every dataset convention and simplification).

---

## File map

### New files (Phase 1 — DONE)
- `tools/run_uncertainty_ident_spec_curve.py` — the pipeline (argparse: `--quick`, `--out`, `--refresh-data`).
- `docs/research/uncertainty_identification/DRAFT.md` — paper skeleton with pipeline-produced numbers.
- `docs/research/uncertainty_identification/output/` — `panel_monthly.csv`, `panel_manifest.json`, `headline_menu.csv`, `spec_curve_results.csv`, `fig_spec_curve.{pdf,png}`, `table_family_medians.{csv,md}`, `summary.json`, `run_log.txt`; Phase 2 adds `fig_gk_overlay.{pdf,png}` + `gk_overlay.csv`.
- `docs/plans/2026-07-19-uncertainty-ident-spec-curve-plan.md` — this plan.

### Modified files
- None. (Hard rule: no package modules, no CHANGELOG, no version bump, no snapshot.)

### Working assumptions (verified 2026-07-19 via signature dumps + live runs)
- `puremacro.inference.spec_curve`: `enumerate_specs(grid)`, `run_spec_curve(data=, specs=, estimator=, ci_level=)` (estimator returns `{"sigma_hat", "se", ...}`), `bootstrap_pvalue_median(curve_df, h0=, B=, rng=)`.
- `puremacro.var.identify`: `cholesky(Y, p=, horizon=, ordering=, n_boot=, ci=, seed=)` → `CholeskySVARResult` with `(H+1, n, n)` IRF arrays; `sign_restrictions`, `proxy`, `identify_maxshare`, `hetero` (Rigobon), `magmav_svar`, `non_gaussian_svar` — all `(H+1, n, n)`.
- `puremacro.lp.la_lp(df, y, x, horizons, n_lags, controls, alpha)` — LHS is `y(t+h) − y(t−1)` (a level change), HC0 SEs.
- `rigobon_svar` band machinery is dead code on this tree: `from ..inference.moving_block_bootstrap` resolves to the nonexistent `puremacro.var.inference`, so `_HAS_MBB is False` and `lower=upper=None` always. The pipeline bootstraps Rigobon bands itself (regime-preserving residual resimulation). *Phase-2 candidate: fix the relative import in `puremacro/var/identify/hetero.py` (one-line package change, needs its own release slot).*
- `magmav_svar`'s convergence gate (`res.fun < 1e-3`) is scale-dependent; the pipeline standardises columns before calling it (B is scale-equivariant, so exact) — without this, real-data cells return `eu=(0,0)` everywhere.
- Local data verified: `data/raw/www.policyuncertainty.com/media_US_Historical_EPU_data.xlsx` (1900-01…2014-10), `media_All_Country_Data.xlsx` (`US` col, 1985+), `worlduncertaintyindex.com/wp-content_uploads_2024_10_WUI_Data.xlsx` (sheet T2 has quarterly `USA`), `www.sydneyludvigson.com/s_MacroFinanceUncertainty_202602Update-3klg.zip` (JLN h=1/3/12, 1960-07…2025-12).
- FRED fredgraph CSV (no key) serves INDPRO, PAYEMS, FEDFUNDS, VIXCLS; `puremacro.fetch._classic.fetch_fred` wraps it.

---

## Phase 1 — pipeline + results + skeleton (DONE 2026-07-19)

- [x] **Task 1: dataset assembly + freeze.** Monthly panel `(epu, wui, jln, vix, ip, emp, ffr)`; EPU ratio-spliced (historical×modern over the 1985–2014 overlap); proxies z-scored on full history; panel frozen to `panel_monthly.csv` + sha256 manifest. Reruns read the frozen csv (offline).
- [x] **Task 2: identification menu on the baseline dataset** (EPU, full sample 1954-07→2025-11, linear-detrended log levels, p=6, H=24): chol-first, chol-last, sign (u↑, IP↓, h=0..2), proxy (partner-proxy AR(6) innovations, censored-proxy zero-fill), max-share (u FEV over 12m), Rigobon (GR+COVID high-vol regime via `regime_dates.REGIMES_US`), MagMav (endogenous breaks), non-Gaussian ICA, la_lp. Unit-effect normalisation (u impact = +1σ). Peak + h=12 with 90% bands → `headline_menu.csv`.
- [x] **Task 3: spec-curve grid.** 4 proxies × 3 samples × 2 detrendings × 9 schemes = 216 cells (deduped when windows coincide, la_lp kept only under `lt`); `run_spec_curve` + `bootstrap_pvalue_median(h0=0)`; figure (sorted estimates, colored by family, validated palette + marker shapes); family-medians table.
- [x] **Task 4: determinism gate.** Reload frozen csv, recompute 12 randomly chosen (seed-7) specs, assert bit-identical `sigma_hat`/`resp_h12` (in-script `AssertionError` otherwise). Passing on both `--quick` and full runs.
- [x] **Task 5: DRAFT.md skeleton** with the actual run numbers (abstract, positioning vs Bloom 2009 / BBD 2016 / JLN 2015 / LMN 2021 / CCM, methods grid, results, limitations).

**Verification:** `python tools/run_uncertainty_ident_spec_curve.py --quick` (< 3 min) and the full run (< 20 min); no tests added under `tests/` (tools are not gated).

---

## Phase 2 — narrative-sign extension (DONE 2026-07-19; unblocked by `puremacro.var.identify.narrative_sign_svar` landing)

- [x] **Task 1:** Add a `narrative` scheme to the menu: narrative sign restrictions à la Antolín-Díaz & Rubio-Ramírez (2018, AER) — implemented via `var.identify.narrative_sign_svar` with plain `(date, shock, sign)` Type-I tuples (not `puremacro.narrative` event indices; the AD-RR module's tuple shorthand made the index adapter unnecessary). Events: 1987-10 (Black Monday), 2001-09 (September 11), 2008-09 (Lehman), 2020-03 (COVID onset), positive uncertainty shock, layered on the same traditional pattern as the `sign` scheme; events outside a cell's window dropped and recorded; ESS + acceptance diagnostics in `headline_menu.csv` / `spec_curve_results.csv`. Menu (baseline): peak −2.72 [−4.18, −0.69], h12 −2.46, ESS 1087 (1087/1436 trad. draws survive).
- [x] **Task 2:** Re-run the grid (full run 187s, determinism gate PASSED). Grid narrative cells restricted to the baseline detrending (lt) per the Phase-2 runtime scope — 11 fd cells skipped and logged; 240 cells, 189 estimated. narrative-sign is the 8th color slot `#e34948` appended LAST in `FAMILY_ORDER` (validator: documented 8-slot order passes adjacent-mode, worst CVD ΔE 9.1 / normal-vision 19.6; inserting red beside green/magenta FAILS the normal-vision floor at 13.2, hence last, with the `*` marker as secondary encoding).
- [x] **Task 3:** Update DRAFT.md numbers + limitations (done; the "no narrative schemes" caveat replaced by narrative-specific caveats: OLS-fixed reduced form, constant Type-I weights ⇒ ESS = surviving count, lt-only grid cells, event list not yet swept). Headline finding: narrative moderates the sign family — less negative in 10/11 matched cells, family median −3.18 vs −3.90, range halved (7.3pp vs 13.2pp).

## Phase 3 — GK identified-set overlay

- [x] **Task 1:** DONE 2026-07-19, delivered as a variation: `run_gk_overlay` computes `var.identify.gk_robust_bands` (traditional sign pattern, `n_var_draws=1`, 5000 rotations) on the baseline dataset and renders a dedicated two-panel figure `output/fig_gk_overlay.{pdf,png}` (+ `gk_overlay.csv`) contrasting posterior-percentile vs identified-set bands for the sign AND narrative schemes — instead of shading the spec-curve figure (the set is a raw-structural-scale object per baseline dataset; overlaying it on the unit-normalised multi-dataset curve would compare incompatible scales). Headline: at h=12 the sign posterior band [−1.26, −0.06] excludes zero, the GK set [−1.36, +0.26] does not; set upper bound positive from h=3 on.
- [ ] **Task 2:** Add an identified-set column to the family table; discuss vs Ludvigson-Ma-Ng set-identification results. (Partially covered: DRAFT.md §3 discusses the GK-vs-LMN reading; the family-table column and the narrative-truncated identified set need `gk_robust_bands` API work — a package change with its own release slot.)

## Phase 4 — robustness + journal targeting

- [ ] Lag-order sweep (p ∈ {3, 6, 12}) and horizon sweep (H ∈ {24, 36}) as extra spec dimensions; report whether family ordering is stable.
- [ ] Quarterly replication (GDP instead of IP) to connect to Carriero-Clark-Marcellino (2018)-style datasets.
- [ ] Target: short empirical-methods outlet (Economics Letters / Economic Inquiry note) or JAE replication section; the deliverable is the pipeline + curve, not a new estimator.

## Explicitly out of scope (all phases)

- Editing `puremacro/` package modules, `puremacro/examples`, `notebooks/`, CHANGELOG, version, or the public-API snapshot.
- Bayesian set-identification (CCM-style large BVARs) — computationally out of the Pyodide-pure budget.
