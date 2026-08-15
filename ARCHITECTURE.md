# puremacro — architecture

This document is the design map for `puremacro`. It complements `README.md` (which is user-facing) by recording (a) the module dependency graph, (b) per-module stability tier, and (c) the Pyodide-compatibility contract — i.e. exactly which optional dependencies are allowed and where they live.

If you are revising the package, read this first. If your change makes anything below stale, update it in the same commit.

The package was significantly extended in **Phase 5** (April–early-May 2026, commits `c4482f2` … `03ecc41`), which absorbed ~22K LOC from the pre-existing `src/` tree into `puremacro/`. The map below reflects that post-Phase-5 reality.

---

## Module map

The tree below is grouped by **intent**, not by alphabet. Estimators and inference machinery come first; data pipelines, presentation, and side-channels are at the bottom.

```
puremacro/
├── __init__.py            ← only re-exports __version__
├── _linalg.py             ← internal linalg primitives (inv_xtx, safe_cholesky)
├── numerics.py            ← MLE / numerical-differentiation primitives
├── state_space.py         ← Kalman filter / smoother
├── mcmc.py                ← MCMC convergence diagnostics
├── posterior.py           ← posterior-draw utilities (IRF fans)
├── experiment.py          ← top-level dispatcher (lp / panel_lp / var_cholesky)
│
│ ── core econometrics ─────────────────────────────────────────────
├── var/
│   ├── estimate.py        ← reduced-form OLS VAR
│   ├── irf.py             ← IRF / FEVD / GFEVD primitives
│   ├── bootstrap.py       ← residual bootstrap for IRF bands
│   ├── bvar.py            ← Minnesota-prior BVAR (dummy obs + Gibbs)
│   ├── vecm.py            ← Engle-Granger, Johansen, VECM
│   ├── tvp.py             ← time-varying-parameter VAR
│   ├── panel.py           ← panel VAR
│   ├── diagnostics.py     ← Granger, stability, residual tests
│   ├── identify/          ← canonical SVAR identifications (cholesky / sign /
│   │                       sign_zero / sign_robust / proxy / hetero /
│   │                       maxshare / non_gaussian / bq); result-object std.
│   │                       `panel.mean_group_svar` (PanelSVARResult) and
│   │                       `maxshare.identify_maxshare` (MaxShareResult)
│   │                       promoted to canonical at 0.43.0.
│   └── regime/            ← regime-switching VARs
│
│
├── lp/
│   ├── _panel_helpers.py  ← two_way_fe_within + panel_lp_horizon_loop
│   ├── jorda.py           ← single-country LP-HAC (pure-numpy)
│   ├── iv.py / iv_helpers.py    ← LP-IV
│   ├── la_lp.py           ← lag-augmented LP (Plagborg-Møller-Wolf)
│   ├── panel.py           ← panel LP, cluster SE (thin wrapper, pure-numpy)
│   ├── panel_dk.py        ← panel LP, Driscoll-Kraay SE (thin wrapper)
│   ├── panel_iv.py        ← panel LP-IV
│   ├── state_dep.py / smooth.py / asymmetric.py / quantile.py
│   ├── garch_state.py / garch_in_mean.py
│   ├── mean_group.py / cce.py
│   │
│   └── lp_*.py            ← OLDER absorbed-from-src/ LP variants (lp_jorda,
│                            lp_iv, lp_panel, lp_panel_dk, lp_state_dep,
│                            lp_smooth, lp_garch_state, lp_garch_in_mean,
│                            garch_utils). R1_02/R1_03/R1_05 notebooks still
│                            import legacy IRF function variants and result
│                            classes; R1_02 is specifically the legacy lp_*
│                            API demo by design. Phase-2.5 follow-up deferred
│                            to 0.44.0 with body-rewrites of those notebooks.
│
├── inference/
│   ├── _ols_helpers.py    ← ols_hac — central OLS path
│   ├── _results.py        ← shared frozen-dataclass result types
│   ├── hac.py             ← Newey-West SE
│   ├── hac_fixed_b.py     ← Kiefer-Vogelsang fixed-b HAC
│   ├── dk.py              ← Driscoll-Kraay panel HAC
│   ├── bootstrap.py / block_bootstrap.py / moving_block.py / wild_bootstrap.py
│   ├── moving_block_bootstrap.py / lp_block_bootstrap.py    ← absorbed
│   ├── weak_iv.py         ← Cragg-Donald, Kleibergen-Paap, AR, MSW
│   ├── over_id.py         ← Hansen-J, Stock-Yogo
│   ├── pesaran.py / pesaran_cce.py / swamy.py / swamy_test.py
│   ├── balanced_panel.py / newey_west.py
│   ├── quandt_andrews.py / spec_curve.py    ← structural-break + sensitivity
│   └── legacy/            ← mirror of the absorbed-from-src/ inference
│                            modules (bootstrap, moving_block_bootstrap,
│                            lp_block_bootstrap, balanced_panel, …). Still
│                            actively imported by lp/lp_state_dep.py and
│                            lp/lp_smooth.py. Not "deprecated" — name is
│                            historical. Retires alongside lp_*.py at 0.44.0.
│
├── garch/                 ← GARCH(1,1) (scipy-only fit), DCC
├── volatility/            ← SigmaObject (MATLAB-port), BEKK, CCC, HAR-RV,
│                            range-based, ARCH-LM / Ljung-Box
├── sigma/                 ← MINIMAL teaching stub for SigmaObject. The
│                            canonical full port lives in volatility/sigma.py
│                            — keep both; sigma_numpy.py is referenced in
│                            teaching material that deliberately uses the
│                            stripped-down API.
│
├── dsge/                  ← klein.py (QZ solver, BK condition); also a
│                            load_dynare.py (Phase 5 absorb) for reading
│                            *_results.mat from Dynare.
├── connectedness/         ← Diebold-Yilmaz spillover
├── forecast/              ← Diebold-Mariano, Giacomini-White, density eval
├── tests/                 ← Bai-Perron breaks + unit-root tests
│                            (distinct from project-level tests/ directory)
├── narrative/             ← fiscal-narrative IV (events, dedup, panel,
│                            replication); pypdf used lazily by sources/
├── nowcast/               ← kalman_dfm, mf_var, forecast combos, CRPS
├── gar/                   ← Growth-at-Risk: QAR, skew-t, FCI
├── did/                   ← Staggered DiD: CS, Sun-Abraham, BJS, SDID
├── dynpanel/              ← Arellano-Bond + Blundell-Bond dynamic-panel GMM
├── hfi/                   ← high-frequency monetary surprises (GK / NS / JK)
│
│ ── estimators absorbed by Phase 5 ────────────────────────────────
├── cycles.py              ← Hamilton 2018 trend-cycle filter
├── cointegration_modern.py← Phillips-Hansen FM-OLS, Stock-Watson DOLS,
│                            Phillips-Ouliaris (frozen-dataclass results)
├── factor.py              ← PCA factors + Bai-Ng (2002) IC for k*
├── korv_gmm.py            ← KORV (2000) system-GMM two-elasticity CES
├── midas.py               ← unrestricted + beta-poly MIDAS
├── spectral.py            ← Welch PSD, cross-spectrum, coherence (numpy.fft)
├── synthetic_control.py   ← Abadie-Diamond-Hainmueller + placebo inference
├── wavelet.py             ← DWT / MODWT Haar variance decomposition
├── realized_vol.py        ← realized variance, bipower, Corsi HAR-RV
├── labor_share.py         ← Gollin (2002) self-employed-adjusted share
├── scale.py               ← IRF scaling / peak finder utilities
│
│ ── new sub-packages absorbed by Phase 5 ──────────────────────────
├── uncertainty/           ← T15 long-history cross-country uncertainty
│                            study: between/within decomposition, regime
│                            helpers, LP-per-country, composite/innovation
│                            builders (DEFAULT_PROXIES, build_backbone, …).
├── instruments/           ← Instrument registry / composition / external
│                            loaders + an SDMX→Instrument adapter.
├── regress/               ← Contains `lp.py`, an independent pure-numpy LP
│                            implementation (NOT a thin re-export of lp.panel
│                            — different signature). 3 callers in
│                            tools/run_*.py use its "split-then-compare"
│                            regime pattern. Soft-legacy; its own follow-up.
├── sa/                    ← Seasonal adjustment: STL (default, pure-numpy
│                            fallback) + X-13ARIMA-SEATS (requires the
│                            x13as binary on PATH). Lazy imports.
├── plotting/              ← Library-API plotting (irf_plot.py + bw_style.py
│                            + helpers). Distinct from plot.py — see below.
├── plot.py                ← Pyodide-friendly grayscale IRF/FEVD plots,
│                            zero styling deps. Used by notebooks that
│                            run on juno.sh / iPad.
├── reports.py             ← table / report builders
│
│ ── data pipelines (entire new category vs the 0.4.0 doc) ─────────
├── fetch/                 ← Public-data fetchers, all routed through
│                            ._http (UA override, SSL fallback, 30s
│                            timeout). Modules for FRED / ALFRED / SDMX
│                            (OECD / Eurostat / ECB / IMF SDMX-Central),
│                            FRED-states, EPU / GPR / WUI / JLN /
│                            Fernald, OECD-MEI / OECD-QNA / OECD-energy
│                            / OECD-FX / OECD-QNA-labor, WB pink-sheet,
│                            ILOSTAT, Yahoo, plus a STL/X-13 seasonal
│                            helper (_seasonal.py — statsmodels lazy).
├── build_panel.py         ← Orchestrates panel_Q + panel_M from fetch/*.
│                            Public entry: build_all(countries, fast,
│                            refresh). Imports arch lazily for GARCH-σ.
├── build_subnational_panel.py
│                          ← US state/county panel (QCEW / LAUS / CES);
│                            ships a build_all entrypoint of its own.
├── bartik/                ← Shift-share construction + sensitivity
│                            (statsmodels lazy in sensitivity.py).
├── klems.py               ← EU-KLEMS 2023 loader (labor/capital, skills).
├── bis_neer.py            ← BIS NEER quarterly aggregator.
├── long_panel.py          ← G9 homogeneous-vintage splicing (PWT10 +
│                            OECD-STAN 1975–2024).
├── vintages.py            ← Real-time vintage snapshots + revision tracking.
├── _http.py               ← Shared HTTP primitives (lazy urllib).
├── _codes.py              ← Country-code canonicalisation; aggregate drop.
├── cache.py               ← Disk-cache abstraction (~/.cache/puremacro/).
├── regime_dates.py        ← Hard-coded regime date dictionaries.
├── regimes.py             ← Regime utility helpers (breaks_to_regimes,
│                            dates_to_regimes); distinct from regime_dates.
│
│ ── research side-channels (NOT in the Pyodide promise) ───────────
├── teaching/              ← MATLAB-parity teaching prototypes that
│                            deliberately wrap statsmodels / linearmodels /
│                            arch (lp_sm, panel_lm, var_sm, garch_arch, …)
│                            so notebooks can compare puremacro's pure-
│                            numpy estimators against canonical packages.
│                            Excluded from the Pyodide compat sweep.
└── examples/              ← ~60 end-to-end replication scripts
                             (Bloom 2009, Mertens-Ravn, …). Out-of-scope
                             for the Pyodide promise; excluded from the
                             compat sweep.
```

### Key dependency edges

These are the load-bearing imports. If you change one of these arrows, double-check the consumers in the same commit.

| Direction | Reason |
|---|---|
| `_linalg`  ← (everywhere OLS happens) | All `(X'X)^{-1}` and Cholesky factorisations route here. |
| `inference/_ols_helpers.ols_hac` ← `lp/jorda`, `lp/iv`, `lp/asymmetric` | Single source of truth for the LP-HAC point estimate + Newey-West SE. |
| `lp/_panel_helpers.panel_lp_horizon_loop` ← `lp/panel`, `lp/panel_dk` | Shared engine; cluster vs. DK SE selected by callback. |
| `var/estimate` + `var/irf` ← `var/identify/*`, `inference/moving_block`, `connectedness/diebold_yilmaz` | Reduced-form VAR is the spine of every SVAR identification scheme and the spillover-index machinery. |
| `lp/garch_utils` ← `lp/garch`, `lp/garch_in_mean` | Public helper used by `R1_02_lp_menu` to fit σ_t before calling lp-garch variants. Remains public; rename to `_garch_utils` deferred. |
| `var/identify/panel.mean_group_svar` uses `safe_cholesky` + `inference/bootstrap` | No `inference.legacy` dependency; canonical path throughout. |
| `var/identify/maxshare.identify_maxshare` uses `var/estimate.estimate_var` + `safe_cholesky` | No statsmodels dependency on the bootstrap path; inline residual resampling. |
| `narrative/types` ← `narrative/{aggregate,dedup,validate,panel,scoring/*,replication/*}` | `NarrativeEvent` / `NarrativeInstrument` are the canonical schema. |
| `narrative/replication/__init__.py` re-exports both `load_*` and `*_csv_to_events` | Examples import the public CSV-to-events helpers from `puremacro.narrative`; do not reach into `replication.<dataset>` directly. |
| `fetch/*` ← `build_panel`, `build_subnational_panel` | All quarterly/monthly panel construction goes through the fetcher layer; no direct HTTP at the build_panel level. |
| `_codes.drop_aggregates` ← `build_panel`, downstream notebooks | Single source of truth for the EA20/EU27/OECD/WLD aggregate filter. |

---

## Stability tiers

| Module | Tier | Notes |
|---|---|---|
| `var/estimate`, `var/irf`, `var/bootstrap` | **Mature** | Stable API; replication-tested (Bloom 2009, Mertens-Ravn). |
| `lp/jorda`, `lp/iv`, `lp/panel`, `lp/panel_dk` | **Mature** | LP-HAC + cluster + DK; parity-tested against `linearmodels.PanelOLS`. |
| `inference/_ols_helpers`, `inference/hac`, `inference/dk`, `inference/weak_iv` | **Mature** | Central inference machinery. Cragg-Donald / Kleibergen-Paap / MSW bands are replication-tested. |
| `garch/fit` (GARCH(1,1)) | **Mature** | scipy-only; bounded L-BFGS-B with variance floor at 1e-10. |
| `dsge/klein` | **Mature** | QZ-based; BK condition enforced. |
| `state_space`, `mcmc` | **Mature** | Kalman filter w/ NaN handling; standard MCMC diagnostics. |
| `var/identify/*` | **Stable, but watch the bootstrap** | Cholesky bootstrap drops non-PD draws and warns above 5% failure rate. All public estimators return frozen-dataclass result-objects 0.42.0+. |
| `var/identify/panel` (`mean_group_svar` + `PanelSVARResult`) | **Stable** | Canova-Ciccarelli 2013 mean-group panel SVAR. Supports `cholesky` and `bq` natively; uses `safe_cholesky` + canonical `inference/bootstrap` — no `inference.legacy` dependency. |
| `var/identify/maxshare` (`identify_maxshare` + `MaxShareResult`) | **Stable** | Faust-Uhlig full pipeline: identification + FEVD + residual bootstrap. `ci` kwarg replaces legacy `q_lo`/`q_hi`. Low-level `maxshare(...)` / `news_maxshare(...)` unchanged. |
| `lp/panel` (`lp_panel_regime_interaction`) | **Stable** | Returns long-form `pd.DataFrame` keyed by `(h, regime)`. Ported from legacy at 0.43.0. |
| `lp/state_dep` (`lp_smooth_transition_irf`) | **Stable** | HAC analytic CIs (replaces legacy block-bootstrap CIs). Ported from legacy at 0.43.0. |
| `var/bvar`, `var/vecm`, `var/tvp` | **Stable** | All `(X'X)^{-1}` calls route through `_linalg.inv_xtx`. |
| `lp/state_dep`, `lp/smooth`, `lp/asymmetric`, `lp/quantile` | **Stable** | Replication-light; exercise these via the example scripts. |
| `inference/{lp_block_bootstrap, moving_block_bootstrap, balanced_panel, swamy_test, pesaran_cce, ...}` | **Stable** | Phase-5 absorbs. `lp_block_bootstrap` lazy-imports `puremacro.teaching.panel_lm` (Phase 0). |
| `inference/quandt_andrews`, `inference/spec_curve` | **Stable** | Structural-break testing + sensitivity-curve helpers. |
| `narrative/{types,aggregate,dedup,validate}` | **Stable** | Offline-deterministic core. |
| `narrative/scoring/{keyword,manual}` | **Stable** | Pure logic. |
| `narrative/replication/*` (live loaders) | **Best-effort** | Network-dependent. Each ships an offline-tested CSV-to-events helper; `load(...)` mirrors are smoke-tested when reachable, skipped otherwise. |
| `narrative/scoring/llm`, `narrative/sources/*` | **Experimental** | LLM and HTTP backends. **Not** in the Pyodide promise. `narrative/sources/_extractors.py` uses `pypdf` lazily; install via `pip install puremacro[narrative]`. |
| `var/regime/*`, `connectedness/diebold_yilmaz`, `forecast/*` | **Stable** | Smaller surface; smoke-tested. |
| `volatility/sigma` (canonical SigmaObject) | **Stable** | 1:1 port of MAV/SigmaObject.m. 13 unit tests pin the MATLAB API. |
| `sigma/sigma_numpy` (teaching stub) | **Stable** | Minimal variance-decomposition SigmaObject (64 LOC). Tested separately; used by teaching material. Not a duplicate of `volatility/sigma` — different scope. |
| `volatility/{multivariate, har, range, diagnostics}` | **Stable** | BEKK / CCC, HAR-RV, range-based, ARCH-LM / Ljung-Box. |
| `nowcast/{dfm, mfvar, combine, scoring}` | **Stable** | Kalman-DFM with ragged edges; MF-VAR; combos + scoring rules. |
| `gar/{qar, skewt, fci}` | **Stable** | Quantile AR; ABG 2019 skew-t; NFCI-style FCI. |
| `did/{callaway_santanna, sun_abraham, borusyak_jaravel_spiess, synthetic_did}` | **Stable** | Modern staggered-DiD set; bootstrap SEs throughout. |
| `hfi/{gk2015, ns2018, jk2020}` | **Stable** | HFI of monetary policy shocks. |
| `cycles` | **Stable** | Hamilton 2018 regression filter; returns `(cycle, trend)` tuple per the two-value carve-out. |
| `cointegration_modern` | **Stable** | FM-OLS / DOLS / Phillips-Ouliaris with frozen-dataclass results (`FMOLSResult`, `DOLSResult`, `PhillipsOuliarisResult`). Distinct from `var/vecm`. |
| `factor` | **Stable** | PCA + Bai-Ng IC; frozen-dataclass result. |
| `korv_gmm` | **Stable** | Two-elasticity system GMM. |
| `midas` | **Stable** | UMidas + beta-poly Midas with frozen-dataclass results. |
| `synthetic_control` | **Stable** | ADH 2010 + placebo. Returns a `dict` today — Phase 3 candidate for wrapping in `SyntheticControlResult`. |
| `spectral` | **Stable** | Welch / cross-spectrum / coherence / gain / phase. Returns `dict`; Phase 3 candidate. |
| `wavelet` | **Stable** | DWT / MODWT Haar variance decomposition. Returns `dict`; Phase 3 candidate. |
| `realized_vol` | **Stable** | Realized variance, bipower, Corsi HAR-RV. |
| `labor_share` | **Stable** | Gollin (2002) labor-share construction. |
| `dynpanel/{ab_gmm, bb_gmm, instruments, diagnostics}` | **Stable** | Two-step Windmeijer (analytic) + Hansen J + AR(1)/AR(2) + Roodman collapse. `GMMResult` frozen dataclass. |
| `uncertainty/*` | **Stable** | T15 cross-country uncertainty study: decomposition, regime helpers, LP-per-country, composite/innovation builders. |
| `instruments/*` | **Stable** | Instrument registry + composition; backbone for the LP-IV pipeline. |
| `bartik/*` | **Stable** | Shift-share construction + sensitivity. `sensitivity.py` lazy-imports statsmodels (Phase 0). |
| `klems`, `bis_neer`, `long_panel`, `vintages`, `realized_vol`, `labor_share` | **Stable** | Single-file data utilities. |
| `fetch/*` | **Best-effort** | Network-dependent. Each fetcher has an offline path / cache layer; live calls smoke-tested where reachable. `fetch/_seasonal.py` and `fetch/fred_states.py` lazy-import statsmodels / arch (Phase 0). |
| `build_panel`, `build_subnational_panel` | **Stable** | Orchestrators on top of `fetch/*`; idempotent against the disk cache. `build_panel` lazy-imports arch for the GARCH-σ derivation. |
| `sa/{stl, x13}` | **Stable** | STL fallback when X-13 is unavailable. Both lazy-import statsmodels (Phase 0). |
| `plotting/*` vs `plot.py` | **Stable / Stable** | Two co-existing presentation paths — see "Legitimate distinctions" below. |
| `cache`, `_http`, `_codes`, `regime_dates`, `regimes`, `scale` | **Stable** | Cross-cutting utilities. |
| `regress/*` | **Soft-legacy** | `regress/lp.py` is an independent pure-numpy LP implementation (not a thin re-export of `lp.panel` — different signature). 3 callers in `tools/run_*.py`; its own follow-up release. |
| `teaching/*` | **Out of Pyodide scope** | Excluded from the Pyodide test sweep. Statsmodels / linearmodels / arch are hard runtime deps here by design. |

---

## Pyodide-compatibility contract

The package's numerical core is **pure numpy + scipy + pandas + matplotlib** — no statsmodels / linearmodels / arch anywhere in the estimator path. That is **load-bearing**: if you add an `import` that breaks it, you are deleting the whole point of the package, and the estimator core stops being importable under Pyodide.

### Allowed runtime dependencies

Declared in `pyproject.toml [project.dependencies]`. This block is parsed and asserted verbatim by `tests/test_pyodide_compat.py::test_pyproject_runtime_deps_match_documentation`, so the two must be edited in the same commit:

```
numpy   >= 1.26
scipy   >= 1.10
pandas  >= 2.0
matplotlib >= 3.7
requests >= 2.31
pyarrow >= 15
```

The first four are the **Pyodide import core**: the only third-party modules a shippable estimator module may import at top level. The last two widen the *install* contract, not the import contract, and each is here for a concrete reason:

- `requests` — the whole `puremacro.fetch` layer (OECD/SDMX, EPU, FRED-CSV, IMF, BEA) and the narrative sources `import requests` at module level by design. Without it a clean `pip install puremacro` dies with `ModuleNotFoundError` on the first fetch call. It is pure Python and installs fine under Pyodide (there are no sockets there, but the offline CSV paths never touch it).
- `pyarrow` — the parquet engine `pandas.read_parquet` needs. `cache.py`, `fetch/labor*.py`, `shock_atlas.py`, `build_panel` / `build_subnational_panel` and the ENOE datasets shipped with the teaching material are all parquet. pandas imports it lazily, so it never lands in `sys.modules` during an import sweep, but it is a hard requirement to *use* those code paths — and the documented student install is a bare `pip install puremacro`, so it cannot live in an extra. **Caveat:** pyarrow has no Pyodide wheel, so an in-browser install must go through `micropip.install("puremacro", deps=False)` plus the deps actually needed; the browser is not a supported deployment target of the teaching material.

**Consequence for the opt-in Pyodide gate:** `tools/pyodide/runner.js` installs the built wheel with `micropip.install("emfs:/tmp/<wheel>")`, i.e. with dependency resolution ON. Since `pyarrow` became a base dependency that resolution can no longer succeed in the browser, so `python tools/release_check.py --pyodide` (gate 6) fails at the install step, not at the import sweep. The runner has to switch to `deps=False` and preload the import core explicitly (`numpy`, `scipy`, `pandas`, `matplotlib` are already `loadPackage`d a few lines above; `requests` would need `micropip`). The in-process guarantee is unaffected: the two sweeps in `tests/test_pyodide_compat.py` are green, because nothing shippable imports `pyarrow` at module level.

Anything else is **dev-only or extra-only** (pytest, statsmodels, linearmodels, arch, pypdf, beautifulsoup4, pdfplumber). These are declared in `[project.optional-dependencies]`:

- `dev = ["pytest", "statsmodels", "linearmodels", "arch"]` — parity tests.
- `narrative = ["pypdf", "beautifulsoup4", "pdfplumber"]` — body-extraction backend for `narrative/sources/_extractors.py`; HTML parsing for connectors (Beige Book, EUR-Lex, EU Parliament, SOTU); PDF text extraction for connectors (ERP, CBO, WARN).

Dev / extra deps **must not** be imported at runtime by code that ships in the wheel without going through a lazy-import guard. The one architecturally-sanctioned exception is `puremacro.narrative.sources.*`, which is the HTTP/scraping side-channel — modules under that path may import `beautifulsoup4` / `pdfplumber` at top level since the Pyodide-compat walker (see "Excluded from the Pyodide sweep" below) explicitly skips them.

### Lazy-import pattern

```python
# Module top: no forbidden top-level import.
def my_estimator(...):
    import statsmodels.api as sm  # lazy: Pyodide contract
    ...
```

For module-attribute access (tests monkeypatching the dependency), define a module-level shim wrapper that does the lazy import inside its body and delegates — see `puremacro/fetch/_seasonal.py:_x13_arima_analysis` for the canonical example.

### Lazy-loaded leaks (Phase 0 — `2026-05-14`)

The Phase 0 sweep converted the following hard top-level imports to lazy form. The Pyodide-compat regression test (`tests/test_pyodide_compat.py`) is green as of this revision.

| File | Optional dep | Pattern |
|---|---|---|
| `bartik/sensitivity.py` | statsmodels | function-scope lazy import |
| `build_panel.py` | arch | function-scope lazy import |
| `fetch/_seasonal.py` | statsmodels (x13) | module-level shim wrapper (preserves monkeypatch contract in `tests/test_fetch_seasonal.py`) |
| `fetch/fred_states.py` | arch | function-scope lazy import |
| `sa/stl.py` | statsmodels (STL) | function-scope lazy import |
| `lp/garch_utils.py` | arch (type hint guarded by `from __future__ import annotations`) | function-scope lazy import |
| `inference/lp_block_bootstrap.py` | transitive via `puremacro.teaching.panel_lm` | function-scope lazy import |

If you add a new optional import, follow the same pattern: lazy import + caller-side gate. **Never** put a guarded import at module top level.

### Excluded from the Pyodide sweep

`tests/test_pyodide_compat.py` walks every shippable submodule and asserts that none of `statsmodels` / `linearmodels` / `arch` show up in `sys.modules` after the import sweep. The walker intentionally skips:

- `puremacro.examples.*` — replication scripts.
- `puremacro.teaching.*` — MATLAB-parity prototypes (intentional statsmodels/linearmodels/arch use).
- `puremacro.narrative.sources.*` — HTTP/scraping side-channel.
- `puremacro.narrative.scoring.llm` — LLM SDK side-channel.
- `puremacro.tests.*` — break / unit-root tests (in-package test submodule, not the project-level `tests/`).

If you put a new module under one of these prefixes, you opt out of the Pyodide promise. Do this deliberately.

---

## Result-object standard (0.4.0+)

All public estimators that return three or more fields (or any non-trivial diagnostic) MUST return a frozen dataclass result object. The contract:

1. **`@dataclass(frozen=True)`** for any return with 3+ fields or non-trivial diagnostics.
2. **Naming:** `<MethodName>Result` in PascalCase (e.g., `GMMResult`, `IRFResult`, `JKResult`, `ProxySVARResult`). Defined in `<subpackage>/_results.py`; re-exported via `<subpackage>/__init__.py`.
3. **Tuple returns** still allowed for genuinely simple two-value returns (`cycle, trend = hamilton_filter(y)`).
4. **Common field vocabulary**: `coefs`, `se`, `cov`, `names: tuple[str, ...]`, `n_obs`, `converged`.
5. **`.summary() -> str`** optional but encouraged.
6. **No `.plot()` method.** Plotting stays in `plot.py` / `plotting/`; result objects are pure data.
7. **No `__post_init__` validation that raises.** The estimator builds a valid result; the dataclass just stores it.
8. **DataFrame carve-out.** Functions returning a single `pandas.DataFrame` with named columns (e.g. every `lp/` estimator) do NOT need to wrap. The DataFrame is already self-documenting.

The public-API freeze test (`tests/test_public_api.py`) snapshots both `__all__` per subpackage and result-class field names per dataclass.

### Phase 3 wrapping (done 2026-05-14)

| Function | Was | Is now |
|---|---|---|
| `spectral.welch_cross` | 7-key `dict` | `WelchCrossResult` |
| `wavelet.wavelet_variance` | 4-key `dict` | `WaveletVarianceResult` |
| `wavelet.wavelet_coherence` | 2-key `dict` | `WaveletCoherenceResult` |
| `synthetic_control.synthetic_control` | 5–6-key `dict` | `SyntheticControlResult` (optional `placebo_gaps`) |

`spectral.welch_psd` returns `(f, Pxx)` and `business_cycle_band_power` returns a float — both covered by the 2-value / scalar carve-outs, no wrapping needed.

### Phase-2 shim status (resolved and retired at 0.43.0)

Phase 2 shipped shims at 0.42.0; the entire `svar/` package (shims + Phase-2.5
banner files) was deleted at 0.43.0. The two functions below are now canonical:

| Function | Returns | Status |
|---|---|---|
| `var.estimate.estimate_var` | `VarEstimateResult` (iterable; `len==5`) | Canonical. `svar/estimate_var` shim deleted at 0.43.0. |
| `var.identify.bq.bq_svar` | frozen dataclass | Canonical. `svar/identify_bq` shim deleted at 0.43.0. |

### Signal contract (0.65.0+)

`puremacro.narrative.RiskIndex` carries two optional companion fields —
`quality: SignalQualityReport | None` and `draws: pd.DataFrame | None` —
that let downstream estimators see how reliable the index reading is
and (in 0.66.0+) propagate measurement uncertainty into IRF bands.
Both default to `None`; every pre-0.65.0 caller is unaffected. Opt in
via `with_quality=True` (Slice 1) or `with_draws='basic'|'full'`
(Slice 2). The single-page reference is `docs/SIGNAL_CONTRACT.md`.

### Data infrastructure (0.66.0+)

`puremacro.credentials` is the single resolver every API-keyed fetcher
uses (env vars + `~/.puremacro/credentials.toml`; raises
`MissingCredentialError` with a researcher-actionable message).
`puremacro._cache_db` hosts the shared SQLite connection (WAL mode)
that backs both the HTTP cache (`puremacro._http_cache`, public API
preserved from 0.65.0) and the ALFRED vintage store
(`puremacro.vintages.AlfredVintageStore`). Narrative connectors
declare `PARSER_SCHEMA_VERSION` and call
`puremacro.narrative.sources._schema_check.assert_landmarks(...)` to
fail loudly on upstream layout drift. References:
`docs/CREDENTIALS.md`, `docs/CACHE_DB.md`.

### F2 closure (0.67.0+)

Slice B closes the F2 sub-project. `puremacro.narrative.sources._fallback`
centralises the live → wayback → playwright fallback chain that 7
connectors previously hand-rolled; each connector declares
`FALLBACK_POLICY` as a module-constant tuple of
`SUPPORTED_STAGES = {"live", "wayback", "playwright"}` and calls
`fetch_with_fallback(url, policy=FALLBACK_POLICY, source="<name>")`.
`puremacro.narrative.sources._telemetry.log_event` records one row per
fetch attempt to the new `connector_events` SQLite table; the 8
Slice-A schema-checked connectors also emit `parser_schema_mismatch`
events. `connector_health(window=, sources=)` aggregates into a
per-source DataFrame. Kill-switch: `PUREMACRO_NARRATIVE_TELEMETRY=0`.
Reference: `docs/CONNECTOR_HEALTH.md`.

### F1 Slice A — SE Asia + Africa CBs (0.68.0+)

First slice of F1 source-coverage expansion. Six new central-bank
connectors: `bi` (Indonesia), `bnm` (Malaysia), `bsp` (Philippines),
`cbn` (Nigeria), `cbe` (Egypt), `cbk` (Kenya). Each adopts the Slice
A schema-versioning contract + the Slice B fallback + telemetry
contracts from inception. Five of six use REST APIs (BNM Open API,
SharePoint OData, Kendo Grid, Sitecore, WordPress) discovered behind
their SPA frontends; only `bi` uses HTML scraping.
Subsequent F1 slices queued: B (business surveys), C (forecaster
surveys), D (alt-data).

#### Local LLM engines (`narrative/_local_engines.py`)

A lazily-imported engine layer lets the two LLM call sites run on a local model
at $0. `resolve_engine("auto")` selects `MLXEngine` (Apple GPU) → `LlamaCppEngine`
(GGUF) → `HTTPEngine` (Ollama / OpenAI-compatible, urllib). `LocalBackend`
(scoring) and `LocalProvider` (indices) are thin wrappers; `MODEL_ALIASES` maps a
friendly model name to per-engine ids. Heavy engine imports stay inside methods
so `import puremacro` remains Pyodide-clean; the `[local-llm]` extra carries the
optional packages.

### `puremacro.vfi` — heterogeneous-agent dynamic programming

A reusable, multi-backend toolkit (a Python port of the MATLAB VFIToolkit),
built as a composable pipeline:

1. **Discretize** an AR(1) shock: `tauchen` / `rouwenhorst` / `farmer_toda`;
   `combine_markov_chains` Kronecker-products independent chains for multi-shock
   models (e.g. persistent + transitory income); `markov_stationary` is the
   chain's ergodic distribution (`discretize.py`). `z_grid` may itself be a LIST
   of grids (multiple exogenous shocks entering the return fn separately), the
   symmetric pair to a list `a_grid`.
2. **Solve** the household problem: `VFIProblem(a_grid, z_grid, P_z, return_fn,
   beta, d_grid, options).solve(backend) -> VFISolution(V, policy_aprime,
   policy_d)`. `a_grid` may be a LIST of 1-D grids for K endogenous states
   (multi-asset): the engine flattens to the C-order product and the kernels run
   unchanged on the flat index; `VFISolution.policy_components()` decodes it.
   Because the policy is a flat index, the distribution, aggregate, GE, and
   permanent-types layers handle multi-asset unchanged (composability tested).
   The return tensor `R(d,a',a,z)` is broadcast on the active
   backend (`returnfn.py`); the driver runs pure Bellman iteration + Howard
   improvement (`kernels.py` + `solve.py`, xp-generic for numpy/mlx/cupy;
   `kernels_numba.py` is the compiled twin). Backend dispatch is the shared
   `puremacro/_backend.py`.
3. **Distribution**: `stationary_distribution(policy, P_z)` — the stationary
   agent measure via the Tan (2020) two-step (`distribution.py`). For CONTINUOUS
   policies (e.g. `solve_egm(...).aprime`), `lottery_distribution` does the same
   with fractional ("lottery") weights splitting each `a'` between its two
   bracketing grid nodes — the bridge from EGM into this layer.
4. **Aggregates / inequality**: `aggregate(fn, mu, ...)`, `lorenz_and_gini`,
   `weighted_quantile` (`aggregate.py`).
5. **General equilibrium**: `stationary_equilibrium(build_problem,
   market_residual, bracket)` — a market-clearing root-find over the above;
   `stationary_equilibrium_types` is the permanent-types variant (solves all
   types per price, clears on the type-weighted aggregate) (`equilibrium.py`).
6. **Finite horizon**: `FiniteHorizonProblem(...).solve()` — life-cycle backward
   induction, also K-asset (`a_grid` as a list) and with optional age-dependent
   `survival` probabilities (mortality: discount `beta*s_j`) (`finite_horizon.py`).
7. **Transition paths**: `transition_path(build_problem, implied_price_path,
   ...)` — deterministic perfect-foresight dynamics between steady states
   (`transition.py`); `push_distribution` is the shared one-step operator.
8. **Distributions over the solved model**: `life_cycle_distribution` /
   `cross_section` / `age_profile` (life-cycle), and `simulate_panel` /
   `empirical_distribution` (Monte Carlo agent panels).

Also: `Case2Problem` (exogenous a'-rule), `value_fn_iter_case1` (VFIToolkit-
signature shim), `estimate_method_of_moments` ((S)MM parameter estimation), and
`solve_permanent_types` (PType -- fixed ex-ante heterogeneity: solve a problem
per type via the `build_problem` seam, then type-weighted aggregates). These
four — endogenous state, exogenous shock, decision, permanent type — are the
VFIToolkit model dimensions. `olg_stationary_equilibrium` (+ `stationary_age_weights`,
`olg_aggregate`) is the overlapping-generations GE: it clears factor markets over
the cohort cross-section of a finite-horizon household, with endogenous labor
(extensive + intensive margins via the `d`=hours decision incl. 0) (`olg.py`).
And `solve_egm` -- the Endogenous Grid Method (Carroll 2006), a second solver
paradigm that inverts the Euler equation for fast, accurate continuous policies
on the one-asset CRRA income-fluctuation problem; `lottery_distribution` then
builds the stationary measure of such a continuous policy. The numba backend
also offers `options={"divide_and_conquer": True}` -- a monotone-policy
divide-and-conquer greedy step (O(n_a' log n_a), exact for supermodular
problems), the signature VFIToolkit accelerator. `EpsteinZinProblem`
(`epstein_zin.py`) is the engine's first NON-time-separable recursion: recursive
preferences that split risk aversion from the EIS (it reduces exactly to
time-separable CRRA when risk aversion = 1/EIS). `firm_dynamics.py` is a distinct
model class -- Hopenhayn (1992) industry dynamics: an optimal-stopping exit
option (`firm_value_with_exit`), a non-mass-conserving entry/exit stationary
measure (`firm_stationary_distribution`), and a free-entry price
(`free_entry_price`); there is no endogenous asset (the state is productivity).
`krusell_smith.py` is heterogeneous agents with AGGREGATE shocks: it folds mean
capital `K` into the exogenous state (so the household is a `VFIProblem`),
simulates the wealth distribution along an aggregate path, and iterates a
log-linear forecast rule `log K'=b0[Z]+b1[Z]log K` to a fixed point (R^2>0.99
approximate aggregation).

`examples.aiyagari_steady_state(...)` runs the whole pipeline in one call;
`examples.huggett_steady_state(...)` is a second worked GE (a pure-exchange bond
economy in zero net supply, clearing on aggregate net assets = 0);
`examples.life_cycle_profile(...)` is a third, finite-horizon, worked example
(hump-shaped earnings, no bequest, the canonical life-cycle asset hump);
`examples.two_asset_profile(...)` is a fourth (liquid + illiquid with an
adjustment cost) showcasing the multiple-endogenous-state API;
`examples.neoclassical_growth(...)` is a fifth (a faithful port of VFIToolkit's
stochastic neoclassical growth model, validated against its analytical steady
state) -- the first representative-agent model; `examples.hopenhayn_equilibrium(...)`
is a sixth (Hopenhayn 1992 firm entry/exit with free entry and selection).
Correctness is pinned by a cross-backend oracle, closed-form economics
(cake-eating, Brock-Mirman, 2-period life-cycle), and a full Aiyagari GE with a
Walras (goods-market) consistency check. Multiple endogenous states (K assets)
are supported in the infinite-horizon `VFIProblem` (flat C-order product).
Deferred (each its own spec): Case-2 / GE multi-asset, low-memory / refinement
accelerators, and exotic asset/type variants (riskyasset / experienceasset).

---

## Legitimate distinctions (don't accidentally merge these)

Some modules look like duplicates at first glance but serve different audiences. Document the distinction here when you keep both.

| Pair | Distinction | Where it lives |
|---|---|---|
| `plot.py` vs `plotting/` | `plot.py` is the Pyodide-pure, zero-styling grayscale plot path used by juno.sh / iPad notebooks. `plotting/` is the library API with `bw_style.py` and richer composition (publication figures). | both top-level |
| `regime_dates.py` vs `regimes.py` | `regime_dates.py` ships the hard-coded date dictionaries (`REGIMES_GLOBAL`, `REGIMES_US`, …) — pure constants. `regimes.py` ships the helpers (`breaks_to_regimes`, `dates_to_regimes`) that take such a dict and an index and return a categorical series. | both top-level |
| `volatility/sigma.py` vs `sigma/sigma_numpy.py` | `volatility/sigma.py` is the canonical 262-line port of MAV/SigmaObject.m (extended API: summary tables, PC1 share, Shapley decomposition, variance-premium accounting). `sigma/sigma_numpy.py` is a 64-line teaching stub — quadratic-form variance only, no decompositions. | two subpackages |

### `inference/legacy/` — retired at 0.44.0

The directory was deleted at 0.44.0 alongside `lp/lp_*.py`. For the record:
six of the eleven files were byte-identical shims re-exporting from non-legacy
siblings; four (`bootstrap.py`, `wild_bootstrap.py`, `block_bootstrap.py`,
`weak_iv.py`) were the canonical home for older bootstrap helpers consumed by
`lp/lp_state_dep.py` and `lp/lp_smooth.py`. All callers migrated at 0.44.0
before deletion. The outer tests (`tests/test_inference.py`, `test_block_bootstrap.py`,
`test_weak_iv.py`, `test_wild_bootstrap.py`) were updated simultaneously.

---

## Known consolidation candidates

These are tracked, not bugs: they sit in the design map as TODOs rather than smells.

The 0.43.0 + 0.44.0 releases retired the `svar/*`, `lp/lp_*.py`, and
`inference/legacy/*` shim layers. The remaining low-priority candidates:

- **`regress/lp.py`** — provides a convenient flat-DataFrame panel-LP interface
  (`y=`, `shock=`, `unit=`, `date=`, `controls=`) with Driscoll-Kraay SEs.
  Complementary to canonical `lp.panel.panel_lp` which expects multi-indexed DataFrames.
- **`lp/garch_utils.py`** — public helper used by `R1_02_lp_menu` to
  fit σ_t before calling lp-garch variants. Could be renamed to
  `_garch_utils.py` (private) once external notebook callers migrate.
- **`ProxySVARResult` axis** — resolved (standardized to `(H+1, n, n)`
  matching all other SVAR identification result classes).

---

## Conventions you should preserve

- **Public API is curated in `__init__.py`.** The top-level package only exports `__version__`; subpackages re-export their stable names. Don't add a re-export to `puremacro/__init__.py` without a reason.
- **Examples and notebooks import via the public API.** If an example or notebook needs a helper that lives in a private module, promote the helper instead of having the caller reach in. (See `narrative/replication/__init__.py` for the pattern.)
- **Test schemas, not exact values, for stochastic functions.** Most LP / SVAR tests check that the point estimate falls within a tolerance window of the truth, not exact equality. Replication-validation tests tighten this where the original paper reports the number.
- **Diagnostic errors over silent garbage.** When a numerical operation fails (singular `X'X`, non-PD Σ, BK violation), raise a `LinAlgError` / `BlanchardKahnError` whose message names the calling function and the likely cause.
- **Match `__version__` across `puremacro/__init__.py` and `pyproject.toml`.** A mismatch slipped through Phase 5 (the `__init__.py` was on 0.40.0 while `pyproject.toml` lingered at 0.12.1); both now read `0.40.0`. If you cut a release, update both.

---

## Out of scope (deliberately)

- **Build pipeline / CI.** Single-author research package; no CI is on by design. Run `pytest` locally before tagging.
- **Sphinx docs.** README + per-module / per-function docstrings are the doc surface. Add a section here instead of bolting on a docs site.
- **Backwards-compatibility shims.** The package is pre-1.0. Rename freely; update consumers in the same commit. The "promote a private helper" pattern applies even to API tightening. 0.43.0 demonstrated that the shim-and-deprecate pattern works cleanly: `svar/` shims shipped at 0.42.0 were deleted on schedule at 0.43.0 with no behaviour change for callers. Future releases can use the same pattern when the migration surface is large enough to warrant a one-release notice window.
