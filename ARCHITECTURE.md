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
│   └── lp_did.py          ← the one surviving `lp_*` module. The older
│                            absorbed-from-src/ variants (lp_jorda, lp_iv,
│                            lp_panel, lp_panel_dk, lp_state_dep, lp_smooth,
│                            lp_garch_state, lp_garch_in_mean, garch_utils)
│                            were retired at 0.44.0 along with
│                            `inference/legacy/`; see "retired at 0.44.0"
│                            below. This block described them as live
│                            until after 1.8.0.
│
├── inference/
│   ├── _ols_helpers.py    ← ols_hac — central OLS path
│   ├── _results.py        ← shared frozen-dataclass result types
│   ├── hac.py             ← Newey-West SE
│   ├── hac_fixed_b.py     ← Kiefer-Vogelsang fixed-b HAC
│   ├── dk.py              ← Driscoll-Kraay panel HAC
│   ├── bootstrap.py / block_bootstrap.py / moving_block.py / wild_bootstrap.py
│   ├── lp_block_bootstrap.py    ← absorbed
│   ├── weak_iv.py         ← Cragg-Donald, Kleibergen-Paap, AR, MSW
│   ├── over_id.py         ← Hansen-J, Stock-Yogo
│   ├── pesaran.py / swamy.py
│   ├── balanced_panel.py / newey_west.py
│   └── quandt_andrews.py / spec_curve.py    ← structural-break + sensitivity
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
│                            *_results.mat from Dynare. build.py (1.2.0)
│                            takes equilibrium conditions as a Python
│                            function, differentiates them by complex step
│                            and hands the matrices to klein_solve.
├── connectedness/         ← Diebold-Yilmaz spillover
├── forecast/              ← Diebold-Mariano, Giacomini-White, density eval
├── tests/                 ← Bai-Perron breaks + unit-root tests
│                            (distinct from project-level tests/ directory)
├── narrative/             ← fiscal-narrative IV (events, dedup, panel,
│                            replication); pypdf used lazily by sources/
├── nowcast/               ← kalman_dfm, mf_var, forecast combos, CRPS
├── gar/                   ← Growth-at-Risk: QAR, skew-t, FCI
├── did/                   ← Staggered DiD: CS, Sun-Abraham, BJS, SDID
├── causal/                ← Double / Debiased Machine Learning & causal inference
│   ├── dml.py             ← DML PLR (Chernozhukov et al. 2018) + cross-fitting
│   └── synthetic_control.py ← Synthetic control methods & placebo inference
├── dynpanel/              ← Arellano-Bond + Blundell-Bond dynamic-panel GMM
├── hfi/                   ← high-frequency monetary surprises (GK / NS / JK)
│
│ ── dynamic programming, continuous projection & spatial GE (3.3.0) ───
├── vfi/                   ← Discrete & continuous dynamic programming
│   ├── problem.py / solve.py / kernels.py / kernels_numba.py
│   ├── discretize.py / returnfn.py / distribution.py / aggregate.py
│   ├── equilibrium.py / finite_horizon.py / transition.py / olg.py
│   ├── egm.py / epstein_zin.py / firm_dynamics.py / krusell_smith.py
│   ├── collocation.py     ← Chebyshev orthogonal polynomial collocation
│   ├── fem.py             ← Finite Element Galerkin + Fischer-Burmeister
│   ├── splines.py         ← Shape-preserving cubic B-splines & Schumaker
│   ├── smolyak.py         ← Smolyak sparse grids for d in [2, 6]
│   ├── dcegm.py           ← Discrete Choice EGM + fast Upper Envelope filter
│   ├── continuous_distribution.py ← Young (2010) continuous density & Aiyagari GE
│   ├── continuous_transition.py   ← Non-linear MIT transitions + sequence Broyden
│   ├── analytic_gradients.py      ← Machine-precision IFT parameter Jacobians
│   ├── deep_macro.py      ← Deep Macro PINNs in pure NumPy + ergodic sampling
│   └── hjb_achdou.py      ← Implicit upwind M-matrix solver, adjoint KFE, continuous Aiyagari GE
├── trade/                 ← Quantitative international trade general equilibrium
│   ├── caliendo_parro.py  ← Caliendo-Parro (2015) exact hat algebra
│   ├── equilibrium.py / solver.py / optimal_tariffs.py / calibration.py
│   └── policy_analytics.py / scenarios.py / tables.py / plot.py
├── spatial/               ← Quantitative spatial economics & spatial econometrics
│   ├── allen_arkolakis.py ← Allen-Arkolakis (2014) spatial gravity GE
│   └── weights.py / models.py / hac.py / panel.py / lp.py / diagnostics.py
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
├── wavelet/               ← DWT / MODWT Haar variance decomposition
│   └── coherence.py       ← boundary-safe pairwise wavelet coherence
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
│   │                        ._http (UA override, SSL fallback, 30s
│   │                        timeout). Modules for FRED / ALFRED / SDMX
│   │                        (OECD / Eurostat / ECB / IMF SDMX-Central),
│   │                        FRED-states, EPU / GPR / WUI / JLN /
│   │                        Fernald, OECD-MEI / OECD-QNA / OECD-energy
│   │                        / OECD-FX / OECD-QNA-labor, WB pink-sheet,
│   │                        emissions, energy_transition, commodities,
│   │                        financial, ILOSTAT, Yahoo, plus a STL/X-13 seasonal
│   │                        helper (_seasonal.py — statsmodels lazy).
│   └── realtime/          ← Real-time central bank vintage connectors:
│                            Banxico, INEGI, Banco Central do Brasil (SGS),
│                            Banco Central de Chile, ALFRED, Bundesbank,
│                            ECB RTD, ONS, StatCan, OECD STES; offline
│                            .pmz cartridges and schema canaries.
├── build_panel.py         ← Orchestrates panel_Q + panel_M from fetch/*.
│                            Public entry: build_all(countries, fast,
│                            refresh). Imports arch lazily for GARCH-σ.
├── build_subnational_panel.py
│                          ← US state/county panel (QCEW / LAUS / CES);
│                            ships a build_all entrypoint of its own.
├── climate_panel.py       ← High-level multi-country climate panel builder
│                            merging emissions, energy balances, transition shares,
│                            and macro aggregates (public: build_climate_panel).
├── financial_panel.py     ← Macroprudential & international financial stability
│                            panel builder (yields, credit gap, property prices,
│                            financial conditions, commodities; build_financial_panel).
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
│ ── runtime adaptation (added 1.2.0) ──────────────────────────────
├── runtime/               ← What this machine can do, and how much to ask
│   │                        of it. The only place that knows the package
│   │                        might not be on a workstation.
│   ├── _capabilities.py   ← host (cpython/pyodide) + device class
│   │                        (workstation/tablet/browser) + the four
│   │                        capabilities that break on a tablet: sockets,
│   │                        parquet, threads, writable fs. Private module
│   │                        so `runtime.capabilities` names the function.
│   ├── budget.py          ← device-sized ceilings on n_boot / n_draws /
│   │                        n_grid / n_sim. Opt-in per call; no estimator
│   │                        consults it, so defaults are unchanged.
│   ├── transport.py       ← HTTP where there are no sockets. Patches
│   │                        `_http._request` + `fetch._http.requests` to
│   │                        run over the browser's XMLHttpRequest.
│   └── store.py           ← DataFrame ⇄ npz codec. The pyarrow-free data
│                            path; also the substrate for pocket/longrun.
├── pocket/                ← `.pmz` data cartridges: zip(manifest.json +
│                            one npz per frame) with sha256 + provenance.
│                            Pack where there is network, open where there
│                            is not. Reads with numpy + stdlib alone.
├── longrun/               ← Chunked, checkpointed jobs that survive an
│                            iPadOS app suspend. Per-item RNG seeding
│                            (`default_rng([seed, i])`) makes results
│                            invariant to chunking and to resumption.
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
| `dsge/klein` | **Mature** | QZ-based; BK condition enforced. `F` and the shock loadings `N`/`L` were corrected at 1.2.0 (see CHANGELOG) and are now pinned against closed-form solutions in `tests/test_dsge/test_klein_analytic.py`. |
| `dsge/build` | **Stable** | Model DSL + complex-step Jacobians -> `klein_solve`. Validated against the closed-form neoclassical growth model (full depreciation, log utility) to 1e-9 on every matrix. Complex-step needs an analytic residual function; the build cross-checks against finite differences and raises when it is not. |
| `runtime/*` | **Stable** | Detection is heuristic by necessity (no API reports "you are in Juno") and every field is overridable by environment variable. `budget` is opt-in: it changes no estimator default. |
| `pocket/*` | **Stable** | Format is versioned (`FORMAT_VERSION`) and self-describing. A transport format, not a trust boundary — checksums catch corruption, not tampering. |
| `longrun/*` | **Stable** | The invariance property (chunking and resumption do not change results) is the contract; `tests/test_longrun.py` pins it. |
| `state_space`, `mcmc` | **Mature** | Kalman filter w/ NaN handling; standard MCMC diagnostics. |
| `var/identify/*` | **Stable, but watch the bootstrap** | Cholesky bootstrap drops non-PD draws and warns above 5% failure rate. All public estimators return frozen-dataclass result-objects 0.42.0+. |
| `var/identify/panel` (`mean_group_svar` + `PanelSVARResult`) | **Stable** | Canova-Ciccarelli 2013 mean-group panel SVAR. Supports `cholesky` and `bq` natively; uses `safe_cholesky` + canonical `inference/bootstrap` — no `inference.legacy` dependency. |
| `var/identify/maxshare` (`identify_maxshare` + `MaxShareResult`) | **Stable** | Faust-Uhlig full pipeline: identification + FEVD + residual bootstrap. `ci` kwarg replaces legacy `q_lo`/`q_hi`. Low-level `maxshare(...)` / `news_maxshare(...)` unchanged. |
| `lp/panel` (`lp_panel_regime_interaction`) | **Stable** | Returns long-form `pd.DataFrame` keyed by `(h, regime)`. Ported from legacy at 0.43.0. |
| `lp/state_dep` (`lp_smooth_transition_irf`) | **Stable** | HAC analytic CIs (replaces legacy block-bootstrap CIs). Ported from legacy at 0.43.0. |
| `var/bvar`, `var/vecm`, `var/tvp` | **Stable** | All `(X'X)^{-1}` calls route through `_linalg.inv_xtx`. |
| `lp/state_dep`, `lp/smooth`, `lp/asymmetric`, `lp/quantile` | **Stable** | Replication-light; exercise these via the example scripts. |
| `inference/{lp_block_bootstrap, balanced_panel, moving_block, swamy, pesaran, ...}` | **Stable** | Phase-5 absorbs. `lp_block_bootstrap` lazy-imports `puremacro.teaching.panel_lm` (Phase 0). The duplicate `moving_block_bootstrap`, `swamy_test` and `pesaran_cce` modules were removed after 1.8.0 — each was a second copy of the module beside it, and the `moving_block_bootstrap` copy's default IRF imported `statsmodels`, a dev-only extra. |
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
| `build_panel`, `build_subnational_panel`, `climate_panel`, `financial_panel` | **Stable** | Orchestrators on top of `fetch/*`; idempotent against the disk cache. `build_panel` lazy-imports arch for the GARCH-σ derivation. `climate_panel` and `financial_panel` provide modular multi-country panels for decarbonization, energy transition, and macroprudential stability. |
| `sa/{stl, x13}` | **Stable** | STL fallback when X-13 is unavailable. Both lazy-import statsmodels (Phase 0). |
| `plotting/*` vs `plot.py` | **Stable / Stable** | Two co-existing presentation paths — see "Legitimate distinctions" below. |
| `cache`, `_http`, `_codes`, `regime_dates`, `regimes`, `scale` | **Stable** | Cross-cutting utilities. |
| `vfi/collocation`, `vfi/fem` | **Stable** | Chebyshev polynomial collocation (Gauss & Lobatto nodes) and Finite Element Galerkin projection with Fischer-Burmeister complementarity for borrowing constraints. |
| `vfi/splines`, `vfi/smolyak` | **Stable** | Shape-preserving cubic B-splines, Schumaker shape-preserving quadratic splines, and Smolyak multi-dimensional sparse grids ($d \in [2, 6]$). |
| `vfi/dcegm` | **Stable** | Iskhakov-Jørgensen-Rust-Schjerning (2017) DC-EGM with fast Upper Envelope sub-optimal branch pruning and extreme value taste shocks. |
| `vfi/continuous_distribution` | **Stable** | Young (2010) non-stochastic continuous density simulation preserving mass conservation ($\sum \mu = 1.0 \pm 10^{-15}$) and Aiyagari continuous GE. |
| `vfi/continuous_transition` | **Stable** | Non-linear transition paths under unexpected MIT shocks combining backward EGM with forward Young operators via sequence-space Broyden. |
| `vfi/analytic_gradients` | **Stable** | Exact parameter Jacobians $\nabla_\theta c^*$ via the Implicit Function Theorem in a single linear solve ($60\times+$ faster than finite differences) for GMM/SMM. |
| `vfi/deep_macro` | **Stable** | Physics-Informed Neural Networks (PINNs) in pure NumPy for ultra-high-dimensional dynamic models (10+ states) with ergodic sampling. |
| `vfi/hjb_achdou` | **Mature** | Canonical implicit upwind finite-difference scheme (sparse M-matrix system $(\rho I - A^n) v^{n+1} = u(c^n)$), adjoint continuous-time KFE stationary distribution $g(a, z)$, and continuous Aiyagari GE. |
| `causal/dml` | **Stable** | Double / Debiased Machine Learning (Chernozhukov et al. 2018) for partially linear regression (`DMLPLR`) with $K$-fold cross-fitting and pure-NumPy regularized learners. |
| `fetch/realtime/{banxico, inegi, bcb, bcch}` | **Stable** | Latin America real-time central bank and statistical agency data connectors with offline `.pmz` cartridges and schema canaries. |
| `trade/caliendo_parro` | **Stable** | Multi-country, multi-sector trade general equilibrium with input-output linkages, intermediate goods, and tariffs solved via exact hat algebra (`CaliendoParroModel`). |
| `spatial/allen_arkolakis` | **Stable** | Continuous geographic general equilibrium with bilateral iceberg trade costs, labor mobility, agglomeration, and congestion (`AllenArkolakisModel`). |
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
```

The first four are the **Pyodide import core**: the only third-party modules a shippable estimator module may import at top level. The fifth widens the *install* contract, not the import contract:

- `requests` — the whole `puremacro.fetch` layer (OECD/SDMX, EPU, FRED-CSV, IMF, BEA) and the narrative sources `import requests` at module level by design. Without it a clean `pip install puremacro` dies with `ModuleNotFoundError` on the first fetch call. It is pure Python and part of the Pyodide distribution (there are no sockets there, but the offline CSV paths never touch it).

Every base dependency must also **ship with the Pyodide distribution itself**. The JupyterLite playground installs with `%pip install puremacro` and PyPI fallback disabled, so a base dependency that Pyodide lacks breaks the first cell of every notebook. `tests/test_pyodide_compat.py::test_runtime_deps_ship_with_pyodide` enforces it.

### Optional file-format engines: the `io` extra

`pip install "puremacro[io]"` adds the two engines pandas needs to read files; `data` and `dev` include them as well.

- `pyarrow` — the parquet engine behind `pandas.read_parquet` / `to_parquet`: `cache.py` (which falls back to the pyarrow-free `.pmz` store without it), the `fetch/labor*.py` caches, `shock_atlas.py`, `build_panel` / `build_subnational_panel`, `climate_panel`, and the ENOE datasets read by course lessons 08 and 24. Pyodide 314.x distributes it; 0.28 does not.
- `openpyxl` — the `.xlsx` engine behind `pandas.read_excel`, needed by **eighteen** shipped modules: `fetch/epu.py`, `epu_states.py`, `epu_news_historical.py`, `wui.py`, `wui_extras.py`, `jln.py`, `lmn.py`, `fernald.py`, `gpr.py`, `wb_pink_sheet.py`, `labor.py`, `oecd_qna_local.py`, `longpanel/ine_es.py`, `realtime/ons.py`, plus `shock_atlas.py`, `long_panel.py`, `narrative/sources/us_warn.py` and `narrative/validation/external_benchmarks.py`. Pure Python, but in no Pyodide distribution.

Both were base dependencies until 3.3.0: `openpyxl` because a bare install once returned silently incomplete panels, `pyarrow` because the documented student install was a bare `pip install puremacro`. They moved because, as base dependencies, they made the playground's install unresolvable (it failed on the live site through 3.2.1). The failure modes that justified promoting them are handled directly instead:

- **Silent partial panels.** Inside `build_all` each producer is wrapped in `try/except Exception: print(...)`, so a missing `openpyxl` once produced a panel silently missing `epu_m`, `epu_q`, `wui_q`, `wui_m`, `jln_macro_*`, `lmn_real`, `lmn_fin`, `tfp_fernald` and every Pink Sheet series. `build_all`, `shock_atlas.load_all_shocks` and `build_climate_panel` now check for the engines they need before doing any work and raise `MissingEngineError` (an `ImportError`) that names the extra. Direct `read_parquet` / `read_excel` calls in fetchers and loaders already fail loudly with pandas' own `ImportError`.
- **The course install.** The course's install instruction is now `pip install "puremacro[io]"` (syllabus, lessons 08 and 24).

**The opt-in Pyodide gate** (`python tools/release_check.py --pyodide`, gate 6) installs the built wheel the way the playground does: `tools/pyodide/runner.js` preloads only `pytest` and `micropip`, runs a dependency-resolving `micropip.install("emfs:/tmp/<wheel>")`, and fails if any dependency resolved from PyPI rather than the Pyodide distribution. It then runs the `pyodide_smoke`-marked tests (31 green under Pyodide 0.28.3 as of 3.3.0), covering the estimator core plus the runtime / pocket / longrun / dsge.build additions. Through 3.2.1 the runner had to install with `deps=False`, because `pyarrow` made a dependency-resolving install impossible, and that is how the playground's broken install went unnoticed.

The in-process guarantee is separate: the two sweeps in `tests/test_pyodide_compat.py` check the import contract, and the subprocess sweep also blocks `pyarrow` and `openpyxl`, so every shippable module must import without the `io` extra.

Anything else is **dev-only or extra-only** (pytest, statsmodels, linearmodels, arch, pypdf, beautifulsoup4, pdfplumber, and the `io` engines pyarrow and openpyxl). These are declared in `[project.optional-dependencies]`:

- `io = ["pyarrow", "openpyxl"]` — file-format engines (see above); also part of `data` and `dev`.
- `dev = ["pytest", "statsmodels", "linearmodels", "arch", ...]` — parity tests, plus the `io` engines so the suite runs with them.
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

## Result-Object Presentation Standard (2.0+)

All public estimators that return three or more fields (or any non-trivial diagnostic) MUST return a frozen dataclass result object adhering to the unified presentation interface:

1. **`@dataclass(frozen=True)`** for any return with 3+ fields or non-trivial diagnostics.
2. **Naming:** `<MethodName>Result` (or `<Problem>Solution`) in PascalCase (e.g., `GMMResult`, `IRFResult`, `JKResult`, `ProxySVARResult`, `VFISolution`, `HJBSolution`). Defined in `<subpackage>/_results.py` (or module); re-exported via `<subpackage>/__init__.py`.
3. **Tuple returns** still allowed for genuinely simple two-value returns (`cycle, trend = hamilton_filter(y)`).
4. **Common field vocabulary**: `coefs`, `se`, `cov`, `names: tuple[str, ...]`, `n_obs`, `converged`.
5. **`.summary() -> pd.DataFrame | str`**: Tabular analytical and diagnostic summary of the model or estimation.
6. **Unified `.plot()` Contract (Headless & WASM-Safe)**:
   Estimators/solvers with dynamic trajectories, distributions, policy functions, or impulse responses implement `.plot()` conforming to strict headless constraints:
   - **Lazy Matplotlib Import**: `import matplotlib.pyplot as plt` inside `.plot()`, never at module level.
   - **No Unsolicited Displays**: Never call `plt.show()` unless explicitly passed `show=True` (default: `False`).
   - **Caller Axes Injection**: Accept `ax=...` (single axis or sequence of axes). If provided, plot into caller's axis and return `ax`.
   - **Standalone Figure Creation**: If `ax is None`, create subplots via `plt.subplots(...)` and return the `fig`.
   - **Headless / Pyodide / WASM Safety**: Operates seamlessly with the `Agg` backend without relying on windowing systems, desktop displays, or interactive loop hooks.
7. **Export Parity Quintet**:
   Result objects expose `.to_dataframe() / .to_frame()`, `.to_markdown()`, `.to_latex()`, and `.to_typst()`. All tabular formatters route through `puremacro.reports` (`df_to_markdown`, `df_to_latex`, `df_to_typst`) to guarantee escaping of LaTeX and Typst special characters and numeric stability.
8. **No `__post_init__` validation that raises.** The estimator builds a valid result; the dataclass just stores it.
9. **DataFrame carve-out.** Functions returning a single `pandas.DataFrame` with named columns (e.g. every `lp/` estimator) do NOT need to wrap. The DataFrame is already self-documenting.

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

### Real-time / vintage layer (1.7.0+, 3.4.0+)

`puremacro.fetch.realtime` is a provider registry, not a single
fetcher. Central bank and statistical agency repositories publish previously-released
values in different shapes; each gets a module exposing a **pure parser**
(bytes → `[date, vintage, value]`, no I/O, offline-testable) plus a
network wrapper, and every result is funnelled through
`_base.normalize_vintage_frame` into one tidy schema
(`country, variable, date, vintage, value, provider, series_id, units`).
`vintage_panel` is the single entry point; `VintagePanel` carries the
revision helpers, which delegate to `puremacro.vintages`.

At 3.4.0, first-class Latin American providers join the registry:
- `banxico`: Banco de México (SIE API) real-time series and revision tracking.
- `inegi`: INEGI (Mexico) national accounts, inflation, and activity indicators.
- `bcb`: Banco Central do Brasil (SGS API) policy rates and aggregates.
- `bcch`: Banco Central de Chile statistical database series.

All providers support offline SQLite caching in `_cache_db` (`realtime_vintages` table)
and `.pmz` offline package cartridges (`cartridge.py`). Active schema canaries
(`canary.py`) run health checks detecting upstream endpoint and layout drift.

Three invariants are load-bearing and easy to break:

1. **Transforms apply within a vintage column, never across.** The
   growth rate a forecaster saw in edition *v* is built from *v*'s own
   levels. This is also what makes the tests immune to rebasing: a
   whole-edition rescaling cancels out of that edition's growth rates.
2. **Units live in the catalogue.** FRED carries both level and
   growth-rate series for the same concept and the identifier does not
   say which; `SeriesSpec.units` decides the default transform so a
   growth rate is not differenced twice.
3. **Vintage-date semantics differ by provider** and are recorded in
   `_base.VINTAGE_SEMANTICS`. Some are genuine national release dates
   (Bundesbank, StatCan, Banxico), some are snapshot months (OECD), one is a
   month-plus-stage label with no day (ONS). Ordering is safe
   everywhere; event-dating is not.

The catalogue is self-auditing via
`tests/test_realtime_providers/test_catalog_live.py` (`-m network`),
because identifiers rot: every non-US country previously resolved to
FRED's OECD-MEI family, which stopped updating in January 2024, and no
offline test could notice. `openpyxl` is needed only by the ONS
workbook reader and is lazy-imported there, so the Pyodide import
contract is unchanged. Reference: `docs/real_time_data.md`.

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

### Continuous Dynamic Programming, Spatial & Trade Architecture (3.3.0)

Milestone 3.3.0 introduces nine high-performance engines spanning continuous state-space dynamic programming, deep macro physics-informed machine learning, and quantitative spatial/trade general equilibrium, all implemented under the strict Pyodide 4-package contract (pure NumPy/SciPy/Pandas/Matplotlib):

#### 1. Continuous Projection Engines: Chebyshev Collocation & Finite Element Galerkin (`vfi/collocation.py`, `vfi/fem.py`)
- **Chebyshev Polynomial Collocation** (`CollocationProblem`): Approximates continuous value functions $V(k)$ or policy rules $c(k)$ via orthogonal Chebyshev polynomials of the first kind $T_n(x) = \cos(n \arccos x)$ defined on the canonical domain $[-1, 1]$. Linearly maps asset domains $[k_{\min}, k_{\max}] \leftrightarrow [-1, 1]$. Collocation nodes are chosen at Gauss-Chebyshev roots $x_k = \cos\left(\frac{2k-1}{2n}\pi\right)$ or Gauss-Chebyshev-Lobatto extrema $x_k = \cos\left(\frac{k\pi}{n}\right)$. Evaluates multidimensional tensor products with fast DCT-based projections. Solves Euler residual conditions $\mathcal{R}(k; \theta) = u'(c(k)) - \beta \mathbb{E}[u'(c(k'))(f'(k') + 1 - \delta)] = 0$ via Powell or Newton-Krylov root-finding.
- **Finite Element Galerkin Projection** (`FEMProblem`): Partitions the continuous state space into localized piecewise-polynomial elements using tent/hat basis functions $\phi_i(k)$ with compact support $\text{supp}(\phi_i) = [k_{i-1}, k_{i+1}]$. Enforces Galerkin orthogonality $\int_{k_{\min}}^{k_{\max}} \mathcal{R}(k; \mathbf{c}) \phi_i(k) \, dk = 0$ via Gauss-Legendre quadrature.
- **Fischer-Burmeister Non-Smooth Complementarity**: Handles borrowing constraints and inequality kinks ($k' \ge \bar{k}$ with multiplier $\lambda \ge 0$ and $\lambda(k' - \bar{k}) = 0$) using the smoothed Fischer-Burmeister NCP operator:
  $$\Psi_{\text{FB}}^\epsilon(a, b) = a + b - \sqrt{a^2 + b^2 + 2\epsilon} = 0$$
  where $a = k' - \bar{k}$ and $b = u'(c) - \beta \mathbb{E}[u'(c') R']$. This smooth reformulation eliminates combinatorial Kuhn-Tucker regime-switching loops and guarantees global convergence in Newton iterations.

#### 2. Shape-Preserving Splines & Smolyak Sparse Grids (`vfi/splines.py`, `vfi/smolyak.py`)
- **Shape-Preserving Splines** (`CubicBSpline`, `SchumakerSpline`): Standard cubic splines often suffer from Runge phenomena and spurious oscillations near policy kinks. puremacro implements:
  1. *Cox-De Boor B-splines* with stable recurrence and tridiagonal linear solves for $C^2$ smooth regions.
  2. *Schumaker (1983) quadratic splines* with adaptive knot insertion that strictly preserve monotonicity ($\partial g / \partial k \ge 0$) and concavity ($\partial^2 g / \partial k^2 \le 0$) over the entire domain, ensuring economically coherent consumption and savings functions.
- **Smolyak Sparse Grid Collocation** (`SmolyakGrid`): Resolves the curse of dimensionality for dynamic models with $d \in [2, 6]$ continuous state variables. Instead of full tensor grids of size $\mathcal{O}(N^d)$, Smolyak sparse grids select multidimensional polynomial combinations based on the index set:
  $$\mathcal{I}(d, \mu) = \left\{ \mathbf{i} \in \mathbb{N}^d : d \le \sum_{j=1}^d i_j \le d + \mu \right\}$$
  where $\mu$ is the approximation level. Combined with nested Clenshaw-Curtis extrema nodes $m(1)=1, m(i)=2^{i-1}+1$, points are reused across levels, reducing grid size to $\mathcal{O}(N (\log N)^{d-1})$ while preserving high polynomial precision.

#### 3. Discrete Choice EGM & Fast Upper Envelope Filtering (`vfi/dcegm.py`)
- **DC-EGM Architecture**: Extends the Endogenous Grid Method (Carroll 2006) to non-convex dynamic programming problems with discrete choices (e.g. labor force participation, retirement) and continuous savings (Iskhakov, Jørgensen, Rust & Schjerning 2017).
- **Upper Envelope Algorithm**: Discrete switches induce non-concave value functions, causing the Euler equation inversion to produce non-monotonic, multi-valued endogenous asset correspondences where multiple candidate savings decisions satisfy the first-order condition. The Upper Envelope filter:
  1. Traverses the endogenous grid in order of post-decision assets $a'$.
  2. Detects backward-bending branches ($M_{t, i+1} < M_{t, i}$).
  3. Computes the candidate value $v_t(M, d) = u(c) + \beta \mathbb{E}[V_{t+1}(M')]$ along each branch.
  4. Discards dominated, sub-optimal branches to extract the global upper envelope $V_t(M) = \max_d \{ v_t(M, d) \}$.
- **Taste Shock Integration**: Extreme value type-I (Gumbel) taste shocks yield closed-form smooth choice probabilities via log-sum-exp softmax aggregations:
  $$P(d | M) = \frac{\exp(v_t(M, d)/\sigma_\epsilon)}{\sum_{d'} \exp(v_t(M, d')/\sigma_\epsilon)}, \quad V_t(M) = \sigma_\epsilon \ln \left( \sum_d \exp(v_t(M, d)/\sigma_\epsilon) \right)$$

#### 4. Young (2010) Continuous Stationary Distributions & General Equilibrium (`vfi/continuous_distribution.py`)
- **Non-Stochastic Density Simulation**: Replaces slow Monte Carlo simulations with Young's (2010) non-stochastic distribution operator on fine continuous grids. Given continuous policy function $g(k, z)$, each point mass at $(k_i, z_j)$ is mapped to $k' = g(k_i, z_j)$.
- **Exact Machine-Precision Mass Conservation**: The policy $k'$ is bracketed by adjacent grid nodes $k_l \le k' \le k_{l+1}$ and partitioned using linear interpolation lottery weights:
  $$\omega_l = \frac{k_{l+1} - k'}{k_{l+1} - k_l}, \quad \omega_{l+1} = \frac{k' - k_l}{k_{l+1} - k_l}$$
  This forward transition matrix $T^*$ is strictly column-stochastic. The stationary distribution $\mu^*$ satisfies $(I - T^*) \mu^* = 0$ subject to $\sum \mu^* = 1$, preserving aggregate mass to floating-point precision ($\sum \mu^* = 1.0 \pm 10^{-15}$). Solved via sparse iterative power iteration or direct sparse linear systems.
- **Aiyagari Continuous General Equilibrium**: Outer root-finding loop solves for the equilibrium rental rate $r^*$ such that continuous capital supply $K^s(r) = \sum_{i,j} k_i \mu^*(k_i, z_j; r)$ clears aggregate Cobb-Douglas capital demand $K^d(r) = \left(\frac{r + \delta}{\alpha}\right)^{\frac{1}{\alpha-1}} L$.

#### 5. Continuous Transition Dynamics Under MIT Shocks (`vfi/continuous_transition.py`)
- **Non-Linear Sequence-Space Formulation**: Solves deterministic transition paths over horizon $t \in [0, T]$ following unexpected aggregate shocks (e.g. TFP shocks, fiscal innovations).
- **Coupled Backward-Forward Iteration**:
  - *Backward Step*: Given an anticipated price sequence $\{r_t, w_t\}_{t=0}^{T-1}$, solves time-varying continuous Euler equations backwards from terminal steady state $g_T = g_{SS}$ to obtain time-dependent policy rules $g_t(k, z)$ for each $t$.
  - *Forward Step*: Propagates the initial continuous wealth distribution $\mu_0 = \mu_{SS}$ forward in time using time-varying Young operators: $\mu_{t+1} = T_t^*(g_t) \mu_t$.
  - *Aggregation*: Integrates capital supply $K_t^s = \int k \, d\mu_t$ at every period.
- **Broyden Quasi-Newton Solver**: Updates the aggregate path $\{K_t\}_{t=1}^T$ to clear goods and factor markets $\mathcal{H}_t(\{K_s\}) = K_t^s - K_t^d = 0$ via sequence-space Broyden updates, bypassing expensive $T \times T$ numerical Jacobian re-evaluations.

#### 6. Exact Analytic IFT Jacobians & Structural Estimation (`vfi/analytic_gradients.py`)
- **Implicit Function Theorem Formulation**: For continuous dynamic models parameterized by structural vector $\theta = (\beta, \gamma, \alpha, \delta, \rho, \sigma)$, the optimal continuous policy $c^*$ satisfies the continuous Euler residual system $\mathcal{R}(c^*; \theta) = \mathbf{0}$. By the Implicit Function Theorem:
  $$\nabla_\theta c^* = - \left[ \frac{\partial \mathcal{R}}{\partial c^*} \right]^{-1} \frac{\partial \mathcal{R}}{\partial \theta}$$
- **Single-Solve Efficiency**: The Jacobian $\partial \mathcal{R}/\partial c^*$ is a tridiagonal or sparse banded matrix. Computing $\nabla_\theta c^*$ requires only a single linear solve per parameter rather than re-solving the full dynamic program, achieving $60\times+$ computational speedups over finite differences.
- **Structural GMM & SMM Estimation**: Propagates exact policy derivatives into moment Jacobians $\nabla_\theta m(\theta)$, enabling fast and robust gradient-based optimization (L-BFGS-B, Gauss-Newton) for structural macroeconometric estimation.

#### 7. Deep Macro & Physics-Informed Neural Networks (`vfi/deep_macro.py`)
- **Pure NumPy MLP Architecture**: Implements neural network approximations of high-dimensional macroeconomic decision rules $c = g_\phi(\mathbf{s})$ entirely in pure NumPy without TensorFlow or PyTorch dependencies, running natively under Pyodide and WebAssembly.
- **Feasibility-Constrained Output Activations**: Hard-codes physical resource constraints directly into network architectures:
  $$c = \text{softplus}(z_c) \cdot W, \quad k' = \text{sigmoid}(z_k) \cdot W$$
  ensuring that consumption is strictly positive, capital is non-negative, and the budget constraint $c + k' \le W$ is satisfied by construction everywhere in state space.
- **Physics-Informed Loss Function**: Trains network weights $\phi$ by minimizing the mean squared residual of economic equilibrium conditions (Euler equations, market clearing):
  $$\mathcal{L}(\phi) = \frac{1}{B} \sum_{b=1}^B \left\| u'(c_b) - \beta \mathbb{E}\left[ u'(c_{b+1}) R_{b+1} \right] \right\|^2$$
- **Ergodic Trajectory Sampling**: Avoids the exponential curse of grid-based discretization by sampling training points $\mathbf{s}_b$ along simulated ergodic paths of the economic system (Maliar, Maliar & Winant 2021), enabling accurate solutions for models with 10+ continuous state variables (e.g. multi-country neoclassical growth).

#### 8. Quantitative Trade General Equilibrium: Caliendo-Parro (2015) (`trade/caliendo_parro.py`)
- **Exact Hat Algebra**: Solves multi-country ($N$), multi-sector ($J$) trade general equilibrium models with input-output linkages, intermediate goods, and tariffs without requiring estimation of unobserved technology levels or bilateral trade costs.
- **Governing Equations in Changes**:
  1. *Sectoral Prices*: $\hat{P}_{n,j} = \left[ \sum_{i=1}^N \pi_{n,i,j} \left( \hat{c}_{i,j} \hat{\tau}_{n,i,j} \right)^{-\theta_j} \right]^{-1/\theta_j}$
  2. *Unit Costs*: $\hat{c}_{i,j} = \hat{w}_i^{\beta_{i,j}} \prod_{k=1}^J \hat{P}_{i,k}^{\gamma_{i,k,j}}$
  3. *Expenditure Shares*: $\hat{\pi}_{n,i,j} = \left( \frac{\hat{c}_{i,j} \hat{\tau}_{n,i,j}}{\hat{P}_{n,j}} \right)^{-\theta_j}$
  4. *Goods Market Clearing & Trade Balances*: Incorporates tariff revenue, deficit adjustments, and cross-sector demand linkages.
- **Numerical Solver**: Employs a damped fixed-point iteration on wages $\hat{\mathbf{w}}$ and prices $\hat{\mathbf{P}}$ with adaptive acceleration, guaranteeing convergence to the unique trade equilibrium.

#### 9. Quantitative Spatial Economics: Allen-Arkolakis (2014) (`spatial/allen_arkolakis.py`)
- **Economic Geography Gravity GE**: Formulates general equilibrium across discrete spatial locations $i \in \{1, \dots, N\}$ on a continuous geographic plane with bilateral iceberg transport costs $\tau_{i,j} \ge 1$, labor mobility, Marshallian agglomeration spillovers ($\alpha$), and amenity congestion ($\beta$).
- **Equilibrium System**:
  1. *Gravity Trade Flows*: $X_{i,j} = \frac{(w_i \tau_{i,j})^{-\theta}}{\sum_k (w_k \tau_{k,j})^{-\theta}} E_j$
  2. *Spatial Wage Equation*: $w_i^{1 + \theta \sigma} L_i^{\theta \alpha} = \sum_{j=1}^N \frac{\tau_{i,j}^{-\theta} w_j L_j}{\sum_k (w_k \tau_{k,j})^{-\theta}}$
  3. *Labor Indifference & Utility Equalization*: $U_i = \frac{w_i A_i L_i^\beta}{P_i} = \bar{U}$
- **Counterfactual Simulations**: Evaluates infrastructure investments (reductions in $\tau_{i,j}$), regional productivity shocks, and amenity policies, computing counterfactual population reallocations $\hat{L}_i$, wage changes $\hat{w}_i$, and aggregate welfare effects $\Delta \ln W$.

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

- **Build pipeline / CI.** CI has existed since v0.92.0 and this line did not keep up. `.github/workflows/ci.yml` runs the suite across {ubuntu, macos, windows} x Python {3.11, 3.12, 3.13}, then the release gate and a strict `mkdocs build` on one leg; `release.yml` publishes to PyPI on a `v*` tag via Trusted Publishing; `pages.yml` deploys the JupyterLite playground. Still run `pytest` locally before tagging — CI runs the same default marker set, so the `slow`, `network`, `reference` and `replication` suites run nowhere but on your machine.
- **Sphinx docs.** README + per-module / per-function docstrings are the doc surface. Add a section here instead of bolting on a docs site.
- **Backwards-compatibility shims.** The package is past 1.0 — 1.8.0 as of this writing — so this is no longer "rename freely". Every name in `tests/fixtures/public_api_snapshot.json` is covered by release gate 3, and `docs/1.0_path.md` promises a `DeprecationWarning` naming the replacement one minor release before removal. Private helpers remain free to move. The "promote a private helper" pattern applies even to API tightening. 0.43.0 demonstrated that the shim-and-deprecate pattern works cleanly: `svar/` shims shipped at 0.42.0 were deleted on schedule at 0.43.0 with no behaviour change for callers. Future releases can use the same pattern when the migration surface is large enough to warrant a one-release notice window.
