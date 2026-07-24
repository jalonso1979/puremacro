# Changelog

This file records user-visible changes to the `puremacro` package at a
release / feature level. It documents the library's public capabilities; it is
not a development diary.

## Unreleased

## 0.94.0 — 2026-07-24

### Added — unit-root tests
- **GLS-detrended unit-root tests** (`tests.unit_root.dfgls_test`,
  `ng_perron_test`): the DF-GLS test (Elliott, Rothenberg & Stock 1996) and the
  Ng-Perron (2001) M-tests (MZa / MZt / MSB / MPT), joining the existing
  ADF / KPSS / PP / Zivot-Andrews suite. Local-to-unity GLS detrending
  (c̄ = −7 constant / −13.5 trend) delivers substantially more power near the
  unit root than ADF — e.g. on 1889–2015 log US real GDP, ADF cannot reject a
  unit root but DF-GLS does at 10%. Three new validation cases (gallery → 66).

### Added — teaching notebooks
- Showcase notebooks **17** (identification spec-curve), **18** (Beveridge
  curve), **19** (Model Confidence Set), and **20** (unit roots with power),
  each in English and Spanish, with their frozen offline datasets.

## 0.93.0 — 2026-07-24

### Added — seasonal adjustment
- **Native X-11/ARIMA engine** (`sa.x11`): a pure-Python
  Shiskin-Young-Musgrave X-11 decomposition with an airline-model regARIMA
  fore/backcast extension, validated against the real X-13ARIMA-SEATS binary
  via frozen goldens. `sa.deseasonalize_x13` now falls back
  binary → native X-11 → STL, so the Pyodide/browser build gets a genuine
  X-11-class adjuster instead of the STL last resort. Exposes `x11_arima`,
  `deseasonalize_x11`, `henderson_weights`, and the frozen `X11Result`.

### Added — forecast evaluation
- **Model Confidence Set** (`forecast.mcs.model_confidence_set`): the Hansen,
  Lunde & Nason (2011) bootstrap elimination procedure — stationary/moving
  block bootstrap of the equivalence statistic (t_max range / semi-quadratic),
  iterative removal of the worst model, running-max MCS p-values, returning the
  set of models statistically indistinguishable from the best. The multi-model
  generalization of the pairwise Diebold-Mariano / Giacomini-White tests in
  `forecast.compare`; `losses_from_forecasts` builds the loss matrix. The
  validation gallery is now 63 cases.

### Added — data fetchers (2026-07-22)
- **JOLTS** (`fetch.jolts.fetch_jolts`): openings, hires, quits, layoffs &
  discharges and total separations — level and rate, SA or NSA, monthly from
  2000-12 — via the key-free FRED fredgraph mirror; total nonfarm plus 13
  live-verified industry supersectors (unverified BLS codes deliberately
  absent; raw 4-digit codes accepted and fail loudly).
- **Eurostat job vacancies** (`fetch.vacancies_eurostat.
  fetch_eurostat_vacancies`): `jvs_q_nace2` through the built-in SDMX layer —
  JVR / JOBVAC / JOBOCC by NACE aggregate and size class, SA or NSA with no
  silent substitution, ISO-3 geo normalization, `csv_path=` for offline use.
  The natural pair with JOLTS for cross-country Beveridge curves.

### Added — econometrics
- **Sup-t simultaneous confidence bands** (`inference.supt`): Montiel Olea &
  Plagborg-Møller (2019) plug-in / bootstrap / Bayes constructions with a
  frozen `SupTBandResult`, wired additively into `var.bootstrap.bootstrap_bands`
  (`band='sup-t'`) and the cumulative-LP block bootstrap. New analytic
  validation case; the validation gallery is now 62 cases.
- **Narrative sign restrictions** (`var.identify.narrative_sign`): Antolín-Díaz
  & Rubio-Ramírez (2018) Type I shock-sign and Type II/III
  historical-decomposition-dominance restrictions with AD-RR importance
  weights, Kish ESS diagnostics, and a `NarrativeEvent` adapter.
- **LP-DiD** (`lp.lp_did`): Dube-Girardi-Jordà-Taylor (2023) local-projections
  difference-in-differences — long-difference event studies on newly-treated vs
  clean controls, equal- and variance-weighted ATTs, free pre-trend horizons,
  clean-control attrition diagnostics, frozen `LPDiDResult`.
- **Generalized IRFs for regime models** (`var.regime.girf`): Koop-Pesaran-Potter
  (1996) generalized impulse responses for TVAR / TVECM / MS-VAR with endogenous
  regime switching, per-regime GIRFs, a bootstrap band on the between-regime
  difference, and Kilian-Vigfusson size/sign asymmetry. Validated against the
  linear VAR IRF in the identical-regime limit to machine precision.
- **District-level Beige Book uncertainty** (`narrative.indices` +
  `_fed_districts`): a 12-district crosswalk and `bbui(level='district',
  output='tidy')`; the Beige Book connector now parses the modern per-district
  pages, the ~2011–2023 single-page layouts, and the 1996–2010 FOMC-era archive.

### Added — replication examples
- `examples/gali_1999_hours.py` — Galí (1999) technology shocks via long-run
  (Blanchard-Quah) identification, on frozen FRED data.
- `examples/kilian_2009_oil.py` — Kilian (2009) oil-market VAR, the first
  example to exercise `var.irf.historical_decomp`.
- `examples/ramey_zubairy_2018_multipliers.py` — Ramey-Zubairy (2018)
  state-dependent fiscal multipliers with the weak-IV pedagogy (Olea-Pflueger
  effective F, Anderson-Rubin / MSW robust bands).
- `examples/narrative_sign_adrr.py` — narrative sign restrictions on a
  monetary VAR (a Volcker-episode restriction tightening the identified set).
- New teaching notebooks: **14** (the tax multiplier three ways), **15** (DiD
  meets local projections), **16** (state-dependent transmission with GIRFs),
  each with an English/Spanish twin.

### Fixed
- Fertility DSGE (`dsge.fertility_adj_costs`) now linearises around the exact
  steady state of its pinned calibration (new public `exact_steady_state` +
  `FERTILITY_PINNED_CALIBRATION`), so the Blanchard-Kahn check is robust across
  BLAS/LAPACK builds. `solve_bgp` is retained for reference.
- `var.identify.hetero`: fixed a dead import that prevented `rigobon_svar` from
  producing bootstrap bands.
- pandas 3.0 compatibility (copy-on-write safe array handling in the
  Smets-Wouters helper; categorical date labels in the narrative report).
- Lazy `fredapi` import in `fetch.fred` (keeps the core import Pyodide-clean).
- SW07 estimation tests replaced brittle short-chain closeness checks with
  platform-robust sampler contracts; the full Table-1A replication is retained
  as an opt-in long-chain variant.

### Packaging
- Standalone repository with CI (Python 3.11/3.12, mypy, a Pyodide import gate,
  and a reference drift-guard against statsmodels / linearmodels / arch),
  a GitHub-Pages playground workflow, and a trusted-publishing release workflow.
- `0.92.0` verified installable from a built wheel/sdist in a clean environment.

## 0.92.0

**Free local-LLM backends for narrative scoring** — run `score_llm` /
`llm_prob_kernel` at $0 on a local model (`narrative._local_engines`:
MLX / llama.cpp / Ollama, auto-selected, with lazy imports so `import puremacro`
stays Pyodide-clean). New `[local-llm]` extra and a desktop showcase notebook.

## 0.49.0 – 0.91.0

Consolidation toward 1.0: real Pyodide CI (booting Pyodide in Node and running a
curated smoke suite against the freshly built wheel), the user-runnable
validation gallery (`puremacro.validation.scorecard()`), the public-API snapshot
freeze, bilingual (English/Spanish) documentation and showcase notebooks, the
JupyterLite browser playground, the heterogeneous-agent / VFI suite
(Aiyagari/Huggett, Krusell-Smith aggregate shocks and transition paths,
life-cycle/OLG, Hopenhayn firm dynamics, Epstein-Zin, EGM), and the linear-DSGE
solvers (Klein QZ, gensys, a Smets-Wouters 2007 port).

## 0.20.0 – 0.48.0

Build-out of the core and modern-extensions surface: reduced-form and Bayesian
VAR with IRF/FEVD/historical decomposition; the SVAR identification menu
(recursive, long-run, sign and sign-plus-zero, proxy / external-instrument,
heteroskedasticity, max-share/news, non-Gaussian, and Giacomini-Kitagawa
identified-set bands); the Jordà local-projection family (lag-augmented, panel,
IV, state-dependent, smoothed, asymmetric, quantile, mean-group / CCE); the
inference layer (HAC / fixed-b / Driscoll-Kraay, weak-IV diagnostics, block /
wild bootstraps, specification curves); staggered DiD, dynamic-panel GMM,
high-frequency monetary surprises, volatility (`SigmaObject`, BEKK/CCC, HAR-RV),
nowcasting (Kalman DFM, MF-VAR), growth-at-risk, cointegration, factors, MIDAS,
spectral / wavelet tools; the narrative-econometrics pipeline (multi-source
uncertainty indices with keyword, embedding, and LLM scoring backends); and the
data-fetching / panel-building / instrument / shift-share infrastructure.

Pre-0.20 releases predate the public package and are not itemised here.
