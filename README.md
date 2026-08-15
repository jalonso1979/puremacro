> 🇬🇧 English · 🇪🇸 [Español](README.es.md)

# puremacro

A **Pyodide-compatible empirical macroeconomics toolbox**: the estimator
code runs on pure numpy + scipy + pandas + matplotlib, so the numerical
core stays importable under Pyodide (iPad / juno.sh, best-effort — see
"juno.sh / iPad" below). The supported target is a local install on a
regular workstation.

## What's in it

**Core econometrics**

- **VAR** — reduced-form OLS, BVAR (Minnesota), VECM (Engle-Granger /
  Johansen), TVP-VAR, panel-VAR; IRF / FEVD / GFEVD; residual,
  block, moving-block, and wild bootstrap bands.
- **SVAR identification** (`var.identify.*`) — Cholesky, Blanchard-Quah,
  sign restrictions (Rubio-Ramirez-Waggoner-Zha), sign + zero
  restrictions (Arias-Rubio Ramirez-Waggoner), sign-robust bands
  (Giacomini-Kitagawa), proxy / external instruments, max-share / news,
  heteroskedasticity (Rigobon), non-Gaussian (Lanne-Meitz-Saikkonen).
  All public estimators return frozen-dataclass `…Result` objects.
- **Local projections** (`lp.*`) — single-country LP-HAC, LP-IV,
  lag-augmented LP (Plagborg-Møller-Wolf), panel LP with cluster /
  Driscoll-Kraay SE, state-dependent LP, smoothed LP (Barnichon-
  Brownlees B-splines), asymmetric LP (Tenreyro-Thwaites), LP-GARCH-
  state, LP-GARCH-in-mean, mean-group, CCE, quantile LP.
- **Inference** (`inference.*`) — central HAC OLS, Newey-West, Kiefer-
  Vogelsang fixed-b, Driscoll-Kraay; weak-IV diagnostics (Cragg-Donald,
  Kleibergen-Paap, Anderson-Rubin, Montiel Olea-Pflueger); Hansen-J /
  Stock-Yogo over-id; Pesaran CD, Swamy slope-homogeneity, Quandt-
  Andrews structural breaks, specification curves.
- **Other estimators** — Diebold-Yilmaz spillover index;
  Diebold-Mariano / Giacomini-White forecast comparison + density-
  forecast scoring (CRPS, log score); Bai-Perron breaks; unit-root
  tests (ADF, KPSS, PP, Zivot-Andrews); Klein QZ solver for linear DSGE
  (Blanchard-Kahn enforced).

**Modern macro extensions**

- **Staggered DiD** (`did.*`) — Callaway-Sant'Anna, Sun-Abraham,
  Borusyak-Jaravel-Spiess, Synthetic-DiD; bootstrap SE throughout.
- **Dynamic-panel GMM** (`dynpanel.*`) — Arellano-Bond, Blundell-Bond
  two-step Windmeijer + Hansen-J + AR(1)/AR(2) + Roodman collapse.
- **High-frequency monetary surprises** (`hfi.*`) — Gertler-Karadi 2015,
  Nakamura-Steinsson 2018, Jarociński-Karadi 2020.
- **Volatility** (`volatility.*`) — `SigmaObject` (1:1 port of the MAV
  MATLAB class with extended decomposition API), BEKK, CCC, HAR-RV,
  range-based, ARCH-LM / Ljung-Box diagnostics.
- **Nowcasting** (`nowcast.*`) — Kalman-DFM (Doz-Giannone-Reichlin)
  with ragged-edge handling, Mariano-Murasawa MF-VAR, forecast
  combinations, probabilistic scoring rules.
- **Growth-at-risk** (`gar.*`) — quantile AR, ABG 2019 skew-t fit, NFCI-
  style FCI.
- **Cycles / cointegration / factors** — Hamilton 2018 trend-cycle
  filter (`cycles`), Phillips-Hansen FM-OLS / Stock-Watson DOLS /
  Phillips-Ouliaris (`cointegration_modern`), PCA factors + Bai-Ng IC
  (`factor`), MIDAS (`midas`), KORV (2000) system-GMM CES (`korv_gmm`),
  synthetic control + placebo inference (`synthetic_control`).
- **Spectral / wavelet** (`spectral`, `wavelet`) — Welch PSD / cross-
  spectrum / coherence (numpy.fft only); MODWT-Haar wavelet variance
  decomposition.
- **Realized volatility** (`realized_vol`) — realized variance, bipower
  variation, Corsi HAR-RV.
- **Heterogeneous-agent / VFI** (`vfi.*`) — value-function iteration with
  EGM, finite-horizon life-cycle, OLG, Krusell-Smith aggregate shocks,
  Hopenhayn firm entry/exit, Epstein-Zin, permanent types, transition
  paths, and method-of-moments estimation; numpy reference backend with
  optional numba / mlx / cupy acceleration. See `notebooks/` for a
  showcase suite.

**Narrative econometrics** (`narrative.*`)

Fiscal- / labor- / uncertainty-narrative IV pipeline: canonical
`NarrativeEvent` / `NarrativeInstrument` schemas, deduplication,
keyword and manual scoring backends, panel construction, replication
loaders for canonical datasets (Romer-Romer, Mertens-Ravn). LLM
scoring backend (`narrative.scoring.llm`) and HTTP source modules
(`narrative.sources.*`) live as out-of-Pyodide side-channels.
Sources include:
- **Beige Book** — Fed Beige Book corpus from federalreserve.gov modern
  + FOMC historical pages, with per-canonical-section + per-district
  parsing (`puremacro.narrative.sources.iter_beige_book`,
  `puremacro.narrative.indices.bbui`).
- **US executive narrative** — Economic Report of the President
  (`iter_erp`), State of the Union (`iter_sotu`), and CBO reports
  (`iter_cbo`); three matching indices `erpui`, `sotuui`, `cboui`.
  CBO body fetches transparently fall back to the Wayback Machine
  when cbo.gov returns a DataDome challenge.
- **EU legislative narrative** — EUR-Lex binding acts (`iter_eurlex`)
  and EU Parliament plenary verbatim (`iter_ep_debates`); two trilingual
  EN/DE/FR indices `eurlex_ui` and `ep_ui`. EUR-Lex enumeration via
  the public Cellar SPARQL endpoint (Wayback-routed per-act fetch
  due to AWS-WAF on the live site); EP via Wayback CDX with coverage
  back to Term 7 (2009-07-14).
- **Bluesky archive** — central-bank governors + finance ministers via
  AT Protocol (`iter_bluesky_posts`, `bluesky_ui`). Hand-curated 29-
  handle seed list (`BLUESKY_KNOWN_HANDLES`); 12 resolved as of 2026-05-25.
  Multilingual via `languages=...` connector kwarg; the index defaults
  to monthly actor-level text aggregation (`aggregate_to="actor_month"`)
  to mitigate LUI's short-text degradation.
- **Cross-source disagreement** — `consensus_disagreement` computes
  the cross-sectional mean + std over any subset of narrative indices;
  `CROSS_SOURCE_GROUPS` documents thematic subsets.

Connectors hit by WAF / bot-protection (EUR-Lex, EU Parliament, CBO) fall back
to the Wayback Machine via the shared `puremacro.narrative.sources._wayback`
helper. Coverage is constrained by what Wayback has snapshotted.

**Data pipelines** (newly absorbed; see `ARCHITECTURE.md`)

- **Fetchers** (`fetch.*`) — FRED / ALFRED (real-time vintages), SDMX-CSV
  (OECD, Eurostat, ECB, IMF SDMX-Central), EPU / GPR / WUI / JLN /
  Fernald, OECD-MEI / QNA / Energy / FX, ILOSTAT, Yahoo, WB pink
  sheet, plus per-state FRED loaders for the US subnational track.
- **Panel builders** (`build_panel`, `build_subnational_panel`) — single
  entry points that materialise quarterly / monthly cross-country and
  US-state panels from the fetchers, with regime tagging, SA
  (X-13 / STL fallback), and a derived GARCH-σ pipeline.
- **Instruments** (`instruments.*`) — instrument registry +
  composition + external loaders (FRED API key path); backbone of the
  LP-IV machinery.
- **Bartik / shift-share** (`bartik.*`) — shares, sensitivities,
  Rotemberg weights, county-level EPU exposure.
- **Misc data utilities** — EU-KLEMS 2023 loader (`klems`), BIS NEER
  aggregator (`bis_neer`), G9 homogeneous-vintage splice
  (`long_panel`), Gollin labor share (`labor_share`), real-time
  vintages (`vintages`), seasonal adjustment (`sa`).
- **Labor flows** — 3-state E/U/N transitions from BLS CPS aggregates
  (`labor_flows`) and 4-state F/I/U/N transitions from ENOE microdata
  for Mexico (`labor_flows_enoe`).

**Teaching artefacts**

`teaching.*` is a research / teaching side-channel that intentionally
wraps `statsmodels` / `linearmodels` / `arch` so notebooks can compare
puremacro's pure-numpy estimators against the canonical packages. **Not
covered by the Pyodide promise.**

## Installation

### From PyPI (users)

```bash
pip install puremacro
```

This pulls the six base dependencies (numpy, scipy, pandas, matplotlib,
requests, pyarrow) — everything the estimators, the `fetch` layer and the
parquet code paths need. Extras are only for the optional features listed
below.

### Local (development)

From the `puremacro/` package directory (the one containing this `README.md`
and `pyproject.toml`):

```bash
pip install -e .
```

To run the dev parity tests, install the optional dev deps too:

```bash
pip install -e '.[dev]'
```

To use the `narrative.sources` PDF body extractor:

```bash
pip install -e '.[narrative]'
```

Other optional extras: `[backend]` (numba + Apple-Silicon mlx), `[cuda]`
(NVIDIA cupy), `[data]` (yfinance / fredapi / xlrd data fetchers), `[llm]`
(Anthropic-backed narrative scoring), `[embeddings]` (sentence-transformers
narrative scoring), `[notebooks]` (jupytext notebook build).

For connectors that want opt-in on-disk caching + per-host throttling,
the variants `safe_get_bytes_cached` and `safe_get_text_cached` apply
a SHA-256-keyed cache at `~/.cache/puremacro/http/`. Set
`PUREMACRO_HTTP_NO_CACHE=1` to bypass.

### juno.sh / iPad (unsupported, best-effort)

Upload the `puremacro/` directory to your juno.sh workspace, then in a
notebook cell:

```python
%pip install ./puremacro
```

**Caveat since `pyarrow` became a base dependency:** that command resolves
the full dependency set and `pyarrow` has no Pyodide wheel, so under a
Pyodide kernel it fails. Install the estimator core without dependency
resolution instead, and add by hand only what you need:

```python
import micropip
await micropip.install("puremacro", deps=False)
await micropip.install(["numpy", "scipy", "pandas", "matplotlib", "requests"])
```

Parquet code paths (`cache`, `fetch.labor*`, `shock_atlas`, `build_panel`)
stay unavailable in the browser. The browser is **not** a supported
deployment target: teaching material assumes a local install.

### Run the LLM features for free (local models)

The narrative LLM features (`score_llm`, `llm_prob_kernel`) run on a **local
model** — no API key, no paid API, $0. Everything else in puremacro is already
free; this closes the last paid gap.

Install an engine once (any one):

```bash
pip install "puremacro[local-llm]"     # MLX (Apple Silicon) + llama.cpp (any OS)
# or install Ollama (https://ollama.com) — no Python deps — then:  ollama pull qwen2.5:3b
```

Then swap in a local backend (same signatures as the paid backends):

```python
from puremacro.narrative.scoring import score_llm, LocalBackend
events = score_llm(records, backend=LocalBackend("qwen2.5-3b-instruct", engine="auto"))

from puremacro.narrative.indices import llm_prob_kernel, LocalProvider
idx = llm_prob_kernel(records, provider=LocalProvider("qwen2.5-3b-instruct"),
                      category="economic uncertainty")
```

`engine="auto"` picks the best installed engine (Apple GPU via MLX → llama.cpp →
a running Ollama server; for LM Studio / vLLM / any OpenAI-compatible server,
pass `engine="openai"` with `base_url=`). Models: `qwen2.5-3b-instruct` (default),
`gemma2-2b` (Google), `llama3.2-3b` (Meta), `phi3.5` (Microsoft), or any raw
engine model id. See `puremacro/examples/narrative_local_llm.py` and the
`local_llm_uncertainty` notebook. (Local inference is desktop-only — it does not
run inside the browser playground.)

## Pyodide compatibility

The runtime promise is: only `numpy + scipy + pandas + matplotlib`
ever get imported by the *estimator* code that ships in the wheel.
`statsmodels`, `linearmodels`, `arch` and `pypdf` are dev-only /
extras-only or lazy-imported behind a guard.

Two further packages are declared as base dependencies in
`pyproject.toml` — six in all — because the wheel cannot function
without them, even though neither touches the estimator path:

- `requests` — imported at module level by `puremacro.fetch.*` and the
  narrative sources. Pure Python; installs under Pyodide.
- `pyarrow` — the parquet engine `pandas.read_parquet` needs
  (`cache`, `fetch.labor*`, `shock_atlas`, `build_panel`, and the
  parquet datasets used by the teaching material). pandas imports it
  lazily, so it never lands in `sys.modules` on an import sweep. It has
  no Pyodide wheel: in the browser use
  `micropip.install("puremacro", deps=False)`.

See `ARCHITECTURE.md` → "Pyodide-compatibility contract" for the full
rationale.

The regression test is `tests/test_pyodide_compat.py` — it walks every
shippable submodule and asserts no forbidden module lands in
`sys.modules`. If you add a new optional dependency, follow the
existing lazy-import pattern (see `narrative.scoring.llm` or
`fetch._seasonal._x13_arima_analysis` for the canonical examples).

## Quickstart

**First 5 minutes — offline, no data files, no API key.** The quickest check
that your install works (a sign-restricted SVAR on a synthetic 3-variable DGP;
no network, no data, fixed seed):

```bash
python -m puremacro.examples.sign_restrictions_uhlig
```

Or, in Python, on a synthetic system you build in three lines:

```python
import numpy as np
import pandas as pd
import puremacro as pm

# A small synthetic 3-variable system (no data files, no API key).
rng = np.random.default_rng(0)
T = 200
Y = rng.standard_normal((T, 3)).cumsum(0)          # ndarray, shape (T, 3)

# Cholesky-identified SVAR with 90% residual-bootstrap bands.
from puremacro.var.identify.cholesky import cholesky_svar
res = cholesky_svar(Y, p=2, horizon=20, n_boot=500, ci=0.9)
print("IRF array shape (H+1, n, n):", res.irf_point.shape)   # (21, 3, 3)
# also available: res.irf_lower, res.irf_upper, res.n_boot, res.n_fail

# Single-country LP-HAC: response of y to a synthetic shock.
panel = pd.DataFrame({"y": Y[:, 0], "shock": rng.standard_normal(T)})
from puremacro.lp.jorda import lp_hac
irf = lp_hac(panel, y="y", x="shock", horizons=range(0, 21), n_lags=2)
print(irf.head())                       # columns: h, beta, se, t, lo, hi
```

Optional API keys are resolved centrally (none are needed for the synthetic
examples above):

```python
from puremacro import credentials
credentials.status()                  # see what's configured (no values leaked)
# credentials.require("fred")         # raises with a signup URL if the key is missing
```

## Rosetta Stone — Macroeconomist's Cheatsheet

If you are transitioning from Stata, MATLAB/Dynare, or statsmodels:

| Task / Estimator | Stata | MATLAB / Dynare | statsmodels / linearmodels | **`puremacro`** |
|---|---|---|---|---|
| **Cholesky SVAR** | `var y1 y2, lags(1/4)` + `irf create` | `varm` / VAR Toolbox | `VAR(Y).fit(4).irf(20)` | `var.identify.cholesky_svar(Y, p=4, horizon=20)` |
| **Blanchard–Quah SVAR** | `svar y1 y2, lreq(...)` | VAR Toolbox `bq_svar` | `SVAR(..., svar_type='B')` | `var.identify.bq_svar(Y, p=4, horizon=20)` |
| **Sign Restrictions** | User plugin | Rubio-Ramírez / VAR Toolbox | — | `var.identify.sign_restrictions(Y, signs, p=4)` |
| **Proxy / External IV SVAR** | `svariv` | Mertens & Ravn SVAR-IV | — | `var.identify.proxy_svar(Y, p=4, instrument_series=z)` |
| **Local Projections (HAC)** | `jorda` / manual OLS | Jordà (2005) code | `OLS(y_h, X).fit(cov_type='HAC')` | `lp.jorda.lp_hac(df, y="y", x="shock", horizons=range(21))` |
| **Panel LP (Driscoll–Kraay)** | `xtscc` | Panel LP toolbox | `PanelOLS(..., cov_type='driscoll-kraay')` | `regress.lp.lp_panel(df, y="y", shock="z", se="driscoll_kraay")` |
| **Dynamic Panel GMM** | `xtabond2 y L.y, gmm(y) two robust` | Arellano–Bond MATLAB | — | `dynpanel.ab_gmm(y, panel_id, time_id, two_step=True, windmeijer=True)` |
| **Staggered DiD** | `csdid y, ivar(id) time(t) gvar(g)` | — | — | `did.callaway_santanna(df, unit="id", time="t", outcome="y", treat_time="g")` |
| **Value Function Iteration** | — | VFIToolkit `ValueFnIter_Case1` | — | `vfi.VFIProblem(a_grid, z_grid, P_z, return_fn, beta).solve()` |
| **Linear DSGE (QZ / BK)** | — | Dynare `stoch_simul` / Klein `solab` | — | `dsge.klein.klein_solve(A, B, C, n_pre=...)` |
| **GLS Unit Root (DF-GLS)** | `dfgls y, maxlag(4)` | ERS (1996) code | `adfuller` | `tests.unit_root.dfgls_test(y, regression="ct")` |
| **Seasonal Adjustment** | `x13 y` | X-13 wrapper | `STL` / `x13` | `sa.stl.stl_sa(y)` / `sa.x11.x11_sa(y)` |

End-to-end replications of canonical papers live under `puremacro/examples/`
— Bloom 2009 (`bloom2009.py`), Mertens-Ravn narrative SVAR
(`svariv_mertens_ravn.py`), Romer-Romer monetary narrative
(`romer_romer_*.py`), and ~60 more. Most (like the Uhlig example above) are
fully synthetic and need no data or keys; a few read bundled or fetched data.

## Documentation

- **`ARCHITECTURE.md`** — module map, stability tiers, Pyodide
  contract, result-object standard. Read this first if you're
  contributing or trying to find where something lives.
- **`CHANGELOG.md`** — per-release diff, including internal-only refactors.
- **Per-function docstrings** are the canonical reference; the module
  docstring of each subpackage explains its scope.

## Conventions

- **Public API per subpackage** is curated via `__init__.py::__all__`;
  the top-level `puremacro` package only re-exports `__version__`.
- **Frozen-dataclass result objects** for any estimator returning 3+
  fields or non-trivial diagnostics (see `ARCHITECTURE.md` § Result-
  object standard). DataFrames returning named columns are exempt.
- **Diagnostic errors over silent garbage** — singular `X'X`, non-PD
  Σ, BK violations, and ill-conditioned bootstrap draws raise or warn
  with a message naming the calling function and the likely cause.

## Status

Pre-1.0; APIs rename freely with consumers updated in the same
commit. Single-author research package. CI workflows (tests, Pyodide
gate, mypy, reference drift-guard, playground deploy, PyPI release)
are defined in `.github/workflows/` and activate once the package is
split into its own repository; while it lives inside the monorepo
they are inert, so run `pytest` (or `python tools/release_check.py`)
locally before tagging a release.
