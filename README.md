> 🇬🇧 English · 🇪🇸 [Español](README.es.md)

# puremacro

A **Pyodide-compatible empirical macroeconomics toolbox**: the estimator
code runs on pure numpy + scipy + pandas + matplotlib, so the numerical
core stays importable under Pyodide (iPad / juno.sh, best-effort — see
"juno.sh / iPad" below). The supported target is a local install on a
regular workstation.

## 5-Minute Quickstart (2.0 Unified API)

`puremacro 2.0` standardizes the macro API around common parameter conventions (`lags`, `horizon`, `ci`), frozen-dataclass result objects, rich visualization (`.plot()`), and direct publication export (`.to_latex()`, `.to_typst()`, `.to_markdown()`):

### 1. Local Projections (LP) & Publication-Grade Export
```python
import numpy as np
import pandas as pd
from puremacro.lp import lp_hac

# Synthetic macro time series
rng = np.random.default_rng(42)
T = 200
shock = rng.standard_normal(T)
gdp = np.cumsum(0.7 * shock + 0.3 * rng.standard_normal(T))
df = pd.DataFrame({"gdp": gdp, "shock": shock})

# Unified API: horizon, lags, confidence level
res = lp_hac(df, y="gdp", x="shock", horizon=12, lags=4, ci=0.90)

# Instant visualization
res.plot(title="Output Response to Monetary Shock")

# Direct table export to LaTeX tabular or Typst table
print(res.to_latex())
print(res.to_typst())
```

### 2. DSGE Higher-Order Approximation & Dynare Parity
Solve nonlinear DSGE models up to second order with Kim, Kim, Schaumburg & Sims (2008) pruning, cross-derivatives ($g_{xu}, g_{uu}$), risk corrections ($g_{\sigma\sigma}$), and Dynare `oo_.dr` parity:
```python
from puremacro.dsge import build_dynare, load_mod

# 1. Define or load a model from .mod file with shocks and stoch_simul options
model = load_mod("rbc.mod")  # or build_dynare(eqs, states, controls, shocks, order=2)

# 2. Solve 2nd-order pruned perturbation
sol = model.solve(order=2)

# 3. Access Dynare-style decision rules (oo_.dr)
print(sol.oo_dr["ghx"])   # first-order state transition
print(sol.oo_dr["ghxx"])  # second-order state curvature
print(sol.oo_dr.summary())

# 4. Analytical theoretical moments & variance decomposition
mom = sol.theoretical_moments()
print(mom.summary())
print(mom.to_latex())
```

### 3. Juno / iPad / Pyodide to Google Colab Offloading
When working on an iPad or client-side Pyodide session with compute or memory constraints, seamlessly offload heavy tasks (e.g. 10,000-draw MCMC or large bootstrap SVARs) to Google Colab:
```python
from puremacro.runtime.colab import (
    generate_colab_notebook,
    show_colab_offload_dialog,
    load_colab_result,
)

# Generate a self-contained Google Colab notebook with auth and Drive mounting
nb = generate_colab_notebook(
    task_code="""
import puremacro as pm
res = pm.dsge.estimate_sw07(n_draws=10000, n_chains=4)
pm.runtime.store.save_frame(res.summary(), "sw07_posterior.pmz")
""",
    mount_drive=True,
    export_result_file="sw07_posterior.pmz",
)

# Open Colab with 1 click in Juno or browser
show_colab_offload_dialog(nb, filename="sw07_offload.ipynb")

# Once Colab finishes, load the .pmz result back into your local session
posterior = load_colab_result("sw07_posterior.pmz")
```

---

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
  and 2nd-order perturbation with Kim et al. (2008) pruning, cross-terms,
  and Dynare `oo_.dr` parity (`dsge.dynare`).

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
- **Heterogeneous-agent / VFI / Sequence-Space HANK** (`vfi.*`, `models.hank_sequence_space`) — value-function iteration with
  EGM, finite-horizon life-cycle, OLG, Krusell-Smith aggregate shocks,
  Hopenhayn firm entry/exit, Epstein-Zin, permanent types, transition
  paths, and method-of-moments estimation. In addition, full sequence-space
  HANK (Auclert et al. 2021) featuring the $\mathcal{O}(T^2)$ Fake News
  Algorithm (`fake_news_algorithm`, `FakeNewsResult`) and targeted fiscal
  transfer simulations across wealth deciles (`simulate_targeted_transfer`,
  `FiscalTransferResult`); numpy reference backend with optional
  numba / mlx / cupy acceleration. See `notebooks/` for a showcase suite.

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

- **Fetchers** (`fetch.*`) — FRED / ALFRED, SDMX-CSV
  (OECD, Eurostat, ECB, IMF SDMX-Central), EPU / GPR / WUI / JLN /
  Fernald, OECD-MEI / QNA / Energy / FX, ILOSTAT, Yahoo, WB pink
  sheet, plus per-state FRED loaders for the US subnational track.
- **Long national accounts** (`fetch.qna_long_panel`) — the OECD spine extended
  backwards per country by ratio-splicing archived national vintages onto it:
  **Spain to 1970Q1** (+100 quarters) and **Japan to 1955Q2** (+155), with
  provenance per series per quarter. The splice preserves the old vintage's
  growth rates and reports how stable each join's ratio is, because a ratio
  that drifts across the overlap means the two vintages disagree about growth
  and the spliced level depends on the anchor. Seven other candidate sources
  were measured to buy zero quarters and the reasons are kept in
  `LONG_PANEL_KNOWN_GAPS`. See `docs/long_panel.md`.
- **Real-time data** (`fetch.vintage_panel`, `fetch.realtime.*`) — published
  *editions* of a series across six providers behind one call: the OECD STES
  revisions archive (42 economies, monthly editions from 1999), ALFRED, the
  Bundesbank Gerda database, the ONS real-time workbook (746 editions back to
  1961), Statistics Canada's vintage tables and the ECB/EABCN database. Comes
  with the revision toolkit — revision triangles, first/latest release,
  `r_t = y_f - y_p`, and the Mankiw-Shapiro news-vs-noise test
  (`vintages.mankiw_shapiro`). Each provider documents what its vintage date
  actually means, because they disagree. See `docs/real_time_data.md`.
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

**Running away from a workstation** (`runtime.*`, `runtime.colab`, `pocket.*`, `longrun.*`)

The package's headline promise is that the estimator core runs on an
iPad. These tools make the promise usable rather than merely true:
`runtime` reports what the machine can actually do (sockets? parquet?
threads?) and routes HTTP over the browser when there are no sockets;
`runtime.colab` provides seamless task offloading to Google Colab with
cloud auth and `.pmz` persistence when tasks exceed mobile memory;
`pocket` packs data into portable, self-verifying `.pmz` cartridges so a
panel built online opens offline; `longrun` runs bootstraps and chains in
resumable chunks that survive the OS suspending the app, with results
invariant to how the work was sliced. See "juno.sh / iPad" below.

**DSGE sketchpad, Dynare parity & frontier engines** (`dsge.build`, `dsge.dynare`, `dsge.cli`, `dsge.occbin`, `dsge.bayesian`, `dsge.perfect_foresight`)

Write equilibrium conditions as a Python function or parse native Dynare
`.mod` files (`load_mod`, `parse_mod`). Solves 1st- and 2nd-order approximations
with complex-step differentiation, Kim-Kim-Schaumburg-Sims (2008) pruning,
cross-derivatives ($g_{xu}, g_{uu}$), risk adjustments ($g_{\sigma\sigma}$),
Dynare `oo_.dr` decision rules, and analytical theoretical moments (`stoch_simul`).
Includes the dedicated `puremacro-dynare` CLI tool, Guerrieri-Iacoviello (2015)
OccBin piecewise-linear algorithm for occasionally binding constraints (ZLB),
Boucekkine-Juillard stacked Newton-Raphson relaxation for non-linear perfect
foresight, and full Bayesian MCMC estimation (Laplace Hessian covariance +
adaptive Random-Walk Metropolis-Hastings). No hand-derived matrices, no
Fortran/C++ compiler, 100% Pyodide-ready.

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

This pulls the seven base dependencies (numpy, scipy, pandas, matplotlib,
requests, pyarrow, openpyxl) — everything the estimators, the `fetch` layer and the
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

#### Finding out what the tablet can actually do

`puremacro.runtime` answers that at run time rather than leaving you to
discover it one traceback at a time:

```python
from puremacro import runtime
print(runtime.report())
#  host       : pyodide 3.12.7 (wasm32)
#  device     : tablet
#  network    : js-fetch (call runtime.enable_browser_network())
#  parquet    : unavailable -> use puremacro.runtime.store / pocket
#  threads    : no (1 cpu, unknown)
#  backends   : numpy
```

Detection is heuristic — no API tells you "this is Juno" — so every field
can be pinned with `PUREMACRO_HOST`, `PUREMACRO_DEVICE`,
`PUREMACRO_SOCKETS` or `PUREMACRO_PARQUET`.

#### The three things that break, and what to do about them

**No sockets.** `requests` and `urllib` cannot open a connection under
Pyodide, so every `fetch.*` call fails even though the estimator core
imports perfectly. One call routes the whole existing fetch layer over
the browser's own networking:

```python
from puremacro import runtime
from puremacro.fetch import fetch_xrate_monthly

runtime.enable_browser_network()
fx = fetch_xrate_monthly(["MEX"])
```

Endpoints must send `Access-Control-Allow-Origin` — some public
statistical APIs do, many WAF-fronted government sites do not. A blocked
request says so and names CORS; `proxy=` routes through a CORS proxy you
control.

**No pyarrow.** Pack the data where the network and pyarrow are, open it
where they are not. A cartridge is one self-verifying file carrying its
own provenance:

```python
from puremacro import pocket

# workstation
pocket.pack(panel, "g7.pmz", source="OECD QNA", vintage="2026-08-19")

# iPad, airplane mode
cart = pocket.load("g7.pmz")
panel = cart.frame()          # sha256-checked on read
cart.provenance.vintage       # '2026-08-19'
```

Getting a *file* onto an iPad is often more friction than the analysis,
so a cartridge also travels as text: `pocket.to_base64("g7.pmz")` on
one machine, `pocket.from_base64(blob, "g7.pmz")` on the other.

**The app gets suspended.** iPadOS stops a backgrounded app, and a
four-minute bootstrap does not survive someone answering a message.
`puremacro.longrun` computes in chunks, persists after each, and resumes
in a later session:

```python
import numpy as np
from puremacro import longrun

job = longrun.bootstrap(one_draw, 2000, checkpoint="irf.ckpt")
job.run(seconds=30)     # 240/2000 · 12% · ~220s of compute left
job.run(seconds=30)     # ... and again after the app was suspended
bands = np.percentile(job.result(), [5, 95], axis=0)
```

Draw *i* always uses `default_rng([seed, i])`, so a job resumed across
five sessions gives bit-identical results to one that ran straight
through — which is what makes a resumed run publishable.

**Sizing the work to the device.** `runtime.fit(n_boot=2000)` returns
what this machine should actually attempt, and
`runtime.budgeted(estimator)` clamps the cost arguments of a call.
Both are opt-in: no estimator default changed, so a script that runs on
your laptop produces the same numbers it always did. Only cost knobs are
clamped — `horizon` changes what is being estimated, so it is left alone.

**Offloading heavy compute to Google Colab.** When tasks exceed tablet memory
or CPU limits (such as full Bayesian MCMC chains or 10,000-draw wild bootstraps),
`puremacro.runtime.colab` offloads the work seamlessly:

```python
from puremacro.runtime.colab import (
    generate_colab_notebook,
    show_colab_offload_dialog,
    load_colab_result,
)

# 1. Package computation into a self-contained notebook with Drive sync
nb = generate_colab_notebook(
    task_code="""
import puremacro as pm
res = pm.dsge.estimate_sw07(n_draws=5000, n_chains=2)
pm.runtime.store.save_frame(res.summary(), "sw07_result.pmz")
""",
    mount_drive=True,
    export_result_file="sw07_result.pmz",
)

# 2. Open directly in Colab from Juno / Jupyter
show_colab_offload_dialog(nb, filename="offload.ipynb")

# 3. Retrieve output back in iPad/Pyodide session via pure-numpy .pmz format
res_summary = load_colab_result("sw07_result.pmz")
```

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

Three further packages are declared as base dependencies in
`pyproject.toml` — seven in all — because the wheel cannot function
without them, even though none touches the estimator path:

- `requests` — imported at module level by `puremacro.fetch.*` and the
  narrative sources. Pure Python; installs under Pyodide.
- `pyarrow` — the parquet engine `pandas.read_parquet` needs
  (`cache`, `fetch.labor*`, `shock_atlas`, `build_panel`, and the
  parquet datasets used by the teaching material). pandas imports it
  lazily, so it never lands in `sys.modules` on an import sweep. It has
  no Pyodide wheel: in the browser use
  `micropip.install("puremacro", deps=False)`.
- `openpyxl` — the `.xlsx` engine `pandas.read_excel` needs. Eighteen
  shipped modules read Excel: the EPU, WUI, JLN, LMN, Fernald, GPR and
  World Bank Pink Sheet fetchers among them. It used to live in the
  `dev` extra, so a plain `pip install puremacro` could not produce
  any of those series — and `build_all` swallows each failure into a
  `print`, so the panel came back quietly missing most of its
  uncertainty proxies. Pure Python; installs under Pyodide.

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
from puremacro.var.identify import cholesky_svar
res = cholesky_svar(Y, p=2, horizon=20, n_boot=500, ci=0.90)
print(res.summary())
res.plot(target_idx=0, shock_idx=0)        # 1-line IRF plot with confidence bands
print(res.to_latex(target_idx=0, shock_idx=0))  # Camera-ready LaTeX table

# Single-country LP-HAC: response of y to a synthetic shock.
panel = pd.DataFrame({"y": Y[:, 0], "shock": rng.standard_normal(T)})
from puremacro.lp import lp_hac
irf = lp_hac(panel, y="y", x="shock", horizon=20, lags=2, ci=0.90)
print(irf.summary())
irf.plot(title="Response of y to Structural Shock")
print(irf.to_latex())                      # Camera-ready LaTeX table
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

| Task / Estimator | Stata | MATLAB / Dynare | statsmodels / linearmodels | **`puremacro 2.0`** |
|---|---|---|---|---|
| **Cholesky SVAR** | `var y1 y2, lags(1/4)` + `irf create` | `varm` / VAR Toolbox | `VAR(Y).fit(4).irf(20)` | `var.identify.cholesky_svar(Y, p=4, horizon=20)` |
| **Blanchard–Quah SVAR** | `svar y1 y2, lreq(...)` | VAR Toolbox `bq_svar` | `SVAR(..., svar_type='B')` | `var.identify.bq_svar(Y, p=4, horizon=20)` |
| **Sign Restrictions** | User plugin | Rubio-Ramírez / VAR Toolbox | — | `var.identify.sign_restrictions(Y, signs, p=4)` |
| **Proxy / External IV SVAR** | `svariv` | Mertens & Ravn SVAR-IV | — | `var.identify.proxy_svar(Y, p=4, instrument_series=z)` |
| **Local Projections (HAC)** | `jorda` / manual OLS | Jordà (2005) code | `OLS(y_h, X).fit(cov_type='HAC')` | `lp.lp_hac(df, y="y", x="shock", horizon=20, lags=4)` |
| **State-Dep LP-IV (Ramey-Zubairy)** | manual 2SLS interaction | — | — | `lp.lp_state_dep_iv(df, y="y", x="g", z="news", state="u")` |
| **Panel LP (Driscoll–Kraay)** | `xtscc` | Panel LP toolbox | `PanelOLS(..., cov_type='driscoll-kraay')` | `lp.panel_lp_dk(df, y="y", x="z", unit_col="id", time_col="t")` |
| **Dynamic Panel GMM** | `xtabond2 y L.y, gmm(y) two robust` | Arellano–Bond MATLAB | — | `dynpanel.ab_gmm(y, panel_id, time_id, two_step=True, windmeijer=True)` |
| **Staggered DiD** | `csdid y, ivar(id) time(t) gvar(g)` | — | — | `did.callaway_santanna(df, unit="id", time="t", outcome="y", treat_time="g")` |
| **Synthetic DiD** | `sdid y id t d` | synthdid R package | — | `did.synthetic_did(df, unit="id", time="t", outcome="y", treatment="d")` |
| **Factor-Augmented VAR (FAVAR)**| — | BBE (2005) MATLAB | — | `var.favar(panel_df, policy_series, n_factors=3, horizon=20)` |
| **Value Function Iteration** | — | VFIToolkit `ValueFnIter_Case1` | — | `vfi.VFIProblem(a_grid, z_grid, P_z, return_fn, beta).solve()` |
| **Linear DSGE (QZ / BK)** | — | Dynare `stoch_simul` / Klein `solab` | — | `dsge.klein.klein_solve(A, B, C, n_pre=...)` |
| **DSGE from equations / .mod** | — | Dynare `.mod` file | — | `dsge.load_mod("rbc.mod")` / `dsge.build_dynare(eqs)` |
| **DSGE 2nd-Order Pruning** | — | Dynare `stoch_simul(order=2, pruning)` | — | `dsge.build_dynare(eqs, order=2)` / `m.solve_second_order()` |
| **`puremacro-dynare` CLI** | — | `dynare model.mod` command-line | — | `puremacro-dynare model.mod --order 2 --fevd --plot` |
| **OccBin (ZLB / piecewise)** | — | Dynare `occbin_solver` / Guerrieri & Iacoviello | — | `dsge.solve_occbin(m_normal, m_zlb, constraint, shocks)` |
| **Non-Linear Perfect Foresight** | — | Dynare `simul` (Boucekkine-Juillard) | — | `dsge.solve_perfect_foresight(m, shocks, T=100)` |
| **Bayesian DSGE (MCMC)** | — | Dynare `estimation(...)` (Metropolis-Hastings) | — | `dsge.estimate_dsge_bayesian(m, data, priors, n_draws=10000)` |
| **Sequence-Space Fake News** | — | SSJ (Auclert et al. 2021) Python/Julia | — | `models.fake_news_algorithm(T=40)` / `models.simulate_targeted_transfer(...)` |
| **GLS Unit Root (DF-GLS)** | `dfgls y, maxlag(4)` | ERS (1996) code | `adfuller` | `unit_root.dfgls_test(y, regression="ct")` |
| **Seasonal Adjustment** | `x13 y` | X-13 wrapper | `STL` / `x13` | `sa.stl_sa(y)` / `sa.x11_sa(y)` |

End-to-end replications of canonical papers live under `puremacro/examples/`
— Bloom 2009 (`bloom2009.py`), Mertens-Ravn narrative SVAR
(`svariv_mertens_ravn.py`), Romer-Romer monetary narrative
(`romer_romer_*.py`), Smets-Wouters 2007 frontier showcase (`41_dynare_frontier_showcase.py`),
and ~75 more. Most (like the Uhlig example above) are
fully synthetic and need no data or keys; a few read bundled or fetched data.

## Documentation

- **`docs/quickstart.md`** — 2-minute quickstart covering core estimators and publication workflows.
- **`docs/dsge_build.md`** — DSGE models from equations, native Dynare `.mod` loader, 2nd-order pruning, `puremacro-dynare` CLI, OccBin ZLB, non-linear relaxation, and Bayesian MCMC.
- **`docs/models.md`** — Structural models: Sequence-Space HANK, Fake News algorithm, targeted transfers, and DMP search-and-matching.
- **`docs/var.md`** — Reduced-form VAR, SVAR identification (Cholesky, signs, narrative, proxy/IV), FAVAR, and bootstrap bands.
- **`docs/lp.md`** — Local Projections guide (LP-HAC, LP-IV, State-Dependent LP-IV, Panel LP, `LPResult`).
- **`docs/did.md`** — Modern Difference-in-Differences (Callaway-Sant'Anna, Sun-Abraham, Borusyak-Jaravel-Spiess, Synthetic DiD).
- **`docs/nowcast.md`** — GDP Nowcasting (Mixed-frequency dynamic factor models, ragged edges, news decomposition).
- **`docs/climate.md`** — Climate macroeconomics: Nordhaus DICE forward simulator and Social Cost of Carbon accounting.
- **`docs/forecast.md`** — Penalized macroeconomic forecasting: Coordinate-descent Elastic Net and Adaptive Lasso.
- **`docs/reporting.md`** — Publication reporting pipeline (LaTeX tabular, Typst tables, Markdown, significance stars).
- **`docs/tablet.md`** — Running on iPad, Juno, and WebAssembly, with Google Colab compute offloading.
- **`docs/benchmarks.md`** — Performance and computational benchmarks across econometric engines.
- **`docs/national_accounts.md`** — OECD Quarterly National Accounts extraction, deflators, and accounting identities.
- **`docs/real_time_data.md`** — Real-time data vintages, revision triangles, and Mankiw-Shapiro news-vs-noise testing.
- **`docs/long_panel.md`** — Historical long national accounts panel (ratio-spliced Spanish and Japanese series).
- **`docs/es/`** — Complete parallel documentation suite in native academic Spanish.
- **`ARCHITECTURE.md`** — module map, stability tiers, Pyodide contract, result-object standard.
- **`CHANGELOG.md`** — per-release diff, including internal-only refactors.
- **`docs/ADVISORY.md`** — correctness advisories: released versions that returned a wrong number.
- **Per-function docstrings** are the canonical reference; the module docstring of each subpackage explains its scope.

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

Production release, shipping **2.3.0**. Covered by release gate 3 in
`docs/1.0_path.md` § 5 lists which subpackages are inside that promise
and which are research-experimental.

CI is live and runs on every push: the suite across three operating
systems and three Python versions, the Pyodide contract, mypy, the
reference drift-guard, `mkdocs build --strict`, the playground deploy,
and a tag-triggered PyPI publish via trusted publishing. See
`.github/workflows/`. Run `python tools/release_check.py` locally
before tagging anyway — gates 5 and 6 are opt-in and CI does not run
them.

When a released version has returned a wrong number, it is recorded in
**[`docs/ADVISORY.md`](docs/ADVISORY.md)**, with the condition under
which the error vanishes so you can rule your own run in or out.
