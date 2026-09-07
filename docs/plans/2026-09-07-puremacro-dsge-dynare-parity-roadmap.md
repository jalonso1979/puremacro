# `puremacro.dsge` → Dynare parity — roadmap

**Doc version:** 1.0 · **Drafted:** 2026-09-07 · **Status:** proposed, not yet approved
**Baseline:** puremacro 2.5.0 (`puremacro/dsge`, 28 modules)
**Target releases:** 2.6.0 → 2.10.0, then 3.0.0
**Driving lens:** a researcher should be able to take a published `.mod` file, run it end to end — solve, diagnose, estimate, decompose — with no MATLAB, no Dynare, no C++ toolchain, inside the four-package Pyodide core (numpy / scipy / pandas / matplotlib).

> Revision history is at the foot of this file. Each release below has an **exit gate**; a release is not cut until its gate is green.

---

## Where we stand (measured, 2026-09-07)

`load_mod` on the bundled `puremacro/dsge/_references/sw07_pfeifer.mod` — 419 lines, 40 endogenous variables, 7 shocks, 54 parameters:

| | time |
|---|---|
| parse + solve, order 1 | **0.057 s** |
| parse + solve, order 2 | **4.6 s** (80×) |

The order-2 cost is almost entirely the central-difference Hessian loop over `K_vars = 3N + n_e = 127` columns (`dynare.py:651-664`). Extrapolated to order 3 that is ~16 000 Jacobian evaluations per solve — minutes. This single fact is why the AST front end (2.7.0) gates the higher-order work (2.8.0).

Parser probes, each reproduced by running it:

| Dynare construct | `parse_mod` today | severity |
|---|---|---|
| `#MU = c^(-gamma);` model-local variable | `NameError: name 'MU' is not defined` — `#` lines are `eval`'d *numerically* at parse time, so any local referencing an endogenous variable is dropped | loud |
| `STEADY_STATE(c)` | `NameError` | loud |
| `normcdf`, `erf`, `abs`, `sign` | `NameError` | loud |
| endogenous variable named `lag` | `AttributeError` — regex substitution collides with the `lead`/`curr`/`lag` namespace | loud |
| `@#define` / `@#for` / `@#include` | **file loads clean and a different model is solved** | **silent** |
| `estimated_params` block | stripped by `block_pattern`, never returned | silent (no estimation path exists to notice) |

What is solid and must not regress: Klein QZ (worst equilibrium residual 2.1e-15 over 292 random determinate models), the KKSS second-order/pruning recursion, the stacked-Newton perfect-foresight solver, the Kalman filter/smoother with exact diffuse initialisation, and the complex-step Jacobians — all re-verified in the 2.5.0 DSGE audit.

**The structural gap:** you can *solve* a `.mod` file and you cannot *estimate* one. `estimate_dsge` requires a hand-written `observation_eq` callable of the kind `sw07_observation.py` provides. Nothing connects `varobs` + `estimated_params` to it.

---

## Release plan

| Release | Theme | Contents | Needs |
|---|---|---|---|
| **2.6.0** | **`.mod` to posterior** | Tier 1 — estimation pipeline, diagnostics, steady-state solver, identification, optimal simple rules | — |
| **2.7.0** | **The front end** | Tier 0 — AST, macro processor, symbolic derivatives | — |
| **2.8.0** | **Higher order & constraints** | Tier 2 — order 3, perfect foresight `endval`/`histval`/MCP, multi-constraint OccBin + piecewise Kalman, SMC, particle filter, BGP detrending, nonlinear Ramsey | 2.7.0 |
| **2.9.0** | **Parity** | Tier 3 — `stoch_simul` filters, extended path, conditional forecast, posterior IRF bands, the parity dashboard | 2.7.0 |
| **3.0.0** | **Past Dynare** | analytic likelihood gradients → NUTS, heterogeneous-agent bridge, browser-native `.mod` | 2.8.0 |

Phases inside a release may ship together or split into point releases; the 2.5.0 spatial work is the precedent (phases 2-4 landed in one release rather than three).

---

## 2.6.0 — Tier 1: `.mod` to posterior

Design spec: `docs/specs/2026-09-07-puremacro-dsge-tier1-design.md`.

Everything here is deliberately reachable **without** the AST, so the highest-value user-facing capability does not wait on the largest refactor.

**Phase A — the estimation pipeline.** Parse `estimated_params` / `estimated_params_init` / `estimated_params_bounds` into the existing `Prior` objects; auto-build the measurement equation from `varobs` (+ measurement errors, `observation_trends`, `prefilter`); `LinearModel.estimate(data)`; `LinearModel.smoother(data)` (Dynare's `calib_smoother`); `LinearModel.forecast()`; marginal likelihood (Laplace + Geweke modified harmonic mean) and `model_comparison`; a mode-search menu (`csminwel`, Nelder-Mead, CMA-ES) plus `mode_check`.

**Phase B — diagnostics and the steady state.** `check()` with an eigenvalue table that names the variables loading on the offending roots; `model_diagnostics()`; `resid()`; a block-triangular steady-state solver (bipartite matching → SCC → topological solve) with Dulmage-Mendelsohn structural-singularity diagnosis, a `solve_algo` menu and homotopy continuation.

**Phase C — identification and simple rules.** `identification()` (Iskrev 2010) — rank, null space, pairwise collinearity, identification strength; `osr()` (optimal simple rules, built on the existing analytical `theoretical_moments()`); `discretionary_policy()` (Dennis 2007) and LQ commitment, both on the first-order matrices `build_dynare` already produces.

**Also in 2.6.0, one line of code:** unrecognised `@#` directives must **raise**. The macro processor itself waits for 2.7.0, but silently solving a different model is not an acceptable interim state.

**Exit gate.** `load_mod("sw07_pfeifer.mod").estimate(data)` reproduces the bespoke `estimate_sw07` posterior to the tolerance stated in the design spec; the generic observation equation matches `sw07_observation.make_state_space` in log-likelihood to 1e-8; the block steady-state solver solves a 50-equation model that `hybr` fails on; `identification()` recovers a planted unidentified parameter pair; the fertility BK diagnosis (`docs/research/fertility_bk_diagnosis/FINDINGS.md`) is reproduced by `resid()` + `check()` in one command.

**Explicit non-goals for 2.6.0.** No AST — `#`-locals over endogenous variables, `STEADY_STATE()`, `normcdf`, and `@#` macros all still fail (loudly). No order 3. No nonlinear Ramsey. No particle filter, no SMC. No time-varying measurement intercept inside the filter (`observation_trends` is handled by detrending the data, as Dynare does).

---

## 2.7.0 — Tier 0: the front end

The keystone. Replace regex + `re.sub` + `exec` with a real parser. ~1000-1500 lines of pure Python; sympy is not an option under the four-package contract and is not needed.

1. **Tokenizer + recursive-descent parser** for the model block → expression DAG (`Var(name, lead)`, `Param`, `Const`, `Call`, `BinOp`). Retires the whole substring-collision class of bugs.
2. **Macro processor** as a separate pre-pass: `@#define`, `@#for`/`@#endfor`, `@#if`/`@#else`, `@#include`, `@{expr}` interpolation. This is the line between "parses toy files" and "parses what people publish".
3. **Symbolic differentiation over the DAG with common-subexpression elimination**, emitting Python source compiled once per model. Order-2 SW07 goes from 4.6 s to ~0.05 s; order 3 becomes feasible; structural sparsity comes free.
4. **Free riders:** `STEADY_STATE()`, `EXPECTATION()`, `diff()`, model-local `#` variables inlined properly, `model(linear)` detection, `write_latex_dynamic_model`, `model_info`, per-equation `resid` keyed by equation tag.

**Exit gate.** Every existing `tests/test_dsge*` and `tests/test_dynare*` test passes unchanged against the new front end; `sw07_pfeifer.mod` order-2 solve under 0.2 s; the probe table at the top of this file is all green; a `@#for`-generated multi-country model parses and solves.

---

## 2.8.0 — Tier 2: higher order and constraints

- **Order 3 with pruning** (Andreasen, Fernández-Villaverde & Rubio-Ramírez 2018) — the reason people reach for `k_order_solver`: risk premia and asset pricing.
- **Perfect foresight, properly:** `histval` / `endval` (permanent shocks with a *different* terminal steady state — `solve_perfect_foresight` currently takes a single terminal vector), `varexo_det`, unanticipated replanning (`shocks(surprise)`), and **MCP complementarity tags** solved by semismooth Newton (Fischer-Burmeister) — Dynare's route to the ZLB in deterministic simulations. The existing stacked Newton is already sparse and block-tridiagonal, so this is boundary conditions plus a step rule.
- **OccBin beyond two regimes:** an `occbin_constraints` block with two constraints → four regimes, plus the **piecewise Kalman filter** (Giovannini et al.) so a model with an occasionally binding constraint can be *estimated*.
- **SMC (Herbst & Schorfheide)** alongside RWMH: robust to multimodality, embarrassingly parallel, and it returns the marginal likelihood as a by-product. Plus a slice sampler (no tuning) and a Haario adaptive-covariance proposal — 2.6.0 still adapts only a scalar.
- **Bootstrap particle filter** on `PrunedDSGESolution` → order-2 likelihoods.
- **Balanced-growth-path detrending:** `trend_var` / `log_trend_var` + `var(deflator=...)`.
- **Nonlinear Ramsey** (`ramsey_model`, `evaluate_planner_objective`) — needs the AST to differentiate the Lagrangian, hence its position here rather than in 2.6.0 phase C.

---

## 2.9.0 — Tier 3: parity and surface area

`stoch_simul` options still missing: `hp_filter`, `one_sided_hp_filter`, `bandpass_filter`, `ar=n` autocorrelation tables, `contemporaneous_correlation`, `simul_replic`, simulated (as opposed to theoretical) moments. Then `extended_path` (Fair-Taylor), `conditional_forecast` (Waggoner-Zha) with `conditional_forecast_paths`, `shock_groups`, `realtime_shock_decomposition`, `initial_condition_decomposition`, posterior IRF bands (`bayesian_irf`), prior predictive analysis, and `qz_criterium` exposed for unit-root models.

**The parity dashboard.** `load_dynare.py` already reads `oo_.mat`. Turn it into CI: run a corpus of published `.mod` files (Pfeifer's `DSGE_mod`, Dynare's own `tests/`) and report per file *parsed / solved / matched*, with max absolute deviation on `ghx`, `ghu`, `ghxx`, theoretical moments and IRFs. A published table reading "we parse 84% of the Pfeifer corpus and match `oo_.dr` to 1e-12 on all of them" is the single most persuasive artifact this project could ship — and it converts the silent `@#` failure into a measured number.

---

## 3.0.0 — past Dynare, not level with it

1. **Analytic likelihood gradients → HMC/NUTS.** With the AST giving ∂f/∂θ, implicit differentiation of the Klein/Sylvester equations gives ∂g_x/∂θ, and the Kalman derivative recursion gives ∂ log L/∂θ. Gradient-based sampling on a 40-parameter DSGE is a different regime from RWMH, and Dynare's own HMC support is thin.
2. **Heterogeneous agents in the same file.** `puremacro.models` already has sequence-space HANK with the fake-news algorithm. Letting a `.mod` declare a het-agent block — or letting the DSGE layer consume sequence-space Jacobians — is something Dynare structurally cannot do.
3. **It runs in a browser.** A Dynare needing no MATLAB licence and no install, `.mod` in and IRFs out on an iPad, has no competitor and is the project's stated identity.

---

## Cross-cutting constraints

- **Four-package core.** numpy, scipy, pandas, matplotlib at import time. No sympy, no jax, no numba, no C extensions in the numerical core. Every algorithm above is chosen to respect this.
- **Presentation contract.** Every new result object ships `.summary()`, `.plot()`, `.to_markdown()`, `.to_latex()`, `.to_typst()`.
- **Every module states what it does not do.** House style since 2.5.0: the docstring discloses the estimator's own limits, including unfavourable measured size.
- **Release mechanics.** Regenerate `tests/fixtures/public_api_snapshot.json` **from a clean checkout**, never from the live tree — the live tree can carry file-sync artifacts that would otherwise be baked into the release fixture. Bump `puremacro/__init__.py`, `pyproject.toml`, `tests/test_import.py` and `CHANGELOG.md` together.
- **Test runner.** conda `work` python with `PYTHONPATH=.`; no GNU `timeout` on macOS.

---

## Revision history

| Doc version | Date | Change |
|---|---|---|
| 1.0 | 2026-09-07 | First draft. Baseline 2.5.0; probes and timings measured on this date. |
