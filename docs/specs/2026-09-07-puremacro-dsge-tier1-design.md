# `puremacro.dsge` Tier 1 — `.mod` to posterior

**Status:** Drafted 2026-09-07. Architectural spec for the first tier of the Dynare-parity roadmap (`docs/plans/2026-09-07-puremacro-dsge-dynare-parity-roadmap.md`).
**Target release:** 2.6.0, in three phases (A, B, C) that may ship together or as 2.6.0/2.6.1/2.6.2.
**Driving lens:** everything in this tier is deliberately reachable **without** the AST front end, so the highest-value user-facing capability — estimating a `.mod` file — does not wait on the largest refactor. Nothing here changes the audited numerical core; it connects parts that already exist and adds the diagnostics that make the connection safe.

## Motivation

`puremacro.dsge` can *solve* a Dynare `.mod` file and cannot *estimate* one. The pieces are all present and unconnected:

- `priors.py` already reads Dynare's `estimated_params` conventions (`priors.py:178`) — but `parse_mod` strips the block (`dynare.py:936`) and never returns it.
- `state_space.py` has a Kalman filter, smoother and simulation smoother with exact-diffuse initialisation and NaN handling.
- `estimate.py::estimate_dsge` is a model-agnostic Bayesian estimator, audited in 2.5.0.
- `decomposition.py::_extract_companion_matrices` already produces exactly the companion form the filter needs.

What is missing is the connector: `varobs` → measurement equation. Today that must be hand-written per model, as `sw07_observation.py` does in 70 lines of index arithmetic. No user is going to do that for their own model, so in practice the estimator is reachable only for SW07.

Three further gaps make estimation unsafe even once it is reachable, and are in scope for the same tier:

1. **No diagnostics.** `docs/research/fertility_bk_diagnosis/FINDINGS.md` records a multi-day investigation whose verdict was that a rank-deficient calibration stage produced a linearisation point that was *not a steady state* — "the port drops Dynare's `steady;` step" — with dynamic residual norm 1.48. Dynare's `resid`, `check` and `model_diagnostics` would have reported that in one command.
2. **One steady-state solver, one guess.** `build.py:1231` is a single `scipy.optimize.root(..., method="hybr")`. It is the weak link on any model larger than a textbook RBC.
3. **No model comparison and no identification.** `estimate_dsge`'s own docstring says "There is no marginal-likelihood estimator, so no model comparison." Nothing checks whether the parameters are identified before sampling them.

## Non-goals

- **No AST.** `#`-locals over endogenous variables, `STEADY_STATE()`, `EXPECTATION()`, `normcdf`, and `@#` macro directives remain unsupported. They must fail **loudly**: this tier adds one guard — an unrecognised `@#` directive raises instead of being silently ignored.
- **No order 3**, no nonlinear Ramsey, no particle filter, no SMC. Those need 2.7.0's symbolic derivatives.
- **No time-varying measurement intercept.** `StateSpaceModel` is time-invariant by construction (`state_space.py:69`), so `observation_trends` is handled by detrending the data, exactly as Dynare's `prefilter`/`observation_trends` do.
- **No change to `estimate_dsge`'s contract.** It keeps taking `observation_eq: Callable[[dict], StateSpaceModel]`. This tier *builds* that callable; it does not rewrite the audited estimator.
- **No new runtime dependency.** numpy, scipy, pandas, matplotlib.

## Architecture

### Module map

```
puremacro/dsge/
├── _estimated_params.py   [new]  estimated_params / _init / _bounds grammar
│                                 -> EstimatedParamSpec, EstimatedParams
├── observation.py         [new]  varobs -> StateSpaceModel; prefilter,
│                                 observation_trends, measurement errors
├── mode.py                [new]  csminwel, cmaes, find_mode, mode_check
├── marginal.py            [new]  laplace_mdd, harmonic_mean_mdd, model_comparison
├── diagnostics.py         [new]  check(), model_diagnostics(), resid()
│                                 -> EigenvalueTable, ModelDiagnosticsResult
├── steady.py              [new]  block-triangular steady state, solve_algo menu,
│                                 homotopy continuation, Dulmage-Mendelsohn report
├── identification.py      [new]  Iskrev (2010) rank / null space / collinearity
├── policy.py              [new]  osr, discretionary_policy, lq_commitment
├── build.py               [ext]  LinearModel.estimate / .smoother / .forecast /
│                                 .check / .diagnostics / .resid / .osr
├── dynare.py              [ext]  parse_mod returns estimated_params, varobs
│                                 metadata, estimation options; raises on @#
├── decomposition.py       [ext]  _extract_companion_matrices promoted to a
│                                 documented shared helper
└── _results.py            [ext]  SmootherResult, DSGEForecastResult,
                                  EigenvalueTable, ModelDiagnosticsResult,
                                  IdentificationResult, OSRResult, PolicyResult

docs/dsge_estimation.md, docs/es/dsge_estimation.md   user guide (runnable)
tests/test_dsge/                                      unit + golden tests
```

---

## Phase A — the estimation pipeline

### A1. `estimated_params` grammar

Accepted statement forms, matching the Dynare manual:

```
[stderr NAME | corr NAME1, NAME2 | NAME] , [INITVAL ,] [LB, UB ,]
    PRIOR_SHAPE , P1 , P2 [, P3 [, P4 [, JSCALE]]] ;
[stderr NAME | corr NAME1, NAME2 | NAME] , INITVAL [, LB, UB] ;   // no prior
```

**Disambiguation rule** (the grammar is comma-separated and positionally ambiguous): consume any leading `stderr` / `corr` keyword and its one or two identifiers first, then split the remainder on commas and locate the first token that is a known `*_pdf` shape. Everything before it is `[]`, `[INITVAL]` or `[INITVAL, LB, UB]` by length; everything after is `P1, P2, P3, P4, JSCALE` by position. This resolves the `corr a, b` comma without lookahead heuristics.

**Prior shape mapping** onto the existing `priors.py` classes:

| Dynare shape | puremacro | notes |
|---|---|---|
| `beta_pdf` | `BetaPrior(mean=P1, std=P2)` | new `shift=P3`, `scale=P4-P3` fields for the generalised beta on `[P3, P4]`; P1/P2 describe the *generalised* variable |
| `gamma_pdf` | `GammaPrior(mean=P1, std=P2)` | new `shift=P3` |
| `normal_pdf` | `NormalPrior(mean=P1, std=P2)` | — |
| `inv_gamma_pdf`, `inv_gamma1_pdf` | `InvGammaPrior` | `priors.py` already carries both the (mean, std) and the (s, ν) parameterisation (`_invgamma_s_nu`, `_logpdf_invgamma_s_nu`) — dispatch on which the block supplies |
| `inv_gamma2_pdf` | `InvGammaPrior(kind="type2")` | new branch |
| `uniform_pdf` | `UniformPrior` | `(P1, P2)` as mean/std, or `(P3, P4)` as bounds when both are given |
| `weibull_pdf` | `WeibullPrior` | new class |
| `dirichlet_pdf` | — | out of scope; raises naming the shape |

**Typed spec.** Each statement becomes an `EstimatedParamSpec`:

```python
@dataclass(frozen=True)
class EstimatedParamSpec:
    kind: str          # "param" | "stderr_shock" | "corr_shock" | "stderr_obs"
    target: tuple      # ("alpha",) | ("eps_a",) | ("eps_a", "eps_b")
    name: str          # "alpha" | "SE_eps_a" | "CORR_eps_a_eps_b" | "ME_dy"
    prior: Prior | None
    init: float | None
    lb: float
    ub: float
    jscale: float = 1.0
```

`name` follows Dynare's display convention so output is recognisable: `SE_<shock>` for a structural standard error, `CORR_<s1>_<s2>` for a correlation, and `ME_<obsvar>` for a measurement error (Dynare writes `SE_EOBS_<var>`; the shorter form is documented in the user guide).

`kind` is what makes estimation fast: **only `kind == "param"` requires a model re-solve.** The other three write into `Q` or `H` directly.

`estimated_params_init;` (with `use_calibration`) overrides `init`; `estimated_params_bounds;` overrides `lb`/`ub`. Both are parsed here.

`parse_mod` gains keys `estimated_params`, `estimated_params_init`, `estimated_params_bounds`, `observation_trends`, `estimation_options`. Existing keys are unchanged; the addition is purely additive.

### A2. The observation equation

This is the technical heart. `_extract_companion_matrices(model)` already returns `(A, B, C, D, ys, variables, shocks, states, Σ_u)` in the timing

```
x_{t+1} = A x_t + B u_t                    (x_t predetermined at the start of t)
v_t     = ys + C x_t + D u_t               (all declared variables)
```

`v_t` depends on the *contemporaneous* innovation `u_t`, but `StateSpaceModel` is
`α_t = c + T α_{t-1} + R ε_t`, `y_t = d + Z α_t + η_t` with `η ⟂ ε`. The clean, exact
resolution is to carry the innovation in the state:

$$\alpha_t \equiv \begin{bmatrix} x_t \\ u_t \end{bmatrix},\qquad
T = \begin{bmatrix} A & B \\ 0 & 0\end{bmatrix},\qquad
R = \begin{bmatrix} 0 \\ I_{n_e}\end{bmatrix},\qquad
\varepsilon_t = u_t,\qquad Q = \Sigma_u$$

Verify: `T α_{t-1} + R u_t = [A x_{t-1} + B u_{t-1};\; 0] + [0;\; u_t] = [x_t;\; u_t] = α_t`. ✓

Then with `O` the row selection for `varobs`:

$$Z = \begin{bmatrix} C_O & D_O \end{bmatrix},\qquad d = ys_O,\qquad H = \operatorname{diag}(\sigma^2_{\text{ME}})$$

and `y_t = ys_O + C_O x_t + D_O u_t = d + Z α_t`. ✓ Dimensions: `m = n_states + n_e`, `r = n_e`, so `R` is `(m, r)` and `Q` is `(r, r)` as `StateSpaceModel.__post_init__` requires.

**Consequences worth stating.**

- Growth-rate observables need no special handling. In a `.mod` file `dy = y - y(-1) + ctrend;` declares `dy` as an endogenous variable, so `varobs dy` picks up row `dy` of `(C, D)` like any other. The 70 lines of index arithmetic in `sw07_observation.py` exist only because that model is hand-built rather than parsed.
- `build_dynare` / `load_mod` models are approximated in **levels** (`LinearModel.units` is all `"level"`), so `d = ys_O` and the data must be in model units. `prefilter=True` sets `d = 0` and demeans the data, as Dynare does.
- `observation_trends` subtracts `a_i + b_i·t` from column `i` of the data before filtering, and the trend is reported back on the smoothed and forecast series. Dynare's own semantics; documented rather than hidden.
- `H = 0` by default (Dynare's default), not the `1e-8` ridge `sw07_observation.py` uses. A ridge is opt-in via `ridge=`; the existing `_check_stochastic_singularity` (`estimate.py:415`) is called first so a singular `F` is reported as the specification error it is, not as a conditioning failure.

**API:**

```python
def make_state_space_from_varobs(
    model, varobs, *, shock_cov=None, measurement_error=None,
    prefilter=False, ridge=0.0,
) -> StateSpaceModel
```

### A3. `LinearModel.estimate` / `.smoother` / `.forecast`

`estimate` builds the `observation_eq` closure that `estimate_dsge` already expects, and delegates. The audited estimator is untouched.

```python
def estimate(self, data, *, priors=None, varobs=None,
             mode_compute="csminwel", n_draws=10_000, n_chains=2,
             burn_in=2_000, prefilter=False, observation_trends=None,
             diffuse_filter=False, seed=0) -> DSGEPosteriorResult
```

`priors` and `varobs` default to the `.mod` file's `estimated_params` and `varobs` blocks, carried on the model. The closure:

1. splits θ by `EstimatedParamSpec.kind` into structural / shock-std / shock-corr / measurement-error blocks;
2. **re-solves only if the structural block changed**, keyed on a hash of that sub-vector;
3. seeds `scipy.optimize.root` with the *previous accepted draw's* steady state rather than the static `initval` guess;
4. assembles `Q` and `H` from the remaining blocks and returns the augmented `StateSpaceModel`.

Steps 2 and 3 are the difference between a usable sampler and an unusable one: at the measured 0.057 s per SW07 solve, 10 000 draws × 2 chains is ~19 minutes of solving alone before the filter runs, and warm-starting the root finder typically removes a large fraction of it. The design does not promise a factor; the benchmark in `benchmarks/` will report the measured one.

`jscale` from the block scales the corresponding row/column of the proposal covariance.

**`smoother(data, *, params=None)`** is Dynare's `calib_smoother`: run the existing Kalman smoother at calibrated (or supplied) parameters. Returns `SmootherResult` with `.states`, `.shocks` (smoothed structural innovations), `.smoothed_obs`, `.filtered_states`, and `.shock_decomposition()` reusing `decomposition.compute_shock_decomposition`.

**`forecast(horizon=8, data=None, ci=0.90)`** returns `DSGEForecastResult`: unconditional forecasts from the smoothed terminal state with bands from the filter's `P`, plus in-sample one-step-ahead forecasts when `data` is given.

### A4. Mode search

`estimate_dsge` today does one bounded L-BFGS-B run and the 2.5.0 audit already found that it could terminate at iteration 1. `mode.py` adds a `mode_compute` menu mirroring Dynare's:

| `mode_compute` | algorithm | notes |
|---|---|---|
| `"lbfgs"` | current bounded L-BFGS-B | default today, kept |
| `"simplex"` | Nelder-Mead with restarts | derivative-free; Dynare's mode 8 |
| `"csminwel"` | Sims' quasi-Newton with the perturbed-Hessian retry | Dynare's default (mode 4); ~150 LOC |
| `"cmaes"` | covariance-matrix adaptation ES | global-ish; ~80 LOC, pure numpy |
| `"none"` | skip; start MCMC at `init` | Dynare's mode 0 |

`find_mode` runs the chosen method, then always re-reports the numerical Hessian through the existing `_nearest_pd` eigenvalue floor, and warns when the mode did not move or the Hessian was not positive definite — the 2.5.0 behaviour, preserved.

`mode_check(result, n_points=20, width=2)` evaluates the posterior along one-parameter slices through the mode and returns a result whose `.plot()` is the grid Dynare draws. A slice whose maximum is not at the mode is the single most common sign that the "mode" is not one.

### A5. Marginal likelihood

`marginal.py`:

- `laplace_mdd(result)` — `log p(y) ≈ log p(y|θ*) + log p(θ*) + (d/2) log 2π + ½ log|Σ*|` with `Σ*` the inverse Hessian at the mode. Cheap and already fully determined by what `estimate_dsge` returns.
- `harmonic_mean_mdd(result, p=(0.1, ..., 0.9))` — Geweke's (1999) modified harmonic mean with a truncated-normal weighting function, evaluated at each truncation level `p`. **The spread across `p` is reported, not hidden**: a modified-harmonic-mean estimate that moves by more than ~1 log point across truncation levels is not converged, and the result object says so.
- `model_comparison([...])` — a posterior-odds table over models with equal or supplied priors, using whichever estimator the caller names, and refusing to mix estimators silently.

The docstring states plainly what the 2.5.0 changelog established: marginal likelihoods computed before 2.5.0 are not comparable, because the Kalman recursion then started from a diffuse `P0` rather than the unconditional covariance.

---

## Phase B — diagnostics and the steady state

### B1. `check()` — the eigenvalue table that names names

`model.check()` returns an `EigenvalueTable`: one row per generalised eigenvalue with modulus, real and imaginary part, and `is_explosive` against `qz_criterium` (exposed, default `1 + 1e-6`). It reports `n_explosive` against `n_forward` and the Blanchard-Kahn verdict — most of which `_bk_status_line` (`cli.py:89`) already computes.

The part that does not exist anywhere today, and is the reason this is in Tier 1: **for a model that fails, the table carries the variable loadings on each offending generalised eigenvector.** Read the eigenvectors off the same `ordqz` factorisation `klein_solve` already computes, take the largest-modulus components, and map them back to variable names. The failure message takes the shape "12 explosive roots for 13 forward-looking variables; the missing root at |λ| = 1.97 loads chiefly on `<var>`, `<var>`, `<var>`" instead of a bare count. (Which variables the fertility model's root actually loads on is what validation item 12 pins; the moduli quoted there are the ones FINDINGS.md records.)

Unit roots (|λ| within tolerance of 1) are detected and reported with a pointer to `diffuse_filter=True`.

### B2. `model_diagnostics()` and `resid()`

`resid()` returns the per-equation steady-state residual as a labelled `pd.Series`, sorted by magnitude, keyed by equation tag where the `.mod` file supplies one. This is the check whose absence the fertility diagnosis identified: a residual norm of 1.48 at the linearisation point is visible in one line.

`model_diagnostics()` returns a `ModelDiagnosticsResult` carrying a list of typed findings:

| check | method |
|---|---|
| steady state does not solve the model | `max\|f(ss,ss,0)\|` against `tol`, per equation |
| static Jacobian rank deficient | SVD; report the near-null combinations of **equations** (collinear equilibrium conditions) and of **variables** |
| variable appears in no equation / equation involves no variable | numeric incidence matrix (below) |
| singular dynamic pencil | reuse `_qz.SingularPencilError` and its `_diagnosis` (`_qz.py:182`) |
| unit roots present | eigenvalue moduli against `qz_criterium` |
| stochastic singularity for the declared `varobs` | `n_varobs > n_shocks + n_measurement_errors` |

**Numeric incidence matrix.** Without an AST, "which variables enter which equation" is obtained by perturbing each variable and watching which residuals move — `N` residual evaluations, negligible. The failure mode is a derivative that happens to vanish at the evaluation point, producing a false zero. Mitigation, and it is specified rather than left to the implementation: evaluate the incidence at the steady state **plus four random points in a neighbourhood, and take the union**. The result object records how many points were used.

### B3. The steady state, properly

`steady.py` replaces "one `hybr` call from one guess" with Dynare's actual approach:

1. **Block-triangular decomposition.** Build the numeric incidence matrix (B2), find a perfect matching between equations and variables by Hopcroft-Karp (~60 LOC), build the directed graph induced by that matching, and take strongly connected components by Tarjan (~40 LOC). Solve the blocks in topological order: singleton blocks by scalar Brent/Newton, larger blocks by `scipy.optimize.root`. A 50-equation model with a recursive block structure becomes a chain of one- and two-dimensional problems.
2. **Structural singularity is a diagnosis, not a failure.** If no perfect matching exists the model is structurally singular; a Dulmage-Mendelsohn decomposition names the over- and under-determined equation/variable sets. That is a far better error than a non-convergent root find — and it is exactly the shape of the fertility model's "mutually inconsistent 13-equation system whose least-squares minimiser is non-isolated".
3. **`solve_algo` menu.** `"block"` (new default when the decomposition succeeds), `"hybr"` (today's behaviour, kept as the fallback and as the compatibility setting), `"lm"`, `"df-sane"`.
4. **Homotopy continuation.** `steady(homotopy={"param": (start, target)}, steps=10)` walks a parameter from a value where the steady state is known to the target, warm-starting each step from the last. The standard fix for a model that only solves near a particular calibration.

`build`/`build_dynare` gain `solve_algo=` and `homotopy=` and default to `"block"`; the existing behaviour is one keyword away, and the golden tests pin that `"hybr"` still returns bit-identical steady states.

---

## Phase C — identification and simple rules

### C1. `identification()` — Iskrev (2010)

Two Jacobians, both by finite differences over θ at the measured 0.057 s per re-solve (54 parameters ≈ 3 s on SW07; the shock-standard-deviation and correlation parameters need no re-solve at all):

- **J₁** = ∂ vec(T, R, Q, Z, H) / ∂θ — identification from the reduced-form solution.
- **J₂** = ∂ m(θ) / ∂θ, with `m` the theoretical first and second moments up to a chosen lag, taken from the existing analytical `theoretical_moments()`.

`IdentificationResult` reports, for each:

- rank against a scaled tolerance, and — when rank-deficient — **the null-space directions expressed as parameter combinations**, which is the answer a user actually needs ("`alpha` and `delta` enter only through `alpha·delta`");
- pairwise and multi-way collinearity: for each parameter, the R² of regressing its column on all others, and the worst offending subset;
- Ratto's normalised identification strength, split into the part due to the prior and the part due to the data.

Run at the prior mean by default, optionally over a Monte Carlo sample of prior draws (Dynare's `prior_mc`).

### C2. `osr()` — optimal simple rules

The cheapest genuinely new capability in the tier, because the objective is already analytic:

$$\min_{\gamma} \; \sum_i w_i \operatorname{Var}(y_i;\gamma) \quad\text{s.t. the model solves at } \gamma$$

with `Var` from `theoretical_moments()`. Optimise over the rule coefficients with Nelder-Mead or Powell, returning a large finite penalty (not `inf`) when Blanchard-Kahn fails at a trial point — the same pattern the 2.5.0 audit installed in the mode search, and for the same reason. At 0.057 s per solve a three-coefficient rule costs ~12 s on SW07.

```python
model.osr(rule_params=["rho_r", "phi_pi", "phi_y"],
          weights={"pinf": 1.0, "y": 0.5, "r": 0.1},
          bounds={...}) -> OSRResult
```

`OSRResult` carries the optimal coefficients, the loss at the optimum and at the calibrated rule, the per-variable variance contributions, and the five presentation methods.

### C3. `discretionary_policy()` and `lq_commitment()`

Both operate on the first-order matrices `build_dynare` already produces (`LinearModel._A_plus`, `._A_0`, `._A_minus`, `._B_u`, promoted to a documented accessor), so neither needs the AST:

- **`discretionary_policy`** — Dennis (2007): iterate on the policy matrices for the model `A₀ y_t = A₁ y_{t-1} + A₂ E_t y_{t+1} + A₃ u_t` under a quadratic loss, with the policy-rule equation removed and the instrument named by the caller.
- **`lq_commitment`** — form the Lagrangian on the linear constraints with the quadratic loss, which yields a larger *linear* rational-expectations system in `(y, λ)`, and hand it to the existing `klein_solve`. The multipliers come back as ordinary variables, so IRFs, moments and FEVDs work on the Ramsey allocation with no new machinery.

Full nonlinear `ramsey_model` — differentiating the planner's Lagrangian through the nonlinear equilibrium conditions — is explicitly deferred to 2.8.0, after the AST.

---

## Result objects

New frozen dataclasses in `_results.py`, each with the full presentation contract (`.summary()`, `.plot()`, `.to_markdown()`, `.to_latex()`, `.to_typst()`): `SmootherResult`, `DSGEForecastResult`, `EigenvalueTable`, `ModelDiagnosticsResult`, `IdentificationResult`, `OSRResult`, `PolicyResult`. `DSGEPosteriorResult` gains `log_mdd_laplace`, `log_mdd_harmonic`, `mdd_harmonic_spread` and `mode_compute`, all optional with `None` defaults so existing pickles and callers are unaffected.

## Validation

Every item below is a test, not an aspiration. Goldens live in `tests/fixtures/`.

| # | Check | Tolerance |
|---|---|---|
| 1 | `make_state_space_from_varobs` on SW07 vs the hand-built `sw07_observation.make_state_space`: log-likelihood at the posterior mode | 1e-8 |
| 2 | `load_mod("sw07_pfeifer.mod").estimate(data, seed=0)` vs `estimate_sw07(seed=0)`, 200 draws, one chain: mode and draws | stated in the test; the two paths differ in state-vector construction, so this pins *equivalence of the likelihood*, not bit-identity of the sampler |
| 3 | `estimated_params` parser against the block in `sw07_pfeifer.mod` (36 entries) and a synthetic block exercising every shape, `corr`, `stderr` on a `varobs`, `P3`/`P4`, and `jscale` | exact |
| 4 | Laplace marginal likelihood on a Gaussian linear model with an analytically known `log p(y)` | 1e-6 |
| 5 | Modified harmonic mean on the same model; spread across truncation levels reported and asserted small | — |
| 6 | Block steady-state solver vs `hybr` on RBC and SW07 | bit-identical |
| 7 | A 50-equation recursive model where `hybr` from the default guess fails and the block solver succeeds | converges |
| 8 | Dulmage-Mendelsohn report on a deliberately over-determined model names the right equation set | exact |
| 9 | `identification()` on a model with a planted unidentified pair (two parameters entering only as a product): the null space recovers the direction | 1e-6 on the direction |
| 10 | `osr()` on a 3-equation NK model with a closed-form optimal Taylor coefficient | 1e-4 |
| 11 | `lq_commitment` on the textbook NK loss reproduces the published commitment IRF sign pattern and the timeless-perspective initial condition | qualitative + pinned array |
| 12 | `resid()` + `check()` on the fertility model at the LM endpoint reproduce the FINDINGS.md verdict: residual norm ≈ 1.48, `n_stable = 11` against 12 required, and the offending eigenvector loads on the reported variables | pinned |
| 13 | An unrecognised `@#` directive raises | exact message |
| 14 | Pyodide import contract: no new top-level third-party import | `tests/test_pyodide_compat.py` |

Benchmarks (`benchmarks/`) record measured wall time for: SW07 solve at order 1, one full likelihood evaluation, 200 MH draws with and without the steady-state warm start, and the block vs `hybr` steady state on the 50-equation model.

## Rollout

| Phase | Contents | Approx. LOC | Gate |
|---|---|---|---|
| **A** | `_estimated_params.py`, `observation.py`, `mode.py`, `marginal.py`, `LinearModel.estimate/.smoother/.forecast`, `@#` guard | ~1400 | validation 1-5, 13, 14 |
| **B** | `diagnostics.py`, `steady.py`, `LinearModel.check/.diagnostics/.resid` | ~1100 | validation 6-8, 12 |
| **C** | `identification.py`, `policy.py`, `LinearModel.osr` | ~900 | validation 9-11 |

Phase A is the release headline and is independently shippable. Phase B makes A safe on models larger than SW07 and should not be deferred past the same release. Phase C is separable and can slip to 2.6.1 without blocking anything.

Documentation: `docs/dsge_estimation.md` and `docs/es/dsge_estimation.md` with runnable blocks, added to `mkdocs.yml`; `CHANGELOG.md` entry stating, per house style, what each estimator deliberately does not do.
