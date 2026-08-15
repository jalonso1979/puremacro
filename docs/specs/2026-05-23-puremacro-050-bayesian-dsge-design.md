# puremacro 0.50.0 — Bayesian estimation of SW07 (`estimate_sw07`)

**Status:** draft 2026-05-23. Target release: **0.50.0**.

## Why

`puremacro.dsge.smets_wouters` ships the Smets-Wouters (2007) DSGE model at the **posterior mode** (the constant `SW07_POSTERIOR_MODE` introduced at 0.45.0). What it does *not* ship is the actual Bayesian estimation that produces that mode — there is no glue tying the existing Kalman likelihood (`state_space.kalman_filter`), the MCMC diagnostics (`mcmc.geweke_z`, `gelman_rubin`, etc.), and the numerical machinery (`numerics.mle_fit`, `numerics.numerical_hessian`) into a working estimator. The model is callable; the estimator that fit it isn't.

This spec closes that gap with a single function `estimate_sw07(data, n_draws, n_chains, burn_in, seed) -> SW07PosteriorResult` that:

1. Loads the bundled US 7-variable dataset (1966Q1–2004Q4), or accepts a caller-supplied `DataFrame`.
2. Builds the SW07 state-space observation equation.
3. Computes the log-posterior (Kalman log-likelihood + log-prior over the 36 estimated parameters from Smets-Wouters 2007 Table 1A).
4. Refines the posterior mode and computes its Hessian.
5. Runs Random-Walk Metropolis-Hastings chains with Hessian-scaled proposals.
6. Returns a frozen-dataclass result with the chains, mode, accept rates, and diagnostics.

The estimator is **SW07-specific**, not generic — the priors, observation equation, and bundled data are all wired to the Smets-Wouters specification. A generic Bayesian DSGE engine is explicitly out of scope; future DSGE additions (HANK, RBC variants, etc.) can either replicate this pattern or motivate a refactor at that time.

This is the **P1** pitch from the 2026-05-22 brainstorm (the last of the four originally picked: P4 → P5 → P3 → P1).

## Scope

One release. Four new modules + one extension + one bundled CSV + one new dataclass (in `_results.py`):

- `puremacro/dsge/sw07_priors.py` — 36 priors per SW07 Table 1A + `log_prior(params: dict) -> float`.
- `puremacro/dsge/sw07_observation.py` — state-space construction `make_state_space(params: dict) -> (G, F, H, Q, D)`.
- `puremacro/mcmc.py` — extend with `random_walk_metropolis(...)` sampler (currently only diagnostics live here).
- `puremacro/dsge/sw07_estimate.py` — driver: `estimate_sw07(data, *, n_draws, n_chains, burn_in, seed) -> SW07PosteriorResult`.
- `puremacro/dsge/_sw07_data.csv` — bundled US 7-variable dataset, 155 quarterly obs, 1966Q1–2004Q4.
- `puremacro/dsge/_results.py` — new `SW07PosteriorResult` frozen dataclass (or extend the existing result-class file in `dsge/` if one is conventional).

Plus tests, version bump, CHANGELOG, public-API snapshot update.

Out of scope: generic Bayesian DSGE engine, HANK linearized solver, alternate samplers (DEMC, NUTS, HMC), Pyodide-Gate-6 coverage of the new tests (the 10K-draw run is far too slow for the Pyodide gate; the existing 8 marker tests stay unchanged), automated FRED fetcher for fresh SW07 data.

## Pre-conditions

- 0.49.0 shipped at tag `v0.49.0` (commit `fa298c4`), pushed to `origin/feature/subnational-labor-uncertainty-us`.
- `puremacro.dsge.smets_wouters.solve_sw07` produces `SWResult(G, Impact, eu, eigenvalues, state_names, control_names, shock_names, params)` per the 0.45.0 + 0.47.0 work. `G_full` is 44×44; `Impact` is 44×7.
- `puremacro.state_space.kalman_filter` accepts a `StateSpaceModel` and returns log-likelihood.
- `puremacro.numerics.mle_fit` and `puremacro.numerics.numerical_hessian` exist and operate on numpy parameter vectors.
- `puremacro.mcmc` currently exports diagnostics only (no sampler). The sampler is the main new addition.
- Six gates green at HEAD (`tools/release_check.py --examples --pyodide`).

## Architecture

One driver (`estimate_sw07`) composes four pieces:

```
estimate_sw07(data, ...)
   │
   ├── sw07_priors.log_prior(params)      ──┐
   │                                        ├── log_posterior(params)
   ├── state_space.kalman_filter(...)     ──┘    via sw07_observation.make_state_space
   │     (uses smets_wouters.solve_sw07 → (G, Impact))
   │
   ├── numerics.mle_fit(-log_posterior, init=SW07_POSTERIOR_MODE)
   │     → mode dict
   ├── numerics.numerical_hessian(-log_posterior, mode)
   │     → proposal_cov = c² * inv(H)
   │
   └── mcmc.random_walk_metropolis(log_posterior, init, proposal_cov, n_draws, ...)
         x n_chains chains
         → SW07PosteriorResult(draws, log_posterior_trace, mode, mode_hessian_inv,
                               accept_rates, n_burn_in, data_n_obs, seed)
```

All pure-numpy + scipy (for `scipy.stats.beta/gamma/norm/invgamma` log-pdfs). Single-threaded chains run sequentially (no multiprocessing — Pyodide does not support `multiprocessing.Process`).

## Components

### `puremacro/dsge/sw07_priors.py` (~80 LOC)

Declarative dict of the 36 SW07 priors from Smets-Wouters 2007 Table 1A. Schema (one entry per parameter):

```python
PRIORS: dict[str, dict] = {
    "phi_p": {"dist": "normal", "mean": 1.25, "std": 0.125, "lb": 1.0, "ub": 3.0},
    "sigma_c": {"dist": "normal", "mean": 1.5, "std": 0.375, "lb": 0.25, "ub": 3.0},
    "h": {"dist": "beta", "mean": 0.7, "std": 0.1, "lb": 0.0, "ub": 1.0},
    "xi_w": {"dist": "beta", "mean": 0.5, "std": 0.1, "lb": 0.0, "ub": 1.0},
    # ... 32 more
    "sigma_a": {"dist": "invgamma", "mean": 0.1, "std": 2.0, "lb": 0.0, "ub": 5.0},
    # ... seven shock-size priors
}
```

Public surface:

- `PRIORS: dict[str, dict]` — the 36-entry dict.
- `log_prior(params: dict) -> float` — sums log-density across all priors. Returns `-np.inf` if any parameter is outside its declared `[lb, ub]` support.
- `prior_means() -> dict[str, float]` — for use as an MLE init when `SW07_POSTERIOR_MODE` is unavailable.
- `prior_stds() -> dict[str, float]` — for use as a fallback proposal-cov when Hessian is non-PD.

Internally uses `scipy.stats.{beta, gamma, norm, invgamma}.logpdf`. Pyodide ships scipy, so this is Pyodide-safe.

### `puremacro/dsge/sw07_observation.py` (~120 LOC)

Maps SW07's 44 model variables to the 7 observed series, building the state-space form for the Kalman filter.

```python
OBSERVED_VARS: tuple[str, ...] = (
    "gdp_growth", "cons_growth", "inv_growth", "wage_growth",
    "log_hours", "infl", "ffr",
)

def make_state_space(params: dict) -> tuple[np.ndarray, ...]:
    """Build (G, F, H, Q, D) for the Kalman filter.

    Returns:
        G : (44, 44)  state transition (from solve_sw07(params).G)
        F : (44, 7)   shock loading on states (from solve_sw07(params).Impact)
        H : (7, 44)   observation matrix — picks observable rows + applies
                       log-difference transforms (e.g., gdp_growth = log_gdp_t - log_gdp_{t-1})
        Q : (7, 7)    shock innovation covariance, diag(σ_a², σ_b², ..., σ_g²)
        D : (7,)      constant offset (steady-state growth + mean inflation)
    """
```

`H`, `Q`, `D` are parameter-dependent. Reference: the published Pfeifer Dynare file at `puremacro/dsge/_references/sw07_pfeifer.mod`.

### `puremacro/mcmc.py` extension (+~150 LOC)

Adds the sampler. Existing diagnostics functions unchanged.

```python
def random_walk_metropolis(
    log_posterior_fn: Callable[[np.ndarray], float],
    init: np.ndarray,
    proposal_cov: np.ndarray,
    n_draws: int,
    *,
    seed: int = 0,
    accept_target: float = 0.25,
    adapt_burnin: int = 0,
) -> dict:
    """Random-Walk Metropolis-Hastings.

    Args:
        log_posterior_fn : params (np.ndarray) -> float. May return -np.inf.
        init             : (n_params,) starting point.
        proposal_cov     : (n_params, n_params) proposal covariance.
        n_draws          : number of post-burn-in iterations to keep.
        seed             : rng seed.
        accept_target    : adaptation target (default 0.25, optimal for d>5).
        adapt_burnin     : if > 0, run that many extra iterations BEFORE n_draws
                           with online scalar-c adaptation of the proposal scale
                           toward accept_target. Adaptation is scalar only (no
                           covariance adaptation) to keep convergence guarantees.

    Returns:
        dict with keys:
            chain : (n_draws, n_params)
            log_post : (n_draws,)
            accept_rate : float (over the n_draws iterations only)
            final_scale : float (the scalar c after adaptation; 1.0 if adapt_burnin=0)
    """
```

The adaptation rule during `adapt_burnin`: every 100 iterations, multiply `c` by `1.1` if recent accept rate > `accept_target * 1.2`, divide by `1.1` if < `accept_target * 0.8`. Frozen once burn-in ends.

### `puremacro/dsge/sw07_estimate.py` (~200 LOC)

The driver. Single public function:

```python
def estimate_sw07(
    data: pd.DataFrame | None = None,
    *,
    n_draws: int = 10_000,
    n_chains: int = 2,
    burn_in: int = 2_000,
    seed: int = 0,
) -> SW07PosteriorResult:
    """Bayesian estimation of Smets-Wouters (2007) via Random-Walk Metropolis.

    Args:
        data     : DataFrame with columns ("gdp_growth", "cons_growth", "inv_growth",
                   "wage_growth", "log_hours", "infl", "ffr"). If None, loads the
                   bundled 1966Q1–2004Q4 US dataset from puremacro/dsge/_sw07_data.csv.
        n_draws  : post-burn-in iterations per chain.
        n_chains : number of independent chains (run sequentially).
        burn_in  : iterations dropped from the start of each chain; also used as
                   the proposal-scale adaptation window.
        seed     : master rng seed. Chain i uses seed + i.

    Returns:
        SW07PosteriorResult with draws (n_chains, n_draws, 36), mode dict,
        Hessian inverse, accept rates, log-posterior traces.

    Raises:
        RuntimeError: posterior-mode search did not converge.
        ValueError  : data DataFrame has wrong columns or insufficient length.
    """
```

### `puremacro/dsge/_sw07_data.csv` (~15 KB)

Bundled 1966Q1–2004Q4 quarterly US dataset. 155 obs × 7 columns. Header lines (as `#` comments — `pd.read_csv(comment="#")`) document:

- FRED series IDs used (e.g., `GDPC1` for real GDP, `PCECC96` for real consumption, …).
- Transformations applied (per-capita, log-difference).
- Vintage date.
- Source: Smets-Wouters 2007 Table 2.

Loaded via `importlib.resources.files("puremacro.dsge") / "_sw07_data.csv"` to keep the path wheel-compatible.

### `puremacro/dsge/_results.py` (or sibling location, ~40 LOC)

```python
@dataclass(frozen=True)
class SW07PosteriorResult:
    draws: np.ndarray                  # (n_chains, n_draws, n_params)
    param_names: tuple[str, ...]       # length n_params (36)
    log_posterior_trace: np.ndarray    # (n_chains, n_draws)
    accept_rates: tuple[float, ...]    # length n_chains
    mode: dict[str, float]             # posterior-mode dict
    mode_hessian_inv: np.ndarray       # (n_params, n_params)
    n_burn_in: int
    data_n_obs: int
    seed: int

    def summary(self) -> pd.DataFrame:
        """Per-parameter mean, std, 5%/50%/95% quantiles across all chains."""
        ...
```

Public API exports added to `puremacro/dsge/__init__.py`:
- `estimate_sw07`
- `SW07PosteriorResult`
- `random_walk_metropolis` exported from `puremacro/mcmc.py::__all__`.

## Data flow

```
estimate_sw07(data, n_draws=10_000, n_chains=2, burn_in=2_000, seed=0)
  │
  ├── Step 1: data resolution
  │     ├── if data is None → load _sw07_data.csv via importlib.resources
  │     └── validate columns == OBSERVED_VARS; n_obs >= 50
  │
  ├── Step 2: log-posterior construction
  │     ├── log_likelihood(params_dict) — wraps solve_sw07 + make_state_space + kalman_filter
  │     │     catches BlanchardKahnError / np.linalg.LinAlgError / ValueError → -inf
  │     └── log_posterior(params_dict) = log_likelihood + log_prior
  │
  ├── Step 3: posterior-mode refinement
  │     ├── init = SW07_POSTERIOR_MODE (from smets_wouters.py)
  │     ├── x0 = dict_to_vec(init, param_names)
  │     ├── res = numerics.mle_fit(neg_log_posterior, x0, method="L-BFGS-B", bounds=prior_bounds)
  │     └── if not res.converged → warn, use SW07_POSTERIOR_MODE directly as mode
  │
  ├── Step 4: proposal covariance
  │     ├── H = numerics.numerical_hessian(neg_log_posterior, mode_vec, eps=1e-5)
  │     ├── H_pd = _nearest_pd(H)  # Higham (2002); identity if H already PD
  │     ├── try invH = np.linalg.inv(H_pd)  # PD ⇒ invertible
  │     ├── if invH non-finite / non-PD: invH = np.diag(prior_stds()² )  # fallback
  │     ├── c0 = 2.38 / sqrt(n_params)  # Roberts-Gelman-Rosenthal optimal
  │     └── proposal_cov = c0² * invH
  │
  ├── Step 5: run n_chains chains sequentially
  │     ├── for chain_idx in range(n_chains):
  │     │     rng = np.random.default_rng(seed + chain_idx)
  │     │     start = mode_vec + rng.multivariate_normal(zeros, 0.0025 * invH)
  │     │     out = random_walk_metropolis(
  │     │         log_posterior_vec, start, proposal_cov,
  │     │         n_draws=n_draws, seed=seed+chain_idx, adapt_burnin=burn_in,
  │     │     )
  │     └── stack: draws.shape == (n_chains, n_draws, n_params)
  │
  └── Step 6: assemble SW07PosteriorResult
```

## Error handling

**`log_likelihood`** catches:
- `BlanchardKahnError` (Klein BK violation) → `-np.inf`
- `np.linalg.LinAlgError` (singular matrices in Kalman update) → `-np.inf`
- `ValueError` (e.g., non-PD covariance in Kalman) → `-np.inf`

Any other exception propagates as a real bug.

**`log_prior`** returns `-np.inf` if any parameter is outside the declared `[lb, ub]` bounds for its prior family.

**`numerics.mle_fit` non-convergence**: emit a warning, fall back to `SW07_POSTERIOR_MODE` as the mode (since the paper's mode is a known good point). Continue to Step 4.

**Non-PD Hessian**: try `_nearest_pd` correction (Higham 2002 nearest-symmetric-positive-definite). If that still fails (e.g., Hessian dominated by NaNs), fall back to `diag(prior_stds²)` and warn.

**MH acceptance rate outside [10%, 50%] after burn-in**: warn (`UserWarning`). Chains are returned regardless; user can decide whether to discard.

**Catastrophically stuck chain** (all draws identical, or accept rate exactly 0): raise `RuntimeError` with chain index. Indicates a real bug.

## Testing

### `puremacro/tests/test_dsge/test_sw07_priors.py`

- `test_log_prior_finite_at_mode` — `log_prior(SW07_POSTERIOR_MODE)` is finite.
- `test_log_prior_minus_inf_out_of_support` — set a beta param to 1.1 → returns `-np.inf`.
- `test_log_prior_density_matches_scipy` — three spot-checks against `scipy.stats.beta.logpdf`, `gamma.logpdf`, `norm.logpdf`, `invgamma.logpdf`.
- `test_log_prior_sum_of_components` — total equals manual sum over 36 individual logpdfs.

### `puremacro/tests/test_dsge/test_sw07_observation.py`

- `test_make_state_space_shapes` — returns `(G, F, H, Q, D)` with shapes `(44,44), (44,7), (7,44), (7,7), (7,)`.
- `test_make_state_space_q_psd` — `Q` eigenvalues ≥ -1e-12.
- `test_make_state_space_h_picks_observables` — `H @ x_steady + D` recovers the steady-state value of each observable (e.g., `gdp_growth_ss ≈ params["gamma"] - 1`).

### `puremacro/tests/test_random_walk_metropolis.py`

- `test_metropolis_recovers_2d_normal` — target = bivariate `N(μ, Σ)` log-density. 20K draws + 2K burn-in adapt. Empirical mean within 0.05 of μ; empirical cov within 10% of Σ. Confirms unbiased sampling.
- `test_metropolis_accept_target_adaptation` — `adapt_burnin=2000` against the 2D normal; post-burn-in accept rate within ±5pp of `accept_target=0.25`.
- `test_metropolis_handles_minus_inf` — log-density returns `-np.inf` for proposed points outside a unit ball; chain stays in the valid region.

### `puremacro/tests/test_dsge/test_sw07_estimate_smoke.py`

Fast integration (~30-60s):

- `test_estimate_sw07_tiny_runs_clean` — `estimate_sw07(n_draws=500, n_chains=1, burn_in=200, seed=0)` runs without error. `result.draws.shape == (1, 500, 36)`. All draws finite. Accept rate in `[0.05, 0.6]`.
- `test_estimate_sw07_with_user_data` — pass a tiny `DataFrame(np.random.standard_normal((60, 7)), columns=OBSERVED_VARS)` → runs without error (data is nonsense but the pipeline doesn't crash).

### `puremacro/tests/test_dsge/test_sw07_estimate_replication.py`

**Slow** acceptance test. Marked `@pytest.mark.slow`. Skipped by default; opt-in via `pytest -m slow`.

- `test_estimate_sw07_posterior_means_close_to_sw07_table1a` — `estimate_sw07(n_draws=10_000, n_chains=2, seed=0)`. For 8 selected stable parameters, assert posterior mean within ±25% of Table 1A means:
  - `gamma` (trend growth): SW07 mean 0.43
  - `beta_const` (discount factor pre-transform): SW07 mean 0.16
  - `h` (habit): SW07 mean 0.71
  - `xi_p` (Calvo prices): SW07 mean 0.65
  - `xi_w` (Calvo wages): SW07 mean 0.73
  - `iota_p` (price indexation): SW07 mean 0.24
  - `iota_w` (wage indexation): SW07 mean 0.59
  - `sigma_c` (intertemporal elasticity): SW07 mean 1.39

The 8 params are picked from Table 1A as the most tightly-identified (smallest posterior std relative to the prior std). Other parameters (e.g., shock persistences) are noisier and excluded from the strict-tolerance test.

### Marker declarations in `pyproject.toml`

Add `slow` to the markers list:

```toml
[tool.pytest.ini_options]
markers = [
    "network: tests requiring live network access (opt-in via `pytest -m network`)",
    "pyodide_smoke: tests safe to run under Pyodide; opt-in via `pytest -m pyodide_smoke`",
    "slow: long-running tests (minutes); opt-in via `pytest -m slow`",
]
```

### NOT applying the `pyodide_smoke` marker

None of the new tests get the marker. RW-MH on 36 parameters with the SW07 Kalman likelihood is too slow for Gate 6 even at `n_draws=500`. Gate 6's existing 8 fast-test set is unchanged.

## Acceptance criteria for 0.50.0

1. `puremacro.dsge.estimate_sw07` and `SW07PosteriorResult` exported from `puremacro/dsge/__init__.py`.
2. `puremacro.mcmc.random_walk_metropolis` exported from `puremacro.mcmc.__all__`.
3. `puremacro/dsge/_sw07_data.csv` shipped in the wheel (verified by `test_sw07_estimate_smoke.py` loading it).
4. `puremacro/dsge/sw07_priors.py` + `puremacro/dsge/sw07_observation.py` exist with the surfaces documented above.
5. All unit tests across the four `test_*` files green under CPython.
6. Smoke test (`n_draws=500, n_chains=1`) passes in <60s.
7. Slow replication test passes on the maintainer's machine when invoked via `pytest -m slow` (the 8-param Table 1A check).
8. Public-API snapshot regenerated to include the new symbols (`estimate_sw07`, `SW07PosteriorResult`, `random_walk_metropolis`).
9. `pyodide_smoke` marker NOT applied to the new tests.
10. `slow` marker declared in `pyproject.toml::[tool.pytest.ini_options].markers`.
11. `CONTRIBUTING.md` documents `pytest -m slow` as the opt-in for long-running tests.
12. CHANGELOG 0.50.0 entry.
13. Version bumped: `pyproject.toml`, `puremacro/__init__.py`, `puremacro/tests/test_import.py` — all `0.50.0`.
14. Six gates green at HEAD (`release_check.py --examples --pyodide` exits 0 at 0.50.0).

## Risks and mitigations

1. **Posterior-mode init may not converge.** `numerics.mle_fit` with `L-BFGS-B` over 36 dims may stall or fail at the SW07 likelihood. *Mitigation:* `SW07_POSTERIOR_MODE` is the known good seed from 0.45.0. If `mle_fit` doesn't converge, fall back to it directly (it IS a posterior mode per the paper).

2. **Hessian-based proposal scaling may not yield the right acceptance rate.** The `c = 2.38/sqrt(n)` formula is optimal for Gaussian targets; SW07's posterior may be non-Gaussian. *Mitigation:* `adapt_burnin` scales `c` during burn-in to hit the 25% target.

3. **Kalman likelihood is slow.** Each MH iteration calls `solve_sw07(params)` (Klein QZ + Sylvester fallback, ~50ms per eval). 22K iterations × 2 chains ≈ 37 min realistically, not the 5-10 min I initially advertised. *Mitigation:* update CHANGELOG to "~20-40 min for the default 10K draws × 2 chains."

4. **Bundled CSV vintage.** SW07 used a frozen FRED vintage; bundling locks it in. *Mitigation:* document the FRED series + vintage in the CSV header comments. Future "fresh-data" replications are deferred to a separate fetcher spec.

5. **Observation equation transcription bugs.** SW07's measurement equation has growth-rate transformations + mean offsets that are easy to get wrong. *Mitigation:* `test_make_state_space_h_picks_observables` asserts H + D recover steady-state observables. Cross-check against `_references/sw07_pfeifer.mod`.

6. **Hessian non-PD.** SW07's mode is at the boundary of some priors (e.g., Calvo `xi_p` close to 1), where finite-difference Hessians can be degenerate. *Mitigation:* `_nearest_pd` correction. If that fails too, fall back to `diag(prior_stds²)` and warn.

7. **`@pytest.mark.slow` is a new marker.** First time we're adding a "slow but real" marker (we already have `network`). *Mitigation:* declare in `pyproject.toml`, document in `CONTRIBUTING.md`. No CI impact (no CI by design); maintainer discipline runs `pytest -m slow` before tagging if posterior-mean drift is a concern.

8. **Pyodide-coverage gap.** Gate 6 covers 8 fast tests; none of the new ones get the marker. SW07 estimation under Pyodide would take far longer than the 6s Gate 6 currently runs. *Mitigation:* documented in the spec + the CHANGELOG; Pyodide users can call `estimate_sw07` themselves at non-CI scale.

## Out of scope (deferred)

- **Generic Bayesian DSGE engine.** SW07-specific is the chosen scope; a generic `(solve_fn, prior_spec, obs_eq, data) -> Posterior` engine can wait until a second DSGE model lands.
- **HANK / TANK linearized solvers** — separate spec if pursued.
- **Alternate samplers** (DEMC, adaptive RW-MH with covariance adaptation, NUTS, HMC). RW-MH matches SW07's published methodology; alternatives are research extensions.
- **Fresh-data FRED fetcher** for SW07. Bundle the canonical vintage; fresh-data work is a separate (small) spec if needed.
- **Pyodide-Gate-6 coverage.** Tests too slow for Gate 6. Existing 8-test smoke is unchanged.
- **PyPI publishing**, **mixed-frequency BVAR (P2)**, **numba JIT (P6)** — separate specs, not picked in the original brainstorm.
