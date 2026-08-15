# puremacro 0.54.0 — fertility DSGE solver (R1b, solver-only)

**Status:** draft 2026-05-24. Target release: **0.54.0**.

## Why

R1b from the 2026-05-23 research-directions brainstorm. R1a (the generic
Bayesian DSGE engine) shipped as 0.53.0. R1 was decomposed into
engine-first (R1a) → fertility-port-second (R1b), and R1b itself is
further decomposed into solver (this release, 0.54.0) → Bayesian
estimation (R1c, 0.55.0).

The user maintains 6 Dynare variants of a fertility DSGE in
`My Drive/Fertility/`. The smallest baseline (`fertility_adj_costs.mod`,
131 LOC) defines an 11-equation 12-variable linear DSGE with capital +
fertility adjustment costs and three exogenous shocks (productivity,
mortality, fertility preference). The Bayesian variant
(`fertility_adj_costs_bayesian_estimation.mod`, 148 LOC) keeps the same
model equations but adds priors, observation equations, and a 13-equation
BGP calibration step (`bgp_fertility_calibration.m`).

This release ships the **solver only** — port the model equations, the
BGP calibration, and the linear-DSGE solve onto pure-numpy/scipy
infrastructure that already exists in puremacro (`klein_solve`,
`gensys`). It does NOT ship Bayesian estimation; that's R1c. Once the
solver is committed, R1c becomes a small additional release that wires
priors + an observation equation onto the engine that shipped in 0.53.0.

The split matches the existing puremacro release rhythm (0.51 → 0.53
all sized ~15-20 tasks each) and keeps the per-release review burden
low.

## Scope

One release. New module `puremacro/dsge/fertility_adj_costs.py` plus a
`FertilitySolution` dataclass and one example demo.

**In scope:**

- `puremacro.dsge.solve_bgp(exogenous, targets, x0, tol) -> dict` —
  scipy.optimize.fsolve port of `bgp_fertility_calibration.m`.
  13-equation system pinning 7 calibrated parameters + 6 steady-state
  values.
- `puremacro.dsge.fertility_adj_costs.model_residuals(z_lead, z, z_lag, eps, params) -> ndarray`
  — the 11 model-equation residuals at given (lead, current, lag, shock)
  vectors, used both for SS verification and numerical-Jacobian
  construction.
- `puremacro.dsge.solve_fertility(params=None, *, shock_stds=None) -> FertilitySolution`
  — top-level entry: BGP → numerical Jacobians → Klein QZ solve →
  FertilitySolution.
- `puremacro.dsge.FertilitySolution` — frozen dataclass with `ss`,
  `params`, `G`, `N`, `F`, `L`, `klein_solution`, `var_names`,
  `shock_names` + `irf()` and `fevd()` methods.
- `puremacro/examples/dsge_fertility_demo.py` — solve the default
  parameterisation, plot IRFs for the three shocks across (log_y, log_n,
  log_b) over 20 quarters.

**Out of scope:**

- **R1c (0.55.0):** Bayesian estimation. Priors dict (from the Bayesian
  Dynare variant), observation equation (cyc_y, cyc_mort, cyc_fert),
  `estimate_fertility` wrapper around `puremacro.dsge.estimate_dsge`.
- Housing model variants (`fertility_housing*.mod`).
- Higher-order perturbation (order2 / order3 .mod files). Could become R1d.
- Cross-country calibration sets (per-country `parameters.mat`).
- Counterfactual / policy experiments.
- **R3 — paleoclimate VARX.** Queued after R1c.

## Pre-conditions

- 0.53.0 shipped at tag `v0.53.0` (commit `aa562cc`), pushed to
  `origin/feature/subnational-labor-uncertainty-us`.
- 6 release-gate gates green at 0.53.0 HEAD.
- `puremacro.dsge.klein_solve(A, B, C, n_pre, ...) -> KleinSolution`
  available. The Klein helper expects the system in the form
  `A · E_t z_{t+1} = B · z_t + C · u_t` with `z_t = [x_t; y_t]`
  (predetermined first, forward-looking second).
- `puremacro.dsge.gensys.gensys(...) -> GensysSolution` available as a
  model-agnostic alternative (we use Klein in this release; gensys is
  available if Klein QZ classification fails on the fertility-DSGE
  eigenvalue structure).
- `scipy.optimize.fsolve` (already a transitive dep — no new package).
- `scipy.optimize.approx_fprime` (same).

## Architecture

```
puremacro/dsge/
  fertility_adj_costs.py    ← NEW (~300 LOC)
  _results.py               ← extend with FertilitySolution
  __init__.py               ← export solve_bgp, solve_fertility, FertilitySolution

puremacro/examples/
  dsge_fertility_demo.py    ← NEW (~80 LOC)

tests/test_dsge/
  test_fertility_bgp.py             ← NEW (~5 tests)
  test_fertility_residuals.py       ← NEW (~3 tests)
  test_solve_fertility.py           ← NEW (~5 tests)
  test_fertility_irf_fevd.py        ← NEW (~3 tests)

tests/test_examples/
  test_dsge_fertility_demo_runs.py  ← NEW (~1 test, smoke import + run)
```

Single-responsibility decomposition: `solve_bgp` is the calibration step,
`model_residuals` is the equation defining the model, `solve_fertility`
is the orchestrator wrapping the linearisation + Klein call. Each piece
is independently testable.

## Component A — `puremacro/dsge/fertility_adj_costs.py` (~300 LOC)

### Constants

```python
VAR_NAMES: tuple[str, ...] = (
    "c", "k", "i", "u", "n", "b", "l_o", "l_w", "y",
    "a", "mun", "ph",
)  # length 12

SHOCK_NAMES: tuple[str, ...] = ("ea", "ep", "en")  # length 3

_VARIABLE_TIMING: dict[str, set[str]] = {
    # exogenous shock states: appear at t-1 and t
    "a":    {"lag", "current"},
    "mun":  {"lag", "current"},
    "ph":   {"lag", "current"},
    # predetermined / capital-stock-like: appear at t-1 and t (and t+1 in Euler)
    "k":    {"lag", "current", "lead"},
    "n":    {"lag", "current", "lead"},
    # forward-looking: appear at t and t+1
    "c":    {"current", "lead"},
    "y":    {"current", "lead"},
    "l_w":  {"current", "lead"},
    "u":    {"current", "lead"},
    # purely static (current only)
    "i":    {"current"},
    "b":    {"current"},
    "l_o":  {"current"},
}

FERTILITY_EXOGENOUS_PARAMS: dict[str, float] = {
    "alpha":   0.4,
    "nu":      2.5,
    "phi":     1.03,
    "g":       0.017,
    "delta_p": 0.075,
    "delta_n": 0.065,
    "omega":   2.5,
    "bara":    0.0,
}

FERTILITY_CALIB_TARGETS: dict[str, float] = {
    "l":              0.30,    # work hours at SS
    "u":              1.00,    # capital utilization at SS
    "depr_rate":      0.10,    # delta_k * u^omega / omega at SS
    "kid_cost_share": 0.18,    # p_n * n / y at SS
    "k_y_ratio":      2.80,    # k / y at SS
    "c_y_ratio":      0.75,    # c / y at SS
    "n_growth":       1.02,    # 1 - delta_p + delta_n * n at SS
}

FERTILITY_SHOCK_PROCESSES: dict[str, dict] = {
    "ea": {"rho": 0.94**4, "sigma": 0.01},
    "en": {"rho": 0.50,    "sigma": 0.07},
    "ep": {"rho": 0.90,    "sigma": 0.07},
}
```

The exogenous + shock-process values mirror the Bayesian Dynare variant's
calibration block (the values that the .mod file uses BEFORE the
estimated_params block kicks in).

### `solve_bgp` — BGP calibration

```python
def solve_bgp(
    exogenous: dict | None = None,
    targets: dict | None = None,
    x0: np.ndarray | None = None,
    tol: float = 1e-12,
) -> dict:
    """Port of bgp_fertility_calibration.m. Solves the 13-equation BGP
    system via scipy.optimize.fsolve.

    Parameters
    ----------
    exogenous : dict, optional
        Pre-calibrated parameters. Defaults to FERTILITY_EXOGENOUS_PARAMS.
        Required keys: alpha, nu, phi, g, delta_p, delta_n, omega, bara.
    targets : dict, optional
        Calibration targets. Defaults to FERTILITY_CALIB_TARGETS. Required
        keys: l, u, depr_rate, kid_cost_share, k_y_ratio, c_y_ratio,
        n_growth.
    x0 : array of 13 floats, optional
        Initial guess for [mu_n, mu_h, beta, tau_n, tau_b, p_n, delta_k,
        c, b, l, u, k, n]. Defaults to the MATLAB script's x0.
    tol : float
        fsolve tolerance.

    Returns
    -------
    dict
        Merged exogenous + calibrated + steady-state values. Concretely:

        — All 8 keys from `exogenous` (alpha, nu, phi, g, delta_p,
          delta_n, omega, bara).
        — The 7 calibrated parameters from the MATLAB script (mapped to
          puremacro names): barn, mu_l (the MATLAB script's mu_h), beta,
          tau_n, tau_b, p_n, delta_k.
        — The 6 BGP fsolve outputs (the MATLAB script's c, b, l, u, k,
          n → our c, b, l_w, u, k, n).
        — Derived SS values for the remaining VAR_NAMES entries: i (from
          investment LoM), l_o (from time constraint), y (from production
          function), a (= bara), mun (= log(barn)), ph (= barp).

    The returned dict has all keys needed by `model_residuals` and
    `solve_fertility`.

    Raises
    ------
    KeyError
        If required keys are missing from exogenous or targets.
    RuntimeError
        If fsolve does not converge (exitflag <= 0) or returns
        non-finite / non-positive values for c, l, k, n, u.
    """
```

Implementation directly mirrors the MATLAB `bgp_system` function: the
13-residual function takes the 13 unpacked unknowns and returns the
13-vector of equation residuals (7 efficiency / resource conditions + 6
calibration targets including the hardcoded n_growth equation).

### `model_residuals` — the 12 equations

```python
def model_residuals(
    z_lead: np.ndarray,
    z: np.ndarray,
    z_lag: np.ndarray,
    eps: np.ndarray,
    params: dict,
) -> np.ndarray:
    """Evaluate the 12 fertility-DSGE residuals at given (lead, current,
    lag, shock) variable vectors.

    Parameters
    ----------
    z_lead, z, z_lag : shape (12,) arrays
        Endogenous-variable vectors at t+1, t, t-1 (ordered by VAR_NAMES).
    eps : shape (3,) array
        Shock realisations at t (ordered by SHOCK_NAMES).
    params : dict
        All structural + calibration parameters merged into one dict.

    Returns
    -------
    ndarray, shape (12,)
        Residuals of the 12 model equations (9 economic + 3 shock AR(1)):
        (1)  Production: y = exp(a) * (u*k_lag)^alpha * l_w^(1-alpha)
        (2)  Investment LoM: i = (1+g)*k - (1 - delta_k*u^omega/omega) * k_lag
        (3)  Resource constraint: c + p_n*n_lag + i + adj_costs = y
        (4)  Children LoM: n = (1-delta_n)*n_lag + exp(-ph) * b
        (5)  Time constraint: l_o + l_w + tau_n*n_lag + tau_b*b = 1
        (6)  Leisure-consumption FOC: mu_l * c * l_o^(-nu) = (1-alpha)*y/l_w
        (7)  Capital efficiency: delta_k * u^omega * k_lag = alpha * y
        (8)  Consumption Euler (intertemporal)
        (9)  Fertility Euler (intertemporal)
        (10) Productivity AR(1): a = (1-rhoa)*bara + rhoa*a_lag + ea
        (11) Fertility-preference AR(1): mun = (1-rhon)*log(barn) + rhon*mun_lag + en
        (12) Mortality AR(1): ph = (1-rhop)*barp + rhop*ph_lag + ep

    The returned vector length (12) matches len(VAR_NAMES), making the
    system square so Klein can solve it.
    """
```

The implementation directly mirrors the .mod file's `model;` block,
EXCLUDING the 6 `log_*` auxiliary identity equations (those are display
helpers in Dynare, not part of the dynamic system).

### `solve_fertility` — top-level entry

```python
def solve_fertility(
    params: dict | None = None,
    *,
    shock_stds: dict | None = None,
    h_for_jacobians: float = 1e-6,
) -> FertilitySolution:
    """Solve the fertility-adj-costs DSGE around its BGP.

    Pipeline:
      1. solve_bgp(...) → steady state + calibrated params.
      2. Merge with shock-process params (rhoa, rhon, rhop, sigmaa,
         sigman, sigmap) from FERTILITY_SHOCK_PROCESSES, overridable
         via the params and shock_stds kwargs.
      3. Build z_ss vector from BGP output.
      4. Numerically differentiate model_residuals at (z_ss, z_ss, z_ss,
         0, params) to get four Jacobians A, B, C, D via central
         differences with step h = max(h_for_jacobians, |z_i|*h_for_jacobians).
      5. Cast into Klein form A · E_t z_{t+1} = -B · z_t - C · z_{t-1}
         - D · eps. Then re-arrange into the canonical
         A_klein · E_t z_{t+1} = B_klein · z_t + C_klein · u_t shape that
         puremacro.dsge.klein_solve expects.
      6. Call klein_solve → KleinSolution.
      7. Extract G (state transition), N (shock impact), F (control
         policy), L (control response to shocks).
      8. Return FertilitySolution.

    Parameters
    ----------
    params : dict, optional
        Overrides for any of the parameters. Unspecified keys are filled
        from solve_bgp() output + FERTILITY_SHOCK_PROCESSES defaults.
    shock_stds : dict, optional
        Per-shock std overrides, e.g. {"ea": 0.012}. Defaults take from
        FERTILITY_SHOCK_PROCESSES.
    h_for_jacobians : float, default 1e-6
        Base step size for central-difference numerical Jacobians.

    Returns
    -------
    FertilitySolution

    Raises
    ------
    RuntimeError
        If the numerical Jacobians contain non-finite entries or if
        Klein's Blanchard-Kahn condition is violated. (klein_solve
        raises BlanchardKahnError; we wrap it with a more informative
        message naming this model.)
    """
```

The variable-timing classification is the highest-risk part of the
implementation. The Klein helper expects `z_t = [x_t; y_t]` with
predetermined first (`x_t`) and forward-looking second (`y_t`). Our
`_VARIABLE_TIMING` dict determines this partition:
- A variable that appears in `{lag, current}` (no lead) is **predetermined**.
- A variable that appears in `{current, lead}` (no lag) is **forward-looking**.
- A variable that appears in `{lag, current, lead}` (e.g., k, n) is
  predetermined (capital-stock-like).
- A variable in `{current}` only is **static** — handled by absorbing
  into the system via the standard Sims-gensys recast (extend the
  state vector with an "identity" equation).

The classification produces:
- `n_pre = 5` (a, mun, ph, k, n) — predetermined.
- `n_fwd = 4` (c, y, l_w, u) — forward-looking.
- `n_static = 3` (i, b, l_o) — static.

The static variables get auxiliary equations added so the final Klein
input has dim 12 = n_pre + n_fwd + n_static.

## Component B — `FertilitySolution` dataclass

Added to `puremacro/dsge/_results.py`:

```python
@dataclass(frozen=True)
class FertilitySolution:
    """Linear solution of the fertility DSGE around its BGP.

    Attributes
    ----------
    ss : dict[str, float]
        Steady-state values keyed by variable name (VAR_NAMES).
    params : dict[str, float]
        All parameters used in the solve (structural + calibration +
        shock-process).
    G : ndarray, shape (n_states, n_states)
        State transition (state at t given state at t-1, no shock).
    N : ndarray, shape (n_states, n_shocks)
        Shock impact on states.
    F : ndarray, shape (n_controls, n_states)
        Control policy (control at t given state at t).
    L : ndarray, shape (n_controls, n_shocks)
        Control response to contemporaneous shock.
    klein_solution : KleinSolution
        Raw QZ output for debugging.
    var_names : tuple of str
        All 12 endogenous variable names (states + controls in some
        canonical order; see the dataclass's state_indices / control_indices
        helper attributes if added).
    shock_names : tuple of str
        Shock names (ea, ep, en).
    """

    ss: dict
    params: dict
    G: np.ndarray
    N: np.ndarray
    F: np.ndarray
    L: np.ndarray
    klein_solution: "KleinSolution"
    var_names: tuple
    shock_names: tuple

    def irf(self, shock: str | int, horizon: int = 20) -> pd.DataFrame:
        """Impulse response to a 1-SD shock.

        Returns DataFrame of shape (horizon+1, n_vars) indexed by
        horizon 0..horizon, columns = var_names (deviation from steady
        state in levels, not logs).
        """

    def fevd(self, horizon: int = 20) -> pd.DataFrame:
        """Forecast-error variance decomposition.

        Returns DataFrame indexed by horizon, with one column per
        (variable, shock) pair giving the variance share contributed
        by that shock at that horizon.
        """
```

## Component C — `puremacro/examples/dsge_fertility_demo.py` (~80 LOC)

```python
"""Solve the fertility DSGE (Alonso-Ortiz, adjustment-costs variant)
around its calibrated BGP and plot IRFs.

Demonstrates puremacro.dsge.solve_fertility on a model OTHER than SW07,
proving the BGP+Klein machinery is composable with new DSGEs.
"""
import matplotlib.pyplot as plt
import numpy as np

from puremacro.dsge import solve_fertility


def main() -> None:
    sol = solve_fertility()
    # Print the BGP
    print("Fertility-adj-costs BGP:")
    for name in ("c", "k", "y", "n", "b", "l_w", "u"):
        print(f"  {name:6s} = {sol.ss[name]:.4f}")
    # Plot IRFs for the 3 shocks on (y, n, b)
    fig, axes = plt.subplots(3, 3, figsize=(11, 9))
    horizon = 20
    plot_vars = ["y", "n", "b"]
    for col, shock in enumerate(sol.shock_names):
        irf = sol.irf(shock, horizon=horizon)
        for row, var in enumerate(plot_vars):
            ax = axes[row, col]
            ax.plot(irf.index, irf[var], "k-")
            ax.axhline(0.0, color="0.7", lw=0.5)
            ax.set_title(f"{var} ← shock {shock}", fontsize=9)
            ax.set_xlabel("quarter" if row == 2 else "")
    fig.suptitle("Fertility DSGE: IRFs to 1-SD shocks", fontsize=11)
    fig.tight_layout()
    fig.savefig("dsge_fertility_demo.png", dpi=150)


if __name__ == "__main__":
    main()
```

The examples gallery auto-discovers the file. The figure is saved next
to the script (matching the existing examples-gallery convention).

## Data flow

```
solve_fertility(params=None)
   │
   1. solve_bgp(exogenous, targets, x0, tol)
       │
       ├── exogenous defaults from FERTILITY_EXOGENOUS_PARAMS
       ├── targets defaults from FERTILITY_CALIB_TARGETS
       ├── x0 defaults from _BGP_X0_DEFAULT
       └── scipy.optimize.fsolve on _bgp_system (13 residuals)
              ↓
       {alpha, nu, ..., mu_h, beta, ..., delta_k, c_ss, k_ss, ..., n_ss}
   │
   2. Merge with shock-process params (rhoa, sigmaa, rhon, sigman, rhop, sigmap)
       from FERTILITY_SHOCK_PROCESSES + user-supplied shock_stds.
   │
   3. Build z_ss = [c_ss, k_ss, i_ss, u_ss, n_ss, b_ss, l_o_ss, l_w_ss,
                    y_ss, 0, log(barn), barp]  (length 12).
   │
   4. For each of (z_lead, z, z_lag, eps), compute Jacobian via central differences:
       A_lead[i,j] = (R[i](z_ss + h_j*e_j at lead) - R[i](z_ss - h_j*e_j at lead)) / (2*h_j)
       B[i,j] similar at current
       C_lag[i,j] similar at lag
       D[i,j] similar at eps
       (where h_j = max(1e-6, |z_ss[j]|*1e-6))
   │
   5. Cast into Klein form. The linearised system is:
        A_lead · E_t Δz_{t+1} + B · Δz_t + C_lag · Δz_{t-1} + D · ε_t = 0
       Rearrange:
        A_lead · E_t Δz_{t+1} = (-B) · Δz_t + (-C_lag) · Δz_{t-1} + (-D) · ε_t
       Reorder z to put predetermined first (x_t) then forward-looking (y_t)
       then static — using _VARIABLE_TIMING to classify.
   │
   6. klein_solve(A_klein, B_klein, C_shocks_klein, n_pre=n_pre+n_static)
       → KleinSolution with policy functions.
   │
   7. Unpack G (state-transition), N (shock-impact on states), F (control
       policy), L (control shock-response) from the Klein output.
   │
   8. return FertilitySolution(ss, params, G, N, F, L, klein_solution,
                                var_names, shock_names)


FertilitySolution.irf(shock, horizon)
   │
   1. Resolve shock to index in shock_names.
   2. state_0 = N[:, shock_idx]; control_0 = L[:, shock_idx]
   3. for h in 1..horizon:
         state_h = G @ state_{h-1}
         control_h = F @ state_h
   4. Stack into (horizon+1, n_states + n_controls) array.
   5. Reorder columns to match VAR_NAMES.
   6. Return as DataFrame.
```

## Error handling

| Failure mode | Component | Handling |
|---|---|---|
| `exogenous` or `targets` missing required key | `solve_bgp` | `KeyError` naming the missing key |
| `fsolve` doesn't converge (exitflag <= 0) | `solve_bgp` | `RuntimeError("BGP fsolve failed: exitflag={exitflag}, residual norm={norm}")` |
| `solve_bgp` solution has non-positive c, l, k, n, or u | `solve_bgp` | `ValueError("BGP solution out of support: {name}={val:.3f}")` |
| Required key missing from `params` | `solve_fertility` | `KeyError` naming the missing key |
| Numerical Jacobian contains non-finite entries | `solve_fertility` | `RuntimeError("Jacobian contains non-finite entries — likely model_residuals is numerically unstable at SS")` |
| Klein QZ rejects (BK violated) | `solve_fertility` | wrap `BlanchardKahnError` with context message `"Klein Blanchard-Kahn condition failed for fertility-adj-costs: n_unstable={n_unst}, n_fwd={n_fwd}"` |
| `model_residuals` called with wrong-size vector | `model_residuals` | `ValueError` with expected vs actual shape |
| `FertilitySolution.irf(shock="bad_name")` | `FertilitySolution.irf` | `ValueError("unknown shock 'bad_name'; expected one of {shock_names}")` |
| `FertilitySolution.irf(horizon < 0)` | `FertilitySolution.irf` | `ValueError("horizon must be >= 0")` |
| `FertilitySolution.irf(horizon=0)` | `FertilitySolution.irf` | OK — returns 1-row DataFrame (impact-only) |

## Testing

Total ~17 new unit tests across 4 files under `tests/test_dsge/`, plus
1 smoke test under `tests/test_examples/`.

### `tests/test_dsge/test_fertility_bgp.py` (~5 tests)

- `test_solve_bgp_converges_with_defaults` — returns dict with the
  expected 22 keys (8 exogenous + 7 calibrated + 6 SS values + i + l_o);
  values finite.
- `test_solve_bgp_satisfies_residuals` — after solve, the 13 residuals
  evaluated at the returned values are all < 1e-10 in absolute value.
- `test_solve_bgp_satisfies_calibration_targets` — `bgp["l"]` ≈ 0.30,
  `bgp["u"]` ≈ 1.00, `bgp["k"]/bgp["y"]` ≈ 2.80, `bgp["c"]/bgp["y"]`
  ≈ 0.75, `bgp["p_n"]*bgp["n"]/bgp["y"]` ≈ 0.18.
- `test_solve_bgp_raises_on_missing_target` — drop `"k_y_ratio"` →
  `KeyError`.
- `test_solve_bgp_custom_targets_override` — set `targets["k_y_ratio"] =
  3.5` → returned BGP has k/y ≈ 3.5 (within fsolve tolerance).

### `tests/test_dsge/test_fertility_residuals.py` (~3 tests)

- `test_model_residuals_zero_at_steady_state` — call `model_residuals`
  with `z_lead = z = z_lag = z_ss`, `eps = 0`. All 12 entries
  in absolute value < 1e-8.
- `test_model_residuals_returns_correct_shape` — returns shape (12,).
- `test_model_residuals_responds_to_perturbation` — perturb z by 0.01 in
  one component; at least one residual changes by more than 1e-6.

### `tests/test_dsge/test_solve_fertility.py` (~5 tests)

- `test_solve_fertility_returns_dataclass` — basic API.
- `test_solve_fertility_blanchard_kahn_satisfied` — `klein_solution.eu
  == (1, 1)` (existence + uniqueness).
- `test_solve_fertility_g_matrix_stable_eigenvalues` — `max(|eig(G)|) <
  1.0` to 6 decimal places. State transition is contractive.
- `test_productivity_shock_positive_output_response` — IRF to `ea` on
  `y` at horizon 0 is positive; at horizon 4 is still positive
  (productivity is a persistent positive shock).
- `test_mortality_shock_negative_fertility_response` — IRF to `ep` on
  `n` at some short horizon (e.g., h=2) is negative (higher mortality
  reduces the children stock).

### `tests/test_dsge/test_fertility_irf_fevd.py` (~3 tests)

- `test_irf_returns_long_dataframe_with_var_columns` — shape (horizon+1,
  12); columns == VAR_NAMES.
- `test_irf_horizon_zero_returns_impact_only` — len(df) == 1.
- `test_irf_unknown_shock_name_raises` — `irf("bad_shock")` →
  `ValueError`.

### `tests/test_examples/test_dsge_fertility_demo_runs.py` (~1 test)

- `test_dsge_fertility_demo_main_runs_without_exception` — imports
  `main` and calls it; asserts no exception. Cleanup the saved figure
  after.

### Markers

- No `@pytest.mark.slow` on any new test. BGP fsolve is <1s, single
  Klein solve <1s, IRF / FEVD trivial. Each test <2s.
- No new `@pytest.mark.pyodide_smoke` tags; the existing 8-test Gate 6
  set is unchanged.

## Acceptance criteria for 0.54.0 (11)

1. `puremacro.dsge.solve_bgp(exogenous, targets, x0, tol) -> dict`
   exported from `puremacro.dsge.__init__`.
2. `puremacro.dsge.solve_fertility(params=None, *, shock_stds=None,
   h_for_jacobians=1e-6) -> FertilitySolution` exported.
3. `puremacro.dsge.FertilitySolution` (frozen dataclass) exported.
4. `puremacro.dsge.fertility_adj_costs` is a public submodule with
   `VAR_NAMES`, `SHOCK_NAMES`, `FERTILITY_EXOGENOUS_PARAMS`,
   `FERTILITY_CALIB_TARGETS`, `FERTILITY_SHOCK_PROCESSES`,
   `model_residuals`, `solve_bgp`, `solve_fertility`.
5. Default `solve_bgp()` produces BGP values within 1e-6 of an
   externally-computed MATLAB reference; the reference values are
   baked into `tests/test_dsge/test_fertility_bgp.py` as a Python
   constant `_EXPECTED_BGP_VALUES_FROM_MATLAB` with a sourced comment.
6. Default `solve_fertility()` produces a Klein solution satisfying
   Blanchard-Kahn, all `|eig(G)| < 1`.
7. ~17 new unit tests green under CPython.
8. Public-API snapshot regenerated (adds the new fertility surface).
9. All 6 release-gate gates green at HEAD.
10. CHANGELOG 0.54.0 entry. **No breaking changes** — strictly additive.
11. Version bumped 0.53.0 → 0.54.0.

## Risks + mitigations (6)

1. **Numerical Jacobians may be inaccurate at the SS.**
   `scipy.optimize.approx_fprime` is forward-difference by default; small
   `h` can amplify floating-point error on Euler equations with steep
   derivatives. *Mitigation:* implement central differences with `h =
   max(1e-6, |z_ss[i]| * 1e-6)` per-component. Cross-check by verifying
   `model_residuals(z_ss, z_ss, z_ss, 0)` is ≤ 1e-8 in absolute value
   (test `test_model_residuals_zero_at_steady_state`). Cross-check Klein
   eigenvalue structure against Blanchard-Kahn (test
   `test_solve_fertility_blanchard_kahn_satisfied`).

2. **BGP fsolve sensitivity to `x0`.** scipy.optimize.fsolve can land
   on spurious local solutions or fail to converge depending on starting
   point. *Mitigation:* use the same `x0 = [0.5, 1.2, 0.9, 0.5, 0.5,
   0.5, 0.08, 1.0, 0.5, 0.30, 1.0, 0.5, 0.5]` as the MATLAB script.
   If `exitflag <= 0`, raise `RuntimeError` with the residual norm — do
   not silently return garbage.

3. **MATLAB `parameters.mat` parity is brittle.** We can't read .mat at
   test time without adding `scipy.io.loadmat` to the test file, and
   the .mat may not be in the puremacro repo at all. *Mitigation:* run
   the .mat once externally, extract the 13 values, bake them as a
   Python constant `_EXPECTED_BGP_VALUES_FROM_MATLAB = {...}` inside
   the test file. Source comment in the test docstring. Tolerance 1e-6
   matches the MATLAB script's fsolve tolerance.

4. **Calibration target `n_growth = 1.02` was hard-coded in the MATLAB
   script.** *Mitigation:* take `n_growth` from `targets["n_growth"]`
   so users can override it. Default 1.02 matches the MATLAB script.

5. **Variable timing classification is the biggest correctness risk.**
   Dynare's `(-1)` / `(+1)` notation maps to specific positions in our
   `(z_lead, z, z_lag)` triple. Misclassifying which variables appear at
   which lag is the #1 source of "Klein didn't satisfy Blanchard-Kahn"
   bugs. *Mitigation:* define explicit `_VARIABLE_TIMING: dict[str,
   set[str]]` mapping each of the 12 variables to which slots it appears
   in. The Jacobian construction iterates over `_VARIABLE_TIMING` and
   skips variables not present in a given slot (sets the corresponding
   column to zero), so a typo in the timing dict will fail the
   Blanchard-Kahn test instead of producing silent garbage. The test
   `test_solve_fertility_blanchard_kahn_satisfied` is the guard.

6. **Euler equations contain `(c/c(+1))` which has a discontinuous
   derivative if c_lead ≈ 0.** At the BGP, `c_ss > 0` so it's fine,
   but the numerical-differentiation step could land too close to zero
   on tiny SS values. *Mitigation:* central differences with `h =
   max(1e-6, |c_ss| * 1e-6)` keeps the perturbed `c` strictly positive
   when `c_ss > 1e-6` (the default BGP gives `c_ss ≈ 1.0`, well above
   the threshold).

## Out of scope (deferred)

- **R1c (0.55.0):** Bayesian estimation. Priors dict (the
  `estimated_params` block of `fertility_adj_costs_bayesian_estimation
  .mod`), observation equation (cyc_y, cyc_mort, cyc_fert),
  `estimate_fertility` wrapper around `puremacro.dsge.estimate_dsge`.
- Housing model variants (`fertility_housing*.mod`).
- Higher-order perturbation (order2 / order3 .mod files). Could become
  R1d.
- Cross-country calibration sets (per-country `parameters.mat`).
- Counterfactual / policy experiments.
- **R3 — paleoclimate VARX / long-run cliometrics.** Queued after R1c.
