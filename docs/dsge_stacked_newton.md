> 🇬🇧 English · 🇪🇸 [Español](es/dsge_stacked_newton.md)

# Stacked-time Newton-Krylov for perfect-foresight models

`puremacro.dsge.stacked_newton` solves the deterministic two-point boundary problem of [`solve_perfect_foresight`](dsge_build.md#8-non-linear-simulation-perfect-foresight), with a given initial state `y_init` and a given terminal state `y_end`,

```
f(y_{t+1}, y_t, y_{t-1}, eps_t) = 0,   t = 1, ..., T,
y_0 = y_init,   y_{T+1} = y_end,
```

without ever assembling the stacked Jacobian. The Newton direction is obtained from Jacobian-vector products by LGMRES, preconditioned with the exact inverse of the block-tridiagonal-in-time Jacobian evaluated at the steady state. It is the model-agnostic core of the IO research engine `dynamic_model/native_numerics.py`, `native_solver.py`, `native_block_preconditioner.py` and `native_krylov_probe.py`, ported with puremacro conventions. It is opt-in and separate: `solve_perfect_foresight` and `extended_path` keep their dense-block SuperLU path unchanged.

## What it computes

| Object | Role |
|---|---|
| `StackedProblem` | Stacked residual `F(Y)` and the matrix-free product `J(Y) v` from an analytic per-date `jvp_fn`, central differences or a complex step; per-date blocks `(A_t, B_t, C_t)` and the steady-state blocks |
| `BlockTridiagonalPreconditioner` | Exact block-Thomas (or SuperLU) inverse of `kron(I, B) + kron(S, A) + kron(S', C)`; `time_block=H` gives block Jacobi over H-date blocks |
| `StructuredBlockTridiagonalPreconditioner` | Protocol for model-specific eliminations: per-date diagonal solves with Jacobi, forward, backward or symmetric Gauss-Seidel sweeps, or a subclass overriding `apply` |
| `preconditioned_lgmres` | LGMRES or restarted GMRES with an explicit preconditioning side and the true residual of the returned direction |
| `solve_stacked_newton_krylov` | Inexact Newton harness: forcing term, bounded step, Armijo line search, linear history, checkpoints |
| `compare_horizons` | Horizon-doubling acceptance `max_{t <= periods} abs(y^T_t - y^2T_t) <= tolerance`; exported at package level as `puremacro.dsge.compare_stacked_horizons` (`puremacro.trade.dynamic` has its own `compare_horizons`) |
| `StackedNewtonResult`, `HorizonComparison` | Frozen result objects with `to_dataframe`, `summary`, `to_markdown`, `to_latex`, `to_typst`; `StackedNewtonResult.to_perfect_foresight_result()` returns a `PerfectForesightResult` |

## The stacked system and its Jacobian action

With `A_t = df_t/dy_{t+1}`, `B_t = df_t/dy_t` and `C_t = df_t/dy_{t-1}`, the stacked Jacobian is block tridiagonal in time and its action on a stacked direction `v` is

```
(J v)_t = A_t v_{t+1} + B_t v_t + C_t v_{t-1},    v_0 = v_{T+1} = 0.
```

`StackedProblem` evaluates this product in one of four ways:

- `jvp_fn(y_next, y_curr, y_prev, eps, v_next, v_curr, v_prev)`: analytic per-date product (exact; the IO engine's `state_jvp` / `jvp_from_states` pattern);
- `stacked_jvp_factory(Y) -> matvec`: a batched product supplied by the model;
- `jvp_method="central"` (default): `(F(Y + h v) - F(Y - h v)) / 2h` with `h = fd_step * max(1, |Y|_inf) / |v|_inf`, two residual evaluations per product, accurate to about 1e-9 relative;
- `jvp_method="complex"`: `Im F(Y + i h v) / h` with `h = 1e-20`, one complex evaluation, exact to machine precision, requires a complex-safe `equations_fn` (the same restriction as `solve_perfect_foresight(method="complex")`); a function that discards the imaginary part (`np.real`, `float()`, a float dtype) is detected and rejected with `TypeError` instead of silently producing a zero product.

The stacking convention is that of `solve_perfect_foresight`: date `t` occupies rows `t*n_vars:(t+1)*n_vars`, `y_prev` at the first date is `y_init` and `y_next` at the last date is `y_end`. Every row of `exogenous_path` is an interior date unless `n_periods` is given: then a path of exactly `T + 2` rows (the `solve_perfect_foresight` layout with both boundary rows) is trimmed to rows `1..T`, so pass `n_periods` whenever the path carries boundary rows, otherwise every shock lands one date later.

A model whose date-`t` residual also needs next date's exogenous values (the IO `NativeEconomy.equations(..., policy_t, policy_next)` pattern) packs both into each row, `exogenous_path[t] = (eps_t, eps_{t+1})`, with the last row repeating `eps_T`. Packing the values themselves, rather than a date index, keeps the held-exogenous rule of `compare_horizons` meaningful.

## The inexact Newton-Krylov iteration

At an iterate `Y` with residual `F` the direction `d` satisfies `|J d + F|_2 <= eta |F|_2` with the forcing term

```
eta = max(eta_min, min(eta_max, sqrt(|F|_inf / s))),    (eta_min, eta_max) = (1e-5, 0.1),
```

solved by SciPy's LGMRES with `inner_m = 40`, `outer_k = 6` and at most `krylov_maxiter = 35` outer iterations (the IO `native_solver._newton` settings). `krylov_method="gmres"` uses restarted GMRES instead; it is an explicit choice, not an automatic fallback. A direction returned at the Krylov cap is not an error: its status and true linear residual are recorded and it is line-searched like any other direction, as in the IO engine. `s` is the residual scale: by default the largest entry of the steady-state blocks floored at one; `jacobian_scale=1.0` gives the IO absolute residual rule. The move is bounded in path units, `step = min(1, max_log_move * max(1, |Y|_inf) / |d|_inf)` with `max_log_move = 0.75`: the IO engine bounds the absolute move of its log variables by 0.75, both rules coincide whenever `|Y|_inf <= 1`, and the relative form keeps a model written in levels of order 1e3 or 1e4 converging in the same six Newton steps as the unit-scale model (with an absolute bound the same model needs more than a thousand steps); `max_log_move=None` disables the bound. The step is accepted by the Armijo test `|F(Y + step d)|_2^2 <= |F(Y)|_2^2 (1 - 1e-4 step)`, halving at most `max_backtracks = 23` times (24 trial evaluations, the IO rule). Every accepted step is recorded in `linear_history` and, when requested, passed to `progress` and `checkpoint`.

Convergence uses the scale-aware rule of `solve_perfect_foresight`: `|F|_inf <= tol * s` and the next Newton direction below `step_tol * max(1, |Y|_inf)` with `step_tol = 1e-10`. A system whose equations are multiplied by 1e-12 or 1e8 therefore returns the same path, and the untouched initial guess is never returned. `step_tol=None` drops the step test and accepts on the residual alone, without a final linear solve; together with `jacobian_scale=1.0` this is the acceptance rule of the IO `native_solver._newton`. When the residual already meets `tol * s` but the step test cannot be met, typically because the residual sits at a noise floor (an inner iterative solve, finite-difference noise), the solve raises `StackedNewtonError` with `converged_by = "step_test_failure"` and a message that names the cause and the `step_tol=None` option. Every failure raises `StackedNewtonError` with the last state attached as `.result` (`converged=False`, `metadata["converged_by"]` naming the cause: `iteration_limit`, `line_search_failure`, `step_test_failure`, `linear_solve_failure` for an error or a nonfinite value inside a Jacobian product or preconditioner application, `nonfinite_direction` or `preconditioner_failure`); a failed solve never returns `converged=True`.

One caveat matters for anyone reading Krylov logs. SciPy's LGMRES calls its callback at the top of each outer iteration, before the residual check. When the iteration cap is exhausted, the direction returned to the caller was never passed to the callback, so the last logged residual does not describe it. `preconditioned_lgmres` therefore recomputes `|J d + F|_2` for the returned direction in original coordinates and the solver records it as `true_linear_residual` next to `requested_tolerance`. This is the IO finding behind `native_krylov_probe.make_lgmres` and its recorded drivers. The callback timing was verified on SciPy 1.18; on SciPy 1.10 and 1.11 the relative tolerance is passed as `tol` instead of `rtol`.

## Preconditioners

**Exact steady-state inverse (default).** With constant blocks `(A, B, C)` at `(y_end, y_end, y_end, eps_T)` the block-Thomas recursion

```
D_1 = B,  U_1 = D_1^{-1} A,  D_t = B - C U_{t-1},  U_t = D_t^{-1} A,
z_1 = D_1^{-1} r_1,  z_t = D_t^{-1} (r_t - C z_{t-1}),
x_T = z_T,  x_t = z_t - U_t x_{t+1},
```

inverts `J0 = kron(I_T, B) + kron(S, A) + kron(S', C)` exactly (`S` is the upper shift). Setup costs `O(T n^3)`, each application `O(T n^2)`, memory `O(T n^2)`: T dense LU factors of size n plus T-1 coupling matrices. This is the Laffargue-Boucekkine-Juillard relaxation written in matrix form; the difference from `solve_perfect_foresight` is that the factorization is done once, at the steady state, and reused by every Krylov iteration of every Newton step, while the current Jacobian only ever acts through products. `method="splu"` factors the same sparse matrix with SuperLU. The block-Thomas recursion needs every leading pivot `D_t` nonsingular, which a nonsingular stacked matrix does not guarantee: a rank-deficient `B` breaks it while SuperLU and `solve_perfect_foresight` both succeed. In floating point such a pivot is rarely exactly zero (rounding leaves entries of order 1e-17), so a pivot is rejected when LAPACK's reciprocal condition estimate falls below `n * eps`, and the finished factorization is checked once on a fixed random right-hand side: a normwise backward error `|J0 x - r|_inf / (|J0|_inf |x|_inf + |r|_inf)` above 1e-10 rejects it as well (a backward-stable solve gives about 1e-16 whatever the conditioning of `J0`; the regular fixtures give at most 2e-16). The named preconditioners of `solve_stacked_newton_krylov` then fall back to SuperLU and record it in the preconditioner name (`block_tridiagonal[splu, exact, thomas_pivot_fallback]`); only a stacked matrix that SuperLU cannot factor either raises `StackedNewtonError`, with the initial state attached.

**Time-block Jacobi.** `time_block=H` drops the links between consecutive H-date blocks: one factorization of length H is reused for every full block and a shorter remainder block gets its own, exactly as the IO `FiniteBlockCapitalPreconditioner` (`labels = repeat(arange(T)//H, n)`). `H = T` is the exact inverse; `H = 1` (the named option `"time_block_jacobi"`) keeps only `B^{-1}` per date.

**Current blocks.** `"current_block_tridiagonal"` refactors the preconditioner from the current per-date blocks at every Newton step. The Krylov solve then converges in one inner cycle, which makes it an exact Newton method at the cost of `3 n` equation calls per date and step, the same cost profile as `solve_perfect_foresight`. It exists mainly as an oracle.

**Structured protocol.** `StructuredBlockTridiagonalPreconditioner(n_periods, n_vars, diag_solve=, lower_apply=, upper_apply=, sweep=)` takes per-date diagonal solves `D_t^{-1} r_t` and the off-diagonal actions `C_t x` and `A_t x`, and applies a block Jacobi, forward `(D + L)^{-1}`, backward `(D + U)^{-1}` or symmetric `(D + U)^{-1} D (D + L)^{-1}` sweep. A subclass may instead override `apply` with a global elimination. The IO `CapitalPreconditioner` (N scalar capital-time tridiagonal solves followed by static national labor and budget LU solves) and `SchurCapitalPreconditioner` (national Schur complement kept block tridiagonal in time) have the structure of this second form, so a model-specific transition solver could supply such an elimination through it. Neither is ported, and in this release no puremacro module uses the protocol: the dynamic MRIO transitions of `puremacro.trade.dynamic` ship their own capital preconditioner and Newton loop. Any object with `as_linear_operator()`, any SciPy `LinearOperator`, or a factory `f(Y) -> operator` evaluated at every Newton iterate is accepted by `solve_stacked_newton_krylov`.

### Why the time coupling belongs in the preconditioner

Measured on the 20-sector Cobb-Douglas growth transition of `tests/test_dsge_stacked_newton.py` (`n_vars = 40`, `T = 100`, 4,000 unknowns, `tol = 1e-9`, `krylov_maxiter = 40`, development laptop, timings indicative only):

| Linear solver | Newton steps | LGMRES outer iterations per step | Seconds | Max abs difference to `solve_perfect_foresight` |
|---|---|---|---|---|
| `solve_perfect_foresight` (dense FD blocks, SuperLU) | 5 | not applicable | 1.22 | reference |
| `"steady_block_tridiagonal"` | 5 | 2, 2, 2, 2, 2, 2 | 0.15 | 3.5e-11 |
| `"steady_splu"` | 5 | 2, 2, 2, 2, 2, 2 | 0.11 | 3.5e-11 |
| `"time_block_jacobi"` (H = 1) | 6 | 4, 5, 5, 6, 7, 10, 11 | 3.4 | 1.2e-12 |
| `"current_block_tridiagonal"` | 4 | 2, 2, 2, 2, 2 | 1.4 | 8.9e-15 |
| `None` | fails | 40, 40, 40, 40, 40, 40, 40 (cap) | 24 | residual stuck at 0.31 |

The outer-iteration counts follow the IO convention (outer iterations entered, one more than the completed inner cycles of a converged solve); each list includes the final convergence-check solve. The exact steady inverse needs one inner cycle per Newton step; dropping the time links multiplies the Krylov work; dropping the preconditioner altogether makes LGMRES fail on a 4,000-unknown system. This reproduces the IO lesson that let the research engine solve an EXIOBASE transition with 4.48 million unknowns: the time coupling must be inside the preconditioner.

## Runnable example

The Ramsey model of `tests/test_dynare_perfect_foresight.py`, a transition from half the steady-state capital stock:

```python
import numpy as np
from puremacro.dsge import solve_perfect_foresight
from puremacro.dsge.stacked_newton import StackedProblem, solve_stacked_newton_krylov, compare_horizons

alpha, beta, delta, sigma = 0.33, 0.96, 0.10, 1.0
r_ss = 1 / beta - (1 - delta)
k_ss = (alpha / r_ss) ** (1 / (1 - alpha))
c_ss = k_ss ** alpha - delta * k_ss

def equations_fn(y_plus, y_curr, y_lag, eps):
    c_p, k_p = y_plus
    c, k = y_curr
    c_m, k_m = y_lag
    A = float(eps)
    euler = c ** (-sigma) - beta * c_p ** (-sigma) * (alpha * A * k ** (alpha - 1) + 1 - delta)
    resource = k - (A * k_m ** alpha + (1 - delta) * k_m - c)
    return [euler, resource]

y_ss = np.array([c_ss, k_ss])
y_init = np.array([c_ss, 0.5 * k_ss])
T = 60
problem = StackedProblem(equations_fn, y_init, y_ss, np.ones(T), variable_names=["c", "k"])
result = solve_stacked_newton_krylov(problem, tol=1e-8)
print(result.summary())
print(result.krylov_outer_iterations())          # (2, 2, 2, 2, 2, 2, 2)

reference = solve_perfect_foresight(equations_fn, y_init=y_init, y_ss=y_ss,
                                    exogenous_path=np.ones(T), n_periods=T, tol=1e-8,
                                    variable_names=["c", "k"])
print(np.max(np.abs(result.to_numpy() - reference.path.to_numpy())))   # about 5e-11

longer = solve_stacked_newton_krylov(
    StackedProblem(equations_fn, y_init, y_ss, np.ones(2 * T), variable_names=["c", "k"]), tol=1e-8)
print(compare_horizons(result, longer, periods=20, tolerance=1e-5).summary())
```

The horizon comparison passes at 60 versus 120 dates (`max_abs_difference` about 1e-7) and fails at 30 versus 60 (3e-3): the terminal condition is felt inside the first twenty dates of a 30-date solve. Passing at one horizon pair is evidence of horizon stability of the retained variables over the window only; it is not a certificate for the infinite-horizon problem. When both inputs are `StackedNewtonResult` objects, `compare_horizons` requires converged inputs, identical boundaries and the same exogenous path held beyond the short horizon (the IO rules); arrays, DataFrames and `PerfectForesightResult` inputs carry no boundaries or exogenous path, so they are compared as given (a `PerfectForesightResult` must still be converged).

## When to use it, and its limits

- Use it when the model already has cheap residual evaluations and either an analytic `jvp_fn` or a complex-safe `equations_fn`, and `T * n_vars` is large enough that assembling `3 n_vars` finite-difference columns per date and factoring the stacked matrix at every Newton step dominates. With central-difference products the linear solve is accurate to roughly 1e-9 relative, which is enough for the Newton iteration but not for a 1e-12 exact-inverse identity; use `jvp_fn` or `jvp_method="complex"` for that.
- Memory of the exact steady inverse is `O(T n^2)`: `n_vars = 200`, `T = 500` needs about 320 MB; `n_vars = 1000`, `T = 200` about 3.2 GB (`BlockTridiagonalPreconditioner.memory_bytes` reports the exact figure). Native MRIO systems with 7,000 to 9,500 unknowns per date are out of reach for this generic operator, and a structured elimination (the `StructuredBlockTridiagonalPreconditioner` route) is mandatory there. Quoting the IO documentation: "Preconditioner speed and memory must be established by executions; block structure alone does not establish native-scale performance."
- No native-scale MRIO transition is solved by this module. The IO record that motivated the port is mixed. The structured capital preconditioner carried an EXIOBASE transition to `T = 640` (4.48 million unknowns); its report and independent audit accept the first 20 dates and the horizon stability of stationary-tail welfare (IO status `verified_window_and_welfare`), while "full terminal-state convergence is not certified". The exact finite time-block inverse solved GTAP at `T = 10` and `T = 20`, and, quoting the IO README, "No GTAP horizon is accepted yet." The seven IO preconditioner variants beyond the exact time-block inverse (capital, Schur, periodic, swept, Galerkin, deflated, Jacobi-deflated) are not ported; the IO diagnostics recorded an amplification of about 2.5e8 by the swept block SGS smoother (GTAP minimal calibration, 40 dates) and warn that "These linear identities do not guarantee numerical accuracy or nonlinear convergence."
- Mixed complementarity (`mcp=True`) and the `surprise` rolling mode of `solve_perfect_foresight` are not available here.
- `direct_below` reproduces the IO MINPACK pre-solve for stacked systems of at most that many unknowns (the IO default is 240; puremacro's default 0 keeps every solve on the Krylov path so that `linear_history` is always complete).

## What is and is not validated

`tests/test_dsge_stacked_newton.py` checks, with stated tolerances:

- linear-algebra oracles: block-Thomas and SuperLU inverses equal dense solves to 1e-12 for constant and per-date blocks, time-block Jacobi equals the masked block-diagonal solve for constant and per-date blocks, linearity and scale invariance, the operator copies its blocks (mutating the inputs after construction changes nothing), SuperLU on a singular assembly raises `ValueError`, the four structured sweeps against the dense `D`, `L`, `U` formulas;
- pivot safeguards: a rank-deficient `B = U V` whose LU pivot is rounding-level rather than zero (two random seeds, stacked condition numbers 616 and 941) is rejected by block-Thomas, inverted by SuperLU to 1e-12, and solved by the default solver through the SuperLU fallback to the `solve_perfect_foresight` path to 1e-8; a pivot that passes the conditioning test but makes the unpivoted elimination lose about ten digits (`B = [[1, 1], [1, 1 + 1e-12]]`, stacked condition number 1.3e3) is rejected by the backward-error check; the check never fires on the regular fixtures (backward error at most 1e-14, observed 2e-16);
- stacked products: central differences agree with the column-wise dense Jacobian built with `solve_perfect_foresight`'s own step rule to 1e-6, complex step and the analytic Ramsey Jacobian to 1e-7 against it (the finite-difference reference itself is accurate to about 2e-9) and to 1e-12 against each other; an `equations_fn` that discards the imaginary part is rejected under the complex step;
- the exact-inverse identity: a preconditioner from the current blocks makes LGMRES and GMRES converge to relative 1e-12 in one inner cycle on random right-hand sides, for both sides;
- the callback-timing caveat: the IO 4-by-4 case where the cap is exhausted and the recomputed residual differs from the last callback residual by more than 0.01; the pre-1.12 SciPy signature (`tol` instead of `rtol`) is exercised by substituting legacy-signature solvers;
- parity with `solve_perfect_foresight` on the Ramsey transition, an anticipated TFP shock at `t = 5`, a histval-to-endval transition and the 20-sector growth fixture, all to 1e-8 (observed 5e-11 to 7e-11 on Ramsey, 3.5e-11 on the growth fixture); right and left preconditioning and GMRES agree with it to 1e-9; equation-scaling invariance to 1e-9 for scales from 1e-12 to 1e8 (the step test accepts a final direction up to `1e-10 * max(1, |Y|_inf)`, so two converged solves may differ by that much; observed 2.6e-10); variable-scaling invariance: the Ramsey model in units of 1e-3, 1, 1e3 and 1e4 converges with default options in at most ten steps to the unit-scale path to 1e-9 relative, with the same Newton and Krylov counts for the large units;
- the failure contract: a failed line search (24 trial evaluations by default, the recorded step being the last one tried), the iteration limit, an error or a nonfinite value inside a linear solve (a structured preconditioner turning nonfinite with right and left preconditioning, a `jvp_fn` that raises or returns NaN, a user operator that raises) and a residual at a noise floor that meets the tolerance while the step test cannot be met each raise `StackedNewtonError` with the last state attached and the cause in `converged_by`; an exhausted Krylov cap alone is recorded and not fatal (a capped solve still matches `solve_perfect_foresight` to 1e-8); a steady `B` of rank 1 with a nonsingular stacked matrix converges through the SuperLU fallback (and with `steady_splu` and `current_block_tridiagonal`) to the `solve_perfect_foresight` path, while a steady matrix that cannot be factored raises `StackedNewtonError` with the initial state attached; `step_tol=None` accepts on the residual alone;
- result objects: the stored trajectory and history records are read-only, copies are writeable, and results survive pickling, deep copies and JSON export of the history;
- horizon comparison rules and the 30/60/120-date statistic;
- parity with the IO engine on its analytic two-country fixture when the IO research workspace is available (`PUREMACRO_IO_ROOT`; the tests skip otherwise): `StackedProblem` reproduces `PathProblem` exactly (residual and analytic JVP), the steady block inverse reproduces `FiniteBlockCapitalPreconditioner(block_size=T)` to 5e-11 (observed 2e-13, T from 1 to 13, with and without adjustment costs, including block size 5 on 13 dates); with `side="left"` and `jacobian_scale=1.0` the harness reproduces `native_solver._newton` to 1e-13 (observed 0.0 on the development machine) on a 40-date announced-tariff transition, with the same Krylov counts on the shared iterations, and with `step_tol=None` added the whole Krylov history matches; the same path agrees with `solve_transition` (IO capital preconditioner) to 1e-9 (observed 3e-14); on a residual with a 1e-12 noise floor (300 unknowns) `step_tol=None` reproduces the IO acceptance and path (observed 0.0) where the default step test raises `step_test_failure`; and `compare_horizons` on the port's own results, with the boundary and held-exogenous rules active, returns the IO statistic on IO `solve_transition` solutions to 1e-12 with the same verdict for 20 versus 40 dates (1.2e-3, observed difference 2e-17) and 40 versus 80 dates (4.3e-5, observed difference 2e-15);
- a `slow`-marked preconditioner ranking regression.

Not validated: any model beyond these analytic fixtures, any native-scale system, memory or speed at scale, SciPy releases older than 1.18 beyond the signature substitution above, and the delegation from `solve_perfect_foresight` (see below).

## Relation to `solve_perfect_foresight` and `extended_path`

The two solvers share the problem statement and the stacking convention, and `StackedNewtonResult.to_perfect_foresight_result()` converts a result into the existing container. The planned integration, not implemented in this release, is a `linear_solver="krylov"` option on `solve_perfect_foresight` that builds a `StackedProblem` from its resolved `eq_fn`, `y_init_arr`, `y_ss_arr` and `exo_sim`, delegates to `solve_stacked_newton_krylov` and converts the result, with MCP mode rejecting the option, and the same option on `extended_path` factoring the steady-state preconditioner once and reusing it at every date. Until then, call this module directly with the callable form of the equations, exactly as in the example above.

## Related documentation

- [DSGE sketchpad and Dynare](dsge_build.md), section 8, for `solve_perfect_foresight`, MCP transitions and `extended_path`.
- [Structural validation status](STRUCTURAL_VALIDATION_STATUS.md) for the supported calculation and acceptance contract of this surface.
- [Quantitative spatial and trade general equilibrium](spatial_and_trade_ge.md) for the static trade solvers; the dynamic MRIO transitions of `puremacro.trade.dynamic` have their own capital preconditioner and Newton loop and do not use this module.
