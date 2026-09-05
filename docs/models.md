> 🇬🇧 English · 🇪🇸 [Español](es/models.md)

# Sequence-Space HANK

`puremacro.models` is the structural side of the package: models you solve for a
steady state and then push along a transition path, with no compiler, no MEX
toolchain and no Dynare in the loop. Everything on this page is offline — none
of it touches the network.

```python
from puremacro.models import solve_hank_sequence_space

res = solve_hank_sequence_space(T=40, n_a=100, beta=0.985, r_ss=0.01,
                                phi_pi=1.5, kappa=0.1,
                                shock_magnitude=0.0025, shock_rho=0.7)
print(res.summary())
```

That is a 25bp contractionary monetary shock in a one-asset heterogeneous-agent
New Keynesian economy, and it returns in **0.21 s**.

## What the package ships

| module | import path | what it solves |
|---|---|---|
| `hank_sequence_space` | `puremacro.models` | one-asset HANK: EGM steady state, exact $\mathcal{O}(T^2)$ Fake News algorithm, targeted fiscal transfers, and GE transition by one `T x T` linear solve |
| `dmp_regime_dependent` | `puremacro.models` | representative-firm DMP, Hall sticky wages, regime-dependent vacancy cost and TFP bump |
| `nested_dmp` (package) | `puremacro.models.nested_dmp` | heterogeneous-firm DMP with Bayesian beliefs about the policy rule: steady state, perfect-foresight IRF, IRF-matching estimation, recursive stochastic solve, welfare sweep |
| `smm` | `puremacro.models.smm` | moment loader and objective for estimating `dmp_regime_dependent` against local-projection IRFs |

`puremacro.models.__all__` re-exports:
`DMPParameters`, `DMPState`, `dmp_steady_state`, `dmp_irf`,
`SequenceSpaceHANKResult`, `solve_hank_sequence_space`,
`FakeNewsResult`, `FiscalTransferResult`,
`fake_news_algorithm`, and `simulate_targeted_transfer`. `nested_dmp` and `smm`
are reached by their own paths.

## What the sequence-space method buys

A global solution of a heterogeneous-agent model carries the distribution as a
state. You approximate it (Krusell-Smith moments, a projection basis, a neural
net), you solve a Bellman equation on the joint (idiosyncratic, aggregate) space,
and the cost grows with the aggregate state dimension. Auclert, Bardóczy,
Rognlie & Straub (2021) observe that if you only want the **linear** response to
an aggregate shock, you never need that object. You need one `T x T` matrix per
input — the Jacobian of aggregate consumption with respect to the date-`s` value
of each price — and equilibrium is then a linear system in the *sequence* of
aggregates, not in a state vector.

Two consequences, both visible in the wall clock:

- **Horizon is cheap.** At `n_a=20` the whole solve costs 0.056 s at `T=40`,
  0.182 s at `T=600` and 0.415 s at `T=1000`. Twenty-five times the horizon for
  7.4x the time: the `T x T` solve is BLAS-bound and the Jacobian fill is
  `O(T^2)`.
- **The household side is priced separately.** Its cost scales with `n_a`, not
  `T` — 0.12 s at `n_a=50`, 0.92 s at `n_a=400` — because the stationary
  distribution is a Python double loop over `(n_a, n_s)`. At any `n_a` above
  ~100 the household block, not the equilibrium solve, is what you are paying
  for.

The price of the linearity is the usual one: this is a first-order response
around a single steady state. It cannot tell you about a ZLB episode, a
state-dependent multiplier, or anything where the size or sign of the shock
changes the propagation.

---

## 1. Exact Fake News Algorithm (`fake_news_algorithm`)

Auclert et al. (2021) showed that computing the $T \times T$ sequence-space Jacobian directly by brute-force numerical simulation requires $T$ separate forward simulations. The **Fake News Algorithm** reduces this computational complexity to $\mathcal{O}(T^2)$ using backward expectation iteration.

The algorithm defines the *fake news matrix* $\mathcal{F}$ where $\mathcal{F}_{t,s}$ represents the revision in the agent's expectation of period-$t$ consumption upon receiving news at date 0 about an innovation occurring at date $s$:
$$\mathcal{F}_{t,s} = (\mathbf{D}_{ss} \mathcal{E}_t)' \cdot d\mathbf{a}^*_s$$

Once $\mathcal{F}$ is formed, the full sequence-space Jacobian $\mathcal{J}$ is recovered through the fundamental cumulation identity:
$$\mathcal{J}_{t,s} = \mathcal{J}_{t-1,s-1} + \mathcal{F}_{t,s}$$

```python
from puremacro.models import fake_news_algorithm

# Compute exact sequence-space consumption Jacobians
fn_res = fake_news_algorithm(T=40, n_a=100, beta=0.985, r_ss=0.01)
print(fn_res.summary())

# Access matrices as tidy DataFrames
df_jac = fn_res.to_frame(which="jacobian")
df_f   = fn_res.to_frame(which="fake_news")

# 1-line publication visualization: heatmaps of F and J
fn_res.plot()

# Export table to LaTeX or Typst
print(fn_res.to_latex())
```

---

## 2. Targeted Fiscal Transfers (`simulate_targeted_transfer`)

Heterogeneous-agent models allow realistic macro-policy simulations that representative-agent models cannot evaluate, such as the macroeconomic stimulus effects of targeted stimulus checks across income and wealth deciles.

`puremacro.models.simulate_targeted_transfer` simulates the aggregate and distributional consequences of an emergency fiscal transfer:

```python
from puremacro.models import simulate_targeted_transfer

# Stimulus targeted to bottom 30% wealth constrained households
transfer_res = simulate_targeted_transfer(
    transfer_amount=500.0,
    target_deciles=[1, 2, 3],
    T=30,
)

print(transfer_res.summary())
print("Aggregate impact MPC:", transfer_res.impact_mpc)
print("Cumulative fiscal multiplier:", transfer_res.cumulative_multiplier)

# Dual-panel publication visualization: consumption IRF and wealth-decile incidence
transfer_res.plot()
```

---

## 3. Parametric Sequence-Space Solver (`solve_hank_sequence_space`)

In addition to the exact Fake News engine, `solve_hank_sequence_space` provides an ultra-fast closed-form general equilibrium solver that ties steady-state EGM policy to analytical intertemporal Jacobians:

```
decay = 1 - agg_mpc

t >= s:   J_C_Y[t, s] = agg_mpc * decay**(t - s)
          J_C_r[t, s] = -(1/gamma) * beta**(t - s + 1) * 0.8**(t - s)

t <  s:   J_C_Y[t, s] = agg_mpc * 0.5**(s - t)
          J_C_r[t, s] = -(1/gamma) * 0.5**(s - t)
```

The `0.8` and the `0.5` are constants in the source, not model objects. So:

- Every argument that touches the household problem — `n_a`, `a_max`, `r_ss`,
  and the hardcoded income process — reaches the impulse responses **only
  through `agg_mpc`**. Raising `n_a` from 30 to 400 moves the MPC from 0.1040 to
  0.1026 and the peak output response from −0.00255 to −0.00254.
- Two calibrations with the same aggregate MPC and different wealth
  distributions return identical IRFs.
- `mpc_distribution`, the decile profile, enters no equilibrium object. It is
  reported, not used.

The equilibrium block above the Jacobians is a genuine sequence-space solve, and
the household block is a genuine Aiyagari-Bewley-Huggett solve. What is
approximated is the bridge between them. Treat the IRFs as a calibrated
two-parameter (`agg_mpc`, `gamma`) intertemporal-MPC model, and the wealth
distribution and decile MPCs as the real heterogeneous-agent output.

## `solve_hank_sequence_space` — arguments

Keyword-only, all of them.

| argument | default | what it does |
|---|---|---|
| `T` | `40` | truncation horizon: length of every IRF, side of every Jacobian |
| `beta` | `0.985` | household discount factor **and** the NKPC discount factor — one knob for both |
| `gamma` | `1.0` | CRRA. Enters `J_C_r` as `1/gamma`, so it is the intertemporal-substitution elasticity |
| `r_ss` | `0.01` | steady-state quarterly real rate. An input, not an equilibrium outcome |
| `phi_pi` | `1.5` | Taylor-rule inflation coefficient |
| `kappa` | `0.1` | NK Phillips curve slope |
| `shock_magnitude` | `0.0025` | monetary shock at `t=0`. Positive = contraction |
| `shock_rho` | `0.7` | AR(1) persistence of the shock, `eps_t = shock * rho**t` |
| `n_a` | `50` | asset grid points |
| `a_max` | `30.0` | top of the asset grid |

What is **not** an argument, because it is fixed in the source: the income
process (two states, `[0.5, 1.5]`, symmetric transition with 0.9 persistence),
the steady-state wage (`w_ss = 1.0`), the borrowing limit (`a' >= 0`), and every
tolerance and iteration cap.

There is no asset-market clearing condition. `r_ss` is imposed and the implied
aggregate saving is whatever it is — at the defaults, mean assets of 7.02
against mean labour income of 1.0. If you need a bond-supply-consistent `r`, you
have to search over `r_ss` yourself.

## The GE block

Three equations, assembled as matrices and solved once.

```
NKPC        pi = K_pi @ dY,            K_pi[t, s] = kappa * beta**(s - t) for s >= t
Taylor      i_t = phi_pi * pi_t + eps_t,   r_t = i_t - pi_{t+1}
            M_r_Y = phi_pi * K_pi - shift(K_pi)
clearing    dY = dC = J_C_Y @ dY + J_C_r @ (M_r_Y @ dY + eps)
```

which rearranges to `(I - J_C_Y - J_C_r @ M_r_Y) dY = J_C_r @ eps` and goes
straight into `np.linalg.solve`. Goods market clearing is `Y = C` with no
investment, no government and no labour block, which means **`irf_consumption`
is a bit-identical copy of `irf_output`**. Comparing them is not a check on
anything.

## What comes back

`SequenceSpaceHANKResult`, a frozen dataclass.

| field | shape | what it is |
|---|---|---|
| `irf_output` | `(T,)` | GE output response `dY` |
| `irf_consumption` | `(T,)` | identical to `irf_output` (see above) |
| `irf_inflation` | `(T,)` | `K_pi @ dY` |
| `irf_rate` | `(T,)` | real rate `M_r_Y @ dY + eps` |
| `jacobian_c_r` | `(T, T)` | consumption Jacobian w.r.t. the real rate |
| `jacobian_c_y` | `(T, T)` | consumption Jacobian w.r.t. aggregate income |
| `steady_state_mpc` | scalar | aggregate quarterly MPC, `sum(mpc_grid * D)` |
| `mpc_distribution` | `pd.Series`, 10 rows | average MPC by wealth decile, indexed `"Decile 1"`…`"Decile 10"` |
| `asset_grid` | `(n_a,)` | the geometric asset grid, `0` to `a_max` |
| `steady_state_wealth_dist` | `(n_a,)` | stationary marginal distribution over assets, sums to 1 |

`res.summary()` prints the horizon, the aggregate MPC, the peak output and
inflation responses, and the ten deciles.

At the shipped defaults: aggregate MPC 0.1026, bottom-decile MPC 0.672,
top-decile 0.020, peak output response −0.00254 at `t=1`. (Read the `n_a`
section before quoting the deciles — at `n_a=50` one of them is not real.)

## Numerical knobs, and what a too-small one does

### `T` — the truncation horizon

The solve assumes the economy is back at the steady state at `T`, so `T` has to
outlast the shock. Measured against a `T=400` reference at the default
`shock_rho=0.7`:

| `T` | impact-response error | size of the last IRF entry, `abs(dY[T-1])` |
|---|---|---|
| 8 | 5.97% | 1.7e−3 (72% of that solve's own impact response) |
| 12 | 1.22% | 8.0e−4 |
| 20 | 0.07% | 2.5e−4 |
| 40 *(default)* | 0.00% | 3.7e−5 |
| 100 | 0.00% | 1.7e−7 |

`abs(res.irf_output[-1])` is the diagnostic to check: if the last entry of the
IRF is not small relative to the impact response, `T` is too short and the
boundary condition is being imposed on a live path. Persistence moves the requirement —
at `shock_rho=0.95`, `T=20` is off by 3.5% and `T=40` by 0.015%.

A second, subtler bound comes from the Jacobian itself. Column `s` of `J_C_Y`
sums to `1 + agg_mpc` ≈ 1.10 in the interior but falls to 0.205 — twice
`agg_mpc` — in the last column: the income response is cut off before it has
played out. The shortfall is `(1 - agg_mpc)**(T - s)`, a function of `T - s`
alone, so **the truncated band is the same width whatever `T` is** — at the
default MPC of 0.102 column `T-20` is still 10% short, and the trailing sums at
`T=40` and at `T=100` are identical to machine precision. `T` has to leave a
usable interior beyond that band:

```python
res = solve_hank_sequence_space()          # shipped defaults, T=40
col_sums = res.jacobian_c_y.sum(axis=0)
col_sums[:5]    # 0.987 1.037 1.061 1.072 1.076 — climbing toward 1.1026
col_sums[-5:]   # 0.521 0.454 0.380 0.297 0.205 — truncated
```

### `n_a` — asset grid points

The aggregate MPC is stable in `n_a` (0.115 at 10, 0.103 at 50, 0.102 at 100,
0.103 at 400).
The **decile profile is not**, and the failure is silent. `mpc_distribution` bins
by cumulative mass with `np.searchsorted`; when two decile boundaries land on the
same grid index, the bucket has zero mass and the code falls back to writing the
*aggregate* MPC into that decile:

| `n_a` | deciles reading the aggregate MPC instead of their own |
|---|---|
| 10 | 6 of 10 |
| 20 | 3 |
| 30 | 2 |
| 50 *(default)* | 1 — decile 6 reads 0.1026, the aggregate |
| 100 | 0 |
| 200, 400 | 0 |

**At the default `n_a=50`, one of the ten deciles you plot is not a decile MPC.**
Use `n_a >= 100` whenever the decile profile is the output you care about; it
costs 0.21 s instead of 0.12 s.

### `a_max` — the top of the grid

Assets are censored at `a_max`, and the censored mass piles on the last node:

| `a_max` | mass on the top grid point | mean assets |
|---|---|---|
| 5 | 0.398 | 3.18 |
| 10 | 0.231 | 5.25 |
| 30 *(default)* | 0.0091 | 7.02 |
| 100 | 0.0000 | 7.16 |

Check `res.steady_state_wealth_dist[-1]` after any change to `beta` or `r_ss`.
The default already carries 0.9% of households censored at the top; anything
appreciably above that means the grid is binding and the right tail of the
wealth distribution is an artefact of where you cut it.

### `beta` — the admissibility bound nothing enforces

A stationary Aiyagari distribution needs `beta * (1 + r_ss) < 1`. At the default
`r_ss = 0.01` that is `beta < 1/1.01 = 0.990099`. The solver does not check it:

| `beta` | `beta(1+r)` | aggregate MPC | mass at `a_max` |
|---|---|---|---|
| 0.985 *(default)* | 0.994850 | 0.1026 | 0.009 |
| 0.99 | 0.999900 | 0.0069 | 0.561 |
| 0.9901 | 1.000001 | 0.0053 | 0.604 |
| 0.995 | 1.004950 | 0.0000 | **1.000** |

At `beta=0.995` every household sits on the top grid point and the MPC is
exactly zero, and the function returns without complaint. The IRFs barely move
(impact −0.00210 against −0.00222) because, per the section above, the household
block reaches them only through that one scalar — so a degenerate wealth
distribution does not announce itself in the output either.

### The tolerances you cannot reach

Both inner loops are hardcoded and neither reports failure:

| loop | tolerance | cap | behaviour at the defaults |
|---|---|---|---|
| EGM policy iteration | `1e-6` sup-norm on `c` | 300 | **exits on the cap, not the tolerance** — the residual is still 9.7e−6 at iteration 299 |
| stationary distribution | `1e-8` sup-norm on `D` | 500 | converges at iteration 467 |

At the shipped calibration neither matters much (raising the caps tenfold moves
the aggregate MPC from 0.102609 to 0.102619). At `beta=0.99` **both** loops
exhaust their caps. There is no `converged` flag on the result to check, so if
you move `beta` or `r_ss` far from the defaults, verify the wealth distribution
by eye rather than trusting the return.

---

# The nested heterogeneous-firm DMP

`puremacro.models.nested_dmp` is a different animal: a search-and-matching model
with a productivity distribution over firms, a non-convex hiring cost, and firms
that are **Bayesian about the central bank's reaction function**. The question it
is built to answer is why an uncertainty shock raises unemployment in one
monetary regime and lowers it in another.

The mechanism is one line of `kernels.expected_discount`. Firms do not know
whether the Fed is a dove (`phi_D < 0`, cuts into uncertainty) or a hawk
(`phi_H > 0`, hikes). They discount at the belief-weighted rate

```
E[beta | pi, sigma] = pi/(1 + r_D) + (1 - pi)/(1 + r_H),
    r_tau = r_star + (phi_tau - phi_sigma) * sigma
```

whose slope in uncertainty `sigma` flips sign at
`pi* = phi_H / (phi_H - phi_D)` — **0.319 at the default calibration**. Above
`pi*` more uncertainty makes firms more patient and job creation expands; below
it, the reverse. At `sigma = 0` the two type-rates coincide, so beliefs are
inert and every `pi` gives the same equilibrium. That is the whole model in a
comparative static:

```python
import numpy as np
from puremacro.models.nested_dmp import comparative_statics
from puremacro.models.nested_dmp.estimation import default_baseline

p  = default_baseline()          # see "Setting up a steady state" below
cs = comparative_statics(p, pi_grid=np.linspace(0.05, 0.95, 7),
                         sigma_grid=np.array([0.0, 0.25, 0.5]))
cs.pi_star          # 0.319
cs.theta[:, 0]      # sigma=0: flat at 3.5666 for every pi
cs.theta[:, 2]      # sigma=0.5: 3.461 ... 3.867, crossing 3.5666 near pi*
```

Because free entry means the cross-sectional distribution never feeds back into
prices, the aggregate state is just the two-dimensional `(sigma, pi)`. There is
no Krusell-Smith problem here, and the transition path is exact rather than
approximated.

## Setting up a steady state

The default `NestedDMPParameters` are a *theory* calibration, not a
data-matched one, and they do not produce usable dynamics: at the defaults the
job-finding rate is `f(theta_ss) = 2.07` — above 100% — which makes the forward
unemployment law of motion divergent rather than a contraction. `simulate_irf`
refuses to run on it:

```
ValueError: f(theta_ss)=2.074 >= 1: the forward unemployment law of motion
is unstable. Normalize mu via calibrate_mu for stable IRF dynamics.
```

Three normalizers are shipped. `calibrate_mu` pins `theta_ss = 1` (a stable
anchor, but `f` lands at 0.035). `calibrate_mu_to_f(p, f_target=0.5)` pins the
job-finding rate instead. `estimation.baseline_calibration` is the one to use:
it wraps `calibrate_mu_to_f` in an outer root-find and solves jointly for
`(mu, s_bar)` hitting a steady-state unemployment rate (`u_target=0.058`) and a
job-finding rate (`f_target=0.7`), because the two are coupled — changing
`s_bar` moves `theta_ss` and hence `f`.

```python
from puremacro.models.nested_dmp import solve_steady_state
from puremacro.models.nested_dmp.estimation import default_baseline

p  = default_baseline()          # targets u = 0.058, f = 0.70; takes ~60 s
                                 # mu -> 0.3672, s_bar -> 0.01412
ss = solve_steady_state(p, sigma=0.0)
ss.theta, ss.urate, ss.beta_bar, ss.free_entry_residual
# 3.567, 0.0499, 0.9901, -1.7e-14
```

`default_baseline` and `baseline_calibration` live in `estimation.__all__` but
are **not** re-exported by `puremacro.models.nested_dmp`, so they need the full
module path.

`SteadyState` carries the state it was solved at (`sigma`, `pi`), the tightness
`theta`, the belief-weighted discount `beta_bar`, the separation threshold
`x_sep` and posting threshold `x_post`, levels `u`/`n`/`v`/`N` (with `.E` and
`.U` as aliases of `n` and `u`), rates `urate`/`jf_rate`/`jd_rate`/`s_eff`,
`lfpr`, `output`, the worker values `W_U`/`W_E`, and the arrays `grid`, `J`,
`phi`. Note that `u` and `n` are **levels** — scaled by `lfpr` — while `urate`
is the rate within the labour force; with `h_max=0` (the default) `lfpr` is 1
and the two coincide. `free_entry_residual` is recomputed at the solution and
should be ~1e−14; `converged` is set unconditionally to `True`, so it is not a
check.

### Two steady states that do not agree

`equilibrium.solve_steady_state` discretizes the productivity process with
**Rouwenhorst**; the dynamics, estimation and welfare code all run on a **fixed
Rouwenhorst grid with Tauchen transitions** (`dynamics._fixed_grid` plus
`_P_sigma`). The two do not give the same answer:

| route | `theta` | `u` | `f` |
|---|---|---|---|
| `dynamics._grid_steady_state` (what everything dynamic uses) | 3.634 | 0.0586 | 0.700 |
| `equilibrium.solve_steady_state` | 3.567 | 0.0499 | 0.694 |

`default_baseline` calibrates against the *first* row, and lands at 0.0586
rather than exactly 0.058: the posting threshold `x_post` is read off the
productivity grid, so `u` steps discontinuously in `s_bar` and the outer
root-find brackets a jump rather than a crossing. Quoting
`solve_steady_state(...).urate` as the calibrated unemployment rate is off by
0.9pp. The design reason for the split is real — dynamics needs a grid that
stays put while the innovation sd varies over a transition, which Rouwenhorst
cannot give — but the two routes are not interchangeable.

## Taking a transition path

```python
from puremacro.models.nested_dmp import simulate_irf

dove = simulate_irf(p, sigma0=1.0, fed_type="dove", horizon=8)
hawk = simulate_irf(p, sigma0=1.0, fed_type="hawk", horizon=8)

dove.log_urate   # -0.036 -0.068 -0.083 -0.090 -0.093 -0.094 -0.096 -0.097 -0.098
hawk.log_urate   # +0.012 +0.011 +0.009 +0.006 +0.004 +0.003 +0.002 +0.002 +0.001
```

That is the sign flip, and it takes 0.53 s for the pair.

The mechanics: `sigma_t = sigma0 * rho_sigma**t`, zero at `T_simul`; beliefs
follow `belief_path` from a regime-specific prior (`pi_loose=0.9`,
`pi_hawk=0.1`, chosen to straddle `pi*`); `theta` and `J` come from a backward
sweep with the terminal condition at the fixed-grid steady state; `u` and `phi`
come from a forward simulation with `u` **predetermined**.

Three alignment facts that will otherwise cost you an afternoon:

- **The `t=0` unemployment response is mechanically zero**, because `u` is a
  state. `IRFResult` therefore returns model periods `1 .. horizon+1` against
  empirical horizons `0 .. horizon` — the standard `+1` DMP-vs-LP shift, applied
  inside the function.
- **The base is a flat simulation at the `sigma=0` steady state**, not the same
  path with `sigma0=0`. With slow job finding `u` may not have returned by
  `T_simul`, and differencing against a decaying base would put that residual in
  the IRF.
- `T_simul` (default 60) **must exceed `horizon + 1`** or the call raises. It is
  also where the terminal condition is imposed, so it plays the role `T` plays
  in the HANK block: short `T_simul` pushes the steady-state boundary into a
  live path.

`IRFResult` carries `fed_type`, `horizons`, and the log-deviation paths
`log_urate`, `log_v`, `log_theta`, `log_lfpr`.

### `n_x` — where the firm grid bites

`n_x` (default 25) is the productivity grid. The separation threshold `x_sep` is
read off as the lowest grid point with `J > 0`, so it can only move in grid
steps, and the unemployment rate inherits that jitter:

| `n_x` | `theta` | `urate` | `x_sep` |
|---|---|---|---|
| 5 | 3.595 | 0.0493 | 0.000 |
| 15 | 3.590 | 0.0665 | 0.000 |
| 25 *(default)* | 3.567 | 0.0499 | −0.131 |
| 51 | 3.581 | 0.0650 | −0.091 |
| 101 | 3.576 | 0.0591 | −0.128 |
| 201 | 3.576 | 0.0608 | −0.136 |

`theta` is settled to within 1% by `n_x = 5`. The **unemployment rate is not** —
the default `n_x = 25` reads 0.0499 against 0.0608 at `n_x = 201`, a 1.1pp
(20% relative) gap that does not shrink monotonically. Any calibration target
expressed in `u` is a target conditional on `n_x`. `n_x = 201` costs 0.11 s per
steady state against 0.02 s, so there is no reason not to check.

## Estimation

`estimation` matches the model IRFs to state-dependent local projections in
**normalized-shape space** — each curve divided by its own peak — with the
magnitude gap reported rather than targeted, on the stated grounds that the lean
free-entry model reproduces the sign flip but under-responds in level.

```python
from puremacro.models.nested_dmp.estimation import (
    load_empirical_irf_targets, fit_report, estimate_shape_match, ablate_beliefs,
)

targets = load_empirical_irf_targets("tests/fixtures/companion")
rep = fit_report(p, targets)
rep.sign_match        # {"urate": bool, "v": ..., "theta": ...}
rep.shape_distance    # inverse-LP-variance-weighted SSE on peak-normalized curves
rep.magnitude_ratio   # |empirical peak| / |model peak|, per regime and outcome
rep.within_band       # fraction of horizons inside the empirical 90% band
```

`estimate_shape_match` minimizes `shape_distance` over `("signal_sd",
"rho_sigma")` by bounded Nelder-Mead, clamping each draw into its admissible
region and charging `1e12` for a non-converging one. `ablate_beliefs` is the
identification check: run both regimes at one common belief (`pi_common=0.5`)
and the sign flip disappears, because it was the two regimes sitting on opposite
sides of `pi*` that produced it.

`load_empirical_irf_targets` reads `urate_lp_targets.csv` and
`vacancies_lp_targets.csv` from the directory you hand it, converts the
unemployment-rate coefficients from percentage points to log deviations by
dividing by `MEAN_URATE_PP = 5.8`, and derives `theta = log V - log U`. The only
copies of the two CSVs in the repository are the ones under
`tests/fixtures/companion/`, which is why the example points there.

## Welfare and the policy lever

`welfare` adds the piece the perfect-foresight IRF cannot give: a genuine
stochastic steady state, so you can rank policies by long-run average welfare.
`phi_sigma` is the lever — a systematic "lean against uncertainty" term that
cuts *both* type-rates by `phi_sigma * sigma`, raising `E[beta]` uniformly while
leaving the type gap, and therefore `pi*`, untouched.

```python
import numpy as np
from puremacro.models.nested_dmp import solve_recursive, ergodic_welfare, optimal_phi_sigma

sol = solve_recursive(p, n_sigma=11, n_pi=21)      # 0.8 s
w   = ergodic_welfare(p, sol, T=10_000, burn=1_000, seed=0)
rep = optimal_phi_sigma(p, grid=np.array([0.0, 0.005, 0.01]))
```

`solve_recursive` returns `RecursiveSolution` with `theta` of shape
`(n_pi, n_sigma_kept)`, `J` of shape `(n_x, n_pi, n_sigma_kept)` — `(21, 4)` and
`(25, 21, 4)` at the call above, because the sigma grid is clipped before it is
allocated (first bullet below) — and every transition object it built —
`Psig`, `Ppi_dove`, `Ppi_hawk` — so the simulator does not rebuild them.
`ergodic_welfare` returns `welfare` (average utilitarian flow:
output plus home production minus vacancy cost), `mean_urate`, `mean_theta`,
`std_urate`. `optimal_phi_sigma` returns `WelfareReport` with the full
`welfare_curve`, `phi_sigma_star`, `welfare_star`, `welfare_zero` and `cev_pct`.

Six things to know before reading a number off any of them:

- **The sigma grid gets clipped hard, and the docstring understates it.** The
  Bellman map contracts only where `beta_bar * (1-delta) * (1-s_bar) < 1`, and a
  dovish belief at high `sigma` pushes `beta_bar` above 1. `solve_recursive`
  therefore drops every node above `sigma_safe`, which is **0.768** at the
  default calibration. The unclipped Tauchen grid spans `[0, 2.10]`, so **4 of
  11 nodes survive at `n_sigma=11`** (2 of 5 at `n_sigma=5`, 8 of 21 at
  `n_sigma=21`). The source comment says this "clips off the top ~15-20%"; it
  clips off 64% of the nodes. Raising `n_sigma` buys resolution inside
  `[0, 0.768]`, not more coverage of the tail — the ceiling is `sigma_safe`
  whatever `n_sigma` is.
- The clip bound is computed at `phi_sigma = 0` on purpose, so every point of a
  `phi_sigma` sweep sits on the same grid and `(jp, js)` indices are comparable
  across the sweep. `phi_sigma > 0` raises `beta_bar`, and mild exceedances
  (`disc` up to 1.05) are tolerated; past that it raises.
- The `sigma` transition is rebuilt by Tauchen on the *clipped* support, so all
  the mass that would have gone above the top surviving node (0.63 at
  `n_sigma=11`) lands on it instead. The simulated `sigma` process is not the
  AR(1) you parameterized.
- `ergodic_welfare` refuses to run when `f(theta_ss) >= 1`, for the same reason
  `simulate_irf` does — it will not mask an unstable law of motion by clipping
  `phi` back into the simplex. Calibrate first.
- `optimal_phi_sigma` re-solves and re-simulates at every grid point, uses the
  **same seed** at each so the ranking reflects the lever rather than sampling
  noise, requires `0.0` to be in the grid as the CEV reference, and records
  infeasible points as `-inf` rather than aborting the sweep. Budget ~1.9 s per
  point at `n_pi=25, T=2000`, ~6 s at the default `T=10_000`.
- `cev_pct` is `100 * (welfare_star / welfare_zero - 1)` — a ratio of average
  utilitarian *flows*, not a consumption-equivalent variation derived from a
  utility function. On a 3-point sweep at the baseline it reads 0.004%.

## The parameter object

`NestedDMPParameters` is frozen and validates in `__post_init__`; every bound
below raises rather than clipping. Seven layers:

| layer | parameters | notes |
|---|---|---|
| matching / DMP core | `beta=0.99`, `alpha=0.5`, `mu=1.0`, `s_bar=0.03`, `b=0.4` | Hosios is the baseline: bargaining share = matching elasticity = `alpha` |
| heterogeneous firms | `rho_x=0.95`, `sigma_x=0.10`, `n_x=25`, `sigma_x_loading=0.0` | `sigma_x_loading=0` turns the Bloom mean-preserving-spread channel **off**, so `sigma` reaches the model through beliefs alone; set `1.0` to restore it |
| non-convex hiring | `f_fixed=0.05`, `c_convex=0.20` | `f_fixed` is the entry hurdle that creates the posting threshold `x_post` |
| wage rigidity | `psi_down=0.0` | asymmetric downward rigidity; an appendix lever, not the baseline (see below) |
| information / beliefs | `r_star=0.01`, `phi_D=-0.0320`, `phi_H=0.015`, `signal_sd=1.0`, `prior_pi0=0.5` | `phi_D`/`phi_H` are **decimal rate units**, not percentage points: −320bp and +150bp |
| policy lever + shock | `phi_sigma=0.0`, `rho_sigma=0.7`, `delta=0.0`, `h_max=0.0` | `h_max=0` turns participation off and collapses to the two-state `{E, U}` core |
| exogenous sigma process | `sigma_bar=0.0`, `sigma_innov_sd=0.5`, `p_regime=0.9` | the welfare backbone |

Two validations are worth calling out because they encode economics rather than
type-checking. `phi_D >= 0` and `phi_H <= 0` both raise — the sign convention
"dove cuts, hawk hikes" is enforced, not assumed. And the constructor checks
that the perceived rate at the dovish extreme (`sigma = 2`) stays above `-1`, or
`beta(r) = 1/(1+r)` crosses its pole.

`psi_down > 0` is documented in the source as a robustness lever rather than a
baseline: because separation is a hard threshold in `x`, a sticky wage that
stays high as the dove expansion fades can dump a dense mass of firms across the
threshold in one step and produce a spurious one-period unemployment spike.
`default_baseline` keeps it at 0.

## Backends

`nested_dmp.backend` is a shim onto `puremacro._backend`, whose `SUPPORTED`
tiers are `("numpy", "numba", "mlx", "cupy")`. Three of them are array
namespaces `get_array_namespace` can hand back; `"numba"` is not — it is
compiled kernels, and asking for its namespace raises.

```python
from puremacro.models.nested_dmp import available_backends, backend_available

available_backends()      # ('numpy', 'numba') on a machine without MLX or CuPy
```

Only the Bellman VFI has a compiled counterpart —
`kernels_numba.bellman_iterate_numba` and a `prange` batch version — and it
mirrors `kernels.bellman_iterate` exactly, validated against the NumPy oracle in
the cross-backend tests. `rouwenhorst` and `tauchen_transition` are NumPy-only
by design: one-time setup, in-place matrix assembly, never on a hot path.
Importing the Numba kernels needs the optional `[backend]` extra.

---

# The representative-firm DMP

`dmp_regime_dependent` is the single-firm counterpart, and the model the SMM
machinery in `models/smm.py` estimates. Cobb-Douglas matching, free entry, Hall
(2005) sticky wages, and two regime channels: a precautionary vacancy-cost bump
under a tight regime (`c_eff = c(1 + eta*sigma)` when `R = H`) and a TFP bump
under a loose one (`y_eff = y_bar + kappa*sigma` when `R = L`).

```python
from puremacro.models import DMPParameters, dmp_steady_state, dmp_irf

p = DMPParameters(beta_bar=0.996, alpha=0.5, eta_b=0.5, s=0.034, z=0.51,
                  y_bar=1.0, c=0.5, mu=0.5, eta=0.067, kappa=0.040,
                  rho_sigma=0.7, psi=0.5)

ss = dmp_steady_state(p, sigma=0.0, regime="H")
# theta 0.841, u 0.069, v 0.058, J 0.921, w 0.965

dmp_irf(p, sigma_shock=1.0, regime="H").log_u   # +0.024 +0.027 +0.024 ...
dmp_irf(p, sigma_shock=1.0, regime="L").log_u   # -0.033 -0.029 -0.020 ...
```

The regime effect is a sign flip in the unemployment response, arrived at by a
different route from the nested model's: here it is two separate exogenous
channels, there it is one belief that crosses a threshold.

Solution is a damped fixed point on the wage path — backward sweep for `J` and
`theta` given wages, forward sweep for `u` and the Nash wage, then
`w_{k+1} = (1-lam) w_k + lam * w_tilde` with `fp_damping=0.3`. The damping is
not cosmetic: the composed map has Jacobian modulus above 1 through the `c_eff`
channel, and `lam = 1` oscillates. Damping changes the approach, never the fixed
point.

Two structural notes from the source worth carrying into any read of the output:

- **Even at `psi = 0` there is one period of wage rigidity**, because `w_0` is
  pinned at `w_ss` — the wage was set at `t = -1`, before the shock. That is the
  explicit lagged-wage state of the Hall construction. An earlier brentq solver
  without that boundary produced larger same-period `theta` responses; the
  fixed-point solver is the structurally correct one.
- **The `+1` alignment is the same as the nested model's.** `u` is
  predetermined, so model `t = h + 1` maps to empirical horizon `h`, and
  `T_simul` (default 60) must exceed `horizon + 1` or the call raises.

The extra channels are all off by default and each is a single scalar:
`phi_AD` (aggregate-demand feedback, fires under `L` only), `alpha_k` and
`lambda_k` (capital), `alpha_N` and `lambda_N` (firm mass, fires under `H`
only), `lambda_mu` (regime-asymmetric matching efficiency — the structural
counterpart of the Beveridge-curve sign reversal), `phi_v` (quadratic
vacancy-adjustment cost, which leaves the steady state invariant but gives the
IRF a hump), `alpha_react` (lagged-`u` search friction) and `nu_lfpr`
(participation, populating `DMPIRF.log_lfpr` only when non-zero).

## SMM

`models/smm.py` builds the moment vector from the local-projection output
tables and scores `dmp_regime_dependent` against it.

```python
from pathlib import Path
from puremacro.models.smm import (
    load_empirical_moments, theoretical_moments, smm_objective,
)

emp = load_empirical_moments(Path("."))     # reads <root>/notebooks/output_tables/*.csv
smm_objective(p, emp)                       # sum_i w_i (model_i - data_i)^2
```

`notebooks/output_tables/` is a generated directory and is not in the
repository, so on a fresh checkout that loader returns 36 `NaN`s with 36 zero
weights and the objective is `0.0`. That is by design — see the first bullet
below — but it is not a fit.

`MomentVector` is two blocks: **36 time-series moments** (9 horizons x 2 regimes
x `{log u, log v}`) and **6 cross-section moments** (3 horizons x 2 regimes,
`Q4 − Q1` AIOE-quartile differences), each with inverse-variance weights.
`load_empirical_moments_monthly` builds the same shape from
`paper_monthly_lp.csv` at quarterly-spaced monthly horizons instead.

Three behaviours to know:

- **Missing CSVs are not an error.** The moments become `NaN` with **zero
  weight**, so the objective ignores them and the loader runs before every
  upstream LP has been built. When a fit looks suspiciously good, count the
  non-zero entries of `weights_time_series` before reading the objective.
- **The cross-section maps months onto quarters** — 6m to 2q, 12m to
  4q, 18m to 6q — and generates the quartile spread by scaling the shock itself
  by `aioe_quartile_scaling`, default `(-1.5, +1.5)` standard deviations. The
  AIOE quartile is a shock size in the model, not a separate parameter.
- **`smm_objective` swallows every exception and returns `1e10`** for a
  parameter draw that violates DMP admissibility, so an optimizer walks away
  from the boundary rather than crashing on it. A returned `1e10` means
  infeasible, not merely a bad fit.
