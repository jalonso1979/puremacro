# DSGE sketchpad: models from equations, not matrices

`puremacro.dsge.klein_solve` has always solved linear rational-expectations
models with no Dynare and no compiler — but it takes `A`, `B`, `C`, and getting
those means differentiating the equilibrium conditions by hand. That derivation
is the step people get wrong, and it is exactly the step a tablet is worst at:
no algebra system, no MATLAB, no patience.

`dsge.build` removes it. Write the conditions as they appear in the paper.

```python
import numpy as np
from puremacro import dsge

def eqs(xp, x, e, p):
    # xp = t+1, x = t, e = shocks, p = params
    return [
        1 / x.c - p.beta * (p.alpha * xp.z * xp.k ** (p.alpha - 1)) / xp.c,
        x.c + xp.k - x.z * x.k ** p.alpha,
        xp.z - x.z ** p.rho * np.exp(e.eps),
    ]

m = dsge.build(
    eqs,
    variables=["c", "k", "z"],
    states=["k", "z"],
    shocks=["eps"],
    params=dict(alpha=0.33, beta=0.98, rho=0.9),
    guess=dict(c=0.5, k=0.1, z=1.0),
)
print(m.summary())
m.irf("eps", horizon=20)     # DataFrame: horizons × variables
```

`xp`, `x`, `e` and `p` each support attribute access (`x.k`), string indexing
(`x["k"]`), positional indexing and tuple-unpacking, so a model reads the way
it is written. A misspelled name raises an error listing what was declared.

## What you get back

`LinearModel` is a frozen dataclass carrying the steady state, the units, the
Klein-form matrices and the underlying `KleinSolution`:

| member | meaning |
|---|---|
| `.steady_state` | `pandas.Series`, verified rather than trusted |
| `.units` | per-variable `"log"` or `"level"` |
| `.irf(shock, horizon, size)` | impulse responses, horizons × variables |
| `.simulate(periods, sigma, seed)` | stochastic simulation |
| `.policy()` | decision rules as a labelled table |
| `.decision_rules()` / `.dynare_dr` | `DynareDR` object matching Dynare's `oo_.dr` (`ghx`, `ghu`, `ys`) |
| `.theoretical_moments()` | analytical moments, correlations, autocorrelations & FEVD (Dynare `stoch_simul`) |
| `.fevd(horizons)` | analytical forecast error variance decomposition across finite & asymptotic horizons |
| `.solution` | the `KleinSolution` — `G`, `F`, `N`, `L` |
| `.A`, `.B`, `.C` | the matrices the Jacobians produced |

Steady states are solved from `guess=`, or supplied via `steady_state=` and then
**checked** against the equations (`max|f(ss, ss, 0)| <= tol`) rather than taken
at face value. Log-linearisation is per variable, falling back to level
deviations where a steady state is not strictly positive; `.units` records which
is which.

## Complex-step differentiation, and its one restriction

Jacobians come from `Im f(x + ih) / h` with `h = 1e-20` — machine-precision
derivatives from one evaluation per argument, with no step-size trade-off and no
cancellation error.

The catch is that this is exact only if the residual function is **analytic**,
and it fails *silently* when it is not: `Im` through an `abs()` is identically
zero, so the derivative comes back as zero with nothing raised anywhere. Four
things break it — `abs`, `min`/`max`, a comparison that branches on a perturbed
value, and any `float()` / `np.real()` cast that discards the imaginary part.

`build` therefore cross-checks every complex-step Jacobian against a finite
difference in a random direction and raises `ModelError` naming the offending
block if they disagree. Models with occasionally-binding constraints violate
analyticity by construction: pass `method="central"` and accept ~1e-8 accuracy
instead of ~1e-15. `verify_derivatives=False` opts out of the check.

## Timing

Equations are `E_t f(z_{t+1}, z_t, u_t) = 0`, matching `klein_solve`. An
exogenous process written the usual way, `z' = rho*z + eps`, moves the *state*
into the next period, so every state is zero in the `h=0` row of an IRF and
jumps at `h=1`.

Forward-looking **controls** are a different matter, and this trips people up:
they generally *do* move at `h=0`, because the innovation is in the agents'
information set and they depend on expectations of `t+1`. In the three-equation
New Keynesian model a demand shock leaves the natural rate at zero on impact
while the output gap and inflation jump immediately — agents reacting to the
higher natural rate they already know is coming. That jump is Klein's `L`
loading. Whether a control moves at `h=0` is a property of the model, not of the
convention; only the states are guaranteed to be zero there.

## Blanchard-Kahn as an outcome

Determinacy is not an assumption you assert but a property of the solve. In the
worked New Keynesian example, violating the Taylor principle is a raised
exception rather than a footnote:

```python
solve(phi_pi=0.9)
# BlanchardKahnError: Blanchard-Kahn indeterminacy: 2 unstable generalised
# eigenvalues vs 3 forward-looking variables
```

Pass `strict=False` to get the Sims/gensys-style soft failure (zero matrices
with `eu` flagged) instead.

## Validation

The neoclassical growth model with full depreciation and log utility is the one
textbook case with a closed form — `k' = αβzk^α`, `c = (1−αβ)zk^α`, so
`G = [[α, 1], [0, ρ]]` and `F = [α, 1]` exactly. `build` reproduces every entry
of `G`, `F`, `N` and `L` to 1e-9, and the steady state to 1e-12.

For a model with genuinely forward-looking controls, the New Keynesian block is
checked the strongest way available: the whole IRF path is substituted back into
the structural equations — expectations included, since the economy is
deterministic after impact — and every equation holds to 1e-10.

Both live in `tests/test_dsge/test_build.py`; the worked example is
`python -m puremacro.examples.dsge_nk_sketchpad`.

!!! note "A note on `klein_solve` at 1.2.0"
    Building models from natural-order equations surfaced a long-standing bug in
    `klein_solve`'s policy function. `F` used a formula inconsistent with the
    `G` returned beside it, the guard meant to catch that partitioned `A` and
    `B` by *row* when the split is over *variables*, and `L` was zero whenever a
    shock entered a control equation contemporaneously. All four are fixed and
    pinned against closed-form solutions in
    `tests/test_dsge/test_klein_analytic.py`. Measured impact on existing
    models: SW07 moves by 1.2e-12, and the course notebooks that call
    `klein_solve` reproduce byte-identical output — the old residual guard did
    fire on those, and its fallback recovered a correct `F`.

## Dynare Compatibility: Decision Rules & Theoretical Moments

To match the output of Dynare's `stoch_simul(order=1)`, `LinearModel` provides:

### 1. Decision Rules (`m.decision_rules()` / `m.dynare_dr`)
Maps Klein's state transition and policy functions into Dynare's standard representation:
$$y_t = y^* + g_{x} (x_{t-1} - x^*) + g_{u} u_t$$

```python
dr = m.decision_rules()
print(dr.summary())
```
Outputs the exact Dynare `POLICY AND TRANSITION FUNCTIONS` table with `Constant`, lagged states (`k(-1)`, `z(-1)`), and structural shocks.

### 2. Theoretical Moments (`m.theoretical_moments()`)
Solves the discrete Lyapunov equation $\Sigma_x = G \Sigma_x G' + N \Sigma_u N'$ for analytical unconditional moments without relying on stochastic simulation:

```python
mom = m.theoretical_moments(lags=5)
print(mom.summary())
```
Renders the 4 standard Dynare output blocks:
- **THEORETICAL MOMENTS**: Mean, Standard Deviation, and Variance for each variable.
- **MATRIX OF CORRELATIONS**: Full cross-correlation matrix.
- **COEFFICIENTS OF AUTOCORRELATION**: Exact theoretical autocorrelations for lags 1 through 5.
- **VARIANCE DECOMPOSITION**: Forecast error variance decomposition shares (in percent) across finite horizons (1, 4, 8, 16, 32) and asymptotic infinity.

