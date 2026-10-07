> 🇬🇧 English · 🇪🇸 [Español](es/structural_bridge.md)

# From empirical moments to structural parameters

`puremacro.structural` fits a callable structural model to empirical moment
estimates, keeping their full covariance, labels, units and research metadata.
It connects existing empirical estimates to a model; it does not supply a new
economic model or identify one merely by optimizing it.

The covariance contract is **Cov(empirical moment estimators)**. For example,
the diagonal contains the squared standard errors of estimated impulse
responses, and the off-diagonals contain covariance between those estimates.
Do not pass the covariance of raw observations or a covariance multiplied by
sample size. `fit_structural` never applies a sample-size multiplier.

## A complete small example

```python
import numpy as np
from puremacro.structural import MomentTargets, fit_structural

h = np.arange(4)
empirical_irf = np.array([0.80, 0.61, 0.47, 0.34])
covariance = 0.01 * 0.5 ** np.abs(h[:, None] - h[None, :])
targets = MomentTargets.from_irf(
    empirical_irf,
    covariance=covariance,
    response="output",
    shock="monetary",
    units="percent per 25bp tightening",
    metadata={"sample": "illustrative", "shock_size": 25, "shock_unit": "bp"},
)

# This reduced example has a closed-form model response. A DSGE, VFI or
# HANK model wrapper can return its selected moments through the same API.
def moments_at(theta):
    impact, persistence = theta
    return impact * persistence ** h

result = fit_structural(
    moments_at, targets, [0.7, 0.7],
    parameter_names=("impact", "persistence"),
    bounds=[(0, None), (0, 0.99)],
    moment_labels=targets.labels,
)
print(result.summary())
print(result.moment_fit())
print(result.to_latex())
```

This is synthetic input, not an empirical monetary-policy estimate. Model
responses must use the same variables, horizon frequency, cumulative/level
convention, transformation and shock normalization as the targets. The adapter
records these choices; it does not infer or reconcile them.

## Constructing and combining targets

| Entry point | Contract |
|---|---|
| `MomentTargets(values, covariance, labels, units=..., metadata=...)` | One-dimensional values, full square covariance, unique string labels. |
| `MomentTargets.from_frame(frame, covariance=...)` | Columns `label`, `estimate`, `unit`; index labels and unspecified units are fallback choices. Column names can be changed with `label=`, `value=`, `unit=`. |
| `MomentTargets.from_irf(irf, covariance=..., response=..., shock=..., units=...)` | One selected response to one shock: a vector or an LP table with `h` and `beta`. Select a VAR tensor slice explicitly first. |
| `MomentTargets.from_draws(draws, values=..., labels=...)` | Rows are joint sampling/bootstrap replicates, columns are moments. Sample covariance uses `ddof=1` without division by the number of replicates. |
| `targets.select(labels)` | Subsets/reorders both the moment vector and covariance axes. |
| `targets.rescale(factors, units=...)` | Applies `D m` and `D S D'` together. Metadata records the previous units and scaling. |
| `targets.stack(other, cross_covariance=...)` | Explicit cross block in the two label orders. Alternatively use `assume_independent=True`, which records that assumption. |

Pointwise IRF bands and standard errors do not determine a joint covariance.
For bootstrap targets, reestimate **all** moments on each valid joint
replicate, retaining the dependence across horizons and responses. Posterior
draws and independently resampled horizons are not automatically valid inputs
to `from_draws`. The supplied `values` remain the original point estimates.
There is no automatic removal of failed/nonfinite replicates.

Labeled pandas inputs must have exactly the declared order. Plain arrays are
positional: the caller guarantees their order. A model returning a pandas
Series is checked against target labels on every evaluation. An optional
`moment_labels=` declaration catches a mismatched model order before fitting.

Target arrays are copied and made read-only. Mapping/list/array metadata is
copied and frozen; `from_frame` and `from_irf` preserve table attributes.
Stacking preserves both component metadata records, including separate samples
and shock normalizations. A stack does not make incompatible units compatible.

## Joint local-projection covariance

`lp_moment_targets` estimates the empirical responses and their joint
covariance directly from a regularly spaced time-series table:

```python
from puremacro.structural import lp_moment_targets

import pandas as pd
# Synthetic, already-transformed data for a runnable API illustration.
rng = np.random.default_rng(19)
data = pd.DataFrame(rng.normal(size=(160, 3)),
                    columns=["output", "inflation", "monetary_innovation"],
                    index=pd.period_range("1980Q1", periods=160, freq="Q"))
empirical = lp_moment_targets(
    data,
    responses=("output", "inflation"),
    shock="monetary_innovation",
    horizons=(0, 1, 2, 3),
    response_units={"output": "log points", "inflation": "percentage points"},
    shock_unit="percentage points",
    frequency="Q",
    lags=2,
    bandwidth=4,
    shock_size=0.25,
)
```

Like `lp_hac`, the dependent variable is **`y[t+h] - y[t-1]`**. The design
includes the contemporaneous shock, response/shock lags, and any declared
controls both contemporaneously and at each lag. A structural response must
match that convention. Moment order is response first, then the supplied
horizon order; labels are `response:shock:h=H`.

Coefficient influence contributions from all regressions are aligned on the
original time grid, with zeros outside their estimation windows. Applying a
single Bartlett HAC kernel to that matrix retains cross-horizon and
cross-response covariance. The default bandwidth is `max(horizons)+1`, an
explicit heuristic; no small-sample correction is applied. This estimator
requires an exogenous shock conditional on the controls and regular
time-series HAC conditions. It is neither LP-IV nor inference robust to weak
identification. The covariance may be singular when too many moments are
included; the subsequent fit will refuse it.

Inputs must be finite and their index unique/increasing. Datetime/Period
indices are checked against `frequency`; other indices rely on the caller's
explicit frequency declaration. The function does not drop missing periods
or infer percentage conversions from variable names. Metadata retains the
sample, frequency, per-moment observation counts, transformation, bandwidth,
units and shock magnitude.

## Criterion and local covariance

Let `S` denote the supplied covariance, `r(theta) = m(theta) - m_hat`, and
`G = dm/dtheta`. The implementation minimizes

\[
Q(\theta)=r(\theta)' W r(\theta), \qquad W=S^{-1}\text{ by default}.
\]

Writing `S = D L L' D`, with `D` containing target standard errors and `L` the
Cholesky factor of the correlation matrix, the weighted residual is
`L^{-1} D^{-1} r`. Standardizing before factorization makes the covariance
checks insensitive to different moment units. The reported objective is the
full quadratic criterion using the original weights.

The optimizer uses coordinates centered at `theta0`, scaled by the initial
weighted Jacobian, and adaptive Jacobian trust-region scaling. It also divides
all residuals by one common sampling-uncertainty scale. This preserves relative
weights and the minimizer while preventing parameter units, large offsets or
arbitrary scalar multiples of custom weights from causing premature numerical
convergence. These internal normalizations do not change the reported
objective or parameter covariance.

For a fixed supplied positive-definite `weight=W`, the covariance is

\[
A=(G'WG)^{-1}G'W,\qquad
\widehat{\operatorname{Cov}}(\hat\theta)=A S A'.
\]

With optimal weighting this reduces to `(G' S^{-1} G)^{-1}`. The implementation
uses the SVD of the weighted Jacobian and covariance factors, avoiding an
explicit inverse of its normal equations. It never estimates a residual
variance or rescales the supplied covariance by fit quality.

These are local, correctly specified minimum-distance formulas; regular
asymptotic inference requires a consistent moment covariance and full-rank
interior identification. See Newey and McFadden (1994), Sections 3.1, 4.1 and
5.2, for the minimum-distance framework, and Section 9.5 for the
overidentification result. [Original chapter](https://users.ssc.wisc.edu/~xshi/Newey_Mcfadden_handbook.pdf).

`jac(theta)` may supply the **unweighted** moments-by-parameters Jacobian.
Otherwise the bridge uses second-order finite differences, with one-sided
stencils near parameter bounds. This does not implement missing derivatives
through heterogeneous-agent distributions or general equilibrium. Stochastic,
nonsmooth, or grid-discontinuous model moments need a different inference
design; fixing a simulation seed alone does not eliminate simulation error.
Optimization uses a local trust-region reflective solve and does not establish
a global optimum. [SciPy least_squares reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).

## Acceptance and identification diagnostics

`result.success` is numerical optimizer success. `result.inference_valid`
means the local numerical checks permit the stated conditional formulas.
Neither asserts that the economic model is correct.

Normal standard errors and intervals are returned as NaN, with a warning and
named diagnostic reasons, if the optimizer fails, the weighted Jacobian lacks
column rank, its condition number exceeds `condition_limit`, any parameter
is at/near a bound, or an interior solution fails a numerical first-order
condition (`nonstationary_solution`). The latter checks the residual projection
onto the local weighted Jacobian's column space; overidentified models may
retain residuals orthogonal to that space. The result retains estimates, moment residuals and
optimizer information for inspection. Rank uses
`s_i > rank_rtol * s_max` (default `rank_rtol=1e-10`); the default condition
limit is `1e8`. Conditioning depends on the supplied parameter units. This
is a numerical diagnostic, **not a statistical test of weak identification**;
full local rank also does not establish global identification.

Covariances and weights must be finite, symmetric and positive semidefinite
as data; fitting additionally requires numerically positive-definite
factorizations. Singular covariances are refused rather than inverted with a
pseudoinverse. No ridge, eigenvalue clipping or automatic moment deletion is
performed. Redundant targets can be explicitly removed with `select`, with
the economic reason for the selection recorded by the researcher.

Model nonfinite outputs, incompatible dimensions and invalid derivatives raise
errors. Failed economic solves should likewise raise in `moments_at`; the
bridge does not convert an invalid solve into a moment vector or a penalty.

## Overidentification and held-out moments

For `q > p`, the minimized criterion has the usual asymptotic chi-square
reference with `q-p` degrees of freedom only under the relevant regular
correct-specification conditions and optimal weighting. To request it, set
`assume_correct_specification=True`. This records an assumption, not a test
that the assumption holds. `j_statistic` and `j_pvalue` stay NaN for custom
weights, exact/underidentification, or failed numerical inference checks.
The criterion is always available as `objective`; `j_df` is `max(q-p, 0)`.

```python
unused = MomentTargets([0.2], [[0.02]], ("output_later",), targets.units[0])
checked = fit_structural(
    moments_at, targets, [0.7, 0.7],
    bounds=[(0, None), (0, 0.99)],
    held_out=unused,
    held_out_moments_at=lambda theta: [theta[0] * theta[1] ** 6],
)
print(checked.moment_fit(held_out=True))
```

Held-out labels must be distinct from fitted labels. Their comparison never
affects the objective. It reports targets, target standard errors, model
predictions and residuals, with no p-value or predictive interval: uncertainty
in fitted parameters and cross-covariance with held-out estimates have not
been supplied. Holding out moments from the same sample does not imply
statistical independence.

## Evidence and limits

`tests/test_structural_bridge.py` checks a closed-form correlated GLS solution
and covariance, arbitrary-weight sandwich covariance, the correctly scaled J
statistic, exact nonlinear impulse-response recovery, mixed-unit invariance,
adapters and named failure cases. These are analytical and software checks.
They do not constitute an empirical model replication or establish nominal
coverage for every estimator producing targets. HA distribution derivatives,
weak-identification-robust confidence sets, simulation-noise corrections,
global search and inference under misspecification remain outside this bridge.
