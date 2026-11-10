> 🇬🇧 English · 🇪🇸 [Español](es/heterogeneous_expectations.md)

# Heterogeneous information and New Keynesian forecasts

`puremacro.expectations` represents **dated Gaussian information sets**.
Different groups receive different signals, retain past observations, update
beliefs and forecast linear economic outcomes. `models.information_nk` applies
these operators to a finite present-value New Keynesian system.

The information structure is an explicit **processing constraint**: agents use
only the listed signals. Market prices, income and other groups' observations
do not automatically become information. This implementation solves the
conditional-forecast system under that restriction. It does not solve
endogenous market learning, optimal attention, or a fully specified
microeconomic information equilibrium. Its first-order paths do not identify
welfare. These distinctions matter when interpreting monetary-policy results.

## A dated information experiment

The primitive state contains policy news and two independent group noises.
Group A receives a more precise signal than B at date zero. Both observe the
policy news exactly when the rate changes at date 8.

```python
import numpy as np
from puremacro.expectations import (
    GaussianSignal, build_gaussian_information, audit_gaussian_information,
)
from puremacro.models import solve_information_nk, audit_information_nk

signals = (
    (GaussianSignal(0, [1., .5, 0.], "initial"),
     GaussianSignal(8, [1., 0., 0.], "reveal")),
    (GaussianSignal(0, [1., 0., 2.], "initial"),
     GaussianSignal(8, [1., 0., 0.], "reveal")),
)
info = build_gaussian_information(
    np.eye(3), signals, horizon=12, weights=[.5, .5],
    group_names=("A", "B"), state_names=("news", "noise_A", "noise_B"),
)
rate_loadings = np.zeros((12, 3))
rate_loadings[8, 0] = -.0025
solution = solve_information_nk(info, rate_loadings)
assert audit_gaussian_information(info)["accepted"]
assert solution.converged and audit_information_nk(solution)["accepted"]
path = solution.evaluate([1., 0., 0.])
print(path.to_frame().head())
print(path.forecast_frame().head())
print(info.belief_frame([1., 0., 0.]).head())
```

Rates and inflation are per-period deviations. Here a quarterly rate change
of −.0025 means −25 quarterly basis points. Output is a log deviation; multiplying
by 100 is a local percentage approximation, not an exact level percentage.
The realization above sets group noises to zero. In this linear independent-noise
example it also equals the conditional mean response given policy news one.
It is not the unconditional mean of the zero-mean Gaussian experiment.

## What a signal means

Let $z\sim N(0,\Sigma)$ contain **all** primitive shocks, including signal
noise. A `GaussianSignal(date, loading, label)` reveals $h z$ to its group at
that date. The function does not add another noise draw. A separate primitive
component represents an independent new measurement; reusing the same loading
represents the same observation, even if its label or date differs.

Groups share observations internally. A group-specific noise is therefore
common within that group and need not average to zero in one realization.
The covariance can encode correlation across primitives. Its dimension must
match every loading. Group weights are nonnegative and sum to one; zero weights
are permitted. Dates are integer values from zero to `horizon-1`.

Stack each group's released signal rows in $H_{gt}$. The conditional-mean map
and posterior covariance are

$$
P_{gt}=\Sigma H_{gt}'(H_{gt}\Sigma H_{gt}')^+H_{gt},\qquad
E[z\mid\mathcal I_{gt}]=P_{gt}z,\qquad
V_{gt}=\Sigma-P_{gt}\Sigma.
$$

For an outcome $a z$, `info.forecast(a, z, date=t, group=g)` returns
$aP_{gt}z$. `beliefs` returns date × group × primitive conditional means.
`belief_frame` reports means and posterior standard deviations. No observations
give a zero mean and the prior covariance; full-state observation gives
$P=I$ and zero conditional variance.

The prior must be positive definite with condition number at most $10^{12}$.
The builder uses Cholesky whitening, observation-row normalization and SVD.
Exact redundancy is recorded through the rank. Relative singular values in
$[10^{-13},10^{-11}]$ are rejected as ambiguous around the $10^{-12}$ rank
threshold. No jitter is added. Dense projectors and covariance arrays are
limited to two million entries each, and the signal count to 4,096.

The independent audit checks ranks, observed signals, idempotence, covariance
orthogonality, posterior positivity and nested information. Individual
conditional expectations obey the tower property. The average map
$\bar P_t=\sum_gw_gP_{gt}$ generally does **not**: $\bar P_t\bar P_s$ can differ
from $\bar P_t$, even when each individual's information is nested.

## The finite monetary application

Consumer weights come from `info.weights`; optional `firm_weights` can differ.
There is one exogenous nominal-rate path $i_t=R_tz$, zero natural-rate
deviations, and an exact zero continuation for output, inflation and rates
at every date at or beyond $T$. Include the complete policy support in the
input horizon. The model has no Taylor rule or endogenous policy choice.

Defaults are discount factor $\beta=.99$, intertemporal elasticity $s=1$,
Calvo persistence $\theta=.75$, and inverse Frisch elasticity $\epsilon=1$.
The implied Phillips slope is
$\kappa=(1-\theta)(1-\beta\theta)(\epsilon+1/s)/\theta$.
All defaults are illustrative assumptions.

The demand and pricing present values follow the distinction in
[Angeletos and Lian (2018), equations 5–6](https://economics.mit.edu/sites/default/files/publications/Angeletos%20Lian%20(AER%202018).pdf).
The finite boundary and maintained signal sets are explicit restrictions of
this implementation; the paper's endogenous market-signal equilibrium is
not reproduced. With consumer and firm averages $\bar E^c_t$ and $\bar E^f_t$:

$$
y_t=-s\sum_{j=t}^{T-1}\beta^{j-t}\bar E^c_t[i_j-\pi_{j+1}]
 +(1-\beta)\sum_{j=t+1}^{T-1}\beta^{j-t-1}\bar E^c_t[y_j],
$$

$$
\pi_t=\kappa y_t+\sum_{j=t+1}^{T-1}(\beta\theta)^{j-t}
 \left(\kappa\bar E^f_t[y_j]+\frac{1-\theta}{\theta}\bar E^f_t[\pi_j]\right).
$$

The code solves outcome loadings backwards with discounted-sum accumulators.
`audit_information_nk` reconstructs the explicit sums for every primitive
direction, independently of those accumulators. Acceptance requires information
and coefficient errors at most $10^{-9}$ by default. The result retains
coefficient arrays, parameters, information and model fingerprints, plus
diagnostics. `evaluate` returns realized paths and group forecasts of next-date
output, inflation and the nominal rate; final next-date forecasts are zero.
For a longer forecast horizon, apply `info.forecast` to the desired date's
outcome loading. Tables support Markdown, LaTeX and Typst.

Under full information the equations reduce to the textbook recursions
$y_t=y_{t+1}-s(i_t-\pi_{t+1})$ and
$\pi_t=\kappa y_t+\beta\pi_{t+1}$. Tests check them directly. Under dispersed
information, replacing the textbook Euler expectation with an average forecast
generally gives a different model. This is why the implementation keeps the
present values. Individual forecasts are Bayesian under the declared signals;
conditional forecast errors in a particular realization need not be zero.

## Reproduce and interpret the illustration

```bash
python -m puremacro.examples.heterogeneous_expectations --output expectations-example --lang en
```

The example compares full information, no advance news, heterogeneous initial
signals, and a second independent signal at date 4. All groups learn the policy
news at implementation. It exports coefficients, paths, beliefs, forecasts,
audits, complete array files, parameters, source hashes and PNG/PDF figures.
A seeded 50,000-draw experiment checks analytic posterior variances, zero
unconditional forecast errors and orthogonality to observed signals, with
Monte Carlo uncertainty reported and fixed acceptance criteria.

For a date-8 rate cut, initial output responses (log deviation × 100) are
approximately 3.9586 under full information, 0 without advance news, 0.8028
with heterogeneous signals and 1.1838 with the second update. The initial
average policy forecast in the last two cases is half the full-information
forecast. The figure therefore also reports the response ratio divided by the
policy-forecast ratio, separating first-order news attenuation from subsequent
expectation feedback. This ratio is undefined when advance news is absent.

The horizon sweep retains large responses: at date 16 the full-information
linear output response exceeds one log unit. These values are sensitivities
of the linear system, not credible quantitative forecasts at that shock size.
The CSV flags absolute output/inflation deviations above .05; that flag is
descriptive, not a bound on approximation error. Zero-tail padding preserves
earlier coefficients exactly up to numerical precision, but does not establish
infinite-horizon determinacy. No empirical calibration, policy recommendation,
welfare, or HANK integration is claimed by this first implementation.
