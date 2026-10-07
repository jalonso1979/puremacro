# Exact finite-sample expectations without fourth-moment work

The implementation is `puremacro/structural/sw07_expectations.py`. It evaluates
the expectation part of the already audited `sw07_finite_sample_moments` oracle
without building dense time-covariance blocks or the covariance matrix of the
sample moments. [Validation and local timing](helper_validation.md) are recorded
separately.

## Estimand and stationary covariance

Let a centered stationary state satisfy

\[
x_t = T x_{t-1} + R\varepsilon_t,
\qquad y_t = Zx_t,
\qquad S = TST' + RQR'.
\]

For nonnegative lags define

\[
\Gamma_h = \operatorname{Cov}(y_t,y_{t-h}) = ZT^hSZ',
\qquad \Gamma_{-h}=\Gamma_h'.
\]

Observation intercepts cancel from central moments. The numerical Kalman
measurement-error ridge is excluded: economic measurement error is zero.
The helper reuses the sampling module's parameter, Blanchard–Kahn,
stationarity, Lyapunov-residual and positive-semidefiniteness checks.

With `N = nobs` and `L = common_max_lag`, the requested moment `(a,b,h)` is

\[
\widehat m_{ab,h} = \frac{1}{N-L}\sum_{t=L}^{N-1}
  (y_{a,t}-\bar y_a)(y_{b,t-h}-\bar y_b),
\qquad
\bar y=\frac1N\sum_{t=0}^{N-1}y_t.
\]

The means use **all N dates**, even when the product window starts at `L > 0`.
Every moment in the study uses `N=156` and `L=4`, giving 152 product rows.
Selecting only the nine fit moments must therefore retain `common_max_lag=4`.

## Prefix-sum derivation

Let `A_t` denote the cumulative positive-lag covariance through date `t`, with
`A_0=0`:

\[
A_t=\sum_{h=1}^{t}\Gamma_h.
\]

The covariance of the observation at date `t` with the full-sample mean is

\[
F_t=\operatorname{Cov}(y_t,\bar y)
  =\frac{\Gamma_0+A_t+A_{N-1-t}'}{N}.
\]

The transpose on the final term is necessary: observations later than `t`
enter through negative-lag covariances. This matters for nonsymmetric VAR
cross-lags. Averaging `F_t` over all dates gives the sample-mean covariance:

\[
V_{\bar y}=\frac1N\sum_{t=0}^{N-1}F_t
 =\frac{N\Gamma_0+\sum_{h=1}^{N-1}(N-h)(\Gamma_h+\Gamma_h')}{N^2}.
\]

Expanding the demeaned product then gives

\[
E[\widehat m_{ab,h}]
 = (\Gamma_h)_{ab}
 -\frac{1}{N-L}\sum_{t=L}^{N-1}
   \left[(F_t)_{ab}+(F_{t-h})_{ba}\right]
 +(V_{\bar y})_{ab}.
\]

This is the implemented expression. It is exact up to numerical arithmetic;
no simulation, interpolation, burn-in or asymptotic centering approximation is
used. Positive-lag matrices are formed by repeatedly propagating `Z` through
`T`; cumulative sums provide all `F_t`. Storage for these observation-level
arrays is `O(N q²)` for `q` selected observables, rather than the full oracle's
`O(N² q²)` time-covariance blocks. State-system solution costs remain shared
with the full oracle.

For `demean=False`, the known model population means are removed, so the
answer is simply `(Γ_h)ab`; only the requested lag horizon needs propagation.
This is a diagnostic control, not raw products including observation
intercepts. In the IID scalar case with variance `v`, the formula gives
`v(1−1/N)` at lag zero and `−v/N` at every positive lag, including a truncated
common product window.

## Role in the paired estimator experiment

The finite-expectation variants evaluate this formula at each candidate
`(crr, em)` while holding the remaining calibration fixed. They use exactly
the observed moment definitions and window. The population variants retain
their original stationary covariance targets. Each sample supplies the same
moment vector and HAC estimate to the four variants.

The oracle-weight variants use the **full exact covariance evaluated once at
the known generating calibration**, held fixed during optimization. This is
a separate control from correcting the expectations; it is not a candidate-
dependent weight matrix and not a feasible empirical covariance estimator.

The expectation derivation requires stationary second moments, not Gaussian
fourth moments. The full exact covariance used for the weight control does
rely on the experiment's Gaussian law. Neither exact expectations nor exact
first two moments establish chi-square specification inference, normal
parameter intervals, or inference robust to an uncertain calibration.
