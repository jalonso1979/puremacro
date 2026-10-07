"""Markov-switching VAR with regime-dependent intercept and covariance.

Model
-----
``ms_var_fit`` estimates an MSIH(K)-VAR(p) in Krolzig's (1997) notation
(Markov-switching Intercept and Heteroskedasticity) with one
autoregressive matrix shared by all regimes::

    y_t = mu_{s_t} + A_1 y_{t-1} + ... + A_p y_{t-p} + e_t,
    e_t | s_t = k  ~  N(0, Sigma_k),
    P[i, j] = Pr(s_t = j | s_{t-1} = i).

Only the intercepts ``mu_k`` and the covariances ``Sigma_k`` switch;
``A = [A_1, ..., A_p]`` does not. The likelihood is conditional on the
first ``p`` observations, and the distribution of the first modelled
regime, ``pi0 = Pr(s_{p+1})``, is estimated as a free parameter. With
``p = 0`` the intercept is the mean, and the model is a Gaussian hidden
Markov model with switching mean and covariance.

Estimation: EM in its ECM form
------------------------------
E-step: Hamilton filter, computed in log-scaled form, plus the Kim
smoother. They give the smoothed regime probabilities
``xi_{t|T}`` and the pairwise probabilities
``Pr(s_t = i, s_{t+1} = j | Y_T)``.

M-step: three conditional maximisations (CM steps) of the EM auxiliary
function ``Q``. Each one maximises ``Q`` over one block given the
others, so the log-likelihood never falls across iterations (the ECM
property, Meng and Rubin 1993):

1. ``B = [mu_1, ..., mu_K, A]`` jointly, by GLS. The weights are the
   smoothed probabilities times ``Sigma_k^{-1}`` of the current iterate::

       vec(B) = [sum_k (Z_k' W_k Z_k) kron Sigma_k^{-1}]^{-1}
                vec(sum_k Sigma_k^{-1} Y' W_k Z_k),

   with ``Z_k = [1_T e_k', X]`` (T x (K + n p): ones in intercept
   column ``k``, zeros in the other intercept columns, then the lags)
   and ``W_k = diag(xi_{t|T}(k))``.
   When all ``Sigma_k`` are equal this reduces to equation-by-equation
   weighted least squares. When they differ, an unweighted update of
   ``A`` does not maximise ``Q``. Up to release 4.3.0 this module used
   that unweighted update, and its log-likelihood could fall.
2. ``Sigma_k = sum_t xi_{t|T}(k) r_tk r_tk' / sum_t xi_{t|T}(k)``,
   subject to the variance floor below. Clipping the eigenvalues at the
   floor is the exact maximiser of ``Q`` under that constraint.
3. ``P[i, j] = sum_t Pr(s_t=i, s_{t+1}=j | Y_T) / sum_t xi_{t|T}(i)`` and
   ``pi0 = xi_{p+1|T}``.

The fit declares convergence only when the log-likelihood increase is
below ``tol``. A fall larger than ``1e-8 * max(1, |loglik|)`` (rounding
error) raises a ``RuntimeWarning``. ``MSVARResult.loglik_path`` records
the log-likelihood of every iterate.

Units and the variance floor
----------------------------
Let ``d_j`` be the pooled residual variance of variable ``j``: the mean
squared residual of a no-intercept OLS VAR(p) (of the demeaned data when
``p = 0``), and ``D = diag(d_1, ..., d_n)``. The fit runs internally on
``D^{-1/2} y_t`` and maps the estimates back, so it is equivariant to the
units of each variable. Rescaling variable ``j`` by ``c_j``
(``y_t -> C y_t`` with ``C = diag(c)``) maps ``mu_k -> C mu_k``,
``A_i -> C A_i C^{-1}`` and ``Sigma_k -> C Sigma_k C``, leaves the regime
probabilities unchanged, and shifts the log-likelihood by
``-(T - p) * sum_j log|c_j|``. The floor is

    D^{-1/2} Sigma_k D^{-1/2}  >=  1e-8 * I   (in the matrix order),

i.e. every regime variance is at least ``1e-8`` times the pooled
residual variance in every direction of the standardised data. Up to
release 4.3.0 an absolute ridge ``1e-8 * I`` was added to every
``Sigma_k`` instead, which depends on the units: it distorted the fit of
any variable whose regime variances are not large relative to ``1e-8``
(e.g. rates in decimals).

The MSIH likelihood is unbounded: a regime whose ``Sigma_k`` collapses
onto a few observations drives it to infinity, and only the floor keeps
it finite. From the data-driven start, EM normally converges to an
interior local maximum. When a returned ``Sigma_k`` has an eigenvalue
within a factor 10 of the floor, ``ms_var_fit`` raises a
``RuntimeWarning`` naming the regime and sets
``MSVARResult.sigma_at_floor[k]``: that regime has collapsed and its
estimates are not an interior maximum.

Relation to Hamilton (1989)
---------------------------
This is not the model of Hamilton (1989). Hamilton's eq. (4.3) (p. 367)
switches the *mean* of an autoregression,
``y_t = alpha_1 s_t + alpha_0 + z_t``, where ``z_t`` is an AR(r)
(``r = 4`` in the application) with one constant innovation variance
``sigma^2``. That is an MSM(2)-AR(4) in Krolzig's notation. Its filter
runs over the joint state ``(s_t, ..., s_{t-4})``, and Hamilton
maximised the likelihood numerically (Davidon-Fletcher-Powell, p. 372,
fn. 7), not by EM.
``ms_var_fit`` therefore does not reproduce Hamilton's Table I; for that
use statsmodels ``MarkovAutoregression(k_regimes=2, order=4,
switching_ar=False)``. For a univariate series, the model estimated
here is statsmodels ``MarkovRegression(trend='c', exog=<p lags>,
switching_exog=False, switching_variance=True)``. The one difference is
that statsmodels starts the filter from the ergodic regime
distribution, while ``ms_var_fit`` estimates ``pi0``.

References
----------
Hamilton, J.D. (1989). A new approach to the economic analysis of
    nonstationary time series and the business cycle. Econometrica
    57(2), 357-384.
Hamilton, J.D. (1990). Analysis of time series subject to changes in
    regime. Journal of Econometrics 45(1-2), 39-70.
Kim, C.-J. (1994). Dynamic linear models with Markov-switching. Journal
    of Econometrics 60(1-2), 1-22.
Krolzig, H.-M. (1997). Markov-Switching Vector Autoregressions.
    Springer Lecture Notes 454.
Meng, X.-L. and Rubin, D.B. (1993). Maximum likelihood estimation via
    the ECM algorithm: a general framework. Biometrika 80(2), 267-278.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..._linalg import safe_cholesky

# Falls in the log-likelihood smaller than this multiple of max(1, |ll|)
# are attributed to rounding and do not count as decreases.
_DECREASE_RTOL = 1e-8
# A regime whose total smoothed weight is below this is treated as empty:
# its mu_k and Sigma_k are kept, and it adds no mu_k column to the GLS
# system, which would otherwise be singular.
_EMPTY_REGIME_WEIGHT = 1e-8
# Eigenvalue floor for Sigma_k in the standardised units D^{-1/2} y_t,
# where D holds the per-variable pooled OLS residual variances: the
# constraint is D^{-1/2} Sigma_k D^{-1/2} >= _VAR_FLOOR_REL * I. It keeps
# the Gaussian densities finite and is invariant to the units of y.
_VAR_FLOOR_REL = 1e-8
# A returned Sigma_k whose smallest standardised eigenvalue is below this
# multiple of the floor is reported as collapsed (sigma_at_floor, warning).
_FLOOR_WARN_FACTOR = 10.0
# A variable whose pooled residual variance is below this fraction of its
# mean square is (numerically) constant or an exact function of the lags;
# the Gaussian likelihood is then degenerate and the fit is refused.
_DEGENERATE_RESID_REL = 1e-20
_LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass
class MSVARResult:
    """Output of ``ms_var_fit`` (an MSIH(K)-VAR(p) with a shared ``A``).

    Attributes
    ----------
    K : int, the number of regimes.
    A : (n, n*p) ndarray, the shared VAR coefficient matrix
        ``[A_1, ..., A_p]``. It multiplies ``[y_{t-1}; ...; y_{t-p}]``.
    mu : (K, n) ndarray, the regime-specific intercepts, in the units of
        ``Y``. When ``p > 0`` these are intercepts, not means. If regime
        ``k`` lasted forever, the process would settle at
        ``(I - A_1 - ... - A_p)^{-1} mu_k``.
    Sigma : (K, n, n) ndarray, the regime-specific innovation
        covariances.
    P : (K, K) ndarray, the transition matrix,
        ``P[i, j] = Pr(s_t = j | s_{t-1} = i)`` (rows sum to one).
    smoothed_probs : (T - p, K) ndarray, the Kim-smoother probabilities
        ``Pr(s_t = k | Y_T)`` for ``t = p+1, ..., T``.
    filtered_probs : (T - p, K) ndarray, the Hamilton-filter
        probabilities ``Pr(s_t = k | Y_t)``.
    loglik : float, the Gaussian log-likelihood of ``Y`` (in its own
        units) at the returned parameters, conditional on the first ``p``
        observations. Rescaling variable ``j`` by ``c_j`` shifts it by
        ``-(T - p) * log|c_j|``.
    n_iter : int, the number of EM iterations (each is one M-step plus
        one E-step).
    converged : bool. True only if the last log-likelihood change was
        in ``[-rounding, tol)``.
    pi0 : (K,) ndarray, the estimated distribution of the first
        modelled regime, ``Pr(s_{p+1})``. It is the filter's initial
        prediction.
    loglik_path : (n_iter,) ndarray, the log-likelihood of every EM
        iterate. It is non-decreasing up to rounding, and
        ``loglik_path[-1] == loglik``.
    sigma_at_floor : (K,) bool ndarray. True for a regime whose
        ``Sigma_k`` has an eigenvalue within a factor 10 of the variance
        floor ``D^{-1/2} Sigma_k D^{-1/2} >= 1e-8 I`` (module docstring):
        the regime collapsed onto a few observations, and its estimates
        are not an interior maximum. ``ms_var_fit`` also warns.
    """
    K: int
    A: np.ndarray
    mu: np.ndarray
    Sigma: np.ndarray
    P: np.ndarray
    smoothed_probs: np.ndarray
    filtered_probs: np.ndarray
    loglik: float
    n_iter: int
    converged: bool
    pi0: Optional[np.ndarray] = None
    loglik_path: Optional[np.ndarray] = None
    sigma_at_floor: Optional[np.ndarray] = None


def _gauss_density(y: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """Gaussian density N(y; mean, cov). Legacy helper; ``ms_var_fit``
    works with log densities (``_log_densities``)."""
    n = len(y)
    diff = y - mean
    try:
        L = safe_cholesky(cov + 1e-12 * np.eye(n), name="ms_var density")
    except np.linalg.LinAlgError:
        # Per the ms_var Hamilton-filter contract, return a tiny but
        # non-zero density rather than propagating the failure.
        return 1e-12
    z = np.linalg.solve(L, diff)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    return float(np.exp(-0.5 * (n * np.log(2 * np.pi) + log_det + z @ z)))


def _log_densities(Y_dep: np.ndarray, X_lag: np.ndarray, A: np.ndarray,
                   mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """log N(y_t; mu_k + A x_t, Sigma_k) for every t and k, shape (T, K)."""
    T, n = Y_dep.shape
    K = mu.shape[0]
    base = Y_dep - X_lag @ A.T
    out = np.empty((T, K))
    tiny = np.finfo(float).tiny
    for k in range(K):
        lam, V = np.linalg.eigh(0.5 * (Sigma[k] + Sigma[k].T))
        lam = np.maximum(lam, tiny)
        z = ((base - mu[k]) @ V) / np.sqrt(lam)
        out[:, k] = -0.5 * (n * _LOG_2PI + np.log(lam).sum()
                            + np.einsum("ti,ti->t", z, z))
    return out


def _hamilton_filter_log(log_dens: np.ndarray, P: np.ndarray,
                         pi0: np.ndarray):
    """Hamilton filter on log densities (T, K), scaled per period so that
    no density underflows. Returns (filtered, predicted, loglik), where
    predicted[t] = Pr(s_t | Y_{t-1}) and predicted[0] = pi0."""
    T, K = log_dens.shape
    filt = np.empty((T, K))
    pred = np.empty((T, K))
    ll = 0.0
    pred_t = np.asarray(pi0, dtype=float)
    for t in range(T):
        with np.errstate(divide="ignore"):
            a = np.log(pred_t) + log_dens[t]
        c = a.max()
        if not np.isfinite(c):
            # No regime has both positive predicted probability and a
            # finite density. Mirror the legacy fallback: uniform filtered
            # probabilities and a density of 1e-12 per regime.
            filt[t] = 1.0 / K
            ll += np.log(K * 1e-12)
        else:
            joint = np.exp(a - c)
            s = joint.sum()
            ll += c + np.log(s)
            filt[t] = joint / s
        pred[t] = pred_t
        pred_t = filt[t] @ P
    return filt, pred, float(ll)


def _hamilton_filter(densities: np.ndarray, P: np.ndarray, pi0: np.ndarray):
    """Forward filter on densities (T, K). Returns (filtered, predicted, ll).

    Thin wrapper over ``_hamilton_filter_log``. Zero densities are
    allowed; a period in which every regime has zero density falls back
    to uniform filtered probabilities."""
    with np.errstate(divide="ignore"):
        log_dens = np.log(np.maximum(np.asarray(densities, dtype=float), 0.0))
    return _hamilton_filter_log(log_dens, P, pi0)


def _kim_smoother(filt: np.ndarray, P: np.ndarray):
    """Kim (1994) backward recursion. Returns (smoothed, pairwise), where
    pairwise[t, i, j] = Pr(s_t = i, s_{t+1} = j | Y_T)."""
    T, K = filt.shape
    sm = np.empty((T, K))
    sm[-1] = filt[-1]
    pairwise = np.zeros((max(T - 1, 0), K, K))
    for t in range(T - 2, -1, -1):
        pred_next = filt[t] @ P
        ratio = np.divide(sm[t + 1], pred_next, out=np.zeros(K),
                          where=pred_next > 0)
        pairwise[t] = filt[t][:, None] * P * ratio[None, :]
        sm[t] = pairwise[t].sum(axis=1)
    return sm, pairwise


def _lag_matrix(Y: np.ndarray, p: int) -> np.ndarray:
    """[y_{t-1}', ..., y_{t-p}'] for t = p+1..T, shape (T - p, n*p)."""
    T = Y.shape[0]
    if p == 0:
        return np.empty((T, 0))
    return np.column_stack([Y[p - k - 1: T - k - 1] for k in range(p)])


def _floor_cov(S: np.ndarray, floor: float) -> np.ndarray:
    """Symmetrise S and lift its eigenvalues to at least ``floor``.

    This is the exact maximiser of the Gaussian Q-term
    ``-(w/2) [log det Sigma + tr(Sigma^{-1} S)]`` under
    ``Sigma >= floor * I``. ``ms_var_fit`` calls it on standardised data
    (each variable divided by its pooled residual standard deviation), so
    the constraint is ``D^{-1/2} Sigma D^{-1/2} >= floor * I`` in the
    original units."""
    S = 0.5 * (S + S.T)
    lam, V = np.linalg.eigh(S)
    if lam.min() >= floor:
        return S
    lam = np.maximum(lam, floor)
    return (V * lam) @ V.T


def _m_step(Y_dep: np.ndarray, X_lag: np.ndarray, smoothed: np.ndarray,
            pairwise: Optional[np.ndarray], current: dict,
            var_floor: float,
            sigma_default: Optional[np.ndarray] = None) -> dict:
    """One ECM cycle: (mu, A) by GLS given the current Sigma_k, then
    Sigma_k, then (P, pi0). Returns a new parameter dict.

    ``current["Sigma"] is None`` marks the first pass, which starts from
    the initial clustering. The GLS weights are then the identity, which
    gives equation-by-equation weighted least squares.
    ``sigma_default`` (n x n, default the identity) is the covariance
    given to a regime that is empty on the first pass; ``ms_var_fit``
    passes the pooled residual covariance of the standardised data.

    If the GLS system is not finite (overflowing weights), CM-1 keeps the
    current ``(mu, A)``; that still does not lower ``Q``.
    """
    T_eff, n = Y_dep.shape
    K = smoothed.shape[1]
    m = X_lag.shape[1]
    w_sum = smoothed.sum(axis=0)
    active = w_sum > _EMPTY_REGIME_WEIGHT
    act = np.flatnonzero(active)
    col = {k: j for j, k in enumerate(act)}
    q = len(act) + m
    Sigma_cur = current["Sigma"]
    mu = np.array(current["mu"], dtype=float, copy=True)
    A = np.array(current["A"], dtype=float, copy=True)

    # --- CM 1: B = [mu_active, A] by GLS given Sigma_cur ------------------
    if q > 0:
        H = np.zeros((n * q, n * q))
        G = np.zeros((n, q))
        for k in range(K):
            w = smoothed[:, k]
            Sinv = (np.eye(n) if Sigma_cur is None
                    else np.linalg.inv(Sigma_cur[k]))
            Sinv = 0.5 * (Sinv + Sinv.T)
            Z = np.zeros((T_eff, q))
            Z[:, len(act):] = X_lag
            target = Y_dep
            if k in col:
                Z[:, col[k]] = 1.0
            else:
                target = Y_dep - mu[k]        # inactive: mu_k held fixed
            Zw = Z * w[:, None]
            H += np.kron(Zw.T @ Z, Sinv)
            G += Sinv @ (target.T @ Zw)
        g = G.reshape(-1, order="F")
        b = None
        if np.all(np.isfinite(H)) and np.all(np.isfinite(g)):
            # Symmetric diagonal scaling: Sigma_k^{-1} and the regime
            # weights can differ by many orders of magnitude.
            dh = np.diag(H)
            d = np.ones_like(dh)
            pos = dh > 0
            d[pos] = 1.0 / np.sqrt(dh[pos])
            Hs = (H * d[:, None]) * d[None, :]
            try:
                b = d * np.linalg.solve(Hs, d * g)
                if not np.all(np.isfinite(b)):
                    raise np.linalg.LinAlgError("non-finite GLS solution")
            except np.linalg.LinAlgError:
                try:
                    b = d * np.linalg.lstsq(Hs, d * g, rcond=None)[0]
                except np.linalg.LinAlgError:
                    b = None
            if b is not None and not np.all(np.isfinite(b)):
                b = None
        if b is not None:
            B = b.reshape(n, q, order="F")
            for k, j in col.items():
                mu[k] = B[:, j]
            A = B[:, len(act):]

    # --- CM 2: Sigma_k given (mu, A) --------------------------------------
    fitted = X_lag @ A.T
    Sigma = np.empty((K, n, n))
    for k in range(K):
        if not active[k]:
            if Sigma_cur is not None:
                Sigma[k] = Sigma_cur[k]
            else:
                Sigma[k] = _floor_cov(np.eye(n) if sigma_default is None
                                      else sigma_default, var_floor)
            continue
        w = smoothed[:, k]
        r = Y_dep - mu[k] - fitted
        Sigma[k] = _floor_cov((r * w[:, None]).T @ r / w_sum[k], var_floor)

    # --- CM 3: transition matrix and initial distribution ------------------
    P = np.array(current["P"], dtype=float, copy=True)
    pi0 = np.array(current["pi0"], dtype=float, copy=True)
    if pairwise is not None:
        if pairwise.shape[0] > 0:
            num = pairwise.sum(axis=0)
            rows = num.sum(axis=1)
            ok = rows > 0
            P[ok] = num[ok] / rows[ok, None]
        pi0 = smoothed[0] / smoothed[0].sum()
    return {"A": A, "mu": mu, "Sigma": Sigma, "P": P, "pi0": pi0}


def _em_step_status(ll_prev: float, ll: float, tol: float):
    """Classify one EM step. Returns (converged, decreased).

    A fall larger than ``_DECREASE_RTOL * max(1, |ll_prev|)`` is a
    decrease. It must not happen in exact arithmetic, and it never counts
    as convergence. Otherwise the step has converged when
    ``|ll - ll_prev| < tol``.
    """
    delta = ll - ll_prev
    decreased = bool(delta < -_DECREASE_RTOL * max(1.0, abs(ll_prev)))
    converged = (not decreased) and bool(abs(delta) < tol)
    return converged, decreased


def _pooled_residual_scale(Y: np.ndarray, p: int):
    """Per-variable pooled residual standard deviations and residuals.

    Returns ``(s, r)``: ``s`` (n,) holds ``sqrt(d_j)``, the root mean
    squared residual of variable ``j`` in a no-intercept OLS VAR(p) (in
    the demeaned data when ``p = 0``), and ``r`` (T - p, n) holds those
    residuals divided by ``s``. The regression runs on each column divided
    by its root mean square, so the result is equivariant to the units of
    each variable: rescaling column ``j`` by ``c_j`` multiplies ``s_j`` by
    ``|c_j|`` and leaves ``|r|`` unchanged.

    Raises ValueError for a variable that is identically zero, or whose
    pooled residual variance is below ``_DEGENERATE_RESID_REL`` times its
    mean square (a constant series, or one that the lags predict exactly).
    Its Gaussian likelihood would be degenerate.
    """
    rms = np.sqrt(np.mean(Y ** 2, axis=0))
    zero = np.flatnonzero(~(rms > 0))
    if zero.size:
        raise ValueError(
            f"ms_var_fit: variable(s) {zero.tolist()} are identically zero")
    Yr = Y / rms
    Yd = Yr[p:]
    if p > 0:
        X = _lag_matrix(Yr, p)
        beta = np.linalg.lstsq(X, Yd, rcond=None)[0]
        r = Yd - X @ beta
    else:
        r = Yd - Yd.mean(axis=0)
    d = np.mean(r ** 2, axis=0)
    bad = np.flatnonzero(~(d > _DEGENERATE_RESID_REL))
    if bad.size:
        raise ValueError(
            f"ms_var_fit: variable(s) {bad.tolist()} have (numerically) zero "
            f"residual variance in a VAR({p}); the series is constant or an "
            "exact function of the lags, and the Gaussian likelihood is "
            "degenerate")
    sd = np.sqrt(d)
    return rms * sd, r / sd


def ms_var_fit(
    Y: np.ndarray,
    K: int = 2,
    p: int = 1,
    *,
    n_iter: int = 500,
    tol: float = 1e-5,
    seed: int = 0,
    verbose: bool = False,
) -> MSVARResult:
    """EM (ECM) estimation of an MSIH(K)-VAR(p) with a shared AR matrix.

    Model (see the module docstring for the estimator, the variance
    floor and the relation to Hamilton 1989)::

        y_t = mu_{s_t} + A_1 y_{t-1} + ... + A_p y_{t-p} + e_t,
        e_t | s_t = k ~ N(0, Sigma_k),   Pr(s_t = j | s_{t-1} = i) = P[i, j].

    Parameters
    ----------
    Y : (T, n) array_like. A 1-D array is treated as one variable. The
        data must be finite. The estimates are equivariant to the units
        of each variable: the fit runs on each variable divided by its
        pooled residual standard deviation and is mapped back.
    K : int >= 1, the number of regimes.
    p : int >= 0, the VAR lag order (shared across regimes). ``p = 0``
        gives a switching mean-and-covariance (hidden Markov) model.
        ``T - p`` must exceed ``K + n*p``, the number of regression
        coefficients per equation.
    n_iter : int >= 1, the maximum number of EM iterations. The default
        was 100 up to release 4.3.0. EM converges linearly and can crawl
        across flat stretches before it reaches a maximum. On Hamilton's
        GNP series with ``K=2, p=4``, the corrected EM needs about 120
        iterations to reach ``tol=1e-5``.
    tol : float, the absolute tolerance on the log-likelihood change
        between iterations. ``converged`` is True only if the change is
        below ``tol`` and is not a fall. A loose ``tol`` can stop EM on
        such a flat stretch.
    seed : int, the seed for the jitter added to the initial regime
        clustering. Periods are first assigned to regimes by the quantiles
        of ``sum_j r_tj^2 / d_j``, the squared residuals of a no-intercept
        OLS VAR (demeaned data when ``p = 0``), each divided by that
        variable's pooled residual variance ``d_j``. Regime 0 therefore
        starts as the low-volatility regime.
    verbose : bool. If True, print the log-likelihood at each iteration.

    Returns
    -------
    MSVARResult. ``loglik``, ``smoothed_probs`` and ``filtered_probs``
    are those of the returned parameters. ``loglik_path`` holds every
    iterate's log-likelihood, and ``sigma_at_floor`` flags regimes whose
    covariance sits at the variance floor.

    Raises
    ------
    ValueError
        For non-finite data, ``K < 1``, ``p < 0``, ``n_iter < 1``,
        ``T - p <= K + n*p``, or a variable that is identically zero,
        constant, or predicted exactly by the lags.

    Warns
    -----
    RuntimeWarning
        If the log-likelihood falls by more than rounding error in any
        iteration. That breaks the ECM guarantee and points to a
        numerical failure, for example a singular GLS system.
    RuntimeWarning
        If a returned ``Sigma_k`` has an eigenvalue within a factor 10 of
        the variance floor: the regime collapsed onto a few observations
        (``sigma_at_floor[k]`` is True).

    Notes
    -----
    Regime labels are not identified. Order them after estimation, for
    example by ``mu`` or by ``Sigma``.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    if Y.ndim != 2:
        raise ValueError(f"ms_var_fit: Y must be 1-D or 2-D, got ndim={Y.ndim}")
    if not np.all(np.isfinite(Y)):
        raise ValueError("ms_var_fit: Y contains NaN or inf")
    K = int(K)
    p = int(p)
    n_iter = int(n_iter)
    if K < 1:
        raise ValueError(f"ms_var_fit: K must be >= 1, got {K}")
    if p < 0:
        raise ValueError(f"ms_var_fit: p must be >= 0, got {p}")
    if n_iter < 1:
        raise ValueError(f"ms_var_fit: n_iter must be >= 1, got {n_iter}")
    T, n = Y.shape
    n_coef = K + n * p
    if T - p <= n_coef:
        raise ValueError(
            "ms_var_fit: need more observations than regression "
            "coefficients per equation, T - p > K + n*p; got "
            f"T - p = {T - p}, K + n*p = {n_coef}")
    rng = np.random.default_rng(seed)

    # --- standardise: variable j divided by its pooled residual sd s_j ---
    # All estimation runs on y_t / s; the estimates are mapped back at the
    # end, and log f(y) = log f(y / s) - (T - p) * sum_j log s_j.
    s, resid_init = _pooled_residual_scale(Y, p)
    Ys = Y / s
    # y_t = mu_{s_t} + A x_t + e_t, x_t = [y_{t-1}; ...; y_{t-p}]
    Y_dep = Ys[p:]
    X_lag = _lag_matrix(Ys, p)
    T_eff = Y_dep.shape[0]
    ll_shift = -T_eff * float(np.sum(np.log(s)))

    # --- initialisation ---
    # Soft-cluster the periods by the size of the standardised residuals
    # of a single no-intercept VAR, plus a small jitter so EM can move.
    score = (resid_init ** 2).sum(axis=1)
    qs = np.quantile(score, np.linspace(0, 1, K + 1)[1:-1])
    sm_init = np.zeros((T_eff, K))
    sm_init[np.arange(T_eff), np.digitize(score, qs)] = 1.0
    sm_init = sm_init + 0.05 * rng.uniform(size=sm_init.shape)
    sm_init = sm_init / sm_init.sum(axis=1, keepdims=True)

    # Pooled residual covariance of the standardised data (unit diagonal)
    # and the variance floor, both in standardised units.
    S0 = resid_init.T @ resid_init / T_eff
    var_floor = _VAR_FLOOR_REL

    params = {
        "A": np.zeros((n, n * p)),
        "mu": np.zeros((K, n)),
        "Sigma": None,
        "P": np.full((K, K), 1.0 / K),
        "pi0": np.full(K, 1.0 / K),
    }
    smoothed, pairwise = sm_init, None
    path: list[float] = []
    converged = False
    n_decreases = 0
    worst_fall = 0.0

    for it in range(n_iter):
        # ----- M-step (ECM cycle), then E-step at the new parameters -----
        params = _m_step(Y_dep, X_lag, smoothed, pairwise, params, var_floor,
                         sigma_default=S0)
        log_dens = _log_densities(Y_dep, X_lag, params["A"], params["mu"],
                                  params["Sigma"])
        filt, _, ll_std = _hamilton_filter_log(log_dens, params["P"],
                                               params["pi0"])
        smoothed, pairwise = _kim_smoother(filt, params["P"])
        ll = ll_std + ll_shift
        path.append(ll)

        if verbose:
            print(f"  iter {it+1:>3d}: loglik = {ll:.6f}")

        if it > 0:
            conv, decreased = _em_step_status(path[-2], ll, tol)
            if decreased:
                n_decreases += 1
                worst_fall = min(worst_fall, ll - path[-2])
            if conv:
                converged = True
                break

    if n_decreases:
        warnings.warn(
            f"ms_var_fit: the EM log-likelihood decreased in {n_decreases} "
            f"iteration(s) (largest fall {worst_fall:.3e}); the returned "
            "estimates may not be a local maximum.",
            RuntimeWarning, stacklevel=2,
        )

    # --- floor check (standardised units) and map back to the units of Y ---
    Sigma_std = params["Sigma"]
    lam_min = np.array([np.linalg.eigvalsh(S).min() for S in Sigma_std])
    at_floor = lam_min <= _FLOOR_WARN_FACTOR * var_floor
    if at_floor.any():
        warnings.warn(
            f"ms_var_fit: the covariance of regime(s) "
            f"{np.flatnonzero(at_floor).tolist()} has an eigenvalue at the "
            f"variance floor ({var_floor:g} times the pooled residual "
            "variance, per standardised variable). The regime has collapsed "
            "onto a few observations, where the MSIH likelihood is "
            "unbounded, so its estimates are not an interior maximum. Try "
            "another seed, fewer regimes or a longer sample.",
            RuntimeWarning, stacklevel=2,
        )
    A = params["A"] * s[:, None] / np.tile(s, p)[None, :]
    mu = params["mu"] * s[None, :]
    Sigma = Sigma_std * s[None, :, None] * s[None, None, :]

    return MSVARResult(
        K=K, A=A, mu=mu, Sigma=Sigma,
        P=params["P"], smoothed_probs=smoothed, filtered_probs=filt,
        loglik=float(ll), n_iter=it + 1, converged=converged,
        pi0=params["pi0"], loglik_path=np.asarray(path, dtype=float),
        sigma_at_floor=at_floor,
    )


__all__ = ["ms_var_fit", "MSVARResult"]
