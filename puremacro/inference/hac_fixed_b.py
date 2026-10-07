"""Fixed-b HAC inference with the Bartlett kernel (Kiefer and Vogelsang 2005).

``hac_fixed_b`` is the OLS Newey-West (Bartlett-kernel) sandwich with a
bandwidth that is a fixed fraction ``b`` of the sample size.
``llsw_critical_value`` gives the critical value to use with its
t-statistics in place of the standard normal quantile.

Fixed-b asymptotics
-------------------
Hold the bandwidth ratio ``b = M/T`` in (0, 1] fixed as T grows. Under H0,
Theorem 3 of Kiefer and Vogelsang (2005) gives the limit of the t-statistic::

    t*_b  =>  W(1) / sqrt(Q_1(b)),
    Q_1(b) = (2/b) int_0^1 B(r)^2 dr - (2/b) int_0^{1-b} B(r+b) B(r) dr

(Bartlett kernel, KV 2005, Definition 1). Here W is a standard Brownian
motion and B(r) = W(r) - r W(1) is a Brownian bridge, independent of W(1).
The limit is symmetric and has heavier tails than N(0, 1), and it depends on b.

Critical values
---------------
* **90%, 95%, 97.5% and 99% quantiles.** These are transcribed from Kiefer, N.M.
  and T.J. Vogelsang (2005), "A New Asymptotic Theory for Heteroskedasticity-
  Autocorrelation Robust Tests", Econometric Theory 21(6), 1130-1164,
  Table I, "Asymptotic Critical Values for t*_b Using Bartlett Kernel", for
  b = 0.02, 0.04, ..., 1.0. The copy read was the working-paper version,
  Cornell CAE Working Paper 05-08 (updated January 2005), printed p. 24
  (PDF p. 26), https://cae.economics.cornell.edu/05-08.pdf. The table notes
  say the b = 1 entries are analytical. The others are KV's simulations
  (50,000 replications, T = 1,000, bandwidth M = bT).
* **How a test level maps to a row.** Table I gives right-tail quantiles of a
  symmetric law. A two-sided test at level alpha therefore uses the
  1 - alpha/2 quantile: alpha = 0.20, 0.10, 0.05, 0.02 map to the 90%, 95%,
  97.5% and 99% rows. A one-sided test (``two_sided=False``) uses the 1 - alpha
  quantile.
* **99.5% quantile.** A two-sided 1% test needs this row, and Table I does not
  have it. It was computed for puremacro from the exact law of the same
  discretisation KV simulate: T = 2,000 normalised partial sums of iid N(0, 1),
  Bartlett weights with M = bT. Because W(1) is independent of B,
  P(|t| > c) = P(Z^2 - c^2 Q > 0), with Z ~ N(0, 1) and
  Q = sum_j lambda_j chi2_j(1), where lambda_j are the eigenvalues of the
  double-centred Bartlett kernel matrix divided by T. This probability was
  evaluated by Imhof's (1961) numerical inversion and solved for c.
* **Checks on the computed row.**
  - The same computation reproduces KV's analytical b = 1 values
    (2.740, 3.764, 4.771, 6.090) to three decimals.
  - It matches the simulated rows of Table I to within their Monte Carlo
    error (largest |difference| 0.061).
  - The 99.5% values move by at most 0.0004 between T = 500 and T = 4,000
    (checked at b = 0.02, 0.1, 0.5 and 1).
  - An independent conditional Monte Carlo (200,000 draws, T = 500) agrees
    to within two standard errors at b = 0.1, 0.3, 0.5, 0.7 and 1.0
    (for example b = 0.5: 5.162, s.e. 0.005, against 5.153).
* **Values of b between and below the grid.** Between grid points, values are
  interpolated linearly in b. KV 2005, section 3.3, say: "Critical values can
  be interpolated for values of b that fall between the values on the grid".
  For 0 < b < 0.02, the value is interpolated linearly between the N(0, 1)
  quantile (the b -> 0 limit) and the b = 0.02 entry.

Bandwidth convention
--------------------
``hac_fixed_b`` uses L = max(1, floor(bT)) lags with Newey-West (1987) weights
1 - l/(L+1). This is the Bartlett kernel k(l/M) = 1 - l/M with M = L + 1. KV
2005, section 3.3: "given the kernel and bandwidth, M, we recommend that the
critical value corresponding to b = M/T be used". ``hac_fixed_b`` therefore
reports ``b_eff = min(1, (L+1)/T)`` and looks its critical values up at
``b_eff``.

The name ``llsw_critical_value`` is historical. Lazarus, Lewis, Stock and
Watson (2018, JBES 36, 541-559) recommend Newey-West tests with KV fixed-b
critical values. Their Table 2 lists constants for bandwidth rules, not
critical values, so it is not the source of these numbers.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

from .._linalg import inv_xtx

# Bandwidth ratios b = M/T of KV (2005) Table I: 0.02, 0.04, ..., 1.00.
# k/50 is correctly rounded, so each entry equals the decimal literal.
_KV_B = np.arange(1, 51) / 50.0

# Kiefer-Vogelsang (2005) Table I, Bartlett kernel: right-tail quantiles of
# t*_b, keyed by quantile level. Each row lists b = 0.02 ... 1.00 in the
# paper's five blocks of ten. Source: CAE WP 05-08, printed p. 24 (PDF p. 26).
_KV_TABLE_I = {
    0.90: (
        1.323, 1.343, 1.368, 1.390, 1.414, 1.442, 1.469, 1.498, 1.529, 1.563,
        1.587, 1.615, 1.644, 1.674, 1.709, 1.737, 1.767, 1.796, 1.830, 1.865,
        1.901, 1.931, 1.963, 1.994, 2.022, 2.053, 2.086, 2.117, 2.149, 2.183,
        2.211, 2.242, 2.272, 2.297, 2.325, 2.359, 2.389, 2.418, 2.450, 2.477,
        2.506, 2.533, 2.560, 2.591, 2.618, 2.649, 2.678, 2.706, 2.733, 2.740,
    ),
    0.95: (
        1.690, 1.731, 1.772, 1.813, 1.861, 1.902, 1.944, 1.988, 2.030, 2.081,
        2.124, 2.179, 2.222, 2.274, 2.324, 2.367, 2.412, 2.459, 2.505, 2.556,
        2.601, 2.651, 2.696, 2.739, 2.781, 2.828, 2.872, 2.913, 2.956, 3.007,
        3.048, 3.082, 3.124, 3.162, 3.198, 3.245, 3.291, 3.330, 3.367, 3.408,
        3.444, 3.494, 3.537, 3.579, 3.616, 3.654, 3.692, 3.727, 3.764, 3.764,
    ),
    0.975: (
        2.018, 2.072, 2.125, 2.179, 2.235, 2.296, 2.355, 2.417, 2.481, 2.553,
        2.617, 2.678, 2.736, 2.805, 2.878, 2.930, 2.999, 3.067, 3.129, 3.192,
        3.253, 3.318, 3.382, 3.447, 3.514, 3.567, 3.636, 3.684, 3.740, 3.783,
        3.834, 3.880, 3.934, 3.995, 4.054, 4.101, 4.153, 4.208, 4.260, 4.312,
        4.367, 4.417, 4.470, 4.524, 4.568, 4.617, 4.664, 4.713, 4.764, 4.771,
    ),
    0.99: (
        2.377, 2.459, 2.537, 2.627, 2.709, 2.792, 2.882, 2.961, 3.051, 3.140,
        3.229, 3.315, 3.385, 3.476, 3.580, 3.707, 3.791, 3.858, 3.942, 4.038,
        4.111, 4.212, 4.306, 4.399, 4.480, 4.567, 4.645, 4.711, 4.762, 4.831,
        4.912, 4.981, 5.041, 5.124, 5.190, 5.279, 5.333, 5.376, 5.445, 5.493,
        5.554, 5.609, 5.672, 5.724, 5.782, 5.868, 5.933, 5.998, 6.058, 6.090,
    ),
}

# 99.5% quantile (two-sided 1%, one-sided 0.5%). Table I does not have this row.
# It was computed for puremacro by exact Imhof inversion of the T = 2,000
# discretisation (see the module docstring) and rounded to three decimals.
_Q995_COMPUTED = (
    2.669, 2.765, 2.864, 2.965, 3.069, 3.174, 3.281, 3.389, 3.498, 3.607,
    3.717, 3.826, 3.935, 4.043, 4.151, 4.257, 4.363, 4.466, 4.569, 4.670,
    4.769, 4.868, 4.964, 5.060, 5.153, 5.245, 5.336, 5.424, 5.511, 5.596,
    5.680, 5.761, 5.842, 5.921, 5.998, 6.075, 6.150, 6.224, 6.297, 6.370,
    6.441, 6.513, 6.584, 6.655, 6.726, 6.797, 6.869, 6.941, 7.012, 7.083,
)

_FIXED_B_QUANTILES = {**_KV_TABLE_I, 0.995: _Q995_COMPUTED}


def _quantile_level(alpha: float, two_sided: bool) -> float:
    """Map a test level to a tabulated right-tail quantile level, or raise."""
    level = 1.0 - alpha / 2.0 if two_sided else 1.0 - alpha
    for q in _FIXED_B_QUANTILES:
        if abs(level - q) < 1e-9:
            return q
    if two_sided:
        supported = sorted(round(2.0 * (1.0 - q), 6) for q in _FIXED_B_QUANTILES)
    else:
        supported = sorted(round(1.0 - q, 6) for q in _FIXED_B_QUANTILES)
    kind = "two-sided" if two_sided else "one-sided"
    raise ValueError(
        f"alpha={alpha!r} is not a tabulated {kind} fixed-b level; "
        f"supported {kind} alpha values are {supported}."
    )


def llsw_critical_value(b: float, alpha: float = 0.10, *, two_sided: bool = True) -> float:
    """Bartlett-kernel fixed-b critical value (Kiefer-Vogelsang 2005, Table I).

    Parameters
    ----------
    b : float
        Bandwidth ratio M/T in (0, 1], where M is the Bartlett bandwidth
        (weights k(j/M) = 1 - j/M). For output from ``hac_fixed_b``, pass its
        ``b_eff``.
    alpha : float, default 0.10
        Test level. Two-sided levels are 0.20, 0.10, 0.05, 0.02 and 0.01, and
        use the 1 - alpha/2 quantile. One-sided levels are 0.10, 0.05, 0.025,
        0.01 and 0.005, and use the 1 - alpha quantile. Any other value raises
        ``ValueError``.
    two_sided : bool, default True
        If True, return c with P(|t*_b| > c) = alpha. If False, return c with
        P(t*_b > c) = alpha. The limit law is symmetric, so the left-tail
        value is -c.

    Returns
    -------
    float
        The critical value, on the scale of the t-statistic.

    Notes
    -----
    The 90/95/97.5/99% quantiles come from KV (2005) Table I (CAE WP 05-08,
    p. 24). The b = 1 entries there are analytical, and the rest are
    simulated. The 99.5% quantile is computed exactly, as described in the
    module docstring. Between grid points (b = 0.02, 0.04, ..., 1) values are
    interpolated linearly in b. Below 0.02 they are interpolated linearly
    towards the N(0, 1) quantile at b = 0. Example: b = 1 and alpha = 0.05
    (two-sided) give 4.771; b = 0.1 and alpha = 0.10 give 1.861.
    """
    b = float(b)
    alpha = float(alpha)
    if not (math.isfinite(b) and 0.0 < b <= 1.0):
        raise ValueError(f"b must lie in (0, 1]; got b={b!r}.")
    q = _quantile_level(alpha, bool(two_sided))
    row = _FIXED_B_QUANTILES[q]
    if b < _KV_B[0]:
        z = NormalDist().inv_cdf(q)
        return float(z + (b / _KV_B[0]) * (row[0] - z))
    return float(np.interp(b, _KV_B, row))


def hac_fixed_b(y, X, b: float = 0.10) -> dict:
    """OLS with Bartlett (Newey-West) HAC standard errors, bandwidth a fixed fraction b of T.

    Parameters
    ----------
    y : array_like, shape (T,)
    X : array_like, shape (T, k)
        Regressors. Include a column of ones for an intercept.
    b : float, default 0.10
        Bandwidth ratio in (0, 1]. The estimator uses L = max(1, floor(bT))
        autocovariance lags with weights 1 - l/(L+1), which is the Bartlett
        kernel k(l/M) = 1 - l/M with bandwidth M = L + 1 (Newey-West 1987).

    Returns
    -------
    dict
        ``beta``, ``se``, ``t`` (= beta/se, for H0: beta_j = 0), ``vcov``,
        ``residuals``, ``n_obs``, ``b`` (as passed), ``lags`` (L),
        ``bandwidth`` (M = L + 1) and ``b_eff`` (min(1, M/T), the ratio that
        fixed-b critical values should be looked up at; see KV 2005, section
        3.3). It also holds the two-sided fixed-b critical values at
        ``b_eff``: ``fixed_b_cv_90``, ``fixed_b_cv_95`` and ``fixed_b_cv_99``
        (alpha = 0.10, 0.05 and 0.01). ``llsw_cv_90`` is kept as a legacy
        alias of ``fixed_b_cv_90``.

    Notes
    -----
    For inference, compare |t| with a fixed-b critical value, not a normal
    quantile. Use one of the returned values, or
    ``llsw_critical_value(out["b_eff"], alpha)`` for other levels. See the
    module docstring for sources.
    """
    b = float(b)
    if not (math.isfinite(b) and 0.0 < b <= 1.0):
        raise ValueError(f"b must lie in (0, 1]; got b={b!r}.")
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    T, k = X.shape
    # The 1e-9 guard keeps floor(bT) exact when bT is an integer in decimal
    # arithmetic but not in binary (0.29 * 100 = 28.999999999999996).
    L = max(1, int(np.floor(b * T + 1e-9)))
    M = L + 1
    b_eff = min(1.0, M / T)
    XtX_inv = inv_xtx(X, name="hac_fixed_b")
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    S = (X * u[:, None]).T @ (X * u[:, None])  # ell=0
    for ell in range(1, L + 1):
        w = 1.0 - ell / (L + 1.0)
        Gamma = (X[ell:] * u[ell:, None]).T @ (X[:-ell] * u[:-ell, None])
        S += w * (Gamma + Gamma.T)
    vcov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(vcov))
    cv90 = float(llsw_critical_value(b_eff, alpha=0.10))
    return {
        "beta": beta,
        "se": se,
        "t": beta / se,
        "vcov": vcov,
        "residuals": u,
        "n_obs": int(T),
        "b": float(b),
        "lags": int(L),
        "bandwidth": int(M),
        "b_eff": float(b_eff),
        "fixed_b_cv_90": cv90,
        "fixed_b_cv_95": float(llsw_critical_value(b_eff, alpha=0.05)),
        "fixed_b_cv_99": float(llsw_critical_value(b_eff, alpha=0.01)),
        "llsw_cv_90": cv90,
    }


__all__ = ["hac_fixed_b", "llsw_critical_value"]
