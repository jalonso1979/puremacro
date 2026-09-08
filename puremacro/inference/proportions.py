"""Confidence intervals for a binomial proportion.

A pure-numpy/scipy drop-in for
``statsmodels.stats.proportion.proportion_confint``, covering the five
closed-form methods: ``normal``, ``agresti_coull``, ``beta``
(Clopper-Pearson), ``wilson`` and ``jeffreys``.

Which one you pick matters most exactly where a macro application needs it
most — small ``nobs``, and counts at or near the boundary. The Wald
(``normal``) interval degenerates to the single point ``[p, p]`` when
``count`` is 0 or ``nobs``, which is not a confidence interval at all; the
Wilson score interval, which inverts the score test instead of the Wald
test, stays a genuine interval there and is why the corpus uses it
(``N04_macro_shocks_and_volcanoes.py:126``, the share of volcanic years
followed by a cold anomaly).

Notes
-----
Verified expression by expression against statsmodels 0.14.6
(``statsmodels/stats/proportion.py:111``). None of the five methods applies
a continuity correction; ``normal`` and ``agresti_coull`` are clipped to
[0, 1] and the other three are not, which is statsmodels' rule and is
reproduced here rather than tidied up.

References
----------
Wilson, E. B. (1927). Probable inference, the law of succession, and
    statistical inference. JASA 22(158), 209-212.
Clopper, C. J. and Pearson, E. S. (1934). The use of confidence or fiducial
    limits illustrated in the case of the binomial. Biometrika 26(4).
Agresti, A. and Coull, B. A. (1998). Approximate is better than "exact" for
    interval estimation of binomial proportions. The American Statistician
    52(2), 119-126.
Brown, L. D., Cai, T. T. and DasGupta, A. (2001). Interval estimation for a
    binomial proportion. Statistical Science 16(2), 101-133.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["proportion_confint"]

_METHODS = ("normal", "agresti_coull", "beta", "wilson", "jeffreys")


def proportion_confint(count, nobs, alpha=0.05, method="normal"):
    """Confidence interval for a binomial proportion.

    Drop-in replacement for
    ``statsmodels.stats.proportion.proportion_confint`` (statsmodels
    0.14.6) for the five closed-form methods.

    Parameters
    ----------
    count : int, float, array_like, or pandas.Series
        Number of successes.
    nobs : int, float, or array_like
        Number of trials. Broadcast against ``count``.
    alpha : float, default 0.05
        Significance level; the interval has nominal coverage ``1 - alpha``.
    method : {'normal', 'agresti_coull', 'beta', 'wilson', 'jeffreys'}
        Interval to compute. ``'beta'`` is the Clopper-Pearson exact
        interval. Any string starting ``'jeff'`` selects Jeffreys, matching
        statsmodels' deliberately forgiving prefix match.

    Returns
    -------
    ci_low, ci_upp : tuple
        Two floats when both ``count`` and ``nobs`` are scalars; two
        pandas objects (indexed like ``count``) when ``count`` is a Series
        or DataFrame; two ndarrays otherwise.

    Raises
    ------
    ValueError
        If ``alpha`` is not in ``(0, 1)``, if ``nobs <= 0``, or if any
        ``count`` is negative or exceeds its ``nobs``.
    NotImplementedError
        If ``method`` is not one of the five implemented here. In
        particular ``'binom_test'`` (statsmodels' numerical inversion of the
        exact binomial test) is not implemented; use ``'beta'``, which is
        the closed-form exact interval, or call
        ``scipy.stats.binomtest(...).proportion_ci()``.

    Notes
    -----
    **Wilson at the boundaries.** With ``count = 0`` the Wilson centre and
    half-width are both ``c^2 / (2n) / (1 + c^2/n)``, so the lower limit is
    zero to within one rounding step and the upper limit is
    ``c^2 / n / (1 + c^2/n)`` — a strictly positive, informative bound.
    Symmetrically at ``count = nobs``. statsmodels does **not** clip the
    Wilson interval, so the lower limit at ``count = 0`` can come back as a
    value of order ``1e-17`` rather than exactly ``0``; this function
    reproduces that arithmetic in the same order and returns the same bits
    rather than clipping to a prettier zero.

    **Clopper-Pearson at the boundaries.** ``beta`` would evaluate
    ``Beta(0, n+1)`` at ``count = 0``, which is undefined; statsmodels
    substitutes 0 for the lower limit (and 1 for the upper limit when
    ``count = nobs``). That substitution is reproduced. It also means the
    boundary intervals are one-sided, so their coverage is ``1 - alpha/2``,
    not ``1 - alpha``.

    **Jeffreys is not adjusted at the boundaries** — with ``count = 0`` the
    Beta(0.5, n+0.5) posterior gives a strictly positive lower limit. That
    is intended: it is a Bayesian credible interval, not an inversion.

    **Input validation is stricter than statsmodels.** ``nobs <= 0`` or
    ``count`` outside ``[0, nobs]`` raise here; statsmodels returns ``nan``
    or a nonsensical interval. No legitimate call is affected.

    Examples
    --------
    >>> lo, hi = proportion_confint(7, 25, alpha=0.05, method="wilson")
    >>> round(lo, 6), round(hi, 6)
    (0.142839, 0.475766)

    The Wald interval collapses at the boundary; Wilson does not:

    >>> proportion_confint(0, 20, method="normal")
    (0.0, 0.0)
    >>> lo, hi = proportion_confint(0, 20, method="wilson")
    >>> round(lo, 12), round(hi, 6)
    (0.0, 0.161125)

    Vectorised over an array of counts:

    >>> import numpy as np
    >>> lo, hi = proportion_confint(np.array([0, 5, 20]), 20, method="wilson")
    >>> np.round(hi, 4)
    array([0.1611, 0.4687, 1.    ])
    """
    is_scalar = np.isscalar(count) and np.isscalar(nobs)
    is_pandas = isinstance(count, (pd.Series, pd.DataFrame))

    # statsmodels routes both arguments through ``array_like(..., ndim=None)``,
    # whose default dtype is float64. Matching that matters: it decides
    # whether ``crit2 / (4 * nobs ** 2)`` is evaluated in integer or float
    # arithmetic on large nobs.
    count_a = np.asarray(count, dtype=float)
    nobs_a = np.asarray(nobs, dtype=float)

    if not np.isscalar(alpha) or not (0.0 < float(alpha) < 1.0):
        raise ValueError(
            f"proportion_confint: alpha must be a scalar in (0, 1), got "
            f"{alpha!r}."
        )
    if np.any(nobs_a <= 0):
        raise ValueError(
            "proportion_confint: nobs must be strictly positive; a "
            "proportion is undefined with no trials."
        )
    if np.any(count_a < 0) or np.any(count_a > nobs_a):
        raise ValueError(
            "proportion_confint: every count must satisfy 0 <= count <= "
            "nobs."
        )

    q_ = count_a / nobs_a
    alpha_2 = 0.5 * alpha

    if method == "normal":
        std_ = np.sqrt(q_ * (1 - q_) / nobs_a)
        dist = stats.norm.isf(alpha / 2.0) * std_
        ci_low = q_ - dist
        ci_upp = q_ + dist

    elif method == "beta":
        # Clopper-Pearson by inverting the Beta CDF. Undefined at the
        # boundaries (shape parameter 0), where statsmodels substitutes the
        # degenerate limit.
        ci_low = stats.beta.ppf(alpha_2, count_a, nobs_a - count_a + 1)
        ci_upp = stats.beta.isf(alpha_2, count_a + 1, nobs_a - count_a)
        if np.ndim(ci_low) > 0:
            ci_low = np.asarray(ci_low)
            ci_upp = np.asarray(ci_upp)
            ci_low.flat[q_.flat == 0] = 0
            ci_upp.flat[q_.flat == 1] = 1
        else:
            ci_low = 0 if q_ == 0 else ci_low
            ci_upp = 1 if q_ == 1 else ci_upp

    elif method == "agresti_coull":
        crit = stats.norm.isf(alpha / 2.0)
        nobs_c = nobs_a + crit ** 2
        q_c = (count_a + crit ** 2 / 2.0) / nobs_c
        std_c = np.sqrt(q_c * (1.0 - q_c) / nobs_c)
        dist = crit * std_c
        ci_low = q_c - dist
        ci_upp = q_c + dist

    elif method == "wilson":
        # Inversion of the score test. The order of operations below is
        # statsmodels' verbatim; ``dist`` is formed and *then* divided by
        # ``denom`` (rather than folded into one expression) because that is
        # where the last-bit behaviour at count=0 comes from.
        crit = stats.norm.isf(alpha / 2.0)
        crit2 = crit ** 2
        denom = 1 + crit2 / nobs_a
        center = (q_ + crit2 / (2 * nobs_a)) / denom
        dist = crit * np.sqrt(
            q_ * (1.0 - q_) / nobs_a + crit2 / (4.0 * nobs_a ** 2)
        )
        dist /= denom
        ci_low = center - dist
        ci_upp = center + dist

    elif isinstance(method, str) and method[:4] == "jeff":
        # Central credible interval of the Beta(1/2, 1/2) reference prior
        # posterior. statsmodels matches on the prefix, not the full name.
        ci_low, ci_upp = stats.beta.interval(
            1 - alpha, count_a + 0.5, nobs_a - count_a + 0.5
        )

    else:
        raise NotImplementedError(
            f"proportion_confint: method {method!r} is not available. "
            f"Implemented: {', '.join(_METHODS)}."
        )

    if method in ["normal", "agresti_coull"]:
        ci_low = np.clip(ci_low, 0, 1)
        ci_upp = np.clip(ci_upp, 0, 1)
    if is_pandas:
        container = pd.Series if isinstance(count, pd.Series) else pd.DataFrame
        ci_low = container(ci_low, index=count.index)
        ci_upp = container(ci_upp, index=count.index)
    if is_scalar:
        return float(ci_low), float(ci_upp)
    return ci_low, ci_upp
