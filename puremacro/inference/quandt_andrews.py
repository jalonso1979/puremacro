"""Quandt-Andrews supF test for parameter constancy.

For a regression y = X β + ε, tests H0: β constant over t vs.
H1: β changes once at unknown date τ ∈ [trim·T, (1-trim)·T].
The supF statistic is sup_τ F(τ) where F(τ) is the Chow F at τ.
Asymptotic critical values per Andrews (1993, Econometrica), Table 1.

This module wraps :func:`puremacro.tests.breaks.bai_perron_regression`
restricted to ``max_breaks=1, force_m=1`` and returns the supF +
Hansen-approximated asymptotic p-value.
"""
from __future__ import annotations

import numpy as np

from puremacro.tests.breaks import bai_perron_regression


# Andrews (1993) Econometrica Table 1, asymptotic 5% / 1% CV for the supF
# statistic as a function of (q, π) where q = number of regressors tested
# = ncol(X) and π = trim. Hard-coded for the three trim values in common
# use; (q, π) outside the table returns NaN-typed sentinels.
_ANDREWS_CV = {
    (1, 0.15): {"5pct": 8.85, "1pct": 12.16},
    (2, 0.15): {"5pct": 11.79, "1pct": 15.55},
    (3, 0.15): {"5pct": 14.27, "1pct": 18.36},
    (4, 0.15): {"5pct": 16.51, "1pct": 21.04},
    (5, 0.15): {"5pct": 18.72, "1pct": 23.59},
    (1, 0.10): {"5pct": 9.84, "1pct": 13.42},
    (1, 0.20): {"5pct": 7.78, "1pct": 11.23},
}


def _hansen_pvalue_approx(supF: float, q: int, trim: float) -> float:
    """Hansen-style asymptotic p-value approximation.

    Andrews (1993) Table I tabulates critical values for the Wald-style
    statistic ``q × F``, which is the asymptotic limit of the LM/LR
    test statistics. Multiply the small-sample F by ``q`` before
    comparing to those CVs:
      - q×supF ≥ 1pct CV  ->  0.005
      - q×supF ≥ 5pct CV  ->  0.025
      - q×supF <  5pct CV ->  linear interpolation calibrated for ~5% size
    """
    cv = _ANDREWS_CV.get((q, round(trim, 2)))
    if cv is None:
        return float("nan")
    wald_stat = q * supF
    if wald_stat >= cv["1pct"]:
        return 0.005
    if wald_stat >= cv["5pct"]:
        return 0.025
    # Below 5% CV: linear interpolation in (wald_stat, p-value) space
    # anchored at (0, 1.0) and (cv_5pct, 0.05). Order-preserving and
    # gives ~nominal 5% size under H0.
    ratio = wald_stat / max(cv["5pct"], 1e-6)
    p_raw = 1.0 - 0.95 * ratio
    return float(max(0.05, min(1.0, p_raw)))


def quandt_andrews_supF(
    y,
    X,
    *,
    trim: float = 0.15,
) -> dict:
    """Quandt-Andrews supF test of H0: no break vs H1: one break.

    Parameters
    ----------
    y : array-like, shape (T,)
        Dependent variable.
    X : array-like, shape (T, q)
        Design matrix (include a constant column explicitly if intended).
    trim : float
        Fraction of T trimmed from each end of the candidate-break range.

    Returns
    -------
    dict with keys:
        ``supF`` — sup_τ F(τ), the small-sample Chow F-statistic.
        ``breakdate`` — integer index of arg max (in [trim·T, (1-trim)·T])
        ``pvalue`` — Hansen-approximated asymptotic p-value (compares
                     ``q × supF`` against Andrews' Wald-scale CVs).
        ``cv_5pct``, ``cv_1pct`` — Andrews (1993) Table I CVs, on the
                                   Wald scale (multiply ``supF`` by
                                   ``q`` for direct comparison). NaN
                                   if ``(q, trim)`` is outside the table.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    T, q = X.shape
    bp = bai_perron_regression(y, X, max_breaks=1, trim=trim, force_m=1)
    # bp["breaks"] is dict[int, list[int]]; m=1 entry is the single-break list.
    one_break = bp["breaks"].get(1, [])
    breakdate = int(one_break[0]) if one_break else int(T // 2)
    # Restricted SSR: full-sample OLS
    beta_full = np.linalg.lstsq(X, y, rcond=None)[0]
    ssr_restricted = float(((y - X @ beta_full) ** 2).sum())
    # Unrestricted SSR: total SSR at the optimal one-break partition.
    ssr_unrestricted = float(bp["ssrs"].get(1, ssr_restricted))
    df1 = q
    df2 = T - 2 * q
    if df2 <= 0 or ssr_unrestricted <= 0:
        return {
            "supF": float("nan"),
            "breakdate": breakdate,
            "pvalue": float("nan"),
            "cv_5pct": float("nan"),
            "cv_1pct": float("nan"),
        }
    supF = ((ssr_restricted - ssr_unrestricted) / df1) / (ssr_unrestricted / df2)
    cv = _ANDREWS_CV.get(
        (q, round(trim, 2)),
        {"5pct": float("nan"), "1pct": float("nan")},
    )
    return {
        "supF": float(supF),
        "breakdate": breakdate,
        "pvalue": _hansen_pvalue_approx(supF, q, trim),
        "cv_5pct": cv["5pct"],
        "cv_1pct": cv["1pct"],
    }


__all__ = ["quandt_andrews_supF"]
