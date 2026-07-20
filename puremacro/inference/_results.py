"""Frozen-dataclass result objects for puremacro.inference."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ARTestResult:
    """Result of :func:`puremacro.inference.weak_iv.anderson_rubin_test`.

    Attributes
    ----------
    stat : float
        Anderson-Rubin F-statistic.
    p_value : float
        F-distribution p-value of the test.
    df_num : int
        Numerator degrees of freedom (number of instruments).
    df_den : int
        Denominator degrees of freedom (T − k_unrestricted).
    residual_ss : float
        Residual sum of squares of the unrestricted regression.

    References
    ----------
    Anderson, T.W. and Rubin, H. (1949). Estimation of the parameters of
        a single equation in a complete system of stochastic equations.
        Annals of Mathematical Statistics 20(1), 46-63.
    """

    stat: float
    p_value: float
    df_num: int
    df_den: int
    residual_ss: float

    def summary(self) -> str:
        return (
            f"Anderson-Rubin test\n"
            f"  F-statistic       : {self.stat:.4f}\n"
            f"  p-value           : {self.p_value:.4f}\n"
            f"  df (num, den)     : ({self.df_num}, {self.df_den})\n"
            f"  residual SS       : {self.residual_ss:.4f}\n"
        )


@dataclass(frozen=True)
class SupTBandResult:
    """Result of :func:`puremacro.inference.supt.supt_band`.

    Sup-t simultaneous confidence band of Montiel Olea & Plagborg-Møller
    (2019): ``lower/upper = center ± crit_value * scale``, where
    ``crit_value`` is calibrated so the whole path is covered jointly with
    probability 1 - alpha.

    Attributes
    ----------
    lower : ndarray, shape (H,)
        Lower simultaneous band, ``center - crit_value * scale``.
    upper : ndarray, shape (H,)
        Upper simultaneous band, ``center + crit_value * scale``.
    center : ndarray, shape (H,)
        Band center: ``theta`` (plugin/bootstrap) or the posterior mean
        of the draws (bayes; also the bootstrap fallback when ``theta``
        is omitted).
    scale : ndarray, shape (H,)
        Per-coordinate standard errors: ``sqrt(diag(Sigma))`` (plugin) or
        the sample std of the draws (bootstrap/bayes).
    crit_value : float
        Sup-t critical value ``c`` — the (1 - alpha) quantile of the max
        absolute studentized deviation. Always >= the pointwise normal
        critical value and <= the Bonferroni one (up to MC error).
    method : str
        'plugin', 'bootstrap', or 'bayes' (Algorithms 1-3 of the paper).
    alpha : float
        Simultaneous two-sided level (0.10 -> 90% band).
    n_draws : int
        Monte Carlo draws used (plugin) or number of bootstrap/posterior
        draws supplied.

    References
    ----------
    Montiel Olea, J.L. and Plagborg-Møller, M. (2019). Simultaneous
        confidence bands: Theory, implementation, and an application to
        SVARs. Journal of Applied Econometrics 34(1), 1-17.
    """

    lower: np.ndarray
    upper: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    crit_value: float
    method: str
    alpha: float
    n_draws: int

    def summary(self) -> str:
        width = self.upper - self.lower
        return (
            f"Sup-t simultaneous band ({100 * (1 - self.alpha):.0f}%)\n"
            f"  method            : {self.method}\n"
            f"  coordinates (H)   : {self.lower.shape[0]}\n"
            f"  crit value (c)    : {self.crit_value:.4f}\n"
            f"  draws used        : {self.n_draws}\n"
            f"  mean band width   : {width.mean():.4f}\n"
        )


@dataclass(frozen=True)
class LewbelIVResult:
    """Result of :func:`puremacro.inference.lewbel_iv.lewbel_iv`.

    Lewbel (2012) 2SLS using instruments constructed from
    heteroskedasticity in exogenous drivers.

    Attributes
    ----------
    beta : ndarray, shape (k_endog + k_exog,)
        2SLS coefficients. Endogenous-regressor coefficients come first,
        in the order given to ``lewbel_iv``; exogenous follow.
    se : ndarray, shape (k_endog + k_exog,)
        Standard errors (homoskedastic 2SLS; HAC variant deferred).
    t : ndarray, shape (k_endog + k_exog,)
        t-statistics, ``beta / se``.
    n_obs : int
        Sample size after dropping rows with NaN inputs.
    n_iv_constructed : int
        Number of Lewbel instruments constructed (k_z × k_endog).
    first_stage_F : float
        First-stage F statistic for the **first** endogenous regressor only
        (joint significance of the constructed IVs). Multi-regressor extension
        is deferred to a future release; with `k_endog > 1`, inspect the
        coefficients on each regressor separately for instrument relevance.
    lewbel_diagnostic : dict
        Breusch-Pagan-style identification test. Keys: ``stat``,
        ``p_value``. Small p indicates the constructed instrument is
        strong; ``p > 0.10`` is treated as weak identification.

    References
    ----------
    Lewbel, A. (2012). Using heteroscedasticity to identify and estimate
        mismeasured and endogenous regressor models. Journal of Business
        and Economic Statistics 30(1), 67-80.
    """

    beta: np.ndarray
    se: np.ndarray
    t: np.ndarray
    n_obs: int
    n_iv_constructed: int
    first_stage_F: float
    lewbel_diagnostic: dict

    def summary(self) -> str:
        diag = self.lewbel_diagnostic
        strength = "STRONG" if diag["p_value"] < 0.10 else "WEAK"
        return (
            f"Lewbel-IV result\n"
            f"  obs (n_obs)              : {self.n_obs}\n"
            f"  Lewbel IVs constructed   : {self.n_iv_constructed}\n"
            f"  first-stage F            : {self.first_stage_F:.2f}\n"
            f"  Lewbel diag p-value      : {diag['p_value']:.4f} [{strength}]\n"
        )
