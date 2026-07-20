"""Validation cases for the ``inference`` subsystem (robust SEs + weak-IV).

Scope (seven cases across the mechanisms that admit a SOUND, INDEPENDENT check):

* PACKAGE   ``ols_hac`` / ``newey_west`` HAC standard errors vs the
            statsmodels ``OLS(...).fit(cov_type="HAC")`` Bartlett-kernel SEs on
            byte-identical data. statsmodels is an independent implementation of
            the same Newey-West (1987) sandwich; the reference is the frozen
            golden in ``goldens/inference.json`` and the live statsmodels call
            lives only in the ``reference``-marked drift-guard.
* PUBLISHED ``stock_yogo_cv`` returns the Stock-Yogo (2005, Table 5.2) 2SLS
            relative-bias critical values for the single-endogenous-regressor
            case — asserted EXACTLY against the published constants.
* ANALYTICAL ``cragg_donald_f`` collapses to the textbook first-stage F in the
            just-identified, no-controls design; and ``newey_west_se`` with
            bandwidth 0 collapses to the White (HC0) heteroskedasticity-robust
            SE. Both references are closed forms computed live with numpy.
            ``supt_band`` (plugin): under an identity correlation matrix the
            sup-t critical value has the closed form
            Phi^{-1}((1 + (1-alpha)^{1/H}) / 2), since max_h |Z_h| of iid
            standard normals has CDF (2*Phi(c) - 1)^H; the Monte Carlo
            quantile must match it (reference evaluated live via scipy's
            normal ppf — scipy is a runtime dependency).
* INTERNAL  ``driscoll_kraay`` (panel HAC) reduces to the Newey-West "meat"
            matrix when each period holds a single cross-sectional unit, since
            the time-aggregation step is then a no-op.

``kleibergen_paap_f`` is deliberately omitted: its returned summary statistic
is a non-standard ``vec(Pi)' V^{-1} vec(Pi)/(l k)`` quadratic form (orders of
magnitude away from the textbook rk Wald F) for which no independent reference
reproduces the same number — see ``skipped`` in the task report.

Pure deps only at import time: this module imports numpy + stdlib. Every
puremacro / statsmodels touch is *inside* a compute callable.
"""
from __future__ import annotations

import numpy as np

from ._model import Mechanism, Tol, ValidationCase


# --------------------------------------------------------------------------- #
# Seeded demo data (kept local; the shared _fixtures.py is not edited).        #
# Identical bytes feed both the puremacro computation and the golden/drift    #
# generators, so PACKAGE comparisons are like-for-like.                        #
# --------------------------------------------------------------------------- #
def hac_demo_data() -> dict:
    """OLS design with AR(1) errors, where HAC SEs genuinely differ from OLS.

    Returns a dict with ``X`` (T,3) including an intercept, ``y`` (T,), the
    chosen Bartlett bandwidth ``lags``, and the OLS ``resid`` (so the standalone
    ``newey_west_se(X, resid, bw)`` and ``ols_hac(y, X, lags)`` see the same
    residuals).
    """
    rng = np.random.default_rng(20260531)
    T = 200
    x1 = rng.standard_normal(T)
    x2 = rng.standard_normal(T)
    X = np.column_stack([np.ones(T), x1, x2])
    # AR(1) errors so serial correlation is present and HAC bites.
    u = np.zeros(T)
    e = rng.standard_normal(T)
    for t in range(1, T):
        u[t] = 0.6 * u[t - 1] + e[t]
    y = 1.0 + 0.5 * x1 - 0.3 * x2 + u
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    return {"X": X, "y": y, "resid": resid, "lags": 4}


def hac_white_demo_data() -> dict:
    """Heteroskedastic (no serial correlation) OLS design for the bw=0 = HC0 case."""
    rng = np.random.default_rng(990531)
    T = 150
    x1 = rng.standard_normal(T)
    x2 = rng.standard_normal(T)
    X = np.column_stack([np.ones(T), x1, x2])
    # Multiplicative heteroskedasticity in x1.
    u = rng.standard_normal(T) * (1.0 + np.abs(x1))
    y = 1.0 + 2.0 * x1 - 1.0 * x2 + u
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    return {"X": X, "y": y, "resid": resid}


def cd_just_identified_data() -> dict:
    """Just-identified IV design: 1 endogenous regressor, 1 instrument, no controls.

    Matches the Cragg-Donald implementation's assumption of no included exogenous
    regressors, so the statistic must equal the textbook first-stage F.
    """
    rng = np.random.default_rng(11)
    T = 400
    z = rng.standard_normal((T, 1))
    X = z * 0.7 + 0.5 * rng.standard_normal((T, 1))   # strong first stage
    Y = X[:, 0] + rng.standard_normal(T)
    return {"Y": Y, "X": X, "Z": z}


def dk_panel_score_data() -> dict:
    """Panel score (T,k) with a single cross-sectional unit per time period.

    Under one-unit-per-period the Driscoll-Kraay time aggregation is the
    identity, so the DK 'meat' must coincide with the Newey-West 'meat'.
    """
    rng = np.random.default_rng(3)
    T, k = 60, 2
    score = rng.standard_normal((T, k))
    time_keys = np.arange(T)   # exactly one observation per period
    return {"score": score, "time_keys": time_keys, "lags": 4}


# --------------------------------------------------------------------------- #
# Compute callables (puremacro imports live here, not at module top).          #
# --------------------------------------------------------------------------- #
def _ols_hac_se() -> dict:
    from puremacro.inference._ols_helpers import ols_hac

    d = hac_demo_data()
    res = ols_hac(d["y"], d["X"], lags=d["lags"])
    return {"se": np.asarray(res["se"], dtype=float)}


def _newey_west_se_alias() -> dict:
    # Top-level alias `puremacro.inference.newey_west` == hac.newey_west_se.
    from puremacro.inference import newey_west

    d = hac_demo_data()
    se = newey_west(d["X"], d["resid"], bw=d["lags"])
    return {"se": np.asarray(se, dtype=float)}


def _stock_yogo_published() -> dict:
    from puremacro.inference.over_id import stock_yogo_cv

    # A spread of (n_instruments, max_bias_pct) keys spanning the table.
    keys = [(3, 5), (3, 10), (3, 20), (3, 30), (4, 10), (5, 10), (6, 10)]
    return {"cv": np.array([stock_yogo_cv(l, b) for (l, b) in keys], dtype=float)}


def _cragg_donald_just_identified() -> dict:
    from puremacro.inference.weak_iv import cragg_donald_f

    d = cd_just_identified_data()
    return {"F": float(cragg_donald_f(d["Y"], d["X"], d["Z"]))}


def _cragg_donald_reference() -> dict:
    """Closed-form first-stage F for the just-identified, no-constant design.

    Regress X on the single instrument z (no intercept); the concentration
    parameter X̂'X̂ divided by the first-stage error variance σ̂² (with n-l dof,
    l=1 here) is exactly the Cragg-Donald minimum-eigenvalue statistic when
    k=1.
    """
    d = cd_just_identified_data()
    X = d["X"][:, 0]
    z = d["Z"][:, 0]
    T = X.shape[0]
    b = (z @ X) / (z @ z)
    x_hat = z * b
    x_til = X - x_hat
    sigma2 = float(np.sum(x_til**2) / (T - 1))   # n - l, with l = 1
    F = float(np.sum(x_hat**2) / sigma2)
    return {"F": F}


def _newey_west_bw0() -> dict:
    from puremacro.inference.hac import newey_west_se

    d = hac_white_demo_data()
    se = newey_west_se(d["X"], d["resid"], bw=0)
    return {"se": np.asarray(se, dtype=float)}


def _white_hc0_reference() -> dict:
    """White (1980) HC0 robust SE: sqrt(diag((X'X)^-1 X'diag(u^2)X (X'X)^-1))."""
    d = hac_white_demo_data()
    X, u = d["X"], d["resid"]
    XtX_inv = np.linalg.inv(X.T @ X)
    meat = (X * u[:, None]).T @ (X * u[:, None])
    vcov = XtX_inv @ meat @ XtX_inv
    return {"se": np.sqrt(np.diag(vcov))}


def _driscoll_kraay_meat() -> dict:
    from puremacro.inference.dk import driscoll_kraay

    d = dk_panel_score_data()
    S = np.asarray(driscoll_kraay(d["score"], d["time_keys"], d["lags"]), dtype=float)
    return {"meat": S.ravel()}


def _newey_west_meat_reference() -> dict:
    """Newey-West (1987) Bartlett 'meat' on the un-aggregated score series.

    With one cross-sectional unit per period the Driscoll-Kraay aggregation is a
    no-op, so this must equal the DK meat byte-for-byte.
    """
    d = dk_panel_score_data()
    score, lags = d["score"], d["lags"]
    S = score.T @ score
    for ell in range(1, lags + 1):
        w = 1.0 - ell / (lags + 1.0)
        G = score[ell:].T @ score[:-ell]
        S += w * (G + G.T)
    return {"meat": S.ravel()}


_SUPT_H = 6
_SUPT_ALPHA = 0.10


def _supt_plugin_iid_mc() -> dict:
    """Sup-t plugin critical value via Monte Carlo on an identity correlation."""
    from puremacro.inference.supt import supt_band

    res = supt_band(np.zeros(_SUPT_H), np.eye(_SUPT_H), alpha=_SUPT_ALPHA,
                    n_mc=200_000, rng=20260718)
    return {"crit": float(res.crit_value)}


def _supt_iid_closed_form() -> dict:
    """Closed-form sup-t critical value for H iid standard normals.

    P(max_h |Z_h| <= c) = (2*Phi(c) - 1)^H, so the (1-alpha) quantile is
    c* = Phi^{-1}((1 + (1-alpha)^{1/H}) / 2). scipy (a runtime dependency)
    supplies Phi^{-1}; the formula itself is exact.
    """
    from scipy.stats import norm

    c = norm.ppf(0.5 * (1.0 + (1.0 - _SUPT_ALPHA) ** (1.0 / _SUPT_H)))
    return {"crit": float(c)}


# --------------------------------------------------------------------------- #
CASES: list[ValidationCase] = [
    ValidationCase(
        id="inference.ols_hac_vs_statsmodels",
        subsystem="inference",
        title="OLS Newey-West HAC standard errors match statsmodels",
        title_es="Los errores estándar HAC de Newey-West de MCO coinciden con statsmodels",
        mechanism=Mechanism.PACKAGE,
        compute=_ols_hac_se,
        reference="inference:ols_hac_vs_statsmodels",  # frozen golden
        tol=Tol.TIGHT,
        citation=(
            "statsmodels 0.14.6 OLS(y, X).fit(cov_type='HAC', "
            "cov_kwds={'maxlags': L}).bse (Newey & West 1987, Bartlett kernel)."
        ),
        notes="Independent sandwich implementation; same Bartlett bandwidth L.",
    ),
    ValidationCase(
        id="inference.newey_west_vs_statsmodels",
        subsystem="inference",
        title="Standalone newey_west() SEs match statsmodels HAC",
        title_es="Los EE de newey_west() (función directa) coinciden con HAC de statsmodels",
        mechanism=Mechanism.PACKAGE,
        compute=_newey_west_se_alias,
        reference="inference:newey_west_vs_statsmodels",  # frozen golden
        tol=Tol.TIGHT,
        citation=(
            "statsmodels 0.14.6 OLS(y, X).fit(cov_type='HAC', "
            "cov_kwds={'maxlags': L}).bse on identical OLS residuals."
        ),
        notes="Validates puremacro.inference.newey_west (alias of hac.newey_west_se).",
    ),
    ValidationCase(
        id="inference.stock_yogo_published_cv",
        subsystem="inference",
        title="Stock-Yogo critical values match the published table",
        title_es="Los valores críticos de Stock-Yogo coinciden con la tabla publicada",
        mechanism=Mechanism.PUBLISHED,
        compute=_stock_yogo_published,
        # Published constants (Stock & Yogo 2005, Table 5.2; single endogenous
        # regressor, 2SLS, maximal relative bias). Keys, in order:
        # (3,5),(3,10),(3,20),(3,30),(4,10),(5,10),(6,10).
        reference=lambda: {
            "cv": np.array([13.91, 9.08, 6.46, 5.39, 10.27, 10.83, 11.12], dtype=float)
        },
        tol=Tol.EXACT,
        citation=(
            "Stock, J.H. & Yogo, M. (2005), 'Testing for Weak Instruments in "
            "Linear IV Regression', in Andrews & Stock (eds.), Cambridge UP, "
            "Table 5.2 (2SLS relative bias, one endogenous regressor)."
        ),
    ),
    ValidationCase(
        id="inference.cragg_donald_just_identified_F",
        subsystem="inference",
        title="Cragg-Donald F equals the first-stage F (just-identified case)",
        title_es="El F de Cragg-Donald iguala el F de primera etapa (caso exactamente identificado)",
        mechanism=Mechanism.ANALYTICAL,
        compute=_cragg_donald_just_identified,
        reference=_cragg_donald_reference,
        tol=Tol.TIGHT,
        citation=(
            "Cragg, J.G. & Donald, S.G. (1993), 'Testing Identifiability and "
            "Specification in Instrumental Variable Models', Econometric Theory "
            "9(2), 222-240: with one endogenous regressor the minimum-eigenvalue "
            "statistic is the first-stage concentration F."
        ),
    ),
    ValidationCase(
        id="inference.newey_west_bw0_equals_hc0",
        subsystem="inference",
        title="Newey-West at bandwidth 0 equals White (HC0) robust SE",
        title_es="Newey-West con ancho de banda 0 iguala el EE robusto de White (HC0)",
        mechanism=Mechanism.ANALYTICAL,
        compute=_newey_west_bw0,
        reference=_white_hc0_reference,
        tol=Tol.TIGHT,
        citation=(
            "White, H. (1980), 'A Heteroskedasticity-Consistent Covariance "
            "Matrix Estimator and a Direct Test for Heteroskedasticity', "
            "Econometrica 48(4), 817-838: the Bartlett HAC with bandwidth 0 "
            "drops all autocovariance terms, leaving the HC0 sandwich."
        ),
    ),
    ValidationCase(
        id="inference.driscoll_kraay_reduces_to_newey_west",
        subsystem="inference",
        title="Driscoll-Kraay meat equals Newey-West meat (one unit per period)",
        title_es="La matriz central de Driscoll-Kraay iguala la de Newey-West (una unidad por periodo)",
        mechanism=Mechanism.INTERNAL,
        compute=_driscoll_kraay_meat,
        reference=_newey_west_meat_reference,
        tol=Tol.EXACT,
        citation=(
            "Driscoll, J.C. & Kraay, A.C. (1998), 'Consistent Covariance Matrix "
            "Estimation with Spatially Dependent Panel Data', RESTAT 80(4), "
            "549-560: the estimator is a cross-sectionally aggregated Newey-West, "
            "so it collapses to Newey & West (1987) when N_t = 1 for all t."
        ),
    ),
    ValidationCase(
        id="inference.supt_plugin_iid_closed_form",
        subsystem="inference",
        title="Sup-t plugin critical value matches the iid closed form",
        title_es="El valor crítico sup-t (plugin) coincide con la forma cerrada iid",
        mechanism=Mechanism.ANALYTICAL,
        compute=_supt_plugin_iid_mc,
        reference=_supt_iid_closed_form,
        tol=Tol.NUMERIC,
        citation=(
            "Montiel Olea, J.L. & Plagborg-Møller, M. (2019), 'Simultaneous "
            "confidence bands: Theory, implementation, and an application to "
            "SVARs', Journal of Applied Econometrics 34(1), 1-17: with an "
            "identity correlation matrix the sup-t critical value is "
            "Phi^{-1}((1 + (1-alpha)^{1/H})/2), the (1-alpha) quantile of the "
            "max of H iid |N(0,1)| variables."
        ),
        notes=(
            "MC quantile with 200k seeded draws vs the exact quantile; MC "
            "standard error ~0.003 well inside the NUMERIC tier (~0.024 here)."
        ),
    ),
]
