"""Validation cases for the ``lp`` (local projections) subsystem.

Scope: Jordà (2005) single-country LP with Newey-West HAC SE (``lp_hac``),
LP-IV (Stock-Watson 2018 / Plagborg-Møller-Wolf 2021, ``lp_iv``), and the
two-way fixed-effects panel LP (``panel_lp``).

Independent references — none is circular (each was checked with
``inspect.getsourcefile`` against the puremacro estimator it benchmarks):

* PACKAGE — ``lp_hac`` β_h and HAC SE reproduce, horizon-by-horizon,
  ``statsmodels`` OLS of ``y_{t+h}-y_{t-1}`` on the shock + lag controls with
  ``cov_type="HAC"`` (Bartlett kernel, ``maxlags=h+1``, ``use_correction=False``
  — the raw Newey-West sandwich, no small-sample factor).
* PACKAGE — ``lp_iv`` β_h reproduces ``linearmodels.IV2SLS`` (a genuine,
  independent two-stage-least-squares implementation).
* PACKAGE — ``panel_lp`` β_h reproduces ``linearmodels.PanelOLS`` with
  ``EntityEffects + TimeEffects`` (the two-way within estimator).
* INTERNAL — when the LP-IV instrument equals the regressor the first stage is
  an identity projection, so 2SLS collapses to OLS: ``lp_iv(z=x)`` must equal
  ``lp_hac`` exactly (cross-method identity, no external package).

PACKAGE references are frozen in ``goldens/lp.json``; the live
statsmodels/linearmodels comparison lives only in
``tests/validation/test_reference_drift_lp.py`` (``-m reference``). This module
imports nothing heavier than numpy/pandas at top level; puremacro estimators
and reference packages are imported inside the compute callables to keep the
import path Pyodide-light.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._model import Mechanism, Tol, ValidationCase

# Horizons and lag count shared by every case AND the golden generator, so the
# puremacro computation and its frozen reference are built from identical specs.
LP_HORIZONS: list[int] = [0, 1, 2, 4]
LP_N_LAGS: int = 2


# --------------------------------------------------------------------------- #
# Seeded demo data (local to this module — the shared _fixtures.py is untouched)
# --------------------------------------------------------------------------- #
def lp_ts_demo() -> pd.DataFrame:
    """A stable single-country series for ``lp_hac``.

    ``y`` follows an AR(1) driven by a contemporaneous + one-lag shock ``x``.
    Seeded → byte-identical every call.
    """
    rng = np.random.default_rng(424242)
    T = 200
    x = rng.standard_normal(T)
    y = np.zeros(T)
    eps = rng.standard_normal(T) * 0.7
    for t in range(1, T):
        y[t] = 0.6 * y[t - 1] + 0.8 * x[t] + 0.3 * x[t - 1] + eps[t]
    return pd.DataFrame({"y": y, "x": x})


def lp_iv_demo() -> pd.DataFrame:
    """A stable single-country series for ``lp_iv``.

    ``x`` is endogenous (correlated with the second-stage error through the
    common shock ``nu``); ``z`` is a strong, valid instrument for ``x``.
    Seeded → byte-identical every call.
    """
    rng = np.random.default_rng(20260531)
    T = 220
    z = rng.standard_normal(T)
    nu = rng.standard_normal(T) * 0.5
    x = 0.9 * z + nu
    y = np.zeros(T)
    eps = rng.standard_normal(T) * 0.6
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] + 0.7 * x[t] + 0.2 * x[t - 1] + eps[t] + 0.4 * nu[t]
    return pd.DataFrame({"y": y, "x": x, "z": z})


def lp_panel_demo() -> pd.DataFrame:
    """A stable balanced panel for ``panel_lp``.

    8 entities x 60 quarters with entity and time fixed effects, an AR(1)
    outcome, and a contemporaneous + one-lag shock. Index is
    ``(code, date)`` with a *datetime* time level so the linearmodels
    reference (which rejects period indices) consumes it unchanged.
    Seeded → byte-identical every call.
    """
    rng = np.random.default_rng(7777)
    N, T = 8, 60
    codes = [f"C{i}" for i in range(N)]
    dates = pd.period_range("2000Q1", periods=T, freq="Q").to_timestamp()
    mu = rng.standard_normal(N) * 0.5     # entity fixed effects
    tau = rng.standard_normal(T) * 0.3    # time fixed effects
    rows = []
    for i, c in enumerate(codes):
        y_prev = x_prev = 0.0
        for t in range(T):
            x = rng.standard_normal()
            eps = rng.standard_normal() * 0.6
            y = mu[i] + tau[t] + 0.5 * y_prev + 0.7 * x + 0.2 * x_prev + eps
            rows.append({"code": c, "date": dates[t], "y": y, "x": x})
            y_prev, x_prev = y, x
    return pd.DataFrame(rows).set_index(["code", "date"]).sort_index()


# --------------------------------------------------------------------------- #
# Compute callables (puremacro side)
# --------------------------------------------------------------------------- #
def _lp_hac_beta() -> dict:
    from puremacro.lp import lp_hac

    res = lp_hac(lp_ts_demo(), "y", "x", horizons=LP_HORIZONS, n_lags=LP_N_LAGS)
    return {"beta": res["beta"].to_numpy(dtype=float)}


def _lp_hac_se() -> dict:
    from puremacro.lp import lp_hac

    res = lp_hac(lp_ts_demo(), "y", "x", horizons=LP_HORIZONS, n_lags=LP_N_LAGS)
    return {"se": res["se"].to_numpy(dtype=float)}


def _lp_iv_beta() -> dict:
    from puremacro.lp import lp_iv

    res = lp_iv(lp_iv_demo(), "y", "x", "z", horizons=LP_HORIZONS, n_lags=LP_N_LAGS)
    return {"beta": res["beta"].to_numpy(dtype=float)}


def _panel_lp_beta() -> dict:
    from puremacro.lp import panel_lp

    res = panel_lp(
        lp_panel_demo(), "y", "x", horizons=LP_HORIZONS, n_lags=LP_N_LAGS, controls=()
    )
    return {"beta": res["beta"].to_numpy(dtype=float)}


def _lp_iv_equals_ols_when_instrument_is_regressor() -> dict:
    """LP-IV with z == x collapses to LP-OLS: report β and SE gaps (→ 0)."""
    from puremacro.lp import lp_hac, lp_iv

    df = lp_ts_demo().copy()
    df["z"] = df["x"]  # instrument == endogenous regressor
    ols = lp_hac(df, "y", "x", horizons=LP_HORIZONS, n_lags=LP_N_LAGS)
    iv = lp_iv(df, "y", "x", "z", horizons=LP_HORIZONS, n_lags=LP_N_LAGS)
    return {
        "beta_gap": np.abs(iv["beta"].to_numpy(float) - ols["beta"].to_numpy(float)),
        "se_gap": np.abs(iv["se"].to_numpy(float) - ols["se"].to_numpy(float)),
    }


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #
CASES: list[ValidationCase] = [
    ValidationCase(
        id="lp.jorda_hac_beta_vs_statsmodels",
        subsystem="lp",
        title="Jordà LP coefficients match horizon-by-horizon statsmodels OLS",
        title_es="Los coeficientes de la PL de Jordà coinciden con MCO de statsmodels horizonte a horizonte",
        mechanism=Mechanism.PACKAGE,
        compute=_lp_hac_beta,
        reference="lp:jorda_hac_beta_vs_statsmodels",  # frozen golden
        tol=Tol.TIGHT,
        citation=(
            "Jordà (2005, AER 95(1):161-182), local projections. Reference: "
            "statsmodels 0.14.6 OLS(y_{t+h}-y_{t-1} ~ shock + lags).fit() — the "
            "β on the shock equals lp_hac's point estimate by construction."
        ),
        notes="OLS point estimate is identification-free; matches to machine precision.",
    ),
    ValidationCase(
        id="lp.jorda_hac_se_vs_statsmodels",
        subsystem="lp",
        title="Jordà LP Newey-West HAC standard errors match statsmodels",
        title_es="Los errores estándar HAC de Newey-West de la PL de Jordà coinciden con statsmodels",
        mechanism=Mechanism.PACKAGE,
        compute=_lp_hac_se,
        reference="lp:jorda_hac_se_vs_statsmodels",  # frozen golden
        tol=Tol.TIGHT,  # HAC SE agrees to ~1e-15; NUMERIC (1e-2) would accept a 1%-wrong SE
        citation=(
            "Newey & West (1987, Econometrica 55(3):703-708); Plagborg-Møller & "
            "Wolf (2021, Econometrica 89(2):955-980) bandwidth L=h+1. Reference: "
            "statsmodels 0.14.6 OLS.fit(cov_type='HAC', maxlags=h+1, "
            "use_correction=False) — raw Bartlett sandwich."
        ),
        notes="HAC SE compared at NUMERIC tier (cross-package sandwich arithmetic).",
    ),
    ValidationCase(
        id="lp.iv_beta_vs_linearmodels_iv2sls",
        subsystem="lp",
        title="LP-IV coefficients match linearmodels IV2SLS",
        title_es="Los coeficientes de la PL-VI coinciden con IV2SLS de linearmodels",
        mechanism=Mechanism.PACKAGE,
        compute=_lp_iv_beta,
        reference="lp:iv_beta_vs_linearmodels_iv2sls",  # frozen golden
        tol=Tol.TIGHT,
        citation=(
            "Stock & Watson (2018, EJ 128(610):917-948); Plagborg-Møller & Wolf "
            "(2021, Econometrica 89(2):955-980). Reference: linearmodels 7.0 "
            "IV2SLS(dependent, exog=lags, endog=shock, instruments=z).fit() — an "
            "independent 2SLS implementation."
        ),
        notes="2SLS point estimate matches to machine precision.",
    ),
    ValidationCase(
        id="lp.panel_two_way_fe_beta_vs_linearmodels",
        subsystem="lp",
        title="Panel LP coefficients match linearmodels PanelOLS two-way FE",
        title_es="Los coeficientes de la PL de panel coinciden con PanelOLS de efectos fijos bidireccionales de linearmodels",
        mechanism=Mechanism.PACKAGE,
        compute=_panel_lp_beta,
        reference="lp:panel_two_way_fe_beta_vs_linearmodels",  # frozen golden
        tol=Tol.TIGHT,
        citation=(
            "Two-way within estimator (Wooldridge 2010, Econometric Analysis of "
            "Cross Section and Panel Data, 2e, §10.5). Reference: linearmodels 7.0 "
            "PanelOLS(EntityEffects + TimeEffects).fit() — the β on the shock "
            "equals panel_lp's point estimate."
        ),
        notes="Iterative within-demeaning reproduces linearmodels' FE β to machine precision.",
    ),
    ValidationCase(
        id="lp.iv_reduces_to_ols_when_instrument_is_regressor",
        subsystem="lp",
        title="LP-IV reduces to LP-OLS when the instrument equals the regressor",
        title_es="La PL-VI se reduce a la PL-MCO cuando el instrumento coincide con el regresor",
        mechanism=Mechanism.INTERNAL,
        compute=_lp_iv_equals_ols_when_instrument_is_regressor,
        reference=lambda: {
            "beta_gap": np.zeros(len(LP_HORIZONS)),
            "se_gap": np.zeros(len(LP_HORIZONS)),
        },
        tol=Tol.TIGHT,
        citation=(
            "2SLS identity: with a single instrument equal to the endogenous "
            "regressor the first stage is an identity projection (x̂=x), so the "
            "IV estimator collapses to OLS (Wooldridge 2010, §5.1)."
        ),
        notes="Cross-method identity — no external reference package needed.",
    ),
]
