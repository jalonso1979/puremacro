"""Validation cases for modern cointegration estimators (FM-OLS and DOLS).

Verifies asymptotic consistency on planted cointegrating vectors and
cross-method numerical agreement under second-order endogeneity.
"""
from __future__ import annotations

import numpy as np

from ._model import Mechanism, Tol, ValidationCase


def _cointegration_demo_data(seed: int = 20260904, T: int = 350) -> dict:
    """Generate cointegrated data with planted cointegrating vector beta = 2.0."""
    rng = np.random.default_rng(seed)
    # Innovation with non-zero covariance (second-order endogeneity)
    Sigma_e = np.array([[1.0, 0.4], [0.4, 1.0]])
    L = np.linalg.cholesky(Sigma_e)
    e = rng.standard_normal((T, 2)) @ L.T

    x = np.cumsum(e[:, 0])
    y = 2.0 * x + e[:, 1]
    return {"y": y, "x": x, "true_beta": 2.0}


def _fmols_beta() -> dict:
    from puremacro.cointegration_modern import fm_ols

    d = _cointegration_demo_data()
    res = fm_ols(d["y"], d["x"], lags=2)
    return {"beta": float(res.beta[0])}


def _dols_beta() -> dict:
    from puremacro.cointegration_modern import dols

    d = _cointegration_demo_data()
    res = dols(d["y"], d["x"], leads=2, lags=2)
    return {"beta": float(res.beta[0])}


def _cross_method_diff() -> dict:
    b_fm = _fmols_beta()["beta"]
    b_dols = _dols_beta()["beta"]
    return {"abs_diff": abs(b_fm - b_dols)}


def _dols_mitigates_endogeneity() -> dict:
    from puremacro.cointegration_modern import dols

    d = _cointegration_demo_data(seed=20260904, T=350)
    y, x, true_beta = d["y"], d["x"], d["true_beta"]
    b_ols = float(np.dot(x, y) / np.dot(x, x))
    res_dols = dols(y, x, leads=2, lags=2)
    b_dols = float(res_dols.beta[0])
    bias_dols = abs(b_dols - true_beta)
    bias_ols = abs(b_ols - true_beta)
    return {"dols_bias_strictly_smaller": 1.0 if bias_dols < bias_ols else 0.0}


CASES: list[ValidationCase] = [
    ValidationCase(
        id="cointegration.fmols_planted_beta",
        subsystem="cointegration",
        title="Phillips-Hansen FM-OLS recovers planted cointegrating vector",
        title_es="FM-OLS de Phillips-Hansen recupera el vector de cointegración",
        mechanism=Mechanism.INTERNAL,
        compute=_fmols_beta,
        reference=lambda: {"beta": _cointegration_demo_data()["true_beta"]},
        tol=Tol.NUMERIC,
        citation="Phillips and Hansen (1990, RES 57(1):99-125) Fully Modified OLS.",
    ),
    ValidationCase(
        id="cointegration.dols_planted_beta",
        subsystem="cointegration",
        title="Stock-Watson DOLS recovers planted cointegrating vector",
        title_es="DOLS de Stock-Watson recupera el vector de cointegración",
        mechanism=Mechanism.INTERNAL,
        compute=_dols_beta,
        reference=lambda: {"beta": _cointegration_demo_data()["true_beta"]},
        tol=Tol.NUMERIC,
        citation="Stock and Watson (1993, Econometrica 61(4):783-820) Dynamic OLS.",
    ),
    ValidationCase(
        id="cointegration.fmols_dols_agreement",
        subsystem="cointegration",
        title="FM-OLS and DOLS point estimates agree under endogeneity",
        title_es="Estimaciones de FM-OLS y DOLS coinciden bajo endogeneidad",
        mechanism=Mechanism.INTERNAL,
        compute=_fmols_beta,
        reference=_dols_beta,
        tol=Tol.NUMERIC,
        citation="Cross-estimator asymptotic equivalence of FM-OLS and DOLS (Stock & Watson 1993).",
    ),
    ValidationCase(
        id="cointegration.dols_mitigates_endogeneity_bias",
        subsystem="cointegration",
        title="Stock-Watson DOLS eliminates second-order endogeneity bias vs static OLS",
        title_es="DOLS de Stock-Watson elimina el sesgo de endogeneidad de segundo orden frente a MCO estático",
        mechanism=Mechanism.INTERNAL,
        compute=_dols_mitigates_endogeneity,
        reference=lambda: {"dols_bias_strictly_smaller": 1.0},
        tol=Tol.EXACT,
        citation="Stock & Watson (1993, Econometrica 61(4):783-820); Hayashi (2000, Econometrics §10.3).",
    ),
]

