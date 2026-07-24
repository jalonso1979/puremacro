"""Validation cases for the ``tests.unit_root`` GLS-detrended unit-root tests.

References are ANALYTICAL closed-form identities of the ERS (1996) GLS local
detrending step — the deterministic backbone that DF-GLS and Ng-Perron share —
plus an INTERNAL cross-statistic identity of the Ng-Perron M-tests. No PACKAGE
goldens: the detrending recovers a *known* deterministic input exactly, so the
honest yardstick is the planted truth itself.

All imports of ``puremacro.tests.unit_root`` live inside the callables to keep
module import pyodide-light.
"""
from __future__ import annotations

import numpy as np

from ._model import Mechanism, Tol, ValidationCase


# GLS detrending is linear in the deterministic regressors, so quasi-differencing
# a purely deterministic series and regressing back recovers its coefficients
# EXACTLY (no noise, no local-to-unity bias). This is the cleanest planted-truth
# check of the ERS detrending both new tests depend on.
_A, _B = 2.0, 0.5   # intercept, slope of the planted linear trend
_T = 200


def _dfgls_trend_coef() -> dict:
    from puremacro.tests.unit_root import dfgls_test

    y = _A + _B * np.arange(1, _T + 1, dtype=float)
    return {"gls_coef": dfgls_test(y, regression="ct")["gls_coef"]}


def _dfgls_const_mean() -> dict:
    """The constant-case GLS demeaning recovers the mean of a level series.

    Uses the ERS detrending step directly (a constant series drives the *test*'s
    downstream DF regression degenerate, but the detrending identity itself is
    exact and is what we assert here).
    """
    from puremacro.tests.unit_root import _gls_detrend

    y = np.full(_T, 3.25)
    _, beta = _gls_detrend(y, "c")
    return {"gls_mean": float(beta[0])}


def _ng_perron_stats() -> dict:
    from puremacro.tests.unit_root import ng_perron_test

    rng = np.random.default_rng(20260724)
    y = np.cumsum(rng.standard_normal(_T))  # any series; identity is structural
    return ng_perron_test(y, regression="ct")["statistics"]


def _ng_perron_mzt() -> dict:
    """LHS of the identity: the reported MZt statistic."""
    return {"mzt": _ng_perron_stats()["MZt"]}


def _ng_perron_mza_times_msb() -> dict:
    """RHS of the identity: MZa * MSB, recomputed independently."""
    s = _ng_perron_stats()
    return {"mzt": s["MZa"] * s["MSB"]}


CASES: list[ValidationCase] = [
    ValidationCase(
        id="unit_root.dfgls_gls_detrend_recovers_trend",
        subsystem="unit_root",
        title="ERS GLS detrending recovers a deterministic trend's coefficients exactly",
        title_es=(
            "El desentendencia GLS de ERS recupera exactamente los coeficientes "
            "de una tendencia deterministica"
        ),
        mechanism=Mechanism.ANALYTICAL,
        compute=_dfgls_trend_coef,
        reference=lambda: {"gls_coef": [_A, _B]},
        tol=Tol.TIGHT,
        citation=(
            "Elliott, Rothenberg & Stock (1996, Econometrica 64:813-836): GLS "
            "quasi-differencing is linear in the deterministic regressors, so a "
            "noiseless y=a+bt is detrended back to (a,b) exactly."
        ),
    ),
    ValidationCase(
        id="unit_root.dfgls_gls_mean_of_constant",
        subsystem="unit_root",
        title="ERS GLS demeaning of a constant series returns the constant",
        title_es=(
            "La eliminacion de media GLS de ERS de una serie constante devuelve "
            "la constante"
        ),
        mechanism=Mechanism.ANALYTICAL,
        compute=_dfgls_const_mean,
        reference=lambda: {"gls_mean": 3.25},
        tol=Tol.TIGHT,
        citation="Elliott, Rothenberg & Stock (1996, Econometrica 64:813-836), c-bar=-7 case.",
    ),
    ValidationCase(
        id="unit_root.ng_perron_mzt_equals_mza_times_msb",
        subsystem="unit_root",
        title="Ng-Perron identity: MZt = MZa * MSB",
        title_es="Identidad de Ng-Perron: MZt = MZa * MSB",
        mechanism=Mechanism.INTERNAL,
        compute=_ng_perron_mzt,
        reference=_ng_perron_mza_times_msb,
        tol=Tol.EXACT,
        citation="Ng & Perron (2001, Econometrica 69:1519-1554), Table 1 M-test definitions.",
    ),
]
