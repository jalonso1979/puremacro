"""Validation cases for the ``var`` subsystem.

Worked-example scope: two INTERNAL identities + one PACKAGE cross-check
(Cholesky IRF vs statsmodels). Pure deps only — the PACKAGE reference is the
frozen golden in ``goldens/var.json`` (live statsmodels is touched only by the
``reference``-marked drift-guard in ``tests/validation/test_reference_drift.py``).
"""
from __future__ import annotations

import numpy as np

from ._fixtures import var_demo_data
from ._model import Mechanism, Tol, ValidationCase


def _fevd_row_sums() -> dict:
    from puremacro.var import fevd, fit_var

    d = var_demo_data()
    vr = fit_var(d["Y"], d["p"])
    B0 = np.linalg.cholesky(vr.Sigma)
    fe = np.asarray(fevd(vr.A_list, B0, d["horizon"]), dtype=float)  # (H+1, k, k)
    return {"row_sums": fe.sum(axis=-1).ravel()}  # variance shares sum to 1 over shocks


def _stability_agreement() -> dict:
    from puremacro.var import companion, fit_var, is_stable

    d = var_demo_data()
    vr = fit_var(d["Y"], d["p"])
    max_eig = float(np.max(np.abs(np.linalg.eigvals(np.asarray(companion(vr.A_list))))))
    agree = float(bool(is_stable(vr.A_list)) == (max_eig < 1.0))
    return {"agree": agree}


def _cholesky_irf_point() -> dict:
    from puremacro.var.identify import cholesky

    d = var_demo_data()
    sol = cholesky(d["Y"], p=d["p"], horizon=d["horizon"], n_boot=10, seed=0)
    return {"irf": np.asarray(sol.irf_point, dtype=float)}


CASES: list[ValidationCase] = [
    ValidationCase(
        id="var.fevd_sums_to_one",
        subsystem="var",
        title="FEVD variance shares sum to 1",
        title_es="Las cuotas de varianza de la FEVD suman 1",
        mechanism=Mechanism.INTERNAL,
        compute=_fevd_row_sums,
        reference=lambda: {"row_sums": np.ones(2 * (var_demo_data()["horizon"] + 1))},
        tol=Tol.EXACT,
        citation="Forecast-error variance decomposition identity (Lütkepohl 2005, §2.3.3).",
    ),
    ValidationCase(
        id="var.stability_iff_spectral_radius",
        subsystem="var",
        title="is_stable() agrees with companion spectral radius < 1",
        title_es="is_stable() coincide con radio espectral del companion < 1",
        mechanism=Mechanism.INTERNAL,
        compute=_stability_agreement,
        reference=lambda: {"agree": 1.0},
        tol=Tol.EXACT,
        citation="VAR stationarity ⟺ all companion eigenvalues inside unit circle (Lütkepohl 2005, §2.1).",
    ),
    ValidationCase(
        id="var.cholesky_irf_vs_statsmodels",
        subsystem="var",
        title="Cholesky IRF matches statsmodels orth_irfs",
        title_es="La FIR de Cholesky coincide con orth_irfs de statsmodels",
        mechanism=Mechanism.PACKAGE,
        compute=_cholesky_irf_point,
        reference="var:cholesky_irf_vs_statsmodels",  # frozen golden
        tol=Tol.TIGHT,
        citation="statsmodels 0.14.6 VAR(df).fit(p).irf(H).orth_irfs (recursive identification).",
    ),
]
