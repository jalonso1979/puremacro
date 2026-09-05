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


def _sign_restrictions_impact() -> dict:
    from puremacro.var.identify.sign import sign_restriction_svar

    d = var_demo_data()
    res = sign_restriction_svar(d["Y"], p=1, horizon=2, restrictions={0: [+1, -1]}, n_draws=100, seed=42)
    # Impact signs for shock 0 must be +1 on var 0 and -1 on var 1
    s0 = 1.0 if res.irf_median[0, 0, 0] > 0 else 0.0
    s1 = 1.0 if res.irf_median[0, 1, 0] < 0 else 0.0
    return {"signs_match": np.array([s0, s1])}


def _sign_zero_orthogonality() -> dict:
    from puremacro.var.identify.sign_zero import sign_zero
    from puremacro.var.estimate import estimate_var

    d = var_demo_data()
    vr = estimate_var(d["Y"], p=1)
    out = sign_zero(vr.A_list, vr.Sigma, zero_constraints=[(0, 1)], sign_constraints={(0, 0): +1},
                    n_draws=50, rng=np.random.default_rng(42))
    zero_val = float(out.B0[0, 1]) if out.success else 0.0
    return {"zero_res": zero_val}


def _rigobon_hetero_positive_variances() -> dict:
    from puremacro.var.identify.hetero import rigobon_svar

    d = var_demo_data()
    Y = d["Y"]
    regime = np.zeros(len(Y), dtype=int)
    regime[len(Y) // 2:] = 1
    out = rigobon_svar(Y, regime_indicator=regime, p=1, horizon=2, n_boot=0)
    all_pos = float(bool(np.all(out.variance_ratios > 0)))
    return {"all_positive": all_pos}


def _bvar_minnesota_diffuse_limit() -> dict:
    from puremacro.var.bvar import minnesota_posterior
    from puremacro.var.estimate import estimate_var
    import pandas as pd

    d = var_demo_data()
    Y = d["Y"]
    vr = estimate_var(Y, p=1)
    df_Y = pd.DataFrame(Y, columns=["y1", "y2"])
    bvar_res = minnesota_posterior(df_Y, p=1, lambda1=1e5)
    diff = np.max(np.abs(bvar_res["A_list"][0] - vr.A_list[0]))
    return {"max_diff": float(diff)}


def _narrative_sign_restrictions() -> dict:
    from puremacro.var.identify.narrative_sign import narrative_sign_svar

    d = var_demo_data()
    res = narrative_sign_svar(
        d["Y"],
        p=1,
        horizon=2,
        sign_matrix={0: [+1.0, -1.0]},
        restrictions=[(10, 0, +1)],
        n_draws=300,
        seed=42,
    )
    s0 = 1.0 if res.irf_median[0, 0, 0] > 0 else 0.0
    s1 = 1.0 if res.irf_median[0, 1, 0] < 0 else 0.0
    has_accepted = 1.0 if res.n_narrative_accepted > 0 else 0.0
    return {"signs_and_accepted": np.array([s0, s1, has_accepted])}


def _bvar_minnesota_analytical() -> dict:
    from puremacro.var.bvar import minnesota_posterior, _build_minnesota_dummies, _univariate_sigma
    import pandas as pd

    d = var_demo_data()
    Y = d["Y"]
    df_Y = pd.DataFrame(Y, columns=["y1", "y2"])
    res = minnesota_posterior(df_Y, p=1, lambda1=0.2, lambda2=1.0)
    return {"A_post": np.asarray(res["A_list"][0], dtype=float)}


def _bvar_minnesota_analytical_ref() -> dict:
    from puremacro.var.bvar import _build_minnesota_dummies, _univariate_sigma

    d = var_demo_data()
    Y = d["Y"]
    T, n = Y.shape
    p = 1
    sigmas = np.array([_univariate_sigma(Y[:, i], p) for i in range(n)])
    Y_dep, X, Yd, Xd = _build_minnesota_dummies(Y, p, sigmas, 0.2, 1.0, 1.0, 100.0)
    Y_aug = np.vstack([Y_dep, Yd])
    X_aug = np.vstack([X, Xd])
    B_closed = np.linalg.solve(X_aug.T @ X_aug, X_aug.T @ Y_aug)
    A_closed = B_closed[1:].T
    return {"A_post": np.asarray(A_closed, dtype=float)}


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
    ValidationCase(
        id="var.sign_restrictions_impact_signs",
        subsystem="var",
        title="Sign restriction SVAR satisfies prescribed impact signs",
        title_es="SVAR de restricciones de signo satisface signos en impacto",
        mechanism=Mechanism.INTERNAL,
        compute=_sign_restrictions_impact,
        reference=lambda: {"signs_match": np.array([1.0, 1.0])},
        tol=Tol.EXACT,
        citation="Uhlig (2005) sign restrictions definition on contemporaneous impact matrix.",
    ),
    ValidationCase(
        id="var.sign_zero_orthogonality",
        subsystem="var",
        title="Sign-and-zero identification satisfies contemporaneous zero restrictions",
        title_es="Identificación de signo y cero satisface restricciones contemporáneas de cero",
        mechanism=Mechanism.INTERNAL,
        compute=_sign_zero_orthogonality,
        reference=lambda: {"zero_res": 0.0},
        tol=Tol.TIGHT,
        citation="Arias, Rubio-Ramírez and Waggoner (2018) sign-and-zero restrictions.",
    ),
    ValidationCase(
        id="var.rigobon_hetero_positive_variances",
        subsystem="var",
        title="Rigobon heteroskedastic SVAR structural variance ratios are strictly positive",
        title_es="Ratios de varianza estructural de SVAR por heterocedasticidad de Rigobon son positivos",
        mechanism=Mechanism.INTERNAL,
        compute=_rigobon_hetero_positive_variances,
        reference=lambda: {"all_positive": 1.0},
        tol=Tol.EXACT,
        citation="Rigobon (2003) identification through heteroskedasticity.",
    ),
    ValidationCase(
        id="var.bvar_minnesota_diffuse_limit",
        subsystem="var",
        title="Minnesota BVAR posterior mean converges to OLS in diffuse prior limit",
        title_es="Media posterior de BVAR Minnesota converge a MCO bajo a priori difusa",
        mechanism=Mechanism.INTERNAL,
        compute=_bvar_minnesota_diffuse_limit,
        reference=lambda: {"max_diff": 0.0},
        tol=Tol.TIGHT,
        citation="Banbura, Giannone and Reichlin (2010) Minnesota prior dummy observations limit.",
    ),
    ValidationCase(
        id="var.narrative_sign_restrictions",
        subsystem="var",
        title="Narrative sign restrictions SVAR satisfies historical shock and impact signs",
        title_es="SVAR con restricciones de signo narrativas satisface signos de shocks históricos e impacto",
        mechanism=Mechanism.INTERNAL,
        compute=_narrative_sign_restrictions,
        reference=lambda: {"signs_and_accepted": np.array([1.0, 1.0, 1.0])},
        tol=Tol.EXACT,
        citation="Antolín-Díaz and Rubio-Ramírez (2018, AER 108(10):2802-2829).",
    ),
    ValidationCase(
        id="var.bvar_minnesota_analytical_posterior",
        subsystem="var",
        title="Minnesota BVAR posterior mean matches analytical augmented dummy OLS",
        title_es="Media posterior de BVAR Minnesota coincide con MCO analítico de datos aumentados",
        mechanism=Mechanism.INTERNAL,
        compute=_bvar_minnesota_analytical,
        reference=_bvar_minnesota_analytical_ref,
        tol=Tol.TIGHT,
        citation="Banbura, Giannone and Reichlin (2010, JAE 25(1):71-92) conjugate Normal-Inverse-Wishart.",
    ),
]

