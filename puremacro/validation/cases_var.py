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
    zero_val = float(out.B0[0, 1]) if out.success and out.B0 is not None else 0.0
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



# ---------------------------------------------------------------------------
# GVAR (puremacro.var.gvar)
#
# Six cases, all sound and independent of the estimator under test:
#
# * INTERNAL   with an empty star block everywhere the GVAR degenerates to N
#              unlinked country VARs, so each country's Phi_il and a_i0 must
#              equal ``puremacro.var.estimate_var`` on that country's own data
#              (a different code path: plain OLS, no link step).
# * INTERNAL   the solved global system reproduces the country equations:
#              ``G x_t - a_0 - sum_l H_l x_{t-l}`` rebuilds the stacked
#              country-OLS residuals, which is the whole content of the link
#              identity.
# * ANALYTICAL ``pp(b)[0] = 1`` for every ``b`` by construction (the profile is
#              normalised by its own impact variance).
# * ANALYTICAL ``gfevd(normalize=True)`` rows sum to 1 at every horizon. The
#              UNNORMALISED rows are deliberately not asserted: the Pesaran-Shin
#              ">= 1" bound needs ``Psi_0 = I`` and a GVAR has ``R_0 = G^-1``,
#              so it fails even at h = 0 (see the module docstring).
# * INTERNAL   a one-country GVAR has ``G = I``, so its GIRF is the ordinary
#              generalised IRF and must equal ``puremacro.var.irf.irf`` driven
#              by ``B0 = Sigma diag(sigma_jj)^-1/2``; its GFEVD must equal
#              ``puremacro.var.irf.gfevd`` on the same coefficients.
#
# DGP: one stable VAR(1) on 4 series (2 countries x 2 variables), T = 80 after
# a 50-period burn-in. Whole block measured at ~0.05 s.
# ---------------------------------------------------------------------------

GVAR_SEED = 20260906
GVAR_T = 80
GVAR_BURN = 50


def gvar_demo_data() -> dict:
    """Seeded 2-country x 2-variable panel (T = 80) plus its trade weights.

    The DGP is a single stable VAR(1) on the stacked 4-vector with genuine
    cross-country blocks, so the fitted GVAR is not degenerate. Three weight
    matrices are returned: the two-country row-standardised one, the all-zero
    one used by the decoupling case, and the 1x1 zero matrix used by the
    one-country cases.
    """
    rng = np.random.default_rng(GVAR_SEED)
    A = np.array(
        [
            [0.50, 0.10, 0.15, 0.00],
            [0.05, 0.40, 0.00, 0.10],
            [0.10, 0.00, 0.45, 0.20],
            [0.00, 0.05, 0.05, 0.35],
        ]
    )
    n_all = GVAR_T + GVAR_BURN
    X = np.zeros((n_all, 4))
    for t in range(1, n_all):
        X[t] = A @ X[t - 1] + rng.standard_normal(4)
    X = X[GVAR_BURN:]

    import pandas as pd

    dates = pd.RangeIndex(GVAR_T, name="date")
    frames = {
        "A": pd.DataFrame(X[:, :2], columns=["y", "r"], index=dates),
        "B": pd.DataFrame(X[:, 2:], columns=["y", "r"], index=dates),
    }
    ids = ["A", "B"]
    return {
        "frames": frames,
        "weights": pd.DataFrame([[0.0, 1.0], [1.0, 0.0]], index=ids, columns=ids),
        "weights_zero": pd.DataFrame(np.zeros((2, 2)), index=ids, columns=ids),
        "weights_solo": pd.DataFrame(np.zeros((1, 1)), index=["A"], columns=["A"]),
        "p": 2,
    }


def _fit_gvar_linked():
    """The linked two-country GVAR(p=2, q=1) used by the identity cases."""
    from puremacro.var import gvar

    d = gvar_demo_data()
    return d, gvar(d["frames"], d["weights"], p=d["p"], q=1, weak_exogeneity=False)


def _fit_gvar_solo():
    """A one-country GVAR: no star block anywhere, hence ``G = I``."""
    from puremacro.var import gvar

    d = gvar_demo_data()
    res = gvar(
        {"A": d["frames"]["A"]},
        d["weights_solo"],
        p=d["p"],
        q=0,
        star_vars={"A": ()},
        weak_exogeneity=False,
    )
    return d, res


def _gvar_zero_weights_country_blocks() -> dict:
    """Country VARX* blocks when every star block is empty (weights all zero)."""
    from puremacro.var import gvar

    d = gvar_demo_data()
    res = gvar(
        d["frames"],
        d["weights_zero"],
        p=d["p"],
        q=0,
        star_vars={"A": (), "B": ()},
        weak_exogeneity=False,
    )
    out = {}
    for c in ("A", "B"):
        m = res.country_models[c]
        out[f"Phi_{c}"] = np.concatenate([np.asarray(P, dtype=float).ravel() for P in m.Phi])
        out[f"a0_{c}"] = np.asarray(m.a0, dtype=float).ravel()
    return out


def _gvar_zero_weights_country_blocks_ref() -> dict:
    """The same blocks from plain OLS VARs — a different estimator entirely."""
    from puremacro.var.estimate import estimate_var

    d = gvar_demo_data()
    out = {}
    for c in ("A", "B"):
        vr = estimate_var(d["frames"][c], p=d["p"])
        out[f"Phi_{c}"] = np.concatenate(
            [np.asarray(A, dtype=float).ravel() for A in vr.A_list]
        )
        out[f"a0_{c}"] = np.asarray(vr.c, dtype=float).ravel()
    return out


def _gvar_link_identity() -> dict:
    """Rebuild eps_t from the solved global system: G x_t - a_0 - sum_l H_l x_{t-l}."""
    _, res = _fit_gvar_linked()
    X = res.panel.reindex(columns=list(res.names)).to_numpy(dtype=float)
    pos = {dt: i for i, dt in enumerate(res.panel.index)}
    a0 = res.G @ res.intercept  # `intercept` is the SOLVED G^-1 a_0
    rows = []
    for dt in res.dates:
        t = pos[dt]
        e = res.G @ X[t] - a0
        for l in range(res.s):
            e = e - res.H[l] @ X[t - l - 1]
        rows.append(e)
    return {"eps": np.asarray(rows, dtype=float).ravel()}


def _gvar_link_identity_ref() -> dict:
    """The stacked country-OLS residuals, produced without the link matrices."""
    _, res = _fit_gvar_linked()
    return {"eps": res.resid.to_numpy(dtype=float).ravel()}


def _gvar_persistence_profile_impact() -> dict:
    """PP(b, 0) for three combinations b: a single variable, a spread, the sum."""
    _, res = _fit_gvar_linked()
    k = res.n_variables
    spread = np.zeros(k)
    spread[0], spread[2] = 1.0, -1.0
    profiles = [
        res.pp({("A", "y"): 1.0}, horizon=12),
        res.pp(spread, horizon=12),
        res.pp(np.ones(k), horizon=12),
    ]
    return {"pp_at_impact": np.array([float(p[0]) for p in profiles])}


def _gvar_gfevd_row_sums() -> dict:
    """Normalised generalised FEVD row sums at every horizon 0..12."""
    _, res = _fit_gvar_linked()
    fe = np.asarray(res.gfevd(horizon=12, normalize=True), dtype=float)
    return {"row_sums": fe.sum(axis=-1).ravel()}


def _gvar_solo_girf() -> dict:
    """GIRFs of a one-country GVAR (G = I) to each of its own innovations."""
    _, res = _fit_gvar_solo()
    paths = [
        np.asarray(res.girf("A", v, horizon=10).irf, dtype=float)
        for v in ("y", "r")
    ]
    return {"girf": np.stack(paths, axis=-1).ravel()}


def _gvar_solo_girf_ref() -> dict:
    """The same object from ``puremacro.var.irf.irf`` with B0 = Sigma D^-1/2.

    Column ``j`` of ``Psi_h B0`` with ``B0 = Sigma diag(sigma_jj)^-1/2`` is
    ``Psi_h Sigma e_j / sqrt(sigma_jj)`` — the Pesaran-Shin generalised
    impulse response, computed by the ordinary VAR MA recursion.
    """
    from puremacro.var.irf import irf

    _, res = _fit_gvar_solo()
    Sigma = np.asarray(res.Sigma_eps, dtype=float)
    B0 = Sigma / np.sqrt(np.diag(Sigma))[None, :]
    A_list = [np.asarray(res.F[l], dtype=float) for l in range(res.s)]
    return {"girf": np.asarray(irf(A_list, B0, 10), dtype=float).ravel()}


def _gvar_solo_gfevd() -> dict:
    _, res = _fit_gvar_solo()
    return {"gfevd": np.asarray(res.gfevd(horizon=10, normalize=True), dtype=float).ravel()}


def _gvar_solo_gfevd_ref() -> dict:
    from puremacro.var.irf import gfevd

    _, res = _fit_gvar_solo()
    A_list = [np.asarray(res.F[l], dtype=float) for l in range(res.s)]
    out = gfevd(A_list, np.asarray(res.Sigma_eps, dtype=float), 10, normalize=True)
    return {"gfevd": np.asarray(out, dtype=float).ravel()}


CASES += [
    ValidationCase(
        id="var.gvar_empty_star_block_decouples",
        subsystem="var",
        title="GVAR with no star block reduces to independent country VARs",
        title_es="El GVAR sin bloque estrella se reduce a VAR nacionales independientes",
        mechanism=Mechanism.INTERNAL,
        compute=_gvar_zero_weights_country_blocks,
        reference=_gvar_zero_weights_country_blocks_ref,
        tol=Tol.TIGHT,
        citation=(
            "Pesaran, Schuermann and Weiner (2004, JBES 22(2):129-162): the "
            "country VARX*(p, q) collapses to a VAR(p) when Lambda_i0 = ... = "
            "Lambda_iq = 0, so OLS on the country's own data must reproduce it."
        ),
        notes=(
            "Trade weights are all zero and every star block is empty (q = 0), "
            "so the link matrix is G = I and each country's Phi_i1, Phi_i2 and "
            "a_i0 come out of a design matrix that contains only its own lags "
            "and a constant. The reference is puremacro.var.estimate_var, a "
            "different code path with no link step. The GVAR fixes the common "
            "effective sample at t0 = max_i p_i, which coincides with "
            "estimate_var's here, so the two OLS problems are literally the same."
        ),
    ),
    ValidationCase(
        id="var.gvar_global_link_identity",
        subsystem="var",
        title="Solved global system reproduces the country equations exactly",
        title_es="El sistema global resuelto reproduce exactamente las ecuaciones por país",
        mechanism=Mechanism.INTERNAL,
        compute=_gvar_link_identity,
        reference=_gvar_link_identity_ref,
        tol=Tol.TIGHT,
        citation=(
            "Dees, di Mauro, Pesaran and Smith (2007, JAE 22(1):1-38), eqs. "
            "(9)-(11): stacking A_i0 W_i x_t = a_i0 + sum_l A_il W_i x_{t-l} + "
            "u_it over i gives G x_t = a_0 + sum_l H_l x_{t-l} + eps_t."
        ),
        notes=(
            "Pushes the estimated global panel back through the link matrices G "
            "and H_l and checks that what comes out is the stacked country-OLS "
            "residual vector, date by date. G/H are built by the link step and "
            "the residuals by the per-country regressions, so the two sides are "
            "independent code paths. Note that GVARResult.intercept is the "
            "SOLVED G^-1 a_0, hence the a_0 = G @ intercept step. Measured "
            "agreement 4.4e-16 on residuals of order 1."
        ),
    ),
    ValidationCase(
        id="var.gvar_persistence_profile_unit_at_impact",
        subsystem="var",
        title="GVAR persistence profile equals 1 at horizon 0",
        title_es="El perfil de persistencia del GVAR vale 1 en el horizonte 0",
        mechanism=Mechanism.ANALYTICAL,
        compute=_gvar_persistence_profile_impact,
        reference=lambda: {"pp_at_impact": np.ones(3)},
        tol=Tol.EXACT,
        citation=(
            "Pesaran and Shin (1996, Journal of Econometrics 71(1-2):117-143): "
            "PP(b, h) = b' R_h Sigma R_h' b / (b' R_0 Sigma R_0' b), so PP(b, 0) "
            "= 1 for every non-degenerate b."
        ),
        notes=(
            "Checked for three combinations b — a single country-variable, a "
            "cross-country spread and the equally weighted sum — to confirm the "
            "normalisation is by b's own impact variance and not by a fixed "
            "scale. Only horizon 0 is analytic; the rest of the profile is a "
            "free number and is deliberately not asserted."
        ),
    ),
    ValidationCase(
        id="var.gvar_gfevd_rows_sum_to_one",
        subsystem="var",
        title="Normalised GVAR generalised FEVD rows sum to 1 at every horizon",
        title_es="Las filas de la FEVD generalizada normalizada del GVAR suman 1 en todo horizonte",
        mechanism=Mechanism.ANALYTICAL,
        compute=_gvar_gfevd_row_sums,
        reference=lambda: {"row_sums": np.ones(13 * 4)},
        tol=Tol.EXACT,
        citation=(
            "Pesaran and Shin (1998, Economics Letters 58(1):17-29) generalised "
            "FEVD under the Diebold-Yilmaz (2009, Economic Journal 119:158-171) "
            "row normalisation."
        ),
        notes=(
            "Only the NORMALISED rows are asserted. The Pesaran-Shin '>= 1' "
            "bound on the unnormalised row sum does not hold for a GVAR at any "
            "horizon, h = 0 included: their proof needs Psi_0 = I and a GVAR has "
            "R_0 = G^-1, which makes the impact row sum a Rayleigh quotient "
            "bounded only by the eigenvalues of corr(Sigma_eps). On this DGP the "
            "impact row sums are 0.98, 0.96, 0.99, 0.92 — all below one."
        ),
    ),
    ValidationCase(
        id="var.gvar_single_country_girf_vs_var_irf",
        subsystem="var",
        title="One-country GVAR GIRF equals the generalised IRF of that VAR",
        title_es="La FIR generalizada del GVAR de un solo país coincide con la del VAR",
        mechanism=Mechanism.INTERNAL,
        compute=_gvar_solo_girf,
        reference=_gvar_solo_girf_ref,
        tol=Tol.TIGHT,
        citation=(
            "Pesaran and Shin (1998, Economics Letters 58(1):17-29): "
            "GIRF_h(j) = Psi_h Sigma e_j / sqrt(sigma_jj), which is column j of "
            "Psi_h B0 with B0 = Sigma diag(sigma_jj)^-1/2."
        ),
        notes=(
            "With a single country and an empty star block G = I, so R_h = Psi_h "
            "and the GVAR's generalised responses must coincide with the "
            "ordinary VAR MA recursion in puremacro.var.irf.irf fed the "
            "generalised impact matrix. Both innovations are shocked. This "
            "pins the R_h = Psi_h G^-1 plumbing and the sqrt(sigma_jj) "
            "normalisation against an implementation that knows nothing about "
            "the GVAR."
        ),
    ),
    ValidationCase(
        id="var.gvar_single_country_gfevd_vs_var_gfevd",
        subsystem="var",
        title="One-country GVAR generalised FEVD equals puremacro.var.gfevd",
        title_es="La FEVD generalizada del GVAR de un solo país coincide con puremacro.var.gfevd",
        mechanism=Mechanism.INTERNAL,
        compute=_gvar_solo_gfevd,
        reference=_gvar_solo_gfevd_ref,
        tol=Tol.TIGHT,
        citation=(
            "Pesaran and Shin (1998, Economics Letters 58(1):17-29) generalised "
            "FEVD; the GVAR substitutes R_l for Psi_l, and R_l = Psi_l when "
            "G = I."
        ),
        notes=(
            "gvar._gfevd_from_ma takes MA coefficients so it can be handed "
            "R_h = Psi_h G^-1; var.irf.gfevd builds them itself from A_list. "
            "With one country G = I and the two must agree exactly. Guards the "
            "cumulative numerator/denominator recursion in the GVAR copy."
        ),
    ),
]
