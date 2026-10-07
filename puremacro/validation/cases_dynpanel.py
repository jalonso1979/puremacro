"""Validation cases for the ``dynpanel`` subsystem (dynamic-panel GMM).

Scope: Arellano-Bond difference GMM (:func:`puremacro.dynpanel.ab_gmm`),
Blundell-Bond system GMM (:func:`puremacro.dynpanel.bb_gmm`), and their
diagnostics (Hansen J overidentification test, Arellano-Bond AR(m) serial-
correlation test) living in ``puremacro.dynpanel.diagnostics``.

Reference strategy — fully INDEPENDENT and NON-CIRCULAR.  No mainstream
offline package (statsmodels 0.14, linearmodels 7.0, arch 8.0) ships an
Arellano-Bond / Blundell-Bond difference- or system-GMM estimator, so a
direct coefficient cross-check against an external implementation is not
available; that PACKAGE/PUBLISHED check is deliberately SKIPPED rather than
faked.  Instead every reference here is one of:

* **simulate-then-recover** on a seeded dynamic panel with a KNOWN ``rho``
  (the data-generating process is the reference, not another estimator);
* a **sampling-theory band** ``|rho_hat - rho| <= c * se`` — the statistically
  correct statement of "recovers the truth within sampling tolerance";
* a **construction property** of the model (differencing manufactures AR(1)
  but not AR(2); valid instruments leave the Hansen J non-rejecting);
* a **closed-form GMM identity** (the overidentification statistic vanishes
  under exact identification);
* a **cross-method evaluation**: the Windmeijer (2005) WC-robust standard
  errors recomputed by a self-contained numpy implementation that takes the
  derivative of the two-step estimator by finite differences.

The published Stata [XT] xtabond results on ``abdata`` (Examples 1, 2 and 4,
reproduced to every printed digit) are checked in
``tests/test_dynpanel/test_ab_1991_replication.py`` and
``tests/test_dynpanel/test_fix_dynpanel_stata_xtabond.py``; the data are
third-party and not bundled, so those checks run only where a copy of
``abdata`` is supplied (``PUREMACRO_ABDATA``) and are not gallery cases.

None of these references calls any ``puremacro`` function, so there is no
risk of the circular-reference trap.  Estimator imports live INSIDE the
compute callables so importing this module stays pyodide-light (no
scipy/statsmodels at module import time).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ._model import Mechanism, Tol, ValidationCase


# ---------------------------------------------------------------------
# Seeded demo data: a stationary linear dynamic panel
#   y_{i,t} = rho * y_{i,t-1} + a_i + e_{i,t},   a_i ~ N(0, sig_a^2),
#   e_{i,t} ~ N(0, sig_e^2)  iid.
# A 30-period burn-in is discarded so the retained sample is (close to)
# its stationary distribution.  Seeded -> byte-identical every call.
# Defined HERE (not in the shared _fixtures.py) per the case-author brief.
# ---------------------------------------------------------------------
def dynpanel_demo_data(
    *,
    rho: float,
    N: int,
    T: int,
    seed: int,
    sig_a: float = 1.0,
    sig_e: float = 1.0,
    burn_in: int = 30,
) -> dict:
    """Simulate a balanced AR(1) dynamic panel in long format.

    Returns a dict with ``y``, ``panel_id``, ``time_id`` (each shape
    ``(N*T,)``) plus the generating ``rho`` for convenience.
    """
    rng = np.random.default_rng(seed)
    y_rows: list[float] = []
    pid_rows: list[int] = []
    tid_rows: list[int] = []
    # stationary marginal sd of y given the fixed effect a_i
    sd0 = sig_e / np.sqrt(max(1.0 - rho * rho, 1e-12))
    for i in range(N):
        a_i = rng.normal(0.0, sig_a)
        y = a_i / (1.0 - rho) + rng.normal(0.0, sd0)
        series: list[float] = []
        for _ in range(T + burn_in):
            y = rho * y + a_i + rng.normal(0.0, sig_e)
            series.append(y)
        for t, val in enumerate(series[burn_in:]):
            y_rows.append(val)
            pid_rows.append(i)
            tid_rows.append(t)
    return {
        "y": np.asarray(y_rows, dtype=float),
        "panel_id": np.asarray(pid_rows, dtype=int),
        "time_id": np.asarray(tid_rows, dtype=int),
        "rho": float(rho),
    }


# Locked DGP settings per case (chosen so the property holds with a
# comfortable margin; see the smoke-test notes in the case docstrings).
_RECOVER: dict[str, Any] = dict(rho=0.5, N=800, T=15, seed=4)   # point recovery (COARSE)
_BAND: dict[str, Any] = dict(rho=0.8, N=800, T=12, seed=4)      # sampling band (high persistence)
_DIAG: dict[str, Any] = dict(rho=0.5, N=600, T=12, seed=1)      # AR + Hansen diagnostics
_EXACTID: dict[str, Any] = dict(rho=0.5, N=300, T=6, seed=1)    # exact identification (J == 0)
_WC: dict[str, Any] = dict(rho=0.5, N=200, T=7, seed=1)         # Windmeijer vs finite differences


# ---------------------------------------------------------------------
# Compute callables
# ---------------------------------------------------------------------
def _ab_rho_point() -> dict:
    from puremacro.dynpanel import ab_gmm

    d = dynpanel_demo_data(**_RECOVER)
    r = ab_gmm(d["y"], d["panel_id"], d["time_id"])
    return {"rho": float(r.coefs[0])}


def _bb_rho_point() -> dict:
    from puremacro.dynpanel import bb_gmm

    d = dynpanel_demo_data(**_RECOVER)
    r = bb_gmm(d["y"], d["panel_id"], d["time_id"])
    return {"rho": float(r.coefs[0])}


def _ab_sampling_band() -> dict:
    """Slack of the two-sided sampling band ``c*se - |rho_hat - rho|``.

    Non-negative slack <=> the point estimate lies within ``c`` reported
    standard errors of the truth (here ``c = 3``, a ~99.7% Gaussian band).
    At high persistence (rho=0.8) AB difference GMM has its largest
    finite-sample bias, so this is the demanding regime for the band.
    """
    from puremacro.dynpanel import ab_gmm

    c = 3.0
    d = dynpanel_demo_data(**_BAND)
    r = ab_gmm(d["y"], d["panel_id"], d["time_id"])
    slack = c * float(r.se[0]) - abs(float(r.coefs[0]) - d["rho"])
    return {"band_slack": slack}


def _ar_pattern() -> dict:
    """Differenced residuals: AR(1) present, AR(2) absent (by construction).

    First-differencing an iid-innovation panel manufactures lag-1 serial
    correlation in Δe (Cov(Δe_t, Δe_{t-1}) = -sig_e^2 < 0) but leaves
    lag-2 uncorrelated (Cov(Δe_t, Δe_{t-2}) = 0).  We report the AR(1)
    rejection strength ``1 - ar1_p`` and the AR(2) non-rejection ``ar2_p``.
    """
    from puremacro.dynpanel import ab_gmm

    d = dynpanel_demo_data(**_DIAG)
    r = ab_gmm(d["y"], d["panel_id"], d["time_id"])
    return {"ar1_reject": 1.0 - float(r.ar1_p), "ar2_no_reject": float(r.ar2_p)}


def _hansen_does_not_reject() -> dict:
    """Hansen J p-value under VALID (correctly excluded) instruments.

    The lagged-level instruments are orthogonal to the differenced error
    by construction, so the overidentifying restrictions hold and the J
    test should not reject.
    """
    from puremacro.dynpanel import ab_gmm

    d = dynpanel_demo_data(**_DIAG)
    r = ab_gmm(d["y"], d["panel_id"], d["time_id"])
    return {"hansen_p": float(r.hansen_j_p)}


def _hansen_exact_id() -> dict:
    """Hansen J under EXACT identification (#instruments == #regressors).

    With a single collapsed lag-2 instrument for the single lagged-y
    regressor the GMM moment conditions are solved exactly, so the
    sample moments ``Z'u`` are numerically zero and the overidentification
    statistic ``J = (Z'u)' W (Z'u)`` is zero with zero degrees of freedom.
    """
    from puremacro.dynpanel import ab_gmm

    d = dynpanel_demo_data(**_EXACTID)
    r = ab_gmm(
        d["y"],
        d["panel_id"],
        d["time_id"],
        gmm_lag_window=(2, 2),  # only the lag-2 instrument
        collapse=True,          # -> exactly one instrument column
    )
    return {"J": float(r.hansen_j), "df": float(r.hansen_j_df)}


def _windmeijer_wc_se() -> dict:
    """Two-step WC-robust SEs on the Stata-default uncollapsed layout.

    ``lags(2)`` with uncollapsed GMM instruments L(2/.).y (Stata
    ``xtabond y, lags(2) twostep vce(robust) noconstant``). Exercises both
    the dead-instrument pruning (the (lag 2, t=2) column is empty once the
    rows lost to the second lag are dropped) and the Windmeijer derivative.
    """
    from puremacro.dynpanel import ab_gmm

    d = dynpanel_demo_data(**_WC)
    r = ab_gmm(
        d["y"], d["panel_id"], d["time_id"],
        lag_dep_var=2, collapse=False, two_step=True, windmeijer=True,
    )
    return {"se": [float(s) for s in r.se], "n_instruments": float(r.n_instruments)}


def _windmeijer_fd_reference() -> dict:
    """Independent numpy reference (no puremacro call) for ``_windmeijer_wc_se``.

    Builds the Arellano-Bond (1991) difference-GMM arrays of a balanced
    panel from scratch -- rows t = 3..T-1, regressors (dy_{t-1}, dy_{t-2}),
    one instrument column per (t, s) holding the level y_s for s <= t-2 --
    then runs one-step GMM with the AB H matrix, the two-step estimator
    ``beta_2(b) = (X'Z S(b)^-1 Z'X)^-1 X'Z S(b)^-1 Z'y`` with
    ``S(b) = sum_i Z_i'u_i(b)u_i(b)'Z_i``, and Windmeijer's

        Var_c = V_2 + D V_2 + V_2 D' + D V_1 D',   D = d beta_2(b)/db' at b = beta_1,

    with D obtained by CENTRAL FINITE DIFFERENCES of beta_2(b) rather than
    the analytic derivative used by puremacro (Windmeijer 2000, IFS WP00/19,
    eqs. (3.2)-(3.3)). The instrument count is the closed form
    ``sum_{t=3}^{T-1} (t-1)``.
    """
    d = dynpanel_demo_data(**_WC)
    N, T = int(_WC["N"]), int(_WC["T"])
    Y = np.empty((N, T))
    Y[d["panel_id"], d["time_id"]] = d["y"]
    P = 2
    ts = list(range(P + 1, T))
    cols = [(t, s) for t in ts for s in range(0, t - 1)]
    Zs, Xs, ys = [], [], []
    for i in range(N):
        Zi = np.zeros((len(ts), len(cols)))
        for j, (t, s) in enumerate(cols):
            Zi[ts.index(t), j] = Y[i, s]
        Zs.append(Zi)
        Xs.append(np.array([[Y[i, t - p] - Y[i, t - p - 1] for p in (1, 2)] for t in ts]))
        ys.append(np.array([Y[i, t] - Y[i, t - 1] for t in ts]))
    n_i = len(ts)
    H = 2.0 * np.eye(n_i) - np.eye(n_i, k=1) - np.eye(n_i, k=-1)
    ZX = sum(Z.T @ X for Z, X in zip(Zs, Xs))
    Zy = sum(Z.T @ y for Z, y in zip(Zs, ys))

    def gmm(W):
        A = ZX.T @ W @ ZX
        return np.linalg.solve(A, ZX.T @ W @ Zy), np.linalg.inv(A)

    def S(b):
        g = [Z.T @ (y - X @ b) for Z, X, y in zip(Zs, Xs, ys)]
        return sum(np.outer(gi, gi) for gi in g)

    W1 = np.linalg.inv(sum(Z.T @ H @ Z for Z in Zs))
    b1, A1inv = gmm(W1)
    V1 = A1inv @ ZX.T @ W1 @ S(b1) @ W1 @ ZX @ A1inv
    _b2, V2 = gmm(np.linalg.inv(S(b1)))
    D = np.zeros((P, P))
    for j in range(P):
        h = 1e-5 * max(1.0, abs(b1[j]))
        e = np.zeros(P)
        e[j] = h
        D[:, j] = (gmm(np.linalg.inv(S(b1 + e)))[0] - gmm(np.linalg.inv(S(b1 - e)))[0]) / (2 * h)
    Vc = V2 + D @ V2 + V2 @ D.T + D @ V1 @ D.T
    return {
        "se": [float(s) for s in np.sqrt(np.diag(Vc))],
        "n_instruments": float(sum(t - 1 for t in ts)),
    }


# ---------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------
CASES: list[ValidationCase] = [
    ValidationCase(
        id="dynpanel.ab_recovers_rho",
        subsystem="dynpanel",
        title="Arellano-Bond difference GMM recovers the known AR(1) coefficient",
        title_es="El GMM en diferencias de Arellano-Bond recupera el coeficiente AR(1) conocido",
        mechanism=Mechanism.INTERNAL,
        compute=_ab_rho_point,
        reference=lambda: {"rho": dynpanel_demo_data(**_RECOVER)["rho"]},
        tol=Tol.COARSE,
        citation=(
            "Simulate-then-recover on y_it = rho*y_{i,t-1} + a_i + e_it with rho=0.5 "
            "(N=800, T=15, seed=4). Arellano & Bond (1991, Rev. Econ. Stud. 58(2), 277-297), "
            "Monte Carlo design of Table 1."
        ),
        notes=(
            "DGP is the reference (no external estimator implements AB GMM offline). "
            "COARSE band 0.05; worst |bias| over 6 seeds was 0.029."
        ),
    ),
    ValidationCase(
        id="dynpanel.bb_recovers_rho",
        subsystem="dynpanel",
        title="Blundell-Bond system GMM recovers the known AR(1) coefficient",
        title_es="El GMM en sistema de Blundell-Bond recupera el coeficiente AR(1) conocido",
        mechanism=Mechanism.INTERNAL,
        compute=_bb_rho_point,
        reference=lambda: {"rho": dynpanel_demo_data(**_RECOVER)["rho"]},
        tol=Tol.COARSE,
        citation=(
            "Simulate-then-recover on y_it = rho*y_{i,t-1} + a_i + e_it with rho=0.5 "
            "(N=800, T=15, seed=4). Blundell & Bond (1998, J. Econometrics 87(1), 115-143), "
            "system-GMM Monte Carlo design (sec. 4)."
        ),
        notes=(
            "DGP is the reference. System GMM is more efficient: worst |bias| over the same "
            "6 seeds was 0.017 (< AB)."
        ),
    ),
    ValidationCase(
        id="dynpanel.ab_within_sampling_band",
        subsystem="dynpanel",
        title="Arellano-Bond estimate lies within 3 standard errors of the truth (high persistence)",
        title_es="La estimacion de Arellano-Bond cae dentro de 3 errores estandar del valor verdadero (alta persistencia)",
        mechanism=Mechanism.INTERNAL,
        compute=_ab_sampling_band,
        reference=lambda: {"band_slack": 0.0},  # slack >= 0 required
        tol=Tol.QUALITATIVE,
        citation=(
            "Sampling-theory band |rho_hat - rho| <= 3*se at rho=0.8 (N=800, T=12, seed=4). "
            "Standard-error theory for two-step Windmeijer-corrected GMM: Windmeijer (2005, "
            "J. Econometrics 126, 25-51); Roodman (2009, Stata J. 9(1), 86-136)."
        ),
        notes=(
            "QUALITATIVE lower bound: metric is the band slack c*se - |rho_hat - rho| with c=3. "
            "Worst |t|=(rho_hat-rho)/se over 6 seeds was ~2.0, comfortably inside the 3-sigma band."
        ),
    ),
    ValidationCase(
        id="dynpanel.ar_test_ar1_present_ar2_absent",
        subsystem="dynpanel",
        title="Arellano-Bond AR test: AR(1) present, AR(2) absent in differenced residuals",
        title_es="Prueba AR de Arellano-Bond: AR(1) presente y AR(2) ausente en los residuos en diferencias",
        mechanism=Mechanism.INTERNAL,
        compute=_ar_pattern,
        reference=lambda: {"ar1_reject": 0.95, "ar2_no_reject": 0.10},
        tol=Tol.QUALITATIVE,
        citation=(
            "Construction property: first-differencing iid innovations makes Cov(De_t, De_{t-1})<0 "
            "but Cov(De_t, De_{t-2})=0. Arellano & Bond (1991, Rev. Econ. Stud. 58(2)), sec. 5 "
            "specification tests; Roodman (2009, Stata J. 9(1)), eq. (10)."
        ),
        notes=(
            "QUALITATIVE lower bounds: AR(1) rejection strength 1-ar1_p >= 0.95 (observed 1.0, "
            "p_AR1=0.000) and AR(2) non-rejection ar2_p >= 0.10 (observed ~0.37)."
        ),
    ),
    ValidationCase(
        id="dynpanel.hansen_j_does_not_reject",
        subsystem="dynpanel",
        title="Hansen J does not reject under valid instruments",
        title_es="La J de Hansen no rechaza con instrumentos validos",
        mechanism=Mechanism.INTERNAL,
        compute=_hansen_does_not_reject,
        reference=lambda: {"hansen_p": 0.05},  # p >= 0.05: fail to reject
        tol=Tol.QUALITATIVE,
        citation=(
            "Overidentification under correct specification: lagged-level instruments are "
            "orthogonal to the differenced error, so J ~ chi2(df) does not reject. "
            "Hansen (1982, Econometrica 50(4), 1029-1054); Arellano & Bond (1991, sec. 5)."
        ),
        notes=(
            "QUALITATIVE lower bound hansen_p >= 0.05 (fail-to-reject at 5%). Observed p=0.978 "
            "for the locked seed; min over 6 seeds was 0.11."
        ),
    ),
    ValidationCase(
        id="dynpanel.hansen_j_zero_when_exactly_identified",
        subsystem="dynpanel",
        title="Hansen J is zero under exact identification",
        title_es="La J de Hansen es cero bajo identificacion exacta",
        mechanism=Mechanism.ANALYTICAL,
        compute=_hansen_exact_id,
        reference=lambda: {"J": 0.0, "df": 0.0},
        tol=Tol.EXACT,
        citation=(
            "Closed-form GMM identity: when #instruments == #parameters the moment conditions "
            "Z'u = 0 are solved exactly, so J = (Z'u)' W (Z'u) = 0 with df=0. "
            "Hansen (1982, Econometrica 50(4), 1029-1054)."
        ),
        notes=(
            "One collapsed lag-2 instrument for one lagged-y regressor (gmm_lag_window=(2,2), "
            "collapse=True). Observed J ~ 4e-30 (< EXACT atol 1e-12), df=0."
        ),
    ),
    ValidationCase(
        id="dynpanel.windmeijer_matches_finite_difference",
        subsystem="dynpanel",
        title=(
            "Windmeijer WC-robust two-step SEs equal an independent finite-difference "
            "evaluation (uncollapsed lags(2) layout)"
        ),
        title_es=(
            "Los errores estandar WC-robustos de Windmeijer en dos etapas coinciden con una "
            "evaluacion independiente por diferencias finitas (instrumentos sin colapsar, lags(2))"
        ),
        mechanism=Mechanism.INTERNAL,
        compute=_windmeijer_wc_se,
        reference=_windmeijer_fd_reference,
        tol=Tol.TIGHT,
        citation=(
            "Windmeijer (2000, IFS WP00/19, eqs. (3.2)-(3.3); 2005, J. Econometrics 126, 25-51): "
            "Var_c = V2 + D V2 + V2 D' + D V1 D' with D = d beta_2(W(b))/db' at the one-step "
            "estimate. Reference: numpy-only AB (1991) GMM with D by central finite differences, "
            "on y_it = 0.5*y_{i,t-1} + a_i + e_it (N=200, T=7, seed=1). Stata [XT] xtabond "
            "Example 4 (abdata) is reproduced to every printed digit by the same code "
            "(tests/test_dynpanel/test_fix_dynpanel_stata_xtabond.py, data not bundled)."
        ),
        notes=(
            "TIGHT rtol 1e-6 on both SEs and on the instrument count (14 = sum_{t=3}^{6}(t-1); "
            "the empty (lag 2, t=2) column is pruned). Observed agreement ~1e-11. "
            "Discriminating: evaluating dS/db at the step-2 residuals (puremacro <= 4.3.0) "
            "misses by 2.1e-3 relative, and the unpruned layout raised LinAlgError. "
            "The WC correction need not inflate the SE (it can shrink it), so no "
            "'se_wc >= se' check is made."
        ),
    ),
]
