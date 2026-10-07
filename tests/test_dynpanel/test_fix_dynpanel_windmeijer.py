"""Windmeijer (2005) WC-robust covariance: evaluation point of dS/dbeta.

Windmeijer (2000, IFS WP00/19, eqs. (3.2)-(3.3)) defines D as the
derivative of the two-step estimator with respect to the argument of its
weight matrix, at the one-step estimate:

    beta_2(b) = (X'Z S(b)^{-1} Z'X)^{-1} X'Z S(b)^{-1} Z'y,
    S(b) = sum_i Z_i'(y_i - X_i b)(y_i - X_i b)'Z_i,
    D = d beta_2(b) / d b'  at  b = beta_1,
    Var_c = V_2 + D V_2 + V_2 D' + D V_1 D'.

The reference below builds the instruments itself (numpy only, balanced
panel) and obtains D by central finite differences of beta_2(b), so it
shares no code with puremacro. Before the fix puremacro evaluated dS/db at
the step-2 residuals and missed the reference by 1e-4..4e-3 (relative);
after the fix it agrees to ~1e-11.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from puremacro.dynpanel import ab_gmm, bb_gmm
from puremacro.dynpanel.diagnostics import windmeijer_correction

from .conftest import simulate_dynamic_panel


# ---------------------------------------------------------------------
# Independent reference (numpy only)
# ---------------------------------------------------------------------
def _ab_arrays(Y: np.ndarray, P: int, collapse: bool):
    """Per-panel (Z_i, X_i, y_i) of AB difference GMM on a balanced panel.

    Rows t = P+1..T-1 (0-based); regressors dy_{t-1..t-P}; GMM instruments
    are the levels y_{t-2}, y_{t-3}, ..., y_0: one column per (t, s) when
    uncollapsed (Holtz-Eakin/Newey/Rosen), one per lag when collapsed.
    """
    N, T = Y.shape
    ts = list(range(P + 1, T))
    Zs, Xs, ys = [], [], []
    for i in range(N):
        yi = np.array([Y[i, t] - Y[i, t - 1] for t in ts])
        Xi = np.array([[Y[i, t - p] - Y[i, t - p - 1] for p in range(1, P + 1)] for t in ts])
        if collapse:
            lags = range(2, T)
            Zi = np.array([[Y[i, t - l] if t - l >= 0 else 0.0 for l in lags] for t in ts])
        else:
            cols = [(t, s) for t in ts for s in range(0, t - 1)]
            Zi = np.zeros((len(ts), len(cols)))
            for j, (t, s) in enumerate(cols):
                Zi[ts.index(t), j] = Y[i, s]
        Zs.append(Zi)
        Xs.append(Xi)
        ys.append(yi)
    return Zs, Xs, ys


def _two_step_fd_reference(Zs, Xs, ys, W1):
    """Two-step GMM with WC covariance, D by central finite differences."""
    ZX = sum(Z.T @ X for Z, X in zip(Zs, Xs))
    Zy = sum(Z.T @ y for Z, y in zip(Zs, ys))

    def gmm(W):
        A = ZX.T @ W @ ZX
        return np.linalg.solve(A, ZX.T @ W @ Zy), np.linalg.inv(A)

    def S(b):
        g = [Z.T @ (y - X @ b) for Z, X, y in zip(Zs, Xs, ys)]
        return sum(np.outer(gi, gi) for gi in g)

    b1, A1inv = gmm(W1)
    V1 = A1inv @ ZX.T @ W1 @ S(b1) @ W1 @ ZX @ A1inv
    b2, V2 = gmm(np.linalg.inv(S(b1)))
    k = len(b1)
    D = np.zeros((k, k))
    for j in range(k):
        h = 1e-5 * max(1.0, abs(b1[j]))
        e = np.zeros(k)
        e[j] = h
        D[:, j] = (gmm(np.linalg.inv(S(b1 + e)))[0] - gmm(np.linalg.inv(S(b1 - e)))[0]) / (2 * h)
    Vc = V2 + D @ V2 + V2 @ D.T + D @ V1 @ D.T
    return {"b1": b1, "b2": b2, "V1": V1, "V2": V2, "Vc": Vc, "A1inv": A1inv}


def _ab_reference(Y, P, collapse):
    Zs, Xs, ys = _ab_arrays(Y, P, collapse)
    n_i = Zs[0].shape[0]
    H = 2.0 * np.eye(n_i) - np.eye(n_i, k=1) - np.eye(n_i, k=-1)
    W1 = np.linalg.inv(sum(Z.T @ H @ Z for Z in Zs))
    ref = _two_step_fd_reference(Zs, Xs, ys, W1)
    ref["m"] = Zs[0].shape[1]
    ref["resid1"] = np.concatenate([y - X @ ref["b1"] for X, y in zip(Xs, ys)])
    ref["n"] = sum(len(y) for y in ys)
    return ref


def _long(Y):
    N, T = Y.shape
    return Y.ravel(), np.repeat(np.arange(N), T), np.tile(np.arange(T), N)


CASES = [
    # (N, T, rho, seed, lag_dep_var, collapse)
    (200, 7, 0.5, 1, 1, True),     # default layout
    (80, 8, 0.5, 0, 1, True),      # existing test_ab_windmeijer_increases_se panel
    (150, 8, 0.8, 3, 1, False),    # uncollapsed, high persistence
    (200, 7, 0.5, 1, 2, False),    # uncollapsed lags(2): needs dead-column pruning
    (60, 6, 0.5, 5, 2, False),     # small N
]


@pytest.mark.parametrize("N,T,rho,seed,P,collapse", CASES)
def test_ab_wc_se_matches_finite_difference_reference(N, T, rho, seed, P, collapse):
    sim = simulate_dynamic_panel(N=N, T=T, rho=rho, seed=seed)
    Y = sim["y"].reshape(N, T)
    ref = _ab_reference(Y, P, collapse)
    y, pid, tid = _long(Y)
    res = ab_gmm(y, pid, tid, lag_dep_var=P, collapse=collapse)
    assert res.n_instruments == ref["m"]
    np.testing.assert_allclose(res.coefs, ref["b2"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(res.se, np.sqrt(np.diag(ref["Vc"])), rtol=1e-7)
    np.testing.assert_allclose(res.cov, ref["Vc"], rtol=1e-6, atol=1e-12)


def test_ab_one_step_robust_and_homoskedastic_match_reference():
    N, T = 120, 7
    sim = simulate_dynamic_panel(N=N, T=T, rho=0.5, seed=11)
    Y = sim["y"].reshape(N, T)
    ref = _ab_reference(Y, 2, False)
    y, pid, tid = _long(Y)
    rob = ab_gmm(y, pid, tid, lag_dep_var=2, collapse=False, two_step=False)
    np.testing.assert_allclose(rob.coefs, ref["b1"], rtol=1e-9)
    np.testing.assert_allclose(rob.cov, ref["V1"], rtol=1e-8, atol=1e-14)
    hom = ab_gmm(y, pid, tid, lag_dep_var=2, collapse=False, two_step=False, robust=False)
    k = len(ref["b1"])
    sigma2 = ref["resid1"] @ ref["resid1"] / (ref["n"] - k)
    # Stata xtdpd p.19: sigma2 * W1^{-1} with H_i = (1, -0.5) -> sigma2/2 with (2, -1)
    np.testing.assert_allclose(hom.cov, 0.5 * sigma2 * ref["A1inv"], rtol=1e-8)
    np.testing.assert_allclose(hom.coefs, rob.coefs, rtol=1e-12)


def test_robust_flag_ignored_for_two_step():
    sim = simulate_dynamic_panel(N=100, T=6, rho=0.5, seed=0)
    a = ab_gmm(sim["y"], sim["panel_id"], sim["time_id"])
    b = ab_gmm(sim["y"], sim["panel_id"], sim["time_id"], robust=False)
    np.testing.assert_array_equal(a.se, b.se)


@pytest.mark.parametrize("collapse", [True, False])
def test_bb_wc_se_matches_finite_difference_reference(collapse):
    """bb_gmm shares windmeijer_correction; check it on bb's own system."""
    import sys

    bbm = sys.modules["puremacro.dynpanel.bb_gmm"]
    sim = simulate_dynamic_panel(N=150, T=7, rho=0.7, seed=2)
    y, pid, tid = sim["y"], sim["panel_id"], sim["time_id"]
    sysd = bbm._build_system(
        y, pid, tid, lag_dep_var=1, X_endog=None, X_pred=None, X_exog=None,
        gmm_lag_window=(2, None), collapse=collapse,
    )
    Z, X, yy, rows = sysd["Z"], sysd["X"], sysd["y"], sysd["rows"]
    groups: dict = {}
    for r, d in enumerate(rows):
        groups.setdefault(d["panel"], []).append(r)
    Zs = [Z[g] for g in groups.values()]
    Xs = [X[g] for g in groups.values()]
    ys = [yy[g] for g in groups.values()]
    W1 = np.linalg.inv(Z.T @ Z)  # bb_gmm's step-1 weight
    ref = _two_step_fd_reference(Zs, Xs, ys, W1)
    res = bb_gmm(y, pid, tid, collapse=collapse)
    np.testing.assert_allclose(res.coefs, ref["b2"], rtol=1e-9)
    np.testing.assert_allclose(res.se, np.sqrt(np.diag(ref["Vc"])), rtol=1e-7)


def _pieces(N=80, T=7, seed=4):
    """Two-step pieces from the independent reference for direct calls."""
    sim = simulate_dynamic_panel(N=N, T=T, rho=0.5, seed=seed)
    Y = sim["y"].reshape(N, T)
    Zs, Xs, ys = _ab_arrays(Y, 1, True)
    Z, X, yv = np.vstack(Zs), np.vstack(Xs), np.concatenate(ys)
    rows = [{"panel": i, "t": t} for i in range(N) for t in range(2, T)]
    ref = _ab_reference(Y, 1, True)
    b1, b2 = ref["b1"], ref["b2"]
    u1, u2 = yv - X @ b1, yv - X @ b2
    S1 = sum(np.outer(Zi.T @ (yi - Xi @ b1), Zi.T @ (yi - Xi @ b1)) for Zi, Xi, yi in zip(Zs, Xs, ys))
    W2 = np.linalg.inv(S1)
    kw = dict(beta_hat=b2, Z=Z, X_diff=X, y_diff=yv, W2=W2, residuals_step2=u2,
              cov_uncorrected=ref["V2"], diff_rows=rows, cov_step1=ref["V1"])
    return kw, u1, ref


def test_windmeijer_correction_with_step1_residuals_matches_reference():
    kw, u1, ref = _pieces()
    V = windmeijer_correction(**kw, residuals_step1=u1)
    np.testing.assert_allclose(V, ref["Vc"], rtol=1e-6, atol=1e-13)


def test_windmeijer_correction_without_step1_residuals_warns_and_differs():
    kw, u1, ref = _pieces()
    with pytest.warns(FutureWarning, match="residuals_step1"):
        V_legacy = windmeijer_correction(**kw)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        V = windmeijer_correction(**kw, residuals_step1=u1)
    assert not np.allclose(V_legacy, V, rtol=1e-6, atol=0)


def test_windmeijer_correction_rejects_misshaped_step1_residuals():
    kw, u1, _ = _pieces()
    with pytest.raises(ValueError, match="residuals_step1"):
        windmeijer_correction(**kw, residuals_step1=u1[:-1])


def test_estimators_do_not_emit_legacy_warning():
    sim = simulate_dynamic_panel(N=80, T=7, rho=0.5, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        ab_gmm(sim["y"], sim["panel_id"], sim["time_id"])
        bb_gmm(sim["y"], sim["panel_id"], sim["time_id"])
