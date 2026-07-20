"""Cholesky-identified VAR using statsmodels.

Pedagogical wrapper: lag-selection by BIC, IRFs via ``VARResults.irf``,
bootstrap confidence bands via ``irf.errband_mc``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


def fit_var(Y: pd.DataFrame, maxlags: int = 8, ic: str = "bic"):
    """Fit a reduced-form VAR(p) with lag chosen by information criterion."""
    if Y.isna().any().any():
        raise ValueError("Y contains NaNs; drop them before calling fit_var")
    model = VAR(Y)
    sel = model.select_order(maxlags=maxlags)
    p = getattr(sel, ic)
    if p is None or p < 1:
        p = 1
    return model.fit(p)


def cholesky_irf(
    res,
    horizon: int = 20,
    shock: str | None = None,
    n_boot: int = 500,
    seed: int | None = None,
    signif: float = 0.10,
) -> dict:
    """Orthogonalized IRF with Monte Carlo error bands (statsmodels).

    Returns dict with keys ``point``, ``lower``, ``upper`` — each a
    DataFrame indexed by horizon 0..H with one column per response
    variable. Only the column of the named ``shock`` is kept (responses
    of every variable to that one shock). If ``shock`` is None the first
    variable in the VAR ordering is used.

    ``signif`` is the two-sided significance level for the error bands;
    the default 0.10 yields 90% bands (the Bloom-2009 / Caldara-et-al
    convention). Pass 0.05 for 95% bands or 0.32 for one-σ bands.
    """
    names = list(res.names)
    if shock is None:
        shock = names[0]
    if shock not in names:
        raise KeyError(f"shock {shock!r} not in {names}")
    k = names.index(shock)

    # Note on the ``seed`` argument: statsmodels 0.14's ``irf_resim`` passes the
    # same ``seed`` to every Monte Carlo replication, collapsing the band. We
    # seed the global numpy RNG once and then call ``errband_mc(seed=None)`` so
    # that each replication draws different innovations.
    if seed is not None:
        np.random.seed(seed)
    irf = res.irf(horizon)
    orth = irf.orth_irfs  # shape (H+1, n, n); [h, response, shock]
    point = orth[:, :, k]
    lower, upper = irf.errband_mc(orth=True, repl=n_boot, signif=signif, seed=None)
    lower = lower[:, :, k]
    upper = upper[:, :, k]

    idx = np.arange(horizon + 1)
    return {
        "point": pd.DataFrame(point, index=idx, columns=names),
        "lower": pd.DataFrame(lower, index=idx, columns=names),
        "upper": pd.DataFrame(upper, index=idx, columns=names),
        "shock": shock,
    }


def fit_varx(Y: pd.DataFrame, X: pd.DataFrame, *, p: int, q: int = 0) -> dict:
    """Fit VARX(p, q) by OLS equation-by-equation.

    Model: ``Y_t = c + sum_{i=1..p} A_i Y_{t-i} + sum_{j=0..q} B_j X_{t-j} + e_t``.

    Returns dict with:
        ``A_endog``    : list of ``p`` (n×n) matrices, lag 1..p.
        ``B_exog``     : list of ``q+1`` (n×m) matrices, lag 0..q.
        ``c``          : (n,) intercept.
        ``Sigma``      : (n×n) residual covariance (T_eff − k DOF correction).
        ``residuals``  : (T_eff × n) residuals.
        ``endog_names``, ``exog_names``.
    """
    if p < 1:
        raise ValueError("p must be >= 1")
    if q < 0:
        raise ValueError("q must be >= 0")
    Y = Y.copy()
    X = X.copy()
    common = Y.dropna().index.intersection(X.dropna().index)
    Y = Y.loc[common]
    X = X.loc[common]
    Y_arr = Y.values
    X_arr = X.values
    T, n = Y_arr.shape
    m = X_arr.shape[1]
    start = max(p, q)
    rows = []
    for t in range(start, T):
        row = [1.0]
        for i in range(1, p + 1):
            row.extend(Y_arr[t - i])
        for j in range(0, q + 1):
            row.extend(X_arr[t - j])
        rows.append(row)
    Z = np.array(rows)
    Y_dep = Y_arr[start:]
    beta, *_ = np.linalg.lstsq(Z, Y_dep, rcond=None)
    fitted = Z @ beta
    resid = Y_dep - fitted
    dof = max(1, resid.shape[0] - Z.shape[1])
    Sigma = (resid.T @ resid) / dof

    # beta has shape (1 + n*p + m*(q+1)) × n. Each block of n (or m) consecutive
    # rows corresponds to one lag; the COEFFICIENT MATRIX for that lag is the
    # transpose (rows of beta are regressors of equation k stacked column-wise).
    c = beta[0]
    A_endog = []
    idx = 1
    for _ in range(p):
        A_endog.append(beta[idx:idx + n].T)  # (n_eq × n_reg) → matrix in usual notation
        idx += n
    B_exog = []
    for _ in range(q + 1):
        B_exog.append(beta[idx:idx + m].T)
        idx += m

    return {
        "A_endog": A_endog,
        "B_exog": B_exog,
        "c": c,
        "Sigma": Sigma,
        "residuals": resid,
        "endog_names": list(Y.columns),
        "exog_names": list(X.columns),
    }


def cholesky_irf_varx(res: dict, *, horizon: int) -> np.ndarray:
    """IRF of the endogenous block of a VARX, identified by Cholesky on Sigma.

    The exogenous regressors X enter via ``B_exog``. They affect the reduced-form
    dynamics but are not shocked here — IRFs are with respect to a unit
    Cholesky shock in the endogenous innovations.

    Returns array of shape ``(horizon+1, n, n)`` where ``[h, i, k]`` is the
    response of variable ``i`` at horizon ``h`` to a unit Cholesky shock in
    structural equation ``k``.
    """
    A_list = res["A_endog"]
    Sigma = res["Sigma"]
    p = len(A_list)
    n = Sigma.shape[0]
    P = np.linalg.cholesky(Sigma)
    Phi = [np.eye(n)]
    for h in range(1, horizon + 1):
        s = np.zeros((n, n))
        for j in range(1, min(h, p) + 1):
            s = s + Phi[h - j] @ A_list[j - 1]
        Phi.append(s)
    return np.array([phi @ P for phi in Phi])


def fevd_cholesky(var_res, *, horizon: int) -> np.ndarray:
    """Cholesky-FEVD wrapper around ``puremacro.var.irf.fevd``.

    Accepts either:
      * a ``statsmodels`` ``VARResults`` object (output of :func:`fit_var`), or
      * a dict with keys ``A_endog`` (or ``A_list``) and ``Sigma``
        (output of :func:`fit_varx`).

    Returns ``(horizon+1, n, n)`` where ``[h, i, k]`` is the share of
    forecast-error variance of variable ``i`` at horizon ``h`` attributable
    to Cholesky shock ``k``. Each ``[h, i, :]`` row sums to 1.
    """
    import sys
    from pathlib import Path
    pm = Path(__file__).resolve().parents[2] / "puremacro"
    if pm.exists() and str(pm) not in sys.path:
        sys.path.insert(0, str(pm))
    from puremacro.var.irf import fevd as _fevd  # noqa: E402

    if isinstance(var_res, dict):
        A_list = var_res.get("A_endog") or var_res.get("A_list")
        Sigma = var_res["Sigma"]
    else:
        # statsmodels VARResults
        coefs = np.asarray(var_res.coefs)  # (p, n, n)
        A_list = [coefs[i] for i in range(coefs.shape[0])]
        Sigma = np.asarray(var_res.sigma_u)
    P = np.linalg.cholesky(Sigma)
    return _fevd(A_list, P, horizon)
