"""IRF (MA recursion), FEVD, historical decomposition for VAR(p)."""
from __future__ import annotations

import numpy as np


def irf(A_list, B0, horizon: int) -> np.ndarray:
    """Orthogonalised IRF: shape (H+1, n, n) where [h, i, j] is the
    response of variable i at horizon h to a 1-SD shock in structural
    equation j."""
    n = B0.shape[0]
    p = len(A_list)
    Phi = [np.eye(n)]
    for h in range(1, horizon + 1):
        Ph = np.zeros((n, n))
        for j in range(1, min(h, p) + 1):
            Ph += Phi[h - j] @ A_list[j - 1]
        Phi.append(Ph)
    return np.stack([Ph @ B0 for Ph in Phi])


def fevd(A_list, B0, horizon: int) -> np.ndarray:
    """Forecast-error variance decomposition: shape (H+1, n, n) where
    [h, i, j] is the share of Var(y_{i,t+h}) explained by shock j."""
    irf_arr = irf(A_list, B0, horizon)  # (H+1, n, n)
    cum_sq = np.cumsum(irf_arr ** 2, axis=0)
    total = cum_sq.sum(axis=2, keepdims=True)
    total = np.where(total == 0, 1.0, total)
    return cum_sq / total


def gfevd(A_list, Sigma, horizon: int, normalize: bool = True) -> np.ndarray:
    """Pesaran-Shin (1998) generalised FEVD — order-invariant.

    For each horizon h, the contribution of shock j to the forecast-error
    variance of variable i is

        theta_{ij}(h) = sigma_jj^{-1} * sum_{l=0..h} (e_i' Phi_l Sigma e_j)^2
                        / sum_{l=0..h} (e_i' Phi_l Sigma Phi_l' e_i)

    Rows do not sum to 1; with ``normalize=True`` (default), each row is
    rescaled to sum to 1 (the Diebold-Yilmaz convention).

    Parameters
    ----------
    A_list : list of (n, n) ndarrays — VAR coefficient matrices.
    Sigma  : (n, n) ndarray — reduced-form residual covariance.
    horizon : int — H.
    normalize : bool — divide each row by its sum (DY convention).

    Returns
    -------
    (H+1, n, n) ndarray of GFEVD shares.
    """
    n = Sigma.shape[0]
    p = len(A_list)
    Phi_list: list[np.ndarray] = [np.eye(n)]
    for h in range(1, horizon + 1):
        Ph = np.zeros((n, n))
        for j in range(1, min(h, p) + 1):
            Ph += Phi_list[h - j] @ A_list[j - 1]
        Phi_list.append(Ph)
    Phi = np.stack(Phi_list)  # (H+1, n, n)
    sigma_jj = np.diag(Sigma)
    # The estimand above is exactly invariant to a change of units
    # y -> D y for diagonal positive D: under it A_i -> D A_i D^-1 and
    # Sigma -> D Sigma D, so the numerator picks up d_i^2 d_j^2 / d_j^2 = d_i^2
    # and the denominator picks up the same d_i^2, term by term.
    #
    # An ABSOLUTE floor breaks that invariance, because whether a variance
    # falls under it depends on the units the caller happened to choose. A
    # 1e-12 floor here used to move the Diebold-Yilmaz total connectedness
    # index from 13.43 to 39.48 on one fixed VAR when a single variable was
    # rescaled -- a pure units change, and a spillover index that depends on
    # whether GDP is measured in billions or trillions is not measuring
    # spillovers. It was silent, and the result stayed plausible.
    #
    # A degenerate variance is a real condition, so it is now reported rather
    # than floored: for a positive-definite Sigma every sigma_jj is positive
    # and no floor is needed at all.
    bad = np.flatnonzero(~(sigma_jj > 0.0))
    if bad.size:
        raise np.linalg.LinAlgError(
            f"gfevd: Sigma has a non-positive diagonal entry at variable(s) "
            f"{bad.tolist()} (values {sigma_jj[bad].tolist()}). The "
            f"Pesaran-Shin GFEVD divides by sigma_jj, so those shocks have no "
            f"defined contribution. Drop the degenerate variable(s) or check "
            f"the residual covariance."
        )
    out = np.zeros((horizon + 1, n, n))
    num_cum = np.zeros((n, n))
    den_cum = np.zeros(n)
    for h in range(horizon + 1):
        Ph = Phi[h]
        PS = Ph @ Sigma  # (n, n)
        num_cum += (PS ** 2) / sigma_jj[None, :]
        # var(y_i, h-step) accumulator
        var_i = np.einsum("ij,jk,ik->i", Ph, Sigma, Ph)
        den_cum += np.maximum(var_i, 0.0)
        # den_cum >= diag(Sigma) > 0 from the h = 0 term (Phi_0 = I), so the
        # guard below can only fire on a Sigma that is not positive definite,
        # which the check above has already excluded.
        theta = num_cum / den_cum[:, None]
        if normalize:
            row_sum = theta.sum(axis=1, keepdims=True)
            theta = theta / np.where(row_sum == 0, 1.0, row_sum)
        out[h] = theta
    return out


def historical_decomp(A_list, B0, residuals, *, init_y=None, intercept=None):
    """Historical decomposition of y_t into structural-shock contributions.

    Reconstructs the path of y from the structural shocks ε_t = B0^{-1} u_t
    plus a deterministic / initial-conditions piece. By construction:
        y_t = deterministic_t + Σ_{j} shocks_t[:, j]

    Parameters
    ----------
    A_list : list of (n, n) ndarrays
        VAR coefficient matrices A_1, ..., A_p.
    B0 : (n, n) ndarray
        Structural impact matrix.
    residuals : (T_eff, n) ndarray
        Reduced-form residuals (T_eff rows, where T_eff = T - p).
    init_y : (p, n) ndarray, optional
        Pre-sample observations (the first p observations of y). If None,
        zeros are assumed (so the deterministic piece encodes only the
        accumulated intercept).
    intercept : (n,) ndarray, optional
        VAR intercept c. If None, treated as zero.

    Returns
    -------
    dict with keys:
        shocks : (T_eff, n, n) — shocks[t, i, j] is variable i's value at
                 time t attributable to structural shock j.
        deterministic : (T_eff, n) — deterministic + initial-condition
                 component (i.e., y_t under the counterfactual ε_t = 0 ∀ t).
    """
    residuals = np.asarray(residuals, dtype=float)
    T_eff, n = residuals.shape
    p = len(A_list)
    B0_inv = np.linalg.inv(B0)
    eps = residuals @ B0_inv.T  # (T_eff, n) structural shocks

    if init_y is None:
        init_y = np.zeros((p, n))
    init_y = np.asarray(init_y, dtype=float)
    if intercept is None:
        intercept = np.zeros(n)
    intercept = np.asarray(intercept, dtype=float).ravel()

    # Compute MA coefficients up to T_eff
    Phi = [np.eye(n)]
    for h in range(1, T_eff):
        ph = np.zeros((n, n))
        for j in range(1, min(h, p) + 1):
            ph += Phi[h - j] @ A_list[j - 1]
        Phi.append(ph)
    Phi = np.stack(Phi)  # (T_eff, n, n)

    # Shock contributions: shocks[t, :, j] = Σ_{s=0..t} Phi[t-s] B0[:, j] eps[s, j]
    shocks = np.zeros((T_eff, n, n))
    for j in range(n):
        b_j = B0[:, j]
        for t in range(T_eff):
            for s in range(t + 1):
                shocks[t, :, j] += Phi[t - s] @ b_j * eps[s, j]

    # Deterministic / initial-condition piece: simulate y_t with eps = 0
    Y_det = np.zeros((p + T_eff, n))
    Y_det[:p] = init_y
    for t in range(p, p + T_eff):
        y_t = intercept.copy()
        for k in range(p):
            y_t = y_t + A_list[k] @ Y_det[t - 1 - k]
        Y_det[t] = y_t
    deterministic = Y_det[p:]

    return {"shocks": shocks, "deterministic": deterministic}
