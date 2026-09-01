"""Linear Gaussian state-space: Kalman filter, RTS / disturbance smoother,
log-likelihood, simulation smoother (Durbin-Koopman 2002).

State-space form
----------------
    alpha_{t+1} = T_mat @ alpha_t + c + R_mat @ eta_t,   eta_t ~ N(0, Q)
    y_t         = Z_mat @ alpha_t + d + eps_t,           eps_t ~ N(0, H)

System matrices may be supplied as 2-D arrays (time-invariant) or 3-D
arrays of shape (T, ...) (time-varying). Missing observations are
encoded as ``np.nan`` in y; the filter drops the corresponding rows of
Z, H, d for that period.

This module is the foundation for: Stock-Watson dynamic factor models,
unobserved-components / Beveridge-Nelson / Watson UC, nowcasting
(Giannone-Reichlin-Small), and linearised DSGE likelihoods.

References
----------
Durbin, J. and Koopman, S. J. (2012). *Time Series Analysis by State
Space Methods*, 2nd ed., Oxford University Press. The recursions
follow chapters 4 (filter) and 4.5 (disturbance smoother).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ._linalg import safe_cholesky


# ---------------------------------------------------------------------------
# System container
# ---------------------------------------------------------------------------
@dataclass
class StateSpaceModel:
    """Time-invariant linear Gaussian state-space.

    Attributes
    ----------
    T : (m, m) ndarray
        State transition.
    Z : (n, m) ndarray
        Observation loading.
    R : (m, r) ndarray, optional
        Selection matrix for the state shock (default = identity m).
    Q : (r, r) ndarray
        State-shock covariance.
    H : (n, n) ndarray
        Measurement-error covariance (set to small ridge for non-noisy y).
    c : (m,) ndarray, optional
        State intercept.
    d : (n,) ndarray, optional
        Measurement intercept.
    """
    T: np.ndarray
    Z: np.ndarray
    Q: np.ndarray
    H: np.ndarray
    R: Optional[np.ndarray] = None
    c: Optional[np.ndarray] = None
    d: Optional[np.ndarray] = None

    def __post_init__(self):
        m = self.T.shape[0]
        n = self.Z.shape[0]
        if self.R is None:
            self.R = np.eye(m)
        if self.c is None:
            self.c = np.zeros(m)
        if self.d is None:
            self.d = np.zeros(n)



def _kalman_diffuse_update(
    a_t: np.ndarray,
    P_t: np.ndarray,
    P_inf: np.ndarray,
    y_t: np.ndarray,
    obs: np.ndarray,
    Z_t: np.ndarray,
    d_t: np.ndarray,
    H_t: np.ndarray,
    Tm: np.ndarray,
    c: np.ndarray,
    RQR: np.ndarray,
    n: int,
    log2pi: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    n_obs = Z_t.shape[0]
    a_curr = a_t.copy()
    P_curr = P_t.copy()
    P_inf_curr = P_inf.copy()
    ll_t = 0.0
    for k in range(n_obs):
        z_k = Z_t[k]                          # (m,)
        d_k = d_t[k]
        h_k = H_t[k, k]
        v_k = float(y_t[obs][k] - z_k @ a_curr - d_k)
        F_inf_k = float(z_k @ P_inf_curr @ z_k)
        F_star_k = float(z_k @ P_curr @ z_k + h_k)
        if F_inf_k > 1e-12:
            # Diffuse update (Koopman-Durbin eq. 5.18)
            M_inf = P_inf_curr @ z_k          # (m,)
            K0 = M_inf / F_inf_k
            a_curr = a_curr + K0 * v_k
            P_inf_curr = P_inf_curr - np.outer(K0, M_inf)
            P_curr = P_curr + (F_star_k / F_inf_k) * np.outer(K0, K0) \
                      - np.outer(K0, P_curr @ z_k) \
                      - np.outer(P_curr @ z_k, K0)
            ll_t += -0.5 * (log2pi + np.log(F_inf_k))
        else:
            if F_star_k <= 0:
                F_star_k = 1e-10
            K = P_curr @ z_k / F_star_k
            a_curr = a_curr + K * v_k
            P_curr = P_curr - np.outer(K, P_curr @ z_k)
            ll_t += -0.5 * (log2pi + np.log(F_star_k)
                             + v_k * v_k / F_star_k)
    a_filt_t = a_curr
    P_filt_t = P_curr
    a_pred_next = Tm @ a_curr + c
    P_pred_next = Tm @ P_curr @ Tm.T + RQR
    P_inf_next = Tm @ P_inf_curr @ Tm.T
    # Clean tiny negative diagonals from numerical drift.
    np.fill_diagonal(P_inf_next, np.maximum(np.diag(P_inf_next), 0.0))

    innov_t = np.full(n, np.nan)
    innov_t[obs] = y_t[obs] - Z_t @ a_t - d_t

    return a_filt_t, P_filt_t, a_pred_next, P_pred_next, P_inf_next, ll_t, innov_t


def _kalman_standard_update(
    a_t: np.ndarray,
    P_t: np.ndarray,
    y_t: np.ndarray,
    obs: np.ndarray,
    Z_t: np.ndarray,
    d_t: np.ndarray,
    H_t: np.ndarray,
    Tm: np.ndarray,
    c: np.ndarray,
    RQR: np.ndarray,
    n: int,
    m: int,
    log2pi: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    n_obs = Z_t.shape[0]
    v = y_t[obs] - Z_t @ a_t - d_t
    F = Z_t @ P_t @ Z_t.T + H_t
    try:
        F_chol = np.linalg.cholesky(F)
    except np.linalg.LinAlgError:
        F = F + 1e-10 * np.eye(n_obs)
        F_chol = safe_cholesky(F, name="kalman filter F (augmented)")
    F_inv_v = np.linalg.solve(F_chol.T, np.linalg.solve(F_chol, v))
    F_inv_Zt = np.linalg.solve(F_chol.T, np.linalg.solve(F_chol, Z_t @ P_t.T))
    K = (Tm @ P_t @ Z_t.T) @ np.linalg.inv(F)

    a_filt_t = a_t + P_t @ Z_t.T @ F_inv_v
    P_filt_t = P_t - P_t @ Z_t.T @ F_inv_Zt
    a_pred_next = Tm @ a_t + c + K @ v
    P_pred_next = Tm @ P_t @ Tm.T + RQR - K @ F @ K.T

    innov_full = np.full(n, np.nan)
    innov_full[obs] = v
    F_full = np.zeros((n, n))
    F_full[np.ix_(obs, obs)] = F
    K_full = np.zeros((m, n))
    K_full[:, obs] = K

    log_det = 2.0 * np.sum(np.log(np.diag(F_chol)))
    ll_t = -0.5 * (n_obs * log2pi + log_det + v @ F_inv_v)

    return a_filt_t, P_filt_t, a_pred_next, P_pred_next, innov_full, F_full, K_full, ll_t


# ---------------------------------------------------------------------------
# Kalman filter
# ---------------------------------------------------------------------------
def kalman_filter(
    y: np.ndarray,
    model: StateSpaceModel,
    a0: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None,
    diffuse_scale: float = 1e6,
    diffuse_states: Optional[list] = None,
) -> dict:
    """Run the Kalman filter on y (shape (T, n)).

    NaN entries in y are treated as missing.

    Parameters
    ----------
    a0 : initial state mean. Defaults to zeros.
    P0 : initial state covariance. Defaults to ``diffuse_scale * I_m``
         (approximately diffuse — a clean implementation of exact diffuse
         init is deferred to v2).

    Returns
    -------
    dict with
        a_pred : (T+1, m) — one-step-ahead state means
        P_pred : (T+1, m, m) — one-step-ahead state covariances
        a_filt : (T, m)     — filtered state means
        P_filt : (T, m, m)  — filtered state covariances
        innov  : (T, n)     — innovations (NaN where y missing)
        F      : (T, n, n)  — innovation covariances
        K      : (T, m, n)  — Kalman gains
        loglik : float       — sum of period-by-period log-likelihoods
    """
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    T_obs, n = y.shape
    m = model.T.shape[0]
    Tm, Z, R, Q, H = model.T, model.Z, model.R, model.Q, model.H
    c, d = model.c, model.d
    assert R is not None, "StateSpaceModel.R must not be None after __post_init__"
    assert c is not None, "StateSpaceModel.c must not be None after __post_init__"
    assert d is not None, "StateSpaceModel.d must not be None after __post_init__"
    RQR = R @ Q @ R.T

    if a0 is None:
        a0 = np.zeros(m)
    if P0 is None:
        P0 = diffuse_scale * np.eye(m)

    # Exact-diffuse initialisation (Koopman-Durbin 2003) for the
    # univariate-observation case. ``diffuse_states`` lists the indices
    # of α whose prior is diffuse (infinite variance). We carry an
    # extra "P_inf" matrix that captures the diffuse part exactly; once
    # all diffuse directions are pinned down by data, P_inf collapses
    # to zero and the standard recursion takes over.
    if diffuse_states is None:
        diffuse_indices = []
    else:
        diffuse_indices = list(diffuse_states)
    use_exact_diffuse = bool(diffuse_indices)
    P_inf = np.zeros((m, m))
    if use_exact_diffuse:
        # Replace the diffuse rows/cols of P0 with finite zeros and put
        # the unit "infinity weight" into P_inf instead.
        P0 = P0.copy()
        for i in diffuse_indices:
            for j in range(m):
                P0[i, j] = 0.0
                P0[j, i] = 0.0
            P_inf[i, i] = 1.0

    a_pred = np.zeros((T_obs + 1, m))
    P_pred = np.zeros((T_obs + 1, m, m))
    a_filt = np.zeros((T_obs, m))
    P_filt = np.zeros((T_obs, m, m))
    innov = np.full((T_obs, n), np.nan)
    F_arr = np.zeros((T_obs, n, n))
    K_arr = np.zeros((T_obs, m, n))

    a_pred[0] = a0
    P_pred[0] = P0
    loglik = 0.0
    log2pi = np.log(2.0 * np.pi)

    for t in range(T_obs):
        a_t = a_pred[t]
        P_t = P_pred[t]
        y_t = y[t]
        obs = ~np.isnan(y_t)
        n_obs = int(obs.sum())
        if n_obs == 0:
            a_filt[t] = a_t
            P_filt[t] = P_t
            a_pred[t + 1] = Tm @ a_t + c
            P_pred[t + 1] = Tm @ P_t @ Tm.T + RQR
            continue

        Z_t = Z[obs]
        d_t = d[obs]
        H_t = H[np.ix_(obs, obs)]

        # ---- Exact-diffuse path: process this period one obs at a time ----
        if use_exact_diffuse and np.any(np.diag(P_inf) > 0.0):
            (
                a_filt[t],
                P_filt[t],
                a_pred[t + 1],
                P_pred[t + 1],
                P_inf,
                ll_t,
                innov[t],
            ) = _kalman_diffuse_update(
                a_t, P_t, P_inf, y_t, obs, Z_t, d_t, H_t, Tm, c, RQR, n, log2pi
            )
            loglik += ll_t
            # F and K bookkeeping (skip during diffuse since the matrices
            # are not the standard ones; downstream smoother will read
            # P_filt / a_filt instead).
            continue

        # ---- Standard non-diffuse update ----
        (
            a_filt[t],
            P_filt[t],
            a_pred[t + 1],
            P_pred[t + 1],
            innov[t],
            F_arr[t],
            K_arr[t],
            ll_t,
        ) = _kalman_standard_update(
            a_t, P_t, y_t, obs, Z_t, d_t, H_t, Tm, c, RQR, n, m, log2pi
        )
        loglik += ll_t

    return {
        "a_pred": a_pred,
        "P_pred": P_pred,
        "a_filt": a_filt,
        "P_filt": P_filt,
        "innov": innov,
        "F": F_arr,
        "K": K_arr,
        "loglik": float(loglik),
    }


# ---------------------------------------------------------------------------
# RTS smoother (state means + covariances)
# ---------------------------------------------------------------------------
def kalman_smoother(
    y: np.ndarray,
    model: StateSpaceModel,
    a0: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None,
    diffuse_scale: float = 1e6,
    diffuse_states: Optional[list] = None,
) -> dict:
    """Filter then RTS smooth. Returns filter dict augmented with
    ``a_smooth`` (T, m) and ``P_smooth`` (T, m, m)."""
    out = kalman_filter(y, model, a0=a0, P0=P0,
                         diffuse_scale=diffuse_scale,
                         diffuse_states=diffuse_states)
    a_pred = out["a_pred"]
    P_pred = out["P_pred"]
    a_filt = out["a_filt"]
    P_filt = out["P_filt"]
    Tm = model.T

    T_obs, m = a_filt.shape
    a_sm = np.zeros_like(a_filt)
    P_sm = np.zeros_like(P_filt)
    a_sm[-1] = a_filt[-1]
    P_sm[-1] = P_filt[-1]

    for t in range(T_obs - 2, -1, -1):
        P_pred_next = P_pred[t + 1]
        try:
            J = P_filt[t] @ Tm.T @ np.linalg.pinv(P_pred_next)
        except np.linalg.LinAlgError:
            J = np.zeros((m, m))
        a_sm[t] = a_filt[t] + J @ (a_sm[t + 1] - a_pred[t + 1])
        P_sm[t] = P_filt[t] + J @ (P_sm[t + 1] - P_pred_next) @ J.T

    out["a_smooth"] = a_sm
    out["P_smooth"] = P_sm
    return out


# ---------------------------------------------------------------------------
# Simulation smoother (Durbin-Koopman 2002)
# ---------------------------------------------------------------------------
def simulation_smoother(
    y: np.ndarray,
    model: StateSpaceModel,
    rng: Optional[np.random.Generator] = None,
    a0: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None,
    diffuse_scale: float = 1e6,
) -> np.ndarray:
    """One draw from p(alpha_{1:T} | y_{1:T}) via Durbin-Koopman.

    Algorithm:
      1. Simulate alpha+, y+ from the model with the same dimensions as y.
      2. Run smoother on y - y+ to get a_hat = E[alpha | y - y+].
      3. Output alpha+ + a_hat.
    """
    if rng is None:
        rng = np.random.default_rng()
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    T_obs, n = y.shape
    m = model.T.shape[0]
    Tm, Z, R, Q, H = model.T, model.Z, model.R, model.Q, model.H
    c, d = model.c, model.d
    assert R is not None, "StateSpaceModel.R must not be None after __post_init__"
    assert c is not None, "StateSpaceModel.c must not be None after __post_init__"
    assert d is not None, "StateSpaceModel.d must not be None after __post_init__"
    r = Q.shape[0]

    a_plus = np.zeros((T_obs, m))
    y_plus = np.zeros((T_obs, n))
    if a0 is None:
        a0 = np.zeros(m)
    if P0 is None:
        P0 = diffuse_scale * np.eye(m)
    L0 = safe_cholesky(P0 + 1e-10 * np.eye(m),
                        name="simulation smoother P0")
    a_curr = a0 + L0 @ rng.standard_normal(m)
    LQ = safe_cholesky(Q + 1e-12 * np.eye(r),
                        name="simulation smoother Q")
    LH = safe_cholesky(H + 1e-12 * np.eye(n),
                        name="simulation smoother H")

    for t in range(T_obs):
        eps = LH @ rng.standard_normal(n)
        y_plus[t] = Z @ a_curr + d + eps
        eta = LQ @ rng.standard_normal(r)
        a_next = Tm @ a_curr + c + R @ eta
        a_plus[t] = a_curr
        a_curr = a_next

    y_star = y - y_plus
    out = kalman_smoother(y_star, model, a0=np.zeros(m), P0=P0,
                          diffuse_scale=diffuse_scale)
    return a_plus + out["a_smooth"]


__all__ = [
    "StateSpaceModel",
    "kalman_filter",
    "kalman_smoother",
    "simulation_smoother",
]
