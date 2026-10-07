"""Linear Gaussian state-space: Kalman filter, RTS / disturbance smoother,
log-likelihood, simulation smoother (Durbin-Koopman 2002).

State-space form
----------------
    alpha_{t+1} = T_mat @ alpha_t + c + R_mat @ eta_t,   eta_t ~ N(0, Q)
    y_t         = Z_mat @ alpha_t + d + eps_t,           eps_t ~ N(0, H)

System matrices are **time-invariant** 2-D arrays: ``T`` (m, m),
``Z`` (n, m), ``R`` (m, r), ``Q`` (r, r), ``H`` (n, n), plus the
intercepts ``c`` (m,) and ``d`` (n,). Time-varying (3-D, ``(T, ...)``)
system matrices are not supported; :class:`StateSpaceModel` raises a
``ValueError`` when given one. Missing observations are encoded as
``np.nan`` in y; the filter drops the corresponding rows of Z, H, d for
that period.

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
        self.T = np.asarray(self.T, dtype=float)
        self.Z = np.asarray(self.Z, dtype=float)
        self.Q = np.asarray(self.Q, dtype=float)
        self.H = np.asarray(self.H, dtype=float)
        for name, mat in (("T", self.T), ("Z", self.Z), ("Q", self.Q), ("H", self.H)):
            if mat.ndim != 2:
                raise ValueError(
                    f"StateSpaceModel.{name} must be a 2-D (time-invariant) array; "
                    f"got shape {mat.shape}. Time-varying (T, ...) system matrices "
                    "are not supported."
                )
        m = self.T.shape[0]
        n = self.Z.shape[0]
        if self.T.shape != (m, m):
            raise ValueError(f"StateSpaceModel.T must be square (m, m); got {self.T.shape}")
        if self.Z.shape[1] != m:
            raise ValueError(f"StateSpaceModel.Z must be (n, m) with m = {m}; got {self.Z.shape}")
        if self.H.shape != (n, n):
            raise ValueError(f"StateSpaceModel.H must be (n, n) with n = {n}; got {self.H.shape}")
        if self.R is None:
            self.R = np.eye(m)
        else:
            self.R = np.asarray(self.R, dtype=float)
            if self.R.ndim != 2 or self.R.shape[0] != m:
                raise ValueError(f"StateSpaceModel.R must be (m, r) with m = {m}; got {self.R.shape}")
        r = self.R.shape[1]
        if self.Q.shape != (r, r):
            raise ValueError(f"StateSpaceModel.Q must be (r, r) with r = {r}; got {self.Q.shape}")
        if self.c is None:
            self.c = np.zeros(m)
        else:
            self.c = np.asarray(self.c, dtype=float).reshape(-1)
            if self.c.shape != (m,):
                raise ValueError(f"StateSpaceModel.c must be (m,) with m = {m}; got {self.c.shape}")
        if self.d is None:
            self.d = np.zeros(n)
        else:
            self.d = np.asarray(self.d, dtype=float).reshape(-1)
            if self.d.shape != (n,):
                raise ValueError(f"StateSpaceModel.d must be (n,) with n = {n}; got {self.d.shape}")



# The diffuse part of the state covariance, P_inf, is carried as
# A_inf A_inf' with A_inf (m, q), one column per diffuse direction not yet
# pinned down by data. A diffuse update removes one column exactly, so no
# rounding residual is left in a resolved direction, and the diffuse phase
# ends when no column is left.
#
# Cut-off on F_inf = |A'z|^2 relative to its rounding bound (|A|'|z|)^2:
# A'z can cancel to rounding level only relative to |A|'|z|, so below
# sqrt(eps) of that bound z carries no diffuse information.
_DIFFUSE_TOL = float(np.finfo(float).eps)


def _propagate_A_inf(Tm: np.ndarray, A_inf: np.ndarray) -> np.ndarray:
    """T P_inf T' in factored form, dropping directions T annihilates
    (numerical rank, as in ``np.linalg.matrix_rank``)."""
    A_next = Tm @ A_inf
    if A_next.shape[1] == 0:
        return A_next
    U, sv, _ = np.linalg.svd(A_next, full_matrices=False)
    keep = sv > sv[0] * max(A_next.shape) * np.finfo(float).eps
    if keep.all():
        return A_next
    return U[:, keep] * sv[keep]


def _kalman_diffuse_update(
    a_t: np.ndarray,
    P_t: np.ndarray,
    A_inf: np.ndarray,
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
    A_curr = A_inf.copy()
    ll_t = 0.0
    # Observations are processed one at a time (Durbin-Koopman 2012, §6.4),
    # which needs uncorrelated measurement errors: rotate the period onto
    # the eigenvectors of H_t. The rotation is orthogonal, so the
    # likelihood needs no Jacobian term.
    y_dm = y_t[obs] - d_t
    if np.count_nonzero(H_t - np.diag(np.diag(H_t))):
        h_diag, U = np.linalg.eigh(H_t)
        h_diag = np.maximum(h_diag, 0.0)
        Z_u, y_u = U.T @ Z_t, U.T @ y_dm
    else:
        h_diag, Z_u, y_u = np.diag(H_t), Z_t, y_dm
    for k in range(n_obs):
        z_k = Z_u[k]                          # (m,)
        v_k = float(y_u[k] - z_k @ a_curr)
        w_k = A_curr.T @ z_k                  # (q,)
        F_inf_k = float(w_k @ w_k)
        F_star_k = float(z_k @ P_curr @ z_k + h_diag[k])
        bound = np.abs(A_curr).T @ np.abs(z_k)
        if F_inf_k > _DIFFUSE_TOL * float(bound @ bound):
            # Diffuse update (Koopman-Durbin eq. 5.18):
            # P_* <- P_* + F_* K0 K0' - K0 M_*' - M_* K0'.
            M_inf = A_curr @ w_k              # (m,)
            M_star = P_curr @ z_k             # (m,)
            K0 = M_inf / F_inf_k
            a_curr = a_curr + K0 * v_k
            P_curr = P_curr + F_star_k * np.outer(K0, K0) \
                      - np.outer(K0, M_star) \
                      - np.outer(M_star, K0)
            # P_inf - M_inf M_inf' / F_inf = A (I - w w' / w'w) A': keep
            # the orthogonal complement of w in the factor's column space.
            Q_w, _ = np.linalg.qr(w_k[:, None], mode="complete")
            A_curr = A_curr @ Q_w[:, 1:]
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
    A_inf_next = _propagate_A_inf(Tm, A_curr)

    innov_t = np.full(n, np.nan)
    innov_t[obs] = y_t[obs] - Z_t @ a_t - d_t

    return a_filt_t, P_filt_t, a_pred_next, P_pred_next, A_inf_next, ll_t, innov_t


def _kalman_standard_update(
    a_t: np.ndarray,
    P_t: np.ndarray,
    y_t: np.ndarray,
    obs: np.ndarray | None,
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
    v = (y_t - Z_t @ a_t - d_t) if obs is None else (y_t[obs] - Z_t @ a_t - d_t)
    F = Z_t @ P_t @ Z_t.T + H_t
    try:
        F_chol = np.linalg.cholesky(F)
    except np.linalg.LinAlgError:
        F = F + 1e-10 * np.eye(n_obs)
        F_chol = safe_cholesky(F, name="kalman filter F (augmented)")
    Zt_Pt = Z_t @ P_t.T
    P_Zt = Zt_Pt.T
    F_inv_v = np.linalg.solve(F_chol.T, np.linalg.solve(F_chol, v))
    F_inv_Zt = np.linalg.solve(F_chol.T, np.linalg.solve(F_chol, Zt_Pt))
    K = Tm @ F_inv_Zt.T

    a_filt_t = a_t + P_Zt @ F_inv_v
    P_filt_t = P_t - P_Zt @ F_inv_Zt
    a_pred_next = Tm @ a_t + c + K @ v
    P_pred_next = Tm @ P_t @ Tm.T + RQR - (Tm @ P_Zt) @ K.T

    if obs is None or n_obs == n:
        innov_full = v
        F_full = F
        K_full = K
    else:
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
         (approximately diffuse). Pass ``diffuse_states=[...]`` for the
         exact-diffuse initialisation of those state indices instead.

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
    out, _ = _kalman_filter(y, model, a0, P0, diffuse_scale, diffuse_states)
    return out


def _kalman_filter(
    y: np.ndarray,
    model: StateSpaceModel,
    a0: Optional[np.ndarray],
    P0: Optional[np.ndarray],
    diffuse_scale: float,
    diffuse_states: Optional[list],
) -> tuple[dict, np.ndarray]:
    """:func:`kalman_filter`, plus a ``(T,)`` mask of the periods updated by
    the exact-diffuse recursion (they store no ``F`` / ``K``)."""
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

    # Exact-diffuse initialisation (Koopman-Durbin 2003), with the
    # observations processed one at a time while it lasts (Durbin-Koopman
    # 2012, §6.4). ``diffuse_states`` lists the indices
    # of α whose prior is diffuse (infinite variance). We carry the
    # diffuse part exactly, as P_inf = A_inf A_inf' (see _DIFFUSE_TOL);
    # once all diffuse directions are pinned down by data, A_inf has no
    # columns left and the standard recursion takes over.
    if diffuse_states is None:
        diffuse_indices = []
    else:
        diffuse_indices = list(diffuse_states)
    use_exact_diffuse = bool(diffuse_indices)
    A_inf = np.zeros((m, 0))
    if use_exact_diffuse:
        # Replace the diffuse rows/cols of P0 with finite zeros and put
        # the unit "infinity weight" into P_inf instead.
        P0 = P0.copy()
        for i in diffuse_indices:
            for j in range(m):
                P0[i, j] = 0.0
                P0[j, i] = 0.0
        A_inf = np.eye(m)[:, sorted(set(diffuse_indices))]

    a_pred = np.zeros((T_obs + 1, m))
    P_pred = np.zeros((T_obs + 1, m, m))
    a_filt = np.zeros((T_obs, m))
    P_filt = np.zeros((T_obs, m, m))
    innov = np.full((T_obs, n), np.nan)
    F_arr = np.zeros((T_obs, n, n))
    K_arr = np.zeros((T_obs, m, n))

    diffuse = np.zeros(T_obs, dtype=bool)

    a_pred[0] = a0
    P_pred[0] = P0
    loglik = 0.0
    log2pi = np.log(2.0 * np.pi)

    has_nans = bool(np.isnan(y).any())
    if not has_nans and A_inf.shape[1] == 0:
        Z_t, d_t, H_t = Z, d, H
        for t in range(T_obs):
            a_t = a_pred[t]
            P_t = P_pred[t]
            y_t = y[t]
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
                a_t, P_t, y_t, None, Z_t, d_t, H_t, Tm, c, RQR, n, m, log2pi
            )
            loglik += ll_t
    else:
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
                if A_inf.shape[1]:
                    # The diffuse part moves with the state too.
                    A_inf = _propagate_A_inf(Tm, A_inf)
                continue

            Z_t = Z[obs]
            d_t = d[obs]
            H_t = H[np.ix_(obs, obs)]

            # ---- Exact-diffuse path: process this period one obs at a time ----
            if A_inf.shape[1]:
                (
                    a_filt[t],
                    P_filt[t],
                    a_pred[t + 1],
                    P_pred[t + 1],
                    A_inf,
                    ll_t,
                    innov[t],
                ) = _kalman_diffuse_update(
                    a_t, P_t, A_inf, y_t, obs, Z_t, d_t, H_t, Tm, c, RQR, n, log2pi
                )
                loglik += ll_t
                # F and K bookkeeping (skip during diffuse since the matrices
                # are not the standard ones; downstream smoother will read
                # P_filt / a_filt instead).
                diffuse[t] = True
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
    }, diffuse


# ---------------------------------------------------------------------------
# Fixed-interval state smoother (state means + covariances)
# ---------------------------------------------------------------------------
def kalman_smoother(
    y: np.ndarray,
    model: StateSpaceModel,
    a0: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None,
    diffuse_scale: float = 1e6,
    diffuse_states: Optional[list] = None,
) -> dict:
    """Filter then smooth. Returns filter dict augmented with
    ``a_smooth`` (T, m) and ``P_smooth`` (T, m, m).

    The smoother is the Durbin-Koopman backward recursion (Durbin and Koopman,
    *Time Series Analysis by State Space Methods*, 2nd ed., section 4.4):

        r_{t-1} = Z_t' F_t^{-1} v_t + L_t' r_t,          L_t = T - K_t Z_t
        N_{t-1} = Z_t' F_t^{-1} Z_t + L_t' N_t L_t
        a_smooth_t = a_t + P_t r_{t-1},   P_smooth_t = P_t - P_t N_{t-1} P_t

    with ``r_T = 0``, ``N_T = 0``, ``(a_t, P_t)`` the one-step predictions and
    ``Z_t``, ``F_t``, ``K_t`` restricted to the entries observed at ``t``; a
    period with nothing observed reduces to ``r_{t-1} = T' r_t``,
    ``N_{t-1} = T' N_t T``. It inverts only ``F_t``, which the filter has
    already factored, never ``P_t``.

    It replaces the Rauch-Tung-Striebel form, whose gain
    ``P_filt_t T' P_{t+1}^{-1}`` was built from ``pinv(P_{t+1})``. The two are
    algebraically identical, but whenever the state carries the structural
    innovation (DSGE models filtered on ``[x_t; u_t]``) ``P_{t+1}`` is
    rank-deficient by construction, and on a small RBC its smallest singular
    value sat at pinv's default cutoff: whether that direction was kept was
    decided by last-bit differences in the SVD, so the first smoothed period
    differed by ~1e-6 between LAPACK builds and the interior by ~1e-8.

    Periods handled by the exact-diffuse initialisation (``diffuse_states``)
    carry no standard ``F_t`` / ``K_t``; the smoother keeps the RTS step over
    that initial stretch, from the Durbin-Koopman estimate at its end.
    """
    out, diffuse = _kalman_filter(y, model, a0, P0, diffuse_scale, diffuse_states)
    a_pred = out["a_pred"]
    P_pred = out["P_pred"]
    a_filt = out["a_filt"]
    P_filt = out["P_filt"]
    innov = out["innov"]
    F_arr = out["F"]
    K_arr = out["K"]
    Tm, Z = model.T, model.Z

    T_obs, m = a_filt.shape
    a_sm = np.zeros_like(a_filt)
    P_sm = np.zeros_like(P_filt)

    # The exact-diffuse periods are an initial stretch: once P_inf has
    # collapsed it stays zero.
    n_diffuse = int(np.flatnonzero(diffuse)[-1]) + 1 if diffuse.any() else 0

    r = np.zeros(m)
    N = np.zeros((m, m))
    for t in range(T_obs - 1, n_diffuse - 1, -1):
        obs = ~np.isnan(innov[t])
        if obs.any():
            Z_t = Z[obs]
            v_t = innov[t][obs]
            K_t = K_arr[t][:, obs]
            # The same factor the filter used: it stores F after any ridge it
            # had to add, and Cholesky is deterministic in its input.
            F_chol = np.linalg.cholesky(F_arr[t][np.ix_(obs, obs)])
            F_inv_v = np.linalg.solve(F_chol.T, np.linalg.solve(F_chol, v_t))
            F_inv_Z = np.linalg.solve(F_chol.T, np.linalg.solve(F_chol, Z_t))
            L_t = Tm - K_t @ Z_t
            r = Z_t.T @ F_inv_v + L_t.T @ r
            N = Z_t.T @ F_inv_Z + L_t.T @ N @ L_t
        else:
            r = Tm.T @ r
            N = Tm.T @ N @ Tm
        a_sm[t] = a_pred[t] + P_pred[t] @ r
        P_sm[t] = P_pred[t] - P_pred[t] @ N @ P_pred[t]

    if n_diffuse == T_obs:
        a_sm[-1] = a_filt[-1]
        P_sm[-1] = P_filt[-1]
        n_diffuse -= 1
    for t in range(n_diffuse - 1, -1, -1):
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
      2. Run the smoother on ``y - y+`` **with the intercepts and the initial
         mean set to zero**, to get the linear part of E[alpha | y - y+].
      3. Output alpha+ + a_hat.

    Step 2's zeroing is load-bearing, and it used to zero only ``a0``. Every
    recursion in :func:`kalman_filter` and :func:`kalman_smoother` is jointly
    linear in ``(y, a0, c, d)``, so the smoother is an affine map
    ``a_hat(y) = A y + b`` whose gains depend on the second moments alone. The
    Durbin-Koopman identity is
    ``alpha_tilde = a_hat(y) + [alpha+ - a_hat(y+)] = alpha+ + A(y - y+)``:
    the offset ``b`` cancels between the two smoothing passes, so the single
    pass on ``y*`` must evaluate the *homogeneous* model. Passing the original
    ``model`` added ``b`` back a second time and shifted every draw by the whole
    intercept — on a local level with ``d = 5`` the draws sat exactly ``-5.0``
    from the posterior mean at every ``t``, against a Monte Carlo standard
    error of 0.012.

    The generation loop below correctly keeps ``c`` and ``d``: they cancel in
    the difference, and only the covariance structure of ``(alpha+, y+)``
    matters. Where ``c`` and ``d`` are both zero the result is bit-identical to
    before.
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
    # y* has already differenced away every deterministic term, so the pass
    # below must run the homogeneous model: zero intercepts as well as a zero
    # initial mean. See the docstring for why.
    model_zero = StateSpaceModel(T=Tm, Z=Z, Q=Q, H=H, R=R,
                                 c=np.zeros(m), d=np.zeros(n))
    out = kalman_smoother(y_star, model_zero, a0=np.zeros(m), P0=P0,
                          diffuse_scale=diffuse_scale)
    return a_plus + out["a_smooth"]


__all__ = [
    "StateSpaceModel",
    "kalman_filter",
    "kalman_smoother",
    "simulation_smoother",
]
