"""Time-varying-parameter VAR (Primiceri 2005, simplified).

Model
-----
    y_t = X_t β_t + ε_t,            ε_t ~ N(0, Sigma)
    β_{t+1} = β_t + η_t,             η_t ~ N(0, Q)

where ``X_t = I_n ⊗ x_t'`` with ``x_t = [1, y_{t-1}', ..., y_{t-p}']``.
The state vector β_t stacks all VAR coefficients (intercepts + lag
matrices) in vec form; it follows a random walk so the VAR slowly
drifts over time.

This v1 keeps Σ constant. The full Primiceri (2005) specification adds
stochastic-volatility paths for Σ_t and time-varying contemporaneous
correlations; that extension is sketched in :func:`tvp_var_sv` (a stub
left for v2 that documents the natural slot in the same Gibbs).

Gibbs sampler
-------------
For each iteration:
  1. Forward-filter-backward-sample (Carter-Kohn 1994) on β_t given Σ, Q.
  2. Sigma | β: inverse-Wishart on the VAR residuals.
  3. Q     | β: inverse-Wishart on the random-walk increments Δβ_t.

References
----------
Primiceri, G. (2005). Time varying structural vector autoregressions
    and monetary policy. Review of Economic Studies 72(3).
Carter, C. and Kohn, R. (1994). On Gibbs sampling for state space
    models. Biometrika 81(3).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._linalg import safe_cholesky


@dataclass
class TVPVARResult:
    beta_draws: np.ndarray         # (n_draws, T_eff, k_state)
    Sigma_draws: np.ndarray        # (n_draws, n, n)
    Q_draws: np.ndarray            # (n_draws, k_state, k_state)
    n: int
    p: int
    T_eff: int
    n_draws: int


def _kalman_ffbs(
    Y_dep: np.ndarray,
    X_full: np.ndarray,
    Sigma: np.ndarray,
    Q: np.ndarray,
    a0: np.ndarray,
    P0: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Carter-Kohn FFBS for β_t.

    Y_dep : (T_eff, n) — VAR LHS.
    X_full : (T_eff, n, k_state) — design matrix per period (so that
        Y_dep[t] = X_full[t] @ β_t + noise).
    Returns one draw of β_{1..T_eff}, shape (T_eff, k_state).
    """
    T_eff, n = Y_dep.shape
    k_state = a0.shape[0]
    a_pred = np.zeros((T_eff + 1, k_state))
    P_pred = np.zeros((T_eff + 1, k_state, k_state))
    a_filt = np.zeros((T_eff, k_state))
    P_filt = np.zeros((T_eff, k_state, k_state))
    a_pred[0] = a0
    P_pred[0] = P0
    Sig = Sigma + 1e-10 * np.eye(n)

    for t in range(T_eff):
        Z = X_full[t]
        v = Y_dep[t] - Z @ a_pred[t]
        F = Z @ P_pred[t] @ Z.T + Sig
        try:
            F_inv = np.linalg.inv(F)
        except np.linalg.LinAlgError:
            F_inv = np.linalg.pinv(F)
        K = P_pred[t] @ Z.T @ F_inv
        a_filt[t] = a_pred[t] + K @ v
        P_filt[t] = P_pred[t] - K @ Z @ P_pred[t]
        a_pred[t + 1] = a_filt[t]
        P_pred[t + 1] = P_filt[t] + Q

    # Backward sample.
    beta_draw = np.zeros((T_eff, k_state))
    P_T = (P_filt[-1] + P_filt[-1].T) / 2 + 1e-10 * np.eye(k_state)
    L_T = safe_cholesky(P_T, name="tvp FFBS terminal state", jitter=1e-6)
    beta_draw[-1] = a_filt[-1] + L_T @ rng.standard_normal(k_state)
    for t in range(T_eff - 2, -1, -1):
        P_pred_next = P_filt[t] + Q
        try:
            J = P_filt[t] @ np.linalg.inv(P_pred_next)
        except np.linalg.LinAlgError:
            J = P_filt[t] @ np.linalg.pinv(P_pred_next)
        m = a_filt[t] + J @ (beta_draw[t + 1] - a_filt[t])
        P = P_filt[t] - J @ P_filt[t]
        P = (P + P.T) / 2 + 1e-10 * np.eye(k_state)
        L = safe_cholesky(P, name="tvp FFBS smoother", jitter=1e-6)
        beta_draw[t] = m + L @ rng.standard_normal(k_state)
    return beta_draw


def _inv_wishart(S, df, rng):
    n = S.shape[0]
    S_inv = np.linalg.inv(S + 1e-12 * np.eye(n))
    L = safe_cholesky(S_inv, name="tvp inverse-Wishart draw", jitter=1e-10)
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = np.sqrt(rng.chisquare(df - i))
        for j in range(i):
            A[i, j] = rng.standard_normal()
    W = L @ A @ A.T @ L.T
    return np.linalg.inv(W)


def tvp_var_fit(
    Y: np.ndarray,
    p: int = 1,
    *,
    n_draws: int = 500,
    burn: int = 500,
    Q_prior_scale: float = 0.0001,
    Sigma_prior_scale: float = 1.0,
    rng: np.random.Generator | None = None,
) -> TVPVARResult:
    """Gibbs sampler for a TVP-VAR(p) with constant Σ.

    Returns posterior draws of β_t (the time-varying coefficient path),
    Σ, and Q.
    """
    if rng is None:
        rng = np.random.default_rng()
    Y = np.asarray(Y, dtype=float)
    T, n = Y.shape
    Y_dep = Y[p:]
    T_eff = Y_dep.shape[0]
    # Static design x_t = [1, y_{t-1}', ..., y_{t-p}']
    rows = []
    for t in range(p, T):
        row = [1.0]
        for k in range(1, p + 1):
            row.extend(Y[t - k])
        rows.append(row)
    x = np.array(rows)         # (T_eff, k_x) where k_x = 1 + n*p
    k_x = x.shape[1]
    k_state = n * k_x          # one β per equation

    # Build X_full[t] as the (n, k_state) Kronecker mapping.
    X_full = np.zeros((T_eff, n, k_state))
    for t in range(T_eff):
        for i in range(n):
            X_full[t, i, i * k_x:(i + 1) * k_x] = x[t]

    # Initial mean / cov: pre-sample OLS on the first 30 obs (or fewer if T small).
    n_pre = min(40, T_eff // 4 + 1)
    if n_pre >= k_x + 1:
        # Run per-equation OLS on first n_pre periods.
        beta0 = np.zeros(k_state)
        for i in range(n):
            B_i, *_ = np.linalg.lstsq(x[:n_pre], Y_dep[:n_pre, i], rcond=None)
            beta0[i * k_x:(i + 1) * k_x] = B_i
    else:
        beta0 = np.zeros(k_state)
    P0 = 4.0 * np.eye(k_state)

    Sigma = Sigma_prior_scale * np.eye(n)
    Q = Q_prior_scale * np.eye(k_state)

    keep_beta = np.empty((n_draws, T_eff, k_state))
    keep_Sigma = np.empty((n_draws, n, n))
    keep_Q = np.empty((n_draws, k_state, k_state))
    total = n_draws + burn
    keep_idx = 0

    for it in range(total):
        beta = _kalman_ffbs(Y_dep, X_full, Sigma, Q, beta0, P0, rng)
        # Sigma posterior: residuals using time-varying β.
        resid = np.array([Y_dep[t] - X_full[t] @ beta[t] for t in range(T_eff)])
        S_sig = resid.T @ resid + Sigma_prior_scale * np.eye(n)
        Sigma = _inv_wishart(S_sig, T_eff + n + 2, rng)

        # Q posterior: increments Δβ_t = β_t - β_{t-1}.
        dbeta = np.diff(beta, axis=0)
        S_q = dbeta.T @ dbeta + Q_prior_scale * np.eye(k_state)
        Q = _inv_wishart(S_q, dbeta.shape[0] + k_state + 2, rng)

        if it >= burn:
            keep_beta[keep_idx] = beta
            keep_Sigma[keep_idx] = Sigma
            keep_Q[keep_idx] = Q
            keep_idx += 1

    return TVPVARResult(
        beta_draws=keep_beta,
        Sigma_draws=keep_Sigma,
        Q_draws=keep_Q,
        n=n, p=p, T_eff=T_eff,
        n_draws=n_draws,
    )


# ===========================================================================
# Stochastic-volatility extension: Kim-Shephard-Chib (1998) mixture sampler
# ===========================================================================

# KSC (1998) Table 4 — 7-component mixture-of-normals approximation to
# log χ²_1. The mixture means are shifted by E[log χ²_1] = -1.2704 to
# match the actual moments of log χ²_1 (this matches the canonical
# reproduction used in Kastner's ``stochvol`` R package and OCSN 2007).
_KSC_WEIGHTS = np.array([0.00730, 0.10556, 0.00002, 0.04395,
                          0.34001, 0.24566, 0.25750])
_KSC_MEANS   = np.array([-11.40039, -5.24321, -9.83726, 1.50746,
                           -0.65098,  0.52478, -2.35859])
_KSC_VARS    = np.array([5.79596, 2.61369, 5.17950, 0.16735,
                           0.64009, 0.34023, 1.26261])


@dataclass
class TVPVARSVResult:
    beta_draws: np.ndarray         # (D, T_eff, k_state)
    log_h_draws: np.ndarray        # (D, T_eff, n)
    A_draws: np.ndarray            # (D, n, n) — constant lower-triangular
    Q_draws: np.ndarray            # (D, k_state, k_state)
    W_draws: np.ndarray            # (D, n) — variance of log-h increments
    n: int
    p: int
    T_eff: int
    n_draws: int


def _sample_mixture_indicators(y_star, h, rng):
    """Sample s_t ∈ {0..6} from the KSC mixture posterior."""
    T = len(y_star)
    diff = (y_star - h)[:, None] - _KSC_MEANS[None, :]
    log_phi = -0.5 * np.log(2 * np.pi * _KSC_VARS)[None, :] - 0.5 * diff ** 2 / _KSC_VARS[None, :]
    log_post = np.log(_KSC_WEIGHTS)[None, :] + log_phi
    log_post = log_post - log_post.max(axis=1, keepdims=True)
    p = np.exp(log_post)
    p = p / p.sum(axis=1, keepdims=True)
    cum = np.cumsum(p, axis=1)
    u = rng.uniform(size=T)[:, None]
    return (u > cum).sum(axis=1)


def _ffbs_sv(y_star, m_t, v_t, w, h0, p0, rng):
    """FFBS for log h: y*_t = h_t + N(m_t, v_t); h_{t+1} = h_t + N(0, w)."""
    T = len(y_star)
    a_pred = np.empty(T + 1); P_pred = np.empty(T + 1)
    a_filt = np.empty(T); P_filt = np.empty(T)
    a_pred[0] = h0; P_pred[0] = p0
    for t in range(T):
        v = y_star[t] - a_pred[t] - m_t[t]
        F = P_pred[t] + v_t[t]
        K = P_pred[t] / F
        a_filt[t] = a_pred[t] + K * v
        P_filt[t] = P_pred[t] * (1.0 - K)
        a_pred[t + 1] = a_filt[t]
        P_pred[t + 1] = P_filt[t] + w
    h_draw = np.empty(T)
    h_draw[-1] = a_filt[-1] + np.sqrt(max(P_filt[-1], 1e-12)) * rng.standard_normal()
    for t in range(T - 2, -1, -1):
        var_pred = P_filt[t] + w
        K_b = P_filt[t] / var_pred
        m = a_filt[t] + K_b * (h_draw[t + 1] - a_filt[t])
        v = P_filt[t] - K_b * P_filt[t]
        h_draw[t] = m + np.sqrt(max(v, 1e-12)) * rng.standard_normal()
    return h_draw


def _kalman_ffbs_tv(Y_dep, X_full, Sigma_t, Q, a0, P0, rng):
    """Carter-Kohn FFBS with *time-varying* observation covariance Sigma_t."""
    T_eff, n = Y_dep.shape
    k_state = a0.shape[0]
    a_pred = np.zeros((T_eff + 1, k_state))
    P_pred = np.zeros((T_eff + 1, k_state, k_state))
    a_filt = np.zeros((T_eff, k_state))
    P_filt = np.zeros((T_eff, k_state, k_state))
    a_pred[0] = a0
    P_pred[0] = P0
    for t in range(T_eff):
        Z = X_full[t]
        Sig = Sigma_t[t] + 1e-10 * np.eye(n)
        v = Y_dep[t] - Z @ a_pred[t]
        F = Z @ P_pred[t] @ Z.T + Sig
        try:
            F_inv = np.linalg.inv(F)
        except np.linalg.LinAlgError:
            F_inv = np.linalg.pinv(F)
        K = P_pred[t] @ Z.T @ F_inv
        a_filt[t] = a_pred[t] + K @ v
        P_filt[t] = P_pred[t] - K @ Z @ P_pred[t]
        a_pred[t + 1] = a_filt[t]
        P_pred[t + 1] = P_filt[t] + Q
    beta_draw = np.zeros((T_eff, k_state))
    P_T = (P_filt[-1] + P_filt[-1].T) / 2 + 1e-10 * np.eye(k_state)
    L_T = safe_cholesky(P_T, name="tvp-sv FFBS terminal state", jitter=1e-6)
    beta_draw[-1] = a_filt[-1] + L_T @ rng.standard_normal(k_state)
    for t in range(T_eff - 2, -1, -1):
        P_pred_next = P_filt[t] + Q
        try:
            J = P_filt[t] @ np.linalg.inv(P_pred_next)
        except np.linalg.LinAlgError:
            J = P_filt[t] @ np.linalg.pinv(P_pred_next)
        m = a_filt[t] + J @ (beta_draw[t + 1] - a_filt[t])
        P = P_filt[t] - J @ P_filt[t]
        P = (P + P.T) / 2 + 1e-10 * np.eye(k_state)
        L = safe_cholesky(P, name="tvp-sv FFBS smoother", jitter=1e-6)
        beta_draw[t] = m + L @ rng.standard_normal(k_state)
    return beta_draw


def tvp_var_sv_fit(
    Y: np.ndarray,
    p: int = 1,
    *,
    n_draws: int = 500,
    burn: int = 500,
    Q_prior_scale: float = 0.0001,
    W_prior_scale: float = 0.01,
    rng: np.random.Generator | None = None,
) -> TVPVARSVResult:
    """TVP-VAR with stochastic volatility (Kim-Shephard-Chib mixture sampler).

    Specification (constant lower-triangular A; Primiceri 2005 with the
    constant-A simplification):

        y_t = X_t β_t + A^{-1} D_t^{1/2} ε_t,           ε_t ~ N(0, I_n)
        β_{t+1} = β_t + η_t,                             η_t ~ N(0, Q)
        log h_{i,t+1} = log h_{i,t} + ξ_{i,t},           ξ_{i,t} ~ N(0, W_i)

    where D_t = diag(h_{1,t}, ..., h_{n,t}). The SV equation
    log(ν_{i,t}^2) = log h_{i,t} + log(ε_{i,t}^2) is non-Gaussian;
    KSC's 7-component mixture-of-normals approximation makes it
    conditionally Gaussian, after which a Kalman FFBS draws the log-h
    paths.

    Gibbs blocks
    ------------
    1. β | y, A, h, Q  via FFBS with time-varying Σ_t = A^{-1} D_t A^{-T}.
    2. A | y, β, h     via weighted OLS, row-by-row.
    3. h | ν, A, W     via mixture-of-normals + FFBS, equation-by-equation.
    4. W | h           via inverse-Gamma per i.
    5. Q | β           via inverse-Wishart on the β-increments.

    Returns
    -------
    ``TVPVARSVResult`` with ``beta_draws``, ``log_h_draws``, ``A_draws``,
    ``Q_draws``, ``W_draws``.
    """
    if rng is None:
        rng = np.random.default_rng()
    Y = np.asarray(Y, dtype=float)
    T_full, n = Y.shape
    Y_dep = Y[p:]
    T_eff = Y_dep.shape[0]
    rows = []
    for t in range(p, T_full):
        row = [1.0]
        for k in range(1, p + 1):
            row.extend(Y[t - k])
        rows.append(row)
    x = np.array(rows)
    k_x = x.shape[1]
    k_state = n * k_x
    X_full = np.zeros((T_eff, n, k_state))
    for t in range(T_eff):
        for i in range(n):
            X_full[t, i, i * k_x:(i + 1) * k_x] = x[t]

    # Initial values.
    n_pre = min(40, T_eff // 4 + 1)
    if n_pre >= k_x + 1:
        beta0 = np.zeros(k_state)
        for i in range(n):
            B_i, *_ = np.linalg.lstsq(x[:n_pre], Y_dep[:n_pre, i], rcond=None)
            beta0[i * k_x:(i + 1) * k_x] = B_i
    else:
        beta0 = np.zeros(k_state)
    P0 = 4.0 * np.eye(k_state)

    Q = Q_prior_scale * np.eye(k_state)
    A = np.eye(n)
    log_h = np.zeros((T_eff, n))
    W = W_prior_scale * np.ones(n)

    keep_beta = np.empty((n_draws, T_eff, k_state))
    keep_log_h = np.empty((n_draws, T_eff, n))
    keep_A = np.empty((n_draws, n, n))
    keep_Q = np.empty((n_draws, k_state, k_state))
    keep_W = np.empty((n_draws, n))
    keep_idx = 0
    total = n_draws + burn

    for it in range(total):
        # --- 1. β | y, A, h, Q via FFBS with time-varying Sigma_t. ---
        Sigma_t = np.empty((T_eff, n, n))
        A_inv = np.linalg.inv(A)
        for t in range(T_eff):
            D = np.diag(np.exp(log_h[t]))
            Sigma_t[t] = A_inv @ D @ A_inv.T
        beta = _kalman_ffbs_tv(Y_dep, X_full, Sigma_t, Q, beta0, P0, rng)

        # --- 2. Reduced-form residuals. ---
        u = np.array([Y_dep[t] - X_full[t] @ beta[t] for t in range(T_eff)])

        # --- 3. Sample A row-by-row by weighted OLS on orthogonalised eqs.
        # Equation i (>=1): u_{i,t} = - sum_{j<i} A[i,j] u_{j,t} + sqrt(h_i) ε.
        for i in range(1, n):
            X_a = u[:, :i]
            y_a = u[:, i]
            w = np.exp(-log_h[:, i])
            Xw = X_a * np.sqrt(w[:, None])
            yw = y_a * np.sqrt(w)
            try:
                a_i, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
                A[i, :i] = -a_i
            except np.linalg.LinAlgError:
                pass

        # --- 4. Orthogonalised residuals + 5. SV step per equation. ---
        nu = u @ A.T
        offset = 1e-6
        for i in range(n):
            y_star = np.log(nu[:, i] ** 2 + offset)
            s = _sample_mixture_indicators(y_star, log_h[:, i], rng)
            m_t = _KSC_MEANS[s]
            v_t = _KSC_VARS[s]
            log_h[:, i] = _ffbs_sv(y_star, m_t, v_t, W[i],
                                     h0=0.0, p0=10.0, rng=rng)

        # --- 6. W | log h: inverse-Gamma per equation. ---
        for i in range(n):
            d_log_h = np.diff(log_h[:, i])
            ssr = float((d_log_h ** 2).sum())
            shape = (T_eff - 1) / 2 + 1.0
            scale = ssr / 2 + 0.01
            W[i] = scale / max(rng.gamma(shape), 1e-12)

        # --- 7. Q | β: inverse-Wishart on β-increments. ---
        d_beta = np.diff(beta, axis=0)
        S_q = d_beta.T @ d_beta + Q_prior_scale * np.eye(k_state)
        Q = _inv_wishart(S_q, d_beta.shape[0] + k_state + 2, rng)

        if it >= burn:
            keep_beta[keep_idx] = beta
            keep_log_h[keep_idx] = log_h
            keep_A[keep_idx] = A
            keep_Q[keep_idx] = Q
            keep_W[keep_idx] = W
            keep_idx += 1

    return TVPVARSVResult(
        beta_draws=keep_beta,
        log_h_draws=keep_log_h,
        A_draws=keep_A,
        Q_draws=keep_Q,
        W_draws=keep_W,
        n=n, p=p, T_eff=T_eff,
        n_draws=n_draws,
    )


__all__ = ["tvp_var_fit", "TVPVARResult", "tvp_var_sv_fit", "TVPVARSVResult"]
