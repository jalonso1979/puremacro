"""Shared first-order moment and variance-decomposition kernels.

Every solved model in :mod:`puremacro.dsge` reduces, to first order, to

    x_{t+1} = G x_t + N u_t,        v_t = M_x x_t + M_u u_t,

where ``x_t`` is the predetermined vector at the *start* of period ``t``
(known before ``u_t`` is drawn), ``u_t`` the innovations with covariance
``Sigma_u`` and ``v_t`` the vector the caller wants moments for. Because
``x_t`` is independent of ``u_t`` there is no cross term between them:

    Var(v)   = M_x Sigma_x M_x' + M_u Sigma_u M_u'
    Gamma_k  = cov(v_t, v_{t-k}) = M_x G^{k-1} (G Sigma_x M_x' + N Sigma_u M_u')

with ``Sigma_x`` the solution of the discrete Lyapunov equation
``Sigma_x = G Sigma_x G' + N Sigma_u N'``.

Two loadings cover both timing conventions used in the package:

* Dynare timing (``build_dynare`` / ``load_mod``): ``x_t`` is the lagged
  state ``s_{t-1}``, ``v_t = [s_t; c_t]`` so ``M_x = [G; F] = ghx`` and
  ``M_u = [N; L] = ghu`` — exactly Dynare's ``oo_.dr``.
* Klein timing (``build``): ``x_t`` is the state itself, ``v_t = [x_t; y_t]``
  so ``M_x = [I; F]`` and ``M_u = [0; L]``.

The forecast-error variance decomposition follows Dynare's conditional
variance decomposition: conditioning on ``x_t`` and ``u_t``, the forecast
error of ``v_{t+h}`` made at ``t`` is ``sum_{j=0}^{h-1} Psi_j u_{t+h-j}`` with
``Psi_0 = M_u`` and ``Psi_j = M_x G^{j-1} N``. :func:`conditional_fevd` takes
that companion form as ``(A, B, C, D) = (G, N, M_x, M_u)``, so the loadings
must be the ones the caller *reports*: ``(ghx, ghu)`` under Dynare timing and
``([I; F], [0; L])`` under Klein timing. Handing it the Dynare loadings for a
Klein-timed model dates the state rows one period later than the control rows
and than :func:`first_order_moments` above.
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd
import scipy.linalg

__all__ = [
    "first_order_moments",
    "conditional_fevd",
    "ZeroVarianceWarning",
    "spectral_moments",
    "one_sided_hp_filter",
    "compute_autocorr_matrices",
]


class ZeroVarianceWarning(UserWarning):
    """A variable has no forecast-error variance at some horizon, so its
    variance shares are undefined (reported as NaN)."""


def first_order_moments(
    G: np.ndarray,
    N: np.ndarray,
    M_x: np.ndarray,
    M_u: np.ndarray,
    sigma_u: np.ndarray,
    lags: int,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Unconditional covariance and autocovariances of ``v_t = M_x x_t + M_u u_t``.

    Returns ``(Sigma_x, Gamma_0, [Gamma_1, ..., Gamma_lags])`` where
    ``Gamma_k = cov(v_t, v_{t-k})``.
    """
    G = np.asarray(G, dtype=float)
    N = np.asarray(N, dtype=float)
    sigma_u = np.asarray(sigma_u, dtype=float)
    n_x = G.shape[0]
    if n_x == 0:
        sigma_x = np.zeros((0, 0))
    else:
        sigma_x = scipy.linalg.solve_discrete_lyapunov(G, N @ sigma_u @ N.T)
        sigma_x = 0.5 * (sigma_x + sigma_x.T)
    gamma_0 = M_x @ sigma_x @ M_x.T + M_u @ sigma_u @ M_u.T
    gamma_0 = 0.5 * (gamma_0 + gamma_0.T)
    # cov(x_t, v_{t-1}) = G Sigma_x M_x' + N Sigma_u M_u'
    base = G @ sigma_x @ M_x.T + N @ sigma_u @ M_u.T
    gammas: list[np.ndarray] = []
    g_pow = np.eye(n_x)
    for _ in range(int(lags)):
        gammas.append(M_x @ g_pow @ base)
        g_pow = g_pow @ G
    return sigma_x, gamma_0, gammas


def spectral_moments(
    G: np.ndarray,
    N: np.ndarray,
    M_x: np.ndarray,
    M_u: np.ndarray,
    sigma_u: np.ndarray,
    lags: int = 5,
    *,
    filter_type: str | None = None,
    hp_lambda: float = 1600.0,
    bandpass: tuple[float, float] | None = None,
    n_quad: int = 128,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Evaluate theoretical filtered or unfiltered moments via Gauss-Legendre quadrature.

    Integrates the power spectral density matrix:
        S_y(omega) = (1 / 2*pi) H_y(omega) Sigma_u H_y(omega)^H
    over [0, pi] for HP/unfiltered, or [omega_L, omega_H] for Baxter-King bandpass:
        Gamma_k = 2 * int |H(omega)|^2 Re[S_y(omega) e^{i omega k}] d omega

    Parameters
    ----------
    G : np.ndarray, shape (n_x, n_x)
        State transition companion matrix.
    N : np.ndarray, shape (n_x, n_u)
        Shock impact matrix on states.
    M_x : np.ndarray, shape (n_v, n_x)
        Observation loading on predetermined states.
    M_u : np.ndarray, shape (n_v, n_u)
        Observation loading on contemporaneous innovations.
    sigma_u : np.ndarray, shape (n_u, n_u)
        Innovation covariance matrix.
    lags : int, default 5
        Number of autocovariance lags to compute.
    filter_type : {"hp", "bandpass"} or None, optional
        Filter specification. If None, computes unfiltered moments.
    hp_lambda : float, default 1600.0
        Smoothing parameter lambda for HP filter.
    bandpass : tuple of (float, float), optional
        Periodicity bounds (low, high) for Baxter-King bandpass filter.
    n_quad : int, default 128
        Number of Gauss-Legendre quadrature nodes.

    Returns
    -------
    sigma_x : np.ndarray, shape (n_x, n_x)
        Filtered or unfiltered state covariance matrix.
    gamma_0 : np.ndarray, shape (n_v, n_v)
        Contemporaneous covariance matrix.
    gammas : list of np.ndarray, length `lags`
        Autocovariance matrices Gamma_k = cov(v_t, v_{t-k}) for k=1..lags.
    """
    G = np.asarray(G, dtype=float)
    N = np.asarray(N, dtype=float)
    M_x = np.asarray(M_x, dtype=float)
    M_u = np.asarray(M_u, dtype=float)
    sigma_u = np.asarray(sigma_u, dtype=float)
    lags = int(lags)
    n_quad = int(n_quad)

    n_x = G.shape[0]
    n_v = M_x.shape[0]

    if n_x > 0:
        g_eigs = np.abs(scipy.linalg.eigvals(G))
        if filter_type is None and bandpass is None:
            if np.any(g_eigs >= 1.0 - 1e-7):
                bad = g_eigs[g_eigs >= 1.0 - 1e-7]
                raise ValueError(
                    f"State transition matrix G has non-stationary eigenvalues (|λ| >= 1.0: {bad}); "
                    "unconditional stationary moments do not exist."
                )
        else:
            if np.any(g_eigs > 1.0 + 1e-7):
                bad = g_eigs[g_eigs > 1.0 + 1e-7]
                raise ValueError(
                    f"State transition matrix G has explosive eigenvalues (|λ| > 1.0: {bad}); "
                    "filtered moments do not exist."
                )

    # Determine integration domain [a, b]
    if filter_type == "bandpass" or (filter_type is None and bandpass is not None):
        filter_type = "bandpass"
        if bandpass is None:
            raise ValueError("bandpass parameter required for bandpass filter")
        low, high = float(bandpass[0]), float(bandpass[1])
        if low <= 0 or high <= low:
            raise ValueError(f"bandpass requires 0 < low < high, got low={low}, high={high}")
        omega_L = max(2.0 * np.pi / high, 0.0)
        omega_H = min(2.0 * np.pi / low, np.pi)
        a, b = omega_L, omega_H
    else:
        a, b = 0.0, np.pi

    # Gauss-Legendre quadrature nodes and weights mapped to [a, b]
    x_nodes, w_weights = np.polynomial.legendre.leggauss(n_quad)
    omega = 0.5 * (b - a) * x_nodes + 0.5 * (b + a)
    weights = 0.5 * (b - a) * w_weights

    # Squared gain transfer function |H(omega)|^2
    if filter_type == "hp":
        if hp_lambda <= 0:
            raise ValueError(f"hp_lambda must be positive, got {hp_lambda}")
        cos_w = np.cos(omega)
        term = 4.0 * float(hp_lambda) * ((1.0 - cos_w) ** 2)
        gain2 = term / (1.0 + term)
    elif filter_type == "bandpass":
        gain2 = np.ones_like(omega)
    else:
        gain2 = np.ones_like(omega)

    S_tilde = np.zeros((n_quad, n_v, n_v), dtype=complex)
    Sx_tilde = np.zeros((n_quad, n_x, n_x), dtype=complex)

    if n_x == 0:
        S_const = (1.0 / (2.0 * np.pi)) * (M_u @ sigma_u @ M_u.T)
        for j in range(n_quad):
            S_tilde[j] = (2.0 * weights[j] * gain2[j]) * S_const
        sigma_x = np.zeros((0, 0))
    else:
        Inx = np.eye(n_x)
        for j in range(n_quad):
            wj = omega[j]
            # Resolvent (e^{i omega_j} I - G) X_j = N
            Zj = np.exp(1j * wj) * Inx - G
            X_j = np.linalg.solve(Zj, N)
            H_j = M_x @ X_j + M_u
            S_j = (1.0 / (2.0 * np.pi)) * (H_j @ sigma_u @ H_j.conj().T)
            S_tilde[j] = (2.0 * weights[j] * gain2[j]) * S_j
            Sx_j = (1.0 / (2.0 * np.pi)) * (X_j @ sigma_u @ X_j.conj().T)
            Sx_tilde[j] = (2.0 * weights[j] * gain2[j]) * Sx_j

        sigma_x = np.sum(Sx_tilde.real, axis=0)
        sigma_x = 0.5 * (sigma_x + sigma_x.T)
        diag_x = np.diag(sigma_x)
        if np.any(diag_x < 0):
            np.fill_diagonal(sigma_x, np.maximum(diag_x, 0.0))
        if sigma_x.size > 0:
            wx, vx = np.linalg.eigh(sigma_x)
            if np.any(wx < 0):
                sigma_x = (vx * np.maximum(wx, 0.0)) @ vx.T
                sigma_x = 0.5 * (sigma_x + sigma_x.T)

    k_vals = np.arange(lags + 1)
    E = np.exp(1j * np.outer(omega, k_vals))  # shape (n_quad, lags + 1)
    Gamma = np.einsum("jvu, jk -> kvu", S_tilde, E).real

    gamma_0 = 0.5 * (Gamma[0] + Gamma[0].T)
    diag_g0 = np.diag(gamma_0)
    if np.any(diag_g0 < 0):
        np.fill_diagonal(gamma_0, np.maximum(diag_g0, 0.0))
    if gamma_0.size > 0:
        w0, v0 = np.linalg.eigh(gamma_0)
        if np.any(w0 < 0):
            gamma_0 = (v0 * np.maximum(w0, 0.0)) @ v0.T
            gamma_0 = 0.5 * (gamma_0 + gamma_0.T)

    gammas = [Gamma[k] for k in range(1, lags + 1)]
    return sigma_x, gamma_0, gammas


def one_sided_hp_filter(
    y: np.ndarray | pd.Series | pd.DataFrame,
    lamb: float = 1600.0,
) -> tuple[np.ndarray | pd.Series | pd.DataFrame, np.ndarray | pd.Series | pd.DataFrame]:
    """One-sided Hodrick-Prescott filter via recursive forward Kalman filter.

    Implements the causal state-space filter of Stock & Watson (1999, p. 301)
    and Hamilton (1994, Ch. 13), matching Dynare's one_sided_hp_filter:
        Delta^2 tau_t = eta_t,   y_t = tau_t + epsilon_t,   sigma_eps^2 / sigma_eta^2 = lambda

    Parameters
    ----------
    y : array-like, Series, or DataFrame
        Time series data of shape (T,) or (T, n). T must be >= 4.
    lamb : float, default 1600.0
        Smoothing parameter lambda (e.g. 1600 for quarterly data).

    Returns
    -------
    cycle, trend : tuple matching input container type
        Extracted cyclical component (y - trend) and trend component.
    """
    if lamb <= 0:
        raise ValueError(f"one_sided_hp_filter parameter lambda must be positive, got {lamb}")

    is_series = isinstance(y, pd.Series)
    is_df = isinstance(y, pd.DataFrame)
    orig = y

    arr = np.asarray(y, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    T, n_vars = arr.shape

    if T < 4:
        raise ValueError(f"one_sided_hp_filter requires at least 4 observations, got {T}")

    q = 1.0 / float(lamb)
    F = np.array([[2.0, -1.0], [1.0, 0.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[q, 0.0], [0.0, 0.0]])
    R = 1.0

    ytrend = np.empty((T, n_vars), dtype=float)

    for k in range(n_vars):
        yk = arr[:, k]
        # Backwards linear extrapolation initialization matching Dynare
        x = np.array([2.0 * yk[0] - yk[1], 3.0 * yk[0] - 2.0 * yk[1]])
        P = np.eye(2) * 1e5

        for j in range(T):
            obs = yk[j]
            S = (H @ P @ H.T)[0, 0] + R
            K = (F @ P @ H.T) / S
            v = obs - (H @ x)[0]
            x = F @ x + (K * v).flatten()
            Temp = F - K @ H
            P = Temp @ P @ Temp.T + Q + K * R @ K.T
            ytrend[j, k] = x[1]  # Second state element is tau_{t|t}

    ycycle = arr - ytrend

    if is_series:
        c_out = pd.Series(ycycle[:, 0], index=orig.index, name=orig.name)
        t_out = pd.Series(ytrend[:, 0], index=orig.index, name=orig.name)
        return c_out, t_out
    elif is_df:
        c_out = pd.DataFrame(ycycle, index=orig.index, columns=orig.columns)
        t_out = pd.DataFrame(ytrend, index=orig.index, columns=orig.columns)
        return c_out, t_out
    else:
        if orig.ndim == 1:
            return ycycle[:, 0], ytrend[:, 0]
        return ycycle, ytrend


def compute_autocorr_matrices(
    gamma_0: np.ndarray,
    gammas: Sequence[np.ndarray],
    variable_names: Sequence[str],
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Compute contemporaneous correlation matrix R(0) and autocorrelation matrices R(k).

    Parameters
    ----------
    gamma_0 : np.ndarray, shape (N, N)
        Contemporaneous covariance matrix.
    gammas : Sequence of np.ndarray, each shape (N, N)
        Autocovariance matrices for lags 1..n.
    variable_names : Sequence of str
        Endogenous variable names for DataFrame indexing.

    Returns
    -------
    df_corr : pd.DataFrame, shape (N, N)
        Contemporaneous correlation matrix R(0) with 1.0 diagonal and bounds [-1, 1].
    autocorr_matrices : list of pd.DataFrame, each shape (N, N)
        Cross-variable autocorrelation matrices R(k) = D^{-1/2} Gamma_k D^{-1/2}.
    """
    vars_list = list(variable_names)
    variances = np.diag(gamma_0)
    stds = np.sqrt(np.maximum(variances, 0.0))
    std_outer = np.outer(stds, stds)

    with np.errstate(divide="ignore", invalid="ignore"):
        inv_outer = np.where(std_outer > 1e-14, 1.0 / std_outer, np.nan)

    corr_mat = gamma_0 * inv_outer
    corr_mat = 0.5 * (corr_mat + corr_mat.T)
    np.fill_diagonal(corr_mat, 1.0)
    corr_mat = np.clip(corr_mat, -1.0, 1.0)
    df_corr = pd.DataFrame(corr_mat, index=vars_list, columns=vars_list)

    autocorr_mats: list[pd.DataFrame] = []
    for gamma_k in gammas:
        R_k = np.clip(gamma_k * inv_outer, -1.0, 1.0)
        autocorr_mats.append(pd.DataFrame(R_k, index=vars_list, columns=vars_list))

    return df_corr, autocorr_mats


def _is_asymptotic(h) -> bool:
    if h is None:
        return True
    if isinstance(h, str):
        return h.lower() in ("inf", "infinity", "none")
    return bool(isinstance(h, (float, np.floating)) and np.isinf(h))


def _validate_horizon(h) -> int:
    """Return ``h`` as a positive integer, or raise a named ``ValueError``.

    Only finite horizons reach here; ``None`` / ``inf`` / ``"Infinity"`` are
    filtered out by :func:`_is_asymptotic` first.
    """
    if isinstance(h, (bool, np.bool_)):
        raise ValueError(
            f"conditional_fevd: horizon must be a positive integer or None "
            f"(asymptotic), got {h!r}."
        )
    if isinstance(h, (int, np.integer)):
        h_int = int(h)
    elif isinstance(h, (float, np.floating)):
        if not float(h).is_integer():
            raise ValueError(
                f"conditional_fevd: horizon {h!r} is not an integer number of "
                "periods. Forecast horizons are whole periods; pass "
                f"{int(h)} or {int(h) + 1} explicitly rather than relying on "
                "truncation."
            )
        h_int = int(h)
    else:
        raise TypeError(
            f"conditional_fevd: horizon must be an int, a float with an "
            f"integral value, or None/inf for the asymptotic decomposition; "
            f"got {type(h).__name__} ({h!r})."
        )
    if h_int < 1:
        raise ValueError(
            f"conditional_fevd: horizon must be >= 1, got {h_int}. The "
            "h-step-ahead forecast error is only defined for h >= 1; there is "
            "no forecast error to decompose at h <= 0."
        )
    return h_int


def _shock_variances(sigmas: np.ndarray, shocks: Sequence[str]) -> np.ndarray:
    """Per-shock innovation variances from ``sigmas``.

    ``sigmas`` may be the vector of innovation standard deviations or the full
    ``(n_u, n_u)`` innovation covariance matrix. A covariance matrix with
    non-negligible off-diagonal entries is **refused**: an orthogonal variance
    decomposition of correlated innovations is not defined without an
    orthogonalisation convention, and silently dropping the correlation would
    decompose a total that differs from the model's own variance.
    """
    sig = np.asarray(sigmas, dtype=float)
    if sig.ndim == 1:
        return sig ** 2
    if sig.ndim != 2 or sig.shape[0] != sig.shape[1]:
        raise ValueError(
            "conditional_fevd: `sigmas` must be a vector of innovation "
            f"standard deviations or a square covariance matrix, got shape {sig.shape}."
        )
    diag = np.diag(sig)
    off = np.abs(sig - np.diag(diag))
    tol = 1e-12 * max(float(np.max(np.abs(diag))), 1.0)
    if float(np.max(off)) > tol:
        i, j = np.unravel_index(int(np.argmax(off)), off.shape)
        names = list(shocks)
        ni = names[i] if i < len(names) else str(i)
        nj = names[j] if j < len(names) else str(j)
        denom = np.sqrt(max(diag[i], 0.0) * max(diag[j], 0.0))
        rho = float(sig[i, j] / denom) if denom > 0.0 else float("nan")
        raise ValueError(
            "conditional_fevd: the declared innovation covariance is not "
            f"diagonal (corr({ni}, {nj}) = {rho:.6g}). A forecast-error "
            "variance decomposition attributes variance to one shock at a "
            "time, which is only defined for uncorrelated innovations; "
            "dropping the off-diagonal would decompose a total that differs "
            "from the model's own variance. Re-specify the shocks as "
            "orthogonal, or pass explicit standard deviations via `sigma=` to "
            "decompose the diagonal part deliberately."
        )
    return np.clip(diag, 0.0, None)


def conditional_fevd(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    sigmas: np.ndarray,
    horizons: Sequence[int | None],
    variables: Sequence[str],
    shocks: Sequence[str],
    *,
    warn: bool = True,
) -> pd.DataFrame:
    """Dynare-style conditional variance decomposition (shares as fractions).

    ``(A, B, C, D)`` describes the reported vector in the timing the caller
    reports it: ``x_{t+1} = A x_t + B u_t`` and ``v_t = C x_t + D u_t``. For
    Dynare timing ``x_t`` is the lagged state ``s_{t-1}`` and ``(C, D)`` are
    ``(ghx, ghu)``; for Klein timing ``x_t`` is the state itself and
    ``(C, D) = ([I; F], [0; L])``. Passing the wrong pair dates the state rows
    one period away from the control rows.

    ``sigmas`` is either the vector of innovation standard deviations or the
    full innovation covariance matrix; a covariance with non-zero off-diagonal
    entries is refused (see :func:`_shock_variances`). ``None`` (or ``inf``, or
    ``"Infinity"``) in ``horizons`` requests the asymptotic (unconditional)
    shares; repeated entries are collapsed to a single row.

    Finite horizons must be integers ``>= 1``. Asymptotic shares require a
    stationary ``A``: a unit or explosive root raises ``ValueError`` rather
    than returning shares built from a divergent Lyapunov "solution".

    Rows whose forecast-error variance is zero have no defined shares and are
    reported as ``NaN`` (with a :class:`ZeroVarianceWarning`) rather than
    being padded with a uniform ``1/n_shocks``. The zero test is made per row
    against that row's own largest total across the requested horizons, so a
    well-scaled variable is never erased because some *other* variable in the
    model is large.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float)
    variables = list(variables)
    shocks = list(shocks)
    n_v, n_u = len(variables), len(shocks)
    var_u = _shock_variances(sigmas, shocks)
    if var_u.shape[0] != n_u:
        raise ValueError(
            f"conditional_fevd: got {var_u.shape[0]} innovation variances for "
            f"{n_u} shock names {shocks}."
        )

    # Normalise the requested horizons: validate, and collapse duplicates
    # (repeated asymptotic entries used to produce a non-unique table index).
    labels: list[str | int] = []
    seen: set[str | int] = set()
    for h in horizons:
        label: str | int = "Infinity" if _is_asymptotic(h) else _validate_horizon(h)
        if label in seen:
            continue
        seen.add(label)
        labels.append(label)

    finite = [h for h in labels if h != "Infinity"]
    max_h = max(finite, default=0)  # type: ignore[type-var]

    if "Infinity" in seen and A.shape[0] > 0:
        eigs = np.abs(scipy.linalg.eigvals(A))
        if np.any(eigs >= 1.0 - 1e-7):
            bad = eigs[eigs >= 1.0 - 1e-7]
            raise ValueError(
                "conditional_fevd: the asymptotic (unconditional) variance "
                "decomposition was requested but the state transition matrix "
                f"has non-stationary eigenvalues (|λ| >= 1.0: {bad}); the "
                "unconditional forecast-error variance is infinite, so "
                "asymptotic variance shares do not exist. Request finite "
                "horizons instead."
            )

    # Psi_0 = D, Psi_k = C A^(k-1) B
    psi: list[np.ndarray] = []
    if max_h > 0:
        psi.append(D)
        a_pow = np.eye(A.shape[0])
        for _ in range(1, max_h):
            psi.append(C @ a_pow @ B)
            a_pow = a_pow @ A

    # Pass 1: forecast-error variance contributions at every requested horizon.
    contributions: list[np.ndarray] = []
    for label in labels:
        v_shocks = np.zeros((n_v, n_u))
        if label == "Infinity":
            for j in range(n_u):
                b_j = B[:, [j]]
                d_j = D[:, [j]]
                var_j = var_u[j]
                if A.shape[0] > 0:
                    sig_s = scipy.linalg.solve_discrete_lyapunov(A, var_j * (b_j @ b_j.T))
                    v_shocks[:, j] = var_j * d_j[:, 0] ** 2 + np.diag(C @ sig_s @ C.T)
                else:
                    v_shocks[:, j] = var_j * d_j[:, 0] ** 2
        else:
            for k in range(int(label)):
                v_shocks += (psi[k] ** 2) * var_u
        contributions.append(v_shocks)

    # Pass 2: the zero test is per variable, relative to that variable's own
    # largest forecast-error variance across the requested horizons.
    totals = np.column_stack([v.sum(axis=1) for v in contributions]) if contributions \
        else np.zeros((n_v, 0))
    row_ref = np.max(totals, axis=1) if totals.size else np.zeros(n_v)

    rows: list[dict] = []
    undefined: list[tuple[str, object]] = []
    for label, v_shocks in zip(labels, contributions):
        tot = v_shocks.sum(axis=1)
        defined = tot > 1e-14 * row_ref
        shares = np.full((n_v, n_u), np.nan)
        shares[defined] = v_shocks[defined] / tot[defined, None]
        for i, var in enumerate(variables):
            if not defined[i]:
                undefined.append((var, label))
            row: dict = {"Variable": var, "Horizon": label}
            for j, s in enumerate(shocks):
                row[s] = float(shares[i, j])
            rows.append(row)

    if undefined and warn:
        shown = ", ".join(f"{v}@h={hh}" for v, hh in undefined[:6])
        more = "" if len(undefined) <= 6 else f" (+{len(undefined) - 6} more)"
        warnings.warn(
            f"forecast-error variance is zero for {shown}{more}; the variance "
            "shares of those rows are undefined and reported as NaN.",
            ZeroVarianceWarning,
            stacklevel=3,
        )

    return pd.DataFrame(rows).set_index(["Variable", "Horizon"])
