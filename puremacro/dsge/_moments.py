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
variance decomposition: at horizon ``h`` the forecast error of ``v_{t+h}``
made at ``t`` is ``sum_{j=0}^{h-1} Psi_j u_{t+h-j}`` with ``Psi_0 = D`` and
``Psi_j = C A^{j-1} B`` for the Dynare-timed companion form
``(A, B, C, D) = (G, N, ghx, ghu)``.
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd
import scipy.linalg

__all__ = ["first_order_moments", "conditional_fevd", "ZeroVarianceWarning"]


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


def _is_asymptotic(h) -> bool:
    if h is None:
        return True
    if isinstance(h, str):
        return h.lower() in ("inf", "infinity", "none")
    return bool(isinstance(h, (float, np.floating)) and np.isinf(h))


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

    ``(A, B, C, D)`` is the Dynare-timed companion form: ``s_t = A s_{t-1} +
    B u_t`` and ``v_t = C s_{t-1} + D u_t``; ``sigmas`` are the innovation
    standard deviations. ``None`` (or ``inf``) in ``horizons`` requests the
    asymptotic (unconditional) shares.

    Rows whose forecast-error variance is zero have no defined shares and are
    reported as ``NaN`` (with a :class:`ZeroVarianceWarning`) rather than
    being padded with a uniform ``1/n_shocks``.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    variables = list(variables)
    shocks = list(shocks)
    n_v, n_u = len(variables), len(shocks)

    finite = [int(h) for h in horizons if h is not None and not _is_asymptotic(h)]
    max_h = max(finite, default=0)

    # Psi_0 = D, Psi_k = C A^(k-1) B
    psi: list[np.ndarray] = []
    if max_h > 0:
        psi.append(D)
        a_pow = np.eye(A.shape[0])
        for _ in range(1, max_h):
            psi.append(C @ a_pow @ B)
            a_pow = a_pow @ A

    rows: list[dict] = []
    undefined: list[tuple[str, object]] = []
    for h in horizons:
        if _is_asymptotic(h):
            label: str | int = "Infinity"
            v_shocks = np.zeros((n_v, n_u))
            for j in range(n_u):
                b_j = B[:, [j]]
                d_j = D[:, [j]]
                var_j = sigmas[j] ** 2
                if A.shape[0] > 0:
                    sig_s = scipy.linalg.solve_discrete_lyapunov(A, var_j * (b_j @ b_j.T))
                    v_shocks[:, j] = var_j * d_j[:, 0] ** 2 + np.diag(C @ sig_s @ C.T)
                else:
                    v_shocks[:, j] = var_j * d_j[:, 0] ** 2
        else:
            assert h is not None  # asymptotic horizons were handled above
            label = int(h)
            v_shocks = np.zeros((n_v, n_u))
            for k in range(int(h)):
                v_shocks += (psi[k] ** 2) * (sigmas ** 2)

        tot = v_shocks.sum(axis=1, keepdims=True)
        scale = max(float(np.max(np.abs(v_shocks))), 1.0)
        defined = (tot[:, 0] > 1e-14 * scale)
        shares = np.full((n_v, n_u), np.nan)
        shares[defined] = v_shocks[defined] / tot[defined]
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
