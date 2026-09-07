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
