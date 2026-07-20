"""Sims (2002) gensys: QZ solution for linear RE models.

Solves systems of the form

    Γ_0 * z_t = Γ_1 * z_{t-1} + Ψ * ε_t + Π * η_t

where z_t is the full endogenous-variable vector, ε_t are exogenous
shocks, and η_t = z_t - E_{t-1} z_t are expectation errors (to be
determined by the solution).

This formulation is model-agnostic: variables need not be pre-classified
into predetermined / forward-looking. The QZ decomposition identifies
the stable and unstable modes automatically, and the Blanchard-Kahn (1980)
order condition is verified from the eigenvalue count.

Solution form:
    z_t = G * z_{t-1} + Impact * ε_t

Reference
---------
Sims, C. (2002). Solving linear rational expectations models.
Computational Economics 20, 1-20.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg


@dataclass(frozen=True)
class GensysSolution:
    """Output of gensys.

    Attributes
    ----------
    G      : (n, n) state-transition matrix — z_t = G z_{t-1} + ...
    Impact : (n, n_eps) shock-impact matrix  — z_t = ... + Impact ε_t
    eu     : tuple (exist, unique) — both 1 iff unique stable solution.
    eigenvalues : sorted |generalised eigenvalues| of (Γ_0, Γ_1).
    """
    G: np.ndarray
    Impact: np.ndarray
    eu: tuple
    eigenvalues: np.ndarray


def gensys(
    Gamma0: np.ndarray,
    Gamma1: np.ndarray,
    Psi: np.ndarray,
    Pi: np.ndarray,
    *,
    div: float = 1.0 + 1e-8,
) -> GensysSolution:
    """Sims (2002) gensys — QZ solution for Γ_0 z = Γ_1 z_{-1} + Ψ ε + Π η.

    Parameters
    ----------
    Gamma0 : (n, n) current-period coefficient matrix.
    Gamma1 : (n, n) lagged-variable coefficient matrix.
    Psi    : (n, n_eps) shock coefficient matrix.
    Pi     : (n, n_eta) expectation-error coefficient matrix.
    div    : stability threshold (generalised |eigenvalue| < div is stable).
             Default 1 + ε to exclude unit roots from stable set.

    Returns
    -------
    GensysSolution
    """
    Gamma0 = np.asarray(Gamma0, dtype=float)
    Gamma1 = np.asarray(Gamma1, dtype=float)
    Psi    = np.asarray(Psi,    dtype=float)
    Pi     = np.asarray(Pi,     dtype=float)

    n = Gamma0.shape[0]
    n_eta = Pi.shape[1]
    n_eps = Psi.shape[1]

    # Generalised Schur decomposition of (Γ_0, Γ_1):
    # Γ_0 = Q S Z^H,   Γ_1 = Q T Z^H
    # Sort so that stable generalised eigenvalues (|T_ii/S_ii| < div) come first.
    S, T, alpha, beta, Q, Z = scipy.linalg.ordqz(
        Gamma0, Gamma1,
        sort=lambda a, b: (abs(b) < div * abs(a)),  # stable (|b/a|<div) first
        output="complex",
    )

    # Generalised eigenvalues |β/α|
    with np.errstate(divide="ignore", invalid="ignore"):
        eigvals = np.where(np.abs(alpha) > 1e-12,
                           np.abs(beta / alpha), np.inf)
    eigvals_sorted = np.sort(eigvals)

    n_stable = int(np.sum(eigvals < div))
    n_unstable = n - n_stable

    eu = [0, 0]
    if n_eta != n_unstable:
        # Blanchard-Kahn order condition fails
        G      = np.zeros((n, n))
        Impact = np.zeros((n, n_eps))
        return GensysSolution(
            G=G, Impact=Impact, eu=tuple(eu), eigenvalues=eigvals_sorted
        )
    eu[0] = 1

    # Partition Z and T,S into stable (1..n_stable) and unstable (n_stable+1..n)
    # Z = [Z1 | Z2] with Z1 corresponding to stable eigenvalues
    Z1 = Z[:, :n_stable]   # (n, n_stable)
    Z2 = Z[:, n_stable:]   # (n, n_unstable) — corresponds to n_eta = n_unstable cols

    # Rotated expectations: Q^H Π
    QH_Pi = Q.conj().T @ Pi   # (n, n_eta)

    # The unstable block of Q^H Π must be zero (Blanchard-Kahn rank condition).
    QH_Pi_unstable = QH_Pi[n_stable:, :]   # (n_unstable, n_eta)

    # Check rank of Z2^H (the "uniqueness" check):
    sv_Z2 = np.linalg.svd(Z2, compute_uv=False)
    tol_Z2 = max(Z2.shape) * sv_Z2.max() * np.finfo(float).eps * 10
    if np.all(sv_Z2 > tol_Z2):
        eu[1] = 1

    if eu[1] == 0:
        G      = np.zeros((n, n))
        Impact = np.zeros((n, n_eps))
        return GensysSolution(
            G=G, Impact=Impact, eu=tuple(eu), eigenvalues=eigvals_sorted
        )

    # Extract stable block
    S11 = S[:n_stable, :n_stable]   # (n_stable, n_stable)
    T11 = T[:n_stable, :n_stable]   # (n_stable, n_stable)

    # State transition (stable block in Schur coordinates):
    # z_stable_t = S11^{-1} T11 z_stable_{t-1} + ...
    # In original coordinates: G = Z1 S11^{-1} T11 Z1^H
    S11_inv = np.linalg.solve(S11.T, np.eye(n_stable)).T  # = inv(S11)
    G_core = Z1 @ S11_inv @ T11 @ Z1.conj().T
    G = G_core.real

    # Shock impact:  z_t = G z_{t-1} + Impact ε_t
    # Impact = (Γ_0 - G Γ_1)^{-1} Ψ  [from the forward recursion]
    LHS = Gamma0 - G @ Gamma1
    try:
        Impact = np.linalg.solve(LHS, Psi).real
    except np.linalg.LinAlgError:
        Impact = np.linalg.lstsq(LHS, Psi, rcond=None)[0].real

    return GensysSolution(
        G=G,
        Impact=Impact,
        eu=tuple(eu),
        eigenvalues=eigvals_sorted,
    )


__all__ = ["gensys", "GensysSolution"]
