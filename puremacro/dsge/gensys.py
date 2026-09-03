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

from ._qz import ordqz_sorted


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
    S, T, alpha, beta, Q, Z = ordqz_sorted(
        Gamma0, Gamma1,
        lambda a, b: (abs(b) < div * abs(a)),  # stable (|b/a|<div) first
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

    # Existence and uniqueness both live on Q2 Pi, not on Z2.
    #
    # Premultiplying the system by Q^H and writing w_t = Z^H z_t gives
    # S w_t = T w_{t-1} + Q^H Psi eps + Q^H Pi eta. Stability requires the
    # unstable block w2 to be identically zero, and setting it to zero in its
    # own equation leaves
    #       0 = Q2 Psi eps + Q2 Pi eta,
    # so the expectation errors are pinned down by the shocks through
    # Q2 Pi. A solution exists iff Q2 Psi lies in the column space of Q2 Pi,
    # and it is unique iff Q2 Pi has full column rank. The order condition
    # above has already forced Q2 Pi to be square (n_eta == n_unstable), so
    # here both conditions reduce to its nonsingularity.
    #
    # The previous test took singular values of `Z2`, a column slice of the
    # unitary Z returned by the QZ. Those are all exactly 1 by construction,
    # so `np.all(sv_Z2 > tol)` was true for every model ever passed in and
    # eu[1] could not be 0: the indeterminacy branch below was unreachable.
    if n_unstable == 0:
        # No unstable modes to zero, so no expectation errors to solve for.
        # (The order condition has already established n_eta == 0.)
        eu[1] = 1
        Q2_Pi_sv = np.array([1.0])
    else:
        Q2_Pi_sv = np.linalg.svd(QH_Pi_unstable, compute_uv=False)
        tol_Pi = (max(QH_Pi_unstable.shape) * Q2_Pi_sv.max()
                  * np.finfo(float).eps * 10)
        if Q2_Pi_sv.min() > tol_Pi:
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

    # Shock impact:  z_t = G z_{t-1} + Impact eps_t
    #
    # Matching the eps terms in Gamma0 z_t = Gamma1 z_{t-1} + Psi eps + Pi eta
    # gives Gamma0 * Impact = Psi + Pi * N, where N is the loading of the
    # expectation errors on the shocks. This used to read
    # `Impact = (Gamma0 - G Gamma1)^{-1} Psi`, which drops the `Pi N` term
    # entirely -- i.e. it solves the system as if the expectation errors did
    # not respond to the shocks, when responding to them is the whole content
    # of the rational-expectations solution.
    #
    # N is what zeroes the unstable modes: 0 = Q2 Psi eps + Q2 Pi eta forces
    # eta = N eps with N = -(Q2 Pi)^-1 Q2 Psi. Substituting into the stable
    # block, w1_t = S11^-1 T11 w1_{t-1} + S11^-1 (Q1 Psi + Q1 Pi N) eps, and
    # z_t = Z1 w1_t.
    #
    # On y_t = a E_t y_{t+1} + eps_t, whose unique stable solution is
    # y_t = eps_t, the old expression returned an impact of [0, -2] at
    # a = 0.5: the variable did not respond to its own shock at all, and the
    # sign was wrong on the other element. The error is zero only when
    # Pi N = 0, i.e. when the expectation errors do not load on the shocks --
    # which is to say, for models with no forward-looking behaviour left in
    # them.
    QH_Psi = Q.conj().T @ Psi
    Q1_Psi = QH_Psi[:n_stable, :]
    Q1_Pi = QH_Pi[:n_stable, :]
    if n_unstable > 0:
        Q2_Psi = QH_Psi[n_stable:, :]
        # Square and nonsingular: the order condition and the uniqueness
        # check above have both been passed by the time we get here.
        N = -np.linalg.solve(QH_Pi_unstable, Q2_Psi)   # (n_eta, n_eps)
        forcing = Q1_Psi + Q1_Pi @ N
    else:
        forcing = Q1_Psi
    Impact = (Z1 @ S11_inv @ forcing).real

    return GensysSolution(
        G=G,
        Impact=Impact,
        eu=tuple(eu),
        eigenvalues=eigvals_sorted,
    )


__all__ = ["gensys", "GensysSolution"]
