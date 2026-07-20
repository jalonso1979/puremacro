"""Klein (2000) QZ method for linear rational-expectations models.

Solves systems of the form

    A * E_t z_{t+1} = B * z_t + C * u_t,    z_t = [x_t; y_t]

where ``x_t`` is an ``n_pre``-vector of predetermined / state variables
and ``y_t`` is an ``n_fwd``-vector of forward-looking / control
variables. Under the Blanchard-Kahn condition (number of unstable
generalised eigenvalues == n_fwd), the unique stable solution is

    x_{t+1} = G x_t + N u_t
    y_t     = F x_t + L u_t

This is a pure-numpy/scipy implementation suitable for the iPad — no
Dynare, no compiled DSGE toolbox. For canonical-form systems
(``Gamma_0 z_t = Gamma_1 z_{t-1} + Psi z_t + Pi eta_t``) reformulate
into the Klein form first.

Eigenvalue classification
~~~~~~~~~~~~~~~~~~~~~~~~~
The ``_select_stable`` function uses strict ``|β/α| < 1.0`` to classify
stable eigenvalues.  Pairs with ``|β/α| == 1.0`` (unit generalised
eigenvalues) are therefore classified as *unstable*.

For the SW07 model this is not an issue: the lag-state equations
(``kpf_lag(t+1) = kpf(t)``, etc.) produce inf generalised eigenvalues
(because the static control equations have ``A[row,:]=0``), not unit
eigenvalues.  The 16 inf eigenvalues count as unstable and, together
with 8 genuinely forward-looking roots (from the dynamic equations),
account for all 24 forward-looking variables.  The QZ stable block of
size 20 cleanly aligns with the 20 predetermined states.

The ``G = Z11 @ inv(S11) @ T11 @ inv(Z11)`` formula is numerically
verified to equal ``G1_x + G1_y @ F`` at machine precision for SW07
(difference < 3e-13).

Reference
---------
Klein, P. (2000). Using the generalised Schur form to solve a
multivariate linear rational expectations model. JEDC 24, 1405-1423.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.linalg


@dataclass(frozen=True)
class KleinSolution:
    """Klein QZ solution.

    Attributes
    ----------
    G : (n_pre, n_pre) ndarray — state transition.
    F : (n_fwd, n_pre) ndarray — policy function (y = F x).
    N : (n_pre, n_u) ndarray   — shock loading on states (zeros if no C).
    L : (n_fwd, n_u) ndarray   — shock loading on controls (zeros if no C).
    eu : tuple[int, int]       — (existence, uniqueness) flags.
                                 Both 1 ⇒ unique stable solution; 0 ⇒ violation.
    eigenvalues : sorted generalised eigenvalues of (B, A) — for diagnostics.
    """
    G: np.ndarray
    F: np.ndarray
    N: np.ndarray
    L: np.ndarray
    eu: tuple
    eigenvalues: np.ndarray


def _select_stable(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Boolean array marking stable generalised eigenvalues |beta/alpha| < 1.

    Pairs with ``alpha == 0`` are treated as infinity (unstable).
    """
    out = np.zeros(len(alpha), dtype=bool)
    for i in range(len(alpha)):
        if alpha[i] == 0:
            out[i] = False
        else:
            out[i] = abs(beta[i] / alpha[i]) < 1.0
    return out


def _solve_F_sylvester(
    A: np.ndarray,
    B: np.ndarray,
    G: np.ndarray,
    n_pre: int,
) -> np.ndarray:
    """Recover the policy function F from the equilibrium Sylvester equation.

    Klein's closed-form  ``F = -inv(Z22) @ Z21``  presumes the QZ stable
    block of size ``n_pre`` cleanly aligns with the predetermined
    subspace. In systems with multiple ``A[row,:] = 0`` (static-control)
    rows — most notably Smets-Wouters style models with many lag-state
    equations — the QZ ordering mixes spurious near-zero eigenvalues
    (the finite-side counterparts of the inf generalised eigenvalues)
    into the stable block, biasing the closed-form F even though
    ``cond(Z11)`` remains small. The remedy is to recover F directly
    from the equilibrium condition that any valid solution must satisfy.

    For the control rows of  ``A E_t z_{t+1} = B z_t``  with the
    substitutions ``y_t = F x_t``  and ``x_{t+1} = G x_t``:

        A21 @ G + A22 @ F @ G  =  B21 + B22 @ F

    Rearranging:

        A22 @ F @ G - B22 @ F  =  B21 - A21 @ G                  (*)

    This is a generalised Sylvester equation in F. Vectorising with the
    identity ``vec(M X N) = (N^T ⊗ M) vec(X)`` gives the linear system

        (G^T ⊗ A22 − I ⊗ B22) vec(F) = vec(B21 − A21 @ G)

    of size ``n_fwd * n_pre`` (480 for SW07: 24*20). Solved via
    ``np.linalg.lstsq`` for robustness against rank-deficient A22 (which
    is the SW07 case — half of the control rows are static, so A22 has
    16 zero rows).

    Parameters
    ----------
    A, B : (n, n) ndarrays — Klein-form coefficient matrices.
    G    : (n_pre, n_pre) ndarray — already-solved state transition.
    n_pre : int — number of predetermined variables.

    Returns
    -------
    F : (n_fwd, n_pre) ndarray — policy function such that  y_t = F x_t.

    Notes
    -----
    Ported from ``smets_wouters._solve_F_sylvester`` (puremacro 0.46.0),
    which exploited the SW07-specific static-vs-dynamic control split
    to keep the linear system block-diagonal. The generic port here
    drops that optimisation: the full ``(n_fwd * n_pre)``-by-
    ``(n_fwd * n_pre)`` system is solved as one block. The two
    formulations are mathematically equivalent — SW07 just had a
    convenient zero-pattern that made the block split cheap.
    """
    n = A.shape[0]
    n_fwd = n - n_pre

    A21 = A[n_pre:, :n_pre]
    A22 = A[n_pre:, n_pre:]
    B21 = B[n_pre:, :n_pre]
    B22 = B[n_pre:, n_pre:]

    # Vectorised form: (G^T kron A22 - I kron B22) vec(F) = vec(B21 - A21 G).
    # Use Fortran ordering for column-major vec convention used in the
    # SW07 reference implementation (matches np.kron's semantics).
    I_pre = np.eye(n_pre)
    M = np.kron(G.T, A22) - np.kron(I_pre, B22)        # (n_fwd*n_pre, n_fwd*n_pre)
    rhs_mat = B21 - A21 @ G                            # (n_fwd, n_pre)
    rhs = rhs_mat.flatten(order="F")

    f_vec, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    F = f_vec.reshape(n_fwd, n_pre, order="F")
    return F


class BlanchardKahnError(RuntimeError):
    """Raised by :func:`klein_solve` (in ``strict=True`` mode) when the
    Blanchard-Kahn condition fails.

    Carries the offending counts and eigenvalues so the caller can
    diagnose whether the model has too few unstable roots
    (indeterminacy) or too many (no stable solution).
    """

    def __init__(self, kind: str, n_unstable: int, n_fwd: int,
                 eigenvalues: np.ndarray):
        self.kind = kind
        self.n_unstable = n_unstable
        self.n_fwd = n_fwd
        self.eigenvalues = eigenvalues
        super().__init__(
            f"Blanchard-Kahn {kind}: {n_unstable} unstable generalised "
            f"eigenvalues vs {n_fwd} forward-looking variables. "
            f"|eigvals| sorted (first 8): "
            f"{np.array2string(np.abs(eigenvalues)[:8], precision=4)}."
        )


def klein_solve(
    A: np.ndarray,
    B: np.ndarray,
    n_pre: int,
    C: Optional[np.ndarray] = None,
    *,
    strict: bool = False,
) -> KleinSolution:
    """Solve A E_t z_{t+1} = B z_t + C u_t via QZ decomposition.

    Parameters
    ----------
    A, B : (n, n) ndarrays — coefficient matrices.
    n_pre : int — number of predetermined variables (states); the
        remaining ``n - n_pre`` rows are forward-looking.
    C : (n, n_u) ndarray, optional — shock loading.
    strict : bool, default False
        When True, raise :class:`BlanchardKahnError` on existence
        (n_unstable > n_fwd) or uniqueness (n_unstable < n_fwd, or
        Z22 rank-deficient) failures instead of returning zero
        matrices with ``eu`` flagged. Recommended for production
        callers; the default preserves the Sims/gensys-style
        soft-failure convention used by exploratory notebooks.

    Returns
    -------
    KleinSolution. Inspect ``eu`` to confirm existence/uniqueness:
        eu = (1, 1) ⇒ unique stable solution.
        eu = (1, 0) ⇒ exists but indeterminate (multiple solutions).
        eu = (0, 0) ⇒ no stable solution (Blanchard-Kahn violated).
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    n = A.shape[0]
    n_fwd = n - n_pre
    if C is None:
        C = np.zeros((n, 0))
    n_u = C.shape[1]

    # Generalised Schur (QZ): A = Q S Z',  B = Q T Z'
    # We want stable eigenvalues at the top-left.
    S, T, alpha, beta, Q, Z = scipy.linalg.ordqz(
        A, B, sort=lambda a, b: abs(b) < abs(a),  # stable first
    )

    # Diagnostics
    with np.errstate(divide="ignore", invalid="ignore"):
        eigvals = np.where(alpha != 0, beta / alpha, np.inf)
    eigvals_sorted = np.sort(np.abs(eigvals))
    n_stable = int(np.sum(_select_stable(alpha, beta)))
    n_unstable = n - n_stable

    eu = [0, 0]
    # Blanchard-Kahn: number of unstable eigenvalues must equal n_fwd.
    if n_unstable == n_fwd:
        eu[0] = 1
        # Partition Z (column-wise transformations of [x; y]):
        # Z = [[Z11, Z12], [Z21, Z22]]
        Z11 = Z[:n_pre, :n_pre]
        Z21 = Z[n_pre:, :n_pre]
        Z22 = Z[n_pre:, n_pre:]
        S11 = S[:n_pre, :n_pre]
        T11 = T[:n_pre, :n_pre]
        # Uniqueness: Z22 invertible. Use SVD-based rank with a tolerance
        # tied to the largest singular value, not just matrix_rank's
        # default (which can over-report rank on near-singular blocks).
        if Z22.size > 0:
            sv = np.linalg.svd(Z22, compute_uv=False)
            tol = max(Z22.shape) * sv.max() * np.finfo(float).eps
            z22_rank = int(np.sum(sv > tol))
        else:
            z22_rank = 0
        if z22_rank == n_fwd:
            eu[1] = 1
            # Policy function
            try:
                Z11_inv = np.linalg.inv(Z11)
            except np.linalg.LinAlgError:
                Z11_inv = np.linalg.pinv(Z11)
            try:
                S11_inv = np.linalg.inv(S11)
            except np.linalg.LinAlgError:
                S11_inv = np.linalg.pinv(S11)
            G = Z11 @ S11_inv @ T11 @ Z11_inv
            G = G.real
            try:
                Z22_inv = np.linalg.inv(Z22)
            except np.linalg.LinAlgError:
                Z22_inv = np.linalg.pinv(Z22)
            F = (-Z22_inv @ Z21).real

            # Z-partition degeneracy detection. The closed-form
            # F = -inv(Z22) Z21 above is exact when the QZ stable block
            # cleanly aligns with the predetermined subspace, but it is
            # corrupted when systems with many static-control rows
            # (A[row,:] = 0, producing inf generalised eigenvalues) mix
            # the resulting near-zero finite-side counterparts into the
            # stable block. cond(Z11) does NOT diagnose this — SW07 has
            # cond(Z11) ~ 16 yet closed-form residuals of O(10). The
            # principled test is whether the closed-form F satisfies
            # the equilibrium condition derived from the control rows:
            #
            #   A21 G + A22 F G  =  B21 + B22 F           (residual r)
            #
            # If ||r||_inf exceeds a tolerance, recover F by solving
            # the equilibrium Sylvester equation directly.
            if n_fwd > 0:
                A21 = A[n_pre:, :n_pre]
                A22 = A[n_pre:, n_pre:]
                B21 = B[n_pre:, :n_pre]
                B22 = B[n_pre:, n_pre:]
                resid = A21 @ G + A22 @ F @ G - B21 - B22 @ F
                # Scale by the magnitude of the input matrices so the
                # tolerance is relative rather than absolute.
                scale_candidates = [1.0]
                if A21.size:
                    scale_candidates.append(float(np.max(np.abs(A21))))
                if B21.size:
                    scale_candidates.append(float(np.max(np.abs(B21))))
                scale = max(scale_candidates)
                if float(np.max(np.abs(resid))) > 1e-6 * scale:
                    F = _solve_F_sylvester(A, B, G, n_pre=n_pre)

            # Shock loadings (impact response). For a generic Klein
            # form with C != 0, the response is found by solving for
            # forward-looking decisions under the unstable block.
            if n_u > 0:
                Q_ = Q.conj().T  # Klein notation: Q' A Z = S
                QC = Q_ @ C
                Q1C = QC[:n_pre]
                Q2C = QC[n_pre:]
                S22 = S[n_pre:, n_pre:]
                T22 = T[n_pre:, n_pre:]
                # Forward-block decision: y impact uses (S22, T22) inversion.
                # Following Klein (2000) eq. (33): when shocks are iid the
                # forward-looking impact reduces to
                #     L = -inv(Z22) inv(S22) Q2C  (state-impact-free part).
                try:
                    L = -Z22_inv @ np.linalg.solve(S22, Q2C)
                except np.linalg.LinAlgError:
                    L = np.zeros((n_fwd, n_u))
                # State response: N = inv(S11) (Q1C + T11 inv(Z11) ... ) — for iid u_t
                # the simplest correct expression for state-only impact is
                #     N = Z11 inv(S11) Q1C
                try:
                    N = (Z11 @ np.linalg.solve(S11, Q1C)).real
                except np.linalg.LinAlgError:
                    N = np.zeros((n_pre, n_u))
                L = L.real
            else:
                N = np.zeros((n_pre, 0))
                L = np.zeros((n_fwd, 0))
        else:
            # Indeterminacy (or near singularity)
            G = np.zeros((n_pre, n_pre))
            F = np.zeros((n_fwd, n_pre))
            N = np.zeros((n_pre, n_u))
            L = np.zeros((n_fwd, n_u))
    else:
        G = np.zeros((n_pre, n_pre))
        F = np.zeros((n_fwd, n_pre))
        N = np.zeros((n_pre, n_u))
        L = np.zeros((n_fwd, n_u))

    if strict and tuple(eu) != (1, 1):
        if eu[0] == 0:
            kind = "no stable solution" if n_unstable > n_fwd else "indeterminacy"
        else:
            kind = "indeterminacy (Z22 rank-deficient)"
        raise BlanchardKahnError(kind, n_unstable, n_fwd, eigvals_sorted)

    return KleinSolution(
        G=G, F=F, N=N, L=L,
        eu=tuple(eu),
        eigenvalues=eigvals_sorted,
    )


__all__ = ["klein_solve", "KleinSolution", "BlanchardKahnError"]
