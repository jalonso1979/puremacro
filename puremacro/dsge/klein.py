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

    Substituting ``y_t = F x_t`` and ``x_{t+1} = G x_t`` into *every*
    equation of ``A E_t z_{t+1} = B z_t`` and collecting the ``x_t``
    terms:

        (A1 + A2 @ F) @ G  =  B1 + B2 @ F

    where ``A1 = A[:, :n_pre]`` and ``A2 = A[:, n_pre:]`` (and likewise
    for B) split the matrices by *variable*, i.e. by column. Rearranging:

        A2 @ F @ G - B2 @ F  =  B1 - A1 @ G                       (*)

    This is a generalised Sylvester equation in F. Vectorising with the
    identity ``vec(M X N) = (N^T ⊗ M) vec(X)`` gives the linear system

        (G^T ⊗ A2 − I ⊗ B2) vec(F) = vec(B1 − A1 @ G)

    of ``n * n_pre`` equations in ``n_fwd * n_pre`` unknowns — over-
    determined, and solved by ``np.linalg.lstsq``, which also absorbs a
    rank-deficient ``A2`` (the SW07 case: half the control rows are
    static, so ``A2`` has 16 zero rows).

    All ``n`` rows enter. Rows of A and B are *equations*, while the
    ``n_pre`` / ``n_fwd`` split partitions *variables*, so there is no
    general correspondence between "the last n_fwd rows" and "the
    control equations" — restricting the system to ``rows[n_pre:]``
    drops equations that constrain F and can leave it underdetermined.

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

    A1 = A[:, :n_pre]
    A2 = A[:, n_pre:]
    B1 = B[:, :n_pre]
    B2 = B[:, n_pre:]

    # Vectorised form: (G^T kron A2 - I kron B2) vec(F) = vec(B1 - A1 G).
    # Fortran ordering matches np.kron's column-major vec convention.
    I_pre = np.eye(n_pre)
    M = np.kron(G.T, A2) - np.kron(I_pre, B2)          # (n*n_pre, n_fwd*n_pre)
    rhs_mat = B1 - A1 @ G                              # (n, n_pre)
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
            # Policy function. The stable subspace is spanned by the
            # first n_pre columns of Z, so a point on it is x_t = Z11 s,
            # y_t = Z21 s, giving y_t = Z21 inv(Z11) x_t. This is the
            # partner of the G formula two lines up (both read the
            # solution off the same Z11-parameterised subspace); the
            # -inv(Z22) Z21 form used before 1.2.0 belongs to a
            # different partition convention and does not satisfy the
            # model's own equilibrium condition — see
            # tests/test_dsge/test_klein_analytic.py.
            F = (Z21 @ Z11_inv).real

            # Verify F against the condition every solution must satisfy,
            # collecting the x_t terms of A E_t z_{t+1} = B z_t after
            # substituting y_t = F x_t and x_{t+1} = G x_t:
            #
            #   (A1 + A2 F) G = B1 + B2 F            (residual r)
            #
            # A1/A2 split A by *column* (variable), and all n rows
            # (equations) enter: there is no general correspondence
            # between "the last n_fwd rows" and "the control equations",
            # so the row-subset check used before 1.2.0 could pass an F
            # that solves nothing. If ||r||_inf is too large — the
            # degenerate case where the QZ stable block does not cleanly
            # align with the predetermined subspace, as in SW07's many
            # static-control rows — recover F from the Sylvester
            # equation instead.
            if n_fwd > 0:
                A1 = A[:, :n_pre]
                A2 = A[:, n_pre:]
                B1 = B[:, :n_pre]
                B2 = B[:, n_pre:]
                resid = (A1 + A2 @ F) @ G - B1 - B2 @ F
                # Relative tolerance: scale by the inputs' magnitude.
                scale = max(1.0, float(np.max(np.abs(A))), float(np.max(np.abs(B))))
                if float(np.max(np.abs(resid))) > 1e-6 * scale:
                    F = _solve_F_sylvester(A, B, G, n_pre=n_pre)

            # Shock loadings. Collecting the u_t terms of the same
            # substitution (E_t y_{t+1} = F(G x_t + N u_t), so u_{t+1}
            # drops out under E_t):
            #
            #   (A1 + A2 F) N - B2 L = C
            #
            # which is n equations in the n unknowns [N; L] per shock —
            # exactly determined, and correct whether the shock enters
            # through a state transition (N), contemporaneously through
            # a control equation (L), or both. The pre-1.2.0 expressions
            # returned L = 0 for the contemporaneous case.
            if n_u > 0:
                A1 = A[:, :n_pre]
                A2 = A[:, n_pre:]
                B2 = B[:, n_pre:]
                M_shock = np.hstack([A1 + A2 @ F, -B2])
                try:
                    NL = np.linalg.solve(M_shock, C)
                except np.linalg.LinAlgError:
                    # Singular impact system (redundant equations): the
                    # least-squares solution is the informative answer.
                    NL, *_ = np.linalg.lstsq(M_shock, C, rcond=None)
                N = np.asarray(NL[:n_pre]).real
                L = np.asarray(NL[n_pre:]).real
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
