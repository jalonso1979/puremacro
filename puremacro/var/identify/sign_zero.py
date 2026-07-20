"""Arias-Rubio Ramírez-Waggoner (2018) sign + zero restrictions SVAR.

Zero restrictions are imposed on the impact matrix B0 itself (i.e.,
restrictions on IRFs at horizon 0). The algorithm:

  1. Draw a Haar-uniform Q via QR of a standard Gaussian.
  2. Form B0 = chol(Σ_u) Q.
  3. For each (i, j) in zero_constraints, project B0's column j onto
     the subspace orthogonal to e_i, re-normalise the column, and
     re-orthonormalise the remaining columns of Q against the modified
     column.
  4. Re-form B0; check that the sign restrictions hold; accept if so.

For v1, only zero restrictions on B0 (horizon-0 impacts) are supported.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from ..._linalg import safe_cholesky
from ._results import SignZeroResult


def _enforce_zero_in_column(B0: np.ndarray, P: np.ndarray, i: int, j: int) -> np.ndarray | None:
    """Replace the j-th column of Q (= P^{-1} B0) so that B0[i, j] = 0,
    while keeping Q orthogonal. Returns the modified Q, or None if the
    constraint is infeasible (e.g., col is forced to zero).
    """
    n = B0.shape[0]
    # The current column of Q is q_j = P^{-1} B0[:, j].
    Q = np.linalg.solve(P, B0)
    q_j = Q[:, j].copy()
    # Constraint: e_i^T B0[:, j] = (P[i, :] @ q_j) = 0.
    # The constraint vector in q-space is c = P[i, :].
    c = P[i, :].copy()
    c_norm = np.linalg.norm(c)
    if c_norm < 1e-12:
        return None
    c_unit = c / c_norm
    # Project q_j onto the hyperplane orthogonal to c
    q_j_new = q_j - (q_j @ c_unit) * c_unit
    nq = np.linalg.norm(q_j_new)
    if nq < 1e-10:
        return None
    q_j_new = q_j_new / nq
    Q[:, j] = q_j_new
    # Re-orthonormalise the remaining columns of Q against the new q_j
    # via Gram-Schmidt
    for k in range(n):
        if k == j:
            continue
        v = Q[:, k]
        v = v - (v @ q_j_new) * q_j_new
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            return None
        Q[:, k] = v / nv
    # Final clean-up: re-QR to recover exact orthonormality (Gram-Schmidt
    # accumulates numerical drift across many constraints).
    Q_clean, _ = np.linalg.qr(Q)
    # Match column signs to Q so the constrained column survives
    for col in range(n):
        if Q_clean[:, col] @ Q[:, col] < 0:
            Q_clean[:, col] *= -1
    return Q_clean


def sign_zero(
    A_list,
    Sigma,
    *,
    zero_constraints: Iterable[tuple[int, int]] | None = None,
    sign_constraints: dict[tuple[int, int], int] | None = None,
    n_draws: int = 1000,
    rng: np.random.Generator | None = None,
) -> SignZeroResult:
    """Search for an orthogonal rotation Q of chol(Σ_u) such that
    B0 = chol(Σ_u) @ Q satisfies all zero and sign restrictions.

    Parameters
    ----------
    A_list, Sigma : VAR coefficients and reduced-form residual covariance.
    zero_constraints : iterable of (i, j) pairs
        Impose B0[i, j] = 0 (variable i has zero impact response to shock j).
    sign_constraints : dict {(i, j): +1 or -1}
        Impose sign(B0[i, j]) = +1 or -1.
    n_draws : int
        Maximum number of Haar draws to try.
    rng : np.random.Generator
        Random source.

    Returns
    -------
    SignZeroResult
        Frozen dataclass. ``success`` is ``True`` iff a draw satisfying
        every constraint was found within ``n_draws`` attempts; on
        failure, ``B0`` and ``Q`` are ``None``.

    References
    ----------
    Arias, J., Rubio-Ramírez, J. and Waggoner, D. (2018). Inference
        based on structural vector autoregressions identified with sign
        and zero restrictions. Econometrica 86(2), 685-720.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    Sigma = np.asarray(Sigma)
    P = safe_cholesky(Sigma, name="sign_zero_svar Σ")
    n = P.shape[0]
    zeros = list(zero_constraints or [])
    signs = dict(sign_constraints or {})

    for d in range(n_draws):
        # Haar Q via QR of Gaussian
        G = rng.standard_normal((n, n))
        Q, R = np.linalg.qr(G)
        Q = Q @ np.diag(np.sign(np.diag(R)))
        B0 = P @ Q

        # Enforce zeros one by one
        feasible = True
        for (i, j) in zeros:
            Q_new = _enforce_zero_in_column(B0, P, i, j)
            if Q_new is None:
                feasible = False
                break
            Q = Q_new
            B0 = P @ Q

        if not feasible:
            continue

        # Check zero constraints actually hold
        zero_ok = all(abs(B0[i, j]) < 1e-8 for (i, j) in zeros)
        if not zero_ok:
            continue

        # Check sign constraints
        sign_ok = True
        for (i, j), required in signs.items():
            if required > 0 and B0[i, j] < 0:
                # Try flipping the column
                B0[:, j] *= -1
                Q[:, j] *= -1
            elif required < 0 and B0[i, j] > 0:
                B0[:, j] *= -1
                Q[:, j] *= -1
            if (required > 0 and B0[i, j] <= 0) or (required < 0 and B0[i, j] >= 0):
                sign_ok = False
                break
        if not sign_ok:
            continue

        # Verify zeros still hold after sign flips (they should — column flip
        # preserves zero entries)
        zero_ok2 = all(abs(B0[i, j]) < 1e-8 for (i, j) in zeros)
        if not zero_ok2:
            continue

        return SignZeroResult(success=True, B0=B0, Q=Q, n_draws_used=d + 1)

    return SignZeroResult(success=False, B0=None, Q=None, n_draws_used=n_draws)


__all__ = ["sign_zero"]
