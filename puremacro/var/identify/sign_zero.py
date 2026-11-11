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

from collections.abc import Mapping
from numbers import Integral, Real
from typing import Iterable

import numpy as np

from ..._linalg import safe_cholesky
from ._results import SignZeroResult


def _constraint_pair(pair, n: int, name: str) -> tuple[int, int]:
    """Validate coordinates without accepting negative NumPy indexing."""
    try:
        i, j = pair
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} entries must be (row, column) pairs") from exc
    if any(isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral)
           or not 0 <= int(k) < n for k in (i, j)):
        raise ValueError(f"{name} indices must be integers in [0, {n})")
    return int(i), int(j)


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
        Only impact restrictions are supported, so ``A_list`` is not used.
        ``Sigma`` must be a finite real symmetric positive-definite matrix.
    zero_constraints : iterable of (i, j) pairs
        Impose B0[i, j] = 0 (variable i has zero impact response to shock j).
        Indices are zero-based. Numerical zeros are checked relative to the
        residual standard deviation of variable i, with tolerance 1e-8.
    sign_constraints : dict {(i, j): +1 or -1}
        Impose sign(B0[i, j]) = +1 or -1; zero is not an unrestricted entry.
    n_draws : int
        Nonnegative integer maximum number of Haar draws to try.
    rng : np.random.Generator
        Random source.

    Returns
    -------
    SignZeroResult
        Frozen dataclass. ``success`` is ``True`` iff a draw satisfying
        every constraint was found within ``n_draws`` attempts; on
        failure, ``B0`` and ``Q`` are ``None``.

    Notes
    -----
    A failed finite search does not prove the restrictions infeasible. The
    sequential zero-projection search can miss feasible rotations; every
    returned success is checked against all restrictions and the covariance.
    A strict sign and a zero on the same entry are contradictory and return
    failure without drawing. Malformed constraints raise ``ValueError``.

    References
    ----------
    Arias, J., Rubio-Ramírez, J. and Waggoner, D. (2018). Inference
        based on structural vector autoregressions identified with sign
        and zero restrictions. Econometrica 86(2), 685-720.
    """
    if isinstance(n_draws, (bool, np.bool_)) or not isinstance(n_draws, Integral) or n_draws < 0:
        raise ValueError("n_draws must be a nonnegative integer")
    n_draws = int(n_draws)
    if np.iscomplexobj(Sigma):
        raise ValueError("Sigma must be a real covariance matrix")
    try:
        Sigma = np.asarray(Sigma, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Sigma must be a real covariance matrix") from exc
    if Sigma.ndim != 2 or Sigma.shape[0] == 0 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("Sigma must be a nonempty square covariance matrix")
    if not np.isfinite(Sigma).all():
        raise ValueError("Sigma must contain only finite values")
    if not np.allclose(Sigma, Sigma.T, rtol=1e-12, atol=0):
        raise ValueError("Sigma must be symmetric")
    n = Sigma.shape[0]
    try:
        raw_zeros = [] if zero_constraints is None else list(zero_constraints)
    except TypeError as exc:
        raise ValueError("zero_constraints must be an iterable of index pairs") from exc
    zeros = [_constraint_pair(pair, n, "zero_constraints") for pair in raw_zeros]
    if sign_constraints is not None and not isinstance(sign_constraints, Mapping):
        raise ValueError("sign_constraints must map index pairs to +1 or -1")
    signs = {}
    for pair, value in ({} if sign_constraints is None else sign_constraints).items():
        ij = _constraint_pair(pair, n, "sign_constraints")
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or value not in (-1, 1):
            raise ValueError("sign_constraints values must be +1 or -1")
        signs[ij] = int(value)
    P = safe_cholesky(Sigma, name="sign_zero_svar Σ")
    scales = np.sqrt(np.diag(Sigma))
    correlation = Sigma / scales[:, None] / scales[None, :]
    if any(pair in signs for pair in zeros):
        return SignZeroResult(success=False, B0=None, Q=None, n_draws_used=0)
    by_column: dict[int, list[tuple[int, int]]] = {}
    for (i, j), required in signs.items():
        by_column.setdefault(j, []).append((i, required))
    if rng is None:
        rng = np.random.default_rng(0)

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
        zero_ok = all(abs(B0[i, j]) / scales[i] < 1e-8 for (i, j) in zeros)
        if not zero_ok:
            continue

        # Choose one orientation per column. Flipping separately for each
        # entry can undo a previously satisfied restriction in that column.
        sign_ok = True
        for j, restrictions in by_column.items():
            signed_entries = np.array([required * B0[i, j] for i, required in restrictions])
            if np.all(signed_entries > 0):
                continue
            if np.all(signed_entries < 0):
                B0[:, j] *= -1
                Q[:, j] *= -1
            else:
                sign_ok = False
                break
        if not sign_ok:
            continue

        # Acceptance is an independent final check of the complete candidate.
        # Row normalization makes the zero/covariance checks invariant to a
        # change in measurement units and avoids absolute covariance floors.
        scaled_B0 = B0 / scales[:, None]
        if not (np.isfinite(Q).all() and np.isfinite(scaled_B0).all()):
            continue
        if not all(required * B0[i, j] > 0 for (i, j), required in signs.items()):
            continue
        if not all(abs(scaled_B0[i, j]) < 1e-8 for (i, j) in zeros):
            continue
        if not np.allclose(Q.T @ Q, np.eye(n), rtol=0, atol=1e-10):
            continue
        if not np.allclose(scaled_B0 @ scaled_B0.T, correlation, rtol=0, atol=1e-10):
            continue

        return SignZeroResult(success=True, B0=B0, Q=Q, n_draws_used=d + 1)

    return SignZeroResult(success=False, B0=None, Q=None, n_draws_used=n_draws)


__all__ = ["sign_zero"]
