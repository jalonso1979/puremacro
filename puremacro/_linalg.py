"""Internal linear-algebra primitives with rank-aware diagnostics.

Wraps the few ``np.linalg`` calls that show up in OLS / panel / SVAR
estimators so a singular ``X'X`` or non-PD ``Σ`` produces a message
that names *which* call failed and *why*, instead of an opaque
``LinAlgError: Singular matrix``.
"""
from __future__ import annotations

import numpy as np


def inv_xtx(X: np.ndarray, *, name: str = "OLS") -> np.ndarray:
    """Return ``(X'X)^{-1}`` or raise a diagnostic ``LinAlgError``.

    Uses Cholesky as the singularity gate, not ``np.linalg.inv``: the
    latter happily returns garbage on near-singular matrices (the
    inverse can have entries of order 1e13 with no exception raised),
    while Cholesky fails cleanly the moment ``X'X`` ceases to be
    numerically positive definite. The inverse is then assembled from
    the triangular factor.

    On failure, reports the rank of ``X`` and the columns most aligned
    with its null space — i.e. the regressors most likely to be
    collinear.

    Parameters
    ----------
    X : (n, k) regressor matrix.
    name : caller label that gets prefixed onto the error message.
    """
    XtX = X.T @ X
    chol_failed = False
    try:
        L = np.linalg.cholesky(XtX)
    except np.linalg.LinAlgError:
        chol_failed = True

    # Cholesky on its own is too permissive — LAPACK's potrf accepts a
    # matrix whose condition number is ~1e19, which is effectively
    # singular for OLS purposes. Reject if the triangular factor is
    # poorly conditioned (cond(L)^2 = cond(X'X); 1e-7 here ≈ cond > 1e14).
    if not chol_failed:
        diag = np.abs(np.diag(L))
        if diag.min() < 1e-6 * diag.max():
            chol_failed = True

    if chol_failed:
        n_cols = X.shape[1]
        rank = int(np.linalg.matrix_rank(X))
        if rank < n_cols:
            _, s, vh = np.linalg.svd(X, full_matrices=False)
            deficit = n_cols - rank
            null_dirs = vh[-deficit:]
            loadings = np.abs(null_dirs).sum(axis=0)
            culprits = np.argsort(loadings)[::-1][:deficit].tolist()
            raise np.linalg.LinAlgError(
                f"{name}: X'X is singular (rank {rank} of {n_cols}). "
                f"Columns most aligned with the null space: {culprits}. "
                "Drop a redundant regressor or use a regularised estimator."
            )
        raise np.linalg.LinAlgError(
            f"{name}: X'X is numerically singular at full rank {rank} "
            f"(condition number ≳ 1e14). Try rescaling regressors."
        )
    # (X'X)^{-1} = L^{-T} L^{-1}, computed via triangular solves.
    eye = np.eye(L.shape[0])
    L_inv = np.linalg.solve(L, eye)
    return L_inv.T @ L_inv


def safe_cholesky(
    A: np.ndarray,
    *,
    name: str = "Cholesky",
    jitter: float = 0.0,
) -> np.ndarray:
    """Lower-triangular Cholesky factor with a diagnostic ``LinAlgError``.

    If ``jitter > 0`` the call retries once with ``A + jitter·I``. On
    final failure, reports the smallest eigenvalue of the symmetrised
    matrix so the caller can see how far from PD they are.

    The call also rejects matrices that are *numerically* singular even
    when LAPACK's ``potrf`` accepts them: if the smallest pivot of the
    factor is more than 1e-7 times the largest (equivalent to
    ``cond(A) ≳ 1e14``), the matrix is treated as non-PD. This mirrors
    ``inv_xtx``'s conditioning check and gives consumers a single
    consistent threshold across the package — see ``ARCHITECTURE.md``
    on "Diagnostic errors over silent garbage".
    """
    chol_failed = False
    L = None
    try:
        L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        chol_failed = True
        if jitter > 0.0:
            try:
                L = np.linalg.cholesky(A + jitter * np.eye(A.shape[0]))
                chol_failed = False
            except np.linalg.LinAlgError:
                pass

    if L is not None and not chol_failed:
        # LAPACK potrf accepts factors whose smallest pivot is ~1e-12
        # while the largest is ~1; that is effectively a singular Σ for
        # SVAR / GMM purposes. Reject when the ratio drops below 1e-7
        # (≈ cond(A) > 1e14).
        diag = np.abs(np.diag(L))
        if diag.size and diag.min() < 1e-7 * diag.max():
            chol_failed = True

    if not chol_failed:
        assert L is not None  # guarded above: chol_failed=True whenever L is None
        return L

    if A.size:
        eigs = np.linalg.eigvalsh((A + A.T) / 2.0)
        min_eig = float(eigs.min())
    else:
        min_eig = float("nan")
    raise np.linalg.LinAlgError(
        f"{name}: matrix is not positive definite "
        f"(min eigenvalue ≈ {min_eig:.3e}, shape {A.shape}). "
        "Likely causes: too few observations for the dimension, "
        "perfectly correlated variables, or a singular reduced-form Σ."
    )


__all__ = ["inv_xtx", "safe_cholesky"]


def markov_stationary_gth(P: np.ndarray) -> np.ndarray:
    """Stationary distribution of a row-stochastic matrix by GTH elimination.

    Grassmann, Taksar and Heyman (1985): Gaussian elimination on the censored chain,
    with each pivot taken as the sum of the off-diagonal probabilities instead of
    ``1 - p_kk``. Only non-negative numbers are added, so there is no cancellation and
    the result is accurate even when states communicate through probabilities far below
    machine epsilon, where the eigenvector of ``P.T`` for the eigenvalue nearest 1 is an
    arbitrary mixture (sometimes with negative entries).

    ``P`` must already be validated as square, non-negative and row-stochastic. A chain
    with one closed communicating class gets all its mass there; with several the
    stationary distribution is not unique and a ``ValueError`` names the classes.
    """
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    A = P.copy()
    for k in range(n - 1, 0, -1):
        s = A[k, :k].sum()
        if s <= 0.0:
            return _stationary_on_closed_class(P)
        A[:k, k] /= s
        A[:k, :k] += np.outer(A[:k, k], A[k, :k])
    pi = np.zeros(n)
    pi[0] = 1.0
    for k in range(1, n):
        pi[k] = pi[:k] @ A[:k, k]
    return pi / pi.sum()


def _stationary_on_closed_class(P: np.ndarray) -> np.ndarray:
    """GTH on the unique closed class of a reducible chain; raise if there is more than one."""
    from scipy.sparse.csgraph import connected_components

    adjacency = P > 0.0
    n_comp, label = connected_components(adjacency, directed=True, connection="strong")
    closed = [c for c in range(n_comp)
              if not adjacency[np.ix_(label == c, label != c)].any()]
    if len(closed) != 1:
        groups = [np.flatnonzero(label == c).tolist() for c in closed]
        raise ValueError(
            f"stationary distribution is not unique: the chain has {len(closed)} closed "
            f"classes of states {groups}"
        )
    members = np.flatnonzero(label == closed[0])
    pi = np.zeros(P.shape[0])
    if members.size == 1:
        pi[members] = 1.0
    else:
        pi[members] = markov_stationary_gth(P[np.ix_(members, members)])
    return pi
