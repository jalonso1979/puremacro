"""Generalised Schur decomposition with a reordering that survives LAPACK.

Every QZ-based solver here (``gensys``, ``klein``, the fertility model) needs
the same thing: ``scipy.linalg.ordqz`` with the stable generalised eigenvalues
moved to the top-left. On some LAPACK builds that reordering fails on pencils
that are not remotely ill-conditioned:

    ValueError: Reordering of (A, B) failed because the transformed matrix
    pair (A, B) would be too far from generalized Schur form; the problem is
    very ill-conditioned.

The four-variable block-diagonal test model raises it under the OpenBLAS that
ships with the numpy 2.5 wheels, while its generalised eigenvalues are
0.5, 0.6, 1.11 and 1.18 — cleanly separated, nowhere near degenerate. The same
call succeeds against Accelerate and against the LAPACK on the CI runners, so
which models a machine can solve has been a property of its BLAS.

What isolates it is that the *real* reordering (``dtgsen``) handles the same
pencil and the same selection without complaint; only the complex one
(``ztgsen``) refuses. So a complex reordering that fails is retried in real
arithmetic. That is not a downgrade: the real generalised Schur form spans the
same deflating subspace, ``ordqz`` moves 2x2 blocks whole so a conjugate pair
is never split across the stable/unstable partition, and every consumer here
ends in ``.real`` anyway. On a healthy LAPACK the retry never runs.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg
from numpy.linalg import LinAlgError

__all__ = ["ordqz_sorted"]

_REORDER_FAILURE = "Reordering of (A, B) failed"


def ordqz_sorted(A, B, sort_ab, *, output: str = "real"):
    """``scipy.linalg.ordqz(A, B, sort=sort_ab)``, retried in real arithmetic.

    Parameters
    ----------
    A, B : (n, n) array_like
        The matrix pencil.
    sort_ab : callable
        ``sort_ab(alpha, beta) -> bool``, exactly as ``ordqz`` takes it.
    output : {"real", "complex"}
        Requested arithmetic. A failed ``"complex"`` reordering falls back to
        ``"real"``; a failed ``"real"`` one has nowhere left to go.

    Returns
    -------
    (S, T, alpha, beta, Q, Z), matching ``scipy.linalg.ordqz``. After a
    fallback these are real arrays rather than complex ones — same
    decomposition, same deflating subspace, different storage.

    Raises
    ------
    numpy.linalg.LinAlgError
        If the reordering fails in both arithmetics, which is the case where
        the conditioning complaint is real.
    """
    try:
        return scipy.linalg.ordqz(A, B, sort=sort_ab, output=output)
    except ValueError as exc:
        if _REORDER_FAILURE not in str(exc):
            raise
        if output != "complex":
            raise LinAlgError(_diagnosis(A, B)) from exc
        complex_failure = exc

    try:
        return scipy.linalg.ordqz(A, B, sort=sort_ab, output="real")
    except ValueError as exc:
        if _REORDER_FAILURE not in str(exc):
            raise
        raise LinAlgError(_diagnosis(A, B)) from complex_failure


def _diagnosis(A, B) -> str:
    """Say whether the pencil really is degenerate, since LAPACK will not."""
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            alpha, beta = scipy.linalg.eig(
                np.asarray(A, dtype=float), np.asarray(B, dtype=float),
                left=False, right=False, homogeneous_eigvals=True,
            )
        both_vanish = int(np.sum((np.abs(alpha) < 1e-12) & (np.abs(beta) < 1e-12)))
    except Exception:      # diagnosis must never mask the failure it explains
        both_vanish = 0
    tail = (
        f" {both_vanish} eigenvalue(s) have both alpha and beta vanishing, "
        f"which leaves the stable/unstable split genuinely undefined."
        if both_vanish else
        " Its generalised eigenvalues look well separated, so this is more "
        "likely a LAPACK build problem than a property of the model."
    )
    return (
        "the generalised Schur form of this pencil could not be reordered, in "
        "complex or in real arithmetic." + tail
    )
