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

# ``SingularPencilError`` and ``_check_regular`` below are the success-path
# counterpart of ``_diagnosis``: the same "both sides vanish" test, but run on
# every solve rather than only after LAPACK has already refused to reorder.

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


class SingularPencilError(LinAlgError):
    """The pencil ``(A, B)`` is singular: ``det(A - lambda B) == 0`` for every
    ``lambda``.

    A generalised eigenvalue whose diagonal block vanishes in *both* Schur
    factors ("coincident zeros", in Sims's phrase) means the pencil has no
    well-defined spectrum, so no stable/unstable split exists and no QZ-based
    solver can answer. In a DSGE model it almost always means a variable
    appears in no equation, or two equations are linearly dependent.

    Carries ``n_singular`` (how many eigenvalues vanish) and ``n_total``.
    """

    def __init__(self, message: str, n_singular: int, n_total: int):
        self.n_singular = n_singular
        self.n_total = n_total
        super().__init__(message)


def _check_regular(S, T, *, where: str, tol: float = 1e-12) -> None:
    """Raise if the pencil behind the generalised Schur factors is singular.

    A generalised eigenvalue is the ratio of a diagonal block of ``S`` to the
    matching block of ``T``. A block that vanishes in *both* factors is not
    "eigenvalue zero" and not "eigenvalue infinity" — it is the signature of a
    *singular* pencil, where ``det(A - lambda B)`` vanishes identically and
    every lambda is an eigenvalue. Neither Klein nor gensys can classify such
    a direction as stable or unstable, and both would otherwise hand back a
    plausible-looking matrix (typically a zero row) for a variable that the
    model does not restrict at all.

    The test reads the Schur factors rather than ``ordqz``'s ``alpha`` /
    ``beta``, because LAPACK scales each ``(alpha_i, beta_i)`` pair by an
    arbitrary common factor: on SW07 one pair comes back as
    ``alpha = -0.50+1.21j, beta = 1.0e17`` while every entry of the model's
    own matrices is below 7. Magnitudes of ``alpha`` and ``beta`` are
    therefore not comparable across pairs, whereas ``S`` and ``T`` are
    unitarily equivalent to ``A`` and ``B`` and so carry their scale.

    Thresholds are relative to ``max|S|`` and ``max|T|`` so the verdict
    survives multiplying the whole system by a constant; an absolute
    threshold would fire on any model written in small enough units.

    With ``output="real"`` the factors are quasi-triangular, so the walk below
    keeps a 2x2 conjugate-pair block whole: its diagonal entries can each be
    negligible while the block itself is perfectly nonsingular.

    Parameters
    ----------
    S, T : (n, n) ndarrays
        Generalised Schur factors, as returned by ``ordqz``.
    where : str
        Name of the calling solver, quoted in the error message.
    tol : float
        Relative vanishing threshold.

    Raises
    ------
    SingularPencilError
        (a subclass of ``numpy.linalg.LinAlgError``) when at least one block
        vanishes in both factors.
    """
    S = np.asarray(S)
    T = np.asarray(T)
    n = S.shape[0]
    if n == 0:
        return
    s_scale = float(np.abs(S).max()) or 1.0
    t_scale = float(np.abs(T).max()) or 1.0
    if n > 1:
        coupled = (np.abs(np.diag(S, -1)) + np.abs(np.diag(T, -1))) > 0.0
    else:
        coupled = np.zeros(0, dtype=bool)

    n_bad = 0
    i = 0
    while i < n:
        j = i + 2 if (i + 1 < n and coupled[i]) else i + 1
        blk_s = float(np.abs(S[i:j, i:j]).max())
        blk_t = float(np.abs(T[i:j, i:j]).max())
        if blk_s <= tol * s_scale and blk_t <= tol * t_scale:
            n_bad += j - i
        i = j
    if n_bad == 0:
        return
    raise SingularPencilError(
        f"{where}: the matrix pencil (A, B) is singular — {n_bad} of {n} "
        f"generalised eigenvalues have a vanishing diagonal block in BOTH "
        f"Schur factors (|S block| <= {tol:g}*max|S| and |T block| <= "
        f"{tol:g}*max|T|), so det(A - lambda B) == 0 for every lambda. The "
        f"stable/unstable split is undefined, and the affected directions are "
        f"unrestricted by the model: every value satisfies the equations "
        f"equally well, so there is no solution to report. This normally means "
        f"a variable appears in no equation, or two equations are linearly "
        f"dependent.",
        n_bad, int(n),
    )


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
