"""Sims (2002) gensys: QZ solution for linear RE models.

Solves systems of the form

    Γ_0 * z_t = Γ_1 * z_{t-1} + Ψ * ε_t + Π * η_t

where z_t is the full endogenous-variable vector, ε_t are exogenous
shocks, and η_t = z_t - E_{t-1} z_t are expectation errors (to be
determined by the solution).

This formulation is model-agnostic: variables need not be pre-classified
into predetermined / forward-looking. The QZ decomposition identifies
the stable and unstable modes automatically, and existence/uniqueness
are decided by Sims's two singular-value tests on the rotated Π.

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

from ._qz import ordqz_sorted, _check_regular


@dataclass(frozen=True)
class GensysSolution:
    """Output of gensys.

    Attributes
    ----------
    G      : (n, n) state-transition matrix — z_t = G z_{t-1} + ...
             Sims's ``G1``: the full decision rule, valid for *any*
             z_{t-1}, not only for one already on the stable manifold.
             Zeros when ``eu != (1, 1)``.
    Impact : (n, n_eps) shock-impact matrix  — z_t = ... + Impact ε_t
    eu     : tuple (exist, unique) — both 1 iff unique stable solution.
             (1, 0) is indeterminacy (a stable solution exists but is not
             unique); (0, 0) is non-existence. These are different
             diagnoses and are reported separately.
    eigenvalues : sorted |generalised eigenvalues| of (Γ_0, Γ_1).
    """
    G: np.ndarray
    Impact: np.ndarray
    eu: tuple
    eigenvalues: np.ndarray


def _truncated_svd(M: np.ndarray, tol: float):
    """SVD of ``M`` truncated at ``tol`` *relative* to its largest singular
    value, returned as ``(U_k, s_k, V_k)`` with ``M ≈ U_k diag(s_k) V_k^H``.

    Sims's ``gensys.m`` truncates at an absolute ``1e-6``, which makes the
    existence/uniqueness verdict depend on the units the model is written in.
    Scaling by ``s.max()`` makes the same test scale-invariant.
    """
    if M.size == 0:
        return (np.zeros((M.shape[0], 0), dtype=complex),
                np.zeros(0),
                np.zeros((M.shape[1], 0), dtype=complex))
    U, s, Vh = np.linalg.svd(M)
    k = int(np.sum(s > tol * s.max())) if s.max() > 0 else 0
    return U[:, :k], s[:k], Vh.conj().T[:, :k]


def gensys(
    Gamma0: np.ndarray,
    Gamma1: np.ndarray,
    Psi: np.ndarray,
    Pi: np.ndarray,
    *,
    div: float = 1.0 + 1e-8,
    tol: float = 1e-6,
    qz_criterium: float | None = None,
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
    tol    : relative truncation tolerance for the rank decisions inside the
             existence and uniqueness tests (Sims's ``realsmall``, here
             applied relative to the largest singular value rather than
             absolutely, so the verdict does not depend on the model's units).
    qz_criterium : float, optional
             Alias for div (stability threshold). If provided, overrides div.

    Returns
    -------
    GensysSolution

    Notes
    -----
    ``eu`` follows Sims's two singular-value tests, *not* the
    Blanchard-Kahn root count:

    * ``eu[0] = 1`` (existence) iff ``Q2 Ψ`` lies in the column space of
      ``Q2 Π`` — i.e. the expectation errors are able to absorb every shock
      that would otherwise excite an unstable mode.
    * ``eu[1] = 1`` (uniqueness) iff every direction of ``Π`` that shows up in
      the *stable* block is already pinned down by the unstable block; a
      "loose" direction is a sunspot.

    The root count ``n_eta == n_unstable`` is necessary but not sufficient
    for either, and its failure does not say which one failed. Before 2.5.0
    this function collapsed every count mismatch to ``eu = (0, 0)``, which
    reported indeterminacy — a model that *has* stable solutions, just too
    many — as non-existence.

    ``G`` and ``Impact`` are returned as zeros unless ``eu == (1, 1)``.

    Raises
    ------
    numpy.linalg.LinAlgError
        If the pencil ``(Γ_0, Γ_1)`` is singular (see
        ``_qz.SingularPencilError``) or if the QZ reordering could not be
        made consistent with the stability count.
    """
    Gamma0 = np.asarray(Gamma0, dtype=float)
    Gamma1 = np.asarray(Gamma1, dtype=float)
    Psi    = np.asarray(Psi,    dtype=float)
    Pi     = np.asarray(Pi,     dtype=float)

    n = Gamma0.shape[0]
    n_eta = Pi.shape[1]
    n_eps = Psi.shape[1]

    if qz_criterium is not None:
        div = float(qz_criterium)

    # Generalised Schur decomposition of (Γ_0, Γ_1):
    # Γ_0 = Q S Z^H,   Γ_1 = Q T Z^H
    # Sort so that stable generalised eigenvalues (|T_ii/S_ii| < div) come first.
    S, T, alpha, beta, Q, Z = ordqz_sorted(
        Gamma0, Gamma1,
        lambda a, b: (abs(b) < div * abs(a)),  # stable (|b/a|<div) first
        output="complex",
    )

    # A diagonal block that vanishes in both S and T makes the whole spectrum
    # meaningless; refuse rather than hand back a zero row for a variable that
    # no equation restricts.
    _check_regular(S, T, where="gensys")

    # Generalised eigenvalues |β/α|, for reporting only. The alpha guard is
    # relative: an absolute floor would call every eigenvalue infinite once
    # the system is written in small enough units.
    a_abs = np.abs(alpha)
    a_scale = float(a_abs.max()) or 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        eigvals = np.where(a_abs > np.finfo(float).eps * a_scale,
                           np.abs(beta / alpha), np.inf)
    eigvals_sorted = np.sort(eigvals)

    # Count stability with the SAME predicate that ordered the pencil. Using a
    # different rule here (the pre-2.5.0 code compared the reported |beta/alpha|
    # against div after an absolute 1e-12 guard on |alpha|) lets the count
    # disagree with the ordering, and then the Z1/Z2 slice below is taken in
    # the wrong place.
    stable = np.abs(beta) < div * np.abs(alpha)
    n_stable = int(np.sum(stable))
    n_unstable = n - n_stable
    if not bool(np.all(stable[:n_stable])):
        raise np.linalg.LinAlgError(
            "gensys: the QZ reordering did not put the stable generalised "
            f"eigenvalues first — {n_stable} are stable under |beta| < "
            f"{div!r}*|alpha| but they are not the leading diagonal entries. "
            "Slicing Z into stable/unstable blocks would silently take the "
            "wrong columns, so the solve is refused."
        )

    # Rotated expectation errors and shocks.
    QH = Q.conj().T
    QH_Pi = QH @ Pi              # (n, n_eta)
    QH_Psi = QH @ Psi            # (n, n_eps)
    Q1_Pi,  Q2_Pi  = QH_Pi[:n_stable, :],  QH_Pi[n_stable:, :]
    Q1_Psi, Q2_Psi = QH_Psi[:n_stable, :], QH_Psi[n_stable:, :]

    # --- Existence and uniqueness (Sims 2002, section 4) --------------------
    #
    # Premultiplying by Q^H and writing w_t = Z^H z_t gives
    #     S w_t = T w_{t-1} + Q^H Psi eps + Q^H Pi eta.
    # Stability requires the unstable block w2 to be identically zero, and
    # setting it to zero in its own equation leaves
    #     0 = Q2 Psi eps + Q2 Pi eta.
    # A solution EXISTS iff Q2 Psi lies in col(Q2 Pi), so that the expectation
    # errors can absorb every shock. It is UNIQUE iff no direction of Pi that
    # appears in the stable block is left free once the unstable block has
    # been satisfied — Sims's "loose endogenous errors" test.
    #
    # Both are rank statements, and neither reduces to the Blanchard-Kahn
    # count n_eta == n_unstable: that count can hold while existence fails
    # (a rank-deficient Q2 Pi), and it can fail while a perfectly good — if
    # non-unique — stable solution exists (indeterminacy).
    U_eta, s_eta, V_eta = _truncated_svd(Q2_Pi, tol)
    _, _, V_eta1 = _truncated_svd(Q1_Pi, tol)

    if Q2_Psi.size == 0:
        exists = True
    else:
        unabsorbed = Q2_Psi - U_eta @ (U_eta.conj().T @ Q2_Psi)
        norm_q2psi = float(np.linalg.norm(Q2_Psi))
        exists = bool(np.linalg.norm(unabsorbed) <= tol * max(norm_q2psi, 1e-300))

    if V_eta1.shape[1] == 0:
        unique = True
    else:
        loose = V_eta1 - V_eta @ (V_eta.conj().T @ V_eta1)
        # V_eta1's columns are orthonormal, so this norm is already unitless.
        unique = bool(np.linalg.norm(loose) <= tol * np.sqrt(V_eta1.shape[1]))

    eu = (int(exists), int(exists and unique))

    if eu != (1, 1):
        return GensysSolution(
            G=np.zeros((n, n)),
            Impact=np.zeros((n, n_eps)),
            eu=eu,
            eigenvalues=eigvals_sorted,
        )

    # --- Decision rule ------------------------------------------------------
    #
    # Sims's tmat maps the full rotated state onto the stable block after
    # substituting out the expectation errors:
    #
    #     tmat = [ I_ns  |  -(Q1 Pi) pinv(Q2 Pi) ]
    #
    # and the transition in Schur coordinates is
    #
    #     [ tmat S ] w_t = [ tmat T ] w_{t-1} + [ tmat Q^H Psi ] eps
    #     [ 0   I  ]       [   0    ]           [       0      ]
    #
    # whose top-left block of the left-hand side is exactly S11 (S is upper
    # triangular, so tmat S has S11 in its leading n_stable columns) and whose
    # bottom row forces w2_t = 0. Rotating back with Z:
    #
    #     G      = Z1 inv(S11) (tmat T) Z^H
    #     Impact = Z1 inv(S11) (tmat Q^H Psi)
    #
    # Before 2.5.0 the first line read `G = Z1 inv(S11) T11 Z1^H`, which keeps
    # only the [tmat T][:, :n_stable] = T11 block and drops [tmat T][:, n_stable:]
    # — i.e. it is the correct rule composed with the orthogonal projector onto
    # span(Z1). The two agree on any z_{t-1} that already lies on the stable
    # manifold (so IRFs seeded from Impact, the Lyapunov variance and every
    # autocovariance were unaffected), and they have the same spectrum, but
    # they differ on an arbitrary z_{t-1}: the projected form does not satisfy
    # the model's own equations off the manifold.
    S11 = S[:n_stable, :n_stable]
    Z1 = Z[:, :n_stable]

    if n_unstable == 0:
        tmat = np.eye(n_stable, dtype=complex)
    else:
        # Truncated pseudo-inverse of Q2 Pi. Uniqueness above has already
        # established that it has full row rank, so this is the genuine
        # inverse whenever Q2 Pi is square and nonsingular.
        if s_eta.size:
            pinv_Q2_Pi = V_eta @ ((1.0 / s_eta)[:, None] * U_eta.conj().T)
        else:
            pinv_Q2_Pi = np.zeros((n_eta, n_unstable), dtype=complex)
        tmat = np.hstack([np.eye(n_stable, dtype=complex),
                          -(Q1_Pi @ pinv_Q2_Pi)])

    # inv(S11) applied on the left, via a solve rather than an explicit inverse.
    G = (Z1 @ np.linalg.solve(S11, tmat @ T) @ Z.conj().T).real
    Impact = (Z1 @ np.linalg.solve(S11, tmat @ QH_Psi)).real

    return GensysSolution(
        G=G,
        Impact=Impact,
        eu=eu,
        eigenvalues=eigvals_sorted,
    )


__all__ = ["gensys", "GensysSolution"]
