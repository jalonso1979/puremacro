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
Stability is decided by ``|β| < div |α|``, with ``div`` a keyword argument
of :func:`klein_solve` (default ``1 + 1e-8``, matching :func:`gensys` and
in the spirit of Dynare's ``qz_criterium``). The same predicate both
*orders* the QZ and *counts* the stable block, so the two can never
disagree.

Before 2.5.0 the threshold was a hard ``|β/α| < 1.0``, which classified an
exact unit root as unstable and made every unit-root model — random-walk
technology, permanent shocks, I(1) debt — unsolvable through Klein while
``gensys`` in the same package solved them. It also made the answer jump
discontinuously between ``rho = 1 - 1e-12`` (solved) and ``rho = 1.0``
(refused). Pass ``div=1.0`` to recover the old knife edge.

Admitting a unit root means the returned ``G`` is *not* stationary: it has
an eigenvalue on the unit circle, so unconditional moments do not exist.
:func:`klein_solve` emits a ``RuntimeWarning`` naming the offending roots
when that happens, rather than letting a downstream variance decomposition
quietly divide by an infinity.

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

Post-solve verification
~~~~~~~~~~~~~~~~~~~~~~~
Whatever the QZ returns, the pair ``(G, F)`` is checked against the
equilibrium condition it must satisfy,

    (A1 + A2 F) G = B1 + B2 F      and      (A1 + A2 F) N - B2 L = C

with the residual of each *equation* divided by that equation's own row
scale ``max|A row| + max|B row|``. The achieved value is reported on
``KleinSolution.residual``; a value above ``1e-6`` raises
:class:`KleinResidualError` rather than returning a matrix that does not
solve the model. Row scaling matters: a bar scaled by the largest entry
anywhere in the system lets a badly scaled row hide a 33%-of-row-scale
violation, which is exactly how a LAPACK accuracy loss on an
ill-conditioned pencil used to be returned as a confident answer.

Reference
---------
Klein, P. (2000). Using the generalised Schur form to solve a
multivariate linear rational expectations model. JEDC 24, 1405-1423.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ._qz import ordqz_sorted, _check_regular

# Row-scaled equilibrium-residual thresholds. Measured margins: SW07 at the
# posterior mode achieves 7e-15, and over 1300 random determinate systems the
# median is 4e-16 with a worst case of 4e-10 — so the refusal bar has three
# orders of headroom over anything the package can currently produce, while
# the 0.33 violation that a LAPACK breakdown on an ill-conditioned pencil
# produces is five orders above it.
_RESID_FALLBACK = 1e-6   # above this, re-solve F from the Sylvester equation
_RESID_WARN     = 1e-8   # above this, say so
_RESID_REFUSE   = 1e-6   # above this after the fallback, refuse to answer


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
                                 (1, 1) ⇒ unique stable solution;
                                 (1, 0) ⇒ a stable solution exists but is not
                                          unique (indeterminacy);
                                 (0, 0) ⇒ no stable solution.
    eigenvalues : sorted generalised eigenvalues of (B, A) — for diagnostics.
    residual : float — largest row-scaled violation of the equilibrium
        conditions ``(A1 + A2 F) G = B1 + B2 F`` and
        ``(A1 + A2 F) N - B2 L = C`` by the returned matrices, each equation
        divided by its own scale ``max|A row| + max|B row|``. 0.0 when no
        solution was computed (``eu != (1, 1)``). Callers that need a
        guarantee can assert on it; ``klein_solve`` itself refuses to return
        anything above 1e-6.
    """
    G: np.ndarray
    F: np.ndarray
    N: np.ndarray
    L: np.ndarray
    eu: tuple
    eigenvalues: np.ndarray
    residual: float = 0.0


def _select_stable(alpha: np.ndarray, beta: np.ndarray,
                   div: float = 1.0) -> np.ndarray:
    """Boolean array marking stable generalised eigenvalues ``|beta| < div |alpha|``.

    Written in homogeneous form rather than as ``|beta/alpha| < div`` so that
    ``alpha == 0`` (an infinite generalised eigenvalue) falls out as unstable
    without a special case and without a division. This is *the same
    expression* that ``klein_solve`` hands to ``ordqz`` as its sort predicate:
    ordering the pencil and counting the stable block must not be able to
    disagree, because the disagreement would be silent and would slice Z in
    the wrong place.
    """
    return np.abs(np.asarray(beta)) < div * np.abs(np.asarray(alpha))


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


class KleinResidualError(RuntimeError):
    """Raised by :func:`klein_solve` when the matrices it computed do not
    satisfy the model's own equilibrium conditions.

    The Blanchard-Kahn count can be satisfied and the QZ can still break down
    — LAPACK loses accuracy on an ill-conditioned pencil, and the reordered
    Schur form no longer spans the stable deflating subspace. What comes back
    then is not a solution: it is a matrix that does not solve the equations.
    Since it cannot be made right, it is not returned.

    Carries ``residual`` (the largest row-scaled violation), ``tol`` (the bar)
    and ``row`` (the worst offending equation index).
    """

    def __init__(self, residual: float, tol: float, row: int, quantity: str):
        self.residual = float(residual)
        self.tol = float(tol)
        self.row = int(row)
        self.quantity = quantity
        super().__init__(
            f"klein_solve: the solution it computed violates the model's own "
            f"equilibrium condition. The largest row-scaled residual of "
            f"{quantity} is {residual:.3e} at equation {row} (tolerance "
            f"{tol:g}), where the residual of each equation is divided by that "
            f"equation's own scale max|A row| + max|B row|. The Blanchard-Kahn "
            f"count was satisfied, so this is a numerical breakdown of the QZ "
            f"decomposition — typically an ill-conditioned pencil, e.g. a "
            f"variable whose units differ from the rest of the model by many "
            f"orders of magnitude. Rescale the offending variable or equation "
            f"and re-solve; a matrix that does not satisfy the equations is "
            f"not returned."
        )


def klein_solve(
    A: np.ndarray,
    B: np.ndarray,
    n_pre: int,
    C: Optional[np.ndarray] = None,
    *,
    strict: bool = False,
    div: float = 1.0 + 1e-8,
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
    div : float, default 1 + 1e-8
        Stability threshold: a generalised eigenvalue is stable when
        ``|beta| < div * |alpha|``. The default admits an exact unit root
        into the stable block, which is what makes random-walk and other
        I(1) specifications solvable (``gensys`` uses the same convention,
        as does Dynare's ``qz_criterium``); the returned ``G`` is then
        non-stationary and a ``RuntimeWarning`` says so. Pass ``div=1.0``
        for the strict pre-2.5.0 behaviour, which classified a unit root as
        unstable.

    Returns
    -------
    KleinSolution. Inspect ``eu`` to confirm existence/uniqueness:
        eu = (1, 1) ⇒ unique stable solution.
        eu = (1, 0) ⇒ exists but indeterminate (multiple solutions);
                      too few unstable roots, or a rank-deficient Z22.
        eu = (0, 0) ⇒ no stable solution (too many unstable roots).

    Raises
    ------
    KleinResidualError
        When the computed matrices violate the equilibrium conditions by more
        than 1e-6 row-scaled, in both the closed-form and the Sylvester path.
        Raised regardless of ``strict``: this is a numerical breakdown, not a
        property of the model, and there is no honest soft answer to give.
    numpy.linalg.LinAlgError
        When the pencil (A, B) is singular — some generalised eigenvalue pair
        has both alpha and beta vanishing, so a variable is restricted by no
        equation and the stable/unstable split is undefined.
    BlanchardKahnError
        Under ``strict=True``, when ``eu != (1, 1)``.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    n = A.shape[0]
    n_fwd = n - n_pre
    if C is None:
        C = np.zeros((n, 0))
    C = np.asarray(C, dtype=float)
    n_u = C.shape[1]

    # Generalised Schur (QZ): A = Q S Z',  B = Q T Z'
    # We want stable eigenvalues at the top-left. The predicate handed to
    # ordqz here and the one `_select_stable` counts with below are the same
    # expression with the same `div`, on purpose.
    S, T, alpha, beta, Q, Z = ordqz_sorted(
        A, B, lambda a, b: abs(b) < div * abs(a),  # stable first
    )

    # A diagonal block vanishing in both Schur factors means the pencil is
    # singular: some variable is restricted by no equation at all, every value
    # of it solves the model, and F = 0 is not "the" answer — it is one of
    # infinitely many.
    _check_regular(S, T, where="klein_solve")

    # Diagnostics
    with np.errstate(divide="ignore", invalid="ignore"):
        eigvals = np.where(alpha != 0, beta / alpha, np.inf)
    eigvals_sorted = np.sort(np.abs(eigvals))
    stable_mask = _select_stable(alpha, beta, div)
    n_stable = int(np.sum(stable_mask))
    n_unstable = n - n_stable
    if not bool(np.all(stable_mask[:n_stable])):
        raise np.linalg.LinAlgError(
            "klein_solve: the QZ reordering did not put the stable generalised "
            f"eigenvalues first — {n_stable} of {n} are stable under "
            f"|beta| < {div!r}*|alpha|, but they are not the leading diagonal "
            "entries. Partitioning Z into stable/unstable blocks would take "
            "the wrong columns, so the solve is refused rather than answered "
            "from the wrong subspace."
        )

    eu = [0, 0]
    # Blanchard-Kahn: number of unstable eigenvalues must equal n_fwd.
    #   n_unstable < n_fwd -> too few unstable roots: stable solutions exist,
    #                         but a continuum of them (indeterminacy) -> (1, 0)
    #   n_unstable > n_fwd -> no stable solution at all                -> (0, 0)
    # Before 2.5.0 both collapsed to (0, 0), contradicting this function's own
    # docstring and telling a user with an indeterminate model that their model
    # has no solution — the opposite diagnosis, and the opposite remedy.
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

            # Verify the PAIR (G, F) against the condition every solution
            # must satisfy, collecting the x_t terms of A E_t z_{t+1} = B z_t
            # after substituting y_t = F x_t and x_{t+1} = G x_t:
            #
            #   (A1 + A2 F) G = B1 + B2 F            (residual r)
            #
            # A1/A2 split A by *column* (variable), and all n rows
            # (equations) enter: there is no general correspondence
            # between "the last n_fwd rows" and "the control equations",
            # so the row-subset check used before 1.2.0 could pass an F
            # that solves nothing. If the residual is too large — the
            # degenerate case where the QZ stable block does not cleanly
            # align with the predetermined subspace, as in SW07's many
            # static-control rows — recover F from the Sylvester
            # equation instead.
            #
            # Each equation's residual is divided by that equation's OWN
            # scale, max|A row| + max|B row|. Before 2.5.0 the bar was
            # scaled by the largest entry anywhere in A or B, which is not
            # a property of the violated equation at all: on a pencil with
            # a 1e9 entry it let a violation worth 33% of its own row's
            # scale pass as "0.5 < 1500". G itself was never checked.
            A1 = A[:, :n_pre]
            A2 = A[:, n_pre:]
            B1 = B[:, :n_pre]
            B2 = B[:, n_pre:]
            rowscale = np.abs(A).max(axis=1) + np.abs(B).max(axis=1)
            rowscale = np.where(rowscale > 0.0, rowscale, 1.0)[:, None]

            def _scaled(r: np.ndarray) -> tuple[float, int]:
                """(worst row-scaled |entry|, the row it sits in)."""
                if r.size == 0:
                    return 0.0, 0
                scaled = np.abs(r) / rowscale
                flat = int(np.argmax(scaled))
                return float(scaled.flat[flat]), flat // r.shape[1]

            def _state_resid(F_: np.ndarray) -> tuple[float, int]:
                if n_pre == 0:
                    return 0.0, 0
                return _scaled((A1 + A2 @ F_) @ G - B1 - B2 @ F_)

            residual, worst_row = _state_resid(F)
            worst_what = "(A1 + A2 F) G - B1 - B2 F"
            if residual > _RESID_FALLBACK and n_fwd > 0 and n_pre > 0:
                F_syl = _solve_F_sylvester(A, B, G, n_pre=n_pre)
                resid_syl, row_syl = _state_resid(F_syl)
                # lstsq happily returns a least-squares fit to an
                # inconsistent system, so take the fallback only when it
                # actually improves matters — and check it either way.
                if resid_syl < residual:
                    F, residual, worst_row = F_syl, resid_syl, row_syl

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
                M_shock = np.hstack([A1 + A2 @ F, -B2])
                try:
                    NL = np.linalg.solve(M_shock, C)
                except np.linalg.LinAlgError:
                    # Singular impact system (redundant equations): the
                    # least-squares solution is the informative answer.
                    NL, *_ = np.linalg.lstsq(M_shock, C, rcond=None)
                N = np.asarray(NL[:n_pre]).real
                L = np.asarray(NL[n_pre:]).real
                # ...and it is only informative if it solves the system,
                # which lstsq does not promise. Check it too.
                r_shock, row_shock = _scaled(
                    (A1 + A2 @ F) @ N - B2 @ L - C
                )
                if r_shock > residual:
                    residual, worst_row = r_shock, row_shock
                    worst_what = "(A1 + A2 F) N - B2 L - C"
            else:
                N = np.zeros((n_pre, 0))
                L = np.zeros((n_fwd, 0))

            if residual > _RESID_REFUSE:
                raise KleinResidualError(
                    residual, _RESID_REFUSE, worst_row, worst_what,
                )
            if residual > _RESID_WARN:
                warnings.warn(
                    f"klein_solve: the returned solution satisfies the model's "
                    f"own equilibrium condition {worst_what} = 0 only to a "
                    f"row-scaled residual of {residual:.3e}, at equation "
                    f"{worst_row} — far above the ~1e-15 a well-conditioned "
                    f"pencil achieves. The decision rules are probably "
                    f"accurate to only a few digits. Check the scaling of the "
                    f"model's variables and equations.",
                    RuntimeWarning, stacklevel=2,
                )

            # An admitted unit root (|lambda| in [1, div)) means G is not
            # stationary: unconditional moments do not exist and any variance
            # decomposition computed from this G is meaningless. Say so.
            n_unit = int(np.sum((eigvals_sorted >= 1.0)
                                & (eigvals_sorted < div)))
            if n_unit:
                warnings.warn(
                    f"klein_solve: {n_unit} generalised eigenvalue(s) lie in "
                    f"[1, div={div!r}) and were admitted to the stable block, "
                    f"so the returned state transition G has a root on or "
                    f"outside the unit circle. The solution is a valid "
                    f"non-explosive rational-expectations path, but the model "
                    f"is NOT stationary: unconditional moments, the "
                    f"unconditional variance decomposition and the Lyapunov "
                    f"variance do not exist for it. Pass div=1.0 to refuse "
                    f"such models instead.",
                    RuntimeWarning, stacklevel=2,
                )
        else:
            # Z22 rank-deficient: the count is right but the stable subspace
            # is not a graph over the states, so the solution is not unique.
            # Stable paths exist, so this is indeterminacy, eu = (1, 0).
            residual = 0.0
            G = np.zeros((n_pre, n_pre))
            F = np.zeros((n_fwd, n_pre))
            N = np.zeros((n_pre, n_u))
            L = np.zeros((n_fwd, n_u))
    else:
        # Too FEW unstable roots is indeterminacy — stable solutions exist,
        # there is simply a continuum of them — so existence holds and only
        # uniqueness fails. Too many is genuine non-existence.
        eu[0] = 1 if n_unstable < n_fwd else 0
        residual = 0.0
        G = np.zeros((n_pre, n_pre))
        F = np.zeros((n_fwd, n_pre))
        N = np.zeros((n_pre, n_u))
        L = np.zeros((n_fwd, n_u))

    if strict and tuple(eu) != (1, 1):
        if n_unstable > n_fwd:
            kind = "no stable solution"
        elif n_unstable < n_fwd:
            kind = "indeterminacy"
        else:
            kind = "indeterminacy (Z22 rank-deficient)"
        raise BlanchardKahnError(kind, n_unstable, n_fwd, eigvals_sorted)

    return KleinSolution(
        G=G, F=F, N=N, L=L,
        eu=tuple(eu),
        eigenvalues=eigvals_sorted,
        residual=float(residual),
    )


__all__ = [
    "klein_solve", "KleinSolution", "BlanchardKahnError", "KleinResidualError",
]
