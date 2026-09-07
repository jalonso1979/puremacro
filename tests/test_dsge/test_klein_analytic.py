"""Klein solver vs. closed-form solutions.

``klein_solve`` returns four matrices — G, F, N, L — and until 1.2.0 only
G was checked against an analytic benchmark. The other three were
validated indirectly, through SW07, whose equation ordering happens to
mask two errors:

* ``F`` was computed as ``-inv(Z22) @ Z21``, which is not the partner of
  the ``G = Z11 inv(S11) T11 inv(Z11)`` the same function returns. The
  consistent formula reads the solution off the same Z11-parameterised
  stable subspace: ``F = Z21 @ inv(Z11)``.
* The guard that was supposed to catch a bad ``F`` — and the Sylvester
  fallback behind it — partitioned A and B by *row* at ``n_pre``. Rows
  are equations and the split is over variables, so in any model whose
  equations are not incidentally ordered to match, the guard inspects
  the wrong equations and the fallback solves an underdetermined system.
* ``L`` came from a Klein (2000) eq.-(33) expression that returns zero
  whenever a shock enters a control equation contemporaneously.

The two models below are small enough to have exact solutions, so they
pin all four matrices rather than one.
"""
import numpy as np
import pytest

from puremacro.dsge.klein import klein_solve


# --- model 1: neoclassical growth, full depreciation, log utility -------
# The one textbook case with a closed form:
#     k_{t+1} = alpha*beta*z_t*k_t^alpha,  c_t = (1-alpha*beta)*z_t*k_t^alpha
# so in log deviations   k^_{t+1} = alpha*k^_t + z^_t   and   c^_t = alpha*k^_t + z^_t.
ALPHA, BETA, RHO = 0.33, 0.98, 0.9


def _growth_model():
    """Log-linear Klein form with variables ordered [k, z, c]."""
    k = (ALPHA * BETA) ** (1 / (1 - ALPHA))
    y = k ** ALPHA
    c = y - k
    # Rows: Euler, resource constraint, AR(1).
    Fp = np.array([[-(ALPHA - 1), -1.0, 1.0],
                   [k, 0.0, 0.0],
                   [0.0, 1.0, 0.0]])
    Fc = np.array([[0.0, 0.0, -1.0],
                   [-y * ALPHA, -y, c],
                   [0.0, -RHO, 0.0]])
    Fu = np.array([[0.0], [0.0], [-1.0]])
    return Fp, -Fc, -Fu


def test_growth_model_matches_closed_form():
    A, B, C = _growth_model()
    sol = klein_solve(A, B, n_pre=2, C=C, strict=True)

    assert sol.eu == (1, 1)
    np.testing.assert_allclose(sol.G, [[ALPHA, 1.0], [0.0, RHO]], atol=1e-12)
    np.testing.assert_allclose(sol.F, [[ALPHA, 1.0]], atol=1e-12)
    np.testing.assert_allclose(sol.N.ravel(), [0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(sol.L.ravel(), [0.0], atol=1e-12)


def test_growth_model_solution_satisfies_its_own_equations():
    """The equilibrium condition, checked directly rather than via a formula.

    Any correct solution satisfies ``(A1 + A2 F) G == B1 + B2 F`` for the
    state block and ``(A1 + A2 F) N - B2 L == C`` for the shock block.
    The pre-1.2.0 F failed the first by ~0.8.
    """
    A, B, C = _growth_model()
    n_pre = 2
    sol = klein_solve(A, B, n_pre=n_pre, C=C, strict=True)
    A1, A2 = A[:, :n_pre], A[:, n_pre:]
    B1, B2 = B[:, :n_pre], B[:, n_pre:]

    state_resid = (A1 + A2 @ sol.F) @ sol.G - B1 - B2 @ sol.F
    shock_resid = (A1 + A2 @ sol.F) @ sol.N - B2 @ sol.L - C
    assert np.max(np.abs(state_resid)) < 1e-10
    assert np.max(np.abs(shock_resid)) < 1e-10


# --- model 2: a shock that hits a control contemporaneously -------------
# z = [s, c]; s is predetermined, c is static:
#     s_{t+1} = rho*s_t + u_t      =>  G = [rho],  N = [1]
#     c_t     = theta*s_t + u_t    =>  F = [theta], L = [1]
RHO_S, THETA = 0.8, 2.0


def test_contemporaneous_shock_loads_on_the_control():
    A = np.array([[1.0, 0.0], [0.0, 0.0]])
    B = np.array([[RHO_S, 0.0], [THETA, -1.0]])
    C = np.array([[1.0], [1.0]])

    sol = klein_solve(A, B, n_pre=1, C=C, strict=True)

    np.testing.assert_allclose(sol.G, [[RHO_S]], atol=1e-12)
    np.testing.assert_allclose(sol.F, [[THETA]], atol=1e-12)
    np.testing.assert_allclose(sol.N.ravel(), [1.0], atol=1e-12)
    # This is the assertion that failed before 1.2.0: L was 0.
    np.testing.assert_allclose(sol.L.ravel(), [1.0], atol=1e-12)


def test_no_shock_block_returns_empty_loadings():
    A = np.array([[1.0, 0.0], [0.0, 0.0]])
    B = np.array([[RHO_S, 0.0], [THETA, -1.0]])
    sol = klein_solve(A, B, n_pre=1, strict=True)
    assert sol.N.shape == (1, 0)
    assert sol.L.shape == (1, 0)


# ---------------------------------------------------------------------------
# Post-solve verification of the returned matrices (added in 2.5.0).
#
# Before 2.5.0 `G` was never checked at all, and the guard on `F` compared
# the residual against `1e-6 * max(1, max|A|, max|B|)` — the largest entry
# ANYWHERE in the system, which is not a property of the equation that is
# being violated. On a pencil carrying a 1e9 entry that bar is 1.5e3, so a
# violation worth a third of its own equation's scale sailed through.
# ---------------------------------------------------------------------------
def test_klein_refuses_a_solution_that_does_not_solve_the_model():
    """B = W diag(0.5, 2) inv(W) with W = [[1, 0], [1e9, 1]], A = I, n_pre = 1.

    The model is a well-posed determinate one — ``scipy.linalg.eig(B, A)``
    returns 0.5 and 2.0, the stable eigenvector is [1, 1e9], so the answer is
    G = 0.5, F = 1e9. LAPACK's ``ordqz`` loses accuracy on this pencil
    (cond(W) ~ 1e9) and reports generalised eigenvalues [3.6e-17, 2.5]; the
    old code returned eu=(1, 1) with G = 3.6e-17 and F = 7.5e8 — a state
    transition wrong by every digit and a policy rule wrong by 25%.

    Equation 0 of ``(A1 + A2 F) G = B1 + B2 F`` is violated by 0.5 against its
    own row scale of 1.5. The Sylvester fallback cannot repair it (it
    conditions on the same wrong G), so the only honest answer is a refusal.
    """
    import pytest
    from puremacro.dsge.klein import klein_solve, KleinResidualError

    W = np.array([[1.0, 0.0], [1e9, 1.0]])
    B = W @ np.diag([0.5, 2.0]) @ np.linalg.inv(W)
    A = np.eye(2)

    with pytest.raises(KleinResidualError) as exc:
        klein_solve(A, B, n_pre=1)
    assert exc.value.residual > 0.1
    assert "equilibrium condition" in str(exc.value)

    # ...and it refuses in strict mode too, rather than raising the wrong error.
    with pytest.raises(KleinResidualError):
        klein_solve(A, B, n_pre=1, strict=True)


def test_klein_reports_the_achieved_residual():
    """`KleinSolution.residual` is the row-scaled equilibrium violation, so a
    caller can assert on the quality of the answer instead of trusting it."""
    A, B, C = _growth_model()
    sol = klein_solve(A, B, n_pre=2, C=C, strict=True)
    assert sol.residual < 1e-12, sol.residual
    assert sol.residual >= 0.0


def test_klein_residual_is_scaled_row_by_row_not_by_the_whole_system():
    """The reported residual divides each equation by ITS OWN scale.

    Multiply one equation of a solved model by 1e9. The equilibrium residual
    of that equation scales with it, so a row-scaled measure is unchanged
    while a raw one grows by 1e9 — and the pre-2.5.0 guard, whose bar was
    ``1e-6 * max(1, max|A|, max|B|)``, grew its bar for EVERY OTHER equation
    by the same 1e9 at the same time. That is how a violation worth a third
    of its own row's scale passed as ``0.5 < 1500``.
    """
    A, B, C = _growth_model()
    n_pre = 2
    scaled_A, scaled_B, scaled_C = A.copy(), B.copy(), C.copy()
    scaled_A[1] *= 1e9
    scaled_B[1] *= 1e9
    scaled_C[1] *= 1e9
    sol = klein_solve(scaled_A, scaled_B, n_pre=n_pre, C=scaled_C, strict=True)

    A1, A2 = scaled_A[:, :n_pre], scaled_A[:, n_pre:]
    B1, B2 = scaled_B[:, :n_pre], scaled_B[:, n_pre:]
    rowscale = (np.abs(scaled_A).max(axis=1)
                + np.abs(scaled_B).max(axis=1))[:, None]
    state = (A1 + A2 @ sol.F) @ sol.G - B1 - B2 @ sol.F
    shock = (A1 + A2 @ sol.F) @ sol.N - B2 @ sol.L - scaled_C
    expected = max(float(np.max(np.abs(state) / rowscale)),
                   float(np.max(np.abs(shock) / rowscale)))
    np.testing.assert_allclose(sol.residual, expected, rtol=1e-12, atol=0.0)

    assert sol.residual < 1e-8

    # And the point of the change: the old global scale, and therefore the old
    # tolerance, is inflated by the rescaled equation even for equations that
    # have nothing to do with it.
    old_scale = max(1.0, float(np.max(np.abs(scaled_A))),
                    float(np.max(np.abs(scaled_B))))
    assert 1e-6 * old_scale > 1e2, (
        "the pre-2.5.0 guard would have accepted an absolute violation of "
        f"{1e-6 * old_scale:g} in ANY equation of this system"
    )


# ---------------------------------------------------------------------------
# eu semantics: klein.py:226 has documented (1, 0) for indeterminacy since
# 1.2.0, and the code returned (0, 0) — "no stable solution" — for exactly the
# case the docstring names.
# ---------------------------------------------------------------------------
def test_klein_reports_indeterminacy_as_indeterminacy():
    """One state, one control, both roots stable: too few unstable roots.

    Stable solutions exist — a continuum of them — so existence holds and only
    uniqueness fails. Too MANY unstable roots is the genuine non-existence
    case and keeps (0, 0).
    """
    indeterminate = klein_solve(np.eye(2), np.diag([0.5, 0.4]), n_pre=1)
    assert indeterminate.eu == (1, 0)

    no_solution = klein_solve(np.eye(2), np.diag([2.0, 3.0]), n_pre=1)
    assert no_solution.eu == (0, 0)

    # the Z22 rank-deficient route (a state given the explosive root) is also
    # indeterminacy, and already reported (1, 0)
    misdeclared = klein_solve(np.eye(2), np.diag([2.0, 0.5]), n_pre=1)
    assert misdeclared.eu == (1, 0)


def test_klein_strict_names_the_right_kind_of_failure():
    """The BlanchardKahnError message must not call an order-condition
    indeterminacy a "Z22 rank-deficient" one now that eu[0] is 1 for it."""
    import pytest
    from puremacro.dsge.klein import BlanchardKahnError

    with pytest.raises(BlanchardKahnError) as exc:
        klein_solve(np.eye(2), np.diag([0.5, 0.4]), n_pre=1, strict=True)
    assert exc.value.kind == "indeterminacy"

    with pytest.raises(BlanchardKahnError) as exc:
        klein_solve(np.eye(2), np.diag([2.0, 3.0]), n_pre=1, strict=True)
    assert exc.value.kind == "no stable solution"


# ---------------------------------------------------------------------------
# Unit roots: `div` / qz_criterium.
# ---------------------------------------------------------------------------
def _unit_root_euler(rho, a=0.95, phi=0.5):
    """x_{t+1} = rho x_t + eps ; y_t = a E_t y_{t+1} + phi x_t.

    Closed form: G = rho, F = phi / (1 - a rho).
    """
    A = np.array([[1.0, 0.0], [0.0, a]])
    B = np.array([[rho, 0.0], [-phi, 1.0]])
    C = np.array([[1.0], [0.0]])
    return A, B, C


def test_klein_solves_a_random_walk_state():
    """rho = 1 exactly: the single most common non-stationary macro
    specification, and klein could not solve one at all.

    With the hard |beta/alpha| < 1 the unit root counted as unstable, the
    Blanchard-Kahn count came out 2 against 1 forward-looking variable, and
    the answer was eu=(0, 0) with zero matrices — while `gensys` in the same
    package solved it. The default div = 1 + 1e-8 admits it.
    """
    import warnings

    A, B, C = _unit_root_euler(1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sol = klein_solve(A, B, n_pre=1, C=C, strict=True)
    assert sol.eu == (1, 1)
    np.testing.assert_allclose(sol.G, [[1.0]], atol=1e-10)
    np.testing.assert_allclose(sol.F, [[0.5 / (1.0 - 0.95)]], atol=1e-8)


def test_klein_is_continuous_across_the_unit_root():
    """|F(rho = 1 - 1e-12) - F(rho = 1)| must be small.

    It used to be the whole answer: 10 on one side of the knife edge and 0
    (with eu=(0, 0)) on the other, from a 1e-12 change in a parameter.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        below = klein_solve(*_unit_root_euler(1.0 - 1e-12)[:2], n_pre=1)
        at = klein_solve(*_unit_root_euler(1.0)[:2], n_pre=1)
    assert below.eu == at.eu == (1, 1)
    assert abs(float(below.F[0, 0]) - float(at.F[0, 0])) < 1e-6


def test_klein_warns_that_a_unit_root_solution_is_not_stationary():
    """Admitting the root is right; letting a caller compute an unconditional
    variance from it without a word is not."""
    import warnings
    import pytest

    A, B, C = _unit_root_euler(1.0)
    with pytest.warns(RuntimeWarning, match="NOT stationary"):
        klein_solve(A, B, n_pre=1, C=C)

    # A stationary model must stay silent.
    A, B, C = _unit_root_euler(0.9)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        klein_solve(A, B, n_pre=1, C=C)


def test_klein_div_one_restores_the_strict_pre_2_5_0_classification():
    """The old knife edge is still reachable, and is still what it was."""
    A, B, C = _unit_root_euler(1.0)
    sol = klein_solve(A, B, n_pre=1, C=C, div=1.0)
    assert sol.eu == (0, 0)


# ---------------------------------------------------------------------------
# Degenerate shapes.
# ---------------------------------------------------------------------------
def test_klein_refuses_a_singular_pencil():
    """A = diag(1, 0), B = diag(0.5, 0): variable y is restricted by NO
    equation, so every F solves the model and F = 0 is not "the" answer.

    det(A - lambda B) is identically zero — verified independently by
    ``scipy.linalg.eig(A, B, homogeneous_eigvals=True)``, which returns one
    pair with both alpha and beta vanishing. The old code returned eu=(1, 1),
    G = [0.5], F = [0.0], and strict=True did not raise.
    """
    import pytest

    with pytest.raises(np.linalg.LinAlgError, match="singular"):
        klein_solve(np.diag([1.0, 0.0]), np.diag([0.5, 0.0]), n_pre=1)


def test_klein_handles_a_model_with_no_predetermined_variables():
    """n_pre = 0: y_t = a E_t y_{t+1} + u_t, whose solution is y_t = L u_t.

    This used to die inside a numpy reduction — "zero-size array to reduction
    operation maximum which has no identity" — from the residual guard, with
    no mention of n_pre anywhere in the traceback.
    """
    a = 0.5
    A = np.array([[a]])
    B = np.array([[1.0]])
    C = np.array([[1.0]])
    sol = klein_solve(A, B, n_pre=0, C=C, strict=True)
    assert sol.eu == (1, 1)
    assert sol.G.shape == (0, 0)
    assert sol.F.shape == (1, 0)
    # A E y_{t+1} = B y_t + C u_t with E_t u_{t+1} = 0 gives B L = -C.
    np.testing.assert_allclose(sol.L, [[-1.0]], atol=1e-12)
    assert sol.residual < 1e-12


def test_every_returned_klein_solution_solves_its_own_model():
    """Property test over random determinate systems.

    Nothing in the suite asserted the equilibrium residual of a *returned*
    solution; F was only ever compared against three hand-computed fixtures.
    Over ~1300 random determinate draws the row-scaled residual has a median
    of 4e-16 and a worst case of 4e-10, so a 1e-8 bar here is loose but
    decisive: the LAPACK breakdown of the ill-conditioned pencil above scores
    0.33.
    """
    import warnings

    rng = np.random.default_rng(20260906)
    checked = 0
    for _ in range(300):
        n = int(rng.integers(2, 6))
        n_pre = int(rng.integers(1, n))
        A = rng.standard_normal((n, n))
        B = rng.standard_normal((n, n))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                sol = klein_solve(A, B, n_pre)
        except (np.linalg.LinAlgError, RuntimeError):
            continue
        if sol.eu != (1, 1):
            continue
        checked += 1
        A1, A2 = A[:, :n_pre], A[:, n_pre:]
        B1, B2 = B[:, :n_pre], B[:, n_pre:]
        rowscale = (np.abs(A).max(axis=1) + np.abs(B).max(axis=1))[:, None]
        r = float(np.max(np.abs((A1 + A2 @ sol.F) @ sol.G - B1 - B2 @ sol.F)
                         / rowscale))
        assert r < 1e-8, f"returned eu=(1,1) with a row-scaled residual of {r:g}"
        np.testing.assert_allclose(sol.residual, r, rtol=1e-10, atol=1e-18)
    assert checked > 50, f"only {checked} determinate draws — sweep too thin"


def test_klein_is_invariant_to_a_uniform_rescaling_of_the_system():
    """Multiplying A, B and C by s does not change the model, so it must not
    change the answer, the eu flags or the reported residual."""
    A, B, C = _growth_model()
    base = klein_solve(A, B, n_pre=2, C=C, strict=True)
    for s in (1e-12, 1e-6, 1.0, 1e6, 1e12):
        got = klein_solve(s * A, s * B, n_pre=2, C=s * C, strict=True)
        assert got.eu == base.eu, f"eu changed under a uniform rescale by {s:g}"
        np.testing.assert_allclose(got.G, base.G, atol=1e-10)
        np.testing.assert_allclose(got.F, base.F, atol=1e-10)
        np.testing.assert_allclose(got.N, base.N, atol=1e-10)
        np.testing.assert_allclose(got.L, base.L, atol=1e-10)
