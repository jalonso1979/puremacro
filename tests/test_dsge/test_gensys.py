

# ---------------------------------------------------------------------------
# Regression tests for the gensys impact matrix (fixed after 1.9.0).
#
# `Impact` used to be computed as (Gamma0 - G Gamma1)^-1 Psi, which drops the
# Pi N term: it solves the system as if the expectation errors did not respond
# to the shocks, when responding to them is the entire content of the
# rational-expectations solution.
#
# gensys had two test files and neither could catch it.
# `tests/test_dsge_gensys_coverage.py` asserts shapes, dtypes, that `eu` is a
# tuple of length 2, and that each entry is in (0, 1) — nothing about the
# numbers. `tests/test_dsge/test_qz_fallback.py` asserts that the real and
# complex QZ paths return the SAME Impact, which is a self-consistency check:
# both paths run the same wrong formula, so it holds exactly as well when the
# formula is wrong. No fixture anywhere compared Impact against a model whose
# answer is known in closed form, which is the only check that can fail.
# ---------------------------------------------------------------------------
def test_gensys_impact_matches_a_closed_form_solution():
    """x_t = rho x_{t-1} + eps ; y_t = a E_t y_{t+1} + c x_t.

    The unique stable solution is y_t = c/(1 - a rho) x_t, so from eps = 1 with
    x_{-1} = 0 the impact vector is [1, k, rho k] with k = c/(1 - a rho).
    The old expression returned [0.6032, -0.8547, -1.6802] against a truth of
    [1, 3.0769, 2.1538] — the wrong sign on y, and an x that did not even
    follow the AR(1) that defines it.
    """
    import numpy as np
    from puremacro.dsge.gensys import gensys

    rho, a, c = 0.7, 0.5, 2.0
    k = c / (1.0 - a * rho)
    truth = np.array([1.0, k, rho * k])

    # z = (x, y, xi) with xi_t = E_t y_{t+1}
    Gamma0 = np.array([[1.0, 0.0, 0.0], [-c, 1.0, -a], [0.0, 1.0, 0.0]])
    Gamma1 = np.array([[rho, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    Psi = np.array([[1.0], [0.0], [0.0]])
    Pi = np.array([[0.0], [0.0], [1.0]])

    sol = gensys(Gamma0, Gamma1, Psi, Pi)
    assert sol.eu == (1, 1)
    np.testing.assert_allclose(sol.Impact.ravel(), truth, atol=1e-10)

    # and the whole path, not just impact: z_h = G^h Impact
    z = sol.Impact[:, 0].copy()
    for h in range(6):
        np.testing.assert_allclose(
            z[:2], [rho ** h, k * rho ** h], atol=1e-10,
            err_msg=f"IRF departs from the closed form at horizon {h}",
        )
        z = sol.G @ z


def test_gensys_impact_on_a_purely_forward_looking_model():
    """y_t = a E_t y_{t+1} + eps_t has the unique stable solution y_t = eps_t.

    The simplest possible case, and the old code returned [0, -2] at a = 0.5:
    the variable did not respond to its own shock at all.
    """
    import numpy as np
    from puremacro.dsge.gensys import gensys

    a = 0.5
    sol = gensys(
        np.array([[1.0, -a], [1.0, 0.0]]),
        np.array([[0.0, 0.0], [0.0, 1.0]]),
        np.array([[1.0], [0.0]]),
        np.array([[0.0], [1.0]]),
    )
    assert sol.eu == (1, 1)
    np.testing.assert_allclose(sol.Impact.ravel(), [1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(sol.G, np.zeros((2, 2)), atol=1e-12)


def test_gensys_solves_a_model_with_no_unstable_roots():
    """A purely backward-looking model used to raise on an empty Z2 slice.

    `sv_Z2.max()` on the zero-column slice raised
    "zero-size array to reduction operation maximum", so gensys could not
    solve x_t = rho x_{t-1} + eps at all.
    """
    import numpy as np
    from puremacro.dsge.gensys import gensys

    sol = gensys(np.array([[1.0]]), np.array([[0.7]]),
                 np.array([[1.0]]), np.zeros((1, 0)))
    assert sol.eu == (1, 1)
    np.testing.assert_allclose(sol.G, [[0.7]], atol=1e-12)
    np.testing.assert_allclose(sol.Impact, [[1.0]], atol=1e-12)


# ---------------------------------------------------------------------------
# The decision rule itself (fixed after 2.4.0).
#
# `G` used to be computed as `Z1 inv(S11) T11 Z1^H`, which is Sims's rule
# composed with the ORTHOGONAL PROJECTOR onto span(Z1) — it drops the
# `[tmat T][:, n_stable:]` block. The two agree exactly on any z_{t-1} that
# already lies on the stable manifold, which is why every test in the package
# passed: `test_gensys_impact_matches_a_closed_form_solution` above seeds its
# iteration from `Impact`, `tests/test_dsge_gensys_coverage.py` never asserts a
# numeric G, and `validation/cases_dsge.py::_klein_gensys_cross` compares only
# eigenvalues — and the projected form has exactly the same spectrum.
#
# Off the manifold — which is the whole point of a state-transition matrix —
# it disagreed with Sims on 300/300 random determinate models, worst relative
# error 0.994, and did not satisfy the model's own equations.
# ---------------------------------------------------------------------------
def _ar1_euler_model():
    """x_t = rho x_{t-1} + eps ; y_t = a E_t y_{t+1} + c x_t, with z=(x,y,xi)."""
    import numpy as np

    rho, a, c = 0.7, 0.5, 2.0
    Gamma0 = np.array([[1.0, 0.0, 0.0], [-c, 1.0, -a], [0.0, 1.0, 0.0]])
    Gamma1 = np.array([[rho, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    Psi = np.array([[1.0], [0.0], [0.0]])
    Pi = np.array([[0.0], [0.0], [1.0]])
    return Gamma0, Gamma1, Psi, Pi, rho, a, c


def test_gensys_G_is_the_closed_form_decision_rule():
    """The unique stable solution is y_t = k x_t with k = c/(1 - a rho), and
    x is the only thing that carries state, so

        G = [[rho, 0, 0], [k rho, 0, 0], [k rho^2, 0, 0]]

    (row 2 is xi_t = E_t y_{t+1} = k x_{t+1} = k rho x_t = k rho^2 x_{t-1}).
    Nothing but x carries state, so columns 1 and 2 are exactly zero. The
    projected form returned
    [[0.1241, 0, 0.2674], [0.3820, 0, 0.8227], [0.2674, 0, 0.5759]] — an AR(1)
    coefficient of 0.124 on a process the model literally defines as 0.7, and
    a spurious dependence on xi_{t-1}.
    """
    import numpy as np
    from puremacro.dsge.gensys import gensys

    Gamma0, Gamma1, Psi, Pi, rho, a, c = _ar1_euler_model()
    k = c / (1.0 - a * rho)
    sol = gensys(Gamma0, Gamma1, Psi, Pi)
    assert sol.eu == (1, 1)
    truth = np.array([[rho, 0.0, 0.0],
                      [k * rho, 0.0, 0.0],
                      [k * rho ** 2, 0.0, 0.0]])
    np.testing.assert_allclose(sol.G, truth, atol=1e-12)


def test_gensys_G_satisfies_the_model_off_the_stable_manifold():
    """Model-free check, valid for any determinate model: for EVERY z_{t-1},

        Gamma0 (G z_{t-1}) - Gamma1 z_{t-1}  must lie in col(Pi)

    because the only thing the equations leave free is Pi eta_t. Seeding from
    `Impact` cannot detect a violation — that state is already on the stable
    manifold, where the projected and the true rule agree to 5e-15.
    """
    import numpy as np
    from puremacro.dsge.gensys import gensys

    Gamma0, Gamma1, Psi, Pi, *_ = _ar1_euler_model()
    sol = gensys(Gamma0, Gamma1, Psi, Pi)
    proj = Pi @ np.linalg.pinv(Pi)
    rng = np.random.default_rng(20260906)
    for _ in range(25):
        z = rng.standard_normal(3)
        target = Gamma0 @ (sol.G @ z) - Gamma1 @ z
        np.testing.assert_allclose(
            target - proj @ target, 0.0, atol=1e-12,
            err_msg="G must solve the model from an arbitrary state, not only "
                    "from one already on the stable manifold",
        )


# ---------------------------------------------------------------------------
# eu: indeterminacy and non-existence are different diagnoses.
# ---------------------------------------------------------------------------
def _nk_model(phi_pi):
    """Three-equation NK model in gensys form.

    z = (x, pi, i, v, E x, E pi); eta on the two expectations; v is the AR(1)
    monetary shock. Determinate iff the Taylor principle holds.
    """
    import numpy as np

    sigma, kappa, beta, rho_v = 1.0, 0.1, 0.99, 0.5
    n = 6
    ix, ipi, ii, iv, iEx, iEpi = range(n)
    G0 = np.zeros((n, n)); G1 = np.zeros((n, n))
    # IS:  x_t = E_t x_{t+1} - (1/sigma)(i_t - E_t pi_{t+1})
    G0[0, ix] = 1.0; G0[0, iEx] = -1.0
    G0[0, ii] = 1.0 / sigma; G0[0, iEpi] = -1.0 / sigma
    # NKPC: pi_t = beta E_t pi_{t+1} + kappa x_t
    G0[1, ipi] = 1.0; G0[1, iEpi] = -beta; G0[1, ix] = -kappa
    # Taylor: i_t = phi_pi pi_t + v_t
    G0[2, ii] = 1.0; G0[2, ipi] = -phi_pi; G0[2, iv] = -1.0
    # AR(1) shock
    G0[3, iv] = 1.0; G1[3, iv] = rho_v
    # expectation definitions: x_t = E_{t-1} x_t + eta_x
    G0[4, ix] = 1.0; G1[4, iEx] = 1.0
    G0[5, ipi] = 1.0; G1[5, iEpi] = 1.0
    Psi = np.zeros((n, 1)); Psi[3, 0] = 1.0
    Pi = np.zeros((n, 2)); Pi[4, 0] = 1.0; Pi[5, 1] = 1.0
    return G0, G1, Psi, Pi


def test_gensys_reports_indeterminacy_as_indeterminacy_not_non_existence():
    """A passive Taylor rule makes the NK model indeterminate — stable
    solutions exist, there is a continuum of them. The old code answered
    eu=(0, 0), "no stable solution", which is the opposite diagnosis and
    points a user at the opposite fix.
    """
    from puremacro.dsge.gensys import gensys

    assert gensys(*_nk_model(0.8)).eu == (1, 0)
    assert gensys(*_nk_model(1.5)).eu == (1, 1)


def test_gensys_solves_a_model_with_a_redundant_expectation_error():
    """Two identical columns in Pi describe the same expectation error twice.

    The model is unchanged and still determinate, but the root count
    n_eta == n_unstable now fails, and the pre-2.5.0 order-condition gate
    refused it outright with eu=(0, 0). Sims's rank tests see through the
    duplication: the answer is the single-column answer.
    """
    import numpy as np
    from puremacro.dsge.gensys import gensys

    Gamma0, Gamma1, Psi, Pi, rho, a, c = _ar1_euler_model()
    k = c / (1.0 - a * rho)
    sol = gensys(Gamma0, Gamma1, Psi, np.hstack([Pi, Pi]))
    assert sol.eu == (1, 1)
    np.testing.assert_allclose(sol.Impact.ravel(), [1.0, k, rho * k], atol=1e-10)


def test_gensys_reports_non_existence_when_Q2Pi_is_rank_deficient():
    """Two unstable roots and two Pi columns, but only ONE independent
    expectation-error direction.

    z1_t = 2 z1_{t-1} + eps + (eta1 + eta2)
    z2_t = 3 z2_{t-1} +       (eta1 + eta2)

    Both roots are explosive, so stability forces z == 0 from a zero initial
    state, which needs eta1 + eta2 = -eps from the first equation and
    eta1 + eta2 = 0 from the second. There is no solution. But the root count
    n_eta == n_unstable == 2 is satisfied, so the pre-2.5.0 order-condition
    gate set eu[0] = 1 — asserting existence — and blamed the failure on
    uniqueness, returning eu=(1, 0). Sims's existence test looks at whether
    Q2 Psi lies in col(Q2 Pi), which is the question that actually matters.
    """
    import numpy as np
    from puremacro.dsge.gensys import gensys

    sol = gensys(
        np.eye(2),
        np.diag([2.0, 3.0]),
        np.array([[1.0], [0.0]]),
        np.array([[1.0, 1.0], [1.0, 1.0]]),
    )
    assert sol.eu == (0, 0), (
        "Q2 Pi is rank 1 against 2 unstable roots and Q2 Psi is not in its "
        "column space, so no stable solution exists"
    )
    np.testing.assert_array_equal(sol.G, np.zeros((2, 2)))


def test_gensys_is_invariant_to_a_uniform_rescaling_of_the_system():
    """Multiplying (Gamma0, Gamma1, Psi, Pi) by s does not change the model.

    The stability count used to be taken from |beta/alpha| guarded by an
    ABSOLUTE |alpha| > 1e-12, while the pencil had been ordered by the
    relative |beta| < div|alpha|. At s = 1e-13 every |alpha| falls under the
    absolute floor, every eigenvalue is reported infinite, n_stable collapses
    to 0 and the solver refused a model it solves at s = 1.
    """
    import numpy as np
    from puremacro.dsge.gensys import gensys

    Gamma0, Gamma1, Psi, Pi, *_ = _ar1_euler_model()
    base = gensys(Gamma0, Gamma1, Psi, Pi)
    for s in (1e-14, 1e-13, 1e-8, 1.0, 1e8, 1e13):
        scaled = gensys(s * Gamma0, s * Gamma1, s * Psi, s * Pi)
        assert scaled.eu == base.eu, f"eu changed under a uniform rescale by {s:g}"
        np.testing.assert_allclose(scaled.G, base.G, atol=1e-10,
                                   err_msg=f"G changed under a rescale by {s:g}")
        np.testing.assert_allclose(scaled.Impact, base.Impact, atol=1e-10,
                                   err_msg=f"Impact changed under a rescale by {s:g}")


def test_gensys_refuses_a_singular_pencil():
    """z2 appears in no equation, so every value of it solves the model.

    Both alpha and beta vanish for that direction: det(Gamma0 - lambda Gamma1)
    is identically zero and there is no stable/unstable split to take. The old
    code answered eu=(1, 1) with a zero row for z2 — a plausible-looking
    number for a quantity the model does not determine.
    """
    import numpy as np
    import pytest
    from puremacro.dsge.gensys import gensys

    with pytest.raises(np.linalg.LinAlgError, match="singular"):
        gensys(np.diag([1.0, 0.0]), np.diag([0.5, 0.0]),
               np.array([[1.0], [0.0]]), np.zeros((2, 0)))
