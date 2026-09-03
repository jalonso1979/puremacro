

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
