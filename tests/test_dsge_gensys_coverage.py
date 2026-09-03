"""Coverage tests for puremacro.dsge.gensys — Sims (2002) QZ solver.

Tests are organised as:
  - GensysSolution dataclass (field types, frozen immutability)
  - Output shape & dtype contracts
  - Blanchard-Kahn (BK) order-condition failures
  - Mathematical identities that hold for every valid solution:
      (1) max|eig(G)| < div (stability)
      (2) (G0 - G @ G1) @ Impact = Psi (shock-impact formula)
      (3) eigenvalues are sorted ascending
      (4) max|eig(G)| <= max stable generalised eigenvalue
  - Known-value tests derived from block-structured models:
      (5) Block isolation — orthogonal shocks do not cross-contaminate
      (6) Geometric IRF decay — AR state decays at its own persistence rate
  - Input-conversion robustness (Python lists, integer arrays)
  - Multiple shocks (n_eps > 1) and multiple expectation errors (n_eta > 1)
  - The `div` threshold parameter
  - Zero-shock model (n_eps = 0)

Model construction note (Sims form)
------------------------------------
The solver takes Gamma0, Gamma1, Psi, Pi in the convention
    Gamma0 z_t = Gamma1 z_{t-1} + Psi eps_t + Pi eta_t
where eta_t = z_t - E_{t-1}[z_t] is the CURRENT-period expectation error.

For a forward-looking variable y with persistence parameter beta:
    Structural (at t-1):  y_{t-1} = beta * E_{t-1}[y_t] + ...
    Substituting E_{t-1}[y_t] = y_t - eta_{y,t}:
        y_{t-1} = beta*(y_t - eta_y) + ...
    => Gamma0 row:  [0, beta]  Gamma1 row: [..., 1]  Pi col:  [0, beta]^T

The stable generalised eigenvalue of this row is beta, and the unstable one
is 1/beta.  Blanchard-Kahn (BK) requires n_eta == n_unstable.  Hence for each
forward-looking variable added we must have exactly one column in Pi.
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.dsge.gensys import gensys, GensysSolution


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model_2var():
    """2-variable forward-looking model: one AR(1) state + one control.

    State z_1: z1_t = rho * z1_{t-1} + eps_t
    Control z_2: (Sims-shifted) beta*z2_t = -z1_{t-1} + z2_{t-1} + beta*eta_z2

    Analytical: unique stable solution, eu=(1,1).
    Stable gen.eig = rho, unstable gen.eig = 1/beta.
    """
    rho = 0.7
    beta = 0.8
    G0 = np.array([[1.0, 0.0], [0.0, beta]])
    G1 = np.array([[rho, 0.0], [-1.0, 1.0]])
    Psi = np.array([[1.0], [0.0]])
    Pi = np.array([[0.0], [beta]])
    params = dict(rho=rho, beta=beta)
    return G0, G1, Psi, Pi, params


@pytest.fixture(scope="module")
def model_4var():
    """4-variable block-diagonal model: two independent 2-variable subsystems.

    State z_1: z1_t = rho1 * z1_{t-1} + eps1_t
    State z_2: z2_t = rho2 * z2_{t-1} + eps2_t
    Control z_3: driven by z_1, unique stable solution
    Control z_4: driven by z_2, unique stable solution

    The two subsystems are ORTHOGONAL: shock eps1 cannot affect z_2 or z_4,
    and shock eps2 cannot affect z_1 or z_3.
    """
    rho1, rho2 = 0.6, 0.5
    b1, b2 = 0.9, 0.85

    G0 = np.zeros((4, 4))
    G0[0, 0] = G0[1, 1] = 1.0
    G0[2, 2] = b1
    G0[3, 3] = b2

    G1 = np.zeros((4, 4))
    G1[0, 0] = rho1
    G1[1, 1] = rho2
    G1[2, 0] = -1.0
    G1[2, 2] = 1.0
    G1[3, 1] = -1.0
    G1[3, 3] = 1.0

    Psi = np.zeros((4, 2))
    Psi[0, 0] = 1.0
    Psi[1, 1] = 1.0

    Pi = np.zeros((4, 2))
    Pi[2, 0] = b1
    Pi[3, 1] = b2

    params = dict(rho1=rho1, rho2=rho2, b1=b1, b2=b2)
    return G0, G1, Psi, Pi, params


# ---------------------------------------------------------------------------
# GensysSolution dataclass
# ---------------------------------------------------------------------------


class TestGensysSolution:
    """Tests for the GensysSolution frozen dataclass."""

    def test_fields_are_accessible(self):
        """All four fields can be read after construction."""
        sol = GensysSolution(
            G=np.eye(2),
            Impact=np.ones((2, 1)),
            eu=(1, 1),
            eigenvalues=np.array([0.5, 1.2]),
        )
        assert sol.G.shape == (2, 2)
        assert sol.Impact.shape == (2, 1)
        assert sol.eu == (1, 1)
        assert sol.eigenvalues.shape == (2,)

    def test_frozen_immutability(self):
        """GensysSolution is frozen: assigning to any field must raise."""
        from dataclasses import FrozenInstanceError

        sol = GensysSolution(
            G=np.eye(2),
            Impact=np.ones((2, 1)),
            eu=(1, 1),
            eigenvalues=np.array([0.5]),
        )
        with pytest.raises((FrozenInstanceError, AttributeError)):
            sol.eu = (0, 0)  # type: ignore[misc]

    def test_eu_is_tuple(self, model_2var):
        """eu field returned by gensys is a tuple, not a list."""
        G0, G1, Psi, Pi, _ = model_2var
        sol = gensys(G0, G1, Psi, Pi)
        assert isinstance(sol.eu, tuple)

    def test_eu_has_length_two(self, model_2var):
        """eu field has exactly two elements."""
        G0, G1, Psi, Pi, _ = model_2var
        sol = gensys(G0, G1, Psi, Pi)
        assert len(sol.eu) == 2

    def test_eu_values_in_zero_one(self, model_2var):
        """Each element of eu is either 0 or 1."""
        G0, G1, Psi, Pi, _ = model_2var
        sol = gensys(G0, G1, Psi, Pi)
        assert sol.eu[0] in (0, 1)
        assert sol.eu[1] in (0, 1)


# ---------------------------------------------------------------------------
# Output shape and dtype contracts
# ---------------------------------------------------------------------------


class TestOutputContracts:
    """Shape, dtype, and basic structural tests on gensys output."""

    def test_G_shape_n_by_n(self, model_2var):
        """G has shape (n, n)."""
        G0, G1, Psi, Pi, _ = model_2var
        n = G0.shape[0]
        sol = gensys(G0, G1, Psi, Pi)
        assert sol.G.shape == (n, n)

    def test_Impact_shape_n_by_n_eps(self, model_2var):
        """Impact has shape (n, n_eps)."""
        G0, G1, Psi, Pi, _ = model_2var
        n = G0.shape[0]
        n_eps = Psi.shape[1]
        sol = gensys(G0, G1, Psi, Pi)
        assert sol.Impact.shape == (n, n_eps)

    def test_eigenvalues_length_n(self, model_2var):
        """eigenvalues has length n (one per generalised eigenvalue)."""
        G0, G1, Psi, Pi, _ = model_2var
        n = G0.shape[0]
        sol = gensys(G0, G1, Psi, Pi)
        assert sol.eigenvalues.shape == (n,)

    def test_G_dtype_float(self, model_2var):
        """G is real float64."""
        G0, G1, Psi, Pi, _ = model_2var
        sol = gensys(G0, G1, Psi, Pi)
        assert np.issubdtype(sol.G.dtype, np.floating)

    def test_Impact_dtype_float(self, model_2var):
        """Impact is real float64."""
        G0, G1, Psi, Pi, _ = model_2var
        sol = gensys(G0, G1, Psi, Pi)
        assert np.issubdtype(sol.Impact.dtype, np.floating)

    def test_eigenvalues_real_nonneg(self, model_2var):
        """eigenvalues are real (imaginary part zero) and nonneg (they are |eig|)."""
        G0, G1, Psi, Pi, _ = model_2var
        sol = gensys(G0, G1, Psi, Pi)
        assert np.issubdtype(sol.eigenvalues.dtype, np.floating)
        assert np.all(sol.eigenvalues >= 0.0)

    def test_multiple_shocks_impact_shape(self):
        """When n_eps=2, Impact has shape (n, 2)."""
        G0 = np.array([[1.0, 0.0], [0.0, 0.8]])
        G1 = np.array([[0.7, 0.0], [-1.0, 1.0]])
        Psi = np.array([[1.0, 0.5], [0.0, 0.3]])  # 2 shocks
        Pi = np.array([[0.0], [0.8]])
        sol = gensys(G0, G1, Psi, Pi)
        assert sol.Impact.shape == (2, 2)

    def test_zero_shocks_impact_shape(self):
        """When n_eps=0, Impact has shape (n, 0)."""
        G0 = np.array([[1.0, 0.0], [0.0, 0.8]])
        G1 = np.array([[0.7, 0.0], [-1.0, 1.0]])
        Psi = np.zeros((2, 0))
        Pi = np.array([[0.0], [0.8]])
        sol = gensys(G0, G1, Psi, Pi)
        assert sol.Impact.shape == (2, 0)

    def test_four_variable_shapes(self, model_4var):
        """4x4 model: G is (4,4), Impact is (4,2)."""
        G0, G1, Psi, Pi, _ = model_4var
        sol = gensys(G0, G1, Psi, Pi)
        assert sol.G.shape == (4, 4)
        assert sol.Impact.shape == (4, 2)


# ---------------------------------------------------------------------------
# Blanchard-Kahn order-condition failures
# ---------------------------------------------------------------------------


class TestBlancharKahnFailure:
    """When BK order condition fails, eu=(0,0) and G, Impact are zero."""

    def test_too_many_expectation_errors_eu_zero(self):
        """n_eta > n_unstable: all eigenvalues stable but Pi has 1 column -> BK fails."""
        G0 = np.eye(2)
        G1 = np.diag([0.5, 0.7])   # both eigenvalues < 1 (stable)
        Psi = np.eye(2)
        Pi = np.ones((2, 1))        # 1 expectation error, 0 unstable eigs
        sol = gensys(G0, G1, Psi, Pi)
        assert sol.eu == (0, 0)

    def test_too_few_expectation_errors_eu_zero(self):
        """n_eta < n_unstable: one unstable eigenvalue but Pi has 0 columns -> BK fails."""
        G0 = np.eye(2)
        G1 = np.diag([2.0, 0.7])   # one unstable eigenvalue = 2
        Psi = np.eye(2)
        Pi = np.zeros((2, 0))       # 0 expectation errors, 1 unstable eig
        sol = gensys(G0, G1, Psi, Pi)
        assert sol.eu == (0, 0)

    def test_bk_failure_G_is_zero(self):
        """When BK fails, G is a zero matrix (no solution computed)."""
        G0 = np.eye(2)
        G1 = np.diag([0.5, 0.7])
        Psi = np.eye(2)
        Pi = np.ones((2, 1))
        sol = gensys(G0, G1, Psi, Pi)
        np.testing.assert_array_equal(sol.G, np.zeros((2, 2)))

    def test_bk_failure_Impact_is_zero(self):
        """When BK fails, Impact is a zero matrix (no solution computed)."""
        G0 = np.eye(2)
        G1 = np.diag([0.5, 0.7])
        Psi = np.eye(2)
        Pi = np.ones((2, 1))
        sol = gensys(G0, G1, Psi, Pi)
        np.testing.assert_array_equal(sol.Impact, np.zeros((2, 2)))

    def test_three_var_n_eta_2_all_stable(self):
        """3-variable system, n_eta=2, all stable eigenvalues -> BK fails."""
        G0 = np.eye(3)
        G1 = np.diag([0.3, 0.5, 0.7])
        Psi = np.eye(3)
        Pi = np.ones((3, 2))
        sol = gensys(G0, G1, Psi, Pi)
        assert sol.eu == (0, 0)

    def test_exist_flag_only_eu_zero(self):
        """eu[0] = 0 when BK order condition fails."""
        G0 = np.eye(2)
        G1 = np.diag([0.5, 0.6])
        Psi = np.eye(2)
        Pi = np.ones((2, 1))
        sol = gensys(G0, G1, Psi, Pi)
        assert sol.eu[0] == 0


# ---------------------------------------------------------------------------
# Mathematical identity: stability of G
# ---------------------------------------------------------------------------


class TestGStability:
    """When eu=(1,1), all eigenvalues of G are < div (default 1+eps)."""

    def test_max_eigval_less_than_one_2var(self, model_2var):
        """2-var model: max|eig(G)| < 1."""
        G0, G1, Psi, Pi, _ = model_2var
        sol = gensys(G0, G1, Psi, Pi)
        assert sol.eu == (1, 1)
        max_eig = float(np.abs(np.linalg.eigvals(sol.G)).max())
        assert max_eig < 1.0

    def test_max_eigval_less_than_one_4var(self, model_4var):
        """4-var model: max|eig(G)| < 1."""
        G0, G1, Psi, Pi, _ = model_4var
        sol = gensys(G0, G1, Psi, Pi)
        assert sol.eu == (1, 1)
        max_eig = float(np.abs(np.linalg.eigvals(sol.G)).max())
        assert max_eig < 1.0

    def test_max_eigval_matches_largest_stable_gen_eig(self, model_2var):
        """max|eig(G)| matches the largest stable generalised eigenvalue."""
        G0, G1, Psi, Pi, p = model_2var
        sol = gensys(G0, G1, Psi, Pi)
        max_eig_G = float(np.abs(np.linalg.eigvals(sol.G)).max())
        stable_gen_eigs = sol.eigenvalues[sol.eigenvalues < 1.0 + 1e-6]
        np.testing.assert_allclose(
            max_eig_G, stable_gen_eigs.max(), atol=1e-10,
            err_msg="max|eig(G)| should equal max stable generalised eigenvalue",
        )

    def test_max_eigval_matches_rho_4var(self, model_4var):
        """4-var model: max|eig(G)| = rho1 (the dominant persistence)."""
        G0, G1, Psi, Pi, p = model_4var
        sol = gensys(G0, G1, Psi, Pi)
        max_eig_G = float(np.abs(np.linalg.eigvals(sol.G)).max())
        np.testing.assert_allclose(
            max_eig_G, p["rho1"], atol=1e-10,
            err_msg=f"max|eig(G)| should equal rho1={p['rho1']}",
        )


# ---------------------------------------------------------------------------
# Mathematical identity: shock-impact formula
# ---------------------------------------------------------------------------


def _shock_identity_residual(G0, Psi, Pi, sol):
    """Residual of  Gamma0 @ Impact = Psi + Pi @ N  after projecting out Pi.

    Substituting z_t = G z_{t-1} + Impact eps_t into
    Gamma0 z_t = Gamma1 z_{t-1} + Psi eps_t + Pi eta_t and matching the eps
    terms gives Gamma0 @ Impact = Psi + Pi @ N, where N is the loading of the
    expectation errors on the shocks. N is not known a priori, so the testable
    content is that (Gamma0 @ Impact - Psi) lies in the column space of Pi.
    """
    target = G0 @ sol.Impact - Psi
    if Pi.shape[1] == 0:
        return target
    N, *_ = np.linalg.lstsq(Pi, target, rcond=None)
    return target - Pi @ N


class TestShockImpactFormula:
    """Gamma0 @ Impact - Psi lies in the column space of Pi.

    This class used to assert `(G0 - G @ G1) @ Impact == Psi`, which is not an
    identity of the model at all: it is the old, incorrect `Impact` formula
    restated as a test. It was derived from the code rather than from the
    equations, so it could only ever confirm whatever the code already did.

    The check is decisive rather than a matter of taste. On the model
    x_t = rho x_{t-1} + eps, y_t = a E_t y_{t+1} + c x_t -- whose unique stable
    solution y_t = c/(1 - a rho) x_t is known in closed form -- the CORRECT
    impact vector [1, 3.0769, 2.1538] gives
    (G0 - G G1) @ Impact = [0.3372, -2.0393, 1.6494], nowhere near Psi =
    [1, 0, 0]. The analytically right answer failed the old assertion, and the
    wrong one passed it.
    """

    def test_shock_identity_2var(self, model_2var):
        """Shock identity holds for the 2-variable model."""
        G0, G1, Psi, Pi, _ = model_2var
        sol = gensys(G0, G1, Psi, Pi)
        assert sol.eu == (1, 1)
        np.testing.assert_allclose(
            _shock_identity_residual(G0, Psi, Pi, sol), 0.0, atol=1e-10,
            err_msg="Gamma0 @ Impact - Psi must lie in the column space of Pi",
        )

    def test_shock_identity_4var(self, model_4var):
        """Shock identity holds for the 4-variable model."""
        G0, G1, Psi, Pi, _ = model_4var
        sol = gensys(G0, G1, Psi, Pi)
        assert sol.eu == (1, 1)
        np.testing.assert_allclose(
            _shock_identity_residual(G0, Psi, Pi, sol), 0.0, atol=1e-10,
            err_msg="Gamma0 @ Impact - Psi must lie in the column space of Pi",
        )

    def test_shock_identity_multiple_shocks(self):
        """Shock identity holds when n_eps=2."""
        G0 = np.array([[1.0, 0.0], [0.0, 0.8]])
        G1 = np.array([[0.7, 0.0], [-1.0, 1.0]])
        Psi = np.array([[1.0, 0.5], [0.0, 0.3]])
        Pi = np.array([[0.0], [0.8]])
        sol = gensys(G0, G1, Psi, Pi)
        assert sol.eu == (1, 1)
        np.testing.assert_allclose(
            _shock_identity_residual(G0, Psi, Pi, sol), 0.0, atol=1e-10,
            err_msg="Shock identity must hold for a 2-shock system",
        )


# ---------------------------------------------------------------------------
# Mathematical identity: eigenvalues sorted ascending
# ---------------------------------------------------------------------------


class TestEigenvaluesSorted:
    """eigenvalues field is sorted in ascending order."""

    def test_eigenvalues_sorted_2var(self, model_2var):
        """Generalised eigenvalues are sorted ascending in 2-var model."""
        G0, G1, Psi, Pi, _ = model_2var
        sol = gensys(G0, G1, Psi, Pi)
        assert np.all(
            sol.eigenvalues[:-1] <= sol.eigenvalues[1:]
        ), "eigenvalues should be sorted ascending"

    def test_eigenvalues_sorted_4var(self, model_4var):
        """Generalised eigenvalues are sorted ascending in 4-var model."""
        G0, G1, Psi, Pi, _ = model_4var
        sol = gensys(G0, G1, Psi, Pi)
        assert np.all(
            sol.eigenvalues[:-1] <= sol.eigenvalues[1:]
        ), "eigenvalues should be sorted ascending"

    def test_stable_eigs_first(self, model_2var):
        """All stable generalised eigenvalues come before unstable ones."""
        G0, G1, Psi, Pi, p = model_2var
        sol = gensys(G0, G1, Psi, Pi)
        # The sorted array should have stable (< 1+eps) eigenvalues first
        n_stable = int(np.sum(sol.eigenvalues < 1.0 + 1e-6))
        stable_part = sol.eigenvalues[:n_stable]
        unstable_part = sol.eigenvalues[n_stable:]
        assert np.all(stable_part < 1.0 + 1e-6)
        assert np.all(unstable_part >= 1.0)

    def test_stable_eig_equals_rho_2var(self, model_2var):
        """First (stable) generalised eigenvalue equals rho for 2-var model."""
        G0, G1, Psi, Pi, p = model_2var
        sol = gensys(G0, G1, Psi, Pi)
        np.testing.assert_allclose(
            sol.eigenvalues[0], p["rho"], atol=1e-10,
            err_msg="Smallest gen.eigenvalue should equal rho (the AR persistence)",
        )

    def test_unstable_eig_equals_inv_beta_2var(self, model_2var):
        """Second (unstable) generalised eigenvalue equals 1/beta for 2-var model."""
        G0, G1, Psi, Pi, p = model_2var
        sol = gensys(G0, G1, Psi, Pi)
        np.testing.assert_allclose(
            sol.eigenvalues[1], 1.0 / p["beta"], atol=1e-10,
            err_msg="Unstable gen.eigenvalue should equal 1/beta",
        )


# ---------------------------------------------------------------------------
# Known-value: block isolation in 4-variable model
# ---------------------------------------------------------------------------


class TestBlockIsolation:
    """Orthogonal subsystems: shock j cannot affect variables in subsystem k≠j.

    The 4-var model has two independent 2-variable subsystems:
      subsystem 1: z[0] (state), z[2] (control) — driven by eps1
      subsystem 2: z[1] (state), z[3] (control) — driven by eps2
    By construction, the two subsystems are orthogonal.  The gensys Impact
    matrix must reflect this: Impact[:, j] has zeros for variables not in
    subsystem j.
    """

    def test_shock1_does_not_affect_z1_on_impact(self, model_4var):
        """Shock eps1 must not affect z[1] (state in subsystem 2) on impact."""
        G0, G1, Psi, Pi, _ = model_4var
        sol = gensys(G0, G1, Psi, Pi)
        np.testing.assert_allclose(
            sol.Impact[1, 0], 0.0, atol=1e-12,
            err_msg="eps1 should not affect z[1] (subsystem 2 state)",
        )

    def test_shock1_does_not_affect_z3_on_impact(self, model_4var):
        """Shock eps1 must not affect z[3] (control in subsystem 2) on impact."""
        G0, G1, Psi, Pi, _ = model_4var
        sol = gensys(G0, G1, Psi, Pi)
        np.testing.assert_allclose(
            sol.Impact[3, 0], 0.0, atol=1e-12,
            err_msg="eps1 should not affect z[3] (subsystem 2 control)",
        )

    def test_shock2_does_not_affect_z0_on_impact(self, model_4var):
        """Shock eps2 must not affect z[0] (state in subsystem 1) on impact."""
        G0, G1, Psi, Pi, _ = model_4var
        sol = gensys(G0, G1, Psi, Pi)
        np.testing.assert_allclose(
            sol.Impact[0, 1], 0.0, atol=1e-12,
            err_msg="eps2 should not affect z[0] (subsystem 1 state)",
        )

    def test_shock2_does_not_affect_z2_on_impact(self, model_4var):
        """Shock eps2 must not affect z[2] (control in subsystem 1) on impact."""
        G0, G1, Psi, Pi, _ = model_4var
        sol = gensys(G0, G1, Psi, Pi)
        np.testing.assert_allclose(
            sol.Impact[2, 1], 0.0, atol=1e-12,
            err_msg="eps2 should not affect z[2] (subsystem 1 control)",
        )

    def test_shock1_irf_never_spreads_to_z1(self, model_4var):
        """At every IRF horizon, eps1 shock does not reach z[1]."""
        G0, G1, Psi, Pi, _ = model_4var
        sol = gensys(G0, G1, Psi, Pi)
        irf = np.zeros((20, 4))
        irf[0] = sol.Impact[:, 0]
        for t in range(1, 20):
            irf[t] = sol.G @ irf[t - 1]
        np.testing.assert_allclose(
            irf[:, 1], 0.0, atol=1e-12,
            err_msg="z[1] IRF to eps1 shock should be zero at all horizons",
        )

    def test_shock1_irf_never_spreads_to_z3(self, model_4var):
        """At every IRF horizon, eps1 shock does not reach z[3]."""
        G0, G1, Psi, Pi, _ = model_4var
        sol = gensys(G0, G1, Psi, Pi)
        irf = np.zeros((20, 4))
        irf[0] = sol.Impact[:, 0]
        for t in range(1, 20):
            irf[t] = sol.G @ irf[t - 1]
        np.testing.assert_allclose(
            irf[:, 3], 0.0, atol=1e-12,
            err_msg="z[3] IRF to eps1 shock should be zero at all horizons",
        )


# ---------------------------------------------------------------------------
# Known-value: geometric IRF decay at AR persistence rate
# ---------------------------------------------------------------------------


class TestGeometricDecay:
    """The AR(1) state decays at exactly its own persistence rho_i after h >= 2.

    In the 4-variable block model the AR states decouple: the IRF of z[0]
    to shock eps1 decays as rho1^h once the transient settles, and similarly
    for z[1] to shock eps2.  The ratio irf[h+1]/irf[h] must converge to the
    respective rho_i.
    """

    def test_z0_irf_decays_at_rho1(self, model_4var):
        """z[0] IRF ratio to shock eps1 equals rho1 after h=2."""
        G0, G1, Psi, Pi, p = model_4var
        sol = gensys(G0, G1, Psi, Pi)
        irf = np.zeros((15, 4))
        irf[0] = sol.Impact[:, 0]
        for t in range(1, 15):
            irf[t] = sol.G @ irf[t - 1]
        # Compute decay ratios starting from h=2 (skip impact and first step)
        ratios = irf[3:, 0] / irf[2:-1, 0]
        np.testing.assert_allclose(
            ratios, p["rho1"], atol=1e-10,
            err_msg=f"z[0] IRF should decay at rho1={p['rho1']} after h=2",
        )

    def test_z2_irf_decays_at_rho1(self, model_4var):
        """z[2] (control) IRF ratio to shock eps1 equals rho1 after h=3."""
        G0, G1, Psi, Pi, p = model_4var
        sol = gensys(G0, G1, Psi, Pi)
        irf = np.zeros((15, 4))
        irf[0] = sol.Impact[:, 0]
        for t in range(1, 15):
            irf[t] = sol.G @ irf[t - 1]
        ratios = irf[4:, 2] / irf[3:-1, 2]
        np.testing.assert_allclose(
            ratios, p["rho1"], atol=1e-10,
            err_msg=f"z[2] IRF should decay at rho1={p['rho1']}",
        )

    def test_z1_irf_decays_at_rho2(self, model_4var):
        """z[1] IRF ratio to shock eps2 equals rho2 after h=2."""
        G0, G1, Psi, Pi, p = model_4var
        sol = gensys(G0, G1, Psi, Pi)
        irf = np.zeros((15, 4))
        irf[0] = sol.Impact[:, 1]
        for t in range(1, 15):
            irf[t] = sol.G @ irf[t - 1]
        ratios = irf[3:, 1] / irf[2:-1, 1]
        np.testing.assert_allclose(
            ratios, p["rho2"], atol=1e-10,
            err_msg=f"z[1] IRF should decay at rho2={p['rho2']}",
        )

    def test_z3_irf_decays_at_rho2(self, model_4var):
        """z[3] (control) IRF ratio to shock eps2 equals rho2 after h=3."""
        G0, G1, Psi, Pi, p = model_4var
        sol = gensys(G0, G1, Psi, Pi)
        irf = np.zeros((15, 4))
        irf[0] = sol.Impact[:, 1]
        for t in range(1, 15):
            irf[t] = sol.G @ irf[t - 1]
        ratios = irf[4:, 3] / irf[3:-1, 3]
        np.testing.assert_allclose(
            ratios, p["rho2"], atol=1e-10,
            err_msg=f"z[3] IRF should decay at rho2={p['rho2']}",
        )

    def test_irf_decays_to_zero(self, model_2var):
        """IRF converges to zero for stable system (last 5 horizons near 0)."""
        G0, G1, Psi, Pi, _ = model_2var
        sol = gensys(G0, G1, Psi, Pi)
        irf = np.zeros((60, 2))
        irf[0] = sol.Impact[:, 0]
        for t in range(1, 60):
            irf[t] = sol.G @ irf[t - 1]
        np.testing.assert_allclose(
            irf[-5:], 0.0, atol=1e-8,
            err_msg="IRF should converge to zero for stable model",
        )


# ---------------------------------------------------------------------------
# Input robustness: type conversion
# ---------------------------------------------------------------------------


class TestInputConversion:
    """gensys accepts Python lists and integer arrays; outputs are float64."""

    def test_python_lists_accepted(self):
        """Python nested lists are accepted as input (not just numpy arrays)."""
        G0 = [[1.0, 0.0], [0.0, 0.8]]
        G1 = [[0.7, 0.0], [-1.0, 1.0]]
        Psi = [[1.0], [0.0]]
        Pi = [[0.0], [0.8]]
        sol = gensys(G0, G1, Psi, Pi)
        assert sol.eu == (1, 1)

    def test_G_output_is_float64_from_float_input(self):
        """G and Impact are float64 regardless of input dtype."""
        G0 = np.array([[1.0, 0.0], [0.0, 0.8]])
        G1 = np.array([[0.7, 0.0], [-1.0, 1.0]])
        Psi = np.array([[1.0], [0.0]])
        Pi = np.array([[0.0], [0.8]])
        sol = gensys(G0, G1, Psi, Pi)
        assert sol.G.dtype == np.float64
        assert sol.Impact.dtype == np.float64

    def test_integer_arrays_produce_float_output(self):
        """Integer-valued numpy arrays are upcast to float internally."""
        G0 = np.eye(2, dtype=int)
        G1 = np.diag([0, 0])  # int zeros
        Psi = np.eye(2, dtype=int)
        Pi = np.zeros((2, 1), dtype=int)
        sol = gensys(G0, G1, Psi, Pi)
        # Output must be float even when inputs are int
        assert np.issubdtype(sol.G.dtype, np.floating)
        assert np.issubdtype(sol.Impact.dtype, np.floating)


# ---------------------------------------------------------------------------
# div threshold parameter
# ---------------------------------------------------------------------------


class TestDivParameter:
    """The `div` keyword shifts the stable/unstable classification boundary."""

    def test_default_div_gives_unique_solution(self, model_2var):
        """Default div produces eu=(1,1) for the standard model."""
        G0, G1, Psi, Pi, _ = model_2var
        sol = gensys(G0, G1, Psi, Pi)
        assert sol.eu == (1, 1)

    def test_div_explicit_one_gives_unique_solution(self, model_2var):
        """div=1.0 gives the same eu=(1,1) as the default for the 2-var model."""
        G0, G1, Psi, Pi, _ = model_2var
        sol = gensys(G0, G1, Psi, Pi, div=1.0)
        assert sol.eu == (1, 1)

    def test_large_div_reclassifies_unstable_as_stable(self, model_2var):
        """With div >> 1/beta, the unstable eigenvalue is classified stable,
        n_unstable = 0 != n_eta = 1, and BK fails -> eu=(0,0)."""
        G0, G1, Psi, Pi, p = model_2var
        # 1/beta ≈ 1.25; with div=2.0 all eigenvalues (0.7, 1.25) are 'stable'
        sol = gensys(G0, G1, Psi, Pi, div=2.0)
        assert sol.eu == (0, 0)

    def test_div_changes_eigenvalue_count(self, model_2var):
        """Changing div shifts how many eigenvalues are counted as stable."""
        G0, G1, Psi, Pi, p = model_2var
        sol_default = gensys(G0, G1, Psi, Pi)
        sol_large   = gensys(G0, G1, Psi, Pi, div=2.0)
        # With default div, 1 stable eig; with div=2, 2 stable eigs
        n_stable_default = int(np.sum(sol_default.eigenvalues < 1.0 + 1e-6))
        n_stable_large   = int(np.sum(sol_large.eigenvalues < 2.0 + 1e-6))
        assert n_stable_default == 1
        assert n_stable_large == 2


# ---------------------------------------------------------------------------
# BK success (eu=(1,1)) smoke test
# ---------------------------------------------------------------------------


class TestBKSuccess:
    """Sanity checks confirming eu=(1,1) for correctly specified models."""

    def test_2var_eu_is_one_one(self, model_2var):
        """2-variable model has unique stable solution."""
        G0, G1, Psi, Pi, _ = model_2var
        sol = gensys(G0, G1, Psi, Pi)
        assert sol.eu == (1, 1)

    def test_4var_eu_is_one_one(self, model_4var):
        """4-variable model has unique stable solution."""
        G0, G1, Psi, Pi, _ = model_4var
        sol = gensys(G0, G1, Psi, Pi)
        assert sol.eu == (1, 1)

    def test_multiple_shocks_eu_is_one_one(self):
        """2-variable model with 2 shocks has unique stable solution."""
        G0 = np.array([[1.0, 0.0], [0.0, 0.8]])
        G1 = np.array([[0.7, 0.0], [-1.0, 1.0]])
        Psi = np.array([[1.0, 0.5], [0.0, 0.3]])
        Pi = np.array([[0.0], [0.8]])
        sol = gensys(G0, G1, Psi, Pi)
        assert sol.eu == (1, 1)

    def test_zero_shocks_eu_is_one_one(self):
        """Zero-shock model still returns eu=(1,1) when BK is satisfied."""
        G0 = np.array([[1.0, 0.0], [0.0, 0.8]])
        G1 = np.array([[0.7, 0.0], [-1.0, 1.0]])
        Psi = np.zeros((2, 0))
        Pi = np.array([[0.0], [0.8]])
        sol = gensys(G0, G1, Psi, Pi)
        assert sol.eu == (1, 1)
