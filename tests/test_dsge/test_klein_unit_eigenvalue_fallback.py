"""Klein-solver fallback test: when the Z-partition is degenerate (the
closed-form ``F = -inv(Z22) Z21`` is corrupted by static-control rows
producing inf generalised eigenvalues whose finite-side counterparts
mix into the QZ stable block), ``klein_solve`` should detect the
condition via the equilibrium residual and recover F via the Sylvester
equilibrium equation.

These tests parallel the SW07 pathology in miniature. In SW07, the
16 static-control equations each have ``A[row, :] = 0``, producing inf
generalised eigenvalues. The QZ stable-first ordering surfaces the
near-zero finite counterparts inside Z11, biasing the closed-form F
even though ``cond(Z11) ~ 16`` (well-conditioned by the usual norm).
The degenerate small case below reproduces exactly this structure:
A row 2 is zero (static control), making the closed-form F have a
~3% bias relative to the true policy function.

Note on the construction
~~~~~~~~~~~~~~~~~~~~~~~~
For the degenerate case we use a static-control Sims form

    [I, 0; 0, 0] E_t z_{t+1} = [G_x, 0; -F_true, I] z_t

which encodes ``x_{t+1} = G_x x_t`` and ``y_t = F_true x_t`` directly.
The zero second-block row of A is the SW07-style "static control"
pattern; it is what mixes the QZ stable block. The Sylvester fallback
is the canonical fix.

For the well-conditioned case we use a forward-looking Euler equation
(``y_t = beta E_t y_{t+1} + phi x_t``), which has a strictly stable
forward root and a clean Z partition; the closed-form Klein returns
the exact F there.
"""
import numpy as np
import pytest

from puremacro.dsge.klein import klein_solve


def test_klein_unit_eigenvalue_triggers_sylvester_fallback():
    """Static-control system whose QZ stable block is mixed by an inf
    generalised eigenvalue. The closed-form F is biased; the Sylvester
    fallback recovers F_true to machine precision.

    The same pathology as SW07 (cond(Z11) is fine, but the closed-form
    F has a non-trivial equilibrium residual), at minimum size.
    """
    n_pre = 2
    n_fwd = 1
    n = n_pre + n_fwd

    G_x = np.array([[0.7, 0.1], [0.0, 0.5]])
    F_true = np.array([[0.3, -0.2]])

    # Sims form: A E_t z_{t+1} = B z_t.
    # Rows 0..n_pre-1: x_{t+1} = G_x x_t  →  A row = e_i, B row = G_x row.
    # Rows n_pre..   : 0 = -F_true x_t + y_t  →  A row = 0, B row = [-F_true, I].
    A = np.zeros((n, n))
    A[:n_pre, :n_pre] = np.eye(n_pre)
    B = np.zeros((n, n))
    B[:n_pre, :n_pre] = G_x
    B[n_pre:, :n_pre] = -F_true
    B[n_pre:, n_pre:] = np.eye(n_fwd)

    sol = klein_solve(A, B, n_pre=n_pre, strict=False)

    # eu must indicate a valid solution.
    assert sol.eu == (1, 1), f"BK condition failed: eu={sol.eu}"

    # G recovers G_x.
    np.testing.assert_allclose(sol.G, G_x, atol=1e-12)

    # F recovers F_true via the fallback. The closed-form Klein F is
    # biased by the static-control inf eigenvalue mixing into the QZ
    # stable block; the Sylvester fallback recovers the true F.
    np.testing.assert_allclose(sol.F, F_true, atol=1e-10)


@pytest.mark.pyodide_smoke
def test_klein_stable_system_uses_closed_form_path():
    """Forward-looking Euler equation:  y_t = beta E_t y_{t+1} + phi x1_t,
    with two stable AR(1) exogenous states. Closed-form Klein returns
    the exact F = phi/(1 - beta*rho1) on x1 (zero on x2).

    The detection guard inspects the equilibrium residual; for this
    well-conditioned case the residual is identically zero and the
    closed-form path is taken (the Sylvester branch is not exercised).
    """
    n_pre = 2
    n_fwd = 1
    n = n_pre + n_fwd

    beta_p = 0.95
    phi = 0.5
    rho1, rho2 = 0.7, 0.5
    G_x = np.diag([rho1, rho2])
    F_true = np.array([[phi / (1.0 - beta_p * rho1), 0.0]])

    # Sims form: state rows are AR(1); forward row is the Euler
    #   beta E_t y_{t+1} = y_t - phi x1_t
    # so A[ctrl, ctrl] = beta, B[ctrl, :n_pre] = [-phi, 0], B[ctrl, ctrl] = 1.
    A = np.zeros((n, n))
    A[:n_pre, :n_pre] = np.eye(n_pre)
    A[n_pre, n_pre] = beta_p
    B = np.zeros((n, n))
    B[:n_pre, :n_pre] = G_x
    B[n_pre, 0] = -phi
    B[n_pre, n_pre] = 1.0

    sol = klein_solve(A, B, n_pre=n_pre, strict=False)

    assert sol.eu == (1, 1)
    np.testing.assert_allclose(sol.G, G_x, atol=1e-12)
    np.testing.assert_allclose(sol.F, F_true, atol=1e-10)
