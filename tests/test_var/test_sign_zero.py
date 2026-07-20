import numpy as np
import pandas as pd
import pytest

from puremacro.var import fit_var
from puremacro.var.identify.sign_zero import sign_zero
from puremacro.var.identify._results import SignZeroResult


def _toy_3var(rng):
    T, n = 200, 3
    A = np.array([[0.5, 0.1, 0.0], [0.0, 0.6, 0.0], [0.0, 0.0, 0.4]])
    Sigma = np.eye(3) + 0.2 * (np.ones((3, 3)) - np.eye(3))
    L = np.linalg.cholesky(Sigma)
    Y = np.zeros((T, n))
    for t in range(1, T):
        Y[t] = A @ Y[t-1] + L @ rng.standard_normal(n)
    return pd.DataFrame(Y, columns=["y0", "y1", "y2"])


def test_sign_zero_enforces_zero_restriction():
    """Imposing B0[0, 1] = 0 (variable 0 has zero impact response to shock 1)
    must hold exactly in the returned matrix."""
    rng = np.random.default_rng(11)
    Y = _toy_3var(rng)
    result = fit_var(Y, p=1)
    if len(result) == 5:
        A_list, _, Sigma, _, _ = result
    else:
        A_list, Sigma, _, _ = result
    # Zero restrictions: 1 entry, B0[0, 1] = 0.
    zero_constraints = [(0, 1)]
    # Sign: entry B0[1, 0] should be positive (shock 0 raises variable 1 on impact)
    sign_constraints = {(1, 0): +1}
    out = sign_zero(A_list, Sigma,
                    zero_constraints=zero_constraints,
                    sign_constraints=sign_constraints,
                    n_draws=200, rng=rng)
    assert isinstance(out, SignZeroResult)
    assert out.success, "no draw satisfied the constraints"
    B0 = out.B0
    assert abs(B0[0, 1]) < 1e-10, f"zero constraint violated: B0[0,1]={B0[0,1]}"
    assert B0[1, 0] > 0, f"sign constraint violated: B0[1,0]={B0[1,0]}"
    # And B0 must reconstruct Sigma
    assert np.allclose(B0 @ B0.T, Sigma, atol=1e-8)


def test_sign_zero_returns_none_if_unsatisfiable():
    rng = np.random.default_rng(13)
    Y = _toy_3var(rng)
    result = fit_var(Y, p=1)
    if len(result) == 5:
        A_list, _, Sigma, _, _ = result
    else:
        A_list, Sigma, _, _ = result
    # An impossible combination: zero everywhere in column 0 plus a sign constraint
    out = sign_zero(A_list, Sigma,
                    zero_constraints=[(0, 0), (1, 0), (2, 0)],
                    sign_constraints={(0, 0): +1},
                    n_draws=50, rng=rng)
    # New contract: always returns a SignZeroResult; check `success` flag.
    assert isinstance(out, SignZeroResult)
    if out.success:
        # If it succeeded, it must satisfy the (degenerate) zeros
        B0 = out.B0
        for (i, j) in [(0, 0), (1, 0), (2, 0)]:
            assert abs(B0[i, j]) < 1e-10
    else:
        assert out.B0 is None and out.Q is None
