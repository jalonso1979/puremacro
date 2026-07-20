import numpy as np
import pandas as pd

from puremacro.var import fit_var
from puremacro.var.irf import historical_decomp


def test_historical_decomp_sum_recovers_y():
    """The sum across shock contributions + deterministic = original y_t."""
    rng = np.random.default_rng(11)
    T, n = 200, 2
    A = np.array([[0.5, 0.1], [0.0, 0.6]])
    Sigma = np.array([[1.0, 0.2], [0.2, 1.0]])
    L = np.linalg.cholesky(Sigma)
    Y = np.zeros((T, n))
    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + L @ rng.standard_normal(n)
    df = pd.DataFrame(Y, columns=["y0", "y1"])
    res = fit_var(df, p=1)
    if len(res) == 5:
        A_list, c, Sigma_hat, residuals, _ = res
    else:
        A_list, Sigma_hat, residuals, c = res
    # Cholesky factor:
    B0 = np.linalg.cholesky(Sigma_hat)
    out = historical_decomp(A_list, B0, residuals, init_y=Y[:1], intercept=c)
    # Sum across shocks (axis=2 of [t, var, shock]) + deterministic
    # should recover Y[1:T_eff+1] (after the first p observations are
    # used as initial conditions).
    p = len(A_list)
    Y_eff = Y[p: p + out["shocks"].shape[0]]
    reconstructed = out["shocks"].sum(axis=2) + out["deterministic"]
    assert np.allclose(reconstructed, Y_eff, atol=1e-8), (
        f"max diff = {np.max(np.abs(reconstructed - Y_eff))}"
    )
