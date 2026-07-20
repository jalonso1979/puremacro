import numpy as np

from puremacro.var import fit_var, irf
from puremacro.var.identify.maxshare import maxshare, news_maxshare


def test_maxshare_returns_valid_B0(toy_var2):
    result = fit_var(toy_var2, p=1)
    if len(result) == 5:
        A_list, _, Sigma, _, _ = result
    else:
        A_list, Sigma, _, _ = result
    B0 = maxshare(A_list, Sigma, target_var=0, horizon=8)
    assert np.allclose(B0 @ B0.T, Sigma, atol=1e-8)


def test_maxshare_first_shock_dominates_target_variance():
    """The optimal rotation must, by construction, have the first shock
    explaining the largest share of Var(y_target, t+h*) at the chosen h*."""
    rng = np.random.default_rng(7)
    T, n = 200, 3
    A = np.array([[0.5, 0.1, 0.0], [0.0, 0.6, 0.05], [0.0, 0.0, 0.4]])
    Sigma = np.array([[1.0, 0.2, 0.1], [0.2, 1.0, 0.1], [0.1, 0.1, 1.0]])
    L = np.linalg.cholesky(Sigma)
    Y = np.zeros((T, n))
    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + L @ rng.standard_normal(n)
    import pandas as pd
    Y_df = pd.DataFrame(Y, columns=["y0", "y1", "y2"])
    result = fit_var(Y_df, p=1)
    if len(result) == 5:
        A_list, _, Sigma_hat, _, _ = result
    else:
        A_list, Sigma_hat, _, _ = result
    target = 0; H = 6
    B0 = maxshare(A_list, Sigma_hat, target_var=target, horizon=H)
    irf_arr = irf(A_list, B0, horizon=H)
    # Cumulative squared response of variable `target` to each shock
    cum_sq = (irf_arr[:H + 1, target, :] ** 2).sum(axis=0)
    # First shock should have the largest cumulative squared response
    assert cum_sq.argmax() == 0


def test_news_maxshare_uses_long_horizon():
    rng = np.random.default_rng(13)
    T, n = 200, 2
    A = np.array([[0.7, 0.1], [0.0, 0.5]])
    Sigma = np.eye(2)
    Y = np.zeros((T, n))
    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + rng.standard_normal(n)
    import pandas as pd
    Y_df = pd.DataFrame(Y, columns=["y0", "y1"])
    result = fit_var(Y_df, p=1)
    A_list, *_, Sigma_hat = (result[0], None, result[2], result[3], result[4]) if len(result) == 5 else (result[0], None, result[1], result[2], None)
    if len(result) == 5:
        A_list, _, Sigma_hat, _, _ = result
    else:
        A_list, Sigma_hat, _, _ = result
    B0 = news_maxshare(A_list, Sigma_hat, target_var=0, horizon=40)
    assert np.allclose(B0 @ B0.T, Sigma_hat, atol=1e-8)
