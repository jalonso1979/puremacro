import numpy as np
import pandas as pd

from puremacro.var.bvar import minnesota_posterior


def test_minnesota_posterior_returns_documented_keys():
    rng = np.random.default_rng(11)
    T, n = 200, 2
    A_true = np.array([[0.7, 0.1], [0.0, 0.5]])
    Y = np.zeros((T, n))
    for t in range(1, T):
        Y[t] = A_true @ Y[t - 1] + rng.standard_normal(n)
    df = pd.DataFrame(Y, columns=["y0", "y1"])
    out = minnesota_posterior(df, p=2, lambda1=0.2)
    for k in ("A_list", "intercept", "Sigma", "lambda1", "lambda2", "lambda3"):
        assert k in out
    assert len(out["A_list"]) == 2
    assert out["A_list"][0].shape == (2, 2)


def test_minnesota_shrinkage_pulls_toward_random_walk():
    """As λ₁ → 0 (tight prior), the posterior coefficients shrink toward the
    AR(1) prior (1 on own first lag, 0 elsewhere)."""
    rng = np.random.default_rng(13)
    T, n = 100, 3
    A_true = 0.5 * np.eye(n) + 0.05 * rng.standard_normal((n, n))
    Y = np.zeros((T, n))
    for t in range(1, T):
        Y[t] = A_true @ Y[t - 1] + rng.standard_normal(n) * 0.5
    df = pd.DataFrame(Y, columns=[f"y{i}" for i in range(n)])
    out_loose = minnesota_posterior(df, p=1, lambda1=10.0)
    out_tight = minnesota_posterior(df, p=1, lambda1=0.001)
    # Diagonal of A1 should be closer to 1 under tight prior
    diag_loose = np.diag(out_loose["A_list"][0])
    diag_tight = np.diag(out_tight["A_list"][0])
    assert np.mean(np.abs(diag_tight - 1.0)) < np.mean(np.abs(diag_loose - 1.0))
    # Off-diagonal of A1 should be closer to 0 under tight prior
    A_loose = out_loose["A_list"][0]
    A_tight = out_tight["A_list"][0]
    off_loose = A_loose[~np.eye(n, dtype=bool)]
    off_tight = A_tight[~np.eye(n, dtype=bool)]
    assert np.mean(np.abs(off_tight)) < np.mean(np.abs(off_loose))


def test_minnesota_loose_prior_close_to_ols():
    """As λ₁ → ∞, the posterior should converge to OLS."""
    from puremacro.var import fit_var
    rng = np.random.default_rng(17)
    T, n = 200, 2
    A_true = np.array([[0.6, 0.1], [0.05, 0.55]])
    Y = np.zeros((T, n))
    for t in range(1, T):
        Y[t] = A_true @ Y[t - 1] + rng.standard_normal(n) * 0.4
    df = pd.DataFrame(Y, columns=["y0", "y1"])
    res = fit_var(np.asarray(df), p=1)
    if len(res) == 5:
        A_ols = res[0][0]
    else:
        A_ols = res[0][0]
    out = minnesota_posterior(df, p=1, lambda1=1e6)
    A_bvar = out["A_list"][0]
    # Should be very close to OLS
    assert np.max(np.abs(A_bvar - A_ols)) < 1e-3
