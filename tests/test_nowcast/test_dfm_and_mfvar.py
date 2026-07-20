"""Tests for kalman_dfm and mf_var nowcasting estimators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.nowcast import kalman_dfm, mf_var


def _factor_panel(rng, T=300, n=8, k=2, idio_sd=0.4):
    """Synthetic dataset with k true factors and n indicators."""
    F_true = np.cumsum(rng.standard_normal((T, k)) * 0.05, axis=0)
    Lambda_true = rng.standard_normal((n, k))
    X = F_true @ Lambda_true.T + idio_sd * rng.standard_normal((T, n))
    return X, F_true, Lambda_true


def test_kalman_dfm_recovers_factor_space(rng):
    """The smoothed factors should span the same subspace as the true
    ones — measurable via canonical correlations."""
    X, F_true, _ = _factor_panel(rng, T=300, n=10, k=2, idio_sd=0.3)
    out = kalman_dfm(X, n_factors=2, p=1)
    F_hat = out["factors"]
    # Stack and compute canonical correlations between F_hat and F_true.
    A = F_hat - F_hat.mean(axis=0)
    B = F_true - F_true.mean(axis=0)
    A /= A.std(axis=0)
    B /= B.std(axis=0)
    # First canonical correlation via SVD of cross-cov.
    cross = (A.T @ B) / A.shape[0]
    s = np.linalg.svd(cross, compute_uv=False)
    assert s[0] > 0.85  # strong alignment of first canonical direction


def test_kalman_dfm_handles_ragged_edge(rng):
    """The Kalman approach must fill in NaNs at the bottom of the
    panel — the defining feature relative to static PCA."""
    X, _, _ = _factor_panel(rng, T=200, n=8, k=2)
    # Knock out the last 3 rows on half the columns (ragged edge).
    X_ragged = X.copy()
    X_ragged[-3:, :4] = np.nan
    out = kalman_dfm(X_ragged, n_factors=2, p=1)
    # X_filled should have no NaNs.
    assert not np.isnan(out["X_filled"]).any()
    # Filled values should approximate the true X within sane bounds.
    err = np.abs(out["X_filled"][-3:, :4] - X[-3:, :4]).max()
    assert err < 5 * X.std()


def test_kalman_dfm_returns_documented_keys(rng):
    X, _, _ = _factor_panel(rng, T=120, n=5, k=1)
    out = kalman_dfm(X, n_factors=1, p=1)
    for k in ("factors", "loadings", "A", "Q", "H", "X_filled",
              "means", "stds"):
        assert k in out


def test_kalman_dfm_dataframe_input_returns_dataframe(rng):
    X, _, _ = _factor_panel(rng, T=120, n=5, k=1)
    idx = pd.date_range("2010-01-01", periods=120, freq="MS")
    df = pd.DataFrame(X, index=idx, columns=[f"x{i}" for i in range(5)])
    out = kalman_dfm(df, n_factors=1, p=1)
    assert "factors_df" in out
    assert isinstance(out["factors_df"], pd.DataFrame)
    assert out["factors_df"].index.equals(idx)


# ---------------------------------------------------------------------------
# mf_var
# ---------------------------------------------------------------------------


def _mf_panel(rng, T_months=120, sigma_m=0.2, sigma_q=0.1):
    """Two monthly indicators + a quarterly variable observed every
    third month as the 3-month average of a latent monthly path."""
    # True latent monthly system.
    A = np.array([[0.7, 0.0, 0.1],
                   [0.0, 0.6, 0.0],
                   [0.1, 0.0, 0.5]])
    e = np.column_stack([
        sigma_m * rng.standard_normal(T_months),
        sigma_m * rng.standard_normal(T_months),
        sigma_q * rng.standard_normal(T_months),
    ])
    M = np.zeros((T_months, 3))
    for t in range(1, T_months):
        M[t] = A @ M[t - 1] + e[t]
    # Build the observed panel: monthly indicators each month, quarterly
    # column = 3-month average at quarter-end months only.
    idx = pd.date_range("2000-01-01", periods=T_months, freq="MS")
    df = pd.DataFrame(M, index=idx, columns=["m1", "m2", "q_latent"])
    df["q"] = np.nan
    for t in range(2, T_months):
        if (t + 1) % 3 == 0:                  # March, June, … (0-indexed t=2,5,…)
            df.iloc[t, df.columns.get_loc("q")] = (
                df["q_latent"].iloc[t - 2: t + 1].mean()
            )
    df_obs = df[["m1", "m2", "q"]].copy()
    return df_obs, df["q_latent"]


def test_mf_var_runs_and_returns_documented_keys(rng):
    df_obs, _ = _mf_panel(rng)
    out = mf_var(df_obs, quarterly_col="q", p=3)
    for k in ("A", "Q", "Z", "H", "factors_monthly",
              "df_monthly", "df_filled"):
        assert k in out


def test_mf_var_aggregation_constraint_holds_at_quarter_ends(rng):
    """At quarter-end months the smoothed monthly path of the quarterly
    variable should average to the published quarterly value."""
    df_obs, _ = _mf_panel(rng, T_months=180)
    out = mf_var(df_obs, quarterly_col="q", p=3)
    monthly_q = out["df_monthly"]["q_monthly"]
    for t in range(2, len(df_obs)):
        published = df_obs["q"].iloc[t]
        if not np.isnan(published):
            avg = monthly_q.iloc[t - 2: t + 1].mean()
            # The smoother enforces this exactly up to numerical noise.
            assert abs(avg - published) < 1e-3


def test_mf_var_rejects_p_below_3():
    df = pd.DataFrame({"m1": np.zeros(10), "q": np.zeros(10)})
    with pytest.raises(ValueError, match="p must be >= 3"):
        mf_var(df, quarterly_col="q", p=2)
