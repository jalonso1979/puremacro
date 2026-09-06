"""Shift-share IV: point estimate, Rotemberg weights and AKM coverage."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from puremacro.bartik import ShiftShareIVResult, shift_share_iv


def _dgp(rng, n=300, K=20, beta=1.0, sector_noise=1.0):
    S = rng.dirichlet(np.full(K, 0.5), size=n)
    g = rng.standard_normal(K)
    z = S @ g
    x = z + rng.standard_normal(n)
    eta = rng.standard_normal(K) * sector_noise          # sector-level unobservable
    eps = S @ eta + 0.5 * rng.standard_normal(n)
    y = beta * x + eps
    return pd.DataFrame({"y": y, "x": x, "c1": rng.standard_normal(n)}), S, g


def test_point_estimate_matches_manual_2sls_and_weights_sum_to_one():
    rng = np.random.default_rng(0)
    df, S, g = _dgp(rng)
    res = shift_share_iv(df, "y", "x", S, g, controls=["c1"])
    assert isinstance(res, ShiftShareIVResult)
    C = np.column_stack([np.ones(len(df)), df["c1"]])
    M = np.eye(len(df)) - C @ np.linalg.pinv(C)
    z, x, y = M @ (S @ g), M @ df["x"].to_numpy(), M @ df["y"].to_numpy()
    assert res.beta == pytest.approx(float(z @ y / (z @ x)), rel=1e-10)
    assert res.rotemberg_weights.sum() == pytest.approx(1.0, abs=1e-10)
    assert res.first_stage_F > 10
    assert res.n_units == 300 and res.n_sectors == 20
    assert res.se == res.se_akm and res.se_type == "akm"
    rob = shift_share_iv(df, "y", "x", S, g, controls=["c1"], se="robust")
    assert rob.se == rob.se_robust and rob.beta == pytest.approx(res.beta)


def test_akm_covers_when_robust_under_covers():
    """Sector-level unobservables make unit-level residuals correlated through
    the shares; the AKM shock-level error stays close to nominal while the
    robust error under-covers (Adão, Kolesár & Morales 2019)."""
    rng = np.random.default_rng(11)
    z = stats.norm.ppf(0.975)
    reps, hit_akm, hit_rob = 300, 0, 0
    for _ in range(reps):
        df, S, g = _dgp(rng, n=400, K=50)
        res = shift_share_iv(df, "y", "x", S, g)
        hit_akm += abs(res.beta - 1.0) <= z * res.se_akm
        hit_rob += abs(res.beta - 1.0) <= z * res.se_robust
    cov_akm, cov_rob = hit_akm / reps, hit_rob / reps
    # 400 replications at these settings give ~0.97 (AKM) vs ~0.90 (robust);
    # with only 25 sectors the AKM interval itself drops to ~0.92, the
    # few-shocks under-coverage documented in AKM's own simulations.
    assert cov_akm >= 0.93, cov_akm
    assert cov_rob < cov_akm - 0.04, (cov_rob, cov_akm)


def test_pandas_alignment_and_validation():
    rng = np.random.default_rng(2)
    df, S, g = _dgp(rng, n=80, K=6)
    df.index = [f"u{i}" for i in range(len(df))]
    shares = pd.DataFrame(S, index=df.index, columns=[f"s{k}" for k in range(6)])
    shocks = pd.Series(g, index=shares.columns)
    a = shift_share_iv(df, "y", "x", shares, shocks)
    b = shift_share_iv(df, "y", "x", shares.iloc[::-1], shocks.iloc[::-1])  # shuffled labels
    assert a.beta == pytest.approx(b.beta) and a.se_akm == pytest.approx(b.se_akm)
    assert list(a.rotemberg_weights.index) == list(shares.columns)
    with pytest.raises(KeyError):
        shift_share_iv(df, "y", "x", shares.iloc[:-1], shocks)
    with pytest.raises(ValueError, match="sectors"):
        shift_share_iv(df, "y", "x", S, g[:-1])
    with pytest.raises(ValueError, match="non-negative"):
        shift_share_iv(df, "y", "x", -S, g)
    with pytest.raises(ValueError, match="se must"):
        shift_share_iv(df, "y", "x", S, g, se="cluster")
    with pytest.raises(ValueError, match="two sectors"):
        shift_share_iv(df, "y", "x", S[:, :1], g[:1])


def test_presentation_contract():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(5)
    df, S, g = _dgp(rng, n=120, K=8)
    res = shift_share_iv(df, "y", "x", S, g, weights=np.ones(120))
    assert "Shift-share IV" in res.summary()
    assert res.to_markdown().startswith("|")
    assert "tabular" in res.to_latex()
    assert res.to_typst().startswith("#table(")
    assert {"statistic", "value"} <= set(res.to_frame().columns)
    assert res.plot() is not None
    plt.close("all")
