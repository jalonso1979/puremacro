import numpy as np
import pandas as pd
import pytest

from puremacro.lp.panel import panel_lp
from puremacro.lp.panel_dk import panel_lp_dk
from puremacro.lp.panel_iv import panel_lp_iv


def _balanced_panel(rng, N=8, T=60, beta=0.5):
    """y_{c,t+1} = β x_{c,t} + entity_fe + time_fe + noise."""
    rows = []
    entity_fe = rng.standard_normal(N) * 1.5
    time_fe = rng.standard_normal(T) * 0.3
    for i in range(N):
        x = rng.standard_normal(T)
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = entity_fe[i] + time_fe[t] + beta * x[t - 1] + rng.standard_normal() * 0.4
        for t in range(T):
            rows.append({
                "code": f"E{i:02d}",
                "date": pd.Timestamp("2010-01-01") + pd.DateOffset(months=t),
                "x": x[t], "y": y[t],
            })
    return pd.DataFrame(rows).set_index(["code", "date"]).sort_index()


def test_panel_lp_recovers_known_slope():
    rng = np.random.default_rng(7)
    df = _balanced_panel(rng, N=10, T=80, beta=0.5)
    out = panel_lp(df, y="y", x="x", horizons=[0, 1, 2], n_lags=1)
    # IRF at h=1: dy_{t+1} - dy_{t-1} per unit x_t. With y_{t+1} = 0.5 x_t + ...
    # the LP β at h=1 should be ≈ 0.5.
    assert abs(out.loc[out["h"] == 1, "beta"].iloc[0] - 0.5) < 0.15


def test_panel_lp_dk_returns_documented_columns():
    rng = np.random.default_rng(11)
    df = _balanced_panel(rng, N=8, T=60, beta=0.4)
    out = panel_lp_dk(df, y="y", x="x", horizons=range(0, 5), n_lags=1)
    assert {"h", "beta", "se", "lo", "hi"}.issubset(out.columns)
    assert np.all(np.isfinite(out["se"]))


def test_panel_lp_dk_engine_does_not_redo_demeaning():
    """Regression guard for the 0.2.0 panel-LP refactor.

    Both ``panel_lp`` and ``panel_lp_dk`` now share
    ``panel_lp_horizon_loop``; the DK variant reuses the within-projected
    design returned by ``two_way_fe_within`` instead of redoing the
    iterative demeaning a second time. If a future change reintroduces
    the duplicate work, the DK runtime will balloon — this test caps
    the ratio.
    """
    import time

    rng = np.random.default_rng(31)
    df = _balanced_panel(rng, N=12, T=120, beta=0.5)
    horizons = list(range(0, 21))

    # Warm import / numpy state.
    panel_lp(df, y="y", x="x", horizons=horizons, n_lags=2)
    panel_lp_dk(df, y="y", x="x", horizons=horizons, n_lags=2)

    t0 = time.perf_counter()
    for _ in range(3):
        panel_lp(df, y="y", x="x", horizons=horizons, n_lags=2)
    cluster_dt = (time.perf_counter() - t0) / 3

    t0 = time.perf_counter()
    for _ in range(3):
        panel_lp_dk(df, y="y", x="x", horizons=horizons, n_lags=2)
    dk_dt = (time.perf_counter() - t0) / 3

    # The DK variant has additional per-horizon work (Driscoll-Kraay
    # meat construction) so a small overhead is expected; a 3× factor
    # would mean the iterative demeaning is being repeated.
    assert dk_dt < 3.0 * cluster_dt, (
        f"panel_lp_dk took {dk_dt:.3f}s vs panel_lp {cluster_dt:.3f}s "
        f"(ratio {dk_dt / cluster_dt:.2f}). The shared-engine refactor "
        "may have regressed — check that two_way_fe_within's "
        "X_within / XtX_inv are still being reused in _focal_dk_se."
    )


def test_panel_lp_and_panel_lp_dk_share_beta():
    """The cluster and Driscoll-Kraay variants differ only in the SE
    formula, not in the point estimate. After the shared-engine refactor
    this should hold to machine precision."""
    rng = np.random.default_rng(17)
    df = _balanced_panel(rng, N=8, T=70, beta=0.6)
    cl = panel_lp(df, y="y", x="x", horizons=range(0, 6), n_lags=1)
    dk = panel_lp_dk(df, y="y", x="x", horizons=range(0, 6), n_lags=1)
    np.testing.assert_allclose(cl["beta"].values, dk["beta"].values,
                               rtol=1e-12, atol=1e-12)
    # SEs should not be identical (different estimators) — sanity check.
    assert not np.allclose(cl["se"].values, dk["se"].values)


def test_panel_lp_iv_returns_first_stage_f():
    rng = np.random.default_rng(13)
    rows = []
    for i in range(6):
        z = rng.standard_normal(50)
        x = 0.7 * z + 0.3 * rng.standard_normal(50)
        y = np.zeros(50)
        for t in range(1, 50):
            y[t] = -0.4 * x[t - 1] + rng.standard_normal() * 0.4
        for t in range(50):
            rows.append({"code": f"E{i:02d}",
                         "date": pd.Timestamp("2010-01-01") + pd.DateOffset(months=t),
                         "x": x[t], "y": y[t], "z": z[t]})
    df = pd.DataFrame(rows).set_index(["code", "date"]).sort_index()
    out = panel_lp_iv(df, y="y", x="x", z="z",
                      horizons=range(0, 5), n_lags=1)
    assert "first_stage_f" in out.columns
    assert out["first_stage_f"].iloc[0] > 5  # strong instrument design
