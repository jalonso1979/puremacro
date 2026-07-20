"""Smoke tests for the F3 example scripts.

These are intentionally permissive: they verify only that each example
runs end-to-end without errors and returns sensibly shaped output.
Tight numerical assertions live in the dedicated replication tests.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PARQUET = Path("data/processed/panel_Q.parquet")


@pytest.fixture(scope="session")
def parquet_or_skip():
    if not PARQUET.exists():
        pytest.skip("panel_Q.parquet not available")
    return PARQUET


# ---------------------------------------------------------------------------
# 1. Caldara-Iacoviello (real data; skip if parquet missing)
# ---------------------------------------------------------------------------
def test_caldara_iacoviello_smoke(parquet_or_skip):
    from puremacro.examples.caldara_iacoviello import run_caldara_iacoviello

    out = run_caldara_iacoviello(panel_path=parquet_or_skip)
    assert "irf_point" in out
    assert isinstance(out["irf_point"], pd.DataFrame)
    assert "log_gdp_real" in out["irf_point"].columns
    # At least one finite value in the GDP IRF column.
    assert np.isfinite(out["irf_point"]["log_gdp_real"].values).any()


# ---------------------------------------------------------------------------
# 2. Sign restrictions (Uhlig 2005), synthetic
# ---------------------------------------------------------------------------
def test_sign_restrictions_uhlig_smoke():
    from puremacro.examples.sign_restrictions_uhlig import run_uhlig

    out = run_uhlig(seed=0, T=300, horizon=12, n_draws=300)
    assert "median_irf" in out
    median = out["median_irf"]
    assert isinstance(median, np.ndarray)
    # Shape (H+1, n, n) with n = 3 and H = 12.
    assert median.shape == (13, 3, 3)
    # Sign restrictions imposed at h=0: var0 (output) and var1 (prices)
    # should be non-positive; var2 (rate) non-negative for the target shock.
    assert median[0, 0, 0] <= 0
    assert median[0, 1, 0] <= 0
    assert median[0, 2, 0] >= 0


# ---------------------------------------------------------------------------
# 3. Panel LP with Driscoll-Kraay (real data; skip if parquet missing)
# ---------------------------------------------------------------------------
def test_lp_panel_dk_smoke(parquet_or_skip):
    from puremacro.examples.lp_panel_dk import run_panel_dk

    irf = run_panel_dk(panel_path=parquet_or_skip,
                        horizons=range(0, 5), n_lags=2)
    assert isinstance(irf, pd.DataFrame)
    for col in ("h", "beta", "se", "lo", "hi"):
        assert col in irf.columns
    assert len(irf) == 5
    assert np.isfinite(irf["beta"].values).any()


# ---------------------------------------------------------------------------
# 4. Smoothed LP (Barnichon-Brownlees), synthetic
# ---------------------------------------------------------------------------
def test_lp_smooth_demo_smoke():
    from puremacro.examples.lp_smooth_demo import run_smooth_lp

    out = run_smooth_lp(T=300, seed=0, horizons=range(0, 11), n_knots=5)
    irf = out["irf_smooth"]
    assert isinstance(irf, pd.DataFrame)
    for col in ("h", "beta", "se", "lo", "hi", "lambda"):
        assert col in irf.columns
    assert len(irf) == 11
    assert np.isfinite(irf["beta"].values).all()


# ---------------------------------------------------------------------------
# 5. Asymmetric LP (Tenreyro-Thwaites), synthetic
# ---------------------------------------------------------------------------
def test_lp_asymmetric_tenreyro_smoke():
    from puremacro.examples.lp_asymmetric_tenreyro import run_asymmetric_lp

    out = run_asymmetric_lp(T=400, seed=0, horizons=range(0, 7))
    assert "irf_pos" in out and "irf_neg" in out
    pos = out["irf_pos"]
    neg = out["irf_neg"]
    assert isinstance(pos, pd.DataFrame) and isinstance(neg, pd.DataFrame)
    assert "beta_pos" in pos.columns and "beta_neg" in neg.columns
    assert len(pos) == 7 and len(neg) == 7
    assert np.isfinite(pos["beta_pos"].values).any()
    assert np.isfinite(neg["beta_neg"].values).any()


# ---------------------------------------------------------------------------
# 6. Cointegration (Engle-Granger + Johansen + VECM), synthetic
# ---------------------------------------------------------------------------
def test_cointegration_johansen_smoke():
    from puremacro.examples.cointegration_johansen import run_johansen

    out = run_johansen(T=400, seed=0, p=2)
    eg = out["engle_granger"]
    joh = out["johansen"]
    vecm = out["vecm"]
    assert "cointegrating_beta" in eg
    assert "trace_stats" in joh and "crit_values" in joh
    # In a system with one common stochastic trend and r=1 cointegrating
    # rank, the trace stat at H_0: r=0 should exceed its critical value.
    assert joh["trace_stats"][0] > joh["crit_values"][0]
    # VECM: Pi has the right shape (n, n).
    assert vecm["Pi"].shape == (3, 3)


# ---------------------------------------------------------------------------
# 7. Diebold-Yilmaz spillover index, synthetic
# ---------------------------------------------------------------------------
def test_diebold_yilmaz_spillover_smoke():
    from puremacro.examples.diebold_yilmaz_spillover import run_diebold_yilmaz

    out = run_diebold_yilmaz(T=400, seed=0, var_lags=2, fevd_horizon=8)
    pm = out["pairwise_matrix"]
    assert pm.shape == (4, 4)
    assert 0.0 <= float(out["total"]) <= 100.0
    for key in ("directional_to", "directional_from", "net"):
        assert out[key].shape == (4,)
