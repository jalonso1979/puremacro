"""Tests for puremacro.dynpanel.ab_gmm."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from puremacro.dynpanel import GMMResult, ab_gmm

from .conftest import simulate_dynamic_panel


def test_ab_returns_GMMResult():
    sim = simulate_dynamic_panel(N=80, T=6, rho=0.3, seed=0)
    res = ab_gmm(sim["y"], sim["panel_id"], sim["time_id"])
    assert isinstance(res, GMMResult)
    # frozen
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.coefs = np.zeros_like(res.coefs)


def test_ab_summary_runs():
    sim = simulate_dynamic_panel(N=80, T=6, rho=0.5, seed=0)
    res = ab_gmm(sim["y"], sim["panel_id"], sim["time_id"])
    s = res.summary()
    assert isinstance(s, str)
    assert len(s) > 0
    assert "Arellano-Bond" in s
    assert "L1.y" in s


def test_ab_recovers_persistence_low_persistence():
    """ρ=0.3 with N=200, T=8 — AB should be within 0.1 of truth."""
    sim = simulate_dynamic_panel(N=200, T=8, rho=0.3, seed=0)
    res = ab_gmm(sim["y"], sim["panel_id"], sim["time_id"])
    rho_hat = res.coefs[0]
    assert abs(rho_hat - 0.3) < 0.1, (
        f"ρ̂ = {rho_hat:.3f} for ρ_true = 0.3"
    )


def test_ab_estimator_label():
    sim = simulate_dynamic_panel(N=80, T=6, rho=0.5, seed=0)
    res = ab_gmm(sim["y"], sim["panel_id"], sim["time_id"])
    assert res.estimator == "ab"


def test_ab_high_persistence_downward_biased():
    """ρ=0.95 (high persistence): AB GMM is known to be downward-biased
    (Blundell-Bond 1998 Section 3, Roodman 2009). Loose threshold 0.85.
    """
    sim = simulate_dynamic_panel(N=200, T=8, rho=0.95, seed=42)
    res = ab_gmm(sim["y"], sim["panel_id"], sim["time_id"])
    rho_hat = res.coefs[0]
    assert rho_hat < 0.85, (
        f"AB at ρ=0.95: expected downward bias (<0.85), got {rho_hat:.3f}"
    )


def test_ab_lag_window_reduces_instruments():
    sim = simulate_dynamic_panel(N=100, T=8, rho=0.5, seed=0)
    res_full = ab_gmm(
        sim["y"], sim["panel_id"], sim["time_id"],
        gmm_lag_window=(2, None),
    )
    res_short = ab_gmm(
        sim["y"], sim["panel_id"], sim["time_id"],
        gmm_lag_window=(2, 3),
    )
    assert res_short.n_instruments < res_full.n_instruments


def test_ab_with_exogenous_regressor():
    """Strictly exogenous regressor recovered within tolerance."""
    sim = simulate_dynamic_panel(N=200, T=8, rho=0.3, beta_exog=0.7, seed=0)
    res = ab_gmm(
        sim["y"], sim["panel_id"], sim["time_id"],
        X_exog=sim["x"],
    )
    assert len(res.coefs) == 2
    rho_hat = res.coefs[0]
    beta_hat = res.coefs[1]
    assert abs(rho_hat - 0.3) < 0.15, f"ρ̂={rho_hat:.3f}"
    assert abs(beta_hat - 0.7) < 0.15, f"β̂={beta_hat:.3f}"


def test_ab_unbalanced_panel():
    """Drop ~10% of cells at random; estimator runs and recovers truth."""
    sim = simulate_dynamic_panel(N=200, T=8, rho=0.3, seed=42)
    rng = np.random.default_rng(7)
    keep = rng.uniform(0, 1, size=len(sim["y"])) > 0.10
    res = ab_gmm(
        sim["y"][keep],
        sim["panel_id"][keep],
        sim["time_id"][keep],
    )
    rho_hat = res.coefs[0]
    assert abs(rho_hat - 0.3) < 0.15, f"ρ̂={rho_hat:.3f} on unbalanced panel"


def test_ab_two_lag_dep_var_runs():
    """Configuration with lag_dep_var=2 should run without error."""
    sim = simulate_dynamic_panel(N=150, T=8, rho=0.5, seed=0)
    res = ab_gmm(sim["y"], sim["panel_id"], sim["time_id"], lag_dep_var=2)
    assert len(res.coefs) == 2
    # First coef is L1.y, second is L2.y
    assert "L1.y" in res.names
    assert "L2.y" in res.names


def test_ab_one_step_works():
    """One-step robust path runs and reports step=1."""
    sim = simulate_dynamic_panel(N=100, T=6, rho=0.3, seed=0)
    res = ab_gmm(
        sim["y"], sim["panel_id"], sim["time_id"], two_step=False,
    )
    assert res.step == 1
    assert res.windmeijer is False


def test_ab_windmeijer_increases_se():
    """Windmeijer correction increases the two-step SE relative to
    no-correction in finite samples."""
    sim = simulate_dynamic_panel(N=80, T=8, rho=0.5, seed=0)
    res_w = ab_gmm(sim["y"], sim["panel_id"], sim["time_id"], windmeijer=True)
    res_no = ab_gmm(sim["y"], sim["panel_id"], sim["time_id"], windmeijer=False)
    assert res_w.se[0] > res_no.se[0], (
        f"Windmeijer SE {res_w.se[0]:.4f} should exceed uncorrected "
        f"SE {res_no.se[0]:.4f}"
    )


def test_ab_user_provided_names():
    sim = simulate_dynamic_panel(N=80, T=6, rho=0.5, seed=0)
    res = ab_gmm(
        sim["y"], sim["panel_id"], sim["time_id"],
        names=["my_rho"],
    )
    assert res.names == ("my_rho",)


def test_ab_too_short_panel_raises():
    """Panel with all units having only 1 time period should raise."""
    rng = np.random.default_rng(0)
    N = 30
    y = rng.normal(size=N)
    panel_id = np.arange(N)
    time_id = np.zeros(N, dtype=int)
    with pytest.raises(ValueError):
        ab_gmm(y, panel_id, time_id)
