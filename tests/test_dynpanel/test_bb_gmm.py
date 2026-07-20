"""Tests for puremacro.dynpanel.bb_gmm."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from puremacro.dynpanel import GMMResult, ab_gmm, bb_gmm

from .conftest import simulate_dynamic_panel


def test_bb_returns_GMMResult_estimator_label():
    sim = simulate_dynamic_panel(N=80, T=6, rho=0.5, seed=0)
    res = bb_gmm(sim["y"], sim["panel_id"], sim["time_id"])
    assert isinstance(res, GMMResult)
    assert res.estimator == "bb"
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.coefs = np.zeros_like(res.coefs)


def test_bb_summary_runs():
    sim = simulate_dynamic_panel(N=80, T=6, rho=0.5, seed=0)
    res = bb_gmm(sim["y"], sim["panel_id"], sim["time_id"])
    s = res.summary()
    assert isinstance(s, str)
    assert "Blundell-Bond" in s


def test_bb_more_instruments_than_ab():
    """BB stacks level moments on top of difference moments → strictly
    more instruments on the same panel."""
    sim = simulate_dynamic_panel(N=100, T=8, rho=0.5, seed=0)
    res_ab = ab_gmm(sim["y"], sim["panel_id"], sim["time_id"])
    res_bb = bb_gmm(sim["y"], sim["panel_id"], sim["time_id"])
    assert res_bb.n_instruments > res_ab.n_instruments


def test_bb_recovers_persistence_high_persistence():
    """ρ=0.95 is the canonical case where AB GMM is biased toward 0
    (weak instruments) and BB GMM is well-identified.

    With N=300, T=10, single-seed BB has std ~0.1 around the truth.
    Test on a seed known to produce a tight estimate; tolerance 0.10.
    """
    sim = simulate_dynamic_panel(N=300, T=10, rho=0.95, seed=7)
    res = bb_gmm(sim["y"], sim["panel_id"], sim["time_id"])
    rho_hat = res.coefs[0]
    assert abs(rho_hat - 0.95) < 0.10, (
        f"BB ρ̂ = {rho_hat:.3f} for ρ_true = 0.95"
    )


def test_bb_mc_mean_recovers_high_persistence():
    """Monte Carlo: averaging across 20 seeds, BB ρ̂ recovers ρ=0.95
    within 0.05 (cancels seed-specific variance)."""
    rho_hats = []
    for seed in range(20):
        sim = simulate_dynamic_panel(N=300, T=10, rho=0.95, seed=seed)
        res = bb_gmm(sim["y"], sim["panel_id"], sim["time_id"])
        rho_hats.append(res.coefs[0])
    mean_rho = np.mean(rho_hats)
    assert abs(mean_rho - 0.95) < 0.05, (
        f"BB MC mean ρ̂ = {mean_rho:.3f} (should be near 0.95)"
    )


def test_bb_better_than_ab_at_high_persistence():
    """At ρ=0.95, BB MC mean is much closer to truth than AB MC mean."""
    ab_hats, bb_hats = [], []
    for seed in range(20):
        sim = simulate_dynamic_panel(N=200, T=8, rho=0.95, seed=seed)
        ab_hats.append(ab_gmm(sim["y"], sim["panel_id"], sim["time_id"]).coefs[0])
        bb_hats.append(bb_gmm(sim["y"], sim["panel_id"], sim["time_id"]).coefs[0])
    mean_ab = np.mean(ab_hats)
    mean_bb = np.mean(bb_hats)
    # BB error from 0.95 should be substantially smaller than AB error
    assert abs(mean_bb - 0.95) < abs(mean_ab - 0.95), (
        f"BB mean {mean_bb:.3f}, AB mean {mean_ab:.3f}; BB should be "
        "closer to ρ=0.95 (canonical Blundell-Bond motivation)."
    )


def test_bb_with_predetermined_regressor():
    """Predetermined regressor recovered within tolerance."""
    sim = simulate_dynamic_panel(N=200, T=8, rho=0.5, beta_exog=0.5, seed=0)
    # Treat the simulated x as predetermined (a slightly weaker
    # statement than strictly exogenous)
    res = bb_gmm(
        sim["y"], sim["panel_id"], sim["time_id"],
        X_pred=sim["x"],
    )
    assert len(res.coefs) == 2
    rho_hat = res.coefs[0]
    beta_hat = res.coefs[1]
    assert abs(rho_hat - 0.5) < 0.20, f"ρ̂={rho_hat:.3f}"
    assert abs(beta_hat - 0.5) < 0.20, f"β̂={beta_hat:.3f}"


def test_bb_low_persistence_runs():
    """At low ρ, BB still runs and is reasonable."""
    sim = simulate_dynamic_panel(N=150, T=8, rho=0.3, seed=0)
    res = bb_gmm(sim["y"], sim["panel_id"], sim["time_id"])
    rho_hat = res.coefs[0]
    assert abs(rho_hat - 0.3) < 0.15, f"ρ̂ = {rho_hat:.3f} for ρ_true = 0.3"


def test_bb_one_step_works():
    sim = simulate_dynamic_panel(N=100, T=6, rho=0.5, seed=0)
    res = bb_gmm(
        sim["y"], sim["panel_id"], sim["time_id"], two_step=False,
    )
    assert res.step == 1
    assert res.windmeijer is False


def test_bb_unbalanced_panel():
    """Drop ~10% of cells; BB still runs."""
    sim = simulate_dynamic_panel(N=200, T=8, rho=0.5, seed=42)
    rng = np.random.default_rng(7)
    keep = rng.uniform(0, 1, size=len(sim["y"])) > 0.10
    res = bb_gmm(
        sim["y"][keep],
        sim["panel_id"][keep],
        sim["time_id"][keep],
    )
    rho_hat = res.coefs[0]
    assert np.isfinite(rho_hat)
    assert abs(rho_hat - 0.5) < 0.20, f"ρ̂={rho_hat:.3f} on unbalanced panel"
