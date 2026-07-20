from __future__ import annotations

import numpy as np

from puremacro.vfi.examples import life_cycle_profile


def test_shapes_and_cohort_mass():
    out = life_cycle_profile(horizon=30, n_z=5, n_a=100)
    sol = out["solution"]
    J = 30
    assert sol.V.shape == (J, 100, 5)
    assert out["life_cycle_dist"].shape == (J, 100, 5)
    # each cohort age-slice is a probability measure (mass 1)
    np.testing.assert_allclose(out["life_cycle_dist"].sum(axis=(1, 2)),
                               np.ones(J), atol=1e-10)
    assert out["assets_by_age"].shape == (J,)
    assert out["consumption_by_age"].shape == (J,)
    assert out["cross_section"].shape == (100, 5)
    np.testing.assert_allclose(out["cross_section"].sum(), 1.0, atol=1e-10)


def test_no_bequest_spends_everything_at_last_age():
    # with terminal continuation 0, the last-age agent maximizes current utility:
    # c is decreasing in a', so the optimal a' is the borrowing limit (index 0).
    out = life_cycle_profile(horizon=25, n_z=5, n_a=100)
    assert np.all(out["solution"].policy_aprime[-1] == 0)


def test_assets_hump_shaped_over_life():
    out = life_cycle_profile(horizon=40, n_z=5, n_a=120)
    assets = out["assets_by_age"]
    # newborns start at a=0 (lowest grid node)
    assert assets[0] == 0.0
    # hump: the peak is in the interior of life, above the (zero) starting level
    peak_age = int(np.argmax(assets))
    assert 0 < peak_age < 39
    assert assets[peak_age] > 0.0
    assert assets[-1] < assets[peak_age]            # run assets down toward the end


def test_consumption_positive_and_smoother_than_income():
    out = life_cycle_profile(horizon=40, n_z=5, n_a=120)
    cons = out["consumption_by_age"]
    assert np.all(cons > 0.0)
    # consumption is smoother than the (hump-shaped) deterministic income profile:
    # its peak-to-trough log range is smaller than income's.
    ages = np.arange(40)
    kappa = np.exp(0.1 * ages - 0.0025 * ages ** 2)
    cons_range = np.log(cons.max()) - np.log(cons.min())
    inc_range = np.log(kappa.max()) - np.log(kappa.min())
    assert cons_range < inc_range


def test_life_cycle_exported():
    from puremacro.vfi import life_cycle_profile as fn

    assert fn is life_cycle_profile
