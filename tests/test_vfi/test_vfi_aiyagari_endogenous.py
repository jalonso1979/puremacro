"""Aiyagari with endogenous labor: the equilibrium must actually be solved for.

Until 1.3.2 ``solve_aiyagari_endogenous`` assigned ``r_curr = r_guess``, never
touched it again, and returned it as ``r_star``. Every test here fails against
that version; the first one fails by construction.
"""
import numpy as np
import pytest

from puremacro.vfi.aiyagari_endogenous import (
    _factor_prices,
    solve_aiyagari_endogenous,
)

SMALL = dict(Na=30, Ne=3)


def test_r_star_does_not_depend_on_the_guess():
    """The bug in one line: r_star used to be whatever the caller passed in."""
    rates = [solve_aiyagari_endogenous(**SMALL, r_guess=g)["r_star"]
             for g in (0.010, 0.035, 0.050)]
    assert max(rates) - min(rates) < 1e-8
    assert not any(abs(r - g) < 1e-9
                   for r, g in zip(rates, (0.010, 0.035, 0.050)))


def test_capital_market_clears_at_r_star():
    """Household supply equals firm demand at the SAME prices, to grid error."""
    res = solve_aiyagari_endogenous(Na=100, Ne=11)
    kl, _ = _factor_prices(res["r_star"], 0.36, 0.08)
    residual = res["K_star"] - kl * res["L_star"]
    assert abs(residual) / res["K_star"] < 1e-3
    assert residual == pytest.approx(res["excess_demand"], abs=1e-12)


def test_clearing_improves_with_the_asset_grid():
    """The remaining residual is discretisation: a' is a grid index, so excess
    demand steps rather than crosses. It must shrink as the grid refines."""
    coarse = solve_aiyagari_endogenous(Na=30, Ne=5)
    fine = solve_aiyagari_endogenous(Na=150, Ne=5)
    assert (abs(fine["excess_demand"]) / fine["K_star"]
            < abs(coarse["excess_demand"]) / coarse["K_star"])


def test_wage_is_consistent_with_the_solved_rate():
    res = solve_aiyagari_endogenous(**SMALL)
    _, w = _factor_prices(res["r_star"], 0.36, 0.08)
    assert res["w_star"] == pytest.approx(w, rel=1e-12)


def test_r_star_below_the_rate_of_time_preference():
    """Aiyagari's central result: precautionary saving pushes r* under 1/beta-1."""
    beta = 0.95
    res = solve_aiyagari_endogenous(**SMALL, beta=beta)
    assert 0.0 < res["r_star"] < 1.0 / beta - 1.0


def test_more_patience_lowers_the_equilibrium_rate():
    impatient = solve_aiyagari_endogenous(**SMALL, beta=0.94)
    patient = solve_aiyagari_endogenous(**SMALL, beta=0.96)
    assert patient["r_star"] < impatient["r_star"]
    assert patient["K_star"] > impatient["K_star"]


def test_aggregate_labor_is_weighted_by_the_distribution():
    """L* is int e*n(a,e) dmu, not mean(n) * mean(e).

    The two agree only if hours and productivity are independent AND households
    are uniform over the grid; neither holds, which is what made the old
    number wrong rather than merely imprecise.
    """
    res = solve_aiyagari_endogenous(**SMALL)
    mu, n, e = res["distribution"], res["policy_n"], res["e_grid"]
    assert res["L_star"] == pytest.approx(float(np.sum(mu * n * e[None, :])), rel=1e-12)
    assert res["L_star"] != pytest.approx(float(np.mean(n) * np.mean(e)), rel=1e-3)


def test_distribution_is_a_distribution():
    res = solve_aiyagari_endogenous(**SMALL)
    mu = res["distribution"]
    assert mu.shape == (30, 3)
    assert np.all(mu >= 0.0)
    assert float(mu.sum()) == pytest.approx(1.0, abs=1e-10)


def test_a_binding_asset_grid_is_named_not_silently_returned():
    """An a_max too low to clear the market must raise, not report a number."""
    with pytest.raises(ValueError, match="does not change sign"):
        solve_aiyagari_endogenous(Na=20, Ne=3, a_max=0.05)


def test_hours_fall_with_wealth_and_rise_with_productivity():
    """The economics the notebook draws: a negative wealth effect on hours and
    a positive substitution effect from productivity.

    Stated as averages, not pointwise. ``n`` is read at the CHOSEN ``a'``,
    which is a grid index, so the policy inherits that discreteness and is not
    monotone cell by cell -- and at the bottom of the asset grid the least
    productive households are pinned at the ``n <= 1`` corner, working more
    than middling ones rather than less.
    """
    res = solve_aiyagari_endogenous(Na=60, Ne=5)
    n = res["policy_n"]
    assert n[-10:, :].mean() < n[:10, :].mean()      # wealth effect
    assert n[:, -1].mean() > n[:, 0].mean()          # substitution effect
    assert np.all((n >= 0.0) & (n <= 1.0))           # time endowment respected
