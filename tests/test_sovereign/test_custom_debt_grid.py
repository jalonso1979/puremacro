"""Nonuniform grids must preserve the stated Bellman problem and reentry state."""
import numpy as np
import pytest

from puremacro.sovereign import evaluate_debt_restructuring, solve_sovereign_default
from puremacro.sovereign.arellano import _make_debt_grid


GRID = np.array([-.15, -.08, -.025, 0., .000001, .0001, .001, .01, .04, .08, .15, .24, .33])
PARAMETERS = dict(n_y=7, sigma_eps=0., y_penalty_type="arellano", theta=.5,
                  enforce_monotonicity=False, tol=1e-10)


@pytest.mark.parametrize("grid", [
    np.array([]), np.array([0.]), np.array([[0., .1]]),
    np.array([0., .1, .1]), np.array([.1, 0.]),
    np.array([-.1, .1]), np.array([0., np.nan]),
    np.array([0., np.inf]), np.array([0., .1 + .01j]),
])
def test_invalid_custom_grid_is_rejected(grid):
    with pytest.raises(ValueError, match="debt_grid"):
        solve_sovereign_default(debt_grid=grid)


def test_nonuniform_grid_matches_dense_and_independent_bellman_equations():
    supplied = GRID.copy()
    fast = solve_sovereign_default(debt_grid=supplied, policy_search="monotone",
                                    n_b=1, b_min=10., b_max=-10., **PARAMETERS)
    dense = solve_sovereign_default(debt_grid=GRID, **PARAMETERS)
    supplied[:] = 999.  # The result owns its coordinates.
    np.testing.assert_array_equal(fast.b_grid, GRID)
    assert fast.converged and dense.converged
    assert fast.metadata["debt_grid_supplied"]
    assert fast.metadata["simulation_reset_debt"] == 0.
    for name in ("v_repay", "v_default", "v_expected", "bond_price", "policy_b", "default_prob"):
        np.testing.assert_array_equal(getattr(fast, name), getattr(dense, name))

    # Recompute the economic equations directly, including all debt choices.
    total = np.maximum(fast.v_repay, fast.v_default[None, :])
    for i, debt in enumerate(GRID):
        for j, income in enumerate(fast.y_grid):
            consumption = income - debt + fast.bond_price[:, j] * GRID
            values = 1. - 1. / consumption + .953 * (total @ fast.p_trans[j])
            assert fast.v_repay[i, j] == pytest.approx(values.max(), abs=2e-8, rel=0.)
    default = (fast.v_default[None, :] > fast.v_repay).astype(float)
    np.testing.assert_allclose(fast.bond_price, (1. - default) @ fast.p_trans.T / 1.01,
                               atol=2e-8, rtol=0.)
    zero = int(np.flatnonzero(GRID == 0.)[0])
    expected_default = (1. - 1. / fast.metadata["y_default_vec"]
                        + .953 * (.5 * total[zero] + .5 * fast.v_default) @ fast.p_trans.T)
    np.testing.assert_allclose(fast.v_default, expected_default, atol=2e-8, rtol=0.)


def test_explicit_default_grid_preserves_equilibrium_and_random_paths():
    grid, _ = _make_debt_grid(-.15, .33, 65)
    generated = solve_sovereign_default(n_b=65, b_min=-.15, b_max=.33,
                                         policy_search="monotone", **PARAMETERS)
    supplied = solve_sovereign_default(debt_grid=grid, policy_search="monotone", **PARAMETERS)
    assert "debt_grid_supplied" not in generated.metadata
    for name in ("v_repay", "v_default", "v_expected", "bond_price", "policy_b", "default_prob"):
        np.testing.assert_array_equal(getattr(supplied, name), getattr(generated, name))
    assert generated.simulate(T=500, seed=1979).equals(supplied.simulate(T=500, seed=1979))


def test_nonuniform_price_tangents_use_spacing_weighted_harmonic_slopes():
    result = solve_sovereign_default(debt_grid=GRID, policy_search="monotone", **PARAMETERS)
    h = np.diff(GRID)
    secant = np.diff(result.bond_price, axis=0) / h[:, None]
    expected = np.zeros_like(result.bond_price[1:-1])
    for i in range(1, len(GRID) - 1):
        for j in range(len(result.y_grid)):
            left, right = secant[i - 1, j], secant[i, j]
            if left * right > 0.:
                w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
                expected[i - 1, j] = min((w1 + w2) / (w1 / left + w2 / right), 0.)
    np.testing.assert_allclose(result.dq_db[1:-1], expected, atol=1e-12, rtol=1e-13)


def test_maturity_restructuring_preserves_custom_grid():
    original = solve_sovereign_default(debt_grid=GRID, n_y=5, sigma_eps=.02, tol=1e-7)
    changed = evaluate_debt_restructuring(original, haircut=.2, maturity_extension=.1, horizon=3)
    solved = changed.metadata["res_restructured"]
    assert solved.converged
    assert solved.metadata["debt_grid_supplied"]
    np.testing.assert_array_equal(solved.b_grid, GRID)
