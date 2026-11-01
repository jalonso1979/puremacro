"""Independent differences, dense equilibria and accounting for the Krylov backend."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from puremacro.models import (HANKEquilibriumWorkspace, calibrate_one_asset_hank,
                              monetary_scenario_tree, solve_hank_equilibrium,
                              solve_hank_scenario_tree, solve_one_asset_hank)
from puremacro.models._hank_tangent import (_forward_jvp, _forward_tape,
                                          _household_jvp, _household_tape)
from puremacro.models.one_asset_hank import _backward, _forward


@pytest.fixture(scope="module")
def steady():
    return calibrate_one_asset_hank(n_a=80, n_e=3, a_max=150.)


def branching_tree():
    return pd.DataFrame(dict(
        parent=[None, "root", "root", "low", "low", "high", "high"],
        conditional_probability=[1., .4, .6, .25, .75, .8, .2],
        monetary_shock=[.0003, -.004, .004, -.003, .001, -.001, .003],
    ), index=["root", "low", "high", "ll", "lh", "hl", "hh"])


@pytest.mark.parametrize("eis,frisch", [(.5, .5), (.7, .4)])
def test_household_tangent_including_binding_labor_budget(steady, eis, frisch):
    ss = steady
    p = dict(ss.calibration, eis=eis, frisch=frisch)
    a, e, P = ss.asset_grid, ss.e_grid, ss.transition
    r, w, transfer = p["r"] + .001, p["w"] * 1.01, ss.aggregates["Div"] - ss.aggregates["Tax"]
    expected = .7 * ss.marginal_value * (1 + np.random.default_rng(3).normal(0, 1e-4, ss.marginal_value.shape))
    direction = .1 * expected
    prices_dot = np.array([.02, -.03, .04])
    base = _backward(expected, a, e, P, p, r, w, transfer)
    tape = _household_tape(expected, a, e, P, p, r, w, transfer)
    assert tape.constrained.any() and (~tape.constrained).any()
    tangent = _household_jvp(tape, direction, a, e, P, p, r, w, base[2], base[3], *prices_dot)
    h = 1e-6
    plus = _backward(expected + h * direction, a, e, P, p, *(np.array([r, w, transfer]) + h * prices_dot))
    minus = _backward(expected - h * direction, a, e, P, p, *(np.array([r, w, transfer]) - h * prices_dot))
    for actual, hi, lo in zip(tangent, plus, minus):
        np.testing.assert_allclose(actual, (hi - lo) / (2 * h), atol=5e-8, rtol=2e-7)
    _, da, dc, dn = tangent
    np.testing.assert_array_equal(da[tape.constrained], 0.)
    np.testing.assert_allclose(dc + da, prices_dot[0] * a[:, None] + prices_dot[1] * e * base[3]
                               + w * e * dn + prices_dot[2] * e, atol=2e-13)
    foc_dot = dn / base[3] - frisch * (prices_dot[1] / w - dc / (eis * base[2]))
    np.testing.assert_allclose(foc_dot[tape.constrained], 0., atol=2e-13)


def test_transport_tangent_conserves_mass_and_asset_first_moment(steady):
    a, P = steady.asset_grid, steady.transition
    rng = np.random.default_rng(9)
    d = rng.uniform(size=steady.distribution.shape)
    d /= d.sum()
    dd = rng.normal(size=d.shape)
    dd -= dd.mean()
    dd *= .001
    # Interior lotteries avoid kinks, with both moving policies and moving mass.
    policy = np.broadcast_to((.7 * a[:-1] + .3 * a[1:])[:, None], d[:-1].shape)
    policy = np.vstack([policy, policy[-1]])
    da = rng.normal(size=d.shape) * .03
    actual = _forward_jvp(d, dd, da, _forward_tape(policy, a), P)
    h = 1e-6
    expected = (_forward(d + h * dd, policy + h * da, a, P)
                - _forward(d - h * dd, policy - h * da, a, P)) / (2 * h)
    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-7)
    assert abs(actual.sum()) < 1e-14
    assert np.sum(actual * a[:, None]) == pytest.approx(np.sum(dd * policy + d * da), abs=2e-13)


@pytest.mark.pyodide_smoke
@pytest.mark.parametrize("tree", [branching_tree(), monetary_scenario_tree(horizon=12)])
def test_full_equilibrium_jvp_matches_directional_differences(steady, tree):
    workspace = HANKEquilibriumWorkspace(steady, tree)
    x = np.random.default_rng(45).normal(0, .0001, workspace.size)
    direction = np.random.default_rng(46).normal(size=workspace.size)
    analytic = workspace.jacobian_vector_product(x, direction)
    h = 1e-7
    numeric = (workspace.residual(x + h * direction) - workspace.residual(x - h * direction)) / (2 * h)
    np.testing.assert_allclose(analytic, numeric, atol=5e-8, rtol=3e-7)
    # A JVP must be a linear map at a fixed active branch, including the zero direction.
    np.testing.assert_allclose(workspace.jacobian_vector_product(x, -2 * direction), -2 * analytic, atol=1e-12)
    np.testing.assert_array_equal(workspace.jacobian_vector_product(x, np.zeros(workspace.size)), 0.)


@pytest.mark.parametrize("magnitude", [-.015, .015])
def test_chain_matches_published_model_nonlinear_solver(steady, magnitude):
    tree = monetary_scenario_tree(horizon=10, innovations=(magnitude,), probabilities=(1.,), branch_date=0)
    result = solve_hank_equilibrium(steady, tree)
    reference = solve_one_asset_hank(steady, horizon=10, magnitude=magnitude, nonlinear=True)
    assert result.converged and reference.converged
    np.testing.assert_allclose(result.paths, reference.paths[result.paths.columns], atol=8e-8, rtol=0)


@pytest.mark.pyodide_smoke
@pytest.mark.parametrize("tree", [branching_tree(), monetary_scenario_tree(
    horizon=12, innovations=(-.01, .005), probabilities=(.3, .7))])
def test_full_nonlinear_solution_matches_dense_reference(steady, tree):
    result = solve_hank_equilibrium(steady, tree)
    reference = solve_hank_scenario_tree(steady, tree)
    assert result.converged and reference.converged
    np.testing.assert_allclose(result.paths, reference.paths, atol=5e-8, rtol=0)
    for name in ("policy_a", "policy_c", "policy_n", "distributions"):
        np.testing.assert_allclose(getattr(result, name), getattr(reference, name), atol=5e-7, rtol=0)
    assert result.diagnostics["evaluations"] < reference.diagnostics["evaluations"] / 3
    assert not result.diagnostics["dense_jacobian_assembled"]
    assert result.diagnostics["preconditioner_factor_nnz"] < 25 * len(tree)
    assert not result.metadata["general_stationary_aggregate_risk_solver"]
    assert set(["pi", "w", "Y", "C", "A", "NE", "r", "Div", "Tax", "goods_mkt", "asset_mkt", "labor_mkt", "nkpc_res"]) == set(result.paths)
    np.testing.assert_allclose(result.to_frame().probability, result.nodes.probability)
    # Reconstruct the budget from reported prices, rather than accepting the diagnostic alone.
    p, a, e = steady.calibration, steady.asset_grid, steady.e_grid
    for i, row in enumerate(result.paths.itertuples()):
        budget = ((1 + p["r"] + row.r) * a[:, None] + (p["w"] + row.w) * e * result.policy_n[i]
                  + (steady.aggregates["Div"] + row.Div - steady.aggregates["Tax"] - row.Tax) * e)
        np.testing.assert_allclose(result.policy_c[i] + result.policy_a[i], budget, atol=1e-10)
    np.testing.assert_allclose(result.distributions.sum(axis=(1, 2)), 1, atol=1e-13)


def test_preconditioner_preserves_solution_and_reduces_products(steady):
    workspace = HANKEquilibriumWorkspace(steady, monetary_scenario_tree(horizon=12))
    pre = workspace.solve()
    plain = workspace.solve(preconditioner="none")
    assert pre.converged and plain.converged
    np.testing.assert_allclose(pre.paths, plain.paths, atol=5e-8, rtol=0)
    assert pre.diagnostics["jacobian_vector_products"] < plain.diagnostics["jacobian_vector_products"] / 2


def test_reuse_warm_start_and_owned_inputs(steady):
    ss, tree = deepcopy(steady), monetary_scenario_tree(horizon=10)
    workspace = HANKEquilibriumWorkspace(ss, tree)
    first = workspace.solve()
    shocks = tree.monetary_shock * 1.02
    ss.calibration["phi"] = 100
    ss.marginal_value[:] = -1
    tree.loc[:, "monetary_shock"] = 100
    nodes_copy = workspace.nodes
    nodes_copy.loc[:, "monetary_shock"] = 200
    warm = workspace.solve(monetary_shocks=shocks, initial=first)
    cold = workspace.solve(monetary_shocks=shocks)
    assert first.converged and warm.converged and cold.converged
    assert warm.diagnostics["preconditioner_reused"]
    assert warm.diagnostics["newton_iterations"] < cold.diagnostics["newton_iterations"]
    np.testing.assert_allclose(warm.paths, cold.paths, atol=5e-8, rtol=0)
    np.testing.assert_array_equal(warm.nodes.monetary_shock, shocks)
    # Result mutations likewise do not change the workspace's terminal data.
    first.steady_state.calibration["phi"] = 200
    again = workspace.solve()
    np.testing.assert_array_equal(first.paths, again.paths)


def test_zero_shock_and_history_count_above_dense_cap(steady):
    tree = monetary_scenario_tree(horizon=140)
    workspace = HANKEquilibriumWorkspace(steady, tree)
    assert len(tree) > 256
    zero = workspace.solve(monetary_shocks=np.zeros(len(tree)))
    assert zero.converged and zero.diagnostics["newton_iterations"] == 0
    result = workspace.solve()
    assert result.converged
    assert result.diagnostics["market_residual_max_abs"] < 1e-8
    assert result.diagnostics["preconditioner_factor_nnz"] < 25 * len(tree)


@pytest.mark.parametrize("options,reason", [
    ({"max_iterations": 1}, "maximum_iterations"),
    ({"preconditioner": "none", "krylov_restart": 1, "krylov_max_cycles": 1}, "linear_solve_failed"),
])
def test_failure_retains_iterate_and_explicit_reason(steady, options, reason):
    workspace = HANKEquilibriumWorkspace(steady, monetary_scenario_tree(horizon=12))
    result = workspace.solve(**options)
    assert not result.converged
    assert result.diagnostics["optimizer_message"] == reason
    assert result.diagnostics["market_residual_max_abs"] > result.diagnostics["tolerance"]
    assert np.isfinite(result.paths.to_numpy()).all()
    if reason == "linear_solve_failed":
        h = result.diagnostics["history"][-1]
        assert h["step_length"] == 0 and h["linear_residual_norm"] > 0


@pytest.mark.parametrize("mutation", [
    lambda t: t.assign(conditional_probability=.7), lambda t: t.assign(monetary_shock=np.nan),
    lambda t: t.rename(index={t.index[1]: t.index[0]}), lambda t: t.iloc[:-1],
    lambda t: t.assign(parent=None), lambda t: t.assign(conditional_probability=0.),
    lambda t: t.rename(index={t.index[1]: 123}),
])
def test_invalid_tree_refused(steady, mutation):
    with pytest.raises(ValueError):
        HANKEquilibriumWorkspace(steady, mutation(monetary_scenario_tree(horizon=4)))


@pytest.mark.parametrize("options", [{"tolerance": 0}, {"tolerance": np.nan}, {"max_iterations": True},
    {"max_iterations": 0}, {"krylov_restart": 0}, {"krylov_max_cycles": 1.5}, {"preconditioner": "diagonal"},
    {"initial": [np.nan]}, {"monetary_shocks": [np.nan]}, {"initial": np.ones(21, dtype=complex)}])
def test_bad_solver_options_refused(steady, options):
    workspace = HANKEquilibriumWorkspace(steady, branching_tree())
    with pytest.raises(ValueError):
        workspace.solve(**options)


def test_memory_and_workspace_identity_guards(steady):
    tree = monetary_scenario_tree(horizon=4)
    with pytest.raises(ValueError, match="max_workspace_bytes"):
        HANKEquilibriumWorkspace(steady, tree, max_workspace_bytes=1)
    with pytest.raises(ValueError, match="converged"):
        HANKEquilibriumWorkspace(replace(steady, converged=False), tree)
    with pytest.raises(ValueError, match="real"):
        HANKEquilibriumWorkspace(replace(steady, asset_grid=steady.asset_grid.astype(complex)), tree)
    workspace = HANKEquilibriumWorkspace(steady, tree)
    with pytest.raises(ValueError, match="row order"):
        workspace.solve(monetary_shocks=tree.monetary_shock.iloc[::-1])
    first = workspace.solve()
    other = HANKEquilibriumWorkspace(steady, tree.assign(conditional_probability=[1., .4, .6, 1., 1., 1., 1.]))
    with pytest.raises(ValueError, match="calibration, grid and topology"):
        other.solve(initial=first)
    with pytest.raises(ValueError, match="positive-price"):
        workspace.solve(monetary_shocks=np.full(len(tree), -2.))
