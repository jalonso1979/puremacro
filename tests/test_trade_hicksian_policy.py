"""Policy objectives, independent tariff grids, and audited solver recovery."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from puremacro.trade import (calibrate_trade_model, compute_hicksian_welfare,
    compute_unilateral_optimal_tariff, solve_multilateral_nash_tariffs,
    compute_welfare_payoff_matrix, evaluate_national_welfare,
    solve_policy_equilibrium, PolicyEquilibriumError)
from puremacro.trade.solver import build_initial_guess
from tools.reference_validation.validate_trade_welfare import consumption_benchmark
from tools.reference_validation.validate_hicksian_policy import validate, validate_oecd_policy, PolicyOracle
from tools.reference_validation.validate_trade_accounting import scalar_reference


@pytest.fixture(scope="module")
def model():
    data, tau, fd = consumption_benchmark()
    c = calibrate_trade_model(data, ns=1, nc=2, nfd=3, country_codes=["A", "B"])
    b = solve_policy_equilibrium(c, sigma=2., tol=1e-9)
    return data, c, tau, fd, b


def settings(base=None):
    return dict(metric="hicksian_ev", sigma=2., consumption_categories=(0, 2),
                ge_tol=1e-9, base_equilibrium=base)


def test_independent_tariff_grid_and_nash_reference():
    report = validate()
    assert report["passed"], report


def test_frozen_oecd_policy_world_aggregation_and_explicit_tolerance():
    report = validate_oecd_policy()
    assert report["passed"], report
    assert not report["one_iteration_candidate_converged"]
    assert not report["native_archive_reloaded_this_run"]
    for attempt in report["strict_tolerance_probe"]["attempts"]:
        assert attempt["requested_tolerance"] == 1e-8
        if attempt["accepted"]:
            assert attempt["audited_max_residual"] <= 1e-8


def test_fixed_baseline_ev_and_meaningful_percentage(model):
    _, c, _, _, b = model
    opt = compute_unilateral_optimal_tariff(c, country_idx="A", tariff_max=.2,
                                           num_grid=5, **settings(b))
    w = compute_hicksian_welfare(c, opt.equilibrium, base_result=b, consumption_categories=(0, 2))
    assert opt.baseline_welfare == 0.
    assert opt.optimal_welfare == pytest.approx(w.ev, abs=1e-9)
    assert opt.welfare_gain_pct == pytest.approx(w.ev_pct_consumption, abs=1e-9)
    assert opt.welfare_gain_pct > 0
    direct = evaluate_national_welfare(c, opt.equilibrium, metric="hicksian_ev", country_idx="A",
                                       base_equilibrium=b, consumption_categories=(0, 2))
    assert direct == opt.optimal_welfare
    assert opt.metadata["baseline_equilibrium"] is b
    assert opt.metadata["sigma"] == 2.
    with pytest.raises(ValueError, match="baseline"):
        evaluate_national_welfare(c, opt.equilibrium, metric="hicksian_ev", country_idx="A")


def test_small_tariff_is_not_silently_replaced_by_zero(model):
    _, c, _, _, b = model
    opt = compute_unilateral_optimal_tariff(c, country_idx="A", tariff_max=1e-7,
                                           num_grid=2, method="grid", **settings(b))
    assert opt.optimal_tariff_rate == 1e-7
    assert opt.welfare_curve[1] != 0
    np.testing.assert_allclose(opt.equilibrium.metadata["final_tariff_multipliers"][1, :, 0],
                               1+1e-7, rtol=0, atol=0)


def test_supplied_baseline_must_pass_requested_ge_tolerance(model):
    _, c, _, _, b = model
    options = {**settings(b), "ge_tol": 1e-18}
    with pytest.raises(ValueError, match="baseline audit"):
        compute_unilateral_optimal_tariff(c, country_idx="A", num_grid=3, **options)


@pytest.mark.parametrize("mode", ["final_only", "intermediate_only"])
def test_separate_policy_instruments_against_independent_equilibrium(model, mode):
    _, c, _, _, b = model
    opt = compute_unilateral_optimal_tariff(c, country_idx="A", tariff_max=.2,
        num_grid=5, method="grid", policy_mode=mode, **settings(b))
    oracle = PolicyOracle()
    expected = []
    for rate in opt.tariff_grid:
        ta = np.ones((2, 1, 2)); tf = np.ones((2, 3, 2))
        if mode == "final_only": tf[1, :, 0] = 1+rate
        else: ta[1, 0, 0] = 1+rate
        state = scalar_reference(oracle.data, ta, tf, sigma=2.)
        q = np.min(state["F"][:, [0, 2]]/oracle.baskets, axis=0)
        utility = np.prod(q**oracle.weights, axis=0)
        expected.append(oracle.unit_cost[0]*(utility[0]-oracle.base_utility[0]))
    np.testing.assert_allclose(opt.welfare_curve, expected, atol=1e-7, rtol=1e-8)


def test_nash_regret_world_aggregation_and_tiny_relaxation(model):
    _, c, _, _, b = model
    result = solve_multilateral_nash_tariffs(c, player_countries=["A"], tariff_max=.2,
        best_response_grid_size=5, max_iter=2, relaxation=1e-8, tol=1e-4, **settings(b))
    assert not result.converged
    assert result.outer_error > .1
    assert result.metadata["relative_max_regret"] > 1e-3
    w = [compute_hicksian_welfare(c, result.equilibrium, base_result=b, target_country=i,
                                 consumption_categories=(0, 2)) for i in range(2)]
    assert result.world_welfare_change_pct == pytest.approx(
        100*sum(r.ev for r in w)/sum(r.consumption_base for r in w), abs=1e-10)
    assert result.metadata["world_welfare_scope"].startswith("all calibrated countries")
    assert result.baseline_welfares == {"A": 0.}
    assert result.metadata["relative_max_regret"] == pytest.approx(
        result.metadata["player_regrets"]["A"]/w[0].consumption_base)


def test_currency_scaling_preserves_choices_and_normalized_regret(model):
    data, c, _, _, b = model
    reference = solve_multilateral_nash_tariffs(c, player_countries=["A", "B"], tariff_max=.2,
        best_response_grid_size=5, max_iter=1, relaxation=.5, **settings(b))
    scaled = calibrate_trade_model(data*.001, ns=1, nc=2, nfd=3, country_codes=["A", "B"])
    result = solve_multilateral_nash_tariffs(scaled, player_countries=["A", "B"], tariff_max=.2,
        best_response_grid_size=5, max_iter=1, relaxation=.5,
        metric="hicksian_ev", sigma=2., consumption_categories=(0, 2), ge_tol=1e-12)
    assert result.nash_tariffs == pytest.approx(reference.nash_tariffs, abs=1e-7)
    assert result.welfare_changes_pct == pytest.approx(reference.welfare_changes_pct, abs=1e-6)
    assert result.metadata["relative_max_regret"] == pytest.approx(reference.metadata["relative_max_regret"], abs=1e-8)
    for p in result.strategic_players:
        assert result.player_welfares[p] == pytest.approx(.001*reference.player_welfares[p], abs=1e-8)


def test_matrix_uses_fixed_actions_and_common_baseline(model):
    _, c, _, _, b = model
    result = compute_welfare_payoff_matrix(c, player_a="A", player_b="B",
                                           nash_a=.1, nash_b=.2, **settings(b))
    profiles = result.metadata["cell_profiles"]
    assert profiles["DC"]["A"] == profiles["DD"]["A"] == .1
    assert profiles["CD"]["B"] == profiles["DD"]["B"] == .2
    assert result.metadata["continuous_nash_verified"] is False
    assert "Nash" not in " ".join(result.summary_df.index)
    for i, j, key in ((0, 0, "CC"), (1, 0, "DC"), (0, 1, "CD"), (1, 1, "DD")):
        for country in range(2):
            w = compute_hicksian_welfare(c, result.scenarios[key], base_result=b,
                target_country=country, consumption_categories=(0, 2))
            assert result.payoff_matrix[i, j, country] == pytest.approx(w.ev_pct_consumption, abs=1e-9)
    with pytest.raises(ValueError, match="one tariff"):
        compute_welfare_payoff_matrix(c, player_a="A", player_b="B", optimal_a=.1,
                                      nash_a=.2, optimal_b=.1, **settings(b))


def test_real_newton_failure_recovers_without_changing_equations(model):
    _, c, ta, tf, _ = model
    result = solve_policy_equilibrium(c, ta, tf, sigma=.5, tol=1e-10)
    attempts = result.metadata["policy_solver_attempts"]
    assert result.converged
    assert attempts[-1]["accepted"]
    assert attempts[-1]["audited_max_residual"] <= 1e-10
    assert result.metadata["sigma"] == .5
    np.testing.assert_array_equal(result.metadata["intermediate_tariff_multipliers"], ta)
    np.testing.assert_array_equal(result.metadata["final_tariff_multipliers"], tf)
    # If Newton improves later, this test still permits its audited success.
    if len(attempts) > 1:
        assert not attempts[0]["accepted"]
        assert result.metadata["policy_solver_fallback_used"]


def test_fallback_resets_poisoned_iterates_and_reaches_continuation(monkeypatch, model):
    from puremacro.trade import policy_solver
    _, c, ta, tf, _ = model
    base = solve_policy_equilibrium(c, sigma=.5, tol=1e-10)
    actual = policy_solver.solve_trade_equilibrium
    calls = []
    def fake(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            return replace(base, converged=False, x_sol=np.full_like(base.x_sol, np.nan))
        return actual(*args, **kwargs)
    monkeypatch.setattr(policy_solver, "solve_trade_equilibrium", fake)
    warm = base.x_sol.copy(); warm[0] += .001
    result = solve_policy_equilibrium(c, ta, tf, sigma=.5, tol=1e-10, x0=warm)
    assert result.metadata["policy_solver_method"] == "keller_pac"
    assert [a["method"] for a in calls] == ["newton", "hybr", "keller_pac"]
    np.testing.assert_array_equal(calls[0]["x0"], warm)
    for call in calls[1:]:
        np.testing.assert_array_equal(call["x0"], build_initial_guess(c))
    for call in calls:
        assert call["sigma"] == .5 and call["tol"] == 1e-10
        np.testing.assert_array_equal(call["tau"], ta)
        np.testing.assert_array_equal(call["tau_fd"], tf)


@pytest.mark.parametrize("corruption", ["failed", "price", "schedule", "physical"])
def test_false_success_exhausts_attempts_with_no_payoff(monkeypatch, model, corruption):
    from puremacro.trade import policy_solver
    _, c, _, _, base = model
    if corruption == "failed": bad = replace(base, converged=False)
    elif corruption == "price": bad = replace(base, p_sol=base.p_sol*1.1)
    elif corruption == "schedule":
        meta = dict(base.metadata); meta["final_tariff_multipliers"] = np.ones_like(meta["final_tariff_multipliers"])*1.1
        bad = replace(base, metadata=meta)
    else:
        x = base.x_sol.copy(); x[2] += .01
        bad = replace(base, x_sol=x, converged=True, max_residual=0.)
    monkeypatch.setattr(policy_solver, "solve_trade_equilibrium", lambda *a, **k: bad)
    with pytest.raises(PolicyEquilibriumError) as failure:
        solve_policy_equilibrium(c, sigma=2.)
    assert len(failure.value.attempts) == 3
    assert not any(a["accepted"] for a in failure.value.attempts)


def test_invalid_inputs_do_not_trigger_fallback(monkeypatch, model):
    from puremacro.trade import policy_solver
    _, c, ta, tf, _ = model
    def forbidden(*a, **k): raise AssertionError("solver must not be called")
    monkeypatch.setattr(policy_solver, "solve_trade_equilibrium", forbidden)
    wrong = ta.copy(); wrong[0, 0, 0] = 1.1
    with pytest.raises(ValueError, match="domestic"):
        solve_policy_equilibrium(c, wrong, tf)
    with pytest.raises(ValueError, match="method"):
        solve_policy_equilibrium(c, method="unknown")


@pytest.mark.parametrize("search", ["unilateral", "nash", "matrix"])
def test_unresolved_policy_point_aborts_instead_of_becoming_bad_payoff(monkeypatch, model, search):
    from puremacro.trade import _hicksian_policy
    _, c, _, _, b = model
    def failed(*a, **k):
        raise PolicyEquilibriumError("all attempts failed", [{"accepted": False}])
    monkeypatch.setattr(_hicksian_policy, "solve_policy_equilibrium", failed)
    with pytest.raises(PolicyEquilibriumError) as error:
        if search == "unilateral":
            compute_unilateral_optimal_tariff(c, country_idx="A", tariff_max=.2, num_grid=3, **settings(b))
        elif search == "nash":
            solve_multilateral_nash_tariffs(c, player_countries=["A", "B"], tariff_max=.2,
                                           best_response_grid_size=3, max_iter=1, **settings(b))
        else:
            compute_welfare_payoff_matrix(c, player_a="A", player_b="B", optimal_a=.1, optimal_b=.1, **settings(b))
    assert any(v > 0 for v in error.value.profile.values())


def test_failed_refinement_is_not_reported_as_optimal(monkeypatch, model):
    from puremacro.trade import _hicksian_policy
    _, c, _, _, b = model
    monkeypatch.setattr(_hicksian_policy, "minimize_scalar", lambda *a, **k: SimpleNamespace(success=False))
    with pytest.raises(RuntimeError, match="refinement"):
        compute_unilateral_optimal_tariff(c, country_idx="A", tariff_max=.2, num_grid=3, **settings(b))


@pytest.mark.parametrize("extra", [{"accounting": "legacy"}, {"consumption_categories": (1,)},
    {"policy_mode": "sectoral"}, {"ge_tol": np.nan}, {"tariff_max": np.inf},
    {"num_grid": 2.5}, {"sigma": -1}, {"consumption_categoriess": (0,)}])
def test_invalid_or_misspelled_policy_options_fail(model, extra):
    _, c, _, _, b = model
    kw = settings(b); kw.update(extra)
    with pytest.raises((ValueError, TypeError)):
        compute_unilateral_optimal_tariff(c, country_idx="A", **kw)


def test_overlapping_players_and_changed_reference_are_rejected(model):
    _, c, ta, tf, b = model
    with pytest.raises(ValueError, match="non-overlapping"):
        solve_multilateral_nash_tariffs(c, player_countries=["A", 0], **settings(b))
    changed = solve_policy_equilibrium(c, ta, tf, sigma=2.)
    with pytest.raises(ValueError, match="zero-tariff"):
        compute_unilateral_optimal_tariff(c, country_idx="A", **settings(changed))


def test_bloc_does_not_report_first_members_terms_of_trade(model):
    _, c, _, _, _ = model
    bloc = replace(c, country_codes=("DEU", "FRA"))
    opt = compute_unilateral_optimal_tariff(bloc, country_idx="EUR", tariff_max=.1,
                                          num_grid=3, **settings())
    # All trade in this two-member example is internal to the customs union.
    assert opt.optimal_welfare == 0.
    assert opt.optimal_tariff_rate == 0.
    assert np.isnan(opt.terms_of_trade_initial) and np.isnan(opt.terms_of_trade_optimal)
    assert "unavailable" in opt.metadata["terms_of_trade_scope"]
    with pytest.raises(ValueError, match="non-overlapping"):
        solve_multilateral_nash_tariffs(bloc, player_countries=["EUR", "DEU"], **settings())
