"""Independent two-country tariff-grid check for Hicksian policy searches.

The oracle uses a scalar relative-price equation and origin-level expenditure
minimization. It never calls puremacro's GE residual, tariff builder or policy
optimizer. Runtime results are compared with the entire independent grid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.reference_validation.validate_trade_accounting import scalar_reference
from tools.reference_validation.validate_trade_welfare import consumption_benchmark, primal_expenditure


class PolicyOracle:
    def __init__(self, sigma=2.):
        self.data, _, _ = consumption_benchmark()
        self.sigma = sigma
        F = self.data[:2, 2:].reshape(2, 2, 3).transpose(0, 2, 1)
        tax = self.data[2, 2:].reshape(2, 3).T
        total = F.sum(0)+tax
        self.weights = total[[0, 2]]/total[[0, 2]].sum(0)[None]
        self.baskets = F[:, [0, 2]]/F[:, [0, 2]].sum(0)[None]
        self.unit_cost = []
        for c in range(2):
            prices = np.ones((2, 2))/(1-tax[[0, 2], c]/total[[0, 2], c])[None]
            self.unit_cost.append(primal_expenditure(prices, self.baskets[:, :, c], self.weights[:, c], 1.))
        self.base_utility = self.utility((0., 0.))
        self.base_expenditure = self.base_utility*np.array(self.unit_cost)

    def state(self, rates):
        # Construct the complete bilateral schedules independently.
        ta = np.ones((2, 1, 2)); tf = np.ones((2, 3, 2))
        ta[1, 0, 0] = 1+rates[0]; ta[0, 0, 1] = 1+rates[1]
        tf[1, :, 0] = 1+rates[0]; tf[0, :, 1] = 1+rates[1]
        return scalar_reference(self.data, ta, tf, sigma=self.sigma)

    def utility(self, rates):
        state = self.state(rates)
        q = np.min(state["F"][:, [0, 2]]/self.baskets, axis=0)
        return np.prod(q**self.weights, axis=0)

    def payoff(self, rates):
        return np.array(self.unit_cost)*(self.utility(rates)-self.base_utility)


def validate_oecd_policy():
    """Multi-sector integration check; no independent empirical policy oracle."""
    from puremacro.trade import (compute_unilateral_optimal_tariff,
        solve_multilateral_nash_tariffs, solve_policy_equilibrium, PolicyEquilibriumError)
    from puremacro.trade.data import package_mrio_to_calibration_result
    from puremacro.trade._oecd_icio import condense_final_demand
    from tools.reference_validation.validate_oecd import load_fixture
    calib = package_mrio_to_calibration_result(condense_final_demand(load_fixture()))
    # Record a strict-tolerance probe separately. It must never relax its own
    # tolerance; the subsequent experiment explicitly declares a different one.
    try:
        strict = solve_policy_equilibrium(calib, sigma=2., tol=1e-8)
        probe = {"accepted": True, "attempts": strict.metadata["policy_solver_attempts"]}
    except PolicyEquilibriumError as exc:
        probe = {"accepted": False, "attempts": exc.attempts}
    options = dict(metric="hicksian_ev", sigma=2., ge_tol=1e-5)
    opt = compute_unilateral_optimal_tariff(calib, country_idx="USA",
        tariff_max=.1, num_grid=5, method="grid", **options)
    base = opt.metadata["baseline_equilibrium"]
    options["base_equilibrium"] = base
    candidate = solve_multilateral_nash_tariffs(calib, player_countries=["USA", "CHN"],
        tariff_max=.1, best_response_grid_size=5, max_iter=1, **options)
    # With one selected fixed-composition basket, directly value the change
    # in its physical quantity at the baseline purchaser price.
    def direct_ev(eq):
        return base.Pfd_final[0, 0, :]*(eq.c_fd[0, 0, :]-base.c_fd[0, 0, :])
    direct = direct_ev(candidate.equilibrium)
    world_pct = float(100*direct.sum()/candidate.metadata["baseline_consumption_by_country"].sum())
    unilateral_error = abs(opt.optimal_welfare-float(direct_ev(opt.equilibrium)[0]))
    world_error = abs(candidate.world_welfare_change_pct-world_pct)
    return {"native_archive_reloaded_this_run": False, "unit": calib.metadata.get("unit"),
        "dimensions": [calib.nc, calib.ns], "consumption_categories": [0],
        "consumption_mapping": "HFCE + NPISH + GGFC (includes government consumption)",
        "ge_tol": 1e-5, "strict_tolerance_probe": probe,
        "unilateral_grid_rate": opt.optimal_tariff_rate, "unilateral_boundary": opt.metadata["boundary"],
        "unilateral_direct_ev_error": unilateral_error,
        "world_scope": candidate.metadata["world_welfare_scope"], "world_direct_percent_ev_error": world_error,
        "one_iteration_candidate_converged": candidate.converged,
        "candidate_max_residual": candidate.equilibrium.max_residual,
        "passed": bool(unilateral_error < 1e-7 and world_error < 1e-10
                       and candidate.equilibrium.max_residual <= options["ge_tol"])}


def validate():
    from puremacro.trade import (calibrate_trade_model, compute_unilateral_optimal_tariff,
        solve_multilateral_nash_tariffs, compute_welfare_payoff_matrix, solve_policy_equilibrium)
    oracle = PolicyOracle()
    calib = calibrate_trade_model(oracle.data, ns=1, nc=2, nfd=3, country_codes=["A", "B"])
    options = dict(metric="hicksian_ev", sigma=2., consumption_categories=(0, 2), ge_tol=1e-9)
    grid = np.linspace(0., .4, 41)
    surface = np.array([[oracle.payoff((a, b)) for b in grid] for a in grid])
    report = {"oracle": "scalar CES GE + primal baseline expenditure; 41 x 41 complete tariff grid",
              "sigma": 2., "tariff_bounds": [0., .4], "consumption_categories": [0, 2], "unilateral": {}}
    for i, player in enumerate(("A", "B")):
        opt = compute_unilateral_optimal_tariff(calib, country_idx=player,
                tariff_max=.4, num_grid=21, tol=1e-6, **options)
        rates = [0., 0.]; rates[i] = opt.optimal_tariff_rate
        direct = oracle.payoff(rates)[i]
        curve = surface[:, 0, i] if i == 0 else surface[0, :, i]
        independent_rate = grid[np.argmax(curve)]
        error = abs(opt.optimal_welfare-direct)
        report["unilateral"][player] = {"runtime_rate": opt.optimal_tariff_rate,
            "independent_grid_rate": float(independent_rate), "ev_error": float(error),
            "largest_grid_improvement_ev": float(max(0., curve.max()-direct)),
            "percent_ev": opt.welfare_gain_pct, "boundary": opt.metadata["boundary"],
            "passed": bool(error < 1e-7 and abs(opt.optimal_tariff_rate-independent_rate) < .011
                           and curve.max()-direct < 1e-7)}
    nash = solve_multilateral_nash_tariffs(calib, player_countries=("A", "B"),
            tariff_max=.4, best_response_grid_size=9, tol=1e-5, regret_tol=1e-6,
            relaxation=.8, max_iter=30, **options)
    rates = tuple(nash.nash_tariffs[p] for p in ("A", "B"))
    direct = oracle.payoff(rates)
    regrets = []
    for i in range(2):
        deviations = [oracle.payoff((t, rates[1]) if i == 0 else (rates[0], t))[i] for t in grid]
        regrets.append(max(0., max(deviations)-direct[i]))
    br_a = np.argmax(surface[:, :, 0], axis=0)
    br_b = np.argmax(surface[:, :, 1], axis=1)
    discrete_nash = [(float(grid[a]), float(grid[b])) for a in range(len(grid))
                     for b in range(len(grid)) if br_a[b] == a and br_b[a] == b]
    errors = [abs(nash.player_welfares[p]-direct[i]) for i, p in enumerate(("A", "B"))]
    relative_regret = np.array(regrets)/oracle.base_expenditure
    report["nash"] = {"converged": bool(nash.converged), "runtime_rates": nash.nash_tariffs,
        "independent_discrete_nash": discrete_nash, "ev_errors": errors,
        "independent_grid_regrets_ev": regrets, "independent_relative_regrets": relative_regret.tolist(),
        "reported_relative_regret": nash.metadata["relative_max_regret"],
        "reported_policy_gap": nash.outer_error,
        "passed": bool(nash.converged and max(errors) < 1e-7 and relative_regret.max() <= 1e-6)}
    matrix = compute_welfare_payoff_matrix(calib, player_a="A", player_b="B",
                optimal_a=.1, optimal_b=.2, **options)
    expected = np.array([[100*oracle.payoff((a, b))/oracle.base_expenditure for b in (0., .2)] for a in (0., .1)])
    matrix_error = float(np.max(np.abs(matrix.payoff_matrix-expected)))
    report["fixed_action_matrix"] = {"max_percent_ev_error": matrix_error, "passed": matrix_error < 1e-7}
    # Real previously failing Newton case: preserve requested tariffs/sigma/tol.
    _, ta, tf = consumption_benchmark()
    recovered = solve_policy_equilibrium(calib, ta, tf, sigma=.5, tol=1e-10)
    reference = scalar_reference(oracle.data, ta, tf, sigma=.5)
    error = float(np.max(np.abs(recovered.p_sol.ravel()-reference["p"])))
    report["fallback"] = {"attempts": recovered.metadata["policy_solver_attempts"],
        "price_error": error, "passed": bool(recovered.converged and error < 1e-9)}
    report["passed"] = bool(all(v["passed"] for v in report["unilateral"].values())
                            and all(report[k]["passed"] for k in ("nash", "fixed_action_matrix", "fallback")))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate()
    result["oecd_aggregation"] = validate_oecd_policy()
    result["passed"] = result["passed"] and result["oecd_aggregation"]["passed"]
    content = json.dumps(result, indent=2)+"\n"
    if args.output:
        args.output.write_text(content)
    print(content)
    raise SystemExit(0 if result["passed"] else 1)
