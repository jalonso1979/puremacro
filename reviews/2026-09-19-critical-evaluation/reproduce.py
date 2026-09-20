"""Small review probes; prints observations without changing library code.

Run from the repository root:
    MPLCONFIGDIR=/tmp/puremacro-review-mpl .venv/bin/python \
        reviews/2026-09-19-critical-evaluation/reproduce.py

These are deliberately separate from the project's regression suite. Expected
values come from closed forms or from explicit public API invariants. They are
not a claim to have externally validated every model in puremacro.
"""
from __future__ import annotations

import json
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    from puremacro.dsge import load_mod
    from puremacro.dsge.parity import verify_dynare_parity
    from puremacro.dsge.pruning import canonical_growth_2nd_order
    from puremacro.trade import solve_trade_equilibrium
    from puremacro.trade.caliendo_parro import CaliendoParroModel
    from puremacro.trade.data import (
        generate_synthetic_mrio,
        load_figaro,
        package_mrio_to_calibration_result,
    )
    from puremacro.trade.equilibrium import _get_ces_weights
    from puremacro.trade.policy_analytics import (
        decompose_hicksian_ev_3way,
        verify_theorems_1_to_4,
    )
    from puremacro.trade.regularize import compute_spectral_radius
    from puremacro.vfi import CollocationProblem
    from puremacro.vfi.deep_macro import (
        DeepMacroMLP,
        DeepMacroModel,
        solve_deep_macro,
    )

    observations = {}

    # R=1 has no root. MINPACK's least-squares success is not root convergence.
    problem = CollocationProblem(
        domain=(1.0, 2.0), orders=3, method="euler",
        euler_residual_fn=lambda policy, s, params: np.ones_like(s),
        options={"tol": 1e-10},
    )
    sol = problem.solve()
    observations["vfi_false_convergence"] = {
        "converged": bool(sol.converged), "residual": sol.residual_norm,
        "expected_converged": False,
    }

    # Analytic deterministic log-utility growth steady-state value.
    alpha, beta, productivity = 0.36, 0.96, 2.0
    k_ss = (alpha * beta * productivity) ** (1.0 / (1.0 - alpha))
    c_ss = productivity * k_ss**alpha - k_ss
    sol = CollocationProblem(
        domain=(0.5 * k_ss, 1.5 * k_ss), orders=9, beta=beta,
        params={"alpha": alpha, "delta": 1.0, "sigma": 1.0, "z": productivity},
    ).solve()
    observations["vfi_value_productivity"] = {
        "reported": float(sol.value(k_ss)),
        "analytic": float(np.log(c_ss) / (1.0 - beta)),
    }

    # The exact solution is y=A*x^3+B*x, B includes innovation variance.
    risk_rows = []
    for rho in (0.5, 0.8):
        beta, sd = 0.95, 0.1
        src = f"""
        var x y; varexo e; parameters rho beta;
        rho={rho}; beta={beta};
        model; x=rho*x(-1)+e; y=beta*y(+1)+x^3; end;
        initval; x=0; y=0; end;
        shocks; var e; stderr {sd}; end;
        """
        sol = load_mod(src, order=3)
        a = 1.0 / (1.0 - beta * rho**3)
        b = 3.0 * beta * a * rho * sd**2 / (1.0 - beta * rho)
        risk_rows.append({
            "rho": rho,
            "ghxss_reported": float(sol.ghxss[1, 0]),
            "ghxss_analytic": 2.0 * b * rho,
            "ghuss_reported": float(sol.ghuss[1, 0]),
            "ghuss_analytic": 2.0 * b,
            "max_cubic_error": float(np.max(np.abs(
                np.array([sol.ghxxx[1, 0], sol.ghxxu[1, 0],
                          sol.ghxuu[1, 0], sol.ghuuu[1, 0]])
                - 6.0 * a * np.array([rho**3, rho**2, rho, 1.0])
            ))),
        })
    observations["dsge_third_order_risk"] = risk_rows

    dr = canonical_growth_2nd_order().decision_rules()
    altered = replace(dr, ys=dr.ys + 1000.0, ghxu=dr.ghxu + 1000.0,
                      ghuu=dr.ghuu + 1000.0)
    parity = verify_dynare_parity(dr, altered, order=2)
    observations["dynare_false_parity"] = {
        "passed": bool(parity.passed), "score": parity.score,
        "expected_passed": False,
    }

    raw = generate_synthetic_mrio("wiod", custom_c=2, custom_s=2)
    calib = package_mrio_to_calibration_result(raw)
    country = calib.country_codes[0]
    base = solve_trade_equilibrium(calib, tol=1e-9, max_iter=100)
    tariff = solve_trade_equilibrium(
        calib, tau_fd=np.array([0.2, 0.0]), tol=1e-9, max_iter=100,
    )
    pac = solve_trade_equilibrium(
        calib, tau_fd=np.array([0.2, 0.0]), method="keller_pac",
        tol=1e-9, max_steps=2,
    )
    observations["pac_ignores_final_demand_tariff"] = {
        "all_converged": bool(base.converged and tariff.converged and pac.converged),
        "newton_tariff_revenue": tariff.tariffs.tolist(),
        "pac_tariff_revenue": pac.tariffs.tolist(),
        "pac_max_difference_from_zero_tariff_baseline": float(np.max(np.abs(pac.x_sol - base.x_sol))),
    }

    ev = decompose_hicksian_ev_3way(calib, tariff, base_result=tariff,
                                  target_country=country)
    observations["ev_identical_equilibria"] = {
        "reported_ev": ev.ev_usd, "reported_identity_residual": ev.residual,
        "expected_ev": 0.0,
    }
    theorem = verify_theorems_1_to_4(calib, tau_override=np.array([0.0]),
                                    target_country=country)
    observations["theorem_zero_tariff"] = {
        "passed": bool(theorem.all_passed),
        "reported_loe_welfare_gain": theorem.details[1]["loe_net_welfare"],
        "expected_gain": 0.0,
    }

    tau = np.ones((4, 2, 2))
    tau[2:, :, 0] = 1.2
    ces = solve_trade_equilibrium(calib, tau=tau, sigma=2.0,
                                  tol=1e-9, max_iter=100)
    _, omega = _get_ces_weights(calib)
    purchaser_prices = ces.p_sol.flatten(order="F")[:, None, None] * tau
    price_index = np.sum(omega / purchaser_prices, axis=0)**(-1)
    ces_quantities = calib.a * ces.y_sol * (price_index[None, :, :] / purchaser_prices)**2
    expected_flows = ces_quantities.reshape((2, 2, 2, 2), order="F")
    leontief_flows = (calib.a * ces.y_sol).reshape((2, 2, 2, 2), order="F")
    observations["ces_postprocessing"] = {
        "converged": bool(ces.converged), "solver_residual": ces.max_residual,
        "flow_error_against_ces_demand": float(np.max(np.abs(ces.intermediate_flows - expected_flows))),
        "flow_error_against_leontief_demand": float(np.max(np.abs(ces.intermediate_flows - leontief_flows))),
    }

    cp = CaliendoParroModel(
        trade_shares=np.array([[[0.8, 0.2], [0.2, 0.8]]]),
        gamma_va=np.full((2, 1), 0.5), gamma_io=np.full((2, 1, 1), 0.5),
        alpha=np.ones((2, 1)), theta=np.array([4.0]), labor_income=np.array([100.0, 100.0]),
    )
    tariffs_new = np.array([[[1.0, 1.2], [1.2, 1.0]]])
    cf = cp.solve_counterfactual(tariffs_new=tariffs_new, d_hat=1.0 / tariffs_new)
    _, _, revenue = cp._solve_expenditures(np.ones(2), cp.trade_shares,
                                          tariffs_new, np.zeros(2))
    observations["cp_offsetting_cost_changes"] = {
        "reported_revenue": cf.tariff_revenue_prime.tolist(),
        "revenue_from_expenditure_equations_at_returned_wages_and_shares": revenue.tolist(),
        "reported_iterations": cf.iterations,
    }

    with warnings.catch_warnings(record=True) as caught:
        fake = load_figaro(file_path="/definitely/missing.csv",
                           fallback_to_synthetic=False, custom_c=2, custom_s=2)
    observations["mrio_disabled_fallback_bypassed"] = {
        "returned_countries": fake.n_countries, "returned_sectors": fake.n_sectors,
        "warnings": len(caught), "metadata": fake.metadata,
        "expected": "Missing-file error, or explicit rejection of conflicting options",
    }

    matrix = np.array([[0.0, 2.0], [0.2, 0.0]])
    estimate, low, high = compute_spectral_radius(matrix, max_iter=60, tol=1e-6)
    observations["periodic_matrix_spectral_radius"] = {
        "reported": estimate, "lower": low, "upper": high,
        "analytic": float(np.sqrt(0.4)),
    }

    from puremacro.trade.optimal_tariffs import (
        build_strategic_tariffs,
        evaluate_national_welfare,
        solve_multilateral_nash_tariffs,
    )
    nash = solve_multilateral_nash_tariffs(
        calib, player_countries=calib.country_codes, relaxation=1e-8,
        tol=1e-4, max_iter=2, metric="equivalent_variation",
    )
    game_base = solve_trade_equilibrium(calib, method="condensed")
    payoffs = []
    for rate in (0.0, 0.2):
        ta, tf = build_strategic_tariffs(
            calib, {calib.country_codes[0]: rate, calib.country_codes[1]: 0.0},
        )
        equilibrium = solve_trade_equilibrium(calib, tau=ta, tau_fd=tf, method="condensed")
        payoffs.append(evaluate_national_welfare(
            calib, equilibrium, country_idx=country, metric="equivalent_variation",
            base_equilibrium=game_base, tau=ta, tau_fd=tf,
        ))
    observations["nash_premature_convergence"] = {
        "converged": bool(nash.converged), "tariffs": nash.nash_tariffs,
        "outer_error": nash.outer_error, "payoffs_at_zero_and_20pct_tariff": payoffs,
    }

    from puremacro.dsge.markov_switching import solve_ms_dsge
    ms = solve_ms_dsge(
        [np.array([[1.0]])] * 2, [np.array([[-3.0]])] * 2,
        [np.array([[2.5]])] * 2, [np.array([[1.0]])] * 2,
        np.full((2, 2), 0.5), initial_T=[np.array([[1.8]])] * 2,
    )
    observations["ms_stability_after_failed_solve"] = {
        "converged": bool(ms.converged), "mean_square_stable": bool(ms.mean_square_stable),
        "reported_residual": ms.diff,
        "actual_residual": float((ms.T[0]**2 - 3.0 * ms.T[0] + 2.5).item()),
        "finite_ergodic_covariance": bool(np.isfinite(ms.ergodic_cov.values).all()),
    }

    # Observe the actual training loop. Compare its backward result with a
    # backward pass using the saved current-state input and identical upstream
    # derivatives; restore caches and return the original gradient unchanged.
    history, gradient_errors = [], []
    original_forward, original_backward = DeepMacroMLP.forward, DeepMacroMLP.backward

    def record_forward(self, inputs):
        history.append(np.asarray(inputs).copy())
        return original_forward(self, inputs)

    def inspect_backward(self, derivative):
        actual = original_backward(self, derivative)
        saved_a, saved_z = self._cache_a, self._cache_z
        original_forward(self, history[-2])
        correct = original_backward(self, derivative)
        self._cache_a, self._cache_z = saved_a, saved_z
        actual_flat = np.concatenate([v.ravel() for pair in actual for v in pair])
        correct_flat = np.concatenate([v.ravel() for pair in correct for v in pair])
        gradient_errors.append(float(np.linalg.norm(actual_flat - correct_flat)
                                     / max(np.linalg.norm(correct_flat), 1e-15)))
        return actual

    DeepMacroMLP.forward, DeepMacroMLP.backward = record_forward, inspect_backward
    try:
        solve_deep_macro(DeepMacroModel(n_states=1, n_controls=1), hidden_dims=(4,),
                         n_epochs=1, trajectory_length=10, burn_in=2, batch_size=4)
    finally:
        DeepMacroMLP.forward, DeepMacroMLP.backward = original_forward, original_backward
    observations["deep_macro_wrong_forward_cache"] = {
        "relative_gradient_error": gradient_errors,
        "expected": 0.0,
    }

    print(json.dumps(observations, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
