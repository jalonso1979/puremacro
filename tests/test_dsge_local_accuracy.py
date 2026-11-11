"""Taylor tensor contractions checked against independent Kronecker algebra."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from puremacro.dsge import load_mod
from puremacro.dsge.local_accuracy import diagnose_perturbation, evaluate_perturbation


def polynomial_solution():
    variables, states, shocks = ("x", "z", "c"), ("x", "z"), ("u", "v")
    rng = np.random.default_rng(912)
    attributes = dict(variable_names=variables, state_variables=states, shock_names=shocks,
                      ys=pd.Series([2, 3, 4], index=variables))
    for name, width in [("ghx", 2), ("ghu", 2), ("ghxx", 4), ("ghxu", 4),
                        ("ghuu", 4), ("ghxxx", 8), ("ghxxu", 8), ("ghxuu", 8),
                        ("ghuuu", 8), ("ghxss", 2), ("ghuss", 2)]:
        attributes[name] = pd.DataFrame(rng.normal(size=(3, width)), index=variables)
    attributes["ghs2"] = pd.Series(rng.normal(size=3), index=variables)
    rules = SimpleNamespace(**attributes)
    return SimpleNamespace(decision_rules=lambda: rules), rules


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.pyodide_smoke
def test_taylor_contractions_match_dense_oracle_for_batch_and_single(order):
    solution, rules = polynomial_solution()
    x = np.array([[0.1, -0.2], [0.3, 0.05]])
    u = np.array([[0.02, 0.03], [-0.04, 0.01]])
    sigma = 0.7
    expected = []
    for state, innovation in zip(x, u):
        y = rules.ys.to_numpy() + rules.ghx.to_numpy() @ state + rules.ghu.to_numpy() @ innovation
        if order >= 2:
            y += rules.ghxx.to_numpy() @ np.kron(state, state) / 2
            y += rules.ghxu.to_numpy() @ np.kron(state, innovation)
            y += rules.ghuu.to_numpy() @ np.kron(innovation, innovation) / 2
            y += rules.ghs2.to_numpy() * sigma**2 / 2
        if order >= 3:
            y += rules.ghxxx.to_numpy() @ np.kron(np.kron(state, state), state) / 6
            y += rules.ghxxu.to_numpy() @ np.kron(np.kron(state, state), innovation) / 2
            y += rules.ghxuu.to_numpy() @ np.kron(np.kron(state, innovation), innovation) / 2
            y += rules.ghuuu.to_numpy() @ np.kron(np.kron(innovation, innovation), innovation) / 6
            y += (rules.ghxss.to_numpy() @ state + rules.ghuss.to_numpy() @ innovation) * sigma**2 / 2
        expected.append(y)
    levels = x + np.array([2, 3])
    np.testing.assert_allclose(evaluate_perturbation(solution, levels, u, sigma=sigma, order=order), expected, atol=1e-14)
    np.testing.assert_allclose(evaluate_perturbation(solution, x, u, sigma=sigma, order=order, deviations=True), expected, atol=1e-14)
    np.testing.assert_allclose(evaluate_perturbation(solution, levels[0], u[0], sigma=sigma, order=order), expected[0], atol=1e-14)


def test_raw_rule_matches_first_step_with_zero_risk_and_rejects_invalid_inputs():
    model = load_mod("var x c; varexo e; model; x=0.8*x(-1)+e; c=0.4*x; end; initval; x=0; c=0; end; steady;")
    np.testing.assert_allclose(evaluate_perturbation(model, [2], [0.1]), [1.7, 0.68], atol=1e-12)
    np.testing.assert_allclose(evaluate_perturbation(model, [[2], [1]], [0.1]), [[1.7, 0.68], [0.9, 0.36]], atol=1e-12)
    for kwargs in ({"order": 2}, {"sigma": -1}, {"sigma": np.nan}, {"order": True}):
        with pytest.raises(ValueError):
            evaluate_perturbation(model, [2], [0.1], **kwargs)
    with pytest.raises(ValueError, match="lagged_states"):
        evaluate_perturbation(model, [2, 1])
    with pytest.raises(ValueError, match="shocks"):
        evaluate_perturbation(model, [2], [np.nan])


def test_risk_scale_changes_expectation_terms_without_rescaling_realized_innovations():
    solution, rules = polynomial_solution()
    a = evaluate_perturbation(solution, [2, 3], [0.1, -0.2], sigma=0, order=2)
    b = evaluate_perturbation(solution, [2, 3], [0.1, -0.2], sigma=1, order=2)
    np.testing.assert_allclose(b-a, rules.ghs2.to_numpy()/2, atol=1e-14)


def test_near_unit_roots_and_rank_deficiency_remain_visible():
    model = load_mod("var x c; varexo e; model; x=0.999999*x(-1)+e; c=0.4*x; end; initval; x=0; c=0; end; steady;")
    report = diagnose_perturbation(model)
    assert report.roots.near_unit.all()
    assert report.jacobian_rank == report.equation_count == 2
    assert any("unit circle" in warning for warning in report.warnings)
    deficient = SimpleNamespace(G=np.array([[0.8]]), _A_minus=np.zeros((2, 2)),
                                _A_plus=np.zeros((2, 2)), _A_0=np.array([[1, 0], [2, 0]]))
    report = diagnose_perturbation(deficient)
    assert report.jacobian_rank == 1
    assert any("rank deficient" in warning for warning in report.warnings)
    absent = diagnose_perturbation(SimpleNamespace(G=np.array([[0.8]])))
    assert absent.jacobian_rank is None
    assert any("not checked" in warning for warning in absent.warnings)
    pd.testing.assert_frame_equal(report.to_frame(), report.roots)
    pd.testing.assert_frame_equal(report.to_dataframe(), report.roots)
    assert isinstance(report.to_markdown(), str)
    assert "\\begin{tabular}" in report.to_latex()
    assert "#table(" in report.to_typst()
    frame = report.to_frame()
    frame.loc[0, "modulus"] = -99
    assert report.roots.loc[0, "modulus"] >= 0


def test_diagnose_perturbation_distinguishes_schur_blocks_from_full_curvature_operator():
    """Non-normal state transitions amplify full higher-order operator conditioning beyond diagonal Schur blocks."""
    # Diagonal Schur blocks have eigenvalue 0.5 -> A_hat + 0.5^k A_plus is well-conditioned (cond ~ 1.5),
    # while large off-diagonal Schur coupling (1e4) makes the full Sylvester operator ill-conditioned.
    h_x = np.array([
        [0.5, 1.0e4],
        [0.0, 0.5],
    ])
    A_hat = np.eye(2)
    A_plus = 0.4 * np.eye(2)
    sol = SimpleNamespace(
        G=h_x,
        _A_minus=np.zeros((2, 2)),
        _A_0=A_hat,
        _A_plus=A_plus,
        A_hat=A_hat,
        A_plus=A_plus,
    )
    report = diagnose_perturbation(sol, condition_warning=1e3, max_curvature_order=3)

    for k in (1, 2, 3):
        assert report.schur_block_condition[k] < 10.0
        assert report.curvature_operator_condition[k] > 1e3
        assert report.curvature_coupling_amplification[k] > 1e3
        assert report.curvature_condition_method[k] == "exact_2norm"

    assert any(
        "despite well-conditioned diagonal Schur blocks" in w
        for w in report.warnings
    )
    curv_df = report.curvature_frame()
    assert list(curv_df["order"]) == [1, 2, 3]
    assert (curv_df["coupling_amplification"] > 1e3).all()

    # Verify matrix-free 1-norm condition estimation path when max_exact_curvature_dim is small
    report_est = diagnose_perturbation(
        sol, condition_warning=1e3, max_curvature_order=3, max_exact_curvature_dim=2,
    )
    for k in (1, 2, 3):
        assert report_est.curvature_condition_method[k] == "onenormest"
        assert report_est.curvature_operator_condition[k] > 1e3


def test_local_order_1_2_3_accuracy_map_zero_risk_vs_stochastic_information():
    """Compare order 1-3 local rules against zero-risk PF paths and stochastic global policies."""
    from puremacro.dsge import (
        read_mod,
        solve_perfect_foresight,
        solve_time_iteration,
    )
    from puremacro.dsge.global_solution import EquilibriumProblem, gaussian_quadrature

    rbc_mod = """
        var k c l a;
        varexo e;
        parameters beta alpha delta deltapsi rho sigma_e;
        beta = 0.96;
        alpha = 0.33;
        delta = 0.08;
        deltapsi = 1.0;
        rho = 0.8;
        sigma_e = 0.02;
        model;
            1 / c = beta * (1 / c(+1)) * (alpha * exp(a(+1)) * (k ^ (alpha - 1)) * (l(+1) ^ (1 - alpha)) + 1 - delta);
            deltapsi * (l ^ 1.0) = (1 - alpha) * exp(a) * (k(-1) ^ alpha) * (l ^ (-alpha)) / c;
            k = (1 - delta) * k(-1) + exp(a) * (k(-1) ^ alpha) * (l ^ (1 - alpha)) - c;
            a = rho * a(-1) + sigma_e * e;
        end;
        steady_state_model;
            a = 0;
            rk = 1 / beta - (1 - delta);
            k_l = (alpha / rk) ^ (1 / (1 - alpha));
            y_l = k_l ^ alpha;
            c_l = y_l - delta * k_l;
            l = ((1 - alpha) * y_l / (deltapsi * c_l)) ^ 0.5;
            k = k_l * l;
            c = c_l * l;
        end;
        shocks;
            var e; stderr 1.0;
        end;
    """
    model = read_mod(rbc_mod)
    sol3 = load_mod(rbc_mod, order=3)
    diag = diagnose_perturbation(sol3, max_curvature_order=3)
    assert diag.jacobian_rank == 4
    for key in ("ghxx", "ghs2", "ghxxx", "ghxss", "ghuss"):
        assert diag.sylvester_residuals[key] < 1e-12

    ss = sol3.decision_rules().ys
    k_ss = float(ss["k"])
    y_ss = np.array([float(ss[v]) for v in model.variables], dtype=float)

    # 1. Zero-future-risk limit (sigma=0.0): compare 1st, 2nd, 3rd order Taylor step
    # against exact nonlinear deterministic perfect foresight (direct & krylov)
    # across a declared state-deviation ladder.
    errors_by_order = {1: [], 2: [], 3: []}
    for scale in (0.04, 0.08, 0.14):
        k0 = k_ss * (1.0 + scale)
        a_lag = 0.05 * scale / 0.14
        e0 = 0.5 * scale / 0.14
        shocks = np.zeros((80, 1))
        shocks[0, 0] = e0
        y_init = np.array([k0, float(ss["c"]), float(ss["l"]), a_lag], dtype=float)
        pf_direct = solve_perfect_foresight(
            model, y_init=y_init, y_end=y_ss, periods=80, shocks=shocks,
            linear_solver="direct", tol=1e-12,
        )
        pf_krylov = solve_perfect_foresight(
            model, y_init=y_init, y_end=y_ss, periods=80, shocks=shocks,
            linear_solver="krylov", tol=1e-12,
        )
        assert pf_direct.converged and pf_krylov.converged
        np.testing.assert_allclose(pf_krylov.path.to_numpy(), pf_direct.path.to_numpy(), atol=1e-10)
        y_pf_t1 = pf_direct.path.iloc[0].to_numpy(dtype=float)

        state_names = list(sol3.decision_rules().state_variables)
        lag_state_vals = np.array([k0 if s == "k" else a_lag for s in state_names], dtype=float)
        for ord_k in (1, 2, 3):
            y_pert_zero_risk = evaluate_perturbation(sol3, lag_state_vals, [e0], sigma=0.0, order=ord_k)
            errors_by_order[ord_k].append(float(np.max(np.abs(y_pert_zero_risk - y_pf_t1))))

        # Verify that sigma=1.0 vs sigma=0.0 reflects precautionary risk (different information problem),
        # not a deterministic solver discrepancy.
        y_stoch_2 = evaluate_perturbation(sol3, lag_state_vals, [e0], sigma=1.0, order=2)
        y_det_2 = evaluate_perturbation(sol3, lag_state_vals, [e0], sigma=0.0, order=2)
        precautionary_shift = y_stoch_2 - y_det_2
        np.testing.assert_allclose(
            precautionary_shift, 0.5 * sol3.decision_rules().ghs2.to_numpy(dtype=float), atol=1e-14
        )
        assert np.linalg.norm(precautionary_shift) > 1e-6

    # Higher order is strictly more accurate at every point on the zero-risk state ladder
    for idx in range(len(errors_by_order[1])):
        assert errors_by_order[3][idx] < errors_by_order[2][idx] < errors_by_order[1][idx]

    # 2. Stochastic policy comparison under the SAME shock distribution:
    # Solve the stochastic growth specialization with global Chebyshev time iteration
    # and compare stochastic accuracy against orders 1, 2, 3 over a state grid.
    stoch_mod = """
        var y c k a;
        varexo e;
        parameters alpha beta delta;
        alpha = 0.33;
        beta = 0.95;
        delta = 0.10;
        model;
            [name='resources'] c + k = y + (1 - delta) * k(-1);
            [name='production'] y = exp(a) * (k(-1) ^ alpha);
            [name='Euler'] 1 / c = beta * (1 / c(+1)) * (alpha * y(+1) / k + 1 - delta);
            [name='productivity'] a = e;
        end;
        steady_state_model;
            rk = 1 / beta - (1 - delta);
            k = (alpha / rk) ^ (1 / (1 - alpha));
            y = k ^ alpha;
            c = y - delta * k;
            a = 0;
        end;
        shocks;
            var e; stderr 0.02;
        end;
    """
    s_model = read_mod(stoch_mod)
    s_sol3 = load_mod(stoch_mod, order=3)
    k_star = float(s_sol3.decision_rules().ys["k"])
    alpha_s, beta_s, delta_s = 0.33, 0.95, 0.10
    names_s = tuple(s_model.variables)

    def resources_s(state):
        return float(np.exp(state[1]) * (state[0] ** alpha_s) + (1.0 - delta_s) * state[0])

    def current_s(state, control):
        y_val = float(np.exp(state[1]) * (state[0] ** alpha_s))
        tot = y_val + (1.0 - delta_s) * state[0]
        vals = {"y": y_val, "c": tot - control[0], "k": control[0], "a": state[1]}
        return np.array([vals[n] for n in names_s], dtype=float)

    def lagged_s(state):
        return np.array([state[0] if n == "k" else 0.0 for n in names_s], dtype=float)

    q_nodes, q_weights = gaussian_quadrature([[0.02 ** 2]], order=3)
    prob_stoch = EquilibriumProblem.from_model(
        s_model,
        state_names=("inherited_capital", "observed_log_productivity"),
        control_names=("savings",),
        domain=((0.7 * k_star, 1.3 * k_star), (-0.05, 0.05)),
        current=current_s,
        lagged=lagged_s,
        transition=lambda s, x, e_next: np.array([x[0], e_next[0]], dtype=float),
        equations=("Euler",),
        exogenous=lambda s: s[1:2],
        shock_nodes=q_nodes,
        shock_weights=q_weights,
        control_bounds=lambda s: [(1e-4, resources_s(s) - 1e-4)],
        admissible=lambda s, x: 0.0 < x[0] < resources_s(s),
    )
    k_idx = list(s_sol3.decision_rules().variable_names).index("k")
    global_sol = solve_time_iteration(
        prob_stoch,
        lambda s: evaluate_perturbation(s_sol3, [s[0]], [s[1]], sigma=1.0, order=2)[[k_idx]],
        basis="chebyshev",
        orders=(7, 3),
        tol=1e-8,
    )
    assert global_sol.converged

    # Compare policies at moderate state deviations under sigma=1.0
    probe_states = np.array([
        [0.85 * k_star, -0.02],
        [k_star, 0.0],
        [1.15 * k_star, 0.02],
    ])
    k_global = global_sol(probe_states)[:, 0]
    k_ord1 = evaluate_perturbation(s_sol3, probe_states[:, :1], probe_states[:, 1:2], sigma=1.0, order=1)[:, k_idx]
    k_ord2 = evaluate_perturbation(s_sol3, probe_states[:, :1], probe_states[:, 1:2], sigma=1.0, order=2)[:, k_idx]
    k_ord3 = evaluate_perturbation(s_sol3, probe_states[:, :1], probe_states[:, 1:2], sigma=1.0, order=3)[:, k_idx]
    assert np.max(np.abs(k_ord3 - k_global)) < np.max(np.abs(k_ord2 - k_global)) < np.max(np.abs(k_ord1 - k_global))


def test_diagnose_perturbation_warns_on_elevated_sylvester_residual():
    """Corrupted higher-order decision rules trigger a warning in diagnose_perturbation."""
    mod_src = """
    var z x;
    varexo e;
    parameters r b c3;
    r = 0.75; b = 0.90; c3 = 0.6;
    model;
        z = r * z(-1) + e;
        x = b * x(+1) + (1/6) * c3 * z^3;
    end;
    initval;
        z = 0; x = 0;
    end;
    steady;
    """
    sol3 = load_mod(mod_src, order=3)
    clean_diag = diagnose_perturbation(sol3)
    assert clean_diag.sylvester_residuals["ghxxx"] < 1e-12
    assert not any("Sylvester residual" in w for w in clean_diag.warnings)

    for bad_thr in (0.0, -1e-8, np.nan, np.inf):
        with pytest.raises(ValueError, match="residual_warning"):
            diagnose_perturbation(sol3, residual_warning=bad_thr)

    # Perturb ghxxx slightly and verify residual warning fires
    corrupted = SimpleNamespace(**{**vars(sol3), "ghxxx": sol3.ghxxx + 1e-3})
    bad_diag = diagnose_perturbation(corrupted, residual_warning=1e-8)
    assert bad_diag.sylvester_residuals["ghxxx"] > 1e-5
    assert any("Sylvester residual for ghxxx" in w for w in bad_diag.warnings)


