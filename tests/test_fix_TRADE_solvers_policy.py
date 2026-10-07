"""Regression tests for the 2026-09-30 trade fixes (review key TRADE).

- ``solve_cyprus_manifold_step`` reports the residual and convergence flag of the
  step it returns (after the displacement clamp) and no longer defaults to the
  wrong country index (15 = COL in the canonical roster; CYP is 17).
- ``solve_keller_pac`` merges consecutive ``tau_lambda <= 0`` detections into one
  fold event and labels it.
- Hicksian tariff searches forward ``foreign_saving_units`` (so the
  order-invariant ``"world_income"`` closure is usable) and label boundary
  replies correctly when ``tariff_max <= 2 * tol``.
- ``compute_stone_geary_final_demand`` warns when the bundle overspends.
- ``load_icio_data`` records the provenance of the bundled table.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import scipy.linalg as la

from puremacro.trade import (
    calibrate_trade_model,
    compute_unilateral_optimal_tariff,
    compute_welfare_payoff_matrix,
    load_icio_data,
    solve_multilateral_nash_tariffs,
    solve_policy_equilibrium,
)
from puremacro.trade._hicksian_policy import _boundary_label
from puremacro.trade.data import (
    BUNDLED_ICIO_PROVENANCE,
    CANONICAL_COUNTRY_CODES,
    generate_synthetic_mrio,
)
from puremacro.trade.flexible import (
    FlexiblePreferenceConfig,
    _extract_benchmark_household_data,
    compute_stone_geary_final_demand,
)
from puremacro.trade.solver import (
    _merge_fold_detections,
    solve_cyprus_manifold_step,
    solve_keller_pac,
)


# ---------------------------------------------------------------------------
# solve_cyprus_manifold_step
# ---------------------------------------------------------------------------

def _notebook64_stiff_system():
    """Experiment 4 of notebook 64: a planted stiff coordinate at index 14."""
    rng = np.random.default_rng(42)
    nc, idx = 77, 14
    a = rng.standard_normal((nc, nc))
    s = a.T @ a + np.eye(nc)
    s[idx, :] *= 1e-5
    s[:, idx] *= 1e-5
    s[idx, idx] = 1e-6
    rhs = rng.standard_normal(nc)
    return s, rhs, idx


def test_cyprus_step_reports_residual_of_returned_clamped_step():
    s, rhs, idx = _notebook64_stiff_system()
    exact = la.solve(s, rhs)
    assert np.max(np.abs(exact)) > 1e6  # the exact solve is explosive (2.8e6)

    dw, residual, converged, info = solve_cyprus_manifold_step(
        s, rhs, idx_cyp=idx, tol=2.5e-3, max_disp=0.30, return_info=True)

    # The returned step is the exact direction, scaled down to max|dw| = 0.30.
    assert np.max(np.abs(dw)) == pytest.approx(0.30, rel=1e-12)
    np.testing.assert_allclose(dw, exact * info["clamp_scale"], rtol=1e-6, atol=1e-18)
    assert info["clamped"] and 0 < info["clamp_scale"] < 1e-6

    # Its residual is what it leaves unexplained: (1 - s) * rhs, about ||rhs||_inf.
    actual = float(np.max(np.abs(s @ dw - rhs)))
    assert residual == pytest.approx(actual, rel=1e-12)
    assert residual == pytest.approx(float(np.max(np.abs(rhs))), rel=1e-5)
    assert residual > 3.0  # 3.208 on this draw; 4.3.0 reported 4.4e-16
    assert converged is False
    assert info["residual_kind"] == "linear_system"
    assert info["linear_residual"] == residual

    # The unclamped direction is an exact solve (what 4.3.0 reported).
    assert info["direction_converged"] is True
    assert info["direction_residual"] < 1e-6
    np.testing.assert_allclose(info["unclamped_step"], exact, rtol=1e-6)
    assert info["method"] == "block_elimination"


def test_cyprus_step_without_clamp_is_exact_and_converged():
    s, rhs, idx = _notebook64_stiff_system()
    rhs_small = rhs * 1e-9  # exact step below max_disp: no clamp
    dw, residual, converged = solve_cyprus_manifold_step(s, rhs_small, idx_cyp=idx)
    np.testing.assert_allclose(dw, la.solve(s, rhs_small), rtol=1e-6, atol=1e-18)
    assert residual == pytest.approx(float(np.max(np.abs(s @ dw - rhs_small))), abs=1e-18)
    assert residual <= 2.5e-3 and converged is True

    dw_off, res_off, conv_off = solve_cyprus_manifold_step(s, rhs, idx_cyp=idx, max_disp=None)
    assert np.max(np.abs(dw_off)) > 1e6
    assert conv_off is True and res_off < 1e-6


def test_cyprus_secant_residual_describes_returned_candidate():
    nc, idx = 4, 2
    s = np.eye(nc)
    rhs = np.array([0.01, 0.02, 0.03, 0.04])

    def nonlinear(c):
        f = c ** 3 + c - 0.05
        return f, abs(f), None

    dw, residual, converged, info = solve_cyprus_manifold_step(
        s, rhs, eval_cyp_fn=nonlinear, idx_cyp=idx, tol=1e-6, max_secant_iter=20,
        return_info=True)
    assert not info["clamped"]
    assert info["residual_kind"] == "eval_cyp_fn"
    # The reported residual is eval_cyp_fn at the coordinate actually returned.
    assert residual == pytest.approx(nonlinear(dw[idx])[1], abs=1e-15)
    assert converged is True and residual <= 1e-6

    # Clamped: the scaled step is off the manifold eval_cyp_fn evaluates.
    big = np.array([1.0, 2.0, 0.03, 4.0])
    dw_c, res_c, conv_c, info_c = solve_cyprus_manifold_step(
        s, big, eval_cyp_fn=nonlinear, idx_cyp=idx, tol=1e-6, max_secant_iter=20,
        return_info=True)
    assert info_c["clamped"] and np.max(np.abs(dw_c)) == pytest.approx(0.30)
    assert np.isnan(res_c) and conv_c is False
    assert info_c["residual_kind"] == "unavailable_after_clamp"
    assert info_c["direction_converged"] is True and info_c["direction_residual"] <= 1e-6


def test_cyprus_index_must_be_explicit_or_derived_from_codes():
    s, rhs, idx = _notebook64_stiff_system()
    with pytest.raises(ValueError, match="idx_cyp is required"):
        solve_cyprus_manifold_step(s, rhs)

    assert CANONICAL_COUNTRY_CODES[15] == "COL"  # the removed 4.3.0 default
    assert CANONICAL_COUNTRY_CODES[14] == "CMR"
    _, _, _, info = solve_cyprus_manifold_step(
        s, rhs, country_codes=CANONICAL_COUNTRY_CODES, return_info=True)
    assert info["idx_cyp"] == CANONICAL_COUNTRY_CODES.index("CYP") == 17
    assert info["country"] == "CYP"

    _, _, _, info14 = solve_cyprus_manifold_step(
        s, rhs, idx_cyp=14, country_codes=CANONICAL_COUNTRY_CODES, return_info=True)
    assert info14["country"] == "CMR"

    no_cyp = [c for c in CANONICAL_COUNTRY_CODES if c != "CYP"] + ["XXX"]
    with pytest.raises(ValueError, match="CYP"):
        solve_cyprus_manifold_step(s, rhs, country_codes=no_cyp)
    with pytest.raises(ValueError, match="entries"):
        solve_cyprus_manifold_step(s, rhs, country_codes=CANONICAL_COUNTRY_CODES[:10])

    with pytest.warns(RuntimeWarning, match="outside"):
        solve_cyprus_manifold_step(np.eye(3), np.ones(3) * 0.1, idx_cyp=99)


# ---------------------------------------------------------------------------
# solve_keller_pac fold events
# ---------------------------------------------------------------------------

def test_merge_fold_detections_groups_consecutive_steps():
    def det(step, lam, tau=-1e-7, log_price=0.0):
        return {"step": step, "lambda": lam, "tau_lambda": tau, "sigma_min": 1e-7 * step,
                "kappa_2": 1e8 + step, "min_log_factor_price": log_price}

    # Steps 3-5: lambda falls (turning point); 8-9: stall with a vanishing factor price.
    detections = [det(3, 0.50), det(4, 0.49), det(5, 0.47),
                  det(8, 0.60, log_price=-20.0), det(9, 0.6000001, log_price=-40.0)]
    trace = [(i, lam) for i, lam in enumerate(
        [0.0, 0.1, 0.3, 0.50, 0.49, 0.47, 0.46, 0.55, 0.60, 0.6000001])] + [(10, 0.6000002)]
    events = _merge_fold_detections(detections, trace)
    assert len(events) == 2
    first, second = events
    assert (first["first_step"], first["last_step"], first["n_detections"]) == (3, 5, 3)
    assert first["step"] == 3 and first["lambda"] == 0.50  # keys of the first detection
    assert first["lambda_reversed"] and first["kind"] == "lambda_reversal"
    assert first["lambda_min"] == 0.47 and first["lambda_max"] == 0.50
    assert first["sigma_min_min"] == pytest.approx(3e-7) and first["kappa_2_max"] == 1e8 + 5
    assert (second["first_step"], second["last_step"], second["n_detections"]) == (8, 9, 2)
    assert not second["lambda_reversed"]
    assert second["kind"] == "factor_price_boundary"
    assert second["min_log_factor_price"] == -40.0

    noise = _merge_fold_detections([det(2, 0.3)], [(0, 0.1), (1, 0.2), (2, 0.3), (3, 0.31)])
    assert noise[0]["kind"] == "no_lambda_reversal" and noise[0]["n_detections"] == 1
    assert _merge_fold_detections([], [(0, 0.0)]) == []


@pytest.fixture(scope="module")
def notebook64_cge():
    """The two-country, two-sector calibration of notebook 64, Experiment 3."""
    data = np.zeros((7, 10))
    data[:4, :4] = np.array([[20., 10., 5., 2.], [8., 25., 2., 4.],
                             [5., 5., 12., 18.], [10., 10., 18., 22.]])
    y = np.array([100., 150., 120., 180.])
    taxes = 0.05 * y
    va = y - data[:4, :4].sum(axis=0) - taxes
    data[4, :4], data[5, :4], data[6, :4] = taxes, 2 * va / 3, va / 3
    fd = y - data[:4, :4].sum(axis=1)
    for i in range(4):
        home = [.50, .25, .05, .10, .08, .02] if i < 2 else [.10, .08, .02, .50, .25, .05]
        data[i, 4:10] = fd[i] * np.array(home)
    data[4, 4:] = 0.02 * data[:4, 4:].sum(axis=0)
    return calibrate_trade_model(data, ns=2, nc=2, nfd=3, validate=True)


def test_pac_stall_is_one_fold_event_not_one_entry_per_step(notebook64_cge):
    # sigma = 0 and a 232% tariff: a factor price goes to zero and PAC stalls at
    # lambda ~ 0.992. In 4.3.0 every remaining step (55 of 100) was a fold point.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = solve_keller_pac(notebook64_cge, tau_target=2.32, sigma=0.0, tol=1e-9, max_steps=100)
    assert res.converged is False
    events = res.metadata["fold_points"]
    assert len(events) == 1
    event = events[0]
    assert res.metadata["fold_detections"] == event["n_detections"] > 10
    assert event["last_step"] == 99
    assert event["n_detections"] == event["last_step"] - event["first_step"] + 1
    assert event["lambda_reversed"] is False
    assert event["kind"] == "factor_price_boundary"
    assert event["min_log_factor_price"] < np.log(1e-6)
    assert "not by itself evidence" in res.metadata["fold_monitor"]


def test_pac_notebook64_calibration_records_no_fold(notebook64_cge):
    res = solve_keller_pac(notebook64_cge, tau_target=0.25, sigma=0.1238, ds_init=0.05, tol=1e-9)
    assert res.converged is True
    assert res.metadata["fold_points"] == [] and res.metadata["fold_detections"] == 0


def test_solver_docstrings_drop_unsupported_fold_claim():
    import puremacro.trade.solver as solver
    for text in (solver.__doc__, solver.solve_keller_pac.__doc__):
        assert "0.1238" not in text.split("notebook-64")[0]
        assert "where standard Newton-Raphson diverges" not in text
    assert "RATES" in solver.solve_trade_equilibrium.__doc__


# ---------------------------------------------------------------------------
# Hicksian policy searches: foreign_saving_units and boundary labels
# ---------------------------------------------------------------------------

def _teaching_table():
    """Notebook 63's hand-balanced table (nonzero foreign balances)."""
    flows = np.array([[10., 12., 24., 8., 6., 4., 20., 16.],
                      [8., 14., 4.5, 15., 10.5, 35., 18., 15.]])
    taxes = np.array([4., 6., 1.2, 1., .8, 1.8, 2., 1.2])
    value_added = flows.sum(1) - flows[:, :2].sum(0) - taxes[:2]
    return np.vstack([flows, taxes, np.r_[2 * value_added / 3, np.zeros(6)],
                      np.r_[value_added / 3, np.zeros(6)]])


@pytest.fixture(scope="module")
def ordered_models():
    data = _teaching_table()
    # Same economy with the countries listed in the other order.
    swapped = data[[1, 0, 2, 3, 4]][:, [1, 0, 5, 6, 7, 2, 3, 4]]
    ab = calibrate_trade_model(data, ns=1, nc=2, nfd=3, country_codes=["A", "B"])
    ba = calibrate_trade_model(swapped, ns=1, nc=2, nfd=3, country_codes=["B", "A"])
    return ab, ba


OPTIONS = dict(metric="hicksian_ev", sigma=2., consumption_categories=(0, 2), ge_tol=1e-9)


def _matrix(calib, units):
    base = solve_policy_equilibrium(calib, sigma=2., tol=1e-9, foreign_saving_units=units)
    return compute_welfare_payoff_matrix(calib, player_a="A", player_b="B", optimal_a=.10,
                                         optimal_b=.10, base_equilibrium=base,
                                         foreign_saving_units=units, **OPTIONS)


def test_world_income_closure_reaches_the_policy_search_and_is_order_invariant(ordered_models):
    ab, ba = ordered_models
    world_ab, world_ba = _matrix(ab, "world_income"), _matrix(ba, "world_income")
    assert world_ab.metadata["foreign_saving_units"] == "world_income"
    for eq in world_ab.scenarios.values():
        assert eq.metadata["foreign_saving_units"] == "world_income"
    np.testing.assert_allclose(world_ab.payoff_matrix, world_ba.payoff_matrix, rtol=0, atol=1e-8)

    # The default numeraire closure depends on which country is listed first.
    num_ab, num_ba = _matrix(ab, "numeraire"), _matrix(ba, "numeraire")
    assert num_ab.metadata["foreign_saving_units"] == "numeraire"
    assert np.max(np.abs(num_ab.payoff_matrix - num_ba.payoff_matrix)) > 1e-2
    # A's EV from its own 10% tariff: 2.5404% either way under world_income.
    assert world_ab.payoff_matrix[1, 0, 0] == pytest.approx(2.5404, abs=5e-4)


def test_world_income_unilateral_and_nash_searches_run(ordered_models):
    ab, _ = ordered_models
    base = solve_policy_equilibrium(ab, sigma=2., tol=1e-9, foreign_saving_units="world_income")
    opt = compute_unilateral_optimal_tariff(ab, country_idx="A", tariff_max=.2, num_grid=3,
                                            method="grid", base_equilibrium=base,
                                            foreign_saving_units="world_income", **OPTIONS)
    assert opt.metadata["foreign_saving_units"] == "world_income"
    assert opt.equilibrium.metadata["foreign_saving_units"] == "world_income"
    nash = solve_multilateral_nash_tariffs(ab, player_countries=("A", "B"), tariff_max=.2,
                                           best_response_grid_size=3, max_iter=1, relaxation=1.,
                                           base_equilibrium=base,
                                           foreign_saving_units="world_income", **OPTIONS)
    assert nash.metadata["foreign_saving_units"] == "world_income"
    assert nash.equilibrium.metadata["foreign_saving_units"] == "world_income"


def test_policy_search_rejects_mismatched_or_invalid_closure(ordered_models):
    ab, _ = ordered_models
    numeraire_base = solve_policy_equilibrium(ab, sigma=2., tol=1e-9)
    with pytest.raises(ValueError, match="foreign_saving_units"):
        compute_unilateral_optimal_tariff(ab, country_idx="A", num_grid=3, base_equilibrium=numeraire_base,
                                          foreign_saving_units="world_income", **OPTIONS)
    with pytest.raises(ValueError, match="foreign_saving_units"):
        compute_unilateral_optimal_tariff(ab, country_idx="A", num_grid=3, base_equilibrium=numeraire_base,
                                          foreign_saving_units="dollars", **OPTIONS)


@pytest.mark.parametrize("rate, maximum, tol, expected", [
    (0.0, 0.3, 1e-4, "lower"), (0.3, 0.3, 1e-4, "upper"), (0.1, 0.3, 1e-4, "interior"),
    (5e-5, 5e-5, 1e-4, "upper"),        # 4.3.0: "lower"
    (4.8e-5, 5e-5, 1e-4, "upper"),      # 4.3.0: "lower"
    (1e-4, 1e-4, 1e-4, "upper"),        # 4.3.0: "lower"
    (0.0, 5e-5, 1e-4, "lower"),
    (2.5e-5, 5e-5, 1e-4, "interior"),   # midpoint of overlapping bands
    (1.5e-4, 2e-4, 1e-4, "upper"),
])
def test_boundary_label(rate, maximum, tol, expected):
    assert _boundary_label(rate, maximum, tol) == expected


def test_small_ceiling_replies_are_labelled_upper(ordered_models):
    ab, _ = ordered_models
    base = solve_policy_equilibrium(ab, sigma=2., tol=1e-9)
    opt = compute_unilateral_optimal_tariff(ab, country_idx="A", tariff_max=5e-5, num_grid=3,
                                            base_equilibrium=base, **OPTIONS)
    assert opt.optimal_tariff_rate == pytest.approx(5e-5)
    assert opt.metadata["boundary"] == "upper"
    assert opt.metadata["boundary_bands_overlap"] is True
    nash = solve_multilateral_nash_tariffs(ab, player_countries=("A", "B"), tariff_max=5e-5,
                                           best_response_grid_size=3, max_iter=3, relaxation=.8,
                                           tol=1e-4, regret_tol=1e-6, base_equilibrium=base, **OPTIONS)
    assert nash.metadata["best_response_boundaries"] == {"A": "upper", "B": "upper"}


# ---------------------------------------------------------------------------
# Stone-Geary feasibility warning
# ---------------------------------------------------------------------------

def test_stone_geary_warns_when_subsistence_exceeds_budget(notebook64_cge):
    calib = notebook64_cge
    ycon0 = _extract_benchmark_household_data(calib)[0]
    prices = np.ones((1, calib.n_sectors, calib.n_countries))
    pref = FlexiblePreferenceConfig(mu_s=0.9)  # 0.9 > tanh(3)/3: binds at low income
    budget_share = calib.theta[:, 0:1, :]

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        c_ok = compute_stone_geary_final_demand(ycon0, prices, calib, pref)
    np.testing.assert_allclose(np.sum(prices * c_ok, axis=1, keepdims=True),
                               budget_share * np.reshape(ycon0, (1, 1, -1)), rtol=1e-12)

    low = 0.05 * ycon0
    with pytest.warns(RuntimeWarning, match="exceeds the household budget"):
        c_low = compute_stone_geary_final_demand(low, prices, calib, pref)
    spend = np.sum(prices * c_low, axis=1, keepdims=True)
    assert np.all(spend > budget_share * np.reshape(low, (1, 1, -1)))

    with warnings.catch_warnings():  # a uniform mu below tanh(3)/3 never binds
        warnings.simplefilter("error", RuntimeWarning)
        for u in (1e-3, 0.05, 0.5):
            compute_stone_geary_final_demand(u * ycon0, prices, calib, FlexiblePreferenceConfig(mu_s=0.33))


# ---------------------------------------------------------------------------
# Data provenance and the synthetic generator's documentation
# ---------------------------------------------------------------------------

def test_bundled_icio_records_regression_fixture_provenance():
    icio = load_icio_data(source="legacy", return_structured=True)
    assert icio.metadata["is_regression_fixture"] is True
    assert icio.metadata["source_export_md5"] == "d1b887aaafa54ab3f28fde78fcd21cdf"
    assert "ADVISORY" in icio.metadata["advisory"]
    assert icio.metadata == BUNDLED_ICIO_PROVENANCE
    icio.metadata["is_regression_fixture"] = "mutated"  # a copy, not the constant
    assert BUNDLED_ICIO_PROVENANCE["is_regression_fixture"] is True
    assert "regression fixture" in load_icio_data.__doc__


def test_synthetic_mrio_numbers_do_not_depend_on_dataset():
    assert "high-fidelity" not in generate_synthetic_mrio.__doc__.lower()
    tables = [generate_synthetic_mrio(ds, custom_c=3, custom_s=3)
              for ds in ("oecd", "figaro", "exiobase", "wiod", "eora")]
    for other in tables[1:]:
        np.testing.assert_array_equal(other.intermediate_matrix, tables[0].intermediate_matrix)
        np.testing.assert_allclose(other.gross_output, tables[0].gross_output, rtol=1e-14, atol=0)
    share = tables[0].taxes_less_subsidies / tables[0].gross_output
    assert 0.05 < share.min() and share.max() < 0.25
