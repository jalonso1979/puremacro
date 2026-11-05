"""Offline checks against author-code exports and economic moment conventions."""
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest
from scipy.special import ndtr

from tools.reference_validation.arellano_2008 import (
    FIXTURE, GROSS_R, ROOT, Solution, annual_spread, author_grid, compare_original,
    eligible_windows, income_process, load_original, monte_carlo, simulate, solve,
    summarize_path, verified_inputs,
)

GRID_RECORD = FIXTURE / "public_grid_convergence.json"
ARCHIVE_GRID_RECORD = FIXTURE / "archive_grid_convergence.json"
GUIDE_COLUMNS = ("debt_pct", "spread_mean_pct", "annual_default_pct", "spread_std_pp", "corr_tb_spread")


def test_archived_sources_and_external_export_integrity():
    verified_inputs()
    manifest = json.loads((FIXTURE / "reference_manifest.json").read_text())
    assert hashlib.sha256((FIXTURE / "reference.npz").read_bytes()).hexdigest() == manifest["reference_sha256"]
    assert manifest["outer_iterations"] <= 500
    assert manifest["price_error"] < 1e-8


def test_study_solver_matches_actual_author_fortran():
    result = solve()
    comparison = compare_original(result, load_original())
    assert comparison["passed"], comparison
    assert result.diagnostics["bellman_residual"] < 2e-11
    assert result.diagnostics["price_residual"] < 1e-8


def test_public_calibration_study_matches_the_public_solver():
    from puremacro.sovereign import solve_sovereign_default

    reference = solve(n_y=9, timing="canonical", calibration="public")
    result = solve_sovereign_default(beta=.95282, r_star=.017, theta=.282,
                                     n_b=401, n_y=9, b_min=-.15, b_max=.33,
                                     sigma_eps=0, y_penalty_type="arellano",
                                     howard_steps=0, enforce_monotonicity=False,
                                     tol=1e-10, max_iter=10000)
    assert result.converged
    np.testing.assert_allclose(reference.assets, -10*result.b_grid[::-1], atol=1e-14)
    np.testing.assert_allclose(reference.income, 10*result.y_grid, atol=1e-12)
    np.testing.assert_allclose(reference.price, result.q_bond[::-1], atol=1e-8, rtol=0)
    np.testing.assert_allclose(reference.value, .1*(result.v_star[::-1]-1/(1-.95282)), atol=1e-8, rtol=0)
    np.testing.assert_array_equal(reference.default, result.default_prob[::-1] > .5)
    np.testing.assert_allclose(reference.assets[reference.policy], -10*result.policy_b_prime[::-1], atol=1e-12)


@pytest.mark.parametrize("duplicates", [False, True])
def test_monotone_search_matches_exhaustive_cash_choice(duplicates):
    from tools.reference_validation._arellano_monge import maximize

    rng = np.random.default_rng(2026)
    assets = np.linspace(-.3, .2, 63)
    income = np.linspace(1., 2., 7)
    cash = rng.uniform(-.2, .3, (63, 7))
    continuation = rng.normal(scale=.2, size=(63, 7))
    if duplicates:
        cash[:] = 0.
        continuation[:] = 0.
    value, policy = maximize(assets, income, cash, continuation, np.argsort(-cash, axis=0, kind="stable"))
    objective = -1/(income[None, None, :]+assets[:, None, None]+cash[None, :, :])+continuation[None, :, :]
    np.testing.assert_allclose(value, objective.max(axis=1), atol=1e-15, rtol=0)
    np.testing.assert_array_equal(policy, 62-np.argmax(objective[:, ::-1], axis=1))


def test_accelerated_equilibrium_matches_author_fortran_and_exhaustive_check():
    reference = solve(policy_search="monotone")
    assert compare_original(reference, load_original())["passed"]
    assert reference.diagnostics["exhaustive_bellman_residual"] < 2e-11


def test_grid_refinements_hold_support_and_exact_zero():
    original = load_original()
    np.testing.assert_array_equal(author_grid(), original.assets)
    for refinement in (1, 2):
        fine = author_grid(refinement)
        np.testing.assert_array_equal(fine[::2**refinement], original.assets)
        assert np.sum(fine == 0) == 1
        assert np.all(np.diff(fine) > 0)


def test_archived_income_process_matches_rounded_tauchen_inputs():
    z, archived = verified_inputs()
    expected_z = np.linspace(-3*.025/np.sqrt(1-.945**2), 3*.025/np.sqrt(1-.945**2), 21)
    edges = np.r_[-np.inf, (expected_z[1:]+expected_z[:-1])/2, np.inf]
    expected_p = np.diff(ndtr((edges[None, :]-.945*expected_z[:, None])/.025), axis=1)
    np.testing.assert_allclose(z, expected_z, atol=5e-9, rtol=0)
    np.testing.assert_allclose(archived, expected_p, atol=5e-9, rtol=0)
    y, p = income_process(41)
    np.testing.assert_allclose(y[::2], 10*np.exp(z), atol=6e-8, rtol=0)
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=5e-16, rtol=0)


def test_compound_annual_spread_is_not_simple_annualization():
    q = np.array([1/1.017, 1/1.05])
    actual = annual_spread(q, 1.017)
    np.testing.assert_allclose(actual, [0, 1.05**4-1.017**4], atol=1e-15)
    assert actual[1] != pytest.approx(4*(1.05-1.017))
    with pytest.raises(ValueError, match="positive"):
        annual_spread(np.array([0.0]), GROSS_R)


def test_predefault_windows_require_contiguous_market_access():
    price = np.ones(300)
    entry = np.zeros(300, dtype=bool)
    entry[[60, 100, 180, 260]] = True
    price[[60, 61, 100, 101, 180, 181, 200, 260]] = 0
    # 60 has too little history, 100 includes exclusion, 260 contains 200.
    np.testing.assert_array_equal(eligible_windows(dict(price=price, entry=entry)), [180])


def test_exclusion_timing_and_default_entries_are_distinct():
    toy = Solution(np.array([-1., 0.]), np.array([1.]), np.ones((1, 1)), np.array([.9]),
                   np.zeros((2, 1)), np.ones((2, 1)), np.ones((2, 1), dtype=int),
                   np.ones((2, 1), dtype=bool), "archive", theta=1.)
    archived = simulate(toy, seed=1, periods=6, burn=0)
    canonical = simulate(replace(toy, timing="canonical"), seed=1, periods=6, burn=0)
    np.testing.assert_array_equal(archived["entry"], [1, 0, 1, 0, 1, 0])
    np.testing.assert_array_equal(canonical["entry"], np.ones(6, dtype=bool))
    assert np.all(archived["excluded"]) and archived["entry"].sum() == 3
    np.testing.assert_array_equal(archived["consumption"], archived["output"])


def test_moments_match_author_matlab_block_run_in_octave():
    reference = json.loads((FIXTURE / "moments_reference.json").read_text())
    actual = summarize_path(simulate(load_original(), seed=reference["seed"],
                                    periods=reference["periods"], burn=reference["burn"]))
    for name, expected in reference["moments"].items():
        assert actual["moments"][name] == pytest.approx(expected, abs=1e-10, rel=0)
    d = actual["diagnostics"]
    hazard = d["default_entries"]/(d["default_entries"]+d["repayment_periods"])
    assert d["annual_access_hazard_pct"] == pytest.approx(100*(1-(1-hazard)**4))
    assert d["annual_access_hazard_pct"] < actual["moments"]["annual_default_pct"]
    assert d["boundary_choice_share_pct"] == 0


def test_uncertainty_requires_independent_replications_and_enough_episodes():
    with pytest.raises(ValueError, match="distinct seeds"):
        monte_carlo(load_original(), [2, 2], 100, 10)
    path = simulate(load_original(), seed=2, periods=100, burn=10)
    with pytest.raises(ValueError, match="eligible windows"):
        summarize_path(path)


def test_grid_gate_distinguishes_drift_from_simulation_uncertainty():
    from tools.reference_validation.check_arellano_grids import TOLERANCES, paired_comparison

    def runs(debts):
        return dict(runs=[dict(seed=i, moments={name: debt if name == "debt_pct" else 0.
                                              for name in TOLERANCES}) for i, debt in enumerate(debts)])

    baseline = runs([0., 0., 0., 0.])
    assert paired_comparison(baseline, runs([.05]*4))["passed"]
    drift = paired_comparison(baseline, runs([.15]*4))
    assert drift["moments"]["debt_pct"]["status"] == "grid_shift_exceeds_tolerance"
    imprecise = paired_comparison(baseline, runs([-.2, -.2, .2, .2]))
    assert not imprecise["passed"]
    assert imprecise["moments"]["debt_pct"]["status"] == "sampling_precision_insufficient"
    # Pair by seed, not by list position: otherwise an identical experiment
    # reordered on disk spuriously acquires sampling uncertainty.
    reference = runs([-.2, -.2, .2, .2])
    reordered = dict(runs=list(reversed(reference["runs"])))
    assert paired_comparison(reference, reordered)["passed"]
    with pytest.raises(ValueError, match="same distinct seeds"):
        paired_comparison(baseline, dict(runs=reordered["runs"][:-1]))


def test_grid_check_solves_public_cases_with_the_certified_monotone_search(monkeypatch):
    from tools.reference_validation import check_arellano_grids as grids

    calls = []

    def recorder(**kwargs):
        calls.append(kwargs)
        return "solution"

    monkeypatch.setattr(grids.reference, "solve", recorder)
    assert grids.solve_case(2, 321) == "solution"
    assert calls == [dict(refinement=2, n_y=321, timing="canonical",
                          calibration="public", policy_search="monotone")]


def test_public_calibration_monotone_search_matches_dense_search():
    dense = solve(n_y=9, timing="canonical", calibration="public")
    fast = solve(n_y=9, timing="canonical", calibration="public", policy_search="monotone")
    np.testing.assert_array_equal(fast.policy, dense.policy)
    np.testing.assert_array_equal(fast.default, dense.default)
    np.testing.assert_allclose(fast.value, dense.value, atol=1e-11, rtol=0)
    np.testing.assert_allclose(fast.price, dense.price, atol=1e-11, rtol=0)
    assert fast.diagnostics["exhaustive_bellman_residual"] < 2e-11


def test_study_income_refinements_are_optional_odd_and_above_the_archived_grid():
    from tools.reference_validation import run_arellano_study as study

    names = [name for name, _ in study.specifications(None)]
    assert names == ["original_200_21", "refined_399_21", "refined_797_21", "refined_797_41",
                     "refined_797_81", "canonical_200_21", "public_321_21"]
    finer = [name for name, _ in study.specifications(None, [41, 81, 161, 321])]
    assert finer[3:7] == ["refined_797_41", "refined_797_81", "refined_797_161", "refined_797_321"]
    for bad in ([42], [21], [1]):
        with pytest.raises(ValueError, match="odd"):
            study.specifications(None, bad)


def test_frozen_public_grid_record_passes_and_recomputes_from_its_seeds():
    from tools.reference_validation.check_arellano_grids import TOLERANCES, paired_comparison

    record = json.loads(GRID_RECORD.read_text())
    assert record["status"] == "passed" and record["passed"] is True
    assert record["tolerances"] == TOLERANCES
    assert record["calibration"]["timing"] == "canonical"
    assert record["calibration"]["policy_search"] == "monotone"
    finest = record["cases"]["public_6401_641"]
    assert (finest["n_assets"], finest["n_income"]) == (6401, 641)
    for case in record["cases"].values():
        design, diagnostics = case["design"], case["diagnostics"]
        assert design["seeds"] == list(range(1979, 1995))
        assert (design["periods"], design["burn"], design["windows"]) == (500000, 1000, 1000)
        assert diagnostics["policy_search"] == "monotone"
        assert diagnostics["bellman_residual"] < 2e-12
        assert diagnostics["exhaustive_bellman_residual"] < 2e-11
        assert diagnostics["price_residual"] < 1e-8
    comparisons = record["comparisons"]
    assert set(comparisons) == {"final_asset_refinement", "final_income_refinement"}
    assert (comparisons["final_asset_refinement"]["coarse"], comparisons["final_asset_refinement"]["fine"]) \
        == ("public_3201_641", "public_6401_641")
    assert (comparisons["final_income_refinement"]["coarse"], comparisons["final_income_refinement"]["fine"]) \
        == ("public_6401_321", "public_6401_641")
    # Recompute from the per-seed moments: the verdict must not rest on the stored summary.
    for comparison in comparisons.values():
        recomputed = paired_comparison(record["cases"][comparison["coarse"]]["simulation"],
                                       record["cases"][comparison["fine"]]["simulation"])
        assert recomputed["passed"]
        for name, value in recomputed["moments"].items():
            assert value["status"] == comparison["moments"][name]["status"] == "passed"
            assert value["absolute_change_bound"] == pytest.approx(
                comparison["moments"][name]["absolute_change_bound"], abs=1e-12)


@pytest.mark.parametrize("guide,thousands,decimal", [("docs/arellano_replication.md", ",", "."),
                                                     ("docs/es/arellano_replication.md", ".", ",")])
def test_guides_tabulate_the_frozen_public_grid_record(guide, thousands, decimal):
    record = json.loads(GRID_RECORD.read_text())
    text = (ROOT / guide).read_text(encoding="utf-8")
    for case in record["cases"].values():
        moments = case["simulation"]["moments"]
        nodes = f'{case["n_assets"]:,}'.replace(",", thousands)
        row = f'| {nodes} × {case["n_income"]} | ' + " | ".join(
            f'{moments[k]["mean"]:.3f}'.replace(".", decimal) for k in GUIDE_COLUMNS) + " |"
        assert row in text, row


def test_grid_check_solves_archive_cases_with_author_timing_and_grid(monkeypatch):
    from tools.reference_validation import check_arellano_grids as grids

    calls = []

    def recorder(**kwargs):
        calls.append(kwargs)
        return "solution"

    monkeypatch.setattr(grids.reference, "solve", recorder)
    assert grids.solve_case(3, 641, "archive") == "solution"
    assert calls == [dict(refinement=3, n_y=641, timing="archive",
                          calibration="archive", policy_search="monotone")]
    # Midpoint refinement of the author's 200-node grid, not the public 401-node ladder.
    assert [grids.n_assets(r, "archive") for r in (0, 2, 5)] == [200, 797, 6369]
    assert [len(author_grid(r)) for r in (0, 2, 5)] == [200, 797, 6369]
    assert [grids.n_assets(r) for r in (0, 4)] == [401, 6401]


def test_frozen_archive_grid_record_passes_and_recomputes_from_its_seeds():
    from tools.reference_validation.check_arellano_grids import TOLERANCES, paired_comparison

    record = json.loads(ARCHIVE_GRID_RECORD.read_text())
    assert record["calibration_name"] == "archive"
    assert record["status"] == "passed" and record["passed"] is True
    assert record["tolerances"] == TOLERANCES
    calibration = record["calibration"]
    assert calibration["timing"] == "archive" and calibration["policy_search"] == "monotone"
    assert calibration["assets_max"] == pytest.approx(1.53502537, abs=1e-8)
    assert calibration["default_cap"] == pytest.approx(9.69, abs=1e-6)
    expected_cases = {f"archive_{n_b}_{n_y}" for n_b in (797, 1593, 3185, 6369) for n_y in (321, 641)}
    assert set(record["cases"]) == expected_cases
    for case in record["cases"].values():
        design, diagnostics = case["design"], case["diagnostics"]
        assert design["seeds"] == list(range(1979, 1995))
        assert (design["periods"], design["burn"], design["windows"]) == (500000, 1000, 1000)
        assert diagnostics["policy_search"] == "monotone"
        assert diagnostics["bellman_residual"] < 4e-12
        assert diagnostics["exhaustive_bellman_residual"] < 4e-12
        assert diagnostics["price_residual"] < 1e-8
    comparisons = record["comparisons"]
    assert (comparisons["final_asset_refinement"]["coarse"], comparisons["final_asset_refinement"]["fine"]) \
        == ("archive_3185_641", "archive_6369_641")
    assert (comparisons["final_income_refinement"]["coarse"], comparisons["final_income_refinement"]["fine"]) \
        == ("archive_6369_321", "archive_6369_641")
    for comparison in comparisons.values():
        recomputed = paired_comparison(record["cases"][comparison["coarse"]]["simulation"],
                                       record["cases"][comparison["fine"]]["simulation"])
        assert recomputed["passed"]
        for name, value in recomputed["moments"].items():
            assert value["status"] == comparison["moments"][name]["status"] == "passed"
            assert value["absolute_change_bound"] == pytest.approx(
                comparison["moments"][name]["absolute_change_bound"], abs=1e-12)
    # The standard design that triggered the registered precision re-simulation stays on record.
    standard = record["preceding_standard_design"]
    assert standard["status"] == "convergence_not_established"
    assert (standard["design"]["periods"], standard["design"]["windows"]) == (100000, 100)
    statuses = {(name, moment): value["status"] for name, comparison in standard["comparisons"].items()
                for moment, value in comparison["moments"].items()}
    assert {key for key, status in statuses.items() if status != "passed"} == {
        ("final_income_refinement", "debt_pct")}
    assert statuses[("final_income_refinement", "debt_pct")] == "sampling_precision_insufficient"


def test_archive_ladder_reproduces_the_study_797_by_321_row():
    # The standard-design 797 x 321 case is the study's archived-calibration row (same seeds, windows).
    means = json.loads(ARCHIVE_GRID_RECORD.read_text())["preceding_standard_design"]["means"]["archive_797_321"]
    assert [round(means[k], 3) for k in GUIDE_COLUMNS[:4]] == [5.144, 3.321, 2.942, 3.098]


@pytest.mark.parametrize("guide,thousands,decimal", [("docs/arellano_replication.md", ",", "."),
                                                     ("docs/es/arellano_replication.md", ".", ",")])
def test_guides_tabulate_the_frozen_archive_grid_record(guide, thousands, decimal):
    record = json.loads(ARCHIVE_GRID_RECORD.read_text())
    text = (ROOT / guide).read_text(encoding="utf-8")
    for case in record["cases"].values():
        moments = case["simulation"]["moments"]
        nodes = f'{case["n_assets"]:,}'.replace(",", thousands)
        row = f'| {nodes} × {case["n_income"]} | ' + " | ".join(
            f'{moments[k]["mean"]:.3f}'.replace(".", decimal) for k in GUIDE_COLUMNS) + " |"
        assert row in text, row


def _public_monotone(n_b: int, n_y: int):
    """PureMacro's public solver at the reference study's public calibration."""
    from puremacro.sovereign import solve_sovereign_default

    return solve_sovereign_default(beta=.95282, r_star=.017, theta=.282, rho_y=.945, sigma_y=.025,
                                   b_min=-.15, b_max=.33, n_b=n_b, n_y=n_y, sigma_eps=0,
                                   y_penalty_type="arellano", enforce_monotonicity=False,
                                   tol=1e-10, max_iter=10000, policy_search="monotone")


def _assert_public_matches_reference(public, reference: Solution):
    # Public debt index k is reference asset index n_b-1-k; income is scaled by 10,
    # so u_public(c/10) = 1 - 10/c and V_public = 1/(1-beta) + 10 V_reference.
    assert public.converged and public.metadata["monotone_certification_gap"] <= 1e-11
    assert np.allclose(public.y_grid * 10, reference.income, rtol=1e-12, atol=0)
    assert np.max(abs(public.bond_price - reference.price[::-1])) <= 1e-8
    assert np.allclose(public.policy_b, (-reference.assets[reference.policy] / 10)[::-1], rtol=0, atol=1e-12)
    default = reference.default[::-1]
    assert np.array_equal(public.default_prob > .5, default)
    repay = 1 / (1 - .95282) + 10 * reference.value[::-1]
    assert np.max(abs(public.v_repay - repay)[~default]) <= 1e-8


def test_public_monotone_matches_reference_401x21():
    _assert_public_matches_reference(_public_monotone(401, 21),
                                     solve(0, 21, timing="canonical", calibration="public"))


@pytest.mark.slow
def test_public_monotone_matches_reference_1601x21():
    _assert_public_matches_reference(
        _public_monotone(1601, 21),
        solve(2, 21, timing="canonical", calibration="public", policy_search="monotone"))
