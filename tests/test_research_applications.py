"""End-to-end numerical and provenance checks for the two research dossiers."""
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from puremacro import __version__
from puremacro.examples.structural_irf_matching import run_application as run_structural, simulate_data
from puremacro.examples.distributional_trade_enigh import run_application as run_distributional


@pytest.fixture(scope="module")
def structural_study(tmp_path_factory):
    output = tmp_path_factory.mktemp("nk_irf_study")
    result = run_structural(output)  # The public default: 4,000 observations.
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    return output, result, manifest


def test_nk_study_recovers_parameters_from_independent_simulated_data(structural_study):
    output, fit, manifest = structural_study
    assert fit.success and fit.inference_valid and fit.identification_rank == 3
    # Tolerances allow sampling error; they do not pin optimizer-specific digits.
    truth = np.array([1.5, .15, .6])
    assert np.all(np.abs(fit.theta - truth) < np.array([.20, .025, .06]))
    assert np.all(np.isfinite(fit.standard_errors)) and np.all(fit.standard_errors > 0)
    assert fit.parameter_names == ("sigma", "kappa", "rho")
    parameters = pd.read_csv(output / "parameters.csv")
    assert_allclose(parameters.simulation_truth, truth)
    assert_allclose(parameters.estimate, fit.theta)
    assert_allclose(parameters.se, fit.standard_errors)
    assert manifest["observations"] == 4000 and len(pd.read_csv(output / "simulated_data.csv")) == 4000
    assert manifest["is_empirical_replication"] is False
    assert manifest["evidence_kind"] == "synthetic_parameter_recovery"
    assert manifest["package_version"] == __version__
    assert manifest["data_sha256"] == hashlib.sha256((output / "simulated_data.csv").read_bytes()).hexdigest()
    assert manifest["inference_valid"] and 0 <= manifest["j_pvalue"] <= 1
    assert "y = y(+1)" in manifest["model"]
    assert (output / "report.md").is_file() and (output / "irf_fit.png").stat().st_size > 1000


def test_nk_study_keeps_joint_covariance_and_distinguishes_held_out_moments(structural_study):
    output, fit, manifest = structural_study
    covariance = pd.read_csv(output / "joint_covariance.csv", index_col=0)
    targets = pd.read_csv(output / "lp_targets.csv")
    all_labels = [f"{response}:eps:h={h}" for response in ("y", "pi") for h in range(9)]
    fitted = [f"{response}:eps:h={h}" for response in ("y", "pi") for h in range(5)]
    held = [f"{response}:eps:h={h}" for response in ("y", "pi") for h in range(5, 9)]
    assert list(covariance.index) == list(covariance.columns) == targets.label.tolist() == all_labels
    matrix = covariance.to_numpy()
    assert_allclose(matrix, matrix.T, atol=1e-18)
    assert np.linalg.eigvalsh(matrix).min() > 0
    scale = np.sqrt(np.diag(matrix))
    correlations = matrix / scale[:, None] / scale[None, :]
    assert np.max(np.abs(correlations - np.eye(len(matrix)))) > .1
    assert_allclose(covariance.loc[fitted, fitted], fit.targets.covariance, rtol=1e-12, atol=1e-20)
    assert_allclose(covariance.loc[held, held], fit.held_out_targets.covariance, rtol=1e-12, atol=1e-20)
    assert fit.targets.labels == tuple(fitted) and fit.held_out_targets.labels == tuple(held)
    training = pd.read_csv(output / "fitted_moments.csv")
    validation = pd.read_csv(output / "held_out_moments.csv")
    assert training.label.tolist() == fitted and training.used_in_fit.all()
    assert validation.label.tolist() == held and not validation.used_in_fit.any()
    assert not (set(training.label) & set(validation.label))
    assert_allclose(validation.model, fit.held_out_model)
    assert manifest["training_horizons"] == list(range(5))
    assert manifest["held_out_horizons"] == list(range(5, 9))
    assert "one percentage point" in manifest["shock_normalization"]
    assert any("same data" in item and "descriptive" in item for item in manifest["limitations"])


@pytest.fixture(scope="module")
def distributional_study(tmp_path_factory):
    output = tmp_path_factory.mktemp("enigh_incidence_study")
    # Nondefault assumptions make an ignored keyword visible in the outputs.
    options = dict(tariff_rate=.12, food_import_share=.30, pass_through=.8,
                   cash_wage_change=.01, transfer_budget_share=.004)
    results = run_distributional(output, **options)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    return output, results, manifest


def test_enigh_study_preserves_observed_source_and_explicit_scenario_assumptions(distributional_study):
    output, results, manifest = distributional_study
    expected = {f"{rule}_{scenario}" for rule in ("fixed_baskets", "cobb_douglas")
                for scenario in ("no_transfer", "equal_per_household", "bottom_four_deciles")}
    assert set(results) == expected
    combined = pd.read_csv(output / "all_scenarios.csv")
    assert len(combined) == 60 and combined.groupby(["preferences", "scenario"]).size().eq(10).all()
    observed = pd.read_csv(output / "observed_enigh_deciles.csv", index_col=0)
    assert observed.households.sum() == 38_830_230
    source = manifest["observed_data"]
    assert source["is_synthetic"] is False
    assert source["source_sha256"] == "7af7850495255fb1e6a9cb139cf531e5a62c7fd2a9a1de5ba0503f03aa82490b"
    assert source["monetary_unit"] == "MXN" and source["period"] == "quarter"
    assert "INEGI" in source["source"] and source["source_url"].startswith("https://www.inegi.org.mx/")
    assumptions = manifest["assumptions"]
    assert assumptions["tariff_rate"] == .12 and assumptions["food_import_share"] == .30
    assert assumptions["pass_through"] == .8 and assumptions["cash_wage_change"] == .01
    assert_allclose(assumptions["prices"]["food"], 1.0288)
    assert all(value == 1 for key, value in assumptions["prices"].items() if key != "food")
    assert "tariff_rate" not in source and "food_import_share" not in source
    assert any("NOT inferred tariff revenue" in item for item in manifest["limitations"])
    assert any("no sampling confidence intervals" in item for item in manifest["limitations"])
    assert (output / "report.md").is_file() and (output / "distributional_incidence.png").stat().st_size > 1000
    for name, result in results.items():
        assert len(pd.read_csv(output / f"{name}.csv")) == 10
        assert_allclose(result.groups.income_change, .01 * observed.cash_wages, rtol=1e-12)


def test_enigh_scenarios_conserve_transfers_and_match_independent_basket_welfare(distributional_study):
    output, results, manifest = distributional_study
    observed = pd.read_csv(output / "observed_enigh_deciles.csv", index_col=0)
    sectors = list(manifest["observed_data"]["categories"])
    budgets = observed[sectors].sum(axis=1)
    population = observed.households
    expected_pool = .004 * float(population @ budgets)
    assert_allclose(manifest["assumptions"]["transfer_pool_mxn_per_quarter"], expected_pool, rtol=1e-12)
    for rule in ("fixed_baskets", "cobb_douglas"):
        no = results[f"{rule}_no_transfer"]
        equal = results[f"{rule}_equal_per_household"]
        targeted = results[f"{rule}_bottom_four_deciles"]
        assert_allclose(no.groups.transfer, 0.)
        assert_allclose(equal.groups.transfer, expected_pool / population.sum(), rtol=1e-12)
        assert_allclose(targeted.groups.transfer.iloc[:4], expected_pool / population.iloc[:4].sum(), rtol=1e-12)
        assert_allclose(targeted.groups.transfer.iloc[4:], 0.)
        for scenario in (equal, targeted):
            assert_allclose(population @ scenario.groups.transfer, expected_pool, rtol=1e-12)
            assert_allclose(scenario.aggregate.total_transfer, expected_pool, rtol=1e-12)
        # Closed-form expenditure functions independent of the example wrapper.
        food_share = observed.food / budgets
        price_index = 1 + .0288 * food_share if rule == "fixed_baskets" else 1.0288 ** food_share
        final_budget = budgets + .01 * observed.cash_wages
        expected_ev = final_budget / price_index - budgets
        expected_cv = final_budget - budgets * price_index
        assert_allclose(no.groups.ev, expected_ev, rtol=1e-10, atol=1e-8)
        assert_allclose(no.groups.cv, expected_cv, rtol=1e-10, atol=1e-8)
        assert_allclose(no.aggregate.total_ev, population @ expected_ev, rtol=1e-10)


@pytest.mark.parametrize("bad", [
    {"tariff_rate": -.1}, {"food_import_share": 1.1}, {"food_import_share": -.1},
    {"pass_through": -1.}, {"cash_wage_change": -1.01},
    {"transfer_budget_share": -.01}, {"tariff_rate": np.nan}, {"pass_through": np.inf},
])
def test_enigh_application_rejects_bad_assumptions_before_writing(tmp_path, bad):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="invalid"):
        run_distributional(output, **bad)
    assert not output.exists()


@pytest.mark.parametrize("n", [99, 100.5, "4000"])
def test_nk_simulation_requires_an_adequate_integer_sample(n):
    with pytest.raises(ValueError, match="integer of at least 100"):
        simulate_data(n=n)
