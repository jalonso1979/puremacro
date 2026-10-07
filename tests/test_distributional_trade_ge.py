"""Independent accounting and analytic welfare checks for the GE application."""
import hashlib
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from puremacro.datasets.enigh import load_enigh2024_deciles
from puremacro.examples import distributional_trade_ge as application


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    output = tmp_path_factory.mktemp("distributional-ge")
    return output, application.run_application(output)


def test_synthetic_table_is_balanced_and_both_equilibria_are_audited(study):
    _, run = study
    calibration = run["calibration"]
    raw = calibration.data_calibra
    n = calibration.ns * calibration.nc
    # Each producer's total sales equal intermediate costs plus factor payments.
    assert_allclose(raw[:n].sum(axis=1), raw[:, :n].sum(axis=0), atol=2e-13)
    assert_allclose(raw[n], 0., atol=0.)
    assert_allclose(calibration.invforT, 0., atol=2e-13)
    for state in (run["baseline"], run["counterfactual"]):
        assert state.converged
        assert state.metadata["accounting"] == "consistent"
        assert max(state.metadata["account_residuals"].values()) < 1e-8
        assert_allclose(state.p_sol[0, 0, 0], 1., atol=1e-12)


def test_revenue_reconstructed_independently_from_food_import_quantities(study):
    output, run = study
    cf = run["counterfactual"]
    scale = run["manifest"]["units"]["mxn_per_model_unit"]
    # Tensor axes are origin sector, origin country, destination sector/category,
    # destination country. Only REST food delivered to HOME attracts duties.
    food_price = cf.p_sol[0, 0, 1]
    food_inputs = cf.intermediate_flows[0, 1, :, 0].sum()
    food_final = cf.final_demand_flows[0, 1, 0, 0]
    receipts = .10 * food_price * (food_inputs + food_final)
    assert_allclose(cf.tariffs, [receipts, 0.], rtol=1e-12, atol=1e-12)
    assert_allclose(cf.T_sol.ravel(), [receipts, 0.], rtol=1e-10, atol=1e-10)
    assert_allclose(run["manifest"]["fiscal"]["incremental_tariff_revenue_mxn"], receipts * scale, rtol=1e-12)
    table = pd.read_csv(output / "tariff_transactions.csv")
    positive = table[table.tariff_rate > 0]
    assert set(positive.origin) == {"REST"}
    assert set(positive.destination) == {"HOME"}
    assert set(positive.sector) == {"food"}
    assert set(positive.use) == {"intermediate", "final"}
    assert_allclose(table.duty_model_units, table.tariff_rate * table.quantity * table.producer_price, atol=1e-14)
    assert_allclose(positive.duty_model_units.sum(), receipts, rtol=1e-12)


def test_symmetric_fixed_coefficient_economy_matches_closed_form(study):
    _, run = study
    data = load_enigh2024_deciles()
    sectors = list(data.attrs["categories"])
    food_share = float(data.households @ data.food / (data.households @ data[sectors].sum(axis=1)))
    base, cf = run["baseline"], run["counterfactual"]
    # Identical coefficients across sectors make producer prices and physical
    # quantities unchanged. HOME factor prices absorb the input-duty cost.
    # Receipts on intermediate imports exactly offset the aggregate factor loss;
    # final-import duties exactly fund the more expensive consumption basket.
    input_duties = 100 * (.35 / .65) * .20 * food_share * .10
    final_duties = 100 * .25 * food_share * .10
    expected_factor_price = 1 - input_duties / 100
    assert_allclose(cf.p_sol, 1., rtol=0, atol=1e-10)
    assert_allclose(cf.y_sol, base.y_sol, rtol=1e-10, atol=1e-10)
    assert_allclose(cf.w_sol.ravel(), [expected_factor_price, 1.], atol=1e-10)
    assert_allclose(cf.r_sol.ravel(), [expected_factor_price, 1.], atol=1e-10)
    assert_allclose(cf.tariffs[0], input_duties + final_duties, atol=1e-10)
    assert_allclose(cf.metadata["final_expenditure"].ravel()[0], 100 + final_duties, atol=1e-8)


def test_factor_mapping_and_rebates_exhaust_income_without_double_counting(study):
    output, run = study
    data = load_enigh2024_deciles()
    mapping = pd.read_csv(output / "income_mapping.csv", index_col=0)
    calibration, base, cf = run["calibration"], run["baseline"], run["counterfactual"]
    scale = run["manifest"]["units"]["mxn_per_model_unit"]
    assert_allclose(mapping.cash_wages, data.cash_wages, rtol=1e-14)
    assert_allclose(data.households @ mapping.cash_wages,
                    scale * base.w_sol.ravel()[0] * calibration.l_endow.ravel()[0], rtol=1e-13)
    assert_allclose(data.households @ mapping.other_factor_proxy,
                    scale * base.r_sol.ravel()[0] * calibration.k_endow.ravel()[0], rtol=1e-13)
    wage_change = cf.w_sol.ravel()[0] / base.w_sol.ravel()[0] - 1
    rent_change = cf.r_sol.ravel()[0] / base.r_sol.ravel()[0] - 1
    expected = data.cash_wages * wage_change + mapping.other_factor_proxy * rent_change
    assert_allclose(mapping.total_factor_income_change, expected, rtol=1e-12, atol=1e-11)
    for row in run["fiscal_allocation"].itertuples():
        assert abs(row.fiscal_residual_mxn) < 1e-4
        # Large national MXN values amplify the documented GE solve residual.
        assert abs(row.income_residual_mxn) / (scale * 100) < 1e-10
        assert_allclose(row.household_rebate_mxn + row.retained_outside_households_mxn,
                        row.incremental_tariff_revenue_mxn, rtol=1e-13)
        assert_allclose(row.factor_income_change_mxn, data.households @ expected, rtol=1e-12)
        if row.rule == "no_household_rebate":
            assert row.household_rebate_mxn == 0
            assert row.retained_outside_households_mxn > 0
        else:
            assert row.retained_outside_households_mxn == 0


@pytest.mark.parametrize("preference", ["fixed_baskets", "cobb_douglas"])
def test_decile_ev_matches_independent_closed_form_and_targeting(study, preference):
    output, run = study
    data = load_enigh2024_deciles()
    sectors = list(data.attrs["categories"])
    budget = data[sectors].sum(axis=1).to_numpy()
    shares = data[sectors].to_numpy() / budget[:, None]
    prices = pd.read_csv(output / "purchaser_prices.csv", index_col=0)
    ratio = (prices.counterfactual / prices.baseline).to_numpy()
    index = shares @ ratio if preference == "fixed_baskets" else np.exp(shares @ np.log(ratio))
    pool = run["manifest"]["fiscal"]["incremental_tariff_revenue_mxn"]
    equal = pool / data.households.sum()
    for rule in ("no_household_rebate", "equal_per_household", "bottom_four_deciles"):
        result = run["results"][f"{preference}_{rule}"]
        transfer = (np.zeros(10) if rule == "no_household_rebate" else
                    np.full(10, equal) if rule == "equal_per_household" else
                    np.r_[np.full(4, pool / data.households.iloc[:4].sum()), np.zeros(6)])
        assert_allclose(result.groups.transfer, transfer, rtol=1e-12, atol=1e-10)
        expected_ev = (budget + result.groups.income_change.to_numpy() + transfer) / index - budget
        assert_allclose(result.groups.ev, expected_ev, rtol=1e-11, atol=1e-9)
        # rtol 1e-11 was missed by 1.4e-12 on Windows CI (a -1.9e7 sum over ten deciles).
        assert_allclose(result.aggregate.total_ev, data.households @ expected_ev, rtol=1e-10)
        assert_allclose(result.groups[["price_ev", "income_ev", "transfer_ev"]].sum(axis=1),
                        result.groups.ev, rtol=1e-11, atol=1e-10)
        assert result.metadata["is_regression_fixture"] is True


def test_zero_tariff_recovers_zero_incidence_and_zero_fiscal_pool(tmp_path):
    run = application.run_application(tmp_path, tariff_rate=0.)
    assert run["manifest"]["fiscal"]["incremental_tariff_revenue_mxn"] == 0
    for result in run["results"].values():
        assert_allclose(result.groups[["ev", "cv", "income_change", "transfer"]], 0., atol=1e-8)


def test_exports_authenticate_scope_observations_and_physical_evidence(study):
    output, run = study
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["is_synthetic"] and manifest["is_regression_fixture"]
    assert manifest["observed_data"]["is_synthetic"] is False
    assert manifest["model_assumptions"]["empirically_calibrated_trade_economy"] is False
    assert "unobserved proxy" in manifest["income_mapping"]["other_factor_proxy"]
    assert "not a separately solved" in manifest["fiscal"]["no_household_rebate"]
    for filename, evidence in manifest["artifacts"].items():
        content = (output / filename).read_bytes()
        assert hashlib.sha256(content).hexdigest() == evidence["sha256"]
        assert len(content) == evidence["bytes"]
    with np.load(output / "equilibrium_evidence.npz", allow_pickle=False) as arrays:
        assert_allclose(arrays["counterfactual_intermediate_quantities"], run["counterfactual"].intermediate_flows)
        assert_allclose(arrays["calibration_table"], run["calibration"].data_calibra, atol=2e-13)
    assert len(pd.read_csv(output / "all_scenarios.csv")) == 60


@pytest.mark.parametrize("rate", [-.1, np.nan, np.inf])
def test_invalid_tariffs_are_refused_before_artifacts(tmp_path, rate):
    output = tmp_path / "refused"
    with pytest.raises(ValueError, match="finite and nonnegative"):
        application.run_application(output, tariff_rate=rate)
    assert not output.exists()


def test_unconverged_result_cannot_reach_distributional_reporting(tmp_path, monkeypatch):
    solve = application.solve_trade_equilibrium
    monkeypatch.setattr(application, "solve_trade_equilibrium",
                        lambda *args, **kwargs: replace(solve(*args, **kwargs), converged=False))
    with pytest.raises(RuntimeError, match="Both GE states must converge"):
        application.run_application(tmp_path)
    assert not (tmp_path / "manifest.json").exists()
