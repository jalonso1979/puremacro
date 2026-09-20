"""Independent economic-accounting checks for the opt-in coherent CGE model."""
from dataclasses import replace
import numpy as np
import pytest

from puremacro.trade import calibrate_trade_model, solve_trade_equilibrium
from puremacro.trade.equilibrium import compute_equilibrium_residuals, pack_equilibrium_vector
from puremacro.trade.postprocessing import postprocess_trade_equilibrium
from puremacro.trade._accounting import evaluate
from tools.reference_validation.validate_trade_accounting import benchmark_table, scalar_reference, validate
from tools.reference_validation.validate_oecd import load_fixture, validate_aggregate


@pytest.fixture(scope="module")
def model():
    data, tau, tau_fd = benchmark_table()
    calib = calibrate_trade_model(data, ns=1, nc=2, nfd=2, country_codes=["A", "B"], sector_codes=["GOOD"])
    return data, calib, tau, tau_fd


@pytest.fixture(scope="module")
def solution(model):
    _, calib, tau, fd = model
    result = solve_trade_equilibrium(calib, tau=tau, tau_fd=fd, accounting="consistent", tol=1e-12, max_iter=100)
    assert result.converged
    return result


def test_independent_scalar_equilibrium():
    result = validate()
    assert result["passed"], result


@pytest.mark.parametrize("method", ["hybr", "sparse_lu", "keller_pac", "condensed", "broyden", "krylov", "lm"])
def test_solver_methods_use_same_price_and_closure_equations(model, method):
    data, calib, tau, fd = model
    ref = scalar_reference(data, tau, fd)
    result = solve_trade_equilibrium(calib, tau=tau, tau_fd=fd, accounting="consistent",
                                     method=method, tol=1e-9, max_iter=100, max_steps=20)
    assert result.converged, (method, result.max_residual, result.metadata["account_residuals"])
    np.testing.assert_allclose(result.p_sol.ravel(), ref["p"], atol=1e-8)
    np.testing.assert_allclose(result.tariffs, ref["tariffs"], atol=1e-8)
    assert result.metadata["replicate_matlab_precedence"] is False
    assert result.metadata["tariff_revenue_mode"] == "schedule"
    if method == "condensed":
        assert result.metadata["effective_method"] == "newton"


def test_producer_value_plus_duties_and_local_taxes_equals_buyer_spending(model, solution):
    data, calib, tau, fd = model
    r = solution
    p = r.p_sol.ravel()
    Z = r.intermediate_flows.reshape(2, 2, order="F")
    F = r.final_demand_flows.reshape(2, 4, order="F")
    pi, pf = Z*p[:, None], F*p[:, None]
    duties_i = pi*(tau.reshape(2, 2, order="F")-1)
    duties_f = pf*(fd.reshape(2, 4, order="F")-1)
    duties = duties_i.sum(0)+duties_f.sum(0).reshape(2, 2).sum(1)
    np.testing.assert_allclose(r.tariffs, duties, atol=1e-11)
    # Composite final prices must never replace exporter prices in duty values.
    wrong = F*r.p_fd.ravel(order="F")[None]*(fd.reshape(2, 4, order="F")-1)
    assert np.max(np.abs(wrong-duties_f)) > .01
    local_fd = r.metadata["final_tax_receipts"].ravel(order="F")
    spending = r.metadata["final_expenditure"].ravel(order="F")
    np.testing.assert_allclose((pf+duties_f).sum(0)+local_fd, spending, atol=1e-11)
    np.testing.assert_allclose(r.metadata["purchaser_final_values"].sum(0), spending, atol=1e-11)
    # Price times output is the production tax base, not output quantities.
    prod = (calib.tax*r.p_sol*r.y_sol).ravel()
    np.testing.assert_allclose(r.T_sol.ravel(), prod+local_fd.reshape(2, 2).sum(1)+duties, atol=1e-10)
    assert np.max(abs(prod-(calib.tax*r.y_sol).ravel())) > .01
    np.testing.assert_allclose(r.data_tariff_vf[:, :2].sum(0), p*r.y_sol.ravel(), atol=1e-10)
    np.testing.assert_allclose(r.data_tariff_vf[:, 2:].sum(0), spending, atol=1e-10)
    np.testing.assert_allclose(r.metadata["final_tax_share"].ravel(order="F"), data[2, 2:]/(data[:2, 2:].sum(0)+data[2, 2:]))


def test_income_expenditure_gdp_trade_and_foreign_saving_agree(model, solution):
    _, calib, _, _ = model
    r = solution
    spending = r.metadata["final_expenditure"].sum(axis=1).ravel()
    np.testing.assert_allclose(r.gdp, spending+r.exports-r.imports, atol=1e-10)
    np.testing.assert_allclose(r.exports-r.imports, calib.invforT.ravel(), atol=1e-10)
    np.testing.assert_allclose(r.T_sol.ravel(), r.tariffs+r.metadata["production_tax_receipts"].sum(1).ravel()+r.metadata["final_tax_receipts"].sum(1).ravel(), atol=1e-10)
    assert max(r.metadata["account_residuals"].values()) < 1e-10
    assert r.p_sol.ravel()[0] == pytest.approx(1.)


@pytest.mark.parametrize("sigma", [0., 1., 2.])
def test_nominal_homogeneity_of_costs_budgets_and_quantities(model, solution, sigma):
    _, calib, tau, fd = model
    r = solution
    base = evaluate(r.x_sol, calib, tau, fd, None, None, sigma=sigma)
    scale = 3.7
    xx = pack_equilibrium_vector(r.p_sol*scale, r.y_sol, r.r_sol*scale,
                                 r.w_sol*scale, r.T_sol*scale, r.XN_sol*scale)
    scaled = evaluate(xx, calib, tau, fd, None, None, sigma=sigma)
    for field in ("p", "pp", "Q", "P", "income", "expenditure", "tariffs", "government"):
        np.testing.assert_allclose(scaled[field], base[field]*scale, atol=1e-10)
    for field in ("Z", "F", "c", "labor", "capital"):
        np.testing.assert_allclose(scaled[field], base[field], atol=1e-10)


def test_fixed_closure_and_numeraire_remove_jacobian_indeterminacy(model, solution):
    _, calib, tau, fd = model
    x = solution.x_sol
    jac = np.empty((len(x), len(x)))
    for j in range(len(x)):
        dx = np.eye(len(x))[j]*1e-5
        up = compute_equilibrium_residuals(x+dx, calib, tau=tau, tau_fd=fd, accounting="consistent")
        dn = compute_equilibrium_residuals(x-dx, calib, tau=tau, tau_fd=fd, accounting="consistent")
        jac[:, j] = (up-dn)/2e-5
    assert np.linalg.matrix_rank(jac, tol=1e-7) == len(x)


def test_unbalanced_returned_state_cannot_be_reported_as_converged(model, solution):
    _, calib, tau, fd = model
    bad = solution.x_sol.copy(); bad[2] += .01
    r = postprocess_trade_equilibrium(bad, calib, tau=tau, tau_fd=fd, accounting="consistent", converged=True)
    assert not r.converged
    assert r.metadata["account_residuals"]["goods"] > .1


def test_cpi_uses_fixed_purchaser_basket(model, solution):
    data, calib, _, _ = model
    basket = data[:2, 2:].sum(0).reshape(2, 2).T[None]
    P0 = (data[:2, 2:].sum(0)+data[2, 2:])/data[:2, 2:].sum(0)
    P0 = P0.reshape(2, 2).T[None]
    nominal = (solution.Pfd_final*basket).sum((0, 1))/(P0*basket).sum((0, 1))
    np.testing.assert_allclose(solution.cpi, nominal/solution.w_sol.ravel())


@pytest.mark.parametrize("sigma", [0., 1., 2.])
def test_single_country_single_final_use_and_zero_intermediates(sigma):
    data = np.array([[0., 100.], [5., 2.], [60., 0.], [35., 0.]])
    calib = calibrate_trade_model(data, ns=1, nc=1, nfd=1)
    r = solve_trade_equilibrium(calib, accounting="consistent", sigma=sigma, tol=1e-10)
    assert r.converged
    np.testing.assert_allclose(r.gdp, [102.])
    np.testing.assert_allclose(r.Pfd_final, [[[1.02]]])
    np.testing.assert_allclose(r.cpi, [1.])


@pytest.mark.parametrize("options", [{"fiscal_closure": "deficit_reduction"}, {"capacity_margins": {"GOOD": .1}}, {"backend": "mlx"}, {"method": "quasi_condensed"}])
def test_unsupported_extensions_fail_explicitly(model, options):
    _, c, _, _ = model
    with pytest.raises(NotImplementedError):
        solve_trade_equilibrium(c, accounting="consistent", **options)


@pytest.mark.parametrize("corruption", ["domestic", "negative_tariff", "signed_final_use", "negative_theta"])
def test_invalid_economic_inputs_are_rejected(model, corruption):
    data, calib, tau, fd = model
    tau = tau.copy()
    if corruption == "domestic":
        tau[0, 0, 0] = 1.1
    elif corruption == "negative_tariff":
        tau[1, 0, 0] = 0.
    elif corruption == "signed_final_use":
        bad = data.copy(); bad[:2, 2] = -1
        calib = replace(calib, data_calibra=bad)
    else:
        theta = calib.theta.copy(); theta[0, 0, 0] = -.01
        calib = replace(calib, theta=theta)
    with pytest.raises(ValueError):
        solve_trade_equilibrium(calib, tau=tau, tau_fd=fd, accounting="consistent")


def test_consistent_oecd_empirical_aggregation():
    r = validate_aggregate(load_fixture(), accounting="consistent")
    assert r["passed"], r
    assert max(r["economic_accounts"].values()) < 1e-5


@pytest.mark.parametrize("omit_transfer", [False, True])
def test_calibration_without_source_table_preserves_actual_tax_base(model, solution, omit_transfer):
    _, calib, tau, fd = model
    reduced = replace(calib, data_calibra=None, T=None if omit_transfer else calib.T)
    result = solve_trade_equilibrium(reduced, tau=tau, tau_fd=fd,
                                     accounting="consistent", tol=1e-10)
    assert result.converged
    np.testing.assert_allclose(result.p_sol, solution.p_sol, atol=1e-10)
    np.testing.assert_allclose(result.tariffs, solution.tariffs, atol=1e-10)
    np.testing.assert_allclose(result.metadata["final_tax_share"],
                               solution.metadata["final_tax_share"], atol=1e-14)


@pytest.mark.parametrize("sigma", [.5, 1., 2.])
def test_ces_counterfactual_substitution_and_national_accounts(model, sigma):
    _, calib, tau, fd = model
    result = solve_trade_equilibrium(calib, tau=tau, tau_fd=fd, sigma=sigma,
                                     accounting="consistent", tol=1e-12)
    assert result.converged
    assert max(result.metadata["account_residuals"].values()) < 1e-9
    # The CES first-order condition changes the import/domestic quantity ratio
    # by the purchaser-price ratio raised to minus the substitution elasticity.
    Z = result.intermediate_flows.reshape(2, 2, order="F")
    purchaser = result.p_sol.ravel()[:, None]*tau[:, 0, :]
    expected_ratio = (calib.a[1, 0]/calib.a[0, 0])*(purchaser[1]/purchaser[0])**(-sigma)
    np.testing.assert_allclose(Z[1]/Z[0], expected_ratio, atol=1e-12)
    expenditure = result.metadata["purchaser_final_values"].sum(0).reshape(2, 2).sum(1)
    np.testing.assert_allclose(result.gdp, expenditure+result.exports-result.imports, atol=1e-9)
