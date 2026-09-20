"""Economic and failure-contract tests for expenditure-function trade welfare."""
from dataclasses import replace

import numpy as np
import pytest

from puremacro.trade import calibrate_trade_model, solve_trade_equilibrium, compute_hicksian_welfare
from puremacro.trade.data import package_mrio_to_calibration_result
from puremacro.trade._oecd_icio import condense_final_demand
from tools.reference_validation.validate_oecd import load_fixture
from tools.reference_validation.validate_trade_welfare import consumption_benchmark, validate


@pytest.fixture(scope="module")
def model():
    data, tau, fd = consumption_benchmark()
    calib = calibrate_trade_model(data, ns=1, nc=2, nfd=3, country_codes=["A", "B"])
    base = solve_trade_equilibrium(calib, accounting="consistent", tol=1e-10)
    cf = solve_trade_equilibrium(calib, tau=tau, tau_fd=fd, accounting="consistent", tol=1e-10)
    assert base.converged and cf.converged
    return data, calib, tau, fd, base, cf


def test_primal_expenditure_and_six_order_shapley_oracles():
    result = validate()
    assert result["passed"], result


@pytest.mark.parametrize("country", [0, 1])
def test_duality_ev_cv_and_fiscal_accounting(model, country):
    _, c, _, _, b, f = model
    r = compute_hicksian_welfare(c, f, base_result=b, target_country=country, consumption_categories=(0, 2))
    weights = c.theta[0, [0, 2], country]; weights = weights/weights.sum()
    q0, q1 = b.c_fd[0, [0, 2], country], f.c_fd[0, [0, 2], country]
    p0, p1 = b.Pfd_final[0, [0, 2], country], f.Pfd_final[0, [0, 2], country]
    assert r.utility_ratio == pytest.approx(np.prod((q1/q0)**weights))
    assert r.ev == pytest.approx(p0@q0*(r.utility_ratio-1), abs=1e-11)
    assert r.cv == pytest.approx(p1@q1*(1-1/r.utility_ratio), abs=1e-11)
    assert r.cv == pytest.approx(r.price_index*r.ev, abs=1e-11)
    assert r.fiscal_transfer_effect == pytest.approx(r.tariff_rebate_effect+r.domestic_tax_rebate_effect, abs=1e-10)
    assert r.ev == pytest.approx(r.price_effect+r.factor_income_effect+r.fiscal_transfer_effect, abs=1e-10)
    assert r.ev_pct_consumption == pytest.approx(100*r.ev/(p0@q0))
    assert r.ev_pct_gdp == pytest.approx(100*r.ev/b.gdp[country])
    # Tariff receipts are already in transfers; adding them again breaks EV.
    assert abs(r.tariff_rebate_effect) > .01
    assert not np.isclose(r.ev, r.price_effect+r.factor_income_effect+r.fiscal_transfer_effect+r.tariff_rebate_effect)


def test_default_consumption_is_actual_category_zero_not_investment(model):
    _, c, _, _, b, f = model
    r = compute_hicksian_welfare(c, f, base_result=b)
    assert r.ev == pytest.approx(float(b.Pfd_final[0, 0, 0]*(f.c_fd[0, 0, 0]-b.c_fd[0, 0, 0])), abs=1e-11)
    assert r.consumption_categories == (0,)
    assert r.metadata["causal_decomposition"] is False


def test_identity_reverse_and_cpi_independence(model):
    _, c, _, _, b, f = model
    identity = compute_hicksian_welfare(c, b, base_result=b)
    assert identity.ev == identity.cv == identity.price_effect == identity.factor_income_effect == identity.fiscal_transfer_effect == 0.
    forward = compute_hicksian_welfare(c, f, base_result=b)
    reverse = compute_hicksian_welfare(c, b, base_result=f)
    assert reverse.ev == pytest.approx(-forward.cv, abs=1e-11)
    assert reverse.cv == pytest.approx(-forward.ev, abs=1e-11)
    changed_cpi = replace(f, cpi=np.full(c.nc, np.nan))
    assert compute_hicksian_welfare(c, changed_cpi, base_result=b).ev == forward.ev


@pytest.mark.parametrize("scale", [.001, 1000.])
def test_currency_scaling_changes_levels_and_preserves_percentages(model, scale):
    data, c, tau, fd, b, f = model
    reference = compute_hicksian_welfare(c, f, base_result=b, consumption_categories=(0, 2))
    c2 = calibrate_trade_model(data*scale, ns=1, nc=2, nfd=3, country_codes=["A", "B"])
    b2 = solve_trade_equilibrium(c2, accounting="consistent", tol=1e-10*scale)
    f2 = solve_trade_equilibrium(c2, tau=tau, tau_fd=fd, accounting="consistent", tol=1e-10*scale)
    r = compute_hicksian_welfare(c2, f2, base_result=b2, consumption_categories=(0, 2))
    for field in ("ev", "cv", "price_effect", "factor_income_effect", "fiscal_transfer_effect", "tariff_rebate_effect"):
        assert getattr(r, field) == pytest.approx(getattr(reference, field)*scale, rel=1e-8, abs=1e-9)
    assert r.ev_pct_consumption == pytest.approx(reference.ev_pct_consumption, abs=1e-9)
    assert r.ev_pct_gdp == pytest.approx(reference.ev_pct_gdp, abs=1e-9)


@pytest.mark.parametrize("shock", [1e-6, -.05, .05])
def test_small_shocks_and_import_subsidies(model, shock):
    _, c, _, _, b, _ = model
    f = solve_trade_equilibrium(c, tau=np.array([shock, 0.]), tau_fd=np.array([shock, 0.]),
                               accounting="consistent", tol=1e-10)
    r = compute_hicksian_welfare(c, f, base_result=b)
    assert np.isfinite(r.ev)
    assert abs(r.decomposition_residual) < 1e-10
    assert r.ev == pytest.approx(b.Pfd_final[0, 0, 0]*(f.c_fd[0, 0, 0]-b.c_fd[0, 0, 0]), abs=1e-11)


@pytest.mark.parametrize("corruption", ["nonconverged", "legacy", "missing_schedule", "quantity", "price", "income", "gdp", "state", "closure"])
def test_failed_incompatible_and_stale_states_are_rejected(model, corruption):
    _, c, _, _, b, f = model
    meta = dict(f.metadata)
    if corruption == "nonconverged":
        f = replace(f, converged=False)
    elif corruption in ("legacy", "missing_schedule", "closure"):
        if corruption == "legacy": meta["accounting"] = "legacy"
        elif corruption == "closure": meta["fiscal_closure"] = "deficit_reduction"
        else: meta.pop("final_tariff_multipliers")
        f = replace(f, metadata=meta)
    elif corruption == "quantity": f = replace(f, c_fd=f.c_fd*1.01)
    elif corruption == "price": f = replace(f, Pfd_final=f.Pfd_final*1.01)
    elif corruption == "income": f = replace(f, T_sol=f.T_sol+1)
    elif corruption == "gdp": f = replace(f, gdp=f.gdp+1)
    else:
        x = f.x_sol.copy(); x[2] += .01
        f = replace(f, x_sol=x)
    with pytest.raises((ValueError, NotImplementedError)):
        compute_hicksian_welfare(c, f, base_result=b)


@pytest.mark.parametrize("options", [{"consumption_categories": ()}, {"consumption_categories": (1,)},
    {"consumption_categories": (0, 0)}, {"consumption_categories": (0, 9)},
    {"consumption_categories": (0.0,)}, {"target_country": -1}, {"target_country": "MISSING"},
    {"tol": -1}, {"tol": np.nan}])
def test_invalid_welfare_specification_rejected(model, options):
    _, c, _, _, b, f = model
    with pytest.raises((ValueError, TypeError)):
        compute_hicksian_welfare(c, f, base_result=b, **options)


def test_single_country_single_basket(model):
    data = np.array([[0., 100.], [5., 2.], [60., 0.], [35., 0.]])
    c = calibrate_trade_model(data, ns=1, nc=1, nfd=1)
    b = solve_trade_equilibrium(c, accounting="consistent", tol=1e-10)
    r = compute_hicksian_welfare(c, b, base_result=b)
    assert r.ev == 0.
    assert r.consumption_base == pytest.approx(102.)


def test_formatting_and_explicit_units(model):
    _, c, _, _, b, f = model
    r = compute_hicksian_welfare(c, f, base_result=b, target_country="A")
    assert r.country_code == "A"
    assert "calibration value units" in r.summary()
    assert "Fiscal transfers" in r.to_markdown()
    assert "Fiscal transfers" in r.to_latex()
    assert "Fiscal transfers" in r.to_typst()


def test_real_oecd_aggregate_consumption_welfare():
    c = package_mrio_to_calibration_result(condense_final_demand(load_fixture()))
    b = solve_trade_equilibrium(c, accounting="consistent", tol=1e-5)
    cf = solve_trade_equilibrium(c, tau=np.array([.1, 0., 0.]), tau_fd=np.array([.1, 0., 0.]),
                                accounting="consistent", tol=1e-5)
    for i in range(c.nc):
        r = compute_hicksian_welfare(c, cf, base_result=b, target_country=i)
        expected = b.Pfd_final[0, 0, i]*(cf.c_fd[0, 0, i]-b.c_fd[0, 0, i])
        assert r.ev == pytest.approx(expected, abs=1e-7)
        assert abs(r.decomposition_residual) < 1e-6


@pytest.mark.parametrize("sigma", [.5, 1., 2.])
def test_ces_production_preserves_consumption_duality(model, sigma):
    _, c, tau, fd, _, _ = model
    b = solve_trade_equilibrium(c, accounting="consistent", sigma=sigma, tol=1e-10)
    # This strong sigma=.5 shock requires hybrid/continuation from the default
    # initial point. Welfare must not accept Newton's nonconverged state.
    method = "hybr" if sigma == .5 else "newton"
    f = solve_trade_equilibrium(c, tau=tau, tau_fd=fd, accounting="consistent", sigma=sigma, tol=1e-10, method=method)
    r = compute_hicksian_welfare(c, f, base_result=b, consumption_categories=(0, 2))
    assert abs(r.decomposition_residual) < 1e-9
    assert r.cv == pytest.approx(r.price_index*r.ev, abs=1e-9)


def test_zero_weight_basket_is_excluded_without_log_floors(model):
    data, _, tau, fd, _, _ = model
    data = data.copy()
    for c in range(2):
        data[:, 2+3*c] += data[:, 4+3*c]
        data[:, 4+3*c] = 0.
    calib = calibrate_trade_model(data, ns=1, nc=2, nfd=3)
    b = solve_trade_equilibrium(calib, accounting="consistent", tol=1e-10)
    f = solve_trade_equilibrium(calib, tau=tau, tau_fd=fd, accounting="consistent", tol=1e-10)
    active = compute_hicksian_welfare(calib, f, base_result=b)
    with_zero = compute_hicksian_welfare(calib, f, base_result=b, consumption_categories=(0, 2))
    assert with_zero.ev == active.ev


def test_real_failed_solve_cannot_produce_welfare(model):
    _, c, tau, fd, b, _ = model
    failed = solve_trade_equilibrium(c, tau=tau, tau_fd=fd, accounting="consistent", max_iter=0)
    assert not failed.converged
    with pytest.raises(ValueError, match="converged"):
        compute_hicksian_welfare(c, failed, base_result=b)
