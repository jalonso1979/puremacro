"""Production timing, full household budgets, and capital-market equations."""
import numpy as np
import pytest

from puremacro.models.hank_capital import (CapitalHANKSteadyState, _CapitalKernel,
    calibrate_capital_hank, capital_production_block)
from puremacro.models.hank_recursive import _make_basis


@pytest.fixture(scope="module")
def capital():
    return calibrate_capital_hank(n_a=10, n_b=10, a_max=100., b_max=15.)


def test_steady_capital_household_and_aggregate_accounts(capital):
    assert isinstance(capital, CapitalHANKSteadyState) and capital.converged
    p, a, hh = capital.calibration, capital.aggregates, capital._household
    assert a["A"] == pytest.approx(a["K"], abs=1e-10)
    assert a["I"] == pytest.approx(p["delta"]*a["K"])
    assert a["Y"] == pytest.approx(p["Z"]*a["K"]**p["alpha"]*a["N"]**(1-p["alpha"]))
    assert p["r_a"]+p["delta"] == pytest.approx(p["alpha"]*a["Y"]/p["mu"]/a["K"], abs=1e-9)
    assert a["Y"] == pytest.approx(a["C"]+a["CHI"]+a["I"]+a["G"], abs=1e-9)
    assert a["Tax"] == pytest.approx(p["r"]*a["B"]+a["G"])
    assert a["w"]*a["N"]+a["Div"]-a["Tax"] == pytest.approx(p["income"], abs=1e-9)
    kernel = _CapitalKernel(capital)
    prices = kernel.prices_from_state(np.zeros(4), np.zeros(3), lag_q=1.)
    household = kernel.backward(kernel.m, prices)
    residual, report = kernel.markets(kernel.d, household, prices, 0.)
    np.testing.assert_allclose(residual, 0., atol=2e-8)
    np.testing.assert_allclose(kernel.forward(kernel.d, household), kernel.d, atol=2e-9)
    for name in ("goods_mkt", "fiscal_res", "capital_law_res", "investment_foc_res",
                 "return_res", "resource_accounting_res"):
        assert abs(report[name]) < 2e-8, name
    checks = kernel.household_checks(kernel.d, household, prices, kernel.m)
    assert checks["budget_error"] < 1e-10
    assert checks["minimum_consumption"] > 0 and checks["grid_exit_mass"] == 0
    # Euler interpolation error and finite-grid mass are disclosed separately;
    # stationary policy convergence alone does not certify their accuracy.
    assert checks["euler_kkt_max"] >= checks["euler_kkt_mean"] >= 0
    assert capital.diagnostics["upper_illiquid_mass"] == hh.mass_at_a_max


def test_physical_parameters_stay_fixed_when_grid_changes(capital):
    refined = calibrate_capital_hank(n_a=15, n_b=15)
    for name in ("alpha", "delta", "epsI", "mu", "r", "r_a", "government_share"):
        assert refined.calibration[name] == capital.calibration[name]
    assert refined.calibration["beta"] != capital.calibration["beta"]
    assert refined.diagnostics["calibration_unknown"] == "beta"
    assert refined.diagnostics["return_error"] < 1e-8
    assert refined.diagnostics["goods_market_error"] < 1e-8


@pytest.mark.parametrize("growth", [-.01, 0., .015])
def test_installation_return_is_capital_value_derivative(growth):
    """Check the payoff against the envelope derivative, independently of q."""
    K, delta, epsI = 10.2, .02, 4.
    psi = 1/(delta*epsI)
    q = 1+psi*growth
    block = capital_production_block(Y=1.03, w=.64, K=K, q=q, lag_q=.99,
                                    Z=.47, delta=delta, epsI=epsI, pi=.002)
    I = (delta+growth)*K

    def installation_cost(stock, investment):
        # Direct technology in investment and inherited capital coordinates.
        return .5*psi*(investment/stock-delta)**2*stock

    eps = 1e-5
    dPhi_dI = (installation_cost(K, I+eps)-installation_cost(K, I-eps))/(2*eps)
    dPhi_dK = (installation_cost(K+eps, I)-installation_cost(K-eps, I))/(2*eps)
    assert q == pytest.approx(1+dPhi_dI, abs=1e-10)
    envelope = block["rental"]+(1-delta)*q-dPhi_dK
    assert (1+block["r_a"])*.99 == pytest.approx(envelope, abs=1e-10)
    assert block["installation_rent"] == pytest.approx(-K*dPhi_dK, abs=1e-9)
    assert block["Knext"] == pytest.approx(K*(1+growth))
    assert block["investment_goods"] == pytest.approx(I+installation_cost(K, I))
    assert block["div"]+block["rental"]*K+block["adj"]+.64*block["N"] == pytest.approx(1.03)


def test_off_equilibrium_accounts_preserve_actual_inherited_price(capital):
    kernel = _CapitalKernel(capital)
    state, controls = np.array([.001, -.0002, .01]), np.array([.0001, .002, .005, .003])
    quoted_q = 1.013
    prices = kernel.prices_from_state(controls, state, kernel.d, lag_q=quoted_q)
    inferred = kernel.prices_from_state(controls, state, kernel.d)
    assert prices["lag_q"] == quoted_q
    assert inferred["lag_q"] != pytest.approx(quoted_q)
    assert (1+prices["r_a"])*quoted_q == pytest.approx((1+inferred["r_a"])*inferred["lag_q"])
    household = kernel.backward(kernel.m, prices)
    _, report = kernel.markets(kernel.d, household, prices, 0.)
    assert abs(report["resource_accounting_res"]) < 1e-10
    assert abs(report["inherited_capital_mkt"]) > .1
    assert abs(report["goods_mkt"]) > 1e-3  # An audit must retain this failure.
    assert abs(report["fiscal_res"]) < 1e-12
    assert abs(report["return_res"]) < 1e-12
    checks = kernel.household_checks(kernel.d, household, prices, kernel.m)
    assert checks["budget_error"] < 1e-10
    nxt = kernel.next_aggregate_state(prices, state, .7)
    assert kernel.p["K"]*np.exp(nxt[2]) == pytest.approx(prices["Knext"])
    assert nxt[0] == prices["nominal"]-kernel.r and nxt[1] == .7*state[1]


def _nonlinear_period(kernel, basis, cur, nxt, rho):
    k, nm = basis.rank, len(kernel.m)
    ns = k+3
    distribution = basis.decode(cur[:k], check_positive=False)
    prices = kernel.prices_from_state(cur[-4:], cur[k:ns], distribution)
    household = kernel.backward(kernel.m+nxt[ns:ns+nm], prices)
    transported = kernel.forward(distribution, household)
    next_aggregate = kernel.next_aggregate_state(prices, cur[k:ns], rho)
    future_output = kernel.Y*np.exp(nxt[-2])
    future_gross = (1+prices["nominal"])/np.exp(nxt[-4])
    continuation = future_output/prices["Y"]*nxt[-4]/future_gross
    markets, _ = kernel.markets(distribution, household, prices, continuation)
    return np.r_[nxt[:k]-basis.project(transported-basis.center),
        nxt[k:ns]-next_aggregate, kernel.m+cur[ns:ns+nm]-household["m"], markets]


def test_capital_joint_derivative_against_nonlinear_levels(capital):
    kernel = _CapitalKernel(capital)
    basis = _make_basis(kernel, 12)
    dimension = basis.rank+3+len(kernel.m)+4
    rng = np.random.default_rng(27912)
    for _ in range(3):
        cur, nxt = rng.normal(size=(2, dimension))*.01
        eps = 2e-6
        numeric = (_nonlinear_period(kernel, basis, eps*cur, eps*nxt, .7)
                   -_nonlinear_period(kernel, basis, -eps*cur, -eps*nxt, .7))/(2*eps)
        tangent = kernel.linear_residual(cur, nxt, basis, .7)
        np.testing.assert_allclose(tangent, numeric, atol=2e-5, rtol=3e-4)
    assert kernel._derivatives is None


@pytest.mark.parametrize("change", [dict(delta=0.), dict(epsI=-1.), dict(alpha=1.),
                                     dict(beta_bracket=(.99, .98)), dict(government_share=.9)])
def test_invalid_calibration_is_rejected(change):
    with pytest.raises(ValueError):
        calibrate_capital_hank(**change)


def test_missing_capital_state_and_invalid_price_domain_are_rejected(capital):
    kernel = _CapitalKernel(capital)
    with pytest.raises(ValueError, match="inherited log_capital"):
        kernel.prices(np.zeros(4), 0., 0.)
    with pytest.raises(ValueError, match="three aggregate states"):
        kernel.prices_from_state(np.zeros(4), np.zeros(2))
    with pytest.raises(ValueError, match="positive-price domain"):
        capital_production_block(Y=1, w=1, K=1, q=1, lag_q=0., Z=1)
