"""Public research workflows: information timing and independent ledger checks."""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from puremacro import asset_pricing as ap


def sample(seed=11):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=150, freq="B")
    return pd.DataFrame(rng.normal(.0003, [.006, .012, .018], (150, 3)),
                        index=dates, columns=["a", "b", "c"])


def allocation(history):
    return ap.optimize_portfolio(ap.estimate_covariance(history), max_weight=.7).as_series()


@pytest.mark.pyodide_smoke
def test_allocation_execution_prefix_and_recipe():
    returns = sample()
    targets = ap.walk_forward_weights(returns, allocation, min_history=30,
                                     window=60, rebalance_every=10)
    first = ap.backtest_portfolio(returns, targets, cost_bps=20)
    changed = returns.copy()
    changed.iloc[90:] = changed.iloc[90:] * 1.7 - .001
    other_targets = ap.walk_forward_weights(changed, allocation, min_history=30,
                                           window=60, rebalance_every=10)
    other = ap.backtest_portfolio(changed, other_targets, cost_bps=20)
    pd.testing.assert_frame_equal(targets.iloc[:90], other_targets.iloc[:90])
    pd.testing.assert_frame_equal(first.ledger.iloc[:90], other.ledger.iloc[:90])
    # Start-of-interval holdings at 90 are unchanged; later drift may differ.
    pd.testing.assert_frame_equal(first.weights.iloc[:91], other.weights.iloc[:91])
    assert first.ledger.accounting_residual.max() < 2e-14
    assert first.weights.sum(axis=1).max() <= 1 + 1e-12
    pd.testing.assert_frame_equal(first.ledger, ap.replay_backtest(first.to_json()).ledger)


def test_recipe_preserves_timezone_across_daylight_saving_transition():
    dates = pd.date_range("2020-03-01", periods=15, tz="America/New_York")
    returns = pd.DataFrame({"a": np.full(15, .001)}, index=dates)
    targets = pd.DataFrame({"a": [1.]}, index=dates[:1])
    result = ap.backtest_portfolio(returns, targets)
    pd.testing.assert_frame_equal(result.ledger, ap.replay_backtest(result.to_json()).ledger)


def test_recipe_preserves_actual_dates_for_custom_holiday_calendar():
    dates = pd.bdate_range("2020-01-01", periods=10,
                          freq=pd.offsets.CustomBusinessDay(holidays=["2020-01-06"]))
    returns = pd.DataFrame({"a": .001}, index=dates)
    targets = pd.DataFrame({"a": [1.]}, index=dates[:1])
    result = ap.backtest_portfolio(returns, targets)
    replayed = ap.replay_backtest(result.to_json())
    pd.testing.assert_frame_equal(result.ledger, replayed.ledger, check_freq=False)
    assert replayed.ledger.index.freq is None  # No invented default holiday rule.
    assert replayed.input_sha256 == result.input_sha256


@pytest.mark.pyodide_smoke
def test_momentum_to_cash_and_independent_zero_cost_holdings_recursion():
    returns = sample()
    prices = 100 * (1 + returns).cumprod()
    targets = ap.walk_forward_weights(prices, lambda p: ap.momentum_weights(p, lookback=20, top_n=2),
                                     min_history=21, rebalance_every=7)
    result = ap.backtest_portfolio(returns, targets, cash_returns=.0001)
    # An independent dollar-holdings recursion; no implementation helpers used.
    dollars = np.zeros(3)
    cash = 1.
    expected = []
    for t, row in enumerate(returns.to_numpy()):
        decision = targets.iloc[t - 2] if t >= 2 else None
        if decision is not None and decision.notna().all():
            wealth = cash + sum(dollars)
            dollars = wealth * decision.to_numpy()
            cash = wealth * (1 - decision.sum())
        dollars = dollars * (1 + row)
        cash *= 1.0001
        expected.append(cash + sum(dollars))
    assert_allclose(result.ledger.wealth, expected, rtol=2e-15)
    risk = ap.risk_summary(result.ledger.net_return, periods_per_year=252)
    assert np.isfinite(risk.to_dataframe()["max_drawdown"]).all()


@pytest.mark.pyodide_smoke
def test_filing_revision_changes_only_later_valuation_inputs():
    facts = pd.DataFrame({
        "asset": ["a", "a"], "metric": ["free_cash_flow"] * 2, "unit": ["USD"] * 2,
        "period_type": ["annual"] * 2, "period_start": ["2019-01-01"] * 2,
        "period_end": ["2019-12-31"] * 2, "available_at": ["2020-03-01", "2021-03-01"],
        "value": [100., 80.], "source": ["filing-1", "revision-1"],
    })
    early = ap.fundamentals_asof(facts, "2020-04-01")
    later = ap.fundamentals_asof(facts, "2021-04-01")
    value_early = ap.discounted_cash_flow([early.value.iloc[0]], .1)
    value_later = ap.discounted_cash_flow([later.value.iloc[0]], .1)
    assert_allclose([value_early.enterprise_value, value_later.enterprise_value], [100 / 1.1, 80 / 1.1])
    assert early.source.iloc[0] == "filing-1"


@pytest.mark.pyodide_smoke
def test_belief_simulation_can_feed_explicit_positive_price_strategy():
    model = ap.simulate_brock_hommes(100, slopes=[0., .9], biases=[0., .2],
                                   noise_std=.02, seed=8, fundamental_value=10.)
    assert (model.prices > 0).all()
    prices = pd.DataFrame({"simulated": model.prices}, index=pd.date_range("2020", periods=100))
    returns = ap.to_returns(prices)
    targets = ap.walk_forward_weights(prices, lambda p: ap.momentum_weights(p, lookback=10),
                                     min_history=11, rebalance_every=5).reindex(returns.index)
    result = ap.backtest_portfolio(returns, targets, cost_bps=10)
    assert np.isfinite(result.ledger.wealth).all()
    assert result.ledger.accounting_residual.max() < 2e-14
