import json

import numpy as np
import pandas as pd
import pytest

from puremacro.asset_pricing import backtest_portfolio, replay_backtest, walk_forward_weights


def frame(values):
    a = np.asarray(values, dtype=float)
    return pd.DataFrame(a, index=pd.date_range("2020-01-01", periods=len(a)),
                        columns=[f"a{i}" for i in range(a.shape[1])])


def test_cash_costs_and_sale_match_hand_accounting():
    r = frame([[0], [.1], [.5], [0]])
    w = frame([[1], [0], [np.nan], [np.nan]])
    out = backtest_portfolio(r, w, execution_lag=1, cost_bps=100, check_scale=False)
    np.testing.assert_allclose(out.ledger.wealth, [1, 1.1 / 1.01, 1.1 / 1.01 * .99, 1.1 / 1.01 * .99])
    np.testing.assert_allclose(out.ledger.cost, [0, .01/1.01, .01, 0])
    assert out.ledger.accounting_residual.max() < 1e-14


def test_no_trade_drifts_weights_and_default_delay_is_next_close():
    r = frame([[0, 0], [.2, 0], [.2, 0], [.2, 0]])
    w = frame([[.5, .5], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]])
    out = backtest_portfolio(r, w)
    np.testing.assert_allclose(out.ledger.wealth, [1, 1, 1.1, 1.22])
    np.testing.assert_allclose(out.weights.iloc[-1], [6/11, 5/11])
    assert out.ledger.turnover.iloc[-1] == 0


def test_walk_forward_future_perturbation_and_replay():
    r = frame(np.random.default_rng(2).normal(0, .01, (40, 3)))
    seen = []
    def policy(h):
        seen.append(h.index[-1])
        vals = (h.mean() > 0).astype(float)
        return vals / max(1, vals.sum())
    w = walk_forward_weights(r, policy, min_history=5, window=10, rebalance_every=2)
    altered = r.copy()
    altered.iloc[25:] *= -5
    w2 = walk_forward_weights(altered, policy, min_history=5, window=10, rebalance_every=2)
    pd.testing.assert_frame_equal(w.iloc[:25], w2.iloc[:25])
    a = backtest_portfolio(r, w, cost_bps=12)
    b = backtest_portfolio(altered, w2, cost_bps=12)
    pd.testing.assert_frame_equal(a.ledger.iloc[:25], b.ledger.iloc[:25])
    replay = replay_backtest(a.to_json())
    np.testing.assert_allclose(a.ledger, replay.ledger, atol=0, rtol=0)
    assert a.input_sha256 == replay.input_sha256
    tampered = json.loads(a.to_json())
    tampered['recipe']['cost_bps'] = 0
    with pytest.raises(ValueError, match='checksum'):
        replay_backtest(json.dumps(tampered))


def test_missing_unheld_asset_and_explicit_universe():
    r = frame([[0, np.nan], [.1, np.nan], [0, .02]])
    w = frame([[1, 0], [np.nan, np.nan], [np.nan, np.nan]])
    eligible = r.notna()
    assert backtest_portfolio(r, w, execution_lag=1, eligibility=eligible).ledger.wealth.iloc[-1] == 1.1
    w.iloc[0] = [.5, .5]
    with pytest.raises(ValueError, match='ineligible'):
        backtest_portfolio(r, w, execution_lag=1, eligibility=eligible)
    with pytest.raises(ValueError, match='missing return'):
        backtest_portfolio(r, w, execution_lag=1)


@pytest.mark.parametrize('bad', [[-.1, .5], [.7, .7], [np.nan, .5], [np.inf, 0]])
def test_invalid_weights_rejected(bad):
    r = frame([[0, 0], [.01, .01]])
    w = frame([bad, [np.nan, np.nan]])
    with pytest.raises(ValueError):
        backtest_portfolio(r, w)


def test_bad_dates_lag_and_bankruptcy():
    r = frame([[0], [-1]])
    w = frame([[1], [np.nan]])
    with pytest.raises(ValueError, match='exhausted'):
        backtest_portfolio(r, w, execution_lag=1, check_scale=False)
    with pytest.raises(ValueError, match='execution_lag'):
        backtest_portfolio(r, w, execution_lag=0)
    with pytest.raises(ValueError, match='increasing'):
        backtest_portfolio(r.iloc[::-1], w)


def test_cash_income_and_report_exports():
    r = frame([[0], [.1], [-.1]])
    w = frame([[0], [np.nan], [np.nan]])
    out = backtest_portfolio(r, w, cash_returns=.01)
    np.testing.assert_allclose(out.ledger.wealth, 1.01 ** np.arange(1, 4))
    assert out.to_markdown() and out.to_latex() and out.to_typst() and out.summary()
    assert out.plot() is not None
