import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from puremacro.asset_pricing import (fundamentals_asof, fundamental_ratios, discounted_cash_flow,
                                     dividend_discount, valuation_sensitivity)


def facts():
    return pd.DataFrame([
        ['a','earnings','USD','annual','2019-01-01','2019-12-31','2020-03-01',10,'first'],
        ['a','earnings','USD','annual','2019-01-01','2019-12-31','2021-03-01',8,'restated'],
        ['a','earnings','USD','annual','2020-01-01','2020-12-31','2021-03-01',12,'new'],
    ],columns=['asset','metric','unit','period_type','period_start','period_end','available_at','value','source'])


def test_availability_and_restatement_no_lookahead():
    data=facts()
    assert fundamentals_asof(data,'2020-02-29').empty
    early=fundamentals_asof(data,'2020-04-01')
    assert early.value.tolist()==[10] and early.source.tolist()==['first']
    assert fundamentals_asof(data,'2021-04-01').value.tolist()==[12]
    # A later restatement of an old year never displaces the latest year.
    data.loc[1,'available_at']='2022-03-01'
    assert fundamentals_asof(data,'2022-04-01').value.tolist()==[12]
    pd.testing.assert_frame_equal(early,fundamentals_asof(data,'2020-04-01'))


def test_duplicate_columns_and_ratio_overflow_rejected():
    data = facts()
    with pytest.raises(ValueError, match='unique column'):
        fundamentals_asof(pd.concat([data, data[['value']]], axis=1), '2020-04-01')
    with pytest.raises(ValueError, match='overflow'):
        fundamental_ratios(price=1e200, shares_outstanding=1e200, revenue=1,
            net_income=1, book_equity=1, debt=0, cash=0,
            operating_cash_flow=0, capital_expenditure=0)


def test_period_types_and_currencies_remain_separate():
    data=facts().iloc[[0]].copy()
    q=data.copy();q['period_type']='quarterly';q['period_start']='2019-10-01';q['value']=3
    other=data.copy();other['unit']='EUR';other['value']=9
    assert len(fundamentals_asof(pd.concat([data,q,other]),'2020-04-01'))==3
    with pytest.raises(ValueError,match='duplicate'):
        fundamentals_asof(pd.concat([data,data]),'2020-04-01')
    invalid=data.copy();invalid['available_at']='2019-01-01'
    with pytest.raises(ValueError,match='interval'):
        fundamentals_asof(invalid,'2020-04-01')


def test_cashflow_present_value_and_enterprise_equity_bridge():
    value=discounted_cash_flow([100,110],.1,terminal_growth=.02,net_debt=200,shares_outstanding=10)
    pv=100/1.1+110/1.1**2
    terminal=(110*1.02/.08)/1.1**2
    assert_allclose([value.forecast_pv,value.terminal_pv,value.enterprise_value,value.equity_value,value.value_per_share],
                    [pv,terminal,pv+terminal,pv+terminal-200,(pv+terminal-200)/10])
    assert value.to_markdown() and value.to_latex() and value.to_typst()
    equity=discounted_cash_flow([100,110],[.1,.2],kind='equity')
    assert equity.equity_value==pytest.approx(100/1.1+110/(1.1*1.2))
    assert np.isnan(equity.enterprise_value)


def test_gordon_dividend_and_terminal_sensitivity():
    value=dividend_discount(2,.1,.03)
    assert value.value_per_share==pytest.approx(2/.07)
    grid=valuation_sensitivity([100,110],[.08,.12],[.02,.1],net_debt=20,shares_outstanding=10)
    assert grid.status.tolist()==['ok','invalid_terminal_growth','ok','ok']
    assert np.isnan(grid.value_per_share.iloc[1])
    assert grid.value_per_share.iloc[0]>grid.value_per_share.iloc[2]
    assert grid.value_per_share.iloc[3]>grid.value_per_share.iloc[2]


def test_ratios_invalid_denominators_and_cashflow_signs():
    out=fundamental_ratios(price=10,shares_outstanding=100,revenue=500,net_income=-10,
        book_equity=200,debt=100,cash=20,operating_cash_flow=40,capital_expenditure=60).set_index('metric')
    assert out.loc['enterprise_value','value']==1080
    assert np.isnan(out.loc['price_earnings','value'])
    assert out.loc['price_earnings','status']=='nonpositive_denominator'
    assert out.loc['ev_ebitda','status']=='missing_denominator'
    assert out.loc['free_cash_flow_yield','value']==pytest.approx(-.02)
    assert out.loc['profit_margin','value']==pytest.approx(-.02)


@pytest.mark.parametrize('kwargs', [dict(terminal_growth=.1),dict(kind='equity',net_debt=10),
                                    dict(shares_outstanding=0),dict(kind='unknown')])
def test_invalid_valuation_assumptions_rejected(kwargs):
    with pytest.raises(ValueError):
        discounted_cash_flow([100,110],.1,**kwargs)


def test_negative_equity_remains_visible():
    assert discounted_cash_flow([1],.1,net_debt=100).equity_value<0
    with pytest.raises(ValueError):
        dividend_discount(-1,.1)
