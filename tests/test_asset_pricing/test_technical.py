import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from puremacro.asset_pricing import technical_indicators, momentum_weights, compare_strategies


def price(values):
    return pd.DataFrame({'a':values},index=pd.date_range('2000',periods=len(values)))


def test_wilder_rsi_and_sma_seed_against_hand_values():
    p = price([10,11,10,12,11,13,14,13])
    a = technical_indicators(p,fast=2,slow=3,signal=2,rsi_period=3,lookback=3)['a']
    # First average gain=1, loss=1/3; next gain=2/3, loss=5/9.
    assert_allclose(a.rsi.iloc[3:5], [75, 100*6/11])
    assert a.rsi.iloc[:3].isna().all()
    assert_allclose(a.ema_fast.iloc[1:4], [10.5, 10+1/6, 11+7/18])
    assert_allclose(a.band_upper.iloc[2], np.mean([10,11,10])+2*np.std([10,11,10]))
    assert a.breakout_high.iloc[3] == 11  # current close 12 excluded
    assert a.momentum.iloc[3] == pytest.approx(.2)


@pytest.mark.parametrize('values,expected', [(np.arange(1,31),100),(np.arange(30,0,-1),0),(np.ones(30),50)])
def test_rsi_limiting_cases(values,expected):
    assert_allclose(technical_indicators(price(values))['a','rsi'].dropna(),expected)


def test_all_indicators_are_causal_and_macd_convention_is_explicit():
    p = price(100*np.exp(np.random.default_rng(4).normal(0,.01,100).cumsum()))
    a = technical_indicators(p)
    changed = p.copy(); changed.iloc[60:] *= 3
    pd.testing.assert_frame_equal(a.iloc[:60], technical_indicators(changed).iloc[:60])
    v = a['a']
    assert_allclose(v.macd.dropna(), (v.ema_fast-v.ema_slow).dropna())
    assert v.macd_signal.iloc[:33].isna().all()
    assert v.macd_signal.iloc[33] == pytest.approx(v.macd.iloc[25:34].mean())


def test_momentum_ties_cash_and_missing_histories():
    p = price([10,11,12,13]); p['b']=p.a; p['c']=[10,9,8,7]
    assert_allclose(momentum_weights(p,lookback=3,top_n=1),[1,0,0])
    assert_allclose(momentum_weights(p[['c']],lookback=3),[0])
    p.loc[p.index[2],'a']=np.nan
    assert_allclose(momentum_weights(p,lookback=3,top_n=1),[0,1,0])
    with pytest.raises(ValueError):
        technical_indicators(p)


def test_max_test_zero_control_positive_control_and_reproducibility():
    dates = pd.date_range('2000',periods=80)
    benchmark = pd.Series(np.random.default_rng(5).normal(0,.01,80),index=dates)
    candidates = pd.DataFrame({'same':benchmark,'worse':benchmark-.01,'better':benchmark+.02})
    out = compare_strategies(candidates,benchmark,block_length=4,n_boot=199,seed=7)
    assert_allclose(out.mean_difference,[0,-.01,.02])
    assert_allclose(out.adjusted_pvalue,[1,1,.005])
    assert out.global_pvalue == .005
    same = compare_strategies(candidates[['same']],benchmark,n_boot=199)
    assert same.global_pvalue == 1
    a = compare_strategies(candidates,benchmark,n_boot=199,seed=2)
    b = compare_strategies(candidates,benchmark,n_boot=199,seed=2)
    pd.testing.assert_frame_equal(a.to_frame(),b.to_frame())
    assert a.to_markdown() and a.to_typst() and a.to_latex()


def test_candidate_family_increases_adjusted_pvalues():
    rng=np.random.default_rng(4)
    data=pd.DataFrame(rng.normal(0,.01,(200,8)),index=pd.date_range('2000',periods=200),columns=list('abcdefgh'))
    zero=pd.Series(0.,index=data.index)
    single=compare_strategies(data[['a']],zero,n_boot=199,seed=1)
    family=compare_strategies(data,zero,n_boot=199,seed=1)
    assert family.adjusted_pvalue[0]>=single.adjusted_pvalue[0]


def test_comparison_rejects_misalignment():
    p=price(np.ones(30))
    with pytest.raises(ValueError,match='exactly'):
        compare_strategies(p,p.a.iloc[:-1])
