"""Perpetual-inventory capital stocks and the Solow residual built on them.

Built on a hand-made panel rather than a frozen data slice, matching
``tests/test_oecd_qna_tools.py``: the point of most of these is an *analytic*
answer the recursion has to reproduce, which a real panel cannot give.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.capital import (PIM_ASSETS, PIM_DELTAS, qna_capital, qna_tfp)
from puremacro import capital as mod

_Q = pd.period_range("1995Q1", periods=120, freq="Q").to_timestamp(how="start")


def _panel(codes=("AAA",), *, inv=100.0, defl=100.0, growth=0.0,
           drop_defl=(), gdp=1000.0) -> pd.DataFrame:
    """A qna_panel-shaped frame: constant real investment unless told otherwise."""
    frames = []
    for code in codes:
        n = len(_Q)
        t = np.arange(n, dtype=float)
        d = {}
        for a in PIM_ASSETS:
            vol = inv * (1.0 + growth) ** t
            d[f"{a}_real"] = vol
            d[a] = vol * defl / 100.0
            if a not in drop_defl:
                d[f"{a}_defl"] = np.full(n, defl, dtype=float)
        d["inv"] = sum(d[a] for a in PIM_ASSETS)
        d["gdp_real"] = gdp * (1.0 + growth) ** t
        d["gdp_defl"] = np.full(n, 100.0)
        d["gdp_income"] = gdp * (1.0 + growth) ** t
        d["taxes_prod_imp_net"] = 0.1 * d["gdp_income"]
        d["comp_emp"] = 0.5 * d["gdp_income"]
        d["hours"] = np.full(n, 1000.0)
        d["emp"] = np.full(n, 500.0)
        d["emp_employees"] = np.full(n, 400.0)
        d["emp_selfemp"] = np.full(n, 100.0)
        f = pd.DataFrame(d, index=_Q)
        f.index.name = "date"
        f["code"] = code
        frames.append(f.reset_index().set_index(["code", "date"]))
    return pd.concat(frames).sort_index()


def test_depreciation_is_converted_geometrically_not_linearly():
    """delta_a/4 is the tempting wrong answer and it understates depreciation,
    so it biases the steady-state stock up — by ~5% for equipment."""
    for asset, annual in PIM_DELTAS.items():
        dq = mod._quarterly_delta(annual)
        assert (1.0 - dq) ** 4 == pytest.approx(1.0 - annual)
        # The geometric rate is the LARGER number, which is exactly why the
        # linear shortcut understates depreciation: compounding delta/4 over
        # four quarters loses less than delta.
        assert dq > annual / 4.0
        assert 1.0 - (1.0 - annual / 4.0) ** 4 < annual
    assert mod._quarterly_delta(0.130) == pytest.approx(0.03421643, abs=1e-8)
    assert mod._quarterly_delta(0.200) == pytest.approx(0.05425839, abs=1e-8)
    assert mod._quarterly_delta(0.030) == pytest.approx(0.00758588, abs=1e-8)
    assert mod._quarterly_delta(0.011) == pytest.approx(0.00276142, abs=1e-8)


def test_constant_investment_matches_the_closed_form():
    """From K_0 = 0 with I constant, K_n = I (1-(1-d)^n)/d exactly — and the
    limit is the steady state I/d, which the slow assets are nowhere near
    inside a 30-year sample."""
    res = qna_capital(_panel(), k0="zero")
    k = res.stocks.loc["AAA"]
    n = len(k) - 1
    for asset in PIM_ASSETS:
        dq = res.delta_quarterly[asset]
        exact = 100.0 * (1.0 - (1.0 - dq) ** n) / dq
        assert k[mod._STOCK[asset]].iloc[-1] == pytest.approx(exact, rel=1e-12)
    # how far each asset actually gets toward its own steady state in 120q
    reached = {a: k[mod._STOCK[a]].iloc[-1] * res.delta_quarterly[a] / 100.0
               for a in PIM_ASSETS}
    assert reached["inv_ipp"] > 0.99 and reached["inv_equip"] > 0.98
    assert reached["inv_dwell"] < 0.30      # dwellings barely start


def test_harberger_k0_starts_at_the_steady_state_and_stays_there():
    """K_0 = I/(g+delta_q) with g=0 is exactly the steady state, so a constant
    investment series should produce a flat stock from the first quarter."""
    res = qna_capital(_panel(), k0="harberger")
    k = res.stocks.loc["AAA"]["k_equip"]
    assert k.iloc[0] == pytest.approx(k.iloc[-1], rel=1e-9)
    assert k.iloc[0] == pytest.approx(100.0 / res.delta_quarterly["inv_equip"], rel=1e-9)


def test_mid_period_timing_is_a_pure_level_factor():
    """It rescales by (1-delta_q)^(1/2) and must not touch any growth rate —
    which is only true if K_0 is scaled along with the flows."""
    end = qna_capital(_panel(growth=0.01), timing="predetermined")
    mid = qna_capital(_panel(growth=0.01), timing="mid")
    for asset in PIM_ASSETS:
        col = mod._STOCK[asset]
        a = end.stocks.loc["AAA"][col].to_numpy()
        b = mid.stocks.loc["AAA"][col].to_numpy()
        ratio = b / a
        assert np.allclose(ratio, ratio[0], rtol=1e-12)
        assert ratio[0] == pytest.approx((1.0 - end.delta_quarterly[asset]) ** 0.5)


def test_tornqvist_aggregate_survives_a_zero_start():
    """Anchoring the services index on the first quarter would multiply it by
    zero when every stock starts at zero."""
    res = qna_capital(_panel(), k0="zero", aggregate="tornqvist")
    k = res.stocks.loc["AAA"]["k"]
    assert (k.iloc[1:] > 0).all()
    assert np.isfinite(k.iloc[1:]).all()


def test_tornqvist_is_invariant_to_the_price_reference_year():
    """The reason it is the default. Deflators are an index against whatever
    year the source chose; re-referencing them must not move a real quantity."""
    base = qna_capital(_panel(defl=100.0))
    rebased = qna_capital(_panel(defl=137.0))          # same prices, new base year
    a = base.stocks.loc["AAA"]["k"].to_numpy()
    b = rebased.stocks.loc["AAA"]["k"].to_numpy()
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_a_missing_deflator_is_refused_rather_than_silently_summed(monkeypatch):
    """Colombia publishes asset volumes and no current prices at all, so it has
    no deflators. A services index cannot be formed and must not pretend."""
    p = _panel(drop_defl=("inv_ipp",))
    assert qna_capital(p, aggregate="tornqvist").names == ()
    with pytest.raises(ValueError, match="deflator"):
        qna_capital(p, aggregate="tornqvist", strict=True)
    # the sum does not need prices, so it still works
    assert qna_capital(p, aggregate="sum").names == ("AAA",)


def test_k0_sensitivity_reports_the_actual_level_error():
    """It is the headline caveat of the module, so it has to be the real number."""
    res = qna_capital(_panel(growth=0.005))
    reported = res.stocks.loc["AAA"]["k0_sensitivity"].iloc[-1]

    orig = mod._pim
    monkey = lambda inv, dq, k0, mid: orig(inv, dq, 1.5 * k0, mid)   # noqa: E731
    mod._pim = monkey
    try:
        shocked = qna_capital(_panel(growth=0.005))
    finally:
        mod._pim = orig
    actual = abs(shocked.stocks.loc["AAA"]["k"].iloc[-1]
                 / res.stocks.loc["AAA"]["k"].iloc[-1] - 1.0)
    assert reported == pytest.approx(actual, rel=1e-6)


def test_coverage_flags_a_country_whose_assets_do_not_add_up():
    """Australia's four asset classes cover ~71% of its published total GFCF,
    with no further asset code that closes the gap."""
    p = _panel()
    p["inv"] = p["inv"] / 0.71                       # total larger than the parts
    res = qna_capital(p)
    assert res.stocks.loc["AAA"]["coverage_pct"].median() == pytest.approx(71.0, abs=0.5)


def test_tfp_decomposition_adds_back_up():
    """log Y = log A + s log L + (1-s) log K, exactly, or it is not a residual."""
    p = _panel(growth=0.004)
    res = qna_tfp(p)
    f = res.tfp.loc["AAA"]
    y = np.log(mod._output(p.loc["AAA"], "va_factor"))
    np.testing.assert_allclose(
        (f["log_tfp"] + f["contrib_labor"] + f["contrib_capital"]).to_numpy(),
        y.to_numpy(), rtol=1e-10)


def test_gollin_share_exceeds_the_unadjusted_one():
    """The whole point of the correction: imputing labour income to the
    self-employed can only raise the share. Here 400 of 500 workers are
    employees, so the scale factor is 1.25."""
    p = _panel()
    adj = qna_tfp(p, share="gollin").labor_share["AAA"]
    raw = qna_tfp(p, share="unadjusted").labor_share["AAA"]
    assert adj == pytest.approx(raw * 1.25)
    assert adj > raw


def test_labor_share_denominator_is_value_added_at_factor_cost():
    """Not GDP: taxes less subsidies on production and imports are removed, so
    the shares of labour and capital sum to one by construction."""
    p = _panel()
    raw = qna_tfp(p, share="unadjusted").labor_share["AAA"]
    # comp_emp = 0.5*GDP and D2X3 = 0.1*GDP, so the share is 0.5/0.9
    assert raw == pytest.approx(0.5 / 0.9)


def test_result_objects_follow_the_house_standard():
    """Frozen dataclasses with the shared vocabulary, per ARCHITECTURE.md."""
    import dataclasses
    cap = qna_capital(_panel())
    tfp = qna_tfp(_panel())
    for r in (cap, tfp):
        assert dataclasses.is_dataclass(r)
        assert r.__dataclass_params__.frozen
        assert isinstance(r.names, tuple) and r.n_obs > 0
        assert isinstance(r.summary(), pd.DataFrame)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cap.n_obs = 0


@pytest.mark.parametrize("kw,msg", [
    ({"aggregate": "mean"}, "aggregate"),
    ({"capital_gains": "sometimes"}, "capital_gains"),
    ({"timing": "start"}, "timing"),
    ({"k0": "guess"}, "k0"),
])
def test_bad_options_are_refused(kw, msg):
    with pytest.raises(ValueError, match=msg):
        qna_capital(_panel(), **kw)


@pytest.mark.parametrize("kw,msg", [
    ({"share": "raw"}, "share"),
    ({"share_mode": "rolling"}, "share_mode"),
    ({"labor": "workers"}, "labor"),
    ({"output": "gva"}, "output"),
])
def test_bad_tfp_options_are_refused(kw, msg):
    with pytest.raises(ValueError, match=msg):
        qna_tfp(_panel(), **kw)
