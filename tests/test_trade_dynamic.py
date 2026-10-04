"""Tests for puremacro.trade.dynamic (dynamic MRIO with sector capital).

The two-country analytic accounts are a software fixture, never empirical
evidence. Every test of the IO engine's ``test_native_dynamics.py`` that
concerns the ported modules is reproduced here on the same fixture; the
parity tests import the IO engine from the research volume when it is
mounted and skip otherwise. Slow tests (bundled 77x11 determinacy and the
160/320/640 horizon ladder) are opt-in via ``pytest -m slow``.
"""
from __future__ import annotations

import importlib
import json
import os
import pathlib
import pickle
import subprocess
import sys
import types
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from puremacro.trade.dynamic import (
    ConsumptionEquivalentResult,
    DynamicAccounts,
    DynamicCalibration,
    DynamicDeterminacyError,
    DynamicEconomy,
    DynamicSolveError,
    DynamicStabilityResult,
    DynamicSteadyStateResult,
    DynamicTariff,
    DynamicTransitionResult,
    EconomicDomainError,
    HorizonComparisonResult,
    HorizonLadderResult,
    analytic_two_country_accounts,
    calibrate_dynamic,
    compare_horizons,
    consumption_equivalent_welfare,
    discounted_endpoint_capital_value,
    run_horizon_ladder,
    solve_dynamic_steady_state,
    solve_dynamic_transition,
    stability_report,
    tariff_path,
)
from puremacro.trade.dynamic.accounts import FINAL_CATEGORIES, _merchandise
from puremacro.trade.dynamic.stationary import StationaryProblem, solve_condensed_steady
from puremacro.trade.dynamic.transition import (PathProblem, _default_start, _installation_rates, path_residual,
                                                steady_linearization)

# The IO research workspace; parity tests skip when it is absent. Override with PUREMACRO_IO_ROOT.
IO_ROOT = pathlib.Path(os.environ.get("PUREMACRO_IO_ROOT", "/Volumes/BIGDATA/Research/IO"))
REPO = pathlib.Path(__file__).resolve().parent.parent
PYTHON = sys.executable
ENV_THREADS = {"OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2", "VECLIB_MAXIMUM_THREADS": "2"}


def _io_module(subpath, name):
    if not (IO_ROOT / subpath).exists():
        pytest.skip("IO research volume not mounted")
    sys.path.insert(0, str(IO_ROOT / subpath))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.pop(0)


# ---------------------------------------------------------------------------
# Welfare (ported from the IO engine's tests)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sigma", [0.5, 1.0, 1.0 - 1e-9, 1.0 + 1e-9, 2.0, 7.0])
def test_welfare_constant_consumption_change_has_exact_equivalent(sigma):
    c0 = np.array([20.0, 70.0])
    change = np.array([0.85, 1.12])
    result = consumption_equivalent_welfare(
        np.tile(change * c0, (7, 1)), baseline_consumption=c0,
        beta=0.96, risk_aversion=sigma, terminal_consumption=change * c0,
        baseline_price=np.array([2.0, 3.0]), baseline_gdp=np.array([100.0, 500.0]),
    )
    np.testing.assert_allclose(result.consumption_equivalent_ratio, change, rtol=2e-14)
    np.testing.assert_allclose(result.equivalent_annual_expenditure, [-6.0, 25.2], rtol=2e-14)
    np.testing.assert_allclose(result.ev_pct_gdp, [-6.0, 5.04], rtol=2e-14)
    np.testing.assert_allclose(result.equivalent_present_value_expenditure,
                               result.equivalent_annual_expenditure / 0.04, rtol=2e-14)


@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0, 6.0])
def test_welfare_equivalent_reproduces_direct_discounted_utility(sigma):
    beta = 0.93
    c0 = np.array([3.0, 4.0])
    ratio = np.array([[0.7, 1.1], [1.2, 0.93], [1.05, 0.85]])
    terminal = np.array([1.04, 1.13])
    result = consumption_equivalent_welfare(
        ratio * c0, baseline_consumption=c0, beta=beta,
        risk_aversion=sigma, terminal_consumption=terminal * c0,
    )

    def utility(x):
        return np.log(x) if sigma == 1 else (x ** (1 - sigma) - 1) / (1 - sigma)

    direct = np.sum(beta ** np.arange(3)[:, None] * utility(ratio), axis=0)
    direct += beta ** 3 / (1 - beta) * utility(terminal)
    equivalent = utility(result.consumption_equivalent_ratio) / (1 - beta)
    np.testing.assert_allclose(equivalent, direct, rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(result.normalized_lifetime_utility, direct, rtol=2e-13)


def test_welfare_sigma_two_is_discounted_harmonic_mean_not_ies_power():
    beta = 0.8
    ratio = np.array([[0.8], [1.3]])
    result = consumption_equivalent_welfare(ratio, baseline_consumption=np.ones(1), beta=beta, risk_aversion=2)
    expected = (1 + beta) / (1 / ratio[0, 0] + beta / ratio[1, 0])
    assert result.consumption_equivalent_ratio[0] == pytest.approx(expected, rel=1e-14)
    assert not result.tail_included
    assert "Finite-horizon" in result.interpretation


def test_welfare_stationary_tail_begins_after_last_recorded_period():
    beta = 0.9
    result = consumption_equivalent_welfare(
        np.array([[0.8], [0.9]]), baseline_consumption=np.ones(1),
        beta=beta, risk_aversion=1, terminal_consumption=np.array([1.2]),
    )
    expected_log = (1 - beta) * (np.log(0.8) + beta * np.log(0.9)) + beta ** 2 * np.log(1.2)
    assert result.consumption_equivalent_ratio[0] == pytest.approx(np.exp(expected_log))
    assert result.terminal_discount_weight == pytest.approx(beta ** 2 / (1 - beta))
    assert result.total_discount_weight == pytest.approx(1 / (1 - beta))
    assert result.tail_is_approximation
    assert "accuracy require separate validation" in result.interpretation


def test_welfare_unit_changes_leave_real_and_gdp_equivalents_unchanged():
    c0 = np.array([20.0, 40.0])
    consumption = c0 * np.array([[0.98, 1.01], [0.97, 1.02]])
    arguments = dict(beta=0.96, risk_aversion=2)
    original = consumption_equivalent_welfare(
        consumption, baseline_consumption=c0, terminal_consumption=c0 * [0.96, 1.03],
        baseline_price=[1.2, 0.8], baseline_gdp=[100, 200], **arguments,
    )
    scaled = consumption_equivalent_welfare(
        consumption * 1e9, baseline_consumption=c0 * 1e9,
        terminal_consumption=c0 * [0.96, 1.03] * 1e9,
        baseline_price=[1.2, 0.8], baseline_gdp=np.array([100, 200]) * 1e9, **arguments,
    )
    np.testing.assert_allclose(scaled.consumption_equivalent_pct, original.consumption_equivalent_pct,
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(scaled.ev_pct_gdp, original.ev_pct_gdp, rtol=1e-12, atol=1e-12)


def test_welfare_rejects_nonfinite_nonpositive_or_misaligned_quantities():
    defaults = dict(baseline_consumption=[1.0, 2.0], beta=0.96, risk_aversion=2)
    for values in ([[1.0, 0.0]], [[1.0, np.nan]], [[1.0, np.inf]], [1.0, 2.0], []):
        with pytest.raises(ValueError):
            consumption_equivalent_welfare(values, **defaults)
    with pytest.raises(ValueError, match="terminal_consumption"):
        consumption_equivalent_welfare([[1.0, 2.0]], terminal_consumption=[1.0], **defaults)
    with pytest.raises(ValueError, match="baseline_price"):
        consumption_equivalent_welfare([[1.0, 2.0]], baseline_gdp=[10, 20], **defaults)
    with pytest.raises(ValueError, match="country_codes"):
        consumption_equivalent_welfare([[1.0, 2.0]], country_codes=("A",), **defaults)
    for keyword, value in [("beta", 1.0), ("beta", np.nan), ("risk_aversion", 0), ("risk_aversion", np.inf)]:
        with pytest.raises(ValueError):
            consumption_equivalent_welfare([[1.0, 2.0]], **(defaults | {keyword: value}))


def test_welfare_extending_an_already_stationary_tail_is_invariant():
    initial = np.array([[0.91, 1.1], [0.93, 1.04]])
    terminal = np.array([0.96, 1.02])
    arguments = dict(baseline_consumption=np.ones(2), beta=0.96, risk_aversion=2, terminal_consumption=terminal)
    short = consumption_equivalent_welfare(initial, **arguments)
    long = consumption_equivalent_welfare(np.vstack((initial, np.tile(terminal, (50, 1)))), **arguments)
    np.testing.assert_allclose(short.consumption_equivalent_pct, long.consumption_equivalent_pct,
                               rtol=2e-14, atol=2e-14)


def test_welfare_result_is_frozen_and_renders_the_report_quintet():
    result = consumption_equivalent_welfare(
        np.array([[0.98, 1.01], [0.97, 1.02]]), baseline_consumption=np.ones(2), beta=0.96, risk_aversion=2,
        terminal_consumption=[0.96, 1.03], baseline_price=[1.0, 2.0], baseline_gdp=[10.0, 20.0],
        country_codes=("AAA", "BBB"))
    assert isinstance(result, ConsumptionEquivalentResult)
    frame = result.to_dataframe()
    assert list(frame.index) == ["AAA", "BBB"] and "ev_pct_gdp" in frame.columns
    for text in (result.to_markdown(), result.to_latex(), result.to_typst()):
        assert "AAA" in text
    assert "AAA=" in result.summary()
    assert result.to_dict()["country_codes"] == ["AAA", "BBB"]
    with pytest.raises((AttributeError, TypeError)):
        result.beta = 0.5  # type: ignore[misc]
    assert not result.consumption_equivalent_pct.flags.writeable


# ---------------------------------------------------------------------------
# Fixtures and independent scalar accounts
# ---------------------------------------------------------------------------
@pytest.fixture
def native_fixture():
    raw = analytic_two_country_accounts()
    calibration = calibrate_dynamic(raw)
    return raw, calibration, DynamicEconomy(calibration, adjustment_cost=2, risk_aversion=2)


def asymmetric_policy(calibration):
    rates = np.zeros((calibration.n_cells, calibration.n_countries))
    rates[calibration.country == 0, 1] = .06
    rates[calibration.country == 1, 0] = .08
    # Distinct final-use duties expose use-category and axis mistakes.
    return DynamicTariff.build(rates, consumption_rates=rates * .7, investment_rates=rates * 1.3,
                               label="analytic tariff")


def check_independent_accounts(calibration, model, state, following, policy, atol=2e-9):
    """Rebuild small-account values with scalar source/destination loops.

    This deliberately does not call residuals, certificates, price-network
    methods, or the model's stored tax, budget, and Euler residual fields.
    """
    d, s, sn = calibration, state, following
    n, nc = d.n_cells, d.n_countries
    a = d.Z.toarray() / d.y0[None, :]
    inputs = a * s["y"][None, :]
    final_c = d.omegaC * s["C"]
    taxable_c = d.omegaCtax * s["C"]
    final_i = d.omegaI * s["Inational"]
    assert np.count_nonzero(d.qOther) == 0
    np.testing.assert_allclose(s["y"], inputs.sum(axis=1) + (final_c + final_i).sum(axis=1), atol=atol)
    primal = (s["K"] / d.K0) ** d.alpha * (s["labor_j"] / d.L0sector) ** (1 - d.alpha)
    np.testing.assert_allclose(s["y"] / d.y0, primal, atol=atol, rtol=atol)
    country_labor = np.zeros(nc)
    factor_income = np.zeros(nc)
    tariffs = np.zeros(nc)
    taxes = np.zeros(nc)
    final_expenditure = np.zeros(nc)
    actual_net_exports = np.zeros(nc)
    for j in range(n):
        c = int(d.country[j])
        country_labor[c] += s["labor_j"][j]
        factor_income[c] += s["R"][j] * s["K"][j] + s["w"][c] * s["labor_j"][j]
        taxes[c] += d.tax[j] * s["p"][j] * s["y"][j]
        intermediate_cost = 0.0
        for i in range(n):
            trade_value = s["p"][i] * inputs[i, j]
            tariffs[c] += policy.rates[i, c] * trade_value
            intermediate_cost += (1 + policy.rates[i, c]) * trade_value
            if d.country[i] != c:
                actual_net_exports[int(d.country[i])] += trade_value
                actual_net_exports[c] -= trade_value
        producer_receipts = s["p"][j] * s["y"][j]
        factor_cost = s["R"][j] * s["K"][j] + s["w"][c] * s["labor_j"][j]
        assert producer_receipts == pytest.approx(intermediate_cost + factor_cost + d.tax[j] * producer_receipts,
                                                  rel=atol, abs=atol)
    for c in range(nc):
        basic_c = basic_i = 0.0
        for i in range(n):
            c_tariff = policy.consumption_rates[i, c] * s["p"][i] * taxable_c[i, c]
            i_tariff = policy.investment_rates[i, c] * s["p"][i] * final_i[i, c]
            tariffs[c] += c_tariff + i_tariff
            basic_c += s["p"][i] * final_c[i, c] + c_tariff
            basic_i += s["p"][i] * final_i[i, c] + i_tariff
            if d.country[i] != c:
                trade_value = s["p"][i] * (final_c[i, c] + final_i[i, c])
                actual_net_exports[int(d.country[i])] += trade_value
                actual_net_exports[c] -= trade_value
        c_tax = basic_c * d.tC[c] / (1 - d.tC[c])
        i_tax = basic_i * d.tI[c] / (1 - d.tI[c])
        taxes[c] += c_tax + i_tax
        final_expenditure[c] = basic_c + basic_i + c_tax + i_tax
        assert basic_c + c_tax == pytest.approx(s["PC"][c] * s["C"][c], rel=atol, abs=atol)
        assert basic_i + i_tax == pytest.approx(s["PI"][c] * s["Inational"][c], rel=atol, abs=atol)
    np.testing.assert_allclose(country_labor, d.L0, atol=atol, rtol=atol)
    np.testing.assert_allclose(tariffs, s["TR"], atol=atol, rtol=atol)
    np.testing.assert_allclose(taxes, s["PT"], atol=atol, rtol=atol)
    # Includes the country's budget that the solver drops for the numeraire.
    transfer = d.XN0 * s["w"][-1]
    np.testing.assert_allclose(final_expenditure + transfer, factor_income + taxes + tariffs, atol=atol, rtol=atol)
    np.testing.assert_allclose(actual_net_exports, transfer, atol=atol, rtol=atol)
    assert abs(transfer.sum()) < atol

    x, xn = s["I"] / s["K"], sn["I"] / sn["K"]
    phi = model.adjustment_cost
    installed = x - .5 * phi * (x - d.delta) ** 2
    installed_next = xn - .5 * phi * (xn - d.delta) ** 2
    marginal = 1 - phi * (x - d.delta)
    marginal_next = 1 - phi * (xn - d.delta)
    np.testing.assert_allclose(s["Knext"], s["K"] * (1 - d.delta + installed), atol=atol, rtol=atol)
    assert np.all(x > 0) and np.all(marginal > 0)
    # Derived in levels rather than calling the log-Euler residual.
    q = s["PI"][d.country] / marginal
    qn = sn["PI"][d.country] / marginal_next
    sdf = d.beta * (sn["C"] / s["C"]) ** (-model.risk_aversion) * s["PC"] / sn["PC"]
    payoff = sn["R"] + qn * (1 - d.delta + installed_next - xn * marginal_next)
    np.testing.assert_allclose(q, sdf[d.country] * payoff, atol=atol, rtol=atol)


# ---------------------------------------------------------------------------
# Accounts container and bridges
# ---------------------------------------------------------------------------
def test_analytic_accounts_follow_the_shared_table_contract():
    acc = analytic_two_country_accounts()
    assert acc.fd_codes == FINAL_CATEGORIES and acc.F.shape == (4, 2, 5) and acc.TFD.shape == (2, 5)
    assert acc.Z.shape == (4, 4) and not acc.Z.flags.writeable and not acc.F.flags.writeable
    assert acc.countries == acc.country_codes == ("AAA", "BBB") and acc.cell_labels[1] == "AAA:S2"
    np.testing.assert_allclose(acc.output, acc.Z.sum(1) + acc.final_demand.sum(1), atol=1e-13)
    report = acc.accounting_report()
    assert report["row_output_max_absolute_gap"] < 1e-13 and report["active_cells"] == 4
    assert report["final_category_totals"]["V"] == pytest.approx(1.1)
    np.testing.assert_allclose(acc.capital_share, [.35, .48, .30, .52], atol=1e-13)
    assert acc.merchandise_mask.tolist() == [True, False]
    with pytest.raises((AttributeError, TypeError)):
        acc.dataset = "x"  # type: ignore[misc]
    frame = acc.to_dataframe()
    assert frame.index.name == "check" and frame.loc["active_cells", "value"] == 4
    assert frame.loc["final_total_V", "value"] == pytest.approx(1.1)
    assert "analytic_test_fixture" in acc.to_markdown()
    assert "analytic\\_test\\_fixture" in acc.to_latex() and "final\\_total\\_V" in acc.to_typst()


def test_from_arrays_maps_source_codes_merges_categories_and_applies_investment_policy():
    acc = analytic_two_country_accounts()
    M, N = acc.n_cells, acc.n_countries
    # Split C into three EXIOBASE-style pieces, merge V and VAL as FIGARO's P5M.
    F = np.concatenate([acc.F[:, :, :1] * .5, acc.F[:, :, :1] * .3, acc.F[:, :, :1] * .2,
                        acc.F[:, :, 1:2], acc.F[:, :, 2:3], (acc.F[:, :, 3] + acc.F[:, :, 4])[:, :, None]], axis=2)
    TFD = np.column_stack([acc.TFD[:, 0] * .5, acc.TFD[:, 0] * .3, acc.TFD[:, 0] * .2,
                           acc.TFD[:, 1], acc.TFD[:, 2], acc.TFD[:, 3] + acc.TFD[:, 4]])
    codes = ("HFCE", "NPISH", "GGFC", "GFCF", "DPABR", "P5M")
    rebuilt = DynamicAccounts.from_arrays(
        country_codes=acc.countries, sector_codes=acc.sectors, Z=acc.Z, F=F.reshape(M, N * 6), TFD=TFD.ravel(),
        fd_codes=codes, VA=acc.VA, TLS=acc.TLS, output=acc.output, production_taxes=acc.production_taxes,
        labor_compensation=acc.labor_compensation, operating_surplus=acc.operating_surplus)
    np.testing.assert_allclose(rebuilt.final_consumption, acc.final_consumption, atol=1e-13)
    np.testing.assert_allclose(rebuilt.inventory_changes, acc.inventory_changes + acc.valuables, atol=1e-13)
    np.testing.assert_allclose(rebuilt.TFD[:, 3], acc.TFD[:, 3] + acc.TFD[:, 4], atol=1e-13)
    assert rebuilt.metadata["final_use_mapping"]["P5M"] == "V"
    with pytest.raises(ValueError, match="no mapping"):
        DynamicAccounts.from_arrays(country_codes=acc.countries, sector_codes=acc.sectors, Z=acc.Z, F=F,
                                    TFD=TFD, fd_codes=("HFCE", "NPISH", "GGFC", "GFCF", "DPABR", "MYSTERY"),
                                    VA=acc.VA, TLS=acc.TLS, output=acc.output)
    custom = DynamicAccounts.from_arrays(country_codes=acc.countries, sector_codes=acc.sectors, Z=acc.Z, F=F,
                                         TFD=TFD, fd_codes=("HFCE", "NPISH", "GGFC", "GFCF", "DPABR", "MYSTERY"),
                                         VA=acc.VA, TLS=acc.TLS, output=acc.output, fd_map={"MYSTERY": "VAL"})
    np.testing.assert_allclose(custom.valuables, acc.inventory_changes + acc.valuables, atol=1e-13)
    negative = np.array(acc.F, copy=True)
    negative[0, 1, 1] = -.5
    with pytest.raises(ValueError, match="negative fixed-investment"):
        DynamicAccounts.from_arrays(country_codes=acc.countries, sector_codes=acc.sectors, Z=acc.Z, F=negative,
                                    TFD=acc.TFD, VA=acc.VA, TLS=acc.TLS, output=acc.output)
    moved = DynamicAccounts.from_arrays(country_codes=acc.countries, sector_codes=acc.sectors, Z=acc.Z, F=negative,
                                        TFD=acc.TFD, VA=acc.VA, TLS=acc.TLS, output=acc.output,
                                        negative_investment="to_inventory")
    assert moved.final_investment[0, 1] == 0 and moved.inventory_changes[0, 1] == pytest.approx(acc.inventory_changes[0, 1] - .5)
    assert moved.metadata["negative_investment"] == {"policy": "to_inventory", "cells": 1, "mass": -.5,
                                                     "note": "negative GFCF entries moved into signed inventory changes"}
    np.testing.assert_allclose(moved.final_demand, negative.sum(axis=2), atol=1e-13)
    for bad in (dict(Z=-acc.Z), dict(VA=acc.VA[:3]), dict(output=np.r_[acc.output[:3], np.nan])):
        with pytest.raises(ValueError):
            DynamicAccounts.from_arrays(**{**dict(country_codes=acc.countries, sector_codes=acc.sectors, Z=acc.Z,
                                                  F=acc.F, TFD=acc.TFD, VA=acc.VA, TLS=acc.TLS, output=acc.output), **bad})


def test_from_arrays_rejects_a_transposed_tax_table_and_an_unmapped_export_column():
    acc = analytic_two_country_accounts()
    base = dict(country_codes=acc.countries, sector_codes=acc.sectors, Z=acc.Z, F=acc.F, VA=acc.VA,
                TLS=acc.TLS, output=acc.output)
    # A (K, N) = (5, 2) table has N*K entries but is not (N, K): it must not be
    # silently reshaped into scrambled purchaser taxes.
    with pytest.raises(ValueError, match=r"TFD must have shape \(2, 5\)"):
        DynamicAccounts.from_arrays(TFD=acc.TFD.T, **base)
    flat = DynamicAccounts.from_arrays(TFD=acc.TFD.ravel(), **base)
    np.testing.assert_array_equal(flat.TFD, acc.TFD)
    # The EXIOBASE EXPORT column is not residents' purchases abroad; it needs fd_map.
    three = {**base, "F": acc.F[:, :, :3], "TFD": acc.TFD[:, :3], "fd_codes": ("HFCE", "GFCF", "EXPORT")}
    with pytest.raises(ValueError, match="'EXPORT' has no mapping"):
        DynamicAccounts.from_arrays(**three)
    mapped = DynamicAccounts.from_arrays(fd_map={"EXPORT": "X"}, **three)
    np.testing.assert_array_equal(mapped.purchases_abroad, acc.purchases_abroad)
    assert mapped.metadata["final_use_mapping"]["EXPORT"] == "X"


def test_from_icio_accepts_a_section_two_table_object():
    acc = analytic_two_country_accounts()
    bridged = DynamicAccounts.from_icio(acc)
    assert bridged.dataset == "analytic_test_fixture" and bridged.metadata["source"] == "DynamicAccounts"
    np.testing.assert_array_equal(bridged.F, acc.F)
    np.testing.assert_array_equal(bridged.TFD, acc.TFD)
    assert bridged.merchandise_mask.tolist() == [True, False]
    np.testing.assert_array_equal(bridged.operating_surplus, acc.operating_surplus)
    # A bare object with the contract attributes, three categories and a
    # negative investment cell that the explicit policy moves into inventories.
    F3 = np.array(acc.F[:, :, :3], copy=True)
    F3[1, 0, 1] = -.25
    table = types.SimpleNamespace(Z=np.array(acc.Z), F=F3, fd_codes=("C", "G", "X"), VA=np.array(acc.VA),
                                  TLS=np.array(acc.TLS), TFD=np.array(acc.TFD[:, :3]), output=F3.sum((1, 2)) + acc.Z.sum(1),
                                  country_codes=acc.countries, sector_codes=acc.sectors,
                                  merchandise_mask=np.array([False, True]), metadata={"year": 2019})
    with pytest.raises(ValueError, match="negative fixed-investment"):
        DynamicAccounts.from_icio(table)
    duck = DynamicAccounts.from_icio(table, negative_investment="to_inventory")
    assert duck.dataset == "SimpleNamespace" and duck.metadata["source"] == "SimpleNamespace"
    assert duck.metadata["year"] == 2019 and duck.metadata["negative_investment"]["cells"] == 1
    assert duck.final_investment[1, 0] == 0 and duck.inventory_changes[1, 0] == pytest.approx(-.25)
    assert duck.merchandise_mask.tolist() == [False, True] and duck.labor_compensation is None
    np.testing.assert_array_equal(duck.valuables, 0)
    np.testing.assert_array_equal(duck.TFD[:, :3], acc.TFD[:, :3])
    np.testing.assert_array_equal(duck.TFD[:, 3:], 0)


def test_from_icio_raw_table_with_oecd_codes_and_assumed_factor_split():
    from puremacro.trade.data import RawIOData

    acc = analytic_two_country_accounts()
    M, N = acc.n_cells, acc.n_countries
    F6 = np.stack([acc.F[:, :, 0] * .6, acc.F[:, :, 0] * .1, acc.F[:, :, 0] * .3, acc.F[:, :, 1],
                   acc.F[:, :, 3], acc.F[:, :, 2]], axis=2)  # HFCE NPISH GGFC GFCF INVNT DPABR (VAL dropped)
    T6 = np.column_stack([acc.TFD[:, 0] * .6, acc.TFD[:, 0] * .1, acc.TFD[:, 0] * .3, acc.TFD[:, 1],
                          acc.TFD[:, 3], acc.TFD[:, 2]])
    raw = RawIOData(countries=list(acc.countries), sectors=list(acc.sectors), intermediate_matrix=np.array(acc.Z),
                    final_demand_matrix=F6.reshape(M, N * 6), value_added=np.array(acc.VA), taxes_less_subsidies=np.array(acc.TLS),
                    gross_output=np.array(acc.output), fd_categories=["HFCE", "NPISH", "GGFC", "GFCF", "INVNT", "DPABR"],
                    year=2019, source="hand", taxes_less_subsidies_fd=T6.ravel())
    bridged = DynamicAccounts.from_icio(raw, merchandise_sectors=("S1",))
    np.testing.assert_allclose(bridged.final_consumption, acc.final_consumption, atol=1e-13)
    np.testing.assert_allclose(bridged.purchases_abroad, acc.purchases_abroad, atol=1e-13)
    np.testing.assert_allclose(bridged.inventory_changes, acc.inventory_changes, atol=1e-13)
    assert np.all(bridged.valuables == 0)
    np.testing.assert_allclose(bridged.labor_compensation, 2 / 3 * acc.VA, atol=1e-13)
    assert "assumed labour/capital split" in bridged.metadata["factor_detail"]
    assert bridged.merchandise_mask.tolist() == [True, False]
    explicit = DynamicAccounts.from_icio(raw, labor_compensation=acc.labor_compensation,
                                         operating_surplus=acc.operating_surplus, merchandise_sectors=None)
    np.testing.assert_allclose(explicit.operating_surplus, acc.operating_surplus)
    assert explicit.merchandise_mask is None
    with pytest.raises(ValueError, match="factor_split"):
        DynamicAccounts.from_icio(raw, factor_split=(0.5, 0.6))
    with pytest.raises(TypeError):
        DynamicAccounts.from_icio(object())


FIGARO_64 = (
    "A01", "A02", "A03", "B", "C10T12", "C13T15", "C16", "C17", "C18", "C19", "C20", "C21",
    "C22", "C23", "C24", "C25", "C26", "C27", "C28", "C29", "C30", "C31_32", "C33", "D35",
    "E36", "E37T39", "F", "G45", "G46", "G47", "H49", "H50", "H51", "H52", "H53", "I", "J58",
    "J59_60", "J61", "J62_63", "K64", "K65", "K66", "L", "M69_70", "M71", "M72", "M73",
    "M74_75", "N77", "N78", "N79", "N80T82", "O84", "P85", "Q86", "Q87_88", "R90T92", "R93",
    "S94", "S95", "S96", "T", "U",
)
ISIC_SECTION_11 = ("A", "B", "C", "DE", "F", "G", "H", "J", "KL", "OPQ", "REST")


def test_isic_a_c_goods_rule_on_figaro_isic_sections_bundled_and_exiobase_codes():
    figaro = _merchandise(FIGARO_64, "isic_a_c")
    goods = [s for s, g in zip(FIGARO_64, figaro) if g]
    # Mining "B" is goods; repair and installation "C33" is not a merchandise shipment.
    assert goods == ["A01", "A02", "A03", "B", "C10T12", "C13T15", "C16", "C17", "C18", "C19", "C20", "C21",
                     "C22", "C23", "C24", "C25", "C26", "C27", "C28", "C29", "C30", "C31_32"]
    assert _merchandise(ISIC_SECTION_11, "isic_a_c").tolist() == [True] * 3 + [False] * 8
    bundled = ("AGRI", "MINQ", "MANU", "ENEG", "CONS", "GOV")
    assert _merchandise(bundled, "isic_a_c").tolist() == [True, True, True, False, False, False]
    oecd = ("A01_02", "B09", "C31T33", "D35", "CA")   # "CA" starts with C but is not a goods code
    assert _merchandise(oecd, "isic_a_c").tolist() == [True, True, True, False, False]
    assert not _merchandise(("i01.a", "i15", "i37"), "isic_a_c").any()   # EXIOBASE needs an explicit list
    with pytest.raises(ValueError):
        _merchandise(bundled, "isic")
    # The bridge records the rule and the resulting goods list.
    acc = analytic_two_country_accounts()
    table = types.SimpleNamespace(Z=np.array(acc.Z), F=np.array(acc.F), fd_codes=acc.fd_codes, VA=np.array(acc.VA),
                                  TLS=np.array(acc.TLS), TFD=np.array(acc.TFD), output=np.array(acc.output),
                                  country_codes=acc.countries, sector_codes=("B", "C33"))
    bridged = DynamicAccounts.from_icio(table)
    assert bridged.merchandise_mask.tolist() == [True, False]
    assert bridged.metadata["merchandise_sectors"]["goods"] == ["B"]
    assert "except C33" in bridged.metadata["merchandise_sectors"]["rule"]
    assert DynamicAccounts.from_icio(acc).metadata["merchandise_sectors"]["rule"] == "the source table's merchandise_mask"


def test_from_icio_bundled_table_requires_an_explicit_negative_investment_policy():
    from puremacro.trade import calibrate_trade_model, load_icio_data

    icio = load_icio_data(source="legacy", return_structured=True)
    with pytest.raises(ValueError, match="3 negative fixed-investment cells"):
        DynamicAccounts.from_icio(icio)
    acc = DynamicAccounts.from_icio(icio, negative_investment="to_inventory")
    assert acc.n_cells == 847 and acc.metadata["negative_investment"]["cells"] == 3
    assert acc.metadata["negative_investment"]["mass"] == pytest.approx(-11648179.3182, abs=1e-3)
    assert np.all(acc.final_investment >= 0) and (acc.inventory_changes < 0).sum() == 3
    np.testing.assert_allclose(acc.labor_compensation / (acc.labor_compensation + acc.operating_surplus), 2 / 3, atol=1e-12)
    assert acc.merchandise_mask.tolist() == [s in ("AGRI", "MINQ", "MANU") for s in acc.sectors]
    assert np.all(acc.production_taxes == 0)
    calib = calibrate_trade_model(icio.matrix)
    bridged = DynamicAccounts.from_trade_calibration(calib, negative_investment="to_inventory")
    np.testing.assert_array_equal(bridged.F, acc.F)
    np.testing.assert_allclose(bridged.Z, acc.Z, rtol=1e-9, atol=1e-5)
    np.testing.assert_allclose(bridged.VA, acc.VA, rtol=1e-9, atol=1e-3)
    assert bridged.dataset == "TradeCalibrationResult" and "inventories" in bridged.metadata["investment_note"]
    with pytest.raises(ValueError, match="negative fixed-investment"):
        DynamicAccounts.from_trade_calibration(calib)


@pytest.mark.parametrize("loader", ["load_figaro", "load_wiod", "load_eora", "load_exiobase"])
def test_from_trade_calibration_never_labels_provider_inventories_as_purchases_abroad(loader):
    import puremacro.trade.data as trade_data

    calib = getattr(trade_data, loader)(custom_c=3, custom_s=2, seed=1, fallback_to_synthetic=True)
    M, n = calib.n_countries * calib.n_sectors, calib.n_countries
    cx = np.asarray(calib.data_calibra)[:M, M:].reshape(M, n, 3)[:, :, 2]
    if loader == "load_exiobase":
        # Its Cx includes the export column, which cannot be split after condensation.
        with pytest.raises(ValueError, match=r"EXPORT.*cx_category"):
            DynamicAccounts.from_trade_calibration(calib, negative_investment="to_inventory")
        acc = DynamicAccounts.from_trade_calibration(calib, negative_investment="to_inventory", cx_category="V")
    else:
        acc = DynamicAccounts.from_trade_calibration(calib, negative_investment="to_inventory")
    x, v = FINAL_CATEGORIES.index("X"), FINAL_CATEGORIES.index("V")
    assert not acc.F[:, :, x].any()
    np.testing.assert_array_equal(acc.F[:, :, v], cx)
    assert acc.metadata["final_use_bridge"]["cx_category"] == "V"
    assert "inventories" not in acc.metadata["investment_note"]


def test_from_trade_calibration_keeps_signed_provider_inventories_in_v():
    import puremacro.trade.data as trade_data

    calib = trade_data.load_figaro(custom_c=3, custom_s=2, seed=1, fallback_to_synthetic=True)
    M = calib.n_countries * calib.n_sectors
    D = np.array(calib.data_calibra, dtype=float, copy=True)
    cx_col, c_col = M + 3 * 1 + 2, M + 3 * 1
    D[0, c_col] += D[0, cx_col] + 1.0          # keep the row total
    D[0, cx_col] = -1.0                        # one inventory drawdown
    acc = DynamicAccounts.from_trade_calibration(replace(calib, data_calibra=D), negative_investment="to_inventory")
    assert acc.F[0, 1, FINAL_CATEGORIES.index("V")] == -1.0
    assert not acc.F[:, :, FINAL_CATEGORIES.index("X")].any()
    assert acc.metadata["negative_investment"]["cells"] == 0


def test_from_trade_calibration_oecd_mapping_matches_the_unmapped_bundled_convention():
    from puremacro.trade.data import load_oecd_icio_granular

    calib = load_oecd_icio_granular(custom_c=3, custom_s=2, seed=1, fallback_to_synthetic=True)
    assert calib.metadata["final_use_mapping"]["Cx"] == ["DPABR"]
    acc = DynamicAccounts.from_trade_calibration(calib, negative_investment="to_inventory")
    unmapped = replace(calib, metadata={k: v for k, v in calib.metadata.items() if k != "final_use_mapping"})
    legacy = DynamicAccounts.from_trade_calibration(unmapped, negative_investment="to_inventory")
    for name in ("Z", "F", "VA", "TLS", "TFD", "output"):
        np.testing.assert_array_equal(getattr(acc, name), getattr(legacy, name))
    assert acc.F[:, :, FINAL_CATEGORIES.index("X")].sum() > 0
    odd = replace(calib, metadata={**calib.metadata, "final_use_mapping": {"C": ["C"], "I": ["I"], "Cx": ["FOO"]}})
    with pytest.raises(ValueError, match=r"DynamicAccounts.from_trade_calibration.*\['FOO'\]"):
        DynamicAccounts.from_trade_calibration(odd, negative_investment="to_inventory")


def test_from_icio_native_oecd_fixture_calibrates_strictly():
    path = REPO / "tests" / "fixtures" / "trade" / "oecd_2023" / "2019_3regions_3sectors.npz"
    if not path.is_file():
        pytest.skip("bundled OECD fixture missing")
    from puremacro.trade.data import RawIOData

    with np.load(path, allow_pickle=False) as r:
        raw = RawIOData(countries=list(r["countries"]), sectors=list(r["sectors"]), fd_categories=list(r["fd_categories"]),
                        intermediate_matrix=r["Z"], final_demand_matrix=r["F"], value_added=r["VA"],
                        taxes_less_subsidies=r["TLS"], gross_output=r["Y"], taxes_less_subsidies_fd=r["TLS_fd"],
                        year=2019, source="OECD_ICIO_2023_regular_aggregated")
    acc = DynamicAccounts.from_icio(raw)
    assert acc.n_cells == 9 and acc.metadata["source_fd_categories"] == ["HFCE", "NPISH", "GGFC", "GFCF", "INVNT", "DPABR"]
    cal = calibrate_dynamic(acc)
    assert cal.report["factor_adjusted_cells"] == 0 and cal.report["investment_basket_reallocated_countries"] == []
    eco = DynamicEconomy(cal)
    z = np.zeros(eco.n_vars)
    assert np.max(np.abs(eco.equations(z, z, z))) < 1e-12
    steady = solve_dynamic_steady_state(eco, DynamicTariff.uniform(cal, "USA", .1), tol=1e-10)
    assert steady.converged and max(steady.certificate.values()) < 1e-8
    assert stability_report(eco, steady).determinate


# ---------------------------------------------------------------------------
# Calibration, economy and solvers on the analytic fixture (IO ports)
# ---------------------------------------------------------------------------
def test_stationary_calibration_preserves_source_deliveries_and_factor_taxes(native_fixture):
    raw, d, model = native_fixture
    assert isinstance(d, DynamicCalibration)
    z = np.zeros(model.n_vars)
    policy = DynamicTariff.zero(d)
    state = model.state(z, z, policy)
    np.testing.assert_allclose(state["p"], 1, atol=1e-13)
    np.testing.assert_allclose(state["y"], raw.output, atol=1e-12)
    np.testing.assert_allclose(d.omegaC * d.C0 + d.omegaI * d.Inational0, raw.final_demand, atol=1e-13)
    np.testing.assert_allclose(d.L0sector, raw.labor_compensation, atol=1e-13)
    np.testing.assert_allclose(d.R0 * d.K0, raw.operating_surplus, atol=1e-13)
    np.testing.assert_allclose(d.b * d.y0, raw.VA - raw.production_taxes, atol=1e-13)
    np.testing.assert_allclose(d.tax * d.y0, raw.TLS + raw.production_taxes, atol=1e-13)
    assert state["PT"].sum() == pytest.approx((raw.TLS + raw.production_taxes).sum() + raw.TFD.sum())
    check_independent_accounts(d, model, state, state, policy, atol=1e-12)
    ledger = d.to_dataframe()
    assert list(ledger.index) == ["AAA", "BBB"]
    assert ledger.loc["AAA", "inventory_to_consumption"] == pytest.approx(.6)
    assert d.to_dataframe("factors").empty and d.to_dataframe("investment").empty
    assert "4 active cells" in d.summary() and "AAA" in d.to_markdown()
    assert not d.K0.flags.writeable


def test_tariff_steady_state_clears_all_independently_rebuilt_accounts(native_fixture):
    _, d, model = native_fixture
    policy = asymmetric_policy(d)
    result = solve_dynamic_steady_state(model, policy, tol=2e-11)
    assert result.converged and result.max_residual < 2e-11
    check_independent_accounts(d, model, result.state, result.state, policy)
    np.testing.assert_allclose(result.state["R"] / result.state["PI"][d.country], 1 / d.beta - 1 + d.delta, rtol=2e-10)


def test_state_and_euler_jvps_match_independent_central_differences(native_fixture):
    _, d, model = native_fixture
    policy = asymmetric_policy(d)
    rng = np.random.default_rng(8701)
    lag, current, lead = rng.normal(0, .002, (3, model.n_vars))
    vlag, vcurrent, vlead = rng.normal(0, 1, (3, model.n_vars))
    h = 2e-6
    state_derivative = model.state_jvp(current, lag, vcurrent, vlag, policy)
    plus = model.state(current + h * vcurrent, lag + h * vlag, policy)
    minus = model.state(current - h * vcurrent, lag - h * vlag, policy)
    for name in state_derivative:
        numeric = (plus[name] - minus[name]) / (2 * h)
        np.testing.assert_allclose(state_derivative[name], numeric, rtol=2e-6, atol=2e-7,
                                   err_msg=f"State derivative: {name}")
    analytic = model.equations_jvp(lead, current, lag, vlead, vcurrent, vlag, policy, policy)
    numeric = (model.equations(lead + h * vlead, current + h * vcurrent, lag + h * vlag, policy, policy)
               - model.equations(lead - h * vlead, current - h * vcurrent, lag - h * vlag, policy, policy)) / (2 * h)
    np.testing.assert_allclose(analytic, numeric, rtol=2e-6, atol=2e-7)


def test_anticipated_tariff_changes_only_forward_conditions_at_fixed_state(native_fixture):
    _, d, model = native_fixture
    zero = DynamicTariff.zero(d)
    tariff = asymmetric_policy(d)
    z = np.zeros(model.n_vars)
    unannounced = model.equations(z, z, z, zero, zero)
    announced = model.equations(z, z, z, zero, tariff)
    np.testing.assert_allclose(announced[:2 * d.n_countries], unannounced[:2 * d.n_countries], atol=1e-14)
    assert np.max(np.abs(announced[2 * d.n_countries:] - unannounced[2 * d.n_countries:])) > 1e-4


def test_terminal_that_is_not_stationary_is_rejected(native_fixture):
    _, d, model = native_fixture
    tariff = asymmetric_policy(d)
    with pytest.raises(DynamicSolveError, match="not stationary"):
        solve_dynamic_transition(model, [tariff] * 5, terminal=np.zeros(model.n_vars), tol=1e-10)


def test_installation_and_factor_domains_are_enforced(native_fixture):
    raw, d, model = native_fixture
    z = np.zeros(model.n_vars)
    infeasible = z.copy()
    infeasible[model.k_slice] = np.log(2)
    with pytest.raises(EconomicDomainError, match="increasing installation branch"):
        model.state(infeasible, z, DynamicTariff.zero(d))
    labor, capital = raw.labor_compensation.copy(), raw.operating_surplus.copy()
    labor[0] += capital[0] + 1
    capital[0] = -1
    loss_accounts = replace(raw, labor_compensation=labor, operating_surplus=capital)
    with pytest.raises(ValueError, match="nonpositive labor/capital"):
        calibrate_dynamic(loss_accounts, factor_policy="strict")
    explicit = calibrate_dynamic(loss_accounts, factor_policy="reclassify_losses")
    assert explicit.report["factor_adjusted_cells"] == 1
    assert explicit.report["factor_adjustments"][0]["capital_observed"] == -1
    assert explicit.to_dataframe("factors").loc[0, "cell"] == "AAA:S1"
    for bad in (dict(beta=1.0), dict(delta=0), dict(factor_policy="x"), dict(investment_policy="x"),
                dict(accounting_policy="x"), dict(capital_share_bounds=(.5, .4)), dict(factor_floor=0)):
        with pytest.raises(ValueError):
            calibrate_dynamic(raw, **bad)


def test_accounting_policy_thresholds_on_the_fixture(native_fixture):
    raw, _, _ = native_fixture
    # (a) A material gap (one unit on a 53-unit cell) is rejected under both policies.
    material = raw.TLS.copy()
    material[0] += 1.0
    for policy in ("strict", "reconcile_rounding"):
        with pytest.raises(ValueError, match="do not balance: max relative gap 0.0189"):
            calibrate_dynamic(replace(raw, TLS=material), accounting_policy=policy)
    # (b) A rounding gap of 1e-8 is 1.9e-10 relative to its cell: strict rejects
    # it below accounting_tolerance, reconcile_rounding folds it into the tax
    # wedge and reports it (sum 4.8e-11 of world output, max 1.9e-10 of the
    # largest output, both inside the documented thresholds).
    rounding = raw.TLS.copy()
    rounding[0] += 1e-8
    noisy = replace(raw, TLS=rounding)
    with pytest.raises(ValueError, match="do not balance"):
        calibrate_dynamic(noisy, accounting_policy="strict", accounting_tolerance=1e-11)
    reconciled = calibrate_dynamic(noisy, accounting_policy="reconcile_rounding", accounting_tolerance=1e-11)
    assert reconciled.report["accounting_policy"] == "reconcile_rounding"
    assert reconciled.report["source_cost_reconciliation_total"] == pytest.approx(-1e-8, rel=1e-5)
    assert reconciled.report["source_cost_gap_sum_over_world_output"] == pytest.approx(1e-8 / raw.output.sum(), rel=1e-5)
    assert reconciled.report["source_cost_gap_max_over_largest_output"] == pytest.approx(1e-8 / raw.output.max(), rel=1e-5)
    assert reconciled.report["accounting_largest_output_tolerance"] == 1e-6
    # The implied output-tax wedge y0 - inputs - VA0 absorbs the gap: factor
    # income is untouched and the wedge differs from the recorded tax by -1e-8.
    clean = calibrate_dynamic(raw)
    np.testing.assert_allclose(reconciled.VA0, clean.VA0, atol=0)
    np.testing.assert_allclose(reconciled.tax, clean.tax, atol=0)
    recorded = rounding + raw.production_taxes
    assert reconciled.tax[0] * reconciled.y0[0] - recorded[0] == pytest.approx(-1e-8, rel=1e-5)
    # The default strict tolerance (1e-6) accepts the same gap as source rounding.
    assert calibrate_dynamic(noisy).report["source_cost_gap_max_relative"] < 1e-6
    # (c) The world-output threshold binds: 4.8e-11 exceeds 1e-12.
    with pytest.raises(ValueError, match="do not balance"):
        calibrate_dynamic(noisy, accounting_policy="reconcile_rounding", accounting_tolerance=1e-11,
                          accounting_world_tolerance=1e-12)
    # (d) Invalid world tolerance.
    for bad in (-1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="accounting_world_tolerance"):
            calibrate_dynamic(raw, accounting_world_tolerance=bad)


def test_gdp_is_not_replaced_by_larger_factor_income(native_fixture):
    raw, _, _ = native_fixture
    # A production subsidy increases gross factor receipts while leaving GDP
    # and every source good's observed output unchanged.
    tax, va, labor = raw.TLS.copy(), raw.VA.copy(), raw.labor_compensation.copy()
    tax[0] -= 20
    va[0] += 20
    labor[0] += 20
    subsidized = replace(raw, TLS=tax, VA=va, labor_compensation=labor)
    d = calibrate_dynamic(subsidized)
    observed_gdp = np.bincount(raw.country_index, weights=va + tax) + raw.TFD.sum(axis=1)
    np.testing.assert_allclose(d.Y0, observed_gdp, atol=1e-12)
    assert d.budget_scale[0] > d.Y0[0]


def test_inactive_cells_are_retained_in_output_registry_without_phantom_capital(native_fixture):
    raw, _, _ = native_fixture
    mapping = np.array([0, 1, 3, 4])
    z = np.zeros((6, 6))
    z[np.ix_(mapping, mapping)] = raw.Z
    F = np.zeros((6, 2, 5))
    F[mapping] = raw.F
    vectors = {}
    for name in ("VA", "TLS", "production_taxes", "labor_compensation", "operating_surplus", "output"):
        values = np.zeros(6)
        values[mapping] = getattr(raw, name)
        vectors[name] = values
    # A genuinely positive, very small source cell must not be turned into zero
    # solely because another industry is large.
    F[2, 0, 0] = 1e-10
    vectors["output"][2] = vectors["VA"][2] = 1e-10
    vectors["labor_compensation"][2] = vectors["operating_surplus"][2] = 5e-11
    data = DynamicAccounts.from_arrays(country_codes=raw.countries, sector_codes=("S1", "S2", "S3"), Z=z, F=F,
                                       TFD=raw.TFD, merchandise_mask=np.array([True, False, False]), **vectors)
    d = calibrate_dynamic(data)
    np.testing.assert_array_equal(d.active_indices, [0, 1, 2, 3, 4])
    assert d.report["original_cells"] == 6
    assert d.report["inactive_cells"] == 1
    assert np.all(d.K0 > 0)
    model = DynamicEconomy(d)
    state = model.state(np.zeros(model.n_vars), np.zeros(model.n_vars), DynamicTariff.zero(d))
    expanded = d.expand_cells(state["y"])
    assert expanded.shape == (6,) and expanded[5] == 0
    np.testing.assert_allclose(expanded, data.output, rtol=2e-13, atol=1e-20)
    assert d.cell_labels[2] == "AAA:S3"


def test_cached_path_jvp_includes_terminal_lag_and_preserves_linearization(native_fixture):
    _, d, model = native_fixture
    rng = np.random.default_rng(8471)
    zero = np.zeros(model.n_vars)
    path = rng.normal(0, .002, (4, model.n_vars))
    policies = [DynamicTariff.zero(d), DynamicTariff.zero(d), asymmetric_policy(d), asymmetric_policy(d)]
    terminal = rng.normal(0, .001, model.n_vars)
    problem = PathProblem(model, policies, zero, terminal)
    flat = path.ravel()
    np.testing.assert_allclose(problem.evaluate(flat), path_residual(model, path, policies, zero, terminal).ravel())
    derivative = problem.linearize(flat)
    directions = [rng.normal(size=path.shape), np.zeros_like(path)]
    directions[1][-1, model.k_slice] = rng.normal(size=d.n_cells)
    for direction in directions:
        h = 2e-6
        expected = (path_residual(model, path + h * direction, policies, zero, terminal)
                    - path_residual(model, path - h * direction, policies, zero, terminal)).ravel() / (2 * h)
        # Updating the object's residual cache must not mutate an already built
        # Newton operator's captured evaluation point.
        problem.evaluate((path + .0001 * direction).ravel())
        np.testing.assert_allclose(derivative(direction.ravel()), expected, rtol=2e-6, atol=2e-7)
    z = rng.normal(0, .002, model.n_vars)
    direction = rng.normal(size=model.n_vars)
    policy = policies[-1]
    h = 2e-6
    expected = (model.equations(z + h * direction, z + h * direction, z + h * direction, policy, policy)
                - model.equations(z - h * direction, z - h * direction, z - h * direction, policy, policy)) / (2 * h)
    np.testing.assert_allclose(steady_linearization(model, z, policy)(direction), expected, rtol=2e-6, atol=2e-7)


def test_no_shock_transition_is_stationary(native_fixture):
    _, d, model = native_fixture
    result = solve_dynamic_transition(model, [DynamicTariff.zero(d)] * 6, tol=1e-11)
    assert result.converged and isinstance(result, DynamicTransitionResult)
    np.testing.assert_allclose(result.path, 0, atol=1e-13)
    np.testing.assert_allclose(np.array([s["K"] for s in result.states]), np.tile(d.K0, (6, 1)))
    np.testing.assert_allclose(result.welfare.consumption_equivalent_pct, 0, atol=1e-13)
    assert result.terminal_max_log_gap < 1e-13
    assert result.stability is not None and result.stability.determinate


def test_anticipated_transition_has_predetermined_capital_and_responds_before_tariff(native_fixture):
    _, d, model = native_fixture
    zero, policy = DynamicTariff.zero(d), asymmetric_policy(d)
    terminal = solve_dynamic_steady_state(model, policy, tol=1e-11)
    policies = tariff_path(zero, policy, horizon=24, announcement=2)
    result = solve_dynamic_transition(model, policies, terminal=terminal, tol=1e-10)
    np.testing.assert_allclose(result.states[0]["K"], d.K0, atol=1e-13)
    assert np.max(np.abs(result.states[0]["C"] / d.C0 - 1)) > 1e-4
    assert np.max(np.abs(result.states[0]["I"] / d.I0 - 1)) > 1e-4
    for t, state in enumerate(result.states):
        lead = result.states[t + 1] if t + 1 < len(result.states) else model.state(terminal.z, result.path[-1], policy)
        check_independent_accounts(d, model, state, lead, policies[t])
        if t:
            np.testing.assert_allclose(state["K"], result.states[t - 1]["Knext"], atol=1e-13)
    np.testing.assert_allclose(result.boundary_state["K"], result.states[-1]["Knext"], atol=1e-13)
    assert max(result.certificate.values()) <= 1e-8 and set(result.certificate) == set(terminal.certificate)


def test_welfare_tail_requires_separate_horizon_check(native_fixture):
    _, d, model = native_fixture
    zero, policy = DynamicTariff.zero(d), asymmetric_policy(d)
    terminal = solve_dynamic_steady_state(model, policy, tol=1e-11)
    solutions = [solve_dynamic_transition(model, tariff_path(zero, policy, horizon=h, announcement=2),
                                          terminal=terminal, tol=1e-10) for h in (80, 160, 320)]
    first = compare_horizons(solutions[0], solutions[1], periods=20, tolerance=1e-5)
    assert first.window_passed
    # Small equation errors and stable early decisions do not certify the tail.
    assert first.welfare_max_difference_pp > 1e-3 and not first.passed and first.status == "not_verified"
    second = compare_horizons(solutions[1], solutions[2], periods=20, tolerance=1e-5)
    assert second.window_passed and second.welfare_max_difference_pp < 1e-4
    assert np.max(np.abs(solutions[2].path[-1] - terminal.z)) < 1e-4
    assert second.terminal_passed and second.passed and second.status == "verified"
    # The explicitly weaker discounted-wealth criterion is a different check:
    # here the endpoint shadow value (about 1e-5) still exceeds its tolerance.
    weaker = compare_horizons(solutions[1], solutions[2], terminal_check="discounted_wealth")
    assert weaker.max_discounted_wealth == pytest.approx(np.max(solutions[2].discounted_endpoint_capital_value))
    assert not weaker.passed and "no rigorous infinite-tail error bound" in weaker.validation_scope
    frame = second.to_dataframe()
    assert list(frame.index) == ["early_window_max_log_difference", "welfare_max_difference_pp",
                                 "terminal_max_log_gap", "max_discounted_wealth"]
    assert "status verified" in second.summary() and "terminal_max_log_gap" in second.to_markdown()
    assert "terminal\\_max\\_log\\_gap" in second.to_latex() and "welfare" in second.to_typst()


def test_frictionless_capital_q_equals_investment_price(native_fixture):
    _, d, _ = native_fixture
    model = DynamicEconomy(d, adjustment_cost=0, risk_aversion=.5)
    result = solve_dynamic_steady_state(model, asymmetric_policy(d), tol=1e-11)
    state = result.state
    np.testing.assert_allclose(state["q"], state["PI"][d.country], atol=1e-13)
    assert np.max(np.abs(state["q"] - 1)) > 1e-3
    check_independent_accounts(d, model, state, state, asymmetric_policy(d))


def test_stationary_condensation_matches_full_solution_and_finite_difference_jacobian(native_fixture):
    _, d, model = native_fixture
    policy = asymmetric_policy(d)
    full = solve_dynamic_steady_state(model, policy, method="full", tol=1e-11)
    condensed = solve_dynamic_steady_state(model, policy, method="condensed", tol=1e-11)
    assert full.method == "full" and condensed.method == "condensed"
    np.testing.assert_allclose(condensed.z, full.z, rtol=1e-8, atol=2e-10)
    check_independent_accounts(d, model, condensed.state, condensed.state, policy)
    auto = solve_dynamic_steady_state(model, policy, tol=1e-11)
    assert auto.method == "full"  # n_vars = 8 <= 240
    rng = np.random.default_rng(7144)
    u = rng.normal(0, .01, 2 * d.n_countries)
    problem = StationaryProblem(model, policy)
    derivative = problem.jacobian(u).copy()
    h = 2e-6
    numeric = np.column_stack([(problem.evaluate(u + h * v) - problem.evaluate(u - h * v)) / (2 * h)
                               for v in np.eye(len(u))])
    np.testing.assert_allclose(derivative, numeric, rtol=1e-6, atol=5e-8)
    with pytest.raises(ValueError):
        solve_condensed_steady(model, policy, start=np.zeros(3))
    with pytest.raises(ValueError):
        solve_dynamic_steady_state(model, policy, method="nope")


def test_batched_states_and_derivatives_preserve_all_date_specific_policies(native_fixture):
    _, d, model = native_fixture
    rng = np.random.default_rng(4129)
    current, lag, direction, lag_direction = [rng.normal(0, .002, (5, model.n_vars)) for _ in range(4)]
    zero, tariff = model.zero_policy(), asymmetric_policy(d)
    policies = [zero, tariff, zero, tariff, tariff]
    batch = model.states(current, lag, policies)
    changes = model.states_jvp(current, lag, direction, lag_direction, policies, batch)
    for t in range(5):
        scalar = model.state(current[t], lag[t], policies[t])
        scalar_change = model.state_jvp(current[t], lag[t], direction[t], lag_direction[t], policies[t])
        for name in scalar:
            np.testing.assert_allclose(batch[t][name], scalar[name], rtol=1e-12, atol=1e-12,
                                       err_msg=f"Batched date {t}, {name}")
        for name in scalar_change:
            np.testing.assert_allclose(changes[t][name], scalar_change[name], rtol=1e-12, atol=1e-12,
                                       err_msg=f"Batched derivative date {t}, {name}")
    assert len(model._networks) == 2  # one factorisation per distinct fingerprint


def test_baseline_anchor_preserves_tiny_positive_labor_share(native_fixture):
    raw, _, _ = native_fixture
    labor, capital = raw.labor_compensation.copy(), raw.operating_surplus.copy()
    total = labor[0] + capital[0]
    labor[0], capital[0] = total * 1e-9, total * (1 - 1e-9)
    d = calibrate_dynamic(replace(raw, labor_compensation=labor, operating_surplus=capital))
    model = DynamicEconomy(d)
    zero, policy = np.zeros(model.n_vars), model.zero_policy()
    assert 0 < 1 - d.alpha[0] < 2e-9
    assert d.report["factor_adjusted_cells"] == 0
    assert np.max(np.abs(model.equations(zero, zero, zero, policy, policy))) < 2e-12
    for state in model.states(np.tile(zero, (3, 1)), np.tile(zero, (3, 1)), [policy] * 3):
        np.testing.assert_allclose(state["y"], d.y0, rtol=1e-14)
        np.testing.assert_allclose(state["R"], d.R0, rtol=2e-12)
        np.testing.assert_allclose(state["p"], 1, atol=2e-12)


def test_horizon_comparison_rejects_changed_boundaries_or_announcements(native_fixture):
    _, d, model = native_fixture
    zero = model.zero_policy()
    short = solve_dynamic_transition(model, [zero] * 6, tol=1e-11)
    long = solve_dynamic_transition(model, [zero] * 8, tol=1e-11)
    assert compare_horizons(short, long, periods=4).passed
    for name in ("initial", "terminal"):
        changed = getattr(long, name).copy()
        changed[-1] += .001
        with pytest.raises(ValueError, match=f"identical {name} boundaries"):
            compare_horizons(short, replace(long, **{name: changed}), periods=4)
    with pytest.raises(ValueError, match="complete policy schedules"):
        compare_horizons(short, replace(long, policies=long.policies[:-1]), periods=4)
    # A newly announced shock after the short run ends is a different economic
    # experiment even though its early policy dates and terminal state match.
    different_policy = list(long.policies)
    different_policy[6] = asymmetric_policy(d)
    with pytest.raises(ValueError, match="same announced policy"):
        compare_horizons(short, replace(long, policies=tuple(different_policy)), periods=4)
    for bad_tolerance in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError, match="finite and positive"):
            compare_horizons(short, long, periods=4, tolerance=bad_tolerance)
    with pytest.raises(ValueError, match="longer horizon"):
        compare_horizons(long, short, periods=4)
    with pytest.raises(ValueError, match="terminal_check"):
        compare_horizons(short, long, periods=4, terminal_check="hope")
    with pytest.raises(ValueError, match="welfare"):
        compare_horizons(short, replace(long, welfare=None), periods=4)


def test_redundant_budget_row_choice_preserves_equilibrium_and_all_accounts(native_fixture):
    _, d, _ = native_fixture
    policy = asymmetric_policy(d)
    stationary, paths = [], []
    for omitted in (0, 1):
        model = DynamicEconomy(d)
        model.omitted_budget_index = omitted
        assert model.numeraire_index == d.n_countries - 1
        ss = solve_condensed_steady(model, policy, tol=1e-11)
        check_independent_accounts(d, model, ss.state, ss.state, policy)
        policy_path = [model.zero_policy()] * 2 + [policy] * 10
        run = solve_dynamic_transition(model, policy_path, terminal=ss, tol=1e-10)
        # This explicitly includes whichever country's budget is absent from
        # the root system, while maintaining the same wage numeraire.
        assert run.certificate["budget_all"] < 1e-9 and run.certificate["current_account"] < 1e-9
        np.testing.assert_allclose(np.array([s["w"][-1] for s in run.states]), 1, atol=1e-10)
        for t, state in enumerate(run.states):
            lead = run.states[t + 1] if t + 1 < len(run.states) else run.boundary_state
            check_independent_accounts(d, model, state, lead, policy_path[t])
        stationary.append(ss.z)
        paths.append(run.path)
    np.testing.assert_allclose(stationary[0], stationary[1], rtol=0, atol=2e-9)
    np.testing.assert_allclose(paths[0], paths[1], rtol=0, atol=2e-9)


def test_discounted_capital_diagnostic_has_correct_timing_and_numeraire(native_fixture):
    _, d, model = native_fixture
    horizon = 7
    arrays = {"z": np.zeros((horizon, model.n_vars)), "C": np.tile(d.C0, (horizon, 1)),
              "PC": np.tile(d.PC0, (horizon, 1)), "Knext": np.tile(d.K0, (horizon, 1)),
              "q": np.tile(d.PI0[d.country], (horizon, 1))}
    expected = np.zeros(d.n_countries)
    for j in range(d.n_cells):
        c = d.country[j]
        expected[c] += d.beta ** (horizon - 1) * d.PI0[c] * d.K0[j] / (d.PC0[c] * d.C0[c])
    actual = discounted_endpoint_capital_value(arrays, d, 2)
    np.testing.assert_allclose(actual, expected, rtol=1e-14)
    changed_prices = arrays | {"q": arrays["q"] * 3, "PC": arrays["PC"] * 3}
    np.testing.assert_allclose(discounted_endpoint_capital_value(changed_prices, d, 2), expected, rtol=1e-14)
    changed_units = arrays | {"C": arrays["C"] * 1e6, "Knext": arrays["Knext"] * 1e6}
    np.testing.assert_allclose(discounted_endpoint_capital_value(changed_units, replace(d, C0=d.C0 * 1e6), 2),
                               expected, rtol=1e-14)
    # A solved no-shock path of the same length reproduces the same diagnostic.
    run = solve_dynamic_transition(model, [model.zero_policy()] * horizon, tol=1e-11)
    np.testing.assert_allclose(run.discounted_endpoint_capital_value, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
def test_tariff_constructors_validate_and_fingerprint(native_fixture):
    _, d, model = native_fixture
    zero = DynamicTariff.zero(d)
    assert zero.is_zero() and zero.label == "baseline" and len(zero.fingerprint) == 64
    assert zero.fingerprint == DynamicTariff.build(np.zeros((4, 2))).fingerprint
    uniform = DynamicTariff.uniform(d, "AAA", .1)
    assert uniform.rates[d.country == 1, 0].tolist() == [.1, .1]
    assert np.all(uniform.rates[d.country == 0] == 0) and np.all(uniform.rates[:, 1] == 0)
    np.testing.assert_array_equal(uniform.consumption_rates, uniform.rates)
    goods = DynamicTariff.uniform(d, 0, .1, sectors="merchandise", uses=("intermediate",))
    assert goods.rates[:, 0].tolist() == [0, 0, .1, 0] and goods.consumption_rates.max() == 0
    by_code = DynamicTariff.uniform(d, "AAA", .1, sectors=("S1",), exporters=("BBB",))
    np.testing.assert_array_equal(by_code.rates, goods.rates)
    table = uniform.to_dataframe(d)
    assert set(table["source"]) == {"BBB:S1", "BBB:S2"} and set(table["destination"]) == {"AAA"}
    assert "BBB:S1" in uniform.to_markdown(d) and "BBB:S1" in uniform.to_latex(d) and "BBB:S1" in uniform.to_typst(d)
    explicit = DynamicTariff.from_rates(d, uniform.rates, consumption_rates=uniform.rates * 0, label="explicit")
    assert explicit.fingerprint != uniform.fingerprint and explicit.consumption_rates.max() == 0
    for bad in (dict(importer="ZZZ", rate=.1), dict(importer="AAA", rate=-1.5), dict(importer="AAA", rate=.1, uses=("x",)),
                dict(importer="AAA", rate=.1, sectors=("S9",)), dict(importer="AAA", rate=.1, exporters=("ZZZ",)),
                dict(importer=5, rate=.1), dict(importer=1.5, rate=.1)):
        with pytest.raises((ValueError, TypeError)):
            DynamicTariff.uniform(d, **bad)
    domestic = np.zeros((4, 2))
    domestic[0, 0] = .1
    with pytest.raises(ValueError, match="domestic"):
        DynamicTariff.from_rates(d, domestic)
    with pytest.raises(ValueError):
        DynamicTariff.from_rates(d, np.zeros((3, 2)))
    with pytest.raises(ValueError, match="exceed -1"):
        DynamicTariff.build(np.full((4, 2), -1.0))
    with pytest.raises(ValueError):
        DynamicTariff.build(np.zeros(4))
    with pytest.raises(TypeError):
        model.state(np.zeros(model.n_vars), np.zeros(model.n_vars), "baseline")
    # The fingerprint is derived from the tables, never passed in: a direct
    # construction gets the same key as build(); invalid tables get none and
    # are refused on use.
    direct = DynamicTariff(np.zeros((4, 2)), np.zeros((4, 2)), None)
    assert direct.fingerprint == zero.fingerprint and not direct.rates.flags.writeable
    np.testing.assert_array_equal(direct.investment_rates, direct.rates)
    with pytest.raises(TypeError):
        DynamicTariff(np.zeros((4, 2)), np.zeros((4, 2)), np.zeros((4, 2)), fingerprint="x")  # type: ignore[call-arg]
    invalid = DynamicTariff(np.full((4, 2), -2.0), np.zeros((4, 2)), np.zeros((4, 2)))
    assert invalid.fingerprint == ""
    with pytest.raises(ValueError, match="no fingerprint"):
        model.state(np.zeros(model.n_vars), np.zeros(model.n_vars), invalid)
    subsidy = DynamicTariff.uniform(d, "AAA", -.05)
    assert subsidy.rates.min() == -.05 and len(subsidy.fingerprint) == 64
    with pytest.raises(ValueError, match="shape"):
        model.state(np.zeros(model.n_vars), np.zeros(model.n_vars), DynamicTariff.build(np.zeros((3, 2))))
    no_mask = replace(d, merchandise_mask=None)
    with pytest.raises(ValueError, match="merchandise mask"):
        DynamicTariff.uniform(no_mask, "AAA", .1, sectors="merchandise")


def test_uniform_tariff_rejects_an_empty_selection(native_fixture):
    _, d, _ = native_fixture
    # The importer listed as the only exporter, an all-false goods mask and an
    # empty exporter tuple would all yield a zero policy under a tariff label.
    with pytest.raises(ValueError, match="contains no foreign source cells"):
        DynamicTariff.uniform(d, "AAA", .1, exporters=("AAA",))
    with pytest.raises(ValueError, match="contains no foreign source cells"):
        DynamicTariff.uniform(d, "AAA", .1, exporters=())
    no_goods = replace(d, merchandise_mask=np.array([False, False]))
    with pytest.raises(ValueError, match="contains no foreign source cells"):
        DynamicTariff.uniform(no_goods, "BBB", .1, sectors="merchandise")
    # A zero rate on a nonempty selection is a legitimate (labelled) baseline.
    assert DynamicTariff.uniform(d, "AAA", 0.0).is_zero()
    assert not d.merchandise_mask.flags.writeable


def test_replaced_or_forged_policies_never_reuse_another_policys_prices(native_fixture):
    _, d, model = native_fixture
    policy = asymmetric_policy(d)
    original = solve_dynamic_steady_state(model, policy, tol=1e-11)   # caches the network of `policy`
    doubled = replace(policy, rates=policy.rates * 2, consumption_rates=policy.consumption_rates * 2,
                      investment_rates=policy.investment_rates * 2, label="doubled")
    honest = DynamicTariff.build(policy.rates * 2, consumption_rates=policy.consumption_rates * 2,
                                 investment_rates=policy.investment_rates * 2, label="doubled")
    assert doubled.fingerprint == honest.fingerprint != policy.fingerprint
    assert not doubled.rates.flags.writeable and not doubled.investment_rates.flags.writeable
    via_replace = solve_dynamic_steady_state(model, doubled, tol=1e-11)
    fresh_model = DynamicEconomy(d)
    fresh = solve_dynamic_steady_state(fresh_model, honest, tol=1e-11)
    np.testing.assert_allclose(via_replace.z, fresh.z, rtol=0, atol=1e-12)
    np.testing.assert_allclose(via_replace.state["TR"], fresh.state["TR"], rtol=1e-12)
    assert np.max(np.abs(via_replace.z - original.z)) > 1e-3
    zero = model.zero_policy()
    run = solve_dynamic_transition(model, tariff_path(zero, doubled, horizon=24, announcement=2),
                                   terminal=via_replace, tol=1e-10)
    reference = solve_dynamic_transition(fresh_model, tariff_path(zero, honest, horizon=24, announcement=2),
                                         terminal=fresh, tol=1e-10)
    np.testing.assert_allclose(run.path, reference.path, rtol=0, atol=1e-12)
    np.testing.assert_allclose(run.welfare.consumption_equivalent_pct, reference.welfare.consumption_equivalent_pct,
                               rtol=0, atol=1e-12)
    # A distinct object with equal tables is verified once and shares the factorisation.
    alias = DynamicTariff.build(policy.rates, consumption_rates=policy.consumption_rates,
                                investment_rates=policy.investment_rates, label="alias")
    model.state(original.z, original.z, alias)
    assert id(alias) in model._network(policy).aliases
    # Tables altered behind the constructor's back never select another policy's prices.
    forged = DynamicTariff.build(policy.rates * 3, label="forged")
    object.__setattr__(forged, "fingerprint", policy.fingerprint)
    with pytest.raises(ValueError, match="fingerprint of a different policy"):
        model.state(np.zeros(model.n_vars), np.zeros(model.n_vars), forged)


def test_tariff_path_schedules_and_temporary_policy_returns_to_baseline(native_fixture):
    _, d, model = native_fixture
    zero, shock = DynamicTariff.zero(d), asymmetric_policy(d)
    path = tariff_path(zero, shock, horizon=10, announcement=2, duration=5)
    assert [p is shock for p in path] == [False, False, True, True, True, True, True, False, False, False]
    assert tariff_path(zero, shock, horizon=4)[0] is shock
    for bad in (dict(horizon=0), dict(horizon=5, announcement=5), dict(horizon=5, announcement=2, duration=3),
                dict(horizon=5, announcement=-1), dict(horizon=5, duration=0), dict(horizon=2.5)):
        with pytest.raises(ValueError):
            tariff_path(zero, shock, **bad)
    with pytest.raises(TypeError):
        tariff_path(zero, "shock", horizon=3)
    # Temporary policy (dates 2-6) with the baseline as terminal state.
    result = solve_dynamic_transition(model, tariff_path(zero, shock, horizon=30, announcement=2, duration=5), tol=1e-10)
    assert result.converged and np.allclose(result.terminal, 0, atol=1e-12)
    assert result.metadata["terminal_policy"] == "baseline"
    ratio = result.consumption_ratio()
    # Consumption moves while the duty is in force and decays back towards the
    # baseline afterwards (slowly: the fixture's slowest stable root is 0.979).
    assert np.max(np.abs(ratio[4] - 1)) > 3e-3 and np.max(np.abs(ratio[-1] - 1)) < 2e-3
    assert np.max(np.abs(ratio[-1] - 1)) < np.max(np.abs(ratio[10] - 1)) < np.max(np.abs(ratio[4] - 1))
    assert result.terminal_max_log_gap < 5e-3
    assert result.welfare is not None and result.welfare.tail_included


# ---------------------------------------------------------------------------
# Result objects, failure contracts and diagnostics
# ---------------------------------------------------------------------------
def test_steady_state_and_transition_results_render_and_plot_headlessly(native_fixture):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, d, model = native_fixture
    policy = asymmetric_policy(d)
    steady = solve_dynamic_steady_state(model, policy, tol=1e-11)
    assert isinstance(steady, DynamicSteadyStateResult) and not steady.z.flags.writeable
    country = steady.to_dataframe()
    assert list(country.index) == ["AAA", "BBB"] and "consumption_change_pct" in country.columns
    cells = steady.to_dataframe(level="cell")
    assert list(cells.index) == list(d.cell_labels) and "tobin_q" in cells.columns
    assert "analytic tariff" in steady.summary()
    for text in (steady.to_markdown(), steady.to_latex(), steady.to_typst(), steady.to_markdown(level="cell")):
        assert "AAA" in text
    with pytest.raises(ValueError):
        steady.to_dataframe(level="planet")
    run = solve_dynamic_transition(model, tariff_path(DynamicTariff.zero(d), policy, horizon=12, announcement=1),
                                   terminal=steady, tol=1e-10)
    wide = run.to_dataframe()
    assert wide.shape == (12, 10) and wide.index.name == "date" and "C:AAA" in wide.columns
    cell_wide = run.to_dataframe(level="cell", variables=("K", "x"))
    assert cell_wide.shape == (12, 8) and "x:BBB:S2" in cell_wide.columns
    with pytest.raises(ValueError):
        run.to_dataframe(variables=("nope",))
    assert "T=12" in run.summary() and "CE pct" in run.summary() and run.z is run.path
    assert "C:AAA" in run.to_markdown() and "C:AAA" in run.to_typst()
    fig, ax = plt.subplots()
    assert run.plot(ax=ax, countries=("AAA",)) is ax
    assert len(ax.lines) == 2  # one country plus the zero line
    figure = run.plot(variable="Inational")
    assert figure is not None
    plt.close("all")
    with pytest.raises(ValueError):
        run.plot(ax=ax, variable="q")
    with pytest.raises(ValueError):
        run.plot(ax=ax, countries=("ZZZ",))
    assert run.metadata["policy_labels"][0] == "baseline" and run.metadata["determinacy_checked"]


def test_results_hold_read_only_state_payloads(native_fixture):
    _, d, model = native_fixture
    policy = asymmetric_policy(d)
    for method in ("full", "condensed"):
        steady = solve_dynamic_steady_state(model, policy, method=method, tol=1e-11)
        assert all(not v.flags.writeable for v in steady.state.values() if isinstance(v, np.ndarray))
        assert not steady.benchmark["Inational0"].flags.writeable
    assert not d.Inational0.flags.writeable
    run = solve_dynamic_transition(model, tariff_path(DynamicTariff.zero(d), policy, horizon=8, announcement=1),
                                   terminal=steady, tol=1e-10)
    for state in run.states + (run.boundary_state,):
        assert all(not v.flags.writeable for v in state.values() if isinstance(v, np.ndarray))
    assert not run.discounted_endpoint_capital_value.flags.writeable
    np.testing.assert_allclose(run.discounted_endpoint_capital_value,
                               discounted_endpoint_capital_value(run, d, model.risk_aversion), rtol=1e-14)
    with pytest.raises((AttributeError, TypeError)):
        run.discounted_endpoint_capital_value = np.zeros(2)  # type: ignore[misc]


def test_summaries_list_only_the_largest_consumption_equivalents():
    pct = np.array([0.5, -3.0, 0.1, 2.0, -0.2, 1.0, 0.05])
    codes = tuple("ABCDEFG")
    result = consumption_equivalent_welfare(np.exp(np.log1p(pct / 100))[None, :], baseline_consumption=np.ones(7),
                                            beta=.96, risk_aversion=2, country_codes=codes)
    text = result.summary()
    assert "B=-3.0000, D=+2.0000, F=+1.0000, A=+0.5000, E=-0.2000 and 2 more" in text
    assert "C=" not in text and "G=" not in text and "to_dataframe()" in text
    two = consumption_equivalent_welfare(np.ones((1, 2)), baseline_consumption=np.ones(2), beta=.96, risk_aversion=2,
                                         country_codes=("AAA", "BBB"))
    assert "more" not in two.summary() and "AAA=" in two.summary()


def test_package_import_leaves_matplotlib_out_of_sys_modules():
    code = ("import sys; import puremacro.trade.dynamic as m; "
            "print(int('matplotlib' in sys.modules), int('torch' in sys.modules))")
    out = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True, cwd=str(REPO),
                         env={**dict(__import__('os').environ), "PYTHONPATH": str(REPO), **ENV_THREADS})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "0 0"


def test_failed_steady_state_raises_with_iterate_and_never_returns_converged(native_fixture):
    _, d, model = native_fixture
    huge = DynamicTariff.uniform(d, "AAA", 1.0)
    with pytest.raises(DynamicSolveError) as excinfo:
        solve_dynamic_steady_state(model, huge, tol=1e-10)
    error = excinfo.value
    assert error.residual is not None and error.iterate is not None and error.iterate.shape == (model.n_vars,)
    assert isinstance(error.history, tuple)
    # The line-search failure names the equation with the largest residual.
    assert "Largest stationary residual" in str(error) and error.location["method"] == "full"
    assert error.location["block"] in {"labor_clearing", "budget", "numeraire", "capital_euler"}
    assert error.location["country"] in d.countries
    with pytest.raises(DynamicSolveError) as condensed:
        solve_dynamic_steady_state(model, huge, tol=1e-10, method="condensed")
    assert condensed.value.iterate.shape == (2 * model.nc,)
    assert condensed.value.location["method"] == "condensed" and condensed.value.location["country"] in d.countries
    assert condensed.value.location["block"] in {"investment_price", "budget", "numeraire"}
    with pytest.raises(DynamicSolveError, match="iteration limit"):
        solve_dynamic_steady_state(model, asymmetric_policy(d), tol=1e-14, max_iter=1, method="full",
                                   start=np.full(model.n_vars, .05))


def test_determinacy_gate_raises_a_structured_error_on_an_indeterminate_fixture(native_fixture):
    _, d, model = native_fixture
    extreme = DynamicTariff.uniform(d, "AAA", 8.0)
    steady = solve_dynamic_steady_state(model, extreme, tol=1e-10)
    report = stability_report(model, steady)
    assert not report.determinate and report.n_stable == 3 and report.n_predetermined == 4
    with pytest.raises(DynamicDeterminacyError) as excinfo:
        solve_dynamic_transition(model, tariff_path(model.zero_policy(), extreme, horizon=30, announcement=2),
                                 terminal=steady, tol=1e-9)
    assert isinstance(excinfo.value.report, DynamicStabilityResult) and excinfo.value.report.n_stable == 3
    assert isinstance(excinfo.value, DynamicSolveError)
    assert extreme.fingerprint in model._stability_cache
    model.clear_caches()
    assert not model._stability_cache and not model._networks and not hasattr(model, "_capital_preconditioner")
    with pytest.raises(ValueError, match="max_order"):
        stability_report(model, steady, max_order=4)
    with pytest.raises(ValueError):
        stability_report(model, steady, tolerance=0)
    with pytest.raises(ValueError):
        stability_report(model, np.zeros(3))


def test_stability_fixture_is_determinate_at_baseline_and_tariff_states(native_fixture):
    _, d, model = native_fixture
    baseline = stability_report(model, np.zeros(model.n_vars))
    assert baseline.determinate and (baseline.n_stable, baseline.n_predetermined, baseline.n_unit) == (4, 4, 0)
    assert baseline.slowest_stable_root == pytest.approx(0.97866, abs=2e-4)
    assert baseline.n_variables == 12 and baseline.eigenvalues.shape == (12,) and not baseline.eigenvalues.flags.writeable
    tariff = stability_report(model, solve_dynamic_steady_state(model, asymmetric_policy(d), tol=1e-11))
    assert tariff.determinate and tariff.n_stable == 4 and tariff.metadata["policy"] == "analytic tariff"
    countries = baseline.to_dataframe()
    assert list(countries.index) == ["AAA", "BBB"] and countries["loading_share"].sum() == pytest.approx(1.0)
    loadings = baseline.to_dataframe("loadings")
    assert loadings.loc[0, "loading"] == 1.0 and loadings.loc[0, "variable"].startswith("K:")
    assert "determinate" in baseline.summary() and "AAA" in baseline.to_markdown()
    with pytest.raises(ValueError):
        baseline.to_dataframe("nope")


def test_stability_report_refuses_points_that_are_not_stationary_under_the_policy(native_fixture):
    _, d, model = native_fixture
    tariff = DynamicTariff.uniform(d, "AAA", .3)
    steady = solve_dynamic_steady_state(model, tariff, tol=1e-11)
    with pytest.raises(ValueError, match="not stationary under policy 'baseline'"):
        stability_report(model, np.array(steady.z))           # policy omitted: the baseline is assumed
    explicit = stability_report(model, np.array(steady.z), tariff)
    carried = stability_report(model, steady)                 # a result carries its own policy
    assert explicit.determinate and explicit.metadata["stationary_residual"] <= 1e-8
    assert carried.slowest_stable_root == explicit.slowest_stable_root
    with pytest.raises(ValueError, match="stationarity_tol"):
        stability_report(model, np.full(model.n_vars, .01))
    loose = stability_report(model, np.full(model.n_vars, .01), stationarity_tol=1.0)   # explicit opt-out
    assert loose.metadata["stationary_residual"] > 1e-3 and loose.metadata["stationarity_tol"] == 1.0
    for bad in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError):
            stability_report(model, steady, stationarity_tol=bad)


def test_run_horizon_ladder_reports_the_fixture_horizon_pins(native_fixture):
    _, d, model = native_fixture
    zero, policy = DynamicTariff.zero(d), asymmetric_policy(d)
    terminal = solve_dynamic_steady_state(model, policy, tol=1e-11)
    seen = []
    ladder = run_horizon_ladder(model, lambda h: tariff_path(zero, policy, horizon=h, announcement=2),
                                horizons=(24, 48, 96), terminal=terminal, tol=1e-10, progress=seen.append)
    assert isinstance(ladder, HorizonLadderResult) and ladder.horizons == (24, 48, 96)
    assert ladder.status == "solved_horizon_not_validated" and ladder.accepted is None
    assert [c.max_log_difference for c in ladder.comparisons] == pytest.approx([3.953e-3, 7.08e-5], rel=2e-2)
    assert [c.welfare_max_difference_pp for c in ladder.comparisons] == pytest.approx([0.156, 0.0388], rel=3e-2)
    assert "extend the horizon" in ladder.validation_scope
    frame = ladder.to_dataframe()
    assert frame["passed"].tolist() == [False, False] and frame["long"].tolist() == [48, 96]
    assert any(item.get("phase") == "horizon_comparison" for item in seen)
    assert "solved_horizon_not_validated" in ladder.summary() and "24" in ladder.to_markdown()
    # Warm-started longer horizons agree with cold solves of the same problem.
    cold = solve_dynamic_transition(model, tariff_path(zero, policy, horizon=48, announcement=2), terminal=terminal, tol=1e-10)
    np.testing.assert_allclose(ladder.solutions[1].path, cold.path, atol=1e-8)
    with pytest.raises(ValueError):
        run_horizon_ladder(model, lambda h: [zero] * h, horizons=(10,))
    with pytest.raises(ValueError, match="one policy per date"):
        run_horizon_ladder(model, lambda h: [zero] * (h + 1), horizons=(6, 8))
    with pytest.raises(ValueError, match="same terminal policy"):
        run_horizon_ladder(model, lambda h: [zero] * (h - 1) + [policy if h == 8 else zero], horizons=(6, 8))


def test_default_start_falls_back_to_a_linear_blend_and_names_infeasible_horizons(native_fixture):
    _, d, model = native_fixture
    extreme = DynamicTariff.uniform(d, "AAA", 10.0)
    steady = solve_dynamic_steady_state(model, extreme, tol=1e-10)
    initial, terminal = np.zeros(model.n_vars), np.array(steady.z)
    # The IO exponential blend disinvests at date 0 (x <= 0 at BBB:S1) for this duty.
    horizon = 60
    fraction = (1 - np.exp(-np.arange(1, horizon + 1) / 12))[:, None]
    exponential = initial + fraction * (terminal - initial)
    assert _installation_rates(model, exponential, initial, terminal).min() <= 0
    start, kind = _default_start(model, initial, terminal, horizon)
    assert kind == "linear_blend" and _installation_rates(model, start, initial, terminal).min() > 0
    # Over 20 dates even the linear blend cannot keep investment interior: no
    # path of that horizon can, and the solver says so instead of a bare domain error.
    with pytest.raises(DynamicSolveError, match="default initial guess cannot keep investment interior over 20") as excinfo:
        solve_dynamic_transition(model, tariff_path(model.zero_policy(), extreme, horizon=20), terminal=steady,
                                 require_determinacy=False)
    location = excinfo.value.location
    assert location["default_start"] == "linear_blend" and location["cell"] == "BBB:S1" and location["country"] == "BBB"
    assert location["average_log_capital_step"] < location["boundary_log_growth"] < 0
    assert excinfo.value.iterate.shape == (20, model.n_vars)
    # Ordinary runs keep the IO start and record which guess was used.
    zero, policy = model.zero_policy(), asymmetric_policy(d)
    run = solve_dynamic_transition(model, tariff_path(zero, policy, horizon=8), tol=1e-10)
    assert run.metadata["start"] == "exponential_blend"
    again = solve_dynamic_transition(model, tariff_path(zero, policy, horizon=8), tol=1e-10, start=np.array(run.path))
    assert again.metadata["start"] == "user" and again.iterations == 0


def test_horizon_comparison_requires_the_same_economy(native_fixture):
    _, d, model = native_fixture
    zero, policy = model.zero_policy(), asymmetric_policy(d)
    # The steady state does not depend on the adjustment cost (x = delta), so
    # both economies share identical boundaries: only the economy differs.
    steady = solve_dynamic_steady_state(model, policy, tol=1e-11)
    stiff = DynamicEconomy(d, adjustment_cost=4.0)
    short = solve_dynamic_transition(model, tariff_path(zero, policy, horizon=10, announcement=2), terminal=steady,
                                     tol=1e-10)
    long = solve_dynamic_transition(stiff, tariff_path(zero, policy, horizon=20, announcement=2), terminal=steady,
                                    tol=1e-10)
    assert short.metadata["economy_fingerprint"] != long.metadata["economy_fingerprint"]
    with pytest.raises(ValueError, match="same economy"):
        compare_horizons(short, long, periods=5)
    # The omitted redundant budget is not part of the economy: Walras' law.
    other_row = DynamicEconomy(d)
    other_row.omitted_budget_index = 1 - other_row.omitted_budget_index
    same = solve_dynamic_transition(other_row, tariff_path(zero, policy, horizon=20, announcement=2), terminal=steady,
                                    tol=1e-10)
    assert same.metadata["economy_fingerprint"] == short.metadata["economy_fingerprint"]
    assert compare_horizons(short, same, periods=5).max_log_difference < 1e-2


def test_structured_errors_survive_pickling_for_worker_processes(native_fixture):
    _, d, model = native_fixture
    error = DynamicSolveError("boom", residual=.5, iterate=np.ones(3), location={"date": 2},
                              history=[{"iteration": 1}], certificate={"root": .1})
    back = pickle.loads(pickle.dumps(error))
    assert str(back) == "boom" and back.residual == .5 and back.location == {"date": 2}
    assert back.iterate.tolist() == [1.0, 1.0, 1.0] and back.history == ({"iteration": 1},)
    report = stability_report(model, np.zeros(model.n_vars))
    determinacy = pickle.loads(pickle.dumps(DynamicDeterminacyError("indeterminate", report)))
    assert isinstance(determinacy, DynamicSolveError) and str(determinacy) == "indeterminate"
    assert determinacy.report.n_stable == report.n_stable
    assert pickle.loads(pickle.dumps(DynamicDeterminacyError("bare"))).report is None


def _sparse_three_by_six_calibration():
    """A seeded 3x6 table whose input matrix has density about .16 (iterative backends)."""
    rng = np.random.default_rng(20260923)
    N, S = 3, 6
    M = N * S
    Z = np.diag(rng.uniform(1, 3, M))
    mask = rng.random((M, M)) < 0.12
    Z[mask] += rng.uniform(.2, 2, mask.sum())
    F = rng.uniform(1, 5, (M, N, 5))
    F[:, :, 2] *= .05
    F[:, :, 3] *= .05
    F[:, :, 4] *= .02
    output = Z.sum(1) + F.sum((1, 2))
    TLS = .02 * output
    VA = output - Z.sum(0) - TLS
    alpha = rng.uniform(.25, .45, M)
    accounts = DynamicAccounts.from_arrays(
        country_codes=("A", "B", "C"), sector_codes=tuple(f"S{i}" for i in range(S)), Z=Z, F=F, VA=VA, TLS=TLS,
        output=output, TFD=.03 * F.sum(0), labor_compensation=(1 - alpha) * VA, operating_surplus=alpha * VA)
    return calibrate_dynamic(accounts)


def test_iterative_price_and_goods_backends_match_sparse_lu():
    cal = _sparse_three_by_six_calibration()
    assert cal.A.nnz / cal.n_cells ** 2 < .25
    results = {}
    for threshold in (600, 5):
        economy = DynamicEconomy(cal, direct_threshold=threshold)
        policy = DynamicTariff.uniform(cal, "A", .01)
        steady = solve_dynamic_steady_state(economy, policy, tol=1e-10, method="full")
        run = solve_dynamic_transition(economy, tariff_path(economy.zero_policy(), policy, horizon=12, announcement=1),
                                       terminal=steady, tol=1e-9, require_determinacy=False)
        assert max(run.certificate.values()) < 1e-8
        results[threshold] = (economy.goods_backend, economy._network(policy).backend, steady.z, run.path)
    assert results[600][:2] == ("sparse_lu_batch", "sparse_lu")
    assert results[5][:2] == ("sparse_iteration_batch", "iterative")
    np.testing.assert_allclose(results[5][2], results[600][2], rtol=0, atol=1e-10)
    np.testing.assert_allclose(results[5][3], results[600][3], rtol=0, atol=1e-10)
    # Numerical failures of the iterative solves are structured, not bare RuntimeErrors.
    with pytest.raises(DynamicSolveError, match="goods response") as goods:
        DynamicEconomy(cal, direct_threshold=5, network_tolerance=1e-30)
    assert goods.value.location["block"] == "goods_response"
    economy = DynamicEconomy(cal, direct_threshold=5)
    economy.network_tolerance = 1e-30
    economy.clear_caches()
    with pytest.raises(DynamicSolveError, match="Price-network solve failed") as prices:
        economy.state(np.zeros(economy.n_vars), np.zeros(economy.n_vars), DynamicTariff.uniform(cal, "B", .02))
    assert prices.value.location["block"] == "price_network" and prices.value.location["backend"] == "iterative"


# ---------------------------------------------------------------------------
# Parity with the IO engine and with puremacro's stacked Newton
# ---------------------------------------------------------------------------
def test_parity_with_io_engine_on_the_fixture_and_a_24_date_transition(native_fixture):
    ne = _io_module("", "dynamic_model.native_economy")
    ns = _io_module("", "dynamic_model.native_solver")
    nst = _io_module("", "dynamic_model.native_stationary")
    raw, d, model = native_fixture
    io_cal = ne.calibrate_native(raw)  # DynamicAccounts is duck-compatible with the IO container
    for name in ("y0", "VA0", "b", "tax", "alpha", "omegaC", "omegaCtax", "omegaI", "tC", "tI", "C0", "I0",
                 "K0", "R0", "L0sector", "L0", "PC0", "PI0", "Y0", "XN0"):
        np.testing.assert_array_equal(np.asarray(getattr(d, name)), np.asarray(getattr(io_cal, name)), err_msg=name)
    assert set(d.report) == set(io_cal.report)
    io_model = ne.NativeEconomy(io_cal, adjustment_cost=2, risk_aversion=2)
    rates = asymmetric_policy(d).rates
    policy = asymmetric_policy(d)
    io_policy = ne.NativePolicy(rates, "analytic tariff", consumption_rates=rates * .7, investment_rates=rates * 1.3)
    assert policy.fingerprint == io_policy.fingerprint
    z = np.zeros(model.n_vars)
    np.testing.assert_array_equal(model.equations(z, z, z, policy, policy), io_model.equations(z, z, z, io_policy, io_policy))
    steady = solve_dynamic_steady_state(model, policy, tol=2e-11)
    io_steady = ns.solve_steady_state(io_model, io_policy, tol=2e-11)
    assert np.max(np.abs(steady.z - io_steady.z)) <= 1e-13
    assert steady.certificate.keys() == io_steady.certificate.keys()
    condensed = solve_condensed_steady(model, policy, tol=1e-11)
    io_condensed = nst.solve_condensed_steady(io_model, io_policy, tol=1e-11)
    assert np.max(np.abs(condensed.z - io_condensed.z)) <= 1e-13 and condensed.iterations == io_condensed.iterations
    zero = model.zero_policy()
    run = solve_dynamic_transition(model, tariff_path(zero, policy, horizon=24, announcement=2), terminal=steady, tol=1e-10)
    io_run = ns.solve_transition(io_model, [io_model.zero_policy()] * 2 + [io_policy] * 22, terminal=io_steady.z, tol=1e-10)
    assert np.max(np.abs(run.path - io_run.z)) <= 1e-12
    assert run.iterations == io_run.iterations
    for key, value in io_run.certificate.items():
        assert run.certificate[key] == pytest.approx(value, abs=1e-15)
    io_states = io_run.states
    for state, io_state in zip(run.states, io_states):
        for name in ("C", "w", "K", "p", "TR", "budget"):
            np.testing.assert_allclose(state[name], io_state[name], rtol=0, atol=1e-12)


def test_parity_with_io_engine_on_a_preconditioned_sixty_date_transition(native_fixture):
    """480 unknowns: no MINPACK, lgmres with the capital preconditioner (the path large tables use)."""
    ne = _io_module("", "dynamic_model.native_economy")
    ns = _io_module("", "dynamic_model.native_solver")
    nn = _io_module("", "dynamic_model.native_numerics")
    raw, d, model = native_fixture
    io_model = ne.NativeEconomy(ne.calibrate_native(raw), adjustment_cost=2, risk_aversion=2)
    policy = asymmetric_policy(d)
    rates = np.array(policy.rates)
    io_policy = ne.NativePolicy(rates, "analytic tariff", consumption_rates=rates * .7, investment_rates=rates * 1.3)
    steady = solve_dynamic_steady_state(model, policy, tol=1e-11)
    io_steady = ns.solve_steady_state(io_model, io_policy, tol=1e-11)
    from puremacro.trade.dynamic.transition import _capital_preconditioner

    rng = np.random.default_rng(6060)
    x = rng.normal(size=7 * model.n_vars)
    np.testing.assert_array_equal(_capital_preconditioner(model).operator(7).matvec(x),
                                  nn.CapitalPreconditioner(io_model).operator(7).matvec(x))
    run = solve_dynamic_transition(model, tariff_path(model.zero_policy(), policy, horizon=60, announcement=3),
                                   terminal=steady, tol=1e-10)
    io_run = ns.solve_transition(io_model, [io_model.zero_policy()] * 3 + [io_policy] * 57, terminal=io_steady.z, tol=1e-10)
    assert all(item.get("linear_method") == "lgmres" for item in run.history)
    assert np.max(np.abs(run.path - io_run.z)) <= 1e-12 and run.iterations == io_run.iterations
    for key, value in io_run.certificate.items():
        assert run.certificate[key] == pytest.approx(value, abs=1e-15)


def test_parity_with_solve_perfect_foresight_at_forty_dates(native_fixture):
    from puremacro.dsge.perfect_foresight import solve_perfect_foresight

    _, d, model = native_fixture
    zero, policy = model.zero_policy(), asymmetric_policy(d)
    steady = solve_dynamic_steady_state(model, policy, tol=1e-11)
    horizon = 40
    run = solve_dynamic_transition(model, tariff_path(zero, policy, horizon=horizon, announcement=2), terminal=steady, tol=1e-10)
    lookup = {0: zero, 1: policy}
    exogenous = np.array([[0 if t < 2 else 1, 0 if t + 1 < 2 else 1] for t in range(horizon)], dtype=float)

    def equations(yp, yc, yl, eps):
        return model.equations(np.asarray(yp, float), np.asarray(yc, float), np.asarray(yl, float),
                               lookup[int(eps[0])], lookup[int(eps[1])])
    reference = solve_perfect_foresight(equations_fn=equations, y_init=np.zeros(model.n_vars), y_ss=steady.z,
                                        exogenous_path=exogenous, n_periods=horizon, tol=1e-10, method="central")
    assert reference.converged
    assert np.max(np.abs(reference.path.to_numpy() - run.path)) <= 1e-10


# ---------------------------------------------------------------------------
# Bundled 77x11 table
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bundled_economy():
    from puremacro.trade import load_icio_data

    accounts = DynamicAccounts.from_icio(load_icio_data(source="legacy", return_structured=True), negative_investment="to_inventory")
    with pytest.raises(ValueError, match="ARG"):
        calibrate_dynamic(accounts)
    calibration = calibrate_dynamic(accounts, factor_policy="reclassify_losses", investment_policy="reallocate",
                                    accounting_policy="reconcile_rounding")
    return calibration, DynamicEconomy(calibration)


def test_bundled_77x11_calibration_ledger_steady_states_and_failure_contract(bundled_economy):
    cal, eco = bundled_economy
    assert cal.n_cells == 847 and cal.report["factor_adjusted_cells"] == 0
    assert len(cal.report["investment_basket_reallocated_countries"]) == 21
    assert cal.report["investment_reallocation_L1_over_observed_world_investment"] == pytest.approx(0.01002, abs=2e-4)
    assert np.allclose(cal.alpha, 1 / 3)
    assert cal.to_dataframe()["rebasketed"].sum() == 21 and len(cal.to_dataframe("investment")) == 21
    assert eco.n_vars == 1001 and eco.goods_backend == "dense_lu_batch"
    base = solve_dynamic_steady_state(eco)
    assert base.method == "condensed" and base.iterations == 0 and max(base.certificate.values()) < 1e-12
    ten = solve_dynamic_steady_state(eco, DynamicTariff.uniform(cal, "USA", .10, sectors="merchandise"))
    assert ten.converged and ten.iterations <= 6 and max(ten.certificate.values()) < 1e-8
    assert ten.to_dataframe().loc["USA", "consumption_change_pct"] != 0
    zero = eco.zero_policy()
    one = DynamicTariff.uniform(cal, "USA", .01, sectors="merchandise")
    steady_one = solve_dynamic_steady_state(eco, one)
    run = solve_dynamic_transition(eco, tariff_path(zero, one, horizon=40), terminal=steady_one, require_determinacy=False)
    assert run.converged and run.iterations <= 6 and max(run.certificate.values()) < 1e-8
    assert run.metadata["determinacy_checked"] is False and run.stability is None
    assert run.welfare.consumption_equivalent_pct[cal.countries.index("USA")] < 0
    assert "and 72 more" in run.summary() and len(run.summary()) < 500
    five = DynamicTariff.uniform(cal, "USA", .05, sectors="merchandise")
    steady_five = solve_dynamic_steady_state(eco, five)
    with pytest.raises(DynamicSolveError) as excinfo:
        solve_dynamic_transition(eco, tariff_path(zero, five, horizon=40), terminal=steady_five, require_determinacy=False)
    error = excinfo.value
    assert "IRL:GOV" in str(error) and "terminal boundary date 40" in str(error)
    assert error.location["cell"] == "IRL:GOV" and error.location["date"] == 39
    assert error.location["min_investment_rate"] < 1e-6 and error.location["min_investment_rate_date"] == 40
    assert error.location["full_step_min_investment_rate"] < 0 and error.location["rejected_trials_domain"] == 24
    assert error.iterate.shape == (40 * eco.n_vars,) and error.residual > 1e-3


@pytest.mark.slow
def test_bundled_77x11_is_locally_indeterminate_under_the_native_closure(bundled_economy):
    cal, eco = bundled_economy
    base = solve_dynamic_steady_state(eco)
    report = stability_report(eco, base)
    assert (report.n_stable, report.n_predetermined, report.n_unit) == (846, 847, 0) and not report.determinate
    assert report.smallest_unstable_root == pytest.approx(1.00185, abs=2e-4)
    shares = report.to_dataframe()
    assert shares.index[0] == "IRL" and shares.iloc[0, 0] > 0.85 and shares.index[1] == "LUX"
    assert report.top_loadings[0][0] == "K:IRL:GOV"
    one = DynamicTariff.uniform(cal, "USA", .01, sectors="merchandise")
    steady_one = solve_dynamic_steady_state(eco, one)
    with pytest.raises(DynamicDeterminacyError, match="846 stable roots for 847"):
        solve_dynamic_transition(eco, tariff_path(eco.zero_policy(), one, horizon=40), terminal=steady_one)
    assert one.fingerprint in eco._stability_cache


@pytest.mark.slow
def test_bundled_77x11_horizon_ladder_with_one_percent_tariff(bundled_economy):
    cal, eco = bundled_economy
    zero = eco.zero_policy()
    one = DynamicTariff.uniform(cal, "USA", .01, sectors="merchandise")
    ladder = run_horizon_ladder(eco, lambda h: tariff_path(zero, one, horizon=h), horizons=(160, 320, 640),
                                require_determinacy=False)
    assert ladder.status == "solved_horizon_not_validated" and len(ladder.comparisons) == 2
    last = ladder.comparisons[-1]
    assert last.max_log_difference <= 1e-12 and last.welfare_max_difference_pp <= 1e-5
    assert last.terminal_max_log_gap > 1e-2 and not last.terminal_passed
    assert last.max_discounted_wealth <= 1e-6
    weaker = compare_horizons(ladder.solutions[1], ladder.solutions[2], terminal_check="discounted_wealth")
    assert weaker.passed and weaker.status == "verified_window_and_welfare"
    welfare = ladder.solutions[-1].welfare
    usa, mex = cal.countries.index("USA"), cal.countries.index("MEX")
    # Pinned from this implementation (2026-09-22): the assessment prototype
    # reported -0.0243 / +0.0326 with its own policy construction; the
    # difference (about 1.5e-3 pp) is within the looser second tolerance.
    assert welfare.consumption_equivalent_pct[usa] == pytest.approx(-0.02558, abs=5e-4)
    assert welfare.consumption_equivalent_pct[mex] == pytest.approx(0.03456, abs=5e-4)
    assert welfare.consumption_equivalent_pct[usa] == pytest.approx(-0.0243, abs=3e-3)
    assert welfare.consumption_equivalent_pct[mex] == pytest.approx(0.0326, abs=3e-3)
    # Inexact-Newton counts depend on BLAS threading and scipy versions; pin a
    # bound (this machine: 4, 4, 5) rather than the exact sequence.
    assert all(1 <= s.iterations <= 6 for s in ladder.solutions)


def test_parity_with_io_engine_on_the_bundled_77x11_table(bundled_economy):
    """Permissive calibration (reclassify/reallocate/reconcile), dense-LU backend and a 1% 40-date path.

    About 3 s beyond the shared module fixture, so it runs by default when the
    research volume is mounted (and skips otherwise).
    """
    ne = _io_module("", "dynamic_model.native_economy")
    ns = _io_module("", "dynamic_model.native_solver")
    from puremacro.trade import load_icio_data

    cal, eco = bundled_economy
    accounts = DynamicAccounts.from_icio(load_icio_data(source="legacy", return_structured=True), negative_investment="to_inventory")
    io_cal = ne.calibrate_native(accounts, factor_policy="reclassify_losses", investment_policy="reallocate",
                                 accounting_policy="reconcile_rounding")
    for name in ("y0", "VA0", "b", "tax", "alpha", "omegaC", "omegaCtax", "omegaI", "tC", "tI", "C0", "I0",
                 "K0", "R0", "L0sector", "L0", "PC0", "PI0", "Y0", "XN0", "active_indices", "country"):
        np.testing.assert_array_equal(np.asarray(getattr(cal, name)), np.asarray(getattr(io_cal, name)), err_msg=name)
    assert abs(cal.A - io_cal.A).max() == 0 and abs(cal.Z - io_cal.Z).max() == 0
    assert set(cal.report) == set(io_cal.report)
    io_eco = ne.NativeEconomy(io_cal)
    one = DynamicTariff.uniform(cal, "USA", .01, sectors="merchandise")
    io_one = ne.NativePolicy(np.array(one.rates), one.label, consumption_rates=np.array(one.consumption_rates),
                             investment_rates=np.array(one.investment_rates))
    assert eco._network(one).backend == io_eco._network(io_one).backend == "dense_lu"
    rng = np.random.default_rng(7711)
    z0, z1, z2 = (rng.normal(0, 2e-3, eco.n_vars) for _ in range(3))
    np.testing.assert_array_equal(eco.equations(z2, z1, z0, one, eco.zero_policy()),
                                  io_eco.equations(z2, z1, z0, io_one, io_eco.zero_policy()))
    steady = solve_dynamic_steady_state(eco, one)
    io_steady = ns.solve_steady_state(io_eco, io_one)
    assert np.max(np.abs(steady.z - io_steady.z)) <= 1e-13 and steady.iterations == io_steady.iterations
    run = solve_dynamic_transition(eco, tariff_path(eco.zero_policy(), one, horizon=40), terminal=steady,
                                   require_determinacy=False)
    io_run = ns.solve_transition(io_eco, [io_one] * 40, terminal=io_steady.z)
    assert np.max(np.abs(run.path - io_run.z)) <= 1e-12 and run.iterations == io_run.iterations
