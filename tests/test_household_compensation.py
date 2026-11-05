"""Fiscal conservation and independently calculated household welfare."""
import numpy as np
import pandas as pd
import pytest

from puremacro.trade.compensation import balanced_compensation
from puremacro.trade.distributional import prepare_household_groups, compute_distributional_welfare


def households():
    return prepare_household_groups(
        pd.DataFrame([[60., 40.], [40., 160.]], index=["poor", "rich"], columns=["food", "other"]),
        pd.Series({"poor": 3., "rich": 1.}), monetary_unit="pesos", period="quarter",
        provenance={"source": "analytical fixture", "is_synthetic": True})


def test_balanced_plan_and_independent_cobb_douglas_welfare():
    g = households()
    plan = balanced_compensation(g, 30., benefit_scores=pd.Series({"rich": 0., "poor": 1.}),
                                 tax_base=g.baseline_budget)
    np.testing.assert_allclose(plan.households.benefit, [10, 0])
    np.testing.assert_allclose(plan.households.tax, [6, 12])
    np.testing.assert_allclose(plan.net_transfers, [4, -12])
    assert plan.ledger.net_transfers == pytest.approx(0)
    p0 = pd.Series({"food": 1., "other": 1.})
    p1 = pd.Series({"food": 1.2, "other": 1.})
    r = compute_distributional_welfare(g, p0, p1, household_transfers=plan.net_transfers,
                                      rule="cobb_douglas")
    expected = np.array([104 / 1.2**.6 - 100, 188 / 1.2**.2 - 200])
    np.testing.assert_allclose(r.groups.ev, expected, atol=1e-12)
    assert r.aggregate.total_ev == pytest.approx(np.dot([3, 1], expected))
    assert r.metadata["transfer_mode"] == "supplied_signed_per_unit"


def test_no_price_change_conserves_total_money_metric_welfare():
    g = households()
    plan = balanced_compensation(g, 13., benefit_scores=pd.Series(1., index=g.expenditure.index),
                                 tax_base=g.baseline_budget)
    p = pd.Series(1., index=g.expenditure.columns)
    r = compute_distributional_welfare(g, p, p, household_transfers=plan.net_transfers)
    assert r.aggregate.total_ev == pytest.approx(0, abs=1e-11)
    assert r.groups.loc["poor", "ev"] > 0
    assert r.groups.loc["rich", "ev"] < 0


def test_fiscal_scaling_and_label_alignment():
    g = households()
    a = balanced_compensation(g, 19., benefit_scores=pd.Series({"rich": 4., "poor": 1.}),
                             tax_base=g.baseline_budget.iloc[::-1])
    b = balanced_compensation(g, 190., benefit_scores=pd.Series({"poor": 10., "rich": 40.}),
                             tax_base=g.baseline_budget * 1e6)
    np.testing.assert_allclose(b.net_transfers, 10 * a.net_transfers)


@pytest.mark.parametrize("bad", [-1, np.nan, np.inf, [1]])
def test_invalid_gross_budget(bad):
    g = households()
    with pytest.raises(ValueError):
        balanced_compensation(g, bad, benefit_scores=pd.Series(1., index=g.expenditure.index),
                              tax_base=g.baseline_budget)


def test_missing_tax_base_and_unbalanced_signed_transfers_rejected():
    g = households()
    with pytest.raises(ValueError, match="positive"):
        balanced_compensation(g, 1., benefit_scores=g.baseline_budget,
                              tax_base=g.baseline_budget * 0)
    p = pd.Series(1., index=g.expenditure.columns)
    with pytest.raises(ValueError, match="conserve"):
        compute_distributional_welfare(g, p, p, household_transfers=pd.Series({"poor": 1., "rich": -1.}))
    with pytest.raises(ValueError, match="mutually exclusive"):
        compute_distributional_welfare(g, p, p, household_transfers=pd.Series({"poor": 1., "rich": -3.}),
                                      transfer_weights=g.baseline_budget)


@pytest.mark.pyodide_smoke
def test_portable_compensation():
    test_balanced_plan_and_independent_cobb_douglas_welfare()
