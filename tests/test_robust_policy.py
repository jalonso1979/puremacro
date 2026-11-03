"""Hand-calculated minimax regret, dominance and missing-evidence guards."""
import numpy as np
import pandas as pd
import pytest

from puremacro.study.robust import compare_policy_payoffs

CONTRACT = dict(population="same households", horizon="one quarter", numeraire="base prices",
                welfare_definition="population mean equivalent variation")


def compare(frame, **kw):
    return compare_policy_payoffs(frame, objective="mean EV", units="pesos", comparability=CONTRACT, **kw)


def table():
    return pd.DataFrame([[10., 0.], [6., 6.], [5., 5.]], index=["risky", "balanced", "dominated"],
                        columns=["good", "bad"])


def test_independent_regret_and_pareto_calculations():
    r = compare(table())
    np.testing.assert_array_equal(r.regrets, [[0, 6], [4, 0], [5, 1]])
    assert r.summary.minimax_regret.to_dict() == {"risky": False, "balanced": True, "dominated": False}
    assert r.summary.maximin.to_dict() == {"risky": False, "balanced": True, "dominated": False}
    assert r.dominance.loc["balanced", "dominated"]
    assert not r.dominance.loc["risky", "balanced"]
    assert "expected_payoff" not in r.summary


def test_declared_probabilities_and_minimization():
    p = pd.Series({"bad": .1, "good": .9})
    r = compare(table(), probabilities=p)
    np.testing.assert_allclose(r.summary.expected_payoff, [9, 6, 5])
    n = compare(-table(), probabilities=p, maximize=False)
    pd.testing.assert_frame_equal(n.regrets, r.regrets)
    pd.testing.assert_frame_equal(n.dominance, r.dominance)
    np.testing.assert_array_equal(n.summary.minimax_regret, r.summary.minimax_regret)


def test_equal_policies_are_ties_and_state_duplication_does_not_imply_probability():
    f = table()
    f.loc["tie"] = f.loc["balanced"]
    f["duplicate_good"] = f["good"]
    r = compare(f)
    assert r.summary.loc[["balanced", "tie"], "minimax_regret"].all()
    assert not r.dominance.loc["balanced", "tie"]
    assert "expected_payoff" not in r.summary


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_failed_payoff_cannot_disappear(value):
    f = table()
    f.iloc[0, 0] = value
    with pytest.raises(ValueError, match="every policy-state"):
        compare(f)


def test_failed_status_blocks_even_finite_placeholder():
    f = table()
    status = pd.DataFrame("passed", index=f.index, columns=f.columns)
    status.iloc[0, 0] = "failed"
    with pytest.raises(ValueError, match="passed"):
        compare(f, statuses=status)


@pytest.mark.parametrize("p", [pd.Series({"good": .5}), pd.Series({"good": .7, "bad": .7}),
                              pd.Series({"good": 1.1, "bad": -.1})])
def test_implicit_or_invalid_probabilities_refused(p):
    with pytest.raises(ValueError, match="probabilities"):
        compare(table(), probabilities=p)


@pytest.mark.pyodide_smoke
def test_portable_policy_comparison():
    test_independent_regret_and_pareto_calculations()
