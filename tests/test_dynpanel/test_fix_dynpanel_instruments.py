"""Uncollapsed (Stata xtabond) instrument layout: dead columns are pruned.

With ``collapse=False`` the (lag, t) basis used to be laid out from every
usable difference time before the rows lost to ``lag_dep_var >= 2`` (or to
lagged exogenous regressors) were dropped, leaving an all-zero column that
made Z'HZ singular: ``ab_gmm``/``bb_gmm`` raised LinAlgError. Stata builds
the instrument matrix on the estimation sample and never counts such a
column ([XT] xtdpd, Methods and formulas, p.18; [XT] xtabond Example 1:
41 instruments on abdata).
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.dynpanel import ab_gmm, bb_gmm
from puremacro.dynpanel.instruments import _prune_zero_columns, build_instruments

from .conftest import simulate_dynamic_panel


def _n_uncollapsed(T: int, P: int) -> int:
    """HENR count: rows t = P+1..T-1 each use y_0..y_{t-2} (t-1 columns)."""
    return sum(t - 1 for t in range(P + 1, T))


def test_prune_zero_columns_unit():
    Z = np.array([[1.0, 0.0, 2.0], [3.0, 0.0, 0.0]])
    Zk, kept, dropped = _prune_zero_columns(Z, ["a", "b", "c"])
    np.testing.assert_array_equal(Zk, Z[:, [0, 2]])
    assert kept == ["a", "c"] and dropped == ["b"]
    Zk, kept, dropped = _prune_zero_columns(Z[:, [0, 2]], ["a", "c"])
    assert dropped == [] and kept == ["a", "c"]


@pytest.mark.parametrize("T", [5, 7, 9])
def test_lags2_uncollapsed_has_no_dead_column(T):
    sim = simulate_dynamic_panel(N=40, T=T, rho=0.5, seed=0)
    b = build_instruments(
        sim["y"], sim["panel_id"], sim["time_id"],
        lag_dep_var=2, collapse=False,
    )
    Z = b["Z"]
    assert np.all(np.any(Z != 0.0, axis=0))
    assert Z.shape[1] == _n_uncollapsed(T, 2)
    assert b["dropped_instr_labels"] == ["_yL0_L2@t2"]
    assert len(b["instr_labels"]) == Z.shape[1]


def test_lags1_uncollapsed_unchanged():
    sim = simulate_dynamic_panel(N=10, T=5, rho=0.5, seed=0)
    b = build_instruments(sim["y"], sim["panel_id"], sim["time_id"], collapse=False)
    assert b["Z"].shape[1] == 6 == _n_uncollapsed(5, 1)
    assert b["dropped_instr_labels"] == []


@pytest.mark.parametrize("two_step", [True, False])
def test_ab_gmm_lags2_uncollapsed_runs_and_reports_note(two_step):
    T = 7
    sim = simulate_dynamic_panel(N=200, T=T, rho=0.5, seed=1)
    res = ab_gmm(
        sim["y"], sim["panel_id"], sim["time_id"],
        lag_dep_var=2, collapse=False, two_step=two_step,
    )
    assert res.n_instruments == _n_uncollapsed(T, 2) == 14
    assert res.hansen_j_df == 14 - 2
    assert np.all(np.isfinite(res.se)) and res.converged
    assert len(res.notes) == 1 and "_yL0_L2@t2" in res.notes[0]
    assert "note:" in res.summary()


def test_ab_gmm_default_fit_has_no_notes():
    sim = simulate_dynamic_panel(N=80, T=6, rho=0.5, seed=0)
    res = ab_gmm(sim["y"], sim["panel_id"], sim["time_id"])
    assert res.notes == ()
    assert "note:" not in res.summary()


def test_ab_gmm_uncollapsed_with_lagged_exogenous_regressor():
    """A twice-lagged exogenous regressor removes the difference rows at
    t = 1, 2, killing the (lag 2, t = 2) column even with lag_dep_var=1."""
    N, T = 150, 7
    sim = simulate_dynamic_panel(N=N, T=T, rho=0.4, beta_exog=0.6, seed=3)
    x = sim["x"].reshape(N, T)
    x_lag2 = np.full_like(x, np.nan)
    x_lag2[:, 2:] = x[:, :-2]
    X = np.column_stack([x.ravel(), x_lag2.ravel()])
    res = ab_gmm(
        sim["y"], sim["panel_id"], sim["time_id"],
        X_exog=X, collapse=False,
    )
    assert res.converged
    assert res.notes and "_yL0_L2@t2" in res.notes[0]
    # rows t = 3..T-1; GMM columns for t: y_0..y_{t-2}; plus 2 exog columns
    assert res.n_instruments == sum(t - 1 for t in range(3, T)) + 2
    assert abs(res.coefs[0] - 0.4) < 0.15
    assert abs(res.coefs[1] - 0.6) < 0.15
    assert abs(res.coefs[2]) < 0.15


@pytest.mark.parametrize("two_step", [True, False])
def test_bb_gmm_lags2_uncollapsed_runs(two_step):
    sim = simulate_dynamic_panel(N=200, T=7, rho=0.5, seed=1)
    res = bb_gmm(
        sim["y"], sim["panel_id"], sim["time_id"],
        lag_dep_var=2, collapse=False, two_step=two_step,
    )
    assert res.converged and np.all(np.isfinite(res.se))
    assert res.n_instruments == 14 + 2  # difference block + 2 level moments
    assert res.notes and "_yL0_L2@t2" in res.notes[0]
