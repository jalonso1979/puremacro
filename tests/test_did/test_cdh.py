"""Tests for de Chaisemartin-D'Haultfoeuille (2020) DID_M / DID_M^l.

Covers:
  - Result-object contract (frozen dataclass).
  - Recovery of a known homogeneous treatment effect.
  - Distribution of the switchers placebo p-value under a no-effect DGP.
  - Switcher counting on a known panel.
  - summary() smoke test.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from puremacro.did import cdh_did, CdHResult


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------


def _switching_panel(
    rng,
    *,
    n_units: int = 30,
    T: int = 12,
    switch_time: int = 6,
    treat_share: float = 0.5,
    true_att: float = 1.0,
    sigma: float = 0.4,
    allow_switch_back: bool = False,
):
    """Two-way FE panel where a fraction of units switch from 0->1 at
    ``switch_time``. Returns y, treatment, panel_id, time_id arrays.
    """
    units = np.arange(n_units)
    treated = rng.random(n_units) < treat_share
    alpha = rng.standard_normal(n_units)
    lam = rng.standard_normal(T) * 0.3

    y_list, d_list, p_list, t_list = [], [], [], []
    for i in units:
        for t in range(T):
            if treated[i]:
                d = 1 if t >= switch_time else 0
                if allow_switch_back and t >= switch_time + 3:
                    d = 0
            else:
                d = 0
            y = (alpha[i] + lam[t] + true_att * d
                 + sigma * rng.standard_normal())
            y_list.append(y)
            d_list.append(d)
            p_list.append(int(i))
            t_list.append(int(t))
    return (np.asarray(y_list, dtype=float),
            np.asarray(d_list, dtype=int),
            np.asarray(p_list, dtype=int),
            np.asarray(t_list, dtype=int))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_cdh_returns_CdHResult(rng):
    y, d, p, t = _switching_panel(rng, n_units=20, T=10, switch_time=5)
    res = cdh_did(y, d, p, t, placebo=False, n_boot=20, seed=1)
    assert isinstance(res, CdHResult)
    assert dataclasses.is_dataclass(res)
    # Frozen dataclass: assignment must fail.
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        res.att_M = 999.0


def test_cdh_recovers_homogeneous_effect(rng):
    y, d, p, t = _switching_panel(rng, n_units=80, T=12, switch_time=6,
                                    treat_share=0.5, true_att=2.0, sigma=0.3)
    res = cdh_did(y, d, p, t, placebo=False, n_boot=50, seed=0)
    # DID_M is the instantaneous switching effect; should sit near 2.0.
    assert abs(res.att_M - 2.0) < 0.5, f"att_M={res.att_M}"


def test_cdh_n_switchers_correct(rng):
    """Construct a panel with a known number of switchers."""
    n_units = 20
    n_treated = 8                                  # exactly 8 units switch
    T = 8
    switch_time = 4

    units = np.arange(n_units)
    treated = np.zeros(n_units, dtype=bool)
    treated[:n_treated] = True
    rng.shuffle(treated)
    alpha = rng.standard_normal(n_units)

    y_list, d_list, p_list, t_list = [], [], [], []
    for i in units:
        for tt in range(T):
            d = 1 if (treated[i] and tt >= switch_time) else 0
            y = alpha[i] + 0.3 * rng.standard_normal() + d
            y_list.append(y)
            d_list.append(d)
            p_list.append(int(i))
            t_list.append(int(tt))
    y = np.asarray(y_list); d = np.asarray(d_list)
    p = np.asarray(p_list); t = np.asarray(t_list)
    res = cdh_did(y, d, p, t, placebo=False, n_boot=10, seed=0)
    # Number of switching events at t=switch_time equals n_treated.
    assert res.n_switchers == n_treated, (
        f"expected {n_treated} switchers, got {res.n_switchers}"
    )


def test_cdh_placebo_p_uniform_under_no_effect():
    """Under a null DGP (no real treatment effect), the switchers
    placebo p-value should be roughly uniform on [0, 1]."""
    base_seed = 12345
    n_sim = 100
    pvals = np.empty(n_sim)
    for s in range(n_sim):
        rng = np.random.default_rng(base_seed + s)
        y, d, p, t = _switching_panel(
            rng, n_units=25, T=10, switch_time=5,
            treat_share=0.5, true_att=0.0, sigma=0.5,
        )
        res = cdh_did(y, d, p, t, placebo=True, n_boot=80,
                       seed=int(base_seed + s))
        pvals[s] = res.placebo_p
    # KS test against uniform: loose threshold (finite-sample non-uniformity
    # is real for bootstrap p-values).
    ks_p = stats.kstest(pvals, "uniform").pvalue
    assert ks_p > 0.01, (
        f"placebo p-values not uniform under the null: KS p = {ks_p:.4f}, "
        f"mean(p)={pvals.mean():.3f}"
    )


def test_cdh_summary(rng):
    y, d, p, t = _switching_panel(rng, n_units=15, T=8, switch_time=4)
    res = cdh_did(y, d, p, t, placebo=False, n_boot=10, seed=0)
    s = res.summary()
    assert isinstance(s, str) and len(s) > 0
    assert "Chaisemartin" in s or "DID_M" in s or "CdH" in s
