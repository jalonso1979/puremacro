"""Tests for the multi-cohort SDID aggregator."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from puremacro.did import (
    sdid_multi_cohort, synthetic_did,
    SDIDMultiResult, SyntheticDiDResult,
)


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def _staggered_arrays(
    rng,
    *,
    n_units: int = 30,
    T: int = 18,
    cohorts: tuple = (6, 11, None),
    true_att_by_cohort: dict | None = None,
    sigma: float = 0.3,
):
    """Multi-cohort panel as four arrays (y, treatment, panel_id, time_id)."""
    if true_att_by_cohort is None:
        true_att_by_cohort = {c: 1.5 for c in cohorts if c is not None}
    units = np.arange(n_units)
    cohort_assign = rng.choice(cohorts, size=n_units, replace=True)
    alpha = rng.standard_normal(n_units)
    lam = rng.standard_normal(T) * 0.3

    y_list, d_list, p_list, t_list = [], [], [], []
    for i in units:
        g = cohort_assign[i]
        for t in range(T):
            if g is None:
                d = 0
                tau = 0.0
            else:
                d = 1 if t >= g else 0
                tau = true_att_by_cohort[g]
            y = (alpha[i] + lam[t] + tau * d
                 + sigma * rng.standard_normal())
            y_list.append(y); d_list.append(d)
            p_list.append(int(i)); t_list.append(int(t))
    return (np.asarray(y_list, dtype=float),
            np.asarray(d_list, dtype=int),
            np.asarray(p_list, dtype=int),
            np.asarray(t_list, dtype=int))


def _single_cohort_arrays(rng, *, n_units=20, T=20, treat_time=10,
                            n_treated=5, true_att=1.5, sigma=0.4):
    units = np.arange(n_units)
    is_treated = np.zeros(n_units, dtype=bool)
    is_treated[:n_treated] = True
    alpha = rng.standard_normal(n_units)
    lam = rng.standard_normal(T) * 0.3
    y_list, d_list, p_list, t_list = [], [], [], []
    for i in units:
        for t in range(T):
            d = 1 if (is_treated[i] and t >= treat_time) else 0
            y = (alpha[i] + lam[t] + true_att * d
                 + sigma * rng.standard_normal())
            y_list.append(y); d_list.append(d)
            p_list.append(int(i)); t_list.append(int(t))
    return (np.asarray(y_list, dtype=float),
            np.asarray(d_list, dtype=int),
            np.asarray(p_list, dtype=int),
            np.asarray(t_list, dtype=int)), is_treated


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sdid_multi_returns_SDIDMultiResult(rng):
    y, d, p, t = _staggered_arrays(rng, n_units=20, T=14)
    res = sdid_multi_cohort(y, d, p, t, n_boot=10, seed=0)
    assert isinstance(res, SDIDMultiResult)
    assert dataclasses.is_dataclass(res)
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        res.att = 999.0


def test_sdid_multi_cohort_weights_sum_to_one(rng):
    y, d, p, t = _staggered_arrays(rng, n_units=30, T=15,
                                     cohorts=(5, 9, 13, None))
    res = sdid_multi_cohort(y, d, p, t, n_boot=10, seed=0)
    assert res.cohort_weights.ndim == 1
    assert res.cohort_weights.shape[0] == res.cohort_atts.shape[0]
    np.testing.assert_allclose(res.cohort_weights.sum(), 1.0, atol=1e-10)


def test_sdid_multi_agrees_with_single_cohort_when_only_one_cohort(rng):
    """Degenerate case: a single cohort. Multi-cohort result must match
    `synthetic_did` within float tolerance."""
    arrays, _ = _single_cohort_arrays(
        rng, n_units=18, T=18, treat_time=10, n_treated=4, true_att=1.5,
    )
    y, d, p, t = arrays

    multi = sdid_multi_cohort(y, d, p, t, n_boot=15, seed=7)
    # Build the equivalent DataFrame for synthetic_did.
    treat_time_per_unit = {}
    for unit_i in np.unique(p):
        mask = (p == unit_i) & (d == 1)
        if mask.any():
            treat_time_per_unit[int(unit_i)] = float(t[mask].min())
        else:
            treat_time_per_unit[int(unit_i)] = np.nan
    df = pd.DataFrame({
        "unit": p, "time": t, "y": y,
        "treat_time": [treat_time_per_unit[int(pi)] for pi in p],
    })
    single = synthetic_did(df, n_boot=15, seed=7)
    assert isinstance(single, SyntheticDiDResult)
    assert multi.cohort_weights.shape[0] == 1
    np.testing.assert_allclose(multi.cohort_weights.sum(), 1.0, atol=1e-10)
    np.testing.assert_allclose(multi.att, single.tau, atol=1e-8)
    np.testing.assert_allclose(multi.cohort_atts[0], single.tau, atol=1e-8)


def test_sdid_multi_recovers_known_effect(rng):
    y, d, p, t = _staggered_arrays(
        rng, n_units=60, T=18, cohorts=(6, 11, None),
        true_att_by_cohort={6: 2.0, 11: 2.0}, sigma=0.25,
    )
    res = sdid_multi_cohort(y, d, p, t, aggregation="att",
                             n_boot=20, seed=0)
    assert abs(res.att - 2.0) < 0.7, f"got att={res.att}"


def test_sdid_multi_att_g_t_shape(rng):
    y, d, p, t = _staggered_arrays(
        rng, n_units=30, T=14, cohorts=(5, 9, None),
    )
    res = sdid_multi_cohort(y, d, p, t, aggregation="att_g_t",
                             n_boot=10, seed=0)
    assert res.att_g_t is not None
    assert isinstance(res.att_g_t, pd.DataFrame)
    expected = {"cohort", "event_time", "att"}
    assert expected.issubset(set(res.att_g_t.columns))
    # Cohorts in att_g_t must equal the identified cohort_times.
    cohorts_in_grid = set(res.att_g_t["cohort"].unique().tolist())
    assert cohorts_in_grid.issubset(set(res.cohort_times.tolist()))


def test_sdid_multi_summary(rng):
    y, d, p, t = _staggered_arrays(rng, n_units=20, T=14)
    res = sdid_multi_cohort(y, d, p, t, n_boot=10, seed=0)
    s = res.summary()
    assert isinstance(s, str) and len(s) > 0
    assert "Synthetic" in s or "SDID" in s or "Multi" in s
