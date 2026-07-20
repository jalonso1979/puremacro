"""Coverage tests for puremacro.lp.lp_did (DGJT 2023 LP-DiD).

Module contract
---------------
lp_did(panel, y, treat, unit, time, horizons, *, pre_window, clean_controls,
       weights, controls, cluster, alpha) -> LPDiDResult
    Per-horizon regressions of y_{i,t+h} - y_{i,t-1} on the treatment-switch
    indicator with period effects, restricted to newly-treated units and
    clean controls (not yet treated through t + max(h, 0)). Pre-trend leads
    h = -pre_window..-2 come from the same regression; h = -1 is a mechanical
    zero (base period). weights='equal' is the equally-weighted ATT (DGJT
    reweighting); weights='vw' is plain OLS (variance-weighted, convex
    period weights). Cluster-robust SEs by unit.

Tests cover:
  - planted-truth recovery: heterogeneous dynamic ATTs across cohorts are
    recovered per horizon while a naive static TWFE (inline, exact balanced
    two-way demeaning) is badly biased on the same panel;
  - clean-control discipline: later-treated units drop out of the control
    pool at exactly the documented horizons (hand-counted n_clean), and a
    horizon with no clean controls left returns NaN with zero counts;
  - pre-trends: ~0 under parallel trends, detectably nonzero (right sign
    and magnitude) under a planted differential-trend violation;
  - equal vs vw: differ under cohort heterogeneity (exact analytic values
    with sigma = 0), coincide exactly under homogeneity;
  - group_weights diagnostics: normalized period shares match the analytic
    n_tr*n_cl/(n_tr+n_cl) and n_tr formulas;
  - cross-check: LP-DiD event-study path close to callaway_santanna's on
    the same DGP (loose tolerance — different control pools/aggregation);
  - clean_controls=False reproduces contaminated (forbidden-comparison)
    estimates that visibly differ under dynamic effects;
  - input validation: missing columns, negative horizons, non-binary or
    NaN treatment, non-absorbing treatment, bad pre_window / weights /
    cluster / alpha, duplicate (unit, time) rows;
  - result object: frozen dataclass, h = -1 normalization row, post /
    pretrend slices, summary() content.

All DGPs are seeded; no bootstrap is involved (analytic cluster SEs), so
every assertion is deterministic.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from puremacro.lp import lp_did, LPDiDResult


# ---------------------------------------------------------------------------
# Shared DGP
# ---------------------------------------------------------------------------

def _adoption_panel(
    cohort_sizes: dict,
    T: int,
    effect,
    sigma: float,
    seed: int = 0,
    pre_slope: float = 0.0,
) -> pd.DataFrame:
    """Two-way FE staggered-adoption panel with known dynamic effects.

    Y_{it} = alpha_i + lambda_t + tau(g, t - g)*1{t >= g}
             + pre_slope*t*1{ever treated} + sigma*eps.

    cohort_sizes maps first-treatment period g (None = never treated) to
    the number of units in that cohort. `effect(g, e)` is the true dynamic
    ATT at event time e >= 0 for cohort g.
    """
    rng = np.random.default_rng(seed)
    lam = rng.standard_normal(T) * 0.3
    rows = []
    uid = 0
    for g, m in cohort_sizes.items():
        for _ in range(m):
            a = rng.standard_normal()
            for t in range(1, T + 1):
                treated = (g is not None) and (t >= g)
                tau = effect(g, t - g) if treated else 0.0
                slope = pre_slope * t if g is not None else 0.0
                rows.append({
                    "unit": uid, "time": t,
                    "y": a + lam[t - 1] + slope + tau
                         + sigma * rng.standard_normal(),
                    "D": 1.0 if treated else 0.0,
                    "treat_time": np.nan if g is None else float(g),
                })
            uid += 1
    return pd.DataFrame(rows)


def _twfe_static(panel: pd.DataFrame) -> float:
    """Naive static TWFE beta via exact two-way demeaning (balanced panel)."""
    Y = panel.pivot(index="unit", columns="time", values="y").to_numpy()
    D = panel.pivot(index="unit", columns="time", values="D").to_numpy()

    def dem(M):
        return M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean()

    yd, dd = dem(Y), dem(D)
    return float((dd * yd).sum() / (dd * dd).sum())


# Heterogeneous-dynamics DGP shared by the recovery and TWFE-bias tests:
# early cohort's effect grows with event time, late cohort's is flat.
_HET = {"cohort_sizes": {8: 30, 14: 30, None: 40}, "T": 24,
        "effect": lambda g, e: (1.0 + 0.3 * e) if g == 8 else 0.5,
        "sigma": 0.3, "seed": 3}


# ---------------------------------------------------------------------------
# Planted-truth recovery vs naive TWFE
# ---------------------------------------------------------------------------

def test_recovers_heterogeneous_dynamic_att():
    pan = _adoption_panel(**_HET)
    res = lp_did(pan, "y", "D", "unit", "time", range(0, 9), pre_window=4)
    assert isinstance(res, LPDiDResult)
    # Equally-weighted truth: cohorts have 30 events each at every h <= 8.
    truth = np.array([(30 * (1.0 + 0.3 * h) + 30 * 0.5) / 60 for h in range(9)])
    est = res.post.sort_values("h")["beta"].to_numpy()
    assert np.abs(est - truth).max() < 0.25, (est, truth)
    # CI covers the truth at most horizons (seeded: all here).
    post = res.post
    assert ((post["lo"] <= truth) & (truth <= post["hi"])).mean() > 0.7


def test_pretrends_near_zero_under_parallel_trends():
    pan = _adoption_panel(**_HET)
    res = lp_did(pan, "y", "D", "unit", "time", range(0, 5), pre_window=4)
    pre = res.pretrend
    assert len(pre) == 3 and set(pre["h"]) == {-4, -3, -2}
    assert pre["beta"].abs().max() < 0.20
    assert np.nanmax(np.abs(pre["t"])) < 2.5


def test_naive_twfe_is_biased_where_lp_did_is_not():
    # Homogeneous but strongly *dynamic* effects: the classic Goodman-Bacon
    # (2021) failure mode for static TWFE under staggered adoption.
    effect = lambda g, e: 0.4 * (e + 1)
    pan = _adoption_panel({6: 20, 10: 20, 14: 20, None: 15}, 20, effect,
                          sigma=0.2, seed=7)
    twfe = _twfe_static(pan)
    # True average effect over treated (i, t) cells.
    cell_effects = [effect(g, t - g)
                    for g, m in {6: 20, 10: 20, 14: 20}.items()
                    for t in range(g, 21) for _ in range(m)]
    cell_avg = float(np.mean(cell_effects))
    assert abs(twfe - cell_avg) > 0.8, (twfe, cell_avg)   # observed: ~1.46

    res = lp_did(pan, "y", "D", "unit", "time", range(0, 6), pre_window=0)
    truth = np.array([effect(None, h) for h in range(6)])
    err = np.abs(res.post.sort_values("h")["beta"].to_numpy() - truth)
    assert err.max() < 0.25, err


# ---------------------------------------------------------------------------
# Clean-control discipline
# ---------------------------------------------------------------------------

def test_clean_controls_drop_out_at_documented_horizons():
    # No never-treated units: cohorts 6 (4 units), 10 (5), 14 (6); T = 20.
    pan = _adoption_panel({6: 4, 10: 5, 14: 6}, 20, lambda g, e: 1.0,
                          sigma=0.2, seed=2)
    res = lp_did(pan, "y", "D", "unit", "time", range(0, 9), pre_window=0)
    est = res.estimates.set_index("h")

    # h = 0..3: event t=6 sees cohorts {10, 14} clean (11 units); event t=10
    # sees {14} (6); the t=14 event has no clean control -> dropped.
    for h in range(0, 4):
        assert est.loc[h, "n_treated"] == 9 and est.loc[h, "n_clean"] == 17, h
        assert est.loc[h, "n_periods"] == 2
    # h = 4..7: cohort 10 is treated by t=6+h -> only {14} remains for the
    # t=6 event (6 units); the t=10 event loses everyone.
    for h in range(4, 8):
        assert est.loc[h, "n_treated"] == 4 and est.loc[h, "n_clean"] == 6, h
    # h = 8: nobody is still untreated at t + 8 -> inestimable, NaN + zeros.
    assert np.isnan(est.loc[8, "beta"]) and est.loc[8, "n_clean"] == 0
    # Monotone discipline: the clean pool never grows with the horizon.
    n_clean = res.post.sort_values("h")["n_clean"].to_numpy()
    assert (np.diff(n_clean) <= 0).all()


def test_never_treated_units_restore_long_horizons():
    pan = _adoption_panel({6: 4, 10: 5, 14: 6, None: 3}, 20,
                          lambda g, e: 1.0, sigma=0.2, seed=2)
    res = lp_did(pan, "y", "D", "unit", "time", [8], pre_window=0)
    row = res.estimates.iloc[-1]
    # Events at t=6 and t=10 (t=14 needs t+8=22 > T): 3 never-treated each.
    assert np.isfinite(row["beta"])
    assert row["n_treated"] == 9 and row["n_clean"] == 6


# ---------------------------------------------------------------------------
# Pre-trend violation detection
# ---------------------------------------------------------------------------

def test_pretrends_flag_planted_violation():
    # Ever-treated units drift up by 0.15/period relative to never-treated
    # controls, so the lead coefficient at h is ~ 0.15*(h+1) < 0.
    pan = _adoption_panel({8: 30, None: 40}, 20, lambda g, e: 1.0,
                          sigma=0.3, seed=5, pre_slope=0.15)
    res = lp_did(pan, "y", "D", "unit", "time", range(0, 5), pre_window=4)
    pre = res.pretrend.set_index("h")
    assert pre.loc[-4, "beta"] < -0.25          # planted: -3*0.15 = -0.45
    assert abs(pre.loc[-4, "t"]) > 3.0
    # Violation grows with the lead length (more trend accumulates).
    assert pre.loc[-4, "beta"] < pre.loc[-2, "beta"] + 0.1


# ---------------------------------------------------------------------------
# Weighting schemes
# ---------------------------------------------------------------------------

def test_equal_vs_vw_differ_under_heterogeneity_exactly():
    # sigma = 0: every period-specific DiD is exact, so both weightings have
    # closed forms. Cohort 6 (30 units, tau=0.5) vs cohort 12 (5 units,
    # tau=2.0), 10 never-treated.
    pan = _adoption_panel({6: 30, 12: 5, None: 10}, 20,
                          lambda g, e: 0.5 if g == 6 else 2.0,
                          sigma=0.0, seed=1)
    eq = lp_did(pan, "y", "D", "unit", "time", range(0, 5), pre_window=0)
    vw = lp_did(pan, "y", "D", "unit", "time", range(0, 5), pre_window=0,
                weights="vw")
    # equal: (30*0.5 + 5*2)/35;  vw period weights 30*15/45 : 5*10/15.
    assert np.allclose(eq.post["beta"], (30 * 0.5 + 5 * 2.0) / 35, atol=1e-8)
    w1, w2 = 30 * 15 / 45, 5 * 10 / 15
    assert np.allclose(vw.post["beta"], (w1 * 0.5 + w2 * 2.0) / (w1 + w2),
                       atol=1e-8)
    assert (vw.post["beta"] - eq.post["beta"]).abs().min() > 0.1


def test_equal_and_vw_coincide_under_homogeneity():
    pan = _adoption_panel({6: 30, 12: 5, None: 10}, 20, lambda g, e: 1.0,
                          sigma=0.0, seed=1)
    eq = lp_did(pan, "y", "D", "unit", "time", range(0, 5), pre_window=0)
    vw = lp_did(pan, "y", "D", "unit", "time", range(0, 5), pre_window=0,
                weights="vw")
    assert np.allclose(eq.post["beta"], vw.post["beta"], atol=1e-10)
    assert np.allclose(eq.post["beta"], 1.0, atol=1e-10)


def test_group_weights_diagnostics_match_analytic_shares():
    pan = _adoption_panel({6: 30, 12: 5, None: 10}, 20,
                          lambda g, e: 0.5 if g == 6 else 2.0,
                          sigma=0.0, seed=1)
    res = lp_did(pan, "y", "D", "unit", "time", [0], pre_window=0)
    gw = res.group_weights[res.group_weights["h"] == 0].set_index("time")
    assert set(gw.index) == {6, 12}
    assert gw.loc[6, "n_treated"] == 30 and gw.loc[6, "n_clean"] == 15
    assert np.isclose(gw.loc[6, "w_vw"], 0.75)      # 10 / (10 + 10/3)
    assert np.isclose(gw.loc[6, "w_equal"], 30 / 35)
    assert np.isclose(gw["w_vw"].sum(), 1.0) and np.isclose(gw["w_equal"].sum(), 1.0)


# ---------------------------------------------------------------------------
# Cross-check against Callaway-Sant'Anna
# ---------------------------------------------------------------------------

def test_event_study_path_close_to_callaway_santanna():
    from puremacro.did import callaway_santanna

    pan = _adoption_panel({6: 25, 12: 25, None: 30}, 20, lambda g, e: 1.2,
                          sigma=0.3, seed=11)
    lp = lp_did(pan, "y", "D", "unit", "time", range(0, 6), pre_window=0)
    cs = callaway_santanna(pan, unit="unit", time="time", outcome="y",
                           treat_time="treat_time", n_boot=20, seed=0)
    es = cs.att_event_study.set_index("event_time")
    lp_path = lp.post.set_index("h")["beta"]
    for h in range(0, 6):
        # Loose tolerance: CS uses never-treated controls and an unweighted
        # cohort mean; LP-DiD pools not-yet + never-treated with event weights.
        assert abs(lp_path.loc[h] - es.loc[h, "att"]) < 0.25, h


# ---------------------------------------------------------------------------
# clean_controls=False reproduces the forbidden comparisons
# ---------------------------------------------------------------------------

def test_dirty_controls_contaminate_under_dynamic_effects():
    effect = lambda g, e: 0.4 * (e + 1)
    pan = _adoption_panel({6: 20, 10: 20, 14: 20, None: 15}, 20, effect,
                          sigma=0.2, seed=7)
    clean = lp_did(pan, "y", "D", "unit", "time", range(0, 4), pre_window=0)
    dirty = lp_did(pan, "y", "D", "unit", "time", range(0, 4), pre_window=0,
                   clean_controls=False)
    # Already-treated "controls" have rising outcomes -> downward bias.
    gap = (clean.post["beta"] - dirty.post["beta"]).to_numpy()
    assert gap.min() > 0.15, gap
    # And the dirty control pool is strictly larger.
    assert (dirty.post["n_clean"] > clean.post["n_clean"]).all()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _tiny_panel():
    return _adoption_panel({3: 3, None: 3}, 8, lambda g, e: 1.0,
                           sigma=0.1, seed=0)


def test_validation_errors():
    pan = _tiny_panel()
    with pytest.raises(ValueError, match="lp_did.*not in panel"):
        lp_did(pan, "nope", "D", "unit", "time", [0])
    with pytest.raises(ValueError, match="lp_did.*horizons.*>= 0"):
        lp_did(pan, "y", "D", "unit", "time", [-1, 0])
    with pytest.raises(ValueError, match="lp_did.*horizons is empty"):
        lp_did(pan, "y", "D", "unit", "time", [])
    with pytest.raises(ValueError, match="lp_did.*pre_window"):
        lp_did(pan, "y", "D", "unit", "time", [0], pre_window=1)
    with pytest.raises(ValueError, match="lp_did.*weights"):
        lp_did(pan, "y", "D", "unit", "time", [0], weights="banana")
    with pytest.raises(ValueError, match="lp_did.*cluster"):
        lp_did(pan, "y", "D", "unit", "time", [0], cluster="no_such_col")
    with pytest.raises(ValueError, match="lp_did.*alpha"):
        lp_did(pan, "y", "D", "unit", "time", [0], alpha=1.5)

    bad = pan.copy()
    bad.loc[0, "D"] = 0.5
    with pytest.raises(ValueError, match="lp_did.*binary"):
        lp_did(bad, "y", "D", "unit", "time", [0])
    bad = pan.copy()
    bad.loc[0, "D"] = np.nan
    with pytest.raises(ValueError, match="lp_did.*NaN"):
        lp_did(bad, "y", "D", "unit", "time", [0])
    dup = pd.concat([pan, pan.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="lp_did.*duplicate"):
        lp_did(dup, "y", "D", "unit", "time", [0])


def test_non_absorbing_treatment_raises_diagnostic():
    pan = _tiny_panel()
    # Force a 1 -> 0 reversal on the first treated unit.
    u0 = pan.loc[pan["D"] == 1.0, "unit"].iloc[0]
    idx = pan[(pan["unit"] == u0)].index[-1]
    pan.loc[idx, "D"] = 0.0
    with pytest.raises(ValueError, match="lp_did.*non-absorbing"):
        lp_did(pan, "y", "D", "unit", "time", [0])


# ---------------------------------------------------------------------------
# Result object contract
# ---------------------------------------------------------------------------

def test_result_object_contract():
    pan = _adoption_panel({5: 8, None: 8}, 14, lambda g, e: 1.0,
                          sigma=0.2, seed=4)
    res = lp_did(pan, "y", "D", "unit", "time", range(0, 4), pre_window=3)

    assert type(res).__dataclass_params__.frozen
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.weights = "vw"

    est = res.estimates
    assert list(est.columns) == ["h", "beta", "se", "t", "lo", "hi",
                                 "n_treated", "n_clean", "n_obs", "n_periods"]
    assert est["h"].tolist() == [-3, -2, -1, 0, 1, 2, 3]
    base = est[est["h"] == -1].iloc[0]
    assert base["beta"] == 0.0 and base["se"] == 0.0     # mechanical zero
    assert base["n_treated"] == 8                        # one cohort of 8
    assert set(res.pretrend["h"]) == {-3, -2}
    assert set(res.post["h"]) == {0, 1, 2, 3}
    assert res.n_units == 16 and res.n_events == 8
    assert res.weights == "equal" and res.clean_controls is True

    s = res.summary()
    assert "LP-DiD" in s and "equal" in s and "pre-trends" in s

    # cluster= an explicit column identical to unit reproduces cluster='unit'.
    pan2 = pan.assign(state=pan["unit"])
    res2 = lp_did(pan2, "y", "D", "unit", "time", range(0, 4), pre_window=3,
                  cluster="state")
    pd.testing.assert_frame_equal(res.estimates, res2.estimates)


def test_pre_window_zero_skips_pretrends():
    pan = _adoption_panel({5: 8, None: 8}, 14, lambda g, e: 1.0,
                          sigma=0.2, seed=4)
    res = lp_did(pan, "y", "D", "unit", "time", range(0, 3), pre_window=0)
    assert res.estimates["h"].min() == 0
    assert len(res.pretrend) == 0
