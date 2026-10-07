"""Monte Carlo coverage of the Callaway-Sant'Anna and Sun-Abraham intervals.

Design: three cohorts of different sizes (drawn per unit with probabilities
0.10 / 0.35 / 0.20, never treated 0.35) with heterogeneous, dynamic effects,
and unit-specific linear trends as the error. The trends make every cohort's
2x2 estimate share the control units' trend, so the cohort estimates are
positively correlated: the design in which treating them as independent (the
Sun-Abraham aggregate se of puremacro <= 4.3.0) understates the standard error.

A 1000-replication run of the same design (n_boot = 200) gave: CS
theta^O_sel percentile coverage 0.886 at nominal 0.90 (MC s.e. 0.010), SA
theta^O_W 0.897, event-study se / MC sd 0.97-0.99 with coverage 0.88-0.90; the
old aggregate se was 0.74-0.85 of the MC sd, with coverage 0.77-0.84. This
test runs a cheaper, seeded version (deterministic, about 10 s).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from puremacro.did import callaway_santanna

T = 8
TIMES = np.arange(1, T + 1)
GROUPS = np.array([3.0, 5.0, 6.0])
PROBS = np.array([0.10, 0.35, 0.20])
N = 300
TAU = {3.0: (0.5, 0.10), 5.0: (2.0, 0.30), 6.0: (1.0, -0.20)}


def _att(g, t):
    lvl, slope = TAU[g]
    return lvl + slope * (t - g) if t >= g else 0.0


def _truth():
    p = PROBS / PROBS.sum()
    sel = [np.mean([_att(g, t) for t in TIMES if t >= g]) for g in GROUPS]
    group = float(p @ np.array(sel))
    cells = [(i, g, t) for i, g in enumerate(GROUPS) for t in TIMES if t >= g]
    simple = sum(p[i] * _att(g, t) for i, g, t in cells) / sum(p[i] for i, _, _ in cells)
    es = {}
    for e in range(3):
        w = p / p.sum()          # every cohort is observed at e = 0, 1, 2
        es[e] = float(sum(w[i] * _att(g, g + e) for i, g in enumerate(GROUPS)))
    return group, simple, es


def _panel(rng):
    u = rng.random(N)
    g = np.full(N, np.nan)
    cum = np.cumsum(PROBS)
    for k in range(len(GROUPS) - 1, -1, -1):
        g[u < cum[k]] = GROUPS[k]
    a = rng.normal(size=N)
    b = rng.normal(size=N)
    eff = np.array([[_att(gi, t) if np.isfinite(gi) else 0.0 for t in TIMES] for gi in g])
    y = (a[:, None] + 0.2 * TIMES[None, :] + eff + 0.5 * b[:, None] * TIMES[None, :]
         + 0.3 * rng.normal(size=(N, T)))
    return pd.DataFrame({"unit": np.repeat(np.arange(N), T), "time": np.tile(TIMES, N),
                         "treat_time": np.repeat(g, T), "y": y.ravel()}), g


def test_overall_and_event_study_intervals_cover_under_shared_controls():
    R, NB, alpha = 200, 100, 0.10
    z = norm.ppf(1 - alpha / 2)
    group_true, simple_true, es_true = _truth()
    rng = np.random.default_rng(20260930)
    rec = []
    for r in range(R):
        df, g = _panel(rng)
        cs = callaway_santanna(df, n_boot=NB, seed=r, alpha=alpha)
        es = cs.att_event_study.set_index("event_time")
        agg = cs.overall_aggregations.set_index("aggregation")
        # the Sun-Abraham aggregate se of puremacro <= 4.3.0: sqrt(sum_g w_g^2 se_g^2)
        n_g = np.array([np.sum(g == c) for c in GROUPS], dtype=float)
        w = n_g / n_g.sum()
        gt = cs.att_gt.set_index(["g", "event_time"])["se"]
        row = {"cs": cs.att_overall, "cs_lo": cs.att_overall_lo, "cs_hi": cs.att_overall_hi,
               "sa": agg.loc["simple", "att"], "sa_se": agg.loc["simple", "se"]}
        for e in es_true:
            row[f"es{e}"] = es.loc[e, "att"]
            row[f"se{e}"] = es.loc[e, "se"]
            row[f"old{e}"] = float(np.sqrt(np.sum(
                w ** 2 * np.array([gt.loc[(c, e)] for c in GROUPS]) ** 2)))
        rec.append(row)
    d = pd.DataFrame(rec)

    # CS overall (theta^O_sel, eq. 3.11): percentile band
    cover = np.mean((d.cs_lo <= group_true) & (group_true <= d.cs_hi))
    assert 0.84 <= cover <= 0.96, cover
    # SA overall (theta^O_W, eq. 3.10): normal band from the joint-bootstrap se
    cover = np.mean(np.abs(d.sa - simple_true) <= z * d.sa_se)
    assert 0.84 <= cover <= 0.96, cover
    assert 0.85 <= d.sa_se.mean() / d.sa.std() <= 1.15

    for e, truth in es_true.items():
        est, se, old = d[f"es{e}"], d[f"se{e}"], d[f"old{e}"]
        ratio = se.mean() / est.std()
        assert 0.85 <= ratio <= 1.15, (e, ratio)
        cover = np.mean(np.abs(est - truth) <= z * se)
        assert cover >= 0.84, (e, cover)
        # the design discriminates: the independent-cohort formula understates
        assert old.mean() / est.std() < 0.85, (e, old.mean() / est.std())
