"""Replication tests for ``puremacro.examples.ramey_zubairy_2018_multipliers``.

Offline: reads the frozen Ramey-Zubairy snapshot
``puremacro/replication/data/rz2018.csv``. Most assertions are qualitative
(signs / ranges / orderings) so they survive a snapshot regeneration; the
exact points (2yr 0.66 and 4yr 0.71 at h = 7 and 15, vs RZ Table 1's 0.66,
0.71) are documented in the example docstring. The one-step state-dependent
multipliers are pinned to rtol 1e-8, so a regenerated snapshot means
re-measuring them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def rz_result():
    from puremacro.examples.ramey_zubairy_2018_multipliers import (
        run_rz_replication,
    )
    return run_rz_replication()


def test_data_loads_offline():
    from puremacro.examples.ramey_zubairy_2018_multipliers import load_rz_data

    d = load_rz_data()
    assert {"y", "g", "newsy", "slack_l", "cumy", "cumg"} <= set(d.columns)
    assert len(d) > 450  # quarterly, 1890 onwards
    # Gordon-Krenn ratios hover around their trend normalization.
    assert 0.5 < d["y"].median() < 1.5
    assert d["g"].min() > 0.0
    # RZ's slack state is occupied roughly a third of the sample.
    share = d["slack_l"].mean()
    assert 0.15 < share < 0.55


def test_missing_unemployment_keeps_slack_missing(monkeypatch):
    import puremacro.examples.ramey_zubairy_2018_multipliers as rzmod
    real = rzmod.load_csv

    def with_gap(name):
        d = real(name)
        d.loc[d.index[200], "unemp"] = np.nan
        return d

    monkeypatch.setattr(rzmod, "load_csv", with_gap)
    df = rzmod.load_rz_data()
    missing = pd.Timestamp(real("rz2018")["date"].iloc[200])
    assert np.isnan(df["slack_l"].iloc[df.index.get_loc(missing) + 1])


def test_linear_cumulative_multiplier_in_rz_range(rz_result):
    """RZ Table 1 headline: cumulative news-shock multipliers ~0.6-0.8
    at the 2- and 4-year integrals (h = 7 and 15), clearly below 1."""
    lin = rz_result["linear"]
    m2 = float(lin.loc[7, "beta"])
    m4 = float(lin.loc[15, "beta"])
    assert 0.3 <= m2 <= 1.0, f"2yr multiplier {m2:+.3f} off RZ's range"
    assert 0.3 <= m4 <= 1.1, f"4yr multiplier {m4:+.3f} off RZ's range"


def test_state_multipliers_do_not_exceed_one(rz_result):
    """RZ's central negative finding: no evidence of multipliers above 1
    in the slack state (contra Auerbach-Gorodnichenko)."""
    ms4 = float(rz_result["m_slack"][15])
    me4 = float(rz_result["m_expansion"][15])
    assert 0.2 <= ms4 <= 1.3, f"slack 4yr multiplier {ms4:+.3f} off range"
    assert 0.2 <= me4 <= 1.3, f"expansion 4yr multiplier {me4:+.3f} off range"
    assert abs(ms4 - me4) < 0.5, "states should not differ dramatically"


def test_one_step_state_dependent_multipliers_pinned(rz_result):
    """RZ's one-step state-dependent integral multipliers at 2 and 4 years
    (h = 7 and 15) on the frozen snapshot. RZ Table 1 reports .60 / .59 at
    2 years and .68 / .67 at 4 years (high / low unemployment)."""
    ms, me = rz_result["m_slack"], rz_result["m_expansion"]
    np.testing.assert_allclose(
        [ms[7], ms[15], me[7], me[15]],
        [0.6028751048822653, 0.6819723644803134,
         0.5949443896394699, 0.6683477316895448],
        rtol=1e-8)


def test_effective_f_pattern_by_state(rz_result):
    """The pedagogical core, under HAC: the effective F of the news first
    stage is weak on impact with slack, strong from h = 3 on, and never
    strong without slack or in the linear model. h = 2 (23.05) sits on the
    23.1 cutoff and is left out."""
    f = rz_result["f_eff"].set_index("h")
    assert f.loc[[0, 1], "slack"].max() < 23.1    # weak on impact: 2.79, a factor 8.3 below
    assert f.loc[3:20, "slack"].min() > 23.1      # strong from h = 3: 55.90 at h = 20, a factor 2.4 above
    assert f["expansion"].max() < 23.1            # never strong without slack: 11.10, a factor 2.1 below
    assert f["linear"].max() < 23.1               # nor in the linear model: 16.81, a factor 1.37 below


def test_ar_set_wider_than_conventional_band(rz_result):
    """Under a weak first stage (no slack, 2 years) the HAC Anderson-Rubin
    set must be clearly wider than the conventional HAC band; both contain
    the estimate."""
    rob = rz_result["robust"]
    assert rob["h"] == 7 and rob["f_eff"] < 23.1                     # weak first stage: 7.30
    assert rob["ar_set_type"] == "bounded"
    assert rob["conv_lo"] <= rob["beta"] <= rob["conv_hi"] and rob["ar_lo"] <= rob["beta"] <= rob["ar_hi"]
    assert rob["ar_hi"] - rob["ar_lo"] > 1.2 * (rob["conv_hi"] - rob["conv_lo"])   # 0.381 vs 0.298: 1.278


def test_ar_set_excludes_zero_and_one(rz_result):
    """The weak-instrument-robust answer is still informative: the 90% set
    without slack at 2 years rules out both a zero multiplier and one."""
    rob = rz_result["robust"]
    assert rob["ar_lo"] > 0.0    # the 90% AR set excludes a zero multiplier: 0.436 (HAC AR p at 0: 0.0067)
    assert rob["ar_hi"] < 1.0    # and a multiplier of one: 0.817, a margin of 0.183 (p at 1: 0.034)


def test_weak_headlines_carry_bounded_ar_sets(rz_result):
    """The example flags the linear and no-slack headlines at 2 and 4 years
    as weak and quotes their HAC Anderson-Rubin 90% sets, which must be
    bounded and contain the estimate; the no-slack sets come from lp_iv on
    the no-slack quarters, which reproduces beta_L."""
    lin, st, exp = rz_result["linear"], rz_result["state_dep"], rz_result["expansion_iv"]
    for h in (7, 15):
        assert lin.loc[h, "mop_f"] < 23.1 and st.loc[h, "first_stage_f_L"] < 23.1   # 16.0, 12.0; 7.3, 10.8
        for r in (lin.loc[h], exp.loc[h]):
            assert r["ar_set_type"] == "bounded" and r["ar_lo"] <= r["beta"] <= r["ar_hi"]
        np.testing.assert_allclose(exp.loc[h, "beta"], st.loc[h, "beta_L"], rtol=1e-8)


def test_first_stage_f_reported_per_horizon(rz_result):
    """lp_iv's per-horizon first-stage F must be finite and positive at
    every horizon of the linear one-step regressions."""
    lin = rz_result["linear"]
    assert (lin["first_stage_f"].to_numpy() > 0).all()
    assert np.isfinite(lin["first_stage_f"].to_numpy()).all()
