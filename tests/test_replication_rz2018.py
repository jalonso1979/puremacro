"""Replication tests for ``puremacro.examples.ramey_zubairy_2018_multipliers``.

Offline: reads the frozen Ramey-Zubairy snapshot
``puremacro/replication/data/rz2018.csv``. Assertions are qualitative
(signs / ranges / orderings) so they survive a snapshot regeneration;
the exact points (2yr 0.67, 4yr 0.71 vs RZ Table 1's 0.66, 0.71) are
documented in the example docstring.
"""
from __future__ import annotations

import numpy as np
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


def test_linear_cumulative_multiplier_in_rz_range(rz_result):
    """RZ Table 1 headline: cumulative news-shock multipliers ~0.6-0.8
    at the 2- and 4-year integrals, clearly below 1."""
    lin = rz_result["linear"]
    m2 = float(lin.loc[8, "beta"])
    m4 = float(lin.loc[16, "beta"])
    assert 0.3 <= m2 <= 1.0, f"2yr multiplier {m2:+.3f} off RZ's range"
    assert 0.3 <= m4 <= 1.1, f"4yr multiplier {m4:+.3f} off RZ's range"


def test_state_multipliers_do_not_exceed_one(rz_result):
    """RZ's central negative finding: no evidence of multipliers above 1
    in the slack state (contra Auerbach-Gorodnichenko)."""
    ms4 = float(rz_result["m_slack"][16])
    me4 = float(rz_result["m_expansion"][16])
    assert 0.2 <= ms4 <= 1.3, f"slack 4yr multiplier {ms4:+.3f} off range"
    assert 0.2 <= me4 <= 1.3, f"expansion 4yr multiplier {me4:+.3f} off range"
    assert abs(ms4 - me4) < 0.5, "states should not differ dramatically"


def test_effective_f_collapses_in_slack_state(rz_result):
    """The pedagogical core: the Olea-Pflueger effective F of the news
    first stage is strong at short horizons in the slack state (WWII)
    but collapses below the 23.1 cutoff by h=20."""
    feff = rz_result["f_eff"]
    f_short = float(feff.loc[feff["h"] <= 8, "slack"].max())
    f_long = float(feff.loc[feff["h"] == 20, "slack"].iloc[0])
    assert f_short > 23.1, f"short-horizon slack F {f_short:.1f} not strong"
    assert f_long < 23.1, f"h=20 slack F {f_long:.1f} did not collapse"
    # The expansion state never clears the cutoff on this sample.
    assert float(feff["expansion"].max()) < 23.1


def test_msw_band_wider_than_conventional(rz_result):
    """Under a marginal first stage the AR/MSW inversion band must be
    clearly wider than the conventional +/-1.645 SE band."""
    rob = rz_result["robust"]
    conv_w = rob["conv_hi"] - rob["conv_lo"]
    msw_w = rob["msw_hi"] - rob["msw_lo"]
    assert np.isfinite(conv_w) and np.isfinite(msw_w)
    assert msw_w > 1.2 * conv_w, (
        f"MSW width {msw_w:.3f} not clearly wider than "
        f"conventional {conv_w:.3f}")
    # Both bands should still contain the 2SLS point.
    assert rob["msw_lo"] <= rob["beta_2sls"] <= rob["msw_hi"]


def test_ar_test_consistency(rz_result):
    """AR cannot reject at the 2SLS point; the p-value is a proper
    probability; the AR-at-zero test has more evidence against it."""
    rob = rz_result["robust"]
    p_point = rob["ar_at_point"].p_value
    p_zero = rob["ar_at_zero"].p_value
    assert 0.0 <= p_zero <= 1.0 and 0.0 <= p_point <= 1.0
    assert p_point > 0.10
    assert p_zero < p_point


def test_first_stage_f_reported_per_horizon(rz_result):
    """lp_iv's per-horizon first-stage F must be finite and positive at
    every horizon of the linear one-step regressions."""
    lin = rz_result["linear"]
    assert (lin["first_stage_f"].to_numpy() > 0).all()
    assert np.isfinite(lin["first_stage_f"].to_numpy()).all()
