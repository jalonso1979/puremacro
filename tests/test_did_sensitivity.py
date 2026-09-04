"""Unit tests for Honest DiD sensitivity analysis (Rambachan & Roth 2023)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.did import (
    callaway_santanna,
    honest_did_sensitivity,
    HonestDiDResult,
)


def test_honest_did_basic_relative_magnitude():
    # Synthetic event-study series
    # pre-periods: -3, -2, -1 (base)
    # post-periods: 0, 1, 2
    event_time = [-3, -2, -1, 0, 1, 2]
    # pre-trends: max abs deviation = 0.2
    beta = [0.15, -0.20, 0.0, 1.50, 1.40, 1.30]
    se = [0.10, 0.10, 0.0, 0.15, 0.18, 0.20]

    res = honest_did_sensitivity(
        event_time=event_time,
        beta=beta,
        se=se,
        target_horizon=0,
        method="relative_magnitude",
        m_grid=[0.0, 0.5, 1.0, 2.0],
        ci=0.95,
    )

    assert isinstance(res, HonestDiDResult)
    assert res.pre_trend_max == pytest.approx(0.20)
    assert res.method == "relative_magnitude"
    assert res.ci == 0.95

    tbl = res.to_frame()
    assert len(tbl) == 4
    assert list(tbl["M"]) == [0.0, 0.5, 1.0, 2.0]

    # At M=0, identified set is point [1.50, 1.50]
    m0_row = tbl[tbl["M"] == 0.0].iloc[0]
    assert m0_row["id_lo"] == pytest.approx(1.50)
    assert m0_row["id_hi"] == pytest.approx(1.50)
    # 95% CI is 1.50 +/- 1.95996 * 0.15
    assert m0_row["ci_lo"] == pytest.approx(1.50 - 1.95996 * 0.15, rel=1e-3)
    assert m0_row["ci_hi"] == pytest.approx(1.50 + 1.95996 * 0.15, rel=1e-3)
    assert bool(m0_row["significant"]) is True

    # At M=1.0, identified set is [1.50 - 0.20, 1.50 + 0.20] = [1.30, 1.70]
    m1_row = tbl[tbl["M"] == 1.0].iloc[0]
    assert m1_row["id_lo"] == pytest.approx(1.30)
    assert m1_row["id_hi"] == pytest.approx(1.70)
    assert m1_row["ci_lo"] < 1.30
    assert m1_row["ci_hi"] > 1.70

    # Breakdown value M*
    # 1.50 - M* * 0.20 - c(M*) * 0.15 = 0
    # For large M*, c(M*) ~ 1.645
    # 1.50 - M* * 0.20 - 1.645 * 0.15 = 0 => M* ~ (1.50 - 0.24675) / 0.20 = 6.266
    assert isinstance(res.breakdown_value, float)
    assert 6.0 < res.breakdown_value < 6.5


def test_honest_did_with_callaway_santanna_result():
    # Build simple panel
    rng = np.random.default_rng(42)
    units = []
    for u in range(16):
        treat_yr = 2012 if u < 6 else (2014 if u < 12 else np.nan)
        for yr in range(2009, 2016):
            d = 1.0 if not np.isnan(treat_yr) and yr >= treat_yr else 0.0
            y = 3.0 * d + 0.2 * (yr - 2009) + rng.standard_normal() * 0.5
            units.append({"unit": f"U{u}", "year": yr, "treat_time": treat_yr, "outcome": y})

    df_panel = pd.DataFrame(units)
    cs_res = callaway_santanna(df_panel, unit="unit", time="year", outcome="outcome", treat_time="treat_time")

    sens = honest_did_sensitivity(cs_res, target_horizon=0, ci=0.90)
    assert isinstance(sens, HonestDiDResult)
    assert sens.target_horizons == [0]
    assert len(sens.table) > 0

    # Formatting checks
    summ = sens.summary()
    assert "Honest DiD Sensitivity Analysis" in summ
    assert "Breakdown Value" in summ

    md = sens.to_markdown()
    assert "| M |" in md or "| M" in md or "M" in md

    ltx = sens.to_latex()
    assert "\\begin{tabular}" in ltx

    typ = sens.to_typst()
    assert "#table" in typ

    ascii_art = sens.plot_ascii()
    assert "Honest DiD Confidence Intervals vs M" in ascii_art
    assert "|" in ascii_art


def test_honest_did_smoothness_method():
    event_time = [-3, -2, -1, 0, 1]
    beta = [0.1, -0.05, 0.0, 1.2, 1.5]
    se = [0.1, 0.1, 0.0, 0.2, 0.2]

    sens = honest_did_sensitivity(
        event_time=event_time,
        beta=beta,
        se=se,
        target_horizon=[0, 1],
        method="smoothness",
        m_grid=[0.0, 0.1, 0.2],
    )
    assert sens.method == "smoothness"
    assert sens.pre_trend_slope is not None
    assert isinstance(sens.breakdown_value, dict)
    assert 0 in sens.breakdown_value
    assert 1 in sens.breakdown_value


def test_honest_did_already_insignificant():
    # If baseline estimate is not significant, breakdown value M* = 0.0
    event_time = [-2, -1, 0]
    beta = [0.1, 0.0, 0.05]
    se = [0.1, 0.0, 0.5]  # large standard error

    sens = honest_did_sensitivity(
        event_time=event_time,
        beta=beta,
        se=se,
        target_horizon=0,
    )
    assert sens.breakdown_value == 0.0
    assert "already not statistically distinguishable" in sens.summary()


def test_honest_did_validation_errors():
    with pytest.raises(ValueError, match="method must be"):
        honest_did_sensitivity(event_time=[-1, 0], beta=[0, 1], se=[0, 0.1], method="unknown")

    with pytest.raises(ValueError, match="must provide either 'result'"):
        honest_did_sensitivity()

    with pytest.raises(ValueError, match="at least one pre-treatment period"):
        honest_did_sensitivity(event_time=[-1, 0, 1], beta=[0.0, 1.0, 1.2], se=[0.0, 0.1, 0.1], base_period=-1)

    with pytest.raises(ValueError, match="equal lengths"):
        honest_did_sensitivity(event_time=[-2, -1, 0], beta=[0.1, 0.0], se=[0.1, 0.1])
