"""Slice (d): STATE_AI_EXPOSURE_2019 sanity checks."""
from __future__ import annotations


def test_state_ai_exposure_2019_table_complete():
    from puremacro.fetch.state_industry_panel import STATE_AI_EXPOSURE_2019
    # 50 states + DC = 51
    assert len(STATE_AI_EXPOSURE_2019) == 51
    # No NaN; all values are plausible occupation shares (0 < x < 0.20)
    for state, share in STATE_AI_EXPOSURE_2019.items():
        assert 0.0 < share < 0.20, f"{state}: implausible share {share}"


def test_state_ai_exposure_rank_pattern_matches_bls():
    """Top tier should include DC/VA/MD/WA/CA/MA/CO; bottom MS/AR/WV/MT."""
    from puremacro.fetch.state_industry_panel import STATE_AI_EXPOSURE_2019
    ranked = sorted(STATE_AI_EXPOSURE_2019.items(), key=lambda kv: kv[1], reverse=True)
    top_10 = {s for s, _ in ranked[:10]}
    bottom_5 = {s for s, _ in ranked[-5:]}
    assert {"DC", "VA", "MD", "WA"} <= top_10
    assert {"MS", "WV"} <= bottom_5


def test_state_ai_exposure_in_public_api():
    from puremacro.fetch import state_industry_panel as mod
    assert "STATE_AI_EXPOSURE_2019" in mod.__all__
