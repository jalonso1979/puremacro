"""Tests for state-industry panel fetcher."""
from __future__ import annotations
import pandas as pd
import pytest


def test_iter_national_industry_emp_q_returns_supersectors(monkeypatch):
    """Mock fetch_fred; verify the fetcher emits records for each industry."""
    from puremacro.fetch import state_industry_panel as sip

    dates = pd.date_range("2024-01-01", periods=12, freq="MS")
    calls = []
    def fake_fetch_fred(sid, timeout=30.0):
        calls.append(sid)
        return pd.Series([1000.0 + i for i in range(12)], index=dates, name=sid)
    monkeypatch.setattr(sip, "fetch_fred", fake_fetch_fred)

    rows = list(sip.iter_national_industry_emp_q())
    # 10 supersectors × 4 quarters in 12 months = 40 records.
    industries = {r[0] for r in rows}
    assert len(industries) == 10
    assert "MANEMP" in industries
    assert "USTPU" in industries
    # Records are 5-tuples.
    assert all(len(r) == 5 for r in rows)


def test_iter_national_industry_emp_q_subset(monkeypatch):
    from puremacro.fetch import state_industry_panel as sip

    dates = pd.date_range("2024-01-01", periods=6, freq="MS")
    def fake_fetch_fred(sid, timeout=30.0):
        return pd.Series([100.0 + i for i in range(6)], index=dates, name=sid)
    monkeypatch.setattr(sip, "fetch_fred", fake_fetch_fred)

    rows = list(sip.iter_national_industry_emp_q(supersectors=["MANEMP"]))
    assert {r[0] for r in rows} == {"MANEMP"}


def test_state_industry_shares_2005_is_complete():
    """The hardcoded baseline shares table covers all 51 states × 10 supersectors,
    and each state's shares sum to ~1.0 (rounding within ±0.05)."""
    from puremacro.fetch.state_industry_panel import (
        STATE_INDUSTRY_SHARES_2005,
        SUPERSECTORS,
        _FIPS,
    )
    states = set(_FIPS)
    assert states == set(STATE_INDUSTRY_SHARES_2005)
    for st, shares in STATE_INDUSTRY_SHARES_2005.items():
        assert set(shares) == set(SUPERSECTORS), (
            f"{st}: industries {set(shares)} != {set(SUPERSECTORS)}"
        )
        total = sum(shares.values())
        assert 0.95 < total < 1.05, f"{st}: shares sum {total:.3f} (need ~1.0)"


def test_iter_national_industry_emp_q_skips_on_error(monkeypatch):
    """If fetch_fred raises, fetcher skips silently per project rule."""
    from puremacro.fetch import state_industry_panel as sip
    def fake_fetch_fred(sid, timeout=30.0):
        raise RuntimeError("network down")
    monkeypatch.setattr(sip, "fetch_fred", fake_fetch_fred)
    rows = list(sip.iter_national_industry_emp_q(supersectors=["MANEMP"]))
    assert rows == []


@pytest.mark.network
def test_iter_national_industry_emp_q_live_smoke():
    """One live FRED call to confirm MANEMP parses."""
    from puremacro.fetch.state_industry_panel import iter_national_industry_emp_q
    rows = list(iter_national_industry_emp_q(supersectors=["MANEMP"]))
    if not rows:
        pytest.skip("FRED returned empty live response")
    assert all(r[2] > 0 for r in rows[-3:])  # log_emp positive


def test_top_counties_by_state_is_complete():
    """Every state has at least 1 county in the TOP_COUNTIES_BY_STATE list."""
    from puremacro.fetch.state_industry_panel import TOP_COUNTIES_BY_STATE, _FIPS
    assert set(TOP_COUNTIES_BY_STATE) == set(_FIPS)
    for st, counties in TOP_COUNTIES_BY_STATE.items():
        assert len(counties) >= 1
        fips_prefix = _FIPS[st]
        for fips in counties:
            assert fips.startswith(fips_prefix), f"{st}: {fips} doesn't start with {fips_prefix}"
            assert len(fips) == 5, f"FIPS code {fips} not 5 chars"


def test_iter_county_urate_q_offline(monkeypatch):
    """Mock fetch_fred; verify urate derivation = U/(U+E)*100."""
    import pandas as pd
    from puremacro.fetch import state_industry_panel as sip
    dates = pd.date_range("2024-01-01", periods=6, freq="MS")
    def fake_fetch_fred(sid, timeout=30.0):
        if sid.endswith("04"):
            return pd.Series([100.0]*6, index=dates, name=sid)  # unemployment
        if sid.endswith("05"):
            return pd.Series([1900.0]*6, index=dates, name=sid)  # employment
        raise ValueError(sid)
    monkeypatch.setattr(sip, "fetch_fred", fake_fetch_fred)
    rows = list(sip.iter_county_urate_q(states=["CA"]))
    assert len(rows) >= 1
    fips, st, qdate, urate, src, meta = rows[0]
    assert st == "CA"
    assert abs(urate - 100/2000*100) < 0.01  # = 5.0


def test_state_demographics_2005_complete():
    """All 51 states have demographic baseline values with plausible magnitudes."""
    from puremacro.fetch.state_industry_panel import (
        STATE_DEMOGRAPHICS_2005, _FIPS,
    )
    assert set(STATE_DEMOGRAPHICS_2005) == set(_FIPS)
    for st, dem in STATE_DEMOGRAPHICS_2005.items():
        assert set(dem) == {"ba_share", "prime_age_share", "foreign_born_share"}
        assert 0.10 < dem["ba_share"] < 0.55, f"{st} ba_share {dem['ba_share']}"
        assert 0.35 < dem["prime_age_share"] < 0.50, f"{st} prime_age_share"
        assert 0.0 < dem["foreign_born_share"] < 0.35, f"{st} foreign_born_share"
