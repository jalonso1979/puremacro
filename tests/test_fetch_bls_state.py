"""Tests for state-panel fetchers (via FRED CSV)."""
from __future__ import annotations
import io
import pandas as pd
import pytest


def _fake_fred_series(start="2024-01-01", n=12, base=4.0):
    """Return a pd.Series with the FRED-style monthly index."""
    dates = pd.date_range(start, periods=n, freq="MS")
    vals = [base + 0.1 * i for i in range(n)]
    s = pd.Series(vals, index=dates, name="fake")
    return s


def test_iter_state_urate_q_offline(monkeypatch):
    """Fetcher delegates to fetch_fred and averages monthly→quarterly."""
    from puremacro.fetch import bls_state_panel as bsp
    calls = []
    def fake_fetch_fred(sid, timeout=30.0):
        calls.append(sid)
        return _fake_fred_series(n=6, base=4.0)  # 6 months → 2 quarters
    monkeypatch.setattr(bsp, "fetch_fred", fake_fetch_fred)

    rows = list(bsp.iter_state_urate_q(states=["AL"]))
    assert calls == ["ALUR"]
    assert len(rows) == 2  # 2 quarters
    st, qdate, urate, url, meta = rows[0]
    assert st == "AL"
    assert isinstance(qdate, pd.Timestamp)
    assert 0 < float(urate) < 30
    assert "fredgraph.csv" in url


def test_iter_state_employment_q_offline(monkeypatch):
    from puremacro.fetch import bls_state_panel as bsp
    def fake_fetch_fred(sid, timeout=30.0):
        return _fake_fred_series(n=6, base=2000.0)  # thousands
    monkeypatch.setattr(bsp, "fetch_fred", fake_fetch_fred)

    rows = list(bsp.iter_state_employment_q(states=["AL"]))
    assert len(rows) == 2
    st, qdate, log_emp, url, meta = rows[0]
    assert st == "AL"
    assert log_emp > 0  # log(thousands)
    assert meta["measure"].startswith("log_")


def test_iter_state_participation_q_offline(monkeypatch):
    from puremacro.fetch import bls_state_panel as bsp
    captured_sid = []
    def fake_fetch_fred(sid, timeout=30.0):
        captured_sid.append(sid)
        return _fake_fred_series(n=6, base=60.0)
    monkeypatch.setattr(bsp, "fetch_fred", fake_fetch_fred)

    rows = list(bsp.iter_state_participation_q(states=["CA"]))
    assert captured_sid == ["LBSSA06"]  # CA FIPS=06
    assert len(rows) == 2
    st, qdate, lfpr, url, meta = rows[0]
    assert 40 < float(lfpr) < 80


def test_iter_state_urate_q_skips_on_empty(monkeypatch):
    """If fetch_fred raises, fetcher continues to next state per the project's
    fail-loud-via-skip rule."""
    from puremacro.fetch import bls_state_panel as bsp
    def fake_fetch_fred(sid, timeout=30.0):
        raise RuntimeError("network down")
    monkeypatch.setattr(bsp, "fetch_fred", fake_fetch_fred)
    rows = list(bsp.iter_state_urate_q(states=["AL", "AK"]))
    assert rows == []


def test_participation_a_alias_points_to_q():
    """Backward-compat: iter_state_participation_a == iter_state_participation_q."""
    from puremacro.fetch import bls_state_panel as bsp
    assert bsp.iter_state_participation_a is bsp.iter_state_participation_q


@pytest.mark.network
def test_iter_state_urate_q_live_smoke():
    """One live FRED call to confirm series ID + parsing."""
    from puremacro.fetch.bls_state_panel import iter_state_urate_q
    rows = list(iter_state_urate_q(states=["CA"]))
    if not rows:
        pytest.skip("FRED returned empty live response")
    # Recent quarter, plausible urate
    assert all(0 < float(r[2]) < 30 for r in rows[-3:])
