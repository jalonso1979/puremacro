"""Slice (b): BBD AIEU benchmark fetcher (mocked + one live smoke)."""
from __future__ import annotations

from unittest import mock

import pandas as pd
import pytest


# Daily fixture covering Jan 2023 + Feb 2023 — enough to verify
# daily → monthly resampling.
_CSV_FIXTURE = "\n".join([
    "day,month,year,daily_ai_index,daily_aie_index,daily_aieu_index",
    "1,1,2023,100.0,50.0,10.0",
    "15,1,2023,200.0,150.0,30.0",
    "31,1,2023,300.0,100.0,50.0",
    "1,2,2023,400.0,200.0,80.0",
    "15,2,2023,200.0,100.0,40.0",
]) + "\n"


def _fake_get(url, **kwargs):
    resp = mock.Mock()
    resp.status_code = 200
    resp.text = _CSV_FIXTURE
    resp.raise_for_status = lambda: None
    return resp


def test_fetch_bbd_ai_epu_monthly_returns_monthly_series(tmp_path):
    """Daily CSV is resampled to monthly mean per the AIEU column."""
    from puremacro.narrative.validation.external_benchmarks import (
        fetch_bbd_ai_epu_monthly,
    )
    with mock.patch("puremacro.narrative.validation.external_benchmarks.requests.get",
                    side_effect=_fake_get):
        ser = fetch_bbd_ai_epu_monthly(cache_dir=tmp_path)

    assert isinstance(ser, pd.Series)
    assert ser.name == "aieu"
    # Two months in fixture.
    assert list(ser.index) == [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-02-01")]
    # Jan: mean(10, 30, 50) = 30.0; Feb: mean(80, 40) = 60.0
    assert ser.iloc[0] == pytest.approx(30.0)
    assert ser.iloc[1] == pytest.approx(60.0)


def test_fetch_bbd_ai_epu_caches_on_disk(tmp_path):
    """Second call (same cache_dir) doesn't re-fetch; it reads parquet."""
    from puremacro.narrative.validation.external_benchmarks import (
        fetch_bbd_ai_epu_monthly,
    )
    with mock.patch("puremacro.narrative.validation.external_benchmarks.requests.get",
                    side_effect=_fake_get) as mocked:
        fetch_bbd_ai_epu_monthly(cache_dir=tmp_path)
        first_calls = mocked.call_count
        ser2 = fetch_bbd_ai_epu_monthly(cache_dir=tmp_path)
        second_calls = mocked.call_count

    assert second_calls == first_calls  # no extra GET
    assert ser2.iloc[0] == pytest.approx(30.0)


def test_fetch_bbd_ai_epu_column_override(tmp_path):
    """Switching to 'ai' selects the AI-only sub-index."""
    from puremacro.narrative.validation.external_benchmarks import (
        fetch_bbd_ai_epu_monthly,
    )
    with mock.patch("puremacro.narrative.validation.external_benchmarks.requests.get",
                    side_effect=_fake_get):
        ser = fetch_bbd_ai_epu_monthly(cache_dir=tmp_path, column="ai")
    # Jan mean(100, 200, 300) = 200
    assert ser.iloc[0] == pytest.approx(200.0)
    assert ser.name == "ai"


def test_fetch_bbd_ai_epu_rejects_unknown_column(tmp_path):
    from puremacro.narrative.validation.external_benchmarks import (
        fetch_bbd_ai_epu_monthly,
    )
    with pytest.raises(ValueError, match="column must be one of"):
        fetch_bbd_ai_epu_monthly(cache_dir=tmp_path, column="unknown")


@pytest.mark.network
def test_fetch_bbd_ai_epu_live_smoke(tmp_path):
    """Live smoke: skip if endpoint unreachable or empty."""
    from puremacro.narrative.validation.external_benchmarks import (
        fetch_bbd_ai_epu_monthly,
    )
    try:
        ser = fetch_bbd_ai_epu_monthly(cache_dir=tmp_path)
    except Exception as e:
        pytest.skip(f"AIEU endpoint unreachable: {e}")
    if ser.empty:
        pytest.skip("AIEU fetcher returned empty")
    assert ser.index.min() <= pd.Timestamp("2020-01-01")
    assert ser.index.max() >= pd.Timestamp("2024-01-01")
