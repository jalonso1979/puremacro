"""Tests for the X13-ARIMA seasonal-adjustment helper."""
from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest


def _has_x13_binary() -> bool:
    """X13 requires the x13as binary on PATH."""
    return shutil.which("x13as") is not None or shutil.which("x13as_ascii") is not None


def _make_seasonal_quarterly_series(
    n_periods: int = 40, *, start: str = "2010-01-01"
) -> pd.Series:
    """Synthetic quarterly series with a clear seasonal pattern.

    Trend: linear up. Seasonal: +2.0 in Q1, -2.0 in Q3. Mild noise.
    """
    rng = np.random.default_rng(42)
    idx = pd.date_range(start, periods=n_periods, freq="QS")
    trend = np.linspace(100, 120, n_periods)
    quarter = idx.quarter
    seasonal = np.where(quarter == 1, 2.0,
                np.where(quarter == 3, -2.0, 0.0))
    noise = rng.normal(0, 0.3, n_periods)
    return pd.Series(trend + seasonal + noise, index=idx, name="emp")


def test_x13_smoke_returns_same_shape_when_binary_available():
    """If x13as is installed, the wrapper returns an SA series with the same index."""
    if not _has_x13_binary():
        pytest.skip("x13as binary not installed; install via `brew install x13as`")
    from puremacro.fetch._seasonal import x13_seasonal_adjust
    raw = _make_seasonal_quarterly_series()
    sa = x13_seasonal_adjust(raw, freq="Q")
    assert isinstance(sa, pd.Series)
    assert sa.index.equals(raw.index)
    assert sa.name == raw.name
    # SA series should reduce the Q1-vs-Q3 swing meaningfully.
    raw_q1_q3 = raw[raw.index.quarter == 1].mean() - raw[raw.index.quarter == 3].mean()
    sa_q1_q3 = sa[sa.index.quarter == 1].mean() - sa[sa.index.quarter == 3].mean()
    assert abs(sa_q1_q3) < abs(raw_q1_q3) * 0.5


def test_x13_missing_binary_raises_with_install_hint(monkeypatch):
    """When the X13 binary is missing, we re-raise with an actionable message."""
    from puremacro.fetch import _seasonal

    def _raise(*args, **kwargs):
        raise FileNotFoundError("x13as not found")

    monkeypatch.setattr(_seasonal, "_x13_arima_analysis", _raise)
    with pytest.raises(RuntimeError, match=r"x13as.*brew install"):
        _seasonal.x13_seasonal_adjust(
            _make_seasonal_quarterly_series(), freq="Q"
        )


def test_x13_too_short_series_raises_clear_error():
    """X13 requires a minimum length; short input gets a clear error."""
    if not _has_x13_binary():
        pytest.skip("x13as binary not installed")
    from puremacro.fetch._seasonal import x13_seasonal_adjust
    short = _make_seasonal_quarterly_series(n_periods=6)
    with pytest.raises(RuntimeError, match=r"insufficient|short|length"):
        x13_seasonal_adjust(short, freq="Q")
