"""Tests for puremacro.sa.x13 — X-13ARIMA-SEATS seasonal adjustment for panel data.

Strategy:
- Known-DGP: build synthetic quarterly/monthly series with a known additive
  seasonal pattern; assert that deseasonalize_x13 reduces the seasonal swing
  (the SA series has a smaller Q1-Q3 or Jan-Jul spread than the raw).
- Properties: output shape, dtype, index alignment, NaN preservation.
- Contracts: min_obs threshold drops short units to NaN; freq validation raises
  ValueError; multi-unit panel produces correct per-unit handling.
- x13_available / _resolve_x13_dir: pure-Python binary-location logic.
- _x13_one fallback paths: all-NaN, too-short, no binary (monkeypatched).
- re-export: residual_seasonality_F is accessible from puremacro.sa.x13.
- No spurious network calls; when X-13 is present it is allowed to run (it is
  a local binary with no network dependency).
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import puremacro.sa.x13 as _x13_mod
from puremacro.sa.x13 import (
    _resolve_x13_dir,
    _x13_one,
    deseasonalize_x13,
    residual_seasonality_F,
    x13_available,
)

RNG = np.random.default_rng(2025)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_quarterly_panel(
    units: list[str],
    n_periods: int = 40,
    *,
    seasonal_amplitude: float = 2.0,
    noise_std: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic quarterly panel with additive seasonal pattern.

    Seasonal: +amplitude in Q1, -amplitude in Q3, 0 otherwise.
    Trend: linear from 100 to 110 per unit (shifted by unit index).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=n_periods, freq="QS")
    quarters = np.array([d.quarter for d in dates])
    seasonal = np.where(
        quarters == 1, seasonal_amplitude,
        np.where(quarters == 3, -seasonal_amplitude, 0.0),
    )
    rows = []
    for i, unit in enumerate(units):
        trend = np.linspace(100 + i * 10, 110 + i * 10, n_periods)
        noise = rng.normal(0, noise_std, n_periods)
        for j, (d, v) in enumerate(zip(dates, trend + seasonal + noise)):
            rows.append({"unit": unit, "date": d, "y": v})
    return pd.DataFrame(rows)


def _make_monthly_panel(
    units: list[str],
    n_periods: int = 72,
    *,
    seed: int = 43,
) -> pd.DataFrame:
    """Synthetic monthly panel with sinusoidal annual seasonality."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=n_periods, freq="MS")
    t = np.arange(n_periods)
    seasonal = np.sin(2 * np.pi * t / 12) * 5.0
    rows = []
    for i, unit in enumerate(units):
        trend = np.linspace(100 + i * 5, 115 + i * 5, n_periods)
        noise = rng.normal(0, 0.3, n_periods)
        for d, v in zip(dates, trend + seasonal + noise):
            rows.append({"unit": unit, "date": d, "y": v})
    return pd.DataFrame(rows)


def _quarterly_dates(n: int, start: str = "2000-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="QS")


def _monthly_dates(n: int, start: str = "2000-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="MS")


# ===========================================================================
# 1. x13_available and _resolve_x13_dir
# ===========================================================================

class TestX13Available:
    """Tests for the x13_available() helper and _resolve_x13_dir()."""

    def test_returns_bool(self):
        """x13_available() returns a Python bool."""
        result = x13_available()
        assert isinstance(result, bool)

    def test_consistent_with_module_level_dir(self):
        """x13_available() is True iff _X13_DIR is not None at import time."""
        assert x13_available() == (_x13_mod._X13_DIR is not None)

    def test_resolve_x13_dir_returns_str_or_none(self):
        """_resolve_x13_dir returns a str or None."""
        result = _resolve_x13_dir()
        assert result is None or isinstance(result, str)

    def test_env_var_file_path_resolved_to_parent(self, monkeypatch, tmp_path):
        """X13PATH pointing to the binary file itself resolves to its parent dir."""
        fake_binary = tmp_path / "x13as"
        fake_binary.write_text("", encoding="utf-8")  # create the file

        monkeypatch.setenv("X13PATH", str(fake_binary))
        result = _resolve_x13_dir()
        assert result == str(tmp_path)

    def test_env_var_dir_with_binary_resolves(self, monkeypatch, tmp_path):
        """X13PATH pointing to a directory that contains x13as resolves to that dir."""
        (tmp_path / "x13as").write_text("", encoding="utf-8")

        monkeypatch.setenv("X13PATH", str(tmp_path))
        result = _resolve_x13_dir()
        assert result == str(tmp_path)

    def test_env_var_dir_without_binary_falls_through(self, monkeypatch, tmp_path):
        """X13PATH pointing to a dir without x13as or x13ashtml does not return it."""
        # tmp_path exists but has no x13as/x13ashtml
        monkeypatch.setenv("X13PATH", str(tmp_path))
        # Also remove x13as from PATH so we fall through to ~/.local/bin check
        monkeypatch.delenv("X13PATH", raising=False)  # reset first
        monkeypatch.setenv("X13PATH", str(tmp_path))
        # Patch shutil.which to return None and ~/.local/bin/x13as to not exist
        with patch.object(shutil, "which", return_value=None):
            with patch.object(Path, "exists", return_value=False):
                result = _resolve_x13_dir()
                # Without a valid binary anywhere, should return None or the env dir
                # Contract: the dir must CONTAIN x13as to be returned
                assert result != str(tmp_path)

    def test_no_env_var_no_which_no_local_bin_returns_none(self, monkeypatch):
        """When no X13PATH, no which match, and no ~/.local/bin/x13as, returns None."""
        monkeypatch.delenv("X13PATH", raising=False)
        with patch.object(shutil, "which", return_value=None):
            # Patch the existence check for ~/.local/bin/x13as
            original_exists = Path.exists
            def _mock_exists(self):
                if "x13as" in str(self):
                    return False
                return original_exists(self)
            with patch.object(Path, "exists", _mock_exists):
                result = _resolve_x13_dir()
                assert result is None

    def test_which_x13as_resolves_to_parent(self, monkeypatch, tmp_path):
        """If x13as is on PATH (via shutil.which), its parent directory is returned."""
        fake_binary = tmp_path / "x13as"
        fake_binary.write_text("", encoding="utf-8")

        monkeypatch.delenv("X13PATH", raising=False)
        with patch.object(shutil, "which", side_effect=lambda n: str(fake_binary) if n == "x13as" else None):
            result = _resolve_x13_dir()
            assert result == str(tmp_path)

    def test_env_var_html_variant_resolves(self, monkeypatch, tmp_path):
        """X13PATH dir containing x13ashtml (not x13as) still resolves."""
        (tmp_path / "x13ashtml").write_text("", encoding="utf-8")

        monkeypatch.setenv("X13PATH", str(tmp_path))
        result = _resolve_x13_dir()
        assert result == str(tmp_path)

    def test_local_bin_fallback_resolves(self, monkeypatch, tmp_path):
        """When no env var and no PATH match, ~/.local/bin/x13as is found as last resort."""
        monkeypatch.delenv("X13PATH", raising=False)
        fake_home_local_bin = tmp_path / ".local" / "bin"
        fake_home_local_bin.mkdir(parents=True)
        (fake_home_local_bin / "x13as").write_text("", encoding="utf-8")

        with patch.object(shutil, "which", return_value=None):
            with patch.object(Path, "home", return_value=tmp_path):
                result = _resolve_x13_dir()
                assert result == str(fake_home_local_bin)


# ===========================================================================
# 2. _x13_one — internal per-series adjuster
# ===========================================================================

class TestX13One:
    """Tests for _x13_one(values, dates, period)."""

    def test_all_nan_returns_nan_array(self):
        """All-NaN input returns all-NaN output of same shape."""
        values = np.full(20, np.nan)
        dates = _quarterly_dates(20)
        result = _x13_one(values, dates, period=4)
        assert result.shape == values.shape
        assert np.all(np.isnan(result))

    def test_all_nan_output_is_float(self):
        """All-NaN input produces float output."""
        values = np.full(12, np.nan)
        dates = _quarterly_dates(12)
        result = _x13_one(values, dates, period=4)
        assert np.issubdtype(result.dtype, np.floating)

    def test_too_short_falls_back_to_stl(self, monkeypatch):
        """Series shorter than 3*period falls back to STL silently."""
        # 3*4 = 12, so 11 obs is too short
        n = 11
        values = np.linspace(100, 110, n) + RNG.normal(0, 0.1, n)
        dates = _quarterly_dates(n)
        # Even with x13 available, short series should still return an array
        result = _x13_one(values, dates, period=4)
        assert result.shape == (n,)
        assert np.all(np.isfinite(result))

    def test_sufficient_length_quarterly_returns_finite(self):
        """Series of length >= 3*4 = 12 returns finite values."""
        n = 40
        rng = np.random.default_rng(10)
        t = np.arange(n)
        seasonal = 3.0 * np.sin(2 * np.pi * t / 4)
        values = 100.0 + seasonal + rng.normal(0, 0.2, n)
        dates = _quarterly_dates(n)
        result = _x13_one(values, dates, period=4)
        assert result.shape == (n,)
        assert np.all(np.isfinite(result))

    def test_output_shape_matches_input(self):
        """Output array has the same shape as the input."""
        n = 36
        values = RNG.normal(100, 1, n)
        dates = _quarterly_dates(n)
        result = _x13_one(values, dates, period=4)
        assert result.shape == (n,)

    def test_nan_positions_preserved_in_output(self):
        """NaN positions in the input are preserved in the output."""
        n = 40
        rng = np.random.default_rng(20)
        values = 100.0 + rng.normal(0, 0.5, n)
        values[5] = np.nan
        values[15] = np.nan
        dates = _quarterly_dates(n)
        result = _x13_one(values, dates, period=4)
        assert np.isnan(result[5]), "NaN at index 5 must be preserved"
        assert np.isnan(result[15]), "NaN at index 15 must be preserved"

    def test_without_x13_binary_falls_back_to_stl(self, monkeypatch):
        """When _X13_DIR is None, falls back to STL and returns finite values."""
        n = 40
        rng = np.random.default_rng(30)
        values = 100.0 + rng.normal(0, 0.5, n)
        dates = _quarterly_dates(n)
        monkeypatch.setattr(_x13_mod, "_X13_DIR", None)
        result = _x13_one(values, dates, period=4)
        assert result.shape == (n,)
        assert np.all(np.isfinite(result))

    def test_monthly_period_returns_finite(self):
        """Monthly (period=12) data of sufficient length returns finite values."""
        n = 60
        rng = np.random.default_rng(40)
        t = np.arange(n)
        values = 100.0 + 5.0 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.3, n)
        dates = _monthly_dates(n)
        result = _x13_one(values, dates, period=12)
        assert result.shape == (n,)
        assert np.all(np.isfinite(result))

    def test_duplicate_date_index_falls_back_to_stl(self):
        """Series with duplicate date entries after period conversion falls back to STL."""
        # Create a dates array with duplicate entries after to_period/to_timestamp conversion.
        # Duplicate entries within a quarter produce duplicates after .to_period('Q').to_timestamp().
        n = 40
        rng = np.random.default_rng(55)
        values = 100.0 + rng.normal(0, 0.5, n)
        # Create a DatetimeIndex where two consecutive dates land in the same quarter period
        dates = pd.DatetimeIndex(
            [pd.Timestamp("2010-01-01"), pd.Timestamp("2010-02-01")] * (n // 2)
        )
        # This should trigger s_x.index.has_duplicates after to_period('Q').to_timestamp()
        result = _x13_one(values, dates, period=4)
        assert result.shape == (n,)
        assert np.all(np.isfinite(result))

    @pytest.mark.slow
    def test_seasonal_swing_reduced_quarterly(self):
        """_x13_one removes quarterly seasonality: Q1-Q3 swing shrinks after SA."""
        rng = np.random.default_rng(42)
        n = 48
        t = np.arange(n)
        quarter = t % 4
        seasonal = np.array([3.0, 0.5, -3.0, 0.5])[quarter]
        trend = np.linspace(100, 110, n)
        values = trend + seasonal + rng.normal(0, 0.2, n)
        dates = _quarterly_dates(n)

        result = _x13_one(values, dates, period=4)

        raw_q1_q3 = values[quarter == 0].mean() - values[quarter == 2].mean()
        sa_q1_q3 = result[quarter == 0].mean() - result[quarter == 2].mean()
        assert abs(sa_q1_q3) < abs(raw_q1_q3) * 0.4, (
            f"Seasonal swing should shrink: raw={raw_q1_q3:.3f}, sa={sa_q1_q3:.3f}"
        )


# ===========================================================================
# 3. deseasonalize_x13 — panel-level public API
# ===========================================================================

class TestDeseasonalizeX13:
    """Tests for the public deseasonalize_x13 function."""

    def test_returns_series(self):
        """Output is a pd.Series."""
        df = _make_quarterly_panel(["A"])
        result = deseasonalize_x13(df, "y", by="unit", date_col="date", freq="Q")
        assert isinstance(result, pd.Series)

    def test_output_shape_matches_input(self):
        """Output has the same length as the input DataFrame."""
        df = _make_quarterly_panel(["A"])
        result = deseasonalize_x13(df, "y", by="unit", date_col="date", freq="Q")
        assert len(result) == len(df)

    def test_output_index_equals_input_index(self):
        """Output index equals the input DataFrame's index."""
        df = _make_quarterly_panel(["A"])
        result = deseasonalize_x13(df, "y", by="unit", date_col="date", freq="Q")
        assert result.index.equals(df.index)

    def test_non_default_index_preserved(self):
        """Custom DataFrame index is preserved in the output."""
        df = _make_quarterly_panel(["A"])
        df.index = np.arange(500, 500 + len(df))
        result = deseasonalize_x13(df, "y", by="unit", date_col="date", freq="Q")
        assert result.index.equals(df.index)

    def test_output_dtype_is_float(self):
        """Output Series has float dtype."""
        df = _make_quarterly_panel(["A"])
        result = deseasonalize_x13(df, "y", by="unit", date_col="date", freq="Q")
        assert np.issubdtype(result.dtype, np.floating)

    def test_single_unit_returns_finite(self):
        """A single unit with sufficient obs returns finite seasonally-adjusted values."""
        df = _make_quarterly_panel(["A"], n_periods=40)
        result = deseasonalize_x13(df, "y", by="unit", date_col="date", freq="Q")
        assert result.notna().all(), "Single unit with 40 obs should be fully adjusted"

    def test_multi_unit_all_adjusted(self):
        """Multiple units each receive independent X-13/STL fits."""
        df = _make_quarterly_panel(["A", "B", "C"], n_periods=40)
        result = deseasonalize_x13(df, "y", by="unit", date_col="date", freq="Q")
        assert len(result) == len(df)
        assert result.notna().all(), "All units with 40 obs should be adjusted"

    def test_short_unit_below_min_obs_returns_nan(self):
        """Units with fewer than min_obs (default 24) non-NaN obs return NaN."""
        df = _make_quarterly_panel(["Short"], n_periods=15)
        result = deseasonalize_x13(df, "y", by="unit", date_col="date", freq="Q")
        # 15 < 24 -> all NaN
        assert result.isna().all(), "Unit with 15 obs should be all NaN (min_obs=24)"

    def test_long_unit_above_min_obs_returns_finite(self):
        """Units with at least min_obs obs return finite SA values."""
        df = _make_quarterly_panel(["Long"], n_periods=30)
        result = deseasonalize_x13(df, "y", by="unit", date_col="date", freq="Q")
        assert result.notna().all(), "Unit with 30 obs (>=24) should have finite SA values"

    def test_custom_min_obs_strict_drops_unit(self):
        """Custom min_obs higher than unit obs drops that unit to NaN."""
        df = _make_quarterly_panel(["A"], n_periods=20)
        result = deseasonalize_x13(df, "y", by="unit", date_col="date", freq="Q", min_obs=30)
        assert result.isna().all(), "Unit with 20 obs should be NaN when min_obs=30"

    def test_custom_min_obs_lenient_accepts_short_unit(self):
        """Custom low min_obs allows a shorter unit to be adjusted."""
        df = _make_quarterly_panel(["A"], n_periods=15)
        result = deseasonalize_x13(df, "y", by="unit", date_col="date", freq="Q", min_obs=10)
        # With min_obs=10, 15 obs > 10 so unit should be processed (not all NaN)
        assert result.notna().any(), "min_obs=10 should accept a 15-obs unit"

    def test_mixed_units_long_and_short(self):
        """Long unit is adjusted; short unit is NaN — independently."""
        df_long = _make_quarterly_panel(["Long"], n_periods=30)
        df_short = _make_quarterly_panel(["Short"], n_periods=10)
        df_mixed = pd.concat([df_long, df_short], ignore_index=True)

        result = deseasonalize_x13(df_mixed, "y", by="unit", date_col="date", freq="Q")

        long_mask = df_mixed["unit"] == "Long"
        short_mask = df_mixed["unit"] == "Short"
        assert result[long_mask].notna().all(), "Long unit should be fully adjusted"
        assert result[short_mask].isna().all(), "Short unit should be all NaN"

    def test_monthly_freq_works(self):
        """freq='M' runs X-13/STL with period=12 and returns finite values."""
        df = _make_monthly_panel(["A"])
        result = deseasonalize_x13(df, "y", by="unit", date_col="date", freq="M")
        assert isinstance(result, pd.Series)
        assert result.notna().all()

    def test_monthly_index_preserved(self):
        """Monthly output index matches input DataFrame index."""
        df = _make_monthly_panel(["A"])
        result = deseasonalize_x13(df, "y", by="unit", date_col="date", freq="M")
        assert result.index.equals(df.index)

    def test_invalid_freq_raises_value_error(self):
        """Unsupported freq value raises ValueError with informative message."""
        df = _make_quarterly_panel(["A"])
        with pytest.raises(ValueError, match=r"freq must be"):
            deseasonalize_x13(df, "y", by="unit", date_col="date", freq="W")  # type: ignore[arg-type]

    def test_invalid_freq_daily_raises_value_error(self):
        """'D' frequency also raises ValueError."""
        df = _make_quarterly_panel(["A"])
        with pytest.raises(ValueError, match=r"freq must be"):
            deseasonalize_x13(df, "y", by="unit", date_col="date", freq="D")  # type: ignore[arg-type]

    def test_output_starts_all_nan_before_filling(self):
        """Output starts as NaN; units below min_obs stay NaN (coverage of init path)."""
        df = _make_quarterly_panel(["A", "B"], n_periods=5)  # both below min_obs=24
        result = deseasonalize_x13(df, "y", by="unit", date_col="date", freq="Q")
        assert result.isna().all(), "Both units with 5 obs should remain NaN"

    def test_multi_unit_index_covers_all_rows(self):
        """Output index covers all rows from all units."""
        df = _make_quarterly_panel(["A", "B"], n_periods=40)
        result = deseasonalize_x13(df, "y", by="unit", date_col="date", freq="Q")
        assert result.index.equals(df.index)

    def test_default_date_col_is_date(self):
        """Default date_col='date' is correct; calling with that explicit arg works."""
        df = _make_quarterly_panel(["A"])
        result = deseasonalize_x13(df, "y", by="unit", date_col="date")
        assert isinstance(result, pd.Series)

    def test_without_binary_falls_back_to_stl(self, monkeypatch):
        """When _X13_DIR is None, deseasonalize_x13 falls back to STL."""
        monkeypatch.setattr(_x13_mod, "_X13_DIR", None)
        df = _make_quarterly_panel(["A"], n_periods=40)
        result = deseasonalize_x13(df, "y", by="unit", date_col="date", freq="Q")
        assert isinstance(result, pd.Series)
        assert result.notna().all()

    @pytest.mark.slow
    def test_seasonal_swing_reduced_quarterly(self):
        """deseasonalize_x13 reduces the Q1-Q3 swing for a panel with strong seasonality."""
        df = _make_quarterly_panel(["A"], seasonal_amplitude=3.0, n_periods=48)
        dates_col = df["date"]
        quarters = np.array([d.quarter for d in dates_col])

        result = deseasonalize_x13(df, "y", by="unit", date_col="date", freq="Q")
        raw = df["y"].values

        raw_q1_q3 = raw[quarters == 1].mean() - raw[quarters == 3].mean()
        sa_q1_q3 = result.values[quarters == 1].mean() - result.values[quarters == 3].mean()
        assert abs(sa_q1_q3) < abs(raw_q1_q3) * 0.4, (
            f"deseasonalize_x13 should reduce Q1-Q3 swing: raw={raw_q1_q3:.3f}, sa={sa_q1_q3:.3f}"
        )

    @pytest.mark.slow
    def test_monthly_seasonal_swing_reduced(self):
        """Monthly X-13/STL reduces cross-month dispersion of group means."""
        df = _make_monthly_panel(["A"], n_periods=72)
        dates_col = df["date"]
        months = np.array([d.month for d in dates_col])
        raw = df["y"].values

        result = deseasonalize_x13(df, "y", by="unit", date_col="date", freq="M")

        raw_group_means = [raw[months == m].mean() for m in range(1, 13)]
        sa_group_means = [result.values[months == m].mean() for m in range(1, 13)]
        raw_sd = float(np.std(raw_group_means))
        sa_sd = float(np.std(sa_group_means))
        assert sa_sd < raw_sd * 0.5, (
            f"Monthly SA should reduce cross-month dispersion: raw_sd={raw_sd:.3f}, sa_sd={sa_sd:.3f}"
        )


# ===========================================================================
# 4. residual_seasonality_F — re-exported from stl
# ===========================================================================

class TestResidualSeasonalityFReexport:
    """Confirm residual_seasonality_F is accessible from puremacro.sa.x13."""

    def test_importable_from_x13(self):
        """residual_seasonality_F is re-exported from puremacro.sa.x13."""
        from puremacro.sa.x13 import residual_seasonality_F as f
        assert callable(f)

    def test_returns_4_tuple(self):
        """Re-exported function returns a tuple of exactly 4 elements."""
        s = pd.Series(RNG.normal(5, 1, 40))
        result = residual_seasonality_F(s, freq="Q")
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_return_types(self):
        """Returns (float, float, int, int)."""
        s = pd.Series(RNG.normal(5, 1, 40))
        F, p, dof_num, dof_den = residual_seasonality_F(s, freq="Q")
        assert isinstance(F, float)
        assert isinstance(p, float)
        assert isinstance(dof_num, int)
        assert isinstance(dof_den, int)

    def test_f_statistic_nonnegative(self):
        """F statistic is non-negative for valid input."""
        s = pd.Series(RNG.normal(5, 1, 40))
        F, p, _, _ = residual_seasonality_F(s, freq="Q")
        assert F >= 0.0

    def test_p_value_in_unit_interval(self):
        """p-value is in [0, 1]."""
        s = pd.Series(RNG.normal(5, 1, 40))
        _, p, _, _ = residual_seasonality_F(s, freq="Q")
        assert 0.0 <= p <= 1.0

    def test_invalid_freq_raises(self):
        """Invalid freq raises ValueError."""
        s = pd.Series(RNG.normal(5, 1, 40))
        with pytest.raises(ValueError, match=r"freq must be"):
            residual_seasonality_F(s, freq="W")  # type: ignore[arg-type]

    def test_seasonal_series_higher_f_than_noise(self):
        """F for a strongly seasonal series exceeds F for white noise."""
        rng = np.random.default_rng(77)
        n = 60
        t = np.arange(n)
        seasonal = pd.Series(np.sin(2 * np.pi * t / 4) * 8.0 + rng.normal(0, 0.1, n))
        noise = pd.Series(rng.normal(0, 1.0, n))

        F_seasonal, _, _, _ = residual_seasonality_F(seasonal, freq="Q")
        F_noise, _, _, _ = residual_seasonality_F(noise, freq="Q")
        assert F_seasonal > F_noise, (
            f"Seasonal F ({F_seasonal:.3f}) should exceed noise F ({F_noise:.3f})"
        )


# ===========================================================================
# 5. __all__ public surface
# ===========================================================================

class TestPublicSurface:
    """Confirm __all__ matches the documented public API."""

    def test_all_contains_expected_names(self):
        """__all__ includes the three documented public names."""
        expected = {"deseasonalize_x13", "residual_seasonality_F", "x13_available"}
        assert expected.issubset(set(_x13_mod.__all__))

    def test_all_names_are_importable(self):
        """Each name in __all__ is actually importable from the module."""
        import importlib
        mod = importlib.import_module("puremacro.sa.x13")
        for name in mod.__all__:
            assert hasattr(mod, name), f"{name!r} in __all__ but not in module"
