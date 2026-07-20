"""Focused coverage tests for puremacro.connectedness.diebold_yilmaz.

Missing lines targeted (from first coverage run at 59%):
  64-76  rolling-window branch (window is not None)
  83-85  cholesky identification branch
  90     unknown identification raise ValueError

Test strategy
-------------
- known_value / math: verify total, directional, and net from theta via formulas
  with a known synthetic dataset.
- property: shapes, dtypes, bounds [0,100], row-sum normalisation of theta,
  net sums to zero, total_series is a pd.Series with correct length.
- contract: order-invariance of gfevd (Pesaran-Shin) vs. order-sensitivity of
  cholesky; consistency between window=None and window=T.
- edge: bad identification string raises ValueError; window=T (single window).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.connectedness.diebold_yilmaz import spillover_index


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_panel(n: int = 3, T: int = 200, seed: int = 42) -> pd.DataFrame:
    """Independent Gaussian series (low spillover expected)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.standard_normal((T, n)),
        columns=[f"y{i}" for i in range(n)],
    )


def _make_connected_panel(n: int = 3, T: int = 300, noise_scale: float = 0.1, seed: int = 77) -> pd.DataFrame:
    """Series driven by a common factor (high spillover expected)."""
    rng = np.random.default_rng(seed)
    common = rng.standard_normal(T)
    cols = {}
    for i in range(n):
        cols[f"y{i}"] = common + noise_scale * rng.standard_normal(T)
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# Tests for the cholesky identification branch (lines 83-85)
# ---------------------------------------------------------------------------

class TestCholeskyIdentification:
    def test_cholesky_returns_all_keys(self):
        df = _make_panel()
        out = spillover_index(df, var_lags=2, fevd_horizon=10, identification="cholesky")
        for key in ("total", "directional_to", "directional_from", "net", "pairwise_matrix"):
            assert key in out, f"missing key: {key}"

    def test_cholesky_total_in_valid_range(self):
        df = _make_panel()
        out = spillover_index(df, var_lags=2, fevd_horizon=10, identification="cholesky")
        assert 0.0 <= out["total"] <= 100.0

    def test_cholesky_output_shapes(self):
        n = 4
        df = _make_panel(n=n, T=200)
        out = spillover_index(df, var_lags=2, fevd_horizon=10, identification="cholesky")
        assert out["directional_to"].shape == (n,)
        assert out["directional_from"].shape == (n,)
        assert out["net"].shape == (n,)
        assert out["pairwise_matrix"].shape == (n, n)

    def test_cholesky_theta_row_sums_to_one(self):
        """Cholesky FEVD is normalised: each row of theta sums to 1."""
        df = _make_panel()
        out = spillover_index(df, var_lags=2, fevd_horizon=10, identification="cholesky")
        theta = out["pairwise_matrix"]
        np.testing.assert_allclose(theta.sum(axis=1), np.ones(theta.shape[0]), atol=1e-6)

    def test_cholesky_pairwise_nonnegative(self):
        df = _make_panel()
        out = spillover_index(df, var_lags=2, fevd_horizon=10, identification="cholesky")
        assert np.all(out["pairwise_matrix"] >= -1e-10)

    def test_cholesky_net_sums_to_zero(self):
        """net = dir_to - dir_from; sum(dir_to) == sum(dir_from), so sum(net) ~ 0."""
        df = _make_panel()
        out = spillover_index(df, var_lags=2, fevd_horizon=10, identification="cholesky")
        assert abs(out["net"].sum()) < 1e-10

    def test_cholesky_math_consistency(self):
        """total should equal 100 * (off-diag sum of theta) / n."""
        df = _make_panel()
        out = spillover_index(df, var_lags=2, fevd_horizon=10, identification="cholesky")
        theta = out["pairwise_matrix"]
        n = theta.shape[0]
        expected_total = 100.0 * (theta.sum() - np.trace(theta)) / n
        assert abs(out["total"] - expected_total) < 1e-10

    def test_cholesky_order_sensitivity(self):
        """Cholesky FEVD depends on column ordering — reversing columns changes total."""
        df = _make_connected_panel()
        df_rev = df[df.columns[::-1]]
        out1 = spillover_index(df, var_lags=2, fevd_horizon=10, identification="cholesky")
        out2 = spillover_index(df_rev, var_lags=2, fevd_horizon=10, identification="cholesky")
        # They should differ (ordering matters for Cholesky) — not identical
        assert not np.isclose(out1["total"], out2["total"], atol=1e-4), (
            "Cholesky totals were identical despite reversed ordering — expected order-sensitivity"
        )


# ---------------------------------------------------------------------------
# Tests for the unknown identification branch (line 90)
# ---------------------------------------------------------------------------

class TestUnknownIdentification:
    def test_bad_identification_raises_value_error(self):
        df = _make_panel()
        with pytest.raises(ValueError, match="unknown identification"):
            spillover_index(df, var_lags=2, fevd_horizon=10, identification="bad_id")

    def test_bad_identification_message_includes_name(self):
        df = _make_panel()
        with pytest.raises(ValueError, match="unexpected_method"):
            spillover_index(df, var_lags=2, fevd_horizon=10, identification="unexpected_method")


# ---------------------------------------------------------------------------
# Tests for the rolling window branch (lines 64-76)
# ---------------------------------------------------------------------------

class TestRollingWindow:
    def test_window_adds_total_series_key(self):
        df = _make_panel(T=120)
        out = spillover_index(df, var_lags=2, fevd_horizon=10, window=60)
        assert "total_series" in out

    def test_no_window_no_total_series_key(self):
        df = _make_panel(T=120)
        out = spillover_index(df, var_lags=2, fevd_horizon=10, window=None)
        assert "total_series" not in out

    def test_window_total_series_is_pd_series(self):
        df = _make_panel(T=120)
        out = spillover_index(df, var_lags=2, fevd_horizon=10, window=60)
        assert isinstance(out["total_series"], pd.Series)

    def test_window_total_series_length(self):
        """Series should have T - window + 1 entries (one per valid window end)."""
        T, window = 100, 40
        df = _make_panel(T=T)
        out = spillover_index(df, var_lags=2, fevd_horizon=10, window=window)
        assert len(out["total_series"]) == T - window + 1

    def test_window_total_series_values_finite(self):
        df = _make_panel(T=120)
        out = spillover_index(df, var_lags=2, fevd_horizon=10, window=60)
        assert np.all(np.isfinite(out["total_series"].values))

    def test_window_total_series_in_valid_range(self):
        df = _make_panel(T=120)
        out = spillover_index(df, var_lags=2, fevd_horizon=10, window=60)
        ts = out["total_series"]
        assert (ts >= 0).all() and (ts <= 100).all()

    def test_window_total_series_index_matches_panel_index(self):
        """The last date of each window should align with the panel's own index."""
        T, window = 80, 40
        dates = pd.date_range("2010Q1", periods=T, freq="QE")
        rng = np.random.default_rng(55)
        df = pd.DataFrame(rng.standard_normal((T, 2)), index=dates, columns=["y1", "y2"])
        out = spillover_index(df, var_lags=2, fevd_horizon=10, window=window)
        ts = out["total_series"]
        # First entry's date should be the date of the last obs in the first window
        assert ts.index[0] == dates[window - 1]
        # Last entry's date should be the last date of the full panel
        assert ts.index[-1] == dates[-1]

    def test_window_equal_to_t_gives_length_one_series(self):
        T = 80
        df = _make_panel(T=T)
        out = spillover_index(df, var_lags=2, fevd_horizon=10, window=T)
        assert len(out["total_series"]) == 1

    def test_window_full_sample_total_equals_no_window_total(self):
        """window=T should give total == full-sample total (same data)."""
        T = 100
        df = _make_panel(T=T)
        out_full = spillover_index(df, var_lags=2, fevd_horizon=10, window=None)
        out_win = spillover_index(df, var_lags=2, fevd_horizon=10, window=T)
        assert np.isclose(out_full["total"], out_win["total"], atol=1e-8)

    def test_window_still_returns_other_keys(self):
        """Rolling-window output should contain all standard keys plus total_series."""
        df = _make_panel(T=100)
        out = spillover_index(df, var_lags=2, fevd_horizon=10, window=50)
        for key in ("total", "directional_to", "directional_from", "net", "pairwise_matrix"):
            assert key in out

    def test_window_skips_failing_subwindow(self):
        """Rolling window silently skips sub-windows where estimation fails
        (except np.linalg.LinAlgError, ValueError: continue).
        We monkeypatch _compute_one to raise on the first sub-window call,
        then succeed on remaining calls; the total_series should be shorter.
        """
        import puremacro.connectedness.diebold_yilmaz as _mod

        T, window = 80, 40
        df = _make_panel(T=T)
        expected_full_len = T - window + 1  # 41

        call_count = [0]
        original_compute_one = _mod._compute_one

        def _flaky_compute_one(Y, p, H, identification="gfevd"):
            call_count[0] += 1
            # Raise on first sub-window call (call_count==1); succeed otherwise
            if call_count[0] == 1:
                raise np.linalg.LinAlgError("simulated singular matrix")
            return original_compute_one(Y, p, H, identification)

        _mod._compute_one = _flaky_compute_one
        try:
            out = spillover_index(df, var_lags=2, fevd_horizon=10, window=window)
        finally:
            _mod._compute_one = original_compute_one

        ts = out["total_series"]
        # One sub-window was skipped, so total_series is one element shorter
        assert len(ts) == expected_full_len - 1
        # Result is still a valid pd.Series
        assert isinstance(ts, pd.Series)
        assert np.all(np.isfinite(ts.values))


# ---------------------------------------------------------------------------
# Tests for gfevd identification (order-invariance property)
# ---------------------------------------------------------------------------

class TestGfevdOrderInvariance:
    def test_gfevd_total_is_order_invariant(self):
        """GFEVD (Pesaran-Shin) total spillover should be the same regardless
        of column ordering."""
        df = _make_connected_panel()
        df_rev = df[df.columns[::-1]]
        out1 = spillover_index(df, var_lags=2, fevd_horizon=10, identification="gfevd")
        out2 = spillover_index(df_rev, var_lags=2, fevd_horizon=10, identification="gfevd")
        np.testing.assert_allclose(out1["total"], out2["total"], atol=1e-6)

    def test_gfevd_theta_row_sums_to_one(self):
        df = _make_panel()
        out = spillover_index(df, var_lags=2, fevd_horizon=10, identification="gfevd")
        theta = out["pairwise_matrix"]
        np.testing.assert_allclose(theta.sum(axis=1), np.ones(theta.shape[0]), atol=1e-6)

    def test_gfevd_math_consistency(self):
        """total = 100 * off_diag_sum / n; dir_to - dir_from == net."""
        df = _make_panel()
        out = spillover_index(df, var_lags=2, fevd_horizon=10, identification="gfevd")
        theta = out["pairwise_matrix"]
        n = theta.shape[0]
        expected_total = 100.0 * (theta.sum() - np.trace(theta)) / n
        np.testing.assert_allclose(out["total"], expected_total, atol=1e-10)

        expected_dir_to = 100.0 * (theta.sum(axis=0) - np.diag(theta)) / n
        expected_dir_from = 100.0 * (theta.sum(axis=1) - np.diag(theta)) / n
        np.testing.assert_allclose(out["directional_to"], expected_dir_to, atol=1e-10)
        np.testing.assert_allclose(out["directional_from"], expected_dir_from, atol=1e-10)
        np.testing.assert_allclose(out["net"], expected_dir_to - expected_dir_from, atol=1e-10)

    def test_gfevd_net_sums_to_zero(self):
        df = _make_panel()
        out = spillover_index(df, var_lags=2, fevd_horizon=10, identification="gfevd")
        assert abs(out["net"].sum()) < 1e-10

    def test_gfevd_total_in_valid_range(self):
        df = _make_panel()
        out = spillover_index(df, var_lags=2, fevd_horizon=10, identification="gfevd")
        assert 0.0 <= out["total"] <= 100.0

    def test_connected_vs_independent_monotonicity_gfevd(self):
        """Highly connected panel -> higher total spillover than independent panel."""
        df_low = _make_panel(n=3, T=300)
        df_high = _make_connected_panel(n=3, T=300, noise_scale=0.1)
        out_low = spillover_index(df_low, var_lags=2, fevd_horizon=10, identification="gfevd")
        out_high = spillover_index(df_high, var_lags=2, fevd_horizon=10, identification="gfevd")
        assert out_high["total"] > out_low["total"]

    def test_connected_vs_independent_monotonicity_cholesky(self):
        """Same monotonicity check with cholesky identification."""
        df_low = _make_panel(n=3, T=300)
        df_high = _make_connected_panel(n=3, T=300, noise_scale=0.1)
        out_low = spillover_index(df_low, var_lags=2, fevd_horizon=10, identification="cholesky")
        out_high = spillover_index(df_high, var_lags=2, fevd_horizon=10, identification="cholesky")
        assert out_high["total"] > out_low["total"]
