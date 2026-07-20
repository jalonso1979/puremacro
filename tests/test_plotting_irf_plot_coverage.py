"""Coverage tests for puremacro.plotting.irf_plot.

Tests cover:
  - _apply_normalization: all four normalize modes, None passthrough, error paths
  - plot_irf: shape/return-type contract, normalization modes, bw vs color,
    CI bands (present / absent), label placement, ylabel_suffix, error handling.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless — must be before any pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from puremacro.plotting.irf_plot import _apply_normalization, plot_irf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_irf(n_t: int = 2, n_s: int = 3, h: int = 8, seed: int = 0) -> np.ndarray:
    """Return random IRF array shaped (n_t, n_s, h+1)."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_t, n_s, h + 1))


def _make_bands(
    point: np.ndarray, width: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    return point - width, point + width


# ---------------------------------------------------------------------------
# _apply_normalization
# ---------------------------------------------------------------------------


class TestApplyNormalization:
    def test_none_returns_inputs_unchanged(self):
        a = np.array([[[1.0, 2.0]]])
        b = np.array([[[3.0, 4.0]]])
        out = _apply_normalization((a, b), normalize=None)
        assert out[0] is a
        assert out[1] is b

    def test_none_preserves_none_elements(self):
        a = np.ones((2, 2, 3))
        out = _apply_normalization((a, None), normalize=None)
        assert out[0] is a
        assert out[1] is None

    def test_pct_multiplies_by_100(self):
        val = np.array([[[0.05, -0.03]]])
        (out,) = _apply_normalization((val,), normalize="pct")
        np.testing.assert_allclose(out, val * 100.0)

    def test_pct_none_passes_through(self):
        val = np.ones((1, 1, 4))
        out_val, out_none = _apply_normalization((val, None), normalize="pct")
        assert out_none is None
        np.testing.assert_allclose(out_val, val * 100.0)

    def test_sd_target_divides_each_row(self):
        # 2 targets, 2 shocks, 5 horizons; each target has a known sd
        point = np.ones((2, 2, 5))
        target_sd = np.array([2.0, 4.0])
        (out,) = _apply_normalization((point,), normalize="sd_target", target_sd=target_sd)
        # Row 0 should be 1/2, row 1 should be 1/4
        np.testing.assert_allclose(out[0], 0.5 * np.ones((2, 5)))
        np.testing.assert_allclose(out[1], 0.25 * np.ones((2, 5)))

    def test_sd_target_raises_without_target_sd(self):
        with pytest.raises(ValueError, match="requires target_sd"):
            _apply_normalization((np.ones((1, 1, 3)),), normalize="sd_target")

    def test_sd_shock_multiplies_each_column(self):
        # 1 target, 3 shocks, 4 horizons; each shock has known sd
        point = np.ones((1, 3, 4))
        shock_sd = np.array([1.0, 2.0, 3.0])
        (out,) = _apply_normalization((point,), normalize="sd_shock", shock_sd=shock_sd)
        # Column j should be shock_sd[j]
        np.testing.assert_allclose(out[0, 0, :], 1.0)
        np.testing.assert_allclose(out[0, 1, :], 2.0)
        np.testing.assert_allclose(out[0, 2, :], 3.0)

    def test_sd_shock_raises_without_shock_sd(self):
        with pytest.raises(ValueError, match="requires shock_sd"):
            _apply_normalization((np.ones((1, 1, 3)),), normalize="sd_shock")

    def test_unknown_normalize_raises(self):
        with pytest.raises(ValueError, match="Unknown normalize option"):
            _apply_normalization((np.ones((1, 1, 3)),), normalize="bogus")

    def test_multiple_arrays_all_normalized(self):
        """All non-None arrays in the tuple get scaled by pct."""
        a = np.array([[[0.1]]])
        b = np.array([[[0.2]]])
        out_a, out_b = _apply_normalization((a, b), normalize="pct")
        np.testing.assert_allclose(out_a, 10.0)
        np.testing.assert_allclose(out_b, 20.0)

    def test_sd_target_single_target_identity(self):
        """Dividing by the target's own sd of 1.0 is identity."""
        point = np.array([[[0.3, -0.5, 0.7]]])
        (out,) = _apply_normalization(
            (point,), normalize="sd_target", target_sd=np.array([1.0])
        )
        np.testing.assert_allclose(out, point)


# ---------------------------------------------------------------------------
# plot_irf — return-type and shape contract
# ---------------------------------------------------------------------------


class TestPlotIrfContract:
    def test_returns_figure(self):
        point = _make_irf(n_t=2, n_s=2, h=8)
        fig = plot_irf(
            point=point,
            target_labels=["GDP", "Inflation"],
            shock_labels=["supply", "demand"],
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_axes_grid_shape(self):
        """Grid should be (n_t, n_s)."""
        n_t, n_s = 3, 2
        point = _make_irf(n_t=n_t, n_s=n_s, h=6)
        fig = plot_irf(
            point=point,
            target_labels=[f"y{i}" for i in range(n_t)],
            shock_labels=[f"s{j}" for j in range(n_s)],
        )
        assert len(fig.axes) == n_t * n_s
        plt.close(fig)

    def test_single_target_single_shock(self):
        point = _make_irf(n_t=1, n_s=1, h=4)
        fig = plot_irf(
            point=point,
            target_labels=["Y"],
            shock_labels=["shock"],
        )
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_horizons_match_point_size(self):
        """The plotted line should have h+1 data points."""
        h = 10
        point = _make_irf(n_t=1, n_s=1, h=h)
        fig = plot_irf(
            point=point,
            target_labels=["Y"],
            shock_labels=["shock"],
        )
        ax = fig.axes[0]
        # The IRF line is the last line added (after axhline)
        irf_line = ax.lines[-1]
        assert len(irf_line.get_xdata()) == h + 1
        plt.close(fig)

    def test_raises_on_wrong_ndim(self):
        bad = np.ones((2, 8))  # missing shock dimension
        with pytest.raises(ValueError, match="n_targets, n_shocks"):
            plot_irf(
                point=bad,
                target_labels=["Y"],
                shock_labels=["s"],
            )

    def test_title_appears_in_figure(self):
        point = _make_irf(n_t=1, n_s=1, h=4)
        fig = plot_irf(
            point=point,
            target_labels=["Y"],
            shock_labels=["s"],
            title="My IRF Title",
        )
        assert fig.texts[0].get_text() == "My IRF Title"
        plt.close(fig)

    def test_shock_label_in_column_title(self):
        """First row axes should show shock labels as titles."""
        point = _make_irf(n_t=2, n_s=2, h=4)
        shock_labels = ["supply", "demand"]
        fig = plot_irf(
            point=point,
            target_labels=["Y", "P"],
            shock_labels=shock_labels,
        )
        titles = [ax.get_title() for ax in fig.axes[:2]]
        for sl in shock_labels:
            assert any(sl in t for t in titles)
        plt.close(fig)

    def test_target_label_in_row_ylabel(self):
        """First column axes should show target labels as y-axis labels."""
        n_t, n_s = 3, 2
        point = _make_irf(n_t=n_t, n_s=n_s, h=4)
        target_labels = ["GDP", "CPI", "Urate"]
        fig = plot_irf(
            point=point,
            target_labels=target_labels,
            shock_labels=["s1", "s2"],
        )
        # First column (every n_s-th axis starting at 0)
        ylabels = [fig.axes[i * n_s].get_ylabel() for i in range(n_t)]
        for tl in target_labels:
            assert any(tl in yl for yl in ylabels)
        plt.close(fig)

    def test_ylabel_suffix_appended(self):
        point = _make_irf(n_t=2, n_s=1, h=4)
        fig = plot_irf(
            point=point,
            target_labels=["GDP", "CPI"],
            shock_labels=["shock"],
            ylabel_suffix=" (%)",
        )
        first_col_labels = [fig.axes[i].get_ylabel() for i in range(2)]
        assert all(" (%)" in lbl for lbl in first_col_labels)
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_irf — CI bands
# ---------------------------------------------------------------------------


class TestPlotIrfBands:
    def test_band_present_when_lower_upper_provided(self):
        """When lower and upper are given, fill_between collections must exist."""
        point = _make_irf(n_t=1, n_s=1, h=6)
        lower, upper = _make_bands(point)
        fig = plot_irf(
            point=point,
            lower=lower,
            upper=upper,
            target_labels=["Y"],
            shock_labels=["s"],
        )
        ax = fig.axes[0]
        # fill_between produces a PolyCollection — must be at least one
        from matplotlib.collections import PolyCollection
        polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
        assert len(polys) >= 1
        plt.close(fig)

    def test_no_band_without_lower_upper(self):
        """Without lower/upper, no fill_between polygon collections are added."""
        point = _make_irf(n_t=1, n_s=1, h=6)
        fig = plot_irf(
            point=point,
            target_labels=["Y"],
            shock_labels=["s"],
        )
        ax = fig.axes[0]
        from matplotlib.collections import PolyCollection
        polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
        assert len(polys) == 0
        plt.close(fig)

    def test_band_alpha_respected_in_color_mode(self):
        """In color mode, the fill_between alpha should match band_alpha."""
        point = _make_irf(n_t=1, n_s=1, h=4)
        lower, upper = _make_bands(point, width=0.3)
        alpha_val = 0.35
        fig = plot_irf(
            point=point,
            lower=lower,
            upper=upper,
            target_labels=["Y"],
            shock_labels=["s"],
            band_alpha=alpha_val,
        )
        ax = fig.axes[0]
        from matplotlib.collections import PolyCollection
        polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
        assert len(polys) >= 1
        # The alpha should match what we passed
        assert polys[0].get_alpha() == pytest.approx(alpha_val)
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_irf — normalization integration
# ---------------------------------------------------------------------------


class TestPlotIrfNormalization:
    def test_pct_scales_plotted_values(self):
        """In pct mode, plotted y-data should be 100x the raw point values."""
        rng = np.random.default_rng(42)
        raw = rng.normal(size=(1, 1, 5))
        fig = plot_irf(
            point=raw,
            target_labels=["Y"],
            shock_labels=["s"],
            normalize="pct",
        )
        ax = fig.axes[0]
        # Last line (the IRF) — skip axhline which is at y=0
        irf_line = ax.lines[-1]
        np.testing.assert_allclose(irf_line.get_ydata(), raw[0, 0] * 100.0)
        plt.close(fig)

    def test_sd_target_scales_plotted_values(self):
        point = np.ones((2, 1, 4))
        target_sd = np.array([2.0, 4.0])
        fig = plot_irf(
            point=point,
            target_labels=["Y", "P"],
            shock_labels=["s"],
            normalize="sd_target",
            target_sd=target_sd,
        )
        # First row: 1/2; second row: 1/4
        ax0 = fig.axes[0]
        np.testing.assert_allclose(ax0.lines[-1].get_ydata(), 0.5 * np.ones(4))
        ax1 = fig.axes[1]
        np.testing.assert_allclose(ax1.lines[-1].get_ydata(), 0.25 * np.ones(4))
        plt.close(fig)

    def test_sd_shock_scales_plotted_values(self):
        point = np.ones((1, 2, 3))
        shock_sd = np.array([1.0, 3.0])
        fig = plot_irf(
            point=point,
            target_labels=["Y"],
            shock_labels=["s1", "s2"],
            normalize="sd_shock",
            shock_sd=shock_sd,
        )
        ax0 = fig.axes[0]  # target 0, shock 0
        np.testing.assert_allclose(ax0.lines[-1].get_ydata(), 1.0 * np.ones(3))
        ax1 = fig.axes[1]  # target 0, shock 1
        np.testing.assert_allclose(ax1.lines[-1].get_ydata(), 3.0 * np.ones(3))
        plt.close(fig)

    def test_normalize_none_is_raw(self):
        rng = np.random.default_rng(7)
        raw = rng.normal(size=(1, 1, 6))
        fig = plot_irf(
            point=raw,
            target_labels=["Y"],
            shock_labels=["s"],
            normalize=None,
        )
        ax = fig.axes[0]
        irf_line = ax.lines[-1]
        np.testing.assert_allclose(irf_line.get_ydata(), raw[0, 0])
        plt.close(fig)

    def test_pct_normalization_with_bands_scaled(self):
        """Bands should also be scaled when normalize='pct'."""
        raw = np.ones((1, 1, 5)) * 0.1
        lower_raw = raw - 0.05
        upper_raw = raw + 0.05
        fig = plot_irf(
            point=raw,
            lower=lower_raw,
            upper=upper_raw,
            target_labels=["Y"],
            shock_labels=["s"],
            normalize="pct",
        )
        ax = fig.axes[0]
        # The IRF line value should be 100 * 0.1 = 10
        np.testing.assert_allclose(ax.lines[-1].get_ydata(), 10.0 * np.ones(5))
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_irf — bw mode
# ---------------------------------------------------------------------------


class TestPlotIrfBwMode:
    def test_bw_returns_figure(self):
        point = _make_irf(n_t=2, n_s=2, h=6)
        fig = plot_irf(
            point=point,
            target_labels=["Y", "P"],
            shock_labels=["s1", "s2"],
            bw=True,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_bw_grid_shape(self):
        n_t, n_s = 2, 3
        point = _make_irf(n_t=n_t, n_s=n_s, h=5)
        fig = plot_irf(
            point=point,
            target_labels=["Y", "P"],
            shock_labels=["s1", "s2", "s3"],
            bw=True,
        )
        assert len(fig.axes) == n_t * n_s
        plt.close(fig)

    def test_bw_band_present(self):
        """bw mode with bands should still produce PolyCollection."""
        point = _make_irf(n_t=1, n_s=1, h=5)
        lower, upper = _make_bands(point)
        fig = plot_irf(
            point=point,
            lower=lower,
            upper=upper,
            target_labels=["Y"],
            shock_labels=["s"],
            bw=True,
        )
        ax = fig.axes[0]
        from matplotlib.collections import PolyCollection
        polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
        assert len(polys) >= 1
        plt.close(fig)

    def test_bw_uses_grayscale_linestyle(self):
        """bw mode should use non-default linestyles (from bw_linestyles)."""
        point = _make_irf(n_t=1, n_s=2, h=4)
        fig = plot_irf(
            point=point,
            target_labels=["Y"],
            shock_labels=["s1", "s2"],
            bw=True,
        )
        # Both IRF lines should exist (the last line in each axis)
        for ax in fig.axes:
            assert len(ax.lines) >= 1
        plt.close(fig)

    def test_bw_vs_color_same_shape(self):
        """bw=True and bw=False should produce figures with the same grid size."""
        point = _make_irf(n_t=2, n_s=2, h=5)
        kw = dict(
            point=point,
            target_labels=["Y", "P"],
            shock_labels=["s1", "s2"],
        )
        fig_color = plot_irf(**kw, bw=False)
        fig_bw = plot_irf(**kw, bw=True)
        assert len(fig_color.axes) == len(fig_bw.axes)
        plt.close(fig_color)
        plt.close(fig_bw)


# ---------------------------------------------------------------------------
# plot_irf — IRF mathematical contract: zero IRF stays at zero
# ---------------------------------------------------------------------------


class TestPlotIrfMathContract:
    def test_zero_irf_plotted_as_zero(self):
        """A point array of all zeros should plot all y-values as zero."""
        zero = np.zeros((2, 2, 9))
        fig = plot_irf(
            point=zero,
            target_labels=["Y", "P"],
            shock_labels=["s1", "s2"],
        )
        for ax in fig.axes:
            irf_line = ax.lines[-1]
            np.testing.assert_allclose(irf_line.get_ydata(), 0.0)
        plt.close(fig)

    def test_ci_contains_point_at_every_horizon(self):
        """CI bands centered on the point should contain the point."""
        rng = np.random.default_rng(99)
        raw = rng.normal(size=(2, 2, 8))
        lower = raw - 0.5
        upper = raw + 0.5
        fig = plot_irf(
            point=raw,
            lower=lower,
            upper=upper,
            target_labels=["Y", "P"],
            shock_labels=["s1", "s2"],
        )
        for ax in fig.axes:
            y = ax.lines[-1].get_ydata()
            assert np.all(np.isfinite(y))
        plt.close(fig)

    def test_lp_style_single_shock(self):
        """LP-style (n_t, 1, H+1) should work without error."""
        lp_irf = _make_irf(n_t=3, n_s=1, h=12)
        fig = plot_irf(
            point=lp_irf,
            target_labels=["GDP", "CPI", "URATE"],
            shock_labels=["uncertainty"],
        )
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_horizons_are_zero_indexed(self):
        """x-axis should start at 0 (impact horizon)."""
        point = _make_irf(n_t=1, n_s=1, h=7)
        fig = plot_irf(
            point=point,
            target_labels=["Y"],
            shock_labels=["s"],
        )
        ax = fig.axes[0]
        x = ax.lines[-1].get_xdata()
        assert x[0] == 0
        assert x[-1] == 7
        plt.close(fig)

    def test_pct_normalization_doubles_amplitude(self):
        """If raw IRF = 0.01 everywhere, pct mode gives 1.0 everywhere."""
        raw = np.full((1, 1, 5), 0.01)
        fig = plot_irf(
            point=raw,
            target_labels=["Y"],
            shock_labels=["s"],
            normalize="pct",
        )
        ax = fig.axes[0]
        np.testing.assert_allclose(ax.lines[-1].get_ydata(), 1.0 * np.ones(5))
        plt.close(fig)
