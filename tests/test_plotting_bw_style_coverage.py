"""Coverage tests for puremacro.plotting.bw_style.

Public API under test:
    bw_colors(n)         -> list[str] of n grayscale shade strings
    bw_linestyles(n)     -> list of n matplotlib linestyle specs
    bw_hatches(n)        -> list[str] of n hatch pattern strings
    apply_bw_style(*, dpi, serif) -> sets matplotlib rcParams globally
    bw_context(*, dpi, serif)     -> context manager that applies + restores style
    save_pub(fig, path_no_ext, dpi) -> saves PDF+PNG, returns list of 2 paths

Tests cover:
  - Known-value: bw_colors return values match _GRAYS for small n; match
    np.linspace(0.0, 0.8, n) for n > 8 (the two distinct code branches).
  - Properties: length, string format ("0.xxx"), values in [0, 1) for gray palette.
  - bw_linestyles: correct slicing for n <= 8; wrapping/cycling for n > 8.
  - bw_hatches: correct slicing for n <= 8; wrapping/cycling for n > 8.
  - apply_bw_style: verifies key rcParam values are set; serif vs sans-serif.
  - bw_context: rcParams are restored after the block; context yields control;
    restoration happens even when an exception is raised.
  - save_pub: both files are created on disk; the returned list has 2 paths
    with correct extensions; sub-directory creation; returned paths exist.
"""
from __future__ import annotations

import os
import tempfile

import matplotlib

matplotlib.use("Agg")  # headless — must precede any pyplot import

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

from puremacro.plotting.bw_style import (
    _GRAYS,
    _HATCHES,
    _LINESTYLES,
    apply_bw_style,
    bw_colors,
    bw_context,
    bw_hatches,
    bw_linestyles,
    save_pub,
)


# ===========================================================================
# bw_colors
# ===========================================================================


class TestBwColors:
    """Known-value and property tests for bw_colors(n)."""

    # -----------------------------------------------------------------------
    # Return type and format
    # -----------------------------------------------------------------------

    def test_returns_list(self):
        result = bw_colors(4)
        assert isinstance(result, list)

    def test_returns_strings(self):
        """Every element must be a string (matplotlib color spec)."""
        for c in bw_colors(6):
            assert isinstance(c, str)

    def test_format_three_decimal_places(self):
        """Each element must match the '0.ddd' format."""
        import re
        pattern = re.compile(r"^\d+\.\d{3}$")
        for c in bw_colors(8):
            assert pattern.match(c), f"{c!r} does not match expected format"

    # -----------------------------------------------------------------------
    # Length contract
    # -----------------------------------------------------------------------

    def test_length_equals_n_small(self):
        for n in range(1, 9):
            assert len(bw_colors(n)) == n

    def test_length_equals_n_large(self):
        """For n > 8 (linspace branch) length must still equal n."""
        for n in (9, 10, 15, 20):
            assert len(bw_colors(n)) == n

    # -----------------------------------------------------------------------
    # Known-value: n <= len(_GRAYS) uses _GRAYS directly
    # -----------------------------------------------------------------------

    def test_n1_equals_first_gray(self):
        result = bw_colors(1)
        expected = f"{_GRAYS[0]:.3f}"
        assert result[0] == expected

    def test_n4_matches_grays_slice(self):
        result = bw_colors(4)
        for i, color in enumerate(result):
            assert color == f"{_GRAYS[i]:.3f}"

    def test_n8_matches_full_grays(self):
        """Exactly 8 colors — uses the _GRAYS list without linspace."""
        result = bw_colors(8)
        assert len(result) == 8
        for i, color in enumerate(result):
            assert color == f"{_GRAYS[i]:.3f}"

    # -----------------------------------------------------------------------
    # Known-value: n > len(_GRAYS) uses np.linspace(0.0, 0.8, n)
    # -----------------------------------------------------------------------

    def test_n9_uses_linspace(self):
        """For n=9 the result switches to linspace."""
        result = bw_colors(9)
        expected_floats = np.linspace(0.0, 0.8, 9)
        for color, expected in zip(result, expected_floats):
            assert color == f"{expected:.3f}"

    def test_n10_uses_linspace(self):
        result = bw_colors(10)
        expected_floats = np.linspace(0.0, 0.8, 10)
        for color, expected in zip(result, expected_floats):
            assert color == f"{expected:.3f}"

    def test_n20_uses_linspace(self):
        result = bw_colors(20)
        expected_floats = np.linspace(0.0, 0.8, 20)
        for color, expected in zip(result, expected_floats):
            assert color == f"{expected:.3f}"

    # -----------------------------------------------------------------------
    # Value range: all shades in [0.0, 0.8]
    # -----------------------------------------------------------------------

    def test_all_values_in_valid_range_small(self):
        """Grayscale floats must be in [0.0, 0.8] (avoids pure white)."""
        for color in bw_colors(8):
            val = float(color)
            assert 0.0 <= val <= 0.8, f"{color!r} out of [0, 0.8]"

    def test_all_values_in_valid_range_large(self):
        for color in bw_colors(12):
            val = float(color)
            assert 0.0 <= val <= 0.8, f"{color!r} out of [0, 0.8]"

    def test_values_are_valid_matplotlib_colors(self):
        """Matplotlib must accept these strings without raising."""
        colors = bw_colors(8)
        fig, ax = plt.subplots()
        for c in colors:
            ax.plot([0, 1], color=c)  # should not raise
        plt.close(fig)


# ===========================================================================
# bw_linestyles
# ===========================================================================


class TestBwLinestyles:
    """Tests for bw_linestyles(n)."""

    def test_returns_list(self):
        assert isinstance(bw_linestyles(4), list)

    def test_length_equals_n_small(self):
        for n in range(1, 9):
            assert len(bw_linestyles(n)) == n

    def test_length_equals_n_large(self):
        """Wrapping branch: length must still equal n."""
        for n in (9, 12, 16):
            assert len(bw_linestyles(n)) == n

    def test_n8_is_full_linestyles(self):
        """Exactly 8 returns all of _LINESTYLES."""
        result = bw_linestyles(8)
        assert result == _LINESTYLES

    def test_n4_is_prefix_of_linestyles(self):
        """First 4 linestyles are a prefix of _LINESTYLES."""
        result = bw_linestyles(4)
        assert result == _LINESTYLES[:4]

    def test_n1_is_solid(self):
        """First linestyle is '-' (solid)."""
        result = bw_linestyles(1)
        assert result[0] == "-"

    # -----------------------------------------------------------------------
    # Wrapping / cycling for n > 8
    # -----------------------------------------------------------------------

    def test_n9_wraps_correctly(self):
        """9th element should equal the 1st (cycling from beginning)."""
        result = bw_linestyles(9)
        assert result[8] == _LINESTYLES[0]

    def test_n16_wraps_correctly(self):
        result = bw_linestyles(16)
        assert len(result) == 16
        # Element at index 8 should wrap back to index 0
        assert result[8] == _LINESTYLES[0]
        # Element at index 9 should map to index 1
        assert result[9] == _LINESTYLES[1]

    def test_wrapping_period_equals_8(self):
        """The cycle period is exactly len(_LINESTYLES) == 8."""
        result = bw_linestyles(16)
        for i in range(8):
            assert result[i] == result[i + 8]

    # -----------------------------------------------------------------------
    # Usability with matplotlib
    # -----------------------------------------------------------------------

    def test_linestyles_accepted_by_matplotlib(self):
        """All 8 canonical linestyles must be accepted by matplotlib."""
        styles = bw_linestyles(8)
        fig, ax = plt.subplots()
        x = np.linspace(0, 1, 5)
        for ls in styles:
            ax.plot(x, x, linestyle=ls)  # must not raise
        plt.close(fig)


# ===========================================================================
# bw_hatches
# ===========================================================================


class TestBwHatches:
    """Tests for bw_hatches(n)."""

    def test_returns_list(self):
        assert isinstance(bw_hatches(4), list)

    def test_returns_strings(self):
        for h in bw_hatches(6):
            assert isinstance(h, str)

    def test_length_equals_n_small(self):
        for n in range(1, 9):
            assert len(bw_hatches(n)) == n

    def test_length_equals_n_large(self):
        for n in (9, 12, 16):
            assert len(bw_hatches(n)) == n

    def test_n8_is_full_hatches(self):
        assert bw_hatches(8) == _HATCHES

    def test_n4_is_prefix(self):
        assert bw_hatches(4) == _HATCHES[:4]

    def test_n1_is_empty_string(self):
        """First hatch is '' (no hatch), used for the first band."""
        result = bw_hatches(1)
        assert result[0] == ""

    # -----------------------------------------------------------------------
    # Wrapping / cycling for n > 8
    # -----------------------------------------------------------------------

    def test_n9_wraps_correctly(self):
        result = bw_hatches(9)
        assert result[8] == _HATCHES[0]

    def test_wrapping_period_equals_8(self):
        result = bw_hatches(16)
        for i in range(8):
            assert result[i] == result[i + 8]


# ===========================================================================
# apply_bw_style
# ===========================================================================


class TestApplyBwStyle:
    """Property tests: apply_bw_style sets the correct rcParams."""

    def setup_method(self):
        """Save original rcParams before each test."""
        self._saved = mpl.rcParams.copy()

    def teardown_method(self):
        """Restore original rcParams after each test."""
        mpl.rcParams.update(self._saved)

    def test_dpi_set_to_requested_value(self):
        apply_bw_style(dpi=300)
        assert mpl.rcParams["figure.dpi"] == 300

    def test_savefig_dpi_matches(self):
        apply_bw_style(dpi=150)
        assert mpl.rcParams["savefig.dpi"] == 150

    def test_default_dpi_is_600(self):
        apply_bw_style()
        assert mpl.rcParams["figure.dpi"] == 600

    def test_serif_true_sets_serif_font_family(self):
        apply_bw_style(serif=True)
        assert mpl.rcParams["font.family"] == ["serif"]

    def test_serif_false_sets_sans_serif_font_family(self):
        apply_bw_style(serif=False)
        assert mpl.rcParams["font.family"] == ["sans-serif"]

    def test_figure_facecolor_is_white(self):
        apply_bw_style()
        assert mpl.rcParams["figure.facecolor"] == "white"

    def test_axes_facecolor_is_white(self):
        apply_bw_style()
        assert mpl.rcParams["axes.facecolor"] == "white"

    def test_axes_edgecolor_is_black(self):
        apply_bw_style()
        assert mpl.rcParams["axes.edgecolor"] == "black"

    def test_axes_labelcolor_is_black(self):
        apply_bw_style()
        assert mpl.rcParams["axes.labelcolor"] == "black"

    def test_text_color_is_black(self):
        apply_bw_style()
        assert mpl.rcParams["text.color"] == "black"

    def test_grid_is_enabled(self):
        apply_bw_style()
        assert mpl.rcParams["axes.grid"] is True

    def test_grid_linestyle_is_dotted(self):
        apply_bw_style()
        assert mpl.rcParams["grid.linestyle"] == ":"

    def test_top_spine_is_hidden(self):
        apply_bw_style()
        assert mpl.rcParams["axes.spines.top"] is False

    def test_right_spine_is_hidden(self):
        apply_bw_style()
        assert mpl.rcParams["axes.spines.right"] is False

    def test_xtick_direction_is_in(self):
        apply_bw_style()
        assert mpl.rcParams["xtick.direction"] == "in"

    def test_ytick_direction_is_in(self):
        apply_bw_style()
        assert mpl.rcParams["ytick.direction"] == "in"

    def test_legend_frameon_is_false(self):
        apply_bw_style()
        assert mpl.rcParams["legend.frameon"] is False

    def test_savefig_format_is_pdf(self):
        apply_bw_style()
        assert mpl.rcParams["savefig.format"] == "pdf"

    def test_pdf_fonttype_is_42(self):
        apply_bw_style()
        assert mpl.rcParams["pdf.fonttype"] == 42

    def test_mathtext_fontset_is_cm(self):
        apply_bw_style()
        assert mpl.rcParams["mathtext.fontset"] == "cm"

    def test_prop_cycle_has_8_colors(self):
        """Property cycle should be set to bw_colors(8) — 8 grayscale shades."""
        apply_bw_style()
        cycle = mpl.rcParams["axes.prop_cycle"]
        colors = [p["color"] for p in cycle]
        assert len(colors) == 8

    def test_prop_cycle_colors_match_bw_colors(self):
        """Property cycle colors must match bw_colors(8) exactly."""
        apply_bw_style()
        cycle = mpl.rcParams["axes.prop_cycle"]
        cycle_colors = [p["color"] for p in cycle]
        expected = bw_colors(8)
        assert cycle_colors == expected

    def test_lines_linewidth_is_1_3(self):
        apply_bw_style()
        assert mpl.rcParams["lines.linewidth"] == pytest.approx(1.3)


# ===========================================================================
# bw_context
# ===========================================================================


class TestBwContext:
    """Tests for the bw_context context manager."""

    def setup_method(self):
        self._saved = mpl.rcParams.copy()

    def teardown_method(self):
        mpl.rcParams.update(self._saved)

    def test_applies_style_inside_block(self):
        """Inside the with-block the bw DPI is set."""
        mpl.rcParams.update({"figure.dpi": 100})
        with bw_context(dpi=300):
            assert mpl.rcParams["figure.dpi"] == 300

    def test_restores_dpi_after_block(self):
        """After the with-block the original DPI is restored."""
        mpl.rcParams.update({"figure.dpi": 72})
        with bw_context(dpi=600):
            pass
        assert mpl.rcParams["figure.dpi"] == 72

    def test_restores_font_family_after_block(self):
        """Font family is restored after the block."""
        mpl.rcParams.update({"font.family": "monospace"})
        with bw_context(serif=True):
            assert mpl.rcParams["font.family"] == ["serif"]
        assert mpl.rcParams["font.family"] == ["monospace"]

    def test_restores_on_exception(self):
        """rcParams are restored even when an exception is raised in the block."""
        mpl.rcParams.update({"figure.dpi": 42})
        try:
            with bw_context(dpi=600):
                raise RuntimeError("intentional test error")
        except RuntimeError:
            pass
        assert mpl.rcParams["figure.dpi"] == 42

    def test_context_yields_none(self):
        """bw_context yields None (no value is produced by the generator)."""
        with bw_context() as val:
            assert val is None

    def test_serif_false_inside_context(self):
        mpl.rcParams.update({"font.family": "serif"})
        with bw_context(serif=False):
            assert mpl.rcParams["font.family"] == ["sans-serif"]
        # restored after
        assert mpl.rcParams["font.family"] == ["serif"]

    def test_nested_contexts_restore_correctly(self):
        """Nested contexts should each restore their respective saved state."""
        mpl.rcParams.update({"figure.dpi": 50})
        with bw_context(dpi=200):
            assert mpl.rcParams["figure.dpi"] == 200
            with bw_context(dpi=400):
                assert mpl.rcParams["figure.dpi"] == 400
            # Inner context restores to 200 (what was set before the inner with)
            assert mpl.rcParams["figure.dpi"] == 200
        # Outer context restores to 50
        assert mpl.rcParams["figure.dpi"] == 50

    def test_dpi_parameter_is_forwarded(self):
        """dpi= kwarg is passed through to apply_bw_style."""
        with bw_context(dpi=150):
            assert mpl.rcParams["figure.dpi"] == 150
            assert mpl.rcParams["savefig.dpi"] == 150


# ===========================================================================
# save_pub
# ===========================================================================


class TestSavePub:
    """Tests for save_pub(fig, path_no_ext, dpi)."""

    def _make_fig(self) -> plt.Figure:
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [0, 1, 0])
        return fig

    # -----------------------------------------------------------------------
    # Return value contract
    # -----------------------------------------------------------------------

    def test_returns_list_of_two(self):
        fig = self._make_fig()
        with tempfile.TemporaryDirectory() as d:
            paths = save_pub(fig, os.path.join(d, "fig"), dpi=72)
        plt.close(fig)
        assert isinstance(paths, list)
        assert len(paths) == 2

    def test_first_path_is_pdf(self):
        fig = self._make_fig()
        with tempfile.TemporaryDirectory() as d:
            paths = save_pub(fig, os.path.join(d, "fig"), dpi=72)
        plt.close(fig)
        assert paths[0].endswith(".pdf")

    def test_second_path_is_png(self):
        fig = self._make_fig()
        with tempfile.TemporaryDirectory() as d:
            paths = save_pub(fig, os.path.join(d, "fig"), dpi=72)
        plt.close(fig)
        assert paths[1].endswith(".png")

    # -----------------------------------------------------------------------
    # Files actually created on disk
    # -----------------------------------------------------------------------

    def test_pdf_file_exists(self):
        fig = self._make_fig()
        with tempfile.TemporaryDirectory() as d:
            paths = save_pub(fig, os.path.join(d, "fig"), dpi=72)
            assert os.path.exists(paths[0])
        plt.close(fig)

    def test_png_file_exists(self):
        fig = self._make_fig()
        with tempfile.TemporaryDirectory() as d:
            paths = save_pub(fig, os.path.join(d, "fig"), dpi=72)
            assert os.path.exists(paths[1])
        plt.close(fig)

    def test_files_are_non_empty(self):
        """Both output files must have non-zero size."""
        fig = self._make_fig()
        with tempfile.TemporaryDirectory() as d:
            paths = save_pub(fig, os.path.join(d, "fig"), dpi=72)
            for p in paths:
                assert os.path.getsize(p) > 0
        plt.close(fig)

    # -----------------------------------------------------------------------
    # Sub-directory creation
    # -----------------------------------------------------------------------

    def test_creates_nested_subdirectory(self):
        """save_pub must create the parent directory if it does not exist."""
        fig = self._make_fig()
        with tempfile.TemporaryDirectory() as d:
            nested = os.path.join(d, "level1", "level2", "fig")
            paths = save_pub(fig, nested, dpi=72)
            for p in paths:
                assert os.path.exists(p)
        plt.close(fig)

    def test_path_stem_is_prefix(self):
        """The returned paths must start with the supplied path_no_ext."""
        fig = self._make_fig()
        stem = None
        with tempfile.TemporaryDirectory() as d:
            stem = os.path.join(d, "myfigure")
            paths = save_pub(fig, stem, dpi=72)
        plt.close(fig)
        for p in paths:
            assert p.startswith(stem)

    def test_returned_paths_match_disk(self):
        """The returned paths are the same files that end up on disk."""
        fig = self._make_fig()
        with tempfile.TemporaryDirectory() as d:
            paths = save_pub(fig, os.path.join(d, "fig"), dpi=72)
            on_disk = {os.path.join(d, "fig.pdf"), os.path.join(d, "fig.png")}
            returned = set(paths)
        plt.close(fig)
        assert returned == on_disk

    # -----------------------------------------------------------------------
    # Default dpi
    # -----------------------------------------------------------------------

    def test_default_dpi_creates_files(self):
        """save_pub(fig, path) with default dpi=600 must still create files."""
        fig = self._make_fig()
        with tempfile.TemporaryDirectory() as d:
            paths = save_pub(fig, os.path.join(d, "fig"))
            for p in paths:
                assert os.path.exists(p)
        plt.close(fig)

    def test_path_in_current_dir(self):
        """path_no_ext with no directory component (just a stem) should work.

        The module uses ``os.path.dirname(path_no_ext) or '.'`` to handle
        this case — verify it does not raise.
        """
        fig = self._make_fig()
        with tempfile.TemporaryDirectory() as d:
            original_cwd = os.getcwd()
            try:
                os.chdir(d)
                paths = save_pub(fig, "myfig", dpi=72)
                for p in paths:
                    assert os.path.exists(p)
            finally:
                os.chdir(original_cwd)
        plt.close(fig)
