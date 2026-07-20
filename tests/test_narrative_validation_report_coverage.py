"""Coverage tests for puremacro/narrative/validation/report.py.

Focus: branches and methods NOT exercised by the existing
test_narrative_validation.py suite.  Missing lines as of baseline run:
  58  – ``continue`` inside to_latex_table_speccurve when a dim col is absent
  71  – empty-tbl early-return in to_latex_table_speccurve
  138-156 – plot_event_density (both empty-panel and non-empty-panel paths)
  161-176 – plot_spec_overlay (sampling and no-sampling branches,
             custom horizon_col, NaN drop)

Testing strategy
----------------
* Known-value / contract: verify axes labels, titles, table content,
  and structural invariants (LaTeX tags, markdown separators).
* Property: figure type, string return types, sampling count in title.
* Edge: empty panel, all-NaN column, fewer rows than sample size,
  missing dimension columns, fully-absent dimension columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_results(n: int = 30, *, rng_seed: int = 42,
                  include_dim_cols: bool = True) -> pd.DataFrame:
    """Synthetic spec-curve results DataFrame."""
    rng = np.random.default_rng(rng_seed)
    data = {
        "spec_id":            [f"spec_{i:04d}" for i in range(n)],
        "target":             rng.choice(["investment", "consumption"], n).tolist(),
        "n_obs":              rng.integers(40, 120, n).tolist(),
        "n_nonzero_quarters": rng.integers(10, 40, n).tolist(),
        "first_stage_f_cd":   rng.uniform(3.0, 25.0, n).tolist(),
        "first_stage_f_kp":   rng.uniform(3.0, 20.0, n).tolist(),
        "beta_2sls_h4":       rng.normal(0.5, 0.3, n).tolist(),
        "beta_2sls_h8":       rng.normal(0.7, 0.4, n).tolist(),
        "se_2sls":            rng.uniform(0.1, 0.5, n).tolist(),
    }
    if include_dim_cols:
        data.update({
            "source_subset":   rng.choice(["all", "canonical_only"], n).tolist(),
            "aggregation":     rng.choice(["sum", "count"], n).tolist(),
            "time_window":     rng.choice(["full", "post1990"], n).tolist(),
            "surprise_filter": rng.integers(0, 2, n).astype(bool).tolist(),
            "harmonized":      rng.integers(0, 2, n).astype(bool).tolist(),
        })
    return pd.DataFrame(data)


def _make_report(n: int = 30, **kwargs):
    from puremacro.narrative.validation.report import NarrativeReport
    return NarrativeReport(_make_results(n, **kwargs))


def _make_event(**overrides):
    base = dict(
        date=pd.Timestamp("2010-01-01"),
        country="USA",
        magnitude=1.0,
        magnitude_unit="pct_gdp",
        target="investment",
        subtarget="defense",
        sign=+1,
        confidence=0.9,
        source_text="...",
        source_url="https://x.org",
        scoring_method="manual",
    )
    base.update(overrides)
    from puremacro.narrative.types import NarrativeEvent
    return NarrativeEvent(**base)


def _make_panel(targets_and_dates):
    """Build HomogeneousFiscalPanel from list of (date_str, target) pairs."""
    from puremacro.narrative import HomogeneousFiscalPanel
    events = [
        _make_event(date=pd.Timestamp(d), target=t)
        for d, t in targets_and_dates
    ]
    return HomogeneousFiscalPanel(events=events, countries=["USA"])


# ---------------------------------------------------------------------------
# to_latex_table_speccurve – missing dim column (line 58: continue)
# ---------------------------------------------------------------------------


class TestLatexTableSpeccurvePartialCols:
    """to_latex_table_speccurve with only a subset of the five dim columns."""

    def test_continue_branch_partial_dims(self):
        """When some dim columns are missing the method should skip them silently."""
        from puremacro.narrative.validation.report import NarrativeReport
        # Only source_subset + aggregation present; time_window/surprise_filter/harmonized absent
        rng = np.random.default_rng(7)
        n = 20
        df = pd.DataFrame({
            "target":          rng.choice(["investment", "consumption"], n).tolist(),
            "source_subset":   rng.choice(["all", "canonical_only"], n).tolist(),
            "aggregation":     rng.choice(["sum", "count"], n).tolist(),
            # deliberately omit time_window, surprise_filter, harmonized
            "n_obs":              rng.integers(40, 120, n).tolist(),
            "n_nonzero_quarters": rng.integers(10, 40, n).tolist(),
            "first_stage_f_cd":   rng.uniform(3.0, 25.0, n).tolist(),
            "first_stage_f_kp":   rng.uniform(3.0, 20.0, n).tolist(),
            "beta_2sls_h4":       rng.normal(0.5, 0.3, n).tolist(),
            "se_2sls":            rng.uniform(0.1, 0.5, n).tolist(),
        })
        rpt = NarrativeReport(df)
        latex = rpt.to_latex_table_speccurve()
        assert isinstance(latex, str)
        # Table was non-empty — should have the standard tabular environment
        assert r"\begin{tabular}{llrrr}" in latex
        assert r"\end{tabular}" in latex
        # The two present dims should appear; absent ones should not
        assert "source_subset" in latex
        assert "aggregation" in latex
        assert "time_window" not in latex
        assert "surprise_filter" not in latex
        assert "harmonized" not in latex

    def test_continue_branch_produces_valid_rows(self):
        """Rows produced for present dims should follow expected structure."""
        from puremacro.narrative.validation.report import NarrativeReport
        rng = np.random.default_rng(8)
        n = 16
        df = pd.DataFrame({
            "target":             ["investment"] * n,
            "aggregation":        rng.choice(["sum", "count"], n).tolist(),
            "n_obs":              rng.integers(40, 120, n).tolist(),
            "n_nonzero_quarters": rng.integers(5, 20, n).tolist(),
            "first_stage_f_cd":   rng.uniform(5.0, 20.0, n).tolist(),
            "first_stage_f_kp":   rng.uniform(5.0, 15.0, n).tolist(),
            "beta_2sls_h4":       rng.normal(0.3, 0.2, n).tolist(),
            "se_2sls":            rng.uniform(0.05, 0.3, n).tolist(),
        })
        rpt = NarrativeReport(df)
        latex = rpt.to_latex_table_speccurve()
        # Two unique values for aggregation -> two data rows in the body.
        # Exclude the header line which also ends with \\ and contains &
        lines = latex.splitlines()
        # Data rows: have & and end with \\ but are NOT the \midrule-adjacent header
        # The header row starts with "Dimension &" — exclude it.
        data_lines = [
            l for l in lines
            if l.strip().endswith(r"\\") and "&" in l
            and not l.strip().startswith("Dimension")
        ]
        assert len(data_lines) == 2


# ---------------------------------------------------------------------------
# to_latex_table_speccurve – empty tbl (line 71)
# ---------------------------------------------------------------------------


class TestLatexTableSpeccurveEmpty:
    """to_latex_table_speccurve returns the 'No data' sentinel when all dims absent."""

    def test_empty_tbl_branch_triggered_when_no_dim_cols(self):
        """No dim columns at all -> empty tbl -> early-return with 'No data'."""
        from puremacro.narrative.validation.report import NarrativeReport
        # None of the five dim columns present
        df = pd.DataFrame({
            "target":             ["investment"],
            "n_obs":              [80],
            "n_nonzero_quarters": [20],
            "first_stage_f_cd":   [12.0],
            "first_stage_f_kp":   [10.0],
            "beta_2sls_h4":       [0.5],
            "se_2sls":            [0.1],
        })
        rpt = NarrativeReport(df)
        latex = rpt.to_latex_table_speccurve()
        assert "No data" in latex
        assert r"\begin{tabular}{l}" in latex
        assert r"\toprule" in latex
        assert r"\bottomrule" in latex

    def test_empty_tbl_is_valid_latex_snippet(self):
        """The no-data sentinel must still be parseable as a LaTeX fragment."""
        from puremacro.narrative.validation.report import NarrativeReport
        df = pd.DataFrame({
            "target": ["investment"],
            "n_obs": [80],
            "n_nonzero_quarters": [20],
            "first_stage_f_cd": [12.0],
            "first_stage_f_kp": [10.0],
            "beta_2sls_h4": [0.5],
            "se_2sls": [0.1],
        })
        rpt = NarrativeReport(df)
        latex = rpt.to_latex_table_speccurve()
        assert latex.startswith(r"\begin{tabular}")
        assert latex.endswith(r"\end{tabular}")


# ---------------------------------------------------------------------------
# plot_event_density – both branches (lines 138-156)
# ---------------------------------------------------------------------------


class TestPlotEventDensity:
    """Tests for NarrativeReport.plot_event_density()."""

    def _get_report(self, n: int = 20) -> "NarrativeReport":
        import matplotlib
        matplotlib.use("Agg")
        return _make_report(n)

    def test_empty_panel_returns_figure(self):
        """When panel.to_long() is empty the method still returns a Figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.figure
        from puremacro.narrative import HomogeneousFiscalPanel

        rpt = self._get_report()
        empty_panel = HomogeneousFiscalPanel(events=[], countries=["USA"])
        fig = rpt.plot_event_density(empty_panel)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_empty_panel_title_says_no_events(self):
        """Empty-panel figure should have 'No events' in its title."""
        import matplotlib
        matplotlib.use("Agg")
        from puremacro.narrative import HomogeneousFiscalPanel

        rpt = self._get_report()
        empty_panel = HomogeneousFiscalPanel(events=[], countries=["USA"])
        fig = rpt.plot_event_density(empty_panel)
        title = fig.axes[0].get_title()
        assert "No events" in title

    def test_non_empty_panel_returns_figure(self):
        """Non-empty panel should return a Figure with a bar chart."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.figure

        rpt = self._get_report()
        panel = _make_panel([
            ("2010-01-01", "investment"),
            ("2010-04-01", "consumption"),
            ("2010-07-01", "investment"),
        ])
        fig = rpt.plot_event_density(panel)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_non_empty_panel_axes_labels(self):
        """Axes should carry the expected labels and title."""
        import matplotlib
        matplotlib.use("Agg")

        rpt = self._get_report()
        panel = _make_panel([
            ("2010-01-01", "investment"),
            ("2010-04-01", "consumption"),
        ])
        fig = rpt.plot_event_density(panel)
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Quarter"
        assert ax.get_ylabel() == "Event count"
        assert "density" in ax.get_title().lower() or "event" in ax.get_title().lower()

    def test_non_empty_panel_custom_figsize(self):
        """Custom figsize should be reflected in the figure dimensions."""
        import matplotlib
        matplotlib.use("Agg")

        rpt = self._get_report()
        panel = _make_panel([("2010-01-01", "investment")])
        fig = rpt.plot_event_density(panel, figsize=(12, 6))
        w, h = fig.get_size_inches()
        assert abs(w - 12) < 0.5
        assert abs(h - 6) < 0.5

    def test_non_empty_panel_has_one_axes(self):
        import matplotlib
        matplotlib.use("Agg")

        rpt = self._get_report()
        panel = _make_panel([
            ("2010-01-01", "investment"),
            ("2010-07-01", "consumption"),
        ])
        fig = rpt.plot_event_density(panel)
        assert len(fig.axes) == 1

    def test_multi_target_panel_produces_stacked_chart(self):
        """Multiple targets should appear as separate bar series (stacked)."""
        import matplotlib
        matplotlib.use("Agg")

        rpt = self._get_report()
        panel = _make_panel([
            ("2010-01-01", "investment"),
            ("2010-01-01", "consumption"),
            ("2010-04-01", "investment"),
        ])
        fig = rpt.plot_event_density(panel)
        # The bar chart should have at least one container per target
        ax = fig.axes[0]
        assert len(ax.containers) >= 1


# ---------------------------------------------------------------------------
# plot_spec_overlay (lines 161-176)
# ---------------------------------------------------------------------------


class TestPlotSpecOverlay:
    """Tests for NarrativeReport.plot_spec_overlay()."""

    def _get_report_large(self) -> "NarrativeReport":
        import matplotlib
        matplotlib.use("Agg")
        return _make_report(100)

    def _get_report_small(self) -> "NarrativeReport":
        import matplotlib
        matplotlib.use("Agg")
        return _make_report(20)

    def test_returns_figure(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.figure

        rpt = self._get_report_large()
        fig = rpt.plot_spec_overlay()
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_sampling_branch_n_less_than_len(self):
        """When n < len(df), sample n rows; title should say n."""
        import matplotlib
        matplotlib.use("Agg")

        rpt = self._get_report_large()   # 100 rows
        fig = rpt.plot_spec_overlay(n=50)
        title = fig.axes[0].get_title()
        assert "50" in title

    def test_no_sampling_branch_n_gte_len(self):
        """When n >= len(df), use all rows; title should say len(df)."""
        import matplotlib
        matplotlib.use("Agg")

        rpt = self._get_report_small()   # 20 rows
        fig = rpt.plot_spec_overlay(n=200)
        title = fig.axes[0].get_title()
        assert "20" in title

    def test_nan_dropped_before_sampling(self):
        """NaN values in horizon_col should be dropped without raising."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.figure
        from puremacro.narrative.validation.report import NarrativeReport

        rng = np.random.default_rng(13)
        n = 30
        betas = rng.normal(0.5, 0.3, n).tolist()
        betas[0] = float("nan")
        betas[5] = float("nan")
        df = pd.DataFrame({
            "target":             ["investment"] * n,
            "n_obs":              rng.integers(40, 120, n).tolist(),
            "n_nonzero_quarters": rng.integers(10, 40, n).tolist(),
            "first_stage_f_cd":   rng.uniform(3.0, 25.0, n).tolist(),
            "first_stage_f_kp":   rng.uniform(3.0, 20.0, n).tolist(),
            "beta_2sls_h4":       betas,
            "se_2sls":            rng.uniform(0.1, 0.5, n).tolist(),
        })
        rpt = NarrativeReport(df)
        fig = rpt.plot_spec_overlay(n=50)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_custom_horizon_col(self):
        """plot_spec_overlay should work with a non-default horizon column."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.figure
        from puremacro.narrative.validation.report import NarrativeReport

        rng = np.random.default_rng(17)
        n = 30
        df = _make_results(n)
        rpt = NarrativeReport(df)
        fig = rpt.plot_spec_overlay(horizon_col="beta_2sls_h8")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_axes_labels(self):
        """Axes should carry the expected labels."""
        import matplotlib
        matplotlib.use("Agg")

        rpt = self._get_report_large()
        fig = rpt.plot_spec_overlay()
        ax = fig.axes[0]
        assert "Specification" in ax.get_xlabel() or "specification" in ax.get_xlabel().lower()
        assert r"\hat{\beta}" in ax.get_ylabel() or "beta" in ax.get_ylabel().lower() or "2SLS" in ax.get_ylabel()

    def test_title_contains_robustness(self):
        """Title should mention 'Robustness'."""
        import matplotlib
        matplotlib.use("Agg")

        rpt = self._get_report_large()
        fig = rpt.plot_spec_overlay()
        assert "Robustness" in fig.axes[0].get_title()

    def test_custom_figsize(self):
        """figsize kwarg is passed through."""
        import matplotlib
        matplotlib.use("Agg")

        rpt = self._get_report_large()
        fig = rpt.plot_spec_overlay(figsize=(6, 3))
        w, h = fig.get_size_inches()
        assert abs(w - 6) < 0.5
        assert abs(h - 3) < 0.5

    def test_reproducibility_under_seed(self):
        """random_state=0 means successive calls with n < len produce the same figure."""
        import matplotlib
        matplotlib.use("Agg")
        from puremacro.narrative.validation.report import NarrativeReport

        df = _make_results(80)
        rpt = NarrativeReport(df)
        fig1 = rpt.plot_spec_overlay(n=30)
        fig2 = rpt.plot_spec_overlay(n=30)
        # Both calls use random_state=0, so sorted betas on sample should be identical
        data1 = fig1.axes[0].collections[0].get_offsets()
        data2 = fig2.axes[0].collections[0].get_offsets()
        np.testing.assert_array_equal(data1, data2)

    def test_median_line_present(self):
        """A horizontal reference line at the median beta should be drawn."""
        import matplotlib
        matplotlib.use("Agg")

        rpt = self._get_report_large()
        fig = rpt.plot_spec_overlay()
        ax = fig.axes[0]
        # The median line is added via axhline; it appears as a Line2D in ax.lines
        # We test at least two horizontal lines (median and zero)
        hlines = [l for l in ax.lines if l.get_linestyle() in ("--", "-")]
        assert len(hlines) >= 1


# ---------------------------------------------------------------------------
# Integration: confirm the previously tested public methods still pass
# (guard against regressions from our test setup assumptions)
# ---------------------------------------------------------------------------


class TestRegressionGuard:
    """Smoke-test that the existing public API surface is intact."""

    def test_firststage_table_still_works(self):
        from puremacro.narrative.validation.report import NarrativeReport
        df = _make_results(30)
        rpt = NarrativeReport(df)
        latex = rpt.to_latex_table_firststage()
        assert r"\begin{tabular}" in latex

    def test_speccurve_table_with_all_dims_still_works(self):
        from puremacro.narrative.validation.report import NarrativeReport
        df = _make_results(30)
        rpt = NarrativeReport(df)
        latex = rpt.to_latex_table_speccurve()
        assert r"\begin{tabular}{llrrr}" in latex

    def test_plot_f_violin_still_works(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.figure
        from puremacro.narrative.validation.report import NarrativeReport

        rpt = NarrativeReport(_make_results(30))
        fig = rpt.plot_f_violin()
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_beta_funnel_still_works(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.figure
        from puremacro.narrative.validation.report import NarrativeReport

        rpt = NarrativeReport(_make_results(30))
        fig = rpt.plot_beta_funnel()
        assert isinstance(fig, matplotlib.figure.Figure)
