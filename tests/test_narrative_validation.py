"""Tests for narrative validation module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.narrative.types import NarrativeEvent
from puremacro.narrative import HomogeneousFiscalPanel


def _make_event(**overrides) -> NarrativeEvent:
    base = dict(
        date=pd.Timestamp("2010-01-01"),
        country="USA",
        magnitude=1.0, magnitude_unit="pct_gdp",
        target="investment", subtarget="defense",
        sign=+1, confidence=0.9,
        source_text="...", source_url="https://x.org",
        scoring_method="manual",
    )
    base.update(overrides)
    return NarrativeEvent(**base)


# ---------------------------------------------------------------------------
# GapFiller
# ---------------------------------------------------------------------------


def _panel_with_gap():
    """USA: events in 2000Q1 and 2002Q1 — six-quarter gap in between."""
    events = [
        _make_event(date=pd.Timestamp("2000-01-01")),
        _make_event(date=pd.Timestamp("2002-01-01")),
    ]
    return HomogeneousFiscalPanel(events=events, countries=["USA"])


def test_gap_filler_finds_long_gaps():
    from puremacro.narrative.validation.gap_filler import find_gaps
    panel = _panel_with_gap()
    gaps = find_gaps(panel, max_gap_q=4)
    assert len(gaps) > 0
    assert "country" in gaps.columns
    assert "gap_start" in gaps.columns
    assert "n_quarters" in gaps.columns
    usa_gaps = gaps[gaps["country"] == "USA"]
    assert len(usa_gaps) == 1
    assert usa_gaps.iloc[0]["n_quarters"] >= 6


def test_gap_filler_returns_empty_for_dense_panel():
    from puremacro.narrative.validation.gap_filler import find_gaps
    # Event every quarter from 2000Q1 to 2001Q4.
    events = [
        _make_event(date=pd.Timestamp(f"200{y}-{m:02d}-01"))
        for y in range(2) for m in [1, 4, 7, 10]
    ]
    panel = HomogeneousFiscalPanel(events=events, countries=["USA"])
    gaps = find_gaps(panel, max_gap_q=4)
    assert gaps.empty


def test_gap_filler_target_filter():
    from puremacro.narrative.validation.gap_filler import find_gaps
    events = [
        _make_event(date=pd.Timestamp("2000-01-01"), target="investment"),
        _make_event(date=pd.Timestamp("2001-01-01"), target="consumption"),
        _make_event(date=pd.Timestamp("2002-01-01"), target="investment"),
    ]
    panel = HomogeneousFiscalPanel(events=events, countries=["USA"])
    gaps_inv = find_gaps(panel, max_gap_q=4, target="investment")
    # Investment events at 2000Q1 and 2002Q1 (consumption at 2001Q1 filtered out).
    # The gap between them spans 2000Q2-2002Q0, so 4 quarters.
    usa_gaps = gaps_inv[gaps_inv["country"] == "USA"]
    assert len(usa_gaps) > 0
    assert usa_gaps.iloc[0]["n_quarters"] >= 4


# ---------------------------------------------------------------------------
# NarrativeSpecificationCurve
# ---------------------------------------------------------------------------


def _make_macro_df(n: int = 80) -> pd.DataFrame:
    """Synthetic macro panel: single country, n quarters."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2000-01-01", periods=n, freq="QS")
    return pd.DataFrame({
        "log_gdp":        np.cumsum(rng.normal(0, 0.01, n)),
        "log_gov_invest": np.cumsum(rng.normal(0, 0.02, n)),
        "log_gov_consume": np.cumsum(rng.normal(0, 0.02, n)),
    }, index=idx)


def _make_single_country_panel(n: int = 80) -> HomogeneousFiscalPanel:
    rng = np.random.default_rng(1)
    idx = pd.date_range("2000-01-01", periods=n, freq="QS")
    # Sparse instrument: nonzero at every 4th quarter.
    events = []
    for i, d in enumerate(idx):
        if i % 4 == 0:
            events.append(_make_event(
                date=d, country="USA",
                magnitude=rng.uniform(0.1, 2.0),
                sign=int(rng.choice([-1, 1])),
                target="investment",
            ))
    return HomogeneousFiscalPanel(events=events, countries=["USA"])


def test_spec_curve_runs_and_returns_dataframe():
    from puremacro.narrative.validation.spec_curve import NarrativeSpecificationCurve
    panel = _make_single_country_panel()
    df = _make_macro_df()
    curve = NarrativeSpecificationCurve(
        panel=panel,
        df=df,
        y="log_gdp",
        x_invest="log_gov_invest",
        x_consume="log_gov_consume",
        horizon=4,
        n_boot=0,        # skip bootstrap for speed
        max_specs=4,     # tiny subsample
    )
    results = curve.run()
    assert isinstance(results, pd.DataFrame)
    assert len(results) >= 1
    required = {"spec_id", "target", "n_obs", "first_stage_f_cd", "beta_2sls_h4"}
    assert required.issubset(results.columns)


def test_spec_curve_single_spec_matches_weak_iv_summary():
    """With max_specs=1 the curve's CD-F must match weak_iv_summary directly."""
    from puremacro.narrative.validation.spec_curve import NarrativeSpecificationCurve
    from puremacro.narrative.validate import weak_iv_summary
    panel = _make_single_country_panel(n=60)
    df = _make_macro_df(n=60)

    curve = NarrativeSpecificationCurve(
        panel=panel, df=df,
        y="log_gdp", x_invest="log_gov_invest", x_consume="log_gov_consume",
        horizon=4, n_boot=0, max_specs=1,
    )
    results = curve.run()
    if results.empty:
        pytest.skip("Randomly selected spec produced no usable rows — OK with max_specs=1")
    # Build the same instrument manually and call weak_iv_summary.
    z = panel.to_quarterly_panel(target="investment").reindex(df.index).fillna(0.0)
    if z.empty or z.shape[1] == 0:
        pytest.skip("No investment events — synthetic panel has no usable z")
    z_series = z.iloc[:, 0]
    diag = weak_iv_summary(
        z_series.values,
        df["log_gov_invest"].values,
        df["log_gdp"].values,
    )
    row = results.iloc[0]
    assert abs(row["first_stage_f_cd"] - diag["first_stage_f_cd"]) < 0.5


def test_spec_curve_sum_sign_differs_from_sum():
    """sum_sign (sign-only) must produce a different series from sum (magnitude-weighted)."""
    from puremacro.narrative.validation.spec_curve import NarrativeSpecificationCurve
    # Build panel with mixed-magnitude events so sum != sum_sign.
    rng = np.random.default_rng(99)
    idx = pd.date_range("2000-01-01", periods=40, freq="QS")
    events = []
    for i, d in enumerate(idx):
        if i % 3 == 0:
            events.append(_make_event(
                date=d, country="USA",
                magnitude=rng.uniform(1.0, 10.0),  # varying magnitudes
                sign=1, target="investment",
            ))
    panel = HomogeneousFiscalPanel(events=events, countries=["USA"])
    df = _make_macro_df(n=40)

    # Build two 1-spec curves that differ only in aggregation.
    def _single_spec_z(aggregation):
        curve = NarrativeSpecificationCurve(
            panel=panel, df=df,
            y="log_gdp", x_invest="log_gov_invest", x_consume="log_gov_consume",
            horizon=4, n_boot=0, max_specs=None, seed=0,
        )
        # Directly call _instrument_for_spec with a fixed spec.
        spec = {
            "source_subset": "all",
            "aggregation": aggregation,
            "confidence_thr": 0.0,
            "target": "investment",
            "time_window": "full",
            "surprise_filter": False,
            "harmonized": False,
        }
        return curve._instrument_for_spec(spec)

    z_sum = _single_spec_z("sum")
    z_sign = _single_spec_z("sum_sign")

    # Both should be non-None for this dense panel.
    assert z_sum is not None and z_sign is not None
    # Since magnitudes vary (1–10), signed-sum ≠ sign-sum in general.
    assert not np.allclose(z_sum.dropna().values, z_sign.dropna().reindex(z_sum.dropna().index).fillna(0).values)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _make_results_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 30
    return pd.DataFrame({
        "spec_id":            [f"spec_{i:04d}" for i in range(n)],
        "source_subset":      rng.choice(["canonical_only", "all"], n),
        "aggregation":        rng.choice(["sum", "count"], n),
        "confidence_thr":     rng.choice([0.5, 0.7], n),
        "target":             rng.choice(["investment", "consumption"], n),
        "time_window":        rng.choice(["full", "post1990"], n),
        "surprise_filter":    rng.integers(0, 2, n).astype(bool),
        "harmonized":         rng.integers(0, 2, n).astype(bool),
        "n_obs":              rng.integers(40, 120, n),
        "n_nonzero_quarters": rng.integers(10, 40, n),
        "first_stage_f_cd":   rng.uniform(3.0, 25.0, n),
        "first_stage_f_kp":   rng.uniform(3.0, 20.0, n),
        "beta_2sls_h4":       rng.normal(0.5, 0.3, n),
        "beta_2sls_h8":       rng.normal(0.7, 0.4, n),
        "AR_p_value_at_2sls": rng.uniform(0.01, 0.4, n),
        "se_2sls":            rng.uniform(0.1, 0.5, n),
    })


def test_report_latex_firststage_contains_country_rows():
    from puremacro.narrative.validation.report import NarrativeReport
    results = _make_results_df()
    rpt = NarrativeReport(results)
    latex = rpt.to_latex_table_firststage()
    assert isinstance(latex, str)
    assert "\\begin{tabular}" in latex
    assert "F_{CD}" in latex or "CD" in latex


def test_report_latex_speccurve_contains_dimension_rows():
    from puremacro.narrative.validation.report import NarrativeReport
    results = _make_results_df()
    rpt = NarrativeReport(results)
    latex = rpt.to_latex_table_speccurve()
    assert "source_subset" in latex or "Source" in latex


def test_report_plot_f_violin_returns_figure():
    import matplotlib
    matplotlib.use("Agg")
    from puremacro.narrative.validation.report import NarrativeReport
    results = _make_results_df()
    rpt = NarrativeReport(results)
    fig = rpt.plot_f_violin()
    import matplotlib.figure
    assert isinstance(fig, matplotlib.figure.Figure)


def test_report_plot_beta_funnel_returns_figure():
    import matplotlib
    matplotlib.use("Agg")
    from puremacro.narrative.validation.report import NarrativeReport
    results = _make_results_df()
    rpt = NarrativeReport(results)
    fig = rpt.plot_beta_funnel()
    import matplotlib.figure
    assert isinstance(fig, matplotlib.figure.Figure)


def test_panel_specification_curve_method_exists():
    from puremacro.narrative import HomogeneousFiscalPanel
    panel = _make_single_country_panel()
    df = _make_macro_df()
    assert hasattr(panel, "specification_curve")
    curve = panel.specification_curve(
        df,
        y="log_gdp",
        x_invest="log_gov_invest",
        x_consume="log_gov_consume",
        horizon=4,
        n_boot=0,
        max_specs=2,
    )
    from puremacro.narrative.validation.spec_curve import NarrativeSpecificationCurve
    assert isinstance(curve, NarrativeSpecificationCurve)


def test_validation_namespace_imports():
    from puremacro.narrative.validation import gap_filler
    from puremacro.narrative.validation.spec_curve import NarrativeSpecificationCurve
    from puremacro.narrative.validation.report import NarrativeReport
    assert callable(gap_filler.find_gaps)
    assert NarrativeSpecificationCurve is not None
    assert NarrativeReport is not None


def test_spec_curve_grid_includes_iv_kind_dimension():
    """The grid must include both iv_kind values when not locked."""
    from puremacro.narrative.validation.spec_curve import (
        NarrativeSpecificationCurve,
    )
    sc_dims = NarrativeSpecificationCurve.dimensions()
    assert "iv_kind" in sc_dims
    assert set(sc_dims["iv_kind"]) == {"announcement", "implementation"}
    assert "within_year_rule" in sc_dims
    assert set(sc_dims["within_year_rule"]) >= {"uniform", "front_loaded", "fiscal_year"}


def test_spec_curve_surprise_filter_flag_changes_output():
    """surprise_filter=True must produce a different instrument from False."""
    from puremacro.narrative.validation.spec_curve import NarrativeSpecificationCurve
    import pandas as pd
    # Build panel with budget-month events (should be decayed by SurpriseFilter).
    # GBR budget month = 10; we give events in October with expenditure subtarget.
    from puremacro.narrative.types import NarrativeEvent
    from puremacro.narrative import HomogeneousFiscalPanel
    events = [
        NarrativeEvent(
            date=pd.Timestamp(f"200{y}-10-01"), country="GBR",
            magnitude=2.0, magnitude_unit="pct_gdp",
            target="investment", subtarget="expenditure",
            sign=-1, confidence=0.9,
            source_text="...", source_url="https://x.org",
            scoring_method="manual",
        )
        for y in range(5)
    ]
    panel = HomogeneousFiscalPanel(events=events, countries=["GBR"])
    df = _make_macro_df(n=40)

    curve = NarrativeSpecificationCurve(panel=panel, df=df,
        y="log_gdp", x_invest="log_gov_invest", x_consume="log_gov_consume",
        horizon=4, n_boot=0, max_specs=None)

    # Use confidence_thr=0.3 so that after decay (0.9 * 0.25 = 0.225 < 0.3)
    # the SurpriseFilter hard-drops all October GBR events, yielding None.
    # Without the filter the same events survive the 0.3 threshold fine.
    spec_no_filter = {
        "source_subset": "all", "aggregation": "sum", "confidence_thr": 0.3,
        "target": "investment", "time_window": "full",
        "surprise_filter": False, "harmonized": False,
    }
    spec_with_filter = {**spec_no_filter, "surprise_filter": True}

    z_raw  = curve._instrument_for_spec(spec_no_filter)
    z_filt = curve._instrument_for_spec(spec_with_filter)
    # z_raw should have events (confidence 0.9 >= 0.3); z_filt should be None
    # because SurpriseFilter decays confidence to 0.225 which falls below the
    # confidence_floor=0.3 passed to it, so all events are hard-dropped.
    assert z_raw is not None, "Raw instrument should have events at confidence_thr=0.3"
    assert z_filt is None, (
        "surprise_filter=True should have dropped all October GBR events "
        "(decayed confidence 0.225 < confidence_floor 0.3)"
    )


def test_spec_curve_dimensions_returns_isolated_copy():
    """Caller mutation of the returned dict must not corrupt the class-level constants."""
    from puremacro.narrative.validation.spec_curve import (
        NarrativeSpecificationCurve,
    )
    d1 = NarrativeSpecificationCurve.dimensions()
    d1["iv_kind"].append("fake")
    d2 = NarrativeSpecificationCurve.dimensions()
    assert "fake" not in d2["iv_kind"]
    assert set(d2["iv_kind"]) == {"announcement", "implementation"}
