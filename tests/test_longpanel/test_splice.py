"""The splice core: what it preserves, and what it refuses to do.

A ratio splice is only legitimate when the two vintages agree about
growth and differ only in level. These tests are built around that
distinction, because it is the one the whole feature rests on:

- a constant ratio (a rebasing, a units change, an annualisation
  factor) must pass through silently and exactly;
- a drifting ratio must be flagged, because the spliced level then
  depends on an arbitrary anchor;
- and three situations must be refused outright rather than papered
  over with a number.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from puremacro.fetch.longpanel._splice import (
    MIN_OVERLAP,
    Seam,
    expenditure_residual,
    overlap_ratio,
    ratio_splice,
    splice_frame,
)


def _path(start="1990Q1", end="2000Q4", g=0.01, base=100.0):
    idx = pd.period_range(start, end, freq="Q").to_timestamp()
    return pd.Series(base * np.exp(np.cumsum(np.full(len(idx), g))), index=idx)


TRUTH = _path()
NEW = TRUTH["1995Q1":]
OLD = TRUTH[:"1996Q4"]


# ---------------------------------------------------------------------------
# what a splice must preserve
# ---------------------------------------------------------------------------
def test_a_constant_ratio_is_absorbed_exactly():
    """A rebasing/units/annualisation factor is a constant and must
    vanish into the ratio without disturbing anything."""
    res = ratio_splice([("current", NEW), ("old", OLD * 0.8)])
    assert res.seams[0].ratio == pytest.approx(1.25)
    assert res.seams[0].ratio_drift == pytest.approx(0.0, abs=1e-12)
    assert res.stable


def test_spine_levels_are_published_levels():
    """The newest segment is the spine: its numbers are never rescaled."""
    res = ratio_splice([("current", NEW), ("old", OLD * 0.8)])
    pd.testing.assert_series_equal(
        res.series[NEW.index[0]:], NEW, check_names=False)


def test_the_old_segment_keeps_its_own_growth_rates():
    """Levels are rescaled; growth is what survives, and it must survive
    exactly — that is the entire point of a ratio splice."""
    old = OLD * 0.8
    res = ratio_splice([("current", NEW), ("old", old)])
    back = res.series[:"1994Q4"]
    np.testing.assert_allclose(back.pct_change().dropna().to_numpy(),
                               old[:"1994Q4"].pct_change().dropna().to_numpy())


def test_the_series_is_extended_backwards():
    res = ratio_splice([("current", NEW), ("old", OLD * 0.8)])
    assert res.series.index.min() == OLD.index.min()
    assert len(res.series) > len(NEW)


def test_provenance_labels_every_quarter():
    res = ratio_splice([("current", NEW), ("old", OLD * 0.8)])
    assert set(res.provenance.unique()) == {"current", "old"}
    assert res.provenance.index.equals(res.series.index)
    assert res.provenance.loc[NEW.index[-1]] == "current"
    assert res.provenance.loc[OLD.index[0]] == "old"


def test_a_scale_factor_of_any_size_works():
    """Japan's ESRI files are billions at annual rates against a spine in
    millions per quarter — a factor of 250. Nothing may special-case it."""
    res = ratio_splice([("spine", NEW), ("esri_like", OLD / 250.0)])
    assert res.seams[0].ratio == pytest.approx(250.0)
    assert res.stable


# ---------------------------------------------------------------------------
# drift: the test of whether a splice is legitimate at all
# ---------------------------------------------------------------------------
def _drifting_old(lo=1 / 0.754, hi=1 / 0.837):
    """An old segment whose ratio moves ACROSS THE OVERLAP.

    Drift outside the overlap is invisible to the estimator and must
    not be used to build this fixture — that mistake makes the test
    pass against a stable ratio.
    """
    old = OLD.copy()
    inside = old.index >= NEW.index[0]
    f = np.ones(len(old))
    f[inside] = np.linspace(lo, hi, inside.sum())
    return old * f


def test_a_drifting_ratio_is_flagged_and_warned():
    with pytest.warns(UserWarning, match="disagree about growth"):
        res = ratio_splice([("current", NEW), ("cepal", _drifting_old())])
    assert not res.stable
    assert not res.seams[0].stable
    assert res.seams[0].ratio_drift > 0.02


def test_the_drift_fixture_actually_drifts_inside_the_overlap():
    """Positive control: if this stops being true the test above passes
    vacuously against a stable ratio."""
    _, n, drift, lo, hi = overlap_ratio(_drifting_old(), NEW)
    assert n >= MIN_OVERLAP
    # Measured: drift 3.66%, ratio ranging 0.754-0.837 across the 8
    # overlapping quarters. Comfortably above the 2% threshold, so the
    # test above is exercising the unstable branch and not the stable one.
    assert drift > 0.02
    assert hi - lo > 0.05


def test_a_constant_ratio_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ratio_splice([("current", NEW), ("old", OLD * 0.8)])


def test_drift_threshold_is_honoured():
    old = _drifting_old()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loose = ratio_splice([("c", NEW), ("o", old)], drift_warn=0.99)
        strict = ratio_splice([("c", NEW), ("o", old)], drift_warn=1e-9)
    assert loose.stable
    assert not strict.stable


# ---------------------------------------------------------------------------
# what must be refused
# ---------------------------------------------------------------------------
def test_no_overlap_is_refused():
    with pytest.raises(ValueError, match="do not overlap"):
        ratio_splice([("a", TRUTH["1998Q1":]), ("b", TRUTH[:"1995Q4"])])


def test_too_little_overlap_is_refused():
    with pytest.raises(ValueError, match="overlapping quarter"):
        ratio_splice([("a", TRUTH["1996Q1":]), ("b", TRUTH[:"1996Q2"])])


def test_a_definitional_break_is_refused_by_default():
    """West Germany is not unified Germany. Rescaling one onto the other
    fabricates an economy that did not exist."""
    with pytest.raises(ValueError, match="refusing to splice"):
        ratio_splice(
            [("unified", NEW), ("west", OLD * 0.8)],
            definitional_breaks={"west": "West Germany only."})


def test_a_definitional_break_can_be_overridden_deliberately():
    res = ratio_splice(
        [("unified", NEW), ("west", OLD * 0.8)],
        definitional_breaks={"west": "West Germany only."},
        allow_definitional_break=True)
    assert len(res.series) > len(NEW)


def test_empty_input_is_empty_output_not_an_error():
    res = ratio_splice([])
    assert res.series.empty and res.seams == []


# ---------------------------------------------------------------------------
# overlap_ratio
# ---------------------------------------------------------------------------
def test_overlap_ratio_uses_the_mean_not_a_single_anchor():
    """One revised quarter must not move the whole backcast."""
    old = OLD * 0.8
    spiked = old.copy()
    spiked.iloc[-1] *= 2.0                       # one wild quarter
    r_clean, *_ = overlap_ratio(old, NEW)
    r_spiked, n, drift, _, _ = overlap_ratio(spiked, NEW)
    assert abs(r_spiked - r_clean) < 0.5 * r_clean
    assert drift > 0.02                          # and it shows up as drift


def test_overlap_ratio_on_disjoint_series():
    r, n, drift, lo, hi = overlap_ratio(TRUTH[:"1992Q4"], TRUTH["1998Q1":])
    assert n == 0 and np.isnan(r)


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------
def _frame(series_map):
    return pd.DataFrame(series_map)


def test_columns_are_spliced_independently():
    """Each column keeps its own source's growth. They are deliberately
    not reconciled — see expenditure_residual."""
    new = _frame({"gdp": NEW, "cons_hh": NEW * 0.6})
    old = _frame({"gdp": OLD * 0.8, "cons_hh": OLD * 0.6 * 0.5})
    vals, prov, seams = splice_frame([("cur", new), ("old", old)])
    assert seams["gdp"][0].ratio == pytest.approx(1.25)
    assert seams["cons_hh"][0].ratio == pytest.approx(2.0)
    assert set(prov.columns) == {"gdp", "cons_hh"}


def test_a_column_missing_from_the_old_segment_keeps_the_spine():
    new = _frame({"gdp": NEW, "cons_gov": NEW * 0.2})
    old = _frame({"gdp": OLD * 0.8})
    vals, prov, seams = splice_frame([("cur", new), ("old", old)])
    assert vals["cons_gov"].dropna().index.min() == NEW.index.min()
    assert vals["gdp"].dropna().index.min() == OLD.index.min()


def test_an_unspliceable_column_warns_and_falls_back_to_the_spine():
    new = _frame({"gdp": NEW, "x": NEW})
    old = _frame({"gdp": OLD * 0.8, "x": TRUTH[:"1992Q4"]})   # no overlap on x
    with pytest.warns(UserWarning, match="kept from"):
        vals, prov, seams = splice_frame([("cur", new), ("old", old)])
    assert seams["x"] == []
    # `x` exists only over the spine's span, so it is NaN in the quarters
    # that only `gdp` reached back into — absence, correctly labelled.
    assert set(prov["x"].dropna().unique()) == {"cur"}
    assert vals["x"].dropna().index.min() == NEW.index.min()


def test_expenditure_residual_is_zero_on_a_consistent_frame():
    idx = NEW.index
    f = pd.DataFrame({
        "cons_hh": pd.Series(60.0, idx), "cons_gov": pd.Series(20.0, idx),
        "capform": pd.Series(25.0, idx), "exports": pd.Series(30.0, idx),
        "imports": pd.Series(35.0, idx),
    })
    f["gdp"] = 60 + 20 + 25 + 30 - 35
    np.testing.assert_allclose(expenditure_residual(f).to_numpy(), 0.0,
                               atol=1e-12)


def test_expenditure_residual_reports_a_real_gap():
    idx = NEW.index
    f = pd.DataFrame({
        "gdp": pd.Series(105.0, idx), "cons_hh": pd.Series(60.0, idx),
        "cons_gov": pd.Series(20.0, idx), "capform": pd.Series(25.0, idx),
        "exports": pd.Series(30.0, idx), "imports": pd.Series(35.0, idx),
    })
    np.testing.assert_allclose(expenditure_residual(f).to_numpy(), 5.0)


def test_expenditure_residual_missing_columns_raises():
    with pytest.raises(ValueError, match="missing"):
        expenditure_residual(pd.DataFrame({"gdp": NEW}))


def test_seam_is_frozen():
    s = Seam(date=pd.Timestamp("1995-01-01"), older="a", newer="b",
             overlap_n=8, ratio=1.0, ratio_drift=0.0, ratio_min=1.0,
             ratio_max=1.0, stable=True)
    with pytest.raises(Exception):
        s.ratio = 2.0
