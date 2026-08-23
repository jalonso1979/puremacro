"""Revision econometrics: triangle construction and the Mankiw-Shapiro pair.

The tests that matter here are the ones that would still pass if the
code were subtly wrong, so each is written against a DGP whose answer
is known analytically rather than against a golden number:

- ``test_transform_is_applied_within_vintage`` fails if the growth
  transform ever mixes a numerator from one vintage with a denominator
  from another. That is the single easiest way to get this module
  wrong and it produces revisions nobody published.
- the two DGP tests fail if the news and noise regressions are
  swapped, which is the second easiest way.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.vintages import (
    MankiwShapiroResult,
    first_release,
    latest_release,
    mankiw_shapiro,
    revision_frame,
    revision_test,
    revision_triangle,
)


def _panel(rows):
    return pd.DataFrame(rows, columns=["date", "vintage", "value"])


def _toy_panel(n_dates=24, n_releases=3, seed=0):
    """A realistic archive: every edition republishes the whole history.

    Real statistical offices do not publish one quarter per edition —
    each release restates the full back-series, with the most recent
    quarters still provisional. Building the fixture that way matters:
    a fixture whose editions each carried a single quarter would make
    every column of the revision triangle one cell long, so no
    within-edition transform (a growth rate, a year-on-year change)
    could be computed at all, and the tests would pass vacuously.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-01", periods=n_dates, freq="QS")
    truth = 100 * np.exp(np.cumsum(0.005 + 0.004 * rng.standard_normal(n_dates)))
    # One edition a quarter, each published a quarter after the period
    # it first covers, running a year past the last reference period.
    editions = pd.date_range(dates[0] + pd.DateOffset(months=4),
                             dates[-1] + pd.DateOffset(months=13), freq="QS")
    rows = []
    for v in editions:
        for i, d in enumerate(dates):
            if d + pd.DateOffset(months=4) > v:
                continue                      # not yet published
            # How many editions have passed since this period first
            # appeared: the estimate settles after n_releases of them.
            k = len(pd.date_range(d + pd.DateOffset(months=4), v, freq="QS")) - 1
            err = 0.0 if k >= n_releases - 1 else rng.standard_normal() * 0.3
            rows.append({"date": d, "vintage": v, "value": truth[i] + err})
    return _panel(rows), dates, truth


# ---------------------------------------------------------------------------
# Triangle construction
# ---------------------------------------------------------------------------
def test_triangle_is_dates_by_vintages_with_asof_semantics():
    panel, dates, _ = _toy_panel(n_dates=6, n_releases=2)
    tri = revision_triangle(panel)
    assert tri.index.tolist() == list(dates)
    assert list(tri.columns) == sorted(panel["vintage"].unique())
    # A period is NaN in every vintage that predates its first publication.
    first_v = panel.groupby("date")["vintage"].min()
    for d in dates:
        earlier = [c for c in tri.columns if c < first_v[d]]
        assert tri.loc[d, earlier].isna().all()


def test_carry_forward_is_opt_in_and_changes_what_a_column_means():
    """Two different objects. A column is one edition by default; with
    ``carry_forward`` it is everything known on that date.

    Differencing down a carried column mixes editions — which is how a
    peseta level next to a euro level produced a -594% quarterly
    "growth rate" on real OECD data.
    """
    q1, q2 = pd.Timestamp("2020-01-01"), pd.Timestamp("2020-04-01")
    v1, v2 = pd.Timestamp("2020-05-01"), pd.Timestamp("2020-08-01")
    panel = _panel([
        {"date": q1, "vintage": v1, "value": 10.0},
        # v2 drops Q1 from its own history but publishes Q2.
        {"date": q2, "vintage": v2, "value": 20.0},
    ])
    plain = revision_triangle(panel)
    assert pd.isna(plain.loc[q1, v2]), "an edition must not gain a period"
    carried = revision_triangle(panel, carry_forward=True)
    assert carried.loc[q1, v2] == 10.0
    # And the consequence for a growth rate: nothing vs a spliced number.
    assert pd.isna(revision_triangle(panel, transform="pct_change").loc[q2, v2])
    assert revision_triangle(panel, transform="pct_change",
                             carry_forward=True).loc[q2, v2] == pytest.approx(100.0)


def test_transform_is_applied_within_vintage():
    """The growth rate in column v uses only column v's levels.

    Built so that the within-vintage answer and the across-vintage
    answer differ: the level for 2020Q1 is revised 10.0 -> 12.0, so a
    Q2 growth rate that took its denominator from the wrong vintage
    would come out different.
    """
    q1, q2 = pd.Timestamp("2020-01-01"), pd.Timestamp("2020-04-01")
    v1, v2 = pd.Timestamp("2020-05-01"), pd.Timestamp("2020-08-01")
    panel = _panel([
        {"date": q1, "vintage": v1, "value": 10.0},
        {"date": q1, "vintage": v2, "value": 12.0},   # Q1 revised up
        {"date": q2, "vintage": v2, "value": 24.0},
    ])
    tri = revision_triangle(panel, transform="pct_change")
    # Within v2: 24/12 - 1 = +100%. Across vintages it would be 24/10 - 1 = +140%.
    assert tri.loc[q2, v2] == pytest.approx(100.0)
    assert tri.loc[q2, v2] != pytest.approx(140.0)


def test_unknown_transform_raises():
    panel, _, _ = _toy_panel(n_dates=4)
    with pytest.raises(ValueError, match="unknown transform"):
        revision_triangle(panel, transform="not_a_transform")


@pytest.mark.parametrize(
    "transform", ["level", "diff", "log_diff_pct", "pct_change",
                  "pct_change_ann", "yoy"],
)
def test_every_advertised_transform_runs(transform):
    panel, _, _ = _toy_panel(n_dates=16)
    tri = revision_triangle(panel, transform=transform)
    assert not tri.empty
    assert tri.notna().to_numpy().any(), f"{transform} produced all-NaN"


def test_empty_panel_returns_empty_frames():
    empty = _panel([])
    assert revision_triangle(empty).empty
    out = revision_frame(empty)
    assert out.empty
    assert list(out.columns) == ["preliminary", "final", "revision"]


# ---------------------------------------------------------------------------
# first / latest release and the revision frame
# ---------------------------------------------------------------------------
def test_first_and_latest_release_pick_the_right_editions():
    panel, dates, truth = _toy_panel(n_dates=8, n_releases=3, seed=3)
    first = first_release(panel)
    last = latest_release(panel)
    # The final release is the truth by construction of the fixture.
    np.testing.assert_allclose(last.to_numpy(), truth, rtol=1e-12)
    # The first release is not (otherwise there is nothing to test).
    observable = first.notna()
    assert not np.allclose(first[observable].to_numpy(),
                           truth[observable.to_numpy()])
    # Where observable, it matches the earliest vintage actually present.
    earliest = panel.sort_values("vintage").groupby("date")["value"].first()
    np.testing.assert_allclose(first[observable].to_numpy(),
                               earliest[observable.to_numpy()].to_numpy())


def test_first_release_is_censored_before_the_archive_starts():
    """A period that ended before the earliest edition has no observable
    first release — the earliest column already holds a revised number.

    Without this the OECD archive (editions from 1999, periods from
    1980) and the ONS workbook (1961 vs 1955) would report a near-final
    estimate as the initial one for two decades of quarters, and every
    revision computed from them would be understated.
    """
    panel, dates, _ = _toy_panel(n_dates=8, n_releases=3, seed=3)
    first_vintage = panel["vintage"].min()
    first = first_release(panel)
    before = first.index < first_vintage
    assert before.any(), "fixture must contain periods predating the archive"
    assert first[before].isna().all()
    assert first[~before].notna().all()
    # Opting out restores the naive reading, so the guard is doing the work.
    naive = revision_frame(panel, transform="level",
                           require_observable_first=False)
    assert len(naive) > len(revision_frame(panel, transform="level"))


def test_revision_frame_revision_is_final_minus_preliminary():
    panel, _, _ = _toy_panel(n_dates=12)
    out = revision_frame(panel, transform="level")
    np.testing.assert_allclose(
        out["revision"].to_numpy(),
        (out["final"] - out["preliminary"]).to_numpy(),
    )


def test_release_index_selects_the_kth_edition():
    panel, _, _ = _toy_panel(n_dates=10, n_releases=3, seed=5)
    r0 = revision_frame(panel, transform="level", release=0)
    r1 = revision_frame(panel, transform="level", release=1)
    ordered = panel.sort_values(["date", "vintage"]).groupby("date")["value"]
    nth0 = ordered.apply(lambda g: g.iloc[0])
    nth1 = ordered.apply(lambda g: g.iloc[1])
    np.testing.assert_allclose(r0["preliminary"].to_numpy(),
                               nth0.reindex(r0.index).to_numpy())
    np.testing.assert_allclose(r1["preliminary"].to_numpy(),
                               nth1.reindex(r1.index).to_numpy())
    assert not np.allclose(r0["preliminary"], r1["preliminary"])


def test_release_index_counts_distinct_editions_not_columns():
    """Triangle columns carry as-of values, so an edition that did not
    restate a period repeats the previous number. Counting columns would
    hand back the same estimate twice and call them two releases."""
    q = pd.Timestamp("2020-01-01")
    rows = [
        # published once, then two editions that do not restate it,
        # then a genuine revision.
        {"date": q, "vintage": pd.Timestamp("2020-05-01"), "value": 100.0},
        {"date": q, "vintage": pd.Timestamp("2021-05-01"), "value": 105.0},
    ]
    # An unrelated period forces extra vintage columns into the triangle,
    # which as-of semantics then forward-carry across the 2020Q1 row.
    other = pd.Timestamp("2020-04-01")
    rows += [
        {"date": other, "vintage": pd.Timestamp("2020-08-01"), "value": 1.0},
        {"date": other, "vintage": pd.Timestamp("2020-11-01"), "value": 2.0},
    ]
    panel = _panel(rows)
    tri = revision_triangle(panel, carry_forward=True)
    carried = tri.loc[q].dropna()
    assert len(carried) > 2, "fixture must produce forward-carried columns"
    # Censoring is a separate concern; switch it off so this test is
    # about edition counting alone.
    out = revision_frame(panel, transform="level", release=1,
                         require_observable_first=False, carry_forward=True)
    # The 2nd distinct edition of 2020Q1 is 105.0, not a repeat of 100.0.
    assert out.loc[q, "preliminary"] == 105.0


def test_negative_release_raises():
    panel, _, _ = _toy_panel(n_dates=4)
    with pytest.raises(ValueError, match="release must be >= 0"):
        revision_frame(panel, release=-1)


def test_negative_release_raises_even_on_an_empty_panel():
    """The guard must precede the triangle build, or an empty panel
    returns a frame and the bad argument is never reported."""
    with pytest.raises(ValueError, match="release must be >= 0"):
        revision_frame(_panel([]), release=-1)


def test_all_zero_revisions_raise_rather_than_reporting_a_verdict():
    """A single-edition panel has nothing to test; regressing a
    identically-zero revision would otherwise emit a confident verdict
    built on a degenerate fit."""
    y = np.arange(20.0)
    with pytest.raises(ValueError, match="every revision is exactly zero"):
        mankiw_shapiro(y, y)


def test_log_transform_warns_when_it_drops_non_positive_values():
    """Net exports and balances go negative; log_diff_pct cannot take
    them and must say so instead of returning a sparser sample."""
    q = pd.date_range("2020-01-01", periods=4, freq="QS")
    rows = [{"date": d, "vintage": d + pd.DateOffset(months=4), "value": v}
            for d, v in zip(q, [-5.0, 3.0, -1.0, 2.0])]
    rows += [{"date": d, "vintage": d + pd.DateOffset(months=7), "value": v}
             for d, v in zip(q, [-5.1, 3.1, -1.1, 2.1])]
    with pytest.warns(UserWarning, match="non-positive"):
        revision_triangle(_panel(rows), transform="log_diff_pct")


# ---------------------------------------------------------------------------
# The news / noise test pair, against DGPs with known answers
# ---------------------------------------------------------------------------
def test_pure_news_dgp_is_called_news():
    """preliminary is an efficient forecast: revision orthogonal to it."""
    rng = np.random.default_rng(7)
    n = 4000
    y_p = rng.standard_normal(n)
    r = 0.6 * rng.standard_normal(n)
    res = mankiw_shapiro(y_p, y_p + r)
    assert res.verdict == "news"
    assert not res.rejects_news
    assert res.rejects_noise
    assert res.beta_on_preliminary == pytest.approx(0.0, abs=0.05)
    assert res.beta_on_final > 0.15


def test_pure_noise_dgp_is_called_noise_and_recovers_the_noise_share():
    """preliminary = truth + independent measurement error."""
    rng = np.random.default_rng(11)
    n = 4000
    sigma_u = 0.5
    y_f = rng.standard_normal(n)
    y_p = y_f + sigma_u * rng.standard_normal(n)
    res = mankiw_shapiro(y_p, y_f)
    assert res.verdict == "noise"
    assert res.rejects_news
    assert not res.rejects_noise
    # Under pure noise beta on preliminary = -var(u)/var(preliminary),
    # which lives strictly inside (-1, 0) -- it is NOT -1.
    theory = sigma_u ** 2 / (1.0 + sigma_u ** 2)
    assert res.beta_on_preliminary == pytest.approx(-theory, abs=0.03)
    assert res.noise_share == pytest.approx(theory, abs=0.03)
    assert -1.0 < res.beta_on_preliminary < 0.0


def test_biased_revisions_are_detected_by_the_mean_revision_test():
    rng = np.random.default_rng(13)
    n = 2000
    y_p = rng.standard_normal(n)
    biased = mankiw_shapiro(y_p, y_p + 0.6 * rng.standard_normal(n) + 0.25)
    assert biased.mean_revision == pytest.approx(0.25, abs=0.05)
    assert biased.p_mean_revision < 0.01


def test_mean_revision_test_has_roughly_correct_size_under_the_null():
    """Unbiased revisions must not be flagged much more than 5% of the time.

    Asserted over replications rather than on one draw: a single-seed
    null test is 5% flaky by construction, and a test that is flaky by
    construction gets muted rather than fixed.
    """
    rng = np.random.default_rng(101)
    rejections = 0
    rejections_at_half = 0
    reps = 200
    for _ in range(reps):
        y_p = rng.standard_normal(250)
        res = mankiw_shapiro(y_p, y_p + 0.6 * rng.standard_normal(250))
        rejections += res.p_mean_revision < 0.05
        rejections_at_half += res.p_mean_revision < 0.5
    # Binomial(200, 0.05) sits under 0.15 with room to spare; a test that
    # always rejected would land at 1.0.
    assert 0.0 < rejections / reps < 0.15
    # Under the null a two-sided p is U(0,1), so it falls below 0.5 about
    # half the time. A one-sided p would be U(0,0.5) and land at 1.0 --
    # this is what catches a dropped factor of two in the p-value.
    assert 0.3 < rejections_at_half / reps < 0.7


def test_result_is_a_frozen_dataclass():
    res = mankiw_shapiro([1.0, 2.0, 3.0, 4.0], [1.1, 2.3, 2.9, 4.4])
    assert isinstance(res, MankiwShapiroResult)
    with pytest.raises(Exception):
        res.beta_on_preliminary = 0.0  # frozen


def test_too_few_observations_raises_rather_than_returning_nan():
    with pytest.raises(ValueError, match="at least 3 aligned observations"):
        mankiw_shapiro([1.0, 2.0], [1.1, 2.1])


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="same length"):
        mankiw_shapiro([1.0, 2.0, 3.0], [1.0, 2.0])


def test_series_inputs_are_aligned_on_index_not_position():
    """n_obs alone cannot tell an index join from positional pairing.

    Under correct alignment the revision is a constant -1.5 at every
    date; under positional pairing it is +0.5. Assert the values.
    """
    idx_a = pd.date_range("2000-01-01", periods=10, freq="QS")
    idx_b = idx_a[2:]
    a = pd.Series(np.arange(10.0), index=idx_a)
    b = pd.Series(np.arange(8.0) + 0.5, index=idx_b)
    res = mankiw_shapiro(a, b)
    assert res.n_obs == 8
    assert res.mean_revision == pytest.approx(-1.5)
    assert res.std_revision == pytest.approx(0.0, abs=1e-12)


def test_nan_rows_are_dropped():
    y_p = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    y_f = np.array([1.1, 2.2, 3.3, np.nan, 5.5])
    res = mankiw_shapiro(y_p, y_f)
    assert res.n_obs == 3


def test_auto_hac_lags_uses_the_newey_west_rule():
    rng = np.random.default_rng(17)
    n = 400
    y_p = rng.standard_normal(n)
    res = mankiw_shapiro(y_p, y_p + rng.standard_normal(n), hac_lags="auto")
    # Independent of the implementation: the Newey-West plug-in rule at
    # n=400 gives floor(4 * 4^(2/9)) = 5.
    assert res.hac_lags == 5
    # And it must actually grow with the sample rather than be constant.
    small = mankiw_shapiro(y_p[:60], (y_p + rng.standard_normal(n))[:60],
                           hac_lags="auto")
    assert small.hac_lags < res.hac_lags


def test_hac_lags_change_the_standard_errors_not_the_point_estimate():
    rng = np.random.default_rng(19)
    n = 500
    y_p = rng.standard_normal(n)
    y_f = y_p + rng.standard_normal(n)
    a = mankiw_shapiro(y_p, y_f, hac_lags=0)
    b = mankiw_shapiro(y_p, y_f, hac_lags=8)
    assert a.beta_on_preliminary == pytest.approx(b.beta_on_preliminary)
    assert a.se_beta_on_preliminary != pytest.approx(b.se_beta_on_preliminary)


def test_negative_hac_lags_raise():
    with pytest.raises(ValueError, match="hac_lags must be >= 0"):
        mankiw_shapiro([1.0, 2.0, 3.0, 4.0], [1.0, 2.1, 3.2, 4.1], hac_lags=-1)


def test_significance_level_is_honoured():
    rng = np.random.default_rng(23)
    n = 300
    y_p = rng.standard_normal(n)
    y_f = y_p + 0.05 * y_p + rng.standard_normal(n)  # weak news violation
    strict = mankiw_shapiro(y_p, y_f, significance=0.001)
    loose = mankiw_shapiro(y_p, y_f, significance=0.5)
    # The p-value sits strictly between the two levels, so both flags are
    # pinned: `A or not B` was a tautology (B implies A) and passed even
    # when `significance` was ignored entirely.
    assert 0.001 < strict.p_beta_on_preliminary < 0.5
    assert not strict.rejects_news
    assert loose.rejects_news
    assert strict.significance == 0.001
    assert loose.verdict != strict.verdict


def test_revision_test_end_to_end_matches_the_two_step_route():
    panel, _, _ = _toy_panel(n_dates=40, seed=29)
    one_step = revision_test(panel, transform="log_diff_pct")
    frame = revision_frame(panel, transform="log_diff_pct")
    two_step = mankiw_shapiro(frame["preliminary"], frame["final"],
                              transform="log_diff_pct")
    assert one_step.beta_on_preliminary == pytest.approx(
        two_step.beta_on_preliminary)
    assert one_step.n_obs == two_step.n_obs
    assert one_step.transform == "log_diff_pct"
