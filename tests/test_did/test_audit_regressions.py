"""Regression tests for the v2.3.0 audit findings on ``puremacro.did``.

Each test documents the failing behaviour of the pre-fix code in its
docstring so the defect is legible without the audit report:

* C25  ``sdid_multi_cohort`` kept later-treated donors after their own
       adoption date (the truncation promised in a comment did not exist).
* C26  ``borusyak_jaravel_spiess`` silently set ``lambda_t = 0`` /
       ``alpha_i = 0`` for unidentified fixed effects.
* M49  ``synthetic_did`` omitted the intercepts of Arkhangelsky et al.,
       so it was not invariant to additive level shifts.
* M51  ``synthetic_did`` returned ``tau = NaN`` on an unbalanced panel.
* M50  ``CdHResult`` / ``SDIDMultiResult`` had no export trio or plot,
       ``SyntheticDiDResult`` had no plot; CS / SA / BJS exporters
       emitted a spurious ``index`` column.
* C22  ``control_group=`` (the documented spelling) raised ``TypeError``.
"""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from puremacro.did import (
    borusyak_jaravel_spiess,
    callaway_santanna,
    cdh_did,
    sdid_multi_cohort,
    sun_abraham,
    synthetic_did,
)


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------


def _two_cohort_panel(seed=0, n_nt=0, T=15, sigma=0.2):
    """Audit DGP: cohort g=5 (tau=1), cohort g=10 (tau=5), ``n_nt`` never-
    treated units, time trend 0.3 per period, unit fixed effects."""
    rng = np.random.default_rng(seed)
    cohorts = [5] * 20 + [10] * 20 + [None] * n_nt
    lam = 0.3 * np.arange(T) + rng.normal(scale=0.2, size=T)
    rows = []
    for i, g in enumerate(cohorts):
        a = rng.normal(scale=1.0)
        for t in range(T):
            eff = 0.0 if g is None or t < g else (1.0 if g == 5 else 5.0)
            rows.append({"unit": i, "time": t,
                         "y": a + lam[t] + eff + rng.normal(scale=sigma),
                         "treat_time": np.nan if g is None else float(g)})
    df = pd.DataFrame(rows)
    D = ((~df.treat_time.isna()) & (df.time >= df.treat_time)).astype(int).to_numpy()
    return df, D


def _single_cohort_panel(seed=7, T=16, n_units=45, n_treated=15, g=8):
    """Single cohort, unit FE (sd 1.5), time trend 0.4/period, dynamic
    effect 1.0 + 0.3 * e (mean over post periods = 2.05 for T=16, g=8)."""
    rng = np.random.default_rng(seed)
    lam = 0.4 * np.arange(T) + rng.normal(scale=0.2, size=T)
    rows = []
    for i in range(n_units):
        gi = g if i < n_treated else None
        a = rng.normal(scale=1.5)
        for t in range(T):
            eff = (1.0 + 0.3 * (t - gi)) if gi is not None and t >= gi else 0.0
            rows.append({"unit": i, "time": t,
                         "y": a + lam[t] + eff + rng.normal(scale=0.3),
                         "treat_time": np.nan if gi is None else float(gi)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# C25: sdid_multi_cohort donor window
# ---------------------------------------------------------------------------


def test_sdid_multi_truncates_window_at_next_cohort_switch():
    """Old code: cohort-5 ATT = -1.563 (truth 1.0) because the cohort-10
    donors stayed in the pool after t=10, when they carried tau=5.
    With no never-treated units the only valid donors are the not-yet-
    treated ones over t < 10; the estimate must recover 1.0."""
    df, D = _two_cohort_panel()
    res = sdid_multi_cohort(df.y.values, D, df.unit.values, df.time.values,
                            n_boot=10, seed=1)
    assert list(res.cohort_times) == [5.0]          # last cohort has no donors
    assert abs(res.cohort_atts[0] - 1.0) < 0.25, res.cohort_atts
    assert abs(res.att - 1.0) < 0.25


def test_sdid_multi_prefers_never_treated_donors_when_available():
    """Old code: cohort-5 ATT = -0.297 with 20 never-treated units present,
    because not-yet-treated donors were still used past their switch. With
    ``control="auto"`` and >= 2 never-treated units the donor pool is the
    never-treated group over the full window and both cohorts recover."""
    df, D = _two_cohort_panel(n_nt=20)
    res = sdid_multi_cohort(df.y.values, D, df.unit.values, df.time.values,
                            n_boot=10, seed=1)
    atts = dict(zip(res.cohort_times, res.cohort_atts))
    assert abs(atts[5.0] - 1.0) < 0.25, atts
    assert abs(atts[10.0] - 5.0) < 0.35, atts
    # equals the manual single-cohort SDID with never-treated donors only
    sub = df[(df.treat_time == 5) | df.treat_time.isna()]
    manual = synthetic_did(sub, n_boot=0).tau
    np.testing.assert_allclose(atts[5.0], manual, atol=1e-8)


def test_sdid_multi_not_yet_treated_control_uses_truncated_window():
    """``control="not_yet_treated"`` must equal a manual single-cohort SDID
    whose donors are the later cohort *restricted to t < 10*; the old code
    (no truncation) differed from it by more than 2.5."""
    df, D = _two_cohort_panel(n_nt=20)
    res = sdid_multi_cohort(df.y.values, D, df.unit.values, df.time.values,
                            n_boot=0, control="not_yet_treated")
    sub = df[df.time < 10].copy()
    sub.loc[sub.treat_time == 10, "treat_time"] = np.nan
    manual = synthetic_did(sub, n_boot=0).tau
    atts = dict(zip(res.cohort_times, res.cohort_atts))
    np.testing.assert_allclose(atts[5.0], manual, atol=1e-8)
    assert abs(atts[5.0] - 1.0) < 0.25


def test_sdid_multi_control_validation():
    df, D = _two_cohort_panel()
    with pytest.raises(ValueError, match="control must be one of"):
        sdid_multi_cohort(df.y.values, D, df.unit.values, df.time.values,
                          n_boot=0, control="bogus")
    with pytest.raises(ValueError, match="never-treated"):
        sdid_multi_cohort(df.y.values, D, df.unit.values, df.time.values,
                          n_boot=0, control="never_treated")


# ---------------------------------------------------------------------------
# C26: BJS unidentified fixed effects
# ---------------------------------------------------------------------------


def test_bjs_raises_on_periods_without_untreated_cells():
    """Old code: on the two-cohort panel without never-treated units the
    event study was off by up to 2.81 (e=8: 3.68 vs truth 1.0) with no
    warning, because lambda_t := 0 for t >= 10. Now it raises and names
    the unidentified periods."""
    df, _ = _two_cohort_panel()
    with pytest.raises(ValueError) as excinfo:
        borusyak_jaravel_spiess(df, n_boot=0)
    msg = str(excinfo.value)
    assert "period(s) [10, 11, 12, 13, 14]" in msg
    assert "unidentified='drop'" in msg


def test_bjs_drop_mode_warns_and_excludes_unidentified_cells():
    """With ``unidentified="drop"`` the unidentified cells (t >= 10) leave
    every aggregate: only cohort-5 event times 0..4 remain and they
    recover tau = 1.0."""
    df, _ = _two_cohort_panel()
    with pytest.warns(UserWarning, match="not identified"):
        res = borusyak_jaravel_spiess(df, n_boot=0, unidentified="drop")
    es = res.att_event_study
    assert set(es.event_time.tolist()) == set(range(5))
    assert (es.n_obs == 20).all()
    assert np.abs(es.att - 1.0).max() < 0.25
    assert abs(res.att_overall - 1.0) < 0.2
    assert (res.tau_it.time < 10).all()
    assert np.isfinite(res.tau_it.tau).all()


def test_bjs_identified_panel_unchanged_and_no_warning():
    """Panels that are identified everywhere neither raise nor warn."""
    df, _ = _two_cohort_panel(n_nt=20)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = borusyak_jaravel_spiess(df, n_boot=5, seed=0)
    es = res.att_event_study
    truth = {e: (20 * 1.0 + (20 * 5.0 if e <= 4 else 0)) / (20 + (20 if e <= 4 else 0))
             for e in range(10)}
    worst = max(abs(r.att - truth[int(r.event_time)]) for _, r in es.iterrows())
    assert worst < 0.3


def test_bjs_rejects_unknown_unidentified_choice():
    df, _ = _two_cohort_panel(n_nt=20)
    with pytest.raises(ValueError, match="unidentified must be one of"):
        borusyak_jaravel_spiess(df, n_boot=0, unidentified="ignore")


def test_bjs_ci_kwarg_sets_alpha():
    df, _ = _two_cohort_panel(n_nt=20)
    r90 = borusyak_jaravel_spiess(df, n_boot=20, seed=0)
    r95 = borusyak_jaravel_spiess(df, n_boot=20, seed=0, ci=0.95)
    w90 = (r90.att_event_study.hi - r90.att_event_study.lo).mean()
    w95 = (r95.att_event_study.hi - r95.att_event_study.lo).mean()
    assert w95 > w90


# ---------------------------------------------------------------------------
# M49: SDID intercepts / shift invariance
# ---------------------------------------------------------------------------


def test_sdid_invariant_to_unit_and_time_level_shifts():
    """Old code: tau = 1.9895 -> 2.1586 after adding +5 to the treated
    units' outcome, 2.2185 after +50 (omega moved by up to 0.50), 2.0647
    under common time shifts. With the intercepts of Arkhangelsky et al.
    every additive unit / time shift leaves tau and omega unchanged."""
    df = _single_cohort_panel()
    base = synthetic_did(df, n_boot=0)
    shifted = {
        "treated +50": df.assign(y=df.y + 50.0 * df.treat_time.notna()),
        "unit shifts": df.assign(y=df.y + df.unit.map(lambda u: 3.0 * (u % 7))),
        "time shifts": df.assign(y=df.y + df.time.map(lambda t: 2.0 * (t % 3))),
    }
    for name, d in shifted.items():
        res = synthetic_did(d, n_boot=0)
        assert abs(res.tau - base.tau) < 1e-6, (name, res.tau, base.tau)
        assert np.abs(res.omega.values - base.omega.values).max() < 1e-6, name
        assert np.abs(res.lambda_w.values - base.lambda_w.values).max() < 1e-6, name
    # and the estimate is right (truth 2.05 = mean of 1.0 + 0.3 e, e = 0..7)
    assert abs(base.tau - 2.05) < 0.15


def test_sdid_weights_on_simplex_and_ci_kwarg():
    df = _single_cohort_panel()
    res = synthetic_did(df, n_boot=30, seed=1, ci=0.95)
    np.testing.assert_allclose(res.omega.sum(), 1.0, atol=1e-8)
    np.testing.assert_allclose(res.lambda_w.sum(), 1.0, atol=1e-8)
    assert (res.omega >= -1e-12).all() and (res.lambda_w >= -1e-12).all()
    r90 = synthetic_did(df, n_boot=30, seed=1)
    assert (res.hi - res.lo) > (r90.hi - r90.lo)
    assert np.isfinite(res.se)


def test_sdid_trajectories_and_plot():
    """The result carries the treated-mean and synthetic paths and plots them."""
    df = _single_cohort_panel()
    res = synthetic_did(df, n_boot=0)
    assert res.y_treated is not None and res.y_synthetic is not None
    assert len(res.y_treated) == 16 and len(res.y_synthetic) == 16
    # post-period gap minus lambda-weighted pre gap reproduces tau
    gap = res.y_treated - res.y_synthetic
    post = gap[gap.index >= res.treatment_time].mean()
    pre = float((res.lambda_w * gap[gap.index < res.treatment_time]).sum())
    np.testing.assert_allclose(post - pre, res.tau, atol=1e-10)
    fig = res.plot()
    assert isinstance(fig, Figure)
    plt.close("all")


# ---------------------------------------------------------------------------
# M51: SDID unbalanced panel
# ---------------------------------------------------------------------------


def test_sdid_unbalanced_panel_raises_naming_cells():
    """Old code: dropping 10% of rows gave tau = NaN with no error."""
    df = _single_cohort_panel()
    dfu = df.drop(df.sample(frac=0.1, random_state=0).index)
    with pytest.raises(ValueError, match="balanced panel") as excinfo:
        synthetic_did(dfu, n_boot=0)
    assert "missing" in str(excinfo.value) and "(" in str(excinfo.value)


def test_sdid_duplicate_cells_raise():
    df = _single_cohort_panel()
    dup = pd.concat([df, df.iloc[[3]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        synthetic_did(dup, n_boot=0)


# ---------------------------------------------------------------------------
# M50: presentation contract
# ---------------------------------------------------------------------------


def _check_exports(res):
    md = res.to_markdown()
    assert isinstance(md, str) and md.startswith("|")
    header = md.splitlines()[0]
    assert "index" not in header.split("|")[1:2] and "| index |" not in header
    assert "\\begin{tabular}" in res.to_latex()
    assert "#table(" in res.to_typst()
    assert isinstance(res.to_frame(), pd.DataFrame)
    fig = res.plot()
    assert isinstance(fig, Figure)
    plt.close("all")


def test_cdh_result_exports_and_plot():
    """Old code: CdHResult had no to_frame/to_markdown/to_latex/to_typst/plot."""
    df, D = _two_cohort_panel(n_nt=20)
    res = cdh_did(df.y.values, D, df.unit.values, df.time.values,
                  n_boot=10, seed=0, horizons=(1, 2, 3))
    _check_exports(res)
    tab = res.to_frame()
    assert list(tab.columns) == ["estimand", "horizon", "att", "se"]
    assert tab.iloc[0]["estimand"] == "DID_M" and len(tab) == 4


def test_sdid_multi_result_exports_and_plot():
    """Old code: SDIDMultiResult had no to_frame/to_markdown/to_latex/to_typst/plot."""
    df, D = _two_cohort_panel(n_nt=20)
    res = sdid_multi_cohort(df.y.values, D, df.unit.values, df.time.values,
                            n_boot=5, seed=1)
    _check_exports(res)
    tab = res.to_frame()
    assert list(tab.columns) == ["cohort", "weight", "att", "se"]
    assert tab.iloc[-1]["cohort"] == "aggregate"
    np.testing.assert_allclose(tab.iloc[-1]["att"], res.att)


def test_cs_sa_bjs_exporters_have_no_index_column():
    """Old code: to_markdown/to_latex/to_typst on CS / SA / BJS results
    started with a spurious ``index`` column holding 0, 1, 2, ..."""
    df, _ = _two_cohort_panel(n_nt=20)
    cs = callaway_santanna(df, n_boot=3)
    sa = sun_abraham(df, n_boot=3)
    bjs = borusyak_jaravel_spiess(df, n_boot=3)
    for res in (cs, sa, bjs):
        header = res.to_markdown().splitlines()[0]
        cells = [c.strip() for c in header.strip("|").split("|")]
        assert cells[0] == "event_time", cells
        assert "index" not in cells
        assert res.to_latex().splitlines()[1].startswith("event\\_time")
        assert "[* index *]" not in res.to_typst()
        # opt back in explicitly
        assert "index" in res.to_markdown(index=True).splitlines()[0]


# ---------------------------------------------------------------------------
# C22: control_group alias
# ---------------------------------------------------------------------------


def test_control_group_alias_matches_control():
    """Old code: callaway_santanna(control_group=...) -> TypeError, the
    spelling used by docs/did.md and docs/quickstart.md."""
    df, _ = _two_cohort_panel(n_nt=20)
    a = callaway_santanna(df, control="not_yet_treated", n_boot=0)
    b = callaway_santanna(df, control_group="not_yet_treated", n_boot=0)
    pd.testing.assert_frame_equal(a.att_gt, b.att_gt)
    sa_a = sun_abraham(df, control="not_yet_treated", n_boot=0)
    sa_b = sun_abraham(df, control_group="not_yet_treated", n_boot=0)
    pd.testing.assert_frame_equal(sa_a.att_event_study, sa_b.att_event_study)
    with pytest.raises(ValueError, match="disagree"):
        callaway_santanna(df, control="not_yet_treated",
                          control_group="never_treated", n_boot=0)
