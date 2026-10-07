"""Callaway-Sant'Anna aggregation weights and Sun-Abraham aggregate standard errors.

Two defects, fixed together because the Sun-Abraham estimator is built on the
Callaway-Sant'Anna draws.

1. ``callaway_santanna`` used to average the cohorts *equally* at every event
   time and report the plain mean of the post-treatment ATT(g, t) cells as
   ``att_overall``. Callaway & Sant'Anna (2021) weight by cohort size instead:
   the event study is their eq. 3.4, with weights P(G = g | G + e <= T), and the
   overall summary they recommend is theta^O_sel (eq. 3.11), the cohort-size
   weighted mean of the cohort averages theta_sel(g) (eq. 3.7). Equations and
   page numbers are those of arXiv:1803.09015v4 (PDF pp. 15-19).

2. ``sun_abraham`` aggregated the per-cohort standard errors as if the cohorts
   were independent, sqrt(sum_g w_g^2 se_g^2). Every cohort's 2x2 DiD subtracts
   the same control units, so the cohort estimates are correlated and that
   formula is wrong in either direction. Sun & Abraham (2021), Online Appendix,
   Proposition 6 (arXiv:1804.05785, PDF p. 48): the variance of the IW estimator
   is a full quadratic form in the weights plus a weight-estimation term.

The oracles below are hand-computed from a noiseless DGP (every ATT(g, t) is
recovered exactly), from the planted cohort sizes, and from a closed-form
variance under a unit-specific-trend DGP.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.did import callaway_santanna, sun_abraham

# ---------------------------------------------------------------------------
# Noiseless DGP with heterogeneous, dynamic effects and unequal cohort sizes
# ---------------------------------------------------------------------------
# t = 1..6. Cohort g = 3 has 10 units, cohort g = 5 has 40, 50 never treated.
#   ATT(3, t) = 1 + 0.5 (t - 3):  t = 3, 4, 5, 6 -> 1.0, 1.5, 2.0, 2.5
#   ATT(5, t) = 3 + 1.0 (t - 5):  t = 5, 6       -> 3.0, 4.0
# P(G = 3 | G <= T) = 10/50 = 0.2 and P(G = 5 | G <= T) = 0.8.
#
# eq. 3.4  theta_es(0) = 0.2*1.0 + 0.8*3.0 = 2.6
#          theta_es(1) = 0.2*1.5 + 0.8*4.0 = 3.5
#          theta_es(2) = 2.0,  theta_es(3) = 2.5   (cohort 3 only)
# eq. 3.7  theta_sel(3) = (1 + 1.5 + 2 + 2.5)/4 = 1.75,  theta_sel(5) = 3.5
# eq. 3.11 theta^O_sel = 0.2*1.75 + 0.8*3.5 = 3.15
# eq. 3.10 theta^O_W   = (10*7 + 40*7) / (4*10 + 2*40) = 350/120 = 2.916666...
# eq. 3.12 theta^O_es  = (2.6 + 3.5 + 2.0 + 2.5)/4 = 2.65
#          theta^O_c   = (1.0 + 1.5 + 2.8 + 3.7)/4 = 2.25
#          (theta_c(5) = 0.2*2 + 0.8*3 = 2.8, theta_c(6) = 0.2*2.5 + 0.8*4 = 3.7)
# legacy   unweighted mean of the six post cells = 14/6 = 2.333...
#          unweighted event study e = 0: (1 + 3)/2 = 2.0, e = 1: (1.5 + 4)/2 = 2.75
_N3, _N5, _NNEVER = 10, 40, 50
_ATT = {3: {3: 1.0, 4: 1.5, 5: 2.0, 6: 2.5}, 5: {5: 3.0, 6: 4.0}}

ES_EQ_3_4 = {0: 2.6, 1: 3.5, 2: 2.0, 3: 2.5}
THETA_SEL = {3.0: 1.75, 5.0: 3.5}
OVERALL = {
    "group": 3.15,
    "simple": 350.0 / 120.0,
    "dynamic": 2.65,
    "calendar": 2.25,
    "unweighted": 14.0 / 6.0,
}


def _noiseless_panel() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    cohorts = [3.0] * _N3 + [5.0] * _N5 + [np.nan] * _NNEVER
    rows = []
    for i, g in enumerate(cohorts):
        a_i = rng.normal()
        for t in range(1, 7):
            eff = _ATT[int(g)].get(t, 0.0) if np.isfinite(g) else 0.0
            rows.append({"unit": i, "time": t, "treat_time": g,
                         "y": a_i + 0.3 * t + eff})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def panel():
    return _noiseless_panel()


def _es(res) -> dict:
    es = res.att_event_study
    return dict(zip(es["event_time"].astype(int), es["att"].astype(float)))


# ---------------------------------------------------------------------------
# 1. Callaway-Sant'Anna point estimates follow the paper
# ---------------------------------------------------------------------------
def test_cs_event_study_is_eq_3_4(panel):
    """Old code: 2.0 and 2.75 at e = 0, 1 (equal cohort weights)."""
    es = _es(callaway_santanna(panel, n_boot=0))
    for e, target in ES_EQ_3_4.items():
        assert es[e] == pytest.approx(target, abs=1e-12), (e, es[e], target)
    # pre-periods are exact zeros under noiseless parallel trends
    for e in (-4, -3, -2, -1):
        assert es[e] == pytest.approx(0.0, abs=1e-12)


def test_cs_overall_default_is_theta_sel_eq_3_11(panel):
    """Old code: att_overall = 2.3333, the unweighted mean of the post cells."""
    res = callaway_santanna(panel, n_boot=0)
    assert res.aggregation == "group"
    assert res.att_overall == pytest.approx(OVERALL["group"], abs=1e-12)


@pytest.mark.parametrize("aggregation", sorted(OVERALL))
def test_cs_every_aggregation_matches_its_equation(panel, aggregation):
    res = callaway_santanna(panel, n_boot=0, aggregation=aggregation)
    assert res.aggregation == aggregation
    assert res.att_overall == pytest.approx(OVERALL[aggregation], abs=1e-12)
    table = res.overall_aggregations.set_index("aggregation")
    for name, target in OVERALL.items():
        assert table.loc[name, "att"] == pytest.approx(target, abs=1e-12), name


def test_cs_unweighted_reproduces_the_legacy_event_study(panel):
    es = _es(callaway_santanna(panel, n_boot=0, aggregation="unweighted"))
    assert es[0] == pytest.approx(2.0, abs=1e-12)
    assert es[1] == pytest.approx(2.75, abs=1e-12)


def test_cs_group_effects_eq_3_7(panel):
    grp = callaway_santanna(panel, n_boot=0).att_group.set_index("g")
    for g, target in THETA_SEL.items():
        assert grp.loc[g, "att"] == pytest.approx(target, abs=1e-12)
    assert grp.loc[3.0, "n_g"] == _N3 and grp.loc[5.0, "n_g"] == _N5
    np.testing.assert_allclose(grp["weight"].to_numpy(), [0.2, 0.8], atol=1e-12)


def test_cs_rejects_unknown_aggregation(panel):
    with pytest.raises(ValueError, match="aggregation"):
        callaway_santanna(panel, n_boot=0, aggregation="cohort")


def test_cs_legacy_array_form_weights_overall_by_cohort_size(panel):
    """The four-array form's ``overall_att`` follows the same default."""
    g = panel["treat_time"].fillna(np.inf).to_numpy()
    args = (panel["y"].to_numpy(), g, panel["time"].to_numpy(dtype=float),
            panel["unit"].to_numpy(dtype=float))
    out = callaway_santanna(*args)
    assert out["overall_att"] == pytest.approx(OVERALL["group"], abs=1e-10)
    assert out["aggregation"] == "group"
    legacy = callaway_santanna(*args, aggregation="unweighted")
    assert legacy["overall_att"] == pytest.approx(OVERALL["unweighted"], abs=1e-10)
    simple = callaway_santanna(*args, aggregation="simple")
    assert simple["overall_att"] == pytest.approx(OVERALL["simple"], abs=1e-10)


# ---------------------------------------------------------------------------
# 2. The bootstrap propagates the estimated weights
# ---------------------------------------------------------------------------
def test_bootstrap_propagates_cohort_share_uncertainty(panel):
    """With no noise every resampled ATT(g, t) equals the planted value, so the
    only bootstrap variation left in theta_es(0) comes from re-estimating the
    cohort shares in each draw (the xi^w term of CS Corollary 2, the
    delta' Sigma_f delta term of SA Proposition 6).

    theta_es(0) = 3 - 2 p with p = n_3 / (n_3 + n_5). Resampling 100 units with
    probabilities (0.1, 0.4, 0.5), p has variance ~ 0.2*0.8/50 given
    n_3 + n_5 ~ 50, so se(theta_es(0)) ~ 2*sqrt(0.0032) = 0.113 and
    se(theta_es(1)) ~ 2.5*sqrt(0.0032) = 0.141. Fixed weights would give
    exactly zero.
    """
    res = callaway_santanna(panel, n_boot=600, seed=11)
    gt = res.att_gt[res.att_gt["event_time"] >= 0]
    assert float(gt["se"].max()) < 1e-10          # cells are exact in every draw
    es = res.att_event_study.set_index("event_time")
    assert es.loc[0, "se"] == pytest.approx(0.113, rel=0.2)
    assert es.loc[1, "se"] == pytest.approx(0.141, rel=0.2)   # 3.5 = 4 - 2.5 p
    assert es.loc[2, "se"] < 1e-10                 # one cohort: weight is 1
    # theta^O_sel = 3.5 - 1.75 p  ->  se ~ 1.75 * 0.0566 = 0.099
    assert res.att_overall_se == pytest.approx(0.099, rel=0.2)
    assert res.att_overall_lo < res.att_overall < res.att_overall_hi


def test_cs_overall_interval_is_reported(panel):
    res = callaway_santanna(panel, n_boot=50, seed=0)
    assert np.isfinite(res.att_overall_se) and res.att_overall_se > 0
    table = res.overall_aggregations.set_index("aggregation")
    assert table.loc["group", "se"] == pytest.approx(res.att_overall_se)
    assert "3.11" in table.loc["group", "estimand"]
    assert "overall ATT" in res.summary() and "se" in res.summary()


def test_no_bootstrap_gives_nan_overall_se(panel):
    res = callaway_santanna(panel, n_boot=0)
    assert np.isnan(res.att_overall_se)


# ---------------------------------------------------------------------------
# 3. Sun-Abraham: IW point estimates and joint-bootstrap standard errors
# ---------------------------------------------------------------------------
def test_sa_event_study_is_the_interaction_weighted_estimator(panel):
    """SA eq. (27) with Pr{E_i = e | E_i in h_l} weights = CS eq. 3.4 here."""
    sa = sun_abraham(panel, n_boot=0)
    es = _es(sa)
    for e, target in ES_EQ_3_4.items():
        assert es[e] == pytest.approx(target, abs=1e-12)
    assert sa.aggregation == "simple"
    assert sa.att_overall == pytest.approx(OVERALL["simple"], abs=1e-12)
    assert sun_abraham(panel, n_boot=0, aggregation="dynamic").att_overall == \
        pytest.approx(OVERALL["dynamic"], abs=1e-12)


def _trend_panel(seed: int, n_per_cohort: int = 100, n_never: int = 40) -> pd.DataFrame:
    """Unit-specific linear trends, no treatment effect, three cohorts sharing
    one never-treated control group: the design where the shared controls
    make the cohort estimates positively correlated."""
    rng = np.random.default_rng(seed)
    cohorts = ([3.0] * n_per_cohort + [4.0] * n_per_cohort + [5.0] * n_per_cohort
               + [np.nan] * n_never)
    n = len(cohorts)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    t = np.arange(1, 9)
    y = a[:, None] + b[:, None] * t[None, :]
    return pd.DataFrame({
        "unit": np.repeat(np.arange(n), len(t)),
        "time": np.tile(t, n),
        "treat_time": np.repeat(cohorts, len(t)),
        "y": y.ravel(),
    })


def test_sa_se_is_the_se_of_the_aggregate_not_of_independent_cohorts():
    """y_it = a_i + b_i t. The IW estimate at event time e is
    (e + 1) [sum_g w_g mean_g(b) - mean_never(b)], whose sampling variance
    given the sample is (e + 1)^2 [sum_g w_g^2 s_g^2 / n_g + s_c^2 / n_c]
    (s^2 = within-group variance of b). The old formula summed w_g^2 se_g^2 and
    so counted the control term once per cohort with weight sum_g w_g^2 = 1/3:
    ratio sqrt((1/3)(1/40 + 1/100) / (1/40 + 1/300)) = 0.64 of the truth.
    """
    df = _trend_panel(seed=5)
    sa = sun_abraham(df, n_boot=2000, seed=3)
    es = sa.att_event_study.set_index("event_time")

    unit = df.drop_duplicates("unit")
    b = (df[df["time"] == 2].set_index("unit")["y"]
         - df[df["time"] == 1].set_index("unit")["y"]).loc[unit["unit"]].to_numpy()
    g = unit["treat_time"].to_numpy()
    never = np.isnan(g)
    var_c = b[never].var() / never.sum()
    for e in (0, 1, 2):
        members = [c for c in (3.0, 4.0, 5.0) if c + e <= 8]
        n_g = np.array([np.sum(g == c) for c in members], dtype=float)
        w = n_g / n_g.sum()
        var_t = sum(wi ** 2 * b[g == c].var() / ni for wi, c, ni in zip(w, members, n_g))
        truth = (e + 1) * np.sqrt(var_t + var_c)
        diag_only = (e + 1) * np.sqrt(sum(
            wi ** 2 * (b[g == c].var() / ni + b[never].var() / never.sum())
            for wi, c, ni in zip(w, members, n_g)))
        assert diag_only / truth < 0.7                 # the old formula's error
        assert es.loc[e, "se"] == pytest.approx(truth, rel=0.07), (e, es.loc[e, "se"], truth)


def test_sa_and_cs_event_study_share_the_joint_draws(panel):
    """Same seed, same draws, same weights: identical point and se."""
    df = _trend_panel(seed=9, n_per_cohort=30, n_never=20)
    cs = callaway_santanna(df, n_boot=200, seed=4)
    sa = sun_abraham(df, n_boot=200, seed=4)
    np.testing.assert_allclose(sa.att_event_study["att"], cs.att_event_study["att"],
                               atol=1e-12)
    np.testing.assert_allclose(sa.att_event_study["se"], cs.att_event_study["se"],
                               rtol=1e-12, atol=1e-14)


def test_sa_overall_has_a_joint_bootstrap_se(panel):
    sa = sun_abraham(panel, n_boot=300, seed=2)
    # theta^O_W = (70 p' + 280 (1-p')) / ... varies with the resampled shares
    assert np.isfinite(sa.att_overall_se) and sa.att_overall_se > 0
    from scipy.stats import norm
    z = norm.ppf(0.95)
    assert sa.att_overall_hi - sa.att_overall == pytest.approx(z * sa.att_overall_se)
    assert sa.att_overall - sa.att_overall_lo == pytest.approx(z * sa.att_overall_se)


def test_sa_rejects_the_non_iw_unweighted_aggregation(panel):
    with pytest.raises(ValueError, match="aggregation"):
        sun_abraham(panel, n_boot=0, aggregation="unweighted")


def test_event_study_vcov_diagonal_matches_se():
    df = _trend_panel(seed=9, n_per_cohort=30, n_never=20)
    for res in (callaway_santanna(df, n_boot=150, seed=1),
                sun_abraham(df, n_boot=150, seed=1)):
        V = res.event_study_vcov
        es = res.att_event_study.set_index("event_time")
        assert list(V.index) == list(es.index) == list(V.columns)
        np.testing.assert_allclose(np.sqrt(np.diag(V.to_numpy())), es["se"].to_numpy(),
                                   rtol=1e-10, atol=1e-12)


def test_event_study_vcov_feeds_honest_did():
    """The full bootstrap covariance is a valid ``sigma=`` for honest_did."""
    import warnings

    from puremacro.did import honest_did

    rng = np.random.default_rng(12)
    rows = []
    for i in range(150):
        g = (4.0, 6.0, np.nan)[i % 3]
        a_i = rng.normal()
        for t in range(1, 10):
            eff = 1.0 if np.isfinite(g) and t >= g else 0.0
            rows.append({"unit": i, "time": t, "treat_time": g,
                         "y": a_i + 0.1 * t + eff + rng.normal(0, 0.5)})
    cs = callaway_santanna(pd.DataFrame(rows), n_boot=200, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        diag = honest_did(cs, target_horizon=0).to_frame()
        full = honest_did(cs, sigma=cs.event_study_vcov, target_horizon=0).to_frame()
    assert np.isfinite(full[["ci_lo", "ci_hi"]].to_numpy()).all()
    assert (full["orig_estimate"] == diag["orig_estimate"]).all()
