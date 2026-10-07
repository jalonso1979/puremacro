"""Sun-Abraham (2021) interaction-weighted (IW) event-study estimator.

Sun and Abraham estimate every cohort-specific effect
``CATT(e, l) = E[Y_{e+l} − Y_{e-1} | E = e] − (control trend)`` (their eq. 28
with pre-period ``s = e − 1``) and average them at each relative period ``l``
with the **sample share of each cohort** among the cohorts observed there
(their eq. 27; the weights are Pr{E_i = e | E_i in h_l}):

    nu_l = Σ_e  (n_e / Σ_{e' in h_l} n_{e'}) · CATT(e, l)

Without covariates and with never-treated controls, Sun and Abraham note that
their approach "coincides with Callaway and Sant'Anna" (arXiv:1804.05785,
p. 24): the CATT(e, l) are the Callaway-Sant'Anna ATT(g, g+l) and nu_l is CS's
event study theta_es(l) (CS 2021 eq. 3.4), which also weights by cohort size.
The two estimators do **not** differ in how they weight cohorts. This function
therefore reuses the group-time estimates *and the joint bootstrap draws* of
:func:`puremacro.did.callaway_santanna`.

Inference
---------
The IW estimate is a weighted sum of cohort estimates that share the same
control units, so they are correlated. Its variance (SA Online Appendix,
Proposition 6, eqs. 72-73, p. 48) is the full quadratic form in the weights
plus a term for estimating the weights. The standard error reported here is
the standard deviation of the aggregate over joint unit-level bootstrap draws,
in which both the cohort effects and the cohort shares are re-estimated, so
both parts are included. ``lo``/``hi`` are the normal band ``att -/+ z se``.
Up to puremacro 4.3.0 the aggregate ``se`` was ``sqrt(Σ w² se_g²)``, which
treats the cohorts as independent; it was between 0.6 and 1.4 times the
Monte Carlo truth depending on the error process.

References
----------
Sun, L. and Abraham, S. (2021). Estimating dynamic treatment effects
    in event studies with heterogeneous treatment effects. JoE 225(2).
Callaway, B. and Sant'Anna, P.H.C. (2021). Difference-in-differences with
    multiple time periods. JoE 225(2), 200-230.
"""
from __future__ import annotations

import pandas as pd
from scipy.stats import norm as _norm

from .callaway_santanna import (
    _AGGREGATIONS,
    _ESTIMANDS,
    _att_gt_frame,
    _cs_fit,
    _event_vcov_frame,
    _maybe_int,
    _n_cohorts,
    _percentile_band,
    _resolve_aggregation,
    _resolve_control,
)
from ._results import SunAbrahamResult

#: Overall summaries built from interaction-weighted (cohort-share) weights.
_SA_AGGREGATIONS = ("simple", "group", "dynamic", "calendar")


def sun_abraham(
    df: pd.DataFrame,
    *,
    unit: str = "unit",
    time: str = "time",
    outcome: str = "y",
    treat_time: str = "treat_time",
    control: str = "never_treated",
    n_boot: int = 200,
    alpha: float = 0.10,
    seed: int = 0,
    ci: float | None = None,
    control_group: str | None = None,
    aggregation: str = "simple",
) -> SunAbrahamResult:
    """Sun-Abraham interaction-weighted event study.

    Averages the cohort-specific effects estimated by
    :func:`callaway_santanna` at each relative period, with the sample share
    of each cohort among the cohorts observed at that period as weights
    (Sun & Abraham 2021, eq. 27).

    Parameters
    ----------
    df : DataFrame
        Long-format panel.
    unit, time, outcome, treat_time : str
        Column names. ``treat_time`` is the per-unit first-treatment
        period (NaN for never-treated controls).
    control : {"never_treated", "not_yet_treated"}, default "never_treated"
        Control group used by the underlying 2x2 comparisons.
    n_boot : int, default 200
        Panel-bootstrap replications (whole units are resampled; the cohort
        effects and the cohort shares are re-estimated jointly on each draw).
    alpha : float, default 0.10
        Two-sided coverage = ``1 − α`` (so 0.10 ⇒ 90 % CIs).
    seed : int, default 0
        RNG seed for the bootstrap. With the same seed the draws are the
        ones :func:`callaway_santanna` uses, so the two event-study ``se``
        columns coincide.
    ci : float, optional
        Confidence interval coverage (alpha = 1.0 - ci).
    control_group : str, optional
        Alias for ``control`` (the ``csdid`` / R ``did`` spelling).
    aggregation : {"simple", "group", "dynamic", "calendar"}, default "simple"
        Overall summary reported as ``att_overall`` (cohort-size weights
        throughout; equation numbers of Callaway & Sant'Anna 2021):
        ``"simple"`` — every post-treatment cell weighted by cohort size
        (eq. 3.10, the value this function has always returned);
        ``"group"`` — eq. 3.11; ``"dynamic"`` — the mean of the event-study
        coefficients over ``e >= 0``, which is Sun and Abraham's own
        nu_g (eq. 25) for g = the post-treatment periods; ``"calendar"`` —
        eq. 3.12.

    Returns
    -------
    SunAbrahamResult
        Frozen dataclass with ``att_gt`` (group-time effects, identical
        to the CS estimator), ``att_event_study`` (interaction-weighted
        nu_l with joint-bootstrap ``se`` and normal band), ``att_overall``
        with ``att_overall_se``/``_lo``/``_hi``, ``overall_aggregations``
        and ``event_study_vcov``.

    References
    ----------
    Sun, L. and Abraham, S. (2021). Estimating dynamic treatment effects
        in event studies with heterogeneous treatment effects. JoE
        225(2), 175-199. IW estimator: eq. 27; variance: Online Appendix,
        Proposition 6.
    """
    if ci is not None:
        alpha = 1.0 - ci
    control = _resolve_control(control, control_group)
    try:
        aggregation = _resolve_aggregation(aggregation, _SA_AGGREGATIONS)
    except ValueError as exc:
        raise ValueError(
            f"{exc}. The interaction-weighted estimator weights cohorts by "
            "their share by construction; the equal-weight 'unweighted' rule "
            "is available only as callaway_santanna(aggregation='unweighted')"
        ) from None

    fit = _cs_fit(
        df, unit=unit, time=time, outcome=outcome, treat_time=treat_time,
        control=control, n_boot=n_boot, seed=seed,
    )
    z = float(_norm.ppf(1.0 - alpha / 2.0))

    # Event study: nu_l, with the SD of the aggregate over the joint draws.
    es_point = fit.point["es"]
    se_es = _percentile_band(fit.boot_es, es_point, alpha)[0]
    es_df = pd.DataFrame({
        "event_time": [_maybe_int(e) for e in fit.lay.event_levels],
        "att": es_point,
        "se": se_es,
        "lo": es_point - z * se_es,
        "hi": es_point + z * se_es,
        "n_cohorts": _n_cohorts(fit),
    }).sort_values("event_time").reset_index(drop=True)

    cols = [_AGGREGATIONS.index(a) for a in _SA_AGGREGATIONS]
    o_point = fit.point["overall"][cols]
    o_se = _percentile_band(fit.boot_overall[:, cols], o_point, alpha)[0]
    overall = pd.DataFrame({
        "aggregation": list(_SA_AGGREGATIONS),
        "estimand": [_ESTIMANDS[a] for a in _SA_AGGREGATIONS],
        "att": o_point,
        "se": o_se,
        "lo": o_point - z * o_se,
        "hi": o_point + z * o_se,
    })
    k = _SA_AGGREGATIONS.index(aggregation)

    return SunAbrahamResult(
        att_gt=_att_gt_frame(fit, alpha),
        att_event_study=es_df,
        att_overall=float(o_point[k]),
        att_overall_se=float(o_se[k]),
        att_overall_lo=float(o_point[k] - z * o_se[k]),
        att_overall_hi=float(o_point[k] + z * o_se[k]),
        aggregation=aggregation,
        alpha=float(alpha),
        overall_aggregations=overall,
        event_study_vcov=_event_vcov_frame(fit, fit.boot_es),
    )


__all__ = ["sun_abraham"]
