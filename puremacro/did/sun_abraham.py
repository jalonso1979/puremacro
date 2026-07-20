"""Sun-Abraham (2021) interaction-weighted estimator.

The SA estimator is mathematically equivalent to a Callaway-Sant'Anna
event-study aggregation: it computes cohort-specific average treatment
effects ``ATT(e, g) = E[Y_{g+e} − Y_{g-1} | G = g] − (control trend)``
and aggregates across cohorts at each event time ``e`` using the
**share of treated units in cohort g**:

    ATT_SA(e) = Σ_g  (n_g / Σ_{g'} n_{g'}) · ATT(e, g)

Implemented here as a thin wrapper over
:func:`puremacro.did.callaway_santanna` — the underlying ATT(g, t)
matrix is the same; SA differs only in the aggregation weights
(equal-cohort-share vs. CS's default unweighted cohort-mean).

References
----------
Sun, L. and Abraham, S. (2021). Estimating dynamic treatment effects
    in event studies with heterogeneous treatment effects. JoE 225(2).
"""
from __future__ import annotations

import pandas as pd

from .callaway_santanna import callaway_santanna
from ._results import SunAbrahamResult


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
) -> SunAbrahamResult:
    """Sun-Abraham interaction-weighted event-study aggregation.

    Re-aggregates the same group-time effects estimated by
    :func:`callaway_santanna` using cohort-size shares as weights.

    Parameters
    ----------
    df : DataFrame
        Long-format panel.
    unit, time, outcome, treat_time : str
        Column names. ``treat_time`` is the per-unit first-treatment
        period (NaN for never-treated controls).
    control : {"never_treated", "not_yet_treated"}, default "never_treated"
        Control group used by the underlying CS step.
    n_boot : int, default 200
        Panel-bootstrap replications for SEs (units are resampled).
    alpha : float, default 0.10
        Two-sided coverage = ``1 − α`` (so 0.10 ⇒ 90 % CIs).
    seed : int, default 0
        RNG seed for the bootstrap.

    Returns
    -------
    SunAbrahamResult
        Frozen dataclass with ``att_gt`` (group-time effects, identical
        to the CS estimator), ``att_event_study`` (cohort-share-weighted
        aggregation), and ``att_overall``.

    References
    ----------
    Sun, L. and Abraham, S. (2021). Estimating dynamic treatment effects
        in event studies with heterogeneous treatment effects. JoE
        225(2), 175-199.
    """
    cs = callaway_santanna(
        df, unit=unit, time=time, outcome=outcome,
        treat_time=treat_time, control=control,
        n_boot=n_boot, alpha=alpha, seed=seed,
    )
    att_gt = cs.att_gt

    # Cohort sizes.
    cohort_sizes = (
        df.dropna(subset=[treat_time])
          .groupby(treat_time)[unit]
          .nunique()
          .rename("n_g")
    )
    att_gt = att_gt.merge(cohort_sizes, left_on="g", right_index=True, how="left")

    # Cohort-share-weighted aggregation per event time.
    es_rows = []
    for e, sub in att_gt.groupby("event_time"):
        w = sub["n_g"].values.astype(float)
        w = w / w.sum() if w.sum() > 0 else w
        es_rows.append({
            "event_time": e,
            "att": float((w * sub["att"].values).sum()),
            "se":  float(((w ** 2) * (sub["se"].values ** 2)).sum() ** 0.5),
            "lo":  float((w * sub["lo"].values).sum()),
            "hi":  float((w * sub["hi"].values).sum()),
            "n_cohorts": int(len(sub)),
        })
    es_df = pd.DataFrame(es_rows).sort_values("event_time").reset_index(drop=True)

    post = att_gt[att_gt["event_time"] >= 0]
    if len(post) == 0:
        att_overall = float("nan")
    else:
        w = post["n_g"].values.astype(float)
        w = w / w.sum() if w.sum() > 0 else w
        att_overall = float((w * post["att"].values).sum())

    return SunAbrahamResult(
        att_gt=att_gt.drop(columns=["n_g"]),
        att_event_study=es_df,
        att_overall=att_overall,
    )


__all__ = ["sun_abraham"]
