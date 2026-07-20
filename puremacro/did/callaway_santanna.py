"""Callaway-Sant'Anna (2021) staggered-DiD estimator.

The CS estimator targets group-time average treatment effects on the
treated:

    ATT(g, t) = E[ Y_t(g) − Y_t(0) | G = g ]

— the average causal effect at calendar time ``t`` for the cohort
first treated at ``g``. Identification uses an outcome-regression
DiD on **not-yet-treated** units (or never-treated, depending on
the chosen control group):

    ATT̂(g, t) = E[ Y_t − Y_{g-1} | G = g ]
                  − E[ Y_t − Y_{g-1} | C ]

where ``C`` is the control set. Aggregations over (g, t) yield event-
study profiles, calendar-time profiles, and an overall ATT.

This is the **default workhorse** for staggered-DiD in modern applied
economics. The implementation supports the never-treated control
specification with a simple panel bootstrap for SEs.

References
----------
Callaway, B. and Sant'Anna, P.H.C. (2021). Difference-in-differences
    with multiple time periods. JoE 225(2), 200-230.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .types import PanelDiD
from ._results import CallawaySantannaResult


def _att_gt(panel: PanelDiD, g, t, control_units: np.ndarray) -> float:
    """Compute ATT(g, t) for a given cohort and time, using the supplied
    control units (must not yet be treated at time ``t``)."""
    df = panel.df
    pre = g - 1                                       # base period
    # Cohort g, periods pre and t
    cohort_unit_ids = df.loc[df[panel.treat_time] == g, panel.unit].unique()
    if len(cohort_unit_ids) == 0:
        return float("nan")
    y_pre_g = df.loc[df[panel.unit].isin(cohort_unit_ids)
                       & (df[panel.time] == pre), panel.outcome].mean()
    y_t_g = df.loc[df[panel.unit].isin(cohort_unit_ids)
                     & (df[panel.time] == t), panel.outcome].mean()

    y_pre_c = df.loc[df[panel.unit].isin(control_units)
                       & (df[panel.time] == pre), panel.outcome].mean()
    y_t_c = df.loc[df[panel.unit].isin(control_units)
                     & (df[panel.time] == t), panel.outcome].mean()

    return float((y_t_g - y_pre_g) - (y_t_c - y_pre_c))


def callaway_santanna(
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
) -> CallawaySantannaResult:
    """Estimate ATT(g, t) for every observed cohort × time pair.

    Parameters
    ----------
    df : DataFrame in long format.
    unit, time, outcome, treat_time : column names.
    control : ``"never_treated"`` (default) or ``"not_yet_treated"``.
        Note: ``"not_yet_treated"`` uses period-specific control sets
        (units whose treatment time exceeds the current ``t``), which
        is more inclusive but increases dependence on parallel trends
        across cohorts.
    n_boot : panel-bootstrap replications for SEs (resampling units).
    alpha : two-sided coverage = 1 − α.

    Returns
    -------
    CallawaySantannaResult
        Frozen dataclass with ``att_gt`` (DataFrame of
        ``[g, t, event_time, att, se, lo, hi]``), ``att_event_study``
        (cohort-mean aggregation), and ``att_overall`` (simple-mean
        of post-treatment ATTs).

    References
    ----------
    Callaway, B. and Sant'Anna, P.H.C. (2021). Difference-in-differences
        with multiple time periods. Journal of Econometrics 225(2), 200-230.
    """
    panel = PanelDiD(df=df, unit=unit, time=time,
                      outcome=outcome, treat_time=treat_time)
    rng = np.random.default_rng(seed)
    cohorts = panel.cohorts
    times = panel.times
    cohort_of = panel.cohort_of()

    if control == "never_treated":
        controls_pool = cohort_of.index[cohort_of.isna()].values
    elif control == "not_yet_treated":
        controls_pool = None  # period-specific; built per (g, t)
    else:
        raise ValueError(f"unknown control {control!r}")

    def _control_units_for(g, t) -> np.ndarray:
        if controls_pool is not None:
            return controls_pool
        # Not-yet-treated: any unit whose treat_time > t (or NaN).
        mask = (cohort_of > t) | cohort_of.isna()
        return cohort_of.index[mask].values

    # 1) Point estimate of ATT(g, t).
    rows = []
    for g in cohorts:
        for t in times:
            if t < g - 1:
                continue                              # pre-base period
            ctrl = _control_units_for(g, t)
            if len(ctrl) == 0:
                continue
            att = _att_gt(panel, g, t, ctrl)
            rows.append({"g": g, "t": t, "event_time": t - g, "att": att})
    att_df = pd.DataFrame(rows)

    # 2) Bootstrap SEs by resampling units.
    all_units = panel.units
    boot_atts = {(r["g"], r["t"]): np.empty(n_boot) for _, r in att_df.iterrows()}
    for b in range(n_boot):
        idx = rng.choice(all_units, size=len(all_units), replace=True)
        # Build the bootstrapped panel; need to relabel duplicate units
        # so groupby works.
        boot_dfs = []
        for new_id, u in enumerate(idx):
            d = panel.df[panel.df[unit] == u].copy()
            d[unit] = f"boot_{new_id}"
            boot_dfs.append(d)
        boot_df = pd.concat(boot_dfs, ignore_index=True)
        boot_panel = PanelDiD(df=boot_df, unit=unit, time=time,
                                outcome=outcome, treat_time=treat_time)
        boot_cohort_of = boot_panel.cohort_of()
        if control == "never_treated":
            boot_pool = boot_cohort_of.index[boot_cohort_of.isna()].values
        else:
            boot_pool = None
        for _, r in att_df.iterrows():
            g, t = r["g"], r["t"]
            if boot_pool is not None:
                ctrl = boot_pool
            else:
                mask = (boot_cohort_of > t) | boot_cohort_of.isna()
                ctrl = boot_cohort_of.index[mask].values
            if len(ctrl) == 0:
                boot_atts[(g, t)][b] = np.nan
            else:
                boot_atts[(g, t)][b] = _att_gt(boot_panel, g, t, ctrl)

    se_col, lo_col, hi_col = [], [], []
    for _, r in att_df.iterrows():
        draws = boot_atts[(r["g"], r["t"])]
        se_col.append(float(np.nanstd(draws, ddof=0)))
        lo_col.append(float(np.nanpercentile(draws, 100 * alpha / 2)))
        hi_col.append(float(np.nanpercentile(draws, 100 * (1 - alpha / 2))))
    att_df["se"] = se_col; att_df["lo"] = lo_col; att_df["hi"] = hi_col

    # 3) Event-study aggregation: simple cohort-mean at each event time.
    es_rows = []
    for e, sub in att_df.groupby("event_time"):
        es_rows.append({
            "event_time": e,
            "att": float(sub["att"].mean()),
            "se":  float(np.sqrt((sub["se"] ** 2).sum()) / len(sub)),
            "lo":  float(sub["lo"].mean()),
            "hi":  float(sub["hi"].mean()),
            "n_cohorts": int(len(sub)),
        })
    es_df = pd.DataFrame(es_rows).sort_values("event_time").reset_index(drop=True)

    # 4) Overall post-treatment ATT.
    post = att_df[att_df["event_time"] >= 0]
    att_overall = float(post["att"].mean()) if len(post) else float("nan")

    return CallawaySantannaResult(
        att_gt=att_df,
        att_event_study=es_df,
        att_overall=att_overall,
    )


__all__ = ["callaway_santanna"]
