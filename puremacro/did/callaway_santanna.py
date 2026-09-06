"""Callaway-Sant'Anna (2021) staggered DiD: group-time ATT(g, t).

Primary API — the one documented in :mod:`puremacro.did` and used by
:func:`puremacro.did.sun_abraham`, the examples and the notebooks — takes a
**long-format DataFrame** with columns ``(unit, time, outcome, treat_time)``
and returns a :class:`~puremacro.did._results.CallawaySantannaResult`::

    res = callaway_santanna(df, unit="unit", time="time", outcome="y",
                            treat_time="treat_time")
    res.att_gt           # DataFrame [g, t, event_time, att, se, lo, hi]
    res.att_event_study  # DataFrame [event_time, att, se, lo, hi, n_cohorts]
    res.att_overall      # float

A legacy **four-array** form is still accepted for backwards compatibility::

    out = callaway_santanna(y, group_g, period_t, unit_id)   # -> dict

with ``group_g = np.inf`` marking never-treated units. It returns the plain
dict (``groups``/``periods``/``att_gt``/``se_gt``/``overall_att``) that the
array implementation has always returned, with analytic (non-bootstrap)
standard errors. New code should prefer the DataFrame form.

Estimator
---------
For every treatment cohort ``g`` and period ``t`` the effect is a
2x2 difference-in-differences against the **universal base period**
``g - 1`` (the last observed period strictly before ``g``):

    ATT(g, t) = E[Y_t − Y_{g−1} | G = g] − E[Y_t − Y_{g−1} | control]

The control group is either the never-treated units (default) or the
not-yet-treated ones. Standard errors and bands come from a **panel
bootstrap**: whole units are resampled with replacement, so the within-unit
serial correlation is preserved.

References
----------
Callaway, B. and Sant'Anna, P.H.C. (2021). Difference-in-differences with
    multiple time periods. Journal of Econometrics 225(2), 200-230.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from ._results import CallawaySantannaResult
from .types import PanelDiD

_CONTROL_CHOICES = ("never_treated", "not_yet_treated")


def _resolve_control(control: str, control_group: str | None) -> str:
    """Merge the ``control`` keyword with its ``control_group`` alias."""
    if control_group is None:
        return control
    if control != "never_treated" and control != control_group:
        raise ValueError(
            f"control={control!r} and control_group={control_group!r} "
            "disagree; pass only one of them"
        )
    return control_group


def _maybe_int(x):
    """Return ``int(x)`` when ``x`` is an integral float, else ``x``."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return x
    return int(xf) if np.isfinite(xf) and float(xf).is_integer() else xf


def _att_matrix(
    Y: np.ndarray,
    cohort: np.ndarray,
    times: np.ndarray,
    groups: np.ndarray,
    base_pos: np.ndarray,
    control: str,
) -> np.ndarray:
    """ATT(g, t) matrix of shape ``(len(groups), len(times))``.

    ``Y`` is the wide (unit x time) outcome matrix, ``cohort`` the per-unit
    first-treatment period (``np.inf`` = never treated) and ``base_pos[gi]``
    the column index of cohort ``gi``'s base period (``-1`` if unavailable).
    """
    n_g, n_t = len(groups), len(times)
    A = np.full((n_g, n_t), np.nan)
    never = np.isinf(cohort)

    for gi, g in enumerate(groups):
        b = int(base_pos[gi])
        if b < 0:
            continue
        treated = cohort == g
        if not treated.any():
            continue
        base_col = Y[:, b]
        for ti in range(n_t):
            if ti == b:
                # Normalisation: the base period is zero by construction.
                A[gi, ti] = 0.0
                continue
            dy = Y[:, ti] - base_col
            ok = np.isfinite(dy)
            if control == "not_yet_treated":
                # Not treated by either endpoint of the long difference,
                # and never the cohort under evaluation.
                thresh = max(times[ti], times[b])
                ctrl = (cohort > thresh) & (cohort != g)
            else:
                ctrl = never
            tm = treated & ok
            cm = ctrl & ok
            if tm.sum() == 0 or cm.sum() == 0:
                continue
            A[gi, ti] = float(dy[tm].mean() - dy[cm].mean())
    return A


def _aggregate(
    A: np.ndarray,
    cells: list[tuple[int, int]],
    event_of_cell: np.ndarray,
    event_levels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten ``A`` over ``cells`` and take the cohort mean per event time."""
    flat = np.array([A[gi, ti] for gi, ti in cells], dtype=float)
    es = np.array(
        [
            np.nan if not np.any(m := (event_of_cell == e) & np.isfinite(flat))
            else float(flat[m].mean())
            for e in event_levels
        ],
        dtype=float,
    )
    return flat, es


def _callaway_santanna_df(
    df: pd.DataFrame,
    *,
    unit: str,
    time: str,
    outcome: str,
    treat_time: str,
    control: str,
    n_boot: int,
    alpha: float,
    seed: int,
) -> CallawaySantannaResult:
    if control not in _CONTROL_CHOICES:
        raise ValueError(f"control must be one of {_CONTROL_CHOICES}; got {control!r}")

    pan = PanelDiD(df=df, unit=unit, time=time, outcome=outcome,
                   treat_time=treat_time)
    W = pan.wide_outcome()                       # units x times
    Y = W.to_numpy(dtype=float)
    times = np.asarray(W.columns)
    cohort = pan.cohort_of().reindex(W.index).to_numpy(dtype=float)
    cohort = np.where(np.isnan(cohort), np.inf, cohort)
    groups = np.sort(np.unique(cohort[np.isfinite(cohort)]))

    if groups.size == 0:
        raise ValueError(
            f"no treated units: column {treat_time!r} is NaN everywhere"
        )
    if control == "never_treated" and not np.isinf(cohort).any():
        raise ValueError(
            "control='never_treated' but the panel has no never-treated "
            f"units (no NaN in {treat_time!r}); use control='not_yet_treated'"
        )

    # Base period for each cohort: the last observed period before g.
    base_pos = np.array(
        [int(np.max(np.flatnonzero(times < g))) if np.any(times < g) else -1
         for g in groups]
    )

    A = _att_matrix(Y, cohort, times, groups, base_pos, control)

    # Cells that the point estimate could identify (base period included: it
    # is the exact zero that anchors the event study).
    cells = [(gi, ti) for gi in range(len(groups)) for ti in range(len(times))
             if np.isfinite(A[gi, ti])]
    if not cells:
        raise ValueError(
            "no ATT(g, t) cell is identified — check that treated cohorts have "
            "a pre-period and that the control group is non-empty"
        )
    event_of_cell = np.array([times[ti] - groups[gi] for gi, ti in cells],
                             dtype=float)
    event_levels = np.unique(event_of_cell)

    att_flat, es_point = _aggregate(A, cells, event_of_cell, event_levels)

    # ---- panel bootstrap (resample whole units) --------------------------
    rng = np.random.default_rng(seed)
    n_u = Y.shape[0]
    boot_gt = np.full((max(n_boot, 0), len(cells)), np.nan)
    boot_es = np.full((max(n_boot, 0), len(event_levels)), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n_u, size=n_u)
        Ab = _att_matrix(Y[idx], cohort[idx], times, groups, base_pos, control)
        boot_gt[b], boot_es[b] = _aggregate(Ab, cells, event_of_cell,
                                            event_levels)

    def _band(draws: np.ndarray, point: np.ndarray):
        if draws.shape[0] == 0 or np.all(np.isnan(draws)):
            nan = np.full(point.shape, np.nan)
            return nan, nan.copy(), nan.copy()
        with np.errstate(invalid="ignore"):
            se = np.nanstd(draws, axis=0, ddof=0)
            lo = np.nanpercentile(draws, 100 * alpha / 2, axis=0)
            hi = np.nanpercentile(draws, 100 * (1 - alpha / 2), axis=0)
        return se, lo, hi

    se_gt, lo_gt, hi_gt = _band(boot_gt, att_flat)
    se_es, lo_es, hi_es = _band(boot_es, es_point)

    # The base period is an exact normalisation, not an estimate.
    base_cell = np.array([ti == base_pos[gi] for gi, ti in cells])
    se_gt = np.where(base_cell, 0.0, se_gt)
    lo_gt = np.where(base_cell, 0.0, lo_gt)
    hi_gt = np.where(base_cell, 0.0, hi_gt)

    att_gt = pd.DataFrame({
        "g": [float(groups[gi]) for gi, _ in cells],
        "t": [times[ti] for _, ti in cells],
        "event_time": [_maybe_int(e) for e in event_of_cell],
        "att": att_flat,
        "se": se_gt,
        "lo": lo_gt,
        "hi": hi_gt,
    }).sort_values(["g", "t"]).reset_index(drop=True)

    n_cohorts = np.array([
        int(np.sum((event_of_cell == e) & np.isfinite(att_flat)))
        for e in event_levels
    ])
    att_event_study = pd.DataFrame({
        "event_time": [_maybe_int(e) for e in event_levels],
        "att": es_point,
        "se": se_es,
        "lo": lo_es,
        "hi": hi_es,
        "n_cohorts": n_cohorts,
    }).sort_values("event_time").reset_index(drop=True)

    post = att_gt[att_gt["event_time"] >= 0]["att"].to_numpy(dtype=float)
    att_overall = float(np.nanmean(post)) if np.any(np.isfinite(post)) else float("nan")

    return CallawaySantannaResult(
        att_gt=att_gt,
        att_event_study=att_event_study,
        att_overall=att_overall,
    )


def _callaway_santanna_arrays(
    y: np.ndarray,
    group_g: np.ndarray,
    period_t: np.ndarray,
    unit_id: np.ndarray,
) -> Dict[str, Any]:
    """Legacy four-array form. Returns the historical plain dict."""
    y = np.asarray(y, dtype=float).ravel()
    group_g = np.asarray(group_g, dtype=float).ravel()
    period_t = np.asarray(period_t, dtype=float).ravel()
    unit_id = np.asarray(unit_id, dtype=float).ravel()

    groups = np.sort(np.unique(group_g[~np.isinf(group_g)]))
    periods = np.sort(np.unique(period_t))

    n_groups = len(groups)
    n_periods = len(periods)

    att_matrix = np.full((n_groups, n_periods), np.nan)
    se_matrix = np.full((n_groups, n_periods), np.nan)

    never_treated = np.isinf(group_g)

    for g_idx, g in enumerate(groups):
        base_period = g - 1.0
        if base_period not in periods:
            continue

        cohort_mask = (group_g == g)

        for t_idx, t in enumerate(periods):
            if t == base_period:
                att_matrix[g_idx, t_idx] = 0.0
                se_matrix[g_idx, t_idx] = 0.0
                continue

            sample_mask = (cohort_mask | never_treated) & (
                (period_t == t) | (period_t == base_period)
            )

            y_sub = y[sample_mask]
            g_sub = group_g[sample_mask]
            t_sub = period_t[sample_mask]
            u_sub = unit_id[sample_mask]

            units = np.unique(u_sub)
            n_u = len(units)

            dy = np.zeros(n_u)
            treat = np.zeros(n_u)

            for u_i, uid in enumerate(units):
                m_t = (u_sub == uid) & (t_sub == t)
                m_b = (u_sub == uid) & (t_sub == base_period)

                if np.any(m_t) and np.any(m_b):
                    dy[u_i] = y_sub[m_t][0] - y_sub[m_b][0]
                    treat[u_i] = 1.0 if g_sub[m_t][0] == g else 0.0

            n_treat = int(np.sum(treat == 1.0))
            n_ctrl = int(np.sum(treat == 0.0))

            if n_treat > 0 and n_ctrl > 0:
                att_val = float(np.mean(dy[treat == 1.0]) - np.mean(dy[treat == 0.0]))
                se_val = float(np.sqrt(
                    np.var(dy[treat == 1.0], ddof=1) / n_treat
                    + np.var(dy[treat == 0.0], ddof=1) / n_ctrl
                ))

                att_matrix[g_idx, t_idx] = att_val
                se_matrix[g_idx, t_idx] = se_val

    post_mask = np.zeros_like(att_matrix, dtype=bool)
    for g_idx, g in enumerate(groups):
        post_mask[g_idx, periods >= g] = True

    valid_post = post_mask & ~np.isnan(att_matrix)
    overall_att = float(np.mean(att_matrix[valid_post])) if np.any(valid_post) else 0.0

    return {
        "groups": groups,
        "periods": periods,
        "att_gt": att_matrix,
        "se_gt": se_matrix,
        "overall_att": overall_att,
    }


def callaway_santanna(
    df,
    group_g=None,
    period_t=None,
    unit_id=None,
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
):
    """Callaway-Sant'Anna (2021) group-time average treatment effects.

    Parameters
    ----------
    df : DataFrame
        Long-format panel, one row per ``(unit, time)``. **This is the
        documented API.** Passing four 1-D arrays
        ``(y, group_g, period_t, unit_id)`` instead selects the legacy
        form described below.
    unit, time, outcome, treat_time : str
        Column names. ``treat_time`` is the per-unit first-treatment
        period, ``NaN`` for never-treated controls.
    control : {"never_treated", "not_yet_treated"}, default "never_treated"
        Comparison group for each 2x2 cell.
    n_boot : int, default 200
        Panel-bootstrap replications for SEs and bands (units resampled).
    alpha : float, default 0.10
        Two-sided coverage = ``1 − α`` (so 0.10 ⇒ 90 % bands).
    seed : int, default 0
        RNG seed for the bootstrap.
    ci : float, optional
        Confidence-interval coverage; when given, ``alpha = 1 − ci``.
    control_group : str, optional
        Alias for ``control`` (the ``csdid`` / R ``did`` spelling).
        Passing both with different non-default values is an error.

    Returns
    -------
    CallawaySantannaResult
        Frozen dataclass with ``att_gt`` (columns ``g, t, event_time, att,
        se, lo, hi``), ``att_event_study`` (unweighted cohort mean per event
        time, columns ``event_time, att, se, lo, hi, n_cohorts``) and
        ``att_overall`` (simple mean of the post-treatment ATT(g, t)).

    Legacy array form
    -----------------
    ``callaway_santanna(y, group_g, period_t, unit_id)`` — four aligned 1-D
    arrays with ``group_g = np.inf`` for never-treated units — returns a
    plain ``dict`` with keys ``groups``, ``periods``, ``att_gt`` (matrix),
    ``se_gt`` (analytic, not bootstrapped) and ``overall_att``. Kept so that
    older call sites keep working; new code should pass a DataFrame.

    References
    ----------
    Callaway, B. and Sant'Anna, P.H.C. (2021). Difference-in-differences
        with multiple time periods. Journal of Econometrics 225(2), 200-230.
    """
    if ci is not None:
        alpha = 1.0 - ci
    control = _resolve_control(control, control_group)

    if isinstance(df, pd.DataFrame):
        if group_g is not None or period_t is not None or unit_id is not None:
            raise TypeError(
                "callaway_santanna(df, ...) takes the panel as its only "
                "positional argument; pass column names as keywords "
                "(unit=, time=, outcome=, treat_time=)"
            )
        return _callaway_santanna_df(
            df, unit=unit, time=time, outcome=outcome, treat_time=treat_time,
            control=control, n_boot=n_boot, alpha=alpha, seed=seed,
        )

    if group_g is None or period_t is None or unit_id is None:
        raise TypeError(
            "callaway_santanna expects either a long-format DataFrame "
            "(preferred) or the legacy four arrays "
            "(y, group_g, period_t, unit_id); got a non-DataFrame first "
            "argument with missing companions"
        )
    return _callaway_santanna_arrays(df, group_g, period_t, unit_id)


__all__ = ["callaway_santanna"]
