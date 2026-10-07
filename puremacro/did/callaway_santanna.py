"""Callaway-Sant'Anna (2021) staggered DiD: group-time ATT(g, t) and its aggregations.

Primary API — the one documented in :mod:`puremacro.did` and used by
:func:`puremacro.did.sun_abraham`, the examples and the notebooks — takes a
**long-format DataFrame** with columns ``(unit, time, outcome, treat_time)``
and returns a :class:`~puremacro.did._results.CallawaySantannaResult`::

    res = callaway_santanna(df, unit="unit", time="time", outcome="y",
                            treat_time="treat_time")
    res.att_gt               # DataFrame [g, t, event_time, att, se, lo, hi]
    res.att_event_study      # DataFrame [event_time, att, se, lo, hi, n_cohorts]
    res.att_overall          # float: theta^O_sel (eq. 3.11) by default
    res.att_overall_se       # its bootstrap standard error (+ _lo / _hi)
    res.att_group            # theta_sel(g) per cohort (eq. 3.7)
    res.overall_aggregations # every overall summary, with se / lo / hi

A legacy **four-array** form is still accepted for backwards compatibility::

    out = callaway_santanna(y, group_g, period_t, unit_id)   # -> dict

with ``group_g = np.inf`` marking never-treated units. It returns the plain
dict (``groups``/``periods``/``att_gt``/``se_gt``/``overall_att``/
``aggregation``) with analytic (non-bootstrap) cell standard errors. New code
should prefer the DataFrame form.

Estimator
---------
For every treatment cohort ``g`` and period ``t`` the effect is a
2x2 difference-in-differences against the **universal base period**
``g - 1`` (the last observed period strictly before ``g``):

    ATT(g, t) = E[Y_t − Y_{g−1} | G = g] − E[Y_t − Y_{g−1} | control]

The control group is either the never-treated units (default) or the
not-yet-treated ones.

Aggregation
-----------
Callaway and Sant'Anna (2021), Section 3; equation numbers are those of
arXiv:1803.09015v4 (pp. 15-19). Every aggregate weights the cohorts by their
**size**: with ``n_g`` the number of units in cohort ``g``, ``n_g / sum n``
estimates P(G = g | G <= T).

* Event study, eq. 3.4:
  ``theta_es(e) = sum_g n_g ATT(g, g+e) / sum_g n_g`` over the cohorts whose
  cell at event time ``e`` is identified. Pre-periods ``e < 0`` use the same
  weights (as the authors' R package ``did::aggte`` does).
* ``aggregation="group"`` (default), eq. 3.11:
  ``theta^O_sel = sum_g n_g theta_sel(g) / sum_g n_g``, where
  ``theta_sel(g)`` (eq. 3.7) is the mean of cohort ``g``'s post-treatment
  cells. This is the summary the paper recommends ("the average effect of
  participating in the treatment experienced by all units that ever
  participated"), the analogue of the 2x2 ATT.
* ``"simple"``, eq. 3.10: ``theta^O_W``, every post-treatment cell weighted by
  its cohort size. It puts more weight on early cohorts, which have more
  post-treatment cells.
* ``"dynamic"``, eq. 3.12: ``theta^O_es``, the plain mean of ``theta_es(e)``
  over the identified ``e >= 0``.
* ``"calendar"``, eq. 3.12: ``theta^O_c``, the plain mean of the calendar-time
  effects ``theta_c(t)`` (eq. 3.8) over the periods in which some cohort is
  treated.
* ``"unweighted"``: the rule puremacro 4.3.0 and earlier used. Every cohort
  counts once in the event study and ``att_overall`` is the plain mean of the
  post-treatment cells. It is **not** a Callaway-Sant'Anna estimand and is
  kept only to reproduce old numbers.

Inference
---------
Standard errors and bands come from a **panel bootstrap**: whole units are
resampled with replacement, so within-unit serial correlation is preserved.
Every aggregate is recomputed on each draw, including the cohort sizes
``n_g``, so the uncertainty from estimating the weights (the ``xi^w`` term of
the paper's Corollary 2, p. 26) is propagated, and the covariance between
cohorts that share control units is kept. ``se`` is the bootstrap standard
deviation and ``lo``/``hi`` are pointwise percentile bands.

References
----------
Callaway, B. and Sant'Anna, P.H.C. (2021). Difference-in-differences with
    multiple time periods. Journal of Econometrics 225(2), 200-230.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd

from ._results import CallawaySantannaResult
from .types import PanelDiD

_CONTROL_CHOICES = ("never_treated", "not_yet_treated")

#: Overall-ATT aggregations, in the column order of the bootstrap draws.
_AGGREGATIONS = ("group", "simple", "dynamic", "calendar", "unweighted")

#: What each aggregation estimates (equation numbers: arXiv:1803.09015v4).
_ESTIMANDS = {
    "group": "theta^O_sel, CS (2021) eq. 3.11: cohort-size-weighted mean of the "
             "cohort averages theta_sel(g) (eq. 3.7)",
    "simple": "theta^O_W, CS (2021) eq. 3.10: post-treatment ATT(g,t) cells "
              "weighted by cohort size",
    "dynamic": "theta^O_es, CS (2021) eq. 3.12: mean of the event study "
               "theta_es(e) (eq. 3.4) over e >= 0",
    "calendar": "theta^O_c, CS (2021) eq. 3.12: mean over treated periods of "
                "theta_c(t) (eq. 3.8)",
    "unweighted": "puremacro <= 4.3.0 rule, not a CS (2021) estimand: plain "
                  "mean of the post-treatment ATT(g,t) cells",
}


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


def _resolve_aggregation(aggregation: str, allowed=_AGGREGATIONS) -> str:
    """Validate the ``aggregation`` keyword (case-insensitive)."""
    key = str(aggregation).lower().strip()
    if key not in allowed:
        raise ValueError(
            f"aggregation must be one of {tuple(allowed)}; got {aggregation!r}"
        )
    return key


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


def _cohort_sizes(cohort: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Number of units in each cohort (the n_g behind P(G = g | G <= T))."""
    return np.array([np.sum(cohort == g) for g in groups], dtype=float)


def _weighted_mean_by(index: np.ndarray, values: np.ndarray,
                      weights: np.ndarray, n_bins: int) -> np.ndarray:
    """Weighted mean of ``values`` within each bin of ``index``.

    Cells whose value is not finite or whose weight is zero are left out, and
    the weights are renormalised over the remaining ones; an empty bin is NaN.
    """
    ok = np.isfinite(values) & (weights > 0)
    w = np.where(ok, weights, 0.0)
    num = np.bincount(index, weights=w * np.where(ok, values, 0.0), minlength=n_bins)
    den = np.bincount(index, weights=w, minlength=n_bins)
    out = np.full(n_bins, np.nan)
    np.divide(num, den, out=out, where=den > 0)
    return out


@dataclass(frozen=True)
class _Layout:
    """Where each identified ATT(g, t) cell sits (cohort, period, event time)."""

    cells: list
    cell_g: np.ndarray          # cohort index of each cell
    cell_t: np.ndarray          # period index of each cell
    event_of_cell: np.ndarray   # e = t - g
    event_levels: np.ndarray    # sorted distinct event times
    event_idx: np.ndarray       # position of each cell's e in event_levels
    post: np.ndarray            # e >= 0
    n_groups: int
    n_times: int


def _aggregate(flat: np.ndarray, n_g: np.ndarray, lay: _Layout) -> dict:
    """Every Callaway-Sant'Anna aggregate of one ATT(g, t) vector.

    ``flat`` holds ATT(g, t) on ``lay.cells`` (NaN where a bootstrap draw
    could not identify a cell) and ``n_g`` the cohort sizes. Returns the
    cohort-size-weighted event study (eq. 3.4), the equal-weight legacy event
    study, the cohort effects theta_sel(g) (eq. 3.7) and the overall summaries
    in the order of :data:`_AGGREGATIONS`.
    """
    n_e = len(lay.event_levels)
    w_cell = n_g[lay.cell_g]
    es = _weighted_mean_by(lay.event_idx, flat, w_cell, n_e)
    es_equal = _weighted_mean_by(lay.event_idx, flat, np.ones_like(flat), n_e)

    p = lay.post
    fp = flat[p]
    theta_sel = _weighted_mean_by(lay.cell_g[p], fp, np.ones_like(fp), lay.n_groups)

    def _wmean(x: np.ndarray, w: np.ndarray) -> float:
        ok = np.isfinite(x) & (w > 0)
        return float(np.sum(w[ok] * x[ok]) / np.sum(w[ok])) if ok.any() else np.nan

    def _mean(x: np.ndarray) -> float:
        ok = np.isfinite(x)
        return float(x[ok].mean()) if ok.any() else np.nan

    theta_c = _weighted_mean_by(lay.cell_t[p], fp, w_cell[p], lay.n_times)
    overall = np.array([
        _wmean(theta_sel, n_g),                          # group     eq. 3.11
        _wmean(fp, w_cell[p]),                           # simple    eq. 3.10
        _mean(es[lay.event_levels >= 0]),                # dynamic   eq. 3.12
        _mean(theta_c),                                  # calendar  eq. 3.12
        _mean(fp),                                       # unweighted (legacy)
    ])
    return {"es": es, "es_equal": es_equal, "group": theta_sel, "overall": overall}


@dataclass(frozen=True)
class _CSFit:
    """Point estimates and joint bootstrap draws shared by CS and SA."""

    groups: np.ndarray
    times: np.ndarray
    n_g: np.ndarray
    lay: _Layout
    base_cell: np.ndarray
    point: dict
    boot_gt: np.ndarray        # (n_boot, n_cells)
    boot_es: np.ndarray        # (n_boot, n_event_times), eq. 3.4 weights
    boot_es_equal: np.ndarray  # (n_boot, n_event_times), legacy equal weights
    boot_group: np.ndarray     # (n_boot, n_groups), theta_sel(g)
    boot_overall: np.ndarray   # (n_boot, len(_AGGREGATIONS))


def _cs_fit(
    df: pd.DataFrame,
    *,
    unit: str,
    time: str,
    outcome: str,
    treat_time: str,
    control: str,
    n_boot: int,
    seed: int,
) -> _CSFit:
    """Estimate every ATT(g, t) and run the joint panel bootstrap."""
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
    cell_g = np.array([gi for gi, _ in cells], dtype=int)
    cell_t = np.array([ti for _, ti in cells], dtype=int)
    event_of_cell = np.array([times[ti] - groups[gi] for gi, ti in cells],
                             dtype=float)
    event_levels = np.unique(event_of_cell)
    lay = _Layout(
        cells=cells, cell_g=cell_g, cell_t=cell_t, event_of_cell=event_of_cell,
        event_levels=event_levels,
        event_idx=np.searchsorted(event_levels, event_of_cell),
        post=event_of_cell >= 0, n_groups=len(groups), n_times=len(times),
    )

    n_g = _cohort_sizes(cohort, groups)
    att_flat = A[cell_g, cell_t]
    point = _aggregate(att_flat, n_g, lay)
    point["gt"] = att_flat

    # ---- panel bootstrap (resample whole units) --------------------------
    # Each draw re-estimates every ATT(g, t) AND the cohort sizes, so the
    # aggregates carry the cross-cohort covariance (shared controls) and the
    # weight-estimation uncertainty.
    rng = np.random.default_rng(seed)
    n_u = Y.shape[0]
    nb = max(int(n_boot), 0)
    boot_gt = np.full((nb, len(cells)), np.nan)
    boot_es = np.full((nb, len(event_levels)), np.nan)
    boot_es_equal = np.full((nb, len(event_levels)), np.nan)
    boot_group = np.full((nb, len(groups)), np.nan)
    boot_overall = np.full((nb, len(_AGGREGATIONS)), np.nan)
    for b in range(nb):
        idx = rng.integers(0, n_u, size=n_u)
        Ab = _att_matrix(Y[idx], cohort[idx], times, groups, base_pos, control)
        flat_b = Ab[cell_g, cell_t]
        agg_b = _aggregate(flat_b, _cohort_sizes(cohort[idx], groups), lay)
        boot_gt[b] = flat_b
        boot_es[b] = agg_b["es"]
        boot_es_equal[b] = agg_b["es_equal"]
        boot_group[b] = agg_b["group"]
        boot_overall[b] = agg_b["overall"]

    base_cell = np.array([ti == base_pos[gi] for gi, ti in cells])
    return _CSFit(
        groups=groups, times=times, n_g=n_g, lay=lay, base_cell=base_cell,
        point=point, boot_gt=boot_gt, boot_es=boot_es,
        boot_es_equal=boot_es_equal, boot_group=boot_group,
        boot_overall=boot_overall,
    )


def _percentile_band(draws: np.ndarray, point: np.ndarray, alpha: float):
    """``(se, lo, hi)``: bootstrap SD and pointwise percentile band."""
    point = np.asarray(point, dtype=float)
    if draws.shape[0] == 0 or np.all(np.isnan(draws)):
        nan = np.full(point.shape, np.nan)
        return nan, nan.copy(), nan.copy()
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        # all-NaN columns (cells a draw could not identify) are expected
        warnings.simplefilter("ignore", RuntimeWarning)
        se = np.nanstd(draws, axis=0, ddof=0)
        lo = np.nanpercentile(draws, 100 * alpha / 2, axis=0)
        hi = np.nanpercentile(draws, 100 * (1 - alpha / 2), axis=0)
    return se, lo, hi


def _nan_cov(draws: np.ndarray) -> np.ndarray:
    """Pairwise-complete bootstrap covariance (ddof = 0).

    Its diagonal is the squared bootstrap SD reported in the ``se`` column,
    so it is the joint counterpart of that column.
    """
    k = draws.shape[1]
    if draws.shape[0] == 0:
        return np.full((k, k), np.nan)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        centred = draws - np.nanmean(draws, axis=0)
    ok = np.isfinite(centred)
    c0 = np.where(ok, centred, 0.0)
    num = c0.T @ c0
    cnt = ok.T.astype(float) @ ok.astype(float)
    V = np.full((k, k), np.nan)
    np.divide(num, cnt, out=V, where=cnt > 0)
    return V


def _att_gt_frame(fit: _CSFit, alpha: float) -> pd.DataFrame:
    """Group-time ATT table with percentile bootstrap bands."""
    lay = fit.lay
    se, lo, hi = _percentile_band(fit.boot_gt, fit.point["gt"], alpha)
    # The base period is an exact normalisation, not an estimate.
    se = np.where(fit.base_cell, 0.0, se)
    lo = np.where(fit.base_cell, 0.0, lo)
    hi = np.where(fit.base_cell, 0.0, hi)
    return pd.DataFrame({
        "g": [float(fit.groups[gi]) for gi in lay.cell_g],
        "t": [fit.times[ti] for ti in lay.cell_t],
        "event_time": [_maybe_int(e) for e in lay.event_of_cell],
        "att": fit.point["gt"],
        "se": se,
        "lo": lo,
        "hi": hi,
    }).sort_values(["g", "t"]).reset_index(drop=True)


def _n_cohorts(fit: _CSFit) -> np.ndarray:
    """Number of cohorts identifying each event time in the point estimate."""
    lay = fit.lay
    return np.array([
        int(np.sum((lay.event_of_cell == e) & np.isfinite(fit.point["gt"])))
        for e in lay.event_levels
    ])


def _event_vcov_frame(fit: _CSFit, draws: np.ndarray) -> pd.DataFrame:
    """Bootstrap covariance of the event-study coefficients, labelled by e."""
    labels = [_maybe_int(e) for e in fit.lay.event_levels]
    return pd.DataFrame(_nan_cov(draws), index=pd.Index(labels, name="event_time"),
                        columns=pd.Index(labels, name="event_time"))


def _group_frame(fit: _CSFit, se, lo, hi) -> pd.DataFrame:
    """theta_sel(g) per cohort with its cohort size and eq. 3.11 weight."""
    theta = fit.point["group"]
    keep = np.isfinite(theta) & (fit.n_g > 0)
    w = np.where(keep, fit.n_g, 0.0)
    w = w / w.sum() if w.sum() > 0 else w
    return pd.DataFrame({
        "g": fit.groups[keep].astype(float),
        "n_g": fit.n_g[keep].astype(int),
        "weight": w[keep],
        "att": theta[keep],
        "se": np.asarray(se)[keep],
        "lo": np.asarray(lo)[keep],
        "hi": np.asarray(hi)[keep],
    })


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
    aggregation: str = "group",
) -> CallawaySantannaResult:
    aggregation = _resolve_aggregation(aggregation)
    fit = _cs_fit(df, unit=unit, time=time, outcome=outcome,
                  treat_time=treat_time, control=control, n_boot=n_boot,
                  seed=seed)

    # The legacy rule weights the event study equally; every CS aggregation
    # uses eq. 3.4.
    legacy = aggregation == "unweighted"
    es_point = fit.point["es_equal"] if legacy else fit.point["es"]
    es_draws = fit.boot_es_equal if legacy else fit.boot_es
    se_es, lo_es, hi_es = _percentile_band(es_draws, es_point, alpha)

    att_event_study = pd.DataFrame({
        "event_time": [_maybe_int(e) for e in fit.lay.event_levels],
        "att": es_point,
        "se": se_es,
        "lo": lo_es,
        "hi": hi_es,
        "n_cohorts": _n_cohorts(fit),
    }).sort_values("event_time").reset_index(drop=True)

    se_o, lo_o, hi_o = _percentile_band(fit.boot_overall, fit.point["overall"], alpha)
    overall = pd.DataFrame({
        "aggregation": list(_AGGREGATIONS),
        "estimand": [_ESTIMANDS[a] for a in _AGGREGATIONS],
        "att": fit.point["overall"],
        "se": se_o,
        "lo": lo_o,
        "hi": hi_o,
    })
    k = _AGGREGATIONS.index(aggregation)

    se_g, lo_g, hi_g = _percentile_band(fit.boot_group, fit.point["group"], alpha)

    return CallawaySantannaResult(
        att_gt=_att_gt_frame(fit, alpha),
        att_event_study=att_event_study,
        att_overall=float(fit.point["overall"][k]),
        att_overall_se=float(se_o[k]),
        att_overall_lo=float(lo_o[k]),
        att_overall_hi=float(hi_o[k]),
        aggregation=aggregation,
        alpha=float(alpha),
        att_group=_group_frame(fit, se_g, lo_g, hi_g),
        overall_aggregations=overall,
        event_study_vcov=_event_vcov_frame(fit, es_draws),
    )


def _callaway_santanna_arrays(
    y: np.ndarray,
    group_g: np.ndarray,
    period_t: np.ndarray,
    unit_id: np.ndarray,
    aggregation: str = "group",
) -> Dict[str, Any]:
    """Legacy four-array form. Returns the historical plain dict.

    ``overall_att`` follows ``aggregation`` exactly as the DataFrame form
    does (default ``"group"``, eq. 3.11), with cohort sizes counted as
    distinct ``unit_id`` values per cohort; ``"unweighted"`` reproduces the
    plain mean of the post-treatment cells that this form returned up to
    puremacro 4.3.0.
    """
    aggregation = _resolve_aggregation(aggregation)
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
    if np.any(valid_post):
        cells = [(gi, ti) for gi in range(n_groups) for ti in range(n_periods)
                 if np.isfinite(att_matrix[gi, ti])]
        cell_g = np.array([gi for gi, _ in cells], dtype=int)
        cell_t = np.array([ti for _, ti in cells], dtype=int)
        event_of_cell = periods[cell_t] - groups[cell_g]
        event_levels = np.unique(event_of_cell)
        lay = _Layout(
            cells=cells, cell_g=cell_g, cell_t=cell_t,
            event_of_cell=event_of_cell, event_levels=event_levels,
            event_idx=np.searchsorted(event_levels, event_of_cell),
            post=event_of_cell >= 0, n_groups=n_groups, n_times=n_periods,
        )
        n_g = np.array([np.unique(unit_id[group_g == g]).size for g in groups],
                       dtype=float)
        agg = _aggregate(att_matrix[cell_g, cell_t], n_g, lay)
        overall_att = float(agg["overall"][_AGGREGATIONS.index(aggregation)])
    else:
        overall_att = 0.0

    return {
        "groups": groups,
        "periods": periods,
        "att_gt": att_matrix,
        "se_gt": se_matrix,
        "overall_att": overall_att,
        "aggregation": aggregation,
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
    aggregation: str = "group",
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
        Panel-bootstrap replications for SEs and bands (units resampled;
        the cohort sizes behind every aggregation weight are re-estimated
        on each draw).
    alpha : float, default 0.10
        Two-sided coverage = ``1 − α`` (so 0.10 ⇒ 90 % bands).
    seed : int, default 0
        RNG seed for the bootstrap.
    ci : float, optional
        Confidence-interval coverage; when given, ``alpha = 1 − ci``.
    control_group : str, optional
        Alias for ``control`` (the ``csdid`` / R ``did`` spelling).
        Passing both with different non-default values is an error.
    aggregation : {"group", "simple", "dynamic", "calendar", "unweighted"}, default "group"
        Which overall summary ``att_overall`` reports (equation numbers of
        arXiv:1803.09015v4; ``n_g`` = units in cohort ``g``):

        * ``"group"`` — theta^O_sel, eq. 3.11, the paper's recommended
          summary: the ``n_g``-weighted mean of the cohort averages
          theta_sel(g) (eq. 3.7);
        * ``"simple"`` — theta^O_W, eq. 3.10: post-treatment cells weighted
          by ``n_g``;
        * ``"dynamic"`` — theta^O_es, eq. 3.12: mean of theta_es(e), e >= 0;
        * ``"calendar"`` — theta^O_c, eq. 3.12: mean of theta_c(t) (eq. 3.8)
          over the treated periods;
        * ``"unweighted"`` — the puremacro <= 4.3.0 rule, **not** a CS
          estimand: plain mean of the post-treatment cells, and an event
          study that weights every cohort equally. Use it only to reproduce
          old numbers.

        Every aggregation except ``"unweighted"`` reports the eq. 3.4
        event study. All five overall summaries are always available in
        ``overall_aggregations``.

    Returns
    -------
    CallawaySantannaResult
        Frozen dataclass with

        * ``att_gt`` (columns ``g, t, event_time, att, se, lo, hi``);
        * ``att_event_study`` — theta_es(e), eq. 3.4: the ATT(g, g+e) of the
          cohorts identifying event time ``e``, weighted by cohort size
          (columns ``event_time, att, se, lo, hi, n_cohorts``);
        * ``att_overall`` with ``att_overall_se``, ``att_overall_lo``,
          ``att_overall_hi`` — the summary chosen by ``aggregation``;
        * ``att_group`` — theta_sel(g) per cohort with ``n_g`` and its
          eq. 3.11 weight;
        * ``overall_aggregations`` — every overall summary with its
          estimand, se and band;
        * ``event_study_vcov`` — bootstrap covariance of the event-study
          coefficients (for :func:`puremacro.did.honest_did`'s ``sigma=``).

        ``se`` is the bootstrap standard deviation; ``lo``/``hi`` are
        pointwise percentile bands at level ``1 − α``.

    Legacy array form
    -----------------
    ``callaway_santanna(y, group_g, period_t, unit_id)`` — four aligned 1-D
    arrays with ``group_g = np.inf`` for never-treated units — returns a
    plain ``dict`` with keys ``groups``, ``periods``, ``att_gt`` (matrix),
    ``se_gt`` (analytic, not bootstrapped), ``overall_att`` (following
    ``aggregation``) and ``aggregation``. Kept so that older call sites keep
    working; new code should pass a DataFrame.

    References
    ----------
    Callaway, B. and Sant'Anna, P.H.C. (2021). Difference-in-differences
        with multiple time periods. Journal of Econometrics 225(2), 200-230.
        Aggregations: Section 3, eqs. 3.4, 3.7, 3.8, 3.10-3.12 and Table 1;
        inference for aggregates: Corollary 2.
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
            aggregation=aggregation,
        )

    if group_g is None or period_t is None or unit_id is None:
        raise TypeError(
            "callaway_santanna expects either a long-format DataFrame "
            "(preferred) or the legacy four arrays "
            "(y, group_g, period_t, unit_id); got a non-DataFrame first "
            "argument with missing companions"
        )
    return _callaway_santanna_arrays(df, group_g, period_t, unit_id,
                                     aggregation=aggregation)


__all__ = ["callaway_santanna"]
