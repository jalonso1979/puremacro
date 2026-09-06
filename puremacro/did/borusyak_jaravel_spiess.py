"""Borusyak-Jaravel-Spiess (2022) imputation estimator.

The BJS estimator works in two steps:

  1. **Impute** the untreated potential outcome ``Y_{i,t}(0)`` for
     every (i, t) by fitting a unit + time fixed-effects model on the
     **untreated** observations only — never-treated units and the
     pre-treatment observations of ever-treated units. The fitted
     model gives the counterfactual ``Ŷ(0)`` everywhere — including
     the treated cells we want to evaluate.

  2. **Aggregate**: for each treated cell, the individual treatment
     effect is ``τ̂_{i,t} = Y_{i,t} − Ŷ_{i,t}(0)``. Average these to
     get cohort-time, event-time, or overall ATTs.

This is the cleanest staggered-DiD estimator conceptually: it imposes
the "no anticipation" + parallel-trends assumption explicitly via the
imputation step, and the resulting estimator is efficient under
homoskedastic errors. The trade-off is that it requires untreated
observations covering every unit and every period that carries a
treated cell: a treated cell ``(i, t)`` is **identified only if** unit
``i`` has at least one untreated observation (so ``α_i`` is estimable)
and period ``t`` has at least one untreated observation (so ``λ_t`` is
estimable). In a panel with no never-treated units, every period from
the last cohort's switch onwards violates the second condition. By
default such cells raise a ``ValueError``; ``unidentified="drop"``
warns and excludes them from every aggregate instead.

References
----------
Borusyak, K., Jaravel, X. and Spiess, J. (2024). Revisiting event-
    study designs: robust and efficient estimation. Review of Economic
    Studies 91(6), 3253-3285.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ._results import BorusyakJaravelSpiessResult

_UNIDENTIFIED_CHOICES = ("raise", "drop")


def _impute_two_way_fe(
    df: pd.DataFrame, *, unit: str, time: str, outcome: str,
    untreated_mask: pd.Series,
) -> pd.Series:
    """Fit Y_{i,t} = α_i + λ_t + ε on the untreated cells, then return
    fitted values everywhere (treated + untreated).

    Cells whose unit has no untreated observation (``α_i`` not
    estimable) or whose period has none (``λ_t`` not estimable) get
    ``NaN`` — never a silent zero.
    """
    sub = df.loc[untreated_mask, [unit, time, outcome]].copy()
    units = sub[unit].unique()
    times = sub[time].unique()
    u_to_i = {u: i for i, u in enumerate(units)}
    t_to_j = {t: j for j, t in enumerate(times)}
    n_u, n_t = len(units), len(times)
    rows = sub[unit].map(u_to_i).values
    cols = sub[time].map(t_to_j).values

    # Iterative two-way demeaning to avoid building a (n_u + n_t)-wide design.
    y = sub[outcome].values.astype(float)
    alpha = np.zeros(n_u)
    lam = np.zeros(n_t)
    grand = float(y.mean())
    for _ in range(200):
        # Update alpha.
        diff = y - lam[cols] - grand
        new_alpha = np.bincount(rows, weights=diff, minlength=n_u) / np.maximum(
            np.bincount(rows, minlength=n_u), 1
        )
        # Update lam.
        diff = y - new_alpha[rows] - grand
        new_lam = np.bincount(cols, weights=diff, minlength=n_t) / np.maximum(
            np.bincount(cols, minlength=n_t), 1
        )
        if (np.max(np.abs(new_alpha - alpha)) < 1e-9
                and np.max(np.abs(new_lam - lam)) < 1e-9):
            alpha, lam = new_alpha, new_lam
            break
        alpha, lam = new_alpha, new_lam

    # Predict for *every* (unit, time) row of the original df. Units or
    # times absent from the untreated set have no estimable fixed effect:
    # their imputation is NaN and the caller decides whether to raise or
    # drop those cells.
    full_unit_idx = df[unit].map(lambda u: u_to_i.get(u, -1)).values
    full_time_idx = df[time].map(lambda t: t_to_j.get(t, -1)).values
    a = np.where(full_unit_idx >= 0, alpha[np.maximum(full_unit_idx, 0)], np.nan)
    l = np.where(full_time_idx >= 0, lam[np.maximum(full_time_idx, 0)], np.nan)
    return pd.Series(grand + a + l, index=df.index)


def _unidentified_cells(
    df: pd.DataFrame, *, unit: str, time: str, untreated_mask: np.ndarray,
    treated_mask: np.ndarray,
) -> tuple[np.ndarray, list, list]:
    """Treated cells whose unit or period has no untreated observation.

    Returns ``(mask_over_df, bad_units, bad_times)``.
    """
    units_ok = set(df.loc[untreated_mask, unit].unique().tolist())
    times_ok = set(df.loc[untreated_mask, time].unique().tolist())
    unit_bad = ~df[unit].isin(units_ok).to_numpy()
    time_bad = ~df[time].isin(times_ok).to_numpy()
    bad = treated_mask & (unit_bad | time_bad)
    bad_units = sorted(df.loc[treated_mask & unit_bad, unit].unique().tolist())
    bad_times = sorted(df.loc[treated_mask & time_bad, time].unique().tolist())
    return bad, bad_units, bad_times


def borusyak_jaravel_spiess(
    df: pd.DataFrame,
    *,
    unit: str = "unit",
    time: str = "time",
    outcome: str = "y",
    treat_time: str = "treat_time",
    n_boot: int = 200,
    alpha: float = 0.10,
    seed: int = 0,
    unidentified: str = "raise",
    ci: float | None = None,
) -> BorusyakJaravelSpiessResult:
    """BJS imputation event-study.

    The "untreated" cells are (a) all observations of never-treated
    units and (b) pre-treatment observations of ever-treated units.
    A two-way fixed-effects model is fit on those cells and used to
    impute ``Ŷ_{i,t}(0)`` for the treated cells, giving per-cell
    treatment effects ``τ̂_{i,t} = Y_{i,t} − Ŷ_{i,t}(0)``.

    Parameters
    ----------
    df : DataFrame
        Long-format panel.
    unit, time, outcome, treat_time : str
        Column names. ``treat_time`` is the per-unit first-treatment
        period (NaN for never-treated controls).
    n_boot : int, default 200
        Panel-bootstrap replications for SEs (units are resampled).
    alpha : float, default 0.10
        Two-sided coverage = ``1 − α`` (so 0.10 ⇒ 90 % CIs).
    seed : int, default 0
        RNG seed for the bootstrap.
    unidentified : {"raise", "drop"}, default "raise"
        What to do with treated cells whose time fixed effect (period
        with no untreated observation — e.g. every period after the
        last cohort switches in a panel without never-treated units) or
        unit fixed effect (unit with no pre-treatment observation) is
        not estimable from the untreated cells. ``"raise"`` stops with
        a ``ValueError`` naming the periods / units; ``"drop"`` emits a
        ``UserWarning`` and excludes those cells from ``tau_it``,
        ``att_event_study`` and ``att_overall``.
    ci : float, optional
        Confidence-interval coverage; when given, ``alpha = 1 − ci``
        (same convention as :func:`callaway_santanna`).

    Returns
    -------
    BorusyakJaravelSpiessResult
        Frozen dataclass with ``tau_it`` (per-treated-cell estimates),
        ``att_event_study`` (event-time aggregation, ``e >= 0`` only —
        BJS evaluates τ̂ on treated cells, so there are no pre-trend
        rows), and ``att_overall`` (cell-weighted mean over ``e >= 0``).

    References
    ----------
    Borusyak, K., Jaravel, X. and Spiess, J. (2024). Revisiting event-
        study designs: robust and efficient estimation. Review of
        Economic Studies 91(6), 3253-3285.
    """
    if unidentified not in _UNIDENTIFIED_CHOICES:
        raise ValueError(
            f"unidentified must be one of {_UNIDENTIFIED_CHOICES}; "
            f"got {unidentified!r}"
        )
    if ci is not None:
        alpha = 1.0 - ci
    df = df.copy()
    rng = np.random.default_rng(seed)
    treat = df[treat_time].values.astype(float)
    is_treated_cell = (~np.isnan(treat)) & (df[time].values >= treat)
    untreated_mask = pd.Series(~is_treated_cell, index=df.index)

    bad_cells, bad_units, bad_times = _unidentified_cells(
        df, unit=unit, time=time, untreated_mask=~is_treated_cell,
        treated_mask=is_treated_cell,
    )
    if bad_cells.any():
        parts = []
        if bad_times:
            parts.append(
                f"period(s) {bad_times} have no untreated observation "
                "(time fixed effect not estimable)"
            )
        if bad_units:
            parts.append(
                f"unit(s) {bad_units} have no untreated observation "
                "(unit fixed effect not estimable)"
            )
        msg = (
            f"{int(bad_cells.sum())} treated cell(s) are not identified by "
            "the untreated cells: " + "; ".join(parts) + ". "
        )
        if unidentified == "raise":
            raise ValueError(
                msg + "Add never-treated units, drop those periods / units, "
                "or pass unidentified='drop' to exclude these cells from "
                "every aggregate (with a warning)."
            )
        warnings.warn(
            msg + "These cells are dropped from tau_it, att_event_study "
            "and att_overall.",
            UserWarning, stacklevel=2,
        )

    yhat0 = _impute_two_way_fe(df, unit=unit, time=time, outcome=outcome,
                                untreated_mask=untreated_mask)
    df["__yhat0__"] = yhat0
    df["__tau__"] = df[outcome] - df["__yhat0__"]
    df["__event_time__"] = df[time] - df[treat_time]
    df["__treated_cell__"] = is_treated_cell & ~bad_cells

    treated = df[df["__treated_cell__"]].copy()

    # Bootstrap by resampling units.
    units = df[unit].unique()
    boot_es: dict[int, np.ndarray] = {}        # event_time → (n_boot,) draws
    # Group once, index per draw. `df[df[unit] == u]` is a full-frame scan, and
    # it ran once per sampled unit per draw -- O(n_boot x n_units x N). The
    # groups are identical across draws, so the scan is loop-invariant.
    unit_groups = {u: g for u, g in df.groupby(unit, sort=False)}
    for b in range(n_boot):
        idx = rng.choice(units, size=len(units), replace=True)
        boot_dfs = []
        for new_id, u in enumerate(idx):
            d = unit_groups[u].copy()
            d[unit] = f"boot_{new_id}"
            boot_dfs.append(d)
        boot_df = pd.concat(boot_dfs, ignore_index=True)
        b_treat = boot_df[treat_time].values.astype(float)
        b_treated_cell = (~np.isnan(b_treat)) & (boot_df[time].values >= b_treat)
        b_untreated = pd.Series(~b_treated_cell, index=boot_df.index)
        b_yhat0 = _impute_two_way_fe(boot_df, unit=unit, time=time,
                                       outcome=outcome,
                                       untreated_mask=b_untreated)
        # A resampled panel can lose every untreated observation of some
        # period; those cells are NaN and simply drop out of the draw.
        boot_df["__tau__"] = boot_df[outcome] - b_yhat0
        boot_df["__event_time__"] = boot_df[time] - boot_df[treat_time]
        b_keep = b_treated_cell & np.isfinite(boot_df["__tau__"].to_numpy(dtype=float))
        for e, sub in boot_df[b_keep].groupby("__event_time__"):
            boot_es.setdefault(int(e), np.full(n_boot, np.nan))[b] = float(sub["__tau__"].mean())

    rows = []
    for e, sub in treated.groupby("__event_time__"):
        att = float(sub["__tau__"].mean())
        draws = boot_es.get(int(e), np.array([]))
        if draws.size and np.isfinite(draws).any():
            se = float(np.nanstd(draws, ddof=0))
            lo = float(np.nanpercentile(draws, 100 * alpha / 2))
            hi = float(np.nanpercentile(draws, 100 * (1 - alpha / 2)))
        else:
            se = lo = hi = float("nan")
        rows.append({"event_time": int(e), "att": att,
                     "se": se, "lo": lo, "hi": hi,
                     "n_obs": int(len(sub))})
    if rows:
        es_df = pd.DataFrame(rows).sort_values("event_time").reset_index(drop=True)
    else:
        es_df = pd.DataFrame(columns=["event_time", "att", "se", "lo", "hi", "n_obs"])

    post = es_df[es_df["event_time"] >= 0]
    att_overall = float((post["att"] * post["n_obs"]).sum() / post["n_obs"].sum()) \
                   if len(post) and post["n_obs"].sum() > 0 else float("nan")

    return BorusyakJaravelSpiessResult(
        tau_it=treated[[unit, time, "__event_time__", "__tau__"]].rename(
            columns={"__event_time__": "event_time", "__tau__": "tau"},
        ),
        att_event_study=es_df,
        att_overall=att_overall,
    )


__all__ = ["borusyak_jaravel_spiess"]
