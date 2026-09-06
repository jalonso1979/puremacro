"""Multi-cohort Synthetic-DiD aggregation.

Wraps the single-cohort SDID estimator already in
:mod:`puremacro.did.synthetic_did` (Arkhangelsky et al. 2021) for the
common applied case of staggered adoption. The procedure is:

  1. Identify treatment **cohorts** — groups of units that first switch
     to treatment at the same calendar time ``g``.
  2. For each cohort ``g``, run single-cohort SDID using cohort
     members as the treated block and a donor pool that is **untreated
     throughout the SDID window** (see ``control`` below):

     - ``"never_treated"`` — never-treated units over the full panel
       (the convention of the ``synthdid`` staggered-adoption vignette).
     - ``"not_yet_treated"`` — never-treated **and** not-yet-treated
       units (``treat_time > g``), with the window truncated at the
       next cohort's switch date so every donor is still untreated in
       every period used. This guarantees a non-empty donor pool even
       in fully-staggered designs without never-treated units (so long
       as the last cohort is not the only one).
     - ``"auto"`` (default) — never-treated donors when at least two
       exist, otherwise the not-yet-treated rule.

  3. Aggregate the cohort-specific ATTs:

     - ``"att"`` — cohort-size-weighted mean of the per-cohort ATTs.
     - ``"att_g_t"`` — full ``cohort × event-time`` grid (per-cohort
       SDID is itself a *single* number; we report it as event-time-0
       and average of event-times >= 0 is identical to the cohort
       ATT).

  4. Standard errors via cluster bootstrap (resampling panel units
     with replacement). The full pipeline (cohort identification +
     per-cohort SDID + aggregation) is repeated on each bootstrap
     replicate so that uncertainty in the cohort weights is reflected
     in the SE.

References
----------
Arkhangelsky, D., Athey, S., Hirshberg, D., Imbens, G. and Wager, S.
    (2021). Synthetic difference-in-differences. American Economic
    Review 111(12), 4088-4118.
Roth, J., Sant'Anna, P., Bilinski, A. and Poe, J. (2023). What's
    trending in difference-in-differences? A synthesis of the recent
    econometrics literature. Journal of Econometrics 235(2),
    2218-2244.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from ._results import SDIDMultiResult
from .synthetic_did import synthetic_did

_CONTROL_CHOICES = ("auto", "never_treated", "not_yet_treated")


def _identify_cohorts(d: np.ndarray, p: np.ndarray, t: np.ndarray):
    """For each unit, the first time at which ``D = 1`` (or ``NaN`` if
    the unit is never treated). Returns ``(cohort_of, units, times)``
    where ``cohort_of`` is a dict ``{unit -> first-treatment-time}``.
    """
    units = np.unique(p)
    times = np.unique(t)
    cohort_of: dict = {}
    for u in units:
        mask = (p == u) & (d == 1)
        if mask.any():
            cohort_of[u] = float(t[mask].min())
        else:
            cohort_of[u] = np.nan
    return cohort_of, units, times


def _build_long_df(y, d, p, t, cohort_of):
    """Assemble the long-format DataFrame ``synthetic_did`` consumes."""
    return pd.DataFrame({
        "unit": p,
        "time": t,
        "y": y,
        "treat_time": np.array(
            [cohort_of[u] for u in p], dtype=float
        ),
    })


def _cohort_donor_window(
    df_full: pd.DataFrame, g: float, cohort_members: np.ndarray, control: str,
) -> tuple[np.ndarray, Optional[float]]:
    """Donor units for cohort ``g`` and the (exclusive) end of the SDID
    window, ``None`` meaning the full panel.

    Donors are only ever used over periods in which they are untreated:
    not-yet-treated donors force the window to stop at the earliest of
    their own switch dates.
    """
    unit_tt = df_full.groupby("unit")["treat_time"].first()
    nt_units = unit_tt.index[unit_tt.isna()].to_numpy()
    nyt_units = unit_tt.index[unit_tt > g].to_numpy()
    nt_units = np.setdiff1d(nt_units, cohort_members)
    nyt_units = np.setdiff1d(nyt_units, cohort_members)

    if control == "never_treated" or (control == "auto" and len(nt_units) >= 2):
        return nt_units, None
    donors = np.concatenate([nt_units, nyt_units])
    if len(nyt_units) == 0:
        return donors, None
    window_end = float(unit_tt.loc[nyt_units].min())
    return donors, window_end


def _sdid_per_cohort(
    df_full: pd.DataFrame,
    *,
    seed: int,
    control: str = "auto",
):
    """Run single-cohort SDID for each cohort on a donor pool that is
    untreated throughout the window. Returns (cohort_times_array,
    cohort_atts_array, cohort_sizes_array).
    """
    cohort_times = np.sort(df_full.loc[~df_full["treat_time"].isna(),
                                          "treat_time"].unique())
    cohort_atts = []
    cohort_sizes = []
    valid_cohort_times = []
    for g in cohort_times:
        # Treated units: cohort members.
        cohort_members = df_full.loc[df_full["treat_time"] == g,
                                       "unit"].unique()
        if len(cohort_members) == 0:
            continue
        donor_units, window_end = _cohort_donor_window(
            df_full, float(g), cohort_members, control,
        )
        if len(donor_units) < 2:
            continue
        keep = np.concatenate([cohort_members, donor_units])
        sub = df_full[df_full["unit"].isin(keep)].copy()
        if window_end is not None:
            # Drop periods from the next cohort's switch onwards so every
            # not-yet-treated donor stays untreated throughout the window.
            sub = sub[sub["time"] < window_end]
        # Re-write treat_time so only cohort members carry g; donors carry NaN.
        sub_treat_time = np.where(sub["unit"].isin(cohort_members), g, np.nan)
        sub["treat_time"] = sub_treat_time
        try:
            with warnings.catch_warnings():
                # n_boot=0 inside synthetic_did triggers harmless
                # "empty slice / ddof <= 0" warnings from nanstd /
                # nanpercentile; we don't use SE/CI from the inner
                # call so suppress them.
                warnings.simplefilter("ignore", RuntimeWarning)
                res = synthetic_did(sub, n_boot=0, seed=seed)
        except ValueError:
            # Too few pre / post periods or donors inside the window.
            continue
        cohort_atts.append(float(res.tau))
        cohort_sizes.append(int(len(cohort_members)))
        valid_cohort_times.append(float(g))
    return (np.asarray(valid_cohort_times),
            np.asarray(cohort_atts),
            np.asarray(cohort_sizes))


def sdid_multi_cohort(
    y,
    treatment,
    panel_id,
    time_id,
    *,
    aggregation: str = "att",
    control: str = "auto",
    n_boot: int = 500,
    seed: Optional[int] = None,
) -> SDIDMultiResult:
    """Multi-cohort Synthetic-DiD aggregator.

    Parameters
    ----------
    y : array-like, shape (N,)
        Outcome.
    treatment : array-like, shape (N,)
        Binary treatment indicator (0/1). Treatment is assumed
        absorbing (once on, stays on); the first ``D = 1`` observation
        per unit defines that unit's cohort.
    panel_id : array-like, shape (N,)
        Panel-unit identifier.
    time_id : array-like, shape (N,)
        Time identifier.
    aggregation : {"att", "att_g_t"}, default "att"
        ``"att"`` returns a single cohort-size-weighted ATT. ``"att_g_t"``
        additionally returns the full ``cohort × event-time`` grid in
        the ``att_g_t`` field.
    control : {"auto", "never_treated", "not_yet_treated"}, default "auto"
        Donor pool for each cohort. ``"never_treated"`` uses never-treated
        units over the full panel; ``"not_yet_treated"`` also admits units
        treated later than ``g`` but truncates the SDID window at their
        earliest switch date so no donor is ever treated inside the
        window; ``"auto"`` picks ``"never_treated"`` when at least two
        never-treated units exist and ``"not_yet_treated"`` otherwise.
    n_boot : int, default 500
        Cluster-bootstrap replications.
    seed : int | None
        RNG seed for the bootstrap.

    Returns
    -------
    SDIDMultiResult
        Frozen dataclass with the aggregated ATT and per-cohort
        breakdown. See :class:`puremacro.did.SDIDMultiResult`.

    Notes
    -----
    A cohort is skipped when fewer than two donors remain or its window
    holds fewer than two pre-periods / one post-period; with
    ``control="not_yet_treated"`` and no never-treated units the last
    cohort therefore never has an estimate. A ``ValueError`` is raised
    when no cohort at all is identifiable.

    Single-cohort SDID is run on each cohort with seed ``0`` so that
    repeated invocations are deterministic. The bootstrap loop uses
    ``seed + b`` per replicate.

    References
    ----------
    Arkhangelsky, D., Athey, S., Hirshberg, D., Imbens, G. and Wager,
        S. (2021). Synthetic difference-in-differences. American
        Economic Review 111(12), 4088-4118.
    Roth, J., Sant'Anna, P., Bilinski, A. and Poe, J. (2023). What's
        trending in difference-in-differences? Journal of
        Econometrics 235(2), 2218-2244.
    """
    if aggregation not in ("att", "att_g_t"):
        raise ValueError(
            f"aggregation must be 'att' or 'att_g_t', got {aggregation!r}"
        )
    if control not in _CONTROL_CHOICES:
        raise ValueError(
            f"control must be one of {_CONTROL_CHOICES}; got {control!r}"
        )

    y = np.asarray(y, dtype=float)
    d = np.asarray(treatment).astype(int)
    p = np.asarray(panel_id)
    t = np.asarray(time_id)
    if not (y.shape == d.shape == p.shape == t.shape) or y.ndim != 1:
        raise ValueError("y, treatment, panel_id, time_id must be 1-D arrays "
                          "of equal length")

    cohort_of, units, _times = _identify_cohorts(d, p, t)
    df_full = _build_long_df(y, d, p, t, cohort_of)
    n_never = int(sum(1 for u in units if np.isnan(cohort_of[u])))
    if control == "never_treated" and n_never < 2:
        raise ValueError(
            "control='never_treated' needs at least two never-treated units "
            f"(found {n_never}); use control='not_yet_treated' or 'auto'"
        )

    point_seed = 0 if seed is None else int(seed)
    cohort_times, cohort_atts, cohort_sizes = _sdid_per_cohort(
        df_full, seed=point_seed, control=control,
    )
    if len(cohort_atts) == 0:
        raise ValueError(
            "no identifiable cohorts: every cohort has fewer than two "
            "untreated donors or fewer than two pre-periods / one "
            "post-period inside its donor window"
        )
    weights = cohort_sizes / cohort_sizes.sum()
    att_point = float(np.sum(weights * cohort_atts))

    # ---------------- Bootstrap ----------------
    rng = np.random.default_rng(seed)
    boot_atts = np.full(n_boot, np.nan)
    n_units = len(units)
    if n_boot > 0:
        # Same loop-invariant full-frame scan as
        # `did/borusyak_jaravel_spiess.py`: group once, index per draw.
        unit_groups = {u: g for u, g in df_full.groupby("unit", sort=False)}
        for b in range(n_boot):
            idx = rng.integers(0, n_units, size=n_units)
            sampled_units = units[idx]
            # Build a long DF for the bootstrapped panel; relabel
            # duplicated units so groupbys don't merge them.
            boot_rows = []
            for new_id, u in enumerate(sampled_units):
                rows = unit_groups[u].copy()
                rows["unit"] = f"boot_{new_id}"
                boot_rows.append(rows)
            boot_df = pd.concat(boot_rows, ignore_index=True)
            try:
                _, b_atts, b_sizes = _sdid_per_cohort(
                    boot_df, seed=point_seed + b + 1, control=control,
                )
                if len(b_atts) > 0 and b_sizes.sum() > 0:
                    w_b = b_sizes / b_sizes.sum()
                    boot_atts[b] = float(np.sum(w_b * b_atts))
            except Exception:
                continue
    if n_boot > 0 and np.isfinite(boot_atts).any():
        se = float(np.nanstd(boot_atts, ddof=0))
    else:
        se = float("nan")

    # ---------------- att_g_t grid ----------------
    if aggregation == "att_g_t":
        rows = []
        for g, atau in zip(cohort_times, cohort_atts):
            # Single-cohort SDID is one number per cohort (an average
            # across post-treatment event-times). We expose it in the
            # grid at event_time = 0 as a single-row representation
            # per cohort. Users wanting per-event-time profiles should
            # use BJS (already in the package).
            rows.append({"cohort": float(g), "event_time": 0,
                          "att": float(atau)})
        att_g_t_df = pd.DataFrame(rows)
    else:
        att_g_t_df = None

    names = tuple(f"g={g}" for g in cohort_times)

    return SDIDMultiResult(
        att=att_point,
        se=se,
        cohort_weights=np.asarray(weights, dtype=float),
        cohort_atts=np.asarray(cohort_atts, dtype=float),
        cohort_times=np.asarray(cohort_times, dtype=float),
        att_g_t=att_g_t_df,
        aggregation=aggregation,
        n_boot=int(n_boot),
        names=names,
    )


__all__ = ["sdid_multi_cohort"]
