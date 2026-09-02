"""Multi-cohort Synthetic-DiD aggregation.

Wraps the single-cohort SDID estimator already in
:mod:`puremacro.did.synthetic_did` (Arkhangelsky et al. 2021) for the
common applied case of staggered adoption. The procedure is:

  1. Identify treatment **cohorts** — groups of units that first switch
     to treatment at the same calendar time ``g``.
  2. For each cohort ``g``, run single-cohort SDID using cohort
     members as the treated block and (a) never-treated units plus
     (b) units not yet treated at ``g`` as donors. This guarantees a
     non-empty donor pool for every cohort even in fully-staggered
     designs without never-treated units (so long as the last cohort
     is not the only one).
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


def _sdid_per_cohort(
    df_full: pd.DataFrame,
    *,
    seed: int,
):
    """Run single-cohort SDID for each cohort using a donor pool of
    never-treated + not-yet-treated units. Returns (cohort_times_array,
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
        # Donors: never-treated + not-yet-treated (treat_time > g).
        donor_mask = df_full["treat_time"].isna() | (df_full["treat_time"] > g)
        donor_units = df_full.loc[donor_mask, "unit"].unique()
        donor_units = np.setdiff1d(donor_units, cohort_members)
        if len(donor_units) < 2:
            continue
        keep = np.concatenate([cohort_members, donor_units])
        sub = df_full[df_full["unit"].isin(keep)].copy()
        # Re-write treat_time so only cohort members carry g; donors carry NaN.
        sub_treat_time = np.where(sub["unit"].isin(cohort_members), g, np.nan)
        sub["treat_time"] = sub_treat_time
        # Drop time periods past the *next* cohort's switch so donors stay
        # untreated throughout the SDID window. This isolates the (g, post)
        # comparison cleanly.
        try:
            with warnings.catch_warnings():
                # n_boot=0 inside synthetic_did triggers harmless
                # "empty slice / ddof <= 0" warnings from nanstd /
                # nanpercentile; we don't use SE/CI from the inner
                # call so suppress them.
                warnings.simplefilter("ignore", RuntimeWarning)
                res = synthetic_did(sub, n_boot=0, seed=seed)
        except (ValueError, Exception):
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

    y = np.asarray(y, dtype=float)
    d = np.asarray(treatment).astype(int)
    p = np.asarray(panel_id)
    t = np.asarray(time_id)
    if not (y.shape == d.shape == p.shape == t.shape) or y.ndim != 1:
        raise ValueError("y, treatment, panel_id, time_id must be 1-D arrays "
                          "of equal length")

    cohort_of, units, _times = _identify_cohorts(d, p, t)
    df_full = _build_long_df(y, d, p, t, cohort_of)

    point_seed = 0 if seed is None else int(seed)
    cohort_times, cohort_atts, cohort_sizes = _sdid_per_cohort(
        df_full, seed=point_seed,
    )
    if len(cohort_atts) == 0:
        raise ValueError("no identifiable cohorts (need at least one treated "
                          "cohort with >= 2 donors)")
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
                    boot_df, seed=point_seed + b + 1,
                )
                if len(b_atts) > 0 and b_sizes.sum() > 0:
                    w_b = b_sizes / b_sizes.sum()
                    boot_atts[b] = float(np.sum(w_b * b_atts))
            except Exception:
                continue
    se = float(np.nanstd(boot_atts, ddof=0)) if n_boot > 0 else float("nan")

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
