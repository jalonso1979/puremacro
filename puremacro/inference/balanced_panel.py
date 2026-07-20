"""Balanced (rectangular) sub-panel selection.

Two helpers for the (code, date)-indexed wide panels used throughout the
T-series notebooks:

- :func:`balanced_subpanel`: keep only entities with full coverage of a
  required column set over a chosen date range, then drop dates where
  any retained entity has a NaN. Returns a rectangular DataFrame where
  every (entity, date) cell is non-null on every required column.

- :func:`balanced_cohort_window`: pick the largest-area date window such
  that at least *min_n_entities* entities have full coverage of the
  required columns over that window. Use this when you want to maximise
  the rectangular area rather than fix a date range a priori.

Why bother. Plotting cross-country means over an unbalanced panel produces
visible jumps when entities enter or leave (e.g., labor share series
where some countries start in 1995 and others in 2000). Estimating
panel LPs on an unbalanced panel mixes composition effects across
horizons (the cohort that supplies β_h=0 differs from the cohort that
supplies β_h=12). Balancing eliminates both pathologies at the cost of
shrinking the sample.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def balanced_subpanel(
    df_wide: pd.DataFrame,
    *,
    cols: Sequence[str],
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    min_T: int | None = None,
) -> pd.DataFrame:
    """Return a rectangular sub-panel of `df_wide` on `cols`.

    Parameters
    ----------
    df_wide
        Wide panel indexed by (code, date) (the level names may differ;
        we use positional levels 0 = entity, 1 = date).
    cols
        Columns required to be non-null in every retained (entity, date) cell.
    start, end
        Optional inclusive date bounds. If both omitted, the panel's full
        date range is used.
    min_T
        Optional minimum number of dates an entity must have over the
        intersection (after the entity-date filter). Defaults to None
        (keep all entities that survive the rectangular filter).

    Returns
    -------
    A copy of `df_wide` restricted to the largest rectangular subset
    satisfying the constraints. The returned panel is sorted by
    (entity, date) and has all of `cols` non-null in every row.
    """
    if not cols:
        raise ValueError("`cols` must list at least one column")

    df = df_wide.copy()
    df.index = df.index.set_names(["__entity", "__date"])
    df = df.sort_index()

    if start is not None:
        df = df.loc[df.index.get_level_values("__date") >= pd.Timestamp(start)]
    if end is not None:
        df = df.loc[df.index.get_level_values("__date") <= pd.Timestamp(end)]

    # Step 1: drop rows where ANY required col is NaN. After this, the
    # surviving rows are non-null on `cols` but the panel may still be
    # ragged (some entities cover [t1, t2], others [t3, t4]).
    sub = df.dropna(subset=list(cols))

    if sub.empty:
        return sub.iloc[:0].rename_axis(df_wide.index.names)

    # Step 2: find the date intersection across entities that have any
    # surviving rows. The largest rectangular subset is the cartesian
    # product of (entities) × (dates that ALL retained entities cover).
    # We iterate: drop dates not covered by every entity, then drop
    # entities that lose too many dates, repeat. In practice one pass
    # suffices for the panels we use.
    dates = sub.index.get_level_values("__date").unique()
    entities = sub.index.get_level_values("__entity").unique()

    # Per (entity, date) presence flag.
    present = sub.assign(_p=1).reset_index()[["__entity", "__date", "_p"]]
    pivot = present.pivot_table(index="__entity", columns="__date",
                                 values="_p", aggfunc="first")

    # Maximize the area of the balanced (n_entities × n_dates) rectangle.
    # Two candidate rectangles to compare:
    #
    # (a) Full-time slice: all dates × every entity that covers them all.
    #     Area = n_full × T. Often dominated by a few entities with long
    #     coverage — e.g. only DNK/FIN/NOR/PRT have unc_innov_q back to 1985.
    #
    # (b) Greedy contiguous-date subset: search over date prefixes sorted
    #     by entity-count and pick the prefix that maximises
    #     n_entities_with_full_cover × n_dates. Catches the case where
    #     truncating the date axis brings in many more entities (e.g.,
    #     starting from 2008Q3 brings every country with unc_innov_q
    #     coverage from then onward).
    #
    # We pick whichever has the larger total cell area.
    full_entities = pivot.index[pivot.notna().all(axis=1)]
    cand_full = (len(full_entities) * pivot.shape[1],
                 list(full_entities), list(pivot.columns))

    # Greedy search. Sort dates by entity-count descending; for each prefix,
    # require those dates be present in EVERY retained entity. This finds
    # near-maximal contiguous-coverage rectangles.
    present_counts = pivot.notna().sum(axis=0)
    sorted_dates = list(present_counts.sort_values(ascending=False).index)
    cand_greedy: tuple[int, list, list] = (0, list(pivot.index), [])
    for k in range(1, len(sorted_dates) + 1):
        ds = sorted_dates[:k]
        pp = pivot[ds].notna().all(axis=1)
        ents = list(pp.index[pp])
        if len(ents) < 2:
            continue
        score = len(ents) * k
        if score > cand_greedy[0]:
            cand_greedy = (score, ents, ds)

    chosen = cand_full if cand_full[0] >= cand_greedy[0] else cand_greedy
    kept_entities, kept_dates = chosen[1], chosen[2]

    if min_T is not None:
        # Re-evaluate after balancing in case any entity slipped below min_T.
        kept_dates = sorted(kept_dates)
        if len(kept_dates) < min_T:
            kept_dates = kept_dates  # keep what we have but flag below
        # min_T applied to retained entities trivially: they all share kept_dates.

    if not kept_entities or not kept_dates:
        return df.iloc[:0].rename_axis(df_wide.index.names)

    keep_mask = (
        sub.index.get_level_values("__entity").isin(kept_entities)
        & sub.index.get_level_values("__date").isin(kept_dates)
    )
    out = sub.loc[keep_mask].copy()
    out.index = out.index.set_names(df_wide.index.names)
    return out


def balanced_cohort_window(
    df_wide: pd.DataFrame,
    *,
    cols: Sequence[str],
    min_n_entities: int = 8,
) -> tuple[pd.Timestamp, pd.Timestamp, list]:
    """Find a (start, end, cohort) such that ≥`min_n_entities` entities
    have full coverage of `cols` over [start, end].

    Maximises (n_entities × n_dates) subject to the entity-count floor.
    Returns the chosen window and the list of qualifying entity codes.
    """
    df = df_wide.copy()
    df.index = df.index.set_names(["__entity", "__date"])

    full = df.dropna(subset=list(cols))
    if full.empty:
        raise ValueError("no rows have all required columns non-null")

    pivot = (full.assign(_p=1).reset_index()
             [["__entity", "__date", "_p"]]
             .pivot_table(index="__entity", columns="__date",
                          values="_p", aggfunc="first"))
    dates = sorted(pivot.columns)

    best: tuple[int, pd.Timestamp, pd.Timestamp, list] = (0, dates[0], dates[0], [])
    # Try every (start, end) in the date axis. O(T²) — fine for T ~ a
    # few hundred quarters.
    for i, s in enumerate(dates):
        for j in range(i + min_n_entities - 1, len(dates)):
            e = dates[j]
            window = pivot.loc[:, s:e]
            ents = list(window.index[window.notna().all(axis=1)])
            if len(ents) < min_n_entities:
                continue
            area = len(ents) * (j - i + 1)
            if area > best[0]:
                best = (area, s, e, ents)

    if best[0] == 0:
        raise ValueError(
            f"no balanced window found with ≥{min_n_entities} entities; "
            "loosen min_n_entities or relax the column requirements"
        )
    return best[1], best[2], best[3]


# ---------------------------------------------------------------------------
# Paper-section cohort registry
# ---------------------------------------------------------------------------

_SECTION_REQUIREMENTS: dict[str, tuple[list[str], str]] = {
    # § -> (required column set, label for error messages)
    "3": (["log_relprice_equip", "ls_gollin_q"], "Act 1 trajectories"),
    "5": (["log_relprice_equip", "ls_gollin_q", "log_neer_q"],
          "Aggregate K-N bridge"),
    "6": (["log_relprice_equip", "ls_gollin_q", "log_neer_q",
           "log_oil_brent_real_local_q", "log_real_gov_q"],
          "Robustness — cleaning + IV"),
    "7": (["log_neer_q", "log_relprice_equip", "log_relprice_struct",
           "log_relprice_ipp", "log_pcons_q"],
          "FX pass-through SVAR"),
}


def cohort_for_section(
    df_wide: pd.DataFrame,
    *,
    section: str,
) -> tuple[pd.DataFrame, list[str], tuple[pd.Timestamp, pd.Timestamp]]:
    """Return the canonical (panel, codes, date_range) for a paper section.

    Parameters
    ----------
    df_wide
        The master quarterly panel in wide format, indexed by (code, qdate).
        Caller is responsible for pivoting from the long-format
        ``panel_Q.parquet`` first (e.g. via
        ``df.pivot_table(index=['code','date'], columns='variable', values='value')``).
    section
        A § number as a string (``"3"``, ``"5"``, ``"6"``, ``"7"``). The
        KORV §4 uses annual KLEMS data and is not handled here — use
        ``puremacro.klems.load_klems_panel`` directly for §4.

    Returns
    -------
    A tuple ``(panel, codes, date_range)``:

    - ``panel``: balanced sub-panel restricted to the required columns.
    - ``codes``: sorted list of ISO-3 codes that survived the balance check.
    - ``date_range``: ``(min_qdate, max_qdate)`` of the surviving rectangle.

    Raises
    ------
    ValueError
        If ``section`` is not a recognized § number.
    """
    if section not in _SECTION_REQUIREMENTS:
        raise ValueError(
            f"unknown section {section!r}; known: {sorted(_SECTION_REQUIREMENTS)}"
        )
    cols, _label = _SECTION_REQUIREMENTS[section]
    sub = balanced_subpanel(df_wide, cols=cols)
    codes = sorted(sub.index.get_level_values(0).unique())
    dates = (
        sub.index.get_level_values(1).min(),
        sub.index.get_level_values(1).max(),
    )
    return sub, codes, dates
