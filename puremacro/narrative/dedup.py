"""Fuzzy deduplication of narrative events across sources.

Two events refer to the same fiscal action if they (a) share country
+ target + same direction (sign), (b) fall within a date window
(typically a quarter), and (c) have magnitudes within a ratio bound.

The deduplicator returns clusters, each containing one or more events.
A representative event per cluster can be chosen by source priority
(canonical replication > national press > LLM-scored news).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd

from .types import NarrativeEvent


# Default precedence: canonical-series replications outrank LLM and
# keyword scorers because they have been hand-validated by published
# authors. Within each tier the more recent / specific dataset wins.
_DEFAULT_PRIORITY = {
    "manual": 0,        # canonical replications (DGLP, RR, MR, …)
    "llm":    1,
    "keyword": 2,
}


def cluster_events(
    events: Iterable[NarrativeEvent],
    *,
    date_tol_days: int = 90,
    magnitude_ratio_tol: float = 4.0,
    require_same_target: bool = True,
    require_same_sign: bool = True,
) -> list[list[NarrativeEvent]]:
    """Group events that plausibly refer to the same fiscal action.

    Parameters
    ----------
    events : iterable of ``NarrativeEvent``.
    date_tol_days : two events ≤ this many days apart are candidates
        for the same cluster (default 90 ≈ one quarter).
    magnitude_ratio_tol : if both events have non-zero magnitude, the
        ratio max/min must be ≤ this; default 4 lets a $50bn estimate
        cluster with a $200bn estimate but not a $1tn one.
    require_same_target : if True (default), only collapse events with
        identical ``target``.
    require_same_sign : if True (default), only collapse events with
        the same sign (rejects an LLM hallucination's flip).

    Returns
    -------
    list of clusters; each cluster is a list of events (size ≥ 1).
    Singletons (one-event "clusters") are included.
    """
    events_list = list(events)
    if not events_list:
        return []
    by_country: dict[str, list[NarrativeEvent]] = defaultdict(list)
    for ev in events_list:
        by_country[ev.country].append(ev)

    clusters: list[list[NarrativeEvent]] = []
    for country_events in by_country.values():
        country_events.sort(key=lambda e: e.date)
        local_clusters: list[list[NarrativeEvent]] = []
        # `ev_quarters` does not depend on the cluster, and the representative's
        # quarter-set is recomputed for the same event object on every pass, so
        # both are hoisted/memoised. Rebuilding `pd.Period` sets inside the
        # inner loop made clustering quadratic in Period construction.
        quarters_cache: dict[int, set] = {}

        def _quarters(e) -> set:
            key = id(e)
            got = quarters_cache.get(key)
            if got is None:
                got = {pd.Period(d, freq="Q") for d, _ in e.effective_profile}
                quarters_cache[key] = got
            return got

        for ev in country_events:
            placed = False
            ev_quarters = _quarters(ev)
            for cluster in local_clusters:
                rep = cluster[-1]
                # Date window OR profile-quarter set intersection.
                # Cross-convention duplicates (e.g., DGLP tranche-year stamp
                # vs RR2017 quarterly-announcement for the same fiscal
                # episode) merge via profile overlap even when announcement
                # dates are >90 days apart. For events with empty profiles,
                # effective_profile falls back to [(date, 1.0)], so the
                # quarter-set has one element and overlap reduces to "same
                # announcement quarter" — a strictly tighter test than the
                # date window.
                ann_close = abs((rep.date - ev.date).days) <= date_tol_days
                # Short-circuit: the profile-overlap test is only reached when
                # the cheap date-window test has already failed.
                if not ann_close and not (_quarters(rep) & ev_quarters):
                    continue
                if require_same_target and rep.target != ev.target:
                    continue
                if require_same_sign and rep.sign != ev.sign:
                    continue
                if rep.magnitude > 0 and ev.magnitude > 0:
                    ratio = max(rep.magnitude, ev.magnitude) / min(
                        rep.magnitude, ev.magnitude
                    )
                    if ratio > magnitude_ratio_tol:
                        continue
                cluster.append(ev)
                placed = True
                break
            if not placed:
                local_clusters.append([ev])
        clusters.extend(local_clusters)
    return clusters


def representative(
    cluster: list[NarrativeEvent],
    *,
    priority: dict[str, int] | None = None,
) -> NarrativeEvent:
    """Pick one event per cluster.

    Priority rule: minimise (priority[scoring_method], -confidence).
    Ties broken by `magnitude` (largest wins) and `date` (earliest wins).
    """
    p = priority or _DEFAULT_PRIORITY
    return min(
        cluster,
        key=lambda e: (
            p.get(e.scoring_method, 99),
            -float(e.confidence),
            -float(e.magnitude),
            e.date,
        ),
    )


def deduplicate(
    events: Iterable[NarrativeEvent],
    *,
    date_tol_days: int = 90,
    magnitude_ratio_tol: float = 4.0,
    priority: dict[str, int] | None = None,
) -> tuple[list[NarrativeEvent], pd.DataFrame]:
    """Cluster events and keep one representative per cluster.

    Returns
    -------
    (representatives, audit_df)
        ``representatives`` is the deduplicated event list.
        ``audit_df`` is a DataFrame with one row per *non-representative*
        duplicate, joining it to its representative event for review.
    """
    clusters = cluster_events(
        events,
        date_tol_days=date_tol_days,
        magnitude_ratio_tol=magnitude_ratio_tol,
    )
    reps: list[NarrativeEvent] = []
    audit_rows: list[dict] = []
    for c in clusters:
        rep = representative(c, priority=priority)
        reps.append(rep)
        for ev in c:
            if ev is rep:
                continue
            audit_rows.append({
                "rep_country": rep.country,
                "rep_date": rep.date,
                "rep_method": rep.scoring_method,
                "rep_magnitude": rep.magnitude,
                "rep_sign": rep.sign,
                "dup_date": ev.date,
                "dup_method": ev.scoring_method,
                "dup_magnitude": ev.magnitude,
                "dup_sign": ev.sign,
                "dup_url": ev.source_url,
            })
    return reps, pd.DataFrame(audit_rows)


__all__ = ["cluster_events", "representative", "deduplicate"]
