"""GapFiller: identify zero-event runs in a HomogeneousFiscalPanel."""
from __future__ import annotations

import pandas as pd


def find_gaps(
    panel,
    *,
    max_gap_q: int = 4,
    target: str | None = None,
) -> pd.DataFrame:
    """Find consecutive zero-event quarters longer than ``max_gap_q``.

    Parameters
    ----------
    panel : HomogeneousFiscalPanel
    max_gap_q : flag runs of this many or more consecutive zero quarters.
    target : optional target filter (``"investment"`` / ``"consumption"``).
        ``None`` means use all events regardless of target.

    Returns
    -------
    DataFrame with columns: country, gap_start, gap_end, n_quarters.
    Ordered by n_quarters descending (longest gaps first).
    """
    rows: list[dict] = []

    for country in panel.countries:
        evs = [e for e in panel.events if e.country == country]
        if target is not None:
            evs = [e for e in evs if e.target == target or e.target == "both"]
        if not evs:
            continue

        dates = sorted({e.date for e in evs})
        if len(dates) < 2:
            continue

        start = pd.Period(dates[0], freq="Q")
        end   = pd.Period(dates[-1], freq="Q")
        full_range = pd.period_range(start, end, freq="Q")

        event_quarters = {pd.Period(d, freq="Q") for d in dates}
        zero_mask = pd.Series(
            [0 if q in event_quarters else 1 for q in full_range],
            index=full_range,
        )

        # Find runs of consecutive zeros.
        in_gap = False
        gap_start_q: pd.Period | None = None
        run = 0
        for q, is_zero in zero_mask.items():
            if is_zero:
                if not in_gap:
                    gap_start_q = q
                    in_gap = True
                run += 1
            else:
                if in_gap and run >= max_gap_q:
                    assert gap_start_q is not None
                    rows.append({
                        "country":    country,
                        "gap_start":  gap_start_q.to_timestamp(),
                        "gap_end":    (q - 1).to_timestamp(),
                        "n_quarters": run,
                    })
                in_gap = False
                run = 0
        if in_gap and run >= max_gap_q:
            assert gap_start_q is not None
            rows.append({
                "country":    country,
                "gap_start":  gap_start_q.to_timestamp(),
                "gap_end":    full_range[-1].to_timestamp(),
                "n_quarters": run,
            })

    if not rows:
        return pd.DataFrame(columns=["country", "gap_start", "gap_end", "n_quarters"])

    df = pd.DataFrame(rows).sort_values("n_quarters", ascending=False).reset_index(drop=True)
    return df


__all__ = ["find_gaps"]
