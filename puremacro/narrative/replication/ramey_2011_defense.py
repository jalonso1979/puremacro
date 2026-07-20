"""Replication of Ramey (2011) defense-news quarterly value.

Ramey's identification rests on a multi-quarter lag between defense
*news* (anticipation of future spending) and *contracts* (actual
fiscal flow). This loader emits the news IV — the announcement-side
shock. Implementation-side flow lives in DoD contract series, not
here.

Each event's metadata carries ``is_news_shock=True``. The
``implementation_profile`` is left empty (default) — by design this
loader does NOT attempt to retime news events to their implementation
quarter, since that would erase the very identification Ramey relies on.

Source: Ramey, V. (2011). "Identifying Government Spending Shocks:
It's All in the Timing." QJE 126(1), 1-50.

The full historical CSV is hosted in Ramey's replication archive at
``https://econweb.ucsd.edu/~vramey/research/Ramey_HOM_govspend.zip``.
This loader either (a) reads a local CSV path the user supplies or
(b) attempts a one-shot fetch of the public CSV mirror. If neither
succeeds we raise a clear error pointing at the canonical source.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from ..sources._http import safe_get_bytes
from ..types import NarrativeEvent, NarrativeInstrument


# Mirror used by several replication packages — keeps things working
# even if Ramey's UCSD page changes layout.
_MIRROR = (
    "https://raw.githubusercontent.com/qmd-co/macro-data/main/"
    "ramey_2011_defense_news.csv"
)


def ramey_csv_to_events(df: pd.DataFrame) -> list[NarrativeEvent]:
    """Coerce a Ramey-format CSV into NarrativeEvents.

    Expected columns (case-insensitive): ``date`` (YYYY-Qn or YYYY-MM-DD)
    and ``news_q`` (or ``defense_news_q``).
    """
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date") or cols.get("quarter")
    val_col = cols.get("news_q") or cols.get("defense_news_q") or cols.get("ramey_news")
    if date_col is None or val_col is None:
        raise ValueError(
            "Ramey CSV missing date/value columns; expected 'date' and "
            "'news_q' (or 'defense_news_q', 'ramey_news')."
        )
    out: list[NarrativeEvent] = []
    for _, row in df.iterrows():
        try:
            date = pd.Period(str(row[date_col]), freq="Q").to_timestamp()
        except Exception:
            try:
                date = pd.Timestamp(row[date_col])
            except Exception:
                continue
        v = float(row[val_col])
        if v == 0.0 or pd.isna(v):
            continue
        out.append(NarrativeEvent(
            date=date,
            country="USA",
            magnitude=abs(v),
            magnitude_unit="pct_gdp",
            target="both",
            subtarget="defense",
            sign=int(1 if v > 0 else -1),
            confidence=1.0,
            source_text="Ramey (2011) defense-news quarterly value.",
            source_url="https://econweb.ucsd.edu/~vramey/research/",
            scoring_method="manual",
            metadata={"replication": "ramey_2011", "is_news_shock": True},
        ))
    return out


def load(
    *,
    csv_path: str | Path | None = None,
    target: str = "both",
) -> NarrativeInstrument:
    """Load the Ramey (2011) defense-news instrument.

    Parameters
    ----------
    csv_path : optional local path to the CSV. When None, attempt the
        mirror download.
    target : passed through to :class:`NarrativeInstrument`. Default
        ``"both"`` because defense purchases hit both consumption (DoD
        operating) and investment (procurement).
    """
    if csv_path is not None:
        df = pd.read_csv(csv_path)
    else:
        try:
            raw = safe_get_bytes(_MIRROR)
            df = pd.read_csv(io.BytesIO(raw))
        except Exception as e:
            raise RuntimeError(
                "Could not fetch Ramey 2011 defense-news series. "
                "Download the CSV from "
                "https://econweb.ucsd.edu/~vramey/research/ "
                "and pass its path as csv_path=."
            ) from e
    events = ramey_csv_to_events(df)
    return NarrativeInstrument.from_events(events, target=target,
                                              aggregation="sum")


__all__ = ["load"]
