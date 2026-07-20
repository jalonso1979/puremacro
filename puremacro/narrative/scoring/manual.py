"""Helpers for hand-coded narrative events.

The CSV schema mirrors ``NarrativeEvent`` fields. Use ``load_manual_csv``
to materialise the file as a list of validated events; use
``validate_events`` for sanity-checking events from any source.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..types import NarrativeEvent, VALID_SIGNS, VALID_TARGETS


REQUIRED_COLS = (
    "date", "country", "magnitude", "magnitude_unit",
    "target", "sign", "confidence", "source_url",
)


def load_manual_csv(path: str | Path) -> list[NarrativeEvent]:
    """Read a CSV and return validated ``NarrativeEvent`` objects.

    The CSV must contain at minimum the columns in ``REQUIRED_COLS``;
    optional columns ``subtarget``, ``source_text``, ``scoring_method``
    are honoured when present.
    """
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    out = []
    for _, row in df.iterrows():
        out.append(NarrativeEvent(
            date=pd.Timestamp(row["date"]),
            country=str(row["country"]),
            magnitude=float(row["magnitude"]),
            magnitude_unit=str(row["magnitude_unit"]),
            target=str(row["target"]),
            subtarget=row.get("subtarget") if "subtarget" in row else None,
            sign=int(row["sign"]),
            confidence=float(row["confidence"]),
            source_text=str(row.get("source_text", "")),
            source_url=str(row["source_url"]),
            scoring_method=str(row.get("scoring_method", "manual")),
        ))
    return out


def validate_events(events: list[NarrativeEvent]) -> dict:
    """Sanity check: distribution of signs/targets, sorted-by-date,
    duplicate detection."""
    if not events:
        return {"n": 0}
    df = pd.DataFrame([e.to_dict() for e in events])
    out = {
        "n": len(events),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "by_target": df["target"].value_counts().to_dict(),
        "by_sign": df["sign"].value_counts().to_dict(),
        "by_country": df["country"].value_counts().to_dict(),
        "by_method": df["scoring_method"].value_counts().to_dict(),
        "n_duplicates": int(df.duplicated(
            subset=["date", "country", "target", "magnitude"],
        ).sum()),
        "issues": [],
    }
    bad_targets = set(df["target"]) - VALID_TARGETS
    if bad_targets:
        out["issues"].append(f"unknown targets: {bad_targets}")
    bad_signs = set(df["sign"]) - VALID_SIGNS
    if bad_signs:
        out["issues"].append(f"unknown signs: {bad_signs}")
    return out


__all__ = ["load_manual_csv", "validate_events", "REQUIRED_COLS"]
