"""Chain SurpriseFilter → MagnitudeHarmonizer → TargetAuditor."""
from __future__ import annotations

from typing import Any

import pandas as pd

from ..types import NarrativeEvent
from .surprise_filter import SurpriseFilter
from .magnitude_harmonizer import MagnitudeHarmonizer
from .target_auditor import TargetAuditor

_AUDIT_COLUMNS = ["country", "date", "old_target", "new_target", "llm_confidence", "reasoning"]


def run_pipeline(
    events: list[NarrativeEvent],
    *,
    surprise_filter: bool = True,
    harmonize_magnitudes: bool = True,
    audit_targets: bool = True,
    llm_backend: Any | None = None,
    confidence_floor: float = 0.5,
    decay_factor: float = 0.25,
    surprise_confidence_floor: float = 0.1,
) -> tuple[list[NarrativeEvent], dict, pd.DataFrame]:
    """Apply quality filters in order.

    Returns
    -------
    (filtered_events, metadata_dict, audit_df)
    """
    meta: dict = {"n_raw": len(events)}
    audit_rows: list[pd.DataFrame] = []

    if surprise_filter:
        sf = SurpriseFilter(decay_factor=decay_factor, confidence_floor=surprise_confidence_floor)
        events = sf.apply(events)
    meta["n_after_surprise_filter"] = len(events)

    if harmonize_magnitudes:
        mh = MagnitudeHarmonizer()
        events = mh.apply(events)
    meta["n_after_harmonizer"] = len(events)

    n_reclass_inv = 0
    n_reclass_con = 0
    if audit_targets:
        ta = TargetAuditor(backend=llm_backend, confidence_threshold=0.7)
        events, audit_df = ta.apply(events)
        if not audit_df.empty:
            n_reclass_inv = int((audit_df["new_target"] == "investment").sum())
            n_reclass_con = int((audit_df["new_target"] == "consumption").sum())
            audit_rows.append(audit_df)
    meta["n_after_target_auditor"] = len(events)
    meta["n_reclassified_investment"] = n_reclass_inv
    meta["n_reclassified_consumption"] = n_reclass_con

    # Drop events below confidence_floor.
    before_drop = len(events)
    events = [e for e in events if e.confidence >= confidence_floor]
    meta["n_dropped_low_confidence"] = before_drop - len(events)

    combined_audit = (
        pd.concat(audit_rows, ignore_index=True)
        if audit_rows else pd.DataFrame(columns=_AUDIT_COLUMNS)
    )
    return events, meta, combined_audit


__all__ = ["run_pipeline"]
