"""TargetAuditor: LLM-assisted reclassification of target="both" events."""
from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from ..types import NarrativeEvent


_AUDIT_COLUMNS = ["country", "date", "old_target", "new_target", "llm_confidence", "reasoning"]


@runtime_checkable
class _TargetBackend(Protocol):
    def classify_target(self, source_text: str) -> dict[str, Any]: ...


class TargetAuditor:
    """Re-classify ``target="both"`` events using an LLM backend.

    Parameters
    ----------
    backend : object with a ``classify_target(source_text) → dict``
        method. The dict must have keys ``target``, ``confidence``,
        ``reasoning``. Pass ``None`` to disable (no-op).
    confidence_threshold : minimum LLM confidence to accept the
        reclassification. Default 0.7.
    """

    def __init__(
        self,
        *,
        backend: Any | None = None,
        confidence_threshold: float = 0.7,
    ) -> None:
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be in [0, 1], got {confidence_threshold!r}"
            )
        self.backend = backend
        self.confidence_threshold = confidence_threshold

    def apply(
        self,
        events: list[NarrativeEvent],
    ) -> tuple[list[NarrativeEvent], pd.DataFrame]:
        """Reclassify events; return (updated_events, audit_df).

        ``audit_df`` has one row per reclassified event with columns:
        country, date, old_target, new_target, llm_confidence, reasoning.
        Never mutates input events.
        """
        if self.backend is None:
            return list(events), pd.DataFrame(columns=_AUDIT_COLUMNS)

        out: list[NarrativeEvent] = []
        audit_rows: list[dict] = []

        for ev in events:
            if ev.target != "both":
                out.append(ev)
                continue
            try:
                resp = self.backend.classify_target(ev.source_text)
            except Exception as exc:
                warnings.warn(
                    f"TargetAuditor backend failed for {ev.country}/{ev.date}: {exc!r}",
                    RuntimeWarning, stacklevel=2,
                )
                out.append(ev)
                continue

            new_target = str(resp.get("target", "both"))
            conf       = float(resp.get("confidence", 0.0))
            reasoning  = str(resp.get("reasoning", ""))

            if new_target in ("investment", "consumption") and conf >= self.confidence_threshold:
                ev = replace(
                    ev,
                    target=new_target,
                    metadata={**ev.metadata, "target_auditor_confidence": conf,
                               "target_auditor_reasoning": reasoning},
                )
                audit_rows.append({
                    "country":        ev.country,
                    "date":           ev.date,
                    "old_target":     "both",
                    "new_target":     new_target,
                    "llm_confidence": conf,
                    "reasoning":      reasoning,
                })
            out.append(ev)

        audit_df = pd.DataFrame(audit_rows) if audit_rows else pd.DataFrame(columns=_AUDIT_COLUMNS)
        return out, audit_df


__all__ = ["TargetAuditor"]
