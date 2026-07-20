"""SurpriseFilter: down-weight anticipated fiscal events."""
from __future__ import annotations

from dataclasses import replace

from ..types import NarrativeEvent

# Budget announcement month by ISO3 country.
BUDGET_MONTHS: dict[str, int] = {
    "AUS": 5, "AUT": 10, "BEL": 10, "BRA": 8, "CAN": 3, "CHL": 9,
    "COL": 7, "DEU": 11, "DNK": 8, "ESP": 9, "FIN": 9, "FRA": 9,
    "GBR": 10, "IDN": 8, "IND": 2, "IRL": 10, "ITA": 10, "JPN": 12,
    "KOR": 8, "MEX": 9, "NLD": 9, "NZL": 5, "POL": 9, "PRT": 10,
    "SWE": 9, "THA": 8, "TUR": 10, "USA": 2, "ZAF": 2,
}

_BUDGET_SUBTARGETS = {"transfers", "wages", "expenditure", "general"}


class SurpriseFilter:
    """Decay confidence of anticipated fiscal events.

    Parameters
    ----------
    decay_factor : multiplied into ``confidence`` when the anticipation
        heuristic fires. Default 0.25.
    confidence_floor : events with ``confidence < confidence_floor``
        after decay are hard-dropped. Default 0.1.
    """

    def __init__(
        self,
        *,
        decay_factor: float = 0.25,
        confidence_floor: float = 0.1,
    ) -> None:
        if not (0.0 < decay_factor <= 1.0):
            raise ValueError(f"decay_factor must be in (0, 1], got {decay_factor!r}")
        if not (0.0 <= confidence_floor < 1.0):
            raise ValueError(f"confidence_floor must be in [0, 1), got {confidence_floor!r}")
        self.decay_factor = decay_factor
        self.confidence_floor = confidence_floor

    def _is_anticipated(self, event: NarrativeEvent) -> bool:
        if event.metadata.get("unanticipated"):
            return False
        # Profile-driven: any non-trivial schedule signals anticipation.
        if len(event.implementation_profile) > 1:
            return True
        if event.delay_quarters > 0:
            return True
        # Legacy budget-calendar fallback for events with no profile data.
        budget_month = BUDGET_MONTHS.get(event.country.upper())
        if budget_month is None:
            return False
        in_budget_month = (event.date.month == budget_month)
        in_budget_subtarget = (event.subtarget or "").lower() in _BUDGET_SUBTARGETS
        return in_budget_month and in_budget_subtarget

    def apply(self, events: list[NarrativeEvent]) -> list[NarrativeEvent]:
        """Return filtered/decayed copy of events. Never mutates input."""
        out: list[NarrativeEvent] = []
        for ev in events:
            if self._is_anticipated(ev):
                new_conf = ev.confidence * self.decay_factor
                meta = {**ev.metadata, "anticipation_flag": True}
                ev = replace(ev, confidence=new_conf, metadata=meta)
            if ev.confidence < self.confidence_floor:
                continue
            out.append(ev)
        return out


__all__ = ["SurpriseFilter", "BUDGET_MONTHS"]
