"""MagnitudeHarmonizer: convert all event magnitudes to real 2015 % of GDP."""
from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np
import pandas as pd

from ..types import NarrativeEvent


class MagnitudeHarmonizer:
    """Re-scale event magnitudes to real 2015 % of GDP.

    Parameters
    ----------
    gdp_deflator : optional DataFrame indexed by (country, date) with a
        ``deflator`` column (1.0 = 2015 level). When None, events that
        are already in ``"pct_gdp"`` pass through unchanged (only stub
        events and z-score events are handled).
    stub_magnitude : replacement magnitude for stub events
        (original magnitude == 0). Default 0.05.
    """

    def __init__(
        self,
        *,
        gdp_deflator: pd.DataFrame | None = None,
        stub_magnitude: float = 0.05,
    ) -> None:
        self.gdp_deflator = gdp_deflator
        self.stub_magnitude = stub_magnitude

    def _get_deflator(self, country: str, date: pd.Timestamp) -> float | None:
        if self.gdp_deflator is None:
            return None
        idx = (country.upper(), pd.Timestamp(date))
        if idx in self.gdp_deflator.index:
            return float(self.gdp_deflator.loc[idx, "deflator"])
        # Nearest-quarter fallback.
        try:
            sub = self.gdp_deflator.xs(country.upper(), level="country")
            nearest = sub.index[np.argmin(np.abs(sub.index - date))]
            return float(sub.loc[nearest, "deflator"])
        except (KeyError, ValueError):
            return None

    def _harmonize_one(self, ev: NarrativeEvent) -> NarrativeEvent:
        unit = ev.magnitude_unit

        if unit == "z":
            warnings.warn(
                f"z-score magnitude for {ev.country}/{ev.date} cannot be "
                "converted to real % of GDP — left unchanged.",
                UserWarning, stacklevel=4,
            )
            return ev

        if ev.magnitude == 0.0:
            return replace(ev, magnitude=self.stub_magnitude,
                           magnitude_unit="stub_pct_gdp")

        if unit == "stub_pct_gdp":
            return ev

        if unit == "pct_gdp":
            d = self._get_deflator(ev.country, ev.date)
            if d is not None and d > 0:
                return replace(ev, magnitude=ev.magnitude / d)
            return ev

        # Fallthrough: unknown unit — return unchanged.
        return ev

    def apply(self, events: list[NarrativeEvent]) -> list[NarrativeEvent]:
        """Return harmonized copies. Never mutates input."""
        return [self._harmonize_one(ev) for ev in events]


__all__ = ["MagnitudeHarmonizer"]
