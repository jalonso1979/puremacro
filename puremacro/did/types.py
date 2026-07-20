"""Common helpers for the DiD estimators.

The estimators in this module all consume a long-format panel with
columns ``(unit, time, outcome)`` plus a per-unit first-treatment time
``treat_time`` (``np.nan`` ⇒ never treated). This file centralises the
ingestion / validation logic so each estimator stays focused on its
specific aggregation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class PanelDiD:
    """Long-format panel with treatment-cohort metadata.

    Attributes
    ----------
    df : DataFrame
        Long format with one row per (unit, time).
    unit, time, outcome : column names.
    treat_time : column name holding the first treatment period for
        each unit (constant within a unit). ``NaN`` denotes never-
        treated controls.
    """
    df: pd.DataFrame
    unit: str
    time: str
    outcome: str
    treat_time: str

    def __post_init__(self) -> None:
        for c in (self.unit, self.time, self.outcome, self.treat_time):
            if c not in self.df.columns:
                raise ValueError(f"column {c!r} not in df")
        # Sort & sanity check.
        self.df = self.df.sort_values([self.unit, self.time]).reset_index(drop=True)

    @property
    def units(self) -> np.ndarray:
        """Unique unit identifiers (in first-seen order)."""
        return self.df[self.unit].unique()

    @property
    def times(self) -> np.ndarray:
        """Distinct time periods, sorted ascending."""
        return np.sort(self.df[self.time].unique())

    @property
    def cohorts(self) -> np.ndarray:
        """Distinct first-treatment times among ever-treated units."""
        s = self.df.groupby(self.unit)[self.treat_time].first()
        return np.sort(s.dropna().unique())

    def cohort_of(self) -> pd.Series:
        """Per-unit first-treatment time (``NaN`` for never-treated)."""
        return self.df.groupby(self.unit)[self.treat_time].first()

    def never_treated_mask(self) -> pd.Series:
        """Boolean mask (indexed by unit) marking never-treated units."""
        return self.cohort_of().isna()

    def wide_outcome(self) -> pd.DataFrame:
        """Pivot the outcome into a ``(unit, time)`` matrix.

        Returns
        -------
        pandas.DataFrame
            Rows are units, columns are time periods, values are the
            outcome. Useful for estimators that operate matrix-style.
        """
        return self.df.pivot(index=self.unit, columns=self.time,
                              values=self.outcome)


__all__ = ["PanelDiD"]
