"""Static state-level classifications used for heterogeneity splits."""
from __future__ import annotations

import pandas as pd

# Right-to-work states as of 2023 (NLRB + state DOLs). Static for this exercise.
_RTW_STATES = {
    "01", "04", "05", "12", "13", "16", "18", "19", "20", "21", "22",
    "27", "28", "29", "31", "35", "37", "38", "40", "45", "46", "47",
    "48", "49", "51", "54", "56",  # 27 states
}

# Tradable NAICS-2 supersectors: Manufacturing (31-33), Mining (21), Agriculture (11)
# QCEW NAICS-3 codes have these as their first 2 digits.
_TRADABLE_NAICS2_PREFIX = {"11", "21", "31", "32", "33"}
_MANUF_NAICS2_PREFIX = {"31", "32", "33"}


def right_to_work_flag() -> pd.DataFrame:
    return pd.DataFrame({
        "state_fips": sorted(_RTW_STATES),
        "right_to_work_state": [1] * len(_RTW_STATES),
    })


def tradable_naics2_prefix() -> set[str]:
    return set(_TRADABLE_NAICS2_PREFIX)


def manuf_naics2_prefix() -> set[str]:
    return set(_MANUF_NAICS2_PREFIX)
