"""Country / aggregate code canonicalization.

Used everywhere the panel might receive aggregate codes from upstream
fetchers (OECD `EA20`, `EU27`, `OECD`; WUI / GPR `WLD`; etc.) so the
panel only ever contains pure-country observations.
"""
from __future__ import annotations

import pandas as pd

IS_AGGREGATE: frozenset[str] = frozenset({
    "EA", "EA12", "EA17", "EA19", "EA20",
    "EU", "EU27", "EU27_2020", "EU28",
    "OECD", "OECDE", "OECDA",
    "G7", "G20", "BRICS",
    "WLD", "WORLD", "ADV", "EME",
    "LATAM", "ASIA", "AFRICA", "MENA",
})


def is_country(code: object) -> bool:
    """True iff *code* is a length-3 alpha string and not in IS_AGGREGATE."""
    if not isinstance(code, str):
        return False
    return len(code) == 3 and code.isalpha() and code.upper() not in IS_AGGREGATE


def drop_aggregates(df: pd.DataFrame, code_col: str = "code") -> pd.DataFrame:
    """Return a copy of *df* with rows whose *code_col* fails ``is_country`` removed."""
    mask = df[code_col].apply(is_country)
    return df.loc[mask].reset_index(drop=True)
