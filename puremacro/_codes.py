"""Country / aggregate code canonicalization.

Used everywhere the panel might receive aggregate codes from upstream
fetchers (OECD `EA20`, `EU27`, `OECD`; WUI / GPR `WLD`; the World Bank's
79 region, income and lending groups; etc.) so the
panel only ever contains pure-country observations.
"""
from __future__ import annotations

import pandas as pd

#: The 79 aggregate ids of the World Bank country list (``/v2/country``,
#: ``region.id == "NA"``, WDI release of 13 July 2026): regions, income and
#: lending groups, demographic-dividend groups, small states and the world.
#: Three letters each, so a code-length rule cannot catch them; none is an
#: ISO 3166 alpha-3 country code.
WB_AGGREGATES: frozenset[str] = frozenset({
    "AFE", "AFR", "AFW", "ARB", "BEA", "BEC", "BHI", "BLA", "BMN", "BSS",
    "CAA", "CEA", "CEB", "CEU", "CLA", "CME", "CSA", "CSS", "DEA", "DEC",
    "DLA", "DMN", "DNS", "DSA", "DSF", "DSS", "EAP", "EAR", "EAS", "ECA",
    "ECS", "EMU", "EUU", "FCV", "FXS", "HIC", "HPC", "IBB", "IBD", "IBT",
    "IDA", "IDB", "IDX", "INX", "LAC", "LCN", "LDC", "LIC", "LMC", "LMY",
    "LTE", "MDE", "MEA", "MIC", "MNA", "NAC", "NAF", "NRS", "NXS", "OED",
    "OSS", "PRE", "PSS", "PST", "RRS", "SAS", "SSA", "SSF", "SST", "SXZ",
    "TEA", "TEC", "TLA", "TMN", "TSA", "TSS", "UMC", "WLD", "XZN",
})

IS_AGGREGATE: frozenset[str] = frozenset({
    "EA", "EA12", "EA17", "EA19", "EA20",
    "EU", "EU27", "EU27_2020", "EU28",
    "OECD", "OECDE", "OECDA",
    "G7", "G20", "BRICS",
    "WLD", "WORLD", "ADV", "EME",
    "LATAM", "ASIA", "AFRICA", "MENA",
}) | WB_AGGREGATES


def is_country(code: object) -> bool:
    """True iff *code* is a length-3 alpha string and not in IS_AGGREGATE."""
    if not isinstance(code, str):
        return False
    return len(code) == 3 and code.isalpha() and code.upper() not in IS_AGGREGATE


def drop_aggregates(df: pd.DataFrame, code_col: str = "code") -> pd.DataFrame:
    """Return a copy of *df* with rows whose *code_col* fails ``is_country`` removed."""
    mask = df[code_col].apply(is_country)
    return df.loc[mask].reset_index(drop=True)
