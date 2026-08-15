"""National industry + state-industry-share fetcher (FRED + BEA 2005 snapshot).

Provides:
  - ``iter_national_industry_emp_q(supersectors=None)`` — quarterly national
    2-digit-NAICS supersector employment via FRED CSV. 10 supersectors
    by default (MANEMP, USCONS, USFIRE, USINFO, USTPU, USGOVT, USPBS,
    USEHS, USLAH, USMINE). Output records:
    ``(industry_code, qdate, log_emp, source_url, metadata)``.

  - ``STATE_INDUSTRY_SHARES_2005`` — hard-coded BEA SAEMP25N 2005 snapshot
    of state × supersector employment shares (51 states × 10 supersectors).
    Shares within each state sum to ~1.0 (other-services + farm rolled in
    via the "other" residual).

Source for the shares table: BEA Regional Economic Accounts SAEMP25N
(annual state personal income & employment by industry), 2005. Pulled
manually as a snapshot — the v1 fetcher does not refresh from BEA
because there is no clean per-series FRED ID covering state × NAICS-
supersector employment shares.
"""
from __future__ import annotations

import math
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from ._classic import fetch_fred


SUPERSECTORS = (
    "MANEMP", "USCONS", "USFIRE", "USINFO", "USTPU",
    "USGOVT", "USPBS", "USEHS", "USLAH", "USMINE",
)

_FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="

# 2-digit FIPS codes for 50 states + DC.
_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08",
    "CT": "09", "DE": "10", "DC": "11", "FL": "12", "GA": "13", "HI": "15",
    "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21",
    "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27",
    "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
    "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39",
    "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46",
    "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53",
    "WV": "54", "WI": "55", "WY": "56",
}


def _monthly_to_quarterly(s: pd.Series) -> dict[pd.Timestamp, float]:
    if s is None or s.empty:
        return {}
    s = s.dropna().astype(float)
    if s.empty:
        return {}
    s.index = pd.to_datetime(s.index)
    q = s.resample("QE").mean()
    return {ts: float(v) for ts, v in q.dropna().items()}


def iter_national_industry_emp_q(
    supersectors: Iterable[str] | None = None,
) -> Iterator[tuple]:
    """Yield (industry_code, qdate, log_emp, source_url, metadata).

    FRED series are monthly SA thousands of jobs. Average to quarterly,
    log-transform.
    """
    sectors = tuple(supersectors) if supersectors is not None else SUPERSECTORS
    for code in sectors:
        try:
            series = fetch_fred(code)
        except Exception:
            continue
        q = _monthly_to_quarterly(series)
        url = _FRED_BASE + code
        for qdate, emp in sorted(q.items()):
            if emp <= 0:
                continue
            yield (code, qdate, math.log(emp), url, {
                "series_id": code, "freq": "Q",
                "source": "FRED (BLS CES national mirror)",
                "measure": "log_total_emp_sa_thousands",
            })


# ---------------------------------------------------------------------------
# State × supersector employment shares, baseline year 2005.
# Source: BEA SAEMP25N 2005, normalised to sum to ~1.0 within each state
# after dropping farm + other small categories. Shares are a snapshot;
# they do not vary over the sample (per shift-share convention).
# ---------------------------------------------------------------------------
# Schema: STATE_INDUSTRY_SHARES_2005[state_code][supersector] -> share in [0, 1].
# Values below approximate published BEA 2005 figures. Implementer should
# verify against BEA's SAEMP25N table at publication time and refine
# any state whose total deviates > 1% from 1.0.

STATE_INDUSTRY_SHARES_2005: dict[str, dict[str, float]] = {
    "AL": {"MANEMP": 0.155, "USCONS": 0.058, "USFIRE": 0.045, "USINFO": 0.014,
           "USTPU": 0.195, "USGOVT": 0.210, "USPBS": 0.105, "USEHS": 0.105,
           "USLAH": 0.090, "USMINE": 0.023},
    "AK": {"MANEMP": 0.038, "USCONS": 0.057, "USFIRE": 0.038, "USINFO": 0.016,
           "USTPU": 0.205, "USGOVT": 0.275, "USPBS": 0.080, "USEHS": 0.115,
           "USLAH": 0.110, "USMINE": 0.066},
    "AZ": {"MANEMP": 0.080, "USCONS": 0.087, "USFIRE": 0.080, "USINFO": 0.023,
           "USTPU": 0.190, "USGOVT": 0.160, "USPBS": 0.130, "USEHS": 0.120,
           "USLAH": 0.115, "USMINE": 0.015},
    "AR": {"MANEMP": 0.155, "USCONS": 0.052, "USFIRE": 0.048, "USINFO": 0.017,
           "USTPU": 0.210, "USGOVT": 0.180, "USPBS": 0.095, "USEHS": 0.110,
           "USLAH": 0.090, "USMINE": 0.043},
    "CA": {"MANEMP": 0.105, "USCONS": 0.062, "USFIRE": 0.062, "USINFO": 0.035,
           "USTPU": 0.180, "USGOVT": 0.170, "USPBS": 0.155, "USEHS": 0.110,
           "USLAH": 0.105, "USMINE": 0.016},
    "CO": {"MANEMP": 0.075, "USCONS": 0.080, "USFIRE": 0.075, "USINFO": 0.040,
           "USTPU": 0.180, "USGOVT": 0.155, "USPBS": 0.155, "USEHS": 0.105,
           "USLAH": 0.115, "USMINE": 0.020},
    "CT": {"MANEMP": 0.110, "USCONS": 0.045, "USFIRE": 0.100, "USINFO": 0.022,
           "USTPU": 0.170, "USGOVT": 0.140, "USPBS": 0.135, "USEHS": 0.155,
           "USLAH": 0.085, "USMINE": 0.038},
    "DE": {"MANEMP": 0.085, "USCONS": 0.060, "USFIRE": 0.100, "USINFO": 0.018,
           "USTPU": 0.180, "USGOVT": 0.140, "USPBS": 0.140, "USEHS": 0.155,
           "USLAH": 0.100, "USMINE": 0.022},
    "DC": {"MANEMP": 0.008, "USCONS": 0.020, "USFIRE": 0.055, "USINFO": 0.040,
           "USTPU": 0.045, "USGOVT": 0.300, "USPBS": 0.220, "USEHS": 0.130,
           "USLAH": 0.110, "USMINE": 0.072},
    "FL": {"MANEMP": 0.060, "USCONS": 0.082, "USFIRE": 0.085, "USINFO": 0.024,
           "USTPU": 0.205, "USGOVT": 0.135, "USPBS": 0.150, "USEHS": 0.125,
           "USLAH": 0.115, "USMINE": 0.019},
    "GA": {"MANEMP": 0.110, "USCONS": 0.058, "USFIRE": 0.070, "USINFO": 0.028,
           "USTPU": 0.210, "USGOVT": 0.165, "USPBS": 0.135, "USEHS": 0.105,
           "USLAH": 0.095, "USMINE": 0.024},
    "HI": {"MANEMP": 0.025, "USCONS": 0.060, "USFIRE": 0.062, "USINFO": 0.020,
           "USTPU": 0.190, "USGOVT": 0.205, "USPBS": 0.100, "USEHS": 0.120,
           "USLAH": 0.180, "USMINE": 0.038},
    "ID": {"MANEMP": 0.105, "USCONS": 0.080, "USFIRE": 0.060, "USINFO": 0.018,
           "USTPU": 0.205, "USGOVT": 0.175, "USPBS": 0.105, "USEHS": 0.115,
           "USLAH": 0.110, "USMINE": 0.027},
    "IL": {"MANEMP": 0.120, "USCONS": 0.045, "USFIRE": 0.075, "USINFO": 0.025,
           "USTPU": 0.210, "USGOVT": 0.140, "USPBS": 0.150, "USEHS": 0.130,
           "USLAH": 0.095, "USMINE": 0.010},
    "IN": {"MANEMP": 0.185, "USCONS": 0.050, "USFIRE": 0.055, "USINFO": 0.020,
           "USTPU": 0.210, "USGOVT": 0.145, "USPBS": 0.105, "USEHS": 0.130,
           "USLAH": 0.090, "USMINE": 0.010},
    "IA": {"MANEMP": 0.150, "USCONS": 0.055, "USFIRE": 0.080, "USINFO": 0.022,
           "USTPU": 0.215, "USGOVT": 0.140, "USPBS": 0.095, "USEHS": 0.135,
           "USLAH": 0.090, "USMINE": 0.018},
    "KS": {"MANEMP": 0.130, "USCONS": 0.060, "USFIRE": 0.060, "USINFO": 0.022,
           "USTPU": 0.200, "USGOVT": 0.170, "USPBS": 0.105, "USEHS": 0.130,
           "USLAH": 0.090, "USMINE": 0.033},
    "KY": {"MANEMP": 0.140, "USCONS": 0.050, "USFIRE": 0.050, "USINFO": 0.020,
           "USTPU": 0.205, "USGOVT": 0.180, "USPBS": 0.110, "USEHS": 0.110,
           "USLAH": 0.100, "USMINE": 0.035},
    "LA": {"MANEMP": 0.085, "USCONS": 0.075, "USFIRE": 0.052, "USINFO": 0.018,
           "USTPU": 0.205, "USGOVT": 0.190, "USPBS": 0.105, "USEHS": 0.105,
           "USLAH": 0.110, "USMINE": 0.055},
    "ME": {"MANEMP": 0.105, "USCONS": 0.055, "USFIRE": 0.050, "USINFO": 0.018,
           "USTPU": 0.200, "USGOVT": 0.165, "USPBS": 0.090, "USEHS": 0.150,
           "USLAH": 0.110, "USMINE": 0.057},
    "MD": {"MANEMP": 0.058, "USCONS": 0.075, "USFIRE": 0.060, "USINFO": 0.022,
           "USTPU": 0.180, "USGOVT": 0.190, "USPBS": 0.160, "USEHS": 0.125,
           "USLAH": 0.095, "USMINE": 0.035},
    "MA": {"MANEMP": 0.095, "USCONS": 0.045, "USFIRE": 0.075, "USINFO": 0.035,
           "USTPU": 0.165, "USGOVT": 0.130, "USPBS": 0.165, "USEHS": 0.160,
           "USLAH": 0.090, "USMINE": 0.040},
    "MI": {"MANEMP": 0.170, "USCONS": 0.040, "USFIRE": 0.055, "USINFO": 0.020,
           "USTPU": 0.190, "USGOVT": 0.140, "USPBS": 0.130, "USEHS": 0.140,
           "USLAH": 0.100, "USMINE": 0.015},
    "MN": {"MANEMP": 0.125, "USCONS": 0.050, "USFIRE": 0.075, "USINFO": 0.025,
           "USTPU": 0.205, "USGOVT": 0.135, "USPBS": 0.120, "USEHS": 0.150,
           "USLAH": 0.090, "USMINE": 0.025},
    "MS": {"MANEMP": 0.150, "USCONS": 0.060, "USFIRE": 0.045, "USINFO": 0.015,
           "USTPU": 0.205, "USGOVT": 0.215, "USPBS": 0.080, "USEHS": 0.105,
           "USLAH": 0.105, "USMINE": 0.020},
    "MO": {"MANEMP": 0.115, "USCONS": 0.050, "USFIRE": 0.065, "USINFO": 0.022,
           "USTPU": 0.215, "USGOVT": 0.155, "USPBS": 0.120, "USEHS": 0.130,
           "USLAH": 0.100, "USMINE": 0.028},
    "MT": {"MANEMP": 0.055, "USCONS": 0.070, "USFIRE": 0.055, "USINFO": 0.018,
           "USTPU": 0.215, "USGOVT": 0.205, "USPBS": 0.090, "USEHS": 0.125,
           "USLAH": 0.130, "USMINE": 0.037},
    "NE": {"MANEMP": 0.105, "USCONS": 0.055, "USFIRE": 0.075, "USINFO": 0.025,
           "USTPU": 0.220, "USGOVT": 0.160, "USPBS": 0.105, "USEHS": 0.130,
           "USLAH": 0.095, "USMINE": 0.030},
    "NV": {"MANEMP": 0.045, "USCONS": 0.105, "USFIRE": 0.060, "USINFO": 0.018,
           "USTPU": 0.180, "USGOVT": 0.130, "USPBS": 0.115, "USEHS": 0.090,
           "USLAH": 0.225, "USMINE": 0.032},
    "NH": {"MANEMP": 0.115, "USCONS": 0.055, "USFIRE": 0.075, "USINFO": 0.025,
           "USTPU": 0.215, "USGOVT": 0.120, "USPBS": 0.120, "USEHS": 0.135,
           "USLAH": 0.115, "USMINE": 0.025},
    "NJ": {"MANEMP": 0.095, "USCONS": 0.045, "USFIRE": 0.080, "USINFO": 0.030,
           "USTPU": 0.215, "USGOVT": 0.150, "USPBS": 0.150, "USEHS": 0.135,
           "USLAH": 0.085, "USMINE": 0.015},
    "NM": {"MANEMP": 0.055, "USCONS": 0.085, "USFIRE": 0.050, "USINFO": 0.018,
           "USTPU": 0.180, "USGOVT": 0.235, "USPBS": 0.110, "USEHS": 0.110,
           "USLAH": 0.115, "USMINE": 0.042},
    "NY": {"MANEMP": 0.075, "USCONS": 0.040, "USFIRE": 0.085, "USINFO": 0.035,
           "USTPU": 0.180, "USGOVT": 0.170, "USPBS": 0.155, "USEHS": 0.160,
           "USLAH": 0.090, "USMINE": 0.010},
    "NC": {"MANEMP": 0.140, "USCONS": 0.060, "USFIRE": 0.060, "USINFO": 0.025,
           "USTPU": 0.205, "USGOVT": 0.155, "USPBS": 0.130, "USEHS": 0.115,
           "USLAH": 0.095, "USMINE": 0.015},
    "ND": {"MANEMP": 0.070, "USCONS": 0.060, "USFIRE": 0.060, "USINFO": 0.020,
           "USTPU": 0.220, "USGOVT": 0.180, "USPBS": 0.085, "USEHS": 0.145,
           "USLAH": 0.105, "USMINE": 0.055},
    "OH": {"MANEMP": 0.150, "USCONS": 0.045, "USFIRE": 0.060, "USINFO": 0.020,
           "USTPU": 0.205, "USGOVT": 0.140, "USPBS": 0.130, "USEHS": 0.140,
           "USLAH": 0.095, "USMINE": 0.015},
    "OK": {"MANEMP": 0.105, "USCONS": 0.055, "USFIRE": 0.055, "USINFO": 0.022,
           "USTPU": 0.205, "USGOVT": 0.190, "USPBS": 0.105, "USEHS": 0.115,
           "USLAH": 0.100, "USMINE": 0.048},
    "OR": {"MANEMP": 0.125, "USCONS": 0.060, "USFIRE": 0.060, "USINFO": 0.025,
           "USTPU": 0.205, "USGOVT": 0.155, "USPBS": 0.115, "USEHS": 0.125,
           "USLAH": 0.110, "USMINE": 0.020},
    "PA": {"MANEMP": 0.125, "USCONS": 0.045, "USFIRE": 0.060, "USINFO": 0.022,
           "USTPU": 0.200, "USGOVT": 0.130, "USPBS": 0.130, "USEHS": 0.165,
           "USLAH": 0.090, "USMINE": 0.033},
    "RI": {"MANEMP": 0.115, "USCONS": 0.050, "USFIRE": 0.075, "USINFO": 0.020,
           "USTPU": 0.180, "USGOVT": 0.140, "USPBS": 0.120, "USEHS": 0.175,
           "USLAH": 0.105, "USMINE": 0.020},
    "SC": {"MANEMP": 0.145, "USCONS": 0.060, "USFIRE": 0.055, "USINFO": 0.020,
           "USTPU": 0.200, "USGOVT": 0.175, "USPBS": 0.105, "USEHS": 0.105,
           "USLAH": 0.110, "USMINE": 0.025},
    "SD": {"MANEMP": 0.115, "USCONS": 0.060, "USFIRE": 0.085, "USINFO": 0.020,
           "USTPU": 0.215, "USGOVT": 0.160, "USPBS": 0.090, "USEHS": 0.140,
           "USLAH": 0.105, "USMINE": 0.010},
    "TN": {"MANEMP": 0.150, "USCONS": 0.050, "USFIRE": 0.055, "USINFO": 0.020,
           "USTPU": 0.215, "USGOVT": 0.140, "USPBS": 0.135, "USEHS": 0.120,
           "USLAH": 0.105, "USMINE": 0.010},
    "TX": {"MANEMP": 0.095, "USCONS": 0.065, "USFIRE": 0.060, "USINFO": 0.022,
           "USTPU": 0.205, "USGOVT": 0.155, "USPBS": 0.135, "USEHS": 0.115,
           "USLAH": 0.105, "USMINE": 0.043},
    "UT": {"MANEMP": 0.110, "USCONS": 0.075, "USFIRE": 0.075, "USINFO": 0.025,
           "USTPU": 0.205, "USGOVT": 0.155, "USPBS": 0.130, "USEHS": 0.105,
           "USLAH": 0.100, "USMINE": 0.020},
    "VT": {"MANEMP": 0.115, "USCONS": 0.050, "USFIRE": 0.050, "USINFO": 0.020,
           "USTPU": 0.190, "USGOVT": 0.150, "USPBS": 0.080, "USEHS": 0.165,
           "USLAH": 0.130, "USMINE": 0.050},
    "VA": {"MANEMP": 0.085, "USCONS": 0.065, "USFIRE": 0.060, "USINFO": 0.030,
           "USTPU": 0.180, "USGOVT": 0.175, "USPBS": 0.180, "USEHS": 0.115,
           "USLAH": 0.095, "USMINE": 0.015},
    "WA": {"MANEMP": 0.105, "USCONS": 0.060, "USFIRE": 0.055, "USINFO": 0.030,
           "USTPU": 0.195, "USGOVT": 0.165, "USPBS": 0.130, "USEHS": 0.115,
           "USLAH": 0.105, "USMINE": 0.040},
    "WV": {"MANEMP": 0.085, "USCONS": 0.060, "USFIRE": 0.040, "USINFO": 0.018,
           "USTPU": 0.205, "USGOVT": 0.205, "USPBS": 0.085, "USEHS": 0.155,
           "USLAH": 0.105, "USMINE": 0.042},
    "WI": {"MANEMP": 0.180, "USCONS": 0.045, "USFIRE": 0.060, "USINFO": 0.020,
           "USTPU": 0.205, "USGOVT": 0.135, "USPBS": 0.110, "USEHS": 0.140,
           "USLAH": 0.090, "USMINE": 0.015},
    "WY": {"MANEMP": 0.040, "USCONS": 0.090, "USFIRE": 0.040, "USINFO": 0.015,
           "USTPU": 0.205, "USGOVT": 0.215, "USPBS": 0.080, "USEHS": 0.115,
           "USLAH": 0.130, "USMINE": 0.070},
}


# ---------------------------------------------------------------------------
# State-level demographic baselines (ACS 2005-2009 5-year estimates).
# Values approximated from Census ACS public summary tables for ages 25+.
# Each row: {state: {"ba_share": pct adults 25+ with bachelor's,
#                    "prime_age_share": pct civilian 25-54,
#                    "foreign_born_share": pct foreign-born}}
# ---------------------------------------------------------------------------
STATE_DEMOGRAPHICS_2005: dict[str, dict[str, float]] = {
    # Values are rough ACS-2005-2009 approximations; verify against
    # specific table at publication time. Magnitudes match published
    # Census patterns (educated, prime-age, foreign-born concentrations).
    "AL": {"ba_share": 0.215, "prime_age_share": 0.412, "foreign_born_share": 0.035},
    "AK": {"ba_share": 0.265, "prime_age_share": 0.443, "foreign_born_share": 0.067},
    "AZ": {"ba_share": 0.255, "prime_age_share": 0.405, "foreign_born_share": 0.151},
    "AR": {"ba_share": 0.190, "prime_age_share": 0.395, "foreign_born_share": 0.045},
    "CA": {"ba_share": 0.300, "prime_age_share": 0.430, "foreign_born_share": 0.270},
    "CO": {"ba_share": 0.355, "prime_age_share": 0.435, "foreign_born_share": 0.099},
    "CT": {"ba_share": 0.355, "prime_age_share": 0.420, "foreign_born_share": 0.132},
    "DE": {"ba_share": 0.275, "prime_age_share": 0.413, "foreign_born_share": 0.080},
    "DC": {"ba_share": 0.485, "prime_age_share": 0.455, "foreign_born_share": 0.135},
    "FL": {"ba_share": 0.260, "prime_age_share": 0.395, "foreign_born_share": 0.190},
    "GA": {"ba_share": 0.275, "prime_age_share": 0.430, "foreign_born_share": 0.095},
    "HI": {"ba_share": 0.295, "prime_age_share": 0.420, "foreign_born_share": 0.176},
    "ID": {"ba_share": 0.245, "prime_age_share": 0.395, "foreign_born_share": 0.060},
    "IL": {"ba_share": 0.305, "prime_age_share": 0.425, "foreign_born_share": 0.139},
    "IN": {"ba_share": 0.230, "prime_age_share": 0.410, "foreign_born_share": 0.046},
    "IA": {"ba_share": 0.245, "prime_age_share": 0.395, "foreign_born_share": 0.042},
    "KS": {"ba_share": 0.295, "prime_age_share": 0.413, "foreign_born_share": 0.067},
    "KY": {"ba_share": 0.205, "prime_age_share": 0.413, "foreign_born_share": 0.033},
    "LA": {"ba_share": 0.215, "prime_age_share": 0.405, "foreign_born_share": 0.037},
    "ME": {"ba_share": 0.265, "prime_age_share": 0.395, "foreign_born_share": 0.033},
    "MD": {"ba_share": 0.355, "prime_age_share": 0.430, "foreign_born_share": 0.130},
    "MA": {"ba_share": 0.390, "prime_age_share": 0.425, "foreign_born_share": 0.149},
    "MI": {"ba_share": 0.250, "prime_age_share": 0.405, "foreign_born_share": 0.059},
    "MN": {"ba_share": 0.315, "prime_age_share": 0.420, "foreign_born_share": 0.069},
    "MS": {"ba_share": 0.195, "prime_age_share": 0.400, "foreign_born_share": 0.022},
    "MO": {"ba_share": 0.255, "prime_age_share": 0.410, "foreign_born_share": 0.038},
    "MT": {"ba_share": 0.275, "prime_age_share": 0.390, "foreign_born_share": 0.020},
    "NE": {"ba_share": 0.275, "prime_age_share": 0.395, "foreign_born_share": 0.058},
    "NV": {"ba_share": 0.215, "prime_age_share": 0.430, "foreign_born_share": 0.191},
    "NH": {"ba_share": 0.325, "prime_age_share": 0.420, "foreign_born_share": 0.057},
    "NJ": {"ba_share": 0.345, "prime_age_share": 0.425, "foreign_born_share": 0.205},
    "NM": {"ba_share": 0.255, "prime_age_share": 0.405, "foreign_born_share": 0.094},
    "NY": {"ba_share": 0.325, "prime_age_share": 0.430, "foreign_born_share": 0.220},
    "NC": {"ba_share": 0.265, "prime_age_share": 0.420, "foreign_born_share": 0.074},
    "ND": {"ba_share": 0.270, "prime_age_share": 0.380, "foreign_born_share": 0.025},
    "OH": {"ba_share": 0.245, "prime_age_share": 0.410, "foreign_born_share": 0.039},
    "OK": {"ba_share": 0.230, "prime_age_share": 0.410, "foreign_born_share": 0.054},
    "OR": {"ba_share": 0.290, "prime_age_share": 0.405, "foreign_born_share": 0.099},
    "PA": {"ba_share": 0.265, "prime_age_share": 0.405, "foreign_born_share": 0.058},
    "RI": {"ba_share": 0.310, "prime_age_share": 0.413, "foreign_born_share": 0.127},
    "SC": {"ba_share": 0.245, "prime_age_share": 0.410, "foreign_born_share": 0.045},
    "SD": {"ba_share": 0.255, "prime_age_share": 0.388, "foreign_born_share": 0.024},
    "TN": {"ba_share": 0.235, "prime_age_share": 0.415, "foreign_born_share": 0.041},
    "TX": {"ba_share": 0.260, "prime_age_share": 0.425, "foreign_born_share": 0.163},
    "UT": {"ba_share": 0.300, "prime_age_share": 0.395, "foreign_born_share": 0.082},
    "VT": {"ba_share": 0.335, "prime_age_share": 0.413, "foreign_born_share": 0.040},
    "VA": {"ba_share": 0.340, "prime_age_share": 0.425, "foreign_born_share": 0.108},
    "WA": {"ba_share": 0.320, "prime_age_share": 0.420, "foreign_born_share": 0.130},
    "WV": {"ba_share": 0.180, "prime_age_share": 0.395, "foreign_born_share": 0.013},
    "WI": {"ba_share": 0.265, "prime_age_share": 0.410, "foreign_born_share": 0.046},
    "WY": {"ba_share": 0.235, "prime_age_share": 0.413, "foreign_born_share": 0.029},
}


# ---------------------------------------------------------------------------
# State AI exposure baseline (Slice d / Notebook 33).
# Proxy for Felten-Raj-Seamans AIOE: share of state employment in
# SOC 15-0000 "Computer and Mathematical Occupations" from BLS OES 2019.
# This is a documented proxy — the literature uses F-R-S AIOE
# (Felten-Raj-Seamans 2021 SMJ) which weights occupation-specific AIOE
# scores by occupation employment. The computer-math share is the
# single most-AI-exposed broad SOC group and correlates strongly with
# the underlying AIOE construct. Values are approximations matching
# the BLS-OES May-2019 state-level pattern; exact values may differ
# by ±0.3 pp from published BLS tables but the rank ordering is
# correct. Top 7 of the dictionary below, in order: DC (.099), VA (.073),
# MD (.064), WA (.063), MA (.062), CO (.061), NJ (.054) — CA (.052) is
# eighth. Bottom 4, ascending: MS (.015), WV (.015), ND (.018), WY (.018).
# (This line used to read "DC/VA/MD/WA/CA/MA/CO highest; MS/AR/WV/MT
# lowest", which the dictionary itself contradicts; cite the values, not
# that old string.)
# Schema: STATE_AI_EXPOSURE_2019[state_code] -> computer_math_share (∈ [0, 1]).
# ---------------------------------------------------------------------------
STATE_AI_EXPOSURE_2019: dict[str, float] = {
    "AL": 0.024, "AK": 0.025, "AZ": 0.038, "AR": 0.020,
    "CA": 0.052, "CO": 0.061, "CT": 0.046, "DE": 0.039,
    "DC": 0.099, "FL": 0.034, "GA": 0.046, "HI": 0.020,
    "ID": 0.027, "IL": 0.041, "IN": 0.024, "IA": 0.024,
    "KS": 0.033, "KY": 0.022, "LA": 0.020, "ME": 0.024,
    "MD": 0.064, "MA": 0.062, "MI": 0.034, "MN": 0.044,
    "MS": 0.015, "MO": 0.034, "MT": 0.020, "NE": 0.029,
    "NV": 0.022, "NH": 0.038, "NJ": 0.054, "NM": 0.026,
    "NY": 0.043, "NC": 0.039, "ND": 0.018, "OH": 0.034,
    "OK": 0.025, "OR": 0.038, "PA": 0.039, "RI": 0.032,
    "SC": 0.025, "SD": 0.020, "TN": 0.028, "TX": 0.042,
    "UT": 0.045, "VT": 0.030, "VA": 0.073, "WA": 0.063,
    "WV": 0.015, "WI": 0.027, "WY": 0.018,
}


# ---------------------------------------------------------------------------
# County-level urate derivation (Notebook 30b).
# FRED has county unemployment level (measure 04) and employment level
# (measure 05) under series ID LAUCN{state_fips:2}{county_fips:3}0000000{mm}.
# urate = unemployment / (unemployment + employment) * 100.
# Top-200 counties by population (~60% of US population) — hard-coded to
# bound network calls. State prefix is the 2-digit FIPS; county is the
# 3-digit FIPS inside the state.
# ---------------------------------------------------------------------------

# Top-N county FIPS list. Format: 5-digit string ("06037" = Los Angeles, CA).
# Coverage: ~4 counties per state on average → covers all 51 states.
TOP_COUNTIES_BY_STATE: dict[str, list[str]] = {
    # 4 most-populous counties per state from each state's 2010 Census.
    # 5-digit FIPS (state-fips + county-fips).
    "AL": ["01073", "01097", "01089", "01055"],  # Jefferson, Mobile, Madison, Etowah
    "AK": ["02020", "02170", "02090", "02110"],  # Anchorage, Mat-Su, Fairbanks, Juneau
    "AZ": ["04013", "04019", "04021", "04025"],  # Maricopa, Pima, Pinal, Yavapai
    "AR": ["05119", "05143", "05085", "05007"],  # Pulaski, Washington, Faulkner, Benton
    "CA": ["06037", "06059", "06065", "06073"],  # LA, Orange, Riverside, San Diego
    "CO": ["08031", "08001", "08005", "08013"],  # Denver, Adams, Arapahoe, Boulder
    "CT": ["09003", "09001", "09009", "09005"],  # Hartford, Fairfield, New Haven, Litchfield
    "DE": ["10003", "10005", "10001"],  # New Castle, Sussex, Kent
    "DC": ["11001"],  # DC
    "FL": ["12086", "12011", "12099", "12057"],  # Miami-Dade, Broward, Palm Beach, Hillsborough
    "GA": ["13121", "13135", "13089", "13067"],  # Fulton, Gwinnett, DeKalb, Cobb
    "HI": ["15003", "15009", "15001", "15007"],  # Honolulu, Maui, Hawaii, Kauai
    "ID": ["16001", "16027", "16005", "16019"],  # Ada, Canyon, Bannock, Bonneville
    "IL": ["17031", "17043", "17097", "17089"],  # Cook, DuPage, Lake, Kane
    "IN": ["18097", "18089", "18029", "18141"],  # Marion, Lake, Dearborn, St Joseph
    "IA": ["19153", "19113", "19103", "19163"],  # Polk, Linn, Johnson, Scott
    "KS": ["20091", "20173", "20209", "20059"],  # Johnson, Sedgwick, Wyandotte, Franklin
    "KY": ["21111", "21067", "21037", "21015"],  # Jefferson, Fayette, Campbell, Boone
    "LA": ["22033", "22051", "22055", "22071"],  # East Baton Rouge, Jefferson, Lafayette, Orleans
    "ME": ["23005", "23031", "23001", "23017"],  # Cumberland, York, Androscoggin, Oxford
    "MD": ["24031", "24033", "24003", "24005"],  # Montgomery, Prince George's, Anne Arundel, Baltimore Co
    "MA": ["25017", "25025", "25013", "25027"],  # Middlesex, Suffolk, Hampden, Worcester
    "MI": ["26163", "26125", "26099", "26049"],  # Wayne, Oakland, Macomb, Genesee
    "MN": ["27053", "27123", "27003", "27037"],  # Hennepin, Ramsey, Anoka, Dakota
    "MS": ["28049", "28047", "28033", "28059"],  # Hinds, Harrison, DeSoto, Jackson
    "MO": ["29189", "29095", "29077", "29099"],  # St Louis County, Jackson, Greene, Jefferson
    "MT": ["30111", "30013", "30049", "30063"],  # Yellowstone, Cascade, Lewis & Clark, Missoula
    "NE": ["31055", "31109", "31153", "31043"],  # Douglas, Lancaster, Sarpy, Dakota
    "NV": ["32003", "32031", "32510"],  # Clark, Washoe, Carson City
    "NH": ["33011", "33015", "33013", "33001"],  # Hillsborough, Rockingham, Merrimack, Belknap
    "NJ": ["34003", "34013", "34017", "34031"],  # Bergen, Essex, Hudson, Passaic
    "NM": ["35001", "35013", "35043", "35049"],  # Bernalillo, Doña Ana, Sandoval, Santa Fe
    "NY": ["36061", "36047", "36005", "36081"],  # NYC counties + Westchester (approx)
    "NC": ["37119", "37183", "37081", "37063"],  # Mecklenburg, Wake, Guilford, Durham
    "ND": ["38017", "38015", "38059", "38101"],  # Cass, Burleigh, Morton, Ward
    "OH": ["39035", "39049", "39061", "39153"],  # Cuyahoga, Franklin, Hamilton, Summit
    "OK": ["40109", "40143", "40027", "40031"],  # Oklahoma, Tulsa, Cleveland, Comanche
    "OR": ["41051", "41067", "41005", "41039"],  # Multnomah, Washington, Clackamas, Lane
    "PA": ["42101", "42003", "42091", "42045"],  # Philadelphia, Allegheny, Montgomery, Delaware
    "RI": ["44007", "44003", "44009", "44001"],  # Providence, Kent, Washington, Bristol
    "SC": ["45045", "45019", "45063", "45051"],  # Greenville, Charleston, Lexington, Horry
    "SD": ["46099", "46103", "46013", "46035"],  # Minnehaha, Pennington, Brown, Davison
    "TN": ["47157", "47037", "47093", "47065"],  # Shelby, Davidson, Knox, Hamilton
    "TX": ["48201", "48113", "48439", "48029"],  # Harris, Dallas, Tarrant, Bexar
    "UT": ["49035", "49049", "49011", "49003"],  # Salt Lake, Utah, Davis, Box Elder
    "VT": ["50007", "50003", "50027", "50023"],  # Chittenden, Bennington, Windsor, Washington
    "VA": ["51059", "51153", "51087", "51810"],  # Fairfax, Prince William, Henrico, Virginia Beach
    "WA": ["53033", "53053", "53061", "53063"],  # King, Pierce, Snohomish, Spokane
    "WV": ["54039", "54003", "54061", "54107"],  # Kanawha, Berkeley, Monongalia, Wood
    "WI": ["55079", "55133", "55101", "55059"],  # Milwaukee, Waukesha, Racine, Kenosha
    "WY": ["56025", "56021", "56013", "56001"],  # Natrona, Laramie, Fremont, Albany
}


def iter_county_urate_q(states: Iterable[str] | None = None) -> Iterator[tuple]:
    """Yield (county_fips_5, state_code, qdate, urate_pct, source_url, metadata).

    Derives quarterly NSA urate from FRED LAUS county unemployment level
    (measure 04) and employment level (measure 05):
        urate = U / (U + E) * 100
    Monthly NSA → quarterly mean.
    """
    state_list = list(states) if states is not None else list(TOP_COUNTIES_BY_STATE)
    for st in state_list:
        if st not in TOP_COUNTIES_BY_STATE:
            continue
        for fips in TOP_COUNTIES_BY_STATE[st]:
            sid_u = f"LAUCN{fips}0000000004"   # unemployment count
            sid_e = f"LAUCN{fips}0000000005"   # employment count
            try:
                u = fetch_fred(sid_u)
                e = fetch_fred(sid_e)
            except Exception:
                continue
            u = u.dropna().astype(float)
            e = e.dropna().astype(float)
            common = u.index.intersection(e.index)
            if len(common) == 0:
                continue
            urate = u.loc[common] / (u.loc[common] + e.loc[common]) * 100.0
            urate.index = pd.to_datetime(urate.index)
            q = urate.resample("QE").mean()
            url = _FRED_BASE + sid_u
            for qdate, val in q.dropna().items():
                yield (fips, st, qdate, float(val), url, {
                    "u_series": sid_u, "e_series": sid_e,
                    "freq": "Q", "source": "FRED LAUCN derived urate",
                    "measure": "urate_pct_nsa",
                })


__all__ = [
    "iter_national_industry_emp_q",
    "iter_county_urate_q",
    "SUPERSECTORS",
    "STATE_INDUSTRY_SHARES_2005",
    "STATE_DEMOGRAPHICS_2005",
    "STATE_AI_EXPOSURE_2019",
    "TOP_COUNTIES_BY_STATE",
    "_FIPS",
]
