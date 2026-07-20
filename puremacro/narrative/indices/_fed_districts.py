"""Federal Reserve district crosswalk (12 districts <-> states).

Source
------
Board of Governors of the Federal Reserve System, "The Twelve Federal
Reserve Districts" (official district map and per-Reserve-Bank district
descriptions), https://www.federalreserve.gov/aboutthefed/federal-reserve-system.htm
(section "Federal Reserve Banks"). District compositions follow the
descriptions published on each Reserve Bank's "about" page.

Conventions and simplifications
-------------------------------
* District boundaries are defined by the Fed at the *county* level; this
  module works at the *state* level, so 14 states that straddle two
  districts are "split states".
* ``STATE_TO_DISTRICT`` resolves each split state to a single **primary**
  district: the district containing the larger share of the state's
  resident population. The famous case is Missouri, which hosts two
  Reserve Banks (St. Louis and Kansas City); eastern Missouri (St. Louis
  metro's Missouri side, ~2.1M) outweighs western Missouri (Kansas City
  metro's Missouri side, ~1.3M), so ``MO -> "St. Louis"``. The closest
  call is Kentucky (Louisville/western KY in the 8th vs Lexington and the
  Cincinnati suburbs in the 4th); it is assigned to St. Louis, which
  operates the Louisville Branch.
* ``STATE_TO_DISTRICTS_FULL`` exposes the full many-to-many state ->
  districts mapping so users can choose their own resolution rule.
* Coverage is the 50 states + the District of Columbia (51 keys).
  Territories (PR, VI -> New York; GU, AS, MP -> San Francisco) live in
  the separate ``TERRITORY_TO_DISTRICT`` map and are excluded from the
  51-key invariant.

District names match ``puremacro.narrative.sources.beige_book.DISTRICTS``
exactly (e.g. ``"St. Louis"``, ``"Kansas City"``), so this crosswalk can
be joined directly onto Beige Book records and BBUI output.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FedDistrict:
    """One Federal Reserve district.

    Attributes
    ----------
    number : int
        Official district number (1 = Boston ... 12 = San Francisco).
    name : str
        Canonical short name (matches Beige Book connector districts).
    states : tuple[str, ...]
        USPS codes of states whose *primary* assignment is this district
        (state-level population-majority convention; see module docstring).
    partial_states : tuple[str, ...]
        USPS codes of split states that overlap this district but whose
        primary assignment is elsewhere.
    """
    number: int
    name: str
    states: tuple[str, ...]
    partial_states: tuple[str, ...] = ()


FED_DISTRICTS: tuple[FedDistrict, ...] = (
    FedDistrict(1, "Boston",
                states=("CT", "MA", "ME", "NH", "RI", "VT")),
    FedDistrict(2, "New York",
                states=("NJ", "NY"),
                partial_states=("CT",)),          # Fairfield County, CT
    FedDistrict(3, "Philadelphia",
                states=("DE", "PA"),
                partial_states=("NJ",)),          # southern New Jersey
    FedDistrict(4, "Cleveland",
                states=("OH",),
                partial_states=("KY", "PA", "WV")),  # eastern KY, western PA,
                                                     # WV northern panhandle
    FedDistrict(5, "Richmond",
                states=("DC", "MD", "NC", "SC", "VA", "WV")),
    FedDistrict(6, "Atlanta",
                states=("AL", "FL", "GA", "LA", "MS", "TN")),
    FedDistrict(7, "Chicago",
                states=("IA", "IL", "IN", "MI", "WI")),
    FedDistrict(8, "St. Louis",
                states=("AR", "KY", "MO"),
                partial_states=("IL", "IN", "MS", "TN")),  # southern IL/IN,
                                                           # northern MS,
                                                           # western TN
    FedDistrict(9, "Minneapolis",
                states=("MN", "MT", "ND", "SD"),
                partial_states=("MI", "WI")),     # Upper Peninsula MI,
                                                  # northwestern WI
    FedDistrict(10, "Kansas City",
                states=("CO", "KS", "NE", "NM", "OK", "WY"),
                partial_states=("MO",)),          # western Missouri
    FedDistrict(11, "Dallas",
                states=("TX",),
                partial_states=("LA", "NM")),     # northern LA, southern NM
    FedDistrict(12, "San Francisco",
                states=("AK", "AZ", "CA", "HI", "ID", "NV",
                        "OR", "UT", "WA")),
)

#: Canonical district names ordered by district number (1..12).
DISTRICT_NAMES: tuple[str, ...] = tuple(d.name for d in FED_DISTRICTS)

#: District name -> official district number.
DISTRICT_NUMBER: dict[str, int] = {d.name: d.number for d in FED_DISTRICTS}

#: State USPS code -> primary district name (50 states + DC, 51 keys).
STATE_TO_DISTRICT: dict[str, str] = {
    s: d.name for d in FED_DISTRICTS for s in d.states
}

#: State USPS code -> every district it overlaps (many-to-many, 51 keys).
STATE_TO_DISTRICTS_FULL: dict[str, frozenset[str]] = {
    s: frozenset(
        {d.name for d in FED_DISTRICTS if s in d.states}
        | {d.name for d in FED_DISTRICTS if s in d.partial_states}
    )
    for s in STATE_TO_DISTRICT
}

#: States that straddle two districts (14 states as of the current map).
SPLIT_STATES: frozenset[str] = frozenset(
    s for s, ds in STATE_TO_DISTRICTS_FULL.items() if len(ds) > 1
)

#: U.S. territories -> district (kept out of the 51-key invariant).
TERRITORY_TO_DISTRICT: dict[str, str] = {
    "PR": "New York",        # Puerto Rico
    "VI": "New York",        # U.S. Virgin Islands
    "GU": "San Francisco",   # Guam
    "AS": "San Francisco",   # American Samoa
    "MP": "San Francisco",   # Northern Mariana Islands
}


def district_states(district: str, *, include_partial: bool = False
                    ) -> tuple[str, ...]:
    """States mapped to ``district``.

    Parameters
    ----------
    district : canonical district name (see ``DISTRICT_NAMES``).
    include_partial : if True, also include split states whose primary
        assignment is elsewhere but which partially lie in this district.
    """
    for d in FED_DISTRICTS:
        if d.name == district:
            if include_partial:
                return tuple(sorted(set(d.states) | set(d.partial_states)))
            return d.states
    raise KeyError(
        f"unknown Federal Reserve district {district!r}; "
        f"expected one of {DISTRICT_NAMES}"
    )


def state_district(state: str, *, all_districts: bool = False
                   ) -> str | frozenset[str]:
    """District(s) for a state USPS code.

    Returns the primary district name by default; with
    ``all_districts=True`` returns the full (possibly >1) frozenset of
    overlapping districts. Territories resolve via
    ``TERRITORY_TO_DISTRICT`` (never split).
    """
    code = state.strip().upper()
    if code in STATE_TO_DISTRICT:
        if all_districts:
            return STATE_TO_DISTRICTS_FULL[code]
        return STATE_TO_DISTRICT[code]
    if code in TERRITORY_TO_DISTRICT:
        d = TERRITORY_TO_DISTRICT[code]
        return frozenset({d}) if all_districts else d
    raise KeyError(
        f"unknown state/territory code {state!r}; expected a USPS "
        f"two-letter code (50 states + DC, or PR/VI/GU/AS/MP)"
    )


def district_crosswalk() -> pd.DataFrame:
    """Tidy state <-> district crosswalk as a DataFrame.

    Returns
    -------
    DataFrame with one row per (state, overlapping district) pair and
    columns ``state``, ``district``, ``district_number``, ``primary``
    (bool: is this the state's primary district?). Exactly 51 rows have
    ``primary=True`` (50 states + DC); split states contribute one extra
    non-primary row per additional district.
    """
    rows = []
    for state in sorted(STATE_TO_DISTRICT):
        primary = STATE_TO_DISTRICT[state]
        for district in sorted(STATE_TO_DISTRICTS_FULL[state]):
            rows.append({
                "state": state,
                "district": district,
                "district_number": DISTRICT_NUMBER[district],
                "primary": district == primary,
            })
    return pd.DataFrame(rows, columns=["state", "district",
                                       "district_number", "primary"])


__all__ = [
    "FedDistrict",
    "FED_DISTRICTS",
    "DISTRICT_NAMES",
    "DISTRICT_NUMBER",
    "STATE_TO_DISTRICT",
    "STATE_TO_DISTRICTS_FULL",
    "SPLIT_STATES",
    "TERRITORY_TO_DISTRICT",
    "district_states",
    "state_district",
    "district_crosswalk",
]
