#!/usr/bin/env python
"""Build the frozen county->Fed-district crosswalk for the 14 split states.

The package's ``puremacro.narrative.indices._fed_districts`` maps STATES
to districts (with a many-to-many table for the 14 split states); the
Fed itself defines district boundaries at the COUNTY level. This
builder assembles the county-level assignment for the split states from
per-Reserve-Bank source files curated from the banks' own territory
descriptions (see ``crosswalk_sources/*.json``; each carries its source
URLs and transcription notes), matches names to Census FIPS codes, and
assigns every remaining county of each split state to the state's other
district. Any unmatched county name is a hard error — nothing drops
silently.

Inputs:
    docs/research/main_street_uncertainty/output/crosswalk_sources/*.json
        {"district": <bank>, "source_urls": [...], "notes": ...,
         "<ST>": ["County Name", ...], ...}
    Census national_county.txt (via the on-disk HTTP cache).

Output:
    docs/research/main_street_uncertainty/output/county_district_crosswalk.csv
        state, county_fips, county_name, district, assignment
        (assignment = 'listed' if the county came from a bank source
         file, 'complement' if it is the split state's other district)
"""
from __future__ import annotations

import io
import json
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from puremacro._http import safe_get_bytes_cached  # noqa: E402
from puremacro.narrative.indices._fed_districts import (  # noqa: E402
    SPLIT_STATES, STATE_TO_DISTRICTS_FULL,
)

OUT_DIR = (REPO_ROOT / "docs" / "research" / "main_street_uncertainty"
           / "output")
SRC_DIR = OUT_DIR / "crosswalk_sources"
CENSUS_URL = ("https://www2.census.gov/geo/docs/reference/codes/files/"
              "national_county.txt")

#: Independent cities / non-county equivalents that belong with a listed
#: county's district but are separate FIPS entities. Only Missouri's
#: St. Louis City matters in the split states; the source lists name it
#: explicitly, so no special-casing is needed beyond name matching.


def norm(name: str) -> str:
    """Canonical form for county-name matching: casefold, strip accents,
    drop 'County'/'Parish'/periods, collapse spaces ('De Kalb'=='DeKalb',
    'St. Clair'=='Saint Clair'? NO — 'Saint' is never used by Census in
    these states; 'St.' normalizes to 'st')."""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.casefold().replace(".", "")
    # NOTE: ' city' is deliberately NOT stripped — 'St. Louis City'
    # (29510) and 'St. Louis County' (29189) must stay distinct; the
    # caller's independent-city fallback appends ' city' when needed.
    for suffix in (" county", " parish"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s.replace(" ", "").replace("-", "")


def main() -> int:
    census = pd.read_csv(
        io.BytesIO(safe_get_bytes_cached(CENSUS_URL)), header=None,
        names=["state", "stfp", "cofp", "name", "class"], dtype=str)
    census["fips"] = census["stfp"] + census["cofp"]
    census["norm"] = census["name"].map(norm)

    src_files = sorted(SRC_DIR.glob("*.json"))
    if not src_files:
        raise SystemExit(f"no source files under {SRC_DIR}")
    sources = [json.loads(p.read_text()) for p in src_files]

    rows = []
    listed: dict[str, dict[str, str]] = {}   # state -> fips -> district
    for src in sources:
        bank = src["district"]
        for st, names in src.items():
            if st in ("district", "source_urls", "notes"):
                continue
            if st not in SPLIT_STATES:
                raise SystemExit(f"{bank}: {st} is not a split state")
            if bank not in STATE_TO_DISTRICTS_FULL[st]:
                raise SystemExit(f"{bank}: not a district of {st}")
            cs = census[census["state"] == st]
            lookup = dict(zip(cs["norm"], zip(cs["fips"], cs["name"])))
            for nm in names:
                key = norm(nm)
                if key not in lookup:
                    # Independent-city fallback (e.g. 'St. Louis City').
                    key_city = norm(nm + " city")
                    if key_city in lookup:
                        key = key_city
                    else:
                        raise SystemExit(
                            f"{bank}/{st}: county {nm!r} not matched to "
                            f"Census FIPS (norm {key!r})")
                fips, census_name = lookup[key]
                prev = listed.setdefault(st, {})
                if fips in prev and prev[fips] != bank:
                    raise SystemExit(
                        f"{st} {census_name}: listed for both {prev[fips]} "
                        f"and {bank}")
                prev[fips] = bank
                rows.append({"state": st, "county_fips": fips,
                             "county_name": census_name, "district": bank,
                             "assignment": "listed"})

    # Complements: every unlisted county of a split state goes to the
    # state's other district. Requires exactly one side to be listed.
    for st in sorted(SPLIT_STATES):
        dists = sorted(STATE_TO_DISTRICTS_FULL[st])
        got = set(listed.get(st, {}).values())
        if not got:
            raise SystemExit(f"{st}: no bank source file covers it")
        other = [d for d in dists if d not in got]
        cs = census[census["state"] == st]
        if len(other) == 0:
            # Both sides listed (e.g. KY from both St. Louis and
            # Cleveland sources): must already partition the state.
            missing = cs[~cs["fips"].isin(listed[st])]
            if len(missing):
                raise SystemExit(
                    f"{st}: both districts listed but "
                    f"{len(missing)} counties unassigned: "
                    f"{sorted(missing['name'])[:5]}...")
            continue
        if len(other) != 1:
            raise SystemExit(
                f"{st}: need exactly one unlisted district, have "
                f"listed={sorted(got)} of {dists}")
        for _, r in cs.iterrows():
            if r["fips"] not in listed[st]:
                rows.append({"state": st, "county_fips": r["fips"],
                             "county_name": r["name"],
                             "district": other[0],
                             "assignment": "complement"})

    xwalk = (pd.DataFrame(rows)
             .sort_values(["state", "county_fips"]).reset_index(drop=True))
    counts = xwalk.groupby(["state", "district"]).size()
    print(counts.to_string())
    totals = xwalk.groupby("state").size()
    expect = census[census["state"].isin(SPLIT_STATES)].groupby("state").size()
    mismatch = totals[totals != expect.reindex(totals.index)]
    if len(mismatch):
        raise SystemExit(f"county totals mismatch Census: {mismatch}")
    out = OUT_DIR / "county_district_crosswalk.csv"
    xwalk.to_csv(out, index=False)
    print(f"wrote {out} ({len(xwalk)} counties, "
          f"{int((xwalk['assignment'] == 'listed').sum())} listed / "
          f"{int((xwalk['assignment'] == 'complement').sum())} complement)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
