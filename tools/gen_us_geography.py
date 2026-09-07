"""Regenerate the shipped US geography tables (dev-only; needs network on first run).

Writes two files that `puremacro.datasets` loads:

  puremacro/datasets/data/us_county_centroids.csv
      The published Census county internal points, verbatim apart from column
      renaming and rounding.
  puremacro/datasets/data/us_state_centroids.csv
      State internal points DERIVED from those county points, plus the
      land-border adjacency held as literal data below.

Source: US Census Bureau, 2023 Gazetteer Files, national counties
(``2023_Gaz_counties_national.zip``). Public domain. The archive is downloaded
once and cached in the system temp directory; pass ``--zip <path>`` to use a
local copy instead (needed on a machine without network).

Two things this script gets right that a naive rewrite would not:

* **The state point is a spherical mean, not a planar one.** Alaska's Aleutians
  West Census Area sits at longitude ``+179.62``, east of the antimeridian.
  Averaging the raw longitude column puts Alaska's point 132 km from the true
  land-area-weighted mean. The state points are therefore the land-area-weighted
  mean of the county positions as UNIT VECTORS on the sphere, converted back to
  degrees; that is wrap-safe everywhere and differs from the planar mean only
  negligibly for the compact states.
* **Connecticut is the 2023 vintage.** The 2023 Gazetteer replaced CT's eight
  legacy counties (09001-09015) with nine planning regions (09110-09190). The
  shipped file follows the source; a merge against a pre-2024 county panel will
  not match on those rows.

Run from the repo root:  python tools/gen_us_geography.py
"""
from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

GAZETTEER_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
    "2023_Gazetteer/2023_Gaz_counties_national.zip"
)
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "puremacro" / "datasets" / "data"

STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
    "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska",
    "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "PR": "Puerto Rico",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming",
}

# Land-border adjacency, standard US political geography. Held as literal data
# because it is not derivable from the Gazetteer, which ships no geometry.
#
# Convention recorded so a future maintainer does not have to guess: this is
# QUEEN contiguity, so the Four Corners point counts as a border and Arizona
# lists Colorado (and Utah lists New Mexico). Under rook contiguity, which
# requires a shared edge rather than a shared point, those two pairs would drop
# and Missouri would stand alone at eight neighbours.
#
# Alaska, Hawaii and Puerto Rico have no land border with any other row and are
# islands in this graph — which is what makes them a useful island test for
# `SpatialWeights`.
ADJACENCY = {
    "AL": "FL GA MS TN", "AK": "", "AZ": "CA CO NM NV UT", "AR": "LA MO MS OK TN TX",
    "CA": "AZ NV OR", "CO": "AZ KS NE NM OK UT WY", "CT": "MA NY RI",
    "DE": "MD NJ PA", "DC": "MD VA", "FL": "AL GA", "GA": "AL FL NC SC TN", "HI": "",
    "ID": "MT NV OR UT WA WY", "IL": "IA IN KY MO WI", "IN": "IL KY MI OH",
    "IA": "IL MN MO NE SD WI", "KS": "CO MO NE OK", "KY": "IL IN MO OH TN VA WV",
    "LA": "AR MS TX", "ME": "NH", "MD": "DC DE PA VA WV", "MA": "CT NH NY RI VT",
    "MI": "IN OH WI", "MN": "IA ND SD WI", "MS": "AL AR LA TN",
    "MO": "AR IA IL KS KY NE OK TN", "MT": "ID ND SD WY", "NE": "CO IA KS MO SD WY",
    "NV": "AZ CA ID OR UT", "NH": "MA ME VT", "NJ": "DE NY PA", "NM": "AZ CO OK TX UT",
    "NY": "CT MA NJ PA VT", "NC": "GA SC TN VA", "ND": "MN MT SD",
    "OH": "IN KY MI PA WV", "OK": "AR CO KS MO NM TX", "OR": "CA ID NV WA",
    "PA": "DE MD NJ NY OH WV", "PR": "", "RI": "CT MA", "SC": "GA NC",
    "SD": "IA MN MT ND NE WY", "TN": "AL AR GA KY MO MS NC VA", "TX": "AR LA NM OK",
    "UT": "AZ CO ID NM NV WY", "VT": "MA NH NY", "VA": "DC KY MD NC TN WV",
    "WA": "ID OR", "WV": "KY MD OH PA VA", "WI": "IA IL MI MN",
    "WY": "CO ID MT NE SD UT",
}


def _fetch_gazetteer(local_zip: Path | None) -> bytes:
    if local_zip is not None:
        return local_zip.read_bytes()
    cache = Path("/tmp") / "2023_Gaz_counties_national.zip"
    if cache.exists():
        print(f"using cached {cache}")
        return cache.read_bytes()
    print(f"downloading {GAZETTEER_URL}")
    with urllib.request.urlopen(GAZETTEER_URL, timeout=60) as response:
        payload = response.read()
    cache.write_bytes(payload)
    return payload


def build_counties(payload: bytes) -> pd.DataFrame:
    """The published county internal points, renamed and rounded."""
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        text = archive.read(archive.namelist()[0]).decode("latin-1")
    df = pd.read_csv(io.StringIO(text), sep="\t")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "USPS": "state", "GEOID": "county_fips", "NAME": "county_name",
        "INTPTLAT": "lat", "INTPTLONG": "lon", "ALAND_SQMI": "land_sqmi",
    })
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    df["state_fips"] = df["county_fips"].str[:2]
    out = df[["county_fips", "state_fips", "state", "county_name",
              "lat", "lon", "land_sqmi"]].copy()
    out["lat"] = out["lat"].round(5)
    out["lon"] = out["lon"].round(5)
    out["land_sqmi"] = out["land_sqmi"].round(3)
    return out.sort_values("county_fips").reset_index(drop=True)


def _spherical_mean(lat_deg, lon_deg, weights) -> tuple[float, float]:
    """Weighted mean position on the sphere, via unit vectors.

    Wrap-safe: averaging raw longitudes puts Alaska's point 132 km off, because
    Aleutians West sits at +179.62 while the rest of the state is negative.
    """
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    w = np.asarray(weights, dtype=float)
    x = float(np.sum(np.cos(lat) * np.cos(lon) * w))
    y = float(np.sum(np.cos(lat) * np.sin(lon) * w))
    z = float(np.sum(np.sin(lat) * w))
    return (float(np.degrees(np.arctan2(z, np.hypot(x, y)))),
            float(np.degrees(np.arctan2(y, x))))


def build_states(counties: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (state_fips, state), block in counties.groupby(["state_fips", "state"], sort=True):
        lat, lon = _spherical_mean(block["lat"], block["lon"], block["land_sqmi"])
        rows.append({
            "state_fips": state_fips,
            "state": state,
            "state_name": STATE_NAMES[state],
            "lat": round(lat, 5),
            "lon": round(lon, 5),
            "land_sqmi": round(float(block["land_sqmi"].sum()), 3),
            "n_counties": int(len(block)),
            "neighbors": " ".join(sorted(ADJACENCY[state].split())),
        })
    return pd.DataFrame(rows).sort_values("state_fips").reset_index(drop=True)


def check(counties: pd.DataFrame, states: pd.DataFrame) -> None:
    """Invariants that must hold before either file is written."""
    assert counties["county_fips"].is_unique, "duplicate county FIPS"
    assert (counties["county_fips"].str[:2] == counties["state_fips"]).all()
    assert counties[["lat", "lon", "land_sqmi"]].notna().all().all(), "NaN in county table"
    assert counties["lat"].between(-90, 90).all()
    assert counties["lon"].between(-180, 180).all()

    assert states["state"].is_unique, "duplicate state"
    assert int(states["n_counties"].sum()) == len(counties)
    assert set(states["state"]) == set(STATE_NAMES) == set(ADJACENCY)
    assert states["lat"].between(-90, 90).all()
    assert states["lon"].between(-180, 180).all()

    adjacency = {s: v.split() for s, v in states.set_index("state")["neighbors"].items()}
    for state, others in adjacency.items():
        assert len(set(others)) == len(others), f"{state} lists a neighbour twice"
        for other in others:
            assert other in adjacency, f"{state} lists unknown neighbour {other}"
            assert state in adjacency[other], f"{state}-{other} border is not mutual"
    counts = {s: len(v) for s, v in adjacency.items()}
    # Known invariants of US geography under queen contiguity.
    assert max(counts.values()) == 8, "MO and TN should top out at eight neighbours"
    assert {s for s, c in counts.items() if c == 8} == {"MO", "TN"}
    assert adjacency["ME"] == ["NH"], "Maine borders only New Hampshire"
    assert {s for s, c in counts.items() if c == 0} == {"AK", "HI", "PR"}
    n_edges = sum(counts.values()) // 2
    print(f"  adjacency: {n_edges} mutual land borders, "
          f"{sum(1 for c in counts.values() if c == 0)} islands")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=None,
                        help="local copy of 2023_Gaz_counties_national.zip (skips the download)")
    args = parser.parse_args()

    counties = build_counties(_fetch_gazetteer(args.zip))
    states = build_states(counties)
    check(counties, states)

    for frame, name in ((counties, "us_county_centroids.csv"),
                        (states, "us_state_centroids.csv")):
        path = DATA_DIR / name
        frame.to_csv(path, index=False)
        print(f"wrote {path} ({len(frame)} rows, {path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
