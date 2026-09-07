"""Tests for built-in macroeconomic dataset loaders."""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from puremacro.datasets import (
    load_gali1999,
    load_narrative_tax_shocks,
    load_macro_quarterly,
    load_macro_monthly,
    load_dice_parameters,
    list_datasets,
)


def test_load_gali1999():
    df = load_gali1999()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "dlprod" in df.columns or len(df.columns) >= 2
    assert len(df) > 100


def test_load_narrative_tax_shocks():
    df = load_narrative_tax_shocks()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) > 100


def test_load_macro_quarterly():
    df = load_macro_quarterly()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "real_gdp" in df.columns
    assert "fed_funds" in df.columns
    assert len(df) > 100


def test_load_macro_monthly():
    df = load_macro_monthly()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "cpi" in df.columns
    assert "unemployment_rate" in df.columns
    assert "fed_funds" in df.columns
    assert len(df) > 100


def test_load_banxico_stance():
    from puremacro.datasets import load_banxico_stance
    df = load_banxico_stance()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "banxico_direction" in df.columns
    assert len(df) > 100


def test_load_dice_parameters():
    params = load_dice_parameters()
    assert isinstance(params, dict)
    assert "capital_elasticity_alpha" in params
    assert params["capital_elasticity_alpha"] == 0.30
    assert "carbon_transition_matrix" in params
    assert params["carbon_transition_matrix"].shape == (3, 3)


def test_list_datasets():
    catalog = list_datasets()
    assert isinstance(catalog, pd.DataFrame)
    assert len(catalog) >= 5
    assert "Dataset Name" in catalog.columns
    assert "Citation" in catalog.columns


# ---------------------------------------------------------------------------
# US geography (Census 2023 Gazetteer) — the input to puremacro.spatial
# ---------------------------------------------------------------------------
def test_load_us_state_centroids_shape_and_keys():
    from puremacro.datasets import load_us_state_centroids

    geo = load_us_state_centroids()
    assert geo.index.name == "state"
    assert len(geo) == 52  # 50 states + DC + PR
    assert geo.index.is_unique
    assert {"state_fips", "state_name", "lat", "lon", "land_sqmi",
            "n_counties", "neighbors"} <= set(geo.columns)
    # FIPS stays a zero-padded string, not an int that lost its leading zero.
    assert geo.loc["AL", "state_fips"] == "01"
    assert geo["state_fips"].str.len().eq(2).all()


def test_us_state_centroids_coordinates_are_plausible():
    from puremacro.datasets import load_us_state_centroids

    geo = load_us_state_centroids()
    # Latitude first, longitude second, in degrees — the convention every
    # spatial builder assumes. A transposed pair would break this.
    assert geo["lat"].between(17.0, 72.0).all()
    assert geo["lon"].between(-180.0, -65.0).all()
    # Spot checks against known geography, generous enough to survive a
    # re-derivation from a later Gazetteer vintage.
    assert abs(geo.loc["TX", "lat"] - 31.5) < 1.5
    assert abs(geo.loc["TX", "lon"] + 99.3) < 1.5
    assert geo.loc["AK", "lat"] > 60.0
    assert geo.loc["HI", "lon"] < -150.0
    assert int(geo.loc["TX", "n_counties"]) == 254
    assert geo["n_counties"].sum() == 3222


def test_us_state_neighbors_are_symmetric_and_known():
    from puremacro.datasets import load_us_state_centroids

    geo = load_us_state_centroids()
    nb = {s: v.split() for s, v in geo["neighbors"].items()}
    # Every listed border is mutual, and every neighbour is a real row.
    for state, others in nb.items():
        for other in others:
            assert other in nb, f"{state} lists unknown neighbour {other}"
            assert state in nb[other], f"{state}-{other} border is not mutual"
    # Missouri and Tennessee tie for the most land borders (eight each);
    # Maine has exactly one; the islands have none.
    counts = {s: len(v) for s, v in nb.items()}
    assert max(counts.values()) == 8
    assert {s for s, c in counts.items() if c == 8} == {"MO", "TN"}
    assert nb["ME"] == ["NH"]
    assert {s for s, c in counts.items() if c == 0} == {"AK", "HI", "PR"}


def test_state_geography_drives_the_spatial_builders():
    from puremacro.datasets import load_us_state_centroids
    from puremacro.spatial import contiguity_weights, distance_weights, knn_weights

    geo = load_us_state_centroids()
    lower48 = geo.drop(index=["AK", "HI", "PR"])
    nb = {s: [n for n in v.split() if n in lower48.index]
          for s, v in lower48["neighbors"].items()}
    W = contiguity_weights(nb)
    assert W.n == 49  # lower 48 + DC
    assert W.n_islands == 0
    assert W.binary().to_dense().sum() == 218  # 109 mutual land borders
    Wk = knn_weights(lower48[["lat", "lon"]], k=4)
    assert Wk.n == 49 and (Wk.n_neighbors == 4).all()
    Wd = distance_weights(lower48[["lat", "lon"]], cutoff=600.0)
    assert Wd.n_islands == 0


def test_load_us_county_centroids():
    from puremacro.datasets import load_us_county_centroids

    counties = load_us_county_centroids()
    assert counties.index.name == "county_fips"
    assert len(counties) == 3222
    assert counties.index.is_unique
    assert (counties.index.str.len() == 5).all()
    assert counties.loc["01001", "county_name"] == "Autauga County"
    assert counties["lat"].between(17.0, 72.0).all()
    assert counties["lon"].between(-180.0, 180.0).all()
    # The Aleutians straddle the antimeridian, so county longitudes are NOT
    # all negative: Aleutians West sits just east of +179. haversine_km takes
    # the difference inside a sine, so it wraps correctly; a Euclidean metric
    # on these columns would put that county ~360 degrees from its neighbours.
    assert counties["lon"].max() > 179.0
    assert (counties["lon"] > 0).sum() < 5
    # The county FIPS carries its state FIPS as the first two characters —
    # the join key puremacro.bartik.build_county_epu relies on.
    assert (counties.index.str[:2] == counties["state_fips"]).all()


def test_state_point_uses_a_spherical_mean_not_a_planar_one():
    """Alaska is the one row where the difference bites.

    Aleutians West sits at longitude +179.62, east of the antimeridian, so a
    plain weighted mean of the longitude column lands ~132 km from the
    land-area-weighted mean position on the sphere. `tools/gen_us_geography.py`
    averages the county positions as unit vectors; this pins that it did.
    """
    import numpy as np

    from puremacro.datasets import load_us_county_centroids, load_us_state_centroids
    from puremacro.spatial import haversine_km

    counties = load_us_county_centroids()
    states = load_us_state_centroids()
    ak = counties[counties["state"] == "AK"]
    assert (ak["lon"] > 0).sum() == 1, "expected exactly one AK county east of the antimeridian"

    lat, lon, w = np.radians(ak["lat"]), np.radians(ak["lon"]), ak["land_sqmi"].to_numpy()
    x = float(np.sum(np.cos(lat) * np.cos(lon) * w))
    y = float(np.sum(np.cos(lat) * np.sin(lon) * w))
    z = float(np.sum(np.sin(lat) * w))
    want = np.array([[np.degrees(np.arctan2(z, np.hypot(x, y))),
                      np.degrees(np.arctan2(y, x))]])
    got = states.loc[["AK"], ["lat", "lon"]].to_numpy(dtype=float)
    assert float(haversine_km(got, want)[0, 0]) < 1.0

    naive = np.array([[np.average(ak["lat"], weights=w), np.average(ak["lon"], weights=w)]])
    assert float(haversine_km(naive, want)[0, 0]) > 100.0, (
        "the planar mean should be far off — if it is not, this test no longer pins anything")


def test_connecticut_is_the_2023_planning_region_vintage():
    """Documented in the loader: CT is the nine 2023 planning regions, not the
    eight legacy counties. A merge against a pre-2024 panel will miss them."""
    from puremacro.datasets import load_us_county_centroids

    ct = load_us_county_centroids().query("state == 'CT'")
    assert len(ct) == 9
    assert set(ct.index) == {f"091{n}0" for n in range(1, 10)}
    assert ct["county_name"].str.endswith("Planning Region").all()
