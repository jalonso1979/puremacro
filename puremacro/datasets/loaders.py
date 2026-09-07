"""Canonical Macroeconomic Research Datasets.

Provides clean, zero-external-dependency access to empirical benchmark datasets:
- Galí (1999, *AER*): Technology shocks, labor productivity, and hours worked.
- Mertens & Ravn (2013, *AER*): Narrative tax shocks (anticipated & unanticipated).
- Romer & Romer (2004, 2017): Narrative monetary policy shocks.
- US Macroeconomic Quarterly Panel (GDP, Consumption, Investment, Inflation, Fed Funds).
- US Macroeconomic Monthly Panel (IP, CPI, Unemployment, FFR, NFCI, VIX).
- DICE-2016 Climate-Macro Model Calibrations (Nordhaus 2018).
"""
from __future__ import annotations

import pathlib
import pandas as pd
import numpy as np


# Datasets ship *inside* the wheel. Before 1.2.0 this pointed three levels up
# at <repo>/notebooks/course/data, which is not packaged — so every loader here
# raised FileNotFoundError (or a bare "No objects to concatenate") on any
# install that was not the author's own checkout, the iPad included.
_DATA_DIR = pathlib.Path(__file__).parent / "data"

# Kept as a fallback so a working tree still resolves files that have not been
# vendored into the package.
_REPO_DATA_DIR = pathlib.Path(__file__).parent.parent.parent / "notebooks" / "course" / "data"


def _resolve(filename: str) -> pathlib.Path:
    """Locate a bundled dataset file, packaged copy first.

    Raises
    ------
    FileNotFoundError
        Naming the file and both places that were searched — a missing
        dataset is a packaging bug, and the message should say so rather
        than surfacing later as an empty concat.
    """
    for base in (_DATA_DIR, _REPO_DATA_DIR):
        candidate = base / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"bundled dataset {filename!r} is missing from this puremacro "
        f"install (looked in {_DATA_DIR} and {_REPO_DATA_DIR}). "
        f"Reinstall the package, or report it: the file should ship in "
        f"the wheel."
    )


def load_gali1999() -> pd.DataFrame:
    """Load Galí (1999, AER) quarterly US dataset (1948Q1 - 1994Q4).

    Columns:
    - 'dlprod': Growth rate of labor productivity (d log(Y/N))
    - 'hours': Log hours worked per capita (level)
    - Raw components: 'ophnfb', 'hoanbs', 'cnp16ov'

    Returns
    -------
    pd.DataFrame
        Indexed by quarterly PeriodIndex ('1948Q2' to '1994Q4').
    """
    path = _resolve("gali1999.csv")
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df.index = df.index.to_period("Q-DEC")

    if "ophnfb" in df.columns:
        df["dlprod"] = np.log(df["ophnfb"]).diff() * 100.0
        if "cnp16ov" in df.columns and "hoanbs" in df.columns:
            df["hours"] = np.log(df["hoanbs"] / df["cnp16ov"]) * 100.0
        elif "hoanbs" in df.columns:
            df["hours"] = np.log(df["hoanbs"]) * 100.0
        df = df.dropna()

    return df


def load_narrative_tax_shocks() -> pd.DataFrame:
    """Load Mertens & Ravn (2013, AER) narrative tax liability shocks.

    Columns:
    - 'unanticipated' / 'mtr_u': Exogenous unanticipated tax changes (% of GDP)
    - 'anticipated' / 'mtr_a': Anticipated future tax changes (% of GDP)
    - 'romer_romer_exog' / 'rr_exog': Romer & Romer (2010) narrative tax series

    Returns
    -------
    pd.DataFrame
        Indexed by quarterly PeriodIndex ('1945Q1' to '2007Q4').
    """
    path = _resolve("tax14_narrative_tax_shocks.csv")
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.PeriodIndex(df["date"], freq="Q-DEC")
        df = df.set_index("date")

    if "mtr_u" in df.columns:
        df["unanticipated"] = df["mtr_u"]
    if "mtr_a" in df.columns:
        df["anticipated"] = df["mtr_a"]
    if "rr_exog" in df.columns:
        df["romer_romer_exog"] = df["rr_exog"]

    return df


def load_macro_quarterly() -> pd.DataFrame:
    """Load US quarterly macroeconomic aggregate panel.

    Combines:
    - Real GDP (GDPC1)
    - Real Gross Private Domestic Investment (GPDIC1)
    - GDP Deflator Inflation (A191RL1Q225SBEA)
    - Effective Federal Funds Rate (FEDFUNDS quarterly average)

    Returns
    -------
    pd.DataFrame
        Quarterly macroeconomic series aligned and cleaned.
    """
    series_map = {
        "GDPC1": "real_gdp",
        "GPDIC1": "real_investment",
        "A191RL1Q225SBEA": "gdp_growth_annualized",
    }
    dfs = []
    for code, col_name in series_map.items():
        p = _resolve(f"{code}.csv")
        df_s = pd.read_csv(p)
        date_col = df_s.columns[0]
        val_col = df_s.columns[1]
        df_s["date"] = pd.to_datetime(df_s[date_col])
        df_s = df_s.set_index("date")
        df_s[col_name] = pd.to_numeric(df_s[val_col], errors="coerce")
        df_q = df_s[[col_name]].resample("QE").last()
        df_q.index = df_q.index.to_period("Q-DEC")
        dfs.append(df_q)

    # Quarterly average of Fed Funds
    p_ff = _resolve("FEDFUNDS.csv")
    df_ff = pd.read_csv(p_ff)
    date_col = df_ff.columns[0]
    val_col = df_ff.columns[1]
    df_ff["date"] = pd.to_datetime(df_ff[date_col])
    df_ff = df_ff.set_index("date")
    df_ff["fed_funds"] = pd.to_numeric(df_ff[val_col], errors="coerce")
    df_ff_q = df_ff[["fed_funds"]].resample("QE").mean()
    df_ff_q.index = df_ff_q.index.to_period("Q-DEC")
    dfs.append(df_ff_q)

    df_merged = pd.concat(dfs, axis=1).dropna()
    return df_merged


def load_macro_monthly() -> pd.DataFrame:
    """Load US monthly macroeconomic indicator panel.

    Combines:
    - Headline CPI (CPIAUCSL)
    - Core CPI (CPILFESL)
    - Civilian Unemployment Rate (UNRATE)
    - Effective Federal Funds Rate (FEDFUNDS)
    - Chicago Fed National Financial Conditions Index (NFCI)

    Returns
    -------
    pd.DataFrame
        Monthly macroeconomic panel aligned and cleaned.
    """
    codes = {
        "CPIAUCSL": "cpi",
        "CPILFESL": "core_cpi",
        "UNRATE": "unemployment_rate",
        "FEDFUNDS": "fed_funds",
        "NFCI": "nfci",
    }
    dfs = []
    for code, col_name in codes.items():
        p = _resolve(f"{code}.csv")
        df_s = pd.read_csv(p)
        date_col = df_s.columns[0]
        val_col = df_s.columns[1]
        df_s["date"] = pd.to_datetime(df_s[date_col])
        df_s = df_s.set_index("date")
        df_s[col_name] = pd.to_numeric(df_s[val_col], errors="coerce")
        # If weekly (like NFCI), resample to monthly mean
        df_m = df_s[[col_name]].resample("MS").mean()
        df_m.index = df_m.index.to_period("M")
        dfs.append(df_m)

    df_merged = pd.concat(dfs, axis=1).dropna()
    return df_merged


def load_dice_parameters() -> dict[str, float | np.ndarray]:
    """Load calibrated parameters for the Nordhaus DICE-2016 Climate-Macro model.

    Returns
    -------
    dict
        Dictionary containing carbon cycle matrix Phi, radiative forcing eta,
        equilibrium climate sensitivity, capital elasticity alpha, and damage coefficients.
    """
    return {
        "capital_elasticity_alpha": 0.30,
        "depreciation_rate_5yr": 0.50,
        "savings_rate": 0.22,
        "climate_sensitivity": 3.0,
        "damage_coefficient": 0.00236,
        "radiative_forcing_eta": 3.68,
        "preindustrial_carbon_gtc": 588.0,
        "initial_carbon_reservoirs_gtc": np.array([851.0, 460.0, 1740.0]),
        "carbon_transition_matrix": np.array([
            [0.88, 0.12, 0.00],
            [0.05, 0.94, 0.01],
            [0.00, 0.002, 0.998],
        ]),
    }


def load_banxico_stance() -> pd.DataFrame:
    """Load Bank of Mexico monetary policy stance and narrative direction series.

    Columns:
    - 'banxico_direction': Directional decision indicator (+1 hike, 0 hold, -1 cut)
    - 'banxico_stance_signed': Cumulative policy stance
    - 'tight_stance_narr': Narrative tight stance dummy (1 if restrictive)

    Returns
    -------
    pd.DataFrame
        Monthly PeriodIndex ('2000M01' to '2024M12').
    """
    path = _resolve("banxico_stance_monthly.csv")
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df.index = df.index.to_period("M")
    return df


def load_us_state_centroids() -> pd.DataFrame:
    """US state geography: internal points, land area and land-border neighbours.

    The building block for the ``puremacro.spatial`` weights builders on a US
    state panel: pass the ``lat``/``lon`` columns to
    :func:`~puremacro.spatial.distance_weights` or
    :func:`~puremacro.spatial.knn_weights`, or the ``neighbors`` column to
    :func:`~puremacro.spatial.contiguity_weights`.

    Columns
    -------
    state_fips : str
        Two-digit state FIPS code, zero-padded.
    state : str
        Two-letter USPS postal abbreviation (the frame is indexed by this).
    state_name : str
    lat, lon : float
        Internal point in decimal degrees, computed as the land-area-weighted
        mean of the state's county internal points (see the note below).
    land_sqmi : float
        Total county land area, square miles.
    n_counties : int
    neighbors : str
        Space-separated postal codes of the states sharing a land border.
        Empty for Alaska, Hawaii and Puerto Rico, which are islands in this
        graph and are a useful test of island handling in ``SpatialWeights``.

    Notes
    -----
    Coordinates and land areas come from the US Census Bureau 2023 national
    county Gazetteer file (``INTPTLAT`` / ``INTPTLONG``, the published county
    internal points). The Census does not publish a state Gazetteer at the same
    location, so the state points here are **derived**: the land-area-weighted
    mean of the constituent county internal points, taken on the sphere (the
    counties are averaged as unit vectors, then converted back to degrees).
    They are representative interior points suitable for distance weights, not
    official Census state internal points, and they are not centroids of the
    state polygon.

    The spherical mean matters for exactly one row. Alaska's Aleutians West
    Census Area sits at longitude ``+179.62``, east of the antimeridian;
    averaging the raw longitude column instead would place Alaska's point
    132 km away. Every other state moves less than 7 km (median 1.2 km).

    Regenerate both this file and the county file with
    ``python tools/gen_us_geography.py``.

    The land-border adjacency in ``neighbors`` is standard US political
    geography, symmetrised (Missouri and Tennessee have the maximum, eight
    neighbours each; Maine has one).

    Returns
    -------
    pd.DataFrame
        52 rows (50 states, DC and Puerto Rico), indexed by postal code.

    Examples
    --------
    >>> from puremacro.datasets import load_us_state_centroids
    >>> geo = load_us_state_centroids()
    >>> int(geo.loc["TX", "n_counties"])
    254
    """
    path = _resolve("us_state_centroids.csv")
    df = pd.read_csv(path, dtype={"state_fips": str})
    df["neighbors"] = df["neighbors"].fillna("")
    return df.set_index("state")


def load_us_county_centroids() -> pd.DataFrame:
    """US county internal points and land area (Census 2023 Gazetteer).

    The county-level geography behind the spatial worked examples: distances
    between these points feed :func:`~puremacro.spatial.distance_weights`,
    :func:`~puremacro.spatial.conley_cov` and the exposure rings of
    :func:`~puremacro.did.spatial_did`. ``county_fips`` matches the crosswalk
    key that :func:`puremacro.bartik.build_county_epu` expects.

    Columns
    -------
    county_fips : str
        Five-digit county FIPS, zero-padded (the frame is indexed by it).
    state_fips : str
        Two-digit state FIPS.
    state : str
        Two-letter USPS postal abbreviation.
    county_name : str
    lat, lon : float
        Published county internal point, decimal degrees (``INTPTLAT`` /
        ``INTPTLONG``). An internal point is a point guaranteed to lie inside
        the county, which for a concave county is not its centroid.
    land_sqmi : float

    Returns
    -------
    pd.DataFrame
        3,222 rows (50 states, DC and Puerto Rico), indexed by county FIPS.

    Notes
    -----
    Source: US Census Bureau, 2023 Gazetteer Files, national counties
    (``2023_Gaz_counties_national.zip``). Public domain. Regenerate with
    ``python tools/gen_us_geography.py``.

    Two vintage details worth knowing before you merge on ``county_fips``:

    * **Connecticut is the 2023 vintage.** CT appears as the nine planning
      regions ``09110``-``09190``, which replaced the eight legacy counties
      ``09001``-``09015``; a merge against a pre-2024 county panel will not
      match those rows.
    * **Longitudes are not all negative.** Aleutians West (``02016``) sits at
      ``+179.62``, east of the antimeridian. :func:`~puremacro.spatial.haversine_km`
      takes the longitude difference inside a sine and wraps correctly;
      ``metric="euclidean"`` on these columns would place that county roughly
      360 degrees from its neighbours.

    Examples
    --------
    >>> from puremacro.datasets import load_us_county_centroids
    >>> counties = load_us_county_centroids()
    >>> len(counties)
    3222
    """
    path = _resolve("us_county_centroids.csv")
    df = pd.read_csv(path, dtype={"county_fips": str, "state_fips": str})
    return df.set_index("county_fips")


def list_datasets() -> pd.DataFrame:
    """List all available empirical benchmark datasets in puremacro.

    Returns
    -------
    pd.DataFrame
        Summary catalog of available datasets, frequencies, and citations.
    """
    catalog = [
        {
            "Dataset Name": "load_gali1999()",
            "Frequency": "Quarterly",
            "Periods": "1948Q1 - 1994Q4",
            "Citation": "Galí (1999, AER)",
            "Description": "Labor productivity growth and hours worked for technology SVAR identification.",
        },
        {
            "Dataset Name": "load_narrative_tax_shocks()",
            "Frequency": "Quarterly",
            "Periods": "1950Q1 - 2006Q4",
            "Citation": "Mertens & Ravn (2013, AER)",
            "Description": "Anticipated and unanticipated narrative federal tax liability changes (% GDP).",
        },
        {
            "Dataset Name": "load_macro_quarterly()",
            "Frequency": "Quarterly",
            "Periods": "1954Q3 - Present",
            "Citation": "FRED / BEA / Federal Reserve",
            "Description": "Quarterly panel of Real GDP, Investment, GDP Deflator Inflation, and Fed Funds.",
        },
        {
            "Dataset Name": "load_macro_monthly()",
            "Frequency": "Monthly",
            "Periods": "1971M01 - Present",
            "Citation": "FRED / BLS / Chicago Fed",
            "Description": "Monthly panel of Headline CPI, Core CPI, Unemployment, Fed Funds, and NFCI.",
        },
        {
            "Dataset Name": "load_us_state_centroids()",
            "Frequency": "Cross-section",
            "Periods": "2023 vintage",
            "Citation": "US Census Bureau 2023 Gazetteer",
            "Description": "US state internal points, land area and land-border neighbours for spatial weights.",
        },
        {
            "Dataset Name": "load_us_county_centroids()",
            "Frequency": "Cross-section",
            "Periods": "2023 vintage",
            "Citation": "US Census Bureau 2023 Gazetteer",
            "Description": "3,222 county internal points and land areas, keyed by county FIPS.",
        },
        {
            "Dataset Name": "load_dice_parameters()",
            "Frequency": "5-Year Calibrated",
            "Periods": "2020 - 2170",
            "Citation": "Nordhaus (2018, PNAS)",
            "Description": "Calibrated 3-box carbon cycle, climate sensitivity, and damage coefficients.",
        },
    ]
    return pd.DataFrame(catalog)


__all__ = [
    "load_gali1999",
    "load_narrative_tax_shocks",
    "load_macro_quarterly",
    "load_macro_monthly",
    "load_banxico_stance",
    "load_dice_parameters",
    "load_us_state_centroids",
    "load_us_county_centroids",
    "list_datasets",
]
