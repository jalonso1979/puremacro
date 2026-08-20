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
        df.index = df.index.to_period("Q")

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
        df["date"] = pd.PeriodIndex(df["date"], freq="Q")
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
        df_q.index = df_q.index.to_period("Q")
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
    df_ff_q.index = df_ff_q.index.to_period("Q")
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
    "load_dice_parameters",
    "list_datasets",
]
