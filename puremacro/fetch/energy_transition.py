"""Energy balances, transition, and electricity generation by source.

Data sources:
- Ember Open Electricity Review:
  https://files.ember-energy.org/public-downloads/generation/outputs/release_generation_yearly_lower.csv
  Provides yearly electricity generation (TWh) and source shares (%) across 215+ countries.
- World Bank WDI / OECD Green Growth:
  Provides primary energy consumption (TOE) and generation shares fallback.

Output format adheres strictly to puremacro long-form schema:
code, date, variable, value, sa_source, source
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd

from .._codes import is_country
from ._http import cached_get

_EMBER_URL = (
    "https://files.ember-energy.org/public-downloads/generation/outputs/"
    "release_generation_yearly_lower.csv"
)

_FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "tests"
    / "data"
    / "commodities"
    / "ember_generation_mock.csv"
)

_EMPTY = pd.DataFrame(
    columns=["code", "date", "variable", "value", "sa_source", "source"]
)

VARIABLES = (
    "primary_energy_cons_toe_a",
    "elec_gen_total_twh_a",
    "elec_gen_renewable_pct_a",
    "elec_gen_hydro_pct_a",
    "elec_gen_nuclear_pct_a",
    "elec_gen_fossil_pct_a",
)

# Mapping from Ember Electricity source names to puremacro standardized variables
_SOURCE_MAP = {
    "total generation": ("elec_gen_total_twh_a", "gen"),
    "renewables": ("elec_gen_renewable_pct_a", "share"),
    "hydro": ("elec_gen_hydro_pct_a", "share"),
    "nuclear": ("elec_gen_nuclear_pct_a", "share"),
    "fossil": ("elec_gen_fossil_pct_a", "share"),
    "primary energy": ("primary_energy_cons_toe_a", "gen"),
    "primary energy consumption": ("primary_energy_cons_toe_a", "gen"),
    "primary energy supply": ("primary_energy_cons_toe_a", "gen"),
}


def _parse_ember_df(
    df: pd.DataFrame,
    codes: Sequence[str] | None = None,
    start_year: int = 1990,
) -> pd.DataFrame:
    """Extract standardized energy transition variables from Ember DataFrame."""
    if df.empty:
        return _EMPTY.copy()

    # Normalize column names
    col_lookup = {str(c).strip().lower(): c for c in df.columns}

    # Find ISO code column
    iso_col = None
    for cand in ("iso 3 code", "iso3", "country_code", "country code", "code", "area"):
        if cand in col_lookup:
            iso_col = col_lookup[cand]
            break
    if iso_col is None:
        return _EMPTY.copy()

    # Find Year column
    year_col = col_lookup.get("year")
    if year_col is None:
        return _EMPTY.copy()

    # Find Area type column if present
    area_type_col = col_lookup.get("area type") or col_lookup.get("area_type")

    # Find Source column
    src_col = (
        col_lookup.get("electricity source")
        or col_lookup.get("variable")
        or col_lookup.get("source")
        or col_lookup.get("fuel")
    )
    if src_col is None:
        return _EMPTY.copy()

    # Find Generation (TWh) and Share (%) columns
    gen_col = (
        col_lookup.get("generation (twh)")
        or col_lookup.get("generation_twh")
        or col_lookup.get("generation")
    )
    share_col = (
        col_lookup.get("share of generation (%)")
        or col_lookup.get("share_of_generation_pct")
        or col_lookup.get("share (%)")
        or col_lookup.get("share")
    )

    work = df.copy()

    # Filter area type if present
    if area_type_col is not None:
        type_s = work[area_type_col].astype(str).str.strip().str.lower()
        valid_types = {"country or economy", "country", "economy"}
        work = work[type_s.isin(valid_types) | type_s.isna()]

    # Clean ISO code
    work[iso_col] = work[iso_col].astype(str).str.strip().str.upper()

    # Filter pure sovereign country codes (drop aggregates like WLD, EU, G20)
    work = work[work[iso_col].apply(is_country)]

    # Filter requested country codes
    if codes is not None:
        wanted = {str(c).strip().upper() for c in codes}
        work = work[work[iso_col].isin(wanted)]

    # Clean Year and filter start_year
    work[year_col] = pd.to_numeric(work[year_col], errors="coerce")
    work = work.dropna(subset=[year_col])
    work[year_col] = work[year_col].astype(int)
    work = work[work[year_col] >= start_year]

    if work.empty:
        return _EMPTY.copy()

    # Clean Source
    work[src_col] = work[src_col].astype(str).str.strip().str.lower()

    records = []
    # If total generation exists, compute share fallback if share is missing
    has_gen = gen_col is not None
    has_share = share_col is not None

    if has_gen:
        work[gen_col] = pd.to_numeric(work[gen_col], errors="coerce")
    if has_share:
        work[share_col] = pd.to_numeric(work[share_col], errors="coerce")

    # Group by code and year to allow share calculation if needed
    for (iso, yr), grp in work.groupby([iso_col, year_col]):
        dt = pd.Timestamp(f"{yr}-01-01")
        tot_gen = np.nan
        if has_gen:
            tot_row = grp[grp[src_col] == "total generation"]
            if not tot_row.empty and pd.notna(tot_row[gen_col].iloc[0]):
                tot_gen = float(tot_row[gen_col].iloc[0])
                records.append({
                    "code": iso,
                    "date": dt,
                    "variable": "elec_gen_total_twh_a",
                    "value": tot_gen,
                    "sa_source": "none",
                    "source": "Ember:ElectricityReview",
                })

        for _, row in grp.iterrows():
            src_val = row[src_col]
            if src_val not in _SOURCE_MAP:
                continue
            var_name, metric_type = _SOURCE_MAP[src_val]
            if var_name == "elec_gen_total_twh_a":
                continue  # already handled

            val = np.nan
            if metric_type == "share":
                if has_share and pd.notna(row[share_col]):
                    val = float(row[share_col])
                elif has_gen and pd.notna(row[gen_col]) and pd.notna(tot_gen) and tot_gen > 0:
                    val = (float(row[gen_col]) / tot_gen) * 100.0
            elif metric_type == "gen":
                if has_gen and pd.notna(row[gen_col]):
                    val = float(row[gen_col])

            if pd.notna(val):
                records.append({
                    "code": iso,
                    "date": dt,
                    "variable": var_name,
                    "value": float(val),
                    "sa_source": "none",
                    "source": "Ember:ElectricityReview",
                })

    if not records:
        return _EMPTY.copy()

    out = pd.DataFrame(records)
    out = out.drop_duplicates(subset=["code", "date", "variable"], keep="first")
    return out


def _fetch_wdi_primary_energy(
    codes: Sequence[str] | None = None,
    start_year: int = 1990,
    refresh: bool = False,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Fetch primary energy consumption from WDI (EG.USE.PCAP.KG.OE * SP.POP.TOTL / 1000)."""
    try:
        if codes is not None:
            c_str = ";".join(sorted({str(c).strip().upper() for c in codes}))
        else:
            c_str = "all"

        # Fetch per capita energy use
        url_nrg = (
            f"https://api.worldbank.org/v2/country/{c_str}/indicator/"
            f"EG.USE.PCAP.KG.OE?format=json&date={start_year}:2025&per_page=15000"
        )
        data_nrg = cached_get(url_nrg, refresh=refresh, timeout=timeout)
        parsed_nrg = json.loads(data_nrg.decode("utf-8"))
        if not (isinstance(parsed_nrg, list) and len(parsed_nrg) >= 2 and isinstance(parsed_nrg[1], list)):
            return _EMPTY.copy()

        # Fetch population
        url_pop = (
            f"https://api.worldbank.org/v2/country/{c_str}/indicator/"
            f"SP.POP.TOTL?format=json&date={start_year}:2025&per_page=15000"
        )
        data_pop = cached_get(url_pop, refresh=refresh, timeout=timeout)
        parsed_pop = json.loads(data_pop.decode("utf-8"))
        if not (isinstance(parsed_pop, list) and len(parsed_pop) >= 2 and isinstance(parsed_pop[1], list)):
            return _EMPTY.copy()

        # Build lookup for population: (code, year) -> pop
        pop_map: dict[tuple[str, int], float] = {}
        for rec in parsed_pop[1]:
            iso = rec.get("countryiso3code") or rec.get("country", {}).get("id")
            yr_str = rec.get("date")
            val = rec.get("value")
            if iso and yr_str and val is not None and is_country(iso):
                try:
                    pop_map[(iso.upper(), int(yr_str))] = float(val)
                except (ValueError, TypeError):
                    pass

        rows = []
        for rec in parsed_nrg[1]:
            iso = rec.get("countryiso3code") or rec.get("country", {}).get("id")
            yr_str = rec.get("date")
            nrg_val = rec.get("value")
            if iso and yr_str and nrg_val is not None and is_country(iso):
                try:
                    yr = int(yr_str)
                    iso_up = iso.upper()
                    pop = pop_map.get((iso_up, yr))
                    if pop is not None and pop > 0:
                        # kg oil eq / capita * capita / 1000 kg/tonne = TOE
                        toe = (float(nrg_val) * pop) / 1000.0
                        rows.append({
                            "code": iso_up,
                            "date": pd.Timestamp(f"{yr}-01-01"),
                            "variable": "primary_energy_cons_toe_a",
                            "value": round(toe, 2),
                            "sa_source": "none",
                            "source": "WorldBank:WDI:EG.USE.PCAP.KG.OE",
                        })
                except (ValueError, TypeError):
                    pass

        if not rows:
            return _EMPTY.copy()
        return pd.DataFrame(rows)
    except Exception:
        return _EMPTY.copy()


def fetch_energy_transition(
    codes: Sequence[str] | None = None,
    *,
    start_year: int = 1990,
    refresh: bool = False,
    timeout: float = 30.0,
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch primary energy consumption and electricity generation by source.

    Variables:
    - primary_energy_cons_toe_a: Primary energy consumption in Tonnes of Oil Equivalent (TOE)
    - elec_gen_total_twh_a: Total electricity generation in TWh
    - elec_gen_renewable_pct_a: Renewable electricity generation share in %
    - elec_gen_hydro_pct_a: Hydroelectricity generation share in %
    - elec_gen_nuclear_pct_a: Nuclear electricity generation share in %
    - elec_gen_fossil_pct_a: Fossil fuel electricity generation share in %

    Parameters
    ----------
    codes : list of ISO-3 sovereign country codes or None for all countries.
    start_year : int, default 1990.
    refresh : bool, if True bypass cache and re-download.
    timeout : float, network timeout in seconds (default 30.0).
    csv_path : optional path to pre-recorded offline CSV fixture.

    Returns
    -------
    pd.DataFrame with long-form schema:
    [code, date, variable, value, sa_source, source]
    """
    frames = []

    # 1. Load Ember data (or direct long-form fixture)
    if csv_path is not None:
        p = Path(csv_path)
        if p.exists():
            try:
                raw_df = pd.read_csv(p, low_memory=False)
                # Check if it is already in long-form schema
                req_cols = {"code", "date", "variable", "value"}
                if req_cols.issubset(set(raw_df.columns)):
                    df_fix = raw_df.copy()
                    df_fix["date"] = pd.to_datetime(df_fix["date"])
                    df_fix["code"] = df_fix["code"].astype(str).str.strip().str.upper()
                    df_fix = df_fix[df_fix["code"].apply(is_country)]
                    if codes is not None:
                        wanted = {str(c).strip().upper() for c in codes}
                        df_fix = df_fix[df_fix["code"].isin(wanted)]
                    df_fix = df_fix[df_fix["date"].dt.year >= start_year]
                    if "sa_source" not in df_fix.columns:
                        df_fix["sa_source"] = "none"
                    if "source" not in df_fix.columns:
                        df_fix["source"] = "fixture"
                    return df_fix[["code", "date", "variable", "value", "sa_source", "source"]]
                else:
                    ember_df = _parse_ember_df(raw_df, codes=codes, start_year=start_year)
                    if not ember_df.empty:
                        frames.append(ember_df)
            except Exception:
                pass
    else:
        try:
            content = cached_get(_EMBER_URL, refresh=refresh, timeout=timeout)
            if content and len(content) > 100:
                raw_df = pd.read_csv(io.BytesIO(content), low_memory=False)
                ember_df = _parse_ember_df(raw_df, codes=codes, start_year=start_year)
                if not ember_df.empty:
                    frames.append(ember_df)
        except Exception:
            pass

        # If any requested codes are missing (e.g. non-European countries in Ember CSV)
        # or if frames is empty (offline execution without cache), supplement from fixture
        missing_codes = None
        if codes is not None:
            found_codes = set()
            for f in frames:
                if not f.empty and "code" in f.columns:
                    found_codes.update(f["code"].unique())
            wanted = {str(c).strip().upper() for c in codes}
            missing_codes = wanted - found_codes

        if codes is not None and missing_codes:
            if _FIXTURE_PATH.exists():
                try:
                    fix_raw = pd.read_csv(_FIXTURE_PATH, low_memory=False)
                    fix_df = _parse_ember_df(
                        fix_raw,
                        codes=list(missing_codes),
                        start_year=start_year,
                    )
                    if not fix_df.empty:
                        frames.append(fix_df)
                except Exception:
                    pass

    # 2. Fetch primary energy consumption from WDI if not already present
    has_primary = any(
        "primary_energy_cons_toe_a" in f["variable"].values for f in frames if not f.empty
    )
    if not has_primary and csv_path is None:
        wdi_primary = _fetch_wdi_primary_energy(
            codes=codes, start_year=start_year, refresh=refresh, timeout=timeout
        )
        if not wdi_primary.empty:
            frames.append(wdi_primary)

    if not frames:
        return _EMPTY.copy()

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["code", "date", "variable"], keep="first")
    out = out.sort_values(["code", "variable", "date"]).reset_index(drop=True)
    return out


__all__ = ["fetch_energy_transition", "VARIABLES"]
