"""OECD REST fetcher for cross-country real variables.

Uses the OECD SDMX REST API v2 (sdmx.oecd.org/public/rest, 2024+).
pandasdmx is NOT used — it is incompatible with Python 3.13 (pydantic forward-ref
issue). The REST endpoints are called through
:func:`puremacro.fetch._oecd_sdmx.get_sdmx_csv`, which owns the on-disk cache and
defers its ``requests`` import, so importing this module needs no network stack.

Dataflows used
--------------
- QNA expenditure (quarterly GDP + GFCF):
    OECD.SDD.NAD, DSD_NAMAIN1@DF_QNA_EXPENDITURE_INDICES
    Volume indices (PRICE_BASE=LR, chain-linked, reference year), ADJUSTMENT=Y (SA).
- Consumer prices (monthly CPI):
    OECD.SDD.TPS, DSD_PRICES@DF_PRICES_ALL
    Total-economy index (EXPENDITURE=_T, UNIT_MEASURE=IX).  SA (ADJUSTMENT=S) where
    available; falls back to NSA for countries where OECD does not publish SA CPI
    (e.g. Mexico), with sa_source='none' so the build-time X-13 audit can handle it.
- Labor market (monthly employment + unemployment rate):
    OECD.SDD.TPS, DSD_LFS@DF_IALFS_INDIC
    MEASURE=EMP (persons, UNIT_MEASURE=PS) and MEASURE=UNE_LF / UNE_LF_M
    (unemployment as % of labour force, UNIT_MEASURE=PT_LF_SUB), both SA
    (ADJUSTMENT=Y), sex=total (_T), age Y_GE15.  Note: not all countries publish
    monthly employment; those with quarterly-only data will not appear in log_emp
    for the monthly frame (the panel builder will handle aggregation/interpolation).

All returned DataFrames are in long form with columns:
    code, date, variable, value, sa_source, source
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------



def _get_csv(agency_flow: str, key: str, start_period: str,
              *, refresh: bool = False) -> pd.DataFrame:
    """Fetch one OECD SDMX URL via cached helper. Empty on persistent failure.

    The shared cache survives 429 rate-limits and process restarts, so a
    single successful fetch is enough to feed every subsequent build_all.
    """
    from ._oecd_sdmx import get_sdmx_csv
    return get_sdmx_csv(agency_flow, key, start_period, refresh=refresh)


def _long_frame(
    wide: pd.Series,
    *,
    code_col: str = "REF_AREA",
    date_col: str = "TIME_PERIOD",
    value_col: str = "OBS_VALUE",
    variable: str,
    sa_source: str,
    source: str,
    date_freq: str = "Q",
) -> pd.DataFrame:
    """Convert a filtered slice of an OECD CSV to the project's long-form schema."""
    df = wide[[code_col, date_col, value_col]].copy()
    df.columns = ["code", "date", "value"]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    if date_freq == "Q":
        df["date"] = pd.PeriodIndex(df["date"].astype(str), freq="Q").to_timestamp()
    else:
        df["date"] = pd.to_datetime(df["date"].astype(str))
    df["variable"] = variable
    df["sa_source"] = sa_source
    df["source"] = source
    return df[["code", "date", "variable", "value", "sa_source", "source"]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_qna_expenditure(codes: Iterable[str] | None = None, *,
                          refresh: bool = False) -> pd.DataFrame:
    """Fetch quarterly real GDP, real GFCF, and CPI for *codes*.

    Returns a long-form DataFrame with columns:
        code, date, variable, value, sa_source, source

    Variables returned
    ------------------
    NOTE: these are volume *indices* (PRICE_BASE="LR"), not levels, so they are
    not interchangeable with the same-named variables that
    ``oecd_qna_expenditure`` and the local workbook emit in XDC millions — the
    two differ by roughly 9 log points. ``build_panel`` deliberately does not
    use this function for that reason; it is kept for direct/interactive use.

    log_gdp_real  — log of chain-linked volume index for GDP (B1GQ), SA
    log_gfcf_real — log of chain-linked volume index for GFCF (P51G), SA
    log_cpi       — log of total CPI index; SA where OECD publishes it,
                    NSA otherwise (sa_source='none' to trigger X-13 downstream)
    """
    if codes is None:
        # OECD SDMX accepts an empty key segment as "all". Build a fully-wild key.
        code_key = ""
        codes = []  # we'll discover the actual list from the response
    else:
        codes = [c.upper() for c in codes]
        code_key = "+".join(codes)
    frames: list[pd.DataFrame] = []

    # ------------------------------------------------------------------
    # 1.  Real GDP and real GFCF — quarterly SA volume indices
    # ------------------------------------------------------------------
    # Dataflow: OECD.SDD.NAD, DSD_NAMAIN1@DF_QNA_EXPENDITURE_INDICES
    # Key dims (13):
    #   FREQ . ADJUSTMENT . REF_AREA . SECTOR . COUNTERPART_SECTOR .
    #   TRANSACTION . INSTR_ASSET . ACTIVITY . EXPENDITURE .
    #   UNIT_MEASURE . PRICE_BASE . TRANSFORMATION . TABLE_IDENTIFIER
    # We fix: FREQ=Q, ADJUSTMENT=Y, and leave all others wild.
    try:
        agency_flow = "OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_INDICES,"
        key = f"Q.Y.{code_key}............."
        raw = _get_csv(agency_flow, key, "1995", refresh=refresh)
        if raw.empty:
            print("[oecd] QNA indices: no results")
        else:
            # Keep only chain-linked volumes (LR = reference year price base)
            # and total economy sector with no industry split
            # GDP uses ACTIVITY=_Z; GFCF (investment) uses ACTIVITY=_T
            _activity_for = {"B1GQ": "_Z", "P51G": "_T"}
            for var, txn in (("log_gdp_real", "B1GQ"), ("log_gfcf_real", "P51G")):
                act = _activity_for[txn]
                sub = raw[
                    (raw["TRANSACTION"] == txn)
                    & (raw["SECTOR"] == "S1")
                    & (raw["PRICE_BASE"] == "LR")
                    & (raw["ACTIVITY"] == act)
                ].copy()
                if sub.empty:
                    print(f"[oecd] QNA {var}: empty after filtering")
                    continue
                # Deduplicate by code+date (take first if multiple COUNTERPART rows)
                sub = sub.drop_duplicates(subset=["REF_AREA", "TIME_PERIOD"], keep="first")
                sub["log_value"] = np.log(sub["OBS_VALUE"].astype(float))
                long = _long_frame(
                    sub.assign(OBS_VALUE=sub["log_value"]),
                    variable=var,
                    sa_source="oecd",
                    source=f"OECD:DSD_NAMAIN1@DF_QNA_EXPENDITURE_INDICES:{txn}",
                    date_freq="Q",
                )
                frames.append(long)
    except Exception as exc:
        print(f"[oecd] QNA GDP/GFCF fetch failed: {exc}")

    # ------------------------------------------------------------------
    # 2.  CPI — monthly, total economy, SA preferred
    # ------------------------------------------------------------------
    # Dataflow: OECD.SDD.TPS, DSD_PRICES@DF_PRICES_ALL
    # Key dims (8): REF_AREA . FREQ . METHODOLOGY . MEASURE . UNIT_MEASURE .
    #               EXPENDITURE . ADJUSTMENT . TRANSFORMATION
    # We request monthly (M) data and all other dims wild.
    try:
        agency_flow = "OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL,"
        key = f"{code_key}.M........."
        raw_cpi = _get_csv(agency_flow, key, "1995", refresh=refresh)
        if raw_cpi.empty:
            print("[oecd] CPI: no results")
        else:
            # Total economy CPI index
            cpi_tot = raw_cpi[
                (raw_cpi["EXPENDITURE"] == "_T")
                & (raw_cpi["UNIT_MEASURE"] == "IX")
            ].copy()
            cpi_parts: list[pd.DataFrame] = []
            seen_codes = sorted(cpi_tot["REF_AREA"].unique()) if not codes else codes
            for code in seen_codes:
                sub = cpi_tot[cpi_tot["REF_AREA"] == code].copy()
                if sub.empty:
                    print(f"[oecd] CPI: no data for {code}")
                    continue
                sa_available = (sub["ADJUSTMENT"] == "S").any()
                if sa_available:
                    sub = sub[sub["ADJUSTMENT"] == "S"]
                    sa_src = "oecd"
                else:
                    # Use NSA — typical for some emerging markets
                    sub = sub[sub["ADJUSTMENT"] == "N"]
                    sa_src = "none"
                # Use METHODOLOGY=N (national) preferred over HICP
                if "N" in sub["METHODOLOGY"].values:
                    sub = sub[sub["METHODOLOGY"] == "N"]
                # Deduplicate
                sub = sub.drop_duplicates(subset=["REF_AREA", "TIME_PERIOD"], keep="first")
                sub["log_value"] = np.log(sub["OBS_VALUE"].astype(float))
                long = _long_frame(
                    sub.assign(OBS_VALUE=sub["log_value"]),
                    variable="log_cpi",
                    sa_source=sa_src,
                    source="OECD:DSD_PRICES@DF_PRICES_ALL",
                    date_freq="M",
                )
                cpi_parts.append(long)
            if cpi_parts:
                frames.append(pd.concat(cpi_parts, ignore_index=True))
    except Exception as exc:
        print(f"[oecd] CPI fetch failed: {exc}")

    if not frames:
        return pd.DataFrame(columns=["code", "date", "variable", "value", "sa_source", "source"])
    return pd.concat(frames, ignore_index=True)


def fetch_labor_monthly(codes: Iterable[str] | None = None, *,
                        refresh: bool = False) -> pd.DataFrame:
    """Fetch monthly employment (log) and unemployment rate for *codes*.

    Returns a long-form DataFrame with columns:
        code, date, variable, value, sa_source, source

    Variables returned
    ------------------
    log_emp — log of total employment in persons (SA); only for countries where
              OECD publishes monthly LFS employment (e.g. USA yes, DEU quarterly-only)
    urate   — unemployment as % of labour force, ages 15+, total sex, SA

    Note: not all OECD countries have monthly employment.  The panel builder is
    expected to handle gaps (e.g. quarterly interpolation or exclusion).
    """
    if codes is None:
        # OECD SDMX accepts an empty key segment as "all". Build a fully-wild key.
        code_key = ""
    else:
        codes = [c.upper() for c in codes]
        code_key = "+".join(codes)
    frames: list[pd.DataFrame] = []

    # ------------------------------------------------------------------
    # Dataflow: OECD.SDD.TPS, DSD_LFS@DF_IALFS_INDIC
    # Key dims (9): REF_AREA . MEASURE . UNIT_MEASURE . TRANSFORMATION .
    #               ADJUSTMENT . SEX . AGE . ACTIVITY . FREQ
    # ------------------------------------------------------------------
    try:
        agency_flow = "OECD.SDD.TPS,DSD_LFS@DF_IALFS_INDIC,"
        key = f"{code_key}........"
        raw = _get_csv(agency_flow, key, "1990", refresh=refresh)
        if raw.empty:
            print("[oecd] Labor: no results")
            return pd.DataFrame(columns=["code", "date", "variable", "value", "sa_source", "source"])

        # --- Unemployment rate ---
        # MEASURE: UNE_LF (rate as % LF, USA) or UNE_LF_M (DEU and others);
        # both are in UNIT_MEASURE=PT_LF_SUB.  Take either, prefer UNE_LF.
        une_mask = (
            raw["MEASURE"].isin(["UNE_LF", "UNE_LF_M"])
            & (raw["FREQ"] == "M")
            & (raw["ADJUSTMENT"] == "Y")
            & (raw["SEX"] == "_T")
            & (raw["UNIT_MEASURE"] == "PT_LF_SUB")
            & (raw["AGE"] == "Y_GE15")
        )
        une = raw[une_mask].copy()
        if not une.empty:
            # Deduplicate: prefer UNE_LF over UNE_LF_M, then keep first
            une = une.sort_values("MEASURE")  # UNE_LF < UNE_LF_M alphabetically
            une = une.drop_duplicates(subset=["REF_AREA", "TIME_PERIOD"], keep="first")
            long_une = _long_frame(
                une,
                variable="urate",
                sa_source="oecd",
                source="OECD:DSD_LFS@DF_IALFS_INDIC:UNE_LF",
                date_freq="M",
            )
            frames.append(long_une)
        else:
            print("[oecd] Labor: no unemployment rate data found after filtering")

        # --- Employment (log) — monthly only ---
        emp_mask = (
            (raw["MEASURE"] == "EMP")
            & (raw["FREQ"] == "M")
            & (raw["ADJUSTMENT"] == "Y")
            & (raw["SEX"] == "_T")
            & (raw["UNIT_MEASURE"] == "PS")
            & (raw["ACTIVITY"] == "_Z")
        )
        emp = raw[emp_mask].copy()
        if not emp.empty:
            # Prefer broadest age group: Y15T74 > Y_GE15 > _T > others
            age_priority = {"Y15T74": 0, "Y_GE15": 1, "_T": 2}
            emp["_age_pri"] = emp["AGE"].map(age_priority).fillna(99)
            emp = emp.sort_values(["REF_AREA", "TIME_PERIOD", "_age_pri"])
            emp = emp.drop_duplicates(subset=["REF_AREA", "TIME_PERIOD"], keep="first")
            emp["log_value"] = np.log(emp["OBS_VALUE"].astype(float))
            long_emp = _long_frame(
                emp.assign(OBS_VALUE=emp["log_value"]),
                variable="log_emp",
                sa_source="oecd",
                source="OECD:DSD_LFS@DF_IALFS_INDIC:EMP",
                date_freq="M",
            )
            frames.append(long_emp)
        else:
            print("[oecd] Labor: no monthly employment data found after filtering")

    except Exception as exc:
        print(f"[oecd] Labor fetch failed: {exc}")

    if not frames:
        return pd.DataFrame(columns=["code", "date", "variable", "value", "sa_source", "source"])
    return pd.concat(frames, ignore_index=True)
