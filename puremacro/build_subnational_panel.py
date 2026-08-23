"""Orchestrator for the US subnational panels.

Produces three artifacts in data/processed/:
    state_panel_M.parquet
    county_panel_Q.parquet
    county_industry_Q.parquet   (long-form, used by bartik/ only)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from puremacro.fetch import qcew, laus, ces_states  # subnational fetchers live in src.fetch (not yet absorbed into puremacro)
from puremacro.fetch.epu_states import fetch as fetch_state_epu
from puremacro.fetch._static_tables import (
    right_to_work_flag,
    tradable_naics2_prefix,
    manuf_naics2_prefix,
)
from .bartik import build_county_epu, build_exposure_iv
from puremacro.regime_dates import assign_regime  # assign_regime not yet absorbed into puremacro.regimes
from .sa import deseasonalize, residual_seasonality_F

logger = logging.getLogger(__name__)


# Variables in the county panel that are published NSA by BLS and that we
# deseasonalize per-county via X-13 ARIMA-SEATS (with STL fallback inside
# ``src.sa.deseasonalize`` when X-13 declines a series).
_QCEW_NSA_COLS = ("emp_qcew_total", "aww_qcew_total", "wages_qcew_total",
                  "estab_qcew_total")
_LAUS_NSA_COLS = ("lf_laus_q", "emp_laus_q", "une_laus_q", "urate_laus_q")


# US Census 2-letter state code → 2-digit FIPS
_STATE_CODE_TO_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06",
    "CO": "08", "CT": "09", "DE": "10", "DC": "11", "FL": "12",
    "GA": "13", "HI": "15", "ID": "16", "IL": "17", "IN": "18",
    "IA": "19", "KS": "20", "KY": "21", "LA": "22", "ME": "23",
    "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28",
    "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
    "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38",
    "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44",
    "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49",
    "VT": "50", "VA": "51", "WA": "53", "WV": "54", "WI": "55",
    "WY": "56",
}


def _normalize_state_epu(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert the fetcher's long-form (state, date, variable, value) schema
    to the (state_fips, date, epu_state_m) schema consumed by the builder.

    The BBD news-based EPU is published unadjusted and is treated as a
    *shock measure*, so it is NOT deseasonalized here. We pass it through
    unmodified.
    """
    if "state_fips" in raw.columns and "epu_state_m" in raw.columns:
        df = raw[["state_fips", "date", "epu_state_m"]].copy()
    else:
        df = raw.copy()
        if "state" in df.columns:
            df["state_fips"] = df["state"].map(_STATE_CODE_TO_FIPS)
            unmapped = df["state_fips"].isna().sum()
            if unmapped > 0:
                logger.warning(
                    "%d rows have unmapped 2-letter state codes (dropped): %s",
                    unmapped, df.loc[df["state_fips"].isna(), "state"].unique()[:10],
                )
                df = df.dropna(subset=["state_fips"])
        if "variable" in df.columns and "value" in df.columns:
            df = df[df["variable"].str.contains("epu", case=False, na=False)]
            df = df.rename(columns={"value": "epu_state_m"})
        df = df[["state_fips", "date", "epu_state_m"]].copy()

    df = df.sort_values(["state_fips", "date"]).reset_index(drop=True)
    # epu_state_m stays raw NSA: uncertainty indices are shock measures.
    return df


def _derive_state_from_county(county_fips: pd.Series) -> pd.Series:
    return county_fips.astype(str).str[:2]


def build_state_monthly_panel(
    *,
    laus_path: str | Path | None = None,
    ces_path: str | Path | None = None,
    epu_path: str | Path | None = None,
    laus_df: pd.DataFrame | None = None,
    ces_df: pd.DataFrame | None = None,
    epu_df: pd.DataFrame | None = None,
    state_births_m: pd.DataFrame | None = None,
    state_births_a: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build monthly state panel: CES supersector 00 + LAUS state + state EPU.

    `state_births_m`: optional (state_fips, date, births_state_m) frame.
    `state_births_a`: optional (state_fips, year, births_state_a) frame —
        broadcast onto every month of the matching year.
    """
    if laus_df is None:
        if laus_path is None:
            raise ValueError("laus_path is required when laus_df is not provided")
        laus_df = laus.parse_laus_series(laus_path, geography="state")
    if ces_df is None:
        if ces_path is None:
            raise ValueError("ces_path is required when ces_df is not provided")
        ces_df = ces_states.parse_ces_states(ces_path)
    if epu_df is None:
        epu_df = pd.read_csv(epu_path, dtype={"state_fips": str}, parse_dates=["date"])

    # CES total (supersector "00"). BLS publishes hrs/ahe only at finer industry
    # levels for many states; emp ('01') is the only universally-populated measure
    # at supersector × industry='000000' aggregation. Missing measures are added
    # as all-NaN columns so downstream code can rely on a stable schema.
    ces_total = ces_df[ces_df["supersector"] == "00"].copy()
    ces_total = ces_total.rename(columns={
        "emp": "emp_ces_total",
        "hrs": "hrs_ces_total",
        "ahe": "ahe_ces_total",
    })
    for col in ("emp_ces_total", "hrs_ces_total", "ahe_ces_total"):
        if col not in ces_total.columns:
            ces_total[col] = np.nan
    ces_total = ces_total[["state_fips", "date", "emp_ces_total",
                           "hrs_ces_total", "ahe_ces_total"]]

    laus_keep = laus_df.rename(columns={
        "lf": "lf_laus",
        "emp": "emp_laus",
        "une": "une_laus",
        "urate": "urate_laus",
    })[["state_fips", "date", "lf_laus", "emp_laus", "une_laus", "urate_laus"]]

    merged = ces_total.merge(laus_keep, on=["state_fips", "date"], how="outer")
    merged = merged.merge(epu_df, on=["state_fips", "date"], how="left")

    missing_epu_states = merged[merged["epu_state_m"].isna()]["state_fips"].nunique()
    if missing_epu_states > 0:
        logger.info(
            "%d state_fips have no EPU coverage in the merged panel",
            missing_epu_states,
        )

    for col in ("emp_ces_total", "ahe_ces_total", "hrs_ces_total", "lf_laus"):
        merged[f"log_{col}"] = np.log(merged[col].replace(0, np.nan))

    if state_births_m is not None and not state_births_m.empty:
        bm = state_births_m[["state_fips", "date", "births_state_m"]].copy()
        bm["state_fips"] = bm["state_fips"].astype(str).str.zfill(2)
        bm["date"] = pd.to_datetime(bm["date"]).dt.to_period("M").dt.to_timestamp(how="start")
        merged["date"] = pd.to_datetime(merged["date"]).dt.to_period("M").dt.to_timestamp(how="start")
        merged = merged.merge(bm, on=["state_fips", "date"], how="left")
        merged["log_births_state_m"] = np.log(merged["births_state_m"].replace(0, np.nan))
    else:
        merged["births_state_m"] = np.nan
        merged["log_births_state_m"] = np.nan

    if state_births_a is not None and not state_births_a.empty:
        ba = state_births_a[["state_fips", "year", "births_state_a"]].copy()
        ba["state_fips"] = ba["state_fips"].astype(str).str.zfill(2)
        merged["year"] = merged["date"].dt.year
        merged = merged.merge(ba, on=["state_fips", "year"], how="left")
        merged = merged.drop(columns=["year"])
        merged["log_births_state_a"] = np.log(merged["births_state_a"].replace(0, np.nan))
    else:
        merged["births_state_a"] = np.nan
        merged["log_births_state_a"] = np.nan

    merged = merged.sort_values(["state_fips", "date"]).reset_index(drop=True)
    merged["regime"] = assign_regime(merged["date"]).values
    return merged


def build_county_quarterly_panel(
    *,
    qcew_path: str | Path | None = None,
    laus_path: str | Path | None = None,
    epu_path: str | Path | None = None,
    shares_path: str | Path | None = None,
    sens_path: str | Path | None = None,
    qcew_df: pd.DataFrame | None = None,
    laus_df: pd.DataFrame | None = None,
    epu_df: pd.DataFrame | None = None,
    shares_df: pd.DataFrame | None = None,
    sens_df: pd.DataFrame | None = None,
    births_df: pd.DataFrame | None = None,
    state_births_q: pd.DataFrame | None = None,
    state_births_a: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build quarterly county panel + county × industry long-form panel."""
    if qcew_df is None:
        if qcew_path is None:
            raise ValueError("qcew_path is required when qcew_df is not provided")
        qcew_df = qcew.parse_qcew_csv(qcew_path)
    if laus_df is None:
        if laus_path is None:
            raise ValueError("laus_path is required when laus_df is not provided")
        laus_df = laus.parse_laus_series(laus_path, geography="county")
    if epu_df is None:
        epu_df = pd.read_csv(epu_path, dtype={"state_fips": str}, parse_dates=["date"])
    if shares_df is None:
        shares_df = pd.read_csv(shares_path, dtype={"county_fips": str, "naics": str})
    if sens_df is None:
        sens_df = pd.read_csv(sens_path, dtype={"naics": str})

    # County-industry long-form (Bartik uses this). Exclude NAICS total '10'.
    ind_panel = qcew_df[qcew_df["naics"] != "10"].copy()

    # County totals (NAICS code '10') -> county x quarter wide.
    totals = qcew_df[qcew_df["naics"] == "10"].copy()
    totals["date"] = pd.to_datetime(
        totals["year"].astype(str) + "-" +
        (totals["qtr"] * 3 - 2).astype(str).str.zfill(2) + "-01"
    )
    totals["state_fips"] = _derive_state_from_county(totals["county_fips"])
    totals = totals.rename(columns={
        "emp_avg": "emp_qcew_total",
        "aww": "aww_qcew_total",
        "total_wages": "wages_qcew_total",
        "estabs": "estab_qcew_total",
    })[["county_fips", "state_fips", "date",
        "emp_qcew_total", "aww_qcew_total", "wages_qcew_total", "estab_qcew_total"]]

    # State EPU -> Q
    epu_df_q = epu_df.copy()
    epu_df_q["period"] = epu_df_q["date"].dt.to_period("Q")
    state_epu_q = (epu_df_q.groupby(["state_fips", "period"], as_index=False)["epu_state_m"]
                            .mean()
                            .rename(columns={"epu_state_m": "epu_state_q"}))
    state_epu_q["date"] = state_epu_q["period"].dt.to_timestamp(how="start")
    state_epu_q = state_epu_q[["state_fips", "date", "epu_state_q"]]
    totals = totals.merge(state_epu_q, on=["state_fips", "date"], how="left")

    # LAUS county -> Q
    laus_q = laus_df.copy()
    laus_q["period"] = laus_q["date"].dt.to_period("Q")
    laus_q = (
        laus_q.groupby(["county_fips", "period"], as_index=False)
              [["lf", "emp", "une", "urate"]]
              .mean()
              .rename(columns={
                  "lf": "lf_laus_q",
                  "emp": "emp_laus_q",
                  "une": "une_laus_q",
                  "urate": "urate_laus_q",
              })
    )
    laus_q["date"] = laus_q["period"].dt.to_timestamp(how="start")
    laus_q = laus_q.drop(columns=["period"])
    totals = totals.merge(laus_q, on=["county_fips", "date"], how="left")

    # Bartik county EPU
    cw = totals[["county_fips", "state_fips"]].drop_duplicates()
    county_epu = build_county_epu(
        shares_df, cw,
        epu_df[["state_fips", "date", "epu_state_m"]],
        freq="Q",
    )[["county_fips", "date", "epu_county_bartik_q"]]
    totals = totals.merge(county_epu, on=["county_fips", "date"], how="left")

    # Scalar exposure
    exposure = build_exposure_iv(shares_df, sens_df)
    totals = totals.merge(exposure, on="county_fips", how="left")

    # Structural covariates for heterogeneity splits
    totals = attach_structural_covariates(totals, shares_df)

    # County × year human births (annual, 1968-2025 mostly):
    #   1968-2000: NCHS via NBER (full pre-1989; 100K+ post-1988).
    #   2001-2009, 2011-2019, 2021-2025: Census PEP (all counties).
    #   2010 / 2020: gaps (Census vintage transitions).
    # See `src/fetch/cdc_births_county.py::_write_combined_artifact`.
    # Forward-filled across the four quarters of each year.
    if births_df is not None and not births_df.empty:
        b = births_df[["county_fips", "year", "births"]].copy()
        b["county_fips"] = b["county_fips"].astype(str).str.zfill(5)
        b = b.rename(columns={"births": "births_county_a"})
        totals["year"] = totals["date"].dt.year
        totals = totals.merge(b, on=["county_fips", "year"], how="left")
        totals = totals.drop(columns=["year"])
    else:
        totals["births_county_a"] = np.nan

    if state_births_q is not None and not state_births_q.empty:
        sbq = state_births_q[["state_fips", "date", "births_state_q"]].copy()
        sbq["state_fips"] = sbq["state_fips"].astype(str).str.zfill(2)
        sbq["date"] = pd.to_datetime(sbq["date"])
        totals = totals.merge(sbq, on=["state_fips", "date"], how="left")
    else:
        totals["births_state_q"] = np.nan

    if state_births_a is not None and not state_births_a.empty:
        sba = state_births_a[["state_fips", "year", "births_state_a"]].copy()
        sba["state_fips"] = sba["state_fips"].astype(str).str.zfill(2)
        totals["year"] = totals["date"].dt.year
        totals = totals.merge(sba, on=["state_fips", "year"], how="left")
        totals = totals.drop(columns=["year"])
    else:
        totals["births_state_a"] = np.nan

    # Seasonal adjustment of NSA series (BLS publishes county QCEW and county
    # LAUS only unadjusted). STL per county_fips, period 4. Counties with <24
    # observations get NaN. We keep raw NSA columns and add `_sa` siblings;
    # `log_*` columns are derived from the SA versions because all downstream
    # LP/SVAR work assumes seasonal noise is removed.
    sa_audit_rows = []
    for col in _QCEW_NSA_COLS + _LAUS_NSA_COLS:
        if col not in totals.columns:
            continue
        sa_col = f"{col}_sa"
        totals[sa_col] = deseasonalize(
            totals, col, by="county_fips", date_col="date", freq="Q",
        )
        # One-shot diagnostic on the largest-county series so the audit isn't
        # 3,000 rows long. We pick the county with the most non-NaN obs.
        cov = totals.groupby("county_fips")[col].count()
        if not cov.empty and cov.max() > 24:
            top = cov.idxmax()
            sub = totals.loc[totals["county_fips"] == top, sa_col]
            F, p, dn, dd = residual_seasonality_F(sub, freq="Q")
            sa_audit_rows.append({
                "scope": "county", "geo": top, "variable": col,
                "method": "stl", "freq": "Q",
                "F_residual_seasonality": F, "p": p,
                "dof_num": dn, "dof_den": dd,
            })

    # Log outcomes are built from the SA versions, with a NSA fallback for
    # counties too short for STL (<24 quarters of obs).
    for col in ("emp_qcew_total", "aww_qcew_total", "wages_qcew_total"):
        sa_col = f"{col}_sa"
        if sa_col in totals.columns:
            src = totals[sa_col].where(totals[sa_col].notna(), totals[col])
        else:
            src = totals[col]
        totals[f"log_{col}"] = np.log(src.replace(0, np.nan))
    if "births_county_a" in totals.columns:
        totals["log_births_county_a"] = np.log(
            totals["births_county_a"].replace(0, np.nan)
        )
    if "births_state_q" in totals.columns:
        totals["log_births_state_q"] = np.log(
            totals["births_state_q"].replace(0, np.nan)
        )
    if "births_state_a" in totals.columns:
        totals["log_births_state_a"] = np.log(
            totals["births_state_a"].replace(0, np.nan)
        )

    # Stash audit on the frame so build_all() can write it to disk.
    totals.attrs["_sa_audit_rows"] = sa_audit_rows

    totals = totals.sort_values(["county_fips", "date"]).reset_index(drop=True)
    totals["regime"] = assign_regime(totals["date"]).values
    return totals, ind_panel


def _compute_county_structural(shares: pd.DataFrame) -> pd.DataFrame:
    """Tradable and manufacturing share from shares panel.

    Uses the first 2 chars of the NAICS code as a NAICS-2 proxy. For SIC-era
    codes (1990-2000 QCEW data) this is approximate — the resulting shares
    are still meaningful as ordinal exposure measures even if not exactly
    matching the modern NAICS taxonomy.
    """
    shares = shares.copy()
    shares["naics2"] = shares["naics"].astype(str).str[:2]
    tradable_set = tradable_naics2_prefix()
    manuf_set = manuf_naics2_prefix()

    tradable = (
        shares[shares["naics2"].isin(tradable_set)]
              .groupby("county_fips")["share"].sum()
              .rename("tradable_share_c")
    )
    manuf = (
        shares[shares["naics2"].isin(manuf_set)]
              .groupby("county_fips")["share"].sum()
              .rename("manuf_share_c")
    )
    out = pd.concat([tradable, manuf], axis=1).reset_index().fillna(0.0)
    return out


def attach_structural_covariates(
    county_df: pd.DataFrame, shares: pd.DataFrame
) -> pd.DataFrame:
    """Attach tradable_share_c, manuf_share_c, right_to_work_state to county_df."""
    structural = _compute_county_structural(shares)
    rtw = right_to_work_flag()
    out = county_df.merge(structural, on="county_fips", how="left")
    out = out.merge(rtw, on="state_fips", how="left")
    out["right_to_work_state"] = out["right_to_work_state"].fillna(0).astype(int)
    out["tradable_share_c"] = out["tradable_share_c"].fillna(0.0)
    out["manuf_share_c"] = out["manuf_share_c"].fillna(0.0)
    return out


def _emit_coverage_report(
    state_df: pd.DataFrame,
    county_df: pd.DataFrame,
    ind_df: pd.DataFrame,
    path: str | Path,
) -> None:
    """Per-year summary of suppression and variable coverage."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 1. QCEW suppression by year
    ind_suppression = (
        ind_df.assign(year=ind_df["year"].astype(int))
              .groupby("year", as_index=False)["suppressed"]
              .mean()
              .rename(columns={"suppressed": "pct_suppressed"})
              .assign(kind="ind_suppression")
    )

    def _nonnull_by_year(df: pd.DataFrame) -> pd.DataFrame:
        df = df.assign(year=df["date"].dt.year)
        # Build per-year non-null fractions explicitly to avoid pandas
        # `groupby().apply()` quirks across versions.
        rows = []
        for yr, group in df.groupby("year"):
            stats = group.drop(columns=["year"]).notna().mean()
            row = stats.to_dict()
            row["year"] = int(yr)
            rows.append(row)
        out = pd.DataFrame(rows)
        # Move 'year' to the front
        cols = ["year"] + [c for c in out.columns if c != "year"]
        return out[cols]

    # 2. State panel non-null coverage by year
    state_nonnull = _nonnull_by_year(state_df)
    state_nonnull["kind"] = "state_nonnull"

    # 3. County panel non-null coverage by year
    county_nonnull = _nonnull_by_year(county_df)
    county_nonnull["kind"] = "county_nonnull"

    combined = pd.concat([ind_suppression, state_nonnull, county_nonnull],
                         axis=0, ignore_index=True)
    combined.to_csv(path, index=False)


def build_all(
    *,
    data_dir: str | Path = "data",
    refresh: bool = False,
) -> dict[str, Path]:
    """End-to-end builder — fetches all network data, writes parquets, returns paths."""
    data_dir = Path(data_dir)
    processed = data_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    state_epu_raw = fetch_state_epu(refresh=refresh)
    state_epu_df = _normalize_state_epu(state_epu_raw)

    # State × month, × quarter, × year births.
    # Monthly: NBER 1968-2002 + Socrata 2023-2024 (gap 2003-2022).
    # Quarterly: derived from monthly above.
    # Yearly: NBER 1968-2002 + Census PEP 2001-2025 (gaps 2010, 2020).
    from .fetch import cdc_births_state
    try:
        state_births_m = cdc_births_state.fetch_monthly()
        state_births_q = cdc_births_state.fetch_quarterly()
        state_births_a = cdc_births_state.fetch_yearly()
    except Exception as e:
        logger.warning("state births fetch failed: %s — building without state births", e)
        state_births_m = state_births_q = state_births_a = None

    state_df = build_state_monthly_panel(
        laus_df=laus.fetch_state(refresh=refresh),
        ces_df=ces_states.fetch(refresh=refresh),
        epu_df=state_epu_df,
        state_births_m=state_births_m,
        state_births_a=state_births_a,
    )
    state_path = processed / "state_panel_M.parquet"
    state_df.to_parquet(state_path)

    shares_path = processed / "bartik_shares_1990.parquet"
    sens_path = processed / "industry_sensitivities.parquet"
    if not shares_path.exists() or not sens_path.exists():
        raise FileNotFoundError(
            f"Bartik artifacts not built yet. Run "
            f"`python tools/build_bartik_artifacts.py` (Task 12) first. "
            f"Missing: {shares_path if not shares_path.exists() else sens_path}"
        )

    # County path loads QCEW year by year
    years = list(range(1990, pd.Timestamp.today().year))
    qcew_df = qcew.fetch(years, refresh=refresh)

    # County × year births: combined NCHS (1968-2000) + Census PEP
    # (2001-2025 minus vintage-base years). Read from the combined
    # canonical CSV at Fertility/data/PROCESSED/county_year_births.csv,
    # which is (re)built when cdc_births_county.fetch() runs.
    try:
        # Inside the try: on a build without `requests` this import used to raise
        # straight out of build_all rather than degrading like the fetch it guards.
        from .fetch import cdc_births_county
        cdc_births_county.fetch(refresh=refresh)  # rebuilds canonical artifact
        from .fetch.cdc_births_county import _COMBINED_ARTIFACT
        if _COMBINED_ARTIFACT.exists():
            combined = pd.read_csv(_COMBINED_ARTIFACT, dtype={"county_fips": str})
            births_df = combined.rename(columns={
                "births": "births",
            })[["county_fips", "year", "births"]]
            if "source" in combined.columns:
                # Keep source for diagnostics but not for the merge.
                births_df = births_df.assign(source=combined["source"])
        else:
            births_df = None
    except Exception as e:
        logger.warning("county births fetch failed: %s — building without births", e)
        births_df = None

    county_df, ind_df = build_county_quarterly_panel(
        qcew_df=qcew_df,
        laus_df=laus.fetch_county(refresh=refresh),
        epu_df=state_epu_df,
        shares_df=pd.read_parquet(shares_path),
        sens_df=pd.read_parquet(sens_path),
        births_df=births_df,
        state_births_q=state_births_q,
        state_births_a=state_births_a,
    )
    county_path = processed / "county_panel_Q.parquet"
    ind_path = processed / "county_industry_Q.parquet"
    county_df.to_parquet(county_path)
    ind_df.to_parquet(ind_path)

    # Append the SA audit accumulated by build_county_quarterly_panel.
    sa_audit_path = processed / "sa_audit.csv"
    audit_rows = county_df.attrs.get("_sa_audit_rows", [])
    if audit_rows:
        new_audit = pd.DataFrame(audit_rows)
        if sa_audit_path.exists():
            old = pd.read_csv(sa_audit_path)
            combined_audit = pd.concat([old, new_audit], axis=0, ignore_index=True)
        else:
            combined_audit = new_audit
        combined_audit.to_csv(sa_audit_path, index=False)

    coverage_path = processed / "subnational_coverage.csv"
    _emit_coverage_report(state_df, county_df, ind_df, coverage_path)

    return {
        "state": state_path,
        "county": county_path,
        "industry": ind_path,
        "coverage": coverage_path,
        "sa_audit": sa_audit_path,
    }
