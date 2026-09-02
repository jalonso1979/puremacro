"""G9 homogeneous-vintage long panel 1975–2024 — splice loader.

A 20-year rolling K-N long-difference bridge regression runs
on a single homogeneous panel covering
1975–2024 across nine OECD economies (USA, JPN, DEU, FRA, GBR, ITA,
CAN, NLD, AUS).

This module contains the constituent loaders and the splicer that joins
them into ``data/processed/long_panel_1975_2024_g9.csv``:

* :func:`load_pwt10` — Penn World Table 10.0 for relative-investment
  prices via ``log(pl_i / pl_c)``.
* :func:`load_oecd_stan_ls` — OECD-STAN labor share 1970–present.
* :func:`load_klems_legacy` — EU-KLEMS 2008/2009 release labor share
  for the 1975–2007 back-fill.
* :func:`build_g9_long_panel` — splicer + write CSV.
* :func:`splice_audit` — per-country MAD in the overlap window.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from puremacro._http import safe_get_bytes

# Repo root. This file is at `<repo>/puremacro/long_panel.py`, so the root is
# parents[1]. It read parents[2] — correct back when the package was nested as
# `puremacro/puremacro/`, and one level too high ever since the split, which
# pointed every data path at whatever directory happens to sit above the
# checkout. The failure is invisible on a machine where that directory exists.
_ROOT = Path(__file__).resolve().parents[1]

G9 = ("USA", "JPN", "DEU", "FRA", "GBR", "ITA", "CAN", "NLD", "AUS")


def _pwt10_path() -> Path:
    return _ROOT / "data" / "raw" / "pwt10" / "pwt100.dta"


def load_pwt10(
    *,
    countries: Iterable[str] | None = None,
    start: int = 1970,
    end: int = 2019,
) -> pd.DataFrame:
    """Load PWT 10.0 relative-investment-price series and labor share.

    Returns a long DataFrame with columns
    ``[code, year, pl_i, pl_c, log_relprice_equip, labsh, log_LS_pwt]``.

    ``labsh`` is the PWT-published labor share — Feenstra-Inklaar-Timmer
    methodology, includes self-employment imputation. K-N (2014, QJE)
    used PWT's labor share as their primary LS measure, so PWT-`labsh`
    on the long panel is methodologically homogeneous with K-N's headline
    series. ``log_LS_pwt = log(labsh)``.

    PWT 10.0 stops in 2019 by construction (the 2023 release covers
    1950–2019).

    Raises
    ------
    FileNotFoundError
        If ``data/raw/pwt10/pwt100.dta`` is absent. Tests should skip
        cleanly via ``pytest.skip`` when this happens.
    """
    path = _pwt10_path()
    if not path.exists():
        raise FileNotFoundError(
            f"PWT 10.0 .dta not found at {path}. Download pwt100.dta from "
            "https://www.rug.nl/ggdc/productivity/pwt/ and place it there; "
            "it is not redistributable, so it is not shipped with the repo."
        )
    df = pd.read_stata(
        path, columns=["countrycode", "year", "pl_i", "pl_c", "labsh"]
    )
    df = df.rename(columns={"countrycode": "code"})
    df["year"] = df["year"].astype(int)
    df = df[(df["year"] >= start) & (df["year"] <= end)]
    if countries is not None:
        df = df[df["code"].isin(set(countries))]
    df = df.copy()
    # Relative-equipment-price column (requires both price levels).
    pk_ok = (
        df["pl_i"].notna() & df["pl_c"].notna()
        & (df["pl_i"] > 0) & (df["pl_c"] > 0)
    )
    df["log_relprice_equip"] = np.where(
        pk_ok, np.log(df["pl_i"]) - np.log(df["pl_c"]), np.nan
    )
    # Labor-share column (requires labsh > 0).
    ls_ok = df["labsh"].notna() & (df["labsh"] > 0)
    df["log_LS_pwt"] = np.where(ls_ok, np.log(df["labsh"]), np.nan)
    df = df.sort_values(["code", "year"]).reset_index(drop=True)
    return df[
        ["code", "year", "pl_i", "pl_c", "log_relprice_equip",
         "labsh", "log_LS_pwt"]
    ]


# OECD-STAN modern SDMX endpoint. Dataflow IDs change across vintages;
# see https://sdmx.oecd.org/public/rest/dataflow/OECD.STD.NAD/all/all
# for the current registry. The constants below target the STAN i4
# (ISIC4) 2020 release; if it has been retired, the loader returns
# empty (handled by callers via skip-on-empty).
_OECD_STAN_URL = (
    "https://sdmx.oecd.org/public/rest/data/"
    "OECD.STD.NAD,DSD_STAN_ARCH@DF_STAN_I4_2020,1.0/"
    "{country}.{var}.{agg}.A"
    "/?dimensionAtObservation=AllDimensions"
    "&format=csvfilewithlabels"
)
_STAN_AGG_CANDIDATES = ("DTOTAL", "TOTAL")
_STAN_LABR_VAR = "LABR"
_STAN_VALU_VAR = "VALU"


def _fetch_stan_csv(country: str, var: str, agg: str, *, timeout: float = 30.0) -> pd.DataFrame:
    """Fetch one (country, var, aggregation) STAN series via OECD SDMX.

    Returns a DataFrame with columns ``[year, value]`` on success, or
    empty on any network/parse failure.
    """
    url = _OECD_STAN_URL.format(country=country, var=var, agg=agg)
    try:
        raw = safe_get_bytes(url, timeout=timeout)
    except Exception:
        return pd.DataFrame(columns=["year", "value"])
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=["year", "value"])
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        return pd.DataFrame(columns=["year", "value"])
    # OECD CSV format: look for time/period column + value column.
    time_col = next((c for c in df.columns if c.upper() in {"TIME_PERIOD", "TIME", "OBS_TIME"}), None)
    val_col = next((c for c in df.columns if c.upper() in {"OBS_VALUE", "VALUE"}), None)
    if time_col is None or val_col is None:
        return pd.DataFrame(columns=["year", "value"])
    out = pd.DataFrame({
        "year": pd.to_numeric(df[time_col], errors="coerce"),
        "value": pd.to_numeric(df[val_col], errors="coerce"),
    }).dropna()
    out["year"] = out["year"].astype(int)
    return out


def load_oecd_stan_ls(
    *,
    countries: Iterable[str] | None = None,
    start: int = 1970,
    end: int = 2019,
) -> pd.DataFrame:
    """Load OECD-STAN labor share = LABR / VALU at total economy.

    Returns ``[code, year, log_LS_stan]``. Empty for countries where
    the SDMX endpoint returns no data (network failure, retired
    dataflow, missing series). Callers should treat an empty return
    as "no coverage" rather than as an error.

    The G9 long-panel splicer (Task 4) uses EU-KLEMS as the primary
    LS source and falls back to STAN where KLEMS doesn't cover a
    (country, year). STAN failures are non-fatal for the bridge figure.
    """
    if countries is None:
        countries = G9
    rows = []
    for code in countries:
        labr = pd.DataFrame()
        valu = pd.DataFrame()
        for agg in _STAN_AGG_CANDIDATES:
            labr_try = _fetch_stan_csv(code, _STAN_LABR_VAR, agg)
            valu_try = _fetch_stan_csv(code, _STAN_VALU_VAR, agg)
            if len(labr_try) > 0 and len(valu_try) > 0:
                labr, valu = labr_try, valu_try
                break
        if len(labr) == 0 or len(valu) == 0:
            continue
        merged = labr.rename(columns={"value": "labr"}).merge(
            valu.rename(columns={"value": "valu"}), on="year", how="inner"
        )
        merged = merged[(merged["year"] >= start) & (merged["year"] <= end)]
        merged = merged[(merged["labr"] > 0) & (merged["valu"] > 0)]
        if len(merged) == 0:
            continue
        merged["log_LS_stan"] = np.log(merged["labr"]) - np.log(merged["valu"])
        merged["code"] = code
        rows.append(merged[["code", "year", "log_LS_stan"]])
    if not rows:
        return pd.DataFrame(columns=["code", "year", "log_LS_stan"])
    return pd.concat(rows, ignore_index=True).sort_values(["code", "year"]).reset_index(drop=True)


def _klems_legacy_dir() -> Path:
    return _ROOT / "data" / "raw" / "euklems_legacy"


def load_klems_legacy(
    *,
    countries: Iterable[str] | None = None,
    start: int = 1970,
    end: int = 2007,
) -> pd.DataFrame:
    """Load EU-KLEMS 2008/2009-release labor share for the total economy.

    Reads ``<ISO3>_output_08I.xls`` per country from
    ``data/raw/euklems_legacy/``. Returns ``[code, year, log_LS_klems_leg]``.

    Silently returns empty if the archive is absent or unreadable —
    callers should treat that as "no coverage" and fall back to
    OECD-STAN or EU-KLEMS 2023.
    """
    if countries is None:
        countries = G9
    archive = _klems_legacy_dir()
    if not archive.exists():
        return pd.DataFrame(columns=["code", "year", "log_LS_klems_leg"])
    rows = []
    for code in countries:
        path = archive / f"{code}_output_08I.xls"
        if not path.exists():
            continue
        try:
            sheet = pd.read_excel(path, sheet_name="TOT", engine="xlrd")
        except Exception:
            continue
        # KLEMS-legacy 'TOT' sheet: variable name in column 0, years as columns.
        sheet = sheet.rename(columns={sheet.columns[0]: "var"})
        years = [c for c in sheet.columns if isinstance(c, (int, float))]
        if not years:
            continue
        long = sheet.melt(id_vars="var", value_vars=years,
                          var_name="year", value_name="value")
        long["year"] = long["year"].astype(int)
        long = long[(long["year"] >= start) & (long["year"] <= end)]
        labr = long[long["var"].astype(str).str.upper() == "LAB"].set_index("year")["value"]
        valu = long[long["var"].astype(str).str.upper() == "VA"].set_index("year")["value"]
        df = pd.concat({"labr": labr, "valu": valu}, axis=1).dropna()
        df = df[(df["labr"] > 0) & (df["valu"] > 0)]
        if len(df) == 0:
            continue
        df["log_LS_klems_leg"] = np.log(df["labr"]) - np.log(df["valu"])
        df = df.reset_index().rename(columns={"index": "year"})
        df["code"] = code
        rows.append(df[["code", "year", "log_LS_klems_leg"]])
    if not rows:
        return pd.DataFrame(columns=["code", "year", "log_LS_klems_leg"])
    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["code", "year"])
        .reset_index(drop=True)
    )


def build_g9_long_panel(
    *,
    start: int = 1975,
    end: int = 2024,
    countries: Iterable[str] = G9,
    write_csv: bool = True,
) -> pd.DataFrame:
    """Splice PWT 10.0 + OECD-STAN + EU-KLEMS legacy + EU-KLEMS 2023 into a
    single homogeneous-vintage long panel for the bridge figure.

    Splice rule for ``log_LS`` (priority order, first non-NaN wins):
        0. PWT 10.0 ``labsh`` (1950–2019, homogeneous Feenstra-Inklaar-Timmer
           methodology, K-N (2014) used the same source — this is the
           bridge-figure headline LS).
        1. EU-KLEMS 2023 (1995/2008–2024 in the G9 cohort).
        2. OECD-STAN (1970–present where covered).
        3. EU-KLEMS legacy 2008/2009 release (1970–2007).

    Splice rule for ``log_relprice_equip``:
        0. PWT 10.0 (1970–2019, homogeneous through the period).
        For 2020–2024 the relative price is left as NaN; constructing it
        from KLEMS-2023 requires a separate consumer-price denominator
        (OECD-QNA infrastructure) that is out of scope for Sprint 1.

    The ``vintage_LS`` and ``vintage_PK`` columns flag which source
    supplied each (code, year) cell, for the splice audit.
    """
    # ---- log_relprice_equip + log_LS_pwt: PWT 10.0 ----
    pk = pd.DataFrame(columns=["code", "year", "log_relprice_equip", "vintage_PK"])
    ls_pwt = pd.DataFrame(columns=["code", "year", "log_LS", "vintage_LS"])
    try:
        pwt = load_pwt10(countries=list(countries), start=start, end=2019)
        if len(pwt) > 0:
            pk = pwt.dropna(subset=["log_relprice_equip"])[
                ["code", "year", "log_relprice_equip"]
            ].copy()
            pk["vintage_PK"] = "pwt10"
            ls_pwt_src = pwt.dropna(subset=["log_LS_pwt"])[
                ["code", "year", "log_LS_pwt"]
            ].copy()
            ls_pwt_src = ls_pwt_src.rename(columns={"log_LS_pwt": "log_LS"})
            ls_pwt_src["vintage_LS"] = "pwt10_labsh"
            ls_pwt = ls_pwt_src
    except FileNotFoundError:
        pass

    # ---- log_LS: KLEMS-2023 (priority 1) ----
    ls_klems2023 = pd.DataFrame(columns=["code", "year", "log_LS", "vintage_LS"])
    try:
        from puremacro.klems import load_klems_panel
        cache_dir = _ROOT / "data" / "raw" / "euklems"
        kdf = load_klems_panel(cache_dir=cache_dir, industry="TOT", equip_def="tangible",
                               include_investment=False)
        if len(kdf) > 0:
            sub = kdf.dropna(subset=["comp_total", "va"]).copy()
            sub = sub[(sub["comp_total"] > 0) & (sub["va"] > 0)]
            sub["log_LS"] = np.log(sub["comp_total"]) - np.log(sub["va"])
            sub["vintage_LS"] = "klems2023"
            ls_klems2023 = sub[["code", "year", "log_LS", "vintage_LS"]]
    except Exception:
        pass

    # ---- log_LS: OECD-STAN (priority 2) ----
    ls_stan = load_oecd_stan_ls(
        countries=list(countries), start=start, end=end
    ).rename(columns={"log_LS_stan": "log_LS"})
    if len(ls_stan) > 0:
        ls_stan = ls_stan.copy()
        ls_stan["vintage_LS"] = "oecd_stan"
    else:
        ls_stan = pd.DataFrame(columns=["code", "year", "log_LS", "vintage_LS"])

    # ---- log_LS: EU-KLEMS legacy (priority 3) ----
    ls_klems_leg = load_klems_legacy(
        countries=list(countries), start=start, end=2007
    ).rename(columns={"log_LS_klems_leg": "log_LS"})
    if len(ls_klems_leg) > 0:
        ls_klems_leg = ls_klems_leg.copy()
        ls_klems_leg["vintage_LS"] = "klems_legacy_08"
    else:
        ls_klems_leg = pd.DataFrame(columns=["code", "year", "log_LS", "vintage_LS"])

    # Concatenate, filling NaN where one source provides a column the others don't.
    sources = [s for s in [ls_pwt, ls_klems2023, ls_stan, ls_klems_leg] if len(s) > 0]
    ls_all = pd.concat(sources, ignore_index=True) if sources else pd.DataFrame(columns=["code", "year", "log_LS", "vintage_LS"])
    if len(ls_all) > 0:
        priority = {
            "pwt10_labsh": 0,
            "klems2023": 1,
            "oecd_stan": 2,
            "klems_legacy_08": 3,
        }
        ls_all["pri"] = ls_all["vintage_LS"].map(priority)
        ls_all = (
            ls_all.sort_values(["code", "year", "pri"])
            .drop_duplicates(["code", "year"], keep="first")
            .drop(columns="pri")
        )

    # ---- merge LS + PK on (code, year) outer ----
    if len(ls_all) > 0 and len(pk) > 0:
        panel = ls_all.merge(pk, on=["code", "year"], how="outer")
    elif len(ls_all) > 0:
        panel = ls_all.copy()
        panel["log_relprice_equip"] = np.nan
        panel["vintage_PK"] = pd.NA
    elif len(pk) > 0:
        panel = pk.copy()
        panel["log_LS"] = np.nan
        panel["vintage_LS"] = pd.NA
    else:
        panel = pd.DataFrame(
            columns=["code", "year", "log_LS", "vintage_LS",
                     "log_relprice_equip", "vintage_PK"]
        )

    if len(panel) > 0:
        panel = panel[(panel["year"] >= start) & (panel["year"] <= end)]
        panel = panel[panel["code"].isin(set(countries))]
        panel = panel.sort_values(["code", "year"]).reset_index(drop=True)

    if write_csv and len(panel) > 0:
        out_path = _ROOT / "data" / "processed" / "long_panel_1975_2024_g9.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(out_path, index=False)

    return panel


def splice_audit(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-country mean-absolute-deviation in the LS splice overlap window.

    Compares OECD-STAN against EU-KLEMS legacy in years where both
    sources cover the same (country, year). Returns
    ``[code, overlap_n, mad_LS_pp, mad_PK_pp]``. Empty / all-NaN when
    both auxiliary sources are unavailable on disk.

    KLEMS-2023 is the trusted modern reference; we don't audit STAN
    vs KLEMS-2023 separately because the body's bridge figure prefers
    KLEMS-2023 for that period anyway.
    """
    if len(panel) == 0:
        return pd.DataFrame(columns=["code", "overlap_n", "mad_LS_pp", "mad_PK_pp"])
    rows = []
    for code, g in panel.groupby("code"):
        ymin, ymax = int(g["year"].min()), int(g["year"].max())
        stan = load_oecd_stan_ls(countries=[code], start=ymin, end=ymax)
        klems_leg = load_klems_legacy(countries=[code], start=ymin, end=ymax)
        if len(stan) == 0 or len(klems_leg) == 0:
            rows.append({"code": code, "overlap_n": 0,
                         "mad_LS_pp": np.nan, "mad_PK_pp": np.nan})
            continue
        merged = stan.merge(klems_leg, on=["code", "year"], how="inner")
        n = len(merged)
        if n == 0:
            rows.append({"code": code, "overlap_n": 0,
                         "mad_LS_pp": np.nan, "mad_PK_pp": np.nan})
            continue
        ls_stan = np.exp(merged["log_LS_stan"])
        ls_legacy = np.exp(merged["log_LS_klems_leg"])
        mad_ls_pp = float(np.mean(np.abs(ls_stan - ls_legacy)) * 100.0)
        rows.append({"code": code, "overlap_n": n,
                     "mad_LS_pp": mad_ls_pp, "mad_PK_pp": np.nan})
    return pd.DataFrame(rows)
