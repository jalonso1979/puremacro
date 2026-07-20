"""Regenerate the frozen data behind notebook 14 (tax multiplier three ways).

Two snapshots, both written to ``puremacro/replication/data/`` (shipped as
package data via the existing ``"puremacro.replication.data" = ["*.csv"]``
glob in pyproject.toml):

1. ``tax14_us_fiscal.csv`` — quarterly US fiscal aggregates from the public,
   key-free FRED fredgraph CSV endpoint (via :func:`puremacro.fetch.fetch_fred`).
   Exact FRED series ids:

   - ``GDPC1``            Real GDP, chained dollars, SAAR, quarterly.
   - ``GDP``              Nominal GDP, SAAR, quarterly.
   - ``GDPDEF``           GDP implicit price deflator (index), quarterly.
   - ``W006RC1Q027SBEA``  Federal government current tax receipts, nominal,
                          SAAR, quarterly (BP's "taxes" concept, simplified —
                          Blanchard-Perotti use net taxes = receipts minus
                          transfers; we state that simplification in the
                          notebook).
   - ``FGEXPND``          Federal government current expenditures, nominal,
                          SAAR, quarterly.
   - ``CPIAUCSL``         CPI all urban (monthly, aggregated to quarterly
                          means) — the alternative deflator swept in the
                          notebook's specification curve.

2. ``tax14_narrative_tax_shocks.csv`` — the narrative tax-shock measures,
   quarterly, in percent of GDP, positive = legislated tax increase:

   - ``rr_exog``  Romer-Romer (2010 AER) exogenous tax changes
                  (deficit-driven + long-run motives).
   - ``mtr_u``    Mertens-Ravn unanticipated tax changes (their RED 2012
                  vintage of the narrative measure used as the proxy in
                  Mertens-Ravn 2013 AER).
   - ``mtr_a``    Mertens-Ravn anticipated tax changes (same vintage).

   Source: Valerie Ramey's public "Macroeconomic Shocks and Their
   Propagation" (Handbook of Macroeconomics, 2016) tax archive,
   https://econweb.ucsd.edu/~vramey/research/Ramey_HOM_tax.zip
   (sheet ``homtaxdat``, columns EXOGENRRATIO / TAXU / taxnews). This
   snapshot exists because the GitHub mirror hard-coded in
   ``puremacro.narrative.replication.{romer_romer_2010,mertens_ravn_2013}``
   is dead (404); the notebook passes ``csv_path=`` pointing here instead.
   The column names match what those loaders' ``*_csv_to_events`` helpers
   expect (``date`` + ``rr_exog`` / ``mtr_u`` / ``mtr_a``).

Dev-only: needs network + openpyxl (for the xlsx). Run from the package dir:

    python tools/gen_notebook_data_tax14.py
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

from puremacro.fetch import fetch_fred
from puremacro.fetch._classic import _safe_urlopen

RAMEY_HOM_TAX_ZIP = "https://econweb.ucsd.edu/~vramey/research/Ramey_HOM_tax.zip"
DATA_DIR = (
    Path(__file__).resolve().parents[1] / "puremacro" / "replication" / "data"
)


def build_fiscal() -> Path:
    quarterly = {
        "gdpc1": fetch_fred("GDPC1"),
        "gdp": fetch_fred("GDP"),
        "gdpdef": fetch_fred("GDPDEF"),
        "fedtax": fetch_fred("W006RC1Q027SBEA"),
        "fedspend": fetch_fred("FGEXPND"),
    }
    # CPI is monthly; aggregate to quarterly means aligned to quarter starts.
    cpi_q = fetch_fred("CPIAUCSL").dropna().resample("QS").mean().round(4)
    out = pd.concat({**quarterly, "cpi": cpi_q}, axis=1).dropna().reset_index()
    out.columns = ["date"] + list(quarterly.keys()) + ["cpi"]
    dest = DATA_DIR / "tax14_us_fiscal.csv"
    out.to_csv(dest, index=False)
    print(f"wrote {dest} ({len(out)} quarterly obs, "
          f"{out['date'].min().date()}..{out['date'].max().date()})")
    return dest


def build_narrative() -> Path:
    raw = _safe_urlopen(RAMEY_HOM_TAX_ZIP, timeout=120.0)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        with zf.open("homtaxdat.xlsx") as fh:
            df = pd.read_excel(fh, sheet_name="homtaxdat")
    # 'quarter' is a float (1948.25 = 1948Q2). Convert to 'YYYYQq' strings,
    # which pd.Period(s, freq='Q') — used by the loaders — parses directly.
    year = df["quarter"].astype(int)
    q = ((df["quarter"] - year) * 4).round().astype(int) + 1
    out = pd.DataFrame({
        "date": [f"{y}Q{k}" for y, k in zip(year, q)],
        "rr_exog": df["EXOGENRRATIO"].fillna(0.0).round(6),
        "mtr_u": df["TAXU"].fillna(0.0).round(6),
        "mtr_a": df["taxnews"].fillna(0.0).round(6),
    })
    # RR's narrative record ends in 2007; drop the padding zeros after it.
    out = out[out["date"] <= "2007Q4"].reset_index(drop=True)
    dest = DATA_DIR / "tax14_narrative_tax_shocks.csv"
    out.to_csv(dest, index=False)
    nz = (out[["rr_exog", "mtr_u", "mtr_a"]] != 0).sum()
    print(f"wrote {dest} ({len(out)} quarters, nonzero events: "
          f"rr_exog={nz['rr_exog']}, mtr_u={nz['mtr_u']}, mtr_a={nz['mtr_a']})")
    return dest


def main() -> None:
    build_fiscal()
    build_narrative()


if __name__ == "__main__":
    main()
