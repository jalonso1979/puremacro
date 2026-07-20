"""tools/build_sw07_data.py — produce the bundled SW07 dataset.

Run once by the maintainer with FRED_API_KEY in the environment; the
resulting CSV gets committed to puremacro/dsge/_sw07_data.csv.
End-users invoking estimate_sw07() load that committed CSV, not this
script.

Series sourced from FRED (Smets-Wouters 2007 Table 2 specification):

    gdp        GDPC1     Real GDP, chained 2017 dollars
    cons       PCECC96   Real personal consumption expenditures
    inv        GPDIC1    Real gross private domestic investment
    hours      HOANBS    Nonfarm business hours (per-capita demeaned)
    wage_nom   COMPNFB   Nonfarm business compensation
    deflator   GDPDEF    GDP implicit price deflator
    ffr        FEDFUNDS  Federal funds rate (effective)
    pop        CNP16OV   Civilian non-institutional population
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from puremacro.fetch.fred import fetch_series


SERIES = {
    "gdp":      "GDPC1",
    "cons":     "PCECC96",
    "inv":      "GPDIC1",
    "hours":    "HOANBS",
    "wage_nom": "COMPNFB",
    "deflator": "GDPDEF",
    "ffr":      "FEDFUNDS",
    "pop":      "CNP16OV",
}

START = "1965-10-01"  # need one lag for log-diff at 1966Q1
END = "2004-12-31"


def _to_period_index(s: pd.Series) -> pd.Series:
    """Convert a DatetimeIndex series to a PeriodIndex (quarterly) for alignment."""
    s = s.copy()
    s.index = pd.PeriodIndex(s.index, freq="Q")
    return s


def main(out_path: Path) -> None:
    raw: dict[str, pd.Series] = {}
    for name, fred_id in SERIES.items():
        df_long = fetch_series(fred_id, variable=name, code="USA")
        s = df_long.set_index("date")["value"]
        s.index = pd.to_datetime(s.index)
        s = s[(s.index >= START) & (s.index <= END)]
        # Resample to quarterly, then convert to PeriodIndex so all series align
        # regardless of whether the source uses QS-OCT, QE, or MS timestamps.
        # SW07 convention: average within quarter for FFR; quarterly aggregates
        # already published as quarterly for GDPC1/PCECC96/GPDIC1; monthly for FFR/CNP16OV.
        if name in ("ffr", "pop"):
            s = s.resample("QE").mean()
        elif name == "hours":
            # Hours is quarterly already; resample last to get QE alignment.
            s = s.resample("QE").last()
        else:
            # Quarterly level series (GDPC1, PCECC96, etc.) arrive as QS-OCT;
            # resample with .last() so they land on quarter-end too.
            s = s.resample("QE").last()
        raw[name] = _to_period_index(s.dropna())

    df = pd.concat(raw, axis=1).dropna()
    # Per-capita transformations.
    for nm in ("gdp", "cons", "inv", "hours"):
        df[nm + "_pc"] = df[nm] / df["pop"]
    df["wage_real_pc"] = df["wage_nom"] / df["deflator"]

    out = pd.DataFrame(index=df.index)
    out["gdp_growth"]   = 100 * np.log(df["gdp_pc"] / df["gdp_pc"].shift(1))
    out["cons_growth"]  = 100 * np.log(df["cons_pc"] / df["cons_pc"].shift(1))
    out["inv_growth"]   = 100 * np.log(df["inv_pc"] / df["inv_pc"].shift(1))
    out["wage_growth"]  = 100 * np.log(df["wage_real_pc"] / df["wage_real_pc"].shift(1))
    out["log_hours"]    = np.log(df["hours_pc"])
    out["log_hours"]   -= out["log_hours"].mean()
    out["infl"]         = 100 * np.log(df["deflator"] / df["deflator"].shift(1))
    out["ffr"]          = df["ffr"] / 4.0

    out = out.dropna()
    # Filter to 1966Q1–2004Q4 on the PeriodIndex.
    out = out[(out.index >= pd.Period("1966Q1", freq="Q")) & (out.index <= pd.Period("2004Q4", freq="Q"))]

    header = [
        "# puremacro SW07 dataset — 1966Q1 to 2004Q4 (155 quarterly obs)",
        "# Source FRED series:",
    ]
    for nm, fred_id in SERIES.items():
        header.append(f"#   {nm}: {fred_id}")
    header.append("# Transformations: per-capita, 100*log-diff; hours demeaned; FFR quarterly")
    header.append(f"# Built: {pd.Timestamp.utcnow().strftime('%Y-%m-%d')}")
    header.append("# Reference: Smets & Wouters (2007), AER")

    with out_path.open("w") as fh:
        for line in header:
            fh.write(line + "\n")
        out.to_csv(fh, index_label="date")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parent.parent / "puremacro" / "dsge" / "_sw07_data.csv"
    )
    main(out)
    print(f"wrote {out}")
