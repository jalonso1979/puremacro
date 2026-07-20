"""Geopolitical Risk Index (Caldara-Iacoviello).

Home: https://www.matteoiacoviello.com/gpr.htm

Uses a single source file (data_gpr_export.xls) which contains both:
  - Global GPR (column "GPR", monthly from 1985)
  - Country GPR (columns "GPRC_<ISO3>", monthly from 1985, 44 countries)

The older .xls format requires the ``xlrd`` package.
"""

from __future__ import annotations

from io import BytesIO

import pandas as pd

from ._http import cached_get

_GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"


def fetch(refresh: bool = False) -> pd.DataFrame:
    """Return GPR monthly indices in long form.

    Columns: code, date, variable, value, sa_source, source.
    variable is 'gpr_global_m' for the world index (code='WLD')
    and 'gpr_m' for individual country series (code=ISO-3).
    """
    content = cached_get(_GPR_URL, refresh=refresh)
    try:
        raw = pd.read_excel(BytesIO(content), engine="xlrd")
    except ImportError as exc:
        raise ImportError(
            "Reading GPR .xls file requires xlrd. "
            "Run: pip install 'xlrd>=2.0.1'"
        ) from exc

    # The file has metadata rows before actual data; drop rows where GPR is NaN.
    date_col = "month"
    if date_col not in raw.columns:
        raise ValueError(f"GPR file missing 'month' column. Got: {list(raw.columns)}")

    # Parse dates on a copy to avoid fragmenting the large raw frame.
    dates = pd.to_datetime(raw[date_col])
    data_mask = raw["GPR"].notna()

    # --- Global series ---
    global_df = pd.DataFrame({
        "date": dates[data_mask].values,
        "value": raw.loc[data_mask, "GPR"].values,
    })
    global_df = global_df.rename(columns={"GPR": "value"})
    global_df["code"] = "WLD"
    global_df["variable"] = "gpr_global_m"
    global_df["sa_source"] = "none"
    global_df["source"] = "GPR"

    # --- Country series (GPRC_<ISO3>) ---
    gprc_cols = [c for c in raw.columns if str(c).upper().startswith("GPRC_")]
    if not gprc_cols:
        raise ValueError("GPR file has no GPRC_* country columns.")

    country_raw = pd.concat(
        [dates[data_mask].rename("date"), raw.loc[data_mask, gprc_cols]],
        axis=1,
    )
    long = country_raw.melt(id_vars="date", var_name="col", value_name="value")
    long["code"] = (
        long["col"].astype(str).str.upper().str.replace("GPRC_", "", regex=False)
    )
    # Keep only valid 3-letter ISO codes.
    long = long[long["code"].str.len() == 3].dropna(subset=["value"])
    long["variable"] = "gpr_m"
    long["sa_source"] = "none"
    long["source"] = "GPR"

    country_df = long[["code", "date", "variable", "value", "sa_source", "source"]]
    global_out = global_df[["code", "date", "variable", "value", "sa_source", "source"]]

    return pd.concat([global_out, country_df], ignore_index=True)
