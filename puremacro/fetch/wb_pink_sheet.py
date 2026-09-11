"""World Bank Pink Sheet — monthly and quarterly commodity benchmarks since 1960.

Source: https://www.worldbank.org/en/research/commodity-markets
Single Excel file (CMO-Historical-Data-Monthly.xlsx).
Sheets:
- 'Monthly Prices': nominal commodity price benchmarks across energy, industrial metals,
  precious metals, agricultural staples, and fertilizers.
- 'Monthly Indices': headline and sub-sector commodity price indices (base 2010=100).

Returns long-form schema with code='WLD' (a single global series per variable):
code, date, variable, value, sa_source, source
"""
from __future__ import annotations

import re
import warnings
from io import BytesIO
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ._http import cached_get

#: The page that always links to the current workbook.
_LANDING = "https://www.worldbank.org/en/research/commodity-markets"

#: Any thedocs.worldbank.org link to the monthly historical workbook.
_XLSX_RE = re.compile(
    r"https://thedocs\.worldbank\.org/[^\"'\s<>\\]*?CMO-Historical-Data-Monthly\.xlsx"
)

#: Last known good document id, used only when landing page cannot be read.
_FALLBACK_URL = (
    "https://thedocs.worldbank.org/en/doc/"
    "5d903e848db1d1b83e0ec8f744e55570-0350012021/related/"
    "CMO-Historical-Data-Monthly.xlsx"
)

#: Warn when the newest observation is older than this.
_STALE_AFTER_DAYS = 185


def _resolve_url(*, refresh: bool = False) -> str:
    """The current workbook URL, from the landing page; fallback if unreadable."""
    try:
        page = cached_get(_LANDING, refresh=refresh, timeout=60)
    except Exception:
        return _FALLBACK_URL
    try:
        text = page.decode("utf-8", errors="ignore")
    except Exception:
        return _FALLBACK_URL
    hit = _XLSX_RE.search(text)
    return hit.group(0) if hit else _FALLBACK_URL


def _warn_if_stale(out: pd.DataFrame) -> None:
    """Say so when the resolved workbook has stopped advancing."""
    if out.empty:
        return
    last = pd.Timestamp(out["date"].max())
    age = (pd.Timestamp.today().normalize() - last).days
    if age > _STALE_AFTER_DAYS:
        warnings.warn(
            f"wb_pink_sheet: the resolved workbook ends at "
            f"{last.date()}, {age} days ago. The World Bank re-issues the Pink "
            f"Sheet under a new document id monthly; this one has stopped "
            f"advancing, so every commodity benchmark below is frozen there "
            f"while merging into the panel as if current. Check "
            f"{_LANDING} for the current file.",
            UserWarning,
            stacklevel=3,
        )


# Column-name substring -> raw monthly price variable name. Match is case-insensitive.
_COL_MAP: dict[str, str] = {
    # Energy
    "crude oil, average": "oil_avg_m",
    "crude oil, brent": "oil_brent_m",
    "crude oil, dubai": "oil_dubai_m",
    "crude oil, wti": "oil_wti_m",
    "coal, australian": "coal_au_m",
    "coal, south african": "coal_za_m",
    "natural gas, us": "gas_us_m",
    "natural gas, europe": "gas_eu_m",
    "liquefied natural gas, japan": "gas_jp_lng_m",
    # Industrial Metals
    "copper": "copper_m",
    "aluminum": "aluminum_m",
    "iron ore, cfr spot": "iron_ore_m",
    "iron ore": "iron_ore_m",
    "lead": "lead_m",
    "tin": "tin_m",
    "nickel": "nickel_m",
    "zinc": "zinc_m",
    # Precious Metals
    "gold": "gold_m",
    "silver": "silver_m",
    "platinum": "platinum_m",
    # Agriculture
    "wheat, us hrw": "wheat_hrw_m",
    "wheat, us srw": "wheat_srw_m",
    "maize": "maize_m",
    "rice, thai 5%": "rice_m",
    "soybeans": "soybeans_m",
    "soybean oil": "soybean_oil_m",
    "soybean meal": "soybean_meal_m",
    "barley": "barley_m",
    # Fertilizers
    "phosphate rock": "phosphate_rock_m",
    "dap": "dap_m",
    "tsp": "tsp_m",
    "urea": "urea_m",
    "potassium chloride": "potassium_chloride_m",
}

# Standardized commodity benchmarks mapping to target canonical names
_STANDARDIZED_PRICE_MAP: dict[str, str] = {
    "oil_brent_m": "brent",
    "oil_wti_m": "wti",
    "gas_us_m": "natgas_us",
    "gas_eu_m": "natgas_eu",
    "coal_au_m": "coal_au",
    "coal_za_m": "coal_za",
    "copper_m": "copper",
    "aluminum_m": "aluminum",
    "iron_ore_m": "iron_ore",
    "gold_m": "gold",
    "silver_m": "silver",
    "wheat_hrw_m": "wheat",
    "wheat_srw_m": "wheat_srw",
    "maize_m": "maize",
    "rice_m": "rice",
    "soybeans_m": "soybeans",
    "phosphate_rock_m": "phosphate_rock",
    "dap_m": "dap",
    "urea_m": "urea",
}

# Sheet 'Monthly Indices' positional column index (1-based) to canonical index name (2010=100)
_INDEX_COL_MAP: dict[int, str] = {
    1: "index_total",
    2: "index_energy",
    3: "index_non_energy",
    4: "index_agri",
    5: "index_beverages",
    6: "index_food",
    7: "index_oils_meals",
    8: "index_grains",
    9: "index_other_food",
    10: "index_raw_materials",
    11: "index_timber",
    12: "index_other_raw_mat",
    13: "index_fert",
    14: "index_metals",
    15: "index_base_metals",
    16: "index_precious_metals",
}

# Canonical categories
COMMODITY_CATEGORIES: dict[str, list[str]] = {
    "energy": ["brent", "wti", "natgas_us", "natgas_eu", "coal_au", "coal_za"],
    "industrial_metals": ["copper", "aluminum", "iron_ore"],
    "precious_metals": ["gold", "silver"],
    "agriculture": ["wheat", "maize", "rice", "soybeans"],
    "fertilizers": ["phosphate_rock", "dap", "urea"],
    "indices": ["index_total", "index_energy", "index_non_energy", "index_agri", "index_metals", "index_fert"],
}

_CATEGORY_ALIASES: dict[str, list[str]] = {
    "metals": ["industrial_metals", "precious_metals"],
    "metal": ["industrial_metals", "precious_metals"],
    "agri": ["agriculture"],
    "fert": ["fertilizers"],
}

_EMPTY = pd.DataFrame(
    columns=["code", "date", "variable", "value", "sa_source", "source"]
)


def _get_workbook_content(
    refresh: bool = False,
    workbook_bytes: bytes | None = None,
    file_path: str | Path | None = None,
) -> bytes | None:
    """Retrieve workbook content from bytes, file path, or cache/remote."""
    if workbook_bytes is not None:
        return workbook_bytes
    if file_path is not None:
        p = Path(file_path)
        if p.exists():
            return p.read_bytes()
    url = _resolve_url(refresh=refresh)
    try:
        return cached_get(url, refresh=refresh, timeout=120)
    except Exception as e:
        print(f"[wb_pink_sheet] download failed: {e}")
        return None


def fetch_prices(
    refresh: bool = False,
    workbook_bytes: bytes | None = None,
    file_path: str | Path | None = None,
) -> pd.DataFrame:
    """Return Pink Sheet monthly commodity prices in long-form schema (code='WLD')."""
    content = _get_workbook_content(
        refresh=refresh, workbook_bytes=workbook_bytes, file_path=file_path
    )
    if content is None or len(content) < 100:
        return _EMPTY.copy()

    try:
        df = pd.read_excel(
            BytesIO(content), sheet_name="Monthly Prices", skiprows=4, header=0
        )
    except Exception as e:
        print(f"[wb_pink_sheet] excel parse failed: {e}")
        return _EMPTY.copy()

    if df.empty:
        return _EMPTY.copy()

    # First col is date (YYYYMMM, e.g. 1960M01). Skip unit row at index 0.
    date_col = df.columns[0]
    df = df.iloc[1:].copy()

    # Coerce date to string and keep only proper YYYYM## rows.
    df[date_col] = df[date_col].astype(str).str.strip()
    df = df[df[date_col].str.match(r"^\d{4}M\d{2}$")]
    df["date"] = pd.to_datetime(df[date_col].str.replace("M", "-") + "-01")

    rows = []
    for col in df.columns:
        if col == date_col or col == "date":
            continue
        col_str = str(col).strip().lower()
        var_name = None
        for needle, name in _COL_MAP.items():
            if needle in col_str:
                var_name = name
                break
        if var_name is None:
            continue
        s = pd.to_numeric(
            df[col].replace({"…": np.nan, "..": np.nan, "...": np.nan, "n.a.": np.nan}),
            errors="coerce",
        )
        sub = pd.DataFrame({
            "code": "WLD",
            "date": df["date"].values,
            "variable": var_name,
            "value": s.values,
            "sa_source": "none",
            "source": "WorldBank:PinkSheet:MonthlyPrices",
        }).dropna(subset=["value"])
        if not sub.empty:
            rows.append(sub)

    if not rows:
        return _EMPTY.copy()

    out = pd.concat(rows, ignore_index=True)
    out = out.drop_duplicates(subset=["variable", "date"], keep="first")
    if workbook_bytes is None and file_path is None:
        _warn_if_stale(out)
    return out


def fetch_indices(
    refresh: bool = False,
    workbook_bytes: bytes | None = None,
    file_path: str | Path | None = None,
) -> pd.DataFrame:
    """Return World Bank Monthly Commodity Price Indices (base 2010=100).

    Parses sheet 'Monthly Indices' covering 16 headline and sub-sector commodity
    price indices in long-form schema: [code='WLD', date, variable, value, sa_source, source].
    """
    content = _get_workbook_content(
        refresh=refresh, workbook_bytes=workbook_bytes, file_path=file_path
    )
    if content is None or len(content) < 100:
        return _EMPTY.copy()

    try:
        df = pd.read_excel(BytesIO(content), sheet_name="Monthly Indices", header=None)
    except Exception as e:
        print(f"[wb_pink_sheet] monthly indices parse failed: {e}")
        return _EMPTY.copy()

    if df.empty or df.shape[1] < 2:
        return _EMPTY.copy()

    # Locate rows where first column matches YYYYM##
    date_mask = df[0].astype(str).str.strip().str.match(r"^\d{4}M\d{2}$")
    df_data = df[date_mask].copy()
    if df_data.empty:
        return _EMPTY.copy()

    dates = pd.to_datetime(df_data[0].astype(str).str.strip().str.replace("M", "-") + "-01")

    rows = []
    for col_idx, var_name in _INDEX_COL_MAP.items():
        if col_idx >= df_data.shape[1]:
            continue
        vals = pd.to_numeric(
            df_data[col_idx].replace({"…": np.nan, "..": np.nan, "...": np.nan, "n.a.": np.nan}),
            errors="coerce",
        )
        sub = pd.DataFrame({
            "code": "WLD",
            "date": dates.values,
            "variable": var_name,
            "value": vals.values,
            "sa_source": "none",
            "source": "WorldBank:PinkSheet:MonthlyIndices",
        }).dropna(subset=["value"])
        if not sub.empty:
            rows.append(sub)

    if not rows:
        return _EMPTY.copy()

    out = pd.concat(rows, ignore_index=True)
    out = out.drop_duplicates(subset=["variable", "date"], keep="first")
    out = out.reset_index(drop=True)
    return out


# Alias fetch to fetch_prices for complete backwards compatibility
fetch = fetch_prices


def fetch_commodity_benchmarks(
    *,
    frequency: str = "M",
    start_date: str = "1960-01-01",
    categories: Sequence[str] | None = None,
    include_indices: bool = True,
    refresh: bool = False,
    workbook_bytes: bytes | None = None,
    file_path: str | Path | None = None,
    variable_suffix: bool | None = None,
) -> pd.DataFrame:
    """Fetch standardized commodity benchmark prices and indices.

    Covers:
    - Energy: brent, wti, natgas_us, natgas_eu, coal_au, coal_za
    - Industrial metals: copper, aluminum, iron_ore
    - Precious metals: gold, silver
    - Agriculture & Fertilizers: wheat, maize, rice, soybeans, phosphate_rock, dap, urea
    - Indices (base 2010=100): index_total, index_energy, index_non_energy, index_agri,
      index_metals, index_fert

    Parameters
    ----------
    frequency : 'M' (monthly) or 'Q' (quarterly). If 'Q', aggregates monthly series
                via quarterly mean rollup (resample('QS').mean()).
    start_date : str, default '1960-01-01'. Filter dates on or after this date.
    categories : sequence of category names (e.g. ['energy', 'industrial_metals',
                 'precious_metals', 'agriculture', 'fertilizers', 'indices']).
                 Aliases 'metals', 'agri', 'fert' are supported.
    include_indices : bool, default True. Whether to include price indices.
    refresh : bool, default False. If True, bypass disk cache.
    workbook_bytes : optional raw bytes of CMO workbook.
    file_path : optional path to offline CMO workbook file.
    variable_suffix : optional bool. If True, appends '_m' or '_q' suffix to variable names.
                      Default: True when frequency=='Q' (e.g. 'brent_q'), False when
                      frequency=='M' (e.g. 'brent').

    Returns
    -------
    pd.DataFrame with schema:
    [code='WLD', date, variable, value, sa_source, source]
    """
    freq = frequency.upper().strip()
    if freq not in {"M", "Q"}:
        raise ValueError(f"frequency must be 'M' or 'Q', got {frequency!r}")

    if variable_suffix is None:
        variable_suffix = True if freq == "Q" else False

    # 1. Fetch monthly prices
    df_prices = fetch_prices(
        refresh=refresh, workbook_bytes=workbook_bytes, file_path=file_path
    )

    frames = []

    if not df_prices.empty:
        # Standardize price variable names
        price_records = []
        for raw_var, std_var in _STANDARDIZED_PRICE_MAP.items():
            sub = df_prices[df_prices["variable"] == raw_var].copy()
            if not sub.empty:
                sub["variable"] = std_var
                price_records.append(sub)
        if price_records:
            frames.append(pd.concat(price_records, ignore_index=True))

    # 2. Fetch monthly indices if requested
    want_indices = include_indices
    if categories is not None:
        norm_cats = [c.lower().strip() for c in categories]
        expanded_cats = set()
        for c in norm_cats:
            if c in _CATEGORY_ALIASES:
                expanded_cats.update(_CATEGORY_ALIASES[c])
            else:
                expanded_cats.add(c)
        if "indices" in expanded_cats:
            want_indices = True
        elif not include_indices:
            want_indices = False

    if want_indices:
        df_indices = fetch_indices(
            refresh=refresh, workbook_bytes=workbook_bytes, file_path=file_path
        )
        if not df_indices.empty:
            frames.append(df_indices)

    if not frames:
        return _EMPTY.copy()

    merged = pd.concat(frames, ignore_index=True)

    # 3. Filter categories if specified
    if categories is not None:
        norm_cats = [c.lower().strip() for c in categories]
        expanded_cats = set()
        for c in norm_cats:
            if c in _CATEGORY_ALIASES:
                expanded_cats.update(_CATEGORY_ALIASES[c])
            else:
                expanded_cats.add(c)

        allowed_vars = set()
        for cat in expanded_cats:
            if cat in COMMODITY_CATEGORIES:
                allowed_vars.update(COMMODITY_CATEGORIES[cat])
        merged = merged[merged["variable"].isin(allowed_vars)]

    # 4. Filter start_date
    start_ts = pd.Timestamp(start_date)
    merged = merged[merged["date"] >= start_ts]

    if merged.empty:
        return _EMPTY.copy()

    # 5. Handle quarterly frequency rollup if requested
    if freq == "Q":
        q_rows = []
        for var, grp in merged.groupby("variable"):
            grp = grp.sort_values("date")
            s = grp.set_index("date")["value"]
            s_q = s.resample("QS").mean().dropna()
            if s_q.empty:
                continue
            var_name = f"{var}_q" if variable_suffix else var
            orig_source = grp["source"].iloc[0] if not grp.empty else "WorldBank:PinkSheet"
            source_q = f"resampled_from_M:{orig_source}"
            sub = pd.DataFrame({
                "code": "WLD",
                "date": s_q.index,
                "variable": var_name,
                "value": s_q.values,
                "sa_source": "none",
                "source": source_q,
            })
            q_rows.append(sub)
        if not q_rows:
            return _EMPTY.copy()
        out = pd.concat(q_rows, ignore_index=True)
    else:
        if variable_suffix:
            merged["variable"] = merged["variable"].apply(
                lambda v: f"{v}_m" if not v.endswith("_m") else v
            )
        out = merged

    out = out.drop_duplicates(subset=["variable", "date"], keep="first")
    out = out.sort_values(["variable", "date"]).reset_index(drop=True)
    return out


__all__ = [
    "fetch",
    "fetch_prices",
    "fetch_indices",
    "fetch_commodity_benchmarks",
    "COMMODITY_CATEGORIES",
]
