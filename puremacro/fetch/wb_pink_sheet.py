"""World Bank Pink Sheet — monthly and quarterly commodity benchmarks since 1960.

Source: https://www.worldbank.org/en/research/commodity-markets
Single Excel file (CMO-Historical-Data-Monthly.xlsx).
Sheets:
- 'Monthly Prices': nominal commodity price benchmarks across energy, industrial metals,
  precious metals, agricultural staples, and fertilizers.
- 'Monthly Indices': headline and sub-sector commodity price indices (base 2010=100).

Returns long-form schema with code='WLD' (a single global series per variable):
code, date, variable, value, sa_source, source

The workbook is read with the standard library alone (``zipfile`` plus
``xml.etree``, see :func:`_read_xlsx_sheet`), so neither ``openpyxl`` nor
``requests`` is needed and the module runs on a tablet with only numpy and
pandas. Downloads go through :func:`puremacro._http.safe_get_bytes_cached`
(urllib, SQLite cache outside the repository): nothing is written into the
package tree.
"""
from __future__ import annotations

import re
import warnings
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Sequence
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

from puremacro._http import safe_get_bytes_cached

#: Seconds the landing page is trusted before it is re-read for a newer link.
#: The World Bank re-issues the workbook under a new document id every month.
_LANDING_TTL = 7 * 24 * 3600

#: Seconds a downloaded workbook is trusted (``refresh=True`` bypasses it).
_WORKBOOK_TTL = 30 * 24 * 3600


def cached_get(url: str, *, refresh: bool = False, timeout: float = 60) -> bytes:
    """Fetch ``url`` through the house urllib cache; raise on failure.

    This module-level name is the seam the tests patch. It keeps the
    ``(url, *, refresh, timeout)`` shape of the ``requests``-based helper it
    replaces, so callers and fixtures did not change.
    """
    ttl = _LANDING_TTL if url == _LANDING else _WORKBOOK_TTL
    return safe_get_bytes_cached(
        url, timeout,
        headers={"Accept-Encoding": "gzip"},
        ttl_seconds=ttl,
        rate_limit_seconds=1.0,
        refresh=refresh,
    )


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
    except (ValueError, ArithmeticError, np.linalg.LinAlgError, OSError, Exception):
        return _FALLBACK_URL
    try:
        text = page.decode("utf-8", errors="ignore")
    except (ValueError, ArithmeticError, np.linalg.LinAlgError, Exception):
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


_XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_PKG_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_CELL_REF = re.compile(r"^([A-Z]+)(\d+)$")


def _col_index(letters: str) -> int:
    """``"A"`` -> 0, ``"Z"`` -> 25, ``"AA"`` -> 26."""
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def _text_of(node) -> str:
    """All ``<t>`` text under ``node`` (rich-text runs are concatenated)."""
    return "".join(t.text or "" for t in node.iter(_XLSX_NS + "t"))


def _read_xlsx_sheet(content: bytes, sheet_name: str) -> list[list]:
    """Return one worksheet of an ``.xlsx`` as a dense list of rows.

    Row ``i`` of the result is spreadsheet row ``i + 1`` and column ``j`` is
    column letter ``j`` (A = 0), so positions match ``pd.read_excel(...,
    header=None)``. Numbers come back as ``float``, text as ``str`` (shared,
    inline and formula strings alike), empty cells as ``None``. Only the
    standard library is used.

    Raises ``KeyError`` when the sheet does not exist and
    ``zipfile.BadZipFile`` / ``ET.ParseError`` on a damaged file; the callers
    turn those into the empty schema.
    """
    with zipfile.ZipFile(BytesIO(content)) as zf:
        names = set(zf.namelist())
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        rid = None
        for sh in wb.iter(_XLSX_NS + "sheet"):
            if sh.get("name", "").strip() == sheet_name:
                rid = sh.get(_REL_NS + "id")
                break
        if rid is None:
            raise KeyError(f"no sheet named {sheet_name!r}")
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        target = None
        for rel in rels.iter(_PKG_REL_NS + "Relationship"):
            if rel.get("Id") == rid:
                target = rel.get("Target", "")
                break
        if not target:
            raise KeyError(f"sheet {sheet_name!r} has no part")
        path = target.lstrip("/") if target.startswith("/") else "xl/" + target
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            sst = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            shared = [_text_of(si) for si in sst.iter(_XLSX_NS + "si")]
        sheet = ET.fromstring(zf.read(path))

    rows: dict[int, dict[int, object]] = {}
    next_row = 0
    for row in sheet.iter(_XLSX_NS + "row"):
        r_attr = row.get("r")
        r = int(r_attr) - 1 if r_attr else next_row
        next_row = r + 1
        cells: dict[int, object] = {}
        next_col = 0
        for c in row.iter(_XLSX_NS + "c"):
            ref = c.get("r")
            m = _CELL_REF.match(ref) if ref else None
            j = _col_index(m.group(1)) if m else next_col
            next_col = j + 1
            kind = c.get("t", "n")
            v = c.find(_XLSX_NS + "v")
            if kind == "inlineStr":
                is_ = c.find(_XLSX_NS + "is")
                value: object = _text_of(is_) if is_ is not None else None
            elif v is None or v.text is None:
                value = None
            elif kind == "s":
                value = shared[int(v.text)]
            elif kind in ("str", "e"):
                value = v.text
            elif kind == "b":
                value = float(v.text)
            else:
                try:
                    value = float(v.text)
                except ValueError:
                    value = v.text
            if value is not None:
                cells[j] = value
        if cells:
            rows[r] = cells
    if not rows:
        return []
    width = max(max(cells) for cells in rows.values()) + 1
    out: list[list] = []
    for r in range(max(rows) + 1):
        cells = rows.get(r, {})
        out.append([cells.get(j) for j in range(width)])
    return out


_PERIOD_RE = re.compile(r"^\d{4}M\d{2}$")


def _is_period(x: object) -> bool:
    return isinstance(x, str) and bool(_PERIOD_RE.match(x.strip()))


def _price_table(content: bytes) -> tuple[list[str], list[str], list[list]]:
    """Header, units and data rows of the 'Monthly Prices' sheet.

    The data rows are those whose first cell reads ``YYYYMmm``; the header is
    two rows above the first of them and the units row sits in between,
    which is the layout of every edition since 2010 (title block, header,
    units, data).
    """
    grid = _read_xlsx_sheet(content, "Monthly Prices")
    first = next((i for i, row in enumerate(grid) if row and _is_period(row[0])), None)
    if first is None or first < 2:
        return [], [], []
    header = ["" if x is None else str(x) for x in grid[first - 2]]
    units = ["" if x is None else str(x) for x in grid[first - 1]]
    data = [row for row in grid[first:] if row and _is_period(row[0])]
    return header, units, data


def _match_price_column(header: str) -> str | None:
    """Raw variable name for a 'Monthly Prices' header, or ``None``.

    A needle must start a word: "tin" used to match inside "Platinum" and
    platinum was silently dropped as a duplicate of tin.
    """
    col = header.strip().lower()
    for needle, name in _COL_MAP.items():
        if re.search(r"(?<![a-z])" + re.escape(needle), col):
            return name
    return None


def _to_float(values: Sequence[object]) -> np.ndarray:
    return pd.to_numeric(pd.Series(list(values), dtype=object), errors="coerce").to_numpy(dtype=float)


def _price_units(content: bytes) -> dict[str, str]:
    """Raw variable name -> unit label from the workbook's units row."""
    header, units, _ = _price_table(content)
    out: dict[str, str] = {}
    for h, u in zip(header, units):
        name = _match_price_column(h)
        if name and name not in out:
            out[name] = u.strip().strip("()")
    return out


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
    except Exception as e:  # noqa: BLE001 - any provider failure -> empty schema
        warnings.warn(f"wb_pink_sheet: download failed ({type(e).__name__}: {e})",
                      UserWarning, stacklevel=3)
        return None


def _clean_numeric(values: Sequence[object]) -> np.ndarray:
    """Floats, with the workbook's '…', '..', 'n.a.' placeholders as NaN."""
    return _to_float(values)


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
        header, _units, data = _price_table(content)
    except Exception as e:  # noqa: BLE001 - damaged or unexpected workbook
        warnings.warn(f"wb_pink_sheet: 'Monthly Prices' parse failed ({type(e).__name__}: {e})",
                      UserWarning, stacklevel=2)
        return _EMPTY.copy()
    if not data:
        return _EMPTY.copy()

    dates = pd.to_datetime([str(row[0]).strip().replace("M", "-") + "-01" for row in data])

    rows = []
    for j, col in enumerate(header):
        if j == 0:
            continue
        var_name = _match_price_column(col)
        if var_name is None:
            continue
        vals = _clean_numeric([row[j] if j < len(row) else None for row in data])
        sub = pd.DataFrame({
            "code": "WLD",
            "date": dates,
            "variable": var_name,
            "value": vals,
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
        grid = _read_xlsx_sheet(content, "Monthly Indices")
    except Exception as e:  # noqa: BLE001 - damaged or unexpected workbook
        warnings.warn(f"wb_pink_sheet: 'Monthly Indices' parse failed ({type(e).__name__}: {e})",
                      UserWarning, stacklevel=2)
        return _EMPTY.copy()

    data = [row for row in grid if row and _is_period(row[0])]
    if not data or len(data[0]) < 2:
        return _EMPTY.copy()

    dates = pd.to_datetime([str(row[0]).strip().replace("M", "-") + "-01" for row in data])

    rows = []
    for col_idx, var_name in _INDEX_COL_MAP.items():
        if col_idx >= len(data[0]):
            continue
        vals = _clean_numeric([row[col_idx] for row in data])
        sub = pd.DataFrame({
            "code": "WLD",
            "date": dates,
            "variable": var_name,
            "value": vals,
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
