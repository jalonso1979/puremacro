"""OECD QNA expenditure fetcher: real GDP + expenditure components.

Pulls from OECD SDMX dataflow ``DSD_NAMAIN1@DF_QNA_EXPENDITURE_NATIO_CURR`` with:

  ADJUSTMENT  = Y (calendar + seasonally adjusted) preferred, else N (NSA, x13_pending)
  SECTOR      = S1 (total economy)
  PRICE_BASE  = L (chain-linked volume; the ``real`` series the panel uses)
  UNIT_MEASURE= XDC (national currency, in millions)

Returns long-form DataFrame ``code, date, variable, value, sa_source, source``
with ``value = log(OBS_VALUE)`` so the panel-level convention (log-real
expenditure) is preserved.

TRANSACTION → variable mapping mirrors the local Volatility/QNA.xlsx workbook:

  B1GQ  → log_gdp_real         (Gross domestic product, expenditure approach)
  P51G  → log_gfcf_real        (Gross fixed capital formation)
  P3    → log_con_real         (Final consumption expenditure)
  P32   → log_govcon_real      (Government final consumption)
  P6    → log_exports_real     (Exports of goods and services)
  P7    → log_imports_real     (Imports of goods and services)

Used by ``src.build_panel`` to fill expenditure-side coverage gaps for
countries not present in the curated local Volatility/QNA.xlsx workbook
(typically ~30+ additional OECD members + accession countries).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from ._oecd_sdmx import get_sdmx_csv

_EMPTY = pd.DataFrame(columns=["code", "date", "variable", "value", "sa_source", "source"])

# (transaction, sector) → output variable name. P3 needs sector disambiguation:
# P3 + S1 (total economy) is total final consumption; P3 + S13 (general
# government) is government final consumption — the dataflow does not expose
# the alternative P32 code, so we have to filter by institutional sector.
_TRANSACTION_TO_VAR: dict[tuple[str, str], str] = {
    ("B1GQ", "S1"):  "log_gdp_real",
    ("P51G", "S1"):  "log_gfcf_real",
    ("P3",   "S1"):  "log_con_real",
    ("P3",   "S13"): "log_govcon_real",
    ("P6",   "S1"):  "log_exports_real",
    ("P7",   "S1"):  "log_imports_real",
}

_TRANSACTIONS_NEEDED = {t for (t, _) in _TRANSACTION_TO_VAR}
_SECTORS_NEEDED = {s for (_, s) in _TRANSACTION_TO_VAR}


def _quarter_to_date(s: pd.Series) -> pd.Series:
    return pd.PeriodIndex(s.astype(str), freq="Q").to_timestamp(how="start")


def fetch_qna_expenditure(codes: Iterable[str] | None = None,
                          *, start_period: str = "1995") -> pd.DataFrame:
    """Fetch real expenditure components for the given country codes.

    Returns a long-form DataFrame with rows for each variable in
    ``_TRANSACTION_TO_VAR`` per (code, date) pair available. Empty frame
    on any error or for codes with no data.

    Strategy: one SDMX request per chunk of ~10 countries (the OECD
    endpoint truncates very wide REF_AREA filters). All transactions /
    sectors / price bases come back in the same response; we filter
    post-hoc.
    """
    if codes is None:
        code_chunks: list[list[str] | None] = [None]
    else:
        codes = [c.upper() for c in codes if c]
        code_chunks = [codes[i:i + 10] for i in range(0, len(codes), 10)] or [[]]

    agency_flow = "OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_NATIO_CURR,"

    parts: list[pd.DataFrame] = []
    for chunk in code_chunks:
        if chunk is None:
            code_key = ""
        elif not chunk:
            continue
        else:
            code_key = "+".join(chunk)
        # 13 dims: FREQ . ADJUSTMENT . REF_AREA . SECTOR . COUNTERPART_SECTOR .
        #          TRANSACTION . INSTR_ASSET . ACTIVITY . EXPENDITURE .
        #          UNIT_MEASURE . PRICE_BASE . TRANSFORMATION . TABLE_IDENTIFIER
        # We constrain FREQ=Q + REF_AREA; everything else floats and we filter
        # post-hoc. Two requests are needed because P32 (govt consumption) is
        # exposed as P3 with SECTOR=S13 rather than as a distinct transaction.
        for sector in sorted(_SECTORS_NEEDED):
            key = f"Q..{code_key}.{sector}.........."
            raw = get_sdmx_csv(agency_flow, key, start_period)
            if raw.empty:
                continue
            parts.append(raw)

    if not parts:
        return _EMPTY.copy()
    raw = pd.concat(parts, ignore_index=True)

    required_cols = {"TRANSACTION", "PRICE_BASE", "UNIT_MEASURE", "ADJUSTMENT",
                     "REF_AREA", "TIME_PERIOD", "OBS_VALUE", "SECTOR"}
    if not required_cols.issubset(set(raw.columns)):
        return _EMPTY.copy()

    # Filter to chain-linked volume in national currency.
    base = raw[(raw["TRANSACTION"].isin(_TRANSACTIONS_NEEDED))
               & (raw["PRICE_BASE"] == "L")
               & (raw["UNIT_MEASURE"] == "XDC")
               & (raw["SECTOR"].isin(_SECTORS_NEEDED))].copy()
    if base.empty:
        return _EMPTY.copy()

    # If ACTIVITY column is present, prefer total-economy rows. The
    # expenditure flow typically uses ACTIVITY=_T or _Z; both refer to total.
    if "ACTIVITY" in base.columns:
        base = base[base["ACTIVITY"].isin(["_T", "_Z"])]

    out_parts: list[pd.DataFrame] = []
    for (txn, sector), var in _TRANSACTION_TO_VAR.items():
        sub_t = base[(base["TRANSACTION"] == txn) & (base["SECTOR"] == sector)]
        if sub_t.empty:
            continue

        # Prefer SA (ADJUSTMENT=Y) over NSA per code.
        sub_y = sub_t[sub_t["ADJUSTMENT"] == "Y"]
        sub_n = sub_t[sub_t["ADJUSTMENT"] == "N"]
        codes_with_sa = set(sub_y["REF_AREA"].unique())
        sub_n = sub_n[~sub_n["REF_AREA"].isin(codes_with_sa)]

        for sa_label, sub in (("oecd", sub_y), ("x13_pending", sub_n)):
            if sub.empty:
                continue
            sub = sub.copy()
            sub["OBS_VALUE"] = pd.to_numeric(sub["OBS_VALUE"], errors="coerce")
            sub = sub.dropna(subset=["OBS_VALUE"])
            sub = sub[sub["OBS_VALUE"] > 0]
            if sub.empty:
                continue
            sub = sub.drop_duplicates(subset=["REF_AREA", "TIME_PERIOD"], keep="first")
            sub["date"] = _quarter_to_date(sub["TIME_PERIOD"])
            sub["value"] = np.log(sub["OBS_VALUE"].astype(float))
            sub["code"] = sub["REF_AREA"]
            sub["variable"] = var
            sub["sa_source"] = sa_label
            sub["source"] = f"OECD:DSD_NAMAIN1@DF_QNA_EXPENDITURE_NATIO_CURR:{txn}/{sector}/L/XDC"
            out_parts.append(sub[["code", "date", "variable", "value", "sa_source", "source"]])

    if not out_parts:
        return _EMPTY.copy()
    return pd.concat(out_parts, ignore_index=True)


__all__ = ["fetch_qna_expenditure"]
