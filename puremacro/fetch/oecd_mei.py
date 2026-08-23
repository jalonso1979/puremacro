"""OECD STES (Short-Term Economic Statistics) fetcher.

Three monthly variables across ~50 OECD members from 1990 onward:

- ``log_ip``  -- log industrial production index, SA, dataflow
                ``OECD.SDD.STES,DSD_STES@DF_INDSERV,4.3`` with
                ``MEASURE=PRVM`` (production volume) at
                ``ACTIVITY=BTE`` (industry except construction).
- ``bci_m``   -- business confidence indicator (amplitude-adjusted),
                dataflow ``OECD.SDD.STES,DSD_STES@DF_CLI`` with
                ``MEASURE=BCICP``.
- ``cci_m``   -- consumer confidence indicator (amplitude-adjusted),
                dataflow ``OECD.SDD.STES,DSD_STES@DF_CLI`` with
                ``MEASURE=CCICP``.

OECD restructured SDMX in late 2024; the prior dataflows
``DSD_STES@DF_INDICATORS`` and ``DSD_STES@DF_BCICP`` now return 404.
The dimensional structure (9-dim key for indicators, identical for CLI)
is unchanged but MEASURE / ACTIVITY / ADJUSTMENT codes were updated:

* IP: PRMNTO01 → PRVM × ACTIVITY=BTE, ADJUSTMENT remains Y.
* BCI/CCI: ADJUSTMENT changed from Y to AA (amplitude adjusted).

The network helper is the same pattern as ``src.fetch.oecd._get_csv``;
duplicated here to keep this module independent of internal helpers.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


_EMPTY = pd.DataFrame(
    columns=["code", "date", "variable", "value", "sa_source", "source"]
)


def _get_csv(agency_flow: str, key: str, start_period: str) -> pd.DataFrame:
    """Fetch one OECD SDMX URL via cached helper. Empty on persistent failure."""
    from ._oecd_sdmx import get_sdmx_csv
    return get_sdmx_csv(agency_flow, key, start_period)


def _to_long(raw: pd.DataFrame, variable: str, *, log: bool, sa: bool, source: str) -> pd.DataFrame:
    if raw.empty:
        return _EMPTY.copy()
    df = raw[["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]].copy()
    df.columns = ["code", "date", "value"]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df["date"] = pd.to_datetime(df["date"].astype(str))
    if log:
        df = df[df["value"] > 0]
        df["value"] = np.log(df["value"])
    df["variable"] = variable
    df["sa_source"] = "oecd" if sa else "none"
    df["source"] = source
    return df[["code", "date", "variable", "value", "sa_source", "source"]]


def fetch(codes: Iterable[str] | None = None, *, start_period: str = "1990") -> pd.DataFrame:
    """Return monthly log_ip + bci_m + cci_m for *codes* (None -> all available)."""
    code_key = "" if codes is None else "+".join([c.upper() for c in codes])
    frames: list[pd.DataFrame] = []

    # 1. Industrial production: PRVM at ACTIVITY=BTE (industry except construction),
    #    monthly, calendar+seasonally adjusted (ADJUSTMENT=Y), unit Index (IX).
    try:
        agency = "OECD.SDD.STES,DSD_STES@DF_INDSERV,4.3"
        # 9-dim key: REF_AREA . FREQ . MEASURE . UNIT_MEASURE . ACTIVITY .
        #            ADJUSTMENT . TRANSFORMATION . TIME_HORIZ . METHODOLOGY
        # Fix REF_AREA, FREQ=M, MEASURE=PRVM, UNIT_MEASURE=IX, ACTIVITY=BTE,
        # ADJUSTMENT=Y; others wild.
        key = f"{code_key}.M.PRVM.IX.BTE.Y...."
        raw = _get_csv(agency, key, start_period)
        if not raw.empty:
            # Server already constrains by key; the filter is a defensive guard
            # against future schema additions (extra MEASURE codes etc.).
            sub = raw.copy()
            if "MEASURE" in sub.columns:
                sub = sub[sub["MEASURE"] == "PRVM"]
            if "ACTIVITY" in sub.columns:
                sub = sub[sub["ACTIVITY"] == "BTE"]
            if "ADJUSTMENT" in sub.columns:
                sub = sub[sub["ADJUSTMENT"] == "Y"]
            if "FREQ" in sub.columns:
                sub = sub[sub["FREQ"] == "M"]
            sub = sub.drop_duplicates(subset=["REF_AREA", "TIME_PERIOD"], keep="first")
            frames.append(_to_long(
                sub, variable="log_ip", log=True, sa=True,
                source="OECD:DSD_STES@DF_INDSERV:PRVM:BTE",
            ))
    except Exception as exc:
        print(f"[oecd_mei] IP fetch failed: {exc}")

    # 2. Business / consumer confidence (BCICP / CCICP) live in DSD_STES@DF_CLI.
    #    Amplitude-adjusted index (ADJUSTMENT=AA, UNIT_MEASURE=IX).
    try:
        agency = "OECD.SDD.STES,DSD_STES@DF_CLI,"
        # 9-dim key — same shape as DSD_STES@DF_INDSERV. MEASURE wild so we
        # fetch BCICP and CCICP in one call.
        key = f"{code_key}.M..IX..AA...."
        raw = _get_csv(agency, key, start_period)
        if not raw.empty:
            for measure, var in (("BCICP", "bci_m"), ("CCICP", "cci_m")):
                sub = raw[raw["MEASURE"] == measure].copy()
                if sub.empty:
                    continue
                if "FREQ" in sub.columns:
                    sub = sub[sub["FREQ"] == "M"]
                sub = sub.drop_duplicates(subset=["REF_AREA", "TIME_PERIOD"], keep="first")
                frames.append(_to_long(
                    sub, variable=var, log=False, sa=True,
                    source=f"OECD:DSD_STES@DF_CLI:{measure}",
                ))
    except Exception as exc:
        print(f"[oecd_mei] BCI/CCI fetch failed: {exc}")

    if not frames:
        return _EMPTY.copy()
    return pd.concat(frames, ignore_index=True)


__all__ = ["fetch"]
