"""Realized volatility per country from daily stock-index returns (Yahoo Finance).

Annualized realized vol:
    RV_t = sqrt(252 * sum_{d in t} r_d^2)
where r_d = log(P_d) - log(P_{d-1}) and t is the month or quarter.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

# `yfinance` is imported lazily inside fetch_realized_vol so that
# `import puremacro.fetch.yahoo` stays Pyodide-clean: yfinance pulls in bs4 and
# needs network, so importing this module must not require it (only calling the
# fetcher does).

#: ISO-3 -> primary stock index ticker on Yahoo Finance.
#: Checked against the v8 chart endpoint on 6 Oct 2026 (``interval=1mo``,
#: ``range=max``). Six tickers were replaced because the old ones answer 404
#: or carry a single observation: NOR ^OSEAX -> OSEBX.OL (2013-03), GRC
#: ATHEX.AT -> GD.AT (1997-06), PRT ^PSI20 -> PSI20.LS (2013-04), TUR ^XU100 ->
#: XU100.IS (1997-06), SWE ^OMXS30 -> ^OMX (2008-11) and ZAF ^JN0U.JO (a USD
#: index from 2017) -> ^J203.JO (FTSE/JSE All Share, ZAR, 2012-02). Countries
#: with no working free ticker are listed in :data:`DEAD_TICKERS` instead.
ISO3_TO_INDEX = {
    "USA": "^GSPC",    "CAN": "^GSPTSE", "MEX": "^MXX",
    "GBR": "^FTSE",    "DEU": "^GDAXI",  "FRA": "^FCHI",
    "ITA": "FTSEMIB.MI","ESP": "^IBEX",  "NLD": "^AEX",
    "CHE": "^SSMI",    "SWE": "^OMX",    "NOR": "OSEBX.OL",
    "FIN": "^OMXH25",  "DNK": "^OMXC25", "AUT": "^ATX",
    "BEL": "^BFX",     "GRC": "GD.AT",   "IRL": "^ISEQ",
    "PRT": "PSI20.LS", "TUR": "XU100.IS",
    "JPN": "^N225",    "KOR": "^KS11",   "HKG": "^HSI",
    "SGP": "^STI",     "AUS": "^AXJO",   "NZL": "^NZ50",
    "IND": "^BSESN",   "IDN": "^JKSE",   "MYS": "^KLSE",
    "CHN": "000001.SS",
    "BRA": "^BVSP",    "ARG": "^MERV",
    "ZAF": "^J203.JO", "ISR": "^TA125.TA",
}

#: Tickers that used to be in :data:`ISO3_TO_INDEX` and no longer return a
#: usable monthly history on Yahoo (404, or a single bar), as of 6 Oct 2026.
#: Kept so a caller asking for these countries is told why they are absent.
DEAD_TICKERS = {
    "POL": "^WIG20",   "CZE": "^PX",      "HUN": "^BUX",
    "RUS": "IMOEX.ME", "THA": "^SET.BK",  "PHL": "PSEI.PS",
    "CHL": "^IPSA",    "COL": "^COLCAP",  "PER": "^SPBLPGPT",
}


def fetch_realized_vol(codes: Iterable[str] | None = None, *, freq: str = "M", start: str = "1990-01-01") -> pd.DataFrame:
    import yfinance as yf  # lazy: keeps `import ...fetch.yahoo` Pyodide-clean (see top of file)

    if freq not in ("M", "Q"):
        raise ValueError("freq must be 'M' or 'Q'")
    if codes is None:
        codes = list(ISO3_TO_INDEX.keys())
    frames = []
    for code in codes:
        ticker = ISO3_TO_INDEX.get(code)
        if not ticker:
            continue
        try:
            px = yf.download(ticker, start=start, progress=False, auto_adjust=True)["Close"]
        except Exception as e:  # noqa: BLE001
            print(f"[yahoo] {code} ({ticker}) failed: {e}")
            continue
        if px.empty:
            continue
        # yfinance sometimes returns a DataFrame when multiple tickers; coerce to Series.
        if isinstance(px, pd.DataFrame):
            px = px.iloc[:, 0]
        ret = np.log(px).diff().dropna()
        sq = ret ** 2
        rule = "ME" if freq == "M" else "QE"  # pandas 2.2+ prefers M/Q-end rules
        rv = np.sqrt(252 * sq.resample(rule).sum())
        # Align to period start for consistency with other monthly/quarterly series.
        rv.index = rv.index.to_period(freq).to_timestamp()
        out = rv.reset_index()
        out.columns = ["date", "value"]
        out["code"] = code
        out["variable"] = f"rv_{freq.lower()}"
        out["sa_source"] = "none"
        out["source"] = f"Yahoo:{ticker}"
        frames.append(out[["code", "date", "variable", "value", "sa_source", "source"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["code", "date", "variable", "value", "sa_source", "source"]
    )
