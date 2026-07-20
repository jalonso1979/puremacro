"""Regenerate puremacro/replication/data/kilian2009.csv from FRED (dev-only).

Freezes the monthly series behind the Kilian (2009 AER) oil-market VAR
replication (``puremacro.examples.kilian_2009_oil``):

- ``IPG211111S`` — Industrial Production: Mining: Crude Oil (US, SA index).
  *Documented proxy*: Kilian's world crude-oil production series (EIA) is not
  served by the public no-key fredgraph endpoint (candidate world-production
  ids 404), so we freeze the US crude-oil production index instead. The
  discontinuation of IPG211111S in 2021 is irrelevant here — the frozen
  sample ends 2007-12, Kilian's sample end.
- ``IGREA``      — Kilian's own index of global real economic activity
  (corrected 2019 vintage, as published by FRED).
- ``WTISPLC``    — Spot crude oil price: WTI. *Documented fallback*: the
  refiners' acquisition cost of imported crude used by Kilian (RACIM-style
  ids) 404s on fredgraph, so WTI is used, deflated by CPI in the example.
- ``CPIAUCSL``   — CPI, all urban consumers, SA (deflator).

Sample frozen at 1971-01..2007-12: one year of pre-sample cushion for growth
rates plus Kilian's estimation sample (1973:2-2007:12 in the paper).

Uses the public no-API-key fredgraph CSV endpoint via
:func:`puremacro.fetch.fetch_fred` — no FRED_API_KEY needed. Run from the
package dir:

    python tools/gen_replication_data_kilian2009.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from puremacro.fetch import fetch_fred


_START = "1971-01-01"
_END = "2007-12-01"


def main() -> None:
    cols = {
        "oilprod": fetch_fred("IPG211111S").dropna(),
        "igrea": fetch_fred("IGREA").dropna(),
        "wti": fetch_fred("WTISPLC").dropna(),
        "cpi": fetch_fred("CPIAUCSL").dropna(),
    }
    out = pd.concat(cols, axis=1).loc[_START:_END].dropna().reset_index()
    out.columns = ["date", "oilprod", "igrea", "wti", "cpi"]
    dest = (
        Path(__file__).resolve().parents[1]
        / "puremacro" / "replication" / "data" / "kilian2009.csv"
    )
    out.to_csv(dest, index=False)
    print(f"wrote {dest} ({len(out)} monthly obs, "
          f"{out['date'].min().date()}..{out['date'].max().date()})")


if __name__ == "__main__":
    main()
