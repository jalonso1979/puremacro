"""Regenerate puremacro/replication/data/gali1999.csv from FRED (dev-only).

Freezes the three public series behind the Gali (1999 AER) technology-shock
replication (``puremacro.examples.gali_1999_hours``):

- ``OPHNFB``  — Nonfarm Business Sector: Labor Productivity (output per hour),
                quarterly index.
- ``HOANBS``  — Nonfarm Business Sector: Hours Worked for All Workers,
                quarterly index.
- ``CNP16OV`` — Civilian Noninstitutional Population 16+, monthly (thousands),
                aggregated here to quarterly means.

Uses the public no-API-key fredgraph CSV endpoint via
:func:`puremacro.fetch.fetch_fred` — no FRED_API_KEY needed (unlike the okun
generator). Run from the package dir:

    python tools/gen_replication_data_gali1999.py

The frozen snapshot is what ships in the wheel; the example and its tests run
offline from it.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from puremacro.fetch import fetch_fred


def main() -> None:
    ophnfb = fetch_fred("OPHNFB").dropna()
    hoanbs = fetch_fred("HOANBS").dropna()
    pop_m = fetch_fred("CNP16OV").dropna()
    # Monthly population -> quarterly mean, aligned to quarter starts like
    # the quarterly BLS series.
    pop_q = pop_m.resample("QS").mean().round(4)

    out = pd.concat(
        {"ophnfb": ophnfb, "hoanbs": hoanbs, "cnp16ov": pop_q}, axis=1
    ).dropna().reset_index()
    out.columns = ["date", "ophnfb", "hoanbs", "cnp16ov"]
    dest = (
        Path(__file__).resolve().parents[1]
        / "puremacro" / "replication" / "data" / "gali1999.csv"
    )
    out.to_csv(dest, index=False)
    print(f"wrote {dest} ({len(out)} quarterly obs, "
          f"{out['date'].min().date()}..{out['date'].max().date()})")


if __name__ == "__main__":
    main()
