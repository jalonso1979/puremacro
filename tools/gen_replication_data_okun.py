"""Regenerate puremacro/replication/data/okun.csv from FRED (dev-only).

Needs a FRED key in the environment (FRED_API_KEY). The key is NEVER written to
the snapshot — only the public series are. Run from the package dir with the key
loaded:
    export $(grep '^FRED_API_KEY=' ../.env | xargs)
    python tools/gen_replication_data_okun.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from puremacro.fetch.fred import fetch_series


def _series(fid: str, var: str) -> pd.Series:
    df = fetch_series(fid, variable=var)
    df = df[["date", "value"]].dropna().sort_values("date")
    return df.set_index("date")["value"]


def main() -> None:
    ip = _series("INDPRO", "ip")
    ur = _series("UNRATE", "ur")
    out = pd.concat({"indpro": ip, "unrate": ur}, axis=1).dropna().reset_index()
    out.columns = ["date", "indpro", "unrate"]
    dest = Path(__file__).resolve().parents[1] / "puremacro" / "replication" / "data" / "okun.csv"
    out.to_csv(dest, index=False)
    print(f"wrote {dest} ({len(out)} monthly obs, {out['date'].min()}..{out['date'].max()})")


if __name__ == "__main__":
    main()
