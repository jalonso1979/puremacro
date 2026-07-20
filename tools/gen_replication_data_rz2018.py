"""Regenerate puremacro/replication/data/rz2018.csv (dev-only).

Freezes the historical quarterly US dataset behind the Ramey-Zubairy
(2018 JPE) government-spending-multiplier replication
(``puremacro.examples.ramey_zubairy_2018_multipliers``).

Source: Valerie Ramey's public replication archive for Ramey, V.A. and
Zubairy, S. (2018). "Government Spending Multipliers in Good Times and
in Bad: Evidence from US Historical Data." Journal of Political Economy
126(2), 850-901.

    https://econweb.ucsd.edu/~vramey/research/Ramey_Zubairy_replication_codes.zip

The zip ships the full dataset both as ``RZDAT.xlsx`` and as
``rzdatnew.csv`` (identical content; see their ``jordagk.do``). We read
the CSV member — no openpyxl needed — and freeze the eight columns the
example uses:

- ``quarter``     decimal quarter (1889.00 = 1889Q1), kept as ``date``
                  (quarter-start timestamps) in the snapshot.
- ``news``        military news shock: present value of expected changes
                  in defense spending (nominal $), RZ's instrument.
- ``ngov``        nominal government purchases.
- ``ngdp``        nominal GDP.
- ``pgdp``        GDP deflator.
- ``rgdp``        real GDP.
- ``rgdp_pott6``  potential real GDP (6th-degree polynomial trend fitted
                  excluding 1930-1946), RZ's baseline normalization.
- ``unemp``       unemployment rate (%), used for the slack state
                  (unemp >= 6.5).

Rows before 1889Q1 (all-missing) are dropped. Run from the package dir:

    python tools/gen_replication_data_rz2018.py

The frozen snapshot is what ships in the wheel; the example and its
tests run offline from it.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

from puremacro.fetch._classic import _safe_urlopen

RZ_ZIP = (
    "https://econweb.ucsd.edu/~vramey/research/"
    "Ramey_Zubairy_replication_codes.zip"
)
CSV_MEMBER = "Ramey_Zubairy_replication_codes/rzdatnew.csv"
COLS = ["news", "ngov", "ngdp", "pgdp", "rgdp", "rgdp_pott6", "unemp"]


def main() -> None:
    raw = _safe_urlopen(RZ_ZIP, timeout=180.0)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        with zf.open(CSV_MEMBER) as fh:
            df = pd.read_csv(fh)

    df = df[df["quarter"] >= 1889.0].reset_index(drop=True)
    year = df["quarter"].astype(int)
    q = ((df["quarter"] - year) * 4).round().astype(int) + 1
    out = pd.DataFrame({
        "date": pd.PeriodIndex(
            [f"{y}Q{k}" for y, k in zip(year, q)], freq="Q"
        ).to_timestamp(),
    })
    for c in COLS:
        out[c] = df[c].round(6)

    dest = (
        Path(__file__).resolve().parents[1]
        / "puremacro" / "replication" / "data" / "rz2018.csv"
    )
    out.to_csv(dest, index=False)
    print(f"wrote {dest} ({len(out)} quarterly obs, "
          f"{out['date'].min().date()}..{out['date'].max().date()}, "
          f"nonzero news quarters: {(out['news'].fillna(0) != 0).sum()})")


if __name__ == "__main__":
    main()
