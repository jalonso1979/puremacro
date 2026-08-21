#!/usr/bin/env python
"""Freeze the Notebook-40 (quarterly national accounts) panel.

One snapshot into ``puremacro/replication/data/qna40_panel.csv``: six
countries from 1995, built by ``qna_panel(..., output=True, income=True,
real=True)`` — so all three OECD measurements of GDP travel together, each
with its own components, deflators and volume measures.

The six are chosen for what they show, not for coverage:

- ``USA`` is absent from ``DF_QNA_BY_ACTIVITY_OUTPUT`` entirely (the industry
  accounts are a separate BEA release), so its output columns are NaN — the
  honest shape of a country that does not publish an approach.
- ``JPN`` publishes an output-flow GDP that disagrees with its
  expenditure-flow GDP, which is what ``crossflow_output`` is for.
- ``DEU`` does the same on the income side.
- ``ITA`` and ``USA`` sit at opposite ends of the unadjusted labour share,
  for the Gollin (2002) point about self-employment income.
- ``MEX`` and ``ESP`` reference their volumes to different base years, which
  is what ``qna_rebase`` exists to reconcile.

Rerun to refresh; the fetch goes through the on-disk SDMX cache. Floats are
written to six significant digits, which is well past the source's own
precision and keeps the snapshot under half a megabyte.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from puremacro.fetch import qna_meta, qna_panel  # noqa: E402

CODES = ["USA", "JPN", "DEU", "ITA", "MEX", "ESP"]
START = "1995"
DATA_DIR = REPO_ROOT / "puremacro" / "replication" / "data"
OUT = DATA_DIR / "qna40_panel.csv"
OUT_META = DATA_DIR / "qna40_meta.csv"


def main() -> int:
    panel = qna_panel(CODES, start=START, output=True, income=True, real=True)
    if panel.empty:
        print("qna_panel returned nothing — no network, or the flow moved.")
        return 1

    flat = panel.reset_index()
    flat["date"] = pd.to_datetime(flat["date"]).dt.strftime("%Y-%m-%d")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    flat.to_csv(OUT, index=False, float_format="%.6g")

    # qna_meta reads panel.attrs, and .attrs does not survive a CSV round
    # trip -- so the metadata that says which base year each country's
    # volumes are referenced to has to travel as its own file.
    qna_meta(panel).to_csv(OUT_META, index=False)

    size_kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT.relative_to(REPO_ROOT)}: "
          f"{len(flat):,} rows x {flat.shape[1]} columns, {size_kb:,.0f} KB")
    print(f"wrote {OUT_META.relative_to(REPO_ROOT)}: "
          f"{len(CODES)} countries of provenance")
    print(f"  countries: {', '.join(sorted(flat['code'].unique()))}")
    print(f"  quarters:  {flat['date'].min()} .. {flat['date'].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
