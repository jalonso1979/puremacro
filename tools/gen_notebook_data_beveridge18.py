#!/usr/bin/env python
"""Freeze the Notebook-18 (Beveridge curve) datasets.

Two snapshots into ``puremacro/replication/data/`` via the batch-5a
fetchers plus key-free fredgraph:

- ``beveridge18_us.csv`` — monthly US: LAUS urate (UNRATE) and
  unemployment level (UNEMPLOY, thousands), JOLTS openings
  (level + rate), hires and quits (levels + rates), SA, 2000-12+.
- ``beveridge18_eu.csv`` — quarterly Europe: Eurostat job vacancy rate
  (jvs_q_nace2, JVR, NACE B-S, TOTAL size class, SA) merged with the
  quarterly mean of the Eurostat monthly LFS urate (une_rt_m, total
  sex/age), ISO-3 codes, countries only.

Rerun to refresh; both fetch paths go through the on-disk HTTP caches.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from puremacro.fetch._http import cached_get  # noqa: E402
from puremacro.fetch.jolts import fetch_jolts  # noqa: E402
from puremacro.fetch.labor_eurostat import fetch_eurostat_lfs_panel  # noqa: E402
from puremacro.fetch.vacancies_eurostat import fetch_eurostat_vacancies  # noqa: E402

OUT = REPO_ROOT / "puremacro" / "replication" / "data"
_FREDGRAPH = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"


def _fred_m(sid: str) -> pd.Series:
    df = pd.read_csv(io.BytesIO(cached_get(_FREDGRAPH.format(sid))))
    s = pd.to_numeric(df[df.columns[1]], errors="coerce")
    s.index = pd.to_datetime(df[df.columns[0]])
    return s.dropna()


def main() -> int:
    # --- US -----------------------------------------------------------------
    j = fetch_jolts(elements=("openings", "hires", "quits"),
                    industries=("total",))
    wide = j.pivot(index="date", columns="element",
                   values=["level", "rate"])
    us = pd.DataFrame({
        "urate": _fred_m("UNRATE"),
        "unemp_level": _fred_m("UNEMPLOY"),
        "v_level": wide[("level", "openings")],
        "v_rate": wide[("rate", "openings")],
        "hires_level": wide[("level", "hires")],
        "hires_rate": wide[("rate", "hires")],
        "quits_rate": wide[("rate", "quits")],
    }).dropna()
    us.index.name = "date"
    us.reset_index().to_csv(OUT / "beveridge18_us.csv", index=False,
                            float_format="%.4f")
    print(f"beveridge18_us.csv: {len(us)} months "
          f"{us.index.min().date()}..{us.index.max().date()}")

    # --- Europe ---------------------------------------------------------------
    jvr = fetch_eurostat_vacancies()          # JVR, B-S, TOTAL, SA
    lfs = fetch_eurostat_lfs_panel()
    u = lfs[(lfs["sex"] == "_T") & (lfs["age"] == "Y_GE15")][
        ["code", "date", "urate"]].dropna()
    # House labor panels store rates as FRACTIONS; JVR is percent —
    # put both axes of the Beveridge plot in percent.
    u["urate"] = u["urate"] * 100.0
    u["q"] = u["date"].dt.to_period("Q")
    uq = (u.groupby(["code", "q"])
           .agg(urate=("urate", "mean"), n_m=("urate", "size"))
           .reset_index())
    uq = uq[uq["n_m"] == 3]
    uq["date"] = uq["q"].dt.to_timestamp(how="start")
    eu = jvr.merge(uq[["code", "date", "urate"]], on=["code", "date"],
                   how="inner")
    eu.to_csv(OUT / "beveridge18_eu.csv", index=False,
              float_format="%.4f")
    print(f"beveridge18_eu.csv: {eu['code'].nunique()} countries, "
          f"{len(eu)} country-quarters "
          f"{eu['date'].min().date()}..{eu['date'].max().date()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
