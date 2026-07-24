#!/usr/bin/env python
"""Freeze X-13ARIMA-SEATS goldens for the native X-11 engine.

Runs the REAL X-13ARIMA-SEATS binary (v1.1.57; located via
``puremacro.sa.x13``'s resolver — see the batch-6 plan for the local
install) on a reference battery, with the spec PINNED to the native v1
configuration:

    transform{function=log|none}   (log where strictly positive)
    arima{model=(0 1 1)(0 1 1)}    (airline; no automdl)
    outlier                        ABSENT (v1 scope)
    forecast{maxlead=60 maxback=60}
    x11{mode=mult|add seasonalma=msr save=(d10 d11 d12)}

and freezes (date, y, d10, d11, d12) per series under
``tests/fixtures/sa_goldens/``. The battery is real data, all fetched
through the on-disk HTTP cache (no hand-typed series):

  * six county LAUS unemployment-level series from the batch-4 border
    panel spanning large/medium/tiny counties (Allegheny PA, Fayette KY,
    Marquette MI, Caddo LA, Keweenaw MI, Hidalgo NM);
  * NSA industrial production (FRED ``IPB50001N``), monthly;
  * two quarterly series: quarterly means of the Allegheny and Keweenaw
    U levels (quarterly-mode coverage).

CI and Pyodide never run the binary — only the frozen CSVs. Regenerate
only when the pinned spec or the battery changes; the tolerance table
lives in ``tests/test_sa/test_x11_native.py`` and was measured, not
promised (see the golden-metadata JSON written next to the CSVs).
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from puremacro._http import safe_get_bytes_cached  # noqa: E402
from puremacro.sa.x13 import _resolve_x13_dir  # noqa: E402

GOLD_DIR = REPO_ROOT / "tests" / "fixtures" / "sa_goldens"
_FREDGRAPH = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

COUNTY_U = {
    "allegheny_pa_u": "42003",
    "fayette_ky_u": "21067",
    "marquette_mi_u": "26103",
    "caddo_la_u": "22017",
    "keweenaw_mi_u": "26083",
    "hidalgo_nm_u": "35023",
}


def fetch_monthly(series_id: str) -> pd.Series:
    raw = safe_get_bytes_cached(_FREDGRAPH.format(series_id))
    df = pd.read_csv(io.BytesIO(raw))
    s = pd.to_numeric(df[df.columns[1]], errors="coerce")
    s.index = pd.to_datetime(df[df.columns[0]])
    s = s.dropna()
    # gap-free calendar (X-13 and the native engine both require it)
    full = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    return s.reindex(full).interpolate(limit_direction="both")


def run_binary(y: np.ndarray, start: pd.Timestamp, period: int,
               use_log: bool, workdir: Path,
               maxback: int = 60) -> dict[str, np.ndarray]:
    x13dir = _resolve_x13_dir()
    if x13dir is None:
        raise SystemExit("X-13 binary not found — see the batch-6 plan "
                         "for the local install.")
    binary = None
    for name in ("x13ashtml", "x13as"):
        cand = Path(x13dir) / name
        if cand.exists():
            binary = cand
            break
    assert binary is not None
    lines = [" " + " ".join(f"{v:.6f}" for v in y[i:i + 6])
             for i in range(0, len(y), 6)]
    startstr = (f"{start.year}.{start.month:02d}" if period == 12
                else f"{start.year}.{(start.month - 1) // 3 + 1}")
    mode = "mult" if use_log else "add"
    spc = (
        'series{ title="g" start=' + startstr
        + f" period={period} data=(\n" + "\n".join(lines) + " ) }\n"
        + "transform{ function=" + ("log" if use_log else "none") + " }\n"
        + "arima{ model=(0 1 1)(0 1 1) }\n"
        + f"forecast{{ maxlead=60 maxback={maxback} }}\n"
        + "x11{ mode=" + mode + " seasonalma=msr save=(d10 d11 d12) }\n")
    (workdir / "g.spc").write_text(spc)
    proc = subprocess.run([str(binary), "g"], cwd=workdir,
                          capture_output=True, text=True, timeout=300)
    out = {}
    for tab in ("d10", "d11", "d12"):
        p = workdir / f"g.{tab}"
        if not p.exists():
            raise RuntimeError(
                f"binary produced no {tab} (stdout tail: "
                f"{proc.stdout[-400:]})")
        df = pd.read_csv(p, sep="\t", skiprows=2, header=None,
                         names=["date", "val"])
        out[tab] = df["val"].to_numpy(float)
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--left-end", action="store_true",
                    help="Batch-7 leg B goldens: maxback=0 (the binary's "
                         "true left-end behavior — it never uses "
                         "backcasts inside X-11), 4-series subset, "
                         "written to sa_goldens_lb/")
    args = ap.parse_args()
    global GOLD_DIR
    maxback = 0 if args.left_end else 60
    if args.left_end:
        GOLD_DIR = GOLD_DIR.parent / "sa_goldens_lb"
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    meta: dict = {"pinned_spec": "log|none + airline + no outlier + "
                                 f"maxlead 60 / maxback {maxback} + "
                                 "x11 msr",
                  "binary": "X-13ARIMA-SEATS 1.1 build 57 "
                            "(x13org/x13prebuilt mac-arm64)",
                  "series": {}}

    battery: list[tuple[str, pd.Series, int]] = []
    for name, fips in COUNTY_U.items():
        s = fetch_monthly(f"LAUCN{fips}0000000004")
        battery.append((name, s, 12))
    # X-13 caps series length at 780 observations; IPB50001N runs from
    # 1919, so keep the last 60 years.
    battery.append(("indpro_nsa", fetch_monthly("IPB50001N").iloc[-720:],
                    12))
    for name, fips in (("allegheny_pa_u_q", "42003"),
                       ("keweenaw_mi_u_q", "26083")):
        s = fetch_monthly(f"LAUCN{fips}0000000004")
        q = s.resample("QS").mean()
        battery.append((name, q, 4))

    for name, s, period in battery:
        y = s.to_numpy(float)
        use_log = bool((y > 0).all())
        with tempfile.TemporaryDirectory() as td:
            tabs = run_binary(y, s.index[0], period, use_log, Path(td),
                              maxback=maxback)
        n = len(y)
        assert all(len(v) == n for v in tabs.values()), (
            name, n, {k: len(v) for k, v in tabs.items()})
        out = pd.DataFrame({"date": s.index.strftime("%Y-%m-%d"), "y": y,
                            **tabs})
        out.to_csv(GOLD_DIR / f"{name}.csv", index=False,
                   float_format="%.10g")
        meta["series"][name] = {"period": period, "n": n,
                                "transform": "log" if use_log else "none",
                                "start": str(s.index[0].date())}
        print(f"golden {name}: n={n}, period={period}, "
              f"transform={'log' if use_log else 'none'}")
    (GOLD_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {len(battery)} goldens -> {GOLD_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
