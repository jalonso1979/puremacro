#!/usr/bin/env python
"""Main Street Uncertainty — Phase 5: real-time (ALFRED) outcome vintages.

Batch-4 left one identification question ranked first among the
remaining candidates: does the surviving own-district response (peak
+0.025 at h = 5, WC p = 0.048 in the LOO horse race) — and the robust
LOO-national response (+0.035 at h = 12, WC p = 0.007) — hold when the
OUTCOMES are measured as first published, rather than after decades of
LAUS/CES benchmark revisions? Beige Book text is never revised, so the
shock side is real-time by construction; only the outcome side needs
vintages.

Data: the keyed ALFRED API with ``output_type=4`` (initial release
only) returns, in ONE call per series, every observation at the value
it carried when FIRST published, stamped with its publication date
(``realtime_start``). Key-free access does not exist: ``fredgraph.csv``
silently ignores ``vintage_date`` (probe 2026-07-21) and
``alfredgraph.csv`` 404s. The key comes from the house credentials
layer (service ``fred``): ``export FRED_API_KEY=...`` or
``~/.puremacro/credentials.toml`` ``[fred] api_key = "..."`` — free
signup at https://fred.stlouisfed.org/docs/api/api_key.html.

First-release discipline: ALFRED archives for state LAUS/CES series
start well after 1992; observations older than the archive appear with
``realtime_start`` equal to the archive's first capture, not their true
publication date. An observation is treated as genuinely first-release
only when ``realtime_start - date <= --max-pub-lag`` days (default 120);
everything else is dropped from the real-time sample and counted in
``realtime_coverage.csv``. Long differences y(t+h) - y(t-1) are formed
from first-release values at BOTH ends (the Croushore-Stark 2001
"first-release diagonal" convention) — this measures what a
contemporaneous observer's data would have shown, not any single
vintage's revision of history.

Comparison design: the phase-3 baseline (expo × own) and the phase-4
horse race (expo × own + expo × LOO) are re-estimated twice on the SAME
(state, quarter) rows — once with first-release outcomes (``rt_``
specs), once with current-vintage outcomes restricted to that row set
(``cv_`` specs). Differences between ``cv_`` and the frozen full-sample
phase-4 numbers are window effects; differences between ``rt_`` and
``cv_`` are vintage effects. Both gaps are reported separately.

Estimator: imported from run_main_street_phase3 / run_main_street_phase4
(identical demeaning, DK bandwidth, restricted wild cluster over the 12
districts).

Outputs (docs/research/main_street_uncertainty/output/):
    realtime_first_release_panel.csv.gz   state-quarter rt outcomes
    realtime_coverage.csv                 per-series archive start + kept share
    irf_realtime.csv                      rt_ and cv_ LP paths + DK + WC p
    fig_phase5_realtime.(png|pdf)
    phase5_manifest.json, phase5_summary.json, run_log_phase5.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from puremacro import credentials  # noqa: E402
from puremacro._http import safe_get_bytes_cached  # noqa: E402

import run_main_street_phase3 as p3  # noqa: E402
import run_main_street_phase4 as p4  # noqa: E402

OUT_DIR = p3.OUT_DIR
MERGED_CSV = p3.MERGED_CSV
EXPO_CSV = OUT_DIR / "exposure_state.csv"
PH4_SUMMARY = OUT_DIR / "phase4_summary.json"

# output_type=4 = "initial release only". The full realtime span is
# REQUIRED: with the default (today-today) real-time period the API
# 400s ("No vintage dates exist for the specified real-time period").
ALFRED_OBS = ("https://api.stlouisfed.org/fred/series/observations"
              "?series_id={sid}&api_key={key}&file_type=json&output_type=4"
              "&realtime_start=1776-07-04&realtime_end=9999-12-31")

HORIZONS = p3.HORIZONS
N_LAGS = p3.N_LAGS
BASELINE_START = p3.BASELINE_START
ALL_STATES = p3.ALL_STATES

_T0 = time.time()
_LOG_LINES: list[str] = []


def log(msg: str) -> None:
    line = f"[{time.time() - _T0:7.1f}s] {msg}"
    print(line, flush=True)
    _LOG_LINES.append(line)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_key() -> str:
    key = credentials.get("fred")
    if not key:
        raise SystemExit(
            "run_main_street_phase5_realtime: no FRED/ALFRED API key "
            "configured. Get a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html and either\n"
            "  export FRED_API_KEY=<key>\n"
            "or add to ~/.puremacro/credentials.toml:\n"
            "  [fred]\n  api_key = \"<key>\"\n"
            "then re-run. (Key-free vintage access does not exist — "
            "fredgraph.csv ignores vintage_date; probed 2026-07-21.)")
    return key


def fetch_first_release(sid: str, key: str) -> pd.DataFrame:
    """Initial-release observations for one series (output_type=4).

    Returns DataFrame(date, value, pub_date, pub_lag_days). The API key
    is interpolated into the URL only for the request; never logged and
    never written to any output file. The on-disk HTTP cache keys on the
    full URL (local file, same trust domain as the key's env/config)."""
    url = ALFRED_OBS.format(sid=urllib.parse.quote(sid), key=key)
    raw = safe_get_bytes_cached(url)
    obj = json.loads(raw)
    obs = obj.get("observations", [])
    rows = []
    for o in obs:
        v = o.get("value", ".")
        if v in (".", "", None):
            continue
        rows.append({"date": o["date"], "value": float(v),
                     "pub_date": o["realtime_start"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["pub_date"] = pd.to_datetime(df["pub_date"])
    df["pub_lag_days"] = (df["pub_date"] - df["date"]).dt.days
    return df


def build_rt_panel(key: str, max_pub_lag: int
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """First-release state urate / emp100, quarterly, + coverage table."""
    panels, cov = [], []
    for st in ALL_STATES:
        rec: dict = {"state": st}
        series = {}
        for label, sid in (("urate", f"{st}UR"), ("emp", f"{st}NA")):
            df = fetch_first_release(sid, key)
            if df.empty:
                rec[f"{label}_archive_start"] = None
                rec[f"{label}_kept_share"] = 0.0
                continue
            kept = df[df["pub_lag_days"] <= max_pub_lag]
            rec[f"{label}_archive_start"] = str(df["pub_date"].min().date())
            rec[f"{label}_kept_share"] = round(len(kept) / len(df), 3)
            rec[f"{label}_first_kept"] = (
                str(kept["date"].min().date()) if len(kept) else None)
            s = kept.set_index("date")["value"].sort_index()
            series[label] = s
        cov.append(rec)
        if "urate" not in series and "emp" not in series:
            continue
        # Quarterly means over months whose first release is in-sample;
        # require all 3 months so a quarter is never a 1-month average.
        frames = {}
        for label, s in series.items():
            q = s.groupby(s.index.to_period("Q")).agg(["mean", "size"])
            q = q[q["size"] == 3]["mean"]
            frames[label] = q
        out = pd.DataFrame(frames)
        out["state"] = st
        out["date"] = out.index.to_timestamp(how="start")
        panels.append(out.reset_index(drop=True))
    panel = pd.concat(panels, ignore_index=True)
    if "emp" in panel.columns:
        panel["emp100"] = 100.0 * np.log(panel["emp"] * 1.0)
    coverage = pd.DataFrame(cov)
    n_ur = int(coverage["urate_kept_share"].gt(0).sum())
    log(f"real-time panel: {panel['state'].nunique()} states with any "
        f"first-release quarters ({n_ur} with urate); "
        f"{len(panel)} state-quarters")
    return panel, coverage


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--wc-draws", type=int, default=999)
    ap.add_argument("--seed", type=int, default=42,
                    help="keep 42: wc draw sequence then matches the "
                         "phase-3/4 runs")
    ap.add_argument("--max-pub-lag", type=int, default=120,
                    help="days between observation month and first "
                         "publication for an observation to count as "
                         "genuinely first-release (default 120)")
    ap.add_argument("--probe", action="store_true",
                    help="fetch ONE series (CAUR), print shape/coverage, "
                         "exit — cheap validation for the first keyed run")
    ap.add_argument("--fast", action="store_true",
                    help="quick mode: --wc-draws 199")
    args = ap.parse_args(argv)
    B = 199 if args.fast else args.wc_draws

    key = require_key()

    if args.probe:
        df = fetch_first_release("CAUR", key)
        print(df.head(3))
        print(df.tail(3))
        print(f"rows={len(df)}, archive start={df['pub_date'].min()}, "
              f"share pub_lag<={args.max_pub_lag}d: "
              f"{(df['pub_lag_days'] <= args.max_pub_lag).mean():.2f}")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rt, coverage = build_rt_panel(key, args.max_pub_lag)
    coverage.to_csv(OUT_DIR / "realtime_coverage.csv", index=False)
    rt.to_csv(OUT_DIR / "realtime_first_release_panel.csv.gz",
              index=False, float_format="%.6f")

    merged = pd.read_csv(MERGED_CSV, parse_dates=["date"])
    expo = pd.read_csv(EXPO_CSV)
    loo_long, _ = p4.build_loo(merged)

    # Shock/exposure side identical to phase 4; outcomes swapped in.
    base = merged.merge(expo[["state", "mfg_z"]], on="state")
    base = (base.rename(columns={"mfg_z": "expo_z"})
            [["state", "district", "date", "urate", "emp100",
              "bbui_innov", "expo_z"]]
            .dropna(subset=["expo_z"]))
    base = base[base["date"] >= BASELINE_START]
    base = base.merge(loo_long[["district", "date", "bbui_loo_z"]],
                      on=["district", "date"], how="left")

    rt_rows = rt.dropna(subset=["urate"])[["state", "date", "urate"]]
    rt_rows = rt_rows.rename(columns={"urate": "urate_rt"})
    both = base.merge(rt_rows, on=["state", "date"], how="inner")
    log(f"common window: {both['date'].min().date()} .. "
        f"{both['date'].max().date()}, {len(both)} state-quarters "
        f"({both['state'].nunique()} states)")

    wc_draws = rng.choice([-1.0, 1.0], size=(B, 12))

    cv = both.rename(columns={"urate": "urate_use"})
    rt_p = both.rename(columns={"urate_rt": "urate_use"})
    irf_frames = []
    for tag, pnl in (("cv", cv), ("rt", rt_p)):
        irf_frames.append(p4.run_spec_multi(
            pnl, spec=f"{tag}_urate_own", outcome="urate_use",
            shock_cols=["bbui_innov"], horizons=HORIZONS,
            wc_draws=wc_draws))
        irf_frames.append(p4.run_spec_multi(
            pnl, spec=f"{tag}_urate_hr_own", outcome="urate_use",
            shock_cols=["bbui_innov", "bbui_loo_z"], horizons=HORIZONS,
            wc_draws=wc_draws))
        irf_frames.append(p4.run_spec_multi(
            pnl, spec=f"{tag}_urate_hr_loo", outcome="urate_use",
            shock_cols=["bbui_loo_z", "bbui_innov"], horizons=HORIZONS,
            wc_draws=wc_draws))
    irf = pd.concat(irf_frames, ignore_index=True)
    irf.to_csv(OUT_DIR / "irf_realtime.csv", index=False)

    # --- headline comparison at the batch-4 anchor horizons -----------------
    ph4 = json.loads(PH4_SUMMARY.read_text())

    def _at(spec: str, h: int) -> dict:
        r = irf[(irf["spec"] == spec) & (irf["h"] == h)].iloc[0]
        return {"beta": round(float(r["beta"]), 5),
                "se_dk": round(float(r["se_dk"]), 5),
                "p_wc": round(float(r["p_wc"]), 3)}

    anchors = {
        "own_h5": {"frozen_phase4": ph4["paths"]["urate_hr_own"]["5"],
                   "cv": _at("cv_urate_hr_own", 5),
                   "rt": _at("rt_urate_hr_own", 5)},
        "own_h9": {"frozen_phase4": ph4["paths"]["urate_hr_own"]["9"],
                   "cv": _at("cv_urate_hr_own", 9),
                   "rt": _at("rt_urate_hr_own", 9)},
        "loo_h12": {"frozen_phase4": ph4["paths"]["urate_hr_loo"]["12"],
                    "cv": _at("cv_urate_hr_loo", 12),
                    "rt": _at("rt_urate_hr_loo", 12)},
    }
    for name, a in anchors.items():
        log(f"{name}: frozen {a['frozen_phase4']['beta']:+.4f} | "
            f"window-matched cv {a['cv']['beta']:+.4f} "
            f"(p {a['cv']['p_wc']:.2f}) | first-release rt "
            f"{a['rt']['beta']:+.4f} (p {a['rt']['p_wc']:.2f})")

    # --- figure ---------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), sharey=True)
    for ax, (spec_stub, title) in zip(axes, [
            ("urate_hr_own", "Own district | LOO controlled"),
            ("urate_hr_loo", "LOO national")]):
        cv_r = irf[irf["spec"] == f"cv_{spec_stub}"].sort_values("h")
        rt_r = irf[irf["spec"] == f"rt_{spec_stub}"].sort_values("h")
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.fill_between(rt_r["h"], rt_r["lo_dk"], rt_r["hi_dk"],
                        color="C0", alpha=0.22)
        ax.plot(rt_r["h"], rt_r["beta"], color="C0", lw=2.0, marker="o",
                ms=4, label="first-release outcomes")
        ax.plot(cv_r["h"], cv_r["beta"], color="0.45", lw=1.6, ls="--",
                marker="s", ms=3, label="current vintage, same window")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("quarters after shock", fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("pp per (1 s.d. exposure) x (1 s.d. shock)",
                       fontsize=9)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Phase 5: do the horse-race responses survive on data "
                 "as first published? (ALFRED initial releases)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for suffix in (".png", ".pdf"):
        fig.savefig((OUT_DIR / "fig_phase5_realtime.png")
                    .with_suffix(suffix),
                    dpi=200 if suffix == ".png" else None)
    plt.close(fig)
    log("figure written")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "alfred": "api.stlouisfed.org fred/series/observations "
                      "output_type=4 (initial release), series {ST}UR + "
                      "{ST}NA x 51; key via puremacro.credentials('fred') "
                      "— never written to outputs",
            "state_district_panel_quarterly.csv": {
                "sha256": sha256_file(MERGED_CSV)},
            "exposure_state.csv": {"sha256": sha256_file(EXPO_CSV)},
            "phase4_summary.json": {"sha256": sha256_file(PH4_SUMMARY)},
        },
        "first_release_rule": f"pub_lag <= {args.max_pub_lag} days; "
                              "quarters require all 3 months",
        "long_difference_convention": "Croushore-Stark first-release "
                                      "diagonal: y(t+h) and y(t-1) both "
                                      "first-release",
        "estimator": "imported from run_main_street_phase3/4",
        "wc_draws": B, "seed": args.seed,
    }
    (OUT_DIR / "phase5_manifest.json").write_text(
        json.dumps(manifest, indent=2))
    summary = {
        "mode": "fast" if args.fast else "full",
        "max_pub_lag_days": args.max_pub_lag,
        "common_window": [str(both["date"].min().date()),
                          str(both["date"].max().date())],
        "n_state_quarters": int(len(both)),
        "anchors": anchors,
        "runtime_seconds": round(time.time() - _T0, 1),
    }
    (OUT_DIR / "phase5_summary.json").write_text(
        json.dumps(summary, indent=2))
    (OUT_DIR / "run_log_phase5.txt").write_text("\n".join(_LOG_LINES) + "\n")
    log("summary written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
