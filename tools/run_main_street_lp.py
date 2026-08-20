#!/usr/bin/env python
"""Main Street Uncertainty — Phase 2 first-pass local projections.

Panel LP of state labor outcomes on district-level Beige Book
uncertainty (BBUI) innovations:

* Shock: within-district AR(2) innovation of the quarterly district BBUI
  (raw sentence-cooccurrence index from
  ``data/processed/bbui_district_panel.csv``), z-scored within district —
  :func:`puremacro.uncertainty.lp_long.derive_innovation_shock`.
* Outcomes: state unemployment rate (pp) and 100·log state nonfarm
  employment, quarterly, from the key-free FRED CSV mirror of BLS
  LAUS/CES (``docs/research/main_street_uncertainty/output/
  state_labor_quarterly.csv``, built by ``/tmp`` fetch script from
  ``puremacro.fetch.bls_state_panel``).
* Crosswalk: states -> primary Fed district via ``STATE_TO_DISTRICT``
  (population-majority convention for the 14 split states; see
  ``puremacro.narrative.indices._fed_districts``).
* Estimator: pooled panel Jordà (2005, AER) LP, horizons 0-12 quarters,
  two-way (entity + time) fixed effects with Driscoll-Kraay (1998,
  REStat) HAC standard errors — :func:`puremacro.lp.panel_dk.panel_lp_dk`
  (LHS is the long difference y_{t+h} − y_{t-1}; 4 lags of shock and
  outcome as controls). NOTE the time FE absorb the common national
  component of the shock each quarter: coefficients are identified off
  *relative* cross-district variation only.
* Simultaneous inference: sup-t band across the 13 horizons (Montiel
  Olea & Plagborg-Møller 2019, Journal of Applied Econometrics 34(1),
  1-17, Algorithm 2) via :func:`puremacro.inference.supt.supt_band`
  on district-cluster bootstrap draws (districts resampled with
  replacement — the shock varies at the district level, so the district
  is the honest resampling unit; only 12 clusters, a stated caveat).
* Per-district IRFs: equal-weighted mean of member-state outcomes per
  district, time-series Jordà LP with HAC (bandwidth h+1) via
  :func:`puremacro.lp.jorda.lp_hac` — the 12-district small multiples.

Descriptive dynamic correlations only — no identification claims.

Outputs (docs/research/main_street_uncertainty/output/):
    state_district_panel_quarterly.csv   merged LP input panel
    bbui_coverage_by_decade.csv          district-panel coverage table
    irf_pooled.csv                       pooled LP + DK + sup-t bands
    irf_by_district.csv                  per-district LP paths
    fig_pooled_irf.(png|pdf)             headline IRF figure
    fig_district_irf_grid.(png|pdf)      12-district small multiples
    panel_manifest.json, summary.json, run_log.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from puremacro.narrative.indices._fed_districts import (  # noqa: E402
    DISTRICT_NAMES, STATE_TO_DISTRICT,
)
from puremacro.uncertainty.lp_long import derive_innovation_shock  # noqa: E402
from puremacro.lp.panel_dk import panel_lp_dk  # noqa: E402
from puremacro.lp.jorda import lp_hac  # noqa: E402
from puremacro.inference.supt import supt_band  # noqa: E402

BBUI_CSV = REPO_ROOT / "data" / "processed" / "bbui_district_panel.csv"
OUT_DIR = (REPO_ROOT / "docs" / "research" / "main_street_uncertainty"
           / "output")
STATE_CSV = OUT_DIR / "state_labor_quarterly.csv"

HORIZONS = list(range(0, 13))
N_LAGS = 4
ALPHA = 0.10  # 90% bands throughout

_T0 = time.time()
_LOG_LINES: list[str] = []


def log(msg: str) -> None:
    line = f"[{time.time() - _T0:7.1f}s] {msg}"
    print(line, flush=True)
    _LOG_LINES.append(line)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def coverage_by_decade(bbui: pd.DataFrame) -> pd.DataFrame:
    """District-panel coverage per decade (districts only, no National)."""
    d = bbui[bbui["district"] != "National"].copy()
    d = d[d["n_docs"] > 0]
    d["year"] = d["date"].dt.year
    d["decade"] = (d["year"] // 10) * 10
    rows = []
    for dec, grp in d.groupby("decade"):
        qs = grp["date"].nunique()
        # possible quarters in-decade given the panel's own span
        lo, hi = grp["date"].min(), grp["date"].max()
        possible = int(((hi.year - lo.year) * 4
                        + (hi.quarter - lo.quarter)) + 1)
        full = int((grp.groupby("date")["district"].nunique() == 12).sum())
        rows.append({
            "decade": f"{dec}s",
            "span": f"{lo.year}Q{lo.quarter}-{hi.year}Q{hi.quarter}",
            "quarters_covered": qs,
            "quarters_possible_in_span": possible,
            "quarters_all_12_districts": full,
            "district_quarter_cells": len(grp),
            "docs_parsed": int(grp["n_docs"].sum()),
            "mean_docs_per_cell": round(float(grp["n_docs"].mean()), 1),
        })
    return pd.DataFrame(rows)


def missing_quarters(bbui: pd.DataFrame) -> list[str]:
    """Quarters inside the panel span with < 12 districts reporting."""
    d = bbui[(bbui["district"] != "National") & (bbui["n_docs"] > 0)]
    got = d.groupby("date")["district"].nunique()
    full_range = pd.period_range(d["date"].min(), d["date"].max(), freq="Q")
    out = []
    for p in full_range:
        ts = p.to_timestamp(how="start")
        n = int(got.get(ts, 0))
        if n < 12:
            out.append(f"{p} ({n}/12)")
    return out


# ---------------------------------------------------------------------------
# Panel assembly
# ---------------------------------------------------------------------------

def build_merged_panel(bbui: pd.DataFrame, states: pd.DataFrame
                       ) -> pd.DataFrame:
    dist = bbui[(bbui["district"] != "National")
                & (bbui["n_docs"] > 0)].copy()

    long = dist.rename(columns={"district": "code"})[
        ["code", "date", "value"]]
    long["variable"] = "bbui_q"
    shocks = derive_innovation_shock(long, "bbui_q")
    shocks = shocks.rename(columns={"code": "district",
                                    "value": "bbui_innov"})
    shocks = shocks[["district", "date", "bbui_innov"]]
    log(f"AR(2) innovations: {shocks['district'].nunique()} districts, "
        f"{len(shocks)} district-quarters")

    st = states.copy()
    st["date"] = pd.to_datetime(st["date"])
    st["district"] = st["state"].map(STATE_TO_DISTRICT)
    st["emp100"] = 100.0 * st["log_emp"]

    merged = st.merge(
        dist.rename(columns={"value": "bbui"})[["district", "date", "bbui"]],
        on=["district", "date"], how="inner",
    ).merge(shocks, on=["district", "date"], how="left")
    merged = merged.sort_values(["state", "date"]).reset_index(drop=True)
    return merged[["state", "district", "date", "urate", "emp100",
                   "d4_log_emp", "bbui", "bbui_innov"]]


# ---------------------------------------------------------------------------
# LPs
# ---------------------------------------------------------------------------

def pooled_lp(merged: pd.DataFrame, outcome: str) -> pd.DataFrame:
    wide = (merged[["state", "date", outcome, "bbui_innov"]]
            .dropna()
            .set_index(["state", "date"])
            .sort_index())
    res = panel_lp_dk(wide, y=outcome, x="bbui_innov", horizons=HORIZONS,
                      n_lags=N_LAGS, alpha=ALPHA,
                      entity_level="state", time_level="date")
    return res


def district_bootstrap_draws(merged: pd.DataFrame, outcome: str,
                             B: int, seed: int) -> np.ndarray:
    """District-cluster bootstrap of the pooled LP beta path.

    Resamples the 12 districts with replacement (Cameron-Gelbach-Miller
    2008 cluster bootstrap at the level of the shock); every drawn
    district contributes all of its member states, relabelled per copy.
    """
    rng = np.random.default_rng(seed)
    districts = sorted(merged["district"].unique())
    by_dist = {d: g for d, g in merged.groupby("district")}
    draws = np.full((B, len(HORIZONS)), np.nan)
    n_fail = 0
    for b in range(B):
        picks = rng.choice(districts, size=len(districts), replace=True)
        parts = []
        for k, d in enumerate(picks):
            g = by_dist[d].copy()
            g["state"] = g["state"] + f"~{k}"
            parts.append(g)
        bp = pd.concat(parts, ignore_index=True)
        try:
            res = pooled_lp(bp, outcome)
            draws[b, :] = res["beta"].to_numpy()
        except Exception:
            n_fail += 1
    if n_fail:
        log(f"  bootstrap[{outcome}]: {n_fail}/{B} draws failed (dropped)")
    return draws[np.isfinite(draws).all(axis=1)]


def per_district_lp(merged: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Equal-weighted district aggregate outcome, time-series LP w/ HAC."""
    rows = []
    for d, g in merged.groupby("district"):
        agg = (g.groupby("date")
                .agg(y=(outcome, "mean"), x=("bbui_innov", "first"))
                .reset_index()
                .sort_values("date"))
        agg = agg.dropna()
        if len(agg) < 4 * N_LAGS + max(HORIZONS) + 8:
            log(f"  per-district LP: skipping {d} (T={len(agg)})")
            continue
        res = lp_hac(agg, y="y", x="x", horizons=HORIZONS,
                     n_lags=N_LAGS, alpha=ALPHA)
        res["district"] = d
        res["n_quarters"] = len(agg)
        rows.append(res)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_pooled(results: dict, fig_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "urate": ("Unemployment rate", "pp per 1 s.d. BBUI innovation"),
        "emp100": ("Nonfarm employment", "% per 1 s.d. BBUI innovation"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for ax, (outcome, r) in zip(axes, results.items()):
        h = r["irf"]["h"].to_numpy()
        beta = r["irf"]["beta"].to_numpy()
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.fill_between(h, r["irf"]["lo"], r["irf"]["hi"],
                        color="C0", alpha=0.25,
                        label="90% pointwise (Driscoll-Kraay)")
        ax.plot(h, r["supt_lo"], color="C0", ls="--", lw=1.0,
                label="90% sup-t simultaneous")
        ax.plot(h, r["supt_hi"], color="C0", ls="--", lw=1.0)
        ax.plot(h, beta, color="C0", lw=2.0, marker="o", ms=3)
        title, unit = labels[outcome]
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("quarters after shock")
        ax.set_ylabel(unit, fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("State labor response to district Beige Book uncertainty "
                 "innovations (two-way FE panel LP; relative variation)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for suffix in (".png", ".pdf"):
        fig.savefig(fig_path.with_suffix(suffix),
                    dpi=200 if suffix == ".png" else None)
    plt.close(fig)


def fig_districts(irf_d: pd.DataFrame, pooled: pd.DataFrame,
                  fig_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 4, figsize=(13, 8), sharex=True,
                             sharey=True)
    for ax, d in zip(axes.ravel(), DISTRICT_NAMES):
        sub = irf_d[irf_d["district"] == d]
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.plot(pooled["h"], pooled["beta"], color="0.7", lw=1.2,
                label="pooled")
        if not sub.empty:
            ax.fill_between(sub["h"], sub["lo"], sub["hi"], color="C0",
                            alpha=0.20)
            ax.plot(sub["h"], sub["beta"], color="C0", lw=1.6)
            nq = int(sub["n_quarters"].iloc[0])
            ax.set_title(f"{d} (T={nq})", fontsize=9)
        else:
            ax.set_title(d, fontsize=9)
        ax.tick_params(labelsize=7)
    fig.suptitle("Unemployment response to own-district BBUI innovation — "
                 "district-aggregated Jordà LP, 90% HAC bands "
                 "(grey: pooled panel-LP path)", fontsize=11)
    fig.supxlabel("quarters after shock", fontsize=9)
    fig.supylabel("pp per 1 s.d. BBUI innovation", fontsize=9)
    fig.tight_layout(rect=(0.01, 0.01, 1, 0.95))
    for suffix in (".png", ".pdf"):
        fig.savefig(fig_path.with_suffix(suffix),
                    dpi=200 if suffix == ".png" else None)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--boot", type=int, default=300,
                    help="district-cluster bootstrap draws (default 300)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fast", action="store_true",
                    help="quick mode: --boot 40")
    args = ap.parse_args(argv)
    B = 40 if args.fast else args.boot

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bbui = pd.read_csv(BBUI_CSV, parse_dates=["date"])
    states = pd.read_csv(STATE_CSV, parse_dates=["date"])
    log(f"BBUI panel: {bbui['district'].nunique()} districts x "
        f"{bbui['date'].nunique()} quarters "
        f"({bbui['date'].min().date()}..{bbui['date'].max().date()})")
    log(f"state panel: {states['state'].nunique()} states x "
        f"{states['date'].nunique()} quarters")

    cov = coverage_by_decade(bbui)
    cov.to_csv(OUT_DIR / "bbui_coverage_by_decade.csv", index=False)
    gaps = missing_quarters(bbui)
    log(f"coverage table written; {len(gaps)} quarters with <12 districts")

    merged = build_merged_panel(bbui, states)
    merged.to_csv(OUT_DIR / "state_district_panel_quarterly.csv",
                  index=False)
    log(f"merged panel: {merged['state'].nunique()} states x "
        f"{merged['date'].nunique()} quarters, {len(merged)} rows "
        f"({merged['date'].min().date()}..{merged['date'].max().date()})")

    results: dict[str, dict] = {}
    irf_frames = []
    for outcome in ("urate", "emp100"):
        irf = pooled_lp(merged, outcome)
        n_eff = len(merged.dropna(subset=[outcome, "bbui_innov"]))
        log(f"pooled LP [{outcome}]: h0={irf['beta'].iloc[0]:+.4f} "
            f"(se {irf['se'].iloc[0]:.4f}), "
            f"h8={irf['beta'].iloc[8]:+.4f}, h12={irf['beta'].iloc[12]:+.4f}"
            f"  (n={n_eff})")
        draws = district_bootstrap_draws(merged, outcome, B=B,
                                         seed=args.seed)
        band = supt_band(theta=irf["beta"].to_numpy(), draws=draws,
                         method="bootstrap", alpha=ALPHA)
        log(f"sup-t [{outcome}]: crit={band.crit_value:.2f} "
            f"({draws.shape[0]} usable draws), "
            f"excl. zero at h: "
            f"{[int(h) for h, lo, hi in zip(irf['h'], band.lower, band.upper) if lo > 0 or hi < 0]}")
        irf = irf.copy()
        irf["outcome"] = outcome
        irf["supt_lo"] = band.lower
        irf["supt_hi"] = band.upper
        irf["supt_crit"] = band.crit_value
        irf["boot_draws"] = draws.shape[0]
        irf_frames.append(irf)
        results[outcome] = {"irf": irf, "supt_lo": band.lower,
                            "supt_hi": band.upper,
                            "supt_crit": float(band.crit_value)}
    pd.concat(irf_frames, ignore_index=True).to_csv(
        OUT_DIR / "irf_pooled.csv", index=False)

    irf_d = per_district_lp(merged, "urate")
    irf_d.to_csv(OUT_DIR / "irf_by_district.csv", index=False)
    log(f"per-district LPs: {irf_d['district'].nunique()} districts")

    fig_pooled(results, OUT_DIR / "fig_pooled_irf.png")
    fig_districts(irf_d, results["urate"]["irf"],
                  OUT_DIR / "fig_district_irf_grid.png")
    log("figures written")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "bbui_district_panel.csv": {
                "sha256": sha256_file(BBUI_CSV),
                "rows": int(len(bbui)),
                "span": f"{bbui['date'].min().date()}..{bbui['date'].max().date()}",
            },
            "state_labor_quarterly.csv": {
                "sha256": sha256_file(STATE_CSV),
                "rows": int(len(states)),
                "source": "FRED fredgraph CSV (no-key mirror of BLS "
                          "LAUS {ST}UR / CES {ST}NA), quarterly means",
            },
        },
        "crosswalk": "STATE_TO_DISTRICT primary mapping "
                     "(population-majority for 14 split states)",
        "shock": "AR(2) innovation of quarterly district BBUI, z-scored "
                 "within district (derive_innovation_shock)",
        "estimator": "panel_lp_dk: two-way FE + Driscoll-Kraay SE, "
                     f"n_lags={N_LAGS}, horizons 0-{max(HORIZONS)}",
        "band": "sup-t (Montiel Olea & Plagborg-Møller 2019 JAE, Alg. 2) "
                f"on {B} district-cluster bootstrap draws",
    }
    (OUT_DIR / "panel_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    ur = results["urate"]["irf"]
    summary = {
        "mode": "fast" if args.fast else "full",
        "n_states": int(merged["state"].nunique()),
        "n_districts": int(merged["district"].nunique()),
        "panel_span": f"{merged['date'].min().date()}..{merged['date'].max().date()}",
        "n_state_quarters": int(len(merged)),
        "urate_h0": float(ur["beta"].iloc[0]),
        "urate_h4": float(ur["beta"].iloc[4]),
        "urate_h8": float(ur["beta"].iloc[8]),
        "urate_h12": float(ur["beta"].iloc[12]),
        "urate_peak_h": int(ur.loc[ur["beta"].abs().idxmax(), "h"]),
        "urate_peak": float(ur.loc[ur["beta"].abs().idxmax(), "beta"]),
        "urate_supt_excludes_zero": [
            int(h) for h, lo, hi in zip(ur["h"], ur["supt_lo"],
                                        ur["supt_hi"])
            if lo > 0 or hi < 0],
        "emp_h8": float(results["emp100"]["irf"]["beta"].iloc[8]),
        "supt_crit_urate": results["urate"]["supt_crit"],
        "coverage_gaps_quarters": len(gaps),
        "runtime_seconds": round(time.time() - _T0, 1),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT_DIR / "run_log.txt").write_text("\n".join(_LOG_LINES) + "\n", encoding="utf-8")
    (OUT_DIR / "coverage_gaps.txt").write_text("\n".join(gaps) + "\n", encoding="utf-8")
    log(f"summary: {json.dumps(summary, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
