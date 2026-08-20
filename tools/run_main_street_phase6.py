#!/usr/bin/env python
"""Main Street Uncertainty — Phase 6: full shift-share exposure vector.

Phase 3-5 used a SINGLE pre-determined industry share (1990-91
manufacturing, mining as robustness) and flagged the full shift-share
vector as future work gated on a keyed source. It turns out no key is
needed: the same key-free FRED CES state mirrors used since phase 3
carry the ENTIRE 11-supersector NAICS partition from 1990-01
({ST}MFG, {ST}NRMN, {ST}CONS, {ST}TRAD, {ST}INFO, {ST}FIRE, {ST}PBSV,
{ST}EDUH, {ST}LEIH, {ST}SRVO, {ST}GOVT; SA first, NSA fallback —
verified live 2026-07-22, e.g. California information only exists as
CAINFON). Same source, same {ST}NA denominator, same 1990Q1-1991Q4
frozen base period as the phase-3 manufacturing share — the phase-3
number is reproduced as a hard gate before anything new runs.

Design. The phase-4 specification with the single expo x shock
interaction generalized to the exposure VECTOR: the 11 shares sum to
~1 per state, so one division (other services, SRVO) is omitted as
the reference and K = 10 z-scored share interactions enter jointly:

    y_{s,t+h} - y_{s,t-1} = sum_k b_k^h (expo_{s,k} x shock_{d(s),t})
        + state FE + district-x-quarter FE + 4 lags of (each
        interaction, outcome) + e

Model A: own-district shock only. Model B: adds the 10 LOO-national
interactions (phase-4 horse race, vectorized). Inference: per-division
Driscoll-Kraay and wild-cluster bootstrap-t p (CGM 2008, focal-column
reordering, B draws shared with the phase-3/4 seed), plus a JOINT
Driscoll-Kraay Wald test that all 10 own-interaction coefficients are
zero at each horizon. Headline question: does the manufacturing
differential survive with the other 9 divisions' exposures controlled,
and which divisions carry independent signal?

Outputs (docs/research/main_street_uncertainty/output/):
    ces_supersector_shares_9091.csv   frozen 51-state x 11-sector table
    irf_shiftshare.csv                per-division paths, models A + B
    wald_shiftshare.csv               joint DK Wald by horizon
    fig_phase6_shiftshare.(png|pdf)
    phase6_manifest.json, phase6_summary.json, run_log_phase6.txt
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from puremacro.inference.dk import driscoll_kraay  # noqa: E402
from puremacro._linalg import inv_xtx  # noqa: E402

import run_main_street_phase3 as p3  # noqa: E402
import run_main_street_phase4 as p4  # noqa: E402

OUT_DIR = p3.OUT_DIR
MERGED_CSV = p3.MERGED_CSV
EXPO3_CSV = OUT_DIR / "exposure_state.csv"
SHARES_CSV = OUT_DIR / "ces_supersector_shares_9091.csv"

#: NAICS supersectors on the FRED CES state mirrors (SA id stem; the
#: NSA fallback appends 'N', phase-3 convention). SRVO is the omitted
#: reference division in the regressions.
SECTORS: dict[str, str] = {
    "mfg": "MFG", "mining": "NRMN", "construction": "CONS",
    "trade_transport": "TRAD", "information": "INFO",
    "financial": "FIRE", "prof_business": "PBSV",
    "education_health": "EDUH", "leisure_hosp": "LEIH",
    "other_services": "SRVO", "government": "GOVT",
}
OMIT = "other_services"
K_SECTORS = [s for s in SECTORS if s != OMIT]

HORIZONS = p3.HORIZONS
N_LAGS = p3.N_LAGS
BASE_START, BASE_END = p3.BASE_START, p3.BASE_END
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


# ---------------------------------------------------------------------------
# Exposure vector
# ---------------------------------------------------------------------------

def build_shares(merged: pd.DataFrame, *, refresh: bool) -> pd.DataFrame:
    if SHARES_CSV.exists() and not refresh:
        out = pd.read_csv(SHARES_CSV)
        log(f"shares reused from {SHARES_CSV.name} ({len(out)} states; "
            "--refresh-shares to rebuild)")
        return out
    emp = merged[["state", "date", "emp100"]].dropna().copy()
    emp["emp_thousands"] = np.exp(emp["emp100"] / 100.0)
    base = emp[(emp["date"] >= BASE_START) & (emp["date"] <= BASE_END)]
    rows = []
    for st in ALL_STATES:
        nf = base[base["state"] == st].set_index("date")["emp_thousands"]
        rec: dict = {"state": st}
        for name, stem in SECTORS.items():
            share = np.nan
            used = None
            for sid in (f"{st}{stem}", f"{st}{stem}N"):
                try:
                    q = p3.fetch_fred_q(sid)
                except Exception:
                    continue
                qb = q[(q.index >= BASE_START) & (q.index <= BASE_END)]
                common = qb.index.intersection(nf.index)
                if len(common) < 4:
                    continue
                share = float((qb.loc[common] / nf.loc[common]).mean())
                used = sid
                break
            rec[name] = share
            rec[f"{name}_series_id"] = used
        rows.append(rec)
    out = pd.DataFrame(rows)
    out["share_sum"] = out[list(SECTORS)].sum(axis=1)
    log(f"shares: coverage per sector: "
        + ", ".join(f"{s}={int(out[s].notna().sum())}" for s in SECTORS))
    log(f"share sums: min {out['share_sum'].min():.3f}, "
        f"median {out['share_sum'].median():.3f}, "
        f"max {out['share_sum'].max():.3f}")
    out.to_csv(SHARES_CSV, index=False)
    return out


# ---------------------------------------------------------------------------
# Joint Driscoll-Kraay Wald on the first K columns
# ---------------------------------------------------------------------------

def dk_wald(sub: pd.DataFrame, x_cols: list[str], k: int) -> dict:
    g_state = p3._codes(sub["state"])
    dt_key = (sub["district"].astype("string") + "|"
              + sub["date"].astype("string"))
    g_dt = p3._codes(dt_key)
    M = np.column_stack([sub["dy"].to_numpy(float)]
                        + [sub[c].to_numpy(float) for c in x_cols])
    W = p3._two_way_demean(M, g_state, g_dt)
    y_w, X_w = W[:, 0], W[:, 1:]
    n = len(y_w)
    XtX_inv = inv_xtx(X_w, name="run_main_street_phase6.dk_wald")
    beta = XtX_inv @ X_w.T @ y_w
    u = y_w - X_w @ beta
    score = X_w * u[:, None]
    L = max(1, int(round(4 * (n / 100) ** (2 / 9))))
    S = driscoll_kraay(score, sub["date"].to_numpy(), lags=L)
    V = XtX_inv @ S @ XtX_inv
    bk, Vk = beta[:k], V[:k, :k]
    try:
        stat = float(bk @ np.linalg.solve(Vk, bk))
        pval = float(stats.chi2.sf(stat, df=k))
    except np.linalg.LinAlgError:
        stat, pval = np.nan, np.nan
    return {"wald": stat, "df": k, "p_wald_dk": pval,
            "betas": bk.tolist(), "n_obs": int(n)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--wc-draws", type=int, default=999)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--refresh-shares", action="store_true")
    ap.add_argument("--fast", action="store_true",
                    help="quick mode: --wc-draws 199")
    args = ap.parse_args(argv)
    B = 199 if args.fast else args.wc_draws

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    merged = pd.read_csv(MERGED_CSV, parse_dates=["date"])
    shares = build_shares(merged, refresh=args.refresh_shares)

    # --- gate: reproduce the phase-3 manufacturing share --------------------
    expo3 = pd.read_csv(EXPO3_CSV)
    chk = shares.merge(expo3[["state", "mfg_share_9091"]], on="state")
    gap = float(np.nanmax(np.abs(chk["mfg"] - chk["mfg_share_9091"])))
    if gap > 1e-9:
        raise RuntimeError(
            f"phase6 gate FAILED: mfg share vs frozen phase-3 "
            f"exposure_state.csv max |diff| = {gap:.3e}")
    log(f"phase-3 replication gate PASSED (mfg share max diff {gap:.1e})")

    # --- panel: z-scored exposure vector + shocks ----------------------------
    loo_long, _ = p4.build_loo(merged)
    z = shares[["state"] + list(SECTORS)].copy()
    for s in SECTORS:
        v = z[s]
        z[s] = (v - v.mean()) / v.std(ddof=0)
    base = merged.merge(z, on="state")
    base = base[["state", "district", "date", "urate", "bbui_innov"]
                + list(SECTORS)].dropna(subset=K_SECTORS)
    base = base[base["date"] >= BASELINE_START]
    base = base.merge(loo_long[["district", "date", "bbui_loo_z"]],
                      on=["district", "date"], how="left")
    log(f"panel: {base['state'].nunique()} states, {len(base)} rows")

    wc_draws = rng.choice([-1.0, 1.0], size=(B, 12))

    def design(panel: pd.DataFrame, h: int, sectors: list[str],
               shocks: list[str], focal: str, focal_shock: str
               ) -> tuple[pd.DataFrame, list[str]]:
        """Multi-sector interacted design; focal interaction first."""
        df = panel.sort_values(["state", "date"]).copy()
        inter_cols = []
        for sc in shocks:
            for s in sectors:
                cn = f"i_{s}_{sc}"
                df[cn] = df[s] * df[sc]
                inter_cols.append(cn)
        fc = f"i_{focal}_{focal_shock}"
        ordered = [fc] + [c for c in inter_cols if c != fc]
        g = df.groupby("state", sort=False)
        df["dy"] = g["urate"].shift(-h) - g["urate"].shift(1)
        cols = list(ordered)
        for lag in range(1, N_LAGS + 1):
            for cn in inter_cols:
                df[f"{cn}_L{lag}"] = g[cn].shift(lag)
                cols.append(f"{cn}_L{lag}")
            df[f"y_L{lag}"] = g["urate"].shift(lag)
            cols.append(f"y_L{lag}")
        keep = ["state", "district", "date", "dy"] + cols
        return df[keep].dropna().reset_index(drop=True), cols

    # --- Model A: own shock, joint Wald + per-division focal stats ----------
    irf_rows, wald_rows = [], []
    for h in HORIZONS:
        sub, cols = design(base, h, K_SECTORS, ["bbui_innov"],
                           K_SECTORS[0], "bbui_innov")
        w = dk_wald(sub, cols, k=len(K_SECTORS))
        w["h"] = h
        wald_rows.append(w)
    wald = pd.DataFrame(wald_rows)
    wald.to_csv(OUT_DIR / "wald_shiftshare.csv", index=False)
    sig_h = [int(r["h"]) for _, r in wald.iterrows()
             if r["p_wald_dk"] < 0.10]
    log(f"Model A joint DK Wald (df=10): p<0.10 at h={sig_h}")

    for model, shocks in (("A_own", ["bbui_innov"]),
                          ("B_horserace", ["bbui_innov", "bbui_loo_z"])):
        for s in K_SECTORS:
            for h in HORIZONS:
                sub, cols = design(base, h, K_SECTORS, shocks,
                                   s, "bbui_innov")
                res = p3.estimate_horizon(sub, cols, wc_draws=wc_draws)
                res.update({"model": model, "sector": s, "h": h})
                irf_rows.append(res)
            b9 = [r for r in irf_rows
                  if r["model"] == model and r["sector"] == s]
            pk = max(b9, key=lambda r: abs(r["beta"]))
            log(f"{model}/{s}: peak h={pk['h']} beta={pk['beta']:+.4f} "
                f"(WC p={pk['p_wc']:.3f})")
    irf = pd.DataFrame(irf_rows)
    irf.to_csv(OUT_DIR / "irf_shiftshare.csv", index=False)

    # --- figure: forest of h=5 and h=9 own-shock coefficients ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=True)
    from scipy.stats import norm as _norm
    zc = _norm.ppf(0.95)
    for ax, model, title in (
            (axes[0], "A_own", "Model A: own-district shock"),
            (axes[1], "B_horserace",
             "Model B: own | LOO-national controlled")):
        sub = irf[(irf["model"] == model) & (irf["h"] == 5)]
        ypos = np.arange(len(K_SECTORS))[::-1]
        for y, s in zip(ypos, K_SECTORS):
            r = sub[sub["sector"] == s].iloc[0]
            c = "#e34948" if s == "mfg" else "#52514e"
            ax.plot([r["beta"] - zc * r["se_dk"],
                     r["beta"] + zc * r["se_dk"]], [y, y], color=c,
                    lw=2.0, solid_capstyle="butt")
            ax.plot(r["beta"], y, "o", color=c, ms=5)
        ax.axvline(0, color="0.6", lw=0.8)
        ax.set_yticks(ypos)
        ax.set_yticklabels(K_SECTORS, fontsize=8)
        ax.set_title(title + " (h=5)", fontsize=10)
        ax.set_xlabel("pp per (1 s.d. share) x (1 s.d. shock)", fontsize=8)
    fig.suptitle("Phase 6: the full supersector exposure vector "
                 "(90% DK intervals; reference division: other services)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for suffix in (".png", ".pdf"):
        fig.savefig((OUT_DIR / "fig_phase6_shiftshare.png")
                    .with_suffix(suffix),
                    dpi=200 if suffix == ".png" else None)
    plt.close(fig)
    log("figure written")

    # --- manifest + summary ---------------------------------------------------
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "exposure": "FRED CES state mirrors, 11 NAICS supersectors, "
                    "SA-then-NSA fallback, base 1990Q1-1991Q4 frozen, "
                    "share of {ST}NA; reference division omitted: "
                    f"{OMIT}",
        "inputs": {
            "ces_supersector_shares_9091.csv": {
                "sha256": sha256_file(SHARES_CSV)},
            "state_district_panel_quarterly.csv": {
                "sha256": sha256_file(MERGED_CSV)},
            "exposure_state.csv (gate)": {
                "sha256": sha256_file(EXPO3_CSV)},
        },
        "estimator": "phase-3/4 estimator imported; K=10 z-scored share "
                     f"interactions jointly; {N_LAGS} lags of every "
                     "interaction + outcome; joint DK Wald df=10",
        "wc_draws": B, "seed": args.seed,
    }
    (OUT_DIR / "phase6_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    def _peak(model: str, s: str) -> dict:
        rows = irf[(irf["model"] == model) & (irf["sector"] == s)]
        pk = rows.loc[rows["beta"].abs().idxmax()]
        return {"h": int(pk["h"]), "beta": round(float(pk["beta"]), 5),
                "se_dk": round(float(pk["se_dk"]), 5),
                "p_wc": round(float(pk["p_wc"]), 3)}

    summary = {
        "mode": "fast" if args.fast else "full",
        "wald_p_by_h": {int(r["h"]): round(float(r["p_wald_dk"]), 4)
                        for _, r in wald.iterrows()},
        "peaks_model_A": {s: _peak("A_own", s) for s in K_SECTORS},
        "peaks_model_B": {s: _peak("B_horserace", s) for s in K_SECTORS},
        "mfg_at_h5_h9_A": {int(h): round(float(row["beta"]), 5)
                           for h, row in irf[(irf["model"] == "A_own")
                                             & (irf["sector"] == "mfg")
                                             & (irf["h"].isin([5, 9]))]
                           .set_index("h").iterrows()},
        "runtime_seconds": round(time.time() - _T0, 1),
    }
    (OUT_DIR / "phase6_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    (OUT_DIR / "run_log_phase6.txt").write_text("\n".join(_LOG_LINES) + "\n", encoding="utf-8")
    log("summary written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
