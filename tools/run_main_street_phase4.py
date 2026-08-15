#!/usr/bin/env python
"""Main Street Uncertainty — Phase 4: leave-own-district-out horse race.

Phase 3's shuffle-district placebo was *half-alive*: wrong-district
placebo interactions recovered ~half the true beta because district
BBUI innovations share a national component (mean pairwise correlation
+0.10; corr(own, leave-one-out mean) ~ +0.30). The design therefore
could not distinguish "own-district uncertainty" from "correlated
national uncertainty loading on manufacturing states". Phase 4 makes
that distinction the estimand:

    LOO shock:  loo_{d,t} = mean of the 11 OTHER districts'
                AR(2)-purged BBUI innovations (purge FIRST, then
                average — preserves each district's own AR dynamics;
                decision recorded in the batch-4 plan). z-scored over
                the 1992Q1+ estimation cells (sd recorded), so
                coefficients read per-s.d. like the own shock. The
                own coefficient's survival is invariant to this
                rescaling.

    Horse race: y_{s,t+h} - y_{s,t-1} = b_h (expo_s x own_{d(s),t})
                                      + c_h (expo_s x loo_{d(s),t})
                + state FE + district-x-quarter FE + 4 lags of (both
                interactions, outcome) + e

Pre-registered decision rule (batch-4 plan, 2026-07-21): if at the
frozen Phase-3 peak (h* = 9) the own-district coefficient keeps >= half
its baseline magnitude with the same sign, the "own-district
uncertainty" reading survives; if the LOO term absorbs it, the paper's
framing changes to "national uncertainty loading on exposed states" —
either way the verdict goes in the DRAFT abstract.

Placebo closure: the Phase-3 derangement placebo is re-run with the
TRUE LOO interaction held as a control, using the SAME seed and hence
the SAME derangement sequence as Phase 3 — a paired comparison. If the
half-alive placebo was pure national-component contamination, the
placebo mean should collapse toward zero once LOO is controlled.

Lead closure: Phase 3 found a -0.015 (WC p = 0.04) lead at h = -2. The
lead specs are re-run with the LOO control to see whether the
pre-shock deterioration of exposed states loads on the national
component rather than the own-district shock.

Everything numeric is imported from ``run_main_street_phase3``
(``estimate_horizon``, ``_two_way_demean``, ``derangement``) — the
estimator is identical by construction, and this script ASSERTS that
its single-shock specs reproduce the frozen Phase-3 ``irf_exposure.csv``
betas/SEs to 1e-9 before running anything new.

Inputs (all frozen Phase-2/3 outputs; no network):
    output/state_district_panel_quarterly.csv   panel incl. bbui_innov
    output/exposure_state.csv                   frozen 1990-91 shares
    output/phase3_summary.json                  frozen h*, beta*
    output/irf_exposure.csv                     replication reference
    output/placebo_shuffle.csv                  Phase-3 placebo (paired)
    output/pretrends.csv                        Phase-3 leads reference

Outputs (docs/research/main_street_uncertainty/output/):
    bbui_loo_innovations.csv    district x quarter own/LOO innovations
    irf_loo_horserace.csv       all Phase-4 LP paths + DK + WC p
    placebo_shuffle_loo.csv     paired placebo betas under LOO control
    fig_phase4_loo.(png|pdf)    horse race + placebo closure figure
    phase4_manifest.json, phase4_summary.json, run_log_phase4.txt
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

import run_main_street_phase3 as p3  # noqa: E402  (estimator core, reused verbatim)

OUT_DIR = p3.OUT_DIR
MERGED_CSV = p3.MERGED_CSV
EXPO_CSV = OUT_DIR / "exposure_state.csv"
PH3_SUMMARY = OUT_DIR / "phase3_summary.json"
PH3_IRF = OUT_DIR / "irf_exposure.csv"
PH3_PLACEBO = OUT_DIR / "placebo_shuffle.csv"
PH3_PRETRENDS = OUT_DIR / "pretrends.csv"

HORIZONS = p3.HORIZONS
LEADS = p3.LEADS
N_LAGS = p3.N_LAGS
BASELINE_START = p3.BASELINE_START

_T0 = time.time()
_LOG_LINES: list[str] = []


def log(msg: str) -> None:
    line = f"[{time.time() - _T0:7.1f}s] {msg}"
    print(line, flush=True)
    _LOG_LINES.append(line)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# LOO shock construction
# ---------------------------------------------------------------------------

def build_loo(merged: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Leave-one-out national innovation per (district, quarter).

    Purge-first-then-average: the per-district AR(2)-purged, per-district
    standardized innovations from the Phase-2 panel are averaged over the
    11 OTHER districts. Strict donor rule: the LOO value is missing
    unless all 11 donors are present that quarter (post-1992 this never
    binds). z-scored (ddof=0) over the 1992Q1+ cells actually used in
    estimation; sd and mean recorded in the manifest."""
    iw = (merged[["district", "date", "bbui_innov"]]
          .drop_duplicates(["district", "date"])
          .pivot(index="date", columns="district", values="bbui_innov"))
    loo_cols, donor_cols = {}, {}
    for d in iw.columns:
        others = iw.drop(columns=d)
        loo_cols[d] = others.mean(axis=1)
        donor_cols[d] = others.notna().sum(axis=1)
    loo = pd.DataFrame(loo_cols)
    donors = pd.DataFrame(donor_cols)
    loo = loo.where(donors == 11)

    est = loo[loo.index >= BASELINE_START].stack()
    mu, sd = float(est.mean()), float(est.std(ddof=0))
    loo_z = (loo - mu) / sd

    long = (loo.stack().rename("bbui_loo").reset_index()
            .rename(columns={"level_1": "district"}))
    long = long.merge(
        loo_z.stack().rename("bbui_loo_z").reset_index()
        .rename(columns={"level_1": "district"}),
        on=["date", "district"])
    long = long.merge(
        donors.stack().rename("n_donors").reset_index()
        .rename(columns={"level_1": "district"}),
        on=["date", "district"])
    long = long.merge(
        iw.stack().rename("bbui_innov_own").reset_index()
        .rename(columns={"level_1": "district"}),
        on=["date", "district"], how="left")

    post = iw[iw.index >= BASELINE_START]
    loo_post = loo[loo.index >= BASELINE_START]
    per_d = {d: float(post[d].corr(loo_post[d])) for d in iw.columns}
    own_s, loo_s = post.stack().align(loo_post.stack(), join="inner")
    diag = {
        "loo_mean_pre_z": mu, "loo_sd_pre_z": sd,
        "corr_own_loo_pooled": float(np.corrcoef(own_s, loo_s)[0, 1]),
        "corr_own_loo_by_district": {k: round(v, 3)
                                     for k, v in sorted(per_d.items())},
    }
    log(f"LOO shock: sd={sd:.4f} (pre z-score), corr(own, loo) pooled "
        f"{diag['corr_own_loo_pooled']:+.3f}, per-district range "
        f"[{min(per_d.values()):+.3f}, {max(per_d.values()):+.3f}]")
    return long, diag


# ---------------------------------------------------------------------------
# Multi-shock LP design (reduces to Phase 3's _design for one shock)
# ---------------------------------------------------------------------------

def _design_multi(panel: pd.DataFrame, outcome: str, shock_cols: list[str],
                  h: int, n_lags: int, use_y_lags: bool = True
                  ) -> tuple[pd.DataFrame, list[str]]:
    """Per-horizon LP design with one ``expo_z x shock`` interaction per
    shock column; the FIRST shock is focal. Lags of EVERY interaction
    (symmetric treatment) and optionally the outcome enter as controls.
    With a single shock column this reproduces Phase 3's ``_design``
    row-for-row and column-for-column (names aside)."""
    df = panel.sort_values(["state", "date"]).copy()
    inter_cols = []
    for j, sc in enumerate(shock_cols):
        cn = f"inter{j}"
        df[cn] = df["expo_z"] * df[sc]
        inter_cols.append(cn)
    g = df.groupby("state", sort=False)
    df["dy"] = g[outcome].shift(-h) - g[outcome].shift(1)
    cols = list(inter_cols)
    for lag in range(1, n_lags + 1):
        for cn in inter_cols:
            df[f"{cn}_L{lag}"] = g[cn].shift(lag)
            cols.append(f"{cn}_L{lag}")
        if use_y_lags:
            df[f"y_L{lag}"] = g[outcome].shift(lag)
            cols.append(f"y_L{lag}")
    keep = ["state", "district", "date", "dy"] + cols
    return df[keep].dropna().reset_index(drop=True), cols


def run_spec_multi(panel: pd.DataFrame, *, spec: str, outcome: str,
                   shock_cols: list[str], horizons: list[int],
                   wc_draws: np.ndarray | None, n_lags: int = N_LAGS,
                   use_y_lags: bool = True) -> pd.DataFrame:
    rows = []
    for h in horizons:
        sub, x_cols = _design_multi(panel, outcome, shock_cols, h, n_lags,
                                    use_y_lags=use_y_lags)
        res = p3.estimate_horizon(sub, x_cols, wc_draws=wc_draws)
        res.update({"spec": spec, "h": h})
        rows.append(res)
    out = pd.DataFrame(rows)
    hs = out.set_index("h")
    peak = int(hs["beta"].abs().idxmax())
    log(f"spec[{spec}]: peak h={peak} beta={hs.loc[peak, 'beta']:+.4f} "
        f"(DK se {hs.loc[peak, 'se_dk']:.4f}, "
        f"WC p={hs.loc[peak, 'p_wc']:.3f}) n={int(hs.loc[peak, 'n_obs'])}")
    return out


# ---------------------------------------------------------------------------
# Placebo closure (paired with Phase 3 by seed)
# ---------------------------------------------------------------------------

def placebo_closure(panel: pd.DataFrame, *, h_star: int, n_perm: int,
                    seed: int) -> pd.DataFrame:
    """Phase-3 shuffle-district placebo re-run with the TRUE LOO
    interaction as a control. Same seed => same derangement sequence as
    Phase 3's ``placebo_shuffle.csv`` => draws pair one-to-one."""
    rng = np.random.default_rng(seed)
    districts = sorted(panel["district"].unique())
    shock_wide = (panel[["district", "date", "bbui_innov"]]
                  .drop_duplicates(["district", "date"])
                  .pivot(index="date", columns="district",
                         values="bbui_innov"))
    rows = []
    for r in range(n_perm):
        p = p3.derangement(rng, len(districts))
        mapping = {districts[i]: districts[p[i]]
                   for i in range(len(districts))}
        pl = panel.copy()
        placebo_shock = shock_wide.stack().rename("placebo_innov")
        pl["placebo_district"] = pl["district"].map(mapping)
        pl = pl.merge(placebo_shock.reset_index()
                      .rename(columns={"district": "placebo_district"}),
                      on=["date", "placebo_district"], how="left")
        sub, x_cols = _design_multi(pl, "urate",
                                    ["placebo_innov", "bbui_loo_z"],
                                    h_star, N_LAGS)
        res = p3.estimate_horizon(sub, x_cols, wc_draws=None)
        rows.append({"draw": r, "beta": res["beta"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def fig_phase4(irf: pd.DataFrame, plc3: pd.DataFrame, plc4: pd.DataFrame,
               h_star: int, beta_hr: float, fig_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))

    ax = axes[0]
    ax.axhline(0.0, color="0.6", lw=0.8)
    base = irf[irf["spec"] == "urate_own"].sort_values("h")
    hr = irf[irf["spec"] == "urate_hr_own"].sort_values("h")
    loo = irf[irf["spec"] == "urate_hr_loo"].sort_values("h")
    ax.plot(base["h"], base["beta"], color="0.45", lw=1.6, ls="--",
            marker="o", ms=3, label="own shock, no control (Phase 3)")
    ax.fill_between(hr["h"], hr["lo_dk"], hr["hi_dk"], color="C0",
                    alpha=0.22)
    ax.plot(hr["h"], hr["beta"], color="C0", lw=2.0, marker="o", ms=4,
            label="own shock | LOO-national controlled")
    ax.plot(loo["h"], loo["beta"], color="C3", lw=1.6, ls=":", marker="s",
            ms=3, label="LOO-national coefficient")
    ax.set_title("Horse race: own-district vs leave-one-out national",
                 fontsize=10)
    ax.set_xlabel("quarters after shock", fontsize=9)
    ax.set_ylabel("pp per (1 s.d. exposure) x (1 s.d. shock)", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[1]
    bins = np.histogram_bin_edges(
        np.concatenate([plc3["beta"], plc4["beta"]]), bins=25)
    ax.hist(plc3["beta"], bins=bins, color="0.75", edgecolor="white",
            alpha=0.9, label=f"Phase 3 (mean {plc3['beta'].mean():+.3f})")
    ax.hist(plc4["beta"], bins=bins, color="C0", edgecolor="white",
            alpha=0.55, label=f"+ LOO control (mean {plc4['beta'].mean():+.3f})")
    ax.axvline(beta_hr, color="C3", lw=2.0,
               label=f"true own β | LOO (h={h_star})")
    ax.axvline(0.0, color="0.4", lw=0.8)
    ax.set_title("Placebo closure (same derangements, paired)", fontsize=10)
    ax.set_xlabel(f"placebo β at h={h_star}", fontsize=9)
    ax.set_ylabel("draws", fontsize=9)
    ax.legend(fontsize=7)

    for ax in axes:
        ax.tick_params(labelsize=8)
    fig.suptitle("Phase 4: purging the national component of district "
                 "BBUI innovations", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for suffix in (".png", ".pdf"):
        fig.savefig(fig_path.with_suffix(suffix),
                    dpi=200 if suffix == ".png" else None)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--wc-draws", type=int, default=999,
                    help="wild-cluster Rademacher draws (default 999, "
                         "matching Phase 3)")
    ap.add_argument("--placebo", type=int, default=200,
                    help="shuffle-district permutations (default 200, "
                         "paired with Phase 3)")
    ap.add_argument("--seed", type=int, default=42,
                    help="MUST stay 42 for the paired-placebo and "
                         "replication gates against the frozen Phase-3 "
                         "outputs")
    ap.add_argument("--fast", action="store_true",
                    help="quick mode: --wc-draws 199 --placebo 40 "
                         "(replication gate relaxed to betas/SEs only)")
    args = ap.parse_args(argv)
    B = 199 if args.fast else args.wc_draws
    n_perm = 40 if args.fast else args.placebo

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    merged = pd.read_csv(MERGED_CSV, parse_dates=["date"])
    expo = pd.read_csv(EXPO_CSV)
    ph3 = json.loads(PH3_SUMMARY.read_text())
    irf3 = pd.read_csv(PH3_IRF)
    plc3 = pd.read_csv(PH3_PLACEBO)
    h_star = int(ph3["h_star"])
    beta_star = float(ph3["beta_star"])
    log(f"frozen Phase-3 anchors: h*={h_star}, beta*={beta_star:+.5f}")

    # --- LOO shock ---------------------------------------------------------
    loo_long, loo_diag = build_loo(merged)
    loo_long.to_csv(OUT_DIR / "bbui_loo_innovations.csv", index=False)

    # --- panel (replicates Phase 3's base_mfg construction exactly) --------
    base = merged.merge(expo[["state", "mfg_z"]], on="state")
    base_mfg = (base.rename(columns={"mfg_z": "expo_z"})
                [["state", "district", "date", "urate", "emp100",
                  "bbui_innov", "expo_z"]]
                .dropna(subset=["expo_z"]))
    base_mfg = base_mfg[base_mfg["date"] >= BASELINE_START]
    base_mfg = base_mfg.merge(
        loo_long[["district", "date", "bbui_loo_z"]],
        on=["district", "date"], how="left")
    n_loo_missing = int(base_mfg["bbui_loo_z"].isna().sum())
    if n_loo_missing:
        raise RuntimeError(
            f"phase4: {n_loo_missing} post-{BASELINE_START} rows without "
            f"a LOO value — donor rule should never bind here")

    # --- wild-cluster draws: same seed + draw order as Phase 3 -------------
    wc_draws = rng.choice([-1.0, 1.0], size=(B, 12))

    # --- specs --------------------------------------------------------------
    irf_frames = []
    irf_frames.append(run_spec_multi(
        base_mfg, spec="urate_own", outcome="urate",
        shock_cols=["bbui_innov"], horizons=HORIZONS, wc_draws=wc_draws))
    irf_frames.append(run_spec_multi(
        base_mfg, spec="urate_loo_alone", outcome="urate",
        shock_cols=["bbui_loo_z"], horizons=HORIZONS, wc_draws=wc_draws))
    irf_frames.append(run_spec_multi(
        base_mfg, spec="urate_hr_own", outcome="urate",
        shock_cols=["bbui_innov", "bbui_loo_z"], horizons=HORIZONS,
        wc_draws=wc_draws))
    irf_frames.append(run_spec_multi(
        base_mfg, spec="urate_hr_loo", outcome="urate",
        shock_cols=["bbui_loo_z", "bbui_innov"], horizons=HORIZONS,
        wc_draws=wc_draws))
    emp_p = base_mfg.dropna(subset=["emp100"])
    irf_frames.append(run_spec_multi(
        emp_p, spec="emp_own", outcome="emp100",
        shock_cols=["bbui_innov"], horizons=HORIZONS, wc_draws=wc_draws))
    irf_frames.append(run_spec_multi(
        emp_p, spec="emp_hr_own", outcome="emp100",
        shock_cols=["bbui_innov", "bbui_loo_z"], horizons=HORIZONS,
        wc_draws=wc_draws))
    # Lead closure: Phase-3 lead convention (no lag controls).
    irf_frames.append(run_spec_multi(
        base_mfg, spec="urate_hr_leads", outcome="urate",
        shock_cols=["bbui_innov", "bbui_loo_z"], horizons=LEADS,
        wc_draws=wc_draws, n_lags=0, use_y_lags=False))
    irf = pd.concat(irf_frames, ignore_index=True)

    # --- replication gate against frozen Phase-3 outputs -------------------
    ref = (irf3[irf3["spec"] == "urate_mfg"].set_index("h")
           .loc[HORIZONS, ["beta", "se_dk"]])
    got = (irf[irf["spec"] == "urate_own"].set_index("h")
           .loc[HORIZONS, ["beta", "se_dk"]])
    gap = float(np.max(np.abs(ref.to_numpy() - got.to_numpy())))
    if gap > 1e-9:
        raise RuntimeError(
            f"phase4 replication gate FAILED: urate_own vs frozen "
            f"urate_mfg max |diff| = {gap:.3e} (> 1e-9)")
    ref_e = (irf3[irf3["spec"] == "emp_mfg"].set_index("h")
             .loc[HORIZONS, ["beta", "se_dk"]])
    got_e = (irf[irf["spec"] == "emp_own"].set_index("h")
             .loc[HORIZONS, ["beta", "se_dk"]])
    gap_e = float(np.max(np.abs(ref_e.to_numpy() - got_e.to_numpy())))
    if gap_e > 1e-9:
        raise RuntimeError(
            f"phase4 replication gate FAILED: emp_own vs frozen emp_mfg "
            f"max |diff| = {gap_e:.3e} (> 1e-9)")
    log(f"replication gate PASSED: urate {gap:.2e}, emp {gap_e:.2e} "
        f"vs frozen Phase-3 irf_exposure.csv")
    irf.to_csv(OUT_DIR / "irf_loo_horserace.csv", index=False)

    # --- decision rule (pre-registered) -------------------------------------
    b_base = float(irf[(irf["spec"] == "urate_own")
                       & (irf["h"] == h_star)]["beta"].iloc[0])
    hr_row = irf[(irf["spec"] == "urate_hr_own") & (irf["h"] == h_star)]
    b_hr = float(hr_row["beta"].iloc[0])
    ratio = b_hr / b_base if b_base != 0 else np.nan
    same_sign = bool(np.sign(b_hr) == np.sign(b_base))
    survives = bool(same_sign and abs(b_hr) >= 0.5 * abs(b_base))
    verdict = ("own-district reading SURVIVES" if survives
               else "LOO-national ABSORBS the differential")
    log(f"decision rule @ h={h_star}: baseline {b_base:+.4f} -> "
        f"horse-race own {b_hr:+.4f} (ratio {ratio:+.2f}, "
        f"WC p={float(hr_row['p_wc'].iloc[0]):.3f}) => {verdict}")

    # --- placebo closure (paired) -------------------------------------------
    plc4 = placebo_closure(base_mfg, h_star=h_star, n_perm=n_perm,
                           seed=args.seed + 1)
    plc4.to_csv(OUT_DIR / "placebo_shuffle_loo.csv", index=False)
    n_pair = min(len(plc3), len(plc4))
    paired = (plc3["beta"].iloc[:n_pair].to_numpy()
              - plc4["beta"].iloc[:n_pair].to_numpy())
    p_ri = float((1 + int((plc4["beta"].abs() >= abs(b_hr)).sum()))
                 / (len(plc4) + 1))
    log(f"placebo closure: Phase-3 mean {plc3['beta'].mean():+.4f} -> "
        f"with LOO control {plc4['beta'].mean():+.4f} "
        f"(paired mean drop {paired.mean():+.4f} over {n_pair} draws); "
        f"randomization p={p_ri:.3f}")

    # --- lead closure summary ------------------------------------------------
    pre3 = pd.read_csv(PH3_PRETRENDS)
    lead3 = (pre3[pre3["spec"] == "urate_mfg_leads"]
             .set_index("h")[["beta", "p_wc"]])
    lead4 = (irf[irf["spec"] == "urate_hr_leads"]
             .set_index("h")[["beta", "p_wc"]])
    for h in LEADS:
        log(f"lead h={h}: Phase 3 {lead3.loc[h, 'beta']:+.4f} "
            f"(p_wc {lead3.loc[h, 'p_wc']:.2f}) -> | LOO "
            f"{lead4.loc[h, 'beta']:+.4f} (p_wc {lead4.loc[h, 'p_wc']:.2f})")

    # --- figure ---------------------------------------------------------------
    fig_phase4(irf, plc3, plc4, h_star, b_hr,
               OUT_DIR / "fig_phase4_loo.png")
    log("figure written")

    # --- manifest + summary ----------------------------------------------------
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "state_district_panel_quarterly.csv": {
                "sha256": sha256_file(MERGED_CSV)},
            "exposure_state.csv": {"sha256": sha256_file(EXPO_CSV)},
            "phase3_summary.json": {"sha256": sha256_file(PH3_SUMMARY)},
            "irf_exposure.csv": {"sha256": sha256_file(PH3_IRF)},
            "placebo_shuffle.csv": {"sha256": sha256_file(PH3_PLACEBO)},
            "pretrends.csv": {"sha256": sha256_file(PH3_PRETRENDS)},
        },
        "loo_definition": "mean of the 11 other districts' AR(2)-purged, "
                          "per-district-standardized BBUI innovations "
                          "(purge first, then average); strict 11-donor "
                          "rule; z-scored ddof=0 over 1992Q1+ cells",
        "loo_z_moments": {"mean": loo_diag["loo_mean_pre_z"],
                          "sd": loo_diag["loo_sd_pre_z"]},
        "estimator": "identical to Phase 3 (imported estimate_horizon); "
                     f"horse race adds the LOO interaction + its {N_LAGS} "
                     "lags as controls",
        "replication_gate": {"urate_max_absdiff": gap,
                             "emp_max_absdiff": gap_e, "tol": 1e-9},
        "wc_draws": B, "placebo_perms": n_perm, "seed": args.seed,
        "paired_with_phase3_placebo": True,
    }
    (OUT_DIR / "phase4_manifest.json").write_text(
        json.dumps(manifest, indent=2))

    def _path(spec: str) -> dict:
        s = irf[irf["spec"] == spec].set_index("h")
        return {int(h): {"beta": round(float(r["beta"]), 5),
                         "se_dk": round(float(r["se_dk"]), 5),
                         "p_wc": (None if pd.isna(r["p_wc"])
                                  else round(float(r["p_wc"]), 3))}
                for h, r in s.iterrows()}

    summary = {
        "mode": "fast" if args.fast else "full",
        "h_star_frozen": h_star,
        "decision_rule": {
            "h": h_star,
            "baseline_beta": b_base,
            "horserace_own_beta": b_hr,
            "horserace_own_p_wc": float(hr_row["p_wc"].iloc[0]),
            "horserace_own_se_dk": float(hr_row["se_dk"].iloc[0]),
            "ratio": ratio,
            "same_sign": same_sign,
            "threshold": 0.5,
            "own_survives": survives,
            "verdict": verdict,
        },
        "loo_diagnostics": loo_diag,
        "placebo_closure": {
            "phase3_mean": float(plc3["beta"].mean()),
            "phase4_mean": float(plc4["beta"].mean()),
            "phase4_sd": float(plc4["beta"].std(ddof=1)),
            "paired_mean_drop": float(paired.mean()),
            "n_paired": int(n_pair),
            "collapse_ratio": (float(plc4["beta"].mean()
                                     / plc3["beta"].mean())
                               if plc3["beta"].mean() != 0 else None),
            "randomization_p_vs_horserace_beta": p_ri,
        },
        "lead_closure": {
            int(h): {"phase3_beta": round(float(lead3.loc[h, "beta"]), 5),
                     "phase3_p_wc": round(float(lead3.loc[h, "p_wc"]), 3),
                     "phase4_beta": round(float(lead4.loc[h, "beta"]), 5),
                     "phase4_p_wc": round(float(lead4.loc[h, "p_wc"]), 3)}
            for h in LEADS
        },
        "paths": {spec: _path(spec) for spec in irf["spec"].unique()},
        "runtime_seconds": round(time.time() - _T0, 1),
    }
    (OUT_DIR / "phase4_summary.json").write_text(
        json.dumps(summary, indent=2))
    (OUT_DIR / "run_log_phase4.txt").write_text("\n".join(_LOG_LINES) + "\n")
    log("summary written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
