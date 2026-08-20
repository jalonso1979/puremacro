#!/usr/bin/env python
"""Main Street Distress --- Phase 8: two inference-robustness checks.

Two referee points about the leave-one-out horse race (Section~loo), neither
touching the frozen Phase-3/4 pipeline:

1. CLUSTER DIMENSION. The leave-one-out national regressor
   $x_s\\,\\varepsilon^{loo}_{d,t}$ is, up to the $-1/11$ reweighting, a common
   time factor scaled by exposure: it varies mostly in $t$. Clustering the
   wild bootstrap on 12 districts draws independent signs per district and so
   breaks the cross-district correlation a common factor is made of, which can
   make the district-clustered $p$ too small. This script re-runs the national
   and own coefficients clustering the wild bootstrap by QUARTER (the
   dimension the national regressor's variation lives in) and reports both
   $p$-values side by side. If the national result survives quarter
   clustering, the "most robust coefficient" claim holds; if it inflates, the
   claim was a clustering artifact.

2. COMMON-WINDOW STANDARDIZATION. Phase 4 standardizes the own shock over its
   full 1970-2025 history but the leave-one-out shock over the 1992+
   estimation cells. The own coefficient's SURVIVAL ratio is invariant to own
   scaling, but the amount it falls when the national control enters is not.
   This script re-standardizes BOTH shocks pooled over the common 1992+ window
   and re-runs the horse race, reporting whether the roughly-half decomposition
   is robust to putting the two shocks on the same scale.

Estimator: the Phase-3 within-OLS + Driscoll-Kraay + wild-cluster machinery,
copied here with the cluster dimension parameterized (district / quarter). A
gate asserts that, under district clustering and the frozen standardization,
this reproduces the frozen Phase-4 horse-race betas/SEs to 1e-9 before any new
number is reported.

Inputs (all frozen; no network):
    output/state_district_panel_quarterly.csv
    output/exposure_state.csv
    output/bbui_loo_innovations.csv         Phase-4 leave-one-out innovations
    output/irf_loo_horserace.csv            Phase-4 replication reference

Outputs (docs/research/main_street_uncertainty/output/):
    irf_phase8_inference.csv    all specs x cluster-dimension paths
    phase8_summary.json, run_log_phase8.txt
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

import run_main_street_phase3 as p3  # noqa: E402
import run_main_street_phase4 as p4  # noqa: E402
from run_main_street_phase3 import (  # noqa: E402
    ALPHA, driscoll_kraay, inv_xtx,
)

OUT_DIR = p3.OUT_DIR
MERGED_CSV = p3.MERGED_CSV
EXPO_CSV = OUT_DIR / "exposure_state.csv"
LOO_CSV = OUT_DIR / "bbui_loo_innovations.csv"
PH4_IRF = OUT_DIR / "irf_loo_horserace.csv"
HORIZONS = p3.HORIZONS
N_LAGS = p3.N_LAGS
BASELINE_START = p3.BASELINE_START

_T0 = time.time()
_LOG: list[str] = []


def log(msg: str) -> None:
    line = f"[{time.time() - _T0:7.1f}s] {msg}"
    print(line, flush=True)
    _LOG.append(line)


def _codes(values) -> np.ndarray:
    return pd.factorize(values, sort=True)[0]


def estimate_horizon_cl(sub: pd.DataFrame, x_cols: list[str], *,
                        wc_draws: np.ndarray | None,
                        cluster: str = "district") -> dict:
    """Phase-3 estimator with the wild-cluster dimension parameterized.

    ``cluster='district'`` reproduces ``run_main_street_phase3.estimate_horizon``
    exactly; ``cluster='quarter'`` clusters the CRVE and the wild bootstrap on
    the quarter (``date``) instead. ``wc_draws`` must have as many columns as
    the chosen cluster has levels in ``sub``."""
    g_state = _codes(sub["state"])
    dt_key = sub["district"].astype("string") + "|" + sub["date"].astype("string")
    g_dt = _codes(dt_key)
    g_cl = _codes(sub[("district" if cluster == "district" else "date")])
    n_cl = int(g_cl.max()) + 1

    M = np.column_stack([sub["dy"].to_numpy(float)]
                        + [sub[c].to_numpy(float) for c in x_cols])
    W = p3._two_way_demean(M, g_state, g_dt)
    y_w, X_w = W[:, 0], W[:, 1:]
    n, k = X_w.shape
    XtX_inv = inv_xtx(X_w, name="phase8")
    beta = XtX_inv @ X_w.T @ y_w
    u = y_w - X_w @ beta

    score = X_w * u[:, None]
    L = max(1, int(round(4 * (n / 100) ** (2 / 9))))
    S = driscoll_kraay(score, sub["date"].to_numpy(), lags=L)
    vcov_dk = XtX_inv @ S @ XtX_inv
    se_dk = float(np.sqrt(np.diag(vcov_dk))[0])

    a = XtX_inv[:, 0]
    z = X_w @ a
    ghat = np.array([float(z[g_cl == c] @ u[g_cl == c]) for c in range(n_cl)])
    cfac = n_cl / (n_cl - 1)
    se_cr = math.sqrt(cfac * float(ghat @ ghat))
    t_cr = float(beta[0]) / se_cr if se_cr > 0 else np.nan

    p_wc = np.nan
    if wc_draws is not None and se_cr > 0:
        Xr = X_w[:, 1:]
        if Xr.shape[1] > 0:
            XtXr = inv_xtx(Xr, name="phase8.restricted")
            br = XtXr @ Xr.T @ y_w
            fr, ur = Xr @ br, y_w - Xr @ br
        else:
            fr, ur = np.zeros_like(y_w), y_w
        V = wc_draws[:, g_cl].T
        Ystar = fr[:, None] + p3._two_way_demean(ur[:, None] * V, g_state, g_dt)
        C = XtX_inv @ (X_w.T @ Ystar)
        Ustar = Ystar - X_w @ C
        var_b = np.zeros(Ustar.shape[1])
        for c in range(n_cl):
            m = g_cl == c
            var_b += (z[m] @ Ustar[m]) ** 2
        se_b = np.sqrt(cfac * var_b)
        ok = se_b > 0
        tstar = np.abs(C[0, ok] / se_b[ok])
        p_wc = float((1 + int((tstar >= abs(t_cr)).sum())) / (ok.sum() + 1))

    zc = norm.ppf(1 - ALPHA / 2)
    return {"beta": float(beta[0]), "se_dk": se_dk,
            "lo_dk": float(beta[0]) - zc * se_dk,
            "hi_dk": float(beta[0]) + zc * se_dk,
            "p_wc": p_wc, "n_obs": n, "n_clusters": n_cl}


def run_spec_cl(panel, *, spec, shock_cols, wc_draws, cluster, horizons):
    rows = []
    for h in horizons:
        sub, xc = p4._design_multi(panel, "urate", shock_cols, h, N_LAGS)
        r = estimate_horizon_cl(sub, xc, wc_draws=wc_draws, cluster=cluster)
        r.update({"spec": spec, "h": h, "cluster": cluster})
        rows.append(r)
    return pd.DataFrame(rows)


def _sumsq(z: np.ndarray, U: np.ndarray, g: np.ndarray, G: int):
    """sum_c (sum_{i in cluster c} z_i U_i)^2. U is (n,) -> float, or
    (n,B) -> (B,). Vectorized scatter-add over clusters."""
    if U.ndim == 1:
        S = np.zeros(G)
        np.add.at(S, g, z * U)
        return float((S ** 2).sum())
    S = np.zeros((G, U.shape[1]))
    np.add.at(S, g, z[:, None] * U)
    return (S ** 2).sum(axis=0)


def estimate_horizon_2way(sub: pd.DataFrame, x_cols: list[str], *,
                          wc_dist: np.ndarray, wc_time: np.ndarray) -> dict:
    """Two-way (district x quarter) cluster-robust wild bootstrap-t.

    Point variance is Cameron-Gelbach-Miller (2011): $V_A + V_B - V_{A\\cap B}$,
    each term the CRVE of the focal coefficient clustered on districts,
    quarters, and district-quarter cells respectively (per-dimension
    $G/(G-1)$). The null-imposed wild bootstrap follows MacKinnon-Nash-Webb
    (2021): each replication multiplies the restricted residual by the PRODUCT
    of an independent district Rademacher sign and an independent quarter
    Rademacher sign, and the bootstrap $t$ uses the same two-way CRVE. With
    twelve districts this is the honest but still small-$G$ inference."""
    g_state = _codes(sub["state"])
    dt_key = sub["district"].astype("string") + "|" + sub["date"].astype("string")
    g_dt = _codes(dt_key)
    g_dist = _codes(sub["district"])
    g_time = _codes(sub["date"])
    nA, nB, nAB = int(g_dist.max()) + 1, int(g_time.max()) + 1, int(g_dt.max()) + 1
    cA, cB, cAB = nA / (nA - 1), nB / (nB - 1), nAB / (nAB - 1)

    M = np.column_stack([sub["dy"].to_numpy(float)]
                        + [sub[c].to_numpy(float) for c in x_cols])
    W = p3._two_way_demean(M, g_state, g_dt)
    y_w, X_w = W[:, 0], W[:, 1:]
    n = X_w.shape[0]
    XtX_inv = inv_xtx(X_w, name="phase8.2way")
    beta = XtX_inv @ X_w.T @ y_w
    u = y_w - X_w @ beta

    score = X_w * u[:, None]
    L = max(1, int(round(4 * (n / 100) ** (2 / 9))))
    S = driscoll_kraay(score, sub["date"].to_numpy(), lags=L)
    se_dk = float(np.sqrt(np.diag(XtX_inv @ S @ XtX_inv))[0])

    a = XtX_inv[:, 0]
    z = X_w @ a

    def crve2(U):
        return (cA * _sumsq(z, U, g_dist, nA)
                + cB * _sumsq(z, U, g_time, nB)
                - cAB * _sumsq(z, U, g_dt, nAB))

    var0 = crve2(u)
    se_cr = math.sqrt(var0) if var0 > 0 else np.nan
    t_cr = float(beta[0]) / se_cr if se_cr and se_cr > 0 else np.nan

    p_wc = np.nan
    if se_cr and se_cr > 0:
        Xr = X_w[:, 1:]
        if Xr.shape[1] > 0:
            XtXr = inv_xtx(Xr, name="phase8.2way.r")
            br = XtXr @ Xr.T @ y_w
            fr, ur = Xr @ br, y_w - Xr @ br
        else:
            fr, ur = np.zeros_like(y_w), y_w
        V = wc_dist[:, g_dist].T * wc_time[:, g_time].T      # (n,B) product signs
        Ystar = fr[:, None] + p3._two_way_demean(ur[:, None] * V, g_state, g_dt)
        C = XtX_inv @ (X_w.T @ Ystar)
        Ustar = Ystar - X_w @ C
        varb = crve2(Ustar)                                   # (B,)
        ok = varb > 0
        tstar = np.abs(C[0, ok] / np.sqrt(varb[ok]))
        p_wc = float((1 + int((tstar >= abs(t_cr)).sum())) / (int(ok.sum()) + 1))

    zc = norm.ppf(1 - ALPHA / 2)
    return {"beta": float(beta[0]), "se_dk": se_dk,
            "lo_dk": float(beta[0]) - zc * se_dk,
            "hi_dk": float(beta[0]) + zc * se_dk,
            "p_wc": p_wc, "n_obs": n, "n_clusters": nA}


def run_spec_2way(panel, *, spec, shock_cols, wc_dist, wc_time, horizons):
    rows = []
    for h in horizons:
        sub, xc = p4._design_multi(panel, "urate", shock_cols, h, N_LAGS)
        r = estimate_horizon_2way(sub, xc, wc_dist=wc_dist, wc_time=wc_time)
        r.update({"spec": spec, "h": h, "cluster": "twoway"})
        rows.append(r)
    return pd.DataFrame(rows)


def zscore_pooled(panel: pd.DataFrame, col: str) -> pd.Series:
    """Pooled z-score over the 1992+ cells (ddof=0), matching how Phase 4
    standardizes the leave-one-out shock."""
    v = panel.loc[panel["date"] >= BASELINE_START, col]
    return (panel[col] - v.mean()) / v.std(ddof=0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--wc-draws", type=int, default=999)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args(argv)
    B = 199 if args.fast else args.wc_draws
    rng = np.random.default_rng(args.seed)

    merged = pd.read_csv(MERGED_CSV, parse_dates=["date"])
    expo = pd.read_csv(EXPO_CSV)
    loo = pd.read_csv(LOO_CSV, parse_dates=["date"])
    ph4 = pd.read_csv(PH4_IRF)

    base = merged.merge(expo[["state", "mfg_z"]], on="state")
    base = (base.rename(columns={"mfg_z": "expo_z"})
            [["state", "district", "date", "urate", "emp100",
              "bbui_innov", "expo_z"]].dropna(subset=["expo_z"]))
    base = base[base["date"] >= BASELINE_START]
    base = base.merge(loo[["district", "date", "bbui_loo_z"]],
                      on=["district", "date"], how="left")
    if int(base["bbui_loo_z"].isna().sum()):
        raise RuntimeError("phase8: missing LOO values post-1992")

    # Phase-7 distress/uncertainty component shocks (optional; present once
    # Phase 7 has run) so the sentiment horse race can be quarter-clustered too.
    p7 = OUT_DIR / "phase7_innovations.csv"
    have_p7 = p7.exists()
    if have_p7:
        pin = pd.read_csv(p7, parse_dates=["date"])
        base = base.merge(pin[["district", "date", "innov_unc", "innov_sent"]],
                          on=["district", "date"], how="left")

    n_dist = base["district"].nunique()
    n_q = base["date"].nunique()
    wc_dist = rng.choice([-1.0, 1.0], size=(B, n_dist))
    wc_time = rng.choice([-1.0, 1.0], size=(B, n_q))
    log(f"clusters: {n_dist} districts, {n_q} quarters; B={B}")

    frames = []

    # (0) Replication gate: district clustering, frozen standardization, must
    #     reproduce the frozen Phase-4 national coefficient.
    loo_alone = run_spec_cl(base, spec="loo_alone_dist",
                            shock_cols=["bbui_loo_z"], wc_draws=wc_dist,
                            cluster="district", horizons=HORIZONS)
    ref = (ph4[ph4["spec"] == "urate_loo_alone"].set_index("h")
           .loc[HORIZONS, ["beta", "se_dk"]])
    got = loo_alone.set_index("h").loc[HORIZONS, ["beta", "se_dk"]]
    gap = float(np.max(np.abs(ref.to_numpy() - got.to_numpy())))
    if gap > 1e-9:
        raise RuntimeError(f"phase8 gate FAILED: national coef vs frozen "
                           f"Phase-4 max|diff| {gap:.2e} > 1e-9")
    log(f"replication gate PASSED (national coef vs frozen Phase-4 {gap:.1e})")
    frames.append(loo_alone)

    # (1) CLUSTER DIMENSION: national + own under quarter clustering.
    frames.append(run_spec_cl(base, spec="loo_alone_quarter",
                              shock_cols=["bbui_loo_z"], wc_draws=wc_time,
                              cluster="quarter", horizons=HORIZONS))
    frames.append(run_spec_cl(base, spec="own_alone_dist",
                              shock_cols=["bbui_innov"], wc_draws=wc_dist,
                              cluster="district", horizons=HORIZONS))
    frames.append(run_spec_cl(base, spec="own_alone_quarter",
                              shock_cols=["bbui_innov"], wc_draws=wc_time,
                              cluster="quarter", horizons=HORIZONS))
    # horse race, both cluster dimensions (focal = national, then own)
    for cl, wc in (("district", wc_dist), ("quarter", wc_time)):
        frames.append(run_spec_cl(base, spec=f"hr_loo_{cl}",
                                  shock_cols=["bbui_loo_z", "bbui_innov"],
                                  wc_draws=wc, cluster=cl, horizons=HORIZONS))
        frames.append(run_spec_cl(base, spec=f"hr_own_{cl}",
                                  shock_cols=["bbui_innov", "bbui_loo_z"],
                                  wc_draws=wc, cluster=cl, horizons=HORIZONS))

    # (2) COMMON-WINDOW STANDARDIZATION: both shocks pooled over 1992+.
    base_sw = base.copy()
    base_sw["own_sw"] = zscore_pooled(base_sw, "bbui_innov")
    base_sw["loo_sw"] = zscore_pooled(base_sw, "bbui_loo_z")
    frames.append(run_spec_cl(base_sw, spec="sw_own_base",
                              shock_cols=["own_sw"], wc_draws=wc_dist,
                              cluster="district", horizons=HORIZONS))
    frames.append(run_spec_cl(base_sw, spec="sw_hr_own",
                              shock_cols=["own_sw", "loo_sw"], wc_draws=wc_dist,
                              cluster="district", horizons=HORIZONS))
    frames.append(run_spec_cl(base_sw, spec="sw_hr_loo",
                              shock_cols=["loo_sw", "own_sw"], wc_draws=wc_dist,
                              cluster="district", horizons=HORIZONS))
    # (3) SENTIMENT HORSE RACE under quarter clustering (is the Phase-7
    #     uncertainty-component result also a district-clustering artifact?)
    if have_p7:
        u_ok = base.dropna(subset=["innov_unc", "innov_sent"])
        for cl, wc in (("district", wc_dist), ("quarter", wc_time)):
            frames.append(run_spec_cl(u_ok, spec=f"hr_unc_{cl}",
                                      shock_cols=["innov_unc", "innov_sent"],
                                      wc_draws=wc, cluster=cl, horizons=HORIZONS))
            frames.append(run_spec_cl(u_ok, spec=f"hr_sent_{cl}",
                                      shock_cols=["innov_sent", "innov_unc"],
                                      wc_draws=wc, cluster=cl, horizons=HORIZONS))

    # (4) TWO-WAY (district x quarter) cluster-robust wild bootstrap on the
    #     decision-relevant coefficients (CGM 2011 variance, MNW 2021 bootstrap).
    frames.append(run_spec_2way(base, spec="loo_alone_twoway",
                                shock_cols=["bbui_loo_z"],
                                wc_dist=wc_dist, wc_time=wc_time,
                                horizons=HORIZONS))
    frames.append(run_spec_2way(base, spec="hr_loo_twoway",
                                shock_cols=["bbui_loo_z", "bbui_innov"],
                                wc_dist=wc_dist, wc_time=wc_time,
                                horizons=HORIZONS))
    frames.append(run_spec_2way(base, spec="hr_own_twoway",
                                shock_cols=["bbui_innov", "bbui_loo_z"],
                                wc_dist=wc_dist, wc_time=wc_time,
                                horizons=HORIZONS))
    if have_p7:
        u_ok2 = base.dropna(subset=["innov_unc", "innov_sent"])
        frames.append(run_spec_2way(u_ok2, spec="hr_unc_twoway",
                                    shock_cols=["innov_unc", "innov_sent"],
                                    wc_dist=wc_dist, wc_time=wc_time,
                                    horizons=HORIZONS))
        frames.append(run_spec_2way(u_ok2, spec="hr_sent_twoway",
                                    shock_cols=["innov_sent", "innov_unc"],
                                    wc_dist=wc_dist, wc_time=wc_time,
                                    horizons=HORIZONS))

    irf = pd.concat(frames, ignore_index=True)
    irf.to_csv(OUT_DIR / "irf_phase8_inference.csv", index=False)

    def at(spec, h):
        r = irf[(irf["spec"] == spec) & (irf["h"] == h)]
        if r.empty:
            return None
        r = r.iloc[0]
        return {"beta": round(float(r["beta"]), 4),
                "se_dk": round(float(r["se_dk"]), 4),
                "p_wc": round(float(r["p_wc"]), 3),
                "n_clusters": int(r["n_clusters"])}

    def peak(spec):
        s = irf[irf["spec"] == spec].set_index("h")
        h = int(s["beta"].abs().idxmax())
        return h, at(spec, h)

    # national at its frozen peak h=12; own at frozen h=9
    h_nat, h_own = 12, 9
    summary = {
        "mode": "fast" if args.fast else "full",
        "replication_gate_max_absdiff": gap,
        "cluster_check": {
            "national_h12": {
                "alone_district": at("loo_alone_dist", h_nat),
                "alone_quarter": at("loo_alone_quarter", h_nat),
                "alone_twoway": at("loo_alone_twoway", h_nat),
                "hr_district": at("hr_loo_district", h_nat),
                "hr_quarter": at("hr_loo_quarter", h_nat),
                "hr_twoway": at("hr_loo_twoway", h_nat),
            },
            "own_h9": {
                "hr_district": at("hr_own_district", h_own),
                "hr_quarter": at("hr_own_quarter", h_own),
                "hr_twoway": at("hr_own_twoway", h_own),
            },
        },
        "common_window": {
            "own_base_h9": at("sw_own_base", h_own),
            "hr_own_h9": at("sw_hr_own", h_own),
            "hr_loo_h12": at("sw_hr_loo", h_nat),
            "own_survival_ratio_h9": (
                round(at("sw_hr_own", h_own)["beta"]
                      / at("sw_own_base", h_own)["beta"], 3)
                if at("sw_own_base", h_own)["beta"] else None),
        },
        "runtime_seconds": round(time.time() - _T0, 1),
    }
    if have_p7:
        h_unc = 7
        summary["sentiment_cluster_check"] = {
            "uncertainty_h7": {
                "hr_district": at("hr_unc_district", h_unc),
                "hr_quarter": at("hr_unc_quarter", h_unc),
                "hr_twoway": at("hr_unc_twoway", h_unc),
            },
            "distress_h9": {
                "hr_district": at("hr_sent_district", h_own),
                "hr_quarter": at("hr_sent_quarter", h_own),
                "hr_twoway": at("hr_sent_twoway", h_own),
            },
        }
    (OUT_DIR / "phase8_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log("=== CLUSTER DIMENSION (national, h=12) ===")
    log(f"  district-clustered: {at('loo_alone_dist', h_nat)}")
    log(f"  quarter-clustered : {at('loo_alone_quarter', h_nat)}")
    log(f"  two-way (DxQ)     : {at('loo_alone_twoway', h_nat)}")
    log(f"  horse race district: {at('hr_loo_district', h_nat)}")
    log(f"  horse race quarter : {at('hr_loo_quarter', h_nat)}")
    log(f"  horse race two-way : {at('hr_loo_twoway', h_nat)}")
    log("=== COMMON-WINDOW STANDARDIZATION (own, h=9) ===")
    log(f"  own baseline: {at('sw_own_base', h_own)}")
    log(f"  own | national: {at('sw_hr_own', h_own)} "
        f"(survival ratio {summary['common_window']['own_survival_ratio_h9']})")
    log(f"  national | own (h=12): {at('sw_hr_loo', h_nat)}")
    if have_p7:
        log("=== SENTIMENT HORSE RACE cluster check ===")
        log(f"  uncertainty|distress (h=7) district: {at('hr_unc_district', 7)}")
        log(f"  uncertainty|distress (h=7) quarter : {at('hr_unc_quarter', 7)}")
        log(f"  uncertainty|distress (h=7) two-way : {at('hr_unc_twoway', 7)}")
        log(f"  distress|uncertainty (h=9) district: {at('hr_sent_district', h_own)}")
        log(f"  distress|uncertainty (h=9) quarter : {at('hr_sent_quarter', h_own)}")
        log(f"  distress|uncertainty (h=9) two-way : {at('hr_sent_twoway', h_own)}")

    (OUT_DIR / "run_log_phase8.txt").write_text("\n".join(_LOG) + "\n", encoding="utf-8")
    log("summary written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
