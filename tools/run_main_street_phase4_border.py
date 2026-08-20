#!/usr/bin/env python
"""Main Street Uncertainty — Phase 4b: split-state border contrasts.

Fourteen states straddle two Fed districts, so within one state and
quarter, counties on either side of the district boundary carry
DIFFERENT district BBUI shocks while sharing the same state government,
state labor-market institutions, and state-level demand conditions. The
crosswalk noise that Phase 1 documented becomes the design:

    y_{c,t+h} - y_{c,t-1} = beta_h shock_{d(c),t}
                          + county FE + state-x-quarter FE
                          + 4 lags of (shock, outcome) + e

restricted to counties of the 14 split states. The state-x-quarter FE
absorb everything common to a state that quarter; beta_h is identified
purely off the within-state, cross-district contrast in district
uncertainty shocks. No exposure interaction: the district shock itself
is the regressor (it survives the FE because a state's counties sit in
two districts).

Two samples, reported side by side:
  * all counties of the 14 split states;
  * BORDER counties only — counties with at least one same-state
    Census-adjacent neighbor assigned to the other district (the
    sharpest version of the contrast; Census county-adjacency file).

Notes / honest limitations (also in DRAFT.md):
  * The leave-one-out national shock is NOT a possible control here:
    within a state-quarter, loo_A - loo_B = (own_B - own_A)/11 exactly,
    i.e. collinear with the boundary contrast of own shocks. The border
    design estimates the effect of own-district relative to
    neighbor-district uncertainty by construction.
  * County LAUS urate is NSA (derived U/(U+E) from LAUCN levels, the
    same construction as ``puremacro.fetch.state_industry_panel.
    iter_county_urate_q``, applied to ALL counties of the split states
    rather than that helper's top-counties subset). State-x-quarter FE
    absorb state-common seasonality; county-idiosyncratic seasonality
    remains in the error. Unweighted regressions (each county equal);
    small-county measurement noise attenuates.
  * Clustered at the district level (the level of the shock), G = 11
    districts present among split states (San Francisco has none);
    wild-cluster bootstrap-t (CGM 2008) with the MacKinnon-Webb
    few-cluster caveat stated, not solved. Driscoll-Kraay reported
    side by side.
  * Estimator adapted from ``run_main_street_phase3.estimate_horizon``
    (same demeaning, same DK bandwidth rule, same restricted
    wild-cluster scheme) with the group structure changed to
    county FE + state-x-quarter FE and an explicit district cluster
    vector, and the bootstrap chunked over draw blocks to bound
    memory on the ~10x larger county panel. A runtime self-check
    asserts chunked == unchunked on the first horizon.

Inputs:
    output/county_district_crosswalk.csv   (build_fed_county_crosswalk.py)
    output/state_district_panel_quarterly.csv   (district bbui_innov)
    FRED LAUCN{fips}0000000004/05 via the on-disk HTTP cache
    Census county_adjacency.txt via the on-disk HTTP cache

Outputs (docs/research/main_street_uncertainty/output/):
    county_border_panel.csv      county-quarter estimation panel
    irf_border_pairs.csv         LP paths, both samples + leads
    fig_phase4_border.(png|pdf)
    phase4b_manifest.json, phase4b_summary.json, run_log_phase4b.txt
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

from puremacro._http import safe_get_bytes_cached  # noqa: E402
from puremacro._linalg import inv_xtx  # noqa: E402
from puremacro.inference.dk import driscoll_kraay  # noqa: E402
import puremacro.sa.x13 as _sax13  # noqa: E402
from puremacro.sa.x13 import deseasonalize_x13, x13_available  # noqa: E402

import run_main_street_phase3 as p3  # noqa: E402  (_two_way_demean, _codes)

OUT_DIR = p3.OUT_DIR
XWALK_CSV = OUT_DIR / "county_district_crosswalk.csv"
MERGED_CSV = p3.MERGED_CSV

_FREDGRAPH = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
ADJACENCY_URL = ("https://www2.census.gov/geo/docs/reference/"
                 "county_adjacency.txt")

HORIZONS = p3.HORIZONS
LEADS = p3.LEADS
N_LAGS = p3.N_LAGS
ALPHA = p3.ALPHA
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
# Data assembly
# ---------------------------------------------------------------------------

def fetch_county_ue_m(fips: str) -> pd.DataFrame | None:
    """Monthly NSA county unemployment / employment levels (LAUCN)."""
    try:
        ru = safe_get_bytes_cached(
            _FREDGRAPH.format(f"LAUCN{fips}0000000004"))
        re_ = safe_get_bytes_cached(
            _FREDGRAPH.format(f"LAUCN{fips}0000000005"))
    except Exception:
        return None
    out = []
    for raw in (ru, re_):
        df = pd.read_csv(io.BytesIO(raw))
        s = pd.to_numeric(df[df.columns[1]], errors="coerce")
        s.index = pd.to_datetime(df[df.columns[0]])
        out.append(s.dropna())
    u, e = out
    common = u.index.intersection(e.index)
    if len(common) == 0:
        return None
    return pd.DataFrame({"date": common, "u": u.loc[common].to_numpy(),
                         "e": e.loc[common].to_numpy()})


def fetch_county_urate_q(fips: str) -> pd.Series | None:
    """Quarterly NSA county urate from LAUCN U/E levels (same derivation
    as state_industry_panel.iter_county_urate_q, quarter-start index)."""
    m = fetch_county_ue_m(fips)
    if m is None:
        return None
    urate = pd.Series((m["u"] / (m["u"] + m["e"]) * 100.0).to_numpy(),
                      index=pd.DatetimeIndex(m["date"]))
    return urate.resample("QS").mean().dropna()


def build_county_panel(xwalk: pd.DataFrame,
                       shock: pd.DataFrame) -> pd.DataFrame:
    parts = []
    n_missing = 0
    for _, r in xwalk.iterrows():
        q = fetch_county_urate_q(r["county_fips"])
        if q is None or len(q) < 40:
            n_missing += 1
            log(f"  county {r['county_fips']} {r['county_name']} "
                f"({r['state']}): no usable LAUS series — dropped")
            continue
        parts.append(pd.DataFrame({
            "county_fips": r["county_fips"], "state": r["state"],
            "district": r["district"], "date": q.index,
            "urate": q.to_numpy(),
        }))
    panel = pd.concat(parts, ignore_index=True)
    panel = panel.merge(shock, on=["district", "date"], how="left")
    log(f"county panel: {panel['county_fips'].nunique()} counties "
        f"({n_missing} dropped), {panel['date'].nunique()} quarters, "
        f"{len(panel)} rows")
    return panel


SA_MONTHLY_GZ = OUT_DIR / "county_ue_monthly_sa.csv.gz"


def build_sa_monthly(xwalk: pd.DataFrame, *, refresh: bool) -> tuple[pd.DataFrame, dict]:
    """Monthly county U/E X-13-adjusted panel (frozen; reused unless
    --refresh-sa). U and E levels are adjusted SEPARATELY per county
    (statsmodels auto transform, outlier detection on, no trading-day —
    the house `puremacro.sa.x13` defaults), then urate_sa = U/(U+E).
    STL fallbacks are counted, not silent: this tool refuses to run
    without the real binary, and reports how many series X-13 rejected."""
    diag: dict = {}
    if SA_MONTHLY_GZ.exists() and not refresh:
        long = pd.read_csv(SA_MONTHLY_GZ, dtype={"county_fips": str},
                           parse_dates=["date"])
        # The freeze stores only the SA levels (raw U/E are one cached
        # HTTP fetch away; the ratio is one line) — recompute here.
        long["urate_sa"] = (long["u_sa"]
                            / (long["u_sa"] + long["e_sa"]) * 100.0)
        log(f"SA monthly panel reused from {SA_MONTHLY_GZ.name} "
            f"({long['county_fips'].nunique()} counties; pass "
            "--refresh-sa to rebuild)")
        diag["reused_frozen"] = True
        return long, diag

    parts = []
    n_gap_months = 0
    for _, r in xwalk.iterrows():
        m = fetch_county_ue_m(r["county_fips"])
        if m is None or len(m) < 40:
            continue
        # X-13 needs a gap-free monthly calendar; LAUS county series have
        # occasional missing months (the value-level interpolation inside
        # puremacro.sa.x13 cannot repair INDEX gaps — statsmodels would
        # write a literal 'nan' row into the spc and X-13 errors out,
        # which is exactly the silent-fallback bug this gate exists for).
        m = m.set_index("date").sort_index()
        full = pd.date_range(m.index.min(), m.index.max(), freq="MS")
        n_gap_months += len(full) - len(m)
        m = m.reindex(full).interpolate(limit_direction="both")
        m.index.name = "date"
        m = m.reset_index()
        m["county_fips"] = r["county_fips"]
        parts.append(m)
    long = pd.concat(parts, ignore_index=True)

    # Count STL fallbacks inside deseasonalize_x13 (module-global hook).
    n_fb = {"n": 0}
    orig_stl = _sax13._stl_one

    def counting_stl(values, period):
        n_fb["n"] += 1
        return orig_stl(values, period=period)

    _sax13._stl_one = counting_stl
    try:
        t0 = time.time()
        long["u_sa"] = deseasonalize_x13(long, "u", by="county_fips",
                                         date_col="date", freq="M")
        log(f"X-13 on U levels: {time.time() - t0:.0f}s")
        t0 = time.time()
        long["e_sa"] = deseasonalize_x13(long, "e", by="county_fips",
                                         date_col="date", freq="M")
        log(f"X-13 on E levels: {time.time() - t0:.0f}s")
    finally:
        _sax13._stl_one = orig_stl

    bad = (long["u_sa"] < 0) | (long["e_sa"] <= 0)
    n_bad = int(bad.sum())
    long.loc[bad, ["u_sa", "e_sa"]] = np.nan
    long["urate_sa"] = long["u_sa"] / (long["u_sa"] + long["e_sa"]) * 100.0
    n_runs = 2 * long["county_fips"].nunique()
    diag.update({"n_stl_fallback_series": n_fb["n"],
                 "n_x13_runs": n_runs,
                 "n_gap_months_interpolated": int(n_gap_months),
                 "n_negative_sa_months_nulled": n_bad,
                 "reused_frozen": False})
    log(f"SA panel: {long['county_fips'].nunique()} counties, "
        f"{n_fb['n']} STL fallbacks (of {n_runs} X-13 runs), "
        f"{n_gap_months} gap months interpolated, "
        f"{n_bad} negative-SA months nulled")
    # Hard gate: '--sa' promises X-13ARIMA-SEATS. A high fallback share
    # means the outputs would really be STL-adjusted — refuse.
    if n_fb["n"] > 0.05 * n_runs:
        raise SystemExit(
            f"build_sa_monthly: {n_fb['n']}/{n_runs} series fell back to "
            "STL — the '_sa' outputs would misrepresent 'X-13 adjusted'. "
            "Diagnose puremacro.sa.x13 (binary reachable? series gaps?) "
            "and re-run with --refresh-sa.")
    # Freeze only the SA levels: raw U/E are refetchable from the HTTP
    # cache and urate_sa is recomputed on load — halves the artifact.
    long[["county_fips", "date", "u_sa", "e_sa"]].to_csv(
        SA_MONTHLY_GZ, index=False, float_format="%.6f")
    return long, diag


def sa_quarterly_panel(long: pd.DataFrame, xwalk: pd.DataFrame,
                       shock: pd.DataFrame) -> pd.DataFrame:
    q = long.dropna(subset=["urate_sa"]).copy()
    q["qdate"] = q["date"].dt.to_period("Q").dt.to_timestamp(how="start")
    agg = (q.groupby(["county_fips", "qdate"])["urate_sa"].mean()
           .rename("urate").reset_index()
           .rename(columns={"qdate": "date"}))
    meta = xwalk[["county_fips", "state", "district"]]
    panel = agg.merge(meta, on="county_fips", how="inner")
    panel = panel.merge(shock, on=["district", "date"], how="left")
    keep = panel.groupby("county_fips")["urate"].transform("count") >= 40
    panel = panel[keep]
    log(f"SA quarterly panel: {panel['county_fips'].nunique()} counties, "
        f"{len(panel)} rows")
    return panel[["county_fips", "state", "district", "date", "urate",
                  "shk"]]


def border_county_set(xwalk: pd.DataFrame) -> set[str]:
    """Counties with >=1 same-state Census-adjacent neighbor in the
    OTHER district. Census county_adjacency.txt (tab-delimited: name,
    fips, neighbor-name, neighbor-fips; county field blank on
    continuation rows)."""
    raw = safe_get_bytes_cached(ADJACENCY_URL)
    adj_pairs: list[tuple[str, str]] = []
    current = None
    for line in raw.decode("latin-1").splitlines():
        f = line.split("\t")
        if len(f) < 4:
            continue
        if f[1].strip('"').strip():
            current = f[1].strip('"').strip()
        nb = f[3].strip('"').strip()
        if current and nb and current != nb:
            adj_pairs.append((current, nb))
    dmap = dict(zip(xwalk["county_fips"], xwalk["district"]))
    smap = dict(zip(xwalk["county_fips"], xwalk["state"]))
    border = set()
    for a, b in adj_pairs:
        if (a in dmap and b in dmap and smap[a] == smap[b]
                and dmap[a] != dmap[b]):
            border.add(a)
            border.add(b)
    return border


# ---------------------------------------------------------------------------
# Estimator (adapted from run_main_street_phase3.estimate_horizon:
# county FE + state-x-quarter FE, explicit district clusters, chunked WC)
# ---------------------------------------------------------------------------

def _design_cty(panel: pd.DataFrame, h: int, n_lags: int,
                use_y_lags: bool = True) -> tuple[pd.DataFrame, list[str]]:
    df = panel.sort_values(["county_fips", "date"]).copy()
    g = df.groupby("county_fips", sort=False)
    df["dy"] = g["urate"].shift(-h) - g["urate"].shift(1)
    cols = ["shk"]
    for lag in range(1, n_lags + 1):
        df[f"shk_L{lag}"] = g["shk"].shift(lag)
        cols.append(f"shk_L{lag}")
        if use_y_lags:
            df[f"y_L{lag}"] = g["urate"].shift(lag)
            cols.append(f"y_L{lag}")
    keep = ["county_fips", "state", "district", "date", "dy"] + cols
    return df[keep].dropna().reset_index(drop=True), cols


def estimate_horizon_cty(sub: pd.DataFrame, x_cols: list[str], *,
                         wc_draws: np.ndarray | None,
                         chunk: int = 128) -> dict:
    g_cty = p3._codes(sub["county_fips"])
    st_key = sub["state"].astype("string") + "|" + sub["date"].astype("string")
    g_stq = p3._codes(st_key)
    g_dist = p3._codes(sub["district"])
    n_dist = int(g_dist.max()) + 1

    M = np.column_stack([sub["dy"].to_numpy(float)]
                        + [sub[c].to_numpy(float) for c in x_cols])
    W = p3._two_way_demean(M, g_cty, g_stq)
    y_w, X_w = W[:, 0], W[:, 1:]
    n, _ = X_w.shape
    XtX_inv = inv_xtx(X_w, name="run_main_street_phase4_border")
    beta = XtX_inv @ X_w.T @ y_w
    u = y_w - X_w @ beta

    score = X_w * u[:, None]
    L = max(1, int(round(4 * (n / 100) ** (2 / 9))))
    S = driscoll_kraay(score, sub["date"].to_numpy(), lags=L)
    vcov_dk = XtX_inv @ S @ XtX_inv
    se_dk = float(np.sqrt(np.diag(vcov_dk))[0])

    a = XtX_inv[:, 0]
    z = X_w @ a
    ghat = np.array([float(z[g_dist == d] @ u[g_dist == d])
                     for d in range(n_dist)])
    cfac = n_dist / (n_dist - 1)
    se_cr = math.sqrt(cfac * float(ghat @ ghat))
    t_cr = float(beta[0]) / se_cr if se_cr > 0 else np.nan

    p_wc = np.nan
    if wc_draws is not None and se_cr > 0:
        Xr = X_w[:, 1:]
        if Xr.shape[1] > 0:
            XtXr = inv_xtx(Xr, name="run_main_street_phase4_border.wc")
            br = XtXr @ Xr.T @ y_w
            fr, ur = Xr @ br, y_w - Xr @ br
        else:
            fr, ur = np.zeros_like(y_w), y_w
        n_exceed = 0
        n_ok = 0
        for j0 in range(0, wc_draws.shape[0], chunk):
            Vb = wc_draws[j0:j0 + chunk][:, g_dist].T      # (n, b)
            Ystar = fr[:, None] + p3._two_way_demean(ur[:, None] * Vb,
                                                     g_cty, g_stq)
            C = XtX_inv @ (X_w.T @ Ystar)
            Ustar = Ystar - X_w @ C
            var_b = np.zeros(Ustar.shape[1])
            for d in range(n_dist):
                m = g_dist == d
                var_b += (z[m] @ Ustar[m]) ** 2
            se_b = np.sqrt(cfac * var_b)
            ok = se_b > 0
            n_ok += int(ok.sum())
            n_exceed += int((np.abs(C[0, ok] / se_b[ok])
                             >= abs(t_cr)).sum())
        p_wc = float((1 + n_exceed) / (n_ok + 1))

    zc = norm.ppf(1 - ALPHA / 2)
    return {
        "beta": float(beta[0]), "se_dk": se_dk,
        "lo_dk": float(beta[0]) - zc * se_dk,
        "hi_dk": float(beta[0]) + zc * se_dk,
        "t_crve": t_cr, "p_wc": p_wc,
        "n_obs": int(n), "n_counties": int(sub["county_fips"].nunique()),
        "n_districts": int(n_dist), "n_states": int(sub["state"].nunique()),
    }


def run_spec_cty(panel: pd.DataFrame, *, spec: str, horizons: list[int],
                 wc_draws: np.ndarray | None, n_lags: int = N_LAGS,
                 use_y_lags: bool = True) -> pd.DataFrame:
    rows = []
    for h in horizons:
        sub, x_cols = _design_cty(panel, h, n_lags, use_y_lags=use_y_lags)
        res = estimate_horizon_cty(sub, x_cols, wc_draws=wc_draws)
        res.update({"spec": spec, "h": h})
        rows.append(res)
    out = pd.DataFrame(rows)
    hs = out.set_index("h")
    peak = int(hs["beta"].abs().idxmax())
    log(f"spec[{spec}]: peak h={peak} beta={hs.loc[peak, 'beta']:+.4f} "
        f"(DK se {hs.loc[peak, 'se_dk']:.4f}, "
        f"WC p={hs.loc[peak, 'p_wc']:.3f}) n={int(hs.loc[peak, 'n_obs'])}, "
        f"{int(hs.loc[peak, 'n_counties'])} counties / "
        f"{int(hs.loc[peak, 'n_states'])} states / "
        f"{int(hs.loc[peak, 'n_districts'])} districts")
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def fig_border(irf: pd.DataFrame, fig_path: Path,
               title_note: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("border_all", "All split-state counties"),
              ("border_only", "Border counties only")]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), sharey=True)
    for ax, (spec, title) in zip(axes, panels):
        r = irf[irf["spec"] == spec].sort_values("h")
        lead = irf[irf["spec"] == spec + "_leads"].sort_values("h")
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.axvline(-0.5, color="0.8", lw=0.8, ls=":")
        ax.fill_between(r["h"], r["lo_dk"], r["hi_dk"], color="C0",
                        alpha=0.22, label="90% pointwise (DK)")
        sig = r["p_wc"] < ALPHA
        ax.plot(r["h"], r["beta"], color="C0", lw=2.0)
        ax.plot(r.loc[~sig, "h"], r.loc[~sig, "beta"], "o", color="white",
                mec="C0", ms=5, label="wild-cluster p ≥ 0.10")
        ax.plot(r.loc[sig, "h"], r.loc[sig, "beta"], "o", color="C0",
                ms=5, label="wild-cluster p < 0.10")
        if len(lead):
            ax.errorbar(lead["h"], lead["beta"],
                        yerr=norm.ppf(1 - ALPHA / 2) * lead["se_dk"],
                        fmt="s", color="0.35", ms=4, lw=1.2, capsize=2,
                        label="leads")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("quarters after shock", fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("pp per 1 s.d. district shock", fontsize=9)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Split-state border contrast: county unemployment after "
                 "a district BBUI innovation (county FE + state-x-quarter "
                 f"FE){title_note}", fontsize=11)
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
    ap.add_argument("--wc-draws", type=int, default=999)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fast", action="store_true",
                    help="quick mode: --wc-draws 199")
    ap.add_argument("--sa", action="store_true",
                    help="X-13ARIMA-SEATS county U/E levels before the "
                         "urate ratio; outputs suffixed _sa. Requires the "
                         "real x13as/x13ashtml binary — hard error "
                         "otherwise (a silent STL fallback would "
                         "misrepresent 'X-13 adjusted').")
    ap.add_argument("--refresh-sa", action="store_true",
                    help="rebuild the frozen SA monthly panel")
    args = ap.parse_args(argv)
    B = 199 if args.fast else args.wc_draws
    sfx = "_sa" if args.sa else ""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    xwalk = pd.read_csv(XWALK_CSV, dtype={"county_fips": str})
    merged = pd.read_csv(MERGED_CSV, parse_dates=["date"])
    shock = (merged[["district", "date", "bbui_innov"]]
             .drop_duplicates(["district", "date"])
             .rename(columns={"bbui_innov": "shk"}))
    log(f"crosswalk: {len(xwalk)} counties in {xwalk['state'].nunique()} "
        f"split states, {xwalk['district'].nunique()} districts")

    sa_diag: dict = {}
    if args.sa:
        if not x13_available():
            raise SystemExit(
                "--sa requires the X-13ARIMA-SEATS binary. Install the "
                "x13org/x13prebuilt mac-arm64 x13ashtml under "
                "~/.local/x13ashtml/ with the ~/.local/bin/x13as "
                "naming-bridge wrapper (see the batch-5a plan), or set "
                "X13PATH.")
        sa_monthly, sa_diag = build_sa_monthly(xwalk,
                                               refresh=args.refresh_sa)
        panel = sa_quarterly_panel(sa_monthly, xwalk, shock)
    else:
        panel = build_county_panel(xwalk, shock)
    panel = panel[panel["date"] >= BASELINE_START]
    panel = panel.dropna(subset=["shk"])

    border = border_county_set(xwalk)
    panel["is_border"] = panel["county_fips"].isin(border)
    n_border = panel.loc[panel["is_border"], "county_fips"].nunique()
    log(f"border counties (same-state cross-district Census adjacency): "
        f"{n_border} of {panel['county_fips'].nunique()}")
    # gzipped: 147k county-quarter rows are ~9.5 MB raw, ~2 MB compressed;
    # pandas reads .csv.gz transparently. Estimation uses the in-memory
    # frame — this file is the reproducibility record.
    panel.to_csv(OUT_DIR / f"county_border_panel{sfx}.csv.gz", index=False,
                 float_format="%.6f")

    n_dist = panel["district"].nunique()
    wc_draws = rng.choice([-1.0, 1.0], size=(B, n_dist))

    # Self-check: chunked WC == unchunked on h=0, all-counties sample.
    sub0, xc0 = _design_cty(panel, 0, N_LAGS)
    r_full = estimate_horizon_cty(sub0, xc0, wc_draws=wc_draws[:64],
                                  chunk=64)
    r_chun = estimate_horizon_cty(sub0, xc0, wc_draws=wc_draws[:64],
                                  chunk=17)
    if not (r_full["p_wc"] == r_chun["p_wc"]
            and abs(r_full["beta"] - r_chun["beta"]) < 1e-14):
        raise RuntimeError("chunked wild-cluster self-check FAILED")
    log("chunked wild-cluster self-check passed")

    irf_frames = []
    irf_frames.append(run_spec_cty(panel, spec="border_all",
                                   horizons=HORIZONS, wc_draws=wc_draws))
    irf_frames.append(run_spec_cty(panel, spec="border_all_leads",
                                   horizons=LEADS, wc_draws=wc_draws,
                                   n_lags=0, use_y_lags=False))
    bpanel = panel[panel["is_border"]]
    irf_frames.append(run_spec_cty(bpanel, spec="border_only",
                                   horizons=HORIZONS, wc_draws=wc_draws))
    irf_frames.append(run_spec_cty(bpanel, spec="border_only_leads",
                                   horizons=LEADS, wc_draws=wc_draws,
                                   n_lags=0, use_y_lags=False))
    irf = pd.concat(irf_frames, ignore_index=True)
    irf.to_csv(OUT_DIR / f"irf_border_pairs{sfx}.csv", index=False)

    fig_border(irf, OUT_DIR / f"fig_phase4_border{sfx}.png",
               title_note=(" — X-13 seasonally adjusted" if args.sa else ""))
    log("figure written")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "county_district_crosswalk.csv": {
                "sha256": sha256_file(XWALK_CSV), "rows": int(len(xwalk))},
            "state_district_panel_quarterly.csv": {
                "sha256": sha256_file(MERGED_CSV)},
            "laus": "FRED LAUCN{fips}0000000004/05 via on-disk HTTP "
                    "cache; urate = U/(U+E)*100, monthly NSA -> "
                    "quarterly mean",
            "adjacency": ADJACENCY_URL,
        },
        "estimator": "county LP; county FE + state-x-quarter FE; "
                     f"{N_LAGS} lags of (district shock, outcome); "
                     "LHS y(t+h)-y(t-1); district-level wild cluster "
                     f"(B={B}, chunked) + Driscoll-Kraay",
        "wc_draws": B, "seed": args.seed,
        "estimation_start": BASELINE_START,
    }
    if args.sa:
        manifest["seasonal_adjustment"] = {
            "method": "X-13ARIMA-SEATS 1.1 build 57 "
                      "(x13org/x13prebuilt mac-arm64 x13ashtml; "
                      "statsmodels naming bridged via ~/.local/bin/x13as)",
            "applied_to": "monthly county U and E levels separately, "
                          "urate_sa = U_sa/(U_sa+E_sa); house wrapper "
                          "puremacro.sa.x13 (auto transform, outlier=True, "
                          "no trading-day)",
            "frozen_panel": SA_MONTHLY_GZ.name,
            **sa_diag,
        }
    (OUT_DIR / f"phase4b_manifest{sfx}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    def _path(spec: str) -> dict:
        s = irf[irf["spec"] == spec].set_index("h")
        return {int(h): {"beta": round(float(r["beta"]), 5),
                         "se_dk": round(float(r["se_dk"]), 5),
                         "p_wc": (None if pd.isna(r["p_wc"])
                                  else round(float(r["p_wc"]), 3))}
                for h, r in s.iterrows()}

    ia = irf[irf["spec"] == "border_all"].set_index("h")
    ib = irf[irf["spec"] == "border_only"].set_index("h")
    summary = {
        "mode": "fast" if args.fast else "full",
        "n_counties": int(panel["county_fips"].nunique()),
        "n_border_counties": int(n_border),
        "n_districts": int(n_dist),
        "peak_all": {"h": int(ia["beta"].abs().idxmax()),
                     "beta": float(ia.loc[ia["beta"].abs().idxmax(),
                                          "beta"]),
                     "p_wc": float(ia.loc[ia["beta"].abs().idxmax(),
                                          "p_wc"])},
        "peak_border": {"h": int(ib["beta"].abs().idxmax()),
                        "beta": float(ib.loc[ib["beta"].abs().idxmax(),
                                             "beta"]),
                        "p_wc": float(ib.loc[ib["beta"].abs().idxmax(),
                                             "p_wc"])},
        "paths": {spec: _path(spec) for spec in irf["spec"].unique()},
        "runtime_seconds": round(time.time() - _T0, 1),
    }
    summary["seasonally_adjusted"] = bool(args.sa)
    (OUT_DIR / f"phase4b_summary{sfx}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    (OUT_DIR / f"run_log_phase4b{sfx}.txt").write_text(
        "\n".join(_LOG_LINES) + "\n", encoding="utf-8")
    log("summary written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
