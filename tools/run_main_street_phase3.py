#!/usr/bin/env python
"""Main Street Uncertainty — Phase 3: exposure-differential identification.

Phase 2 established a descriptive fact: state unemployment rises ~0.04 pp
over 2-3 years after a one-s.d. district Beige Book labor-uncertainty
(BBUI) innovation, identified off relative cross-district variation
(two-way FE). Phase 3 sharpens the design to an *exposure differential*:

    y_{s,t+h} - y_{s,t-1} = beta_h (expo_s x shock_{d(s),t})
                            + state FE + district-x-quarter FE
                            + 4 lags of (interaction, outcome) + e

The district-x-quarter fixed effects absorb the district shock itself
(and any district-level demand/composition shock that quarter), so
beta_h is identified purely off *within-district, cross-state*
differences in pre-determined industry exposure — the shift-share logic
of Bartik (1991) in the "exposure-design" reading of Goldsmith-Pinkham,
Sorkin & Swift (2020, AER 110(8)) with the shock treated as given
(Borusyak, Hull & Jaravel 2022, REStud 89(1)).

Exposure (data-backed, key-free):
  * Manufacturing share: FRED CES state mirrors ``{ST}MFG`` (SA; DC only
    exists NSA as ``DCMFGN``) over total nonfarm ``{ST}NA`` (from the
    Phase-2 state labor CSV), averaged over the *base period 1990Q1-
    1991Q4* — the earliest quarters the CES state series exist — and
    frozen (never updated) per shift-share discipline. The baseline
    estimation sample therefore starts 1992Q1, strictly after the base
    period. z-scored across states (ddof=0).
  * Mining/energy share (robustness, the Phase-2 Dallas suspect):
    ``{ST}NRMN`` (mining & logging, NSA; seasonality washes out in the
    8-quarter base mean), same base period.

Outcomes:
  * State urate (pp) and 100 x log nonfarm employment (Phase-2 merged
    panel ``output/state_district_panel_quarterly.csv``).
  * Mass layoffs, administrative: BLS Mass Layoff Statistics state
    layoff-event counts (series ``MLUMS<SS>NN0001003``; program ran
    1995-04..2013-05, parquet ``../data/processed/warn_bls_mls.parquet``
    built by ``uncertainty_examples/tools/scrape_bls_mls.py`` from
    download.bls.gov flat files), quarterly sums, outcome log(1+events),
    51 states, 1998Q1-2013Q1.
  * Mass layoffs, WARN notices: ``puremacro.narrative.sources.us_warn``
    (state DOL scrapers + BigLocalNews + WARNTracker caches). Coverage
    is state-heterogeneous; the pipeline writes the full quarterly panel
    and coverage table, and runs the exposure LP only on the 2015Q1-
    2021Q4 window x states covered throughout — flagged FRAGILE in all
    outputs (few effective districts; see warn_coverage.csv).

Inference, side by side at every horizon:
  * Driscoll-Kraay (1998, REStat 80(4)) HAC SEs on the within-projected
    design, house bandwidth L = max(1, round(4 (N_panel/100)^(2/9)))
    exactly as ``puremacro.lp._panel_helpers._focal_dk_se``.
  * Wild-cluster bootstrap-t p-values, Rademacher weights at the level
    of the shock (the 12 Fed districts), null imposed (restricted
    residuals), per Cameron, Gelbach & Miller (2008, REStat 90(3)).
    Implemented in-script (the package's ``inference.wild_bootstrap``
    is a time-series residual bootstrap, not a cluster one). The FE
    projection is re-applied to every bootstrap outcome (exact FWL).
    With G=12 clusters Rademacher weights admit 2^12 sign patterns;
    we draw B=999 (seeded) and report p = (1+#{|t*|>=|t|})/(B+1) —
    the MacKinnon-Webb few-cluster caveat is stated, not solved.

Placebos / robustness:
  * Shuffle-district placebo: derangements of the 12 district labels
    (every state gets a *wrong* district's shock; the district-x-quarter
    partition is permutation-invariant, so only the regressor moves).
    Randomization p at the peak horizon.
  * Lead outcomes h in {-4,-3,-2}: LHS y_{t+h} - y_{t-1} (h=-1 is
    identically zero by construction and is skipped). No lag controls in
    the lead specs — the outcome lags ARE the lead LHS mechanically.
  * Drop-one-district jackknife of beta at the peak horizon.

Stated simplifications: single-industry exposure (manufacturing share),
not a full shift-share vector; base shares are 1990-91 (CES state
floor), i.e. *early-sample-fixed*, not pre-1983; current-vintage
outcomes; WARN zeros inside a state's covered span are assumed true
zeros; no identification claim beyond the exposure-differential design
itself (level effects common to all exposure levels are absorbed).

Outputs (docs/research/main_street_uncertainty/output/):
    exposure_state.csv          state exposure shares + z-scores
    warn_state_quarterly.csv    WARN notice counts, all covered states
    warn_coverage.csv           per-state WARN coverage + LP inclusion
    mls_state_quarterly.csv     BLS MLS layoff events, quarterly
    irf_exposure.csv            all exposure-LP paths + DK + WC p
    pretrends.csv               lead-outcome coefficients
    placebo_shuffle.csv         shuffle-district placebo draws
    jackknife_districts.csv     drop-one-district betas at peak h
    fig_exposure_irf.(png|pdf)  headline exposure IRFs
    fig_phase3_robustness.(png|pdf) pre-trends / placebo / jackknife
    phase3_manifest.json, phase3_summary.json, run_log_phase3.txt
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.stats import norm  # noqa: E402

from puremacro._http import safe_get_bytes_cached  # noqa: E402
from puremacro._linalg import inv_xtx  # noqa: E402
from puremacro.inference.dk import driscoll_kraay  # noqa: E402
from puremacro.narrative.indices._fed_districts import (  # noqa: E402
    DISTRICT_NAMES, STATE_TO_DISTRICT,
)
from puremacro.narrative.sources.us_warn import iter_us_warn  # noqa: E402

OUT_DIR = (REPO_ROOT / "docs" / "research" / "main_street_uncertainty"
           / "output")
MERGED_CSV = OUT_DIR / "state_district_panel_quarterly.csv"
BBUI_CSV = REPO_ROOT / "data" / "processed" / "bbui_district_panel.csv"
MLS_PARQUET = (REPO_ROOT.parent / "data" / "processed"
               / "warn_bls_mls.parquet")

_FREDGRAPH = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

HORIZONS = list(range(0, 13))
HORIZONS_SHORT = list(range(0, 9))   # MLS / WARN (shorter samples)
LEADS = [-4, -3, -2]                 # h=-1 is identically 0, skipped
N_LAGS = 4
ALPHA = 0.10
BASE_START, BASE_END = "1990-01-01", "1991-12-31"   # exposure base period
BASELINE_START = "1992-01-01"                        # strictly post-base
WARN_WIN = ("2015-01-01", "2021-12-31")

ALL_STATES = sorted(STATE_TO_DISTRICT)   # 50 states + DC

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

def fetch_fred_q(series_id: str) -> pd.Series:
    """Quarterly mean of a monthly FRED series via the key-free fredgraph
    CSV, through the on-disk HTTP cache (resumable across runs)."""
    raw = safe_get_bytes_cached(_FREDGRAPH.format(series_id))
    df = pd.read_csv(io.BytesIO(raw))
    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
    s = pd.to_numeric(df[df.columns[1]], errors="coerce")
    s.index = df[df.columns[0]]
    q = s.dropna().resample("QS").mean()
    q.name = series_id
    return q


def build_exposure(merged: pd.DataFrame) -> pd.DataFrame:
    """State manufacturing / mining employment shares, 1990Q1-1991Q4 mean.

    Denominator: total nonfarm employment (thousands) recovered from the
    Phase-2 panel's ``emp100`` (= 100 x log thousands).
    """
    emp = merged[["state", "date", "emp100"]].dropna().copy()
    emp["emp_thousands"] = np.exp(emp["emp100"] / 100.0)
    base = emp[(emp["date"] >= BASE_START) & (emp["date"] <= BASE_END)]

    rows = []
    for st in ALL_STATES:
        nf = base[base["state"] == st].set_index("date")["emp_thousands"]
        if nf.empty:
            raise RuntimeError(
                f"build_exposure: no base-period nonfarm employment for "
                f"{st} — Phase-2 panel incomplete?")
        rec: dict = {"state": st, "district": STATE_TO_DISTRICT[st]}
        for label, candidates in (
            ("mfg", (f"{st}MFG", f"{st}MFGN")),
            ("mining", (f"{st}NRMN", f"{st}NRMNN")),
        ):
            share = np.nan
            used = None
            for sid in candidates:
                try:
                    q = fetch_fred_q(sid)
                except Exception:
                    continue
                qb = q[(q.index >= BASE_START) & (q.index <= BASE_END)]
                common = qb.index.intersection(nf.index)
                if len(common) < 4:
                    continue
                share = float((qb.loc[common] / nf.loc[common]).mean())
                used = sid
                break
            rec[f"{label}_share_9091"] = share
            rec[f"{label}_series_id"] = used
        rows.append(rec)
    expo = pd.DataFrame(rows)

    for label in ("mfg", "mining"):
        col = f"{label}_share_9091"
        ok = expo[col].notna()
        vals = expo.loc[ok, col]
        if not ((vals > 0) & (vals < 0.5)).all():
            raise RuntimeError(
                f"build_exposure: implausible {label} shares outside "
                f"(0, 0.5): {expo.loc[ok & ~expo[col].between(0, 0.5), ['state', col]].to_dict('records')}")
        z = (vals - vals.mean()) / vals.std(ddof=0)
        expo.loc[ok, f"{label}_z"] = z
        log(f"exposure[{label}]: {int(ok.sum())}/51 states, share range "
            f"[{vals.min():.3f}, {vals.max():.3f}] "
            f"(missing: {sorted(expo.loc[~ok, 'state'])})")
    return expo


def build_warn_quarterly() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quarterly WARN-notice counts per state + coverage table.

    Zeros are filled only inside a state's own covered span [first
    filing, last filing] — outside it the count is missing, not zero.
    """
    panels, cov_rows = [], []
    for st in ALL_STATES:
        try:
            recs = list(iter_us_warn(state=st, since=None))
        except ValueError:
            cov_rows.append({"state": st, "n_filings": 0,
                             "first_q": None, "last_q": None,
                             "in_warn_lp": False})
            continue
        if not recs:
            cov_rows.append({"state": st, "n_filings": 0,
                             "first_q": None, "last_q": None,
                             "in_warn_lp": False})
            continue
        d = pd.DataFrame({
            "date": pd.to_datetime([r[0] for r in recs]),
            "workers": [r[4] if r[4] is not None else np.nan for r in recs],
        })
        d["q"] = d["date"].dt.to_period("Q")
        agg = d.groupby("q").agg(n_notices=("date", "size"),
                                 workers_affected=("workers", "sum"))
        full = pd.period_range(agg.index.min(), agg.index.max(), freq="Q")
        agg = agg.reindex(full, fill_value=0.0)
        agg["n_notices"] = agg["n_notices"].astype(int)
        out = agg.reset_index().rename(columns={"index": "q"})
        out["state"] = st
        out["date"] = out["q"].dt.to_timestamp(how="start")
        panels.append(out[["state", "date", "n_notices",
                           "workers_affected"]])
        first_q, last_q = str(full[0]), str(full[-1])
        in_lp = (full[0].to_timestamp() <= pd.Timestamp(WARN_WIN[0])
                 and full[-1].to_timestamp() >= pd.Timestamp("2021-10-01"))
        cov_rows.append({"state": st, "n_filings": len(d),
                         "first_q": first_q, "last_q": last_q,
                         "in_warn_lp": bool(in_lp)})
    warn = pd.concat(panels, ignore_index=True)
    cov = pd.DataFrame(cov_rows)
    n_in = int(cov["in_warn_lp"].sum())
    log(f"WARN panel: {warn['state'].nunique()} states with filings, "
        f"{len(warn)} state-quarters; {n_in} states covered through "
        f"{WARN_WIN[0][:4]}-{WARN_WIN[1][:4]} (LP subset)")
    return warn, cov


def build_mls_quarterly() -> pd.DataFrame:
    """BLS Mass Layoff Statistics events, monthly parquet -> quarterly sums.

    The BLS flat files omit state-months with no qualifying events (no
    zero rows exist; the smallest reported count is 3), so absent months
    inside the program window are filled with 0 — a documented
    assumption whose error is bounded at 1-2 events and concentrated in
    small states. Quarters not fully inside the window are dropped (the
    program ends mid-quarter in 2013-05)."""
    mls = pd.read_parquet(MLS_PARQUET)
    mls["date"] = pd.to_datetime(mls["date"])
    lo, hi = mls["date"].min(), mls["date"].max()
    full_m = pd.date_range(lo, hi, freq="MS")
    parts = []
    for st in ALL_STATES:
        s = (mls[mls["state"] == st].set_index("date")["layoff_events"]
             .reindex(full_m, fill_value=0.0))
        parts.append(pd.DataFrame({"state": st, "date": s.index,
                                   "layoff_events": s.to_numpy()}))
    m = pd.concat(parts, ignore_index=True)
    m["q"] = m["date"].dt.to_period("Q")
    g = m.groupby(["state", "q"]).agg(
        layoff_events=("layoff_events", "sum"),
        n_months=("layoff_events", "size")).reset_index()
    g = g[g["n_months"] == 3]
    g["date"] = g["q"].dt.to_timestamp(how="start")
    out = g[["state", "date", "layoff_events"]].reset_index(drop=True)
    log(f"MLS panel: {out['state'].nunique()} states x "
        f"{out['date'].nunique()} quarters "
        f"({out['date'].min().date()}..{out['date'].max().date()}; "
        f"absent months filled as 0)")
    return out


# ---------------------------------------------------------------------------
# Estimator: interacted panel LP, state FE + district-x-time FE
# ---------------------------------------------------------------------------

def _two_way_demean(A: np.ndarray, g1: np.ndarray, g2: np.ndarray,
                    tol: float = 1e-9, max_iter: int = 200) -> np.ndarray:
    """Alternating-projections demeaning by two grouping vectors.

    Works on an (n, B) matrix so bootstrap outcome draws are projected
    in one shot. Group codes must be 0..G-1 integers."""
    A = np.array(A, dtype=float, copy=True)
    if A.ndim == 1:
        A = A[:, None]
    n = A.shape[0]
    ones = np.ones(n)
    S1 = sparse.csr_matrix((ones, (g1, np.arange(n))))
    S2 = sparse.csr_matrix((ones, (g2, np.arange(n))))
    c1 = np.asarray(S1.sum(axis=1)).ravel()
    c2 = np.asarray(S2.sum(axis=1)).ravel()
    for _ in range(max_iter):
        m1 = (S1 @ A) / c1[:, None]
        A -= m1[g1]
        m2 = (S2 @ A) / c2[:, None]
        delta = m2[g2]
        A -= delta
        if np.max(np.abs(delta)) < tol:
            break
    return A


def _design(panel: pd.DataFrame, outcome: str, shock_col: str, h: int,
            n_lags: int, use_y_lags: bool = True
            ) -> tuple[pd.DataFrame, list[str]]:
    """Per-horizon LP design. ``inter`` (focal) = expo_z x shock.

    LHS is the long difference y_{t+h} - y_{t-1} (Jordà 2005 panel
    convention, matching Phase 2)."""
    df = panel.sort_values(["state", "date"]).copy()
    df["inter"] = df["expo_z"] * df[shock_col]
    g = df.groupby("state", sort=False)
    df["dy"] = g[outcome].shift(-h) - g[outcome].shift(1)
    cols = ["inter"]
    for lag in range(1, n_lags + 1):
        df[f"inter_L{lag}"] = g["inter"].shift(lag)
        cols.append(f"inter_L{lag}")
        if use_y_lags:
            df[f"y_L{lag}"] = g[outcome].shift(lag)
            cols.append(f"y_L{lag}")
    keep = ["state", "district", "date", "dy"] + cols
    return df[keep].dropna().reset_index(drop=True), cols


def _codes(values: pd.Series) -> np.ndarray:
    return pd.factorize(values, sort=True)[0]


def estimate_horizon(sub: pd.DataFrame, x_cols: list[str], *,
                     wc_draws: np.ndarray | None) -> dict:
    """One LP horizon: within (state, district-x-date) OLS + DK SE +
    wild-cluster bootstrap-t p over districts (CGM 2008)."""
    g_state = _codes(sub["state"])
    dt_key = sub["district"].astype("string") + "|" + sub["date"].astype("string")
    g_dt = _codes(dt_key)
    g_dist = _codes(sub["district"])
    n_dist = int(g_dist.max()) + 1

    M = np.column_stack([sub["dy"].to_numpy(float)]
                        + [sub[c].to_numpy(float) for c in x_cols])
    W = _two_way_demean(M, g_state, g_dt)
    y_w, X_w = W[:, 0], W[:, 1:]
    n, k = X_w.shape
    XtX_inv = inv_xtx(X_w, name="run_main_street_phase3.estimate_horizon")
    beta = XtX_inv @ X_w.T @ y_w
    u = y_w - X_w @ beta

    # Driscoll-Kraay, house bandwidth (matches _focal_dk_se).
    score = X_w * u[:, None]
    L = max(1, int(round(4 * (n / 100) ** (2 / 9))))
    time_keys = sub["date"].to_numpy()
    S = driscoll_kraay(score, time_keys, lags=L)
    vcov_dk = XtX_inv @ S @ XtX_inv
    se_dk = float(np.sqrt(np.diag(vcov_dk))[0])

    # CRVE by district (small-G factor G/(G-1)) — the WC test statistic.
    a = XtX_inv[:, 0]
    z = X_w @ a                              # (n,) focal influence vector
    ghat = np.array([float(z[g_dist == d] @ u[g_dist == d])
                     for d in range(n_dist)])
    cfac = n_dist / (n_dist - 1)
    se_cr = math.sqrt(cfac * float(ghat @ ghat))
    t_cr = float(beta[0]) / se_cr if se_cr > 0 else np.nan

    p_wc = np.nan
    if wc_draws is not None and se_cr > 0:
        # Null-imposed restricted fit: drop the focal column.
        Xr = X_w[:, 1:]
        if Xr.shape[1] > 0:
            XtXr = inv_xtx(Xr, name="run_main_street_phase3.wc_restricted")
            br = XtXr @ Xr.T @ y_w
            fr, ur = Xr @ br, y_w - Xr @ br
        else:
            fr, ur = np.zeros_like(y_w), y_w
        V = wc_draws[:, g_dist].T                 # (n, B) cluster signs
        Ystar = fr[:, None] + _two_way_demean(ur[:, None] * V,
                                              g_state, g_dt)
        C = XtX_inv @ (X_w.T @ Ystar)             # (k, B) coefficients
        Ustar = Ystar - X_w @ C
        var_b = np.zeros(Ustar.shape[1])
        for d in range(n_dist):
            m = g_dist == d
            var_b += (z[m] @ Ustar[m]) ** 2
        se_b = np.sqrt(cfac * var_b)
        ok = se_b > 0
        tstar = np.abs(C[0, ok] / se_b[ok])
        p_wc = float((1 + int((tstar >= abs(t_cr)).sum())) / (ok.sum() + 1))

    zc = norm.ppf(1 - ALPHA / 2)
    return {
        "beta": float(beta[0]), "se_dk": se_dk,
        "lo_dk": float(beta[0]) - zc * se_dk,
        "hi_dk": float(beta[0]) + zc * se_dk,
        "t_crve": t_cr, "p_wc": p_wc,
        "n_obs": int(n), "n_states": int(sub["state"].nunique()),
        "n_districts": int(n_dist),
    }


def run_spec(panel: pd.DataFrame, *, spec: str, outcome: str,
             shock_col: str, horizons: list[int], wc_draws: np.ndarray,
             use_y_lags: bool = True, wc: bool = True) -> pd.DataFrame:
    rows = []
    for h in horizons:
        sub, x_cols = _design(panel, outcome, shock_col, h, N_LAGS,
                              use_y_lags=use_y_lags)
        res = estimate_horizon(sub, x_cols,
                               wc_draws=wc_draws if wc else None)
        res.update({"spec": spec, "h": h})
        rows.append(res)
    out = pd.DataFrame(rows)
    hs = out.set_index("h")
    peak = int(hs["beta"].abs().idxmax())
    log(f"spec[{spec}]: peak h={peak} beta={hs.loc[peak, 'beta']:+.4f} "
        f"(DK se {hs.loc[peak, 'se_dk']:.4f}, WC p={hs.loc[peak, 'p_wc']:.3f}) "
        f"n={int(hs.loc[peak, 'n_obs'])}, "
        f"{int(hs.loc[peak, 'n_states'])} states / "
        f"{int(hs.loc[peak, 'n_districts'])} districts")
    return out


# ---------------------------------------------------------------------------
# Placebos
# ---------------------------------------------------------------------------

def derangement(rng: np.random.Generator, n: int) -> np.ndarray:
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def shuffle_district_placebo(panel: pd.DataFrame, *, outcome: str,
                             h_star: int, n_perm: int,
                             seed: int) -> pd.DataFrame:
    """Reassign every state the shock of a *wrong* district (derangement
    of district labels). The district-x-quarter partition is invariant
    under relabeling, so the FE structure is identical — only the
    interaction regressor changes. Effects should die."""
    rng = np.random.default_rng(seed)
    districts = sorted(panel["district"].unique())
    shock_wide = (panel[["district", "date", "bbui_innov"]]
                  .drop_duplicates(["district", "date"])
                  .pivot(index="date", columns="district",
                         values="bbui_innov"))
    rows = []
    for r in range(n_perm):
        p = derangement(rng, len(districts))
        mapping = {districts[i]: districts[p[i]]
                   for i in range(len(districts))}
        pl = panel.copy()
        placebo_shock = shock_wide.stack().rename("placebo_innov")
        pl["placebo_district"] = pl["district"].map(mapping)
        pl = pl.merge(placebo_shock.reset_index()
                      .rename(columns={"district": "placebo_district"}),
                      on=["date", "placebo_district"], how="left")
        sub, x_cols = _design(
            pl.assign(bbui_innov_pl=pl["placebo_innov"]),
            outcome, "bbui_innov_pl", h_star, N_LAGS)
        res = estimate_horizon(sub, x_cols, wc_draws=None)
        rows.append({"draw": r, "beta": res["beta"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_exposure(irf: pd.DataFrame, panels: list[tuple[str, str, str]],
                 fig_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.8))
    if len(panels) == 1:
        axes = [axes]
    for ax, (spec, title, unit) in zip(axes, panels):
        r = irf[irf["spec"] == spec].sort_values("h")
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.fill_between(r["h"], r["lo_dk"], r["hi_dk"], color="C0",
                        alpha=0.22, label="90% pointwise (Driscoll-Kraay)")
        sig = r["p_wc"] < ALPHA
        ax.plot(r["h"], r["beta"], color="C0", lw=2.0)
        ax.plot(r.loc[~sig, "h"], r.loc[~sig, "beta"], "o", color="white",
                mec="C0", ms=5, label="wild-cluster p ≥ 0.10")
        ax.plot(r.loc[sig, "h"], r.loc[sig, "beta"], "o", color="C0",
                ms=5, label="wild-cluster p < 0.10")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("quarters after shock", fontsize=9)
        ax.set_ylabel(unit, fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Exposure-differential LP: manufacturing share x district "
                 "BBUI innovation (state FE + district-x-quarter FE)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for suffix in (".png", ".pdf"):
        fig.savefig(fig_path.with_suffix(suffix),
                    dpi=200 if suffix == ".png" else None)
    plt.close(fig)


def fig_robustness(irf_u: pd.DataFrame, pre: pd.DataFrame,
                   plc: pd.DataFrame, jk: pd.DataFrame, h_star: int,
                   beta_star: float, fig_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))

    ax = axes[0]
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.axvline(-0.5, color="0.8", lw=0.8, ls=":")
    pu = pre[pre["spec"] == "urate_mfg_leads"].sort_values("h")
    ax.errorbar(pu["h"], pu["beta"], yerr=norm.ppf(1 - ALPHA / 2) * pu["se_dk"],
                fmt="s", color="0.35", ms=4, lw=1.2, capsize=2,
                label="leads (no lag controls)")
    r = irf_u.sort_values("h")
    ax.fill_between(r["h"], r["lo_dk"], r["hi_dk"], color="C0", alpha=0.22)
    ax.plot(r["h"], r["beta"], color="C0", lw=1.8, marker="o", ms=3,
            label="baseline path")
    ax.set_title("Pre-trends: lead outcomes", fontsize=10)
    ax.set_xlabel("quarters relative to shock", fontsize=9)
    ax.set_ylabel("pp per s.d. x s.d.", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[1]
    ax.hist(plc["beta"], bins=25, color="0.75", edgecolor="white")
    ax.axvline(beta_star, color="C0", lw=2.0,
               label=f"true β(h={h_star})")
    ax.set_title("Shuffle-district placebo", fontsize=10)
    ax.set_xlabel(f"placebo β at h={h_star}", fontsize=9)
    ax.set_ylabel("draws", fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[2]
    jj = jk.sort_values("beta").reset_index(drop=True)
    ax.axvline(beta_star, color="C0", lw=1.4, ls="--")
    ax.axvline(0.0, color="0.6", lw=0.8)
    ax.plot(jj["beta"], np.arange(len(jj)), "o", color="C0", ms=5)
    ax.set_yticks(np.arange(len(jj)))
    ax.set_yticklabels("drop " + jj["dropped_district"], fontsize=7)
    ax.set_title(f"Drop-one-district jackknife (h={h_star})", fontsize=10)
    ax.set_xlabel("β", fontsize=9)

    for ax in axes:
        ax.tick_params(labelsize=8)
    fig.suptitle("Robustness of the manufacturing-exposure unemployment "
                 "response", fontsize=11)
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
                    help="wild-cluster Rademacher draws (default 999)")
    ap.add_argument("--placebo", type=int, default=200,
                    help="shuffle-district permutations (default 200)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fast", action="store_true",
                    help="quick mode: --wc-draws 199 --placebo 40")
    args = ap.parse_args(argv)
    B = 199 if args.fast else args.wc_draws
    n_perm = 40 if args.fast else args.placebo

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    merged = pd.read_csv(MERGED_CSV, parse_dates=["date"])
    assert merged["district"].nunique() == 12, "expected 12 districts"
    log(f"Phase-2 merged panel: {merged['state'].nunique()} states x "
        f"{merged['date'].nunique()} quarters, {len(merged)} rows")

    # Cross-district shock comovement — needed to read the shuffle
    # placebo (correlated shocks mean a wrong-district interaction
    # retains part of the signal by construction).
    innov_wide = (merged[["district", "date", "bbui_innov"]]
                  .drop_duplicates(["district", "date"])
                  .pivot(index="date", columns="district",
                         values="bbui_innov"))
    cmat = innov_wide.corr()
    shock_xcorr = float(cmat.values[np.triu_indices_from(cmat, k=1)].mean())
    log(f"mean pairwise cross-district shock correlation: "
        f"{shock_xcorr:+.3f}")

    # --- exposure ---------------------------------------------------------
    expo = build_exposure(merged)
    expo.to_csv(OUT_DIR / "exposure_state.csv", index=False)

    # --- WARN + MLS panels ------------------------------------------------
    warn, warn_cov = build_warn_quarterly()
    warn.to_csv(OUT_DIR / "warn_state_quarterly.csv", index=False)
    warn_cov.to_csv(OUT_DIR / "warn_coverage.csv", index=False)
    mls = build_mls_quarterly()
    mls.to_csv(OUT_DIR / "mls_state_quarterly.csv", index=False)

    # --- panels for each spec --------------------------------------------
    base = merged.merge(expo[["state", "mfg_z", "mining_z"]], on="state")
    base_mfg = (base.rename(columns={"mfg_z": "expo_z"})
                [["state", "district", "date", "urate", "emp100",
                  "bbui_innov", "expo_z"]]
                .dropna(subset=["expo_z"]))
    base_mfg = base_mfg[base_mfg["date"] >= BASELINE_START]

    base_min = (base.rename(columns={"mining_z": "expo_z"})
                [["state", "district", "date", "urate",
                  "bbui_innov", "expo_z"]]
                .dropna(subset=["expo_z"]))
    base_min = base_min[base_min["date"] >= BASELINE_START]

    mls_p = (mls.merge(base_mfg[["state", "date", "bbui_innov", "expo_z",
                                 "district"]],
                       on=["state", "date"], how="inner"))
    mls_p["log1p_events"] = np.log1p(mls_p["layoff_events"])

    warn_states = sorted(warn_cov.loc[warn_cov["in_warn_lp"], "state"])
    warn_p = warn[warn["state"].isin(warn_states)].copy()
    warn_p = warn_p[(warn_p["date"] >= WARN_WIN[0])
                    & (warn_p["date"] <= WARN_WIN[1])]
    warn_p["log1p_notices"] = np.log1p(warn_p["n_notices"])
    warn_p = warn_p.merge(
        base_mfg[["state", "date", "bbui_innov", "expo_z", "district"]],
        on=["state", "date"], how="inner")
    warn_dists = warn_p.groupby("district")["state"].nunique()
    log(f"WARN LP subset: {len(warn_states)} states, "
        f"{int((warn_dists >= 2).sum())}/12 districts with >=2 states "
        f"(FRAGILE — see warn_coverage.csv)")

    # --- wild-cluster draws (shared across specs, seeded once) ------------
    wc_draws = rng.choice([-1.0, 1.0], size=(B, 12))

    # --- exposure LPs ------------------------------------------------------
    irf_frames = []
    irf_frames.append(run_spec(base_mfg, spec="urate_mfg", outcome="urate",
                               shock_col="bbui_innov", horizons=HORIZONS,
                               wc_draws=wc_draws))
    irf_frames.append(run_spec(base_mfg.dropna(subset=["emp100"]),
                               spec="emp_mfg", outcome="emp100",
                               shock_col="bbui_innov", horizons=HORIZONS,
                               wc_draws=wc_draws))
    irf_frames.append(run_spec(base_min, spec="urate_mining",
                               outcome="urate", shock_col="bbui_innov",
                               horizons=HORIZONS, wc_draws=wc_draws))
    irf_frames.append(run_spec(mls_p, spec="mls_mfg",
                               outcome="log1p_events",
                               shock_col="bbui_innov",
                               horizons=HORIZONS_SHORT, wc_draws=wc_draws))
    if len(warn_states) >= 10:
        irf_frames.append(run_spec(warn_p, spec="warn_mfg_FRAGILE",
                                   outcome="log1p_notices",
                                   shock_col="bbui_innov",
                                   horizons=HORIZONS_SHORT,
                                   wc_draws=wc_draws))
    else:
        log(f"WARN LP skipped: only {len(warn_states)} fully covered "
            f"states (<10)")
    irf = pd.concat(irf_frames, ignore_index=True)
    irf.to_csv(OUT_DIR / "irf_exposure.csv", index=False)

    # --- pre-trends (leads) ------------------------------------------------
    pre_frames = []
    for spec, pnl, outc in (("urate_mfg_leads", base_mfg, "urate"),
                            ("mls_mfg_leads", mls_p, "log1p_events")):
        rows = []
        for h in LEADS:
            sub, x_cols = _design(pnl, outc, "bbui_innov", h, 0,
                                  use_y_lags=False)
            res = estimate_horizon(sub, x_cols, wc_draws=wc_draws)
            res.update({"spec": spec, "h": h})
            rows.append(res)
        pre_frames.append(pd.DataFrame(rows))
        b = pre_frames[-1]
        log(f"leads[{spec}]: " + ", ".join(
            f"h={int(r.h)}: {r.beta:+.4f} (p_wc {r.p_wc:.2f})"
            for r in b.itertuples()))
    pre = pd.concat(pre_frames, ignore_index=True)
    pre.to_csv(OUT_DIR / "pretrends.csv", index=False)

    # --- peak horizon, placebo, jackknife ----------------------------------
    iu = irf[irf["spec"] == "urate_mfg"].set_index("h")
    h_star = int(iu["beta"].abs().idxmax())
    beta_star = float(iu.loc[h_star, "beta"])

    plc = shuffle_district_placebo(base_mfg, outcome="urate",
                                   h_star=h_star, n_perm=n_perm,
                                   seed=args.seed + 1)
    plc.to_csv(OUT_DIR / "placebo_shuffle.csv", index=False)
    p_ri = float((1 + int((plc["beta"].abs() >= abs(beta_star)).sum()))
                 / (len(plc) + 1))
    log(f"placebo: {len(plc)} derangements, mean {plc['beta'].mean():+.4f}, "
        f"sd {plc['beta'].std(ddof=1):.4f}, randomization p={p_ri:.3f}")

    jk_rows = []
    for d in DISTRICT_NAMES:
        pnl = base_mfg[base_mfg["district"] != d]
        sub, x_cols = _design(pnl, "urate", "bbui_innov", h_star, N_LAGS)
        res = estimate_horizon(sub, x_cols, wc_draws=None)
        jk_rows.append({"dropped_district": d, "beta": res["beta"],
                        "se_dk": res["se_dk"]})
    jk = pd.DataFrame(jk_rows)
    jk.to_csv(OUT_DIR / "jackknife_districts.csv", index=False)
    log(f"jackknife: beta(h={h_star}) range "
        f"[{jk['beta'].min():+.4f}, {jk['beta'].max():+.4f}]")

    # --- figures ------------------------------------------------------------
    fig_exposure(irf, [
        ("urate_mfg", "Unemployment rate (1992Q1+)",
         "pp per (1 s.d. exposure) x (1 s.d. shock)"),
        ("emp_mfg", "Nonfarm employment (1992Q1+)",
         "% per (1 s.d. exposure) x (1 s.d. shock)"),
        ("mls_mfg", "Mass-layoff events, BLS MLS (1998-2013)",
         "log(1+events) per (1 s.d. x 1 s.d.)"),
    ], OUT_DIR / "fig_exposure_irf.png")
    fig_robustness(irf[irf["spec"] == "urate_mfg"], pre, plc, jk,
                   h_star, beta_star, OUT_DIR / "fig_phase3_robustness.png")
    log("figures written")

    # --- manifest + summary -------------------------------------------------
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "state_district_panel_quarterly.csv": {
                "sha256": sha256_file(MERGED_CSV), "rows": int(len(merged))},
            "bbui_district_panel.csv": {"sha256": sha256_file(BBUI_CSV)},
            "warn_bls_mls.parquet": {
                "sha256": sha256_file(MLS_PARQUET),
                "source": "BLS MLS flat files (MLUMS<SS>NN0001003), "
                          "built by uncertainty_examples/tools/"
                          "scrape_bls_mls.py"},
            "exposure_series": "FRED fredgraph CSV {ST}MFG(/MFGN), "
                               "{ST}NRMN; base period 1990Q1-1991Q4",
            "warn_connector": "puremacro.narrative.sources.us_warn."
                              "iter_us_warn (per-state caches)",
        },
        "estimator": "interacted panel LP; state FE + district-x-quarter "
                     f"FE; {N_LAGS} lags of interaction and outcome; "
                     "LHS y(t+h)-y(t-1)",
        "inference": {
            "dk": "Driscoll-Kraay 1998, house bandwidth "
                  "4*(N/100)^(2/9) on panel rows",
            "wild_cluster": f"CGM 2008 restricted Rademacher over 12 "
                            f"districts, B={B}, seed={args.seed}",
        },
        "wc_draws": B, "placebo_perms": n_perm, "seed": args.seed,
    }
    (OUT_DIR / "phase3_manifest.json").write_text(
        json.dumps(manifest, indent=2))

    def _spec_block(name: str) -> dict:
        s = irf[irf["spec"] == name]
        if s.empty:
            return {}
        si = s.set_index("h")
        pk = int(si["beta"].abs().idxmax())
        return {
            "peak_h": pk, "peak_beta": float(si.loc[pk, "beta"]),
            "peak_se_dk": float(si.loc[pk, "se_dk"]),
            "peak_p_wc": float(si.loc[pk, "p_wc"]),
            "h_with_wc_p_lt_10": [int(h) for h, p in si["p_wc"].items()
                                  if p < 0.10],
            "n_obs_peak": int(si.loc[pk, "n_obs"]),
            "n_states": int(si.loc[pk, "n_states"]),
            "n_districts": int(si.loc[pk, "n_districts"]),
            "beta_by_h": {int(h): round(float(b), 5)
                          for h, b in si["beta"].items()},
        }

    summary = {
        "mode": "fast" if args.fast else "full",
        "specs": {name: _spec_block(name)
                  for name in irf["spec"].unique()},
        "h_star": h_star, "beta_star": beta_star,
        "placebo": {"n": int(len(plc)),
                    "mean": float(plc["beta"].mean()),
                    "sd": float(plc["beta"].std(ddof=1)),
                    "randomization_p": p_ri,
                    "mean_cross_district_shock_corr": shock_xcorr},
        "jackknife": {"min": float(jk["beta"].min()),
                      "max": float(jk["beta"].max()),
                      "min_district": jk.loc[jk["beta"].idxmin(),
                                             "dropped_district"],
                      "all_same_sign": bool((np.sign(jk["beta"])
                                             == np.sign(beta_star)).all())},
        "pretrends": {spec: {int(r["h"]): {"beta": round(float(r["beta"]), 5),
                                           "p_wc": round(float(r["p_wc"]), 3)}
                             for _, r in pre[pre["spec"] == spec].iterrows()}
                      for spec in pre["spec"].unique()},
        "warn_lp_states": warn_states,
        "warn_districts_with_2plus_states": int((warn_dists >= 2).sum()),
        "exposure": {
            "mfg_states": int(expo["mfg_z"].notna().sum()),
            "mining_states": int(expo["mining_z"].notna().sum()),
            "mfg_share_min": float(expo["mfg_share_9091"].min()),
            "mfg_share_max": float(expo["mfg_share_9091"].max()),
            "mining_missing": sorted(
                expo.loc[expo["mining_z"].isna(), "state"]),
        },
        "runtime_seconds": round(time.time() - _T0, 1),
    }
    (OUT_DIR / "phase3_summary.json").write_text(
        json.dumps(summary, indent=2))
    (OUT_DIR / "run_log_phase3.txt").write_text("\n".join(_LOG_LINES) + "\n")
    log("summary written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
