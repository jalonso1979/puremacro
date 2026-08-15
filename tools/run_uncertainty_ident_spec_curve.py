#!/usr/bin/env python
"""Specification-curve pipeline: how much of the uncertainty-shock literature
is identification choice?

Runs the full SVAR identification menu shipped in ``puremacro.var.identify``
(plus the lag-augmented LP benchmark) on ONE fixed monthly US dataset, then
sweeps proxy choice, sample window and detrending through
``puremacro.inference.spec_curve.run_spec_curve``.

Dataset conventions (documented choices)
----------------------------------------
* Frequency: monthly. Variables (n = 4), in this order:
    0. ``u``    — uncertainty proxy, z-scored on its full available history.
    1. ``act``  — economic activity: 100·log Industrial Production (INDPRO).
    2. ``emp``  — 100·log nonfarm payroll employment (PAYEMS).
    3. ``ffr``  — effective federal funds rate (FEDFUNDS), percent level.
  This is a reduced 4-variable version of the Bloom (2009, Econometrica)
  8-variable VAR; Baker-Bloom-Davis (2016, QJE, Fig. VII) use the same core
  (EPU, log S&P 500, FFR, log employment, log IP). Simplifications: no stock
  index (FRED's no-key CSV endpoint only serves ~10 years of SP500), no
  wages/CPI/hours block.
* Uncertainty proxies:
    - EPU: Baker-Bloom-Davis US news-based EPU, modern series (1985+,
      ``media_All_Country_Data.xlsx``) ratio-spliced back to 1900 with the
      Historical News-Based EPU (``media_US_Historical_EPU_data.xlsx``).
    - WUI: Ahir-Bloom-Furceri World Uncertainty Index, USA column of sheet T2
      (quarterly, forward-filled to months). NOTE: deviates from
      ``puremacro.shock_atlas`` which uses the world F1 average.
    - JLN: Jurado-Ludvigson-Ng (2015, AER) macro uncertainty, h=1, from the
      202602 zip in ``data/raw/www.sydneyludvigson.com``.
    - VIX: CBOE VIX (FRED VIXCLS), monthly mean of daily closes, 1990+.
* Macro series come from the public no-key FRED fredgraph CSV endpoint via
  ``puremacro.fetch._classic.fetch_fred`` (network authorized; results are
  frozen into ``panel_monthly.csv`` so reruns are offline + deterministic).
* VAR: p = 6 lags, constant only, horizon H = 24 months, ci = 0.90.
* Detrending sweep: ``lt`` = within-sample linear detrend of 100·log levels
  (activity + employment); ``fd`` = 100·Δlog growth rates whose IRFs are
  cumulated back to levels (pointwise band cumulation — an approximation).
  ``u`` and ``ffr`` always enter in levels. No HP filter (not in puremacro).
* Normalisation: every scheme's uncertainty-shock IRF is rescaled so the
  impact response of ``u`` equals +1.0 (one full-sample standard deviation of
  the z-scored proxy). ``la_lp``'s coefficient is per unit of ``u`` and is
  already on that scale. Statistical schemes (Rigobon, MagMav, ICA) do not
  label shocks; the "uncertainty shock" is the column with the largest
  |impact loading| on ``u`` — a documented labelling rule.
* Proxy-SVAR instrument: AR(6) innovations of a *different* uncertainty proxy
  (EPU↔JLN partner map), zero-filled outside availability following the
  censored-proxy convention of Stock-Watson (2018, Economic Journal) /
  Mertens-Ravn (2013, AER). Exogeneity of a second uncertainty measure is a
  strong assumption (Ludvigson-Ma-Ng 2021 critique) — flagged in the paper.
* Rigobon (2003) regimes: high-volatility = Great Recession + COVID months
  per ``puremacro.regime_dates.REGIMES_US``; all other months low-volatility.

Identification menu (all on the SAME reduced-form VAR per dataset)
------------------------------------------------------------------
  chol_first  Cholesky, uncertainty ordered first  (Bloom 2009 tradition)
  chol_last   Cholesky, uncertainty ordered last   (slow-moving-macro-first)
  sign        Rubio-Ramirez-Waggoner-Zha rotations; u up, activity down, h=0..2
  narrative   Antolin-Diaz & Rubio-Ramirez (2018) narrative sign restrictions:
              the SAME traditional pattern as `sign`, PLUS "the uncertainty
              shock was positive" in famous dated months — 1987-10 (Black
              Monday), 2001-09 (September 11), 2008-09 (Lehman), 2020-03
              (COVID onset). Events outside a cell's estimation window are
              dropped (recorded in the diagnostics). Weighted-percentile
              bands with AD-RR importance weights; ESS reported. In the
              spec grid the scheme runs only under the baseline detrending
              (lt) — the fd variants are skipped and logged (Phase 2 scope).
  proxy       Mertens-Ravn external instrument (partner-proxy innovations)
  maxshare    Faust-Uhlig max-share: shock maximising u's FEV over 12 months
  rigobon     Rigobon (2003) heteroskedasticity, regime split as above
  magmav      Magnusson-Mavroeidis (2014) endogenous variance breaks
  nongauss    Lanne-Meitz-Saikkonen (2017) FastICA non-Gaussian SVAR
  la_lp       Plagborg-Moller-Wolf (2021) lag-augmented LP (non-VAR benchmark)

GK identified-set overlay (baseline dataset only)
-------------------------------------------------
For the two rotation-based schemes (sign, narrative) the pipeline also
computes Giacomini-Kitagawa (2021) robust bands — the min/max of the IP
response over ALL admissible rotations, i.e. the identified set — via
``var.identify.gk_robust_bands`` with the same traditional sign pattern.
The contrast (posterior percentiles of a Haar prior vs the whole set) is
the robust-Bayes critique made visual. Conventions: reduced-form
uncertainty is OFF (n_var_draws=1) so both objects condition on the same
OLS reduced form, and the comparison is on the RAW structural scale (one-
standard-deviation structural shock, NOT the unit-effect normalisation
used everywhere else) because the set bounds are per-rotation extremes
that cannot be rescaled by a per-draw impact loading after aggregation.
gk_robust_bands imposes only the traditional pattern: the narrative-
truncated identified set (weakly smaller) is not computable from the
public API and is discussed, not plotted.

Outputs (written to --out)
--------------------------
  panel_monthly.csv / panel_manifest.json  — frozen dataset + sources
  headline_menu.csv                        — menu on the baseline dataset
  spec_curve_results.csv                   — the full curve (figure data)
  fig_spec_curve.pdf / .png                — sorted estimates by family
  fig_gk_overlay.pdf / .png                — GK identified set vs percentile bands
  gk_overlay.csv                           — the overlay figure's data
  table_family_medians.csv / .md           — medians per identification family
  summary.json / run_log.txt               — headline numbers + log

References
----------
Bloom, N. (2009). The impact of uncertainty shocks. Econometrica 77(3).
Baker, S., Bloom, N., Davis, S. (2016). Measuring economic policy
    uncertainty. QJE 131(4).
Antolín-Díaz, J., Rubio-Ramírez, J. F. (2018). Narrative sign restrictions
    for SVARs. American Economic Review 108(10), 2802-2829.
Giacomini, R., Kitagawa, T. (2021). Robust Bayesian inference for
    set-identified models. Econometrica 89(4), 1519-1556.
Jurado, K., Ludvigson, S., Ng, S. (2015). Measuring uncertainty. AER 105(3).
Ludvigson, S., Ma, S., Ng, S. (2021). Uncertainty and business cycles:
    exogenous impulse or endogenous response? AEJ: Macro 13(4).
Simonsohn, U., Simmons, J., Nelson, L. (2020). Specification curve analysis.
    Nature Human Behaviour 4, 1208-1214.
Plus the scheme-specific references in each puremacro.var.identify module.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import warnings
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from puremacro.fetch._classic import fetch_fred  # noqa: E402
from puremacro.inference.spec_curve import (  # noqa: E402
    bootstrap_pvalue_median,
    enumerate_specs,
    run_spec_curve,
)
from puremacro.lp.la_lp import la_lp  # noqa: E402
from puremacro.regime_dates import assign_regime  # noqa: E402
from puremacro.var.estimate import estimate_var  # noqa: E402
from puremacro.var.identify import (  # noqa: E402
    NarrativeRestriction,
    cholesky as cholesky_svar,
    gk_robust_bands,
    hetero as rigobon_svar,
    identify_maxshare,
    magmav_svar,
    narrative_sign_svar,
    non_gaussian_svar,
    proxy as proxy_svar,
    sign_restrictions as sign_restriction_svar,
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

P_LAGS = 6
HORIZON = 24
CI = 0.90
Z90 = 1.6448536269514722          # two-sided z for the 90% scheme bands
BASELINE = {"proxy": "epu", "sample": "full", "detrend": "lt"}
IMPACT_GUARD = 1e-3               # |impact on u| below this → normalisation refused

# Traditional sign pattern shared by the sign / narrative / GK machinery:
# uncertainty up, activity down, at horizons 0..2 (emp and ffr unrestricted).
SIGN_PATTERN = {h: [1, -1, 0, 0] for h in range(3)}

# Famous dated uncertainty events (month starts, matching the MS panel index).
# Each becomes an AD-RR Type-I restriction: structural shock 0 (the sign-
# identified uncertainty shock) was POSITIVE in that month.
NARRATIVE_EVENTS = {
    "1987-10-01": "Black Monday",
    "2001-09-01": "September 11 attacks",
    "2008-09-01": "Lehman Brothers failure",
    "2020-03-01": "COVID-19 onset",
}

# Phase 3 (opt-in via --narrative-hd): historical-decomposition dominance
# restrictions (AD-RR Types II/III) at the two events where "uncertainty
# was the dominant shock that month" is defensible from the literature.
# Restricting the HD of the UNCERTAINTY variable itself (index 0): the
# event month's move in the proxy must be driven mostly (Type II) /
# overwhelmingly (Type III) by the uncertainty shock. With any Type II/III
# restriction the AD-RR importance weights become non-constant, so the
# Kish ESS is finally an informative diagnostic (with pure Type I it just
# equals the surviving-draw count).
HD_EVENTS = ("2008-09-01", "2020-03-01")   # Lehman, COVID
COVID_EVENT = "2020-03-01"
HD_SCHEMES = {"narrative_t2": "narrative-sign",
              "narrative_t3": "narrative-sign"}

FAMILY_OF = {
    "chol_first": "recursive",
    "chol_last": "recursive",
    "sign": "sign",
    "narrative": "narrative-sign",
    "proxy": "proxy",
    "maxshare": "max-share",
    "rigobon": "heteroskedasticity",
    "magmav": "heteroskedasticity",
    "nongauss": "non-Gaussian",
    "la_lp": "local-projection",
}
# narrative-sign sits LAST, not next to sign: the 8-slot palette below is the
# dataviz reference instance in its documented order (red appended last),
# which passes the adjacent-pair validator (worst CVD ΔE 9.1, normal-vision
# 19.6); inserting red beside green/magenta fails the normal-vision floor.
FAMILY_ORDER = ["recursive", "sign", "proxy", "max-share",
                "heteroskedasticity", "non-Gaussian", "local-projection",
                "narrative-sign"]
# Validated categorical palette (dataviz reference instance, light mode; the
# three low-contrast slots get marker-edge + legend + CSV table as relief).
FAMILY_COLOR = {
    "recursive": "#2a78d6",
    "sign": "#008300",
    "proxy": "#e87ba4",
    "max-share": "#eda100",
    "heteroskedasticity": "#1baf7a",
    "non-Gaussian": "#eb6834",
    "local-projection": "#4a3aa7",
    "narrative-sign": "#e34948",
}
FAMILY_MARKER = {
    "recursive": "o",
    "sign": "s",
    "proxy": "^",
    "max-share": "D",
    "heteroskedasticity": "v",
    "non-Gaussian": "P",
    "local-projection": "X",
    "narrative-sign": "*",
}
# The star glyph has less visual mass than the other markers at equal area.
FAMILY_MARKER_SIZE = {"narrative-sign": 46}
INSTRUMENT_PARTNER = {"epu": "jln", "wui": "epu", "jln": "epu", "vix": "epu"}

_LOG_LINES: list[str] = []
_T0 = time.time()


def log(msg: str) -> None:
    line = f"[{time.time() - _T0:7.1f}s] {msg}"
    print(line, flush=True)
    _LOG_LINES.append(line)


def _seed_for(tag: str) -> int:
    """Deterministic per-spec seed (crc32 — python's hash() is randomised)."""
    return zlib.crc32(tag.encode()) % (2 ** 31)


def _z(s: pd.Series) -> pd.Series:
    s = s.dropna().astype(float)
    return (s - s.mean()) / s.std(ddof=0)


# --------------------------------------------------------------------------- #
# Panel assembly (sources: data/raw bundles + FRED fredgraph CSV)
# --------------------------------------------------------------------------- #

def load_epu_spliced(root: Path) -> tuple[pd.Series, dict]:
    """US news-based EPU: modern (1985+) spliced back to 1900 with the
    Historical index, scaled by the ratio of means over the 1985-2014 overlap
    (Baker-Bloom-Davis 2016 distribute the two segments separately)."""
    mod = pd.read_excel(root / "data/raw/www.policyuncertainty.com/media_All_Country_Data.xlsx")
    mod = mod[pd.to_numeric(mod["Year"], errors="coerce").notna()].copy()
    mod["date"] = pd.to_datetime(
        mod["Year"].astype(int).astype(str) + "-"
        + mod["Month"].astype(int).astype(str).str.zfill(2) + "-01")
    modern = mod.set_index("date")["US"].astype(float).dropna()

    hist = pd.read_excel(root / "data/raw/www.policyuncertainty.com/media_US_Historical_EPU_data.xlsx")
    hist = hist.dropna(subset=["Month"]).copy()
    hist["date"] = pd.to_datetime(
        hist["Year"].astype(int).astype(str) + "-"
        + hist["Month"].astype(int).astype(str).str.zfill(2) + "-01")
    hcol = "News-Based Historical Economic Policy Uncertainty"
    historical = hist.set_index("date")[hcol].astype(float).dropna()

    overlap = modern.index.intersection(historical.index)
    if len(overlap) < 24:
        raise ValueError(
            "load_epu_spliced: overlap between historical and modern US EPU "
            f"is only {len(overlap)} months; cannot ratio-splice.")
    ratio = modern.loc[overlap].mean() / historical.loc[overlap].mean()
    pre = historical.loc[historical.index < modern.index.min()] * ratio
    spliced = pd.concat([pre, modern]).sort_index()
    meta = {
        "source": ["data/raw/www.policyuncertainty.com/media_All_Country_Data.xlsx (US col)",
                   "data/raw/www.policyuncertainty.com/media_US_Historical_EPU_data.xlsx"],
        "transform": f"ratio-splice (factor {ratio:.4f} over {len(overlap)}-month overlap), then z-score",
        "span": f"{spliced.index.min().date()} -> {spliced.index.max().date()}",
    }
    return _z(spliced), meta


def load_wui_us(root: Path) -> tuple[pd.Series, dict]:
    """World Uncertainty Index, USA column, sheet T2 (quarterly -> monthly ffill)."""
    df = pd.read_excel(
        root / "data/raw/worlduncertaintyindex.com/wp-content_uploads_2024_10_WUI_Data.xlsx",
        sheet_name="T2")
    date_col = df.columns[0]
    raw = df[date_col].astype(str).str.lower().str.replace(" ", "")
    df = df.loc[raw.str.match(r"^\d{4}q[1-4]$", na=False)].copy()
    df["qdate"] = pd.PeriodIndex(df[date_col].astype(str).str.replace(" ", ""),
                                 freq="Q").to_timestamp(how="start")
    q = df.set_index("qdate")["USA"].astype(float).dropna()
    m = q.resample("MS").ffill()
    meta = {
        "source": "data/raw/worlduncertaintyindex.com/wp-content_uploads_2024_10_WUI_Data.xlsx sheet T2 col USA",
        "transform": "quarterly -> monthly forward-fill, z-score",
        "span": f"{m.index.min().date()} -> {m.index.max().date()}",
    }
    return _z(m), meta


def load_jln_macro(root: Path) -> tuple[pd.Series, dict]:
    """Jurado-Ludvigson-Ng macro uncertainty, h=1, from the local zip."""
    zpath = root / "data/raw/www.sydneyludvigson.com/s_MacroFinanceUncertainty_202602Update-3klg.zip"
    with zipfile.ZipFile(zpath) as zf:
        raw = pd.read_excel(io.BytesIO(zf.read("MacroUncertaintyToCirculate.xlsx")))
    s = raw.set_index(pd.to_datetime(raw["Date"]))["h=1"].astype(float).dropna()
    meta = {
        "source": str(zpath.relative_to(root)) + " :: MacroUncertaintyToCirculate.xlsx h=1",
        "transform": "z-score",
        "span": f"{s.index.min().date()} -> {s.index.max().date()}",
    }
    return _z(s), meta


def load_vix_monthly() -> tuple[pd.Series, dict]:
    """CBOE VIX (FRED VIXCLS): monthly mean of daily closes."""
    daily = fetch_fred("VIXCLS")
    m = daily.dropna().resample("MS").mean()
    meta = {
        "source": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS",
        "transform": "monthly mean of daily closes, z-score",
        "span": f"{m.index.min().date()} -> {m.index.max().date()}",
    }
    return _z(m), meta


def assemble_panel(root: Path) -> tuple[pd.DataFrame, dict]:
    """Assemble the frozen monthly panel. Network only touches FRED fredgraph."""
    cols: dict[str, pd.Series] = {}
    manifest: dict[str, dict] = {}

    loaders = {
        "epu": lambda: load_epu_spliced(root),
        "wui": lambda: load_wui_us(root),
        "jln": lambda: load_jln_macro(root),
        "vix": load_vix_monthly,
    }
    for name, fn in loaders.items():
        try:
            s, meta = fn()
            cols[name] = s
            manifest[name] = meta
            log(f"  proxy {name:4s}: n={len(s):4d}  {meta['span']}")
        except Exception as e:  # noqa: BLE001 — a missing proxy narrows the grid, doesn't kill the run
            log(f"  proxy {name:4s}: FAILED ({type(e).__name__}: {e}) — dropped from grid")

    fred_ids = {"ip": ("INDPRO", "100*log"), "emp": ("PAYEMS", "100*log"),
                "ffr": ("FEDFUNDS", "level")}
    for name, (sid, tr) in fred_ids.items():
        s = fetch_fred(sid).dropna().astype(float)
        s.index = pd.to_datetime(s.index)
        if tr == "100*log":
            s = 100.0 * np.log(s)
        cols[name] = s
        manifest[name] = {
            "source": f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}",
            "transform": tr,
            "span": f"{s.index.min().date()} -> {s.index.max().date()}",
        }
        log(f"  macro {name:4s}: {sid} n={len(s):4d}  {manifest[name]['span']}")

    panel = pd.DataFrame(cols)
    panel.index.name = "date"
    return panel, manifest


def freeze_panel(panel: pd.DataFrame, manifest: dict, out: Path) -> None:
    csv = out / "panel_monthly.csv"
    panel.to_csv(csv)  # default float repr round-trips float64 exactly
    sha = hashlib.sha256(csv.read_bytes()).hexdigest()
    meta = {
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "sha256_panel_monthly_csv": sha,
        "n_rows": int(len(panel)),
        "columns": {k: v for k, v in manifest.items()},
        "conventions": {
            "frequency": "monthly (MS)",
            "proxies_z_scored": "each on its full available history at assembly time",
            "activity": "100*log(INDPRO)", "employment": "100*log(PAYEMS)",
            "ffr": "FEDFUNDS level (percent)",
        },
    }
    (out / "panel_manifest.json").write_text(json.dumps(meta, indent=2))
    log(f"froze panel: {csv.name} ({len(panel)} rows, sha256 {sha[:12]}…)")


def load_frozen_panel(out: Path) -> pd.DataFrame:
    df = pd.read_csv(out / "panel_monthly.csv", parse_dates=["date"], index_col="date")
    return df


# --------------------------------------------------------------------------- #
# Dataset construction per spec cell
# --------------------------------------------------------------------------- #

def _linear_detrend(x: np.ndarray) -> np.ndarray:
    t = np.arange(len(x), dtype=float)
    X = np.column_stack([np.ones_like(t), t])
    beta = np.linalg.lstsq(X, x, rcond=None)[0]
    return x - X @ beta


def build_dataset(panel: pd.DataFrame, proxy: str, sample: str, detrend: str) -> dict | None:
    """Return dict(Y, dates, df_lp, key, ...) or None if infeasible."""
    if proxy not in panel.columns:
        return None
    need = [proxy, "ip", "emp", "ffr"]
    df = panel[need].copy()
    if detrend == "fd":
        df["ip"] = df["ip"].diff()
        df["emp"] = df["emp"].diff()
    df = df.dropna()
    if sample == "post1985":
        df = df.loc[df.index >= "1985-01-01"]
    elif sample == "pre2020":
        df = df.loc[df.index <= "2019-12-01"]
    elif sample != "full":
        raise ValueError(f"build_dataset: unknown sample {sample!r}")
    if len(df) < 120:  # ten years minimum for a 6-lag monthly VAR
        return None
    # Contiguity check — a gap would silently corrupt every lag structure.
    expected = pd.date_range(df.index.min(), df.index.max(), freq="MS")
    if len(expected) != len(df) or not (expected == df.index).all():
        raise ValueError(
            f"build_dataset: monthly index for proxy={proxy} sample={sample} "
            "has internal gaps; refusing to build a lagged dataset.")

    u = df[proxy].to_numpy()
    if detrend == "lt":
        act = _linear_detrend(df["ip"].to_numpy())
        emp = _linear_detrend(df["emp"].to_numpy())
    elif detrend == "fd":
        act = df["ip"].to_numpy()
        emp = df["emp"].to_numpy()
    else:
        raise ValueError(f"build_dataset: unknown detrend {detrend!r}")
    Y = np.column_stack([u, act, emp, df["ffr"].to_numpy()])

    # LP frame always uses raw 100*log activity levels: la_lp's LHS
    # y(t+h) - y(t-1) is a level change, so 'lt'/'fd' would be identical.
    lp = panel[[proxy, "ip", "emp", "ffr"]].dropna()
    lp = lp.loc[(lp.index >= df.index.min() - pd.DateOffset(months=P_LAGS))
                & (lp.index <= df.index.max())]
    df_lp = lp.rename(columns={proxy: "u"})

    key = (proxy, str(df.index.min().date()), str(df.index.max().date()), detrend)
    return {"Y": Y, "dates": df.index, "df_lp": df_lp, "proxy": proxy,
            "sample": sample, "detrend": detrend, "key": key,
            "n_obs": len(df)}


def regime_indicator(dates: pd.DatetimeIndex) -> np.ndarray:
    """1 = high-volatility months (GR + COVID per REGIMES_US), else 0."""
    lab = assign_regime(dates, scope="us")
    return lab.isin(["GR", "COVID"]).to_numpy().astype(int)


def ar_innovations(x: np.ndarray, p: int = 6) -> np.ndarray:
    """OLS AR(p) innovations, zero-padded for the first p obs."""
    T = len(x)
    if T <= p + 10:
        raise ValueError("ar_innovations: series too short for AR(6) instrument.")
    X = np.column_stack([np.ones(T - p)] + [x[p - l - 1:T - l - 1] for l in range(p)])
    beta = np.linalg.lstsq(X, x[p:], rcond=None)[0]
    out = np.zeros(T)
    out[p:] = x[p:] - X @ beta
    return out


def narrative_events_in(ds: dict) -> list[str]:
    """Narrative event months usable in this dataset cell: inside the
    estimation window and past the first p observations (no VAR residual
    exists before that). Events outside are dropped, not errors — e.g. the
    VIX sample starts in 1990, after Black Monday."""
    d0, d1 = ds["dates"][P_LAGS], ds["dates"][-1]
    return [ev for ev in NARRATIVE_EVENTS if d0 <= pd.Timestamp(ev) <= d1]


def build_instrument(panel: pd.DataFrame, ds: dict) -> np.ndarray | None:
    """Partner-proxy AR(6) innovations aligned to the dataset rows,
    zero-filled where the partner is unavailable (censored-proxy convention,
    Stock-Watson 2018)."""
    partner = INSTRUMENT_PARTNER[ds["proxy"]]
    if partner not in panel.columns:
        return None
    s = panel[partner].reindex(ds["dates"])
    avail = s.notna().to_numpy()
    if avail.sum() < 120:
        return None
    z = np.zeros(len(s))
    z[avail] = ar_innovations(s.to_numpy()[avail])
    return z


# --------------------------------------------------------------------------- #
# Shock normalisation + scheme runners
# --------------------------------------------------------------------------- #

def _norm(resp_u0: float, scheme: str) -> float:
    """Return scale k with k * resp_u0 = +1. Refuses degenerate loadings."""
    if not np.isfinite(resp_u0) or abs(resp_u0) < IMPACT_GUARD:
        raise ValueError(
            f"run_scheme[{scheme}]: impact of the labelled uncertainty shock on "
            f"u is {resp_u0!r} (< {IMPACT_GUARD}); unit-effect normalisation "
            "would explode — spec marked failed.")
    return 1.0 / resp_u0


def _pick_u_col(B: np.ndarray) -> int:
    """Labelling rule for statistical identifications: the 'uncertainty
    shock' is the column with the largest |impact loading| on u (row 0)."""
    return int(np.argmax(np.abs(B[0, :])))


def _finish(resp, lo, hi, k, detrend):
    """Scale by k (sign-aware band swap) and cumulate fd growth responses."""
    resp = np.asarray(resp, dtype=float) * k
    if lo is not None:
        lo = np.asarray(lo, dtype=float) * k
        hi = np.asarray(hi, dtype=float) * k
        if k < 0:
            lo, hi = hi, lo
    if detrend == "fd":
        resp = np.cumsum(resp)
        if lo is not None:  # pointwise cumulation — approximation, see module doc
            lo, hi = np.cumsum(lo), np.cumsum(hi)
    return resp, lo, hi


def _resim_bootstrap(Y, fit_fn, n_boot, seed, regime=None):
    """Generic residual-resimulation bootstrap returning (lo, hi, n_fail) of
    whatever normalised path ``fit_fn(Yb)`` returns. Resampling is within-
    regime when ``regime`` is given (preserves the heteroskedasticity that
    Rigobon-type identification feeds on)."""
    rng = np.random.default_rng(seed)
    A_list, c, _, resid, _ = estimate_var(Y, P_LAGS)
    T_eff = resid.shape[0]
    paths, n_fail = [], 0
    for _ in range(n_boot):
        if regime is None:
            idx = rng.integers(0, T_eff, size=T_eff)
        else:
            idx = np.arange(T_eff)
            for g in (0, 1):
                pool = np.flatnonzero(regime == g)
                if len(pool):
                    idx[pool] = rng.choice(pool, size=len(pool), replace=True)
        eps = resid[idx]
        Yb = np.zeros_like(Y)
        Yb[:P_LAGS] = Y[:P_LAGS]
        for t in range(P_LAGS, Y.shape[0]):
            yt = c.copy()
            for l in range(P_LAGS):
                yt += A_list[l] @ Yb[t - l - 1]
            Yb[t] = yt + eps[t - P_LAGS]
        try:
            paths.append(fit_fn(Yb))
        except (np.linalg.LinAlgError, ValueError, RuntimeError):
            n_fail += 1
    if not paths:
        return None, None, n_fail
    arr = np.stack(paths)
    lo_q, hi_q = 100 * (1 - CI) / 2, 100 * (1 + CI) / 2
    return (np.percentile(arr, lo_q, axis=0),
            np.percentile(arr, hi_q, axis=0), n_fail)


def run_scheme(scheme: str, ds: dict, panel: pd.DataFrame, draws: dict,
               seed: int) -> dict:
    """Run one identification scheme on one dataset. Returns dict with the
    normalised activity response path (level effect), bands, diagnostics."""
    Y, detrend = ds["Y"], ds["detrend"]
    H = HORIZON
    diag: dict = {}

    if scheme in ("chol_first", "chol_last"):
        ordering = None if scheme == "chol_first" else [1, 2, 3, 0]
        r = cholesky_svar(Y, p=P_LAGS, horizon=H, ordering=ordering,
                          n_boot=draws["var"], ci=CI, seed=seed)
        k = _norm(r.irf_point[0, 0, 0], scheme)
        resp, lo, hi = _finish(r.irf_point[:, 1, 0], r.irf_lower[:, 1, 0],
                               r.irf_upper[:, 1, 0], k, detrend)
        diag["n_fail"] = r.n_fail

    elif scheme == "sign":
        r = sign_restriction_svar(Y, p=P_LAGS, horizon=H,
                                  restrictions=SIGN_PATTERN,
                                  n_draws=draws["sign"], ci=CI, seed=seed)
        k = _norm(r.irf_median[0, 0, 0], scheme)
        resp, lo, hi = _finish(r.irf_median[:, 1, 0], r.irf_lower[:, 1, 0],
                               r.irf_upper[:, 1, 0], k, detrend)
        diag["n_accepted"] = r.n_accepted

    elif scheme == "narrative":
        events = narrative_events_in(ds)
        if not events:
            raise ValueError(
                "run_scheme[narrative]: no narrative event month falls inside "
                f"this cell's window ({ds['key'][1]} -> {ds['key'][2]}); the "
                "scheme would collapse to plain sign restrictions.")
        r = narrative_sign_svar(
            Y, p=P_LAGS, horizon=H, sign_matrix=SIGN_PATTERN,
            restrictions=[(ev, 0, +1) for ev in events], dates=ds["dates"],
            n_draws=draws["narr"], ci=CI, seed=seed)
        k = _norm(r.irf_median[0, 0, 0], scheme)
        resp, lo, hi = _finish(r.irf_median[:, 1, 0], r.irf_lower[:, 1, 0],
                               r.irf_upper[:, 1, 0], k, detrend)
        diag["n_trad"] = r.n_traditional_accepted
        diag["n_narr"] = r.n_narrative_accepted
        diag["ess"] = round(float(r.ess), 1)
        diag["events"] = "+".join(ev[:7] for ev in events)

    elif scheme in ("narrative_t2", "narrative_t3"):
        events = narrative_events_in(ds)
        if not events:
            raise ValueError(
                f"run_scheme[{scheme}]: no narrative event month falls "
                f"inside this cell's window.")
        hd_events = [ev for ev in HD_EVENTS if ev in events]
        if not hd_events:
            raise ValueError(
                f"run_scheme[{scheme}]: neither HD-restricted event "
                f"(Lehman, COVID) falls inside this cell's window — the "
                "scheme would collapse to plain Type-I narrative.")
        if scheme == "narrative_t3" and COVID_EVENT not in hd_events:
            raise ValueError(
                "run_scheme[narrative_t3]: Type III restricts the COVID "
                "month, which is outside this cell's window — cell would "
                "duplicate narrative_t2.")
        restrictions: list = [
            NarrativeRestriction(kind="shock_sign", date=ev, shock=0,
                                 sign=+1) for ev in events]
        for ev in hd_events:
            dom = ("overwhelming" if (scheme == "narrative_t3"
                                      and ev == COVID_EVENT) else "most")
            restrictions.append(NarrativeRestriction(
                kind="hd_dominance", date=ev, shock=0, variable=0,
                window=0, dominance=dom))
        r = narrative_sign_svar(
            Y, p=P_LAGS, horizon=H, sign_matrix=SIGN_PATTERN,
            restrictions=restrictions, dates=ds["dates"],
            n_draws=draws["narr"], ci=CI, seed=seed)
        k = _norm(r.irf_median[0, 0, 0], scheme)
        resp, lo, hi = _finish(r.irf_median[:, 1, 0], r.irf_lower[:, 1, 0],
                               r.irf_upper[:, 1, 0], k, detrend)
        diag["n_trad"] = r.n_traditional_accepted
        diag["n_narr"] = r.n_narrative_accepted
        diag["ess"] = round(float(r.ess), 1)
        diag["events"] = "+".join(ev[:7] for ev in events)
        diag["hd_events"] = "+".join(
            ev[:7] + ("(III)" if (scheme == "narrative_t3"
                                  and ev == COVID_EVENT) else "(II)")
            for ev in hd_events)

    elif scheme == "proxy":
        z = build_instrument(panel, ds)
        if z is None:
            raise ValueError(
                f"run_scheme[proxy]: no instrument partner available for "
                f"proxy={ds['proxy']} on this sample.")
        r = proxy_svar(Y, p=P_LAGS, horizon=H, instrument_series=z,
                       shock_target_idx=0, n_boot=draws["var"], ci=CI, seed=seed)
        k = _norm(r.irf_point[0, 0, 0], scheme)
        resp, lo, hi = _finish(r.irf_point[:, 1, 0], r.irf_lower[:, 1, 0],
                               r.irf_upper[:, 1, 0], k, detrend)
        diag["first_stage_F"] = round(float(r.first_stage_F), 2)

    elif scheme == "maxshare":
        nb = draws["var"]
        r = identify_maxshare(Y, p=P_LAGS, target_idx=0, max_fev_at=12,
                              horizon=H, n_bootstrap=nb, ci=CI, seed=seed)
        k = _norm(r.irfs[0, 0, 0], scheme)
        lo = r.irf_lower[:, 1, 0] if r.irf_lower is not None else None
        hi = r.irf_upper[:, 1, 0] if r.irf_upper is not None else None
        resp, lo, hi = _finish(r.irfs[:, 1, 0], lo, hi, k, detrend)
        diag["fev_share"] = round(float(r.fev_share_at_target), 3)

    elif scheme == "rigobon":
        ri = regime_indicator(ds["dates"])
        r = rigobon_svar(Y, p=P_LAGS, horizon=H, regime_indicator=ri, n_boot=0)
        j = _pick_u_col(r.B)
        k = _norm(r.irfs[0, 0, j], scheme)
        resp, lo, hi = _finish(r.irfs[:, 1, j], None, None, k, detrend)
        diag["u_col"] = j
        diag["var_ratio"] = round(float(r.variance_ratios[j]), 2)
        if draws["custom"] > 0:
            ri_eff = ri[P_LAGS:]

            def _fit(Yb):
                rb = rigobon_svar(Yb, p=P_LAGS, horizon=H,
                                  regime_indicator=ri, n_boot=0)
                jb = _pick_u_col(rb.B)
                kb = _norm(rb.irfs[0, 0, jb], "rigobon-boot")
                pb, _, _ = _finish(rb.irfs[:, 1, jb], None, None, kb, detrend)
                return pb

            lo, hi, nf = _resim_bootstrap(Y, _fit, draws["custom"], seed + 1,
                                          regime=ri_eff)
            diag["n_fail"] = nf

    elif scheme == "magmav":
        # magmav's convergence gate (Frobenius loss < 1e-3) is scale-
        # dependent; B is scale-equivariant, so standardising columns and
        # rescaling the responses afterwards is exact, not an approximation.
        sd = Y.std(axis=0, ddof=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = magmav_svar(Y / sd, p=P_LAGS, horizon=H, k_breaks=None,
                            n_boot=draws["magmav"], ci=CI, seed=seed)
        if r.eu != (1, 1):
            raise ValueError(
                f"run_scheme[magmav]: identification failed (eu={r.eu}, "
                f"k_breaks={r.k_breaks}) — Cholesky fallback rejected as a "
                "distinct scheme; spec marked failed.")
        j = _pick_u_col(r.B)  # row-0 rescale is a common factor; argmax unchanged
        k = _norm(r.irf_point[0, 0, j] * sd[0], scheme)
        lo = hi = None
        if draws["magmav"] > 0 and np.isfinite(r.irf_lower).all():
            lo, hi = r.irf_lower[:, 1, j] * sd[1], r.irf_upper[:, 1, j] * sd[1]
        resp, lo, hi = _finish(r.irf_point[:, 1, j] * sd[1], lo, hi, k, detrend)
        diag["k_breaks"] = r.k_breaks
        diag["u_col"] = j

    elif scheme == "nongauss":
        r = non_gaussian_svar(Y, p=P_LAGS, horizon=H, seed=seed)
        j = _pick_u_col(r.B0)
        k = _norm(r.irf[0, 0, j], scheme)
        resp, lo, hi = _finish(r.irf[:, 1, j], None, None, k, detrend)
        diag["u_col"] = j
        diag["kurt"] = round(float(r.kurtosis[j]), 2)
        if draws["custom"] > 0:
            def _fit(Yb):
                rb = non_gaussian_svar(Yb, p=P_LAGS, horizon=H, seed=seed)
                jb = _pick_u_col(rb.B0)
                kb = _norm(rb.irf[0, 0, jb], "nongauss-boot")
                pb, _, _ = _finish(rb.irf[:, 1, jb], None, None, kb, detrend)
                return pb

            lo, hi, nf = _resim_bootstrap(Y, _fit, draws["custom"], seed + 1)
            diag["n_fail"] = nf

    elif scheme == "la_lp":
        out = la_lp(ds["df_lp"], y="ip", x="u", horizons=range(H + 1),
                    n_lags=P_LAGS, controls=["emp", "ffr"], alpha=0.10)
        resp = out["beta"].to_numpy()          # per +1 z-unit of u — already unit-effect
        se = out["se"].to_numpy()
        lo, hi = resp - Z90 * se, resp + Z90 * se
        diag["p_aug"] = int(out["p_aug"].iloc[0])

    else:
        raise ValueError(f"run_scheme: unknown scheme {scheme!r}")

    return {"resp": resp, "lo": lo, "hi": hi, "diag": diag}


# --------------------------------------------------------------------------- #
# Spec-curve estimator
# --------------------------------------------------------------------------- #

def make_estimator(panel: pd.DataFrame, draws: dict, dataset_cache: dict,
                   seen_keys: dict):
    def estimator(_data, spec: dict) -> dict:
        tag = "{proxy}|{sample}|{detrend}|{scheme}".format(**spec)
        base = {"family": FAMILY_OF[spec["scheme"]], "sigma_hat": np.nan,
                "se": np.nan, "resp_h12": np.nan, "peak_h": -1,
                "n_obs": 0, "error": ""}
        ds_key = (spec["proxy"], spec["sample"], spec["detrend"])
        if ds_key not in dataset_cache:
            dataset_cache[ds_key] = build_dataset(panel, **{k: spec[k] for k in
                                                            ("proxy", "sample", "detrend")})
        ds = dataset_cache[ds_key]
        if ds is None:
            base["error"] = "dataset infeasible (proxy missing or <120 obs)"
            return base
        # Dedupe: identical effective dataset under a different sample label.
        owner = seen_keys.setdefault(ds["key"], spec["sample"])
        if owner != spec["sample"]:
            base["error"] = f"duplicate of sample={owner} (same effective window)"
            return base
        if spec["scheme"] == "la_lp" and spec["detrend"] == "fd":
            base["error"] = "la_lp is detrend-invariant; kept only under lt"
            return base
        if spec["scheme"].startswith("narrative") and spec["detrend"] == "fd":
            base["error"] = ("narrative kept only under lt (baseline "
                             "detrending) — Phase 2 grid scope; skip logged")
            return base
        try:
            r = run_scheme(spec["scheme"], ds, panel, draws, _seed_for(tag))
        except Exception as e:  # noqa: BLE001 — a failed cell is data, not a crash
            base["error"] = f"{type(e).__name__}: {e}"
            base["n_obs"] = ds["n_obs"]
            return base
        resp, lo, hi = r["resp"], r["lo"], r["hi"]
        ph = int(np.argmax(np.abs(resp)))
        se = np.nan
        if lo is not None and np.all(np.isfinite([lo[ph], hi[ph]])):
            se = float((hi[ph] - lo[ph]) / (2 * Z90))
        base.update({
            "sigma_hat": float(resp[ph]), "se": se, "peak_h": ph,
            "resp_h12": float(resp[12]), "n_obs": ds["n_obs"],
            **{f"diag_{k}": v for k, v in r["diag"].items()},
        })
        return base
    return estimator


# --------------------------------------------------------------------------- #
# Headline identification menu (baseline dataset, fuller bands)
# --------------------------------------------------------------------------- #

def run_menu(panel: pd.DataFrame, draws_menu: dict, out: Path) -> pd.DataFrame:
    ds = build_dataset(panel, **BASELINE)
    if ds is None:
        raise RuntimeError("run_menu: baseline dataset infeasible — aborting.")
    log(f"baseline dataset: proxy=epu {ds['key'][1]} -> {ds['key'][2]}, "
        f"lt-detrended, T={ds['n_obs']}")
    rows = []
    for scheme in FAMILY_OF:
        t0 = time.time()
        tag = f"menu|{scheme}"
        try:
            r = run_scheme(scheme, ds, panel, draws_menu, _seed_for(tag))
            resp, lo, hi = r["resp"], r["lo"], r["hi"]
            ph = int(np.argmax(np.abs(resp)))
            row = {
                "scheme": scheme, "family": FAMILY_OF[scheme],
                "peak_h": ph, "peak": resp[ph],
                "peak_lo": lo[ph] if lo is not None else np.nan,
                "peak_hi": hi[ph] if hi is not None else np.nan,
                "h12": resp[12],
                "h12_lo": lo[12] if lo is not None else np.nan,
                "h12_hi": hi[12] if hi is not None else np.nan,
                "diagnostics": json.dumps(r["diag"]),
            }
        except Exception as e:  # noqa: BLE001
            row = {"scheme": scheme, "family": FAMILY_OF[scheme],
                   "peak_h": -1, "peak": np.nan, "peak_lo": np.nan,
                   "peak_hi": np.nan, "h12": np.nan, "h12_lo": np.nan,
                   "h12_hi": np.nan, "diagnostics": f"FAILED {type(e).__name__}: {e}"}
        rows.append(row)
        log(f"  menu {scheme:10s}: peak={row['peak']:+.3f} at h={row['peak_h']:2d}, "
            f"h12={row['h12']:+.3f}  ({time.time() - t0:.1f}s)")
    menu = pd.DataFrame(rows)
    menu.to_csv(out / "headline_menu.csv", index=False)
    return menu


# --------------------------------------------------------------------------- #
# GK identified-set overlay (sign + narrative, baseline dataset)
# --------------------------------------------------------------------------- #

def run_gk_overlay(panel: pd.DataFrame, draws_menu: dict, out: Path) -> dict:
    """Giacomini-Kitagawa (2021) identified-set bands vs the posterior-
    percentile bands of the two rotation-based schemes, baseline dataset.

    Scale and conditioning conventions (also in the module docstring):
    RAW structural scale (one-SD structural shock — the set bounds are per-
    rotation extremes, not renormalisable per draw after aggregation), and
    n_var_draws=1 so the identified set and the percentile bands condition
    on the same OLS reduced form. The menu seeds are reused, so the sign /
    narrative accepted-draw sets here are exactly the menu's.
    """
    ds = build_dataset(panel, **BASELINE)
    if ds is None:
        raise RuntimeError("run_gk_overlay: baseline dataset infeasible.")
    Y, H = ds["Y"], HORIZON
    t0 = time.time()

    r_sign = sign_restriction_svar(Y, p=P_LAGS, horizon=H,
                                   restrictions=SIGN_PATTERN,
                                   n_draws=draws_menu["sign"], ci=CI,
                                   seed=_seed_for("menu|sign"))
    events = narrative_events_in(ds)
    r_narr = narrative_sign_svar(Y, p=P_LAGS, horizon=H,
                                 sign_matrix=SIGN_PATTERN,
                                 restrictions=[(ev, 0, +1) for ev in events],
                                 dates=ds["dates"], n_draws=draws_menu["narr"],
                                 ci=CI, seed=_seed_for("menu|narrative"))
    gk = gk_robust_bands(Y, p=P_LAGS, horizon=H, restrictions=SIGN_PATTERN,
                         shock_idx=0, n_var_draws=1,
                         n_q_per_draw=draws_menu["gk_q"], ci=CI,
                         seed=_seed_for("gk|baseline"))
    n_gk_acc = int(gk.n_accepted_per_draw[0])
    log(f"GK overlay: sign accepted {r_sign.n_accepted}/{draws_menu['sign']}, "
        f"narrative {r_narr.n_narrative_accepted}/{r_narr.n_traditional_accepted}"
        f" trad. (ESS {r_narr.ess:.0f}), GK set from {n_gk_acc}/"
        f"{draws_menu['gk_q']} admissible rotations  ({time.time() - t0:.1f}s)")

    h = np.arange(H + 1)
    cols = {
        "h": h,
        "gk_lo": gk.irf_lower[:, 1], "gk_med": gk.irf_median[:, 1],
        "gk_hi": gk.irf_upper[:, 1],
        "sign_lo": r_sign.irf_lower[:, 1, 0],
        "sign_med": r_sign.irf_median[:, 1, 0],
        "sign_hi": r_sign.irf_upper[:, 1, 0],
        "narr_lo": r_narr.irf_lower[:, 1, 0],
        "narr_med": r_narr.irf_median[:, 1, 0],
        "narr_hi": r_narr.irf_upper[:, 1, 0],
    }
    pd.DataFrame(cols).to_csv(out / "gk_overlay.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True,
                             constrained_layout=True)
    panels = [
        (axes[0], "Sign restrictions (RWZ)", FAMILY_COLOR["sign"],
         cols["sign_lo"], cols["sign_med"], cols["sign_hi"]),
        (axes[1], "Narrative sign restrictions (AD-RR)",
         FAMILY_COLOR["narrative-sign"],
         cols["narr_lo"], cols["narr_med"], cols["narr_hi"]),
    ]
    for ax, title, c, lo, med, hi in panels:
        ax.axhline(0.0, color="#52514e", lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.fill_between(h, cols["gk_lo"], cols["gk_hi"], color="#e7e5df",
                        zorder=2, label="GK identified set (90%)")
        ax.plot(h, cols["gk_lo"], color="#898781", lw=0.9, zorder=3)
        ax.plot(h, cols["gk_hi"], color="#898781", lw=0.9, zorder=3)
        ax.fill_between(h, lo, hi, color=c, alpha=0.30, lw=0, zorder=4,
                        label="posterior 90% percentile band")
        ax.plot(h, med, color=c, lw=2.0, zorder=5, label="posterior median")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("months after shock")
        ax.set_xlim(0, H)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e4e0", lw=0.6, zorder=0)
    axes[0].set_ylabel("IP response, %  (one-SD structural shock)")
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    axes[1].annotate(
        f"{len(events)} events; {r_narr.n_narrative_accepted}/"
        f"{r_narr.n_traditional_accepted} trad. draws survive; "
        f"ESS {r_narr.ess:.0f}",
        xy=(0.02, 0.04), xycoords="axes fraction", fontsize=7.5,
        color="#52514e")
    fig.suptitle("Percentile bands vs the identified set — same data, same "
                 "sign pattern (raw structural scale)", fontsize=11)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_gk_overlay.{ext}", dpi=300)
    plt.close(fig)
    log("wrote fig_gk_overlay.pdf/.png + gk_overlay.csv")

    def _band(lo, hi, at):
        return [round(float(lo[at]), 3), round(float(hi[at]), 3)]

    return {
        "scale": "raw structural (one-SD shock), OLS reduced form fixed",
        "narrative_events_used": [ev[:7] for ev in events],
        "gk_n_admissible": n_gk_acc,
        "h12_band_gk": _band(cols["gk_lo"], cols["gk_hi"], 12),
        "h12_band_sign": _band(cols["sign_lo"], cols["sign_hi"], 12),
        "h12_band_narrative": _band(cols["narr_lo"], cols["narr_hi"], 12),
        "h12_width_ratio_gk_vs_sign": round(
            float((cols["gk_hi"][12] - cols["gk_lo"][12])
                  / (cols["sign_hi"][12] - cols["sign_lo"][12])), 2),
        "gk_upper_positive_share": round(
            float((cols["gk_hi"] > 0).mean()), 3),
        "narrative_ess": round(float(r_narr.ess), 1),
        "narrative_vs_sign_h12_width_ratio": round(
            float((cols["narr_hi"][12] - cols["narr_lo"][12])
                  / (cols["sign_hi"][12] - cols["sign_lo"][12])), 3),
    }


# --------------------------------------------------------------------------- #
# Event sweep (opt-in via --event-sweep): the four Type-I event months are
# themselves a specification choice — sweep 9 configs (all four, each alone,
# each left out) on the baseline dataset and report what each does to the
# h=12 band. Purely additive: new output files, no existing number touched.
# --------------------------------------------------------------------------- #

def run_event_sweep(panel: pd.DataFrame, draws_menu: dict,
                    out: Path) -> dict:
    ds = build_dataset(panel, **BASELINE)
    if ds is None:
        raise RuntimeError("run_event_sweep: baseline dataset infeasible.")
    all_events = narrative_events_in(ds)
    configs: list[tuple[str, list[str]]] = [("all4", all_events)]
    configs += [(f"only_{ev[:7]}", [ev]) for ev in all_events]
    configs += [(f"drop_{ev[:7]}", [e for e in all_events if e != ev])
                for ev in all_events]
    rows = []
    for name, events in configs:
        t0 = time.time()
        r = narrative_sign_svar(
            Y=ds["Y"], p=P_LAGS, horizon=HORIZON, sign_matrix=SIGN_PATTERN,
            restrictions=[(ev, 0, +1) for ev in events], dates=ds["dates"],
            n_draws=draws_menu["narr"], ci=CI,
            seed=_seed_for(f"evsweep|{name}"))
        k = _norm(r.irf_median[0, 0, 0], f"evsweep|{name}")
        resp, lo, hi = _finish(r.irf_median[:, 1, 0], r.irf_lower[:, 1, 0],
                               r.irf_upper[:, 1, 0], k, ds["detrend"])
        ph = int(np.argmax(np.abs(resp)))
        rows.append({
            "config": name,
            "events": "+".join(ev[:7] for ev in events),
            "n_events": len(events),
            "n_trad": r.n_traditional_accepted,
            "n_narr": r.n_narrative_accepted,
            "ess": round(float(r.ess), 1),
            "peak": float(resp[ph]), "peak_h": ph,
            "h12": float(resp[12]),
            "h12_lo": float(lo[12]), "h12_hi": float(hi[12]),
            "width_h12": float(hi[12] - lo[12]),
            "excl_zero_h12": bool(hi[12] < 0 or lo[12] > 0),
        })
        log(f"  evsweep {name:15s}: h12={rows[-1]['h12']:+.3f} "
            f"[{rows[-1]['h12_lo']:+.3f}, {rows[-1]['h12_hi']:+.3f}] "
            f"width {rows[-1]['width_h12']:.3f}, "
            f"narr {rows[-1]['n_narr']}/{rows[-1]['n_trad']} "
            f"({time.time() - t0:.1f}s)")
    sweep = pd.DataFrame(rows)
    sweep.to_csv(out / "event_sweep.csv", index=False)

    # Forest figure: h=12 interval per config, all4 on top.
    fig, ax = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
    ypos = np.arange(len(sweep))[::-1]
    ax.axvline(0.0, color="#52514e", lw=0.8, ls=(0, (4, 3)), zorder=1)
    base_w = float(sweep.loc[0, "width_h12"])
    for y, (_, r) in zip(ypos, sweep.iterrows()):
        c = FAMILY_COLOR["narrative-sign"] if r["config"] == "all4" else "#52514e"
        ax.plot([r["h12_lo"], r["h12_hi"]], [y, y], color=c, lw=2.2,
                solid_capstyle="butt", zorder=2)
        ax.plot(r["h12"], y, marker="*", ms=10, color=c, zorder=3)
        ax.annotate(f"w={r['width_h12']:.2f}  n={r['n_narr']}",
                    xy=(r["h12_hi"], y), xytext=(4, 0),
                    textcoords="offset points", fontsize=7,
                    color="#52514e", va="center")
    ax.set_yticks(ypos)
    ax.set_yticklabels(sweep["config"], fontsize=8)
    ax.set_xlabel("IP response at h = 12, %  (per +1$\\sigma$ uncertainty)")
    ax.set_title("Narrative event-set sweep: 68% band at h = 12 by event "
                 "configuration (baseline dataset)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e5e4e0", lw=0.6, zorder=0)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_event_sweep.{ext}", dpi=300)
    plt.close(fig)
    log("wrote fig_event_sweep.pdf/.png + event_sweep.csv")

    only = sweep[sweep["config"].str.startswith("only_")]
    drop = sweep[sweep["config"].str.startswith("drop_")]
    tightest = only.loc[only["width_h12"].idxmin()]
    loosest_drop = drop.loc[drop["width_h12"].idxmax()]
    return {
        "all4_width_h12": base_w,
        "all4_h12_band": [float(sweep.loc[0, "h12_lo"]),
                          float(sweep.loc[0, "h12_hi"])],
        "single_event_width_range": [float(only["width_h12"].min()),
                                     float(only["width_h12"].max())],
        "tightest_single_event": str(tightest["config"]),
        "most_binding_event_by_loo": str(loosest_drop["config"]),
        "excl_zero_h12_configs": sweep.loc[sweep["excl_zero_h12"],
                                           "config"].tolist(),
    }


# --------------------------------------------------------------------------- #
# Lag-order sweep (opt-in via --ph-sweep): the VAR lag order p = 6 is a
# held-fixed specification choice the draft flags. Sweep p over
# {3, 6, 12} on the baseline dataset for the point-identified fast
# schemes + sign. Purely additive outputs; the horizon stays H = 24
# with the peak searched within it (peak_h is reported, so a separate
# H sweep adds nothing the peak search does not already reveal).
# --------------------------------------------------------------------------- #

def run_ph_sweep(panel: pd.DataFrame, draws_menu: dict,
                 out: Path) -> dict:
    global P_LAGS
    ds = build_dataset(panel, **BASELINE)
    if ds is None:
        raise RuntimeError("run_ph_sweep: baseline dataset infeasible.")
    schemes = ["chol_first", "chol_last", "sign", "proxy", "maxshare"]
    p_grid = [3, 6, 12]
    p_saved = P_LAGS
    rows = []
    try:
        for p_try in p_grid:
            P_LAGS = p_try           # run_scheme reads the module global
            for scheme in schemes:
                tag = f"phsweep|{scheme}|p{p_try}"
                try:
                    r = run_scheme(scheme, ds, panel, draws_menu,
                                   _seed_for(tag))
                    resp = r["resp"]
                    ph = int(np.argmax(np.abs(resp)))
                    rows.append({"scheme": scheme, "p": p_try,
                                 "peak": float(resp[ph]), "peak_h": ph,
                                 "h12": float(resp[12]), "error": ""})
                except Exception as e:  # noqa: BLE001 — a failed cell is data
                    rows.append({"scheme": scheme, "p": p_try,
                                 "peak": np.nan, "peak_h": -1,
                                 "h12": np.nan,
                                 "error": f"{type(e).__name__}: {e}"})
                log(f"  phsweep {scheme:10s} p={p_try:2d}: "
                    f"peak={rows[-1]['peak']:+.3f} h12={rows[-1]['h12']:+.3f}"
                    if rows[-1]["error"] == "" else
                    f"  phsweep {scheme:10s} p={p_try:2d}: FAILED")
    finally:
        P_LAGS = p_saved
    sweep = pd.DataFrame(rows)
    sweep.to_csv(out / "ph_sweep.csv", index=False)

    ok = sweep.dropna(subset=["peak"])
    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    ax.axhline(0.0, color="#52514e", lw=0.8, ls=(0, (4, 3)), zorder=1)
    for scheme in schemes:
        g = ok[ok["scheme"] == scheme]
        if g.empty:
            continue
        fam = FAMILY_OF[scheme]
        ax.plot(g["p"], g["peak"], marker=FAMILY_MARKER[fam],
                color=FAMILY_COLOR[fam], lw=1.4, ms=6, label=scheme)
    ax.set_xticks(p_grid)
    ax.set_xlabel("VAR lag order p (baseline: 6)")
    ax.set_ylabel("Peak IP response, %  (per +1$\\sigma$ uncertainty)")
    ax.set_title("Lag-order sweep, baseline dataset", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e4e0", lw=0.6, zorder=0)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_ph_sweep.{ext}", dpi=300)
    plt.close(fig)
    log("wrote fig_ph_sweep.pdf/.png + ph_sweep.csv")

    per_scheme = {s: {"range": [float(ok[ok['scheme'] == s]['peak'].min()),
                                float(ok[ok['scheme'] == s]['peak'].max())],
                      "all_negative": bool((ok[ok['scheme'] == s]['peak']
                                            < 0).all())}
                  for s in schemes if not ok[ok["scheme"] == s].empty}
    return {"p_grid": p_grid, "schemes": per_scheme,
            "n_failed": int(sweep["error"].ne("").sum())}


# --------------------------------------------------------------------------- #
# Figure + tables
# --------------------------------------------------------------------------- #

def make_figure(curve: pd.DataFrame, out: Path, mode: str) -> None:
    ok = curve.dropna(subset=["sigma_hat"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.2, 4.6), constrained_layout=True)
    ax.axhline(0.0, color="#52514e", lw=0.8, ls=(0, (4, 3)), zorder=1)
    x = np.arange(len(ok))
    # Cap the axis so the body of the curve stays readable; weak-instrument
    # proxy cells reach −47 with ±50 whiskers and would flatten everything
    # else. Clipped specs are drawn ON the boundary and counted in a note.
    y_lo, y_hi = -22.0, 6.0
    clipped = ok["sigma_hat"] < y_lo
    for fam in FAMILY_ORDER:
        m = ((ok["family"] == fam) & ~clipped).to_numpy()
        c = FAMILY_COLOR[fam]
        if m.any():
            ax.vlines(x[m], ok.loc[m, "ci_lo"].clip(lower=y_lo),
                      ok.loc[m, "ci_hi"].clip(upper=y_hi),
                      color=c, lw=1.0, alpha=0.30, zorder=2)
            ax.scatter(x[m], ok.loc[m, "sigma_hat"],
                       s=FAMILY_MARKER_SIZE.get(fam, 26),
                       marker=FAMILY_MARKER[fam], facecolor=c,
                       edgecolor="#0b0b0b", linewidth=0.4, label=fam, zorder=3)
        mc = ((ok["family"] == fam) & clipped).to_numpy()
        if mc.any():
            ax.scatter(x[mc], np.full(mc.sum(), y_lo + 0.6), s=30, marker="v",
                       facecolor="none", edgecolor=c, linewidth=1.2, zorder=3)
    if clipped.any():
        ax.annotate(
            f"$\\triangledown$ {int(clipped.sum())} specs below axis "
            f"(min {ok['sigma_hat'].min():.0f}; weak-instrument proxy cells — "
            "full values in spec_curve_results.csv)",
            xy=(0.02, 0.06), xycoords="axes fraction", fontsize=7.5,
            color="#52514e")
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Specification rank (sorted by estimate)")
    ax.set_ylabel("Peak IP response, %  (per +1$\\sigma$ uncertainty)")
    ax.set_title("Peak activity response to an uncertainty shock across "
                 f"{len(ok)} specifications", fontsize=11)
    ax.legend(frameon=False, ncols=4, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e4e0", lw=0.6, zorder=0)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_spec_curve.{ext}", dpi=300)
    plt.close(fig)
    log(f"wrote fig_spec_curve.pdf/.png ({len(ok)} plotted specs, mode={mode})")


def family_table(curve: pd.DataFrame, out: Path) -> pd.DataFrame:
    ok = curve.dropna(subset=["sigma_hat"])
    rows = []
    for fam in FAMILY_ORDER:
        sub = ok[ok["family"] == fam]
        if sub.empty:
            continue
        rows.append({
            "family": fam,
            "n_specs": len(sub),
            "median_peak": float(sub["sigma_hat"].median()),
            "median_h12": float(sub["resp_h12"].median()),
            "share_peak_negative": float((sub["sigma_hat"] < 0).mean()),
            "min_peak": float(sub["sigma_hat"].min()),
            "max_peak": float(sub["sigma_hat"].max()),
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "table_family_medians.csv", index=False)
    md = ["| family | n | median peak | median h=12 | share peak<0 | min | max |",
          "|---|---|---|---|---|---|---|"]
    for _, r in tab.iterrows():
        md.append(f"| {r['family']} | {r['n_specs']:.0f} | {r['median_peak']:+.3f} "
                  f"| {r['median_h12']:+.3f} | {r['share_peak_negative']:.0%} "
                  f"| {r['min_peak']:+.3f} | {r['max_peak']:+.3f} |")
    (out / "table_family_medians.md").write_text("\n".join(md) + "\n")
    return tab


def variance_decomposition(curve: pd.DataFrame) -> dict:
    """R² of peak estimates on scheme dummies vs data-dimension dummies —
    the paper's 'how much is identification choice' number."""
    ok = curve.dropna(subset=["sigma_hat"])
    y = ok["sigma_hat"].to_numpy()

    def _r2(cols: list[str]) -> float:
        X = pd.get_dummies(ok[cols].astype(str), drop_first=True).astype(float)
        X.insert(0, "const", 1.0)
        b = np.linalg.lstsq(X.to_numpy(), y, rcond=None)[0]
        e = y - X.to_numpy() @ b
        return float(1.0 - e.var() / y.var())

    return {
        "r2_scheme_only": round(_r2(["scheme"]), 4),
        "r2_data_dims_only": round(_r2(["proxy", "sample", "detrend"]), 4),
        "r2_joint": round(_r2(["scheme", "proxy", "sample", "detrend"]), 4),
    }


# --------------------------------------------------------------------------- #
# Determinism check (frozen csv -> identical figure data)
# --------------------------------------------------------------------------- #

def determinism_check(curve: pd.DataFrame, out: Path, draws: dict,
                      n_check: int = 12) -> None:
    """Reload the frozen csv, recompute a fixed subset of specs, assert the
    recomputed estimates equal the stored curve bit-for-bit."""
    log(f"determinism check: reloading frozen csv, recomputing {n_check} specs …")
    panel2 = load_frozen_panel(out)
    est2 = make_estimator(panel2, draws, {}, {})
    ok = curve.dropna(subset=["sigma_hat"]).reset_index(drop=True)
    rng = np.random.default_rng(7)
    picks = rng.choice(len(ok), size=min(n_check, len(ok)), replace=False)
    for i in picks:
        row = ok.iloc[int(i)]
        spec = {k: row[k] for k in ("proxy", "sample", "detrend", "scheme")}
        redo = est2(None, spec)
        for fld in ("sigma_hat", "resp_h12"):
            if not np.isclose(redo[fld], row[fld], rtol=0, atol=0, equal_nan=True):
                raise AssertionError(
                    f"determinism_check: spec {spec} field {fld} changed on "
                    f"rerun from frozen csv: {row[fld]!r} -> {redo[fld]!r}")
    log("determinism check PASSED (same seed -> same figure data)")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true",
                    help="reduced grid + capped draws (< 3 min)")
    ap.add_argument("--out", type=Path,
                    default=REPO / "docs/research/uncertainty_identification/output")
    ap.add_argument("--refresh-data", action="store_true",
                    help="re-assemble the panel even if a frozen csv exists")
    ap.add_argument("--narrative-hd", action="store_true",
                    help="Phase 3: add narrative_t2/t3 (AD-RR Type II/III "
                         "HD-dominance) cells to the menu and grid. "
                         "Additive: existing cell tags/seeds unchanged; "
                         "default run is byte-identical without this flag.")
    ap.add_argument("--event-sweep", action="store_true",
                    help="Phase 3: sweep the Type-I event set (9 configs) "
                         "on the baseline dataset; writes event_sweep.csv "
                         "+ fig_event_sweep. Purely additive outputs.")
    ap.add_argument("--ph-sweep", action="store_true",
                    help="Phase 4: sweep the VAR lag order p over {3,6,12} "
                         "on the baseline dataset (fast schemes + sign); "
                         "writes ph_sweep.csv + fig_ph_sweep. Additive.")
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    mode = "quick" if args.quick else "full"
    log(f"mode={mode}  out={out}")
    if args.narrative_hd:
        FAMILY_OF.update(HD_SCHEMES)
        log("Phase 3: narrative_t2 (Type II @ Lehman+COVID) and "
            "narrative_t3 (Type II @ Lehman, Type III @ COVID) added to "
            "menu + grid")

    # ---- draw budgets ------------------------------------------------------
    if args.quick:
        draws_grid = {"var": 30, "sign": 300, "magmav": 0, "custom": 0,
                      "narr": 300}
        draws_menu = {"var": 50, "sign": 600, "magmav": 10, "custom": 30,
                      "narr": 600, "gk_q": 800}
        pval_B = 500
        grid = {"proxy": ["epu", "vix"], "sample": ["full", "pre2020"],
                "detrend": ["lt"], "scheme": list(FAMILY_OF)}
        log("QUICK mode reductions: grid -> proxies {epu,vix} × samples "
            "{full,pre2020} × detrend {lt}; VAR bootstraps 30 (grid) / 50 "
            "(menu); sign draws 300/600; narrative draws 300/600; magmav "
            "bands 0/10; rigobon+ICA custom bootstrap 0/30; GK rotations "
            "800; median p-value B=500.")
    else:
        draws_grid = {"var": 100, "sign": 1200, "magmav": 0, "custom": 60,
                      "narr": 1200}
        draws_menu = {"var": 300, "sign": 3000, "magmav": 100, "custom": 200,
                      "narr": 6000, "gk_q": 5000}
        pval_B = 2000
        grid = {"proxy": ["epu", "wui", "jln", "vix"],
                "sample": ["full", "post1985", "pre2020"],
                "detrend": ["lt", "fd"], "scheme": list(FAMILY_OF)}
        log("FULL mode caps (runtime budget ~20 min): VAR bootstraps 100 "
            "(grid) / 300 (menu); sign draws 1200/3000; narrative draws "
            "1200/6000 (grid narrative cells restricted to lt); magmav bands "
            "only in the menu (100); rigobon+ICA custom bootstrap 60/200; GK "
            "rotations 5000; median p-value B=2000.")

    # ---- panel: assemble or reload frozen ----------------------------------
    csv = out / "panel_monthly.csv"
    if csv.exists() and not args.refresh_data:
        log(f"using frozen panel {csv.name} (pass --refresh-data to rebuild)")
    else:
        log("assembling panel …")
        panel_raw, manifest = assemble_panel(REPO)
        freeze_panel(panel_raw, manifest, out)
    panel = load_frozen_panel(out)
    log(f"panel loaded: {len(panel)} rows, cols={list(panel.columns)}")

    # ---- headline identification menu --------------------------------------
    menu = run_menu(panel, draws_menu, out)

    # ---- GK identified-set overlay (sign + narrative, baseline) ------------
    gk_summary = run_gk_overlay(panel, draws_menu, out)

    # ---- event sweep (opt-in) ------------------------------------------------
    sweep_summary = None
    if args.event_sweep:
        sweep_summary = run_event_sweep(panel, draws_menu, out)

    # ---- lag-order sweep (opt-in) ---------------------------------------------
    ph_summary = None
    if args.ph_sweep:
        ph_summary = run_ph_sweep(panel, draws_menu, out)

    # ---- spec curve ---------------------------------------------------------
    specs = enumerate_specs(grid)
    log(f"spec grid: {len(specs)} cells")
    dataset_cache: dict = {}
    seen_keys: dict = {}
    estimator = make_estimator(panel, draws_grid, dataset_cache, seen_keys)
    curve = run_spec_curve(data=None, specs=specs, estimator=estimator, ci_level=CI)
    curve.to_csv(out / "spec_curve_results.csv", index=False)
    ok = curve.dropna(subset=["sigma_hat"])
    skipped = curve[curve["sigma_hat"].isna()]
    log(f"curve: {len(ok)} estimated, {len(skipped)} skipped/failed")
    for reason, cnt in skipped["error"].value_counts().items():
        log(f"    skipped ×{cnt}: {reason}")

    pval = bootstrap_pvalue_median(curve, h0=0.0, B=pval_B,
                                   rng=np.random.default_rng(20260719))
    log(f"bootstrap p-value for H0: median peak effect = 0  ->  p = {pval:.4f} "
        f"(B={pval_B})")

    make_figure(curve, out, mode)
    tab = family_table(curve, out)

    # ---- headline numbers ---------------------------------------------------
    menu_ok = menu.dropna(subset=["peak"])
    headline = {
        "mode": mode,
        "menu_peak_range": [float(menu_ok["peak"].min()), float(menu_ok["peak"].max())],
        "menu_all_negative_peak": bool((menu_ok["peak"] < 0).all()),
        "menu_schemes_estimated": int(len(menu_ok)),
        "curve_n_estimated": int(len(ok)),
        "curve_n_skipped": int(len(skipped)),
        "curve_peak_range": [float(ok["sigma_hat"].min()), float(ok["sigma_hat"].max())],
        "curve_median_peak": float(ok["sigma_hat"].median()),
        "curve_share_peak_negative": float((ok["sigma_hat"] < 0).mean()),
        "median_pvalue_h0_zero": float(pval),
        "gk_overlay": gk_summary,
        "variance_decomposition": variance_decomposition(curve),
        "family_medians": {r["family"]: round(r["median_peak"], 4)
                           for _, r in tab.iterrows()},
        "runtime_seconds": round(time.time() - _T0, 1),
    }
    if args.narrative_hd:
        nhd = menu[menu["scheme"].isin(HD_SCHEMES)]
        headline["narrative_hd"] = {
            r["scheme"]: {"peak": (None if pd.isna(r["peak"])
                                   else round(float(r["peak"]), 4)),
                          "h12": (None if pd.isna(r["h12"])
                                  else round(float(r["h12"]), 4)),
                          "diagnostics": r["diagnostics"]}
            for _, r in nhd.iterrows()}
    if sweep_summary is not None:
        headline["event_sweep"] = sweep_summary
    if ph_summary is not None:
        headline["ph_sweep"] = ph_summary
    (out / "summary.json").write_text(json.dumps(headline, indent=2))
    log("summary.json written: " + json.dumps(headline["family_medians"]))

    # ---- determinism ---------------------------------------------------------
    determinism_check(curve, out, draws_grid)

    headline["runtime_seconds"] = round(time.time() - _T0, 1)
    (out / "summary.json").write_text(json.dumps(headline, indent=2))
    (out / "run_log.txt").write_text("\n".join(_LOG_LINES) + "\n")
    log(f"DONE in {headline['runtime_seconds']:.0f}s")


if __name__ == "__main__":
    main()
