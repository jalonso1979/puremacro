"""Builds panel_Q and panel_M from the fetch modules.

Entry point: ``build_all(countries=..., fast=False, refresh=False)``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from . import regime_dates as regimes  # Phase 5B rename — date dicts
from .build_subnational_panel import build_all as _build_subnational
from .fetch import (wui, wui_extras, epu, gpr, jln, lmn, fernald, fred,
                    oecd, yahoo, oecd_qna_local, oecd_mei, imf_ifs, hrs_mpu,
                    wb_pink_sheet, oecd_energy, oecd_fx, oecd_qna_panel,
                    oecd_qna_expenditure)
from ._codes import drop_aggregates as _drop_aggregates, is_country
from .uncertainty import build_backbone, build_composite, build_innovation


def _drop_aggregates_in_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose ``code`` is a regional aggregate (EA20, EU27, OECD, etc.).

    Applied centrally in ``build_all`` so individual fetchers don't need
    to know about the filter.
    """
    return _drop_aggregates(panel, code_col="code")


def _apply_country_filter(panel: pd.DataFrame, countries: Iterable[str] | None) -> pd.DataFrame:
    """Filter *panel* to *countries* iff *countries* is not None.

    With ``countries=None`` (the new default), every country a fetcher
    returned survives — that's the sparse-panel contract.
    """
    if countries is None:
        return panel.reset_index(drop=True)
    return panel[panel["code"].isin(list(countries))].reset_index(drop=True)


def _gap_fill_ifs(panel: pd.DataFrame, ifs_panel: pd.DataFrame,
                  *, oecd_var: str, ifs_var: str) -> pd.DataFrame:
    """Idempotent gap-fill: promote IFS rows for ``ifs_var`` into ``oecd_var``
    only for ``(code, date)`` cells not already populated by OECD.

    Both the raw ``ifs_var`` row and the promoted ``oecd_var`` row are
    emitted for gap cells. For cells where OECD already covers
    ``oecd_var``, the IFS rows for that cell are dropped entirely.
    """
    if ifs_panel.empty:
        return panel
    ifs_only = ifs_panel[ifs_panel["variable"] == ifs_var]
    if ifs_only.empty:
        return panel

    existing = panel[panel["variable"] == oecd_var][["code", "date"]].assign(_in=True)
    candidates = ifs_only.merge(existing, on=["code", "date"], how="left")
    gap_rows = candidates[candidates["_in"].isna()].drop(columns="_in").copy()
    if gap_rows.empty:
        return panel

    promoted = gap_rows.copy()
    promoted["variable"] = oecd_var

    parts = [df for df in [panel, gap_rows, promoted] if not df.empty]
    return pd.concat(parts, ignore_index=True) if parts else panel


def _resolve_data_dir() -> Path:
    """Return the project-root data/processed/ directory.

    Walks up from this file looking for a directory that has both
    pyproject.toml and a non-empty data/processed/ sub-directory with
    panel_Q.parquet present.  Falls back to the legacy package-relative
    path (puremacro/data/processed/) so that installs outside the repo
    still work.
    """
    _here = Path(__file__).resolve()
    for candidate in [_here.parent, *_here.parents]:
        _proc = candidate / "data" / "processed"
        if (candidate / "pyproject.toml").exists() and (_proc / "panel_Q.parquet").exists():
            return _proc
    # legacy fallback: puremacro/data/processed/
    return _here.parent.parent / "data" / "processed"


DATA_DIR = _resolve_data_dir()
PANEL_Q_PATH = DATA_DIR / "panel_Q.parquet"
PANEL_M_PATH = DATA_DIR / "panel_M.parquet"
COV_Q_PATH = DATA_DIR / "coverage_Q.csv"
COV_M_PATH = DATA_DIR / "coverage_M.csv"

UNCERTAINTY_PROXIES_Q = ("wui_q", "epu_q", "gpr_q", "rv_q", "wtui_q", "wpui_q", "mpu_q")
UNCERTAINTY_PROXIES_M = ("wui_m", "epu_m", "gpr_m", "rv_m", "wtui_m", "wpui_m", "mpu_m")

CORE_COUNTRIES = ("USA", "MEX", "CAN", "GBR", "DEU", "FRA", "ITA", "ESP", "JPN", "KOR", "AUS", "BRA")
FAST_COUNTRIES = ("USA", "MEX", "DEU", "JPN", "GBR", "BRA")
# Extended set for the teaching track: adds 5 OECD-Europe countries with WUI + GPR
# + OECD GDP/emp/CPI coverage. Requires a panel rebuild (`build_all(countries=CORE_COUNTRIES_EXT)`)
# before teaching notebooks can read the new countries.
CORE_COUNTRIES_EXT = CORE_COUNTRIES + ("NLD", "SWE", "NOR", "CHE", "POL")

SAMPLE_RECIPES: list[dict] = [
    {"recipe": "t2_monthly_basic",
     "variables": ["unc_m", "log_cpi", "urate"]},
    {"recipe": "t2_monthly_full",
     "variables": ["unc_pc_m", "log_cpi", "urate", "log_ip"]},
    {"recipe": "t1_quarterly_inv",
     "variables": ["unc_q", "log_gfcf_real", "log_hours_qna"]},
    {"recipe": "t1_quarterly_full",
     "variables": ["unc_pc_q", "log_gdp_real", "log_con_real",
                   "log_gfcf_real", "log_govcon_real",
                   "log_exports_real", "log_imports_real",
                   "log_emp_qna", "log_hours_qna"]},
]


def merge_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    panel = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    panel = panel.drop_duplicates(subset=["code", "date", "variable"], keep="first")
    return panel.sort_values(["code", "variable", "date"]).reset_index(drop=True)


def build_coverage_report(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame(columns=["code", "variable", "first_obs", "last_obs",
                                     "n_obs", "pct_missing", "sa_source"])
    grp = panel.groupby(["code", "variable"])
    out = grp.agg(
        first_obs=("date", "min"),
        last_obs=("date", "max"),
        n_obs=("value", "count"),
        sa_source=("sa_source", "first"),
    ).reset_index()

    def _expected_n(row):
        d = pd.date_range(row["first_obs"], row["last_obs"], freq="MS" if row["variable"].endswith("_m") else "QS")
        return max(1, len(d))

    out["pct_missing"] = 1 - out["n_obs"] / out.apply(_expected_n, axis=1)
    out["pct_missing"] = out["pct_missing"].clip(lower=0)
    return out[["code", "variable", "first_obs", "last_obs", "n_obs", "pct_missing", "sa_source"]]


def build_variable_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-variable rollup: how many distinct countries x first/last date."""
    if panel.empty:
        return pd.DataFrame(columns=["variable", "n_countries", "first_obs", "last_obs"])
    grp = panel.groupby("variable")
    return grp.agg(
        n_countries=("code", "nunique"),
        first_obs=("date", "min"),
        last_obs=("date", "max"),
    ).reset_index().sort_values("variable").reset_index(drop=True)


def build_sample_recipes(panel: pd.DataFrame, recipes: list[dict] | None = None) -> pd.DataFrame:
    """For each recipe, count countries with all listed variables present."""
    if recipes is None:
        recipes = SAMPLE_RECIPES
    if panel.empty:
        return pd.DataFrame(columns=["recipe", "variables", "n_countries", "first_obs", "last_obs"])

    rows: list[dict] = []
    for r in recipes:
        var_codes: list[set[str]] = []
        for v in r["variables"]:
            sub = panel[panel["variable"] == v]
            if sub.empty:
                var_codes.append(set())
            else:
                var_codes.append(set(sub["code"].unique()))
        common = set.intersection(*var_codes) if var_codes else set()
        sub_panel = panel[panel["variable"].isin(r["variables"])
                          & panel["code"].isin(common)] if common else panel.iloc[0:0]
        rows.append({
            "recipe": r["recipe"],
            "variables": ",".join(r["variables"]),
            "n_countries": len(common),
            "first_obs": sub_panel["date"].min() if not sub_panel.empty else pd.NaT,
            "last_obs": sub_panel["date"].max() if not sub_panel.empty else pd.NaT,
        })
    return pd.DataFrame(rows)


def _validate_panel(panel: pd.DataFrame) -> None:
    """Build-time sanity asserts. Raises AssertionError on hard failures,
    UserWarning on soft signals (thin coverage).

    Coverage-count thresholds (GPR >= 40, WUI_Q >= 60) are only checked
    when the panel contains at least 20 distinct country codes — below that
    threshold the panel is clearly a test fixture and the thresholds are
    not meaningful.
    """
    if panel.empty:
        return
    # Hard: no aggregate codes
    bad_codes = sorted(c for c in panel["code"].unique() if not is_country(c))
    assert not bad_codes, f"aggregate codes leaked into panel: {bad_codes}"
    # Hard: no (code, date, variable) duplicates
    dup_count = panel.duplicated(subset=["code", "date", "variable"]).sum()
    assert dup_count == 0, f"duplicate (code,date,variable) rows: {dup_count}"
    # Hard: GPR-C must keep its 40+ countries; WUI_Q must keep 60+
    # (skipped for thin test fixtures with < 20 countries total)
    total_countries = panel["code"].nunique()
    if total_countries >= 20:
        gpr_n = panel[panel["variable"] == "gpr_m"]["code"].nunique()
        wuiq_n = panel[panel["variable"] == "wui_q"]["code"].nunique()
        if gpr_n > 0:
            assert gpr_n >= 40, f"GPR-C dropped to {gpr_n} countries (expected >= 40)"
        if wuiq_n > 0:
            assert wuiq_n >= 60, f"WUI-Q dropped to {wuiq_n} countries (expected >= 60)"
    # Soft: warn on any variable with < 5 countries
    for var, n in panel.groupby("variable")["code"].nunique().items():
        if n < 5:
            warnings.warn(f"{var}: thin coverage ({n} countries)", UserWarning,
                          stacklevel=2)


def sa_audit(panel: pd.DataFrame) -> pd.DataFrame:
    """Report Kruskal-Wallis seasonality flags for any non-exempt 'none' SA series. Non-blocking.

    The exempt set excludes proxies that have already been routed through
    X-13 (their ``sa_source`` is ``'x13'``, so the filter on
    ``sa_source=='none'`` skips them anyway).
    """
    exempt = {"rv_q", "rv_m", "vix",
              "jln_macro_1", "jln_macro_3", "jln_macro_12",
              "lmn_real", "lmn_fin", "tfp_fernald"}
    flags = []
    for (code, var), grp in panel.groupby(["code", "variable"]):
        if var in exempt or grp["sa_source"].iloc[0] != "none":
            continue
        try:
            s = grp.set_index("date")["value"].astype(float).dropna()
            if len(s) < 36:
                continue
            from scipy.stats import kruskal
            if s.index.inferred_freq and s.index.inferred_freq.startswith("M"):
                groups = [v.values for _, v in s.groupby(s.index.month)]
            else:
                groups = [v.values for _, v in s.groupby(s.index.quarter)]
            groups = [g for g in groups if len(g) >= 2]
            if len(groups) < 2:
                continue
            stat, p = kruskal(*groups)
            flags.append({"code": code, "variable": var, "kw_stat": stat, "kw_p": p,
                          "sa_flag": "fail" if p < 0.05 else "pass"})
        except Exception:
            continue
    return pd.DataFrame(flags)


# Variables that arrive NSA and need centralized X-13 SA before regressions.
#
# NOTE: uncertainty proxies (EPU, GPR, WUI, VIX, JLN) are *shock measures*,
# not real-economy outcomes. They must NOT be deseasonalized — applying
# X-13 to them removes legitimate signal. Only real-economy outcomes
# (prices, employment, output, hours) are routed through SA when their
# fetcher returned them unadjusted.
_X13_TARGETS_M: set[str] = set()
_X13_TARGETS_Q: set[str] = set()
# Real-economy variables routed through X-13 only when sa_source == 'none'
# (i.e., the row came from a fetcher that didn't ship a SA series — e.g.
# OECD CPI for Mexico, or IMF IFS gap-filled values for non-OECD members).
_X13_FALLBACK_VARS = {"log_cpi", "log_ip", "urate"}


def _apply_x13_to_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Run X-13 ARIMA-SEATS on selected NSA variables, in-place by (code, var).

    The function replaces values for any (code, variable) pair where the
    variable is targeted and the row's ``sa_source`` is currently
    ``'none'``. After successful adjustment, ``sa_source`` is set to
    ``'x13'`` (or ``'stl'`` when the X-13 binary declined the series and
    the helper fell back to STL).
    """
    if panel.empty:
        return panel
    from .sa import deseasonalize_x13, x13_available

    out = panel.copy()
    out["sa_source"] = out["sa_source"].astype(object)
    fallback_used = not x13_available()

    def _is_monthly(s: pd.Series) -> bool:
        d = pd.to_datetime(s).sort_values().drop_duplicates()
        if len(d) < 6:
            return False
        diffs = d.diff().dt.days.dropna()
        return diffs.median() < 60  # quarterly ≈ 90, monthly ≈ 30

    targets: list[tuple[str, str, str]] = []
    for var in sorted(set(out["variable"].unique())
                      & (_X13_TARGETS_M | _X13_TARGETS_Q | _X13_FALLBACK_VARS)):
        sub = out[out["variable"] == var]
        nsa_codes = sorted(sub.loc[sub["sa_source"] == "none", "code"].unique())
        if not nsa_codes:
            continue
        # Decide frequency from the data (not the variable name) so that
        # monthly fallback CPI / quarterly WUI are routed correctly.
        freq_letter = "M" if _is_monthly(sub["date"]) else "Q"
        for code in nsa_codes:
            targets.append((code, var, freq_letter))

    if not targets:
        return out

    print(f"[build] X-13 SA over {len(targets)} (code, variable) cells "
          f"(binary {'available' if x13_available() else 'NOT FOUND — STL fallback'})")

    # Run per (var, freq) to share the deseasonalize_x13 setup.
    for var in {t[1] for t in targets}:
        for freq in ("M", "Q"):
            cells = [(c, v, f) for (c, v, f) in targets if v == var and f == freq]
            if not cells:
                continue
            codes = [c for (c, _, _) in cells]
            mask = (out["variable"] == var) & out["code"].isin(codes)
            sub = out.loc[mask, ["code", "date", "value"]].copy()
            sub = sub.sort_values(["code", "date"]).reset_index(drop=False)
            sa = deseasonalize_x13(
                sub.rename(columns={"value": "_v"}),
                "_v",
                by="code",
                date_col="date",
                freq=freq,
                min_obs=24 if freq == "M" else 12,
            )
            sub["value_sa"] = sa.values
            ok = sub["value_sa"].notna()
            new_idx = sub.loc[ok, "index"].values
            out.loc[new_idx, "value"] = sub.loc[ok, "value_sa"].values
            out.loc[new_idx, "sa_source"] = "x13" if not fallback_used else "stl"
    return out


def compute_garch_sigma(panel: pd.DataFrame, proxy: str) -> pd.DataFrame:
    from arch import arch_model  # lazy: Pyodide contract — see ARCHITECTURE.md

    frames = []
    target_var = f"garch_sigma_{proxy}"
    for code, grp in panel[panel["variable"] == proxy].groupby("code"):
        s = grp.set_index("date")["value"].astype(float).dropna()
        if len(s) < 40:
            continue
        # Scale and take first-differences / returns depending on level vs index.
        # For uncertainty indices we use first differences scaled by 100.
        x = 100 * s.diff().dropna()
        try:
            res = arch_model(x, mean="Zero", vol="GARCH", p=1, q=1, rescale=False).fit(disp="off")
            sigma: pd.Series = pd.Series(res.conditional_volatility / 100.0)
            # Align sigma to the date index of the input
            sigma.index = x.index
        except Exception:
            continue
        out = sigma.reset_index()
        out.columns = ["date", "value"]
        out["code"] = code
        out["variable"] = target_var
        out["sa_source"] = "derived"
        out["source"] = f"GARCH(1,1) on {proxy}"
        frames.append(out[["code", "date", "variable", "value", "sa_source", "source"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["code", "date", "variable", "value", "sa_source", "source"]
    )


def _quarterly_from_monthly(df_m: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Aggregate an already-SA monthly series to quarterly (mean for levels, last for rates)."""
    rule = "last" if variable in ("urate", "short_rate") else "mean"
    out = []
    for code, grp in df_m.groupby("code"):
        s = grp.set_index("date")["value"].astype(float).sort_index()
        q = s.resample("QS").mean() if rule == "mean" else s.resample("QS").last()
        row = q.reset_index()
        row.columns = ["date", "value"]
        row["code"] = code
        row["variable"] = variable
        row["sa_source"] = grp["sa_source"].iloc[0]
        row["source"] = f"resampled_from_M:{grp['source'].iloc[0]}"
        out.append(row[["code", "date", "variable", "value", "sa_source", "source"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["code", "date", "variable", "value", "sa_source", "source"]
    )


def _derive_real_local_oil(panel_m: pd.DataFrame) -> pd.DataFrame:
    """Compute ``log_oil_brent_real_local_m`` per country.

    Definition::

        log_oil_brent_real_local = log(brent_USD) + log(LCU/USD) - log(CPI)

    The Pink Sheet fetcher emits Brent as USD/bbl (price level), so we
    take ``np.log`` here. Requires ``xrate_usd_per_local_m`` and
    ``log_cpi`` for the country (skipped per-country if either missing).
    """
    cols = ["code", "date", "variable", "value", "sa_source", "source"]
    empty = pd.DataFrame(columns=cols)
    if panel_m.empty:
        return empty

    brent = panel_m[(panel_m["code"] == "WLD") & (panel_m["variable"] == "oil_brent_m")]
    if brent.empty:
        return empty
    brent_s = brent.set_index("date")["value"].astype(float).sort_index()
    log_brent = np.log(brent_s.replace(0, np.nan)).dropna()

    rows = []
    for code in panel_m["code"].unique():
        if code == "WLD":
            continue
        fx_g = panel_m[(panel_m["code"] == code) & (panel_m["variable"] == "xrate_usd_per_local_m")]
        cpi_g = panel_m[(panel_m["code"] == code) & (panel_m["variable"] == "log_cpi")]
        if cpi_g.empty:
            continue
        # log_cpi is already in log space.
        log_cpi = cpi_g.set_index("date")["value"].astype(float).sort_index()
        if fx_g.empty:
            # USA's local currency is USD, so OECD doesn't publish an LCU/USD
            # rate for USA — treat log_fx ≡ 0 (1:1) for USA only. Other
            # countries lacking FX are skipped (insufficient inputs).
            if code != "USA":
                continue
            log_fx = pd.Series(0.0, index=log_brent.index, name="log_fx")
        else:
            fx_s = fx_g.set_index("date")["value"].astype(float).sort_index()
            log_fx = np.log(fx_s.replace(0, np.nan)).dropna()
        df = pd.concat([log_brent, log_fx, log_cpi], axis=1, join="inner")
        df.columns = ["log_brent", "log_fx", "log_cpi"]
        df = df.dropna()
        if df.empty:
            continue
        # log(brent_USD) + log(LCU/USD) = log(brent in LCU); subtract log(CPI) for real.
        real = df["log_brent"] + df["log_fx"] - df["log_cpi"]
        out = real.reset_index()
        out.columns = ["date", "value"]
        out["code"] = code
        out["variable"] = "log_oil_brent_real_local_m"
        out["sa_source"] = "derived"
        out["source"] = "log(brent_USD)+log(LCU/USD)-log(CPI)"
        rows.append(out[cols])
    if not rows:
        return empty
    return pd.concat(rows, ignore_index=True)


def _resample_monthly_emp_to_quarterly(panel_m: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly ``log_emp`` to quarterly via simple mean of the months in the quarter.

    Returns rows for ``log_emp_qna`` with ``sa_source='aliased_from_monthly_LFS'``
    only for ``(code, quarter)`` pairs where ALL three months in the quarter
    have data. Used as a fallback when neither the local Volatility/QNA.xlsx
    workbook nor the OECD QNA SDMX fetcher provides quarterly employment for
    a given country (typically AUS / CAN / JPN).
    """
    schema_cols = ["code", "date", "variable", "value", "sa_source", "source"]
    if panel_m.empty or "variable" not in panel_m.columns:
        return pd.DataFrame(columns=schema_cols)
    emp_m = panel_m[panel_m["variable"] == "log_emp"]
    if emp_m.empty:
        return pd.DataFrame(columns=schema_cols)
    rows = []
    for code, grp in emp_m.groupby("code"):
        s = (grp.sort_values("date")
                .drop_duplicates(subset=["date"], keep="first")
                .set_index("date")["value"].astype(float))
        gq = s.groupby(pd.Grouper(freq="QS"))
        means = gq.mean()
        counts = gq.count()
        means = means[counts == 3]
        if means.empty:
            continue
        out = means.reset_index()
        out.columns = ["date", "value"]
        out["code"] = code
        out["variable"] = "log_emp_qna"
        out["sa_source"] = "aliased_from_monthly_LFS"
        out["source"] = "resampled_from_M:OECD_LFS:log_emp"
        rows.append(out[schema_cols])
    if not rows:
        return pd.DataFrame(columns=schema_cols)
    return pd.concat(rows, ignore_index=True)


#: OECD SDMX labour-block column -> panel variable. ``qna_labor`` returns
#: LEVELS (thousands of persons, millions of hours); the panel's contract for
#: these two variables is natural logs, which is what ``oecd_qna_local``, the
#: LFS resample and the FRED HOANBS alias all emit.
_QNA_LABOR_VARS = {"emp": "log_emp_qna", "hours": "log_hours_qna"}

#: ``qna_labor``'s ``sa_source`` vocabulary -> the panel's. ``puremacro``
#: means the fetcher ran the adjustment itself under ``sa="x13"`` — which is
#: what the retired route only ever promised (``x13_pending``) and never did.
#: ``none`` survives for a series too short for the engine, and is honest:
#: ``sa_audit`` acts on ``none`` and would have ignored ``x13_pending``.
_QNA_SA_SOURCE = {"oecd": "oecd", "puremacro": "x13", "none": "none"}

_QNA_LABOR_SOURCE = "OECD:DSD_NAMAIN1@DF_QNA_BY_ACTIVITY_EMPDC"


def _fetch_qna_labor_logs(codes: Sequence[str] | None,
                          *, start: str = "1995") -> pd.DataFrame:
    """QNA total-economy employment and hours as ``log_emp_qna`` / ``log_hours_qna``.

    Adapter over :func:`puremacro.fetch.qna_labor`, which replaced
    ``fetch.oecd_qna_labor.fetch_qna_labor``: it reaches the same two totals
    through a route that actually seasonally adjusts the reference areas
    publishing the labour block raw (``sa="x13"``) instead of labelling them
    ``x13_pending`` and leaving them alone.

    It deliberately calls ``qna_labor`` rather than ``qna_panel(labor=True)``:
    the latter would download the expenditure block as well, run X-13 over it,
    and — because it filters the labour rows to the countries the expenditure
    flow returned — silently drop a country that publishes labour but not
    expenditure, or lose the whole gap-fill if that request came back empty.

    Returns the panel's long schema; an empty frame (never an exception) when
    the download fails or none of the requested countries publish the block.
    """
    schema_cols = ["code", "date", "variable", "value", "sa_source", "source"]
    empty = pd.DataFrame(columns=schema_cols)
    try:
        long = oecd_qna_panel.qna_labor(codes, start=start, sa="x13")
    except Exception:
        # Same contract as the route this replaced: a failed download degrades
        # to "this source contributed nothing", it does not take a build down.
        return empty
    if long.empty:
        return empty
    long = long[long["variable"].isin(_QNA_LABOR_VARS)]
    long = long[pd.to_numeric(long["value"], errors="coerce") > 0]
    if long.empty:
        return empty
    out = pd.DataFrame({
        "code": long["code"].to_numpy(),
        "date": pd.to_datetime(long["date"]).to_numpy(),
        "variable": long["variable"].map(_QNA_LABOR_VARS).to_numpy(),
        "value": np.log(long["value"].astype(float).to_numpy()),
        # per series, not per country: a reference area adjusted at source for
        # heads but not hours gets an honest label on each.
        "sa_source": [_QNA_SA_SOURCE.get(v, "none") for v in long["sa_source"]],
        "source": [f"{_QNA_LABOR_SOURCE}:{v}" for v in long["variable"]],
    })
    return out[schema_cols].reset_index(drop=True)


def build_all(
    countries: Iterable[str] | None = None,
    *,
    fast: bool = False,
    refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # countries=None means "all countries any fetcher returns" (sparse
    # panel). fast=True keeps the legacy 6-country filter for quick local
    # builds. An explicit countries=[...] forces a subset.
    if countries is None and fast:
        countries = list(FAST_COUNTRIES)
    elif countries is not None:
        countries = list(countries)
    # else: countries stays None → no filter
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    frames_all = []

    # WUI (has two fetchers)
    try:
        frames_all.append(wui.fetch_q(refresh=refresh))
    except Exception as e:
        print(f"[build] WUI_Q failed: {e}")
    try:
        frames_all.append(wui.fetch_m(refresh=refresh))
    except Exception as e:
        print(f"[build] WUI_M failed: {e}")
    # WUI extras (WTUI / WPUI — same xlsx, different tabs)
    try:
        frames_all.append(wui_extras.fetch_extras_m())
    except Exception as e:
        print(f"[build] WUI extras failed: {e}")

    # Single-entry fetchers
    for mod, label in [(epu, "EPU"), (gpr, "GPR"), (hrs_mpu, "HRS-MPU")]:
        try:
            frames_all.append(mod.fetch(refresh=refresh))
        except Exception as e:
            print(f"[build] {label} failed: {e}")

    if countries is None or "USA" in countries:
        for mod, label in [(jln, "JLN"), (lmn, "LMN"), (fernald, "Fernald")]:
            try:
                frames_all.append(mod.fetch(refresh=refresh))
            except Exception as e:
                print(f"[build] {label} failed: {e}")
        try:
            frames_all.append(fred.fetch_all_us(refresh=refresh))
        except Exception as e:
            print(f"[build] FRED failed: {e}")

    ifs_frame = pd.DataFrame(columns=["code", "date", "variable", "value", "sa_source", "source"])
    oecd_codes = None if countries is None else [c for c in countries if c != "USA"]
    if oecd_codes is None or oecd_codes:
        try:
            frames_all.append(oecd.fetch_qna_expenditure(oecd_codes))
        except Exception as e:
            print(f"[build] OECD QNA failed: {e}")
        try:
            frames_all.append(oecd.fetch_labor_monthly(oecd_codes))
        except Exception as e:
            print(f"[build] OECD labor failed: {e}")
        try:
            frames_all.append(oecd_mei.fetch(codes=oecd_codes))
        except Exception as e:
            print(f"[build] OECD MEI failed: {e}")
        try:
            ifs_frame = imf_ifs.fetch(codes=oecd_codes)
        except Exception as e:
            print(f"[build] IMF IFS failed: {e}")

    labor_local = pd.DataFrame(columns=["code", "date", "variable", "value", "sa_source", "source"])
    try:
        labor_local = oecd_qna_local.fetch_extended()
        frames_all.append(labor_local)
    except Exception as e:
        print(f"[build] OECD QNA local failed: {e}")

    # --- OECD QNA SDMX fills expenditure variables (real GDP / GFCF /
    # consumption / govt consumption / exports / imports) for countries
    # missing from the local Volatility/QNA.xlsx workbook (which only covers
    # ~12 OECD-core countries). Mirrors the labor-SDMX gap-fill that follows.
    if oecd_codes is None or oecd_codes:
        try:
            local_exp_vars = ["log_gdp_real", "log_gfcf_real", "log_con_real",
                              "log_govcon_real", "log_exports_real", "log_imports_real"]
            local_exp_pairs: set[tuple[str, str]] = set()
            for v in local_exp_vars:
                for c in labor_local.loc[labor_local["variable"] == v, "code"].unique():
                    local_exp_pairs.add((c, v))
            sdmx_exp_codes = oecd_codes
            if sdmx_exp_codes is not None:
                # Only ask about countries missing at least one expenditure variable.
                sdmx_exp_codes = [
                    c for c in sdmx_exp_codes
                    if any((c, v) not in local_exp_pairs for v in local_exp_vars)
                ]
            if sdmx_exp_codes is None or sdmx_exp_codes:
                if sdmx_exp_codes:
                    print(f"[build] fetching OECD QNA SDMX expenditure for: {sdmx_exp_codes}")
                exp_sdmx = oecd_qna_expenditure.fetch_qna_expenditure(codes=sdmx_exp_codes)
                if not exp_sdmx.empty:
                    keep_mask = [
                        (c, v) not in local_exp_pairs
                        for c, v in zip(exp_sdmx["code"], exp_sdmx["variable"])
                    ]
                    exp_sdmx = exp_sdmx[keep_mask]
                    if not exp_sdmx.empty:
                        frames_all.append(exp_sdmx)
        except Exception as e:
            print(f"[build] OECD QNA SDMX expenditure failed: {e}")

    # --- OECD QNA SDMX fills labor variables for countries missing from the
    # local workbook (typically CAN/MEX hours, KOR employment). This goes
    # through ``qna_labor`` rather than the retired
    # ``oecd_qna_labor.fetch_qna_labor``: same two totals under the same two
    # variable names, but seasonally adjusted here for the reference areas
    # that publish the labour block raw — and without dragging the
    # expenditure block along, see ``_fetch_qna_labor_logs`` for why.
    if oecd_codes is None or oecd_codes:
        try:
            local_emp_codes = set(
                labor_local.loc[labor_local["variable"] == "log_emp_qna", "code"].unique()
            )
            local_hrs_codes = set(
                labor_local.loc[labor_local["variable"] == "log_hours_qna", "code"].unique()
            )
            sdmx_codes = oecd_codes
            if sdmx_codes is not None:
                # Only ask SDMX about countries missing at least one labor variable.
                sdmx_codes = [c for c in sdmx_codes
                              if c not in local_emp_codes or c not in local_hrs_codes]
            if sdmx_codes is None or sdmx_codes:
                if sdmx_codes:
                    print(f"[build] fetching OECD QNA SDMX labor for: {sdmx_codes}")
                labor_sdmx = _fetch_qna_labor_logs(sdmx_codes)
                if not labor_sdmx.empty:
                    # Drop SDMX rows for (code, variable) pairs the local workbook
                    # already covers — local wins, SDMX only fills the gaps.
                    local_pairs = set(
                        map(tuple, labor_local[["code", "variable"]].to_numpy().tolist())
                    )
                    keep_mask = [
                        (c, v) not in local_pairs
                        for c, v in zip(labor_sdmx["code"], labor_sdmx["variable"])
                    ]
                    labor_sdmx = labor_sdmx[keep_mask]
                    if not labor_sdmx.empty:
                        frames_all.append(labor_sdmx)
        except Exception as e:
            print(f"[build] OECD QNA SDMX labor failed: {e}")

    try:
        frames_all.append(yahoo.fetch_realized_vol(countries, freq="M"))
    except Exception as e:
        print(f"[build] Yahoo RV M failed: {e}")
    try:
        frames_all.append(yahoo.fetch_realized_vol(countries, freq="Q"))
    except Exception as e:
        print(f"[build] Yahoo RV Q failed: {e}")

    # --- Piece 5 (energy + FX) ---
    # Pink Sheet (global benchmarks under code='WLD'); OECD energy CPI per
    # country (CP045); OECD nominal LCU/USD. Energy CPI / FX use the full
    # country list (USA included — both are published OECD series for the
    # US) rather than the OECD-minus-USA list used by QNA.
    pink_frame = pd.DataFrame(columns=["code", "date", "variable", "value", "sa_source", "source"])
    try:
        pink_frame = wb_pink_sheet.fetch(refresh=refresh)
    except Exception as e:
        print(f"[build] WB Pink Sheet failed: {e}")
    energy_codes = None if countries is None else list(countries)
    if energy_codes is None or energy_codes:
        try:
            frames_all.append(oecd_energy.fetch_energy_cpi(codes=energy_codes))
        except Exception as e:
            print(f"[build] OECD energy CPI failed: {e}")
        try:
            frames_all.append(oecd_fx.fetch_xrate_monthly(codes=energy_codes))
        except Exception as e:
            print(f"[build] OECD FX failed: {e}")

    frames_all = [f for f in frames_all if f is not None and not f.empty]
    panel_all = merge_frames(frames_all)
    panel_all = _drop_aggregates_in_panel(panel_all)
    # ------------------------------------------------------------------
    # Country filter — split into two stages so the uncertainty composite
    # is built on the full WUI/EPU/GPR universe (~143 countries from WUI
    # alone), not the small `countries` roster. Real-economy variables
    # still respect the requested country list.
    #
    # Pink Sheet (code='WLD') is held aside (built into pink_frame above):
    # ``_drop_aggregates`` and ``_validate_panel`` reject the WLD code by
    # design, so we merge the global benchmarks directly into the M/Q
    # panels after validation (further down in this function).
    # ------------------------------------------------------------------
    UNC_PROXY_VARS = set(UNCERTAINTY_PROXIES_Q) | set(UNCERTAINTY_PROXIES_M)
    UNC_DERIVED_VARS = {"unc_m", "unc_q", "unc_pc_m", "unc_pc_q",
                        "unc_innov_m", "unc_innov_q"}
    UNC_KEEP_VARS = UNC_PROXY_VARS | UNC_DERIVED_VARS

    if countries is not None:
        keep_mask = (panel_all["code"].isin(list(countries))
                     | panel_all["variable"].isin(UNC_PROXY_VARS))
        panel_all = panel_all[keep_mask].reset_index(drop=True)
    # else: countries is None → no filter, keep everything.

    # Gap-fill with IMF IFS for countries/dates not covered by OECD.
    panel_all = _gap_fill_ifs(panel_all, ifs_frame, oecd_var="log_cpi", ifs_var="cpi_ifs_m")
    panel_all = _gap_fill_ifs(panel_all, ifs_frame, oecd_var="log_ip", ifs_var="ip_ifs_m")
    panel_all = _gap_fill_ifs(panel_all, ifs_frame, oecd_var="urate", ifs_var="urate_ifs_m")
    panel_all = panel_all.drop_duplicates(subset=["code", "date", "variable"], keep="first")

    # Centralized X-13 ARIMA-SEATS pass over NSA proxies and fallback CPI.
    panel_all = _apply_x13_to_panel(panel_all)

    # Build GPR-based backbone (unc_m), PC1 composite (unc_pc_m), and AR(2)
    # innovation (unc_innov_m). These run on the full uncertainty-proxy
    # universe (no country filter) — see UNC_PROXY_VARS exemption above.
    panel_all = build_backbone(panel_all)
    panel_all = build_composite(panel_all)
    panel_all = build_innovation(panel_all, freq="m")

    # Derive quarterly uncertainty from already-SA monthly proxies (mean within quarter).
    for m_name, q_name in (("epu_m", "epu_q"), ("gpr_m", "gpr_q"),
                            ("wtui_m", "wtui_q"), ("wpui_m", "wpui_q"),
                            ("mpu_m", "mpu_q"),
                            ("unc_m", "unc_q"), ("unc_pc_m", "unc_pc_q")):
        if (panel_all["variable"] == m_name).any():
            qfm = _quarterly_from_monthly(panel_all[panel_all["variable"] == m_name], q_name)
            # sa_source carries through from the monthly source via _quarterly_from_monthly.
            panel_all = merge_frames([panel_all, qfm])

    # Quarterly innovation is built from unc_pc_q (resampled above).
    panel_all = build_innovation(panel_all, freq="q")

    # GARCH conditional volatility for each uncertainty proxy at both frequencies.
    garch_frames = []
    for proxy in UNCERTAINTY_PROXIES_Q + UNCERTAINTY_PROXIES_M:
        if (panel_all["variable"] == proxy).any():
            garch_frames.append(compute_garch_sigma(panel_all, proxy))
    panel_all = merge_frames([panel_all] + [f for f in garch_frames if not f.empty])

    # Split Q and M panels by variable name.
    monthly_vars = {v for v in panel_all["variable"].unique()
                    if v.endswith("_m") or v in {
                        "urate", "short_rate", "log_emp", "log_cpi", "log_ip",
                        "log_hours_manuf", "log_ip_capgoods", "vix",
                        "log_nonres_biz_inv_ind", "jln_macro_1", "jln_macro_3",
                        "jln_macro_12", "lmn_real", "lmn_fin",
                    }}
    panel_m = panel_all[panel_all["variable"].isin(monthly_vars)].copy()
    panel_q = panel_all[~panel_all["variable"].isin(monthly_vars)].copy()

    # Add Q resamples of key monthly real variables.
    for var in ("urate", "short_rate", "log_emp", "log_cpi"):
        if (panel_m["variable"] == var).any():
            qfm = _quarterly_from_monthly(panel_m[panel_m["variable"] == var], var)
            panel_q = merge_frames([panel_q, qfm])

    # --- Piece 5: merge global energy benchmarks (WLD) into the monthly
    # panel, then derive country-specific real-local oil from log_cpi +
    # xrate. Q rollups for everything energy-related follow. WLD bypasses
    # ``_validate_panel`` because that runs against the country-only
    # ``panel_all``.
    if not pink_frame.empty:
        panel_m = merge_frames([panel_m, pink_frame])
    deriv_m = _derive_real_local_oil(panel_m)
    if not deriv_m.empty:
        panel_m = merge_frames([panel_m, deriv_m])
    for var in ("oil_brent_m", "oil_wti_m", "oil_dubai_m", "oil_avg_m",
                "gas_eu_m", "gas_us_m", "gas_jp_lng_m",
                "coal_au_m", "coal_za_m",
                "log_cpi_energy_m", "xrate_usd_per_local_m",
                "log_oil_brent_real_local_m"):
        if (panel_m["variable"] == var).any():
            q_var = var[:-2] + "_q"
            qfm = _quarterly_from_monthly(panel_m[panel_m["variable"] == var], q_var)
            panel_q = merge_frames([panel_q, qfm])

    # --- Piece 2: LFS monthly emp -> quarterly mean as a final fallback for
    # log_emp_qna. Only fills countries that don't already have log_emp_qna
    # from either the local Volatility/QNA.xlsx workbook or the OECD QNA SDMX
    # fetcher. Targets AUS / CAN / JPN in the core roster.
    labor_lfs = _resample_monthly_emp_to_quarterly(panel_m)
    if not labor_lfs.empty:
        have_emp = set(panel_q.loc[panel_q["variable"] == "log_emp_qna", "code"].unique())
        labor_lfs = labor_lfs[~labor_lfs["code"].isin(have_emp)]
        if not labor_lfs.empty:
            print(f"[labor] LFS-resampled emp for: {sorted(labor_lfs['code'].unique())}")
            panel_q = merge_frames([panel_q, labor_lfs])

    # Alias log_hours (FRED HOANBS, quarterly NFB hours, USA) -> log_hours_qna
    # for any country where log_hours is present but log_hours_qna is not.
    have_hrs = set(panel_q.loc[panel_q["variable"] == "log_hours_qna", "code"].unique())
    hrs_alias = panel_q[(panel_q["variable"] == "log_hours") & (~panel_q["code"].isin(have_hrs))].copy()
    if not hrs_alias.empty:
        hrs_alias["variable"] = "log_hours_qna"
        hrs_alias["sa_source"] = "aliased_from_log_hours:FRED:HOANBS"
        hrs_alias["source"] = "aliased_from_log_hours:FRED:HOANBS"
        print(f"[labor] aliased log_hours -> log_hours_qna for: {sorted(hrs_alias['code'].unique())}")
        panel_q = merge_frames([panel_q, hrs_alias])

    # Regime column on both panels.
    panel_q["regime"] = regimes.assign_regime(panel_q["date"]).values
    panel_m["regime"] = regimes.assign_regime(panel_m["date"]).values

    # Coverage reports.
    cov_q = build_coverage_report(panel_q)
    cov_m = build_coverage_report(panel_m)

    panel_q.to_parquet(PANEL_Q_PATH)
    panel_m.to_parquet(PANEL_M_PATH)
    cov_q.to_csv(COV_Q_PATH, index=False)
    cov_m.to_csv(COV_M_PATH, index=False)
    var_cov = build_variable_coverage(panel_all)
    var_cov.to_csv(DATA_DIR / "variable_coverage.csv", index=False)
    recipes = build_sample_recipes(panel_all)
    recipes.to_csv(DATA_DIR / "sample_recipes.csv", index=False)

    # Validate the full panel before writing additional artifacts.
    _validate_panel(panel_all)

    audit = sa_audit(panel_all)
    if not audit.empty:
        audit.to_csv(DATA_DIR / "sa_audit.csv", index=False)

    # Per-country uncertainty diagnostics (tier + ADF/KPSS + AR(2) sigma).
    from .uncertainty.diagnostics import build_diagnostics
    diag_q = build_diagnostics(panel_q, freq="q")
    diag_m = build_diagnostics(panel_m, freq="m")
    diag = pd.concat([diag_q, diag_m], ignore_index=True)
    diag_path = DATA_DIR / "uncertainty_diagnostics.csv"
    diag.to_csv(diag_path, index=False)
    print(f"[diagnostics] wrote {len(diag)} rows to {diag_path}")

    return panel_q, panel_m


def build_all_subnational(*, refresh: bool = False, data_dir: str = "data") -> dict:
    """Convenience wrapper matching build_all's signature for subnational panels.

    Returns a dict of artifact paths: {'state', 'county', 'industry'}.
    Requires Bartik artifacts (bartik_shares_1990.parquet and
    industry_sensitivities.parquet) already present in data/processed/;
    run `python tools/build_bartik_artifacts.py` (Task 12) first.
    """
    return _build_subnational(data_dir=data_dir, refresh=refresh)


def load_country(code: str, freq: str = "Q") -> pd.DataFrame:
    path = PANEL_Q_PATH if freq == "Q" else PANEL_M_PATH
    long = pd.read_parquet(path)
    sub = long[long["code"] == code]
    wide = sub.pivot(index="date", columns="variable", values="value").sort_index()
    return wide
