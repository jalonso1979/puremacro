"""Precompute the REAL EU-KLEMS KORV moment panel for course module 21.

Replicates the canonical KORV moment pipeline (the "Strategy B" system-GMM
inputs used by tools/_archive/make_notebook_T_korv_capital_skill.py) starting
from ``puremacro.klems.load_klems_panel`` on the real EU-KLEMS 2023 cache,
and writes a small CSV with only the moment columns + code + year so the
course notebook ``21_ia_korv_es.py`` can run OFFLINE (no network, no heavy
KLEMS cache) in the browser/iPad.

Moment columns consumed by ``puremacro.korv_gmm.fit_korv_pooled``:

    dlog_ls_lu     Δlog(L_s / L_u)      skilled/unskilled quantity ratio
    dlog_ws_wu     Δlog(w_s / w_u)      skilled/unskilled wage ratio
    dlog_ke_lu     Δlog(K_e / L_u)      equipment/unskilled ratio (I_e headline)
    dlog_wu_pk     Δlog(w_u / P_K)      unskilled wage vs equipment price
    dlog_lsushare  Δlog(LS_u)           unskilled share of labor compensation
    dlog_pk_pc     Δlog(P_K / P_C)      equipment relative price (the ΔQ driver)

Supply-instrument column consumed by ``puremacro.korv_gmm.fit_sigma_su_pooled``
(the Katz–Murphy 2SLS route for σ_su, iv_col='dlog_ns_nu'):

    dlog_ns_nu     Δlog(N_s / N_u)_{t-1}  predetermined growth of the relative
                                          skilled/unskilled labor SUPPLY in
                                          quantity (headcount) — the Katz–Murphy
                                          supply shifter that instruments the
                                          endogenous relative wage in m1.

Skill mapping (canonical): unskilled = LOW (ISCED 0-2);
skilled = MED + HIGH (ISCED 3-8). Wages are compensation / hours per skill.
Equipment price P_K is the EU-KLEMS value-weighted equipment price index
``p_equip_index``; the consumption price P_C is the OECD-QNA consumption
deflator (Q4 of each year) from ``data/processed/panel_Q.parquet``.

Run:  python tools_curso_korv_build.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from puremacro.klems import load_klems_panel
from puremacro.korv_gmm import fit_korv_pooled, fit_sigma_su_pooled

ROOT = Path(__file__).resolve().parent            # .../uncertainty_examples/puremacro
KLEMS_CACHE = ROOT.parent / "data" / "raw" / "euklems"
PANEL_Q = ROOT.parent / "data" / "processed" / "panel_Q.parquet"
OUT_CSV = ROOT / "notebooks" / "course" / "data" / "korv_panel_klems.csv"

MOMENT_COLS = [
    "dlog_ls_lu", "dlog_ws_wu", "dlog_ke_lu",
    "dlog_wu_pk", "dlog_lsushare", "dlog_pk_pc",
]


def _load_pc_annual() -> pd.DataFrame:
    """Annual (Q4) log consumption price P_C per country from panel_Q."""
    pq = pd.read_parquet(PANEL_Q)
    pc = pq[pq["variable"] == "log_conp_deflator_q"].copy()
    if pc.empty:
        pc = pq[pq["variable"] == "log_gdp_deflator_q"].copy()
    pc["year"] = pd.to_datetime(pc["date"]).dt.year
    pc["quarter"] = pd.to_datetime(pc["date"]).dt.quarter
    return (pc[pc["quarter"] == 4]
            .rename(columns={"value": "log_p_c"})[["code", "year", "log_p_c"]])


def build_moments(*, equip_col: str = "i_equip") -> pd.DataFrame:
    """Build the KORV moment panel from real EU-KLEMS.

    ``equip_col='i_equip'`` uses gross equipment investment flows in m2
    (the archived headline spec); ``'k_equip'`` uses the equipment stock.
    """
    k = load_klems_panel(cache_dir=KLEMS_CACHE, industry="TOT",
                         include_investment=True).copy()
    # Skill aggregation: unskilled = LOW, skilled = MED + HIGH.
    k["l_u"] = k["hours_low"]
    k["l_s"] = k["hours_med"] + k["hours_high"]
    k["w_u"] = k["lab_low"] / k["hours_low"].replace(0, np.nan)
    k["w_s"] = (k["lab_med"] + k["lab_high"]) / k["l_s"].replace(0, np.nan)
    k["ls_u_share"] = k["lab_low"] / k["comp_total"].replace(0, np.nan)
    k["k_e"] = k[equip_col]
    k["p_k"] = k["p_equip_index"]

    df = k.merge(_load_pc_annual(), on=["code", "year"], how="left")
    df["log_pk_pc"] = np.log(df["p_k"].clip(lower=1e-9)) - df["log_p_c"]
    df = df.sort_values(["code", "year"])

    ratios = {
        "dlog_ls_lu":    np.log(df["l_s"].clip(lower=1e-9) / df["l_u"].clip(lower=1e-9)),
        "dlog_ws_wu":    np.log(df["w_s"].clip(lower=1e-9) / df["w_u"].clip(lower=1e-9)),
        "dlog_ke_lu":    np.log(df["k_e"].clip(lower=1e-9) / df["l_u"].clip(lower=1e-9)),
        "dlog_wu_pk":    np.log(df["w_u"].clip(lower=1e-9) / df["p_k"].clip(lower=1e-9)),
        "dlog_lsushare": np.log(df["ls_u_share"].clip(lower=1e-9)),
        "dlog_pk_pc":    df["log_pk_pc"],
    }
    for col, src in ratios.items():
        df[col] = src.groupby(df["code"]).diff()

    # --- Katz–Murphy supply shifter (instrument for the skill FOC m1) --------
    # dlog_ns_nu: growth of the relative skilled/unskilled labor SUPPLY in
    # quantity (headcount), used PREDETERMINED (one-year lag) as the excluded
    # instrument for the endogenous relative wage Δlog(w_s/w_u) in m1.
    #
    # In EU-KLEMS 2023 the skill split is published only as an employment
    # (persons) share ``Share_E`` — the same share that allocates hours — so the
    # headcount relative supply N_s/N_u coincides with the hours ratio L_s/L_u.
    # A CONTEMPORANEOUS supply term would therefore equal the m1 dependent
    # variable (dlog_ls_lu) and the 2SLS would degenerate (instrument = LHS).
    # Lagging one year makes it a genuine, if weak, instrument: predetermined
    # relative supply is exogenous to the contemporaneous relative-demand shock
    # — the identifying assumption of Katz & Murphy (1992). fit_sigma_su_pooled
    # then within-country-demeans it before the 2SLS.
    df["dlog_ns_nu"] = df.groupby("code")["dlog_ls_lu"].shift(1)

    df["k_equip_share_inner"] = 0.5   # neutral share for the Allen sigma_es

    keep = ["code", "year"] + MOMENT_COLS + ["dlog_ns_nu", "k_equip_share_inner"]
    return df.dropna(subset=MOMENT_COLS)[keep].reset_index(drop=True)


def main() -> None:
    panel = build_moments(equip_col="i_equip")     # headline: investment-based m2
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")
    print(f"  {len(panel)} obs, {panel.code.nunique()} countries, "
          f"years {int(panel.year.min())}-{int(panel.year.max())}")
    print(f"  countries: {sorted(panel.code.unique())}")

    fit = fit_korv_pooled(panel)
    print("\nfit_korv_pooled on REAL EU-KLEMS (investment-based m2):")
    print(f"  sigma_su (skilled-unskilled)   = {fit.sigma_su:+.3f} (se {fit.se_sigma_su:.3f})"
          "   [self-instrumented FOC, NOT sharply identified]")
    print(f"  sigma_eu (EQUIPMENT-unskilled)  = {fit.sigma_eu:+.3f} (se {fit.se_sigma_eu:.3f})"
          "   [headline, well-identified]")
    print(f"  sigma_es (equip-skilled, Allen) = {fit.sigma_es:+.3f}"
          "   [depends on sigma_su -> fragile]")
    print(f"  Hansen J = {fit.hansen_J:.2f}  (p = {fit.hansen_p:.4f})   "
          f"n = {fit.n_obs}, countries = {fit.n_country}, converged = {fit.converged}")

    # Katz–Murphy supply-IV route for sigma_su (the m1 FOC that fit_korv_pooled
    # can only self-instrument). Uses the predetermined supply shifter dlog_ns_nu.
    iv = fit_sigma_su_pooled(panel, iv_col="dlog_ns_nu")
    print("\nfit_sigma_su_pooled with Katz–Murphy supply IV (iv_col='dlog_ns_nu'):")
    print(f"  sigma_su (IV, Katz–Murphy)      = {iv.sigma_su:+.3f} (se {iv.se_sigma_su:.3f})   "
          f"[n = {iv.n_obs}, countries = {iv.n_country}, converged = {iv.converged}]")
    print(f"  vs self-instrumented sigma_su   = {fit.sigma_su:+.3f}   "
          f"(change {iv.sigma_su - fit.sigma_su:+.3f})")
    print("  -> supply IV attenuates sigma_su toward 0 but does NOT reach KORV's "
          "strong sigma_su~1.4; still weak / not sharply identified.")

    # Robustness: equipment-stock spec for reference.
    fit_k = fit_korv_pooled(build_moments(equip_col="k_equip"))
    print(f"\n  [robustness, equipment-STOCK m2] sigma_eu = {fit_k.sigma_eu:+.3f} "
          f"(se {fit_k.se_sigma_eu:.3f}), Hansen J = {fit_k.hansen_J:.1f}")
    print("\nKORV (2000) structural benchmark: sigma_es ~ 0.67 < 1 < sigma_eu ~ 1.67.")
    print("Aggregate EU-KLEMS lands sigma_eu near/below Cobb-Douglas -> weaker than KORV.")


if __name__ == "__main__":
    main()
