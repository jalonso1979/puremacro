#!/usr/bin/env python
"""Main Street Uncertainty — Phase 7: sentiment vs uncertainty discriminant.

Two referees charged that the BBUI, as implemented, is a labor *negative-
sentiment* index rather than an *uncertainty* index: of the 49 tone terms
only ~11 are genuine-uncertainty words (uncertain, risk, volatile,
downside), the other ~33 are directional-negative (decline, weak, soften,
deteriorate, loss, slowdown ...), and the 130 phrases mix tightness
(labor shortage, war for talent) with weakness (layoffs, rising
unemployment). This phase decomposes the published index along that seam
and asks which component actually carries the exposure-weighted labor
signal.

Three indices are scored on the SAME 1970-2025 corpus, differing ONLY in
the tone word list (labor_domain gate unchanged):

    full   : the published LUI lexicon (uncertainty_tone = all 49 terms,
             130 phrases).                    == data/processed/bbui_district_panel.csv
    unc    : uncertainty_tone restricted to the 11 genuine-uncertainty
             terms, phrases dropped -- pure uncertainty vocabulary.
    sent   : uncertainty_tone restricted to the 33 directional-negative
             terms, phrases kept -- the negative-sentiment residual.

Each is collapsed to a district-quarter panel (normalize='raw'), turned
into a per-district AR(2) innovation z-scored over its own history via
``derive_innovation_shock`` -- the SAME recipe Phase 2 used to build
``bbui_innov`` -- and merged onto the frozen exposure/outcome panel. The
full rebuild is gated against the frozen ``bbui_innov`` to 1e-6 before
anything new runs, so the scoring path is validated.

Estimand (identical estimator to Phases 3-4, imported verbatim):

    y_{s,t+h} - y_{s,t-1} = b_h (expo_s x shock_{d(s),t}) + state FE
                          + district-x-quarter FE + 4 lags + e

run for shock in {full, unc, sent}, plus the head-to-head horse race
expo x unc vs expo x sent (the two are near-orthogonal at the district-
quarter level, r ~ 0.09, so the design CAN separate them). If the
uncertainty-only exposure differential vanishes while the sentiment one
survives, the paper's "uncertainty" reading is not supported and the
measure must be relabeled; if uncertainty-only survives, a genuine
uncertainty signal sits under the sentiment.

Scoring is slow (~4 min/lexicon); scored panels are cached to
``output/phase7_index_*.csv`` and reused unless --rescore is passed.

Inputs (frozen; corpus via the HTTP cache, no live network):
    (corpus)  puremacro.narrative iter_beige_book 1970-2025
    output/state_district_panel_quarterly.csv   panel incl. bbui_innov
    output/exposure_state.csv                   frozen 1990-91 shares
    data/processed/bbui_district_panel.csv       full-lexicon replication ref

Outputs (docs/research/main_street_uncertainty/output/):
    phase7_index_full.csv / _unc.csv / _sent.csv   district-quarter panels
    phase7_innovations.csv        district x quarter full/unc/sent innovations
    phase7_discriminant.json      district-quarter correlations among indices
    irf_phase7_sentiment.csv      all exposure LP paths + DK + WC p
    phase7_manifest.json, phase7_summary.json, run_log_phase7.txt
"""
from __future__ import annotations

import argparse
import copy
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

import run_main_street_phase3 as p3  # noqa: E402  (estimator core, reused)
import run_main_street_phase4 as p4  # noqa: E402  (_design_multi, run_spec_multi)
from puremacro.narrative.indices.beige_book import bbui  # noqa: E402
from puremacro.narrative.indices._lexicons import LEXICONS  # noqa: E402
from puremacro.narrative.sources.beige_book import iter_beige_book  # noqa: E402
from puremacro.uncertainty.lp_long import derive_innovation_shock  # noqa: E402

OUT_DIR = p3.OUT_DIR
MERGED_CSV = p3.MERGED_CSV
EXPO_CSV = OUT_DIR / "exposure_state.csv"
FULL_REF_CSV = REPO_ROOT / "data" / "processed" / "bbui_district_panel.csv"

HORIZONS = p3.HORIZONS
N_LAGS = p3.N_LAGS
BASELINE_START = p3.BASELINE_START

# The 11 genuine-uncertainty terms among the 49 LUI tone words. Everything
# else in uncertainty_tone is directional-negative sentiment.
UNCERT_TERMS = {
    "uncertain", "uncertainties", "uncertainty", "risk", "risks", "risky",
    "volatile", "volatility", "downside", "downside risk", "downside risks",
}

_T0 = time.time()
_LOG_LINES: list[str] = []


def log(msg: str) -> None:
    line = f"[{time.time() - _T0:7.1f}s] {msg}"
    print(line, flush=True)
    _LOG_LINES.append(line)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Lexicon variants
# ---------------------------------------------------------------------------

def lexicon_variants() -> dict[str, dict]:
    base = LEXICONS["lui"]["en"]
    tone = set(base["uncertainty_tone"])
    unc = copy.deepcopy(base)
    unc["uncertainty_tone"] = sorted(tone & UNCERT_TERMS)
    unc["phrases"] = []                       # pure uncertainty vocabulary
    sent = copy.deepcopy(base)
    sent["uncertainty_tone"] = sorted(tone - UNCERT_TERMS)   # directional-neg
    return {"full": base, "unc": unc, "sent": sent}


# ---------------------------------------------------------------------------
# Scoring (cached)
# ---------------------------------------------------------------------------

def score_panels(rescore: bool) -> dict[str, pd.DataFrame]:
    variants = lexicon_variants()
    panels: dict[str, pd.DataFrame] = {}
    need = rescore or any(
        not (OUT_DIR / f"phase7_index_{k}.csv").exists() for k in variants)
    if not need:
        for k in variants:
            panels[k] = pd.read_csv(OUT_DIR / f"phase7_index_{k}.csv",
                                    parse_dates=["date"])
        log("scored panels loaded from cache")
        return panels

    log("scoring corpus 1970-2025 (iter_beige_book) ...")
    recs = list(iter_beige_book(since="1970-01-01", until="2025-12-31"))
    log(f"corpus: {len(recs)} records")
    for k, lx in variants.items():
        df = bbui(recs, level="district", output="tidy", normalize="raw",
                  lexicon=lx)
        df = df[df["district"] != "National"].copy()
        df["date"] = pd.to_datetime(df["date"])
        df.to_csv(OUT_DIR / f"phase7_index_{k}.csv", index=False)
        panels[k] = df
        log(f"scored [{k}]: {len(df)} district-quarters, "
            f"n tone terms={len(lx['uncertainty_tone'])}, "
            f"n phrases={len(lx['phrases'])}")
    return panels


# ---------------------------------------------------------------------------
# Innovations (Phase-2 recipe, verbatim)
# ---------------------------------------------------------------------------

def build_innovation(panel: pd.DataFrame, name: str) -> pd.DataFrame:
    """Per-district AR(2) innovation, z-scored over own history -- exactly
    ``build_merged_panel`` in Phase 2, one index at a time."""
    dist = panel[panel["n_docs"] > 0].copy()
    long = dist.rename(columns={"district": "code"})[["code", "date", "value"]]
    long["variable"] = "idx_q"
    sh = derive_innovation_shock(long, "idx_q")
    sh = sh.rename(columns={"code": "district", "value": name})
    return sh[["district", "date", name]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--wc-draws", type=int, default=999)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rescore", action="store_true",
                    help="force re-scoring the corpus (ignore cached panels)")
    ap.add_argument("--fast", action="store_true",
                    help="quick mode: --wc-draws 199")
    args = ap.parse_args(argv)
    B = 199 if args.fast else args.wc_draws

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # --- score + validate the full rebuild --------------------------------
    panels = score_panels(args.rescore)
    ref = pd.read_csv(FULL_REF_CSV, parse_dates=["date"])
    ref = ref[ref["district"] != "National"]
    chk = panels["full"].merge(
        ref[["district", "date", "value"]], on=["district", "date"],
        suffixes=("_new", "_ref"))
    gap = float(np.max(np.abs(chk["value_new"] - chk["value_ref"])))
    if gap > 1e-6:
        raise RuntimeError(
            f"phase7: full-lexicon rebuild disagrees with frozen "
            f"bbui_district_panel.csv (max |diff| {gap:.2e} > 1e-6)")
    log(f"full-lexicon scoring gate PASSED (max |diff| vs frozen "
        f"panel {gap:.2e}, {len(chk)} cells)")

    # --- innovations -------------------------------------------------------
    innov = None
    for k in ("full", "unc", "sent"):
        sh = build_innovation(panels[k], f"innov_{k}")
        innov = sh if innov is None else innov.merge(
            sh, on=["district", "date"], how="outer")
    innov = innov.sort_values(["district", "date"]).reset_index(drop=True)
    innov.to_csv(OUT_DIR / "phase7_innovations.csv", index=False)

    # gate: rebuilt full innovation must match frozen bbui_innov
    merged = pd.read_csv(MERGED_CSV, parse_dates=["date"])
    frozen = (merged[["district", "date", "bbui_innov"]]
              .drop_duplicates(["district", "date"]))
    ig = innov.merge(frozen, on=["district", "date"], how="inner").dropna(
        subset=["innov_full", "bbui_innov"])
    igap = float(np.max(np.abs(ig["innov_full"] - ig["bbui_innov"])))
    log(f"innovation gate: rebuilt full vs frozen bbui_innov max |diff| "
        f"{igap:.2e} over {len(ig)} cells "
        f"({'PASS' if igap < 1e-6 else 'INFO — build differs'})")

    # --- district-quarter discriminant correlations ------------------------
    est = innov[innov["date"] >= BASELINE_START]
    disc = {
        "corr_full_sent": float(est["innov_full"].corr(est["innov_sent"])),
        "corr_full_unc": float(est["innov_full"].corr(est["innov_unc"])),
        "corr_unc_sent": float(est["innov_unc"].corr(est["innov_sent"])),
        "n_cells": int(len(est.dropna(
            subset=["innov_full", "innov_unc", "innov_sent"]))),
    }
    (OUT_DIR / "phase7_discriminant.json").write_text(json.dumps(disc, indent=2), encoding="utf-8")
    log(f"innovation discriminant (1992+): corr(full,sent)="
        f"{disc['corr_full_sent']:+.3f}, corr(full,unc)="
        f"{disc['corr_full_unc']:+.3f}, corr(unc,sent)="
        f"{disc['corr_unc_sent']:+.3f}")

    # --- exposure panel (replicates Phase-4 base_mfg) ----------------------
    expo = pd.read_csv(EXPO_CSV)
    base = merged.merge(expo[["state", "mfg_z"]], on="state")
    base = (base.rename(columns={"mfg_z": "expo_z"})
            [["state", "district", "date", "urate", "emp100", "expo_z"]]
            .dropna(subset=["expo_z"]))
    base = base[base["date"] >= BASELINE_START]
    base = base.merge(innov[["district", "date", "innov_full",
                             "innov_unc", "innov_sent"]],
                      on=["district", "date"], how="left")
    n_missing = int(base[["innov_full", "innov_unc", "innov_sent"]]
                    .isna().any(axis=1).sum())
    log(f"exposure panel: {len(base)} rows, {n_missing} with a missing shock")

    wc = rng.choice([-1.0, 1.0], size=(B, 12))

    frames = []
    for k in ("full", "unc", "sent"):
        frames.append(p4.run_spec_multi(
            base.dropna(subset=[f"innov_{k}"]),
            spec=f"urate_{k}", outcome="urate",
            shock_cols=[f"innov_{k}"], horizons=HORIZONS, wc_draws=wc))
    # head-to-head: uncertainty vs sentiment (focal = each in turn)
    hh = base.dropna(subset=["innov_unc", "innov_sent"])
    frames.append(p4.run_spec_multi(
        hh, spec="urate_hr_unc", outcome="urate",
        shock_cols=["innov_unc", "innov_sent"], horizons=HORIZONS, wc_draws=wc))
    frames.append(p4.run_spec_multi(
        hh, spec="urate_hr_sent", outcome="urate",
        shock_cols=["innov_sent", "innov_unc"], horizons=HORIZONS, wc_draws=wc))
    irf = pd.concat(frames, ignore_index=True)
    irf.to_csv(OUT_DIR / "irf_phase7_sentiment.csv", index=False)

    def peak(spec: str) -> dict:
        s = irf[irf["spec"] == spec].set_index("h")
        h = int(s["beta"].abs().idxmax())
        return {"h": h, "beta": round(float(s.loc[h, "beta"]), 4),
                "se_dk": round(float(s.loc[h, "se_dk"]), 4),
                "p_wc": round(float(s.loc[h, "p_wc"]), 3)}

    summary = {
        "mode": "fast" if args.fast else "full",
        "discriminant_innovations": disc,
        "scoring_gate_max_absdiff": gap,
        "full_innovation_gate_max_absdiff": igap,
        "peaks": {s: peak(s) for s in irf["spec"].unique()},
        "wc_draws": B, "seed": args.seed,
        "runtime_seconds": round(time.time() - _T0, 1),
    }
    (OUT_DIR / "phase7_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "state_district_panel_quarterly.csv": {"sha256": sha256_file(MERGED_CSV)},
            "exposure_state.csv": {"sha256": sha256_file(EXPO_CSV)},
            "bbui_district_panel.csv": {"sha256": sha256_file(FULL_REF_CSV)},
        },
        "uncertainty_terms": sorted(UNCERT_TERMS),
        "estimator": "identical to Phase 3/4 (imported run_spec_multi)",
        "wc_draws": B, "seed": args.seed,
    }
    (OUT_DIR / "phase7_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for s in irf["spec"].unique():
        p = peak(s)
        log(f"peak[{s}]: h={p['h']} beta={p['beta']:+.4f} "
            f"(DK se {p['se_dk']:.4f}, WC p={p['p_wc']:.3f})")

    (OUT_DIR / "run_log_phase7.txt").write_text("\n".join(_LOG_LINES) + "\n", encoding="utf-8")
    log("summary written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
