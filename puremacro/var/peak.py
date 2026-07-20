"""Cross-country peak / horizon-fixed IRF summaries.

`peak_summary` returns one row per country with the peak (max-|·|) IRF
horizon and value, point-wise IRF values at h ∈ {4, 8, 16}, the cumulative
IRF at h=16 with bootstrap bounds, and the bootstrap n_obs. `peak_distribution`
is a thin projection that exposes only the columns needed for cross-country
distribution comparisons (peak, peak_h, accum at a chosen horizon, n_obs).

Originally lifted from puremacro.teaching.svar_panel.peak_summary; that
location is now a back-compat re-export from here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def peak_summary(
    results: dict[str, dict],
    *,
    shock_idx: int,
    response_idx: int,
    scale: float = 100.0,
    cumulative: bool = False,
    horizons_for_peak: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """For each country, find the peak (max-|·|) IRF response within the
    chosen horizon range and the IRF at a few key horizons. Returns a
    DataFrame indexed by `code` with columns:
        peak_h, peak, peak_lo, peak_hi,
        irf_4, irf_8, irf_16,
        accum_h16, accum_h16_lo, accum_h16_hi, n_obs.

    `peak_lo` / `peak_hi` are the bootstrap CI bounds of the IRF AT the
    peak horizon — they are NOT a CI on argmax (which has a non-standard
    sampling distribution). Use them to draw error bars in forest plots.
    """
    rows = []
    for code, r in results.items():
        ir = r["point"][:, response_idx, shock_idx] * scale
        ir_lo = r["lo"][:, response_idx, shock_idx] * scale
        ir_hi = r["hi"][:, response_idx, shock_idx] * scale
        if cumulative:
            ir = np.cumsum(ir)
            ir_lo = np.cumsum(ir_lo)
            ir_hi = np.cumsum(ir_hi)
        H = len(ir)
        if horizons_for_peak is None:
            lo, hi = 0, H
        else:
            lo, hi = horizons_for_peak
        peak_h = lo + int(np.argmax(np.abs(ir[lo:hi])))
        # Accumulated h=16 from the raw point/lo/hi (cumsum of period IRF).
        if H > 16:
            ac_pt = float(np.cumsum(r["point"][:, response_idx, shock_idx] * scale)[16])
            ac_lo = float(np.cumsum(r["lo"][:, response_idx, shock_idx] * scale)[16])
            ac_hi = float(np.cumsum(r["hi"][:, response_idx, shock_idx] * scale)[16])
        else:
            ac_pt = ac_lo = ac_hi = np.nan
        rows.append({
            "code":         code,
            "peak_h":       int(peak_h),
            "peak":         float(ir[peak_h]),
            "peak_lo":      float(ir_lo[peak_h]),
            "peak_hi":      float(ir_hi[peak_h]),
            "irf_4":        float(ir[4])  if H > 4  else np.nan,
            "irf_8":        float(ir[8])  if H > 8  else np.nan,
            "irf_16":       float(ir[16]) if H > 16 else np.nan,
            "accum_h16":    ac_pt,
            "accum_h16_lo": ac_lo,
            "accum_h16_hi": ac_hi,
            "n_obs":        int(r["n_obs"]),
        })
    return pd.DataFrame(rows).set_index("code") if rows else pd.DataFrame()


def peak_distribution(
    results: dict[str, dict],
    *,
    shock_idx: int,
    response_idx: int,
    scale: float = 100.0,
    cumulative: bool = True,
    h_fixed: int = 16,
) -> pd.DataFrame:
    """Slim per-country distribution DataFrame.

    Columns: ``peak``, ``peak_h``, ``accum`` (cumulative IRF at ``h_fixed``),
    ``h_fixed`` (carried as a column so consumers don't have to guess), and
    ``n_obs``. One row per country in ``results``; index is RangeIndex.

    Implementation: when ``h_fixed == 16`` we delegate to ``peak_summary`` and
    rename ``accum_h16 -> accum``. Otherwise we recompute the cumulative IRF at
    ``h_fixed`` directly to avoid relying on the hard-coded h=16 in
    ``peak_summary``.
    """
    if not results:
        return pd.DataFrame(columns=["peak", "peak_h", "accum", "h_fixed", "n_obs"])

    summary = peak_summary(
        results, shock_idx=shock_idx, response_idx=response_idx,
        scale=scale, cumulative=cumulative,
    )

    if h_fixed == 16:
        out = summary[["peak", "peak_h", "accum_h16", "n_obs"]].rename(
            columns={"accum_h16": "accum"}
        ).copy()
    else:
        accum = []
        for code in summary.index:
            r = results[code]
            ir = r["point"][:, response_idx, shock_idx] * scale
            cs = np.cumsum(ir)
            accum.append(float(cs[h_fixed]) if h_fixed < len(cs) else np.nan)
        out = summary[["peak", "peak_h", "n_obs"]].copy()
        out["accum"] = accum

    out["h_fixed"] = int(h_fixed)
    out = out.reset_index(drop=True)
    return out[["peak", "peak_h", "accum", "h_fixed", "n_obs"]]
