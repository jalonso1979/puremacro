"""Cross-source consensus + disagreement over narrative indices.

Computes at each time t:
  consensus(t)    = mean across z-scored series
  disagreement(t) = std  across z-scored series  (ddof=0)
  n_active(t)     = count of non-null series at t

Operates on a caller-provided dict ``{name: pd.Series}``. The series
can be at heterogeneous native frequencies; the function resamples
to a common grid (monthly by default) via forward-fill within each
series' native period before computing cross-sectional statistics.

The ``GROUPS`` constant documents thematic subsets of the 13+ existing
narrative indices in ``puremacro.narrative.indices``. ``GROUPS['all']``
covers the canonical set.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


# Predefined thematic groups for callers. Each tuple lists the symbol
# names of indices that conceptually belong together. Consumers still
# build the {name: series} dict themselves; GROUPS is reference
# metadata only.
GROUPS: dict[str, tuple[str, ...]] = {
    "macro_uncertainty": ("epu", "mpu", "wui"),
    "labor":             ("lui", "ltui", "lwui"),
    "us_policy":         ("erpui", "sotuui", "cboui", "bbui"),
    "eu_policy":         ("eurlex_ui", "ep_ui"),
    "geopolitical":      ("gpr", "tone"),
    "social":            ("bluesky_ui",),
    "all": (
        "epu", "mpu", "wui",
        "lui", "ltui", "lwui",
        "erpui", "sotuui", "cboui", "bbui",
        "eurlex_ui", "ep_ui",
        "gpr", "tone",
        "bluesky_ui",
    ),
}


def consensus_disagreement(
    series_dict: Mapping[str, pd.Series],
    *,
    freq: str = "ME",
    base_period: tuple[str, str] | None = None,
    min_active: int = 2,
    return_panel: bool = False,
) -> pd.DataFrame:
    """Cross-sectional consensus + disagreement across narrative indices.

    Parameters
    ----------
    series_dict : mapping of ``{name: pd.Series}``. Each series must have
        a DatetimeIndex (or coercible). Empty mappings raise ValueError.
    freq : pandas frequency string for the common output grid; default
        ``"ME"`` (month-end). Series at coarser native frequencies are
        forward-filled within their period to populate the target grid.
    base_period : optional ``(start_iso, end_iso)`` window used to
        compute the per-series z-score mean and std. ``None`` uses the
        full series for the z-score reference.
    min_active : drop rows where ``n_active < min_active`` (default 2 —
        std across <2 series is degenerate).
    return_panel : if True, the returned DataFrame also includes one
        column per input series with its z-scored values, alongside
        ``consensus``, ``disagreement``, ``n_active``.

    Returns
    -------
    pd.DataFrame indexed by the target-grid dates, with at minimum
    columns ``[consensus, disagreement, n_active]``.
    """
    if not series_dict:
        raise ValueError("series_dict is empty")

    # Step 1: resample each series to the target grid via per-period
    # forward-fill (handles quarterly/annual native frequency).
    aligned: dict[str, pd.Series] = {}
    for name, s in series_dict.items():
        if not isinstance(s, pd.Series):
            raise ValueError(f"series_dict[{name!r}] is not a pd.Series")
        s2 = s.copy()
        s2.index = pd.to_datetime(s2.index)
        s2 = s2.sort_index()
        # Forward-fill within native period, then resample to target.
        # Daily-or-finer: take period mean.
        s2_resampled = s2.resample(freq).mean()
        # If the series is coarser than freq, gaps in resample will be
        # NaN; forward-fill those within the calendar year as a
        # reasonable proxy for "still in effect".
        s2_resampled = s2_resampled.ffill(limit=12)
        aligned[name] = s2_resampled

    # Step 2: z-score each series (per base_period if given).
    zscored: dict[str, pd.Series] = {}
    for name, s in aligned.items():
        if base_period is None:
            ref = s.dropna()
        else:
            start, end = base_period
            ref = s.loc[pd.to_datetime(start):pd.to_datetime(end)].dropna()
        if ref.empty or ref.std(ddof=0) == 0:
            # Degenerate reference: keep as NaN to drop from cross-section.
            zscored[name] = pd.Series(np.nan, index=s.index)
            continue
        zscored[name] = (s - ref.mean()) / ref.std(ddof=0)

    # Step 3: assemble panel and compute cross-sectional stats.
    panel = pd.DataFrame(zscored)
    consensus = panel.mean(axis=1, skipna=True)
    disagreement = panel.std(axis=1, ddof=0, skipna=True)
    n_active = panel.notna().sum(axis=1)

    out = pd.DataFrame({
        "consensus": consensus,
        "disagreement": disagreement,
        "n_active": n_active,
    })
    out = out[out["n_active"] >= min_active].copy()

    if return_panel:
        # Add per-series z-scored columns
        out = pd.concat([out, panel.loc[out.index]], axis=1)

    return out


__all__ = ["consensus_disagreement", "GROUPS"]
