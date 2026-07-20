"""Cross-country variance decomposition for long-history uncertainty indices.

Two primitives:

* :func:`between_within_share` — total variance of a long-format panel split
  into a "between-time" (cross-section-mean's time variance) and a
  "within-time" (per-country deviation variance) share.
* :func:`rolling_between_within_share` — same, computed in a rolling
  centered window so the share's time evolution is visible.
* :func:`correlation_with_clustering` — pairwise Pearson on first-differences,
  reordered by single-linkage hierarchical clustering on ``1 - corr`` (signed,
  not absolute, since uncertainty co-movement is expected to be positive
  and a negative pair is genuinely informative).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


def _wide(panel: pd.DataFrame, var: str) -> pd.DataFrame:
    sub = panel[panel["variable"] == var]
    return (sub.pivot_table(index="date", columns="code",
                            values="value", aggfunc="first")
            .sort_index())


def between_within_share(panel: pd.DataFrame, var: str) -> pd.DataFrame:
    """Static decomposition of variance for *var* into between/within shares.

    Returns a single-row DataFrame with columns
    ``between_share, within_share, total_var, n_countries, n_obs``.
    """
    wide = _wide(panel, var)
    full = wide.dropna(axis=1, how="any")
    if full.empty:
        return pd.DataFrame({"between_share": [np.nan], "within_share": [np.nan],
                             "total_var": [np.nan], "n_countries": [0], "n_obs": [0]})
    m = full.mean(axis=1)
    between = float(m.var(ddof=0))
    within = float(full.sub(m, axis=0).var(ddof=0).mean())
    total = between + within
    return pd.DataFrame({
        "between_share": [between / total if total > 0 else np.nan],
        "within_share":  [within / total if total > 0 else np.nan],
        "total_var":     [total],
        "n_countries":   [int(full.shape[1])],
        "n_obs":         [int(full.shape[0])],
    })


def rolling_between_within_share(
    panel: pd.DataFrame, var: str, *, window_months: int = 120,
    min_country_obs: int = 60,
) -> pd.DataFrame:
    """Rolling between/within share. Centered window of *window_months*."""
    wide = _wide(panel, var)
    if wide.empty:
        return pd.DataFrame(columns=["date", "between_share", "within_share", "n_countries"])
    half = window_months // 2
    rows = []
    for i, t in enumerate(wide.index):
        lo = max(0, i - half)
        hi = min(len(wide), i + half + 1)
        if hi - lo < window_months:
            rows.append((t, np.nan, np.nan, 0))
            continue
        block = wide.iloc[lo:hi]
        keep = [c for c in block.columns if block[c].notna().sum() >= min_country_obs]
        if len(keep) < 2:
            rows.append((t, np.nan, np.nan, len(keep)))
            continue
        sub = block[keep].dropna(how="any")
        m = sub.mean(axis=1, skipna=False)
        between = float(m.var(ddof=0))
        within = float(sub.sub(m, axis=0).var(ddof=0).mean())
        total = between + within
        rows.append((t, between / total if total > 0 else np.nan,
                     within / total if total > 0 else np.nan,
                     len(keep)))
    return pd.DataFrame(rows, columns=["date", "between_share", "within_share", "n_countries"])


def correlation_with_clustering(
    panel: pd.DataFrame, var: str, *, method: str = "single",
) -> tuple[pd.DataFrame, list[str]]:
    """Pairwise Pearson on first-differences of *var*, reordered by clustering."""
    wide = _wide(panel, var)
    diff = wide.diff().dropna(how="all")
    keep = [c for c in diff.columns if diff[c].notna().sum() >= 24]
    diff = diff[keep]
    corr = diff.corr(method="pearson")
    if len(keep) < 2:
        return corr, keep
    D = (1.0 - corr).clip(lower=0)
    D_arr = D.values.copy()
    np.fill_diagonal(D_arr, 0.0)
    D = pd.DataFrame(D_arr, index=D.index, columns=D.columns)
    condensed = squareform(D.values, checks=False)
    Z = linkage(condensed, method=method)
    order = [keep[i] for i in leaves_list(Z)]
    return corr, order


__all__ = [
    "between_within_share",
    "rolling_between_within_share",
    "correlation_with_clustering",
]
