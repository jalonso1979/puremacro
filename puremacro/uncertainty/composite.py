"""Derive ``unc_m`` (GPR-C backbone) and ``unc_pc_*`` (per-country PC1)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

DEFAULT_PROXIES: tuple[str, ...] = (
    "gpr_m", "epu_m", "wui_m", "wtui_m", "wpui_m", "mpu_m", "rv_m",
)
# Proxies always (or near-always) available before 2008; used for the
# pre-WUI-era "long segment" of the stitched composite. WUI/WTUI/WPUI are
# excluded because all three start in 2008-Q1.
LONG_PROXY_BASES: tuple[str, ...] = ("gpr", "epu", "rv", "mpu")
TARGET_MEAN = 100.0
TARGET_SD = 20.0
MIN_OBS = 24
# When |loading_GPR| < this, treat GPR as essentially orthogonal to PC1 and
# fall back to the dominant-loading proxy as sign anchor. Calibrated against
# the unit-norm baseline (1/√k for k=6 ≈ 0.41): any proxy whose loading is
# below ~35% of that uniform reference carries no usable sign signal.
GPR_SIGN_FLOOR = 0.15


def _rescale(values: pd.Series, target_mean: float = TARGET_MEAN,
             target_sd: float = TARGET_SD) -> pd.Series:
    s = values.std(ddof=0)
    if not np.isfinite(s) or s == 0:
        return pd.Series(target_mean, index=values.index)
    return (values - values.mean()) / s * target_sd + target_mean


def _drop_existing(panel: pd.DataFrame, variable: str) -> pd.DataFrame:
    return panel[panel["variable"] != variable].reset_index(drop=True)


def _infer_freq_suffix(proxies: Sequence[str]) -> str:
    """Return ``_m`` or ``_q`` based on the dominant suffix of ``proxies``.

    Defaults to ``_m`` if mixed/ambiguous so behaviour matches the historic
    monthly-only call signature.
    """
    suffixes = [p.rsplit("_", 1)[-1] for p in proxies if "_" in p]
    if suffixes and all(s == "q" for s in suffixes):
        return "_q"
    if suffixes and all(s == "m" for s in suffixes):
        return "_m"
    # Mixed or unrecognised — fall back to monthly to preserve legacy behavior.
    return "_m"


def build_backbone(panel: pd.DataFrame) -> pd.DataFrame:
    """Add ``unc_m`` = GPR-C rescaled per country to mean=100, sd=20."""
    if (panel["variable"] == "gpr_m").sum() == 0:
        return panel
    panel = _drop_existing(panel, "unc_m")

    rows: list[pd.DataFrame] = []
    for code, grp in (panel[panel["variable"] == "gpr_m"]
                      .sort_values(["code", "date"])
                      .groupby("code")):
        if len(grp) < MIN_OBS:
            continue
        vals = grp.set_index("date")["value"].astype(float)
        rescaled = _rescale(vals)
        out = rescaled.reset_index()
        out.columns = ["date", "value"]
        out["code"] = code
        out["variable"] = "unc_m"
        out["sa_source"] = "derived"
        out["source"] = "rescaled(gpr_m, mean=100, sd=20)"
        rows.append(out[["code", "date", "variable", "value", "sa_source", "source"]])

    if not rows:
        return panel
    return pd.concat([panel, pd.concat(rows, ignore_index=True)], ignore_index=True)


def _pc1_from_wide(wide: pd.DataFrame, suffix: str) -> tuple[pd.Series, list[str]] | None:
    """Run PC1 with sign-anchor logic on the largest balanced window of *wide*.

    Returns the raw (un-rescaled) signed PC1 series and the list of proxy
    columns used. Returns ``None`` if the balanced window is too short
    or fewer than 2 proxies survive z-scoring.
    """
    block = wide.dropna()
    if len(block) < MIN_OBS:
        return None
    z = (block - block.mean()) / block.std(ddof=0).replace(0, np.nan)
    z = z.dropna(axis=1, how="any")
    if z.shape[1] < 2:
        return None
    u, s, vt = np.linalg.svd(z.values, full_matrices=False)
    pc1 = u[:, 0] * s[0]
    loadings = vt[0, :]
    gpr_name = f"gpr{suffix}"
    cols = list(z.columns)
    gpr_idx = cols.index(gpr_name) if gpr_name in cols else None
    if gpr_idx is not None and abs(loadings[gpr_idx]) >= GPR_SIGN_FLOOR:
        sign = np.sign(loadings[gpr_idx]) or 1.0
    else:
        sign = np.sign(loadings[np.argmax(np.abs(loadings))]) or 1.0
    return pd.Series(pc1 * sign, index=z.index), cols


def build_composite(panel: pd.DataFrame,
                    proxies: Sequence[str] = DEFAULT_PROXIES,
                    *,
                    min_obs: int = MIN_OBS,
                    extend_history: bool = True) -> pd.DataFrame:
    """Add ``unc_pc_<freq>`` = per-country PC1 of available proxies, rescaled.

    The output variable name is inferred from the suffix of ``proxies``:
    ``_m`` proxies → ``unc_pc_m``, ``_q`` proxies → ``unc_pc_q``. The source
    string is annotated with the tier:

    * ``T1`` — three or more proxies overlapped on PC1.
    * ``T2`` — exactly two proxies overlapped on PC1.
    * ``T3`` — single-proxy fallback (only one proxy available, or no
      multi-proxy overlap window of sufficient length).

    A trailing ``+T0`` (e.g. ``T1+T0``) marks rows whose pre-rich-window
    coverage was backfilled from a "long segment" PC1 built on the always-
    available proxy basis (``LONG_PROXY_BASES`` — typically gpr/epu/rv,
    excluding the post-2008 WUI family). The long segment is mean/sd-
    matched to the rich segment over their overlap window, so post-rich-
    start values are bit-identical to those produced with
    ``extend_history=False``.

    Sign convention: PC1 is sign-flipped to be positively correlated with
    ``gpr_<freq>`` when its loading is informative (``|loading| >=
    GPR_SIGN_FLOOR``); otherwise the sign anchor falls back to the
    dominant-loading proxy. This guards against the JPN/KOR-style flip
    where GPR is essentially orthogonal to PC1 and its loading sign is
    numerical noise.

    Single-proxy fallback: countries with only 1 of the listed proxies
    available fall back to that proxy directly (rescaled the same way),
    so the composite column is always at least as wide as the union of
    single proxies.
    """
    proxy_set = set(proxies)
    suffix = _infer_freq_suffix(tuple(proxies))
    out_var = f"unc_pc{suffix}"
    panel = _drop_existing(panel, out_var)
    long_proxies_in_set = {f"{base}{suffix}" for base in LONG_PROXY_BASES} & proxy_set
    rows: list[pd.DataFrame] = []

    for code, grp in panel[panel["variable"].isin(proxy_set)].groupby("code"):
        wide = grp.pivot_table(index="date", columns="variable",
                               values="value", aggfunc="first").sort_index()
        wide = wide.dropna(how="all")
        if wide.empty or len(wide) < min_obs:
            continue
        # Drop columns with too few non-NaN observations
        keep_cols = [c for c in wide.columns if wide[c].notna().sum() >= min_obs]
        if not keep_cols:
            continue
        wide = wide[keep_cols]

        if wide.shape[1] == 1:
            # Single-proxy fallback
            col = wide.columns[0]
            series = wide[col].dropna()
            rescaled = _rescale(series)
            out = rescaled.reset_index()
            out.columns = ["date", "value"]
            out["code"] = code
            out["variable"] = out_var
            out["sa_source"] = "derived"
            out["source"] = (f"single_proxy({col}; tier=T3; "
                             f"rescaled mean={TARGET_MEAN}, sd={TARGET_SD})")
            rows.append(out[["code", "date", "variable", "value", "sa_source", "source"]])
            continue

        rich = _pc1_from_wide(wide, suffix)
        if rich is None:
            # No clean overlap window. Fall back to the proxy with the most
            # observations, single-proxy style (avoids distorting PC1 via
            # mean-imputation).
            counts = wide.notna().sum().sort_values(ascending=False)
            if counts.iloc[0] < min_obs:
                continue
            fallback_col = counts.index[0]
            series = wide[fallback_col].dropna()
            rescaled = _rescale(series)
            out = rescaled.reset_index()
            out.columns = ["date", "value"]
            out["code"] = code
            out["variable"] = out_var
            out["sa_source"] = "derived"
            out["source"] = (f"single_proxy({fallback_col}, no_overlap; tier=T3; "
                             f"rescaled mean={TARGET_MEAN}, sd={TARGET_SD})")
            rows.append(out[["code", "date", "variable", "value", "sa_source", "source"]])
            continue
        pc1_rich, rich_cols = rich
        # Capture rich-segment moments now — they pin the rescaling so that
        # values inside the rich window stay bit-identical even after we
        # bracket the rich window with long-segment backfill.
        rich_mean = pc1_rich.mean()
        rich_sd = pc1_rich.std(ddof=0)

        # Optional: bracket the rich window with a long-segment PC1 built on
        # always-available proxies (gpr/epu/rv/mpu). Mean/sd-match on overlap,
        # then prepend the pre-rich tail and append the post-rich tail. The
        # rich window itself is preserved bit-identical.
        long_cols: list[str] = []
        full_pc1 = pc1_rich
        if extend_history:
            long_present = [c for c in wide.columns if c in long_proxies_in_set]
            if len(long_present) >= 2:
                long_pc = _pc1_from_wide(wide[long_present], suffix)
                if long_pc is not None:
                    pc1_long, long_cols_candidate = long_pc
                    rich_start = pc1_rich.index.min()
                    rich_end = pc1_rich.index.max()
                    overlap = pc1_long.index.intersection(pc1_rich.index)
                    pre = pc1_long.index[pc1_long.index < rich_start]
                    post = pc1_long.index[pc1_long.index > rich_end]
                    a = pc1_long.loc[overlap] if len(overlap) else None
                    if (a is not None and len(overlap) >= min_obs
                            and (len(pre) > 0 or len(post) > 0)
                            and a.std(ddof=0) > 0):
                        b = pc1_rich.loc[overlap]
                        # Affine match long → rich on overlap, then bracket.
                        scale = b.std(ddof=0) / a.std(ddof=0)
                        offset = b.mean() - a.mean() * scale
                        parts: list[pd.Series] = []
                        if len(pre) > 0:
                            parts.append(pc1_long.loc[pre] * scale + offset)
                        parts.append(pc1_rich)
                        if len(post) > 0:
                            parts.append(pc1_long.loc[post] * scale + offset)
                        full_pc1 = pd.concat(parts).sort_index()
                        long_cols = long_cols_candidate

        # Rescale using the rich-segment moments only, so post-rich-window
        # values match the no-extension result exactly.
        if rich_sd == 0 or not np.isfinite(rich_sd):
            rescaled = pd.Series(TARGET_MEAN, index=full_pc1.index)
        else:
            rescaled = (full_pc1 - rich_mean) / rich_sd * TARGET_SD + TARGET_MEAN
        tier = "T1" if len(rich_cols) >= 3 else "T2"
        if long_cols:
            tier = f"{tier}+T0({'+'.join(sorted(long_cols))})"
        out = rescaled.reset_index()
        out.columns = ["date", "value"]
        out["code"] = code
        out["variable"] = out_var
        out["sa_source"] = "derived"
        out["source"] = (f"PC1({'+'.join(rich_cols)}; tier={tier}; "
                         f"rescaled mean={TARGET_MEAN}, sd={TARGET_SD})")
        rows.append(out[["code", "date", "variable", "value", "sa_source", "source"]])

    if not rows:
        return panel
    return pd.concat([panel, pd.concat(rows, ignore_index=True)], ignore_index=True)


def _ar2_residual(series: pd.Series) -> pd.Series:
    """Fit AR(2) with constant via OLS; return residual aligned with input index.

    Pre-residual observations (first 2) are NaN.
    """
    s = series.dropna().sort_index()
    if len(s) < 10:
        return pd.Series(np.nan, index=series.index)
    y = s.values[2:]
    X = np.column_stack([
        np.ones(len(y)),
        s.values[1:-1],
        s.values[:-2],
    ])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    out = pd.Series(np.nan, index=s.index)
    out.iloc[2:] = resid
    return out.reindex(series.index)


def build_innovation(panel: pd.DataFrame, *, freq: str) -> pd.DataFrame:
    """Add ``unc_innov_<freq>`` = AR(2) residual of ``unc_pc_<freq>`` per country.

    Parameters
    ----------
    panel : long-form DataFrame with cols code, date, variable, value, sa_source, source.
    freq  : "q" or "m" — selects which composite to use.

    The residual is rescaled to mean=0, sd=1 (so a "1-SD shock" interpretation
    is direct) per country. Source string carries the tier of the underlying
    composite plus an AR(2) tag.
    """
    if freq not in ("q", "m"):
        raise ValueError(f"freq must be 'q' or 'm', got {freq!r}")
    pc_var = f"unc_pc_{freq}"
    out_var = f"unc_innov_{freq}"
    panel = _drop_existing(panel, out_var)
    pc_rows = panel[panel["variable"] == pc_var]
    if pc_rows.empty:
        return panel
    rows: list[pd.DataFrame] = []
    for code, grp in pc_rows.groupby("code"):
        s = grp.sort_values("date").set_index("date")["value"]
        if len(s.dropna()) < MIN_OBS:
            continue
        resid = _ar2_residual(s).dropna()
        if resid.empty or resid.std(ddof=0) == 0:
            continue
        # Rescale residual to mean 0, sd 1
        z = (resid - resid.mean()) / resid.std(ddof=0)
        # Carry forward the tier from the source string of the underlying composite.
        src_pc = grp["source"].iloc[0]
        tier = "Tx"
        for tag in ("tier=T1", "tier=T2", "tier=T3"):
            if tag in src_pc:
                tier = tag.split("=")[1]
                break
        out_df = z.reset_index()
        out_df.columns = ["date", "value"]
        out_df["code"] = code
        out_df["variable"] = out_var
        out_df["sa_source"] = "derived"
        out_df["source"] = (f"AR(2)_residual({pc_var}; tier={tier}; "
                            "rescaled mean=0, sd=1)")
        rows.append(out_df[["code", "date", "variable", "value", "sa_source", "source"]])
    if not rows:
        return panel
    return pd.concat([panel, pd.concat(rows, ignore_index=True)], ignore_index=True)


__all__ = ["build_backbone", "build_composite", "build_innovation", "DEFAULT_PROXIES"]
