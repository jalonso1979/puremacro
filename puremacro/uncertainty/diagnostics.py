"""Per-country uncertainty diagnostics: tier, ADF, KPSS, AR(2) residual SD."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

# puremacro is a sibling package; ensure it's on path even when
# the worktree's puremacro/ subdir hasn't been pip-installed.
_PUREMACRO_PATH = Path(__file__).resolve().parent.parent.parent / "puremacro"
if _PUREMACRO_PATH.exists() and str(_PUREMACRO_PATH) not in sys.path:
    sys.path.insert(0, str(_PUREMACRO_PATH))

_AdfTestFn = Callable[..., dict]
_KpssTestFn = Callable[..., dict]

adf_test: Optional[_AdfTestFn]
kpss_test: Optional[_KpssTestFn]

try:
    from puremacro.tests.unit_root import adf_test, kpss_test
except ImportError:
    adf_test = None
    kpss_test = None


def _extract_tier(source: str) -> str:
    for tag in ("tier=T1", "tier=T2", "tier=T3"):
        if tag in source:
            return tag.split("=")[1]
    return "?"


def _extract_proxies(source: str) -> str:
    """Pull e.g. ``wui_m+gpr_m+epu_m`` out of ``PC1(wui_m+gpr_m+epu_m; tier=T1; ...)``.

    Tolerates the ``resampled_from_M:`` prefix that quarterly composites
    inherit when they're built by resampling the monthly composite.
    """
    s = source
    if s.startswith("resampled_from_M:"):
        s = s[len("resampled_from_M:"):]
    if s.startswith("PC1("):
        inside = s[4:].split(";")[0]
        return inside
    if s.startswith("single_proxy("):
        inside = s[len("single_proxy("):].split(";")[0].split(",")[0]
        return inside
    return "?"


def build_diagnostics(panel: pd.DataFrame, *, freq: str) -> pd.DataFrame:
    """Return a per-country diagnostics DataFrame for the uncertainty composite.

    Columns: ``code, freq, tier, n_proxies, proxies, n_obs,
    adf_stat, adf_p, kpss_stat, kpss_p, ar2_sigma,
    first_obs, last_obs``.

    ADF / KPSS statistics use ``puremacro.tests.unit_root``; if those tests
    raise (e.g. singular regression for trivial series) the corresponding
    cells are NaN. ``ar2_sigma`` is the (population) standard deviation of
    the level series after AR(2) residualisation — by construction the
    rescaled innovation already has sd≈1, so this column will be ≈1 for
    any country whose innovation series was successfully built.
    """
    if freq not in ("q", "m"):
        raise ValueError(freq)
    pc_var = f"unc_pc_{freq}"
    inv_var = f"unc_innov_{freq}"
    pc_rows = panel[panel["variable"] == pc_var]
    inv_rows = panel[panel["variable"] == inv_var]

    rows = []
    for code, grp in pc_rows.groupby("code"):
        s = grp.sort_values("date").set_index("date")["value"].dropna()
        if len(s) < 10:
            continue
        source = grp["source"].iloc[0]
        tier = _extract_tier(source)
        proxies_str = _extract_proxies(source)
        n_proxies = len(proxies_str.split("+")) if proxies_str != "?" else 0

        if adf_test is not None and kpss_test is not None:
            try:
                adf = adf_test(s.values, regression="c")
                adf_stat, adf_p = float(adf["stat"]), float(adf["p_value"])
            except Exception:
                adf_stat, adf_p = np.nan, np.nan
            try:
                kpss = kpss_test(s.values, regression="c")
                kpss_stat, kpss_p = float(kpss["stat"]), float(kpss["p_value"])
            except Exception:
                kpss_stat, kpss_p = np.nan, np.nan
        else:
            adf_stat = adf_p = kpss_stat = kpss_p = np.nan

        inv_g = inv_rows[inv_rows["code"] == code]
        ar2_sigma = float(inv_g["value"].std(ddof=0)) if len(inv_g) else np.nan

        rows.append({
            "code": code,
            "freq": freq,
            "tier": tier,
            "n_proxies": n_proxies,
            "proxies": proxies_str,
            "n_obs": int(len(s)),
            "adf_stat": adf_stat,
            "adf_p": adf_p,
            "kpss_stat": kpss_stat,
            "kpss_p": kpss_p,
            "ar2_sigma": ar2_sigma,
            "first_obs": pd.Timestamp(s.index.min()).date().isoformat(),
            "last_obs": pd.Timestamp(s.index.max()).date().isoformat(),
        })
    return pd.DataFrame(rows)


__all__ = ["build_diagnostics"]
