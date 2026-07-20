"""Long-history LP wrappers for T15.

Three primitives, all thin layers over :mod:`puremacro.lp`:

* :func:`derive_innovation_shock` — per-country AR(2) residual of an
  uncertainty index, z-scored within country. The same AR(2) trick used
  by ``build_innovation`` for ``unc_pc_<freq>``, generalized to an
  arbitrary input variable (e.g. ``gpr_m``, ``epu_m``).
* :func:`lp_per_country` — country-by-country Jordà LP using
  :func:`puremacro.lp.jorda.lp_hac`. Returns a long DataFrame
  ``[code, h, beta, se, ci_lo, ci_hi, n_obs]``.
* :func:`pooled_with_dk_se` — pooled panel LP with Driscoll-Kraay SEs,
  via :func:`puremacro.lp.panel_dk.panel_lp_dk`.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ..lp.jorda import lp_hac
from ..lp.panel_dk import panel_lp_dk


def _ar2_residual(y: pd.Series) -> pd.Series:
    s = y.dropna().sort_index()
    if len(s) < 10:
        return pd.Series(np.nan, index=y.index)
    yv = s.values[2:]
    X = np.column_stack([np.ones(len(yv)), s.values[1:-1], s.values[:-2]])
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta
    out = pd.Series(np.nan, index=s.index)
    out.iloc[2:] = resid
    return out.reindex(y.index)


def derive_innovation_shock(panel: pd.DataFrame, var: str) -> pd.DataFrame:
    """Per-country AR(2) innovation of *var*, z-scored.

    Returns long-format DataFrame with columns ``code, date, variable, value``
    where ``variable`` is ``f"{base}_innov_{suffix}"``.
    """
    base, _, suffix = var.rpartition("_")
    out_var = f"{base}_innov_{suffix}"
    sub = panel[panel["variable"] == var]
    rows = []
    for code, grp in sub.groupby("code"):
        s = grp.sort_values("date").set_index("date")["value"].astype(float)
        r = _ar2_residual(s).dropna()
        if r.empty or r.std(ddof=0) == 0:
            continue
        z = (r - r.mean()) / r.std(ddof=0)
        for d, v in z.items():
            rows.append({"code": code, "date": d, "variable": out_var,
                         "value": float(v)})
    return pd.DataFrame(rows)


def lp_per_country(panel: pd.DataFrame, *, outcome: str, shock: str,
                   horizons: Iterable[int], lags: int = 4,
                   unit_col: str = "code", date_col: str = "date",
                   controls: Sequence[str] = (),
                   ci: float = 0.9) -> pd.DataFrame:
    """Country-by-country Jordà LP. Returns long DataFrame.

    Columns: ``code, h, beta, se, ci_lo, ci_hi, n_obs``.
    """
    horizons = list(horizons)
    h_max = max(horizons)
    alpha = 1.0 - ci
    rows = []
    for code, grp in panel.groupby(unit_col):
        df = grp.sort_values(date_col).reset_index(drop=True)
        if len(df.dropna(subset=[outcome, shock])) < 4 * lags + h_max + 4:
            continue
        try:
            res = lp_hac(df, y=outcome, x=shock, horizons=horizons,
                         n_lags=lags, controls=list(controls), alpha=alpha)
        except (ValueError, np.linalg.LinAlgError):
            continue
        n_obs = int(df[[outcome, shock]].dropna().shape[0])
        for _, row in res.iterrows():
            rows.append({
                "code": code,
                "h": int(row["h"]),
                "beta": float(row["beta"]),
                "se":   float(row["se"]),
                "ci_lo": float(row["lo"]),
                "ci_hi": float(row["hi"]),
                "n_obs": n_obs,
            })
    return pd.DataFrame(rows)


def pooled_with_dk_se(panel: pd.DataFrame, *, outcome: str, shock: str,
                      horizons: Iterable[int], unit_col: str = "code",
                      date_col: str = "date",
                      controls: Sequence[str] = (),
                      n_lags: int = 4) -> pd.DataFrame:
    """Pooled panel LP with Driscoll-Kraay HAC SEs.

    Wraps :func:`puremacro.lp.panel_dk.panel_lp_dk`. Returns a DataFrame
    with at least ``h, beta, se, n_obs``.
    """
    df_wide = (panel[[unit_col, date_col, outcome, shock] + list(controls)]
               .set_index([unit_col, date_col])
               .sort_index())
    res = panel_lp_dk(df_wide, y=outcome, x=shock,
                      horizons=list(horizons),
                      n_lags=n_lags,
                      controls=tuple(controls),
                      entity_level=unit_col, time_level=date_col)
    res = res.copy()
    res["n_obs"] = int(len(panel))
    # Backwards-compat: rename lo/hi to ci_low/ci_high so existing notebook
    # cells that read ci_low / ci_high keep working.
    if "lo" in res.columns and "ci_low" not in res.columns:
        res = res.rename(columns={"lo": "ci_low", "hi": "ci_high"})
    return res


def lp_state_dependent(panel: pd.DataFrame, *, outcome: str, shock: str,
                       state: str, horizons: Iterable[int],
                       lags: int = 4,
                       unit_col: str = "code", date_col: str = "date",
                       controls: Sequence[str] = ()) -> pd.DataFrame:
    """Per-country state-dependent LP (Auerbach-Gorodnichenko style).

    Splits each shock into low- and high-state components via the smooth
    transition F (column ``state``); recovers β^L and β^H jointly.

    Returns long DataFrame with columns
    ``code, h, beta_L, se_L, beta_H, se_H, n_obs``.
    """
    from ..inference.hac import newey_west_se

    horizons = list(horizons)
    rows = []
    for code, grp in panel.groupby(unit_col):
        df = grp.sort_values(date_col).reset_index(drop=True).copy()
        df["__sL"] = (1.0 - df[state]) * df[shock]
        df["__sH"] = df[state] * df[shock]
        for h in horizons:
            dh = df.copy()
            dh["__y_lead"] = dh[outcome].shift(-h)
            for l in range(1, lags + 1):
                dh[f"__y_lag{l}"] = dh[outcome].shift(l)
                dh[f"__sL_lag{l}"] = dh["__sL"].shift(l)
                dh[f"__sH_lag{l}"] = dh["__sH"].shift(l)
            cols_extra = [f"__y_lag{l}" for l in range(1, lags + 1)] \
                       + [f"__sL_lag{l}" for l in range(1, lags + 1)] \
                       + [f"__sH_lag{l}" for l in range(1, lags + 1)] \
                       + list(controls)
            dh = dh.dropna(subset=["__y_lead", "__sL", "__sH"] + cols_extra)
            if len(dh) < max(30, 4 * lags):
                continue
            X = np.column_stack([np.ones(len(dh)),
                                 dh["__sL"].values,
                                 dh["__sH"].values,
                                 *(dh[c].values for c in cols_extra)])
            y = dh["__y_lead"].values
            try:
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            except np.linalg.LinAlgError:
                continue
            resid = y - X @ beta
            bw = int(np.ceil(1.5 * (h + 1) ** (1.0 / 3.0))) + 1
            se = newey_west_se(X, resid, bw=bw)
            rows.append({
                "code": code, "h": int(h),
                "beta_L": float(beta[1]), "se_L": float(se[1]),
                "beta_H": float(beta[2]), "se_H": float(se[2]),
                "n_obs": int(len(dh)),
            })
    return pd.DataFrame(rows)


__all__ = ["derive_innovation_shock", "lp_per_country", "pooled_with_dk_se",
           "lp_state_dependent"]
