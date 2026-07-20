"""Regenerate puremacro/validation/goldens/lp.json (dev-only; uses
statsmodels + linearmodels).

References (all independent of puremacro):
  * lp_hac  β_h, SE_h  -> statsmodels OLS with cov_type='HAC' (Bartlett kernel,
    maxlags=h+1, use_correction=False) of y_{t+h}-y_{t-1} on shock + lags.
  * lp_iv   β_h        -> linearmodels IV2SLS (genuine 2SLS).
  * panel_lp β_h       -> linearmodels PanelOLS (EntityEffects + TimeEffects).

The demo DGPs and the (horizons, n_lags) spec are imported from
puremacro.validation.cases_lp so the golden and the puremacro computation share
byte-identical inputs. Run from the package dir:

    python tools/gen_validation_goldens_lp.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import statsmodels
import statsmodels.api as sm
import linearmodels
from linearmodels.iv import IV2SLS
from linearmodels.panel import PanelOLS

from puremacro.validation.cases_lp import (
    LP_HORIZONS,
    LP_N_LAGS,
    lp_iv_demo,
    lp_panel_demo,
    lp_ts_demo,
)


def _ref_lp_hac(df, y, x, horizons, n_lags):
    """statsmodels OLS-HAC, replicating lp_hac's design exactly."""
    betas, ses = [], []
    for h in horizons:
        sub = df[[y, x]].copy()
        sub["dy_h"] = sub[y].shift(-h) - sub[y].shift(1)
        for lag in range(1, n_lags + 1):
            sub[f"{x}_L{lag}"] = sub[x].shift(lag)
            sub[f"{y}_L{lag}"] = sub[y].shift(lag)
        sub = sub.dropna()
        n = len(sub)
        cols = [np.ones(n), sub[x].to_numpy()]
        for lag in range(1, n_lags + 1):
            cols.append(sub[f"{x}_L{lag}"].to_numpy())
            cols.append(sub[f"{y}_L{lag}"].to_numpy())
        X = np.column_stack(cols)
        m = sm.OLS(sub["dy_h"].to_numpy(), X).fit(
            cov_type="HAC", cov_kwds={"maxlags": h + 1, "use_correction": False}
        )
        betas.append(float(m.params[1]))
        ses.append(float(m.bse[1]))
    return np.asarray(betas), np.asarray(ses)


def _ref_lp_iv(df, y, x, z, horizons, n_lags):
    """linearmodels IV2SLS β on the endogenous shock (lags are exogenous)."""
    betas = []
    for h in horizons:
        sub = df[[y, x, z]].copy()
        sub["dy_h"] = sub[y].shift(-h) - sub[y].shift(1)
        for lag in range(1, n_lags + 1):
            sub[f"{x}_L{lag}"] = sub[x].shift(lag)
            sub[f"{y}_L{lag}"] = sub[y].shift(lag)
        sub = sub.dropna()
        sub["const"] = 1.0
        exog_cols = (
            ["const"]
            + [f"{x}_L{lag}" for lag in range(1, n_lags + 1)]
            + [f"{y}_L{lag}" for lag in range(1, n_lags + 1)]
        )
        fit = IV2SLS(
            dependent=sub["dy_h"],
            exog=sub[exog_cols],
            endog=sub[[x]],
            instruments=sub[[z]],
        ).fit(cov_type="unadjusted")
        betas.append(float(fit.params[x]))
    return np.asarray(betas)


def _ref_panel_lp(df_wide, y, x, horizons, n_lags):
    """linearmodels PanelOLS two-way FE β on the shock."""
    df = df_wide.copy()
    g = df.groupby(level="code")
    for lag in range(1, n_lags + 1):
        df[f"xL{lag}"] = g[x].shift(lag)
        df[f"yL{lag}"] = g[y].shift(lag)
    betas = []
    for h in horizons:
        d = df.copy()
        d["dy_h"] = g[y].shift(-h) - g[y].shift(1)
        rhs = (
            [x]
            + [f"xL{lag}" for lag in range(1, n_lags + 1)]
            + [f"yL{lag}" for lag in range(1, n_lags + 1)]
        )
        d = d[["dy_h"] + rhs].dropna()
        fit = PanelOLS(
            d["dy_h"], d[rhs], entity_effects=True, time_effects=True
        ).fit()
        betas.append(float(fit.params[x]))
    return np.asarray(betas)


def main() -> None:
    H, NL = LP_HORIZONS, LP_N_LAGS

    b_hac, se_hac = _ref_lp_hac(lp_ts_demo(), "y", "x", H, NL)
    b_iv = _ref_lp_iv(lp_iv_demo(), "y", "x", "z", H, NL)
    b_panel = _ref_panel_lp(lp_panel_demo(), "y", "x", H, NL)

    out = {
        "_meta": {
            "generated_by": "tools/gen_validation_goldens_lp.py",
            "reference": (
                f"statsmodels {statsmodels.__version__} OLS(cov_type='HAC', "
                f"use_correction=False); linearmodels {linearmodels.__version__} "
                "IV2SLS and PanelOLS(EntityEffects+TimeEffects)"
            ),
            "dgp": (
                "puremacro.validation.cases_lp: lp_ts_demo (seed=424242), "
                "lp_iv_demo (seed=20260531), lp_panel_demo (seed=7777); "
                f"horizons={H}, n_lags={NL}"
            ),
            "regenerate": "python tools/gen_validation_goldens_lp.py",
        },
        "jorda_hac_beta_vs_statsmodels": {"beta": b_hac.tolist()},
        "jorda_hac_se_vs_statsmodels": {"se": se_hac.tolist()},
        "iv_beta_vs_linearmodels_iv2sls": {"beta": b_iv.tolist()},
        "panel_two_way_fe_beta_vs_linearmodels": {"beta": b_panel.tolist()},
    }

    dest = (
        Path(__file__).resolve().parents[1]
        / "puremacro"
        / "validation"
        / "goldens"
        / "lp.json"
    )
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {dest}")
    print(f"  lp_hac  beta {b_hac}")
    print(f"  lp_hac  se   {se_hac}")
    print(f"  lp_iv   beta {b_iv}")
    print(f"  panel   beta {b_panel}")


if __name__ == "__main__":
    main()
