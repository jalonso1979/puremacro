"""T6 — research-grade Bartik importance & interactions helpers."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def compute_state_iv_coefs(
    panel: pd.DataFrame,
    *,
    horizon: int = 4,
    outcome: str = "log_emp",
    endog: str = "log_state_epu",
    instruments: Sequence[str] = ("bartik_ind", "bartik_dem"),
    min_obs: int = 40,
) -> pd.DataFrame:
    """Per-state IV: y_{s,t+h} - y_{s,t-1} on endog_{s,t}, instruments listed.

    Returns DataFrame with columns: state, beta, se, n, first_stage_f.
    """
    from linearmodels.iv import IV2SLS
    results = []
    panel = panel.copy().sort_index()
    for st, sub in panel.groupby(level="state"):
        sub = sub.reset_index(level="state", drop=True).sort_index()
        y_future = sub[outcome].shift(-horizon)
        y_past = sub[outcome].shift(1)
        sub = sub.assign(yh=y_future - y_past, const=1.0)
        cols = ["yh", "const", endog, *instruments]
        d = sub[cols].dropna()
        d = d[np.isfinite(d).all(axis=1)]
        if len(d) < min_obs:
            continue
        try:
            fit = IV2SLS(d["yh"], d[["const"]], d[[endog]],
                         d[list(instruments)]).fit(cov_type="robust")
            beta = float(fit.params[endog])
            se = float(fit.std_errors[endog])
            try:
                fstat = float(fit.first_stage.diagnostics.loc[endog, "f.stat"])  # type: ignore[attr-defined]  # IV2SLSResults not in stubs
            except Exception:
                fstat = np.nan
            results.append({
                "state": st, "beta": beta, "se": se,
                "n": len(d), "first_stage_f": fstat,
            })
        except Exception:
            continue
    return pd.DataFrame(results)


def pesaran_smith_heterogeneity(state_results: pd.DataFrame) -> dict:
    """Pesaran-Smith (1995) χ² test of β homogeneity across states."""
    from scipy.stats import chi2 as _chi2
    d = state_results.dropna(subset=["beta", "se"])
    d = d[d["se"] > 0]
    if len(d) < 3:
        return {"chi2": np.nan, "df": 0, "p_value": np.nan,
                "mean_beta": float(d["beta"].mean()) if len(d) else np.nan}
    weights = 1.0 / (d["se"] ** 2)
    mean_beta = float((d["beta"] * weights).sum() / weights.sum())
    stat = float(((d["beta"] - mean_beta) ** 2 * weights).sum())
    df = int(len(d) - 1)
    p = float(1.0 - _chi2.cdf(stat, df))
    return {"chi2": stat, "df": df, "p_value": p, "mean_beta": mean_beta}


def three_instrument_gmm(
    panel: pd.DataFrame, *,
    outcome: str, endog: str,
    instruments: Sequence[str],
) -> dict:
    """IVGMM with given instruments + incremental Hansen J on the last one."""
    from linearmodels.iv import IVGMM
    from scipy.stats import chi2 as _chi2
    d = panel.reset_index().copy()
    d["const"] = 1.0
    cols = [outcome, "const", endog, *instruments]
    d = d[cols].dropna()
    d = d[np.isfinite(d).all(axis=1)]
    if len(d) < 50:
        return {"beta": np.nan, "se": np.nan,
                "hansen_j": np.nan, "hansen_p": np.nan,
                "incremental_j": np.nan, "incremental_p": np.nan,
                "first_stage_f": np.nan, "n": len(d)}

    fit_full = IVGMM(
        d[outcome], d[["const"]], d[[endog]], d[list(instruments)],
    ).fit(cov_type="robust")
    beta = float(fit_full.params[endog])
    se = float(fit_full.std_errors[endog])
    j_full = float(fit_full.j_stat.stat)  # type: ignore[attr-defined]  # IVGMMResults not in stubs
    j_full_p = float(fit_full.j_stat.pval)  # type: ignore[attr-defined]  # IVGMMResults not in stubs
    try:
        fs_f = float(fit_full.first_stage.diagnostics.loc[endog, "f.stat"])  # type: ignore[attr-defined]  # IVGMMResults not in stubs
    except Exception:
        fs_f = np.nan

    incr_j = np.nan
    incr_p = np.nan
    # Incremental Hansen J is only defined when the full model is
    # over-identified AND dropping the last instrument still leaves an
    # over-identified (or at least identified) partial model.  With just
    # 2 instruments, the partial fit is exactly identified (J ≡ 0 by
    # construction) and the incremental statistic is meaningless.
    if len(instruments) >= 3:
        partial = list(instruments)[:-1]
        try:
            fit_partial = IVGMM(
                d[outcome], d[["const"]], d[[endog]], d[partial],
            ).fit(cov_type="robust")
            j_partial = float(fit_partial.j_stat.stat)  # type: ignore[attr-defined]  # IVGMMResults not in stubs
            incr_j = max(j_full - j_partial, 0.0)
            incr_p = float(1.0 - _chi2.cdf(incr_j, df=1))
        except Exception:
            pass

    return {
        "beta": beta, "se": se,
        "hansen_j": j_full, "hansen_p": j_full_p,
        "incremental_j": incr_j, "incremental_p": incr_p,
        "first_stage_f": fs_f, "n": len(d),
    }


def bucket_iv_betas(
    *,
    shares: pd.DataFrame,
    shifts: pd.DataFrame,
    y: pd.DataFrame,
    endog: pd.DataFrame,
    share_state_col: str, share_bucket_col: str, share_value_col: str,
    shift_bucket_col: str, shift_date_col: str, shift_value_col: str,
    y_state_col: str, y_date_col: str, y_value_col: str,
    endog_state_col: str, endog_date_col: str, endog_value_col: str,
) -> pd.DataFrame:
    """Just-identified IV coefficient per bucket."""
    from linearmodels.iv import IV2SLS
    rows = []
    buckets = shares[share_bucket_col].unique()
    y_map = y.set_index([y_state_col, y_date_col])[y_value_col]
    endog_map = endog.set_index([endog_state_col, endog_date_col])[endog_value_col]
    for k in buckets:
        sk = shares[shares[share_bucket_col] == k].set_index(share_state_col)[share_value_col]
        xk = shifts[shifts[shift_bucket_col] == k].set_index(shift_date_col)[shift_value_col]
        frame = []
        for st, share_val in sk.items():
            for dt, shift_val in xk.items():
                key = (st, dt)
                if key not in y_map.index or key not in endog_map.index:
                    continue
                frame.append({
                    "state": st, "date": dt,
                    "y": float(y_map.loc[key]),
                    "endog": float(endog_map.loc[key]),
                    "z": float(share_val * shift_val),
                    "const": 1.0,
                })
        df = pd.DataFrame(frame).dropna()
        df = df[np.isfinite(df[["y","endog","z"]]).all(axis=1)]
        if len(df) < 40:
            continue
        try:
            fit = IV2SLS(df["y"], df[["const"]], df[["endog"]], df[["z"]]).fit(cov_type="robust")
            try:
                fs_f = float(fit.first_stage.diagnostics.loc["endog", "f.stat"])  # type: ignore[attr-defined]  # IV2SLSResults not in stubs
            except Exception:
                fs_f = np.nan
            rows.append({
                "bucket": k,
                "beta_k": float(fit.params["endog"]),
                "se_k": float(fit.std_errors["endog"]),
                "first_stage_f_k": fs_f,
                "n_k": len(df),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def bridge_industry_to_demographic(
    bridge: pd.DataFrame,
    demographic_betas: pd.DataFrame,
    *,
    bridge_industry_col: str = "industry_code",
    bridge_weight_col: str = "weight",
    demo_beta_col: str = "beta_d",
    cell_cols: Sequence[str] = ("age_bin", "educ_bin", "sex"),
) -> pd.DataFrame:
    """For each industry k: π_k = Σ_d bridge[k,d] · β_d_dem."""
    merged = bridge.merge(demographic_betas, on=list(cell_cols), how="inner")
    merged["contrib"] = merged[bridge_weight_col] * merged[demo_beta_col]
    out = (merged.groupby(bridge_industry_col)["contrib"].sum()
                 .reset_index()
                 .rename(columns={bridge_industry_col: "industry_code",
                                  "contrib": "pi_k"}))
    return out
