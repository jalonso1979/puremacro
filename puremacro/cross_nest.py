"""Cross-nest elasticities of substitution for a symmetric four-factor
nested CES production structure.

Background
----------
The structural form is the symmetric nested CES

    Y = [ beta K**theta + (1-beta) L**theta ] ** (1/theta)
    K = [ alpha K_e**rho   + (1-alpha) K_i**rho   ] ** (1/rho)
    L = [ gamma L_s**eta   + (1-gamma) L_u**eta   ] ** (1/eta)

with sigma_KL = 1/(1-theta), sigma_ie = 1/(1-rho), sigma_su = 1/(1-eta).

Under Sato (1967) two-level nesting, the Allen-Hicks cross-nest elasticities
between any capital-nest factor and any labor-nest factor satisfy

    sigma_es = sigma_eu = sigma_is = sigma_iu = sigma_KL    (exactly)

This module estimates the four cross-nest elasticities INDEPENDENTLY from
their relative-demand FOCs, treating the Sato identity as a testable
restriction rather than an imposed structure:

    Delta log(L_s / K_e) = sigma_es * Delta log(r_e / w_s) + epsilon   (m_es)
    Delta log(L_u / K_e) = sigma_eu * Delta log(r_e / w_u) + epsilon   (m_eu)
    Delta log(L_s / K_i) = sigma_is * Delta log(r_i / w_s) + epsilon   (m_is)
    Delta log(L_u / K_i) = sigma_iu * Delta log(r_i / w_u) + epsilon   (m_iu)

The estimator handles both within-country panel-FE OLS (when iv_col=None)
and within-country panel-FE 2SLS with country-clustered SE (when iv_col is
the cumulated structural ISTC shock at the appropriate nest). The four
estimates support a joint Wald test of the Sato restriction.

Identification caveats
----------------------
1. OLS has the standard relative-demand bias: when relative prices rise
   for endogenous productivity reasons, relative quantities respond and
   the OLS slope is biased toward zero. Use 2SLS with an exogenous price
   shock (e.g., cum-ISTC IV from the source nest) for unbiased estimation.
2. At annual frequency on EU-KLEMS the cum-ISTC IV is weak (paper Block-D
   audit reports F approx 3.5). Quarterly identification is stronger but
   requires quarterly skill-decomposed labor data, which EU-KLEMS does not
   provide.
3. The intangibles-margin relative prices r_i and w (any) inherit the
   labor-cost-deflation critique of national-accounts R&D deflators.
   Restrict K_i to K_Soft_DB for the clean test.

References
----------
Sato, K. (1967), A Two-Level Constant-Elasticity-of-Substitution Production
    Function, Review of Economic Studies.
Allen, R.G.D. (1938), Mathematical Analysis for Economists.
Hall, R. and Jorgenson, D. (1967), Tax Policy and Investment Behavior, AER.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CrossNestFit:
    """Result of a single cross-nest elasticity estimation.

    Attributes
    ----------
    sigma : float
        Point estimate of the cross-nest elasticity.
    se : float
        Country-clustered standard error.
    first_stage_F : float | None
        First-stage F (Cragg-Donald variant) when iv_col is supplied; None
        for OLS.
    n_obs : int
        Effective sample size after dropping NaN rows.
    n_country : int
        Number of unique countries in the effective sample.
    estimator : str
        'OLS' or '2SLS'.
    pair : str
        Two-letter pair label ('es', 'eu', 'is', 'iu').
    """
    sigma: float
    se: float
    first_stage_F: Optional[float]
    n_obs: int
    n_country: int
    estimator: str
    pair: str


def _within_demean(df: pd.DataFrame, cols: list[str], group: str = "code") -> pd.DataFrame:
    """Country-FE demean the specified columns in place on a copy."""
    out = df.copy()
    for col in cols:
        out[col + "_demean"] = out.groupby(group)[col].transform(
            lambda s: s - s.mean()
        )
    return out


def _fit_ols(
    df: pd.DataFrame, y: np.ndarray, x: np.ndarray, n: int, n_country: int, pair: str
) -> CrossNestFit:
    var_x = float(np.dot(x, x))
    if var_x < 1e-12:
        return CrossNestFit(
            sigma=np.nan, se=np.nan, first_stage_F=None,
            n_obs=n, n_country=n_country,
            estimator="OLS", pair=pair,
        )
    beta = float(np.dot(x, y) / var_x)
    resid = y - beta * x
    df["_score"] = resid * x
    cluster_sum = df.groupby("code")["_score"].sum().to_numpy()
    var_num = float(np.dot(cluster_sum, cluster_sum))
    var_den = var_x ** 2
    se = float(np.sqrt(var_num / max(var_den, 1e-12)))
    return CrossNestFit(
        sigma=beta, se=se, first_stage_F=None,
        n_obs=n, n_country=n_country,
        estimator="OLS", pair=pair,
    )


def _fit_2sls(
    df: pd.DataFrame, y: np.ndarray, x: np.ndarray, z: np.ndarray, n: int, n_country: int, pair: str
) -> CrossNestFit:
    # First stage: x = pi * z + nu  (in demeaned space).
    var_z = float(np.dot(z, z))
    if var_z < 1e-12:
        return CrossNestFit(
            sigma=np.nan, se=np.nan, first_stage_F=np.nan,
            n_obs=n, n_country=n_country,
            estimator="2SLS", pair=pair,
        )
    pi_hat = float(np.dot(z, x) / var_z)
    nu = x - pi_hat * z
    var_nu = float(np.dot(nu, nu) / max(n - 1, 1))
    se_pi = float(np.sqrt(max(var_nu / max(var_z, 1e-12), 0.0)))
    F = (pi_hat / se_pi) ** 2 if se_pi > 0 else np.nan

    # 2SLS slope: cov(z, y) / cov(z, x).
    sx = float(np.dot(z, x))
    sy = float(np.dot(z, y))
    if abs(sx) < 1e-12:
        return CrossNestFit(
            sigma=np.nan, se=np.nan, first_stage_F=float(F),
            n_obs=n, n_country=n_country,
            estimator="2SLS", pair=pair,
        )
    beta = sy / sx
    resid = y - beta * x

    # Country-clustered SE on 2SLS slope.
    df["_score_iv"] = resid * z
    cluster_sum = df.groupby("code")["_score_iv"].sum().to_numpy()
    var_num = float(np.dot(cluster_sum, cluster_sum))
    var_den = (np.dot(z, x)) ** 2
    se = float(np.sqrt(var_num / max(var_den, 1e-12)))
    return CrossNestFit(
        sigma=float(beta), se=se, first_stage_F=float(F),
        n_obs=n, n_country=n_country,
        estimator="2SLS", pair=pair,
    )


def fit_cross_nest_pooled(
    panel: pd.DataFrame,
    *,
    lhs_col: str,
    rhs_col: str,
    iv_col: str | None = None,
    pair: str = "",
) -> CrossNestFit:
    """Estimate one cross-nest elasticity via within-country panel-FE OLS or 2SLS.

    Model
    -----
        lhs_col = sigma * rhs_col + alpha_country + epsilon

    where lhs_col is typically a relative log-quantity (e.g. Delta log L_s/K_e)
    and rhs_col is the corresponding relative log-price (Delta log r_e/w_s).
    Country fixed effects are absorbed by within demeaning.

    If iv_col is supplied, runs 2SLS using iv_col as the instrument for
    rhs_col (typically a cumulated structural ISTC shock at the source nest).
    Otherwise runs OLS.

    Standard error is country-clustered following Cameron, Gelbach, and
    Miller (2008). For 2SLS, the first-stage F is reported (Cragg-Donald
    variant: the simple t**2 on the demeaned first stage; treat as a
    diagnostic, not a finite-sample-precise statistic).

    Parameters
    ----------
    panel : pd.DataFrame
        Long-format panel with a 'code' column for country and the columns
        lhs_col, rhs_col, optionally iv_col.
    lhs_col, rhs_col, iv_col : str
        Column names. Rows with NaN in any of the supplied columns are
        dropped.
    pair : str
        Optional two-letter label (e.g. 'es') for downstream tabulation.

    Returns
    -------
    CrossNestFit
    """
    needed = ["code", lhs_col, rhs_col] + ([iv_col] if iv_col else [])
    df = panel.dropna(subset=needed).copy()
    if df.empty:
        return CrossNestFit(
            sigma=np.nan, se=np.nan, first_stage_F=None,
            n_obs=0, n_country=0,
            estimator=("2SLS" if iv_col else "OLS"), pair=pair,
        )

    cols_to_demean = [lhs_col, rhs_col] + ([iv_col] if iv_col else [])
    df = _within_demean(df, cols_to_demean)

    y = df[lhs_col + "_demean"].to_numpy()
    x = df[rhs_col + "_demean"].to_numpy()
    n = len(df)
    n_country = df["code"].nunique()

    if iv_col is None:
        return _fit_ols(df, y, x, n, n_country, pair)

    z = df[iv_col + "_demean"].to_numpy()
    return _fit_2sls(df, y, x, z, n, n_country, pair)


def fit_all_cross_nest(
    panel: pd.DataFrame,
    *,
    quantity_cols: dict[str, str],
    price_cols: dict[str, str],
    iv_cols: dict[str, str | None] | None = None,
) -> dict[str, CrossNestFit]:
    """Run the four cross-nest FOC estimations: es, eu, is, iu.

    Parameters
    ----------
    panel : pd.DataFrame
        Long panel with the relative-quantity and relative-price columns
        already constructed.
    quantity_cols : dict
        Keys 'es','eu','is','iu' mapping to column names for the LHS
        Delta log(L_*/K_*) variables. Example:

            {'es': 'dlog_ls_ke', 'eu': 'dlog_lu_ke',
             'is': 'dlog_ls_ki', 'iu': 'dlog_lu_ki'}

    price_cols : dict
        Same-keyed dict for the RHS Delta log(r_*/w_*) variables.
    iv_cols : dict or None
        Optional same-keyed dict for the IV columns; e.g.
        {'es': 'cum_istc_e', 'eu': 'cum_istc_e',
         'is': 'cum_istc_i', 'iu': 'cum_istc_i'}.
        Missing keys or None values fall back to OLS for that pair.

    Returns
    -------
    dict
        Maps each pair label to a CrossNestFit.
    """
    iv_cols = iv_cols or {}
    out: dict[str, CrossNestFit] = {}
    for pair in ("es", "eu", "is", "iu"):
        out[pair] = fit_cross_nest_pooled(
            panel,
            lhs_col=quantity_cols[pair],
            rhs_col=price_cols[pair],
            iv_col=iv_cols.get(pair),
            pair=pair,
        )
    return out


def sato_wald_test(
    estimates: dict[str, CrossNestFit],
    sigma_kl: float | None = None,
) -> dict[str, float]:
    """Joint Wald test of the Sato (1967) cross-nest restriction.

    Two variants:
      - If sigma_kl is None: tests H0: sigma_es = sigma_eu = sigma_is = sigma_iu
        (three linearly independent restrictions, chi-square with df=3).
      - If sigma_kl is supplied: tests H0: sigma_es = sigma_eu = sigma_is =
        sigma_iu = sigma_kl (four restrictions, chi-square with df=4).

    Assumes the four estimates are approximately independent across the
    four FOCs — a reasonable first approximation when each FOC is fit on a
    different relative-quantity/relative-price pair and country-FE absorb
    the within-country common variation. A more rigorous version would
    stack the four moment conditions in a joint GMM with cross-equation
    covariance from the empirical score, which we leave for follow-up.

    Returns
    -------
    dict
        Keys: 'chi2', 'df', 'p_value'.
    """
    pairs = ("es", "eu", "is", "iu")
    sigmas = np.array([estimates[p].sigma for p in pairs])
    ses    = np.array([estimates[p].se for p in pairs])
    if np.any(~np.isfinite(sigmas)) or np.any(~np.isfinite(ses)) or np.any(ses <= 0):
        return {"chi2": float("nan"), "df": float("nan"), "p_value": float("nan")}
    var = ses ** 2

    if sigma_kl is None:
        # Test pairwise equality: 3 restrictions, e.g. sigma_eu - sigma_es,
        # sigma_is - sigma_es, sigma_iu - sigma_es.
        diff = sigmas[1:] - sigmas[0]
        # Approximate (independent) variance of differences.
        var_diff = var[1:] + var[0]
        chi2 = float(np.sum(diff ** 2 / var_diff))
        df = 3
    else:
        # Test against sigma_kl: 4 restrictions.
        diff = sigmas - sigma_kl
        chi2 = float(np.sum(diff ** 2 / var))
        df = 4

    from scipy.stats import chi2 as chi2_dist
    p_value = float(chi2_dist.sf(chi2, df))
    return {"chi2": chi2, "df": df, "p_value": p_value}


def build_cross_nest_columns(
    panel: pd.DataFrame,
    *,
    code_col: str = "code",
    year_col: str = "year",
    L_s_col: str = "L_s",
    L_u_col: str = "L_u",
    K_e_col: str = "K_e",
    K_i_col: str = "K_i",
    w_s_col: str = "w_s",
    w_u_col: str = "w_u",
    r_e_col: str = "r_e",
    r_i_col: str = "r_i",
) -> pd.DataFrame:
    """Construct the four relative log-quantity and relative log-price columns.

    Adds to the panel: dlog_ls_ke, dlog_lu_ke, dlog_ls_ki, dlog_lu_ki,
    dlog_re_ws, dlog_re_wu, dlog_ri_ws, dlog_ri_wu.

    Differences are within-country first differences sorted by year.

    Returns a copy of the panel sorted by (code, year) with the new columns
    appended.
    """
    df = panel.sort_values([code_col, year_col]).reset_index(drop=True).copy()

    def _log_ratio_diff(num: str, den: str, new: str) -> None:
        # Avoid Series-grouped-by-external-Series alignment quirks (which
        # trigger "cannot reindex on an axis with duplicate labels" on some
        # pandas versions). Build the column as a fresh ndarray via groupby
        # on the DataFrame and direct .diff().
        ratio = np.log(df[num].astype(float) / df[den].astype(float))
        df["_tmp_ratio"] = ratio
        df[new] = df.groupby(code_col)["_tmp_ratio"].diff()
        df.drop(columns="_tmp_ratio", inplace=True)

    # Relative quantities.
    _log_ratio_diff(L_s_col, K_e_col, "dlog_ls_ke")
    _log_ratio_diff(L_u_col, K_e_col, "dlog_lu_ke")
    _log_ratio_diff(L_s_col, K_i_col, "dlog_ls_ki")
    _log_ratio_diff(L_u_col, K_i_col, "dlog_lu_ki")
    # Relative prices.
    _log_ratio_diff(r_e_col, w_s_col, "dlog_re_ws")
    _log_ratio_diff(r_e_col, w_u_col, "dlog_re_wu")
    _log_ratio_diff(r_i_col, w_s_col, "dlog_ri_ws")
    _log_ratio_diff(r_i_col, w_u_col, "dlog_ri_wu")

    return df


def crossnest_summary(estimates: dict[str, CrossNestFit]) -> pd.DataFrame:
    """Tabulate the four cross-nest estimates in a compact DataFrame."""
    rows = []
    for pair in ("es", "eu", "is", "iu"):
        f = estimates[pair]
        rows.append(
            dict(
                pair=pair,
                description={
                    "es": "equipment-skilled",
                    "eu": "equipment-unskilled",
                    "is": "intangibles-skilled",
                    "iu": "intangibles-unskilled",
                }[pair],
                estimator=f.estimator,
                sigma=f.sigma,
                se=f.se,
                first_stage_F=f.first_stage_F,
                n_obs=f.n_obs,
                n_country=f.n_country,
            )
        )
    return pd.DataFrame(rows)
