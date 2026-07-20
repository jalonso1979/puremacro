"""Exposure IV scalar E_c = Σ_i w_{c,i} × β_i, and Rotemberg industry weights.

The Rotemberg decomposition (Goldsmith-Pinkham-Sorkin-Swift 2020) expresses
the Bartik IV estimator as a weighted sum over industries:

    α_k = (β_k × Σ_c w_{c,k}² × σ²_k) / (Σ_i β_i × Σ_c w_{c,i}² × σ²_i)

so Σ_k α_k = 1 by construction. Negative α_k indicate the industry pulls the
IV estimate against the sign of the headline effect.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def build_exposure_iv(
    shares: pd.DataFrame,
    sensitivities: pd.DataFrame,
) -> pd.DataFrame:
    """Return county_fips → exposure_c scalar.

    E_c = Σ_i w_{c,i} × β_i.
    Industries in shares without a matching β in sensitivities contribute 0
    (filled via fillna(0)); a warning is logged above 1% unmatched and a
    ValueError is raised above 5% unmatched.
    """
    if shares.duplicated(["county_fips", "naics"]).any():
        raise ValueError(
            "shares contains duplicate (county_fips, naics) rows; "
            "deduplicate before calling"
        )

    shares = shares.copy()
    shares["county_fips"] = shares["county_fips"].astype(str)
    shares["naics"] = shares["naics"].astype(str)
    sensitivities = sensitivities.copy()
    sensitivities["naics"] = sensitivities["naics"].astype(str)

    merged = shares.merge(sensitivities[["naics", "shrunk_beta"]], on="naics", how="left")
    n_unmatched = merged["shrunk_beta"].isna().sum()
    frac_unmatched = n_unmatched / len(merged) if len(merged) else 0
    if frac_unmatched > 0.05:
        raise ValueError(
            f"{n_unmatched}/{len(merged)} ({100*frac_unmatched:.0f}%) shares rows have no "
            "matching β — exposure_c would be biased toward zero; align the industry panels"
        )
    if frac_unmatched > 0.01:
        logger.warning(
            "%d/%d (%.1f%%) shares rows have no matching β — contributing 0 to exposure",
            n_unmatched, len(merged), 100 * frac_unmatched,
        )
    merged["shrunk_beta"] = merged["shrunk_beta"].fillna(0.0)
    merged["contrib"] = merged["share"] * merged["shrunk_beta"]
    out = merged.groupby("county_fips", as_index=False)["contrib"].sum()
    return out.rename(columns={"contrib": "exposure_c"})


def rotemberg_weights(
    shares: pd.DataFrame,
    sensitivities: pd.DataFrame,
    shock_variances: pd.DataFrame,
    *,
    variance_col: str = "sigma_sq_nat",
) -> pd.DataFrame:
    """Per-industry Rotemberg α_k weights for the Bartik IV estimator.

    Output columns: naics, alpha, beta, sum_w_sq.
    By construction Σ_k α_k = 1.
    """
    if shares.duplicated(["county_fips", "naics"]).any():
        raise ValueError(
            "shares contains duplicate (county_fips, naics) rows; "
            "deduplicate before calling"
        )
    if variance_col not in shock_variances.columns:
        raise KeyError(
            f"{variance_col!r} not in shock_variances.columns={list(shock_variances.columns)}"
        )

    shares = shares.copy()
    shares["county_fips"] = shares["county_fips"].astype(str)
    shares["naics"] = shares["naics"].astype(str)
    sensitivities = sensitivities.copy()
    sensitivities["naics"] = sensitivities["naics"].astype(str)
    shock_variances = shock_variances.copy()
    shock_variances["naics"] = shock_variances["naics"].astype(str)

    merged = shares.merge(sensitivities[["naics", "shrunk_beta"]], on="naics", how="left")
    merged = merged.merge(shock_variances[["naics", variance_col]], on="naics", how="left")
    merged["shrunk_beta"] = merged["shrunk_beta"].fillna(0.0)
    merged[variance_col] = merged[variance_col].fillna(0.0)
    merged["w_sq"] = merged["share"] ** 2

    per_ind = merged.groupby("naics").agg(
        sum_w_sq=("w_sq", "sum"),
        beta=("shrunk_beta", "first"),
        sigma_sq=(variance_col, "first"),
    ).reset_index()
    per_ind["numer"] = per_ind["beta"] * per_ind["sum_w_sq"] * per_ind["sigma_sq"]
    total = per_ind["numer"].sum()
    if total == 0:
        logger.warning("Rotemberg denominator is zero; returning NaN alphas")
        per_ind["alpha"] = float("nan")
    else:
        per_ind["alpha"] = per_ind["numer"] / total
    return per_ind[["naics", "alpha", "beta", "sum_w_sq"]]
