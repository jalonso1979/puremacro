"""Synthetic Difference-in-Differences (Arkhangelsky-Athey-Hirshberg-
Imbens-Wager 2021).

SDID combines synthetic-control unit weights with DiD time weights.
For a single treated unit (or block of contemporaneously-treated
units) the estimator solves two weighting problems:

  ω̂ = arg min Σ_t  (Σ_i ω_i Y_{i,t} − Y_{1,t})²  + ζ ‖ω‖²
       s.t. ω ≥ 0,  Σ ω_i = 1                     (units; pre-period match)

  λ̂ = arg min Σ_i  (Σ_t λ_t Y_{i,t} − Y_{i,T_post})²
       s.t. λ ≥ 0,  Σ λ_t = 1                     (times; pre vs post)

and the SDID estimate is

    τ̂_SDID = (Y_{1,post} − Σ_i ω̂_i Y_{i,post})
              − Σ_t λ̂_t (Y_{1,t} − Σ_i ω̂_i Y_{i,t}).

This is the most general framework for a single treated cohort:
synthetic-control plus DiD plus a small ridge for stability.

This implementation supports the **single-cohort** case (one treatment
time common across treated units). For staggered designs, run SDID
once per cohort and aggregate as in CS / SA, or use the BJS imputation
estimator.

References
----------
Arkhangelsky, D., Athey, S., Hirshberg, D.A., Imbens, G.W., Wager, S.
    (2021). Synthetic difference-in-differences. AER 111(12), 4088-4118.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ._results import SyntheticDiDResult


def _solve_simplex_quadratic(Y_pre_donors, y_pre_target, ridge: float) -> np.ndarray:
    """Solve  min ‖Y_pre_donors @ ω − y_pre_target‖² + ridge ‖ω‖²
    s.t.  ω ≥ 0,  Σ ω = 1.

    Active-set / projected-gradient is overkill; we use scipy SLSQP
    with the simplex constraints explicitly.
    """
    n = Y_pre_donors.shape[1]
    omega0 = np.full(n, 1.0 / n)

    def obj(omega):
        r = Y_pre_donors @ omega - y_pre_target
        return float(r @ r + ridge * (omega @ omega))

    def jac(omega):
        r = Y_pre_donors @ omega - y_pre_target
        return 2.0 * (Y_pre_donors.T @ r) + 2.0 * ridge * omega

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * n
    res = minimize(obj, omega0, jac=jac, method="SLSQP",
                    bounds=bounds, constraints=constraints,
                    options={"maxiter": 500, "ftol": 1e-9})
    if not res.success:
        return omega0
    return res.x


def synthetic_did(
    df: pd.DataFrame,
    *,
    unit: str = "unit",
    time: str = "time",
    outcome: str = "y",
    treat_time: str = "treat_time",
    n_boot: int = 200,
    alpha: float = 0.10,
    seed: int = 0,
) -> SyntheticDiDResult:
    """Synthetic-DiD for a single common treatment time.

    Identifies the treatment time as the (single) non-NaN value of
    ``treat_time`` across treated units; raises if multiple cohorts
    coexist (pass each cohort separately or use BJS / CS).

    Parameters
    ----------
    df : DataFrame
        Long-format panel.
    unit, time, outcome, treat_time : str
        Column names. ``treat_time`` is the per-unit first-treatment
        period (NaN for never-treated controls).
    n_boot : int, default 200
        Donor-bootstrap replications for SE / CI.
    alpha : float, default 0.10
        Two-sided coverage = ``1 − α`` (so 0.10 ⇒ 90 % CIs).
    seed : int, default 0
        RNG seed for the bootstrap.

    Returns
    -------
    SyntheticDiDResult
        Frozen dataclass with ``tau`` (point estimate), ``omega``
        (donor-unit weights), ``lambda_w`` (pre-period time weights;
        renamed from ``lambda`` because ``lambda`` is a Python reserved
        keyword), ``se``, ``lo``, ``hi``, ``treatment_time``.

    References
    ----------
    Arkhangelsky, D., Athey, S., Hirshberg, D.A., Imbens, G.W. and
        Wager, S. (2021). Synthetic difference-in-differences. AER
        111(12), 4088-4118.
    """
    rng = np.random.default_rng(seed)
    df = df.sort_values([unit, time]).reset_index(drop=True)

    # Identify treated and control units; require a single treatment time.
    cohort_of = df.groupby(unit)[treat_time].first()
    treated_cohorts = cohort_of.dropna().unique()
    if len(treated_cohorts) != 1:
        raise ValueError(
            f"synthetic_did expects a single common treatment time; "
            f"got cohorts {treated_cohorts}. For staggered designs, "
            "iterate per cohort or use BJS / CS."
        )
    T_treat = float(treated_cohorts[0])
    treated_units = cohort_of.index[~cohort_of.isna()].values
    donor_units = cohort_of.index[cohort_of.isna()].values
    if len(donor_units) < 2:
        raise ValueError("need at least 2 never-treated donor units")

    times_all = np.sort(df[time].unique())
    pre_mask = times_all < T_treat
    post_mask = times_all >= T_treat
    pre_times = times_all[pre_mask]
    post_times = times_all[post_mask]
    if len(pre_times) < 2 or len(post_times) < 1:
        raise ValueError("need ≥ 2 pre-treatment and ≥ 1 post-treatment periods")

    Y_wide = df.pivot(index=unit, columns=time, values=outcome)
    Y_donors_pre = Y_wide.loc[donor_units, pre_times].values            # (J, T_pre)
    Y_donors_post = Y_wide.loc[donor_units, post_times].values          # (J, T_post)
    Y_treated_pre = Y_wide.loc[treated_units, pre_times].mean(axis=0).values
    Y_treated_post = Y_wide.loc[treated_units, post_times].mean(axis=0).values

    # Choose ridge as in the SDID paper: σ̂² = sample variance of
    # first-differences within donors, scaled by N_post · N^{1/4}.
    diff_donors = np.diff(Y_donors_pre, axis=1)
    sigma2 = float(diff_donors.var(ddof=0))
    n_treated = max(len(treated_units), 1)
    ridge = sigma2 * (n_treated * len(post_times)) * len(donor_units) ** 0.25

    # Unit weights: match treated pre-period path.
    omega = _solve_simplex_quadratic(
        Y_donors_pre.T, Y_treated_pre, ridge=ridge,
    )
    # Time weights: match donor post-period mean.
    Y_donors_post_mean = Y_donors_post.mean(axis=1)
    lam = _solve_simplex_quadratic(
        Y_donors_pre, Y_donors_post_mean, ridge=ridge / 4,
    )

    # SDID estimand.
    delta_post = Y_treated_post.mean() - omega @ Y_donors_post.mean(axis=1)
    delta_pre = float((lam * (Y_treated_pre - omega @ Y_donors_pre)).sum())
    tau = float(delta_post - delta_pre)

    # Bootstrap (resample donor units only — the conventional choice).
    boot_taus = np.full(n_boot, np.nan)
    for b in range(n_boot):
        donor_idx = rng.choice(len(donor_units), size=len(donor_units), replace=True)
        Y_d_pre_b = Y_donors_pre[donor_idx]
        Y_d_post_b = Y_donors_post[donor_idx]
        try:
            ome_b = _solve_simplex_quadratic(Y_d_pre_b.T, Y_treated_pre, ridge=ridge)
            lam_b = _solve_simplex_quadratic(
                Y_d_pre_b, Y_d_post_b.mean(axis=1), ridge=ridge / 4,
            )
            d_post = Y_treated_post.mean() - ome_b @ Y_d_post_b.mean(axis=1)
            d_pre = float((lam_b * (Y_treated_pre - ome_b @ Y_d_pre_b)).sum())
            boot_taus[b] = d_post - d_pre
        except Exception:
            continue

    se = float(np.nanstd(boot_taus, ddof=0))
    lo = float(np.nanpercentile(boot_taus, 100 * alpha / 2))
    hi = float(np.nanpercentile(boot_taus, 100 * (1 - alpha / 2)))

    return SyntheticDiDResult(
        tau=tau,
        omega=pd.Series(omega, index=donor_units),
        lambda_w=pd.Series(lam, index=pre_times),
        se=se,
        lo=lo,
        hi=hi,
        treatment_time=T_treat,
    )


__all__ = ["synthetic_did"]
