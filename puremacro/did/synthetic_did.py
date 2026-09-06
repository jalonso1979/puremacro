"""Synthetic Difference-in-Differences (Arkhangelsky-Athey-Hirshberg-
Imbens-Wager 2021).

SDID combines synthetic-control unit weights with DiD time weights.
For a single treated unit (or block of contemporaneously-treated
units) the estimator solves two weighting problems, **each with a free
intercept** (Arkhangelsky et al. 2021, eqs. 4-5):

  (ω̂_0, ω̂) = arg min Σ_{t<T_0} (ω_0 + Σ_i ω_i Y_{i,t} − Ȳ_{tr,t})²
                        + ζ_ω² T_pre ‖ω‖²
       s.t. ω ≥ 0,  Σ ω_i = 1                (units; pre-period match)

  (λ̂_0, λ̂) = arg min Σ_{i∈co} (λ_0 + Σ_{t<T_0} λ_t Y_{i,t} − Ȳ_{i,post})²
                        + ζ_λ² N_co ‖λ‖²
       s.t. λ ≥ 0,  Σ λ_t = 1                (times; pre vs post)

and the SDID estimate is

    τ̂_SDID = (Ȳ_{tr,post} − Σ_i ω̂_i Ȳ_{i,post})
              − Σ_t λ̂_t (Ȳ_{tr,t} − Σ_i ω̂_i Y_{i,t}).

The intercepts are what make SDID invariant to additive unit and time
level shifts: without ω_0 the unit weights would have to match the
treated *level*, not just its pre-trend, and a constant added to the
treated units' outcome would move both ω̂ and τ̂. Profiling out the
intercept is equivalent to centring the pre-period outcome paths
(over time for ω, over control units for λ) before solving the
simplex-constrained ridge problem; that is how it is implemented.

The regularisation follows the paper and the ``synthdid`` R package:
``ζ_ω = (N_tr T_post)^{1/4} σ̂``, ``ζ_λ = 1e-6 σ̂``, with ``σ̂`` the
standard deviation of first differences of the control units'
pre-period outcomes (period-demeaned, so a common time shift does not
move the tuning parameter either).

This implementation supports the **single-cohort** case (one treatment
time common across treated units). For staggered designs use
:func:`puremacro.did.sdid_multi_cohort`, or the BJS / CS estimators.

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


def _solve_simplex_quadratic(
    A: np.ndarray, b: np.ndarray, ridge: float, *, intercept: bool = True,
) -> np.ndarray:
    """Solve  min_{w, w_0} ‖A w + w_0 − b‖² + ridge ‖w‖²
    s.t.  w ≥ 0,  Σ w = 1.

    With ``intercept=True`` the free intercept ``w_0`` is profiled out
    analytically: for any ``w`` the optimal ``w_0`` is the mean residual,
    so the problem is equivalent to the same simplex ridge regression on
    row-centred ``A`` and ``b``. Uses scipy SLSQP with the simplex
    constraints explicitly.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if intercept:
        A = A - A.mean(axis=0, keepdims=True)
        b = b - b.mean()
    n = A.shape[1]
    w0 = np.full(n, 1.0 / n)

    def obj(w):
        r = A @ w - b
        return float(r @ r + ridge * (w @ w))

    def jac(w):
        r = A @ w - b
        return 2.0 * (A.T @ r) + 2.0 * ridge * w

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0,
                    "jac": lambda w: np.ones_like(w)}]
    bounds = [(0.0, 1.0)] * n
    res = minimize(obj, w0, jac=jac, method="SLSQP",
                    bounds=bounds, constraints=constraints,
                    options={"maxiter": 1000, "ftol": 1e-12})
    if not res.success:
        return w0
    w = np.clip(res.x, 0.0, None)
    return w / w.sum()


def _sdid_weights(
    Y_donors_pre: np.ndarray,
    Y_donors_post: np.ndarray,
    Y_treated_pre: np.ndarray,
    *,
    n_treated: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Unit weights ω (over donors) and time weights λ (over pre periods)."""
    J, T_pre = Y_donors_pre.shape
    T_post = Y_donors_post.shape[1]
    # Noise level: sd of first differences of control pre-period outcomes
    # (synthdid's ``noise.level``), with the period mean of each first
    # difference removed. Under the SDID model Y = α_i + β_t + ε the common
    # step β_{t+1} − β_t is not noise, and removing it keeps the ridge —
    # and therefore τ̂ — exactly invariant to additive time-level shifts.
    diffs = np.diff(Y_donors_pre, axis=1)
    diffs = diffs - diffs.mean(axis=0, keepdims=True)
    sigma = float(np.std(diffs, ddof=1)) if diffs.size > 1 else 0.0
    zeta_omega = (n_treated * T_post) ** 0.25 * sigma
    zeta_lambda = 1e-6 * sigma
    ridge_omega = zeta_omega ** 2 * T_pre
    ridge_lambda = zeta_lambda ** 2 * J

    omega = _solve_simplex_quadratic(Y_donors_pre.T, Y_treated_pre,
                                     ridge=ridge_omega)
    lam = _solve_simplex_quadratic(Y_donors_pre, Y_donors_post.mean(axis=1),
                                   ridge=ridge_lambda)
    return omega, lam


def _sdid_tau(
    omega: np.ndarray, lam: np.ndarray,
    Y_donors_pre: np.ndarray, Y_donors_post: np.ndarray,
    Y_treated_pre: np.ndarray, Y_treated_post: np.ndarray,
) -> float:
    delta_post = float(Y_treated_post.mean() - omega @ Y_donors_post.mean(axis=1))
    delta_pre = float((lam * (Y_treated_pre - omega @ Y_donors_pre)).sum())
    return delta_post - delta_pre


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
    ci: float | None = None,
) -> SyntheticDiDResult:
    """Synthetic-DiD for a single common treatment time.

    Identifies the treatment time as the (single) non-NaN value of
    ``treat_time`` across treated units; raises if multiple cohorts
    coexist (use :func:`sdid_multi_cohort`, BJS or CS for staggered
    designs). The panel must be **balanced** over the treated and donor
    units: a missing ``(unit, time)`` cell raises a ``ValueError``
    naming the cell.

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
    ci : float, optional
        Confidence-interval coverage; when given, ``alpha = 1 − ci``
        (same convention as :func:`callaway_santanna`).

    Returns
    -------
    SyntheticDiDResult
        Frozen dataclass with ``tau`` (point estimate), ``omega``
        (donor-unit weights), ``lambda_w`` (pre-period time weights;
        renamed from ``lambda`` because ``lambda`` is a Python reserved
        keyword), ``se``, ``lo``, ``hi``, ``treatment_time``, plus the
        treated-mean and ω-weighted synthetic outcome paths
        (``y_treated``, ``y_synthetic``) used by ``.plot()``.

    Notes
    -----
    Both weight problems include the intercepts of Arkhangelsky et al.
    (2021), so the estimate is invariant to adding a constant to any
    unit's outcome path or a common constant to any period.

    References
    ----------
    Arkhangelsky, D., Athey, S., Hirshberg, D.A., Imbens, G.W. and
        Wager, S. (2021). Synthetic difference-in-differences. AER
        111(12), 4088-4118.
    """
    if ci is not None:
        alpha = 1.0 - ci
    rng = np.random.default_rng(seed)
    df = df.sort_values([unit, time]).reset_index(drop=True)

    # Identify treated and control units; require a single treatment time.
    cohort_of = df.groupby(unit)[treat_time].first()
    treated_cohorts = cohort_of.dropna().unique()
    if len(treated_cohorts) != 1:
        raise ValueError(
            f"synthetic_did expects a single common treatment time; "
            f"got cohorts {treated_cohorts}. For staggered designs, "
            "use sdid_multi_cohort, or iterate per cohort / use BJS / CS."
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

    if df.duplicated(subset=[unit, time]).any():
        dup = df[df.duplicated(subset=[unit, time])].iloc[0]
        raise ValueError(
            f"duplicate ({unit}, {time}) cell: ({dup[unit]!r}, {dup[time]!r}); "
            "synthetic_did needs one row per unit and period"
        )
    Y_wide = df.pivot(index=unit, columns=time, values=outcome)
    missing = Y_wide.isna()
    if missing.to_numpy().any():
        cells = [(u, t) for u, t in zip(*np.nonzero(missing.to_numpy()))]

        def _py(v):
            return v.item() if hasattr(v, "item") else v

        named = [f"({_py(Y_wide.index[u])!r}, {_py(Y_wide.columns[t])!r})"
                 for u, t in cells[:5]]
        more = "" if len(cells) <= 5 else f" and {len(cells) - 5} more"
        raise ValueError(
            f"synthetic_did requires a balanced panel: {len(cells)} missing "
            f"({unit}, {time}) cell(s) {', '.join(named)}{more}. Drop the "
            "affected units or fill the outcome before calling."
        )
    Y_donors_pre = Y_wide.loc[donor_units, pre_times].to_numpy(dtype=float)     # (J, T_pre)
    Y_donors_post = Y_wide.loc[donor_units, post_times].to_numpy(dtype=float)   # (J, T_post)
    Y_treated_pre = Y_wide.loc[treated_units, pre_times].mean(axis=0).to_numpy(dtype=float)
    Y_treated_post = Y_wide.loc[treated_units, post_times].mean(axis=0).to_numpy(dtype=float)

    omega, lam = _sdid_weights(Y_donors_pre, Y_donors_post, Y_treated_pre,
                               n_treated=len(treated_units))
    tau = _sdid_tau(omega, lam, Y_donors_pre, Y_donors_post,
                    Y_treated_pre, Y_treated_post)

    # Bootstrap (resample donor units only — the conventional choice).
    boot_taus = np.full(n_boot, np.nan)
    for b in range(n_boot):
        donor_idx = rng.choice(len(donor_units), size=len(donor_units), replace=True)
        Y_d_pre_b = Y_donors_pre[donor_idx]
        Y_d_post_b = Y_donors_post[donor_idx]
        try:
            ome_b, lam_b = _sdid_weights(Y_d_pre_b, Y_d_post_b, Y_treated_pre,
                                         n_treated=len(treated_units))
            boot_taus[b] = _sdid_tau(ome_b, lam_b, Y_d_pre_b, Y_d_post_b,
                                     Y_treated_pre, Y_treated_post)
        except Exception:
            continue

    if n_boot > 0 and np.isfinite(boot_taus).any():
        se = float(np.nanstd(boot_taus, ddof=0))
        lo = float(np.nanpercentile(boot_taus, 100 * alpha / 2))
        hi = float(np.nanpercentile(boot_taus, 100 * (1 - alpha / 2)))
    else:
        se = lo = hi = float("nan")

    y_treated = pd.Series(
        np.concatenate([Y_treated_pre, Y_treated_post]),
        index=np.concatenate([pre_times, post_times]), name="treated",
    )
    y_synthetic = pd.Series(
        np.concatenate([omega @ Y_donors_pre, omega @ Y_donors_post]),
        index=y_treated.index, name="synthetic",
    )

    return SyntheticDiDResult(
        tau=float(tau),
        omega=pd.Series(omega, index=donor_units),
        lambda_w=pd.Series(lam, index=pre_times),
        se=se,
        lo=lo,
        hi=hi,
        treatment_time=T_treat,
        y_treated=y_treated,
        y_synthetic=y_synthetic,
    )


__all__ = ["synthetic_did"]
