"""Diagnostic tests for dynamic panel GMM.

Implements:

- ``hansen_j(...)`` — Hansen J overidentification test (chi-square).
- ``ar_test(...)`` — Arellano-Bond (1991) AR(m) serial-correlation test.
- ``windmeijer_correction(...)`` — Windmeijer (2005) finite-sample SE
  correction for the two-step efficient GMM estimator.

References
----------
Arellano, M. and Bond, S. (1991). Some tests of specification for panel
    data. Review of Economic Studies 58(2), 277-297.
Windmeijer, F. (2005). A finite sample correction for the variance of
    linear efficient two-step GMM estimators. Journal of Econometrics
    126, 25-51.
Roodman, D. (2009). How to do xtabond2: an introduction to difference
    and system GMM in Stata. Stata Journal 9(1), 86-136.
"""
from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------
# Hansen J test
# ---------------------------------------------------------------------


def hansen_j(
    Z: np.ndarray,
    residuals: np.ndarray,
    W: np.ndarray,
    n_regressors: int,
) -> tuple:
    """Hansen J overidentification test.

    Statistic: ``J = (Σ Z_i' u_i)' W (Σ Z_i' u_i) / 1`` evaluated at the
    two-step weight ``W`` and two-step residuals ``u``. (Standard form
    in Arellano-Bond / Roodman.)

    Returns ``(J, p, df)``.
    """
    Zu = Z.T @ residuals  # (m,)
    J = float(Zu @ W @ Zu)
    df = max(Z.shape[1] - n_regressors, 0)
    if df == 0:
        # exactly identified: J = 0, p = NaN
        return J, float("nan"), 0
    p = float(stats.chi2.sf(J, df=df))
    return J, p, df


# ---------------------------------------------------------------------
# Arellano-Bond AR(m) test
# ---------------------------------------------------------------------


def ar_test(
    residuals: np.ndarray,
    diff_rows: list,
    lag: int,
    *,
    Z: np.ndarray | None = None,
    X_diff: np.ndarray | None = None,
    W: np.ndarray | None = None,
) -> tuple:
    """Arellano-Bond AR(``lag``) serial-correlation test.

    A simplified panel-clustered z-test on the cross-product of
    differenced residuals at lag ``lag``. We use the conventional
    test statistic formula:

        m = (Σ_i d_i' d_{i, .lag}) / sqrt(V_m)

    where the lag-``lag`` cross-products are summed within panel and
    panel-clustered variance is the mean of squared per-panel
    contributions, scaled by N.

    This is the simplified (residual-only) variant: it ignores
    estimation uncertainty in β̂. This conservative-on-Type-II,
    correctly-sized-on-Type-I version is described in Arellano-Bond
    (1991) section 5 and elaborated in Roodman (2009) eq. (10). The
    full plug-in correction is more elaborate and commonly omitted in
    pedagogical implementations; we document the choice here.

    Parameters
    ----------
    residuals : ndarray, shape (n_diff_rows,)
    diff_rows : list of dicts with keys ``panel``, ``t``.
    lag : positive int (1 or 2).

    Returns ``(stat, p_value)`` where ``p_value`` is two-sided normal.
    """
    if lag < 1:
        raise ValueError("lag must be >= 1.")
    n = len(residuals)
    if n != len(diff_rows):
        raise ValueError(
            f"residuals length {n} != diff_rows length {len(diff_rows)}."
        )

    # Group by panel (rows are already in panel-contiguous order)
    panel_groups: dict[object, list[tuple[int, int]]] = {}
    for r, drow in enumerate(diff_rows):
        panel_groups.setdefault(drow["panel"], []).append((drow["t"], r))

    # Collect per-panel pair contributions s_i = Σ_t d_t · d_{t-lag}
    # only when both rows exist (consecutive diff times of distance == lag).
    s_per_panel = []
    valid_pairs = []  # (residual_t, residual_t-lag) flattened
    for pid, rows in panel_groups.items():
        rows_sorted = sorted(rows, key=lambda x: x[0])
        t_to_idx = {t: idx for t, idx in rows_sorted}
        s_i = 0.0
        for t, idx in rows_sorted:
            t_lag = t - lag
            if t_lag in t_to_idx:
                idx_lag = t_to_idx[t_lag]
                p = residuals[idx] * residuals[idx_lag]
                s_i += p
                valid_pairs.append((residuals[idx], residuals[idx_lag]))
        s_per_panel.append(s_i)

    if not valid_pairs or all(s == 0 for s in s_per_panel):
        return float("nan"), float("nan")

    s_arr: np.ndarray = np.array(s_per_panel)
    num = float(s_arr.sum())
    # panel-clustered variance: Σ_i s_i^2 (residual-only Roodman 2009 eq. 10
    # without the X / Z plug-in terms — see docstring caveat).
    var_num = float(np.sum(s_arr ** 2))
    if var_num <= 0:
        return float("nan"), float("nan")
    stat = num / np.sqrt(var_num)
    p = float(2.0 * stats.norm.sf(abs(stat)))
    return stat, p


# ---------------------------------------------------------------------
# Windmeijer (2005) finite-sample SE correction
# ---------------------------------------------------------------------


def windmeijer_correction(
    beta_hat: np.ndarray,
    Z: np.ndarray,
    X_diff: np.ndarray,
    y_diff: np.ndarray,
    W2: np.ndarray,
    residuals_step2: np.ndarray,
    cov_uncorrected: np.ndarray,
    diff_rows: list,
    cov_step1: np.ndarray | None = None,
    *,
    residuals_step1: np.ndarray | None = None,
) -> np.ndarray:
    """Windmeijer (2005) finite-sample corrected ("WC-robust") covariance
    of the linear two-step efficient GMM estimator.

    The two-step estimator uses the weight ``W_2 = S(beta_1)^{-1}`` with
    ``S(b) = sum_i Z_i' u_i(b) u_i(b)' Z_i`` and ``u_i(b) = y_i - X_i b``,
    i.e. the weight is a function of the ONE-STEP estimate ``beta_1``.
    The conventional two-step variance ``V_2 = (X'Z W_2 Z'X)^{-1}`` treats
    that weight as fixed and is downward biased in finite samples.
    Windmeijer (2000, IFS WP00/19, eqs. (3.2)-(3.3); 2005, J. Econometrics
    126) expands ``beta_2`` in ``beta_1 - beta_0`` and obtains

        Var_c(beta_2) = V_2 + D V_2 + V_2 D' + D V_1 D'

    where ``V_1`` is the one-step robust (panel-clustered) covariance and
    ``D`` is the derivative of the two-step estimator with respect to the
    argument of its weight matrix, ``D = d beta_2(W(b)) / d b'`` at
    ``b = beta_1``. For linear moments its k-th column is

        D[:, k] = -V_2 X'Z W_2 (dS/db_k)|_{b = beta_1} W_2 Z'u_2,
        (dS/db_k)|_{beta_1} = -sum_i Z_i' (x_ik u1_i' + u1_i x_ik') Z_i,

    with ``u1 = y - X beta_1`` the STEP-1 residuals (the point at which
    ``W_2`` is evaluated) and ``u_2 = y - X beta_2`` the step-2 residuals
    (which enter only through ``Z'u_2``). Evaluating ``dS/db`` at the
    step-2 residuals instead is not Windmeijer's derivative: on Stata's
    ``abdata`` it inflates the [XT] xtabond Example 4 WC-robust standard
    errors by factors 1.01-1.68, whereas the step-1 form reproduces them
    to about 1e-6 (``tests/test_dynpanel/test_fix_dynpanel_stata_xtabond.py``).

    Scaling: ``S``, ``W_2`` and ``V_2`` are used without the 1/N factors of
    the paper; ``D`` is invariant to that common scaling.

    Parameters
    ----------
    beta_hat : (k,) two-step estimate beta_2 (kept for API symmetry).
    Z : (n, m) instrument matrix (difference, or stacked system, rows).
    X_diff : (n, k) regressor matrix matching the rows of ``Z``.
    y_diff : (n,) outcome (unused; kept for API symmetry).
    W2 : (m, m) two-step weight ``S(beta_1)^{-1}``.
    residuals_step2 : (n,) two-step residuals ``y - X beta_2``.
    cov_uncorrected : (k, k) conventional two-step covariance ``V_2``.
    diff_rows : list of dicts with key ``panel``; rows are clustered by
        panel for the ``sum_i`` terms.
    cov_step1 : (k, k) or None
        One-step robust covariance ``V_1`` (Windmeijer's var(beta_1)).
        If None, ``V_2`` is substituted -- an approximation that does not
        reproduce Stata; ``ab_gmm``/``bb_gmm`` always pass ``V_1``.
    residuals_step1 : (n,) or None, keyword-only
        One-step residuals ``y - X beta_1``, the evaluation point of
        ``dS/db``. Required for the correct correction. If omitted, a
        ``FutureWarning`` is issued and the step-2 residuals are used, which
        reproduces the (incorrect) behaviour of puremacro 4.3.0 and
        earlier.

    Returns
    -------
    cov_corrected : (k, k)

    References
    ----------
    Windmeijer, F. (2000). A finite sample correction for the variance of
        linear two-step GMM estimators. IFS Working Paper W00/19,
        eqs. (2.3)-(2.4) and (3.2)-(3.3).
    Windmeijer, F. (2005). A finite sample correction for the variance
        of linear efficient two-step GMM estimators. Journal of
        Econometrics 126, 25-51.
    StataCorp. Stata [XT] xtabond, Example 4 (WC-robust VCE on abdata).
    """
    n, k = X_diff.shape
    m = Z.shape[1]
    V_2 = cov_uncorrected
    V_1 = cov_step1 if cov_step1 is not None else V_2

    if residuals_step1 is None:
        warnings.warn(
            "windmeijer_correction called without residuals_step1: the "
            "derivative of the weight matrix is evaluated at the step-2 "
            "residuals, which is not Windmeijer's (2005) correction and does "
            "not reproduce Stata's WC-robust standard errors. Pass "
            "residuals_step1=y - X @ beta_step1.",
            FutureWarning,
            stacklevel=2,
        )
        u_w = np.asarray(residuals_step2, dtype=float)
    else:
        u_w = np.asarray(residuals_step1, dtype=float)
        if u_w.shape != np.shape(residuals_step2):
            raise ValueError(
                f"residuals_step1 has shape {u_w.shape}; expected "
                f"{np.shape(residuals_step2)} (one residual per row of Z)."
            )

    # Group rows by panel
    panel_groups: dict[object, list[int]] = {}
    for r, drow in enumerate(diff_rows):
        panel_groups.setdefault(drow["panel"], []).append(r)

    # v = W_2 Z'u_2 (step-2 residuals) and h = X'Z W_2
    XZW = (X_diff.T @ Z) @ W2  # (k, m)
    v = W2 @ (Z.T @ residuals_step2)  # (m,)

    D = np.zeros((k, k))
    for k_idx in range(k):
        # (dS/db_k)|_{beta_1} v
        #   = -sum_i [ Z_i'x_ik (u1_i'Z_i v) + Z_i'u1_i (x_ik'Z_i v) ]
        out = np.zeros(m)
        for rows in panel_groups.values():
            Zi = Z[rows]
            xi = X_diff[rows, k_idx]
            ui = u_w[rows]
            Ziv = Zi @ v
            out -= (Zi.T @ xi) * float(ui @ Ziv) + (Zi.T @ ui) * float(xi @ Ziv)
        D[:, k_idx] = -V_2 @ (XZW @ out)

    # Windmeijer (2000) eq. (3.3): Var_c = V_2 + D V_2 + V_2 D' + D V_1 D'
    cov_corr = V_2 + D @ V_2 + V_2 @ D.T + D @ V_1 @ D.T
    cov_corr = 0.5 * (cov_corr + cov_corr.T)
    return cov_corr


__all__ = ["hansen_j", "ar_test", "windmeijer_correction"]
