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
    p = float(2.0 * (1.0 - stats.norm.cdf(abs(stat))))
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
) -> np.ndarray:
    """Apply the Windmeijer (2005) finite-sample correction to a
    two-step GMM covariance matrix.

    The standard two-step sandwich SE underestimates the true variance
    in finite samples because the optimal weight ``W_2 = W_2(β̂_1)``
    is treated as fixed when in fact it depends on first-step β̂_1.
    Windmeijer (2005) Theorem 1 / eq. (2.5) gives an analytic correction:

        Var_W = V_2 + D · V_2 + V_2 · D' + D · V_1 · D'

    with the analytical Jacobian (Windmeijer 2005 eq. (2.5), Roodman
    2009 eq. (16) up to sign convention):

        D[:, k] = -V_2 · (X'Z) · W_2 · ∂S/∂β_k · W_2 · (Z'û_2)

    where
        V_2 = (X'Z W_2 Z'X)^{-1}
        S(β) = Σ_i Z_i' u_i(β) u_i(β)' Z_i
        ∂S/∂β_k = -Σ_i ( Z_i' x_{ik} u_i' Z_i + Z_i' u_i x_{ik}' Z_i )

    Parameters
    ----------
    beta_hat : (k,) two-step estimate (β̂_2).
    Z : (n, m) instrument matrix.
    X_diff : (n, k) design matrix.
    y_diff : (n,) outcome (unused but kept for API symmetry).
    W2 : (m, m) two-step weight matrix.
    residuals_step2 : (n,) two-step residuals û_2 = y - X β̂_2.
    cov_uncorrected : (k, k) the standard two-step sandwich covariance V_2.
    diff_rows : list of dicts with keys ``panel`` and ``t``; used to
        partition rows by panel for the Σ_i sums.
    cov_step1 : (k, k) or None
        The first-step robust covariance V_1. If None, V_2 is used
        (Roodman 2009 simplification — xtabond2 default).

    Returns
    -------
    cov_corrected : (k, k)

    References
    ----------
    Windmeijer, F. (2005). A finite sample correction for the variance
        of linear efficient two-step GMM estimators. Journal of
        Econometrics 126, 25-51.
    Roodman, D. (2009). How to do xtabond2. Stata Journal 9(1), 86-136.
    """
    n, k = X_diff.shape
    m = Z.shape[1]
    V_2 = cov_uncorrected
    V_1 = cov_step1 if cov_step1 is not None else V_2

    # Group rows by panel
    panel_groups: dict[object, list[int]] = {}
    for r, drow in enumerate(diff_rows):
        panel_groups.setdefault(drow["panel"], []).append(r)

    # Pre-compute X'Z, Z'û_2
    XZ = X_diff.T @ Z  # (k, m)
    Zu = Z.T @ residuals_step2  # (m,)

    # Common factor h = X'Z W_2  ∈ R^{k × m}
    XZW = XZ @ W2

    D = np.zeros((k, k))
    # For each k_idx, build dS/dβ_k and assemble.
    for k_idx in range(k):
        # ∂S/∂β_k = -Σ_i ( Z_i' x_{ik} u_i' Z_i + Z_i' u_i x_{ik}' Z_i )
        # We need the action of dS on W_2 Z'û_2. Define v = W_2 Z'û_2 ∈ R^m.
        v = W2 @ Zu  # (m,)
        # ∂S/∂β_k · v = -Σ_i ( Z_i' x_{ik} (u_i' Z_i v) + Z_i' u_i (x_{ik}' Z_i v) )
        # The result is a vector in R^m.
        out = np.zeros(m)
        for pid, rows in panel_groups.items():
            Zi = Z[rows]
            xi = X_diff[rows, k_idx]  # (n_i,)
            ui = residuals_step2[rows]  # (n_i,)
            ZiTxi = Zi.T @ xi  # (m,)
            ZiTui = Zi.T @ ui  # (m,)
            scalar1 = float(ui @ (Zi @ v))  # u_i' Z_i v
            scalar2 = float(xi @ (Zi @ v))  # x_{ik}' Z_i v
            out += -(ZiTxi * scalar1 + ZiTui * scalar2)
        # D[:, k_idx] = -V_2 · X'Z · W_2 · (∂S/∂β_k · v)... but wait,
        # the factor structure from Windmeijer (2005) eq. 2.5 is:
        #   D[:, k] = -V_2 · X'Z · W_2 · (∂S/∂β_k) · v
        #          = -V_2 · XZW · (out / matrix collapsed)
        # We've already computed (∂S/∂β_k) · v = out. So:
        D[:, k_idx] = -V_2 @ (XZW @ out)

    # Windmeijer (2005) eq. (2.7) / Roodman 2009 eq. (16):
    cov_corr = V_2 + D @ V_2 + V_2 @ D.T + D @ V_1 @ D.T
    cov_corr = 0.5 * (cov_corr + cov_corr.T)
    return cov_corr


__all__ = ["hansen_j", "ar_test", "windmeijer_correction"]
