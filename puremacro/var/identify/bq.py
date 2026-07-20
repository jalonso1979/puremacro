"""Blanchard-Quah (1989) long-run identification extended to n-variable VAR.

Imposes that only the 'permanent' shock (indicated by `permanent_var_idx`) has
non-zero long-run effect on the variable at that index; all other shocks are
restricted to have zero long-run effect on that variable.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.random import default_rng

from ..._linalg import safe_cholesky
from ..estimate import estimate_var
from ..irf import irf as compute_irf
from ._results import BQSVARResult

# Same thresholds as cholesky_svar — see var/identify/cholesky.py for
# rationale on the stricter bootstrap conditioning bar.
_BOOT_FAIL_WARN_THRESHOLD = 0.05
_BOOT_COND_RATIO_FLOOR = 1e-4


def _bq_impact(A_list, Sigma, permanent_var_idx=0):
    n = Sigma.shape[0]
    try:
        Psi_1 = np.linalg.inv(np.eye(n) - sum(A_list))
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "bq_svar: long-run matrix (I - A(1)) is singular — the VAR has "
            "a (near-)unit root or collinear variables, so the long-run "
            "restriction cannot be imposed on this sample"
        ) from exc
    # Permute so that permanent variable comes first.
    perm = [permanent_var_idx] + [i for i in range(n) if i != permanent_var_idx]
    P = np.eye(n)[perm]
    Psi_1_p = P @ Psi_1 @ P.T
    Sigma_p = P @ Sigma @ P.T
    Omega = Psi_1_p @ Sigma_p @ Psi_1_p.T
    L = safe_cholesky(Omega, name="bq_svar Ω")
    B_p = np.linalg.solve(Psi_1_p, L)
    # Un-permute back to original variable order.
    return P.T @ B_p @ P


def bq_svar(
    Y: np.ndarray,
    *,
    p: int,
    horizon: int,
    permanent_var_idx: int = 0,
    n_boot: int = 500,
    ci: float = 0.9,
    seed: int = 0,
) -> BQSVARResult:
    rng = default_rng(seed)
    A_list, c, Sigma, resid, _ = estimate_var(Y, p)
    B = _bq_impact(A_list, Sigma, permanent_var_idx)
    # compute_irf returns (H+1, n, n); cumsum along horizon axis (0)
    point = np.cumsum(compute_irf(A_list, B, horizon), axis=0)

    T, n = Y.shape
    accepted = []
    n_fail = 0
    lo_q = (1 - ci) / 2
    hi_q = 1 - lo_q

    for b in range(n_boot):
        idx = rng.integers(0, len(resid), size=len(resid))
        eps_b = resid[idx]
        Yb = np.zeros_like(Y)
        Yb[:p] = Y[:p]
        for t in range(p, T):
            yt = c.copy()
            for l in range(p):
                yt += A_list[l] @ Yb[t - l - 1]
            yt += eps_b[t - p]
            Yb[t] = yt
        A_b, _, Sigma_b, _, _ = estimate_var(Yb, p)
        # Reject ill-conditioned reduced-form Σ_b before even attempting
        # the long-run impact factorisation: bands built on bootstrap
        # draws with cond(Σ_b) > 1e8 are statistically meaningless. Also
        # guard against LAPACK's silent NaN-factor case on ultra-
        # degenerate Σ_b (see cholesky.py for the same pattern).
        try:
            L_test = safe_cholesky(Sigma_b, name="bq_svar bootstrap Σ_b")
        except np.linalg.LinAlgError:
            n_fail += 1
            continue
        diagL = np.abs(np.diag(L_test))
        if (not np.all(np.isfinite(diagL))
                or (diagL.size
                    and diagL.min() < _BOOT_COND_RATIO_FLOOR * diagL.max())):
            n_fail += 1
            continue
        try:
            B_b = _bq_impact(A_b, Sigma_b, permanent_var_idx)
        except np.linalg.LinAlgError:
            n_fail += 1
            continue
        accepted.append(np.cumsum(compute_irf(A_b, B_b, horizon), axis=0))

    if not accepted:
        raise np.linalg.LinAlgError(
            f"bq_svar: all {n_boot} bootstrap draws produced a non-PD "
            "long-run covariance Ω. The data may be too short or the "
            "VAR may be near non-stationary."
        )
    fail_rate = n_fail / n_boot
    if fail_rate > _BOOT_FAIL_WARN_THRESHOLD:
        warnings.warn(
            f"bq_svar: {n_fail}/{n_boot} bootstrap draws "
            f"({fail_rate:.1%}) produced a non-PD long-run Ω and were "
            "dropped. Bands are computed from the surviving draws and "
            "may be unreliable.",
            stacklevel=2,
        )
    draws = np.stack(accepted, axis=0)
    return BQSVARResult(
        irf_point=point,
        irf_lower=np.quantile(draws, lo_q, axis=0),
        irf_upper=np.quantile(draws, hi_q, axis=0),
        n_boot=n_boot,
        n_fail=n_fail,
        ci=ci,
    )
