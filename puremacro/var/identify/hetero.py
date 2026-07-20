"""Rigobon (2003) identification via heteroskedasticity.

Exploits shifts in residual variance across regimes (e.g. high/low volatility
states identified by a GARCH regime or external dummy) to recover the structural
impact matrix B without imposing recursive or long-run restrictions.

Math summary
------------
Given regime indicator (0/1) with T observations and a VAR(p) fit:

  1. Compute Sigma_0, Sigma_1 -- regime-conditional residual covariance matrices.
  2. Cholesky-factor Sigma_0:  L_0 = chol(Sigma_0, lower=True)
  3. Build the *ratio matrix* M = L_0^{-1} . Sigma_1 . L_0^{-T}
  4. Eigen-decompose (real symmetric):  M = P . diag(lambda) . P^T  (scipy.linalg.eigh)
  5. Structural impact matrix:  B = L_0 . P
  6. Verify:  B . diag(lambda) . B^T approx Sigma_1  and  B . B^T approx Sigma_0

Inference via moving-block bootstrap (requires puremacro.inference, Task A3).
If that module is unavailable, bootstrap bands are skipped (lower=upper=None).

References
----------
Rigobon, R. (2003). Identification through heteroskedasticity.
    Rev. Econ. Stat. 85(4), 777-792.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.linalg

from ..estimate import estimate_var
from ..irf import irf as compute_irf, fevd as compute_fevd

try:
    from ...inference.moving_block_bootstrap import moving_block_bootstrap, bootstrap_percentiles
    _HAS_MBB = True
except ImportError:  # pragma: no cover — inference ships in the same wheel
    _HAS_MBB = False


# --------------------------------------------------------------------------- #
# Return type
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class HeteroResult:
    """Output of Rigobon heteroskedasticity identification.

    Attributes
    ----------
    B : ndarray (n, n)
        Structural impact matrix.  Column j is the impact vector of shock j.
    variance_ratios : ndarray (n,)
        Eigenvalues lambda from the ratio matrix M = L_0^{-1} Sigma_1 L_0^{-T}.
        These are the variance ratios of each shock across the two regimes.
    irfs : ndarray (H+1, n, n)
        Impulse response functions (structural).
    fevd : ndarray (H+1, n, n)
        Forecast error variance decomposition.
    lower : ndarray (H+1, n, n) or None
        Lower bootstrap band.
    upper : ndarray (H+1, n, n) or None
        Upper bootstrap band.
    point : ndarray (H+1, n, n)
        Point IRF.
    """
    B: np.ndarray
    variance_ratios: np.ndarray
    irfs: np.ndarray          # (H+1, n, n)
    fevd: np.ndarray          # (H+1, n, n)
    lower: Optional[np.ndarray]   # (H+1, n, n) or None
    upper: Optional[np.ndarray]   # (H+1, n, n) or None
    point: np.ndarray             # (H+1, n, n)


# --------------------------------------------------------------------------- #
# Math helpers
# --------------------------------------------------------------------------- #

def _regime_covariances(
    residuals: np.ndarray,
    regime_indicator: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute regime-conditional covariance matrices."""
    T_eff, n = residuals.shape
    idx0 = regime_indicator == 0
    idx1 = regime_indicator == 1

    if idx0.sum() < n + 1:
        raise ValueError(
            f"Regime 0 has only {idx0.sum()} observations; need at least n+1 = {n + 1}."
        )
    if idx1.sum() < n + 1:
        raise ValueError(
            f"Regime 1 has only {idx1.sum()} observations; need at least n+1 = {n + 1}."
        )

    e0 = residuals[idx0]
    e1 = residuals[idx1]
    Sigma_0 = e0.T @ e0 / (idx0.sum() - 1)
    Sigma_1 = e1.T @ e1 / (idx1.sum() - 1)
    return Sigma_0, Sigma_1


def _rigobon_impact(
    Sigma_0: np.ndarray,
    Sigma_1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Core Rigobon identification step.

    ``safe_cholesky`` gates Σ_0; once L0 is in hand its inverse is
    well-conditioned (Σ_0's smallest eigenvalue bounds L0's smallest
    diagonal entry from below).
    """
    from ..._linalg import safe_cholesky

    L0 = safe_cholesky(Sigma_0, name="rigobon Σ_0")
    L0_inv = np.linalg.inv(L0)
    M = L0_inv @ Sigma_1 @ L0_inv.T            # symmetric, PSD in theory
    eigenvalues, P = scipy.linalg.eigh(M)      # ascending order, real
    B = L0 @ P
    return B, eigenvalues


# --------------------------------------------------------------------------- #
# Main API
# --------------------------------------------------------------------------- #

def rigobon_svar(
    Y: np.ndarray,
    *,
    p: int,
    horizon: int,
    regime_indicator: np.ndarray,
    n_boot: int = 500,
    ci: float = 0.9,
    seed: int = 0,
    block_len: Optional[int] = None,
) -> HeteroResult:
    """Rigobon (2003) SVAR identified via heteroskedasticity.

    Parameters
    ----------
    Y : ndarray (T, n)
        Data matrix.
    p : int
        VAR lag order.
    horizon : int
        IRF horizon H (output covers h = 0 ... H).
    regime_indicator : array-like of int (T,) or (T-p,)
        Binary (0/1) regime labels.
    n_boot : int
        Moving-block bootstrap repetitions.  Set 0 to skip bootstrap.
    ci : float
        Confidence level for bootstrap bands.
    seed : int
        Random seed.
    block_len : int or None
        Block length.  None -> round((T-p)^(1/3)).

    Returns
    -------
    HeteroResult
    """
    rng = np.random.default_rng(seed)

    A_list, c, Sigma, residuals, _ = estimate_var(Y, p)
    T_eff, n = residuals.shape

    ri = np.asarray(regime_indicator, dtype=int)
    if ri.shape[0] == Y.shape[0]:
        ri = ri[p:]
    if ri.shape[0] != T_eff:
        raise ValueError(
            f"regime_indicator length ({ri.shape[0]}) must be T ({Y.shape[0]}) "
            f"or T-p ({T_eff})."
        )

    Sigma_0, Sigma_1 = _regime_covariances(residuals, ri)
    B, variance_ratios = _rigobon_impact(Sigma_0, Sigma_1)

    irfs = compute_irf(A_list, B, horizon)   # (H+1, n, n)
    fevd_arr = compute_fevd(A_list, B, horizon)  # (H+1, n, n)
    point = irfs

    lower = upper = None
    if n_boot > 0 and _HAS_MBB:
        def _impact_fn(Y_star: np.ndarray, p_: int, H: int) -> np.ndarray:
            A_b, _, _, resid_b, _ = estimate_var(Y_star, p_)
            T_b = resid_b.shape[0]
            ri_b = ri[:T_b] if T_b <= len(ri) else np.resize(ri, T_b)
            try:
                S0_b, S1_b = _regime_covariances(resid_b, ri_b)
                B_b, _ = _rigobon_impact(S0_b, S1_b)
                return compute_irf(A_b, B_b, H)   # (H+1, n, n)
            except (np.linalg.LinAlgError, ValueError):
                return irfs

        boot = moving_block_bootstrap(
            residuals=residuals,
            Y=Y,
            A_list=A_list,
            intercept=c,
            n_draws=n_boot,
            block_len=block_len,
            horizon=horizon,
            irf_fn=_impact_fn,
            rng=rng,
        )

        lo_q = (1 - ci) / 2 * 100
        hi_q = (1 - (1 - ci) / 2) * 100
        lower, _, upper = bootstrap_percentiles(
            boot["draws"], q_lo=lo_q, q_hi=hi_q
        )

    return HeteroResult(
        B=B,
        variance_ratios=variance_ratios,
        irfs=irfs,
        fevd=fevd_arr,
        lower=lower,
        upper=upper,
        point=point,
    )
