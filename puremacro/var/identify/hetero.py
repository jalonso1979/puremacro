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
import warnings
from typing import Optional

import numpy as np
import scipy.linalg

from ..estimate import estimate_var
from ..irf import irf as compute_irf, fevd as compute_fevd

try:
    # `inference.moving_block`, not the retired `inference.moving_block_bootstrap`
    # copy: they had drifted, and only this one's default IRF is statsmodels-free.
    from ...inference.moving_block import moving_block_bootstrap, bootstrap_percentiles
    _HAS_MBB = True
except ImportError:  # pragma: no cover — inference ships in the same wheel
    _HAS_MBB = False


#: Warn above this fraction of failed bootstrap draws. Same threshold and same
#: drop-and-warn pattern as ``var/identify/cholesky.py``.
_BOOT_FAIL_WARN_THRESHOLD = 0.05


def _canonical_signs(B: np.ndarray) -> np.ndarray:
    """Row vector of +/-1 making every diagonal element of ``B`` positive.

    ``scipy.linalg.eigh`` fixes each eigenvector only up to sign, so the same
    data can yield ``B`` or a column-flipped ``B`` run to run. Harmless for the
    point estimate on its own; fatal for a percentile band across draws.
    """
    s = np.sign(np.diag(B))
    s[s == 0] = 1.0
    return s


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
    # `eigh` pins each eigenvector only up to sign. Give the point estimate a
    # canonical one so bootstrap draws can be matched to it below; percentiles
    # taken across draws that disagree on sign are percentiles of a bimodal
    # mixture, not of a sampling distribution.
    B = B * _canonical_signs(B)

    irfs = compute_irf(A_list, B, horizon)   # (H+1, n, n)
    fevd_arr = compute_fevd(A_list, B, horizon)  # (H+1, n, n)
    point = irfs

    lower = upper = None
    if n_boot > 0 and _HAS_MBB:
        n_fail = 0

        def _impact_fn(Y_star: np.ndarray, p_: int, H: int,
                       idx_star: np.ndarray) -> Optional[np.ndarray]:
            nonlocal n_fail
            A_b, _, _, resid_b, _ = estimate_var(Y_star, p_)
            # The regime label must follow the residual it was drawn with.
            # This used to read `ri[:T_b]` -- calendar-order labels pasted onto
            # reshuffled blocks -- so within a draw each "regime" was a random
            # subset of the same mixed distribution. Both bootstrap covariances
            # then converged to the same matrix, the generalised eigenproblem
            # went near-degenerate, and the band was percentiles of arbitrary
            # rotations. On a DGP with true variance ratio 3.0 the point
            # estimate recovered 2.82 while the bootstrap draws averaged 1.14
            # and never once exceeded 1.50 in 500 draws.
            ri_b = ri[idx_star][:resid_b.shape[0]]
            try:
                S0_b, S1_b = _regime_covariances(resid_b, ri_b)
                B_b, _ = _rigobon_impact(S0_b, S1_b)
            except (np.linalg.LinAlgError, ValueError):
                # Drop and warn, per CONTRIBUTING.md; returning `irfs` here put
                # a point mass at the point estimate and narrowed the band.
                n_fail += 1
                return None
            B_b = B_b * _canonical_signs(B_b)
            # Match each column's sign to the point estimate before it enters
            # the percentile stack.
            sgn = np.sign(np.sum(B_b * B, axis=0))
            sgn[sgn == 0] = 1.0
            return compute_irf(A_b, B_b * sgn, H)   # (H+1, n, n)

        boot = moving_block_bootstrap(
            residuals=residuals,
            Y=Y,
            A_list=A_list,
            intercept=c,
            n_draws=n_boot,
            pass_index=True,
            block_len=block_len,
            horizon=horizon,
            irf_fn=_impact_fn,
            rng=rng,
        )

        if not boot["draws"]:
            raise np.linalg.LinAlgError(
                f"rigobon_svar: all {n_boot} bootstrap draws failed to "
                "identify. The two regimes may be too similar, or one of them "
                "too short to estimate a covariance from."
            )
        fail_rate = n_fail / n_boot
        if fail_rate > _BOOT_FAIL_WARN_THRESHOLD:
            warnings.warn(
                f"rigobon_svar: {n_fail}/{n_boot} bootstrap draws "
                f"({fail_rate:.1%}) failed to identify and were dropped. "
                "Bands are computed from the surviving draws and may be "
                "unreliable; consider more data or a sharper regime split.",
                stacklevel=2,
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
