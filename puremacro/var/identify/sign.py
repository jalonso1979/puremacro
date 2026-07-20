"""Rubio-Ramirez-Waggoner-Zha sign-restriction SVAR.

Acceptance: at each specified horizon, the sign of the impulse response of each
variable to the target shock (column 0 after rotation) must match the prior.
Entries set to 0 in the restriction vector are unrestricted.
"""

from __future__ import annotations

import numpy as np
from numpy.random import default_rng

from ..._linalg import safe_cholesky
from ..estimate import estimate_var
from ..irf import irf as compute_irf
from ._results import SignRestrictionResult


def _draw_orthogonal(n: int, rng) -> np.ndarray:
    """Haar-uniform draw from O(n) via QR of a Gaussian matrix."""
    A = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(A)
    # Fix signs so that Q is unique (Diagonal of R positive)
    D = np.diag(np.sign(np.diag(R)))
    return Q @ D


def _check_signs(ir: np.ndarray, restrictions: dict) -> bool:
    # ir shape: (H+1, n, n). Target shock is column (axis-2 index) 0.
    for h, sign_vec in restrictions.items():
        for i, s in enumerate(sign_vec):
            if s == 0:
                continue
            if np.sign(ir[h, i, 0]) != s:
                return False
    return True


def sign_restriction_svar(
    Y: np.ndarray,
    *,
    p: int,
    horizon: int,
    restrictions: dict,
    n_draws: int = 2000,
    ci: float = 0.9,
    seed: int = 0,
) -> SignRestrictionResult:
    rng = default_rng(seed)
    A_list, c, Sigma, resid, _ = estimate_var(Y, p)
    P = safe_cholesky(Sigma, name="sign_restriction_svar")
    n = Sigma.shape[0]

    accepted = []
    for _ in range(n_draws):
        R = _draw_orthogonal(n, rng)
        B = P @ R
        ir = compute_irf(A_list, B, horizon)  # (H+1, n, n)
        if _check_signs(ir, restrictions):
            accepted.append(ir)

    if len(accepted) == 0:
        raise RuntimeError(
            f"No draws satisfied the sign restrictions out of {n_draws}. "
            "Try relaxing the prior or increasing n_draws."
        )
    draws = np.stack(accepted, axis=0)  # (n_accepted, H+1, n, n)
    lo_q = (1 - ci) / 2
    hi_q = 1 - lo_q
    return SignRestrictionResult(
        irf_median=np.median(draws, axis=0),
        irf_lower=np.quantile(draws, lo_q, axis=0),
        irf_upper=np.quantile(draws, hi_q, axis=0),
        n_draws=n_draws,
        n_accepted=len(accepted),
        ci=ci,
    )
