"""Proxy-SVAR / external-instrument identification (Mertens-Ravn 2013, Stock-Watson 2018).

The 0.4.0 release migrates the return type from a 3-tuple (point, lo, hi) to a
:class:`ProxySVARResult` frozen dataclass that additionally carries the impact
matrix ``B`` and the Olea-Pflueger effective F.
"""
from __future__ import annotations

from typing import Optional, Callable, Any

import numpy as np

from ..estimate import estimate_var
from ..._linalg import safe_cholesky
from ._results import ProxySVARResult
from ...inference.weak_iv import olea_pflueger_f

_WildBootstrapFn = Optional[Callable[..., Any]]
wild_bootstrap_var: _WildBootstrapFn
try:
    from ...inference.wild_bootstrap import wild_bootstrap_var
except ImportError:
    wild_bootstrap_var = None


def _proxy_impact_factory(instrument_series, shock_target_idx=0):
    def _impact(A_list, Sigma, resid):
        T_eff = resid.shape[0]
        z = np.asarray(instrument_series)[-T_eff:]
        z = z - z.mean()
        # Under the proxy assumptions E[z eps_1] = phi != 0 and E[z eps_j] = 0,
        # Cov(u, z) = B E[eps z] = phi * b_1, so `Pi` is the impact column up to
        # an unknown scale -- it IS the direction, and needs only normalising.
        #
        # The normalisation is `b_1' Sigma^-1 b_1 = 1`, which follows from
        # Sigma = BB' => B' Sigma^-1 B = I. So with Pi = k*b_1,
        # Pi' Sigma^-1 Pi = k^2 and `Pi / sqrt(Pi' Sigma^-1 Pi)` is +/- b_1.
        #
        # This used to read `(Sigma @ Pi) / sqrt(Pi @ Sigma @ Pi)`, which is
        # proportional to `Sigma b_1` rather than to `b_1` -- the right vector
        # only when Sigma is proportional to the identity, i.e. when b_1 happens
        # to be an eigenvector of Sigma. Nothing caught it because that vector
        # still satisfies b' Sigma^-1 b = 1 exactly, so the SVD completion below
        # still returns BB' = Sigma to machine precision: the scale in the
        # Sigma^-1 metric was right and only the direction was wrong. At
        # T = 400,000 on a DGP with true b_1 = [1, 0.8, -0.5] it returned
        # [0.919, 1.049, -0.608], and converged there rather than to the truth.
        Pi = (resid.T @ z) / (z @ z)
        # Solve rather than invert, through the package's diagnostic factor.
        L = safe_cholesky(Sigma, name="proxy-SVAR Sigma")
        w = np.linalg.solve(L, Pi)          # w'w == Pi' Sigma^-1 Pi
        norm = float(np.sqrt(w @ w))
        if norm < 1e-10:
            raise np.linalg.LinAlgError("Degenerate instrument in proxy-SVAR.")
        B_col1 = Pi / norm
        n = Sigma.shape[0]
        B = np.zeros((n, n))
        B[:, 0] = B_col1
        residual_cov = Sigma - np.outer(B_col1, B_col1)
        u, s, _ = np.linalg.svd(residual_cov)
        rank = int(np.sum(s > 1e-8))
        for k in range(min(rank, n - 1)):
            B[:, 1 + k] = u[:, k] * np.sqrt(max(s[k], 0))
        return B
    return _impact


def proxy_svar(
    Y: np.ndarray,
    *,
    p: int,
    horizon: int,
    instrument_series: np.ndarray,
    shock_target_idx: int = 0,
    n_boot: int = 500,
    ci: float = 0.9,
    seed: int = 0,
) -> ProxySVARResult:
    """Proxy-SVAR identification via external instrument (Mertens-Ravn 2013).

    Parameters
    ----------
    Y : ndarray, shape (T, n)
        VAR data.
    p : int
        VAR lag order.
    horizon : int
        IRF horizon (returns ``horizon+1`` periods).
    instrument_series : ndarray, shape (T,)
        External instrument / proxy. The last ``T - p`` observations are aligned
        to the VAR residuals.
    shock_target_idx : int, default 0
        Index of the structural shock targeted by the proxy. The identified shock
        is placed in column 0 of ``B`` regardless.
    n_boot : int, default 500
        Number of wild-bootstrap draws.
    ci : float, default 0.9
        Confidence-interval level.
    seed : int, default 0
        Bootstrap RNG seed.

    Returns
    -------
    ProxySVARResult
        See :class:`puremacro.var.identify._results.ProxySVARResult`.
    """
    A_list, c, Sigma, resid, _ = estimate_var(Y, p)
    T_eff = resid.shape[0]
    z = np.asarray(instrument_series)[-T_eff:]
    # Olea-Pflueger F is computed on the first VAR residual against the proxy.
    f_eff = olea_pflueger_f(resid[:, shock_target_idx], z.reshape(-1, 1))

    impact_fn = _proxy_impact_factory(instrument_series, shock_target_idx)
    B = impact_fn(A_list, Sigma, resid)

    if wild_bootstrap_var is None:
        raise ImportError(
            "wild_bootstrap_var is not available. "
            "puremacro.inference.wild_bootstrap must be installed."
        )
    point, lo, hi = wild_bootstrap_var(
        Y, p=p, horizon=horizon, impact_fn=impact_fn,
        n_boot=n_boot, ci=ci, seed=seed,
    )
    # wild_bootstrap_var returns (n, n, H+1); canonical *Result convention is (H+1, n, n).
    # See puremacro/var/identify/_results.py for the project-wide axis convention.
    point = np.transpose(point, (2, 0, 1))
    lo = np.transpose(lo, (2, 0, 1))
    hi = np.transpose(hi, (2, 0, 1))
    return ProxySVARResult(
        irf_point=point,
        irf_lower=lo,
        irf_upper=hi,
        B=B,
        first_stage_F=float(f_eff),
        n_boot=n_boot,
        ci=ci,
    )
