"""Proxy-SVAR / external-instrument identification (Mertens-Ravn 2013, Stock-Watson 2018).

Bootstrap confidence intervals default to the centered moving-block bootstrap
(MBB) of Jentsch & Lunsford (2019, AER; 2022, RESTAT), which jointly resamples
blocks of reduced-form VAR residuals and proxy innovations to capture first-stage
and covariance estimation uncertainty.

The 0.4.0 release migrates the return type from a 3-tuple (point, lo, hi) to a
:class:`ProxySVARResult` frozen dataclass that additionally carries the impact
matrix ``B`` and the Olea-Pflueger effective F.
"""
from __future__ import annotations

import math
import warnings
from typing import Optional, Callable, Any

import numpy as np
from numpy.random import default_rng

from ..estimate import estimate_var
from ..._linalg import safe_cholesky
from ._results import ProxySVARResult
from ...inference.bootstrap import _irf_from_var, _ols_var
from ...inference.weak_iv import olea_pflueger_f

_WildBootstrapFn = Optional[Callable[..., Any]]
wild_bootstrap_var: _WildBootstrapFn
try:
    from ...inference.wild_bootstrap import (
        _BOOT_FAIL_WARN_THRESHOLD,
        wild_bootstrap_var,
    )
except ImportError:
    wild_bootstrap_var = None
    _BOOT_FAIL_WARN_THRESHOLD = 0.05


def _identify_proxy_impact(
    Sigma: np.ndarray,
    resid: np.ndarray,
    z_vec: np.ndarray,
) -> np.ndarray:
    """Identify impact matrix ``B`` with proxy-identified shock in column 0."""
    valid = np.isfinite(z_vec)
    if int(valid.sum()) < 2:
        raise np.linalg.LinAlgError(
            "Degenerate instrument in proxy-SVAR: fewer than 2 finite proxy observations."
        )
    z_sub = z_vec[valid]
    z_c = z_sub - z_sub.mean()
    zz = float(z_c @ z_c)
    if not np.isfinite(zz) or zz <= 1e-24:
        raise np.linalg.LinAlgError("Degenerate instrument in proxy-SVAR.")
    resid_sub = resid[valid]
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
    Pi = (resid_sub.T @ z_c) / zz
    # Solve rather than invert, through the package's diagnostic factor.
    L = safe_cholesky(Sigma, name="proxy-SVAR Sigma")
    v = np.linalg.solve(L, Pi)          # v'v == Pi' Sigma^-1 Pi
    norm = float(np.sqrt(v @ v))
    if not np.isfinite(norm) or norm < 1e-10:
        raise np.linalg.LinAlgError("Degenerate instrument in proxy-SVAR.")
    B_col1 = Pi / norm
    n = Sigma.shape[0]
    B = np.zeros((n, n), dtype=float)
    B[:, 0] = B_col1
    residual_cov = Sigma - np.outer(B_col1, B_col1)
    u, s, _ = np.linalg.svd(residual_cov)
    rank = int(np.sum(s > 1e-8))
    for k in range(min(rank, n - 1)):
        B[:, 1 + k] = u[:, k] * np.sqrt(max(s[k], 0.0))
    return B


def _proxy_impact_factory(instrument_series, shock_target_idx=0):
    def _impact(A_list, Sigma, resid, w=None):
        T_eff = resid.shape[0]
        z = np.asarray(instrument_series, dtype=float).reshape(-1)[-T_eff:]
        if w is not None:
            # Wild-bootstrap draw: re-sign the proxy with the same Rademacher
            # weights applied to the residuals (Mertens & Ravn 2013). Without
            # this the draw-level covariance Cov(w*u, z) has a random sign and
            # the bands collapse onto zero.
            z = z * np.asarray(w, dtype=float)[-T_eff:]
        return _identify_proxy_impact(Sigma, resid, z)
    return _impact


def _mbb_bootstrap_proxy_svar(
    Y_arr: np.ndarray,
    *,
    p: int,
    horizon: int,
    A_list: list[np.ndarray],
    c: np.ndarray,
    resid: np.ndarray,
    z_eff: np.ndarray,
    B_point: np.ndarray,
    block_len: int,
    n_boot: int,
    ci: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centered moving-block bootstrap for proxy-SVAR (Jentsch & Lunsford 2019, 2022).

    Returns ``(point, lo, hi)`` each of canonical shape ``(horizon + 1, n, n)``.
    """
    T, n = Y_arr.shape
    T_eff = resid.shape[0]
    point = np.transpose(_irf_from_var(A_list, horizon, B_point), (2, 0, 1))

    n_blocks_avail = T_eff - block_len + 1
    u_blocks = np.empty((n_blocks_avail, block_len, n), dtype=float)
    z_blocks = np.empty((n_blocks_avail, block_len), dtype=float)
    for idx in range(n_blocks_avail):
        u_blocks[idx] = resid[idx : idx + block_len]
        z_blocks[idx] = z_eff[idx : idx + block_len]

    # Position-wise centering across overlapping blocks so E*[u*_t] = 0 and E*[z*_t] = 0
    # at every block offset s in {0, ..., block_len - 1} (Jentsch & Lunsford 2019, p. 2663).
    u_blocks -= u_blocks.mean(axis=0, keepdims=True)
    if np.isfinite(z_blocks).all():
        z_blocks -= z_blocks.mean(axis=0, keepdims=True)
    else:
        overall_z_mean = float(np.nanmean(z_eff))
        for s_off in range(block_len):
            col = z_blocks[:, s_off]
            fin = np.isfinite(col)
            n_fin = int(fin.sum())
            if n_fin >= 2:
                col[fin] = col[fin] - float(np.mean(col[fin]))
            elif n_fin == 1:
                col[fin] = col[fin] - overall_z_mean

    rng = default_rng(seed)
    n_draw_blocks = (T_eff + block_len - 1) // block_len
    blk_idx = rng.integers(0, n_blocks_avail, size=(n_boot, n_draw_blocks))
    E_all = u_blocks[blk_idx].reshape(n_boot, n_draw_blocks * block_len, n)[:, :T_eff, :]
    Z_all = z_blocks[blk_idx].reshape(n_boot, n_draw_blocks * block_len)[:, :T_eff]

    A_stack = np.hstack(A_list)
    Yb_all = np.empty((n_boot, T, n), dtype=float)
    Yb_all[:, :p, :] = Y_arr[:p]
    for t in range(p, T):
        lags_mat = Yb_all[:, t - p : t, :][:, ::-1, :].reshape(n_boot, p * n)
        Yb_all[:, t, :] = lags_mat @ A_stack.T + c + E_all[:, t - p, :]

    accepted: list[np.ndarray] = []
    for b_idx in range(n_boot):
        try:
            A_b, _, Sigma_b, resid_b, _ = _ols_var(Yb_all[b_idx], p)
            B_b = _identify_proxy_impact(Sigma_b, resid_b, Z_all[b_idx])
            irf_b = np.transpose(_irf_from_var(A_b, horizon, B_b), (2, 0, 1))
            accepted.append(irf_b)
        except np.linalg.LinAlgError:
            continue

    n_fail = n_boot - len(accepted)
    if not accepted:
        raise np.linalg.LinAlgError(
            f"proxy_svar: all {n_boot} bootstrap draws failed identification. "
            "Likely causes: a weak proxy/instrument, too few observations for "
            "n*p coefficients, or a singular reduced-form Sigma."
        )
    fail_rate = n_fail / n_boot
    if fail_rate > _BOOT_FAIL_WARN_THRESHOLD:
        warnings.warn(
            f"proxy_svar: {n_fail}/{n_boot} bootstrap draws ({fail_rate:.1%}) "
            "failed identification and were dropped. Bands are computed from "
            "the surviving draws and may be unreliable; consider more data, "
            "fewer lags, or a stronger instrument.",
            RuntimeWarning,
            stacklevel=2,
        )

    draws = np.stack(accepted, axis=0)
    lo_q = (1.0 - ci) / 2.0
    hi_q = 1.0 - lo_q
    lo = np.quantile(draws, lo_q, axis=0)
    hi = np.quantile(draws, hi_q, axis=0)
    return point, lo, hi


def proxy_svar(
    Y: np.ndarray,
    instrument_series: np.ndarray | None = None,
    *,
    p: int | None = None,
    horizon: int = 20,
    shock_target_idx: int = 0,
    n_boot: int = 500,
    ci: float = 0.9,
    seed: int = 0,
    lags: int | None = None,
    sign_var: int = 0,
    sign: float | None = None,
    z: np.ndarray | None = None,
    bootstrap: str = "mbb",
    block_size: int | None = None,
) -> ProxySVARResult:
    """Proxy-SVAR identification via external instrument (Mertens-Ravn 2013).

    Confidence bands default to the centered moving-block bootstrap (``bootstrap="mbb"``)
    of Jentsch & Lunsford (2019, 2022), which is asymptotically valid for proxy-SVAR
    impulse responses under conditional heteroskedasticity and weak serial dependence.

    Parameters
    ----------
    Y : ndarray, shape (T, n)
        VAR data.
    instrument_series : ndarray, shape (T,)
        External instrument / proxy. May contain ``NaN`` on periods where the
        proxy is unavailable; at least two finite post-lag observations are required.
    p : int
        VAR lag order (or pass ``lags=...``).
    horizon : int
        IRF horizon (returns ``horizon+1`` periods).
    shock_target_idx : int, default 0
        Column of ``Y`` whose reduced-form VAR residual enters the proxy's
        first stage: ``first_stage_F`` is the Olea-Pflueger effective F of
        ``resid[:, shock_target_idx]`` regressed on the proxy. Set it to the
        variable the proxy is meant to move (for a monetary proxy, the
        policy-rate column); with the default 0 and the policy variable in
        another column, ``first_stage_F`` is the F of the wrong equation.
        It does not change ``B`` or the impulse responses: the impact column
        is identified from ``Cov(u, z)`` across all residuals and is placed
        in column 0 of ``B`` regardless.
    n_boot : int, default 500
        Number of bootstrap draws.
    ci : float, default 0.9
        Confidence-interval level in ``(0, 1)``.
    seed : int, default 0
        Bootstrap RNG seed.
    lags : int, optional
        Alias for ``p``.
    sign_var : int, default 0
        Variable index for sign normalization.
    sign : float, optional
        Target sign for ``B[sign_var, 0]`` (e.g., ``-1`` for contractionary shock).
    z : ndarray, optional
        Alias for ``instrument_series``.
    bootstrap : {"mbb", "iid", "wild"}, default "mbb"
        Bootstrap scheme for confidence bands:
        - ``"mbb"``: Centered moving-block bootstrap of Jentsch & Lunsford
          (2019, AER; 2022, RESTAT) on joint ``(resid, z)`` pairs.
        - ``"iid"``: Centered i.i.d. pairs bootstrap (equivalent to MBB with
          ``block_size=1``).
        - ``"wild"``: Legacy Mertens-Ravn (2013) Rademacher wild bootstrap
          (emits ``UserWarning``; inconsistent for proxy-SVAR CIs per
          Jentsch & Lunsford 2019).
    block_size : int, optional
        Block length ``ell`` for ``bootstrap="mbb"`` satisfying
        ``1 <= block_size < T_eff``. When ``None``, uses the Jentsch & Lunsford
        rule-of-thumb ``floor(5.03 * T_eff**0.25)`` clamped to ``[1, T_eff - 1]``.

    Returns
    -------
    ProxySVARResult
        See :class:`puremacro.var.identify._results.ProxySVARResult`.
    """
    if lags is not None:
        p = lags
    if p is None:
        p = 2
    if z is not None:
        instrument_series = z
    if instrument_series is None:
        raise ValueError("instrument_series is required for proxy_svar.")
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or int(horizon) < 0:
        raise ValueError(f"horizon must be a non-negative integer; got {horizon!r}")
    horizon = int(horizon)
    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)) or int(n_boot) < 1:
        raise ValueError(f"n_boot must be an integer >= 1; got {n_boot!r}")
    n_boot = int(n_boot)
    if not isinstance(ci, (int, float, np.floating)) or isinstance(ci, bool) or not (0.0 < float(ci) < 1.0):
        raise ValueError(f"ci must lie strictly in (0, 1); got {ci!r}")
    ci = float(ci)
    if not isinstance(bootstrap, str):
        raise ValueError(f"bootstrap must be one of 'mbb', 'iid', or 'wild'; got {bootstrap!r}")
    boot_mode = bootstrap.strip().lower()
    if boot_mode not in ("mbb", "iid", "wild"):
        raise ValueError(f"bootstrap must be one of 'mbb', 'iid', or 'wild'; got {bootstrap!r}")
    if sign is not None:
        if isinstance(sign, bool) or not np.isfinite(float(sign)) or float(sign) == 0.0:
            raise ValueError(f"sign must be a finite non-zero float when provided; got {sign!r}")

    Y_arr = np.asarray(Y, dtype=float)
    A_list, c, Sigma, resid, _ = estimate_var(Y, p)
    T_eff, n = resid.shape
    if T_eff < 2:
        raise ValueError(f"proxy_svar requires T - p >= 2 effective observations; got {T_eff}")
    if not (0 <= shock_target_idx < n):
        raise ValueError(f"shock_target_idx={shock_target_idx} out of bounds for n={n} variables")
    if not (0 <= sign_var < n):
        raise ValueError(f"sign_var={sign_var} out of bounds for n={n} variables")

    z_raw = np.asarray(instrument_series, dtype=float)
    if z_raw.ndim == 2 and z_raw.shape[1] == 1:
        z_raw = z_raw[:, 0]
    elif z_raw.ndim != 1:
        raise ValueError(f"instrument_series must be 1-D (or (T, 1)); got shape {z_raw.shape}")
    if z_raw.shape[0] < T_eff:
        raise ValueError(
            f"instrument_series length ({z_raw.shape[0]}) is shorter than effective VAR sample T_eff={T_eff}."
        )
    z_eff = z_raw[-T_eff:]
    if np.isinf(z_eff).any():
        raise ValueError("instrument_series contains +/-inf values.")
    valid_z = np.isfinite(z_eff)
    if int(valid_z.sum()) < 2:
        raise ValueError("instrument_series must have at least 2 finite observations in the effective VAR sample.")

    if block_size is not None:
        if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)):
            raise ValueError(f"block_size must be an integer when provided; got {block_size!r}")
        ell = int(block_size)
        if boot_mode == "iid" and ell != 1:
            raise ValueError(f"block_size must be None or 1 when bootstrap='iid'; got {block_size}")
        if not (1 <= ell < T_eff):
            raise ValueError(f"block_size must satisfy 1 <= block_size < T_eff ({T_eff}); got {block_size}")
    else:
        if boot_mode == "iid":
            ell = 1
        else:
            ell = min(max(1, T_eff - 1), max(1, int(math.floor(5.03 * (T_eff ** 0.25)))))

    # Olea-Pflueger effective F of the residual of variable shock_target_idx
    # regressed on the proxy (the first stage). It only feeds first_stage_F.
    f_eff = olea_pflueger_f(resid[valid_z, shock_target_idx], z_eff[valid_z].reshape(-1, 1))

    B = _identify_proxy_impact(Sigma, resid, z_eff)

    if boot_mode in ("mbb", "iid"):
        point, lo, hi = _mbb_bootstrap_proxy_svar(
            Y_arr,
            p=p,
            horizon=horizon,
            A_list=A_list,
            c=c,
            resid=resid,
            z_eff=z_eff,
            B_point=B,
            block_len=ell,
            n_boot=n_boot,
            ci=ci,
            seed=seed,
        )
    else:
        warnings.warn(
            "proxy_svar(..., bootstrap='wild'): Rademacher wild bootstrap is "
            "inconsistent for proxy-SVAR confidence intervals because w_t^2 = 1 "
            "leaves sample Cov(u*, z*) and Sigma* invariant up to O_p(T^-1) "
            "(Jentsch & Lunsford 2019, AER; 2022, RESTAT). Use bootstrap='mbb' "
            "(the default) for asymptotically valid inference.",
            UserWarning,
            stacklevel=2,
        )
        if wild_bootstrap_var is None:
            raise ImportError(
                "wild_bootstrap_var is not available. "
                "puremacro.inference.wild_bootstrap must be installed."
            )
        impact_fn = _proxy_impact_factory(z_eff, shock_target_idx)
        point_w, lo_w, hi_w = wild_bootstrap_var(
            Y_arr, p=p, horizon=horizon, impact_fn=impact_fn,
            n_boot=n_boot, ci=ci, seed=seed,
        )
        # wild_bootstrap_var returns (n, n, H+1); canonical *Result convention is (H+1, n, n).
        point = np.transpose(point_w, (2, 0, 1))
        lo = np.transpose(lo_w, (2, 0, 1))
        hi = np.transpose(hi_w, (2, 0, 1))

    if sign is not None and np.sign(B[sign_var, 0]) != np.sign(sign):
        B = B.copy()
        point = point.copy()
        lo = lo.copy()
        hi = hi.copy()
        B[:, 0] = -B[:, 0]
        point[:, :, 0] = -point[:, :, 0]
        lo_new = -hi[:, :, 0]
        hi_new = -lo[:, :, 0]
        lo[:, :, 0] = lo_new
        hi[:, :, 0] = hi_new
    return ProxySVARResult(
        irf_point=point,
        irf_lower=lo,
        irf_upper=hi,
        B=B,
        first_stage_F=float(f_eff),
        n_boot=n_boot,
        ci=ci,
    )
