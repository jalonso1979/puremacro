"""Bootstrap confidence bands for VAR-based IRFs."""
from __future__ import annotations

import numpy as np

from .estimate import estimate_var
from .irf import irf


def _irf_from_var(Y, p, identify_fn, horizon, **id_kwargs):
    """Helper: fit VAR + identify B0 + compute IRF in one call.

    `identify_fn` is callable as identify_fn(A_list, Sigma, **id_kwargs).
    """
    A_list, _, Sigma, residuals, _ = estimate_var(Y, p)
    B0 = identify_fn(A_list, Sigma, **id_kwargs)
    return irf(A_list, B0, horizon)


def _kilian_bias_correct(Y, p, n_pilot=100, rng=None):
    """Kilian (1998) bootstrap bias correction for VAR coefficients.

    Estimate finite-sample bias of (intercept, A_list) by simulating
    ``n_pilot`` recursive bootstrap samples from the observed VAR,
    re-estimating, and averaging the deviation. Subtract the bias from
    the original estimate and shrink towards the original if the
    bias-corrected companion is non-stationary.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    A_list, intercept, Sigma, residuals, _ = estimate_var(Y, p)
    Y_arr = np.asarray(Y)
    T, n = Y_arr.shape
    bias_int = np.zeros_like(intercept)
    bias_A = [np.zeros_like(A) for A in A_list]
    for _ in range(n_pilot):
        idx = rng.integers(0, len(residuals), size=len(residuals))
        u_b = residuals[idx]
        Y_b = np.copy(Y_arr)
        for t in range(p, T):
            yt = intercept.copy()
            for l in range(p):
                yt += A_list[l] @ Y_b[t - l - 1]
            Y_b[t] = yt + u_b[t - p]
        try:
            A_b, c_b, _, _, _ = estimate_var(Y_b, p)
        except np.linalg.LinAlgError:
            continue
        bias_int += (c_b - intercept) / n_pilot
        for k in range(p):
            bias_A[k] += (A_b[k] - A_list[k]) / n_pilot
    A_bc = [A - b for A, b in zip(A_list, bias_A)]
    int_bc = intercept - bias_int
    # Stability shrinkage (Kilian 1998 sec. 3): if companion eigenvalues
    # exceed 1, shrink the bias correction toward zero by delta in (0,1].
    from .estimate import companion
    delta = 1.0
    while delta > 0.0:
        Cc = companion(A_bc)
        eigs = np.abs(np.linalg.eigvals(Cc))
        if eigs.max() < 0.999:
            break
        delta -= 0.01
        A_bc = [A - delta * b for A, b in zip(A_list, bias_A)]
        int_bc = intercept - delta * bias_int
    return A_bc, int_bc


def bootstrap_bands(Y, p, identify_fn, horizon, n_boot=500, alpha=0.10,
                    method="recursive", rng=None, bias_correct=False,
                    n_pilot=100, band="pointwise", **id_kwargs):
    """Bootstrap IRF percentile bands.

    method: 'recursive' (residual recursive bootstrap; Kilian-Lutkepohl S12),
            'wild' (Mammen 1993),
            'block' (moving block bootstrap).

    bias_correct: if True, apply Kilian (1998) bias-corrected bootstrap.
                  The pilot stage uses ``n_pilot`` recursive draws to
                  estimate bias; the main bootstrap then resamples around
                  the bias-corrected DGP.

    band: 'pointwise' (default; per-horizon percentile bands, the historical
          behavior) or 'sup-t' (Montiel Olea & Plagborg-Møller 2019, Journal
          of Applied Econometrics 34(1): simultaneous bands via the
          quantile-of-max-studentized-deviation over the bootstrap draws,
          applied per (response, shock) pair across horizons). With
          'sup-t' the band is centered on the bootstrap median (the same
          "point" this function has always reported — a documented
          simplification of the paper's Algorithm 2, which centers on the
          original-sample estimate), horizons whose draws are exactly
          degenerate (e.g. Cholesky impact zeros) collapse to the point,
          and the returned dict gains two extra keys: ``band`` (str) and
          ``crit_value`` (n x n array of per-pair sup-t critical values;
          NaN where every horizon is degenerate). The 'pointwise' return
          contract is unchanged.
    """
    if band not in ("pointwise", "sup-t"):
        raise ValueError(f"unknown band {band!r}; use 'pointwise' or 'sup-t'")
    if rng is None:
        rng = np.random.default_rng(42)
    if bias_correct:
        A_list, intercept = _kilian_bias_correct(Y, p, n_pilot=n_pilot, rng=rng)
        # Recompute residuals under the bias-corrected DGP for resampling.
        Y_arr = np.asarray(Y)
        T_arr, n_arr = Y_arr.shape
        residuals = np.zeros((T_arr - p, n_arr))
        for t in range(p, T_arr):
            yt = intercept.copy()
            for l in range(p):
                yt += A_list[l] @ Y_arr[t - l - 1]
            residuals[t - p] = Y_arr[t] - yt
        Sigma = (residuals.T @ residuals) / max(1, residuals.shape[0] - 1 - n_arr * p)
    else:
        A_list, intercept, Sigma, residuals, _ = estimate_var(Y, p)
    Y_arr = np.asarray(Y)
    T, n = Y_arr.shape
    irfs = np.zeros((n_boot, horizon + 1, n, n))
    A_stack = np.hstack(A_list)          # (n, n*p), lags newest-first
    for b in range(n_boot):
        if method == "recursive":
            idx = rng.integers(0, len(residuals), size=len(residuals))
            u_b = residuals[idx]
        elif method == "wild":
            xi = rng.choice([-1.0, 1.0], size=len(residuals))
            u_b = residuals * xi[:, None]
        elif method == "block":
            block = max(1, int(round(T ** (1 / 3))))
            n_blocks = int(np.ceil(len(residuals) / block))
            starts = rng.integers(0, len(residuals) - block + 1, size=n_blocks)
            u_b = np.concatenate([residuals[s:s + block] for s in starts])[:len(residuals)]
        else:
            raise ValueError(f"unknown method {method!r}")
        Y_b = np.copy(Y_arr)
        # One (n, n*p) matmul per period instead of p separate (n, n) ones.
        # The stacked coefficient matrix is loop-invariant, so it is built once
        # above the draw loop; the per-period cost was 61-76%% of the bootstrap.
        # `[t-1 : t-1-p : -1]` is lags 1..p newest-first, matching the order the
        # columns of `A_stack` were concatenated in.
        for t in range(p, T):
            Y_b[t] = (intercept
                      + A_stack @ Y_b[t - 1:t - 1 - p if t - 1 - p >= 0 else None:-1].ravel()
                      + u_b[t - p])
        try:
            A_b, _, Sigma_b, _, _ = estimate_var(Y_b, p)
            B_b = identify_fn(A_b, Sigma_b, **id_kwargs)
            irfs[b] = irf(A_b, B_b, horizon)
        except Exception:
            irfs[b] = np.nan
    point = np.nanpercentile(irfs, 50, axis=0)
    if band == "pointwise":
        lower = np.nanpercentile(irfs, alpha * 100 / 2, axis=0)
        upper = np.nanpercentile(irfs, 100 - alpha * 100 / 2, axis=0)
        return {"point": point, "lower": lower, "upper": upper, "draws": irfs}

    # band == 'sup-t': joint band per (response, shock) pair across horizons.
    from puremacro.inference.supt import supt_band  # lazy: avoid import cycle

    finite = ~np.isnan(irfs).any(axis=(1, 2, 3))
    good = irfs[finite]
    if good.shape[0] < 20:
        raise ValueError(
            f"sup-t band needs >= 20 fully-finite bootstrap draws; got "
            f"{good.shape[0]} of {n_boot} (failed re-estimations produce NaN "
            f"draws — increase n_boot or inspect the identification)"
        )
    lower = np.empty_like(point)
    upper = np.empty_like(point)
    crit = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            d = good[:, :, i, j]                      # (n_good, horizon+1)
            # Degenerate horizons (e.g. Cholesky impact restrictions hold in
            # EVERY draw) collapse to the point instead of studentizing 0/0;
            # relative threshold matches supt_band's degeneracy check.
            tiny = 1e-12 * np.maximum(1.0, np.abs(d).max(axis=0))
            active = d.std(axis=0, ddof=1) > tiny
            lower[:, i, j] = point[:, i, j]
            upper[:, i, j] = point[:, i, j]
            if active.any():
                res = supt_band(theta=point[active, i, j],
                                draws=d[:, active],
                                method="bootstrap", alpha=alpha)
                lower[active, i, j] = res.lower
                upper[active, i, j] = res.upper
                crit[i, j] = res.crit_value
    return {"point": point, "lower": lower, "upper": upper, "draws": irfs,
            "band": "sup-t", "crit_value": crit}
