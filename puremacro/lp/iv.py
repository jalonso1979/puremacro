"""Single-country LP-IV (Stock-Watson 2018; Plagborg-Møller-Wolf 2021).

First stage: x_t = π_z' z_t + π_w' W_t + ν_t.
Second stage: y_{t+h} - y_{t-1} = α_h + β_h x_t + γ' W_t + ε_{t,h}
              with x_t replaced by its first-stage projection x̂_t.
HAC SE via Newey-West with bandwidth L = h + 1.

Provides Montiel Olea & Pflueger (2013) effective F-statistic (F_eff)
and weak-IV robust confidence sets (Anderson-Rubin / MOP).
"""
from __future__ import annotations

import warnings
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import chi2, ncx2, norm

from ..inference._ols_helpers import ols_hac, tsls_hac
from ._common import check_regular_index, cumulative_windows, resolve_lp_kwargs


def mop_critical_values(
    k_z: int,
    alpha: float = 0.05,
    *,
    tau: float | Sequence[float] = (0.10, 0.20),
) -> tuple[float, ...]:
    """Montiel Olea & Pflueger (2013) critical values for the effective F-statistic.

    The MOP test rejects "the worst-case Nagar bias of TSLS exceeds ``tau``"
    when ``F_eff`` exceeds the critical value. In the simplified case of
    MOP Section 4 (first-stage covariance ``W`` proportional to the
    identity, which always holds for ``k_z = 1``) the critical value has the
    closed form behind their published table,

        cv(k_z, tau, alpha) = chi2_{k_z, 1 - alpha}(k_z / tau) / k_z,

    the ``1 - alpha`` quantile of a non-central chi-square with ``k_z``
    degrees of freedom and non-centrality ``k_z / tau``, divided by ``k_z``.
    For ``k_z = 1`` at ``alpha = 0.05`` this reproduces the published MOP
    values 37.42 (``tau`` = 5%), 23.11 (10%), 15.06 (20%) and 12.04 (30%),
    and the 23.1 cutoff quoted by :func:`puremacro.inference.weak_iv.olea_pflueger_f`.

    For ``k_z >= 2`` the full MOP critical value is data dependent: it
    replaces ``k_z`` by an effective degrees of freedom ``K_eff`` computed from
    the estimated first-stage HAC covariance (MOP Section 4, Patnaik
    approximation). The value returned here is the ``W = I`` reduction and is
    not that generalised critical value.

    Parameters
    ----------
    k_z : int
        Number of excluded instruments (``>= 1``).
    alpha : float, default 0.05
        Significance level of the weak-instrument test, in (0, 1).
    tau : float or sequence of float, keyword-only, default (0.10, 0.20)
        Worst-case relative bias threshold(s), each in (0, 1). MOP tabulate
        0.05, 0.10, 0.20 and 0.30.

    Returns
    -------
    tuple of float
        One critical value per entry of ``tau``; with the default this is
        ``(cv_10, cv_20)``.

    References
    ----------
    Montiel Olea, J.L. and Pflueger, C. (2013). A robust test for weak
        instruments. Journal of Business & Economic Statistics 31(3), 358-369.
    """
    k = int(k_z)
    if k < 1:
        raise ValueError(f"mop_critical_values: k_z must be >= 1, got {k_z}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"mop_critical_values: alpha must be in (0, 1), got {alpha}")
    taus = (float(tau),) if isinstance(tau, (int, float, np.number)) else tuple(float(t) for t in tau)
    if not taus:
        raise ValueError("mop_critical_values: tau must contain at least one threshold")
    for t in taus:
        if not (0.0 < t < 1.0):
            raise ValueError(f"mop_critical_values: tau must be in (0, 1), got {t}")
    return tuple(float(ncx2.ppf(1.0 - alpha, df=k, nc=k / t)) / k for t in taus)


def compute_mop_effective_f(
    x: np.ndarray,
    Z: np.ndarray,
    W_ctl: np.ndarray,
    lags: int,
) -> float:
    """Compute Montiel Olea & Pflueger (2013) effective F-statistic with Newey-West HAC errors."""
    x_arr = np.asarray(x, dtype=float).ravel()
    Z_arr = np.asarray(Z, dtype=float)
    if Z_arr.ndim == 1:
        Z_arr = Z_arr[:, None]
    W_ctl_arr = np.asarray(W_ctl, dtype=float)

    # Partial out controls W_ctl
    W_inv = np.linalg.pinv(W_ctl_arr.T @ W_ctl_arr)
    x_tilde = x_arr - W_ctl_arr @ (W_inv @ (W_ctl_arr.T @ x_arr))
    Z_tilde = Z_arr - W_ctl_arr @ (W_inv @ (W_ctl_arr.T @ Z_arr))

    ZtZ = Z_tilde.T @ Z_tilde
    ZtZ_inv = np.linalg.pinv(ZtZ)
    pi_hat = ZtZ_inv @ (Z_tilde.T @ x_tilde)
    v_hat = x_tilde - Z_tilde @ pi_hat

    scores = Z_tilde * v_hat[:, None]
    Omega_zz = scores.T @ scores
    for ell in range(1, lags + 1):
        weight = 1.0 - ell / (lags + 1.0)
        gamma_ell = scores[ell:].T @ scores[:-ell]
        Omega_zz += weight * (gamma_ell + gamma_ell.T)

    num = float(pi_hat.T @ ZtZ @ pi_hat)
    denom = float(np.trace(ZtZ_inv @ Omega_zz))
    if denom <= 0:
        return float("inf")
    return float(num / denom)


def _compute_anderson_rubin_ci(
    y_target: np.ndarray,
    x: np.ndarray,
    W_mat: np.ndarray,
    lags: int,
    alpha: float,
) -> tuple[float, float, str]:
    """Exact closed-form Anderson-Rubin (1949) confidence interval for LP-IV."""
    MtM_inv = np.linalg.pinv(W_mat.T @ W_mat)
    w_z = MtM_inv[1, :]

    res_y = ols_hac(y_target, W_mat, lags=lags)
    gamma_y = float(res_y["beta"][1])
    u_y = y_target - W_mat @ res_y["beta"]

    res_x = ols_hac(x, W_mat, lags=lags)
    gamma_x = float(res_x["beta"][1])
    u_x = x - W_mat @ res_x["beta"]

    h_vec = W_mat @ w_z
    score_y = h_vec * u_y
    score_x = h_vec * u_x

    def _nw_var(s1: np.ndarray, s2: np.ndarray) -> float:
        v = float(np.dot(s1, s2))
        for ell in range(1, lags + 1):
            weight = 1.0 - ell / (lags + 1.0)
            gamma_ell = float(np.dot(s1[ell:], s2[:-ell]) + np.dot(s2[ell:], s1[:-ell]))
            v += weight * gamma_ell
        return v

    V_yy = _nw_var(score_y, score_y)
    V_xx = _nw_var(score_x, score_x)
    V_yx = _nw_var(score_y, score_x)

    crit = float(chi2.ppf(1.0 - alpha, df=1))

    A = gamma_x ** 2 - crit * V_xx
    B = -2.0 * (gamma_y * gamma_x - crit * V_yx)
    C = gamma_y ** 2 - crit * V_yy

    disc = B ** 2 - 4.0 * A * C
    if A > 0:
        if disc >= 0:
            r1 = (-B - np.sqrt(disc)) / (2.0 * A)
            r2 = (-B + np.sqrt(disc)) / (2.0 * A)
            return float(min(r1, r2)), float(max(r1, r2)), "bounded"
        else:
            return np.nan, np.nan, "empty"
    else:
        if disc >= 0:
            r1 = (-B - np.sqrt(disc)) / (2.0 * A)
            r2 = (-B + np.sqrt(disc)) / (2.0 * A)
            return float(max(r1, r2)), float(min(r1, r2)), "unbounded_rays"
        else:
            return -np.inf, np.inf, "all_real"


def _bartlett_lrv(A: np.ndarray, B: np.ndarray, lags: int) -> np.ndarray:
    """Bartlett-kernel long-run cross-covariance of score matrices ``A`` and ``B``.

    ``A'B + sum_{l=1}^{lags} (1 - l/(lags+1)) (A[l:]'B[:-l] + A[:-l]'B[l:])``;
    with ``A = B`` this is the Newey-West ``Omega`` used by :func:`ols_hac`,
    and ``lags = 0`` gives the Eicker-Huber-White ``A'B``.
    """
    out = A.T @ B
    for ell in range(1, lags + 1):
        weight = 1.0 - ell / (lags + 1.0)
        out = out + weight * (A[ell:].T @ B[:-ell] + A[:-ell].T @ B[ell:])
    return out


def _invert_ar_statistic(
    ar_stat: Callable[[np.ndarray], np.ndarray],
    crit: float,
    stat_inf: float,
    beta_hat: float,
    se_hat: float,
    *,
    extra_seeds: Sequence[float] = (),
    max_roots: int | None = None,
    n_grid: int = 401,
    max_expand: int = 24,
    outer_decades: int = 8,
) -> tuple[float, float, str]:
    """Invert ``{b : ar_stat(b) <= crit}`` exactly (to root-finding tolerance).

    ``ar_stat`` evaluates the statistic on an array of candidate ``b``.
    ``stat_inf`` is its closed-form limit as ``|b| -> inf``; it decides whether
    the tails are rejected (set bounded or empty) or accepted (two rays or the
    whole line). ``beta_hat``/``se_hat`` seed the search window, and
    ``extra_seeds`` (the per-instrument IV ratios) are tried when the window
    around the seed finds nothing even after ``max_expand`` doublings, so a
    poor 2SLS estimate cannot change the answer. ``max_roots`` is the largest
    number of crossings of ``crit`` the statistic can have (``2 k_z`` for the
    Anderson-Rubin statistic, whose inequality is rational of that degree);
    the scan stops as soon as that many are found. Otherwise every crossing on
    the seed window, on the bracket around the located region and on a
    geometric scan reaching ``10**outer_decades`` times that window on either
    side is located, with narrow dips or bumps between grid points refined by
    a bounded minimiser before root-finding.
    """
    tails_rejected = bool(stat_inf > crit)
    sign = 1.0 if tails_rejected else -1.0

    def f(b: np.ndarray) -> np.ndarray:
        # Negative exactly on the region the tails do NOT belong to: the
        # accepted set when the tails are rejected, the rejected set otherwise.
        return sign * (np.asarray(ar_stat(np.atleast_1d(np.asarray(b, dtype=float))), dtype=float) - crit)

    def f1(b: float) -> float:
        return float(f(np.array([float(b)]))[0])

    # Phase 1: locate one point b_star of the non-tail region, widening the
    # se-scaled window around each seed until the sign of f flips somewhere.
    se_scale = 10.0 * float(se_hat) if (np.isfinite(se_hat) and se_hat > 0) else 0.0
    centre0 = float(beta_hat) if np.isfinite(beta_hat) else 0.0
    seeds: list[tuple[float, float]] = [(centre0, se_scale if se_scale > 0 else max(abs(centre0), 1.0))]
    finite_extra = [float(s) for s in extra_seeds if np.isfinite(s)]
    spread = (max(finite_extra) - min(finite_extra)) if len(finite_extra) > 1 else 0.0
    for s in finite_extra:
        if all(s != c for c, _ in seeds):
            # Window scaled by the data (the spread of the IV ratios or the ratio
            # itself), not by a possibly absurd se_hat.
            seeds.append((s, max(se_scale, spread, abs(s)) or 1.0))
    b_star: float | None = None
    centre, half = seeds[0]
    for centre, half in seeds:
        for _ in range(max_expand):
            grid = np.linspace(centre - half, centre + half, n_grid)
            fv = f(grid)
            finite = np.isfinite(fv)
            if finite.any():
                i = int(np.flatnonzero(finite)[np.argmin(fv[finite])])
                if fv[i] < 0.0:
                    b_star = float(grid[i])
                    break
                lo_i, hi_i = max(i - 1, 0), min(i + 1, n_grid - 1)
                res = minimize_scalar(f1, bounds=(grid[lo_i], grid[hi_i]), method="bounded")
                if res.fun < 0.0:
                    b_star = float(res.x)
                    break
            half *= 2.0
        if b_star is not None:
            break
    if b_star is None:
        if tails_rejected:
            return np.nan, np.nan, "empty"
        return -np.inf, np.inf, "all_real"

    # Phase 2: bracket the located region by stepping outward from b_star
    # until f is positive on each side (guaranteed by the tail limit).
    def _first_positive(direction: float) -> float:
        step = half / 16.0
        b = b_star
        for _ in range(200):
            b = b + direction * step
            if f1(b) > 0.0:
                return float(b)
            step *= 2.0
        return direction * np.inf

    left = _first_positive(-1.0)
    right = _first_positive(1.0)

    # Phase 3: every crossing of the level ``crit`` on the seed window, the
    # bracket and a geometric scan far beyond them (other components of the
    # set need not be adjacent to the one found first).
    lo_w = min(centre - half, left if np.isfinite(left) else b_star - half)
    hi_w = max(centre + half, right if np.isfinite(right) else b_star + half)
    span = hi_w - lo_w
    outer = span * np.logspace(-2.0, float(outer_decades), 25 * (outer_decades + 2))
    grid = np.unique(np.concatenate([np.linspace(lo_w, hi_w, n_grid), lo_w - outer, hi_w + outer, [b_star]]))
    fv = f(grid)
    finite = np.isfinite(fv)
    grid, fv = grid[finite], fv[finite]

    def _n_changes(vals: np.ndarray) -> int:
        neg = vals < 0.0
        return int(np.count_nonzero(neg[1:] != neg[:-1]))

    budget = 2 * max_roots if max_roots is not None else 8
    if max_roots is None or _n_changes(fv) < max_roots:
        # A dip below zero (or a bump above zero) narrower than the grid step
        # shows up as an interior local minimum (maximum) of f on the wrong
        # side of zero; refine the most promising ones with a bounded minimiser.
        interior = np.arange(1, len(grid) - 1)
        dips = interior[(fv[1:-1] > 0.0) & (fv[1:-1] < fv[:-2]) & (fv[1:-1] < fv[2:])]
        bumps = interior[(fv[1:-1] < 0.0) & (fv[1:-1] > fv[:-2]) & (fv[1:-1] > fv[2:])]
        cands = sorted(np.concatenate([dips, bumps]).tolist(), key=lambda j: abs(fv[j]))[:budget]
        new_b: list[float] = []
        new_f: list[float] = []
        for j in cands:
            want_dip = fv[j] > 0.0
            res = minimize_scalar(
                f1 if want_dip else (lambda b: -f1(b)),
                bounds=(grid[j - 1], grid[j + 1]),
                method="bounded",
            )
            val = float(res.fun) if want_dip else -float(res.fun)
            if (val < 0.0) == want_dip and val != 0.0:
                new_b.append(float(res.x))
                new_f.append(val)
                if max_roots is not None and _n_changes(np.concatenate([fv, new_f])) >= max_roots:
                    break
        if new_b:
            order = np.argsort(np.concatenate([grid, new_b]))
            grid = np.concatenate([grid, new_b])[order]
            fv = np.concatenate([fv, new_f])[order]

    def _root(a: float, b: float) -> float:
        # brentq's xtol is absolute: scale it by the root's magnitude, which a
        # same-sign bracket bounds below by its nearer end. A bracket straddling
        # zero gives no such bound, so polish relative to the first result.
        ref = min(abs(a), abs(b)) if a * b > 0.0 else max(abs(a), abs(b))
        tol = 1e-12 * max(ref, 1e-300)
        r = float(brentq(f1, a, b, xtol=tol, maxiter=500))
        if a * b <= 0.0 and r != 0.0 and 1e-12 * abs(r) < tol:
            lo_r, hi_r = r - 2.0 * tol, r + 2.0 * tol
            if (f1(lo_r) < 0.0) != (f1(hi_r) < 0.0):
                r = float(brentq(f1, lo_r, hi_r, xtol=1e-12 * abs(r), maxiter=500))
        return r

    roots: list[float] = []
    for j in range(len(grid) - 1):
        if (fv[j] < 0.0) != (fv[j + 1] < 0.0):
            roots.append(_root(float(grid[j]), float(grid[j + 1])))
    if fv[0] < 0.0:
        roots.insert(0, -np.inf)
    if fv[-1] < 0.0:
        roots.append(np.inf)

    if tails_rejected:
        if len(roots) > 2:
            warnings.warn(
                "Anderson-Rubin confidence set is disconnected; reporting the "
                "enclosing interval [ar_lo, ar_hi].",
                RuntimeWarning,
                stacklevel=3,
            )
        return float(min(roots)), float(max(roots)), "bounded"

    # Tails accepted: the roots delimit rejected gap(s); report (-inf, hi] U [lo, inf)
    # with lo > hi. More than one gap -> keep the widest (a superset of the set).
    gaps = [(roots[j], roots[j + 1]) for j in range(0, len(roots) - 1, 2)]
    if len(gaps) > 1:
        warnings.warn(
            "Anderson-Rubin confidence set has more than two components; "
            "reporting the two rays around the widest rejected gap.",
            RuntimeWarning,
            stacklevel=3,
        )
    gap_lo, gap_hi = max(gaps, key=lambda g: g[1] - g[0])
    return float(gap_hi), float(gap_lo), "unbounded_rays"


def _compute_anderson_rubin_multi(
    y_target: np.ndarray,
    x: np.ndarray,
    W_ctl: np.ndarray,
    Z_mat: np.ndarray,
    lags: int,
    alpha: float,
    beta_hat: float = 0.0,
    se_hat: float = 1.0,
) -> tuple[float, float, str]:
    """Weak-IV-robust Anderson-Rubin confidence set for multi-instrument LP-IV.

    For a candidate ``beta_0`` the statistic is the Newey-West Wald statistic
    for "the instruments have zero coefficient" in the regression of
    ``y_target - beta_0 x`` on ``Z`` (controls partialled out):

        AR(beta_0) = delta(beta_0)' V(beta_0)^{-1} delta(beta_0),

    ``delta(beta_0) = delta_y - beta_0 delta_x`` and ``V(beta_0) = G Omega(beta_0) G``
    with ``Omega(beta_0) = Omega_yy - beta_0 (Omega_yx + Omega_yx') + beta_0^2 Omega_xx``.
    Every block is computed once, so AR is evaluated exactly at any
    ``beta_0`` without a per-point regression. ``lags = 0`` gives the
    Eicker-Huber-White version used by :func:`puremacro.lp.la_lp.la_lp_iv`.

    The set ``{beta_0 : AR(beta_0) <= chi2_{k_z, 1-alpha}}`` is inverted
    exactly (to root-finding tolerance) rather than read off a grid: the
    closed-form limit ``AR(+/-inf) = delta_x' (G Omega_xx G)^{-1} delta_x`` (the
    robust first-stage Wald statistic) classifies the set as bounded/empty
    versus unbounded, and the endpoints are Brent roots of ``AR - crit``. With
    ``k_z >= 2`` the inequality is of degree ``2 k_z`` in ``beta_0`` (not a
    quadratic as in the single-instrument case), so the statistic crosses the
    critical value at most ``2 k_z`` times and the set can have more than one
    component (this does happen in small heteroskedastic samples). Every
    crossing on the window around the 2SLS seed, on the bracket of the region
    found first and on a geometric scan far beyond them is located (see
    :func:`_invert_ar_statistic`); when more than one component is found a
    ``RuntimeWarning`` is issued and the enclosing interval (bounded) or the
    two rays around the widest rejected gap (unbounded) are reported. Only a
    component narrower than the scan resolution and far from every seed can
    escape the scan. ``beta_hat``/``se_hat`` only seed the search, with the
    per-instrument IV ratios ``delta_y / delta_x`` as fallback seeds.
    Returns ``(ar_lo, ar_hi, ar_set_type)``; for ``"unbounded_rays"`` the set
    is ``(-inf, ar_hi] U [ar_lo, inf)`` with ``ar_lo > ar_hi``.
    """
    k_z = Z_mat.shape[1]
    crit = float(chi2.ppf(1.0 - alpha, df=k_z))
    W_inv = np.linalg.pinv(W_ctl.T @ W_ctl)
    y_tilde = y_target - W_ctl @ (W_inv @ (W_ctl.T @ y_target))
    x_tilde = x - W_ctl @ (W_inv @ (W_ctl.T @ x))
    Z_tilde = Z_mat - W_ctl @ (W_inv @ (W_ctl.T @ Z_mat))
    ZtZ_inv = np.linalg.pinv(Z_tilde.T @ Z_tilde)

    delta_y = ZtZ_inv @ (Z_tilde.T @ y_tilde)
    delta_x = ZtZ_inv @ (Z_tilde.T @ x_tilde)
    S_y = Z_tilde * (y_tilde - Z_tilde @ delta_y)[:, None]
    S_x = Z_tilde * (x_tilde - Z_tilde @ delta_x)[:, None]
    Om_yy = _bartlett_lrv(S_y, S_y, lags)
    Om_xx = _bartlett_lrv(S_x, S_x, lags)
    Om_yx = _bartlett_lrv(S_y, S_x, lags)
    Om_cross = Om_yx + Om_yx.T

    def _wald(delta: np.ndarray, Omega: np.ndarray) -> float:
        V_delta = ZtZ_inv @ Omega @ ZtZ_inv
        try:
            return float(delta @ np.linalg.solve(V_delta, delta))
        except np.linalg.LinAlgError:
            return float(delta @ np.linalg.pinv(V_delta) @ delta)

    def ar_stat(b: np.ndarray) -> np.ndarray:
        # AR(b) for a whole array of candidates at once: (m, k_z) deltas and
        # (m, k_z, k_z) covariances, solved in one batched call.
        b = np.asarray(b, dtype=float)
        d = delta_y[None, :] - b[:, None] * delta_x[None, :]
        Om = (
            Om_yy[None, :, :]
            - b[:, None, None] * Om_cross[None, :, :]
            + (b * b)[:, None, None] * Om_xx[None, :, :]
        )
        V = ZtZ_inv[None, :, :] @ Om @ ZtZ_inv[None, :, :]
        try:
            sol = np.linalg.solve(V, d[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            sol = np.stack([np.linalg.pinv(V_i) @ d_i for V_i, d_i in zip(V, d)])
        return np.einsum("ij,ij->i", d, sol)

    stat_inf = _wald(delta_x, Om_xx)
    with np.errstate(divide="ignore", invalid="ignore"):
        iv_ratios = delta_y / delta_x
    return _invert_ar_statistic(
        ar_stat, crit, stat_inf, beta_hat, se_hat, extra_seeds=iv_ratios, max_roots=2 * k_z
    )


def lp_iv(
    df: pd.DataFrame,
    y: str,
    x: str,
    z: str | Sequence[str],
    horizons: Iterable[int] = range(0, 21),
    n_lags: int = 2,
    controls: Sequence[str] | None = None,
    alpha: float = 0.10,
    *,
    anderson_rubin: bool = False,
    weak_iv_robust: bool = False,
    cumulative: bool = False,
    lags: int | None = None,
    horizon: int | None = None,
    ci: float | None = None,
) -> pd.DataFrame:
    """Single-country local projections with instrumental variables (LP-IV).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing outcome, endogenous regressor, instrument(s), and controls.
    y : str
        Outcome variable name.
    x : str
        Endogenous shock / policy variable name.
    z : str or sequence of str
        External instrument name(s).
    horizons : iterable of int, default range(0, 21)
        Impulse response horizons.
    n_lags : int, default 2
        Number of lags of (x, y, controls) included as control variables.
    controls : sequence of str, optional
        Additional contemporaneous or baseline control variables.
    alpha : float, default 0.10
        Significance level for confidence intervals (0.10 -> 90% CI).
    anderson_rubin : bool, default False
        If True, also computes exact weak-instrument-robust Anderson-Rubin (1949)
        confidence intervals (``ar_lo``, ``ar_hi``, ``ar_set_type``).
    weak_iv_robust : bool, default False
        Alias for ``anderson_rubin``.
    cumulative : bool, default False
        If False, the outcome is the long difference ``y_{t+h} - y_{t-1}`` and the
        policy regressor is ``x_t``. If True, estimates the Ramey-Zubairy (2018)
        integral multiplier: the outcome is ``sum_{j=0..h} (y_{t+j} - y_{t-1})`` and
        the policy regressor is ``sum_{j=0..h} x_{t+j}``, so ``beta`` is the
        cumulative response of ``y`` per unit of cumulative ``x`` over horizons
        ``0..h``. With ``y_{t-1}`` among the controls the coefficient equals
        Ramey-Zubairy's ``sum y_{t+j}`` form. Requires a regularly spaced index;
        ``h = 0`` coincides with the default.
    lags : int, optional
        Alias for ``n_lags``. Standardized keyword.
    horizon : int, optional
        If provided, sets ``horizons = range(0, horizon + 1)``.
    ci : float, optional
        Confidence level in (0, 1), e.g. 0.90 for 90% CI. Sets ``alpha = 1.0 - ci``.

    Returns
    -------
    LPResult
        DataFrame subclass with columns:
        ``[h, beta, se, t, lo, hi, first_stage_f, mop_f, mop_cv_10, mop_cv_20]``
        (plus ``[ar_lo, ar_hi, ar_set_type]`` when ``anderson_rubin=True`` or ``weak_iv_robust=True``).
        ``mop_f`` is the Montiel Olea & Pflueger (2013) effective F with the same
        Newey-West bandwidth as the horizon's HAC errors; ``mop_cv_10`` /
        ``mop_cv_20`` are :func:`mop_critical_values` at the 5% level for 10% and
        20% worst-case bias (23.11 / 15.06 for one instrument). ``ar_set_type`` is
        one of ``"bounded"``, ``"unbounded_rays"`` (set ``(-inf, ar_hi] U [ar_lo, inf)``),
        ``"all_real"`` or ``"empty"``.
    """
    horizons, n_lags, alpha = resolve_lp_kwargs(
        horizons, n_lags, alpha, lags=lags, horizon=horizon, ci=ci, name="lp_iv"
    )

    horizons = list(horizons)
    ctl = list(controls or [])
    z_names = [z] if isinstance(z, str) else list(z)
    k_z = len(z_names)
    cv_10, cv_20 = mop_critical_values(k_z)
    z_crit = norm.ppf(1 - alpha / 2)
    do_ar = anderson_rubin or weak_iv_robust
    if cumulative:
        check_regular_index(df.index, "lp_iv")

    rows = []
    for h in horizons:
        sub = df[[y, x] + z_names + ctl].copy()
        # Lags come from the original x and y, before any window replaces them.
        for lag in range(1, n_lags + 1):
            sub[f"{x}_L{lag}"] = sub[x].shift(lag)
            sub[f"{y}_L{lag}"] = sub[y].shift(lag)
            for c in ctl:
                sub[f"{c}_L{lag}"] = sub[c].shift(lag)
        if cumulative:
            sub["dy_h"], sub["__x_w__"] = cumulative_windows(sub[y], sub[x], h)
        else:
            sub["dy_h"] = sub[y].shift(-h) - sub[y].shift(1)
            sub["__x_w__"] = sub[x]
        sub = sub.dropna()
        if sub.empty:
            row: dict[str, Any] = {
                "h": h,
                "beta": np.nan,
                "se": np.nan,
                "t": np.nan,
                "lo": np.nan,
                "hi": np.nan,
                "first_stage_f": np.nan,
                "mop_f": np.nan,
                "mop_cv_10": cv_10,
                "mop_cv_20": cv_20,
            }
            if do_ar:
                row.update({"ar_lo": np.nan, "ar_hi": np.nan, "ar_set_type": "empty"})
            rows.append(row)
            continue

        n = len(sub)
        # Exogenous controls (without instruments)
        W_ctl_list = [np.ones(n)]
        for lag in range(1, n_lags + 1):
            W_ctl_list.append(sub[f"{x}_L{lag}"].values)
            W_ctl_list.append(sub[f"{y}_L{lag}"].values)
            for c in ctl:
                W_ctl_list.append(sub[f"{c}_L{lag}"].values)
        for c in ctl:
            W_ctl_list.append(sub[c].values)
        W_ctl_mat = np.column_stack(W_ctl_list)

        # Instruments
        Z_mat = sub[z_names].values

        # First stage regressor matrix:
        # If single instrument, keep standard order [ones, z, lags...] for exact backward compat
        if k_z == 1:
            W_mat_list = [np.ones(n), sub[z_names[0]].values]
            for lag in range(1, n_lags + 1):
                W_mat_list.append(sub[f"{x}_L{lag}"].values)
                W_mat_list.append(sub[f"{y}_L{lag}"].values)
                for c in ctl:
                    W_mat_list.append(sub[f"{c}_L{lag}"].values)
            for c in ctl:
                W_mat_list.append(sub[c].values)
            W_mat = np.column_stack(W_mat_list)
        else:
            W_mat = np.column_stack([W_ctl_mat, Z_mat])

        # First stage regression
        fs = ols_hac(sub["__x_w__"].values, W_mat, lags=h + 1)
        x_hat = W_mat @ fs["beta"]

        # First-stage F & Montiel Olea & Pflueger effective F
        mop_f = compute_mop_effective_f(sub["__x_w__"].values, Z_mat, W_ctl_mat, lags=h + 1)

        if k_z == 1:
            first_stage_f = (
                float((fs["beta"][1] / fs["se"][1]) ** 2) if fs["se"][1] > 0 else np.nan
            )
        else:
            first_stage_f = mop_f

        # Second stage: x_hat for the estimate, actual x for the residuals behind the SE.
        X2 = [np.ones(n), x_hat]
        for lag in range(1, n_lags + 1):
            X2.append(sub[f"{x}_L{lag}"].values)
            X2.append(sub[f"{y}_L{lag}"].values)
            for c in ctl:
                X2.append(sub[f"{c}_L{lag}"].values)
        for c in ctl:
            X2.append(sub[c].values)
        X2_mat = np.column_stack(X2)
        X_actual = X2_mat.copy()
        X_actual[:, 1] = sub["__x_w__"].values
        out = tsls_hac(sub["dy_h"].values, X_actual, X2_mat, lags=h + 1)
        beta_h = float(out["beta"][1])
        se_h = float(out["se"][1])

        row = {
            "h": h,
            "beta": beta_h,
            "se": se_h,
            "t": beta_h / se_h if se_h > 0 else np.nan,
            "lo": beta_h - z_crit * se_h,
            "hi": beta_h + z_crit * se_h,
            "first_stage_f": first_stage_f,
            "mop_f": mop_f,
            "mop_cv_10": cv_10,
            "mop_cv_20": cv_20,
        }

        if do_ar:
            if k_z == 1:
                ar_lo, ar_hi, ar_kind = _compute_anderson_rubin_ci(
                    sub["dy_h"].values, sub["__x_w__"].values, W_mat, lags=h + 1, alpha=alpha
                )
            else:
                ar_lo, ar_hi, ar_kind = _compute_anderson_rubin_multi(
                    sub["dy_h"].values,
                    sub["__x_w__"].values,
                    W_ctl_mat,
                    Z_mat,
                    lags=h + 1,
                    alpha=alpha,
                    beta_hat=beta_h,
                    se_hat=se_h,
                )
            row.update({"ar_lo": ar_lo, "ar_hi": ar_hi, "ar_set_type": ar_kind})

        rows.append(row)

    from ._results import LPResult

    res = LPResult(rows)
    if "h" in res.columns:
        res.index = res["h"]
    res.y_name = str(y)
    res.x_name = str(x)
    res.method = "LP-IV (cumulative)" if cumulative else "LP-IV"
    res.cumulative = bool(cumulative)
    res.ci_level = 1.0 - alpha
    return res


__all__ = ["lp_iv", "compute_mop_effective_f", "mop_critical_values"]
