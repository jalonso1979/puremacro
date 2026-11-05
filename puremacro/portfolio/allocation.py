"""Portfolio optimization, covariance shrinkage, HRP, and Black-Litterman allocation.

Strictly Pyodide-compatible implementation built on ``numpy`` and ``scipy``:
- Covariance shrinkage: analytical Ledoit-Wolf (2004) and Marčenko-Pastur RMT denoising.
- Classical & modern optimization: Max-Sharpe, Minimum-Variance, Target-Return,
  Equal-Risk-Contribution (Maillard-Roncalli-Teïletche 2010 / Spinu 2013),
  Hierarchical Risk Parity (López de Prado 2016), Rockafellar-Uryasev (2000)
  Minimum-CVaR, Fractional Kelly, and Equal-Weight.
- Bayesian views: Black-Litterman (1992) with He-Litterman (1999) and Idzorek (2005)
  confidence levels.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import linprog, minimize
from scipy.spatial.distance import squareform
from scipy.stats import norm

from puremacro.portfolio._results import AllocationResult

__all__ = [
    "shrink_covariance",
    "optimize_portfolio",
    "markowitz_frontier",
    "hierarchical_risk_parity",
    "black_litterman",
]


def _extract_returns_and_names(
    data: np.ndarray | pd.DataFrame | Sequence[float],
    asset_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Coerce input array or DataFrame and extract asset names."""
    if isinstance(data, pd.DataFrame):
        arr = data.to_numpy(dtype=float)
        names = (
            tuple(str(c) for c in asset_names)
            if asset_names is not None
            else tuple(str(c) for c in data.columns)
        )
        return arr, names

    arr = np.asarray(data, dtype=float)
    n_assets = arr.shape[1] if arr.ndim == 2 else arr.shape[0]
    if asset_names is not None:
        names = tuple(str(c) for c in asset_names)
    else:
        names = tuple(f"asset_{i}" for i in range(n_assets))
    return arr, names


def _ensure_psd(cov: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    """Symmetrize covariance matrix and floor eigenvalues for strict positive definiteness."""
    sym = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    if np.min(eigvals) < floor:
        eigvals = np.maximum(eigvals, floor)
        sym = (eigvecs * eigvals) @ eigvecs.T
        sym = 0.5 * (sym + sym.T)
    return sym


def shrink_covariance(
    returns: np.ndarray | pd.DataFrame,
    *,
    method: str = "ledoit_wolf",
) -> np.ndarray:
    """Estimate a well-conditioned, positive-definite asset return covariance matrix.

    Parameters
    ----------
    returns : np.ndarray | pd.DataFrame
        Matrix of periodic asset returns of shape ``(T, N)``.
    method : str, default ``"ledoit_wolf"``
        Covariance estimator:
        - ``"ledoit_wolf"``: Analytical Ledoit-Wolf (2004) shrinkage toward a
          scaled identity target ``(tr(S) / N) * I_N``.
        - ``"rmt"``: Marčenko-Pastur Random Matrix Theory eigenvalue clipping
          preserving the trace and asset variances.
        - ``"sample"``: Standard unbiased sample covariance matrix.

    Returns
    -------
    np.ndarray
        Positive-definite covariance matrix of shape ``(N, N)``.
    """
    X = returns.to_numpy(dtype=float) if isinstance(returns, pd.DataFrame) else np.asarray(returns, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"returns must be a 2D array of shape (T, N); got shape {X.shape}")
    T, N = X.shape
    if T < 2 or N < 1:
        raise ValueError(f"returns must have T >= 2 and N >= 1; got T={T}, N={N}")

    X_c = X - np.mean(X, axis=0, keepdims=True)
    S_sample = (X_c.T @ X_c) / float(T - 1)
    S_sample = 0.5 * (S_sample + S_sample.T)

    m = method.lower().strip()
    if m == "sample":
        return _ensure_psd(S_sample, floor=1e-12)

    if m == "ledoit_wolf":
        S_ml = (X_c.T @ X_c) / float(T)
        mu_tr = float(np.trace(S_ml)) / float(N)
        F_ml = mu_tr * np.eye(N)
        d2 = float(np.sum((S_ml - F_ml) ** 2)) / float(N)
        if d2 <= 1e-18:
            return _ensure_psd(
                (float(np.trace(S_sample)) / float(N)) * np.eye(N),
                floor=1e-10,
            )

        row_sq_norms = np.sum(X_c ** 2, axis=1)
        b_bar2 = (float(np.sum(row_sq_norms ** 2)) - float(T) * float(np.sum(S_ml ** 2))) / (
            float(N) * float(T ** 2)
        )
        b2 = min(max(b_bar2, 0.0), d2)
        delta = float(np.clip(b2 / d2, 0.0, 1.0))
        if N >= T:
            delta = max(delta, 1e-2)

        target_sample = (float(np.trace(S_sample)) / float(N)) * np.eye(N)
        shrunk = (1.0 - delta) * S_sample + delta * target_sample
        return _ensure_psd(shrunk, floor=1e-10)

    if m == "rmt":
        variances = np.maximum(np.diag(S_sample), 1e-12)
        stds = np.sqrt(variances)
        inv_stds = 1.0 / stds
        corr = S_sample * np.outer(inv_stds, inv_stds)
        corr = 0.5 * (corr + corr.T)
        np.fill_diagonal(corr, 1.0)

        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = np.maximum(eigvals, 0.0)

        q = float(N) / float(T)
        lambda_plus = (1.0 + np.sqrt(q)) ** 2
        noise_mask = eigvals <= lambda_plus
        if np.all(noise_mask):
            noise_mask[-1] = False

        if np.any(noise_mask):
            avg_noise = max(float(np.mean(eigvals[noise_mask])), 1e-4)
            eigvals_clipped = eigvals.copy()
            eigvals_clipped[noise_mask] = avg_noise
        else:
            eigvals_clipped = np.maximum(eigvals, 1e-4)

        eigvals_clipped *= float(N) / float(np.sum(eigvals_clipped))
        corr_clean = (eigvecs * eigvals_clipped) @ eigvecs.T
        corr_clean = 0.5 * (corr_clean + corr_clean.T)
        diag_clean = np.sqrt(np.maximum(np.diag(corr_clean), 1e-12))
        corr_clean = corr_clean / np.outer(diag_clean, diag_clean)
        np.fill_diagonal(corr_clean, 1.0)

        cov_rmt = corr_clean * np.outer(stds, stds)
        return _ensure_psd(cov_rmt, floor=1e-10)

    raise ValueError(
        f"Unknown covariance shrinkage method {method!r}. "
        "Expected one of ('ledoit_wolf', 'rmt', 'sample')."
    )


def _solve_risk_budgeting(
    cov: np.ndarray,
    budgets: np.ndarray | None = None,
    max_iter: int = 500,
    tol: float = 1e-13,
) -> np.ndarray:
    """Solve the strictly convex Spinu (2013) / Roncalli (2013) risk-budgeting problem.

    Finds ``w > 0``, ``sum(w) = 1`` such that ``w_i * (Sigma w)_i / (w' Sigma w) = b_i``
    to machine precision via Cyclical Coordinate Descent.
    """
    N = cov.shape[0]
    if budgets is None:
        b = np.full(N, 1.0 / float(N), dtype=float)
    else:
        b = np.asarray(budgets, dtype=float)
        b = np.maximum(b, 1e-8)
        b = b / float(np.sum(b))

    Sigma = _ensure_psd(cov, floor=1e-12)
    diag_s = np.maximum(np.diag(Sigma), 1e-12)
    x = 1.0 / np.sqrt(diag_s)
    x = x / float(np.sum(x))

    for _ in range(max_iter):
        x_old = x.copy()
        for i in range(N):
            # Sum_{j != i} Sigma_{ij} x_j
            c_i = float(np.dot(Sigma[i, :], x) - Sigma[i, i] * x[i])
            s_ii = float(diag_s[i])
            disc = max(c_i * c_i + 4.0 * s_ii * b[i], 0.0)
            x[i] = (-c_i + np.sqrt(disc)) / (2.0 * s_ii)
        if float(np.max(np.abs(x - x_old))) < tol:
            break

    w = x / float(np.sum(x))
    return w


def _build_allocation_result(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    asset_names: tuple[str, ...],
    method: str,
    rf: float = 0.0,
    cvar_alpha: float = 0.05,
    periods_per_year: int = 12,
    scenario_returns: np.ndarray | None = None,
    frontier_returns: np.ndarray | None = None,
    frontier_vols: np.ndarray | None = None,
    frontier_weights: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> AllocationResult:
    """Compute portfolio diagnostics and construct an AllocationResult."""
    w = np.asarray(weights, dtype=float).ravel()
    exp_ret = float(np.dot(w, mu))
    port_var = max(float(w @ cov @ w), 1e-18)
    port_vol = float(np.sqrt(port_var))
    sharpe = (exp_ret - float(rf)) / port_vol if port_vol > 1e-12 else 0.0

    marginal_contrib = cov @ w
    rc = (w * marginal_contrib) / port_var
    rc_sum = float(np.sum(rc))
    if abs(rc_sum) > 1e-12:
        rc = rc / rc_sum

    asset_vols = np.sqrt(np.maximum(np.diag(cov), 1e-18))
    weighted_vol_sum = float(np.dot(w, asset_vols))
    div_ratio = weighted_vol_sum / port_vol if port_vol > 1e-12 else 1.0

    hhi = max(float(np.sum(w ** 2)), 1e-12)
    eff_n = 1.0 / hhi

    alpha_tail = float(np.clip(cvar_alpha, 1e-4, 0.5))
    if scenario_returns is not None and scenario_returns.ndim == 2:
        port_scen = scenario_returns @ w
        losses = -port_scen
        var_q = float(np.percentile(losses, 100.0 * (1.0 - alpha_tail)))
        tail_losses = losses[losses >= var_q]
        cvar_val = float(np.mean(tail_losses)) if tail_losses.size > 0 else var_q
    else:
        p_yr = max(float(periods_per_year), 1.0)
        mu_p = exp_ret / p_yr
        sig_p = port_vol / np.sqrt(p_yr)
        z_a = float(norm.ppf(alpha_tail))
        es_mult = float(norm.pdf(z_a)) / alpha_tail
        cvar_val = -mu_p + es_mult * sig_p

    return AllocationResult(
        weights=w,
        asset_names=asset_names,
        method=method,
        expected_return=exp_ret,
        volatility=port_vol,
        sharpe_ratio=sharpe,
        cvar_95=cvar_val,
        diversification_ratio=div_ratio,
        effective_n_assets=eff_n,
        risk_contributions=rc,
        frontier_returns=frontier_returns,
        frontier_vols=frontier_vols,
        frontier_weights=frontier_weights,
        metadata=dict(metadata) if metadata is not None else {},
    )


def _parse_mu_cov(
    returns_or_mu: np.ndarray | pd.DataFrame | Sequence[float],
    cov: np.ndarray | None,
    shrink: str | None,
    periods_per_year: int,
    asset_names: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, tuple[str, ...]]:
    """Extract annualized (mu, cov), optional raw scenario returns, and asset names."""
    arr, names = _extract_returns_and_names(returns_or_mu, asset_names)
    if arr.ndim == 2:
        scenario_returns = arr
        mu = np.mean(arr, axis=0) * float(periods_per_year)
        if cov is None:
            raw_cov = shrink_covariance(arr, method=shrink or "sample")
            cov_mat = raw_cov * float(periods_per_year)
        else:
            cov_in = np.asarray(cov, dtype=float)
            # If cov matches periodic scale of arr, annualize it
            sample_var = float(np.mean(np.var(arr, axis=0, ddof=1)))
            cov_var = float(np.mean(np.diag(cov_in)))
            if sample_var > 0 and cov_var < 2.5 * sample_var and periods_per_year > 1:
                cov_mat = _ensure_psd(cov_in * float(periods_per_year), floor=1e-12)
            else:
                cov_mat = _ensure_psd(cov_in, floor=1e-12)
        return mu, cov_mat, scenario_returns, names

    if arr.ndim == 1:
        mu = arr.astype(float)
        N = len(mu)
        if cov is None:
            cov_mat = np.eye(N, dtype=float) * 0.04
        else:
            cov_mat = _ensure_psd(np.asarray(cov, dtype=float), floor=1e-12)
        return mu, cov_mat, None, names

    raise ValueError(f"returns_or_mu must be 1D or 2D; got shape {arr.shape}")


def _solve_min_variance(
    cov: np.ndarray,
    bounds: list[tuple[float, float]],
    mu: np.ndarray | None = None,
    target_return: float | None = None,
) -> np.ndarray:
    """Solve minimum-variance portfolio with optional target return constraint."""
    N = cov.shape[0]
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    w0 = np.clip(np.full(N, 1.0 / float(N)), lb, ub)
    if float(np.sum(w0)) > 0:
        w0 = w0 / float(np.sum(w0))

    constraints: list[dict[str, Any]] = [
        {"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0, "jac": lambda w: np.ones(N)}
    ]
    if mu is not None and target_return is not None:
        constraints.append(
            {
                "type": "eq",
                "fun": lambda w, m=mu, tr=target_return: float(np.dot(w, m)) - float(tr),
                "jac": lambda w, m=mu: m,
            }
        )

    res = minimize(
        fun=lambda w: 0.5 * float(w @ cov @ w),
        x0=w0,
        jac=lambda w: cov @ w,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 500},
    )
    w = np.clip(res.x, lb, ub)
    w = w / float(np.sum(w))
    return w


def optimize_portfolio(
    returns_or_mu: np.ndarray | pd.DataFrame | Sequence[float],
    cov: np.ndarray | None = None,
    *,
    method: str = "max_sharpe",
    rf: float = 0.0,
    target_return: float | None = None,
    cvar_alpha: float = 0.05,
    kelly_fraction: float = 0.5,
    long_only: bool = True,
    max_weight: float = 1.0,
    min_weight: float = 0.0,
    shrink: str | None = None,
    periods_per_year: int = 12,
    asset_names: Sequence[str] | None = None,
) -> AllocationResult:
    """Optimize portfolio weights across classical and modern allocation paradigms.

    Parameters
    ----------
    returns_or_mu : np.ndarray | pd.DataFrame | Sequence[float]
        Either a 2D ``(T, N)`` matrix/DataFrame of periodic asset returns or a
        1D ``(N,)`` vector of annualized expected returns.
    cov : np.ndarray | None, optional
        Asset covariance matrix ``(N, N)``. Estimated from ``returns_or_mu`` when ``None``.
    method : str, default ``"max_sharpe"``
        Optimization rule in ``("max_sharpe", "min_variance", "target_return",
        "risk_parity", "hrp", "min_cvar", "kelly", "equal_weight")``.
    rf : float, default 0.0
        Annualized risk-free rate.
    target_return : float | None, optional
        Target annualized return when ``method="target_return"``.
    cvar_alpha : float, default 0.05
        Tail probability level for CVaR minimization (0.05 = 95% CVaR).
    kelly_fraction : float, default 0.5
        Fractional Kelly multiplier (0.5 = Half-Kelly).
    long_only : bool, default True
        Restrict weights to be non-negative.
    max_weight : float, default 1.0
        Upper bound on individual asset weight.
    min_weight : float, default 0.0
        Lower bound on individual asset weight.
    shrink : str | None, optional
        Covariance shrinkage estimator (``"ledoit_wolf"``, ``"rmt"``, or ``"sample"``).
    periods_per_year : int, default 12
        Number of return observations per year.
    asset_names : Sequence[str] | None, optional
        Explicit asset names.

    Returns
    -------
    AllocationResult
    """
    m = method.lower().strip()
    valid_methods = (
        "max_sharpe",
        "min_variance",
        "target_return",
        "risk_parity",
        "hrp",
        "min_cvar",
        "kelly",
        "equal_weight",
    )
    if m not in valid_methods:
        raise ValueError(f"Unknown optimization method {method!r}. Expected one of {valid_methods}.")

    if m == "hrp":
        arr, names = _extract_returns_and_names(returns_or_mu, asset_names)
        if arr.ndim == 2:
            return hierarchical_risk_parity(
                returns_or_mu,
                is_cov=False,
                periods_per_year=periods_per_year,
                asset_names=names,
            )
        if cov is None:
            raise ValueError("cov must be provided when returns_or_mu is 1D and method='hrp'")
        hrp_res = hierarchical_risk_parity(
            cov,
            is_cov=True,
            periods_per_year=periods_per_year,
            asset_names=names,
        )
        return _build_allocation_result(
            weights=hrp_res.weights,
            mu=arr,
            cov=_ensure_psd(np.asarray(cov, dtype=float), floor=1e-12),
            asset_names=names,
            method="hrp",
            rf=rf,
            cvar_alpha=cvar_alpha,
            periods_per_year=periods_per_year,
            metadata=hrp_res.metadata,
        )

    mu, cov_mat, scenario_returns, names = _parse_mu_cov(
        returns_or_mu, cov, shrink, periods_per_year, asset_names
    )
    N = len(mu)

    lb_val = max(0.0, float(min_weight)) if long_only else (
        float(min_weight) if min_weight != 0.0 else -float(max_weight)
    )
    ub_val = float(max_weight)
    if lb_val > ub_val or lb_val * N > 1.0 + 1e-9 or ub_val * N < 1.0 - 1e-9:
        raise ValueError(
            f"Infeasible weight bounds [{lb_val}, {ub_val}] for N={N} assets summing to 1.0."
        )
    bounds = [(lb_val, ub_val) for _ in range(N)]
    lb_arr = np.full(N, lb_val, dtype=float)
    ub_arr = np.full(N, ub_val, dtype=float)

    if m == "equal_weight":
        w = np.clip(np.full(N, 1.0 / float(N), dtype=float), lb_arr, ub_arr)
        w = w / float(np.sum(w))

    elif m == "min_variance":
        w = _solve_min_variance(cov_mat, bounds)

    elif m == "target_return":
        # Determine feasible return range [r_min, r_max] under bounds
        res_min_r = minimize(
            fun=lambda x: float(np.dot(x, mu)),
            x0=np.full(N, 1.0 / float(N)),
            jac=lambda x: mu,
            bounds=bounds,
            constraints=[{"type": "eq", "fun": lambda x: float(np.sum(x)) - 1.0}],
            method="SLSQP",
        )
        res_max_r = minimize(
            fun=lambda x: -float(np.dot(x, mu)),
            x0=np.full(N, 1.0 / float(N)),
            jac=lambda x: -mu,
            bounds=bounds,
            constraints=[{"type": "eq", "fun": lambda x: float(np.sum(x)) - 1.0}],
            method="SLSQP",
        )
        r_min_feas = float(np.dot(res_min_r.x, mu))
        r_max_feas = float(np.dot(res_max_r.x, mu))
        tgt = float(np.mean(mu)) if target_return is None else float(target_return)
        tgt_clipped = float(np.clip(tgt, r_min_feas + 1e-9, r_max_feas - 1e-9))
        w = _solve_min_variance(cov_mat, bounds, mu=mu, target_return=tgt_clipped)

    elif m == "max_sharpe":
        excess = mu - float(rf)
        w0 = np.clip(np.full(N, 1.0 / float(N)), lb_arr, ub_arr)
        w0 = w0 / float(np.sum(w0))

        def _neg_sharpe(x: np.ndarray) -> float:
            var = max(float(x @ cov_mat @ x), 1e-18)
            vol = np.sqrt(var)
            return -float(np.dot(x, excess)) / vol

        def _neg_sharpe_jac(x: np.ndarray) -> np.ndarray:
            var = max(float(x @ cov_mat @ x), 1e-18)
            vol = np.sqrt(var)
            ret_ex = float(np.dot(x, excess))
            return -(excess * vol - ret_ex * (cov_mat @ x) / vol) / var

        res = minimize(
            fun=_neg_sharpe,
            x0=w0,
            jac=_neg_sharpe_jac,
            bounds=bounds,
            constraints=[
                {"type": "eq", "fun": lambda x: float(np.sum(x)) - 1.0, "jac": lambda x: np.ones(N)}
            ],
            method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 500},
        )
        w = np.clip(res.x, lb_arr, ub_arr)
        w = w / float(np.sum(w))

    elif m == "risk_parity":
        w_erc = _solve_risk_budgeting(cov_mat)
        if np.all(w_erc >= lb_arr - 1e-9) and np.all(w_erc <= ub_arr + 1e-9):
            w = w_erc
        else:
            target_rc = 1.0 / float(N)

            def _rc_obj(x: np.ndarray) -> float:
                var = max(float(x @ cov_mat @ x), 1e-18)
                rc = x * (cov_mat @ x) / var
                return float(np.sum((rc - target_rc) ** 2))

            w_init = np.clip(w_erc, lb_arr, ub_arr)
            w_init = w_init / float(np.sum(w_init))
            res = minimize(
                fun=_rc_obj,
                x0=w_init,
                bounds=bounds,
                constraints=[{"type": "eq", "fun": lambda x: float(np.sum(x)) - 1.0}],
                method="SLSQP",
                options={"ftol": 1e-14, "maxiter": 600},
            )
            w = np.clip(res.x, lb_arr, ub_arr)
            w = w / float(np.sum(w))

    elif m == "min_cvar":
        alpha_tail = float(np.clip(cvar_alpha, 1e-4, 0.5))
        if scenario_returns is not None and scenario_returns.ndim == 2:
            T_scen = scenario_returns.shape[0]
            # Rockafellar-Uryasev LP: variables [w_1..w_N, zeta, u_1..u_T]
            c_lp = np.zeros(N + 1 + T_scen, dtype=float)
            c_lp[N] = 1.0
            c_lp[N + 1 :] = 1.0 / (alpha_tail * float(T_scen))

            # -R w - zeta - u <= 0
            A_ub = np.zeros((T_scen, N + 1 + T_scen), dtype=float)
            A_ub[:, :N] = -scenario_returns
            A_ub[:, N] = -1.0
            A_ub[:, N + 1 :] = -np.eye(T_scen)
            b_ub = np.zeros(T_scen, dtype=float)

            A_eq = np.zeros((1, N + 1 + T_scen), dtype=float)
            A_eq[0, :N] = 1.0
            b_eq = np.array([1.0], dtype=float)

            lp_bounds = bounds + [(None, None)] + [(0.0, None)] * T_scen
            lp_res = linprog(
                c_lp,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=lp_bounds,
                method="highs",
            )
            if lp_res.success and lp_res.x is not None:
                w = np.clip(lp_res.x[:N], lb_arr, ub_arr)
                w = w / float(np.sum(w))
            else:
                w = _solve_min_variance(cov_mat, bounds)
        else:
            p_yr = max(float(periods_per_year), 1.0)
            z_a = float(norm.ppf(alpha_tail))
            es_mult = float(norm.pdf(z_a)) / alpha_tail
            w0 = np.clip(np.full(N, 1.0 / float(N)), lb_arr, ub_arr)
            w0 = w0 / float(np.sum(w0))
            res = minimize(
                fun=lambda x: -float(np.dot(x, mu)) / p_yr
                + es_mult * np.sqrt(max(float(x @ cov_mat @ x), 1e-18) / p_yr),
                x0=w0,
                bounds=bounds,
                constraints=[{"type": "eq", "fun": lambda x: float(np.sum(x)) - 1.0}],
                method="SLSQP",
                options={"ftol": 1e-12, "maxiter": 500},
            )
            w = np.clip(res.x, lb_arr, ub_arr)
            w = w / float(np.sum(w))

    else:  # m == "kelly"
        kappa = max(float(kelly_fraction), 1e-4)
        excess = mu - float(rf)
        w0 = np.clip(np.full(N, 1.0 / float(N)), lb_arr, ub_arr)
        w0 = w0 / float(np.sum(w0))
        res = minimize(
            fun=lambda x: -float(np.dot(x, excess)) + (0.5 / kappa) * float(x @ cov_mat @ x),
            x0=w0,
            jac=lambda x: -excess + (1.0 / kappa) * (cov_mat @ x),
            bounds=bounds,
            constraints=[
                {"type": "eq", "fun": lambda x: float(np.sum(x)) - 1.0, "jac": lambda x: np.ones(N)}
            ],
            method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 500},
        )
        w = np.clip(res.x, lb_arr, ub_arr)
        w = w / float(np.sum(w))

    return _build_allocation_result(
        weights=w,
        mu=mu,
        cov=cov_mat,
        asset_names=names,
        method=m,
        rf=rf,
        cvar_alpha=cvar_alpha,
        periods_per_year=periods_per_year,
        scenario_returns=scenario_returns,
        metadata={
            "rf": rf,
            "long_only": long_only,
            "max_weight": max_weight,
            "min_weight": min_weight,
            "shrink": shrink,
            "periods_per_year": periods_per_year,
        },
    )


def hierarchical_risk_parity(
    returns_or_cov: np.ndarray | pd.DataFrame,
    *,
    is_cov: bool = False,
    linkage_method: str = "single",
    periods_per_year: int = 12,
    asset_names: Sequence[str] | None = None,
) -> AllocationResult:
    """Compute López de Prado (2016) 3-stage Hierarchical Risk Parity (HRP) weights.

    Parameters
    ----------
    returns_or_cov : np.ndarray | pd.DataFrame
        Either a ``(T, N)`` matrix of periodic returns (when ``is_cov=False``)
        or an ``(N, N)`` covariance matrix (when ``is_cov=True``).
    is_cov : bool, default False
        Set ``True`` if ``returns_or_cov`` is already a covariance matrix.
    linkage_method : str, default ``"single"``
        Hierarchical clustering linkage method (``"single"``, ``"ward"``, ``"complete"``, ``"average"``).
    periods_per_year : int, default 12
        Number of return periods per year.
    asset_names : Sequence[str] | None, optional
        Asset names.

    Returns
    -------
    AllocationResult
    """
    arr, names = _extract_returns_and_names(returns_or_cov, asset_names)
    if arr.ndim != 2:
        raise ValueError(f"returns_or_cov must be 2D; got shape {arr.shape}")

    if is_cov:
        cov_mat = _ensure_psd(arr, floor=1e-12)
        mu = np.zeros(cov_mat.shape[0], dtype=float)
        scenario_returns = None
    else:
        scenario_returns = arr
        mu = np.mean(arr, axis=0) * float(periods_per_year)
        cov_mat = shrink_covariance(arr, method="ledoit_wolf") * float(periods_per_year)

    N = cov_mat.shape[0]
    if N == 1:
        return _build_allocation_result(
            weights=np.array([1.0]),
            mu=mu,
            cov=cov_mat,
            asset_names=names,
            method="hrp",
            periods_per_year=periods_per_year,
            scenario_returns=scenario_returns,
        )

    stds = np.sqrt(np.maximum(np.diag(cov_mat), 1e-12))
    corr = cov_mat / np.outer(stds, stds)
    corr = np.clip(0.5 * (corr + corr.T), -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    # Stage 1: Tree clustering on correlation distance d_ij = sqrt(0.5 * (1 - rho_ij))
    dist_mat = np.sqrt(np.maximum(0.5 * (1.0 - corr), 0.0))
    dist_mat = 0.5 * (dist_mat + dist_mat.T)
    np.fill_diagonal(dist_mat, 0.0)
    condensed_dist = squareform(dist_mat, checks=False)
    link = linkage(condensed_dist, method=linkage_method)

    # Stage 2: Quasi-diagonalization (seriation of leaf nodes)
    sort_ix = [int(i) for i in leaves_list(link)]

    # Stage 3: Recursive bisection with inverse-variance cluster allocation
    weights = np.ones(N, dtype=float)
    clusters: list[list[int]] = [sort_ix]

    def _cluster_var(cov_sub: np.ndarray) -> float:
        ivp = 1.0 / np.maximum(np.diag(cov_sub), 1e-12)
        ivp = ivp / float(np.sum(ivp))
        return float(ivp @ cov_sub @ ivp)

    while clusters:
        next_clusters: list[list[int]] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            left = cluster[:mid]
            right = cluster[mid:]
            var_left = _cluster_var(cov_mat[np.ix_(left, left)])
            var_right = _cluster_var(cov_mat[np.ix_(right, right)])
            denom = var_left + var_right
            alpha_split = 1.0 - var_left / denom if denom > 0 else 0.5
            weights[left] *= alpha_split
            weights[right] *= 1.0 - alpha_split
            if len(left) > 1:
                next_clusters.append(left)
            if len(right) > 1:
                next_clusters.append(right)
        clusters = next_clusters

    weights = weights / float(np.sum(weights))
    return _build_allocation_result(
        weights=weights,
        mu=mu,
        cov=cov_mat,
        asset_names=names,
        method="hrp",
        periods_per_year=periods_per_year,
        scenario_returns=scenario_returns,
        metadata={
            "sort_ix": sort_ix,
            "linkage": link,
            "linkage_method": linkage_method,
            "periods_per_year": periods_per_year,
        },
    )


def markowitz_frontier(
    returns_or_mu: np.ndarray | pd.DataFrame | Sequence[float],
    cov: np.ndarray | None = None,
    *,
    n_points: int = 30,
    rf: float = 0.0,
    long_only: bool = True,
    max_weight: float = 1.0,
    periods_per_year: int = 12,
    asset_names: Sequence[str] | None = None,
) -> AllocationResult:
    """Trace the Markowitz mean-variance efficient frontier and return the tangency portfolio.

    Parameters
    ----------
    returns_or_mu : np.ndarray | pd.DataFrame | Sequence[float]
        2D periodic return matrix or 1D annualized expected return vector.
    cov : np.ndarray | None, optional
        Asset covariance matrix ``(N, N)``.
    n_points : int, default 30
        Number of frontier grid points.
    rf : float, default 0.0
        Annualized risk-free rate.
    long_only : bool, default True
        Restrict weights to be non-negative.
    max_weight : float, default 1.0
        Maximum weight on a single asset.
    periods_per_year : int, default 12
        Number of periods per year.
    asset_names : Sequence[str] | None, optional
        Asset names.

    Returns
    -------
    AllocationResult
        Tangency (Max-Sharpe) portfolio with ``frontier_returns``, ``frontier_vols``,
        and ``frontier_weights`` populated.
    """
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2; got {n_points}")

    mu, cov_mat, scenario_returns, names = _parse_mu_cov(
        returns_or_mu, cov, None, periods_per_year, asset_names
    )
    N = len(mu)
    lb_val = 0.0 if long_only else -float(max_weight)
    ub_val = float(max_weight)
    bounds = [(lb_val, ub_val) for _ in range(N)]

    w_gmv = _solve_min_variance(cov_mat, bounds)
    r_gmv = float(np.dot(w_gmv, mu))

    res_max = minimize(
        fun=lambda x: -float(np.dot(x, mu)),
        x0=np.full(N, 1.0 / float(N)),
        jac=lambda x: -mu,
        bounds=bounds,
        constraints=[{"type": "eq", "fun": lambda x: float(np.sum(x)) - 1.0}],
        method="SLSQP",
    )
    r_max = max(float(np.dot(res_max.x, mu)), r_gmv)

    frontier_returns = np.linspace(r_gmv, r_max, n_points)
    frontier_vols = np.zeros(n_points, dtype=float)
    frontier_weights = np.zeros((n_points, N), dtype=float)

    for idx, r_tgt in enumerate(frontier_returns):
        if idx == 0 or abs(r_max - r_gmv) < 1e-12:
            w_pt = w_gmv
        else:
            w_pt = _solve_min_variance(cov_mat, bounds, mu=mu, target_return=float(r_tgt))
        frontier_weights[idx, :] = w_pt
        frontier_vols[idx] = float(np.sqrt(max(float(w_pt @ cov_mat @ w_pt), 1e-18)))

    tangency = optimize_portfolio(
        mu,
        cov_mat,
        method="max_sharpe",
        rf=rf,
        long_only=long_only,
        max_weight=max_weight,
        periods_per_year=periods_per_year,
        asset_names=names,
    )

    return _build_allocation_result(
        weights=tangency.weights,
        mu=mu,
        cov=cov_mat,
        asset_names=names,
        method="markowitz_frontier",
        rf=rf,
        periods_per_year=periods_per_year,
        scenario_returns=scenario_returns,
        frontier_returns=frontier_returns,
        frontier_vols=frontier_vols,
        frontier_weights=frontier_weights,
        metadata={
            "n_points": n_points,
            "rf": rf,
            "gmv_return": r_gmv,
            "gmv_vol": float(frontier_vols[0]),
        },
    )


def black_litterman(
    cov: np.ndarray | pd.DataFrame,
    market_weights: Sequence[float] | np.ndarray,
    *,
    P: np.ndarray,
    Q: Sequence[float] | np.ndarray,
    omega: np.ndarray | None = None,
    confidences: Sequence[float] | np.ndarray | None = None,
    tau: float = 0.05,
    delta: float = 2.5,
    rf: float = 0.0,
    long_only: bool = True,
    max_weight: float = 1.0,
    asset_names: Sequence[str] | None = None,
) -> AllocationResult:
    """Compute Black-Litterman (1992) posterior returns and optimal portfolio weights.

    Supports both He-Litterman (1999) proportional view uncertainty
    ``Omega = diag(P (tau Sigma) P')`` and Idzorek (2005) user confidence levels
    ``c_k in [0, 1]``.

    Parameters
    ----------
    cov : np.ndarray | pd.DataFrame
        Asset return covariance matrix of shape ``(N, N)``.
    market_weights : Sequence[float] | np.ndarray
        Benchmark or market-capitalization portfolio weights of length ``N``.
    P : np.ndarray
        View pick matrix of shape ``(K, N)`` (or 1D of length ``N`` for a single view).
    Q : Sequence[float] | np.ndarray
        View expected excess returns of length ``K``.
    omega : np.ndarray | None, optional
        Explicit view uncertainty covariance matrix ``(K, K)`` (or diagonal vector of length ``K``).
    confidences : Sequence[float] | np.ndarray | None, optional
        Idzorek (2005) view confidence levels in ``[0, 1]`` of length ``K``.
    tau : float, default 0.05
        Scalar scaling factor for prior equilibrium return uncertainty.
    delta : float, default 2.5
        Market risk-aversion coefficient.
    rf : float, default 0.0
        Risk-free rate added to posterior excess returns.
    long_only : bool, default True
        Restrict optimal weights to be non-negative.
    max_weight : float, default 1.0
        Upper bound on individual asset weights.
    asset_names : Sequence[str] | None, optional
        Asset names.

    Returns
    -------
    AllocationResult
    """
    cov_arr, names = _extract_returns_and_names(cov, asset_names)
    Sigma = _ensure_psd(cov_arr, floor=1e-12)
    N = Sigma.shape[0]

    w_mkt = np.asarray(market_weights, dtype=float).ravel()
    if len(w_mkt) != N:
        raise ValueError(f"market_weights length {len(w_mkt)} does not match cov dimension {N}")
    w_sum = float(np.sum(w_mkt))
    if abs(w_sum) > 1e-12:
        w_mkt = w_mkt / w_sum

    # Implied equilibrium excess returns Pi = delta * Sigma @ w_mkt
    pi_excess = float(delta) * (Sigma @ w_mkt)
    tau_sigma = float(tau) * Sigma

    q_vec = np.asarray(Q, dtype=float).ravel()
    K = len(q_vec)
    if K == 0 or np.asarray(P).size == 0:
        mu_bl_excess = pi_excess.copy()
        sigma_bl = Sigma + tau_sigma
    else:
        p_mat = np.asarray(P, dtype=float)
        if p_mat.ndim == 1:
            p_mat = p_mat.reshape(1, -1)
        if p_mat.shape != (K, N):
            raise ValueError(f"P must have shape ({K}, {N}); got {p_mat.shape}")

        v_mat = p_mat @ tau_sigma @ p_mat.T
        v_diag = np.maximum(np.diag(v_mat), 1e-12)

        if confidences is not None:
            c_vec = np.clip(np.asarray(confidences, dtype=float).ravel(), 0.0, 1.0)
            if len(c_vec) != K:
                raise ValueError(f"confidences length {len(c_vec)} must match number of views {K}")
            # Exact Idzorek / Kalman-gain formulation stable on [0, 1] including c_k = 0
            sqrt_c = np.sqrt(c_vec)
            scaled_v = (v_mat * np.outer(sqrt_c, sqrt_c)) + np.diag((1.0 - c_vec) * v_diag)
            scaled_v = _ensure_psd(scaled_v, floor=1e-14)
            inv_term = np.diag(sqrt_c) @ np.linalg.solve(scaled_v, np.diag(sqrt_c))
            gain = tau_sigma @ p_mat.T @ inv_term
            mu_bl_excess = pi_excess + gain @ (q_vec - p_mat @ pi_excess)
            sigma_bl_mean = tau_sigma - gain @ p_mat @ tau_sigma
            sigma_bl = _ensure_psd(Sigma + sigma_bl_mean, floor=1e-12)
        else:
            if omega is None:
                omega_mat = np.diag(v_diag)
            else:
                om_arr = np.asarray(omega, dtype=float)
                omega_mat = np.diag(om_arr) if om_arr.ndim == 1 else om_arr
            total_v = _ensure_psd(v_mat + omega_mat, floor=1e-14)
            gain = tau_sigma @ p_mat.T @ np.linalg.inv(total_v)
            mu_bl_excess = pi_excess + gain @ (q_vec - p_mat @ pi_excess)
            sigma_bl_mean = tau_sigma - gain @ p_mat @ tau_sigma
            sigma_bl = _ensure_psd(Sigma + sigma_bl_mean, floor=1e-12)

    # Optimize Black-Litterman mean-variance utility: max_w w' mu_bl_excess - (delta / 2) w' Sigma w
    # Unconstrained solution when mu_bl_excess == pi_excess is identically w_mkt.
    lb_val = 0.0 if long_only else -float(max_weight)
    ub_val = float(max_weight)
    bounds = [(lb_val, ub_val) for _ in range(N)]
    lb_arr = np.full(N, lb_val, dtype=float)
    ub_arr = np.full(N, ub_val, dtype=float)

    w0 = np.clip(w_mkt, lb_arr, ub_arr)
    w0 = w0 / float(np.sum(w0))

    res = minimize(
        fun=lambda x: -float(np.dot(x, mu_bl_excess)) + 0.5 * float(delta) * float(x @ Sigma @ x),
        x0=w0,
        jac=lambda x: -mu_bl_excess + float(delta) * (Sigma @ x),
        bounds=bounds,
        constraints=[
            {"type": "eq", "fun": lambda x: float(np.sum(x)) - 1.0, "jac": lambda x: np.ones(N)}
        ],
        method="SLSQP",
        options={"ftol": 1e-14, "maxiter": 500},
    )
    w_opt = np.clip(res.x, lb_arr, ub_arr)
    w_opt = w_opt / float(np.sum(w_opt))

    mu_bl_total = mu_bl_excess + float(rf)
    return _build_allocation_result(
        weights=w_opt,
        mu=mu_bl_total,
        cov=Sigma,
        asset_names=names,
        method="black_litterman",
        rf=rf,
        metadata={
            "equilibrium_returns": pi_excess + float(rf),
            "equilibrium_excess_returns": pi_excess,
            "posterior_returns": mu_bl_total,
            "posterior_excess_returns": mu_bl_excess,
            "posterior_cov": sigma_bl,
            "tau": tau,
            "delta": delta,
            "rf": rf,
        },
    )
