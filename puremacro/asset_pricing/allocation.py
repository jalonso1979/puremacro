"""Constrained portfolio allocation and explicitly chosen covariance estimates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

from ._research import ResearchReport, dated_frame, finite, readonly
from ._validate import _validate_returns

__all__ = ["CovarianceResult", "PortfolioAllocationResult", "estimate_covariance",
           "equal_weight", "optimize_portfolio", "efficient_frontier"]


@dataclass(frozen=True)
class CovarianceResult(ResearchReport):
    names: tuple[str, ...]
    covariance: np.ndarray
    means: np.ndarray
    shrinkage: float
    n_obs: int
    method: str

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.covariance, index=self.names, columns=self.names).rename_axis("asset").reset_index()


@dataclass(frozen=True)
class PortfolioAllocationResult(ResearchReport):
    names: tuple[str, ...]
    weights: np.ndarray
    variance: float
    expected_return: float
    turnover: float
    objective: float
    optimality_gap: float
    constraint_violation: float
    converged: bool

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({"asset": self.names, "weight": self.weights})

    def as_series(self) -> pd.Series:
        return pd.Series(self.weights.copy(), index=self.names, name="weight")

    def plot(self, ax: Any = None, show: bool = False):
        from ._results import _new_ax, _finish
        fig, ax, given = _new_ax(ax)
        ax.bar(self.names, self.weights)
        ax.set(ylabel="portfolio weight", title="Constrained allocation")
        return _finish(fig, ax, given, show)


def estimate_covariance(returns: pd.DataFrame, *, method: str = "ledoit_wolf",
                        check_scale: bool = True) -> CovarianceResult:
    """Per-period sample (T-1 divisor) or Ledoit-Wolf identity-target covariance.

    Ledoit-Wolf uses demeaned observations and a T divisor, with target
    trace(S)/N * I. The shrinkage intensity estimates the squared Frobenius
    estimation error, capped by distance to the target (Ledoit & Wolf 2004,
    Journal of Multivariate Analysis 88, 365-411). No observations are dropped.
    Means are descriptive sample means, not certified forecasts.
    """
    frame = dated_frame(returns, "returns")
    _validate_returns(frame, "returns", check_scale)
    if len(frame) < 2:
        raise ValueError("at least two observations are required")
    if method not in ("sample", "ledoit_wolf"):
        raise ValueError("method must be sample or ledoit_wolf")
    x = frame.to_numpy()
    means = x.mean(axis=0)
    z = x - means
    t, n = z.shape
    sample = z.T @ z / t
    intensity = 0.0
    if method == "sample":
        cov = sample * t / (t - 1)
    else:
        target = np.eye(n) * np.trace(sample) / n
        distance = float(np.sum((sample - target) ** 2))
        # sum_i ||z_i z_i' - S||_F^2 / T^2, calculated without a T*N*N cube.
        error = max(0.0, float(np.mean(np.sum(z*z, axis=1)**2) - np.sum(sample*sample))) / t
        if distance > 0 and n > 1:
            intensity = min(1.0, error / distance)
        cov = (1 - intensity) * sample + intensity * target
    return CovarianceResult(tuple(frame.columns), readonly(cov), readonly(means),
                            intensity, t, method)


def equal_weight(assets: Sequence[str]) -> pd.Series:
    """Fully invested equal weights; explicitly supply the eligible asset list."""
    names = tuple(assets)
    if not names or len(set(names)) != len(names) or not all(isinstance(s, str) and s for s in names):
        raise ValueError("assets must contain unique nonempty string names")
    return pd.Series(np.full(len(names), 1 / len(names)), index=names, name="weight")


def _vector(value: Any, names: tuple[str, ...], label: str) -> np.ndarray:
    if isinstance(value, pd.Series):
        if value.index.has_duplicates or set(value.index) != set(names):
            raise ValueError(f"{label} labels must match covariance assets")
        value = value.reindex(names).to_numpy()
    out = np.asarray(value, dtype=float)
    if out.ndim == 0:
        out = np.full(len(names), float(out))
    if out.shape != (len(names),) or not np.isfinite(out).all():
        raise ValueError(f"{label} must have one finite value per asset")
    return out


def optimize_portfolio(
    covariance: CovarianceResult | pd.DataFrame | np.ndarray, *,
    expected_returns: Any = None, risk_aversion: float = 1.0, return_reward: float = 1.0,
    max_weight: Any = 1.0, previous_weights: Any = None,
    turnover_penalty: float = 0.0, target_return: float | None = None,
    tolerance: float = 1e-7,
) -> PortfolioAllocationResult:
    """Solve a long-only, fully invested convex allocation problem.

    Minimize risk_aversion/2 * w' Sigma w - return_reward*mu'w
    + penalty*||w-w_prev||_1
    with 0 <= w <= max_weight, sum(w)=1 and optional mu'w >= target_return.
    Without expected_returns the linear-return term is zero (minimum variance),
    including when a CovarianceResult carries sample means. All moments and
    penalties must be in consistent per-period units. A penalty is an allocation
    preference; realized execution costs belong in backtest_portfolio.

    L1 turnover is represented exactly by linear auxiliary constraints.
    SLSQP finds a candidate; an independent linear minimization of its gradient
    over the feasible polytope bounds its convex objective suboptimality.
    Failures/infeasibility raise rather than returning an equal-weight fallback.
    """
    if isinstance(covariance, CovarianceResult):
        names, sigma = covariance.names, np.array(covariance.covariance, copy=True)
    elif isinstance(covariance, pd.DataFrame):
        if (covariance.index.has_duplicates or covariance.columns.has_duplicates
                or set(covariance.index) != set(covariance.columns)
                or not all(isinstance(c, str) and c for c in covariance.columns)):
            raise ValueError("covariance row/column asset labels must match and be unique strings")
        names = tuple(covariance.columns)
        sigma = covariance.loc[list(names), list(names)].to_numpy(dtype=float)
    else:
        sigma = np.asarray(covariance, dtype=float)
        names = tuple(f"a{i}" for i in range(len(sigma))) if sigma.ndim == 2 else ()
    n = len(names)
    if n == 0 or sigma.shape != (n, n) or not np.isfinite(sigma).all():
        raise ValueError("covariance must be a finite nonempty square matrix")
    magnitude = max(float(np.max(np.abs(sigma))), np.finfo(float).tiny)
    if not np.allclose(sigma, sigma.T, rtol=0, atol=1e-12*magnitude):
        raise ValueError("covariance must be symmetric")
    try:
        eigenvalues = np.linalg.eigvalsh(sigma)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance eigendecomposition failed") from exc
    if eigenvalues.min() < -32*np.finfo(float).eps*n*magnitude:
        raise ValueError("covariance must be positive semidefinite")
    gamma = finite(risk_aversion, "risk_aversion", 0)
    if gamma == 0:
        raise ValueError("risk_aversion must be positive")
    penalty = finite(turnover_penalty, "turnover_penalty", 0)
    reward = finite(return_reward, "return_reward", 0)
    tol = finite(tolerance, "tolerance", 0)
    if tol == 0 or tol > .01:
        raise ValueError("tolerance must be in (0, .01]")
    mu = np.zeros(n) if expected_returns is None else _vector(expected_returns, names, "expected_returns")
    caps = _vector(max_weight, names, "max_weight")
    if np.any(caps < 0) or np.any(caps > 1) or caps.sum() < 1 - 1e-12:
        raise ValueError("max_weight is infeasible or outside [0,1]")
    if previous_weights is None and penalty > 0:
        raise ValueError("turnover penalty requires previous_weights")
    previous = np.zeros(n) if previous_weights is None else _vector(previous_weights, names, "previous_weights")
    if np.any(previous < 0) or previous.sum() > 1 + 1e-12:
        raise ValueError("previous_weights must be long-only and unlevered")
    auxiliary = penalty > 0
    dim = 2*n if auxiliary else n
    eq = np.zeros((1, dim)); eq[0, :n] = 1
    aub, bub = [], []
    if target_return is not None:
        if expected_returns is None:
            raise ValueError("target_return requires expected_returns")
        target = finite(target_return, "target_return")
        row = np.zeros(dim); row[:n] = -mu
        aub.append(row); bub.append(-target)
    if auxiliary:
        for i in range(n):
            row = np.zeros(dim); row[i] = 1; row[n+i] = -1
            aub.append(row); bub.append(previous[i])
            row = np.zeros(dim); row[i] = -1; row[n+i] = -1
            aub.append(row); bub.append(-previous[i])
    A = np.asarray(aub).reshape((-1, dim)) if aub else None
    b = np.asarray(bub) if bub else None
    bounds = [(0., float(c)) for c in caps] + ([(0., None)]*n if auxiliary else [])
    feasible = linprog(np.zeros(dim), A_ub=A, b_ub=b, A_eq=eq, b_eq=[1.], bounds=bounds, method="highs")
    if not feasible.success:
        raise ValueError(f"portfolio constraints are infeasible: {feasible.message}")
    scale = max(gamma*magnitude, reward*float(np.max(np.abs(mu))), penalty, np.finfo(float).tiny)
    def objective(x):
        return float((.5*gamma*x[:n] @ sigma @ x[:n] - reward*mu @ x[:n]
                      + (penalty*x[n:].sum() if auxiliary else 0)) / scale)
    def jac(x):
        gradient = gamma*sigma @ x[:n] - reward*mu
        return np.r_[gradient, np.full(n, penalty)] / scale if auxiliary else gradient / scale
    constraints = [{"type": "eq", "fun": lambda x: eq @ x - 1, "jac": lambda x: eq}]
    if A is not None:
        constraints.append({"type": "ineq", "fun": lambda x: b - A @ x, "jac": lambda x: -A})
    solved = minimize(objective, feasible.x, jac=jac, bounds=bounds, constraints=constraints,
                      method="SLSQP", options={"ftol": min(1e-12, tol*tol), "maxiter": 2000})
    x = solved.x
    x[:n] = np.clip(x[:n], 0, caps)
    violation = max(abs(float((eq @ x)[0] - 1)), max(0., float(np.max(-x[:n]))),
                    max(0., float(np.max(x[:n]-caps))),
                    max(0., float(np.max(A @ x - b))) if A is not None else 0.)
    gradient = jac(x)
    oracle = linprog(gradient, A_ub=A, b_ub=b, A_eq=eq, b_eq=[1.], bounds=bounds, method="highs")
    gap = max(0., float(gradient @ x - oracle.fun)) if oracle.success else np.inf
    if not solved.success or violation > tol or gap > tol:
        raise RuntimeError(f"allocation not certified: {solved.message}; violation={violation:g}, normalized gap={gap:g}")
    weights = x[:n]
    variance = float(weights @ sigma @ weights)
    if not np.isfinite(variance) or variance < 0:
        raise RuntimeError("negative portfolio variance or overflow; covariance is not usable at working precision")
    return PortfolioAllocationResult(names, readonly(weights), variance,
        float(mu @ weights) if expected_returns is not None else np.nan,
        float(np.abs(weights-previous).sum()) if previous_weights is not None else np.nan,
        objective(x)*scale, gap*scale, violation, True)


def efficient_frontier(covariance: CovarianceResult | pd.DataFrame | np.ndarray,
                       expected_returns: Any, targets: Sequence[float], **kwargs: Any) -> pd.DataFrame:
    """Minimum-variance portfolios at explicit return floors; fail on infeasibility.

    Floors below the global minimum-variance return repeat that allocation.
    Each solve uses zero linear reward for returns: mu only defines the floor.
    Turnover penalties, if supplied, change the frontier objective explicitly.
    """
    if any(k in kwargs for k in ("target_return", "risk_aversion", "return_reward")):
        raise ValueError("frontier sets target_return, risk_aversion and return_reward internally")
    rows = []
    for target in targets:
        result = optimize_portfolio(covariance, expected_returns=expected_returns,
                                    target_return=float(target), return_reward=0.0, **kwargs)
        rows.append({"target_return": target, "expected_return": result.expected_return,
                     "volatility": np.sqrt(result.variance), "optimality_gap": result.optimality_gap,
                     **{f"weight:{n}": w for n, w in zip(result.names, result.weights)}})
    return pd.DataFrame(rows)
