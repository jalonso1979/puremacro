"""Stochastic Sovereign Debt Sustainability Analysis (DSA) Engine.

Provides parametric fiscal reaction functions and Monte Carlo simulation
for sovereign debt trajectories under macroeconomic and fiscal uncertainty.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from puremacro.sovereign._fiscal_paths import simulate_paths
from puremacro.sovereign._results import FiscalSovereignResult, StochasticDSAResult

__all__ = [
    "FiscalReactionFunction",
    "simulate_stochastic_dsa",
    "simulate_fiscal_dsa",
    "compare_fiscal_rules_dsa",
]


@dataclass(frozen=True)
class FiscalReactionFunction:
    """Parametric fiscal reaction function for endogenous primary balance.

    Evaluates:
        pb_t = alpha + rho * pb_{t-1} + beta_debt * d_{effective, t-1} + gamma_y * y_gap_t + shock

    where:
    - beta_debt > 0 provides Bohn (1998) solvency discipline.
    - rho captures fiscal policy persistence.
    - gamma_y captures automatic stabilizer sensitivity to the output gap.
    - If fatigue_threshold is specified, debt above the threshold does not
      elicit additional consolidation (fiscal fatigue, Ghosh et al. 2013).
    - If pb_min or pb_max are specified, the primary balance is clamped within bounds.

    Parameters
    ----------
    alpha : float
        Autonomous primary balance constant. Default is 0.0.
    rho : float
        Primary balance autoregressive persistence. Default is 0.70.
    beta_debt : float
        Bohn feedback coefficient on previous debt-to-GDP. Default is 0.03.
    gamma_y : float
        Cyclical sensitivity to the output gap. Default is 0.10.
    pb_min : float | None
        Minimum primary balance floor (e.g. maximum primary deficit). Default is None.
    pb_max : float | None
        Maximum sustained primary surplus ceiling. Default is 5.0 (% of GDP).
    fatigue_threshold : float | None
        Debt ratio threshold beyond which fiscal fatigue caps debt responsiveness.
        Default is None.
    """

    alpha: float = 0.0
    rho: float = 0.70
    beta_debt: float = 0.03
    gamma_y: float = 0.10
    pb_min: float | None = None
    pb_max: float | None = 5.0
    fatigue_threshold: float | None = None

    def evaluate(
        self,
        d_prev: float | np.ndarray,
        y_gap: float | np.ndarray = 0.0,
        pb_prev: float | np.ndarray = 0.0,
        shock: float | np.ndarray = 0.0,
    ) -> Any:
        """Evaluate the primary balance reaction rule for scalar or array inputs."""
        is_array = isinstance(d_prev, np.ndarray) or isinstance(pb_prev, np.ndarray)

        if self.fatigue_threshold is not None:
            if is_array:
                d_eff = np.minimum(d_prev, self.fatigue_threshold)
            else:
                d_eff = min(float(d_prev), float(self.fatigue_threshold))
        else:
            d_eff = d_prev

        pb = (
            self.alpha
            + self.rho * pb_prev
            + self.beta_debt * d_eff
            + self.gamma_y * y_gap
            + shock
        )

        if self.pb_min is not None:
            if is_array:
                pb = np.maximum(pb, self.pb_min)
            else:
                pb = max(float(pb), float(self.pb_min))

        if self.pb_max is not None:
            if is_array:
                pb = np.minimum(pb, self.pb_max)
            else:
                pb = min(float(pb), float(self.pb_max))

        if is_array:
            return pb
        return float(pb)


def simulate_stochastic_dsa(
    *,
    d0: float = 0.65,
    horizon: int = 20,
    n_sims: int = 1000,
    seed: int | None = 42,
    mu: Sequence[float] | np.ndarray = (2.2, 1.8, -1.5),
    A: Sequence[Sequence[float]] | np.ndarray | None = None,
    cov: Sequence[Sequence[float]] | np.ndarray | None = None,
    threshold: float = 0.80,
    pb_shift: float = 0.0,
    m: float = 0.0,
    multiplier_horizon: int = 4,
    shocks: bool = True,
    fiscal_reaction: FiscalReactionFunction | None = None,
    frequency: str = "quarterly",
    exact_compounding: bool = False,
    store_macro_paths: bool = False,
) -> StochasticDSAResult:
    """Simulate stochastic sovereign debt trajectories and compute sustainability metrics.

    Simulates forward paths of sovereign debt-to-GDP ratios using a stationary
    VAR(1) macro process for [growth g, real interest rate r, primary balance pb],
    the debt snowball equation, optional Keynesian multiplier drag, and optional
    endogenous fiscal reaction rules.

    Parameters
    ----------
    d0 : float
        Initial debt-to-GDP ratio (e.g. 0.65 for 65% of GDP).
    horizon : int
        Projection horizon in periods (quarters or years). Default is 20.
    n_sims : int
        Number of stochastic simulation paths. Default is 1000.
    seed : int | None
        Random seed for reproducibility. Default is 42.
    mu : Sequence[float] | np.ndarray
        Long-run unconditional mean vector [growth, interest_rate, primary_balance].
        Default is (2.2, 1.8, -1.5).
    A : Sequence[Sequence[float]] | np.ndarray | None
        3x3 VAR(1) transition matrix. Must satisfy spectral radius rho(A) < 1.0.
    cov : Sequence[Sequence[float]] | np.ndarray | None
        3x3 positive semi-definite innovation covariance matrix.
    threshold : float
        Prudential debt threshold (e.g. 0.80 for 80% of GDP).
    pb_shift : float
        Discretionary primary balance shift (annual percentage points of GDP).
    m : float
        Level fiscal multiplier. When m > 0 and pb_shift != 0, real growth is
        reduced by m * pb_shift during quarters 1..multiplier_horizon.
    multiplier_horizon : int
        Duration (in periods) over which multiplier drag operates. Default is 4.
    shocks : bool
        If True, macroeconomic innovations are added. If False, innovations
        are set to zero while consuming the random stream to preserve alignment.
    fiscal_reaction : FiscalReactionFunction | None
        Optional reaction function determining primary balance endogenously.
    frequency : str
        Simulation frequency: "quarterly" (default) or "annual".
    exact_compounding : bool
        If True, uses exact non-linear compounding:
            d_t = d_{t-1} * (1 + r_t/100) / (1 + g_t/100) - pb_t/100.
        If False and frequency is "quarterly", uses linearized quarterly convention:
            d_t = d_{t-1} * (1 + (r_t - g_t)/400) - pb_t/400.
    store_macro_paths : bool
        If True, stores the simulated macro state variables [g, r, pb] in the result.

    Returns
    -------
    StochasticDSAResult
        Container with simulated debt paths, baseline path, quantiles, breach probabilities,
        Debt-at-Risk (DaR 95), and presentation export methods.
    """
    # 1. Parameter Validation
    if not isinstance(d0, (int, float, np.floating, np.integer)) or not np.isfinite(d0):
        raise ValueError(f"initial debt d0 must be a finite float, got {d0!r}")
    d0_val = float(d0)

    if not isinstance(horizon, (int, np.integer)) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError(f"horizon must be an integer >= 1, got {horizon!r}")
    horizon_val = int(horizon)

    if not isinstance(n_sims, (int, np.integer)) or isinstance(n_sims, bool) or n_sims < 1:
        raise ValueError(f"n_sims must be an integer >= 1, got {n_sims!r}")
    n_sims_val = int(n_sims)

    if not isinstance(threshold, (int, float, np.floating, np.integer)) or not np.isfinite(threshold):
        raise ValueError(f"threshold must be a finite float, got {threshold!r}")
    threshold_val = float(threshold)

    if frequency not in ("quarterly", "annual"):
        raise ValueError(f"frequency must be 'quarterly' or 'annual', got {frequency!r}")

    mu_arr = np.asarray(mu, dtype=float)
    if mu_arr.shape != (3,):
        raise ValueError(f"mu must have length 3 ([growth, interest_rate, primary_balance]), got {mu_arr.shape}")
    if not np.all(np.isfinite(mu_arr)):
        raise ValueError("mu values must be finite numbers")

    # Shift primary balance if discretionary shift is specified
    mu_active = mu_arr.copy()
    if pb_shift != 0.0:
        mu_active[2] += float(pb_shift)

    # Transition matrix A
    if A is None:
        A_mat = np.array(
            [
                [0.60, -0.10, 0.05],
                [0.10, 0.80, -0.05],
                [0.20, -0.10, 0.70],
            ],
            dtype=float,
        )
    else:
        A_mat = np.asarray(A, dtype=float)
        if A_mat.shape != (3, 3):
            raise ValueError(f"A must have shape (3, 3), got {A_mat.shape}")
        if not np.all(np.isfinite(A_mat)):
            raise ValueError("A values must be finite numbers")

    eigvals = np.linalg.eigvals(A_mat)
    rho_A = float(np.max(np.abs(eigvals)))
    if rho_A >= 1.0:
        raise ValueError(
            f"VAR transition matrix A is non-stationary: spectral radius rho(A) = {rho_A:.4f} >= 1.0"
        )

    # Covariance matrix
    if cov is None:
        cov_mat = np.array(
            [
                [2.5, -0.3, 0.4],
                [-0.3, 1.2, -0.2],
                [0.4, -0.2, 1.0],
            ],
            dtype=float,
        )
    else:
        cov_mat = np.asarray(cov, dtype=float)
        if cov_mat.shape != (3, 3):
            raise ValueError(f"cov must have shape (3, 3), got {cov_mat.shape}")
        if not np.all(np.isfinite(cov_mat)):
            raise ValueError("cov values must be finite numbers")

    eigvals_cov = np.linalg.eigvalsh(cov_mat)
    if np.any(eigvals_cov < -1e-10):
        raise ValueError(
            f"Covariance matrix cov is not positive semi-definite (minimum eigenvalue: {np.min(eigvals_cov):.4e})"
        )

    # Factorize covariance
    if np.all(eigvals_cov > 1e-12):
        L = np.linalg.cholesky(cov_mat)
    else:
        eigvals_h, eigvecs_h = np.linalg.eigh(cov_mat)
        L = eigvecs_h @ np.diag(np.sqrt(np.maximum(eigvals_h, 0.0)))

    # 2. Draw Innovations (Common Random Numbers)
    rng = np.random.default_rng(seed)
    z_draws = rng.standard_normal((n_sims_val, horizon_val, 3))
    if shocks:
        innovations = z_draws @ L.T
    else:
        innovations = np.zeros((n_sims_val, horizon_val, 3), dtype=float)

    # 3. Vectorized Simulation across Paths
    paths = np.zeros((n_sims_val, horizon_val + 1), dtype=float)
    paths[:, 0] = d0_val

    if store_macro_paths:
        macro_paths = np.zeros((n_sims_val, horizon_val + 1, 3), dtype=float)
        macro_paths[:, 0, :] = mu_active
    else:
        macro_paths = None

    st = np.tile(mu_active, (n_sims_val, 1))
    pb_prev = np.full(n_sims_val, mu_active[2], dtype=float)

    for t in range(1, horizon_val + 1):
        shk_t = innovations[:, t - 1, :]
        st = mu_active + (st - mu_active) @ A_mat.T + shk_t

        g_t = st[:, 0].copy()
        r_t = st[:, 1].copy()
        pb_t = st[:, 2].copy()

        if m != 0.0 and pb_shift != 0.0 and t <= multiplier_horizon:
            g_t -= m * pb_shift

        if fiscal_reaction is not None:
            y_gap_t = st[:, 0] - mu_arr[0]
            pb_t = fiscal_reaction.evaluate(
                d_prev=paths[:, t - 1],
                y_gap=y_gap_t,
                pb_prev=pb_prev,
                shock=shk_t[:, 2],
            )
            st[:, 2] = pb_t
            pb_prev = pb_t

        if store_macro_paths and macro_paths is not None:
            macro_paths[:, t, 0] = g_t
            macro_paths[:, t, 1] = r_t
            macro_paths[:, t, 2] = pb_t

        d_prev = paths[:, t - 1]
        if frequency == "quarterly" and not exact_compounding:
            paths[:, t] = d_prev * (1.0 + (r_t - g_t) / 400.0) - (pb_t / 400.0)
        else:
            denom = np.maximum(1.0 + g_t / 100.0, 1e-6)
            paths[:, t] = d_prev * (1.0 + r_t / 100.0) / denom - (pb_t / 100.0)

    # 4. Deterministic Baseline Trajectory
    baseline_path = np.zeros(horizon_val + 1, dtype=float)
    baseline_path[0] = d0_val
    st_base = mu_active.copy()
    pb_prev_base = mu_active[2]

    for t in range(1, horizon_val + 1):
        st_base = mu_active + A_mat @ (st_base - mu_active)
        g_t_b = float(st_base[0])
        r_t_b = float(st_base[1])
        pb_t_b = float(st_base[2])

        if m != 0.0 and pb_shift != 0.0 and t <= multiplier_horizon:
            g_t_b -= m * pb_shift

        if fiscal_reaction is not None:
            y_gap_b = float(st_base[0] - mu_arr[0])
            pb_t_b = float(
                fiscal_reaction.evaluate(
                    d_prev=baseline_path[t - 1],
                    y_gap=y_gap_b,
                    pb_prev=pb_prev_base,
                    shock=0.0,
                )
            )
            st_base[2] = pb_t_b
            pb_prev_base = pb_t_b

        d_prev_b = baseline_path[t - 1]
        if frequency == "quarterly" and not exact_compounding:
            baseline_path[t] = d_prev_b * (1.0 + (r_t_b - g_t_b) / 400.0) - (pb_t_b / 400.0)
        else:
            denom_b = max(1.0 + g_t_b / 100.0, 1e-6)
            baseline_path[t] = d_prev_b * (1.0 + r_t_b / 100.0) / denom_b - (pb_t_b / 100.0)

    # 5. Distributional & Fan Chart Metrics
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    quantiles = {p: np.percentile(paths, p, axis=0) for p in percentiles}

    prob_breach = float(np.mean(np.max(paths, axis=1) > threshold_val))
    prob_terminal_breach = float(np.mean(paths[:, -1] > threshold_val))
    prob_non_stabilizing = float(np.mean(paths[:, -1] > d0_val))
    dar_95 = float(np.percentile(paths[:, -1], 95))

    metadata: dict[str, Any] = {
        "d0": d0_val,
        "horizon": horizon_val,
        "n_sims": n_sims_val,
        "seed": seed,
        "mu": mu_arr.tolist(),
        "threshold": threshold_val,
        "pb_shift": pb_shift,
        "m": m,
        "multiplier_horizon": multiplier_horizon,
        "shocks": shocks,
        "frequency": frequency,
        "exact_compounding": exact_compounding,
    }

    return StochasticDSAResult(
        paths=paths,
        baseline_path=baseline_path,
        d0=d0_val,
        threshold=threshold_val,
        horizon=horizon_val,
        frequency=frequency,
        quantiles=quantiles,
        prob_breach=prob_breach,
        prob_terminal_breach=prob_terminal_breach,
        prob_non_stabilizing=prob_non_stabilizing,
        dar_95=dar_95,
        macro_paths=macro_paths,
        metadata=metadata,
    )


_RULE_TOL = 1e-12


def simulate_fiscal_dsa(
    result: FiscalSovereignResult,
    *,
    b0: float = 0.05,
    y0: float | None = None,
    horizon: int = 40,
    n_sims: int = 1000,
    seed: int = 42,
    threshold: float | None = None,
    uniforms: np.ndarray | None = None,
) -> StochasticDSAResult:
    """Simulate structural sovereign debt trajectories from a given initial debt state.

    Uses a solved :class:`~puremacro.sovereign.FiscalSovereignResult` equilibrium
    policy and bond-price schedule to simulate forward debt paths starting at
    initial debt ``b0`` (in units of mean quarterly output, ``E[y] = 1``). This
    separates fiscal rules (such as hard vs. soft debt ceilings) that behave
    identically on the ergodic set ($b \\le 0.02$) but prescribe different
    transition paths when starting from elevated debt ($b_0 > 0.02$).

    Parameters
    ----------
    result : FiscalSovereignResult
        Solved structural fiscal sovereign default model.
    b0 : float, default 0.05
        Initial debt level in units of mean quarterly output (mapped to the
        nearest node on ``result.b_grid``).
    y0 : float, optional
        Initial endowment level in units of mean quarterly output (mapped to the
        nearest node on ``result.y_grid``). If ``None``, selects the median
        income node ``len(result.y_grid) // 2`` when ``b0`` is in the repayment
        region there (``default_prob < 0.5``), or the nearest node to ``1.10``
        (a boom endowment where high initial debt ``b0`` is repaid in good
        standing).
    horizon : int, default 40
        Simulation horizon in quarters.
    n_sims : int, default 1000
        Number of Monte Carlo paths.
    seed : int, default 42
        Random seed for generating common random numbers when ``uniforms`` is None.
    threshold : float, optional
        Debt ceiling / threshold in units of mean quarterly output. Defaults to
        ``result.rule.debt_ceiling`` if present and not None, otherwise ``0.02``.
    uniforms : np.ndarray, optional
        Pre-drawn uniform array of shape ``(3, n_sims, horizon)`` for common
        random numbers across multiple fiscal rules.

    Returns
    -------
    StochasticDSAResult
        Container with simulated debt trajectories (in mean-output units),
        deterministic constant-income baseline path, fan-chart quantiles,
        threshold breach probabilities, and structural transition metadata.
    """
    if not isinstance(result, FiscalSovereignResult):
        raise TypeError(
            f"result must be a FiscalSovereignResult; got {type(result).__name__}"
        )
    if not isinstance(b0, (int, float, np.floating, np.integer)) or not np.isfinite(b0):
        raise ValueError(f"b0 must be a finite float, got {b0!r}")
    b0_val = float(b0)

    if not isinstance(horizon, (int, np.integer)) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError(f"horizon must be an integer >= 1, got {horizon!r}")
    horizon_val = int(horizon)

    if not isinstance(n_sims, (int, np.integer)) or isinstance(n_sims, bool) or n_sims < 1:
        raise ValueError(f"n_sims must be an integer >= 1, got {n_sims!r}")
    n_sims_val = int(n_sims)

    b_idx = int(np.argmin(np.abs(result.b_grid - b0_val)))
    b0_node = float(result.b_grid[b_idx])

    if y0 is None:
        mid_y = len(result.y_grid) // 2
        if float(result.default_prob[b_idx, mid_y]) < 0.5:
            y_idx = mid_y
        else:
            y_idx = int(np.argmin(np.abs(result.y_grid - 1.10)))
    else:
        if not isinstance(y0, (int, float, np.floating, np.integer)) or not np.isfinite(y0):
            raise ValueError(f"y0 must be a finite float or None, got {y0!r}")
        y_idx = int(np.argmin(np.abs(result.y_grid - float(y0))))
    y0_node = float(result.y_grid[y_idx])

    if threshold is None:
        if result.rule is not None and result.rule.debt_ceiling is not None:
            threshold_val = float(result.rule.debt_ceiling)
        else:
            threshold_val = 0.02
    else:
        if not isinstance(threshold, (int, float, np.floating, np.integer)) or not np.isfinite(threshold):
            raise ValueError(f"threshold must be a finite float or None, got {threshold!r}")
        threshold_val = float(threshold)

    if uniforms is None:
        u = np.random.default_rng(seed).random((3, n_sims_val, horizon_val))
    else:
        u = np.asarray(uniforms, dtype=float)
        if u.shape != (3, n_sims_val, horizon_val):
            raise ValueError(
                f"uniforms must have shape (3, {n_sims_val}, {horizon_val}), got {u.shape}"
            )

    fp = simulate_paths(result, b_index=b_idx, y_index=y_idx, uniforms=u)

    # Chosen next-period debt in mean-output units (0.0 in exclusion/default)
    b_next_paths = np.where(fp.repaid, result.b_grid[fp.b_next_index], 0.0)

    paths = np.zeros((n_sims_val, horizon_val + 1), dtype=float)
    paths[:, 0] = b0_node
    paths[:, 1:] = b_next_paths

    # Deterministic constant-income baseline trajectory at y_idx
    zero_idx = int(result.metadata["zero_index"])
    baseline_path = np.zeros(horizon_val + 1, dtype=float)
    baseline_path[0] = b0_node
    b_cur = b_idx
    for step in range(1, horizon_val + 1):
        if float(result.default_prob[b_cur, y_idx]) < 0.5:
            b_next_det = int(result.policy_index[b_cur, y_idx])
            baseline_path[step] = float(result.b_grid[b_next_det])
            b_cur = b_next_det
        else:
            baseline_path[step] = 0.0
            b_cur = zero_idx

    percentiles = [5, 10, 25, 50, 75, 90, 95]
    quantiles = {p: np.percentile(paths, p, axis=0) for p in percentiles}

    # Post-initial choice breach probabilities (respecting _RULE_TOL = 1e-12)
    above_mask = fp.repaid & (b_next_paths > threshold_val + _RULE_TOL)
    prob_breach = float(np.mean(np.any(above_mask, axis=1)))
    prob_terminal_breach = float(np.mean(above_mask[:, -1]))
    prob_non_stabilizing = float(np.mean(paths[:, -1] > b0_node + _RULE_TOL))
    dar_95 = float(np.percentile(paths[:, -1], 95))

    repaid_0 = fp.repaid[:, 0]
    first_period_above_pct = (
        100.0 * float(np.sum(above_mask[:, 0]) / np.sum(repaid_0))
        if np.any(repaid_0)
        else 0.0
    )
    total_repaid = int(np.sum(fp.repaid))
    above_repay_share_pct = (
        100.0 * float(np.sum(above_mask) / total_repaid)
        if total_repaid > 0
        else 0.0
    )

    ceiling_val = (
        float(result.rule.debt_ceiling)
        if (result.rule is not None and result.rule.debt_ceiling is not None)
        else threshold_val
    )
    above_ceiling_mask = fp.repaid & (b_next_paths > ceiling_val + _RULE_TOL)
    above_ceiling_share_pct = (
        100.0 * float(np.sum(above_ceiling_mask) / total_repaid)
        if total_repaid > 0
        else 0.0
    )

    first_period_spending_pct = 100.0 * float(np.mean(fp.g[:, 0] / fp.gdp[:, 0]))
    first_period_b_next = float(np.mean(b_next_paths[:, 0]))
    cumulative_default_prob = float(np.mean(np.any(fp.default_entry, axis=1)))
    terminal_default_prob = float(np.mean(~fp.repaid[:, -1]))

    spread_q50 = np.full(horizon_val, np.nan, dtype=float)
    for t in range(horizon_val):
        valid_sp = fp.spread[fp.repaid[:, t], t]
        if valid_sp.size > 0:
            spread_q50[t] = float(np.median(valid_sp))

    metadata: dict[str, Any] = {
        "model_type": "structural_fiscal_sovereign",
        "b0_requested": b0_val,
        "b0_node": b0_node,
        "b0_index": b_idx,
        "y0_node": y0_node,
        "y0_index": y_idx,
        "horizon": horizon_val,
        "n_sims": n_sims_val,
        "seed": seed,
        "threshold": threshold_val,
        "rule": result.metadata.get("rule"),
        "first_period_b_next": first_period_b_next,
        "first_period_spending_pct": first_period_spending_pct,
        "first_period_above_threshold_pct": first_period_above_pct,
        "above_threshold_repay_share_pct": above_repay_share_pct,
        "above_ceiling_share_pct": above_ceiling_share_pct,
        "cumulative_default_prob": cumulative_default_prob,
        "terminal_default_prob": terminal_default_prob,
        "median_spread_path": spread_q50,
    }

    return StochasticDSAResult(
        paths=paths,
        baseline_path=baseline_path,
        d0=b0_node,
        threshold=threshold_val,
        horizon=horizon_val,
        frequency="quarterly",
        quantiles=quantiles,
        prob_breach=prob_breach,
        prob_terminal_breach=prob_terminal_breach,
        prob_non_stabilizing=prob_non_stabilizing,
        dar_95=dar_95,
        macro_paths=None,
        metadata=metadata,
    )


def compare_fiscal_rules_dsa(
    models: Mapping[str, FiscalSovereignResult],
    *,
    b0: float = 0.05,
    y0: float | None = None,
    horizon: int = 40,
    n_sims: int = 1000,
    seed: int = 42,
    threshold: float = 0.02,
) -> pd.DataFrame:
    """Compare structural debt-sustainability trajectories across fiscal rules using common random numbers.

    Parameters
    ----------
    models : Mapping[str, FiscalSovereignResult]
        Dictionary mapping rule names (e.g. ``"none"``, ``"ceiling"``,
        ``"soft_ceiling"``, ``"ceiling_escape"``, ``"deficit"``) to solved
        :class:`~puremacro.sovereign.FiscalSovereignResult` objects.
    b0 : float, default 0.05
        Initial debt in units of mean quarterly output.
    y0 : float, optional
        Initial endowment in units of mean quarterly output.
    horizon : int, default 40
        Simulation horizon in quarters.
    n_sims : int, default 1000
        Number of Monte Carlo simulation paths.
    seed : int, default 42
        Random seed used once to draw shared uniform shocks across all rules.
    threshold : float, default 0.02
        Debt ceiling / threshold in units of mean quarterly output.

    Returns
    -------
    pd.DataFrame
        Comparison table with one row per rule and columns:
        ``["rule_name", "b0_node", "first_period_b_next", "first_period_g_pct",
        "first_period_above_threshold_pct", "above_threshold_repay_share_pct",
        "median_terminal_debt_pct", "dar_95_pct", "prob_breach_pct",
        "cumulative_default_pct"]``.
    """
    if not models:
        raise ValueError("models mapping must not be empty")
    if not isinstance(horizon, (int, np.integer)) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError(f"horizon must be an integer >= 1, got {horizon!r}")
    if not isinstance(n_sims, (int, np.integer)) or isinstance(n_sims, bool) or n_sims < 1:
        raise ValueError(f"n_sims must be an integer >= 1, got {n_sims!r}")

    uniforms = np.random.default_rng(seed).random((3, int(n_sims), int(horizon)))
    rows: list[dict[str, Any]] = []
    for name, model in models.items():
        dsa = simulate_fiscal_dsa(
            model,
            b0=b0,
            y0=y0,
            horizon=horizon,
            n_sims=n_sims,
            seed=seed,
            threshold=threshold,
            uniforms=uniforms,
        )
        rows.append(
            {
                "rule_name": str(name),
                "b0_node": float(dsa.d0),
                "first_period_b_next": float(dsa.metadata["first_period_b_next"]),
                "first_period_g_pct": float(dsa.metadata["first_period_spending_pct"]),
                "first_period_above_threshold_pct": float(
                    dsa.metadata["first_period_above_threshold_pct"]
                ),
                "above_threshold_repay_share_pct": float(
                    dsa.metadata["above_threshold_repay_share_pct"]
                ),
                "median_terminal_debt_pct": 100.0 * float(dsa.quantiles[50][-1]),
                "dar_95_pct": 100.0 * float(dsa.dar_95),
                "prob_breach_pct": 100.0 * float(dsa.prob_breach),
                "cumulative_default_pct": 100.0 * float(
                    dsa.metadata["cumulative_default_prob"]
                ),
            }
        )
    return pd.DataFrame(rows)
