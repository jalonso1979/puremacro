"""Stochastic Sovereign Debt Sustainability Analysis (DSA) Engine.

Provides parametric fiscal reaction functions and Monte Carlo simulation
for sovereign debt trajectories under macroeconomic and fiscal uncertainty.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from scipy.stats import norm

from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst
from puremacro.sovereign._fiscal_paths import simulate_paths
from puremacro.sovereign._results import FiscalSovereignResult, StochasticDSAResult

__all__ = [
    "FiscalReactionEstimateResult",
    "FiscalReactionFunction",
    "compare_fiscal_rules_dsa",
    "estimate_fiscal_reaction_function",
    "fit_stochastic_dsa_from_data",
    "simulate_fiscal_dsa",
    "simulate_stochastic_dsa",
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


def _step_debt_ratio(
    d_prev: np.ndarray | float,
    g_t: np.ndarray | float,
    r_t: np.ndarray | float,
    pb_t: np.ndarray | float,
    *,
    frequency: str,
    exact_compounding: bool,
    rate_convention: str,
) -> np.ndarray | float:
    """Advance debt-to-GDP ratio by one period under annualized percent macro inputs."""
    periods = 4.0 if frequency == "quarterly" else 1.0
    scale = 100.0 * periods
    if not exact_compounding:
        return d_prev * (1.0 + (r_t - g_t) / scale) - (pb_t / scale)
    if frequency == "quarterly" and rate_convention == "effective":
        num = np.power(np.maximum(1.0 + r_t / 100.0, 1e-6), 0.25)
        denom = np.power(np.maximum(1.0 + g_t / 100.0, 1e-6), 0.25)
        return d_prev * (num / denom) - (pb_t / scale)
    denom = np.maximum(1.0 + g_t / scale, 1e-6)
    return d_prev * (1.0 + r_t / scale) / denom - (pb_t / scale)


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
    rate_convention: str = "apr",
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
        If True, uses exact non-linear compounding scaled to ``frequency``:
        for ``frequency="annual"``, ``(1 + r_t/100) / (1 + g_t/100) - pb_t/100``;
        for ``frequency="quarterly"``, ``(1 + r_t/400) / (1 + g_t/400) - pb_t/400``
        when ``rate_convention="apr"`` (or ``((1 + r_t/100)/(1 + g_t/100))**0.25 - pb_t/400``
        when ``rate_convention="effective"``).
        If False, uses linearized period accumulation ``d_{t-1} * (1 + (r_t - g_t)/s) - pb_t/s``
        with ``s = 400`` (quarterly) or ``s = 100`` (annual).
    rate_convention : str
        Sub-annual compounding convention when ``frequency="quarterly"`` and
        ``exact_compounding=True``: ``"apr"`` (default, ``r/400`` and ``g/400``)
        or ``"effective"`` (geometric ``(1 + r/100)**0.25`` and ``(1 + g/100)**0.25``).
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

    if not isinstance(frequency, str) or frequency not in ("quarterly", "annual"):
        raise ValueError(f"frequency must be 'quarterly' or 'annual', got {frequency!r}")

    if not isinstance(rate_convention, str) or rate_convention not in ("apr", "effective"):
        raise ValueError(f"rate_convention must be 'apr' or 'effective', got {rate_convention!r}")

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
        paths[:, t] = _step_debt_ratio(
            d_prev,
            g_t,
            r_t,
            pb_t,
            frequency=frequency,
            exact_compounding=exact_compounding,
            rate_convention=rate_convention,
        )

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
        baseline_path[t] = float(
            _step_debt_ratio(
                d_prev_b,
                g_t_b,
                r_t_b,
                pb_t_b,
                frequency=frequency,
                exact_compounding=exact_compounding,
                rate_convention=rate_convention,
            )
        )

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
        "rate_convention": rate_convention,
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


def _newey_west_cov_ols(X: np.ndarray, resid: np.ndarray, hac_lags: int) -> np.ndarray:
    """Compute Newey-West HAC covariance matrix for OLS estimator."""
    T, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    scores = X * resid[:, None]
    S = scores.T @ scores
    max_lag = min(int(hac_lags), max(T - 1, 0))
    for lag in range(1, max_lag + 1):
        w = 1.0 - lag / (max_lag + 1.0)
        gamma_l = scores[lag:].T @ scores[:-lag]
        S = S + w * (gamma_l + gamma_l.T)
    dof_scale = T / max(T - k, 1)
    return dof_scale * (XtX_inv @ S @ XtX_inv)


@dataclass(frozen=True)
class FiscalReactionEstimateResult:
    """Empirical Bohn (1998) / Ghosh et al. (2013) fiscal reaction function estimate.

    Parameters
    ----------
    reaction_function : FiscalReactionFunction
        Calibrated parametric :class:`FiscalReactionFunction` ready for DSA simulation.
    params : Mapping[str, float]
        Estimated regression coefficients (``alpha``, ``rho``, ``beta_debt``, and optional
        ``beta_debt_sq``, ``beta_debt_cu``, ``gamma_y``).
    se : Mapping[str, float]
        Newey-West HAC standard errors for each coefficient.
    t_stats : Mapping[str, float]
        HAC t-statistics for each coefficient.
    p_values : Mapping[str, float]
        Two-sided asymptotic p-values for each coefficient.
    bohn_t_stat : float
        t-statistic on the linear debt feedback coefficient ``beta_debt``.
    bohn_p_value_one_sided : float
        One-sided p-value testing ``H0: beta_debt <= 0`` vs ``H1: beta_debt > 0``
        (Bohn 1998 solvency condition).
    bohn_solvency_holds : bool
        True if ``beta_debt > 0`` and ``bohn_p_value_one_sided < 0.05``.
    fatigue_threshold : float | None
        Empirical fiscal-fatigue turning point ``d* > 0`` where ``d(pb)/d(debt) = 0``
        from the cubic specification (if a positive root exists within plausible range).
    r2 : float
        In-sample R-squared of the fiscal reaction regression.
    residual_std : float
        Standard deviation of the primary balance innovation ``epsilon_t``.
    n_obs : int
        Effective sample size used in estimation.
    metadata : Mapping[str, Any]
        Additional diagnostic information.
    """

    reaction_function: FiscalReactionFunction
    params: Mapping[str, float]
    se: Mapping[str, float]
    t_stats: Mapping[str, float]
    p_values: Mapping[str, float]
    bohn_t_stat: float
    bohn_p_value_one_sided: float
    bohn_solvency_holds: bool
    fatigue_threshold: float | None
    r2: float
    residual_std: float
    n_obs: int
    metadata: Mapping[str, Any]

    def to_dataframe(self) -> pd.DataFrame:
        """Return parameter estimates and HAC inference as a DataFrame."""
        rows = []
        for name, val in self.params.items():
            rows.append(
                {
                    "term": name,
                    "coef": float(val),
                    "se_hac": float(self.se[name]),
                    "t_stat": float(self.t_stats[name]),
                    "p_value": float(self.p_values[name]),
                }
            )
        return pd.DataFrame(rows)

    def to_frame(self) -> pd.DataFrame:
        """Alias for :meth:`to_dataframe`."""
        return self.to_dataframe()

    def summary(self) -> str:
        """Return a formatted ASCII summary of the empirical fiscal reaction function."""
        df_str = self.to_dataframe().to_string(index=False, float_format=lambda x: f"{x:.4f}")
        fat_str = f"{self.fatigue_threshold:.4f}" if self.fatigue_threshold is not None else "None"
        lines = [
            "==========================================================================",
            "Empirical Fiscal Reaction Function (Bohn 1998 / Ghosh et al. 2013)",
            "==========================================================================",
            f"Observations (N): {self.n_obs} | R-squared: {self.r2:.4f} | Residual SD: {self.residual_std:.4f}",
            f"Bohn Solvency Test (beta_debt > 0): t = {self.bohn_t_stat:.3f} "
            f"(one-sided p = {self.bohn_p_value_one_sided:.4f}) -> Holds: {self.bohn_solvency_holds}",
            f"Estimated Fiscal Fatigue Threshold (d*): {fat_str}",
            "--------------------------------------------------------------------------",
            df_str,
            "==========================================================================",
        ]
        return "\n".join(lines)

    def to_markdown(self, **kwargs: Any) -> str:
        """Export regression table to GitHub-flavored Markdown."""
        return df_to_markdown(self.to_dataframe(), index=False, **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Export regression table to LaTeX booktabs format."""
        return df_to_latex(self.to_dataframe(), index=False, **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Export regression table to Typst format."""
        return df_to_typst(self.to_dataframe(), index=False, **kwargs)

    def plot(self, *, debt_range: tuple[float, float] = (0.20, 1.60), ax: Any = None):
        """Plot the estimated marginal primary balance schedule as a function of debt."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
        else:
            fig = ax.figure
        d_grid = np.linspace(debt_range[0], debt_range[1], 200)
        b1 = self.params.get("beta_debt", 0.0)
        b2 = self.params.get("beta_debt_sq", 0.0)
        b3 = self.params.get("beta_debt_cu", 0.0)
        alpha = self.params.get("alpha", 0.0)
        pb_poly = alpha + b1 * d_grid + b2 * (d_grid ** 2) + b3 * (d_grid ** 3)
        pb_rule = np.array([self.reaction_function.evaluate(float(d), alpha) for d in d_grid])
        ax.plot(100.0 * d_grid, pb_poly, lw=2.2, color="#1f77b4", label="Estimated Polynomial Schedule")
        ax.plot(100.0 * d_grid, pb_rule, lw=1.8, ls="--", color="#d62728", label="Piecewise Reaction Function")
        if self.fatigue_threshold is not None:
            ax.axvline(
                100.0 * self.fatigue_threshold,
                color="#2ca02c",
                ls=":",
                lw=1.6,
                label=f"Fatigue Threshold ({100.0 * self.fatigue_threshold:.1f}%)",
            )
        ax.set_xlabel("Lagged Debt-to-GDP Ratio (%)")
        ax.set_ylabel("Primary Balance Response (% of GDP)")
        ax.set_title("Empirical Fiscal Reaction Function & Fiscal Fatigue")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
        return fig


def estimate_fiscal_reaction_function(
    df: pd.DataFrame,
    *,
    pb_col: str = "pb",
    debt_col: str = "debt",
    y_gap_col: str | None = None,
    cubic_fatigue: bool = True,
    hac_lags: int = 4,
    pb_max: float | None = 5.0,
    pb_min: float | None = None,
) -> FiscalReactionEstimateResult:
    """Estimate Bohn (1998) linear or Ghosh et al. (2013) cubic fiscal reaction function.

    Estimates:
        pb_t = alpha + rho * pb_{t-1} + beta_1 * d_{t-1}
               [+ beta_2 * d_{t-1}^2 + beta_3 * d_{t-1}^3] [+ gamma_y * y_gap_t] + eps_t

    Parameters
    ----------
    df : pd.DataFrame
        Time-series DataFrame containing primary balance and debt-to-GDP series.
    pb_col : str, default "pb"
        Column name for primary balance (% of GDP or ratio).
    debt_col : str, default "debt"
        Column name for debt-to-GDP ratio (e.g. ``0.60`` for 60% of GDP).
    y_gap_col : str | None, default None
        Optional column name for output gap.
    cubic_fatigue : bool, default True
        If True, includes quadratic and cubic lagged debt terms (Ghosh et al. 2013)
        to estimate an empirical fiscal-fatigue threshold ``d*`` where ``d(pb)/d(debt) = 0``.
    hac_lags : int, default 4
        Newey-West HAC truncation lag.
    pb_max : float | None, default 5.0
        Primary balance ceiling passed to the returned :class:`FiscalReactionFunction`.
    pb_min : float | None, default None
        Primary balance floor passed to the returned :class:`FiscalReactionFunction`.

    Returns
    -------
    FiscalReactionEstimateResult
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    for col in (pb_col, debt_col):
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found in df")
    if y_gap_col is not None and y_gap_col not in df.columns:
        raise ValueError(f"Column {y_gap_col!r} not found in df")

    work = pd.DataFrame(
        {
            "pb": pd.to_numeric(df[pb_col], errors="coerce"),
            "pb_lag": pd.to_numeric(df[pb_col], errors="coerce").shift(1),
            "d_lag": pd.to_numeric(df[debt_col], errors="coerce").shift(1),
        }
    )
    if y_gap_col is not None:
        work["y_gap"] = pd.to_numeric(df[y_gap_col], errors="coerce")
    work = work.dropna()
    if len(work) < 8:
        raise ValueError("estimate_fiscal_reaction_function requires at least 8 valid time-series observations")

    y = work["pb"].to_numpy(dtype=float)
    d_lag = work["d_lag"].to_numpy(dtype=float)
    pb_lag = work["pb_lag"].to_numpy(dtype=float)

    # Fit linear Bohn specification first to get the baseline linear slope and Bohn solvency test
    cols_lin = [np.ones_like(y), pb_lag, d_lag]
    names_lin = ["alpha", "rho", "beta_debt"]
    if y_gap_col is not None:
        cols_lin.append(work["y_gap"].to_numpy(dtype=float))
        names_lin.append("gamma_y")
    X_lin = np.column_stack(cols_lin)
    beta_lin, *_ = np.linalg.lstsq(X_lin, y, rcond=None)
    resid_lin = y - X_lin @ beta_lin
    cov_lin = _newey_west_cov_ols(X_lin, resid_lin, hac_lags=hac_lags)
    se_lin = np.sqrt(np.maximum(np.diag(cov_lin), 1e-16))
    bohn_idx = names_lin.index("beta_debt")
    bohn_t = float(beta_lin[bohn_idx] / se_lin[bohn_idx])
    bohn_p_one = float(norm.sf(bohn_t))

    fatigue_threshold: float | None = None
    if cubic_fatigue and len(work) >= 12:
        cols_cub = [np.ones_like(y), pb_lag, d_lag, d_lag ** 2, d_lag ** 3]
        names = ["alpha", "rho", "beta_debt", "beta_debt_sq", "beta_debt_cu"]
        if y_gap_col is not None:
            cols_cub.append(work["y_gap"].to_numpy(dtype=float))
            names.append("gamma_y")
        X = np.column_stack(cols_cub)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        cov = _newey_west_cov_ols(X, resid, hac_lags=hac_lags)
        se = np.sqrt(np.maximum(np.diag(cov), 1e-16))
        # Solve d(pb)/dd = beta_1 + 2*beta_2*d + 3*beta_3*d^2 = 0 for local maximum (fatigue turning point)
        b1, b2, b3 = float(beta[2]), float(beta[3]), float(beta[4])
        d_min_obs = float(np.min(d_lag))
        d_max_obs = float(np.max(d_lag))
        if abs(b3) > 1e-10:
            roots = np.roots([3.0 * b3, 2.0 * b2, b1])
            real_roots = [
                float(r.real)
                for r in roots
                if abs(r.imag) < 1e-7 and (2.0 * b2 + 6.0 * b3 * float(r.real) < 0.0)
            ]
            valid_roots = [r for r in real_roots if max(0.20, 0.5 * d_min_obs) <= r <= 2.5 * max(d_max_obs, 1.0)]
            if valid_roots:
                fatigue_threshold = float(min(valid_roots))
        elif b2 < -1e-10 and b1 > 0:
            cand = -b1 / (2.0 * b2)
            if max(0.20, 0.5 * d_min_obs) <= cand <= 2.5 * max(d_max_obs, 1.0):
                fatigue_threshold = float(cand)
    else:
        X = X_lin
        names = names_lin
        beta = beta_lin
        resid = resid_lin
        cov = cov_lin
        se = se_lin

    t_stats_arr = beta / se
    p_vals_arr = 2.0 * norm.sf(np.abs(t_stats_arr))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    rss = float(np.sum(resid ** 2))
    r2 = float(1.0 - rss / max(tss, 1e-16))
    resid_std = float(np.std(resid, ddof=len(names))) if len(y) > len(names) else float(np.std(resid))

    params_dict = {k: float(v) for k, v in zip(names, beta)}
    se_dict = {k: float(v) for k, v in zip(names, se)}
    t_dict = {k: float(v) for k, v in zip(names, t_stats_arr)}
    p_dict = {k: float(v) for k, v in zip(names, p_vals_arr)}

    # Build the FiscalReactionFunction using the linear Bohn feedback + estimated fatigue threshold
    rho_clamped = float(np.clip(beta_lin[1], -0.99, 0.99))
    gamma_y_est = float(beta_lin[names_lin.index("gamma_y")]) if "gamma_y" in names_lin else 0.0
    reaction_fn = FiscalReactionFunction(
        alpha=float(beta_lin[0]),
        rho=rho_clamped,
        beta_debt=float(beta_lin[2]),
        gamma_y=gamma_y_est,
        pb_min=pb_min,
        pb_max=pb_max,
        fatigue_threshold=fatigue_threshold,
    )
    return FiscalReactionEstimateResult(
        reaction_function=reaction_fn,
        params=params_dict,
        se=se_dict,
        t_stats=t_dict,
        p_values=p_dict,
        bohn_t_stat=bohn_t,
        bohn_p_value_one_sided=bohn_p_one,
        bohn_solvency_holds=bool(beta_lin[2] > 0.0 and bohn_p_one < 0.05),
        fatigue_threshold=fatigue_threshold,
        r2=r2,
        residual_std=resid_std,
        n_obs=len(y),
        metadata={
            "cubic_fatigue": bool(cubic_fatigue),
            "hac_lags": int(hac_lags),
            "linear_beta_debt": float(beta_lin[2]),
            "linear_se_beta_debt": float(se_lin[2]),
        },
    )


def fit_stochastic_dsa_from_data(
    df: pd.DataFrame,
    *,
    g_col: str = "g",
    r_col: str = "r",
    pb_col: str = "pb",
    debt_col: str | None = "debt",
    y_gap_col: str | None = None,
    d0: float | None = None,
    horizon: int = 20,
    n_sims: int = 1000,
    seed: int = 42,
    threshold: float = 0.80,
    use_bohn_reaction: bool = True,
    cubic_fatigue: bool = True,
    frequency: str = "quarterly",
) -> StochasticDSAResult:
    """Estimate a joint macro-fiscal VAR(1) and optional Bohn reaction function from data and run DSA.

    Parameters
    ----------
    df : pd.DataFrame
        Historical macro-fiscal DataFrame containing ``(g, r, pb)`` and optionally ``debt``.
    g_col, r_col, pb_col : str
        Column names for real GDP growth (%), real interest rate (%), and primary balance (% of GDP).
    debt_col : str | None, default "debt"
        Column name for historical debt-to-GDP ratio (used for initial debt ``d0`` if ``d0 is None``
        and for estimating the Bohn fiscal reaction function when ``use_bohn_reaction=True``).
    y_gap_col : str | None, default None
        Optional output-gap column for the fiscal reaction function.
    d0 : float | None, default None
        Initial debt-to-GDP ratio. If ``None``, uses the last non-missing value of ``df[debt_col]``.
    horizon : int, default 20
        Projection horizon.
    n_sims : int, default 1000
        Number of Monte Carlo simulation paths.
    seed : int, default 42
        Random seed.
    threshold : float, default 0.80
        Debt distress threshold ratio.
    use_bohn_reaction : bool, default True
        If True and ``debt_col`` is present in ``df``, estimates :func:`estimate_fiscal_reaction_function`
        and passes the resulting :class:`FiscalReactionFunction` into :func:`simulate_stochastic_dsa`.
    cubic_fatigue : bool, default True
        Whether to estimate cubic fiscal fatigue when ``use_bohn_reaction=True``.
    frequency : str, default "quarterly"
        Data frequency (``"quarterly"`` or ``"annual"``).

    Returns
    -------
    StochasticDSAResult
    """
    for col in (g_col, r_col, pb_col):
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found in df")
    sub = df[[g_col, r_col, pb_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 8:
        raise ValueError("fit_stochastic_dsa_from_data requires at least 8 complete (g, r, pb) observations")

    Y_all = sub.to_numpy(dtype=float)
    mu_hat = np.mean(Y_all, axis=0)
    Y_dem = Y_all - mu_hat[None, :]
    Y_curr = Y_dem[1:]
    Y_lag = Y_dem[:-1]
    # OLS VAR(1) companion matrix A_hat: Y_curr_t = A_hat @ Y_lag_t + u_t
    A_hat_T, *_ = np.linalg.lstsq(Y_lag, Y_curr, rcond=None)
    A_hat = A_hat_T.T
    # Shrink eigenvalues inside unit circle if nearly non-stationary
    eigvals = np.linalg.eigvals(A_hat)
    max_eig = float(np.max(np.abs(eigvals)))
    if max_eig >= 0.98:
        A_hat = A_hat * (0.95 / max_eig)
    U_hat = Y_curr - Y_lag @ A_hat.T
    Sigma_hat = np.cov(U_hat, rowvar=False, ddof=1)
    Sigma_hat = 0.5 * (Sigma_hat + Sigma_hat.T) + 1e-8 * np.eye(3)

    if d0 is None:
        if debt_col is not None and debt_col in df.columns:
            valid_d = pd.to_numeric(df[debt_col], errors="coerce").dropna()
            if len(valid_d) == 0:
                raise ValueError("No valid observations in debt_col; specify d0 explicitly")
            d0_val = float(valid_d.iloc[-1])
        else:
            raise ValueError("Provide d0 or a valid debt_col in df")
    else:
        d0_val = float(d0)

    reaction_est: FiscalReactionEstimateResult | None = None
    reaction_fn: FiscalReactionFunction | None = None
    if use_bohn_reaction and debt_col is not None and debt_col in df.columns:
        reaction_est = estimate_fiscal_reaction_function(
            df,
            pb_col=pb_col,
            debt_col=debt_col,
            y_gap_col=y_gap_col,
            cubic_fatigue=cubic_fatigue,
        )
        reaction_fn = reaction_est.reaction_function

    dsa_res = simulate_stochastic_dsa(
        d0=d0_val,
        horizon=horizon,
        n_sims=n_sims,
        mu=mu_hat,
        A=A_hat,
        cov=Sigma_hat,
        fiscal_reaction=reaction_fn,
        threshold=threshold,
        seed=seed,
        frequency=frequency,
        store_macro_paths=True,
    )
    meta = dict(dsa_res.metadata)
    meta["var_mu"] = mu_hat
    meta["var_A"] = A_hat
    meta["var_Sigma"] = Sigma_hat
    meta["n_obs_var"] = len(sub)
    if reaction_est is not None:
        meta["fiscal_reaction_estimate"] = reaction_est
    return StochasticDSAResult(
        paths=dsa_res.paths,
        baseline_path=dsa_res.baseline_path,
        d0=dsa_res.d0,
        threshold=dsa_res.threshold,
        horizon=dsa_res.horizon,
        frequency=dsa_res.frequency,
        quantiles=dsa_res.quantiles,
        prob_breach=dsa_res.prob_breach,
        prob_terminal_breach=dsa_res.prob_terminal_breach,
        prob_non_stabilizing=dsa_res.prob_non_stabilizing,
        dar_95=dsa_res.dar_95,
        macro_paths=dsa_res.macro_paths,
        metadata=meta,
    )

