"""Portfolio rebalancing backtester and Monte Carlo wealth glidepath simulator.

Implements:
- Exact weight-drift and rebalancing mechanics across ``"monthly"``, ``"quarterly"``,
  ``"annual"``, ``"buy_and_hold"``, and ``"drift_band"`` policies with proportional
  transaction costs in basis points.
- Monte Carlo forward wealth simulation for accumulation and retirement decumulation,
  including ruin probability and 95%-confidence Safe Withdrawal Rate (SWR).
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from puremacro.portfolio._results import (
    AllocationResult,
    PortfolioBacktestResult,
    WealthSimulationResult,
)
from puremacro.portfolio.allocation import _extract_returns_and_names

__all__ = [
    "backtest_portfolio",
    "simulate_wealth_glidepath",
]


def _max_drawdown_duration(drawdown: np.ndarray) -> int:
    """Compute maximum consecutive periods spent strictly below high-water mark."""
    max_dur = 0
    cur_dur = 0
    for d in drawdown:
        if d < -1e-12:
            cur_dur += 1
            if cur_dur > max_dur:
                max_dur = cur_dur
        else:
            cur_dur = 0
    return int(max_dur)


def backtest_portfolio(
    returns: np.ndarray | pd.DataFrame,
    weights: Sequence[float] | np.ndarray | Mapping[str, float],
    *,
    rebalance: str = "quarterly",
    drift_threshold: float = 0.05,
    tc_bps: float = 10.0,
    rf: float = 0.0,
    periods_per_year: int = 12,
    benchmark: Sequence[float] | np.ndarray | pd.Series | None = None,
    asset_names: Sequence[str] | None = None,
) -> PortfolioBacktestResult:
    """Simulate a multi-asset portfolio with exact weight drift, rebalancing, and transaction costs.

    Parameters
    ----------
    returns : np.ndarray | pd.DataFrame
        Periodic asset return matrix of shape ``(T, N)``.
    weights : Sequence[float] | np.ndarray | Mapping[str, float]
        Target asset allocation weights of length ``N`` (or mapping ``{asset_name: weight}``).
    rebalance : str, default ``"quarterly"``
        Rebalancing rule:
        - ``"monthly"``: Rebalance every month (every ``max(1, periods_per_year // 12)`` periods).
        - ``"quarterly"``: Rebalance every quarter (every ``max(1, periods_per_year // 4)`` periods).
        - ``"annual"`` (or ``"yearly"``): Rebalance once per year (every ``periods_per_year`` periods).
        - ``"buy_and_hold"`` (or ``"none"``): Never rebalance after initial allocation.
        - ``"drift_band"``: Rebalance whenever ``max_i |w_{i,t}^- - w_i^*| > drift_threshold``.
    drift_threshold : float, default 0.05
        Maximum absolute weight deviation before triggering a ``"drift_band"`` rebalance.
    tc_bps : float, default 10.0
        Proportional transaction cost in basis points per unit of turnover
        (``10.0`` bps = ``0.0010 * sum_i |w_{i,t}^+ - w_{i,t}^-|``).
    rf : float, default 0.0
        Annualized risk-free rate.
    periods_per_year : int, default 12
        Number of return periods per year.
    benchmark : Sequence[float] | np.ndarray | pd.Series | None, optional
        Optional 1D periodic benchmark returns of length ``T``.
    asset_names : Sequence[str] | None, optional
        Explicit asset names.

    Returns
    -------
    PortfolioBacktestResult
    """
    if isinstance(returns, pd.DataFrame):
        is_df = True
        time_index = returns.index
    else:
        is_df = False
        time_index = None
    ret_arr, names = _extract_returns_and_names(returns, asset_names)
    if ret_arr.ndim != 2:
        raise ValueError(f"returns must be 2D of shape (T, N); got {ret_arr.shape}")
    T, N = ret_arr.shape
    if T < 1 or N < 1:
        raise ValueError(f"returns must have T >= 1 and N >= 1; got T={T}, N={N}")

    if isinstance(weights, Mapping):
        w_target = np.array([float(weights.get(name, 0.0)) for name in names], dtype=float)
    else:
        w_target = np.asarray(weights, dtype=float).ravel()
    if len(w_target) != N:
        raise ValueError(f"weights length ({len(w_target)}) must match number of assets ({N}).")
    w_sum = float(np.sum(w_target))
    if abs(w_sum) > 1e-12:
        w_target = w_target / w_sum

    reb = rebalance.lower().strip()
    valid_reb = ("monthly", "quarterly", "annual", "yearly", "buy_and_hold", "none", "drift_band")
    if reb not in valid_reb:
        raise ValueError(f"Unknown rebalance policy {rebalance!r}. Expected one of {valid_reb}.")

    step_monthly = max(1, int(periods_per_year) // 12)
    step_quarterly = max(1, int(periods_per_year) // 4)
    step_annual = max(1, int(periods_per_year))
    tc_rate = max(float(tc_bps), 0.0) * 1e-4

    weight_history = np.zeros((T, N), dtype=float)
    turnover_series = np.zeros(T, dtype=float)
    tc_paid_series = np.zeros(T, dtype=float)
    port_returns = np.zeros(T, dtype=float)

    w_minus = w_target.copy()
    for t in range(T):
        if t == 0:
            do_rebalance = False
        elif reb in ("buy_and_hold", "none"):
            do_rebalance = False
        elif reb == "monthly":
            do_rebalance = (t % step_monthly) == 0
        elif reb == "quarterly":
            do_rebalance = (t % step_quarterly) == 0
        elif reb in ("annual", "yearly"):
            do_rebalance = (t % step_annual) == 0
        else:  # drift_band
            do_rebalance = float(np.max(np.abs(w_minus - w_target))) > float(drift_threshold)

        if do_rebalance:
            turnover = float(np.sum(np.abs(w_target - w_minus)))
            tc_cost = tc_rate * turnover
            w_plus = w_target.copy()
        else:
            turnover = 0.0
            tc_cost = 0.0
            w_plus = w_minus.copy()

        weight_history[t, :] = w_plus
        turnover_series[t] = turnover
        tc_paid_series[t] = tc_cost

        gross_ret = float(np.dot(w_plus, ret_arr[t, :]))
        net_ret = (1.0 - tc_cost) * (1.0 + gross_ret) - 1.0
        port_returns[t] = net_ret

        denom = 1.0 + gross_ret
        if abs(denom) > 1e-12:
            w_minus = w_plus * (1.0 + ret_arr[t, :]) / denom
        else:
            w_minus = w_target.copy()

    equity = np.cumprod(1.0 + port_returns)
    running_peak = np.maximum.accumulate(np.maximum(equity, 1.0))
    drawdown = equity / running_peak - 1.0
    max_dd = float(np.max(np.maximum(0.0, -drawdown)))
    max_dd_dur = _max_drawdown_duration(drawdown)

    p_yr = float(periods_per_year)
    terminal_eq = max(float(equity[-1]), 1e-12)
    cagr = float(terminal_eq ** (p_yr / float(T)) - 1.0)
    ann_ret = float(np.mean(port_returns)) * p_yr
    ann_vol = (
        float(np.std(port_returns, ddof=1)) * np.sqrt(p_yr)
        if T > 1
        else 0.0
    )
    sharpe = (ann_ret - float(rf)) / ann_vol if ann_vol > 1e-12 else 0.0

    rf_periodic = float(rf) / p_yr
    excess_periodic = port_returns - rf_periodic
    downside = np.minimum(excess_periodic, 0.0)
    downside_dev = float(np.sqrt(np.mean(downside ** 2))) * np.sqrt(p_yr)
    sortino = (ann_ret - float(rf)) / downside_dev if downside_dev > 1e-12 else sharpe

    calmar = cagr / max_dd if max_dd > 1e-12 else 0.0

    gains = float(np.sum(np.maximum(excess_periodic, 0.0)))
    losses_sum = float(np.sum(np.maximum(-excess_periodic, 0.0)))
    omega = gains / losses_sum if losses_sum > 1e-12 else float(gains > 0.0) * 10.0

    losses_dist = -port_returns
    var_95 = float(np.percentile(losses_dist, 95.0))
    tail_losses = losses_dist[losses_dist >= var_95]
    cvar_95 = float(np.mean(tail_losses)) if tail_losses.size > 0 else var_95
    total_tc = float(np.sum(tc_paid_series))

    bench_eq_out: pd.Series | np.ndarray | None = None
    if benchmark is not None:
        b_arr = np.asarray(benchmark, dtype=float).ravel()
        if len(b_arr) != T:
            raise ValueError(
                f"benchmark length ({len(b_arr)}) must match returns length ({T})."
            )
        b_eq = np.cumprod(1.0 + b_arr)
        bench_eq_out = (
            pd.Series(b_eq, index=time_index, name="benchmark_equity")
            if is_df
            else b_eq
        )

    eq_out: pd.Series | np.ndarray = (
        pd.Series(equity, index=time_index, name="equity") if is_df else equity
    )
    ret_out: pd.Series | np.ndarray = (
        pd.Series(port_returns, index=time_index, name="portfolio_return")
        if is_df
        else port_returns
    )
    dd_out: pd.Series | np.ndarray = (
        pd.Series(drawdown, index=time_index, name="drawdown") if is_df else drawdown
    )

    return PortfolioBacktestResult(
        equity_curve=eq_out,
        portfolio_returns=ret_out,
        drawdown_series=dd_out,
        weight_history=weight_history,
        turnover_series=turnover_series,
        asset_names=names,
        cagr=cagr,
        annualized_return=ann_ret,
        annualized_vol=ann_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        omega_ratio=omega,
        max_drawdown=max_dd,
        max_drawdown_duration=max_dd_dur,
        var_95=var_95,
        cvar_95=cvar_95,
        total_tc_paid=total_tc,
        benchmark_equity=bench_eq_out,
        metadata={
            "rebalance": rebalance,
            "drift_threshold": drift_threshold,
            "tc_bps": tc_bps,
            "rf": rf,
            "periods_per_year": periods_per_year,
        },
    )


def simulate_wealth_glidepath(
    returns_or_alloc: np.ndarray | pd.DataFrame | AllocationResult,
    *,
    weights: Sequence[float] | np.ndarray | None = None,
    initial_wealth: float = 100_000.0,
    horizon_years: int = 30,
    annual_contribution: float = 0.0,
    annual_withdrawal: float = 4_000.0,
    real_return_adjustment: float = 0.0,
    n_sims: int = 1000,
    seed: int = 42,
    periods_per_year: int = 12,
) -> WealthSimulationResult:
    """Simulate Monte Carlo forward wealth paths for accumulation and retirement decumulation.

    Computes fan-chart percentiles ``{5, 10, 25, 50, 75, 90, 95}``, probability of
    ruin, probability of doubling initial wealth, 5% CVaR terminal wealth, and the
    exact 95%-confidence Safe Withdrawal Rate (``safe_withdrawal_rate_95``).

    Parameters
    ----------
    returns_or_alloc : np.ndarray | pd.DataFrame | AllocationResult
        Either an :class:`AllocationResult`, a 2D ``(T, N)`` matrix of periodic
        asset returns, or a 1D ``(T,)`` vector of periodic portfolio returns.
    weights : Sequence[float] | np.ndarray | None, optional
        Portfolio weights when ``returns_or_alloc`` is a 2D return matrix.
    initial_wealth : float, default 100_000.0
        Starting wealth at year 0.
    horizon_years : int, default 30
        Simulation horizon in years.
    annual_contribution : float, default 0.0
        Annual real contribution added at the end of each year.
    annual_withdrawal : float, default 4_000.0
        Annual real withdrawal subtracted at the end of each year.
    real_return_adjustment : float, default 0.0
        Annual inflation or fee drag subtracted from expected return.
    n_sims : int, default 1000
        Number of Monte Carlo trajectories.
    seed : int, default 42
        Random number generator seed.
    periods_per_year : int, default 12
        Number of periods per year when annualizing empirical returns.

    Returns
    -------
    WealthSimulationResult
    """
    if horizon_years < 1:
        raise ValueError(f"horizon_years must be >= 1; got {horizon_years}")
    if n_sims < 1:
        raise ValueError(f"n_sims must be >= 1; got {n_sims}")
    if initial_wealth <= 0.0:
        raise ValueError(f"initial_wealth must be > 0; got {initial_wealth}")

    p_yr = float(periods_per_year)
    if isinstance(returns_or_alloc, AllocationResult):
        mu_ann = float(returns_or_alloc.expected_return)
        vol_ann = float(returns_or_alloc.volatility)
    else:
        arr = (
            returns_or_alloc.to_numpy(dtype=float)
            if isinstance(returns_or_alloc, (pd.DataFrame, pd.Series))
            else np.asarray(returns_or_alloc, dtype=float)
        )
        if arr.ndim == 2:
            N = arr.shape[1]
            if weights is None:
                w = np.full(N, 1.0 / float(N), dtype=float)
            else:
                w = np.asarray(weights, dtype=float).ravel()
                w = w / float(np.sum(w))
            port_r = arr @ w
        elif arr.ndim == 1:
            port_r = arr
        else:
            raise ValueError(f"returns_or_alloc must be 1D or 2D; got shape {arr.shape}")
        mu_ann = float(np.mean(port_r)) * p_yr
        vol_ann = float(np.std(port_r, ddof=1)) * np.sqrt(p_yr) if len(port_r) > 1 else 0.10

    mu_real = mu_ann - float(real_return_adjustment)
    vol_ann = max(vol_ann, 1e-8)

    rng = np.random.default_rng(seed)
    # Lognormal annual gross return multipliers matching arithmetic mean (1 + mu_real) and vol_ann
    drift_log = np.log(max(1.0 + mu_real, 1e-4)) - 0.5 * (vol_ann ** 2)
    z_draws = rng.standard_normal((n_sims, horizon_years))
    gross_returns = np.exp(drift_log + vol_ann * z_draws)

    wealth_paths = np.zeros((n_sims, horizon_years + 1), dtype=float)
    wealth_paths[:, 0] = float(initial_wealth)
    net_cf = float(annual_contribution) - float(annual_withdrawal)

    ruined = np.zeros(n_sims, dtype=bool)
    for y in range(1, horizon_years + 1):
        prev_w = wealth_paths[:, y - 1]
        next_w = prev_w * gross_returns[:, y - 1] + net_cf
        if net_cf <= 0.0:
            newly_ruined = next_w <= 0.0
            ruined = ruined | newly_ruined
            next_w = np.where(ruined, 0.0, np.maximum(next_w, 0.0))
        else:
            next_w = np.maximum(next_w, 0.0)
            ruined = ruined | (next_w <= 0.0)
        wealth_paths[:, y] = next_w

    prob_ruin = float(np.mean(ruined))
    terminal_wealth = wealth_paths[:, -1]
    prob_double = float(np.mean(terminal_wealth >= 2.0 * float(initial_wealth)))
    median_terminal = float(np.median(terminal_wealth))

    q5_cutoff = float(np.percentile(terminal_wealth, 5.0))
    worst_5 = terminal_wealth[terminal_wealth <= q5_cutoff]
    cvar_5_terminal = float(np.mean(worst_5)) if worst_5.size > 0 else q5_cutoff

    # Exact Milevsky-Robinson present-value factor for sustainable withdrawal rate:
    # A path survives all H years iff swr <= 1 / sum_{k=1}^H (prod_{j=1}^k G_j)^{-1}
    cum_gross = np.cumprod(gross_returns, axis=1)
    annuity_factor = np.sum(1.0 / np.maximum(cum_gross, 1e-12), axis=1)
    max_swr_per_path = 1.0 / np.maximum(annuity_factor, 1e-12)
    swr_95 = float(np.percentile(max_swr_per_path, 5.0))

    quantile_levels = (5, 10, 25, 50, 75, 90, 95)
    quantiles: dict[int, np.ndarray] = {
        q: np.percentile(wealth_paths, float(q), axis=0) for q in quantile_levels
    }

    return WealthSimulationResult(
        wealth_paths=wealth_paths,
        quantiles=quantiles,
        initial_wealth=float(initial_wealth),
        horizon_years=int(horizon_years),
        annual_contribution=float(annual_contribution),
        annual_withdrawal=float(annual_withdrawal),
        prob_ruin=prob_ruin,
        prob_double=prob_double,
        median_terminal_wealth=median_terminal,
        cvar_5_terminal_wealth=cvar_5_terminal,
        safe_withdrawal_rate_95=swr_95,
        metadata={
            "mu_annual": mu_ann,
            "mu_real": mu_real,
            "vol_annual": vol_ann,
            "real_return_adjustment": real_return_adjustment,
            "n_sims": n_sims,
            "seed": seed,
            "periods_per_year": periods_per_year,
        },
    )
