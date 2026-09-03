"""Frozen-dataclass result types for puremacro.dsge."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DSGEPosteriorResult:
    """Result of puremacro.dsge.estimate_dsge (and model-specific wrappers
    like puremacro.dsge.estimate_sw07).

    Attributes
    ----------
    draws : ndarray, shape (n_chains, n_draws, n_params)
        Post-burn-in MCMC draws.
    param_names : tuple of str, length n_params
        Parameter names in column order matching `draws`.
    log_posterior_trace : ndarray, shape (n_chains, n_draws)
        Log-posterior at each retained draw.
    accept_rates : tuple of float, length n_chains
        Per-chain acceptance rate over the retained draws.
    mode : dict[str, float]
        Posterior mode (parameter name → value).
    mode_hessian_inv : ndarray, shape (n_params, n_params)
        Inverse Hessian at the mode when scipy.optimize converges + Hessian is PD;
        otherwise falls back to diag(prior_stds**2). Either way, this is the
        proposal-cov foundation that random_walk_metropolis scales by c0**2.
    n_burn_in : int
        Burn-in iterations dropped (also used as proposal-scale adaptation window).
    data_n_obs : int
        Number of observations in the input dataset.
    seed : int
        Master RNG seed.
    model_name : str, default 'unknown'
        Identifier for the underlying DSGE model. ``estimate_sw07`` sets
        this to ``"SW07"``; ``estimate_dsge`` callers can pass whatever
        string they want. New in 0.53.0.
    """
    draws: np.ndarray
    param_names: Tuple[str, ...]
    log_posterior_trace: np.ndarray
    accept_rates: Tuple[float, ...]
    mode: dict
    mode_hessian_inv: np.ndarray
    n_burn_in: int
    data_n_obs: int
    seed: int
    model_name: str = "unknown"

    def summary(self) -> pd.DataFrame:
        """Per-parameter mean, std, 5%/50%/95% quantiles across all chains."""
        flat = self.draws.reshape(-1, len(self.param_names))
        return pd.DataFrame({
            "mean":  flat.mean(axis=0),
            "std":   flat.std(axis=0),
            "q5":    np.quantile(flat, 0.05, axis=0),
            "q50":   np.quantile(flat, 0.50, axis=0),
            "q95":   np.quantile(flat, 0.95, axis=0),
            "mode":  [self.mode[n] for n in self.param_names],
        }, index=list(self.param_names))


# Backward-compatibility alias for code that imports SW07PosteriorResult.
# Resolves to the same class object; isinstance/pickle/type-hints continue
# to work. Drop in 1.0 if appropriate.
SW07PosteriorResult = DSGEPosteriorResult


@dataclass(frozen=True)
class FertilitySolution:
    """Linear solution of the fertility DSGE around its BGP.

    Attributes
    ----------
    ss : dict[str, float]
        Steady-state values keyed by variable name (matches VAR_NAMES).
    params : dict[str, float]
        All parameters used in the solve (structural + calibration +
        shock-process).
    G : ndarray, shape (n_states, n_states)
        State transition (state at t given state at t-1, no shock).
    N : ndarray, shape (n_states, n_shocks)
        Shock impact on states.
    F : ndarray, shape (n_controls, n_states)
        Control policy on the LAGGED state: y_t = F x_{t-1} + L eps_t, which
        is the partition `solve_fertility` actually builds. This entry used to
        read "control at t given state at t", contradicting the solver, and an
        IRF loop written against that wrong reading put every control one
        period early.
    L : ndarray, shape (n_controls, n_shocks)
        Control response to contemporaneous shock.
    klein_solution : KleinSolution or None
        Raw QZ output for debugging.
    var_names : tuple of str
        All 12 endogenous variable names (states first, then controls).
    shock_names : tuple of str
        Shock names (ea, ep, en).

    Notes
    -----
    The first n_states entries of var_names are the predetermined
    variables (rows of G/N); the remaining are controls (rows of F/L).
    """

    ss: dict
    params: dict
    G: np.ndarray
    N: np.ndarray
    F: np.ndarray
    L: np.ndarray
    klein_solution: object
    var_names: tuple
    shock_names: tuple

    def irf(self, shock, horizon: int = 20) -> pd.DataFrame:
        """Impulse response to a 1-SD shock. See fertility_adj_costs.solve_fertility docstring."""
        from puremacro.dsge.fertility_adj_costs import _compute_irf
        return _compute_irf(self, shock, horizon)

    def fevd(self, horizon: int = 20) -> pd.DataFrame:
        """Forecast-error variance decomposition."""
        from puremacro.dsge.fertility_adj_costs import _compute_fevd
        return _compute_fevd(self, horizon)


__all__ = ["DSGEPosteriorResult", "SW07PosteriorResult", "FertilitySolution"]
