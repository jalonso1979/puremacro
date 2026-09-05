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

    def to_frame(self) -> pd.DataFrame:
        """Return summary statistics as a DataFrame."""
        return self.summary()

    def to_markdown(self, **kwargs) -> str:
        """Render summary table as Markdown."""
        from puremacro.reports import _df_to_markdown

        return _df_to_markdown(self.summary(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        """Render summary table as LaTeX tabular."""
        from puremacro.reports import _df_to_latex

        return _df_to_latex(self.summary(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        """Render summary table as Typst table."""
        from puremacro.reports import _df_to_typst

        return _df_to_typst(self.summary(), **kwargs)


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


@dataclass(frozen=True)
class DynareDR:
    """Decision rule representation matching Dynare's oo_.dr structure.

    First-order approximation around steady state:
        y_t = ys + ghx * (x_{t-1} - xs) + ghu * u_t

    Attributes
    ----------
    ghx : pd.DataFrame
        (n_vars x n_states) matrix of policy derivatives with respect to lagged states.
    ghu : pd.DataFrame
        (n_vars x n_shocks) matrix of policy derivatives with respect to contemporaneous shocks.
    ys : pd.Series
        Steady-state values for all endogenous variables.
    state_variables : tuple[str, ...]
        Names of predetermined state variables.
    variable_names : tuple[str, ...]
        Names of all endogenous variables in model order.
    shock_names : tuple[str, ...]
        Names of structural shocks.
    """

    ghx: pd.DataFrame
    ghu: pd.DataFrame
    ys: pd.Series
    state_variables: tuple[str, ...]
    variable_names: tuple[str, ...]
    shock_names: tuple[str, ...]

    def __getitem__(self, key: str):
        """Allow dict-like access matching Dynare MATLAB struct conventions."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"DynareDR has no field {key!r}")

    def to_frame(self) -> pd.DataFrame:
        """Return transition and policy functions matching Dynare's output layout.

        Rows are [Constant, state(-1)..., shocks...], columns are endogenous variables.
        """
        rows = ["Constant"] + [f"{s}(-1)" for s in self.state_variables] + list(self.shock_names)
        df = pd.DataFrame(index=rows, columns=list(self.variable_names), dtype=float)

        # Constant row
        for v in self.variable_names:
            df.loc["Constant", v] = self.ys.get(v, 0.0)

        # Lagged states rows
        for s in self.state_variables:
            row_lbl = f"{s}(-1)"
            for v in self.variable_names:
                df.loc[row_lbl, v] = self.ghx.loc[v, s]

        # Shocks rows
        for e in self.shock_names:
            for v in self.variable_names:
                df.loc[e, v] = self.ghu.loc[v, e]

        return df

    def summary(self) -> str:
        """Render Dynare-style 'POLICY AND TRANSITION FUNCTIONS' table."""
        df = self.to_frame()
        lines = [
            "POLICY AND TRANSITION FUNCTIONS (Dynare Format)",
            "=" * 72,
            df.round(6).to_string(),
            "=" * 72,
        ]
        return "\n".join(lines)

    def to_markdown(self, **kwargs) -> str:
        """Export decision rules to Markdown table."""
        from puremacro.reports import _df_to_markdown

        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        """Export decision rules to LaTeX tabular format."""
        from puremacro.reports import _df_to_latex

        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        """Export decision rules to Typst table format."""
        from puremacro.reports import _df_to_typst

        return _df_to_typst(self.to_frame(), **kwargs)


@dataclass(frozen=True)
class Dynare2ndDR:
    """Second-order decision rule representation matching Dynare's oo_.dr structure.

    Second-order approximation around steady state:
        y_t = ys + 0.5 * ghs2 * sigma^2 + ghx * (x_{t-1} - xs) + ghu * u_t
              + 0.5 * ghxx * ((x_{t-1} - xs) ⊗ (x_{t-1} - xs))
              + ghxu * ((x_{t-1} - xs) ⊗ u_t)
              + 0.5 * ghuu * (u_t ⊗ u_t)

    Attributes
    ----------
    ghx : pd.DataFrame
        (n_vars x n_states) matrix of first-order state policy derivatives.
    ghu : pd.DataFrame
        (n_vars x n_shocks) matrix of first-order shock policy derivatives.
    ghxx : pd.DataFrame
        (n_vars x n_states^2) matrix of second-order state policy derivatives.
    ghxu : pd.DataFrame
        (n_vars x (n_states * n_shocks)) matrix of cross state-shock derivatives.
    ghuu : pd.DataFrame
        (n_vars x n_shocks^2) matrix of second-order shock derivatives.
    ghs2 : pd.Series
        (n_vars,) vector of volatility / risk correction terms.
    ys : pd.Series
        Steady-state values for all endogenous variables.
    state_variables : tuple[str, ...]
        Names of predetermined state variables.
    variable_names : tuple[str, ...]
        Names of all endogenous variables in model order.
    shock_names : tuple[str, ...]
        Names of structural shocks.
    """

    ghx: pd.DataFrame
    ghu: pd.DataFrame
    ghxx: pd.DataFrame
    ghxu: pd.DataFrame
    ghuu: pd.DataFrame
    ghs2: pd.Series
    ys: pd.Series
    state_variables: tuple[str, ...]
    variable_names: tuple[str, ...]
    shock_names: tuple[str, ...]

    def __getitem__(self, key: str):
        """Allow dict-like access matching Dynare MATLAB struct conventions."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Dynare2ndDR has no field {key!r}")

    def to_frame(self) -> pd.DataFrame:
        """Return transition and policy functions matching Dynare layout."""
        rows = ["Constant", "0.5 * ghs2"]
        rows += [f"{s}(-1)" for s in self.state_variables]
        rows += list(self.shock_names)
        df = pd.DataFrame(index=rows, columns=list(self.variable_names), dtype=float)

        for v in self.variable_names:
            df.loc["Constant", v] = self.ys.get(v, 0.0)
            df.loc["0.5 * ghs2", v] = 0.5 * self.ghs2.get(v, 0.0)
            for s in self.state_variables:
                df.loc[f"{s}(-1)", v] = self.ghx.loc[v, s]
            for e in self.shock_names:
                df.loc[e, v] = self.ghu.loc[v, e]

        return df

    def summary(self) -> str:
        """Render Dynare-style 2nd-order policy functions table."""
        df = self.to_frame()
        lines = [
            "SECOND-ORDER POLICY AND TRANSITION FUNCTIONS (Dynare Format)",
            "=" * 72,
            df.round(6).to_string(),
            "-" * 72,
            f"State cross-terms (ghxx) shape : {self.ghxx.shape}",
            f"State-shock terms (ghxu) shape : {self.ghxu.shape}",
            f"Shock cross-terms (ghuu) shape : {self.ghuu.shape}",
            "=" * 72,
        ]
        return "\n".join(lines)

    def to_markdown(self, **kwargs) -> str:
        from puremacro.reports import _df_to_markdown

        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        from puremacro.reports import _df_to_latex

        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        from puremacro.reports import _df_to_typst

        return _df_to_typst(self.to_frame(), **kwargs)


@dataclass(frozen=True)
class TheoreticalMomentsResult:
    """Analytical theoretical moments matching Dynare's stoch_simul.

    Attributes
    ----------
    moments : pd.DataFrame
        Table of [Mean, Std.Dev., Variance] for each endogenous variable.
    covariance : pd.DataFrame
        Unconditional covariance matrix (n_vars x n_vars).
    correlation : pd.DataFrame
        Unconditional correlation matrix (n_vars x n_vars).
    autocorr : pd.DataFrame
        Theoretical autocorrelation coefficients for lags 1 to n_lags.
    fevd : pd.DataFrame
        Forecast error variance decomposition percentage shares across horizons.
    """

    moments: pd.DataFrame
    covariance: pd.DataFrame
    correlation: pd.DataFrame
    autocorr: pd.DataFrame
    fevd: pd.DataFrame

    def summary(self) -> str:
        """Render complete Dynare-style theoretical moments report."""
        lines = [
            "THEORETICAL MOMENTS (Dynare stoch_simul)",
            "=" * 72,
            self.moments.round(6).to_string(),
            "",
            "MATRIX OF CORRELATIONS",
            "-" * 72,
            self.correlation.round(4).to_string(),
            "",
            "COEFFICIENTS OF AUTOCORRELATION",
            "-" * 72,
            self.autocorr.round(4).to_string(),
            "",
            "VARIANCE DECOMPOSITION (in percent)",
            "-" * 72,
            self.fevd.round(2).to_string(),
            "=" * 72,
        ]
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """Return primary theoretical moments table."""
        return self.moments.copy()

    def to_markdown(self, **kwargs) -> str:
        from puremacro.reports import _df_to_markdown

        return _df_to_markdown(self.moments, **kwargs)

    def to_latex(self, **kwargs) -> str:
        from puremacro.reports import _df_to_latex

        return _df_to_latex(self.moments, **kwargs)

    def to_typst(self, **kwargs) -> str:
        from puremacro.reports import _df_to_typst

        return _df_to_typst(self.moments, **kwargs)


__all__ = [
    "DSGEPosteriorResult",
    "SW07PosteriorResult",
    "FertilitySolution",
    "DynareDR",
    "Dynare2ndDR",
    "TheoreticalMomentsResult",
]
