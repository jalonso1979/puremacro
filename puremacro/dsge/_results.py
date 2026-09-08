"""Frozen-dataclass result types for puremacro.dsge."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple

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
    #: ``log p(y | th*) + log p(th*)`` at the reported mode. ``None`` on
    #: results produced before 2.6.0, which is why :meth:`log_mdd` says so
    #: rather than guessing. New in 2.6.0.
    log_post_mode: float | None = None

    def log_mdd(self, method: str = "laplace") -> float:
        """Log marginal likelihood ``log p(y)``.

        ``method="laplace"`` needs :attr:`log_post_mode` and a
        positive-definite :attr:`mode_hessian_inv`; ``method="harmonic"`` uses
        Geweke's modified harmonic mean over the draws and warns when its
        estimate is not stable across truncation levels. See
        :mod:`puremacro.dsge.marginal`.
        """
        from .marginal import harmonic_mean_mdd, laplace_mdd

        if method == "laplace":
            if self.log_post_mode is None:
                raise ValueError(
                    "log_mdd(method='laplace') needs log_post_mode, which this "
                    "result does not carry (it predates puremacro 2.6.0). "
                    "Re-run the estimation, or use method='harmonic'."
                )
            return laplace_mdd(self.log_post_mode, self.mode_hessian_inv)
        if method == "harmonic":
            return harmonic_mean_mdd(self.draws, self.log_posterior_trace).estimate
        raise ValueError(
            f"unknown method {method!r}; expected 'laplace' or 'harmonic'"
        )

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

    Notes
    -----
    ``moments`` / ``covariance`` / ``correlation`` / ``autocorr`` are stated in
    the timing the model reports its variables in -- the same timing as
    ``irf()`` and ``simulate()``. ``fevd`` as filled in by
    :meth:`~puremacro.dsge.LinearModel.theoretical_moments` is built from the
    Dynare-timed ``ghx`` / ``ghu`` decision rules, so for a **Klein-timed**
    model (one built with :func:`~puremacro.dsge.build`) the state rows of
    ``fevd`` are dated one period later than the same rows of ``covariance``:
    ``fevd`` reports the state at the *end* of the period, the moments report
    it at the start. Control rows agree in both.

    :func:`~puremacro.dsge.compute_fevd` (also reachable as
    ``LinearModel.fevd_result()``) does not have this offset -- it decomposes
    the variables in the timing they are reported in, so a predetermined state
    correctly has no one-step forecast error. Prefer it when the decomposition
    has to line up with the covariance block above.
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


@dataclass(frozen=True)
class StochSimulResult:
    """Consolidated result of Dynare stoch_simul execution.

    Attributes
    ----------
    dr : DynareDR | Dynare2ndDR
        First- or second-order decision rule structure (oo_.dr).
    theoretical_moments : TheoreticalMomentsResult
        Analytical unconditional moments, correlations, autocorrelations, and FEVD.
    simulated_moments : pd.DataFrame | None
        Sample moments if simulation with periods > 0 was requested.
    irfs : dict[str, pd.Series]
        Dictionary of impulse response functions keyed by '{var}_{shock}'.
    order : int
        Approximation order (1 or 2).
    variable_names : tuple[str, ...]
        Names of all endogenous variables.
    shock_names : tuple[str, ...]
        Names of structural shocks.
    """

    dr: DynareDR | Dynare2ndDR | Any
    theoretical_moments: TheoreticalMomentsResult
    simulated_moments: pd.DataFrame | None
    irfs: dict[str, pd.Series]
    order: int
    variable_names: tuple[str, ...]
    shock_names: tuple[str, ...]

    def __getitem__(self, key: str):
        """Allow subscript access matching Dynare struct conventions."""
        if hasattr(self, key):
            return getattr(self, key)
        if key in self.irfs:
            return self.irfs[key]
        raise KeyError(f"StochSimulResult has no attribute or IRF series {key!r}")

    def to_frame(self, shock: str | None = None) -> pd.DataFrame:
        """Return IRFs as a DataFrame.

        If shock is provided, returns (H+1 x n_vars) for that shock.
        Otherwise, returns a wide DataFrame of all '{var}_{shock}' IRF paths.
        """
        if shock is not None:
            cols = {
                v: self.irfs[f"{v}_{shock}"]
                for v in self.variable_names
                if f"{v}_{shock}" in self.irfs
            }
            return pd.DataFrame(cols)
        return pd.DataFrame(self.irfs)

    def summary(self) -> str:
        """Render complete consolidated Dynare stoch_simul report."""
        lines = [
            f"DYNARE STOCH_SIMUL REPORT (Order {self.order})",
            "=" * 72,
            f"Endogenous variables : {len(self.variable_names)}",
            f"Exogenous shocks     : {len(self.shock_names)}",
            "-" * 72,
            self.dr.summary(),
            "",
            self.theoretical_moments.summary(),
        ]
        if self.simulated_moments is not None:
            lines += [
                "",
                "MOMENTS OF SIMULATED VARIABLES",
                "-" * 72,
                self.simulated_moments.round(6).to_string(),
                "=" * 72,
            ]
        return "\n".join(lines)

    def to_markdown(self, **kwargs) -> str:
        """Export primary theoretical moments to Markdown."""
        from puremacro.reports import _df_to_markdown

        return _df_to_markdown(self.theoretical_moments.to_frame(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        """Export primary theoretical moments to LaTeX."""
        from puremacro.reports import _df_to_latex

        return _df_to_latex(self.theoretical_moments.to_frame(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        """Export primary theoretical moments to Typst."""
        from puremacro.reports import _df_to_typst

        return _df_to_typst(self.theoretical_moments.to_frame(), **kwargs)

    def plot(
        self,
        variables: Sequence[str] | None = None,
        shocks: Sequence[str] | None = None,
        *,
        style: str = "publication",
        figsize: tuple[float, float] | None = None,
    ):
        """Plot impulse response functions (IRFs).

        Parameters
        ----------
        variables : Sequence[str], optional
            Variables to include. Defaults to first 6 variables.
        shocks : Sequence[str], optional
            Structural shocks to plot. Defaults to first shock.
        style : str, default 'publication'
            Plot styling theme ('publication', 'dark', 'grayscale').
        figsize : tuple[float, float], optional
            Figure dimensions (width, height).

        Returns
        -------
        matplotlib.figure.Figure | None
        """
        import matplotlib.pyplot as plt
        from puremacro.plot import _palette

        if not self.irfs:
            return None

        sel_shocks = list(shocks) if shocks is not None else list(self.shock_names[:1])
        if not sel_shocks and self.shock_names:
            sel_shocks = [self.shock_names[0]]

        sel_vars = list(variables) if variables is not None else list(self.variable_names[:6])
        if not sel_vars and self.variable_names:
            sel_vars = list(self.variable_names)

        pairs = [
            (v, s)
            for s in sel_shocks
            for v in sel_vars
            if f"{v}_{s}" in self.irfs
        ]
        if not pairs:
            return None

        n_plots = len(pairs)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        if figsize is None:
            figsize = (3.8 * n_cols, 2.8 * n_rows)

        if style == "grayscale":
            colors = _palette(max(1, len(pairs)))
        else:
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"] * (len(pairs) // 6 + 1)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        ax_flat = axes.flatten()

        for idx, (v, s) in enumerate(pairs):
            ax = ax_flat[idx]
            series = self.irfs[f"{v}_{s}"]
            h = np.arange(len(series))
            col = colors[idx % len(colors)]
            ax.plot(h, series.values, color=col, lw=1.8, label=f"{v} ({s})")
            ax.axhline(0, color="gray", linestyle="--", lw=0.8, alpha=0.7)
            ax.set_title(f"{v} to {s}", fontsize=10, fontweight="bold")
            ax.set_xlabel("Horizon (periods)", fontsize=8)
            ax.set_ylabel("Dev from SS", fontsize=8)
            ax.grid(True, linestyle=":", alpha=0.5)

        for idx in range(len(pairs), len(ax_flat)):
            ax_flat[idx].set_visible(False)

        fig.tight_layout()
        return fig


from .perfect_foresight import PerfectForesightResult

__all__ = [
    "DSGEPosteriorResult",
    "SW07PosteriorResult",
    "FertilitySolution",
    "DynareDR",
    "Dynare2ndDR",
    "TheoreticalMomentsResult",
    "StochSimulResult",
    "PerfectForesightResult",
]


@dataclass(frozen=True)
class SmootherResult:
    """Kalman-smoother output at calibrated parameters (Dynare ``calib_smoother``).

    Attributes
    ----------
    states : pandas.DataFrame
        Smoothed model states, ``(T, n_states)``, in deviation units.
    shocks : pandas.DataFrame
        Smoothed structural innovations ``E[u_t | y_{1:T}]``, ``(T, n_shocks)``.
        These are read straight off the smoothed state: the filter carries
        ``alpha_t = [x_t; u_t]``, so the innovations are its last ``n_e`` rows
        and no separate disturbance smoother is involved.
    smoothed_obs : pandas.DataFrame
        Fitted observables, ``d + Z a_smooth``, with any declared observation
        trend added back. Equal to the data to machine precision when there is
        no measurement error.
    filtered_states : pandas.DataFrame
        One-sided ``E[x_t | y_{1:t}]``, ``(T, n_states)``.
    loglik : float
        Log-likelihood of the data under the calibrated parameters.
    varobs : tuple of str
        Observables, in data-column order.
    """

    states: pd.DataFrame
    shocks: pd.DataFrame
    smoothed_obs: pd.DataFrame
    filtered_states: pd.DataFrame
    loglik: float
    varobs: Tuple[str, ...]
    _model: Any = None
    _data: Any = None

    def shock_decomposition(self, **kwargs):
        """Historical shock decomposition for the same model and data.

        Delegates to :func:`~puremacro.dsge.compute_shock_decomposition`, which
        runs its own smoother. The two routes are pinned against each other in
        ``tests/test_dsge/test_smoother.py`` rather than assumed to agree.
        """
        if self._model is None or self._data is None:
            raise ValueError(
                "SmootherResult.shock_decomposition() needs the model and data "
                "this result was built from; it was constructed without them."
            )
        from .decomposition import compute_shock_decomposition

        return compute_shock_decomposition(self._model, self._data, **kwargs)

    def to_frame(self) -> pd.DataFrame:
        """Per-shock summary of the smoothed innovations."""
        s = self.shocks
        return pd.DataFrame(
            {
                "mean": s.mean(),
                "std": s.std(ddof=0),
                "min": s.min(),
                "max": s.max(),
            },
            index=list(s.columns),
        )

    def summary(self) -> str:
        lines = [
            "KALMAN SMOOTHER (calibrated parameters)",
            "=" * 60,
            f"observables : {', '.join(self.varobs)}",
            f"periods     : {len(self.smoothed_obs)}",
            f"log-likelihood: {self.loglik:.6f}",
            "",
            "SMOOTHED STRUCTURAL SHOCKS",
            "-" * 60,
            self.to_frame().round(6).to_string(),
        ]
        return "\n".join(lines)

    def plot(self, ax=None, **kwargs):
        """Smoothed structural shocks, one panel per shock."""
        import matplotlib.pyplot as plt

        cols = list(self.shocks.columns)
        if ax is None:
            _, ax = plt.subplots(len(cols), 1, sharex=True,
                                 figsize=(8, 2.0 * len(cols)), squeeze=False)
            ax = ax.ravel()
        ax = np.atleast_1d(ax)
        for a, c in zip(ax, cols):
            a.plot(self.shocks.index, self.shocks[c], **kwargs)
            a.axhline(0.0, color="0.6", lw=0.8)
            a.set_ylabel(c)
        ax[-1].set_xlabel("period")
        return ax

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
class DSGEForecastResult:
    """Unconditional forecast from a solved model.

    Attributes
    ----------
    mean, lower, upper : pandas.DataFrame
        ``(horizon, n_vars)``, indexed by horizon ``1..horizon``. The band is
        ``mean +/- z_{(1+ci)/2} * sd``, with ``sd`` from the forecast-error
        covariance ``Z P_h Z' + H`` — parameter uncertainty is **not** in it.
    ci : float
        Nominal coverage of the band.
    horizon : int
        Number of periods forecast.
    """

    mean: pd.DataFrame
    lower: pd.DataFrame
    upper: pd.DataFrame
    ci: float
    horizon: int

    def to_frame(self) -> pd.DataFrame:
        out = {}
        for c in self.mean.columns:
            out[c] = self.mean[c]
            out[f"{c}_lower"] = self.lower[c]
            out[f"{c}_upper"] = self.upper[c]
        return pd.DataFrame(out, index=self.mean.index)

    def summary(self) -> str:
        pct = int(round(100 * self.ci))
        lines = [
            f"DSGE FORECAST ({self.horizon} periods, {pct}% band)",
            "=" * 60,
            "The band reflects shock uncertainty only: parameters are held "
            "fixed at the values the model was solved with.",
            "",
            self.to_frame().round(6).to_string(),
        ]
        return "\n".join(lines)

    def plot(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt

        cols = list(self.mean.columns)
        if ax is None:
            _, ax = plt.subplots(len(cols), 1, sharex=True,
                                 figsize=(8, 2.4 * len(cols)), squeeze=False)
            ax = ax.ravel()
        ax = np.atleast_1d(ax)
        for a, c in zip(ax, cols):
            a.plot(self.mean.index, self.mean[c], **kwargs)
            a.fill_between(self.mean.index, self.lower[c], self.upper[c], alpha=0.25)
            a.set_ylabel(c)
        ax[-1].set_xlabel("horizon")
        return ax

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
class ModeCheckResult:
    """One-parameter slices of an objective through a candidate mode.

    Attributes
    ----------
    slices : dict[str, pandas.DataFrame]
        Per parameter, a frame with ``value`` and ``objective`` columns: the
        objective with every other parameter held at the mode.
    peaks_at_mode : dict[str, bool]
        Whether the slice attains its minimum at the mode. A ``False`` here is
        the most common sign that the reported mode is not one.
    mode : dict[str, float]
        The point the slices pass through.
    objective_at_mode : float
        The objective there. ``find_mode`` minimises, so this is a negative log
        posterior, not a log posterior.
    """

    slices: dict
    peaks_at_mode: dict
    mode: dict
    objective_at_mode: float

    @property
    def failures(self) -> Tuple[str, ...]:
        """Parameters whose slice bottoms out away from the mode."""
        return tuple(n for n, ok in self.peaks_at_mode.items() if not ok)

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for name, sl in self.slices.items():
            j = int(sl["objective"].idxmin())
            rows.append({
                "mode": self.mode[name],
                "best_on_slice": float(sl["value"].iloc[j]),
                "objective_gain": float(self.objective_at_mode - sl["objective"].iloc[j]),
                "peaks_at_mode": self.peaks_at_mode[name],
            })
        return pd.DataFrame(rows, index=list(self.slices))

    def summary(self) -> str:
        bad = self.failures
        lines = [
            "MODE CHECK (one-parameter slices)",
            "=" * 60,
            f"objective at the mode: {self.objective_at_mode:.6f}",
            "",
            self.to_frame().round(6).to_string(),
            "",
        ]
        if bad:
            lines.append(
                f"{len(bad)} parameter(s) improve away from the reported mode: "
                f"{', '.join(bad)}. The point supplied is not a mode in those "
                "directions — re-run the search from the better point."
            )
        else:
            lines.append(
                "Every slice bottoms out at the reported mode. That is "
                "necessary, not sufficient: these are one-parameter slices, "
                "so they say nothing about directions in between."
            )
        return "\n".join(lines)

    def plot(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt

        names = list(self.slices)
        if ax is None:
            ncol = min(3, len(names))
            nrow = int(np.ceil(len(names) / ncol))
            _, ax = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.6 * nrow),
                                 squeeze=False)
            ax = ax.ravel()
        ax = np.atleast_1d(ax)
        for a, name in zip(ax, names):
            sl = self.slices[name]
            a.plot(sl["value"], sl["objective"], **kwargs)
            a.axvline(self.mode[name], color="0.5", ls="--", lw=0.9)
            a.set_title(name + ("" if self.peaks_at_mode[name] else "  (!)"))
        return ax

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
class DiagnosticFinding:
    """A single diagnostic issue or verification confirmation."""

    category: str  # "steady_state", "static_rank", "incidence", "pencil", "unit_root", "stochastic_singularity"
    severity: str  # "error", "warning", "info"
    message: str
    details: Any = None

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] ({self.category}) {self.message}"


@dataclass(frozen=True)
class EigenvalueTable:
    """Frozen dataclass representing the DSGE eigenvalue spectrum and determinacy.

    Attributes
    ----------
    eigenvalues : np.ndarray
        Complex 1D array of generalized eigenvalues, sorted by modulus ascending.
    modulus : np.ndarray
        Float 1D array of eigenvalue moduli |lambda_i|.
    real : np.ndarray
        Float 1D array of real components Re(lambda_i).
    imag : np.ndarray
        Float 1D array of imaginary components Im(lambda_i).
    is_explosive : np.ndarray
        Boolean 1D array indicating whether |lambda_i| >= qz_criterium.
    is_unit_root : np.ndarray
        Boolean 1D array indicating whether ||lambda_i| - 1.0| <= 1e-5.
    n_explosive : int
        Total number of explosive eigenvalues.
    n_forward : int
        Total number of forward-looking (control) variables.
    is_determinate : bool
        True if n_explosive == n_forward and pencil is regular.
    bk_status : str
        Human-readable Blanchard-Kahn verdict message.
    loadings : dict[int, dict[str, float]] | None
        Mapping from offending root index to top variable loadings (|v_ij|),
        populated only when is_determinate is False.
    variables : tuple[str, ...]
        Names of all variables in the system.
    qz_criterium : float, default 1.0 + 1e-6
        Threshold modulus above which an eigenvalue is classified as explosive.
    """

    eigenvalues: np.ndarray
    modulus: np.ndarray
    real: np.ndarray
    imag: np.ndarray
    is_explosive: np.ndarray
    is_unit_root: np.ndarray
    n_explosive: int
    n_forward: int
    is_determinate: bool
    bk_status: str
    loadings: dict[int, dict[str, float]] | None
    variables: tuple[str, ...]
    qz_criterium: float = 1.0 + 1e-6

    @property
    def bk_satisfied(self) -> bool:
        """Alias for is_determinate."""
        return self.is_determinate

    @property
    def offending_loadings(self) -> dict[int, dict[str, float]] | None:
        """Alias for loadings."""
        return self.loadings

    @property
    def summary_message(self) -> str:
        """Alias for bk_status."""
        return self.bk_status

    @property
    def n_stable(self) -> int:
        """Total number of stable eigenvalues (|lambda| < qz_criterium)."""
        return int(np.sum(~self.is_explosive))

    def to_frame(self) -> pd.DataFrame:
        """Return eigenvalue details as a DataFrame."""
        idx = [f"root_{i+1}" for i in range(len(self.eigenvalues))]
        return pd.DataFrame(
            {
                "modulus": self.modulus,
                "real": self.real,
                "imag": self.imag,
                "is_explosive": self.is_explosive,
                "is_unit_root": self.is_unit_root,
            },
            index=idx,
        )

    def summary(self) -> str:
        lines = [
            f"EIGENVALUES & BLANCHARD-KAHN DIAGNOSTICS (qz_criterium={self.qz_criterium:.6f})",
            "=" * 72,
            f"Status       : {self.bk_status}",
            f"Determinacy  : {'UNIQUE STABLE EQUILIBRIUM' if self.is_determinate else 'DETERMINACY FAILED'}",
            f"Forward vars : {self.n_forward}",
            f"Explosive    : {self.n_explosive}",
            f"Stable       : {self.n_stable}",
            f"Unit roots   : {int(np.sum(self.is_unit_root))}",
            "",
            "EIGENVALUE SPECTRUM",
            "-" * 72,
            self.to_frame().round(6).to_string(),
        ]
        if self.loadings:
            lines.extend([
                "",
                "OFFENDING ROOT VARIABLE LOADINGS",
                "-" * 72,
            ])
            for idx, lds in self.loadings.items():
                mod = self.modulus[idx] if idx < len(self.modulus) else float("nan")
                top_items = [f"{v}: {w:.4f}" for v, w in lds.items()]
                lines.append(f"Root #{idx+1} (|lambda| = {mod:.4f}): {', '.join(top_items)}")
        return "\n".join(lines)

    def plot(self, ax=None, **kwargs):
        """Plot eigenvalues on the complex plane against the unit circle."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        if len(self.eigenvalues) == 0:
            ax.text(0.5, 0.5, "No eigenvalues to plot", ha="center", va="center")
            return ax

        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), color="#4a7bb0", linestyle="--", linewidth=1.2, label="Unit circle (|lambda| = 1)")

        finite_mask = np.isfinite(self.real) & np.isfinite(self.imag)
        re = self.real[finite_mask]
        im = self.imag[finite_mask]
        exp = self.is_explosive[finite_mask]
        unit = self.is_unit_root[finite_mask]
        stable = (~exp) & (~unit)

        if np.any(stable):
            ax.scatter(re[stable], im[stable], color="#2ca02c", marker="o", s=40, label=f"Stable ({np.sum(stable)})", zorder=3)
        if np.any(unit):
            ax.scatter(re[unit], im[unit], color="#ff7f0e", marker="D", s=50, label=f"Unit root ({np.sum(unit)})", zorder=4)
        if np.any(exp):
            ax.scatter(re[exp], im[exp], color="#d62728", marker="s", s=45, label=f"Explosive ({np.sum(exp)})", zorder=3)

        if self.loadings:
            for idx, lds in self.loadings.items():
                if idx < len(self.eigenvalues) and finite_mask[idx]:
                    rx, ix = self.real[idx], self.imag[idx]
                    ax.scatter([rx], [ix], facecolors="none", edgecolors="#17becf", s=130, linewidth=2.0, zorder=5)
                    top_vars = list(lds.keys())[:2]
                    callout = f"{self.modulus[idx]:.2f} ({', '.join(top_vars)})"
                    ax.annotate(callout, (rx, ix), textcoords="offset points", xytext=(5, 5), fontsize=8, color="#17becf", fontweight="bold")

        n_inf = int(np.sum(~finite_mask))
        title = f"Eigenvalue Spectrum (n_fwd={self.n_forward}, n_exp={self.n_explosive})"
        if n_inf > 0:
            title += f" [{n_inf} infinite deflated]"
        ax.set_title(title)
        ax.set_xlabel("Re(lambda)")
        ax.set_ylabel("Im(lambda)")
        ax.axhline(0, color="0.7", linestyle=":", linewidth=0.8)
        ax.axvline(0, color="0.7", linestyle=":", linewidth=0.8)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.set_aspect("equal", adjustable="datalim")
        return ax

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
class ModelDiagnosticsResult:
    """Frozen dataclass containing the results of DSGE model diagnostics.

    Attributes
    ----------
    passed : bool
        True if no error-severity findings were detected.
    findings : tuple[DiagnosticFinding, ...]
        Tuple of diagnostic findings.
    static_rank : int
        Numerical rank of the static Jacobian.
    static_n_vars : int
        Number of variables evaluated in the static Jacobian.
    collinear_equations : tuple[str, ...]
        Names or combinations of collinear equations identified by SVD.
    collinear_variables : tuple[str, ...]
        Names or combinations of unconstrained variables identified by SVD.
    unused_variables : tuple[str, ...]
        Variables that enter zero equations across all evaluation points.
    unused_equations : tuple[str, ...]
        Equations that depend on zero variables across all evaluation points.
    is_pencil_regular : bool
        True if the dynamic matrix pencil (A, B) is regular.
    stochastic_singularity : bool
        True if n_varobs > n_shocks + n_measurement_errors.
    unit_roots : tuple[int, ...]
        Indices of detected unit roots.
    incidence_matrix : np.ndarray
        Boolean incidence matrix (n_equations, n_variables), union of eval points.
    eval_points : int, default 5
        Number of neighbourhood points evaluated.
    variable_names : tuple[str, ...]
        Names of endogenous variables corresponding to columns of incidence_matrix.
    equation_names : tuple[str, ...]
        Names or labels of equations corresponding to rows of incidence_matrix.
    """

    passed: bool
    findings: tuple[DiagnosticFinding, ...]
    static_rank: int
    static_n_vars: int
    collinear_equations: tuple[str, ...]
    collinear_variables: tuple[str, ...]
    unused_variables: tuple[str, ...] = ()
    unused_equations: tuple[str, ...] = ()
    is_pencil_regular: bool = True
    stochastic_singularity: bool = False
    unit_roots: tuple[int, ...] = ()
    incidence_matrix: np.ndarray | None = None
    eval_points: int = 5
    variable_names: tuple[str, ...] = ()
    equation_names: tuple[str, ...] = ()

    @property
    def rank_deficient(self) -> bool:
        return self.static_rank < self.static_n_vars

    @property
    def n_vars(self) -> int:
        return self.static_n_vars

    @property
    def pencil_regular(self) -> bool:
        return self.is_pencil_regular

    @property
    def redundant_equations(self) -> tuple[str, ...]:
        return self.unused_equations

    @property
    def unit_roots_detected(self) -> int:
        return len(self.unit_roots)

    def to_frame(self) -> pd.DataFrame:
        """Return findings as a DataFrame."""
        if not self.findings:
            return pd.DataFrame([{
                "category": "all",
                "severity": "info",
                "message": "All diagnostic checks passed successfully.",
            }])
        return pd.DataFrame([{
            "category": f.category,
            "severity": f.severity,
            "message": f.message,
        } for f in self.findings])

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        err_count = sum(1 for f in self.findings if f.severity == "error")
        warn_count = sum(1 for f in self.findings if f.severity == "warning")
        lines = [
            "DSGE MODEL DIAGNOSTICS",
            "=" * 72,
            f"Overall status         : {status} ({err_count} errors, {warn_count} warnings)",
            f"Static Jacobian rank   : {self.static_rank} / {self.static_n_vars} "
            f"({'RANK DEFICIENT' if self.rank_deficient else 'FULL RANK'})",
            f"Dynamic pencil regular : {'YES' if self.is_pencil_regular else 'NO (SINGULAR)'}",
            f"Stochastic singularity : {'YES' if self.stochastic_singularity else 'NO'}",
            f"Unit roots detected    : {self.unit_roots_detected}",
            f"Unused variables       : {len(self.unused_variables)}" + (f" ({', '.join(self.unused_variables)})" if self.unused_variables else ""),
            f"Unused equations       : {len(self.unused_equations)}" + (f" ({', '.join(self.unused_equations)})" if self.unused_equations else ""),
        ]
        if self.collinear_equations:
            lines.append("Collinear equations    : " + "; ".join(self.collinear_equations))
        if self.collinear_variables:
            lines.append("Collinear variables    : " + "; ".join(self.collinear_variables))
        lines.extend([
            "",
            "DIAGNOSTIC FINDINGS",
            "-" * 72,
            self.to_frame().to_string(),
        ])
        return "\n".join(lines)

    def plot(self, ax=None, **kwargs):
        """Plot the numeric incidence matrix."""
        import matplotlib.pyplot as plt

        if self.incidence_matrix is None:
            if ax is None:
                fig, ax = plt.subplots(figsize=(5, 3))
            ax.text(0.5, 0.5, "No incidence matrix available", ha="center", va="center")
            return ax

        mat = self.incidence_matrix.astype(float)
        if ax is None:
            n_eq, n_var = mat.shape
            fig, ax = plt.subplots(figsize=(max(5, n_var * 0.4), max(4, n_eq * 0.3)))

        ax.imshow(mat, cmap="Blues", aspect="auto", vmin=0, vmax=1)
        ax.set_title(f"Model Incidence Matrix ({mat.shape[0]} eqs x {mat.shape[1]} vars, {self.eval_points}-pt union)")

        if self.variable_names and len(self.variable_names) == mat.shape[1]:
            ax.set_xticks(range(len(self.variable_names)))
            ax.set_xticklabels(self.variable_names, rotation=90, fontsize=8)
        else:
            ax.set_xlabel("Variables")

        if self.equation_names and len(self.equation_names) == mat.shape[0]:
            ax.set_yticks(range(len(self.equation_names)))
            ax.set_yticklabels(self.equation_names, fontsize=8)
        else:
            ax.set_ylabel("Equations")

        ax.grid(True, which="both", color="0.8", linestyle=":", linewidth=0.5)
        return ax

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
class IdentificationResult:
    """Frozen dataclass containing Iskrev (2010) parameter identification diagnostics.

    Attributes
    ----------
    is_identified : bool
        True if both J1 (reduced-form solution) and J2 (theoretical moments) are full rank.
    j1_rank : int
        Numerical rank of state-space Jacobian J1 = d vec(T, R, Q, Z, H) / d theta.
    j1_n_params : int
        Number of parameters tested in J1.
    j1_null_space : np.ndarray
        Orthonormal basis of J1 null space, shape (n_null, n_params).
    j1_null_combinations : tuple[str, ...]
        Human-readable linear parameter combinations spanning J1 null space.
    j1_collinearity : pd.DataFrame
        Pairwise and multi-way collinearity R^2 for J1 columns.
    j2_rank : int
        Numerical rank of theoretical moment Jacobian J2 = d m(theta) / d theta.
    j2_n_params : int
        Number of parameters tested in J2.
    j2_null_space : np.ndarray
        Orthonormal basis of J2 null space, shape (n_null, n_params).
    j2_null_combinations : tuple[str, ...]
        Human-readable linear parameter combinations spanning J2 null space.
    j2_collinearity : pd.DataFrame
        Pairwise and multi-way collinearity R^2 for J2 columns.
    strength : pd.DataFrame
        Ratto (2011) identification strength and sensitivity per parameter.
    param_names : tuple[str, ...]
        Names of evaluated parameters.
    varobs : tuple[str, ...]
        Observables used in identification analysis.
    lags : int, default 1
        Autocovariance lags included in J2 moments.
    prior_mc_results : dict | None, default None
        Optional Monte Carlo identification results over prior distribution.
    """

    is_identified: bool
    j1_rank: int
    j1_n_params: int
    j1_null_space: np.ndarray
    j1_null_combinations: tuple[str, ...]
    j1_collinearity: pd.DataFrame
    j2_rank: int
    j2_n_params: int
    j2_null_space: np.ndarray
    j2_null_combinations: tuple[str, ...]
    j2_collinearity: pd.DataFrame
    strength: pd.DataFrame
    param_names: tuple[str, ...]
    varobs: tuple[str, ...]
    lags: int = 1
    prior_mc_results: dict | None = None

    @property
    def rank_deficient(self) -> bool:
        """True if either J1 or J2 is rank deficient."""
        return not self.is_identified

    @property
    def j1_rank_deficient(self) -> bool:
        """True if state-space Jacobian J1 is rank deficient."""
        return self.j1_rank < self.j1_n_params

    @property
    def j2_rank_deficient(self) -> bool:
        """True if moment Jacobian J2 is rank deficient."""
        return self.j2_rank < self.j2_n_params

    @property
    def n_params(self) -> int:
        """Total number of parameters evaluated."""
        return len(self.param_names)

    def to_frame(self) -> pd.DataFrame:
        """Return parameter identification summary table as a DataFrame."""
        rows = []
        for p in self.param_names:
            j1_r2 = float(self.j1_collinearity.loc[p, "r2"]) if (p in self.j1_collinearity.index and "r2" in self.j1_collinearity.columns) else 0.0
            j2_r2 = float(self.j2_collinearity.loc[p, "r2"]) if (p in self.j2_collinearity.index and "r2" in self.j2_collinearity.columns) else 0.0
            sens = float(self.strength.loc[p, "sensitivity"]) if (p in self.strength.index and "sensitivity" in self.strength.columns) else 0.0
            st = float(self.strength.loc[p, "strength"]) if (p in self.strength.index and "strength" in self.strength.columns) else 0.0
            norm_st = float(self.strength.loc[p, "normalized_strength"]) if (p in self.strength.index and "normalized_strength" in self.strength.columns) else 0.0
            is_ident = bool((j1_r2 < 0.999) and (j2_r2 < 0.999) and (sens > 1e-8))
            rows.append({
                "j1_collinearity": j1_r2,
                "j2_collinearity": j2_r2,
                "sensitivity": sens,
                "strength": st,
                "normalized_strength": norm_st,
                "identified": is_ident,
            })
        return pd.DataFrame(rows, index=list(self.param_names))

    def summary(self) -> str:
        """Render human-readable identification diagnostics summary."""
        status_str = "IDENTIFIED" if self.is_identified else "UNIDENTIFIED (RANK DEFICIENT)"
        lines = [
            "PARAMETER IDENTIFICATION ANALYSIS (Iskrev 2010 / Ratto 2011)",
            "=" * 72,
            f"Overall status         : {status_str}",
            f"Parameters evaluated   : {len(self.param_names)}",
            f"Observables (varobs)   : {', '.join(self.varobs)}",
            f"Autocovariance lags    : {self.lags}",
            f"J1 (Solution) rank     : {self.j1_rank} / {self.j1_n_params} "
            f"({'FULL RANK' if not self.j1_rank_deficient else f'DEFICIENT by {self.j1_n_params - self.j1_rank}'})",
            f"J2 (Moments) rank      : {self.j2_rank} / {self.j2_n_params} "
            f"({'FULL RANK' if not self.j2_rank_deficient else f'DEFICIENT by {self.j2_n_params - self.j2_rank}'})",
        ]
        if self.j1_null_combinations:
            lines.extend([
                "",
                "J1 NULL SPACE PARAMETER COMBINATIONS",
                "-" * 72,
            ])
            for comb in self.j1_null_combinations:
                lines.append(f"  {comb}")

        if self.j2_null_combinations:
            lines.extend([
                "",
                "J2 NULL SPACE PARAMETER COMBINATIONS",
                "-" * 72,
            ])
            for comb in self.j2_null_combinations:
                lines.append(f"  {comb}")

        if not self.j1_null_combinations and not self.j2_null_combinations:
            lines.extend([
                "",
                "NULL SPACE DIRECTIONS",
                "-" * 72,
                "  None (All parameters locally identified)",
            ])

        lines.extend([
            "",
            "PARAMETER IDENTIFICATION SUMMARY",
            "-" * 72,
            self.to_frame().round(4).to_string(),
            "=" * 72,
        ])
        return "\n".join(lines)

    def plot(self, ax=None, **kwargs):
        """Plot parameter collinearity R^2 and identification strength."""
        import matplotlib.pyplot as plt

        n_p = len(self.param_names)
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, max(4.0, n_p * 0.45)))

        if n_p == 0:
            ax.text(0.5, 0.5, "No parameters evaluated", ha="center", va="center")
            return ax

        y_pos = np.arange(n_p)
        height = 0.35

        r2_vals = (
            self.j2_collinearity["r2"].to_numpy()
            if "r2" in self.j2_collinearity.columns
            else np.zeros(n_p)
        )
        st_vals = (
            self.strength["normalized_strength"].to_numpy()
            if "normalized_strength" in self.strength.columns
            else (
                self.strength["strength"].to_numpy()
                if "strength" in self.strength.columns
                else np.zeros(n_p)
            )
        )

        ax.barh(
            y_pos - height / 2,
            r2_vals,
            height=height,
            color="#4a7bb0",
            alpha=0.85,
            label="Collinearity $R^2$ (J2)",
        )
        ax.barh(
            y_pos + height / 2,
            st_vals,
            height=height,
            color="#2ca02c",
            alpha=0.85,
            label="Norm. Identification Strength",
        )
        ax.axvline(
            1.0,
            color="#d62728",
            linestyle="--",
            linewidth=1.2,
            alpha=0.7,
            label="Collinear (1.0)",
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(list(self.param_names))
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Metric Value [0, 1]")
        ax.set_title("DSGE Parameter Identification: Collinearity vs Strength")
        ax.grid(True, axis="x", linestyle=":", alpha=0.5)
        ax.legend(loc="lower right", fontsize=8)

        return ax

    def to_markdown(self, **kwargs) -> str:
        """Render summary table as Markdown."""
        from puremacro.reports import _df_to_markdown

        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        """Render summary table as LaTeX tabular."""
        from puremacro.reports import _df_to_latex

        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        """Render summary table as Typst table."""
        from puremacro.reports import _df_to_typst

        return _df_to_typst(self.to_frame(), **kwargs)


@dataclass(frozen=True)
class OSRResult:
    """Frozen dataclass containing Optimal Simple Rule optimization results.

    Attributes
    ----------
    optimal_params : dict[str, float]
        Dictionary of optimal rule parameter values.
    initial_params : dict[str, float]
        Dictionary of baseline/initial rule parameter values.
    loss_opt : float
        Value of quadratic loss at the optimum.
    loss_initial : float
        Value of quadratic loss at initial parameters.
    rule_params : tuple[str, ...]
        Names of optimized rule parameters.
    target_vars : tuple[str, ...]
        Names of target variables in loss function.
    weights : dict[str, float]
        Weights assigned to each target variable.
    variance_table : pd.DataFrame
        DataFrame detailing variances, weights, and weighted losses initially and at optimum.
    converged : bool
        True if the optimizer reported successful convergence.
    message : str
        Termination status message from optimizer.
    n_evaluations : int
        Total function evaluations.
    optimal_model : Any = None
        Model re-solved at optimal parameter values.
    """

    optimal_params: dict[str, float]
    initial_params: dict[str, float]
    loss_opt: float
    loss_initial: float
    rule_params: tuple[str, ...]
    target_vars: tuple[str, ...]
    weights: dict[str, float]
    variance_table: pd.DataFrame
    converged: bool
    message: str
    n_evaluations: int
    optimal_model: Any = None

    @property
    def loss_init(self) -> float:
        """Alias for loss_initial."""
        return self.loss_initial

    @property
    def loss_calib(self) -> float:
        """Alias for loss_initial."""
        return self.loss_initial

    def to_frame(self) -> pd.DataFrame:
        """Return canonical DataFrame representation of the variance comparison table."""
        return self.variance_table.copy()

    def summary(self) -> str:
        """Render human-readable summary of OSR optimization results."""
        improvement_pct = 0.0
        if self.loss_initial > 0:
            improvement_pct = 100.0 * (self.loss_initial - self.loss_opt) / self.loss_initial

        lines = [
            "OPTIMAL SIMPLE RULES (OSR) OPTIMIZATION",
            "=" * 72,
            f"Convergence status      : {'CONVERGED' if self.converged else 'TERMINATED'} ({self.message})",
            f"Function evaluations    : {self.n_evaluations}",
            f"Baseline loss           : {self.loss_initial:.6e}",
            f"Optimal loss            : {self.loss_opt:.6e}",
            f"Loss reduction          : {improvement_pct:.2f}%",
            "",
            "RULE PARAMETERS",
            "-" * 72,
            f"{'Parameter':<20} {'Initial':>15} {'Optimal':>15} {'Change':>15}",
            "-" * 72,
        ]
        for p in self.rule_params:
            init_val = self.initial_params.get(p, np.nan)
            opt_val = self.optimal_params.get(p, np.nan)
            chg = opt_val - init_val if not (np.isnan(init_val) or np.isnan(opt_val)) else np.nan
            lines.append(f"{p:<20} {init_val:>15.6f} {opt_val:>15.6f} {chg:>+15.6f}")

        lines.extend([
            "",
            "TARGET VARIABLE VARIANCES",
            "-" * 72,
            self.variance_table.round(6).to_string(),
            "=" * 72,
        ])
        return "\n".join(lines)

    def plot(self, ax=None, **kwargs):
        """Plot grouped bar chart comparing variances under baseline vs optimal rule."""
        import matplotlib.pyplot as plt

        df = self.variance_table
        vars_list = list(df.index)
        n_vars = len(vars_list)

        if ax is None:
            fig, ax = plt.subplots(figsize=(max(6.0, n_vars * 1.5), 4.5))

        if n_vars == 0:
            ax.text(0.5, 0.5, "No target variables to plot", ha="center", va="center")
            return ax

        x = np.arange(n_vars)
        width = 0.35

        var_init = (
            df["var_initial"].to_numpy()
            if "var_initial" in df.columns
            else (df["var_calib"].to_numpy() if "var_calib" in df.columns else np.zeros(n_vars))
        )
        var_opt = (
            df["var_optimal"].to_numpy()
            if "var_optimal" in df.columns
            else (df["var_opt"].to_numpy() if "var_opt" in df.columns else np.zeros(n_vars))
        )

        ax.bar(x - width / 2, var_init, width, label="Baseline", color="#4a7bb0", alpha=0.85)
        ax.bar(x + width / 2, var_opt, width, label="Optimal Rule", color="#2ca02c", alpha=0.85)

        ax.set_ylabel("Theoretical Variance")
        ax.set_title("Optimal Simple Rules: Variance Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(vars_list)
        ax.legend()
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)
        return ax

    def to_markdown(self, **kwargs) -> str:
        """Render summary variance table as Markdown."""
        from puremacro.reports import _df_to_markdown

        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        """Render summary variance table as LaTeX tabular."""
        from puremacro.reports import _df_to_latex

        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        """Render summary variance table as Typst table."""
        from puremacro.reports import _df_to_typst

        return _df_to_typst(self.to_frame(), **kwargs)


@dataclass(frozen=True)
class PolicyResult:
    """Frozen dataclass containing optimal policy regime results.

    Attributes
    ----------
    regime : str
        "discretion" or "commitment".
    target_vars : tuple[str, ...]
        Target variable names.
    weights : dict[str, float]
        Weights on target variables.
    instruments : tuple[str, ...]
        Names of policy instruments.
    beta : float
        Policymaker discount factor.
    loss : float
        Expected unconditional quadratic loss.
    policy_rules : pd.DataFrame
        Reaction function coefficients expressing instrument(s) in terms of states.
    transition_matrix : np.ndarray
        Closed-loop state transition matrix G.
    impact_matrix : np.ndarray
        Closed-loop shock loading matrix N.
    multipliers : tuple[str, ...]
        Names of Lagrange multiplier variables (empty in discretion).
    linear_model : Any
        The solved LinearModel under the policy regime, enabling .irf(), .fevd(),
        .simulate(), and .theoretical_moments().
    """

    regime: str
    target_vars: tuple[str, ...]
    weights: dict[str, float]
    instruments: tuple[str, ...]
    beta: float
    loss: float
    policy_rules: pd.DataFrame
    transition_matrix: np.ndarray
    impact_matrix: np.ndarray
    multipliers: tuple[str, ...]
    linear_model: Any = None

    @property
    def augmented_model(self) -> Any:
        """Alias for linear_model."""
        return self.linear_model

    def to_frame(self) -> pd.DataFrame:
        """Return canonical DataFrame representation of the policy rule reaction coefficients."""
        return self.policy_rules.copy()

    def summary(self) -> str:
        """Render human-readable summary of policy regime equilibrium."""
        lines = [
            f"OPTIMAL POLICY REGIME: {self.regime.upper()}",
            "=" * 72,
            f"Policymaker discount (beta): {self.beta:.4f}",
            f"Expected unconditional loss : {self.loss:.6e}",
            f"Policy instruments          : {', '.join(self.instruments)}",
            f"Target variables            : {', '.join(f'{k} (w={v})' for k, v in self.weights.items())}",
        ]
        if self.multipliers:
            lines.append(f"Lagrange multipliers        : {', '.join(self.multipliers)}")
        lines.extend([
            "",
            "POLICY REACTION FUNCTIONS (Coefficients on States)",
            "-" * 72,
            self.policy_rules.round(6).to_string(),
            "=" * 72,
        ])
        return "\n".join(lines)

    def plot(self, ax=None, periods: int = 16, shock: str | None = None, **kwargs):
        """Plot impulse responses under the optimal policy regime."""
        import matplotlib.pyplot as plt

        if self.linear_model is None:
            if ax is None:
                fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No linear model attached", ha="center", va="center")
            return ax

        shock_name = shock if shock is not None else (self.linear_model.shocks[0] if self.linear_model.shocks else None)
        if shock_name is None:
            if ax is None:
                fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No shocks in model", ha="center", va="center")
            return ax
        irf_df = self.linear_model.irf(shock_name, horizon=periods)
        plot_vars = [v for v in self.target_vars if v in irf_df.columns]
        for inst in self.instruments:
            if inst in irf_df.columns and inst not in plot_vars:
                plot_vars.append(inst)
        for mult in self.multipliers:
            if mult in irf_df.columns and mult not in plot_vars:
                plot_vars.append(mult)

        if not plot_vars:
            plot_vars = list(irf_df.columns[:min(4, len(irf_df.columns))])

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4.5))

        for v in plot_vars:
            ax.plot(irf_df.index, irf_df[v], label=v, linewidth=1.8)

        ax.axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Deviation")
        ax.set_title(f"Optimal Policy ({self.regime.capitalize()}): Impulse Responses")
        ax.legend(loc="best")
        ax.grid(True, linestyle=":", alpha=0.6)
        return ax

    def to_markdown(self, **kwargs) -> str:
        """Render policy rules table as Markdown."""
        from puremacro.reports import _df_to_markdown

        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        """Render policy rules table as LaTeX tabular."""
        from puremacro.reports import _df_to_latex

        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        """Render policy rules table as Typst table."""
        from puremacro.reports import _df_to_typst

        return _df_to_typst(self.to_frame(), **kwargs)



