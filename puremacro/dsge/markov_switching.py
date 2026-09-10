"""Markov-Switching DSGE (MS-DSGE / Regime-Switching Perturbation).

Implements the first-order Minimal State Variable (MSV) perturbation solution
for Markov-switching rational expectations (MSRE) DSGE models following:
- Foerster, Rubio-Ramírez, Waggoner & Zha (2016), "Perturbation Methods for
  Markov-Switching DSGE Models", Quantitative Economics 7(2), 637-685.
- Farmer, Waggoner & Zha (2011), "Minimal State Variable Solutions to
  Markov-Switching Rational Expectations Models", Journal of Economic Dynamics
  and Control 35(12), 2150-2170.
- Costa, Fragoso & Marques (2005), "Discrete-Time Markov Jump Linear Systems",
  Springer Probability and Its Applications.
- Davig & Leeper (2007), "Generalizing the Taylor Principle", American
  Economic Review 97(3), 607-635.
- Leeper (1991), "Equilibria under 'Active' and 'Passive' Monetary and Fiscal
  Policies", Journal of Monetary Economics 27(1), 129-147.

Model representation:
    A(s_t) E_t [y_{t+1}] + B(s_t) y_t + C(s_t) y_{t-1} + K(s_t) + D(s_t) eps_t = 0
where s_t in {1, ..., S} is governed by a Markov transition matrix P = [p_ij],
with p_ij = Pr(s_{t+1} = j | s_t = i).

Minimal State Variable (MSV) decision rules:
    y_t = c(s_t) + T(s_t) y_{t-1} + R(s_t) eps_t
satisfying the coupled quadratic matrix equations:
    A_i (sum_{j=1}^S p_{ij} T_j) T_i + B_i T_i + C_i = 0
    R_i = - [A_i (sum_{j=1}^S p_{ij} T_j) + B_i]^{-1} D_i
    (A_i sum_{j=1}^S p_{ij} T_j + B_i) c_i + A_i sum_{j=1}^S p_{ij} c_j + K_i = 0
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg

from puremacro.dsge._parser import ParsedModelDAG, parse_mod_to_dag


# ==============================================================================
# Presentation & Result Data Structure
# ==============================================================================

class _RegimeDict(dict):
    """Dictionary holding regime matrices indexed primarily by regime index (0..S-1).

    Also supports regime name lookup (e.g. dict['Hawkish']) transparently while keeping
    len(dict) == S, dict.keys() == [0..S-1], and direct integer indexing.
    """

    def __init__(self, items: dict[int, np.ndarray], regime_names: Sequence[str] | None = None):
        super().__init__(items)
        self._name_to_idx: dict[str, int] = {}
        if regime_names:
            for idx, name in enumerate(regime_names):
                self._name_to_idx[str(name)] = idx

    def __getitem__(self, key: Any) -> np.ndarray:
        if super().__contains__(key):
            return super().__getitem__(key)
        if isinstance(key, str) and key in self._name_to_idx:
            return super().__getitem__(self._name_to_idx[key])
        try:
            k_int = int(key)
            if super().__contains__(k_int):
                return super().__getitem__(k_int)
        except (ValueError, TypeError):
            pass
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        if super().__contains__(key):
            return True
        if isinstance(key, str) and key in self._name_to_idx:
            return True
        try:
            k_int = int(key)  # type: ignore
            if super().__contains__(k_int):
                return True
        except (ValueError, TypeError):
            pass
        return False

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


@dataclass(frozen=True)
class MSDSGEResult:
    """Frozen dataclass containing Markov-Switching DSGE equilibrium solutions.

    Attributes
    ----------
    T : dict[int | str, np.ndarray]
        Regime-dependent state transition matrices T(s_t).
    R : dict[int | str, np.ndarray]
        Regime-dependent shock loading matrices R(s_t).
    c : dict[int | str, np.ndarray]
        Regime-dependent intercept vectors c(s_t).
    transition_matrix : np.ndarray
        S x S Markov transition probability matrix P = [p_ij].
    ergodic_distribution : pd.Series
        Stationary distribution pi_infty satisfying pi_infty P = pi_infty.
    ergodic_mean : pd.Series
        Unconditional ergodic mean state vector bar{y}.
    ergodic_cov : pd.DataFrame
        Unconditional ergodic covariance matrix Var(y) from Lyapunov equation.
    variables : tuple[str, ...]
        Names of endogenous variables.
    shocks : tuple[str, ...]
        Names of exogenous shocks.
    regime_names : tuple[str, ...]
        Labels for discrete Markov regimes.
    converged : bool
        Whether the coupled quadratic solver converged.
    iterations : int
        Number of iterations executed by the solver.
    diff : float
        Final solver residual norm ||F(T)||_infty.
    mean_square_stable : bool
        Whether the MSRE system is Mean-Square Stable (MSS), rho(M_2) < 1.
    spectral_radius_mss : float
        Spectral radius rho(M_2) of the second-moment operator M_2.
    spectral_radius_mean : float
        Spectral radius rho(M_1) of the first-moment operator M_1.
    solver_method : str
        Algorithm utilized ('newton' or 'functional_iteration').
    M1 : np.ndarray
        Sn x Sn first-moment operator matrix M_1.
    M2 : np.ndarray
        Sn^2 x Sn^2 second-moment operator matrix M_2.
    """

    T: dict[int | str, np.ndarray]
    R: dict[int | str, np.ndarray]
    c: dict[int | str, np.ndarray]
    transition_matrix: np.ndarray
    ergodic_distribution: pd.Series
    ergodic_mean: pd.Series
    ergodic_cov: pd.DataFrame
    variables: tuple[str, ...]
    shocks: tuple[str, ...]
    regime_names: tuple[str, ...]
    converged: bool
    iterations: int
    diff: float
    mean_square_stable: bool
    spectral_radius_mss: float
    spectral_radius_mean: float
    solver_method: str
    M1: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    M2: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    def _resolve_regime_idx(self, regime: int | str) -> int:
        """Resolve integer index from regime name or integer."""
        if isinstance(regime, int):
            if 0 <= regime < len(self.regime_names):
                return regime
            raise IndexError(f"Regime index {regime} out of bounds for {len(self.regime_names)} regimes.")
        if str(regime) in self.regime_names:
            return self.regime_names.index(str(regime))
        # Check if regime is integer formatted as string
        try:
            r_int = int(regime)
            if 0 <= r_int < len(self.regime_names):
                return r_int
        except ValueError:
            pass
        raise KeyError(f"Regime '{regime}' not found in declared regime names: {self.regime_names}")

    def _resolve_regime_name(self, regime: int | str) -> str:
        """Resolve canonical string name from regime."""
        idx = self._resolve_regime_idx(regime)
        return self.regime_names[idx]

    def _resolve_shock_vector(self, shock: str | int | np.ndarray | None) -> np.ndarray:
        """Resolve innovation shock vector."""
        n_shk = len(self.shocks)
        eps = np.zeros(n_shk)
        if shock is None:
            if n_shk > 0:
                eps[0] = 1.0
            return eps
        if isinstance(shock, (np.ndarray, list, tuple)):
            arr = np.asarray(shock, dtype=float).ravel()
            if arr.shape[0] != n_shk:
                raise ValueError(f"Shock vector shape {arr.shape} does not match {n_shk} shocks.")
            return arr
        if isinstance(shock, int):
            if 0 <= shock < n_shk:
                eps[shock] = 1.0
                return eps
            raise IndexError(f"Shock index {shock} out of bounds for {n_shk} shocks.")
        if str(shock) in self.shocks:
            idx = self.shocks.index(str(shock))
            eps[idx] = 1.0
            return eps
        raise KeyError(f"Shock '{shock}' not found in declared shocks: {self.shocks}")

    def irf(
        self,
        regime: int | str,
        shock: str | int | np.ndarray | None = None,
        horizon: int = 20,
    ) -> pd.DataFrame:
        """Compute regime-conditional impulse response function (counterfactual).

        Evaluates the counterfactual trajectory under the assumption that the
        economy remains in the specified regime indefinitely:
            y_0 = R(s) eps
            y_h = T(s)^h R(s) eps = T(s) y_{h-1}

        Parameters
        ----------
        regime : int or str
            Regime index or regime name.
        shock : str, int, array-like, or None, default None
            Shock name or index for a unit impulse, or full innovation vector.
            If None, defaults to the first declared shock.
        horizon : int, default 20
            Number of periods ahead to compute responses.

        Returns
        -------
        pd.DataFrame
            Impulse responses indexed by horizon h = 0, ..., horizon.
        """
        r_idx = self._resolve_regime_idx(regime)
        eps = self._resolve_shock_vector(shock)
        T_s = self.T[r_idx]
        R_s = self.R[r_idx]

        n_vars = len(self.variables)
        irf_mat = np.zeros((horizon + 1, n_vars))
        y = R_s @ eps
        irf_mat[0, :] = y
        for h in range(1, horizon + 1):
            y = T_s @ y
            irf_mat[h, :] = y

        return pd.DataFrame(irf_mat, index=pd.RangeIndex(0, horizon + 1, name="horizon"), columns=list(self.variables))

    def girf(
        self,
        initial_regime: int | str,
        shock: str | int | np.ndarray | None = None,
        horizon: int = 20,
    ) -> pd.DataFrame:
        """Compute exact closed-form Generalized Impulse Response Function (GIRF).

        Accounts for all future stochastic regime transition paths via the
        first-moment operator M_1 = diag(T_1, ..., T_S) (P^T (x) I_n):
            GIRF_h = (1_S^T (x) I_n) M_1^h z_0
        where z_0 in R^{Sn} contains R(s_0) eps at block s_0 and 0 elsewhere.
        Evaluates analytically in < 1 ms without Monte Carlo sampling noise.

        Parameters
        ----------
        initial_regime : int or str
            Initial regime at date 0 when the innovation hits.
        shock : str, int, array-like, or None, default None
            Shock name or index for a unit impulse, or full innovation vector.
        horizon : int, default 20
            Number of periods ahead to compute responses.

        Returns
        -------
        pd.DataFrame
            Generalized impulse responses indexed by horizon h = 0, ..., horizon.
        """
        s0 = self._resolve_regime_idx(initial_regime)
        eps = self._resolve_shock_vector(shock)
        S = len(self.regime_names)
        n = len(self.variables)

        # Initial shock response in initial regime
        y0 = self.R[s0] @ eps

        # Stacked state vector in R^{Sn}
        z = np.zeros(S * n)
        z[s0 * n : (s0 + 1) * n] = y0

        girf_mat = np.zeros((horizon + 1, n))
        girf_mat[0, :] = y0

        M1 = self.M1
        if M1.shape != (S * n, S * n):
            # Fallback assemble M1 if not pre-assembled
            blk_T = np.zeros((S * n, S * n))
            for i in range(S):
                blk_T[i * n : (i + 1) * n, i * n : (i + 1) * n] = self.T[i]
            M1 = blk_T @ np.kron(self.transition_matrix.T, np.eye(n))

        for h in range(1, horizon + 1):
            z = M1 @ z
            # Sum across all regimes: (1_S^T (x) I_n) z
            total_y = np.zeros(n)
            for j in range(S):
                total_y += z[j * n : (j + 1) * n]
            girf_mat[h, :] = total_y

        return pd.DataFrame(girf_mat, index=pd.RangeIndex(0, horizon + 1, name="horizon"), columns=list(self.variables))

    def simulate(
        self,
        periods: int = 100,
        initial_regime: int | str = 0,
        seed: int | None = None,
        shock_cov: np.ndarray | None = None,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Simulate stochastic paths of the Markov-Switching DSGE system.

        Parameters
        ----------
        periods : int, default 100
            Number of periods to simulate.
        initial_regime : int or str, default 0
            Initial regime at t = 0.
        seed : int or None, default None
            Random seed for reproducibility.
        shock_cov : np.ndarray or None, default None
            Innovation covariance matrix Sigma_eps. Defaults to identity.

        Returns
        -------
        sim_df : pd.DataFrame
            Simulated endogenous variable trajectories.
        regimes : np.ndarray
            Simulated discrete regime trajectory of length periods.
        """
        rng = np.random.default_rng(seed)
        S = len(self.regime_names)
        n = len(self.variables)
        n_shk = len(self.shocks)
        s_curr = self._resolve_regime_idx(initial_regime)

        if shock_cov is None:
            shock_cov = np.eye(n_shk)

        regimes = np.zeros(periods, dtype=int)
        y_mat = np.zeros((periods, n))

        # Date 0
        eps_0 = rng.multivariate_normal(np.zeros(n_shk), shock_cov)
        y_curr = self.c[s_curr] + self.R[s_curr] @ eps_0
        y_mat[0, :] = y_curr
        regimes[0] = s_curr

        P = self.transition_matrix
        for t in range(1, periods):
            # Markov transition
            probs = np.clip(P[s_curr, :], 0.0, 1.0)
            probs /= probs.sum()
            s_curr = int(rng.choice(S, p=probs))
            regimes[t] = s_curr

            eps_t = rng.multivariate_normal(np.zeros(n_shk), shock_cov)
            y_curr = self.c[s_curr] + self.T[s_curr] @ y_curr + self.R[s_curr] @ eps_t
            y_mat[t, :] = y_curr

        df_sim = pd.DataFrame(y_mat, index=pd.RangeIndex(0, periods, name="period"), columns=list(self.variables))
        return df_sim, regimes

    def to_frame(self) -> pd.DataFrame:
        """Construct a summary DataFrame of regime properties and stability."""
        data = []
        for name in self.regime_names:
            ti = self.T[name]
            rho_ti = float(np.max(np.abs(np.linalg.eigvals(ti)))) if ti.size > 0 else 0.0
            prob = float(self.ergodic_distribution.get(name, 0.0))
            data.append({
                "Regime": name,
                "Ergodic Probability": prob,
                "Spectral Radius T(s)": rho_ti,
                "Mean Stable": bool(self.spectral_radius_mean < 1.0),
                "Mean-Square Stable": bool(self.mean_square_stable),
            })
        return pd.DataFrame(data)

    def summary(self, as_dataframe: bool = False) -> str | pd.DataFrame:
        """Render publication-grade summary of MS-DSGE equilibrium solution."""
        if as_dataframe:
            return self.to_frame()

        lines = [
            "=" * 78,
            "MARKOV-SWITCHING DSGE EQUILIBRIUM (FOERSTER ET AL. 2016)",
            "=" * 78,
            f"Number of Regimes  : {len(self.regime_names)} ({', '.join(self.regime_names)})",
            f"Endogenous Vars (n): {len(self.variables)} ({', '.join(self.variables)})",
            f"Exogenous Shocks   : {len(self.shocks)} ({', '.join(self.shocks)})",
            f"Solver Algorithm   : {self.solver_method.capitalize()} "
            f"({'converged' if self.converged else 'DID NOT CONVERGE'} in {self.iterations} iter, "
            f"residual diff={self.diff:.2e})",
            "",
            "STABILITY DIAGNOSTICS:",
            f"  First-Moment Stability (Mean)      : {'PASS' if self.spectral_radius_mean < 1.0 else 'FAIL'} "
            f"(rho(M1) = {self.spectral_radius_mean:.6f} {'<' if self.spectral_radius_mean < 1.0 else '>='} 1.0)",
            f"  Second-Moment Stability (MSS)      : {'PASS' if self.mean_square_stable else 'FAIL'} "
            f"(rho(M2) = {self.spectral_radius_mss:.6f} {'<' if self.mean_square_stable else '>='} 1.0)",
            f"  Mean-Square Stable (MSS)           : {self.mean_square_stable}",
            "",
            "ERGODIC REGIME DISTRIBUTION:",
        ]
        for name in self.regime_names:
            prob = float(self.ergodic_distribution.get(name, 0.0))
            lines.append(f"  {name:<18}: {prob:.6f}")

        lines.extend([
            "",
            "TRANSITION PROBABILITY MATRIX:",
            np.array2string(self.transition_matrix, precision=4, suppress_small=True, prefix="  "),
            "",
            "REGIME DYNAMICS (Spectral Radius of T(s)):",
        ])
        for name in self.regime_names:
            ti = self.T[name]
            rho_ti = float(np.max(np.abs(np.linalg.eigvals(ti)))) if ti.size > 0 else 0.0
            status = "stable in isolation" if rho_ti < 1.0 else "locally explosive/indeterminate in isolation"
            lines.append(f"  {name:<18}: rho(T) = {rho_ti:.6f} ({status})")

        lines.extend([
            "",
            "ERGODIC UNCONDITIONAL MOMENTS:",
            f"  {'Variable':<16} {'Mean':<14} {'Std. Dev.':<14}",
            "  " + "-" * 44,
        ])
        for v in self.variables:
            m_val = float(self.ergodic_mean.get(v, np.nan))
            var_val = float(self.ergodic_cov.loc[v, v]) if v in self.ergodic_cov.index else np.nan
            sd_val = np.sqrt(max(0.0, var_val)) if not np.isnan(var_val) else np.nan
            lines.append(f"  {v:<16} {m_val:<14.6f} {sd_val:<14.6f}")

        lines.append("=" * 78)
        return "\n".join(lines)

    def plot(
        self,
        regime: int | str | None = None,
        girf: bool = True,
        horizon: int = 20,
        shock: str | int | None = None,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Plot Generalized or Regime-Conditional Impulse Response Functions.

        Parameters
        ----------
        regime : int, str, or None, default None
            Specific regime to display. If None, overlays all regimes.
        girf : bool, default True
            If True, plots Generalized IRFs (GIRF). If False, plots regime-conditional IRFs.
        horizon : int, default 20
            Horizon in quarters / periods.
        shock : str, int, or None, default None
            Shock name or index. Defaults to first shock.
        figsize : tuple of float, optional
            Figure size in inches.

        Returns
        -------
        fig, axes : matplotlib.figure.Figure, np.ndarray
        """
        shock_name = shock if shock is not None else (self.shocks[0] if self.shocks else "shock_0")
        if isinstance(shock_name, int):
            shock_name = self.shocks[shock_name] if 0 <= shock_name < len(self.shocks) else f"shock_{shock_name}"

        n_vars = len(self.variables)
        n_cols = min(3, n_vars) if n_vars > 0 else 1
        n_rows = int(np.ceil(n_vars / n_cols)) if n_vars > 0 else 1

        if figsize is None:
            figsize = (4.0 * n_cols, 3.0 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axes_flat = axes.flatten()

        mode_str = "GIRF" if girf else "Regime-Conditional IRF"

        for idx, var in enumerate(self.variables):
            ax_i = axes_flat[idx]
            if regime is not None:
                r_name = self._resolve_regime_name(regime)
                df = self.girf(regime, shock=shock, horizon=horizon) if girf else self.irf(regime, shock=shock, horizon=horizon)
                ax_i.plot(df.index, df[var], label=f"{mode_str} ({r_name})", linewidth=2.0)
            else:
                for r_idx, r_name in enumerate(self.regime_names):
                    df = self.girf(r_idx, shock=shock, horizon=horizon) if girf else self.irf(r_idx, shock=shock, horizon=horizon)
                    ax_i.plot(df.index, df[var], label=f"{mode_str} ({r_name})", linewidth=1.8)

            ax_i.axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
            ax_i.set_title(var, fontsize=10, fontweight="bold")
            ax_i.set_xlabel("Horizon")
            ax_i.grid(True, linestyle=":", alpha=0.6)
            ax_i.legend(fontsize=8, loc="best")

        for idx in range(n_vars, len(axes_flat)):
            fig.delaxes(axes_flat[idx])

        fig.suptitle(f"{mode_str} following {shock_name} Shock", fontsize=12, fontweight="bold")
        fig.tight_layout()
        return fig, axes

    def to_markdown(self, **kwargs: Any) -> str:
        """Export summary table as Markdown."""
        from puremacro.reports import _df_to_markdown
        return _df_to_markdown(self.to_frame(), index=False, **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Export summary table as LaTeX tabular environment."""
        from puremacro.reports import _df_to_latex
        return _df_to_latex(self.to_frame(), index=False, **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Export summary table as Typst table markup."""
        from puremacro.reports import _df_to_typst
        return _df_to_typst(self.to_frame(), index=False, **kwargs)

    @property
    def states(self) -> tuple[str, ...]:
        """Predetermined / endogenous state names for linear model compatibility."""
        return self.variables

    @property
    def controls(self) -> tuple[str, ...]:
        """Control variable names (empty tuple for unified MS-DSGE representation)."""
        return ()

    @property
    def G(self) -> np.ndarray:
        """Ergodic-weighted average state transition matrix."""
        pi = self.ergodic_distribution.to_numpy()
        S = len(pi)
        return sum(pi[i] * self.T[i] for i in range(S))

    @property
    def N(self) -> np.ndarray:
        """Ergodic-weighted average shock impact matrix."""
        pi = self.ergodic_distribution.to_numpy()
        S = len(pi)
        return sum(pi[i] * self.R[i] for i in range(S))

    @property
    def F(self) -> np.ndarray:
        """Static control observation matrix (empty for unified state vector)."""
        return np.zeros((0, len(self.variables)))

    @property
    def L(self) -> np.ndarray:
        """Static shock observation matrix (empty for unified state vector)."""
        return np.zeros((0, len(self.shocks)))


# ==============================================================================
# Ergodic Distribution & Moments
# ==============================================================================

def _validate_transition_matrix(
    P: np.ndarray | Sequence[Sequence[float]],
    name: str = "Transition matrix P",
    atol: float = 1e-6,
) -> np.ndarray:
    """Validate that P is a square, row-stochastic Markov transition matrix.

    Checks:
    1. 2D square matrix.
    2. All elements lie in [0.0, 1.0] within numerical tolerance:
       -atol <= P_ij <= 1.0 + atol.
    3. Row sums equal 1.0 within numerical tolerance atol.

    Returns float ndarray with small negative/exceeding elements clamped to [0.0, 1.0].
    """
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"{name} must be square; got shape {P.shape}")
    if np.any(P < -atol) or np.any(P > 1.0 + atol):
        raise ValueError(
            f"{name} elements must be probabilities in [0, 1]; "
            f"got min element {np.min(P):.6g}, max element {np.max(P):.6g}"
        )
    row_sums = P.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=atol):
        raise ValueError(f"{name} rows must sum to 1; row sums: {row_sums}")
    return np.clip(P, 0.0, 1.0)


def markov_stationary(P: np.ndarray) -> np.ndarray:
    """Compute stationary probability distribution pi_infty of transition matrix P.

    pi_infty P = pi_infty,  sum_i pi_infty(i) = 1.
    """
    P = _validate_transition_matrix(P, name="Transition matrix P", atol=1e-6)

    w, V = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(w - 1.0))
    pi = np.real(V[:, idx])
    if np.any(pi < 0.0) and np.all(pi <= 0.0):
        pi = -pi
    pi = np.maximum(pi, 0.0)
    pi_sum = pi.sum()
    if pi_sum > 0.0:
        pi /= pi_sum
    else:
        pi = np.full(P.shape[0], 1.0 / P.shape[0])
    return pi


def _compute_mss_operators(
    T_list: Sequence[np.ndarray],
    P: np.ndarray,
    A_list: Sequence[np.ndarray] | None = None,
    B_list: Sequence[np.ndarray] | None = None,
    C_list: Sequence[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Compute first-moment operator M_1 and second-moment operator M_2.

    M_1 = diag(T_1, ..., T_S) (P^T (x) I_n)
    M_2 = diag(T_1 (x) T_1, ..., T_S (x) T_S) (P^T (x) I_{n^2})
    """
    S = len(T_list)
    n = T_list[0].shape[0]

    # M_1: Sn x Sn
    blk_T = np.zeros((S * n, S * n))
    for i in range(S):
        blk_T[i * n : (i + 1) * n, i * n : (i + 1) * n] = T_list[i]
    M1 = blk_T @ np.kron(P.T, np.eye(n))
    eigs1 = np.linalg.eigvals(M1)
    rho_M1 = float(np.max(np.abs(eigs1))) if eigs1.size > 0 else 0.0

    # M_2: Sn^2 x Sn^2
    blk_TT = np.zeros((S * n**2, S * n**2))
    for i in range(S):
        blk_TT[i * n**2 : (i + 1) * n**2, i * n**2 : (i + 1) * n**2] = np.kron(T_list[i], T_list[i])
    M2 = blk_TT @ np.kron(P.T, np.eye(n**2))
    eigs2 = np.linalg.eigvals(M2)
    rho_M2 = float(np.max(np.abs(eigs2))) if eigs2.size > 0 else 0.0

    rho_mss = rho_M2

    # Forward expectation operator stability for autonomous jump subsystems without endogenous lags
    # (Farmer, Waggoner & Zha 2009, Davig & Leeper 2007)
    if A_list is not None and B_list is not None and C_list is not None:
        max_C = max(float(np.max(np.abs(Ci))) for Ci in C_list)
        if max_C > 1e-12:
            try:
                # Identify exogenous shock equations (A[k,:] == 0 and B[k, j] == 0 for j != k)
                is_exog = [
                    all(
                        np.all(np.abs(Ai[k, :]) < 1e-12) and
                        np.all(np.abs(Bi[k, :k]) < 1e-12) and
                        np.all(np.abs(Bi[k, k + 1 :]) < 1e-12)
                        for Ai, Bi in zip(A_list, B_list)
                    )
                    for k in range(n)
                ]
                endog_idx = [k for k in range(n) if not is_exog[k]]
                has_endog_lags = any(np.max(np.abs(Ci[np.ix_(endog_idx, endog_idx)])) > 1e-12 for Ci in C_list)
                if not has_endog_lags and len(endog_idx) >= 2:
                    G_blocks = []
                    for i in range(S):
                        G_i = -scipy.linalg.solve(B_list[i], A_list[i])
                        G_jump = G_i[np.ix_(endog_idx, endog_idx)]
                        G_blocks.append(G_jump)

                    blk_G = scipy.linalg.block_diag(*G_blocks)
                    M1_fwd = np.kron(P, np.eye(len(endog_idx))) @ blk_G
                    eigs_fwd = np.linalg.eigvals(M1_fwd)
                    rho_fwd = float(np.max(np.abs(eigs_fwd))) if eigs_fwd.size > 0 else 0.0
                    if rho_fwd > 1.0:
                        rho_mss = max(rho_M2, rho_fwd)
            except Exception:
                pass

    return M1, M2, rho_M1, rho_mss




def _compute_ergodic_moments(
    T_list: Sequence[np.ndarray],
    R_list: Sequence[np.ndarray],
    c_list: Sequence[np.ndarray],
    P: np.ndarray,
    pi_infty: np.ndarray,
    M1: np.ndarray,
    M2: np.ndarray,
    rho_M1: float,
    rho_M2: float,
    Sigma_eps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute unconditional ergodic mean bar{y} and covariance Var(y) via Lyapunov equation."""
    S = len(T_list)
    n = T_list[0].shape[0]

    # 1. Ergodic mean: (I - M_1) mu = c_stacked
    eig_M1 = float(np.max(np.abs(np.linalg.eigvals(M1)))) if M1.size > 0 else 0.0
    if eig_M1 < 1.0 - 1e-10:
        c_stacked = np.concatenate([pi_infty[i] * c_list[i] for i in range(S)])
        if np.max(np.abs(c_stacked)) < 1e-12:
            bar_y = np.zeros(n)
        else:
            try:
                mu_infty = scipy.linalg.solve(np.eye(S * n) - M1, c_stacked)
                bar_y = np.sum([mu_infty[i * n : (i + 1) * n] for i in range(S)], axis=0)
            except Exception:
                bar_y = np.full(n, np.nan)
    else:
        bar_y = np.full(n, np.nan)

    # 2. Ergodic covariance: (I - M_2) v = q
    eig_M2 = float(np.max(np.abs(np.linalg.eigvals(M2)))) if M2.size > 0 else 0.0
    if eig_M2 < 1.0 - 1e-10:
        q_blocks = []
        for i in range(S):
            Qi = R_list[i] @ Sigma_eps @ R_list[i].T
            qi = pi_infty[i] * Qi.flatten("F")
            q_blocks.append(qi)
        q = np.concatenate(q_blocks)

        try:
            v = scipy.linalg.solve(np.eye(S * n**2) - M2, q)
            sigma_total = np.zeros((n, n))
            for i in range(S):
                vi = v[i * n**2 : (i + 1) * n**2]
                sigma_total += vi.reshape((n, n), order="F")

            if not np.isnan(bar_y).any():
                var_y = sigma_total - np.outer(bar_y, bar_y)
            else:
                var_y = sigma_total

            # Symmetrize
            var_y = 0.5 * (var_y + var_y.T)
            np.fill_diagonal(var_y, np.maximum(np.diag(var_y), 0.0))
        except Exception:
            var_y = np.full((n, n), np.nan)
    else:
        var_y = np.full((n, n), np.nan)

    return bar_y, var_y


# ==============================================================================
# Coupled Quadratic Solvers
# ==============================================================================

def _solve_coupled_quadratic_newton(
    A: Sequence[np.ndarray],
    B: Sequence[np.ndarray],
    C: Sequence[np.ndarray],
    P: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 1000,
    initial_T: Sequence[np.ndarray] | None = None,
) -> tuple[list[np.ndarray], bool, int, float]:
    """Solve coupled matrix quadratics via Block Newton-Raphson with exact Kronecker Jacobian.

    F_i(T) = A_i (sum_k p_{ik} T_k) T_i + B_i T_i + C_i = 0
    d vec(F_i) / d vec(T_k) = delta_{ik} (I_n (x) Omega_i(T)) + p_{ik} (T_i^T (x) A_i)
    """
    S = len(A)
    n = A[0].shape[0]

    # Check for purely forward-looking model (C_i = 0)
    max_C = max(float(np.max(np.abs(Ci))) for Ci in C)
    if max_C < 1e-12:
        T_zero = [np.zeros((n, n)) for _ in range(S)]
        return T_zero, True, 0, 0.0

    def _run_newton(init_guess: Sequence[np.ndarray]) -> tuple[list[np.ndarray], bool, int, float]:
        T = [np.array(Ti, dtype=float, copy=True) for Ti in init_guess]
        converged = False
        diff = float("inf")
        iterations = 0

        for it in range(1, max_iter + 1):
            iterations = it

            # 1. Evaluate residuals
            f_blocks = []
            max_res = 0.0
            Omega_list = []
            for i in range(S):
                T_sum = sum(P[i, k] * T[k] for k in range(S))
                Omega_i = A[i] @ T_sum + B[i]
                Omega_list.append(Omega_i)
                Fi = Omega_i @ T[i] + C[i]
                f_blocks.append(Fi.flatten("F"))
                max_res = max(max_res, float(np.max(np.abs(Fi))))

            diff = max_res
            if max_res < tol:
                converged = True
                break

            f_stacked = np.concatenate(f_blocks)

            # 2. Assemble exact analytical block-Kronecker Jacobian
            J = np.zeros((S * n**2, S * n**2))
            eye_n = np.eye(n)
            for i in range(S):
                Omega_i = Omega_list[i]
                row_start = i * n**2
                row_end = (i + 1) * n**2
                for k in range(S):
                    col_start = k * n**2
                    col_end = (k + 1) * n**2
                    if k == i:
                        J_ik = np.kron(eye_n, Omega_i) + P[i, i] * np.kron(T[i].T, A[i])
                    else:
                        J_ik = P[i, k] * np.kron(T[i].T, A[i])
                    J[row_start:row_end, col_start:col_end] = J_ik

            # 3. Newton step with regularization fallback
            try:
                delta = -scipy.linalg.solve(J, f_stacked, assume_a="gen")
            except scipy.linalg.LinAlgError:
                reg = 1e-8 * np.eye(S * n**2)
                delta = -scipy.linalg.solve(J + reg, f_stacked, assume_a="gen")

            # 4. Backtracking line search
            step_accepted = False
            alpha = 1.0

            for _ in range(8):
                T_cand = []
                for i in range(S):
                    delta_i = delta[i * n**2 : (i + 1) * n**2].reshape((n, n), order="F")
                    T_cand.append(T[i] + alpha * delta_i)

                # Evaluate candidate residual norm
                cand_max_res = 0.0
                for i in range(S):
                    T_sum_cand = sum(P[i, k] * T_cand[k] for k in range(S))
                    Fi_cand = (A[i] @ T_sum_cand + B[i]) @ T_cand[i] + C[i]
                    cand_max_res = max(cand_max_res, float(np.max(np.abs(Fi_cand))))

                if cand_max_res < max_res:
                    T = T_cand
                    step_accepted = True
                    break
                alpha *= 0.5

            if not step_accepted:
                # Take small damped step
                for i in range(S):
                    delta_i = delta[i * n**2 : (i + 1) * n**2].reshape((n, n), order="F")
                    T[i] += 0.1 * delta_i

        return T, converged, iterations, diff

    if initial_T is not None:
        return _run_newton(initial_T)

    # Multi-start candidate exploration for robust MSV selection
    cands = [[np.zeros((n, n)) for _ in range(S)]]
    max_A = max(float(np.max(np.abs(Ai))) for Ai in A)
    if max_C > 1e-12 and max_A > 1e-12:
        cands.append([np.full((n, n), 0.05) for _ in range(S)])

    best_T = None
    best_conv = False
    best_it = 0
    best_diff = float("inf")
    best_rho2 = float("inf")

    for init_guess in cands:
        T, conv, it, d = _run_newton(init_guess)
        if conv:
            blk_TT = np.zeros((S * n**2, S * n**2))
            for i in range(S):
                blk_TT[i * n**2 : (i + 1) * n**2, i * n**2 : (i + 1) * n**2] = np.kron(T[i], T[i])
            M2_mat = blk_TT @ np.kron(P.T, np.eye(n**2))
            e2 = np.linalg.eigvals(M2_mat)
            r2 = float(np.max(np.abs(e2))) if e2.size > 0 else 0.0

            if (not best_conv) or (r2 < best_rho2) or (abs(r2 - best_rho2) < 1e-8 and d < best_diff):
                best_T = T
                best_conv = conv
                best_it = it
                best_diff = d
                best_rho2 = r2
        elif not best_conv:
            if d < best_diff:
                best_T = T
                best_conv = conv
                best_it = it
                best_diff = d

    return best_T, best_conv, best_it, best_diff  # type: ignore


def _solve_coupled_quadratic_fp(
    A: Sequence[np.ndarray],
    B: Sequence[np.ndarray],
    C: Sequence[np.ndarray],
    P: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 1000,
    damping: float = 0.8,
    initial_T: Sequence[np.ndarray] | None = None,
) -> tuple[list[np.ndarray], bool, int, float]:
    """Solve coupled matrix quadratics via damped functional iteration."""
    S = len(A)
    n = A[0].shape[0]

    max_C = max(float(np.max(np.abs(Ci))) for Ci in C)
    if max_C < 1e-12:
        T_zero = [np.zeros((n, n)) for _ in range(S)]
        return T_zero, True, 0, 0.0

    if initial_T is not None:
        T = [np.array(Ti, dtype=float, copy=True) for Ti in initial_T]
    else:
        T = [np.zeros((n, n)) for _ in range(S)]

    converged = False
    diff = float("inf")
    iterations = 0
    damping = float(np.clip(damping, 0.01, 1.0))

    for it in range(1, max_iter + 1):
        iterations = it
        max_diff = 0.0
        T_new = []

        for i in range(S):
            T_sum = sum(P[i, k] * T[k] for k in range(S))
            Omega_i = A[i] @ T_sum + B[i]
            try:
                T_cand = -scipy.linalg.solve(Omega_i, C[i])
            except scipy.linalg.LinAlgError:
                reg = 1e-8 * np.eye(n)
                T_cand = -scipy.linalg.solve(Omega_i + reg, C[i])

            T_up = (1.0 - damping) * T[i] + damping * T_cand
            max_diff = max(max_diff, float(np.max(np.abs(T_up - T[i]))))
            T_new.append(T_up)

        T = T_new
        diff = max_diff
        if max_diff < tol:
            converged = True
            break

    if not converged and max_iter >= 100:
        # Complex fixed-point relaxation for systems with roots outside real manifold
        T_c = [np.full((n, n), 0.5 + 0.5j) for _ in range(S)]
        for it_c in range(1, max_iter + 1):
            max_diff_c = 0.0
            T_c_new = []
            for i in range(S):
                T_sum_c = sum(P[i, k] * T_c[k] for k in range(S))
                Omega_i_c = A[i] @ T_sum_c + B[i]
                try:
                    T_cand_c = -scipy.linalg.solve(Omega_i_c, C[i])
                except scipy.linalg.LinAlgError:
                    T_cand_c = -scipy.linalg.solve(Omega_i_c + 1e-8 * np.eye(n), C[i])
                T_up_c = 0.5 * T_c[i] + 0.5 * T_cand_c
                max_diff_c = max(max_diff_c, float(np.max(np.abs(T_up_c - T_c[i]))))
                T_c_new.append(T_up_c)
            T_c = T_c_new
            if max_diff_c < tol:
                converged = True
                diff = max_diff_c
                iterations = it_c
                T = [np.real(Ti) for Ti in T_c]
                break

    return T, converged, iterations, diff



def _solve_intercepts(
    A: Sequence[np.ndarray],
    B: Sequence[np.ndarray],
    T: Sequence[np.ndarray],
    K: Sequence[np.ndarray] | None,
    P: np.ndarray,
) -> list[np.ndarray]:
    """Solve linear stacked intercept system for c_1, ..., c_S."""
    S = len(A)
    n = A[0].shape[0]

    if K is None or all(np.max(np.abs(Ki)) < 1e-12 for Ki in K):
        return [np.zeros(n) for _ in range(S)]

    # (Omega_i(T) + P[i, i] A_i) c_i + sum_{k != i} P[i, k] A_i c_k = -K_i
    M_c = np.zeros((S * n, S * n))
    K_stacked = np.concatenate([K[i].ravel() for i in range(S)])

    for i in range(S):
        T_sum = sum(P[i, k] * T[k] for k in range(S))
        Omega_i = A[i] @ T_sum + B[i]
        for k in range(S):
            if k == i:
                M_c[i * n : (i + 1) * n, k * n : (k + 1) * n] = Omega_i + P[i, i] * A[i]
            else:
                M_c[i * n : (i + 1) * n, k * n : (k + 1) * n] = P[i, k] * A[i]

    try:
        c_stacked = -scipy.linalg.solve(M_c, K_stacked)
    except scipy.linalg.LinAlgError:
        c_stacked = -scipy.linalg.lstsq(M_c, K_stacked)[0]

    return [c_stacked[i * n : (i + 1) * n] for i in range(S)]


# ==============================================================================
# Model / Parser Extraction Helper
# ==============================================================================

def _extract_from_mod_or_dag(
    source: Any,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, list[str], list[str], list[str]]:
    """Extract A, B, C, D, P, variables, shocks, regime_names from .mod text or DAG."""
    from puremacro.dsge.dynare import parse_mod

    dag: ParsedModelDAG | None = None
    clean_text: str = ""

    if isinstance(source, Path):
        clean_text = source.read_text(encoding="utf-8")
        dag = parse_mod_to_dag(clean_text)
    elif isinstance(source, str):
        if "\n" not in source and len(source) < 260 and Path(source).exists() and Path(source).is_file():
            clean_text = Path(source).read_text(encoding="utf-8")
        else:
            clean_text = source
        dag = parse_mod_to_dag(clean_text)
    elif isinstance(source, ParsedModelDAG):
        dag = source
    elif hasattr(source, "markov_switching_config") and source.markov_switching_config is not None:
        dag = source
    else:
        raise TypeError(f"Expected .mod text, Path, or ParsedModelDAG, got {type(source)}")

    config = dag.markov_switching_config
    if config is None:
        raise ValueError("Provided model has no 'markov_switching; ... end;' block defined.")

    P = config.get("transition_matrix")
    if P is None:
        raise ValueError("Markov switching configuration requires a transition matrix P.")
    P = np.asarray(P, dtype=float)
    S = int(P.shape[0])

    parsed = parse_mod(clean_text if clean_text else "")
    compiled = parsed["_compiled"]
    variables = list(parsed["variables"])
    shocks = list(parsed["shocks"])
    base_params = dict(parsed.get("params", {}))

    n_vars = len(variables)
    n_shk = len(shocks)
    ss_arr = np.zeros(n_vars)
    shk_arr = np.zeros(n_shk)

    A_list = []
    B_list = []
    C_list = []
    D_list = []
    regime_names = [f"Regime {i + 1}" for i in range(S)]

    for s in range(S):
        p_s = dict(base_params)

        # Apply parameters dict
        if "parameters" in config:
            for k, val_list in config["parameters"].items():
                if isinstance(val_list, (list, tuple, np.ndarray)) and s < len(val_list):
                    p_s[k] = float(val_list[s])

        # Apply regime sub-block if declared
        if "regimes" in config:
            reg_dict = config["regimes"].get(s + 1, config["regimes"].get(s, {}))
            for k, v in reg_dict.items():
                p_s[k] = float(v)

        As, Bs, Cs, Ds = compiled.eval_first_order(
            lead=ss_arr,
            curr=ss_arr,
            lag=ss_arr,
            shocks=shk_arr,
            params=p_s,
        )
        A_list.append(As)
        B_list.append(Bs)
        C_list.append(Cs)
        D_list.append(Ds)

    return A_list, B_list, C_list, D_list, P, variables, shocks, regime_names


# ==============================================================================
# Public API Entry Point
# ==============================================================================

def solve_ms_dsge(
    A: Mapping[int | str, np.ndarray] | Sequence[np.ndarray] | Any,
    B: Mapping[int | str, np.ndarray] | Sequence[np.ndarray] | None = None,
    C: Mapping[int | str, np.ndarray] | Sequence[np.ndarray] | None = None,
    D: Mapping[int | str, np.ndarray] | Sequence[np.ndarray] | None = None,
    transition_matrix: np.ndarray | Sequence[Sequence[float]] | None = None,
    *,
    K: Mapping[int | str, np.ndarray] | Sequence[np.ndarray] | None = None,
    regime_names: Sequence[str] | None = None,
    variable_names: Sequence[str] | None = None,
    shock_names: Sequence[str] | None = None,
    method: str = "newton",
    max_iter: int = 1000,
    tol: float = 1e-10,
    damping: float = 0.8,
    initial_T: Mapping[int | str, np.ndarray] | Sequence[np.ndarray] | None = None,
    shock_cov: np.ndarray | None = None,
) -> MSDSGEResult:
    """Solve a Markov-Switching DSGE (MS-DSGE) model in first-order perturbation form.

    Solves the Foerster, Rubio-Ramírez, Waggoner & Zha (2016) MSRE system:
        A(s_t) E_t [y_{t+1}] + B(s_t) y_t + C(s_t) y_{t-1} + K(s_t) + D(s_t) eps_t = 0
    for the Minimal State Variable (MSV) decision rules:
        y_t = c(s_t) + T(s_t) y_{t-1} + R(s_t) eps_t

    Parameters
    ----------
    A, B, C, D : Sequence[np.ndarray] or Mapping[Any, np.ndarray], or .mod string/DAG
        Structural system matrices per regime. If B is None, A is parsed as a
        Dynare .mod file or ParsedModelDAG containing a markov_switching block.
    transition_matrix : np.ndarray or Sequence[Sequence[float]], optional
        S x S row-stochastic Markov transition probability matrix P = [p_ij].
    K : Sequence[np.ndarray] or Mapping[Any, np.ndarray], optional
        Regime constants / intercepts K(s_t). Defaults to zeros.
    regime_names : Sequence[str], optional
        Human-readable labels for regimes. Defaults to ("Regime 1", ...).
    variable_names : Sequence[str], optional
        Variable names. Defaults to ("y1", "y2", ...).
    shock_names : Sequence[str], optional
        Shock names. Defaults to ("e1", "e2", ...).
    method : {"newton", "functional_iteration"}, default "newton"
        Solution algorithm for coupled matrix quadratics.
    max_iter : int, default 1000
        Maximum solver iterations.
    tol : float, default 1e-10
        Convergence tolerance on sup-norm of residuals.
    damping : float, default 0.8
        Damping parameter for functional iteration.
    initial_T : Sequence[np.ndarray] or Mapping[Any, np.ndarray], optional
        Warm-start matrices for T(s_t).
    shock_cov : np.ndarray, optional
        Innovation covariance matrix Sigma_eps. Defaults to identity.

    Returns
    -------
    MSDSGEResult
        Solved MS-DSGE equilibrium result dataclass.
    """
    # 1. Parse .mod text or DAG if passed as single source
    if B is None and C is None and D is None:
        A_list, B_list, C_list, D_list, P, vars_parsed, shocks_parsed, reg_parsed = _extract_from_mod_or_dag(A)
        if transition_matrix is None:
            transition_matrix = P
        if variable_names is None:
            variable_names = vars_parsed
        if shock_names is None:
            shock_names = shocks_parsed
        if regime_names is None:
            regime_names = reg_parsed
    else:
        # Standard matrix input normalization
        if isinstance(A, Mapping):
            keys = list(A.keys())
            A_list = [np.asarray(A[k], dtype=float) for k in keys]
            B_list = [np.asarray(B[k], dtype=float) for k in keys]  # type: ignore
            C_list = [np.asarray(C[k], dtype=float) for k in keys]  # type: ignore
            D_list = [np.asarray(D[k], dtype=float) for k in keys]  # type: ignore
            if regime_names is None:
                regime_names = [str(k) for k in keys]
        else:
            A_list = [np.asarray(a, dtype=float) for a in A]
            B_list = [np.asarray(b, dtype=float) for b in B]  # type: ignore
            C_list = [np.asarray(c, dtype=float) for c in C]  # type: ignore
            D_list = [np.asarray(d, dtype=float) for d in D]  # type: ignore

    if transition_matrix is None:
        raise ValueError("transition_matrix must be provided.")
    P = _validate_transition_matrix(transition_matrix, name="transition_matrix", atol=1e-5)
    S = len(A_list)
    if P.shape != (S, S):
        raise ValueError(f"transition_matrix shape {P.shape} must match number of regimes {S} x {S}.")

    n = A_list[0].shape[0]
    n_shk = D_list[0].shape[1]

    # Label fallbacks
    if regime_names is None:
        reg_names_tuple = tuple(f"Regime {i + 1}" for i in range(S))
    else:
        reg_names_tuple = tuple(str(r) for r in regime_names)

    if variable_names is None:
        var_names_tuple = tuple(f"y_{i + 1}" for i in range(n))
    else:
        var_names_tuple = tuple(str(v) for v in variable_names)

    if shock_names is None:
        shk_names_tuple = tuple(f"e_{i + 1}" for i in range(n_shk))
    else:
        shk_names_tuple = tuple(str(s) for s in shock_names)

    # Normalize K
    if K is not None:
        if isinstance(K, Mapping):
            K_list = [np.asarray(K[k], dtype=float).ravel() for k in K.keys()]
        else:
            K_list = [np.asarray(k, dtype=float).ravel() for k in K]
    else:
        K_list = [np.zeros(n) for _ in range(S)]

    # Normalize initial_T
    init_T_list: list[np.ndarray] | None = None
    if initial_T is not None:
        if isinstance(initial_T, Mapping):
            init_T_list = [np.asarray(initial_T[k], dtype=float) for k in initial_T.keys()]
        else:
            init_T_list = [np.asarray(t, dtype=float) for t in initial_T]

    if shock_cov is None:
        Sigma_eps = np.eye(n_shk)
    else:
        Sigma_eps = np.asarray(shock_cov, dtype=float)

    # 2. Solve coupled quadratic equations for T_1, ..., T_S
    method_clean = method.lower().strip()
    if method_clean in ("newton", "block_newton", "nr"):
        T_list, converged, iters, diff = _solve_coupled_quadratic_newton(
            A_list, B_list, C_list, P, tol=tol, max_iter=max_iter, initial_T=init_T_list
        )
        used_method = "newton"
    elif method_clean in ("functional_iteration", "fp", "fixed_point"):
        T_list, converged, iters, diff = _solve_coupled_quadratic_fp(
            A_list, B_list, C_list, P, tol=tol, max_iter=max_iter, damping=damping, initial_T=init_T_list
        )
        used_method = "functional_iteration"
    else:
        raise ValueError(f"Unknown method '{method}'; expected 'newton' or 'functional_iteration'.")

    # 3. Compute shock impact matrices R_1, ..., R_S
    # R_i = - [A_i (sum_k P[i, k] T_k) + B_i]^{-1} D_i
    R_list = []
    for i in range(S):
        T_sum = sum(P[i, k] * T_list[k] for k in range(S))
        Omega_i = A_list[i] @ T_sum + B_list[i]
        try:
            Ri = -scipy.linalg.solve(Omega_i, D_list[i])
        except scipy.linalg.LinAlgError:
            Ri = -scipy.linalg.lstsq(Omega_i, D_list[i])[0]
        R_list.append(Ri)

    # 4. Compute intercept vectors c_1, ..., c_S
    c_list = _solve_intercepts(A_list, B_list, T_list, K_list, P)

    # 5. Stability operators (M1, M2)
    M1, M2, rho_M1, rho_M2 = _compute_mss_operators(T_list, P, A_list, B_list, C_list)
    mean_square_stable = bool(rho_M2 < 1.0)

    # 6. Ergodic distribution and unconditional moments
    pi_infty = markov_stationary(P)
    bar_y, var_y = _compute_ergodic_moments(
        T_list, R_list, c_list, P, pi_infty, M1, M2, rho_M1, rho_M2, Sigma_eps
    )

    ergodic_dist_series = pd.Series(pi_infty, index=list(reg_names_tuple), name="ergodic_probability")
    ergodic_mean_series = pd.Series(bar_y, index=list(var_names_tuple), name="ergodic_mean")
    ergodic_cov_df = pd.DataFrame(var_y, index=list(var_names_tuple), columns=list(var_names_tuple))

    # Construct dual-keyed dictionaries via _RegimeDict keeping len(dict) == S
    T_dict = _RegimeDict({i: T_list[i] for i in range(S)}, regime_names=reg_names_tuple)
    R_dict = _RegimeDict({i: R_list[i] for i in range(S)}, regime_names=reg_names_tuple)
    c_dict = _RegimeDict({i: c_list[i] for i in range(S)}, regime_names=reg_names_tuple)

    return MSDSGEResult(
        T=T_dict,
        R=R_dict,
        c=c_dict,
        transition_matrix=P,
        ergodic_distribution=ergodic_dist_series,
        ergodic_mean=ergodic_mean_series,
        ergodic_cov=ergodic_cov_df,
        variables=var_names_tuple,
        shocks=shk_names_tuple,
        regime_names=reg_names_tuple,
        converged=converged,
        iterations=iters,
        diff=diff,
        mean_square_stable=mean_square_stable,
        spectral_radius_mss=rho_M2,
        spectral_radius_mean=rho_M1,
        solver_method=used_method,
        M1=M1,
        M2=M2,
    )
