"""Simulated Method of Moments (SMM) & Minimum-Distance Calibration for Fiscal Sovereign Default.

Bridges :func:`puremacro.sovereign.solve_fiscal_sovereign_default` with
:func:`puremacro.structural.bridge.fit_structural` and
:func:`puremacro.structural.bridge.ags_sensitivity` (Andrews, Gentzkow & Shapiro 2017).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst
from puremacro.sovereign._fiscal_paths import draw_uniforms, ergodic_moments, simulate_paths
from puremacro.sovereign._results import FiscalSovereignResult
from puremacro.sovereign.fiscal import solve_fiscal_sovereign_default
from puremacro.sovereign.rules import FiscalRule
from puremacro.structural.bridge import (
    AGSSensitivityResult,
    MomentTargets,
    StructuralFitResult,
    ags_sensitivity,
    fit_structural,
)

__all__ = [
    "SovereignSMMResult",
    "calibrate_fiscal_sovereign_smm",
]


@dataclass(frozen=True)
class SovereignSMMResult:
    """Result container for SMM / minimum-distance calibration of the fiscal sovereign default model.

    Parameters
    ----------
    fit : StructuralFitResult
        Minimum-distance estimation result from :func:`puremacro.structural.bridge.fit_structural`.
    sensitivity : AGSSensitivityResult
        Andrews, Gentzkow & Shapiro (2017) sensitivity matrix ``Lambda`` and standardized ``Lambda_tilde``.
    model : FiscalSovereignResult
        Solved :class:`FiscalSovereignResult` evaluated at the calibrated parameter vector ``fit.theta``.
    parameter_dict : Mapping[str, float]
        Dictionary mapping calibrated parameter names to their point estimates.
    metadata : Mapping[str, Any]
        Calibration settings and diagnostic metadata.
    """

    fit: StructuralFitResult
    sensitivity: AGSSensitivityResult
    model: FiscalSovereignResult
    parameter_dict: Mapping[str, float]
    metadata: Mapping[str, Any]

    def to_dataframe(self) -> pd.DataFrame:
        """Return parameter estimates, standard errors, and top identifying moments."""
        df = self.fit.summary().copy()
        top_map = self.sensitivity.metadata.get("top_identifying_moment", {})
        df["top_identifying_moment"] = [top_map.get(p, "") for p in self.fit.parameter_names]
        return df

    def to_frame(self) -> pd.DataFrame:
        """Alias for :meth:`to_dataframe`."""
        return self.to_dataframe()

    def summary(self) -> str:
        """Return a formatted ASCII summary of the SMM calibration and AGS sensitivity."""
        param_str = self.to_dataframe().to_string(index=False, float_format=lambda x: f"{x:.4f}")
        mom_str = self.fit.moment_fit().to_string(index=False, float_format=lambda x: f"{x:.4f}")
        lines = [
            "==========================================================================",
            "Fiscal Sovereign Default SMM / Minimum-Distance Calibration",
            "==========================================================================",
            f"Success: {self.fit.success} | Inference Valid: {self.fit.inference_valid} | "
            f"Objective Q(theta): {self.fit.objective:.6f}",
            f"Condition Number: {self.fit.condition_number:.2e} | Function Evals: {self.fit.n_evals}",
            "--------------------------------------------------------------------------",
            "Calibrated Structural Parameters:",
            param_str,
            "--------------------------------------------------------------------------",
            "Target vs. Model Moments:",
            mom_str,
            "--------------------------------------------------------------------------",
            self.sensitivity.summary(),
        ]
        return "\n".join(lines)

    def to_markdown(self, **kwargs: Any) -> str:
        """Export calibration parameter table to GitHub-flavored Markdown."""
        return df_to_markdown(self.to_dataframe(), index=False, **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Export calibration parameter table to LaTeX booktabs format."""
        return df_to_latex(self.to_dataframe(), index=False, **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Export calibration parameter table to Typst format."""
        return df_to_typst(self.to_dataframe(), index=False, **kwargs)

    def plot(self, *, ax: Any = None):
        """Plot a 2-panel comparison of Target vs. Model Moments and AGS Sensitivity Heatmap."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), layout="constrained")
        else:
            fig = ax.figure
            axes = [ax, ax]

        ax0 = axes[0]
        mfit = self.fit.moment_fit()
        x = np.arange(len(mfit))
        width = 0.36
        ax0.bar(
            x - width / 2,
            mfit["target"],
            width,
            yerr=1.96 * mfit["target_se"],
            label="Empirical Target (±1.96 SE)",
            color="#1f77b4",
            alpha=0.85,
            capsize=3,
        )
        ax0.bar(
            x + width / 2,
            mfit["model"],
            width,
            label="Calibrated Model",
            color="#ff7f0e",
            alpha=0.85,
        )
        ax0.set_xticks(x)
        ax0.set_xticklabels(mfit["label"], rotation=25, ha="right")
        ax0.set_title("Empirical vs. Calibrated Model Moments")
        ax0.legend(frameon=False)
        ax0.grid(axis="y", alpha=0.25)

        if len(axes) > 1 and axes[1] is not axes[0]:
            self.sensitivity.plot(ax=axes[1])
        return fig


def _compute_stationary_or_simulated_moments(
    sol: FiscalSovereignResult,
    labels: tuple[str, ...],
    *,
    use_simulation: bool,
    seeds: Sequence[int],
    periods: int,
    burn: int,
) -> np.ndarray:
    """Compute model moments matching ``labels`` from a solved ``FiscalSovereignResult``."""
    if use_simulation:
        zero_idx = int(sol.metadata.get("zero_index", int(np.argmin(np.abs(sol.b_grid)))))
        y0_idx = len(sol.y_grid) // 2
        u_draws = draw_uniforms(seeds, periods)
        paths = simulate_paths(sol, zero_idx, y0_idx, u_draws)
        per_path = ergodic_moments(paths, burn=burn)
        out = []
        for lbl in labels:
            vals = [float(d[lbl]) for d in per_path if lbl in d and np.isfinite(d[lbl])]
            out.append(float(np.mean(vals)) if vals else 0.0)
        return np.asarray(out, dtype=float)

    # Compute smooth stationary distribution over (b, y) grid via power iteration
    n_b, n_y = sol.default_prob.shape
    p_trans = np.asarray(sol.p_trans, dtype=float)
    rep_prob = np.clip(1.0 - np.asarray(sol.default_prob, dtype=float), 0.0, 1.0)
    zero_idx = int(sol.metadata.get("zero_index", int(np.argmin(np.abs(sol.b_grid)))))
    w = np.full((n_b, n_y), 1.0 / (n_b * n_y), dtype=float)
    b_next_idx = sol.policy_index
    for _ in range(150):
        w_rep_curr = w * rep_prob
        w_def_curr = np.sum(w * (1.0 - rep_prob), axis=0)
        w_next_b = np.zeros((n_b, n_y), dtype=float)
        np.add.at(w_next_b, (b_next_idx, np.arange(n_y)[None, :]), w_rep_curr)
        w_next_b[zero_idx, :] += w_def_curr
        w_new = w_next_b @ p_trans
        s_new = float(np.sum(w_new))
        if s_new > 0:
            w_new /= s_new
        if float(np.max(np.abs(w_new - w))) < 1e-10:
            w = w_new
            break
        w = w_new
    w_rep = w * rep_prob
    w_rep_sum = float(np.sum(w_rep))
    if w_rep_sum <= 1e-12:
        w_rep = np.full_like(w, 1.0 / w.size)
    else:
        w_rep = w_rep / w_rep_sum

    w_all_sum = float(np.sum(w))
    w_all = w / w_all_sum if w_all_sum > 1e-12 else np.full_like(w, 1.0 / w.size)

    debt_pct = 100.0 * float(np.sum(w_rep * np.maximum(sol.debt_to_gdp, 0.0)))
    # Expected spread at chosen b'(b, y)
    b_next_idx = sol.policy_index
    y_idx = np.broadcast_to(np.arange(sol.y_grid.size)[None, :], sol.debt_to_gdp.shape)
    chosen_spread = np.where(np.isfinite(sol.spread[b_next_idx, y_idx]), sol.spread[b_next_idx, y_idx], 0.0)
    spread_mean_pct = 100.0 * float(np.sum(w_rep * chosen_spread))
    spread_var = float(np.sum(w_rep * ((100.0 * chosen_spread - spread_mean_pct) ** 2)))
    spread_std_pp = float(np.sqrt(max(spread_var, 0.0)))
    annual_default_pct = 400.0 * float(np.sum(w_all * sol.default_prob))
    y_mat = np.broadcast_to(sol.y_grid[None, :], sol.g_policy.shape)
    spending_pct = 100.0 * float(np.sum(w_rep * (sol.g_policy / y_mat)))

    pb_ratio = sol.primary_balance / y_mat
    log_y = np.log(y_mat)
    mean_pb = float(np.sum(w_rep * pb_ratio))
    mean_ly = float(np.sum(w_rep * log_y))
    cov_pb_y = float(np.sum(w_rep * (pb_ratio - mean_pb) * (log_y - mean_ly)))
    std_pb = float(np.sqrt(max(np.sum(w_rep * ((pb_ratio - mean_pb) ** 2)), 1e-16)))
    std_ly = float(np.sqrt(max(np.sum(w_rep * ((log_y - mean_ly) ** 2)), 1e-16)))
    corr_pb_y = cov_pb_y / max(std_pb * std_ly, 1e-12)

    moment_map = {
        "debt_pct": debt_pct,
        "spread_mean_pct": spread_mean_pct,
        "spread_std_pp": spread_std_pp,
        "annual_default_pct": annual_default_pct,
        "spending_pct": spending_pct,
        "corr_pb_y": corr_pb_y,
    }
    return np.asarray([moment_map[lbl] for lbl in labels], dtype=float)


def calibrate_fiscal_sovereign_smm(
    targets: MomentTargets,
    *,
    param_names: Sequence[str] = ("beta", "d0"),
    theta0: Sequence[float] = (0.95, -0.188),
    bounds: Sequence[tuple[float | None, float | None]] | None = ((0.80, 0.99), (-0.45, -0.01)),
    tau: float = 0.20,
    psi: float = 1.0,
    n_b: int = 51,
    n_y: int = 9,
    sigma_eps: float = 0.002,
    rule: FiscalRule | None = None,
    use_simulation: bool = False,
    seeds: Sequence[int] = (1979, 1980, 1981, 1982),
    periods: int = 400,
    burn: int = 100,
    max_nfev: int = 25,
    tol: float = 1e-5,
    solver_kwargs: Mapping[str, Any] | None = None,
) -> SovereignSMMResult:
    """Calibrate structural parameters of the fiscal sovereign default model via SMM / minimum distance.

    Wraps :func:`puremacro.sovereign.solve_fiscal_sovereign_default` into
    :func:`puremacro.structural.bridge.fit_structural` and automatically computes the
    Andrews, Gentzkow & Shapiro (2017) sensitivity matrix :func:`ags_sensitivity`.

    Parameters
    ----------
    targets : MomentTargets
        Empirical target moments and their covariance matrix. Supported moment labels:
        ``"debt_pct"``, ``"spread_mean_pct"``, ``"spread_std_pp"``, ``"annual_default_pct"``,
        ``"spending_pct"``, and ``"corr_pb_y"``.
    param_names : Sequence[str], default ("beta", "d0")
        Names of parameters in :func:`solve_fiscal_sovereign_default` to estimate.
    theta0 : Sequence[float], default (0.95, -0.188)
        Initial parameter guess matching ``param_names``.
    bounds : Sequence[tuple[float | None, float | None]] | None
        Box bounds for each parameter.
    tau, psi : float
        Baseline tax rate and fiscal utility weight (if not included in ``param_names``).
    n_b, n_y : int
        Debt and endowment grid sizes for fast structural evaluation.
    sigma_eps : float, default 0.002
        Extreme-value taste shock scale for smooth numerical Jacobians during optimization.
    rule : FiscalRule | None, default None
        Optional fiscal rule active during calibration.
    use_simulation : bool, default False
        If False (default), evaluates moments directly from the smooth stationary distribution
        on the ``(b, y)`` grid. If True, simulates common-random-number paths via ``simulate_paths``.
    seeds : Sequence[int]
        Common random number seeds when ``use_simulation=True``.
    periods, burn : int
        Simulation horizon and burn-in periods when ``use_simulation=True``.
    max_nfev : int, default 25
        Maximum number of optimizer function evaluations.
    tol : float, default 1e-5
        Optimizer convergence tolerance.
    solver_kwargs : Mapping[str, Any] | None
        Additional keyword arguments forwarded to :func:`solve_fiscal_sovereign_default`.

    Returns
    -------
    SovereignSMMResult
    """
    base_kwargs: dict[str, Any] = {
        "tau": float(tau),
        "psi": float(psi),
        "n_b": int(n_b),
        "n_y": int(n_y),
        "sigma_eps": float(sigma_eps),
        "policy_search": "dense" if sigma_eps > 0 else "monotone",
        "tol": 1e-8,
        "max_iter": 2000,
        "rule": rule,
    }
    if solver_kwargs:
        base_kwargs.update(solver_kwargs)

    pnames = tuple(param_names)

    def _solve_at(theta_vec: np.ndarray) -> FiscalSovereignResult:
        kw = dict(base_kwargs)
        for name, val in zip(pnames, theta_vec):
            kw[name] = float(val)
        return solve_fiscal_sovereign_default(**kw)

    def _moments_at(theta_vec: np.ndarray) -> np.ndarray:
        sol = _solve_at(theta_vec)
        return _compute_stationary_or_simulated_moments(
            sol,
            targets.labels,
            use_simulation=use_simulation,
            seeds=seeds,
            periods=periods,
            burn=burn,
        )

    fit = fit_structural(
        _moments_at,
        targets,
        theta0,
        parameter_names=pnames,
        bounds=bounds,
        max_nfev=max_nfev,
        tol=tol,
    )
    sens = ags_sensitivity(fit)
    solved_model = _solve_at(fit.theta)
    param_dict = {name: float(val) for name, val in zip(pnames, fit.theta)}
    return SovereignSMMResult(
        fit=fit,
        sensitivity=sens,
        model=solved_model,
        parameter_dict=param_dict,
        metadata={
            "use_simulation": bool(use_simulation),
            "n_b": int(n_b),
            "n_y": int(n_y),
            "sigma_eps": float(sigma_eps),
        },
    )
