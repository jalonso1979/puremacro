r"""Semi-Structural Policy Counterfactuals and Optimal Policy Evaluation.

Implements two modern semi-structural macroeconomic policy evaluation frameworks
built on empirical impulse responses (LP, LP-IV, or SVAR):

1. **McKay & Wolf (2023, Review of Economic Studies)**:
   *"What Can Time-Series Regressions Tell Us About Policy Counterfactuals?"*
   Constructs Lucas-critique-robust counterfactual paths for macroeconomic targets
   :math:`\mathbf{y}` and policy instruments :math:`\mathbf{i}` under an alternative
   linear policy rule

   .. math::

       \mathbf{A}_i \mathbf{i}^{\text{cf}} + \mathbf{A}_y \mathbf{y}^{\text{cf}} = \mathbf{0}

   by combining empirical impulse responses to multiple policy news/surprise shocks
   :math:`(\boldsymbol{\Theta}_y, \boldsymbol{\Theta}_i)` with weights
   :math:`\mathbf{w}^*` that minimize the counterfactual rule residual (with optional
   Tikhonov/ridge regularization).

2. **Barnichon & Mesters (2020, AER; 2023, JPE)**:
   *Optimal Policy Perturbation (OPP)* statistic and optimal policy adjustment
   :math:`\delta^*` evaluating whether current policy stances minimize a discounted
   quadratic social loss function over target gaps:

   .. math::

       \mathcal{L}(\delta) = \frac{1}{2} \sum_{h=0}^H \beta^h \sum_{k=1}^K w_k
       \left( y_{k, h} + \mathcal{R}_{h, k} \delta \right)^2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    import matplotlib.figure

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst

__all__ = [
    "PolicyCounterfactualResult",
    "OptimalPolicyPerturbationResult",
    "mckay_wolf_counterfactual",
    "barnichon_mesters_opp",
]


@dataclass(frozen=True)
class PolicyCounterfactualResult:
    """Result container for McKay & Wolf (2023) semi-structural policy counterfactuals.

    Parameters
    ----------
    baseline_target : np.ndarray
        Baseline path or impulse response of the target variable(s), shape ``(H+1,)``
        or ``(H+1, n_y)``.
    counterfactual_target : np.ndarray
        Counterfactual path or impulse response of the target variable(s), same shape
        as ``baseline_target``.
    baseline_instrument : np.ndarray
        Baseline path or impulse response of the policy instrument, shape ``(H+1,)``.
    counterfactual_instrument : np.ndarray
        Counterfactual path or impulse response of the policy instrument, shape ``(H+1,)``.
    shock_weights : np.ndarray
        Optimal linear combination weights :math:`\\mathbf{w}^*` on the policy shocks,
        shape ``(n_shocks,)``.
    rule_residual : np.ndarray
        Counterfactual policy rule residual :math:`\\mathbf{d}_0 + \\mathbf{M} \\mathbf{w}^*`,
        shape ``(n_eq,)``.
    rule_rmse : float
        Root-mean-squared error of the counterfactual policy rule residual.
    horizon : int
        Maximum horizon :math:`H` (paths have length :math:`H + 1`).
    ridge : float
        Tikhonov regularization penalty :math:`\\lambda \\ge 0`.
    ci_lower : np.ndarray | None, default None
        Lower confidence band for ``counterfactual_target`` when ``target_irf_se`` is supplied.
    ci_upper : np.ndarray | None, default None
        Upper confidence band for ``counterfactual_target`` when ``target_irf_se`` is supplied.
    metadata : dict[str, Any]
        Additional diagnostic metadata (condition numbers, alpha level, etc.).
    """

    baseline_target: np.ndarray
    counterfactual_target: np.ndarray
    baseline_instrument: np.ndarray
    counterfactual_instrument: np.ndarray
    shock_weights: np.ndarray
    rule_residual: np.ndarray
    rule_rmse: float
    horizon: int
    ridge: float
    ci_lower: np.ndarray | None = None
    ci_upper: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert baseline and counterfactual trajectories to a horizon-indexed DataFrame."""
        T_h = self.horizon + 1
        data: dict[str, Any] = {}

        if self.baseline_target.ndim == 1:
            data["baseline_target"] = self.baseline_target
            data["counterfactual_target"] = self.counterfactual_target
            data["target_effect"] = self.counterfactual_target - self.baseline_target
            if self.ci_lower is not None and self.ci_upper is not None:
                data["ci_lower"] = self.ci_lower
                data["ci_upper"] = self.ci_upper
        else:
            n_y = self.baseline_target.shape[1]
            for k in range(n_y):
                data[f"baseline_target_{k}"] = self.baseline_target[:, k]
                data[f"counterfactual_target_{k}"] = self.counterfactual_target[:, k]
                data[f"target_effect_{k}"] = (
                    self.counterfactual_target[:, k] - self.baseline_target[:, k]
                )
                if self.ci_lower is not None and self.ci_upper is not None:
                    data[f"ci_lower_{k}"] = self.ci_lower[:, k]
                    data[f"ci_upper_{k}"] = self.ci_upper[:, k]

        data["baseline_instrument"] = self.baseline_instrument
        data["counterfactual_instrument"] = self.counterfactual_instrument
        data["instrument_effect"] = self.counterfactual_instrument - self.baseline_instrument

        if self.rule_residual.size == T_h:
            data["rule_residual"] = self.rule_residual

        return pd.DataFrame(data, index=pd.Index(range(T_h), name="horizon"))

    def to_frame(self) -> pd.DataFrame:
        """Alias for :meth:`to_dataframe`."""
        return self.to_dataframe()

    def summary(self) -> pd.DataFrame:
        """Return a summary DataFrame of policy counterfactual diagnostics."""
        target_diff = self.counterfactual_target - self.baseline_target
        inst_diff = self.counterfactual_instrument - self.baseline_instrument
        rows = [
            {"metric": "horizon", "value": float(self.horizon)},
            {"metric": "n_shocks", "value": float(self.shock_weights.size)},
            {"metric": "ridge", "value": float(self.ridge)},
            {"metric": "rule_rmse", "value": float(self.rule_rmse)},
            {
                "metric": "rule_max_abs_error",
                "value": float(np.max(np.abs(self.rule_residual))),
            },
            {
                "metric": "shock_weights_l2_norm",
                "value": float(np.linalg.norm(self.shock_weights)),
            },
            {
                "metric": "max_abs_target_effect",
                "value": float(np.max(np.abs(target_diff))),
            },
            {
                "metric": "max_abs_instrument_effect",
                "value": float(np.max(np.abs(inst_diff))),
            },
        ]
        return pd.DataFrame(rows).set_index("metric")

    def to_markdown(self, **kwargs: Any) -> str:
        """Render the counterfactual trajectory table as Markdown."""
        return df_to_markdown(self.to_dataframe(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Render the counterfactual trajectory table as a LaTeX tabular."""
        return df_to_latex(self.to_dataframe(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Render the counterfactual trajectory table as a Typst table."""
        return df_to_typst(self.to_dataframe(), **kwargs)

    def plot(
        self,
        ax: Any = None,
        show: bool = False,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Plot baseline vs. counterfactual target and policy instrument paths."""
        import matplotlib.pyplot as plt

        horizons = np.arange(self.horizon + 1)
        figsize = kwargs.pop("figsize", (10.5, 4.2))

        if ax is None:
            fig, (ax_y, ax_i) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
        elif isinstance(ax, (list, tuple, np.ndarray)) and len(ax) >= 2:
            ax_y, ax_i = ax[0], ax[1]
            fig = ax_y.figure
        else:
            ax_y = ax
            ax_i = None
            fig = ax_y.figure

        if self.baseline_target.ndim == 1:
            ax_y.plot(
                horizons,
                self.baseline_target,
                label="Baseline Target",
                color="#1f77b4",
                lw=2.0,
            )
            ax_y.plot(
                horizons,
                self.counterfactual_target,
                label="Counterfactual Target",
                color="#d62728",
                lw=2.0,
                ls="--",
            )
            if self.ci_lower is not None and self.ci_upper is not None:
                ax_y.fill_between(
                    horizons,
                    self.ci_lower,
                    self.ci_upper,
                    color="#d62728",
                    alpha=0.2,
                    label="Counterfactual CI",
                )
        else:
            palette = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
            for k in range(self.baseline_target.shape[1]):
                c = palette[k % len(palette)]
                ax_y.plot(
                    horizons,
                    self.baseline_target[:, k],
                    label=f"Baseline Target {k}",
                    color=c,
                    lw=1.8,
                )
                ax_y.plot(
                    horizons,
                    self.counterfactual_target[:, k],
                    label=f"Counterfactual Target {k}",
                    color=c,
                    lw=1.8,
                    ls="--",
                )
                if self.ci_lower is not None and self.ci_upper is not None:
                    ax_y.fill_between(
                        horizons,
                        self.ci_lower[:, k],
                        self.ci_upper[:, k],
                        color=c,
                        alpha=0.15,
                    )

        ax_y.axhline(0.0, color="0.5", ls=":", lw=0.8)
        ax_y.set_title("Macroeconomic Target Path")
        ax_y.set_xlabel("Horizon h")
        ax_y.set_ylabel("Response")
        ax_y.legend(frameon=False, fontsize="small")

        if ax_i is not None:
            ax_i.plot(
                horizons,
                self.baseline_instrument,
                label="Baseline Instrument",
                color="#1f77b4",
                lw=2.0,
            )
            ax_i.plot(
                horizons,
                self.counterfactual_instrument,
                label="Counterfactual Instrument",
                color="#d62728",
                lw=2.0,
                ls="--",
            )
            ax_i.axhline(0.0, color="0.5", ls=":", lw=0.8)
            ax_i.set_title(f"Policy Instrument (Rule RMSE={self.rule_rmse:.2e})")
            ax_i.set_xlabel("Horizon h")
            ax_i.set_ylabel("Instrument Rate / Stance")
            ax_i.legend(frameon=False, fontsize="small")

        if show:
            plt.show()
        return fig


@dataclass(frozen=True)
class OptimalPolicyPerturbationResult:
    """Result container for Barnichon & Mesters (2020, 2023) Optimal Policy Perturbation.

    Parameters
    ----------
    opp_gradient : float
        The Optimal Policy Perturbation (OPP) statistic (marginal loss slope at
        :math:`\\delta = 0`).
    opp_se : float
        Standard error of the OPP statistic.
    opp_tstat : float
        Asymptotic t-statistic :math:`\\text{OPP} / \\text{SE}(\\text{OPP})`.
    opp_pvalue : float
        Two-sided asymptotic normal p-value testing :math:`H_0: \\text{OPP} = 0`.
    optimal_adjustment : float
        Optimal policy perturbation :math:`\\delta^*` minimizing quadratic loss.
    ar_stat : float
        Weak-IV-robust Anderson-Rubin statistic testing :math:`H_0: \\delta^* = 0`.
    ar_pvalue : float
        Asymptotic :math:`\\chi^2_1` p-value for ``ar_stat``.
    loss_baseline : float
        Baseline quadratic loss :math:`\\mathcal{L}(0)`.
    loss_adjusted : float
        Quadratic loss :math:`\\mathcal{L}(\\delta^*)` after applying the optimal policy
        adjustment :math:`\\delta^*`.
    horizon : int
        Maximum horizon :math:`H` (paths have length :math:`H + 1`).
    beta : float
        Period discount factor :math:`\\beta \\in (0, 1]`.
    target_paths_baseline : np.ndarray
        Baseline target gap trajectories, shape ``(H+1, K)`` (or ``(H+1,)`` if 1D input).
    target_paths_optimal : np.ndarray
        Optimal perturbed target gap trajectories :math:`y + \\mathcal{R} \\delta^*`,
        same shape as ``target_paths_baseline``.
    metadata : dict[str, Any]
        Additional diagnostic metadata (target weights, denominator norm, etc.).
    """

    opp_gradient: float
    opp_se: float
    opp_tstat: float
    opp_pvalue: float
    optimal_adjustment: float
    ar_stat: float
    ar_pvalue: float
    loss_baseline: float
    loss_adjusted: float
    horizon: int
    beta: float
    target_paths_baseline: np.ndarray
    target_paths_optimal: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a horizon-indexed DataFrame of baseline and optimal target gap paths."""
        T_h = self.horizon + 1
        data: dict[str, Any] = {}
        if self.target_paths_baseline.ndim == 1:
            data["target_baseline"] = self.target_paths_baseline
            data["target_optimal"] = self.target_paths_optimal
            data["target_adjustment"] = self.target_paths_optimal - self.target_paths_baseline
        else:
            K = self.target_paths_baseline.shape[1]
            for k in range(K):
                data[f"target_{k}_baseline"] = self.target_paths_baseline[:, k]
                data[f"target_{k}_optimal"] = self.target_paths_optimal[:, k]
                data[f"target_{k}_adjustment"] = (
                    self.target_paths_optimal[:, k] - self.target_paths_baseline[:, k]
                )
        return pd.DataFrame(data, index=pd.Index(range(T_h), name="horizon"))

    def to_frame(self) -> pd.DataFrame:
        """Alias for :meth:`to_dataframe`."""
        return self.to_dataframe()

    def summary(self) -> pd.DataFrame:
        """Return a summary DataFrame of OPP test statistics and loss metrics."""
        loss_reduction_pct = (
            100.0 * (self.loss_baseline - self.loss_adjusted) / self.loss_baseline
            if self.loss_baseline > 0.0
            else 0.0
        )
        rows = [
            {"metric": "opp_gradient", "value": float(self.opp_gradient)},
            {"metric": "opp_se", "value": float(self.opp_se)},
            {"metric": "opp_tstat", "value": float(self.opp_tstat)},
            {"metric": "opp_pvalue", "value": float(self.opp_pvalue)},
            {"metric": "optimal_adjustment", "value": float(self.optimal_adjustment)},
            {"metric": "ar_stat", "value": float(self.ar_stat)},
            {"metric": "ar_pvalue", "value": float(self.ar_pvalue)},
            {"metric": "loss_baseline", "value": float(self.loss_baseline)},
            {"metric": "loss_adjusted", "value": float(self.loss_adjusted)},
            {"metric": "loss_reduction_pct", "value": float(loss_reduction_pct)},
            {"metric": "horizon", "value": float(self.horizon)},
            {"metric": "beta", "value": float(self.beta)},
        ]
        return pd.DataFrame(rows).set_index("metric")

    def to_markdown(self, **kwargs: Any) -> str:
        """Render the OPP summary table as Markdown."""
        return df_to_markdown(self.summary(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Render the OPP summary table as a LaTeX tabular."""
        return df_to_latex(self.summary(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Render the OPP summary table as a Typst table."""
        return df_to_typst(self.summary(), **kwargs)

    def plot(
        self,
        ax: Any = None,
        show: bool = False,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Plot baseline vs. optimal perturbed target gap trajectories."""
        import matplotlib.pyplot as plt

        horizons = np.arange(self.horizon + 1)
        figsize = kwargs.pop("figsize", (8.5, 4.2))

        if ax is None:
            fig, ax_main = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        else:
            ax_main = ax[0] if isinstance(ax, (list, tuple, np.ndarray)) else ax
            fig = ax_main.figure

        if self.target_paths_baseline.ndim == 1:
            ax_main.plot(
                horizons,
                self.target_paths_baseline,
                label="Baseline Target Gap",
                color="#1f77b4",
                lw=2.0,
            )
            ax_main.plot(
                horizons,
                self.target_paths_optimal,
                label=f"Optimal Perturbed Path (δ*={self.optimal_adjustment:+.3f})",
                color="#2ca02c",
                lw=2.0,
                ls="--",
            )
        else:
            palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
            for k in range(self.target_paths_baseline.shape[1]):
                c = palette[k % len(palette)]
                ax_main.plot(
                    horizons,
                    self.target_paths_baseline[:, k],
                    label=f"Target {k} (Baseline)",
                    color=c,
                    lw=2.0,
                )
                ax_main.plot(
                    horizons,
                    self.target_paths_optimal[:, k],
                    label=f"Target {k} (Optimal, δ*={self.optimal_adjustment:+.3f})",
                    color=c,
                    lw=2.0,
                    ls="--",
                )

        ax_main.axhline(0.0, color="0.5", ls=":", lw=0.8)
        ax_main.set_title(
            f"Barnichon-Mesters OPP (OPP={self.opp_gradient:+.3f}, p={self.opp_pvalue:.3f})"
        )
        ax_main.set_xlabel("Horizon h")
        ax_main.set_ylabel("Target Gap")
        ax_main.legend(frameon=False, fontsize="small")

        if show:
            plt.show()
        return fig


def mckay_wolf_counterfactual(
    baseline_target: Sequence[float] | np.ndarray,
    baseline_instrument: Sequence[float] | np.ndarray,
    policy_target_irfs: np.ndarray,
    policy_instrument_irfs: np.ndarray,
    *,
    rule_target_weights: Sequence[float] | np.ndarray | float = 1.5,
    rule_instrument_matrix: np.ndarray | None = None,
    rule_target_matrix: np.ndarray | None = None,
    ridge: float = 0.0,
    target_irf_se: np.ndarray | None = None,
    alpha: float = 0.10,
) -> PolicyCounterfactualResult:
    r"""Construct semi-structural policy counterfactuals per McKay & Wolf (2023).

    Given baseline paths (or impulse responses to a non-policy shock) of macro
    target(s) :math:`\mathbf{y}^{\text{base}}` and a policy instrument
    :math:`\mathbf{i}^{\text{base}}`, together with empirical impulse responses
    :math:`(\boldsymbol{\Theta}_y, \boldsymbol{\Theta}_i)` to :math:`n_{\text{shocks}}`
    policy news/surprise shocks, finds the linear combination of policy shocks
    :math:`\mathbf{w}^*` that enforces the counterfactual linear policy rule:

    .. math::

        \mathbf{A}_i \mathbf{i}^{\text{cf}} + \mathbf{A}_y \mathbf{y}^{\text{cf}} = \mathbf{0},
        \qquad \mathbf{y}^{\text{cf}} = \mathbf{y}^{\text{base}} + \boldsymbol{\Theta}_y \mathbf{w},
        \qquad \mathbf{i}^{\text{cf}} = \mathbf{i}^{\text{base}} + \boldsymbol{\Theta}_i \mathbf{w}.

    By Proposition 1 of McKay & Wolf (2023), in any linear structural rational
    expectations model where policy affects the private sector only through the
    current and expected future path of the policy instrument, this semi-structural
    combination exactly recovers the true equilibrium counterfactual without
    estimating the full structural model.

    Parameters
    ----------
    baseline_target : Sequence[float] | np.ndarray
        Baseline impulse response or path of the macroeconomic target(s), shape
        ``(H+1,)`` for a single target or ``(H+1, n_y)`` for multiple targets.
    baseline_instrument : Sequence[float] | np.ndarray
        Baseline impulse response or path of the policy instrument, shape ``(H+1,)``.
    policy_target_irfs : np.ndarray
        Empirical causal impulse responses :math:`\boldsymbol{\Theta}_y` of the
        target(s) to :math:`n_{\text{shocks}}` policy shocks, shape
        ``(H+1, n_shocks)`` or ``(H+1, n_y, n_shocks)``.
    policy_instrument_irfs : np.ndarray
        Empirical causal impulse responses :math:`\boldsymbol{\Theta}_i` of the
        policy instrument to the :math:`n_{\text{shocks}}` policy shocks, shape
        ``(H+1, n_shocks)``.
    rule_target_weights : Sequence[float] | np.ndarray | float, default 1.5
        Taylor/reaction rule weight(s) :math:`\phi_y` when ``rule_target_matrix`` is
        not explicitly provided, encoding :math:`i_h^{\text{cf}} = \sum_k \phi_k y_{k, h}^{\text{cf}}`.
    rule_instrument_matrix : np.ndarray | None, default None
        Matrix :math:`\mathbf{A}_i` of shape ``(n_eq, H+1)`` acting on the counterfactual
        instrument path. Defaults to :math:`\mathbf{I}_{H+1}`.
    rule_target_matrix : np.ndarray | None, default None
        Matrix :math:`\mathbf{A}_y` acting on the counterfactual target path(s).
        Defaults to :math:`-\phi_y \mathbf{I}_{H+1}`.
    ridge : float, default 0.0
        Tikhonov regularization parameter :math:`\lambda \ge 0` penalizing
        :math:`\lambda \|\mathbf{w}\|_2^2` when :math:`n_{\text{shocks}}` policy shocks
        are collinear or noisy.
    target_irf_se : np.ndarray | None, default None
        Standard errors of ``policy_target_irfs`` (or ``baseline_target``) used to
        construct delta-method confidence intervals ``(ci_lower, ci_upper)``.
    alpha : float, default 0.10
        Two-sided significance level in :math:`(0, 1)` for confidence intervals.

    Returns
    -------
    PolicyCounterfactualResult
        Frozen dataclass containing counterfactual paths, optimal shock weights,
        rule residuals, and optional confidence bands.
    """
    ridge_val = float(ridge)
    if not np.isfinite(ridge_val) or ridge_val < 0.0:
        raise ValueError(f"ridge must be a finite non-negative float, got {ridge!r}")
    alpha_val = float(alpha)
    if not (0.0 < alpha_val < 1.0):
        raise ValueError(f"alpha must lie strictly in (0, 1), got {alpha!r}")

    i_base = np.asarray(baseline_instrument, dtype=np.float64)
    if i_base.ndim == 2 and i_base.shape[1] == 1:
        i_base = i_base[:, 0]
    if i_base.ndim != 1 or i_base.size == 0:
        raise ValueError(
            f"baseline_instrument must be a non-empty 1D array of shape (H+1,), got shape {i_base.shape}"
        )
    T_h = int(i_base.size)
    horizon = T_h - 1

    y_base_raw = np.asarray(baseline_target, dtype=np.float64)
    if y_base_raw.ndim == 1:
        is_1d_target = True
        if y_base_raw.shape[0] != T_h:
            raise ValueError(
                f"baseline_target length ({y_base_raw.shape[0]}) must match "
                f"baseline_instrument length ({T_h})"
            )
        y_base_2d = y_base_raw[:, None]
        n_y = 1
    elif y_base_raw.ndim == 2:
        is_1d_target = False
        if y_base_raw.shape[0] != T_h or y_base_raw.shape[1] < 1:
            raise ValueError(
                f"baseline_target must have shape ({T_h}, n_y) with n_y >= 1, got {y_base_raw.shape}"
            )
        y_base_2d = y_base_raw
        n_y = int(y_base_raw.shape[1])
    else:
        raise ValueError(
            f"baseline_target must be 1D (H+1,) or 2D (H+1, n_y), got shape {y_base_raw.shape}"
        )

    theta_i = np.asarray(policy_instrument_irfs, dtype=np.float64)
    if theta_i.ndim == 1:
        theta_i = theta_i[:, None]
    if theta_i.ndim != 2 or theta_i.shape[0] != T_h or theta_i.shape[1] < 1:
        raise ValueError(
            f"policy_instrument_irfs must have shape ({T_h}, n_shocks), got {theta_i.shape}"
        )
    n_shocks = int(theta_i.shape[1])

    theta_y_raw = np.asarray(policy_target_irfs, dtype=np.float64)
    if theta_y_raw.ndim == 1:
        if n_y == 1 and n_shocks == 1 and theta_y_raw.shape[0] == T_h:
            theta_y_3d = theta_y_raw[:, None, None]
        else:
            raise ValueError(
                f"1D policy_target_irfs of shape {theta_y_raw.shape} incompatible with "
                f"T_h={T_h}, n_y={n_y}, n_shocks={n_shocks}"
            )
    elif theta_y_raw.ndim == 2:
        if theta_y_raw.shape[0] != T_h:
            raise ValueError(
                f"policy_target_irfs first dimension ({theta_y_raw.shape[0]}) must equal {T_h}"
            )
        if n_y == 1 and theta_y_raw.shape[1] == n_shocks:
            theta_y_3d = theta_y_raw[:, None, :]
        elif n_shocks == 1 and theta_y_raw.shape[1] == n_y:
            theta_y_3d = theta_y_raw[:, :, None]
        else:
            raise ValueError(
                f"2D policy_target_irfs of shape {theta_y_raw.shape} incompatible with "
                f"n_y={n_y}, n_shocks={n_shocks}; expected ({T_h}, {n_y}, {n_shocks})"
            )
    elif theta_y_raw.ndim == 3:
        if theta_y_raw.shape != (T_h, n_y, n_shocks):
            raise ValueError(
                f"3D policy_target_irfs must have shape ({T_h}, {n_y}, {n_shocks}), "
                f"got {theta_y_raw.shape}"
            )
        theta_y_3d = theta_y_raw
    else:
        raise ValueError(
            f"policy_target_irfs must be 2D or 3D array, got shape {theta_y_raw.shape}"
        )

    if rule_instrument_matrix is None:
        A_i = np.eye(T_h, dtype=np.float64)
    else:
        A_i = np.asarray(rule_instrument_matrix, dtype=np.float64)
        if A_i.ndim != 2 or A_i.shape[1] != T_h:
            raise ValueError(
                f"rule_instrument_matrix must have shape (n_eq, {T_h}), got {A_i.shape}"
            )
    n_eq = int(A_i.shape[0])

    # Parse rule_target_weights
    w_raw = np.asarray(rule_target_weights, dtype=np.float64)
    if w_raw.ndim == 0:
        phi_vec = np.full(n_y, float(w_raw.item()), dtype=np.float64)
    else:
        phi_vec = w_raw

    if rule_target_matrix is None:
        if n_eq != T_h:
            raise ValueError(
                f"rule_target_matrix must be provided when rule_instrument_matrix has "
                f"n_eq={n_eq} != H+1 ({T_h})"
            )
        if phi_vec.ndim == 1 and phi_vec.size == n_y:
            y_base_rule_term = -np.einsum("hk,k->h", y_base_2d, phi_vec)
            theta_y_rule_term = -np.einsum("hks,k->hs", theta_y_3d, phi_vec)
        elif phi_vec.ndim == 1 and phi_vec.size == T_h and n_y == 1:
            y_base_rule_term = -(phi_vec * y_base_2d[:, 0])
            theta_y_rule_term = -(phi_vec[:, None] * theta_y_3d[:, 0, :])
        elif phi_vec.ndim == 2 and phi_vec.shape == (T_h, n_y):
            y_base_rule_term = -np.sum(phi_vec * y_base_2d, axis=1)
            theta_y_rule_term = -np.einsum("hk,hks->hs", phi_vec, theta_y_3d)
        else:
            raise ValueError(
                f"rule_target_weights of shape {phi_vec.shape} incompatible with "
                f"T_h={T_h}, n_y={n_y}"
            )
    else:
        A_y = np.asarray(rule_target_matrix, dtype=np.float64)
        if A_y.ndim == 2 and A_y.shape == (n_eq, T_h):
            if n_y == 1:
                y_base_rule_term = A_y @ y_base_2d[:, 0]
                theta_y_rule_term = A_y @ theta_y_3d[:, 0, :]
            else:
                if phi_vec.ndim != 1 or phi_vec.size != n_y:
                    raise ValueError(
                        f"When rule_target_matrix has shape ({n_eq}, {T_h}) and n_y={n_y}, "
                        f"rule_target_weights must have length {n_y}"
                    )
                y_comb = y_base_2d @ phi_vec
                theta_comb = np.einsum("hks,k->hs", theta_y_3d, phi_vec)
                y_base_rule_term = A_y @ y_comb
                theta_y_rule_term = A_y @ theta_comb
        elif A_y.ndim == 3 and A_y.shape == (n_eq, n_y, T_h):
            y_base_rule_term = np.einsum("ekh,hk->e", A_y, y_base_2d)
            theta_y_rule_term = np.einsum("ekh,hks->es", A_y, theta_y_3d)
        elif A_y.ndim == 2 and A_y.shape == (n_eq, T_h * n_y):
            y_base_rule_term = A_y @ y_base_2d.reshape(-1)
            theta_y_rule_term = A_y @ theta_y_3d.reshape(T_h * n_y, n_shocks)
        else:
            raise ValueError(
                f"rule_target_matrix shape {A_y.shape} incompatible with n_eq={n_eq}, "
                f"T_h={T_h}, n_y={n_y}"
            )

    # Rule deviation r(w) = d_0 + M w
    d_0 = A_i @ i_base + y_base_rule_term
    M = A_i @ theta_i + theta_y_rule_term

    u_svd, s_svd, vt_svd = np.linalg.svd(M, full_matrices=False)
    cond_number = (
        float(s_svd[0] / s_svd[-1])
        if (s_svd.size > 0 and s_svd[-1] > 1e-15)
        else float("inf")
    )

    if ridge_val == 0.0 and M.shape[0] == M.shape[1] and cond_number < 1e12:
        try:
            w_star = np.linalg.solve(M, -d_0)
        except np.linalg.LinAlgError:
            tol = max(M.shape) * np.finfo(np.float64).eps * (float(s_svd[0]) if s_svd.size else 1.0)
            inv_s = np.where(s_svd > tol, 1.0 / np.maximum(s_svd, tol), 0.0)
            w_star = -(vt_svd.T @ (inv_s * (u_svd.T @ d_0)))
    elif ridge_val == 0.0:
        tol = max(M.shape) * np.finfo(np.float64).eps * (float(s_svd[0]) if s_svd.size else 1.0)
        inv_s = np.where(s_svd > tol, 1.0 / np.maximum(s_svd, tol), 0.0)
        w_star = -(vt_svd.T @ (inv_s * (u_svd.T @ d_0)))
    else:
        inv_s = s_svd / (s_svd ** 2 + ridge_val)
        w_star = -(vt_svd.T @ (inv_s * (u_svd.T @ d_0)))

    i_cf = i_base + theta_i @ w_star
    y_cf_2d = y_base_2d + np.einsum("hks,s->hk", theta_y_3d, w_star)
    y_cf = y_cf_2d[:, 0] if is_1d_target else y_cf_2d

    rule_res = d_0 + M @ w_star
    rule_rmse = float(np.sqrt(np.mean(rule_res ** 2)))

    ci_lower: np.ndarray | None = None
    ci_upper: np.ndarray | None = None
    if target_irf_se is not None:
        se_arr = np.asarray(target_irf_se, dtype=np.float64)
        if np.any(se_arr < 0.0):
            raise ValueError("target_irf_se must contain non-negative standard errors.")
        if se_arr.shape == theta_y_3d.shape:
            cf_se_2d = np.sqrt(np.einsum("hks,s->hk", se_arr ** 2, w_star ** 2))
            cf_se = cf_se_2d[:, 0] if is_1d_target else cf_se_2d
        elif is_1d_target and se_arr.ndim == 2 and se_arr.shape == (T_h, n_shocks):
            cf_se = np.sqrt((se_arr ** 2) @ (w_star ** 2))
        elif se_arr.shape == y_cf.shape:
            cf_se = se_arr * float(np.linalg.norm(w_star))
        else:
            raise ValueError(
                f"target_irf_se shape {se_arr.shape} must match policy_target_irfs "
                f"or baseline_target shape {y_cf.shape}"
            )
        z_crit = float(norm.ppf(1.0 - alpha_val / 2.0))
        ci_lower = y_cf - z_crit * cf_se
        ci_upper = y_cf + z_crit * cf_se

    return PolicyCounterfactualResult(
        baseline_target=y_base_raw.copy(),
        counterfactual_target=y_cf,
        baseline_instrument=i_base.copy(),
        counterfactual_instrument=i_cf,
        shock_weights=w_star,
        rule_residual=rule_res,
        rule_rmse=rule_rmse,
        horizon=horizon,
        ridge=ridge_val,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        metadata={
            "n_shocks": n_shocks,
            "n_targets": n_y,
            "n_eq": n_eq,
            "alpha": alpha_val,
            "condition_number": cond_number,
        },
    )


def _coerce_opp_covariance(
    cov: np.ndarray,
    T_h: int,
    K: int,
    name: str,
) -> np.ndarray:
    """Coerce an OPP covariance input into a full (T_h * K, T_h * K) matrix."""
    arr = np.asarray(cov, dtype=np.float64)
    N = T_h * K
    if arr.ndim == 1 and arr.size == N:
        if np.any(arr < 0.0):
            raise ValueError(f"{name} diagonal variances must be non-negative.")
        return np.diag(arr)
    if arr.ndim == 2 and arr.shape == (T_h, K) and N != T_h:
        if np.any(arr < 0.0):
            raise ValueError(f"{name} variances must be non-negative.")
        return np.diag(arr.ravel())
    if arr.ndim == 2 and arr.shape == (N, N):
        return arr
    if arr.ndim == 2 and arr.shape == (T_h, T_h) and K > 1:
        return np.kron(arr, np.eye(K, dtype=np.float64))
    raise ValueError(
        f"{name} shape {arr.shape} incompatible with T_h={T_h}, K={K} (expected ({N}, {N}) or ({T_h}, {K}))"
    )


def barnichon_mesters_opp(
    target_gaps: Sequence[Sequence[float]] | np.ndarray,
    policy_irfs: Sequence[Sequence[float]] | np.ndarray,
    *,
    target_weights: Sequence[float] | np.ndarray | None = None,
    beta: float = 0.99,
    irf_cov: np.ndarray | None = None,
    target_cov: np.ndarray | None = None,
) -> OptimalPolicyPerturbationResult:
    r"""Compute the Barnichon & Mesters (2020, 2023) Optimal Policy Perturbation (OPP) test.

    Evaluates whether a policy authority's baseline outlook :math:`\mathbf{y}_{0:H}`
    satisfies the necessary first-order optimality condition for minimizing the
    intertemporal quadratic loss function:

    .. math::

        \mathcal{L}(\delta) = \frac{1}{2} \sum_{h=0}^H \beta^h \sum_{k=1}^K w_k
        \left( y_{k, h} + \mathcal{R}_{h, k} \delta \right)^2

    where :math:`\mathcal{R}_{h, k} = \partial y_{k, t+h} / \partial \delta_t` is the
    causal impulse response of target gap :math:`k` at horizon :math:`h` to a unit
    policy perturbation :math:`\delta_t`.

    The OPP statistic is the gradient at :math:`\delta = 0`:

    .. math::

        \text{OPP} = \left. \frac{\partial \mathcal{L}(\delta)}{\partial \delta} \right|_{\delta=0}
        = \sum_{h=0}^H \beta^h \sum_{k=1}^K w_k \mathcal{R}_{h, k} y_{k, h},

    and the optimal policy adjustment minimizing :math:`\mathcal{L}(\delta)` is:

    .. math::

        \delta^* = -\frac{\sum_{h=0}^H \beta^h \sum_{k=1}^K w_k \mathcal{R}_{h, k} y_{k, h}}
        {\sum_{h=0}^H \beta^h \sum_{k=1}^K w_k \mathcal{R}_{h, k}^2}.

    Parameters
    ----------
    target_gaps : Sequence[Sequence[float]] | np.ndarray
        Forecasted or realized paths of :math:`K` policy target gaps over horizons
        :math:`h = 0, \dots, H`, shape ``(H+1, K)`` (or 1D ``(H+1,)`` for ``K=1``).
    policy_irfs : Sequence[Sequence[float]] | np.ndarray
        Causal impulse responses :math:`\mathcal{R}_{h, k}` of the :math:`K` targets
        to a unit policy perturbation, matching the shape of ``target_gaps``.
    target_weights : Sequence[float] | np.ndarray | None, default None
        Relative loss weights :math:`w_1, \dots, w_K > 0` across targets. Defaults to
        equal unit weights :math:`\mathbf{1}_K`.
    beta : float, default 0.99
        Period discount factor :math:`\beta \in (0, 1]`.
    irf_cov : np.ndarray | None, default None
        Covariance matrix of the vectorized policy IRFs :math:`\text{vec}(\mathcal{R})`,
        shape ``((H+1)*K, (H+1)*K)`` (or diagonal variances of shape ``(H+1, K)``).
    target_cov : np.ndarray | None, default None
        Covariance matrix of the vectorized target gap forecasts :math:`\text{vec}(\mathbf{y})`,
        shape ``((H+1)*K, (H+1)*K)`` (or diagonal variances of shape ``(H+1, K)``).

    Returns
    -------
    OptimalPolicyPerturbationResult
        Frozen dataclass containing the OPP gradient, standard error, t-statistic,
        p-value, weak-IV-robust Anderson-Rubin statistic, optimal adjustment :math:`\delta^*`,
        baseline vs. adjusted quadratic losses, and optimal target gap paths.
    """
    beta_val = float(beta)
    if not np.isfinite(beta_val) or not (0.0 < beta_val <= 1.0):
        raise ValueError(f"beta must lie in (0, 1], got {beta!r}")

    y_raw = np.asarray(target_gaps, dtype=np.float64)
    r_raw = np.asarray(policy_irfs, dtype=np.float64)

    if y_raw.ndim == 1:
        is_1d = True
        if y_raw.size == 0:
            raise ValueError("target_gaps cannot be empty.")
        y_2d = y_raw[:, None]
        if r_raw.ndim == 1 and r_raw.shape == y_raw.shape:
            r_2d = r_raw[:, None]
        elif r_raw.ndim == 2 and r_raw.shape == (y_raw.size, 1):
            r_2d = r_raw
        else:
            raise ValueError(
                f"policy_irfs shape {r_raw.shape} must match target_gaps shape {y_raw.shape}"
            )
    elif y_raw.ndim == 2:
        is_1d = False
        if y_raw.shape[0] == 0 or y_raw.shape[1] == 0:
            raise ValueError(f"target_gaps must be non-empty, got shape {y_raw.shape}")
        y_2d = y_raw
        if r_raw.ndim == 1 and y_raw.shape[1] == 1 and r_raw.size == y_raw.shape[0]:
            r_2d = r_raw[:, None]
        elif r_raw.ndim == 2 and r_raw.shape == y_raw.shape:
            r_2d = r_raw
        else:
            raise ValueError(
                f"policy_irfs shape {r_raw.shape} must match target_gaps shape {y_raw.shape}"
            )
    else:
        raise ValueError(
            f"target_gaps must be 1D (H+1,) or 2D (H+1, K), got shape {y_raw.shape}"
        )

    T_h, K = int(y_2d.shape[0]), int(y_2d.shape[1])
    horizon = T_h - 1

    if target_weights is None:
        w_k = np.ones(K, dtype=np.float64)
    else:
        w_k = np.asarray(target_weights, dtype=np.float64).reshape(-1)
        if w_k.size != K:
            raise ValueError(
                f"target_weights length ({w_k.size}) must equal number of targets K ({K})"
            )
        if np.any(w_k < 0.0) or float(np.sum(w_k)) <= 0.0:
            raise ValueError("target_weights must be non-negative with positive sum.")

    discount = beta_val ** np.arange(T_h, dtype=np.float64)
    weights_hk = discount[:, None] * w_k[None, :]  # shape (T_h, K)

    opp_gradient = float(np.sum(weights_hk * r_2d * y_2d))
    denom = float(np.sum(weights_hk * (r_2d ** 2)))
    if denom <= 1e-16:
        raise ValueError("policy_irfs has zero weighted energy; optimal adjustment is not identified.")

    optimal_adjustment = float(-opp_gradient / denom)

    y_opt_2d = y_2d + r_2d * optimal_adjustment
    loss_baseline = float(0.5 * np.sum(weights_hk * (y_2d ** 2)))
    loss_adjusted = float(0.5 * np.sum(weights_hk * (y_opt_2d ** 2)))

    # Inference for OPP = sum_{h,k} d_{h,k} R_{h,k} y_{h,k}
    d_vec = weights_hk.ravel()
    y_vec = y_2d.ravel()
    r_vec = r_2d.ravel()
    a_y = d_vec * y_vec  # derivative of OPP w.r.t. vec(R)
    a_r = d_vec * r_vec  # derivative of OPP w.r.t. vec(y)

    N = T_h * K
    var_opp = 0.0
    var_ar = 0.0

    if irf_cov is None and target_cov is None:
        # Default homoscedastic identity covariance when no covariance matrix is provided
        sigma_r = np.eye(N, dtype=np.float64)
        var_opp = float(a_y @ sigma_r @ a_y)
        if var_opp <= 1e-16:
            var_opp = float(a_r @ a_r) if float(a_r @ a_r) > 1e-16 else 1.0
        var_ar = var_opp
    else:
        if irf_cov is not None:
            sigma_r = _coerce_opp_covariance(irf_cov, T_h, K, "irf_cov")
            v_r = float(a_y @ sigma_r @ a_y)
            var_opp += max(0.0, v_r)
            var_ar += max(0.0, v_r)
        if target_cov is not None:
            sigma_y = _coerce_opp_covariance(target_cov, T_h, K, "target_cov")
            v_y = float(a_r @ sigma_y @ a_r)
            var_opp += max(0.0, v_y)
            var_ar += max(0.0, v_y)
        if irf_cov is not None and target_cov is not None:
            # Exact finite-sample / weak-IV product trace term tr(D Sigma_R D Sigma_y)
            d_sigma_r = sigma_r * np.outer(d_vec, d_vec)
            trace_term = max(0.0, float(np.sum(d_sigma_r * sigma_y.T)))
            if var_opp <= 1e-16:
                var_opp += trace_term
            var_ar += trace_term
        if var_opp <= 1e-16:
            var_opp = 1e-16
        if var_ar <= 1e-16:
            var_ar = var_opp

    opp_se = float(np.sqrt(var_opp))
    opp_tstat = float(opp_gradient / opp_se)
    opp_pvalue = float(2.0 * norm.sf(abs(opp_tstat)))

    ar_stat = float((opp_gradient ** 2) / var_ar)
    ar_pvalue = float(chi2.sf(ar_stat, df=1))

    target_paths_baseline = y_raw.copy()
    target_paths_optimal = y_opt_2d[:, 0] if is_1d else y_opt_2d

    orthogonality_check = float(np.sum(weights_hk * r_2d * y_opt_2d))

    return OptimalPolicyPerturbationResult(
        opp_gradient=opp_gradient,
        opp_se=opp_se,
        opp_tstat=opp_tstat,
        opp_pvalue=opp_pvalue,
        optimal_adjustment=optimal_adjustment,
        ar_stat=ar_stat,
        ar_pvalue=ar_pvalue,
        loss_baseline=loss_baseline,
        loss_adjusted=loss_adjusted,
        horizon=horizon,
        beta=beta_val,
        target_paths_baseline=target_paths_baseline,
        target_paths_optimal=target_paths_optimal,
        metadata={
            "n_targets": K,
            "target_weights": w_k.copy(),
            "irf_weighted_norm_sq": denom,
            "orthogonality_residual": orthogonality_check,
        },
    )
