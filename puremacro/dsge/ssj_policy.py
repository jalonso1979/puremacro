r"""Sequence-Space Jacobian (SSJ) Optimal Policy & HANK Welfare Decomposition.

Implements linear-quadratic sequence-space Ramsey optimal policy, optimal simple
rules (OSR), and heterogeneous-agent utilitarian vs. redistributive welfare
decomposition following:

- Auclert, M., Bardóczy, B., Rognlie, M., & Straub, L. (2021). Using the
  Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent Models.
  *Econometrica*, 89(5), 2375–2408.
- McKay, A., & Wolf, C. K. (2023). Optimal Policy Rules in HANK.
- Bhandari, A., Evans, D., Golosov, M., & Sargent, T. J. (2021). Inequality,
  Business Cycles, and Monetary-Fiscal Policy. *Econometrica*, 89(6), 2559–2599.
- Dávila, E., & Korinek, A. (2018). Pecuniary Externalities in Economies with
  Financial Frictions. *Review of Economic Studies*, 85(1), 352–395.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.optimize

from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst

__all__ = [
    "SSJPolicyResult",
    "SSJOSRResult",
    "ssj_ramsey",
    "ssj_osr",
    "ssj_welfare_decomposition",
]


def _normalize_jacobians(
    jacobians: Mapping[str, np.ndarray] | np.ndarray,
) -> tuple[dict[str, np.ndarray], int]:
    """Normalize sequence-space Jacobians into a `{target_name: (T, T) array}` dict."""
    if isinstance(jacobians, Mapping):
        if not jacobians:
            raise ValueError("`jacobians` mapping must not be empty.")
        jac_dict: dict[str, np.ndarray] = {}
        T_val: int | None = None
        for k, v in jacobians.items():
            mat = np.asarray(v, dtype=np.float64)
            if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
                raise ValueError(
                    f"Jacobian for target {k!r} must be square 2D (T, T), got shape {mat.shape}."
                )
            if T_val is None:
                T_val = int(mat.shape[0])
            elif mat.shape[0] != T_val:
                raise ValueError(
                    f"Inconsistent Jacobian dimensions: {mat.shape[0]} vs {T_val}."
                )
            jac_dict[str(k)] = mat
        assert T_val is not None
        return jac_dict, T_val

    arr = np.asarray(jacobians, dtype=np.float64)
    if arr.ndim == 2:
        if arr.shape[0] != arr.shape[1]:
            raise ValueError(f"2D Jacobian must be square (T, T), got shape {arr.shape}.")
        return {"y0": arr}, int(arr.shape[0])
    if arr.ndim == 3:
        if arr.shape[1] != arr.shape[2]:
            raise ValueError(
                f"3D Jacobian array must have shape (K, T, T), got {arr.shape}."
            )
        K, T_val, _ = arr.shape
        return {f"y{k}": arr[k] for k in range(K)}, int(T_val)
    raise ValueError(
        f"`jacobians` must be a Mapping or 2D/3D ndarray, got shape {arr.shape}."
    )


def _normalize_baseline_paths(
    baseline_paths: Mapping[str, Sequence[float] | np.ndarray] | Sequence[float] | np.ndarray,
    target_names: Sequence[str],
    T: int,
) -> dict[str, np.ndarray]:
    """Normalize baseline target deviation paths into `{target_name: (T,) array}`."""
    out: dict[str, np.ndarray] = {}
    if isinstance(baseline_paths, Mapping):
        for k in target_names:
            if k in baseline_paths:
                raw = np.asarray(baseline_paths[k], dtype=np.float64).ravel()
            elif len(baseline_paths) == 1:
                raw = np.asarray(next(iter(baseline_paths.values())), dtype=np.float64).ravel()
            else:
                raise KeyError(f"Target {k!r} not found in `baseline_paths`.")
            vec = np.zeros(T, dtype=np.float64)
            n_copy = min(T, raw.size)
            vec[:n_copy] = raw[:n_copy]
            out[k] = vec
        return out

    arr = np.asarray(baseline_paths, dtype=np.float64)
    if arr.ndim == 1:
        vec = np.zeros(T, dtype=np.float64)
        n_copy = min(T, arr.size)
        vec[:n_copy] = arr[:n_copy]
        for k in target_names:
            out[k] = vec.copy()
        return out
    if arr.ndim == 2:
        if arr.shape[0] == len(target_names):
            for idx, k in enumerate(target_names):
                vec = np.zeros(T, dtype=np.float64)
                n_copy = min(T, arr.shape[1])
                vec[:n_copy] = arr[idx, :n_copy]
                out[k] = vec
            return out
        if arr.shape[1] == len(target_names) and arr.shape[0] == T:
            for idx, k in enumerate(target_names):
                out[k] = arr[:, idx].copy()
            return out
    raise ValueError(
        f"Cannot align `baseline_paths` of shape {arr.shape} with targets {list(target_names)} and T={T}."
    )


def _normalize_target_weights(
    target_weights: Mapping[str, float] | Sequence[float] | np.ndarray | None,
    target_names: Sequence[str],
) -> dict[str, float]:
    """Normalize target weights into `{target_name: float}`."""
    if target_weights is None:
        return {k: 1.0 for k in target_names}
    if isinstance(target_weights, Mapping):
        weights = {k: float(target_weights.get(k, 1.0)) for k in target_names}
    else:
        seq = list(np.asarray(target_weights, dtype=np.float64).ravel())
        if len(seq) != len(target_names):
            raise ValueError(
                f"`target_weights` length ({len(seq)}) must match number of targets ({len(target_names)})."
            )
        weights = {k: float(w) for k, w in zip(target_names, seq)}
    for k, w in weights.items():
        if not np.isfinite(w) or w < 0.0:
            raise ValueError(f"Weight for target {k!r} must be finite and nonnegative, got {w}.")
    return weights


@dataclass(frozen=True)
class SSJPolicyResult:
    """Result container for sequence-space Ramsey optimal policy (`ssj_ramsey`).

    Attributes
    ----------
    instrument_path : np.ndarray
        Optimal commitment instrument path $\\mathbf{i}^*$ of shape `(T,)`.
    target_paths : dict[str, np.ndarray]
        Optimal target variable trajectories $\\mathbf{y}_k^* = \\bar{\\mathbf{y}}_k + J_k \\mathbf{i}^*$.
    baseline_paths : dict[str, np.ndarray]
        Baseline (zero-instrument) target deviation trajectories $\\bar{\\mathbf{y}}_k$.
    loss_optimal : float
        Discounted quadratic loss $\\mathcal{L}(\\mathbf{i}^*)$ under optimal policy.
    loss_baseline : float
        Discounted quadratic loss $\\mathcal{L}(\\mathbf{0})$ under the baseline path.
    welfare_gain : float
        Quadratic loss reduction $\\mathcal{L}(\\mathbf{0}) - \\mathcal{L}(\\mathbf{i}^*) \\ge 0$.
    horizon : int
        Sequence-space truncation horizon $T$.
    beta : float
        Planner discount factor $\\beta \\in (0, 1]$.
    distributional_gains : np.ndarray | None
        Group-level discounted welfare changes when `distributional_jacobian` is supplied.
    timeless_instrument_path : np.ndarray | None
        Timeless-perspective optimal instrument path correcting initial date-0 anticipation.
    metadata : dict[str, Any]
        Diagnostic metadata (FOC gradient infinity norm, condition number, weights).
    """

    instrument_path: np.ndarray
    target_paths: dict[str, np.ndarray]
    baseline_paths: dict[str, np.ndarray]
    loss_optimal: float
    loss_baseline: float
    welfare_gain: float
    horizon: int
    beta: float
    distributional_gains: np.ndarray | None = None
    timeless_instrument_path: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Return a diagnostic summary DataFrame of the sequence-space Ramsey solution."""
        foc_norm = float(self.metadata.get("foc_grad_inf_norm", 0.0))
        rows = [
            ("Horizon T", float(self.horizon)),
            ("Discount Factor beta", float(self.beta)),
            ("Baseline Loss", float(self.loss_baseline)),
            ("Optimal Loss", float(self.loss_optimal)),
            ("Welfare Gain (Loss Reduction)", float(self.welfare_gain)),
            ("Max |Instrument|", float(np.max(np.abs(self.instrument_path)))),
            ("Instrument L2 Norm", float(np.linalg.norm(self.instrument_path))),
            ("FOC Gradient ||grad L||_inf", foc_norm),
        ]
        if self.timeless_instrument_path is not None:
            diff_tl = float(np.max(np.abs(self.timeless_instrument_path - self.instrument_path)))
            rows.append(("Max |Timeless - Ramsey Instrument|", diff_tl))
        if self.distributional_gains is not None:
            rows.append(("Total Distributional Gain", float(np.sum(self.distributional_gains))))
        return pd.DataFrame(
            {"value": [r[1] for r in rows]},
            index=pd.Index([r[0] for r in rows], name="metric"),
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return time-series paths of instruments and targets as a DataFrame."""
        data: dict[str, np.ndarray] = {
            "instrument": np.asarray(self.instrument_path, dtype=np.float64),
        }
        if self.timeless_instrument_path is not None:
            data["timeless_instrument"] = np.asarray(
                self.timeless_instrument_path, dtype=np.float64
            )
        for k, y_opt in self.target_paths.items():
            data[f"{k}_optimal"] = np.asarray(y_opt, dtype=np.float64)
            if k in self.baseline_paths:
                data[f"{k}_baseline"] = np.asarray(self.baseline_paths[k], dtype=np.float64)
        return pd.DataFrame(data, index=pd.RangeIndex(0, self.horizon, name="t"))

    def to_frame(self) -> pd.DataFrame:
        """Alias for `to_dataframe()`."""
        return self.to_dataframe()

    def to_markdown(self, **kwargs: Any) -> str:
        """Render summary table as Markdown."""
        return df_to_markdown(self.summary(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Render summary table as LaTeX tabular."""
        return df_to_latex(self.summary(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Render summary table as Typst table."""
        return df_to_typst(self.summary(), **kwargs)

    def plot(
        self,
        ax: Any = None,
        show: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Plot optimal instrument and target trajectories against baseline paths."""
        import matplotlib.pyplot as plt

        t_grid = np.arange(self.horizon)
        if ax is None:
            fig, ax_plot = plt.subplots(figsize=kwargs.pop("figsize", (8.0, 4.5)))
        else:
            ax_plot = ax
            fig = ax_plot.figure

        ax_plot.plot(
            t_grid,
            self.instrument_path,
            lw=2.2,
            color="#1f77b4",
            label="Optimal Instrument (Ramsey)",
            **kwargs,
        )
        if self.timeless_instrument_path is not None:
            ax_plot.plot(
                t_grid,
                self.timeless_instrument_path,
                lw=1.6,
                ls=":",
                color="#ff7f0e",
                label="Timeless Instrument",
            )
        for idx, (k, y_opt) in enumerate(self.target_paths.items()):
            col = "#2ca02c" if idx == 0 else "#d62728"
            ax_plot.plot(t_grid, y_opt, lw=1.8, color=col, label=f"{k} (optimal)")
            if k in self.baseline_paths:
                ax_plot.plot(
                    t_grid,
                    self.baseline_paths[k],
                    lw=1.3,
                    ls="--",
                    color=col,
                    alpha=0.7,
                    label=f"{k} (baseline)",
                )
        ax_plot.axhline(0.0, color="gray", lw=0.8, ls="--")
        ax_plot.set_xlabel("Horizon t")
        ax_plot.set_ylabel("Deviation")
        ax_plot.set_title("Sequence-Space Ramsey Optimal Policy")
        ax_plot.legend(frameon=False, fontsize=8)
        ax_plot.grid(True, linestyle=":", alpha=0.5)
        if show:
            plt.show()
        return fig


@dataclass(frozen=True)
class SSJOSRResult:
    """Result container for sequence-space Optimal Simple Rules (`ssj_osr`).

    Attributes
    ----------
    coefficients : dict[str, float]
        Optimal feedback rule coefficients $\\phi^*$.
    coefficient_vector : np.ndarray
        Optimal feedback coefficients as a 1D array.
    instrument_path : np.ndarray
        Closed-loop instrument path under the optimal simple rule.
    target_paths : dict[str, np.ndarray]
        Closed-loop target variable paths under the optimal simple rule.
    baseline_paths : dict[str, np.ndarray]
        Uncontrolled baseline target paths ($\\phi = 0$).
    loss_optimal : float
        Discounted quadratic loss under $\\phi^*$.
    loss_baseline : float
        Discounted quadratic loss under $\\phi = 0$.
    loss_naive : float
        Discounted quadratic loss under naive unit-feedback rule.
    welfare_gain : float
        Loss reduction relative to baseline $\\mathcal{L}(\\mathbf{0}) - \\mathcal{L}(\\phi^*)$.
    horizon : int
        Truncation horizon $T$.
    beta : float
        Discount factor $\\beta$.
    lags : tuple[int, ...]
        Lag orders included in the feedback rule.
    converged : bool
        Whether the optimizer converged cleanly.
    iterations : int
        Number of optimizer iterations executed.
    metadata : dict[str, Any]
        Additional optimization metadata.
    """

    coefficients: dict[str, float]
    coefficient_vector: np.ndarray
    instrument_path: np.ndarray
    target_paths: dict[str, np.ndarray]
    baseline_paths: dict[str, np.ndarray]
    loss_optimal: float
    loss_baseline: float
    loss_naive: float
    welfare_gain: float
    horizon: int
    beta: float
    lags: tuple[int, ...]
    converged: bool
    iterations: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Return a summary DataFrame of OSR coefficients and loss metrics."""
        rows: list[tuple[str, float]] = []
        for k, v in self.coefficients.items():
            rows.append((f"coef_{k}", float(v)))
        rows.extend([
            ("loss_optimal", float(self.loss_optimal)),
            ("loss_naive", float(self.loss_naive)),
            ("loss_baseline", float(self.loss_baseline)),
            ("welfare_gain", float(self.welfare_gain)),
            ("horizon", float(self.horizon)),
            ("beta", float(self.beta)),
            ("converged", 1.0 if self.converged else 0.0),
        ])
        return pd.DataFrame(
            {"value": [r[1] for r in rows]},
            index=pd.Index([r[0] for r in rows], name="metric"),
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return closed-loop time-series trajectories as a DataFrame."""
        data: dict[str, np.ndarray] = {
            "instrument": np.asarray(self.instrument_path, dtype=np.float64),
        }
        for k, y_path in self.target_paths.items():
            data[f"{k}_osr"] = np.asarray(y_path, dtype=np.float64)
            if k in self.baseline_paths:
                data[f"{k}_baseline"] = np.asarray(self.baseline_paths[k], dtype=np.float64)
        return pd.DataFrame(data, index=pd.RangeIndex(0, self.horizon, name="t"))

    def to_frame(self) -> pd.DataFrame:
        """Alias for `to_dataframe()`."""
        return self.to_dataframe()

    def to_markdown(self, **kwargs: Any) -> str:
        """Render summary table as Markdown."""
        return df_to_markdown(self.summary(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Render summary table as LaTeX tabular."""
        return df_to_latex(self.summary(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Render summary table as Typst table."""
        return df_to_typst(self.summary(), **kwargs)

    def plot(
        self,
        ax: Any = None,
        show: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Plot OSR instrument and target trajectories."""
        import matplotlib.pyplot as plt

        t_grid = np.arange(self.horizon)
        if ax is None:
            fig, ax_plot = plt.subplots(figsize=kwargs.pop("figsize", (8.0, 4.5)))
        else:
            ax_plot = ax
            fig = ax_plot.figure

        ax_plot.plot(
            t_grid,
            self.instrument_path,
            lw=2.0,
            color="#1f77b4",
            label="OSR Instrument",
            **kwargs,
        )
        for idx, (k, y_opt) in enumerate(self.target_paths.items()):
            col = "#2ca02c" if idx == 0 else "#d62728"
            ax_plot.plot(t_grid, y_opt, lw=1.8, color=col, label=f"{k} (OSR)")
            if k in self.baseline_paths:
                ax_plot.plot(
                    t_grid,
                    self.baseline_paths[k],
                    lw=1.3,
                    ls="--",
                    color=col,
                    alpha=0.7,
                    label=f"{k} (baseline)",
                )
        ax_plot.axhline(0.0, color="gray", lw=0.8, ls="--")
        ax_plot.set_xlabel("Horizon t")
        ax_plot.set_ylabel("Deviation")
        ax_plot.set_title("Sequence-Space Optimal Simple Rule (OSR)")
        ax_plot.legend(frameon=False, fontsize=8)
        ax_plot.grid(True, linestyle=":", alpha=0.5)
        if show:
            plt.show()
        return fig


def _compute_timeless_path(
    jac_dict: Mapping[str, np.ndarray],
    base_dict: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
    b_vec: np.ndarray,
    H_ramsey: np.ndarray,
    rhs_ramsey: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Compute timeless-perspective instrument path correcting date-0 anticipation."""
    T = len(b_vec)
    if T <= 2:
        try:
            return np.linalg.solve(H_ramsey, rhs_ramsey)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(H_ramsey, rhs_ramsey, rcond=None)[0]

    # Measure forward-looking anticipation kernel from an interior column t_ref
    t_ref = min(T - 1, max(1, T // 2))
    H_pre = np.zeros((T, T), dtype=np.float64)
    max_leads = min(t_ref, 16)

    for k, J_k in jac_dict.items():
        w_k = weights[k]
        if w_k <= 0.0:
            continue
        # Extract anticipation profile a_m = J_k[t_ref - m, t_ref] for m = 1 .. max_leads
        lead_profile = np.array([J_k[t_ref - m, t_ref] for m in range(1, max_leads + 1)], dtype=np.float64)
        if np.all(np.abs(lead_profile) < 1e-14):
            continue
        # For each pre-initial period r = 1 .. max_leads (date -r), its response to
        # future instruments i_0, i_1, ... is sum_{t >= 0} lead_profile[t + r - 1] * i_t
        for r in range(1, max_leads + 1):
            row_r = np.zeros(T, dtype=np.float64)
            n_active = max_leads - r + 1
            row_r[: min(T, n_active)] = lead_profile[r - 1 : r - 1 + min(T, n_active)]
            discount_r = float(beta ** (-r)) if beta > 0 else 1.0
            H_pre += w_k * discount_r * np.outer(row_r, row_r)

    H_tl = H_ramsey + H_pre
    try:
        return np.linalg.solve(H_tl, rhs_ramsey)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(H_tl, rhs_ramsey, rcond=None)[0]


def ssj_ramsey(
    jacobians: Mapping[str, np.ndarray] | np.ndarray,
    baseline_paths: Mapping[str, Sequence[float] | np.ndarray] | Sequence[float] | np.ndarray,
    *,
    target_weights: Mapping[str, float] | Sequence[float] | None = None,
    instrument_weight: float = 0.1,
    beta: float = 0.99,
    distributional_jacobian: np.ndarray | None = None,
    welfare_weights: Sequence[float] | np.ndarray | None = None,
    compute_timeless: bool = True,
    incorporate_welfare_gradient: bool = False,
) -> SSJPolicyResult:
    r"""Solve linear-quadratic sequence-space Ramsey optimal policy (McKay & Wolf 2023).

    Minimizes the discounted sequence-space quadratic objective:
    $$\mathcal{L}(\mathbf{i}) = \frac{1}{2} \sum_k w_k (\bar{y}_k + J_k \mathbf{i})' B (\bar{y}_k + J_k \mathbf{i}) + \frac{\lambda_i}{2} \mathbf{i}' B \mathbf{i}$$
    where $B = \operatorname{diag}(1, \beta, \beta^2, \dots, \beta^{T-1})$, via the exact
    first-order linear system:
    $$\left( \sum_k w_k J_k' B J_k + \lambda_i B \right) \mathbf{i}^* = -\sum_k w_k J_k' B \bar{y}_k.$$

    Parameters
    ----------
    jacobians : Mapping[str, np.ndarray] or np.ndarray
        Sequence-space Jacobians $J_k = \partial \mathbf{Y}_k / \partial \mathbf{i}$ of shape `(T, T)`.
    baseline_paths : Mapping[str, Sequence[float] | np.ndarray] or Sequence[float] or np.ndarray
        Baseline deviations $\bar{\mathbf{y}}_k$ of length `T` in the absence of policy action.
    target_weights : Mapping[str, float] or Sequence[float], optional
        Stabilization weights $w_k \ge 0$. Defaults to `1.0` for every target.
    instrument_weight : float, default 0.1
        Penalty weight $\lambda_i \ge 0$ on instrument squared deviations.
    beta : float, default 0.99
        Planner discount factor $\beta \in (0, 1]$.
    distributional_jacobian : np.ndarray, optional
        Group-level consumption/utility Jacobians of shape `(n_groups, T, T)`.
    welfare_weights : Sequence[float] or np.ndarray, optional
        Pareto / marginal-utility weights of shape `(n_groups,)`.
    compute_timeless : bool, default True
        Whether to compute the timeless-perspective instrument path.
    incorporate_welfare_gradient : bool, default False
        If True and `distributional_jacobian` is supplied, subtracts the linear social
        welfare gradient $\sum_g \omega_g J_{g}^{\text{dist}\prime} b$ from the FOC.

    Returns
    -------
    SSJPolicyResult
    """
    if not (0.0 < float(beta) <= 1.0):
        raise ValueError(f"`beta` must be in (0, 1], got {beta}.")
    if float(instrument_weight) < 0.0 or not np.isfinite(instrument_weight):
        raise ValueError(
            f"`instrument_weight` must be finite and nonnegative, got {instrument_weight}."
        )

    jac_dict, T = _normalize_jacobians(jacobians)
    target_names = list(jac_dict.keys())
    base_dict = _normalize_baseline_paths(baseline_paths, target_names, T)
    weights = _normalize_target_weights(target_weights, target_names)
    lam_i = float(instrument_weight)
    beta_f = float(beta)

    b_vec = beta_f ** np.arange(T, dtype=np.float64)
    B = np.diag(b_vec)

    H = lam_i * B.copy()
    g0 = np.zeros(T, dtype=np.float64)
    loss_base = 0.0

    for k in target_names:
        w_k = weights[k]
        J_k = jac_dict[k]
        y0_k = base_dict[k]
        By0 = b_vec * y0_k
        loss_base += 0.5 * w_k * float(np.dot(y0_k, By0))
        if w_k > 0.0:
            BJ_k = b_vec[:, None] * J_k
            H += w_k * (J_k.T @ BJ_k)
            g0 += w_k * (J_k.T @ By0)

    H = 0.5 * (H + H.T)

    social_grad = np.zeros(T, dtype=np.float64)
    dist_J_arr: np.ndarray | None = None
    w_groups: np.ndarray | None = None

    if distributional_jacobian is not None:
        dist_J_arr = np.asarray(distributional_jacobian, dtype=np.float64)
        if dist_J_arr.ndim != 3 or dist_J_arr.shape[1:] != (T, T):
            raise ValueError(
                f"`distributional_jacobian` must have shape (n_groups, {T}, {T}), got {dist_J_arr.shape}."
            )
        n_groups = dist_J_arr.shape[0]
        if welfare_weights is not None:
            w_groups = np.asarray(welfare_weights, dtype=np.float64).ravel()
            if w_groups.size != n_groups:
                raise ValueError(
                    f"`welfare_weights` length ({w_groups.size}) must match n_groups ({n_groups})."
                )
        else:
            w_groups = np.ones(n_groups, dtype=np.float64) / float(n_groups)

        for g in range(n_groups):
            social_grad += w_groups[g] * (dist_J_arr[g].T @ b_vec)

        if incorporate_welfare_gradient:
            g0 = g0 - social_grad

    rhs = -g0

    # Solve H i* = rhs
    i_opt: np.ndarray
    total_w = sum(weights.values())
    if total_w == 0.0 and lam_i > 0.0 and not incorporate_welfare_gradient:
        i_opt = np.zeros(T, dtype=np.float64)
    elif np.all(np.abs(H) < 1e-15):
        i_opt = np.zeros(T, dtype=np.float64)
    else:
        try:
            i_opt = scipy.linalg.solve(H, rhs, assume_a="sym")
            res_vec = rhs - (H @ i_opt)
            i_cand = i_opt + scipy.linalg.solve(H, res_vec, assume_a="sym")
            if float(np.max(np.abs(H @ i_cand - rhs))) < float(np.max(np.abs(res_vec))):
                i_opt = i_cand
        except scipy.linalg.LinAlgError:
            i_opt = np.linalg.lstsq(H, rhs, rcond=None)[0]

    # Compute optimal target paths and loss
    opt_targets: dict[str, np.ndarray] = {}
    loss_opt = 0.5 * lam_i * float(np.dot(i_opt, b_vec * i_opt))
    for k in target_names:
        y_k = base_dict[k] + jac_dict[k] @ i_opt
        opt_targets[k] = y_k
        loss_opt += 0.5 * weights[k] * float(np.dot(y_k, b_vec * y_k))

    if incorporate_welfare_gradient and dist_J_arr is not None:
        loss_opt -= float(np.dot(social_grad, i_opt))

    grad_at_opt = H @ i_opt + g0
    foc_grad_inf_norm = float(np.max(np.abs(grad_at_opt)))

    dist_gains: np.ndarray | None = None
    if dist_J_arr is not None and w_groups is not None:
        n_groups = dist_J_arr.shape[0]
        dist_gains = np.empty(n_groups, dtype=np.float64)
        for g in range(n_groups):
            dc_g = dist_J_arr[g] @ i_opt
            dist_gains[g] = float(w_groups[g] * np.dot(b_vec, dc_g))

    timeless_path: np.ndarray | None = None
    if compute_timeless:
        timeless_path = _compute_timeless_path(
            jac_dict, base_dict, weights, b_vec, H, rhs, beta_f
        )

    welfare_gain = float(loss_base - loss_opt)

    return SSJPolicyResult(
        instrument_path=i_opt,
        target_paths=opt_targets,
        baseline_paths=base_dict,
        loss_optimal=float(loss_opt),
        loss_baseline=float(loss_base),
        welfare_gain=welfare_gain,
        horizon=T,
        beta=beta_f,
        distributional_gains=dist_gains,
        timeless_instrument_path=timeless_path,
        metadata={
            "foc_gradient": grad_at_opt,
            "foc_grad_inf_norm": foc_grad_inf_norm,
            "hessian": H,
            "target_weights": weights,
            "instrument_weight": lam_i,
            "social_welfare_gradient": social_grad if dist_J_arr is not None else None,
        },
    )


def _lag_matrix(T: int, lag: int) -> np.ndarray:
    """Return `(T, T)` lag operator matrix $L_{\\text{lag}}$ such that $(L_{\\text{lag}} y)_t = y_{t - \\text{lag}}$."""
    mat = np.zeros((T, T), dtype=np.float64)
    if abs(lag) >= T:
        return mat
    if lag == 0:
        np.fill_diagonal(mat, 1.0)
    elif lag > 0:
        idx = np.arange(lag, T)
        mat[idx, idx - lag] = 1.0
    else:
        lead = -lag
        idx = np.arange(T - lead)
        mat[idx, idx + lead] = 1.0
    return mat


def ssj_osr(
    jacobians: Mapping[str, np.ndarray],
    shock_paths: Mapping[str, np.ndarray],
    *,
    target_weights: Mapping[str, float] | None = None,
    instrument_weight: float = 0.1,
    beta: float = 0.99,
    lags: Sequence[int] = (0,),
) -> SSJOSRResult:
    r"""Optimize simple linear feedback rule coefficients in sequence space.

    Finds the feedback coefficients $\phi = \{\phi_{k, \ell}\}$ in the rule
    $$i_t = \sum_{k} \sum_{\ell \in \text{lags}} \phi_{k, \ell} \, y_{k, t - \ell}$$
    minimizing the sequence-space discounted quadratic loss via `scipy.optimize.minimize`.

    Parameters
    ----------
    jacobians : Mapping[str, np.ndarray]
        Sequence-space Jacobians $J_k = \partial \mathbf{Y}_k / \partial \mathbf{i}$ of shape `(T, T)`.
    shock_paths : Mapping[str, np.ndarray]
        Baseline target deviation paths $\bar{\mathbf{y}}_k$ (or exogenous shock paths) of length `T`.
    target_weights : Mapping[str, float], optional
        Stabilization weights $w_k \ge 0$. Defaults to `1.0` per target.
    instrument_weight : float, default 0.1
        Instrument penalty weight $\lambda_i \ge 0$.
    beta : float, default 0.99
        Discount factor $\beta \in (0, 1]$.
    lags : Sequence[int], default (0,)
        Lag indices $\ell$ included in the feedback rule.

    Returns
    -------
    SSJOSRResult
    """
    if not (0.0 < float(beta) <= 1.0):
        raise ValueError(f"`beta` must be in (0, 1], got {beta}.")
    if float(instrument_weight) < 0.0:
        raise ValueError(f"`instrument_weight` must be nonnegative, got {instrument_weight}.")

    jac_dict, T = _normalize_jacobians(jacobians)
    target_names = list(jac_dict.keys())
    weights = _normalize_target_weights(target_weights, target_names)
    lam_i = float(instrument_weight)
    beta_f = float(beta)
    lags_tup = tuple(int(l) for l in lags)
    if not lags_tup:
        raise ValueError("`lags` must contain at least one lag index.")

    # Align baseline paths with target_names
    if all(k in shock_paths for k in target_names):
        base_dict = _normalize_baseline_paths(shock_paths, target_names, T)
    elif len(shock_paths) == len(target_names):
        ordered_vals = [np.asarray(v, dtype=np.float64).ravel() for v in shock_paths.values()]
        base_dict = _normalize_baseline_paths(
            {k: val for k, val in zip(target_names, ordered_vals)}, target_names, T
        )
    else:
        combined = np.zeros(T, dtype=np.float64)
        for v in shock_paths.values():
            arr = np.asarray(v, dtype=np.float64).ravel()
            n_c = min(T, arr.size)
            combined[:n_c] += arr[:n_c]
        base_dict = {k: combined.copy() for k in target_names}

    b_vec = beta_f ** np.arange(T, dtype=np.float64)
    eye_T = np.eye(T, dtype=np.float64)
    lag_mats = {l: _lag_matrix(T, l) for l in lags_tup}

    param_keys: list[tuple[str, int]] = [
        (k, l) for k in target_names for l in lags_tup
    ]
    n_params = len(param_keys)

    def _solve_closed_loop(phi: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray] | None:
        R_map: dict[str, np.ndarray] = {
            k: np.zeros((T, T), dtype=np.float64) for k in target_names
        }
        for idx, (k, l) in enumerate(param_keys):
            R_map[k] = R_map[k] + float(phi[idx]) * lag_mats[l]

        M = eye_T.copy()
        rhs = np.zeros(T, dtype=np.float64)
        for k in target_names:
            M = M - R_map[k] @ jac_dict[k]
            rhs = rhs + R_map[k] @ base_dict[k]

        try:
            i_path = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(i_path)) or np.max(np.abs(i_path)) > 1e8:
            return None

        y_paths = {k: base_dict[k] + jac_dict[k] @ i_path for k in target_names}
        return i_path, y_paths, M

    def _loss_and_grad(phi: np.ndarray) -> tuple[float, np.ndarray]:
        solved = _solve_closed_loop(phi)
        if solved is None:
            return 1e12 + 1e4 * float(np.dot(phi, phi)), 2e4 * phi
        i_path, y_paths, M = solved
        loss = 0.5 * lam_i * float(np.dot(i_path, b_vec * i_path))
        dL_di = lam_i * (b_vec * i_path)
        for k in target_names:
            w_k = weights[k]
            y_k = y_paths[k]
            By_k = b_vec * y_k
            loss += 0.5 * w_k * float(np.dot(y_k, By_k))
            if w_k > 0.0:
                dL_di = dL_di + w_k * (jac_dict[k].T @ By_k)

        try:
            mu = np.linalg.solve(M.T, dL_di)
        except np.linalg.LinAlgError:
            mu = np.linalg.lstsq(M.T, dL_di, rcond=None)[0]

        grad = np.empty(n_params, dtype=np.float64)
        for idx, (k, l) in enumerate(param_keys):
            grad[idx] = float(np.dot(mu, lag_mats[l] @ y_paths[k]))
        return float(loss), grad

    loss_baseline, _ = _loss_and_grad(np.zeros(n_params, dtype=np.float64))
    phi_naive = np.full(n_params, -0.5, dtype=np.float64)
    loss_naive_neg, _ = _loss_and_grad(phi_naive)
    loss_naive_pos, _ = _loss_and_grad(np.full(n_params, 0.5, dtype=np.float64))
    loss_naive = min(loss_naive_neg, loss_naive_pos)

    # Optimize starting from best of 0 and small negative/positive seeds
    seeds = [
        np.zeros(n_params, dtype=np.float64),
        np.full(n_params, -0.25, dtype=np.float64),
        np.full(n_params, 0.25, dtype=np.float64),
    ]
    best_res = None
    for s0 in seeds:
        res = scipy.optimize.minimize(
            _loss_and_grad,
            s0,
            method="BFGS",
            jac=True,
            options={"gtol": 1e-10, "maxiter": 300},
        )
        if best_res is None or res.fun < best_res.fun:
            best_res = res

    assert best_res is not None
    phi_opt = np.asarray(best_res.x, dtype=np.float64)
    solved_opt = _solve_closed_loop(phi_opt)
    if solved_opt is None:
        phi_opt = np.zeros(n_params, dtype=np.float64)
        solved_opt = _solve_closed_loop(phi_opt)
        assert solved_opt is not None

    i_opt, y_opt_paths, _ = solved_opt
    loss_opt, grad_opt = _loss_and_grad(phi_opt)

    coef_dict: dict[str, float] = {}
    for idx, (k, l) in enumerate(param_keys):
        if len(lags_tup) == 1 and l == 0:
            coef_dict[k] = float(phi_opt[idx])
        else:
            coef_dict[f"{k}_lag{l}"] = float(phi_opt[idx])
            if l == 0 and k not in coef_dict:
                coef_dict[k] = float(phi_opt[idx])

    return SSJOSRResult(
        coefficients=coef_dict,
        coefficient_vector=phi_opt,
        instrument_path=i_opt,
        target_paths=y_opt_paths,
        baseline_paths=base_dict,
        loss_optimal=float(loss_opt),
        loss_baseline=float(loss_baseline),
        loss_naive=float(loss_naive),
        welfare_gain=float(loss_baseline - loss_opt),
        horizon=T,
        beta=beta_f,
        lags=lags_tup,
        converged=bool(best_res.success or np.max(np.abs(grad_opt)) < 1e-7),
        iterations=int(getattr(best_res, "nit", 1)),
        metadata={
            "grad_inf_norm": float(np.max(np.abs(grad_opt))),
            "message": str(getattr(best_res, "message", "")),
            "target_weights": weights,
            "instrument_weight": lam_i,
        },
    )


def ssj_welfare_decomposition(
    group_jacobians: np.ndarray,
    instrument_path: np.ndarray,
    *,
    group_shares: Sequence[float] | np.ndarray | None = None,
    group_marginal_utilities: Sequence[float] | np.ndarray | None = None,
    beta: float = 0.99,
) -> pd.DataFrame:
    r"""Decompose aggregate sequence-space welfare into Efficiency vs. Redistribution.

    Following Bhandari et al. (2021) and Auclert, Rognlie & Straub, decomposes
    social welfare change $\Delta W = \sum_{g=1}^G \pi_g u'_{c, g} \text{PV}_g$ into:
    $$\Delta W = \underbrace{\bar{u}'_c \sum_{g=1}^G \pi_g \text{PV}_g}_{\text{Aggregate Efficiency}} + \underbrace{\sum_{g=1}^G \pi_g (u'_{c, g} - \bar{u}'_c) \text{PV}_g}_{\text{Redistribution / Inequality}}$$
    where $\text{PV}_g = \sum_{t=0}^{T-1} \beta^t (\mathcal{J}_g \mathbf{i})_t$ and
    $\bar{u}'_c = \sum_{g=1}^G \pi_g u'_{c, g}$.

    Parameters
    ----------
    group_jacobians : np.ndarray
        Array of shape `(n_groups, T, T)` giving each group's consumption/utility
        sequence-space Jacobian with respect to the policy instrument.
    instrument_path : np.ndarray
        Policy instrument path of shape `(T,)`.
    group_shares : Sequence[float] or np.ndarray, optional
        Population shares $\pi_g$ of shape `(n_groups,)`. Normalized to sum to 1.
        Defaults to uniform shares $1 / G$.
    group_marginal_utilities : Sequence[float] or np.ndarray, optional
        Steady-state marginal utilities (or social welfare weights) $u'_{c, g}$ of
        shape `(n_groups,)`. Defaults to `1.0` for all groups.
    beta : float, default 0.99
        Planner discount factor $\beta \in (0, 1]$.

    Returns
    -------
    pd.DataFrame
        Group-level welfare decomposition table with columns:
        `["share", "marginal_utility", "pv_gain", "efficiency", "redistribution", "total_welfare"]`.
        Aggregate totals are stored in `df.attrs`.
    """
    J_grp = np.asarray(group_jacobians, dtype=np.float64)
    i_vec = np.asarray(instrument_path, dtype=np.float64).ravel()
    if J_grp.ndim != 3:
        raise ValueError(
            f"`group_jacobians` must be a 3D array of shape (n_groups, T, T), got {J_grp.shape}."
        )
    n_groups, T1, T2 = J_grp.shape
    if T1 != T2 or T1 != i_vec.size:
        raise ValueError(
            f"Incompatible shapes: `group_jacobians` {J_grp.shape} vs `instrument_path` {i_vec.shape}."
        )
    if not (0.0 < float(beta) <= 1.0):
        raise ValueError(f"`beta` must be in (0, 1], got {beta}.")

    if group_shares is None:
        shares = np.full(n_groups, 1.0 / float(n_groups), dtype=np.float64)
    else:
        shares = np.asarray(group_shares, dtype=np.float64).ravel()
        if shares.size != n_groups:
            raise ValueError(
                f"`group_shares` length ({shares.size}) must match n_groups ({n_groups})."
            )
        if np.any(shares < 0.0) or float(np.sum(shares)) <= 0.0:
            raise ValueError("`group_shares` must be nonnegative with positive sum.")
        shares = shares / float(np.sum(shares))

    if group_marginal_utilities is None:
        mu = np.ones(n_groups, dtype=np.float64)
    else:
        mu = np.asarray(group_marginal_utilities, dtype=np.float64).ravel()
        if mu.size != n_groups:
            raise ValueError(
                f"`group_marginal_utilities` length ({mu.size}) must match n_groups ({n_groups})."
            )

    b_vec = (float(beta) ** np.arange(T1, dtype=np.float64))
    pv_gains = np.array([float(np.dot(b_vec, J_grp[g] @ i_vec)) for g in range(n_groups)], dtype=np.float64)

    mean_mu = float(np.dot(shares, mu))
    efficiency_comp = shares * mean_mu * pv_gains
    redistribution_comp = shares * (mu - mean_mu) * pv_gains
    total_welfare = shares * mu * pv_gains

    df = pd.DataFrame(
        {
            "share": shares,
            "marginal_utility": mu,
            "pv_gain": pv_gains,
            "efficiency": efficiency_comp,
            "redistribution": redistribution_comp,
            "total_welfare": total_welfare,
        },
        index=pd.Index([f"group_{g}" for g in range(n_groups)], name="group"),
    )
    df.attrs["efficiency_total"] = float(np.sum(efficiency_comp))
    df.attrs["redistribution_total"] = float(np.sum(redistribution_comp))
    df.attrs["total_welfare"] = float(np.sum(total_welfare))
    df.attrs["mean_marginal_utility"] = mean_mu
    return df
