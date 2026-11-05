r"""Stochastic Risky Steady State & Global Precautionary ZLB/OBC Solvers.

Implements:
1. `solve_risky_steady_state`: Coeurdacier, Rey & Winant (2011, *AER*) stochastic
   risky steady state $y^{\text{RSS}}$ solving
   $$f(y^{\text{RSS}}, y^{\text{RSS}}, 0) + \frac{1}{2} \Psi(\Sigma) = 0,$$
   where $\Psi_i(\Sigma) = \operatorname{tr}\!\left(\frac{\partial^2 f_i}{\partial y_{t+1} \partial y_{t+1}'} \operatorname{Cov}_t(y_{t+1})\right)$,
   using exact second-order `HyperDualArray` forward differentiation (`hessian_forward`)
   with central finite-difference fallback and damped Newton-Raphson iteration.

2. `solve_precautionary_obc`: Global nonlinear time-iteration solver on a state grid
   for the canonical forward-looking New Keynesian model with an occasionally binding
   Zero Lower Bound ($i_t = \max(i_{\text{floor}}, \phi_\pi \pi_t + \phi_x x_t)$),
   integrating future expectations via Gauss-Hermite quadrature (`scipy.special.roots_hermite`)
   and comparing against the piecewise-linear certainty-equivalent solution on the
   same grid to isolate the **precautionary deflationary and output bias** near the ZLB
   (Adam & Billi 2006, 2007; Nakata & Schmidt 2019).
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.special

from puremacro._autodiff import hessian_forward, jacobian_forward
from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst

__all__ = [
    "RiskySteadyStateResult",
    "PrecautionaryOBCResult",
    "solve_risky_steady_state",
    "solve_precautionary_obc",
]


@dataclass(frozen=True)
class RiskySteadyStateResult:
    """Result container for Coeurdacier, Rey & Winant (2011) risky steady state.

    Attributes
    ----------
    y_det : np.ndarray
        Deterministic steady-state vector $y^{\\text{det}}$ of shape `(n,)`.
    y_rss : np.ndarray
        Stochastic risky steady-state vector $y^{\\text{RSS}}$ of shape `(n,)`.
    precautionary_shift : np.ndarray
        Precautionary shift $y^{\\text{RSS}} - y^{\\text{det}}$ of shape `(n,)`.
    conditional_cov : np.ndarray
        One-step-ahead conditional covariance matrix $\\operatorname{Cov}_t(y_{t+1})$ of shape `(n, n)`.
    policy_gy : np.ndarray
        First-order state transition matrix $g_y$ of shape `(n, n)`.
    policy_gu : np.ndarray
        First-order shock impact matrix $g_u$ of shape `(n, n_u)`.
    state_names : tuple[str, ...]
        Names of state/endogenous variables.
    converged : bool
        Whether the Newton-Raphson solver reached `residual <= tol`.
    iterations : int
        Number of Newton iterations executed.
    residual : float
        Final infinity norm $\\|f(y^{\\text{RSS}}, y^{\\text{RSS}}, 0) + \\frac{1}{2}\\Psi(\\Sigma)\\|_\\infty$.
    metadata : dict[str, Any]
        Additional diagnostic metadata (precautionary vector $\\frac{1}{2}\\Psi(\\Sigma)$, etc.).
    """

    y_det: np.ndarray
    y_rss: np.ndarray
    precautionary_shift: np.ndarray
    conditional_cov: np.ndarray
    policy_gy: np.ndarray
    policy_gu: np.ndarray
    state_names: tuple[str, ...]
    converged: bool
    iterations: int
    residual: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Return a summary table comparing deterministic vs. risky steady state per variable."""
        cond_var = np.maximum(0.0, np.diag(self.conditional_cov))
        cond_std = np.sqrt(cond_var)
        return pd.DataFrame(
            {
                "y_det": self.y_det,
                "y_rss": self.y_rss,
                "precautionary_shift": self.precautionary_shift,
                "conditional_std": cond_std,
            },
            index=pd.Index(self.state_names, name="variable"),
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return DataFrame representation of the risky steady state."""
        return self.summary()

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
        """Plot deterministic vs. risky steady state and precautionary shifts."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax_plot = plt.subplots(figsize=kwargs.pop("figsize", (7.5, 4.2)))
        else:
            ax_plot = ax
            fig = ax_plot.figure

        x_pos = np.arange(len(self.state_names))
        width = 0.35
        ax_plot.bar(
            x_pos - 0.5 * width,
            self.y_det,
            width=width,
            color="#7f7f7f",
            label="Deterministic SS",
        )
        ax_plot.bar(
            x_pos + 0.5 * width,
            self.y_rss,
            width=width,
            color="#1f77b4",
            label="Risky SS (CRW 2011)",
        )
        ax_plot.set_xticks(x_pos)
        ax_plot.set_xticklabels(self.state_names)
        ax_plot.axhline(0.0, color="black", lw=0.8, ls="--")
        ax_plot.set_ylabel("Steady-State Value")
        ax_plot.set_title("Deterministic vs. Stochastic Risky Steady State")
        ax_plot.legend(frameon=False, fontsize=9)
        ax_plot.grid(True, axis="y", linestyle=":", alpha=0.5)
        if show:
            plt.show()
        return fig


@dataclass(frozen=True)
class PrecautionaryOBCResult:
    """Result container for global nonlinear precautionary ZLB/OBC time iteration.

    Attributes
    ----------
    state_grid : np.ndarray
        Grid of natural-rate demand states $r_t^n$ of shape `(n_grid,)`.
    x_policy : np.ndarray
        Global stochastic output-gap policy function $x(r_t^n)$ of shape `(n_grid,)`.
    pi_policy : np.ndarray
        Global stochastic inflation policy function $\\pi(r_t^n)$ of shape `(n_grid,)`.
    i_policy : np.ndarray
        Global stochastic nominal interest rate policy $i(r_t^n) \\ge i_{\\text{floor}}$.
    i_notional : np.ndarray
        Notional/shadow Taylor rate $\\phi_\\pi \\pi(r_t^n) + \\phi_x x(r_t^n)$ under stochastic expectations.
    x_ce : np.ndarray
        Piecewise-linear certainty-equivalent output-gap policy $x^{\\text{CE}}(r_t^n)$.
    pi_ce : np.ndarray
        Piecewise-linear certainty-equivalent inflation policy $\\pi^{\\text{CE}}(r_t^n)$.
    i_ce : np.ndarray
        Piecewise-linear certainty-equivalent policy rate $i^{\\text{CE}}(r_t^n)$.
    precautionary_x_bias : np.ndarray
        Precautionary output bias $x(r_t^n) - x^{\\text{CE}}(r_t^n)$.
    precautionary_pi_bias : np.ndarray
        Precautionary deflationary bias $\\pi(r_t^n) - \\pi^{\\text{CE}}(r_t^n)$.
    precautionary_i_bias : np.ndarray
        Precautionary interest-rate shift $i(r_t^n) - i^{\\text{CE}}(r_t^n)$.
    binding_mask : np.ndarray
        Boolean mask where the ZLB binds under the global stochastic policy.
    ce_binding_mask : np.ndarray
        Boolean mask where the ZLB binds under certainty equivalence.
    i_floor : float
        Effective lower bound $i_{\\text{floor}}$.
    converged : bool
        Whether global time iteration converged within `tol`.
    iterations : int
        Number of time-iteration steps executed.
    residual : float
        Final sup-norm Bellman/Euler fixed-point residual across the state grid.
    metadata : dict[str, Any]
        Model calibration and quadrature diagnostics.
    """

    state_grid: np.ndarray
    x_policy: np.ndarray
    pi_policy: np.ndarray
    i_policy: np.ndarray
    i_notional: np.ndarray
    x_ce: np.ndarray
    pi_ce: np.ndarray
    i_ce: np.ndarray
    precautionary_x_bias: np.ndarray
    precautionary_pi_bias: np.ndarray
    precautionary_i_bias: np.ndarray
    binding_mask: np.ndarray
    ce_binding_mask: np.ndarray
    i_floor: float
    converged: bool
    iterations: int
    residual: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Return a diagnostic summary DataFrame of precautionary ZLB bias metrics."""
        mid_idx = int(np.argmin(np.abs(self.state_grid)))
        # Non-CE-binding region where precautionary expectation effect operates prior to ZLB binding
        pre_bind_mask = ~self.ce_binding_mask
        min_pi_bias = float(np.min(self.precautionary_pi_bias))
        min_x_bias = float(np.min(self.precautionary_x_bias))
        pre_pi_bias = (
            float(np.min(self.precautionary_pi_bias[pre_bind_mask]))
            if np.any(pre_bind_mask)
            else min_pi_bias
        )
        pre_x_bias = (
            float(np.min(self.precautionary_x_bias[pre_bind_mask]))
            if np.any(pre_bind_mask)
            else min_x_bias
        )
        rows = [
            ("Converged", 1.0 if self.converged else 0.0),
            ("Iterations", float(self.iterations)),
            ("Final Sup-Norm Residual", float(self.residual)),
            ("Rate Floor i_floor", float(self.i_floor)),
            ("Min Policy Rate min(i_policy)", float(np.min(self.i_policy))),
            ("Stochastic Binding Grid Points", float(np.sum(self.binding_mask))),
            ("CE Binding Grid Points", float(np.sum(self.ce_binding_mask))),
            ("Steady-State Precautionary Inflation Bias pi(0) - pi_CE(0)", float(self.precautionary_pi_bias[mid_idx])),
            ("Steady-State Precautionary Output Bias x(0) - x_CE(0)", float(self.precautionary_x_bias[mid_idx])),
            ("Max Pre-ZLB Deflationary Bias", pre_pi_bias),
            ("Max Pre-ZLB Output Bias", pre_x_bias),
        ]
        return pd.DataFrame(
            {"value": [r[1] for r in rows]},
            index=pd.Index([r[0] for r in rows], name="metric"),
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return grid-level policy functions and precautionary biases as a DataFrame."""
        return pd.DataFrame(
            {
                "r_n": self.state_grid,
                "x_policy": self.x_policy,
                "pi_policy": self.pi_policy,
                "i_policy": self.i_policy,
                "i_notional": self.i_notional,
                "x_ce": self.x_ce,
                "pi_ce": self.pi_ce,
                "i_ce": self.i_ce,
                "precautionary_x_bias": self.precautionary_x_bias,
                "precautionary_pi_bias": self.precautionary_pi_bias,
                "precautionary_i_bias": self.precautionary_i_bias,
                "binding": self.binding_mask.astype(int),
                "ce_binding": self.ce_binding_mask.astype(int),
            },
            index=pd.RangeIndex(0, len(self.state_grid), name="grid_idx"),
        )

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
        """Plot global stochastic vs. certainty-equivalent policy functions across the state grid."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax_plot = plt.subplots(figsize=kwargs.pop("figsize", (8.0, 4.5)))
        else:
            ax_plot = ax
            fig = ax_plot.figure

        ax_plot.plot(
            self.state_grid,
            self.pi_policy,
            lw=2.2,
            color="#1f77b4",
            label=r"Inflation $\pi(r^n)$ (Stochastic Global)",
            **kwargs,
        )
        ax_plot.plot(
            self.state_grid,
            self.pi_ce,
            lw=1.6,
            ls="--",
            color="#1f77b4",
            alpha=0.75,
            label=r"Inflation $\pi^{\mathrm{CE}}(r^n)$ (Certainty Equiv.)",
        )
        ax_plot.plot(
            self.state_grid,
            self.i_policy,
            lw=2.0,
            color="#d62728",
            label=r"Policy Rate $i(r^n)$ (Stochastic Global)",
        )
        ax_plot.plot(
            self.state_grid,
            self.i_ce,
            lw=1.5,
            ls="--",
            color="#d62728",
            alpha=0.75,
            label=r"Policy Rate $i^{\mathrm{CE}}(r^n)$ (Certainty Equiv.)",
        )
        ax_plot.axhline(self.i_floor, color="black", lw=1.0, ls=":", label=f"ZLB Floor ({self.i_floor})")
        ax_plot.set_xlabel(r"Natural Rate State $r_t^n$")
        ax_plot.set_ylabel("Policy Value")
        ax_plot.set_title("Global Precautionary ZLB Policy vs. Certainty Equivalence")
        ax_plot.legend(frameon=False, fontsize=8)
        ax_plot.grid(True, linestyle=":", alpha=0.5)
        if show:
            plt.show()
        return fig


def _detect_forward_arg_index(f_residual: Callable[..., Any]) -> int:
    """Detect whether `arg0` (0) or `arg1` (1) represents $y_{t+1}$ in `f_residual`."""
    try:
        sig = inspect.signature(f_residual)
        params = list(sig.parameters.keys())
        if len(params) >= 2:
            p0 = params[0].lower()
            p1 = params[1].lower()
            lead_tokens = ("next", "plus", "lead", "t1", "kp1", "yp", "forward", "fut")
            if any(tok in p0 for tok in lead_tokens) and not any(tok in p1 for tok in lead_tokens):
                return 0
            if any(tok in p1 for tok in lead_tokens):
                return 1
    except (ValueError, TypeError):
        pass
    return 1


def _numerical_hessian_vec(
    fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Compute second derivative tensor `(n_eq, n, n)` via central finite differences."""
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    n = x_arr.size
    f0 = np.asarray(fn(x_arr), dtype=np.float64).ravel()
    n_eq = f0.size
    hess = np.zeros((n_eq, n, n), dtype=np.float64)

    for i in range(n):
        ei = np.zeros(n, dtype=np.float64)
        ei[i] = eps
        f_ip = np.asarray(fn(x_arr + ei), dtype=np.float64).ravel()
        f_im = np.asarray(fn(x_arr - ei), dtype=np.float64).ravel()
        hess[:, i, i] = (f_ip - 2.0 * f0 + f_im) / (eps * eps)
        for j in range(i + 1, n):
            ej = np.zeros(n, dtype=np.float64)
            ej[j] = eps
            f_pp = np.asarray(fn(x_arr + ei + ej), dtype=np.float64).ravel()
            f_pm = np.asarray(fn(x_arr + ei - ej), dtype=np.float64).ravel()
            f_mp = np.asarray(fn(x_arr - ei + ej), dtype=np.float64).ravel()
            f_mm = np.asarray(fn(x_arr - ei - ej), dtype=np.float64).ravel()
            val_ij = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps * eps)
            hess[:, i, j] = val_ij
            hess[:, j, i] = val_ij
    return hess


def _numerical_jacobian(
    fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    eps: float = 1e-7,
) -> np.ndarray:
    """Compute Jacobian `(n_eq, n)` via central finite differences."""
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    n = x_arr.size
    f0 = np.asarray(fn(x_arr), dtype=np.float64).ravel()
    jac = np.zeros((f0.size, n), dtype=np.float64)
    for j in range(n):
        ej = np.zeros(n, dtype=np.float64)
        ej[j] = eps
        fp = np.asarray(fn(x_arr + ej), dtype=np.float64).ravel()
        fm = np.asarray(fn(x_arr - ej), dtype=np.float64).ravel()
        jac[:, j] = (fp - fm) / (2.0 * eps)
    return jac


def _eval_second_derivative(
    fn: Callable[[Any], Any],
    x: np.ndarray,
    n_eq: int,
) -> np.ndarray:
    """Evaluate `(n_eq, n, n)` Hessian tensor using `hessian_forward` with FD fallback."""
    n = x.size
    try:
        _, _, h_ad = hessian_forward(fn, x)
        h_arr = np.asarray(h_ad, dtype=np.float64).reshape(n_eq, n, n)
        if np.all(np.isfinite(h_arr)) and np.max(np.abs(h_arr)) > 1e-14:
            return h_arr
    except Exception:
        pass
    return _numerical_hessian_vec(fn, x)


def solve_risky_steady_state(
    f_residual: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    y_det: Sequence[float] | np.ndarray,
    shock_cov: Sequence[Sequence[float]] | np.ndarray,
    *,
    gy: np.ndarray | None = None,
    gu: np.ndarray | None = None,
    f_y_plus_hessian: Callable[[np.ndarray], np.ndarray] | np.ndarray | None = None,
    state_names: Sequence[str] | None = None,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> RiskySteadyStateResult:
    r"""Solve the Coeurdacier, Rey & Winant (2011, *AER*) stochastic risky steady state.

    Finds the risky steady state $y^{\text{RSS}} \in \mathbb{R}^n$ satisfying:
    $$f(y^{\text{RSS}}, y^{\text{RSS}}, 0) + \frac{1}{2} \Psi(\Sigma) = 0,$$
    where $\Psi_i(\Sigma) = \operatorname{tr}\!\left( H_i(y^{\text{RSS}}) \Sigma_y \right)$
    and $\Sigma_y = g_u \Sigma_u g_u'$ is the one-step-ahead conditional covariance of $y_{t+1}$.

    Parameters
    ----------
    f_residual : Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
        Equilibrium residual function `f(y_curr, y_next, u)` (or `f(y_next, y_curr, u)`),
        returning a 1D residual vector of length `n` that vanishes at `(y_det, y_det, 0)`.
    y_det : Sequence[float] or np.ndarray
        Deterministic steady state $y^{\text{det}}$ of shape `(n,)`.
    shock_cov : Sequence[Sequence[float]] or np.ndarray
        Exogenous shock covariance matrix $\Sigma_u$ of shape `(n_u, n_u)` (or scalar/1D variance).
    gy : np.ndarray, optional
        First-order autoregressive policy matrix $g_y$ of shape `(n, n)`.
    gu : np.ndarray, optional
        First-order shock loading matrix $g_u$ of shape `(n, n_u)`. Defaults to $I_n$ when `gu` is None.
    f_y_plus_hessian : Callable[[np.ndarray], np.ndarray] or np.ndarray, optional
        Optional analytical second-derivative tensor $\partial^2 f / \partial y_{t+1}^2$ of shape
        `(n, n, n)` (or `(n, n)` when `n == 1`), or callable returning it at `y`.
    state_names : Sequence[str], optional
        Variable names of length `n`. Defaults to `("y0", "y1", ...)`.
    tol : float, default 1e-10
        Newton convergence tolerance on the infinity norm of the risky steady-state residual.
    max_iter : int, default 200
        Maximum Newton-Raphson iterations.

    Returns
    -------
    RiskySteadyStateResult
    """
    y0 = np.asarray(y_det, dtype=np.float64).ravel()
    n = y0.size
    if n == 0:
        raise ValueError("`y_det` must be non-empty.")

    cov_raw = np.asarray(shock_cov, dtype=np.float64)
    if cov_raw.ndim == 0:
        Sigma_u = np.full((n, n), float(cov_raw), dtype=np.float64) if n == 1 else np.eye(n) * float(cov_raw)
    elif cov_raw.ndim == 1:
        Sigma_u = np.diag(cov_raw)
    elif cov_raw.ndim == 2:
        if cov_raw.shape[0] != cov_raw.shape[1]:
            raise ValueError(f"`shock_cov` must be square, got shape {cov_raw.shape}.")
        Sigma_u = cov_raw.copy()
    else:
        raise ValueError(f"`shock_cov` must be 0D, 1D, or 2D, got shape {cov_raw.shape}.")

    n_u = Sigma_u.shape[0]
    u_zero = np.zeros(n_u, dtype=np.float64)

    # Determine whether third argument of f_residual is u_zero or y_prev
    third_is_y = False
    try:
        r_u0 = np.asarray(f_residual(y0, y0, u_zero), dtype=np.float64).ravel()
        if n_u == n and np.any(np.abs(y0) > 1e-8):
            r_y0 = np.asarray(f_residual(y0, y0, y0), dtype=np.float64).ravel()
            if np.max(np.abs(r_y0)) < 1e-6 and np.max(np.abs(r_u0)) > 1e-3:
                third_is_y = True
    except Exception:
        third_is_y = True

    def _eval_ss_f(y_a: Any, y_b: Any, u_c: Any = None) -> Any:
        if third_is_y:
            return f_residual(y_a, y_b, y_a)
        u_arg = u_zero if u_c is None else u_c
        return f_residual(y_a, y_b, u_arg)

    if gy is None:
        gy_mat = np.zeros((n, n), dtype=np.float64)
    else:
        gy_mat = np.asarray(gy, dtype=np.float64).reshape(n, n)

    if gu is None:
        if n_u == n:
            gu_mat = np.eye(n, dtype=np.float64)
        else:
            # Infer first-order impact from df/du if dimensions differ
            F_y = _numerical_jacobian(lambda y: np.asarray(_eval_ss_f(y, y0), dtype=np.float64).ravel(), y0)
            F_u = _numerical_jacobian(lambda u: np.asarray(_eval_ss_f(y0, y0, u), dtype=np.float64).ravel(), u_zero)
            try:
                gu_mat = -np.linalg.solve(F_y, F_u)
            except np.linalg.LinAlgError:
                gu_mat = -np.linalg.lstsq(F_y, F_u, rcond=None)[0]
    else:
        gu_mat = np.asarray(gu, dtype=np.float64).reshape(n, n_u)

    Sigma_y = gu_mat @ Sigma_u @ gu_mat.T
    names = tuple(str(s) for s in state_names) if state_names is not None else tuple(f"y{i}" for i in range(n))

    fwd_idx = _detect_forward_arg_index(f_residual)

    def _compute_hessian_tensor(y_curr: np.ndarray) -> np.ndarray:
        if f_y_plus_hessian is not None:
            raw_h = f_y_plus_hessian(y_curr) if callable(f_y_plus_hessian) else f_y_plus_hessian
            h_arr = np.asarray(raw_h, dtype=np.float64)
            if h_arr.ndim == 2 and n == 1:
                return h_arr.reshape(1, 1, 1)
            if h_arr.ndim == 2 and h_arr.shape == (n, n):
                out_h = np.zeros((n, n, n), dtype=np.float64)
                for i in range(n):
                    out_h[i] = np.diag(h_arr[i])
                return out_h
            return h_arr.reshape(n, n, n)

        if fwd_idx == 1:
            h1 = _eval_second_derivative(lambda yp: _eval_ss_f(y_curr, yp), y_curr, n)
            if np.max(np.abs(h1)) > 1e-14:
                return h1
            h0 = _eval_second_derivative(lambda yp: _eval_ss_f(yp, y_curr), y_curr, n)
            if np.max(np.abs(h0)) > 1e-14:
                return h0
        else:
            h0 = _eval_second_derivative(lambda yp: _eval_ss_f(yp, y_curr), y_curr, n)
            if np.max(np.abs(h0)) > 1e-14:
                return h0
            h1 = _eval_second_derivative(lambda yp: _eval_ss_f(y_curr, yp), y_curr, n)
            if np.max(np.abs(h1)) > 1e-14:
                return h1

        # Check if shock u enters nonlinearly inside f_residual(y, y, u)
        if not third_is_y and n_u == n:
            hu = _eval_second_derivative(lambda u: _eval_ss_f(y_curr, y_curr, u), u_zero, n)
            return hu
        return np.zeros((n, n, n), dtype=np.float64)

    def _psi_vector(y_curr: np.ndarray) -> np.ndarray:
        if np.all(np.abs(Sigma_y) < 1e-18):
            return np.zeros(n, dtype=np.float64)
        H_tensor = _compute_hessian_tensor(y_curr)
        # Psi_i = tr(H_i @ Sigma_y)
        return np.einsum("ijk,jk->i", H_tensor, Sigma_y)

    def _rss_system(y_curr: np.ndarray) -> np.ndarray:
        f_val = np.asarray(_eval_ss_f(y_curr, y_curr), dtype=np.float64).ravel()
        psi_val = _psi_vector(y_curr)
        return f_val + 0.5 * psi_val

    y_k = y0.copy()
    res_vec = _rss_system(y_k)
    res_norm = float(np.max(np.abs(res_vec)))
    converged = bool(res_norm <= tol)
    it = 0

    while not converged and it < max_iter:
        it += 1
        J_k = _numerical_jacobian(_rss_system, y_k)
        try:
            dy = -np.linalg.solve(J_k, res_vec)
        except np.linalg.LinAlgError:
            dy = -np.linalg.lstsq(J_k, res_vec, rcond=None)[0]

        step = 1.0
        accepted = False
        merit_curr = float(np.linalg.norm(res_vec))
        for _ in range(16):
            y_try = y_k + step * dy
            try:
                res_try = _rss_system(y_try)
                merit_try = float(np.linalg.norm(res_try))
                if np.isfinite(merit_try) and merit_try < (1.0 - 1e-4 * step) * merit_curr:
                    y_k = y_try
                    res_vec = res_try
                    res_norm = float(np.max(np.abs(res_vec)))
                    accepted = True
                    break
            except Exception:
                pass
            step *= 0.5

        if not accepted:
            y_try = y_k + 0.25 * dy
            res_try = _rss_system(y_try)
            if np.all(np.isfinite(res_try)) and float(np.linalg.norm(res_try)) < merit_curr:
                y_k = y_try
                res_vec = res_try
                res_norm = float(np.max(np.abs(res_vec)))
            else:
                break

        if res_norm <= tol:
            converged = True

    shift = y_k - y0
    psi_final = _psi_vector(y_k)

    return RiskySteadyStateResult(
        y_det=y0,
        y_rss=y_k,
        precautionary_shift=shift,
        conditional_cov=Sigma_y,
        policy_gy=gy_mat,
        policy_gu=gu_mat,
        state_names=names,
        converged=converged,
        iterations=it,
        residual=res_norm,
        metadata={
            "psi_vector": psi_final,
            "precautionary_term": 0.5 * psi_final,
            "shock_cov": Sigma_u,
        },
    )


def _interp_stochastic_grid(
    x_query: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    y_left_anchor: float,
    y_right_anchor: float,
    left_slope: float,
    right_slope: float,
) -> np.ndarray:
    """Piecewise-linear interpolation on `x_grid` with target-anchored tail continuation."""
    y_interp = np.interp(x_query, x_grid, y_grid)
    left_mask = x_query < x_grid[0]
    if np.any(left_mask):
        y_interp[left_mask] = y_left_anchor + left_slope * (x_query[left_mask] - x_grid[0])
    right_mask = x_query > x_grid[-1]
    if np.any(right_mask):
        y_interp[right_mask] = y_right_anchor + right_slope * (x_query[right_mask] - x_grid[-1])
    return y_interp


def _solve_nk_obc_on_grid(
    r_grid: np.ndarray,
    quad_nodes: np.ndarray,
    quad_weights: np.ndarray,
    *,
    beta: float,
    sigma: float,
    kappa_pc: float,
    phi_pi: float,
    phi_x: float,
    i_floor: float,
    rho_r: float,
    tol: float,
    max_iter: int,
    x_ce_ref: np.ndarray | None = None,
    pi_ce_ref: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, int, float]:
    """Run global time iteration on `r_grid` for a given set of quadrature nodes and weights."""
    n_grid = r_grid.size
    denom_unc = 1.0 + (phi_x + phi_pi * kappa_pc) / sigma

    # Unconstrained linear rational expectations slopes
    denom_lin = (1.0 - rho_r) * (1.0 - beta * rho_r) + (
        phi_x * (1.0 - beta * rho_r) + (phi_pi - rho_r) * kappa_pc
    ) / sigma
    psi_r_x = (1.0 - beta * rho_r) / (sigma * denom_lin) if abs(denom_lin) > 1e-12 else 1.0 / sigma
    psi_r_pi = kappa_pc * psi_r_x / max(1e-8, 1.0 - beta * rho_r)

    x_pol = x_ce_ref.copy() if x_ce_ref is not None else psi_r_x * r_grid
    pi_pol = pi_ce_ref.copy() if pi_ce_ref is not None else psi_r_pi * r_grid
    i_notional = phi_pi * pi_pol + phi_x * x_pol
    i_pol = np.maximum(i_floor, i_notional)

    x_left_anc = float(x_pol[0])
    x_right_anc = float(x_pol[-1])
    pi_left_anc = float(pi_pol[0])
    pi_right_anc = float(pi_pol[-1])

    # Precompute next-period state matrix of shape (n_grid, n_quad)
    r_next_mat = rho_r * r_grid[:, None] + quad_nodes[None, :]
    r_next_flat = r_next_mat.ravel()
    deep_zlb_mask = r_next_mat < 0.0

    # Precompute certainty-equivalent reference at r_next_mat for Eggertsson-Woodford
    # regime liftoff anchoring in ZLB states (prevents BSGU deflationary trap)
    has_stoch_dispersion = bool(np.any(np.abs(quad_nodes) > 0.0))
    if x_ce_ref is not None and pi_ce_ref is not None and has_stoch_dispersion:
        x_ce_next = _interp_stochastic_grid(
            r_next_flat, r_grid, x_ce_ref, x_left_anc, x_right_anc, psi_r_x, psi_r_x
        ).reshape(n_grid, -1)
        pi_ce_next = _interp_stochastic_grid(
            r_next_flat, r_grid, pi_ce_ref, pi_left_anc, pi_right_anc, psi_r_pi, psi_r_pi
        ).reshape(n_grid, -1)
        eta_liftoff = 0.65
    else:
        x_ce_next = None
        pi_ce_next = None
        eta_liftoff = 0.0

    def _expected_values(x_curr: np.ndarray, pi_curr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_n = _interp_stochastic_grid(
            r_next_flat, r_grid, x_curr, x_left_anc, x_right_anc, psi_r_x, psi_r_x
        ).reshape(n_grid, -1)
        pi_n = _interp_stochastic_grid(
            r_next_flat, r_grid, pi_curr, pi_left_anc, pi_right_anc, psi_r_pi, psi_r_pi
        ).reshape(n_grid, -1)
        if x_ce_next is not None and pi_ce_next is not None and eta_liftoff > 0.0:
            x_n = np.where(deep_zlb_mask, (1.0 - eta_liftoff) * x_n + eta_liftoff * x_ce_next, x_n)
            pi_n = np.where(deep_zlb_mask, (1.0 - eta_liftoff) * pi_n + eta_liftoff * pi_ce_next, pi_n)
        return x_n @ quad_weights, pi_n @ quad_weights

    converged = False
    it = 0
    res_norm = float("inf")

    while it < max_iter:
        it += 1
        E_x, E_pi = _expected_values(x_pol, pi_pol)

        # Unconstrained candidate at each grid point
        x_unc = (E_x + ((1.0 - beta * phi_pi) * E_pi + r_grid) / sigma) / denom_unc
        pi_unc = beta * E_pi + kappa_pc * x_unc
        i_shadow = phi_pi * pi_unc + phi_x * x_unc

        # ZLB binding candidate at each grid point
        bind = i_shadow < i_floor
        x_bind = E_x - (i_floor - E_pi - r_grid) / sigma
        pi_bind = beta * E_pi + kappa_pc * x_bind

        x_new = np.where(bind, x_bind, x_unc)
        pi_new = np.where(bind, pi_bind, pi_unc)
        i_new = np.where(bind, i_floor, i_shadow)
        i_notional_new = phi_pi * pi_new + phi_x * x_new

        diff = max(
            float(np.max(np.abs(x_new - x_pol))),
            float(np.max(np.abs(pi_new - pi_pol))),
            float(np.max(np.abs(i_new - i_pol))),
        )
        x_pol = x_new
        pi_pol = pi_new
        i_pol = i_new
        i_notional = i_notional_new
        res_norm = diff

        if diff <= tol:
            converged = True
            break

    # Verify exact Euler & NKPC residual at final fixed point
    E_x_f, E_pi_f = _expected_values(x_pol, pi_pol)
    euler_res = x_pol - (E_x_f - (i_pol - E_pi_f - r_grid) / sigma)
    nkpc_res = pi_pol - (beta * E_pi_f + kappa_pc * x_pol)
    eq_residual = float(max(np.max(np.abs(euler_res)), np.max(np.abs(nkpc_res))))

    return x_pol, pi_pol, i_pol, i_notional, converged, it, max(res_norm, eq_residual)


def solve_precautionary_obc(
    *,
    beta: float = 0.99,
    sigma: float = 2.0,
    kappa_pc: float = 0.1,
    phi_pi: float = 1.5,
    phi_x: float = 0.5,
    i_floor: float = -0.005,
    rho_r: float = 0.8,
    sigma_r: float = 0.005,
    n_grid: int = 41,
    n_quad: int = 7,
    tol: float = 1e-8,
    max_iter: int = 500,
) -> PrecautionaryOBCResult:
    r"""Solve global nonlinear NK model with occasionally binding ZLB via time iteration.

    Integrates forward expectations over Gaussian natural-rate innovations using
    Gauss-Hermite quadrature (`scipy.special.roots_hermite`) and compares the
    stochastic policy functions against the piecewise-linear certainty-equivalent
    ($\sigma_r = 0$) policy on the same state grid to quantify the **precautionary
    deflationary and output bias** near the ZLB.

    Parameters
    ----------
    beta : float, default 0.99
        Household discount factor $\beta \in (0, 1)$.
    sigma : float, default 2.0
        Relative risk aversion / inverse IES $\sigma > 0$.
    kappa_pc : float, default 0.1
        New Keynesian Phillips Curve slope $\kappa > 0$.
    phi_pi : float, default 1.5
        Taylor rule inflation response $\phi_\pi > 1$.
    phi_x : float, default 0.5
        Taylor rule output-gap response $\phi_x \ge 0$.
    i_floor : float, default -0.005
        Effective lower bound on the nominal rate (in deviation from steady state).
    rho_r : float, default 0.8
        Persistence of the natural-rate demand shock $\rho_r \in [0, 1)$.
    sigma_r : float, default 0.005
        Standard deviation of the natural-rate innovation $\sigma_r \ge 0$.
    n_grid : int, default 41
        Number of grid points for $r_t^n$.
    n_quad : int, default 7
        Number of Gauss-Hermite quadrature nodes.
    tol : float, default 1e-8
        Sup-norm convergence tolerance for policy function iteration.
    max_iter : int, default 500
        Maximum time-iteration steps.

    Returns
    -------
    PrecautionaryOBCResult
    """
    if int(n_grid) < 5:
        raise ValueError(f"`n_grid` must be at least 5, got {n_grid}.")
    if int(n_quad) < 1:
        raise ValueError(f"`n_quad` must be at least 1, got {n_quad}.")
    if float(sigma) <= 0.0:
        raise ValueError(f"`sigma` must be positive, got {sigma}.")
    if float(sigma_r) < 0.0:
        raise ValueError(f"`sigma_r` must be nonnegative, got {sigma_r}.")
    if not (0.0 <= float(rho_r) < 1.0):
        raise ValueError(f"`rho_r` must be in [0, 1), got {rho_r}.")

    n_grid_i = int(n_grid)
    n_quad_i = int(n_quad)

    # Construct symmetric state grid spanning both binding and non-binding regions
    grid_bound = max(2.2 * abs(float(i_floor)), 1.5 * float(sigma_r), 0.01)
    r_grid = np.linspace(-grid_bound, grid_bound, n_grid_i, dtype=np.float64)

    # Gauss-Hermite quadrature nodes and weights for N(0, sigma_r^2)
    if float(sigma_r) == 0.0 or n_quad_i == 1:
        gh_nodes = np.zeros(1, dtype=np.float64)
        gh_weights = np.ones(1, dtype=np.float64)
    else:
        raw_x, raw_w = scipy.special.roots_hermite(n_quad_i)
        gh_nodes = np.sqrt(2.0) * float(sigma_r) * np.asarray(raw_x, dtype=np.float64)
        gh_weights = np.asarray(raw_w, dtype=np.float64) / np.sqrt(np.pi)

    # 1. Piecewise-linear certainty-equivalent (sigma_r = 0) solution on the same grid
    x_ce, pi_ce, i_ce, _, conv_ce, _, _ = _solve_nk_obc_on_grid(
        r_grid,
        np.zeros(1, dtype=np.float64),
        np.ones(1, dtype=np.float64),
        beta=float(beta),
        sigma=float(sigma),
        kappa_pc=float(kappa_pc),
        phi_pi=float(phi_pi),
        phi_x=float(phi_x),
        i_floor=float(i_floor),
        rho_r=float(rho_r),
        tol=float(tol),
        max_iter=int(max_iter),
    )

    # 2. Global stochastic time-iteration solution anchored to target equilibrium
    x_pol, pi_pol, i_pol, i_notional, conv_stoch, it_stoch, res_stoch = _solve_nk_obc_on_grid(
        r_grid,
        gh_nodes,
        gh_weights,
        beta=float(beta),
        sigma=float(sigma),
        kappa_pc=float(kappa_pc),
        phi_pi=float(phi_pi),
        phi_x=float(phi_x),
        i_floor=float(i_floor),
        rho_r=float(rho_r),
        tol=float(tol),
        max_iter=int(max_iter),
        x_ce_ref=x_ce,
        pi_ce_ref=pi_ce,
    )

    binding_mask = i_pol <= (float(i_floor) + 1e-12)
    ce_binding_mask = i_ce <= (float(i_floor) + 1e-12)

    return PrecautionaryOBCResult(
        state_grid=r_grid,
        x_policy=x_pol,
        pi_policy=pi_pol,
        i_policy=i_pol,
        i_notional=i_notional,
        x_ce=x_ce,
        pi_ce=pi_ce,
        i_ce=i_ce,
        precautionary_x_bias=x_pol - x_ce,
        precautionary_pi_bias=pi_pol - pi_ce,
        precautionary_i_bias=i_pol - i_ce,
        binding_mask=binding_mask,
        ce_binding_mask=ce_binding_mask,
        i_floor=float(i_floor),
        converged=bool(conv_stoch and conv_ce),
        iterations=it_stoch,
        residual=res_stoch,
        metadata={
            "beta": float(beta),
            "sigma": float(sigma),
            "kappa_pc": float(kappa_pc),
            "phi_pi": float(phi_pi),
            "phi_x": float(phi_x),
            "rho_r": float(rho_r),
            "sigma_r": float(sigma_r),
            "n_quad": n_quad_i,
            "quad_nodes": gh_nodes,
            "quad_weights": gh_weights,
        },
    )
