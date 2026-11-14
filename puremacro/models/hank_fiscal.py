"""Financed fiscal policy closures for the one-asset Sequence-Space HANK model.

Builds explicit, audited fiscal-monetary general-equilibrium closures on top of
the household block and sequence-space primitives in
:mod:`puremacro.models.hank_sequence_space` without modifying that module:

* ``closure="balanced_budget"`` — government debt stays constant at
  ``B_t = B_ss`` and total taxes finance interest plus spending date by date,
  ``T_t = r_t B_ss + G_t``.
* ``closure="debt_rule"`` — government debt evolves endogenously via
  ``B_t = (1 + r_t) B_{t-1} + G_t - T_t`` under the feedback rule
  ``T_t = T_ss + phi_B (B_{t-1} - B_ss) + phi_G dG_t`` subject to the
  stationary debt-stabilization condition ``r_ss < phi_B < 2 + r_ss``.

Tax adjustments ``dT_t = T_t - T_ss`` enter household disposable income either
proportionally to labour earnings (``tax_type="proportional"``) or as uniform
lump-sum transfers/taxes (``tax_type="lump_sum"``).  At every period ``t``,
aggregate household accounting plus the government budget yields the exact
Walras's Law identity

.. math::

    (Y_t - C_t - G_t) = (A_t - B_t) - (1 + r_t)(A_{t-1} - B_{t-1}),

so when the asset grid does not censor upper-tail savings, clearing goods and
satisfying the government budget simultaneously clears the asset market
``A_t = B_t``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import warnings
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "FiscalHANKResult",
    "solve_hank_fiscal",
    "compare_fiscal_closures",
]

_DEFAULT_N_A = 100
_DEFAULT_A_MAX = 100.0
_DEFAULT_BETA = 0.985
_DEFAULT_GAMMA = 1.0
_DEFAULT_R_SS = 0.01
_DEFAULT_PHI_PI = 1.5
_DEFAULT_KAPPA = 0.1
_SS_EGM_TOL = 1e-12
_SS_FISCAL_TOL = 1e-13
_FD_STEP = 1e-4

_CLOSURE_ALIASES = {
    "balanced_budget": "balanced_budget",
    "balanced-budget": "balanced_budget",
    "bb": "balanced_budget",
    "debt_rule": "debt_rule",
    "debt-rule": "debt_rule",
    "debt": "debt_rule",
    "fiscal_rule": "debt_rule",
}

_TAX_ALIASES = {
    "proportional": "proportional",
    "labor": "proportional",
    "labour": "proportional",
    "income": "proportional",
    "proportional_labor": "proportional",
    "lump_sum": "lump_sum",
    "lump-sum": "lump_sum",
    "lumpsum": "lump_sum",
}


class _InfeasibleFiscalTrialError(ValueError):
    """Raised internally when a trial path leaves positive consumption/return support."""


@dataclass
class _FiscalHousehold:
    """Steady-state household parameters and policy arrays used by fiscal solves."""

    a_grid: np.ndarray
    s_grid: np.ndarray
    pi_s: np.ndarray
    beta: float
    gamma: float
    r_ss: float
    w_ss: float
    V_ss: np.ndarray
    c_ss: np.ndarray
    a_ss: np.ndarray
    D_ss: np.ndarray
    Lambda: np.ndarray
    B_ss: float
    N_ss: float
    G_ss: float
    T_ss: float
    Y_ss: float
    C_ss: float
    ss_converged: bool


@dataclass(frozen=True)
class FiscalHANKResult:
    """Results and market-clearing audit for a financed HANK fiscal transition.

    Attributes
    ----------
    Y, C, G, T, B, A, r, w, tau, pi, r_ex_ante : np.ndarray
        Equilibrium level paths over ``t = 0, ..., horizon - 1``. ``r`` is the
        realized return entering ``(1 + r_t) B_{t-1}`` and household cash-on-hand
        (``r[0] = r_ss``, ``r[t] = r_ex_ante[t - 1]`` for ``t >= 1``).
    dY, dC, dG, dT, dB : np.ndarray
        Level deviations from steady state.
    linear_dY, linear_dC : np.ndarray
        First-order sequence-space solution using the exact Fake News Jacobians.
    residuals : np.ndarray
        Goods-market residual sequence ``Y_t - C_t - G_t``.
    goods_market_residual : float
        Sup-norm goods-market residual ``||Y_t - C_t - G_t||_inf``.
    gov_budget_residual : float
        Sup-norm government flow budget residual
        ``||B_t - ((1 + r_t) B_{t-1} + G_t - T_t)||_inf``.
    gov_intertemporal_residual : float
        Absolute present-value intertemporal government budget residual
        ``|B_{-1} - (sum_t Q_t (T_t - G_t) + Q_{T-1} B_{T-1})|``.
    asset_market_residual : float
        Sup-norm Walras/asset-market clearing residual ``||A_t - B_t||_inf``.
    terminal_debt_gap : float
        Terminal debt deviation ``|B_{T-1} - B_ss|``.
    terminal_discounted_debt_gap : float
        Present-value terminal excess debt ``|Q_{T-1} (B_{T-1} - B_ss)|``.
    impact_multiplier : float
        Impact fiscal multiplier ``dY_0 / dG_0``.
    cumulative_multiplier : float
        Cumulative fiscal multiplier ``sum_t dY_t / sum_t dG_t``.
    present_value_multiplier : float
        Present-value multiplier ``sum_t Q_t dY_t / sum_t Q_t dG_t``.
    closure : str
        ``'balanced_budget'`` or ``'debt_rule'``.
    tax_type : str
        ``'proportional'`` or ``'lump_sum'``.
    horizon : int
        Simulation horizon ``T``.
    phi_B : float | None
        Debt feedback coefficient ``phi_B`` (``None`` for ``balanced_budget``).
    phi_G : float | None
        Contemporaneous spending tax coefficient ``phi_G`` (``None`` for ``balanced_budget``).
    converged : bool
        ``True`` iff the nonlinear solver converged, all market residuals are
        below ``tol``, and the fiscal rule is stationary.
    iterations : int
        Number of accepted Broyden updates.
    tol : float
        Convergence tolerance.
    norm_history : list[float]
        ``||Y - C - G||_inf`` at the initial guess and after each accepted step.
    steady_state : dict[str, float]
        Steady-state levels and grid diagnostics.
    steady_state_model : Any
        Underlying :class:`~puremacro.models.hank_sequence_space.SequenceSpaceHANKResult`.
    solver_diagnostics : Any
        :class:`~puremacro.solvers.diagnostics.SolverDiagnosticRecord` separating
        numerical convergence, economic accuracy, and support validity.
    """

    Y: np.ndarray
    C: np.ndarray
    G: np.ndarray
    T: np.ndarray
    B: np.ndarray
    A: np.ndarray
    r: np.ndarray
    w: np.ndarray
    tau: np.ndarray
    pi: np.ndarray
    r_ex_ante: np.ndarray
    dY: np.ndarray
    dC: np.ndarray
    dG: np.ndarray
    dT: np.ndarray
    dB: np.ndarray
    linear_dY: np.ndarray
    linear_dC: np.ndarray
    residuals: np.ndarray
    goods_market_residual: float
    gov_budget_residual: float
    gov_intertemporal_residual: float
    asset_market_residual: float
    terminal_debt_gap: float
    terminal_discounted_debt_gap: float
    impact_multiplier: float
    cumulative_multiplier: float
    present_value_multiplier: float
    closure: str
    tax_type: str
    horizon: int
    phi_B: float | None
    phi_G: float | None
    converged: bool
    iterations: int
    tol: float
    norm_history: list[float]
    steady_state: dict[str, float] = field(default_factory=dict)
    steady_state_model: Any = None
    solver_diagnostics: Any = None

    @property
    def convergence_status(self) -> Any:
        """Numerical :class:`~puremacro.solvers.diagnostics.ConvergenceStatus`."""
        return self.solver_diagnostics.convergence

    @property
    def accuracy_status(self) -> Any:
        """Economic :class:`~puremacro.solvers.diagnostics.AccuracyStatus`."""
        return self.solver_diagnostics.accuracy

    @property
    def support_status(self) -> Any:
        """Domain :class:`~puremacro.solvers.diagnostics.SupportStatus`."""
        return self.solver_diagnostics.support

    def summary(self) -> str:
        """Return a concise summary of paths, multipliers, and clearing residuals."""
        status = (
            self.solver_diagnostics.overall_status
            if self.solver_diagnostics is not None
            else ("PASS" if self.converged else "FAILED")
        )
        rule_desc = (
            "B_t = B_ss (T_t = r_t B_ss + G_t)"
            if self.closure == "balanced_budget"
            else f"phi_B={self.phi_B:.4f}, phi_G={self.phi_G:.4f}"
        )
        lines = [
            "Financed Sequence-Space HANK Fiscal Transition",
            "=" * 72,
            f"Closure / Tax Type              : {self.closure} / {self.tax_type}",
            f"Fiscal Rule                     : {rule_desc}",
            f"Horizon T                       : {self.horizon} quarters",
            f"Overall Diagnostic Status       : {status} (converged={self.converged}, iters={self.iterations})",
            "-" * 72,
            f"Goods Residual ||Y-C-G||_inf    : {self.goods_market_residual:.3e} (tol {self.tol:.1e})",
            f"Gov Budget Residual (flow)      : {self.gov_budget_residual:.3e}",
            f"Gov Intertemporal Residual (PV) : {self.gov_intertemporal_residual:.3e}",
            f"Asset Residual ||A-B||_inf      : {self.asset_market_residual:.3e}",
            f"Terminal Debt Gap |B_T-1-B_ss|  : {self.terminal_debt_gap:.3e}",
            "-" * 72,
            f"Impact Fiscal Multiplier        : {self.impact_multiplier:.4f}",
            f"Cumulative Fiscal Multiplier    : {self.cumulative_multiplier:.4f}",
            f"Present-Value Multiplier        : {self.present_value_multiplier:.4f}",
            f"Peak Output Deviation (dY)      : {float(self.dY[int(np.argmax(np.abs(self.dY)))]):+.6f}",
            f"Peak Consumption Dev. (dC)      : {float(self.dC[int(np.argmax(np.abs(self.dC)))]):+.6f}",
            f"Peak Debt Deviation (dB)        : {float(self.dB[int(np.argmax(np.abs(self.dB)))]):+.6f}",
            "=" * 72,
        ]
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """Return equilibrium paths and period-by-period clearing residuals."""
        b_ss = float(self.steady_state["B_ss"])
        b_prev = np.concatenate(([b_ss], self.B[:-1]))
        gov_res = self.B - ((1.0 + self.r) * b_prev + self.G - self.T)
        return pd.DataFrame(
            {
                "Y": self.Y,
                "C": self.C,
                "G": self.G,
                "T": self.T,
                "B": self.B,
                "A": self.A,
                "r": self.r,
                "r_ex_ante": self.r_ex_ante,
                "w": self.w,
                "tau": self.tau,
                "pi": self.pi,
                "dY": self.dY,
                "dC": self.dC,
                "dG": self.dG,
                "dT": self.dT,
                "dB": self.dB,
                "Goods_Residual": self.residuals,
                "Gov_Budget_Residual": gov_res,
                "Asset_Residual": self.A - self.B,
            },
            index=[f"t={t}" for t in range(self.horizon)],
        )

    def to_markdown(self, **kwargs: Any) -> str:
        """Render transition paths as a Markdown table."""
        from puremacro.reports import _df_to_markdown

        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Render transition paths as a LaTeX tabular environment."""
        from puremacro.reports import _df_to_latex

        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Render transition paths as Typst table markup."""
        from puremacro.reports import _df_to_typst

        return _df_to_typst(self.to_frame(), **kwargs)

    def plot(self, style: str = "publication", figsize: tuple[float, float] = (11.0, 8.0)) -> Any:
        """Plot 4-panel fiscal transition: output/consumption, fiscal flows, debt/assets, rates."""
        import matplotlib.pyplot as plt

        from puremacro.plot import _palette

        colors = _palette(3) if style == "grayscale" else ["#1f77b4", "#d62728", "#2ca02c"]
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        t_grid = np.arange(self.horizon)

        axes[0, 0].plot(t_grid, self.dY, color=colors[0], lw=2.0, label=r"Output $dY_t$")
        axes[0, 0].plot(t_grid, self.dC, color=colors[1], lw=1.8, linestyle="--", label=r"Consumption $dC_t$")
        axes[0, 0].axhline(0.0, color="gray", linestyle=":", lw=0.8)
        axes[0, 0].set_title("Output and Consumption", fontsize=11, fontweight="bold")
        axes[0, 0].set_ylabel("Deviation from SS", fontsize=9)
        axes[0, 0].legend(loc="best", frameon=False, fontsize=9)

        axes[0, 1].plot(t_grid, self.dG, color=colors[0], lw=2.0, label=r"Spending $dG_t$")
        axes[0, 1].plot(t_grid, self.dT, color=colors[1], lw=1.8, linestyle="--", label=r"Taxes $dT_t$")
        axes[0, 1].axhline(0.0, color="gray", linestyle=":", lw=0.8)
        axes[0, 1].set_title(f"Fiscal Flows ({self.tax_type})", fontsize=11, fontweight="bold")
        axes[0, 1].set_ylabel("Deviation from SS", fontsize=9)
        axes[0, 1].legend(loc="best", frameon=False, fontsize=9)

        axes[1, 0].plot(t_grid, self.dB, color=colors[0], lw=2.2, label=r"Debt $B_t - B_{ss}$")
        axes[1, 0].plot(
            t_grid,
            self.A - float(self.steady_state["B_ss"]),
            color=colors[2],
            lw=1.6,
            linestyle=":",
            label=r"Assets $A_t - B_{ss}$",
        )
        axes[1, 0].axhline(0.0, color="gray", linestyle=":", lw=0.8)
        axes[1, 0].set_title("Debt and Asset-Market Clearing", fontsize=11, fontweight="bold")
        axes[1, 0].set_xlabel("Horizon (quarters)", fontsize=9)
        axes[1, 0].set_ylabel("Deviation from SS", fontsize=9)
        axes[1, 0].legend(loc="best", frameon=False, fontsize=9)

        axes[1, 1].plot(
            t_grid,
            self.r_ex_ante - float(self.steady_state["r_ss"]),
            color=colors[0],
            lw=2.0,
            label=r"Ex-ante real rate $dr_t$",
        )
        axes[1, 1].plot(t_grid, self.pi, color=colors[1], lw=1.8, linestyle="--", label=r"Inflation $\pi_t$")
        axes[1, 1].axhline(0.0, color="gray", linestyle=":", lw=0.8)
        axes[1, 1].set_title("Real Rate and Inflation", fontsize=11, fontweight="bold")
        axes[1, 1].set_xlabel("Horizon (quarters)", fontsize=9)
        axes[1, 1].set_ylabel("Quarterly rate deviation", fontsize=9)
        axes[1, 1].legend(loc="best", frameon=False, fontsize=9)

        for ax in axes.ravel():
            ax.grid(True, linestyle=":", alpha=0.5)

        fig.suptitle(
            rf"Financed HANK ({self.closure}, {self.tax_type}): "
            rf"$\|Y-C-G\|_\infty={self.goods_market_residual:.2e}$, "
            rf"$\|A-B\|_\infty={self.asset_market_residual:.2e}$",
            fontsize=11,
        )
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Input validation and steady-state helpers
# ---------------------------------------------------------------------------


def _normalize_closure(closure: str) -> str:
    if not isinstance(closure, str):
        raise TypeError(f"closure must be a string, got {type(closure).__name__}")
    key = _CLOSURE_ALIASES.get(closure.lower().strip())
    if key is None:
        raise ValueError(
            f"Unknown closure: {closure!r}. Choose 'balanced_budget' or 'debt_rule'."
        )
    return key


def _normalize_tax_type(tax_type: str) -> str:
    if not isinstance(tax_type, str):
        raise TypeError(f"tax_type must be a string, got {type(tax_type).__name__}")
    key = _TAX_ALIASES.get(tax_type.lower().strip())
    if key is None:
        raise ValueError(
            f"Unknown tax_type: {tax_type!r}. Choose 'proportional' or 'lump_sum'."
        )
    return key


def _validate_positive_int(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    iv = int(value)
    if iv < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {iv}")
    return iv


def _validate_finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, complex)) or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a finite real float, got {value!r}")
    fv = float(value)
    if not np.isfinite(fv):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return fv


def _coerce_shock_array(
    seq: Sequence[float] | np.ndarray | None,
    name: str,
) -> np.ndarray | None:
    if seq is None:
        return None
    if np.iscomplexobj(seq):
        raise ValueError(f"{name} must be real-valued")
    arr = np.asarray(seq, dtype=float)
    if arr.ndim > 1:
        raise ValueError(f"{name} must be 1-dimensional, got shape {arr.shape}")
    arr = arr.ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite numbers")
    return arr


def _pad_or_copy(arr: np.ndarray, horizon: int) -> np.ndarray:
    out = np.zeros(horizon, dtype=float)
    out[: len(arr)] = arr
    return out


def _solve_fiscal_steady_state(
    *,
    beta: float,
    gamma: float,
    r_ss: float,
    phi_pi: float,
    kappa: float,
    n_a: int,
    a_max: float,
    g_ss: float,
    T_jac: int = 40,
) -> tuple[_FiscalHousehold, Any]:
    """Solve stationary Aiyagari household block with steady spending g_ss >= 0."""
    from puremacro.models.hank_sequence_space import (
        SequenceSpaceHANKResult,
        _W_SS,
        _asset_grid,
        _build_transition_matrix,
        _decile_weights,
        _egm_fixed_point,
        _ge_matrices,
        _income_process,
        _local_mpc,
        _stationary_distribution,
        _stationary_labour,
    )

    if not (0.0 < beta < 1.0):
        raise ValueError(f"beta must be in (0, 1), got {beta}")
    if gamma <= 0.0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    if r_ss <= -1.0:
        raise ValueError(f"r_ss must be > -1, got {r_ss}")
    if n_a < 10:
        raise ValueError(f"n_a must be >= 10, got {n_a}")
    if a_max <= 0.0:
        raise ValueError(f"a_max must be positive, got {a_max}")

    a_grid = _asset_grid(n_a, a_max)
    s_grid, pi_s = _income_process(2)
    N_ss = _stationary_labour(s_grid, pi_s)
    w_ss = float(_W_SS)
    Y_ss = w_ss * N_ss
    if g_ss < 0.0 or g_ss >= Y_ss:
        raise ValueError(f"g_ss must satisfy 0 <= g_ss < Y_ss ({Y_ss:.4f}), got {g_ss}")

    init_w_net = w_ss - g_ss / N_ss
    c = np.maximum(r_ss * a_grid[:, None] + init_w_net * s_grid[None, :], 1e-8)
    V = (1.0 + r_ss) * c ** (-gamma)
    a = np.zeros_like(c)
    Lam = np.zeros((c.size, c.size), dtype=float)
    D = np.zeros_like(c)
    converged = False

    def assets_at(B: float) -> float:
        nonlocal V, c, a, Lam, D, converged
        w_net = w_ss - (r_ss * B + g_ss) / N_ss
        if w_net <= 0.0:
            raise ValueError(
                f"Steady-state tax bill (r_ss*B + g_ss)/N_ss exceeds wage (B={B:.3f}, g_ss={g_ss:.3f})"
            )
        V, c, a, _, converged = _egm_fixed_point(
            V, c, a_grid, s_grid, pi_s, beta, gamma, r_ss, w_net
        )
        Lam = _build_transition_matrix(a, a_grid, pi_s)
        D = _stationary_distribution(Lam).reshape(a.shape)
        return float(np.sum(D.sum(axis=1) * a_grid))

    B_prev = 0.0
    f_prev = assets_at(B_prev) - B_prev
    B_cur = B_prev + f_prev
    f_cur = assets_at(B_cur) - B_cur
    fiscal_converged = abs(f_cur) < _SS_FISCAL_TOL * max(1.0, abs(B_cur))
    for _ in range(80):
        if fiscal_converged:
            break
        denom = f_cur - f_prev
        step = -f_cur * (B_cur - B_prev) / denom if abs(denom) > 1e-15 else f_cur
        B_prev, f_prev = B_cur, f_cur
        B_cur = B_cur + step
        f_cur = assets_at(B_cur) - B_cur
        fiscal_converged = abs(f_cur) < _SS_FISCAL_TOL * max(1.0, abs(B_cur))

    if not converged:
        warnings.warn(
            f"Steady-state EGM did not converge to {_SS_EGM_TOL:.0e} "
            f"(beta*(1+r_ss)={beta * (1.0 + r_ss):.6f}).",
            RuntimeWarning,
            stacklevel=3,
        )
    if not fiscal_converged:
        warnings.warn(
            f"Steady-state asset fixed point B = A_ss did not converge (|A_ss - B| = {abs(f_cur):.2e}).",
            RuntimeWarning,
            stacklevel=3,
        )

    T_ss = r_ss * B_cur + g_ss
    C_ss = float(np.sum(D * c))
    D_a = D.sum(axis=1)
    mpc_grid = _local_mpc(c, a_grid, r_ss)
    agg_mpc = float(np.sum(mpc_grid * D))
    W = _decile_weights(D_a)
    decile_mass = W @ D_a
    decile_mpc = (W @ (mpc_grid * D).sum(axis=1)) / np.where(decile_mass > 0, decile_mass, 1.0)
    mpc_series = pd.Series({f"Decile {d + 1}": float(decile_mpc[d]) for d in range(10)})

    hh = _FiscalHousehold(
        a_grid=a_grid,
        s_grid=s_grid,
        pi_s=pi_s,
        beta=beta,
        gamma=gamma,
        r_ss=r_ss,
        w_ss=w_ss,
        V_ss=V,
        c_ss=c,
        a_ss=a,
        D_ss=D,
        Lambda=Lam,
        B_ss=B_cur,
        N_ss=N_ss,
        G_ss=g_ss,
        T_ss=T_ss,
        Y_ss=Y_ss,
        C_ss=C_ss,
        ss_converged=bool(converged and fiscal_converged),
    )

    T_ss_len = max(1, int(T_jac))
    y_ss_vec = (w_ss - T_ss / N_ss) * s_grid
    J_r_real = _fake_news_fiscal(hh, "r_pure", T_ss_len + 1, y_ss_vec)
    J_w_real = _fake_news_fiscal(hh, "w_net", T_ss_len, y_ss_vec)
    J_C_r = np.ascontiguousarray(J_r_real[:T_ss_len, 1 : T_ss_len + 1])
    J_C_Y = np.ascontiguousarray(J_w_real * (w_ss / max(C_ss, 1e-12)))
    K_pi, M_r_Y = _ge_matrices(T_ss_len, beta, kappa, phi_pi)
    LHS = np.eye(T_ss_len) - J_C_Y - J_C_r @ M_r_Y
    shock_zero = np.zeros(T_ss_len)
    dY_ss = np.linalg.solve(LHS, shock_zero)

    ss_result = SequenceSpaceHANKResult(
        irf_output=dY_ss,
        irf_consumption=dY_ss.copy(),
        irf_inflation=K_pi @ dY_ss,
        irf_rate=M_r_Y @ dY_ss,
        jacobian_c_r=J_C_r,
        jacobian_c_y=J_C_Y,
        steady_state_mpc=agg_mpc,
        mpc_distribution=mpc_series,
        asset_grid=a_grid,
        steady_state_wealth_dist=D_a,
        policy_c=c,
        policy_a=a,
        distribution=D,
        trans_matrix=Lam,
        beta=beta,
        gamma=gamma,
        r_ss=r_ss,
        phi_pi=phi_pi,
        kappa=kappa,
        steady_state_consumption=C_ss,
        w_ss=w_ss,
        government_debt=B_cur,
        tax_rate=T_ss / Y_ss,
        ss_converged=hh.ss_converged,
    )
    return hh, ss_result


def _resolve_fiscal_steady_state(
    ss_model: Any,
    horizon: int,
    *,
    g_ss: float,
    beta: float | None,
    gamma: float | None,
    r_ss: float | None,
    phi_pi: float | None,
    kappa: float | None,
    n_a: int | None,
    a_max: float | None,
    extra_kwargs: Mapping[str, Any],
) -> tuple[_FiscalHousehold, Any, float, float]:
    """Resolve steady-state household block and GE parameters (phi_pi, kappa)."""
    from puremacro.models.hank_sequence_space import (
        SequenceSpaceHANKResult,
        _build_transition_matrix,
        _income_process,
        _stationary_labour,
    )

    if extra_kwargs:
        raise TypeError(
            f"solve_hank_fiscal() got unexpected keyword argument(s): {sorted(extra_kwargs)}"
        )

    if ss_model is None:
        b_val = _DEFAULT_BETA if beta is None else _validate_finite_float(beta, "beta")
        g_val = _DEFAULT_GAMMA if gamma is None else _validate_finite_float(gamma, "gamma")
        r_val = _DEFAULT_R_SS if r_ss is None else _validate_finite_float(r_ss, "r_ss")
        p_val = _DEFAULT_PHI_PI if phi_pi is None else _validate_finite_float(phi_pi, "phi_pi")
        k_val = _DEFAULT_KAPPA if kappa is None else _validate_finite_float(kappa, "kappa")
        na_val = _DEFAULT_N_A if n_a is None else _validate_positive_int(n_a, "n_a", minimum=10)
        amax_val = _DEFAULT_A_MAX if a_max is None else _validate_finite_float(a_max, "a_max")
        hh, ss_res = _solve_fiscal_steady_state(
            beta=b_val,
            gamma=g_val,
            r_ss=r_val,
            phi_pi=p_val,
            kappa=k_val,
            n_a=na_val,
            a_max=amax_val,
            g_ss=g_ss,
            T_jac=min(horizon, 40),
        )
        return hh, ss_res, p_val, k_val

    if isinstance(ss_model, Mapping):
        allowed_map = {
            "T", "beta", "gamma", "r_ss", "phi_pi", "kappa", "n_a", "a_max",
            "shock_magnitude", "shock_rho",
        }
        unknown_map = sorted(set(ss_model) - allowed_map)
        if unknown_map:
            raise ValueError(f"Unsupported ss_model mapping key(s): {unknown_map}")
        b_val = _validate_finite_float(ss_model.get("beta", _DEFAULT_BETA) if beta is None else beta, "beta")
        g_val = _validate_finite_float(ss_model.get("gamma", _DEFAULT_GAMMA) if gamma is None else gamma, "gamma")
        r_val = _validate_finite_float(ss_model.get("r_ss", _DEFAULT_R_SS) if r_ss is None else r_ss, "r_ss")
        p_val = _validate_finite_float(ss_model.get("phi_pi", _DEFAULT_PHI_PI) if phi_pi is None else phi_pi, "phi_pi")
        k_val = _validate_finite_float(ss_model.get("kappa", _DEFAULT_KAPPA) if kappa is None else kappa, "kappa")
        na_val = _validate_positive_int(ss_model.get("n_a", _DEFAULT_N_A) if n_a is None else n_a, "n_a", minimum=10)
        amax_val = _validate_finite_float(ss_model.get("a_max", _DEFAULT_A_MAX) if a_max is None else a_max, "a_max")
        t_jac = _validate_positive_int(ss_model.get("T", min(horizon, 40)), "T", minimum=1)
        hh, ss_res = _solve_fiscal_steady_state(
            beta=b_val,
            gamma=g_val,
            r_ss=r_val,
            phi_pi=p_val,
            kappa=k_val,
            n_a=na_val,
            a_max=amax_val,
            g_ss=g_ss,
            T_jac=t_jac,
        )
        return hh, ss_res, p_val, k_val

    if not isinstance(ss_model, SequenceSpaceHANKResult):
        raise TypeError(
            f"ss_model must be SequenceSpaceHANKResult, Mapping, or None, got {type(ss_model).__name__}"
        )

    cur_beta = float(ss_model.beta)
    cur_gamma = float(ss_model.gamma)
    cur_r_ss = float(ss_model.r_ss)
    cur_n_a = int(len(ss_model.asset_grid))
    cur_a_max = float(ss_model.asset_grid[-1])
    p_val = float(ss_model.phi_pi) if phi_pi is None else _validate_finite_float(phi_pi, "phi_pi")
    k_val = float(ss_model.kappa) if kappa is None else _validate_finite_float(kappa, "kappa")

    b_val = cur_beta if beta is None else _validate_finite_float(beta, "beta")
    g_val = cur_gamma if gamma is None else _validate_finite_float(gamma, "gamma")
    r_val = cur_r_ss if r_ss is None else _validate_finite_float(r_ss, "r_ss")
    na_val = cur_n_a if n_a is None else _validate_positive_int(n_a, "n_a", minimum=10)
    amax_val = cur_a_max if a_max is None else _validate_finite_float(a_max, "a_max")

    needs_resolve = (
        g_ss > 0.0
        or not np.isclose(b_val, cur_beta)
        or not np.isclose(g_val, cur_gamma)
        or not np.isclose(r_val, cur_r_ss)
        or na_val != cur_n_a
        or not np.isclose(amax_val, cur_a_max)
    )
    if needs_resolve:
        hh, ss_res = _solve_fiscal_steady_state(
            beta=b_val,
            gamma=g_val,
            r_ss=r_val,
            phi_pi=p_val,
            kappa=k_val,
            n_a=na_val,
            a_max=amax_val,
            g_ss=g_ss,
            T_jac=min(horizon, 40),
        )
        return hh, ss_res, p_val, k_val

    c_ss = np.asarray(ss_model.policy_c, dtype=float)
    a_ss = np.asarray(ss_model.policy_a, dtype=float)
    D_ss = np.asarray(ss_model.distribution, dtype=float)
    a_grid = np.asarray(ss_model.asset_grid, dtype=float)
    if c_ss.ndim != 2 or c_ss.size == 0 or a_ss.shape != c_ss.shape or D_ss.shape != c_ss.shape:
        raise ValueError(
            "ss_model must carry valid 2D policy_c, policy_a, and distribution arrays"
        )
    s_grid, pi_s = _income_process(c_ss.shape[1])
    N_ss = _stationary_labour(s_grid, pi_s)
    Lam = np.asarray(ss_model.trans_matrix, dtype=float)
    if Lam.shape != (c_ss.size, c_ss.size):
        Lam = _build_transition_matrix(a_ss, a_grid, pi_s)
    B_ss = float(ss_model.government_debt)
    if not np.isfinite(B_ss):
        B_ss = float(np.sum(D_ss.sum(axis=1) * a_grid))
    w_ss = float(ss_model.w_ss)
    Y_ss = w_ss * N_ss
    T_ss = cur_r_ss * B_ss
    C_ss = float(np.sum(D_ss * c_ss))
    hh = _FiscalHousehold(
        a_grid=a_grid,
        s_grid=s_grid,
        pi_s=pi_s,
        beta=cur_beta,
        gamma=cur_gamma,
        r_ss=cur_r_ss,
        w_ss=w_ss,
        V_ss=(1.0 + cur_r_ss) * c_ss ** (-cur_gamma),
        c_ss=c_ss,
        a_ss=a_ss,
        D_ss=D_ss,
        Lambda=Lam,
        B_ss=B_ss,
        N_ss=N_ss,
        G_ss=0.0,
        T_ss=T_ss,
        Y_ss=Y_ss,
        C_ss=C_ss,
        ss_converged=bool(ss_model.ss_converged),
    )
    return hh, ss_model, p_val, k_val


# ---------------------------------------------------------------------------
# Household EGM transition & pure Fake News Jacobians
# ---------------------------------------------------------------------------


def _disposable_income_vec(
    w_t: float,
    T_t: float,
    T_ss: float,
    N_ss: float,
    s_grid: np.ndarray,
    tax_type: str,
) -> np.ndarray:
    """After-tax household income per productivity state s_j."""
    if tax_type == "proportional":
        return (w_t - T_t / N_ss) * s_grid
    return (w_t - T_ss / N_ss) * s_grid - (T_t - T_ss)


def _egm_step_fiscal(
    V_next: np.ndarray,
    a_grid: np.ndarray,
    s_grid: np.ndarray,
    pi_s: np.ndarray,
    beta: float,
    gamma: float,
    r_t: float,
    y_vec: np.ndarray,
    *,
    enforce_feasibility: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One backward EGM step with explicit per-state after-tax income ``y_vec``."""
    R = 1.0 + float(r_t)
    if enforce_feasibility and (not np.isfinite(R) or R <= 1e-6):
        raise _InfeasibleFiscalTrialError(f"Non-positive gross return 1 + r_t = {R:.4e}")
    R_safe = max(R, 1e-6)
    exp_V = V_next @ pi_s.T
    c_endo = (beta * np.maximum(exp_V, 1e-300)) ** (-1.0 / gamma)
    c_t = np.empty_like(c_endo)
    a_t = np.empty_like(c_endo)
    for j in range(len(s_grid)):
        y = float(y_vec[j])
        cash = R_safe * a_grid + y
        if enforce_feasibility and (not np.isfinite(y) or float(cash[0]) <= 1e-12):
            raise _InfeasibleFiscalTrialError(
                f"Infeasible household disposable cash-on-hand at borrowing limit (cash[0]={float(cash[0]):.4e})"
            )
        a_endo = (c_endo[:, j] + a_grid - y) / R_safe
        c = np.interp(a_grid, a_endo, c_endo[:, j])
        c = np.maximum(np.minimum(c, cash), 1e-12)
        c_t[:, j] = c
        a_t[:, j] = cash - c
    V_t = R_safe * c_t ** (-gamma)
    return V_t, c_t, a_t


def _fake_news_fiscal(
    hh: _FiscalHousehold,
    shock_input: str,
    T_fn: int,
    y_ss_vec: np.ndarray,
    h: float = _FD_STEP,
) -> np.ndarray:
    """Exact ABRS Fake News Jacobian w.r.t. pure r, net wage, or lump-sum tax."""
    from puremacro.models.hank_sequence_space import _fake_news_recursion, _forward_step

    args = (hh.a_grid, hh.s_grid, hh.pi_s, hh.beta, hh.gamma)
    base_V, base_c, base_a = _egm_step_fiscal(
        hh.V_ss, *args, hh.r_ss, y_ss_vec, enforce_feasibility=False
    )
    D1_base = _forward_step(hh.D_ss, base_a, hh.a_grid, hh.pi_s)

    if shock_input == "r_pure":
        r_p, y_p = hh.r_ss + h, y_ss_vec
    elif shock_input == "w_net":
        r_p, y_p = hh.r_ss, y_ss_vec + h * hh.s_grid
    elif shock_input == "T_ls":
        r_p, y_p = hh.r_ss, y_ss_vec - h
    else:
        raise ValueError(f"Unknown pure shock_input: {shock_input!r}")

    V_p, c_p, a_p = _egm_step_fiscal(
        hh.V_ss, *args, r_p, y_p, enforce_feasibility=False
    )
    N = hh.c_ss.size
    dc = np.zeros((T_fn, N), dtype=float)
    dD1 = np.zeros((T_fn, N), dtype=float)
    dV = (V_p - base_V) / h
    dc_k = (c_p - base_c) / h
    da_k = (a_p - base_a) / h
    for k in range(T_fn):
        if k > 0:
            V_k, c_k, a_k = _egm_step_fiscal(
                hh.V_ss + h * dV, *args, hh.r_ss, y_ss_vec, enforce_feasibility=False
            )
            dV = (V_k - base_V) / h
            dc_k = (c_k - base_c) / h
            da_k = (a_k - base_a) / h
        dc[k] = dc_k.ravel()
        dD1[k] = (
            (_forward_step(hh.D_ss, base_a + h * da_k, hh.a_grid, hh.pi_s) - D1_base) / h
        ).ravel()
    J, _, _ = _fake_news_recursion(
        T_fn, hh.c_ss.ravel(), hh.Lambda, hh.D_ss.ravel(), dc, dD1
    )
    return J


def _build_linear_system(
    hh: _FiscalHousehold,
    *,
    horizon: int,
    closure: str,
    tax_type: str,
    phi_B: float,
    phi_G: float,
    M_r_Y: np.ndarray,
    dG: np.ndarray,
    eps_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble first-order sequence-space Jacobian J_ss and linear dY, dC."""
    y_ss_vec = (hh.w_ss - hh.T_ss / hh.N_ss) * hh.s_grid
    J_r_real = _fake_news_fiscal(hh, "r_pure", horizon + 1, y_ss_vec)
    J_w_real = _fake_news_fiscal(hh, "w_net", horizon + 1, y_ss_vec)

    J_C_r = J_r_real[:horizon, 1 : horizon + 1]
    J_C_w_on_real = J_w_real[:horizon, : horizon + 1]

    S_shift = np.zeros((horizon + 1, horizon), dtype=float)
    S_shift[1:, :] = np.eye(horizon)
    M_r_real_Y = S_shift @ M_r_Y
    r_real_m = S_shift @ eps_m

    if closure == "balanced_budget":
        M_T_Y = hh.B_ss * M_r_real_Y
        M_T_G = np.zeros((horizon + 1, horizon), dtype=float)
        M_T_G[:horizon, :horizon] = np.eye(horizon)
        dT_from_m = hh.B_ss * r_real_m
    else:
        rho_B = 1.0 + hh.r_ss - phi_B
        idx = np.arange(horizon)
        diff = idx[:, None] - idx[None, :]
        L_B = np.where(diff >= 0, rho_B ** np.maximum(diff, 0), 0.0)
        M_B_Y = L_B @ (hh.B_ss * M_r_real_Y[:horizon, :])
        M_B_G = L_B * (1.0 - phi_G)
        dB_from_m = L_B @ (hh.B_ss * r_real_m[:horizon])

        M_T_Y = phi_B * np.vstack([np.zeros((1, horizon)), M_B_Y])
        M_T_G = phi_B * np.vstack([np.zeros((1, horizon)), M_B_G])
        M_T_G[:horizon, :horizon] += phi_G * np.eye(horizon)
        dT_from_m = phi_B * np.r_[0.0, dB_from_m]

    I_pad = np.vstack([np.eye(horizon), np.zeros((1, horizon))])
    if tax_type == "proportional":
        J_C_Y_eff = J_C_w_on_real @ (I_pad - M_T_Y) / hh.N_ss + J_C_r @ M_r_Y
        dC_shock = (
            J_C_r @ eps_m
            - J_C_w_on_real @ (M_T_G @ dG + dT_from_m) / hh.N_ss
        )
    else:
        J_T_real = _fake_news_fiscal(hh, "T_ls", horizon + 1, y_ss_vec)
        J_C_T_on_real = J_T_real[:horizon, : horizon + 1]
        J_C_Y_eff = (
            J_C_w_on_real @ I_pad / hh.N_ss
            + J_C_T_on_real @ M_T_Y
            + J_C_r @ M_r_Y
        )
        dC_shock = (
            J_C_r @ eps_m
            + J_C_T_on_real @ (M_T_G @ dG + dT_from_m)
        )

    J_ss = np.eye(horizon) - J_C_Y_eff
    dY_lin = np.linalg.solve(J_ss, dG + dC_shock)
    dC_lin = dY_lin - dG
    return J_ss, dY_lin, dC_lin


def _evaluate_fiscal_paths(
    dY: np.ndarray,
    *,
    hh: _FiscalHousehold,
    dG: np.ndarray,
    eps_m: np.ndarray,
    K_pi: np.ndarray,
    M_r_Y: np.ndarray,
    closure: str,
    tax_type: str,
    phi_B: float,
    phi_G: float,
    enforce_feasibility: bool = True,
    audit_final: bool = False,
) -> dict[str, Any]:
    """Evaluate nonlinear fiscal, GE, and household transition paths for a trial dY."""
    from puremacro.models.hank_sequence_space import _forward_step

    horizon = len(dY)
    Y = hh.Y_ss + dY
    w = hh.w_ss * (1.0 + dY / hh.Y_ss)
    if enforce_feasibility and (not np.all(np.isfinite(Y)) or np.any(Y <= 1e-8) or np.any(w <= 1e-8)):
        raise _InfeasibleFiscalTrialError("Output or gross wage dropped to non-positive region")

    pi = K_pi @ dY
    dr_ea = M_r_Y @ dY + eps_m
    r_ex_ante = hh.r_ss + dr_ea
    rr_seq = np.concatenate(([hh.r_ss], r_ex_ante))
    G = hh.G_ss + dG

    if closure == "balanced_budget":
        B = np.full(horizon, hh.B_ss, dtype=float)
        T_path = rr_seq[:horizon] * hh.B_ss + G
        T_T = float(rr_seq[horizon]) * hh.B_ss + hh.G_ss
    else:
        B = np.empty(horizon, dtype=float)
        T_path = np.empty(horizon, dtype=float)
        b_prev = hh.B_ss
        for t in range(horizon):
            T_t = hh.T_ss + phi_B * (b_prev - hh.B_ss) + phi_G * float(dG[t])
            B_t = (1.0 + float(rr_seq[t])) * b_prev + float(G[t]) - T_t
            if not (np.isfinite(T_t) and np.isfinite(B_t)) or abs(B_t) > 1e12:
                raise _InfeasibleFiscalTrialError("Government debt or taxes overflowed")
            T_path[t] = T_t
            B[t] = B_t
            b_prev = B_t
        T_T = hh.T_ss + phi_B * (b_prev - hh.B_ss)

    y_T = _disposable_income_vec(hh.w_ss, T_T, hh.T_ss, hh.N_ss, hh.s_grid, tax_type)
    V_next, c_terminal, _ = _egm_step_fiscal(
        hh.V_ss,
        hh.a_grid,
        hh.s_grid,
        hh.pi_s,
        hh.beta,
        hh.gamma,
        float(rr_seq[horizon]),
        y_T,
        enforce_feasibility=enforce_feasibility,
    )

    n_a, n_s = hh.c_ss.shape
    c_path = np.empty((horizon, n_a, n_s), dtype=float)
    a_path = np.empty((horizon, n_a, n_s), dtype=float)
    y_paths = np.empty((horizon, n_s), dtype=float)
    for t in range(horizon - 1, -1, -1):
        y_t = _disposable_income_vec(
            float(w[t]), float(T_path[t]), hh.T_ss, hh.N_ss, hh.s_grid, tax_type
        )
        y_paths[t] = y_t
        V_next, c_path[t], a_path[t] = _egm_step_fiscal(
            V_next,
            hh.a_grid,
            hh.s_grid,
            hh.pi_s,
            hh.beta,
            hh.gamma,
            float(rr_seq[t]),
            y_t,
            enforce_feasibility=enforce_feasibility,
        )

    D = hh.D_ss.copy()
    C = np.empty(horizon, dtype=float)
    A = np.empty(horizon, dtype=float)
    dist_mass_err = 0.0
    min_dist_mass = float(np.min(D))
    top_grid_mass = float(np.sum(D[-1, :]))
    a_max = float(hh.a_grid[-1])
    active_overshoot = 0.0
    mass_overshoot = 0.0

    for t in range(horizon):
        C[t] = float(np.sum(D * c_path[t]))
        A[t] = float(np.sum(D * a_path[t]))
        if audit_final:
            overshoot_t = np.maximum(a_path[t] - a_max, 0.0)
            active_overshoot = max(
                active_overshoot, float(np.max(np.where(D > 1e-14, overshoot_t, 0.0)))
            )
            mass_overshoot = max(mass_overshoot, float(np.sum(D * overshoot_t)))
        D = _forward_step(D, a_path[t], hh.a_grid, hh.pi_s)
        if audit_final:
            dist_mass_err = max(dist_mass_err, abs(float(np.sum(D)) - 1.0))
            min_dist_mass = min(min_dist_mass, float(np.min(D)))
            top_grid_mass = max(top_grid_mass, float(np.sum(D[-1, :])))

    out: dict[str, Any] = {
        "Y": Y,
        "C": C,
        "G": G,
        "T": T_path,
        "B": B,
        "A": A,
        "r": rr_seq[:horizon],
        "r_ex_ante": r_ex_ante,
        "w": w,
        "pi": pi,
        "H": Y - C - G,
    }

    if audit_final:
        cash_all = (1.0 + rr_seq[:horizon, None, None]) * hh.a_grid[None, :, None] + y_paths[:, None, :]
        hh_budget_err = float(np.max(np.abs(c_path + a_path - cash_all)))
        min_c = float(np.min(c_path))

        following_c = np.concatenate((c_path[1:], c_terminal[None]))
        marginal = (1.0 + rr_seq[1:, None, None]) * following_c ** (-hh.gamma)
        expect = marginal @ hh.pi_s.T
        euler_err = 0.0
        a_min_tol = float(hh.a_grid[0]) + 1e-10
        for t in range(horizon):
            interp_exp = np.column_stack(
                [np.interp(a_path[t, :, j], hh.a_grid, expect[t, :, j]) for j in range(n_s)]
            )
            grad = hh.beta * interp_exp / (c_path[t] ** (-hh.gamma)) - 1.0
            kkt = np.where(a_path[t] <= a_min_tol, np.maximum(grad, 0.0), grad)
            euler_err = max(euler_err, float(np.max(np.abs(kkt))))

        out["audit"] = {
            "household_budget_residual": hh_budget_err,
            "euler_kkt_max": euler_err,
            "dist_mass_err": dist_mass_err,
            "min_dist_mass": min_dist_mass,
            "top_grid_mass": top_grid_mass,
            "max_asset_overshoot": active_overshoot,
            "mass_asset_overshoot": mass_overshoot,
            "min_consumption": min_c,
        }
    return out


# ---------------------------------------------------------------------------
# Public solver & comparison functions
# ---------------------------------------------------------------------------


def solve_hank_fiscal(
    ss_model: Any | None = None,
    shock_seq: Sequence[float] | np.ndarray | None = None,
    *,
    closure: str = "balanced_budget",
    tax_type: str = "proportional",
    horizon: int = 300,
    phi_B: float = 0.08,
    phi_G: float = 0.0,
    g_ss: float = 0.0,
    monetary_shock_seq: Sequence[float] | np.ndarray | None = None,
    beta: float | None = None,
    gamma: float | None = None,
    r_ss: float | None = None,
    phi_pi: float | None = None,
    kappa: float | None = None,
    n_a: int | None = None,
    a_max: float | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
    backtracking: bool = True,
    allow_unstable_rule: bool = False,
    **kwargs: Any,
) -> FiscalHANKResult:
    """Solve non-linear HANK transition under explicit financed fiscal closures.

    Parameters
    ----------
    ss_model : SequenceSpaceHANKResult, Mapping, or None, optional
        Pre-solved steady state or parameter dictionary. When ``None``, solves
        the steady state with ``n_a=100, a_max=100.0`` (avoiding upper-tail
        asset truncation so Walras's Law holds to machine precision).
    shock_seq : Sequence[float] or np.ndarray, optional
        Exogenous fiscal spending shock path ``dG_t``. Defaults to
        ``0.01 * 0.7**t``. Shorter sequences are zero-padded to ``horizon``;
        longer ones extend ``horizon``.
    closure : {'balanced_budget', 'debt_rule'}, default 'balanced_budget'
        Fiscal financing rule:
        * ``'balanced_budget'``: ``B_t = B_ss`` and ``T_t = r_t B_ss + G_t``.
        * ``'debt_rule'``: ``B_t = (1 + r_t) B_{t-1} + G_t - T_t`` with
          ``T_t = T_ss + phi_B * (B_{t-1} - B_ss) + phi_G * dG_t``.
    tax_type : {'proportional', 'lump_sum'}, default 'proportional'
        How tax adjustments ``dT_t`` enter household disposable income.
    horizon : int, default 300
        Transition horizon ``T`` (quarters).
    phi_B : float, default 0.08
        Debt-feedback parameter in ``closure='debt_rule'``. Must satisfy the
        stationary Leeper condition ``r_ss < phi_B < 2 + r_ss`` unless
        ``allow_unstable_rule=True``.
    phi_G : float, default 0.0
        Contemporaneous spending-tax offset in ``closure='debt_rule'``.
    g_ss : float, default 0.0
        Steady-state government spending level in ``[0, Y_ss)``.
    monetary_shock_seq : Sequence[float] or np.ndarray, optional
        Optional contemporaneous Taylor-rule shock ``eps_m_t``.
    beta, gamma, r_ss, phi_pi, kappa, n_a, a_max : optional
        Steady-state and GE parameter overrides.
    max_iter : int, default 100
        Maximum number of Broyden iterations.
    tol : float, default 1e-8
        Tolerance checked against ``||Y - C - G||_inf``, government budget, and
        asset clearing ``||A - B||_inf``.
    backtracking : bool, default True
        Perform monotone line search during Broyden iterations.
    allow_unstable_rule : bool, default False
        If ``False`` (default), ``phi_B <= r_ss`` or ``phi_B >= 2 + r_ss``
        raises ``ValueError``. If ``True``, returns a result explicitly flagged
        ``converged=False`` with ``regime='non_stationary_explosive'``.

    Returns
    -------
    FiscalHANKResult
    """
    from puremacro.models.hank_sequence_space import _ge_matrices
    from puremacro.solvers.diagnostics import make_solver_diagnostic_record

    closure_key = _normalize_closure(closure)
    tax_key = _normalize_tax_type(tax_type)
    horizon = _validate_positive_int(horizon, "horizon", minimum=1)
    max_iter = _validate_positive_int(max_iter, "max_iter", minimum=0)
    tol_val = _validate_finite_float(tol, "tol")
    if tol_val <= 0.0:
        raise ValueError(f"tol must be positive, got {tol_val}")
    phi_B_val = _validate_finite_float(phi_B, "phi_B")
    phi_G_val = _validate_finite_float(phi_G, "phi_G")
    g_ss_val = _validate_finite_float(g_ss, "g_ss")
    if not isinstance(backtracking, (bool, np.bool_)):
        raise TypeError("backtracking must be a boolean")
    if not isinstance(allow_unstable_rule, (bool, np.bool_)):
        raise TypeError("allow_unstable_rule must be a boolean")

    g_arr = _coerce_shock_array(shock_seq, "shock_seq")
    m_arr = _coerce_shock_array(monetary_shock_seq, "monetary_shock_seq")
    if g_arr is not None:
        horizon = max(horizon, len(g_arr))
    if m_arr is not None:
        horizon = max(horizon, len(m_arr))

    dG = (
        0.01 * (0.7 ** np.arange(horizon))
        if g_arr is None
        else _pad_or_copy(g_arr, horizon)
    )
    eps_m = np.zeros(horizon, dtype=float) if m_arr is None else _pad_or_copy(m_arr, horizon)

    hh, ss_res, phi_pi_eff, kappa_eff = _resolve_fiscal_steady_state(
        ss_model,
        horizon,
        g_ss=g_ss_val,
        beta=beta,
        gamma=gamma,
        r_ss=r_ss,
        phi_pi=phi_pi,
        kappa=kappa,
        n_a=n_a,
        a_max=a_max,
        extra_kwargs=kwargs,
    )

    unstable_rule = False
    debt_eigenvalue = 0.0 if closure_key == "balanced_budget" else (1.0 + hh.r_ss - phi_B_val)
    if closure_key == "debt_rule":
        if phi_B_val <= hh.r_ss or phi_B_val >= 2.0 + hh.r_ss:
            unstable_rule = True
            msg = (
                f"Unbacked/explosive fiscal debt rule: phi_B={phi_B_val:.6g} violates stationary "
                f"stability r_ss < phi_B < 2 + r_ss (r_ss={hh.r_ss:.6g}; steady-state debt root "
                f"1 + r_ss - phi_B = {debt_eigenvalue:.6g})."
            )
            if not allow_unstable_rule:
                raise ValueError(msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

    K_pi, M_r_Y = _ge_matrices(horizon, hh.beta, kappa_eff, phi_pi_eff)
    J_ss, dY_lin, dC_lin = _build_linear_system(
        hh,
        horizon=horizon,
        closure=closure_key,
        tax_type=tax_key,
        phi_B=phi_B_val,
        phi_G=phi_G_val,
        M_r_Y=M_r_Y,
        dG=dG,
        eps_m=eps_m,
    )
    B_inv = np.linalg.inv(J_ss)

    ss_truncation_gap = float(abs(np.sum(hh.D_ss * hh.a_ss) - hh.B_ss))
    # Accumulating goods-clearing error over T periods multiplies by ~(1+r)^T;
    # tightening Broyden's inner target keeps ||A - B||_inf < tol whenever the grid does not truncate.
    inner_tol = min(tol_val, tol_val / max(20.0, 0.1 * horizon))

    def try_eval(u: np.ndarray, *, audit_final: bool = False) -> dict[str, Any]:
        return _evaluate_fiscal_paths(
            u,
            hh=hh,
            dG=dG,
            eps_m=eps_m,
            K_pi=K_pi,
            M_r_Y=M_r_Y,
            closure=closure_key,
            tax_type=tax_key,
            phi_B=phi_B_val,
            phi_G=phi_G_val,
            enforce_feasibility=not unstable_rule,
            audit_final=audit_final,
        )

    U = dY_lin.copy()
    try:
        ev = try_eval(U)
    except _InfeasibleFiscalTrialError:
        U = np.zeros(horizon, dtype=float)
        ev = try_eval(U)

    H_val = ev["H"]
    norm = float(np.max(np.abs(H_val)))
    asset_norm = float(np.max(np.abs(ev["A"] - ev["B"])))
    merit = float(np.linalg.norm(H_val))
    norm_history: list[float] = [norm]
    iterations = 0
    stall_reason = ""
    reset_done = False
    B_mat = B_inv.copy()

    def _broyden_done() -> bool:
        if norm >= inner_tol:
            return False
        if ss_truncation_gap < 0.1 * tol_val:
            return asset_norm < tol_val
        return norm < tol_val

    while not _broyden_done() and iterations < max_iter:
        dU = -B_mat @ H_val
        accepted = False
        ev_try: dict[str, Any] = ev
        U_try = U
        H_try = H_val
        m_try = merit
        if backtracking:
            step = 1.0
            for _ in range(12):
                U_cand = U + step * dU
                try:
                    ev_cand = try_eval(U_cand)
                except _InfeasibleFiscalTrialError:
                    step *= 0.5
                    continue
                H_cand = ev_cand["H"]
                if not np.all(np.isfinite(H_cand)):
                    step *= 0.5
                    continue
                m_cand = float(np.linalg.norm(H_cand))
                if m_cand <= (1.0 - 1e-4 * step) * merit:
                    U_try, ev_try, H_try, m_try = U_cand, ev_cand, H_cand, m_cand
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                if not reset_done:
                    B_mat = B_inv.copy()
                    reset_done = True
                    continue
                stall_reason = "line_search_stalled"
                break
        else:
            U_cand = U + dU
            try:
                ev_cand = try_eval(U_cand)
            except _InfeasibleFiscalTrialError as exc:
                stall_reason = f"infeasible_trial: {exc}"
                break
            H_cand = ev_cand["H"]
            if not np.all(np.isfinite(H_cand)):
                stall_reason = "nonfinite_residual"
                break
            U_try, ev_try, H_try, m_try = U_cand, ev_cand, H_cand, float(np.linalg.norm(H_cand))
            accepted = True

        delta_U = U_try - U
        delta_H = H_try - H_val
        u_vec = delta_U - B_mat @ delta_H
        v_vec = delta_U @ B_mat
        denom = float(np.dot(v_vec, delta_H))
        if abs(denom) > 1e-14:
            B_mat = B_mat + np.outer(u_vec, v_vec) / denom
        U, ev, H_val, merit = U_try, ev_try, H_try, m_try
        norm = float(np.max(np.abs(H_val)))
        asset_norm = float(np.max(np.abs(ev["A"] - ev["B"])))
        norm_history.append(norm)
        iterations += 1
        if accepted:
            reset_done = False

    final = try_eval(U, audit_final=True)
    audit = final["audit"]
    Y = final["Y"]
    C = final["C"]
    G = final["G"]
    T_path = final["T"]
    B = final["B"]
    A = final["A"]
    r_real = final["r"]
    r_ex_ante = final["r_ex_ante"]
    w = final["w"]
    pi = final["pi"]
    H_final = final["H"]

    dY = Y - hh.Y_ss
    dC = C - hh.C_ss
    dG_actual = G - hh.G_ss
    dT = T_path - hh.T_ss
    dB = B - hh.B_ss
    tau = (
        T_path / Y
        if tax_key == "proportional"
        else np.full(horizon, hh.T_ss / hh.Y_ss, dtype=float)
    )

    goods_res = float(np.max(np.abs(H_final)))
    b_prev = np.concatenate(([hh.B_ss], B[:-1]))
    gov_res = float(np.max(np.abs(B - ((1.0 + r_real) * b_prev + G - T_path))))
    Q = np.cumprod(1.0 / (1.0 + r_real))
    gov_ibc_res = float(abs(hh.B_ss - (np.sum(Q * (T_path - G)) + Q[-1] * B[-1])))
    asset_res = float(np.max(np.abs(A - B)))
    term_debt_gap = float(abs(B[-1] - hh.B_ss))
    term_disc_debt_gap = float(abs(Q[-1] * (B[-1] - hh.B_ss)))

    imp_mult = float(dY[0] / dG_actual[0]) if abs(dG_actual[0]) > 1e-12 else float("nan")
    sum_dG = float(np.sum(dG_actual))
    cum_mult = float(np.sum(dY) / sum_dG) if abs(sum_dG) > 1e-12 else float("nan")
    pv_dG = float(np.sum(Q * dG_actual))
    pv_mult = float(np.sum(Q * dY) / pv_dG) if abs(pv_dG) > 1e-12 else float("nan")

    solver_ok = (goods_res < tol_val) and (not stall_reason)
    all_markets_ok = (
        goods_res < tol_val
        and gov_res < tol_val
        and asset_res < tol_val
        and audit["household_budget_residual"] < 1e-8
        and audit["euler_kkt_max"] <= 0.01
    )
    converged = bool(solver_ok and all_markets_ok and not unstable_rule)

    supp_warnings: list[str] = []
    is_supported = True
    regime = "interior"
    if unstable_rule:
        is_supported = False
        regime = "non_stationary_explosive"
        supp_warnings.append(
            f"phi_B={phi_B_val:.6g} violates r_ss < phi_B < 2 + r_ss (root={debt_eigenvalue:.6g})"
        )
    elif (
        audit["top_grid_mass"] > 1e-6
        or audit["max_asset_overshoot"] > 1e-6
        or audit["mass_asset_overshoot"] > 1e-8
        or ss_truncation_gap > 1e-6
    ):
        is_supported = False
        regime = "boundary_asset_truncation"
        msg = (
            f"Upper asset grid truncation detected (a_max={float(hh.a_grid[-1]):.1f}, "
            f"top_mass={audit['top_grid_mass']:.2e}, overshoot={audit['max_asset_overshoot']:.2e}, "
            f"||A-B||_inf={asset_res:.2e}); increase a_max >= 50.0 (default 100.0) for exact Walras clearing."
        )
        supp_warnings.append(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    elif (
        audit["min_consumption"] <= 1e-10
        or audit["min_dist_mass"] < -1e-12
        or audit["dist_mass_err"] > 1e-9
    ):
        is_supported = False
        regime = "infeasible_prices"
        supp_warnings.append(
            f"Household support boundary triggered (min_c={audit['min_consumption']:.2e}, "
            f"min_mass={audit['min_dist_mass']:.2e})"
        )
    elif closure_key == "debt_rule" and term_debt_gap > 0.05:
        supp_warnings.append(
            f"Terminal debt gap |B_T-1 - B_ss|={term_debt_gap:.3e} remains non-negligible at horizon={horizon}."
        )

    if not solver_ok and not unstable_rule:
        reason = stall_reason or f"max_iter={max_iter}_reached"
        warnings.warn(
            f"solve_hank_fiscal did not converge: ||Y-C-G||_inf={goods_res:.3e}, "
            f"||A-B||_inf={asset_res:.3e} ({reason}).",
            RuntimeWarning,
            stacklevel=2,
        )

    stopping_reason = (
        "explosive_debt_rule"
        if unstable_rule
        else ("tolerance_met" if (solver_ok and all_markets_ok) else (stall_reason or "residual_above_tol"))
    )
    diag = make_solver_diagnostic_record(
        converged=bool(solver_ok and not unstable_rule),
        iterations=iterations,
        stopping_reason=stopping_reason,
        solver_method=f"sequence_space_broyden_{closure_key}_{tax_key}",
        tolerance=tol_val,
        is_accurate=bool(all_markets_ok),
        max_residual=float(max(goods_res, gov_res, asset_res)),
        residual_norm=float(np.linalg.norm(H_final)),
        residual_type="goods_government_asset_clearing",
        accuracy_tolerance=tol_val,
        residuals={
            "goods_market_inf": goods_res,
            "gov_budget_inf": gov_res,
            "gov_intertemporal": gov_ibc_res,
            "asset_market_inf": asset_res,
            "household_budget_inf": float(audit["household_budget_residual"]),
            "euler_kkt_max": float(audit["euler_kkt_max"]),
            "distribution_mass_err": float(audit["dist_mass_err"]),
        },
        is_supported=is_supported,
        regime=regime,
        warnings=supp_warnings,
        boundary_metrics={
            "debt_eigenvalue": float(debt_eigenvalue),
            "terminal_debt_gap": term_debt_gap,
            "terminal_discounted_debt_gap": term_disc_debt_gap,
            "top_grid_mass": float(audit["top_grid_mass"]),
            "max_asset_overshoot": float(audit["max_asset_overshoot"]),
            "min_consumption": float(audit["min_consumption"]),
            "euler_kkt_max": float(audit["euler_kkt_max"]),
        },
        evidence={
            "closure": closure_key,
            "tax_type": tax_key,
            "horizon": horizon,
            "phi_B": phi_B_val if closure_key == "debt_rule" else None,
            "phi_G": phi_G_val if closure_key == "debt_rule" else None,
        },
    )

    ss_info = {
        "Y_ss": float(hh.Y_ss),
        "C_ss": float(hh.C_ss),
        "G_ss": float(hh.G_ss),
        "T_ss": float(hh.T_ss),
        "B_ss": float(hh.B_ss),
        "r_ss": float(hh.r_ss),
        "w_ss": float(hh.w_ss),
        "tau_ss": float(hh.T_ss / hh.Y_ss),
        "N_ss": float(hh.N_ss),
        "top_grid_mass": float(np.sum(hh.D_ss[-1, :])),
    }

    return FiscalHANKResult(
        Y=Y,
        C=C,
        G=G,
        T=T_path,
        B=B,
        A=A,
        r=r_real,
        w=w,
        tau=tau,
        pi=pi,
        r_ex_ante=r_ex_ante,
        dY=dY,
        dC=dC,
        dG=dG_actual,
        dT=dT,
        dB=dB,
        linear_dY=dY_lin,
        linear_dC=dC_lin,
        residuals=H_final,
        goods_market_residual=goods_res,
        gov_budget_residual=gov_res,
        gov_intertemporal_residual=gov_ibc_res,
        asset_market_residual=asset_res,
        terminal_debt_gap=term_debt_gap,
        terminal_discounted_debt_gap=term_disc_debt_gap,
        impact_multiplier=imp_mult,
        cumulative_multiplier=cum_mult,
        present_value_multiplier=pv_mult,
        closure=closure_key,
        tax_type=tax_key,
        horizon=horizon,
        phi_B=phi_B_val if closure_key == "debt_rule" else None,
        phi_G=phi_G_val if closure_key == "debt_rule" else None,
        converged=converged,
        iterations=iterations,
        tol=tol_val,
        norm_history=norm_history,
        steady_state=ss_info,
        steady_state_model=ss_res,
        solver_diagnostics=diag,
    )


def compare_fiscal_closures(
    ss_model: Any | None = None,
    shock_seq: Sequence[float] | np.ndarray | None = None,
    *,
    horizon: int = 300,
    phi_B: float = 0.08,
    phi_G: float = 0.0,
    include_unfinanced: bool = True,
    tol: float = 1e-8,
    **kwargs: Any,
) -> pd.DataFrame:
    """Compare multipliers and clearing diagnostics across fiscal closures.

    Runs ``balanced_budget`` and ``debt_rule`` under both ``proportional`` and
    ``lump_sum`` tax adjustments on the same steady state and spending shock,
    plus (when ``include_unfinanced=True``) the legacy goods-only
    ``solve_nonlinear_transition(..., shock_var='G')`` baseline for contrast.
    """
    from puremacro.models.hank_sequence_space import solve_nonlinear_transition

    g_arr = _coerce_shock_array(shock_seq, "shock_seq")
    horizon = _validate_positive_int(horizon, "horizon", minimum=1)
    if g_arr is not None:
        horizon = max(horizon, len(g_arr))
    dG = (
        0.01 * (0.7 ** np.arange(horizon))
        if g_arr is None
        else _pad_or_copy(g_arr, horizon)
    )

    specs: list[tuple[str, str, str]] = [
        ("balanced_budget_proportional", "balanced_budget", "proportional"),
        ("balanced_budget_lump_sum", "balanced_budget", "lump_sum"),
        ("debt_rule_proportional", "debt_rule", "proportional"),
        ("debt_rule_lump_sum", "debt_rule", "lump_sum"),
    ]
    rows: list[dict[str, Any]] = []
    base_ss = ss_model
    for label, cl, tt in specs:
        res = solve_hank_fiscal(
            ss_model=base_ss,
            shock_seq=dG,
            closure=cl,
            tax_type=tt,
            horizon=horizon,
            phi_B=phi_B,
            phi_G=phi_G,
            tol=tol,
            **kwargs,
        )
        if base_ss is None:
            base_ss = res.steady_state_model
        rows.append(
            {
                "scenario": label,
                "closure": cl,
                "tax_type": tt,
                "impact_multiplier": res.impact_multiplier,
                "cumulative_multiplier": res.cumulative_multiplier,
                "pv_multiplier": res.present_value_multiplier,
                "peak_dC": float(res.dC[int(np.argmax(np.abs(res.dC)))]),
                "peak_dB": float(res.dB[int(np.argmax(np.abs(res.dB)))]),
                "goods_residual": res.goods_market_residual,
                "gov_budget_residual": res.gov_budget_residual,
                "asset_residual": res.asset_market_residual,
                "status": res.solver_diagnostics.overall_status,
            }
        )

    if include_unfinanced:
        unfin = solve_nonlinear_transition(
            ss_model=base_ss,
            shock_seq=dG,
            shock_var="G",
            horizon=horizon,
            tol=tol,
        )
        dY_u = unfin.nonlinear_path
        dC_u = unfin.irf_consumption_nonlinear
        r_ss_u = float(unfin.steady_state_model.r_ss)
        rr_u = np.concatenate(([r_ss_u], r_ss_u + unfin.irf_rate_nonlinear[:-1]))
        Q_u = np.cumprod(1.0 / (1.0 + rr_u))
        B_ss_u = float(unfin.steady_state_model.government_debt)
        rows.append(
            {
                "scenario": "unfinanced_goods_only",
                "closure": "unfinanced_goods_only",
                "tax_type": "interest_only",
                "impact_multiplier": float(dY_u[0] / dG[0]) if abs(dG[0]) > 1e-12 else float("nan"),
                "cumulative_multiplier": float(np.sum(dY_u) / np.sum(dG)) if abs(np.sum(dG)) > 1e-12 else float("nan"),
                "pv_multiplier": float(np.sum(Q_u * dY_u) / np.sum(Q_u * dG)) if abs(np.sum(Q_u * dG)) > 1e-12 else float("nan"),
                "peak_dC": float(dC_u[int(np.argmax(np.abs(dC_u)))]),
                "peak_dB": float("nan"),
                "goods_residual": float(np.max(np.abs(unfin.residuals))),
                "gov_budget_residual": float(np.max(np.abs(dG))),
                "asset_residual": float(np.max(np.abs(np.cumsum(dG)))),
                "status": "GOODS_ONLY_UNFINANCED" if B_ss_u >= 0 else "GOODS_ONLY_UNFINANCED",
            }
        )

    return pd.DataFrame(rows).set_index("scenario")
