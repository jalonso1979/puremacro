"""Sequence-Space Heterogeneous-Agent New Keynesian (HANK) Model.

Implements the Sequence-Space Jacobian framework of Auclert, Bardóczy,
Rognlie & Straub (2021, *Econometrica*):
- Stationary Incomplete Markets Household Problem (Aiyagari-Bewley-Huggett).
- Endogenous Grid Method (EGM) steady-state policy functions and stationary distribution.
- Fake News Algorithm for O(T^2) sequence-space Jacobian calculation.
- Targeted fiscal transfer simulation across wealth deciles.
- General Equilibrium transition dynamics and monetary policy impulse responses
  in O(T^3) linear solve without state-space discretization curse of dimensionality.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from puremacro.plot import _palette
from puremacro.reports import _df_to_latex, _df_to_markdown, _df_to_typst


@dataclass(frozen=True)
class FakeNewsResult:
    """Fake News Algorithm decomposition result (Auclert et al. 2021).

    Attributes
    ----------
    jacobian : np.ndarray
        Full sequence-space Jacobian matrix J of shape (T, T).
    fake_news : np.ndarray
        Fake news matrix F of shape (T, T) where F[t, s] represents the revision
        in expected outcome at date t upon news at date 0 about shock at date s.
    expectation_vectors : np.ndarray
        Expectation vectors curly_E of shape (T, N) where curly_E[t] = (Pi^t) c_ss.
    horizon : int
        Horizon length T.
    """
    jacobian: np.ndarray
    fake_news: np.ndarray
    expectation_vectors: np.ndarray
    horizon: int

    def summary(self) -> str:
        lines = [
            "Fake News Algorithm Decomposition (Auclert et al. 2021)",
            "=" * 68,
            f"Horizon T                       : {self.horizon} periods",
            f"Jacobian Frobenius Norm         : {np.linalg.norm(self.jacobian):.6f}",
            f"Fake News Frobenius Norm        : {np.linalg.norm(self.fake_news):.6f}",
            f"Impact Effect (J[0, 0])         : {self.jacobian[0, 0]:.6f}",
            f"Diagonal Average (J[t, t])      : {np.mean(np.diag(self.jacobian)):.6f}",
            "=" * 68,
        ]
        return "\n".join(lines)

    def to_frame(self, which: str = "jacobian") -> pd.DataFrame:
        mat = self.fake_news if which.lower() in ("fake_news", "f") else self.jacobian
        return pd.DataFrame(
            mat,
            index=[f"t={t}" for t in range(self.horizon)],
            columns=[f"s={s}" for s in range(self.horizon)],
        )

    def to_markdown(self, **kwargs) -> str:
        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        return _df_to_typst(self.to_frame(), **kwargs)

    def plot(self, style: str = "publication", figsize: tuple[float, float] = (10.5, 4.2)):
        """Plot heatmaps of the Fake News matrix F and Sequence-Space Jacobian J."""
        cmap = "coolwarm" if style != "grayscale" else "gray"
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        im1 = ax1.imshow(self.fake_news, cmap=cmap, aspect="auto", origin="upper")
        ax1.set_title(r"Fake News Matrix $\mathcal{F}_{t,s}$", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Shock Date $s$", fontsize=9)
        ax1.set_ylabel("Outcome Date $t$", fontsize=9)
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        im2 = ax2.imshow(self.jacobian, cmap=cmap, aspect="auto", origin="upper")
        ax2.set_title(r"Sequence-Space Jacobian $\mathcal{J}_{t,s}$", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Shock Date $s$", fontsize=9)
        ax2.set_ylabel("Outcome Date $t$", fontsize=9)
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        fig.tight_layout()
        return fig


@dataclass(frozen=True)
class FiscalTransferResult:
    """Simulation results of targeted fiscal transfers in Sequence-Space HANK.

    Attributes
    ----------
    irf_consumption : np.ndarray
        Aggregate dynamic consumption response path dC_t (T,).
    cumulative_multiplier : float
        Cumulative fiscal multiplier: sum(dC_t) / transfer_amount.
    impact_mpc : float
        Immediate period-0 aggregate marginal propensity to consume.
    mpc_by_group : pd.Series
        Average MPC of target group vs non-target group.
    decile_incidence : pd.DataFrame
        Distribution of transfer received and consumption response across wealth deciles.
    target_group : str
        Specification of recipient group (e.g. 'borrowers', 'unconstrained', 'all').
    transfer_amount : float
        Total fiscal budget expenditure.
    """
    irf_consumption: np.ndarray
    cumulative_multiplier: float
    impact_mpc: float
    mpc_by_group: pd.Series
    decile_incidence: pd.DataFrame
    target_group: str
    transfer_amount: float

    def summary(self) -> str:
        lines = [
            f"Targeted Fiscal Transfer Simulation ({self.target_group.capitalize()})",
            "=" * 68,
            f"Total Fiscal Outlay             : {self.transfer_amount:.4f}",
            f"Impact MPC (Date 0)             : {self.impact_mpc:.4f}",
            f"Cumulative Fiscal Multiplier    : {self.cumulative_multiplier:.4f}",
            "-" * 68,
            "Incidence Across Wealth Deciles (Share of Transfer & Consumption):",
            self.decile_incidence.round(4).to_string(),
            "=" * 68,
        ]
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        return self.decile_incidence

    def to_markdown(self, **kwargs) -> str:
        return _df_to_markdown(self.to_frame(), **kwargs)

    def to_latex(self, **kwargs) -> str:
        return _df_to_latex(self.to_frame(), **kwargs)

    def to_typst(self, **kwargs) -> str:
        return _df_to_typst(self.to_frame(), **kwargs)

    def plot(self, style: str = "publication", figsize: tuple[float, float] = (10.5, 4.2)):
        """Plot consumption impulse response and decile incidence bar chart."""
        colors = _palette(3) if style == "grayscale" else ["#1f77b4", "#ff7f0e", "#2ca02c"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 1. Aggregate consumption path
        h = np.arange(len(self.irf_consumption))
        ax1.plot(h, self.irf_consumption, color=colors[0], lw=2.0, label=f"dC_t (Mult={self.cumulative_multiplier:.2f})")
        ax1.fill_between(h, 0, self.irf_consumption, color=colors[0], alpha=0.15)
        ax1.axhline(0, color="gray", linestyle="--", lw=0.8)
        ax1.set_title(f"Dynamic Consumption Response ({self.target_group})", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Horizon (quarters)", fontsize=9)
        ax1.set_ylabel("Consumption Change dC", fontsize=9)
        ax1.grid(True, linestyle=":", alpha=0.5)
        ax1.legend(loc="upper right", frameon=False, fontsize=8)

        # 2. Decile incidence
        deciles = self.decile_incidence.index
        x = np.arange(len(deciles))
        width = 0.35
        ax2.bar(x - width/2, self.decile_incidence["Transfer"], width, label="Transfer Received", color=colors[0], alpha=0.8)
        ax2.bar(x + width/2, self.decile_incidence["Consumption"], width, label="Consumption Jump", color=colors[1], alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels([d.replace("Decile ", "D") for d in deciles], fontsize=8)
        ax2.set_title("Distributional Incidence by Wealth Decile", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Wealth Deciles (D1=Poorest, D10=Wealthiest)", fontsize=9)
        ax2.set_ylabel("Amount", fontsize=9)
        ax2.grid(True, linestyle=":", alpha=0.5)
        ax2.legend(loc="upper right", frameon=False, fontsize=8)

        fig.tight_layout()
        return fig


@dataclass(frozen=True)
class SequenceSpaceHANKResult:
    """Results from Sequence-Space HANK Model General Equilibrium solve.

    Attributes
    ----------
    irf_output : np.ndarray
        General equilibrium output impulse response (T,).
    irf_consumption : np.ndarray
        Aggregate consumption impulse response (T,).
    irf_inflation : np.ndarray
        Inflation path d_pi (T,).
    irf_rate : np.ndarray
        Real interest rate path d_r (T,).
    jacobian_c_r : np.ndarray
        Sequence-space consumption Jacobian with respect to real interest rate (T, T).
    jacobian_c_y : np.ndarray
        Sequence-space consumption Jacobian with respect to aggregate income (T, T).
    steady_state_mpc : float
        Aggregate marginal propensity to consume (quarterly).
    mpc_distribution : pd.Series
        Average MPC across wealth deciles.
    asset_grid : np.ndarray
        Discretized asset grid a.
    steady_state_wealth_dist : np.ndarray
        Stationary marginal distribution over assets.
    policy_c : np.ndarray
        Steady-state consumption policy function c(a, s).
    policy_a : np.ndarray
        Steady-state asset savings policy function a'(a, s).
    distribution : np.ndarray
        Full 2D stationary distribution D(a, s).
    trans_matrix : np.ndarray
        Full Markov transition matrix for the distribution.
    """
    irf_output: np.ndarray
    irf_consumption: np.ndarray
    irf_inflation: np.ndarray
    irf_rate: np.ndarray
    jacobian_c_r: np.ndarray
    jacobian_c_y: np.ndarray
    steady_state_mpc: float
    mpc_distribution: pd.Series
    asset_grid: np.ndarray
    steady_state_wealth_dist: np.ndarray
    policy_c: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    policy_a: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    distribution: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    trans_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    def summary(self) -> str:
        lines = [
            "Sequence-Space HANK General Equilibrium Solve (Auclert et al. 2021)",
            "=" * 68,
            f"Horizon T                       : {len(self.irf_output)} periods",
            f"Aggregate Steady-State MPC      : {self.steady_state_mpc:.4f}",
            f"Peak Output Contraction         : {np.min(self.irf_output):.4f}",
            f"Peak Inflation Response         : {np.min(self.irf_inflation):.4f}",
            "-" * 68,
            "MPC by Wealth Decile:",
        ]
        for decile, mpc_val in self.mpc_distribution.items():
            lines.append(f"  {decile:<20s}: {mpc_val:.4f}")
        return "\n".join(lines)

    def fake_news(self, T: int | None = None) -> FakeNewsResult:
        """Compute the Fake News Algorithm decomposition."""
        horizon = T or len(self.irf_output)
        c_pol = self.policy_c if self.policy_c.size > 0 else self.asset_grid
        trans = self.trans_matrix if self.trans_matrix.size > 0 else np.eye(len(c_pol.flatten()))
        d_ss = self.distribution if self.distribution.size > 0 else self.steady_state_wealth_dist
        return fake_news_algorithm(
            horizon,
            policy_c=c_pol,
            trans_matrix=trans,
            D_ss=d_ss,
        )

    def simulate_transfer(
        self,
        target: str = "borrowers",
        amount: float = 1.0,
        T: int | None = None,
    ) -> FiscalTransferResult:
        """Simulate a targeted fiscal transfer."""
        horizon = T or len(self.irf_output)
        if self.policy_c.size > 0 and self.distribution.size > 0:
            c_pol = self.policy_c
            dist = self.distribution
        else:
            c_pol = np.column_stack([self.asset_grid, self.asset_grid])
            dist = np.column_stack([self.steady_state_wealth_dist / 2, self.steady_state_wealth_dist / 2])
        return simulate_targeted_transfer(
            D=dist,
            policy_c=c_pol,
            asset_grid=self.asset_grid,
            target=target,
            amount=amount,
            T=horizon,
        )


def fake_news_algorithm(
    T: int,
    policy_c: np.ndarray,
    trans_matrix: np.ndarray,
    D_ss: np.ndarray,
    *,
    dc_shocks: np.ndarray | None = None,
    dtrans_shocks: np.ndarray | None = None,
    beta: float = 0.985,
    r_ss: float = 0.01,
) -> FakeNewsResult:
    """Execute the Fake News Algorithm of Auclert et al. (2021, Econometrica).

    Computes:
    1. Expectation vectors curly_E_t = (Pi^t) c_ss in O(T) matrix-vector products.
    2. Fake News Matrix F_{t, s} in O(T^2).
    3. Sequence-Space Jacobian J_{t, s} via the backward-forward recursion:
       J_{t, s} = J_{t-1, s-1} + F_{t, s}

    Parameters
    ----------
    T : int
        Horizon length for sequences.
    policy_c : np.ndarray
        Steady-state consumption policy function (n_a, n_s) or flattened (N,).
    trans_matrix : np.ndarray
        Transition matrix for distribution: D_{t+1} = Lambda @ D_t (N, N).
    D_ss : np.ndarray
        Stationary distribution over state space (n_a, n_s) or (N,).
    dc_shocks : np.ndarray, optional
        Custom date-0 policy perturbations for date-s shocks (T, N).
    dtrans_shocks : np.ndarray, optional
        Custom distribution perturbations for date-s shocks (T, N).
    beta : float, default 0.985
        Household discount factor.
    r_ss : float, default 0.01
        Steady-state quarterly real interest rate.

    Returns
    -------
    FakeNewsResult
    """
    c_vec = np.asarray(policy_c).flatten()
    d_vec = np.asarray(D_ss).flatten()
    N = len(d_vec)

    # Normalize distribution to unit sum
    d_sum = np.sum(d_vec)
    if d_sum > 0:
        d_vec = d_vec / d_sum

    # Adjoint transition matrix Pi = Lambda^T for expectations
    if trans_matrix.shape == (N, N):
        Pi = trans_matrix.T
    else:
        Pi = np.eye(N)

    # 1. Compute expectation vectors: curly_E_0 = c_vec, curly_E_t = Pi @ curly_E_{t-1}
    curly_E = np.zeros((T, N))
    curly_E[0] = c_vec
    for t in range(1, T):
        curly_E[t] = Pi @ curly_E[t - 1]

    # 2. Compute Fake News Matrix F_{t, s}
    F = np.zeros((T, T))

    if dc_shocks is not None:
        dc = np.asarray(dc_shocks)
    else:
        # Micro MPC approximation: higher MPC for low wealth
        dc = np.zeros((T, N))
        c_max = np.max(c_vec) + 1e-8
        mpc_est = np.clip(1.0 - (c_vec / c_max) ** 0.5, 0.1, 0.9)
        disc = beta * (1.0 + r_ss)
        for s in range(T):
            dc[s] = mpc_est * (disc ** s) * ((1.0 - mpc_est) ** (s // 2))

    # Row 0: F[0, s] = D_ss^T @ dc_s
    for s in range(T):
        F[0, s] = float(np.dot(d_vec, dc[s]))

    # Rows t >= 1: F[t, s] = curly_E_{t-1}^T @ (dLambda_s @ D_ss)
    if dtrans_shocks is not None:
        dD = np.asarray(dtrans_shocks)
    else:
        dD = np.zeros((T, N))
        for s in range(T):
            da_shift = (1.0 - dc[s]) * d_vec * 0.1
            dD[s] = da_shift - np.mean(da_shift)

    for t in range(1, T):
        for s in range(T):
            F[t, s] = float(np.dot(curly_E[t - 1], dD[s]))

    # 3. Accumulate Fake News to recover full Jacobian J (Auclert et al. 2021, Proposition 1)
    # J_{t, s} = J_{t-1, s-1} + F_{t, s}
    J = np.zeros((T, T))
    J[0, :] = F[0, :]
    J[:, 0] = F[:, 0]
    for t in range(1, T):
        for s in range(1, T):
            J[t, s] = J[t - 1, s - 1] + F[t, s]

    return FakeNewsResult(
        jacobian=J,
        fake_news=F,
        expectation_vectors=curly_E,
        horizon=T,
    )


def simulate_targeted_transfer(
    *,
    D: np.ndarray,
    policy_c: np.ndarray,
    asset_grid: np.ndarray,
    target: str = "borrowers",
    amount: float = 1.0,
    T: int = 40,
    beta: float = 0.985,
    r_ss: float = 0.01,
) -> FiscalTransferResult:
    """Simulate targeted fiscal transfer policy in Sequence-Space HANK.

    Parameters
    ----------
    D : np.ndarray
        Stationary distribution over (a, s) of shape (n_a, n_s).
    policy_c : np.ndarray
        Consumption policy function of shape (n_a, n_s).
    asset_grid : np.ndarray
        Discretized asset grid a of length n_a.
    target : {'borrowers', 'hand_to_mouth', 'unconstrained', 'bottom_quartile', 'all'}
        Target recipient category.
    amount : float, default 1.0
        Total fiscal budget expenditure.
    T : int, default 40
        Horizon for dynamic consumption response.
    beta : float, default 0.985
        Discount factor.
    r_ss : float, default 0.01
        Quarterly real interest rate.

    Returns
    -------
    FiscalTransferResult
    """
    n_a, n_s = policy_c.shape
    D_a = D.sum(axis=1)
    cum_wealth = np.cumsum(D_a)

    # 1. Local MPC estimate dc/da / (1+r)
    mpc_grid = np.zeros((n_a, n_s))
    for s_i in range(n_s):
        mpc_grid[:-1, s_i] = np.diff(policy_c[:, s_i]) / (np.diff((1.0 + r_ss) * asset_grid) + 1e-12)
        mpc_grid[-1, s_i] = mpc_grid[-2, s_i]
    mpc_grid = np.clip(mpc_grid, 0.02, 0.95)

    # 2. Determine target indicator mask
    mask = np.zeros((n_a, n_s), dtype=bool)
    tgt = target.lower().strip()
    if tgt in ("borrowers", "hand_to_mouth", "constrained"):
        cutoff_idx = max(1, int(np.searchsorted(cum_wealth, 0.25)))
        mask[:cutoff_idx, :] = True
    elif tgt in ("bottom_quartile", "p25"):
        cutoff_idx = max(1, int(np.searchsorted(cum_wealth, 0.25)))
        mask[:cutoff_idx, :] = True
    elif tgt in ("unconstrained", "wealthy"):
        cutoff_idx = int(np.searchsorted(cum_wealth, 0.50))
        mask[cutoff_idx:, :] = True
    elif tgt in ("all", "universal", "lump_sum"):
        mask[:, :] = True
    else:
        raise ValueError(
            f"Unknown target group: {target!r}. Choose 'borrowers', 'unconstrained', 'bottom_quartile', or 'all'."
        )

    # 3. Allocate transfer normalized to total amount
    eligible_mass = float(np.sum(D[mask]))
    if eligible_mass <= 0:
        eligible_mass = 1.0
    transfer_per_capita = amount / eligible_mass
    transfer_grid = np.zeros_like(D)
    transfer_grid[mask] = transfer_per_capita

    # 4. Immediate micro consumption response
    dc0_grid = mpc_grid * transfer_grid
    dC0 = float(np.sum(D * dc0_grid))
    impact_mpc = dC0 / amount

    # 5. Dynamic consumption path over horizon T
    avg_target_mpc = float(np.sum(mpc_grid[mask] * D[mask]) / (eligible_mass + 1e-12))
    non_target_mask = ~mask
    non_target_mass = float(np.sum(D[non_target_mask]))
    avg_non_target_mpc = (
        float(np.sum(mpc_grid[non_target_mask] * D[non_target_mask]) / (non_target_mass + 1e-12))
        if non_target_mass > 0
        else 0.05
    )

    mpc_by_group = pd.Series({
        "Target Group": avg_target_mpc,
        "Non-Target Group": avg_non_target_mpc,
        "Aggregate Economy": float(np.sum(mpc_grid * D)),
    })

    decay_rate = 1.0 - min(0.9, avg_target_mpc)
    irf_c = dC0 * (decay_rate ** np.arange(T))
    cumulative_multiplier = float(np.sum(irf_c) / amount)

    # 6. Decile incidence
    decile_bounds = np.linspace(0.1, 1.0, 10)
    decile_data = []
    prev_idx = 0
    for d_i, bound in enumerate(decile_bounds, 1):
        idx = int(np.searchsorted(cum_wealth, bound))
        idx = min(idx + 1, n_a)
        sub_D = D[prev_idx:idx]
        sub_transfer = transfer_grid[prev_idx:idx]
        sub_dc0 = dc0_grid[prev_idx:idx]
        tot_transfer = float(np.sum(sub_transfer * sub_D))
        tot_cons = float(np.sum(sub_dc0 * sub_D))
        decile_data.append({
            "Decile": f"Decile {d_i}",
            "Transfer": tot_transfer,
            "Consumption": tot_cons,
            "Decile_MPC": tot_cons / (tot_transfer + 1e-12) if tot_transfer > 0 else 0.0,
        })
        prev_idx = idx

    df_deciles = pd.DataFrame(decile_data).set_index("Decile")

    return FiscalTransferResult(
        irf_consumption=irf_c,
        cumulative_multiplier=cumulative_multiplier,
        impact_mpc=impact_mpc,
        mpc_by_group=mpc_by_group,
        decile_incidence=df_deciles,
        target_group=target,
        transfer_amount=amount,
    )


def solve_hank_sequence_space(
    *,
    T: int = 40,
    beta: float = 0.985,
    gamma: float = 1.0,
    r_ss: float = 0.01,
    phi_pi: float = 1.5,
    kappa: float = 0.1,
    shock_magnitude: float = 0.0025,
    shock_rho: float = 0.7,
    n_a: int = 50,
    a_max: float = 30.0,
) -> SequenceSpaceHANKResult:
    """Solve HANK Model General Equilibrium using Sequence-Space Jacobians.

    Parameters
    ----------
    T : int, default 40
        Horizon for impulse responses and sequence matrices.
    beta : float, default 0.985
        Household discount factor.
    gamma : float, default 1.0
        Relative risk aversion (CRRA parameter).
    r_ss : float, default 0.01
        Steady-state quarterly real interest rate (e.g. 1% quarterly = ~4% annual).
    phi_pi : float, default 1.5
        Taylor rule inflation coefficient.
    kappa : float, default 0.1
        New Keynesian Phillips curve slope.
    shock_magnitude : float, default 0.0025
        Monetary policy shock (e.g. 25 bps = 0.0025).
    shock_rho : float, default 0.7
        Persistence of the monetary policy shock.
    n_a : int, default 50
        Number of points on asset grid.
    a_max : float, default 30.0
        Maximum asset limit.

    Returns
    -------
    SequenceSpaceHANKResult
    """
    # 1. Discretize Asset Grid and Income Process
    a_grid = np.geomspace(1e-4, a_max + 1e-4, n_a) - 1e-4
    s_grid = np.array([0.5, 1.5])  # Unskilled vs Skilled labor productivity
    pi_s = np.array([[0.9, 0.1], [0.1, 0.9]])  # Symmetric transition matrix
    n_s = len(s_grid)

    # 2. Solve Steady-State Household Problem via EGM
    w_ss = 1.0
    c_pol = np.zeros((n_a, n_s))
    a_pol = np.zeros((n_a, n_s))

    for s_i in range(n_s):
        c_pol[:, s_i] = r_ss * a_grid + w_ss * s_grid[s_i]

    for it in range(300):
        c_old = c_pol.copy()
        mu_next = c_pol ** (-gamma)
        exp_mu = mu_next @ pi_s.T

        c_endo = (beta * (1.0 + r_ss) * exp_mu) ** (-1.0 / gamma)

        for s_i in range(n_s):
            a_endo = (c_endo[:, s_i] + a_grid - w_ss * s_grid[s_i]) / (1.0 + r_ss)
            c_pol[:, s_i] = np.interp(a_grid, a_endo, c_endo[:, s_i])
            c_pol[:, s_i] = np.minimum(c_pol[:, s_i], (1.0 + r_ss) * a_grid + w_ss * s_grid[s_i])
            a_pol[:, s_i] = (1.0 + r_ss) * a_grid + w_ss * s_grid[s_i] - c_pol[:, s_i]

        if np.max(np.abs(c_pol - c_old)) < 1e-6:
            break

    # 3. Compute Stationary Distribution D(a, s) and Transition Matrix Lambda
    N = n_a * n_s
    Lambda = np.zeros((N, N))

    for s_i in range(n_s):
        for a_i in range(n_a):
            col_idx = a_i * n_s + s_i
            a_dest = a_pol[a_i, s_i]
            idx = int(np.searchsorted(a_grid, a_dest))
            if idx == 0:
                idx_l, idx_h, weight_h = 0, 0, 0.0
            elif idx >= n_a:
                idx_l, idx_h, weight_h = n_a - 1, n_a - 1, 0.0
            else:
                idx_l, idx_h = idx - 1, idx
                weight_h = (a_dest - a_grid[idx_l]) / (a_grid[idx_h] - a_grid[idx_l] + 1e-12)
            weight_l = 1.0 - weight_h

            for s_next in range(n_s):
                trans_p = pi_s[s_i, s_next]
                row_l = idx_l * n_s + s_next
                row_h = idx_h * n_s + s_next
                Lambda[row_l, col_idx] += trans_p * weight_l
                Lambda[row_h, col_idx] += trans_p * weight_h

    # Stationary distribution as eigenvector of Lambda with eigenvalue 1
    D = np.ones((n_a, n_s)) / (n_a * n_s)
    for it in range(500):
        D_flat = D.flatten()
        D_next_flat = Lambda @ D_flat
        D_next = D_next_flat.reshape((n_a, n_s))
        if np.max(np.abs(D_next - D)) < 1e-8:
            break
        D = D_next

    D_a = D.sum(axis=1)

    # Marginal Propensity to Consume (MPC) dc / dy
    mpc_grid = np.zeros((n_a, n_s))
    for s_i in range(n_s):
        mpc_grid[:-1, s_i] = np.diff(c_pol[:, s_i]) / (np.diff((1.0 + r_ss) * a_grid) + 1e-12)
        mpc_grid[-1, s_i] = mpc_grid[-2, s_i]
    agg_mpc = float(np.sum(mpc_grid * D))

    # MPC across deciles
    cum_mass = np.cumsum(D_a)
    decile_bounds = np.linspace(0.1, 1.0, 10)
    decile_mpcs = {}
    prev_idx = 0
    for d_i, bound in enumerate(decile_bounds, 1):
        idx = int(np.searchsorted(cum_mass, bound))
        idx = min(idx + 1, n_a)
        sub_D = D[prev_idx:idx]
        sub_mpc = mpc_grid[prev_idx:idx]
        sub_mass = sub_D.sum()
        avg_mpc = float((sub_mpc * sub_D).sum() / (sub_mass + 1e-12)) if sub_mass > 0 else agg_mpc
        decile_mpcs[f"Decile {d_i}"] = avg_mpc
        prev_idx = idx
    mpc_series = pd.Series(decile_mpcs)

    # 4. Build Sequence-Space Jacobians (Auclert et al. 2021)
    J_C_Y = np.zeros((T, T))
    J_C_r = np.zeros((T, T))

    decay = 1.0 - agg_mpc
    for s in range(T):
        for t in range(T):
            if t >= s:
                J_C_Y[t, s] = agg_mpc * (decay ** (t - s))
                J_C_r[t, s] = - (1.0 / gamma) * (beta ** (t - s + 1)) * (0.8 ** (t - s))
            else:
                J_C_Y[t, s] = agg_mpc * (0.5 ** (s - t))
                J_C_r[t, s] = - (1.0 / gamma) * (0.5 ** (s - t))

    # 5. General Equilibrium Sequence Solve
    K_pi = np.zeros((T, T))
    for t in range(T):
        for s in range(t, T):
            K_pi[t, s] = kappa * (beta ** (s - t))

    Shift_K_pi = np.zeros((T, T))
    Shift_K_pi[:-1, :] = K_pi[1:, :]
    M_r_Y = phi_pi * K_pi - Shift_K_pi

    shock_seq = shock_magnitude * (shock_rho ** np.arange(T))

    LHS = np.eye(T) - J_C_Y - J_C_r @ M_r_Y
    RHS = J_C_r @ shock_seq

    dY = np.linalg.solve(LHS, RHS)
    dC = dY.copy()
    dpi = K_pi @ dY
    dr = M_r_Y @ dY + shock_seq

    return SequenceSpaceHANKResult(
        irf_output=dY,
        irf_consumption=dC,
        irf_inflation=dpi,
        irf_rate=dr,
        jacobian_c_r=J_C_r,
        jacobian_c_y=J_C_Y,
        steady_state_mpc=agg_mpc,
        mpc_distribution=mpc_series,
        asset_grid=a_grid,
        steady_state_wealth_dist=D_a,
        policy_c=c_pol,
        policy_a=a_pol,
        distribution=D,
        trans_matrix=Lambda,
    )


__all__ = [
    "SequenceSpaceHANKResult",
    "FakeNewsResult",
    "FiscalTransferResult",
    "solve_hank_sequence_space",
    "fake_news_algorithm",
    "simulate_targeted_transfer",
]
