"""Monetary & Macroprudential Transmission Simulator (HANK vs RANK).

Compares monetary policy and balance-sheet transmission between:
1. Heterogeneous-Agent New Keynesian (HANK): Incomplete markets, uninsurable
   idiosyncratic risk, borrowing constraints, and endogenous MPC distribution across deciles.
2. Representative-Agent New Keynesian (RANK): Complete markets, permanent income,
   uniform Euler equation.

Implements the Kaplan-Moll-Violante (2018) transmission decomposition:
    dC_total = dC_direct (J^{C,r} dr) + dC_indirect (J^{C,Y} dY)

References:
    Kaplan, G., Moll, B., and Violante, G. L. (2018). "Monetary Policy According to HANK."
    American Economic Review, 108(3), 697–743.
    Auclert, A., Bardóczy, B., Rognlie, M., and Straub, L. (2021). "Using the Sequence-Space
    Jacobian to Solve and Estimate Heterogeneous-Agent Models." Econometrica, 89(6), 3115–3148.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

import numpy as np
import pandas as pd

from puremacro.models.hank_sequence_space import (
    SequenceSpaceHANKResult,
    _ge_matrices,
    simulate_targeted_transfer,
    solve_hank_sequence_space,
)
from puremacro.reports import _df_to_latex, _df_to_markdown, _df_to_typst


@dataclass(frozen=True)
class MonetaryTransmissionResult:
    """Comparative HANK vs RANK Monetary & Macroprudential Transmission Result.

    Attributes
    ----------
    horizon : int
        Simulation horizon T in quarters.
    shock_type : str
        Type of policy shock: 'rate' or 'balance_sheet'.
    shock_magnitude : float
        Magnitude of policy shock (e.g. 0.0025 for 25 bps interest rate hike).
    shock_rho : float
        Persistence parameter rho of the policy shock.
    irf_output_hank : np.ndarray
        HANK general equilibrium output impulse response (T,).
    irf_output_rank : np.ndarray
        RANK general equilibrium output impulse response (T,).
    irf_consumption_hank : np.ndarray
        HANK consumption impulse response (T,).
    irf_consumption_rank : np.ndarray
        RANK consumption impulse response (T,).
    irf_inflation_hank : np.ndarray
        HANK inflation impulse response (T,).
    irf_inflation_rank : np.ndarray
        RANK inflation impulse response (T,).
    irf_rate_hank : np.ndarray
        HANK ex-ante real interest rate path (T,).
    irf_rate_rank : np.ndarray
        RANK ex-ante real interest rate path (T,).
    mpc_deciles_hank : pd.Series
        HANK marginal propensity to consume across 10 wealth deciles.
    mpc_deciles_rank : pd.Series
        RANK marginal propensity to consume across 10 wealth deciles (uniform).
    aggregate_mpc_hank : float
        Aggregate quarterly MPC in HANK.
    aggregate_mpc_rank : float
        Aggregate quarterly MPC in RANK (1 - beta).
    direct_channel_hank : np.ndarray
        KMV Direct channel (intertemporal substitution) in HANK: J^{C,r} @ dr (T,).
    indirect_channel_hank : np.ndarray
        KMV Indirect channel (GE labor income) in HANK: J^{C,Y} @ dY (T,).
    direct_channel_rank : np.ndarray
        KMV Direct channel in RANK: J^{C,r}_RANK @ dr (T,).
    indirect_channel_rank : np.ndarray
        KMV Indirect channel in RANK: zeros(T,).
    indirect_share_hank : float
        Percentage of initial consumption response driven by GE indirect channel in HANK.
    indirect_share_rank : float
        Percentage of initial consumption response driven by GE indirect channel in RANK (0.0%).
    beta : float
        Household discount factor.
    gamma : float
        Relative risk aversion (CRRA parameter).
    r_ss : float
        Steady-state quarterly real interest rate.
    phi_pi : float
        Taylor rule inflation response coefficient.
    kappa : float
        New Keynesian Phillips curve slope.
    """

    horizon: int
    shock_type: str
    shock_magnitude: float
    shock_rho: float

    irf_output_hank: np.ndarray
    irf_output_rank: np.ndarray
    irf_consumption_hank: np.ndarray
    irf_consumption_rank: np.ndarray
    irf_inflation_hank: np.ndarray
    irf_inflation_rank: np.ndarray
    irf_rate_hank: np.ndarray
    irf_rate_rank: np.ndarray

    mpc_deciles_hank: pd.Series
    mpc_deciles_rank: pd.Series
    aggregate_mpc_hank: float
    aggregate_mpc_rank: float

    direct_channel_hank: np.ndarray
    indirect_channel_hank: np.ndarray
    direct_channel_rank: np.ndarray
    indirect_channel_rank: np.ndarray
    indirect_share_hank: float
    indirect_share_rank: float

    beta: float = 0.985
    gamma: float = 1.0
    r_ss: float = 0.01
    phi_pi: float = 1.5
    kappa: float = 0.1

    def peak_responses(self) -> dict[str, float]:
        """Return peak absolute responses across macroeconomic variables."""
        return {
            "output_hank": float(self.irf_output_hank[np.argmax(np.abs(self.irf_output_hank))]),
            "output_rank": float(self.irf_output_rank[np.argmax(np.abs(self.irf_output_rank))]),
            "inflation_hank": float(self.irf_inflation_hank[np.argmax(np.abs(self.irf_inflation_hank))]),
            "inflation_rank": float(self.irf_inflation_rank[np.argmax(np.abs(self.irf_inflation_rank))]),
            "consumption_hank": float(self.irf_consumption_hank[np.argmax(np.abs(self.irf_consumption_hank))]),
            "consumption_rank": float(self.irf_consumption_rank[np.argmax(np.abs(self.irf_consumption_rank))]),
        }

    def __repr__(self) -> str:
        return (
            f"MonetaryTransmissionResult("
            f"shock_type={self.shock_type!r}, "
            f"magnitude={self.shock_magnitude:+.4f}, "
            f"T={self.horizon}, "
            f"mpc_hank={self.aggregate_mpc_hank:.4f}, "
            f"mpc_rank={self.aggregate_mpc_rank:.4f}, "
            f"indirect_share_hank={self.indirect_share_hank:.1f}%)"
        )

    def summary(self) -> str:
        """Text summary comparing HANK and RANK transmission mechanisms."""
        pk = self.peak_responses()
        pk_y_hank = pk["output_hank"]
        pk_y_rank = pk["output_rank"]
        pk_pi_hank = pk["inflation_hank"]
        pk_pi_rank = pk["inflation_rank"]

        lines = [
            "Monetary & Macroprudential Transmission: HANK vs RANK",
            "=" * 72,
            f"Shock Type                      : {self.shock_type.upper()} (magnitude = {self.shock_magnitude:+.4f}, rho = {self.shock_rho:.2f})",
            f"Horizon T                       : {self.horizon} quarters",
            "-" * 72,
            f"{'Metric':<34s} | {'HANK':>16s} | {'RANK':>16s}",
            "-" * 72,
            f"{'Peak Output Response (dY)':<34s} | {pk_y_hank:>+16.6f} | {pk_y_rank:>+16.6f}",
            f"{'Peak Inflation Response (dpi)':<34s} | {pk_pi_hank:>+16.6f} | {pk_pi_rank:>+16.6f}",
            f"{'Aggregate Quarterly MPC':<34s} | {self.aggregate_mpc_hank:>16.4f} | {self.aggregate_mpc_rank:>16.4f}",
            f"{'Decile 1 MPC (Poorest 10%)':<34s} | {float(self.mpc_deciles_hank.iloc[0]):>16.4f} | {float(self.mpc_deciles_rank.iloc[0]):>16.4f}",
            f"{'Decile 10 MPC (Richest 10%)':<34s} | {float(self.mpc_deciles_hank.iloc[-1]):>16.4f} | {float(self.mpc_deciles_rank.iloc[-1]):>16.4f}",
            "-" * 72,
            "Kaplan-Moll-Violante (2018) Consumption Decomposition (t = 0):",
            f"{'  Total Consumption Response (dC)':<34s} | {self.irf_consumption_hank[0]:>+16.6f} | {self.irf_consumption_rank[0]:>+16.6f}",
            f"{'  Direct Channel (J^{C,r} dr)':<34s} | {self.direct_channel_hank[0]:>+16.6f} | {self.direct_channel_rank[0]:>+16.6f}",
            f"{'  Indirect Channel (J^{C,Y} dY)':<34s} | {self.indirect_channel_hank[0]:>+16.6f} | {self.indirect_channel_rank[0]:>+16.6f}",
            f"{'  Indirect Share (%)':<34s} | {self.indirect_share_hank:>15.2f}% | {self.indirect_share_rank:>15.2f}%",
            "=" * 72,
        ]
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """Return quarterly simulation paths as a pandas DataFrame."""
        return pd.DataFrame(
            {
                "output_hank": self.irf_output_hank,
                "output_rank": self.irf_output_rank,
                "consumption_hank": self.irf_consumption_hank,
                "consumption_rank": self.irf_consumption_rank,
                "inflation_hank": self.irf_inflation_hank,
                "inflation_rank": self.irf_inflation_rank,
                "rate_hank": self.irf_rate_hank,
                "rate_rank": self.irf_rate_rank,
                "direct_hank": self.direct_channel_hank,
                "indirect_hank": self.indirect_channel_hank,
                "direct_rank": self.direct_channel_rank,
                "indirect_rank": self.indirect_channel_rank,
            },
            index=pd.Index(np.arange(self.horizon), name="quarter"),
        )

    def to_markdown(self, index: bool = True, digits: int = 4, **kwargs: Any) -> str:
        """Format simulation paths as a GitHub-flavored Markdown table."""
        return _df_to_markdown(self.to_frame(), index=index, digits=digits, **kwargs)

    def to_latex(self, index: bool = True, digits: int = 4, **kwargs: Any) -> str:
        """Format simulation paths as a LaTeX tabular environment."""
        return _df_to_latex(self.to_frame(), index=index, digits=digits, **kwargs)

    def to_typst(self, index: bool = True, digits: int = 4, **kwargs: Any) -> str:
        """Format simulation paths as a Typst table."""
        return _df_to_typst(self.to_frame(), index=index, digits=digits, **kwargs)

    def plot(
        self,
        kind: str = "all",
        figsize: tuple[float, float] = (12.0, 8.5),
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Plot comparative HANK vs RANK transmission dynamics."""
        import matplotlib.pyplot as plt

        t = np.arange(self.horizon)

        if kind == "all":
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            # Panel 1: Output IRF
            ax1 = axes[0, 0]
            ax1.plot(t, self.irf_output_hank * 100.0, color="#1e40af", linewidth=2.0, label="HANK")
            ax1.plot(t, self.irf_output_rank * 100.0, color="#b91c1c", linestyle="--", linewidth=2.0, label="RANK")
            ax1.axhline(0, color="black", linestyle=":", alpha=0.6)
            ax1.set_title("Output Response (dY, %)")
            ax1.set_xlabel("Quarter")
            ax1.set_ylabel("% deviation")
            ax1.legend()
            ax1.grid(True, linestyle=":", alpha=0.5)

            # Panel 2: Inflation IRF
            ax2 = axes[0, 1]
            ax2.plot(t, self.irf_inflation_hank * 400.0, color="#1e40af", linewidth=2.0, label="HANK")
            ax2.plot(t, self.irf_inflation_rank * 400.0, color="#b91c1c", linestyle="--", linewidth=2.0, label="RANK")
            ax2.axhline(0, color="black", linestyle=":", alpha=0.6)
            ax2.set_title("Annualized Inflation Response (dpi, pp)")
            ax2.set_xlabel("Quarter")
            ax2.set_ylabel("pp")
            ax2.legend()
            ax2.grid(True, linestyle=":", alpha=0.5)

            # Panel 3: MPC by Decile
            ax3 = axes[1, 0]
            dec_x = np.arange(1, 11)
            ax3.bar(dec_x - 0.18, self.mpc_deciles_hank.values, width=0.36, color="#1e40af", label="HANK (Incomplete Markets)")
            ax3.bar(dec_x + 0.18, self.mpc_deciles_rank.values, width=0.36, color="#b91c1c", alpha=0.7, label="RANK (Representative Agent)")
            ax3.set_xticks(dec_x)
            ax3.set_xticklabels([f"D{d}" for d in dec_x])
            ax3.set_title("Marginal Propensity to Consume (MPC) by Wealth Decile")
            ax3.set_xlabel("Wealth Decile (D1 = Poorest, D10 = Richest)")
            ax3.set_ylabel("Quarterly MPC")
            ax3.legend()
            ax3.grid(True, linestyle=":", alpha=0.5, axis="y")

            # Panel 4: KMV Direct vs Indirect Channels in HANK
            ax4 = axes[1, 1]
            ax4.plot(t, self.irf_consumption_hank * 100.0, color="#111827", linewidth=2.5, label="Total dC (HANK)")
            ax4.plot(t, self.direct_channel_hank * 100.0, color="#2563eb", linestyle="-.", linewidth=1.8, label="Direct (Intertemp. Subst.)")
            ax4.plot(t, self.indirect_channel_hank * 100.0, color="#059669", linestyle="--", linewidth=1.8, label="Indirect (GE Labor Income)")
            ax4.axhline(0, color="black", linestyle=":", alpha=0.6)
            ax4.set_title(f"KMV (2018) Decomposition (HANK Indirect Share = {self.indirect_share_hank:.1f}%)")
            ax4.set_xlabel("Quarter")
            ax4.set_ylabel("% deviation")
            ax4.legend()
            ax4.grid(True, linestyle=":", alpha=0.5)

            fig.tight_layout()
            return fig

        elif kind == "irf":
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.get_figure()
            ax.plot(t, self.irf_output_hank * 100.0, color="#1e40af", linewidth=2.0, label="HANK Output")
            ax.plot(t, self.irf_output_rank * 100.0, color="#b91c1c", linestyle="--", linewidth=2.0, label="RANK Output")
            ax.plot(t, self.irf_consumption_hank * 100.0, color="#2563eb", linestyle=":", linewidth=1.8, label="HANK Consumption")
            ax.plot(t, self.irf_consumption_rank * 100.0, color="#f87171", linestyle=":", linewidth=1.8, label="RANK Consumption")
            ax.axhline(0, color="black", linestyle=":", alpha=0.6)
            ax.set_title("HANK vs RANK Impulse Responses")
            ax.set_xlabel("Quarter")
            ax.set_ylabel("% deviation")
            ax.legend()
            ax.grid(True, linestyle=":", alpha=0.5)
            fig.tight_layout()
            return fig

        elif kind == "mpc":
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.get_figure()
            dec_x = np.arange(1, 11)
            ax.bar(dec_x - 0.18, self.mpc_deciles_hank.values, width=0.36, color="#1e40af", label="HANK")
            ax.bar(dec_x + 0.18, self.mpc_deciles_rank.values, width=0.36, color="#b91c1c", alpha=0.7, label="RANK")
            ax.set_xticks(dec_x)
            ax.set_xticklabels([f"D{d}" for d in dec_x])
            ax.set_title("MPC Ladder by Wealth Decile")
            ax.set_xlabel("Wealth Decile")
            ax.set_ylabel("Quarterly MPC")
            ax.legend()
            ax.grid(True, linestyle=":", alpha=0.5, axis="y")
            fig.tight_layout()
            return fig

        elif kind == "kmv":
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.get_figure()
            ax.plot(t, self.irf_consumption_hank * 100.0, color="#111827", linewidth=2.5, label="Total dC (HANK)")
            ax.plot(t, self.direct_channel_hank * 100.0, color="#2563eb", linestyle="-.", linewidth=1.8, label="Direct Channel")
            ax.plot(t, self.indirect_channel_hank * 100.0, color="#059669", linestyle="--", linewidth=1.8, label="Indirect Channel")
            ax.axhline(0, color="black", linestyle=":", alpha=0.6)
            ax.set_title("KMV (2018) Transmission Decomposition (HANK)")
            ax.set_xlabel("Quarter")
            ax.set_ylabel("% deviation")
            ax.legend()
            ax.grid(True, linestyle=":", alpha=0.5)
            fig.tight_layout()
            return fig

        else:
            raise ValueError(f"Unknown plot kind: {kind!r}. Choose from 'all', 'irf', 'mpc', 'kmv'.")


class MonetaryTransmissionSimulator:
    """Monetary & Macroprudential Transmission Simulator comparing HANK and RANK.

    Solves the Sequence-Space general equilibrium for both models under identical
    monetary policy and balance-sheet shocks, extracting the Kaplan-Moll-Violante (2018)
    direct vs indirect decomposition and the heterogeneous MPC distribution.
    """

    def __init__(
        self,
        model: SequenceSpaceHANKResult | None = None,
        beta: float = 0.985,
        gamma: float = 1.0,
        r_ss: float = 0.01,
        phi_pi: float = 1.5,
        kappa: float = 0.1,
        n_a: int = 50,
        a_max: float = 30.0,
    ) -> None:
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.r_ss = float(r_ss)
        self.phi_pi = float(phi_pi)
        self.kappa = float(kappa)
        self.n_a = int(n_a)
        self.a_max = float(a_max)
        self._cached_hank: SequenceSpaceHANKResult | None = model

    @property
    def steady_state_mpc(self) -> float:
        """Steady-state aggregate quarterly MPC in HANK economy."""
        if self._cached_hank is not None:
            return float(self._cached_hank.steady_state_mpc)
        res = self._get_hank_model(T=40, shock_magnitude=0.0025, shock_rho=0.7)
        return float(res.steady_state_mpc)

    def _get_hank_model(self, T: int, shock_magnitude: float, shock_rho: float) -> SequenceSpaceHANKResult:
        """Obtain solved SequenceSpaceHANKResult with the required horizon and shock parameters."""
        if self._cached_hank is not None and len(self._cached_hank.irf_output) >= T:
            return self._cached_hank
        model = solve_hank_sequence_space(
            T=T,
            beta=self.beta,
            gamma=self.gamma,
            r_ss=self.r_ss,
            phi_pi=self.phi_pi,
            kappa=self.kappa,
            shock_magnitude=shock_magnitude,
            shock_rho=shock_rho,
            n_a=self.n_a,
            a_max=self.a_max,
        )
        self._cached_hank = model
        return model

    def simulate_rate_shock(
        self,
        magnitude: float = 0.0025,
        rho: float = 0.7,
        T: int = 40,
    ) -> MonetaryTransmissionResult:
        """Simulate an ex-ante interest rate shock (e.g. monetary policy tightening).

        Parameters
        ----------
        magnitude : float, default 0.0025
            Shock magnitude (0.0025 = 25 bps quarterly).
        rho : float, default 0.7
            Shock persistence parameter.
        T : int, default 40
            Simulation horizon in quarters.

        Returns
        -------
        MonetaryTransmissionResult
        """
        T = int(T)
        if T < 2:
            raise ValueError(f"Simulation horizon T must be at least 2 quarters, got {T}")
        mag = float(magnitude)
        if not np.isfinite(mag):
            raise ValueError(f"magnitude must be a finite float, got {magnitude!r}")
        r_pers = float(rho)
        if not (0.0 <= r_pers < 1.0):
            raise ValueError(f"Shock persistence rho must satisfy 0.0 <= rho < 1.0, got {rho}")

        hank_res = self._get_hank_model(T, mag, r_pers)

        # Truncate or obtain horizon T matrices
        J_C_r_hank = hank_res.jacobian_c_r[:T, :T]
        J_C_Y_hank = hank_res.jacobian_c_y[:T, :T]
        K_pi, M_r_Y = _ge_matrices(T, self.beta, self.kappa, self.phi_pi)
        shock_seq = float(magnitude) * (float(rho) ** np.arange(T))

        # 1. HANK Equilibrium Solve
        LHS_hank = np.eye(T) - J_C_Y_hank - J_C_r_hank @ M_r_Y
        dY_hank = np.linalg.solve(LHS_hank, J_C_r_hank @ shock_seq)
        dC_hank = dY_hank.copy()
        dpi_hank = K_pi @ dY_hank
        dr_hank = M_r_Y @ dY_hank + shock_seq

        # KMV Decomposition in HANK: dC = J^{C,r} dr + J^{C,Y} dY
        dC_direct_hank = J_C_r_hank @ dr_hank
        dC_indirect_hank = J_C_Y_hank @ dY_hank
        ind_share_hank = float((dC_indirect_hank[0] / dC_hank[0]) * 100.0) if abs(dC_hank[0]) > 1e-12 else 0.0

        # 2. RANK Model in Sequence Space
        # Under representative agent Euler equation: c_t = E_t c_{t+1} - 1/gamma dr_t
        # c_t = -1/gamma sum_{s=t}^{T-1} dr_s
        # Jacobian J^{C,r}_RANK[t, s] = -1/gamma for s >= t, 0 for s < t. J^{C,Y}_RANK = 0.
        J_C_r_rank = np.zeros((T, T), dtype=float)
        for t_idx in range(T):
            J_C_r_rank[t_idx, t_idx:] = -1.0 / self.gamma

        LHS_rank = np.eye(T) - J_C_r_rank @ M_r_Y
        dY_rank = np.linalg.solve(LHS_rank, J_C_r_rank @ shock_seq)
        dC_rank = dY_rank.copy()
        dpi_rank = K_pi @ dY_rank
        dr_rank = M_r_Y @ dY_rank + shock_seq

        # KMV Decomposition in RANK: Indirect channel is identically zero
        dC_direct_rank = J_C_r_rank @ dr_rank
        dC_indirect_rank = np.zeros(T, dtype=float)
        ind_share_rank = 0.0

        # MPC across Deciles
        mpc_hank = hank_res.mpc_distribution.copy()
        mpc_rank_val = float(1.0 - self.beta)
        mpc_rank = pd.Series(
            {f"Decile {d + 1}": mpc_rank_val for d in range(10)},
            name="mpc_rank",
        )

        return MonetaryTransmissionResult(
            horizon=T,
            shock_type="rate",
            shock_magnitude=magnitude,
            shock_rho=rho,
            irf_output_hank=dY_hank,
            irf_output_rank=dY_rank,
            irf_consumption_hank=dC_hank,
            irf_consumption_rank=dC_rank,
            irf_inflation_hank=dpi_hank,
            irf_inflation_rank=dpi_rank,
            irf_rate_hank=dr_hank,
            irf_rate_rank=dr_rank,
            mpc_deciles_hank=mpc_hank,
            mpc_deciles_rank=mpc_rank,
            aggregate_mpc_hank=float(hank_res.steady_state_mpc),
            aggregate_mpc_rank=mpc_rank_val,
            direct_channel_hank=dC_direct_hank,
            indirect_channel_hank=dC_indirect_hank,
            direct_channel_rank=dC_direct_rank,
            indirect_channel_rank=dC_indirect_rank,
            indirect_share_hank=ind_share_hank,
            indirect_share_rank=ind_share_rank,
            beta=self.beta,
            gamma=self.gamma,
            r_ss=self.r_ss,
            phi_pi=self.phi_pi,
            kappa=self.kappa,
        )

    def simulate_balance_sheet_intervention(
        self,
        amount: float = 1.0,
        target: str = "borrowers",
        T: int = 40,
    ) -> MonetaryTransmissionResult:
        """Simulate a targeted balance-sheet or fiscal transfer intervention.

        Parameters
        ----------
        amount : float, default 1.0
            Total transfer outlay in model income units.
        target : str, default 'borrowers'
            Target group: 'borrowers' (hand-to-mouth), 'unconstrained' (wealthy), 'all'.
        T : int, default 40
            Horizon in quarters.
        """
        T = int(T)
        if T < 2:
            raise ValueError(f"Simulation horizon T must be at least 2 quarters, got {T}")
        amt = float(amount)
        if not np.isfinite(amt) or amt <= 0.0:
            raise ValueError(f"Transfer amount must be a positive finite float, got {amount!r}")

        valid_targets = {"borrowers", "hand_to_mouth", "bottom_quartile", "unconstrained", "wealthy", "all", "universal"}
        if isinstance(target, str) and target.lower().strip() not in valid_targets:
            raise ValueError(
                f"Unknown target group: {target!r}. Choose from {sorted(valid_targets)} or a sequence of decile ints."
            )

        hank_res = self._get_hank_model(T, 0.0025, 0.7)

        # Simulate targeted transfer in HANK
        tf_res = hank_res.simulate_transfer(target=target, amount=amt, T=T)
        dC_hank = tf_res.irf_consumption[:T].copy()
        dY_hank = dC_hank.copy()

        K_pi, M_r_Y = _ge_matrices(T, self.beta, self.kappa, self.phi_pi)
        dpi_hank = K_pi @ dY_hank
        dr_hank = M_r_Y @ dY_hank

        J_C_r_hank = hank_res.jacobian_c_r[:T, :T]
        J_C_Y_hank = hank_res.jacobian_c_y[:T, :T]
        dC_direct_hank = J_C_r_hank @ dr_hank
        dC_indirect_hank = dC_hank - dC_direct_hank
        ind_share_hank = float((dC_indirect_hank[0] / dC_hank[0]) * 100.0) if abs(dC_hank[0]) > 1e-12 else 0.0

        # In RANK: Under representative agent permanent income, cash windfall has uniform MPC (1 - beta)
        mpc_rank_val = float(1.0 - self.beta)
        dC_rank = np.zeros(T, dtype=float)
        # Decay at savings rate beta
        dC_rank[0] = amt * mpc_rank_val
        for t_idx in range(1, T):
            dC_rank[t_idx] = dC_rank[t_idx - 1] * (1.0 - mpc_rank_val)

        dY_rank = dC_rank.copy()
        dpi_rank = K_pi @ dY_rank
        dr_rank = M_r_Y @ dY_rank

        dC_direct_rank = dC_rank.copy()
        dC_indirect_rank = np.zeros(T, dtype=float)
        ind_share_rank = 0.0

        mpc_hank = hank_res.mpc_distribution.copy()
        mpc_rank = pd.Series(
            {f"Decile {d + 1}": mpc_rank_val for d in range(10)},
            name="mpc_rank",
        )

        return MonetaryTransmissionResult(
            horizon=T,
            shock_type="balance_sheet",
            shock_magnitude=amt,
            shock_rho=0.0,
            irf_output_hank=dY_hank,
            irf_output_rank=dY_rank,
            irf_consumption_hank=dC_hank,
            irf_consumption_rank=dC_rank,
            irf_inflation_hank=dpi_hank,
            irf_inflation_rank=dpi_rank,
            irf_rate_hank=dr_hank,
            irf_rate_rank=dr_rank,
            mpc_deciles_hank=mpc_hank,
            mpc_deciles_rank=mpc_rank,
            aggregate_mpc_hank=float(hank_res.steady_state_mpc),
            aggregate_mpc_rank=mpc_rank_val,
            direct_channel_hank=dC_direct_hank,
            indirect_channel_hank=dC_indirect_hank,
            direct_channel_rank=dC_direct_rank,
            indirect_channel_rank=dC_indirect_rank,
            indirect_share_hank=ind_share_hank,
            indirect_share_rank=ind_share_rank,
            beta=self.beta,
            gamma=self.gamma,
            r_ss=self.r_ss,
            phi_pi=self.phi_pi,
            kappa=self.kappa,
        )

    def simulate_transmission(
        self,
        shock_type: str = "rate",
        shock_path: Sequence[float] | np.ndarray | None = None,
        magnitude: float = 0.0025,
        rho: float = 0.7,
        amount: float = 1.0,
        target: str = "borrowers",
        T: int = 40,
    ) -> MonetaryTransmissionResult:
        """General interface to simulate monetary or balance-sheet transmission."""
        st = shock_type.lower().strip()
        if st in ("rate", "interest_rate", "monetary"):
            if shock_path is not None:
                shock_arr = np.asarray(shock_path, dtype=float)
                T_actual = len(shock_arr)
                if T_actual < 2:
                    raise ValueError(f"Custom shock_path must have length >= 2, got {T_actual}")
                if not np.all(np.isfinite(shock_arr)):
                    raise ValueError("shock_path contains NaN or infinite values.")

                hank_res = self._get_hank_model(T_actual, float(shock_arr[0]), 0.7)
                J_C_r_hank = hank_res.jacobian_c_r[:T_actual, :T_actual]
                J_C_Y_hank = hank_res.jacobian_c_y[:T_actual, :T_actual]
                K_pi, M_r_Y = _ge_matrices(T_actual, self.beta, self.kappa, self.phi_pi)

                LHS_hank = np.eye(T_actual) - J_C_Y_hank - J_C_r_hank @ M_r_Y
                dY_hank = np.linalg.solve(LHS_hank, J_C_r_hank @ shock_arr)
                dC_hank = dY_hank.copy()
                dpi_hank = K_pi @ dY_hank
                dr_hank = M_r_Y @ dY_hank + shock_arr

                dC_direct_hank = J_C_r_hank @ dr_hank
                dC_indirect_hank = J_C_Y_hank @ dY_hank
                ind_share_hank = float((dC_indirect_hank[0] / dC_hank[0]) * 100.0) if abs(dC_hank[0]) > 1e-12 else 0.0

                J_C_r_rank = np.zeros((T_actual, T_actual), dtype=float)
                for t_idx in range(T_actual):
                    J_C_r_rank[t_idx, t_idx:] = -1.0 / self.gamma

                LHS_rank = np.eye(T_actual) - J_C_r_rank @ M_r_Y
                dY_rank = np.linalg.solve(LHS_rank, J_C_r_rank @ shock_arr)
                dC_rank = dY_rank.copy()
                dpi_rank = K_pi @ dY_rank
                dr_rank = M_r_Y @ dY_rank + shock_arr

                dC_direct_rank = J_C_r_rank @ dr_rank
                dC_indirect_rank = np.zeros(T_actual, dtype=float)
                mpc_rank_val = float(1.0 - self.beta)
                mpc_rank = pd.Series({f"Decile {d + 1}": mpc_rank_val for d in range(10)})

                effective_rho = float(rho) if rho is not None else float("nan")

                return MonetaryTransmissionResult(
                    horizon=T_actual,
                    shock_type="rate",
                    shock_magnitude=float(shock_arr[0]),
                    shock_rho=effective_rho,
                    irf_output_hank=dY_hank,
                    irf_output_rank=dY_rank,
                    irf_consumption_hank=dC_hank,
                    irf_consumption_rank=dC_rank,
                    irf_inflation_hank=dpi_hank,
                    irf_inflation_rank=dpi_rank,
                    irf_rate_hank=dr_hank,
                    irf_rate_rank=dr_rank,
                    mpc_deciles_hank=hank_res.mpc_distribution.copy(),
                    mpc_deciles_rank=mpc_rank,
                    aggregate_mpc_hank=float(hank_res.steady_state_mpc),
                    aggregate_mpc_rank=mpc_rank_val,
                    direct_channel_hank=dC_direct_hank,
                    indirect_channel_hank=dC_indirect_hank,
                    direct_channel_rank=dC_direct_rank,
                    indirect_channel_rank=dC_indirect_rank,
                    indirect_share_hank=ind_share_hank,
                    indirect_share_rank=0.0,
                    beta=self.beta,
                    gamma=self.gamma,
                    r_ss=self.r_ss,
                    phi_pi=self.phi_pi,
                    kappa=self.kappa,
                )
            return self.simulate_rate_shock(magnitude=magnitude, rho=rho, T=T)
        elif st in ("balance_sheet", "transfer", "macroprudential"):
            return self.simulate_balance_sheet_intervention(amount=amount, target=target, T=T)
        else:
            raise ValueError(f"Unknown shock_type: {shock_type!r}. Choose 'rate' or 'balance_sheet'.")
