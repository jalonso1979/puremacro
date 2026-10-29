"""Result containers for puremacro.sovereign.

Defines frozen dataclasses for sovereign default, sudden-stop collateral
constraints, and debt restructuring counterfactuals. Each result implements
the presentation contract: ``summary()``, ``to_dataframe()``, ``to_frame()``,
``to_markdown()``, ``to_latex()``, ``to_typst()``, and ``plot()``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import matplotlib.figure

    from puremacro.sovereign.rules import FiscalRule


__all__ = [
    "SovereignDefaultResult",
    "FiscalSovereignResult",
    "SuddenStopResult",
    "RestructuringResult",
]


def _render_df(df: pd.DataFrame, fmt: str) -> str:
    """Render a DataFrame through shared reports helpers with fallback."""
    try:
        from puremacro.reports import _df_to_latex, _df_to_markdown, _df_to_typst

        if fmt == "markdown":
            return _df_to_markdown(df, index=False)
        if fmt == "latex":
            return _df_to_latex(df, index=False)
        if fmt == "typst":
            return _df_to_typst(df, index=False)
    except Exception:
        pass

    if fmt == "markdown":
        return df.to_markdown(index=False)
    if fmt == "latex":
        return df.to_latex(index=False)
    if fmt == "typst":
        return str(df.to_dict(orient="records"))
    raise ValueError(f"Unknown format {fmt!r}")


def _fmt_val(v: Any, digits: int = 4) -> Any:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return round(float(v), digits)
    return v


@dataclass(frozen=True)
class SovereignDefaultResult:
    """Result container for Arellano / Chatterjee-Eyigungor sovereign default models."""

    b_grid: np.ndarray
    y_grid: np.ndarray
    v_r: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    v_d: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    v_star: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    q_bond: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    default_prob: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    policy_b_prime: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    spread: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    spread_mean: float = 0.0
    spread_y_corr: float = 0.0
    bellman_residual: float = 0.0
    converged: bool = False
    v_repay: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    v_default: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    v_expected: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    policy_b: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    bond_price: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    dq_db: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    p_trans: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    maturity_param_delta: float = 1.0
    coupon_rate_kappa: float = 0.0
    taste_shock_scale: float = 0.0
    residual: float = 0.0
    iterations: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.v_repay) == 0 and len(self.v_r) > 0:
            object.__setattr__(self, "v_repay", self.v_r)
        elif len(self.v_r) == 0 and len(self.v_repay) > 0:
            object.__setattr__(self, "v_r", self.v_repay)

        if len(self.v_default) == 0 and len(self.v_d) > 0:
            object.__setattr__(self, "v_default", self.v_d)
        elif len(self.v_d) == 0 and len(self.v_default) > 0:
            object.__setattr__(self, "v_d", self.v_default)

        if len(self.v_expected) == 0 and len(self.v_star) > 0:
            object.__setattr__(self, "v_expected", self.v_star)
        elif len(self.v_star) == 0 and len(self.v_expected) > 0:
            object.__setattr__(self, "v_star", self.v_expected)

        if len(self.policy_b) == 0 and len(self.policy_b_prime) > 0:
            object.__setattr__(self, "policy_b", self.policy_b_prime)
        elif len(self.policy_b_prime) == 0 and len(self.policy_b) > 0:
            object.__setattr__(self, "policy_b_prime", self.policy_b)

        if len(self.bond_price) == 0 and len(self.q_bond) > 0:
            object.__setattr__(self, "bond_price", self.q_bond)
        elif len(self.q_bond) == 0 and len(self.bond_price) > 0:
            object.__setattr__(self, "q_bond", self.bond_price)

        if self.residual == 0.0 and self.bellman_residual != 0.0:
            object.__setattr__(self, "residual", self.bellman_residual)
        elif self.bellman_residual == 0.0 and self.residual != 0.0:
            object.__setattr__(self, "bellman_residual", self.residual)

    def summary(self) -> pd.DataFrame:
        """Return a structured summary DataFrame of the solution."""
        mean_spread = float(self.spread_mean)
        max_spread = float(np.max(self.spread))
        mean_price = float(np.mean(self.bond_price))
        min_price = float(np.min(self.bond_price))
        mean_pdef = float(np.mean(self.default_prob))
        max_pdef = float(np.max(self.default_prob))

        rows = [
            ("converged", self.converged),
            ("iterations", self.iterations),
            ("bellman_residual", self.bellman_residual),
            ("maturity_delta", self.maturity_param_delta),
            ("coupon_kappa", self.coupon_rate_kappa),
            ("taste_shock_sigma", self.taste_shock_scale),
            ("n_debt_points", len(self.b_grid)),
            ("n_income_points", len(self.y_grid)),
            ("debt_min", float(self.b_grid[0])),
            ("debt_max", float(self.b_grid[-1])),
            ("mean_bond_price", mean_price),
            ("min_bond_price", min_price),
            ("mean_spread_bps", mean_spread * 10000.0),
            ("max_spread_bps", max_spread * 10000.0),
            ("spread_y_corr", self.spread_y_corr),
            ("mean_default_prob", mean_pdef),
            ("max_default_prob", max_pdef),
            ("max_dq_db", float(np.max(self.dq_db))),
            ("min_dq_db", float(np.min(self.dq_db))),
        ]
        return pd.DataFrame(
            {"metric": [r[0] for r in rows], "value": [_fmt_val(r[1]) for r in rows]}
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Tabulate policy, price, spread, and value functions across state grid."""
        records = []
        n_b = len(self.b_grid)
        n_y = len(self.y_grid)
        for i in range(n_b):
            b_val = float(self.b_grid[i])
            for j in range(n_y):
                y_val = float(self.y_grid[j])
                records.append({
                    "debt_b": b_val,
                    "endowment_y": y_val,
                    "policy_b_next": float(self.policy_b[i, j]),
                    "bond_price_q": float(self.bond_price[i, j]),
                    "spread_annualized": float(self.spread[i, j]),
                    "spread_bps": float(self.spread[i, j] * 10000.0),
                    "default_prob": float(self.default_prob[i, j]),
                    "dq_db": float(self.dq_db[i, j]),
                    "v_repay": float(self.v_repay[i, j]),
                    "v_default": float(self.v_default[j]),
                    "v_expected": float(self.v_expected[i, j]),
                })
        return pd.DataFrame(records)

    def to_frame(self, digits: int = 6) -> pd.DataFrame:
        """Alias for summary DataFrame formatted to given digits."""
        df = self.summary()
        df["value"] = [_fmt_val(v, digits) for v in df["value"]]
        return df

    def to_markdown(self, digits: int = 6) -> str:
        """Render summary as GitHub-flavored Markdown."""
        return _render_df(self.to_frame(digits), "markdown")

    def to_latex(self, digits: int = 6) -> str:
        """Render summary as a LaTeX tabular."""
        return _render_df(self.to_frame(digits), "latex")

    def to_typst(self, digits: int = 6) -> str:
        """Render summary as a Typst table."""
        return _render_df(self.to_frame(digits), "typst")

    def simulate(
        self,
        T: int | None = None,
        periods: int | None = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Simulate Markov sample path of the sovereign economy.

        Parameters
        ----------
        T : int, optional
            Number of simulation quarters (synonym for ``periods``).
        periods : int, optional
            Number of simulation periods. Defaults to 1000 if both are None.
        seed : int, default 42
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            Simulated paths for endowment, debt, consumption, bond prices,
            spreads, and default indicators. The path starts at the zero-debt
            node (nearest node if the grid omits zero). Reentry returns to that
            node before the current period's default/repayment decision.
        """
        n_periods = 1000
        if periods is not None:
            n_periods = int(periods)
        elif T is not None:
            n_periods = int(T)

        rng = np.random.default_rng(seed)
        n_y = len(self.y_grid)
        delta = self.maturity_param_delta
        kappa = self.coupon_rate_kappa
        theta = float(self.metadata.get("theta", 0.282))
        y_def_vec = self.metadata.get("y_default_vec", 0.969 * self.y_grid)

        y_idx = n_y // 2
        zero_idx = int(np.argmin(np.abs(self.b_grid)))
        b_idx = zero_idx
        in_default = False

        records = []
        for t in range(n_periods):
            # Draw next endowment state
            y_idx = int(rng.choice(n_y, p=self.p_trans[y_idx]))
            y_val = float(self.y_grid[y_idx])

            if in_default:
                if rng.random() < theta:
                    in_default = False
                    b_idx = zero_idx
                else:
                    c_val = float(y_def_vec[y_idx]) if hasattr(y_def_vec, "__getitem__") else float(y_val * 0.969)
                    records.append({
                        "period": t,
                        "y": y_val,
                        "b": 0.0,
                        "debt": 0.0,
                        "b_next": 0.0,
                        "default": True,
                        "repaid": False,
                        "bond_price": 0.0,
                        "q": 0.0,
                        "spread": float(np.nan),
                        "spread_bps": float(np.nan),
                        "consumption": c_val,
                        "c": c_val,
                    })
                    continue

            # Sovereign chooses whether to default
            p_def = float(self.default_prob[b_idx, y_idx])
            if rng.random() < p_def:
                in_default = True
                c_val = float(y_def_vec[y_idx]) if hasattr(y_def_vec, "__getitem__") else float(y_val * 0.969)
                records.append({
                    "period": t,
                    "y": y_val,
                    "b": float(self.b_grid[b_idx]),
                    "debt": float(self.b_grid[b_idx]),
                    "b_next": 0.0,
                    "default": True,
                    "repaid": False,
                    "bond_price": 0.0,
                    "q": 0.0,
                    "spread": float(np.nan),
                    "spread_bps": float(np.nan),
                    "consumption": c_val,
                    "c": c_val,
                })
                continue

            # Sovereign repays and issues new debt
            b_next_val = float(self.policy_b[b_idx, y_idx])
            b_cur_val = float(self.b_grid[b_idx])

            # Find nearest grid index for next debt
            b_next_idx = int(np.argmin(np.abs(self.b_grid - b_next_val)))
            q_val = float(self.bond_price[b_next_idx, y_idx])
            spread_val = float(self.spread[b_next_idx, y_idx])

            c_val = y_val - (delta + (1.0 - delta) * kappa) * b_cur_val + q_val * (b_next_val - (1.0 - delta) * b_cur_val)
            c_val = max(1e-5, c_val)

            records.append({
                "period": t,
                "y": y_val,
                "b": b_cur_val,
                "debt": b_cur_val,
                "b_next": b_next_val,
                "default": False,
                "repaid": True,
                "bond_price": q_val,
                "q": q_val,
                "spread": spread_val,
                "spread_bps": spread_val * 10000.0,
                "consumption": c_val,
                "c": c_val,
            })
            b_idx = b_next_idx

        return pd.DataFrame(records)

    def plot_price_schedule(
        self,
        figsize: tuple[float, float] = (10.5, 4.5),
    ) -> matplotlib.figure.Figure:
        """Plot the equilibrium bond price schedule q(b', y) and spread curves."""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        n_y = len(self.y_grid)
        y_indices = [0, n_y // 2, n_y - 1]
        labels = ["Low Income", "Median Income", "High Income"]
        styles = ["--", "-", "-."]

        for idx, lbl, sty in zip(y_indices, labels, styles):
            ax1.plot(self.b_grid, self.bond_price[:, idx], label=f"{lbl} (y={self.y_grid[idx]:.2f})", linestyle=sty, lw=1.8)
            ax2.plot(self.b_grid, self.spread[:, idx] * 10000.0, label=f"{lbl} (y={self.y_grid[idx]:.2f})", linestyle=sty, lw=1.8)

        ax1.set_xlabel("Next-Period Debt $b'$")
        ax1.set_ylabel("Bond Price $q(b', y)$")
        ax1.set_title("Bond Price Schedule")
        ax1.grid(True, alpha=0.3)
        ax1.legend(frameon=False)

        ax2.set_xlabel("Next-Period Debt $b'$")
        ax2.set_ylabel("Sovereign Spread (bps)")
        ax2.set_title("Sovereign Spread Curve")
        ax2.grid(True, alpha=0.3)
        ax2.legend(frameon=False)

        fig.tight_layout()
        return fig

    def plot_policy(
        self,
        figsize: tuple[float, float] = (10.5, 4.5),
    ) -> matplotlib.figure.Figure:
        """Plot optimal borrowing policy b'(b, y) and default probability."""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        n_y = len(self.y_grid)
        y_indices = [0, n_y // 2, n_y - 1]
        labels = ["Low Income", "Median Income", "High Income"]
        styles = ["--", "-", "-."]

        ax1.plot(self.b_grid, self.b_grid, color="gray", linestyle=":", lw=1.2, label="45° Line")
        for idx, lbl, sty in zip(y_indices, labels, styles):
            ax1.plot(self.b_grid, self.policy_b[:, idx], label=lbl, linestyle=sty, lw=1.8)
            ax2.plot(self.b_grid, self.default_prob[:, idx], label=lbl, linestyle=sty, lw=1.8)

        ax1.set_xlabel("Current Debt $b$")
        ax1.set_ylabel("Chosen Debt $b'$")
        ax1.set_title("Debt Policy Function $b'(b, y)$")
        ax1.grid(True, alpha=0.3)
        ax1.legend(frameon=False)

        ax2.set_xlabel("Debt Choice $b'$")
        ax2.set_ylabel("Default Probability $p_{\\mathrm{def}}$")
        ax2.set_title("Default Hazard")
        ax2.grid(True, alpha=0.3)
        ax2.legend(frameon=False)

        fig.tight_layout()
        return fig

    def plot(
        self,
        figsize: tuple[float, float] = (10.5, 4.5),
    ) -> matplotlib.figure.Figure:
        """Default plot showing price schedule and spread curves."""
        return self.plot_price_schedule(figsize=figsize)


def _empty_index() -> np.ndarray:
    return np.zeros((0,), dtype=int)


@dataclass(frozen=True)
class FiscalSovereignResult(SovereignDefaultResult):
    """Sovereign default result with a government-spending block.

    Adds the fiscal policies implied by the debt choice: public spending
    ``g = tau*y - b + q(b', y) b'``, the primary balance ``b - q b'``
    (``= tau*y - g``), debt-to-output ``b'/y`` and private consumption
    ``(1 - tau) y``. Debt and spending are in units of mean quarterly output.
    """

    tau: float = 0.0
    psi: float = 0.0
    gamma_c: float = 2.0
    gamma_g: float = 2.0
    private_consumption: bool = True
    policy_index: np.ndarray = field(default_factory=_empty_index)
    g_policy: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    g_default: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    primary_balance: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    debt_to_gdp: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    private_consumption_repay: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    private_consumption_default: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    rule: FiscalRule | None = None

    def summary(self) -> pd.DataFrame:
        """Return the sovereign summary plus the fiscal parameters."""
        base = super().summary()
        extra = [("tau", self.tau), ("psi", self.psi), ("gamma_c", self.gamma_c),
                 ("gamma_g", self.gamma_g), ("private_consumption", self.private_consumption)]
        rows = pd.DataFrame({"metric": [r[0] for r in extra], "value": [_fmt_val(r[1]) for r in extra]})
        return pd.concat([base, rows], ignore_index=True)

    def simulate(
        self,
        T: int | None = None,
        periods: int | None = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Simulate one fiscal path from zero debt and the middle income node.

        Uses the vectorized fiscal path engine with one path drawn from
        ``np.random.default_rng(seed).random((3, periods))`` (income, default
        and reentry planes), so the random stream differs from
        :meth:`SovereignDefaultResult.simulate` for the same seed. The first
        decision happens at the starting state; reentry is drawn before that
        period's decision. ``gdp`` is ``y`` when repaying and ``y_def``
        otherwise, and ``g + private_consumption + primary_balance == gdp``.
        """
        from puremacro.sovereign._fiscal_paths import draw_uniforms, simulate_paths

        n_periods = 1000
        if periods is not None:
            n_periods = int(periods)
        elif T is not None:
            n_periods = int(T)
        paths = simulate_paths(self, int(self.metadata["zero_index"]), len(self.y_grid) // 2,
                               draw_uniforms([seed], n_periods))
        repaid = paths.repaid[0]
        y_idx = paths.y_index[0]
        private = np.where(repaid, self.private_consumption_repay[y_idx], self.private_consumption_default[y_idx])
        return pd.DataFrame({
            "period": np.arange(n_periods),
            "y": paths.y[0],
            "gdp": paths.gdp[0],
            "b": self.b_grid[paths.b_index[0]],
            "b_next": self.b_grid[paths.b_next_index[0]],
            "default": paths.default_entry[0],
            "repaid": repaid,
            "excluded": ~paths.access[0],
            "g": paths.g[0],
            "private_consumption": private,
            "primary_balance": paths.primary_balance[0],
            "debt_to_gdp": paths.debt_to_gdp[0],
            "spread": paths.spread[0],
            "bond_price": paths.bond_price[0],
        })


@dataclass(frozen=True)
class SuddenStopResult:
    """Result container for Mendoza (2010) sudden-stop collateral constraint model."""

    k_grid: np.ndarray
    b_grid: np.ndarray
    z_grid: np.ndarray
    policy_k_prime: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    policy_b_prime: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    policy_c: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    policy_q_asset: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    collateral_binding_fraction: float = 0.0
    multiplier_lambda: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    policy_b: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    policy_k: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    asset_price_q: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    kappa_collateral: float = 0.30
    crisis_probability: float = 0.0
    amplification_ratio: float = 1.0
    converged: bool = False
    iterations: int = 0
    p_trans: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.policy_b) == 0 and len(self.policy_b_prime) > 0:
            object.__setattr__(self, "policy_b", self.policy_b_prime)
        elif len(self.policy_b_prime) == 0 and len(self.policy_b) > 0:
            object.__setattr__(self, "policy_b_prime", self.policy_b)

        if len(self.policy_k) == 0 and len(self.policy_k_prime) > 0:
            object.__setattr__(self, "policy_k", self.policy_k_prime)
        elif len(self.policy_k_prime) == 0 and len(self.policy_k) > 0:
            object.__setattr__(self, "policy_k_prime", self.policy_k)

        if len(self.asset_price_q) == 0 and len(self.policy_q_asset) > 0:
            object.__setattr__(self, "asset_price_q", self.policy_q_asset)
        elif len(self.policy_q_asset) == 0 and len(self.asset_price_q) > 0:
            object.__setattr__(self, "policy_q_asset", self.asset_price_q)

        if self.collateral_binding_fraction == 0.0 and self.crisis_probability != 0.0:
            object.__setattr__(self, "collateral_binding_fraction", self.crisis_probability)
        elif self.crisis_probability == 0.0 and self.collateral_binding_fraction != 0.0:
            object.__setattr__(self, "crisis_probability", self.collateral_binding_fraction)

    def summary(self) -> pd.DataFrame:
        """Return a structured summary DataFrame of the sudden stop model."""
        binding_mask = self.multiplier_lambda > 1e-4
        mean_multiplier = float(np.mean(self.multiplier_lambda[binding_mask])) if np.any(binding_mask) else 0.0
        min_q = float(np.min(self.asset_price_q))
        max_q = float(np.max(self.asset_price_q))

        rows = [
            ("converged", self.converged),
            ("iterations", self.iterations),
            ("kappa_collateral", self.kappa_collateral),
            ("collateral_binding_fraction", self.collateral_binding_fraction),
            ("crisis_probability", self.crisis_probability),
            ("amplification_ratio", self.amplification_ratio),
            ("mean_binding_multiplier", mean_multiplier),
            ("min_tobin_q", min_q),
            ("max_tobin_q", max_q),
            ("n_debt_points", len(self.b_grid)),
            ("n_capital_points", len(self.k_grid)),
            ("n_productivity_points", len(self.z_grid)),
            ("precautionary_buffer_pct", float(self.metadata.get("precautionary_buffer_pct", 0.0))),
        ]
        return pd.DataFrame(
            {"metric": [r[0] for r in rows], "value": [_fmt_val(r[1]) for r in rows]}
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Tabulate policies across debt, capital, and shock states."""
        records = []
        n_b = len(self.b_grid)
        n_k = len(self.k_grid)
        n_z = len(self.z_grid)
        for ib in range(n_b):
            b_val = float(self.b_grid[ib])
            for ik in range(n_k):
                k_val = float(self.k_grid[ik])
                for iz in range(n_z):
                    z_val = float(self.z_grid[iz])
                    records.append({
                        "debt_b": b_val,
                        "capital_k": k_val,
                        "prod_z": z_val,
                        "policy_b_next": float(self.policy_b[ib, ik, iz]),
                        "policy_k_next": float(self.policy_k[ib, ik, iz]),
                        "consumption_c": float(self.policy_c[ib, ik, iz]),
                        "multiplier_lambda": float(self.multiplier_lambda[ib, ik, iz]),
                        "tobin_q": float(self.asset_price_q[ib, ik, iz]),
                    })
        return pd.DataFrame(records)

    def to_frame(self, digits: int = 6) -> pd.DataFrame:
        df = self.summary()
        df["value"] = [_fmt_val(v, digits) for v in df["value"]]
        return df

    def to_markdown(self, digits: int = 6) -> str:
        return _render_df(self.to_frame(digits), "markdown")

    def to_latex(self, digits: int = 6) -> str:
        return _render_df(self.to_frame(digits), "latex")

    def to_typst(self, digits: int = 6) -> str:
        return _render_df(self.to_frame(digits), "typst")

    def simulate(
        self,
        T: int | None = None,
        periods: int | None = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Simulate Markov sample path of small open economy with collateral constraint."""
        n_periods = 1000
        if periods is not None:
            n_periods = int(periods)
        elif T is not None:
            n_periods = int(T)

        rng = np.random.default_rng(seed)
        n_b = len(self.b_grid)
        n_k = len(self.k_grid)
        n_z = len(self.z_grid)

        ib = n_b // 4
        ik = n_k // 2
        iz = n_z // 2

        records = []
        for t in range(n_periods):
            iz = int(rng.choice(n_z, p=self.p_trans[iz]))
            z_val = float(self.z_grid[iz])
            b_val = float(self.b_grid[ib])
            k_val = float(self.k_grid[ik])

            b_next = float(self.policy_b[ib, ik, iz])
            k_next = float(self.policy_k[ib, ik, iz])
            c_val = float(self.policy_c[ib, ik, iz])
            lam_val = float(self.multiplier_lambda[ib, ik, iz])
            q_val = float(self.asset_price_q[ib, ik, iz])

            is_crisis = bool(lam_val > 1e-4)
            records.append({
                "period": t,
                "z": z_val,
                "b": b_val,
                "debt": b_val,
                "b_next": b_next,
                "k": k_val,
                "k_next": k_next,
                "consumption": c_val,
                "c": c_val,
                "tobin_q": q_val,
                "multiplier": lam_val,
                "multiplier_lambda": lam_val,
                "crisis": is_crisis,
            })

            ib = int(np.argmin(np.abs(self.b_grid - b_next)))
            ik = int(np.argmin(np.abs(self.k_grid - k_next)))

        return pd.DataFrame(records)

    def plot_crisis_dynamics(
        self,
        figsize: tuple[float, float] = (10.5, 4.5),
    ) -> matplotlib.figure.Figure:
        """Plot Tobin's q and multiplier vs debt showing constraint binding."""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        ik_mid = len(self.k_grid) // 2
        n_z = len(self.z_grid)
        z_indices = [0, n_z // 2, n_z - 1]
        labels = ["Low Prod ($z_{\\mathrm{low}}$)", "Normal Prod ($z_{\\mathrm{mid}}$)", "High Prod ($z_{\\mathrm{high}}$)"]
        styles = ["--", "-", "-."]

        for iz, lbl, sty in zip(z_indices, labels, styles):
            ax1.plot(self.b_grid, self.asset_price_q[:, ik_mid, iz], label=lbl, linestyle=sty, lw=1.8)
            ax2.plot(self.b_grid, self.multiplier_lambda[:, ik_mid, iz], label=lbl, linestyle=sty, lw=1.8)

        ax1.set_xlabel("Foreign Debt $b$")
        ax1.set_ylabel("Tobin's $q$ (Capital Price)")
        ax1.set_title("Fisherian Asset Price Deflation")
        ax1.grid(True, alpha=0.3)
        ax1.legend(frameon=False)

        ax2.set_xlabel("Foreign Debt $b$")
        ax2.set_ylabel("Collateral Multiplier $\\lambda$")
        ax2.set_title("Collateral Constraint Binding")
        ax2.grid(True, alpha=0.3)
        ax2.legend(frameon=False)

        fig.tight_layout()
        return fig

    def plot(
        self,
        figsize: tuple[float, float] = (10.5, 4.5),
    ) -> matplotlib.figure.Figure:
        """Default plot showing crisis dynamics."""
        return self.plot_crisis_dynamics(figsize=figsize)


@dataclass(frozen=True)
class RestructuringResult:
    """Result container for sovereign debt restructuring counterfactuals."""

    baseline_spread: float
    counterfactual_spread: float
    spread_reduction_bps: float
    haircut: float
    maturity_extension_years: float
    pre_default_prob: float
    post_default_prob: float
    debt_gdp_relief_pct: float
    transition_path: pd.DataFrame
    spread_compression: float = 0.0
    restructured_debt: float = 0.0
    initial_debt: float = 0.0
    default_prob_post: float = 0.0
    default_prob_pre: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.spread_compression == 0.0 and self.spread_reduction_bps != 0.0:
            object.__setattr__(self, "spread_compression", self.spread_reduction_bps)
        if self.default_prob_post == 0.0 and self.post_default_prob != 0.0:
            object.__setattr__(self, "default_prob_post", self.post_default_prob)
        if self.default_prob_pre == 0.0 and self.pre_default_prob != 0.0:
            object.__setattr__(self, "default_prob_pre", self.pre_default_prob)

    def summary(self) -> pd.DataFrame:
        """Return a structured summary DataFrame of the restructuring counterfactual."""
        rows = [
            ("haircut_pct", self.haircut * 100.0),
            ("maturity_extension_years", self.maturity_extension_years),
            ("baseline_spread_bps", self.baseline_spread * 10000.0),
            ("counterfactual_spread_bps", self.counterfactual_spread * 10000.0),
            ("spread_reduction_bps", self.spread_reduction_bps),
            ("spread_compression", self.spread_compression),
            ("pre_default_probability_pct", self.pre_default_prob * 100.0),
            ("post_default_probability_pct", self.post_default_prob * 100.0),
            ("default_hazard_reduction_pct", (self.pre_default_prob - self.post_default_prob) * 100.0),
            ("debt_to_gdp_relief_pct", self.debt_gdp_relief_pct),
            ("initial_debt", self.initial_debt),
            ("restructured_debt", self.restructured_debt),
        ]
        return pd.DataFrame(
            {"metric": [r[0] for r in rows], "value": [_fmt_val(r[1]) for r in rows]}
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return the transition path DataFrame."""
        return self.transition_path.copy()

    def to_frame(self, digits: int = 6) -> pd.DataFrame:
        df = self.summary()
        df["value"] = [_fmt_val(v, digits) for v in df["value"]]
        return df

    def to_markdown(self, digits: int = 6) -> str:
        return _render_df(self.to_frame(digits), "markdown")

    def to_latex(self, digits: int = 6) -> str:
        return _render_df(self.to_frame(digits), "latex")

    def to_typst(self, digits: int = 6) -> str:
        return _render_df(self.to_frame(digits), "typst")

    def plot(
        self,
        figsize: tuple[float, float] = (10.5, 4.5),
    ) -> matplotlib.figure.Figure:
        """Plot the recovery path of spreads and debt post-restructuring."""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        df = self.transition_path

        ax1.plot(df["period"], df["spread_bps"], color="black", lw=1.8, label="Restructured Spread")
        ax1.axhline(self.baseline_spread * 10000.0, color="red", linestyle="--", lw=1.2, label="Pre-Restructuring Baseline")
        ax1.set_xlabel("Quarters Post-Restructuring")
        ax1.set_ylabel("Sovereign Spread (bps)")
        ax1.set_title("Spread Compression Trajectory")
        ax1.grid(True, alpha=0.3)
        ax1.legend(frameon=False)

        ax2.plot(df["period"], df["debt"], color="black", lw=1.8, label="Debt Stock")
        ax2.set_xlabel("Quarters Post-Restructuring")
        ax2.set_ylabel("Sovereign Debt Level")
        ax2.set_title("Debt Dynamics Post-Relief")
        ax2.grid(True, alpha=0.3)
        ax2.legend(frameon=False)

        fig.tight_layout()
        return fig
