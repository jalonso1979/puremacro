"""Allen & Arkolakis (2014) Quantitative Spatial General Equilibrium Model.

Continuous geographic spatial general equilibrium with trade costs,
amenities, productivities, and labor mobility across space.

Reference:
    Allen, T. and Arkolakis, C. (2014). "Trade and the Topography of the
    Spatial Economy." The Quarterly Journal of Economics, 129(3), 1085–1140.
"""
from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from puremacro.reports import _df_to_latex, _df_to_markdown, _df_to_typst
from puremacro.spatial.weights import pairwise_distances


@dataclass(frozen=True)
class AllenArkolakisResult:
    """Equilibrium and counterfactual solution of the Allen-Arkolakis spatial model.

    Attributes
    ----------
    wages : np.ndarray
        Equilibrium nominal wages w_i, shape (N,).
    population : np.ndarray
        Equilibrium population distribution L_i, shape (N,).
    price_index : np.ndarray
        Equilibrium CES price index P_i, shape (N,).
    real_wages : np.ndarray
        Equilibrium real wages w_i / P_i, shape (N,).
    welfare : float
        Spatially equalized utility level \\bar{u}.
    trade_shares : np.ndarray
        Bilateral expenditure shares \\pi_{ij} (share of destination i on origin j), shape (N, N).
    consumer_market_access : np.ndarray
        Consumer market access CMA_i = P_i^{-\\theta}, shape (N,).
    firm_market_access : np.ndarray
        Firm market access FMA_i = \\sum_j \\tau_{ij}^{-\\theta} P_j^\\theta w_j L_j, shape (N,).
    converged : bool
        Whether the fixed-point iteration converged within tolerance.
    iterations : int
        Number of fixed-point iterations performed.
    max_residual : float
        Maximum residual across goods and labor market clearing conditions.
    labor_conservation_residual : float
        Absolute error in total population conservation (|\\sum_i L_i - \\bar{L}|).
    spatial_utility_variance : float
        Normalized spatial utility dispersion (std(u_i) / \\bar{u}).
    region_names : tuple[str, ...]
        Tuple of region identifier labels.
    coordinates : np.ndarray | None, optional
        Geographic coordinates (lat, lon or x, y) of regions, shape (N, 2).
    w_hat : np.ndarray | None, optional
        Relative wage changes (w' / w) if counterfactual.
    L_hat : np.ndarray | None, optional
        Relative population changes (L' / L) if counterfactual.
    welfare_hat : float | None, optional
        Relative welfare change (\\bar{u}' / \\bar{u}) if counterfactual.
    welfare_pct : float | None, optional
        Percentage welfare change ((\\bar{u}' / \\bar{u} - 1) * 100) if counterfactual.
    """

    wages: np.ndarray
    population: np.ndarray
    price_index: np.ndarray
    real_wages: np.ndarray
    welfare: float
    trade_shares: np.ndarray
    consumer_market_access: np.ndarray
    firm_market_access: np.ndarray
    converged: bool
    iterations: int
    max_residual: float
    labor_conservation_residual: float
    spatial_utility_variance: float
    region_names: tuple[str, ...]
    coordinates: np.ndarray | None = None
    w_hat: np.ndarray | None = None
    L_hat: np.ndarray | None = None
    welfare_hat: float | None = None
    welfare_pct: float | None = None

    def summary(self) -> pd.DataFrame:
        """Regional summary table of equilibrium variables."""
        data = {
            "wage": self.wages,
            "population": self.population,
            "price_index": self.price_index,
            "real_wage": self.real_wages,
            "cma": self.consumer_market_access,
            "fma": self.firm_market_access,
        }
        if self.w_hat is not None:
            data["wage_hat"] = self.w_hat
        if self.L_hat is not None:
            data["population_hat"] = self.L_hat

        df = pd.DataFrame(data, index=pd.Index(self.region_names, name="region"))
        return df

    def to_frame(self) -> pd.DataFrame:
        """Return the summary DataFrame."""
        return self.summary()

    def to_markdown(self, index: bool = True, digits: int = 4, **kwargs: Any) -> str:
        """Format regional summary as a Markdown table."""
        return _df_to_markdown(self.summary(), index=index, digits=digits, **kwargs)

    def to_latex(self, index: bool = True, digits: int = 4, **kwargs: Any) -> str:
        """Format regional summary as a LaTeX tabular."""
        return _df_to_latex(self.summary(), index=index, digits=digits, **kwargs)

    def to_typst(self, index: bool = True, digits: int = 4, **kwargs: Any) -> str:
        """Format regional summary as a Typst table."""
        return _df_to_typst(self.summary(), index=index, digits=digits, **kwargs)

    def plot(
        self,
        kind: str = "spatial",
        ax: Any | None = None,
        figsize: tuple[float, float] = (9.0, 5.0),
        **kwargs: Any,
    ) -> Any:
        """Plot equilibrium spatial distributions."""
        import matplotlib.pyplot as plt

        regions = list(self.region_names)
        x = np.arange(len(regions))

        if kind == "spatial" and self.coordinates is not None:
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.get_figure()

            coords = self.coordinates
            sizes = 200.0 * (self.population / np.mean(self.population))
            scatter = ax.scatter(
                coords[:, 1],
                coords[:, 0],
                s=sizes,
                c=self.real_wages,
                cmap="viridis",
                edgecolor="black",
                alpha=0.85,
                **kwargs,
            )
            for i, name in enumerate(regions):
                ax.annotate(
                    name,
                    (coords[i, 1], coords[i, 0]),
                    fontsize=8,
                    ha="center",
                    va="center",
                )
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label("Real Wage ($w_i / P_i$)")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title("Allen-Arkolakis Spatial Equilibrium (Size: Pop, Color: Real Wage)")
            ax.grid(True, linestyle=":", alpha=0.5)
            fig.tight_layout()
            return fig

        if kind in ("population", "spatial"):
            if ax is None:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            else:
                fig = ax.get_figure()
                ax1 = ax
                ax2 = None

            ax1.bar(x, self.population, color="#2b5c8f", edgecolor="black", linewidth=0.5)
            ax1.set_xticks(x)
            ax1.set_xticklabels(regions, rotation=45, ha="right")
            ax1.set_ylabel("Population ($L_i$)")
            ax1.set_title("Population Distribution")
            ax1.grid(True, linestyle=":", alpha=0.5, axis="y")

            if ax2 is not None:
                ax2.bar(x, self.real_wages, color="#27ae60", edgecolor="black", linewidth=0.5)
                ax2.set_xticks(x)
                ax2.set_xticklabels(regions, rotation=45, ha="right")
                ax2.set_ylabel("Real Wage ($w_i / P_i$)")
                ax2.set_title("Real Wage Distribution")
                ax2.grid(True, linestyle=":", alpha=0.5, axis="y")
            fig.tight_layout()
            return fig

        if kind == "counterfactual":
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.get_figure()

            if self.L_hat is not None and self.w_hat is not None:
                width = 0.35
                ax.bar(
                    x - width / 2,
                    (self.L_hat - 1.0) * 100.0,
                    width,
                    label="Population Change (%)",
                    color="#2b5c8f",
                    edgecolor="black",
                )
                ax.bar(
                    x + width / 2,
                    (self.w_hat - 1.0) * 100.0,
                    width,
                    label="Wage Change (%)",
                    color="#e67e22",
                    edgecolor="black",
                )
                ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
                ax.set_xticks(x)
                ax.set_xticklabels(regions, rotation=45, ha="right")
                ax.set_ylabel("Percentage Change (%)")
                ax.set_title("Allen-Arkolakis Counterfactual Impact")
                ax.legend()
                ax.grid(True, linestyle=":", alpha=0.5, axis="y")
            else:
                raise ValueError("No counterfactual changes recorded in this result.")
            fig.tight_layout()
            return fig

        raise ValueError(f"Unknown plot kind: {kind!r}. Choose from 'spatial', 'population', 'counterfactual'.")

    @property
    def trade_shares_dest_origin(self) -> np.ndarray:
        """Bilateral expenditure shares where axis 0 is destination i and axis 1 is origin j."""
        return self.trade_shares

    @property
    def trade_shares_origin_dest(self) -> np.ndarray:
        """Bilateral expenditure shares where axis 0 is origin j and axis 1 is destination i (transpose)."""
        return self.trade_shares.T


class AllenArkolakisModel:
    """Allen & Arkolakis (2014) Quantitative Spatial General Equilibrium Model.

    Solves for the spatial distributions of population, wages, prices, market access,
    and equalized aggregate utility under free labor mobility, bilateral trade frictions,
    productivity agglomeration, and amenity congestion.

    Parameters
    ----------
    trade_costs : np.ndarray
        Bilateral iceberg trade costs \\tau_{ij} >= 1.0 of shape (N, N), where index 0
        is origin i and index 1 is destination j. Must satisfy \\tau_{ii} = 1.0.
    fundamental_productivity : np.ndarray
        Exogenous fundamental productivity \\bar{A}_i > 0 of shape (N,).
    fundamental_amenity : np.ndarray
        Exogenous fundamental amenities \\bar{a}_i > 0 of shape (N,).
    theta : float, default 4.0
        Trade elasticity \\theta > 0 (where \\theta = \\sigma - 1).
    alpha : float, default 0.10
        Productivity agglomeration elasticity (A_i = \\bar{A}_i L_i^\\alpha).
    beta : float, default -0.30
        Amenity congestion elasticity (a_i = \\bar{a}_i L_i^\\beta). Typically \\beta < 0.
    total_population : float, default 1.0
        Total mobile labor endowment \\bar{L} = \\sum_i L_i.
    region_names : Sequence[str] | None, optional
        Region identifier names of length N. Defaults to ("R0", "R1", ...).
    coordinates : np.ndarray | None, optional
        Geographic coordinates (e.g. [lat, lon] or [x, y]) of shape (N, 2).
    """

    def __init__(
        self,
        trade_costs: np.ndarray,
        fundamental_productivity: np.ndarray,
        fundamental_amenity: np.ndarray,
        theta: float = 4.0,
        alpha: float = 0.10,
        beta: float = -0.30,
        total_population: float = 1.0,
        region_names: Sequence[str] | None = None,
        coordinates: np.ndarray | None = None,
    ) -> None:
        self.trade_costs = np.asarray(trade_costs, dtype=float)
        self.fundamental_productivity = np.asarray(fundamental_productivity, dtype=float)
        self.fundamental_amenity = np.asarray(fundamental_amenity, dtype=float)
        self.theta = float(theta)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.total_population = float(total_population)

        if self.trade_costs.ndim != 2 or self.trade_costs.shape[0] != self.trade_costs.shape[1]:
            raise ValueError(f"trade_costs must be square (N, N), got shape {self.trade_costs.shape}")
        self.N = self.trade_costs.shape[0]

        if self.fundamental_productivity.shape != (self.N,):
            raise ValueError(
                f"fundamental_productivity must have shape ({self.N},), got {self.fundamental_productivity.shape}"
            )
        if self.fundamental_amenity.shape != (self.N,):
            raise ValueError(
                f"fundamental_amenity must have shape ({self.N},), got {self.fundamental_amenity.shape}"
            )

        if np.any(self.trade_costs < 1.0):
            raise ValueError("Iceberg trade costs must be >= 1.0 everywhere.")
        if np.any(self.fundamental_productivity <= 0.0):
            raise ValueError("Fundamental productivities must be strictly positive.")
        if np.any(self.fundamental_amenity <= 0.0):
            raise ValueError("Fundamental amenities must be strictly positive.")
        if self.theta <= 0.0:
            raise ValueError(f"Trade elasticity theta must be positive, got {self.theta}")
        if self.beta == 0.0:
            raise ValueError(
                "Amenity congestion elasticity beta cannot be zero, as equilibrium inversion "
                "requires finite congestion elasticity (nu = -1/beta)."
            )
        if self.beta > 0.0:
            warnings.warn(
                f"Amenity congestion elasticity beta is non-negative ({self.beta}). "
                "The model typically requires beta < 0 to ensure congestion balances agglomeration.",
                UserWarning,
                stacklevel=2,
            )
        if self.total_population <= 0.0:
            raise ValueError(f"total_population must be positive, got {self.total_population}")

        if region_names is None:
            self.region_names = tuple(f"R{i}" for i in range(self.N))
        else:
            if len(region_names) != self.N:
                raise ValueError(
                    f"region_names length ({len(region_names)}) must equal N ({self.N})"
                )
            self.region_names = tuple(str(r) for r in region_names)

        if coordinates is not None:
            coords_arr = np.asarray(coordinates, dtype=float)
            if coords_arr.shape != (self.N, 2):
                raise ValueError(f"coordinates must have shape ({self.N}, 2), got {coords_arr.shape}")
            self.coordinates = coords_arr
        else:
            self.coordinates = None

        # Pre-flight uniqueness check
        if not self.is_unique:
            warnings.warn(
                f"Parameters alpha={self.alpha}, beta={self.beta}, theta={self.theta} violate "
                f"the uniqueness condition (alpha + beta <= theta / (1 + theta) and alpha <= 1 / theta). "
                "Multiple equilibria or agglomeration collapse may occur.",
                UserWarning,
                stacklevel=2,
            )

    @property
    def is_unique(self) -> bool:
        """Pre-flight check of the Allen & Arkolakis (2014, Theorem 2) uniqueness condition.

        Guaranteed uniqueness holds if:
            alpha + beta <= theta / (1 + theta)  and  alpha <= 1 / theta.
        """
        bound_sum = self.theta / (1.0 + self.theta)
        bound_alpha = 1.0 / self.theta
        return bool((self.alpha + self.beta <= bound_sum + 1e-12) and (self.alpha <= bound_alpha + 1e-12))

    @classmethod
    def from_coordinates(
        cls,
        coords: Any,
        fundamental_productivity: np.ndarray | None = None,
        fundamental_amenity: np.ndarray | None = None,
        distance_cost_param: float = 0.001,
        distance_exponent: float = 0.5,
        metric: str = "haversine",
        theta: float = 4.0,
        alpha: float = 0.10,
        beta: float = -0.30,
        total_population: float = 1.0,
        region_names: Sequence[str] | None = None,
    ) -> AllenArkolakisModel:
        """Construct an Allen-Arkolakis model from geographic coordinates.

        Iceberg trade costs are parametrized as:
            \\tau_{ij} = 1.0 + \\delta \\cdot d_{ij}^\\gamma  (for i != j)
            \\tau_{ii} = 1.0.

        Parameters
        ----------
        coords : array-like
            Coordinates of shape (N, 2). If metric is 'haversine', coordinates are [lat, lon] in degrees.
        fundamental_productivity : np.ndarray | None, optional
            Productivity fundamentals. Default is uniform 1.0.
        fundamental_amenity : np.ndarray | None, optional
            Amenity fundamentals. Default is uniform 1.0.
        distance_cost_param : float, default 0.001
            Friction scaling parameter \\delta.
        distance_exponent : float, default 0.5
            Distance elasticity exponent \\gamma.
        metric : str, default 'haversine'
            Distance metric ('haversine' or 'euclidean').
        theta : float, default 4.0
            Trade elasticity.
        alpha : float, default 0.10
            Agglomeration elasticity.
        beta : float, default -0.30
            Congestion elasticity.
        total_population : float, default 1.0
            Total labor endowment.
        region_names : Sequence[str] | None, optional
            Region names.

        Returns
        -------
        AllenArkolakisModel
            Configured spatial model instance.
        """
        coords_arr = np.asarray(coords, dtype=float)
        if coords_arr.ndim != 2 or coords_arr.shape[1] != 2:
            raise ValueError("coords must be (N, 2) array of coordinates")
        N = coords_arr.shape[0]

        # Compute pairwise distance matrix
        dist_matrix = pairwise_distances(coords_arr, metric=metric)

        # Iceberg trade costs: tau_{ij} = 1 + delta * dist^gamma, with tau_{ii} = 1
        trade_costs = 1.0 + distance_cost_param * (dist_matrix ** distance_exponent)
        np.fill_diagonal(trade_costs, 1.0)

        if fundamental_productivity is None:
            fund_A = np.ones(N, dtype=float)
        else:
            fund_A = np.asarray(fundamental_productivity, dtype=float)

        if fundamental_amenity is None:
            fund_a = np.ones(N, dtype=float)
        else:
            fund_a = np.asarray(fundamental_amenity, dtype=float)

        return cls(
            trade_costs=trade_costs,
            fundamental_productivity=fund_A,
            fundamental_amenity=fund_a,
            theta=theta,
            alpha=alpha,
            beta=beta,
            total_population=total_population,
            region_names=region_names,
            coordinates=coords_arr,
        )

    def solve_equilibrium(
        self,
        tol: float = 1e-8,
        max_iter: int = 2500,
        damping: float = 0.35,
    ) -> AllenArkolakisResult:
        """Solve for the spatial general equilibrium.

        Uses contraction mapping iteration on nominal wages and population distribution
        guaranteeing exact labor mass conservation |\\sum L_i - \\bar{L}| < 10^{-12}
        and spatial utility equalization std(u_i) / \\bar{u} < 10^{-7}.

        Parameters
        ----------
        tol : float, default 1e-8
            Convergence tolerance for wage changes and spatial utility dispersion.
        max_iter : int, default 2500
            Maximum fixed-point iterations.
        damping : float, default 0.35
            Damping relaxation parameter \\lambda \\in (0, 1].

        Returns
        -------
        AllenArkolakisResult
            Frozen dataclass containing the equilibrium allocations and welfare.
        """
        N = self.N
        theta = self.theta
        alpha = self.alpha
        beta = self.beta
        L_bar = self.total_population
        tau = self.trade_costs
        A_bar = self.fundamental_productivity
        a_bar = self.fundamental_amenity

        # Precompute kernel matrix: K_{ji} = tau_{ji}^{-theta}
        # In destination i, variety from origin j arrives with cost tau_{ji}
        # So tau_to_dest[j, i] = tau[j, i]
        kernel_P = tau ** (-theta)  # shape (N, N), element [j, i] is shipping from j to i

        # Initial conditions: uniform population and unit wages
        w = np.ones(N, dtype=float)
        L = np.full(N, L_bar / N, dtype=float)

        converged = False
        iteration = 0
        utility_dispersion = 1.0
        max_residual = 1.0
        nu = -1.0 / beta  # positive sensitivity of population to real wage / amenity

        for it in range(1, max_iter + 1):
            iteration = it

            # 1. Productivities with agglomeration
            A = A_bar * (L ** alpha)

            # 2. Price index P_i:
            # P_i^{-theta} = sum_j tau_{ji}^{-theta} * (w_j / A_j)^{-theta}
            # Term w_j / A_j: shape (N,)
            marginal_cost = w / A
            cost_factor = marginal_cost ** (-theta)
            # P_pow_neg_theta[i] = sum_j kernel_P[j, i] * cost_factor[j]
            P_pow_neg_theta = np.dot(cost_factor, kernel_P)
            P = P_pow_neg_theta ** (-1.0 / theta)

            # 3. Firm Market Access FMA_i:
            # FMA_i = sum_j tau_{ij}^{-theta} * P_j^theta * w_j * L_j
            # Note kernel_P[i, j] = tau_{ij}^{-theta}
            dest_exp = (P ** theta) * w * L
            FMA = np.dot(kernel_P, dest_exp)

            # 4. Wage update from goods market clearing:
            # w_i L_i = (w_i / A_i)^{-theta} * FMA_i = w_i^{-theta} * A_i^theta * FMA_i
            # w_i^{1 + theta} = A_i^theta * L_i^{-1} * FMA_i = A_bar_i^theta * L_i^{alpha * theta - 1} * FMA_i
            w_target = ((A ** theta) * FMA / L) ** (1.0 / (1.0 + theta))

            # Normalize wages: mean wage = 1.0
            w_target = w_target / np.mean(w_target)

            # Damped wage update
            w_next = (1.0 - damping) * w + damping * w_target
            w_next = w_next / np.mean(w_next)

            # 5. Labor reallocation from utility equalization:
            # u_i = a_bar_i * L_i^beta * (w_i / P_i) = u_bar
            # L_i = L_bar * (w_i * a_bar_i / P_i)^nu / sum_k (w_k * a_bar_k / P_k)^nu
            real_comp = (w_next * a_bar) / P
            # Guard against numerical overflow in powers
            log_real_comp = np.log(real_comp)
            z = nu * log_real_comp
            z_max = np.max(z)
            exp_z = np.exp(z - z_max)
            L_target = L_bar * (exp_z / np.sum(exp_z))

            # Damped labor update preserving total population
            L_next = (1.0 - damping) * L + damping * L_target
            # Exact mass conservation normalization:
            L_next = L_bar * (L_next / np.sum(L_next))

            # 6. Convergence diagnostics
            # Compute actual utilities
            u_i = a_bar * (L_next ** beta) * (w_next / P)
            u_bar = float(np.mean(u_i))
            utility_dispersion = float(np.std(u_i) / max(u_bar, 1e-12))

            wage_diff = float(np.max(np.abs(w_next - w)))
            pop_diff = float(np.max(np.abs(L_next - L)) / L_bar)
            max_residual = max(wage_diff, pop_diff, utility_dispersion)

            w = w_next
            L = L_next

            if max_residual < tol and utility_dispersion < 1e-7:
                converged = True
                break

        # Final pass allocations
        A = A_bar * (L ** alpha)
        P_pow_neg_theta = np.dot((w / A) ** (-theta), kernel_P)
        P = P_pow_neg_theta ** (-1.0 / theta)
        CMA = P_pow_neg_theta
        FMA = np.dot(kernel_P, (P ** theta) * w * L)
        real_wages = w / P

        # Utility and spatial price equalization
        u_i = a_bar * (L ** beta) * (w / P)
        welfare = float(np.mean(u_i))
        spatial_utility_variance = float(np.std(u_i) / max(welfare, 1e-12))
        labor_conservation_res = float(np.abs(np.sum(L) - L_bar))

        # Bilateral trade shares: pi[i, j] = share of destination i on origin j
        # pi[i, j] = tau_{ji}^{-theta} * (w_j / A_j)^{-theta} / P_i^{-theta}
        trade_shares = np.zeros((N, N), dtype=float)
        for i in range(N):
            trade_shares[i, :] = kernel_P[:, i] * ((w / A) ** (-theta)) / CMA[i]
            trade_shares[i, :] = trade_shares[i, :] / np.sum(trade_shares[i, :])

        return AllenArkolakisResult(
            wages=w,
            population=L,
            price_index=P,
            real_wages=real_wages,
            welfare=welfare,
            trade_shares=trade_shares,
            consumer_market_access=CMA,
            firm_market_access=FMA,
            converged=converged,
            iterations=iteration,
            max_residual=max_residual,
            labor_conservation_residual=labor_conservation_res,
            spatial_utility_variance=spatial_utility_variance,
            region_names=self.region_names,
            coordinates=self.coordinates,
        )

    def solve_counterfactual(
        self,
        trade_costs_new: np.ndarray | None = None,
        productivity_new: np.ndarray | None = None,
        amenity_new: np.ndarray | None = None,
        tol: float = 1e-8,
        max_iter: int = 2500,
        damping: float = 0.35,
    ) -> AllenArkolakisResult:
        """Solve a counterfactual equilibrium and compute comparative statics against baseline.

        Parameters
        ----------
        trade_costs_new : np.ndarray | None, optional
            Counterfactual trade costs \\tau'_{ij}.
        productivity_new : np.ndarray | None, optional
            Counterfactual fundamental productivity \\bar{A}'_i.
        amenity_new : np.ndarray | None, optional
            Counterfactual fundamental amenities \\bar{a}'_i.
        tol : float, default 1e-8
            Convergence tolerance.
        max_iter : int, default 2500
            Maximum iterations.
        damping : float, default 0.35
            Damping parameter.

        Returns
        -------
        AllenArkolakisResult
            Counterfactual equilibrium result with w_hat, L_hat, welfare_hat, and welfare_pct.
        """
        # Solve baseline equilibrium
        base_res = self.solve_equilibrium(tol=tol, max_iter=max_iter, damping=damping)

        # Create counterfactual model
        cf_tau = self.trade_costs if trade_costs_new is None else np.asarray(trade_costs_new, dtype=float)
        cf_A = self.fundamental_productivity if productivity_new is None else np.asarray(productivity_new, dtype=float)
        cf_a = self.fundamental_amenity if amenity_new is None else np.asarray(amenity_new, dtype=float)

        cf_model = AllenArkolakisModel(
            trade_costs=cf_tau,
            fundamental_productivity=cf_A,
            fundamental_amenity=cf_a,
            theta=self.theta,
            alpha=self.alpha,
            beta=self.beta,
            total_population=self.total_population,
            region_names=self.region_names,
            coordinates=self.coordinates,
        )

        cf_res = cf_model.solve_equilibrium(tol=tol, max_iter=max_iter, damping=damping)

        # Compute comparative statics
        w_hat = cf_res.wages / base_res.wages
        L_hat = cf_res.population / base_res.population
        welfare_hat = cf_res.welfare / base_res.welfare
        welfare_pct = (welfare_hat - 1.0) * 100.0

        return AllenArkolakisResult(
            wages=cf_res.wages,
            population=cf_res.population,
            price_index=cf_res.price_index,
            real_wages=cf_res.real_wages,
            welfare=cf_res.welfare,
            trade_shares=cf_res.trade_shares,
            consumer_market_access=cf_res.consumer_market_access,
            firm_market_access=cf_res.firm_market_access,
            converged=cf_res.converged,
            iterations=cf_res.iterations,
            max_residual=cf_res.max_residual,
            labor_conservation_residual=cf_res.labor_conservation_residual,
            spatial_utility_variance=cf_res.spatial_utility_variance,
            region_names=cf_res.region_names,
            coordinates=cf_res.coordinates,
            w_hat=w_hat,
            L_hat=L_hat,
            welfare_hat=welfare_hat,
            welfare_pct=welfare_pct,
        )

    def simulate_infrastructure_shock(
        self,
        origin: str | int,
        destination: str | int,
        cost_reduction: float = 0.20,
        **kwargs: Any,
    ) -> AllenArkolakisResult:
        """Simulate an infrastructure investment reducing bilateral transport costs.

        Parameters
        ----------
        origin : str | int
            Origin region name or index.
        destination : str | int
            Destination region name or index.
        cost_reduction : float, default 0.20
            Proportional reduction in excess iceberg trade costs (\\tau - 1).
            For instance, a 20% reduction transforms \\tau = 1.5 into 1 + 0.8 * 0.5 = 1.4.

        Returns
        -------
        AllenArkolakisResult
            Counterfactual spatial equilibrium result.
        """
        orig_idx = origin if isinstance(origin, int) else self.region_names.index(origin)
        dest_idx = destination if isinstance(destination, int) else self.region_names.index(destination)

        if orig_idx == dest_idx:
            raise ValueError("Infrastructure shock requires distinct origin and destination regions.")

        trade_costs_new = self.trade_costs.copy()
        # Reduce excess cost bilaterally
        excess_od = trade_costs_new[orig_idx, dest_idx] - 1.0
        excess_do = trade_costs_new[dest_idx, orig_idx] - 1.0
        trade_costs_new[orig_idx, dest_idx] = 1.0 + (1.0 - cost_reduction) * excess_od
        trade_costs_new[dest_idx, orig_idx] = 1.0 + (1.0 - cost_reduction) * excess_do

        return self.solve_counterfactual(trade_costs_new=trade_costs_new, **kwargs)

    def simulate_climate_shock(
        self,
        productivity_shocks: Mapping[str | int, float] | np.ndarray | None = None,
        amenity_shocks: Mapping[str | int, float] | np.ndarray | None = None,
        is_percentage_change: bool = False,
        **kwargs: Any,
    ) -> AllenArkolakisResult:
        """Simulate localized climate or environmental shocks altering productivities or amenities.

        Parameters
        ----------
        productivity_shocks : Mapping[str | int, float] | np.ndarray | None, optional
            Productivity shock. If ``is_percentage_change=True``, values represent relative
            fractional changes (e.g. +0.10 for +10%, -0.10 for -10%). If ``False``, values represent
            gross multipliers (e.g. 1.10 for +10%, 0.85 for -15%) with convenience support for
            negative fractional changes in (-1.0, 0.0).
            Can be given as a mapping {region: factor} or full array of length N.
        amenity_shocks : Mapping[str | int, float] | np.ndarray | None, optional
            Amenity shock. Follows the same conventions as ``productivity_shocks``.
            Can be given as a mapping {region: factor} or full array of length N.
        is_percentage_change : bool, default False
            Whether shock values are interpreted strictly as percentage/fractional changes
            relative to baseline (e.g., +0.10 means +10%, -0.10 means -10%, so factor = 1.0 + shock).
        **kwargs : Any
            Additional solver options passed to :meth:`solve_counterfactual`.

        Returns
        -------
        AllenArkolakisResult
            Counterfactual spatial equilibrium result.
        """
        prod_new = self.fundamental_productivity.copy()
        amen_new = self.fundamental_amenity.copy()

        def _clean_factor(v: float, name: str) -> float:
            val = float(v)
            if is_percentage_change:
                if val <= -1.0:
                    raise ValueError(f"Percentage shock {val} for {name} cannot be <= -1.0 (wipes out or inverts baseline).")
                return 1.0 + val
            if -1.0 < val < 0.0:
                return 1.0 + val
            if val <= -1.0:
                raise ValueError(f"Relative shock {val} for {name} cannot be <= -1.0 (wipes out or inverts baseline).")
            if val <= 0.0:
                raise ValueError(f"Gross shock multiplier for {name} must be strictly positive, got {val}")
            if 0.0 < val < 0.5:
                warnings.warn(
                    f"Shock value {val} for {name} is between 0 and 0.5 and interpreted as a gross multiplier "
                    f"({val * 100:.1f}% of baseline, i.e. a {(1.0 - val) * 100:.1f}% decline). "
                    f"If you intended a +{val * 100:.1f}% increase, pass 1.0 + {val} or set is_percentage_change=True.",
                    UserWarning,
                    stacklevel=3,
                )
            return val

        if productivity_shocks is not None:
            if isinstance(productivity_shocks, Mapping):
                for k, v in productivity_shocks.items():
                    idx = k if isinstance(k, int) else self.region_names.index(k)
                    prod_new[idx] *= _clean_factor(v, f"region {k}")
            else:
                arr = np.asarray(productivity_shocks, dtype=float)
                if arr.shape != (self.N,):
                    raise ValueError(f"productivity_shocks array must have shape ({self.N},)")
                if is_percentage_change:
                    if np.any(arr <= -1.0):
                        raise ValueError("Percentage shock cannot be <= -1.0 (wipes out or inverts baseline).")
                    factors = 1.0 + arr
                else:
                    if np.any(arr <= -1.0):
                        raise ValueError("Relative shock cannot be <= -1.0 (wipes out or inverts baseline).")
                    factors = np.where((-1.0 < arr) & (arr < 0.0), 1.0 + arr, arr)
                    if np.any(factors <= 0.0):
                        raise ValueError("Gross shock multiplier must be strictly positive.")
                prod_new *= factors

        if amenity_shocks is not None:
            if isinstance(amenity_shocks, Mapping):
                for k, v in amenity_shocks.items():
                    idx = k if isinstance(k, int) else self.region_names.index(k)
                    amen_new[idx] *= _clean_factor(v, f"region {k}")
            else:
                arr = np.asarray(amenity_shocks, dtype=float)
                if arr.shape != (self.N,):
                    raise ValueError(f"amenity_shocks array must have shape ({self.N},)")
                if is_percentage_change:
                    if np.any(arr <= -1.0):
                        raise ValueError("Percentage shock cannot be <= -1.0 (wipes out or inverts baseline).")
                    factors = 1.0 + arr
                else:
                    if np.any(arr <= -1.0):
                        raise ValueError("Relative shock cannot be <= -1.0 (wipes out or inverts baseline).")
                    factors = np.where((-1.0 < arr) & (arr < 0.0), 1.0 + arr, arr)
                    if np.any(factors <= 0.0):
                        raise ValueError("Gross shock multiplier must be strictly positive.")
                amen_new *= factors

        return self.solve_counterfactual(
            productivity_new=prod_new, amenity_new=amen_new, **kwargs
        )
