"""Caliendo & Parro (2015) Quantitative Trade General Equilibrium Model.

Multi-country, multi-sector Ricardian trade general equilibrium with
input-output linkages, bilateral trade costs, tariffs, and trade imbalances,
solved via Exact Hat Algebra.

Reference:
    Caliendo, L. and Parro, F. (2015). "Estimates of the Trade and Welfare Effects
    of NAFTA." The Review of Economic Studies, 82(1), 1–44.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import linalg

from puremacro.reports import _df_to_latex, _df_to_markdown, _df_to_typst


@dataclass(frozen=True)
class CaliendoParroResult:
    """Counterfactual general equilibrium result from Caliendo & Parro (2015).

    Attributes
    ----------
    w_hat : np.ndarray
        Gross change in nominal wages (\\hat{w}_n = w_n' / w_n), shape (N,).
    P_hat : np.ndarray
        Gross change in sectoral price indices (\\hat{P}_n^j), shape (N, J).
    P_index_hat : np.ndarray
        Gross change in consumer price indices (\\hat{P}_n), shape (N,).
    real_wage_hat : np.ndarray
        Gross change in real wages (\\hat{w}_n / \\hat{P}_n), shape (N,).
    income_hat : np.ndarray
        Gross change in total national income (I_n' / I_n), shape (N,).
    welfare_hat : np.ndarray
        Gross change in real income / welfare (\\hat{I}_n / \\hat{P}_n), shape (N,).
    welfare_pct : np.ndarray
        Percentage change in welfare ((welfare_hat - 1) * 100), shape (N,).
    pi_prime : np.ndarray
        Counterfactual bilateral trade shares (\\pi_{ni}'^j), shape (J, N, N).
    X_prime : np.ndarray
        Counterfactual sectoral expenditures (X_n'^j), shape (N, J).
    Y_prime : np.ndarray
        Counterfactual sectoral gross outputs (Y_n'^j), shape (N, J).
    tariff_revenue_prime : np.ndarray
        Counterfactual tariff revenue by country (R_n'), shape (N,).
    converged : bool
        Whether the equilibrium wage iteration converged within tolerance.
    iterations : int
        Number of wage tatonnement iterations performed.
    max_residual : float
        Maximum absolute excess demand residual across countries.
    market_clearing_residual : float
        Maximum goods / labor market clearing residual across countries.
    trade_balance_residual : float
        Maximum trade balance residual across countries.
    country_codes : tuple[str, ...]
        Tuple of country identifier codes.
    sector_codes : tuple[str, ...]
        Tuple of sector identifier codes.
    """

    w_hat: np.ndarray
    P_hat: np.ndarray
    P_index_hat: np.ndarray
    real_wage_hat: np.ndarray
    income_hat: np.ndarray
    welfare_hat: np.ndarray
    welfare_pct: np.ndarray
    pi_prime: np.ndarray
    X_prime: np.ndarray
    Y_prime: np.ndarray
    tariff_revenue_prime: np.ndarray
    converged: bool
    iterations: int
    max_residual: float
    market_clearing_residual: float
    trade_balance_residual: float
    country_codes: tuple[str, ...]
    sector_codes: tuple[str, ...]

    def summary(self) -> pd.DataFrame:
        """Country-level summary of equilibrium changes."""
        df = pd.DataFrame(
            {
                "wage_hat": self.w_hat,
                "cpi_hat": self.P_index_hat,
                "real_wage_hat": self.real_wage_hat,
                "income_hat": self.income_hat,
                "welfare_hat": self.welfare_hat,
                "welfare_pct": self.welfare_pct,
                "tariff_revenue_prime": self.tariff_revenue_prime,
            },
            index=pd.Index(self.country_codes, name="country"),
        )
        return df

    def sector_summary(self) -> pd.DataFrame:
        """Detailed country-sector breakdown of prices, expenditures, and outputs."""
        rows = []
        for n, c in enumerate(self.country_codes):
            for j, s in enumerate(self.sector_codes):
                rows.append(
                    {
                        "country": c,
                        "sector": s,
                        "price_hat": float(self.P_hat[n, j]),
                        "expenditure_prime": float(self.X_prime[n, j]),
                        "output_prime": float(self.Y_prime[n, j]),
                    }
                )
        return pd.DataFrame(rows).set_index(["country", "sector"])

    def to_frame(self) -> pd.DataFrame:
        """Return the country summary DataFrame."""
        return self.summary()

    def to_markdown(self, index: bool = True, digits: int = 4, **kwargs: Any) -> str:
        """Format country summary as a GitHub-flavored Markdown table."""
        return _df_to_markdown(self.summary(), index=index, digits=digits, **kwargs)

    def to_latex(self, index: bool = True, digits: int = 4, **kwargs: Any) -> str:
        """Format country summary as a LaTeX tabular environment."""
        return _df_to_latex(self.summary(), index=index, digits=digits, **kwargs)

    def to_typst(self, index: bool = True, digits: int = 4, **kwargs: Any) -> str:
        """Format country summary as a Typst table."""
        return _df_to_typst(self.summary(), index=index, digits=digits, **kwargs)

    def plot(
        self,
        kind: str = "welfare",
        ax: Any | None = None,
        figsize: tuple[float, float] = (8.0, 4.5),
        **kwargs: Any,
    ) -> Any:
        """Plot counterfactual outcome by country."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        countries = list(self.country_codes)
        x = np.arange(len(countries))

        if kind == "welfare":
            vals = self.welfare_pct
            ylabel = "Welfare Change (%)"
            title = "Caliendo-Parro (2015): Counterfactual Welfare Effects"
        elif kind == "real_wage":
            vals = (self.real_wage_hat - 1.0) * 100.0
            ylabel = "Real Wage Change (%)"
            title = "Caliendo-Parro (2015): Real Wage Effects"
        elif kind == "wage":
            vals = (self.w_hat - 1.0) * 100.0
            ylabel = "Nominal Wage Change (%)"
            title = "Caliendo-Parro (2015): Nominal Wage Effects"
        elif kind == "cpi":
            vals = (self.P_index_hat - 1.0) * 100.0
            ylabel = "Consumer Price Index Change (%)"
            title = "Caliendo-Parro (2015): Price Index Effects"
        else:
            raise ValueError(
                f"Unknown plot kind: {kind!r}. Choose from 'welfare', 'real_wage', 'wage', 'cpi'."
            )

        colors = ["#2b5c8f" if v >= 0 else "#c0392b" for v in vals]
        ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5, **kwargs)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(countries)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.5, axis="y")
        fig.tight_layout()
        return fig


class CaliendoParroModel:
    """Caliendo & Parro (2015) Multi-Country Multi-Sector Trade GE Model.

    Solves for counterfactual general equilibrium changes using Exact Hat Algebra
    with intermediate input-output linkages, sector-specific trade elasticities,
    tariffs, and trade imbalances.

    Parameters
    ----------
    trade_shares : np.ndarray
        Bilateral trade shares \\pi_{ni}^j of shape (J, N, N), where index 0 is sector j,
        index 1 is destination country n (importer), and index 2 is origin country i (exporter).
        Satisfies \\sum_{i=1}^N \\pi_{ni}^j = 1 for all n, j.
    gamma_va : np.ndarray
        Value-added (labor) cost share \\gamma_n^j of shape (N, J). Must be strictly positive.
    gamma_io : np.ndarray
        Intermediate input cost share \\gamma_n^{j, k} of shape (N, J, J), where index 0 is country n,
        index 1 is producing sector j, and index 2 is input sector k.
        Satisfies \\gamma_n^j + \\sum_{k=1}^J \\gamma_n^{j, k} = 1 for all n, j.
    alpha : np.ndarray
        Final consumption expenditure share \\alpha_n^j of shape (N, J).
        Satisfies \\sum_{j=1}^J \\alpha_n^j = 1 for all n.
    theta : np.ndarray
        Sectoral trade elasticity \\theta_j of shape (J,). Must be strictly positive.
    labor_income : np.ndarray
        Baseline labor income (value-added) w_n L_n of shape (N,). Must be strictly positive.
    deficits : np.ndarray | None, optional
        Baseline trade deficits D_n of shape (N,). Satisfies \\sum_{n=1}^N D_n = 0.
        If None, assumed to be zero (balanced trade).
    tariffs : np.ndarray | None, optional
        Baseline gross tariffs \\tau_{ni}^j = 1 + t_{ni}^j of shape (J, N, N).
        Must be >= 1.0 with domestic tariffs \\tau_{nn}^j = 1.0. If None, default is all 1.0.
    country_codes : Sequence[str] | None, optional
        Country identifier codes of length N. Defaults to ("C0", "C1", ...).
    sector_codes : Sequence[str] | None, optional
        Sector identifier codes of length J. Defaults to ("S0", "S1", ...).
    nontradables : Sequence[int | str] | None, optional
        Indices or names of non-tradable sectors.
    """

    def __init__(
        self,
        trade_shares: np.ndarray,
        gamma_va: np.ndarray,
        gamma_io: np.ndarray,
        alpha: np.ndarray,
        theta: np.ndarray,
        labor_income: np.ndarray,
        deficits: np.ndarray | None = None,
        tariffs: np.ndarray | None = None,
        country_codes: Sequence[str] | None = None,
        sector_codes: Sequence[str] | None = None,
        nontradables: Sequence[int | str] | None = None,
    ) -> None:
        self.trade_shares = np.asarray(trade_shares, dtype=float)
        self.gamma_va = np.asarray(gamma_va, dtype=float)
        self.gamma_io = np.asarray(gamma_io, dtype=float)
        self.alpha = np.asarray(alpha, dtype=float)
        self.theta = np.asarray(theta, dtype=float)
        self.labor_income = np.asarray(labor_income, dtype=float)

        # Infer dimensions
        if self.trade_shares.ndim != 3:
            raise ValueError(
                f"trade_shares must be 3D (J, N, N), got shape {self.trade_shares.shape}"
            )
        self.J, self.N, N_check = self.trade_shares.shape
        if self.N != N_check:
            raise ValueError(
                f"trade_shares destination and origin dimensions must match: {self.N} != {N_check}"
            )

        # Validate shapes
        if self.gamma_va.shape != (self.N, self.J):
            raise ValueError(
                f"gamma_va must have shape ({self.N}, {self.J}), got {self.gamma_va.shape}"
            )
        if self.gamma_io.shape != (self.N, self.J, self.J):
            raise ValueError(
                f"gamma_io must have shape ({self.N}, {self.J}, {self.J}), got {self.gamma_io.shape}"
            )
        if self.alpha.shape != (self.N, self.J):
            raise ValueError(
                f"alpha must have shape ({self.N}, {self.J}), got {self.alpha.shape}"
            )
        if self.theta.shape != (self.J,):
            raise ValueError(
                f"theta must have shape ({self.J},), got {self.theta.shape}"
            )
        if self.labor_income.shape != (self.N,):
            raise ValueError(
                f"labor_income must have shape ({self.N},), got {self.labor_income.shape}"
            )

        # Check positivity
        if np.any(self.gamma_va <= 0):
            raise ValueError("gamma_va (value-added share) must be strictly positive everywhere.")
        if np.any(self.theta <= 0):
            raise ValueError("theta (trade elasticities) must be strictly positive.")
        if np.any(self.labor_income <= 0):
            raise ValueError("labor_income must be strictly positive.")

        # Normalize trade shares along origin axis (axis 2) to ensure exact summing to 1
        share_sums = np.sum(self.trade_shares, axis=2, keepdims=True)
        if np.any(share_sums <= 0):
            raise ValueError("trade_shares contain non-positive row sums.")
        self.trade_shares = self.trade_shares / share_sums

        # Normalize consumption shares to sum to 1
        alpha_sums = np.sum(self.alpha, axis=1, keepdims=True)
        if np.any(alpha_sums <= 0):
            raise ValueError("alpha contains non-positive country sums.")
        self.alpha = self.alpha / alpha_sums

        # Handle deficits
        if deficits is None:
            self.deficits = np.zeros(self.N, dtype=float)
        else:
            self.deficits = np.asarray(deficits, dtype=float)
            if self.deficits.shape != (self.N,):
                raise ValueError(
                    f"deficits must have shape ({self.N},), got {self.deficits.shape}"
                )
            if abs(np.sum(self.deficits)) > 1e-4 * max(1.0, np.sum(self.labor_income)):
                raise ValueError(
                    f"World trade deficits must sum to zero, got sum = {np.sum(self.deficits)}"
                )

        # Handle tariffs
        if tariffs is None:
            self.tariffs = np.ones((self.J, self.N, self.N), dtype=float)
        else:
            self.tariffs = np.asarray(tariffs, dtype=float)
            if self.tariffs.shape != (self.J, self.N, self.N):
                raise ValueError(
                    f"tariffs must have shape ({self.J}, {self.N}, {self.N}), got {self.tariffs.shape}"
                )
            if np.any(self.tariffs < 1.0):
                raise ValueError("Gross tariffs must be >= 1.0.")

        # Country and sector labels
        if country_codes is None:
            self.country_codes = tuple(f"C{n}" for n in range(self.N))
        else:
            if len(country_codes) != self.N:
                raise ValueError(
                    f"country_codes length ({len(country_codes)}) must equal N ({self.N})"
                )
            self.country_codes = tuple(str(c) for c in country_codes)

        if sector_codes is None:
            self.sector_codes = tuple(f"S{j}" for j in range(self.J))
        else:
            if len(sector_codes) != self.J:
                raise ValueError(
                    f"sector_codes length ({len(sector_codes)}) must equal J ({self.J})"
                )
            self.sector_codes = tuple(str(s) for s in sector_codes)

        # Non-tradables mask
        self.is_nontradable = np.zeros(self.J, dtype=bool)
        if nontradables is not None:
            for item in nontradables:
                if isinstance(item, int):
                    if 0 <= item < self.J:
                        self.is_nontradable[item] = True
                    else:
                        raise ValueError(f"nontradable index {item} out of range [0, {self.J-1}]")
                elif isinstance(item, str):
                    if item in self.sector_codes:
                        self.is_nontradable[self.sector_codes.index(item)] = True
                    else:
                        raise ValueError(f"nontradable sector name {item!r} not in sector_codes")
                else:
                    raise TypeError(f"nontradable entry must be int or str, got {type(item)}")

        # Ensure non-tradables have pi_{nn} = 1 and pi_{ni} = 0 for i != n
        for j in range(self.J):
            if self.is_nontradable[j]:
                self.trade_shares[j] = np.eye(self.N)

        # Baseline baseline tariff revenue:
        # R_n = sum_{j} sum_{i} (tau_{ni}^j - 1)/tau_{ni}^j * pi_{ni}^j * X_n^j
        # Compute baseline expenditures X_n^j consistently from baseline labor income:
        self.baseline_X, self.baseline_Y, self.baseline_R = self._solve_baseline_expenditures()
        self.baseline_income = self.labor_income + self.baseline_R + self.deficits

    def _solve_baseline_expenditures(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve for baseline expenditures X_n^j, gross outputs Y_n^j, and tariff revenue R_n."""
        # Solves X = A X + b where b_nj = alpha_n^j * (w_n L_n + D_n)
        pi = self.trade_shares
        tau = self.tariffs
        N, J = self.N, self.J

        # mu[n, j] = sum_i (tau_{ni}^j - 1) / tau_{ni}^j * pi_{ni}^j
        mu = np.sum((tau - 1.0) / tau * pi, axis=2).T  # shape (N, J)

        # Linear system for X of shape (N * J,)
        dim = N * J
        M = np.eye(dim, dtype=float)
        b = np.zeros(dim, dtype=float)

        for n in range(N):
            wL_plus_D = self.labor_income[n] + self.deficits[n]
            for j in range(J):
                row_idx = n * J + j
                b[row_idx] = self.alpha[n, j] * wL_plus_D

                # Tariff revenue feedback: alpha_n^j * mu_n^k * X_n^k
                for k in range(J):
                    col_idx_same = n * J + k
                    M[row_idx, col_idx_same] -= self.alpha[n, j] * mu[n, k]

                # Intermediate input demand: sum_k gamma_n^{k, j} * sum_m (pi_{mn}^k / tau_{mn}^k) * X_m^k
                for k in range(J):
                    gamma_k_to_j = self.gamma_io[n, k, j]
                    if gamma_k_to_j == 0.0:
                        continue
                    for m in range(N):
                        col_idx_m = m * J + k
                        purch_share = pi[k, m, n] / tau[k, m, n]
                        M[row_idx, col_idx_m] -= gamma_k_to_j * purch_share

        X_vec = linalg.solve(M, b)
        X = X_vec.reshape((N, J))

        # Gross output Y_n^j = sum_m (pi_{mn}^j / tau_{mn}^j) * X_m^j
        Y = np.zeros((N, J), dtype=float)
        for n in range(N):
            for j in range(J):
                Y[n, j] = np.sum((pi[j, :, n] / tau[j, :, n]) * X[:, j])

        R = np.sum(mu * X, axis=1)
        return X, Y, R

    def _solve_inner_price_indices(
        self,
        w_hat: np.ndarray,
        kappa_hat: np.ndarray,
        tol: float = 1e-12,
        max_iter: int = 500,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve inner-tier contraction mapping for sectoral price index changes.

        ln c_i^j = gamma_i^j ln w_hat_i + sum_k gamma_i^{j, k} ln P_i^k
        P_hat_n^j = (sum_i pi_{ni}^j (c_hat_i^j * kappa_hat_{ni}^j)^{-theta_j})^{-1/theta_j}
        """
        N, J = self.N, self.J
        ln_w = np.log(w_hat)
        ln_kappa = np.log(kappa_hat)  # shape (J, N, N)

        # Initial guess: zero log price changes
        ln_P = np.zeros((N, J), dtype=float)

        for _ in range(max_iter):
            # Compute log input bundle cost ln_c of shape (N, J)
            # ln_c[n, j] = gamma_va[n, j] * ln_w[n] + sum_k gamma_io[n, j, k] * ln_P[n, k]
            ln_c = self.gamma_va * ln_w[:, None] + np.einsum("njk,nk->nj", self.gamma_io, ln_P)

            ln_P_next = np.zeros_like(ln_P)
            for j in range(J):
                if self.is_nontradable[j]:
                    # Non-tradable price index equals domestic unit cost
                    ln_P_next[:, j] = ln_c[:, j]
                else:
                    theta_j = self.theta[j]
                    # Exponent term: -theta_j * (ln_c_i^j + ln_kappa_{ni}^j)
                    # Shape: (N, N) where row is destination n, col is origin i
                    ln_cost_term = ln_c[:, j][None, :] + ln_kappa[j, :, :]
                    z = -theta_j * ln_cost_term

                    # Log-sum-exp with weights pi_{ni}^j:
                    # Filter for positive baseline shares
                    pi_j = self.trade_shares[j]
                    z_max = np.max(z, axis=1, keepdims=True)
                    # Stabilized sum: sum_i pi_{ni}^j * exp(z_{ni} - z_max_n)
                    exp_sum = np.sum(pi_j * np.exp(z - z_max), axis=1)
                    ln_P_next[:, j] = -(1.0 / theta_j) * (z_max.ravel() + np.log(exp_sum))

            diff = np.max(np.abs(ln_P_next - ln_P))
            ln_P = ln_P_next
            if diff < tol:
                break

        P_hat = np.exp(ln_P)
        c_hat = np.exp(self.gamma_va * ln_w[:, None] + np.einsum("njk,nk->nj", self.gamma_io, ln_P))
        return P_hat, c_hat

    def _solve_expenditures(
        self,
        w_hat: np.ndarray,
        pi_prime: np.ndarray,
        tau_prime: np.ndarray,
        deficits_prime: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Intermediate tier: solve linear Leontief system for counterfactual expenditures."""
        N, J = self.N, self.J
        mu_prime = np.sum((tau_prime - 1.0) / tau_prime * pi_prime, axis=2).T  # (N, J)

        dim = N * J
        M = np.eye(dim, dtype=float)
        b = np.zeros(dim, dtype=float)

        for n in range(N):
            disp_inc = w_hat[n] * self.labor_income[n] + deficits_prime[n]
            for j in range(J):
                row_idx = n * J + j
                b[row_idx] = self.alpha[n, j] * disp_inc

                # Tariff revenue term: alpha_n^j * mu_n^k * X_n'^k
                for k in range(J):
                    M[row_idx, n * J + k] -= self.alpha[n, j] * mu_prime[n, k]

                # Intermediate demand: sum_k gamma_n^{k, j} * sum_m (pi'_{mn}^k / tau'_{mn}^k) * X_m'^k
                for k in range(J):
                    gamma_k_to_j = self.gamma_io[n, k, j]
                    if gamma_k_to_j == 0.0:
                        continue
                    for m in range(N):
                        col_idx_m = m * J + k
                        purch_share = pi_prime[k, m, n] / tau_prime[k, m, n]
                        M[row_idx, col_idx_m] -= gamma_k_to_j * purch_share

        X_vec = linalg.solve(M, b)
        X_prime = X_vec.reshape((N, J))

        # Counterfactual gross output Y_n'^j = sum_m (pi'_{mn}^j / tau'_{mn}^j) * X_m'^j
        Y_prime = np.zeros((N, J), dtype=float)
        for n in range(N):
            for j in range(J):
                Y_prime[n, j] = np.sum((pi_prime[j, :, n] / tau_prime[j, :, n]) * X_prime[:, j])

        tariff_revenue_prime = np.sum(mu_prime * X_prime, axis=1)
        return X_prime, Y_prime, tariff_revenue_prime

    def _resolve_country_index(self, c: str | int) -> int:
        """Resolve country identifier (name, alias, or index) to integer index."""
        if isinstance(c, (int, np.integer)):
            idx = int(c)
            if 0 <= idx < self.N:
                return idx
            raise ValueError(f"Country index {idx} out of range [0, {self.N - 1}]")
        s = str(c)
        if s in self.country_codes:
            return self.country_codes.index(s)
        s_upper = s.upper()
        for idx, code in enumerate(self.country_codes):
            if code.upper() == s_upper:
                return idx
        _ALIAS_MAP = {
            "USA": 0, "US": 0, "UNITED STATES": 0,
            "CHN": 1, "CN": 1, "CHINA": 1,
            "ROW": 2, "DEU": 2, "DE": 2, "GERMANY": 2,
        }
        if s_upper in _ALIAS_MAP and _ALIAS_MAP[s_upper] < self.N:
            return _ALIAS_MAP[s_upper]
        raise ValueError(f"Country identifier {c!r} not found in country_codes {self.country_codes}")

    def _resolve_sector_index(self, s: str | int) -> int:
        """Resolve sector identifier (name, alias, or index) to integer index."""
        if isinstance(s, (int, np.integer)):
            idx = int(s)
            if 0 <= idx < self.J:
                return idx
            raise ValueError(f"Sector index {idx} out of range [0, {self.J - 1}]")
        st = str(s)
        if st in self.sector_codes:
            return self.sector_codes.index(st)
        s_lower = st.lower()
        for idx, code in enumerate(self.sector_codes):
            if code.lower() == s_lower:
                return idx
        _SEC_MAP = {
            "manuf": 0, "manufacturing": 0, "goods": 0, "mfg": 0,
            "serv": 1, "services": 1, "service": 1, "non-tradable": 1,
        }
        if s_lower in _SEC_MAP and _SEC_MAP[s_lower] < self.J:
            return _SEC_MAP[s_lower]
        raise ValueError(f"Sector identifier {s!r} not found in sector_codes {self.sector_codes}")

    def solve_counterfactual(
        self,
        tariffs_new: np.ndarray | None = None,
        tau_hat: np.ndarray | None = None,
        d_hat: np.ndarray | None = None,
        deficits_new: np.ndarray | None = None,
        deficit_rule: str = "fixed",
        tol: float = 1e-8,
        max_iter: int = 1500,
        damping: float = 0.35,
    ) -> CaliendoParroResult:
        """Solve for counterfactual trade general equilibrium via Exact Hat Algebra.

        Parameters
        ----------
        tariffs_new : np.ndarray | None, optional
            Counterfactual gross tariffs \\tau_{ni}'^j of shape (J, N, N).
        tau_hat : np.ndarray | None, optional
            Relative gross tariff changes \\hat{\\tau}_{ni}^j = \\tau_{ni}'^j / \\tau_{ni}^j.
        d_hat : np.ndarray | None, optional
            Relative iceberg trade cost changes \\hat{d}_{ni}^j of shape (J, N, N).
        deficits_new : np.ndarray | None, optional
            Counterfactual trade deficits D_n'. If provided, overrides deficit_rule.
        deficit_rule : str, default "fixed"
            Rule for trade deficits:
            - "fixed": D_n' = D_n
            - "scaled": D_n' = (\\hat{Y}_W) * D_n
            - "zero": D_n' = 0
        tol : float, default 1e-8
            Convergence tolerance on maximum excess labor demand residual.
        max_iter : int, default 1500
            Maximum outer wage tatonnement iterations.
        damping : float, default 0.35
            Wage adjustment damping parameter \\nu \\in (0, 1].

        Returns
        -------
        CaliendoParroResult
            Frozen dataclass containing counterfactual equilibrium outcomes.
        """
        N, J = self.N, self.J

        # Determine counterfactual tariffs tau_prime and relative changes tau_hat
        if tariffs_new is not None:
            tau_prime = np.asarray(tariffs_new, dtype=float)
            if tau_prime.shape != (J, N, N):
                raise ValueError(f"tariffs_new shape must be (J, N, N), got {tau_prime.shape}")
            tau_hat_arr = tau_prime / self.tariffs
        elif tau_hat is not None:
            tau_hat_arr = np.asarray(tau_hat, dtype=float)
            if tau_hat_arr.shape != (J, N, N):
                raise ValueError(f"tau_hat shape must be (J, N, N), got {tau_hat_arr.shape}")
            tau_prime = self.tariffs * tau_hat_arr
        else:
            tau_prime = self.tariffs.copy()
            tau_hat_arr = np.ones((J, N, N), dtype=float)

        # Iceberg trade cost changes
        if d_hat is not None:
            d_hat_arr = np.asarray(d_hat, dtype=float)
            if d_hat_arr.shape != (J, N, N):
                raise ValueError(f"d_hat shape must be (J, N, N), got {d_hat_arr.shape}")
        else:
            d_hat_arr = np.ones((J, N, N), dtype=float)

        # Total trade cost change kappa_hat
        kappa_hat = tau_hat_arr * d_hat_arr

        # Identity shock short-circuit
        is_zero_shock = (
            np.allclose(kappa_hat, 1.0)
            and (deficits_new is None or np.allclose(deficits_new, self.deficits))
            and (deficit_rule == "fixed" or np.allclose(self.deficits, 0.0))
        )
        if is_zero_shock:
            return CaliendoParroResult(
                w_hat=np.ones(N, dtype=float),
                P_hat=np.ones((N, J), dtype=float),
                P_index_hat=np.ones(N, dtype=float),
                real_wage_hat=np.ones(N, dtype=float),
                income_hat=np.ones(N, dtype=float),
                welfare_hat=np.ones(N, dtype=float),
                welfare_pct=np.zeros(N, dtype=float),
                pi_prime=self.trade_shares.copy(),
                X_prime=self.baseline_X.copy(),
                Y_prime=self.baseline_Y.copy(),
                tariff_revenue_prime=self.baseline_R.copy(),
                converged=True,
                iterations=0,
                max_residual=0.0,
                market_clearing_residual=0.0,
                trade_balance_residual=0.0,
                country_codes=self.country_codes,
                sector_codes=self.sector_codes,
            )

        # Initial wage guess: w_hat = 1.0 for all countries
        w_hat = np.ones(N, dtype=float)
        converged = False
        iteration = 0
        max_residual = 1.0

        # Precompute baseline total world labor income
        world_base_inc = np.sum(self.labor_income)

        # Scale damping dynamically with sectoral trade elasticities
        max_theta = float(np.max(self.theta))
        nu = min(damping, 1.5 / max(max_theta, 4.0))
        prev_residual = float("inf")

        for it in range(1, max_iter + 1):
            iteration = it

            # Deficit rule update
            if deficits_new is not None:
                deficits_prime = np.asarray(deficits_new, dtype=float)
            elif deficit_rule == "fixed":
                deficits_prime = self.deficits.copy()
            elif deficit_rule == "scaled":
                world_inc_scale = np.sum(w_hat * self.labor_income) / world_base_inc
                deficits_prime = self.deficits * world_inc_scale
            elif deficit_rule == "zero":
                deficits_prime = np.zeros(N, dtype=float)
            else:
                raise ValueError(f"Unknown deficit_rule: {deficit_rule!r}")

            # 1. Inner tier: solve price index changes P_hat and input costs c_hat
            P_hat, c_hat = self._solve_inner_price_indices(w_hat, kappa_hat)

            # 2. Counterfactual trade shares pi_prime
            pi_prime = np.zeros((J, N, N), dtype=float)
            for j in range(J):
                if self.is_nontradable[j]:
                    pi_prime[j] = np.eye(N)
                else:
                    theta_j = self.theta[j]
                    # (c_hat_i^j * kappa_hat_{ni}^j / P_hat_n^j)^{-theta_j}
                    cost_ratio = (
                        c_hat[:, j][None, :] * kappa_hat[j, :, :] / P_hat[:, j][:, None]
                    )
                    pi_prime[j] = self.trade_shares[j] * (cost_ratio ** (-theta_j))
                    # Exact normalization
                    row_sums = np.sum(pi_prime[j], axis=1, keepdims=True)
                    pi_prime[j] = pi_prime[j] / row_sums

            # 3. Intermediate tier: linear Leontief solve for expenditures
            X_prime, Y_prime, tariff_revenue_prime = self._solve_expenditures(
                w_hat, pi_prime, tau_prime, deficits_prime
            )

            # 4. Outer tier: check excess labor demand residual
            # Labor demand bill V_n' = sum_j gamma_va[n, j] * Y_prime[n, j]
            labor_demand = np.sum(self.gamma_va * Y_prime, axis=1)
            labor_supply = w_hat * self.labor_income
            excess_demand = labor_demand - labor_supply
            abs_excess = np.abs(excess_demand)
            rel_residual = abs_excess / self.labor_income
            max_residual = float(np.max(rel_residual))
            max_abs_residual = float(np.max(abs_excess))

            if max_residual < tol and max_abs_residual < 1e-6:
                converged = True
                break

            # Adaptive damping adjustment
            if max_residual > prev_residual:
                nu = max(0.01, nu * 0.7)
            else:
                nu = min(damping, nu * 1.05)
            prev_residual = max_residual

            # Tatonnement update: w_hat_new = w_hat * (labor_demand / labor_supply) ** nu
            ratio = np.clip(labor_demand / labor_supply, 0.05, 20.0)
            w_hat_new = w_hat * (ratio ** nu)

            # Numeraire normalization: keep world labor income constant
            norm_factor = world_base_inc / np.sum(w_hat_new * self.labor_income)
            w_hat = w_hat_new * norm_factor

        # Final calculations of aggregate outcomes
        P_index_hat = np.prod(P_hat ** self.alpha, axis=1)
        real_wage_hat = w_hat / P_index_hat

        # National income change: I_n' / I_n
        income_prime = w_hat * self.labor_income + tariff_revenue_prime + deficits_prime
        income_hat = income_prime / self.baseline_income

        # Welfare change: \hat{W}_n = \hat{I}_n / \hat{P}_n
        welfare_hat = income_hat / P_index_hat
        welfare_pct = (welfare_hat - 1.0) * 100.0

        # Market clearing and trade balance residuals
        # Net exports: E_n' - M_n' + D_n' = 0
        exports_prime = np.zeros(N, dtype=float)
        imports_prime = np.zeros(N, dtype=float)
        for n in range(N):
            for m in range(N):
                if m != n:
                    # Exports of n to m: sum_j (pi_{mn}'^j / tau_{mn}'^j) * X_m'^j
                    exports_prime[n] += np.sum(
                        (pi_prime[:, m, n] / tau_prime[:, m, n]) * X_prime[m, :]
                    )
                    # Imports of n from m: sum_j (pi_{nm}'^j / tau_{nm}'^j) * X_n'^j
                    imports_prime[n] += np.sum(
                        (pi_prime[:, n, m] / tau_prime[:, n, m]) * X_prime[n, :]
                    )

        trade_balance_res = float(np.max(np.abs(imports_prime - exports_prime - deficits_prime)))
        market_clearing_res = max_residual

        return CaliendoParroResult(
            w_hat=w_hat,
            P_hat=P_hat,
            P_index_hat=P_index_hat,
            real_wage_hat=real_wage_hat,
            income_hat=income_hat,
            welfare_hat=welfare_hat,
            welfare_pct=welfare_pct,
            pi_prime=pi_prime,
            X_prime=X_prime,
            Y_prime=Y_prime,
            tariff_revenue_prime=tariff_revenue_prime,
            converged=converged,
            iterations=iteration,
            max_residual=max_residual,
            market_clearing_residual=market_clearing_res,
            trade_balance_residual=trade_balance_res,
            country_codes=self.country_codes,
            sector_codes=self.sector_codes,
        )

    def simulate_tariff_shock(
        self,
        importer: str | int,
        exporter: str | int,
        sector: str | int | None = None,
        tariff_rate: float = 0.10,
        **kwargs: Any,
    ) -> CaliendoParroResult:
        """Simulate a bilateral tariff change.

        Parameters
        ----------
        importer : str | int
            Importer country code or index.
        exporter : str | int
            Exporter country code or index.
        sector : str | int | None, optional
            Sector code or index to apply tariff to. If None, applies across all sectors.
        tariff_rate : float, default 0.10
            New ad-valorem tariff rate (e.g. 0.10 for a 10% tariff).
            The gross tariff will be (1 + tariff_rate).

        Returns
        -------
        CaliendoParroResult
            Counterfactual equilibrium result.
        """
        imp_idx = self._resolve_country_index(importer)
        exp_idx = self._resolve_country_index(exporter)

        if imp_idx == exp_idx:
            raise ValueError("Importer and exporter cannot be the same country for tariff shock.")

        tariffs_new = self.tariffs.copy()
        if sector is None:
            # All tradable sectors
            for j in range(self.J):
                if not self.is_nontradable[j]:
                    tariffs_new[j, imp_idx, exp_idx] = 1.0 + tariff_rate
        else:
            sec_idx = self._resolve_sector_index(sector)
            if self.is_nontradable[sec_idx]:
                raise ValueError(f"Cannot apply tariff to non-tradable sector {self.sector_codes[sec_idx]}")
            tariffs_new[sec_idx, imp_idx, exp_idx] = 1.0 + tariff_rate

        return self.solve_counterfactual(tariffs_new=tariffs_new, **kwargs)

    def simulate_trade_war(
        self,
        coalition_a: Sequence[str | int],
        coalition_b: Sequence[str | int],
        tariff_rate: float = 0.25,
        **kwargs: Any,
    ) -> CaliendoParroResult:
        """Simulate a reciprocal trade war between two coalitions of countries.

        Parameters
        ----------
        coalition_a : Sequence[str | int]
            Country codes or indices in Coalition A.
        coalition_b : Sequence[str | int]
            Country codes or indices in Coalition B.
        tariff_rate : float, default 0.25
            Reciprocal ad-valorem tariff rate imposed by both coalitions on each other.

        Returns
        -------
        CaliendoParroResult
            Counterfactual equilibrium result under the trade war.
        """
        idx_a = [self._resolve_country_index(c) for c in coalition_a]
        idx_b = [self._resolve_country_index(c) for c in coalition_b]

        tariffs_new = self.tariffs.copy()
        for j in range(self.J):
            if not self.is_nontradable[j]:
                # A tariffs on B
                for ia in idx_a:
                    for ib in idx_b:
                        tariffs_new[j, ia, ib] = 1.0 + tariff_rate
                # B tariffs on A
                for ib in idx_b:
                    for ia in idx_a:
                        tariffs_new[j, ib, ia] = 1.0 + tariff_rate

        return self.solve_counterfactual(tariffs_new=tariffs_new, **kwargs)
